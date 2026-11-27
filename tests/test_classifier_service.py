from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from lib.bootstrap import build_classifier, build_detection
from lib.config import Settings
from lib.models import resnet_classifier


@pytest.mark.parametrize("name", ["cnn_custom", "resnet18_finetuned"])
def test_training_checkpoint_and_inference(tmp_path, monkeypatch, name):
    # Las pruebas comprueban la interfaz sin descargar pesos de ImageNet.
    make_resnet = resnet_classifier.resnet18
    monkeypatch.setattr(resnet_classifier, "resnet18", lambda weights: make_resnet(weights=None))
    rows = []
    for split in ("train", "valid", "test", "independent"):
        for value, breed in enumerate(("A", "B"), 1):
            path = f"{split}-{breed}.png"
            image = np.full((32, 32, 3), 70 * value, np.uint8)
            assert cv2.imwrite(str(tmp_path / path), image)
            rows.append(dict(path=path, split=split, breed=breed))
    pd.DataFrame(rows).to_csv(tmp_path / "splits_finales.csv", index=False)
    settings = Settings(dataset_path=tmp_path, model_path=tmp_path, output_path=tmp_path,
                        image_size=32, train_epochs=1, train_batch_size=2, classifier_model=name)
    service = build_classifier(settings)
    service.train_classifier()
    service = build_classifier(settings)
    metrics = service.evaluate_classifier("independent")
    assert set(metrics) == {"accuracy", "precision", "recall", "specificity", "f1"}
    assert list(service.last_eval["predictions"].path) == ["independent-A.png", "independent-B.png"]
    assert service.load_model().class_names == ["A", "B"]
    embedding = service.extract_custom_embedding(image)
    assert len(embedding) == 512
    assert np.isclose(np.linalg.norm(embedding), 1)
    detection = build_detection(settings, service)
    path = detection.classify_image(str(tmp_path / "test-B.png"), tmp_path, name)
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    assert result["breed"] in service.load_model().class_names
    assert 0 <= result["score"] <= 1
