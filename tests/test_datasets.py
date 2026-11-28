import cv2
import numpy as np

from lib.datasets import BreedDataset, list_class_names, list_samples


def _make(tmp_path):
    for breed in ("b", "a"):
        d = tmp_path / "train" / breed
        d.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(d / f"{i}.jpg"), np.random.randint(0, 255, (40, 60, 3), np.uint8))
    return tmp_path / "train"


def test_class_names_sorted(tmp_path):
    assert list_class_names(_make(tmp_path)) == ["a", "b"]


def test_limit_and_dataset_item(tmp_path):
    split = _make(tmp_path)
    samples = list_samples(split, ["a", "b"], max_per_class=2)
    assert len(samples) == 4
    x, y = BreedDataset(samples, 32)[0]
    assert x.shape == (3, 32, 32) and y in (0, 1)


def test_gallery_uses_only_manifest_train_rows(tmp_path):
    import pandas as pd
    import pytest
    from lib.tp2_data import cargar_splits, rutas_galeria

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    rows = []
    for split in ("train", "valid", "test"):
        folder = tmp_path / split / "a"
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "ok.jpg"), image)
        rows.append(dict(path=f"{split}/a/ok.jpg", breed="a", split=split))
    cv2.imwrite(str(tmp_path / "train/a/excluida.jpg"), image)
    manifest = tmp_path / "splits_finales.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    assert rutas_galeria(tmp_path) == [(tmp_path / "train/a/ok.jpg", "a")]
    rows[-1]["path"] = rows[0]["path"]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="duplicadas"):
        cargar_splits(tmp_path)
