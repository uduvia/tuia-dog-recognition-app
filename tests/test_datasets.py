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
