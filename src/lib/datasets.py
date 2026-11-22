"""Lectura del dataset de razas (carpeta por raza) con el preprocesamiento compartido."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from lib.preprocessing import preprocess_bgr

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Sample = tuple[Path, int]


def list_class_names(split_dir: Path) -> list[str]:
    """Nombres de raza ordenados alfabeticamente (define el mapeo indice -> raza)."""
    return sorted(d.name for d in Path(split_dir).iterdir() if d.is_dir())


def list_samples(
    split_dir: Path,
    class_names: list[str],
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> list[Sample]:
    """Lista (ruta, indice_de_clase) de un split. `max_per_class` sirve para pruebas rapidas."""
    rng = random.Random(seed)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples: list[Sample] = []
    for name in class_names:
        folder = Path(split_dir) / name
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if max_per_class is not None and len(files) > max_per_class:
            files = rng.sample(files, max_per_class)
        samples.extend((p, class_to_idx[name]) for p in files)
    return samples


def basic_augment(image: np.ndarray) -> np.ndarray:
    """Augmentation minima sobre BGR (provisoria; la definitiva se justifica en la notebook 1)."""
    if random.random() < 0.5:
        image = cv2.flip(image, 1)  # flip horizontal
    alpha = random.uniform(0.8, 1.2)  # contraste
    beta = random.uniform(-20, 20)  # brillo
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


class BreedDataset:
    """Dataset map-style (compatible con DataLoader): devuelve (CHW float32, label)."""

    def __init__(
        self,
        samples: list[Sample],
        image_size: int = 224,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        if self.transform is not None:
            image = self.transform(image)
        return preprocess_bgr(image, self.image_size), label
