"""Preprocesamiento compartido entre las notebooks y los servicios de la app.

Contrato unico (el mismo en entrenamiento e inferencia):
  - Entrada: imagen BGR uint8, HxWx3 (convencion OpenCV / app).
  - Resize directo a (size, size) con INTER_AREA al reducir e INTER_LINEAR al ampliar.
  - BGR -> RGB, escala [0, 1], normalizacion con estadisticas de ImageNet.
  - Salida: float32 CHW.

Se importa desde src/lib para que notebooks y backend usen exactamente las mismas funciones.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Valida la imagen y la lleva a HxWx3 uint8 (acepta gris y BGRA)."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Imagen vacia o invalida")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Forma de imagen no soportada: {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def preprocess_bgr(
    image: np.ndarray,
    size: int = 224,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 CHW normalizado, listo para el modelo (sin batch)."""
    image = ensure_bgr_uint8(image)
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)


def preprocess_batch(images: list[np.ndarray], size: int = 224) -> np.ndarray:
    """Lista de imagenes BGR -> array float32 NCHW."""
    return np.stack([preprocess_bgr(img, size) for img in images], axis=0)


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza L2 un vector 1D."""
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, eps)
