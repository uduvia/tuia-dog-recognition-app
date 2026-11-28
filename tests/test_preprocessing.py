import numpy as np
import pytest

from lib.preprocessing import l2_normalize, preprocess_bgr


def test_shape_dtype():
    img = np.random.randint(0, 255, (300, 120, 3), dtype=np.uint8)
    out = preprocess_bgr(img, 224)
    assert out.shape == (3, 224, 224) and out.dtype == np.float32


def test_bgr_to_rgb_order():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0] = 255  # canal B en BGR
    out = preprocess_bgr(img, 8)
    # tras BGR->RGB el azul queda en el canal 2 (alto), el rojo (canal 0) en el minimo
    assert out[2].mean() > out[0].mean()


def test_gray_and_bgra_supported():
    assert preprocess_bgr(np.zeros((20, 20), np.uint8), 16).shape == (3, 16, 16)
    assert preprocess_bgr(np.zeros((20, 20, 4), np.uint8), 16).shape == (3, 16, 16)


def test_invalid_image():
    with pytest.raises(ValueError):
        preprocess_bgr(None)


def test_l2_normalize():
    assert np.isclose(np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))), 1.0)
