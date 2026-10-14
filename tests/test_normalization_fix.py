import numpy as np

from sign_language.landmarks.normalization import normalize_landmarks


def test_1d_and_2d_shape_preservation():
    # 1D array of 42
    p1 = np.ones(42, dtype=np.float32) * 2.0
    out1 = normalize_landmarks(p1)
    assert out1.shape == (42,)
    assert np.isfinite(out1).all()

    # 1D array of 1659
    p2 = np.ones(1659, dtype=np.float32)
    m2 = np.ones(1659, dtype=np.float32)
    out2 = normalize_landmarks(p2, mask=m2)
    assert out2.shape == (1659,)

    # 2D array of (10, 1659)
    p3 = np.random.randn(10, 1659).astype(np.float32)
    out3 = normalize_landmarks(p3)
    assert out3.shape == (10, 1659)

    # 2D array of (10, 42)
    p4 = np.random.randn(10, 42).astype(np.float32)
    out4 = normalize_landmarks(p4)
    assert out4.shape == (10, 42)


def test_3d_landmarks_with_2d_and_3d_masks():
    # 3D array (10, 553, 3) with 2D mask (10, 1659)
    p = np.random.randn(10, 553, 3).astype(np.float32)
    m_2d = np.ones((10, 1659), dtype=np.float32)
    out_2d_mask = normalize_landmarks(p, mask=m_2d)
    assert out_2d_mask.shape == (10, 553, 3)

    # 3D array (10, 553, 3) with 3D mask (10, 553, 3)
    m_3d = np.ones((10, 553, 3), dtype=np.float32)
    out_3d_mask = normalize_landmarks(p, mask=m_3d)
    assert out_3d_mask.shape == (10, 553, 3)


def test_normalization_modes():
    p = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    out_std = normalize_landmarks(p, mode="standardize")
    assert out_std.shape == (3, 2)
    assert np.allclose(out_std.mean(axis=0), 0.0, atol=1e-5)

    out_minmax = normalize_landmarks(p, mode="minmax")
    assert out_minmax.shape == (3, 2)
    assert np.isclose(out_minmax.min(), 0.0)
    assert np.isclose(out_minmax.max(), 1.0)

    out_none = normalize_landmarks(p, mode="none")
    assert np.array_equal(out_none, p)
