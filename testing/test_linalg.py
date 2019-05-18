import numpy as np

from pysurf import triangle_center


def test_triangle_center():
    """
    Tests the triangle center calculation function from its three points.

    Returns:
        None
    """
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 1])
    c = np.array([3, 1, 2])
    true_center = np.array([2, 2, 2])
    assert np.allclose(triangle_center(a, b, c), true_center)
