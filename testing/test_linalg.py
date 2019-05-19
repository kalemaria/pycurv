import numpy as np

from pysurf import (triangle_center, triangle_area_cross_product,
                    triangle_area_heron, euclidean_distance)


def test_triangle_center():
    """
    Tests the triangle center calculation function from its three points.

    Returns:
        None
    """
    # acute triangle:
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 1])
    c = np.array([3, 1, 2])
    true_center = np.array([2, 2, 2])
    assert np.allclose(triangle_center(a, b, c), true_center)
    # obtuse triangle:
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 9])
    c = np.array([0, 1, 2])
    true_center = np.array([1, 2, 4.667])
    assert np.allclose(triangle_center(a, b, c), true_center, rtol=1e-03)


def test_triangle_area():
    """
    Tests the triangle area calculation functions from its three points.

    Returns:
        None
    """
    # acute triangle:
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 1])
    c = np.array([3, 1, 2])
    true_area = 2.598
    assert round(triangle_area_cross_product(a, b, c), 3) == true_area
    assert round(triangle_area_heron(a, b, c), 3) == true_area
    # obtuse triangle:
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 9])
    c = np.array([0, 1, 2])
    true_area = 3.536
    assert round(triangle_area_cross_product(a, b, c), 3) == true_area
    assert round(triangle_area_heron(a, b, c), 3) == true_area


def test_euclidean_distance():
    """
    Tests the triangle area calculation function from its three points.

    Returns:
        None
    """
    # acute triangle:
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 1])
    c = np.array([3, 1, 2])
    true_ab = 2.449
    true_bc = 2.449
    true_ac = 2.449
    assert round(euclidean_distance(a, b), 3) == true_ab
    assert round(euclidean_distance(b, c), 3) == true_bc
    assert round(euclidean_distance(a, c), 3) == true_ac
    # obtuse triangle:
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 9])
    c = np.array([0, 1, 2])
    true_ab = 6.164
    true_bc = 7.55
    true_ac = 1.732
    assert round(euclidean_distance(a, b), 3) == true_ab
    assert round(euclidean_distance(b, c), 3) == true_bc
    assert round(euclidean_distance(a, c), 3) == true_ac
