import numpy as np
from math import acos


def absolute_error_scalar(true_value, estimated_value):
    """
    Calculates the "absolute error" as measure of accuracy for scalars.

    Args:
        true_value: true / accepted scalar value
        estimated_value: estimated / measured / experimental scalar value

    Returns:
        abs(true_value - estimated_value)
        the lower the error, the more accurate the estimated value
    """
    return abs(true_value - estimated_value)


def relative_error_scalar(true_value, estimated_value):
    """
    Calculates the "relative error" as measure of accuracy for scalars.

    Args:
        true_value: true / accepted scalar value
        estimated_value: estimated / measured / experimental scalar value

    Returns:
        abs((true_value - estimated_value) / true_value)
        if true_value = 0, just abs(true_value - estimated_value)
        the lower the error, the more accurate the estimated value
    """
    if true_value == 0:
        return abs(true_value - estimated_value)
    else:
        return abs((true_value - estimated_value) / true_value)


def error_vector(true_vector, estimated_vector):
    """
    Calculates the error for 3D vectors.

    Args:
        true_vector (numpy.ndarray): true / accepted 3D vector
        estimated_vector (numpy.ndarray): estimated / measured / experimental 3D
            vector

    Returns:
        1 - abs(np.dot(true_vector, estimated_vector))
        0 if the vectors are parallel, 1 if they are perpendicular
    """
    return 1 - abs(np.dot(true_vector, estimated_vector))


def angular_error_vector(true_vector, estimated_vector):
    """
    Calculates the "angular error" for 3D vectors.

    Args:
        true_vector (numpy.ndarray): true / accepted 3D vector
        estimated_vector (numpy.ndarray): estimated / measured / experimental 3D
            vector

    Returns:
        acos(abs(np.dot(true_vector, estimated_vector)))
        angle in radians between two vectors
    """
    try:
        acos_arg = abs(np.dot(true_vector, estimated_vector))
        angular_error = acos(acos_arg)
    except ValueError:
        if acos_arg > 1:
            acos_arg = 1.0
        elif acos_arg < 0:
            acos_arg = 0.0
        angular_error = acos(acos_arg)
    return angular_error


def calculate_histogram_area(counts, bin_edges):
    """
    Calculates normalized area of a cumulative histogram (maximal is 1), where
    maximal count must be 1.

    Args:
        counts (ndarray):  normalized frequency of the values in the bins,
            maximal value must be 1
        bin_edges (Tuple[ndarray, float]): bin edges of the bins, length of this
            array is one more than of the "counts" array

    Returns:
        normalized area
    """
    area = 0.0
    bin_width = 0.0
    for i, count in enumerate(counts):
        bin_width = bin_edges[i+1] - bin_edges[i]
        area += bin_width * count
    max_count = 1
    num_bins = len(counts)
    max_area = bin_width * max_count * num_bins
    normalized_area = area / max_area
    return normalized_area
