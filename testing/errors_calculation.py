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
    return acos(abs(np.dot(true_vector, estimated_vector)))
