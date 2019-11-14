from math import pi, atan, sqrt

"""
Contains functions of different curvature definitions combining the maximal
principal curvature kappa_1 with the minimal principal curvature kappa_2.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def calculate_gauss_curvature(kappa_1, kappa_2):
    """
    Calculates the Gauss curvature from the principal curvatures.

    Args:
        kappa_1 (float): maximal principal curvature
        kappa_2 (float): minimal principal curvature

    Returns:
        Gauss curvature (float)
    """
    return kappa_1 * kappa_2


def calculate_mean_curvature(kappa_1, kappa_2):
    """
    Calculates the mean curvature from the principal curvatures.

    Args:
        kappa_1 (float): maximal principal curvature
        kappa_2 (float): minimal principal curvature

    Returns:
        mean curvature (float)
    """
    return (kappa_1 + kappa_2) / 2


def calculate_shape_index(kappa_1, kappa_2):
    """
    Calculates the shape index (Koenderink and van Doorn et al., Image and
    Vision Computing, 1992) from the principal curvatures.

    Args:
        kappa_1 (float): maximal principal curvature
        kappa_2 (float): minimal principal curvature

    Returns:
        shape index (float)
    """
    if kappa_1 == 0 and kappa_2 == 0:
        return 0
    else:
        return 2 / pi * atan((kappa_1 + kappa_2) / (kappa_1 - kappa_2))


def calculate_curvedness(kappa_1, kappa_2):
    """
    Calculates the curvedness (Koenderink and van Doorn et al., Image and
    Vision Computing, 1992) from the principal curvatures.

    Args:
        kappa_1 (float): maximal principal curvature
        kappa_2 (float): minimal principal curvature

    Returns:
        curvedness (float)
    """
    return sqrt((kappa_1 ** 2 + kappa_2 ** 2) / 2)
