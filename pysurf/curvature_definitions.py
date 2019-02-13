from math import pi, atan, sqrt

"""
Contains functions of different curvature definitions combining the maximal
principal curvature kappa_1 with the minimal principal curvature kappa_2.
"""

__author__ = 'kalemanov'


def calculate_gauss_curvature(kappa_1, kappa_2):  # TODO add docstings!
    return kappa_1 * kappa_2


def calculate_mean_curvature(kappa_1, kappa_2):
    return (kappa_1 + kappa_2) / 2


def calculate_shape_index(kappa_1, kappa_2):
    if kappa_1 == 0 and kappa_2 == 0:
        return 0
    else:
        return 2 / pi * atan((kappa_1 + kappa_2) / (kappa_1 - kappa_2))


def calculate_curvedness(kappa_1, kappa_2):
    return sqrt((kappa_1 ** 2 + kappa_2 ** 2) / 2)
