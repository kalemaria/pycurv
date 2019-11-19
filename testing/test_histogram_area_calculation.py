import numpy as np

from errors_calculation import calculate_histogram_area

"""
A unit test for testing the function calculating a normalized histogram area.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def test_calculate_histogram_area():
    area1 = calculate_histogram_area(counts=np.ones(10),
                                     bin_edges=np.linspace(0, 1, 11))
    assert round(area1, 7) == 1

    area0 = calculate_histogram_area(counts=np.zeros(10),
                                     bin_edges=np.linspace(0, 1, 11))
    assert area0 == 0
