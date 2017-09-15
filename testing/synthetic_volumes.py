import numpy as np
import math
import os

from pysurf_compact import pysurf_io as io
from pysurf_compact import run_gen_surface

"""A set of functions and classes for generating artificial segmentation volumes
(masks) of geometrical objects."""


class SphereMask(object):
    """
    A class for generating a sphere mask.
    """
    @staticmethod
    def generate_sphere_mask(r=10.0, box=23):
        # TODO docstring
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        xx, yy, zz = np.mgrid[low:high, low:high, low:high]
        sq_dist_from_0_0 = xx ** 2 + yy ** 2 + zz ** 2
        sphere = (sq_dist_from_0_0 <= r ** 2).astype(int)  # filled sphere
        return sphere


def main():
    """
    Main function generating some sphere and cylinder masks and from them signed
    surfaces.

    Returns:
        None
    """
    fold = "/fs/pool/pool-ruben/Maria/curvature/synthetic_volumes/"
    if not os.path.exists(fold):
        os.makedirs(fold)

    # Generate a sphere mask
    sm = SphereMask()
    sphere_r10_box23 = sm.generate_sphere_mask()
    io.save_numpy(sphere_r10_box23, fold + "sphere_r10_box23.mrc")

    # From the mask, generate a surface
    run_gen_surface(sphere_r10_box23, fold + "sphere_r10_box23")


if __name__ == "__main__":
    main()
