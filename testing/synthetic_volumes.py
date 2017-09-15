import numpy as np
import math
import os

from pysurf_compact import pysurf_io as io
from pysurf_compact import run_gen_surface, pexceptions

"""A set of functions and classes for generating artificial segmentation volumes
(masks) of geometrical objects."""


class SphereMask(object):
    """
    A class for generating a sphere mask.
    """
    @staticmethod
    def generate_sphere_mask(r=10, box=23):
        # TODO docstring
        # Create a 3D grid with center (0, 0, 0) in the middle of the box
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        xx, yy, zz = np.mgrid[low:high, low:high, low:high]

        # Calculate squared distances from the center
        sq_dist_from_center = xx ** 2 + yy ** 2 + zz ** 2

        # Threshold using squared radius to generate a filled sphere
        sphere = (sq_dist_from_center <= r ** 2).astype(int)
        return sphere


class CylinderMask(object):
    """
    A class for generating a cylinder mask.
    """
    @staticmethod
    def generate_cylinder_mask(r=10, h=21, box=27, t=0):
        # TODO docstring
        if h > box:
            error_msg = "Cylinder high has to be maximum the box size."
            raise pexceptions.PySegInputError(
                expr='CylinderMask.generate_cylinder_mask', msg=error_msg)

        # Create a 2D grid with center (0, 0) in the middle
        # (slice through the box in XY plane)
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        xx, yy = np.mgrid[low:high, low:high]

        # Calculate squared distances from the center
        sq_dist_from_center = xx ** 2 + yy ** 2

        # Threshold using squared radius to generate a filled circle
        circle = (sq_dist_from_center <= r ** 2).astype(int)

        # Generate a cylinder consisting of N=box circles stacked in Z dimension
        cylinder = np.zeros(shape=(box, box, box))
        bottom = int(math.floor((box - h) / 2.0))
        top = bottom + h
        for zz in range(bottom, top):
            cylinder[:, :, zz] = circle

        if t > 0:  # Generate a hollow cylinder with thickness t
            # Generate an inner cylinder smaller by t ...
            if h < box:  # ... in radius and by 2t in high ("closed" cylinder)
                inner_cylinder = CylinderMask.generate_cylinder_mask(
                    r=r-t, h=h-2*t, box=box)
            else:  # ... in radius only (h == box, "open" cylinder)
                inner_cylinder = CylinderMask.generate_cylinder_mask(
                    r=r-t, h=h, box=box)
            # Subtract it from the bigger cylinder
            cylinder = cylinder - inner_cylinder

        return cylinder


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

    # # Generate a filled sphere mask
    # sm = SphereMask()
    # sphere_r10_box23 = sm.generate_sphere_mask()
    # io.save_numpy(sphere_r10_box23, fold + "sphere_r10_box23.mrc")
    #
    # # From the mask, generate a surface (is correct)
    # run_gen_surface(sphere_r10_box23, fold + "sphere_r10_box23")  # 2088 cells

    # Generate a filled cylinder mask
    # cm = CylinderMask()
    # cylinder_r10_h23_box23 = cm.generate_cylinder_mask(r=10, h=23, box=23)
    # io.save_numpy(cylinder_r10_h23_box23, fold + "cylinder_r10_h_23_box23.mrc")
    # cylinder_r10_h21_box23 = cm.generate_cylinder_mask(r=10, h=21, box=23)
    # io.save_numpy(cylinder_r10_h21_box23, fold + "cylinder_r10_h_21_box23.mrc")
    # cylinder_r10_h21_box27 = cm.generate_cylinder_mask(r=10, h=21, box=27)
    # io.save_numpy(cylinder_r10_h21_box27, fold + "cylinder_r10_h21_box27.mrc")
    # cylinder_r10_h21_box33 = cm.generate_cylinder_mask(r=10, h=21, box=33)
    # io.save_numpy(cylinder_r10_h21_box33, fold + "cylinder_r10_h21_box33.mrc")

    # From the mask, generate a surface (is incorrect with different "borders")
    # run_gen_surface(cylinder_r10_h23_box23, fold + "cylinder_r10_h23_box23")
    # # 4044 cells
    # run_gen_surface(cylinder_r10_h21_box23, fold + "cylinder_r10_h21_box23")
    # # 3349 cells
    # run_gen_surface(cylinder_r10_h21_box27, fold + "cylinder_r10_h21_box27")
    # # 3785 cells
    # run_gen_surface(cylinder_r10_h21_box33, fold + "cylinder_r10_h21_box33")
    # # 3152 cells

    # Generate a hollow cylinder mask
    cm = CylinderMask()
    thickness = 1  # (for 2 there were some artifacts)
    # cylinder_r10_h21_box27 = cm.generate_cylinder_mask(
    #     r=10, h=21, box=27, t=thickness)
    # io.save_numpy(cylinder_r10_h21_box27,
    #               "{}cylinder_r10_h21_box27_t{}.mrc".format(fold, thickness))
    # cylinder_r10_h23_box23 = cm.generate_cylinder_mask(
    #     r=10, h=23, box=23, t=thickness)
    # io.save_numpy(cylinder_r10_h23_box23,
    #               "{}cylinder_r10_h23_box23_t{}.mrc".format(fold, thickness))
    cylinder_r10_h27_box27 = cm.generate_cylinder_mask(
        r=10, h=27, box=27, t=thickness)
    io.save_numpy(cylinder_r10_h27_box27,
                  "{}cylinder_r10_h27_box27_t{}.mrc".format(fold, thickness))

    # From the mask, generate a surface
    # run_gen_surface(cylinder_r10_h21_box27,  # 3592 cells
    #                 "{}cylinder_r10_h21_box27_t{}".format(fold, thickness))
    # run_gen_surface(cylinder_r10_h23_box23,  # 3140 cells, little unevenness
    #                 "{}cylinder_r10_h23_box23_t{}".format(fold, thickness))
    run_gen_surface(cylinder_r10_h27_box27,  # 3672 cells, little unevenness
                    "{}cylinder_r10_h27_box27_t{}".format(fold, thickness))


if __name__ == "__main__":
    main()
