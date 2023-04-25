import numpy as np
import math
from scipy import ndimage

from pycurv import pycurv_io as io
from pycurv import pexceptions, run_gen_surface

"""
A set of functions and classes for generating artificial segmentation volumes
(masks) of geometrical objects.

Author: Maria Salfer (Max Planck Institute for Biochemistry)
"""

__author__ = 'Maria Salfer'


class SphereMask(object):
    """
    A class for generating a sphere mask.
    """
    @staticmethod
    def generate_sphere_mask(r=10, box=23, t=0):
        """
        Generates a 3D volume with a sphere mask.

        Args:
            r (int): radius in voxels difference (default 10)
            box (int): size of the box in x, y, and z dimensions in voxels, has
                to be at least 2 * r + 1 (default 23)
            t (int): thickness of a hollow sphere in voxels, if 0 (default) a
                filled sphere is generated

        Returns:
            3D volume with the sphere mask (numpy.ndarray)
        """
        if 2 * r + 1 > box:
            raise pexceptions.PySegInputError(
                expr='SphereMask.generate_sphere_mask',
                msg="Sphere diameter has to fit into the box.")
        # Create a 3D grid with center (0, 0, 0) in the middle of the box
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        x, y, z = np.mgrid[low:high, low:high, low:high]

        # Calculate squared distances from the center
        sq_dist_from_center = x ** 2 + y ** 2 + z ** 2

        # Threshold using squared radius to generate a filled sphere
        sphere = (sq_dist_from_center <= r ** 2).astype(int)

        if t > 0:  # Generate a hollow sphere with thickness t
            # Generate an inner sphere with radius smaller by t
            inner_sphere = SphereMask.generate_sphere_mask(r=r-t, box=box)

            # Subtract it from the bigger sphere
            sphere = sphere - inner_sphere

        return sphere

    @staticmethod
    def generate_gauss_sphere_mask(sg, box):
        """
        Generates a 3D volume with a smooth sphere mask.

        Args:
            sg (float): sigma of the gaussian formula
            box (int): size of the box in x, y, and z dimensions in voxels

        Returns:
            3D volume with the gaussian sphere mask (numpy.ndarray)
        """
        # Create a 3D grid with center (0, 0, 0) in the middle of the box
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        x, y, z = np.mgrid[low:high, low:high, low:high]

        # Calculate the gaussian 3D function with center (0, 0, 0) and amplitude
        # 1 at the center
        gauss_sphere = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * (sg ** 2)))

        return gauss_sphere


class CylinderMask(object):
    """
    A class for generating a cylinder mask.
    """
    @staticmethod
    def generate_cylinder_mask(r=10, h=20, box=23, t=0, opened=False):
        """
        Generates a 3D volume with a cylinder mask.

        Args:
            r (int): radius in voxels difference (default 10)
            h (int): height in voxels difference (default 20)
            box (int): size of the box in x, y, and z dimensions in voxels, has
                to be at least 2 * r + 1 or h (the bigger of them, default 23)
            t (int): thickness of a hollow cylinder in voxels, if 0 (default) a
                filled cylinder is generated
            opened (boolean): if True (default False) and t>0, an "opened"
                cylinder mask without the circular planes is generated

        Returns:
            3D volume with the cylinder mask (numpy.ndarray)
        """
        if 2 * r + 1 > box:
            raise pexceptions.PySegInputError(
                expr='CylinderMask.generate_cylinder_mask',
                msg="Cylinder diameter has to fit into the box.")
        if h > box:
            raise pexceptions.PySegInputError(
                expr='CylinderMask.generate_cylinder_mask',
                msg="Cylinder high has to fit into the box.")

        # Create a 2D grid with center (0, 0) in the middle
        # (slice through the box in XY plane)
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        x, y = np.mgrid[low:high, low:high]

        # Calculate squared distances from the center
        sq_dist_from_center = x ** 2 + y ** 2

        # Threshold using squared radius to generate a filled circle
        circle = (sq_dist_from_center <= r ** 2).astype(int)

        # Generate a cylinder consisting of N=h circles stacked in Z dimension
        cylinder = np.zeros(shape=(box, box, box))
        bottom = int(math.floor((box - h - 1) / 2.0))
        top = bottom + h + 1  # not including
        for z in range(bottom, top):
            cylinder[:, :, z] = circle

        if t > 0:  # Generate a hollow cylinder with thickness t
            # Generate an inner cylinder smaller by t ...
            if opened is True:  # ... in radius only ("open" cylinder)
                inner_cylinder = CylinderMask.generate_cylinder_mask(
                    r=r-t, h=h, box=box)
            else:  # ... in radius and by 2t in height ("closed" cylinder)
                inner_cylinder = CylinderMask.generate_cylinder_mask(
                    r=r-t, h=h-2*t, box=box)

            # Subtract it from the bigger cylinder
            cylinder = cylinder - inner_cylinder

        return cylinder

    @staticmethod
    def generate_gauss_cylinder_mask(sg, box):
        """
        Generates a 3D volume with a smooth cylinder mask.

        Args:
            sg (float): sigma of the gaussian formula
            box (int): size of the box in x, y, and z dimensions in voxels

        Returns:
            3D volume with the gaussian cylinder mask (numpy.ndarray)
        """
        # Create a 3D grid with center (0, 0, 0) in the middle of the box
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        x, y, z = np.mgrid[low:high, low:high, low:high]

        # Calculate the gaussian 3D function with center (0, 0, 0) and amplitude
        # 1 at the center
        gauss_cylinder = np.exp(-(x ** 2 + y ** 2) / (2 * (sg ** 2)))

        return gauss_cylinder


class TorusMask(object):
    """
    A class for generating a torus mask.
    """
    @staticmethod
    def generate_torus_mask(c, a, box, t=0):
        """
        Generates a 3D volume with a torus mask.

        Args:
            c (int or float): ring radius
            a (int or float): cross-section radius
            box (int): size of the box in x, y, and z dimensions in voxels, has
                to be at least 2 * c + 2 * a + 1
            t (int): thickness of a hollow torus in voxels, if 0 (default) a
                filled torus is generated

        Returns:
            3D volume with the torus mask (numpy.ndarray)
        """
        if 2 * c + 2 * a + 1 > box:
            raise pexceptions.PySegInputError(
                expr='TorusMask.generate_torus_mask',
                msg="Torus has to fit into the box.")
        # Create a 3D grid with center (0, 0, 0) in the middle of the box
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        x, y, z = np.mgrid[low:high, low:high, -a:a+1]

        # Threshold to generate a filled torus
        torus = (
                (c - np.sqrt(x ** 2 + y ** 2)) ** 2 + z ** 2 <= a ** 2
        ).astype(int)

        if t > 0:  # Generate a hollow sphere with thickness t
            # Generate an inner sphere with radius smaller by t
            inner_torus = TorusMask.generate_torus_mask(c=c, a=a-t, box=box)

            # Subtract it from the bigger sphere
            torus = torus - inner_torus

        return torus


class ConeMask(object):
    """
    A class for generating a cone mask.
    """
    @staticmethod
    def generate_cone_mask(r=10, h=20, box=23, t=0, opened=False):
        """
        Generates a 3D volume with a cone voxel mask.

        Args:
            r (int): radius in voxels difference (default 10)
            h (int): height in voxels difference (default 20)
            box (int): size of the box in x, y, and z dimensions in voxels, has
                to be at least 2 * r + 1 or h + 1 (the bigger of them, default
                23)
            t (int): thickness of a hollow cone in voxels, if 0 (default) a
                filled cone is generated
            opened (boolean): if True (default False) and t>0, an "opened"
                cone mask without the circular plane is generated

        Returns:
            3D volume with the cone mask (numpy.ndarray)
        """
        if 2 * r + 1 > box:
            raise pexceptions.PySegInputError(
                expr='ConeMask.generate_cylinder_mask',
                msg="Cone diameter has to fit into the box.")
        if h + 1 > box:
            raise pexceptions.PySegInputError(
                expr='ConeMask.generate_cylinder_mask',
                msg="Cone high has to fit into the box.")

        # Generate a cone consisting of N=h different sized circles stacked
        # in Z dimension
        cone = np.zeros(shape=(box, box, box))
        bottom = int(math.floor((box - h - 1) / 2.0))
        for u in range(h + 1):
            # Create a 2D grid with center (0, 0) in the middle
            # (slice through the box in XY plane)
            low = - math.floor(box / 2.0)
            high = math.ceil(box / 2.0)
            x, y = np.mgrid[low:high, low:high]

            # Threshold using implicit cone equation to generate a filled circle
            circle = (x ** 2 + y ** 2) / ((r / float(h)) ** 2) <= (h - u) ** 2
            z = u + bottom
            cone[:, :, z] = circle.astype(int)

        if t > 0:  # Generate a hollow cone with thickness t
            # Generate an inner cone smaller by t ...
            if opened is True:  # ... in radius only ("open" cone)
                inner_cone = ConeMask.generate_cone_mask(
                    r=r-t, h=h-t, box=box)
            else:  # ... in radius and by 2t in height ("closed" cone)
                inner_cone = ConeMask.generate_cone_mask(
                    r=r-t, h=h-2*t, box=box)

            # Subtract it from the bigger cone
            cone = cone - inner_cone

        return cone


def main():
    """
    Code generating some sphere and cylinder masks and from them signed
    surfaces.

    Returns:
        None
    """
    fold = "/fs/pool/pool-ruben/Maria/curvature/synthetic_volumes/"
    fold2 = "/fs/pool/pool-ruben/Maria/curvature/missing_wedge_sphere/"

    # Generate a gaussian sphere mask
    sm = SphereMask()
    r = 10
    sg = r / 3.0
    box = int(math.ceil(r * 2.5))
    gauss_sphere = sm.generate_gauss_sphere_mask(sg, box)
    io.save_numpy(gauss_sphere, "{}gauss_sphere_mask_r{}_box{}.mrc".format(
        fold, r, box))

    # Generate a filled sphere mask
    sm = SphereMask()
    r = 10  # r=10: 2088 cells, r=15: 60 cells, r=20: 0 cells
    sphere = sm.generate_sphere_mask(r, box=(2 * r + 3))
    io.save_numpy(sphere, "{}sphere_r{}.mrc".format(fold, r))

    # From the mask, generate a surface (is correct)
    run_gen_surface(sphere, "{}sphere_r{}".format(fold, r))

    # Generate a hollow sphere mask
    sm = SphereMask()
    r = 20  # r=10: 1856 cells, r=15: 4582 cells, r=20: 8134 cells
    box = int(math.ceil(r * 2.5))
    thickness = 1
    sphere = sm.generate_sphere_mask(r, box, t=thickness)
    io.save_numpy(sphere, "{}sphere_r{}_t{}_box{}.mrc".format(
        fold2, r, thickness, box))
    # From the mask, generate a surface
    run_gen_surface(sphere, "{}sphere_r{}_t{}".format(fold2, r, thickness))
    # Gaussian smoothing
    sigma = 0.2
    smooth_sphere = ndimage.filters.gaussian_filter(
        sphere.astype(np.float32), sigma)
    print(np.min(smooth_sphere))
    print(np.max(smooth_sphere))
    io.save_numpy(smooth_sphere, "{}smooth_sphere_r{}_t{}_box{}.mrc".format(
        fold2, r, thickness, box))

    # Generate a gaussian cylinder mask
    cm = CylinderMask()
    r = 100
    sm_sg = 100 / 3.0
    sm_box = int(r * 2.5)
    gauss_cylinder = cm.generate_gauss_cylinder_mask(sm_sg, sm_box)
    io.save_numpy(gauss_cylinder, "{}gauss_cylinder_sg{}_box{}.mrc".format(
        fold, sm_sg, sm_box))

    # Generate a filled cylinder mask
    r = 10
    h = 20
    t = 0
    surf_filebase = '{}cylinder_r{}_h{}'.format(fold, r, h)
    cm = CylinderMask()
    box = max(2 * r + 1, h + 1) + 2
    cylinder_mask = cm.generate_cylinder_mask(r, h, box, t=t)
    io.save_numpy(cylinder_mask, surf_filebase + ".mrc")

    # Generate a hollow cylinder mask
    r = 10
    h = 20
    t = 1
    surf_filebase = '{}cylinder_r{}_h{}_t{}'.format(fold, r, h, t)
    cm = CylinderMask()
    box = max(2 * r + 1, h + 1) + 2
    cylinder_mask = cm.generate_cylinder_mask(r, h, box, t=t, opened=True)
    io.save_numpy(cylinder_mask, surf_filebase + ".mrc")
    run_gen_surface(cylinder_mask, surf_filebase)

    # Generate a filled cone mask
    r = 6
    h = 8
    t = 0
    surf_filebase = '{}cone/cone_r{}_h{}'.format(fold, r, h)
    cm = ConeMask()
    box = max(2 * r + 1, h + 1) + 2
    cone_mask = cm.generate_cone_mask(r, h, box, t=t)
    io.save_numpy(cone_mask, surf_filebase + ".mrc")

    # Generate a hollow cone mask
    r = 6
    h = 6
    t = 1
    surf_filebase = '{}cone/cone_r{}_h{}_t{}'.format(fold, r, h, t)
    cm = ConeMask()
    box = max(2 * r + 1, h + 1) + 2
    cone_mask = cm.generate_cone_mask(r, h, box, t=t, opened=True)
    io.save_numpy(cone_mask, surf_filebase + ".mrc")
    run_gen_surface(cone_mask, surf_filebase)


if __name__ == "__main__":
    main()
