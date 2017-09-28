import vtk
import math
import numpy as np

from pysurf_compact import pysurf_io as io
from pysurf_compact import pexceptions, PointGraph

"""A set of functions and classes for generating artificial surfaces of
geometrical objects."""


def is_positive_number(arg, input_error=True):
    """
    Checks whether an argument is a positive number.

    Args:
        arg: argument
        input_error (boolean): if True (default), raises PySegInputError

    Returns:
        True if the argument if a positive number, False otherwise
    """
    if (isinstance(arg, int) or isinstance(arg, float)) and arg > 0:
        return True
    else:
        if input_error:
            error_msg = "Argument must be a positive integer or float number."
            raise pexceptions.PySegInputError(expr='is_positive_number',
                                              msg=error_msg)
        return False


def remove_non_triangle_cells(surface):
    """
    Removes non-triangle cells( e.g. lines and vertices) from the given surface.

    Args:
        surface (vtk.vtkPolyData): a surface

    Returns:
        cleaned surface with only triangular cells (vtk.vtkPolyData)
    """
    print('{} cells including non-triangles'.format(surface.GetNumberOfCells()))
    for i in range(surface.GetNumberOfCells()):
        # Get the cell i and remove if if it's not a triangle:
        cell = surface.GetCell(i)
        if not isinstance(cell, vtk.vtkTriangle):
            surface.DeleteCell(i)
    surface.RemoveDeletedCells()
    print('{} cells after deleting non-triangle cells'.format(
        surface.GetNumberOfCells()))
    return surface


def add_gaussian_noise_to_surface(surface, percent=10, only_z=True,
                                  verbose=False):
    """
    Adds Gaussian noise to a surface by moving points coordinates.
    Args:
        surface (vtk.vtkPolyData): input surface
        percent (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 10)
        only_z (boolean, optional): if True (default), noise will be added only
            in the z dimension, otherwise in all dimensions
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        noisy surface (vtk.vtkPolyData)
    """
    # Find the average triangle edge length (l_ave)
    pg = PointGraph(scale_factor_to_nm=1, scale_x=1, scale_y=1, scale_z=1)
    pg.build_graph_from_vtk_surface(surface)
    l_ave = pg.calculate_average_edge_length(verbose=verbose)

    # Variance of the noise is the given percent of l_ave
    var = percent / 100.0 * l_ave
    std = math.sqrt(var)  # standard deviation
    if verbose:
        print ("variance = {}".format(var))

    # Copy the surface and initialize vtkPoints data structure
    new_surface = vtk.vtkPolyData()
    new_surface.DeepCopy(surface)
    points = vtk.vtkPoints()

    # For each point, randomly add noise from Gaussian distribution with the
    # wanted variance
    for i in xrange(surface.GetNumberOfPoints()):
        x, y, z = surface.GetPoint(i)
        if only_z:
            new_z = np.random.normal(loc=z, scale=std)
            points.InsertPoint(i, x, y, new_z)
        else:
            new_x = np.random.normal(loc=x, scale=std)
            new_y = np.random.normal(loc=y, scale=std)
            new_z = np.random.normal(loc=z, scale=std)
            # new_p = old_p + np.random.normal(scale=std) * normal  TODO!
            points.InsertPoint(i, new_x, new_y, new_z)
    # Set the points of the surface copy
    new_surface.SetPoints(points)

    return new_surface


class PlaneGenerator(object):
    """
    A class for generating triangular-mesh surface of a plane.
    """
    @staticmethod
    def generate_plane_surface(half_size=10, res=30):
        """
        Generates a square plane surface with triangular cells.

        The sphere will have a center at (0, 0, 0) and normals (0, 0, 1) -
        parallel to X and Y axes.

        Args:
            half_size (int): half size of the plane (from center to an edge)
            res (int): resolution (number of divisions) in X and Y axes

        Returns:
            a plane surface (vtk.vtkPolyData)
        """
        print("Generating a plane with half size={} and resolution={}".format(
            half_size, res))
        plane = vtk.vtkPlaneSource()
        # plane.SetCenter(0, 0, 0)
        plane.SetNormal(0, 0, 1)
        plane.SetOrigin(-half_size, -half_size, 0)
        plane.SetPoint1(half_size, -half_size, 0)
        plane.SetPoint2(-half_size, half_size, 0)
        plane.SetResolution(res, res)

        # The plane is made of strips, so pass it through a triangle filter
        # to get a triangle mesh
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(plane.GetOutputPort())
        tri.Update()

        plane_surface = tri.GetOutput()
        print('{} cells'.format(plane_surface.GetNumberOfCells()))
        return plane_surface


class SphereGenerator(object):
    """
    A class for generating triangular-mesh surface of a sphere.
    """
    @staticmethod
    def generate_sphere_surface(r=10.0, latitude_res=100,
                                longitude_res=100):
        """
        Generates a sphere surface with only triangular cells.

        Args:
            r (float, optional): sphere radius (default 10.0)
            latitude_res (int, optional): latitude resolution (default 100)
            longitude_res (int, optional): latitude resolution (default 100)

        Returns:
            a sphere surface (vtk.vtkPolyData)
        """
        print("Generating a sphere with radius={}, latitude resolution={} and "
              "longitude resolution={}".format(r, latitude_res,
                                               longitude_res))
        is_positive_number(r)
        sphere = vtk.vtkSphereSource()
        # the origin around which the sphere should be centered
        sphere.SetCenter(0.0, 0.0, 0.0)
        # polygonal discretization in latitude direction
        sphere.SetPhiResolution(latitude_res)
        # polygonal discretization in longitude direction
        sphere.SetThetaResolution(longitude_res)
        # the radius of the sphere
        sphere.SetRadius(r)
        # sphere.LatLongTessellationOn() #doesn't work as expected (default Off)
        # print sphere.GetLatLongTessellation()

        # The sphere is made of strips, so pass it through a triangle filter
        # to get a triangle mesh
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(sphere.GetOutputPort())

        # The sphere has nasty discontinuities from the way the edges are
        # generated, so pass it though a CleanPolyDataFilter to merge any
        # points which are coincident or very close
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(tri.GetOutputPort())
        cleaner.SetTolerance(0.0005)
        cleaner.Update()

        # Might contain non-triangle cells after cleaning - remove them
        sphere_surface = remove_non_triangle_cells(cleaner.GetOutput())
        return sphere_surface


class CylinderGenerator(object):
    """
    A class for generating triangular-mesh surface of a cylinder.
    """
    @staticmethod
    def generate_cylinder_surface(r=10.0, h=20.0, res=100):
        """
        Generates a cylinder surface with only triangular cells.

        Args:
            r (float, optional): cylinder radius (default 10.0)
            h (float, optional): cylinder high (default 20.0)
            res (int, optional): resolution (default 100)

        Returns:
            a cylinder surface (vtk.vtkPolyData)
        """
        print("Generating a cylinder with radius={}, height={} and "
              "resolution={}".format(r, h, res))
        is_positive_number(r)
        is_positive_number(h)
        cylinder = vtk.vtkCylinderSource()
        # the origin around which the cylinder should be centered
        cylinder.SetCenter(0, 0, 0)
        # the radius of the cylinder
        cylinder.SetRadius(r)
        # the high of the cylinder
        cylinder.SetHeight(h)
        # polygonal discretization
        cylinder.SetResolution(res)

        # The cylinder is made of strips, so pass it through a triangle filter
        # to get a triangle mesh
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(cylinder.GetOutputPort())

        # The cylinder has nasty discontinuities from the way the edges are
        # generated, so pass it though a CleanPolyDataFilter to merge any
        # points which are coincident or very close
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(tri.GetOutputPort())
        cleaner.SetTolerance(0.005)
        cleaner.Update()

        # Might contain non-triangle cells after cleaning - remove them
        cylinder_surface = remove_non_triangle_cells(cleaner.GetOutput())
        return cylinder_surface


def main():
    """
    Main function generating some sphere and cylinder surfaces.

    Returns:
        None
    """
    fold = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"

    # Plane
    pg = PlaneGenerator()
    plane = pg.generate_plane_surface(half_size=10, res=30)
    # io.save_vtp(plane, fold + "plane_half_size10_res30.vtp")
    noisy_plane = add_gaussian_noise_to_surface(plane, percent=10, only_z=True)
    io.save_vtp(noisy_plane, fold + "plane_half_size10_res30_noise10%z.vtp")

    # # Sphere
    # sg = SphereGenerator()
    # sphere_r10 = sg.generate_sphere_surface(r=10, latitude_res=50,
    #                                         longitude_res=50)
    # io.save_vtp(sphere_r10, fold + "sphere_r10_res50.vtp")
    #
    # # Cylinder
    # cg = CylinderGenerator()
    # cylinder_r10_h20 = cg.generate_cylinder_surface(r=10, h=20, res=50)
    # io.save_vtp(cylinder_r10_h20, fold + "cylinder_r10_h20_res50.vtp")


if __name__ == "__main__":
    main()
