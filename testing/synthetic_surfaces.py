import vtk
import math
import numpy as np
import os

from pysurf_compact import pysurf_io as io
from pysurf_compact import pexceptions, surface_graphs
from synthetic_volumes import SphereMask, CylinderMask

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


def remove_non_triangle_cells(surface, verbose=False):
    """
    Removes non-triangle cells( e.g. lines and vertices) from the given surface.

    Args:
        surface (vtk.vtkPolyData): a surface
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        cleaned surface with only triangular cells (vtk.vtkPolyData)
    """
    if verbose:
        print('{} cells including non-triangles'.format(
            surface.GetNumberOfCells()))
    for i in range(surface.GetNumberOfCells()):
        # Get the cell i and remove it if it's not a triangle:
        cell = surface.GetCell(i)
        if not isinstance(cell, vtk.vtkTriangle):
            surface.DeleteCell(i)
    surface.RemoveDeletedCells()
    if verbose:
        print('{} cells after deleting non-triangle cells'.format(
            surface.GetNumberOfCells()))
    return surface


def add_gaussian_noise_to_surface(surface, percent=10, verbose=False):
    """
    Adds Gaussian noise to a surface by moving each triangle point in the
    direction of its normal vector.

    Args:
        surface (vtk.vtkPolyData): input surface
        percent (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 10)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        noisy surface (vtk.vtkPolyData)
    """
    # Find the average triangle edge length (l_ave)
    pg = surface_graphs.PointGraph(surface, scale_factor_to_nm=1,
                                   scale_x=1, scale_y=1, scale_z=1)
    pg.build_graph_from_vtk_surface()
    l_ave = pg.calculate_average_edge_length(verbose=verbose)

    # Variance of the noise is the given percent of l_ave
    var = percent / 100.0 * l_ave
    std = math.sqrt(var)  # standard deviation
    if verbose:
        print ("variance = {}".format(var))

    # Get the point normals of the surface
    point_normals = __get_point_normals(surface)
    if point_normals is None:
        print "No point normals were found. Computing normals..."
        surface = __compute_point_normals(surface)
        point_normals = __get_point_normals(surface)
        if point_normals is None:
            print "Failed to compute point normals! Exiting..."
            exit(0)
        else:
            print "Successfully computed point normals!"
    else:
        print "Point normals were found!"

    # Copy the surface and initialize vtkPoints data structure
    new_surface = vtk.vtkPolyData()
    new_surface.DeepCopy(surface)
    points = vtk.vtkPoints()

    # For each point, get its normal and randomly add noise from Gaussian
    # distribution with the wanted variance in the normal direction
    for i in xrange(new_surface.GetNumberOfPoints()):
        old_p = np.asarray(new_surface.GetPoint(i))
        normal_p = np.asarray(point_normals.GetTuple3(i))
        new_p = old_p + np.random.normal(scale=std) * normal_p
        new_x, new_y, new_z = new_p
        points.InsertPoint(i, new_x, new_y, new_z)

    # Set the points of the surface copy and return it
    new_surface.SetPoints(points)
    return new_surface


def __get_point_normals(surface):
    normals = surface.GetPointData().GetNormals()
    if normals is None:
        normals = surface.GetPointData().GetArray("Normals")
    return normals


def __compute_point_normals(surface):
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(surface)
    normal_generator.ComputePointNormalsOn()
    normal_generator.Update()
    surface = normal_generator.GetOutput()
    return surface


def __copy_and_name_array(da, name):
    """
    Copies data array and gives it a new name.

    Args:
        da (vtkDataArray): data array
        name (str): wanted name for the array

    Returns:
        copy of the data array with the name or None, if input was None
    """
    if da is not None:
        outda = da.NewInstance()
        outda.DeepCopy(da)
        outda.SetName(name)
        return outda
    else:
        return None


def is_coordinate_on_sphere_surface(x, y, z, r, error=0.0):
    """
    Checks whether a coordinate is on smooth sphere surface. Only works if the
    sphere is centered at (0, 0, 0)!
    Args:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
        r (int or float): smooth sphere radius
        error (float): allowed relative +/- error of the radius

    Returns:
        True if the coordinate is close to the sphere surface within the error,
        False if it is not
    """
    sq_dist_from_center = x ** 2 + y ** 2 + z ** 2
    if ((1 - error) * r) ** 2 <= sq_dist_from_center <= ((1 + error) * r) ** 2:
        return True
    else:
        return False


def are_triangle_vertices_on_smooth_sphere_surface(surface, r, center=[0, 0, 0],
                                                   error=0.0):
    """
    Checks and prints out how many triangle vertices (from total) of the given
    surface are on smooth sphere surface.
    Args:
        surface (vtk.vtkPolyData): sphere surface approximated by triangles
        r (int or float): smooth sphere radius
        center (float[3]): coordinates in the center of the sphere surface
        error (float): allowed relative +/- error of the radius

    Returns:
        None
    """
    num_points_on_smooth_sphere_surface = 0
    # for each triangle vertex (point)
    # After subtracting center from all points, the new center becomes (0, 0, 0)
    for i in xrange(surface.GetNumberOfPoints()):
        point = np.asarray(surface.GetPoint(i))
        x = point[0] - center[0]
        y = point[1] - center[1]
        z = point[2] - center[2]
        if is_coordinate_on_sphere_surface(x, y, z, r, error=error) is True:
            num_points_on_smooth_sphere_surface += 1

    print "From {} points, {} are on smooth sphere surface".format(
        surface.GetNumberOfPoints(), num_points_on_smooth_sphere_surface)


class PlaneGenerator(object):
    """
    A class for generating triangular-mesh surface of a plane.
    """
    @staticmethod
    def generate_plane_surface(half_size=10, res=30):
        """
        Generates a square plane surface with triangular cells.

        The plane has a center at (0, 0, 0) and normals (0, 0, 1), i.e. the
        plane is parallel to X and Y axes.

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
    def generate_UV_sphere_surface(r=10.0, latitude_res=100,
                                   longitude_res=100, verbose=False):
        """
        Generates a UV sphere surface with only triangular cells.

        Args:
            r (float, optional): sphere radius (default 10.0)
            latitude_res (int, optional): latitude resolution (default 100)
            longitude_res (int, optional): latitude resolution (default 100)
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            a sphere surface (vtk.vtkPolyData)
        """
        if verbose:
            print("Generating a sphere with radius={}, latitude resolution={} "
                  "and longitude resolution={}".format(r, latitude_res,
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
        sphere_surface = remove_non_triangle_cells(cleaner.GetOutput(),
                                                   verbose=verbose)
        return sphere_surface

    @staticmethod
    def generate_sphere_surface(r):
        """
        Generates a sphere surface with a given radius using a filled binary
        sphere mask and the Marching Cubes method. The resulting surface tends
        to follow the sharp voxels outline.

        Args:
            r (int): sphere radius

        Returns:
            a sphere surface (vtk.vtkPolyData)
        """
        # Generate a sphere mask with radius == r
        box = int(math.ceil(r * 2.5))
        sm = SphereMask()
        mask_np = sm.generate_sphere_mask(r, box)
        mask_vti = io.numpy_to_vti(mask_np)

        # Iso-surface
        surfaces = vtk.vtkMarchingCubes()
        surfaces.SetInputData(mask_vti)
        surfaces.ComputeNormalsOn()
        surfaces.ComputeGradientsOn()
        surfaces.SetValue(0, 1)
        surfaces.Update()

        return surfaces.GetOutput()

    @staticmethod
    def generate_gauss_sphere_surface(r):
        """
        Generates a sphere surface with a given radius using a gaussian sphere
        mask and the Marching Cubes method. The resulting surface is smooth.

        Args:
            r (int): sphere radius

        Returns:
            a sphere surface (vtk.vtkPolyData)
        """
        # Generate a gauss sphere mask with 3 * sigma == radius r
        sg = r / 3.0
        box = int(math.ceil(r * 2.5))
        sm = SphereMask()
        mask_np = sm.generate_gauss_sphere_mask(sg, box)
        # center_voxel = [box / 2, box / 2, box / 2]
        # print center_voxel
        mask_vti = io.numpy_to_vti(mask_np)

        # Calculate threshold th to get the desired radius r
        th = math.exp(-(r ** 2) / (2 * (sg ** 2)))

        # Iso-surface
        surfaces = vtk.vtkMarchingCubes()
        surfaces.SetInputData(mask_vti)
        surfaces.ComputeNormalsOn()
        surfaces.ComputeGradientsOn()
        surfaces.SetValue(0, th)
        surfaces.Update()

        return surfaces.GetOutput()


class CylinderGenerator(object):
    """
    A class for generating triangular-mesh surface of a cylinder.
    """
    @staticmethod
    def generate_cylinder_surface(r=10.0, h=20.0, res=100):
        """
        Generates a cylinder surface with minimal number of triangular cells
        and two circular planes.

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

    @staticmethod
    def generate_gauss_cylinder_surface(r):
        """
        Generates a cylinder surface with a given radius using a gaussian
        cylinder mask and the Marching Cubes method. The resulting surface is
        smooth.

        Args:
            r (int): cylinder radius

        Returns:
            a cylinder surface (vtk.vtkPolyData)
        """
        # Generate a gauss cylinder mask with 3 * sigma == radius r
        sg = r / 3.0
        box = int(math.ceil(r * 2.5))
        cm = CylinderMask()
        mask_np = cm.generate_gauss_cylinder_mask(sg, box)
        mask_vti = io.numpy_to_vti(mask_np)

        # Calculate threshold th to get the desired radius r
        th = math.exp(-(r ** 2) / (2 * (sg ** 2)))

        # Iso-surface
        surfaces = vtk.vtkMarchingCubes()
        surfaces.SetInputData(mask_vti)
        surfaces.ComputeNormalsOn()
        surfaces.ComputeGradientsOn()
        surfaces.SetValue(0, th)
        surfaces.Update()

        return surfaces.GetOutput()


class SaddleGenerator(object):
    """
    A class for generating triangular-mesh surfaces containing a saddle surface
    (negative gaussian curvature).
    """
    # @staticmethod
    # def generate_one_sheeted_hyperboloid(a, c, h):
    #     pass

    @staticmethod
    def generate_parametric_torus(rr, csr):
        """
        Generates a torus surface with triangular cells.

        Args:
            rr (int or float): ring radius
            csr (int or float): cross-section radius

        Returns:
            a torus surface (vtk.vtkPolyData)
        """
        torus = vtk.vtkParametricTorus()
        torus.SetRingRadius(rr)
        torus.SetCrossSectionRadius(csr)

        source = vtk.vtkParametricFunctionSource()
        source.SetParametricFunction(torus)
        source.Update()

        # The surface has nasty discontinuities from the way the edges are
        # generated, so pass it though a CleanPolyDataFilter to merge any
        # points which are coincident or very close
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(source.GetOutputPort())
        cleaner.SetTolerance(0.005)
        cleaner.Update()

        # Might contain non-triangle cells after cleaning - remove them
        surface = remove_non_triangle_cells(cleaner.GetOutput())
        return surface


def main():
    """
    Main function generating some sphere and cylinder surfaces.

    Returns:
        None
    """
    fold = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"

    # # Plane
    # pg = PlaneGenerator()
    # plane = pg.generate_plane_surface(half_size=10, res=30)
    # # io.save_vtp(plane, fold + "plane_half_size10_res30.vtp")
    # noisy_plane = add_gaussian_noise_to_surface(plane, percent=10)
    # io.save_vtp(noisy_plane,
    #             "{}plane_half_size10_res30_noise10.vtp".format(fold))

    # UV Sphere
    sg = SphereGenerator()
    sphere = sg.generate_UV_sphere_surface(r=10, latitude_res=50,
                                           longitude_res=50)
    are_triangle_vertices_on_smooth_sphere_surface(sphere, r=10, error=0.001)
    # sphere_noise = add_gaussian_noise_to_surface(sphere, percent=10)
    # io.save_vtp(sphere_noise, fold + "sphere_r10_res50_noise10.vtp")

    # Sphere from gauss mask
    sg = SphereGenerator()
    sphere = sg.generate_gauss_sphere_surface(r=10)
    # io.save_vtp(sphere, "{}gauss_sphere_r10.vtp".format(fold))
    are_triangle_vertices_on_smooth_sphere_surface(
        sphere, r=10, center=[12, 12, 12], error=0.009)
    # sphere_noise = add_gaussian_noise_to_surface(sphere, percent=10)
    # io.save_vtp(sphere_noise, "{}gauss_sphere_r10_noise10.vtp".format(fold))

    # # Cylinder
    # cg = CylinderGenerator()
    # # cylinder_r10_h20 = cg.generate_cylinder_surface(r=10, h=20, res=50)
    # # io.save_vtp(cylinder_r10_h20, fold + "cylinder_r10_h20_res50.vtp")
    # rad = 10
    # cylinder = cg.generate_gauss_cylinder_surface(rad)
    # io.save_vtp(cylinder, "{}gauss_cylinder_r{}.vtp".format(fold, rad))

    # # icosphere noise addition
    # os.chdir(fold)
    # poly = io.load_poly("sphere/ico1280_noise0/sphere_r10.surface.vtp")
    # poly_noise = add_gaussian_noise_to_surface(poly, percent=10)
    # io.save_vtp(poly_noise, "sphere/ico1280_noise10/sphere_r10.surface.vtp")

    # Torus
    # sg = SaddleGenerator()
    # rr = 25
    # csr = 10
    # torus = sg.generate_parametric_torus(rr, csr)
    # io.save_vtp(torus, "{}torus_rr{}_csr{}.vtp".format(fold, rr, csr))


if __name__ == "__main__":
    main()
