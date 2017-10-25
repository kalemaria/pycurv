import unittest
import time
import os.path
import numpy as np
import math

from pysurf_compact import pysurf_io as io
from pysurf_compact import (
    TriangleGraph, PointGraph, vector_voting, pexceptions)
from synthetic_surfaces import (
    PlaneGenerator, SphereGenerator, CylinderGenerator, SaddleGenerator,
    add_gaussian_noise_to_surface)


def percent_error_scalar(true_value, estimated_value):
    """
    Calculates the "percentage error": relative error as measure of accuracy.

    Args:
        true_value: true / accepted scalar value
        estimated_value: estimated / measured / experimental scalar value

    Returns:
        abs((true_value - estimated_value) / true_value) * 100.0
        the lower the error, the more accurate the estimated value
    """
    return abs((true_value - estimated_value) / true_value) * 100.0


def percent_error_vector(true_vector, estimated_vector):
    """
    Calculated a percentage error for 3D vectors.

    Args:
        true_vector (numpy.ndarray): true / accepted 3D vector
        estimated_vector (numpy.ndarray): estimated / measured / experimental 3D
            vector

    Returns:
        (1 - abs(np.dot(true_vector, estimated_vector))) * 100.0
        0 if the vectors are equal, 100 if they are perpendicular
    """
    return (1 - abs(np.dot(true_vector, estimated_vector))) * 100.0


class VectorVotingTestCase(unittest.TestCase):
    """
    Tests for vector_voting.py, assuming that other used functions are correct.
    """

    def parametric_test_plane_normals(self, half_size, res=30, noise=10,
                                      k=3, g_max=0, epsilon=0, eta=0):
        """
        Tests whether normals are correctly estimated for a plane surface with
        known orientation (parallel to to X and Y axes).

        Args:
            half_size (int): half size of the plane (from center to an edge)
            res (int, optional): resolution (number of divisions) in X and Y
                axes (default 30)
            noise (int, optional): determines variance of the Gaussian noise in
                percents of average triangle edge length (default 10), the noise
                is added on triangle vertex coordinates in its normal direction
            k (int, optional): parameter of Normal Vector Voting algorithm
                determining the geodesic neighborhood radius:
                g_max = k * average weak triangle graph edge length (default 3)
            g_max (float, optional): geodesic neighborhood radius in length unit
                of the graph, here voxels; if positive (default 0) this g_max
                will be used and k will be ignored
            epsilon (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2), default 0
            eta (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2) and "no preferred orientation" (class 3),
                default 0

        Notes:
            * Either g_max or k must be positive (if both are positive, the
              specified g_max will be used).
            * If epsilon = 0 and eta = 0 (default), all triangles will be
              classified as "surface patch" (class 1).

        Returns:
            None
        """
        base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
        if res == 0:
            fold = '{}synthetic_volumes/plane/noise{}/'.format(base_fold, noise)
        else:
            fold = '{}synthetic_surfaces/plane/res{}_noise{}/'.format(
                base_fold, res, noise)
        if not os.path.exists(fold):
            os.makedirs(fold)
        surf_file = '{}plane_half_size{}.surface.vtp'.format(fold, half_size)
        scale_factor_to_nm = 1  # assume it's already in nm
        # Actually can just give in any number for the scales, because they are
        # only used for ribosome density calculation or volumes / .mrc files
        # creation.
        scale_x = 2 * half_size
        scale_y = scale_x
        scale_z = scale_y
        files_fold = '{}files4plotting/'.format(fold)
        if not os.path.exists(files_fold):
            os.makedirs(files_fold)
        base_filename = "{}plane_half_size{}".format(files_fold, half_size)
        vtk_normal_errors_file = '{}.VTK.normal_errors.txt'.format(
            base_filename)
        if g_max > 0:
            surf_vv_file = '{}.VV_g_max{}_epsilon{}_eta{}.vtp'.format(
                base_filename, g_max, epsilon, eta)
            vv_normal_errors_file = (
                '{}.VV_g_max{}_epsilon{}_eta{}.normal_errors.txt'.format(
                    base_filename, g_max, epsilon, eta))
        elif k > 0:
            surf_vv_file = '{}.VV_k{}_epsilon{}_eta{}.vtp'.format(
                base_filename, k, epsilon, eta)
            vv_normal_errors_file = (
                '{}.VV_k{}_epsilon{}_eta{}.normal_errors.txt'.format(
                    base_filename, k, epsilon, eta))
        else:
            error_msg = ("Either g_max or k must be positive (if both are "
                         "positive, the specified g_max will be used).")
            raise pexceptions.PySegInputError(
                expr='parametric_test_plane_normals', msg=error_msg)

        print ("\n*** Generating a surface and a graph for a plane with half-"
               "size {} and {}% noise ***".format(half_size, noise))
        # If the .vtp file with the test surface does not exist, create it:
        if not os.path.isfile(surf_file):
            if res == 0:  # generate surface from a mask with gen_surface
                print "Sorry, not implemented yet"
                exit(0)
            else:  # generate surface directly with VTK
                pg = PlaneGenerator()
                plane = pg.generate_plane_surface(half_size, res)
                if noise > 0:
                    plane = add_gaussian_noise_to_surface(plane, percent=noise)
                io.save_vtp(plane, surf_file)

        # Reading in the .vtp file with the test triangle mesh and transforming
        # it into a triangle graph:
        t_begin = time.time()

        print '\nReading in the surface file to get a vtkPolyData surface...'
        surf = io.load_poly(surf_file)
        print ('\nBuilding the TriangleGraph from the vtkPolyData surface with '
               'curvatures...')
        tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
        tg.build_graph_from_vtk_surface(surf, verbose=False,
                                        reverse_normals=False)
        print tg.graph

        if k > 0:
            print "k = {}".format(k)
            # Find the average triangle edge length (l_ave) and calculate g_max:
            # (Do this here because in vector_voting average weak edge length of
            # the triangle graph is used, since the surface not passed)
            pg = PointGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
            pg.build_graph_from_vtk_surface(surf)
            l_ave = pg.calculate_average_edge_length()
            print "average triangle edge length = {}".format(l_ave)
            g_max = k * l_ave

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm:
        surf_vv = vector_voting(tg, k=0, g_max=g_max, epsilon=epsilon, eta=eta,
                                exclude_borders=False)
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        io.save_vtp(surf_vv, surf_vv_file)

        # Getting the initial and the estimated normals
        pos = [0, 1, 2]  # vector-property value positions
        vtk_normals = tg.graph.vertex_properties["normal"].get_2d_array(pos)
        vv_normals = tg.graph.vertex_properties["N_v"].get_2d_array(pos)
        # The shape is (3, <num_vertices>) - have to transpose to group the
        # respective x, y, z components to sub-arrays
        vtk_normals = np.transpose(vtk_normals)  # shape (<num_vertices>, 3)
        vv_normals = np.transpose(vv_normals)

        # Ground-truth normal is parallel to Z axis
        true_normal = np.array([0, 0, 1])

        # Computing the percentage errors of the initial (VTK) and estimated
        # (VV) normals wrt the true normal:
        vtk_normal_errors = np.array(map(
            lambda x: percent_error_vector(true_normal, x), vtk_normals))
        vv_normal_errors = np.array(map(
            lambda x: percent_error_vector(true_normal, x), vv_normals))

        # Writing the errors into files:
        io.write_values_to_file(vtk_normal_errors, vtk_normal_errors_file)
        io.write_values_to_file(vv_normal_errors, vv_normal_errors_file)

        # Asserting that all estimated normals are close to the true normal,
        # allowing error of 30%:
        for error in vv_normal_errors:
            msg = '{} is > {}%!'.format(error, 30)
            self.assertLessEqual(error, 30, msg=msg)

    def parametric_test_cylinder_T_2_curvatures(
            self, r, inverse=False, res=0, h=0, noise=0,
            k=3, g_max=0, epsilon=0, eta=0):
        """
        Tests whether minimal principal directions (T_2), as well as minimal and
        maximal principal curvatures are correctly estimated
        for an opened cylinder surface (without the circular planes) with known
        orientation (height, i.e. T_2, parallel to the Z axis).

        Args:
            r (int): cylinder radius in voxels
            inverse (boolean, optional): if True (default False), the sphere
                will have normals pointing outwards (negative curvature), else
                the other way around
            res (int, optional): if > 0 determines how many stripes around both
                approximate circles (and then triangles) the cylinder has, the
                surface is generated using vtkCylinderSource; If 0 (default)
                first a gaussian cylinder mask is generated and then surface
                using vtkMarchingCubes, in the latter case the cylinder is as
                high as the used mask box (2.5 * r) and is open at both sides
            h (int, optional): cylinder height in voxels, only needed if res > 0
                for the method using vtkCylinderSource, if res is 0 or h is 0
                (default), height will be set to 2.5 * r rounded up
            noise (int, optional): determines variance of the Gaussian noise in
                percents of average triangle edge length (default 0), the noise
                is added on triangle vertex coordinates in its normal direction
            k (int, optional): parameter of Normal Vector Voting algorithm
                determining the geodesic neighborhood radius:
                g_max = k * average weak triangle graph edge length (default 3)
            g_max (float, optional): geodesic neighborhood radius in length unit
                of the graph, here voxels; if positive (default 0) this g_max
                will be used and k will be ignored
            epsilon (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2), default 0
            eta (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2) and "no preferred orientation" (class 3),
                default 0

        Notes:
            * Either g_max or k must be positive (if both are positive, the
              specified g_max will be used).
            * If epsilon = 0 and eta = 0 (default), all triangles will be
              classified as "surface patch" (class 1).

        Returns:
            None
        """
        base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
        if res == 0:
            fold = '{}synthetic_surfaces/cylinder/noise{}/'.format(
                base_fold, noise)
        else:
            fold = '{}synthetic_surfaces/cylinder/res{}_noise{}/'.format(
                base_fold, res, noise)
        if not os.path.exists(fold):
            os.makedirs(fold)

        if res == 0 and h != 0:
            h = 0  # h has to be also 0 if res is 0
        if h == 0:
            h = int(math.ceil(r * 2.5))  # set h to 2.5 * radius, if not given

        surf_filebase = '{}cylinder_r{}_h{}'.format(fold, r, h)
        surf_file = '{}.surface.vtp'.format(surf_filebase)
        scale_factor_to_nm = 1  # assume it's already in nm
        # Actually can just give in any number for the scales, because they are
        # only used for ribosome density calculation or volumes / .mrc files
        # creation.
        scale_x = 2 * r
        scale_y = scale_x
        scale_z = h
        files_fold = '{}files4plotting/'.format(fold)
        if not os.path.exists(files_fold):
            os.makedirs(files_fold)
        if inverse:
            inverse_str = "inverse_"
        else:
            inverse_str = ""
        base_filename = "{}{}cylinder_r{}_h{}".format(
            files_fold, inverse_str, r, h)
        if g_max > 0:
            surf_vv_file = '{}.VV_g_max{}_epsilon{}_eta{}.vtp'.format(
                base_filename, g_max, epsilon, eta)
            T_2_errors_file = (
                '{}.VV_g_max{}_epsilon{}_eta{}.T_2_errors.txt'.format(
                    base_filename, g_max, epsilon, eta))
            kappa_1_errors_file = (
                '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                    base_filename, g_max, epsilon, eta))
            kappa_2_errors_file = (
                '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                    base_filename, g_max, epsilon, eta))
        elif k > 0:
            surf_vv_file = '{}.VV_k{}_epsilon{}_eta{}.vtp'.format(
                base_filename, k, epsilon, eta)
            T_2_errors_file = (
                '{}.VV_k{}_epsilon{}_eta{}.T_2_errors.txt'.format(
                    base_filename, k, epsilon, eta))
            kappa_1_errors_file = (
                '{}.VV_k{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                    base_filename, k, epsilon, eta))
            kappa_2_errors_file = (
                '{}.VV_k{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                    base_filename, k, epsilon, eta))
        else:
            error_msg = ("Either g_max or k must be positive (if both are "
                         "positive, the specified g_max will be used).")
            raise pexceptions.PySegInputError(
                expr='parametric_test_plane_normals', msg=error_msg)
        vtk_kappa_1_errors_file = ('{}.VTK.kappa_1_errors.txt'
                                   .format(base_filename))
        vtk_kappa_2_errors_file = ('{}.VTK.kappa_2_errors.txt'
                                   .format(base_filename))

        if inverse:
            print ("\n*** Generating a surface and a graph for an inverse "
                   "cylinder with radius {}, height {} and {}% noise ***"
                   .format(r, h, noise))
        else:
            print ("\n*** Generating a surface and a graph for a cylinder with "
                   "radius {}, height {} and {}% noise ***".format(
                    r, h, noise))
        # If the .vtp file with the test surface does not exist, create it:
        if not os.path.isfile(surf_file):
            cg = CylinderGenerator()
            if res == 0:  # generate surface from a mask with gen_surface
                cylinder = cg.generate_gauss_cylinder_surface(r)
            else:  # generate surface directly with VTK
                print "Warning: cylinder contains planes!"
                cylinder = cg.generate_cylinder_surface(r, h, res=50)
            if noise > 0:
                cylinder = add_gaussian_noise_to_surface(cylinder,
                                                         percent=noise)
            io.save_vtp(cylinder, surf_file)

        # Reading in the .vtp file with the test triangle mesh and transforming
        # it into a triangle graph:
        t_begin = time.time()

        print '\nReading in the surface file to get a vtkPolyData surface...'
        surf = io.load_poly(surf_file)
        print ('\nBuilding the TriangleGraph from the vtkPolyData surface with '
               'curvatures...')
        tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
        # VTK has opposite surface normals convention than we use
        # a graph with normals pointing outwards is generated (normal case
        # for this method; negative curvatures)
        if inverse:
            reverse_normals = False
        # a graph with normals pointing inwards is generated (positive
        # curvatures)
        else:
            reverse_normals = True
        tg.build_graph_from_vtk_surface(surf, verbose=False,
                                        reverse_normals=reverse_normals)
        print tg.graph

        if k > 0:
            print "k = {}".format(k)
            # Find the average triangle edge length (l_ave) and calculate g_max:
            # (Do this here because in vector_voting average weak edge length of
            # the triangle graph is used, since the surface not passed)
            pg = PointGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
            pg.build_graph_from_vtk_surface(surf)
            l_ave = pg.calculate_average_edge_length()
            print "average triangle edge length = {}".format(l_ave)
            g_max = k * l_ave

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm:
        surf_vv = vector_voting(tg, k=0, g_max=g_max, epsilon=epsilon, eta=eta,
                                exclude_borders=False)
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        io.save_vtp(surf_vv, surf_vv_file)

        # Getting the estimated (by VV) minimal principal directions (T_2)
        pos = [0, 1, 2]  # vector-property value positions
        if not inverse:
            T_2s = tg.graph.vertex_properties["T_2"].get_2d_array(pos)
        else:
            T_2s = tg.graph.vertex_properties["T_1"].get_2d_array(pos)
        # The shape is (3, <num_vertices>) - have to transpose to group the
        # respective x, y, z components to sub-arrays
        T_2s = np.transpose(T_2s)  # shape (<num_vertices>, 3)

        # Ground-truth T_2 vector is parallel to Z axis
        true_T_2 = np.array([0, 0, 1])

        # Computing the percentage errors of the estimated T_2 vectors wrt the
        # true one and writing them into a file:
        T_2_errors = np.array(map(
            lambda x: percent_error_vector(true_T_2, x), T_2s))
        io.write_values_to_file(T_2_errors, T_2_errors_file)

        # Getting principal curvatures from NVV and VTK from the output graph:
        kappa_1_values = tg.get_vertex_property_array("kappa_1")
        kappa_2_values = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1_values = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2_values = tg.get_vertex_property_array("min_curvature")

        # Calculating average principal curvatures
        kappa_1_avg = np.mean(kappa_1_values)
        kappa_2_avg = np.mean(kappa_2_values)

        # Ground-truth principal curvatures
        if inverse:
            true_kappa_1 = 0.0
            true_kappa_2 = - 1.0 / r
        else:
            true_kappa_1 = 1.0 / r
            true_kappa_2 = 0.0

        # Calculating the percentage errors of the principal curvatures and
        # writing them into files:
        if true_kappa_1 != 0:
            kappa_1_errors = np.array(map(
                lambda x: percent_error_scalar(true_kappa_1, x),
                kappa_1_values))
            io.write_values_to_file(kappa_1_errors, kappa_1_errors_file)
            vtk_kappa_1_errors = np.array(map(
                lambda x: percent_error_scalar(true_kappa_1, x),
                vtk_kappa_1_values))
            io.write_values_to_file(vtk_kappa_1_errors, vtk_kappa_1_errors_file)
        if true_kappa_2 != 0:
            kappa_2_errors = np.array(map(
                lambda x: percent_error_scalar(true_kappa_2, x),
                kappa_2_values))
            io.write_values_to_file(kappa_2_errors, kappa_2_errors_file)
            vtk_kappa_2_errors = np.array(map(
                lambda x: percent_error_scalar(true_kappa_2, x),
                vtk_kappa_2_values))
            io.write_values_to_file(vtk_kappa_2_errors, vtk_kappa_2_errors_file)

        # Asserting that all estimated T_2 vectors are close to the true vector,
        # allowing error of 30%:
        if not inverse:
            print "Testing the minimal principal directions (T_2)..."
        else:
            print "Testing the maximal principal directions (T_1)..."
        for error in T_2_errors:
            msg = '{} is > {}%!'.format(error, 30)
            self.assertLessEqual(error, 30, msg=msg)

        # Asserting that average principal curvatures are close to the correct
        # ones allowing percent error of +-30%
        allowed_error = 0.3 * max(abs(true_kappa_1), abs(true_kappa_2))
        if true_kappa_1 != 0:
            print "Testing the average maximal principal curvature (kappa_1)..."
            msg = '{} is not in [{}, {}]!'.format(
                kappa_1_avg, true_kappa_1 - allowed_error,
                true_kappa_1 + allowed_error)
            self.assertAlmostEqual(kappa_1_avg, true_kappa_1,
                                   delta=allowed_error, msg=msg)
        if true_kappa_2 != 0:
            print "Testing the average minimal principal curvature (kappa_2)..."
            msg = '{} is not in [{}, {}]!'.format(
                kappa_2_avg, true_kappa_2 - allowed_error,
                true_kappa_2 + allowed_error)
            self.assertAlmostEqual(kappa_2_avg, true_kappa_2,
                                   delta=allowed_error, msg=msg)

    def parametric_test_sphere_curvatures(
            self, radius, inverse=False, res=0, ico=0, noise=10,
            k=3, g_max=0, epsilon=0, eta=0, save_areas=False):
        """
        Runs all the steps needed to calculate curvatures for a test sphere
        with a given radius. Tests whether the curvatures are correctly
        estimated using Normal Vector Voting (VV) with a given g_max:

        kappa_1 = kappa_2 = 1/r; allowing some error.

        Args:
            radius (int): radius of the sphere
            inverse (boolean, optional): if True (default False), the sphere
                will have normals pointing outwards (negative curvature), else
                the other way around
            res (int, optional): if > 0 determines how many longitude and
                latitude stripes the UV sphere from vtkSphereSource has, the
                surface is triangulated; If 0 (default) and ico=0, first a
                gaussian sphere mask is generated and then surface using
                vtkMarchingCubes
            ico (int, optional): if > 0 (default 0) and res=0, an icosahedron
                with so many faces is used (1280 faces with radius 1 or 10 are
                available so far)
            noise (int, optional): determines variance of the Gaussian noise in
                percents of average triangle edge length (default 10), the noise
                is added on triangle vertex coordinates in its normal direction
            k (int, optional): parameter of Normal Vector Voting algorithm
                determining the geodesic neighborhood radius:
                g_max = k * average weak triangle graph edge length (default 3)
            g_max (float, optional): geodesic neighborhood radius in length unit
                of the graph, here voxels; if positive (default 0) this g_max
                will be used and k will be ignored
            epsilon (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2), default 0
            eta (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2) and "no preferred orientation" (class 3, see
                Notes), default 0
            save_areas (boolean, optional): if True (default False), also mesh
                triangle ares will be saved to a file

        Notes:
            * Either g_max or k must be positive (if both are positive, the
              specified g_max will be used).
            * If epsilon = 0 and eta = 0 (default), all triangles will be
              classified as "surface patch" (class 1).

        Returns:
            None
        """
        base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
        if res > 0:  # UV sphere is used
            fold = '{}synthetic_surfaces/sphere/res{}_noise{}/'.format(
                base_fold, res, noise)
        elif ico > 0:  # icosahedron sphere with so many faces is used
            fold = '{}synthetic_surfaces/sphere/ico{}_noise{}/'.format(
                base_fold, ico, noise)
        else:  # a "disco" sphere from gaussian mask is used
            fold = '{}synthetic_surfaces/sphere/noise{}/'.format(
                base_fold, noise)

        if not os.path.exists(fold):
            os.makedirs(fold)
        surf_filebase = '{}sphere_r{}'.format(fold, radius)
        surf_file = '{}.surface.vtp'.format(surf_filebase)
        scale_factor_to_nm = 1  # assume it's already in nm
        # Actually can just give in any number for the scales, because they are
        # only used for ribosome density calculation or volumes / .mrc files
        # creation.
        scale_x = 2 * radius
        scale_y = 2 * radius
        scale_z = 2 * radius
        files_fold = '{}files4plotting/'.format(fold)
        if not os.path.exists(files_fold):
            os.makedirs(files_fold)
        if inverse:
            inverse_str = "inverse_"
        else:
            inverse_str = ""
        base_filename = "{}{}sphere_r{}".format(
            files_fold, inverse_str, radius)
        if g_max > 0:
            surf_VV_file = '{}.VV_g_max{}_epsilon{}_eta{}.vtp'.format(
                base_filename, g_max, epsilon, eta)
            kappa_1_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1.txt'.format(
                base_filename, g_max, epsilon, eta)
            kappa_2_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2.txt'.format(
                base_filename, g_max, epsilon, eta)
            kappa_1_errors_file = (
                '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                    base_filename, g_max, epsilon, eta))
            kappa_2_errors_file = (
                '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                    base_filename, g_max, epsilon, eta))
        elif k > 0:
            surf_VV_file = '{}.VV_k{}_epsilon{}_eta{}.vtp'.format(
                base_filename, k, epsilon, eta)
            kappa_1_file = '{}.VV_k{}_epsilon{}_eta{}.kappa_1.txt'.format(
                base_filename, k, epsilon, eta)
            kappa_2_file = '{}.VV_k{}_epsilon{}_eta{}.kappa_2.txt'.format(
                base_filename, k, epsilon, eta)
            kappa_1_errors_file = (
                '{}.VV_k{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                    base_filename, k, epsilon, eta))
            kappa_2_errors_file = (
                '{}.VV_k{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                    base_filename, k, epsilon, eta))
        else:
            error_msg = ("Either g_max or k must be positive (if both are "
                         "positive, the specified g_max will be used).")
            raise pexceptions.PySegInputError(
                expr='parametric_test_sphere_curvatures', msg=error_msg)
        vtk_kappa_1_file = ('{}.VTK.kappa_1.txt'.format(base_filename))
        vtk_kappa_2_file = ('{}.VTK.kappa_2.txt'.format(base_filename))
        vtk_kappa_1_errors_file = ('{}.VTK.kappa_1_errors.txt'
                                   .format(base_filename))
        vtk_kappa_2_errors_file = ('{}.VTK.kappa_2_errors.txt'
                                   .format(base_filename))
        if save_areas:
            triangle_areas_file = ('{}.triangle_areas.txt'
                                   .format(base_filename))

        if inverse:
            print ("\n*** Generating a surface and a graph for an inverse "
                   "sphere with radius {} and {}% noise ***".format(
                    radius, noise))
        else:
            print ("\n*** Generating a surface and a graph for a sphere "
                   "with radius {} and {}% noise ***".format(radius, noise))
        # If the .vtp file with the test surface does not exist, create it:
        if not os.path.isfile(surf_file):
            sg = SphereGenerator()
            if res > 0:  # generate a UV sphere surface directly with VTK
                sphere = sg.generate_UV_sphere_surface(
                    r=radius, latitude_res=res, longitude_res=res)
                if noise > 0:
                    sphere = add_gaussian_noise_to_surface(sphere, percent=noise)
                io.save_vtp(sphere, surf_file)
            elif ico > 0:
                print ("Sorry, you have to generate the icosahedron sphere\n"
                       "beforehand e.g. with Blender, export it as STL file\n"
                       "and convert it to VTP file using the function\n"
                       "pysurf_io.stl_file_to_vtp_file, optionally add noise\n"
                       "with add_gaussian_noise_to_surface and save it as\n{}"
                       .format(surf_file))
                exit(0)
            else:  # generate a sphere surface from a gaussian mask
                sphere = sg.generate_gauss_sphere_surface(radius)
                if noise > 0:
                    sphere = add_gaussian_noise_to_surface(sphere, percent=noise)
                io.save_vtp(sphere, surf_file)

        # Reading in the .vtp file with the test triangle mesh and transforming
        # it into a triangle graph:
        t_begin = time.time()

        print '\nReading in the surface file to get a vtkPolyData surface...'
        surf = io.load_poly(surf_file)
        print ('\nBuilding the TriangleGraph from the vtkPolyData surface with '
               'curvatures...')
        tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
        # VTK has opposite surface normals convention than we use
        # a graph with normals pointing outwards is generated (normal case
        # for VTK; negative curvatures)
        if inverse:
            reverse_normals = False
        # a graph with normals pointing inwards is generated (VTK normals have
        # to be flipped, positive curvatures)
        else:
            reverse_normals = True
        tg.build_graph_from_vtk_surface(surf, verbose=False,
                                        reverse_normals=reverse_normals)
        print tg.graph

        if k > 0:
            print "k = {}".format(k)
            # Find the average triangle edge length (l_ave) and calculate g_max:
            # (Do this here because in vector_voting average weak edge length of
            # the triangle graph is used, since the surface not passed)
            pg = PointGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
            pg.build_graph_from_vtk_surface(surf)
            l_ave = pg.calculate_average_edge_length()
            print "average triangle edge length = {}".format(l_ave)
            g_max = k * l_ave

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm:
        surf_VV = vector_voting(tg, k=0, g_max=g_max, epsilon=epsilon, eta=eta,
                                exclude_borders=False)
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        io.save_vtp(surf_VV, surf_VV_file)

        # Getting principal curvatures from NVV and VTK from the output graph:
        kappa_1_values = tg.get_vertex_property_array("kappa_1")
        kappa_2_values = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1_values = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2_values = tg.get_vertex_property_array("min_curvature")
        if save_areas:
            triangle_areas = tg.get_vertex_property_array("area")

        # Calculating average principal curvatures
        kappa_1_avg = np.mean(kappa_1_values)
        kappa_2_avg = np.mean(kappa_2_values)

        # Ground truth principal curvatures
        true_curvature = 1.0 / radius
        if inverse:
            true_curvature *= -1

        # Calculating the percentage errors of the principal curvatures:
        kappa_1_errors = np.array(map(
            lambda x: percent_error_scalar(true_curvature, x), kappa_1_values))
        kappa_2_errors = np.array(map(
            lambda x: percent_error_scalar(true_curvature, x), kappa_2_values))
        vtk_kappa_1_errors = np.array(map(
            lambda x: percent_error_scalar(true_curvature, x),
            vtk_kappa_1_values))
        vtk_kappa_2_errors = np.array(map(
            lambda x: percent_error_scalar(true_curvature, x),
            vtk_kappa_2_values))

        # Writing all the curvature values and errors into files:
        io.write_values_to_file(kappa_1_values, kappa_1_file)
        io.write_values_to_file(kappa_1_errors, kappa_1_errors_file)
        io.write_values_to_file(kappa_2_values, kappa_2_file)
        io.write_values_to_file(kappa_2_errors, kappa_2_errors_file)
        io.write_values_to_file(vtk_kappa_1_values, vtk_kappa_1_file)
        io.write_values_to_file(vtk_kappa_1_errors, vtk_kappa_1_errors_file)
        io.write_values_to_file(vtk_kappa_2_values, vtk_kappa_2_file)
        io.write_values_to_file(vtk_kappa_2_errors, vtk_kappa_2_errors_file)
        if save_areas:
            io.write_values_to_file(triangle_areas, triangle_areas_file)

        # Asserting that all values of both principal curvatures are close to
        # the true value, allowing percent error of +-30%:
        # allowed_error = 0.3 * abs(true_curvature)
        # print "Testing the maximal principal curvatures (kappa_1)..."
        # for kappa_1 in kappa_1_values:
        #     msg = '{} is not in [{}, {}]!'.format(
        #         kappa_1, true_curvature - allowed_error,
        #         true_curvature + allowed_error)
        #     self.assertAlmostEqual(kappa_1, true_curvature,
        #                            delta=allowed_error, msg=msg)
        # print "Testing the minimal principal curvatures (kappa_2)..."
        # for kappa_2 in kappa_2_values:
        #     msg = '{} is not in [{}, {}]!'.format(
        #         kappa_2, true_curvature - allowed_error,
        #         true_curvature + allowed_error)
        #     self.assertAlmostEqual(kappa_2, true_curvature,
        #                            delta=allowed_error, msg=msg)

        # Asserting that average principal curvatures are close to the correct
        # ones allowing percent error of +-30%
        allowed_error = 0.3 * abs(true_curvature)
        print "Testing the average maximal principal curvature (kappa_1)..."
        msg = '{} is not in [{}, {}]!'.format(
            kappa_1_avg, true_curvature - allowed_error,
            true_curvature + allowed_error)
        self.assertAlmostEqual(kappa_1_avg, true_curvature,
                               delta=allowed_error, msg=msg)
        print "Testing the average minimal principal curvature (kappa_2)..."
        msg = '{} is not in [{}, {}]!'.format(
            kappa_2_avg, true_curvature - allowed_error,
            true_curvature + allowed_error)
        self.assertAlmostEqual(kappa_2_avg, true_curvature,
                               delta=allowed_error, msg=msg)

    def parametric_test_torus_curvatures(self, rr, csr, inverse=False,
                                         k=3, g_max=0, epsilon=0, eta=0):
        """
        Runs all the steps needed to calculate curvatures for a test torus
        with given radii using Normal Vector Voting (VV) with a given g_max.

        Args:
            rr (int): ring radius of the torus
            csr (int): cross-section radius of the torus
            inverse (boolean, optional): if True (default False), the sphere
                will have normals pointing outwards (negative curvature), else
                the other way around (choose in the way that csr < rr - csr)
            k (int, optional): parameter of Normal Vector Voting algorithm
                determining the geodesic neighborhood radius:
                g_max = k * average weak triangle graph edge length (default 3)
            g_max (float, optional): geodesic neighborhood radius in length unit
                of the graph, here voxels; if positive (default 0) this g_max
                will be used and k will be ignored
            epsilon (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2), default 0
            eta (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2) and "no preferred orientation" (class 3, see
                Notes), default 0

        Notes:
            * Either g_max or k must be positive (if both are positive, the
              specified g_max will be used).
            * If epsilon = 0 and eta = 0 (default), all triangles will be
              classified as "surface patch" (class 1).
            * csr should be much smaller than rr (csr < rr - csr).

        Returns:
            None
        """
        fold = '/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/torus/'

        if not os.path.exists(fold):
            os.makedirs(fold)
        surf_filebase = '{}torus_rr{}_csr{}_inner_part'.format(fold, rr, csr)  # TODO remove "_inner_part" when finished debugging
        surf_file = '{}.surface.vtp'.format(surf_filebase)
        scale_factor_to_nm = 1  # assume it's already in nm
        # Actually can just give in any number for the scales, because they are
        # only used for ribosome density calculation or volumes / .mrc files
        # creation.
        scale_x = 2 * (rr + csr)
        scale_y = 2 * scale_x
        scale_z = 2 * csr
        files_fold = '{}files4plotting/'.format(fold)
        if not os.path.exists(files_fold):
            os.makedirs(files_fold)
        if inverse:
            inverse_str = "inverse_"
        else:
            inverse_str = ""
        base_filename = "{}{}torus_rr{}_csr{}_inner_part".format(
            files_fold, inverse_str, rr, csr)  # TODO remove "_inner_part" when finished debugging
        if g_max > 0:
            surf_VV_file = '{}.VV_g_max{}_epsilon{}_eta{}.vtp'.format(
                base_filename, g_max, epsilon, eta)
        elif k > 0:
            surf_VV_file = '{}.VV_k{}_epsilon{}_eta{}.vtp'.format(
                base_filename, k, epsilon, eta)
        else:
            error_msg = ("Either g_max or k must be positive (if both are "
                         "positive, the specified g_max will be used).")
            raise pexceptions.PySegInputError(
                expr='parametric_test_sphere_curvatures', msg=error_msg)

        if inverse:
            print ("\n*** Generating a surface and a graph for an inverse "
                   "torus with ring radius {} and cross-section radius {} ***"
                   .format(rr, csr))
        else:
            print ("\n*** Generating a surface and a graph for a torus "
                   "with ring radius {} and cross-section radius {} ***"
                   .format(rr, csr))
        # If the .vtp file with the test surface does not exist, create it:
        if not os.path.isfile(surf_file):
            sg = SaddleGenerator()
            torus = sg.generate_parametric_torus(rr, csr)
            io.save_vtp(torus, surf_file)

        # Reading in the .vtp file with the test triangle mesh and transforming
        # it into a triangle graph:
        t_begin = time.time()

        print '\nReading in the surface file to get a vtkPolyData surface...'
        surf = io.load_poly(surf_file)
        print ('\nBuilding the TriangleGraph from the vtkPolyData surface with '
               'curvatures...')
        tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
        # VTK has opposite surface normals convention than we use
        # a graph with normals pointing outwards is generated (normal case
        # for VTK; negative curvatures)
        if inverse:
            reverse_normals = False
        # a graph with normals pointing inwards is generated (VTK normals have
        # to be flipped, positive curvatures)
        else:
            reverse_normals = True
        tg.build_graph_from_vtk_surface(surf, verbose=False,
                                        reverse_normals=reverse_normals)
        print tg.graph

        if k > 0:
            print "k = {}".format(k)
            # Find the average triangle edge length (l_ave) and calculate g_max:
            # (Do this here because in vector_voting average weak edge length of
            # the triangle graph is used, since the surface not passed)
            pg = PointGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
            pg.build_graph_from_vtk_surface(surf)
            l_ave = pg.calculate_average_edge_length()
            print "average triangle edge length = {}".format(l_ave)
            g_max = k * l_ave

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm:
        surf_VV = vector_voting(tg, k=0, g_max=g_max, epsilon=epsilon, eta=eta,
                                exclude_borders=True)
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        io.save_vtp(surf_VV, surf_VV_file)

        # Getting principal curvatures from NVV and VTK from the output graph:
        kappa_1_values = tg.get_vertex_property_array("kappa_1")
        kappa_2_values = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1_values = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2_values = tg.get_vertex_property_array("min_curvature")

        # Ground-truth principal curvatures
        irr = rr - csr
        orr = rr + csr
        if inverse:
            true_kappa_1 = (- 1.0 / orr, 1.0 / irr)
            print ("true kappa_1 between {} and {}".format(true_kappa_1[0],
                                                           true_kappa_1[1]))
            true_kappa_2 = - 1.0 / csr
            print ("true kappa_2 = {}".format(true_kappa_2))
        else:
            true_kappa_1 = 1.0 / csr
            print ("true kappa_1 = {}".format(true_kappa_1))
            true_kappa_2 = (- 1.0 / irr, 1.0 / orr)
            print ("true kappa_2 between {} and {}".format(true_kappa_2[0],
                                                           true_kappa_2[1]))

    # *** The following tests will be run by unittest ***

    # def test_plane_normals(self):
    #     """
    #     Tests whether normals are correctly estimated using Normal Vector
    #     Voting with a certain g_max for a plane surface with known orientation
    #     (parallel to to X and Y axes), certain size, resolution and noise
    #     level.
    #     """
    #     for n in [0, 5, 10]:
    #         for k in [3, 5]:  # 1 for noise=0
    #             self.parametric_test_plane_normals(10, res=30, noise=n, k=k)

    # def test_cylinder_T_2_curvatures(self):
    #     """
    #     Tests whether minimal principal directions (T_2) are correctly estimated
    #     using Normal Vector Voting with a certain g_max for an opened cylinder
    #     surface (without the circular planes) with known orientation (height,
    #     i.e. T_2, parallel to the Z axis), certain radius, height, resolution
    #     and noise level.
    #     """
    #     # for k in [3, 5]:
    #     #     self.parametric_test_cylinder_T_2_curvatures(10, noise=0, k=k)
    #     for n in [0]:  # 10, 5
    #         for k in [3]:  # 3, 5
    #             self.parametric_test_cylinder_T_2_curvatures(10, noise=n, k=k)

    def test_inverse_cylinder_T_2_curvatures(self):
        """
        Tests whether minimal principal directions (T_2) are correctly estimated
        using Normal Vector Voting with a certain g_max for an opened cylinder
        surface (without the circular planes) with known orientation (height,
        i.e. T_2, parallel to the Z axis), certain radius, height, resolution
        and noise level.
        """
        for k in [3]:  # 3, 5
            self.parametric_test_cylinder_T_2_curvatures(10, noise=0, k=k,
                                                         inverse=True)

    # def test_sphere_curvatures(self):
    #     """
    #     Tests whether curvatures are correctly estimated using Normal Vector
    #     Voting with a certain g_max for a sphere with a certain radius,
    #     resolution and noise level:
    #
    #     kappa1 = kappa2 = 1/5 = 0.2; 30% of difference is allowed
    #     """
    #     for n in [0]:  # 5, 10
    #         for k in [1]:  # 5, 3
    #             # self.parametric_test_sphere_curvatures(10, res=30, noise=n, k=k,
    #             #                                        save_areas=True)
    #             self.parametric_test_sphere_curvatures(
    #                 10, ico=1280, noise=n, k=k, save_areas=False)

    # def test_inverse_sphere_curvatures(self):
    #     """
    #     Tests whether curvatures are correctly estimated using Normal Vector
    #     Voting with a certain g_max for an inverse sphere with a certain
    #     radius, resolution and noise level:
    #
    #     kappa1 = kappa2 = -1/5 = -0.2; 30% of difference is allowed
    #     """
    #     for k in [3, 5]:
    #         self.parametric_test_sphere_curvatures(10, noise=0, k=k,
    #                                                inverse=True)

    # def test_torus_curvatures(self):
    #     """
    #     Runs parametric_test_torus_curvatures with certain parameters.
    #     """
    #     self.parametric_test_torus_curvatures(25, 10, inverse=False, k=1)


if __name__ == '__main__':
    unittest.main()
