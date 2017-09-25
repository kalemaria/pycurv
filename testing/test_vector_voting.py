import unittest
import time
import os.path
import numpy as np

from pysurf_compact import pysurf_io as io
from pysurf_compact import (
    TriangleGraph, vector_voting, pexceptions, run_gen_surface)
from synthetic_surfaces import SphereGenerator
from synthetic_volumes import SphereMask


def percent_error(true_value, estimated_value):
    """
    Calculates the "percent error": relative error as measure of accuracy.
    Args:
        true_value: true / accepted value
        estimated_value: estimated / measures / experimental value

    Returns:
        abs((true_value - estimated_value) / true_value) * 100.0
    """
    return abs((true_value - estimated_value) / true_value) * 100.0


class VectorVotingTestCase(unittest.TestCase):
    """
    Tests for vector_voting.py, assuming that other used functions are correct.
    """

    def parametric_test_sphere_curvatures(
            self, radius=5, res=50, inverse=False,
            k=3, g_max=0, epsilon=0, eta=0):
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
            res (int): if > 0 (default 50) determines how many stripes (and then
                triangles) the sphere has (longitude and latitude), the surface
                is generated directly using VTK; If 0 first a sphere mask is
                generated and then surface using gen_surface function
            k (int, optional): parameter of Normal Vector Voting algorithm
                determining the geodesic neighborhood radius:
                g_max = k * average weak triangle graph edge length (default 3)
            g_max (float, optional): geodesic neighborhood radius in length unit
                of the graph, here voxels; if positive (default 0.0) this g_max
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
        """
        base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
        if res == 0:
            fold = '{}synthetic_volumes/sphere/'.format(base_fold)
        else:
            fold = '{}synthetic_surfaces/sphere/res{}/'.format(base_fold, res)
        if not os.path.exists(fold):
            os.makedirs(fold)
        surf_file = '{}sphere_r{}.surface.vtp'.format(fold, radius)
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
            surf_VV_file = '{}.VV_k{}_epsilon{}_eta{}.vtp'
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

        if inverse:
            print ("\n*** Generating a surface and a graph for an inverse "
                   "sphere with radius {} ***".format(radius))
        else:
            print ("\n*** Generating a surface and a graph for a sphere "
                   "with radius {} ***".format(radius))
        # If the .vtp file with the test surface does not exist, create it:
        if not os.path.isfile(surf_file):
            if res == 0:
                sm = SphereMask()
                sphere_mask = sm.generate_sphere_mask(
                    r=radius, box=(radius * 2 + 3), t=1)
                run_gen_surface(
                    sphere_mask, '{}sphere_r{}'.format(fold, radius))
            else:
                sg = SphereGenerator()
                sphere = sg.generate_sphere_surface(
                    r=radius, latitude_res=res, longitude_res=res)
                io.save_vtp(sphere, surf_file)

        # Reading in the .vtp file with the test triangle mesh and transforming
        # it into a triangle graph:
        t_begin = time.time()

        print '\nReading in the surface file to get a vtkPolyData surface...'
        surf = io.load_poly(surf_file)
        print ('\nBuilding the TriangleGraph from the vtkPolyData surface with '
               'curvatures...')
        tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
        if res == 0:  # generate surface from a mask with gen_surface
            # a graph with normals pointing outwards is generated (negative
            # curvatures)
            if inverse:
                reverse_normals = True
            # a graph with normals pointing inwards is generated (normal case
            # for this method; positive curvatures)
            else:
                reverse_normals = False
        else:
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

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm:
        surf_VV = vector_voting(tg, k=k, g_max=g_max, epsilon=epsilon, eta=eta,
                                exclude_borders=False)
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        io.save_vtp(surf_VV, surf_VV_file)

        # Getting principal curvatures from NVV and VTK from the output graph:
        kappa_1_values = tg.get_vertex_property_array("kappa_1")
        kappa_2_values = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1_values = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2_values = tg.get_vertex_property_array("min_curvature")

        # Calculating average principal curvatures
        kappa_1_avg = np.mean(kappa_1_values)
        kappa_2_avg = np.mean(kappa_2_values)

        # Correct principal curvatures
        correct_curvature = 1.0 / radius
        if inverse:
            correct_curvature *= -1

        # Calculating the percent errors of the principal curvatures:
        kappa_1_errors = np.array(map(
            lambda x: percent_error(correct_curvature, x), kappa_1_values))
        kappa_2_errors = np.array(map(
            lambda x: percent_error(correct_curvature, x), kappa_2_values))
        vtk_kappa_1_errors = np.array(map(
            lambda x: percent_error(correct_curvature, x), vtk_kappa_1_values))
        vtk_kappa_2_errors = np.array(map(
            lambda x: percent_error(correct_curvature, x), vtk_kappa_2_values))

        # Writing all the curvature values and errors into files:
        io.write_values_to_file(kappa_1_values, kappa_1_file)
        io.write_values_to_file(kappa_1_errors, kappa_1_errors_file)
        io.write_values_to_file(kappa_2_values, kappa_2_file)
        io.write_values_to_file(kappa_2_errors, kappa_2_errors_file)
        io.write_values_to_file(vtk_kappa_1_values, vtk_kappa_1_file)
        io.write_values_to_file(vtk_kappa_1_errors, vtk_kappa_1_errors_file)
        io.write_values_to_file(vtk_kappa_2_values, vtk_kappa_2_file)
        io.write_values_to_file(vtk_kappa_2_errors, vtk_kappa_2_errors_file)

        # # Asserting that all values of both principal curvatures are close to
        # # the correct value, allowing percent error of +-30%:
        # allowed_error = 0.3 * abs(correct_curvature)
        # print "Testing the maximal principal curvatures (kappa_1)..."
        # for kappa_1 in kappa_1_values:
        #     msg = '{} is not in [{}, {}]!'.format(
        #         kappa_1, correct_curvature - allowed_error,
        #         correct_curvature + allowed_error)
        #     self.assertAlmostEqual(kappa_1, correct_curvature,
        #                            delta=allowed_error, msg=msg)
        # print "Testing the minimal principal curvatures (kappa_2)..."
        # for kappa_2 in kappa_2_values:
        #     msg = '{} is not in [{}, {}]!'.format(
        #         kappa_2, correct_curvature - allowed_error,
        #         correct_curvature + allowed_error)
        #     self.assertAlmostEqual(kappa_2, correct_curvature,
        #                            delta=allowed_error, msg=msg)

        # Asserting that average principal curvatures are close to the correct
        # ones allowing percent error of +-30%
        allowed_error = 0.3 * abs(correct_curvature)
        print "Testing the average maximal principal curvature (kappa_1)..."
        msg = '{} is not in [{}, {}]!'.format(
            kappa_1_avg, correct_curvature - allowed_error,
            correct_curvature + allowed_error)
        self.assertAlmostEqual(kappa_1_avg, correct_curvature,
                               delta=allowed_error, msg=msg)
        print "Testing the average minimal principal curvature (kappa_2)..."
        msg = '{} is not in [{}, {}]!'.format(
            kappa_2_avg, correct_curvature - allowed_error,
            correct_curvature + allowed_error)
        self.assertAlmostEqual(kappa_2_avg, correct_curvature,
                               delta=allowed_error, msg=msg)

    def test_sphere_curvatures(self):
        """
        Tests whether curvatures for a sphere with a certain radius and
        resolution are correctly estimated using Normal Vector Voting with a
        certain g_max:

        kappa1 = kappa2 = 1/5 = 0.2; 30% of difference is allowed
        """
        self.parametric_test_sphere_curvatures(
            radius=5, res=50, g_max=1, epsilon=0, eta=0)

    def test_inverse_sphere_curvatures(self):
        """
        Tests whether curvatures for an inverse sphere with a certain radius and
        resolution are correctly estimated using Normal Vector Voting with a
        certain g_max:

        kappa1 = kappa2 = -1/5 = -0.2; 30% of difference is allowed
        """
        self.parametric_test_sphere_curvatures(
            radius=5, res=50, inverse=True, g_max=1, epsilon=0, eta=0)

    def test_sphere_from_volume_curvatures(self):
        """
        Tests whether curvatures for a sphere with a certain radius are
        correctly estimated using Normal Vector Voting with a certain g_max:

        kappa1 = kappa2 = 1/20 = 0.05; 30% of difference is allowed
        """
        self.parametric_test_sphere_curvatures(
            radius=20, res=0, g_max=7, epsilon=0, eta=0)

    def test_inverse_sphere_from_volume_radius20_g_max3_curvatures(self):
        """
        Tests whether curvatures for an inverse sphere with a certain radius
        are correctly estimated using Normal Vector Voting with a certain
        g_max:

        kappa1 = kappa2 = -1/20 = -0.05; 30% of difference is allowed
        """
        self.parametric_test_sphere_curvatures(
            radius=20, res=0, inverse=True, g_max=7, epsilon=0, eta=0)

if __name__ == '__main__':
    unittest.main()
