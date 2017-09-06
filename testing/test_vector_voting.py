import unittest
import time
import os.path
import numpy as np

from pysurf_compact import pysurf_io as io
from pysurf_compact import TriangleGraph, vector_voting, pexceptions
from synthetic_surfaces import SphereGenerator


class VectorVotingTestCase(unittest.TestCase):
    """
    Tests for vector_voting.py, assuming that other used functions are correct.
    """

    def parametric_test_sphere_curvatures(self, radius=5, res=50, inverse=False,
                                          k=3, g_max=0, epsilon=0, eta=0):
        """
        Runs all the steps needed to calculate curvatures for a test sphere
        with a given radius and resolution. Tests whether the curvatures are
        correctly estimated using NVV with a given g_max:

        kappa_1 = kappa_2 = 1/r; 30% of difference is allowed.

        Args:
            radius:
            res:
            inverse:
            k:
            g_max:
            epsilon:
            eta:
        """
        if inverse:
            print ("\n*** Generating a surface and a graph for an inverse "
                   "sphere with radius {} and resolution {} ***"
                   .format(radius, res))
        else:
            print ("\n*** Generating a surface and a graph for a sphere "
                   "with radius {} and resolution {} ***".format(radius, res))

        fold = '/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/'
        surf_file = '{}sphere_r{}_res{}.vtp'.format(fold, radius, res)
        scale_factor_to_nm = 1  # assume it's already in nm
        # Actually can just give in any number for the scales, because they are
        # only used for ribosome density calculation or volumes / .mrc files
        # creation.
        scale_x = 2 * radius
        scale_y = 2 * radius
        scale_z = 2 * radius
        fold2 = '{}files4plotting/'.format(fold)
        if inverse:
            inverse_str = "inverse_"
        else:
            inverse_str = ""
        base_filename = "{}{}sphere_r{}_res{}".format(
            fold2, inverse_str, radius, res)
        if g_max > 0:
            surf_VV_file = '{}.VV_g_max{}_epsilon{}_eta{}.vtp'.format(
                base_filename, g_max, epsilon, eta)
            kappa_1_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1.txt'.format(
                base_filename, g_max, epsilon, eta)
            kappa_2_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2.txt'.format(
                base_filename, g_max, epsilon, eta)
            kappa_1_errors_file = ('{}.VV_g_max{}_epsilon{}_eta{}.kappa_1_'
                                   'errors.txt'.format(
                                    base_filename, g_max, epsilon, eta))
            kappa_2_errors_file = ('{}.VV_g_max{}_epsilon{}_eta{}.kappa_2_'
                                   'errors.txt'.format(
                                    base_filename, g_max, epsilon, eta))
        elif k > 0:
            surf_VV_file = '{}.VV_k{}_epsilon{}_eta{}.vtp'
            kappa_1_file = '{}.VV_k{}_epsilon{}_eta{}.kappa_1.txt'.format(
                base_filename, k, epsilon, eta)
            kappa_2_file = '{}.VV_k{}_epsilon{}_eta{}.kappa_2.txt'.format(
                base_filename, k, epsilon, eta)
            kappa_1_errors_file = ('{}.VV_k{}_epsilon{}_eta{}.kappa_1_errors'
                                   '.txt'.format(
                                    base_filename, k, epsilon, eta))
            kappa_2_errors_file = ('{}.VV_k{}_epsilon{}_eta{}.kappa_2_errors'
                                   '.txt'.format(
                                    base_filename, k, epsilon, eta))
        else:
            error_msg = ("Either g_max or k must be positive (if both are "
                         "positive, the specified g_max will be used).")
            raise pexceptions.PySegInputError(expr='vector_voting',
                                              msg=error_msg)
        vtk_max_curvature_file = ('{}.VTK.max_curvature.txt'
                                  .format(base_filename))
        vtk_min_curvature_file = ('{}.VTK.min_curvature.txt'
                                  .format(base_filename))
        vtk_max_curvature_errors_file = ('{}.VTK.max_curvature_errors.txt'
                                         .format(base_filename))
        vtk_min_curvature_errors_file = ('{}.VTK.min_curvature_errors.txt'
                                         .format(base_filename))

        # If the .vtp file with the test surface does not exist, create it:
        if not os.path.isfile(surf_file):
            sg = SphereGenerator()
            sphere = sg.generate_sphere_surface(radius=radius, latitude_res=res,
                                                longitude_res=res)
            io.save_vtp(sphere, surf_file)

        # Reading in the .vtp file with the test triangle mesh and transforming
        # it into a triangle graph:
        t_begin = time.time()

        print '\nReading in the surface file to get a vtkPolyData surface...'
        surf = io.load_poly(surf_file)
        print ('\nBuilding the TriangleGraph from the vtkPolyData surface with '
               'curvatures...')
        tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
        # a graph with normals pointing outwards is generated (normal case for
        # current sphere generation; negative curvatures)
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
        vtk_max_curvature_values = tg.get_vertex_property_array("max_curvature")
        vtk_min_curvature_values = tg.get_vertex_property_array("min_curvature")

        # Calculating the errors of the principal curvatures in percents:
        correct_curvature = 1.0 / radius
        if inverse:
            correct_curvature *= -1
        kappa_1_errors = np.array(map(
            lambda x: abs(x - correct_curvature) /
            correct_curvature * 100.0, kappa_1_values))
        kappa_2_errors = np.array(map(
            lambda x: abs(x - correct_curvature) /
            correct_curvature * 100.0, kappa_2_values))
        vtk_max_curvature_errors = np.array(map(
            lambda x: abs(x - correct_curvature) /
            correct_curvature * 100.0, vtk_max_curvature_values))
        vtk_min_curvature_errors = np.array(map(
            lambda x: abs(x - correct_curvature) /
            correct_curvature * 100.0, vtk_min_curvature_values))

        # Writing all the curvature values and errors into files:
        io.write_values_to_file(kappa_1_values, kappa_1_file)
        io.write_values_to_file(kappa_1_errors, kappa_1_errors_file)
        io.write_values_to_file(kappa_2_values, kappa_2_file)
        io.write_values_to_file(kappa_2_errors, kappa_2_errors_file)
        io.write_values_to_file(vtk_max_curvature_values,
                                vtk_max_curvature_file)
        io.write_values_to_file(vtk_max_curvature_errors,
                                vtk_max_curvature_errors_file)
        io.write_values_to_file(vtk_min_curvature_values,
                                vtk_min_curvature_file)
        io.write_values_to_file(vtk_min_curvature_errors,
                                vtk_min_curvature_errors_file)

        # Asserting that all values of both principal curvatures are close to
        # the correct value, 30% of difference is allowed:
        allowed_error = 0.3 * abs(correct_curvature)
        print "Testing the maximal principal curvatures (kappa_1)..."
        for kappa_1 in kappa_1_values:
            msg = '{} is not in [{}, {}]!'.format(
                kappa_1, correct_curvature - allowed_error,
                correct_curvature + allowed_error)
            self.assertAlmostEqual(kappa_1, correct_curvature,
                                   delta=allowed_error, msg=msg)
        print "Testing the minimal principal curvatures (kappa_2)..."
        for kappa_2 in kappa_2_values:
            msg = '{} is not in [{}, {}]!'.format(
                kappa_2, correct_curvature - allowed_error,
                correct_curvature + allowed_error)
            self.assertAlmostEqual(kappa_2, correct_curvature,
                                   delta=allowed_error, msg=msg)

    def test_sphere_radius5_res50_g_max1_curvatures(self):
        """
        Tests whether curvatures for a sphere with radius 5 and resolution 50
        are correctly estimated using NVV with g_max = 1:

        kappa1 = kappa2 = 1/5 = 0.2; 30% of difference is allowed
        """
        self.parametric_test_sphere_curvatures(radius=5, res=50,
                                               g_max=1, epsilon=0, eta=0)

    def test_inverse_sphere_radius5_res50_g_max1_curvatures(self):
        """
        Tests whether curvatures for an inverse sphere with radius 5 and
        resolution 50 are correctly estimated using NVV with g_max = 1:

        kappa1 = kappa2 = -1/5 = 0.2; 30% of difference is allowed
        """
        self.parametric_test_sphere_curvatures(radius=5, res=50, inverse=True,
                                               g_max=1, epsilon=0, eta=0)

if __name__ == '__main__':
    unittest.main()
