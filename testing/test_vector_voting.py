import unittest
import time
import os.path
import numpy as np
import math
import pandas as pd

from pysurf_compact import pysurf_io as io
from pysurf_compact import (
    TriangleGraph, vector_voting, vector_voting_curve_fitting,
    vector_curvature_tensor_voting)
from synthetic_surfaces import (
    PlaneGenerator, SphereGenerator, CylinderGenerator, SaddleGenerator,
    add_gaussian_noise_to_surface)


def absolute_error_scalar(true_value, estimated_value):
    """
    Calculates the "absolute error" as measure of accuracy for scalars.

    Args:
        true_value: true / accepted scalar value
        estimated_value: estimated / measured / experimental scalar value

    Returns:
        abs(true_value - estimated_value)
        the lower the error, the more accurate the estimated value
    """
    return abs(true_value - estimated_value)


def relative_error_scalar(true_value, estimated_value):
    """
    Calculates the "relative error" as measure of accuracy for scalars.

    Args:
        true_value: true / accepted scalar value
        estimated_value: estimated / measured / experimental scalar value

    Returns:
        abs((true_value - estimated_value) / true_value)
        if true_value = 0, just abs(true_value - estimated_value)
        the lower the error, the more accurate the estimated value
    """
    if true_value == 0:
        return abs(true_value - estimated_value)
    else:
        return abs((true_value - estimated_value) / true_value)


def error_vector(true_vector, estimated_vector):
    """
    Calculates the error for 3D vectors.

    Args:
        true_vector (numpy.ndarray): true / accepted 3D vector
        estimated_vector (numpy.ndarray): estimated / measured / experimental 3D
            vector

    Returns:
        1 - abs(np.dot(true_vector, estimated_vector))
        0 if the vectors are parallel, 1 if they are perpendicular
    """
    return 1 - abs(np.dot(true_vector, estimated_vector))


def angular_error_vector(true_vector, estimated_vector):
    """
    Calculates the "angular error" for 3D vectors.

    Args:
        true_vector (numpy.ndarray): true / accepted 3D vector
        estimated_vector (numpy.ndarray): estimated / measured / experimental 3D
            vector

    Returns:
        acos(abs(np.dot(true_vector, estimated_vector)))
        angle in radians between two vectors
    """
    return math.acos(abs(np.dot(true_vector, estimated_vector)))


def beautify_number(number, precision=15):
    """
    Rounds an almost zero floating point number to 0 and removes minus.
    Args:
        number (float): input number
        precision (int): desired number of decimal points to use for rounding

    Returns:
        0 if absolute value of the rounded number is 0, else the original number
    """
    if abs(round(number, precision)) == 0:
        return 0
    else:
        return number


def torus_curvatures_and_directions(c, a, x, y, z, verbose=False):
    """
    Calculated true minimal principal curvature (kappa_2), maximal (T_1) and
    minimal (T_2) principal directions at a point on the surface of a torus.

    Args:
        c (int): major (ring) radius
        a (int): minor (cross-section) radius
        x (float): X coordinate of the point on the torus surface
        y (float): Y coordinate of the point on the torus surface
        z (float): Z coordinate of the point on the torus surface
        verbose: (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        kappa_2, T_1, T_2

    Note:
        Maximal principal curvature (kappa_1) is always 1 / a.
    """
    sin = math.sin
    cos = math.cos
    asin = math.asin
    acos = math.acos
    pi = math.pi

    # Find v angle around the small circle, origin at outer torus side,
    # rotation towards positive z
    v = asin(z / a)
    r_xy = math.sqrt(x ** 2 + y ** 2)  # same as c + a * cos(v)
    if r_xy <= c:
        v = pi - v
    if v < 0:
        v += 2 * pi

    # Find u angle around the big circle, origin at positive x and y=0,
    # rotation towards positive y
    cos_u = x / r_xy
    u = acos(cos_u)
    if y < 0:
        u = 2 * pi - u
    # sin_u = y / r_xy  # alternative calculation
    # u_too = asin(sin_u)
    # if x < 0:
    #     u_too = pi - u_too

    # minimal principal curvatures
    if v == pi / 2 or v == 3 * pi / 2:  # top or bottom of torus in Z
        kappa_2 = 0
    else:
        kappa_2 = cos(v) / r_xy

    # # normal to the surface
    # N = [cos(u) * cos(v), sin(u) * cos(v), sin(v)]
    # maximal and minimal principal directions
    T_1 = [- cos(u) * sin(v), - sin(u) * sin(v), cos(v)]
    # assert(round(np.linalg.norm(T_1), 5) == 1.0)
    T_2 = [- sin(u), cos(u), 0]
    # assert(round(np.linalg.norm(T_2), 5) == 1.0)
    # round almost 0 to 0 and remove minus before 0
    # N = [beautify_number(e) for e in N]
    T_1 = [beautify_number(e) for e in T_1]
    T_2 = [beautify_number(e) for e in T_2]

    if verbose:
        print("v = {}".format(v))
        print("u = {}".format(u))
        print("kappa_2 = {}".format(kappa_2))
        # print("N = ({}, {}, {})".format(N[0], N[1], N[2]))
        print("T_1 = ({}, {}, {})".format(T_1[0], T_1[1], T_1[2]))
        print("T_2 = ({}, {}, {})".format(T_2[0], T_2[1], T_2[2]))

    return kappa_2, T_1, T_2


class VectorVotingTestCase(unittest.TestCase):
    """
    Tests for vector_voting.py, assuming that other used functions are correct.
    """

    def parametric_test_plane_normals(self, half_size, radius_hit, res=30,
                                      noise=10):
        """
        Tests whether normals are correctly estimated for a plane surface with
        known orientation (parallel to to X and Y axes).

        Args:
            half_size (int): half size of the plane (from center to an edge)
            radius_hit (float): radius in length unit of the graph, here voxels;
                it should be chosen to correspond to radius of smallest features
                of interest on the surface
            res (int, optional): resolution (number of divisions) in X and Y
                axes (default 30)
            noise (int, optional): determines variance of the Gaussian noise in
                percents of average triangle edge length (default 10), the noise
                is added on triangle vertex coordinates in its normal direction

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
        scale_factor_to_nm = 1.0  # assume it's already in nm
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
        surf_vv_file = '{}.VCTV_rh{}.vtp'.format(base_filename, radius_hit)
        vv_eval_file = '{}.VCTV_rh{}.csv'.format(base_filename, radius_hit)
        vtk_eval_file = '{}.VTK.csv'.format(base_filename)

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

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm (with curvature
        # tensor voting, because it is be the fastest):
        surf_vv = vector_curvature_tensor_voting(
            tg, radius_hit=radius_hit, exclude_borders=True)
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

        # Computing errors of the initial (VTK) and estimated (VV) normals wrt
        # the true normal:
        vtk_normal_errors = np.array(map(
            lambda x: error_vector(true_normal, x), vtk_normals))
        vtk_normal_angular_errors = np.array(map(
            lambda x: angular_error_vector(true_normal, x), vtk_normals))
        vv_normal_errors = np.array(map(
            lambda x: error_vector(true_normal, x), vv_normals))
        vv_normal_angular_errors = np.array(map(
            lambda x: angular_error_vector(true_normal, x), vv_normals))

        # Writing the errors into csv files:
        df = pd.DataFrame()
        df['normalErrors'] = vtk_normal_errors
        df['normalAngularErrors'] = vtk_normal_angular_errors
        df.to_csv(vtk_eval_file, sep=';')

        df = pd.DataFrame()
        df['normalErrors'] = vv_normal_errors
        df['normalAngularErrors'] = vv_normal_angular_errors
        df.to_csv(vv_eval_file, sep=';')

        # Asserting that all estimated normals are close to the true normal,
        # allowing error of 30%:
        for error in vv_normal_errors:
            msg = '{} is > {}!'.format(error, 0.3)
            self.assertLessEqual(error, 0.3, msg=msg)

    def parametric_test_cylinder_directions_curvatures(
            self, r, radius_hit, inverse=False, res=0, h=0, noise=10,
            method='VCTV', page_curvature_formula=False, num_points=None):
        """
        Tests whether minimal principal directions (T_2), as well as minimal and
        maximal principal curvatures are correctly estimated for an opened
        cylinder surface (without the circular planes) with known
        orientation (height, i.e. T_2, parallel to the Z axis) using normal
        vector voting (VV), VV combined with curve fitting (VVCF) or with
        curvature tensor voting (VCTV).

        Args:
            r (int): cylinder radius in voxels
            radius_hit (float): radius in length unit of the graph, here voxels;
                it should be chosen to correspond to radius of smallest features
                of interest on the surface
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
            method (str): tells which method should be used: 'VV' for normal
                vector voting, 'VVCF' for curve fitting in the two principal
                directions estimated by VV to estimate the principal curvatures
                or 'VCTV' (default) for vector and curvature tensor voting to
                estimate the principal direction and curvatures
            page_curvature_formula (boolean, optional): if True (default False)
                normal curvature formula from Page at al. is used for VV or VVCF
                (see collecting_curvature_votes)
            num_points (int): for VVCF, number of points to sample in each
                estimated principal direction in order to fit parabola and
                estimate curvature

        Returns:
            None
        """
        if method != 'VV' and method != 'VVCF' and method != 'VCTV':
            print("The parameter 'method' has to be 'VV', 'VVCF' or 'VCTV'")
            exit(0)
        if method == 'VV' and page_curvature_formula:
            method = 'VV_page_curvature_formula'
        elif method == 'VVCF' and page_curvature_formula:
            method = 'VVCF_page_curvature_formula_{}points'.format(num_points)
        elif method == 'VVCF':
            method = 'VVCF_{}points'.format(num_points)
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
        scale_factor_to_nm = 1.0  # assume it's already in nm
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
        surf_VV_file = '{}.{}_rh{}.vtp'.format(
            base_filename, method, radius_hit)
        VV_eval_file = '{}.{}_rh{}.csv'.format(
            base_filename, method, radius_hit)
        VTK_eval_file = '{}.VTK.csv'.format(base_filename)

        if inverse:
            print ("\n*** Generating a surface and a graph for an inverse "
                   "cylinder with radius {}, height {} and {}% noise using the "
                   "method {}***".format(r, h, noise, method))
        else:
            print ("\n*** Generating a surface and a graph for a cylinder with "
                   "radius {}, height {} and {}% noise using the method {}***"
                   .format(r, h, noise, method))
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

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm:
        if method == 'VV' or method == 'VV_page_curvature_formula':
            script = vector_voting
        elif 'VVCF' in method:
            script = vector_voting_curve_fitting
        else:  # if method == 'VCTV'
            script = vector_curvature_tensor_voting
        kwargs = {'radius_hit': radius_hit, 'exclude_borders': True}
        if 'page_curvature_formula' in method:
            kwargs['page_curvature_formula'] = True
        if 'CF' in method:
            kwargs['num_points'] = num_points
        surf_VV = script(tg, **kwargs)

        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        io.save_vtp(surf_VV, surf_VV_file)

        # Getting the estimated principal directions along cylinder height, T_h:
        pos = [0, 1, 2]  # vector-property value positions
        if not inverse:  # it's the minimal direction
            T_h = tg.graph.vertex_properties["T_2"].get_2d_array(pos)
        else:  # it's the maximal direction
            T_h = tg.graph.vertex_properties["T_1"].get_2d_array(pos)
        # The shape is (3, <num_vertices>) - have to transpose to group the
        # respective x, y, z components to sub-arrays
        T_h = np.transpose(T_h)  # shape (<num_vertices>, 3)

        # Ground-truth T_h vector is parallel to Z axis
        true_T_h = np.array([0, 0, 1])

        # Computing errors of the estimated T_h vectors wrt the true one:
        T_h_errors = np.array(map(
            lambda x: error_vector(true_T_h, x), T_h))
        T_h_angular_errors = np.array(map(
            lambda x: angular_error_vector(true_T_h, x), T_h))

        # Getting principal curvatures from VV and VTK from the output graph:
        kappa_1 = tg.get_vertex_property_array("kappa_1")
        kappa_2 = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1 = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2 = tg.get_vertex_property_array("min_curvature")
        # Calculating estimated average principal curvatures:
        kappa_1_avg = np.mean(kappa_1)
        kappa_2_avg = np.mean(kappa_2)

        # Ground-truth principal curvatures
        if inverse:
            true_kappa_1 = 0.0
            true_kappa_2 = - 1.0 / r
        else:
            true_kappa_1 = 1.0 / r
            true_kappa_2 = 0.0

        # Calculating errors of the principal curvatures:
        if true_kappa_1 != 0:  # not inverse
            abs_kappa_1_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_1, x),
                kappa_1))
            rel_kappa_1_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_1, x),
                kappa_1))
            vtk_abs_kappa_1_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_1, x),
                vtk_kappa_1))
            vtk_rel_kappa_1_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_1, x),
                vtk_kappa_1))
        else:  # inverse
            abs_kappa_2_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_2, x),
                kappa_2))
            rel_kappa_2_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_2, x),
                kappa_2))
            vtk_abs_kappa_2_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_2, x),
                vtk_kappa_2))
            vtk_rel_kappa_2_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_2, x),
                vtk_kappa_2))

        # Writing all the VV curvature values and errors into a csv file:
        df = pd.DataFrame()
        df['kappa1'] = kappa_1
        df['kappa2'] = kappa_2
        if true_kappa_1 != 0:  # not inverse
            df['kappa1AbsErrors'] = abs_kappa_1_errors
            df['kappa1RelErrors'] = rel_kappa_1_errors
            df['T2Errors'] = T_h_errors
            df['T2AngularErrors'] = T_h_angular_errors
        else:  # inverse
            df['kappa2AbsErrors'] = abs_kappa_2_errors
            df['kappa2RelErrors'] = rel_kappa_2_errors
            df['T1Errors'] = T_h_errors
            df['T1AngularErrors'] = T_h_angular_errors
        df.to_csv(VV_eval_file, sep=';')
        # The same for VTK, if the file does not exist yet:
        if not os.path.isfile(VTK_eval_file):
            df = pd.DataFrame()
            df['kappa1'] = vtk_kappa_1
            df['kappa2'] = vtk_kappa_2
            if true_kappa_1 != 0:  # not inverse
                df['kappa1AbsErrors'] = vtk_abs_kappa_1_errors
                df['kappa1RelErrors'] = vtk_rel_kappa_1_errors
            else:  # inverse
                df['kappa2AbsErrors'] = vtk_abs_kappa_2_errors
                df['kappa2RelErrors'] = vtk_rel_kappa_2_errors
            df.to_csv(VTK_eval_file, sep=';')

        # Asserting that all estimated T_h vectors are close to the true vector,
        # allowing error of 30%:
        if not inverse:
            print "Testing the minimal principal directions (T_2)..."
        else:
            print "Testing the maximal principal directions (T_1)..."
        for error in T_h_errors:
            msg = '{} is > {}!'.format(error, 0.3)
            self.assertLessEqual(error, 0.3, msg=msg)

        # Asserting that average principal curvatures are close to the correct
        # ones allowing error of +-30% of the maximal absolute true value
        allowed_error = 0.3 * max(abs(true_kappa_1), abs(true_kappa_2))
        # if true_kappa_1 != 0:  # not inverse
        #     print "Testing the average maximal principal curvature (kappa_1)..."
        #     msg = '{} is not in [{}, {}]!'.format(
        #         kappa_1_avg, true_kappa_1 - allowed_error,
        #         true_kappa_1 + allowed_error)
        #     self.assertAlmostEqual(kappa_1_avg, true_kappa_1,
        #                            delta=allowed_error, msg=msg)
        # else:  # inverse
        #     print "Testing the average minimal principal curvature (kappa_2)..."
        #     msg = '{} is not in [{}, {}]!'.format(
        #         kappa_2_avg, true_kappa_2 - allowed_error,
        #         true_kappa_2 + allowed_error)
        #     self.assertAlmostEqual(kappa_2_avg, true_kappa_2,
        #                            delta=allowed_error, msg=msg)

        # Asserting that all principal curvatures are close to the correct
        # ones allowing error of +-30% of the maximal absolute true value
        if true_kappa_1 != 0:  # not inverse
            print "Testing the maximal principal curvature (kappa_1)..."
            for i, error in enumerate(abs_kappa_1_errors):
                msg = 'triangle {}: {} is > {}%!'.format(i, error, 30)
                self.assertLessEqual(error, allowed_error, msg=msg)
        else:  # inverse
            print "Testing the minimal principal curvature (kappa_2)..."
            for i, error in enumerate(abs_kappa_2_errors):
                msg = 'triangle {}: {} is > {}%!'.format(i, error, 30)
                self.assertLessEqual(error, allowed_error, msg=msg)

    def parametric_test_sphere_curvatures(
            self, radius, radius_hit, inverse=False, binary=False, res=0,
            ico=0, noise=0, save_areas=False, method='VCTV',
            page_curvature_formula=False, num_points=None):
        """
        Runs all the steps needed to calculate curvatures for a test sphere
        with a given radius. Tests whether the curvatures are correctly
        estimated using normal vector voting (VV), VV combined with curve
        fitting (VVCF) or with curvature tensor voting (VCTV):

        kappa_1 = kappa_2 = 1/r; allowing some error.

        Args:
            radius (int): radius of the sphere
            radius_hit (float): radius in length unit of the graph, here voxels;
                it should be chosen to correspond to radius of smallest features
                of interest on the surface
            inverse (boolean, optional): if True (default False), the sphere
                will have normals pointing outwards (negative curvature), else
                the other way around
            binary (boolean, optional): if True (default False), a binary sphere
                is generated (ignoring the next three options)
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
            save_areas (boolean, optional): if True (default False), also mesh
                triangle ares will be saved to a file
            method (str): tells which method should be used: 'VV' for normal
                vector voting, 'VVCF' for curve fitting in the two principal
                directions estimated by VV to estimate the principal curvatures
                or 'VCTV' (default) for vector and curvature tensor voting to
                estimate the principal direction and curvatures
            page_curvature_formula (boolean, optional): if True (default False)
                normal curvature formula from Page et al. is used for VV or VVCF
                (see collecting_curvature_votes)
            num_points (int): for VVCF, number of points to sample in each
                estimated principal direction in order to fit parabola and
                estimate curvature

        Returns:
            None
        """
        if method != 'VV' and method != 'VVCF' and method != 'VCTV':
            print("The parameter 'method' has to be 'VV', 'VVCF' or 'VCTV'")
            exit(0)
        if method == 'VV' and page_curvature_formula:
            method = 'VV_page_curvature_formula'
        elif method == 'VVCF' and page_curvature_formula:
            method = 'VVCF_page_curvature_formula_{}points'.format(num_points)
        elif method == 'VVCF':
            method = 'VVCF_{}points'.format(num_points)
        base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
        if binary:
            fold = '{}synthetic_surfaces/sphere/binary/'.format(base_fold)
        else:
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
        scale_factor_to_nm = 1.0  # assume it's already in nm
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
        base_filename = "{}{}sphere_r{}".format(files_fold, inverse_str, radius)
        surf_VV_file = '{}.{}_rh{}.vtp'.format(
            base_filename, method, radius_hit)
        VV_eval_file = '{}.{}_rh{}.csv'.format(
            base_filename, method, radius_hit)
        VTK_eval_file = '{}.VTK.csv'.format(base_filename)

        if inverse:
            print ("\n*** Generating a surface and a graph for an inverse "
                   "sphere with radius {} and {}% noise using the method {}***"
                   .format(radius, noise, method))
        else:
            print ("\n*** Generating a surface and a graph for a sphere "
                   "with radius {} and {}% noise using the method {}***".format(
                    radius, noise, method))
        # If the .vtp file with the test surface does not exist, create it:
        if not os.path.isfile(surf_file):
            sg = SphereGenerator()
            if binary:
                sphere = sg.generate_sphere_surface(radius)
                io.save_vtp(sphere, surf_file)
            else:
                if res > 0:  # generate a UV sphere surface directly with VTK
                    sphere = sg.generate_UV_sphere_surface(
                        r=radius, latitude_res=res, longitude_res=res)
                    if noise > 0:
                        sphere = add_gaussian_noise_to_surface(sphere,
                                                               percent=noise)
                    io.save_vtp(sphere, surf_file)
                elif ico > 0:
                    print ("Sorry, you have to generate the icosahedron\n"
                           "sphere beforehand e.g. with Blender, export it as\n"
                           "STL file and convert it to VTP file using the\n"
                           "function pysurf_io.stl_file_to_vtp_file,\n"
                           "optionally add noise with\n"
                           "add_gaussian_noise_to_surface and save it as\n{}"
                           .format(surf_file))
                    exit(0)
                else:  # generate a sphere surface from a gaussian mask
                    sphere = sg.generate_gauss_sphere_surface(radius)
                    if noise > 0:
                        sphere = add_gaussian_noise_to_surface(sphere,
                                                               percent=noise)
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

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Running the modified Normal Vector Voting algorithm:
        if method == 'VV' or method == 'VV_page_curvature_formula':
            script = vector_voting
        elif 'VVCF' in method:
            script = vector_voting_curve_fitting
        else:  # if method == 'VCTV'
            script = vector_curvature_tensor_voting
        kwargs = {'radius_hit': radius_hit, 'exclude_borders': False}
        if 'page_curvature_formula' in method:
            kwargs['page_curvature_formula'] = True
        if 'CF' in method:
            kwargs['num_points'] = num_points
        surf_VV = script(tg, **kwargs)

        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        io.save_vtp(surf_VV, surf_VV_file)

        # Ground truth principal curvatures
        true_curvature = 1.0 / radius
        if inverse:
            true_curvature *= -1

        # Getting estimated principal curvatures from the output graph:
        kappa_1 = tg.get_vertex_property_array("kappa_1")
        kappa_2 = tg.get_vertex_property_array("kappa_2")
        # Calculating average principal curvatures
        kappa_1_avg = np.mean(kappa_1)
        kappa_2_avg = np.mean(kappa_2)

        # Calculating errors of the principal curvatures:
        abs_kappa_1_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), kappa_1))
        abs_kappa_2_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), kappa_2))
        rel_kappa_1_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), kappa_1))
        rel_kappa_2_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), kappa_2))

        # Writing all the curvature values and errors into a csv file:
        df = pd.DataFrame()
        df['kappa1'] = kappa_1
        df['kappa1AbsErrors'] = abs_kappa_1_errors
        df['kappa1RelErrors'] = rel_kappa_1_errors
        df['kappa2'] = kappa_2
        df['kappa2AbsErrors'] = abs_kappa_2_errors
        df['kappa2RelErrors'] = rel_kappa_2_errors
        if save_areas:
            triangle_areas = tg.get_vertex_property_array("area")
            df['triangleAreas'] = triangle_areas
        df.to_csv(VV_eval_file, sep=';')

        # The same steps for VTK, if the file does not exist yet:
        if not os.path.isfile(VTK_eval_file):
            vtk_kappa_1_values = tg.get_vertex_property_array("max_curvature")
            vtk_kappa_2_values = tg.get_vertex_property_array("min_curvature")
            vtk_abs_kappa_1_errors = np.array(map(
                lambda x: absolute_error_scalar(true_curvature, x),
                vtk_kappa_1_values))
            vtk_abs_kappa_2_errors = np.array(map(
                lambda x: absolute_error_scalar(true_curvature, x),
                vtk_kappa_2_values))
            vtk_rel_kappa_1_errors = np.array(map(
                lambda x: relative_error_scalar(true_curvature, x),
                vtk_kappa_1_values))
            vtk_rel_kappa_2_errors = np.array(map(
                lambda x: relative_error_scalar(true_curvature, x),
                vtk_kappa_2_values))
            df = pd.DataFrame()
            df['kappa1'] = vtk_kappa_1_values
            df['kappa1AbsErrors'] = vtk_abs_kappa_1_errors
            df['kappa1RelErrors'] = vtk_rel_kappa_1_errors
            df['kappa2'] = vtk_kappa_2_values
            df['kappa2AbsErrors'] = vtk_abs_kappa_2_errors
            df['kappa2RelErrors'] = vtk_rel_kappa_2_errors
            df.to_csv(VTK_eval_file, sep=';')

        # Asserting that average principal curvatures are close to the correct
        # ones allowing absolute error of +-30% of the true curvature
        allowed_error = 0.3 * abs(true_curvature)
        # print "Testing the average maximal principal curvature (kappa_1)..."
        # msg = '{} is not in [{}, {}]!'.format(
        #     kappa_1_avg, true_curvature - allowed_error,
        #     true_curvature + allowed_error)
        # self.assertAlmostEqual(kappa_1_avg, true_curvature,
        #                        delta=allowed_error, msg=msg)
        # print "Testing the average minimal principal curvature (kappa_2)..."
        # msg = '{} is not in [{}, {}]!'.format(
        #     kappa_2_avg, true_curvature - allowed_error,
        #     true_curvature + allowed_error)
        # self.assertAlmostEqual(kappa_2_avg, true_curvature,
        #                        delta=allowed_error, msg=msg)

        # Asserting that all values of both principal curvatures are close to
        # the true value, allowing error of +-30%:
        # print "Testing the maximal principal curvature (kappa_1)..."
        # for i, error in enumerate(abs_kappa_1_errors):
        #     msg = 'triangle {}: {} is > {}%!'.format(i, error, 30)
        #     self.assertLessEqual(error, allowed_error, msg=msg)
        # print "Testing the minimal principal curvature (kappa_2)..."
        # for i, error in enumerate(abs_kappa_2_errors):
        #     msg = 'triangle {}: {} is > {}%!'.format(i, error, 30)
        #     self.assertLessEqual(error, allowed_error, msg=msg)

    def parametric_test_torus_curvatures(
            self, rr, csr, radius_hit, method='VCTV',
            page_curvature_formula=False, num_points=None):
        """
        Runs all the steps needed to calculate curvatures for a test torus
        with given radii using normal vector voting (VV), VV combined with curve
        fitting (VVCF) or with curvature tensor voting (VCTV).

        Args:
            rr (int): ring radius of the torus
            csr (int): cross-section radius of the torus
            radius_hit (float): radius in length unit of the graph, here voxels;
                it should be chosen to correspond to radius of smallest features
                of interest on the surface
            method (str): tells which method should be used: 'VV' for normal
                vector voting, 'VVCF' for curve fitting in the two principal
                directions estimated by VV to estimate the principal curvatures
                or 'VCTV' (default) for vector and curvature tensor voting to
                estimate the principal direction and curvatures
            page_curvature_formula (boolean, optional): if True (default False)
                normal curvature formula from Page et al. is used for VV or VVCF
                (see collecting_curvature_votes)
            num_points (int): for VVCF, number of points to sample in each
                estimated principal direction in order to fit parabola and
                estimate curvature

        Notes:
            * csr should be much smaller than rr (csr < rr - csr).

        Returns:
            None
        """
        if method != 'VV' and method != 'VVCF' and method != 'VCTV':
            print("The parameter 'method' has to be 'VV', 'VVCF' or 'VCTV'")
            exit(0)
        if method == 'VV' and page_curvature_formula:
            method = 'VV_page_curvature_formula'
        elif method == 'VVCF' and page_curvature_formula:
            method = 'VVCF_page_curvature_formula_{}points'.format(num_points)
        elif method == 'VVCF':
            method = 'VVCF_{}points'.format(num_points)
        fold = '/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/torus/'

        if not os.path.exists(fold):
            os.makedirs(fold)
        # TODO add "_inner_part" for a small saddle part:
        surf_filebase = '{}torus_rr{}_csr{}'.format(fold, rr, csr)
        surf_file = '{}.surface.vtp'.format(surf_filebase)
        scale_factor_to_nm = 1.0  # assume it's already in nm
        # Actually can just give in any number for the scales, because they are
        # only used for ribosome density calculation or volumes / .mrc files
        # creation.
        scale_x = 2 * (rr + csr)
        scale_y = scale_x
        scale_z = 2 * csr
        files_fold = '{}files4plotting/'.format(fold)
        if not os.path.exists(files_fold):
            os.makedirs(files_fold)
        # TODO add "_inner_part" for a small saddle part:
        base_filename = "{}torus_rr{}_csr{}".format(files_fold, rr, csr)
        surf_VV_file = '{}.{}_rh{}.vtp'.format(
            base_filename, method, radius_hit)
        VV_eval_file = '{}.{}_rh{}.csv'.format(
            base_filename, method, radius_hit)
        VTK_eval_file = '{}.VTK.csv'.format(base_filename)
        print ("\n*** Generating a surface and a graph for a torus "
               "with ring radius {} and cross-section radius {} "
               "using the method {}***"
               .format(rr, csr, method))
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
        # VTK has opposite surface normals convention than we use,
        # a graph with normals pointing inwards is generated (VTK normals have
        # to be flipped)
        tg.build_graph_from_vtk_surface(surf, verbose=False,
                                        reverse_normals=True)
        print tg.graph

        t_end = time.time()
        duration = t_end - t_begin
        print ('Graph construction from surface took: {} min {} s'.format(
            divmod(duration, 60)[0], divmod(duration, 60)[1]))

        # Ground-truth principal curvatures and directions
        # Vertex properties for storing the true maximal and minimal curvatures
        # and the their directions of the corresponding triangle:
        tg.graph.vp.true_kappa_1 = tg.graph.new_vertex_property("float")
        tg.graph.vp.true_kappa_2 = tg.graph.new_vertex_property("float")
        tg.graph.vp.true_T_1 = tg.graph.new_vertex_property("vector<float>")
        tg.graph.vp.true_T_2 = tg.graph.new_vertex_property("vector<float>")

        # Calculate and fill the properties
        true_kappa_1 = 1.0 / csr  # constant for the whole torus surface
        xyz = tg.graph.vp.xyz
        for v in tg.graph.vertices():
            x, y, z = xyz[v]  # coordinates of triangle center v
            true_kappa_2, true_T_1, true_T_2 = torus_curvatures_and_directions(
                rr, csr, x, y, z)
            tg.graph.vp.true_kappa_1[v] = true_kappa_1
            tg.graph.vp.true_kappa_2[v] = true_kappa_2
            tg.graph.vp.true_T_1[v] = true_T_1
            tg.graph.vp.true_T_2[v] = true_T_2

        # Running the modified Normal Vector Voting algorithm
        if method == 'VV' or method == 'VV_page_curvature_formula':
            script = vector_voting
        elif 'VVCF' in method:
            script = vector_voting_curve_fitting
        else:  # if method == 'VCTV'
            script = vector_curvature_tensor_voting
        kwargs = {'radius_hit': radius_hit, 'exclude_borders': False}
        if 'page_curvature_formula' in method:
            kwargs['page_curvature_formula'] = True
        if 'CF' in method:
            kwargs['num_points'] = num_points
        surf_VV = script(tg, **kwargs)

        # Saving the output (TriangleGraph object) for later inspection:
        io.save_vtp(surf_VV, surf_VV_file)

        # Getting the estimated and true principal directions:
        pos = [0, 1, 2]  # vector-property value positions
        # The shape is (3, <num_vertices>) - have to transpose to group the
        # respective x, y, z components to sub-arrays
        T_1 = np.transpose(tg.graph.vertex_properties["T_1"].get_2d_array(pos))
        T_2 = np.transpose(tg.graph.vertex_properties["T_2"].get_2d_array(pos))
        true_T_1 = np.transpose(
            tg.graph.vertex_properties["true_T_1"].get_2d_array(pos))
        true_T_2 = np.transpose(
            tg.graph.vertex_properties["true_T_2"].get_2d_array(pos))

        # Computing errors of the estimated directions wrt the true ones:
        T_1_errors = np.array(map(
            lambda x, y: error_vector(x, y), true_T_1, T_1))
        T_1_angular_errors = np.array(map(
            lambda x, y: angular_error_vector(x, y), true_T_1, T_1))
        T_2_errors = np.array(map(
            lambda x, y: error_vector(x, y), true_T_2, T_2))
        T_2_angular_errors = np.array(map(
            lambda x, y: angular_error_vector(x, y), true_T_2, T_2))

        # Getting the estimated and true principal curvatures:
        kappa_1 = tg.get_vertex_property_array("kappa_1")
        kappa_2 = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1 = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2 = tg.get_vertex_property_array("min_curvature")
        true_kappa_1 = tg.get_vertex_property_array("true_kappa_1")
        true_kappa_2 = tg.get_vertex_property_array("true_kappa_2")

        # Computing errors of the estimated curvatures wrt the true ones:
        abs_kappa_1_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y), true_kappa_1, kappa_1))
        # abs_kappa_1_errors = []  # the same as map
        # for x, y in zip(true_kappa_1, kappa_1):
        #     abs_kappa_1_error = absolute_error_scalar(x, y)
        #     abs_kappa_1_errors.append(abs_kappa_1_error)
        # abs_kappa_1_errors = np.array(abs_kappa_1_errors)
        rel_kappa_1_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y), true_kappa_1, kappa_1))
        vtk_abs_kappa_1_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y),
            true_kappa_1, vtk_kappa_1))
        vtk_rel_kappa_1_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y),
            true_kappa_1, vtk_kappa_1))
        abs_kappa_2_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y), true_kappa_2, kappa_2))
        rel_kappa_2_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y), true_kappa_2, kappa_2))
        vtk_abs_kappa_2_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y),
            true_kappa_2, vtk_kappa_2))
        vtk_rel_kappa_2_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y),
            true_kappa_2, vtk_kappa_2))

        # Writing all the VV curvature values and errors into a csv file:
        df = pd.DataFrame()
        df['kappa1'] = kappa_1
        df['kappa2'] = kappa_2
        df['kappa1AbsErrors'] = abs_kappa_1_errors
        df['kappa1RelErrors'] = rel_kappa_1_errors
        df['kappa2AbsErrors'] = abs_kappa_2_errors
        df['kappa2RelErrors'] = rel_kappa_2_errors
        df['T1Errors'] = T_1_errors
        df['T1AngularErrors'] = T_1_angular_errors
        df['T2Errors'] = T_2_errors
        df['T2AngularErrors'] = T_2_angular_errors
        df.to_csv(VV_eval_file, sep=';')
        # The same for VTK, if the file does not exist yet:
        if not os.path.isfile(VTK_eval_file):
            df = pd.DataFrame()
            df['kappa1'] = vtk_kappa_1
            df['kappa2'] = vtk_kappa_2
            df['kappa1AbsErrors'] = vtk_abs_kappa_1_errors
            df['kappa1RelErrors'] = vtk_rel_kappa_1_errors
            df['kappa2AbsErrors'] = vtk_abs_kappa_2_errors
            df['kappa2RelErrors'] = vtk_rel_kappa_2_errors
            df.to_csv(VTK_eval_file, sep=';')

        # Asserting that all estimated T_1 and T_2 vectors are close to the
        # corresponding true vector, allowing error of 30%:
        print "Testing the maximal principal directions (T_1)..."
        for i, error in enumerate(T_1_errors):
            msg = 'triangle {}: {} is > {}!'.format(i, error, 0.3)
            self.assertLessEqual(error, 0.3, msg=msg)
        print "Testing the minimal principal directions (T_2)..."
        for i, error in enumerate(T_2_errors):
            msg = 'triangle {}: {} is > {}!'.format(i, error, 0.3)
            self.assertLessEqual(error, 0.3, msg=msg)
        # Asserting that all estimated kappa_1 and kappa_2 values are close to
        # the corresponding true values, allowing error of 30% from the true
        # value (the maximal absolute value in case of kappa_2, because it can
        # be 0 or negative):
        print "Testing the maximal principal curvature (kappa_1)..."
        allowed_error = 0.3 * max(true_kappa_1)
        for i, error in enumerate(abs_kappa_1_errors):
            msg = 'triangle {}: {} is > {}%!'.format(i, error, 30)
            self.assertLessEqual(error, allowed_error, msg=msg)
        print "Testing the minimal principal curvature (kappa_2)..."
        allowed_error = 0.3 * max(abs(true_kappa_2))
        for i, error in enumerate(abs_kappa_2_errors):
            msg = 'triangle {}: {} is > {}%!'.format(i, error, 30)
            self.assertLessEqual(error, allowed_error, msg=msg)

    # *** The following tests will be run by unittest ***

    # def test_plane_normals(self):
    #     """
    #     Tests whether normals are correctly estimated for a plane surface with
    #     known orientation (parallel to to X and Y axes), certain size,
    #     resolution and noise level.
    #     """
    #     for n in [10]:
    #         for rh in [4, 6, 8]:
    #             self.parametric_test_plane_normals(
    #                 10, rh, res=10, noise=n)

    # def test_cylinder_directions_curvatures(self):
    #     """
    #     Tests whether minimal principal directions (T_2) and curvatures are
    #     correctly estimated for an opened cylinder surface (without the circular
    #     planes) with known orientation (height, i.e. T_2, parallel to the Z
    #     axis), certain radius and noise level.
    #     """
    #     p = 50
    #     for n in [0]:
    #         for rh in [3, 4, 5, 6, 7]:  # 8
    #             for m in ['VVCF']:  # 'VV', 'VCTV'
    #                 self.parametric_test_cylinder_directions_curvatures(
    #                     10, rh, noise=n, method=m,
    #                     page_curvature_formula=False, num_points=p)

    # def test_inverse_cylinder_directions_curvatures(self):
    #     """
    #     Tests whether maximal principal directions (T_1) and curvatures are
    #     correctly estimated for an inverse opened cylinder surface (without the
    #     circular planes) with known orientation (height, i.e. T_1, parallel to
    #     the Z axis), certain radius and noise level.
    #     """
    #     p = 50
    #     for rh in [8]:
    #         for m in ['VV', 'VVCF', 'VCTV']:
    #             self.parametric_test_cylinder_directions_curvatures(
    #                 10, rh, noise=0, inverse=True, method=m,
    #                 page_curvature_formula=False, num_points=p)

    def test_sphere_curvatures(self):
        """
        Tests whether curvatures are correctly estimated for a sphere with a
        certain radius and noise level:

        kappa1 = kappa2 = 1/5 = 0.2; 30% of difference is allowed
        """
        # Icosahedron sphere with 1280 faces:
        # for n in [0]:
        #     for rh in [3.5]:  # 1, 2, 3, 3.5, 4, 5, 6, 7, 8, 9
        #         for p in [50]:  # 5, 10, 15, 20, 30, 40, 50
        #             self.parametric_test_sphere_curvatures(
        #                 10, rh, ico=1280, noise=n, method='VVCF',
        #                 page_curvature_formula=False, num_points=p)
        #         for m in ['VV', 'VCTV']:
        #             self.parametric_test_sphere_curvatures(
        #                 10, rh, ico=1280, noise=n, method=m,
        #                 page_curvature_formula=False)
        # Binary sphere with different radii:
        for r in [20, 30]:  # 10, 20, 30
            for rh in [9]:  # 5, 6, 7, 8, 9
                for m in ['VV']:  # 'VCTV',
                    self.parametric_test_sphere_curvatures(r, rh, binary=True,
                                                           method=m)

    # def test_inverse_sphere_curvatures(self):
    #     """
    #     Tests whether curvatures are correctly estimated for an inverse sphere
    #     with a certain radius and noise level:
    #
    #     kappa1 = kappa2 = -1/5 = -0.2; 30% of difference is allowed
    #     """
    #     p = 50
    #     for rh in [9]:
    #         for m in ['VV']:  # 'VVCF', 'VCTV'
    #             self.parametric_test_sphere_curvatures(
    #                 10, rh, ico=1280, noise=0, inverse=True, method=m,
    #                 page_curvature_formula=False, num_points=p)

    # def test_torus_curvatures(self):
    #     """
    #     Runs parametric_test_torus_curvatures with certain parameters.
    #     """
    #     p = 50
    #     for rh in [2, 3, 4]:  # 5, 6, 7, 8, 9
    #         for m in ['VVCF']:  # 'VV', 'VCTV'
    #             self.parametric_test_torus_curvatures(
    #                 25, 10, rh, method=m,
    #                 page_curvature_formula=False, num_points=p)


if __name__ == '__main__':
    unittest.main()
