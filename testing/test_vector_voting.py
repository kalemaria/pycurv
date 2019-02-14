import pytest
import time
import os.path
import math
import pandas as pd
import cProfile
import pstats

from pysurf import pysurf_io as io
from pysurf import (
    TriangleGraph, normals_directions_and_curvature_estimation)
from synthetic_surfaces import (
    PlaneGenerator, SphereGenerator, CylinderGenerator, SaddleGenerator,
    ConeGenerator, add_gaussian_noise_to_surface)
from errors_calculation import *

"""
Scripts for testing of validity of curvature estimation methods using
"synthetic" benchmark surfaces.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'

FOLD = '/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/'


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
    # assert(round(math.sqrt(np.dot(T_1, T_1)), 5) == 1.0)
    T_2 = [- sin(u), cos(u), 0]
    # assert(round(math.sqrt(np.dot(T_2, T_2)), 5) == 1.0)
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


def surface_to_graph(surf_file, scale_factor_to_nm=1, inverse=False):
    """
    Reads in the .vtp file with the triangle mesh surface and transforms it
    into a triangle graph.

    Args:
        surf_file (str): .vtp file with the triangle mesh surface
        scale_factor_to_nm (float, optional): pixel size in nanometers for
            scaling the surface and the graph (default 1)
        inverse (boolean, optional): if True, the graph will have normals
            pointing outwards (negative curvature), if False (default), the
            other way around

    Returns:
        surface (vtk.vtkPolyData) and triangle graph(TriangleGraph)
    """
    t_begin = time.time()

    print('\nReading in the surface file to get a vtkPolyData surface...')
    surf = io.load_poly(surf_file)
    print('\nBuilding the TriangleGraph from the vtkPolyData surface with '
          'curvatures...')
    tg = TriangleGraph()
    # VTK has opposite surface normals convention than we use
    # a graph with normals pointing outwards is generated (normal case
    # for this method; negative curvatures)
    if inverse:
        reverse_normals = False
    # a graph with normals pointing inwards is generated (positive
    # curvatures)
    else:
        reverse_normals = True
    tg.build_graph_from_vtk_surface(surf, scale_factor_to_nm, verbose=False,
                                    reverse_normals=reverse_normals)
    print(tg.graph)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Graph construction from surface took: {} min {} s'.format(
        minutes, seconds))
    return surf, tg


"""
Tests for vector_voting.py, assuming that other used functions are correct.
"""


@pytest.mark.parametrize("half_size, radius_hit, res, noise", [
    (20, 8, 20, 10)
])
def test_plane_normals(half_size, radius_hit, res, noise):
    """
    Tests whether normals are correctly estimated for a plane surface with
    known orientation (parallel to to X and Y axes).
    Allowing error of 30%.

    Args:
        half_size (int): half size of the plane (from center to an edge)
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        res (int): resolution (number of divisions) in X and Y axes
        noise (int): determines variance of the Gaussian noise in
            percents of average triangle edge length, the noise
            is added on triangle vertex coordinates in its normal direction

    Returns:
        None
    """
    fold = '{}plane/res{}_noise{}/'.format(FOLD, res, noise)
    if not os.path.exists(fold):
        os.makedirs(fold)
    surf_file = '{}plane_half_size{}.surface.vtp'.format(fold, half_size)
    files_fold = '{}files4plotting/'.format(fold)
    if not os.path.exists(files_fold):
        os.makedirs(files_fold)
    base_filename = "{}plane_half_size{}".format(files_fold, half_size)
    vv_surf_file = '{}.VCTV_rh{}.vtp'.format(base_filename, radius_hit)
    vv_eval_file = '{}.VCTV_rh{}.csv'.format(base_filename, radius_hit)
    vtk_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.normals.gt'.format(base_filename)

    print("\n*** Generating a surface and a graph for a plane with "
          "half-size {} and {}% noise ***".format(half_size, noise))
    # If the .vtp file with the test surface does not exist, create it:
    if not os.path.isfile(surf_file):
        # generate surface directly with VTK
        pg = PlaneGenerator()
        plane = pg.generate_plane_surface(half_size, res)
        if noise > 0:
            plane = add_gaussian_noise_to_surface(plane, percent=noise)
        io.save_vtp(plane, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, tg = surface_to_graph(surf_file)

    # Running the modified Normal Vector Voting algorithm (with curvature
    # tensor voting, because its second pass is the fastest):
    results = normals_directions_and_curvature_estimation(
        tg, radius_hit, exclude_borders=0, methods=['VCTV'], poly_surf=surf,
        graph_file=temp_normals_graph_file)
    surf_vv = results['VCTV'][1]
    # Saving the output (TriangleGraph object) for later inspection in ParaView:
    io.save_vtp(surf_vv, vv_surf_file)

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
        assert error <= 0.3


@pytest.mark.parametrize("radius,radius_hit,eb,inverse,methods", [
    (10, 8, 5, False, ['VV']),
    (10, 8, 5, True, ['VV']),
])
def test_cylinder_directions_curvatures(
        radius, radius_hit, eb, inverse, methods,
        res=0, h=0, noise=0, page_curvature_formula=False, full_dist_map=False,
        area2=True, cores=4):
    """
    Tests whether minimal principal directions (T_2), as well as minimal and
    maximal principal curvatures are correctly estimated for an opened
    cylinder surface (without the circular planes) with known
    orientation (height, i.e. T_2, parallel to the Z axis) using normal
    vector voting (VV) or VV combined with curvature tensor voting (VCTV).
    Allowing error of +-30% of the maximal absolute true value.

    Args:
        radius (int): cylinder radius in voxels
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        eb (float): distance from borders to exclude in length unit of the
            graph, here voxels
        inverse (boolean): if True, the cylinder will have normals pointing
            outwards (negative curvature), else the other way around
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'VCTV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
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
        page_curvature_formula (boolean, optional): if True (default False)
            normal curvature formula from Page at al. is used for VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map
            is calculated for each vertex (default)
        area2 (boolean, optional): if True (default), votes are
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        cores (int): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """
    if res == 0:
        fold = '{}cylinder/noise{}/'.format(FOLD, noise)
    else:
        fold = '{}cylinder/res{}_noise{}/'.format(FOLD, res, noise)
    if not os.path.exists(fold):
        os.makedirs(fold)

    if res == 0 and h != 0:
        h = 0  # h has to be also 0 if res is 0
    if h == 0:
        h = int(math.ceil(radius * 2.5))  # set h to 2.5 * radius, if not given

    surf_filebase = '{}cylinder_r{}_h{}'.format(fold, radius, h)
    surf_file = '{}.surface.vtp'.format(surf_filebase)
    files_fold = '{}files4plotting/'.format(fold)
    if not os.path.exists(files_fold):
        os.makedirs(files_fold)
    if inverse:
        inverse_str = "inverse_"
    else:
        inverse_str = ""
    base_filename = "{}{}cylinder_r{}_h{}_eb{}".format(
        files_fold, inverse_str, radius, h, eb)
    VTK_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.normals.gt'.format(base_filename)

    if inverse:
        print("\n*** Generating a surface and a graph for an inverse "
              "cylinder with radius {}, height {} and {}% noise ***".format(
                radius, h, noise))
    else:
        print("\n*** Generating a surface and a graph for a cylinder with "
              "radius {}, height {} and {}% noise ***".format(radius, h, noise))
    # If the .vtp file with the test surface does not exist, create it:
    if not os.path.isfile(surf_file):
        cg = CylinderGenerator()
        if res == 0:  # generate surface from a gaussian mask
            cylinder = cg.generate_gauss_cylinder_surface(radius)
        else:  # generate surface directly with VTK
            print("Warning: cylinder contains planes!")
            cylinder = cg.generate_cylinder_surface(radius, h, res)
        if noise > 0:
            cylinder = add_gaussian_noise_to_surface(cylinder, percent=noise)
        io.save_vtp(cylinder, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, tg = surface_to_graph(surf_file, inverse=inverse)

    # Running the modified Normal Vector Voting algorithm:
    method_tg_surf_dict = normals_directions_and_curvature_estimation(
        tg, radius_hit, exclude_borders=eb, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        graph_file=temp_normals_graph_file)

    # Ground-truth T_h vector is parallel to Z axis
    true_T_h = np.array([0, 0, 1])

    # Ground-truth principal curvatures
    if inverse:
        true_kappa_1 = 0.0
        true_kappa_2 = - 1.0 / radius
    else:
        true_kappa_1 = 1.0 / radius
        true_kappa_2 = 0.0

    for method in method_tg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (tg, surf) = method_tg_surf_dict[method]
        if method == 'VV' and page_curvature_formula:
            method = 'VV_page_curvature_formula'
        if (method == 'VV' or method == 'VV_page_curvature_formula') and area2:
            method = '{}_area2'.format(method)
        surf_file = '{}.{}_rh{}.vtp'.format(
            base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)

        # Evaluating each method:
        print("\nEvaluating {}...".format(method))
        eval_file = '{}.{}_rh{}.csv'.format(
            base_filename, method, radius_hit)

        # Getting the estimated principal directions along cylinder height:
        pos = [0, 1, 2]  # vector-property value positions
        if not inverse:  # it's the minimal direction
            T_h = tg.graph.vertex_properties["T_2"].get_2d_array(pos)
        else:  # it's the maximal direction
            T_h = tg.graph.vertex_properties["T_1"].get_2d_array(pos)
        # The shape is (3, <num_vertices>) - have to transpose to group the
        # respective x, y, z components to sub-arrays
        T_h = np.transpose(T_h)  # shape (<num_vertices>, 3)

        # Computing errors of the estimated T_h vectors wrt the true one:
        T_h_errors = np.array(map(lambda x: error_vector(true_T_h, x), T_h))
        T_h_angular_errors = np.array(map(
            lambda x: angular_error_vector(true_T_h, x), T_h))

        # Getting estimated and VTK principal curvatures from the output
        # graph:
        kappa_1 = tg.get_vertex_property_array("kappa_1")
        kappa_2 = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1 = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2 = tg.get_vertex_property_array("min_curvature")

        # Calculating errors of the principal curvatures:
        if not inverse:
            abs_kappa_1_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_1, x), kappa_1))
            rel_kappa_1_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_1, x), kappa_1))
            vtk_abs_kappa_1_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_1, x), vtk_kappa_1))
            vtk_rel_kappa_1_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_1, x), vtk_kappa_1))
        else:  # inverse
            abs_kappa_2_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_2, x), kappa_2))
            rel_kappa_2_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_2, x), kappa_2))
            vtk_abs_kappa_2_errors = np.array(map(
                lambda x: absolute_error_scalar(true_kappa_2, x), vtk_kappa_2))
            vtk_rel_kappa_2_errors = np.array(map(
                lambda x: relative_error_scalar(true_kappa_2, x), vtk_kappa_2))

        # Writing all the curvature values and errors into a csv file:
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
        df.to_csv(eval_file, sep=';')
        # The same for VTK, if the file does not exist yet:
        if not os.path.isfile(VTK_eval_file):
            df = pd.DataFrame()
            df['kappa1'] = vtk_kappa_1
            df['kappa2'] = vtk_kappa_2
            if not inverse:
                df['kappa1AbsErrors'] = vtk_abs_kappa_1_errors
                df['kappa1RelErrors'] = vtk_rel_kappa_1_errors
            else:  # inverse
                df['kappa2AbsErrors'] = vtk_abs_kappa_2_errors
                df['kappa2RelErrors'] = vtk_rel_kappa_2_errors
            df.to_csv(VTK_eval_file, sep=';')

        # Asserting that all estimated T_h vectors are close to the true
        # vector, allowing error of 30%:
        if not inverse:
            print("Testing the minimal principal directions (T_2)...")
        else:
            print("Testing the maximal principal directions (T_1)...")
        for error in T_h_errors:
            assert error <= 0.3

        # Asserting that all principal curvatures are close to the correct
        # ones allowing error of +-30% of the maximal absolute true value
        allowed_error = 0.3 * max(abs(true_kappa_1), abs(true_kappa_2))
        if not inverse:
            print("Testing the maximal principal curvature (kappa_1)...")
            for error in abs_kappa_1_errors:
                assert error <= allowed_error
        else:  # inverse
            print("Testing the minimal principal curvature (kappa_2)...")
            for error in abs_kappa_2_errors:
                assert error <= allowed_error


@pytest.mark.parametrize(
    "radius,radius_hit,inverse,binary,ico,methods, runtimes", [
        (10, 3.5, False, False, 1280, ['VV'], None),  # icosahedron
        (10, 9, False, True, 0, ['VV'],
            "{}sphere/binary/files4plotting/bin_spheres_runtimes.csv".format(
             FOLD)),  # binary
        (10, 9, False, False, 0, ['VV', 'VCTV'], None),  # gaussian
        (10, 8, True, False, 0, ['VV', 'VCTV'], None),  # gaussian inverse
    ])
def test_sphere_curvatures(
        radius, radius_hit, inverse, methods, binary, ico, runtimes,
        res=0, noise=0, save_areas=False, page_curvature_formula=False,
        full_dist_map=False, area2=True, cores=4):
    """
    Runs all the steps needed to calculate curvatures for a test sphere
    with a given radius. Tests whether the curvatures are correctly
    estimated using normal vector voting (VV), VV combined with curvature tensor
    voting (VCTV).
    kappa_1 = kappa_2 = 1/r; allowing error of +-30%.

    Args:
        radius (int): radius of the sphere
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        inverse (boolean): if True, the sphere will have normals pointing
            outwards (negative curvature), else the other way around
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'VCTV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        binary (boolean): if True, a binary sphere is generated (ignoring the
            options ico, res and noise)
        ico (int): if > 0 and res=0, an icosahedron with so many faces is used
            (1280 faces with radius 1 or 10 are available so far)
        runtimes (str): if given, runtimes and some parameters are added to
            this file (otherwise None)
        res (int, optional): if > 0 determines how many longitude and
            latitude stripes the UV sphere from vtkSphereSource has, the
            surface is triangulated; If 0 (default) and ico=0, first a
            gaussian sphere mask is generated and then surface using
            vtkMarchingCubes
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 10), the noise
            is added on triangle vertex coordinates in its normal direction
        save_areas (boolean, optional): if True (default False), also mesh
            triangle ares will be saved to a file
        page_curvature_formula (boolean, optional): if True (default False)
            normal curvature formula from Page et al. is used for VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map
            is calculated for each vertex (default)
        area2 (boolean, optional): if True (default), votes are
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        cores (int): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """
    if binary:
        fold = '{}sphere/binary/'.format(FOLD)
    else:
        if res > 0:  # UV sphere with this longitude and latitude res. is used
            fold = '{}sphere/res{}_noise{}/'.format(FOLD, res, noise)
        elif ico > 0:  # icosahedron sphere with so many faces is used
            fold = '{}sphere/ico{}_noise{}/'.format(FOLD, ico, noise)
        else:  # a sphere generated by a gaussian mask is used
            fold = '{}sphere/noise{}/'.format(FOLD, noise)

    if not os.path.exists(fold):
        os.makedirs(fold)
    surf_filebase = '{}sphere_r{}'.format(fold, radius)
    surf_file = '{}.surface.vtp'.format(surf_filebase)
    files_fold = '{}files4plotting/'.format(fold)
    if not os.path.exists(files_fold):
        os.makedirs(files_fold)
    if inverse:
        inverse_str = "inverse_"
    else:
        inverse_str = ""
    base_filename = "{}{}sphere_r{}".format(files_fold, inverse_str, radius)
    VTK_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.normals.gt'.format(base_filename)

    if inverse:
        print("\n*** Generating a surface and a graph for an inverse "
              "sphere with radius {} and {}% noise***".format(radius, noise))
    else:
        print("\n*** Generating a surface and a graph for a sphere with "
              "radius {} and {}% noise***".format(radius, noise))
    # If the .vtp file with the test surface does not exist, create it:
    if not os.path.isfile(surf_file):
        sg = SphereGenerator()
        if binary:
            sphere = sg.generate_binary_sphere_surface(radius)
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
                print("Sorry, you have to generate the icosahedron\n"
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

    # Reading in the surface and transforming it into a triangle graph
    surf, tg = surface_to_graph(surf_file, inverse=inverse)

    # Running the modified Normal Vector Voting algorithm:
    if runtimes is not None and not os.path.isfile(runtimes):
        with open(runtimes, 'w') as f:
            f.write("num_v;radius_hit;g_max;avg_num_neighbors;cores;"
                    "duration1;method;duration2\n")
    method_tg_surf_dict = normals_directions_and_curvature_estimation(
        tg, radius_hit, exclude_borders=0, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        runtimes=runtimes, graph_file=temp_normals_graph_file)

    # Ground truth principal curvatures
    true_curvature = 1.0 / radius
    if inverse:
        true_curvature *= -1

    for method in method_tg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (tg, surf) = method_tg_surf_dict[method]
        if method == 'VV' and page_curvature_formula:
            method = 'VV_page_curvature_formula'
        if (method == 'VV' or method == 'VV_page_curvature_formula') and area2:
            method = '{}_area2'.format(method)
        surf_file = '{}.{}_rh{}.vtp'.format(base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)

        # Evaluating each method:
        print("\nEvaluating {}...".format(method))
        eval_file = '{}.{}_rh{}.csv'.format(
            base_filename, method, radius_hit)

        # Getting estimated principal curvatures from the output graph:
        kappa_1 = tg.get_vertex_property_array("kappa_1")
        kappa_2 = tg.get_vertex_property_array("kappa_2")

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
        df.to_csv(eval_file, sep=';')

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

        # Asserting that all values of both principal curvatures are close
        # to the true value, allowing error of +-30%:
        allowed_error = 0.3 * abs(true_curvature)
        print("Testing the maximal principal curvature (kappa_1)...")
        for error in abs_kappa_1_errors:
            assert error <= allowed_error
        print("Testing the minimal principal curvature (kappa_2)...")
        for error in abs_kappa_2_errors:
            assert error <= allowed_error


@pytest.mark.parametrize("rr,csr,radius_hit,methods,runtimes,cores", [
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 1),
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 2),
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 3),
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 4),
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 5),
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 6),
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 7),
    (25, 10, 8, ['VV'],
        "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD), 8),
# (25, 10, 8, ['VCTV'],
#         "{}torus/files4plotting/torus_rr25_csr10_runtimes.csv".format(FOLD)),
])
def test_torus_directions_curvatures(
        rr, csr, radius_hit, methods, runtimes, cores,
        page_curvature_formula=False, full_dist_map=False, area2=True):
    """
    Runs all the steps needed to calculate curvatures for a test torus
    with given radii using normal vector voting (VV) or VV combined with
    curvature tensor voting (VCTV).
    Allowing error of +-30%.

    Args:
        rr (int): ring radius of the torus
        csr (int): cross-section radius of the torus
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'VCTV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        runtimes (str): if given, runtimes and some parameters are added to
            this file (otherwise None)
        cores (int): number of cores to run VV in parallel
        page_curvature_formula (boolean, optional): if True (default False)
            normal curvature formula from Page et al. is used for VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map
            is calculated for each vertex (default)
        area2 (boolean, optional): if True (default), votes are
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)

    Notes:
        * csr should be much smaller than rr (csr < rr - csr).

    Returns:
        None
    """
    fold = '{}torus/'.format(FOLD)
    if not os.path.exists(fold):
        os.makedirs(fold)
    surf_filebase = '{}torus_rr{}_csr{}'.format(fold, rr, csr)
    surf_file = '{}.surface.vtp'.format(surf_filebase)
    files_fold = '{}files4plotting/'.format(fold)
    if not os.path.exists(files_fold):
        os.makedirs(files_fold)
    base_filename = "{}torus_rr{}_csr{}".format(files_fold, rr, csr)
    VTK_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.normals.gt'.format(base_filename)

    print("\n*** Generating a surface and a graph for a torus with ring "
          "radius {} and cross-section radius {}***".format(rr, csr))
    # If the .vtp file with the test surface does not exist, create it:
    if not os.path.isfile(surf_file):
        sg = SaddleGenerator()
        torus = sg.generate_parametric_torus(rr, csr)
        io.save_vtp(torus, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, tg = surface_to_graph(surf_file, inverse=False)

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

    # Getting the true principal directions and principal curvatures:
    pos = [0, 1, 2]  # vector-property value positions
    # The shape is (3, <num_vertices>) - have to transpose to group the
    # respective x, y, z components to sub-arrays
    true_T_1 = np.transpose(
        tg.graph.vertex_properties["true_T_1"].get_2d_array(pos))
    true_T_2 = np.transpose(
        tg.graph.vertex_properties["true_T_2"].get_2d_array(pos))
    true_kappa_1 = tg.get_vertex_property_array("true_kappa_1")
    true_kappa_2 = tg.get_vertex_property_array("true_kappa_2")

    # Running the modified Normal Vector Voting algorithm:
    if runtimes is not None and not os.path.isfile(runtimes):
        with open(runtimes, 'w') as f:
            f.write("num_v;radius_hit;g_max;avg_num_neighbors;cores;"
                    "duration1;method;duration2\n")
    method_tg_surf_dict = normals_directions_and_curvature_estimation(
        tg, radius_hit, exclude_borders=0, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        runtimes=runtimes, graph_file=temp_normals_graph_file)

    for method in method_tg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (tg, surf) = method_tg_surf_dict[method]
        if method == 'VV' and page_curvature_formula:
            method = 'VV_page_curvature_formula'
        if (method == 'VV' or method == 'VV_page_curvature_formula') and area2:
            method = '{}_area2'.format(method)
        surf_file = '{}.{}_rh{}.vtp'.format(base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)

        # Evaluating each method:
        print("\nEvaluating {}...".format(method))
        eval_file = '{}.{}_rh{}.csv'.format(base_filename, method, radius_hit)

        # Getting the estimated and true principal directions:
        T_1 = np.transpose(tg.graph.vertex_properties["T_1"].get_2d_array(pos))
        T_2 = np.transpose(tg.graph.vertex_properties["T_2"].get_2d_array(pos))

        # Computing errors of the estimated directions wrt the true ones:
        T_1_errors = np.array(map(
            lambda x, y: error_vector(x, y), true_T_1, T_1))
        T_1_angular_errors = np.array(map(
            lambda x, y: angular_error_vector(x, y), true_T_1, T_1))
        T_2_errors = np.array(map(
            lambda x, y: error_vector(x, y), true_T_2, T_2))
        T_2_angular_errors = np.array(map(
            lambda x, y: angular_error_vector(x, y), true_T_2, T_2))

        # Getting the estimated principal curvatures:
        kappa_1 = tg.get_vertex_property_array("kappa_1")
        kappa_2 = tg.get_vertex_property_array("kappa_2")
        vtk_kappa_1 = tg.get_vertex_property_array("max_curvature")
        vtk_kappa_2 = tg.get_vertex_property_array("min_curvature")

        # Computing errors of the estimated curvatures wrt the true ones:
        abs_kappa_1_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y), true_kappa_1, kappa_1))
        rel_kappa_1_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y), true_kappa_1, kappa_1))
        abs_kappa_2_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y), true_kappa_2, kappa_2))
        rel_kappa_2_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y), true_kappa_2, kappa_2))
        vtk_abs_kappa_1_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y),
            true_kappa_1, vtk_kappa_1))
        vtk_rel_kappa_1_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y),
            true_kappa_1, vtk_kappa_1))
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
        df.to_csv(eval_file, sep=';')
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
        print("Testing the maximal principal directions (T_1)...")
        for error in T_1_errors:
            assert error <= 0.3
        print("Testing the minimal principal directions (T_2)...")
        for error in T_2_errors:
            assert error <= 0.3
        # Asserting that all estimated kappa_1 and kappa_2 values are close
        # to the corresponding true values, allowing error of 30% from the
        # true value (the maximal absolute value in case of kappa_2, because
        # it can be 0 or negative):
        print("Testing the maximal principal curvature (kappa_1)...")
        allowed_error = 0.3 * max(true_kappa_1)
        for error in abs_kappa_1_errors:
            assert error <= allowed_error
        print("Testing the minimal principal curvature (kappa_2)...")
        allowed_error = 0.3 * max(abs(true_kappa_2))
        for error in abs_kappa_2_errors:
            assert error <= allowed_error


@pytest.mark.parametrize("r,h,radius_hit,res,methods", [
    (6, 6, 5, 38, ['VV', 'VCTV']),  # smooth
])
def run_cone(  # does not include assert for true curvature!
        r, h, radius_hit, methods, res=0, noise=0, page_curvature_formula=False,
        full_dist_map=False, area2=True, cores=4):
    """
    Runs all the steps needed to calculate curvatures for a test cone with given
    radius and height using normal vector voting (VV) or VV combined with
    curvature tensor voting (VCTV).
    Writes out kappa_1 and kappa_2 values to a CSV file.

    Args:
        r (int): cone base radius in voxels
        h (int): cone height in voxels
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'VCTV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        res (int, optional): if > 0 determines how many triangles around the
            circular base the cone has, is subdivided and smoothed, the base
            disappears; If 0 (default) a binary cone with the circular base
            is generated
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 0), the noise
            is added on triangle vertex coordinates in its normal direction
            - only for a smoothed cone, res > 0!
        page_curvature_formula (boolean, optional): if True (default False)
            normal curvature formula from Page et al. is used for VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map
            is calculated for each vertex (default)
        area2 (boolean, optional): if True (default), votes are
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        cores (int): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """
    if res == 0:
        noise = 0
        fold = '{}cone/binary/'.format(FOLD)
    else:
        fold = '{}cone/res{}_noise{}/'.format(FOLD, res, noise)
    if not os.path.exists(fold):
        os.makedirs(fold)

    surf_filebase = '{}cone_r{}_h{}'.format(fold, r, h)
    surf_file = '{}.surface.vtp'.format(surf_filebase)
    files_fold = '{}files4plotting/'.format(fold)
    if not os.path.exists(files_fold):
        os.makedirs(files_fold)
    base_filename = "{}cone_r{}_h{}".format(files_fold, r, h)
    # VTK_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.normals.gt'.format(base_filename)

    print("\n*** Generating a surface and a graph for a cone with radius "
          "{}, height {} and {}% noise ***".format(r, h, noise))
    # If the .vtp file with the test surface does not exist, create it:
    if not os.path.isfile(surf_file):
        cg = ConeGenerator()
        if res == 0:  # generate surface from a binary mask
            print("Warning: cone contains a plane!")
            cone = cg.generate_binary_cone_surface(r, h)
        else:  # generate surface directly with VTK
            cone = cg.generate_cone(r, h, res, subdivisions=3, decimate=0.8)
            if noise > 0:
                cone = add_gaussian_noise_to_surface(cone, percent=noise)
        io.save_vtp(cone, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, tg = surface_to_graph(surf_file, inverse=False)

    # Running the modified Normal Vector Voting algorithm:
    method_tg_surf_dict = normals_directions_and_curvature_estimation(
        tg, radius_hit, exclude_borders=1, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        graph_file=temp_normals_graph_file)

    for method in method_tg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (tg, surf) = method_tg_surf_dict[method]
        if method == 'VV' and page_curvature_formula:
            method = 'VV_page_curvature_formula'
        if (method == 'VV' or method == 'VV_page_curvature_formula') and area2:
            method = '{}_area2'.format(method)
        surf_file = '{}.{}_rh{}.vtp'.format(base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)

        # Getting the estimated principal curvatures:
        kappa_1 = tg.get_vertex_property_array("kappa_1")
        kappa_2 = tg.get_vertex_property_array("kappa_2")

        # Writing all the VV curvature values and errors into a csv file:
        df = pd.DataFrame()
        df['kappa1'] = kappa_1
        df['kappa2'] = kappa_2

        csv_file = '{}.{}_rh{}.csv'.format(
            base_filename, method, radius_hit)
        df.to_csv(csv_file, sep=';')


# py.test -n 4   # test on multiple CPUs

if __name__ == "__main__":
    fold = '{}sphere/binary/'.format(FOLD)
    stats_file = '{}sphere_r10.VV_rh9.stats'.format(fold)
    cProfile.run('test_sphere_curvatures(radius=10, radius_hit=9, '
                 'inverse=False, binary=True, ico=0, methods=[\'VV\'], '
                 'runtimes=None, cores=1)', stats_file)

    p = pstats.Stats(stats_file)
    # what algorithms are taking time:
    # p.strip_dirs().sort_stats('cumulative').print_stats(10)
    # what functions were looping a lot, and taking a lot of time:
    p.strip_dirs().sort_stats('time').print_stats(10)
