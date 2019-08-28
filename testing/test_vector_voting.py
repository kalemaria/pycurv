import pytest
import time
import os.path
from os import remove
import math
import pandas as pd
import sys
import vtk
# import cProfile
# import pstats

from curvaturia import curvaturia_io as io
from curvaturia import (
    TriangleGraph, PointGraph, normals_directions_and_curvature_estimation,
    nice_acos, nice_asin)
from synthetic_surfaces import (
    PlaneGenerator, SphereGenerator, CylinderGenerator, SaddleGenerator,
    add_gaussian_noise_to_surface)
from errors_calculation import *

"""
Scripts for testing of validity of curvature estimation methods using
"synthetic" benchmark surfaces.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'

# FOLD = '/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces_benchmarking/'
FOLD = './test_vector_voting_output/'


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
    asin = nice_asin
    acos = nice_acos
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


def surface_to_graph(surf_file, scale=(1, 1, 1), inverse=False,
                     vertex_based=False):
    """
    Reads in the .vtp file with the triangle mesh surface and transforms it
    into a triangle graph.

    Args:
        surf_file (str): .vtp file with the triangle mesh surface
        scale (float, optional): scale factor (X, Y, Z) in given units for
            scaling the surface and the graph (default (1, 1, 1))
        inverse (boolean, optional): if True, the graph will have normals
            pointing outwards (negative curvature), if False (default), the
            other way around
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center.

    Returns:
        surface (vtk.vtkPolyData) and triangle graph(TriangleGraph)
    """
    t_begin = time.time()

    print('\nReading in the surface file to get a vtkPolyData surface...')
    surf = io.load_poly(surf_file)
    if vertex_based:
        print('\nBuilding the PointGraph from the vtkPolyData surface with '
              'curvatures...')
        sg = PointGraph()
    else:
        print('\nBuilding the TriangleGraph from the vtkPolyData surface with '
              'curvatures...')
        sg = TriangleGraph()
    sg.build_graph_from_vtk_surface(surf, scale, verbose=False,
                                    reverse_normals=inverse)
    print(sg.graph)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Graph construction from surface took: {} min {} s'.format(
        minutes, seconds))
    return surf, sg


"""
Tests for vector_voting.py, assuming that other used functions are correct.
"""


@pytest.mark.parametrize("half_size,radius_hit,res,noise,vertex_based,cores", [
    (20, 4, 20, 10, False, 4),
    (20, 8, 20, 10, False, 4),
    # pytest.param(20, 4, 20, 10, True, 4,
    #              marks=pytest.mark.xfail(reason="too high errors")),  # vertex
    # (20, 8, 20, 10, True, 4),  # vertex
])
def test_plane_normals(half_size, radius_hit, res, noise, vertex_based, cores):
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
        vertex_based (boolean): if True, curvature is calculated per triangle
            vertex instead of triangle center
        cores (int): number of cores to run VV in parallel

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
    if vertex_based:
        vertex_based_str = "_vertex_based"
    else:
        vertex_based_str = ""
    base_filename = "{}plane_half_size{}{}".format(
        files_fold, half_size, vertex_based_str)
    vv_surf_file = '{}.SSVV_rh{}.vtp'.format(base_filename, radius_hit)
    vv_eval_file = '{}.SSVV_rh{}.csv'.format(base_filename, radius_hit)
    vtk_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.VV_rh{}_normals.gt'.format(
        base_filename, radius_hit)
    log_file = '{}.SSVV_rh{}.log'.format(base_filename, radius_hit)
    sys.stdout = open(log_file, 'w')

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

    # Reading in the surface and transforming it into a graph
    surf, sg = surface_to_graph(surf_file, vertex_based=vertex_based)

    # Running the modified Normal Vector Voting algorithm (with curvature
    # tensor voting, because its second pass is the fastest):
    results = normals_directions_and_curvature_estimation(
        sg, radius_hit, methods=['SSVV'], poly_surf=surf,
        graph_file=temp_normals_graph_file, cores=cores)
    # Remove the normals graph file, so the next test will start anew
    remove(temp_normals_graph_file)

    sg = results['SSVV'][0]
    surf_vv = results['SSVV'][1]
    # Saving the output (TriangleGraph object) for later inspection in ParaView:
    io.save_vtp(surf_vv, vv_surf_file)

    # Getting the initial and the estimated normals
    pos = [0, 1, 2]  # vector-property value positions
    vtk_normals = sg.graph.vertex_properties["normal"].get_2d_array(pos)
    vv_normals = sg.graph.vertex_properties["N_v"].get_2d_array(pos)
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


# @pytest.mark.parametrize("radius_hit", range(3, 11))
# @pytest.mark.parametrize("radius,eb,inverse,methods,area2,voxel,vertex_based", [
#     # pytest.param(10, 0, False, ['VV'], True, True, False,  # AVV, voxel
#     #              marks=pytest.mark.xfail(reason="too high errors")),
#     # pytest.param(10, 0, False, ['VV'], False, True, False,  # RVV, voxel
#     #              marks=pytest.mark.xfail(reason="too high errors")),
#     # pytest.param(10, 0, False, ['SSVV'], False, True, False,  # RVV, voxel
#     #              marks=pytest.mark.xfail(reason="too high errors")),
#     pytest.param(10, 0, False, ['VV'], True, False, False,  # AVV, smooth
#                  marks=pytest.mark.xfail(reason="too high errors")),
#     # pytest.param(10, 0, False, ['VV'], False, False, False,  # RVV, smooth
#     #              marks=pytest.mark.xfail(reason="too high errors")),
#     # pytest.param(10, 0, False, ['SSVV'], False, False, False,  # RVV, smooth
#     #              marks=pytest.mark.xfail(reason="too high errors")),
# ])
@pytest.mark.parametrize(
    "radius,radius_hit,eb,inverse,methods,area2,voxel,vertex_based", [
        # smooth cylinder
        (10, 4, 5, False, ['VV'], True, False, False),  # AVV
        (10, 4, 5, False, ['VV'], False, False, False),  # RVV
        (10, 6, 5, False, ['SSVV'], False, False, False),
        # pytest.param(10, 5, 0, False, ['VV'], True, False, False,
        #              marks=pytest.mark.xfail(reason="too high errors")),  # AVV
        # pytest.param(10, 5, 0, False, ['VV'], False, False, False,
        #              marks=pytest.mark.xfail(reason="too high errors")),  # RVV
        # (10, 6, 0, False, ['SSVV'], False, False, False),
        # pytest.param(10, 5, 0, False, ['VV'], False, False, True,  # RVV, vertex
        #              marks=pytest.mark.xfail(reason="too high errors")),
        # (10, 6, 0, False, ['SSVV'], False, False, True),  # SSVV, vertex
        # noisy cylinder
        # pytest.param(10, 9, 0, False, ['VV'], True, True, False,
        #              marks=pytest.mark.xfail(reason="too high errors")),  # AVV
        # (10, 9, 5, False, ['VV'], True, True, False),  # AVV
        # (10, 8, 5, False, ['SSVV'], False, True, False),  # SSVV
        # (10, 10, 5, False, ['VV'], False, True, False),  # RVV
    ])
def test_cylinder_directions_curvatures(
        radius, radius_hit, eb, inverse, methods, area2, voxel, vertex_based,
        res=0, h=0, noise=0, page_curvature_formula=False, full_dist_map=False,
        cores=4):
    """
    Tests whether minimal principal directions (T_2), as well as minimal and
    maximal principal curvatures are correctly estimated for an opened
    cylinder surface (without the circular planes) with known
    orientation (height, i.e. T_2, parallel to the Z axis) using normal
    vector voting (VV) or VV combined with curvature tensor voting (SSVV).
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
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation; not possible if vertex_based is True)
        voxel (boolean): if True, a voxel cylinder is generated (ignoring the
            options res and noise)
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'SSVV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        vertex_based (boolean): if True, curvature is calculated per triangle
            vertex instead of triangle center.
        res (int, optional): if > 0 determines how many stripes around both
            approximate circles (and then triangles) the cylinder has, the
            surface is generated using vtkCylinderSource; If 0 (default)
            first a smooth cylinder mask is generated and then surface
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
        cores (int): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """
    if voxel:
        fold = '{}cylinder/voxel/'.format(FOLD)
    else:
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
    if vertex_based:
        area2 = False
        vertex_based_str = "_vertex_based"
    else:
        vertex_based_str = ""
    base_filename = "{}{}cylinder_r{}_h{}_eb{}{}".format(
        files_fold, inverse_str, radius, h, eb, vertex_based_str)
    VTK_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.VV_rh{}_normals.gt'.format(
        base_filename, radius_hit)
    methods_str = ""
    for method in methods:
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        methods_str += (method + '_')
    log_file = '{}.{}rh{}.log'.format(base_filename, methods_str, radius_hit)
    sys.stdout = open(log_file, 'w')

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
        if voxel:
            cylinder = cg.generate_voxel_cylinder_surface(radius)
        else:
            if res == 0:  # generate surface from a smooth mask
                cylinder = cg.generate_gauss_cylinder_surface(radius)
            else:  # generate surface directly with VTK
                print("Warning: cylinder contains planes!")
                cylinder = cg.generate_cylinder_surface(radius, h, res)
            if noise > 0:
                cylinder = add_gaussian_noise_to_surface(cylinder,
                                                         percent=noise)
        io.save_vtp(cylinder, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, sg = surface_to_graph(surf_file, inverse=inverse,
                                vertex_based=vertex_based)

    # Running the modified Normal Vector Voting algorithm:
    method_sg_surf_dict = normals_directions_and_curvature_estimation(
        sg, radius_hit, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        graph_file=temp_normals_graph_file)
    # Remove the normals graph file, so the next test will start anew
    remove(temp_normals_graph_file)

    # Ground-truth T_h vector is parallel to Z axis
    true_T_h = np.array([0, 0, 1])

    # Ground-truth principal curvatures
    if inverse:
        true_kappa_1 = 0.0
        true_kappa_2 = - 1.0 / radius
    else:
        true_kappa_1 = 1.0 / radius
        true_kappa_2 = 0.0

    for method in method_sg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (sg, surf) = method_sg_surf_dict[method]
        if vertex_based is False:  # cannot exclude borders for PointGraph
            # Exclude values at surface borders:
            sg.find_vertices_near_border(eb, purge=True)  # sg is TriangleGraph
            print('\nExcluded triangles that are {} to surface borders.'.format(
                eb))
            print(sg.graph)
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
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
            T_h = sg.graph.vertex_properties["T_2"].get_2d_array(pos)
        else:  # it's the maximal direction
            T_h = sg.graph.vertex_properties["T_1"].get_2d_array(pos)
        # The shape is (3, <num_vertices>) - have to transpose to group the
        # respective x, y, z components to sub-arrays
        T_h = np.transpose(T_h)  # shape (<num_vertices>, 3)

        # Computing errors of the estimated T_h vectors wrt the true one:
        T_h_errors = np.array(map(lambda x: error_vector(true_T_h, x), T_h))
        T_h_angular_errors = np.array(map(
            lambda x: angular_error_vector(true_T_h, x), T_h))

        # Getting estimated and VTK principal curvatures from the output graph:
        kappa_1 = sg.get_vertex_property_array("kappa_1")
        kappa_2 = sg.get_vertex_property_array("kappa_2")
        vtk_kappa_1 = sg.get_vertex_property_array("max_curvature")
        vtk_kappa_2 = sg.get_vertex_property_array("min_curvature")

        # Calculating errors of the principal curvatures:
        abs_kappa_1_errors = np.array(map(
            lambda x: absolute_error_scalar(true_kappa_1, x), kappa_1))
        rel_kappa_1_errors = np.array(map(
            lambda x: relative_error_scalar(true_kappa_1, x), kappa_1))
        vtk_abs_kappa_1_errors = np.array(map(
            lambda x: absolute_error_scalar(true_kappa_1, x), vtk_kappa_1))
        vtk_rel_kappa_1_errors = np.array(map(
            lambda x: relative_error_scalar(true_kappa_1, x), vtk_kappa_1))
        abs_kappa_2_errors = np.array(map(
            lambda x: absolute_error_scalar(true_kappa_2, x), kappa_2))
        rel_kappa_2_errors = np.array(map(
            lambda x: relative_error_scalar(true_kappa_2, x), kappa_2))
        vtk_abs_kappa_2_errors = np.array(map(
            lambda x: absolute_error_scalar(true_kappa_2, x), vtk_kappa_2))
        vtk_rel_kappa_2_errors = np.array(map(
            lambda x: relative_error_scalar(true_kappa_2, x), vtk_kappa_2))

        # Calculating estimated and VTK mean curvatures and their errors:
        true_mean_curv = (true_kappa_1 + true_kappa_2) / 2.0
        mean_curv = [(k_1 + k_2) / 2 for k_1, k_2 in zip(kappa_1, kappa_2)]
        vtk_mean_curv = [(k_1 + k_2) / 2 for k_1, k_2 in zip(
            vtk_kappa_1, vtk_kappa_2)]
        abs_mean_curv_errors = np.array(map(
            lambda x: absolute_error_scalar(true_mean_curv, x), mean_curv))
        rel_mean_curv_errors = np.array(map(
            lambda x: relative_error_scalar(true_mean_curv, x), mean_curv))
        vtk_abs_mean_curv_errors = np.array(map(
            lambda x: absolute_error_scalar(true_mean_curv, x), vtk_mean_curv))
        vtk_rel_mean_curv_errors = np.array(map(
            lambda x: relative_error_scalar(true_mean_curv, x), vtk_mean_curv))

        # Writing all the curvature values and errors into a csv file:
        df = pd.DataFrame()
        df['kappa1'] = kappa_1
        df['kappa2'] = kappa_2
        df['kappa1AbsErrors'] = abs_kappa_1_errors
        df['kappa1RelErrors'] = rel_kappa_1_errors
        df['kappa2AbsErrors'] = abs_kappa_2_errors
        df['kappa2RelErrors'] = rel_kappa_2_errors
        if inverse:
            df['T1Errors'] = T_h_errors
            df['T1AngularErrors'] = T_h_angular_errors
        else:
            df['T2Errors'] = T_h_errors
            df['T2AngularErrors'] = T_h_angular_errors
        df['mean_curvatureAbsErrors'] = abs_mean_curv_errors
        df['mean_curvatureRelErrors'] = rel_mean_curv_errors
        df.to_csv(eval_file, sep=';')
        # The same for VTK:
        df = pd.DataFrame()
        df['kappa1'] = vtk_kappa_1
        df['kappa2'] = vtk_kappa_2
        df['kappa1AbsErrors'] = vtk_abs_kappa_1_errors
        df['kappa1RelErrors'] = vtk_rel_kappa_1_errors
        df['kappa2AbsErrors'] = vtk_abs_kappa_2_errors
        df['kappa2RelErrors'] = vtk_rel_kappa_2_errors
        df['mean_curvatureAbsErrors'] = vtk_abs_mean_curv_errors
        df['mean_curvatureRelErrors'] = vtk_rel_mean_curv_errors
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
        allowed_error = 0.3
        if not inverse:
            print("Testing the maximal principal curvature (kappa_1)...")
            for error in rel_kappa_1_errors:
                assert error <= allowed_error
        else:  # inverse
            print("Testing the minimal principal curvature (kappa_2)...")
            for error in rel_kappa_2_errors:
                assert error <= allowed_error


# @pytest.mark.parametrize("radius_hit", range(5, 11))
# @pytest.mark.parametrize(
#     "radius,inverse,voxel,ico,methods,area2,runtimes,vertex_based", [
#         # (10, False, False, 1280, ['VV'], True, '', False),  # icosahedron
#         (10, False, True, 0, ['VV'], False, '', False),  # voxel, RVV
#         (10, False, True, 0, ['VV'], True, '', False),  # voxel, AVV
#         (10, False, True, 0, ['SSVV'], True, '', False),  # voxel, 'SSVV'
#         # "{}sphere/voxel/files4plotting/bin_spheres_runtimes.csv".format(FOLD)
#         (10, False, False, 0, ['VV'], False, '', False),  # smooth, RVV
#         (10, False, False, 0, ['VV'], True, '', False),  # smooth, AVV
#         (10, False, False, 0, ['SSVV'], True, '', False),  # smooth, SSVV
#         # (10, True, False, 0, ['VV', 'SSVV'], False, '', False), smooth inverse
#     ])
@pytest.mark.parametrize(
    "radius,radius_hit,inverse,voxel,ico,methods,area2,runtimes,vertex_based", [
        # smooth, radius=10:
        # (10, 10, False, False, 0, ['VV'], True, '', False),  # AVV
        # (10, 10, False, False, 0, ['VV'], False, '', False),  # RVV
        # (10, 9, False, False, 0, ['SSVV'], False, '', False),  # SSVV
        # RVV and SSVV, vertex:
        # (10, 9, False, False, 0, ['SSVV', 'VV'], False, '', True),
        # smooth, radius=20:
        (20, 10, False, False, 0, ['VV'], True, '', False),  # AVV
        (20, 10, False, False, 0, ['VV'], False, '', False),  # RVV
        (20, 9, False, False, 0, ['SSVV'], False, '', False),  # SSVV
        # voxel, radius=10:
        # (10, 9, False, True, 0, ['VV'], True, '', False),  # AVV
        # (10, 10, False, True, 0, ['VV'], False, '', False),  # RVV
        # (10, 8, False, True, 0, ['SSVV'], False, '', False),  # SSVV
        # (10, 10, False, True, 0, ['VV'], False, '', True),  # RVV, vertex
        # (10, 8, False, True, 0, ['SSVV'], False, '', True),  # SSVV, vertex
        # voxel, radius=20:
        # (20, 10, False, True, 0, ['VV'], True, '', False),
        # pytest.param(20, 8, False, True, 0, ['SSVV'], True, '', False,
        #              marks=pytest.mark.xfail(reason="too high errors")),
        # voxel, radius=20, radius_hit=18, SSVV & AVV:
        # (20, 18, False, True, 0, ['SSVV', 'VV'], True, '', False),
        # voxel, radius=30:
        (30, 9, False, True, 0, ['VV'], True, '', False),  # AVV
        (30, 10, False, True, 0, ['VV'], False, '', False),  # RVV
        pytest.param(30, 8, False, True, 0, ['SSVV'], False, '', False,  # SSVV
                     marks=pytest.mark.xfail(reason="too high errors")),
        # voxel, radius=30, radius_hit=18
        (30, 28, False, True, 0, ['VV'], True, '', False),  # AVV
        (30, 28, False, True, 0, ['SSVV', 'VV'], False, '', False), # SSVV & RVV
    ])
def test_sphere_curvatures(
        radius, radius_hit, inverse, methods, area2, voxel, ico, runtimes,
        vertex_based, res=0, noise=0, save_areas=False,
        page_curvature_formula=False, full_dist_map=False, cores=4):
    """
    Runs all the steps needed to calculate curvatures for a test sphere with a
    given radius. Tests whether the curvatures are correctly estimated using
    normal vector voting (VV), VV combined with curvature tensor voting (SSVV).
    kappa_1 = kappa_2 = 1/r; allowing error of +-30%.

    Args:
        radius (int): radius of the sphere
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        inverse (boolean): if True, the sphere will have normals pointing
            outwards (negative curvature), else the other way around
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'SSVV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        area2 (boolean): if True (default), votes are weighted by triangle area
            also in the second step (principle directions and curvatures
            estimation; not possible if vertex_based is True)
        voxel (boolean): if True, a voxel sphere is generated (ignoring the
            options ico, res and noise)
        ico (int): if > 0 and res=0, an icosahedron with so many faces is used
            (1280 faces with radius 1 or 10 are available so far)
        runtimes (str): if given, runtimes and some parameters are added to
            this file (otherwise empty string '')
        vertex_based (boolean): if True, curvature is calculated per triangle
            vertex instead of triangle center.
        res (int, optional): if > 0 determines how many longitude and
            latitude stripes the UV sphere from vtkSphereSource has, the
            surface is triangulated; If 0 (default) and ico=0, first a
            smooth sphere mask is generated and then surface using
            vtkMarchingCubes
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 10), the noise
            is added on triangle vertex coordinates in its normal direction
        save_areas (boolean, optional): if True (default False), also mesh
            triangle ares will be saved to a file (not possible if vertex_based
            is True)
        page_curvature_formula (boolean, optional): if True (default False)
            normal curvature formula from Page et al. is used for VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map
            is calculated for each vertex (default)
        cores (int): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """
    if voxel:
        fold = '{}sphere/voxel/'.format(FOLD)
    else:
        if res > 0:  # UV sphere with this longitude and latitude res. is used
            fold = '{}sphere/res{}_noise{}/'.format(FOLD, res, noise)
        elif ico > 0:  # icosahedron sphere with so many faces is used
            fold = '{}sphere/ico{}_noise{}/'.format(FOLD, ico, noise)
        else:  # a sphere generated by a smooth mask is used
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
    if vertex_based:
        area2 = False
        save_areas = False
        vertex_based_str = "_vertex_based"
    else:
        vertex_based_str = ""
    base_filename = "{}{}sphere_r{}{}".format(
        files_fold, inverse_str, radius, vertex_based_str)
    VTK_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.VV_rh{}_normals.gt'.format(
        base_filename, radius_hit)
    methods_str = ""
    for method in methods:
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        methods_str += (method + '_')
    log_file = '{}.{}rh{}.log'.format(base_filename, methods_str, radius_hit)
    sys.stdout = open(log_file, 'w')

    if inverse:
        print("\n*** Generating a surface and a graph for an inverse "
              "sphere with radius {} and {}% noise***".format(radius, noise))
    else:
        print("\n*** Generating a surface and a graph for a sphere with "
              "radius {} and {}% noise***".format(radius, noise))
    # If the .vtp file with the test surface does not exist, create it:
    if not os.path.isfile(surf_file):
        sg = SphereGenerator()
        if voxel:
            sphere = sg.generate_voxel_sphere_surface(radius)
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
            else:  # generate a sphere surface from a smooth mask
                sphere = sg.generate_gauss_sphere_surface(radius)
                if noise > 0:
                    sphere = add_gaussian_noise_to_surface(sphere,
                                                           percent=noise)
                io.save_vtp(sphere, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, sg = surface_to_graph(surf_file, inverse=inverse,
                                vertex_based=vertex_based)

    # Running the modified Normal Vector Voting algorithm:
    if runtimes != '' and not os.path.isfile(runtimes):
        with open(runtimes, 'w') as f:
            f.write("num_v;radius_hit;g_max;avg_num_neighbors;cores;"
                    "duration1;method;duration2\n")
    method_sg_surf_dict = normals_directions_and_curvature_estimation(
        sg, radius_hit, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        runtimes=runtimes, graph_file=temp_normals_graph_file)
    # Remove the normals graph file, so the next test will start anew
    remove(temp_normals_graph_file)

    # Ground truth principal curvatures
    true_curvature = 1.0 / radius
    if inverse:
        true_curvature *= -1

    for method in method_sg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (sg, surf) = method_sg_surf_dict[method]
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        surf_file = '{}.{}_rh{}.vtp'.format(base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)

        # Evaluating each method:
        print("\nEvaluating {}...".format(method))
        eval_file = '{}.{}_rh{}.csv'.format(base_filename, method, radius_hit)

        # Getting estimated principal curvatures from the output graph:
        kappa_1 = sg.get_vertex_property_array("kappa_1")
        kappa_2 = sg.get_vertex_property_array("kappa_2")
        vtk_kappa_1 = sg.get_vertex_property_array("max_curvature")
        vtk_kappa_2 = sg.get_vertex_property_array("min_curvature")

        # Calculating errors of the principal curvatures:
        abs_kappa_1_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), kappa_1))
        abs_kappa_2_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), kappa_2))
        rel_kappa_1_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), kappa_1))
        rel_kappa_2_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), kappa_2))
        vtk_abs_kappa_1_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), vtk_kappa_1))
        vtk_abs_kappa_2_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), vtk_kappa_2))
        vtk_rel_kappa_1_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), vtk_kappa_1))
        vtk_rel_kappa_2_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), vtk_kappa_2))

        # Calculating estimated and VTK mean curvatures and their errors:
        mean_curv = [(k_1 + k_2) / 2 for k_1, k_2 in zip(kappa_1, kappa_2)]
        vtk_mean_curv = [(k_1 + k_2) / 2 for k_1, k_2 in zip(
            vtk_kappa_1, vtk_kappa_2)]
        abs_mean_curv_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), mean_curv))
        rel_mean_curv_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), mean_curv))
        vtk_abs_mean_curv_errors = np.array(map(
            lambda x: absolute_error_scalar(true_curvature, x), vtk_mean_curv))
        vtk_rel_mean_curv_errors = np.array(map(
            lambda x: relative_error_scalar(true_curvature, x), vtk_mean_curv))

        # Writing all the curvature values and errors into a csv file:
        df = pd.DataFrame()
        df['kappa1'] = kappa_1
        df['kappa1AbsErrors'] = abs_kappa_1_errors
        df['kappa1RelErrors'] = rel_kappa_1_errors
        df['kappa2'] = kappa_2
        df['kappa2AbsErrors'] = abs_kappa_2_errors
        df['kappa2RelErrors'] = rel_kappa_2_errors
        df['mean_curvatureAbsErrors'] = abs_mean_curv_errors
        df['mean_curvatureRelErrors'] = rel_mean_curv_errors
        if save_areas:
            triangle_areas = sg.get_vertex_property_array("area")
            df['triangleAreas'] = triangle_areas
        df.to_csv(eval_file, sep=';')
        # The same steps for VTK:
        df = pd.DataFrame()
        df['kappa1'] = vtk_kappa_1
        df['kappa1AbsErrors'] = vtk_abs_kappa_1_errors
        df['kappa1RelErrors'] = vtk_rel_kappa_1_errors
        df['kappa2'] = vtk_kappa_2
        df['kappa2AbsErrors'] = vtk_abs_kappa_2_errors
        df['kappa2RelErrors'] = vtk_rel_kappa_2_errors
        df['mean_curvatureAbsErrors'] = vtk_abs_mean_curv_errors
        df['mean_curvatureRelErrors'] = vtk_rel_mean_curv_errors
        df.to_csv(VTK_eval_file, sep=';')

        # Asserting that all values of both principal curvatures are close
        # to the true value, allowing error of +-30%:
        allowed_error = 0.3
        print("Testing the maximal principal curvature (kappa_1)...")
        for error in rel_kappa_1_errors:
            assert error <= allowed_error
        print("Testing the minimal principal curvature (kappa_2)...")
        for error in rel_kappa_2_errors:
            assert error <= allowed_error


# @pytest.mark.parametrize("radius_hit", range(4, 11))
# @pytest.mark.parametrize(
#     "rr,csr,subdivisions,methods,area2,voxel,runtimes,cores,vertex_based", [
#         # (25, 10, 0, ['VV'], True, True, '', 4, False),  # AVV, voxel
#         # (25, 10, 0, ['VV'], False, True, '', 4, False),  # RVV, voxel
#         # (25, 10, 0, ['SSVV'], False, True, '', 4, False),  # SSVV, voxel
#         (25, 10, 0, ['VV'], True, False, '', 4, False),  # AVV, smooth
#         # (25, 10, 0, ['VV'], False, False, '', 4, False),  # RVV, smooth
#         # (25, 10, 0, ['SSVV'], False, False, '', 4, False),  # SSVV, smooth
#     ])
@pytest.mark.parametrize("cores", range(1, 21))
@pytest.mark.parametrize(
    "rr,csr,subdivisions,radius_hit,methods,area2,voxel,vertex_based,runtimes",
    [
        (25, 10, 0, 9, ['VV'], True, False, False,
         "{}torus/noise0/files4plotting/"
         "torus_rr25_csr10_AVV_runtimes_joined_pass1_and_2.csv".format(FOLD))
    ])
@pytest.mark.xfail(reason="too high errors")
# @pytest.mark.parametrize(
#     "rr,csr,subdivisions,radius_hit,methods,area2,voxel,runtimes,cores,vertex_based", [
#         (25, 10, 0, 9, ['VV'], False, False, '', 4, False),  # RVV
#         (25, 10, 0, 9, ['VV'], True, False, '', 4, False),  # AVV
#         (25, 10, 0, 5, ['SSVV'], False, False, '', 4, False),
#         # (25, 10, 0, 5, ['SSVV'], False, False, '', 4, True),  # SSVV, vertex
#         # (25, 10, 0, 9, ['VV'], False, False, '', 4, True),  # RVV, vertex
#         # (25, 10, 100, 5, ['SSVV'], False, False, '', 4, True),  # SSVV, vertex, finer
#         # (25, 10, 100, 9, ['VV'], False, False, '', 4, True),  # RVV, vertex, finer
#     ])
def test_torus_directions_curvatures(
        rr, csr, subdivisions, radius_hit, methods, area2, voxel, runtimes,
        cores, vertex_based, page_curvature_formula=False, full_dist_map=False):
    """
    Runs all the steps needed to calculate curvatures for a test torus
    with given radii using normal vector voting (VV) or VV combined with
    curvature tensor voting (SSVV).
    Allowing error of +-30%.

    Args:
        rr (int): ring radius of the torus
        csr (int): cross-section radius of the torus
        subdivisions (int): number of subdivisions in all three
            dimensions, if 0, default subdivisions are used
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'SSVV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        area2 (boolean): if True (default), votes are weighted by triangle area
            also in the second step (principle directions and curvatures
            estimation; not possible if vertex_based is True)
        voxel (boolean): if True, a voxel torus is generated (ignoring
            subdivisions parameter)
        runtimes (str): if given, runtimes and some parameters are added to
            this file (otherwise '')
        cores (int): number of cores to run VV in parallel
        vertex_based (boolean): if True, curvature is calculated per triangle
            vertex instead of triangle center.
        page_curvature_formula (boolean, optional): if True (default False)
            normal curvature formula from Page et al. is used for VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map
            is calculated for each vertex (default)

    Notes:
        * csr should be much smaller than rr (csr < rr - csr).

    Returns:
        None
    """
    if voxel:
        fold = '{}torus/voxel/'.format(FOLD)
    else:
        fold = '{}torus/noise0/'.format(FOLD)
    if not os.path.exists(fold):
        os.makedirs(fold)
    if subdivisions > 0:
        subdivisions_str = "_subdivisions{}".format(subdivisions)
    else:
        subdivisions_str = ""
    surf_filebase = '{}torus_rr{}_csr{}{}'.format(
        fold, rr, csr, subdivisions_str)
    surf_file = '{}.surface.vtp'.format(surf_filebase)
    files_fold = '{}files4plotting/'.format(fold)
    if not os.path.exists(files_fold):
        os.makedirs(files_fold)
    if vertex_based:
        area2 = False
        vertex_based_str = "_vertex_based"
    else:
        vertex_based_str = ""
    base_filename = "{}torus_rr{}_csr{}{}{}".format(
        files_fold, rr, csr, subdivisions_str, vertex_based_str)
    VTK_eval_file = '{}.VTK.csv'.format(base_filename)
    temp_normals_graph_file = '{}.VV_rh{}_normals.gt'.format(
        base_filename, radius_hit)
    methods_str = ""
    for method in methods:
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        methods_str += (method + '_')
    log_file = '{}.{}rh{}.log'.format(base_filename, methods_str, radius_hit)
    sys.stdout = open(log_file, 'w')

    print("\n*** Generating a surface and a graph for a torus with ring "
          "radius {} and cross-section radius {}***".format(rr, csr))
    # If the .vtp file with the test surface does not exist, create it:
    if not os.path.isfile(surf_file):
        sgen = SaddleGenerator()
        if voxel:
            torus = sgen.generate_voxel_torus_surface(rr, csr)
        else:
            torus = sgen.generate_parametric_torus(rr, csr, subdivisions)
        io.save_vtp(torus, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, sg = surface_to_graph(surf_file, inverse=False,
                                vertex_based=vertex_based)

    # Ground-truth principal curvatures and directions
    # Vertex properties for storing the true maximal and minimal curvatures
    # and the their directions of the corresponding triangle:
    sg.graph.vp.true_kappa_1 = sg.graph.new_vertex_property("float")
    sg.graph.vp.true_kappa_2 = sg.graph.new_vertex_property("float")
    sg.graph.vp.true_T_1 = sg.graph.new_vertex_property("vector<float>")
    sg.graph.vp.true_T_2 = sg.graph.new_vertex_property("vector<float>")

    if voxel:
        # Map the noisy coordinates to coordinates on smooth torus surface:
        # generate the smooth torus surface
        sgen = SaddleGenerator()
        smooth_torus = sgen.generate_parametric_torus(rr, csr, subdivisions=0)
        # make point locator on the smooth torus surface
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(smooth_torus)
        pointLocator.SetNumberOfPointsPerBucket(10)
        pointLocator.BuildLocator()

    # Calculate and fill the properties
    true_kappa_1 = 1.0 / csr  # constant for the whole torus surface
    xyz = sg.graph.vp.xyz
    for v in sg.graph.vertices():
        x, y, z = xyz[v]  # coordinates of graph vertex v
        if voxel:
            # correct the coordinates to have (0,0,0) in the middle
            x = x - (rr+csr)
            y = y - (rr+csr)
            z = z - csr
            xyz[v] = [x, y, z]
            # find the closest point on the smooth surface
            closest_point_id = pointLocator.FindClosestPoint([x, y, z])
            closest_true_xyz = np.zeros(shape=3)
            smooth_torus.GetPoint(closest_point_id, closest_true_xyz)
            true_kappa_2, true_T_1, true_T_2 = torus_curvatures_and_directions(
                rr, csr, *closest_true_xyz)
        else:
            true_kappa_2, true_T_1, true_T_2 = torus_curvatures_and_directions(
                rr, csr, x, y, z)
        sg.graph.vp.true_kappa_1[v] = true_kappa_1
        sg.graph.vp.true_kappa_2[v] = true_kappa_2
        sg.graph.vp.true_T_1[v] = true_T_1
        sg.graph.vp.true_T_2[v] = true_T_2

    # Getting the true principal directions and principal curvatures:
    pos = [0, 1, 2]  # vector-property value positions
    # The shape is (3, <num_vertices>) - have to transpose to group the
    # respective x, y, z components to sub-arrays
    true_T_1 = np.transpose(
        sg.graph.vertex_properties["true_T_1"].get_2d_array(pos))
    true_T_2 = np.transpose(
        sg.graph.vertex_properties["true_T_2"].get_2d_array(pos))
    true_kappa_1 = sg.get_vertex_property_array("true_kappa_1")
    true_kappa_2 = sg.get_vertex_property_array("true_kappa_2")

    # Running the modified Normal Vector Voting algorithm:
    if runtimes != '' and not os.path.isfile(runtimes):
        with open(runtimes, 'w') as f:
            f.write("num_v;radius_hit;g_max;avg_num_neighbors;cores;duration1;"
                    "method;duration2\n")
    method_sg_surf_dict = normals_directions_and_curvature_estimation(
        sg, radius_hit, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        runtimes=runtimes, graph_file=temp_normals_graph_file)
    # Remove the normals graph file, so the next test will start anew
    remove(temp_normals_graph_file)

    for method in method_sg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (sg, surf) = method_sg_surf_dict[method]
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        surf_file = '{}.{}_rh{}.vtp'.format(base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)

        # Evaluating each method:
        print("\nEvaluating {}...".format(method))
        eval_file = '{}.{}_rh{}.csv'.format(base_filename, method, radius_hit)

        # Getting the estimated and true principal directions:
        T_1 = np.transpose(sg.graph.vertex_properties["T_1"].get_2d_array(pos))
        T_2 = np.transpose(sg.graph.vertex_properties["T_2"].get_2d_array(pos))

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
        kappa_1 = sg.get_vertex_property_array("kappa_1")
        kappa_2 = sg.get_vertex_property_array("kappa_2")
        vtk_kappa_1 = sg.get_vertex_property_array("max_curvature")
        vtk_kappa_2 = sg.get_vertex_property_array("min_curvature")

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

        # Calculating estimated and VTK mean curvatures and their errors:
        true_mean_curv = (true_kappa_1 + true_kappa_2) / 2.0
        mean_curv = [(k_1 + k_2) / 2 for k_1, k_2 in zip(kappa_1, kappa_2)]
        vtk_mean_curv = [(k_1 + k_2) / 2 for k_1, k_2 in zip(
            vtk_kappa_1, vtk_kappa_2)]
        abs_mean_curv_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y),
            true_mean_curv, mean_curv))
        rel_mean_curv_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y),
            true_mean_curv, mean_curv))
        vtk_abs_mean_curv_errors = np.array(map(
            lambda x, y: absolute_error_scalar(x, y),
            true_mean_curv, vtk_mean_curv))
        vtk_rel_mean_curv_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y),
            true_mean_curv, vtk_mean_curv))

        # Writing all the VV curvature values and errors into a csv file:
        df = pd.DataFrame()
        df['true_kappa2'] = true_kappa_2
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
        df['mean_curvatureAbsErrors'] = abs_mean_curv_errors
        df['mean_curvatureRelErrors'] = rel_mean_curv_errors
        df.to_csv(eval_file, sep=';')
        # The same for VTK:
        df = pd.DataFrame()
        df['true_kappa2'] = true_kappa_2
        df['kappa1'] = vtk_kappa_1
        df['kappa2'] = vtk_kappa_2
        df['kappa1AbsErrors'] = vtk_abs_kappa_1_errors
        df['kappa1RelErrors'] = vtk_rel_kappa_1_errors
        df['kappa2AbsErrors'] = vtk_abs_kappa_2_errors
        df['kappa2RelErrors'] = vtk_rel_kappa_2_errors
        df['mean_curvatureAbsErrors'] = vtk_abs_mean_curv_errors
        df['mean_curvatureRelErrors'] = vtk_rel_mean_curv_errors
        df.to_csv(VTK_eval_file, sep=';')

        # Asserting that all estimated T_1 and T_2 vectors are close to the
        # corresponding true vector, allowing error of 30%:
        allowed_error = 0.3
        print("Testing the maximal principal directions (T_1)...")
        for error in T_1_errors:
            assert error <= allowed_error
        print("Testing the minimal principal directions (T_2)...")
        for error in T_2_errors:
            assert error <= allowed_error
        # Asserting that all estimated kappa_1 and kappa_2 values are close
        # to the corresponding true values, allowing error of 30% from the
        # true value (the maximal absolute value in case of kappa_2, because
        # it can be 0 or negative):
        print("Testing the maximal principal curvature (kappa_1)...")
        for error in rel_kappa_1_errors:
            assert error <= allowed_error
        print("Testing the minimal principal curvature (kappa_2)...")
        for error in rel_kappa_2_errors:
            assert error <= allowed_error


def run_cylinder_with_creases(
        radius_hit, inverse, methods, area2, vertex_based, epsilon, eta,
        radius=10, h=25, res=10, subdivisions=3, max_edge=0, max_area=0,
        decimate=0, noise=0,
        page_curvature_formula=False, full_dist_map=False, cores=4):
    """
    Tests whether surface is correctly classified for a cylinder surface with
    circular planes and that neighborhood search does not go over the creases
    for curvature estimation step, using normal vector voting (VV) or VV
    combined with curvature tensor voting (SSVV).

    Args:
        radius_hit (float): radius in length unit of the graph, here voxels;
            it should be chosen to correspond to radius of smallest features
            of interest on the surface
        inverse (boolean): if True, the cylinder will have normals pointing
            outwards (negative curvature), else the other way around
        methods (list): tells which method(s) should be used: 'VV'
            for normal vector voting or 'SSVV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation; not possible if vertex_based is True)
        vertex_based (boolean): if True, curvature is calculated per triangle
            vertex instead of triangle center.
        epsilon (float): parameter of Normal Vector Voting algorithm influencing
            the number of triangles classified as "crease junction" (class 2)
        eta (float): parameter of Normal Vector Voting algorithm influencing the
            number of triangles classified as "crease junction" (class 2) and
            "no preferred orientation" (class 3)
        radius (int, optional): cylinder radius in voxels (>0, default 10)
        h (int, optional): cylinder height in voxels (>0, default 25)
        res (int, optional): determines how many stripes around both
            approximate circles (and then triangles) the cylinder has, the
            surface is generated using vtkCylinderSource (>0, default 10)
        subdivisions (int, optional): if > 0 (default 3) vtkLinearSubdivi-
            sionFilter is applied with this number of subdivisions
        max_edge (float, optional): if > 0 (default 0) vtkAdaptiveSubdivi-
            sionFilter is applied with this maximal triangle edge length
        max_area (float, optional):  if > 0 (default 0) vtkAdaptiveSubdivi-
            sionFilter is applied with this maximal triangle area
        decimate (float, optional): if > 0 (default 0) vtkDecimatePro is
            applied with this target reduction (< 1)
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 0), the noise
            is added on triangle vertex coordinates in its normal direction
        page_curvature_formula (boolean, optional): if True (default False)
            normal curvature formula from Page at al. is used for VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map
            is calculated for each vertex (default)
        cores (int): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """
    fold = '{}cylinder/res{}_noise{}/'.format(FOLD, res, noise)
    if not os.path.exists(fold):
        os.makedirs(fold)

    if subdivisions > 0:
        subdiv_str = "_linear_subdiv{}".format(subdivisions)
    else:
        subdiv_str = ""
    if max_edge > 0:
        max_edge_str = "_max_edge{}".format(max_edge)
    else:
        max_edge_str = ""
    if max_area > 0:
        max_area_str = "_max_area{}".format(max_area)
    else:
        max_area_str = ""
    if decimate > 0:
        decim_str = "_decim{}".format(decimate)
    else:
        decim_str = ""

    surf_filebase = '{}cylinder_r{}_h{}{}{}{}{}'.format(
        fold, radius, h, subdiv_str, max_edge_str, max_area_str, decim_str)
    surf_file = '{}.surface.vtp'.format(surf_filebase)
    files_fold = '{}files4plotting/'.format(fold)
    if not os.path.exists(files_fold):
        os.makedirs(files_fold)
    if inverse:
        inverse_str = "inverse_"
    else:
        inverse_str = ""
    if vertex_based:
        area2 = False
        vertex_based_str = "_vertex_based"
    else:
        vertex_based_str = ""
    base_filename = ("{}{}cylinder_r{}_h{}{}{}{}{}{}_epsilon{}_eta{}".
        format(files_fold, inverse_str, radius, h, subdiv_str, max_edge_str,
               max_area_str, decim_str, vertex_based_str, epsilon, eta))
    temp_normals_graph_file = '{}.VV_rh{}_normals.gt'.format(
        base_filename, radius_hit)
    methods_str = ""
    for method in methods:
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        methods_str += (method + '_')
    log_file = '{}.{}rh{}.log'.format(base_filename, methods_str, radius_hit)
    sys.stdout = open(log_file, 'w')

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
        cylinder = cg.generate_cylinder_surface(
            radius, h, res, subdivisions, max_edge, max_area, decimate,
            verbose=True)
        if noise > 0:
            cylinder = add_gaussian_noise_to_surface(cylinder, percent=noise)
        io.save_vtp(cylinder, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    surf, sg = surface_to_graph(surf_file, inverse=inverse,
                                vertex_based=vertex_based)

    # Running the modified Normal Vector Voting algorithm:
    method_sg_surf_dict = normals_directions_and_curvature_estimation(
        sg, radius_hit, epsilon, eta, methods=methods,
        page_curvature_formula=page_curvature_formula,
        full_dist_map=full_dist_map, area2=area2, poly_surf=surf, cores=cores,
        graph_file=temp_normals_graph_file)
    # Remove the normals graph file, so the next test will start anew
    remove(temp_normals_graph_file)

    for method in method_sg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (sg, surf) = method_sg_surf_dict[method]
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        surf_file = '{}.{}_rh{}.vtp'.format(
            base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)


# py.test -n 4   # test on multiple CPUs

if __name__ == "__main__":
    res = 10
    eta = 0
    epsilon = 5
    run_cylinder_with_creases(
        radius_hit=5, res=res, inverse=False, methods=["VV"], area2=False,
        vertex_based=True, epsilon=epsilon, eta=eta, cores=4)
    run_cylinder_with_creases(
        radius_hit=5, res=res, inverse=False, methods=["VV"], area2=True,
        vertex_based=False, epsilon=epsilon, eta=eta, cores=4)
    # test_plane_normals(
    #     half_size=10, radius_hit=4, res=10, noise=10, vertex_based=True,
    #     cores=1)
    #     fold = '{}sphere/voxel/'.format(FOLD)
#     stats_file = '{}sphere_r10.AVV_rh9.stats'.format(fold)
#     cProfile.run('test_sphere_curvatures(radius=10, radius_hit=9, '
#                  'inverse=False, voxel=True, ico=0, methods=[\'VV\'], '
#                  'runtimes='', cores=1)', stats_file)
#
#     p = pstats.Stats(stats_file)
#     # what algorithms are taking time:
#     # p.strip_dirs().sort_stats('cumulative').print_stats(10)
#     # what functions were looping a lot, and taking a lot of time:
#     p.strip_dirs().sort_stats('time').print_stats(20)
