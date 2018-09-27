import time
import numpy as np
import math
from graph_tool import load_graph
from graph_tool.topology import shortest_distance
import pathos.pools as pp
from functools import partial
from os import remove

from surface_graphs import TriangleGraph

"""
Contains a function implementing the normal vector voting algorithm (Page et
al., 2002) with adaptation.

The main differences to the original method are usage of a graph (TriangleGraph
object) representing the underlying surface and approximating geodesic distances
by shortest distances along the graph edges. Uses several methods of the
TriangleGraph class in the surface_graphs module, which implement triangle-wise
operations of the algorithm.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry),
date: 2017-06-17
"""

__author__ = 'kalemanov'


def vector_voting(tg, radius_hit, epsilon=0, eta=0, exclude_borders=True,
                  page_curvature_formula=False):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, principle curvatures and directions for a surface using its
    triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        exclude_borders (boolean, optional): if True (default), principle
            curvatures and directions are not estimated for triangles at surface
            borders
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used (see
            collecting_curvature_votes)
    Returns:
        the surface of triangles with classified orientation and estimated
        normals or tangents, principle curvatures and directions (vtkPolyData)

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print('\nPreparing for running modified Vector Voting...')

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    g_max = math.pi * radius_hit / 2
    print("radius_hit = {}".format(radius_hit))
    print("g_max = {}".format(g_max))

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A_max = np.max(A)
    print("Maximal triangle area = {}".format(A_max))

    # * Orientation classification parameters *
    print("epsilon = {}".format(epsilon))
    print("eta = {}".format(eta))

    # * Adding vertex properties to be filled in classifying_orientation *
    # vertex property storing the orientation class of the vertex: 1 if it
    # belongs to a surface patch, 2 if it belongs to a crease junction or 3 if
    # it doesn't have a preferred orientation:
    tg.graph.vp.orientation_class = tg.graph.new_vertex_property("int")
    # vertex property for storing the estimated normal of the corresponding
    # triangle (if the vertex belongs to class 1; scaled in nm):
    tg.graph.vp.N_v = tg.graph.new_vertex_property("vector<float>")
    # vertex property for storing the estimated tangent of the corresponding
    # triangle (if the vertex belongs to class 2; scaled in nm):
    tg.graph.vp.T_v = tg.graph.new_vertex_property("vector<float>")

    # * Adding vertex properties to be filled in estimate_curvature *
    # vertex properties for storing the estimated principal directions of the
    # maximal and minimal curvatures of the corresponding triangle (if the
    # vertex belongs to class 1; scaled in nm):
    tg.graph.vp.T_1 = tg.graph.new_vertex_property("vector<float>")
    tg.graph.vp.T_2 = tg.graph.new_vertex_property("vector<float>")
    # vertex properties for storing the estimated maximal and minimal curvatures
    # of the corresponding triangle (if the vertex belongs to class 1; scaled in
    # nm):
    tg.graph.vp.kappa_1 = tg.graph.new_vertex_property("float")
    tg.graph.vp.kappa_2 = tg.graph.new_vertex_property("float")
    # vertex property for storing the Gaussian curvature calculated from kappa_1
    # and kappa_2 at the corresponding triangle:
    tg.graph.vp.gauss_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the mean curvature calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.mean_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the shape index calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.shape_index_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the curvedness calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.curvedness_VV = tg.graph.new_vertex_property("float")

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Preparation took: {} min {} s'.format(minutes, seconds))

    # Main algorithm
    t_begin = time.time()

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print("\nRunning modified Vector Voting for all vertices...")

    print("\nFirst run: classifying orientation and estimating normals for "
          "surface patches and tangents for creases...")
    t_begin1 = time.time()

    collecting_normal_votes = tg.collecting_normal_votes
    all_num_neighbors = []
    classifying_orientation = tg.classifying_orientation
    classes_counts = {}
    for v in tg.graph.vertices():
        neighbor_idx_to_dist, V_v = collecting_normal_votes(
            v, g_max, A_max, sigma, verbose=False)
        all_num_neighbors.append(len(neighbor_idx_to_dist))
        class_v = classifying_orientation(v, V_v, epsilon=epsilon, eta=eta,
                                          verbose=False)
        try:
            classes_counts[class_v] += 1
        except KeyError:
            classes_counts[class_v] = 1

    # Printing out some numbers concerning the first run:
    avg_num_neighbors = np.mean(np.array(all_num_neighbors))
    print("Average number of geodesic neighbors for all vertices: {}".format(
        avg_num_neighbors))
    print("{} surface patches".format(classes_counts[1]))
    if 2 in classes_counts:
        print("{} crease junctions".format(classes_counts[2]))
    if 3 in classes_counts:
        print("{} no preferred orientation".format(classes_counts[3]))

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    minutes, seconds = divmod(duration1, 60)
    print('First run took: {} min {} s'.format(minutes, seconds))

    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or B_v is None"

    if exclude_borders:  # Exclude triangles at the borders from curvatures
        # calculation and remove them in the end

        t_begin0 = time.time()

        print('\nFinding triangles that are at surface borders and excluding '
              'them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Finding graph border took: {} min {} s'.format(minutes, seconds))

        condition1 += " and is_on_border[v] == 0"
        condition2 += " or is_on_border[v] == 1"

    print("\nSecond run: estimating principle curvatures and directions for "
          "surface patches...")
    t_begin2 = time.time()

    orientation_class = tg.graph.vp.orientation_class
    collecting_curvature_votes = tg.collecting_curvature_votes
    estimate_curvature = tg.estimate_curvature
    if exclude_borders:
        is_on_border = tg.graph.vp.is_on_border

    for i, v in enumerate(tg.graph.vertices()):
        # Estimate principal directions and curvatures (and calculate the
        # Gaussian and mean curvatures, shape index and curvedness) for
        # vertices belonging to a surface patch
        B_v = None  # initialization
        if eval(condition1):
            # None is returned if v does not have any neighbor belonging to
            # a surface patch
            B_v = collecting_curvature_votes(
                    v, g_max, sigma, verbose=False,
                    page_curvature_formula=page_curvature_formula)
        if B_v is not None:
            estimate_curvature(v, B_v, verbose=False)
        # For crease, no preferably oriented vertices, vertices on border or
        # vertices lacking neighbors, add placeholders to the corresponding
        # vertex properties
        if eval(condition2):
            tg.graph.vp.T_1[v] = np.zeros(shape=3)
            tg.graph.vp.T_2[v] = np.zeros(shape=3)
            tg.graph.vp.kappa_1[v] = 0
            tg.graph.vp.kappa_2[v] = 0
            tg.graph.vp.gauss_curvature_VV[v] = 0
            tg.graph.vp.mean_curvature_VV[v] = 0
            tg.graph.vp.shape_index_VV[v] = 0
            tg.graph.vp.curvedness_VV[v] = 0

    if exclude_borders:
        tg.find_graph_border(purge=True)

    t_end2 = time.time()
    duration2 = t_end2 - t_begin2
    minutes, seconds = divmod(duration2, 60)
    print('Second run took: {} min {} s'.format(minutes, seconds))

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Modified Vector Voting took: {} min {} s'.format(minutes, seconds))

    # Transforming the resulting graph to a surface with triangles:
    surface_VV = tg.graph_to_triangle_poly()
    return surface_VV


def vector_voting_curve_fitting(
        tg, radius_hit, num_points, epsilon=0, eta=0, exclude_borders=True,
        page_curvature_formula=False):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, principle directions and then principal curvatures using
    curve fitting in the principle directions for a surface using its triangle
    graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        radius_hit (float): radius in length unit of the graph, e.g. nanometers,
            for sampling surface points in tangent directions (distance between
            the points equals to graph's scale); it should be chosen to
            correspond to radius of smallest features of interest on the surface
        num_points (int): number of points to sample in each estimated
            principal direction in order to fit parabola and estimate curvature
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        exclude_borders (boolean, optional): if True (default), principle
            curvatures and directions are not estimated for triangles at surface
            borders
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used (see
            collecting_curvature_votes)

    Returns:
        the surface of triangles with classified orientation and estimated
        normals or tangents, principle curvatures and directions (vtkPolyData)

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print('\nPreparing for running modified Vector Voting...')

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    g_max = math.pi * radius_hit / 2
    print("radius_hit = {}".format(radius_hit))
    print("g_max = {}".format(g_max))

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A_max = np.max(A)
    print("Maximal triangle area = {}".format(A_max))

    # * Orientation classification parameters *
    print("epsilon = {}".format(epsilon))
    print("eta = {}".format(eta))

    # * Adding vertex properties to be filled in classifying_orientation *
    # vertex property storing the orientation class of the vertex: 1 if it
    # belongs to a surface patch, 2 if it belongs to a crease junction or 3 if
    # it doesn't have a preferred orientation:
    tg.graph.vp.orientation_class = tg.graph.new_vertex_property("int")
    # vertex property for storing the estimated normal of the corresponding
    # triangle (if the vertex belongs to class 1; scaled in nm):
    tg.graph.vp.N_v = tg.graph.new_vertex_property("vector<float>")
    # vertex property for storing the estimated tangent of the corresponding
    # triangle (if the vertex belongs to class 2; scaled in nm):
    tg.graph.vp.T_v = tg.graph.new_vertex_property("vector<float>")

    # * Adding vertex properties to be filled in estimate_directions_and_fit
    # curves *
    # vertex properties for storing the estimated principal directions of the
    # maximal and minimal curvatures of the corresponding triangle (if the
    # vertex belongs to class 1; scaled in nm):
    tg.graph.vp.T_1 = tg.graph.new_vertex_property("vector<float>")
    tg.graph.vp.T_2 = tg.graph.new_vertex_property("vector<float>")
    # vertex properties for storing curve fitting errors (variances) in maximal
    # and minimal principal directions at the vertex (belonging to class 1):
    tg.graph.vp.fit_error_1 = tg.graph.new_vertex_property("float")
    tg.graph.vp.fit_error_2 = tg.graph.new_vertex_property("float")
    # vertex properties for storing the estimated maximal and minimal curvatures
    # of the corresponding triangle (if the vertex belongs to class 1; scaled in
    # nm):
    tg.graph.vp.kappa_1 = tg.graph.new_vertex_property("float")
    tg.graph.vp.kappa_2 = tg.graph.new_vertex_property("float")
    # vertex property for storing the Gaussian curvature calculated from kappa_1
    # and kappa_2 at the corresponding triangle:
    tg.graph.vp.gauss_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the mean curvature calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.mean_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the shape index calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.shape_index_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the curvedness calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.curvedness_VV = tg.graph.new_vertex_property("float")

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Preparation took: {} min {} s'.format(minutes, seconds))

    # Main algorithm
    t_begin = time.time()

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print("\nRunning modified Vector Voting for all vertices...")

    print("\nFirst run: classifying orientation and estimating normals for "
          "surface patches and tangents for creases...")
    t_begin1 = time.time()

    collecting_normal_votes = tg.collecting_normal_votes
    all_num_neighbors = []
    classifying_orientation = tg.classifying_orientation
    classes_counts = {}
    for v in tg.graph.vertices():
        neighbor_idx_to_dist, V_v = collecting_normal_votes(
            v, g_max, A_max, sigma, verbose=False)
        all_num_neighbors.append(len(neighbor_idx_to_dist))
        class_v = classifying_orientation(v, V_v, epsilon=epsilon, eta=eta,
                                          verbose=False)
        try:
            classes_counts[class_v] += 1
        except KeyError:
            classes_counts[class_v] = 1

    # Printing out some numbers concerning the first run:
    avg_num_neighbors = np.mean(np.array(all_num_neighbors))
    print("Average number of geodesic neighbors for all vertices: {}".format(
        avg_num_neighbors))
    print("{} surface patches".format(classes_counts[1]))
    if 2 in classes_counts:
        print("{} crease junctions".format(classes_counts[2]))
    if 3 in classes_counts:
        print("{} no preferred orientation".format(classes_counts[3]))

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    minutes, seconds = divmod(duration1, 60)
    print('First run took: {} min {} s'.format(minutes, seconds))

    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or B_v is None"

    if exclude_borders:  # Exclude triangles at the borders from curvatures
        # calculation and remove them in the end

        t_begin0 = time.time()

        print('\nFinding triangles that are at surface borders and excluding '
              'them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Finding graph border took: {} min {} s'.format(minutes, seconds))

        condition1 += " and is_on_border[v] == 0"
        condition2 += " or is_on_border[v] == 1"

    print("\nSecond run: estimating principle directions and curvatures for "
          "surface patches...")
    t_begin2 = time.time()

    orientation_class = tg.graph.vp.orientation_class
    collecting_curvature_votes = tg.collecting_curvature_votes
    estimate_directions_and_fit_curves = tg.estimate_directions_and_fit_curves
    if exclude_borders:
        is_on_border = tg.graph.vp.is_on_border

    for i, v in enumerate(tg.graph.vertices()):
        # Estimate principal directions and curvatures (and calculate the
        # Gaussian and mean curvatures, shape index and curvedness) for
        # vertices belonging to a surface patch
        B_v = None  # initialization
        if eval(condition1):
            # None is returned if v does not have any neighbor belonging to
            # a surface patch
            B_v = collecting_curvature_votes(
                    v, g_max, sigma, verbose=False,
                    page_curvature_formula=page_curvature_formula)
        if B_v is not None:
            estimate_directions_and_fit_curves(v, B_v, radius_hit,
                                               num_points, verbose=False)
        # For crease, no preferably oriented vertices, vertices on border or
        # vertices lacking neighbors, add placeholders to the corresponding
        # vertex properties
        if eval(condition2):
            tg.graph.vp.T_1[v] = np.zeros(shape=3)
            tg.graph.vp.T_2[v] = np.zeros(shape=3)
            tg.graph.vp.fit_error_1[v] = 0
            tg.graph.vp.fit_error_2[v] = 0
            tg.graph.vp.kappa_1[v] = 0
            tg.graph.vp.kappa_2[v] = 0
            tg.graph.vp.gauss_curvature_VV[v] = 0
            tg.graph.vp.mean_curvature_VV[v] = 0
            tg.graph.vp.shape_index_VV[v] = 0
            tg.graph.vp.curvedness_VV[v] = 0

    if exclude_borders:
        tg.find_graph_border(purge=True)

    t_end2 = time.time()
    duration2 = t_end2 - t_begin2
    minutes, seconds = divmod(duration2, 60)
    print('Second run took: {} min {} s'.format(minutes, seconds))

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Modified Vector Voting took: {} min {} s'.format(minutes, seconds))

    # Transforming the resulting graph to a surface with triangles:
    surface_VV = tg.graph_to_triangle_poly()
    return surface_VV


def vector_curvature_tensor_voting(
        tg, poly_surf, radius_hit, epsilon=0, eta=0, exclude_borders=True):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, and the GenCurvVote algorithm to estimate principle curvatures
    and directions for a surface using its triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        poly_surf (vtkPolyData): surface from which the graph was generated,
            scaled to nm
        radius_hit (float): radius in length unit of the graph, e.g. nanometers,
            parameter of gen_curv_vote algorithm; it should be chosen to
            correspond to radius of smallest features of interest on the surface
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        exclude_borders (boolean, optional): if True (default), principle
            curvatures and directions are not estimated for triangles at surface
            borders

    Returns:
        the surface of triangles with classified orientation and estimated
        normals or tangents, principle curvatures and directions (vtkPolyData)

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print('\nPreparing for running modified Vector Voting...')

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    g_max = math.pi * radius_hit / 2
    print("radius_hit = {}".format(radius_hit))
    print("g_max = {}".format(g_max))

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A_max = np.max(A)
    print("Maximal triangle area = {}".format(A_max))

    # * Orientation classification parameters *
    print("epsilon = {}".format(epsilon))
    print("eta = {}".format(eta))

    # * Adding vertex properties to be filled in classifying_orientation *
    # vertex property storing the orientation class of the vertex: 1 if it
    # belongs to a surface patch, 2 if it belongs to a crease junction or 3 if
    # it doesn't have a preferred orientation:
    tg.graph.vp.orientation_class = tg.graph.new_vertex_property("int")
    # vertex property for storing the estimated normal of the corresponding
    # triangle (if the vertex belongs to class 1; scaled in nm):
    tg.graph.vp.N_v = tg.graph.new_vertex_property("vector<float>")
    # vertex property for storing the estimated tangent of the corresponding
    # triangle (if the vertex belongs to class 2; scaled in nm):
    tg.graph.vp.T_v = tg.graph.new_vertex_property("vector<float>")

    # * Adding vertex properties to be filled in estimate_directions_and_fit
    # curves *
    # vertex properties for storing the estimated principal directions of the
    # maximal and minimal curvatures of the corresponding triangle (if the
    # vertex belongs to class 1; scaled in nm):
    tg.graph.vp.T_1 = tg.graph.new_vertex_property("vector<float>")
    tg.graph.vp.T_2 = tg.graph.new_vertex_property("vector<float>")
    # vertex properties for storing the estimated maximal and minimal curvatures
    # of the corresponding triangle (if the vertex belongs to class 1; scaled in
    # nm):
    tg.graph.vp.kappa_1 = tg.graph.new_vertex_property("float")
    tg.graph.vp.kappa_2 = tg.graph.new_vertex_property("float")
    # vertex property for storing the Gaussian curvature calculated from kappa_1
    # and kappa_2 at the corresponding triangle:
    tg.graph.vp.gauss_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the mean curvature calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.mean_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the shape index calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.shape_index_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the curvedness calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.curvedness_VV = tg.graph.new_vertex_property("float")

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Preparation took: {} min {} s'.format(minutes, seconds))

    # Main algorithm
    t_begin = time.time()

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print("\nRunning modified Vector Voting for all vertices...")

    print("\nFirst run: classifying orientation and estimating normals for "
          "surface patches and tangents for creases...")
    t_begin1 = time.time()

    collecting_normal_votes = tg.collecting_normal_votes
    all_num_neighbors = []
    all_neighbor_idx_to_dist = []
    classifying_orientation = tg.classifying_orientation
    classes_counts = {}
    for v in tg.graph.vertices():
        neighbor_idx_to_dist, V_v = collecting_normal_votes(
            v, g_max, A_max, sigma, verbose=False)
        all_num_neighbors.append(len(neighbor_idx_to_dist))
        all_neighbor_idx_to_dist.append(neighbor_idx_to_dist)
        class_v = classifying_orientation(v, V_v, epsilon=epsilon, eta=eta,
                                          verbose=False)
        try:
            classes_counts[class_v] += 1
        except KeyError:
            classes_counts[class_v] = 1

    # Printing out some numbers concerning the first run:
    avg_num_geodesic_neighbors = (sum(x for x in all_num_neighbors) /
                                  len(all_num_neighbors))
    print("Average number of geodesic neighbors for all vertices: {}".format(
        avg_num_geodesic_neighbors))
    print("{} surface patches".format(classes_counts[1]))
    if 2 in classes_counts:
        print("{} crease junctions".format(classes_counts[2]))
    if 3 in classes_counts:
        print("{} no preferred orientation".format(classes_counts[3]))

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    minutes, seconds = divmod(duration1, 60)
    print('First run took: {} min {} s'.format(minutes, seconds))

    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or result is None"

    if exclude_borders:  # Exclude triangles at the borders from curvatures
        # calculation and remove them in the end

        t_begin0 = time.time()

        print('\nFinding triangles that are at surface borders and excluding '
              'them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Finding graph border took: {} min {} s'.format(minutes, seconds))

        condition1 += " and is_on_border[v] == 0"
        condition2 += " or is_on_border[v] == 1"

    print("\nSecond run: estimating principle directions and curvatures for "
          "surface patches...")
    t_begin2 = time.time()

    orientation_class = tg.graph.vp.orientation_class
    gen_curv_vote = tg.gen_curv_vote
    if exclude_borders:
        is_on_border = tg.graph.vp.is_on_border

    for i, v in enumerate(tg.graph.vertices()):
        # Estimate principal directions and curvatures (and calculate the
        # Gaussian and mean curvatures, shape index and curvedness) for
        # vertices belonging to a surface patch
        result = None  # initialization
        if eval(condition1):
            # None is returned if curvature at v cannot be estimated
            result = gen_curv_vote(poly_surf, v, radius_hit, verbose=False)
        # For crease, no preferably oriented vertices or if curvature could
        # not be estimated, add placeholders to the corresponding vertex
        # properties
        if eval(condition2):
            tg.graph.vp.T_1[v] = np.zeros(shape=3)
            tg.graph.vp.T_2[v] = np.zeros(shape=3)
            tg.graph.vp.kappa_1[v] = 0
            tg.graph.vp.kappa_2[v] = 0
            tg.graph.vp.gauss_curvature_VV[v] = 0
            tg.graph.vp.mean_curvature_VV[v] = 0
            tg.graph.vp.shape_index_VV[v] = 0
            tg.graph.vp.curvedness_VV[v] = 0

    if exclude_borders:
        tg.find_graph_border(purge=True)

    t_end2 = time.time()
    duration2 = t_end2 - t_begin2
    minutes, seconds = divmod(duration2, 60)
    print('Second run took: {} min {} s'.format(minutes, seconds))

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Modified Vector Voting took: {} min {} s'.format(minutes, seconds))

    # Transforming the resulting graph to a surface with triangles:
    surface_VV = tg.graph_to_triangle_poly()
    return surface_VV


def normals_directions_and_curvature_estimation(
        tg, radius_hit, epsilon=0, eta=0, exclude_borders=0,
        methods=['VV'], page_curvature_formula=False, num_points=None,
        full_dist_map=False, graph_file='temp.gt', area2=True,
        only_normals=False, poly_surf=None, cores=4, runtimes=None):
    """
    Runs the modified Normal Vector Voting algorithm (with different options for
    the second pass) to estimate surface orientation, principle curvatures and
    directions for a surface using its triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        exclude_borders (int, optional): if > 0, principle curvatures and
            directions are not estimated for triangles within this distance to
            surface borders (default 0)
        methods (list, optional): all methods to run in the second pass ('VV',
            'VVCF' and 'VCTV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV and VVCF
            (see collecting_curvature_votes)
        num_points (int, optional): number of points to sample in each estimated
            principal direction in order to fit parabola and estimate curvature
            (necessary is 'VVCF' is in methods list)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map is
            calculated later for each vertex (default)
        graph_file (string, optional): name for a temporary graph file
            after the first run of the algorithm (default 'temp.gt')
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation)
        only_normals (boolean, optional): if True (default False), only normals
            are estimated, without principal directions and curvatures, only the
            graph with the orientations class, normals or tangents is returned.
        poly_surf (vtkPolyData, optional): surface from which the graph was
            generated, scaled to nm (required only for VCTV, default None)
        cores (int, optional): number of cores to run VV in parallel (default 4)
        runtimes (str, optional): if given, runtimes and some parameters are
            added to this file (default None)
    Returns:
        a dictionary mapping the method name ('VV', 'VVCF' and 'VCTV') to the
        tuple of two elements: TriangleGraph graph and vtkPolyData surface of
        triangles with classified orientation and estimated normals or tangents,
        principle curvatures and directions (if only_normals is False)

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    t_begin = time.time()

    tg = normals_estimation(tg, radius_hit, epsilon, eta, full_dist_map,
                            cores=cores, runtimes=runtimes)

    preparation_for_curvature_estimation(tg, exclude_borders, graph_file)

    if only_normals is False:
        results = {}
        for method in methods:
            tg_curv, surface_curv = curvature_estimation(
                radius_hit, exclude_borders, graph_file, method,
                page_curvature_formula, num_points, area2, poly_surf=poly_surf,
                full_dist_map=full_dist_map, cores=cores, runtimes=runtimes)
            results[method] = (tg_curv, surface_curv)

        t_end = time.time()
        duration = t_end - t_begin
        minutes, seconds = divmod(duration, 60)
        print('Whole method took: {} min {} s'.format(minutes, seconds))
        return results


def normals_estimation(tg, radius_hit, epsilon=0, eta=0, full_dist_map=False,
                       cores=4, runtimes=None):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation (classification in surface patch with normal, crease junction
    with tangent or no preferred orientation) for a surface using its triangle
    graph (first part used by normals_directions_and_curvature_estimation).

    Adds the "orientation_class" (1-3), the estimated normal "N_v" (if class is
    1) and the estimated_tangent "T_v" (if class is 2) as vertex properties
    into the graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map is
            calculated later for each vertex (default)
        cores (int): number of cores to run VV (collecting_normal_votes and
            classifying_orientation) in parallel (default 8)
        runtimes (str): if given, runtimes and some parameters are added to
            this file (default None)

    Returns:
        tg (TriangleGraph): triangle graph with added properties

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    # Preparation (calculations that are the same for the whole graph)
    t_begin0 = time.time()
    print('\nPreparing for running modified Vector Voting...')

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    # g_max is 1/4 of circle circumference with radius=radius_hit
    g_max = math.pi * radius_hit / 2
    print("radius_hit = {}".format(radius_hit))
    print("g_max = {}".format(g_max))

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A = np.array(A)
    A_max = np.max(A)
    print("Maximal triangle area = {}".format(A_max))

    # * Orientation classification parameters *
    print("epsilon = {}".format(epsilon))
    print("eta = {}".format(eta))

    # * Adding vertex properties to be filled in classifying_orientation *
    # vertex property storing the orientation class of the vertex: 1 if it
    # belongs to a surface patch, 2 if it belongs to a crease junction or 3 if
    # it doesn't have a preferred orientation:
    tg.graph.vp.orientation_class = tg.graph.new_vertex_property("int")
    # vertex property for storing the estimated normal of the corresponding
    # triangle (if the vertex belongs to class 1; scaled in nm):
    tg.graph.vp.N_v = tg.graph.new_vertex_property("vector<float>")
    # vertex property for storing the estimated tangent of the corresponding
    # triangle (if the vertex belongs to class 2; scaled in nm):
    tg.graph.vp.T_v = tg.graph.new_vertex_property("vector<float>")

    t_end0 = time.time()
    duration0 = t_end0 - t_begin0
    minutes, seconds = divmod(duration0, 60)
    print('Preparation took: {} min {} s' .format(minutes, seconds))

    if full_dist_map is True:
        # * Distance map between all pairs of vertices *
        t_begin0 = time.time()
        print("\nCalculating the full distance map...")
        full_dist_map = shortest_distance(  # <class 'graph_tool.PropertyMap'>
            tg.graph, weights=tg.graph.ep.distance)
        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Calculation of the full distance map took: {} min {} s'.format(
            minutes, seconds))
    else:
        full_dist_map = None

    # First run of the main_javier algorithm

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print("\nRunning modified Vector Voting for all vertices...")

    print("\nFirst run: classifying orientation and estimating normals for "
          "surface patches and tangents for creases...")
    t_begin1 = time.time()

    collecting_normal_votes = tg.collecting_normal_votes
    classifying_orientation = tg.classifying_orientation
    num_v = tg.graph.num_vertices()
    print("number of vertices: {}".format(num_v))
    classes_counts = {}

    if cores > 1:  # parallel processing
        p = pp.ProcessPool(cores)
        print('Opened a pool with {} processes'.format(cores))

        # output is a list with 2 columns:
        # column 0 = num_neighbors (int)
        # column 1 = V_v (3x3 float array)
        # each row i is of vertex v, its index == i
        # V_v_list = p.map(partial(collecting_normal_votes, # if only V_v output
        results1_list = p.map(partial(collecting_normal_votes,
                                      g_max=g_max, A_max=A_max, sigma=sigma,
                                      full_dist_map=full_dist_map),
                              range(num_v))  # a list of vertex v indices
        results1_array = np.array(results1_list, dtype=object)
        # Calculating average neighbors number:
        num_neighbors_array = results1_array[:, 0]
        avg_num_neighbors = np.mean(num_neighbors_array)
        # Input of the next parallel calculation:
        V_v_array = results1_array[:, 1]

        # output is a list with 3 columns:
        # column 0 = orientation class of the vertex (int)
        # column 1 = N_v (3x1 float array, zeros if class=2 or 3)
        # column 2 = T_v (3x1 float array, zeros if class=1 or 3)
        # each row i is of vertex v, its index == i
        results2_list = p.map(partial(classifying_orientation,
                                      epsilon=epsilon, eta=eta),
                              range(num_v), V_v_array)
        #                      range(num_v), V_v_list)  # if only V_v output
        p.close()
        p.clear()
        results2_array = np.array(results2_list, dtype=object)
        class_v_array = results2_array[:, 0]
        N_v_array = results2_array[:, 1]
        T_v_array = results2_array[:, 2]
        # Adding the properties to the graph tg and counting classes:
        for i in range(num_v):
            v = tg.graph.vertex(i)
            tg.graph.vp.orientation_class[v] = class_v_array[i]
            tg.graph.vp.N_v[v] = N_v_array[i]
            tg.graph.vp.T_v[v] = T_v_array[i]
            try:
                classes_counts[class_v_array[i]] += 1
            except KeyError:
                classes_counts[class_v_array[i]] = 1

    else:  # cores == 1, sequential processing
        sum_num_neighbors = 0
        for i in range(num_v):
            num_neighbors, V_v = collecting_normal_votes(
                i, g_max, A_max, sigma, verbose=False,
                full_dist_map=full_dist_map)
            sum_num_neighbors += num_neighbors
            class_v, N_v, T_v = classifying_orientation(
                i, V_v, epsilon=epsilon, eta=eta, verbose=False)
            # Adding the properties to the graph tg and counting classes:
            v = tg.graph.vertex(i)
            tg.graph.vp.orientation_class[v] = class_v
            tg.graph.vp.N_v[v] = N_v
            tg.graph.vp.T_v[v] = T_v
            try:
                classes_counts[class_v] += 1
            except KeyError:
                classes_counts[class_v] = 1
        avg_num_neighbors = float(sum_num_neighbors) / float(num_v)

    # Printing out some numbers concerning the first run:
    print("Average number of geodesic neighbors for all vertices: {}".format(
        avg_num_neighbors))
    print("{} surface patches".format(classes_counts[1]))
    if 2 in classes_counts:
        print("{} crease junctions".format(classes_counts[2]))
    if 3 in classes_counts:
        print("{} no preferred orientation".format(classes_counts[3]))

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    minutes, seconds = divmod(duration1, 60)
    print('First run took: {} min {} s'.format(minutes, seconds))

    if runtimes is not None:
        with open(runtimes, 'a') as f:
            f.write("{};{};{};{};{};{};".format(
                num_v, radius_hit, g_max, avg_num_neighbors, cores, duration1))

    return tg


def preparation_for_curvature_estimation(
        tg, exclude_borders=0, graph_file='temp.gt'):
    """
    Does preparation for principal directions and curvature estimation after the
    surface orientation estimation (second part used by
    normals_directions_and_curvature_estimation): adds vertex properties to be
    filled by all curvature methods, excludes triangles at the borders from
    curvatures calculation and saves the graph to a file.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        exclude_borders (int, optional): if > 0, principle curvatures and
            directions are not estimated for triangles within this distance to
            surface borders (default 0)
        graph_file (str): file path to save the graph

    Returns:
        resulting triangle graph instance attributes: surface (scaled to nm),
        scale_factor_to_nm
    """
    # * Adding vertex properties to be filled by all curvature methods *
    # vertex properties for storing the estimated principal directions of the
    # maximal and minimal curvatures of the corresponding triangle (if the
    # vertex belongs to class 1; scaled in nm):
    tg.graph.vp.T_1 = tg.graph.new_vertex_property("vector<float>")
    tg.graph.vp.T_2 = tg.graph.new_vertex_property("vector<float>")
    # vertex properties for storing the estimated maximal and minimal curvatures
    # of the corresponding triangle (if the vertex belongs to class 1; scaled in
    # nm):
    tg.graph.vp.kappa_1 = tg.graph.new_vertex_property("float")
    tg.graph.vp.kappa_2 = tg.graph.new_vertex_property("float")
    # vertex property for storing the Gaussian curvature calculated from kappa_1
    # and kappa_2 at the corresponding triangle:
    tg.graph.vp.gauss_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the mean curvature calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.mean_curvature_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the shape index calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.shape_index_VV = tg.graph.new_vertex_property("float")
    # vertex property for storing the curvedness calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    tg.graph.vp.curvedness_VV = tg.graph.new_vertex_property("float")

    if exclude_borders > 0:  # Exclude triangles at the borders from curvatures
        # calculation to remove them in the end

        t_begin0 = time.time()

        print('\nFinding triangles that are {} nm to surface borders and '
              'excluding them from curvatures calculation...'.format(
            exclude_borders))
        # Mark vertices at borders with vertex property 'is_near_border', do not
        # delete
        tg.find_vertices_near_border(exclude_borders, purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Finding borders took: {} min {} s'.format(minutes, seconds))

    # Save the graph to a file for use by different methods in the second run:
    tg.graph.save(graph_file)


def curvature_estimation(
        radius_hit, exclude_borders=0, graph_file='temp.gt', method="VV",
        page_curvature_formula=False, num_points=None, area2=True,
        poly_surf=None, full_dist_map=False, cores=4, runtimes=None):
    """
    Runs the second pass of the modified Normal Vector Voting algorithm with
    the given method to estimate principle curvatures and directions for a
    surface using its triangle graph (third and the last part used by
    normals_directions_and_curvature_estimation).

    Args:
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        exclude_borders (int, optional): if > 0, principle curvatures and
            directions are not estimated for triangles within this distance to
            surface borders (default 0)
        graph_file (string, optional): name for a graph file after the first run
            of the algorithm (default 'temp.gt' will be removed after loading)
        method (str, optional): a method to run in the second pass ('VV', 'VVCF'
            and 'VCTV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV and VVCF
            (see collecting_curvature_votes)
        num_points (int, optional): number of points to sample in each estimated
            principal direction in order to fit parabola and estimate curvature
            (necessary is 'VVCF' is in methods list)
        area2 (boolean, optional): if True (default False), votes are
            weighted by triangle area also in the second pass
        poly_surf (vtkPolyData): surface from which the graph was generated,
            scaled to nm (required only if method="VCTV", default None)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map is
            calculated later for each vertex (default)
        cores (int): number of cores to run VV (collecting_curvature_votes and
            estimate_curvature) in parallel (default 8)
        runtimes (str): if given, runtimes and some parameters are added to
            this file (default None)

    Returns:
        a tuple of TriangleGraph graph and vtkPolyData surface of triangles
        with classified orientation and estimated normals or tangents, principle
        curvatures and directions
    """
    tg = TriangleGraph()
    tg.graph = load_graph(graph_file)
    if graph_file == 'temp.gt':
        remove(graph_file)

    if full_dist_map is True:
        # * Distance map between all pairs of vertices *
        t_begin0 = time.time()
        print("\nCalculating the full distance map...")
        full_dist_map = shortest_distance(  # <class 'graph_tool.PropertyMap'>
            tg.graph, weights=tg.graph.ep.distance)
        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Calculation of the full distance map took: {} min {} s'.format(
            minutes, seconds))
    else:
        full_dist_map = None

    print("\nSecond run: estimating principle curvatures and directions for "
          "surface patches using {}...".format(method))
    t_begin2 = time.time()

    collecting_curvature_votes = tg.collecting_curvature_votes
    gen_curv_vote = tg.gen_curv_vote
    estimate_curvature = tg.estimate_curvature
    estimate_directions_and_fit_curves = tg.estimate_directions_and_fit_curves
    orientation_class = tg.graph.vp.orientation_class
    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or B_v_list[i] is None"
    if exclude_borders > 0:
        is_near_border = tg.graph.vp.is_near_border
        condition1 += " and is_near_border[v] == 0"
        condition2 += " or is_near_border[v] == 1"
    g_max = math.pi * radius_hit / 2.0
    sigma = g_max / 3.0
    if (method == "VV" or method == "VVCF") and area2:
        A, _ = tg.get_areas()
        A = np.array(A)
        A_max = np.max(A)
        print("Maximal triangle area = {}".format(A_max))
    else:
        A_max = 0.0

    # Estimate principal directions and curvatures (and calculate the
    # Gaussian and mean curvatures, shape index and curvedness) for
    # vertices belonging to a surface patch and not on border
    good_vertices_ind = []
    B_v_list = []  # has same length as good_vertices_ind
    for v in tg.graph.vertices():
        if eval(condition1):
            good_vertices_ind.append(int(v))  # tg.graph.vertex_index[v]
            # Voting and curvature estimation for VCTV:
            if method == "VCTV":  # sequential processing, edits the graph
                # None is returned if curvature at v cannot be estimated
                B_v = gen_curv_vote(poly_surf, v, radius_hit, verbose=False)
                B_v_list.append(B_v)
    print("{} vertices to estimate curvature".format(len(good_vertices_ind)))

    if method == "VV" or method == "VVCF":
        if cores > 1:  # parallel processing
            p = pp.ProcessPool(cores)
            print('Opened a pool with {} processes'.format(cores))

            # Curvature votes collection is common for VV and VVCF:
            # None is returned if v does not have any neighbor belonging to
            # a surface patch, then estimate_curvature will return Nones as well
            B_v_list = p.map(partial(collecting_curvature_votes,
                                     g_max=g_max, sigma=sigma, verbose=False,
                                     page_curvature_formula=page_curvature_formula,
                                     A_max=A_max, full_dist_map=full_dist_map),
                             good_vertices_ind)
            # Curvature estimation for VV:
            if method == "VV":
                # results_list has same length as good_vertices_ind
                # columns: T_1, T_2, kappa_1, kappa_2, gauss_curvature,
                # mean_curvature, shape_index, curvedness
                results_list = p.map(partial(estimate_curvature,
                                             verbose=False),
                                     good_vertices_ind, B_v_list)
                p.close()
                p.clear()
                results_array = np.array(results_list, dtype=object)
                T_1_array = results_array[:, 0]
                T_2_array = results_array[:, 1]
                kappa_1_array = results_array[:, 2]
                kappa_2_array = results_array[:, 3]
                gauss_curvature_array = results_array[:, 4]
                mean_curvature_array = results_array[:, 5]
                shape_index_array = results_array[:, 6]
                curvedness_array = results_array[:, 7]
                # Add T_1, T_2, kappa_1, kappa_2, Gaussian and mean curvatures,
                # shape index and curvedness as properties to the graph:
                # (v_ind is vertex v index, i is v_ind index in results arrays)
                for i, v_ind in enumerate(good_vertices_ind):
                    v = tg.graph.vertex(v_ind)
                    if T_1_array[i] is not None:
                        tg.graph.vp.T_1[v] = T_1_array[i]
                        tg.graph.vp.T_2[v] = T_2_array[i]
                        tg.graph.vp.kappa_1[v] = kappa_1_array[i]
                        tg.graph.vp.kappa_2[v] = kappa_2_array[i]
                        tg.graph.vp.gauss_curvature_VV[v] = gauss_curvature_array[i]
                        tg.graph.vp.mean_curvature_VV[v] = mean_curvature_array[i]
                        tg.graph.vp.shape_index_VV[v] = shape_index_array[i]
                        tg.graph.vp.curvedness_VV[v] = curvedness_array[i]

        else:  # cores == 1, sequential processing
            # Curvature votes collection is common for VV and VVCF:
            for v_ind in good_vertices_ind:
                B_v = collecting_curvature_votes(
                    v_ind, g_max, sigma, verbose=False,
                    page_curvature_formula=page_curvature_formula,
                    A_max=A_max, full_dist_map=full_dist_map)
                B_v_list.append(B_v)
            # Curvature estimation for VV:
            if method == "VV":
                for i, v_ind in enumerate(good_vertices_ind):
                    B_v = B_v_list[i]
                    results = estimate_curvature(v_ind, B_v, verbose=False)
                    T_1 = results[0]
                    T_2 = results[1]
                    kappa_1 = results[2]
                    kappa_2 = results[3]
                    gauss_curvature = results[4]
                    mean_curvature = results[5]
                    shape_index = results[6]
                    curvedness = results[7]
                    # Add the properties to the graph:
                    v = tg.graph.vertex(v_ind)
                    if T_1 is not None:
                        tg.graph.vp.T_1[v] = T_1
                        tg.graph.vp.T_2[v] = T_2
                        tg.graph.vp.kappa_1[v] = kappa_1
                        tg.graph.vp.kappa_2[v] = kappa_2
                        tg.graph.vp.gauss_curvature_VV[v] = gauss_curvature
                        tg.graph.vp.mean_curvature_VV[v] = mean_curvature
                        tg.graph.vp.shape_index_VV[v] = shape_index
                        tg.graph.vp.curvedness_VV[v] = curvedness

    # Curvature estimation for VVCF:
    if method == "VVCF":  # sequential processing
        # * Adding vertex properties to be filled in estimate_directions_and_fit
        # curves *
        # vertex properties for storing curve fitting errors (variances) in
        # maximal and minimal principal directions at the vertex (belonging to
        # class 1):
        tg.graph.vp.fit_error_1 = tg.graph.new_vertex_property("float")
        tg.graph.vp.fit_error_2 = tg.graph.new_vertex_property("float")

        for i, v_ind in enumerate(good_vertices_ind):
            B_v = B_v_list[i]
            results = estimate_directions_and_fit_curves(
                poly_surf, v_ind, B_v, radius_hit, num_points, verbose=False)
            T_1 = results[0]
            T_2 = results[1]
            kappa_1 = results[2]
            kappa_2 = results[3]
            gauss_curvature = results[4]
            mean_curvature = results[5]
            shape_index = results[6]
            curvedness = results[7]
            fit_error_1 = results[8]
            fit_error_2 = results[9]
            # Add the properties to the graph:
            v = tg.graph.vertex(v_ind)
            if T_1 is not None:
                tg.graph.vp.T_1[v] = T_1
                tg.graph.vp.T_2[v] = T_2
                tg.graph.vp.kappa_1[v] = kappa_1
                tg.graph.vp.kappa_2[v] = kappa_2
                tg.graph.vp.gauss_curvature_VV[v] = gauss_curvature
                tg.graph.vp.mean_curvature_VV[v] = mean_curvature
                tg.graph.vp.shape_index_VV[v] = shape_index
                tg.graph.vp.curvedness_VV[v] = curvedness
                tg.graph.vp.fit_error_1[v] = fit_error_1
                tg.graph.vp.fit_error_2[v] = fit_error_2

    # For all methods:
    # For crease, no preferably oriented vertices, vertices on border or
    # vertices lacking neighbors, add placeholders to the corresponding
    # vertex properties
    for v in tg.graph.vertices():
        v_ind = int(v)  # index of v in the graph
        if v_ind in good_vertices_ind:
            i = good_vertices_ind.index(v_ind)  # index of v in B_v_list
        if eval(condition2):  # i used here
            tg.graph.vp.T_1[v] = np.zeros(shape=3)
            tg.graph.vp.T_2[v] = np.zeros(shape=3)
            tg.graph.vp.kappa_1[v] = 0
            tg.graph.vp.kappa_2[v] = 0
            tg.graph.vp.gauss_curvature_VV[v] = 0
            tg.graph.vp.mean_curvature_VV[v] = 0
            tg.graph.vp.shape_index_VV[v] = 0
            tg.graph.vp.curvedness_VV[v] = 0

    if exclude_borders > 0:
        tg.find_vertices_near_border(exclude_borders, purge=True)

    # Transforming the resulting graph to a surface with triangles:
    surface_curv = tg.graph_to_triangle_poly()

    t_end2 = time.time()
    duration2 = t_end2 - t_begin2
    minutes, seconds = divmod(duration2, 60)
    print('Second run of {} took: {} min {} s'.format(method, minutes, seconds))

    # adding to the runtimes CSV file:
    # - method
    # - duration2
    if runtimes is not None:
        with open(runtimes, 'a') as f:
            f.write("{};{}\n".format(method, duration2))

    return tg, surface_curv
