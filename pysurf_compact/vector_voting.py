import time
import numpy as np
# from scipy import stats
import math
from graph_tool import load_graph

import pexceptions
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
    print '\nPreparing for running modified Vector Voting...'

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    g_max = math.pi * radius_hit / 2
    print "radius_hit = {}".format(radius_hit)
    print "g_max = {}".format(g_max)

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A_max = np.max(A)
    print "Maximal triangle area = %s" % A_max

    # * Orientation classification parameters *
    print "epsilon = %s" % epsilon
    print "eta = %s" % eta

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
    print 'Preparation took: %s min %s s' % divmod(duration, 60)

    # Main algorithm
    t_begin = time.time()

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print "\nRunning modified Vector Voting for all vertices..."

    print ("\nFirst run: classifying orientation and estimating normals for "
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
    print ("Average number of geodesic neighbors for all vertices: %s"
           % avg_num_geodesic_neighbors)
    print "%s surface patches" % classes_counts[1]
    if 2 in classes_counts:
        print "%s crease junctions" % classes_counts[2]
    if 3 in classes_counts:
        print "%s no preferred orientation" % classes_counts[3]

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    print 'First run took: %s min %s s' % divmod(duration1, 60)

    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or B_v is None"

    if exclude_borders:  # Exclude triangles at the borders from curvatures
        # calculation and remove them in the end

        t_begin0 = time.time()

        print ('\nFinding triangles that are at surface borders and '
               'excluding them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        print 'Finding graph border took: %s min %s s' % divmod(duration0, 60)

        condition1 += " and is_on_border[v] == 0"
        condition2 += " or is_on_border[v] == 1"

    print ("\nSecond run: estimating principle curvatures and directions for "
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
                    v, all_neighbor_idx_to_dist[i], sigma, verbose=False,
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
    print 'Second run took: %s min %s s' % divmod(duration2, 60)

    t_end = time.time()
    duration = t_end - t_begin
    print 'Modified Vector Voting took: %s min %s s' % divmod(duration, 60)

    # Transforming the resulting graph to a surface with triangles:
    surface_VV = tg.graph_to_triangle_poly()
    return surface_VV


def vector_voting_curve_fitting(
        tg, radius_hit, num_points, epsilon=0, eta=0, exclude_borders=True,
        page_curvature_formula=False):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, principle curvatures and directions for a surface using its
    triangle graph.

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
    # TODO modify the docstring!
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print '\nPreparing for running modified Vector Voting...'

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    g_max = math.pi * radius_hit / 2
    print "radius_hit = {}".format(radius_hit)
    print "g_max = {}".format(g_max)

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A_max = np.max(A)
    print "Maximal triangle area = %s" % A_max

    # * Orientation classification parameters *
    print "epsilon = %s" % epsilon
    print "eta = %s" % eta

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
    print 'Preparation took: %s min %s s' % divmod(duration, 60)

    # Main algorithm
    t_begin = time.time()

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print "\nRunning modified Vector Voting for all vertices..."

    print ("\nFirst run: classifying orientation and estimating normals for "
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
    print ("Average number of geodesic neighbors for all vertices: %s"
           % avg_num_geodesic_neighbors)
    print "%s surface patches" % classes_counts[1]
    if 2 in classes_counts:
        print "%s crease junctions" % classes_counts[2]
    if 3 in classes_counts:
        print "%s no preferred orientation" % classes_counts[3]

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    print 'First run took: %s min %s s' % divmod(duration1, 60)

    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or B_v is None"

    if exclude_borders:  # Exclude triangles at the borders from curvatures
        # calculation and remove them in the end

        t_begin0 = time.time()

        print ('\nFinding triangles that are at surface borders and '
               'excluding them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        print 'Finding graph border took: %s min %s s' % divmod(duration0, 60)

        condition1 += " and is_on_border[v] == 0"
        condition2 += " or is_on_border[v] == 1"

    print ("\nSecond run: estimating principle directions and curvatures for "
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
                    v, all_neighbor_idx_to_dist[i], sigma, verbose=False,
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
    print 'Second run took: %s min %s s' % divmod(duration2, 60)

    t_end = time.time()
    duration = t_end - t_begin
    print 'Modified Vector Voting took: %s min %s s' % divmod(duration, 60)

    # Transforming the resulting graph to a surface with triangles:
    surface_VV = tg.graph_to_triangle_poly()
    return surface_VV


def vector_curvature_tensor_voting(
        tg, radius_hit, epsilon=0, eta=0, exclude_borders=True):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, principle curvatures and directions for a surface using its
    triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
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
    # TODO modify the docstring!
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print '\nPreparing for running modified Vector Voting...'

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    g_max = math.pi * radius_hit / 2
    print "radius_hit = {}".format(radius_hit)
    print "g_max = {}".format(g_max)

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A_max = np.max(A)
    print "Maximal triangle area = %s" % A_max

    # * Orientation classification parameters *
    print "epsilon = %s" % epsilon
    print "eta = %s" % eta

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
    print 'Preparation took: %s min %s s' % divmod(duration, 60)

    # Main algorithm
    t_begin = time.time()

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print "\nRunning modified Vector Voting for all vertices..."

    print ("\nFirst run: classifying orientation and estimating normals for "
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
    print ("Average number of geodesic neighbors for all vertices: %s"
           % avg_num_geodesic_neighbors)
    print "%s surface patches" % classes_counts[1]
    if 2 in classes_counts:
        print "%s crease junctions" % classes_counts[2]
    if 3 in classes_counts:
        print "%s no preferred orientation" % classes_counts[3]

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    print 'First run took: %s min %s s' % divmod(duration1, 60)

    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or result is None"

    if exclude_borders:  # Exclude triangles at the borders from curvatures
        # calculation and remove them in the end

        t_begin0 = time.time()

        print ('\nFinding triangles that are at surface borders and '
               'excluding them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        print 'Finding graph border took: %s min %s s' % divmod(duration0, 60)

        condition1 += " and is_on_border[v] == 0"
        condition2 += " or is_on_border[v] == 1"

    print ("\nSecond run: estimating principle directions and curvatures for "
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
            result = gen_curv_vote(v, radius_hit, verbose=False)
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
    print 'Second run took: %s min %s s' % divmod(duration2, 60)

    t_end = time.time()
    duration = t_end - t_begin
    print 'Modified Vector Voting took: %s min %s s' % divmod(duration, 60)

    # Transforming the resulting graph to a surface with triangles:
    surface_VV = tg.graph_to_triangle_poly()
    return surface_VV


def normals_directions_and_curvature_estimation(
        tg, radius_hit, epsilon=0, eta=0, exclude_borders=True,
        methods=['VV'], page_curvature_formula=False, num_points=None):
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
        methods (list, optional): all methods to run in the second pass ('VV',
            'VVCF' and 'VCTV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV and VVCF
            (see collecting_curvature_votes)
        num_points (int, optional): number of points to sample in each estimated
            principal direction in order to fit parabola and estimate curvature
            (necessary is 'VVCF' is in methods list)
    Returns:
        a dictionary mapping the method name ('VV', 'VVCF' and 'VCTV') to the
        tuple of two elements: TriangleGraph graph and vtkPolyData surface of
        triangles with classified orientation and estimated normals or tangents,
        principle curvatures and directions

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    t_begin = time.time()

    # Preparation (calculations that are the same for the whole graph)
    t_begin0 = time.time()
    print '\nPreparing for running modified Vector Voting...'

    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    g_max = math.pi * radius_hit / 2
    print "radius_hit = {}".format(radius_hit)
    print "g_max = {}".format(g_max)

    # * sigma *
    sigma = g_max / 3.0

    # * Maximal triangle area *
    A, _ = tg.get_areas()
    A_max = np.max(A)
    print "Maximal triangle area = %s" % A_max

    # * Orientation classification parameters *
    print "epsilon = %s" % epsilon
    print "eta = %s" % eta

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

    t_end0 = time.time()
    duration0 = t_end0 - t_begin0
    print 'Preparation took: %s min %s s' % divmod(duration0, 60)

    # Main algorithm

    # * For all vertices, collecting normal vector votes, while calculating
    # average number of the geodesic neighbors, and classifying the orientation
    # of each vertex *
    print "\nRunning modified Vector Voting for all vertices..."

    print ("\nFirst run: classifying orientation and estimating normals for "
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
    print ("Average number of geodesic neighbors for all vertices: %s"
           % avg_num_geodesic_neighbors)
    print "%s surface patches" % classes_counts[1]
    if 2 in classes_counts:
        print "%s crease junctions" % classes_counts[2]
    if 3 in classes_counts:
        print "%s no preferred orientation" % classes_counts[3]

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    print 'First run took: %s min %s s' % divmod(duration1, 60)

    condition1 = "orientation_class[v] == 1"
    condition2 = "orientation_class[v] != 1 or (B_v is None and result is None)"

    if exclude_borders:  # Exclude triangles at the borders from curvatures
        # calculation and remove them in the end

        t_begin0 = time.time()

        print ('\nFinding triangles that are at surface borders and '
               'excluding them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        print 'Finding graph border took: %s min %s s' % divmod(duration0, 60)

        condition1 += " and is_on_border[v] == 0"
        condition2 += " or is_on_border[v] == 1"

    t_begin0 = time.time()

    # Save the graph to a file for use by different methods in the second run:
    import os
    cwd = os.getcwd()
    graph_file = cwd + "/temp.gt"
    tg.graph.save(graph_file)

    t_end0 = time.time()
    duration0 = t_end0 - t_begin0
    print '\nSaving the graph took: %s min %s s' % divmod(duration0, 60)
    print tg.graph

    results = {}
    if 'VV' in methods:
        t_begin0 = time.time()

        tg_VV = TriangleGraph(
            surface=tg.surface, scale_factor_to_nm=tg.scale_factor_to_nm,
            scale_x=tg.scale_x, scale_y=tg.scale_y, scale_z=tg.scale_z)
        tg_VV.graph = load_graph(graph_file)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        print '\nLoading the graph took: %s min %s s' % divmod(duration0, 60)
        print tg_VV.graph

        print ("\nSecond run: estimating principle curvatures and directions "
               "for surface patches using vector voting (VV)...")
        t_begin2 = time.time()

        orientation_class = tg_VV.graph.vp.orientation_class
        collecting_curvature_votes = tg_VV.collecting_curvature_votes
        estimate_curvature = tg_VV.estimate_curvature
        if exclude_borders:
            is_on_border = tg_VV.graph.vp.is_on_border

        for i, v in enumerate(tg_VV.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch and not on border
            B_v = None  # initialization
            result = None  # initialization
            if eval(condition1):
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v = collecting_curvature_votes(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False,
                        page_curvature_formula=page_curvature_formula)
            if B_v is not None:
                estimate_curvature(v, B_v, verbose=False)
            # For crease, no preferably oriented vertices, vertices on border or
            # vertices lacking neighbors, add placeholders to the corresponding
            # vertex properties
            if eval(condition2):
                tg_VV.graph.vp.T_1[v] = np.zeros(shape=3)
                tg_VV.graph.vp.T_2[v] = np.zeros(shape=3)
                tg_VV.graph.vp.kappa_1[v] = 0
                tg_VV.graph.vp.kappa_2[v] = 0
                tg_VV.graph.vp.gauss_curvature_VV[v] = 0
                tg_VV.graph.vp.mean_curvature_VV[v] = 0
                tg_VV.graph.vp.shape_index_VV[v] = 0
                tg_VV.graph.vp.curvedness_VV[v] = 0

        if exclude_borders:
            tg_VV.find_graph_border(purge=True)

        # Transforming the resulting graph to a surface with triangles:
        surface_VV = tg_VV.graph_to_triangle_poly()

        results['VV'] = (tg_VV, surface_VV)

        t_end2 = time.time()
        duration2 = t_end2 - t_begin2
        print 'Second run of VV took: %s min %s s' % divmod(duration2, 60)

    if 'VVCF' in methods:
        t_begin0 = time.time()

        tg_VVCF = TriangleGraph(
            surface=tg.surface, scale_factor_to_nm=tg.scale_factor_to_nm,
            scale_x=tg.scale_x, scale_y=tg.scale_y, scale_z=tg.scale_z)
        tg_VVCF.graph = load_graph(graph_file)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        print '\nLoading the graph took: %s min %s s' % divmod(duration0, 60)
        print tg_VVCF.graph

        print ("\nSecond run: estimating principle curvatures and directions "
               "for surface patches using vector voting with curve fitting "
               "(VVCF)...")
        t_begin2 = time.time()

        orientation_class = tg_VVCF.graph.vp.orientation_class
        collecting_curvature_votes = tg_VVCF.collecting_curvature_votes
        estimate_directions_and_fit_curves = \
            tg_VVCF.estimate_directions_and_fit_curves
        if exclude_borders:
            is_on_border = tg_VVCF.graph.vp.is_on_border

        # * Adding vertex properties to be filled in estimate_directions_and_fit
        # curves *
        # vertex properties for storing curve fitting errors (variances) in
        # maximal and minimal principal directions at the vertex (belonging to
        # class 1):
        tg_VVCF.graph.vp.fit_error_1 = tg_VVCF.graph.new_vertex_property(
            "float")
        tg_VVCF.graph.vp.fit_error_2 = tg_VVCF.graph.new_vertex_property(
            "float")

        for i, v in enumerate(tg_VVCF.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch and not on border
            B_v = None  # initialization
            result = None  # initialization
            if eval(condition1):
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v = collecting_curvature_votes(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False,
                        page_curvature_formula=page_curvature_formula)
            if B_v is not None:
                estimate_directions_and_fit_curves(v, B_v, radius_hit,
                                                   num_points, verbose=False)
            # For crease, no preferably oriented vertices, vertices on border or
            # vertices lacking neighbors, add placeholders to the corresponding
            # vertex properties
            if eval(condition2):
                tg_VVCF.graph.vp.T_1[v] = np.zeros(shape=3)
                tg_VVCF.graph.vp.T_2[v] = np.zeros(shape=3)
                tg_VVCF.graph.vp.fit_error_1[v] = 0
                tg_VVCF.graph.vp.fit_error_2[v] = 0
                tg_VVCF.graph.vp.kappa_1[v] = 0
                tg_VVCF.graph.vp.kappa_2[v] = 0
                tg_VVCF.graph.vp.gauss_curvature_VV[v] = 0
                tg_VVCF.graph.vp.mean_curvature_VV[v] = 0
                tg_VVCF.graph.vp.shape_index_VV[v] = 0
                tg_VVCF.graph.vp.curvedness_VV[v] = 0

        if exclude_borders:
            tg_VVCF.find_graph_border(purge=True)

        # Transforming the resulting graph to a surface with triangles:
        surface_VVCF = tg_VVCF.graph_to_triangle_poly()

        results['VVCF'] = (tg_VVCF, surface_VVCF)

        t_end2 = time.time()
        duration2 = t_end2 - t_begin2
        print 'Second run of VVCF took: %s min %s s' % divmod(duration2, 60)

    if 'VCTV' in methods:
        t_begin0 = time.time()

        tg_VCTV = TriangleGraph(
            surface=tg.surface, scale_factor_to_nm=tg.scale_factor_to_nm,
            scale_x=tg.scale_x, scale_y=tg.scale_y, scale_z=tg.scale_z)
        tg_VCTV.graph = load_graph(graph_file)

        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        print '\nLoading the graph took: %s min %s s' % divmod(duration0, 60)
        print tg_VCTV.graph

        print ("\nSecond run: estimating principle curvatures and directions "
               "for surface patches using vector curvature tensor voting "
               "(VCTV)...")
        t_begin2 = time.time()

        orientation_class = tg_VCTV.graph.vp.orientation_class
        gen_curv_vote = tg_VCTV.gen_curv_vote
        if exclude_borders:
            is_on_border = tg_VCTV.graph.vp.is_on_border

        for i, v in enumerate(tg_VCTV.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch and not on border
            B_v = None  # initialization
            result = None  # initialization
            if eval(condition1):
                # None is returned if curvature at v cannot be estimated
                result = gen_curv_vote(v, radius_hit, verbose=False)
            # For crease, no preferably oriented vertices, vertices on border or
            # vertices lacking neighbors, add placeholders to the corresponding
            # vertex properties
            if eval(condition2):
                tg_VCTV.graph.vp.T_1[v] = np.zeros(shape=3)
                tg_VCTV.graph.vp.T_2[v] = np.zeros(shape=3)
                tg_VCTV.graph.vp.kappa_1[v] = 0
                tg_VCTV.graph.vp.kappa_2[v] = 0
                tg_VCTV.graph.vp.gauss_curvature_VV[v] = 0
                tg_VCTV.graph.vp.mean_curvature_VV[v] = 0
                tg_VCTV.graph.vp.shape_index_VV[v] = 0
                tg_VCTV.graph.vp.curvedness_VV[v] = 0

        if exclude_borders:
            tg_VCTV.find_graph_border(purge=True)

        # Transforming the resulting graph to a surface with triangles:
        surface_VCTV = tg_VCTV.graph_to_triangle_poly()

        results['VCTV'] = (tg_VCTV, surface_VCTV)

        t_end2 = time.time()
        duration2 = t_end2 - t_begin2
        print 'Second run of VCTV took: %s min %s s' % divmod(duration2, 60)

    t_end = time.time()
    duration = t_end - t_begin
    print 'Whole method took: %s min %s s' % divmod(duration, 60)
    return results
