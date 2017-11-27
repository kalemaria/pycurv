import time
import numpy as np
# from scipy import stats

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


def vector_voting(tg, k=3, g_max=0.0, epsilon=0, eta=0, exclude_borders=True):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, principle curvatures and directions for a surface using its
    triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        k (int, optional): parameter of Normal Vector Voting algorithm
            determining the geodesic neighborhood radius:
            g_max = k * average weak triangle graph edge length (default k=3)
        g_max (float, optional): geodesic neighborhood radius in length unit of
            the graph, e.g. nanometers; if positive (default 0.0) this g_max
            will be used and k will be ignored
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
        * Either g_max or k must be positive (if both are positive, the
          specified g_max will be used).
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print '\nPreparing for running modified Vector Voting...'

    if g_max > 0:
        # * Maximal geodesic distance g_max (given directly) *
        print "g_max = {}".format(g_max)
    elif k > 0:
        # * k, integer multiple of the average length of the triangle edges *
        print "k = {}".format(k)

        # * Average length of the triangle edges *
        # weak edges of triangle graph are the most similar in length to surface
        # triangle edges
        avg_weak_edge_length = tg.calculate_average_edge_length(
            prop_e="is_strong", value=0)
        try:
            assert(avg_weak_edge_length > 0)
        except AssertionError:
            print ("Something wrong in the graph construction has happened: "
                   "average weak edge length is 0.")
            exit(1)

        # * Maximal geodesic distance g_max (depend on k but is also the same
        # for the whole graph) *
        g_max = k * avg_weak_edge_length
        print "g_max = {}".format(g_max)
    else:
        error_msg = ("Either g_max or k must be positive (if both are "
                     "positive, the specified g_max will be used).")
        raise pexceptions.PySegInputError(expr='vector_voting',
                                          msg=error_msg)

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

    collecting_votes = tg.collecting_votes
    all_num_neighbors = []
    all_neighbor_idx_to_dist = []
    classifying_orientation = tg.classifying_orientation
    classes_counts = {}
    for v in tg.graph.vertices():
        neighbor_idx_to_dist, V_v = collecting_votes(v, g_max, A_max, sigma,
                                                     verbose=False)
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

    print ("\nSecond run: estimating principle curvatures and directions for "
           "surface patches...")
    t_begin2 = time.time()

    orientation_class = tg.graph.vp.orientation_class
    collecting_votes2 = tg.collecting_votes2
    estimate_curvature = tg.estimate_curvature
    if exclude_borders is False:
        for i, v in enumerate(tg.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch
            B_v = None  # initialization
            if orientation_class[v] == 1:
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v = collecting_votes2(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False)
            if B_v is not None:
                estimate_curvature(v, B_v, verbose=False)
            # For crease, no preferably oriented vertices or vertices lacking
            # neighbors, add placeholders to the corresponding vertex properties
            if orientation_class[v] != 1 or B_v is None:
                tg.graph.vp.T_1[v] = np.zeros(shape=3)
                tg.graph.vp.T_2[v] = np.zeros(shape=3)
                tg.graph.vp.kappa_1[v] = 0
                tg.graph.vp.kappa_2[v] = 0
                tg.graph.vp.gauss_curvature_VV[v] = 0
                tg.graph.vp.mean_curvature_VV[v] = 0
                tg.graph.vp.shape_index_VV[v] = 0
                tg.graph.vp.curvedness_VV[v] = 0
    # Exclude triangles at the borders from curvatures calculation and remove
    # them in the end
    else:
        print ('\nFinding triangles that are at surface borders and excluding '
               'them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)
        is_on_border = tg.graph.vp.is_on_border
        for i, v in enumerate(tg.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch and not on border
            B_v = None  # initialization
            if orientation_class[v] == 1 and is_on_border[v] == 0:
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v = collecting_votes2(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False)
            if B_v is not None:
                estimate_curvature(v, B_v, verbose=False)
            # For crease, no preferably oriented vertices, vertices on border or
            # vertices lacking neighbors, add placeholders to the corresponding
            # vertex properties
            if orientation_class[v] != 1 or is_on_border[v] == 1 or B_v is None:
                tg.graph.vp.T_1[v] = np.zeros(shape=3)
                tg.graph.vp.T_2[v] = np.zeros(shape=3)
                tg.graph.vp.kappa_1[v] = 0
                tg.graph.vp.kappa_2[v] = 0
                tg.graph.vp.gauss_curvature_VV[v] = 0
                tg.graph.vp.mean_curvature_VV[v] = 0
                tg.graph.vp.shape_index_VV[v] = 0
                tg.graph.vp.curvedness_VV[v] = 0
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


def vector_voting_sign_correction(tg, k=3, g_max=0.0, epsilon=0, eta=0,
                                  exclude_borders=True):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, principle curvatures and directions for a surface using its
    triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        k (int, optional): parameter of Normal Vector Voting algorithm
            determining the geodesic neighborhood radius:
            g_max = k * average weak triangle graph edge length (default k=3)
        g_max (float, optional): geodesic neighborhood radius in length unit of
            the graph, e.g. nanometers; if positive (default 0.0) this g_max
            will be used and k will be ignored
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
        * Either g_max or k must be positive (if both are positive, the
          specified g_max will be used).
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    TriangleGraph.num_curvature_is_negated = 0
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print '\nPreparing for running modified Vector Voting...'

    if g_max > 0:
        # * Maximal geodesic distance g_max (given directly) *
        print "g_max = {}".format(g_max)
    elif k > 0:
        # * k, integer multiple of the average length of the triangle edges *
        print "k = {}".format(k)

        # * Average length of the triangle edges *
        # weak edges of triangle graph are the most similar in length to surface
        # triangle edges
        avg_weak_edge_length = tg.calculate_average_edge_length(
            prop_e="is_strong", value=0)
        try:
            assert(avg_weak_edge_length > 0)
        except AssertionError:
            print ("Something wrong in the graph construction has happened: "
                   "average weak edge length is 0.")
            exit(1)

        # * Maximal geodesic distance g_max (depend on k but is also the same
        # for the whole graph) *
        g_max = k * avg_weak_edge_length
        print "g_max = {}".format(g_max)
    else:
        error_msg = ("Either g_max or k must be positive (if both are "
                     "positive, the specified g_max will be used).")
        raise pexceptions.PySegInputError(expr='vector_voting',
                                          msg=error_msg)

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

    # * Adding vertex properties to be filled in sign_voting *
    # vertex property telling whether the vertex is locally planar(0),
    # elliptic (1), hyperbolic (2) or parabolic (3):
    # tg.graph.vp.local_shape = tg.graph.new_vertex_property("int")
    # TODO remove after testing:
    # tg.graph.vp.mu = tg.graph.new_vertex_property("float")
    # tg.graph.vp.sigma = tg.graph.new_vertex_property("float")

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

    collecting_votes = tg.collecting_votes
    all_num_neighbors = []
    all_neighbor_idx_to_dist = []
    classifying_orientation = tg.classifying_orientation
    classes_counts = {}
    for v in tg.graph.vertices():
        neighbor_idx_to_dist, V_v = collecting_votes(v, g_max, A_max, sigma,
                                                     verbose=False)
        all_num_neighbors.append(len(neighbor_idx_to_dist))
        all_neighbor_idx_to_dist.append(neighbor_idx_to_dist)
        # assert len(all_neighbor_idx_to_dist) - 1 == tg.graph.vertex_index[v]
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

    # print '\nRun in the middle: estimating curvature sign...'
    # sign_voting = tg.sign_voting
    # # shapes_counts = {}
    # all_mu = []
    # all_traceS = []
    # for i, v in enumerate(tg.graph.vertices()):
    #     # if tg.graph.vertex_index[v] == 174:
    #     #     verb = True
    #     # else:
    #     #     verb = False
    #     verb = False
    #     mu, traceS = sign_voting(v, all_neighbor_idx_to_dist[i], verbose=verb)
    #     # try:
    #     #     shapes_counts[shape_v] += 1
    #     # except KeyError:
    #     #     shapes_counts[shape_v] = 1
    #     all_mu.append(mu)
    #     all_traceS.append(traceS)
    # # for shape in shapes_counts.keys():
    # #     print "{}: {}".format(shape, shapes_counts[shape])
    # print "mu: [{}, {}]".format(min(all_mu), max(all_mu))
    # # print stats.describe(np.array(all_mu))
    # print "traceS: [{}, {}]".format(min(all_traceS), max(all_traceS))
    # # print stats.describe(np.array(all_traceS))

    print ("\nSecond run: estimating principle curvatures and directions for "
           "surface patches...")
    t_begin2 = time.time()

    orientation_class = tg.graph.vp.orientation_class
    collecting_votes2 = tg.collecting_votes2_sign_correction
    estimate_curvature = tg.estimate_curvature_sign_correction
    all_neighbor_idx_to_orientation = []
    if exclude_borders is False:
        for i, v in enumerate(tg.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch
            B_v = None  # initialization
            if orientation_class[v] == 1:
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v, curvature_is_negated, neighbor_idx_to_orientation = \
                    collecting_votes2(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False)
                all_neighbor_idx_to_orientation.append(
                    neighbor_idx_to_orientation)
            if B_v is not None:
                estimate_curvature(v, B_v, curvature_is_negated, verbose=False)
            # For crease, no preferably oriented vertices or vertices lacking
            # neighbors, add placeholders to the corresponding vertex properties
            if orientation_class[v] != 1 or B_v is None:
                tg.graph.vp.T_1[v] = np.zeros(shape=3)
                tg.graph.vp.T_2[v] = np.zeros(shape=3)
                tg.graph.vp.kappa_1[v] = 0
                tg.graph.vp.kappa_2[v] = 0
                tg.graph.vp.gauss_curvature_VV[v] = 0
                tg.graph.vp.mean_curvature_VV[v] = 0
                tg.graph.vp.shape_index_VV[v] = 0
                tg.graph.vp.curvedness_VV[v] = 0
    # Exclude triangles at the borders from curvatures calculation and remove
    # them in the end
    else:
        print ('\nFinding triangles that are at surface borders and excluding '
               'them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)
        is_on_border = tg.graph.vp.is_on_border
        for i, v in enumerate(tg.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch and not on border
            B_v = None  # initialization
            if orientation_class[v] == 1 and is_on_border[v] == 0:
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v, curvature_is_negated, neighbor_idx_to_orientation = \
                    collecting_votes2(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False)
                all_neighbor_idx_to_orientation.append(
                    neighbor_idx_to_orientation)
            if B_v is not None:
                estimate_curvature(v, B_v, curvature_is_negated, verbose=False)
            # For crease, no preferably oriented vertices, vertices on border or
            # vertices lacking neighbors, add placeholders to the corresponding
            # vertex properties
            if orientation_class[v] != 1 or is_on_border[v] == 1 or B_v is None:
                tg.graph.vp.T_1[v] = np.zeros(shape=3)
                tg.graph.vp.T_2[v] = np.zeros(shape=3)
                tg.graph.vp.kappa_1[v] = 0
                tg.graph.vp.kappa_2[v] = 0
                tg.graph.vp.gauss_curvature_VV[v] = 0
                tg.graph.vp.mean_curvature_VV[v] = 0
                tg.graph.vp.shape_index_VV[v] = 0
                tg.graph.vp.curvedness_VV[v] = 0
        tg.find_graph_border(purge=True)

    t_end2 = time.time()
    duration2 = t_end2 - t_begin2
    print 'Second run took: %s min %s s' % divmod(duration2, 60)

    t_end = time.time()
    duration = t_end - t_begin
    print 'Modified Vector Voting took: %s min %s s' % divmod(duration, 60)
    print "kappa was negated %s times" % TriangleGraph.num_curvature_is_negated

    # Transforming the resulting graph to a surface with triangles:
    surface_VV = tg.graph_to_triangle_poly()
    return surface_VV


def vector_voting_curve_fitting(tg, k=3, g_max=0.0, epsilon=0, eta=0,
                                exclude_borders=True):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation, principle curvatures and directions for a surface using its
    triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        k (int, optional): parameter of Normal Vector Voting algorithm
            determining the geodesic neighborhood radius:
            g_max = k * average weak triangle graph edge length (default k=3)
        g_max (float, optional): geodesic neighborhood radius in length unit of
            the graph, e.g. nanometers; if positive (default 0.0) this g_max
            will be used and k will be ignored
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
        * Either g_max or k must be positive (if both are positive, the
          specified g_max will be used).
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    # Preparation (calculations that are the same for the whole graph)
    t_begin = time.time()
    print '\nPreparing for running modified Vector Voting...'

    if g_max > 0:
        # * Maximal geodesic distance g_max (given directly) *
        print "g_max = {}".format(g_max)
    elif k > 0:
        # * k, integer multiple of the average length of the triangle edges *
        print "k = {}".format(k)

        # * Average length of the triangle edges *
        # weak edges of triangle graph are the most similar in length to surface
        # triangle edges
        avg_weak_edge_length = tg.calculate_average_edge_length(
            prop_e="is_strong", value=0)
        try:
            assert(avg_weak_edge_length > 0)
        except AssertionError:
            print ("Something wrong in the graph construction has happened: "
                   "average weak edge length is 0.")
            exit(1)

        # * Maximal geodesic distance g_max (depend on k but is also the same
        # for the whole graph) *
        g_max = k * avg_weak_edge_length
        print "g_max = {}".format(g_max)
    else:
        error_msg = ("Either g_max or k must be positive (if both are "
                     "positive, the specified g_max will be used).")
        raise pexceptions.PySegInputError(expr='vector_voting',
                                          msg=error_msg)

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

    collecting_votes = tg.collecting_votes
    all_num_neighbors = []
    all_neighbor_idx_to_dist = []
    classifying_orientation = tg.classifying_orientation
    classes_counts = {}
    for v in tg.graph.vertices():
        neighbor_idx_to_dist, V_v = collecting_votes(v, g_max, A_max, sigma,
                                                     verbose=False)
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

    print ("\nSecond run: estimating principle directions and curvatures for "
           "surface patches...")
    t_begin2 = time.time()

    orientation_class = tg.graph.vp.orientation_class
    collecting_votes2 = tg.collecting_votes2
    estimate_directions_and_fit_curves = tg.estimate_directions_and_fit_curves
    if exclude_borders is False:
        for i, v in enumerate(tg.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch
            B_v = None  # initialization
            if orientation_class[v] == 1:
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v = collecting_votes2(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False)
            if B_v is not None:
                estimate_directions_and_fit_curves(
                    v, B_v, g_max, all_neighbor_idx_to_dist[i], verbose=False)
            # For crease, no preferably oriented vertices or vertices lacking
            # neighbors, add placeholders to the corresponding vertex properties
            if orientation_class[v] != 1 or B_v is None:
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
    # Exclude triangles at the borders from curvatures calculation and remove
    # them in the end
    else:
        print ('\nFinding triangles that are at surface borders and excluding '
               'them from curvatures calculation...')
        # Mark vertices at borders with vertex property 'is_on_border', do not
        # delete
        tg.find_graph_border(purge=False)
        is_on_border = tg.graph.vp.is_on_border
        for i, v in enumerate(tg.graph.vertices()):
            # Estimate principal directions and curvatures (and calculate the
            # Gaussian and mean curvatures, shape index and curvedness) for
            # vertices belonging to a surface patch and not on border
            B_v = None  # initialization
            if orientation_class[v] == 1 and is_on_border[v] == 0:
                # None is returned if v does not have any neighbor belonging to
                # a surface patch
                B_v = collecting_votes2(
                        v, all_neighbor_idx_to_dist[i], sigma, verbose=False)
            if B_v is not None:
                estimate_directions_and_fit_curves(
                    v, B_v, g_max, all_neighbor_idx_to_dist[i], verbose=False)
            # For crease, no preferably oriented vertices, vertices on border or
            # vertices lacking neighbors, add placeholders to the corresponding
            # vertex properties
            if orientation_class[v] != 1 or is_on_border[v] == 1 or B_v is None:
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
