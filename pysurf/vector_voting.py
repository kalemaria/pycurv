import time
import numpy as np
import math
from graph_tool import load_graph
from graph_tool.topology import shortest_distance
import pathos.pools as pp
from functools import partial
from os import remove
from os.path import isfile

from surface_graphs import TriangleGraph, PointGraph

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


def normals_directions_and_curvature_estimation(
        sg, radius_hit, epsilon=0, eta=0, methods=['VV'],
        page_curvature_formula=False, full_dist_map=False, graph_file='temp.gt',
        area2=True, only_normals=False, poly_surf=None, cores=4, runtimes=''):
    """
    Runs the modified Normal Vector Voting algorithm (with different options for
    the second pass) to estimate surface orientation, principle curvatures and
    directions for a surface using its triangle graph.

    Args:
        sg (TriangleGraph or PointGraph): triangle or point graph generated
            from a surface of interest
        radius_hit (float): radius in length unit of the graph;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (float, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (float, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph (not possible for PointGraph),
            otherwise a local distance map is calculated later for each vertex
            (default)
        graph_file (string, optional): name for a temporary graph file
            after the first run of the algorithm (default 'temp.gt')
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation; not possible for PointGraph)
        only_normals (boolean, optional): if True (default False), only normals
            are estimated, without principal directions and curvatures, only the
            graph with the orientations, normals or tangents is returned.
        poly_surf (vtkPolyData, optional): surface from which the graph was
            generated, scaled to given units (required only for SSVV, default
            None)
        cores (int, optional): number of cores to run VV in parallel (default 4)
        runtimes (str, optional): if given, runtimes and some parameters are
            added to this file (default '')

    Returns:
        a dictionary mapping the method name ('VV' and 'SSVV') to the
        tuple of two elements: TriangleGraph or PointGraph (if pg was given)
        graph and vtkPolyData surface of triangles with classified orientation
        and estimated normals or tangents, principle curvatures and directions
        (if only_normals is False)

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    t_begin = time.time()

    normals_estimation(sg, radius_hit, epsilon, eta, full_dist_map, cores=cores,
                       runtimes=runtimes, graph_file=graph_file)
    if sg.__class__.__name__ == "PointGraph":
        vertex_based = True
        area2 = False
        full_dist_map = False
    else:
        vertex_based = False

    if only_normals is False:
        if len(methods) > 1:
            # load the graph from file, so each method starts anew
            sg = None
        results = {}
        for method in methods:
            sg_curv, surface_curv = curvature_estimation(
                radius_hit, graph_file=graph_file, method=method,
                page_curvature_formula=page_curvature_formula, area2=area2,
                poly_surf=poly_surf, full_dist_map=full_dist_map, cores=cores,
                runtimes=runtimes, vertex_based=vertex_based, sg=sg)
            results[method] = (sg_curv, surface_curv)
        if graph_file == 'temp.gt' and isfile(graph_file):
            remove(graph_file)

        t_end = time.time()
        duration = t_end - t_begin
        minutes, seconds = divmod(duration, 60)
        print('Whole method took: {} min {} s'.format(minutes, seconds))
        return results


def normals_estimation(sg, radius_hit, epsilon=0, eta=0, full_dist_map=False,
                       cores=4, runtimes='', graph_file='temp.gt'):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation (classification in surface patch with normal, crease junction
    with tangent or no preferred orientation) for a surface using its graph.

    Adds the "orientation_class" (1-3), the estimated normal "N_v" (if class is
    1) and the estimated_tangent "T_v" (if class is 2) as vertex properties
    into the graph.

    Adds the estimated normal "N_v" as vertex property into the graph.

    Args:
        sg (TriangleGraph or PointGraph): triangle or point graph generated
            from a surface of interest
        radius_hit (float): radius in length unit of the graph;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (float, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (float, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph (not possible for PointGraph),
            otherwise a local distance map is calculated later for each vertex
            (default)
        cores (int, optional): number of cores to run VV (collect_normal_votes
            and estimate_normal) in parallel (default 8)
        runtimes (str, optional): if given, runtimes and some parameters are
            added to this file (default '')
        graph_file (str, optional): file path to save the graph

    Returns:
        None

    Note:
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
    A_max = sg.graph.gp.max_triangle_area
    print("Maximal triangle area = {}".format(A_max))

    # * Orientation classification parameters *
    print("epsilon = {}".format(epsilon))
    print("eta = {}".format(eta))

    # * Adding vertex properties to be filled in estimate_normal *
    # vertex property storing the orientation class of the vertex: 1 if it
    # belongs to a surface patch, 2 if it belongs to a crease junction or 3 if
    # it doesn't have a preferred orientation:
    sg.graph.vp.orientation_class = sg.graph.new_vertex_property("int")
    # vertex property for storing the estimated normal of the corresponding
    # vertex (if the vertex belongs to class 1):
    sg.graph.vp.N_v = sg.graph.new_vertex_property("vector<float>")
    # vertex property for storing the estimated tangent of the corresponding
    # vertex (if the vertex belongs to class 2):
    sg.graph.vp.T_v = sg.graph.new_vertex_property("vector<float>")

    t_end0 = time.time()
    duration0 = t_end0 - t_begin0
    minutes, seconds = divmod(duration0, 60)
    print('Preparation took: {} min {} s' .format(minutes, seconds))

    if full_dist_map is True and sg.__class__.__name__ == "TriangleGraph":
        # * Distance map between all pairs of vertices *
        t_begin0 = time.time()
        print("\nCalculating the full distance map...")
        full_dist_map = shortest_distance(  # <class 'graph_tool.PropertyMap'>
            sg.graph, weights=sg.graph.ep.distance)
        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Calculation of the full distance map took: {} min {} s'.format(
            minutes, seconds))
    else:
        full_dist_map = None

    # * For all vertices, collecting normal vector votes, calculating
    # average number of the geodesic neighbors and classifying the orientation
    # of each vertex *
    print("\nRunning modified Vector Voting for all vertices...")

    if eta == 0 and epsilon == 0:
        print("\nFirst pass: estimating normals...")
    else:
        print("\nFirst pass: classifying orientation and estimating normals for"
              " surface patches and tangents for creases...")
    t_begin1 = time.time()

    collect_normal_votes = sg.collect_normal_votes
    estimate_normal = sg.estimate_normal
    num_v = sg.graph.num_vertices()
    print("number of vertices: {}".format(num_v))
    classes_counts = {}
    vertex = sg.graph.vertex
    vp_orientation_class = sg.graph.vp.orientation_class
    vp_N_v = sg.graph.vp.N_v
    vp_T_v = sg.graph.vp.T_v

    if cores > 1:  # parallel processing
        t_begin1_1 = time.time()
        p = pp.ProcessPool(cores)
        print('Opened a pool with {} processes'.format(cores))

        # output is a list with 2 columns:
        # column 0 = num_neighbors (int)
        # column 1 = V_v (3x3 float array)
        # each row i is of vertex v, its index == i
        if sg.__class__.__name__ == "TriangleGraph":
            results1_list = p.map(partial(collect_normal_votes,
                                          g_max=g_max, A_max=A_max, sigma=sigma,
                                          full_dist_map=full_dist_map),
                                  range(num_v))  # a list of vertex v indices
        else:  # PointGraph
            results1_list = p.map(partial(collect_normal_votes,
                                          g_max=g_max, A_max=A_max, sigma=sigma),
                                  range(num_v))  # a list of vertex v indices
        results1_array = np.array(results1_list, dtype=object)
        # Calculating average neighbors number:
        num_neighbors_array = results1_array[:, 0]
        avg_num_neighbors = np.mean(num_neighbors_array)
        # Input of the next parallel calculation:
        V_v_array = results1_array[:, 1]
        t_end1_1 = time.time()
        duration1_1 = t_end1_1 - t_begin1_1

        t_begin1_2 = time.time()
        # output is a list with 3 columns:
        # column 0 = orientation class of the vertex (int)
        # column 1 = N_v (3x1 float array, zeros if class=2 or 3)
        # column 2 = T_v (3x1 float array, zeros if class=1 or 3)
        # each row i is of vertex v, its index == i
        results2_list = p.map(
            partial(estimate_normal, epsilon=epsilon, eta=eta),
            range(num_v), V_v_array)
        p.close()
        p.clear()
        results2_array = np.array(results2_list, dtype=object)
        class_v_array = results2_array[:, 0]
        N_v_array = results2_array[:, 1]
        T_v_array = results2_array[:, 2]
        t_end1_2 = time.time()
        duration1_2 = t_end1_2 - t_begin1_2

        t_begin1_3 = time.time()
        # Adding the estimated properties to the graph and counting classes:
        for i in range(num_v):
            v = vertex(i)
            vp_orientation_class[v] = class_v_array[i]
            vp_N_v[v] = N_v_array[i]
            vp_T_v[v] = T_v_array[i]
            try:
                classes_counts[class_v_array[i]] += 1
            except KeyError:
                classes_counts[class_v_array[i]] = 1
        t_end1_3 = time.time()
        duration1_3 = t_end1_3 - t_begin1_3

    else:  # cores == 1, sequential processing
        sum_num_neighbors = 0
        for i in range(num_v):
            if sg.__class__.__name__ == "TriangleGraph":
                num_neighbors, V_v = collect_normal_votes(
                    i, g_max, A_max, sigma, full_dist_map=full_dist_map)
            else:  # PointGraph
                num_neighbors, V_v = collect_normal_votes(
                    i, g_max, A_max, sigma)
            sum_num_neighbors += num_neighbors
            class_v, N_v, T_v = estimate_normal(i, V_v, epsilon, eta)
            # Adding the estimated properties to the graph and counting classes:
            v = vertex(i)
            vp_orientation_class[v] = class_v
            vp_N_v[v] = N_v
            vp_T_v[v] = T_v
            try:
                classes_counts[class_v] += 1
            except KeyError:
                classes_counts[class_v] = 1
        avg_num_neighbors = float(sum_num_neighbors) / float(num_v)
        duration1_1 = "NA"
        duration1_2 = "NA"
        duration1_3 = "NA"

    # Printing out some numbers concerning the first pass:
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
    print('First pass took: {} min {} s'.format(minutes, seconds))

    if runtimes != '':
        with open(runtimes, 'a') as f:
            f.write("{};{};{};{};{};{};{};{};{};".format(
                num_v, radius_hit, g_max, avg_num_neighbors, cores, duration1,
                duration1_1, duration1_2, duration1_3))

    # Save the graph to a file for use by different methods in the second run:
    sg.graph.save(graph_file)


def curvature_estimation(
        radius_hit, graph_file='temp.gt', method='VV',
        page_curvature_formula=False, area2=True, poly_surf=None,
        full_dist_map=False, cores=4, runtimes='', vertex_based=False, sg=None):
    """
    Runs the second pass of the modified Normal Vector Voting algorithm with
    the given method to estimate principle curvatures and directions for a
    surface using its triangle graph (third and the last part used by
    normals_directions_and_curvature_estimation).

    Args:
        radius_hit (float): radius in length unit of the graph;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        graph_file (string, optional): name for a graph file after the first run
            of the algorithm (default 'temp.gt' will be removed after loading)
        method (str, optional): a method to run in the second pass ('VV' and
            'SSVV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        area2 (boolean, optional): if True (default False), votes are
            weighted by triangle area also in the second pass (not possible for
            vertex-based approach)
        poly_surf (vtkPolyData): scaled surface from which the graph was
            generated, (required only if method="SSVV", default None)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph (not possible for vertex-based
            approach), otherwise a local distance map is calculated later for
            each vertex (default)
        cores (int): number of cores to run VV (collect_curvature_votes and
            estimate_curvature) in parallel (default 4)
        runtimes (str): if given, runtimes and some parameters are added to
            this file (default '')
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center.
        sg (TriangleGraph or PointGraph): if given (default None), this graph
            object will be used instead of loading from the 'graph_file' file

    Returns:
        a tuple of TriangleGraph or PointGraph (if pg was given) graph and
        vtkPolyData surface of triangles with classified orientation and
        estimated normals or tangents, principle curvatures and directions
    """
    if sg is None:
        if vertex_based:
            sg = PointGraph()
        else:
            sg = TriangleGraph()
        sg.graph = load_graph(graph_file)

    if vertex_based:
        # cannot weight by triangle area in vertex-based approach
        area2 = False

    if full_dist_map is True and sg.__class__.__name__ == "TriangleGraph":
        # * Distance map between all pairs of vertices *
        t_begin0 = time.time()
        print("\nCalculating the full distance map...")
        full_dist_map = shortest_distance(  # <class 'graph_tool.PropertyMap'>
            sg.graph, weights=sg.graph.ep.distance)
        t_end0 = time.time()
        duration0 = t_end0 - t_begin0
        minutes, seconds = divmod(duration0, 60)
        print('Calculation of the full distance map took: {} min {} s'.format(
            minutes, seconds))
    else:
        full_dist_map = None

    if method == 'VV':
        if page_curvature_formula:
            method_print = 'NVV'
        elif area2:
            method_print = 'AVV'
        else:
            method_print = 'RVV'
    else:  # method == 'SSVV'
        method_print = 'SSVV'
    print("\nSecond pass: estimating principle curvatures and directions for "
          "surface patches using {}...".format(method_print))
    t_begin2 = time.time()

    g_max = math.pi * radius_hit / 2.0
    sigma = g_max / 3.0
    if method == "VV" and area2:
        A_max = sg.graph.gp.max_triangle_area
        print("Maximal triangle area = {}".format(A_max))
    else:
        A_max = 0.0

    # * Adding vertex properties to be filled by all curvature methods *
    # vertex properties for storing the estimated principal directions of the
    # maximal and minimal curvatures of the corresponding triangle:
    sg.graph.vp.T_1 = sg.graph.new_vertex_property("vector<float>")
    sg.graph.vp.T_2 = sg.graph.new_vertex_property("vector<float>")
    # vertex properties for storing the estimated maximal and minimal curvatures
    # of the corresponding triangle:
    sg.graph.vp.kappa_1 = sg.graph.new_vertex_property("float")
    sg.graph.vp.kappa_2 = sg.graph.new_vertex_property("float")
    # vertex property for storing the Gaussian curvature calculated from kappa_1
    # and kappa_2 at the corresponding triangle:
    sg.graph.vp.gauss_curvature_VV = sg.graph.new_vertex_property("float")
    # vertex property for storing the mean curvature calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    sg.graph.vp.mean_curvature_VV = sg.graph.new_vertex_property("float")
    # vertex property for storing the shape index calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    sg.graph.vp.shape_index_VV = sg.graph.new_vertex_property("float")
    # vertex property for storing the curvedness calculated from kappa_1 and
    # kappa_2 at the corresponding triangle:
    sg.graph.vp.curvedness_VV = sg.graph.new_vertex_property("float")

    # shortcuts
    vertices = sg.graph.vertices
    vertex = sg.graph.vertex
    collect_curvature_votes = sg.collect_curvature_votes
    gen_curv_vote = sg.gen_curv_vote
    estimate_curvature = sg.estimate_curvature
    orientation_class = sg.graph.vp.orientation_class
    add_curvature_descriptors_to_vertex = sg.add_curvature_descriptors_to_vertex
    graph_to_triangle_poly = sg.graph_to_triangle_poly

    # Estimate principal directions and curvatures (and calculate the
    # Gaussian and mean curvatures, shape index and curvedness) for vertices
    # belonging to a surface patch
    good_vertices_ind = []
    for v in vertices():
        if orientation_class[v] == 1:
            good_vertices_ind.append(int(v))
            # Voting and curvature estimation for SSVV:
            if method == "SSVV":  # sequential processing, edits the graph
                # curvatures saved in the graph, placeholders where error
                gen_curv_vote(poly_surf, v, radius_hit)
        else:  # add placeholders for vertices classified as crease or noise
            add_curvature_descriptors_to_vertex(
                v, None, None, None, None, None, None, None, None)
    print("{} vertices to estimate curvature".format(len(good_vertices_ind)))

    if method == "VV":
        if cores > 1:  # parallel processing
            t_begin2_1 = time.time()
            p = pp.ProcessPool(cores)
            print('Opened a pool with {} processes'.format(cores))

            # Curvature votes collection for VV:
            # None is returned if v does not have any neighbor belonging to
            # a surface patch, then estimate_curvature will return Nones as well
            B_v_list = p.map(partial(
                collect_curvature_votes, g_max=g_max, sigma=sigma,
                page_curvature_formula=page_curvature_formula, A_max=A_max,
                full_dist_map=full_dist_map),
                good_vertices_ind)
            t_end2_1 = time.time()
            duration2_1 = t_end2_1 - t_begin2_1

            t_begin2_2 = time.time()
            # Curvature estimation for VV:
            # results_list has same length as good_vertices_ind
            # columns: T_1, T_2, kappa_1, kappa_2, gauss_curvature,
            # mean_curvature, shape_index, curvedness
            results_list = p.map(partial(estimate_curvature),
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
            t_end2_2 = time.time()
            duration2_2 = t_end2_2 - t_begin2_2

            t_begin2_3 = time.time()
            # Add the curvature descriptors as properties to the graph:
            # (v_ind is vertex v index, i is v_ind index in results arrays)
            for i, v_ind in enumerate(good_vertices_ind):
                v = vertex(v_ind)
                add_curvature_descriptors_to_vertex(
                    v, T_1_array[i], T_2_array[i], kappa_1_array[i],
                    kappa_2_array[i], gauss_curvature_array[i],
                    mean_curvature_array[i], shape_index_array[i],
                    curvedness_array[i])
            t_end2_3 = time.time()
            duration2_3 = t_end2_3 - t_begin2_3

        else:  # cores == 1, sequential processing
            # Curvature votes collection and estimation for VV:
            for i, v_ind in enumerate(good_vertices_ind):
                B_v = collect_curvature_votes(
                    v_ind, g_max, sigma,
                    page_curvature_formula=page_curvature_formula,
                    A_max=A_max, full_dist_map=full_dist_map)
                results = estimate_curvature(v_ind, B_v)
                # Add the properties to the graph:
                v = vertex(v_ind)
                add_curvature_descriptors_to_vertex(v, *results)
            duration2_1 = "NA"
            duration2_2 = "NA"
            duration2_3 = "NA"

    # Transforming the resulting graph to a surface with triangles:
    surface_curv = graph_to_triangle_poly(verbose=False)

    t_end2 = time.time()
    duration2 = t_end2 - t_begin2
    minutes, seconds = divmod(duration2, 60)
    print('Second run of {} took: {} min {} s'.format(method, minutes, seconds))

    # adding to the runtimes CSV file:
    # - method
    # - duration2
    if runtimes != '':
        with open(runtimes, 'a') as f:
            f.write("{};{};{};{};{}\n".format(
                method, duration2, duration2_1, duration2_2, duration2_3))

    return sg, surface_curv
