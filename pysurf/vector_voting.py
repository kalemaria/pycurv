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
        tg, radius_hit, methods=['VV'], page_curvature_formula=False,
        full_dist_map=False, graph_file='temp.gt', area2=True,
        only_normals=False, poly_surf=None, cores=4, runtimes='', pg=None):
    """
    Runs the modified Normal Vector Voting algorithm (with different options for
    the second pass) to estimate surface orientation, principle curvatures and
    directions for a surface using its triangle graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        radius_hit (float): radius in length unit of the graph;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map is
            calculated later for each vertex (default)
        graph_file (string, optional): name for a temporary graph file
            after the first run of the algorithm (default 'temp.gt')
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation; not possible if pg is not None)
        only_normals (boolean, optional): if True (default False), only normals
            are estimated, without principal directions and curvatures, only the
            graph with the orientations, normals or tangents is returned.
        poly_surf (vtkPolyData, optional): surface from which the graph was
            generated, scaled to given units (required only for SSVV, default
            None)
        cores (int, optional): number of cores to run VV in parallel (default 4)
        runtimes (str, optional): if given, runtimes and some parameters are
            added to this file (default '')
        pg (PointGraph, optional): if given (default None), curvature
            is calculated per triangle vertex instead of triangle center.
    Returns:
        a dictionary mapping the method name ('VV' and 'SSVV') to the
        tuple of two elements: TriangleGraph or PointGraph (if pg was given)
        graph and vtkPolyData surface of triangles with classified orientation
        and estimated normals or tangents, principle curvatures and directions
        (if only_normals is False)

    Notes:
        Maximal geodesic neighborhood distance g_max for normal vector voting
        will be derived from radius_hit: g_max = pi * radius_hit / 2
    """
    t_begin = time.time()

    normals_estimation(tg, radius_hit, full_dist_map, cores=cores,
                       runtimes=runtimes, pg=pg)

    preparation_for_curvature_estimation(tg, graph_file, pg=pg)

    if pg is None:
        vertex_based = False
    else:
        vertex_based = True

    if only_normals is False:
        if len(methods) == 1:
            if pg is None:
                sg = tg
            else:
                sg = pg
        else:
            sg = None
        results = {}
        for method in methods:
            tg_curv, surface_curv = curvature_estimation(
                radius_hit, graph_file, method, page_curvature_formula, area2,
                poly_surf=poly_surf, full_dist_map=full_dist_map, cores=cores,
                runtimes=runtimes, vertex_based=vertex_based, sg=sg)
            results[method] = (tg_curv, surface_curv)
        if graph_file == 'temp.gt' and isfile(graph_file):
            remove(graph_file)

        t_end = time.time()
        duration = t_end - t_begin
        minutes, seconds = divmod(duration, 60)
        print('Whole method took: {} min {} s'.format(minutes, seconds))
        return results


def normals_estimation(tg, radius_hit, full_dist_map=False,
                       cores=4, runtimes='', pg=None):
    """
    Runs the modified Normal Vector Voting algorithm to estimate true normals
    for a surface using its triangle graph.

    Adds the estimated normal "N_v" as vertex property into the graph.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
        radius_hit (float): radius in length unit of the graph;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph, otherwise a local distance map is
            calculated later for each vertex (default)
        cores (int): number of cores to run VV (collect_normal_votes and
            estimate_normal) in parallel (default 8)
        runtimes (str): if given, runtimes and some parameters are added to
            this file (default '')
        pg (PointGraph, optional): if given (default None), curvature
            is calculated per triangle vertex instead of triangle center.

    Returns:
        None

    Note:
        Maximal geodesic neighborhood distance g_max for normal vector voting
        will be derived from radius_hit: g_max = pi * radius_hit / 2
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

    if pg is None:
        sg = tg
    else:
        sg = pg

    # * Adding vertex properties to be filled in estimate_normal *
    # vertex property for storing the estimated normal of the corresponding
    # triangle:
    sg.graph.vp.N_v = sg.graph.new_vertex_property("vector<float>")

    t_end0 = time.time()
    duration0 = t_end0 - t_begin0
    minutes, seconds = divmod(duration0, 60)
    print('Preparation took: {} min {} s' .format(minutes, seconds))

    if full_dist_map is True and pg is None:
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
    # average number of the geodesic neighbors and estimating normals*
    print("\nRunning modified Vector Voting for all vertices...")

    print("\nFirst pass: estimating normals...")
    t_begin1 = time.time()

    collect_normal_votes = sg.collect_normal_votes
    estimate_normal = sg.estimate_normal
    num_v = sg.graph.num_vertices()
    print("number of vertices: {}".format(num_v))
    vertex = sg.graph.vertex
    vp_N_v = sg.graph.vp.N_v

    if cores > 1:  # parallel processing
        p = pp.ProcessPool(cores)
        print('Opened a pool with {} processes'.format(cores))

        # output is a list with 2 columns:
        # column 0 = num_neighbors (int)
        # column 1 = V_v (3x3 float array)
        # each row i is of vertex v, its index == i
        # V_v_list = p.map(partial(collect_normal_votes, # if only V_v output
        if pg is None:  # TriangleGraph
            results1_list = p.map(partial(collect_normal_votes,
                                          g_max=g_max, A_max=A_max, sigma=sigma,
                                          full_dist_map=full_dist_map),
                                  range(num_v))  # a list of vertex v indices
        else:  # PointGraph
            results1_list = p.map(partial(collect_normal_votes,
                                          g_max=g_max, A_max=A_max, sigma=sigma,
                                          tg=tg),
                                  range(num_v))  # a list of vertex v indices
        results1_array = np.array(results1_list, dtype=object)
        # Calculating average neighbors number:
        num_neighbors_array = results1_array[:, 0]
        avg_num_neighbors = np.mean(num_neighbors_array)
        # Input of the next parallel calculation:
        V_v_array = results1_array[:, 1]

        # each row i is of vertex v, its index == i
        N_v_list = p.map(partial(estimate_normal),
                         range(num_v), V_v_array)
        #                range(num_v), V_v_list)  # if only V_v output
        p.close()
        p.clear()
        N_v_array = np.array(N_v_list, dtype=object)
        # Adding the estimated normal property to the graph:
        for i in range(num_v):
            v = vertex(i)
            vp_N_v[v] = N_v_array[i]

    else:  # cores == 1, sequential processing
        sum_num_neighbors = 0
        for i in range(num_v):
            if pg is None:  # TriangleGraph
                num_neighbors, V_v = collect_normal_votes(
                    i, g_max, A_max, sigma, full_dist_map=full_dist_map)
            else:  # PointGraph
                num_neighbors, V_v = collect_normal_votes(
                    i, g_max, A_max, sigma, tg=tg)
            sum_num_neighbors += num_neighbors
            N_v = estimate_normal(i, V_v)
            # Adding the estimated normal property to the graph:
            v = tg.graph.vertex(i)
            vp_N_v[v] = N_v

        avg_num_neighbors = float(sum_num_neighbors) / float(num_v)

    # Printing out some numbers concerning the first pass:
    print("Average number of geodesic neighbors for all vertices: {}".format(
        avg_num_neighbors))

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    minutes, seconds = divmod(duration1, 60)
    print('First pass took: {} min {} s'.format(minutes, seconds))

    if runtimes != '':
        with open(runtimes, 'a') as f:
            f.write("{};{};{};{};{};{};".format(
                num_v, radius_hit, g_max, avg_num_neighbors, cores, duration1))


def preparation_for_curvature_estimation(
        tg, graph_file='temp.gt', pg=None):
    """
    Does preparation for principal directions and curvature estimation after the
    surface orientation estimation (second part used by
    normals_directions_and_curvature_estimation): adds vertex properties to be
    filled by all curvature methods and saves the graph to a file.

    Args:
        tg (TriangleGraph): triangle graph generated from a surface of interest
            graph_file (str): file path to save the graph
        pg (PointGraph, optional): if given (default None), curvature
            is calculated per triangle vertex instead of triangle center.

    Returns:
        None
    """
    if pg is None:
        sg = tg
    else:
        sg = pg

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
            calculated for the whole graph, otherwise a local distance map is
            calculated later for each vertex (default)
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
            # cannot weight by triangle area in vertex-based approach
            area2 = False
        else:
            sg = TriangleGraph()
        sg.graph = load_graph(graph_file)

    if full_dist_map is True:
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

    print("\nSecond pass: estimating principle curvatures and directions for "
          "surface patches using {}...".format(method))
    t_begin2 = time.time()

    collect_curvature_votes = sg.collect_curvature_votes
    gen_curv_vote = sg.gen_curv_vote
    estimate_curvature = sg.estimate_curvature
    g_max = math.pi * radius_hit / 2.0
    sigma = g_max / 3.0
    if method == "VV" and area2:
        A, _ = sg.get_areas()
        A = np.array(A)
        A_max = np.max(A)
        print("Maximal triangle area = {}".format(A_max))
    else:
        A_max = 0.0

    # Estimate principal directions and curvatures (and calculate the
    # Gaussian and mean curvatures, shape index and curvedness) for vertices
    vertices_ind = []
    for v in sg.graph.vertices():
        vertices_ind.append(int(v))
        # Voting and curvature estimation for SSVV:
        if method == "SSVV":  # sequential processing, edits the graph
            # curvatures saved in the graph, placeholders where error
            gen_curv_vote(poly_surf, v, radius_hit)
    print("{} vertices to estimate curvature".format(len(vertices_ind)))

    if method == "VV":
        if cores > 1:  # parallel processing
            p = pp.ProcessPool(cores)
            print('Opened a pool with {} processes'.format(cores))

            # Curvature votes collection for VV:
            # None is returned if v does not have any neighbors, then
            # estimate_curvature will return Nones as well
            B_v_list = p.map(partial(collect_curvature_votes,
                                     g_max=g_max, sigma=sigma,
                                     page_curvature_formula=page_curvature_formula,
                                     A_max=A_max, full_dist_map=full_dist_map),
                             vertices_ind)
            # Curvature estimation for VV:
            # results_list has same length as vertices_ind
            # columns: T_1, T_2, kappa_1, kappa_2, gauss_curvature,
            # mean_curvature, shape_index, curvedness
            results_list = p.map(partial(estimate_curvature),
                                 vertices_ind, B_v_list)
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
            # Add the curvature descriptors as properties to the graph:
            # (v_ind is vertex v index, i is v_ind index in results arrays)
            for i, v_ind in enumerate(vertices_ind):
                v = sg.graph.vertex(v_ind)
                sg._add_curvature_descriptors_to_vertex(
                    v, T_1_array[i], T_2_array[i], kappa_1_array[i],
                    kappa_2_array[i], gauss_curvature_array[i],
                    mean_curvature_array[i], shape_index_array[i],
                    curvedness_array[i])

        else:  # cores == 1, sequential processing
            # Curvature votes collection and estimation for VV:
            for i, v_ind in enumerate(vertices_ind):
                B_v = collect_curvature_votes(
                    v_ind, g_max, sigma,
                    page_curvature_formula=page_curvature_formula,
                    A_max=A_max, full_dist_map=full_dist_map)
                results = estimate_curvature(v_ind, B_v)
                # Add the properties to the graph:
                v = sg.graph.vertex(v_ind)
                sg._add_curvature_descriptors_to_vertex(v, *results)

    # Transforming the resulting graph to a surface with triangles:
    surface_curv = sg.graph_to_triangle_poly(verbose=False)

    t_end2 = time.time()
    duration2 = t_end2 - t_begin2
    minutes, seconds = divmod(duration2, 60)
    print('Second run of {} took: {} min {} s'.format(method, minutes, seconds))

    # adding to the runtimes CSV file:
    # - method
    # - duration2
    if runtimes != '':
        with open(runtimes, 'a') as f:
            f.write("{};{}\n".format(method, duration2))

    return sg, surface_curv
