import sys
import time
import numpy as np
import math
from graph_tool import load_graph
from graph_tool.topology import shortest_distance
import pathos.pools as pp
from functools import partial
from os import remove
from os.path import isfile
from tqdm import tqdm
from scipy import sparse
import tempfile
import os

from .surface_graphs import TriangleGraph, PointGraph

# Worker-local cache for loaded graphs (avoids reloading for each chunk)
_worker_graph_cache = {}


def _chunk_indices(indices, chunk_size):
    """Split indices into chunks for batched parallel processing.

    Args:
        indices: Iterable of indices to chunk
        chunk_size: Maximum size of each chunk

    Returns:
        List of lists, each containing up to chunk_size indices
    """
    indices = list(indices)
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]


def _get_cached_graph(graph_file, graph_class):
    """Load graph from file, caching in worker-local storage.

    Each worker process loads the graph once and reuses it for all chunks.
    This avoids the overhead of pickling the graph with each task.
    """
    global _worker_graph_cache
    cache_key = (graph_file, graph_class)

    if cache_key not in _worker_graph_cache:
        if graph_class == 'TriangleGraph':
            sg = TriangleGraph()
        else:
            sg = PointGraph()
        sg.graph = load_graph(graph_file)
        _worker_graph_cache[cache_key] = sg

    return _worker_graph_cache[cache_key]


def _process_vertex_chunk_first_pass(vertex_indices, sg, g_max, a_max, sigma, full_dist_map):
    """Process a chunk of vertices for the first pass (normal estimation).

    Args:
        vertex_indices: List of vertex indices to process
        sg: Surface graph object OR tuple (graph_file, graph_class) for lazy loading
        g_max: Maximum geodesic distance
        a_max: Maximum triangle area
        sigma: Sigma parameter for weighting
        full_dist_map: Full distance map or None

    Returns:
        List of (num_neighbors, class_v, n_v, t_v) tuples for each vertex
    """
    # Support lazy loading from file to reduce IPC overhead
    if isinstance(sg, tuple):
        graph_file, graph_class = sg
        sg = _get_cached_graph(graph_file, graph_class)

    results = []
    for idx in vertex_indices:
        result = sg.first_pass(idx, g_max=g_max, a_max=a_max,
                               sigma=sigma, full_dist_map=full_dist_map)
        results.append(result)
    return results


def _process_vertex_chunk_second_pass(vertex_indices, sg, g_max, sigma,
                                       page_curvature_formula, a_max, full_dist_map):
    """Process a chunk of vertices for the second pass (curvature estimation).

    Args:
        vertex_indices: List of vertex indices to process
        sg: Surface graph object OR tuple (graph_file, graph_class) for lazy loading
        g_max: Maximum geodesic distance
        sigma: Sigma parameter for weighting
        page_curvature_formula: Whether to use Page's curvature formula
        a_max: Maximum triangle area (0.0 to disable area weighting)
        full_dist_map: Full distance map or None

    Returns:
        List of (t_1, t_2, kappa_1, kappa_2, gauss, mean, shape, curvedness) tuples
    """
    # Support lazy loading from file to reduce IPC overhead
    if isinstance(sg, tuple):
        graph_file, graph_class = sg
        sg = _get_cached_graph(graph_file, graph_class)

    results = []
    for idx in vertex_indices:
        result = sg.second_pass(idx, g_max=g_max, sigma=sigma,
                                page_curvature_formula=page_curvature_formula,
                                a_max=a_max, full_dist_map=full_dist_map)
        results.append(result)
    return results

"""
Contains a function implementing the normal vector voting algorithm (Page et
al., 2002) with adaptation.

The main differences to the original method are usage of a graph (TriangleGraph
object) representing the underlying surface and approximating geodesic distances
by shortest distances along the graph edges. Uses several methods of the
TriangleGraph class in the surface_graphs module, which implement triangle-wise
operations of the algorithm.

Author: Maria Salfer (Max Planck Institute for Biochemistry),
date: 2017-06-17
"""

__author__ = 'Maria Salfer'


def normals_directions_and_curvature_estimation(
        sg, radius_hit, epsilon=0, eta=0, methods=['VV'],
        page_curvature_formula=False, full_dist_map=False, graph_file='temp.gt',
        area2=True, only_normals=False, poly_surf=None, cores=10, runtimes=''):
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
        cores (int, optional): number of cores to run VV in parallel (default 10)
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
                       cores=10, runtimes='', graph_file='temp.gt'):
    """
    Runs the modified Normal Vector Voting algorithm to estimate surface
    orientation (classification in surface patch with normal, crease junction
    with tangent or no preferred orientation) for a surface using its graph.

    Adds the "orientation_class" (1-3), the estimated normal "n_v" (if class is
    1) and the estimated_tangent "t_v" (if class is 2) as vertex properties
    into the graph.

    Adds the estimated normal "n_v" as vertex property into the graph.

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
            and estimate_normal) in parallel (default 10)
        runtimes (str, optional): if given, runtimes and some parameters are
            added to this file (default '')
        graph_file (str, optional): file path to save the graph, default file
            'temp.gt'

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
    a_max = sg.graph.gp.max_triangle_area
    print("Maximal triangle area = {}".format(a_max))

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
    sg.graph.vp.n_v = sg.graph.new_vertex_property("vector<float>")
    # vertex property for storing the estimated tangent of the corresponding
    # vertex (if the vertex belongs to class 2):
    sg.graph.vp.t_v = sg.graph.new_vertex_property("vector<float>")
    # vertex property for storing local average of estimated normals
    sg.graph.vp.avg_normals = sg.graph.new_vertex_property('vector<float>')

    if full_dist_map is True and sg.__class__.__name__ == "TriangleGraph":
        # * Distance map between all pairs of vertices *
        t_begin0_2 = time.time()
        print("\nCalculating the full distance map...")
        full_dist_map = shortest_distance(  # <class 'graph_tool.PropertyMap'>
            sg.graph, weights=sg.graph.ep.distance)
        t_end0_2 = time.time()
        duration0_2 = t_end0_2 - t_begin0_2
        minutes, seconds = divmod(duration0_2, 60)
        print('Calculation of the full distance map took: {} min {} s'.format(
            minutes, seconds))
    else:
        full_dist_map = None

    # Initialize distance cache for efficient neighbor search on large graphs
    # The distance cache stores Dijkstra results to reduce redundant computations
    graph_num_vertices = sg.graph.num_vertices()

    t_end0 = time.time()
    duration0 = t_end0 - t_begin0
    minutes, seconds = divmod(duration0, 60)
    print('Preparation took: {} min {} s'.format(minutes, seconds))

    # * For all vertices, collecting normal vector votes, calculating
    # average number of the geodesic neighbors and classifying the orientation
    # of each vertex *
    t_begin1 = time.time()
    print("\nRunning modified Vector Voting for all vertices...")

    if eta == 0 and epsilon == 0:
        print("\nFirst pass: estimating normals...")
    else:
        print("\nFirst pass: classifying orientation and estimating normals for"
              " surface patches and tangents for creases...")

    collect_normal_votes = sg.collect_normal_votes
    estimate_normal = sg.estimate_normal
    first_pass = sg.first_pass
    num_v = sg.graph.num_vertices()
    print("number of vertices: {}".format(num_v))
    classes_counts = {}
    vertex = sg.graph.vertex
    vp_orientation_class = sg.graph.vp.orientation_class
    vp_n_v = sg.graph.vp.n_v
    vp_t_v = sg.graph.vp.t_v

    if cores > 1:  # parallel processing
        # Save graph to temp file - workers load from file instead of pickling
        # This significantly reduces IPC overhead for large graphs
        temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix='.gt')
        os.close(temp_graph_fd)
        sg.graph.save(temp_graph_file)
        graph_class = sg.__class__.__name__
        sg_ref = (temp_graph_file, graph_class)  # lightweight reference

        p = pp.ProcessPool(cores)
        print('Opened a pool with {} processes'.format(cores))

        # Use chunked processing to reduce IPC overhead
        # Each chunk is processed as a batch, reducing task count from num_v to ~num_v/chunk_size
        chunk_size = max(100, num_v // (cores * 6))
        vertex_chunks = _chunk_indices(range(num_v), chunk_size)
        print('Processing {} vertices in {} chunks of ~{} vertices each'.format(
            num_v, len(vertex_chunks), chunk_size))

        # Explicit progress message for stderr (shows even if tqdm doesn't render)
        print('First pass: processing {} chunks...'.format(
            len(vertex_chunks)), file=sys.stderr)

        chunk_results = list(tqdm(
            p.imap(
                partial(_process_vertex_chunk_first_pass, sg=sg_ref,
                        g_max=g_max, a_max=a_max, sigma=sigma,
                        full_dist_map=full_dist_map),
                vertex_chunks),
            total=len(vertex_chunks),
            desc="First pass",
            unit="chunk",
            file=sys.stderr))
        p.close()
        p.clear()
        print('First pass: completed {} chunks'.format(len(vertex_chunks)), file=sys.stderr)

        # Clean up temp file
        if isfile(temp_graph_file):
            remove(temp_graph_file)

        # Flatten chunk results into a single list
        results_list = [result for chunk in chunk_results for result in chunk]
        results_array = np.array(results_list, dtype=object)
        # Calculating average neighbors number:
        num_neighbors_array = results_array[:, 0]
        avg_num_neighbors = np.mean(num_neighbors_array)
        class_v_array = results_array[:, 1]
        n_v_array = results_array[:, 2]
        t_v_array = results_array[:, 3]

        # Adding the estimated properties to the graph and counting classes:
        for i in range(num_v):
            v = vertex(i)
            vp_orientation_class[v] = class_v_array[i]
            vp_n_v[v] = n_v_array[i]
            vp_t_v[v] = t_v_array[i]
            try:
                classes_counts[class_v_array[i]] += 1
            except KeyError:
                classes_counts[class_v_array[i]] = 1

    else:  # cores == 1, sequential processing
        sum_num_neighbors = 0
        for i in tqdm(range(num_v), desc="First pass", unit="vtx", file=sys.stderr):
            if sg.__class__.__name__ == "TriangleGraph":
                num_neighbors, V_v = collect_normal_votes(
                    i, g_max, a_max, sigma, full_dist_map=full_dist_map)
            else:  # PointGraph
                num_neighbors, V_v = collect_normal_votes(
                    i, g_max, a_max, sigma)
            sum_num_neighbors += num_neighbors
            class_v, n_v, t_v = estimate_normal(i, V_v, epsilon, eta)
            # Adding the estimated properties to the graph and counting classes:
            v = vertex(i)
            vp_orientation_class[v] = class_v
            vp_n_v[v] = n_v
            vp_t_v[v] = t_v
            try:
                classes_counts[class_v] += 1
            except KeyError:
                classes_counts[class_v] = 1
        avg_num_neighbors = float(sum_num_neighbors) / float(num_v)
    
    # Post processing to correct orientation of normals
    # Fully vectorized using sparse matrix operations
    print("\nPost-processing: correcting normal orientations...")

    # Extract all normals into numpy array (N x 3)
    normals = np.array([vp_n_v[v] for v in sg.graph.vertices()])

    # Build sparse adjacency matrix with distance filtering
    # Only include edges where distance <= g_max
    rows = []
    cols = []
    ep_distance = sg.graph.ep.distance
    for e in sg.graph.edges():
        dist = ep_distance[e]
        if dist <= g_max:
            src, tgt = int(e.source()), int(e.target())
            rows.extend([src, tgt])
            cols.extend([tgt, src])

    # Create sparse adjacency matrix
    data = np.ones(len(rows), dtype=np.float32)
    adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_v, num_v))

    # Row-normalize: divide each row by number of neighbors
    row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    inv_row_sums = 1.0 / row_sums
    normalizer = sparse.diags(inv_row_sums)
    adj_normalized = normalizer @ adj_matrix

    # Compute average normals via sparse matrix multiplication
    avg_normals = adj_normalized @ normals  # (N x 3)

    # Normalize the averaged normals
    norms = np.linalg.norm(avg_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    avg_normals = avg_normals / norms

    # Compute dot products between original and average normals
    dot_products = np.sum(normals * avg_normals, axis=1)

    # Flip normals where dot product is negative
    flip_mask = dot_products < 0
    normals[flip_mask] *= -1
    num_flipped = np.sum(flip_mask)
    print(f"Flipped {num_flipped} normals for consistency")

    # Write back to graph properties
    vp_avg_normals = sg.graph.vp.avg_normals
    for i, v in enumerate(sg.graph.vertices()):
        vp_n_v[v] = normals[i]
        vp_avg_normals[v] = avg_normals[i]

    # Printing out some numbers concerning the first pass:
    print("Average number of geodesic neighbors for all vertices: {}".format(
        avg_num_neighbors))
    print("{} surface patches".format(classes_counts[1]))
    if 2 in classes_counts:
        print("{} crease junctions".format(classes_counts[2]))
    if 3 in classes_counts:
        print("{} no preferred orientation".format(classes_counts[3]))

    # Save the graph to a file for use by different methods in the second run:
    sg.graph.save(graph_file)

    t_end1 = time.time()
    duration1 = t_end1 - t_begin1
    minutes, seconds = divmod(duration1, 60)
    print('First pass took: {} min {} s'.format(minutes, seconds))

    if runtimes != '':
        with open(runtimes, 'a') as f:
            f.write("{};{};{};{};{};{};".format(
                num_v, radius_hit, g_max, avg_num_neighbors, cores, duration1))


def curvature_estimation(
        radius_hit, graph_file='temp.gt', method='VV',
        page_curvature_formula=False, area2=True, poly_surf=None,
        full_dist_map=False, cores=10, runtimes='', vertex_based=False, sg=None):
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
            estimate_curvature) in parallel (default 10)
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
    # Preparation (calculations that are the same for the whole graph)
    t_begin0 = time.time()
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
        t_begin0_2 = time.time()
        print("\nCalculating the full distance map...")
        full_dist_map = shortest_distance(  # <class 'graph_tool.PropertyMap'>
            sg.graph, weights=sg.graph.ep.distance)
        t_end0_2 = time.time()
        duration0_2 = t_end0_2 - t_begin0_2
        minutes, seconds = divmod(duration0_2, 60)
        print('Calculation of the full distance map took: {} min {} s'.format(
            minutes, seconds))
    else:
        full_dist_map = None

    g_max = math.pi * radius_hit / 2.0
    sigma = g_max / 3.0
    if method == "VV" and area2:
        a_max = sg.graph.gp.max_triangle_area
        print("Maximal triangle area = {}".format(a_max))
    else:
        a_max = 0.0

    # * Adding vertex properties to be filled by all curvature methods *
    # vertex properties for storing the estimated principal directions of the
    # maximal and minimal curvatures of the corresponding triangle:
    sg.graph.vp.t_1 = sg.graph.new_vertex_property("vector<float>")
    sg.graph.vp.t_2 = sg.graph.new_vertex_property("vector<float>")
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

    t_end0 = time.time()
    duration0 = t_end0 - t_begin0
    minutes, seconds = divmod(duration0, 60)
    print('Preparation took: {} min {} s'.format(minutes, seconds))

    t_begin2 = time.time()
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

    # shortcuts
    vertices = sg.graph.vertices
    vertex = sg.graph.vertex
    collect_curvature_votes = sg.collect_curvature_votes
    gen_curv_vote = sg.gen_curv_vote
    estimate_curvature = sg.estimate_curvature
    second_pass = sg.second_pass
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
            # Save graph to temp file - workers load from file instead of pickling
            # This significantly reduces IPC overhead for large graphs
            temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix='.gt')
            os.close(temp_graph_fd)
            sg.graph.save(temp_graph_file)
            graph_class = sg.__class__.__name__
            sg_ref = (temp_graph_file, graph_class)  # lightweight reference

            p = pp.ProcessPool(cores)
            print('Opened a pool with {} processes'.format(cores))

            # Use chunked processing to reduce IPC overhead
            num_good = len(good_vertices_ind)
            chunk_size = max(100, num_good // (cores * 6))
            vertex_chunks = _chunk_indices(good_vertices_ind, chunk_size)
            print('Processing {} vertices in {} chunks of ~{} vertices each'.format(
                num_good, len(vertex_chunks), chunk_size))

            # Explicit progress message for stderr (shows even if tqdm doesn't render)
            print('Second pass: processing {} chunks...'.format(
                len(vertex_chunks)), file=sys.stderr)

            # results_list has same length as good_vertices_ind
            # columns: t_1, t_2, kappa_1, kappa_2, gauss_curvature,
            # mean_curvature, shape_index, curvedness
            chunk_results = list(tqdm(
                p.imap(
                    partial(_process_vertex_chunk_second_pass, sg=sg_ref,
                            g_max=g_max, sigma=sigma,
                            page_curvature_formula=page_curvature_formula,
                            a_max=a_max, full_dist_map=full_dist_map),
                    vertex_chunks),
                total=len(vertex_chunks),
                desc="Second pass",
                unit="chunk",
                file=sys.stderr))
            p.close()
            p.clear()
            print('Second pass: completed {} chunks'.format(len(vertex_chunks)), file=sys.stderr)

            # Clean up temp file
            if isfile(temp_graph_file):
                remove(temp_graph_file)

            # Flatten chunk results into a single list
            results_list = [result for chunk in chunk_results for result in chunk]
            results_array = np.array(results_list, dtype=object)
            t_1_array = results_array[:, 0]
            t_2_array = results_array[:, 1]
            kappa_1_array = results_array[:, 2]
            kappa_2_array = results_array[:, 3]
            gauss_curvature_array = results_array[:, 4]
            mean_curvature_array = results_array[:, 5]
            shape_index_array = results_array[:, 6]
            curvedness_array = results_array[:, 7]

            # Add the curvature descriptors as properties to the graph:
            # (v_ind is vertex v index, i is v_ind index in results arrays)
            for i, v_ind in enumerate(good_vertices_ind):
                v = vertex(v_ind)
                add_curvature_descriptors_to_vertex(
                    v, t_1_array[i], t_2_array[i], kappa_1_array[i],
                    kappa_2_array[i], gauss_curvature_array[i],
                    mean_curvature_array[i], shape_index_array[i],
                    curvedness_array[i])

        else:  # cores == 1, sequential processing
            # Curvature votes collection and estimation for VV:
            for i, v_ind in tqdm(enumerate(good_vertices_ind), total=len(good_vertices_ind),
                                 desc="Second pass", unit="vtx", file=sys.stderr):
                B_v = collect_curvature_votes(
                    v_ind, g_max, sigma,
                    page_curvature_formula=page_curvature_formula,
                    a_max=a_max, full_dist_map=full_dist_map)
                results = estimate_curvature(v_ind, B_v)
                # Add the properties to the graph:
                v = vertex(v_ind)
                add_curvature_descriptors_to_vertex(v, *results)

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
            f.write("{};{}\n".format(method_print, duration2))

    return sg, surface_curv
