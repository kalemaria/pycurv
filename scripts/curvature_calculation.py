import sys
import time
from os.path import isfile
from graph_tool import load_graph
import gzip
from os import remove
import pandas as pd
import numpy as np
from scipy import ndimage
import vtk  # for Anna's workflow
# import cProfile

from pysurf import (
    pexceptions, normals_directions_and_curvature_estimation, run_gen_surface,
    TriangleGraph, preparation_for_curvature_estimation, curvature_estimation)
from pysurf import pysurf_io as io

"""
A script with an example application of the PySurf package for estimation of
membrane curvature.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


# CONSTANTS
MAX_DIST_SURF = 3
"""int: a constant determining the maximal distance in pixels of a point on the
surface from the segmentation mask, used in gen_isosurface and gen_surface
functions.
"""
THRESH_SIGMA1 = 0.699471735
""" float: when convoluting a binary mask with a gaussian kernel with sigma 1,
values 1 at the boundary with 0's become this value.
"""


def convert_vtp_to_stl_surface_and_mrc_curvatures(
        surf_vtp_file, outfile_base, pixel_size, scale_x, scale_y, scale_z):
    """
    Converts the '.vtp' surface file to '.stl' file and converts selected
    vtkPolyData cell arrays from the '.vtp' file as 3-D volumes in '.mrc' files.
    The selected arrays are: "kappa_1", "kappa_2", "curvedness_VV".

    Args:
        surf_vtp_file (str): surface .vtp file, should contain the final surface
            with curvatures
        outfile_base (str): base name for the output .mrc and .log files
        pixel_size (float): pixel size in nanometer of the membrane mask
        scale_x (int): size of the membrane mask in X dimension
        scale_y (int): size of the membrane mask in Y dimension
        scale_z (int): size of the membrane mask in Z dimension

    Returns:
        None
    """
    # Converting the '.vtp' surface file to '.stl' file:
    surf_stl_file = (surf_vtp_file[0:-4] + '.stl')
    if not isfile(surf_stl_file):
        io.vtp_file_to_stl_file(surf_vtp_file, surf_stl_file)
        print("The '.vtp' file {} was converted to .stl format".format(
            surf_vtp_file))

    # Converting vtkPolyData selected cell arrays from the '.vtp' file as 3-D
    # volumes in '.mrc' files (and saving them as '.mrc.gz' files).
    # max voxel value & .log files:
    _vtp_arrays_to_mrc_volumes(
        surf_vtp_file, outfile_base, pixel_size, scale_x, scale_y, scale_z,
        log_files=True)
    # mean voxel value & no .log files:
    _vtp_arrays_to_mrc_volumes(
        surf_vtp_file, outfile_base, pixel_size, scale_x, scale_y, scale_z,
        mean=True)


def _vtp_arrays_to_mrc_volumes(
        surf_vtp_file, outfile_base, pixel_size, scale_x, scale_y, scale_z,
        mean=False, log_files=False, compress=False):
    """
    This function converts selected vtkPolyData cell arrays from the '.vtp' file
    as 3-D volumes in '.mrc' files. The selected arrays are: "kappa_1",
    "kappa_2", "curvedness_VV".

    Args:
        surf_vtp_file (str): surface .vtp file, should contain the final surface
            with curvatures
        outfile_base (str): base name for the output .mrc (and .log) files
        pixel_size (float): pixel size in nanometer of the membrane mask
        scale_x (int): size of the membrane mask in X dimension
        scale_y (int): size of the membrane mask in Y dimension
        scale_z (int): size of the membrane mask in Z dimension
        mean (boolean, optional): if True (default False), in case multiple
            triangles map to the same voxel, takes the mean value, else the
            maximal value
        log_files (boolean, optional): if True (default False), writes the log
            files for such cases
        compress (boolean, optional): if True (default False), compresses the
            '.mrc' files with gzip.

    Returns:
        None
    """
    array_names = ["kappa_1", "kappa_2", "curvedness_VV"]
    names = ["max_curvature", "min_curvature", "curvedness"]

    if mean:
        voxel_value_str = "voxel_mean"
    else:
        voxel_value_str = "voxel_max"
    mrcfilenames = []
    logfilenames = []
    for name in names:
        mrcfilename = "{}.{}.{}.mrc".format(outfile_base, name, voxel_value_str)
        mrcfilenames.append(mrcfilename)
        if log_files:
            logfilename = "{}.{}.{}.log".format(outfile_base, name,
                                                voxel_value_str)
        else:
            logfilename = None
        logfilenames.append(logfilename)

    # Load the vtkPolyData object from the '.vtp' file, calculate the volumes
    # from arrays, write '.log' files, and save the volumes as '.mrc' files:
    poly = io.load_poly(surf_vtp_file)
    for i, array_name in enumerate(array_names):
        volume = io.poly_array_to_volume(
            poly, array_name, pixel_size, scale_x, scale_y, scale_z,
            logfilename=logfilenames[i], mean=mean)
        io.save_numpy(volume, mrcfilenames[i])

    if compress:
        # Gunzip the '.mrc' files and delete the uncompressed files:
        for mrcfilename in mrcfilenames:
            with open(mrcfilename) as f_in, \
                    gzip.open(mrcfilename + '.gz', 'wb') as f_out:
                f_out.writelines(f_in)
            remove(mrcfilename)
            print('Archive {}.gz was written'.format(mrcfilename))


def new_workflow(
        fold, base_filename, pixel_size_nm, radius_hit,
        epsilon=0, eta=0, methods=['VV'],
        seg_file=None, label=1, holes=0, remove_wrong_borders=True,
        min_component=100, only_normals=False, cores=4,
        runtimes=None):
    """
    A script for running all processing steps to estimate membrane curvature.

    The three steps are: 1. signed surface generation, 2. surface cleaning using
    a graph, 3. curvature calculation using a graph generated from the clean
    surface.

    It was written for Javier's data. Segmentation is not split into regions.
    Step 3. of VV, consisting of normals and curvature calculations, can run in
    parallel on multiple cores.

    Args:
        fold (str): path where the input membrane segmentation is and where the
            output will be written
        base_filename (str): base file name for saving the output files
        pixel_size_nm (float): pixel size in nanometer of the membrane mask
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3), default 0
        methods (list, optional): all methods to run in the second pass ('VV',
            'VVCF' and 'VCTV' are possible, default is 'VV')
        seg_file (str, optional): membrane segmentation mask
        label (int, optional): label to be considered in the membrane mask
            (default 1)
        holes (int, optional): if > 0, small holes in the segmentation are
            closed with a cube of that size in pixels before curvature
            estimation (default 0)
        remove_wrong_borders (boolean, optional): if True (default), wrong
            artefact surface borders will be removed
        min_component (int, optional): if > 0 (default 100), small
            disconnected surface components having triangles within this number
            will be removed
        only_normals (boolean, optional): if True (default False), only normals
            are estimated, without principal directions and curvatures, only the
            graph with the orientations class, normals or tangents is returned.
        cores (int, optional): number of cores to run VV in parallel (default 4)
        runtimes (str, optional): if given, runtimes and some parameters are
            added to this file (default None)

    Returns:
        None
    """
    log_file = '{}{}.{}_rh{}_epsilon{}_eta{}.log'.format(
                fold, base_filename, methods[0], radius_hit, epsilon, eta)
    sys.stdout = open(log_file, 'a')

    t_begin = time.time()

    surf_file = base_filename + ".surface.vtp"
    if not isfile(fold + surf_file):
        if seg_file is None or not isfile(fold + seg_file):
            text = "The segmentation file not given or not found"
            raise pexceptions.PySegInputError(
                expr="new_workflow", msg=eval(text))

        seg = io.load_tomo(fold + seg_file)
        assert(isinstance(seg, np.ndarray))
        data_type = seg.dtype

        if label == 2:  # and np.max(seg) == 4:  if cER (and filled seg. exists)
            # Surface generation with filled segmentation using vtkMarchingCubes
            # and applying the mask of unfilled segmentation
            print("\nMaking filled and unfilled binary segmentations...")
            # have to combine the outer and inner seg. for the filled one:
            filled_binary_seg = np.logical_or(seg == 2, seg == 3).astype(
                data_type)
            binary_seg = (seg == 2).astype(data_type)
            print("\nGenerating a surface...")
            surf = run_gen_surface(
                filled_binary_seg, fold + base_filename, lbl=1,
                other_mask=binary_seg, isosurface=True, sg=1, thr=THRESH_SIGMA1)
            # Write the resulting binary segmentations into a file:
            filled_binary_seg_file = "{}{}.filled_binary_seg.mrc".format(
                fold, base_filename)
            io.save_numpy(filled_binary_seg, filled_binary_seg_file)
            binary_seg_file = "{}{}.binary_seg.mrc".format(fold, base_filename)
            io.save_numpy(binary_seg, binary_seg_file)

        else:  # Surface generation with vtkSurfaceReconstructionFilter method
            # Load the segmentation numpy array from a file and get only the
            # requested labels as 1 and the background as 0:
            print("\nMaking the segmentation binary...")
            binary_seg = (seg == label).astype(data_type)
            if holes != 0:  # reduce / increase holes in the segmentation
                cube_size = abs(holes)
                cube = np.ones((cube_size, cube_size, cube_size))
                if holes > 0:  # close (reduce) holes
                    print("\nReducing holes in the segmentation...")
                    binary_seg = ndimage.binary_closing(
                        binary_seg, structure=cube, iterations=1).astype(
                        data_type)
                    # Write the resulting binary segmentation into a file:
                    binary_seg_file = "{}{}.binary_seg.mrc".format(
                        fold, base_filename)
                    io.save_numpy(binary_seg, binary_seg_file)
            print("\nGenerating a surface from the binary segmentation...")
            surf = run_gen_surface(binary_seg, fold + base_filename, lbl=1)
    else:
        print('\nReading in the surface from file...')
        surf = io.load_poly(fold + surf_file)

    clean_graph_file = '{}.scaled_cleaned.gt'.format(base_filename)
    clean_surf_file = '{}.scaled_cleaned.vtp'.format(base_filename)
    if not isfile(fold + clean_graph_file) or not isfile(fold + clean_surf_file):
        print('\nBuilding a triangle graph from the surface...')
        tg = TriangleGraph()
        tg.build_graph_from_vtk_surface(surf, pixel_size_nm)
        print('The graph has {} vertices and {} edges'.format(
            tg.graph.num_vertices(), tg.graph.num_edges()))

        # Remove the wrong borders (surface generation artefact)
        b = 0
        if remove_wrong_borders:
            b += MAX_DIST_SURF  # "padding" from masking in surface generation
        # if holes < 0:
        #     b += abs(holes)
        if b > 0:
            print('\nFinding triangles that are {} pixels to surface borders...'
                  .format(b))
            tg.find_vertices_near_border(b * pixel_size_nm, purge=True)
            print('The graph has {} vertices and {} edges'.format(
                tg.graph.num_vertices(), tg.graph.num_edges()))

        # Filter out possibly occurring small disconnected fragments
        if min_component > 0:
            print('\nFinding small connected components of the graph...')
            tg.find_small_connected_components(
                threshold=min_component, purge=True, verbose=True)
            print('The graph has {} vertices and {} edges'.format(
                tg.graph.num_vertices(), tg.graph.num_edges()))

        # Saving the scaled (and cleaned) graph and surface:
        tg.graph.save(fold + clean_graph_file)
        surf_clean = tg.graph_to_triangle_poly()
        io.save_vtp(surf_clean, fold + clean_surf_file)
    else:
        print('\nReading in the cleaned graph and surface from files...')
        surf_clean = io.load_poly(fold + clean_surf_file)
        tg = TriangleGraph()
        tg.graph = load_graph(fold + clean_graph_file)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Surface and graph generation (and cleaning) took: {} min {} s'
          .format(minutes, seconds))

    # Running the modified Normal Vector Voting algorithms:
    gt_file1 = '{}{}.NVV_rh{}_epsilon{}_eta{}.gt'.format(
            fold, base_filename, radius_hit, epsilon, eta)
    method_tg_surf_dict = {}
    if not isfile(gt_file1):
        if runtimes is not None:
            with open(runtimes, 'w') as f:
                f.write("num_v;radius_hit;g_max;avg_num_neighbors;cores;"
                        "duration1;method;duration2\n")
        method_tg_surf_dict = normals_directions_and_curvature_estimation(
            tg, radius_hit, epsilon=epsilon, eta=eta, exclude_borders=0,
            methods=methods, full_dist_map=False, graph_file=gt_file1,
            area2=True, only_normals=only_normals, poly_surf=surf_clean,
            cores=cores, runtimes=runtimes)
    elif only_normals is False:
        if runtimes is not None:
            with open(runtimes, 'w') as f:
                f.write("method;duration2\n")
        for method in methods:
            tg_curv, surface_curv = curvature_estimation(
                radius_hit, exclude_borders=0, graph_file=gt_file1,
                method=method, poly_surf=surf_clean, cores=cores,
                runtimes=runtimes)
            method_tg_surf_dict[method] = (tg_curv, surface_curv)

    if only_normals is False:
        for method in method_tg_surf_dict.keys():
            # Saving the output (graph and surface object) for later
            # filtering or inspection in ParaView:
            (tg, surf) = method_tg_surf_dict[method]
            if method == 'VV':
                method = 'VV_area2'
            gt_file = '{}{}.{}_rh{}_epsilon{}_eta{}.gt'.format(
                fold, base_filename, method, radius_hit, epsilon, eta)
            tg.graph.save(gt_file)
            surf_file = '{}{}.{}_rh{}_epsilon{}_eta{}.vtp'.format(
                fold, base_filename, method, radius_hit, epsilon, eta)
            io.save_vtp(surf, surf_file)


def calculate_PM_curvatures(fold, base_filename, radius_hit, cores=4):
    gt_file_normals = "{}{}.NVV_rh{}.gt".format(fold, base_filename, radius_hit)
    tg = TriangleGraph()
    tg.graph = load_graph(gt_file_normals)

    preparation_for_curvature_estimation(tg, graph_file=gt_file_normals)

    tg_curv, surf_curv = curvature_estimation(
        radius_hit, graph_file=gt_file_normals, method='VV', cores=cores)

    gt_file_curv = "{}{}.VV_area2_rh{}_epsilon0_eta0.gt".format(
        fold, base_filename, radius_hit)
    tg_curv.graph.save(gt_file_curv)
    surf_file_curv = "{}{}.VV_area2_rh{}_epsilon0_eta0.vtp".format(
        fold, base_filename, radius_hit)
    io.save_vtp(surf_curv, surf_file_curv)


def extract_curvatures_after_new_workflow(
        fold, base_filename, radius_hit, epsilon=0, eta=0, methods=['VV'],
        exclude_borders=0, categorize_shape_index=False):
    """
    Extracts curvature information from a .gt file generated by new_workflow
    into a .csv file. Optionally, values near surface borders can be excluded
    and shape index can be categorized.

    Args:
        fold (str): path where the input membrane segmentation is and where the
            output will be written
        base_filename (str): base file name for saving the output files
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3), default 0
        methods (list, optional): all methods to run in the second pass ('VV',
            'VVCF' and 'VCTV' are possible, default is 'VV')
        exclude_borders (int, optional): if > 0, triangles within this distance
            from borders in nm and corresponding values will be excluded from
            the output files (graph .gt, surface.vtp file and .csv)
        categorize_shape_index (boolean, optional): if True (default False),
            shape index categories will be added to the input graph .gt and
            surface .vtp files as well as the output .csv file

    Returns:
        None
    """
    log_file = '{}{}.{}_rh{}_epsilon{}_eta{}.log'.format(
                fold, base_filename, methods[0], radius_hit, epsilon, eta)
    sys.stdout = open(log_file, 'a')

    for method in methods:
        if method == 'VV':
            method = 'VV_area2'
        print("Method: {}".format(method))
        # input graph and surface files
        gt_infile = '{}{}.{}_rh{}_epsilon{}_eta{}.gt'.format(
            fold, base_filename, method, radius_hit, epsilon, eta)
        vtp_infile = '{}{}.{}_rh{}_epsilon{}_eta{}.vtp'.format(
            fold, base_filename, method, radius_hit, epsilon, eta)
        # output csv, gt and vtp files
        csv_outfile = '{}{}.{}_rh{}_epsilon{}_eta{}.csv'.format(
            fold, base_filename, method, radius_hit, epsilon, eta)
        gt_outfile = None
        vtp_outfile = None
        if exclude_borders > 0:
            eb = "_excluding{}borders".format(exclude_borders)
            gt_outfile = '{}{}.{}_rh{}_epsilon{}_eta{}{}.gt'.format(
                fold, base_filename, method, radius_hit, epsilon, eta, eb)
            csv_outfile = '{}{}.{}_rh{}_epsilon{}_eta{}{}.csv'.format(
                fold, base_filename, method, radius_hit, epsilon, eta, eb)
            vtp_outfile = '{}{}.{}_rh{}_epsilon{}_eta{}{}.vtp'.format(
                fold, base_filename, method, radius_hit, epsilon, eta, eb)
        elif categorize_shape_index:  # overwrite the input files
            gt_outfile = gt_infile
            vtp_outfile = vtp_infile

        # Create TriangleGraph object and load the graph file
        tg = TriangleGraph()
        tg.graph = load_graph(gt_infile)

        _extract_curvatures_from_graph(
            tg, csv_outfile, exclude_borders, gt_outfile, vtp_outfile,
            categorize_shape_index=categorize_shape_index)


def _extract_curvatures_from_graph(
        tg, csv_file, exclude_borders, gt_file=None, vtp_file=None,
        categorize_shape_index=False):
    # If don't want to include curvatures near borders, filter out those
    if exclude_borders > 0:
        tg.find_vertices_near_border(exclude_borders, purge=True)

    # List of shape class labels of all vertices for the csv file:
    shape_index_class = []
    if categorize_shape_index:
        # Add a new property: categorical shape index (one value for class)
        tg.graph.vp.shape_index_cat = tg.graph.new_vertex_property("float")
        for v in tg.graph.vertices():
            si_v = tg.graph.vp.shape_index_VV[v]
            si_cat_v, si_class_v = _shape_index_classifier(si_v)
            tg.graph.vp.shape_index_cat[v] = si_cat_v
            shape_index_class.append(si_class_v)

    # Saving the changes into graph and surface files, if specified:
    if gt_file is not None:
        tg.graph.save(gt_file)
    if vtp_file is not None:
        # Transforming the resulting graph to a surface with triangles:
        surf = tg.graph_to_triangle_poly()
        io.save_vtp(surf, vtp_file)

    # Getting estimated principal curvatures from the output graph:
    kappa_1 = tg.get_vertex_property_array("kappa_1")
    kappa_2 = tg.get_vertex_property_array("kappa_2")
    gauss_curvature = tg.get_vertex_property_array("gauss_curvature_VV")
    mean_curvature = tg.get_vertex_property_array("mean_curvature_VV")
    shape_index = tg.get_vertex_property_array("shape_index_VV")
    curvedness = tg.get_vertex_property_array("curvedness_VV")
    triangle_areas = tg.get_vertex_property_array("area")

    # Writing all the curvature values and errors into a csv file:
    df = pd.DataFrame()
    df["kappa1"] = kappa_1
    df["kappa2"] = kappa_2
    df["gauss_curvature"] = gauss_curvature
    df["mean_curvature"] = mean_curvature
    df["shape_index"] = shape_index
    if categorize_shape_index:  # want the shape class labels
        df["shape_index_class"] = shape_index_class
    df["curvedness"] = curvedness
    df["triangleAreas"] = triangle_areas
    df.to_csv(csv_file, sep=';')


def _shape_index_classifier(x):
    """
    Maps shape index value to the representative (middle) value of each shape
    class and the class label.

    Args:
        x (float): shape index value, should be in range [-1, 1]

    Returns:
        A tuple of the representative (middle) value of each shape class and
        the class label, e.g. 0, 'Saddle' fro values in range [-1/8, +1/8)
    """
    if x < -1:
        return None, None
    elif -1 <= x < -7/8.0:
        return -1, 'Spherical cup'
    elif -7/8.0 <= x < -5/8.0:
        return -0.75, 'Trough'
    elif -5/8.0 <= x < -3/8.0:
        return -0.5, 'Rut'
    elif -3/8.0 <= x < -1/8.0:
        return -0.25, 'Saddle rut'
    elif -1/8.0 <= x < 1/8.0:
        return 0, 'Saddle'
    elif 1/8.0 <= x < 3/8.0:
        return 0.25, 'Saddle ridge'
    elif 3/8.0 <= x < 5/8.0:
        return 0.5, 'Ridge'
    elif 5/8.0 <= x < 7/8.0:
        return 0.75, 'Dome'
    elif 7/8.0 <= x <= 1:
        return 1, 'Spherical cap'
    else:  # x > 1
        return None, None


def annas_workflow(
        fold, base_filename, radius_hit, seg_file=None, scale_factor_to_nm=1.368,
        epsilon=0, eta=0, methods=['VV'], thr=0.4, cores=4):
    """
    A script for running all processing steps to estimate membrane curvature.

    The three steps are: 1. isosurface generation from a filled and smoothed
    segmentation, 2. graph generation, 3. curvature calculation using the graph.

    It was written for Dr. Anna Rast's data: filled and smoothed with a Gaussian
    filter segmentation (with Matlab). Step 3. of VV, consisting of normals and
    curvature calculations, can run in parallel on multiple cores.

    Args:
        fold (str): path where the input membrane segmentation is and where the
            output will be written
        base_filename (str): base file name for saving the output files
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        seg_file (str, optional): membrane segmentation mask
        scale_factor_to_nm (float, optional): pixel size in nanometer of the
            membrane mask (default 1.368)
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3), default 0
        methods (list, optional): all methods to run in the second pass ('VV',
            'VVCF' and 'VCTV' are possible, default is 'VV')
        thr (float, optional): value threshold in the input segmentation where
            to generate the isosurface (default 0.4)
        cores (int, optional): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """

    t_begin = time.time()

    surf_file = base_filename + ".surface.vtp"
    if not isfile(fold + surf_file):
        if seg_file is None or not isfile(fold + seg_file):
            text = "The segmentation file {} not given or not found".format(
                    fold + seg_file)
            raise pexceptions.PySegInputError(
                expr="new_workflow", msg=eval(text))

        seg = io.load_tomo(fold + seg_file)
        assert(isinstance(seg, np.ndarray))

        print("\nGenerating a surface...")
        # Generate isosurface from the smoothed segmentation
        seg_vti = io.numpy_to_vti(seg)
        surfaces = vtk.vtkMarchingCubes()
        surfaces.SetInputData(seg_vti)
        surfaces.ComputeNormalsOn()
        surfaces.ComputeGradientsOn()
        surfaces.SetValue(0, thr)
        surfaces.Update()

        # Sometimes the contouring algorithm can create a volume whose gradient
        # vector and ordering of polygon (using the right hand rule) are
        # inconsistent. vtkReverseSense cures this problem.
        reverse = vtk.vtkReverseSense()
        reverse.SetInputConnection(surfaces.GetOutputPort())
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.Update()
        surf = reverse.GetOutput()

        # Writing the vtkPolyData surface into a VTP file
        io.save_vtp(surf, fold + surf_file)
        print('Surface was written to the file {}'.format(surf_file))

    else:
        print('\nReading in the surface from file...')
        surf = io.load_poly(fold + surf_file)

    clean_graph_file = '{}.scaled_cleaned.gt'.format(base_filename)
    clean_surf_file = '{}.scaled_cleaned.vtp'.format(base_filename)
    if not isfile(fold + clean_graph_file) or not isfile(fold + clean_surf_file):
        print('\nBuilding a triangle graph from the surface...')
        tg = TriangleGraph()
        tg.build_graph_from_vtk_surface(surf, scale_factor_to_nm)
        print('The graph has {} vertices and {} edges'.format(
            tg.graph.num_vertices(), tg.graph.num_edges()))

        # Saving the scaled (and cleaned) graph and surface:
        tg.graph.save(fold + clean_graph_file)
        surf_clean = tg.graph_to_triangle_poly()
        io.save_vtp(surf_clean, fold + clean_surf_file)
    else:
        print('\nReading in the cleaned graph and surface from files...')
        surf_clean = io.load_poly(fold + clean_surf_file)
        tg = TriangleGraph()
        tg.graph = load_graph(fold + clean_graph_file)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Surface and graph generation (and cleaning) took: {} min {} s'
          .format(minutes, seconds))

    gt_file = '{}{}.{}_rh{}_epsilon{}_eta{}.gt'.format(
        fold, base_filename, 'VV_area2', radius_hit, epsilon, eta)
    surf_file = '{}{}.{}_rh{}_epsilon{}_eta{}.vtp'.format(
        fold, base_filename, 'VV_area2', radius_hit, epsilon, eta)
    if not isfile(gt_file) or not isfile(surf_file):
        # Running the modified Normal Vector Voting algorithms:
        gt_file1 = '{}{}.NVV_rh{}_epsilon{}_eta{}.gt'.format(
                fold, base_filename, radius_hit, epsilon, eta)
        method_tg_surf_dict = {}
        if not isfile(gt_file1):
            method_tg_surf_dict = normals_directions_and_curvature_estimation(
                tg, radius_hit, epsilon=epsilon, eta=eta, exclude_borders=0,
                methods=methods, full_dist_map=False, graph_file=gt_file1,
                area2=True, poly_surf=surf_clean, cores=cores)

        for method in method_tg_surf_dict.keys():
            # Saving the output (graph and surface object) for later
            # filtering or inspection in ParaView:
            (tg, surf) = method_tg_surf_dict[method]
            if method == 'VV':
                method = 'VV_area2'
            gt_file = '{}{}.{}_rh{}_epsilon{}_eta{}.gt'.format(
                fold, base_filename, method, radius_hit, epsilon, eta)
            tg.graph.save(gt_file)
            surf_file = '{}{}.{}_rh{}_epsilon{}_eta{}.vtp'.format(
                fold, base_filename, method, radius_hit, epsilon, eta)
            io.save_vtp(surf, surf_file)
    else:
        print("\nOutput files {} and {} are already there.".format(
            gt_file, surf_file))


def main_javier(membrane, radius_hit):
    """
    Main function for running the new_workflow function for Javier's cER or PM.

    Args:
        membrane (string): what membrane segmentation to use 'cER' or 'PM'
        radius_hit (int): RadiusHit parameter (in nm)

    Returns:
        None
    """
    t_begin = time.time()

    base_fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/"

    # The "famous" tcb (done cER RH=6, 10 and 15 + PM RH=6, PM RH=15)
    # (~ 12 voxels diameter at the base of high curvature regions):
    # fold = "{}tcb_t3_ny01/new_workflow/".format(base_fold)
    # seg_file = "t3_ny01_lbl.Labels_cropped.mrc"
    # base_filename = "t3_ny01_cropped_{}".format(membrane)
    # pixel_size = 1.044  # from old data set!

    # The huge one tcb (done cER RH=6, but many holes, surface splits):
    # tomo = "tcb_170924_l1_t3_cleaned_pt_ny01"
    # fold = "{}{}/".format(base_fold, tomo)
    # seg_file = "{}_lbl.labels.mrc".format(tomo)
    # base_filename = "{}_{}".format(tomo, membrane)

    # The good one tcb (RadiusHit=10 and 15):
    # fold = "{}TCB/170924_TITAN_l2_t2/".format(base_fold)
    # seg_file = "t2_ny01_lbl.labels.mrc"
    # base_filename = "TCBl2t2_{}".format(membrane)

    # Another tcb with surface generation problems:
    # fold = "{}TCB/170924_TITAN_l1_t1/smooth/".format(base_fold)
    # seg_file = "t1_cleaned_pt_lbl.labels_FILLED.mrc"
    # base_filename = "TCBl1t1_{}".format(membrane)

    # tcb for Ruben:
    fold = "{}TCB/180830_TITAN_l2_t2/smooth/".format(base_fold)
    seg_file = "t2_ny01_lbl.labels_FILLED.mrc"
    base_filename = "TCBl2t2_180830_{}".format(membrane)
    scale_x = 928
    scale_y = scale_x
    scale_z = 123

    # The "sheety" scs (done cER RH=6, but holes and ridges):
    # tomo = "scs_171108_l2_t4_ny01"
    # fold = "{}{}/cropped_ends/".format(base_fold, tomo)
    # # fold = "{}{}/small_cutout/".format(base_fold, tomo)  # RH=6, 10, 15, 20
    # # lbl = 1
    # fold = "{}SCS/171108_TITAN_l2_t4/smooth/".format(base_fold)
    # seg_file = "t4_ny01_lbl.labels_FILLED.mrc"
    # base_filename = "SCSl2t4_{}".format(membrane)

    # The "good one" scs (done cER RH=6, RH=15 and RH=10;
    # normals estimation for PM RH=15):
    # tomo = "scs_171108_l1_t2_ny01"
    # fold = "{}{}/".format(base_fold, tomo)
    # # fold = "{}{}/small_cutout/".format(base_fold, tomo)  # RH=6,10,12,15,18
    # # lbl = 1
    # seg_file = "{}_lbl.labels.mrc".format(tomo)
    # base_filename = "{}_{}".format(tomo, membrane)

    # same for all:
    pixel_size = 1.368  # same for whole new data set
    holes = 3  # surface was better for the "sheety" one with 3 than with 0 or 5
    min_component = 100

    if membrane == "PM":
        lbl = 1
        print("\nEstimating normals for {}".format(base_filename))
        new_workflow(
            fold, base_filename, pixel_size, radius_hit, methods=['VV'],
            seg_file=seg_file, label=lbl, holes=holes,
            min_component=min_component, only_normals=True)
    elif membrane == "cER":
        lbl = 2
        print("\nCalculating curvatures for {}".format(base_filename))
        new_workflow(
            fold, base_filename, pixel_size, radius_hit, methods=['VV'],
            seg_file=seg_file, label=lbl, holes=holes,
            min_component=min_component)

        for b in range(0, 2):
            print("\nExtracting curvatures for {} without {} nm from border"
                  .format(membrane, b))
            extract_curvatures_after_new_workflow(
                fold, base_filename, radius_hit, methods=['VV'],
                exclude_borders=b, categorize_shape_index=True)

        surf_vtp_file = '{}{}.{}_rh{}_epsilon0_eta0.vtp'.format(
            fold, base_filename, 'VV_area2', radius_hit)
        outfile_base = '{}{}.{}_rh{}'.format(
            fold, base_filename, 'VV_area2', radius_hit)
        convert_vtp_to_stl_surface_and_mrc_curvatures(
            surf_vtp_file, outfile_base, pixel_size, scale_x, scale_y, scale_z)
    else:
        print("Membrane not known.")
        exit(0)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))


def main_felix():
    """
    Main function for running the new_workflow function for Felix' data.

    Returns:
        None
    """
    t_begin = time.time()

    # Felix's vesicle:
    base_filename = "t74_vesicle3"
    pixel_size = 2.526
    radius_hit = 10  # nm
    fold = ('/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_method/'
            'vesicle3_t74/')
    tomo = "t74"
    seg_file = "{}{}_vesicle3_bin6.Labels.mrc".format(fold, tomo)
    lbl = 1
    min_component = 100
    num_cores = 4
    runtimes_file = "{}{}_runtimes.csv".format(fold, base_filename)
    new_workflow(
            fold, base_filename, pixel_size, radius_hit, methods=['VV'],
            seg_file=seg_file, label=lbl, holes=0,
            min_component=min_component, only_normals=False,
            cores=num_cores, runtimes=runtimes_file)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))


def main_missing_wedge():
    """
    Main function for running the new_workflow function for normal vs. missing
    wedge containing sphere surface.

    Returns:
        None
    """
    t_begin = time.time()

    fold = '/fs/pool/pool-ruben/Maria/curvature/missing_wedge_sphere/'
    rh = 8

    print("\nNormal sphere (control)")
    base_filename = 'bin_sphere_r20_t1_thresh0.6'
    new_workflow(
        fold, base_filename, pixel_size_nm=1, radius_hit=rh, epsilon=0,
        eta=0, methods=['VCTV', 'VV'], remove_wrong_borders=False)
    print("\nExtracting all curvatures")
    extract_curvatures_after_new_workflow(
        fold, base_filename, radius_hit=rh, epsilon=0,
        eta=0, methods=['VCTV', 'VV'], exclude_borders=0)

    print("\nSphere with missing wedge")
    base_filename = 'bin_sphere_r20_t1_with_wedge30deg_thresh0.6'
    new_workflow(fold, base_filename, pixel_size_nm=1, radius_hit=rh,
                 epsilon=0, eta=0, methods=['VCTV', 'VV'],
                 remove_wrong_borders=True)
    for b in range(0, 9):
        print("\nExtracting curvatures without {} nm from border".format(b))
        extract_curvatures_after_new_workflow(
            fold, base_filename, radius_hit=rh,
            epsilon=0, eta=0, methods=['VCTV', 'VV'], exclude_borders=b)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))


def main_anna():
    """
    Main function for running the annas_workflow function for Anna's data.

    Returns:
        None
    """
    fold = "/fs/pool/pool-EMpub/4Maria/fromAnna/"
    seg_file = "membrane_filter.mrc"
    base_filename = "membrane_filter"
    radius_hit = 15
    annas_workflow(fold, base_filename, radius_hit, seg_file)


if __name__ == "__main__":
    # membrane = sys.argv[1]
    # rh = int(sys.argv[2])
    # main_javier(membrane, rh)
    # for t in [4, 7]:
    #     for n in range(2):
    #         extract_curvatures_after_new_workflow(
    #             fold="/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/WT/"
    #                  "171002_TITAN_l2_t{}/".format(t),
    #             base_filename="WT_171002_l2_t{}.PM".format(t), radius_hit=10,
    #             exclude_borders=n)
    subfold = "/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/WT/" \
              "171002_TITAN_l2_t2/"
    base_filename = "WT_171002_l2_t2.PM"
    calculate_PM_curvatures(subfold, base_filename, radius_hit=10, cores=1)

    # fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/new_workflow/"
    # stats_file = '{}t3_ny01_cropped_{}.VCTV_VV_area2_rh{}.stats'.format(
    #     fold, membrane, rh)
    # cProfile.run('main_javier(membrane, rh)', stats_file)

    # fold = ('/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_method/'
    #         'vesicle3_t74/')
    # stats_file = '{}t74_vesicle3.NVV_rh10.stats'.format(fold)
    # cProfile.run('main_felix()', stats_file)

    # main_felix()

    # main_anna()
