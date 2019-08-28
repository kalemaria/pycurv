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
import os

from curvaturia import (
    pexceptions, normals_directions_and_curvature_estimation, run_gen_surface,
    TriangleGraph, PointGraph,
    curvature_estimation)
from curvaturia import curvaturia_io as io

"""
A script with an example application of the curvaturia package for estimation of
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
        surf_vtp_file, outfile_base, scale, size):
    """
    Converts the '.vtp' surface file to '.stl' file and converts selected
    vtkPolyData cell arrays from the '.vtp' file as 3-D volumes in '.mrc' files.
    The selected arrays are: "kappa_1", "kappa_2", "curvedness_VV".

    Args:
        surf_vtp_file (str): surface .vtp file, should contain the final surface
            with curvatures
        outfile_base (str): base name for the output .mrc and .log files
        scale (tuple): pixel size (X, Y, Z) of the membrane mask in units of the
            surface
        size (tuple): size (X, Y, Z) of the membrane mask

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
        surf_vtp_file, outfile_base, scale, size, log_files=True)
    # mean voxel value & no .log files:
    _vtp_arrays_to_mrc_volumes(
        surf_vtp_file, outfile_base, scale, size, mean=True)


def _vtp_arrays_to_mrc_volumes(
        surf_vtp_file, outfile_base, scale, size,
        mean=False, log_files=False, compress=False):
    """
    This function converts selected vtkPolyData cell arrays from the '.vtp' file
    as 3-D volumes in '.mrc' files. The selected arrays are: "kappa_1",
    "kappa_2", "curvedness_VV".

    Args:
        surf_vtp_file (str): surface .vtp file, should contain the final surface
            with curvatures
        outfile_base (str): base name for the output .mrc (and .log) files
        scale (tuple): pixel size (X, Y, Z) of the membrane mask in units of the
            surface
        size (tuple): size (X, Y, Z) of the membrane mask
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
        volume = io.poly_array_to_volume(poly, array_name, scale, size,
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
        fold, base_filename, pixel_size, radius_hit, methods=['VV'],
        page_curvature_formula=False, area2=True,
        seg_file=None, label=1, filled_label=None, unfilled_mask=None, holes=0,
        remove_wrong_borders=True, min_component=100, only_normals=False,
        cores=4, runtimes=''):
    """
    A script for running all processing steps to estimate membrane curvature.

    The three steps are: 1. signed surface generation, 2. surface cleaning using
    a graph, 3. curvature calculation using a graph generated from the clean
    surface.

    It was written for Javier's data. Segmentation is not split into regions.
    Second pass, consisting of normals and curvature calculations, can run in
    parallel on multiple cores (for RVV and AVV, but not for SSVV).

    Args:
        fold (str): path where the input membrane segmentation is and where the
            output will be written
        base_filename (str): base file name for saving the output files
        pixel_size (float): pixel size in nanometer of the membrane mask
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation)
        seg_file (str, optional): membrane segmentation mask
        label (int, optional): label to be considered in the membrane mask
            (default 1)
        filled_label (int, optional): if the membrane mask was filled with this
            label (default None), a better surface generation will be used (with
            a slight smoothing; holes are closed automatically by the filling.)
        unfilled_mask (numpy.ndarray, optional): if given (default None), apply
            this mask on the extracted surface from an unfilled segmentation,
            instead of the segmentation itself; not used if filled_label is
            given
        holes (int, optional): if > 0, small holes in the unfilled segmentation
            are closed with a cube of that size in pixels before curvature
            estimation (default 0); not used if filled_label is given
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
            added to this file (default '')

    Returns:
        None
    """
    log_file = '{}{}.{}_rh{}.log'.format(
                fold, base_filename, methods[0], radius_hit)
    sys.stdout = open(log_file, 'a')

    t_begin = time.time()

    surf_file = base_filename + ".surface.vtp"
    if not isfile(fold + surf_file):
        if seg_file is None or not isfile(fold + seg_file):
            raise pexceptions.PySegInputError(
                expr="new_workflow",
                msg="The segmentation file not given or not found")

        seg = io.load_tomo(fold + seg_file)
        assert(isinstance(seg, np.ndarray))
        data_type = seg.dtype

        if filled_label is not None:  # if filled seg. given
            # Surface generation with filled segmentation using vtkMarchingCubes
            # and applying the mask of unfilled segmentation
            print("\nMaking filled and unfilled binary segmentations...")
            binary_seg = (seg == label).astype(data_type)
            if not np.any(binary_seg):
                raise pexceptions.PySegInputError(
                    expr="new_workflow",
                    msg="Label not found in the segmentation!")
            # have to combine the outer and inner seg. for the filled one:
            filled_binary_seg = np.logical_or(
                seg == label, seg == filled_label).astype(data_type)
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
            if not np.any(binary_seg):
                raise pexceptions.PySegInputError(
                    expr="new_workflow",
                    msg="Label not found in the segmentation!")
            if holes > 0:  # close (reduce) holes in the segmentation
                cube_size = abs(holes)
                cube = np.ones((cube_size, cube_size, cube_size))
                print("\nReducing holes in the segmentation...")
                binary_seg = ndimage.binary_closing(
                    binary_seg, structure=cube, iterations=1).astype(data_type)
            # Write the resulting binary segmentation into a file:
            binary_seg_file = "{}{}.binary_seg.mrc".format(
                fold, base_filename)
            io.save_numpy(binary_seg, binary_seg_file)
            print("\nGenerating a surface from the binary segmentation...")
            surf = run_gen_surface(binary_seg, fold + base_filename, lbl=1,
                                   other_mask=unfilled_mask)
    else:
        print('\nReading in the surface from file...')
        surf = io.load_poly(fold + surf_file)

    clean_graph_file = '{}.scaled_cleaned.gt'.format(base_filename)
    clean_surf_file = '{}.scaled_cleaned.vtp'.format(base_filename)
    if not isfile(fold + clean_graph_file) or not isfile(fold + clean_surf_file):
        print('\nBuilding a triangle graph from the surface...')
        tg = TriangleGraph()
        scale = (pixel_size, pixel_size, pixel_size)
        tg.build_graph_from_vtk_surface(surf, scale)
        if tg.graph.num_vertices() == 0:
            raise pexceptions.PySegInputError(
                expr="new_workflow", msg="The graph is empty!")
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
            tg.find_vertices_near_border(b * pixel_size, purge=True)
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
    gt_file1 = '{}{}.NVV_rh{}.gt'.format(
            fold, base_filename, radius_hit)
    method_tg_surf_dict = {}
    if not isfile(gt_file1):
        if runtimes != '':
            with open(runtimes, 'w') as f:
                f.write("num_v;radius_hit;g_max;avg_num_neighbors;cores;"
                        "duration1;method;duration2\n")
        method_tg_surf_dict = normals_directions_and_curvature_estimation(
            tg, radius_hit, methods=methods, full_dist_map=False,
            graph_file=gt_file1, page_curvature_formula=page_curvature_formula,
            area2=area2, only_normals=only_normals, poly_surf=surf_clean,
            cores=cores, runtimes=runtimes)
    elif only_normals is False:
        if runtimes != '':
            with open(runtimes, 'w') as f:
                f.write("method;duration2\n")
        for method in methods:
            tg_curv, surface_curv = curvature_estimation(
                radius_hit, graph_file=gt_file1, method=method,
                page_curvature_formula=page_curvature_formula, area2=area2,
                poly_surf=surf_clean, cores=cores, runtimes=runtimes)
            method_tg_surf_dict[method] = (tg_curv, surface_curv)

    if only_normals is False:
        for method in method_tg_surf_dict.keys():
            # Saving the output (graph and surface object) for later
            # filtering or inspection in ParaView:
            (tg, surf) = method_tg_surf_dict[method]
            if method == 'VV':
                if page_curvature_formula and (area2 is False):
                    method = 'NVV'
                elif page_curvature_formula is False:
                    if area2 is False:
                        method = 'RVV'
                    else:
                        method = 'AVV'

            gt_file = '{}{}.{}_rh{}.gt'.format(
                fold, base_filename, method, radius_hit)
            tg.graph.save(gt_file)
            surf_file = '{}{}.{}_rh{}.vtp'.format(
                fold, base_filename, method, radius_hit)
            io.save_vtp(surf, surf_file)


def calculate_PM_curvatures(fold, base_filename, radius_hit, cores=4):
    gt_file_normals = "{}{}.NVV_rh{}.gt".format(fold, base_filename, radius_hit)
    tg = TriangleGraph()
    tg.graph = load_graph(gt_file_normals)

    tg_curv, surf_curv = curvature_estimation(
        radius_hit, graph_file=gt_file_normals, method='VV', cores=cores, sg=tg)

    gt_file_curv = "{}{}.AVV_rh{}.gt".format(fold, base_filename, radius_hit)
    tg_curv.graph.save(gt_file_curv)
    surf_file_curv = "{}{}.AVV_rh{}.vtp".format(fold, base_filename, radius_hit)
    io.save_vtp(surf_curv, surf_file_curv)


def extract_curvatures_after_new_workflow(
        fold, base_filename, radius_hit, methods=['VV'],
        page_curvature_formula=False, area2=True,
        exclude_borders=0, categorize_shape_index=False):
    """
    Extracts curvature information from a .gt file generated by new_workflow
    into a .csv file. Optionally, values near surface borders can be excluded
    and shape index can be categorized.

    Args:
        fold (str): path where the input membrane segmentation is and where the
            output will be written
        base_filename (str): base file name for saving the output files
        radius_hit (float): radius in length unit of the graph, here nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation)
        exclude_borders (int, optional): if > 0, triangles within this distance
            from borders in dist and corresponding values will be excluded from
            the output files (graph .gt, surface.vtp file and .csv), iteratively
            starting from 0 until maximally this distance (integer by integer)
        categorize_shape_index (boolean, optional): if True (default False),
            shape index categories will be added to the input graph .gt and
            surface .vtp files as well as the output .csv file

    Returns:
        None
    """
    log_file = '{}{}.{}_rh{}.log'.format(
                fold, base_filename, methods[0], radius_hit)
    sys.stdout = open(log_file, 'a')

    for method in methods:
        if method == 'VV':
            if page_curvature_formula and (area2 is False):
                method = 'NVV'
            elif page_curvature_formula is False:
                if area2 is False:
                    method = 'RVV'
                else:
                    method = 'AVV'
        print("Method: {}".format(method))
        # input graph and surface files
        gt_infile = '{}{}.{}_rh{}.gt'.format(
            fold, base_filename, method, radius_hit)
        vtp_infile = '{}{}.{}_rh{}.vtp'.format(
            fold, base_filename, method, radius_hit)
        # output csv, gt and vtp files (without excluding borders)
        csv_outfile = '{}{}.{}_rh{}.csv'.format(
            fold, base_filename, method, radius_hit)
        if categorize_shape_index:  # overwrite the input files
            gt_outfile = gt_infile
            vtp_outfile = vtp_infile
        else:
            gt_outfile = None
            vtp_outfile = None
        for dist in range(exclude_borders + 1):
            if dist > 0:
                eb = "_excluding{}borders".format(dist)
                csv_outfile = '{}{}.{}_rh{}{}.csv'.format(
                    fold, base_filename, method, radius_hit, eb)
                gt_outfile = '{}{}.{}_rh{}{}.gt'.format(
                    fold, base_filename, method, radius_hit, eb)
                vtp_outfile = '{}{}.{}_rh{}{}.vtp'.format(
                    fold, base_filename, method, radius_hit, eb)

            # Create TriangleGraph object and load the graph file
            tg = TriangleGraph()
            tg.graph = load_graph(gt_infile)

            _extract_curvatures_from_graph(
                tg, csv_outfile, dist, gt_outfile, vtp_outfile,
                categorize_shape_index=categorize_shape_index)


def _extract_curvatures_from_graph(
        sg, csv_file, exclude_borders=0, gt_file=None, vtp_file=None,
        categorize_shape_index=False):
    # If don't want to include curvatures near borders, filter out those
    if exclude_borders > 0 and sg.__class__.__name__ == "TriangleGraph":
        sg.find_vertices_near_border(exclude_borders, purge=True)

    # List of shape class labels of all vertices for the csv file:
    shape_index_class = []
    if categorize_shape_index:
        # Add a new property: categorical shape index (one value for class)
        sg.graph.vp.shape_index_cat = sg.graph.new_vertex_property("float")
        for v in sg.graph.vertices():
            si_v = sg.graph.vp.shape_index_VV[v]
            si_cat_v, si_class_v = _shape_index_classifier(si_v)
            sg.graph.vp.shape_index_cat[v] = si_cat_v
            shape_index_class.append(si_class_v)

    # Saving the changes into graph and surface files, if specified:
    if gt_file is not None:
        sg.graph.save(gt_file)
    if vtp_file is not None:
        # Transforming the resulting graph to a surface with triangles:
        surf = sg.graph_to_triangle_poly()
        io.save_vtp(surf, vtp_file)

    # Getting estimated principal curvatures from the output graph:
    kappa_1 = sg.get_vertex_property_array("kappa_1")
    kappa_2 = sg.get_vertex_property_array("kappa_2")
    gauss_curvature = sg.get_vertex_property_array("gauss_curvature_VV")
    mean_curvature = sg.get_vertex_property_array("mean_curvature_VV")
    shape_index = sg.get_vertex_property_array("shape_index_VV")
    curvedness = sg.get_vertex_property_array("curvedness_VV")

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
    if sg.__class__.__name__ == "TriangleGraph":
        triangle_areas = sg.get_vertex_property_array("area")
        df["triangleAreas"] = triangle_areas
    df.to_csv(csv_file, sep=';')


def extract_areas_from_graph(
        tg, csv_file, exclude_borders, gt_file=None, vtp_file=None):
    # If don't want to include triangles near borders, filter out those
    if exclude_borders > 0:
        tg.find_vertices_near_border(exclude_borders, purge=True)

    # Saving the changes into graph and surface files, if specified:
    if gt_file is not None:
        tg.graph.save(gt_file)
    if vtp_file is not None:
        # Transforming the resulting graph to a surface with triangles:
        surf = tg.graph_to_triangle_poly()
        io.save_vtp(surf, vtp_file)

    # Getting areas from the graph:
    triangle_areas = tg.get_vertex_property_array("area")

    # Writing all the curvature values and errors into a csv file:
    df = pd.DataFrame()
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
        fold, base_filename, radius_hit, seg_file=None, pixel_size=1.368,
        methods=['VV'], thr=0.4, cores=4):
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
        radius_hit (float): radius in length unit of the graph, here nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        seg_file (str, optional): membrane segmentation mask
        pixel_size (float, optional): pixel size in nanometer of the
            membrane mask (default 1.368)
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
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
            raise pexceptions.PySegInputError(
                expr="new_workflow",
                msg="The segmentation file {} not given or not found".format(
                    fold + seg_file))

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
        scale = (pixel_size, pixel_size, pixel_size)
        tg.build_graph_from_vtk_surface(surf, scale)
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

    gt_file = '{}{}.{}_rh{}.gt'.format(
        fold, base_filename, 'AVV', radius_hit)
    surf_file = '{}{}.{}_rh{}.vtp'.format(
        fold, base_filename, 'AVV', radius_hit)
    if not isfile(gt_file) or not isfile(surf_file):
        # Running the modified Normal Vector Voting algorithms:
        gt_file1 = '{}{}.NVV_rh{}.gt'.format(
                fold, base_filename, radius_hit)
        method_tg_surf_dict = {}
        if not isfile(gt_file1):
            method_tg_surf_dict = normals_directions_and_curvature_estimation(
                tg, radius_hit, methods=methods, full_dist_map=False,
                graph_file=gt_file1, area2=True, poly_surf=surf_clean,
                cores=cores)

        for method in method_tg_surf_dict.keys():
            # Saving the output (graph and surface object) for later
            # filtering or inspection in ParaView:
            (tg, surf) = method_tg_surf_dict[method]
            if method == 'VV':
                method = 'AVV'
            gt_file = '{}{}.{}_rh{}.gt'.format(
                fold, base_filename, method, radius_hit)
            tg.graph.save(gt_file)
            surf_file = '{}{}.{}_rh{}.vtp'.format(
                fold, base_filename, method, radius_hit)
            io.save_vtp(surf, surf_file)
    else:
        print("\nOutput files {} and {} are already there.".format(
            gt_file, surf_file))


def from_ply_workflow(
        ply_file, radius_hit, scale=(1, 1, 1), page_curvature_formula=False,
        methods=["VV"], area2=True, cores=4):
    """
    Estimates curvature for each triangle in a triangle mesh in PLY format.

    Args:
        ply_file (str): PLY file with the surface
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        scale (tuple, optional): pixel size (X, Y, Z) in given units for
            scaling the surface if it is not scaled (default (1, 1, 1))
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation)
        cores (int, optional): number of cores to run VV in parallel (default 4)

    Returns:
        None
    """
    base_filename = os.path.splitext(ply_file)[0]
    log_file = '{}.{}_rh{}.log'.format(
        base_filename, methods[0], radius_hit)
    sys.stdout = open(log_file, 'a')

    # Transforming PLY to VTP surface format
    surf_file = base_filename + ".vtp"
    io.ply_file_to_vtp_file(ply_file, surf_file)

    # Reading in the surface and transforming it into a triangle graph
    print('\nReading in the surface file to get a vtkPolyData surface...')
    surf = io.load_poly(surf_file)
    print('\nBuilding a triangle graph from the surface...')
    tg = TriangleGraph()
    tg.build_graph_from_vtk_surface(surf, scale)
    if tg.graph.num_vertices() == 0:
        raise pexceptions.PySegInputError(
            expr="new_workflow", msg="The graph is empty!")
    print('The graph has {} vertices and {} edges'.format(
        tg.graph.num_vertices(), tg.graph.num_edges()))

    # Running the modified Normal Vector Voting algorithm:
    temp_normals_graph_file = '{}.VV_rh{}_normals.gt'.format(
        base_filename, radius_hit)
    method_tg_surf_dict = normals_directions_and_curvature_estimation(
        tg, radius_hit, methods=methods,
        page_curvature_formula=page_curvature_formula, area2=area2,
        poly_surf=surf, cores=cores, graph_file=temp_normals_graph_file)

    for method in method_tg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (tg, surf) = method_tg_surf_dict[method]
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        surf_file = '{}.{}_rh{}.vtp'.format(base_filename, method, radius_hit)
        io.save_vtp(surf, surf_file)
        gt_file = '{}.{}_rh{}.gt'.format(base_filename, method, radius_hit)
        tg.graph.save(gt_file)
        csv_file = '{}.{}_rh{}.csv'.format(base_filename, method, radius_hit)
        _extract_curvatures_from_graph(tg, csv_file)


def from_vtk_workflow(
        vtk_file, radius_hit, vertex_based, epsilon, eta, scale=(1, 1, 1),
        page_curvature_formula=False, methods=["VV"], area2=True, cores=4,
        reverse_normals=False):
    """
    Estimates curvature for each triangle in a triangle mesh in VTK format.

    Args:
        vtk_file (str): path to the VTK file with the surface
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        vertex_based (boolean): if True, curvature is calculated per triangle
            vertex instead of triangle center
        epsilon (float): parameter of Normal Vector Voting algorithm influencing
            the number of triangles classified as "crease junction" (class 2)
        eta (float): parameter of Normal Vector Voting algorithm influencing the
            number of triangles classified as "crease junction" (class 2) and
            "no preferred orientation" (class 3)
        scale (tuple, optional): pixel size (X, Y, Z) in given units for
            scaling the surface if it is not scaled (default (1, 1, 1))
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation; not possible if vertex_based is True)
        cores (int, optional): number of cores to run VV in parallel (default 4)
        reverse_normals (boolean, optional): if True (default False), original
            surface normals will be reversed

        Returns:
            None
        """
    if reverse_normals:
        reverse_normals_str = "_reversed_normals"
    else:
        reverse_normals_str = ""
    vtk_filename = os.path.basename(vtk_file)
    base_filename = os.path.splitext(vtk_filename)[0][:-15]+reverse_normals_str
    log_file = '{}.{}_rh{}_epsilon{}_eta{}.log'.format(
        base_filename, methods[0], radius_hit, epsilon, eta)
    sys.stdout = open(log_file, 'a')

    print('\nReading in the surface file to get a vtkPolyData surface...')
    surf = io.load_poly_from_vtk(vtk_file)

    # Running the modified Normal Vector Voting algorithm:
    normals_graph_file = '{}.VV_rh{}_epsilon{}_eta{}_normals.gt'.format(
        base_filename, radius_hit, epsilon, eta)
    method_tg_surf_dict = {}
    if not isfile(normals_graph_file):
        # Make or read in the graph first:
        if not vertex_based:
            triangle_graph_file = base_filename + ".gt"
            if not isfile(triangle_graph_file):
                # uses TriangleGraph's point_in_cells and triangle_cell_ids
                print('\nBuilding a triangle graph from the surface...')
                tg = TriangleGraph()
                tg.build_graph_from_vtk_surface(
                    surf, scale, reverse_normals=reverse_normals)
                if tg.graph.num_vertices() == 0:
                    raise pexceptions.PySegInputError(
                        expr="new_workflow", msg="The graph is empty!")
                print('The graph has {} vertices and {} edges'.format(
                    tg.graph.num_vertices(), tg.graph.num_edges()))
                tg.graph.save(triangle_graph_file)
            else:
                print('\nReading in the triangle graph from file...')
                tg = TriangleGraph()
                tg.graph = load_graph(triangle_graph_file)
            sg = tg
        else:  # vertex_based
            area2 = False
            point_graph_file = base_filename + "_point.gt"
            if not isfile(point_graph_file):
                print('\nBuilding a point graph from the surface...')
                pg = PointGraph()
                pg.build_graph_from_vtk_surface(
                    surf, scale, reverse_normals=reverse_normals)
                if pg.graph.num_vertices() == 0:
                    raise pexceptions.PySegInputError(
                        expr="new_workflow", msg="The graph is empty!")
                print('The graph has {} vertices and {} edges'.format(
                    pg.graph.num_vertices(), pg.graph.num_edges()))
                pg.graph.save(point_graph_file)
            else:
                print('\nReading in the point graph from file...')
                pg = PointGraph()
                pg.graph = load_graph(point_graph_file)
            sg = pg
        # Estimate normals, directions and curvatures:
        method_tg_surf_dict = normals_directions_and_curvature_estimation(
            sg, radius_hit, epsilon, eta, methods=methods,
            page_curvature_formula=page_curvature_formula,
            area2=area2, poly_surf=surf, cores=cores,
            graph_file=normals_graph_file)
    else:
        # Estimate directions and curvatures using the graph file with normals:
        for method in methods:
            sg_curv, surface_curv = curvature_estimation(
                radius_hit, graph_file=normals_graph_file, method=method,
                page_curvature_formula=page_curvature_formula, area2=area2,
                poly_surf=surf, cores=cores, vertex_based=vertex_based)
            method_tg_surf_dict[method] = (sg_curv, surface_curv)

    for method in method_tg_surf_dict.keys():
        # Saving the output (TriangleGraph object) for later inspection in
        # ParaView:
        (sg_curv, surface_curv) = method_tg_surf_dict[method]
        if method == 'VV':
            if page_curvature_formula:
                method = 'NVV'
            elif area2:
                method = 'AVV'
            else:
                method = 'RVV'
        surf_file = '{}.{}_rh{}_epsilon{}_eta{}.vtp'.format(
            base_filename, method, radius_hit, epsilon, eta)
        io.save_vtp(surface_curv, surf_file)
        gt_file = '{}.{}_rh{}_epsilon{}_eta{}.gt'.format(
            base_filename, method, radius_hit, epsilon, eta)
        sg_curv.graph.save(gt_file)
        csv_file = '{}.{}_rh{}_epsilon{}_eta{}.csv'.format(
            base_filename, method, radius_hit, epsilon, eta)
        _extract_curvatures_from_graph(sg_curv, csv_file)


def from_nii_workflow(
        nii_file, outfold, radius_hit, page_curvature_formula=False,
        methods=["VV"], area2=True, cores=4):
    """
    Extracts surface for every label > 0 in the segmentation in NII format,
    after applying a Gaussian filter with sigma of 1.
    For each surface, estimates curvature for each triangle in a triangle mesh.

    Args:
        nii_file (str): NII file with the segmentation
        outfold (str): output folder
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation)
        cores (int, optional): number of cores to run VV in parallel (default 4)

        Returns:
            None
        """
    base_filename = os.path.splitext(
        os.path.splitext(os.path.basename(nii_file))[0]
    )[0]  # without the path and without ".nii.gz" extensions
    log_file = '{}.{}_rh{}.log'.format(
        base_filename, methods[0], radius_hit)
    sys.stdout = open(log_file, 'a')

    # Reading in the data and getting the data type and average scaling in mm:
    seg, _, header = io.load_nii(nii_file)
    assert (isinstance(seg, np.ndarray))
    data_type = seg.dtype
    scale = header.get_zooms()
    print("pixel size in mm (x, y, z) = {}".format(scale))

    # Save as MRC file:
    mrc_file = str(os.path.join(outfold, base_filename+".mrc"))
    if not isfile(mrc_file):
        io.save_numpy(seg, mrc_file)

    for label in range(1, np.max(seg)+1):
        print("Label {}".format(label))
        # output base file name with the path and with the label:
        outfile_base = str(os.path.join(outfold, base_filename+str(label)))

        # Surface generation around the filled segmentation using
        # vtkMarchingCubes
        surf_file = outfile_base + ".surface.vtp"
        if not isfile(surf_file):
            filled_binary_seg = (seg == label).astype(data_type)
            if not np.any(filled_binary_seg):
                raise pexceptions.PySegInputError(
                    expr="from_nii_workflow",
                    msg="Label not found in the segmentation!")
            print("\nGenerating a surface...")
            surf = run_gen_surface(
                filled_binary_seg, outfile_base, lbl=1,
                other_mask=None, isosurface=True, sg=1, thr=THRESH_SIGMA1)
        else:
            print('\nReading in the surface from file...')
            surf = io.load_poly(surf_file)

        # Transforming the surface into a triangle graph
        print('\nBuilding a triangle graph from the surface...')
        tg = TriangleGraph()
        tg.build_graph_from_vtk_surface(surf, scale)
        if tg.graph.num_vertices() == 0:
            raise pexceptions.PySegInputError(
                expr="new_workflow", msg="The graph is empty!")
        print('The graph has {} vertices and {} edges'.format(
            tg.graph.num_vertices(), tg.graph.num_edges()))

        # Running the modified Normal Vector Voting algorithm:
        temp_normals_graph_file = '{}.VV_rh{}_normals.gt'.format(
            outfile_base, radius_hit)
        method_tg_surf_dict = normals_directions_and_curvature_estimation(
            tg, radius_hit, methods=methods,
            page_curvature_formula=page_curvature_formula, area2=area2,
            poly_surf=surf, cores=cores, graph_file=temp_normals_graph_file)

        for method in method_tg_surf_dict.keys():
            # Saving the output (TriangleGraph object) for later inspection in
            # ParaView:
            (tg, surf) = method_tg_surf_dict[method]
            if method == 'VV':
                if page_curvature_formula:
                    method = 'NVV'
                elif area2:
                    method = 'AVV'
                else:
                    method = 'RVV'
            surf_file = '{}.{}_rh{}.vtp'.format(
                outfile_base, method, radius_hit)
            io.save_vtp(surf, surf_file)
            gt_file = '{}.{}_rh{}.gt'.format(outfile_base, method, radius_hit)
            tg.graph.save(gt_file)
            csv_file = '{}.{}_rh{}.csv'.format(outfile_base, method, radius_hit)
            _extract_curvatures_from_graph(tg, csv_file)


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
    size = (928, 928, 123)

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

        surf_vtp_file = '{}{}.{}_rh{}.vtp'.format(
            fold, base_filename, 'AVV', radius_hit)
        outfile_base = '{}{}.{}_rh{}'.format(
            fold, base_filename, 'AVV', radius_hit)
        scale = (pixel_size, pixel_size, pixel_size)
        convert_vtp_to_stl_surface_and_mrc_curvatures(
            surf_vtp_file, outfile_base, scale, size)
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
        fold, base_filename, pixel_size=1, radius_hit=rh,
        methods=['SSVV', 'VV'], remove_wrong_borders=False)
    print("\nExtracting all curvatures")
    extract_curvatures_after_new_workflow(
        fold, base_filename, radius_hit=rh, methods=['SSVV', 'VV'],
        exclude_borders=0)

    print("\nSphere with missing wedge")
    base_filename = 'bin_sphere_r20_t1_with_wedge30deg_thresh0.6'
    new_workflow(fold, base_filename, pixel_size=1, radius_hit=rh,
                 methods=['SSVV', 'VV'], remove_wrong_borders=True)
    for b in range(0, 9):
        print("\nExtracting curvatures without {} nm from border".format(b))
        extract_curvatures_after_new_workflow(
            fold, base_filename, radius_hit=rh, methods=['SSVV', 'VV'],
            exclude_borders=b)

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

    # Get triangle areas from a scaled cleaned graph, excluding 1 nm from border
    # folder = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/" \
    #          "WT_HS/181127_TITAN_l1_t2/"
    # graph_file = "{}WT_HS_181127_l1_t2.cER.scaled_cleaned.gt".format(folder)
    # csv_file = "{}WT_HS_181127_l1_t2.cER.areas_excluding1borders.csv".format(
    #     folder)
    # tg = TriangleGraph()
    # tg.graph = load_graph(graph_file)
    # extract_areas_from_graph(tg, csv_file, exclude_borders=1)

    # One test peak run:
    # subsubfold = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/" \
    #              "TCB/180830_TITAN_l2_t2peak/filled/"
    # base_filename = "TCB_180830_l2_t2peak.cER"
    # pixel_size = 1.368
    # radius_hit = 2
    # seg_filename = "t2_ny01_lbl.labels_FILLEDpeak.mrc"
    # lbl = 2
    # filled_lbl = 3
    # min_component = 50
    # new_workflow(subsubfold, base_filename, pixel_size, radius_hit,
    #              methods=['SSVV'], seg_file=seg_filename,
    #              label=lbl, filled_label=filled_lbl,
    #              min_component=min_component, cores=4)

    # membrane = sys.argv[1]
    # rh = int(sys.argv[2])
    # main_javier(membrane, rh)
    # subfold = "/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/WT/" \
    #           "171002_TITAN_l2_t2/"
    # base_filename = "WT_171002_l2_t2.PM"
    # calculate_PM_curvatures(subfold, base_filename, radius_hit=10, cores=1)

    # fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/new_workflow/"
    # stats_file = '{}t3_ny01_cropped_{}.SSVV_AVV_rh{}.stats'.format(
    #     fold, membrane, rh)
    # cProfile.run('main_javier(membrane, rh)', stats_file)

    # fold = ('/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_method/'
    #         'vesicle3_t74/')
    # stats_file = '{}t74_vesicle3.NVV_rh10.stats'.format(fold)
    # cProfile.run('main_felix()', stats_file)

    # main_felix()

    # main_anna()

    # from_ply_workflow(
    #     ply_file="/fs/pool/pool-ruben/Maria/curvature/TestImages-LimeSeg/"
    #              "LimeSegOutput/DubSeg/cell_11/T_10.ply",
    #     radius_hit=10, scale=(1, 1, 1))

    # from_nii_workflow(
    #     nii_file="/fs/pool/pool-ruben/Maria/curvature/HVSMR2016_training_data/"
    #              "GroundTruth/training_axial_full_pat0-label.nii.gz",
    #     outfold="/fs/pool/pool-ruben/Maria/curvature/HVSMR2016_training_data/"
    #             "GroundTruthOutput",
    #     radius_hit=5)

    vtk_file = sys.argv[1]
    radius_hit = float(sys.argv[2])  # 2 mm, Mindboggle's default? "radius disk"
    if len(sys.argv) > 3:
        epsilon = float(sys.argv[3])
    else:
        epsilon = 0
    if len(sys.argv) > 4:
        eta = float(sys.argv[4])
    else:
        eta = 0
    from_vtk_workflow(
        vtk_file, radius_hit, vertex_based=True, epsilon=epsilon, eta=eta,
        scale=(1, 1, 1), methods=["VV"], cores=4, reverse_normals=False)
