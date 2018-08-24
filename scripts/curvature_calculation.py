import sys
import time
from os.path import isfile
from graph_tool import load_graph
import gzip
from os import remove
import pandas as pd
import numpy as np
from scipy import ndimage
# import cProfile

from pysurf import (
    pexceptions, normals_directions_and_curvature_estimation, vector_voting,
    run_gen_surface, TriangleGraph, split_segmentation, normals_estimation,
    preparation_for_curvature_estimation, curvature_estimation, rescale_surface)
from pysurf import pysurf_io as io

"""
A script with an example application of the PySurf package for estimation of
membrane curvature.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


# when convoluting a binary mask with a gaussian kernel with sigma 1, values 1
# at the boundary with 0's become this value:
THRESH_SIGMA1 = 0.699471735


def workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z,
             radius_hit):
    """
    Function for running all processing steps to estimate membrane curvature.

    The three steps are: 1. signed surface generation, 2. surface cleaning using
    a graph, 3. curvature calculation using a graph generated from the clean
    surface.

    Args:
        fold (str): path where the input membrane segmentation is and where the
            output will be written
        tomo (str): tomogram name with which the segmentation file starts
        seg_file (str): membrane segmentation mask (may contain 'fold' and
            'tomo')
        label (int): label to be considered in the membrane mask
        pixel_size (float): pixel size in nanometer of the membrane mask
        scale_x (int): size of the membrane mask in X dimension
        scale_y (int): size of the membrane mask in Y dimension
        scale_z (int): size of the membrane mask in Z dimension
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface

    Returns:
        None
    """
    epsilon = 0
    eta = 0
    region_file_base = "%s%s_ERregion" % (fold, tomo)
    all_kappa_1_values = []
    all_kappa_2_values = []
    all_gauss_curvature_VV_values = []
    all_mean_curvature_VV_values = []
    all_shape_index_VV_values = []
    all_curvedness_VV_values = []
    region_surf_with_borders_files = []
    region_surf_VV_files = []
    all_file_base = "%s%s_ER" % (fold, tomo)
    all_kappa_1_file = ("%s.VV_rh%s_epsilon%s_eta%s.max_curvature.txt"
                        % (all_file_base, radius_hit, epsilon, eta))
    all_kappa_2_file = ("%s.VV_rh%s_epsilon%s_eta%s.min_curvature.txt"
                        % (all_file_base, radius_hit, epsilon, eta))
    all_gauss_curvature_VV_file = (
        "%s.VV_rh%s_epsilon%s_eta%s.gauss_curvature.txt" % (
            all_file_base, radius_hit, epsilon, eta))
    all_mean_curvature_VV_file = ("%s.VV_rh%s_epsilon%s_eta%s.mean_curvature.txt"
                                  % (all_file_base, radius_hit, epsilon, eta))
    all_shape_index_VV_file = ("%s.VV_rh%s_epsilon%s_eta%s.shape_index.txt"
                               % (all_file_base, radius_hit, epsilon, eta))
    all_curvedness_VV_file = ("%s.VV_rh%s_epsilon%s_eta%s.curvedness.txt"
                              % (all_file_base, radius_hit, epsilon, eta))
    all_surf_with_borders_vtp_file = ('%s.cleaned_surface_with_borders_nm.vtp'
                                      % all_file_base)
    all_surf_VV_vtp_file = ('%s.cleaned_surface_nm.VV_rh%s_epsilon%s_eta%s.vtp'
                            % (all_file_base, radius_hit, epsilon, eta))

    # Split the segmentation into regions:
    regions, mask_file = split_segmentation(
        seg_file, lbl=label, close=True, close_cube_size=3, close_iter=1,
        min_region_size=100)
    for i, region in enumerate(regions):
        print "\n\nRegion %s" % (i + 1)

        # ***Part 1: surface generation***
        surf_file_base = "%s%s" % (region_file_base, i + 1)
        # region surface file, output of run_gen_surface
        surf_file = surf_file_base + '.surface.vtp'
        surf = None
        if not isfile(surf_file):
            print "\nGenerating a surface..."
            surf = run_gen_surface(region, surf_file_base, lbl=label,
                                   save_input_as_vti=True)

        # ***Part 2: surface cleaning***
        scale_factor_to_nm = pixel_size
        cleaned_scaled_surf_file = (
            surf_file_base + '.cleaned_surface_with_borders_nm.vtp')
        cleaned_scaled_graph_file = (
            surf_file_base + '.cleaned_triangle_graph_with_borders_nm.gt')
        # The cleaned scaled surface .vtp file does not exist yet if no cleaned
        # scaled surface .vtp file was generated, then also no cleaned scaled
        # graph .gt file was written -> have to generate the scaled graph from
        # the original surface and clean it:
        if not isfile(cleaned_scaled_surf_file):
            # this is the case if surface was generated in an earlier run
            if surf is None:
                surf = io.load_poly(surf_file)
                print 'A surface was loaded from the file %s' % surf_file

            print ('\nBuilding the TriangleGraph from the vtkPolyData surface '
                   'with curvatures...')
            tg = TriangleGraph()
            surf_nm = tg.build_graph_from_vtk_surface(
                surf, scale_factor_to_nm, verbose=False)
            print ('The graph has %s vertices and %s edges'
                   % (tg.graph.num_vertices(), tg.graph.num_edges()))

            io.save_vtp(surf_nm, surf_file[0:-4] + "_nm.vtp")
            print ('The surface scaled to nm was written into the file %s_nm'
                   '.vtp' % surf_file[0:-4])

            print '\nFinding triangles that are 3 pixels to surface borders...'
            tg.find_vertices_near_border(3 * scale_factor_to_nm, purge=True)
            print ('The graph has %s vertices and %s edges'
                   % (tg.graph.num_vertices(), tg.graph.num_edges()))

            print '\nFinding small connected components of the graph...'
            tg.find_small_connected_components(threshold=100, purge=True)

            if tg.graph.num_vertices() > 0:
                print ('The graph has %s vertices and %s edges and following '
                       'properties'
                       % (tg.graph.num_vertices(), tg.graph.num_edges()))
                tg.graph.list_properties()
                tg.graph.save(cleaned_scaled_graph_file)
                print ('Cleaned and scaled graph with outer borders was '
                       'written into the file %s' % cleaned_scaled_graph_file)

                poly_triangles_filtered_with_borders = \
                    tg.graph_to_triangle_poly()
                io.save_vtp(poly_triangles_filtered_with_borders,
                            cleaned_scaled_surf_file)
                print ('Cleaned and scaled surface with outer borders was '
                       'written into the file %s' % cleaned_scaled_surf_file)
                calculate_curvature = True
            else:
                print ("Region %s was completely filtered out and will be "
                       "omitted." % (i + 1))
                calculate_curvature = False
        # This is the case if graph generation and cleaning was done in an
        # earlier run and the cleaned scaled surface .vtp file exists:
        else:
            # the graph has vertices for sure if the .vtp file was written
            calculate_curvature = True
            # the graph was not saved and has to be reversed-engineered from the
            # surface
            if not isfile(cleaned_scaled_graph_file):
                surf = io.load_poly(cleaned_scaled_surf_file)
                print ('The cleaned and scaled surface with outer borders was '
                       'loaded from the file %s' % cleaned_scaled_surf_file)

                print '\nBuilding the triangle graph from the surface...'
                tg = TriangleGraph()
                tg.build_graph_from_vtk_surface(
                    surf, scale_factor_to_nm=1, verbose=False)
                print ('The graph has %s vertices and %s edges'
                       % (tg.graph.num_vertices(), tg.graph.num_edges()))
                tg.graph.list_properties()
                tg.graph.save(cleaned_scaled_graph_file)
                print ('Cleaned and scaled graph with outer borders was '
                       'written into the file %s' % cleaned_scaled_graph_file)
            # cleaned scaled graph can just be loaded from the found .gt file
            else:
                tg = TriangleGraph()
                tg.graph = load_graph(cleaned_scaled_graph_file)
                print ('Cleaned and scaled graph with outer borders was loaded '
                       'from the file %s' % cleaned_scaled_graph_file)
                print ('The graph has %s vertices and %s edges'
                       % (tg.graph.num_vertices(), tg.graph.num_edges()))

        # ***Part 3: curvature calculation***
        if calculate_curvature:
            # Running the Normal Vector Voting algorithm and saving the output:
            cleaned_scaled_graph_VV_file = (
                surf_file_base +
                '.cleaned_triangle_graph_nm.VV_rh%s_epsilon%s_eta%s.gt'
                % (radius_hit, epsilon, eta)
            )
            cleaned_scaled_surf_VV_file = (
                surf_file_base +
                '.cleaned_surface_nm.VV_rh%s_epsilon%s_eta%s.vtp'
                % (radius_hit, epsilon, eta)
            )
            kappa_1_file = "%s%s.VV_rh%s_epsilon%s_eta%s.max_curvature.txt" % (
                region_file_base, i + 1, radius_hit, epsilon, eta)
            kappa_2_file = "%s%s.VV_rh%s_epsilon%s_eta%s.min_curvature.txt" % (
                region_file_base, i + 1, radius_hit, epsilon, eta)
            gauss_curvature_VV_file = (
                "%s%s.VV_rh%s_epsilon%s_eta%s.gauss_curvature.txt"
                % (region_file_base, i + 1, radius_hit, epsilon, eta))
            mean_curvature_VV_file = (
                "%s%s.VV_rh%s_epsilon%s_eta%s.mean_curvature.txt"
                % (region_file_base, i + 1, radius_hit, epsilon, eta))
            shape_index_VV_file = (
                "%s%s.VV_rh%s_epsilon%s_eta%s.shape_index.txt" % (
                    region_file_base, i + 1, radius_hit, epsilon, eta))
            curvedness_VV_file = (
                "%s%s.VV_rh%s_epsilon%s_eta%s.curvedness.txt" % (
                    region_file_base, i + 1, radius_hit, epsilon, eta))
            # does not calculate curvatures for triangles at borders and removes
            # them in the end
            surf_VV = vector_voting(tg, radius_hit, epsilon=epsilon, eta=eta,
                                    exclude_borders=True)
            print ('The graph without outer borders and with VV curvatures has '
                   '%s vertices and %s edges'
                   % (tg.graph.num_vertices(), tg.graph.num_edges()))
            tg.graph.list_properties()
            tg.graph.save(cleaned_scaled_graph_VV_file)
            print ('The graph without outer borders and with VV curvatures was '
                   'written into the file %s' % cleaned_scaled_graph_VV_file)

            io.save_vtp(surf_VV, cleaned_scaled_surf_VV_file)
            print ('The surface without outer borders and with VV curvatures '
                   'was written into the file %s' % cleaned_scaled_surf_VV_file)

            # Making a list of all the region .vtp files
            region_surf_with_borders_files.append(cleaned_scaled_surf_file)
            region_surf_VV_files.append(cleaned_scaled_surf_VV_file)

            # Getting the VV curvatures from the output graph (without outer
            # borders), and merging the respective values for all regions:
            kappa_1_values = tg.get_vertex_property_array("kappa_1")
            all_kappa_1_values.extend(kappa_1_values.tolist())
            kappa_2_values = tg.get_vertex_property_array("kappa_2")
            all_kappa_2_values.extend(kappa_2_values.tolist())
            gauss_curvature_VV_values = tg.get_vertex_property_array(
                "gauss_curvature_VV")
            all_gauss_curvature_VV_values.extend(
                gauss_curvature_VV_values.tolist())
            mean_curvature_VV_values = tg.get_vertex_property_array(
                "mean_curvature_VV")
            all_mean_curvature_VV_values.extend(
                mean_curvature_VV_values.tolist())
            shape_index_VV_values = tg.get_vertex_property_array(
                "shape_index_VV")
            all_shape_index_VV_values.extend(shape_index_VV_values.tolist())
            curvedness_VV_values = tg.get_vertex_property_array("curvedness_VV")
            all_curvedness_VV_values.extend(curvedness_VV_values.tolist())

            # Writing all the region curvature values into files:
            io.write_values_to_file(kappa_1_values, kappa_1_file)
            io.write_values_to_file(kappa_2_values, kappa_2_file)
            io.write_values_to_file(gauss_curvature_VV_values,
                                    gauss_curvature_VV_file)
            io.write_values_to_file(mean_curvature_VV_values,
                                    mean_curvature_VV_file)
            io.write_values_to_file(shape_index_VV_values, shape_index_VV_file)
            io.write_values_to_file(curvedness_VV_values, curvedness_VV_file)
            print ('All the curvature values for the region were written into '
                   'files')

    # Writing all the joint curvature values into files:
    io.write_values_to_file(all_kappa_1_values, all_kappa_1_file)
    io.write_values_to_file(all_kappa_2_values, all_kappa_2_file)
    io.write_values_to_file(all_gauss_curvature_VV_values,
                            all_gauss_curvature_VV_file)
    io.write_values_to_file(all_mean_curvature_VV_values,
                            all_mean_curvature_VV_file)
    io.write_values_to_file(all_shape_index_VV_values, all_shape_index_VV_file)
    io.write_values_to_file(all_curvedness_VV_values, all_curvedness_VV_file)
    print ('All the curvature values for the whole tomogram were written into '
           'files')

    # Merging all region '.vtp' files (once with outer borders before VV, if it
    # has not been done yet, and once after VV):
    if not isfile(all_surf_with_borders_vtp_file):
        io.merge_vtp_files(region_surf_with_borders_files,
                           all_surf_with_borders_vtp_file)
        print ("Done merging all the found region cleaned and scaled surface "
               "with outer borders '.vtp' files into the file %s"
               % all_surf_with_borders_vtp_file)
    io.merge_vtp_files(region_surf_VV_files, all_surf_VV_vtp_file)
    print ("Done merging all the found region cleaned and scaled surface with "
           "curvatures '.vtp' files into the file %s" % all_surf_VV_vtp_file)

    # Converting the '.vtp' tomogram files to '.stl' files:
    all_surf_with_borders_stl_file = (all_surf_with_borders_vtp_file[0:-4] +
                                      '.stl')
    if not isfile(all_surf_with_borders_stl_file):
        io.vtp_file_to_stl_file(all_surf_with_borders_vtp_file,
                                all_surf_with_borders_stl_file)
        print ("The '.vtp' file %s was converted to .stl format"
               % all_surf_with_borders_vtp_file)
    all_surf_VV_stl_file = all_surf_VV_vtp_file[0:-4] + '.stl'
    io.vtp_file_to_stl_file(all_surf_VV_vtp_file, all_surf_VV_stl_file)
    print ("The '.vtp' file %s was converted to .stl format"
           % all_surf_VV_vtp_file)

    # Converting vtkPolyData selected cell arrays from the '.vtp' file to 3-D
    # volumes and saving them as '.mrc.gz' files.
    # max voxel value & .log files:
    __vtp_arrays_to_mrc_volumes(
        all_file_base, all_surf_VV_vtp_file, pixel_size, scale_x, scale_y,
        scale_z, radius_hit, epsilon, eta, log_files=True)
    # mean voxel value & no .log files:
    __vtp_arrays_to_mrc_volumes(
        all_file_base, all_surf_VV_vtp_file, pixel_size, scale_x, scale_y,
        scale_z, radius_hit, epsilon, eta, mean=True)


def __vtp_arrays_to_mrc_volumes(
        all_file_base, all_surf_VV_vtp_file, pixel_size, scale_x, scale_y,
        scale_z, radius_hit, epsilon, eta, mean=False, log_files=False):
    """
    This subfunction converts vtkPolyData cell arrays from a '.vtp' file to 3-D
    volumes and saves them as '.mrc.gz' files.

    Most of the parameters are passed from the "parent" function, "workflow".

    Args:
        all_file_base (str): as defined in workflow
        all_surf_VV_vtp_file (str): as defined in workflow
        pixel_size (float): pixel size in nanometer of the membrane mask (as
            passed to workflow)
        scale_x (int): size of the membrane mask in X dimension (as passed to
            workflow)
        scale_y (int): size of the membrane mask in Y dimension (as passed to
            workflow)
        scale_z (int): size of the membrane mask in Z dimension (as passed to
            workflow)
        radius_hit (float): radius in length unit of the graph, e.g. nanometers;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface
        epsilon (int): as defined in workflow
        eta (int): as defined in workflow
        mean (boolean, optional): if True (default False), in case multiple
            triangles map to the same voxel, takes the mean value, else the
            maximal value
        log_files (boolean, optional): if True (default False), writes the log
            files for such cases

    Returns:
        None
    """
    array_name1 = "kappa_1"
    name1 = "max_curvature"
    array_name2 = "kappa_2"
    name2 = "min_curvature"
    array_name3 = "curvedness_VV"
    name3 = "curvedness"

    if mean:
        voxel_mean_str = ".voxel_mean"
    else:
        voxel_mean_str = ""
    mrcfilename1 = '%s.VV_rh%s_epsilon%s_eta%s.%s.volume%s.mrc' % (
        all_file_base, radius_hit, epsilon, eta, name1, voxel_mean_str)
    mrcfilename2 = '%s.VV_rh%s_epsilon%s_eta%s.%s.volume%s.mrc' % (
        all_file_base, radius_hit, epsilon, eta, name2, voxel_mean_str)
    mrcfilename3 = '%s.VV_rh%s_epsilon%s_eta%s.%s.volume%s.mrc' % (
        all_file_base, radius_hit, epsilon, eta, name3, voxel_mean_str)
    if log_files:
        logfilename1 = mrcfilename1[0:-4] + '.log'
        logfilename2 = mrcfilename2[0:-4] + '.log'
        logfilename3 = mrcfilename3[0:-4] + '.log'
    else:
        logfilename1 = None
        logfilename2 = None
        logfilename3 = None

    # Load the vtkPolyData object from the '.vtp' file, calculate the volumes
    # from arrays, write '.log' files, and save the volumes as '.mrc' files:
    poly = io.load_poly(all_surf_VV_vtp_file)
    volume1 = io.poly_array_to_volume(poly, array_name1, pixel_size,
                                      scale_x, scale_y, scale_z,
                                      logfilename=logfilename1, mean=mean)
    io.save_numpy(volume1, mrcfilename1)
    volume2 = io.poly_array_to_volume(poly, array_name2, pixel_size,
                                      scale_x, scale_y, scale_z,
                                      logfilename=logfilename2, mean=mean)
    io.save_numpy(volume2, mrcfilename2)
    volume3 = io.poly_array_to_volume(poly, array_name3, pixel_size,
                                      scale_x, scale_y, scale_z,
                                      logfilename=logfilename3, mean=mean)
    io.save_numpy(volume3, mrcfilename3)

    # Gunzip the '.mrc' files and delete the uncompressed files:
    for mrcfilename in [mrcfilename1, mrcfilename2, mrcfilename3]:
        with open(mrcfilename) as f_in, \
                gzip.open(mrcfilename + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
        remove(mrcfilename)
        print 'Archive %s.gz was written' % mrcfilename


def new_workflow(
        fold, base_filename, scale_factor_to_nm, radius_hit,
        epsilon=0, eta=0, methods=['VV'],
        seg_file=None, label=1, holes=0, remove_wrong_borders=True,
        remove_small_components=100, only_normals=False):
    # TODO docstring - works for Javier data!
    # holes (int): a positive number means closing with a cube of that size,
    # a negative number means removing surface borders of that size (in pixels)
    # before curvature estimation (opening with a cube of that size removes
    # everything)

    # log_file = '{}{}.{}_rh{}_epsilon{}_eta{}.log'.format(
    #             fold, base_filename, methods[0], radius_hit, epsilon, eta)
    # sys.stdout = open(log_file, 'a')

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
        data_type = seg.dtype

        if label == 2 and np.max(seg) == 3:  # if cER and filled cER seg. exists
            # Surface generation with filled segmentation using vtkMarchingCubes
            # and applying the mask of unfilled segmentation
            print ("\nMaking filled and unfilled binary segmentations...")
            # have to combine the outer and inner seg. for the filled one:
            filled_binary_seg = np.logical_or(seg == 2, seg == 3).astype(
                data_type)
            binary_seg = (seg == 2).astype(data_type)
            print ("\nGenerating a surface...")
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
            print ("\nMaking the segmentation binary...")
            binary_seg = (seg == label).astype(data_type)
            if holes != 0:  # reduce / increase holes in the segmentation
                cube_size = abs(holes)
                cube = np.ones((cube_size, cube_size, cube_size))
                if holes > 0:  # close (reduce) holes
                    print ("\nReducing holes in the segmentation...")
                    binary_seg = ndimage.binary_closing(
                        binary_seg, structure=cube, iterations=1).astype(
                        data_type)
                    # Write the resulting binary segmentation into a file:
                    binary_seg_file = "{}{}.binary_seg.mrc".format(
                        fold, base_filename)
                    io.save_numpy(binary_seg, binary_seg_file)
                # else:  # open (increase) holes - removed everything
                #     print ("\nIncreasing holes in the segmentation...")
                #     binary_seg = ndimage.binary_opening(
                #         binary_seg, structure=cube, iterations=1).astype(
                #         data_type)
            print ("\nGenerating a surface from the binary segmentation...")
            surf = run_gen_surface(binary_seg, fold + base_filename, lbl=1)
    else:
        print ('\nReading in the surface from file...')
        surf = io.load_poly(fold + surf_file)

    clean_graph_file = '{}.scaled_cleaned.gt'.format(base_filename)
    clean_surf_file = '{}.scaled_cleaned.vtp'.format(base_filename)
    if not isfile(fold + clean_graph_file) or not isfile(fold + clean_surf_file):
        print ('\nBuilding a triangle graph from the surface...')
        tg = TriangleGraph()
        tg.build_graph_from_vtk_surface(surf, scale_factor_to_nm)
        print ('The graph has {} vertices and {} edges'.format(
            tg.graph.num_vertices(), tg.graph.num_edges()))

        # Remove the wrong borders (surface generation artefact)
        b = 0
        if remove_wrong_borders:
            b += 2  # because of MAX_DIST_SURF parameter in surface generation
        if holes < 0:
            b += abs(holes)
        if b > 0:
            print ('\nFinding triangles that are {} pixels to surface '
                   'borders...'.format(b))
            tg.find_vertices_near_border(b * scale_factor_to_nm, purge=True)
            print ('The graph has {} vertices and {} edges'.format(
                tg.graph.num_vertices(), tg.graph.num_edges()))

        # Filter out possibly occurring small disconnected fragments
        if remove_small_components > 0:
            print ('\nFinding small connected components of the graph...')
            tg.find_small_connected_components(
                threshold=remove_small_components, purge=True, verbose=True)
            print ('The graph has {} vertices and {} edges'.format(
                tg.graph.num_vertices(), tg.graph.num_edges()))

        # Saving the scaled (and cleaned) graph and surface:
        tg.graph.save(fold + clean_graph_file)
        surf_clean = tg.graph_to_triangle_poly()
        io.save_vtp(surf_clean, fold + clean_surf_file)
    else:
        print ('\nReading in the cleaned graph and surface from files...')
        surf_clean = io.load_poly(fold + clean_surf_file)
        tg = TriangleGraph()
        tg.graph = load_graph(fold + clean_graph_file)

    t_end = time.time()
    duration = t_end - t_begin
    print ('Surface and graph generation (and cleaning) took: {} min {} s'
           .format(divmod(duration, 60)[0], divmod(duration, 60)[1]))

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
                area2=True, only_normals=only_normals, poly_surf=surf_clean)
        elif only_normals is False:
            for method in methods:
                tg_curv, surface_curv = curvature_estimation(
                    radius_hit, exclude_borders=0, graph_file=gt_file1,
                    method=method, poly_surf=surf_clean)
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
    else:
        print("\nOutput files {} and {} are already there.".format(
            gt_file, surf_file))


def extract_curvatures_after_new_workflow(
        fold, base_filename, radius_hit, epsilon=0, eta=0, methods=['VV'],
        exclude_borders=0, categorize_shape_index=False):
    # TODO docstring if remains
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

        __extract_curvatures_from_graph(
            tg, csv_outfile, exclude_borders, gt_outfile, vtp_outfile,
            categorize_shape_index=categorize_shape_index)


def __extract_curvatures_from_graph(
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
            si_cat_v, si_class_v = __shape_index_classifier(si_v)
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


def __shape_index_classifier(x):
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


def main(membrane, rh):
    """
    Main function for running the workflow function for Javier's cER or PM.

    Args:
        membrane(string): what membrane segmentation to use 'cER' or 'PM'
        rh(int): RadiusHit parameter (in nm)

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
    fold = "{}TCB/170924_TITAN_l1_t1/smooth/".format(base_fold)
    seg_file = "t1_cleaned_pt_lbl.labels_FILLED.mrc"
    base_filename = "TCBl1t1_{}".format(membrane)

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
    radius_hit = rh
    min_component = 100

    if membrane == "PM":
        lbl = 1
        print("\nEstimating normals for {}".format(base_filename))
        new_workflow(
            fold, base_filename, pixel_size, radius_hit, methods=['VV'],
            seg_file=seg_file, label=lbl, holes=holes,
            remove_small_components=min_component, only_normals=True)
    elif membrane == "cER":
        lbl = 2
        print("\nCalculating curvatures for {}".format(base_filename))
        new_workflow(
            fold, base_filename, pixel_size, radius_hit, methods=['VV'],
            seg_file=seg_file, label=lbl, holes=holes,
            remove_small_components=min_component)

        for b in range(0, 2):
            print("\nExtracting curvatures for {} without {} nm from border"
                  .format(membrane, b))
            extract_curvatures_after_new_workflow(
                fold, base_filename, pixel_size, radius_hit, methods=['VV'],
                exclude_borders=b, categorize_shape_index=True)
    else:
        print("Membrane not known.")
        exit(0)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nTotal elapsed time: %s min %s s' % divmod(duration, 60)


def main_smoothed(membrane):
    """
    Main function for running the workflow function for real data.

    Args:
        membrane(string): what membrane segmentation to use 'cER' or 'PM'

    Returns:
        None
    """
    t_begin = time.time()

    fold = ("/fs/pool/pool-ruben/Maria/curvature/Javier/scs_171108_l2_t4_ny01/"
            "small_cutout_smoothed/")
    base_filename = "scs_171108_l2_t4_ny01_{}_holes3.smoothed300".format(
        membrane)
    pixel_size = 1  # because the smoothed surface is already scaled (& cleaned)
    radius_hit = 6

    print("\nCalculating curvatures for {}".format(membrane))
    new_workflow(
        fold, base_filename, pixel_size, radius_hit, epsilon=0, eta=0,
        methods=['VCTV', 'VV'], remove_wrong_borders=False,
        remove_small_components=0)

    for b in range(0, 3):
        print("\nExtracting curvatures for {} without {} nm from border".format(
            membrane, b))
        extract_curvatures_after_new_workflow(
            fold, base_filename, radius_hit, epsilon=0, eta=0,
            methods=['VCTV', 'VV'], exclude_borders=b)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nTotal elapsed time: %s min %s s' % divmod(duration, 60)


def main2():
    t_begin = time.time()

    # Change those parameters for each tomogram & label:
    # fold = \
    #     "/fs/pool/pool-ruben/Maria/curvature/Felix/new_workflow/diffuseHtt97Q/"
    # tomo = "t112"
    # seg_file = "%s%s_final_ER1_vesicles2_notER3_NE4.Labels.mrc" % (fold, tomo)
    # label = 1
    # pixel_size = 2.526
    # scale_x = 620
    # scale_y = 620
    # scale_z = 80
    # radius_hit = 3
    # workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z,
    #          radius_hit)

    # Felix's vesicle:
    base_filename = "t74_vesicle3"
    pixel_size = 2.526
    radius_hit = 10  # nm
    fold = ('/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_method/'
            'vesicle3_t74/')
    tomo = "t74"
    seg_file = "%s%s_vesicle3_bin6.Labels.mrc" % (fold, tomo)
    lbl = 1
    min_component = 100
    new_workflow(
            fold, base_filename, pixel_size, radius_hit, methods=['VV'],
            seg_file=seg_file, label=lbl, holes=0,
            remove_small_components=min_component, only_normals=True)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nTotal elapsed time: %s min %s s' % divmod(duration, 60)


def main3():
    t_begin = time.time()

    fold = '/fs/pool/pool-ruben/Maria/curvature/missing_wedge_sphere/'
    rh = 8

    print("\nNormal sphere (control)")
    base_filename = 'bin_sphere_r20_t1_thresh0.6'
    new_workflow(
        fold, base_filename, scale_factor_to_nm=1, radius_hit=rh, epsilon=0,
        eta=0, methods=['VCTV', 'VV'], remove_wrong_borders=False)
    print("\nExtracting all curvatures")
    extract_curvatures_after_new_workflow(
        fold, base_filename, radius_hit=rh, epsilon=0,
        eta=0, methods=['VCTV', 'VV'], exclude_borders=0)

    print("\nSphere with missing wedge")
    base_filename = 'bin_sphere_r20_t1_with_wedge30deg_thresh0.6'
    new_workflow(fold, base_filename, scale_factor_to_nm=1, radius_hit=rh,
                 epsilon=0, eta=0, methods=['VCTV', 'VV'],
                 remove_wrong_borders=True)
    for b in range(0, 9):
        print("\nExtracting curvatures without {} nm from border".format(b))
        extract_curvatures_after_new_workflow(
            fold, base_filename, radius_hit=rh,
            epsilon=0, eta=0, methods=['VCTV', 'VV'], exclude_borders=b)

    t_end = time.time()
    duration = t_end - t_begin
    print('\nTotal elapsed time: {} min {} s'.format(
        divmod(duration, 60)[0], divmod(duration, 60)[1]))

if __name__ == "__main__":
    # main("cER", 10)
    # membrane = sys.argv[1]
    # rh = int(sys.argv[2])
    # main(membrane, rh)
    # fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/new_workflow/"
    # stats_file = '{}t3_ny01_cropped_{}.VCTV_VV_area2_rh{}.stats'.format(
    #     fold, membrane, rh)
    # cProfile.run('main(membrane, rh)', stats_file)

    # fold = ('/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_method/'
    #         'vesicle3_t74/')
    # stats_file = '{}t74_vesicle3.NVV_rh10.stats'.format(fold)
    # cProfile.run('main2()', stats_file)

    main2()
