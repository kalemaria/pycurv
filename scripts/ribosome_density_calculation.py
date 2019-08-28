import time
from datetime import datetime
from graph_tool import load_graph

from curvaturia import (read_in_mask, get_target_voxels_in_membrane_mask,
                        VoxelGraph)
from curvaturia import curvaturia_io as io

"""
A script with an example application of the curvaturia package for calculation of
membrane-bound ribosome density.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def run_build_graph_from_np_ndarray(mem_mask, mem_graph_file, verbose=False):
    """
    Builds a graph from a membrane mask and writes the graph to a file.

    Args:
        mem_mask (numpy ndarray): binary membrane mask in form of 3D array,
            segmenting the underlying tomogram into membrane (voxels with value
            1) and background (voxels with value 0)
        mem_graph_file (str): name for the output membrane graph file
            (preferably with '.gt' or '.graphml' extension)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        None
    """
    t_begin = time.time()

    # Build a graph from the membrane mask:
    now = datetime.now()
    print('\nStarting building the membrane graph on: {}-{}-{} {}:{}:{}'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second))
    vg = VoxelGraph()
    vg.build_graph_from_np_ndarray(mem_mask, verbose)
    now = datetime.now()
    print('\nFinished building the graph on: {}-{}-{} {}:{}:{}'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second))
    print(vg.graph)
    vg.graph.list_properties()

    # Write the graph to the file:
    print('\nWriting the graph to the file {}'.format(mem_graph_file))
    vg.graph.save(mem_graph_file)

    t_end = time.time()
    duration = t_end - t_begin
    print('Elapsed time: {} s'.format(duration))


def run_calculate_density(mem_graph_file, ribo_mask, scale_factor_to_nm=1,
                          vtp_files_base=None, verbose=False):
    """
    Reads in the membrane graph from a file and calculates ribosome density for
    each membrane voxel.

    Args:
        mem_graph_file (str): name of the input membrane graph file (preferably
            with '.gt' or '.graphml' extension)
        ribo_mask (numpy ndarray): binary mask of ribosomes centers on membrane
            in form of 3D array, where a voxel with value 1 means a particle is
            present at that membrane coordinate
        scale_factor_to_nm (float, optional): pixel size in nanometers, default
            1 (if the graph was not scaled to nanometers)
        vtp_files_base (str, optional): If not None (default None), the
            VoxelGraph is converted to VTK PolyData points and lines objects and
            written to '<vtp_files_base>.vertices.vtp' and
            '<vtp_files_base>.edges.vtp' files, respectively
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        a numpy ndarray (with the same shape as the mask) with the densities
    """
    t_begin = time.time()

    # Read in the graph from the file:
    print('\nReading in the graph from the file {}'.format(mem_graph_file))
    vg = VoxelGraph()
    vg.graph = load_graph(mem_graph_file)
    print(vg.graph)
    vg.graph.list_properties()

    # Fill the dictionary of VoxelGraph, coordinates_to_vertex_index:
    vg.update_coordinates_to_vertex_index()
    print('Size of coordinates_to_vertex_index: {}'.format(
        len(vg.coordinates_to_vertex_index)))

    # Calculate shortest distances for each node in the graph (membrane voxel)
    # to each reachable voxel of ribosome center mapped on the membrane, and
    # from the distances a density measure of ribosomes at each membrane voxel:
    now = datetime.now()
    print('\nStarting calculating shortest distances to each reachable '
          'ribosome center and ribosome density for each membrane voxel on: '
          '{}-{}-{} {}:{}:{}'.format(
           now.year, now.month, now.day, now.hour, now.minute, now.second))
    densities = vg.calculate_density(
        ribo_mask.shape[0], ribo_mask.shape[1], ribo_mask.shape[2],
        scale_factor_to_nm, mask=ribo_mask, verbose=verbose)
    now = datetime.now()
    print('\nFinished calculating the shortest distances and density on: '
          '{}-{}-{} {}:{}:{}'.format(
           now.year, now.month, now.day, now.hour, now.minute, now.second))
    vg.graph.list_properties()

    if vtp_files_base is not None:
        print('\nConverting the VoxelGraph to a VTK PolyData object and '
              'writing it to .vtp files...')
        poly_verts, poly_lines = vg.graph_to_points_and_lines_polys()
        io.save_vtp(poly_verts, vtp_files_base + '.vertices.vtp')
        io.save_vtp(poly_lines, vtp_files_base + '.edges.vtp')

    t_end = time.time()
    duration = t_end - t_begin
    print('Elapsed time: {} s'.format(duration))

    return densities


def run_build_graph_from_np_ndarray_and_calculate_density(
        mem_mask, ribo_mask, scale_factor_to_nm=1, vtp_files_base=None,
        verbose=False):
    """
    Builds a graph from a membrane mask and calculates ribosome density for each
    membrane voxel.

    Args:
        mem_mask (numpy ndarray): binary membrane mask in form of 3D array,
            segmenting the underlying tomogram into membrane (voxels with value
            1) and background (voxels with value 0)
        ribo_mask (numpy ndarray): binary mask of ribosomes centers on membrane
            in form of 3D array, where a voxel with value 1 means a particle is
            present at that membrane coordinate
        scale_factor_to_nm (float, optional): pixel size in nanometers, default
            1 (if no scaling to nanometers is desired)
        vtp_files_base (str, optional): If not None (default None), the
            VoxelGraph is converted to VTK PolyData points and lines objects and
            written to '<vtp_files_base>.vertices.vtp' and
            '<vtp_files_base>.edges.vtp' files, respectively
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        a numpy ndarray (with the same shape as the masks) with the densities

    Note:
        Both masks - mem_mask and ribo_mask - have to have the same shape.
    """
    t_begin = time.time()
    assert (mem_mask.shape == ribo_mask.shape)

    # Build a graph from the membrane mask:
    now = datetime.now()
    print('\nStarting building the VoxelGraph on: {}-{}-{} {}:{}:{}'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second))
    vg = VoxelGraph()
    vg.build_graph_from_np_ndarray(mem_mask, verbose)
    now = datetime.now()
    print('\nFinished building the graph on: {}-{}-{} {}:{}:{}'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second))
    print(vg.graph)
    vg.graph.list_properties()

    # Calculate shortest distances for each node in the graph (membrane voxel)
    # to each reachable voxel of ribosome center mapped on the membrane, and
    # from the distances a density measure of ribosomes at each membrane voxel:
    now = datetime.now()
    print('\nStarting calculating shortest distances to each reachable '
          'ribosome center and ribosome density for each membrane voxel on: '
          '{}-{}-{} {}:{}:{}'.format(
           now.year, now.month, now.day, now.hour, now.minute, now.second))
    densities = vg.calculate_density(
        mem_mask.shape[0], mem_mask.shape[1], mem_mask.shape[2],
        scale_factor_to_nm, mask=ribo_mask, verbose=verbose)
    now = datetime.now()
    print('\nFinished calculating the shortest distances and density on: '
          '{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day,
                                     now.hour, now.minute, now.second))
    vg.graph.list_properties()

    if vtp_files_base is not None:
        print('\nConverting the VoxelGraph to a VTK PolyData object and '
              'writing it to .vtp files...')
        poly_verts, poly_lines = vg.graph_to_points_and_lines_polys()
        io.save_vtp(poly_verts, vtp_files_base + '.vertices.vtp')
        io.save_vtp(poly_lines, vtp_files_base + '.edges.vtp')

    t_end = time.time()
    duration = t_end - t_begin
    print('Elapsed time: {} s'.format(duration))

    return densities


def main():
    """
    Main function with an examplary calculation of ribosome density on
    ER-membranes for a tomogram.

    Ribosome coordinates had been rescaled from bin 3 to bin 6 and bin 6
    membrane segmentation mask was used.

    Returns:
        None
    """
    # Change those parameters for each tomogram:
    tomo = 't85'
    tm_fold = ('/fs/pool/pool-ruben/Maria/Felix_ribosomes_and_Htt/'
               'bin3OPS1.263nm/02_template_matching/' + tomo +
               '/etomo_cleaned_notcorr_Felix/')
    class_fold = ('/fs/pool/pool-ruben/Maria/Felix_ribosomes_and_Htt/'
                  'bin3OPS1.263nm/04_classifications_refined_membrane/' + tomo +
                  '/6classes/')
    output_fold = ('/fs/pool/pool-ruben/Maria/Felix_ribosomes_and_Htt/'
                   'bin3OPS1.263nm/05_ribosome_density/' + tomo + '/')
    membrane_mask_file = tm_fold + 'mask_membrane_final_bin6.mrc'
    membrane_graph_file = output_fold + 'graph_membrane_final_bin6.gt'
    ribosome_mask_file = (class_fold +
                          'sec61_centers_inside_membrane_r4_bin6.mrc')
    ribosome_densities_file = (output_fold +
                               'densities_sec61_centers_r4_bin6.mrc')
    vtp_files_base = output_fold + 'densities_sec61_centers_r4_bin6'
    pixel_size_nm = 2.526

    # To check how many of the target voxels from the ribosome mask are inside
    # the membrane mask:
    # membrane_mask = read_in_mask(membrane_mask_file)
    # ribosome_mask = read_in_mask(ribosome_mask_file)
    # get_target_voxels_in_membrane_mask(ribosome_mask, membrane_mask)

    # To generate the graph and calculate the densities:
    # membrane_mask = read_in_mask(membrane_mask_file)
    # ribosome_mask = read_in_mask(ribosome_mask_file)
    # ribosome_densities = \
    # run_build_graph_from_np_ndarray_and_calculate_density(
    #     membrane_mask, ribosome_mask, pixel_size_nm, vtp_files_base
    # )
    # io.save_numpy(ribosome_densities, ribosome_densities_file)

    # To generate and save the graph only:
    membrane_mask = read_in_mask(membrane_mask_file)
    run_build_graph_from_np_ndarray(membrane_mask, membrane_graph_file,
                                    pixel_size_nm)

    # To calculate the densities using the saved graph:
    ribosome_mask = read_in_mask(ribosome_mask_file)
    ribosome_densities = run_calculate_density(
        membrane_graph_file, ribosome_mask, pixel_size_nm, vtp_files_base
    )
    io.save_numpy(ribosome_densities, ribosome_densities_file)


if __name__ == "__main__":
    main()
