import time
from graph_tool import Graph, load_graph
import pandas as pd
from os.path import isfile
import click
import sys

from pysurf import (pysurf_io as io, TriangleGraph, calculate_distances,
                    calculate_thicknesses, normals_estimation)


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


def generate_mem1_mem2_graphs_and_surface(
        segmentation_mrc_file, scale_factor_to_nm, mem1_graph_outfile,
        mem2_surf_outfile, mem2_graph_outfile, mem1_surf_outfile=None,
        lbl_mem1=1, lbl_mem2=2, lbl_between_mem1_mem2=4, mem1="PM", mem2="cER"):
    """
    Extracts two membrane surfaces from a segmentations with labels for
    both membranes and a space between them.

    Args:
        segmentation_mrc_file (string): segmentation '.mrc' file path
        scale_factor_to_nm (float): pixel size in nanometers for scaling the
            surface and the graph
        mem1_graph_outfile (string): first surface graph '.gt' output file
        mem2_surf_outfile (string): second surface '.vtp' output file
        mem2_graph_outfile (string): second surface graph '.gt' output file
        mem1_surf_outfile (string, optional): first surface '.vtp' output file,
            if None (default) not generated
        lbl_mem1 (int, optional): label of first membrane (default 1)
        lbl_mem2 (int, optional): label of second membrane (default 2)
        lbl_between_mem1_mem2 (int, optional): label of inter-membrane space
            (default 4)
        mem1 (str, optional): name of the first membrane (default "PM")
        mem2 (str, optional): name of the second membrane (default "cER")

    Returns:
        None
    """
    # Extract the three masks:
    segmentation = io.load_tomo(segmentation_mrc_file)
    # Generate isosurface around the mask in between the membranes,
    # first applying the first membrane mask:
    mem1_surface = io.gen_isosurface(
        segmentation, lbl_between_mem1_mem2, mask=lbl_mem1, sg=1,
        thr=THRESH_SIGMA1)
    # second applying the second membrane mask:
    mem2_surface = io.gen_isosurface(
        segmentation, lbl_between_mem1_mem2, mask=lbl_mem2, sg=1,
        thr=THRESH_SIGMA1)
    # Generate graphs and remove 3 pixels from borders:
    mem1_tg = TriangleGraph()
    mem1_tg.build_graph_from_vtk_surface(mem1_surface, scale_factor_to_nm)
    print('The raw {} graph has {} vertices and {} edges'.format(
            mem1, mem1_tg.graph.num_vertices(), mem1_tg.graph.num_edges()))
    mem1_tg.find_vertices_near_border(
        MAX_DIST_SURF * scale_factor_to_nm, purge=True)
    print('The cleaned {} graph has {} vertices and {} edges'.format(
            mem1, mem1_tg.graph.num_vertices(), mem1_tg.graph.num_edges()))
    if mem1_tg.graph.num_vertices() == 0:
        raise IOError("Graph does not have vertices")

    mem2_tg = TriangleGraph()
    mem2_tg.build_graph_from_vtk_surface(mem2_surface, scale_factor_to_nm)
    print('The raw {} graph has {} vertices and {} edges'.format(
            mem2, mem2_tg.graph.num_vertices(), mem2_tg.graph.num_edges()))
    mem2_tg.find_vertices_near_border(
        MAX_DIST_SURF * scale_factor_to_nm, purge=True)
    print('The cleaned {} graph has {} vertices and {} edges'.format(
            mem2, mem2_tg.graph.num_vertices(), mem2_tg.graph.num_edges()))
    if mem2_tg.graph.num_vertices() == 0:
        raise IOError("Graph does not have vertices")

    # Save final graphs as .gt and surfaces as .vtp files:
    mem1_tg.graph.save(mem1_graph_outfile)
    mem2_tg.graph.save(mem2_graph_outfile)
    if mem1_surf_outfile is not None:
        mem1_surf_clean = mem1_tg.graph_to_triangle_poly()
        io.save_vtp(mem1_surf_clean, mem1_surf_outfile)
    mem2_surf_clean = mem2_tg.graph_to_triangle_poly()
    io.save_vtp(mem2_surf_clean, mem2_surf_outfile)


def generate_mem_lumen_graph_and_surface(
        segmentation_mrc_file, scale_factor_to_nm, mem_surf_outfile,
        mem_graph_outfile, lbl_mem=2, lbl_mem_lumen=3, mem="cER"):
    """
    Extracts inner membrane surface from a segmentation with labels for
    the membrane and its lumen.

    Args:
        segmentation_mrc_file (string): segmentation '.mrc' file path
        scale_factor_to_nm (float): pixel size in nanometers for scaling the
            surface and the graph
        mem_surf_outfile (string): membrane surface '.vtp' output file
        mem_graph_outfile (string): membrane graph '.gt' output file
        lbl_mem (int, optional): label of the membrane (default 2)
        lbl_mem_lumen (int, optional): label of the membrane lumen (default=3)
        mem (str, optional): name of the first membrane (default "cER")

    Returns:
        None
    """
    # Extract the three masks:
    segmentation = io.load_tomo(segmentation_mrc_file)
    # Generate isosurface around the mask of membrane lumen:
    mem_surface = io.gen_isosurface(
        segmentation, lbl_mem_lumen, mask=lbl_mem, sg=1, thr=THRESH_SIGMA1)
    # Generate graph and remove 3 pixels from borders:
    mem_tg = TriangleGraph()
    mem_tg.build_graph_from_vtk_surface(mem_surface, scale_factor_to_nm)
    print('The raw {} graph has {} vertices and {} edges'.format(
            mem, mem_tg.graph.num_vertices(), mem_tg.graph.num_edges()))
    mem_tg.find_vertices_near_border(
        MAX_DIST_SURF * scale_factor_to_nm, purge=True)
    print('The cleaned {} graph has {} vertices and {} edges'.format(
            mem, mem_tg.graph.num_vertices(), mem_tg.graph.num_edges()))
    if mem_tg.graph.num_vertices() == 0:
        raise IOError("Graph does not have vertices")

    # Save final graph as .gt and surface as .vtp files:
    mem_tg.graph.save(mem_graph_outfile)
    mem_surf_clean = mem_tg.graph_to_triangle_poly()
    io.save_vtp(mem_surf_clean, mem_surf_outfile)


def run_calculate_distances(
        mem1_graph_file, mem2_surf_file, mem2_graph_file, mem2_surf_outfile,
        mem2_graph_outfile, distances_outfile, maxdist, offset=0,
        both_directions=True, reverse_direction=False, mem1="PM",
        verbose=False):
    """
    A script running calculate_distances with graphs and surface loaded from
    files, transforming the resulting graph to a surface with triangles and
    saving the resulting graph and surface into files.

    Args:
        mem1_graph_file (str): .gt input file with the first membrane's
            TriangleGraph with corrected normals
        mem2_surf_file (str): .vtp input file with the second membrane's
            vtkPolyData surface
        mem2_graph_file (str): .gt input file with the second membrane's
            TriangleGraph
        mem2_surf_outfile (str): .vtp output file for the second membrane's
            vtkPolyData surface with distances
        mem2_graph_outfile (str): .gt output file for the second membrane's
            TriangleGraph with distances
        distances_outfile (str): .csv output file for the distances list
        maxdist (float): maximal distance (nm) from the first to the second
            membrane
        offset (float, optional): positive or negative offset (nm, default 0)
            to add to the distances, depending on how the surfaces where
            generated and/or in order to account for membrane thickness
        both_directions (boolean, optional): if True, look in both directions of
            each first membrane's normal (default), otherwise only in the normal
            direction
        reverse_direction (boolean, optional): if True, look in opposite
            direction of each first membrane's normals (default=False;
            if both_directions True, will look in both directions)
        mem1 (str, optional): name of the first membrane (default "PM")
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        None
    """
    # Load the input files:
    tg_mem1 = TriangleGraph()
    tg_mem1.graph = load_graph(mem1_graph_file)
    surf_mem2 = io.load_poly(mem2_surf_file)
    tg_mem2 = TriangleGraph()
    tg_mem2.graph = load_graph(mem2_graph_file)

    # Calculate distances:
    d1s = calculate_distances(
        tg_mem1, tg_mem2, surf_mem2, maxdist, offset, both_directions,
        reverse_direction, mem1, verbose)
    print("{} d1s".format(len(d1s)))
    # Save the distances into distances_outfile:
    df = pd.DataFrame()
    df["d1"] = d1s
    df.to_csv(distances_outfile, sep=';')

    # Transform the modified graph to a surface with triangles:
    mem2_surf_dist = tg_mem2.graph_to_triangle_poly()
    # Save the modified graph and surface into files:
    tg_mem2.graph.save(mem2_graph_outfile)
    io.save_vtp(mem2_surf_dist, mem2_surf_outfile)


def run_calculate_thicknesses(
        mem1_graph_file, mem2_surf_file, mem2_graph_file,
        mem2_surf_outfile, mem2_graph_outfile, thicknesses_outfile,
        maxdist, maxthick, offset=0.0, both_directions=True,
        reverse_direction=False, mem2="cER", verbose=False):
    """
    A script running calculate_thicknesses with graphs and surface loaded from
    files, transforming the resulting graph to a surface with triangles and
    saving the resulting graph and surface into files.

    Args:
        mem1_graph_file (str): .gt input file with the first membrane's
            TriangleGraph with corrected normals
        mem2_surf_file (str): .vtp input file with the the second membrane's
        vtkPolyData surface
        mem2_graph_file (str): .gt input file with the  the second membrane's
            TriangleGraph
        mem2_surf_outfile: .vtp output file with the the second membrane's
            vtkPolyData surface with thicknesses
        mem2_graph_outfile: .gt output file with the the second membrane's
            TriangleGraph with thicknesses
        thicknesses_outfile: .csv output file for the thicknesses list
        maxdist (float): maximal distance (nm) from the first to the second
            membrane
        maxthick (float): maximal thickness (nm) of the second organelle
        offset (float, optional): positive or negative offset (nm, default 0)
            to add to the distances, depending on how the surfaces where
            generated and/or in order to account for membrane thickness
        both_directions (boolean, optional): if True, look in both directions of
            each first membrane's normal (default), otherwise only in the normal
            direction
        reverse_direction (boolean, optional): if True, look in opposite
            direction of each first membrane's normals (default=False;
            if both_directions True, will look in both directions)
        mem2 (str, optional): name of the second membrane (default "cER")
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        None
    """
    # Load the input files:
    tg_mem1 = TriangleGraph()
    tg_mem1.graph = load_graph(mem1_graph_file)
    surf_mem2 = io.load_poly(mem2_surf_file)
    tg_mem2 = TriangleGraph()
    tg_mem2.graph = load_graph(mem2_graph_file)

    # Calculate distances:
    d2s = calculate_thicknesses(
        tg_mem1, tg_mem2, surf_mem2, maxdist, maxthick, offset,
        both_directions, reverse_direction, mem2, verbose)
    print("{} d2s".format(len(d2s)))
    # Save the distances into distances_outfile:
    df = pd.DataFrame()
    df["d2"] = d2s
    df.to_csv(thicknesses_outfile, sep=';')

    # Transform the modified graph to a surface with triangles:
    mem2_surf_thick = tg_mem2.graph_to_triangle_poly()
    # Save the modified graph and surface into files:
    tg_mem2.graph.save(mem2_graph_outfile)
    io.save_vtp(mem2_surf_thick, mem2_surf_outfile)


def extract_distances(
        fold, base_filename, name, exclude_borders=1):
    """
    Extracts distances information from a .gt file into a .csv file. By default,
    values within 1 nm to surface borders are excluded.

    Args:
        fold (str): path where the input is and where the output will be written
        base_filename (str): base file name for input and output files
        name (str): name of the property to extract (e.g., 'PMdistance' or
            'cERthickness')
        exclude_borders (int, optional): if > 0, triangles within this distance
            from borders in nm and corresponding values will be excluded from
            the output files (graph .gt, surface.vtp file and .csv)

    Returns:
        None
    """
    # input graph and surface files
    gt_infile = '{}{}.gt'.format(fold, base_filename)
    # output csv, gt and vtp files
    csv_outfile = '{}{}.csv'.format(fold, base_filename)
    gt_outfile = None
    vtp_outfile = None
    if exclude_borders > 0:
        eb = "_excluding{}borders".format(exclude_borders)
        gt_outfile = '{}{}{}.gt'.format(fold, base_filename, eb)
        csv_outfile = '{}{}{}.csv'.format(fold, base_filename, eb)
        vtp_outfile = '{}{}{}.vtp'.format(fold, base_filename, eb)

    # Create TriangleGraph object and load the graph file
    tg = TriangleGraph()
    tg.graph = load_graph(gt_infile)

    _extract_distances_from_graph(
        tg, csv_outfile, exclude_borders, name, gt_outfile, vtp_outfile)


def _extract_distances_from_graph(
        tg, csv_file, exclude_borders, name, gt_file=None, vtp_file=None):
    # If don't want to include curvatures near borders, filter out those
    if exclude_borders > 0:
        tg.find_vertices_near_border(exclude_borders, purge=True)

    # Saving the changes into graph and surface files, if specified:
    if gt_file is not None:
        tg.graph.save(gt_file)
    if vtp_file is not None:
        # Transforming the resulting graph to a surface with triangles:
        surf = tg.graph_to_triangle_poly()
        io.save_vtp(surf, vtp_file)

    # Getting estimated principal curvatures from the output graph:
    distances = tg.get_vertex_property_array(name)
    triangle_areas = tg.get_vertex_property_array("area")

    # Writing all the curvature values and errors into a csv file:
    df = pd.DataFrame()
    df[name] = distances
    df["triangleAreas"] = triangle_areas
    df.to_csv(csv_file, sep=';')


# @click.command()
# @click.argument('fold', type=str)
# @click.argument('segmentation_file', type=str)
# @click.argument('base_filename', type=str)
# @click.option('-lbl_mem1', type=int, default=1,
#               help="label of the first membrane (default=1)")
# @click.option('-lbl_mem2', type=int, default=2,
#               help="label of the second membrane (default=2)")
# @click.option('-lbl_between_mem1_mem2', type=int, default=4,
#               help="label of inter-membrane space (default=4)")
# @click.option('-lbl_mem2_lumen', type=int, default=3,
#               help="label of the second lumen (default=3)")
# @click.option('-pixel_size_nm', type=float, default=1.368,
#               help="pixel size in nm of the segmentation (default=1.368)")
# @click.option('-radius_hit', type=float, default=10,
#               help="neighborhood parameter for the first membrane's normals "
#                    "estimation by VV (default=10)")
# @click.option('-maxdist_nm', type=float, default=50,
#               help="maximal distance in nm, should be bigger than the largest "
#                    "possible distance, for the algorithm to stop searching "
#                    "(default=50)")
# @click.option('-maxthick_nm', type=float, default=80,
#               help="maximal distance between the two sides of the second "
#                    "membrane in nm, should be bigger than the largest possible "
#                    "distance, for the algorithm to stop searching (default=80)")
# @click.option('-offset_voxels', type=int, default=1,
#               help="offset in voxels, will be added to the distances, "
#                    "(default=1, because surfaces are generated 1/2 voxel off "
#                    "the membrane segmentation boundary towards the "
#                    "inter-membrane space)")
# @click.option('-both_directions', type=bool, default=True,
#               help="if True, look in both directions of each the first "
#                    "membrane's normal (default), otherwise only in the normal "
#                    "direction")
# @click.option('-reverse_direction', type=bool, default=False,
#               help="if True, look in opposite direction of each the first "
#                    "membrane's normals (default=False; if both_directions "
#                    "True, will look in both directions)")
def distances_and_thicknesses_calculation(
        fold, segmentation_file, base_filename,
        lbl_mem1=1, lbl_mem2=2, lbl_between_mem1_mem2=4, lbl_mem2_lumen=3,
        pixel_size_nm=1.368, radius_hit=10, maxdist_nm=50, maxthick_nm=80,
        offset_voxels=1, both_directions=True, reverse_direction=False,
        mem1="PM", mem2="cER"):
    """Takes input/output folder, input segmentation MRC file and base name for
    output files and calculates distances between the first and the second
    membrane and thicknesses between two sides of the second membrane."""
    log_file = '{}{}.distances_and_thicknesses_calculation.log'.format(
        fold, base_filename)
    sys.stdout = open(log_file, 'a')

    offset_nm = offset_voxels * pixel_size_nm
    if not fold.endswith('/'):
        fold += '/'
    segmentation_file = '{}{}'.format(fold, segmentation_file)
    mem1_surf_file = '{}{}.{}.vtp'.format(fold, base_filename, mem1)
    mem1_graph_file = '{}{}.{}.gt'.format(fold, base_filename, mem1)
    mem2_surf_file = '{}{}.{}.vtp'.format(fold, base_filename, mem2)
    mem2_graph_file = '{}{}.{}.gt'.format(fold, base_filename, mem2)
    mem1_normals_surf_file = '{}{}.{}.NVV_rh{}.vtp'.format(
        fold, base_filename, mem1, radius_hit)
    mem1_normals_graph_file = '{}{}.{}.NVV_rh{}.gt'.format(
        fold, base_filename, mem1, radius_hit)
    mem2_dist_surf_file = '{}{}.{}.distancesFrom{}.vtp'.format(
        fold, base_filename, mem2, mem1)
    mem2_dist_graph_file = '{}{}.{}.distancesFrom{}.gt'.format(
        fold, base_filename, mem2, mem1)
    distances_outfile = '{}{}.{}.distancesFrom{}.csv'.format(
        fold, base_filename, mem2, mem1)
    inner_mem2_surf_file = '{}{}.inner{}.vtp'.format(fold, base_filename, mem2)
    inner_mem2_graph_file = '{}{}.inner{}.gt'.format(fold, base_filename, mem2)
    inner_mem2_thick_surf_file = '{}{}.inner{}.thicknesses.vtp'.format(
        fold, base_filename, mem2)
    inner_mem2_thick_graph_file = '{}{}.inner{}.thicknesses.gt'.format(
        fold, base_filename, mem2)
    thicknesses_outfile = '{}{}.inner{}.thicknesses.csv'.format(
        fold, base_filename, mem2)

    if (not isfile(mem1_surf_file) or not isfile(mem1_graph_file) or not
            isfile(mem2_surf_file) or not isfile(mem2_graph_file)):
        print('Generating {} and {} graphs and surface files'.format(
            mem1, mem2))
        generate_mem1_mem2_graphs_and_surface(
            segmentation_file, pixel_size_nm,
            mem1_graph_file, mem2_surf_file, mem2_graph_file, mem1_surf_file,
            lbl_mem1, lbl_mem2, lbl_between_mem1_mem2, mem1, mem2)
    if not isfile(mem1_normals_graph_file):
        print('Estimating normals for {} graph'.format(mem1))
        mem1_tg = TriangleGraph()
        mem1_tg.graph = load_graph(mem1_graph_file)
        normals_estimation(mem1_tg, radius_hit)
        mem1_tg.graph.save(mem1_normals_graph_file)
        mem1_surf = mem1_tg.graph_to_triangle_poly()
        io.save_vtp(mem1_surf, mem1_normals_surf_file)
    print('Calculating and saving distances between {} and {}'.format(
        mem1, mem2))
    run_calculate_distances(
        mem1_normals_graph_file, mem2_surf_file, mem2_graph_file,
        mem2_dist_surf_file, mem2_dist_graph_file, distances_outfile,
        maxdist_nm, offset_nm, both_directions, reverse_direction, mem1)
    if not isfile(inner_mem2_surf_file) or not isfile(inner_mem2_graph_file):
        print('Generating inner {} graphs and surface files'.format(mem2))
        generate_mem_lumen_graph_and_surface(
            segmentation_file, pixel_size_nm, inner_mem2_surf_file,
            inner_mem2_graph_file, lbl_mem2, lbl_mem2_lumen, mem2)
    print('Calculating and saving {} thicknesses'.format(mem2))
    run_calculate_thicknesses(
        mem1_normals_graph_file, inner_mem2_surf_file, inner_mem2_graph_file,
        inner_mem2_thick_surf_file, inner_mem2_thick_graph_file,
        thicknesses_outfile, maxdist_nm, maxthick_nm, offset_nm,
        both_directions, reverse_direction, mem2)


if __name__ == "__main__":
    t_begin = time.time()

    distances_and_thicknesses_calculation()

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))
