import time
from graph_tool import Graph, load_graph
import pandas as pd
from os.path import isfile
import click

from pysurf import (pysurf_io as io, TriangleGraph, calculate_distances,
                    calculate_distances_and_thicknesses, calculate_thicknesses,
                    normals_estimation)


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


def generate_pm_er_graphs_and_surface(
        segmentation_mrc_file, scale_factor_to_nm,
        pm_graph_outfile, er_surf_outfile, er_graph_outfile,
        pm_surf_outfile=None, lbl_pm=1, lbl_er=2, lbl_between_pm_er=4):
    """
    Extracts PM and cER membrane surfaces from a segmentations with labels for
    both membranes and a space between them.

    Args:
        segmentation_mrc_file (string): segmentation '.mrc' file path
        scale_factor_to_nm (float): pixel size in nanometers for scaling the
            surface and the graph
        pm_graph_outfile (string): PM graph '.gt' output file
        er_surf_outfile (string): cER surface '.vtp' output file
        er_graph_outfile (string): cER graph '.gt' output file
        pm_surf_outfile (string, optional): PM surface '.vtp' output file,
            if None (default) not generated
        lbl_pm (int, optional): label of PM membrane (default 1)
        lbl_er (int, optional): label of cER membrane (default 2)
        lbl_between_pm_er (int, optional): label of inter-membrane space
            (default 4)

    Returns:
        None
    """
    # Extract the three masks:
    segmentation = io.load_tomo(segmentation_mrc_file)
    # Generate isosurface around the mask in between the membranes,
    # first applying the PM mask:
    pm_surface = io.gen_isosurface(segmentation, lbl_between_pm_er, mask=lbl_pm,
                                   sg=1, thr=THRESH_SIGMA1)
    # second applying the cER mask:
    er_surface = io.gen_isosurface(segmentation, lbl_between_pm_er, mask=lbl_er,
                                   sg=1, thr=THRESH_SIGMA1)
    # Generate graphs and remove 3 pixels from borders:
    pm_tg = TriangleGraph()
    pm_tg.build_graph_from_vtk_surface(pm_surface, scale_factor_to_nm)
    print('The raw PM graph has {} vertices and {} edges'.format(
            pm_tg.graph.num_vertices(), pm_tg.graph.num_edges()))
    pm_tg.find_vertices_near_border(MAX_DIST_SURF * scale_factor_to_nm,
                                    purge=True)
    print('The cleaned PM graph has {} vertices and {} edges'.format(
            pm_tg.graph.num_vertices(), pm_tg.graph.num_edges()))
    if pm_tg.graph.num_vertices() == 0:
        raise IOError("Graph does not have vertices")

    er_tg = TriangleGraph()
    er_tg.build_graph_from_vtk_surface(er_surface, scale_factor_to_nm)
    print('The raw cER graph has {} vertices and {} edges'.format(
            er_tg.graph.num_vertices(), er_tg.graph.num_edges()))
    er_tg.find_vertices_near_border(MAX_DIST_SURF * scale_factor_to_nm,
                                    purge=True)
    print('The cleaned cER graph has {} vertices and {} edges'.format(
            er_tg.graph.num_vertices(), er_tg.graph.num_edges()))
    if er_tg.graph.num_vertices() == 0:
        raise IOError("Graph does not have vertices")

    # Save final PM and cER graphs as .gt and PM and cER surface as .vtp files:
    pm_tg.graph.save(pm_graph_outfile)
    er_tg.graph.save(er_graph_outfile)
    if pm_surf_outfile is not None:
        pm_surf_clean = pm_tg.graph_to_triangle_poly()
        io.save_vtp(pm_surf_clean, pm_surf_outfile)
    er_surf_clean = er_tg.graph_to_triangle_poly()
    io.save_vtp(er_surf_clean, er_surf_outfile)


def generate_er_lumen_graph_and_surface(
        segmentation_mrc_file, scale_factor_to_nm,
        er_surf_outfile, er_graph_outfile, lbl_er=2, lbl_er_lumen=3):
    """
    Extracts inner cER membrane surface from a segmentation with labels for
    cER membrane and its lumen.

    Args:
        segmentation_mrc_file (string): segmentation '.mrc' file path
        scale_factor_to_nm (float): pixel size in nanometers for scaling the
            surface and the graph
        er_surf_outfile (string): cER surface '.vtp' output file
        er_graph_outfile (string): cER graph '.gt' output file
        lbl_er (int, optional): label of cER membrane (default 2)
        lbl_er_lumen (int, optional): label of cER lumen (default=3)

    Returns:
        None
    """
    # Extract the three masks:
    segmentation = io.load_tomo(segmentation_mrc_file)
    # Generate isosurface around the mask of cER lumen:
    er_surface = io.gen_isosurface(segmentation, lbl_er_lumen, mask=lbl_er,
                                   sg=1, thr=THRESH_SIGMA1)
    # Generate cER graph and remove 3 pixels from borders:
    er_tg = TriangleGraph()
    er_tg.build_graph_from_vtk_surface(er_surface, scale_factor_to_nm)
    print('The raw cER graph has {} vertices and {} edges'.format(
            er_tg.graph.num_vertices(), er_tg.graph.num_edges()))
    er_tg.find_vertices_near_border(MAX_DIST_SURF * scale_factor_to_nm,
                                    purge=True)
    print('The cleaned cER graph has {} vertices and {} edges'.format(
            er_tg.graph.num_vertices(), er_tg.graph.num_edges()))
    if er_tg.graph.num_vertices() == 0:
        raise IOError("Graph does not have vertices")

    # Save final cER graph as .gt and cER surface as .vtp files:
    er_tg.graph.save(er_graph_outfile)
    er_surf_clean = er_tg.graph_to_triangle_poly()
    io.save_vtp(er_surf_clean, er_surf_outfile)


def run_calculate_distances(
        pm_graph_file, er_surf_file, er_graph_file, er_surf_outfile,
        er_graph_outfile, distances_outfile, maxdist, offset=0,
        both_directions=True, reverse_direction=False, verbose=False):
    """
    A script running calculate_distances with graphs and surface loaded from
    files, transforming the resulting graph to a surface with triangles and
    saving the resulting graph and surface into files.

    Args:
        pm_graph_file (str): .gt input file with the PM TriangleGraph with
            corrected normals
        er_surf_file (str): .vtp input file with the cER vtkPolyData surface
        er_graph_file (str): .gt input file with the cER TriangleGraph
        er_surf_outfile (str): .vtp output file for the cER vtkPolyData surface
            with distances
        er_graph_outfile (str): .gt output file for the cER TriangleGraph with
            distances
        distances_outfile (str): .csv output file for the distances list
        maxdist (float): maximal distance (nm) from PM to the cER membrane
        offset (float, optional): positive or negative offset (nm, default 0)
            to add to the distances, depending on how the surfaces where
            generated and/or in order to account for membrane thickness
        both_directions (boolean, optional): if True, look in both directions of
            each PM normal (default), otherwise only in the normal direction
        reverse_direction (boolean, optional): if True, look in opposite
            direction of each PM normals (default=False; if both_directions
            True, will look in both directions)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        None
    """
    # Load the input files:
    tg_PM = TriangleGraph()
    tg_PM.graph = load_graph(pm_graph_file)
    poly_cER = io.load_poly(er_surf_file)
    tg_cER = TriangleGraph()
    tg_cER.graph = load_graph(er_graph_file)

    # Calculate distances:
    d1s = calculate_distances(tg_PM, tg_cER, poly_cER, maxdist, offset,
                              both_directions, reverse_direction, verbose)
    print("{} d1s".format(len(d1s)))
    # Save the distances into distances_outfile:
    df = pd.DataFrame()
    df["d1"] = d1s
    df.to_csv(distances_outfile, sep=';')

    # Transform the modified graph to a surface with triangles:
    cER_surf_dist = tg_cER.graph_to_triangle_poly()
    # Save the modified graph and surface into files:
    tg_cER.graph.save(er_graph_outfile)
    io.save_vtp(cER_surf_dist, er_surf_outfile)


def run_calculate_thicknesses(
        pm_graph_file, er_surf_file, er_graph_file,
        er_surf_outfile, er_graph_outfile, thicknesses_outfile,
        maxdist, maxthick, offset=0.0, both_directions=True,
        reverse_direction=False, verbose=False):
    """
    A script running calculate_thicknesses with graphs and surface loaded from
    files, transforming the resulting graph to a surface with triangles and
    saving the resulting graph and surface into files.

    Args:
        pm_graph_file (str): .gt input file with the PM TriangleGraph with
            corrected normals
        er_surf_file (str): .vtp input file with the cER vtkPolyData surface
        er_graph_file (str): .gt input file with the  cER TriangleGraph
        er_surf_outfile: .vtp output file with the cER vtkPolyData surface
            with thicknesses
        er_graph_outfile: .gt output file with the cER TriangleGraph with
            thicknesses
        thicknesses_outfile: .csv output file for the thicknesses list
        maxdist (float): maximal distance (nm) from PM to the cER membrane
        maxthick (float): maximal distance (nm) from first to second cER
            membrane
        offset (float, optional): positive or negative offset (nm, default 0)
            to add to the distances, depending on how the surfaces where
            generated and/or in order to account for membrane thickness
        both_directions (boolean, optional): if True, look in both directions of
            each PM normal (default), otherwise only in the normal direction
        reverse_direction (boolean, optional): if True, look in opposite
            direction of each PM normals (default=False; if both_directions
            True, will look in both directions)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        None
    """
    # Load the input files:
    tg_PM = TriangleGraph()
    tg_PM.graph = load_graph(pm_graph_file)
    poly_cER = io.load_poly(er_surf_file)
    tg_cER = TriangleGraph()
    tg_cER.graph = load_graph(er_graph_file)

    # Calculate distances:
    d2s = calculate_thicknesses(
        tg_PM, tg_cER, poly_cER, maxdist, maxthick, offset,
        both_directions, reverse_direction, verbose)
    print("{} d2s".format(len(d2s)))
    # Save the distances into distances_outfile:
    df = pd.DataFrame()
    df["d2"] = d2s
    df.to_csv(thicknesses_outfile, sep=';')

    # Transform the modified graph to a surface with triangles:
    poly_cER_thick = tg_cER.graph_to_triangle_poly()
    # Save the modified graph and surface into files:
    tg_cER.graph.save(er_graph_outfile)
    io.save_vtp(poly_cER_thick, er_surf_outfile)


def run_calculate_distances_and_thicknesses(
        pm_graph_file, er_surf_file, er_graph_file, er_surf_outfile,
        er_graph_outfile, distances_outfile, maxdist, maxthick, verbose=False):
    """
    A script running calculate_distances_and_thicknesses with graphs and surface
    loaded from files, transforming the resulting graph to a surface with
    triangles and saving the resulting graph and surface into files.

    Args:
        pm_graph_file (str): .gt input file with the PM TriangleGraph with
            corrected normals
        er_surf_file (str): .vtp input file with the cER vtkPolyData surface
        er_graph_file (str): .gt input file with the cER TriangleGraph
        er_surf_outfile (str): .vtp output file for the cER vtkPolyData surface
        er_graph_outfile (str): .gt output file for the cER TriangleGraph
        distances_outfile (str): .csv output file for the two distances lists
        maxdist (float): maximal distance (nm) from PM to first cER membrane
        maxthick (float): maximal distance (nm) from first to second cER
            membrane
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        None
    """
    # Load the input files:
    tg_PM = TriangleGraph()
    tg_PM.graph = load_graph(pm_graph_file)
    poly_cER = io.load_poly(er_surf_file)
    tg_cER = TriangleGraph()
    tg_cER.graph = load_graph(er_graph_file)

    # Calculate distances:
    d1s, d2s = calculate_distances_and_thicknesses(
        tg_PM, tg_cER, poly_cER, maxdist, maxthick, verbose)
    print("{} d1s".format(len(d1s)))
    print("{} d2s".format(len(d2s)))
    # Save the distances into distances_outfile:
    df = pd.DataFrame()
    df["d1"] = d1s
    df["d2"] = d2s
    df.to_csv(distances_outfile, sep=';')

    # Transform the resulting graph to a surface with triangles:
    cER_surf_dist = tg_cER.graph_to_triangle_poly()
    # Save the resulting graph and surface into files:
    tg_cER.graph.save(er_graph_outfile)
    io.save_vtp(cER_surf_dist, er_surf_outfile)


@click.command()
@click.argument('fold', type=str)
@click.argument('segmentation_file', type=str)
@click.argument('base_filename', type=str)
@click.option('-lbl_pm', type=int, default=1,
              help="label of PM membrane (default=1)")
@click.option('-lbl_er', type=int, default=2,
              help="label of cER membrane (default=2)")
@click.option('-lbl_between_pm_er', type=int, default=4,
              help="label of inter-membrane space (default=4)")
@click.option('-lbl_er_lumen', type=int, default=3,
              help="label of cER lumen (default=3)")
@click.option('-pixel_size_nm', type=float, default=1.368,
              help="pixel size in nm of the segmentation (default=1.368)")
@click.option('-radius_hit', type=float, default=10,
              help="neighborhood parameter for PM normals estimation by VV "
                   "(default=10)")
@click.option('-maxdist_nm', type=float, default=50,
              help="maximal distance in nm, should be bigger than the largest "
                   "possible distance, for the algorithm to stop searching "
                   "(default=50)")
@click.option('-maxthick_nm', type=float, default=80,
              help="maximal distance between the two cER membrane sides in nm, "
                   "should be bigger than the largest possible distance, for "
                   "the algorithm to stop searching (default=80)")
@click.option('-offset_voxels', type=int, default=1,
              help="offset in voxels, will be added to the distances, "
                   "(default=1, because surfaces are generated 1/2 voxel off "
                   "the membrane segmentation boundary towards the "
                   "inter-membrane space)")
@click.option('-both_directions', type=bool, default=True,
              help="if True, look in both directions of each PM normal "
                   "(default), otherwise only in the normal direction")
@click.option('-reverse_direction', type=bool, default=False,
              help="if True, look in opposite direction of each PM normals "
                   "(default=False; if both_directions True, will look in both "
                   "directions)")
def distances_and_thicknesses_calculation(
        fold, segmentation_file, base_filename,
        lbl_pm=1, lbl_er=2, lbl_between_pm_er=4, lbl_er_lumen=3,
        pixel_size_nm=1.368, radius_hit=10, maxdist_nm=50, maxthick_nm=80,
        offset_voxels=1, both_directions=True, reverse_direction=False):
    """Takes input/output folder, input segmentation MRC file and base name for
    output files and calculates distances between two cER membrane sides."""
    offset_nm = offset_voxels * pixel_size_nm
    if not fold.endswith('/'):
        fold += '/'
    segmentation_file = '{}{}'.format(fold, segmentation_file)
    pm_surf_file = '{}{}.PM.vtp'.format(fold, base_filename)
    pm_graph_file = '{}{}.PM.gt'.format(fold, base_filename)
    er_surf_file = '{}{}.cER.vtp'.format(fold, base_filename)
    er_graph_file = '{}{}.cER.gt'.format(fold, base_filename)
    pm_normals_surf_file = '{}{}.PM.NVV_rh{}.vtp'.format(
        fold, base_filename, radius_hit)
    pm_normals_graph_file = '{}{}.PM.NVV_rh{}.gt'.format(
        fold, base_filename, radius_hit)
    er_dist_surf_file = '{}{}.cER.distancesFromPM.vtp'.format(
        fold, base_filename)
    er_dist_graph_file = '{}{}.cER.distancesFromPM.gt'.format(
        fold, base_filename)
    distances_outfile = '{}{}.cER.distancesFromPM.csv'.format(
        fold, base_filename)
    inner_er_surf_file = '{}{}.innercER.vtp'.format(fold, base_filename)
    inner_er_graph_file = '{}{}.innercER.gt'.format(fold, base_filename)
    inner_er_thick_surf_file = '{}{}.innercER.thicknesses.vtp'.format(
        fold, base_filename)
    inner_er_thick_graph_file = '{}{}.innercER.thicknesses.gt'.format(
        fold, base_filename)
    thicknesses_outfile = '{}{}.innercER.thicknesses.csv'.format(
        fold, base_filename)

    if (not isfile(pm_surf_file) or not isfile(pm_graph_file) or not
            isfile(er_surf_file) or not isfile(er_graph_file)):
        print('Generating PM and cER graphs and surface files')
        generate_pm_er_graphs_and_surface(
            segmentation_file, pixel_size_nm,
            pm_graph_file, er_surf_file, er_graph_file, pm_surf_file,
            lbl_pm, lbl_er, lbl_between_pm_er)
    if not isfile(pm_normals_graph_file):
        print('Estimating normals for PM graph')
        pm_tg = TriangleGraph()
        pm_tg.graph = load_graph(pm_graph_file)
        normals_estimation(pm_tg, radius_hit)
        pm_tg.graph.save(pm_normals_graph_file)
        pm_surf = pm_tg.graph_to_triangle_poly()
        io.save_vtp(pm_surf, pm_normals_surf_file)
    if not isfile(distances_outfile):
        print('Calculating and saving distances between PM and cER')
        run_calculate_distances(
            pm_normals_graph_file, er_surf_file, er_graph_file,
            er_dist_surf_file, er_dist_graph_file, distances_outfile,
            maxdist_nm, offset_nm, both_directions, reverse_direction)
    if not isfile(inner_er_surf_file) or not isfile(inner_er_graph_file):
        print('Generating inner cER graphs and surface files')
        generate_er_lumen_graph_and_surface(
            segmentation_file, pixel_size_nm,
            inner_er_surf_file, inner_er_graph_file, lbl_er, lbl_er_lumen)
    # if not isfile(thicknesses_outfile):
    print('Calculating and saving cER thicknesses')
    run_calculate_thicknesses(
        pm_normals_graph_file, inner_er_surf_file, inner_er_graph_file,
        inner_er_thick_surf_file, inner_er_thick_graph_file,
        thicknesses_outfile, maxdist_nm, maxthick_nm, offset_nm,
        both_directions, reverse_direction)


def main_distances_and_thickness():
    # Input parameters:
    rh = 10
    pixel_size_nm = 1.368
    maxdist_nm = 80
    maxthick_nm = 80
    base_fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/"
    # The "famous" tcb (done cER RH=6 and 10 + PM RH=6):
    # fold = "{}tcb_t3_ny01/new_workflow/".format(base_fold)
    # base_filename = "t3_ny01_cropped_"
    # rh = 6
    # The new tcb (done cropped cER and PM with RH=15):
    # tomo = "tcb_170924_l2_t2_ny01"
    # fold = "{}{}/".format(base_fold, tomo)
    # base_filename = "{}_cropped_".format(tomo)
    fold = "{}TCB/170924_TITAN_l1_t1/".format(base_fold)
    base_filename = "TCBl1t1_"
    # The good scs (done cER and PM with RH=15):
    # tomo = "scs_171108_l1_t2_ny01"
    # fold = "{}{}/".format(base_fold, tomo)
    # base_filename = "{}_".format(tomo)

    # should exist:
    # File with scaled PM graph with corrected normals:
    PM_graph_file = "{}{}PM.NVV_rh{}_epsilon0_eta0.gt".format(
        fold, base_filename, rh)
    # Files with scaled cER surface and graph, after curvature calculation:
    cER_surf_file = "{}{}cER.VV_area2_rh{}_epsilon0_eta0.vtp".format(
        fold, base_filename, rh)
    cER_graph_file = "{}{}cER.VV_area2_rh{}_epsilon0_eta0.gt".format(
        fold, base_filename, rh)
    # will be generated:
    cER_surf_outfile = "{}.PMdist_maxdist{}_maxthick{}.vtp".format(
        cER_surf_file[0:-4], maxdist_nm, maxthick_nm)
    cER_graph_outfile = "{}.PMdist_maxdist{}_maxthick{}.gt".format(
        cER_graph_file[0:-3], maxdist_nm, maxthick_nm)
    distances_outfile = "{}.PMdist_maxdist{}_maxthick{}.csv".format(
        cER_surf_file[0:-4], maxdist_nm, maxthick_nm)

    run_calculate_distances_and_thicknesses(
        PM_graph_file, cER_surf_file, cER_graph_file, cER_surf_outfile,
        cER_graph_outfile, distances_outfile, maxdist_nm, maxthick_nm,
        verbose=False)


if __name__ == "__main__":
    t_begin = time.time()

    distances_and_thicknesses_calculation()
    # main_distances_and_thickness()

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))
