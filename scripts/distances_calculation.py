import time
from graph_tool import Graph, load_graph
import pandas as pd

from pysurf import (pysurf_io as io, TriangleGraph, calculate_distances,
                    calculate_distances_and_thicknesses, normals_estimation)


__author__ = 'kalemanov'

# CONSTANTS
MAX_DIST_SURF = 3
"""int: a constant determining the maximal distance in pixels of a point on the
surface from the segmentation mask, used in gen_isosurface and gen_surface
functions.
"""


def generate_graphs_and_surface(
        segmentation_mrc_file, scale_factor_to_nm,
        pm_graph_outfile, er_surf_outfile, er_graph_outfile,
        lbl_pm=1, lbl_er=2, lbl_between_pm_er=4):
    # Extract the three masks:
    segmentation = io.load_tomo(segmentation_mrc_file)
    # Generate isosurface around the mask in between the membranes,
    # first applying the PM mask:
    pm_surface = io.gen_isosurface(segmentation, lbl_between_pm_er, mask=lbl_pm)
    # second applying the cER mask:
    er_surface = io.gen_isosurface(segmentation, lbl_between_pm_er, mask=lbl_er)
    # Generate graphs and remove 3 pixels from borders:
    pm_tg = TriangleGraph()
    pm_tg.build_graph_from_vtk_surface(pm_surface, scale_factor_to_nm)
    pm_tg.find_vertices_near_border(MAX_DIST_SURF * scale_factor_to_nm,
                                    purge=True)
    er_tg = TriangleGraph()
    er_tg.build_graph_from_vtk_surface(er_surface, scale_factor_to_nm)
    er_tg.find_vertices_near_border(MAX_DIST_SURF * scale_factor_to_nm,
                                    purge=True)
    # Save final PM and cER graphs as .gt and cER surface as .vtp files:
    pm_tg.graph.save(pm_graph_outfile)
    er_tg.graph.save(er_graph_outfile)
    er_surf_clean = er_tg.graph_to_triangle_poly()
    io.save_vtp(er_surf_clean, er_surf_outfile)


def run_calculate_distances(
        pm_graph_file, er_surf_file, er_graph_file, er_surf_outfile,
        er_graph_outfile, distances_outfile, maxdist, verbose=False):
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
        er_graph_outfile (str): .gt output file for the cER TriangleGraph
        distances_outfile (str): .csv output file for the distances list
        maxdist (int): maximal distance (nm) from PM to the cER membrane
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
    d1s = calculate_distances(tg_PM, tg_cER, poly_cER, maxdist, verbose)
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
        maxdist (int): maximal distance (nm) from PM to first cER membrane
        maxthick (int): maximal distance (nm) from first to second cER membrane
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


def main_surfaces_normals_and_distances():
    fold = '/fs/pool/pool-ruben/Maria/4Javier/'
    segmentation_mrc_file = '{}t1_cropped.labels_thick_MARIA.mrc'.format(fold)
    pixel_size = 1.368
    pm_graph_file = '{}t1_cropped.PM.gt'.format(fold)
    er_surf_file = '{}t1_cropped.cER.vtp'.format(fold)
    er_graph_file = '{}t1_cropped.cER.gt'.format(fold)
    radius_hit = 10
    pm_normals_graph_file = '{}t1_cropped.PM.VVnormals.gt'.format(fold)
    er_dist_surf_file = '{}t1_cropped.cER.distancesFromPM.vtp'.format(fold)
    er_dist_graph_file = '{}t1_cropped.cER.distancesFromPM.gt'.format(fold)
    distances_outfile = '{}t1_cropped.cER.distancesFromPM.csv'.format(fold)
    maxdist_voxels = 60
    maxdist_nm = int(maxdist_voxels * pixel_size)

    print ('Generating PM and cER graphs and cER surface files')
    generate_graphs_and_surface(segmentation_mrc_file, pixel_size,
                                pm_graph_file, er_surf_file, er_graph_file)
    print('Estimating normals for PM graph')
    pm_tg = TriangleGraph()
    pm_tg.graph = load_graph(pm_graph_file)
    normals_estimation(pm_tg, radius_hit)
    pm_tg.graph.save(pm_normals_graph_file)
    print('Calculating and saving distances between PM and cER')
    run_calculate_distances(
        pm_normals_graph_file, er_surf_file, er_graph_file, er_dist_surf_file,
        er_dist_graph_file, distances_outfile, maxdist_nm)


def main_only_distances():
    fold = '/fs/pool/pool-ruben/Maria/4Javier/distances_test/'
    pm_normals_graph_file = '{}t1_cropped.PM.VVnormals.gt'.format(fold)
    er_surf_file = '{}t1_cropped.cER.vtp'.format(fold)
    er_graph_file = '{}t1_cropped.cER.gt'.format(fold)
    er_dist_surf_file = '{}t1_cropped.cER.distancesFromPM.vtp'.format(fold)
    er_dist_graph_file = '{}t1_cropped.cER.distancesFromPM.gt'.format(fold)
    distances_outfile = '{}t1_cropped.cER.distancesFromPM.csv'.format(fold)
    pixel_size = 1.368
    maxdist_voxels = 60
    maxdist_nm = int(maxdist_voxels * pixel_size)

    print('Calculating and saving distances between PM and cER')
    run_calculate_distances(
        pm_normals_graph_file, er_surf_file, er_graph_file, er_dist_surf_file,
        er_dist_graph_file, distances_outfile, maxdist_nm)


def main_distances_and_thickness():
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
    rh = 10
    pixel_size = 1.368

    # Input files:
    # File with scaled PM graph with corrected normals:
    PM_graph_file = "{}{}PM.NVV_rh{}_epsilon0_eta0.gt".format(
        fold, base_filename, rh)
    # Files with scaled cER surface and graph, after curvature calculation:
    cER_surf_file = "{}{}cER.VV_area2_rh{}_epsilon0_eta0.vtp".format(
        fold, base_filename, rh)
    cER_graph_file = "{}{}cER.VV_area2_rh{}_epsilon0_eta0.gt".format(
        fold, base_filename, rh)

    # Input parameters:
    maxdist_voxels = 60
    maxthick_voxels = 60
    maxdist_nm = int(maxdist_voxels * pixel_size)
    maxthick_nm = int(maxthick_voxels * pixel_size)

    # Output files:
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

    main_distances_and_thickness()
    # main_surfaces_normals_and_distances()
    # main_only_distances()

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))
