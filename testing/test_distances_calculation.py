import unittest
import numpy as np
import pandas as pd
from graph_tool import Graph, load_graph

from pysurf import TriangleGraph, pysurf_io as io
from scripts import (generate_graphs_and_surface, normals_estimation,
                     run_calculate_distances)

"""
Scripts for testing methods calculating distances between membrane surfaces
using "synthetic" benchmark segmentations.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def generate_block_segmentation(
        size, padding=2, lbl_pm=1, lbl_er=2, lbl_between_pm_er=4):
    """
    Generates a simple phantom segmentation shaped as a 3d cube, with a plane
    PM membrane, a plane ER membrane (parallel to each other and both 1 voxel
    thick, for simplicity) and a block-shaped volume between them.
    Also calculates the distance in pixels between PM and ER membranes.

    Args:
        size (int): size of the cubic segmentation in voxels
        padding (int, optional): padding of all the labels from the volume
            border in voxels (default 2)
        lbl_pm (int, optional): label of PM membrane (default 1)
        lbl_er (int, optional): label of ER membrane (default 2)
        lbl_between_pm_er (int, optional): label of inter-membrane space
            (default 4)

    Returns:
        segmentation (nd.array), dist_between_pm_er_pixels (int)
    """
    pm = padding
    er = size - padding - 1
    segmentation = np.zeros(dtype=int, shape=(size, size, size))
    segmentation[padding:-padding, pm, padding:-padding] = lbl_pm
    segmentation[padding:-padding, er, padding:-padding] = lbl_er
    segmentation[
        padding:-padding, pm+1:er, padding:-padding] = lbl_between_pm_er
    dist_between_pm_er_pixels = er - pm - 1  # (er-1)-(pm+1)+1
    return segmentation, dist_between_pm_er_pixels


class DistancesCalculationTestCase(unittest.TestCase):
    """
    Tests for run_calculate_distances.py, assuming that other used functions are
    correct.
    """

    def test_surfaces_normals_and_distances(self):
        """
        A tests for run_calculate_distances.py, using a simple phantom
        segmentation shaped as a 3d cube, with a plane PM membrane, a plane ER
        membrane (parallel to each other and both 1 voxel thick, for simplicity)
        and a block-shaped volume between them with known thickness.
        """
        # should exist:
        fold = '/fs/pool/pool-ruben/Maria/4Javier/distances_phantom_test/'
        # will be generated:
        segmentation_mrc_file = '{}phantom_block_segmentation.mrc'.format(fold)
        pm_surf_file = '{}phantom_plane.PM.vtp'.format(fold)
        pm_graph_file = '{}phantom_plane.PM.gt'.format(fold)
        er_surf_file = '{}phantom_plane.cER.vtp'.format(fold)
        er_graph_file = '{}phantom_plane.cER.gt'.format(fold)
        pm_normals_graph_file = '{}phantom_plane.PM.VVnormals.gt'.format(fold)
        er_dist_surf_file = '{}phantom_plane.cER.distancesFromPM.vtp'.format(
            fold)
        er_dist_graph_file = '{}phantom_plane.cER.distancesFromPM.gt'.format(
            fold)
        distances_outfile = '{}phantom_plane.cER.distancesFromPM.csv'.format(
            fold)
        # parameters:
        size = 30  # of segmentation_mrc_file
        pixel_size_nm = 1.368  # of segmentation_mrc_file
        radius_hit = 5  # for PM normals estimation by VV (have a perfect plane)
        maxdist_voxels = size
        maxdist_nm = maxdist_voxels * pixel_size_nm
        offset_voxels = 1  # because surfaces are generated 1/2 voxel off the
        # membrane segmentation boundary towards the inter-membrane space
        offset_nm = offset_voxels * pixel_size_nm

        print ('Generating a phantom segmentation')
        segmentation, true_dist_pixels = generate_block_segmentation(size)
        io.save_numpy(segmentation, segmentation_mrc_file)
        true_dist_nm = true_dist_pixels * pixel_size_nm
        print("True distance = {}".format(true_dist_nm))
        print ('Generating PM and cER graphs and cER surface files')
        generate_graphs_and_surface(
            segmentation_mrc_file, pixel_size_nm,
            pm_graph_file, er_surf_file, er_graph_file, pm_surf_file)
        print('Estimating normals for PM graph')
        pm_tg = TriangleGraph()
        pm_tg.graph = load_graph(pm_graph_file)
        normals_estimation(pm_tg, radius_hit)
        pm_tg.graph.save(pm_normals_graph_file)
        print('Calculating and saving distances between PM and cER')
        run_calculate_distances(
            pm_normals_graph_file, er_surf_file, er_graph_file,
            er_dist_surf_file, er_dist_graph_file, distances_outfile,
            maxdist_nm, offset_nm)
        print('Reading in and checking distances between PM and cER')
        df = pd.read_csv(distances_outfile, sep=';')
        distances = df['d1']
        for distance in distances:
            msg = "distance == {} != {}".format(distance, true_dist_nm)
            self.assertEqual(round(distance, 3), round(true_dist_nm, 3), msg)


if __name__ == '__main__':
    unittest.main()
