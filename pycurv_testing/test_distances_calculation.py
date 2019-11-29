import numpy as np
import pandas as pd
import os
import shutil

from pycurv import pycurv_io as io
from pycurv_scripts import distances_and_thicknesses_calculation

"""
A function generating a synthetic segmentation and an integration test for
testing the functions calculating distances and thicknesses between membrane
surfaces.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def generate_pm_er_lumen_segmentation(
        dist, thick, padding=2, lbl_pm=1, lbl_er=2, lbl_between_pm_er=4,
        lbl_er_lumen=3):
    """
    Generates a simple phantom segmentation shaped as a 3d cube, with a plane
    PM membrane, a hollow block-shaped ER membrane (parallel to each other and
    both 1 voxel thick, for simplicity), a block-shaped volume between PM and
    ER and a block-shaped volume inside the ER (ER lumen).
    Also calculates the distance in pixels between PM and ER membranes.

    Args:
        dist (int): distance from PM to ER (excluding the membranes) in voxels
        thick (int): thickness of ER lumen (excluding the membranes) in voxels
        padding (int, optional): padding of all the labels from the volume
            border in voxels (default 2)
        lbl_pm (int, optional): label of PM membrane (default 1)
        lbl_er (int, optional): label of ER membrane (default 2)
        lbl_between_pm_er (int, optional): label of inter-membrane space
            (default 4)
        lbl_er_lumen (int, optional): label of inter-membrane space (default 3)
    Returns:
        segmentation (nd.array)
    """
    size = 2 * padding + 3 + dist + thick
    segmentation = np.zeros(dtype=int, shape=(size, size, size))  # z, y, x
    pm = padding
    segmentation[padding:-padding, pm, padding:-padding] = lbl_pm
    er_top = pm + dist + 1
    segmentation[
        padding:-padding, pm+1:er_top, padding:-padding] = lbl_between_pm_er
    er_bottom = er_top + thick + 1
    segmentation[
        padding:-padding, er_top:er_bottom+1, padding:-padding] = lbl_er
    segmentation[
        padding+1:-padding-1, er_top+1:er_bottom,
        padding+1:-padding-1] = lbl_er_lumen

    return segmentation


def test_distances_and_thicknesses_calculation():
    """
    Tests for run_calculate_distances.py, assuming that other used functions are
    correct.
    """
    # will be generated:
    fold = './test_distances_calculation_output/'
    if os.path.isdir(fold):
        shutil.rmtree(fold)
    os.mkdir(fold)  # start always with an empty directory
    segmentation_file = 'phantom_segmentation.mrc'
    base_filename = "phantom"
    distances_outfile = '{}.cER.distancesFromPM.csv'.format(base_filename)
    thicknesses_outfile = '{}.innercER.thicknesses.csv'.format(
        base_filename)

    # parameters:
    true_dist_pixels = 5
    true_thick_pixels = 10
    pixel_size_nm = 1.368
    true_dist_nm = true_dist_pixels * pixel_size_nm
    true_thick_nm = true_thick_pixels * pixel_size_nm
    radius_hit = 5  # for PM normals estimation by VV (have a perfect plane)
    maxdist_voxels = true_dist_pixels * 2
    maxdist_nm = maxdist_voxels * pixel_size_nm
    maxthick_voxels = true_thick_pixels * 2
    maxthick_nm = maxthick_voxels * pixel_size_nm
    offset_voxels = 1  # because surfaces are generated 1/2 voxel off the
    # membrane segmentation boundary towards the inter-membrane space

    print ('Generating a phantom segmentation')
    print("True distance = {}".format(true_dist_pixels))
    print("True thickness = {}".format(true_thick_pixels))
    segmentation = generate_pm_er_lumen_segmentation(
        true_dist_pixels, true_thick_pixels)
    io.save_numpy(segmentation, fold+segmentation_file)

    print("Applying the script distances_and_thicknesses_calculation")
    distances_and_thicknesses_calculation(
        fold, segmentation_file, base_filename,
        pixel_size=pixel_size_nm, radius_hit=radius_hit,
        maxdist=maxdist_nm, maxthick=maxthick_nm,
        offset_voxels=offset_voxels, smooth=False)

    print('Reading in and checking distances between PM and cER')
    df = pd.read_csv(fold+distances_outfile, sep=';', index_col=0)
    distances = df['d1']
    for distance in distances:
        assert round(distance, 3) == round(true_dist_nm, 3)

    print('Reading in and checking cER thicknesses')
    df = pd.read_csv(fold+thicknesses_outfile, sep=';', index_col=0)
    thicknesses = df['d2']
    for thickness in thicknesses:
        assert round(thickness, 3) == round(true_thick_nm, 3)
