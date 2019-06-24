import pandas as pd
import numpy as np
from os.path import join
from os import chdir
import vtk

from errors_calculation import relative_error_scalar
from test_vector_voting import torus_curvatures_and_directions
from synthetic_surfaces import SaddleGenerator


def calculate_curvature_errors_sphere(
        radius, curvatures_csv_file, errors_csv_file):
    """
    Function calculating relative curvature errors for a sphere surface.

    Args:
        radius (int): radius of the sphere
        curvatures_csv_file (string): input curvatures file with columns labeled
            "kappa1", "kappa2", "mean_curvature" and "gauss_curvature".
        errors_csv_file (string): output error file with "RelErrors" appended to
            the aforementioned curvature column labels.

    Returns:
        None
    """
    # read in the curvatures from the CSV file:
    df = pd.read_csv(curvatures_csv_file, sep=';')
    mean_curv = df["mean_curvature"].values  # is always there
    kappa_1 = None
    kappa_2 = None
    gauss_curv = None
    if 'kappa1' in df.columns:
        kappa_1 = df["kappa1"].values
    if 'kappa2' in df.columns:
        kappa_2 = df["kappa2"].values
    if 'gauss_curvature' in df.columns:
        gauss_curv = df["gauss_curvature"].values

    # calculate the true curvatures:
    true_curvature = 1.0 / radius  # for kappa_1, kappa_2 and mean curvature
    true_gauss_curvature = true_curvature * true_curvature

    # Calculating relative errors of the curvatures:
    rel_mean_curv_errors = np.array(
        [relative_error_scalar(true_curvature, x) for x in mean_curv])
    df = pd.DataFrame()
    df['mean_curvatureRelErrors'] = rel_mean_curv_errors
    if kappa_1 is not None:
        rel_kappa_1_errors = np.array(
            [relative_error_scalar(true_curvature, x) for x in kappa_1])
        df['kappa1RelErrors'] = rel_kappa_1_errors
    if kappa_2 is not None:
        rel_kappa_2_errors = np.array(
            [relative_error_scalar(true_curvature, x) for x in kappa_2])
        df['kappa2RelErrors'] = rel_kappa_2_errors
    if gauss_curv is not None:
        rel_gauss_curv_errors = np.array(
            [relative_error_scalar(true_gauss_curvature, x) for x in gauss_curv])
        df['gauss_curvatureRelErrors'] = rel_gauss_curv_errors

    # Writing all the curvature errors into a second CSV file:
    df.to_csv(errors_csv_file, sep=';')


def calculate_curvature_errors_cylinder(
        radius, curvatures_csv_file, errors_csv_file):
    """
    Function calculating relative curvature errors for a cylinder surface.

    Args:
        radius (int): radius of the cylinder
        curvatures_csv_file (string): input curvatures file with columns labeled
            "kappa1", "kappa2", "mean_curvature" and "gauss_curvature".
        errors_csv_file (string): output error file with "RelErrors" appended to
            the aforementioned curvature column labels.

    Returns:
        None
    """
    # read in the curvatures from the CSV file:
    df = pd.read_csv(curvatures_csv_file, sep=';')
    mean_curv = df["mean_curvature"].values  # is always there
    kappa_1 = None
    kappa_2 = None
    gauss_curv = None
    if 'kappa1' in df.columns:
        kappa_1 = df["kappa1"].values
    if 'kappa2' in df.columns:
        kappa_2 = df["kappa2"].values
    if 'gauss_curvature' in df.columns:
        gauss_curv = df["gauss_curvature"].values

    # calculate the true curvatures:
    true_kappa_1 = 1.0 / radius
    true_kappa_2 = 0.0
    true_mean_curv = (true_kappa_1 + true_kappa_2) / 2.0
    true_gauss_curvature = true_kappa_1 * true_kappa_2

    # Calculating relative errors of the curvatures:
    rel_mean_curv_errors = np.array(
        [relative_error_scalar(true_mean_curv, x) for x in mean_curv])
    df = pd.DataFrame()
    df['mean_curvatureRelErrors'] = rel_mean_curv_errors
    if kappa_1 is not None:
        rel_kappa_1_errors = np.array(
            [relative_error_scalar(true_kappa_1, x) for x in kappa_1])
        df['kappa1RelErrors'] = rel_kappa_1_errors
    if kappa_2 is not None:
        rel_kappa_2_errors = np.array(
            [relative_error_scalar(true_kappa_2, x) for x in kappa_2])
        df['kappa2RelErrors'] = rel_kappa_2_errors
    if gauss_curv is not None:
        rel_gauss_curv_errors = np.array(
            [relative_error_scalar(true_gauss_curvature, x) for x in gauss_curv])
        df['gauss_curvatureRelErrors'] = rel_gauss_curv_errors

    # Writing all the curvature errors into a second CSV file:
    df.to_csv(errors_csv_file, sep=';')


def calculate_curvature_errors_torus(
        rr, csr, curvatures_csv_file, errors_csv_file, voxel=False):
    """
    Function calculating relative curvature errors for a torus surface.

    Args:
        rr (int): ring radius of the torus
        csr (int): cross-section radius of the torus
        curvatures_csv_file (string): input curvatures file with columns labeled
            "kappa1", "kappa2", "mean_curvature" and "gauss_curvature",
            including coordinates columns labeled 'x', 'y' and 'z'
        errors_csv_file (string): output error file with "RelErrors" appended to
            the aforementioned curvature column labels.
        voxel (boolean): if True, a voxel torus is generated (default False)

    Returns:
        None
    """
    # read in the curvatures from the CSV file:
    df = pd.read_csv(curvatures_csv_file, sep=';')
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    if voxel:
        # correct the coordinates to have (0,0,0) in the middle
        x = x - (rr + csr)
        y = y - (rr + csr)
        z = z - csr
        # Map the noisy coordinates to coordinates on smooth torus surface:
        # generate the smooth torus surface
        sgen = SaddleGenerator()
        smooth_torus = sgen.generate_parametric_torus(rr, csr, subdivisions=0)
        # make point locator on the smooth torus surface
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(smooth_torus)
        pointLocator.SetNumberOfPointsPerBucket(10)
        pointLocator.BuildLocator()
    mean_curv = df["mean_curvature"].values  # is always there
    kappa_1 = None
    kappa_2 = None
    gauss_curv = None
    if 'kappa1' in df.columns:
        kappa_1 = df["kappa1"].values
    if 'kappa2' in df.columns:
        kappa_2 = df["kappa2"].values
    if 'gauss_curvature' in df.columns:
        gauss_curv = df["gauss_curvature"].values

    # calculate the true curvatures:
    one_true_kappa_1 = 1.0 / csr  # constant for the whole torus surface
    true_kappa_1 = []
    true_kappa_2 = []
    for i in range(x.size):
        true_kappa_1.append(one_true_kappa_1)
        if voxel:
            # find the closest point on the smooth surface
            closest_point_id = pointLocator.FindClosestPoint([x[i], y[i], z[i]])
            closest_true_xyz = np.zeros(shape=3)
            smooth_torus.GetPoint(closest_point_id, closest_true_xyz)
            true_kappa_2_i, _, _ = torus_curvatures_and_directions(
                rr, csr, *closest_true_xyz)
        else:
            true_kappa_2_i, _, _ = torus_curvatures_and_directions(
                rr, csr, x[i], y[i], z[i])
        true_kappa_2.append(true_kappa_2_i)
    true_kappa_1 = np.array(true_kappa_1)
    true_kappa_2 = np.array(true_kappa_2)
    true_mean_curv = (true_kappa_1 + true_kappa_2) / 2.0
    true_gauss_curv = true_kappa_1 * true_kappa_2

    # Calculating relative errors of the principal curvatures:
    rel_mean_curv_errors = np.array(map(
        lambda x, y: relative_error_scalar(x, y), true_mean_curv, mean_curv))
    df = pd.DataFrame()
    df['true_mean_curv'] = true_mean_curv
    df['mean_curvatureRelErrors'] = rel_mean_curv_errors
    if kappa_1 is not None:
        rel_kappa_1_errors = np.array(
            [relative_error_scalar(one_true_kappa_1, x) for x in kappa_1])
        df['kappa1RelErrors'] = rel_kappa_1_errors
    if kappa_2 is not None:
        rel_kappa_2_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y), true_kappa_2, kappa_2))
        df['true_kappa2'] = true_kappa_2
        df['kappa2RelErrors'] = rel_kappa_2_errors
    if gauss_curv is not None:
        rel_gauss_curv_errors = np.array(map(
            lambda x, y: relative_error_scalar(x, y), true_gauss_curv, gauss_curv))
        df['true_gauss_curv'] = true_gauss_curv
        df['gauss_curvatureRelErrors'] = rel_gauss_curv_errors

    # Writing all the curvature errors into a second CSV file:
    df.to_csv(errors_csv_file, sep=';')


if __name__ == '__main__':
    test_mindboggle_output_fold = (
        "/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/"
        "comparison_to_mindboggle/test_mindboggle_output/")
    surface_bases = ['torus_rr25_csr10.surface.', 'smooth_sphere_r10.surface.',
                 'noisy_sphere_r10.surface.', 'cylinder_r10_h25.surface.']
    subfolds = ['torus', 'smooth_sphere', 'noisy_sphere', 'cylinder']
    m = 0
    ns = range(1, 10, 2)  # range(2, 21, 2)
    for surface_base, subfold in zip(surface_bases, subfolds):
        fold = join(test_mindboggle_output_fold, subfold)
        chdir(fold)
        for n in ns:
            out_base = "{}mindboggle_m{}_n{}".format(surface_base, m, n)
            curvatures_csv_file = join(
                fold, "{}_curvatures.csv".format(out_base))
            errors_csv_file = join(
                fold, "{}_curvature_errors.csv".format(out_base))
            if out_base.startswith('torus'):
                calculate_curvature_errors_torus(
                    25, 10, curvatures_csv_file, errors_csv_file)
            elif 'sphere' in out_base:
                calculate_curvature_errors_sphere(
                    10, curvatures_csv_file, errors_csv_file)
            else:  # cylinder
                calculate_curvature_errors_cylinder(
                    10, curvatures_csv_file, errors_csv_file)
