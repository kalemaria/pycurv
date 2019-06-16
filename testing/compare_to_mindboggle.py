import pandas as pd
import numpy as np
from os.path import join

from errors_calculation import relative_error_scalar
from test_vector_voting import torus_curvatures_and_directions


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


def calculate_curvature_errors_torus(
        rr, csr, curvatures_csv_file, errors_csv_file):
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

    Returns:
        None
    """
    # read in the curvatures from the CSV file:
    df = pd.read_csv(curvatures_csv_file, sep=';')
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
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
    base_fold = "/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/" \
                "comparison_to_mindboggle/"
    for m in [0, 2]:
        for n in [0.75, 2, 9, 10]:
            fold = "{}test_surfaces_mindboggle_output_n{}".format(base_fold, n)

            sphere_radius = 10
            sphere_base = "noisy_sphere_r{}.surface.mindboggle_m{}_n{}".format(
                sphere_radius, m, n)
            sphere_curvatures_csv_file = join(
                fold, "{}_curvatures.csv".format(sphere_base))
            sphere_errors_csv_file = join(
                fold, "{}_curvature_errors.csv".format(sphere_base))
            calculate_curvature_errors_sphere(
                sphere_radius, sphere_curvatures_csv_file,
                sphere_errors_csv_file)

            torus_rr = 25
            torus_csr = 10
            torus_base = "torus_rr{}_csr{}.surface.mindboggle_m{}_n{}".format(
                torus_rr, torus_csr, m, n)
            torus_curvatures_csv_file = join(
                fold, "{}_curvatures.csv".format(torus_base))
            torus_errors_csv_file = join(
                fold, "{}_curvature_errors.csv".format(torus_base))
            calculate_curvature_errors_torus(
                torus_rr, torus_csr, torus_curvatures_csv_file,
                torus_errors_csv_file)
