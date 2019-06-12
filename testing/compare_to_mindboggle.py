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
    kappa_1 = df["kappa1"].values
    kappa_2 = df["kappa2"].values
    mean_curv = df["mean_curvature"].values
    gauss_curv = df["gauss_curvature"].values

    # calculate the true curvatures:
    true_curvature = 1.0 / radius  # for kappa_1, kappa_2 and mean curvature
    true_gauss_curvature = true_curvature * true_curvature

    # Calculating relative errors of the principal curvatures:
    rel_kappa_1_errors = np.array(
        [relative_error_scalar(true_curvature, x) for x in kappa_1])
    rel_kappa_2_errors = np.array(
        [relative_error_scalar(true_curvature, x) for x in kappa_2])
    rel_mean_curv_errors = np.array(
        [relative_error_scalar(true_curvature, x) for x in mean_curv])
    rel_gauss_curv_errors = np.array(
        [relative_error_scalar(true_gauss_curvature, x) for x in gauss_curv])

    # Writing all the curvature errors into a second CSV file:
    df = pd.DataFrame()
    df['kappa1RelErrors'] = rel_kappa_1_errors
    df['kappa2RelErrors'] = rel_kappa_2_errors
    df['mean_curvatureRelErrors'] = rel_mean_curv_errors
    df['gauss_curvatureRelErrors'] = rel_gauss_curv_errors
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
    kappa_1 = df["kappa1"].values
    kappa_2 = df["kappa2"].values
    mean_curv = df["mean_curvature"].values
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
    rel_kappa_1_errors = np.array(
        [relative_error_scalar(one_true_kappa_1, x) for x in kappa_1])
    rel_kappa_2_errors = np.array(map(
        lambda x, y: relative_error_scalar(x, y), true_kappa_2, kappa_2))
    rel_mean_curv_errors = np.array(map(
        lambda x, y: relative_error_scalar(x, y), true_mean_curv, mean_curv))
    rel_gauss_curv_errors = np.array(map(
        lambda x, y: relative_error_scalar(x, y), true_gauss_curv, gauss_curv))

    # Writing all the curvature errors into a second CSV file:
    df = pd.DataFrame()
    df['kappa1RelErrors'] = rel_kappa_1_errors
    df['kappa2RelErrors'] = rel_kappa_2_errors
    df['mean_curvatureRelErrors'] = rel_mean_curv_errors
    df['gauss_curvatureRelErrors'] = rel_gauss_curv_errors
    df.to_csv(errors_csv_file, sep=';')


if __name__ == '__main__':
    n = 9
    fold = (
        "/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/"
        "comparison_to_mindboggle/test_surfaces_mindboggle_output_n{}".format(
            n))

    # sphere_radius = 10
    # sphere_curvatures_csv_file = join(
    #     fold, "noisy_sphere_r{}.surface.mindboggle_n{}_curvatures.csv".format(
    #         sphere_radius, n))
    # sphere_errors_csv_file = join(
    #     fold,
    #     "noisy_sphere_r{}.surface.mindboggle_n{}_curvature_errors.csv".format(
    #         sphere_radius, n))
    # calculate_curvature_errors_sphere(
    #     sphere_radius, sphere_curvatures_csv_file, sphere_errors_csv_file)

    torus_rr = 25
    torus_csr = 10
    torus_curvatures_csv_file = join(
        fold,
        "torus_rr{}_csr{}.surface.mindboggle_n{}_curvatures.csv".format(
            torus_rr, torus_csr, n))
    torus_errors_csv_file = join(
        fold,
        "torus_rr{}_csr{}.surface.mindboggle_n{}_curvature_errors.csv".format(
            torus_rr, torus_csr, n))
    calculate_curvature_errors_torus(
        torus_rr, torus_csr, torus_curvatures_csv_file, torus_errors_csv_file)
