import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from pysurf_compact import pexceptions


def plot_hist(value_list, num_bins, title, xlabel="Value", ylabel="Counts",
              value_range=None, outfile=None):
    """
    Plots a histogram of the values with the given number of bins and plot
    title.

    Args:
        value_list: a list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Counts")
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    if value_range is None:
        plt.hist(value_list, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        plt.hist(value_list, bins=num_bins, range=value_range)
    else:
        error_msg = "Range has to be a tuple of two numbers (min, max)."
        raise pexceptions.PySegInputError(expr='plot_hist', msg=error_msg)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)


def plot_line_hist(value_list, num_bins, title, xlabel="Value", ylabel="Counts",
                   value_range=None, label=None, outfile=None, max_val=None):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        value_list: a list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Counts")
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        label (str, optional): legend label for the value list (default None)
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    if max_val is not None:
        value_list = [max_val if val > max_val else val for val in value_list]
    if value_range is None:
        counts, bin_edges = np.histogram(value_list, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        counts, bin_edges = np.histogram(value_list, bins=num_bins,
                                         range=value_range)
    else:
        error_msg = "Range has to be a tuple of two numbers (min, max)."
        raise pexceptions.PySegInputError(expr='plot_hist', msg=error_msg)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bincenters, counts, ls='-', marker='^', c="b", label=label,
             linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)


def add_line_hist(values, num_bins, value_range=None, max_val=None,
                  label=None, ls='-', marker='^', c='b', freq=False,
                  cumulative=False):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
        num_bins (int): number of bins for the histogram
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value
        label (str, optional): legend label for the value list (default None)
        ls (str, optional): line style (default '-')
        marker (str, optional): plotting character (default '^')
        c (str, optional): color (default 'b' for blue)
        freq (boolean, optional): if True (default False), frequencies instead
            of counts will be plotted
        cumulative (boolean, optional): if True (default False), cumulative
            counts or frequencies will be plotted

    Returns:
        None
    """
    if max_val is not None:
        values = [max_val if val > max_val else val for val in values]
    if value_range is None:
        counts, bin_edges = np.histogram(values, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        counts, bin_edges = np.histogram(values, bins=num_bins,
                                         range=value_range)
    else:
        error_msg = "Range has to be a tuple of two numbers (min, max)."
        raise pexceptions.PySegInputError(expr='plot_hist', msg=error_msg)
    # bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if cumulative is True:  # TODO activate again!
        # counts = np.array([np.sum(counts[0:i+1]) for i in range(len(counts))])
        counts = np.cumsum(counts)
    if freq is True:
        counts = counts / float(len(values))  # normalized to  max 1
    plt.plot(bin_edges[:-1], counts, ls=ls, marker=marker, c=c, label=label,
             linewidth=2)


def plot_errors(
        labels, line_styles, markers, colors,  # all lists
        title, xlabel, ylabel,
        error_arrays=None, error_files=None,  # all lists
        num_bins=20, value_range=None, max_val=None, freq=False, outfile=None,
        cumulative=False):
    # TODO docstring from the docstrings of the two previous functions
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()

    if error_files is not None:
        for i, error_file in enumerate(error_files):
            # Reading in the error values from files:
            if not os.path.exists(error_file):
                print ("File {} not found!".format(error_file))
                exit(0)
            errors = np.loadtxt(error_file)
            add_line_hist(
                errors, num_bins, value_range=value_range, max_val=max_val,
                label=labels[i], ls=line_styles[i], marker=markers[i],
                c=colors[i], freq=freq, cumulative=cumulative)
    elif error_arrays is not None:
        for i, error_array in enumerate(error_arrays):
            add_line_hist(
                error_array, num_bins, value_range=value_range, max_val=max_val,
                label=labels[i], ls=line_styles[i], marker=markers[i],
                c=colors[i], freq=freq, cumulative=cumulative)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.grid(True)
    plt.margins(0.05)
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)
    print ("The plot was saved as {}".format(outfile))


if __name__ == "__main__":
    # plane normals
    # n = 10  # noise in %
    # fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #         "plane/res10_noise{}/files4plotting/".format(n))
    # plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #              "plane/res10_noise{}/plots/".format(n))
    # VCTV_rh8_normal_errors = pd.read_csv(fold+"plane_half_size10.VCTV_rh8.csv",
    #                                      sep=';')["normalErrors"].tolist()
    # VCTV_rh4_normal_errors = pd.read_csv(fold+"plane_half_size10.VCTV_rh4.csv",
    #                                      sep=';')["normalErrors"].tolist()
    # VTK_normal_errors = pd.read_csv(fold+"plane_half_size10.VTK.csv", sep=';'
    #                                 )["normalErrors"].tolist()
    # plot_errors(
    #     error_arrays=[VCTV_rh8_normal_errors,
    #                   VCTV_rh4_normal_errors,
    #                   VTK_normal_errors],
    #     labels=["VV rh=8", "VV rh=4", "VTK"],
    #     line_styles=['-', '--', ':'], markers=['^', 'v', 's'],
    #     colors=['b', 'c', 'r'],
    #     title="Plane ({}% noise)".format(n),
    #     xlabel="Normal orientation error", ylabel="Frequency",
    #     outfile="{}plane_res10_noise{}.VV_vs_VTK.normal_errors_bin20_freq.png"
    #             .format(plot_fold, n),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True
    # )

    # # cylinder T_2 and kappa_1 errors
    # n = 0  # noise in %
    # fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #         "cylinder/noise0/files4plotting/".format(n))
    # plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #              "cylinder/noise0/plots/".format(n))
    # df = pd.read_csv(fold + "cylinder_r10_h25.VCTV_rh8.csv", sep=';')
    # VCTV_T_2_errors = df["T2Errors"].tolist()
    # VCTV_kappa_1_errors = df["kappa1RelErrors"].tolist()
    #
    # VVCF_kappa_1_errors = pd.read_csv(fold + "cylinder_r10_h25.VVCF_rh8.csv",
    #                                   sep=';')["kappa1RelErrors"].tolist()
    # df = pd.read_csv(fold + "cylinder_r10_h25.VV_rh8.csv", sep=';')
    # VV_T_2_errors = df["T2Errors"].tolist()
    # VV_kappa_1_errors = df["kappa1RelErrors"].tolist()
    #
    # VTK_kappa_1_errors = pd.read_csv(fold + "cylinder_r10_h25.VTK.csv",
    #                                  sep=';')["kappa1RelErrors"].tolist()
    # plot_errors(
    #     error_arrays=[VCTV_T_2_errors, VV_T_2_errors],
    #     labels=["VCTV rh=8", "VV rh=8"],
    #     line_styles=['-', '--'], markers=['^', 'v'],
    #     colors=['b', 'c'],
    #     title="Cylinder ({}% noise)".format(n),
    #     xlabel="Minimal principal direction error",
    #     ylabel="Frequency",
    #     outfile="{}cylinder_r10_noise{}.VV_VCTV_rh8.T_2_errors_bins20_freq.png"
    #             .format(plot_fold, n),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True
    # )
    # plot_errors(
    #     error_arrays=[VCTV_kappa_1_errors, VVCF_kappa_1_errors,
    #                   VV_kappa_1_errors, VTK_kappa_1_errors],
    #     labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
    #     line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
    #     colors=['b', 'g', 'c', 'r'],
    #     title="Cylinder ({}% noise)".format(n),
    #     xlabel="Maximal principal curvature relative error",
    #     ylabel="Frequency",
    #     outfile=("{}cylinder_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
    #              "kappa_1_errors_bins20_freq.png".format(plot_fold, n)),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True
    # )

    # # inverse cylinder T_1 and kappa_2 errors
    # n = 0  # noise in %
    # fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #         "cylinder/noise0/files4plotting/".format(n))
    # plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #              "cylinder/noise0/plots/".format(n))
    # df = pd.read_csv(fold + "inverse_cylinder_r10_h25.VCTV_rh8.csv", sep=';')
    # VCTV_T_1_errors = df["T1Errors"].tolist()
    # VCTV_kappa_2_errors = df["kappa2RelErrors"].tolist()
    #
    # VVCF_kappa_2_errors = pd.read_csv(
    #     fold + "inverse_cylinder_r10_h25.VVCF_rh8.csv", sep=';'
    # )["kappa2RelErrors"].tolist()
    #
    # df = pd.read_csv(fold + "inverse_cylinder_r10_h25.VV_rh8.csv", sep=';')
    # VV_T_1_errors = df["T1Errors"].tolist()
    # VV_kappa_2_errors = df["kappa2RelErrors"].tolist()
    #
    # VTK_kappa_2_errors = pd.read_csv(fold + "inverse_cylinder_r10_h25.VTK.csv",
    #                                  sep=';')["kappa2RelErrors"].tolist()
    # plot_errors(
    #     error_arrays=[
    #         VCTV_T_1_errors,
    #         VV_T_1_errors],
    #     labels=["VCTV rh=8", "VV rh=8"],
    #     line_styles=['-', '--'], markers=['^', 'v'],
    #     colors=['b', 'c'],
    #     title="Inverse cylinder ({}% noise)".format(n),
    #     xlabel="Maximal principal direction error",
    #     ylabel="Frequency",
    #     outfile="{}inverse_cylinder_r10_noise{}.VV_VCTV_rh8.T_1_errors"
    #             "_bins20_freq.png".format(plot_fold, n),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True
    # )
    # plot_errors(
    #     error_arrays=[
    #         VCTV_kappa_2_errors,
    #         VVCF_kappa_2_errors,
    #         VV_kappa_2_errors,
    #         VTK_kappa_2_errors],
    #     labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
    #     line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
    #     colors=['b', 'g', 'c', 'r'],
    #     title="Inverse cylinder ({}% noise)".format(n),
    #     xlabel="Minimal principal curvature relative error",
    #     ylabel="Number of triangles",
    #     outfile=("{}inverse_cylinder_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
    #              "kappa_2_errors_bins20_freq.png".format(plot_fold, n)),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True
    # )

    # # sphere kappa_1 and kappa_2 values using different methods
    n = 0  # noise in %
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "sphere/ico1280_noise{}/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "sphere/ico1280_noise{}/plots/".format(n))
    # method = 'VVCF'  # done: 'VCTV', 'VV'
    # kappa_1_rh_n1 = pd.read_csv(fold + "sphere_r10.{}_rh6.5.csv".format(method),
    #                             sep=';')["kappa1"].tolist()
    # kappa_1_rh_n2 = pd.read_csv(fold + "sphere_r10.{}_rh7.csv".format(method),
    #                             sep=';')["kappa1"].tolist()
    # kappa_1_rh_n3 = pd.read_csv(fold + "sphere_r10.{}_rh7.5.csv".format(method),
    #                             sep=';')["kappa1"].tolist()
    # kappa_1_rh_n4 = pd.read_csv(fold + "sphere_r10.{}_rh8.csv".format(method),
    #                             sep=';')["kappa1"].tolist()
    # kappa_1_rh_n5 = pd.read_csv(fold + "sphere_r10.{}_rh8.5.csv".format(method),
    #                             sep=';')["kappa1"].tolist()
    # kappa_2_rh_n1 = pd.read_csv(fold + "sphere_r10.{}_rh6.5.csv".format(method),
    #                             sep=';')["kappa2"].tolist()
    # kappa_2_rh_n2 = pd.read_csv(fold + "sphere_r10.{}_rh7.csv".format(method),
    #                             sep=';')["kappa2"].tolist()
    # kappa_2_rh_n3 = pd.read_csv(fold + "sphere_r10.{}_rh7.5.csv".format(method),
    #                             sep=';')["kappa2"].tolist()
    # kappa_2_rh_n4 = pd.read_csv(fold + "sphere_r10.{}_rh8.csv".format(method),
    #                             sep=';')["kappa2"].tolist()
    # kappa_2_rh_n5 = pd.read_csv(fold + "sphere_r10.{}_rh8.5.csv".format(method),
    #                             sep=';')["kappa2"].tolist()
    # plot_errors(
    #     error_arrays=[kappa_1_rh_n1 + kappa_2_rh_n1,
    #                   kappa_1_rh_n2 + kappa_2_rh_n2,
    #                   kappa_1_rh_n3 + kappa_2_rh_n3,
    #                   kappa_1_rh_n4 + kappa_2_rh_n4,
    #                   kappa_1_rh_n5 + kappa_2_rh_n5],
    #     labels=["RadiusHit=6.5", "RadiusHit=7", "RadiusHit=7.5", "RadiusHit=8",
    #             "RadiusHit=8.5"],
    #     line_styles=['-.', '-.', '--', '-', ':'],
    #     markers=['x', 'v', '^', 's', 'o'],
    #     colors=['b', 'c', 'g', 'y', 'r'],
    #     title="{} on sphere (icosahedron 1280, {}% noise)".format(method, n),
    #     xlabel="Estimated principal curvatures",
    #     ylabel="Counts",
    #     outfile=("{}icosphere_r10_noise{}.{}_rh6.5-8.5."
    #              "kappa_1_and_2.png".format(plot_fold, n, method)),
    #     num_bins=5, value_range=None, max_val=None, freq=False
    # )
    # sphere kappa_1 and kappa_2 errors
    VCTV_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VCTV_rh8.csv", sep=';'
                                      )["kappa1RelErrors"].tolist()
    VCTV_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VCTV_rh8.csv", sep=';'
                                      )["kappa2RelErrors"].tolist()
    VVCF_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VVCF_rh8.csv", sep=';'
                                      )["kappa1RelErrors"].tolist()
    VVCF_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VVCF_rh8.csv", sep=';'
                                      )["kappa2RelErrors"].tolist()
    VV_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VV_rh8.csv", sep=';'
                                    )["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VV_rh8.csv", sep=';'
                                    )["kappa2RelErrors"].tolist()
    VTK_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VTK.csv", sep=';'
                                     )["kappa1RelErrors"].tolist()
    VTK_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VTK.csv", sep=';'
                                     )["kappa2RelErrors"].tolist()
    plot_errors(
        error_arrays=[VCTV_kappa_1_errors + VCTV_kappa_2_errors,
                      VVCF_kappa_1_errors + VVCF_kappa_2_errors,
                      VV_kappa_1_errors + VV_kappa_2_errors,
                      VTK_kappa_1_errors + VTK_kappa_2_errors],
        labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
        line_styles=['-', '-.', '--', ':'],
        markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Sphere (icosahedron 1280, {}% noise)".format(n),
        xlabel="Principal curvatures relative error",
        ylabel="Frequency",
        outfile=("{}icosphere_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
                 "kappa_1_and_2_errors_20bins_freq.png".format(plot_fold, n)),
        num_bins=20, value_range=(0, 1), max_val=1, freq=True
    )
    plot_errors(  # cumulative
        error_arrays=[VCTV_kappa_1_errors + VCTV_kappa_2_errors,
                      VVCF_kappa_1_errors + VVCF_kappa_2_errors,
                      VV_kappa_1_errors + VV_kappa_2_errors,
                      VTK_kappa_1_errors + VTK_kappa_2_errors],
        labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
        line_styles=['-', '-.', '--', ':'],
        markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Sphere (icosahedron 1280, {}% noise)".format(n),
        xlabel="Principal curvatures relative error",
        ylabel="Cumulative frequency",
        outfile=("{}icosphere_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
                 "kappa_1_and_2_errors_20bins_cum_freq.png".format(
                  plot_fold, n)),
        num_bins=20, value_range=(0, 1), max_val=1, freq=True, cumulative=True
    )

    # # inverse sphere kappa_1 errors
    # n = 0  # noise in %
    # fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #         "sphere/ico1280_noise{}/files4plotting/".format(n))
    # plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #              "sphere/ico1280_noise{}/plots/".format(n))
    # VCTV_kappa_1_errors = pd.read_csv(fold + "inverse_sphere_r10.VCTV_rh8.csv",
    #                                   sep=';')["kappa1RelErrors"].tolist()
    #
    # VVCF_kappa_1_errors = pd.read_csv(fold + "inverse_sphere_r10.VVCF_rh8.csv",
    #                                   sep=';')["kappa1RelErrors"].tolist()
    #
    # VV_kappa_1_errors = pd.read_csv(fold + "inverse_sphere_r10.VV_rh8.csv",
    #                                 sep=';')["kappa1RelErrors"].tolist()
    #
    # VTK_kappa_1_errors = pd.read_csv(fold + "inverse_sphere_r10.VTK.csv",
    #                                  sep=';')["kappa1RelErrors"].tolist()
    # plot_errors(
    #     error_arrays=[VCTV_kappa_1_errors, VVCF_kappa_1_errors,
    #                   VV_kappa_1_errors, VTK_kappa_1_errors],
    #     labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
    #     line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
    #     colors=['b', 'g', 'c', 'r'],
    #     title="Inverse sphere (icosahedron 1280, {}% noise)".
    #           format(n),
    #     xlabel="Maximal principal curvature relative error",
    #     ylabel="Frequency",
    #     outfile=("{}inverse_icosphere_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
    #              "kappa_1_errors_20bins_freq.png".format(plot_fold, n)),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True
    # )
