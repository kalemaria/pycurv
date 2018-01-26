import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from pysurf_compact import pexceptions


def plot_hist(values, num_bins, title, xlabel="Value", ylabel="Counts",
              value_range=None, outfile=None):
    """
    Plots a histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
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
        plt.hist(values, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        plt.hist(values, bins=num_bins, range=value_range)
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


def plot_line_hist(values, num_bins, title, xlabel="Value", ylabel="Counts",
                   value_range=None, label=None, max_val=None, outfile=None):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Counts")
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        label (str, optional): legend label for the value list (default None)
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
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
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if cumulative is True:
        # counts = np.array([np.sum(counts[0:i+1]) for i in range(len(counts))])
        counts = np.cumsum(counts)
    if freq is True:
        counts = counts / float(len(values))  # normalized to  max 1
    plt.plot(bincenters, counts, ls=ls, marker=marker, c=c, label=label,
             linewidth=2)


def plot_composite_line_hist(
        labels, line_styles, markers, colors,
        title, xlabel, ylabel,
        data_arrays=None, data_files=None,
        num_bins=20, value_range=None, max_val=None, freq=False,
        cumulative=False, outfile=None):
    """
    Plots several data sets as line histograms in one plot.
    Args:
        labels: list of legend labels (str) for the data sets
        line_styles: list of line styles (str)
        markers: list of plotting characters (str)
        colors: list of colors (str)
        title (str): title of the plot
        xlabel (str): X axis label
        ylabel (str): Y axis label
        data_arrays: list of data arrays
        data_files: list of data file names (str)
        num_bins (int, optional): number of bins for the histogram (default 20)
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value
        freq (boolean, optional): if True (default False), frequencies instead
            of counts will be plotted
        cumulative (boolean, optional): if True (default False), cumulative
            counts or frequencies will be plotted
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

    Returns:
        None
    Note:
        either data_arrays or data_files has to be given

    """
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()

    if data_files is not None:
        for i, data_file in enumerate(data_files):
            # Reading in the error values from files:
            if not os.path.exists(data_file):
                print ("File {} not found!".format(data_file))
                exit(0)
            errors = np.loadtxt(data_file)
            add_line_hist(
                errors, num_bins, value_range=value_range, max_val=max_val,
                label=labels[i], ls=line_styles[i], marker=markers[i],
                c=colors[i], freq=freq, cumulative=cumulative)
    elif data_arrays is not None:
        for i, data_array in enumerate(data_arrays):
            add_line_hist(
                data_array, num_bins, value_range=value_range, max_val=max_val,
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
    # plot_composite_line_hist(
    #     data_arrays=[VCTV_rh8_normal_errors,
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

    # # cylinder kappa_1 values for different RadiusHit and methods
    # n = 0  # noise in %
    # fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #         "cylinder/noise0/files4plotting/".format(n))
    # plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #              "cylinder/noise0/plots/".format(n))
    # for method in ['VV', 'VVCF', 'VCTV']:  # TODO for VVCF_50points
    #     kappa_1_rh_n1 = pd.read_csv(fold + "cylinder_r10_h25.{}_rh5.csv".format(
    #         method), sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n2 = pd.read_csv(fold + "cylinder_r10_h25.{}_rh6.csv".format(
    #         method), sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n3 = pd.read_csv(fold + "cylinder_r10_h25.{}_rh7.csv".format(
    #         method), sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n4 = pd.read_csv(fold + "cylinder_r10_h25.{}_rh8.csv".format(
    #         method), sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n5 = pd.read_csv(fold + "cylinder_r10_h25.{}_rh9.csv".format(
    #         method), sep=';')["kappa1"].tolist()
    #     plot_composite_line_hist(
    #         data_arrays=[kappa_1_rh_n1, kappa_1_rh_n2, kappa_1_rh_n3,
    #                       kappa_1_rh_n4, kappa_1_rh_n5],
    #         labels=["RadiusHit=5", "RadiusHit=6", "RadiusHit=7",
    #                 "RadiusHit=8", "RadiusHit=9"],
    #         line_styles=['-.', '-.', '--', '-', ':'],
    #         markers=['x', 'v', '^', 's', 'o'],
    #         colors=['b', 'c', 'g', 'y', 'r'],
    #         title="{} on cylinder ({}% noise)".format(method, n),
    #         xlabel="Estimated maximal principal curvature",
    #         ylabel="Counts",
    #         outfile=("{}cylinder_r10_h25_noise{}.{}_rh5-9.kappa_1.png".format(
    #             plot_fold, n, method)),
    #         num_bins=5, value_range=None, max_val=None, freq=False
    #     )
    # # cylinder T_2 and kappa_1 errors
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
    # plot_composite_line_hist(
    #     data_arrays=[VCTV_T_2_errors, VV_T_2_errors],
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
    # plot_composite_line_hist(  # cumulative
    #     data_arrays=[VCTV_T_2_errors, VV_T_2_errors],
    #     labels=["VCTV rh=8", "VV rh=8"],
    #     line_styles=['-', '--'], markers=['^', 'v'],
    #     colors=['b', 'c'],
    #     title="Cylinder ({}% noise)".format(n),
    #     xlabel="Minimal principal direction error",
    #     ylabel="Cumulative frequency",
    #     outfile="{}cylinder_r10_noise{}.VV_VCTV_rh8.T_2_errors_bins20_cum_freq."
    #             "png".format(plot_fold, n),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True, cumulative=True
    # )
    # plot_composite_line_hist(
    #     data_arrays=[VCTV_kappa_1_errors, VVCF_kappa_1_errors,
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
    # plot_composite_line_hist(  # cumulative
    #     data_arrays=[VCTV_kappa_1_errors, VVCF_kappa_1_errors,
    #                   VV_kappa_1_errors, VTK_kappa_1_errors],
    #     labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
    #     line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
    #     colors=['b', 'g', 'c', 'r'],
    #     title="Cylinder ({}% noise)".format(n),
    #     xlabel="Maximal principal curvature relative error",
    #     ylabel="Cumulative frequency",
    #     outfile=("{}cylinder_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
    #              "kappa_1_errors_bins20_cum_freq.png".format(plot_fold, n)),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True, cumulative=True
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
    # plot_composite_line_hist(
    #     data_arrays=[
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
    # plot_composite_line_hist(
    #     data_arrays=[
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

    # # sphere kappa_1 and kappa_2 values for different RadiusHit and methods
    # n = 0  # noise in %
    # fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #         "sphere/ico1280_noise{}/files4plotting/".format(n))
    # plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
    #              "sphere/ico1280_noise{}/plots/".format(n))
    # for method in ['VV', 'VCTV', 'VVCF_50points']:
    #     kappa_1_rh_n1 = pd.read_csv(fold + "sphere_r10.{}_rh5.csv".format(method),
    #                                 sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n2 = pd.read_csv(fold + "sphere_r10.{}_rh6.csv".format(method),
    #                                 sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n3 = pd.read_csv(fold + "sphere_r10.{}_rh7.csv".format(method),
    #                                 sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n4 = pd.read_csv(fold + "sphere_r10.{}_rh8.csv".format(method),
    #                                 sep=';')["kappa1"].tolist()
    #     kappa_1_rh_n5 = pd.read_csv(fold + "sphere_r10.{}_rh9.csv".format(method),
    #                                 sep=';')["kappa1"].tolist()
    #     kappa_2_rh_n1 = pd.read_csv(fold + "sphere_r10.{}_rh5.csv".format(method),
    #                                 sep=';')["kappa2"].tolist()
    #     kappa_2_rh_n2 = pd.read_csv(fold + "sphere_r10.{}_rh6.csv".format(method),
    #                                 sep=';')["kappa2"].tolist()
    #     kappa_2_rh_n3 = pd.read_csv(fold + "sphere_r10.{}_rh7.csv".format(method),
    #                                 sep=';')["kappa2"].tolist()
    #     kappa_2_rh_n4 = pd.read_csv(fold + "sphere_r10.{}_rh8.csv".format(method),
    #                                 sep=';')["kappa2"].tolist()
    #     kappa_2_rh_n5 = pd.read_csv(fold + "sphere_r10.{}_rh9.csv".format(method),
    #                                 sep=';')["kappa2"].tolist()
    #     plot_composite_line_hist(
    #         data_arrays=[kappa_1_rh_n1 + kappa_2_rh_n1,
    #                      kappa_1_rh_n2 + kappa_2_rh_n2,
    #                      kappa_1_rh_n3 + kappa_2_rh_n3,
    #                      kappa_1_rh_n4 + kappa_2_rh_n4,
    #                      kappa_1_rh_n5 + kappa_2_rh_n5],
    #         labels=["RadiusHit=5", "RadiusHit=6", "RadiusHit=7",
    #                 "RadiusHit=8", "RadiusHit=9"],
    #         line_styles=['-.', '-.', '--', '-', ':'],
    #         markers=['x', 'v', '^', 's', 'o'],
    #         colors=['b', 'c', 'g', 'y', 'r'],
    #         title="{} on sphere (icosahedron 1280, {}% noise)".format(method, n),
    #         xlabel="Estimated principal curvatures",
    #         ylabel="Counts",
    #         outfile=("{}icosphere_r10_noise{}.{}_rh5-9."
    #                  "kappa_1_and_2.png".format(plot_fold, n, method)),
    #         num_bins=5, value_range=None, max_val=None, freq=False
    #     )

    # # sphere kappa_1 and kappa_2 values for a fixed RadiusHit but different
    # # number of points used for fitting
    # method = 'VVCF'
    # kappa_1_n1 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 10), sep=';')["kappa1"].tolist()
    # kappa_1_n2 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 20), sep=';')["kappa1"].tolist()
    # kappa_1_n3 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 30), sep=';')["kappa1"].tolist()
    # kappa_1_n4 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 40), sep=';')["kappa1"].tolist()
    # kappa_1_n5 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 50), sep=';')["kappa1"].tolist()
    # kappa_2_n1 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 10), sep=';')["kappa2"].tolist()
    # kappa_2_n2 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 20), sep=';')["kappa2"].tolist()
    # kappa_2_n3 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 30), sep=';')["kappa2"].tolist()
    # kappa_2_n4 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 40), sep=';')["kappa2"].tolist()
    # kappa_2_n5 = pd.read_csv(fold + "sphere_r10.{}_{}points_rh8.csv".format(
    #     method, 50), sep=';')["kappa2"].tolist()
    # plot_composite_line_hist(
    #     data_arrays=[kappa_1_n1 + kappa_2_n1,
    #                   kappa_1_n2 + kappa_2_n2,
    #                   kappa_1_n3 + kappa_2_n3,
    #                   kappa_1_n4 + kappa_2_n4,
    #                   kappa_1_n5 + kappa_2_n5],
    #     labels=["10 points", "20 points", "30 points", "40 points", "50 points"],
    #     line_styles=['-.', '-.', '--', '-', ':'],
    #     markers=['x', 'v', '^', 's', 'o'],
    #     colors=['b', 'c', 'g', 'y', 'r'],
    #     title="{}, rh=8 on sphere (icosahedron 1280, {}% noise)".
    #           format(method, n),
    #     xlabel="Estimated principal curvatures",
    #     ylabel="Counts",
    #     outfile=("{}icosphere_r10_noise{}.{}_10-50points_rh8."
    #              "kappa_1_and_2.png".format(plot_fold, n, method)),
    #     num_bins=5, value_range=None, max_val=None, freq=False
    # )

    # # sphere kappa_1 and kappa_2 errors
    # VCTV_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VCTV_rh8.csv", sep=';'
    #                                   )["kappa1RelErrors"].tolist()
    # VCTV_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VCTV_rh8.csv", sep=';'
    #                                   )["kappa2RelErrors"].tolist()
    # VVCF_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VVCF_rh8.csv", sep=';'
    #                                   )["kappa1RelErrors"].tolist()
    # VVCF_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VVCF_rh8.csv", sep=';'
    #                                   )["kappa2RelErrors"].tolist()
    # VV_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VV_rh8.csv", sep=';'
    #                                 )["kappa1RelErrors"].tolist()
    # VV_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VV_rh8.csv", sep=';'
    #                                 )["kappa2RelErrors"].tolist()
    # VTK_kappa_1_errors = pd.read_csv(fold + "sphere_r10.VTK.csv", sep=';'
    #                                  )["kappa1RelErrors"].tolist()
    # VTK_kappa_2_errors = pd.read_csv(fold + "sphere_r10.VTK.csv", sep=';'
    #                                  )["kappa2RelErrors"].tolist()
    # plot_composite_line_hist(
    #     data_arrays=[VCTV_kappa_1_errors + VCTV_kappa_2_errors,
    #                   VVCF_kappa_1_errors + VVCF_kappa_2_errors,
    #                   VV_kappa_1_errors + VV_kappa_2_errors,
    #                   VTK_kappa_1_errors + VTK_kappa_2_errors],
    #     labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
    #     line_styles=['-', '-.', '--', ':'],
    #     markers=['^', 'o', 'v', 's'],
    #     colors=['b', 'g', 'c', 'r'],
    #     title="Sphere (icosahedron 1280, {}% noise)".format(n),
    #     xlabel="Principal curvatures relative error",
    #     ylabel="Frequency",
    #     outfile=("{}icosphere_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
    #              "kappa_1_and_2_errors_20bins_freq.png".format(plot_fold, n)),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True
    # )
    # plot_composite_line_hist(  # cumulative
    #     data_arrays=[VCTV_kappa_1_errors + VCTV_kappa_2_errors,
    #                   VVCF_kappa_1_errors + VVCF_kappa_2_errors,
    #                   VV_kappa_1_errors + VV_kappa_2_errors,
    #                   VTK_kappa_1_errors + VTK_kappa_2_errors],
    #     labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
    #     line_styles=['-', '-.', '--', ':'],
    #     markers=['^', 'o', 'v', 's'],
    #     colors=['b', 'g', 'c', 'r'],
    #     title="Sphere (icosahedron 1280, {}% noise)".format(n),
    #     xlabel="Principal curvatures relative error",
    #     ylabel="Cumulative frequency",
    #     outfile=("{}icosphere_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
    #              "kappa_1_and_2_errors_20bins_cum_freq.png".format(
    #               plot_fold, n)),
    #     num_bins=20, value_range=(0, 1), max_val=1, freq=True, cumulative=True
    # )

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
    # plot_composite_line_hist(
    #     data_arrays=[VCTV_kappa_1_errors, VVCF_kappa_1_errors,
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

    # torus kappa_1 values for different RadiusHit and methods
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "torus/files4plotting/")
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "torus/plots/")
    base_filename = "torus_rr25_csr10"
    principal_components = {1: "maximal", 2: "minimal"}
    # for method in ['VVCF_50points']:  # 'VV', 'VCTV'
    #     for i in principal_components.keys():
    #         kappa_arrays = []
    #         labels = []
    #         for rh in range(1, 6):  # 5, 10
    #             kappa_array = pd.read_csv(fold + "{}.{}_rh{}.csv".format(
    #                 base_filename, method, rh), sep=';')["kappa{}".format(
    #                     i)].tolist()
    #             kappa_arrays.append(kappa_array)
    #             label = "RadiusHit={}".format(rh)
    #             labels.append(label)
    #         plot_composite_line_hist(
    #             data_arrays=kappa_arrays,
    #             labels=labels,
    #             line_styles=['-.', '-.', '--', '-', ':'],
    #             markers=['x', 'v', '^', 's', 'o'],
    #             colors=['b', 'c', 'g', 'y', 'r'],
    #             title="{} on torus (major radius=25, minor radius=10)".format(
    #                 method),
    #             xlabel="Estimated {} principal curvature".format(
    #                 principal_components[i]),
    #             ylabel="Counts",
    #             outfile=("{}{}.{}_rh1-5.kappa_{}.png".format(  # TODO rh range!
    #                 plot_fold, base_filename, method, i)),
    #             num_bins=5, value_range=None, max_val=None, freq=False
    #         )
    # torus T_1, T_2, kappa_1 and kappa_2 errors (with best rh for each method)
    for i in principal_components.keys():
        df = pd.read_csv("{}{}.VCTV_rh8.csv".format(fold, base_filename), sep=';')
        VCTV_T_errors = df["T{}Errors".format(i)].tolist()
        VCTV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VVCF_kappa_errors = pd.read_csv("{}{}.VVCF_50points_rh3.csv".format(
            fold, base_filename), sep=';')["kappa{}RelErrors".format(
                i)].tolist()

        df = pd.read_csv("{}{}.VV_rh8.csv".format(fold, base_filename), sep=';')
        VV_T_errors = df["T{}Errors".format(i)].tolist()  # same for VVCF
        VV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VTK_kappa_errors = pd.read_csv("{}{}.VTK.csv".format(
            fold, base_filename), sep=';')["kappa{}RelErrors".format(
                i)].tolist()
        data = [VCTV_T_errors, VV_T_errors]
        plot_composite_line_hist(  # cumulative
            data_arrays=data,
            labels=["VCTV rh=8", "VV rh=8"],
            line_styles=['-', '--'], markers=['^', 'v'],
            colors=['b', 'c'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal direction error".format(
                principal_components[i]),
            ylabel="Cumulative frequency",
            outfile="{}{}.VV_VCTV_rh8.T_{}_errors_bins20_cum_freq."
                    "png".format(plot_fold, base_filename, i),
            num_bins=20, freq=True, cumulative=True,
            value_range=(0, max([max(d) for d in data]))
            # , max_val=1
        )
        data = [VCTV_kappa_errors, VVCF_kappa_errors,
                VV_kappa_errors, VTK_kappa_errors]
        plot_composite_line_hist(  # cumulative
            data_arrays=data,
            labels=["VCTV rh=8", "VVCF rh=3", "VV rh=8", "VTK"],
            line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
            colors=['b', 'g', 'c', 'r'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal curvature relative error".format(
                principal_components[i]),
            ylabel="Cumulative frequency",
            outfile=("{}{}.VV_VVCF_50points_VCTV_rh8_vs_VTK.kappa_{}_errors_"
                     "bins20_cum_freq.png".format(plot_fold, base_filename, i)),
            num_bins=20, freq=True, cumulative=True,
            value_range=(0, max([max(d) for d in data]))
            # , max_val=1
        )
