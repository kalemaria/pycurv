import matplotlib.pyplot as plt
# print(plt.style.available)
plt.style.use('presentation')
import numpy as np
import os
import pandas as pd

from pysurf import pexceptions

"""
Functions for plotting estimation errors of curvature estimation methods using
"synthetic" benchmark surfaces.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


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
        raise pexceptions.PySegInputError(
            expr='plot_hist',
            msg="Range has to be a tuple of two numbers (min, max).")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
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
        raise pexceptions.PySegInputError(
            expr='plot_hist',
            msg="Range has to be a tuple of two numbers (min, max).")
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bincenters, counts, ls='-', marker='^', c="b", label=label,
             linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
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
        raise pexceptions.PySegInputError(
            expr='plot_hist',
            msg="Range has to be a tuple of two numbers (min, max).")
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if cumulative is True:
        # counts = np.array([np.sum(counts[0:i+1]) for i in range(len(counts))])
        counts = np.cumsum(counts)
    if freq is True:
        counts = counts / float(len(values))  # normalized to max 1
    plt.plot(bincenters, counts, ls=ls, marker=marker, c=c, label=label,
             linewidth=2)


def plot_composite_line_hist(
        labels, line_styles, markers, colors,
        xlabel, ylabel, title=None,
        data_arrays=None, data_files=None,
        num_bins=20, value_range=None, y_range=None, max_val=None, freq=False,
        cumulative=False, outfile=None):
    """
    Plots several data sets as line histograms in one plot.
    Args:
        labels: list of legend labels (str) for the data sets
        line_styles: list of line styles (str)
        markers: list of plotting characters (str)
        colors: list of colors (str)
        xlabel (str): X axis label
        ylabel (str): Y axis label
        title (str, optional): title of the plot
        data_arrays (list, optional): list of data arrays
        data_files (list, optional): list of data file names (str)
        num_bins (int, optional): number of bins for the histogram (default 20)
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
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
                print("File {} not found!".format(data_file))
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
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_range is not None:
        plt.ylim(y_range)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.grid(True)
    plt.tight_layout()
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)
        print("The plot was saved as {}".format(outfile))


def plot_plane_normals(n=10, y_range=None, res=20):
    """ Plots estimated normals errors by VV versus original face normals
    (calculated by VTK) on a noisy plane surface.

    Args:
        n (int, optional): noise in % (default 10)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        res (int, optional): defines the size of the square plane in pixels and
            triangle division: 2*res
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "plane/res{}_noise{}/files4plotting/with_borders/".format(res, n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "plane/res{}_noise{}/plots/with_borders/".format(res, n))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "plane_half_size{}".format(res)
    VCTV_rh8_normal_errors = pd.read_csv("{}{}.VCTV_rh8.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    VCTV_rh4_normal_errors = pd.read_csv("{}{}.VCTV_rh4.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    VTK_normal_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                    sep=';')["normalErrors"].tolist()
    data = [VCTV_rh8_normal_errors, VCTV_rh4_normal_errors, VTK_normal_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VV RadiusHit=8", "VV RadiusHit=4", "VTK"],
        line_styles=['-', '--', ':'], markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Plane ({}% noise)".format(n),
        xlabel="Normal orientation error", ylabel="Cumulative frequency",
        outfile="{}plane_res{}_noise{}.VV_vs_VTK.normal_errors_bin20_cum_freq"
                ".png".format(plot_fold, res, n),
        num_bins=20, freq=True, cumulative=True,
        value_range=(0, max([max(d) for d in data])), y_range=y_range
    )


def plot_cylinder_kappa_1_diff_rh(n=0):
    """Plots estimated kappa_1 values histograms on a cylinder surface by
    different methods (VV, VVCF and VCTV) using different RadiusHit.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "cylinder/noise0/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "cylinder/noise0/plots/".format(n))
    basename = "cylinder_r10_h25_eb5"
    for method in ['VCTV', 'VV_area2']:  # 'VV', 'VVCF_50points'
        kappa_arrays = []
        labels = []
        for rh in range(5, 10):
            kappa_array = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')["kappa1"].tolist()
            kappa_arrays.append(kappa_array)
            label = "RadiusHit={}".format(rh)
            labels.append(label)
        if method == 'VV_area2':
            method = 'VV2'
        plot_composite_line_hist(
            data_arrays=kappa_arrays,
            labels=labels,
            line_styles=[':', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on cylinder ({}% noise)".format(method, n),
            xlabel="Estimated maximal principal curvature",
            ylabel="Counts",
            outfile=("{}{}_noise{}.{}_rh5-9.kappa_1.png".format(
                plot_fold, basename, n, method)),
            num_bins=5, value_range=None, max_val=None, freq=False
        )


def plot_cylinder_T_2_and_kappa_1_errors_VV_VCTV(
        n=0, y_range=None, value_range_T=None, value_range_kappa=None,
        area2=True, exclude_borders=5):
    """Plots estimated kappa_2 and T_1 errors histograms on a cylinder surface
    for different methods (VV, VVCF and VCTV) and optimal RadiusHit for each
    method.

    Args:
        n (int, optional): noise in % (default 0)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        value_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        value_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        area2 (boolean, optional): if True (default), votes are
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        exclude_borders (int, optional): how many voxels from border were
            excluded for curvature calculation (default 5)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "cylinder/noise0/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "cylinder/noise0/plots/".format(n))
    basename = "cylinder_r10_h25_eb{}".format(exclude_borders)
    df = pd.read_csv("{}{}.VCTV_rh7.csv".format(fold, basename), sep=';')
    VCTV_T_2_errors = df["T2Errors"].tolist()
    VCTV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    if area2:
        a = "_area2"
        line_styles = ['-', '-.']
        markers = ['^', 'o']
        colors = ['b', 'orange']
    else:
        a = ""
        line_styles = ['-', '--']
        markers = ['^', 'v']
        colors = ['b', 'c']
    df = pd.read_csv("{}{}.VV{}_rh5.csv".format(fold, basename, a), sep=';')
    VV_T_2_errors = df["T2Errors"].tolist()
    VV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VTK_kappa_1_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa1RelErrors"].tolist()
    data = [VCTV_T_2_errors, VV_T_2_errors]
    if value_range_T is None:
        value_range_T = (0, max([max(d) for d in data]))
    if area2:
        a = "2"
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV RadiusHit=7", "VV{} RadiusHit=5".format(a)],
        line_styles=line_styles, markers=markers, colors=colors,
        title="Cylinder ({}% noise)".format(n),
        xlabel="Minimal principal direction error",
        ylabel="Cumulative frequency",
        outfile="{}{}_noise{}.VV{}_rh5_VCTV_rh7.T_2_errors_bins20_cum_freq"
                "_slides.png".format(plot_fold, basename, n, a),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=value_range_T,
        y_range=y_range
    )
    data = [VCTV_kappa_1_errors,
            VV_kappa_1_errors, VTK_kappa_1_errors]
    line_styles.append(':')
    markers.append('s')
    colors.append('r')
    if value_range_kappa is None:
        value_range_kappa = (0, max([max(d) for d in data]))
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV RadiusHit=7", "VV{} RadiusHit=5".format(a), "VTK"],
        line_styles=line_styles, markers=markers, colors=colors,
        title="Cylinder ({}% noise)".format(n),
        xlabel="Maximal principal curvature relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}_noise{}.VV{}_VCTV_best_rh_vs_VTK.kappa_1_errors_bins20_"
                 "cum_freq_slides.png".format(plot_fold, basename, n, a)),
        num_bins=20, freq=True, cumulative=True,
        value_range=value_range_kappa, y_range=y_range
    )


def plot_cylinder_T_2_and_kappa_1_errors_VV_VVarea2_VCTV(n=0, y_range=None,
                                                         exclude_borders=5):
    """Plots estimated kappa_2 and T_1 errors histograms on a cylinder surface
    for different methods (VV, VVCF and VCTV) and optimal RadiusHit for each
    method.

    Args:
        n (int, optional): noise in % (default 0)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        exclude_borders (int, optional): how many voxels from border were
            excluded for curvature calculation (default 5)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "cylinder/noise0/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "cylinder/noise0/plots/".format(n))
    basename = "cylinder_r10_h25_eb{}".format(exclude_borders)
    df = pd.read_csv("{}{}.VCTV_rh7.csv".format(fold, basename), sep=';')
    VCTV_T_2_errors = df["T2Errors"].tolist()
    VCTV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.VV_rh5.csv".format(fold, basename), sep=';')
    VV_T_2_errors = df["T2Errors"].tolist()
    VV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.VV_area2_rh5.csv".format(fold, basename), sep=';')
    VV_area2_T_2_errors = df["T2Errors"].tolist()
    VV_area2_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VTK_kappa_1_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa1RelErrors"].tolist()
    data = [VCTV_T_2_errors, VV_T_2_errors, VV_area2_T_2_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV rh=7", "VV rh=5", "VV area2 rh=5"],
        line_styles=['-', '--', '-.'], markers=['^', 'v', 'o'],
        colors=['b', 'c', 'g'],
        title="Cylinder ({}% noise)".format(n),
        xlabel="Minimal principal direction error",
        ylabel="Cumulative frequency",
        outfile="{}{}_noise{}.VV_VVarea2_rh5_VCTV_rh7.T_2_errors_bins20"
                "_cum_freq.png".format(plot_fold, basename, n),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, 0.005),  # (0, max([max(d) for d in data]))
        y_range=y_range
    )
    data = [VCTV_kappa_1_errors, VV_kappa_1_errors, VV_area2_kappa_1_errors,
            VTK_kappa_1_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV rh=7", "VV rh=5", "VV area2 rh=5", "VTK"],  # "VVCF rh=4",
        line_styles=['-', '--', '-.', ':'], markers=['^', 'v', 'o', 's'],
        colors=['b', 'c', 'g', 'r'],
        title="Cylinder ({}% noise)".format(n),
        xlabel="Maximal principal curvature relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}_noise{}.VV_VVarea2_VCTV_best_rh_vs_VTK.kappa_1_errors_"
                 "bins20_cum_freq.png".format(plot_fold, basename, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data])), y_range=y_range
    )


def plot_inverse_cylinder_T_1_and_kappa_2_errors(n=0):
    """Plots estimated kappa_2 and T_1 errors histograms on an inverse cylinder
    surface for different methods (VV, VVCF and VCTV) and optimal RadiusHit for
    each method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "cylinder/noise0/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "cylinder/noise0/plots/".format(n))
    basename = "inverse_cylinder_r10_h25"
    df = pd.read_csv("{}{}.VCTV_rh8.csv".format(fold, basename), sep=';')
    VCTV_T_1_errors = df["T1Errors"].tolist()
    VCTV_kappa_2_errors = df["kappa2RelErrors"].tolist()

    VVCF_kappa_2_errors = pd.read_csv("{}{}.VVCF_rh8.csv".format(
        fold, basename), sep=';')["kappa2RelErrors"].tolist()

    df = pd.read_csv("{}{}.VV_rh8.csv".format(fold, basename), sep=';')
    VV_T_1_errors = df["T1Errors"].tolist()
    VV_kappa_2_errors = df["kappa2RelErrors"].tolist()

    VTK_kappa_2_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa2RelErrors"].tolist()
    data = [VCTV_T_1_errors,
            VV_T_1_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV rh=8", "VV rh=8"],
        line_styles=['-', '--'], markers=['^', 'v'],
        colors=['b', 'c'],
        title="Inverse cylinder ({}% noise)".format(n),
        xlabel="Maximal principal direction error",
        ylabel="Cumulative frequency",
        outfile="{}{}_noise{}.VV_VCTV_rh8.T_1_errors"
                "_bins20_cum_freq.png".format(plot_fold, basename, n),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data]))
    )
    data = [VCTV_kappa_2_errors,
            VVCF_kappa_2_errors,
            VV_kappa_2_errors,
            VTK_kappa_2_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
        line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Inverse cylinder ({}% noise)".format(n),
        xlabel="Minimal principal curvature relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}_noise{}.VV_VVCF_VCTV_rh8_vs_VTK.kappa_2_errors_bins20"
                 "_cum_freq.png".format(plot_fold, basename, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data]))
    )


def plot_sphere_kappa_1_and_2_diff_rh(r=10, n=0, ico=1280, binary=False,
                                      methods=['VV'], rhs=range(5, 10)):
    """Plots estimated kappa_1 and kappa_2 values for a sphere surface
     by different methods (VV, VVCF and VCTV) using different RadiusHit.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (default 1280), icosahedron results with so
            many faces are used; if 0, Gaussian sphere results are used
        binary (boolean, optional): if True (default False), binary sphere
            results are used (ignoring the other options)
        methods (list, optional): tells which method(s) should be used: 'VV'
            for normal vector voting (default), 'VVCF' for curve fitting in
            the two principal directions estimated by VV to estimate the
            principal curvatures or 'VCTV' for vector and curvature tensor
            voting to estimate the principal directions and curvatures
        rhs (list, optional): wanted RadiusHit parameter values (default 5-9)
    """
    first = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/sphere/"
    if binary:
        second = "binary/"
        type = "binary, radius={}".format(r)
    elif ico > 0:
        second = "ico{}_noise{}/".format(ico, n)
        type = "icosahedron {}, radius={}, {}% noise".format(ico, r, n)
    else:
        second = "noise{}/".format(n)
        type = "Gaussian, radius={}, {}% noise".format(r, n)
    fold = ("{}{}files4plotting/".format(first, second))
    plot_fold = ("{}{}plots/".format(first, second))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)
    for method in methods:
        kappa_1_arrays = []
        kappa_2_arrays = []
        kappas_arrays = []
        labels = []
        for rh in rhs:
            df = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')
            kappa_1_array = df["kappa1"].tolist()
            kappa_2_array = df["kappa2"].tolist()
            kappa_1_arrays.append(kappa_1_array)
            kappa_2_arrays.append(kappa_2_array)
            kappas_arrays.append(kappa_1_array + kappa_2_array)
            label = "RadiusHit={}".format(rh)
            labels.append(label)
        if method == "VV_area2":
            method = "VV2"
        plot_composite_line_hist(  # kappa_1
            data_arrays=kappa_1_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on sphere ({})".format(method, type),
            xlabel="Estimated maximal principal curvature",
            ylabel="Counts",
            outfile=("{}{}.{}_rh{}-{}.kappa_1_slides.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=5, value_range=None, max_val=None, freq=False
        )
        plot_composite_line_hist(  # kappa_2
            data_arrays=kappa_2_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on sphere ({})".format(method, type),
            xlabel="Estimated minimal principal curvature",
            ylabel="Counts",
            outfile=("{}{}.{}_rh{}-{}.kappa_2_slides.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=5, value_range=None, max_val=None, freq=False
        )
        plot_composite_line_hist(  # kappa_1 + kappa_2
            data_arrays=kappas_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on sphere ({})".format(method, type),
            xlabel="Estimated principal curvatures",
            ylabel="Counts",
            outfile=("{}{}.{}_rh{}-{}.kappa_1_and_2_slides.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=5, value_range=None, max_val=None, freq=False
        )


def plot_sphere_kappa_1_and_2_fitting_diff_num_points(n=0):
    """ Plots estimated kappa_1 and kappa_2 values for a icosahedron sphere
    surface by VVCF using a fixed RadiusHit but different number of points used
    for fitting.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "sphere/ico1280_noise{}/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "sphere/ico1280_noise{}/plots/".format(n))
    method = 'VVCF'
    basename = "sphere_r10"
    kappas_arrays = []
    labels = []
    for p in range(10, 60, 10):
        df = pd.read_csv("{}{}.{}_{}points_rh8.csv".format(
            fold, basename, method, p), sep=';')
        kappa_1_array = df["kappa1"].tolist()
        kappa_2_array = df["kappa2"].tolist()
        kappas_arrays.append(kappa_1_array + kappa_2_array)
        label = "{} points".format(p)
        labels.append(label)
    plot_composite_line_hist(
        data_arrays=kappas_arrays,
        labels=labels,
        line_styles=['-.', '-.', '--', '-', ':'],
        markers=['x', 'v', '^', 's', 'o'],
        colors=['b', 'c', 'g', 'y', 'r'],
        title="{}, rh=8 on sphere (icosahedron 1280, {}% noise)".
              format(method, n),
        xlabel="Estimated principal curvatures",
        ylabel="Counts",
        outfile=("{}ico{}_noise{}.{}_10-50points_rh8.kappa_1_and_2.png".format(
            plot_fold, basename, n, method)),
        num_bins=5, value_range=None, max_val=None, freq=False
    )


def plot_sphere_kappa_1_and_2_errors(r=10, n=0, ico=1280, binary=False):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (VV, VVCF, VCTV and VTK) and an optimal RadiusHit for
    each method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (default 1280), icosahedron results with so
            many faces are used; if 0, Gaussian sphere results are used
        binary (boolean, optional): if True (default False), binary sphere
            results are used (ignoring the other options)
    """
    first = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/sphere/"
    if binary:
        second = "binary/"
        type = "binary, radius={}".format(r)
    elif ico > 0:
        second = "ico{}_noise{}/".format(ico, n)
        type = "icosahedron {}, radius={}, {}% noise".format(ico, r, n)
    else:
        second = "noise{}/".format(n)
        type = "Gaussian, radius={}, {}% noise".format(r, n)
    fold = ("{}{}files4plotting/".format(first, second))
    plot_fold = ("{}{}plots/".format(first, second))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df_VCTV = pd.read_csv("{}{}.VCTV_rh8.csv".format(fold, basename), sep=';')
    VCTV_kappa_1_errors = df_VCTV["kappa1RelErrors"].tolist()
    VCTV_kappa_2_errors = df_VCTV["kappa2RelErrors"].tolist()

    df_VVCF = pd.read_csv("{}{}.VVCF_50points_rh3.5.csv".format(
        fold, basename), sep=';')
    VVCF_kappa_1_errors = df_VVCF["kappa1RelErrors"].tolist()
    VVCF_kappa_2_errors = df_VVCF["kappa2RelErrors"].tolist()

    df_VV = pd.read_csv("{}{}.VV_rh9.csv".format(fold, basename), sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    df_VTK = pd.read_csv("{}{}.VTK.csv".format(fold, basename), sep=';')
    VTK_kappa_1_errors = df_VTK["kappa1RelErrors"].tolist()
    VTK_kappa_2_errors = df_VTK["kappa2RelErrors"].tolist()

    data = [VCTV_kappa_1_errors + VCTV_kappa_2_errors,
            VVCF_kappa_1_errors + VVCF_kappa_2_errors,
            VV_kappa_1_errors + VV_kappa_2_errors,
            VTK_kappa_1_errors + VTK_kappa_2_errors]
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["VCTV rh=8", "VVCF 50 p. rh=3.5", "VV rh=9", "VTK"],
        line_styles=['-', '-.', '--', ':'],
        markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Sphere ({})".format(type),
        xlabel="Principal curvatures relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}.VV_VVCF50p_VCTV_vs_VTK."
                 "kappa_1_and_2_errors_20bins_cum_freq.png".format(
                  plot_fold, basename)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data]))
    )


def plot_sphere_kappa_1_and_2_errors_noVVCF(
        r=10, rhVV=8, rhVCTV=8, n=0, ico=1280, binary=False, y_range=None):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (VV, VCTV and VTK) and an optimal RadiusHit for each
    method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 8)
        rhVCTV (int, optional): radius_hit for VCTV (default 8)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (default 1280), icosahedron results with so
            many faces are used; if 0, Gaussian sphere results are used
        binary (boolean, optional): if True (default False), binary sphere
            results are used (ignoring the other options)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
    """
    first = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/sphere/"
    if binary:
        second = "binary/"
        type = "binary, radius={}".format(r)
    elif ico > 0:
        second = "ico{}_noise{}/".format(ico, n)
        type = "icosahedron {}, radius={}, {}% noise".format(ico, r, n)
    else:
        second = "noise{}/".format(n)
        type = "Gaussian, radius={}, {}% noise".format(r, n)
    fold = ("{}{}files4plotting/".format(first, second))
    plot_fold = ("{}{}plots/".format(first, second))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df_VCTV = pd.read_csv("{}{}.VCTV_rh{}.csv".format(fold, basename, rhVCTV),
                          sep=';')
    VCTV_kappa_1_errors = df_VCTV["kappa1RelErrors"].tolist()
    VCTV_kappa_2_errors = df_VCTV["kappa2RelErrors"].tolist()

    df_VV = pd.read_csv("{}{}.VV_rh{}.csv".format(fold, basename, rhVV),
                        sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    df_VV2 = pd.read_csv("{}{}.VV_area2_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
    VV_area2_kappa_1_errors = df_VV2["kappa1RelErrors"].tolist()
    VV_area2_kappa_2_errors = df_VV2["kappa2RelErrors"].tolist()

    df_VTK = pd.read_csv("{}{}.VTK.csv".format(fold, basename), sep=';')
    VTK_kappa_1_errors = df_VTK["kappa1RelErrors"].tolist()
    VTK_kappa_2_errors = df_VTK["kappa2RelErrors"].tolist()

    data = [VCTV_kappa_1_errors + VCTV_kappa_2_errors,
            VV_kappa_1_errors + VV_kappa_2_errors,
            VV_area2_kappa_1_errors + VV_area2_kappa_2_errors,
            VTK_kappa_1_errors + VTK_kappa_2_errors]
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["VCTV RadiusHit={}".format(rhVCTV),
                "VV RadiusHit={}".format(rhVV),
                "VV2 RadiusHit={}".format(rhVV), "VTK"],
        line_styles=['-', '--', '-.', ':'],
        markers=['^', 'v', 'o', 's'],
        colors=['b', 'c', 'orange', 'r'],
        title="Sphere ({})".format(type),
        xlabel="Principal curvatures relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}.VV_VVarea2rh{}_VCTVrh{}_vs_VTK."
                 "kappa_1_and_2_errors_20bins_cum_freq_slides.png".format(
                  plot_fold, basename, rhVV, rhVCTV)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data])), y_range=y_range
    )


def plot_sphere_kappa_1_and_2_errors_VV_VCTV(
        r=10, rhVV=8, rhVCTV=8, n=0, ico=1280, binary=False, value_range=None,
        y_range=None, area2=False):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (VV and VCTV) and an optimal RadiusHit for each
    method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 8)
        rhVCTV (int, optional): radius_hit for VCTV (default 8)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (default 1280), icosahedron results with so
            many faces are used; if 0, Gaussian sphere results are used
        binary (boolean, optional): if True (default False), binary sphere
            results are used (ignoring the other options)
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        area2 (boolean, optional): if True (default False), votes are
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
    """
    first = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/sphere/"
    if binary:
        second = "binary/"
        type = "binary, radius={}".format(r)
    elif ico > 0:
        second = "ico{}_noise{}/".format(ico, n)
        type = "icosahedron {}, radius={}, {}% noise".format(ico, r, n)
    else:
        second = "noise{}/".format(n)
        type = "Gaussian, radius={}, {}% noise".format(r, n)
    fold = ("{}{}files4plotting/".format(first, second))
    plot_fold = ("{}{}plots/".format(first, second))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df_VCTV = pd.read_csv("{}{}.VCTV_rh{}.csv".format(fold, basename, rhVCTV),
                          sep=';')
    VCTV_kappa_1_errors = df_VCTV["kappa1RelErrors"].tolist()
    VCTV_kappa_2_errors = df_VCTV["kappa2RelErrors"].tolist()

    if area2:
        a = "_area2"
        line_styles = ['-', '-.']
        markers = ['^', 'o']
        colors = ['b', 'orange']
    else:
        a = ""
        line_styles = ['-', '--']
        markers = ['^', 'v']
        colors = ['b', 'c']
    df_VV = pd.read_csv("{}{}.VV{}_rh{}.csv".format(fold, basename, a, rhVV),
                        sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    data = [VCTV_kappa_1_errors + VCTV_kappa_2_errors,
            VV_kappa_1_errors + VV_kappa_2_errors]
    if value_range is None:
        value_range = (0, max([max(d) for d in data]))
    if area2:
        a = "2"
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["VCTV RadiusHit={}".format(rhVCTV),
                "VV{} RadiusHit={}".format(a, rhVV)],
        line_styles=line_styles, markers=markers, colors=colors,
        title="Sphere ({})".format(type),
        xlabel="Principal curvatures relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}.VV{}rh{}_vs_VCTVrh{}."
                 "kappa_1_and_2_errors_20bins_cum_freq_range{}_slides.png".
                  format(plot_fold, basename, a, rhVV, rhVCTV, value_range[1])),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=value_range, y_range=y_range
    )


def plot_sphere_kappa_1_and_2_errors_VV_VVarea2_VCTV(
        r=10, rhVV=8, rhVCTV=8, n=0, ico=1280, binary=False, value_range=None,
        y_range=None):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (VV and VCTV) and an optimal RadiusHit for each
    method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 8)
        rhVCTV (int, optional): radius_hit for VCTV (default 8)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (default 1280), icosahedron results with so
            many faces are used; if 0, Gaussian sphere results are used
        binary (boolean, optional): if True (default False), binary sphere
            results are used (ignoring the other options)
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
    """
    first = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/sphere/"
    if binary:
        second = "binary/"
        type = "binary, radius={}".format(r)
    elif ico > 0:
        second = "ico{}_noise{}/".format(ico, n)
        type = "icosahedron {}, radius={}, {}% noise".format(ico, r, n)
    else:
        second = "noise{}/".format(n)
        type = "Gaussian, radius={}, {}% noise".format(r, n)
    fold = ("{}{}files4plotting/".format(first, second))
    plot_fold = ("{}{}plots/".format(first, second))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df_VCTV = pd.read_csv("{}{}.VCTV_rh{}.csv".format(fold, basename, rhVCTV),
                          sep=';')
    VCTV_kappa_1_errors = df_VCTV["kappa1RelErrors"].tolist()
    VCTV_kappa_2_errors = df_VCTV["kappa2RelErrors"].tolist()

    df_VV = pd.read_csv("{}{}.VV_rh{}.csv".format(fold, basename, rhVV),
                        sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    df_VV2 = pd.read_csv("{}{}.VV_area2_rh{}.csv".format(fold, basename, rhVV),
                        sep=';')
    VV_area2_kappa_1_errors = df_VV2["kappa1RelErrors"].tolist()
    VV_area2_kappa_2_errors = df_VV2["kappa2RelErrors"].tolist()

    data = [VCTV_kappa_1_errors + VCTV_kappa_2_errors,
            VV_kappa_1_errors + VV_kappa_2_errors,
            VV_area2_kappa_1_errors + VV_area2_kappa_2_errors]
    if value_range is None:
        value_range = (0, max([max(d) for d in data]))
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["VCTV RadiusHit={}".format(rhVCTV),
                "VV RadiusHit={}".format(rhVV),
                "VV2 RadiusHit={}".format(rhVV)],
        line_styles=['-', '--', '-.'],
        markers=['^', 'v', 'o'],
        colors=['b', 'c', 'orange'],
        title="Sphere ({})".format(type),
        xlabel="Principal curvatures relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}.VV_VVarea2rh{}_vs_VCTVrh{}."
                 "kappa_1_and_2_errors_20bins_cum_freq_range{}_slides.png"
                 .format(plot_fold, basename, rhVV, rhVCTV, value_range[1])),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=value_range, y_range=y_range
    )


def plot_inverse_sphere_kappa_1_and_2_errors(n=0):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on an inverse
    icosahedron sphere surface for different methods (VV, VVCF and VCTV) and an
    optimal RadiusHit for each method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "sphere/ico1280_noise{}/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "sphere/ico1280_noise{}/plots/".format(n))
    basename = "inverse_sphere_r10"

    df_VCTV = pd.read_csv("{}{}.VCTV_rh8.csv".format(fold, basename), sep=';')
    VCTV_kappa_1_errors = df_VCTV["kappa1RelErrors"].tolist()
    VCTV_kappa_2_errors = df_VCTV["kappa2RelErrors"].tolist()

    df_VVCF = pd.read_csv("{}{}.VVCF_50points_rh3.5.csv".format(
        fold, basename), sep=';')
    VVCF_kappa_1_errors = df_VVCF["kappa1RelErrors"].tolist()
    VVCF_kappa_2_errors = df_VVCF["kappa2RelErrors"].tolist()

    df_VV = pd.read_csv("{}{}.VV_rh9.csv".format(fold, basename), sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    df_VTK = pd.read_csv("{}{}.VTK.csv".format(fold, basename), sep=';')
    VTK_kappa_1_errors = df_VTK["kappa1RelErrors"].tolist()
    VTK_kappa_2_errors = df_VTK["kappa2RelErrors"].tolist()

    data = [VCTV_kappa_1_errors + VCTV_kappa_2_errors,
            VVCF_kappa_1_errors + VVCF_kappa_2_errors,
            VV_kappa_1_errors + VV_kappa_2_errors,
            VTK_kappa_1_errors + VTK_kappa_2_errors]
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["VCTV rh=8", "VVCF 50 p. rh=3.5", "VV rh=9", "VTK"],
        line_styles=['-', '-.', '--', ':'],
        markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Inverse sphere (icosahedron 1280, {}% noise)".format(n),
        xlabel="Principal curvatures relative error",
        ylabel="Cumulative frequency",
        outfile=("{}inverse_icosphere_r10_noise{}.VV_VVCF50p_VCTV_vs_VTK."
                 "kappa_1_and_2_errors_20bins_cum_freq.png".format(
                  plot_fold, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data]))
    )


def plot_torus_kappa_1_and_2_diff_rh():
    """Plots estimated kappa_1 and kappa_2 values for a torus surface
     by different methods (VV, VVCF and VCTV) using different RadiusHit.
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "torus/files4plotting/")
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "torus/plots/")
    basename = "torus_rr25_csr10"
    principal_components = {1: "maximal", 2: "minimal"}
    for method in ['VV', 'VCTV', 'VVCF_50points']:
        for i in principal_components.keys():
            kappas_arrays = []
            labels = []
            for rh in range(5, 10):
                kappa_array = pd.read_csv(fold + "{}.{}_rh{}.csv".format(
                    basename, method, rh), sep=';')["kappa{}".format(
                        i)].tolist()
                kappas_arrays.append(kappa_array)
                label = "RadiusHit={}".format(rh)
                labels.append(label)
            plot_composite_line_hist(
                data_arrays=kappas_arrays,
                labels=labels,
                line_styles=['-.', '-.', '--', '-', ':'],
                markers=['x', 'v', '^', 's', 'o'],
                colors=['b', 'c', 'g', 'y', 'r'],
                title="{} on torus (major radius=25, minor radius=10)".format(
                    method),
                xlabel="Estimated {} principal curvature".format(
                    principal_components[i]),
                ylabel="Counts",
                outfile=("{}{}.{}_rh5-9.kappa_{}.png".format(
                    plot_fold, basename, method, i)),
                num_bins=5, value_range=None, max_val=None, freq=False
            )


def plot_torus_kappa_1_and_2_T_1_and_2_errors_noVVCF(
        rhVV=8, rhVCTV=8, n=0, value_range_T=None, value_range_kappa=None,
        y_range=None, area2=False):
    """
    Plots estimated kappa_1 and kappa_2 as well as T_1 and T_2 errors histograms
    on a torus surface for different methods (VV, and VCTV) and an optimal
    RadiusHit for each method.

    Args:
        rhVV (int, optional): radius_hit for VV (default 8)
        rhVCTV (int, optional): radius_hit for VCTV (default 8)
        n (int, optional): noise in % (default 0)
        value_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        value_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        area2 (boolean, optional): if True (default False), votes are
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "torus/files4plotting/")
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "torus/plots/")
    basename = "torus_rr25_csr10"
    principal_components = {1: "maximal", 2: "minimal"}
    for i in principal_components.keys():
        df = pd.read_csv("{}{}.VCTV_rh{}.csv".format(fold, basename, rhVCTV),
                         sep=';')
        VCTV_T_errors = df["T{}Errors".format(i)].tolist()
        VCTV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        # VVCF_kappa_errors = pd.read_csv("{}{}.VVCF_50points_rh3.csv".format(
        #     fold, basename), sep=';')["kappa{}RelErrors".format(i)].tolist()

        if area2:
            a = "_area2"
        else:
            a = ""
        df = pd.read_csv("{}{}.VV{}_rh{}.csv".format(fold, basename, a, rhVV),
                         sep=';')
        VV_T_errors = df["T{}Errors".format(i)].tolist()  # same for VVCF
        VV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VTK_kappa_errors = pd.read_csv("{}{}.VTK.csv".format(
            fold, basename), sep=';')["kappa{}RelErrors".format(
                i)].tolist()
        data = [VCTV_T_errors, VV_T_errors]
        if value_range_T is None:
            value_range_T = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["VCTV rh={}".format(rhVCTV), "VV{} rh={}".format(a, rhVV)],
            line_styles=['-', '--'], markers=['^', 'v'],
            colors=['b', 'c'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal direction error".format(
                principal_components[i]),
            ylabel="Cumulative frequency",
            outfile="{}{}.VVrh{}{}_vs_VCTVrh{}.T_{}_errors_bins20_cum_freq.png"
                    .format(plot_fold, basename, rhVV, a, rhVCTV, i),
            num_bins=20, freq=True, cumulative=True,
            value_range=value_range_T,
            y_range=y_range  # , max_val=1
        )
        data = [VCTV_kappa_errors,  # VVCF_kappa_errors,
                VV_kappa_errors, VTK_kappa_errors]
        if value_range_kappa is None:
            value_range_kappa = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["VCTV rh={}".format(rhVCTV), "VV{} rh={}".format(a, rhVV),
                    "VTK"],  # "VVCF rh=3",
            line_styles=['-', '--', ':'], markers=['^', 'v', 's'],  # '-.', 'o',
            colors=['b', 'c', 'r'],  # 'g',
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal curvature relative error".format(
                principal_components[i]),
            ylabel="Cumulative frequency",
            outfile=("{}{}.VVrh{}{}_VCTVrh{}_vs_VTK."  # VVCF_50points_
                     "kappa_{}_errors_bins20_cum_freq.png".format(
                      plot_fold, basename, rhVV, a, rhVCTV, i)),
            num_bins=20, freq=True, cumulative=True,
            value_range=value_range_kappa,
            y_range=y_range  # , max_val=1
        )


def plot_torus_kappa_1_and_2_T_1_and_2_errors_VTK_VV_VVarea2_VCTV(
        rhVV=8, rhVCTV=8, n=0, value_range_T=None, value_range_kappa=None,
        y_range=None):
    """
    Plots estimated kappa_1 and kappa_2 as well as T_1 and T_2 errors histograms
    on a torus surface for different methods (VV, VVarea2 and VCTV) and an
    optimal RadiusHit for each method.

    Args:
        rhVV (int, optional): radius_hit for VV (default 8)
        rhVCTV (int, optional): radius_hit for VCTV (default 8)
        n (int, optional): noise in % (default 0)
        value_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        value_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "torus/files4plotting/")
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "torus/plots/")
    basename = "torus_rr25_csr10"
    principal_components = {1: "maximal", 2: "minimal"}
    for i in principal_components.keys():
        df = pd.read_csv("{}{}.VCTV_rh{}.csv".format(fold, basename, rhVCTV),
                         sep=';')
        VCTV_T_errors = df["T{}Errors".format(i)].tolist()
        VCTV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        df = pd.read_csv("{}{}.VV_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
        VV_T_errors = df["T{}Errors".format(i)].tolist()
        VV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        df = pd.read_csv("{}{}.VV_area2_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
        VV_area2_T_errors = df["T{}Errors".format(i)].tolist()
        VV_area2_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VTK_kappa_errors = pd.read_csv("{}{}.VTK.csv".format(
            fold, basename), sep=';')["kappa{}RelErrors".format(
                i)].tolist()

        data = [VCTV_T_errors, VV_T_errors, VV_area2_T_errors]
        if value_range_T is None:
            value_range = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["VCTV RadiusHit={}".format(rhVCTV),
                    "VV RadiusHit={}".format(rhVV),
                    "VV2 RadiusHit={}".format(rhVV)],
            line_styles=['-', '--', '-.'], markers=['^', 'v', 'o'],
            colors=['b', 'c', 'orange'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal direction error".format(
                principal_components[i].capitalize()),
            ylabel="Cumulative frequency",
            outfile="{}{}.VV_VV2rh{}_vs_VCTVrh{}.T_{}_errors_bins20_cum_freq"
                    "_slides.png".format(plot_fold, basename, rhVV, rhVCTV, i),
            num_bins=20, freq=True, cumulative=True,
            value_range=value_range,
            y_range=y_range  # , max_val=1
        )
        data = [VCTV_kappa_errors, VV_kappa_errors,
                VV_area2_kappa_errors, VTK_kappa_errors]
        if value_range_kappa is None:
            value_range = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["VCTV RadiusHit={}".format(rhVCTV),
                    "VV RadiusHit={}".format(rhVV),
                    "VV2 RadiusHit={}".format(rhVV), "VTK"],
            line_styles=['-', '--', '-.', ':'], markers=['^', 'v', 'o', 's'],
            colors=['b', 'c', 'orange', 'r'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal curvature relative error".format(
                principal_components[i].capitalize()),
            ylabel="Cumulative frequency",
            outfile=("{}{}.VV_VV2rh{}_VCTVrh{}_vs_VTK."
                     "kappa_{}_errors_bins20_cum_freq_slides.png".format(
                      plot_fold, basename, rhVV, rhVCTV, i)),
            num_bins=20, freq=True, cumulative=True,
            value_range=value_range,
            y_range=y_range  # , max_val=1
        )


if __name__ == "__main__":
    # plot_plane_normals(y_range=(-0.05, 1.05))
    # plot_sphere_kappa_1_and_2_fitting_diff_num_points()
    # plot_sphere_kappa_1_and_2_diff_rh(ico=0)
    # plot_sphere_kappa_1_and_2_errors()
    # plot_inverse_sphere_kappa_1_and_2_errors()
    # plot_cylinder_kappa_1_diff_rh()
    # plot_cylinder_T_2_and_kappa_1_errors_VV_VVarea2_VCTV(
    #     y_range=(-0.05, 1.05), exclude_borders=0)
    # plot_cylinder_T_2_and_kappa_1_errors_VV_VCTV(
    #     y_range=(-0.05, 1.05), value_range_T=(0, 0.006),
    #     value_range_kappa=(0, 1.0), exclude_borders=0, area2=True)
    # plot_inverse_cylinder_T_1_and_kappa_2_errors()
    # plot_torus_kappa_1_and_2_diff_rh()
    # plot_torus_kappa_1_and_2_T_1_and_2_errors_noVVCF(y_range=(-0.05, 1.05),
    #                                                  area2=True)
    # plot_torus_kappa_1_and_2_T_1_and_2_errors_VTK_VV_VVarea2_VCTV(
    #     y_range=(-0.05, 1.05))

    # gaussian sphere
    # for r in [10, 20, 30]:
    #     plot_sphere_kappa_1_and_2_errors_noVVCF(
    #         r=r, rhVV=9, rhVCTV=9, ico=0, binary=False, y_range=(-0.05, 1.05))
        #     r=r, rhVV=9, rhVCTV=9, ico=0, binary=False, value_range=(0, 0.178),
        #     y_range=(-0.05, 1.05))
        # plot_sphere_kappa_1_and_2_errors_VV_VCTV(
        #     r=r, rhVV=9, rhVCTV=9, ico=0, binary=False, value_range=(0, 0.05),
        #     y_range=(-0.05, 1.05), area2=True)

    # binary sphere
    plot_sphere_kappa_1_and_2_diff_rh(
        r=10, binary=True, methods=["VV_area2"], rhs=range(8, 13))  # "VCTV", "VV"
    # for r in [10, 20, 30]:
    #     # plot_sphere_kappa_1_and_2_errors_noVVCF(
    #     #     r=r, rhVV=9, rhVCTV=8, ico=0, binary=True, y_range=(-0.05, 1.05))
    #     plot_sphere_kappa_1_and_2_errors_VV_VCTV(
    #         r=r, rhVV=9, rhVCTV=8, ico=0, binary=True, y_range=(-0.05, 1.05),
    #         value_range=(0, 0.65), area2=True)
    # plot_sphere_kappa_1_and_2_errors_VV_VCTV(
    #     r=20, rhVV=18, rhVCTV=18, ico=0, binary=True, value_range=(0, 0.65),
    #     y_range=(-0.05, 1.05), area2=True)
    # plot_sphere_kappa_1_and_2_errors_VV_VCTV(
    #     r=30, rhVV=28, rhVCTV=28, ico=0, binary=True, value_range=(0, 0.65),
    #     y_range=(-0.05, 1.05), area2=True)
