import numpy as np
import os
import pandas as pd

from pysurf import pexceptions

import matplotlib.pyplot as plt
plt.style.use('presentation')  # print(plt.style.available)

# import seaborn as sns
# sns.set_context("poster", font_scale=2.2)
# sns.set_style("ticks")

from matplotlib import rcParams
# rcParams["figure.figsize"] = (12, 10)
rcParams['mathtext.default'] = 'regular'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams["axes.linewidth"] = 2
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 2

"""
Functions for plotting estimation errors of curvature estimation methods using
"synthetic" benchmark surfaces.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'

FOLD = '/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces_benchmarking/'
LINEWIDTH = 4


def plot_hist(values, num_bins, title, xlabel="Value", ylabel="Frequency",
              x_range=None, outfile=None):
    """
    Plots a histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Frequency")
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    if x_range is None:
        plt.hist(values, bins=num_bins)
    elif isinstance(x_range, tuple) and len(x_range) == 2:
        plt.hist(values, bins=num_bins, range=x_range)
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


def plot_line_hist(values, num_bins, title, xlabel="Value", ylabel="Frequency",
                   x_range=None, label=None, max_val=None, outfile=None):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Frequency")
        x_range (tuple, optional): a tuple of two values to limit the range
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
    if x_range is None:
        counts, bin_edges = np.histogram(values, bins=num_bins)
    elif isinstance(x_range, tuple) and len(x_range) == 2:
        counts, bin_edges = np.histogram(values, bins=num_bins,
                                         range=x_range)
    else:
        raise pexceptions.PySegInputError(
            expr='plot_hist',
            msg="Range has to be a tuple of two numbers (min, max).")
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bincenters, counts, ls='-', marker='^', c="b", label=label,
             linewidth=LINEWIDTH, clip_on=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)


def add_line_hist(values, num_bins, x_range=None, max_val=None,
                  label=None, ls='-', marker='^', c='b', freq=False,
                  cumulative=False):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
        num_bins (int): number of bins for the histogram
        x_range (tuple, optional): a tuple of two values to limit the range
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
    if x_range is None:
        counts, bin_edges = np.histogram(values, bins=num_bins)
    elif isinstance(x_range, tuple) and len(x_range) == 2:
        counts, bin_edges = np.histogram(values, bins=num_bins,
                                         range=x_range)
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
             linewidth=LINEWIDTH, clip_on=False)


def plot_composite_line_hist(
        labels, line_styles, markers, colors,
        xlabel, ylabel, title=None,
        data_arrays=None, data_files=None,
        num_bins=20, x_range=None, y_range=None, max_val=None, freq=False,
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
        x_range (tuple, optional): a tuple of two values to limit the range
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
    fig, ax = plt.subplots()

    if data_files is not None:
        for i, data_file in enumerate(data_files):
            # Reading in the error values from files:
            if not os.path.exists(data_file):
                print("File {} not found!".format(data_file))
                exit(0)
            errors = np.loadtxt(data_file)
            add_line_hist(
                errors, num_bins, x_range=x_range, max_val=max_val,
                label=labels[i], ls=line_styles[i], marker=markers[i],
                c=colors[i], freq=freq, cumulative=cumulative)
    elif data_arrays is not None:
        for i, data_array in enumerate(data_arrays):
            add_line_hist(
                data_array, num_bins, x_range=x_range, max_val=max_val,
                label=labels[i], ls=line_styles[i], marker=markers[i],
                c=colors[i], freq=freq, cumulative=cumulative)
    if title is not None:
        ax.set_title(title)
        ttl = ax.title
        ttl.set_position([.5, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_range is not None:
        plt.ylim(y_range)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)  # frameon=False
    plt.grid(True)
    plt.tight_layout()
    plt.tick_params(top='off', right='off', which='both')  # sns.despine()
    plt.tick_params(direction='in')
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)
        print("The plot was saved as {}".format(outfile))


def plot_plane_normals(n=10, y_range=(0, 1), res=20):
    """ Plots estimated normals errors by VV versus original face normals
    (calculated by VTK) on a noisy plane surface.

    Args:
        n (int, optional): noise in % (default 10)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        res (int, optional): defines the size of the square plane in pixels and
            triangle division: 2*res
    """
    fold = ("{}plane/res{}_noise{}/files4plotting/".format(
        FOLD, res, n))
    plot_fold = ("{}plane/res{}_noise{}/plots/".format(
        FOLD, res, n))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "plane_half_size{}".format(res)
    SSVV_rh8_normal_errors = pd.read_csv("{}{}.SSVV_rh8.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    SSVV_rh4_normal_errors = pd.read_csv("{}{}.SSVV_rh4.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    VTK_normal_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                    sep=';')["normalErrors"].tolist()
    data = [SSVV_rh8_normal_errors, SSVV_rh4_normal_errors, VTK_normal_errors]
    print([max(d) for d in data])
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VV RadiusHit=8", "VV RadiusHit=4", "VTK"],
        line_styles=['-', '--', ':'], markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Plane ({}% noise)".format(n),
        xlabel="Normal orientation error",
        ylabel="Cumulative relative frequency",
        outfile="{}plane_res{}_noise{}.VV_vs_VTK.normal_errors.png".format(
            plot_fold, res, n),
        num_bins=20, freq=True, cumulative=True,
        x_range=(0, max([max(d) for d in data])), y_range=y_range
    )


def plot_cylinder_kappa_1_diff_rh(n=0):
    """Plots estimated kappa_1 values histograms on a cylinder surface by
    different methods (AVV and SSVV) using different RadiusHit.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("{}cylinder/noise0/files4plotting/".format(FOLD, n))
    plot_fold = ("{}cylinder/noise0/plots/".format(FOLD, n))
    basename = "cylinder_r10_h25_eb0"
    for method in ['SSVV', 'AVV']:
        kappa_arrays = []
        labels = []
        for rh in range(5, 10):
            kappa_array = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')["kappa1"].tolist()
            kappa_arrays.append(kappa_array)
            label = "RadiusHit={}".format(rh)
            labels.append(label)
        plot_composite_line_hist(
            data_arrays=kappa_arrays,
            labels=labels,
            line_styles=[':', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on cylinder ({}% noise)".format(method, n),
            xlabel=r"$\kappa_1$",
            ylabel="Frequency",
            outfile=("{}{}_noise{}.{}_rh5-9.kappa_1.png".format(
                plot_fold, basename, n, method)),
            num_bins=5, x_range=None, max_val=None, freq=False
        )


def plot_cylinder_T_2_and_kappa_1_errors(
        n=0, y_range=(0, 1), x_range_T=None, x_range_kappa=None,
        RorAVV="AVV", exclude_borders=5):
    """Plots estimated kappa_2 and T_1 errors histograms on a cylinder surface
    for different methods (RVV or AVV and SSVV) and optimal RadiusHit for each
    method.

    Args:
        n (int, optional): noise in % (default 0)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        x_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        x_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        RorAVV (str, optional): RVV or AVV (default)
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        exclude_borders (int, optional): how many voxels from border were
            excluded for curvature calculation (default 5)
    """
    fold = ("{}cylinder/noise0/files4plotting/".format(FOLD, n))
    plot_fold = ("{}cylinder/noise0/plots/".format(FOLD, n))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "cylinder_r10_h25_eb{}".format(exclude_borders)
    df = pd.read_csv("{}{}.SSVV_rh7.csv".format(fold, basename), sep=';')
    SSVV_T_2_errors = df["T2Errors"].tolist()
    SSVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    if RorAVV == "AVV":
        line_styles = ['-', '-.']
        markers = ['^', 'o']
        colors = ['b', 'orange']
    else:
        line_styles = ['-', '--']
        markers = ['^', 'v']
        colors = ['b', 'c']
    df = pd.read_csv("{}{}.{}_rh5.csv".format(fold, basename, RorAVV),
                     sep=';')
    VV_T_2_errors = df["T2Errors"].tolist()
    VV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VTK_kappa_1_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa1RelErrors"].tolist()
    data = [SSVV_T_2_errors, VV_T_2_errors]
    if x_range_T is None:
        x_range_T = (0, max([max(d) for d in data]))
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV RadiusHit=7", "{} RadiusHit=5".format(RorAVV)],
        line_styles=line_styles, markers=markers, colors=colors,
        title="Cylinder ({}% noise)".format(n),
        xlabel=r"$T_2\ error$",
        ylabel="Cumulative relative frequency",
        outfile="{}{}_noise{}.{}_rh5_SSVV_rh7.T_2_errors.png".format(
            plot_fold, basename, n, RorAVV),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=x_range_T,
        y_range=y_range
    )
    data = [SSVV_kappa_1_errors,
            VV_kappa_1_errors, VTK_kappa_1_errors]
    line_styles.append(':')
    markers.append('s')
    colors.append('r')
    if x_range_kappa is None:
        x_range_kappa = (0, max([max(d) for d in data]))
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV RadiusHit=7", "{} RadiusHit=5".format(RorAVV), "VTK"],
        line_styles=line_styles, markers=markers, colors=colors,
        title="Cylinder ({}% noise)".format(n),
        xlabel=r"$\kappa_1\ relative\ error$",
        ylabel="Cumulative relative frequency",
        outfile=("{}{}_noise{}.{}_rh5_SSVV_rh7_vs_VTK.kappa_1_errors.png"
                 .format(plot_fold, basename, n, RorAVV)),
        num_bins=20, freq=True, cumulative=True,
        x_range=x_range_kappa, y_range=y_range
    )


def plot_cylinder_T_2_and_kappa_1_errors_allVV(n=0, y_range=(0, 1),
                                               exclude_borders=5):
    """Plots estimated kappa_2 and T_1 errors histograms on a cylinder surface
    for different methods (RVV, AVV and SSVV) and optimal RadiusHit for each
    method.

    Args:
        n (int, optional): noise in % (default 0)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        exclude_borders (int, optional): how many voxels from border were
            excluded for curvature calculation (default 5)
    """
    fold = ("{}cylinder/noise0/files4plotting/".format(FOLD, n))
    plot_fold = ("{}cylinder/noise0/plots/".format(FOLD, n))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "cylinder_r10_h25_eb{}".format(exclude_borders)
    df = pd.read_csv("{}{}.SSVV_rh7.csv".format(fold, basename), sep=';')
    SSVV_T_2_errors = df["T2Errors"].tolist()
    SSVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.RVV_rh5.csv".format(fold, basename), sep=';')
    RVV_T_2_errors = df["T2Errors"].tolist()
    RVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.AVV_rh5.csv".format(fold, basename), sep=';')
    AVV_T_2_errors = df["T2Errors"].tolist()
    AVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VTK_kappa_1_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa1RelErrors"].tolist()
    data = [SSVV_T_2_errors, RVV_T_2_errors, AVV_T_2_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=7", "RVV rh=5", "AVV rh=5"],
        line_styles=['-', '--', '-.'], markers=['^', 'v', 'o'],
        colors=['b', 'c', 'g'],
        title="Cylinder ({}% noise)".format(n),
        xlabel=r"$T_2\ error$",
        ylabel="Cumulative relative frequency",
        outfile="{}{}_noise{}.RVV_AVV_rh5_SSVV_rh7.T_2_errors.png".format(
            plot_fold, basename, n),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=(0, 0.005),  # (0, max([max(d) for d in data]))
        y_range=y_range
    )
    data = [SSVV_kappa_1_errors, RVV_kappa_1_errors, AVV_kappa_1_errors,
            VTK_kappa_1_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=7", "RVV rh=5", "AVV rh=5", "VTK"],
        line_styles=['-', '--', '-.', ':'], markers=['^', 'v', 'o', 's'],
        colors=['b', 'c', 'g', 'r'],
        title="Cylinder ({}% noise)".format(n),
        xlabel=r"$\kappa_1\ relative\ error$",
        ylabel="Cumulative relative frequency",
        outfile=("{}{}_noise{}.RVV_AVV_rh5_SSVV_rh7_vs_VTK.kappa_1_errors.png"
                 .format(plot_fold, basename, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=(0, max([max(d) for d in data])), y_range=y_range
    )


def plot_inverse_cylinder_T_1_and_kappa_2_errors(n=0):
    """Plots estimated kappa_2 and T_1 errors histograms on an inverse cylinder
    surface for different methods (RVV and SSVV) and optimal RadiusHit for
    each method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("{}cylinder/noise0/files4plotting/".format(FOLD, n))
    plot_fold = ("{}cylinder/noise0/plots/".format(FOLD, n))
    basename = "inverse_cylinder_r10_h25"
    df = pd.read_csv("{}{}.SSVV_rh8.csv".format(fold, basename), sep=';')
    SSVV_T_1_errors = df["T1Errors"].tolist()
    SSVV_kappa_2_errors = df["kappa2RelErrors"].tolist()

    df = pd.read_csv("{}{}.RVV_rh8.csv".format(fold, basename), sep=';')
    VV_T_1_errors = df["T1Errors"].tolist()
    VV_kappa_2_errors = df["kappa2RelErrors"].tolist()

    VTK_kappa_2_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa2RelErrors"].tolist()
    data = [SSVV_T_1_errors,
            VV_T_1_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=8", "RVV rh=8"],
        line_styles=['-', '--'], markers=['^', 'v'],
        colors=['b', 'c'],
        title="Inverse cylinder ({}% noise)".format(n),
        xlabel="T_1\ error",
        ylabel="Cumulative relative frequency",
        outfile="{}{}_noise{}.RVV_SSVV_rh8.T_1_errors.png".format(
            plot_fold, basename, n),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=(0, max([max(d) for d in data]))
    )
    data = [SSVV_kappa_2_errors,
            VV_kappa_2_errors,
            VTK_kappa_2_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=8", "RVV rh=8", "VTK"],
        line_styles=['-', '--', ':'], markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Inverse cylinder ({}% noise)".format(n),
        xlabel=r"$\kappa_2\ relative\ error$",
        ylabel="Cumulative relative frequency",
        outfile=("{}{}_noise{}.RVV_SSVV_rh8_vs_VTK.kappa_2_errors.png".format(
            plot_fold, basename, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=(0, max([max(d) for d in data]))
    )


def plot_sphere_kappa_1_and_2_diff_rh(
        r=10, n=0, ico=0, voxel=False, methods=["RVV", "AVV", "SSVV"],
        rhs=range(5, 10)):
    """Plots estimated kappa_1 and kappa_2 values for a sphere surface
     by different methods (RVV, AVV and SSVV) using different RadiusHit.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (e.g. 1280), icosahedron results with so
            many faces are used; if 0 (default), smooth sphere results are used
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used (ignoring the other options)
        methods (list, optional): tells which method(s) should be used: 'RVV' or
            'AVV' for normal vector voting (default) or 'SSVV' for vector and
            curvature tensor voting to estimate the principal directions and
            curvatures
        rhs (list, optional): wanted RadiusHit parameter values (default 5-9)
    """
    if voxel:
        subfolds = "sphere/voxel/"
        type = "voxel sphere, radius={}".format(r)
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
        type = "icosahedron {} sphere, radius={}, {}% noise".format(ico, r, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
        type = "smooth sphere, radius={}, {}% noise".format(r, n)
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
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
        plot_composite_line_hist(  # kappa_1
            data_arrays=kappa_1_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on {}".format(method, type),
            xlabel=r"$\kappa_1$",
            ylabel="Frequency",
            outfile=("{}{}.{}_rh{}-{}.kappa_1.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=5, x_range=None, max_val=None, freq=False
        )
        plot_composite_line_hist(  # kappa_2
            data_arrays=kappa_2_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on {}".format(method, type),
            xlabel=r"$\kappa_2$",
            ylabel="Frequency",
            outfile=("{}{}.{}_rh{}-{}.kappa_2.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=5, x_range=None, max_val=None, freq=False
        )
        plot_composite_line_hist(  # kappa_1 + kappa_2
            data_arrays=kappas_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on {}".format(method, type),
            xlabel=r"$\kappa_1\ and\ \kappa_2$",
            ylabel="Frequency",
            outfile=("{}{}.{}_rh{}-{}.kappa_1_and_2.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=5, x_range=None, max_val=None, freq=False
        )


def plot_sphere_kappa_1_and_2_errors(
        r=10, rhVV=8, rhSSVV=8, n=0, ico=0, voxel=False, y_range=(0, 1)):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (RVV, AVV, SSVV and VTK) and an optimal RadiusHit for
    each method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 8)
        rhSSVV (int, optional): radius_hit for SSVV (default 8)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (i.e. 1280), icosahedron results with so
            many faces are used; if 0 (default), smooth sphere results are used
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used (ignoring the other options)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
    """
    if voxel:
        subfolds = "sphere/voxel/"
        type = "Voxel sphere (radius={})".format(r)
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
        type = "Icosahedron {} sphere (radius={}, {}% noise)".format(ico, r, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
        type = "Smooth sphere (radius={}, {}% noise)".format(r, n)
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df_SSVV = pd.read_csv("{}{}.SSVV_rh{}.csv".format(fold, basename, rhSSVV),
                          sep=';')
    SSVV_kappa_1_errors = df_SSVV["kappa1RelErrors"].tolist()
    SSVV_kappa_2_errors = df_SSVV["kappa2RelErrors"].tolist()

    df_RVV = pd.read_csv("{}{}.RVV_rh{}.csv".format(fold, basename, rhVV),
                        sep=';')
    RVV_kappa_1_errors = df_RVV["kappa1RelErrors"].tolist()
    RVV_kappa_2_errors = df_RVV["kappa2RelErrors"].tolist()

    df_AVV = pd.read_csv("{}{}.AVV_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
    AVV_kappa_1_errors = df_AVV["kappa1RelErrors"].tolist()
    AVV_kappa_2_errors = df_AVV["kappa2RelErrors"].tolist()

    df_VTK = pd.read_csv("{}{}.VTK.csv".format(fold, basename), sep=';')
    VTK_kappa_1_errors = df_VTK["kappa1RelErrors"].tolist()
    VTK_kappa_2_errors = df_VTK["kappa2RelErrors"].tolist()

    data = [SSVV_kappa_1_errors + SSVV_kappa_2_errors,
            AVV_kappa_1_errors + AVV_kappa_2_errors,
            RVV_kappa_1_errors + RVV_kappa_2_errors,
            VTK_kappa_1_errors + VTK_kappa_2_errors]
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["SSVV RadiusHit={}".format(rhSSVV),
                "AVV RadiusHit={}".format(rhVV),
                "RVV RadiusHit={}".format(rhVV), "VTK"],
        line_styles=['-', '-.', '--', ':'],
        markers=['^', 'o', 'v', 's'],
        colors=['b', 'orange', 'c', 'r'],
        title=type,
        xlabel=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        ylabel="Cumulative relative frequency",
        outfile=("{}{}.RVV_AVVrh{}_SSVVrh{}_vs_VTK."
                 "kappa_1_and_2_errors.png".format(
                    plot_fold, basename, rhVV, rhSSVV)),
        num_bins=20, freq=True, cumulative=True,
        x_range=(0, max([max(d) for d in data])), y_range=y_range
    )


def plot_sphere_kappa_1_and_2_errors_noVTK(
        r=10, rhVV=8, rhSSVV=8, n=0, ico=0, voxel=False, x_range=None,
        y_range=(0, 1), RorAVV="AVV"):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (RVV, AVV and SSVV) and an optimal RadiusHit for each
    method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 8)
        rhSSVV (int, optional): radius_hit for SSVV (default 8)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (e.g. 1280), icosahedron results with so
            many faces are used; if 0 (default), smooth sphere results are used
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used (ignoring the other options)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        RorAVV (str, optional): RVV or AVV (default)
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
    """
    if voxel:
        subfolds = "sphere/voxel/"
        type = "Voxel sphere (radius={})".format(r)
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
        type = "Icosahedron {} sphere (radius={}, {}% noise)".format(ico, r, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
        type = "Smooth sphere (radius={}, {}% noise)".format(r, n)
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df_SSVV = pd.read_csv("{}{}.SSVV_rh{}.csv".format(fold, basename, rhSSVV),
                          sep=';')
    SSVV_kappa_1_errors = df_SSVV["kappa1RelErrors"].tolist()
    SSVV_kappa_2_errors = df_SSVV["kappa2RelErrors"].tolist()

    if RorAVV == 'AVV':
        line_styles = ['-', '-.']
        markers = ['^', 'o']
        colors = ['b', 'orange']
    else:
        line_styles = ['-', '--']
        markers = ['^', 'v']
        colors = ['b', 'c']
    df_VV = pd.read_csv("{}{}.{}_rh{}.csv".format(fold, basename, RorAVV, rhVV),
                        sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    data = [SSVV_kappa_1_errors + SSVV_kappa_2_errors,
            VV_kappa_1_errors + VV_kappa_2_errors]
    if x_range is None:
        x_range = (0, max([max(d) for d in data]))
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["SSVV RadiusHit={}".format(rhSSVV),
                "{} RadiusHit={}".format(RorAVV, rhVV)],
        line_styles=line_styles, markers=markers, colors=colors,
        title=type,
        xlabel=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        ylabel="Cumulative relative frequency",
        outfile=("{}{}.{}rh{}_vs_SSVVrh{}."
                 "kappa_1_and_2_errors_range{}.png".
                 format(plot_fold, basename, RorAVV, rhVV, rhSSVV, x_range[1])),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=x_range, y_range=y_range
    )


def plot_sphere_kappa_1_and_2_errors_noVTK_allVV(
        r=10, rhVV=8, rhSSVV=8, n=0, ico=0, voxel=False, value_range=None,
        y_range=(0, 1)):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (VV and SSVV) and an optimal RadiusHit for each
    method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 8)
        rhSSVV (int, optional): radius_hit for SSVV (default 8)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (e.g. 1280), icosahedron results with so
            many faces are used; if 0 (default), smooth sphere results are used
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used (ignoring the other options)
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
    """
    if voxel:
        subfolds = "sphere/voxel/"
        type = "Voxel sphere (radius={})".format(r)
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
        type = "Icosahedron {} sphere (radius={}, {}% noise)".format(ico, r, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
        type = "Smooth sphere (radius={}, {}% noise)".format(r, n)
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df_SSVV = pd.read_csv("{}{}.SSVV_rh{}.csv".format(fold, basename, rhSSVV),
                          sep=';')
    SSVV_kappa_1_errors = df_SSVV["kappa1RelErrors"].tolist()
    SSVV_kappa_2_errors = df_SSVV["kappa2RelErrors"].tolist()

    df_RVV = pd.read_csv("{}{}.RVV_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
    RVV_kappa_1_errors = df_RVV["kappa1RelErrors"].tolist()
    RVV_kappa_2_errors = df_RVV["kappa2RelErrors"].tolist()

    df_AVV = pd.read_csv("{}{}.AVV_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
    AVV_kappa_1_errors = df_AVV["kappa1RelErrors"].tolist()
    AVV_kappa_2_errors = df_AVV["kappa2RelErrors"].tolist()

    data = [SSVV_kappa_1_errors + SSVV_kappa_2_errors,
            AVV_kappa_1_errors + AVV_kappa_2_errors,
            RVV_kappa_1_errors + RVV_kappa_2_errors]
    if value_range is None:
        value_range = (0, max([max(d) for d in data]))
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["SSVV RadiusHit={}".format(rhSSVV),
                "AVV RadiusHit={}".format(rhVV),
                "RVV RadiusHit={}".format(rhVV)],
        line_styles=['-', '-.', '--'],
        markers=['^', 'o', 'v'],
        colors=['b', 'orange', 'c'],
        title=type,
        xlabel=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        ylabel="Cumulative relative frequency",
        outfile=("{}{}.RVV_AVVrh{}_vs_SSVVrh{}.kappa_1_and_2_errors_range{}.png"
                 .format(plot_fold, basename, rhVV, rhSSVV, value_range[1])),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=value_range, y_range=y_range
    )


def plot_inverse_sphere_kappa_1_and_2_errors(n=0):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on an inverse
    icosahedron sphere surface for different methods (RVV and SSVV) and an
    optimal RadiusHit for each method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("{}sphere/ico1280_noise{}/files4plotting/".format(FOLD, n))
    plot_fold = ("{}sphere/ico1280_noise{}/plots/".format(FOLD, n))
    basename = "inverse_sphere_r10"

    df_SSVV = pd.read_csv("{}{}.SSVV_rh8.csv".format(fold, basename), sep=';')
    SSVV_kappa_1_errors = df_SSVV["kappa1RelErrors"].tolist()
    SSVV_kappa_2_errors = df_SSVV["kappa2RelErrors"].tolist()

    df_VV = pd.read_csv("{}{}.RVV_rh9.csv".format(fold, basename), sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    df_VTK = pd.read_csv("{}{}.VTK.csv".format(fold, basename), sep=';')
    VTK_kappa_1_errors = df_VTK["kappa1RelErrors"].tolist()
    VTK_kappa_2_errors = df_VTK["kappa2RelErrors"].tolist()

    data = [SSVV_kappa_1_errors + SSVV_kappa_2_errors,
            VV_kappa_1_errors + VV_kappa_2_errors,
            VTK_kappa_1_errors + VTK_kappa_2_errors]
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["SSVV rh=8", "RVV rh=9", "VTK"],
        line_styles=['-', '--', ':'],
        markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Inverse sphere (icosahedron 1280, {}% noise)".format(n),
        xlabel=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        ylabel="Cumulative relative frequency",
        outfile=("{}inverse_icosphere_r10_noise{}.RVV_SSVV_vs_VTK."
                 "kappa_1_and_2_errors.png".format(plot_fold, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        x_range=(0, max([max(d) for d in data]))
    )


def plot_torus_kappa_1_and_2_T_1_and_2_errors(
        rhVV=8, rhSSVV=8, x_range_T=None, x_range_kappa=None,
        y_range=(0, 1), RorAVV="AVV"):
    """
    Plots estimated kappa_1 and kappa_2 as well as T_1 and T_2 errors histograms
    on a torus surface for different methods (VV and SSVV) and an optimal
    RadiusHit for each method.

    Args:
        rhVV (int, optional): radius_hit for VV (default 8)
        rhSSVV (int, optional): radius_hit for SSVV (default 8)
        x_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        x_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        RorAVV (str, optional): RVV or AVV (default)
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
    """
    fold = ("{}torus/files4plotting/".format(FOLD))
    plot_fold = ("{}torus/plots/".format(FOLD))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "torus_rr25_csr10"
    principal_components = {1: "maximal", 2: "minimal"}
    for i in principal_components.keys():
        df = pd.read_csv("{}{}.SSVV_rh{}.csv".format(fold, basename, rhSSVV),
                         sep=';')
        SSVV_T_errors = df["T{}Errors".format(i)].tolist()
        SSVV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        df = pd.read_csv("{}{}.{}_rh{}.csv".format(
            fold, basename, RorAVV, rhVV), sep=';')
        VV_T_errors = df["T{}Errors".format(i)].tolist()
        VV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VTK_kappa_errors = pd.read_csv("{}{}.VTK.csv".format(
            fold, basename), sep=';')["kappa{}RelErrors".format(i)].tolist()
        data = [SSVV_T_errors, VV_T_errors]
        if x_range_T is None:
            x_range_T = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["SSVV rh={}".format(rhSSVV), "{} rh={}".format(
                RorAVV, rhVV)],
            line_styles=['-', '--'], markers=['^', 'v'],
            colors=['b', 'c'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal direction error".format(
                principal_components[i]),
            ylabel="Cumulative relative frequency",
            outfile="{}{}.{}rh{}_vs_SSVVrh{}.T_{}_errors.png"
                .format(plot_fold, basename, RorAVV, rhVV, rhSSVV, i),
            num_bins=20, freq=True, cumulative=True,
            x_range=x_range_T,
            y_range=y_range
        )
        data = [SSVV_kappa_errors, VV_kappa_errors, VTK_kappa_errors]
        if x_range_kappa is None:
            x_range_kappa = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["SSVV rh={}".format(rhSSVV), "{} rh={}".format(
                RorAVV, rhVV), "VTK"],
            line_styles=['-', '--', ':'], markers=['^', 'v', 's'],
            colors=['b', 'c', 'r'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal curvature relative error".format(
                principal_components[i]),
            ylabel="Cumulative relative frequency",
            outfile=("{}{}.{}rh{}_SSVVrh{}_vs_VTK.kappa_{}_errors.png".format(
                        plot_fold, basename, RorAVV, rhVV, rhSSVV, i)),
            num_bins=20, freq=True, cumulative=True,
            x_range=x_range_kappa,
            y_range=y_range
        )


def plot_torus_kappa_1_and_2_T_1_and_2_errors_allVV(
        rhVV=8, rhSSVV=8, x_range_T=None, x_range_kappa=None,
        y_range=(0, 1)):
    """
    Plots estimated kappa_1 and kappa_2 as well as T_1 and T_2 errors histograms
    on a torus surface for different methods (RVV, AVV and SSVV) and an
    optimal RadiusHit for each method.

    Args:
        rhVV (int, optional): radius_hit for VV (default 8)
        rhSSVV (int, optional): radius_hit for SSVV (default 8)
        x_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        x_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
    """
    fold = ("{}torus/files4plotting/".format(FOLD))
    plot_fold = ("{}torus/plots/".format(FOLD))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "torus_rr25_csr10"
    for i in [1, 2]:
        df = pd.read_csv("{}{}.SSVV_rh{}.csv".format(fold, basename, rhSSVV),
                         sep=';')
        SSVV_T_errors = df["T{}Errors".format(i)].tolist()
        SSVV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        df = pd.read_csv("{}{}.RVV_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
        RVV_T_errors = df["T{}Errors".format(i)].tolist()
        RVV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        df = pd.read_csv("{}{}.AVV_rh{}.csv".format(fold, basename, rhVV),
                         sep=';')
        AVV_T_errors = df["T{}Errors".format(i)].tolist()
        AVV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VTK_kappa_errors = pd.read_csv("{}{}.VTK.csv".format(
            fold, basename), sep=';')["kappa{}RelErrors".format(i)].tolist()

        data = [SSVV_T_errors, AVV_T_errors, RVV_T_errors]
        if x_range_T is None:
            x_range = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["SSVV RadiusHit={}".format(rhSSVV),
                    "AVV RadiusHit={}".format(rhVV),
                    "RVV RadiusHit={}".format(rhVV)],
            line_styles=['-', '-.', '--'], markers=['^', 'o', 'v'],
            colors=['b', 'orange', 'c'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel=r"$T_{}\ error$".format(i),
            ylabel="Cumulative relative frequency",
            outfile="{}{}.RVV_AVVrh{}_vs_SSVVrh{}.T_{}_errors.png".format(
                plot_fold, basename, rhVV, rhSSVV, i),
            num_bins=20, freq=True, cumulative=True,
            x_range=x_range,
            y_range=y_range
        )
        data = [VTK_kappa_errors, SSVV_kappa_errors,
                AVV_kappa_errors, RVV_kappa_errors]
        if x_range_kappa is None:
            x_range = (0, max([max(d) for d in data]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=["VTK", "SSVV RadiusHit={}".format(rhSSVV),
                    "AVV RadiusHit={}".format(rhVV),
                    "RVV RadiusHit={}".format(rhVV)],
            line_styles=[':', '-', '-.', '--'], markers=['s', '^', 'o', 'v'],
            colors=['r', 'b', 'orange', 'c'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel=r"$\kappa_{}\ relative\ error$".format(i),
            ylabel="Cumulative relative frequency",
            outfile=("{}{}.RVV_AVVrh{}_SSVVrh{}_vs_VTK.kappa_{}_errors.png"
                     .format(plot_fold, basename, rhVV, rhSSVV, i)),
            num_bins=20, freq=True, cumulative=True,
            x_range=x_range,
            y_range=y_range
        )


if __name__ == "__main__":
    # plot_plane_normals()
    # plot_inverse_sphere_kappa_1_and_2_errors()  # not used
    plot_cylinder_kappa_1_diff_rh()
    # plot_cylinder_T_2_and_kappa_1_errors(
    #     x_range_T=(0, 0.006), x_range_kappa=(0, 1.0), exclude_borders=5)
    # plot_cylinder_T_2_and_kappa_1_errors(
    #     x_range_T=(0, 0.006), x_range_kappa=(0, 1.0), exclude_borders=0)
    # plot_inverse_cylinder_T_1_and_kappa_2_errors()  # not used
    # plot_torus_kappa_1_and_2_diff_rh()  # not implemented
    # plot_torus_kappa_1_and_2_T_1_and_2_errors_allVV()

    # smooth sphere
    # plot_sphere_kappa_1_and_2_diff_rh(
    #     r=10, methods=["RVV", "AVV", "SSVV"], rhs=range(5, 10))
    # for r in [10, 20]:
    #     plot_sphere_kappa_1_and_2_errors(
    #         r=r, rhVV=9, rhSSVV=9, voxel=False)
    #     plot_sphere_kappa_1_and_2_errors_noVTK_allVV(
    #         r=r, rhVV=9, rhSSVV=9, voxel=False)

    # voxel sphere
    # plot_sphere_kappa_1_and_2_diff_rh(
    #     r=10, voxel=True, methods=["RVV", "AVV", "SSVV"], rhs=range(5, 10))
    # for r in [10, 20, 30]:
    #     plot_sphere_kappa_1_and_2_errors_noVTK(
    #         r=r, rhVV=9, rhSSVV=8, voxel=True, x_range=(0, 0.65))
    # plot_sphere_kappa_1_and_2_errors_noVTK(
    #     r=20, rhVV=18, rhSSVV=18, voxel=True, x_range=(0, 0.65))
    # plot_sphere_kappa_1_and_2_errors_noVTK(
    #     r=30, rhVV=28, rhSSVV=28, voxel=True, x_range=(0, 0.65))
