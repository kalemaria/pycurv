import numpy as np
import os
from os.path import join
import pandas as pd
from pathlib import PurePath
from itertools import cycle

from pycurv import pexceptions
from pycurv_testing import calculate_histogram_area

import matplotlib.pyplot as plt
plt.style.use('classic')
plt.style.use('presentation')  # print(plt.style.available)

from matplotlib import rcParams
rcParams['mathtext.default'] = 'regular'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams["axes.linewidth"] = 2
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 2

"""
Functions for plotting estimation errors of all the tested curvature estimation
algorithms using "synthetic" benchmark surfaces.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'

FOLDPEAKS = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/TCB/" \
            "180830_TITAN_l2_t2peak/"
FOLDCER = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/TCB/" \
          "180830_TITAN_l2_t2half/"
FOLDPEAKSPLOTS = '/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/' \
                 'plots_peaks/'
FOLD = '/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/' \
       'comparison_to_others/test_vector_voting_output/'
FOLDMB = "/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/" \
         "comparison_to_others/test_mindboggle_output/"
FOLDFS = "/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/" \
         "comparison_to_others/test_freesurfer_output/"
FOLDPLOTS = "/fs/pool/pool-ruben/Maria/workspace/github/my_tests_output/" \
            "comparison_to_others/plots/"
LINEWIDTH = 4
MARKERS = ['*', 'v', '^', 's', 'o', 'v', '^', 's', 'o', '*']
COLORS = ['b', 'c', 'g', 'y', 'r', 'b', 'c', 'g', 'y', 'r']


def plot_hist(values, num_bins, title, x_label="Value", y_label="Frequency",
              x_range=None, outfile=None):
    """
    Plots a histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        x_label (str, optional): X axis label (default "Value")
        y_label (str, optional): Y axis label (default "Frequency")
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)


def plot_line_hist(values, weights=None, num_bins=20, title=None,
                   x_label="Value", y_label="Frequency", x_range=None,
                   label=None, ls='-', marker='^', c='b', max_val=None,
                   normalize=False, cumulative=False, outfile=None):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        values: a list of numerical values
        weights (numpy.ndarray, optional): if given, values will be weighted
        num_bins (int): number of bins for the histogram (default 10)
        title (str, optional): title of the plot
        x_label (str, optional): X axis label (default "Value")
        y_label (str, optional): Y axis label (default "Frequency")
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        label (str, optional): legend label for the value list (default None)
        ls (str, optional): line style (default '-')
        marker (str, optional): plotting character (default '^')
        c (str, optional): color (default 'b' for blue)
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value
        normalize (boolean, optional): if True (default False), relative
            frequencies instead of frequencies will be plotted
        cumulative (boolean, optional): if True (default False), cumulative
            counts or frequencies will be plotted
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    if max_val is not None:
        values = [max_val if val > max_val else val for val in values]

    params = {}
    if x_range is not None:
        if isinstance(x_range, tuple) and len(x_range) == 2:
            params["range"] = x_range
        else:
            raise pexceptions.PySegInputError(
                expr='plot_hist',
                msg="Range has to be a tuple of two numbers (min, max).")
    if weights is not None:
        params["weights"] = weights
    counts, bin_edges = np.histogram(values, bins=num_bins, **params)

    if normalize is True:
        if weights is None:
            counts = counts / float(len(values))  # normalized to max 1
        else:
            counts = counts / sum(weights)
    if cumulative is True:
        counts = np.cumsum(counts)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bincenters, counts, ls=ls, marker=marker, c=c, label=label,
             linewidth=LINEWIDTH, clip_on=False)
    if title is not None:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if label is not None:
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.grid(True)
    plt.tight_layout()
    # plt.tick_params(top='off', right='off', which='both')  # stopped to work
    # Only show ticks on the left and bottom spines
    plt.yaxis.set_ticks_position('left')
    plt.xaxis.set_ticks_position('bottom')

    plt.tick_params(direction='in')
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)
        print("The plot was saved as {}".format(outfile))


def add_line_hist(values, weights=None, num_bins=20, x_range=None, max_val=None,
                  label=None, ls='-', marker='^', c='b', normalize=False,
                  cumulative=False, zorder=None):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        values (numpy.ndarray): a list of numerical values
        weights (numpy.ndarray, optional): if given, values will be weighted
        num_bins (int, optional): number of bins for the histogram (default 20)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value
        label (str, optional): legend label for the value list (default None)
        ls (str, optional): line style (default '-')
        marker (str, optional): plotting character (default '^')
        c (str, optional): color (default 'b' for blue)
        normalize (boolean, optional): if True (default False), relative
            frequencies instead of frequencies will be plotted
        cumulative (boolean, optional): if True (default False), cumulative
            counts or frequencies will be plotted
        zorder (int, optional): integer indicating the order of the data line
            on the plot

    Returns:
        hist_area, if cumulative, else None
    """
    if max_val is not None:
        values = [max_val if val > max_val else val for val in values]

    params = {}
    if x_range is not None:
        if isinstance(x_range, tuple) and len(x_range) == 2:
            params["range"] = x_range
        else:
            raise pexceptions.PySegInputError(
                expr='plot_hist',
                msg="Range has to be a tuple of two numbers (min, max).")
    if weights is not None:
        params["weights"] = weights
    counts, bin_edges = np.histogram(values, bins=num_bins, **params)

    if normalize:
        if weights is None:
            counts = counts / float(len(values))  # normalized to max 1
        else:
            counts = counts / sum(weights)
    hist_area = None
    if cumulative:
        counts = np.cumsum(counts)
        hist_area = calculate_histogram_area(counts, bin_edges)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bincenters, counts, ls=ls, marker=marker, c=c, label=label,
             linewidth=LINEWIDTH, clip_on=False, zorder=zorder)
    return hist_area


def plot_composite_line_hist(
        labels, line_styles, markers, colors,
        x_label, y_label, title=None,
        data_arrays=None, data_files=None, weights_arrays=None,
        num_bins=20, x_range=None, y_range=None, max_val=None,
        normalize=False, cumulative=False, outfile=None, legend_loc='best',
        num_x_values=0, zorders=None, fontsize=30, ncol=1):
    """
    Plots several data sets as line histograms in one plot.
    Args:
        labels: list of legend labels (str) for the data sets
        line_styles: list of line styles (str)
        markers: list of plotting characters (str)
        colors: list of colors (str)
        x_label (str): X axis label
        y_label (str): Y axis label
        title (str, optional): title of the plot
        data_arrays (list, optional): list of data arrays
        data_files (list, optional): list of data file names (str)
        weights_arrays (list, optional): if given, data will be weighted,
            should be in the same format as the data arrays (default None)
        num_bins (int, optional): number of bins for the histogram (default 20)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value
        normalize (boolean, optional): if True (default False), relative
            frequencies instead of frequencies will be plotted
        cumulative (boolean, optional): if True (default False), cumulative
            counts or frequencies will be plotted
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path
        legend_loc (str, optional): legend location (default 'best')
        num_x_values (int, optional): if > 0 (default 0), plot this number of
            ticks on X axis
        zorders (list, optional): list of integers indicating the order of data
            lines on the plot
        fontsize (int, optional): fontsize (default 30)
        ncol (int, optional): number of legend columns (default 1)

    Returns:
        hist_areas: list of histogram areas, values are Null in not cumulative
    Note:
        either data_arrays or data_files has to be given

    """
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    hist_areas = []
    if data_files is not None:
        data_arrays = []
        for i, data_file in enumerate(data_files):
            # Reading in the error values from files:
            if not os.path.exists(data_file):
                print("File {} not found!".format(data_file))
                exit(0)
            errors = np.loadtxt(data_file)
            data_arrays.append(errors)

    for i, data_array in enumerate(data_arrays):
        if weights_arrays is None:
            weights_array = None
        else:
            weights_array = weights_arrays[i]
        if zorders is None:
            zorder = None
        else:
            zorder = zorders[i]
        hist_area = add_line_hist(
            data_array, weights=weights_array, num_bins=num_bins,
            x_range=x_range, max_val=max_val,
            label=labels[i], ls=line_styles[i], marker=markers[i],
            c=colors[i], normalize=normalize, cumulative=cumulative,
            zorder=zorder)
        hist_areas.append(hist_area)
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=fontsize)
        ttl = ax.title
        ttl.set_position([.5, 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_range is not None:
        plt.ylim(y_range)
    plt.legend(loc=legend_loc, fancybox=True, framealpha=0.5, fontsize=18,
               ncol=ncol, columnspacing=1, handletextpad=0.2, borderpad=0.2)
    # plt.grid(True)
    plt.tight_layout()
    # plt.tick_params(top='off', right='off', which='both')  # stopped to work
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.tick_params(direction='in')
    ax.tick_params(axis='x', which='major', pad=8)  # space to X-labels
    if num_x_values > 0:
        plt.locator_params(axis='x', nbins=num_x_values)
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)
        print("The plot was saved as {}".format(outfile))
    return hist_areas


def plot_plane_normals(
        n=10, rand_dir=False, x_range=None, y_range=(0, 1), res=20,
        vertex_based=False):
    """ Plots estimated normals errors by VV versus original face normals
    (calculated by VTK) on a noisy plane surface for several radius_hit values.

    Args:
        n (int, optional): noise in % (default 10)
        rand_dir (boolean, optional): if True (default False), results where
            each point was moved in a random direction instead of the direction
            of its normal will be plotted.
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        res (int, optional): defines the size of the square plane in pixels and
            triangle division: 2*res
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center
    """
    if rand_dir:
        base_fold = "{}plane/res{}_noise{}_rand_dir/".format(FOLD, res, n)
    else:
        base_fold = "{}plane/res{}_noise{}/".format(FOLD, res, n)
    fold = "{}files4plotting/".format(base_fold)
    plot_fold = "{}plots/".format(base_fold)
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)

    plot_file = "{}plane_res{}_noise{}.VV_vs_VTK.normal_errors.png".format(
        plot_fold, res, n)
    if vertex_based:
        plot_file = os.path.splitext(plot_file)[0] + "_vertex_based.png"
    if x_range is not None:
        plot_file = os.path.splitext(plot_file)[0] + "_{}-{}.png".format(
            x_range[0], x_range[1])

    basename = "plane_half_size{}".format(res)
    if vertex_based:
        basename += "_vertex_based"
    SSVV_rh8_normal_errors = pd.read_csv("{}{}.SSVV_rh8.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    SSVV_rh4_normal_errors = pd.read_csv("{}{}.SSVV_rh4.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    VTK_normal_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                    sep=';')["normalErrors"].tolist()
    data = [VTK_normal_errors, SSVV_rh4_normal_errors, SSVV_rh8_normal_errors]
    print("maximal values: {}".format([max(d) for d in data]))
    if x_range is None:
        # Find minimal and maximal value to set the X-range:
        min_value = min([min(d) for d in data])
        max_value = max([max(d) for d in data])
        x_range_1 = (min_value, max_value)
    else:
        x_range_1 = x_range
    plot_composite_line_hist(
        data_arrays=data,
        labels=["Initial normals", r"VV rh=4", r"VV rh=8"],
        line_styles=['-', '-', '-'], markers=['s', 'v', '^'],
        colors=['r', 'c', 'b'],
        title=None,
        x_label="Normal orientation error",
        y_label="Cumulative relative frequency",
        outfile=plot_file,
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range_1, y_range=y_range,
        legend_loc='lower right'
    )


def plot_plane_normals_different_noise(
        rh=8, rand_dir=False, x_range=None, y_range=(0, 1), res=20,
        vertex_based=False):
    """ Plots estimated normals errors by VV on a noisy plane surface for
    several noise levels (with a fixed radius_hit value).

    Args:
        rh (int, optional): radius_hit parameter in pixels (default 8)
        rand_dir (boolean, optional): if True (default False), results where
            each point was moved in a random direction instead of the direction
            of its normal will be plotted.
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        res (int, optional): defines the size of the square plane in pixels and
            triangle division: 2*res
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center
    """
    plot_file = "{}plane/plane_res{}_rh{}.VV.normal_errors.png".format(
        FOLD, res, rh)
    if rand_dir:
        plot_file = os.path.splitext(plot_file)[0] + "_rand_dir.png"
    if vertex_based:
        plot_file = os.path.splitext(plot_file)[0] + "_vertex_based.png"
    if x_range is not None:
        plot_file = os.path.splitext(plot_file)[0] + "_{}-{}.png".format(
            x_range[0], x_range[1])

    data = []
    for n in [5, 10, 20, 30]:  # noise levels in %
        if rand_dir:
            base_fold = "{}plane/res{}_noise{}_rand_dir/".format(FOLD, res, n)
        else:
            base_fold = "{}plane/res{}_noise{}/".format(FOLD, res, n)
        fold = "{}files4plotting/".format(base_fold)
        plot_fold = "{}plots/".format(base_fold)
        if not os.path.exists(plot_fold):
            os.makedirs(plot_fold)

        basename = "plane_half_size{}".format(res)
        if vertex_based:
            basename += "_vertex_based"
        SSVV_normal_errors = pd.read_csv("{}{}.SSVV_rh{}.csv".format(
            fold, basename, rh), sep=';')["normalErrors"].tolist()
        data.append(SSVV_normal_errors)

    print("maximal values: {}".format([max(d) for d in data]))
    if x_range is None:
        # Find minimal and maximal value to set the X-range:
        min_value = min([min(d) for d in data])
        max_value = max([max(d) for d in data])
        x_range_1 = (min_value, max_value)
    else:
        x_range_1 = x_range
    plot_composite_line_hist(
        data_arrays=data,
        labels=["5% noise", "10% noise", "20% noise", "30% noise"],
        line_styles=['-', '-', '-', '-'], markers=['s', '^', 'v', '*'],
        colors=['g', 'b', 'y', 'r'],
        title=None,
        x_label="Normal orientation error",
        y_label="Cumulative relative frequency",
        outfile=plot_file,
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range_1, y_range=y_range,
        legend_loc='lower right'
    )


def plot_plane_normals_different_noise_and_rh(
        noise_levels=[5, 10, 20, 30], rand_dir=False, x_range=None,
        y_range=(0, 1), res=20, vertex_based=False):
    """ Plots areas under the curve of histograms of the estimated normals
    errors by VV on a noisy plane surface for several noise levels and
    radius_hit values.

    Args:
        noise_levels (list, optional): noise levels in %, default 5, 10, 20, 30
        rand_dir (boolean, optional): if True (default False), results where
            each point was moved in a random direction instead of the direction
            of its normal will be plotted.
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis of the underlying histograms (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis of the underlying histograms (default (0, 1))
        res (int, optional): defines the size of the square plane in pixels and
            triangle division: 2*res
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center
    """
    plot_file = "{}plane/plane_res{}.VV.normal_errors_AUC.png".format(
        FOLD, res)
    if rand_dir:
        plot_file = os.path.splitext(plot_file)[0] + "_rand_dir.png"
    if vertex_based:
        plot_file = os.path.splitext(plot_file)[0] + "_vertex_based.png"
    if x_range is not None:
        plot_file = os.path.splitext(plot_file)[0] + "_{}-{}.png".format(
            x_range[0], x_range[1])

    rhs = [4, 8]
    hist_areas_for_rhs = []
    for rh in rhs:
        hist_file = "{}plane/plane_res{}_rh{}.VV.normal_errors.png".format(
            FOLD, res, rh)
        if rand_dir:
            hist_file = os.path.splitext(hist_file)[0] + "_rand_dir.png"
        if vertex_based:
            hist_file = os.path.splitext(hist_file)[0] + "_vertex_based.png"
        if x_range is not None:
            hist_file = os.path.splitext(hist_file)[0] + "_{}-{}.png".format(
                x_range[0], x_range[1])
        data = []
        for n in noise_levels:
            if rand_dir:
                base_fold = "{}plane/res{}_noise{}_rand_dir/".format(
                    FOLD, res, n)
            else:
                base_fold = "{}plane/res{}_noise{}/".format(FOLD, res, n)
            fold = "{}files4plotting/".format(base_fold)
            plot_fold = "{}plots/".format(base_fold)
            if not os.path.exists(plot_fold):
                os.makedirs(plot_fold)

            basename = "plane_half_size{}".format(res)
            if vertex_based:
                basename += "_vertex_based"
            SSVV_normal_errors = pd.read_csv("{}{}.SSVV_rh{}.csv".format(
                fold, basename, rh), sep=';')["normalErrors"].tolist()
            data.append(SSVV_normal_errors)

        print("maximal values: {}".format([max(d) for d in data]))
        if x_range is None:
            # Find minimal and maximal value to set the X-range:
            min_value = min([min(d) for d in data])
            max_value = max([max(d) for d in data])
        else:
            min_value, max_value = x_range
        x_range_1 = (min_value, max_value)
        hist_areas = plot_composite_line_hist(
            data_arrays=data,
            labels=["5% noise", "10% noise", "20% noise", "30% noise"],
            line_styles=['-', '-', '-', '-'], markers=['s', '^', 'v', '*'],
            colors=['g', 'b', 'y', 'r'],
            title=None,
            x_label="Normal orientation error",
            y_label="Cumulative relative frequency",
            outfile=hist_file,
            num_bins=20, normalize=True, cumulative=True,
            x_range=x_range_1, y_range=y_range,
        )
        hist_areas_for_rhs.append(hist_areas)

    # Do the plot:
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()

    for rh, hist_areas, m, c in zip(
            rhs, hist_areas_for_rhs, ['*', '^'], ['cyan', 'b']):
        plt.plot(noise_levels, hist_areas, ls='-', marker=m, c=c,
                 label='VV rh={}'.format(rh), linewidth=LINEWIDTH, clip_on=False)

    plt.xlabel("Noise (%)")
    plt.ylabel("Area of cumulative\nnormal error histogram < {}".format(
        max_value))
    if y_range is not None:
        plt.ylim(y_range)
    plt.legend(loc="lower right", fancybox=True, framealpha=0.5, fontsize=18,
               ncol=1, columnspacing=1, handletextpad=0.2, borderpad=0.2)
    plt.tight_layout()
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.tick_params(direction='in')
    ax.tick_params(axis='x', which='major', pad=8)  # space to X-labels
    # plt.show()
    fig.savefig(plot_file)
    print("The plot was saved as {}".format(plot_file))


def plot_cylinder_kappa_1_diff_rh(n=0, x_range=None, num_bins=20):
    """Plots estimated kappa_1 values histograms on a cylinder surface by
    different methods (AVV and SSVV) using different radius_hit.

    Args:
        n (int, optional): noise in % (default 0)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        num_bins (int, optional): number of bins for the histogram (default 20)
    """
    fold = ("{}cylinder/noise{}/files4plotting/".format(FOLD, n))
    plot_fold = ("{}cylinder/noise{}/plots/".format(FOLD, n))
    basename = "cylinder_r10_h25_eb0"
    for method in ['SSVV', 'AVV']:
        plot_file = "{}{}_noise{}.{}_rh5-9.kappa_1_bins{}.png".format(
            plot_fold, basename, n, method, num_bins)
        if x_range is not None:
            plot_file = os.path.splitext(plot_file)[0] + "_{}-{}.png".format(
                x_range[0], x_range[1])
        kappa_arrays = []
        labels = []
        for rh in range(5, 10):
            kappa_array = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')["kappa1"].tolist()
            kappa_arrays.append(kappa_array)
            label = r"rh={}".format(rh)
            labels.append(label)
        if x_range is None:
            # Find minimal and maximal value to set the X-range:
            min_value = min([min(d) for d in kappa_arrays])
            max_value = max([max(d) for d in kappa_arrays])
            x_range_1 = (min_value, max_value)
        else:
            x_range_1 = x_range
        plot_composite_line_hist(
            data_arrays=kappa_arrays, labels=labels,
            line_styles=['-', '-', '-', '-', '-'],
            markers=['*', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on cylinder ({}% noise)".format(method, n),
            x_label=r"$\kappa_1$", x_range=x_range_1,
            y_label="Frequency",
            num_bins=num_bins, normalize=False, outfile=plot_file
        )


def plot_cylinder_curvature_errors_diff_rh(
        methods=["RVV", "AVV", "SSVV"], curvature="kappa1", rhs=list(range(5, 10)),
        x_range=None, y_range=(0, 1), num_bins=20,
        legend_loc="lower right", csv=None, voxel=False, *args, **kwargs):
    """
    Plots estimated curvature errors histograms on a cylinder surface
    for different methods and radius_hit.

    Args:
        methods (list, optional): tells which method(s) should be used
            (default=["RVV", "AVV", "SSVV"])
        curvature (str, optional): "kappa1" (default), "kappa2" or
            "mean_curvature"
        rhs (list, optional): wanted radius_hit parameter values (default 5-9)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        num_bins (int, optional): number of bins for the histogram (default 20)
        legend_loc (str, optional): legend location (default 'lower right')
        csv (str, optional): csv file for saving cumulative histogram areas
        voxel (boolean, optional): if noisy cylinder should be plotted
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist
    """
    if voxel:
        subfolds = "cylinder/voxel/"
    else:
        subfolds = "cylinder/noise0/"
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "cylinder_r10_h25_eb0"
    y_label = "Cumulative relative frequency"

    method_col = []
    rh_col = []
    hist_area_curv_col = []
    for i, method in enumerate(methods):
        print(method)
        curv_arrays = []
        labels = []
        for rh in rhs:
            df = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')
            curv_array = df["{}RelErrors".format(curvature)].tolist()
            curv_arrays.append(curv_array)
            label = r"rh={}".format(rh)
            labels.append(label)
        if x_range is None:
            # Find the maximal value to set the X-range:
            max_value = max([max(d) for d in curv_arrays])
            x_range_method = (0, max_value)
        else:
            x_range_method = x_range
        data_size = len(curv_arrays)
        line_styles = ['-'] * data_size
        markers_iter = cycle(MARKERS)
        markers = [next(markers_iter) for i in range(data_size)]
        colors_iter = cycle(COLORS)
        colors = [next(colors_iter) for i in range(data_size)]
        if curvature == "kappa1":
            formatted_curvature = r"$\kappa_1$"
        elif curvature == "kappa2":
            formatted_curvature = r"$\kappa_2$"
        else:
            formatted_curvature = "mean curvature"

        hist_areas = plot_composite_line_hist(
            data_arrays=curv_arrays, labels=labels,
            line_styles=line_styles, markers=markers, colors=colors,
            title=method,
            x_label="Relative {} error".format(formatted_curvature),
            y_label=y_label,
            outfile="{}{}.{}_rh{}-{}.{}_errors_range{}-{}.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1], curvature,
                x_range_method[0], x_range_method[1]),
            num_bins=num_bins, x_range=x_range_method, y_range=y_range,
            normalize=True, cumulative=True, legend_loc=legend_loc,
            *args, **kwargs
        )
        method_col += [method] * data_size
        rh_col += rhs
        hist_area_curv_col += hist_areas
        i = hist_areas.index(max(hist_areas))
        print("Best performance for {} is for radius_hit={}".format(
            curvature, rhs[i]))
    if csv is not None:
        df = pd.DataFrame(index=None)
        df["method"] = method_col
        df["RadiusHit"] = rh_col
        df["hist_area_{}".format(curvature)] = hist_area_curv_col
        df.to_csv(csv, sep=';')


def plot_cylinder_t_2_and_kappa_1_errors(
        RorAVV="AVV", rhVV=5, rhSSVV=6, n=None, voxel=False, exclude_borders=5,
        y_range=(0, 1), x_range_T=None, x_range_kappa=None,
        legend_loc='lower right', vertex_based=False, *args, **kwargs):
    """Plots estimated kappa_2 and t_1 errors histograms on a cylinder surface
    for different methods (RVV or AVV and SSVV) and optimal radius_hit for each
    method.

    Args:
        RorAVV (str, optional): RVV or AVV (default)
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        rhVV (int, optional): radius_hit for VV (default 5)
        rhSSVV (int, optional): radius_hit for SSVV (default 6)
        n (int, optional): n for Mindboggle (MB) -m 0 (default None=not plotted)
        voxel (boolean, optional): if True (default False), voxel cylinder
            results are used
        exclude_borders (int, optional): how many voxels from border were
            excluded for curvature calculation (default 5)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        x_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        x_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        legend_loc (str, optional): legend location (default 'lower right')
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist
    """
    if voxel:
        fold = ("{}cylinder/voxel/files4plotting/".format(FOLD))
        mb_fold = "{}noisy_cylinder/".format(FOLDMB)
        mb_csv = ("noisy_cylinder_r10_h25.surface.mindboggle_m0_n{}_"
                  "curvature_errors.csv".format(n))
        plot_fold = "{}noisy_cylinder/".format(FOLDPLOTS)
    else:
        fold = ("{}cylinder/noise0/files4plotting/".format(FOLD))
        mb_fold = "{}smooth_cylinder/".format(FOLDMB)
        mb_csv = ("cylinder_r10_h25.surface.mindboggle_m0_n{}_"
                  "curvature_errors.csv".format(n))
        plot_fold = "{}smooth_cylinder/".format(FOLDPLOTS)
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "cylinder_r10_h25_eb{}".format(exclude_borders)
    if vertex_based:
        basename += "_vertex_based"
    df = pd.read_csv("{}{}.SSVV_rh{}.csv".format(
        fold, basename, rhSSVV), sep=';')
    SSVV_t_2_errors = df["T2Errors"].tolist()
    SSVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.{}_rh{}.csv".format(fold, basename, RorAVV, rhVV),
                     sep=';')
    VV_t_2_errors = df["T2Errors"].tolist()
    VV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VTK_kappa_1_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa1RelErrors"].tolist()

    if n is not None:
        MB_kappa_errors = pd.read_csv("{}{}".format(
            mb_fold, mb_csv), sep=';')["kappa1RelErrors"].tolist()

    # directions
    data = [VV_t_2_errors, SSVV_t_2_errors]
    outfile = "{}{}.{}_rh{}_SSVV_rh{}.t_2_errors.png".format(
        plot_fold, basename, RorAVV, rhVV, rhSSVV)
    if RorAVV == "AVV":
        markers = ['o']
        colors = ['orange']
    else:  # RVV
        markers = ['v']
        colors = ['c']
    markers.append('^')  # SSVV
    colors.append('b')
    title = ("Including borders" if exclude_borders == 0
             else "Excluding borders ({} voxels)".format(exclude_borders))
    if x_range_T is None:
        x_range_T = (0, max([max(d) for d in data]))
    else:
        outfile = outfile.replace(
            ".png", "{}-{}.png".format(x_range_T[0], x_range_T[1]))
    plot_composite_line_hist(
        data_arrays=data,
        labels=[r"{} rh={}".format(RorAVV, rhVV),
                r"SSVV rh={}".format(rhSSVV)],
        zorders=[2, 1],
        line_styles=['-']*len(data), markers=markers, colors=colors,
        title=title,
        x_label=r"$\vec t_2\ error$",
        y_label="Cumulative relative frequency",
        outfile=outfile,
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range_T, y_range=y_range, legend_loc=legend_loc,
        *args, **kwargs
    )

    # curvatures
    data = [VV_kappa_1_errors, SSVV_kappa_1_errors, VTK_kappa_1_errors]
    outfile = "{}{}.{}_rh{}_SSVV_rh{}_VTK.kappa_1_errors.png".format(
        plot_fold, basename, RorAVV, rhVV, rhSSVV)
    markers.append('s')  # VTK
    colors.append('r')
    labels = [r"{} rh={}".format(RorAVV, rhVV),
              r"SSVV rh={}".format(rhSSVV), "VTK"]
    zorders = [3, 2, 1]
    if n is not None:
        data.append(MB_kappa_errors)  # Mindboggle
        markers.append('*')
        colors.append('purple')
        labels.append("MB n={}".format(n))
        zorders.append(4)
        outfile = outfile.replace("VTK", "VTK_MB")
    if x_range_kappa is None:
        x_range_kappa = (0, max([max(d) for d in data]))
    else:
        outfile = outfile.replace(
            ".png", "{}-{}.png".format(x_range_kappa[0], x_range_kappa[1]))
    plot_composite_line_hist(
        data_arrays=data,
        labels=labels, zorders=zorders,
        line_styles=['-']*len(data), markers=markers, colors=colors,
        title=title,
        x_label=r"$Relative\ \kappa_1\ error$",
        y_label="Cumulative relative frequency",
        outfile=outfile,
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range_kappa, y_range=y_range, legend_loc=legend_loc,
        *args, **kwargs
    )


def plot_cylinder_t_2_and_kappa_1_errors_allVV(n=0, y_range=(0, 1),
                                               exclude_borders=5):
    """Plots estimated kappa_2 and t_1 errors histograms on a cylinder surface
    for different methods (RVV, AVV and SSVV) and optimal radius_hit for each
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
    SSVV_t_2_errors = df["T2Errors"].tolist()
    SSVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.RVV_rh5.csv".format(fold, basename), sep=';')
    RVV_t_2_errors = df["T2Errors"].tolist()
    RVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.AVV_rh5.csv".format(fold, basename), sep=';')
    AVV_t_2_errors = df["T2Errors"].tolist()
    AVV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VTK_kappa_1_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa1RelErrors"].tolist()
    data = [SSVV_t_2_errors, RVV_t_2_errors, AVV_t_2_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=7", "RVV rh=5", "AVV rh=5"],
        line_styles=['-', '-', '-'], markers=['^', 'v', 'o'],
        colors=['b', 'c', 'g'],
        title="Cylinder ({}% noise)".format(n),
        x_label=r"$t_2\ error$",
        y_label="Cumulative relative frequency",
        outfile="{}{}_noise{}.RVV_AVV_rh5_SSVV_rh7.t_2_errors.png".format(
            plot_fold, basename, n),
        num_bins=20, normalize=True, cumulative=True,  # max_val=1
        x_range=(0, 0.005),  # (0, max([max(d) for d in data]))
        y_range=y_range
    )
    data = [SSVV_kappa_1_errors, RVV_kappa_1_errors, AVV_kappa_1_errors,
            VTK_kappa_1_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=7", "RVV rh=5", "AVV rh=5", "VTK"],
        line_styles=['-', '-', '-', '-'], markers=['^', 'v', 'o', 's'],
        colors=['b', 'c', 'g', 'r'],
        title="Cylinder ({}% noise)".format(n),
        x_label=r"$\kappa_1\ relative\ error$",
        y_label="Cumulative relative frequency",
        outfile=("{}{}_noise{}.RVV_AVV_rh5_SSVV_rh7_vs_VTK.kappa_1_errors.png"
                 .format(plot_fold, basename, n)),
        num_bins=20, normalize=True, cumulative=True,  # max_val=1
        x_range=(0, max([max(d) for d in data])), y_range=y_range
    )


def plot_inverse_cylinder_t_1_and_kappa_2_errors(n=0):
    """Plots estimated kappa_2 and t_1 errors histograms on an inverse cylinder
    surface for different methods (RVV and SSVV) and optimal radius_hit for
    each method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("{}cylinder/noise0/files4plotting/".format(FOLD, n))
    plot_fold = ("{}cylinder/noise0/plots/".format(FOLD, n))
    basename = "inverse_cylinder_r10_h25"
    df = pd.read_csv("{}{}.SSVV_rh8.csv".format(fold, basename), sep=';')
    SSVV_t_1_errors = df["T1Errors"].tolist()
    SSVV_kappa_2_errors = df["kappa2RelErrors"].tolist()

    df = pd.read_csv("{}{}.RVV_rh8.csv".format(fold, basename), sep=';')
    VV_t_1_errors = df["T1Errors"].tolist()
    VV_kappa_2_errors = df["kappa2RelErrors"].tolist()

    VTK_kappa_2_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa2RelErrors"].tolist()
    data = [SSVV_t_1_errors,
            VV_t_1_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=8", "RVV rh=8"],
        line_styles=['-', '-'], markers=['^', 'v'],
        colors=['b', 'c'],
        title="Inverse cylinder ({}% noise)".format(n),
        x_label="t_1 error",
        y_label="Cumulative relative frequency",
        outfile="{}{}_noise{}.RVV_SSVV_rh8.t_1_errors.png".format(
            plot_fold, basename, n),
        num_bins=20, normalize=True, cumulative=True,  # max_val=1
        x_range=(0, max([max(d) for d in data]))
    )
    data = [SSVV_kappa_2_errors,
            VV_kappa_2_errors,
            VTK_kappa_2_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["SSVV rh=8", "RVV rh=8", "VTK"],
        line_styles=['-', '-', '-'], markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Inverse cylinder ({}% noise)".format(n),
        x_label=r"$\kappa_2\ relative\ error$",
        y_label="Cumulative relative frequency",
        outfile=("{}{}_noise{}.RVV_SSVV_rh8_vs_VTK.kappa_2_errors.png".format(
            plot_fold, basename, n)),
        num_bins=20, normalize=True, cumulative=True,  # max_val=1
        x_range=(0, max([max(d) for d in data]))
    )


def plot_sphere_kappa_1_and_2_diff_rh(
        r=10, n=0, ico=0, voxel=False, methods=["RVV", "AVV", "SSVV"],
        rhs=list(range(5, 10)), x_range=None, y_range=None, num_bins=20,
        legend_loc='upper left'):
    """Plots estimated kappa_1 and kappa_2 values for a sphere surface
     by different methods (RVV, AVV and SSVV) using different radius_hit.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (e.g. 1280), icosahedron results with so
            many faces are used; if 0 (default), smooth sphere results are used
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used (ignoring the other options)
        methods (list, optional): tells which method(s) should be used
            (default=["RVV", "AVV", "SSVV"])
        rhs (list, optional): wanted radius_hit parameter values (default 5-9)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        num_bins (int, optional): number of bins for the histogram (default 20)
        legend_loc (str, optional): legend location (default 'upper left')
    """
    if voxel:
        subfolds = "sphere/voxel/"
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)
    y_label = "Relative frequency"
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
            label = r"rh={}".format(rh)
            labels.append(label)
        if x_range is None:
            # Find minimal and maximal value to set the X-range:
            min_value = min([min(d) for d in kappa_1_arrays])
            max_value = max([max(d) for d in kappa_1_arrays])
            x_range_1 = (min_value, max_value)
        else:
            x_range_1 = x_range
        plot_composite_line_hist(  # kappa_1
            data_arrays=kappa_1_arrays,
            labels=labels,
            line_styles=['-', '-', '-', '-', '-'],
            markers=['*', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title=None,
            x_label=r"$\kappa_1$",
            y_label=y_label,
            outfile="{}{}.{}_rh{}-{}.kappa_1.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1]),
            num_bins=num_bins, x_range=x_range_1, max_val=None, normalize=True,
            y_range=y_range, legend_loc=legend_loc
        )
        if x_range is None:
            # Find minimal and maximal value to set the X-range:
            min_value = min([min(d) for d in kappa_2_arrays])
            max_value = max([max(d) for d in kappa_2_arrays])
            x_range_2 = (min_value, max_value)
        else:
            x_range_2 = x_range
        plot_composite_line_hist(  # kappa_2
            data_arrays=kappa_2_arrays,
            labels=labels,
            line_styles=['-', '-', '-', '-', '-'],
            markers=['*', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title=None,
            x_label=r"$\kappa_2$",
            y_label=y_label,
            outfile=("{}{}.{}_rh{}-{}.kappa_2.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=num_bins, x_range=x_range_2, max_val=None, normalize=True,
            y_range=y_range, legend_loc=legend_loc
        )
        if x_range is None:
            # Find minimal and maximal value to set the X-range:
            min_value = min([min(d) for d in kappas_arrays])
            max_value = max([max(d) for d in kappas_arrays])
            x_range_1_2 = (min_value, max_value)
        else:
            x_range_1_2 = x_range
        plot_composite_line_hist(  # kappa_1 + kappa_2
            data_arrays=kappas_arrays,
            labels=labels,
            line_styles=['-', '-', '-', '-', '-'],
            markers=['*', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title=None,
            x_label=r"$\kappa_1\ and\ \kappa_2$",
            y_label=y_label,
            outfile=("{}{}.{}_rh{}-{}.kappa_1_and_2.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])),
            num_bins=num_bins, x_range=x_range_1_2, max_val=None,
            normalize=True, y_range=y_range, legend_loc=legend_loc
        )


def plot_sphere_curvature_errors_diff_rh(
        r=10, methods=["AVV", "RVV", "SSVV"], curvature="both",
        rhs=list(range(5, 10)), n=0, ico=0, voxel=False, x_range=None,
        y_range=(0, 1), num_bins=20, legend_loc="lower right", csv=None, *args,
        **kwargs):
    """
    Plots estimated curvature errors histograms on a sphere surface
    for different methods and radius_hit.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        methods (list, optional): tells which method(s) should be used
            (default=["RVV", "AVV", "SSVV"])
        curvature (str, optional): "kappa1", "kappa2", "both" (default) or
            "mean_curvature"
        rhs (list, optional): wanted radius_hit parameter values (default 5-9)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (e.g. 1280), icosahedron results with so
            many faces are used; if 0 (default), smooth sphere results are used
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used (ignoring the other options)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        num_bins (int, optional): number of bins for the histogram (default 20)
        legend_loc (str, optional): legend location (default 'lower right')
        csv (str, optional): csv file for saving cumulative histogram areas
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist
    """
    if voxel:
        subfolds = "sphere/voxel/"
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)
    y_label = "Cumulative relative frequency"

    method_col = []
    rh_col = []
    hist_area_curv_col = []
    for i, method in enumerate(methods):
        print(method)
        curv_arrays = []
        labels = []
        for rh in rhs:
            df = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')
            if curvature == "both":
                kappa_1_array = df["kappa1RelErrors"].tolist()
                kappa_2_array = df["kappa2RelErrors"].tolist()
                curv_arrays.append(kappa_1_array + kappa_2_array)
            else:
                curv_array = df["{}RelErrors".format(curvature)].tolist()
                curv_arrays.append(curv_array)
            label = r"rh={}".format(rh)
            labels.append(label)
        if x_range is None:
            # Find the maximal value to set the X-range:
            max_value = max([max(d) for d in curv_arrays])
            x_range_method = (0, max_value)
        else:
            x_range_method = x_range
        data_size = len(curv_arrays)
        line_styles = ['-'] * data_size
        markers_iter = cycle(MARKERS)
        markers = [next(markers_iter) for i in range(data_size)]
        colors_iter = cycle(COLORS)
        colors = [next(colors_iter) for i in range(data_size)]
        if curvature == "kappa1":
            formatted_curvature = r"$\kappa_1$"
        elif curvature == "kappa2":
            formatted_curvature = r"$\kappa_2$"
        elif curvature == "mean_curvature":
            formatted_curvature = "mean curvature"
        else:
            formatted_curvature = r"$\kappa_1\ and\ \kappa_2$"

        hist_areas = plot_composite_line_hist(
            data_arrays=curv_arrays, labels=labels,
            line_styles=line_styles, markers=markers, colors=colors,
            title=method,
            x_label="Relative {} error".format(formatted_curvature),
            y_label=y_label,
            outfile="{}{}.{}_rh{}-{}.{}_errors_range{}-{}.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1], curvature,
                x_range_method[0], x_range_method[1]),
            num_bins=num_bins, x_range=x_range_method, y_range=y_range,
            normalize=True, cumulative=True, legend_loc=legend_loc,
            *args, **kwargs
        )
        method_col += [method] * data_size
        rh_col += rhs
        hist_area_curv_col += hist_areas
        i = hist_areas.index(max(hist_areas))
        print("Best performance for {} is for radius_hit={}".format(
            curvature, rhs[i]))
    if csv is not None:
        df = pd.DataFrame(index=None)
        df["method"] = method_col
        df["RadiusHit"] = rh_col
        df["hist_area_{}".format(curvature)] = hist_area_curv_col
        df.to_csv(csv, sep=';')


def plot_sphere_kappa_1_and_2_errors(
        r=10, rhVV=9, rhSSVV=9, n=0, ico=0, voxel=False, x_range=None,
        y_range=(0, 1), RorAVV="AVV", vertex_based=False):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (RVV, AVV, SSVV and VTK) and an optimal radius_hit for
    each method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 9)
        rhSSVV (int, optional): radius_hit for SSVV (default 9)
        n (int, optional): noise in % (default 0)
        ico (int, optional): if > 0 (i.e. 1280), icosahedron results with so
            many faces are used; if 0 (default), smooth sphere results are used
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used (ignoring the other options)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        RorAVV (str, optional): RVV or AVV (default)
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center
    """
    if voxel:
        subfolds = "sphere/voxel/"
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)
    if vertex_based:
        basename += "_vertex_based"

    df_SSVV = pd.read_csv("{}{}.SSVV_rh{}.csv".format(fold, basename, rhSSVV),
                          sep=';')
    SSVV_kappa_1_errors = df_SSVV["kappa1RelErrors"].tolist()
    SSVV_kappa_2_errors = df_SSVV["kappa2RelErrors"].tolist()

    df_VV = pd.read_csv("{}{}.{}_rh{}.csv".format(fold, basename, RorAVV, rhVV),
                         sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    df_VTK = pd.read_csv("{}{}.VTK.csv".format(fold, basename), sep=';')
    VTK_kappa_1_errors = df_VTK["kappa1RelErrors"].tolist()
    VTK_kappa_2_errors = df_VTK["kappa2RelErrors"].tolist()

    data = [VV_kappa_1_errors + VV_kappa_2_errors,
            SSVV_kappa_1_errors + SSVV_kappa_2_errors,
            VTK_kappa_1_errors + VTK_kappa_2_errors]
    if x_range is None:
        x_range = (0, max([max(d) for d in data]))
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["{} rh={}".format(RorAVV, rhVV),
                "SSVV rh={}".format(rhSSVV), "VTK"],
        line_styles=['-', '-', '-'],
        markers=['v', '^', 's'],
        colors=['c', 'b', 'r'],
        title="Sphere radius={}".format(r),
        x_label=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        y_label="Cumulative relative frequency",
        outfile=("{}{}.{}rh{}_SSVVrh{}_vs_VTK."
                 "kappa_1_and_2_errors.png".format(
                    plot_fold, basename, RorAVV, rhVV, rhSSVV)),
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range, y_range=y_range
    )


def plot_sphere_curvature_errors_allVV(
        r=10, rhRVV=9, rhAVV=9, rhSSVV=9, n=8, voxel=False,
        curvature="both", x_range=None, y_range=(0, 1), onlyVV=False,
        *args, **kwargs):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (RVV, AVV, SSVV and VTK) and an optimal radius_hit for
    each method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhRVV (int, optional): radius_hit for RVV (default 9)
        rhAVV (int, optional): radius_hit for AVV (default 9)
        rhSSVV (int, optional): radius_hit for SSVV (default 9)
        n (int, optional): n for Mindboggle (MB) -m 0 (default 8)
        voxel (boolean, optional): if True (default False), voxel sphere
            results are used
        curvature (str, optional): "kappa1", "kappa2", "both" (default) or
            "mean_curvature"
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist
    """
    if voxel:
        subfolds = "sphere/voxel/"
        which_sphere = "noisy_sphere"
    else:
        subfolds = "sphere/noise0/"
        which_sphere = "smooth_sphere"
    mb_fold = join(FOLDMB, which_sphere)
    mb_csv = ("{}_r{}.surface.mindboggle_m0_n{}_"
              "curvature_errors.csv".format(which_sphere, r, n))
    fs_fold = join(FOLDFS, which_sphere, "csv")
    fs_csv = "{}_r{}.freesurfer_curvature_errors.csv".format(which_sphere, r)
    plot_fold = join(FOLDPLOTS, which_sphere)
    fold = join(FOLD, subfolds, "files4plotting")
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "sphere_r{}".format(r)

    df = pd.read_csv("{}/{}.SSVV_rh{}.csv".format(
        fold, basename, rhSSVV), sep=';')
    if curvature == "both":
        SSVV_kappa_1_errors = df["kappa1RelErrors"].tolist()
        SSVV_kappa_2_errors = df["kappa2RelErrors"].tolist()
        SSVV_curv_errors = SSVV_kappa_1_errors + SSVV_kappa_2_errors
        assert len(SSVV_curv_errors) == (len(SSVV_kappa_1_errors) +
                                         len(SSVV_kappa_2_errors))
    else:
        SSVV_curv_errors = df["{}RelErrors".format(curvature)].tolist()

    df = pd.read_csv("{}/{}.RVV_rh{}.csv".format(fold, basename, rhRVV), sep=';')
    if curvature == "both":
        RVV_kappa_1_errors = df["kappa1RelErrors"].tolist()
        RVV_kappa_2_errors = df["kappa2RelErrors"].tolist()
        RVV_curv_errors = RVV_kappa_1_errors + RVV_kappa_2_errors
    else:
        RVV_curv_errors = df["{}RelErrors".format(curvature)].tolist()

    df = pd.read_csv("{}/{}.AVV_rh{}.csv".format(fold, basename, rhAVV), sep=';')
    if curvature == "both":
        AVV_kappa_1_errors = df["kappa1RelErrors"].tolist()
        AVV_kappa_2_errors = df["kappa2RelErrors"].tolist()
        AVV_curv_errors = AVV_kappa_1_errors + AVV_kappa_2_errors
    else:
        AVV_curv_errors = df["{}RelErrors".format(curvature)].tolist()

    df = pd.read_csv("{}/{}.VTK.csv".format(fold, basename), sep=';')
    if curvature == "both":
        VTK_kappa_1_errors = df["kappa1RelErrors"].tolist()
        VTK_kappa_2_errors = df["kappa2RelErrors"].tolist()
        VTK_curv_errors = VTK_kappa_1_errors + VTK_kappa_2_errors
    else:
        VTK_curv_errors = df["{}RelErrors".format(curvature)].tolist()

    df = pd.read_csv("{}/{}".format(mb_fold, mb_csv), sep=';')
    if curvature == "both":
        MB_kappa_1_errors = df["kappa1RelErrors"].tolist()
        MB_kappa_2_errors = df["kappa2RelErrors"].tolist()
        MB_curv_errors = MB_kappa_1_errors + MB_kappa_2_errors
    else:
        MB_curv_errors = df["{}RelErrors".format(curvature)].tolist()

    df = pd.read_csv("{}/{}".format(fs_fold, fs_csv), sep=';')
    if curvature == "both":
        FS_kappa_1_errors = df["kappa1RelErrors"].tolist()
        FS_kappa_2_errors = df["kappa2RelErrors"].tolist()
        FS_curv_errors = FS_kappa_1_errors + FS_kappa_2_errors
    else:
        FS_curv_errors = df["{}RelErrors".format(curvature)].tolist()

    if onlyVV is False:
        data = [RVV_curv_errors, AVV_curv_errors, SSVV_curv_errors,
                VTK_curv_errors, MB_curv_errors, FS_curv_errors]

        outfile = ("{}/{}.RVVrh{}_AVVrh{}_SSVVrh{}_VTK_MBn{}_FS.{}_errors.png"
            .format(plot_fold, basename, rhRVV, rhAVV, rhSSVV, n, curvature))
        if x_range is None:
            x_range = (0, max([max(d) for d in data]))
        else:
            outfile = outfile.replace(
                ".png", "{}-{}.png".format(x_range[0], x_range[1]))

        if curvature == "kappa1":
            formatted_curvature = r"$\kappa_1$"
        elif curvature == "kappa2":
            formatted_curvature = r"$\kappa_2$"
        elif curvature == "mean_curvature":
            formatted_curvature = "mean curvature"
        else:
            formatted_curvature = r"$\kappa_1\ and\ \kappa_2$"

        plot_composite_line_hist(
            data_arrays=data,
            labels=[r"RVV rh={}".format(rhRVV),
                    r"AVV rh={}".format(rhAVV),
                    r"SSVV rh={}".format(rhSSVV),
                    "VTK",
                    "MB n={}".format(n),
                    "FS"],
            zorders=[2, 4, 3, 1, 5, 6],
            line_styles=['-']*len(data), markers=['v', 'o', '^', 's', '*', '.'],
            colors=['c', 'orange', 'b', 'r', 'purple', 'g'],
            title="Sphere radius={}".format(r),
            x_label="Relative {} error".format(formatted_curvature),
            y_label="Cumulative relative frequency",
            outfile=outfile,
            num_bins=20, normalize=True, cumulative=True,
            x_range=x_range, y_range=y_range,
            *args, **kwargs
        )
    else:
        data = [RVV_curv_errors, AVV_curv_errors, SSVV_curv_errors]

        outfile = (
        "{}/{}.RVVrh{}_AVVrh{}_SSVVrh{}.{}_errors.png".format(
            plot_fold, basename, rhRVV, rhAVV, rhSSVV, curvature))
        if x_range is None:
            x_range = (0, max([max(d) for d in data]))
        else:
            outfile = outfile.replace(
                ".png", "{}-{}.png".format(x_range[0], x_range[1]))

        if curvature == "kappa1":
            formatted_curvature = r"$\kappa_1$"
        elif curvature == "kappa2":
            formatted_curvature = r"$\kappa_2$"
        elif curvature == "mean_curvature":
            formatted_curvature = "mean curvature"
        else:
            formatted_curvature = r"$\kappa_1\ and\ \kappa_2$"

        plot_composite_line_hist(
            data_arrays=data,
            labels=[r"RVV rh={}".format(rhRVV),
                    r"AVV rh={}".format(rhAVV),
                    r"SSVV rh={}".format(rhSSVV)],
            zorders=[1, 3, 2],
            line_styles=['-'] * len(data), markers=['v', 'o', '^'],
            colors=['c', 'orange', 'b'],
            title="Sphere radius={}".format(r),
            x_label="Relative {} error".format(formatted_curvature),
            y_label="Cumulative relative frequency",
            outfile=outfile,
            num_bins=20, normalize=True, cumulative=True,
            x_range=x_range, y_range=y_range,
            *args, **kwargs
        )


def plot_sphere_kappa_1_and_2_errors_noVTK(
        r=10, rhVV=9, rhSSVV=9, n=0, ico=0, voxel=False, x_range=None,
        y_range=(0, 1), RorAVV="AVV", legend_loc='lower right'):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (RVV, AVV and SSVV) and an optimal radius_hit for each
    method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 9)
        rhSSVV (int, optional): radius_hit for SSVV (default 9)
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
        legend_loc (str, optional): legend location (default 'lower right')
    """
    if voxel:
        subfolds = "sphere/voxel/"
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
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
        line_styles = ['-', '-']
        markers = ['o', '^']
        colors = ['orange', 'b']
    else:
        line_styles = ['-', '-']
        markers = ['v', '^']
        colors = ['c', 'b']
    df_VV = pd.read_csv("{}{}.{}_rh{}.csv".format(fold, basename, RorAVV, rhVV),
                        sep=';')
    VV_kappa_1_errors = df_VV["kappa1RelErrors"].tolist()
    VV_kappa_2_errors = df_VV["kappa2RelErrors"].tolist()

    data = [VV_kappa_1_errors + VV_kappa_2_errors,
            SSVV_kappa_1_errors + SSVV_kappa_2_errors]
    if x_range is None:
        x_range = (0, max([max(d) for d in data]))
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["{} rh={}".format(RorAVV, rhVV),
                "SSVV rh={}".format(rhSSVV)],
        line_styles=line_styles, markers=markers, colors=colors,
        title="Sphere radius={}".format(r),
        x_label=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        y_label="Cumulative relative frequency",
        outfile=("{}{}.{}rh{}_vs_SSVVrh{}.kappa_1_and_2_errors_range{}.png".
                 format(plot_fold, basename, RorAVV, rhVV, rhSSVV, x_range[1])),
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range, y_range=y_range, legend_loc=legend_loc
    )


def plot_sphere_kappa_1_and_2_errors_noVTK_allVV(
        r=10, rhVV=9, rhSSVV=9, n=0, ico=0, voxel=False, value_range=None,
        y_range=(0, 1)):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on a sphere surface
    for different methods (VV and SSVV) and an optimal radius_hit for each
    method.

    Args:
        r (int, optional): radius of the sphere in voxels (default 10)
        rhVV (int, optional): radius_hit for VV (default 9)
        rhSSVV (int, optional): radius_hit for SSVV (default 9)
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
    elif ico > 0:
        subfolds = "sphere/ico{}_noise{}/".format(ico, n)
    else:
        subfolds = "sphere/noise{}/".format(n)
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

    data = [RVV_kappa_1_errors + RVV_kappa_2_errors,
            AVV_kappa_1_errors + AVV_kappa_2_errors,
            SSVV_kappa_1_errors + SSVV_kappa_2_errors]
    if value_range is None:
        value_range = (0, max([max(d) for d in data]))
    plot_composite_line_hist(  # kappa_1 + kappa_2
        data_arrays=data,
        labels=["RVV rh={}".format(rhVV),
                "AVV rh={}".format(rhVV),
                "SSVV rh={}".format(rhSSVV)],
        line_styles=['-', '-', '-'],
        markers=['v', 'o', '^'],
        colors=['c', 'orange', 'b'],
        title=None,
        x_label=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        y_label="Cumulative relative frequency",
        outfile=("{}{}.RVV_AVVrh{}_vs_SSVVrh{}.kappa_1_and_2_errors_range{}.png"
                 .format(plot_fold, basename, rhVV, rhSSVV, value_range[1])),
        num_bins=20, normalize=True, cumulative=True,
        x_range=value_range, y_range=y_range
    )


def plot_inverse_sphere_kappa_1_and_2_errors(n=0):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on an inverse
    icosahedron sphere surface for different methods (RVV and SSVV) and an
    optimal radius_hit for each method.

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
        line_styles=['-', '-', '-'],
        markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Inverse sphere (icosahedron 1280, {}% noise)".format(n),
        x_label=r"$\kappa_1\ and\ \kappa_2\ relative\ error$",
        y_label="Cumulative relative frequency",
        outfile=("{}inverse_icosphere_r10_noise{}.RVV_SSVV_vs_VTK."
                 "kappa_1_and_2_errors.png".format(plot_fold, n)),
        num_bins=20, normalize=True, cumulative=True,
        x_range=(0, max([max(d) for d in data]))
    )


def plot_torus_kappa_1_and_2_diff_rh(
        methods=["RVV", "AVV", "SSVV"],
        rhs=list(range(5, 10)), x_range=None, num_bins=20):
    """Plots estimated kappa_1 values for a torus surface
     by different methods (RVV, AVV and SSVV) using different radius_hit.

    Args:
        methods (list, optional): tells which method(s) should be used: 'RVV' or
            'AVV' for normal vector voting (default) or 'SSVV' for vector and
            curvature tensor voting to estimate the principal directions and
            curvatures
        rhs (list, optional): wanted radius_hit parameter values (default 5-9)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        num_bins (int, optional): number of bins for the histogram (default 20)
    """
    subfolds = "torus/"
    type = "torus (major radius=25, minor radius=10)"
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "torus_rr25_csr10"
    for method in methods:
        plot_file = "{}{}.{}_rh{}-{}.kappa_1.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1])
        if x_range is not None:
            plot_file = os.path.splitext(plot_file)[0] + "_{}-{}.png".format(
                x_range[0], x_range[1])
        kappa_1_arrays = []
        labels = []
        for rh in rhs:
            df = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')
            kappa_1_array = df["kappa1"].tolist()
            kappa_1_arrays.append(kappa_1_array)
            label = r"rh={}".format(rh)
            labels.append(label)
        if x_range is None:
            # Find minimal and maximal value to set the X-range:
            min_value = min([min(d) for d in kappa_1_arrays])
            max_value = max([max(d) for d in kappa_1_arrays])
            x_range_1 = (min_value, max_value)
        else:
            x_range_1 = x_range
        plot_composite_line_hist(
            data_arrays=kappa_1_arrays,
            labels=labels,
            line_styles=['-', '-', '-', '-', '-'],
            markers=['*', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on {}".format(method, type),
            x_label=r"$\kappa_1$",
            y_label="Frequency",
            outfile=plot_file,
            num_bins=num_bins, x_range=x_range_1, max_val=None, normalize=False
        )


def plot_torus_curvature_errors_diff_rh(
        methods=["RVV", "AVV", "SSVV"], curvature="kappa1", rhs=list(range(5, 10)),
        x_range=None, y_range=(0, 1), num_bins=20,
        legend_loc="lower right", csv=None, voxel=False, *args, **kwargs):
    """
    Plots estimated curvature errors histograms on a torus surface
    for different methods and radius_hit.

    Args:
        methods (list, optional): tells which method(s) should be used
            (default=["RVV", "AVV", "SSVV"])
        curvature (str, optional): "kappa1" (default), "kappa2" or
            "mean_curvature"
        rhs (list, optional): wanted radius_hit parameter values (default 5-9)
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        num_bins (int, optional): number of bins for the histogram (default 20)
        legend_loc (str, optional): legend location (default 'lower right')
        csv (str, optional): csv file for saving cumulative histogram areas
        voxel (boolean, optional): if noisy cylinder should be plotted
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist
    """
    if voxel:
        subfolds = "torus/voxel/"
    else:
        subfolds = "torus/noise0/"
    fold = ("{}{}files4plotting/".format(FOLD, subfolds))
    plot_fold = ("{}{}plots/".format(FOLD, subfolds))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "torus_rr25_csr10"
    y_label = "Cumulative relative frequency"

    method_col = []
    rh_col = []
    hist_area_curv_col = []
    for i, method in enumerate(methods):
        print(method)
        curv_arrays = []
        labels = []
        for rh in rhs:
            df = pd.read_csv("{}{}.{}_rh{}.csv".format(
                fold, basename, method, rh), sep=';')
            curv_array = df["{}RelErrors".format(curvature)].tolist()
            curv_arrays.append(curv_array)
            label = r"rh={}".format(rh)
            labels.append(label)
        if x_range is None:
            # Find the maximal value to set the X-range:
            max_value = max([max(d) for d in curv_arrays])
            x_range_method = (0, max_value)
        else:
            x_range_method = x_range
        data_size = len(curv_arrays)
        line_styles = ['-'] * data_size
        markers_iter = cycle(MARKERS)
        markers = [next(markers_iter) for i in range(data_size)]
        colors_iter = cycle(COLORS)
        colors = [next(colors_iter) for i in range(data_size)]
        if curvature == "kappa1":
            formatted_curvature = r"$\kappa_1$"
        elif curvature == "kappa2":
            formatted_curvature = r"$\kappa_2$"
        else:
            formatted_curvature = "mean curvature"

        hist_areas = plot_composite_line_hist(
            data_arrays=curv_arrays, labels=labels,
            line_styles=line_styles, markers=markers, colors=colors,
            title=method,
            x_label="Relative {} error".format(formatted_curvature),
            y_label=y_label,
            outfile="{}{}.{}_rh{}-{}.{}_errors_range{}-{}.png".format(
                plot_fold, basename, method, rhs[0], rhs[-1], curvature,
                x_range_method[0], x_range_method[1]),
            num_bins=num_bins, x_range=x_range_method, y_range=y_range,
            normalize=True, cumulative=True, legend_loc=legend_loc,
            *args, **kwargs
        )
        method_col += [method] * data_size
        rh_col += rhs
        hist_area_curv_col += hist_areas
        i = hist_areas.index(max(hist_areas))
        print("Best performance for {} is for radius_hit={}".format(
            curvature, rhs[i]))
    if csv is not None:
        df = pd.DataFrame(index=None)
        df["method"] = method_col
        df["RadiusHit"] = rh_col
        df["hist_area_{}".format(curvature)] = hist_area_curv_col
        df.to_csv(csv, sep=';')


def plot_torus_kappa_1_and_2_t_1_and_2_errors(
        rhVV=9, rhSSVV=5, subdivisions=0, x_range_T=None, x_range_kappa=None,
        y_range=(0, 1), RorAVV="AVV", vertex_based=False, voxel=False):
    """
    Plots estimated kappa_1 and kappa_2 as well as t_1 and t_2 errors histograms
    on a torus surface for different methods (VV and SSVV) and an optimal
    radius_hit for each method.

    Args:
        rhVV (int, optional): radius_hit for VV (default 9)
        rhSSVV (int, optional): radius_hit for SSVV (default 5)
        subdivisions (int, optional): number of subdivisions in all three torus
            dimensions, if 0 (default), default subdivisions are used
        x_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        x_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        RorAVV (str, optional): RVV or AVV (default)
            weighted by triangle area also in the second step (principle
            directions and curvatures estimation)
        vertex_based (boolean, optional): if True (default False), curvature is
            calculated per triangle vertex instead of triangle center
        voxel (boolean, optional): if noisy torus should be plotted
    """
    if voxel:
        fold = ("{}torus/voxel/files4plotting/".format(FOLD))
    else:
        fold = ("{}torus/noise0/files4plotting/".format(FOLD))
    plot_fold = ("{}torus/plots/".format(FOLD))
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "torus_rr25_csr10"
    if subdivisions > 0:
        basename += "_subdivisions{}".format(subdivisions)
    if vertex_based:
        basename += "_vertex_based"
    for i in [1, 2]:
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
            x_range = (0, max([max(d) for d in data]))
            range_str = ""
        else:
            x_range = x_range_T
            range_str = "_range{}-{}".format(x_range[0], x_range[1])
        plot_composite_line_hist(
            data_arrays=data,
            labels=["SSVV rh={}".format(rhSSVV), "{} rh={}".format(
                RorAVV, rhVV)],
            line_styles=['-', '-'], markers=['^', 'v'],
            colors=['b', 'c'],
            title=None,
            x_label=r"$\vec t_{}\ error$".format(i),
            y_label="Cumulative relative frequency",
            outfile="{}{}.{}rh{}_vs_SSVVrh{}.T_{}_errors{}.png".format(
                plot_fold, basename, RorAVV, rhVV, rhSSVV, i, range_str),
            num_bins=20, normalize=True, cumulative=True,
            x_range=x_range,
            y_range=y_range,
            num_x_values=7
        )
        data = [VV_kappa_errors, VTK_kappa_errors, SSVV_kappa_errors]
        if x_range_kappa is None:
            x_range = (0, max([max(d) for d in data]))
            range_str = ""
        else:
            x_range = x_range_kappa
            range_str = "_range{}-{}".format(x_range[0], x_range[1])
        plot_composite_line_hist(
            data_arrays=data,
            labels=["{} rh={}".format(RorAVV, rhVV), "VTK",
                    "SSVV rh={}".format(rhSSVV), ],
            line_styles=['-', '-', '-'], markers=['v', 's', '^'],
            colors=['c', 'r', 'b'],
            title=None,
            x_label=r"$\kappa_{}\ relative\ error$".format(i),
            y_label="Cumulative relative frequency",
            outfile=("{}{}.{}rh{}_SSVVrh{}_vs_VTK.kappa_{}_errors{}.png".format(
                plot_fold, basename, RorAVV, rhVV, rhSSVV, i, range_str)),
            num_bins=20, normalize=True, cumulative=True,
            x_range=x_range,
            y_range=y_range
        )


def plot_torus_kappa_1_and_2_t_1_and_2_errors_allVV(
        rhRVV=9, rhAVV=9, rhSSVV=5, n=4, x_range_T=None, x_range_kappa=None,
        y_range=(0, 1), voxel=False, *args, **kwargs):
    """
    Plots estimated kappa_1 and kappa_2 as well as t_1 and t_2 errors histograms
    on a torus surface for different methods (RVV, AVV and SSVV) and an
    optimal radius_hit for each method. VTK is included for curvatures.

    Args:
        rhRVV (int, optional): radius_hit for RVV (default 9)
        rhAVV (int, optional): radius_hit for AVV (default 9)
        rhSSVV (int, optional): radius_hit for SSVV (default 5)
        n (int, optional): n for Mindboggle (MB) -m 0 (default 4)
        x_range_T (tuple, optional): a tuple of two values to limit the
            range at X axis of principal directions plots (default None)
        x_range_kappa (tuple, optional): a tuple of two values to limit the
            range at X axis of principal curvatures plots (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        voxel (boolean, optional): if noisy torus should be plotted
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist
    """
    if voxel:
        fold = "{}torus/voxel/files4plotting/".format(FOLD)
        mb_fold = "{}noisy_torus/".format(FOLDMB)
        mb_csv = ("noisy_torus_rr25_csr10.surface.mindboggle_m0_n{}_"
                  "curvature_errors.csv".format(n))
        fs_fold = "{}noisy_torus/csv/".format(FOLDFS)
        fs_csv = ("noisy_torus_rr25_csr10.freesurfer_"
                  "curvature_errors.csv".format(n))
        plot_fold = "{}noisy_torus/".format(FOLDPLOTS)
    else:
        fold = "{}torus/noise0/files4plotting/".format(FOLD)
        mb_fold = "{}smooth_torus/".format(FOLDMB)
        mb_csv = ("torus_rr25_csr10.surface.mindboggle_m0_n{}_"
                  "curvature_errors.csv".format(n))
        fs_fold = "{}smooth_torus/csv/".format(FOLDFS)
        fs_csv = ("smooth_torus_rr25_csr10.freesurfer_"
                  "curvature_errors.csv".format(n))
        plot_fold = "{}smooth_torus/".format(FOLDPLOTS)
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    basename = "torus_rr25_csr10"
    for i in [1, 2]:
        df = pd.read_csv("{}{}.SSVV_rh{}.csv".format(fold, basename, rhSSVV),
                         sep=';')
        SSVV_T_errors = df["T{}Errors".format(i)].tolist()
        SSVV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        df = pd.read_csv("{}{}.RVV_rh{}.csv".format(fold, basename, rhRVV),
                         sep=';')
        RVV_T_errors = df["T{}Errors".format(i)].tolist()
        RVV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        df = pd.read_csv("{}{}.AVV_rh{}.csv".format(fold, basename, rhAVV),
                         sep=';')
        AVV_T_errors = df["T{}Errors".format(i)].tolist()
        AVV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VTK_kappa_errors = pd.read_csv("{}{}.VTK.csv".format(
            fold, basename), sep=';')["kappa{}RelErrors".format(i)].tolist()

        MB_kappa_errors = pd.read_csv("{}{}".format(
            mb_fold, mb_csv), sep=';')["kappa{}RelErrors".format(i)].tolist()

        FS_kappa_errors = pd.read_csv("{}{}".format(
            fs_fold, fs_csv), sep=';')["kappa{}RelErrors".format(i)].tolist()

        # directions
        data = [RVV_T_errors, AVV_T_errors, SSVV_T_errors]
        outfile = "{}{}.RVVrh{}_AVVrh{}_vs_SSVVrh{}.T_{}_errors.png".format(
            plot_fold, basename, rhRVV, rhAVV, rhSSVV, i)
        if x_range_T is None:
            x_range = (0, max([max(d) for d in data]))
        else:
            x_range = x_range_T
            outfile = outfile.replace(
                ".png", "{}-{}.png".format(x_range[0], x_range[1]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=[r"RVV rh={}".format(rhRVV),
                    r"AVV rh={}".format(rhAVV),
                    r"SSVV rh={}".format(rhSSVV)],
            zorders=[1, 3, 2],
            line_styles=['-']*len(data), markers=['v', 'o', '^'],
            colors=['c', 'orange', 'b'],
            title=None,
            x_label=r"$\vec t_{}\ error$".format(i),
            y_label="Cumulative relative frequency",
            outfile=outfile,
            num_bins=20, normalize=True, cumulative=True,
            x_range=x_range, y_range=y_range,
            *args, **kwargs
        )

        # curvatures
        data = [RVV_kappa_errors, AVV_kappa_errors, SSVV_kappa_errors,
                VTK_kappa_errors, MB_kappa_errors, FS_kappa_errors]
        outfile = (
            "{}{}.RVVrh{}_AVVrh{}_SSVVrh{}_VTK_MBn{}_FS.kappa_{}_errors.png"
            .format(plot_fold, basename, rhRVV, rhAVV, rhSSVV, n, i))
        if x_range_kappa is None:
            x_range = (0, max([max(d) for d in data]))
        else:
            x_range = x_range_kappa
            outfile = outfile.replace(
                ".png", "{}-{}.png".format(x_range[0], x_range[1]))
        plot_composite_line_hist(
            data_arrays=data,
            labels=[r"RVV rh={}".format(rhRVV),
                    r"AVV rh={}".format(rhAVV),
                    r"SSVV rh={}".format(rhSSVV),
                    "VTK",
                    "MB n={}".format(n),
                    "FS"],
            zorders=[2, 4, 3, 1, 5, 6],
            line_styles=['-']*len(data), markers=['v', 'o', '^', 's', '*', '.'],
            colors=['c', 'orange', 'b', 'r', 'purple', 'g'],
            title=None,
            x_label=r"$Relative\ \kappa_{}\ error$".format(i),
            y_label="Cumulative relative frequency",
            outfile=outfile,
            num_bins=20, normalize=True, cumulative=True,
            x_range=x_range, y_range=y_range
        )


def plot_errors_different_parameter(
        errors_csv_file_template, parameter_name, parameter_values, plot_fold,
        curvature="kappa1", x_range=None, y_range=(0, 1), title=None, csv=None,
        *args, **kwargs):
    """

    Args:
        errors_csv_file_template: errors CSV file template with
            'X' instead of the parameter value
        parameter_name (str): parameter name
        parameter_values (list): wanted parameter values
        plot_fold (str): output folder
        curvature (str, optional): "kappa1" (default), "kappa2",
            "mean_curvature" or "both" for kappa1 and kappa2
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        title (str, optional): plot title (default None - no title)
        csv (str, optional): csv file for saving cumulative histogram areas
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist

    Returns:
        None
    """
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)

    base_filename = os.path.splitext(
        os.path.basename(errors_csv_file_template))[0]
    outfile = str(os.path.join(plot_fold, base_filename)) + '.png'
    outfile = outfile.replace("curvature", curvature)
    outfile = outfile.replace("X", "{}-{}".format(
        parameter_values[0], parameter_values[-1]))

    data = []
    labels = []
    for i in parameter_values:
        errors_csv_file = errors_csv_file_template.replace('X', str(i))

        # read in the curvature errors from the CSV file:
        df = pd.read_csv(errors_csv_file, sep=';')
        if curvature == "both":
            rel_kappa_1_errors = df["kappa1RelErrors"].tolist()
            rel_kappa_2_errors = df["kappa2RelErrors"].tolist()
            data.append(rel_kappa_1_errors + rel_kappa_2_errors)
        else:
            rel_curv_errors = df["{}RelErrors".format(curvature)].values
            data.append(rel_curv_errors)

        labels.append("{}={}".format(parameter_name, i))

    if x_range is None:
        x_range = (0, max([max(d) for d in data]))

    data_size = len(data)
    line_styles = ['-'] * data_size
    markers_iter = cycle(MARKERS)
    markers = [next(markers_iter) for i in range(data_size)]
    colors_iter = cycle(COLORS)
    colors = [next(colors_iter) for i in range(data_size)]
    if curvature == "kappa1":
        formatted_curvature = r"$\kappa_1$"
    elif curvature == "kappa2":
        formatted_curvature = r"$\kappa_2$"
    elif curvature == "mean_curvature":
        formatted_curvature = "mean curvature"
    else:
        formatted_curvature = r"$\kappa_1\ and\ \kappa_2$"

    hist_areas = plot_composite_line_hist(
        data_arrays=data, labels=labels,
        line_styles=line_styles, markers=markers, colors=colors,
        title=title,
        x_label="Relative {} error".format(formatted_curvature),
        y_label="Cumulative relative frequency",
        outfile=outfile,
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range, y_range=y_range,
        *args, **kwargs
    )
    i = hist_areas.index(max(hist_areas))
    print("Best performance for {} is for n={}".format(
        curvature, parameter_values[i]))

    if csv is not None:
        df = pd.DataFrame(index=None)
        df["n"] = parameter_values
        df["hist_area_{}".format(curvature)] = hist_areas
        df.to_csv(csv, sep=';')


def plot_mindboggle_errors(
        mb_errors_csv_file, avv_errors_csv_file, vtk_errors_csv_file, plot_fold,
        curvature="kappa1", x_range=None, y_range=(0, 1), title=None,
        *args, **kwargs):
    """
    Plots relative curvature errors for Mindboggle, AVV and VTK.

    Args:
        mb_errors_csv_file (str): Mindboggle errors CSV file
        avv_errors_csv_file (str): AVV errors CSV file
        vtk_errors_csv_file (str): VTK errors CSV file
        plot_fold (str): output folder
        curvature (str, optional): "kappa1" (default), "kappa2" or
            "mean_curvature"
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default (0, 1))
        title (str, optional): plot title (default None - no title)
        *args: other arguments passed to plot_composite_line_hist
        **kwargs: other keyword arguments passed to plot_composite_line_hist

    Returns:
        None
    """
    if not os.path.exists(plot_fold):
        os.makedirs(plot_fold)
    # read in the curvature errors from the CSV files:
    mb_df = pd.read_csv(mb_errors_csv_file, sep=';')
    mb_rel_curv_errors = mb_df["{}RelErrors".format(curvature)].values
    avv_df = pd.read_csv(avv_errors_csv_file, sep=';')
    avv_rel_curv_errors = avv_df["{}RelErrors".format(curvature)].values
    vtk_df = pd.read_csv(vtk_errors_csv_file, sep=';')
    vtk_rel_curv_errors = vtk_df["{}RelErrors".format(curvature)].values

    data = [avv_rel_curv_errors, mb_rel_curv_errors, vtk_rel_curv_errors]

    base_filename = os.path.splitext(os.path.basename(mb_errors_csv_file))[0]
    outfile = str(os.path.join(plot_fold, base_filename)) + '.png'
    outfile = outfile.replace("curvature", curvature)
    if x_range is None:
        x_range = (0, max([max(d) for d in data]))
    else:
        outfile = outfile.replace(
            ".png", "{}-{}.png".format(x_range[0], x_range[1]))

    if curvature == "kappa1":
        formatted_curvature = r"$\kappa_1$"
    elif curvature == "kappa2":
        formatted_curvature = r"$\kappa_2$"
    else:
        formatted_curvature = "mean curvature"

    plot_composite_line_hist(
        data_arrays=data, labels=["AVV", "MB", "VTK"],
        zorders=[2, 3, 1],
        line_styles=['-', '-', '-'], markers=['o', '*', 's'],
        colors=['orange', 'g', 'r'],
        title=title,
        x_label="Relative {} error".format(formatted_curvature),
        y_label="Cumulative relative frequency",
        outfile=outfile,
        num_bins=20, normalize=True, cumulative=True,
        x_range=x_range, y_range=y_range,
        *args, **kwargs
    )


def plot_peak_curvature_diff_rh(
        df, segmentation="filled", method="AVV", curvature="kappa1",
        weights=None, x_label=r"$\kappa_{1}\ (nm^{-1})$", x_range=None,
        y_range=None, num_bins=20, title=None, plot_fold=None):
    """
    Plots curvature data of a cER sub-surface with a peak, generated
    using a regular or compartment segmentation, estimated by AVV or SSVV and
    different radius_hit

    Args:
        df (pandas.DataFrame): DataFrame containing all the data in special
            format, generated in read_in_and_plot_peak_curvatures
        segmentation (str, optional): segmentation used for surface generation,
            default "filled"
        method (str, optional): curvature method used, default "AVV"
        curvature (str, optional): curvature to be plotted, default "kappa1"
        weights (str, optional): if given, curvatures will be weighted by this
            column from the DataFrame (default None)
        x_label (str, optional): X-label, default r"$\kappa_{1}\ (nm^{-1})$"
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        num_bins (int, optional): number of bins for the histogram (default 20)
        title (str, optional): plot title (default None - no title)
        plot_fold (str, optional): folder where to save the plot, if wished
            (default None)

    Returns:
        None
    """
    if plot_fold is not None:
        plot_file = "{}peak_{}_{}_{}_diffRadiusHit.png".format(
            plot_fold, segmentation, method, curvature)
        if x_range is not None:
            plot_file = os.path.splitext(plot_file)[0] + "_X{}-{}.png".format(
                x_range[0], x_range[1])
        if y_range is not None:
            plot_file = os.path.splitext(plot_file)[0] + "_Y{}-{}.png".format(
                y_range[0], y_range[1])
        if weights is not None:
            plot_file = (os.path.splitext(plot_file)[0] +
                         "_weighted_by_{}.png".format(weights))
    else:
        plot_file = None
    curvatures_arrays = []
    labels = []
    if weights is not None:
        weights_arrays = []
    else:
        weights_arrays = None
    for radius_hit in [2, 5, 10, 15, 20]:
        selection = df.query('(segmentation==@segmentation) & (method==@method)'
                             '& (radius_hit==@radius_hit)')
        curvatures = selection[curvature].values[0]
        curvatures_array = np.array(curvatures).astype(np.float)
        curvatures_arrays.append(curvatures_array)
        label = r"rh={}".format(radius_hit)
        labels.append(label)
        if weights is not None:
            weights = selection[weights].values[0]
            weights_array = np.array(weights).astype(np.float)
            weights_arrays.append(weights_array)

    if x_range is None:
        # Find minimal and maximal value to set the X-range:
        min_value = min([min(d) for d in curvatures_arrays])
        max_value = max([max(d) for d in curvatures_arrays])
        x_range = (min_value, max_value)

    y_label = "Relative frequency"
    if weights is not None:
        y_label += " weighted by area"

    plot_composite_line_hist(
        data_arrays=curvatures_arrays, labels=labels,
        line_styles=['-', '-', '-', '-', '-'],
        markers=['*', 'v', '^', 's', 'o'],
        colors=['b', 'c', 'g', 'y', 'r'],
        x_label=x_label, x_range=x_range, y_label=y_label, y_range=y_range,
        normalize=True, num_bins=num_bins, title=title, outfile=plot_file,
        weights_arrays=weights_arrays
    )


def read_in_and_plot_peak_curvatures(x_range=None, y_range=None, num_bins=20,
                                     weights=None):
    """
    Reads in curvature data of a cER sub-surface with a peak, generated
    using a regular or compartment segmentation, estimated by AVV or SSVV and
    different radius_hit and plots the curvatures.

    Args:
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        y_range (tuple, optional): a tuple of two values to limit the range
            at Y axis (default None)
        num_bins (int, optional): number of bins for the histogram (default 20)
        weights (str, optional): if given, curvatures will be weighted by this
            property (default None)

    Returns:
        None
    """
    folder = FOLDPEAKS
    plot_fold = FOLDPEAKSPLOTS
    segmentations = ["unfilled_using_peakplus", "filled"]
    methods = ["AVV", "SSVV"]
    radius_hits = [2, 5, 10, 15, 20]

    # Read in and combine the data to one data frame
    super_df = pd.DataFrame(
        columns=["segmentation", "method", "radius_hit", "kappa1", "kappa2",
                 "gauss_curvature", "mean_curvature", "shape_index",
                 "curvedness", "triangleAreas"])
    i = 0
    for segmentation in segmentations:  # 2
        for method in methods:  # 2
            for radius_hit in radius_hits:  # 5
                csv_name = "TCB_180830_l2_t2peak.cER.{}_rh{}.csv".format(
                    method, radius_hit)
                csv_p = PurePath(folder, segmentation, csv_name)
                csv = str(csv_p)
                df = pd.read_csv(csv, sep=";", index_col=0)
                row = [segmentation, method, radius_hit]
                for key in list(df.keys()):
                    row.append(df[key].tolist())
                super_df.loc[i] = row  # explicit index, in this case the same
                # as implicit one, starting from 0
                i += 1

    # Plot the peak curvatures
    for i, segmentation in enumerate(segmentations):
        for method in methods:
            plot_peak_curvature_diff_rh(
                super_df, segmentation, method, curvature="kappa1",
                x_label=r"$\kappa_{1}\ (nm^{-1})$", x_range=x_range,
                y_range=y_range, num_bins=num_bins, title=None,
                plot_fold=plot_fold, weights=weights)


def read_in_and_plot_surface_curvature(
        num_bins=20, weights=None, curvature="kappa1",
        x_label=r"$\kappa_{1}\ (nm^{-1})$"):
    """
    Reads in curvature data of a cER surface, generated using a compartment
    segmentation, estimated by AVV and radius_hit=10 and plots the curvature.

    Args:
        num_bins (int, optional): number of bins for the histogram (default 20)
        weights (str, optional): if given, curvatures will be weighted by this
            column of the DataFrame (default None)
        curvature (str, optional): curvature to be plotted, default "kappa1"
        x_label (str, optional): X-label, default r"$\kappa_{1}\ (nm^{-1})$"

    Returns:
        None
    """
    folder = FOLDCER
    plot_fold = FOLDPEAKSPLOTS
    method = "AVV"
    radius_hit = 10
    csv = "{}TCB_180830_l2_t2half.cER.{}_rh{}_excluding1borders.csv".format(
        folder, method, radius_hit)
    plot_file = "{}TCB_180830_l2_t2half.cER.{}_rh{}_{}.png".format(
        plot_fold, method, radius_hit, curvature)
    y_label = "Frequency"

    df = pd.read_csv(csv, sep=";", index_col=0)
    curvatures = df[curvature]

    if weights is not None:
        plot_file = (os.path.splitext(plot_file)[0] +
                     "_weighted_by_{}.png".format(weights))
        y_label += " weighted by area"
        weights = df[weights]

    plot_line_hist(
        curvatures, weights=weights, num_bins=num_bins, title=None,
        x_label=x_label, y_label=y_label,
        x_range=None, label=None, ls='--', marker='^', c='g',
        normalize=False, cumulative=False, outfile=plot_file)


def read_in_and_plot_surface_curvatures(
        x_range=None, num_bins=20, weights=None, method="AVV", radius_hit=10,
        borders=0):
    """
    Reads in curvature data of a cER surface, generated using a compartment
    segmentation, estimated by the given method and radius_hit, and plots the
    curvature (minimal and maximal principal as well as curvedness).

    Args:
        x_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        num_bins (int, optional): number of bins for the histogram (default 20)
        weights (str, optional): if given, curvatures will be weighted by this
            column of the DataFrame (default None)
        method (str, optional): which method to plot (default "AVV")
        radius_hit (int, optional): which radius_hit to plot (default 10)
        borders (int, optional): how much to exclude from borders in nm
            (default 0)

    Returns:
        None
    """
    folder = FOLDCER
    plot_fold = FOLDPEAKSPLOTS
    if borders == 0:
        csv = "{}TCB_180830_l2_t2half.cER.{}_rh{}.csv".format(
            folder, method, radius_hit)
    else:
        csv = ("{}TCB_180830_l2_t2half.cER.{}_rh{}_excluding{}borders.csv"
               .format(folder, method, radius_hit, borders))
    plot_file = "{}TCB_180830_l2_t2half.cER.{}_rh{}_excluding{}borders" \
                "_curvatures.png".format(plot_fold, method, radius_hit, borders)
    if x_range is not None:
        plot_file = os.path.splitext(plot_file)[0] + "_{}-{}.png".format(
            x_range[0], x_range[1])
    y_label = "Relative frequency"

    df = pd.read_csv(csv, sep=";", index_col=0)
    if weights is not None:
        plot_file = (os.path.splitext(plot_file)[0] +
                     "_weighted_by_{}.png".format(weights))
        y_label += " weighted by area"
        weights = df[weights]
    curvatures_arrays = []
    weights_arrays = []
    for curvature in ["kappa1", "kappa2", "curvedness"]:
        curvatures = df[curvature]
        curvatures_arrays.append(curvatures)
        weights_arrays.append(weights)

    labels = [r"$\kappa_{1}$", r"$\kappa_{2}$", "Curvedness"]
    x_label = r"$AVV\ curvature\ (nm^{-1})$"
    if x_range is None:
        # Find minimal and maximal value to set the X-range:
        min_value = min([min(d) for d in curvatures_arrays])
        max_value = max([max(d) for d in curvatures_arrays])
        x_range = (min_value, max_value)

    plot_composite_line_hist(
        labels,
        line_styles=['-', '-', '-'],
        markers=['^', 's', 'o'],
        colors=['g', 'y', 'r'],
        x_label=x_label, y_label=y_label, title=None,
        data_arrays=curvatures_arrays, weights_arrays=weights_arrays,
        num_bins=num_bins, x_range=x_range, y_range=None,
        normalize=True, cumulative=False, outfile=plot_file)


def plot_excluding_borders(method="AVV", radius_hit=10):
    """
    Plots maximal absolute curvatures and percent surface depending on distance
    filtered from border.

    Args:
        method (str, optional): which method to plot (default "AVV")
        radius_hit (int, optional): which radius_hit to plot (default 10)

    Returns:
        None
    """
    folder = FOLDCER
    plot_fold = FOLDPEAKSPLOTS
    plot_file = "{}TCB_180830_l2_t2half.cER.{}_rh{}_excluding_borders.png"\
        .format(plot_fold, method, radius_hit)
    csv = "{}TCB_180830_l2_t2half.cER.{}_rh{}.csv".format(
        folder, method, radius_hit)

    df = pd.read_csv(csv, sep=";")
    kappa1 = df["kappa1"]
    kappa2 = df["kappa2"]
    areas = df["triangleAreas"]
    border_dist = [0]
    percent_surface = [100]
    max_kappa1 = [max(kappa1)]
    min_kappa2 = [abs(min(kappa2))]
    for b in range(1, 8):
        border_dist.append(b)
        csv_b = os.path.splitext(csv)[0] + "_excluding{}borders.csv".format(b)
        df_b = pd.read_csv(csv_b, sep=";")
        kappa1_b = df_b["kappa1"]
        kappa2_b = df_b["kappa2"]
        areas_b = df_b["triangleAreas"]
        percent = float(sum(areas_b)) / float(sum(areas)) * 100.0
        percent_surface.append(percent)
        max_kappa1.append(max(kappa1_b))
        min_kappa2.append(abs(min(kappa2_b)))
        print("{}% after excluding {} nm from border, max(kappa_1)={}, "
              "min(kappa_2)={}".format(
                percent, b, max(kappa1_b), min(kappa2_b)))

    fig, ax1 = plt.subplots()
    rcParams['axes.spines.top'] = True
    rcParams['axes.spines.right'] = True
    color = 'red'
    ax1.set_xlabel('Distance filtered from border (nm)')
    ax1.set_ylabel(r'Absolute extreme curvature $(nm^{-1})$', color=color)
    ax1.plot(border_dist, max_kappa1, color=color, marker='^',
             linestyle='None', label=r"|maximal $\kappa_1$|")
    ax1.plot(border_dist, min_kappa2, color=color, marker='v',
             linestyle='None', label=r"|minimal $\kappa_2$|")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper right', fancybox=True, framealpha=0.5)
    ax1.set_ylim(0, max(max(max_kappa1), max(min_kappa2)) + 0.1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'blue'
    ax2.set_ylabel('Surface (%)',
                   color=color)  # we already handled the x-label with ax1
    ax2.plot(border_dist, percent_surface, color=color, marker='*',
             linestyle='None')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(min(percent_surface) - 1, max(percent_surface) + 1)

    plt.xlim(-0.1, 7.1)
    plt.tight_layout()
    plt.tick_params(direction='in')

    fig.savefig(plot_file)
    print("The plot was saved as {}".format(plot_file))


if __name__ == "__main__":
    # **Benchmark data**
    # Fig 4:
    # plot_plane_normals(x_range=(0, 0.4))
    # plot_plane_normals(x_range=(0, 0.4), rand_dir=True)

    # plot_plane_normals_different_noise(rh=8, x_range=(0, 0.4))
    # plot_plane_normals_different_noise(rh=8, x_range=(0, 0.1))
    # plot_plane_normals_different_noise(rh=4, x_range=(0, 0.4))
    # plot_plane_normals_different_noise(rh=4, x_range=(0, 0.1))

    plot_plane_normals_different_noise_and_rh(x_range=(0, 0.4))
    plot_plane_normals_different_noise_and_rh(x_range=(0, 0.1))

    # # voxel sphere - Fig 5
    # kwargs = {}
    # kwargs["num_x_values"] = 6
    # kwargs["fontsize"] = 23
    # kwargs["ncol"] = 2
    # for method in ["AVV", "RVV", "SSVV"]:
    #     for curvature in ["kappa1", "kappa2", "both"]:
    #         plot_sphere_curvature_errors_diff_rh(
    #             voxel=True, methods=[method], curvature=curvature,
    #             rhs=list(range(5, 11)), x_range=(0, 0.5), csv=join(
    #                 FOLD, "sphere/voxel/files4plotting",
    #                 "sphere_r10_{}_RadiusHit5-10_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # # Fig 8:
    # for r in [10, 30]:
    #     plot_sphere_curvature_errors_allVV(  # both curvatures
    #         r=r, rhRVV=10, rhAVV=10, rhSSVV=8, n=2, voxel=True,
    #         x_range=(0, 0.7), legend_loc="center right"
    #     )
    # plot_sphere_curvature_errors_allVV(  # both curvatures
    #     r=30, rhRVV=28, rhAVV=28, rhSSVV=28, n=2, voxel=True, onlyVV=True,
    #     x_range=(0, 0.7), legend_loc="center right"
    # )
    #
    # # smooth torus
    # kwargs = {}
    # kwargs["num_x_values"] = 6
    # for method in ["AVV", "RVV"]:
    #     for curvature in ["kappa1", "kappa2", "mean_curvature"]:
    #         plot_torus_curvature_errors_diff_rh(
    #             methods=[method], curvature=curvature,
    #             rhs=list(range(4, 11)), csv=join(
    #                 FOLD, "torus/noise0/files4plotting",
    #                 "torus_rr25_csr10_{}_RadiusHit4-10_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # for method in ["SSVV"]:
    #     for curvature in ["kappa1", "kappa2", "mean_curvature"]:
    #         plot_torus_curvature_errors_diff_rh(
    #             methods=[method], curvature=curvature,
    #             rhs=list(range(4, 10)), csv=join(
    #                 FOLD, "torus/noise0/files4plotting",
    #                 "torus_rr25_csr10_{}_RadiusHit4-9_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # # Fig 6:
    # plot_torus_kappa_1_and_2_t_1_and_2_errors_allVV(
    #     rhRVV=10, rhAVV=9, rhSSVV=6, n=4, x_range_kappa=(0, 0.08),
    #     legend_loc="lower right")  # range for kappa1
    # plot_torus_kappa_1_and_2_t_1_and_2_errors_allVV(
    #     rhRVV=10, rhAVV=9, rhSSVV=6, n=4, x_range_kappa=(0, 1.4),
    #     legend_loc="lower right")  # range for kappa2
    #
    # # smooth sphere
    # kwargs = {}
    # kwargs["num_x_values"] = 6
    # kwargs["fontsize"] = 23
    # for method in ["AVV", "RVV"]:
    #     for curvature in ["kappa1", "kappa2", "both", "mean_curvature"]:
    #         plot_sphere_curvature_errors_diff_rh(
    #             methods=[method], curvature=curvature,
    #             rhs=list(range(5, 11)), csv=join(
    #                 FOLD, "sphere/noise0/files4plotting",
    #                 "sphere_r10_{}_RadiusHit5-10_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # for method in ["SSVV"]:
    #     for curvature in ["kappa1", "kappa2", "both", "mean_curvature"]:
    #         plot_sphere_curvature_errors_diff_rh(
    #             methods=[method], curvature=curvature,
    #             rhs=list(range(5, 10)), csv=join(
    #                 FOLD, "sphere/noise0/files4plotting",
    #                 "sphere_r10_{}_RadiusHit5-9_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # # Fig 7:
    # for r in [10, 20]:
    #     plot_sphere_curvature_errors_allVV(  # both curvatures
    #         r=r, rhRVV=10, rhAVV=10, rhSSVV=9, n=2, voxel=False,
    #         x_range=(0, 0.25), **kwargs
    #     )
    # for r in [10, 20]:
    #     plot_sphere_curvature_errors_allVV(  # both curvatures
    #         r=r, rhRVV=10, rhAVV=10, rhSSVV=9, n=2, voxel=False,
    #         x_range=(0, 8), **kwargs
    #     )
    #
    # # smooth cylinder
    # kwargs = {}
    # kwargs["fontsize"] = 23
    # plot_cylinder_curvature_errors_diff_rh(
    #     x_range=(0, 0.25), methods=["AVV", "SSVV"], rhs=list(range(2, 11)),
    #     csv=join(FOLD, "cylinder/noise0/files4plotting",
    #              "cylinder_r10_AVV_SSVV_RadiusHit2-10_xmax0.25.csv"))
    # for method in ["AVV", "RVV", "SSVV"]:
    #     for curvature in ["kappa1", "kappa2", "mean_curvature"]:
    #         plot_cylinder_curvature_errors_diff_rh(
    #             methods=[method], curvature=curvature,
    #             rhs=list(range(3, 11)), csv=join(
    #                 FOLD, "cylinder/noise0/files4plotting",
    #                 "cylinder_r10_{}_RadiusHit3-10_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # # Fig 9:
    # plot_cylinder_t_2_and_kappa_1_errors(
    #     x_range_T=(0, 0.006), x_range_kappa=(0, 1.0),
    #     exclude_borders=0, rhVV=5, rhSSVV=6, n=2, **kwargs)
    # plot_cylinder_t_2_and_kappa_1_errors(
    #     x_range_T=(0, 0.006), x_range_kappa=(0, 1.0),
    #     exclude_borders=5, rhVV=5, rhSSVV=6, **kwargs)
    #
    # # **Real data**
    # read_in_and_plot_peak_curvatures(x_range=(-0.1, 0.4), y_range=(0, 0.8),
    #                                  num_bins=25)
    # for method in ["AVV"]:  # "NVV", "RVV", "SSVV"
    #     # plot_excluding_borders(method=method)
    #     read_in_and_plot_surface_curvatures(
    #         num_bins=25, method=method, borders=0, x_range=(-0.1, 0.15))

    # **Extra tests**
    # *voxel torus*
    # kwargs = {}
    # kwargs["num_x_values"] = 6
    # for method in ["AVV", "RVV"]:
    #     for curvature in ["kappa1", "kappa2", "mean_curvature"]:
    #         plot_torus_curvature_errors_diff_rh(
    #             voxel=True, methods=[method], curvature=curvature,
    #             rhs=range(4, 11), csv=join(
    #                 FOLD, "torus/voxel/files4plotting",
    #                 "torus_rr25_csr10_{}_RadiusHit4-10_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # for method in ["SSVV"]:
    #     for curvature in ["kappa1", "kappa2", "mean_curvature"]:
    #         plot_torus_curvature_errors_diff_rh(
    #             voxel=True, methods=[method], curvature=curvature,
    #             rhs=range(4, 10), csv=join(
    #                 FOLD, "torus/voxel/files4plotting",
    #                 "torus_rr25_csr10_{}_RadiusHit4-9_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # plot_torus_kappa_1_and_2_t_1_and_2_errors_allVV(
    #     # best for kappa1, no legend, because overlaps
    #     rhRVV=10, rhAVV=10, rhSSVV=9, n=2, voxel=True)  # , x_range_kappa=(0, 4)
    # plot_torus_kappa_1_and_2_t_1_and_2_errors_allVV(
    #     rhRVV=7, rhAVV=8, rhSSVV=4, n=2, voxel=True)  # best for kappa2
    #
    # *voxel cylinder*
    # for method in ["AVV", "RVV", "SSVV"]:
    #     for curvature in ["kappa1", "kappa2", "mean_curvature"]:
    #         plot_cylinder_curvature_errors_diff_rh(
    #             voxel=True, methods=[method], curvature=curvature,
    #             rhs=range(3, 11), csv=join(
    #                 FOLD, "cylinder/voxel/files4plotting",
    #                 "cylinder_r10_{}_RadiusHit3-10_{}_area.csv".format(
    #                     method, curvature)), **kwargs)
    # plot_cylinder_t_2_and_kappa_1_errors(
    #     x_range_T=(0, 0.01), x_range_kappa=(0, 4.0),
    #     voxel=True, exclude_borders=0, rhVV=9, rhSSVV=8, n=2)
    # plot_cylinder_t_2_and_kappa_1_errors(
    #     x_range_T=(0, 0.01), x_range_kappa=(0, 4.0),
    #     voxel=True, exclude_borders=5, rhVV=9, rhSSVV=8)
    #
    # *Mindboggle*
    # surface_bases = [
    #     'torus_rr25_csr10.surface.', 'noisy_torus_rr25_csr10.surface.',
    #     'smooth_sphere_r10.surface.', 'noisy_sphere_r10.surface.',
    #     'cylinder_r10_h25.surface.', 'noisy_cylinder_r10_h25.surface.'
    #     ]
    # subfolds = ['smooth_torus', 'noisy_torus',
    #             'smooth_sphere', 'noisy_sphere',
    #             'smooth_cylinder', 'noisy_cylinder'
    #             ]
    # m = 0
    # ns = range(1, 11)
    # curvature = "kappa1"  # "kappa2", "both", "mean_curvature"
    # plot_fold_n_choice = FOLDPLOTS + "/mindboggle_n_choice/n1-10"
    #
    # for surface_base, subfold in zip(surface_bases, subfolds):
    #     fold = join(FOLDMB, subfold)
    #     out_base = "{}mindboggle_m{}_nX".format(surface_base, m)
    #     mb_errors_csv_file_template = join(
    #         fold, "{}_curvature_errors.csv".format(out_base))
    #     n_area_csv_file = join(
    #         fold, "{}_{}_area.csv".format(out_base, curvature))
    #     kwargs = {}
    #     if surface_base.startswith("smooth_torus"):
    #         kwargs["num_x_values"] = 5
    #     plot_errors_different_parameter(
    #         mb_errors_csv_file_template, 'n', ns, plot_fold_n_choice,
    #         curvature=curvature, csv=n_area_csv_file, **kwargs)
    #
    # best_ns = [5, 10, 8, 4, 2, 8]  # for mean curvature
    # # best_radius_hits = [6, 9, 10, 9, 4, 8]
    # avv_errors_csv_files = [
    #     "torus/noise0/files4plotting/torus_rr25_csr10.AVV_rh6.csv",
    #     "torus/voxel/files4plotting/torus_rr25_csr10.AVV_rh9.csv",
    #     "sphere/noise0/files4plotting/sphere_r10.AVV_rh10.csv",
    #     "sphere/voxel/files4plotting/sphere_r10.AVV_rh9.csv",
    #     "cylinder/noise0/files4plotting/cylinder_r10_h25_eb0.AVV_rh4.csv",
    #     "cylinder/voxel/files4plotting/cylinder_r10_h25_eb0.AVV_rh8.csv"]
    # vtk_errors_csv_files = [
    #     "torus/noise0/files4plotting/torus_rr25_csr10.VTK.csv",
    #     "torus/voxel/files4plotting/torus_rr25_csr10.VTK.csv",
    #     "sphere/noise0/files4plotting/sphere_r10.VTK.csv",
    #     "sphere/voxel/files4plotting/sphere_r10.VTK.csv",
    #     "cylinder/noise0/files4plotting/cylinder_r10_h25_eb0.VTK.csv",
    #     "cylinder/voxel/files4plotting/cylinder_r10_h25_eb0.VTK.csv"]
    #
    # for i, surface_base in enumerate(surface_bases):
    #     fold = join(FOLDMB, subfolds[i])
    #     out_base = "{}mindboggle_m{}_n{}".format(
    #         surface_base, m, best_ns[i])
    #     mb_errors_csv_file = join(
    #         fold, "{}_curvature_errors.csv".format(out_base))
    #     avv_errors_csv_file = join(FOLD, avv_errors_csv_files[i])
    #     vtk_errors_csv_file = join(FOLD, vtk_errors_csv_files[i])
    #     plot_mindboggle_errors(
    #         mb_errors_csv_file=mb_errors_csv_file,
    #         avv_errors_csv_file=avv_errors_csv_file,
    #         vtk_errors_csv_file=vtk_errors_csv_file,
    #         plot_fold=FOLDPLOTS, curvature=curvature, x_range=(0, 4))
    #
    # *FreeSurfer*
    # surface_bases = [
    #     # 'torus_rr25_csr10.', 'noisy_torus_rr25_csr10.',
    #     # 'smooth_sphere_r10.',
    #     'noisy_sphere_r10.',
    #     # 'cylinder_r10_h25.surface.', 'noisy_cylinder_r10_h25.surface.'
    #     ]
    # subfolds = [# 'smooth_torus', 'noisy_torus',
    #             # 'smooth_sphere',
    #             'noisy_sphere',
    #             # 'smooth_cylinder', 'noisy_cylinder'
    #             ]
    # a_s = range(0, 11)
    # curvature = "both"  # "kappa1", "kappa2", "mean_curvature"
    # plot_fold_a_choice = join(FOLDPLOTS, "freesurfer_a_choice/a0-10")
    #
    # for surface_base, subfold in zip(surface_bases, subfolds):
    #     fold = join(FOLDFS, subfold, "csv")
    #     out_base = "{}freesurfer_aX".format(surface_base)
    #     fs_errors_csv_file_template = join(
    #         fold, "{}_curvature_errors.csv".format(out_base))
    #     a_area_csv_file = join(
    #         fold, "{}_{}_area.csv".format(out_base, curvature))
    #     kwargs = {}
    #     if surface_base.startswith("smooth_torus"):
    #         kwargs["num_x_values"] = 5
    #     plot_errors_different_parameter(
    #         fs_errors_csv_file_template, 'a', a_s, plot_fold_a_choice,
    #         curvature=curvature, csv=a_area_csv_file, **kwargs)
