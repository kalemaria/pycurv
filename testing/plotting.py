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


def plot_plane_normals(n=10):
    """ Plots estimated normals errors by VV versus original face normals
    (calculated by VTK) on a noisy plane surface.

    Args:
        n (int, optional): noise in % (default 10)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "plane/res10_noise{}/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "plane/res10_noise{}/plots/".format(n))
    basename = "plane_half_size10"
    VCTV_rh8_normal_errors = pd.read_csv("{}{}.VCTV_rh8.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    VCTV_rh4_normal_errors = pd.read_csv("{}{}.VCTV_rh4.csv".format(
        fold, basename), sep=';')["normalErrors"].tolist()
    VTK_normal_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                    sep=';')["normalErrors"].tolist()
    data = [VCTV_rh8_normal_errors, VCTV_rh4_normal_errors, VTK_normal_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VV rh=8", "VV rh=4", "VTK"],
        line_styles=['-', '--', ':'], markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Plane ({}% noise)".format(n),
        xlabel="Normal orientation error", ylabel="Cumulative frequency",
        outfile="{}plane_res10_noise{}.VV_vs_VTK.normal_errors_bin20_cum_freq"
                ".png".format(plot_fold, n),
        num_bins=20, freq=True, cumulative=True,
        value_range=(0, max([max(d) for d in data]))
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
    basename = "cylinder_r10_h25"
    for method in ['VV', 'VCTV', 'VVCF']:  # TODO 'VVCF_50points'?
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
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on cylinder ({}% noise)".format(method, n),
            xlabel="Estimated maximal principal curvature",
            ylabel="Counts",
            outfile=("{}{}_noise{}.{}_rh5-9.kappa_1.png".format(
                plot_fold, basename, n, method)),
            num_bins=5, value_range=None, max_val=None, freq=False
        )


def plot_cylinder_T_2_and_kappa_1_errors(n=0):
    """Plots estimated kappa_2 and T_1 errors histograms on a cylinder surface
    for different methods (VV, VVCF and VCTV) and optimal RadiusHit for each
    method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "cylinder/noise0/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "cylinder/noise0/plots/".format(n))
    basename = "cylinder_r10_h25"
    df = pd.read_csv("{}{}.VCTV_rh5.csv".format(fold, basename), sep=';')
    VCTV_T_2_errors = df["T2Errors"].tolist()
    VCTV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VVCF_kappa_1_errors = pd.read_csv("{}{}.VVCF_rh5.csv".format(
        fold, basename), sep=';')["kappa1RelErrors"].tolist()

    df = pd.read_csv("{}{}.VV_rh5.csv".format(fold, basename), sep=';')
    VV_T_2_errors = df["T2Errors"].tolist()
    VV_kappa_1_errors = df["kappa1RelErrors"].tolist()

    VTK_kappa_1_errors = pd.read_csv("{}{}.VTK.csv".format(fold, basename),
                                     sep=';')["kappa1RelErrors"].tolist()
    data = [VCTV_T_2_errors, VV_T_2_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV rh=5", "VV rh=5"],
        line_styles=['-', '--'], markers=['^', 'v'],
        colors=['b', 'c'],
        title="Cylinder ({}% noise)".format(n),
        xlabel="Minimal principal direction error",
        ylabel="Cumulative frequency",
        outfile="{}{}_noise{}.VV_VCTV_rh5.T_2_errors_bins20_cum_freq."
                "png".format(plot_fold, basename, n),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data]))
    )
    data = [VCTV_kappa_1_errors, VVCF_kappa_1_errors,
            VV_kappa_1_errors, VTK_kappa_1_errors]
    plot_composite_line_hist(
        data_arrays=data,
        labels=["VCTV rh=5", "VVCF rh=5", "VV rh=5", "VTK"],
        line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Cylinder ({}% noise)".format(n),
        xlabel="Maximal principal curvature relative error",
        ylabel="Cumulative frequency",
        outfile=("{}{}_noise{}.VV_VVCF50p_VCTV_rh5_vs_VTK.kappa_1_errors_bins20"
                 "_cum_freq.png".format(plot_fold, basename, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data]))
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


def plot_sphere_kappa_1_and_2_diff_rh(n=0):
    """Plots estimated kappa_1 and kappa_2 values for an icosahedron surface
     by different methods (VV, VVCF and VCTV) using different RadiusHit.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "sphere/ico1280_noise{}/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "sphere/ico1280_noise{}/plots/".format(n))
    basename = "sphere_r10"
    for method in ['VVCF_50points']:  # 'VV', 'VCTV',
        kappa_1_arrays = []
        kappa_2_arrays = []
        kappas_arrays = []
        labels = []
        for rh in [2, 3, 3.5, 4, 5]:  # range(2, 7)
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
            title="{} on sphere (icosahedron 1280, {}% noise)".format(
                method, n),
            xlabel="Estimated maximal principal curvature",
            ylabel="Counts",
            outfile=("{}ico{}_noise{}.{}_rh2-5.kappa_1.png".format(
                plot_fold, basename, n, method)),  # TODO change rh range
            num_bins=5, value_range=None, max_val=None, freq=False
        )
        plot_composite_line_hist(  # kappa_2
            data_arrays=kappa_2_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on sphere (icosahedron 1280, {}% noise)".format(
                method, n),
            xlabel="Estimated minimal principal curvature",
            ylabel="Counts",
            outfile=("{}ico{}_noise{}.{}_rh2-5.kappa_2.png".format(
                plot_fold, basename, n, method)),  # TODO change rh range
            num_bins=5, value_range=None, max_val=None, freq=False
        )
        plot_composite_line_hist(  # kappa_1 + kappa_2
            data_arrays=kappas_arrays,
            labels=labels,
            line_styles=['-.', '-.', '--', '-', ':'],
            markers=['x', 'v', '^', 's', 'o'],
            colors=['b', 'c', 'g', 'y', 'r'],
            title="{} on sphere (icosahedron 1280, {}% noise)".format(
                method, n),
            xlabel="Estimated principal curvatures",
            ylabel="Counts",
            outfile=("{}ico{}_noise{}.{}_rh2-5.kappa_1_and_2.png".format(
                plot_fold, basename, n, method)),  # TODO change rh range
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


def plot_sphere_kappa_1_and_2_errors(n=0):
    """
    Plots estimated kappa_1 and kappa_2 errors histograms on an icosahedron
    sphere surface for different methods (VV, VVCF and VCTV) and an optimal
    RadiusHit for each method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "sphere/ico1280_noise{}/files4plotting/".format(n))
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "sphere/ico1280_noise{}/plots/".format(n))
    basename = "sphere_r10"

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
        title="Sphere (icosahedron 1280, {}% noise)".format(n),
        xlabel="Principal curvatures relative error",
        ylabel="Cumulative frequency",
        outfile=("{}icosphere_r10_noise{}.VV_VVCF50p_VCTV_vs_VTK."
                 "kappa_1_and_2_errors_20bins_cum_freq.png".format(
                  plot_fold, n)),
        num_bins=20, freq=True, cumulative=True,  # max_val=1
        value_range=(0, max([max(d) for d in data]))
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


def plot_torus_kappa_1_and_2_diff_rh(n=0):
    """Plots estimated kappa_1 and kappa_2 values for a torus surface
     by different methods (VV, VVCF and VCTV) using different RadiusHit.

    Args:
        n (int, optional): noise in % (default 0)
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
                outfile=("{}{}.{}_rh5-9.kappa_{}.png".format(  # TODO rh range!
                    plot_fold, basename, method, i)),
                num_bins=5, value_range=None, max_val=None, freq=False
            )


def plot_torus_kappa_1_and_2_T_1_and_2_errors(n=0):
    """
    Plots estimated kappa_1 and kappa_2 as well as T_1 and T_2 errors histograms
    on a torus surface for different methods (VV, VVCF and VCTV) and an optimal
    RadiusHit for each method.

    Args:
        n (int, optional): noise in % (default 0)
    """
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "torus/files4plotting/")
    plot_fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
                 "torus/plots/")
    basename = "torus_rr25_csr10"
    principal_components = {1: "maximal", 2: "minimal"}
    for i in principal_components.keys():
        df = pd.read_csv("{}{}.VCTV_rh8.csv".format(fold, basename), sep=';')
        VCTV_T_errors = df["T{}Errors".format(i)].tolist()
        VCTV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VVCF_kappa_errors = pd.read_csv("{}{}.VVCF_50points_rh3.csv".format(
            fold, basename), sep=';')["kappa{}RelErrors".format(
                i)].tolist()

        df = pd.read_csv("{}{}.VV_rh8.csv".format(fold, basename), sep=';')
        VV_T_errors = df["T{}Errors".format(i)].tolist()  # same for VVCF
        VV_kappa_errors = df["kappa{}RelErrors".format(i)].tolist()

        VTK_kappa_errors = pd.read_csv("{}{}.VTK.csv".format(
            fold, basename), sep=';')["kappa{}RelErrors".format(
                i)].tolist()
        data = [VCTV_T_errors, VV_T_errors]
        plot_composite_line_hist(
            data_arrays=data,
            labels=["VCTV rh=8", "VV rh=8"],
            line_styles=['-', '--'], markers=['^', 'v'],
            colors=['b', 'c'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal direction error".format(
                principal_components[i]),
            ylabel="Cumulative frequency",
            outfile="{}{}.VV_VCTV_rh8.T_{}_errors_bins20_cum_freq."
                    "png".format(plot_fold, basename, i),
            num_bins=20, freq=True, cumulative=True,
            value_range=(0, max([max(d) for d in data]))
            # , max_val=1
        )
        data = [VCTV_kappa_errors, VVCF_kappa_errors,
                VV_kappa_errors, VTK_kappa_errors]
        plot_composite_line_hist(
            data_arrays=data,
            labels=["VCTV rh=8", "VVCF rh=3", "VV rh=8", "VTK"],
            line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
            colors=['b', 'g', 'c', 'r'],
            title="Torus (major radius=25, minor radius=10)",
            xlabel="{} principal curvature relative error".format(
                principal_components[i]),
            ylabel="Cumulative frequency",
            outfile=("{}{}.VV_VVCF_50points_VCTV_rh8_vs_VTK.kappa_{}_errors_"
                     "bins20_cum_freq.png".format(plot_fold, basename, i)),
            num_bins=20, freq=True, cumulative=True,
            value_range=(0, max([max(d) for d in data]))
            # , max_val=1
        )


if __name__ == "__main__":
    # plot_plane_normals()
    # plot_sphere_kappa_1_and_2_fitting_diff_num_points()
    # plot_sphere_kappa_1_and_2_diff_rh()
    # plot_sphere_kappa_1_and_2_errors()
    # plot_inverse_sphere_kappa_1_and_2_errors()
    # plot_cylinder_kappa_1_diff_rh()  # TODO later VVCF_50points with lower rh
    # plot_cylinder_T_2_and_kappa_1_errors()  # TODO VVCF_50points, optimal rh
    # plot_inverse_cylinder_T_1_and_kappa_2_errors()  # TODO VVCF_50points, optimal rh
    # plot_torus_kappa_1_and_2_diff_rh()
    plot_torus_kappa_1_and_2_T_1_and_2_errors()
