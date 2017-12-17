import matplotlib.pyplot as plt
import numpy as np
import os

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


def add_line_hist(value_list, num_bins, value_range=None, max_val=None,
                  label=None, ls='-', marker='^', c='b'):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        value_list: a list of numerical values
        num_bins (int): number of bins for the histogram
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        max_val (float, optional): if given (default None), values higher than
            this value will be set to this value
        label (str, optional): legend label for the value list (default None)
        ls (str, optional): line style (default '-')
        marker (str, optional): plotting character (default '^')
        c (str, optional): color (default 'b' for blue)

    Returns:
        None
    """
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
    plt.plot(bincenters, counts, ls=ls, marker=marker, c=c, label=label,
             linewidth=2)


def plot_errors(
        error_files, labels, line_styles, markers, colors,  # all lists
        title, xlabel, ylabel, num_bins=10, value_range=(0, 100), max_val=100,
        outfile=None, ):
    # TODO docstring from the docstrings of the two previous functions
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()

    for i, error_file in enumerate(error_files):
        # Reading in the error values from files:
        if not os.path.exists(error_file):
            print ("File {} not found!".format(error_file))
            exit(0)
        errors = np.loadtxt(error_file)
        add_line_hist(
            errors, num_bins, value_range=value_range, max_val=max_val,
            label=labels[i], ls=line_styles[i], marker=markers[i], c=colors[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)
    print ("The plot was saved as {}".format(outfile))


if __name__ == "__main__":
    plot_fold = "/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/plots/"
    # plane normals
    n = 5  # noise in %
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "plane/res10_noise{}/files4plotting/".format(n))
    plot_errors(
        error_files=[fold + "plane_half_size10.VCTV_rh4.normal_errors.txt",
                     fold + "plane_half_size10.VCTV_rh2.normal_errors.txt",
                     fold + "plane_half_size10.VTK.normal_errors.txt"],
        labels=["VV rh=4", "VV rh=2", "VTK"],
        line_styles=['-', '--', ':'], markers=['^', 'v', 's'],
        colors=['b', 'c', 'r'],
        title="Plane ({}% noise)".format(n),
        xlabel="Normal orientation error (%)", ylabel="Number of triangles",
        outfile="{}plane_res10_noise{}.VV_vs_VTK.normal_errors.png".format(
            plot_fold, n)
    )
    # cylinder T_2 and kappa_1 errors
    n = 0  # noise in %
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "cylinder/noise0/files4plotting/".format(n))
    plot_errors(
        error_files=[fold + "cylinder_r10_h25.VCTV_rh8.T_h_errors.txt",
                     fold + "cylinder_r10_h25.VV_rh8.T_h_errors.txt"],
        labels=["VCTV rh=8", "VV rh=8"],
        line_styles=['-', '--'], markers=['^', 'v'],
        colors=['b', 'c'],
        title="Cylinder ({}% noise)".format(n),
        xlabel="Minimal principal direction error (%)",
        ylabel="Number of triangles",
        outfile="{}cylinder_r10_noise{}.VV_VCTV_rh8.T_2_errors.png".format(
            plot_fold, n)
    )
    plot_errors(
        error_files=[fold + "cylinder_r10_h25.VCTV_rh8.kappa_1_errors.txt",
                     fold + "cylinder_r10_h25.VVCF_rh8.kappa_1_errors.txt",
                     fold + "cylinder_r10_h25.VV_rh8.kappa_1_errors.txt",
                     fold + "cylinder_r10_h25.VTK.kappa_1_errors.txt"],
        labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
        line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Cylinder ({}% noise)".format(n),
        xlabel="Maximal principal curvature error (%)",
        ylabel="Number of triangles",
        outfile=("{}cylinder_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
                 "kappa_1_errors.png".format(plot_fold, n))
    )
    # inverse cylinder T_1 and kappa_2 errors
    n = 0  # noise in %
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "cylinder/noise0/files4plotting/".format(n))
    plot_errors(
        error_files=[
            fold + "inverse_cylinder_r10_h25.VCTV_rh8.T_h_errors.txt",
            fold + "inverse_cylinder_r10_h25.VV_rh8.T_h_errors.txt"],
        labels=["VCTV rh=8", "VV rh=8"],
        line_styles=['-', '--'], markers=['^', 'v'],
        colors=['b', 'c'],
        title="Inverse cylinder ({}% noise)".format(n),
        xlabel="Maximal principal direction error (%)",
        ylabel="Number of triangles",
        outfile="{}inverse_cylinder_r10_noise{}.VV_VCTV_rh8.T_1_errors.png"
                .format(plot_fold, n)
    )
    plot_errors(
        error_files=[
            fold + "inverse_cylinder_r10_h25.VCTV_rh8.kappa_2_errors.txt",
            fold + "inverse_cylinder_r10_h25.VVCF_rh8.kappa_2_errors.txt",
            fold + "inverse_cylinder_r10_h25.VV_rh8.kappa_2_errors.txt",
            fold + "inverse_cylinder_r10_h25.VTK.kappa_2_errors.txt"],
        labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
        line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Inverse cylinder ({}% noise)".format(n),
        xlabel="Minimal principal curvature error (%)",
        ylabel="Number of triangles",
        outfile=("{}inverse_cylinder_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
                 "kappa_2_errors.png".format(plot_fold, n))
    )
    # sphere kappa_1 errors
    n = 0  # noise in %
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "sphere/ico1280_noise{}/files4plotting/".format(n))
    plot_errors(
        error_files=[fold + "sphere_r10.VCTV_rh8.kappa_1_errors.txt",
                     fold + "sphere_r10.VVCF_rh8.kappa_1_errors.txt",
                     fold + "sphere_r10.VV_rh8.kappa_1_errors.txt",
                     fold + "sphere_r10.VTK.kappa_1_errors.txt"],
        labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
        line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Sphere (icosahedron 1280, {}% noise)".format(n),
        xlabel="Maximal principal curvature error (%)",
        ylabel="Number of triangles",
        outfile=("{}icosphere_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
                 "kappa_1_errors.png".format(plot_fold, n))
    )
    # inverse sphere kappa_1 errors
    n = 0  # noise in %
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/"
            "sphere/ico1280_noise{}/files4plotting/".format(n))
    plot_errors(
        error_files=[fold + "inverse_sphere_r10.VCTV_rh8.kappa_1_errors.txt",
                     fold + "inverse_sphere_r10.VVCF_rh8.kappa_1_errors.txt",
                     fold + "inverse_sphere_r10.VV_rh8.kappa_1_errors.txt",
                     fold + "inverse_sphere_r10.VTK.kappa_1_errors.txt"],
        labels=["VCTV rh=8", "VVCF rh=8", "VV rh=8", "VTK"],
        line_styles=['-', '-.', '--', ':'], markers=['^', 'o', 'v', 's'],
        colors=['b', 'g', 'c', 'r'],
        title="Inverse sphere (icosahedron 1280, {}% noise)".
              format(n),
        xlabel="Maximal principal curvature error (%)",
        ylabel="Number of triangles",
        outfile=("{}inverse_icosphere_r10_noise{}.VV_VVCF_VCTV_rh8_vs_VTK."
                 "kappa_1_errors.png".format(plot_fold, n))
    )
