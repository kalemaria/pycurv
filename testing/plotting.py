import matplotlib.pyplot as plt
import numpy as np
import os

from pysurf_compact import pysurf_io as io
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
        xlabel (str): X axis label
        ylabel (str): Y axis label
        value_range: a tuple of two values to limit the range at X axis
        outfile (str): if given (default None), the plot with be saved as a file
            under this path

    Returns:
        None
    """
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
                   value_range=None, outfile=None):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        value_list: a list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str): X axis label
        ylabel (str): Y axis label
        value_range: a tuple of two values to limit the range at X axis
        outfile (str): if given (default None), the plot with be saved as a file
            under this path

    Returns:
        None
    """
    fig = plt.figure()
    counts = []
    bin_edges = []
    if value_range is None:
        counts, bin_edges = np.histogram(value_list, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        counts, bin_edges = np.histogram(value_list, bins=num_bins,
                                         range=value_range)
    else:
        error_msg = "Range has to be a tuple of two numbers (min, max)."
        raise pexceptions.PySegInputError(expr='plot_hist', msg=error_msg)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bincenters, counts, ls='-', marker='.')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)


def plot_double_line_hist(value_list1, value_list2, num_bins, title,
                          xlabel="Value", ylabel="Counts", value_range=None,
                          label1="values 1", label2="values 2", outfile=None):
    """
    Plots a line histogram of two value lists with the given number of bins and
    plot title.

    Args:
        value_list1: first list of numerical values
        value_list2: second list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str): X axis label
        ylabel (str): Y axis label
        value_range: a tuple of two values to limit the range at X axis
        label1 (str): legend label for the first value list
        label2 (str): legend label for the second value list
        outfile (str): if given (default None), the plot with be saved as a file
            under this path

    Returns:
        None
    """
    fig = plt.figure()
    counts1 = []
    bin_edges1 = []
    counts2 = []
    bin_edges2 = []
    if value_range is None:
        counts1, bin_edges1 = np.histogram(value_list1, bins=num_bins)
        counts2, bin_edges2 = np.histogram(value_list2, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        counts1, bin_edges1 = np.histogram(value_list1, bins=num_bins,
                                           range=value_range)
        counts2, bin_edges2 = np.histogram(value_list2, bins=num_bins,
                                           range=value_range)
    else:
        error_msg = "Range has to be a tuple of two numbers (min, max)."
        raise pexceptions.PySegInputError(expr='plot_hist', msg=error_msg)
    bincenters1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
    bincenters2 = 0.5 * (bin_edges2[1:] + bin_edges2[:-1])
    plt.plot(bincenters1, counts1, ls='-', marker='.', c="blue", label=label1)
    plt.plot(bincenters2, counts2, ls='--', marker='.', c="red", label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)


def main():
    """
    Main method for plotting Normal Vector Voting test results.

    Returns:
        None
    """
    radius = 20
    inverse = False
    g_max = 13
    epsilon = 0
    eta = 0

    fold = '/fs/pool/pool-ruben/Maria/curvature/synthetic_volumes/good/'
    files_fold = '{}files4plotting/'.format(fold)
    plots_fold = '{}plots/'.format(fold)
    if inverse:
        inverse_str = "inverse_"
    else:
        inverse_str = ""
    base_filename = "{}{}sphere_r{}".format(files_fold, inverse_str, radius)
    kappa_1_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1.txt'.format(
        base_filename, g_max, epsilon, eta)
    kappa_2_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2.txt'.format(
        base_filename, g_max, epsilon, eta)
    kappa_1_errors_file = (
        '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
            base_filename, g_max, epsilon, eta))
    kappa_2_errors_file = (
        '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
            base_filename, g_max, epsilon, eta))

    vtk_max_curvature_file = ('{}.VTK.max_curvature.txt'.format(base_filename))
    vtk_min_curvature_file = ('{}.VTK.min_curvature.txt'.format(base_filename))
    vtk_max_curvature_errors_file = ('{}.VTK.max_curvature_errors.txt'.format(
        base_filename))
    vtk_min_curvature_errors_file = ('{}.VTK.min_curvature_errors.txt'.format(
        base_filename))

    # Reading in the values from files:
    kappa_1_values = io.read_values_from_file(kappa_1_file)
    kappa_2_values = io.read_values_from_file(kappa_2_file)
    kappa_1_errors = io.read_values_from_file(kappa_1_errors_file)
    kappa_2_errors = io.read_values_from_file(kappa_2_errors_file)
    vtk_max_curvature_values = io.read_values_from_file(vtk_max_curvature_file)
    vtk_min_curvature_values = io.read_values_from_file(vtk_min_curvature_file)
    vtk_max_curvature_errors = io.read_values_from_file(
        vtk_max_curvature_errors_file)
    vtk_min_curvature_errors = io.read_values_from_file(
        vtk_min_curvature_errors_file)

    # Plotting:
    root, ext = os.path.splitext(kappa_1_file)
    head, tail = os.path.split(root)
    kappa_1_plot = "{}{}.png".format(plots_fold, tail)
    # print len(kappa_1_values)  # debug
    plot_hist(kappa_1_values, 30,
              "Sphere with radius {}, VV with g_max={}, epsilon={}, eta={}"
              .format(radius, g_max, epsilon, eta),
              xlabel="Maximal principal curvature", ylabel="Number of vertices",
              outfile=kappa_1_plot)
    root, ext = os.path.splitext(kappa_1_errors_file)
    head, tail = os.path.split(root)
    kappa_1_errors_plot = "{}{}.png".format(plots_fold, tail)
    # print len(kappa_1_errors)  # debug
    plot_hist(kappa_1_errors, 10,
              "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}"
              .format(radius, g_max, epsilon, eta),
              xlabel="Maximal principal curvature error (%)",
              ylabel="Number of vertices", value_range=(0, 100),
              outfile=kappa_1_errors_plot)
    # plot_hist(kappa_2_values, 30,
    #           "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Minimal principal curvature", ylabel="Number of vertices")
    # plot_hist(kappa_2_errors, 10,
    #           "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Minimal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100))

    root, ext = os.path.splitext(vtk_max_curvature_file)
    head, tail = os.path.split(root)
    vtk_max_curvature_plot = "{}{}.png".format(plots_fold, tail)
    # print len(vtk_max_curvature_values)  # debug
    plot_hist(vtk_max_curvature_values, 30,
              "Sphere with radius={}, averaging of VTK values for triangle "
              "vertices".format(radius),
              xlabel="Maximal principal curvature", ylabel="Number of vertices",
              outfile=vtk_max_curvature_plot)
    root, ext = os.path.splitext(vtk_max_curvature_errors_file)
    head, tail = os.path.split(root)
    vtk_max_curvature_errors_plot = "{}{}.png".format(plots_fold, tail)
    # print len(vtk_max_curvature_errors)  # debug
    plot_hist(vtk_max_curvature_errors, 10,
              "Sphere with radius={}, averaging of VTK values for triangle "
              "vertices".format(radius),
              xlabel="Maximal principal curvature error (%)",
              ylabel="Number of vertices", value_range=(0, 100),
              outfile=vtk_max_curvature_errors_plot)
    # plot_hist(vtk_min_curvature_values, 30,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Minimal principal curvature", ylabel="Number of vertices")
    # plot_hist(vtk_min_curvature_errors, 10,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Minimal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100))

    tail = "{}sphere_r{}.VV_g_max{}_epsilon{}_eta{}vsVTK.kappa_1_errors".format(
        inverse_str, radius, g_max, epsilon, eta)
    vv_vtk_kappa_1_errors_plot = "{}{}.png".format(plots_fold, tail)
    plot_double_line_hist(
        kappa_1_errors, vtk_max_curvature_errors, 10,
        "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}\n"
        "vs. averaging of VTK values for triangle vertices".format(
            radius, g_max, epsilon, eta),
        xlabel="Maximal principal curvature error (%)",
        ylabel="Number of vertices", value_range=(0, 100),
        label1="VV", label2="VTK", outfile=vv_vtk_kappa_1_errors_plot)

if __name__ == "__main__":
    main()
