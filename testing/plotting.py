import matplotlib.pyplot as plt
import numpy as np

from pysurf_compact import pysurf_io as io
from pysurf_compact import pexceptions


def plot_hist(value_list, num_bins, title, xlabel="Value", ylabel="Counts",
              value_range=None):
    """
    Plots a histogram of the values with the given number of bins and plot
    title.

    Args:
        value_list:
        num_bins:
        title:
        xlabel:
        ylabel:
        value_range:

    Returns:

    """
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
    plt.show()


def plot_line_hist(value_list, num_bins, title, xlabel="Value", ylabel="Counts",
                   value_range=None):
    """
    Plots a line histogram of the values with the given number of bins and plot
    title.

    Args:
        value_list:
        num_bins:
        title:
        xlabel:
        ylabel:
        value_range:

    Returns:

    """
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
    plt.show()


def plot_double_line_hist(value_list1, value_list2, num_bins, title,
                          xlabel="Value", ylabel="Counts", value_range=None,
                          label1="values 1", label2="values 2"):
    """
    Plots a line histogram of two value lists with the given number of bins and
    plot title. # TODO ADD A LEGEND!

    Args:
        value_list1:
        value_list2:
        num_bins:
        title:
        xlabel:
        ylabel:
        value_range:
        label1:
        label2:

    Returns:

    """
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
    plt.show()


# Plotting for vector voting tests
def main():
    radius = 20
    inverse = False
    g_max = 13
    epsilon = 0
    eta = 0

    fold = '/fs/pool/pool-ruben/Maria/curvature/synthetic_volumes/good/'
    fold2 = '{}files4plotting/'.format(fold)
    if inverse:
        inverse_str = "inverse_"
    else:
        inverse_str = ""
    base_filename = "{}{}sphere_r{}".format(fold2, inverse_str, radius)
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
    plot_hist(kappa_1_values, 30,
              "Sphere with radius {}, NVV with g_max={}, epsilon={}, eta={}"
              .format(radius, g_max, epsilon, eta),
              xlabel="Maximal principal curvature", ylabel="Number of vertices")
    plot_hist(kappa_1_errors, 10,
              "Sphere with radius={}, NVV with g_max={}, epsilon={}, eta={}"
              .format(radius, g_max, epsilon, eta),
              xlabel="Maximal principal curvature error (%)",
              ylabel="Number of vertices", value_range=(0, 100))
    # plot_hist(kappa_2_values, 30,
    #           "Sphere with radius={}, NVV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Minimal principal curvature", ylabel="Number of vertices")
    # plot_hist(kappa_2_errors, 10,
    #           "Sphere with radius={}, NVV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Minimal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100))

    # plot_hist(vtk_max_curvature_values, 30,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Maximal principal curvature", ylabel="Number of vertices")
    plot_hist(vtk_max_curvature_errors, 10,
              "Sphere with radius={}, averaging of VTK values for triangle "
              "vertices".format(radius),
              xlabel="Maximal principal curvature error (%)",
              ylabel="Number of vertices", value_range=(0, 100))
    # plot_hist(vtk_min_curvature_values, 30,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Minimal principal curvature", ylabel="Number of vertices")
    # plot_hist(vtk_min_curvature_errors, 10,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Minimal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100))
    
    plot_double_line_hist(
        kappa_1_errors, vtk_max_curvature_errors, 10,
        "Sphere with radius={}, NVV with g_max={}, epsilon={}, eta={}\n"
        "vs. averaging of VTK values for triangle vertices".format(
            radius, g_max, epsilon, eta),
        xlabel="Maximal principal curvature error (%)",
        ylabel="Number of vertices", value_range=(0, 100),
        label1="NVV", label2="VTK")

if __name__ == "__main__":
    main()
