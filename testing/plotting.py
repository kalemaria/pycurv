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


def plot_plane_normal_errors(half_size, res=30, noise=10,
                             g_max=5, epsilon=0, eta=0):
    """
    A method for plotting plane triangle normal errors as estimated by a
    modification of Normal Vector Voting (VV) algorithm vs. calculated by VTK.

    Args:
        half_size (int): half size of the plane (from center to an edge)
        res (int, optional): resolution (number of divisions) in X and Y
            axes (default 30)
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 10), the noise
            is added on triangle vertex coordinates in Z dimension
        g_max (float, optional): geodesic neighborhood radius in length unit
            of the graph, here voxels; if positive (default 0.0) this g_max
            will be used and k will be ignored
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2) and "no preferred orientation" (class 3),
            default 0

    Returns:
        None
    """
    base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
    if res == 0:
        fold = '{}synthetic_volumes/plane/noise{}/'.format(base_fold, noise)
    else:
        fold = '{}synthetic_surfaces/plane/res{}_noise{}/'.format(
            base_fold, res, noise)
    files_fold = '{}files4plotting/'.format(fold)
    base_filename = "{}plane_half_size{}".format(files_fold, half_size)
    vtk_normal_errors_file = '{}.VTK.normal_errors.txt'.format(
        base_filename)
    vv_normal_errors_file = (
        '{}.VV_g_max{}_epsilon{}_eta{}.normal_errors.txt'.format(
            base_filename, g_max, epsilon, eta))

    # Reading in the error values from files:
    if not os.path.exists(vtk_normal_errors_file):
        print ("File {} not found!".format(vtk_normal_errors_file))
        exit(0)
    if not os.path.exists(vv_normal_errors_file):
        print ("File {} not found!".format(vv_normal_errors_file))
        exit(0)

    vtk_normal_errors = io.read_values_from_file(vtk_normal_errors_file)
    vv_normal_errors = io.read_values_from_file(vv_normal_errors_file)

    # Plotting:
    plots_fold = '{}plots/'.format(fold)
    if not os.path.exists(plots_fold):
        os.makedirs(plots_fold)
    base = ("plane_half_size{}.VV_g_max{}_epsilon{}_eta{}vsVTK.normal_errors"
            .format(half_size, g_max, epsilon, eta))
    vv_vtk_normal_errors_plot = "{}{}.png".format(plots_fold, base)
    plot_double_line_hist(
        vv_normal_errors, vtk_normal_errors, 10,
        "Comparison for Plane ({}% noise)".format(noise),
        # "Plane with half-size={}, VV with g_max={}, epsilon={}, eta={}\n"
        # "vs. averaging of VTK values for triangle vertices".format(
        #     half_size, g_max, epsilon, eta),
        xlabel="Normal Orientation Error (%)",
        ylabel="Number of Vertices", value_range=(0, 100),
        label1="VV ({})".format(g_max), label2="VTK",
        outfile=vv_vtk_normal_errors_plot)
    print ("The plot was saved as {}".format(vv_vtk_normal_errors_plot))


def plot_sphere_curv_errors(radius, inverse=False, res=50,
                            g_max=5, epsilon=0, eta=0):
    """
    A method for plotting sphere principal curvature errors as estimated by a
    modification of Normal Vector Voting (VV) algorithm vs. calculated by VTK.

    Args:
        radius (int): radius of the sphere
        inverse (boolean, optional): if True (default False), the sphere
            will have normals pointing outwards (negative curvature), else
            the other way around
        res (int): if > 0 (default 50) determines how many stripes (and then
            triangles) the sphere has (longitude and latitude), the surface
            is generated directly using VTK; If 0 first a sphere mask is
            generated and then surface using gen_surface function
        g_max (float, optional): geodesic neighborhood radius in length unit
            of the graph, here voxels; if positive (default 5) this g_max
            will be used and k will be ignored
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2) and "no preferred orientation" (class 3, see
            Notes), default 0

    Returns:
        None
    """
    base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
    if res == 0:
        fold = '{}synthetic_volumes/sphere/'.format(base_fold)
    else:
        fold = '{}synthetic_surfaces/sphere/res{}/'.format(base_fold, res)
    files_fold = '{}files4plotting/'.format(fold)
    if inverse:
        inverse_str = "inverse_"
    else:
        inverse_str = ""
    base_filename = "{}{}sphere_r{}".format(files_fold, inverse_str, radius)
    # kappa_1_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1.txt'.format(
    #     base_filename, g_max, epsilon, eta)
    # kappa_2_file = '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2.txt'.format(
    #     base_filename, g_max, epsilon, eta)
    kappa_1_errors_file = (
        '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
            base_filename, g_max, epsilon, eta))
    kappa_2_errors_file = (
        '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
            base_filename, g_max, epsilon, eta))

    # vtk_kappa_1_file = ('{}.VTK.kappa_1.txt'.format(base_filename))
    # vtk_kappa_2_file = ('{}.VTK.kappa_2.txt'.format(base_filename))
    vtk_kappa_1_errors_file = ('{}.VTK.kappa_1_errors.txt'.format(
        base_filename))
    vtk_kappa_2_errors_file = ('{}.VTK.kappa_2_errors.txt'.format(
        base_filename))

    # Reading in the values from files:
    if not os.path.exists(kappa_1_errors_file):
        print ("File {} not found!".format(kappa_1_errors_file))
        exit(0)
    if not os.path.exists(vtk_kappa_1_errors_file):
        print ("File {} not found!".format(vtk_kappa_1_errors_file))
        exit(0)
    if not os.path.exists(kappa_2_errors_file):
        print ("File {} not found!".format(kappa_2_errors_file))
        exit(0)
    if not os.path.exists(vtk_kappa_2_errors_file):
        print ("File {} not found!".format(vtk_kappa_2_errors_file))
        exit(0)
    # kappa_1_values = io.read_values_from_file(kappa_1_file)
    # kappa_2_values = io.read_values_from_file(kappa_2_file)
    kappa_1_errors = io.read_values_from_file(kappa_1_errors_file)
    kappa_2_errors = io.read_values_from_file(kappa_2_errors_file)
    # vtk_kappa_1_values = io.read_values_from_file(vtk_kappa_1_file)
    # vtk_kappa_2_values = io.read_values_from_file(vtk_kappa_2_file)
    vtk_kappa_1_errors = io.read_values_from_file(vtk_kappa_1_errors_file)
    vtk_kappa_2_errors = io.read_values_from_file(vtk_kappa_2_errors_file)

    # Plotting:
    plots_fold = '{}plots/'.format(fold)
    if not os.path.exists(plots_fold):
        os.makedirs(plots_fold)
    # root, ext = os.path.splitext(kappa_1_file)
    # head, tail = os.path.split(root)
    # kappa_1_plot = "{}{}.png".format(plots_fold, tail)
    # plot_hist(kappa_1_values, 30,
    #           "Sphere with radius {}, VV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Maximal principal curvature",
    #           ylabel="Number of vertices",
    #           outfile=kappa_1_plot)
    # root, ext = os.path.splitext(kappa_1_errors_file)
    # head, tail = os.path.split(root)
    # kappa_1_errors_plot = "{}{}.png".format(plots_fold, tail)
    # plot_hist(kappa_1_errors, 10,
    #           "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Maximal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100),
    #           outfile=kappa_1_errors_plot)
    # plot_hist(kappa_2_values, 30,
    #           "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Minimal principal curvature",
    #           ylabel="Number of vertices")
    # plot_hist(kappa_2_errors, 10,
    #           "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}"
    #           .format(radius, g_max, epsilon, eta),
    #           xlabel="Minimal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100))

    # root, ext = os.path.splitext(vtk_kappa_1_file)
    # head, tail = os.path.split(root)
    # vtk_max_curvature_plot = "{}{}.png".format(plots_fold, tail)
    # plot_hist(vtk_kappa_1_values, 30,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Maximal principal curvature",
    #           ylabel="Number of vertices",
    #           outfile=vtk_max_curvature_plot)
    # root, ext = os.path.splitext(vtk_kappa_1_errors_file)
    # head, tail = os.path.split(root)
    # vtk_max_curvature_errors_plot = "{}{}.png".format(plots_fold, tail)
    # plot_hist(vtk_kappa_1_errors, 10,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Maximal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100),
    #           outfile=vtk_max_curvature_errors_plot)
    # plot_hist(vtk_kappa_2_values, 30,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Minimal principal curvature",
    #           ylabel="Number of vertices")
    # plot_hist(vtk_kappa_2_errors, 10,
    #           "Sphere with radius={}, averaging of VTK values for triangle "
    #           "vertices".format(radius),
    #           xlabel="Minimal principal curvature error (%)",
    #           ylabel="Number of vertices", value_range=(0, 100))

    base = "{}sphere_r{}.VV_g_max{}_epsilon{}_eta{}vsVTK".format(
        inverse_str, radius, g_max, epsilon, eta)
    kappa_1_errors_plot = "{}{}.kappa_1_errors.png".format(plots_fold, base)
    kappa_2_errors_plot = "{}{}.kappa_2_errors.png".format(plots_fold, base)
    if inverse is True:
        title = "Comparison for Inverse Sphere (0% noise)"
    else:
        title = "Comparison for Sphere (0% noise)"
    # old title:
    # "Sphere with radius={}, VV with g_max={}, epsilon={}, eta={}\n"
    # "vs. averaging of VTK values for triangle vertices".format(
    #     radius, g_max, epsilon, eta),
    plot_double_line_hist(
        kappa_1_errors, vtk_kappa_1_errors, 10, title,
        xlabel="Maximal Principal Curvature Error (%)",
        ylabel="Number of Vertices", value_range=(0, 100),
        label1="VV ({})".format(g_max), label2="VTK",
        outfile=kappa_1_errors_plot)
    print ("The plot was saved as {}".format(kappa_1_errors_plot))
    plot_double_line_hist(
        kappa_2_errors, vtk_kappa_2_errors, 10, title,
        xlabel="Minimal Principal Curvature Error (%)",
        ylabel="Number of Vertices", value_range=(0, 100),
        label1="VV ({})".format(g_max), label2="VTK",
        outfile=kappa_2_errors_plot)
    print ("The plot was saved as {}".format(kappa_2_errors_plot))

if __name__ == "__main__":
    # for n in [5, 10]:
    #     for g in [3, 5]:
    #         plot_plane_normal_errors(10, res=30, noise=n, g_max=g)
    plot_sphere_curv_errors(5, inverse=False, res=50, g_max=1)
    plot_sphere_curv_errors(5, inverse=True, res=50, g_max=1)
