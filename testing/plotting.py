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
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Counts")
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

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
                   value_range=None, label=None, outfile=None):
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

    Returns:
        None
    """
    fig = plt.figure()
    if value_range is None:
        counts, bin_edges = np.histogram(value_list, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        counts, bin_edges = np.histogram(value_list, bins=num_bins,
                                         range=value_range)
    else:
        error_msg = "Range has to be a tuple of two numbers (min, max)."
        raise pexceptions.PySegInputError(expr='plot_hist', msg=error_msg)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bincenters, counts, ls='-', marker='.', label=label)
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
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Counts")
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        label1 (str, optional): legend label for the first value list (default
            "values 1")
        label2 (str, optional): legend label for the second value list (default
            "values 2")
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

    Returns:
        None
    """
    fig = plt.figure()
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


def plot_triple_line_hist(
        value_list1, value_list2, value_list3, num_bins, title, xlabel="Value",
        ylabel="Counts", value_range=None, label1="values 1", label2="values 2",
        label3="values 3", outfile=None):
    """
    Plots a line histogram of two value lists with the given number of bins and
    plot title.

    Args:
        value_list1: first list of numerical values
        value_list2: second list of numerical values
        value_list3: third list of numerical values
        num_bins (int): number of bins for the histogram
        title (str): title of the plot
        xlabel (str, optional): X axis label (default "Value")
        ylabel (str, optional): Y axis label (default "Counts")
        value_range (tuple, optional): a tuple of two values to limit the range
            at X axis (default None)
        label1 (str, optional): legend label for the first value list (default
            "values 1")
        label2 (str, optional): legend label for the second value list (default
            "values 2")
        label3 (str, optional): legend label for the third value list (default
            "values 3")
        outfile (str, optional): if given (default None), the plot will be saved
            as a file under this path

    Returns:
        None
    """
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    if value_range is None:
        counts1, bin_edges1 = np.histogram(value_list1, bins=num_bins)
        counts2, bin_edges2 = np.histogram(value_list2, bins=num_bins)
        counts3, bin_edges3 = np.histogram(value_list3, bins=num_bins)
    elif isinstance(value_range, tuple) and len(value_range) == 2:
        counts1, bin_edges1 = np.histogram(value_list1, bins=num_bins,
                                           range=value_range)
        counts2, bin_edges2 = np.histogram(value_list2, bins=num_bins,
                                           range=value_range)
        counts3, bin_edges3 = np.histogram(value_list3, bins=num_bins,
                                           range=value_range)
    else:
        error_msg = "Range has to be a tuple of two numbers (min, max)."
        raise pexceptions.PySegInputError(expr='plot_hist', msg=error_msg)
    bincenters1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
    bincenters2 = 0.5 * (bin_edges2[1:] + bin_edges2[:-1])
    bincenters3 = 0.5 * (bin_edges3[1:] + bin_edges3[:-1])
    plt.plot(bincenters1, counts1, ls='-', marker='^', c="b", label=label1,
             linewidth=2)
    plt.plot(bincenters2, counts2, ls='--', marker='v', c="c", label=label2,
             linewidth=2)
    plt.plot(bincenters3, counts3, ls=':', marker='s', c="r", label=label3,
             linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    if outfile is None:
        plt.show()
    elif isinstance(outfile, str):
        fig.savefig(outfile)


def plot_plane_normal_errors(half_size, res=30, noise=10,
                             k=3, g_max=0, epsilon=0, eta=0, extra=0):
    """
    A method for plotting plane triangle normal errors as estimated by a
    modification of Normal Vector Voting (VV) algorithm vs. calculated by VTK.

    Args:
        half_size (int): half size of the plane (from center to an edge)
        res (int, optional): resolution (number of divisions) in X and Y
            axes (default 30)
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 10), the noise
            is added on triangle vertex coordinates in its normal direction
        k (int, optional): parameter of Normal Vector Voting algorithm
            determining the geodesic neighborhood radius:
            g_max = k * average weak triangle graph edge length (default 3)
        g_max (float, optional): geodesic neighborhood radius in length unit
            of the graph, here voxels; if positive (default 0) this g_max
            will be used and k will be ignored
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2) and "no preferred orientation" (class 3),
            default 0
        extra (int, optional); if != 0 (default 0), additional errors for k or
            g_max greater by extra than the given one will be plotted

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
    if g_max > 0:
        vv_normal_errors_file = (
            '{}.VV_g_max{}_epsilon{}_eta{}.normal_errors.txt'.format(
                base_filename, g_max, epsilon, eta))
        plot_base = ("plane_half_size{}.VV_g_max{}_epsilon{}_eta{}vsVTK"
                     ".normal_errors".format(half_size, g_max, epsilon, eta))
        label_vv = "VV ({})".format(g_max)
        if extra != 0:
            vv_normal_errors_file2 = (
                '{}.VV_g_max{}_epsilon{}_eta{}.normal_errors.txt'.format(
                    base_filename, g_max + extra, epsilon, eta))
            plot_base = ("plane_half_size{}.VV_g_max{}-{}_epsilon{}_eta{}vsVTK"
                         ".normal_errors".format(
                          half_size, g_max, g_max + extra, epsilon, eta))
            label_vv2 = "VV ({})".format(g_max + extra)
    elif k > 0:
        vv_normal_errors_file = (
            '{}.VV_k{}_epsilon{}_eta{}.normal_errors.txt'.format(
                base_filename, k, epsilon, eta))
        plot_base = ("plane_half_size{}.VV_k{}_epsilon{}_eta{}vsVTK"
                     ".normal_errors".format(half_size, k, epsilon, eta))
        label_vv = "VV ({})".format(k)
        if extra != 0:
            vv_normal_errors_file2 = (
                '{}.VV_k{}_epsilon{}_eta{}.normal_errors.txt'.format(
                    base_filename, k + extra, epsilon, eta))
            plot_base = ("plane_half_size{}.VV_k{}-{}_epsilon{}_eta{}vsVTK"
                         ".normal_errors".format(
                          half_size, k, k + extra, epsilon, eta))
            label_vv2 = "VV ({})".format(k + extra)
    else:
        error_msg = ("Either g_max or k must be positive (if both are "
                     "positive, the specified g_max will be used).")
        raise pexceptions.PySegInputError(
            expr='plot_plane_normal_errors', msg=error_msg)

    # Reading in the error values from files:
    if not os.path.exists(vtk_normal_errors_file):
        print ("File {} not found!".format(vtk_normal_errors_file))
        exit(0)
    if not os.path.exists(vv_normal_errors_file):
        print ("File {} not found!".format(vv_normal_errors_file))
        exit(0)
    vtk_normal_errors = io.read_values_from_file(vtk_normal_errors_file)
    vv_normal_errors = io.read_values_from_file(vv_normal_errors_file)
    if extra != 0:
        if not os.path.exists(vv_normal_errors_file2):
            print ("File {} not found!".format(vv_normal_errors_file2))
            exit(0)
        vv_normal_errors2 = io.read_values_from_file(vv_normal_errors_file2)

    # Plotting:
    plots_fold = '{}plots/'.format(fold)
    if not os.path.exists(plots_fold):
        os.makedirs(plots_fold)
    vv_vtk_normal_errors_plot = "{}{}.png".format(plots_fold, plot_base)
    if extra != 0:
        plot_triple_line_hist(
            vv_normal_errors, vv_normal_errors2, vtk_normal_errors, 10,
            "Comparison for Plane ({}% noise)".format(noise),
            xlabel="Normal Orientation Error (%)",
            ylabel="Number of Vertices", value_range=(0, 100),
            label1=label_vv, label2=label_vv2, label3="VTK",
            outfile=vv_vtk_normal_errors_plot)
    else:
        plot_double_line_hist(
            vv_normal_errors, vtk_normal_errors, 10,
            "Comparison for Plane ({}% noise)".format(noise),
            xlabel="Normal Orientation Error (%)",
            ylabel="Number of Vertices", value_range=(0, 100),
            label1=label_vv, label2="VTK",
            outfile=vv_vtk_normal_errors_plot)
    print ("The plot was saved as {}".format(vv_vtk_normal_errors_plot))


def plot_cylinder_T_2_errors(r, h, res=0, noise=0,
                             k=3, g_max=0, epsilon=0, eta=0):
    """
    A method for plotting cylinder minimal principal direction errors as
    estimated by a modification of Normal Vector Voting (VV) algorithm.

    Args:
        r (int): cylinder radius in voxels
        h (int): cylinder height in voxels
        res (int, optional): if > 0 determines how many stripes around both
            approximate circles (and then triangles) the cylinder has, the
            surface is generated directly using VTK; If 0 (default) first a
            cylinder mask is generated and then surface using gen_surface
            function
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 0), the noise
            is added on triangle vertex coordinates in its normal direction
        k (int, optional): parameter of Normal Vector Voting algorithm
            determining the geodesic neighborhood radius:
            g_max = k * average weak triangle graph edge length (default 3)
        g_max (float, optional): geodesic neighborhood radius in length unit
            of the graph, here voxels; if positive (default 0) this g_max
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
        fold = '{}synthetic_volumes/cylinder/noise{}/'.format(
            base_fold, noise)
    else:
        fold = '{}synthetic_surfaces/cylinder/res{}_noise{}/'.format(
            base_fold, res, noise)
    files_fold = '{}files4plotting/'.format(fold)
    base_filename = "{}cylinder_r{}_h{}".format(files_fold, r, h)
    if g_max > 0:
        vv_T_2_errors_file = (
            '{}.VV_g_max{}_epsilon{}_eta{}.T_2_errors.txt'.format(
                base_filename, g_max, epsilon, eta))
        label_vv = "VV ({})".format(g_max)
    elif k > 0:
        vv_T_2_errors_file = (
            '{}.VV_k{}_epsilon{}_eta{}.T_2_errors.txt'.format(
                base_filename, k, epsilon, eta))
        label_vv = "VV ({})".format(k)
    else:
        error_msg = ("Either g_max or k must be positive (if both are "
                     "positive, the specified g_max will be used).")
        raise pexceptions.PySegInputError(
            expr='plot_cylinder_T_2_errors', msg=error_msg)

    # Reading in the error values from files:
    if not os.path.exists(vv_T_2_errors_file):
        print ("File {} not found!".format(vv_T_2_errors_file))
        exit(0)
    vv_T_2_errors = io.read_values_from_file(vv_T_2_errors_file)

    # Plotting:
    plots_fold = '{}plots/'.format(fold)
    if not os.path.exists(plots_fold):
        os.makedirs(plots_fold)
    root, ext = os.path.splitext(vv_T_2_errors_file)
    head, tail = os.path.split(root)
    vv_T_2_errors_plot = "{}{}.png".format(plots_fold, tail)
    plot_line_hist(vv_T_2_errors, 10,
                   "Comparison for Cylinder ({}% noise)".format(noise),
                   xlabel="Minimal Principal Direction Error (%)",
                   ylabel="Number of Vertices", value_range=(0, 100),
                   label=label_vv,
                   outfile=vv_T_2_errors_plot)
    print ("The plot was saved as {}".format(vv_T_2_errors_plot))


def plot_sphere_curv_errors(radius, inverse=False, res=50, noise=10,
                            k=3, g_max=0, epsilon=0, eta=0, extra=0):
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
        noise (int, optional): determines variance of the Gaussian noise in
            percents of average triangle edge length (default 10), the noise
            is added on triangle vertex coordinates in its normal direction
        k (int, optional): parameter of Normal Vector Voting algorithm
            determining the geodesic neighborhood radius:
            g_max = k * average weak triangle graph edge length (default 3)
        g_max (float, optional): geodesic neighborhood radius in length unit
            of the graph, here voxels; if positive (default 0) this g_max
            will be used and k will be ignored
        epsilon (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2), default 0
        eta (int, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease
            junction" (class 2) and "no preferred orientation" (class 3, see
            Notes), default 0
        extra (int, optional); if != 0 (default 0), additional errors for k or
            g_max greater by extra than the given one will be plotted

    Returns:
        None
    """
    base_fold = '/fs/pool/pool-ruben/Maria/curvature/'
    if res == 0:
        fold = '{}synthetic_volumes/sphere/noise{}/'.format(base_fold, noise)
    else:
        fold = '{}synthetic_surfaces/sphere/res{}_noise{}/'.format(
            base_fold, res, noise)
    files_fold = '{}files4plotting/'.format(fold)
    if inverse:
        inverse_str = "inverse_"
    else:
        inverse_str = ""
    base_filename = "{}{}sphere_r{}".format(files_fold, inverse_str, radius)
    vtk_kappa_1_errors_file = ('{}.VTK.kappa_1_errors.txt'.format(
        base_filename))
    vtk_kappa_2_errors_file = ('{}.VTK.kappa_2_errors.txt'.format(
        base_filename))
    if g_max > 0:
        kappa_1_errors_file = (
            '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                base_filename, g_max, epsilon, eta))
        kappa_2_errors_file = (
            '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                base_filename, g_max, epsilon, eta))
        plot_base = "{}sphere_r{}.VV_g_max{}_epsilon{}_eta{}vsVTK".format(
            inverse_str, radius, g_max, epsilon, eta)
        label_vv = "VV ({})".format(g_max)
        if extra != 0:
            kappa_1_errors_file2 = (
                '{}.VV_g_max{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                    base_filename, g_max + extra, epsilon, eta))
            kappa_2_errors_file2 = (
                '{}.VV_g_max{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                    base_filename, g_max + extra, epsilon, eta))
            plot_base = ("{}sphere_r{}.VV_g_max{}-{}_epsilon{}_eta{}vsVTK"
                         .format(inverse_str, radius, g_max, g_max + extra,
                                 epsilon, eta))
            label_vv2 = "VV ({})".format(g_max + extra)
    elif k > 0:
        kappa_1_errors_file = (
            '{}.VV_k{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                base_filename, k, epsilon, eta))
        kappa_2_errors_file = (
            '{}.VV_k{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                base_filename, k, epsilon, eta))
        plot_base = "{}sphere_r{}.VV_k{}_epsilon{}_eta{}vsVTK".format(
            inverse_str, radius, k, epsilon, eta)
        label_vv = "VV ({})".format(k)
        if extra != 0:
            kappa_1_errors_file2 = (
                '{}.VV_k{}_epsilon{}_eta{}.kappa_1_errors.txt'.format(
                    base_filename, k + extra, epsilon, eta))
            kappa_2_errors_file2 = (
                '{}.VV_k{}_epsilon{}_eta{}.kappa_2_errors.txt'.format(
                    base_filename, k + extra, epsilon, eta))
            plot_base = ("{}sphere_r{}.VV_k{}-{}_epsilon{}_eta{}vsVTK"
                         .format(inverse_str, radius, k, k + extra,
                                 epsilon, eta))
            label_vv2 = "VV ({})".format(k + extra)
    else:
        error_msg = ("Either g_max or k must be positive (if both are "
                     "positive, the specified g_max will be used).")
        raise pexceptions.PySegInputError(
            expr='plot_sphere_curv_errors', msg=error_msg)

    # Reading in the values from files:
    if not os.path.exists(kappa_1_errors_file):
        print ("File {} not found!".format(kappa_1_errors_file))
        exit(0)
    if not os.path.exists(kappa_2_errors_file):
        print ("File {} not found!".format(kappa_2_errors_file))
        exit(0)
    if not os.path.exists(vtk_kappa_1_errors_file):
        print ("File {} not found!".format(vtk_kappa_1_errors_file))
        exit(0)
    if not os.path.exists(vtk_kappa_2_errors_file):
        print ("File {} not found!".format(vtk_kappa_2_errors_file))
        exit(0)
    kappa_1_errors = io.read_values_from_file(kappa_1_errors_file)
    kappa_2_errors = io.read_values_from_file(kappa_2_errors_file)
    vtk_kappa_1_errors = io.read_values_from_file(vtk_kappa_1_errors_file)
    vtk_kappa_2_errors = io.read_values_from_file(vtk_kappa_2_errors_file)
    if extra != 0:
        if not os.path.exists(kappa_1_errors_file2):
            print ("File {} not found!".format(kappa_1_errors_file2))
            exit(0)
        if not os.path.exists(kappa_2_errors_file2):
            print ("File {} not found!".format(kappa_2_errors_file2))
            exit(0)
        kappa_1_errors2 = io.read_values_from_file(kappa_1_errors_file2)
        kappa_2_errors2 = io.read_values_from_file(kappa_2_errors_file2)

    # Plotting:
    plots_fold = '{}plots/'.format(fold)
    if not os.path.exists(plots_fold):
        os.makedirs(plots_fold)
    kappa_1_errors_plot = "{}{}.kappa_1_errors.png".format(plots_fold,
                                                           plot_base)
    kappa_2_errors_plot = "{}{}.kappa_2_errors.png".format(plots_fold,
                                                           plot_base)
    if inverse is True:
        title = "Comparison for Inverse Sphere ({}% noise)".format(noise)
    else:
        title = "Comparison for Sphere ({}% noise)".format(noise)
    if extra != 0:
        plot_triple_line_hist(
            kappa_1_errors, kappa_1_errors2, vtk_kappa_1_errors, 10, title,
            xlabel="Maximal Principal Curvature Error (%)",
            ylabel="Number of Vertices", value_range=(0, 100),
            label1=label_vv, label2=label_vv2, label3="VTK",
            outfile=kappa_1_errors_plot)
        plot_triple_line_hist(
            kappa_2_errors, kappa_2_errors2, vtk_kappa_2_errors, 10, title,
            xlabel="Minimal Principal Curvature Error (%)",
            ylabel="Number of Vertices", value_range=(0, 100),
            label1=label_vv, label2=label_vv2, label3="VTK",
            outfile=kappa_2_errors_plot)
    else:
        plot_double_line_hist(
            kappa_1_errors, vtk_kappa_1_errors, 10, title,
            xlabel="Maximal Principal Curvature Error (%)",
            ylabel="Number of Vertices", value_range=(0, 100),
            label1=label_vv, label2="VTK",
            outfile=kappa_1_errors_plot)
        plot_double_line_hist(
            kappa_2_errors, vtk_kappa_2_errors, 10, title,
            xlabel="Minimal Principal Curvature Error (%)",
            ylabel="Number of Vertices", value_range=(0, 100),
            label1=label_vv, label2="VTK",
            outfile=kappa_2_errors_plot)
    print ("The plot was saved as {}".format(kappa_1_errors_plot))
    print ("The plot was saved as {}".format(kappa_2_errors_plot))


if __name__ == "__main__":
    # for n in [5, 10]:
    #     for k in [5, 3]:
    #         plot_plane_normal_errors(10, res=30, noise=n, k=k)
    # plot_plane_normal_errors(10, res=30, noise=10, k=5, extra=-2)

    # plot_cylinder_T_2_errors(20, 10, res=0, noise=0, k=3)

    # plot_sphere_curv_errors(10, inverse=False, res=30, noise=10, k=3)
    # plot_sphere_curv_errors(10, inverse=False, res=30, noise=10, k=5)
    # plot_sphere_curv_errors(10, inverse=False, res=30, noise=10, k=5, extra=-2)
    plot_sphere_curv_errors(10, inverse=True, res=30, noise=10, k=5, extra=-2)
