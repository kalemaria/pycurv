import vtk
import numpy as np
import time
from scipy import ndimage, optimize
from scipy.linalg import expm3
import math
from graph_tool import Graph, GraphView, incident_edges_op
from graph_tool.topology import (shortest_distance, label_largest_component,
                                 label_components)
import matplotlib.pyplot as plt

import graphs
import pexceptions
from pysurf_io import TypesConverter, save_vtp

"""
Set of functions and classes (abstract SurfaceGraph and derived TriangleGraph)
for representing a surface by a graph, cleaning the surface and triangle-wise
operations of the normal vector voting algorithm.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def add_curvature_to_vtk_surface(surface, curvature_type, invert=False):
    """
    Adds curvatures (Gaussian, mean, maximum or minimum) to each triangle vertex
    of a vtkPolyData surface calculated by VTK.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        curvature_type (str): type of curvature to add: 'Gaussian', 'Mean',
            'Maximum' or 'Minimum'
        invert (boolean, optional): if True (default False), VTK will calculate
            curvatures as for meshes with inward pointing normals (their
            convention is outwards pointing normals, opposite from ours)

    Returns:
        the vtkPolyData surface with '<type>_Curvature' property added to each
        triangle vertex
    """
    if isinstance(surface, vtk.vtkPolyData):
        curvature_filter = vtk.vtkCurvatures()
        curvature_filter.SetInputData(surface)
        if curvature_type == "Gaussian":
            curvature_filter.SetCurvatureTypeToGaussian()
        elif curvature_type == "Mean":
            curvature_filter.SetCurvatureTypeToMean()
        elif curvature_type == "Maximum":
            curvature_filter.SetCurvatureTypeToMaximum()
        elif curvature_type == "Minimum":
            curvature_filter.SetCurvatureTypeToMinimum()
        else:
            raise pexceptions.PySegInputError(
                expr='add_curvature_to_vtk_surface',
                msg=("One of the following strings required as the second "
                     "input: 'Gaussian', 'Mean', 'Maximum' or 'Minimum'."))
        if invert:
            curvature_filter.InvertMeanCurvatureOn()  # default Off
        curvature_filter.Update()
        surface_curvature = curvature_filter.GetOutput()
        return surface_curvature
    else:
        raise pexceptions.PySegInputError(
            expr='add_curvature_to_vtk_surface',
            msg="A vtkPolyData object required as the first input.")
    # How to get the curvatures later, e.g. for point with ID 0:
    # point_data = surface_curvature.GetPointData()
    # curvatures = point_data.GetArray(n)
    # where n = 2 for Gaussian, 3 for Mean, 4 for Maximum or Minimum
    # curvature_point0 = curvatures.GetTuple1(0)


def rescale_surface(surface, scale):
    """
    Rescales the given vtkPolyData surface with a given scaling factor in each
    of the three dimensions.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        scale (float): a scaling factor

    Returns:
        rescaled surface (vtk.vtkPolyData)
    """
    if isinstance(surface, vtk.vtkPolyData):
        transf = vtk.vtkTransform()
        transf.Scale(scale, scale, scale)
        tpd = vtk.vtkTransformPolyDataFilter()
        tpd.SetInputData(surface)
        tpd.SetTransform(transf)
        tpd.Update()
        scaled_surface = tpd.GetOutput()
        return scaled_surface
    else:
        raise pexceptions.PySegInputError(
            expr='rescale_surface',
            msg="A vtkPolyData object required as the first input.")


def signum(number):
    """
    Returns the signum of a number.

    Args:
        number: a number

    Returns:
        -1 if the number is negative, 1 if it is positive, 0 if it is 0
    """
    if number < 0:
        return -1
    elif number > 0:
        return 1
    else:
        return 0


def fit_curve(pos_x_2d, pos_y_2d):
    """
    Fits a parabolic curve to a set ot points in 2D.
    Args:
        pos_x_2d: x coordinates
        pos_y_2d: y coordinates

    Returns:
        optimal values of the parameter a of the canonical parabola (a * x * x)
        variance of the parameter estimate
    """
    # Initial guess of the parameter a is 0 (a straight line)
    x0 = np.array([0.0])
    try:
        popt, pcov = optimize.curve_fit(
            canonical_parabola, pos_x_2d, pos_y_2d, x0, sigma=None)
        a = popt[0]
        var_a = pcov[0][0]
        if var_a == float('Inf'):  # if fit is impossible (e.g. only one point)
            var_a = 1
        return a, var_a
    except RuntimeError as e:
        print "RuntimeError happened:"
        print(e)  # has to be: "Optimal parameters not found: gtol=0.000000 is
        # too small, func(x) is orthogonal to the columns of
        # the Jacobian to machine precision."
        return 0.0, -1.0  # in tests it looked like a perfect straight line


def canonical_parabola(x, a):
    """
    Generates the canonical parabola function.
    Args:
        x: x coordinate
        a: a parameter

    Returns:
        y = a * x * x
    """
    return a * x * x


def perpendicular_vector(iv, debug=False):
    """
    Finds a unit vector perpendicular to a given vector.
    Implementation of algorithm of Ahmed Fasih https://math.stackexchange.com/
    questions/133177/finding-a-unit-vector-perpendicular-to-another-vector

    Args:
        iv (numpy.ndarray): input 3D vector
        debug (boolean): if True (default False), an assertion is done to assure
            that the vectors are perpendicular

    Returns:
        3D vector perpendicular to the input vector (np.ndarray)
    """
    try:
        assert(isinstance(iv, np.ndarray) and iv.shape == (3,))
    except AssertionError:
        print("Requires a 1D numpy.ndarray of length 3 (3D vector)")
        return None
    if iv[0] == iv[1] == iv[2] == 0:
        print("Requires a non-zero 3D vector")
        return None
    ov = np.array([0.0, 0.0, 0.0])
    for m in range(3):
        if iv[m] != 0:
            break
    if m == 2:
        n = 0
    else:
        n = m + 1
    ov[n] = iv[m]
    ov[m] = -iv[n]
    if debug:
        try:
            assert np.dot(iv, ov) == 0
        except AssertionError:
            print("Failed to find a perpendicular vector to the given one")
            print("given vector: ({}, {}, {})".format(iv[0], iv[1], iv[2]))
            print("resulting vector: ({}, {}, {})".format(ov[0], ov[1], ov[2]))
            return None
    len_outv = math.sqrt(np.dot(ov, ov))
    if len_outv == 0:
        print("Resulting vector has length 0")
        print("given vector: ({}, {}, {})".format(iv[0], iv[1], iv[2]))
        print("resulting vector: ({}, {}, {})".format(ov[0], ov[1], ov[2]))
        return None
    return ov / len_outv  # unit length vector


def rotation_matrix(axis, theta):
    """
    Generates a rotation matrix for rotating a 3D vector around an axis by an
    angle. From B. M. https://stackoverflow.com/questions/6802577/
    python-rotation-of-3d-vector

    Args:
        axis (numpy.ndarray): rotational axis (3D vector)
        theta (float): rotational angle (radians)

    Returns:
        3 x 3 rotation matrix
    """
    a = axis / math.sqrt(np.dot(axis, axis))  # unit vector along axis
    A = np.cross(np.eye(3), a)  # skew-symmetric matrix associated to a
    return expm3(A * theta)


def rotate_vector(v, theta, axis=None, matrix=None, debug=False):
    """
    Rotates a 3D vector around an axis by an angle (wrapper function for
    rotation_matrix).

    Args:
        v (numpy.ndarray): input 3D vector
        theta (float): rotational angle (radians)
        axis (numpy.ndarray): rotational axis (3D vector)
        matrix (numpy.ndarray): 3 x 3 rotation matrix
        debug (boolean): if True (default False), an assertion is done to assure
            that the angle is correct

    Returns:
        rotated 3D vector (numpy.ndarray)
    """
    sqrt = math.sqrt
    dot = np.dot
    acos = math.acos
    pi = math.pi

    if matrix is None and axis is not None:
        R = rotation_matrix(axis, theta)
    elif matrix is not None and axis is None:
        R = matrix
    else:
        print("Either the rotation axis or rotation matrix must be given")
        return None

    u = dot(R, v)
    if debug:
        cos_theta = dot(v, u) / sqrt(dot(v, v)) / sqrt(dot(u, u))
        try:
            theta2 = acos(cos_theta)
        except ValueError:
            if cos_theta > 1:
                cos_theta = 1.0
            elif cos_theta < 0:
                cos_theta = 0.0
            theta2 = acos(cos_theta)
        try:
            assert theta - (0.05 * pi) <= theta2 <= theta + (0.05 * pi)
        except AssertionError:
            print("Angle between the input vector and the rotated one is not "
                  "{}, but {}".format(theta, theta2))
            return None
    return u


def collecting_normal_votes_not_oo(vertex_v, tg, g_max, A_max, sigma,
                                   full_dist_map=None, verbose=False):
    """
    For a vertex v, collects the normal votes of all triangles within its
    geodesic neighborhood and calculates the weighted covariance matrix sum
    V_v.

    Implements equations (6), illustrated in figure 6(b), (7) and (8) from
    the paper of Page et al., 2002.

    More precisely, a normal vote N_i of each triangle i (whose centroid c_i
    is lying within the geodesic neighborhood of vertex v) is calculated
    using the normal N assigned to the triangle i. Then, each vote is
    represented by a covariance matrix V_i and votes are collected as a
    weighted matrix sum V_v, where each vote is weighted depending on the
    area of triangle i and the geodesic distance of its centroid c_i from v.

    Here, c_i and v are both centroids of triangles (v is a triangle vertex
    in Page's approach), which are vertices of TriangleGraph generated from
    the triangle surface.

    Args:
        vertex_v (graph_tool.Vertex): the vertex v in the surface
            triangle-graph for which the votes are collected
        tg (TriangleGraph): TriangleGraph object to apply the function on
        g_max (float): the maximal geodesic distance in nanometers
        A_max (float): the area of the largest triangle in the surface
            triangle-graph
        sigma (float): sigma, defined as 3*sigma = g_max, so that votes
            beyond the neighborhood can be ignored
        full_dist_map (graph_tool.PropertyMap, optional): the full distance
            map for the whole graph; if None, a local distance map is
            calculated for this vertex (default)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:mou
        - a dictionary of neighbors of vertex v, mapping index of each
          vertex c_i to its geodesic distance from the vertex v
        - the 3x3 symmetric matrix V_v (numpy.ndarray)
    """
    # To spare function referencing every time in the following for loop:
    vertex = tg.graph.vertex
    normal = tg.graph.vp.normal
    array = np.array
    xyz = tg.graph.vp.xyz
    sqrt = math.sqrt
    dot = np.dot
    outer = np.multiply.outer
    area = tg.graph.vp.area
    exp = math.exp

    # Get the coordinates of vertex v (as numpy array):
    v = xyz[vertex_v]
    v = array(v)

    # Find the neighboring vertices of vertex v to be returned:
    neighbor_idx_to_dist = tg.find_geodesic_neighbors(
        vertex_v, g_max, full_dist_map=full_dist_map)
    try:
        assert len(neighbor_idx_to_dist) > 0
    except AssertionError:
        print ("\nWarning: the vertex v = %s has 0 neighbors. "
               "It will be ignored later." % v)
        # return a placeholder instead of V_v
        return neighbor_idx_to_dist, np.zeros(shape=(3, 3))

    if verbose:
        print "\nv = %s" % v
        print "%s neighbors" % len(neighbor_idx_to_dist)

    # Initialize the weighted matrix sum of all votes for vertex v to be
    # calculated and returned:
    V_v = np.zeros(shape=(3, 3))

    # Let each of the neighboring vertices to cast a vote on vertex v:
    for idx_c_i in neighbor_idx_to_dist.keys():
        # Get neighboring vertex c_i and its coordinates (as numpy array):
        vertex_c_i = vertex(idx_c_i)
        c_i = xyz[vertex_c_i]
        c_i = array(c_i)

        # Calculate the normal vote N_i of c_i on v:
        N = normal[vertex_c_i]
        N = array(N)

        vc_i = c_i - v
        vc_i_len = sqrt(dot(vc_i, vc_i))
        vc_i_norm = vc_i / vc_i_len

        # theta_i is the angle between the vectors N and vc_i
        cos_theta_i = - (dot(N, vc_i)) / vc_i_len

        N_i = N + 2 * cos_theta_i * vc_i_norm

        # Covariance matrix containing one vote of c_i on v:
        V_i = outer(N_i, N_i)

        # Calculate the weight depending on the area of the neighboring
        # triangle i, A_i, and the geodesic distance to the neighboring
        # vertex c_i from vertex v, g_i:
        A_i = area[vertex_c_i]
        g_i = neighbor_idx_to_dist[idx_c_i]
        w_i = A_i / A_max * exp(- g_i / sigma)

        if verbose:
            print "\nc_i = %s" % c_i
            print "N = %s" % N
            print "vc_i = %s" % vc_i
            print "||vc_i|| = %s" % vc_i_len
            print "cos(theta_i) = %s" % cos_theta_i
            print "N_i = %s" % N_i
            print "V_i = %s" % V_i
            print "A_i = %s" % A_i
            print "g_i = %s" % g_i
            print "w_i = %s" % w_i

        # Weigh V_i and add it to the weighted matrix sum:
        V_v += w_i * V_i

    if verbose:
        print "\nV_v: %s" % V_v
    return neighbor_idx_to_dist, V_v
    # return vertex_v, neighbor_idx_to_dist, V_v


class SurfaceGraph(graphs.SegmentationGraph):
    """Class defining the abstract SurfaceGraph object."""

    def build_graph_from_vtk_surface(self, verbose=False):
        """
        Base method for building a graph from a vtkPolyData surface, to be
        implemented by SurfaceGraph subclasses.

        Args:
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        pass


class PointGraph(SurfaceGraph):
    """
    Class defining the PointGraph object, its attributes and methods.

    The constructor requires the following parameters of the underlying
    segmentation that will be used to build the graph.

    Args:
        surface (vtk.vtkPolyData): a signed surface (mesh of triangles)
            generated from the segmentation in voxels
        scale_factor_to_nm (float): pixel size in nanometers for scaling the
            surface and the graph
    """

    def __init__(self, surface, scale_factor_to_nm):
        """
        Constructor.

        Args:
            surface (vtk.vtkPolyData): a signed surface (mesh of triangles)
                generated from the segmentation in voxels
            scale_factor_to_nm (float): pixel size in nanometers for scaling the
                surface and the graph

        Returns:
            None
        """
        graphs.SegmentationGraph.__init__(self, scale_factor_to_nm)

        if isinstance(surface, vtk.vtkPolyData):
            self.surface = surface
            """vtk.vtkPolyData: a signed surface (mesh of triangles) generated
            from the segmentation (in voxels)"""
        else:
            raise pexceptions.PySegInputError(
                expr='SegmentationGraph constructor',
                msg="A vtkPolyData object required as the first input.")

    def build_graph_from_vtk_surface(self, verbose=False):
        """
        Builds the graph from the vtkPolyData surface, which is rescaled to
        nanometers according to the scale factor also specified when creating
        the PointGraph object.

        Every vertex of the graph represents a surface triangle vertex,
        and every edge of the graph connects two adjacent vertices, just like a
        triangle edge.

        Args:
            verbose(boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        # rescale the surface to nm and update the attribute
        surface = rescale_surface(self.surface, self.scale_factor_to_nm)
        self.surface = surface

        if verbose:
            # 0. Check numbers of cells and all points.
            print '%s cells' % surface.GetNumberOfCells()
            print '%s points' % surface.GetNumberOfPoints()

        # 1. Iterate over all cells, adding their points as vertices to the
        # graph and connecting them by edges.
        for i in xrange(surface.GetNumberOfCells()):
            if verbose:
                print 'Cell number %s:' % i

            # Get all points which made up the cell & check that they are 3.
            points_cell = surface.GetCell(i).GetPoints()
            if points_cell.GetNumberOfPoints() == 3:
                # 1a) Add each of the 3 points as vertex to the graph, if
                # it has not been added yet.
                for j in range(0, 3):
                    x, y, z = points_cell.GetPoint(j)
                    p = (x, y, z)
                    if p not in self.coordinates_to_vertex_index:
                        vd = self.graph.add_vertex()  # vertex descriptor
                        self.graph.vp.xyz[vd] = [x, y, z]
                        self.coordinates_to_vertex_index[p] = \
                            self.graph.vertex_index[vd]
                        if verbose:
                            print ('\tThe point (%s, %s, %s) has been added'
                                   ' to the graph as a vertex.' % (x, y, z))

                # 1b) Add an edge with a distance between all 3 pairs of
                # vertices, if it has not been added yet.
                for k in range(0, 2):
                    x1, y1, z1 = points_cell.GetPoint(k)
                    p1 = (x1, y1, z1)
                    # vertex descriptor of the 1st point
                    vd1 = self.graph.vertex(
                        self.coordinates_to_vertex_index[p1])
                    for l in range(k + 1, 3):
                        x2, y2, z2 = points_cell.GetPoint(l)
                        p2 = (x2, y2, z2)
                        # vertex descriptor of the 2nd point
                        vd2 = self.graph.vertex(
                            self.coordinates_to_vertex_index[p2])
                        if not (
                              (p1, p2) in self.coordinates_pair_connected or
                              (p2, p1) in self.coordinates_pair_connected):
                            # edge descriptor
                            ed = self.graph.add_edge(vd1, vd2)
                            self.graph.ep.distance[ed] = \
                                self.distance_between_voxels(p1, p2)
                            self.coordinates_pair_connected[(p1, p2)] = True
                            if verbose:
                                print ('\tThe neighbor points (%s, %s, %s) '
                                       'and (%s, %s, %s) have been '
                                       'connected by an edge with a '
                                       'distance of %s pixels.'
                                       % (x1, y1, z1, x2, y2, z2,
                                          self.graph.ep.distance[ed]))
            else:
                print ('Oops, there are %s points in cell number %s'
                       % (points_cell.GetNumberOfPoints(), i))

        # 2. Check if the numbers of vertices and edges are as they should
        # be:
        assert self.graph.num_vertices() == len(
            self.coordinates_to_vertex_index)
        assert self.graph.num_edges() == len(
            self.coordinates_pair_connected)
        print '%s triangle vertices' % self.graph.num_vertices()


class TriangleGraph(SurfaceGraph):
    """
    Class defining the TriangleGraph object, its attributes and methods.

    The constructor requires the following parameters of the underlying
    segmentation that will be used to build the graph.

    Args:
        surface (vtk.vtkPolyData): a signed surface (mesh of triangles)
            generated from the segmentation in voxels
        scale_factor_to_nm (float, optional): pixel size in nanometers for
            scaling the surface and the graph (default 1)
    """

    def __init__(self, surface, scale_factor_to_nm=1):
        """
        Constructor.

        Args:
            surface (vtk.vtkPolyData): a signed surface (mesh of triangles)
                generated from the segmentation in voxels
            scale_factor_to_nm (float, optional): pixel size in nanometers for
                scaling the surface and the graph (default 1)

        Returns:
            None
        """
        graphs.SegmentationGraph.__init__(self, scale_factor_to_nm)

        if isinstance(surface, vtk.vtkPolyData):
            self.surface = surface
            """vtk.vtkPolyData: a signed surface (mesh of triangles) generated
            from the segmentation (in voxels)"""
        else:
            raise pexceptions.PySegInputError(
                expr='SegmentationGraph constructor',
                msg="A vtkPolyData object required as the first input.")

        # Add more "internal property maps" to the graph.
        # vertex property for storing the area in nanometers squared of the
        # corresponding triangle:
        self.graph.vp.area = self.graph.new_vertex_property("float")
        # vertex property for storing the normal in nanometers of the
        # corresponding triangle:
        self.graph.vp.normal = self.graph.new_vertex_property("vector<float>")
        # vertex property for storing the VTK minimal curvature at the
        # corresponding triangle:
        self.graph.vp.min_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK maximal curvature at the
        # corresponding triangle:
        self.graph.vp.max_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK Gaussian curvature at the
        # corresponding triangle:
        self.graph.vp.gauss_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK mean curvature at the
        # corresponding triangle:
        self.graph.vp.mean_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the coordinates in nanometers of the 3
        # points of the corresponding triangle:
        self.graph.vp.points = self.graph.new_vertex_property("object")
        # edge property storing the "strength" property of the edge: 1 for a
        # "strong" or 0 for a "weak" one:
        self.graph.ep.is_strong = self.graph.new_edge_property("int")

        self.point_in_cells = {}
        """dict: a dictionary mapping a point coordinates (x, y, z) in
        nanometers to a list of triangle-cell indices sharing this point.
        """

    def build_graph_from_vtk_surface(
            self, verbose=False, reverse_normals=False):
        """
        Builds the graph from the vtkPolyData surface, which is rescaled to
        nanometers according to the scale factor also specified when creating
        the TriangleGraph object.

        Every vertex of the graph represents the center of a surface triangle,
        and every edge of the graph connects two adjacent triangles. There are
        two types of edges: a "strong" edge if the adjacent triangles share two
        triangle edges and a "weak" edge if they share only one edge.

        Args:
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out
            reverse_normals (boolean, optional): if True (default False), the
                triangle normals are reversed during graph generation

        Returns:
            rescaled surface to nanometers with VTK curvatures (vtk.vtkPolyData)
        """

        t_begin = time.time()

        # 1. Preparation
        if self.scale_factor_to_nm != 1:
            # rescale the surface to nm and update the attribute
            self.surface = rescale_surface(
                self.surface, self.scale_factor_to_nm)

        print 'Adding curvatures to the vtkPolyData surface...'
        # because VTK and we (gen_surface) have the opposite normal
        # convention: VTK outwards pointing normals, we: inwards pointing
        if reverse_normals:
            invert = False
        else:
            invert = True
        self.surface = add_curvature_to_vtk_surface(
            self.surface, "Minimum", invert)
        self.surface = add_curvature_to_vtk_surface(
            self.surface, "Maximum", invert)

        if verbose:
            # Check numbers of cells and all points.
            print '%s cells' % self.surface.GetNumberOfCells()
            print '%s points' % self.surface.GetNumberOfPoints()

        point_data = self.surface.GetPointData()
        n = point_data.GetNumberOfArrays()
        min_curvatures = None
        max_curvatures = None
        gauss_curvatures = None
        mean_curvatures = None
        for i in range(n):
            if point_data.GetArrayName(i) == "Minimum_Curvature":
                min_curvatures = point_data.GetArray(i)
            if point_data.GetArrayName(i) == "Maximum_Curvature":
                max_curvatures = point_data.GetArray(i)
            if point_data.GetArrayName(i) == "Gauss_Curvature":
                gauss_curvatures = point_data.GetArray(i)
            if point_data.GetArrayName(i) == "Mean_Curvature":
                mean_curvatures = point_data.GetArray(i)

        # 2. Add each triangle cell as a vertex to the graph. Ignore the
        # non-triangle cells and cell with area equal to zero.
        # Make a list of all added triangle cell indices:
        triangle_cell_ids = []
        for cell_id in xrange(self.surface.GetNumberOfCells()):
            # Get the cell i and check if it's a triangle:
            cell = self.surface.GetCell(cell_id)
            if not isinstance(cell, vtk.vtkTriangle):
                print ('Oops, the cell number %s is not a vtkTriangle but '
                       'a %s! It will be ignored.'
                       % (cell_id, cell.__class__.__name__))
                continue
            if verbose:
                print 'Triangle cell number %s' % cell_id

            # Initialize a list for storing the points coordinates making
            # out the cell
            points_xyz = []

            # Get the 3 points which made up the triangular cell:
            points_cell = cell.GetPoints()

            # Calculate the area of the triangle i;
            area = cell.TriangleArea(points_cell.GetPoint(0),
                                     points_cell.GetPoint(1),
                                     points_cell.GetPoint(2))
            try:
                assert(area > 0)
            except AssertionError:
                print ('\tThe cell %s cannot be added to the graph as a vertex,'
                       ' because the triangle area is not positive, but is %s. '
                       % (cell_id, area))
                continue

            # Calculate the centroid of the triangle:
            x_center = 0
            y_center = 0
            z_center = 0
            for j in range(0, 3):
                x, y, z = points_cell.GetPoint(j)
                x_center += x
                y_center += y
                z_center += z

                # Add each point j as a key in point_in_cells and cell index
                # to the value list:
                point_j = (x, y, z)
                if point_j in self.point_in_cells:
                    self.point_in_cells[point_j].append(cell_id)
                else:
                    self.point_in_cells[point_j] = [cell_id]

                # Add each point j into the points coordinates list
                points_xyz.append([x, y, z])
            x_center /= 3
            y_center /= 3
            z_center /= 3

            # Calculate the normal of the triangle i;
            normal = np.zeros(shape=3)
            cell.ComputeNormal(points_cell.GetPoint(0),
                               points_cell.GetPoint(1),
                               points_cell.GetPoint(2), normal)
            if reverse_normals:
                normal *= -1

            # Get the min, max, Gaussian and mean curvatures (calculated by
            # VTK) for each of 3 points of the triangle i and calculate the
            # average curvatures:
            avg_min_curvature = 0
            avg_max_curvature = 0
            avg_gauss_curvature = 0
            avg_mean_curvature = 0
            for j in range(0, 3):
                point_j_id = cell.GetPointId(j)

                min_curvature_point_j = min_curvatures.GetTuple1(point_j_id)
                avg_min_curvature += min_curvature_point_j

                max_curvature_point_j = max_curvatures.GetTuple1(point_j_id)
                avg_max_curvature += max_curvature_point_j

                gauss_curvature_point_j = gauss_curvatures.GetTuple1(
                    point_j_id)
                avg_gauss_curvature += gauss_curvature_point_j

                mean_curvature_point_j = mean_curvatures.GetTuple1(
                    point_j_id)
                avg_mean_curvature += mean_curvature_point_j

            avg_min_curvature /= 3
            avg_max_curvature /= 3
            avg_gauss_curvature /= 3
            avg_mean_curvature /= 3

            # Add the centroid as vertex to the graph, setting its
            # properties:
            vd = self.graph.add_vertex()  # vertex descriptor
            self.graph.vp.xyz[vd] = [x_center, y_center, z_center]
            self.coordinates_to_vertex_index[(
                x_center, y_center, z_center)] = self.graph.vertex_index[vd]
            self.graph.vp.area[vd] = area
            self.graph.vp.normal[vd] = normal
            self.graph.vp.min_curvature[vd] = avg_min_curvature
            self.graph.vp.max_curvature[vd] = avg_max_curvature
            self.graph.vp.gauss_curvature[vd] = avg_gauss_curvature
            self.graph.vp.mean_curvature[vd] = avg_mean_curvature
            self.graph.vp.points[vd] = points_xyz

            if verbose:
                print ('\tThe triangle centroid %s has been added to the '
                       'graph as a vertex. Triangle area = %s, normal = %s,'
                       '\naverage minimal curvature = %s,'
                       'average maximal curvature = %s, points = %s.'
                       % (self.graph.vp.xyz[vd],
                          self.graph.vp.area[vd], self.graph.vp.normal[vd],
                          self.graph.vp.min_curvature[vd],
                          self.graph.vp.max_curvature[vd],
                          self.graph.vp.points[vd]))

            triangle_cell_ids.append(cell_id)

        # 3. Add edges for each cell / vertex.
        for i, cell_id in enumerate(triangle_cell_ids):
            # Note: i corresponds to the vertex number of each cell, because
            # they were added in this order
            cell = self.surface.GetCell(cell_id)
            if verbose:
                print '(Triangle) cell number %s:' % cell_id

            # Find the "neighbor cells" and how many points they share with
            # cell i (1 or 2) as follows.
            # For each point j of cell i, iterate over the neighbor cells
            # sharing that point (excluding the cell i).
            # Add each neighbor cell to the neighbor_cells list if it is not
            # there yet and add 1 into the shared_points list.
            # Otherwise, find the index of the cell in neighbor_cells and
            # increase the counter of shared_points at the same index.
            points_cell = cell.GetPoints()
            neighbor_cells = []
            shared_points = []
            for j in range(points_cell.GetNumberOfPoints()):
                point_j = points_cell.GetPoint(j)
                for neighbor_cell_id in self.point_in_cells[point_j]:
                    if neighbor_cell_id != cell_id:
                        if neighbor_cell_id not in neighbor_cells:
                            neighbor_cells.append(neighbor_cell_id)
                            shared_points.append(1)
                        else:
                            idx = neighbor_cells.index(neighbor_cell_id)
                            shared_points[idx] += 1
            if verbose:
                print "has %s neighbor cells" % len(neighbor_cells)

            # Get the vertex descriptor representing the cell i (vertex i):
            vd_i = self.graph.vertex(i)

            # Get the coordinates of the vertex i:
            p_i = self.graph.vp.xyz[vd_i]  # a list
            p_i = (p_i[0], p_i[1], p_i[2])  # a tuple

            # Iterate over the ready neighbor_cells and shared_points lists,
            # connecting cell i with a neighbor cell x with a "strong" edge
            # if they share 2 edges and with a "weak" edge otherwise (if
            # they share only 1 edge).
            for idx, neighbor_cell_id in enumerate(neighbor_cells):
                # Get the vertex descriptor representing the cell x:
                # vertex index of the current neighbor cell
                x = triangle_cell_ids.index(neighbor_cell_id)
                # vertex descriptor of the current neighbor cell, vertex x
                vd_x = self.graph.vertex(x)

                # Get the coordinates of the vertex x:
                p_x = self.graph.vp.xyz[vd_x]
                p_x = (p_x[0], p_x[1], p_x[2])

                # Add an edge between the vertices i and x, if it has not
                # been added yet:
                if not (((p_i, p_x) in self.coordinates_pair_connected) or
                        ((p_x, p_i) in self.coordinates_pair_connected)):
                    ed = self.graph.add_edge(vd_i, vd_x)  # edge descriptor
                    self.coordinates_pair_connected[(p_i, p_x)] = True

                    # Add the distance of the edge
                    self.graph.ep.distance[ed] = \
                        self.distance_between_voxels(p_i, p_x)

                    # Assign the "strength" property to the edge as
                    # explained above:
                    if shared_points[idx] == 2:
                        is_strong = 1
                        strength = 'strong'
                    else:
                        is_strong = 0
                        strength = 'weak'
                    self.graph.ep.is_strong[ed] = is_strong
                    if verbose:
                        print ('\tThe neighbor vertices (%s, %s, %s) and '
                               '(%s, %s, %s) have been connected by a %s '
                               'edge with a distance of %s pixels.'
                               % (p_i[0], p_i[1], p_i[2],
                                  p_x[0], p_x[1], p_x[2], strength,
                                  self.graph.ep.distance[ed]))

        # 4. Check if the numbers of vertices and edges are as they should be:
        assert self.graph.num_vertices() == len(triangle_cell_ids)
        assert self.graph.num_edges() == len(self.coordinates_pair_connected)
        if verbose:
            print ('Real number of unique points: %s'
                   % len(self.point_in_cells))

        t_end = time.time()
        duration = t_end - t_begin
        print ('Surface graph generation took: %s min %s s'
               % divmod(duration, 60))

    def graph_to_triangle_poly(self, verbose=False):
        """
        Generates a VTK PolyData object from the TriangleGraph object with
        triangle-cells representing the surface triangles.

        Args:
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            vtk.vtkPolyData with triangle-cells
        """
        if self.graph.num_vertices() > 0:
            # Initialization
            poly_triangles = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            vertex_arrays = list()
            # Vertex property arrays
            for prop_key in self.graph.vp.keys():
                data_type = self.graph.vp[prop_key].value_type()
                if (data_type != 'string' and data_type != 'python::object' and
                        prop_key != 'points' and prop_key != 'xyz'):
                    if verbose:
                        print '\nvertex property key: %s' % prop_key
                        print 'value type: %s' % data_type
                    if data_type[0:6] != 'vector':  # scalar
                        num_components = 1
                    else:  # vector
                        num_components = len(
                            self.graph.vp[prop_key][self.graph.vertex(0)])
                    array = TypesConverter().gt_to_vtk(data_type)
                    array.SetName(prop_key)
                    if verbose:
                        print 'number of components: %s' % num_components
                    array.SetNumberOfComponents(num_components)
                    vertex_arrays.append(array)
            if verbose:
                print '\nvertex arrays length: %s' % len(vertex_arrays)

            # Geometry
            # lut[vertex_index, triangle_point_index*] = point_array_index**
            # *ALWAYS 0-2, **0-(NumPoints-1)
            lut = np.zeros(shape=(self.graph.num_vertices(), 3), dtype=np.int)
            i = 0  # next new point index
            # dictionary of points with a key (x, y, z) and the index in VTK
            # points list as a value
            points_dict = {}
            for vd in self.graph.vertices():
                # enumerate over the 3 points of the triangle (vertex)
                for j, [x, y, z] in enumerate(self.graph.vp.points[vd]):
                    # add the new point everywhere & update the index
                    if (x, y, z) not in points_dict:
                        points.InsertPoint(i, x, y, z)
                        lut[self.graph.vertex_index[vd], j] = i
                        points_dict[(x, y, z)] = i
                        i += 1
                    else:  # reference the old point index only in the lut
                        lut[self.graph.vertex_index[vd], j] = points_dict[
                            (x, y, z)]
            if verbose:
                print 'number of points: %s' % points.GetNumberOfPoints()

            # Topology
            # Triangles
            triangles = vtk.vtkCellArray()
            for vd in self.graph.vertices():  # vd = vertex descriptor
                # storing triangles of type Triangle:
                triangle = vtk.vtkTriangle()
                # The first parameter is the index of the triangle vertex which
                # is ALWAYS 0-2.
                # The second parameter is the index into the point (geometry)
                # array, so this can range from 0-(NumPoints-1)
                triangle.GetPointIds().SetId(
                    0, lut[self.graph.vertex_index[vd], 0])
                triangle.GetPointIds().SetId(
                    1, lut[self.graph.vertex_index[vd], 1])
                triangle.GetPointIds().SetId(
                    2, lut[self.graph.vertex_index[vd], 2])
                triangles.InsertNextCell(triangle)
                for array in vertex_arrays:
                    prop_key = array.GetName()
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.graph.vp[prop_key].value_type()
                    data_type = TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.get_vertex_prop_entry(
                        prop_key, vd, n_comp, data_type))
            if verbose:
                print ('number of triangle cells: %s'
                       % triangles.GetNumberOfCells())

            # vtkPolyData construction
            poly_triangles.SetPoints(points)
            poly_triangles.SetPolys(triangles)
            for array in vertex_arrays:
                poly_triangles.GetCellData().AddArray(array)

            return poly_triangles

        else:
            print "The graph is empty!"
            return None

    def find_graph_border(self, purge=False):
        """
        Finds vertices at the graph border, defined as such having less than 3
        strong edges.

        Args:
            purge (boolean, optional): if True, those vertices and their edges
                will be filtered out permanently, if False (default) no
                filtering will be done

        Returns:
            list of indices of vertices at the graph border
        """
        if "is_on_border" not in self.graph.vertex_properties:
            print 'Finding vertices at the graph border...'
            # Add a vertex property for storing the number of strong edges:
            self.graph.vp.num_strong_edges = self.graph.new_vertex_property(
                "int")
            # Sum up the "strong" edges coming out of each vertex and add them
            # to the new property:
            incident_edges_op(self.graph, "out", "sum", self.graph.ep.is_strong,
                              self.graph.vp.num_strong_edges)
            # print ('number of strong edges: min = %s, max = %s'
            #        % (min(num_strong_edges.a), max(num_strong_edges.a)))

            # Add a boolean vertex property telling whether a vertex is on
            # border:
            self.graph.vp.is_on_border = self.graph.new_vertex_property(
                "boolean")
            # indices of vertices with less than 3 strong edges (= vertices on
            # border)
            border_vertices_indices = np.where(
                self.graph.vp.num_strong_edges.a < 3)[0]
            self.graph.vp.is_on_border.a = np.zeros(
                shape=self.graph.num_vertices())
            self.graph.vp.is_on_border.a[border_vertices_indices] = 1
        else:
            border_vertices_indices = np.where(
                self.graph.vp.is_on_border.a == 1)[0]
        print ('%s vertices are at the graph border.'
               % len(border_vertices_indices))

        if purge is True:
            print ('Filtering out the vertices at the graph borders and their '
                   'edges...')
            # Set the filter to get only vertices NOT on border.
            self.graph.set_vertex_filter(self.graph.vp.is_on_border,
                                         inverted=True)
            # Purge the filtered out vertices and edges permanently from the
            # graph:
            self.graph.purge_vertices()
            # Remove the properties used for the filtering that are no longer
            # true:
            del self.graph.vertex_properties["num_strong_edges"]
            del self.graph.vertex_properties["is_on_border"]
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

        return border_vertices_indices

    def find_vertices_near_border(self, b, purge=False):
        """
        Finds vertices that are within a given distance in nanometers to the
        graph border.

        Args:
            b (float): distance from border in nanometers
            purge (boolean, optional): if True, those vertices and their edges
                will be filtered out permanently; if False (default), no
                filtering will be done

        Returns:
            None
        """
        if "is_near_border" not in self.graph.vertex_properties:
            border_vertices_indices = self.find_graph_border()

            print ('For each graph border vertex, finding vertices within '
                   'geodesic distance %s to it...' % b)
            vertex_id_within_b_to_border = dict()
            for border_v_i in border_vertices_indices:
                border_v = self.graph.vertex(border_v_i)
                dist_border_v = shortest_distance(
                    self.graph, source=border_v, target=None,
                    weights=self.graph.ep.distance, max_dist=b)
                dist_border_v = dist_border_v.get_array()

                idxs = np.where(dist_border_v <= b)[0]
                for idx in idxs:
                    dist = dist_border_v[idx]
                    try:
                        vertex_id_within_b_to_border[idx] = min(
                            dist, vertex_id_within_b_to_border[idx])
                    except KeyError:
                        vertex_id_within_b_to_border[idx] = dist
            print ('%s vertices are within distance %s nm to the graph border.'
                   % (len(vertex_id_within_b_to_border), b))

            # Add a boolean vertex property telling whether a vertex is within
            # distance b to border:
            self.graph.vp.is_near_border = self.graph.new_vertex_property(
                "boolean")
            for v in self.graph.vertices():
                v_i = self.graph.vertex_index[v]
                if v_i in vertex_id_within_b_to_border:
                    self.graph.vp.is_near_border[v] = 1
                else:
                    self.graph.vp.is_near_border[v] = 0

        if purge is True:
            print 'Filtering out those vertices and their edges...'
            # Set the filter to get only vertices NOT within distance b to
            # border.
            self.graph.set_vertex_filter(self.graph.vp.is_near_border,
                                         inverted=True)
            # Purge filtered out vertices and edges permanently from the graph:
            self.graph.purge_vertices()
            # Remove the properties used for filtering that are no longer true:
            del self.graph.vertex_properties["num_strong_edges"]
            del self.graph.vertex_properties["is_on_border"]
            del self.graph.vertex_properties["is_near_border"]
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

    def find_vertices_outside_mask(self, mask, label=1, allowed_dist=0):
        """
        Finds vertices that are outside a mask.

        This means that their scaled back to pixels coordinates are further away
        than an allowed distance in pixels to a mask voxel with the given label.

        Args:
            mask (numpy.ndarray): 3D mask of the segmentation from which the
                underlying surface was created
            label (int, optional): the label in the mask to be considered
                (default 1)
            allowed_dist (int, optional): allowed distance in pixels between a
                voxel coordinate and a mask voxel (default 0)

        Returns:
            None
        """
        if isinstance(mask, np.ndarray):
            print '\nFinding vertices outside the membrane mask...'
            # Add a boolean vertex property telling whether a vertex is outside
            # the mask:
            self.graph.vp.is_outside_mask = \
                self.graph.new_vertex_property("boolean")
            # Invert the boolean matrix, because distance_transform_edt
            # calculates distances from '0's, not from '1's!
            maskd = ndimage.morphology.distance_transform_edt(
                np.invert(mask == label)
            )

            num_vertices_outside_mask = 0
            for v in self.graph.vertices():
                v_xyz = self.graph.vp.xyz[v]
                v_pixel_x = int(round(v_xyz[0] / self.scale_factor_to_nm))
                v_pixel_y = int(round(v_xyz[1] / self.scale_factor_to_nm))
                v_pixel_z = int(round(v_xyz[2] / self.scale_factor_to_nm))
                try:
                    if maskd[v_pixel_x, v_pixel_y, v_pixel_z] > allowed_dist:
                        self.graph.vp.is_outside_mask[v] = 1
                        num_vertices_outside_mask += 1
                    else:
                        self.graph.vp.is_outside_mask[v] = 0
                except IndexError:
                    print ("IndexError happened. Vertex with coordinates in nm "
                           "(%s, %s, %s)" % (v_xyz[0], v_xyz[1], v_xyz[2]))
                    print ("was transformed to pixel (%s, %s, %s),"
                           % (v_pixel_x, v_pixel_y, v_pixel_z))
                    print ("which is not inside the mask with shape "
                           "(%s, %s, %s)"
                           % (maskd.shape[0], maskd.shape[1], maskd.shape[2]))
            print ('%s vertices are further away than %s pixel to the mask.'
                   % (num_vertices_outside_mask, allowed_dist))

        else:
            raise pexceptions.PySegInputError(
                expr='find_vertices_outside_mask (TriangleGraph)',
                msg="A a 3D numpy ndarray object required as the first input.")

    def find_vertices_near_border_and_outside_mask(self, b, mask, label=1,
                                                   allowed_dist=0, purge=False):
        """
        Finds vertices that are within distance in nanometers to the graph
        border and outside a mask.

        Outside mask means that scaled back to pixels vertices coordinates are
        further than the allowed distance in pixels to a mask voxel with the
        given label.

        Args:
            b (float): distance from border in nanometers
            mask (numpy.ndarray): 3D mask of the segmentation from which the
                underlying surface was created
            label (int, optional): the label in the mask to be considered
                (default 1)
            allowed_dist (int, optional): allowed distance in pixels between a
                voxel coordinate and a mask voxel (default 0)
            purge (boolean, optional): if True, those vertices and their edges
                will be filtered out permanently; if False (default), no
                filtering will be done

        Returns:
            None
        """
        # don't remove vertices near border, because have to intersect with the
        # mask first!
        self.find_vertices_near_border(b, purge=False)
        self.find_vertices_outside_mask(mask, label=label,
                                        allowed_dist=allowed_dist)

        # Add a boolean vertex property telling whether a vertex within distance
        # b to border and outside mask:
        self.graph.vp.is_near_border_and_outside_mask = \
            self.graph.new_vertex_property("boolean")
        num_vertices_near_border_and_outside_mask = 0
        for v in self.graph.vertices():
            if (self.graph.vp.is_near_border[v] == 1 and
                    self.graph.vp.is_outside_mask[v] == 1):
                self.graph.vp.is_near_border_and_outside_mask[v] = 1
                num_vertices_near_border_and_outside_mask += 1
            else:
                self.graph.vp.is_near_border_and_outside_mask[v] = 0
        print ('%s vertices are within distance %s nm to the graph border and '
               'further than %s pixel from the mask.'
               % (num_vertices_near_border_and_outside_mask, b, allowed_dist))

        if purge is True:
            print 'Filtering out those vertices and their edges...'
            # Set the filter to get only vertices NOT {near border and outside
            # mask}.
            self.graph.set_vertex_filter(
                self.graph.vp.is_near_border_and_outside_mask, inverted=True
            )
            # Purge filtered out vertices and edges permanently from the graph:
            self.graph.purge_vertices()
            # Remove the properties used for filtering that are no longer true:
            del self.graph.vertex_properties["num_strong_edges"]
            del self.graph.vertex_properties["is_on_border"]
            del self.graph.vertex_properties["is_near_border"]
            # until here as in find_vertices_near_border, because not purged
            del self.graph.vertex_properties["is_outside_mask"]
            del self.graph.vertex_properties["is_near_border_and_outside_mask"]
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

    def find_largest_connected_component(self, replace=False):
        """
        Finds the largest connected component (lcc) of the graph.

        Args:
            replace (boolean, optional): if True (default False) and the lcc has
                less vertices, the graph is replaced by its lcc

        Returns:
            the lcc of the graph (graph_tool.GraphView)

        Note:
            Use lcc only if you are sure that the segmentation region was not
            split into multiple parts during surface generation of subsequent
            surface cleaning.
        """
        print ('Total number of vertices in the graph: %s'
               % self.graph.num_vertices())
        is_in_lcc = label_largest_component(self.graph)
        lcc = GraphView(self.graph, vfilt=is_in_lcc)
        print ('Number of vertices in the largest connected component of the '
               'graph: %s' % lcc.num_vertices())

        if replace is True and lcc.num_vertices() < self.graph.num_vertices():
            print ('Filtering out those vertices and edges not belonging to '
                   'the largest connected component...')
            # Set the filter to get only vertices belonging to the lcc.
            self.graph.set_vertex_filter(is_in_lcc, inverted=False)
            # Purge filtered out vertices and edges permanently from the graph:
            self.graph.purge_vertices()
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

        return lcc

    def find_small_connected_components(self, threshold=100, purge=False,
                                        verbose=False):
        """
        Finds small connected components of the graph that are below a given
        threshold size in voxels.

        Args:
            threshold (int, optional): threshold size in voxels (default 100),
                below which connected components are considered small
            purge (boolean, optional): if True (default False) and small
                components were found, their vertices and edges will be filtered
                out permanently; otherwise no filtering will be done
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        t_begin = time.time()

        comp_labels_map, sizes = label_components(self.graph)
        small_components = []
        for i, size in enumerate(sizes):
            comp_label = i
            if size < threshold:
                small_components.append(comp_label)
        print ("The graph has %s components, %s of them have size < %s"
               % (len(sizes), len(small_components), threshold))
        if verbose:
            print "Sizes of components:"
            print sizes

        # Add a boolean vertex property telling whether a vertex belongs to a
        # small component with size below the threshold:
        self.graph.vp.small_component = \
            self.graph.new_vertex_property("boolean")
        num_vertices_in_small_components = 0
        for v in self.graph.vertices():
            if comp_labels_map[v] in small_components:
                self.graph.vp.small_component[v] = 1
                num_vertices_in_small_components += 1
            else:
                self.graph.vp.small_component[v] = 0
        print ("%s vertices are in the small components."
               % num_vertices_in_small_components)

        if len(small_components) > 0 and purge is True:
            print ('Filtering out those vertices and their edges belonging to '
                   'the small components...')
            # Set the filter to get only vertices NOT belonging to a small
            # component.
            self.graph.set_vertex_filter(self.graph.vp.small_component,
                                         inverted=True)
            # Purge filtered out vertices and edges permanently from the graph:
            self.graph.purge_vertices()
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()
        # Remove the properties used for the filtering that are no longer true:
        del self.graph.vertex_properties["small_component"]

        t_end = time.time()
        duration = t_end - t_begin
        print 'Finding small components took: %s min %s s' % divmod(duration,
                                                                    60)

    def get_areas(self, verbose=False):
        """
        Gets all triangle areas in nanometers squared from the vertex properties
        of the graph and calculates the total area.

        Args:
            verbose (boolean, optional): if True (default False), prints out the
                minimal and the maximal triangle area values as well as the the
                total surface area

        Returns:
            - all triangle areas in nanometers squared (numpy.ndarray)
            - the total area in nanometers squared (float)
        """
        triangle_areas = self.graph.vp.area.get_array()
        total_area = np.sum(triangle_areas)
        if verbose:
            print '%s triangle area values' % len(triangle_areas)
            print 'min = %s, max = %s' % (min(triangle_areas),
                                          max(triangle_areas))
            print 'total surface area = %s' % total_area
        return triangle_areas, total_area

    # * The following TriangleGraph methods are implementing with adaptations
    # the normal vector voting algorithm of Page et al., 2002. *

    def collecting_normal_votes(self, vertex_v, g_max, A_max, sigma,
                                full_dist_map=None, verbose=False):
        """
        For a vertex v, collects the normal votes of all triangles within its
        geodesic neighborhood and calculates the weighted covariance matrix sum
        V_v.

        Implements equations (6), illustrated in figure 6(b), (7) and (8) from
        the paper of Page et al., 2002.

        More precisely, a normal vote N_i of each triangle i (whose centroid c_i
        is lying within the geodesic neighborhood of vertex v) is calculated
        using the normal N assigned to the triangle i. Then, each vote is
        represented by a covariance matrix V_i and votes are collected as a
        weighted matrix sum V_v, where each vote is weighted depending on the
        area of triangle i and the geodesic distance of its centroid c_i from v.

        Here, c_i and v are both centroids of triangles (v is a triangle vertex
        in Page's approach), which are vertices of TriangleGraph generated from
        the triangle surface.

        Args:
            vertex_v (graph_tool.Vertex): the vertex v in the surface
                triangle-graph for which the votes are collected
            g_max (float): the maximal geodesic distance in nanometers
            A_max (float): the area of the largest triangle in the surface
                triangle-graph
            sigma (float): sigma, defined as 3*sigma = g_max, so that votes
                beyond the neighborhood can be ignored
            full_dist_map (graph_tool.PropertyMap, optional): the full distance
                map for the whole graph; if None, a local distance map is
                calculated for this vertex (default)
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:mou
            - a dictionary of neighbors of vertex v, mapping index of each
              vertex c_i to its geodesic distance from the vertex v
            - the 3x3 symmetric matrix V_v (numpy.ndarray)
        """
        # To spare function referencing every time in the following for loop:
        vertex = self.graph.vertex
        normal = self.graph.vp.normal
        array = np.array
        xyz = self.graph.vp.xyz
        sqrt = math.sqrt
        dot = np.dot
        outer = np.multiply.outer
        area = self.graph.vp.area
        exp = math.exp

        # Get the coordinates of vertex v (as numpy array):
        v = xyz[vertex_v]
        v = array(v)

        # Find the neighboring vertices of vertex v to be returned:
        neighbor_idx_to_dist = self.find_geodesic_neighbors(
            vertex_v, g_max, full_dist_map=full_dist_map)
        try:
            assert len(neighbor_idx_to_dist) > 0
        except AssertionError:
            print ("\nWarning: the vertex v = %s has 0 neighbors. "
                   "It will be ignored later." % v)
            # return a placeholder instead of V_v
            return neighbor_idx_to_dist, np.zeros(shape=(3, 3))

        if verbose:
            print "\nv = %s" % v
            print "%s neighbors" % len(neighbor_idx_to_dist)

        # Initialize the weighted matrix sum of all votes for vertex v to be
        # calculated and returned:
        V_v = np.zeros(shape=(3, 3))

        # Let each of the neighboring vertices to cast a vote on vertex v:
        for idx_c_i in neighbor_idx_to_dist.keys():
            # Get neighboring vertex c_i and its coordinates (as numpy array):
            vertex_c_i = vertex(idx_c_i)
            c_i = xyz[vertex_c_i]
            c_i = array(c_i)

            # Calculate the normal vote N_i of c_i on v:
            N = normal[vertex_c_i]
            N = array(N)

            vc_i = c_i - v
            vc_i_len = sqrt(dot(vc_i, vc_i))
            vc_i_norm = vc_i / vc_i_len

            # theta_i is the angle between the vectors N and vc_i
            cos_theta_i = - (dot(N, vc_i)) / vc_i_len

            N_i = N + 2 * cos_theta_i * vc_i_norm

            # Covariance matrix containing one vote of c_i on v:
            V_i = outer(N_i, N_i)

            # Calculate the weight depending on the area of the neighboring
            # triangle i, A_i, and the geodesic distance to the neighboring
            # vertex c_i from vertex v, g_i:
            A_i = area[vertex_c_i]
            g_i = neighbor_idx_to_dist[idx_c_i]
            w_i = A_i / A_max * exp(- g_i / sigma)

            if verbose:
                print "\nc_i = %s" % c_i
                print "N = %s" % N
                print "vc_i = %s" % vc_i
                print "||vc_i|| = %s" % vc_i_len
                print "cos(theta_i) = %s" % cos_theta_i
                print "N_i = %s" % N_i
                print "V_i = %s" % V_i
                print "A_i = %s" % A_i
                print "g_i = %s" % g_i
                print "w_i = %s" % w_i

            # Weigh V_i and add it to the weighted matrix sum:
            V_v += w_i * V_i

        if verbose:
            print "\nV_v: %s" % V_v
        return neighbor_idx_to_dist, V_v
        # return vertex_v, neighbor_idx_to_dist, V_v

    # def collecting_normal_votes_star(self, params):
    #     """Convert `f([1,2])` to `f(1,2)` call."""
    #     return self.collecting_normal_votes(*params)

    def classifying_orientation(self, vertex_v, V_v, epsilon=2, eta=2,
                                verbose=False):
        """
        For a vertex v, its calculated matrix V_v (output of collecting_votes)
        and the parameters epsilon and eta (default 2 each), classifies its
        orientation.

        The output classes are 1 if it belongs to a surface patch, 2 if it
        belongs to a crease junction or 3 if it doesn't have a preferred
        orientation.

        This is done using eigen-decomposition of V_v and equations (9) and (10)
        from the paper of Page et al., 2002. Equations (11) and (12) may help to
        choose epsilon and eta.

        Adds the "orientation_class", the estimated normal "N_v" (if class is 1)
        and the estimated_tangent "T_v" (if class is 2) as vertex properties
        into the graph.

        Args:
            vertex_v (graph_tool.Vertex): the vertex v in the surface
                triangle-graph whose orientation is classified
            V_v (numpy.ndarray): the 3x3 symmetric matrix V_v
            epsilon (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2), default 2
            eta (int, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2) and "no preferred orientation" (class 3, see
                Notes), default 2
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            orientation of vertex v (int) - 1 if it belongs to a surface patch,
            2 if it belongs to a crease junction or 3 if it doesn't have a
            preferred orientation

        Notes:
            If epsilon = 0 and eta = 0, all triangles will be classified as
            "surface patch" (class 1).
        """
        # Decompose the symmetric semidefinite matrix V_v:
        # eigenvalues are in increasing order and eigenvectors are in columns of
        # the returned quadratic matrix
        eigenvalues, eigenvectors = np.linalg.eigh(V_v)
        # Eigenvalues from highest to lowest:
        lambda_1 = eigenvalues[2]
        lambda_2 = eigenvalues[1]
        lambda_3 = eigenvalues[0]
        # Eigenvectors, corresponding to the eigenvalues:
        E_1 = eigenvectors[:, 2]
        E_2 = eigenvectors[:, 1]
        E_3 = eigenvectors[:, 0]
        # Saliency maps:
        S_s = lambda_1 - lambda_2  # surface patch
        S_c = lambda_2 - lambda_3  # crease junction
        S_n = lambda_3  # no preferred orientation

        if verbose is True:
            print "\nlambda_1 = %s" % lambda_1
            print "lambda_2 = %s" % lambda_2
            print "lambda_3 = %s" % lambda_3
            print "E_1 = %s" % E_1
            print "E_2 = %s" % E_2
            print "E_3 = %s" % E_3
            print "S_s = %s" % S_s
            print "S_c = %s" % S_c
            print "epsilon * S_c = %s" % str(epsilon * S_c)
            print "S_n = %s" % S_n
            print "epsilon * eta * S_n = %s" % str(epsilon * eta * S_n)

        # Make decision and add the estimated normal or tangent as properties to
        # the graph (add a placeholder [0, 0, 0] to each property, where it does
        # not apply):
        max_saliency = max(S_s, epsilon * S_c, epsilon * eta * S_n)
        if max_saliency == S_s:
            # Eventually have to flip (negate) the estimated normal, because its
            # direction is lost during the matrix generation!
            # Take the one for which the angle to the original normal is smaller
            # (or cosine of the angle is higher):
            normal1 = E_1
            normal2 = -E_1
            orig_normal = self.graph.vp.normal[vertex_v]
            cos_angle1 = np.dot(orig_normal, normal1)
            cos_angle2 = np.dot(orig_normal, normal2)
            if cos_angle1 > cos_angle2:
                N_v = normal1
            else:
                N_v = normal2
            if verbose is True:
                print "surface patch with normal N_v = %s" % N_v
            self.graph.vp.orientation_class[vertex_v] = 1
            self.graph.vp.N_v[vertex_v] = N_v
            self.graph.vp.T_v[vertex_v] = np.zeros(shape=3)
            return 1
        elif max_saliency == (epsilon * S_c):
            if verbose is True:
                print "crease junction with tangent T_v = %s" % E_3
            self.graph.vp.orientation_class[vertex_v] = 2
            self.graph.vp.T_v[vertex_v] = E_3
            self.graph.vp.N_v[vertex_v] = np.zeros(shape=3)
            return 2
        else:
            if verbose is True:
                print "no preferred orientation"
            self.graph.vp.orientation_class[vertex_v] = 3
            self.graph.vp.N_v[vertex_v] = np.zeros(shape=3)
            self.graph.vp.T_v[vertex_v] = np.zeros(shape=3)
            return 3

    def collecting_curvature_votes(
            self, vertex_v, neighbor_idx_to_dist, sigma, verbose=False,
            page_curvature_formula=False, A_max=None):
        """
        For a vertex v, collects the curvature and tangent votes of all
        triangles within its geodesic neighborhood belonging to a surface patch
        and calculates the matrix B_v.

        Implements equations (13) and (15) also illustrated in figure 6(c),
        (16), (17), (14), and (5) from the paper. In short, three components are
        calculated for each neighboring vertex (representing triangle centroid)
        v_i:

        1. Weight w_i depending on the geodesic distance between v_i and v (so
            that all weights sum up to 2 * pi).

        2. Tangent T_i from v in the direction of the arc connecting v and v_i
            (using the estimated normal N_v at v).

        3. Normal curvature kappa_i from Tong and Tang et al. defined as:
            kappa_i = abs(2 * cos((pi - theta) / 2) / vector_length(vv_i)),
            where theta is the turning angle between N_v and the projection n_i
            of the estimated normal of v_i (N_v_i) onto the arc plane (formed by
            v, N_v and v_i); sign of kappa_i is opposite to the one of
            dot(T_i, n_i)

        Those components are incorporated into the 3x3 symmetric matrix B_v
        (Eq. 5), which is returned.

        Args:
            vertex_v (graph_tool.Vertex): the vertex v in the surface
                triangle-graph for which the votes are collected
            neighbor_idx_to_dist (dict): a dictionary of neighbors of vertex v,
                mapping index of each vertex v_i to its geodesic distance from
                the vertex v (output of collecting_votes)
            sigma (float): sigma, defined as 3*sigma = g_max, so that votes
                beyond the neighborhood can be ignored
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out
            page_curvature_formula (boolean, optional): if True (default False)
                normal curvature definition from Page et al. is used:
                the turning angle theta between N_v and the projection n_i of
                the estimated normal of v_i (N_v_i) onto the arc plane (formed
                by v, N_v and v_i) divided by the arc length (geodesic distance
                between v and v_i).
            A_max (boolean, optional): if given (default None), votes are
                weighted by triangle area like in the first step (normals
                estimation)
        Returns:
            the 3x3 symmetric matrix B_v (numpy.ndarray)
        """
        # To spare function referencing every time in the following for loop:
        vertex = self.graph.vertex
        vp_N_v = self.graph.vp.N_v
        orientation_class = self.graph.vp.orientation_class
        exp = math.exp
        xyz = self.graph.vp.xyz
        array = np.array
        dot = np.dot
        sqrt = math.sqrt
        pi = math.pi
        cross = np.cross
        acos = math.acos
        outer = np.multiply.outer
        cos = math.cos
        area = self.graph.vp.area

        # Get the coordinates of vertex v and its estimated normal N_v (as numpy
        # array):
        v = array(xyz[vertex_v])
        N_v = array(vp_N_v[vertex_v])

        # First, calculate the weights w_i of the neighboring triangles
        # belonging to a surface patch, because they have to be normalized to
        # sum up to 2 * pi:
        surface_neighbors_idx = []
        all_w_i = []
        for idx_v_i in neighbor_idx_to_dist.keys():
            # Get the neighboring vertex v_i:
            vertex_v_i = vertex(idx_v_i)

            # Check if the neighboring vertex v_i belongs to a surface patch
            # (otherwise don't consider it):
            if orientation_class[vertex_v_i] == 1:
                surface_neighbors_idx.append(idx_v_i)
                # Weight depending on the geodesic distance to the neighboring
                # vertex v_i from the vertex v, g_i:
                g_i = neighbor_idx_to_dist[idx_v_i]
                if A_max is None:
                    w_i = exp(- g_i / sigma)
                else:
                    A_i = area[vertex_v_i]
                    w_i = A_i / A_max * exp(- g_i / sigma)
                all_w_i.append(w_i)

        all_w_i = array(all_w_i)
        sum_w_i = np.sum(all_w_i)

        try:
            assert(sum_w_i > 0)
        except AssertionError:  # can be 0 if no surface patch neighbors exist
            print ("\nWarning: sum of the weights is not positive, but %s, for "
                   "the vertex v = %s" % (sum_w_i, v))
            print ("%s neighbors in a surface patch with weights w_i:"
                   % len(surface_neighbors_idx))
            print all_w_i
            print "The vertex will be ignored."
            return None

        wanted_sum_w_i = 2 * pi
        factor = wanted_sum_w_i / sum_w_i
        all_w_i = factor * all_w_i  # normalized weights!

        if verbose:
            print "\nv = %s" % v
            print "N_v = %s" % N_v
            print ("%s neighbors in a surface patch with normalized weights "
                   "w_i:" % len(surface_neighbors_idx))
            print all_w_i

        # Initialize the weighted matrix sum of all votes for vertex v to be
        # calculated:
        B_v = np.zeros(shape=(3, 3))

        # Let each of the neighboring triangles belonging to a surface patch
        # (as checked before) to cast a vote on vertex v:
        for i, idx_v_i in enumerate(surface_neighbors_idx):
            # Get the neighboring vertex v_i:
            vertex_v_i = vertex(idx_v_i)

            # Second, calculate tangent directions T_i of each vote:
            v_i = array(xyz[vertex_v_i])
            vv_i = v_i - v
            t_i = vv_i - dot(N_v, vv_i) * N_v
            t_i_len = sqrt(dot(t_i, t_i))
            T_i = t_i / t_i_len

            # Third, calculate the normal curvature kappa_i:
            # P_i: vector perpendicular to the plane that contains both N_v and
            # T_i as well as the normal curve (between v and v_i)
            P_i = cross(N_v, T_i)
            N_v_i = array(vp_N_v[vertex_v_i])  # estimated normal of vertex v_i
            # n_i: projection of N_v_i on the plane containing N_v rooted at v
            # and v_i
            n_i = N_v_i - dot(P_i, N_v_i) * P_i
            n_i_len = sqrt(dot(n_i, n_i))
            # theta is the turning angle between n_i and N_v
            cos_theta = dot(N_v, n_i) / n_i_len
            try:
                theta = acos(cos_theta)
            except ValueError:
                if cos_theta > 1:
                    cos_theta = 1.0
                elif cos_theta < 0:
                    cos_theta = 0.0
                theta = acos(cos_theta)
            # curvature sign has to be like this according to Page's paper:
            # kappa_i_sign = signum(dot(T_i, n_i))
            # but negated according to Tang & Medioni's definition (suitable
            # for our surface normals convention):
            kappa_i_sign = -1 * signum(dot(T_i, n_i))
            if page_curvature_formula:  # formula from Page et al. paper:
                s = neighbor_idx_to_dist[idx_v_i]  # arc length s = g_i
                kappa_i = theta / s
                # decomposition does not work if multiply kappa_i with its sign
            else:  # formula from Tong and Tang paper:
                kappa_i = abs(2 * cos((pi - theta) / 2) / sqrt(dot(vv_i, vv_i)))
                kappa_i *= kappa_i_sign

            # Recover the corresponding weight, which was calculated and
            # normalized before:
            w_i = all_w_i[i]

            if verbose:
                print "\nv_i = %s" % v_i
                print "vv_i = %s" % vv_i
                print "t_i = %s" % t_i
                print "||t_i|| = %s" % t_i_len
                print "T_i = %s" % T_i
                print "P_i = %s" % P_i
                print "N_v_i = %s" % N_v_i
                print "n_i = %s" % n_i
                print "||n_i|| = %s" % n_i_len
                print "theta = %s" % theta
                print "kappa_i = %s" % kappa_i
                print "w_i = %s" % w_i

            # Finally, sum up the components of B_v:
            B_v += w_i * kappa_i * outer(T_i, T_i)
        # and normalize it by 2 * pi:
        B_v /= (2 * pi)

        if verbose:
            print "\nB_v = %s" % B_v

        return B_v

    def estimate_curvature(self, vertex_v, B_v, verbose=False):
        """
        For a vertex v and its calculated matrix B_v (output of
        collecting_votes2), calculates the principal directions (T_1 and T_2)
        and curvatures (kappa_1 and kappa_2) at this vertex.

        This is done using eigen-decomposition of B_v: the eigenvectors
        corresponding to the two largest eigenvalues are the principal
        directions and the principal curvatures are found with linear
        transformations of those eigenvalues (Eq. 4).

        Args:
            vertex_v (graph_tool.Vertex): the vertex v in the surface
                triangle-graph for which the principal directions and curvatures
                are estimated
            B_v (numpy.ndarray): the 3x3 symmetric matrix B_v (output of
                collecting_votes2)
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        # Decompose the symmetric matrix B_v:
        # eigenvalues are in increasing order and eigenvectors are in columns of
        # the returned quadratic matrix
        eigenvalues, eigenvectors = np.linalg.eigh(B_v)
        # Eigenvalues from highest to lowest:
        b_1 = eigenvalues[2]
        b_2 = eigenvalues[1]
        b_3 = eigenvalues[0]
        # Eigenvectors that correspond to the highest two eigenvalues are the
        # estimated principal directions:
        T_1 = eigenvectors[:, 2]
        T_2 = eigenvectors[:, 1]
        T_3 = eigenvectors[:, 0]  # has to be equal to the normal N_v or -N_v
        N_v = np.array(self.graph.vp.N_v[vertex_v])
        try:
            assert(round(abs(T_3[0]), 7) == round(abs(N_v[0]), 7) and
                   round(abs(T_3[1]), 7) == round(abs(N_v[1]), 7) and
                   round(abs(T_3[2]), 7) == round(abs(N_v[2]), 7))
        except AssertionError:
            if verbose:
                print "T_3 has to be equal to the normal |N_v|, but:"
                print("T_1 = {}".format(T_1))
                print("T_2 = {}".format(T_2))
                print("T_3 = {}".format(T_3))
                print("N_v = {}".format(N_v))
                print("lambda_1 = {}".format(b_1))
                print("lambda_2 = {}".format(b_2))
                print("lambda_3 = {}".format(b_3))
            if (round(abs(T_1[0]), 7) == round(abs(N_v[0]), 7) and
                    round(abs(T_1[1]), 7) == round(abs(N_v[1]), 7) and
                    round(abs(T_1[2]), 7) == round(abs(N_v[2]), 7)):
                T_1 = T_3  # T_3 = N_v
                b_1 = b_3  # b_3 = 0
                if verbose:
                    print("Exchanged T_1 with T_3 and b_1 with b_3")
            elif (round(abs(T_2[0]), 7) == round(abs(N_v[0]), 7) and
                    round(abs(T_2[1]), 7) == round(abs(N_v[1]), 7) and
                    round(abs(T_2[2]), 7) == round(abs(N_v[2]), 7)):
                T_2 = T_3  # T_3 = N_v
                b_2 = b_3  # b_3 = 0
                if verbose:
                    print("Exchanged T_2 with T_3 and b_2 with b_3")
            else:
                print("Error: no eigenvector which equals to the normal found")
                print("T_1 = {}".format(T_1))
                print("T_2 = {}".format(T_2))
                print("T_3 = {}".format(T_3))
                print("N_v = {}".format(N_v))
                print("lambda_1 = {}".format(b_1))
                print("lambda_2 = {}".format(b_2))
                print("lambda_3 = {}".format(b_3))
                return None
        # Estimated principal curvatures:
        kappa_1 = 3 * b_1 - b_2
        kappa_2 = 3 * b_2 - b_1
        # Curvatures and directions might be interchanged:
        if kappa_1 < kappa_2:
            T_1, T_2 = T_2, T_1
            kappa_1, kappa_2 = kappa_2, kappa_1

        if verbose:
            print "\nb_1 = %s" % b_1
            print "b_2 = %s" % b_2
            print "T_1 = %s" % T_1
            print "T_2 = %s" % T_2
            print "kappa_1 = %s" % kappa_1
            print "kappa_2 = %s" % kappa_2

        # Add T_1, T_2, kappa_1, kappa_2, derived Gaussian and mean curvatures
        # as well as shape index and curvedness as properties to the graph:
        self.graph.vp.T_1[vertex_v] = T_1
        self.graph.vp.T_2[vertex_v] = T_2
        self.graph.vp.kappa_1[vertex_v] = kappa_1
        self.graph.vp.kappa_2[vertex_v] = kappa_2
        self.graph.vp.gauss_curvature_VV[vertex_v] = kappa_1 * kappa_2
        self.graph.vp.mean_curvature_VV[vertex_v] = (kappa_1 + kappa_2) / 2
        if kappa_1 == 0 and kappa_2 == 0:
            self.graph.vp.shape_index_VV[vertex_v] = 0
        else:
            self.graph.vp.shape_index_VV[vertex_v] = 2 / math.pi * math.atan(
                (kappa_1 + kappa_2) / (kappa_1 - kappa_2))
        self.graph.vp.curvedness_VV[vertex_v] = math.sqrt(
            (kappa_1 ** 2 + kappa_2 ** 2) / 2)

    def estimate_directions_and_fit_curves(self, vertex_v, B_v, radius_hit,
                                           num_points, verbose=False):
        """
        For a vertex v and its calculated matrix B_v, calculates the principal
        directions (T_1 and T_2) and curvatures (kappa_1 and kappa_2) at this
        vertex.

        This is done using eigen-decomposition of B_v: the eigenvectors
        corresponding to the two largest eigenvalues are the principal
        directions. The principal curvatures are estimated from parabolic curves
        fitted to the surface points in the principal directions.

        Args:
            vertex_v (graph_tool.Vertex): the vertex v in the surface
                triangle-graph for which the principal directions and curvatures
                are estimated
            B_v (numpy.ndarray): the 3x3 symmetric matrix B_v (output of
                collecting_votes2)
            radius_hit (float): radius in length unit of the graph, e.g.
                nanometers, for sampling surface points in tangent directions
            num_points (int): number of points to sample in each tangent
                direction in order to fit parabola and estimate curvature
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        # Decompose the symmetric matrix B_v:
        # eigenvalues are in increasing order and eigenvectors are in columns of
        # the returned quadratic matrix
        eigenvalues, eigenvectors = np.linalg.eigh(B_v)
        # Eigenvalues from highest to lowest:
        b_1 = eigenvalues[2]
        b_2 = eigenvalues[1]
        b_3 = eigenvalues[0]
        # Eigenvectors that correspond to the highest two eigenvalues are the
        # estimated principal directions:
        T_1 = eigenvectors[:, 2]
        T_2 = eigenvectors[:, 1]
        T_3 = eigenvectors[:, 0]  # has to be equal to the normal N_v or -N_v
        N_v = np.array(self.graph.vp.N_v[vertex_v])
        try:
            assert(round(abs(T_3[0]), 7) == round(abs(N_v[0]), 7) and
                   round(abs(T_3[1]), 7) == round(abs(N_v[1]), 7) and
                   round(abs(T_3[2]), 7) == round(abs(N_v[2]), 7))
        except AssertionError:
            if verbose:
                print "T_3 has to be equal to the normal |N_v|, but:"
                print("T_1 = {}".format(T_1))
                print("T_2 = {}".format(T_2))
                print("T_3 = {}".format(T_3))
                print("N_v = {}".format(N_v))
                print("lambda_1 = {}".format(b_1))
                print("lambda_2 = {}".format(b_2))
                print("lambda_3 = {}".format(b_3))
            if (round(abs(T_1[0]), 7) == round(abs(N_v[0]), 7) and
                    round(abs(T_1[1]), 7) == round(abs(N_v[1]), 7) and
                    round(abs(T_1[2]), 7) == round(abs(N_v[2]), 7)):
                T_1 = T_3  # T_3 = N_v
                # b_1 = b_3  # b_3 = 0
                if verbose:
                    print("Exchanged T_1 with T_3 and b_1 with b_3")
            elif (round(abs(T_2[0]), 7) == round(abs(N_v[0]), 7) and
                    round(abs(T_2[1]), 7) == round(abs(N_v[1]), 7) and
                    round(abs(T_2[2]), 7) == round(abs(N_v[2]), 7)):
                T_2 = T_3  # T_3 = N_v
                # b_2 = b_3  # b_3 = 0
                if verbose:
                    print("Exchanged T_2 with T_3 and b_2 with b_3")
            else:
                print("Error: no eigenvector equal to the normal |N_v| found")
                print("T_1 = {}".format(T_1))
                print("T_2 = {}".format(T_2))
                print("T_3 = {}".format(T_3))
                print("N_v = {}".format(N_v))
                print("lambda_1 = {}".format(b_1))
                print("lambda_2 = {}".format(b_2))
                print("lambda_3 = {}".format(b_3))
                return None
        # Estimate principal curvatures using curve fitting in the principal
        # directions:
        var_a_1, kappa_1 = self.find_points_in_tangent_direction_and_fit_curve(
            vertex_v, T_1, radius_hit, num_points, verbose=verbose)
        var_a_2, kappa_2 = self.find_points_in_tangent_direction_and_fit_curve(
            vertex_v, T_2, radius_hit, num_points, verbose=verbose)
        # Curvatures and directions might be interchanged:
        if kappa_1 < kappa_2:
            T_1, T_2 = T_2, T_1
            var_a_1, var_a_2 = var_a_2, var_a_1
            kappa_1, kappa_2 = kappa_2, kappa_1

        if verbose:
            print "\nT_1 = {}".format(T_1)
            print "T_2 = {}".format(T_2)
            print "fit_error_1 = {}".format(var_a_1)
            print "fit_error_2 = {}".format(var_a_2)
            print "kappa_1 = {}".format(kappa_1)
            print "kappa_2 = {}".format(kappa_2)

        # Add T_1, T_2, curve fitting errors (variances), kappa_1, kappa_2,
        # derived Gaussian and mean curvatures as well as shape index and
        # curvedness as properties to the graph:
        self.graph.vp.T_1[vertex_v] = T_1
        self.graph.vp.T_2[vertex_v] = T_2
        self.graph.vp.fit_error_1[vertex_v] = var_a_1
        self.graph.vp.fit_error_2[vertex_v] = var_a_2
        self.graph.vp.kappa_1[vertex_v] = kappa_1
        self.graph.vp.kappa_2[vertex_v] = kappa_2
        self.graph.vp.gauss_curvature_VV[vertex_v] = kappa_1 * kappa_2
        self.graph.vp.mean_curvature_VV[vertex_v] = (kappa_1 + kappa_2) / 2
        if kappa_1 == 0 and kappa_2 == 0:
            self.graph.vp.shape_index_VV[vertex_v] = 0
        else:
            self.graph.vp.shape_index_VV[vertex_v] = 2 / math.pi * math.atan(
                (kappa_1 + kappa_2) / (kappa_1 - kappa_2))
        self.graph.vp.curvedness_VV[vertex_v] = math.sqrt(
            (kappa_1 ** 2 + kappa_2 ** 2) / 2)

    def find_points_in_tangent_direction_and_fit_curve(
            self, vertex_v, tangent, radius_hit, num_points,
            poly_file=None, plot_file=None, verbose=False):
        """
        Finds points on the surface in the given direction from the given vertex
        and fits a parabolic curve.

        Args:
            vertex_v (graph_tool.Vertex): the vertex v in the surface
                triangle-graph for which the principal directions and curvatures
                are estimated
            tangent (numpy.ndarray): 3D unit tangent vector (of length 1)
            radius_hit (float): radius in length unit of the graph, e.g.
                nanometers, for sampling surface points in tangent directions
            num_points (int): maximal number of points to sample in the tangent
                direction in order to fit parabola and estimate curvature (in
                each of the two directions, without the central point)
            poly_file (str): if given vtkPolyData containing the found points
                will be constructed and saved
            plot_file (str): if given the found points will be plotted in 2D
                together with the fitted parabola
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            variance of the parabola a parameter estimate
            curvature of the fitted curve (2 * a)
        """
        # Get the coordinates of the central vertex and its estimated normal:
        v = np.array(self.graph.vp.xyz[vertex_v])
        normal = np.array(self.graph.vp.N_v[vertex_v])

        if verbose:
            print "\nv = ({},{},{})".format(v[0], v[1], v[2])
            print "T = [{},{},{}]".format(tangent[0], tangent[1], tangent[2])
            print "N = [{},{},{}]".format(normal[0], normal[1], normal[2])

        # Define a cellLocator to be able to compute intersections between lines
        # and the surface:
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(self.surface)
        locator.BuildLocator()
        tolerance = 0.001

        # Make a list of points, each point is the intersection of a vertical
        # line defined by p1 and p2 (of length 2 radius_hit) and the surface
        points = vtk.vtkPoints()
        positions_2D_x = []
        positions_2D_y = []
        sqrt = math.sqrt
        dot = np.dot
        # dist: distance on the tangent line between the perpendicular lines
        dist = float(radius_hit) / num_points

        # Add the central point v first
        points.InsertNextPoint(v)
        positions_2D_x.append(0.0)
        positions_2D_y.append(0.0)

        # First sample in the tangent direction and then in the opposite one
        for direction in [1, -1]:
            for i in range(1, num_points + 1):
                curr_dist = i * dist
                # point on line from v in tangent / opposite direction
                p0 = v + np.multiply(np.multiply(tangent, direction), curr_dist)
                # point on line from p0 in opposite normal direction
                p1 = p0 - np.multiply(normal, radius_hit)
                # point on line from p0 in normal direction
                p2 = p0 + np.multiply(normal, radius_hit)

                # Outputs (we need only pos, which is the x, y, z position
                # of the intersection:
                t = vtk.mutable(0)
                pos = [0.0, 0.0, 0.0]
                pcoords = [0.0, 0.0, 0.0]
                sub_id = vtk.mutable(0)
                cell_id = vtk.mutable(0)
                locator.IntersectWithLine(p1, p2, tolerance, t, pos, pcoords,
                                          sub_id, cell_id)
                if verbose:
                    print "\nPoint {}:".format(direction * i)

                # If no intersection was found (pos stays like initialized),
                # exclude and stop searching in that direction
                if pos == [0.0, 0.0, 0.0]:
                    if verbose:
                        print "No intersection point"
                    break

                # If euclidean distance between p0 and pos is > radius_hit (i.e.
                # pos is not between p1 and p2), exclude and stop searching in
                # that direction
                pos_p0 = sqrt(dot(p0 - pos, p0 - pos))
                if pos_p0 > radius_hit:
                    if verbose:
                        print "Point NOT within geodesic radius to the tangent"
                    break

                # Check orientation (curvature in or against the normal)
                sign = signum(dot(normal, pos - v))
                # Calculate 2D coordinates of point pos
                pos_2D_x = float(direction * curr_dist)
                pos_2D_y = float(sign * pos_p0)

                if i >= 2:
                    # If a high jump happens, i.e. distance to previous point
                    # is too high, exclude and stop searching in that direction
                    current_pos = np.array([pos_2D_x, pos_2D_y])
                    last_pos = np.array(
                        [positions_2D_x[-1], positions_2D_y[-1]])
                    before_last_pos = np.array(
                        [positions_2D_x[-2], positions_2D_y[-2]])
                    current_pos_dist = sqrt(dot(current_pos - last_pos,
                                                current_pos - last_pos))
                    last_pos_dist = sqrt(dot(last_pos - before_last_pos,
                                             last_pos - before_last_pos))
                    if current_pos_dist > 1.5 * last_pos_dist:
                        if verbose:
                            print "Point too far away from the previous one"
                            print "current distance = {}".format(
                                current_pos_dist)
                            print "last distance = {}".format(last_pos_dist)
                        break

                # Add the x, y, z position of the intersection
                points.InsertNextPoint(pos)
                positions_2D_x.append(pos_2D_x)
                positions_2D_y.append(pos_2D_y)
                if verbose:
                    print "Added a valid point"
                #     print "p0 = ({},{},{})".format(p0[0], p0[1], p0[2])
                #     print "p1 = ({},{},{})".format(p1[0], p1[1], p1[2])
                #     print "p2 = ({},{},{})".format(p2[0], p2[1], p2[2])
                #     print "point = ({}, {}, {})".format(pos[0], pos[1], pos[2])
                #     print "cell id = {}".format(cell_id)
                #     print "sign = {}".format(sign)
                #     print "coordinates in 2D = ({}, {})".format(
                #         pos_2D_x, pos_2D_y)

        # Fit a simple parabola curve:
        a, var_a = fit_curve(positions_2D_x, positions_2D_y)  # a = 1 / (2 * R)
        curvature = 2 * a  # curvature = 1 / R

        if verbose:  # or var_a == 1 or var_a == -1:
            print ("{} intersection points found".format(
                points.GetNumberOfPoints()))
            print "variance = {}".format(var_a)
            print "curvature = {}".format(curvature)

        if poly_file is not None:  # vtkPolyData construction
            poly_verts = vtk.vtkPolyData()
            poly_verts.SetPoints(points)
            verts = vtk.vtkCellArray()  # vertex (1-point) cells
            for i in range(points.GetNumberOfPoints()):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            poly_verts.SetVerts(verts)
            save_vtp(poly_verts, poly_file)

        if plot_file is not None:  # or var_a == 1 or var_a == -1:  # 2D plot
            fig = plt.figure()
            # plot the intersection points
            plt.plot(positions_2D_x, positions_2D_y, 'ro')
            # plot the estimated parabola function
            x = np.linspace(min(positions_2D_x), max(positions_2D_x), 100)
            y = [canonical_parabola(x_i, a) for x_i in x]
            plt.plot(x, y)
            # add grey lines parallel to axes at (0, 0)
            plt.axvline(x=0, color='grey', linewidth=0.5)
            plt.axhline(y=0, color='grey', linewidth=0.5)
            # make axes scale equal and add labels
            plt.axis('equal')
            plt.xlabel("tangent")
            plt.ylabel("normal")
            if plot_file is not None:
                fig.savefig(plot_file)
            else:
                plt.show()

        return var_a, curvature

    def gen_curv_vote(self, vertex_r, radius_hit, verbose=False):
        """
        Implements the third pass of the method of Tong & Tang et al., 2005,
        "Algorithm 5. GenCurvVote". Estimates principal curvatures and
        directions (after normals estimation) using curvature tensor. For the
        given triangle center, eight neighboring points are sampled around it at
        equal angles using tangents of length defined by the RadiusHit
        parameter.

        Args:
            vertex_r (graph_tool.Vertex): the vertex r in the surface
                triangle-graph for which the principal directions and curvatures
                are estimated
            radius_hit (float): radius in length unit of the graph, e.g.
                nanometers, for sampling surface points in tangent directions
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            kappa_1 (float): maximal principal curvature
            kappa_2 (float): minimal principal curvature
            T_1 (3D vector of floats): maximal principal direction
            T_2 (3D vector of floats): minimal principal direction
        """
        # Get the coordinates of vertex r and its estimated normal N_r (as numpy
        # array, input of the original method):
        r = np.array(self.graph.vp.xyz[vertex_r])
        N_r = np.array(self.graph.vp.N_v[vertex_r])
        if verbose:
            print("\nr = ({}, {}, {})".format(r[0], r[1], r[2]))
            print("N_r = ({}, {}, {})".format(N_r[0], N_r[1], N_r[2]))

        # Define a cellLocator to be able to compute intersections between lines
        # and the surface:
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(self.surface)
        locator.BuildLocator()
        tolerance = 0.001

        # Define some frequently used functions in the loop:
        multiply = np.multiply
        sqrt = math.sqrt
        dot = np.dot
        outer = np.multiply.outer
        pi = math.pi

        # a vector on tangent plane T_r(S)
        votedir = perpendicular_vector(N_r, debug=verbose)
        if votedir is None:
            print("Error: calculation of a perpendicular vector failed")
            exit(0)
        # M_r = np.zeros(shape=(2, 2))  # when transform votedir to 2D
        M_r = np.zeros(shape=(3, 3))
        R = rotation_matrix(N_r, pi/4)
        # num_valid_votes = 0
        for i in range(8):
            # rotate the vector by 45 degrees (pi/4 radians) around N_r axis
            votedir = rotate_vector(votedir, pi/4, matrix=R, debug=verbose)
            r_t = r + votedir * radius_hit

            # Find intersection point c between the surface and line segment l
            # going through r_t and parallel to N_r:
            # point on line l from r_t in normal direction
            p1 = r_t + multiply(N_r, radius_hit)
            # point on line l from r_t in opposite normal direction
            p2 = r_t - multiply(N_r, radius_hit)
            # Outputs (we need only c, which is the x, y, z position
            # of the intersection):
            t = vtk.mutable(0)
            c = [0.0, 0.0, 0.0]
            pcoords = [0.0, 0.0, 0.0]
            sub_id = vtk.mutable(0)
            cell_id = vtk.mutable(0)
            locator.IntersectWithLine(p1, p2, tolerance, t, c, pcoords,
                                      sub_id, cell_id)
            # If there is no intersection, c stays like initialized:
            if c == [0.0, 0.0, 0.0]:
                if verbose:
                    print("No intersection point found")
                continue  # in paper "return None", but I think if does not make
                # sense to give up if there is no continuation of the surface at
                # radius_hit distance just in one or some of the 8 directions

            b = sqrt(dot(r_t - c, r_t - c))
            if b > radius_hit:
                if verbose:
                    print("b = {}, higher than RadiusHit = {}".format(
                        b, radius_hit))
                continue  # in paper "return None" ...
            k_rc = 2 * b / (b ** 2 + radius_hit ** 2)
            # sign(c) = 1 if c is above the tangent plane T_r(S)
            #          -1 if c is below T_r(S)
            #           0 if c lies on T_r(S)
            sign_c = signum(dot(N_r, c - r))

            outer_product = outer(votedir, votedir)
            multiplicator = sign_c * k_rc
            M_r += multiply(outer_product, multiplicator)
            # num_valid_votes += 1

            if verbose:
                print("\nvotedir = ({}, {}, {})".format(
                    votedir[0], votedir[1], votedir[2]))
                print("r_t = ({}, {}, {})".format(r_t[0], r_t[1], r_t[2]))
                print("c = ({}, {}, {})".format(c[0], c[1], c[2]))
                print("b = {}".format(b))
                print("k_rc = {}".format(k_rc))
                print("sign_c = {}".format(sign_c))

        M_r /= 8
        # Decompose the symmetric matrix M_r:
        # eigenvalues are in increasing order and eigenvectors are in columns of
        # the returned quadratic matrix
        eigenvalues, eigenvectors = np.linalg.eigh(M_r)
        # Eigenvalues from highest to lowest:
        lambda_1 = eigenvalues[2]  # in 2D eigenvalues[1]
        lambda_2 = eigenvalues[1]  # in 2D eigenvalues[0]
        lambda_3 = eigenvalues[0]
        # Eigenvectors that correspond to the highest two eigenvalues are the
        # estimated principal directions:
        T_1 = eigenvectors[:, 2]  # in 2D eigenvectors[:, 1]
        T_2 = eigenvectors[:, 1]  # in 2D eigenvectors[:, 0]
        T_3 = eigenvectors[:, 0]
        try:
            assert(round(abs(T_3[0]), 7) == round(abs(N_r[0]), 7) and
                   round(abs(T_3[1]), 7) == round(abs(N_r[1]), 7) and
                   round(abs(T_3[2]), 7) == round(abs(N_r[2]), 7))
        except AssertionError:
            if verbose:
                print "T_3 has to be equal to the normal |N_r|, but:"
                print("T_1 = {}".format(T_1))
                print("T_2 = {}".format(T_2))
                print("T_3 = {}".format(T_3))
                print("N_r = {}".format(N_r))
                print("lambda_1 = {}".format(lambda_1))
                print("lambda_2 = {}".format(lambda_2))
                print("lambda_3 = {}".format(lambda_3))
            if (round(abs(T_1[0]), 7) == round(abs(N_r[0]), 7) and
                    round(abs(T_1[1]), 7) == round(abs(N_r[1]), 7) and
                    round(abs(T_1[2]), 7) == round(abs(N_r[2]), 7)):
                T_1 = T_3
                lambda_1 = lambda_3
                # T_3 = N_r; lambda_3 = 0
                if verbose:
                    print("Exchanged T_1 with T_3 and lambda_1 with lambda_3")
            elif (round(abs(T_2[0]), 7) == round(abs(N_r[0]), 7) and
                    round(abs(T_2[1]), 7) == round(abs(N_r[1]), 7) and
                    round(abs(T_2[2]), 7) == round(abs(N_r[2]), 7)):
                T_2 = T_3
                lambda_2 = lambda_3
                # T_3 = N_r; lambda_3 = 0
                if verbose:
                    print("Exchanged T_2 with T_3 and lambda_2 with lambda_3")
            else:
                # print("Error: no eigenvector which equals to the normal found")
                return None
        # Estimated principal curvatures:
        kappa_1 = 3 * lambda_1 - lambda_2
        kappa_2 = 3 * lambda_2 - lambda_1
        # Curvatures and directions might be interchanged:
        if kappa_1 < kappa_2:
            T_1, T_2 = T_2, T_1
            kappa_1, kappa_2 = kappa_2, kappa_1
        if verbose:
            # print("\nNumber valid votes = {}".format(num_valid_votes))
            print("\nT_1 = {}".format(T_1))
            print("T_2 = {}".format(T_2))
            print("kappa_1 = {}".format(kappa_1))
            print("kappa_2 = {}".format(kappa_2))

        # Add T_1, T_2, kappa_1, kappa_2, derived Gaussian and mean curvatures
        # as well as shape index and curvedness as properties to the graph:
        self.graph.vp.T_1[vertex_r] = T_1
        self.graph.vp.T_2[vertex_r] = T_2
        self.graph.vp.kappa_1[vertex_r] = kappa_1
        self.graph.vp.kappa_2[vertex_r] = kappa_2
        self.graph.vp.gauss_curvature_VV[vertex_r] = kappa_1 * kappa_2
        self.graph.vp.mean_curvature_VV[vertex_r] = (kappa_1 + kappa_2) / 2
        if kappa_1 == 0 and kappa_2 == 0:
            self.graph.vp.shape_index_VV[vertex_r] = 0
        else:
            self.graph.vp.shape_index_VV[vertex_r] = 2 / pi * math.atan(
                (kappa_1 + kappa_2) / (kappa_1 - kappa_2))
        self.graph.vp.curvedness_VV[vertex_r] = sqrt(
            (kappa_1 ** 2 + kappa_2 ** 2) / 2)
        return kappa_1, kappa_2, T_1, T_2
