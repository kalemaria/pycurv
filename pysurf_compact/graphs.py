"""
Contains an abstract class (SegmentationGraph) for representing a segmentation by a graph with attributes and methods common for all derived classes.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'

import math
import vtk
import numpy as np
from datetime import datetime
from graph_tool import Graph
from graph_tool.topology import shortest_distance

from pysurf_io import TypesConverter
import pexceptions


class SegmentationGraph(object):
    """
    Class defining the abstract SegmentationGraph object, its attributes and implements methods common to all derived graph classes.

    The constructor requires the following parameters of the underlying segmentation that will be used to build the graph.

    Args:
        scale_factor_to_nm (float): pixel size in nanometers for scaling the graph
        scale_x (int): x axis length in pixels of the segmentation
        scale_y (int): y axis length in pixels of the segmentation
        scale_z (int): z axis length in pixels of the segmentation
    """

    def __init__(self, scale_factor_to_nm, scale_x, scale_y, scale_z):
        """
        Constructor.

        Args:
            scale_factor_to_nm (float): pixel size in nanometers for scaling the graph
            scale_x (int): x axis length in pixels of the segmentation
            scale_y (int): y axis length in pixels of the segmentation
            scale_z (int): z axis length in pixels of the segmentation

        Returns:
            None
        """
        self.graph = Graph(directed=False)
        self.scale_factor_to_nm = scale_factor_to_nm
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z

        # Add "internal property maps" to the graph.
        # vertex property for storing the xyz coordinates of the corresponding vertex scaled in nm:
        self.graph.vp.xyz = self.graph.new_vertex_property("vector<float>")
        # edge property storing the distance between the connected vertices in nm:
        self.graph.ep.distance = self.graph.new_edge_property("float")

        # A dictionary mapping the vertex coordinates (x, y, z) scaled in nm to the vertex index:
        self.coordinates_to_vertex_index = {}
        # A dictionary of connected coordinates pairs in form ((x1, y1, z1), (x2, y2, z2)) scaled in nm -> True
        self.coordinates_pair_connected = {}

    @staticmethod
    def distance_between_voxels(voxel1, voxel2):
        """
        Calculates and returns the Euclidean distance between two voxels.

        Args:
            voxel1 (tuple): first voxel coordinates in form of a tuple of integers of length 3 (x1, y1, z1)
            voxel2 (tuple): second voxel coordinates in form of a tuple of integers of length 3 (x2, y2, z2)

        Returns:
            the Euclidean distance between two voxels (float)

        """
        if isinstance(voxel1, tuple) and (len(voxel1) == 3) and isinstance(voxel2, tuple) and (len(voxel2) == 3):
            sum_of_squared_differences = 0
            for i in range(3):  # for each dimension
                sum_of_squared_differences += (voxel1[i] - voxel2[i]) ** 2
            return math.sqrt(sum_of_squared_differences)
        else:
            error_msg = 'Tuples of integers of length 3 required as first and second input.'
            raise pexceptions.PySegInputError(expr='distance_between_voxels (SegmentationGraph)', msg=error_msg)

    def update_coordinates_to_vertex_index(self):
        """
        Updates graph's dictionary coordinates_to_vertex_index.

        This has to be done after purging the graph, because vertices are renumbered, as well as after reading a graph from a file (e.g. before density calculation).
        Reminder: the dictionary maps the vertex coordinates (x, y, z) scaled in nanometers to the vertex index.

        Returns:
            None
        """
        self.coordinates_to_vertex_index = {}
        for vd in self.graph.vertices():
            [x, y, z] = self.graph.vp.xyz[vd]
            self.coordinates_to_vertex_index[(x, y, z)] = self.graph.vertex_index[vd]

    def calculate_density(self, mask=None, target_coordinates=None, verbose=False):
        """
        Calculates ribosome density for each membrane graph vertex.

        Calculates shortest geodesic distances (d) for each vertex in the graph to each reachable ribosome center mapped on the membrane given by a binary mask with coordinates in pixels
        or an array of coordinates in nm. Then, calculates a density measure of ribosomes at each vertex / membrane voxel: D = sum {over all reachable ribosomes} (1 / (d + 1)).
        Adds the density as vertex PropertyMap to the graph. Returns an array with the same shape as the underlying segmentation with the densities + 1, in order to distinguish membrane
        voxels with 0 density from the background.

        Args:
            mask (numpy.ndarray, optional): a binary mask of the ribosome centers as 3D array where indices are coordinates in pixels (default None)
            target_coordinates (numpy.ndarray, optional): the ribosome centers coordinates in nm as 2D array in format [[x1, y1, z1], [x2, y2, z2], ...] (default None)
            verbose (boolean, optional): if True (default False), some extra information will be printed out

        Returns:
            a 3D numpy ndarray with the densities + 1

        Note:
            One of the first two parameters, mask or target_coordinates, has to be given.
        """
        import ribosome_density as rd
        # If a mask is given, find the set of voxels of ribosome centers mapped on the membrane, 'target_voxels', and rescale them to nm, 'target_coordinates':
        if mask is not None:
            if mask.shape != (self.scale_x, self.scale_y, self.scale_z):
                error_msg = "Scales of the input 'mask' have to be equal to those set during the generation of the graph object."
                raise pexceptions.PySegInputError(expr='calculate_density (SegmentationGraph)', msg=error_msg)
            target_voxels = rd.get_foreground_voxels_from_mask(mask)  # output as a list of tuples: [(x1,y1,z1), (x2,y2,z2), ...] in pixels
            target_ndarray_voxels = rd.tupel_list_to_ndarray_voxels(target_voxels)  # for rescaling have to convert to a ndarray
            target_ndarray_coordinates = target_ndarray_voxels * self.scale_factor_to_nm  # rescale to nm, output a ndarray: [[x1,y1,z1], [x2,y2,z2], ...]
            target_coordinates = rd.ndarray_voxels_to_tupel_list(target_ndarray_coordinates)  # convert to a list of tupels, which are in nm now
        # If target_coordinates are given (in nm), convert them from a numpy ndarray to a list of tuples:
        elif target_coordinates is not None:
            target_coordinates = rd.ndarray_voxels_to_tupel_list(target_coordinates)  # to convert numpy ndarray to a list of tuples: [(x1,y1,z1), (x2,y2,z2), ...]
        # Exit if the target_voxels list is empty:
        if len(target_coordinates) == 0:
            error_msg = "No target voxels were found! Check your input ('mask' or 'target_coordinates')."
            raise pexceptions.PySegInputError(expr='calculate_density (SegmentationGraph)', msg=error_msg)
        print '%s target voxels' % len(target_coordinates)
        if verbose:
            print target_coordinates

        # Pre-filter the target coordinates to those existing in the graph (should already all be in the graph, but just in case):
        target_coordinates_in_graph = []
        for target_xyz in target_coordinates:
            if target_xyz in self.coordinates_to_vertex_index:
                target_coordinates_in_graph.append(target_xyz)
            else:
                error_msg = 'Target (%s, %s, %s) not inside the membrane!' % (target_xyz[0], target_xyz[1], target_xyz[2])
                raise pexceptions.PySegInputWarning(expr='calculate_density (SegmentationGraph)', msg=error_msg)

        print '%s target coordinates in graph' % len(target_coordinates_in_graph)
        if verbose:
            print target_coordinates_in_graph

        # Get all indices of the target coordinates:
        target_vertices_indices = []
        for target_xyz in target_coordinates_in_graph:
            v_target_index = self.coordinates_to_vertex_index[target_xyz]
            target_vertices_indices.append(v_target_index)

        # Density calculation
        # Add a new vertex property to the graph, density:
        self.graph.vp.density = self.graph.new_vertex_property("float")
        # Dictionary mapping voxel coordinates (for the volume returned later) to a list of density values falling within that voxel:
        voxel_to_densities = {}

        # For each vertex in the graph:
        for v_membrane in self.graph.vertices():
            # Get its coordinates:
            membrane_xyz = self.graph.vp.xyz[v_membrane]
            if verbose:
                print 'Membrane vertex (%s, %s, %s)' % (membrane_xyz[0], membrane_xyz[1], membrane_xyz[2])
            # Get a distance map with all pairs of distances between current graph vertex (membrane_xyz) and target vertices (ribosome coordinates):
            dist_map = shortest_distance(self.graph, source=v_membrane, target=target_vertices_indices, weights=self.graph.ep.distance)

            # Iterate over all shortest distances from the membrane vertex to the target vertices, while calculating the density:
            # Initializing: membrane coordinates with no reachable ribosomes will have a value of 0, those with reachable ribosomes > 0.
            density = 0
            # If there is only one target voxel, dist_map is a single value - wrap it into a list.
            if len(target_coordinates_in_graph) == 1:
                dist_map = [dist_map]
            for d in dist_map:
                if verbose:
                    print '\tTarget vertex ...'
                if d == np.finfo(np.float64).max:  # == if unreachable, the maximum float64 is stored: 1.7976931348623157e+308
                    if verbose:
                        print '\t\tunreachable'
                else:
                    if verbose:
                        print '\t\td = %s' % d
                    density += 1 / (d + 1)

            # Add the density of the membrane vertex as a property of the current vertex in the graph:
            self.graph.vp.density[v_membrane] = density

            # Calculate the corresponding voxel of the vertex and add the density to the list keyed by the voxel in the dictionary:
            # Scaling the coordinates back from nm to voxels. (Without round float coordinates are truncated to the next lowest integer.)
            voxel_x = int(round(membrane_xyz[0] / self.scale_factor_to_nm))
            voxel_y = int(round(membrane_xyz[1] / self.scale_factor_to_nm))
            voxel_z = int(round(membrane_xyz[2] / self.scale_factor_to_nm))
            voxel = (voxel_x, voxel_y, voxel_z)
            if voxel in voxel_to_densities:
                voxel_to_densities[voxel].append(density)
            else:
                voxel_to_densities[voxel] = [density]

            if verbose:
                print '\tdensity = %s' % density
            if (self.graph.vertex_index[v_membrane] + 1) % 1000 == 0:
                now = datetime.now()
                print '%s membrane vertices processed on: %s-%s-%s %s:%s:%s' \
                      % (self.graph.vertex_index[v_membrane] + 1, now.year, now.month, now.day, now.hour, now.minute, now.second)

        # Initialize an array scaled like the original segmentation, which will hold in each membrane voxel the maximal density among the corresponding vertex coordinates in the graph plus 1
        # and 0 in each background (non-membrane) voxel:
        densities = np.zeros((self.scale_x, self.scale_y, self.scale_z), dtype=np.float16)
        # The densities array membrane voxels are initialized with 1 in order to distinguish membrane voxels from the background.
        for voxel in voxel_to_densities:
            densities[voxel[0], voxel[1], voxel[2]] = 1 + max(voxel_to_densities[voxel])
        if verbose:
            print 'densities:\n%s' % densities
        return densities

    def graph_to_points_and_lines_polys(self, vertices=True, edges=True, verbose=False):
        """
        Generates a VTK PolyData object from the graph with vertices as vertex-cells (containing 1 point) and edges as line-cells (containing 2 points).

        Args:
            vertices (boolean, optional): if True (default) vertices are stored a VTK PolyData object as vertex-cells
            edges (boolean, optional): if True (default) edges are stored a VTK PolyData object as line-cells
            verbose (boolean, optional): if True (default False), some extra information will be printed out

        Returns:
            - vtk.vtkPolyData with vertex-cells
            - vtk.vtkPolyData with edges as line-cells
        """
        # Initialization
        poly_verts = vtk.vtkPolyData()
        poly_lines = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        vertex_arrays = list()
        edge_arrays = list()
        # Vertex property arrays
        for prop_key in self.graph.vp.keys():
            data_type = self.graph.vp[prop_key].value_type()
            if data_type != 'string' and data_type != 'python::object' and prop_key != 'xyz':
                if verbose:
                    print '\nvertex property key: %s' % prop_key
                    print 'value type: %s' % data_type
                if data_type[0:6] != 'vector':  # scalar
                    num_components = 1
                else:  # vector
                    num_components = len(self.graph.vp[prop_key][self.graph.vertex(0)])
                array = TypesConverter().gt_to_vtk(data_type)
                array.SetName(prop_key)
                if verbose:
                    print 'number of components: %s' % num_components
                array.SetNumberOfComponents(num_components)
                vertex_arrays.append(array)
        # Edge property arrays
        for prop_key in self.graph.ep.keys():
            data_type = self.graph.ep[prop_key].value_type()
            if data_type != 'string' and data_type != 'python::object':
                if verbose:
                    print '\nedge property key: %s' % prop_key
                    print 'value type: %s' % data_type
                if data_type[0:6] != 'vector':  # scalar
                    num_components = 1
                else:  # vector
                    # num_components = len(self.graph.ep[prop_key][self.graph.edge(0, 1)])
                    num_components = 3
                    if verbose:
                        print 'Sorry, not implemented yet, assuming a vector with 3 components.'  # all edge properties so far are scalars
                array = TypesConverter().gt_to_vtk(data_type)
                array.SetName(prop_key)
                if verbose:
                    print 'number of components: %s' % num_components
                array.SetNumberOfComponents(num_components)
                edge_arrays.append(array)
        if verbose:
            print '\nvertex arrays length: %s' % len(vertex_arrays)
            print 'edge arrays length: %s' % len(edge_arrays)

        # Geometry
        lut = np.zeros(shape=self.graph.num_vertices(), dtype=np.int)
        for i, vd in enumerate(self.graph.vertices()):
            [x, y, z] = self.graph.vp.xyz[vd]
            points.InsertPoint(i, x, y, z)
            lut[self.graph.vertex_index[vd]] = i
        if verbose:
            print 'number of points: %s' % points.GetNumberOfPoints()

        # Topology
        # Vertices
        verts = vtk.vtkCellArray()
        if vertices:
            for vd in self.graph.vertices():  # vd = vertex descriptor
                verts.InsertNextCell(1)
                verts.InsertCellPoint(lut[self.graph.vertex_index[vd]])
                for array in vertex_arrays:
                    prop_key = array.GetName()
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.graph.vp[prop_key].value_type()
                    data_type = TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.get_vertex_prop_entry(prop_key, vd, n_comp, data_type))
            if verbose:
                print 'number of vertex cells: %s' % verts.GetNumberOfCells()
        # Edges
        lines = vtk.vtkCellArray()
        if edges:
            for ed in self.graph.edges():  # ed = edge descriptor
                lines.InsertNextCell(2)
                lines.InsertCellPoint(lut[self.graph.vertex_index[ed.source()]])
                lines.InsertCellPoint(lut[self.graph.vertex_index[ed.target()]])
                for array in edge_arrays:
                    prop_key = array.GetName()
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.graph.ep[prop_key].value_type()
                    data_type = TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.get_edge_prop_entry(prop_key, ed, n_comp, data_type))
            if verbose:
                print 'number of line cells: %s' % lines.GetNumberOfCells()

        # vtkPolyData construction
        poly_verts.SetPoints(points)
        poly_lines.SetPoints(points)
        if vertices:
            poly_verts.SetVerts(verts)
        if edges:
            poly_lines.SetLines(lines)
        for array in vertex_arrays:
            poly_verts.GetCellData().AddArray(array)
        for array in edge_arrays:
            poly_lines.GetCellData().AddArray(array)

        return poly_verts, poly_lines

    def get_vertex_prop_entry(self, prop_key, vertex_descriptor, n_comp, data_type):
        """
        Gets a property value of a vertex for inserting into a VTK vtkDataArray object.

        This private function is used by the methods graph_to_points_and_lines_polys and graph_to_triangle_poly (the latter of the derived class surface_graphs.TriangleGraph).

        Args:
            prop_key (str): name of the desired vertex property
            vertex_descriptor (graph_tool.Vertex): vertex descriptor of the current vertex
            n_comp (int): number of components of the array (length of the output tuple)
            data_type: numpy data type converted from a graph-tool property value type by TypesConverter().gt_to_numpy

        Returns:
            a tuple (with length like n_comp) with the property value of the vertex converted to a numpy data type
        """
        prop = list()
        if n_comp == 1:
            prop.append(data_type(self.graph.vp[prop_key][vertex_descriptor]))
        else:
            for i in range(n_comp):
                prop.append(data_type(self.graph.vp[prop_key][vertex_descriptor][i]))
        return tuple(prop)

    def get_edge_prop_entry(self, prop_key, edge_descriptor, n_comp, data_type):
        """
        Gets a property value of an edge for inserting into a VTK vtkDataArray object.

        This private function is used by the method graph_to_points_and_lines_polys.

        Args:
            prop_key (str): name of the desired vertex property
            edge_descriptor (graph_tool.Edge): edge descriptor of the current edge
            n_comp (int): number of components of the array (length of the output tuple)
            data_type: numpy data type converted from a graph-tool property value type by TypesConverter().gt_to_numpy

        Returns:
            a tuple (with length like n_comp) with the property value of the edge converted to a numpy data type
        """
        prop = list()
        if n_comp == 1:
            prop.append(data_type(self.graph.ep[prop_key][edge_descriptor]))
        else:
            for i in range(n_comp):
                prop.append(data_type(self.graph.ep[prop_key][edge_descriptor][i]))
        return tuple(prop)

    # * The following SegmentationGraph methods are needed for the normal vector voting algorithm. *

    def calculate_average_edge_length(self, prop_e=None, value=1):
        """
        Calculates the average edge length in the graph.

        If a special edge property is specified, includes only the edges where this property equals the given value.
        If there are no edges in the graph, the given property does not exist or there are no edges with the given property equaling the given value, None is returned.

        Args:
            prop_e (str, optional): edge property, if specified only edges where this property equals the given value will be considered
            value (int, optional): value of the specified edge property an edge has to have in order to be considered (default 1)

        Returns:
            the average edge length in the graph (float) or None
        """
        total_edge_length = 0
        average_edge_length = None
        if prop_e is None:
            print "Considering all edges:"
            for ed in self.graph.edges():
                total_edge_length += self.graph.ep.distance[ed]
            if self.graph.num_edges() > 0:
                average_edge_length = total_edge_length / self.graph.num_edges()
            else:
                print "There are no edges in the graph!"
        elif prop_e in self.graph.edge_properties:
            print "Considering only edges with property %s equaling value %s " % (prop_e, value)
            num_special_edges = 0
            for ed in self.graph.edges():
                if self.graph.edge_properties[prop_e][ed] == value:
                    num_special_edges += 1
                    total_edge_length += self.graph.ep.distance[ed]
            if num_special_edges > 0:
                average_edge_length = total_edge_length / num_special_edges
            else:
                print "There are no edges with the property %s equaling value %s!" % (prop_e, value)
        print "Average length: %s" % average_edge_length
        return average_edge_length

    def find_geodesic_neighbors(self, v, g_max, verbose=False):
        """
        Finds geodesic neighbor vertices of a given vertex v in the graph that are within a given maximal geodesic distance g_max from it.

        Also finds the corresponding geodesic distances. All edges are considered.

        Args:
            v (graph_tool.Vertex): the source vertex
            g_max: maximal geodesic distance (in nanometers, if the graph was scaled)
            verbose (boolean, optional): if True (default False), some extra information will be printed out

        Returns:
            a dictionary mapping a neighbor vertex index to the geodesic distance from vertex v
        """
        dist_v = shortest_distance(self.graph, source=v, target=None, weights=self.graph.ep.distance, max_dist=g_max)
        dist_v = dist_v.get_array()

        neighbor_id_to_dist = dict()

        idxs = np.where(dist_v <= g_max)[0]
        for idx in idxs:
            dist = dist_v[idx]
            if dist != 0:  # ignore the source vertex itself
                neighbor_id_to_dist[idx] = dist

        if verbose:
            print "%s neighbors" % len(neighbor_id_to_dist)
        return neighbor_id_to_dist

    def get_vertex_property_array(self, property_name):
        """
        Gets a numpy array with all values of a vertex property of the graph, printing out the number of values, the minimal and the maximal value.

        Args:
            property_name (str): vertex property name

        Returns:
            an array (numpy.ndarray) with all values of the vertex property
        """
        if isinstance(property_name, str) and property_name in self.graph.vertex_properties:
            values = self.graph.vertex_properties[property_name].get_array()
            print '%s "%s" values' % (len(values), property_name)
            print 'min = %s, max = %s' % (min(values), max(values))
            return values
        else:
            error_msg = 'The input "%s" is not a str object or is not found in vertex properties of the graph.' % property_name
            raise pexceptions.PySegInputError(expr='get_vertex_property_array (SegmentationGraph)', msg=error_msg)
