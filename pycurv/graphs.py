import math
import vtk
import numpy as np
from datetime import datetime
from graph_tool import Graph
from graph_tool.topology import shortest_distance
from heapq import heappush, heappop

from .pycurv_io import TypesConverter
from . import pexceptions
from .linalg import nice_acos, euclidean_distance

"""
Contains an abstract class (SegmentationGraph) for representing a segmentation
by a graph with attributes and methods common for all derived classes.

Author: Maria Salfer (Max Planck Institute for Biochemistry)
"""

__author__ = 'Maria Salfer'


class SegmentationGraph(object):
    """
    Class defining the abstract SegmentationGraph object, its attributes and
    implements methods common to all derived graph classes.

    The constructor requires the following parameters of the underlying
    segmentation that will be used to build the graph.
    """

    def __init__(self):
        """
        Constructor of the abstract SegmentationGraph object.

        Returns:
            None
        """
        self.graph = Graph(directed=False)
        """graph_tool.Graph: a graph object storing the segmentation graph
        topology, geometry and properties (initially empty).
        """

        # Add "internal property maps" to the graph.
        # vertex property for storing the xyz coordinates of the corresponding
        # vertex:
        self.graph.vp.xyz = self.graph.new_vertex_property("vector<float>")
        # edge property for storing the distance between the connected vertices:
        self.graph.ep.distance = self.graph.new_edge_property("float")

        self.coordinates_to_vertex_index = {}
        """dict: a dictionary mapping the vertex coordinates (x, y, z) to the
        vertex index.
        """
        self.coordinates_pair_connected = set()
        """set: a set storing pairs of vertex coordinates that are
        connected by an edge in a tuple form ((x1, y1, z1), (x2, y2, z2)).
        """

    @staticmethod
    def distance_between_voxels(voxel1, voxel2):
        """
        Calculates and returns the Euclidean distance between two voxels.

        Args:
            voxel1 (tuple): first voxel coordinates in form of a tuple of
                floats of length 3 (x1, y1, z1)
            voxel2 (tuple): second voxel coordinates in form of a tuple of
                floats of length 3 (x2, y2, z2)

        Returns:
            the Euclidean distance between two voxels (float)
        """
        if (isinstance(voxel1, tuple) and (len(voxel1) == 3) and
                isinstance(voxel2, tuple) and (len(voxel2) == 3)):
            sum_of_squared_differences = 0
            for i in range(3):  # for each dimension
                sum_of_squared_differences += (voxel1[i] - voxel2[i]) ** 2
            return math.sqrt(sum_of_squared_differences)
        else:
            raise pexceptions.PySegInputError(
                expr='distance_between_voxels (SegmentationGraph)',
                msg=('Tuples of integers of length 3 required as first and '
                     'second input.'))

    def update_coordinates_to_vertex_index(self):
        """
        Updates graph's dictionary coordinates_to_vertex_index.

        The dictionary maps the vertex coordinates (x, y, z) to the vertex
        index. It has to be updated after purging the graph, because vertices
        are renumbered, as well as after reading a graph from a file (e.g.
        before density calculation).

        Returns:
            None
        """
        self.coordinates_to_vertex_index = {}
        for vd in self.graph.vertices():
            [x, y, z] = self.graph.vp.xyz[vd]
            self.coordinates_to_vertex_index[
                (x, y, z)] = self.graph.vertex_index[vd]

    def calculate_density(self, size, scale, mask=None, target_coordinates=None,
                          verbose=False):
        """
        Calculates ribosome density for each membrane graph vertex.

        Calculates shortest geodesic distances (d) for each vertex in the graph
        to each reachable ribosome center mapped on the membrane given by a
        binary mask with coordinates in pixels or an array of coordinates in
        given units.
        Then, calculates a density measure of ribosomes at each vertex or
        membrane voxel: D = sum {over all reachable ribosomes} (1 / (d + 1)).
        Adds the density as vertex PropertyMap to the graph. Returns an array
        with the same shape as the underlying segmentation with the densities
        plus 1, in order to distinguish membrane voxels with 0 density from the
        background.

        Args:
            size (tuple): size in voxels (X, Y, Z) of the original segmentation
            scale (tuple): pixel size (X, Y, Z) in given units of the original
                segmentation
            mask (numpy.ndarray, optional): a binary mask of the ribosome
                centers as 3D array where indices are coordinates in pixels
                (default None)
            target_coordinates (numpy.ndarray, optional): the ribosome centers
                coordinates in given units as 2D array in format
                [[x1, y1, z1], [x2, y2, z2], ...] (default None)
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            a 3D numpy ndarray with the densities + 1

        Note:
            One of the two parameters, mask or target_coordinates, has to be
            given.
        """
        from . import ribosome_density as rd
        # If a mask is given, find the set of voxels of ribosome centers mapped
        # on the membrane, 'target_voxels', and rescale them to units,
        # 'target_coordinates':
        if mask is not None:
            if mask.shape != size:
                raise pexceptions.PySegInputError(
                    expr='calculate_density (SegmentationGraph)',
                    msg=("Size of the input 'mask' have to be equal to those "
                         "set during the generation of the graph."))
            # output as a list of tuples [(x1,y1,z1), (x2,y2,z2), ...] in pixels
            target_voxels = rd.get_foreground_voxels_from_mask(mask)
            # for rescaling have to convert to an ndarray
            target_ndarray_voxels = rd.tupel_list_to_ndarray_voxels(
                target_voxels)
            # rescale to units, output an ndarray [[x1,y1,z1], [x2,y2,z2], ...]
            target_ndarray_coordinates = (target_ndarray_voxels *
                                          np.asarray(scale))
            # convert to a list of tuples, which are in units now
            target_coordinates = rd.ndarray_voxels_to_tupel_list(
                target_ndarray_coordinates)
        # If target_coordinates are given (in units), convert them from a numpy
        # ndarray to a list of tuples:
        elif target_coordinates is not None:
            target_coordinates = rd.ndarray_voxels_to_tupel_list(
                target_coordinates)
        # Exit if the target_voxels list is empty:
        if len(target_coordinates) == 0:
            raise pexceptions.PySegInputError(
                expr='calculate_density (SegmentationGraph)',
                msg="No target voxels were found! Check your input ('mask' or "
                    "'target_coordinates').")
        print('{} target voxels'.format(len(target_coordinates)))
        if verbose:
            print(target_coordinates)

        # Pre-filter the target coordinates to those existing in the graph
        # (should already all be in the graph, but just in case):
        target_coordinates_in_graph = []
        for target_xyz in target_coordinates:
            if target_xyz in self.coordinates_to_vertex_index:
                target_coordinates_in_graph.append(target_xyz)
            else:
                raise pexceptions.PySegInputWarning(
                    expr='calculate_density (SegmentationGraph)',
                    msg=('Target ({}, {}, {}) not inside the membrane!'.format(
                        target_xyz[0], target_xyz[1], target_xyz[2])))

        print('{} target coordinates in graph'.format(len(
            target_coordinates_in_graph)))
        if verbose:
            print(target_coordinates_in_graph)

        # Get all indices of the target coordinates:
        target_vertices_indices = []
        for target_xyz in target_coordinates_in_graph:
            v_target_index = self.coordinates_to_vertex_index[target_xyz]
            target_vertices_indices.append(v_target_index)

        # Density calculation
        # Add a new vertex property to the graph, density:
        self.graph.vp.density = self.graph.new_vertex_property("float")
        # Dictionary mapping voxel coordinates (for the volume returned later)
        # to a list of density values falling within that voxel:
        voxel_to_densities = {}

        # For each vertex in the graph:
        for v_membrane in self.graph.vertices():
            # Get its coordinates:
            membrane_xyz = self.graph.vp.xyz[v_membrane]
            if verbose:
                print('Membrane vertex ({}, {}, {})'.format(
                    membrane_xyz[0], membrane_xyz[1], membrane_xyz[2]))
            # Get a distance map with all pairs of distances between current
            # graph vertex (membrane_xyz) and target vertices (ribosome
            # coordinates):
            dist_map = shortest_distance(self.graph, source=v_membrane,
                                         target=target_vertices_indices,
                                         weights=self.graph.ep.distance)

            # Iterate over all shortest distances from the membrane vertex to
            # the target vertices, while calculating the density:
            # Initializing: membrane coordinates with no reachable ribosomes
            # will have a value of 0, those with reachable ribosomes > 0.
            density = 0
            # If there is only one target voxel, dist_map is a single value -
            # wrap it into a list.
            if len(target_coordinates_in_graph) == 1:
                dist_map = [dist_map]
            for d in dist_map:
                if verbose:
                    print('\tTarget vertex ...')
                # if unreachable, the maximum float64 is stored
                if d == np.finfo(np.float64).max:
                    if verbose:
                        print('\t\tunreachable')
                else:
                    if verbose:
                        print('\t\td = {}'.format(d))
                    density += 1 / (d + 1)

            # Add the density of the membrane vertex as a property of the
            # current vertex in the graph:
            self.graph.vp.density[v_membrane] = density

            # Calculate the corresponding voxel of the vertex and add the
            # density to the list keyed by the voxel in the dictionary:
            # Scaling the coordinates back from units to voxels. (Without round
            # float coordinates are truncated to the next lowest integer.)
            voxel_x = int(round(membrane_xyz[0] / scale[0]))
            voxel_y = int(round(membrane_xyz[1] / scale[1]))
            voxel_z = int(round(membrane_xyz[2] / scale[2]))
            voxel = (voxel_x, voxel_y, voxel_z)
            if voxel in voxel_to_densities:
                voxel_to_densities[voxel].append(density)
            else:
                voxel_to_densities[voxel] = [density]

            if verbose:
                print('\tdensity = {}'.format(density))
            if (self.graph.vertex_index[v_membrane] + 1) % 1000 == 0:
                now = datetime.now()
                print('{} membrane vertices processed on: {}-{}-{} {}:{}:{}'
                      .format(self.graph.vertex_index[v_membrane] + 1,
                              now.year, now.month, now.day,
                              now.hour, now.minute, now.second))

        # Initialize an array scaled like the original segmentation, which will
        # hold in each membrane voxel the maximal density among the
        # corresponding vertex coordinates in the graph plus 1 and 0 in each
        # background (non-membrane) voxel:
        densities = np.zeros(size, dtype=np.float16)
        # The densities array membrane voxels are initialized with 1 in order to
        # distinguish membrane voxels from the background.
        for voxel in voxel_to_densities:
            densities[voxel[0], voxel[1], voxel[2]] = 1 + max(
                voxel_to_densities[voxel])
        if verbose:
            print('densities:\n{}'.format(densities))
        return densities

    def graph_to_points_and_lines_polys(self, vertices=True, edges=True,
                                        verbose=False):
        """
        Generates a VTK PolyData object from the graph with vertices as
        vertex-cells (containing 1 point) and edges as line-cells (containing 2
        points).

        Args:
            vertices (boolean, optional): if True (default) vertices are stored
                a VTK PolyData object as vertex-cells
            edges (boolean, optional): if True (default) edges are stored a VTK
                PolyData object as line-cells
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

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
        for prop_key in list(self.graph.vp.keys()):
            data_type = self.graph.vp[prop_key].value_type()
            if (data_type != 'string' and data_type != 'python::object' and
                    prop_key != 'xyz'):
                if verbose:
                    print('\nvertex property key: {}'.format(prop_key))
                    print('value type: {}'.format(data_type))
                if data_type[0:6] != 'vector':  # scalar
                    num_components = 1
                else:  # vector
                    num_components = len(
                        self.graph.vp[prop_key][self.graph.vertex(0)])
                array = TypesConverter().gt_to_vtk(data_type)
                array.SetName(prop_key)
                if verbose:
                    print('number of components: {}'.format(num_components))
                array.SetNumberOfComponents(num_components)
                vertex_arrays.append(array)
        # Edge property arrays
        for prop_key in list(self.graph.ep.keys()):
            data_type = self.graph.ep[prop_key].value_type()
            if data_type != 'string' and data_type != 'python::object':
                if verbose:
                    print('\nedge property key: {}'.format(prop_key))
                    print('value type: {}'.format(data_type))
                if data_type[0:6] != 'vector':  # scalar
                    num_components = 1
                else:  # vector (all edge properties so far are scalars)
                    # num_components = len(
                    #     self.graph.ep[prop_key][self.graph.edge(0, 1)])
                    num_components = 3
                    if verbose:
                        print('Sorry, not implemented yet, assuming a vector '
                              'with 3 components.')
                array = TypesConverter().gt_to_vtk(data_type)
                array.SetName(prop_key)
                if verbose:
                    print('number of components: {}'.format(num_components))
                array.SetNumberOfComponents(num_components)
                edge_arrays.append(array)
        if verbose:
            print('\nvertex arrays length: {}'.format(len(vertex_arrays)))
            print('edge arrays length: {}'.format(len(edge_arrays)))

        # Geometry
        lut = np.zeros(shape=self.graph.num_vertices(), dtype=np.int)
        for i, vd in enumerate(self.graph.vertices()):
            [x, y, z] = self.graph.vp.xyz[vd]
            points.InsertPoint(i, x, y, z)
            lut[self.graph.vertex_index[vd]] = i
        if verbose:
            print('number of points: {}'.format(points.GetNumberOfPoints()))

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
                    array.InsertNextTuple(self.get_vertex_prop_entry(
                        prop_key, vd, n_comp, data_type))
            if verbose:
                print('number of vertex cells: {}'.format(
                    verts.GetNumberOfCells()))
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
                    array.InsertNextTuple(self.get_edge_prop_entry(
                        prop_key, ed, n_comp, data_type))
            if verbose:
                print('number of line cells: {}'.format(
                    lines.GetNumberOfCells()))

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

    def get_vertex_prop_entry(self, prop_key, vertex_descriptor, n_comp,
                              data_type):
        """
        Gets a property value of a vertex for inserting into a VTK vtkDataArray
        object.

        This function is used by the methods graph_to_points_and_lines_polys and
        graph_to_triangle_poly (the latter of the derived classes PointGraph and
        TriangleGraph (in surface_graphs).

        Args:
            prop_key (str): name of the desired vertex property
            vertex_descriptor (graph_tool.Vertex): vertex descriptor of the
                current vertex
            n_comp (int): number of components of the array (length of the
                output tuple)
            data_type: numpy data type converted from a graph-tool property
                value type by TypesConverter().gt_to_numpy

        Returns:
            a tuple (with length like n_comp) with the property value of the
            vertex converted to a numpy data type
        """
        prop = list()
        if n_comp == 1:
            prop.append(data_type(self.graph.vp[prop_key][vertex_descriptor]))
        else:
            for i in range(n_comp):
                prop.append(data_type(
                            self.graph.vp[prop_key][vertex_descriptor][i]))
        return tuple(prop)

    def get_edge_prop_entry(self, prop_key, edge_descriptor, n_comp, data_type):
        """
        Gets a property value of an edge for inserting into a VTK vtkDataArray
        object.

        This private function is used by the method
        graph_to_points_and_lines_polys.

        Args:
            prop_key (str): name of the desired vertex property
            edge_descriptor (graph_tool.Edge): edge descriptor of the current
                edge
            n_comp (int): number of components of the array (length of the
                output tuple)
            data_type: numpy data type converted from a graph-tool property
                value type by TypesConverter().gt_to_numpy

        Returns:
            a tuple (with length like n_comp) with the property value of the
            edge converted to a numpy data type
        """
        prop = list()
        if n_comp == 1:
            prop.append(data_type(self.graph.ep[prop_key][edge_descriptor]))
        else:
            for i in range(n_comp):
                prop.append(data_type(
                            self.graph.ep[prop_key][edge_descriptor][i]))
        return tuple(prop)

    # * The following SegmentationGraph methods are needed for the normal vector
    # voting algorithm. *

    def calculate_average_edge_length(self, prop_e=None, value=1,
                                      verbose=False):
        """
        Calculates the average edge length in the graph.

        If a special edge property is specified, includes only the edges where
        this property equals the given value. If there are no edges in the
        graph, the given property does not exist or there are no edges with the
        given property equaling the given value, None is returned.

        Args:
            prop_e (str, optional): edge property, if specified only edges where
                this property equals the given value will be considered
            value (int, optional): value of the specified edge property an edge
                has to have in order to be considered (default 1)
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            the average edge length in the graph (float) or None
        """
        total_edge_length = 0
        average_edge_length = None
        if prop_e is None:
            if verbose:
                print("Considering all edges:")
            if self.graph.num_edges() > 0:
                if verbose:
                    print("{} edges".format(self.graph.num_edges()))
                average_edge_length = np.mean(self.graph.ep.distance.a)
            else:
                print("There are no edges in the graph!")
        elif prop_e in self.graph.edge_properties:
            if verbose:
                print("Considering only edges with property {} equaling value "
                      "{}!".format(prop_e, value))
            num_special_edges = 0
            for ed in self.graph.edges():
                if self.graph.edge_properties[prop_e][ed] == value:
                    num_special_edges += 1
                    total_edge_length += self.graph.ep.distance[ed]
            if num_special_edges > 0:
                if verbose:
                    print("{} such edges".format(num_special_edges))
                average_edge_length = total_edge_length / num_special_edges
            else:
                print("There are no edges with the property {} equaling value "
                      "{}!".format(prop_e, value))
        if verbose:
            print("Average length: {}".format(average_edge_length))
        return average_edge_length

    def find_geodesic_neighbors(self, v, g_max, full_dist_map=None,
                                only_surface=False, verbose=False):
        """
        Finds geodesic neighbor vertices of a given vertex v in the graph that
        are within a given maximal geodesic distance g_max from it.

        Also finds the corresponding geodesic distances. All edges are
        considered. The distances are calculated with a breadth-first search
        (BFS) or Dijkstra's algorithm.

        Args:
            v (graph_tool.Vertex): the source vertex
            g_max: maximal geodesic distance (in the units of the graph)
            full_dist_map (graph_tool.PropertyMap, optional): the full distance
                map for the whole graph; if None, a local distance map is
                calculated for each vertex (default)
            only_surface (boolean, optional): if True (default False), only
                neighbors classified as surface patch (class 1) are considered
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            a dictionary mapping a neighbor vertex index to the geodesic
            distance from vertex v
        """
        if full_dist_map is not None:
            dist_v = full_dist_map[v].get_array()
        else:
            dist_v = shortest_distance(self.graph, source=v, target=None,
                                       weights=self.graph.ep.distance,
                                       max_dist=g_max)
            dist_v = dist_v.get_array()
        # numpy array of distances from v to all vertices, in vertex index order

        vertex = self.graph.vertex
        orientation_class = self.graph.vp.orientation_class
        neighbor_id_to_dist = dict()

        idxs = np.where(dist_v <= g_max)[0]  # others are INF
        for idx in idxs:
            dist = dist_v[idx]
            if dist != 0:  # ignore the source vertex itself
                v_i = vertex(idx)
                if (not only_surface) or orientation_class[v_i] == 1:
                    neighbor_id_to_dist[idx] = dist

        if verbose:
            print("{} neighbors".format(len(neighbor_id_to_dist)))
        return neighbor_id_to_dist

    def find_geodesic_neighbors_exact(
            self, o, g_max, only_surface=False, verbose=False, debug=False):
        """
        Finds geodesic neighbor vertices of the origin vertex o in the graph
        that are within a given maximal geodesic distance g_max from it.

        Also finds the corresponding geodesic distances. All edges and faces are
        considered. The distances are calculated with Sun's and Abidi's
        algorithm, a simplification of Kimmels' and Sethian's fast marching
        algorithm.

        Args:
            o (graph_tool.Vertex): the source vertex
            g_max: maximal geodesic distance (in the units of the graph)
            only_surface (boolean, optional): if True (default False), only
                neighbors classified as surface patch (class 1) are considered
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out
            debug (boolean, optional): if True (default False), some more extra
                information will be printed out

        Returns:
            a dictionary mapping a neighbor vertex index to the geodesic
            distance from vertex o
        """
        # Shortcuts
        xyz = self.graph.vp.xyz
        vertex = self.graph.vertex
        orientation_class = self.graph.vp.orientation_class
        distance_between_voxels = self.distance_between_voxels
        calculate_geodesic_distance = self._calculate_geodesic_distance
        insert_geo_dist_vertex_id = self._insert_geo_dist_vertex_id
        # Initialization
        geo_dist_heap = []  # heap has the smallest geodesic distance first
        # dictionaries to keep track which geodesic distance belongs to which
        # vertex or vertices and vice versa
        geo_dist_to_vertex_ids = {}
        vertex_id_to_geo_dist = {}
        neighbor_id_to_dist = {}  # output dictionary
        # Tag the center point (o) as Alive:
        self.graph.vp.tag = self.graph.new_vertex_property("string")
        tag = self.graph.vp.tag  # shortcut
        tag[o] = "Alive"
        if debug:
            print("Vertex o={}: Alive".format(int(o)))
        vertex_id_to_geo_dist[int(o)] = 0  # need it for geo. dist. calculation
        xyz_o = tuple(xyz[o])
        for n in o.all_neighbours():
            # Tag all neighboring points of the center point (n) as Close
            tag[n] = "Close"
            # Geodesic distance in this case = Euclidean between o and n
            xyz_n = tuple(xyz[n])
            on = distance_between_voxels(xyz_o, xyz_n)
            if debug:
                print("Vertex n={}: Close with distance {}".format(int(n), on))
            heappush(geo_dist_heap, on)
            insert_geo_dist_vertex_id(geo_dist_to_vertex_ids, on, int(n))
            vertex_id_to_geo_dist[int(n)] = on

        # Repeat while the smallest distance is <= g_max
        while len(geo_dist_heap) >= 1 and geo_dist_heap[0] <= g_max:
            if debug:
                print("\n{} distances in heap, first={}".format(
                    len(geo_dist_heap), geo_dist_heap[0]))
            # 1. Change the tag of the point in Close with the smallest
            # geodesic distance (a) from Close to Alive
            smallest_geo_dist = heappop(geo_dist_heap)
            closest_vertices_ids = geo_dist_to_vertex_ids[smallest_geo_dist]
            a = vertex(closest_vertices_ids[0])
            if len(closest_vertices_ids) > 1:  # move the first one (a) to the
                # back, so it's not taken again next time
                closest_vertices_ids.pop(0)
                closest_vertices_ids.append(int(a))
            tag[a] = "Alive"
            # only proceed if a is a surface patch:
            if only_surface and orientation_class[a] != 1:
                continue
            neighbor_id_to_dist[int(a)] = smallest_geo_dist  # add a to output
            if debug:
                print("Vertex a={}: Alive".format(int(a)))
            neighbors_a = set(a.all_neighbours())  # actually don't have
            # duplicates, but like this can use fast sets' intersection method
            for c in neighbors_a:
                # 2. Tag all neighboring points (c) of this point as Close,
                # but skip those which are Alive already
                if tag[c] == "Alive":
                    if debug:
                        print("Skipping Alive neighbor {}".format(int(c)))
                    continue
                tag[c] = "Close"
                if debug:
                    print("Vertex c={}: Close".format(int(c)))
                # 3. Recompute the geodesic distance of these neighboring
                # points, using only values of points that are Alive, and renew
                # it only if the recomputed result is smaller
                # Find Alive point b, belonging to the same triangle as a and c:
                # iterate over an intersection of the neighbors of a and c
                neighbors_c = set(c.all_neighbours())
                common_neighbors_a_c = neighbors_a.intersection(neighbors_c)
                for b in common_neighbors_a_c:
                    # check if b is tagged Alive
                    if tag[b] == "Alive":
                        if debug:
                            print("\tUsing vertex b={}".format(int(b)))
                        new_geo_dist_c = calculate_geodesic_distance(
                            a, b, xyz[c].a, vertex_id_to_geo_dist,
                            verbose=verbose)
                        if int(c) not in vertex_id_to_geo_dist:  # add c
                            if debug:
                                print("\tadding new distance {}".format(
                                    new_geo_dist_c))
                            vertex_id_to_geo_dist[int(c)] = new_geo_dist_c
                            heappush(geo_dist_heap, new_geo_dist_c)
                            insert_geo_dist_vertex_id(
                                geo_dist_to_vertex_ids, new_geo_dist_c, int(c))
                        else:
                            old_geo_dist_c = vertex_id_to_geo_dist[int(c)]
                            if new_geo_dist_c < old_geo_dist_c:  # update c
                                if debug:
                                    print("\tupdating distance {} to {}".format(
                                        old_geo_dist_c, new_geo_dist_c))
                                vertex_id_to_geo_dist[int(c)] = new_geo_dist_c
                                if old_geo_dist_c in geo_dist_heap:
                                    # check because it is sometimes not there
                                    geo_dist_heap.remove(old_geo_dist_c)
                                heappush(geo_dist_heap, new_geo_dist_c)
                                old_geo_dist_vertex_ids = geo_dist_to_vertex_ids[
                                    old_geo_dist_c]
                                if len(old_geo_dist_vertex_ids) == 1:
                                    del geo_dist_to_vertex_ids[old_geo_dist_c]
                                else:
                                    old_geo_dist_vertex_ids.remove(int(c))
                                insert_geo_dist_vertex_id(
                                    geo_dist_to_vertex_ids, new_geo_dist_c,
                                    int(c))
                            elif debug:
                                print("\tkeeping the old distance={}, because "
                                      "it's <= the new={}".format(
                                        old_geo_dist_c, new_geo_dist_c))
                        # if debug:
                        #     print(geo_dist_heap)
                        #     print(geo_dist_to_vertex_ids)
                        #     print(vertex_id_to_geo_dist)
                        #     print(neighbor_id_to_dist)
                        break  # one Alive b is expected, stop iteration
                else:
                    if debug:
                        print("\tNo common neighbors of a and c are Alive")

        del self.graph.vertex_properties["tag"]
        if debug:
            print("Vertex o={} has {} geodesic neighbors".format(
                int(o), len(neighbor_id_to_dist)))
        if verbose:
            print("{} neighbors".format(len(neighbor_id_to_dist)))
        return neighbor_id_to_dist

    def _calculate_geodesic_distance(
            self, a, b, xyz_c, vertex_id_to_geo_dist, verbose=False):
        geo_dist_a = vertex_id_to_geo_dist[int(a)]
        geo_dist_b = vertex_id_to_geo_dist[int(b)]
        xyz_a = self.graph.vp.xyz[a].a
        xyz_b = self.graph.vp.xyz[b].a
        ab = euclidean_distance(xyz_a, xyz_b)
        ac = euclidean_distance(xyz_a, xyz_c)
        bc = euclidean_distance(xyz_b, xyz_c)
        # maybe faster to use linalg.euclidean_distance directly on np.ndarrays
        alpha = nice_acos((ab ** 2 + ac ** 2 - bc ** 2) / (2 * ab * ac))
        beta = nice_acos((ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc))
        if alpha < (math.pi / 2) and beta < (math.pi / 2):
            if verbose:
                print("\ttriangle abc is acute")
            theta = nice_acos((geo_dist_a ** 2 + ab ** 2 - geo_dist_b ** 2) /
                              (2 * ab * geo_dist_a))
            geo_dist_c = math.sqrt(ac ** 2 + geo_dist_a ** 2 -
                                   2 * ac * geo_dist_a * math.cos(alpha + theta))
        else:
            if verbose:
                print("\ttriangle abc is obtuse")
            geo_dist_c = min(geo_dist_a + ac, geo_dist_b + bc)
        return geo_dist_c

    @staticmethod
    def _insert_geo_dist_vertex_id(geo_dist_to_vertices, geo_dist, vertex_ind):
        if geo_dist in geo_dist_to_vertices:
            geo_dist_to_vertices[geo_dist].append(vertex_ind)
        else:
            geo_dist_to_vertices[geo_dist] = [vertex_ind]

    def get_vertex_property_array(self, property_name):
        """
        Gets a numpy array with all values of a vertex property of the graph,
        printing out the number of values, the minimal and the maximal value.

        Args:
            property_name (str): vertex property name

        Returns:
            an array (numpy.ndarray) with all values of the vertex property
        """
        if (isinstance(property_name, str) and
                property_name in self.graph.vertex_properties):
            values = np.array(
                self.graph.vertex_properties[property_name].get_array())
            print('{} "{}" values'.format(len(values), property_name))
            print('min = {}, max = {}, mean = {}'.format(
                min(values), max(values), np.mean(values)))
            return values
        else:
            raise pexceptions.PySegInputError(
                expr='get_vertex_property_array (SegmentationGraph)',
                msg=('The input "{}" is not a str object or is not found in '
                     'vertex properties of the graph.'.format(property_name)))
