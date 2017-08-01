import math
import vtk
import numpy as np
from datetime import datetime
from graph_tool import Graph
from graph_tool.topology import shortest_distance

from pysurf_io import TypesConverter

# Class defining the abstract SegmentationGraph object.
class SegmentationGraph(object):
    # Initializes the object, specifying the parameters of the underlying segmentation that will be used to generate the graph: pixel size in nm and the scales.
    def __init__(self, scale_factor_to_nm, scale_x, scale_y, scale_z):
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
    # Calculates and returns the euclidean distance between two voxels (3D points).
    def distance_between_voxels(voxel1, voxel2):
        if isinstance(voxel1, tuple) and (len(voxel1) == 3) and isinstance(voxel2, tuple) and (len(voxel2) == 3):
            sum_of_squared_differences = 0
            for i in range(3):  # for each dimension
                sum_of_squared_differences += (voxel1[i] - voxel2[i]) ** 2
            return math.sqrt(sum_of_squared_differences)
        else:
            print 'Error: Wrong input data, each voxel has to be a tuple of integers of length 3.'
            exit(1)

    # Returns the coordinates of all vertices in numpy array format.
    def vertices_xyz_to_np_array(self):
        vertices_xyz = []
        for v in self.graph.vertices():
            v_xyz = self.graph.vp.xyz[v]  # [x,y,z]
            vertices_xyz.append(v_xyz)
        vertices_xyz = np.array(vertices_xyz)
        return vertices_xyz

    # Calculates shortest distances for each node in the graph (membrane_xyz) to each reachable voxel of ribosome center mapped on the membrane given by
    # a mask in pixels (3D numpy ndarray) or a numpy ndarray of target coordinates in nm (output of nearest_vertex_for_particles), and from the distances
    # a density measure of ribosomes at each membrane voxel. Returns a numpy ndarray (with the same shape as the underlying segmentation/mask) with the densities.
    def calculate_density(self, mask=None, target_coordinates=None, verbose=False):
        import ribosome_density as rd
        # If a mask is given, find the set of voxels of ribosome centers mapped on the membrane, 'target_voxels', and rescale them to nm, 'target_coordinates':
        if mask is not None:
            assert mask.shape[0] == self.scale_x
            assert mask.shape[1] == self.scale_y
            assert mask.shape[2] == self.scale_z
            target_voxels = rd.get_foreground_voxels_from_mask(mask)  # output as a list of tuples: [(x1,y1,z1), (x2,y2,z2), ...] in pixels
            target_ndarray_voxels = rd.tupel_list_to_ndarray_voxels(target_voxels)  # for rescaling have to convert to a ndarray
            target_ndarray_coordinates = target_ndarray_voxels * self.scale_factor_to_nm  # rescale to nm, output a ndarray: [[x1,y1,z1], [x2,y2,z2], ...]
            target_coordinates = rd.ndarray_voxels_to_tupel_list(target_ndarray_coordinates)  # convert to a list of tupels, which are in nm now
        # If target_coordinates are given (in nm), convert them from a numpy ndarray to a list of tuples:
        elif target_coordinates is not None:
            target_coordinates = rd.ndarray_voxels_to_tupel_list(target_coordinates)  # to convert numpy ndarray to a list of tuples: [(x1,y1,z1), (x2,y2,z2), ...]
        # Exit if the target_voxels list is empty:
        if len(target_coordinates) == 0:
            print 'Error: no target voxels were found!'
            exit(1)
        print '%s target voxels' % len(target_coordinates)
        if verbose:
            print target_coordinates

        # Pre-filter the target coordinates to those existing in the graph (should already all be in the graph, but just in case):
        target_coordinates_in_graph = []
        for target_xyz in target_coordinates:
            if target_xyz in self.coordinates_to_vertex_index:
                target_coordinates_in_graph.append(target_xyz)
            else:
                print 'Warning: Target (%s, %s, %s) not inside the membrane!' % (target_xyz[0], target_xyz[1], target_xyz[2])
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

    # Computes Geodesic Gaussian Filter on the graph
    # sig: sigma for the Gaussian function
    # prop_v: property key for weighting the vertices (e.g. "curvature")
    # prop_e: property key for weighting the edges ("distance")
    # energy: if True energy normalization is active, so the resulting values will be in the same unit scale as input.
    # Adapted from Antonio's ggf from pyseg/graph/gt.py.
    def ggf(self, sig, str_ggf, prop_v=None, prop_e=None, energy=True):

        # Initialization
        if prop_v is not None:
            prop_v_p = self.graph.vertex_properties[prop_v]
            field_v = prop_v_p.get_array()
        else:
            field_v = np.ones(shape=self.graph.num_vertices(), dtype=np.float)
        prop_e_p = None
        if prop_e is not None:
            prop_e_p = self.graph.edge_properties[prop_e]
            # field_e = prop_e_p.get_array()
            # prop_e_p.get_array()[:] = field_e  # Antonio did not know anymore what it is for, works also without it.
        prop_ggf = self.graph.new_vertex_property('float')
        s3 = 3. * sig
        c = (-1.) / (2. * sig * sig)

        # Filtering
        for s in self.graph.vertices():
            dist_map = shortest_distance(self.graph, source=s, weights=prop_e_p, max_dist=s3)
            ids = np.where(dist_map.get_array() < s3)[0]  # IDs of vertices within the 3 sigma neighborhood from the source vertex.
            # Computing energy preserving coefficients
            fields = np.zeros(shape=ids.shape[0], dtype=np.float)  # stores the respective vertex property values for the neighboring vertices
            coeffs = np.zeros(shape=ids.shape[0], dtype=np.float)  # stores the "gaussian distance" coefficients from the source vertex for the neighboring vertices
            hold_sum = 0  # "gaussian distance" coefficients sum over all neighborhood
            for i in range(ids.shape[0]):
                v = self.graph.vertex(ids[i])
                fields[i] = field_v[int(v)]
                dst = dist_map[v]  # distance of the current neighboring vertex from the source vertex
                hold = math.exp(c * dst * dst)  # "gaussian distance" coefficients
                coeffs[i] = hold
                hold_sum += hold
            if hold_sum > 0:  # Not tested for a negative sum.
                # Convolution
                if energy:
                    coeffs *= (1 / hold_sum)  # Every coefficient divided by the sum of coefficients for normalization, so the total sum will be 1.
                    prop_ggf[s] = np.sum(fields * coeffs)
                else:
                    prop_ggf[s] = np.sum(fields * coeffs)  # The same as hold_sum.
            else:
                print "Non-positive hold_sum = %s" % hold_sum  # -- No such cases occurred.

        # Storing the property
        self.graph.vertex_properties[str_ggf] = prop_ggf

    # Generates a VTK PolyData object with vertices as points and edges as lines.
    # vertices: if True (default) vertices are stored as points
    # edges: if True (default) edges are stored as lines
    # (Rewritten from Antonio's pyseg/graph/morse.py -> get_scheme_vtp - he used his own "graph-tool" (gt).)
    def graph_to_points_and_lines_polys(self, vertices=True, edges=True, verbose=False):
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
                    array.InsertNextTuple(self.__get_vertex_prop_entry(prop_key, vd, n_comp, data_type))
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
                    array.InsertNextTuple(self.__get_edge_prop_entry(prop_key, ed, n_comp, data_type))
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

    # Generates a VTK PolyData object with triangle-cells.
    def graph_to_triangle_poly(self, verbose=False):
        if self.graph.num_vertices() > 0:
            # Initialization
            poly_triangles = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            vertex_arrays = list()
            # Vertex property arrays
            for prop_key in self.graph.vp.keys():
                data_type = self.graph.vp[prop_key].value_type()
                if data_type != 'string' and data_type != 'python::object' and prop_key != 'points' and prop_key != 'xyz':
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
            if verbose:
                print '\nvertex arrays length: %s' % len(vertex_arrays)

            # Geometry
            lut = np.zeros(shape=(self.graph.num_vertices(), 3), dtype=np.int)  # lut[vertex_index, triangle_point_index*] = point_array_index**; *ALWAYS 0-2, **0-(NumPoints-1)
            i = 0  # next new point index
            points_dict = {}  # dictionary of points with a key (x, y, z) and the index in VTK points list as a value
            for vd in self.graph.vertices():
                for j, [x, y, z] in enumerate(self.graph.vp.points[vd]):  # enumerate over the 3 points of the triangle (vertex)
                    if (x, y, z) not in points_dict:  # add the new point everywhere & update the index
                        points.InsertPoint(i, x, y, z)
                        lut[self.graph.vertex_index[vd], j] = i
                        points_dict[(x, y, z)] = i
                        i += 1
                    else:  # reference the old point index only in the lut
                        lut[self.graph.vertex_index[vd], j] = points_dict[(x, y, z)]
            if verbose:
                print 'number of points: %s' % points.GetNumberOfPoints()

            # Topology
            # Triangles
            triangles = vtk.vtkCellArray()
            for vd in self.graph.vertices():  # vd = vertex descriptor
                # storing triangles of type Triangle:
                triangle = vtk.vtkTriangle()
                # The first parameter is the index of the triangle vertex which is ALWAYS 0-2.
                # The second parameter is the index into the point (geometry) array, so this can range from 0-(NumPoints-1)
                triangle.GetPointIds().SetId(0, lut[self.graph.vertex_index[vd], 0])
                triangle.GetPointIds().SetId(1, lut[self.graph.vertex_index[vd], 1])
                triangle.GetPointIds().SetId(2, lut[self.graph.vertex_index[vd], 2])
                triangles.InsertNextCell(triangle)
                for array in vertex_arrays:
                    prop_key = array.GetName()
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.graph.vp[prop_key].value_type()
                    data_type = TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.__get_vertex_prop_entry(prop_key, vd, n_comp, data_type))
            if verbose:
                print 'number of triangle cells: %s' % triangles.GetNumberOfCells()

            # vtkPolyData construction
            poly_triangles.SetPoints(points)
            poly_triangles.SetPolys(triangles)
            for array in vertex_arrays:
                poly_triangles.GetCellData().AddArray(array)

            return poly_triangles

        else:
            print "The graph is empty!"
            return None

    # Get the property value of a vertex
    def __get_vertex_prop_entry(self, prop_key, vertex_descriptor, n_comp, data_type):
        prop = list()
        if n_comp == 1:
            prop.append(data_type(self.graph.vp[prop_key][vertex_descriptor]))
        else:
            for i in range(n_comp):
                prop.append(data_type(self.graph.vp[prop_key][vertex_descriptor][i]))
        return tuple(prop)

    # Get the property value of an edge
    def __get_edge_prop_entry(self, prop_key, edge_descriptor, n_comp, data_type):
        prop = list()
        if n_comp == 1:
            prop.append(data_type(self.graph.ep[prop_key][edge_descriptor]))
        else:
            for i in range(n_comp):
                prop.append(data_type(self.graph.ep[prop_key][edge_descriptor][i]))
        return tuple(prop)

    # * The following Graph methods are needed for the vector voting algorithm. *
    # Calculates and returns the average edge length in the graph.
    # If a special edge property is specified as prop_e, include only the edges where this property equals the given value (default 1).
    # If there are no edges in the graph, such a property does not exist or there are no edges with the given property equaling the given value, None is returned.
    def calculate_average_edge_length(self, prop_e=None, value=1):
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

    # Finds geodesic neighbor vertices of a given vertex v in the graph that are within a given maximal geodesic distance g_max from it. Also finds the corresponding geodesic distances. (All edges are considered.)
    # Returns a dictionary mapping a neighbor vertex index to the geodesic distance from vertex v.
    def find_geodesic_neighbors(self, v, g_max, verbose=False):
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

    # Returns a numpy array of all values from the vertex properties of the graph.
    def get_vertex_property_array(self, property_name):
        if isinstance(property_name, str) and property_name in self.graph.vertex_properties:
            values = self.graph.vertex_properties[property_name].get_array()
            print '%s "%s" values' % (len(values), property_name)
            print 'min = %s, max = %s' % (min(values), max(values))
            return values
        else:
            print '"%s" is not a string or not found in vertex properties of the graph.'

    # Converts a vertex property array of the graph to a 3-D array of size like the underlying segmentation.
    # Initializes a 3-D matrix of size like the segmentation with zeros, transforms triangle coordinates in nm to voxels and puts the corresponding property value into the voxel.
    # If more than one triangles map to the same voxel, takes the maximal value. Logs such cases by writing out the voxel coordinates and the value list into a file.
    def property_to_volume(self, property_name, logfilename, verbose=False):
        # Dictionary mapping voxel coordinates (for the volume returned later) to a list of values falling within that voxel:
        voxel_to_values = {}

        # For each vertex in the graph:
        for v in self.graph.vertices():
            # Get its coordinates:
            v_xyz = self.graph.vp.xyz[v]

            # Get the property value assigned to the graph vertex:
            v_value = self.graph.vertex_properties[property_name][v]

            # Calculate the corresponding voxel of the vertex and add the value to the list keyed by the voxel in the dictionary:
            # Scaling the coordinates back from nm to voxels. (Without round float coordinates are truncated to the next lowest integer.)
            voxel_x = int(round(v_xyz[0] / self.scale_factor_to_nm))
            voxel_y = int(round(v_xyz[1] / self.scale_factor_to_nm))
            voxel_z = int(round(v_xyz[2] / self.scale_factor_to_nm))
            voxel = (voxel_x, voxel_y, voxel_z)
            if voxel in voxel_to_values:
                voxel_to_values[voxel].append(v_value)
            else:
                voxel_to_values[voxel] = [v_value]

            if verbose:
                print '\nMembrane vertex (%s, %s, %s)' % (v_xyz[0], v_xyz[1], v_xyz[2])
                print 'voxel (%s, %s, %s)' % (voxel[0], voxel[1], voxel[2])
                print '%s value = %s' % (property_name, v_value)

        print '\n%s voxels mapped from %s vertices' % (len(voxel_to_values), self.graph.num_vertices())

        # Initialize a 3-D array scaled like the original segmentation, which will hold in each voxel the maximal value among the corresponding vertex coordinates in the graph
        # and 0 in all other (background) voxels:
        volume = np.zeros((self.scale_x, self.scale_y, self.scale_z), dtype=np.float32)  # single precision float: sign bit, 8 bit exponent, 23 bits mantissa
        # Open and write the cases with multiple values into a log file:
        with open(logfilename, 'w') as f:
            for voxel in voxel_to_values:
                value_list = voxel_to_values[voxel]
                volume[voxel[0], voxel[1], voxel[2]] = max(value_list)
                if len(value_list) > 1:
                    line = '%s\t%s\t%s\t' % (voxel[0], voxel[1], voxel[2])
                    for value in value_list:
                        line += '%s\t' % value
                    line = line[0:-1] + '\n'
                    f.write(line)

        return volume
