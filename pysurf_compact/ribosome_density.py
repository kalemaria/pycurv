import math
import numpy as np
from scipy import spatial
from datetime import datetime
from graph_tool import Graph

import graphs
import pysurf_io as io
import pexceptions

"""
Set of functions and a class (VoxelGraph) for calculating ribosome density on
membranes.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def read_in_mask(mask_file, verbose=False):
    """
    A wrapper for reading in a membrane segmentation or ribosome centers mask
    (binary tomographic data).

    Args:
        mask_file (str): a mask file in EM, MRC or VTI format
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        the read in mask (numpy.ndarray)
    """
    print '\nReading in the mask %s' % mask_file
    mask = io.load_tomo(mask_file)
    if verbose:
        print 'Shape and data type:'
        print mask.shape
        print mask.dtype
    return mask


def get_foreground_voxels_from_mask(mask):
    """
    Gets foreground (non-zero) voxel coordinates from a mask (binary tomographic
    data).

    Args:
        mask (numpy ndarray): a 3D array holding the mask, which should be
            binary

    Returns:
        a list of foreground (nonzero) voxel coordinates as tuples in form
        (x, y, z)
    """
    voxels = []
    # check that the mask is a 3D numpy array:
    if isinstance(mask, np.ndarray) and (len(mask.shape) == 3):
        indices = mask.nonzero()
        voxels_num = indices[0].size
        for i in xrange(voxels_num):
            voxel_i = (indices[0][i], indices[1][i], indices[2][i])
            voxels.append(voxel_i)
    else:
        error_msg = 'A 3D numpy ndarray object required as input.'
        raise pexceptions.PySegInputError(
            expr='get_foreground_voxels_from_mask', msg=error_msg
        )
    return voxels


def rescale_mask(in_mask_file, out_mask_file, scaling_factor, out_shape):
    """
    Reads in a mask (binary tomographic data) from a file, rescales the mask and
    writes it into a file.

    How rescaling is done: The output (rescaled) mask with a given shape is
    initialized with zeros. Foreground (non-zero) voxel coordinates from the
    original mask are multiplied by the scaling factor, and ones (1) are put at
    the resulting coordinates inside the rescaled mask.

    Args:
        in_mask_file (str): an input mask file
        out_mask_file (str): an output (rescaled) mask file
        scaling_factor (int): a scaling factor by which the foreground voxel
            coordinates in the input file are multiplied
        out_shape (tuple): shape of the output (rescaled) mask
            (size_x, size_y, size_z)

    Returns:
        None
    """
    in_mask = read_in_mask(in_mask_file)
    in_target_voxels = get_foreground_voxels_from_mask(in_mask)

    out_mask = np.zeros(out_shape, dtype=np.uint8)

    for in_voxel in in_target_voxels:
        in_x = in_voxel[0]
        in_y = in_voxel[1]
        in_z = in_voxel[2]
        out_x = math.floor(in_x * scaling_factor)
        out_y = math.floor(in_y * scaling_factor)
        out_z = math.floor(in_z * scaling_factor)
        out_mask[out_x, out_y, out_z] = 1

    io.save_numpy(out_mask, out_mask_file)


def ndarray_voxels_to_tupel_list(voxels_ndarray):
    """
    Turns voxel coordinates from a 2D numpy ndarray in format
    [[x1, y1, z1], [x2, y2, z2], ...] into a list of tuples in format
    [(x1, y1, z1), (x2, y2, z2), ...].

    Args:
        voxels_ndarray (numpy.ndarray): a 2D array containing voxel coordinates
            in format [[x1, y1, z1], [x2, y2, z2], ...]

    Returns:
        a list of tuples containing the voxel coordinates in format
        [(x1, y1, z1), (x2, y2, z2), ...]
    """
    tupel_list = []
    # check that voxels_ndarray is a 2D numpy array with 3 columns:
    if (isinstance(voxels_ndarray, np.ndarray) and
            (len(voxels_ndarray.shape) == 2) and
            (voxels_ndarray.shape[1] == 3)):
        voxels_list = voxels_ndarray.tolist()
        for voxel in voxels_list:
            x = voxel[0]
            y = voxel[1]
            z = voxel[2]
            tupel_list.append((x, y, z))
    else:
        error_msg = 'A 2D numpy ndarray with 3 columns required as input.'
        raise pexceptions.PySegInputError(expr='ndarray_voxels_to_tupel_list',
                                          msg=error_msg)
    return tupel_list


# From a list of foreground voxels as tupels in form (x, y, z), returns a numpy
# ndarray in format [[x1,y1,z1], [x2,y2,z2], ...].
def tupel_list_to_ndarray_voxels(tupel_list):
    """
    Turns voxel coordinates from a list of tuples in format
    [(x1, y1, z1), (x2, y2, z2), ...] into a 2D numpy ndarray in format
    [[x1, y1, z1], [x2, y2, z2], ...].

    Args:
        tupel_list: a list of tuples containing voxel coordinates in format
            [(x1, y1, z1), (x2, y2, z2), ...]

    Returns:
        a 2D array containing the voxel coordinates in format
        [[x1, y1, z1], [x2, y2, z2], ...] (numpy.ndarray)
    """
    voxels_list = []
    for tupel in tupel_list:
        x = tupel[0]
        y = tupel[1]
        z = tupel[2]
        voxels_list.append([x, y, z])
    voxels_ndarray = np.array(voxels_list)
    # check that voxels_ndarray is a 2D array with 3 columns:
    dims = voxels_ndarray.shape
    assert (len(dims) == 2)
    assert (dims[1] == 3)
    return voxels_ndarray


def get_target_voxels_in_membrane_mask(ribo_mask, mem_mask, verbose=False):
    """
    Gets target voxels from a ribosome mask and pre-filters them to those that
    are inside a membrane mask (value 1).

    Prints out the target voxel numbers before and after filtering and warns of
    the voxels that are not inside the membrane mask.

    Args:
        ribo_mask (numpy.ndarray): a ribosome mask
        mem_mask (numpy.ndarray): a membrane mask
        verbose (boolean, optional): it True (default False), additionally
            prints out the target voxels before and after filtering

    Returns:
         a list of the target voxels that are inside the membrane mask as tuples
         in form (x, y, z)
    """
    if (isinstance(ribo_mask, np.ndarray) and (len(ribo_mask.shape) == 3) and
            isinstance(mem_mask, np.ndarray) and (len(mem_mask.shape) == 3)):
        if ribo_mask.shape != mem_mask.shape:
            error_msg = ('Both input 3D numpy ndarray objects have to have the '
                         'same scales.')
            raise pexceptions.PySegInputError(
                expr='get_target_voxels_in_membrane_mask', msg=error_msg
            )
        # Find the set of voxels of ribosome centers mapped on the membrane,
        # called 'target voxels' from now on:
        target_voxels = get_foreground_voxels_from_mask(ribo_mask)
        print '%s target voxels' % len(target_voxels)
        if verbose:
            print target_voxels

        target_voxels_in_membrane_mask = []
        for target_voxel in target_voxels:
            if mem_mask[target_voxel[0], target_voxel[1], target_voxel[2]] == 1:
                target_voxels_in_membrane_mask.append(target_voxel)
            else:
                error_msg = ('Target voxel (%s, %s, %s) not inside the '
                             'membrane!' % (target_voxel[0], target_voxel[1],
                                            target_voxel[2]))
                raise pexceptions.PySegInputWarning(
                    expr='get_target_voxels_in_membrane_mask', msg=error_msg
                )
        print ('%s target voxels in membrane'
               % len(target_voxels_in_membrane_mask))
        if verbose:
            print target_voxels_in_membrane_mask
        return target_voxels_in_membrane_mask
    else:
        error_msg = ('3D numpy ndarray objects required as first and second '
                     'input')
        raise pexceptions.PySegInputError(
            expr='get_target_voxels_in_membrane_mask', msg=error_msg
        )


def particles_xyz_to_np_array(motl_em_file, scaling_factor=1):
    """
    Extracts coordinates of all particles from a motive list EM file and returns
    them in numpy array format.

    Optionally, scales the coordinates by multiplying with a given scaling
    factor.

    Args:
        motl_em_file (str): TOM motive list EM file holding the particle
            coordinates in rows 8-10
        scaling_factor (int, optional): scaling factor by which the coordinates
            are multiplied; if 1 (default), no scaling is performed

    Returns:
        a 2D array containing the particle coordinates in format
        [[x1, y1, z1], [x2, y2, z2], ...] (numpy.ndarray)
    """
    motl = io.load_tomo(motl_em_file)
    particles_xyz = []
    for col in xrange(motl.shape[1]):
        x = scaling_factor * motl[7, col, 0]
        y = scaling_factor * motl[8, col, 0]
        z = scaling_factor * motl[9, col, 0]
        particles_xyz.append([x, y, z])
    particles_xyz = np.array(particles_xyz)
    return particles_xyz


def nearest_vertex_for_particles(vertices_xyz, particles_xyz, radius):
    """
    Finds for each particle coordinates the nearest membrane graph vertices
    coordinates (both sets given by 2D numpy arrays) within a given radius.

    If no vertex exists within the radius, [-1, -1, -1] is returned at the
    respective index. Uses KD trees for a fast search.

    Args:
        vertices_xyz (numpy.ndarray): membrane graph vertices coordinates in
            format [[x1, y1, z1], [x2, y2, z2], ...]
        particles_xyz (numpy.ndarray): particle coordinates in same format as
            the vertices coordinates
        radius (int or float): distance upper bound for searching vertices from
            each particle coordinate

    Returns:
        a 2D array in same format as the inputs with the nearest vertices
        coordinates (numpy.ndarray)

    Note:
        All input parameters have to be in the same scale (either in pixels or
        in nanometers).
    """
    # Construct the KD tree from the vertices coordinates:
    tree = spatial.KDTree(vertices_xyz)
    # Search in the tree for the nearest vertex within the radius to the
    # particles (defaults: k=1 number of nearest neighbors, eps=0 precise
    # distance, p=2 Euclidean distance):
    (distances, nearest_treedata_positions) = tree.query(
        particles_xyz, distance_upper_bound=radius
    )
    nearest_vertices_xyz = []
    for i in nearest_treedata_positions:
        if i < len(vertices_xyz):
            nearest_vertices_xyz.append(tree.data[i])
        else:
            nearest_vertices_xyz.append([-1, -1, -1])
    nearest_vertices_xyz = np.array(nearest_vertices_xyz)
    return nearest_vertices_xyz


class VoxelGraph(graphs.SegmentationGraph):
    """
    Class defining the VoxelGraph object and its methods.

    The constructor requires the following parameters of the underlying
    segmentation that will be used to build the graph.

    Args:
        scale_factor_to_nm (float): pixel size in nanometers for scaling the
            graph
        scale_x (int): x axis length in pixels of the segmentation
        scale_y (int): y axis length in pixels of the segmentation
        scale_z (int): z axis length in pixels of the segmentation
    """

    def build_graph_from_np_ndarray(self, mask, verbose=False):
        """
        Builds a graph from a binary mask of a membrane segmentation, including
        only voxels with value 1 (foreground voxels).

        Each foreground voxel, its foreground neighbor voxels and edges with
        euclidean distances between the voxel and its neighbor voxels (all
        scaled in nm) are added to the graph.

        Args:
            mask (numpy.ndarray): a binary 3D mask
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        if isinstance(mask, np.ndarray) and (len(mask.shape) == 3):
            if mask.shape != (self.scale_x, self.scale_y, self.scale_z):
                error_msg = ('Scales of the input mask have to be equal to '
                             'those set during the generation of the VoxelGraph'
                             'object.')
                raise pexceptions.PySegInputError(
                    expr='build_graph_from_np_ndarray (VoxelGraph)',
                    msg=error_msg
                )
            # Find the set of the membrane voxels, which become the vertices of
            # the graph:
            membrane_voxels = get_foreground_voxels_from_mask(mask)
            print '%s membrane voxels' % len(membrane_voxels)
            if verbose:
                print membrane_voxels
            self.__expand_voxels(mask, membrane_voxels, verbose)
        else:
            error_msg = 'A 3D numpy ndarray object required as first input.'
            raise pexceptions.PySegInputError(
                expr='build_graph_from_np_ndarray (VoxelGraph)', msg=error_msg
            )

    def __expand_voxels(self, mask, remaining_mem_voxels, verbose=False):
        """
        An iterative function used for building the membrane graph of a
        VoxelGraph object.

        This private method should only be called by the method
        build_graph_from_np_ndarray! Expands each foreground voxel, adding it,
        its foreground neighbor voxels and edges with euclidean distances
        between the voxel and its neighbor voxels (all scaled in nm) to the
        graph.

        Args:
            mask (numpy.ndarray): a binary 3D mask
            remaining_mem_voxels: a list of remaining membrane voxel coordinates
                as tuples in form (x, y, z)
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        while len(remaining_mem_voxels) > 0:
            try:
                if verbose:
                    print ('%s remaining membrane voxels'
                           % len(remaining_mem_voxels))
                elif len(remaining_mem_voxels) % 1000 == 0:
                    now = datetime.now()
                    print ('%s remaining membrane voxels on: %s-%s-%s %s:%s:%s'
                           % (len(remaining_mem_voxels), now.year, now.month,
                              now.day, now.hour, now.minute, now.second))

                # get and remove the last voxel on the list of remaining
                # membrane voxels to expand it next
                voxel_to_expand = remaining_mem_voxels.pop()
                if verbose:
                    print ('\nCurrent voxel to expand: (%s, %s, %s)'
                           % (voxel_to_expand[0], voxel_to_expand[1],
                              voxel_to_expand[2]))

                scaled_voxel_to_expand = (
                    voxel_to_expand[0] * self.scale_factor_to_nm,
                    voxel_to_expand[1] * self.scale_factor_to_nm,
                    voxel_to_expand[2] * self.scale_factor_to_nm
                )
                # If the scaled voxel to be expanded has been already added to
                # the graph, get its vertex descriptor:
                if scaled_voxel_to_expand in self.coordinates_to_vertex_index:
                    v_expanded = self.graph.vertex(
                        self.coordinates_to_vertex_index[scaled_voxel_to_expand]
                    )
                # Otherwise, add the scaled voxel to be expanded as vertex to
                # the graph:
                else:
                    v_expanded = self.graph.add_vertex()
                    self.graph.vp.xyz[v_expanded] = [
                        scaled_voxel_to_expand[0], scaled_voxel_to_expand[1],
                        scaled_voxel_to_expand[2]
                    ]
                    self.coordinates_to_vertex_index[scaled_voxel_to_expand] = \
                        self.graph.vertex_index[v_expanded]
                    if verbose:
                        print ('This voxel has been added to the graph as '
                               'vertex.')

                # Get the neighbor membrane voxels of the current voxel:
                neighbor_voxels = self.foreground_neighbors_of_voxel(
                    mask, voxel_to_expand
                )

                # If there is at least one foreground neighbor, do for each:
                if len(neighbor_voxels) > 0:
                    for neighbor_voxel in neighbor_voxels:

                        scaled_neighbor_voxel = (
                            neighbor_voxel[0] * self.scale_factor_to_nm,
                            neighbor_voxel[1] * self.scale_factor_to_nm,
                            neighbor_voxel[2] * self.scale_factor_to_nm
                        )
                        # If the scaled neighbor voxel has been already added to
                        # the graph, get its vertex descriptor:
                        if (scaled_neighbor_voxel in
                                self.coordinates_to_vertex_index):
                            v_neighbor = self.graph.vertex(
                                self.coordinates_to_vertex_index[
                                    scaled_neighbor_voxel
                                ]
                            )
                        # Otherwise, add the neighbor voxel as vertex to the
                        # graph:
                        else:
                            v_neighbor = self.graph.add_vertex()
                            self.graph.vp.xyz[v_neighbor] = [
                                scaled_neighbor_voxel[0],
                                scaled_neighbor_voxel[1],
                                scaled_neighbor_voxel[2]
                            ]
                            self.coordinates_to_vertex_index[
                                scaled_neighbor_voxel
                            ] = self.graph.vertex_index[v_neighbor]
                            if verbose:
                                print ('The neighbor voxel (%s, %s, %s) has '
                                       'been added to the graph as a vertex.'
                                       % (neighbor_voxel[0], neighbor_voxel[1],
                                          neighbor_voxel[2]))

                        # Add an edge with a distance between the expanded
                        # scaled vertex and the scaled neighbor vertex, if it
                        # does not exist yet:
                        if not (((scaled_voxel_to_expand, scaled_neighbor_voxel)
                                 in self.coordinates_pair_connected) or
                                ((scaled_neighbor_voxel, scaled_voxel_to_expand)
                                 in self.coordinates_pair_connected)):
                            # edge descriptor
                            ed = self.graph.add_edge(v_expanded, v_neighbor)
                            self.graph.ep.distance[
                                ed] = self.distance_between_voxels(
                                scaled_voxel_to_expand, scaled_neighbor_voxel)
                            self.coordinates_pair_connected[(
                                scaled_voxel_to_expand,
                                scaled_neighbor_voxel)] = True
                            if verbose:
                                print ('The neighbor voxels (%s, %s, %s) and '
                                       '(%s, %s, %s) have been connected by an '
                                       'edge with a distance of %s pixels.' %
                                       (voxel_to_expand[0], voxel_to_expand[1],
                                        voxel_to_expand[2], neighbor_voxel[0],
                                        neighbor_voxel[1], neighbor_voxel[2],
                                        self.graph.ep.distance[ed]))
            except Exception, exc:
                print "An exception happened: " + str(exc)
                print ('There were %s remaining membrane voxels.'
                       % len(remaining_mem_voxels))
        else:
            if verbose:
                print '0 remaining membrane voxels'

    @staticmethod
    def foreground_neighbors_of_voxel(mask, voxel):
        """
        Returns neighbor voxels with value 1 (foreground) of a given voxel
        inside a binary mask of a membrane segmentation.

        Args:
            mask (numpy.ndarray): a binary 3D mask
            voxel (tuple): voxel coordinates in the mask as a tuple of integers
                of length 3: (x, y, z)

        Returns:
            a list of tuples with neighbor voxels coordinates in format
            [(x1, y1, z1), (x2, y2, z2), ...]
        """
        neighbor_voxels = []
        if isinstance(mask, np.ndarray) and (len(mask.shape) == 3):
            if isinstance(voxel, tuple) and (len(voxel) == 3):
                x = voxel[0]
                y = voxel[1]
                z = voxel[2]
                x_size = mask.shape[0]
                y_size = mask.shape[1]
                z_size = mask.shape[2]
                # iterate over all possible (max 26) neighbor voxels
                # combinations
                for i in (x - 1, x, x + 1):
                    for j in (y - 1, y, y + 1):
                        for k in (z - 1, z, z + 1):
                            # do not add the voxel itself and voxels outside the
                            # border
                            if ((i, j, k) != (x, y, z) and
                                    i in xrange(0, x_size) and
                                    j in xrange(0, y_size) and
                                    k in xrange(0, z_size)):
                                # add only foreground voxels to the neighbors
                                # list
                                if mask[i, j, k] == 1:
                                    neighbor_voxels.append((i, j, k))
            else:
                error_msg = ('A tuple of integers of length 3 required as the'
                             'second input.')
                raise pexceptions.PySegInputError(
                    expr='foreground_neighbors_of_voxel (VoxelGraph)',
                    msg=error_msg
                )
        else:
            error_msg = 'A 3D numpy ndarray required as the first input.'
            raise pexceptions.PySegInputError(
                expr='foreground_neighbors_of_voxel (VoxelGraph)',
                msg=error_msg
            )
        return neighbor_voxels
