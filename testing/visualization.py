import math
from graph_tool.topology import shortest_distance

from pysurf import TriangleGraph
from pysurf import pysurf_io as io

"""
Functions that can be used for visualizing geodesic neighborhood.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry)
"""

__author__ = 'kalemanov'


def calculate_g_max_from_radius_hit(radius_hit):
    """
    Calculates maximal geodesic neighborhood distance g_max from radius_hit:
    g_max is quoter of a circle with radius equal to radius_hit.

    Args:
        radius_hit (float): radius_hit parameter

    Returns:
        g_max
    """
    g_max = math.pi * radius_hit / 2
    print "radius_hit = {}".format(radius_hit)
    print "corresponding g_max = {}".format(g_max)
    return g_max


def calculate_distances_from_triangle(
        surface, scale_factor_to_nm, triangle_index, verbose=False):
    """
    Calculates shortest geodesic distances from a source triangle center to all
    other triangles on the surface.

    Args:
        surface (vtkPolyData): input surface
        scale_factor_to_nm (float): scale factor from voxels to nanometers
        triangle_index (int): index of the source triangle
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        vtkPolyData with "dist_to_v" property
    """
    # Build a triangle graph from the surface:
    tg = TriangleGraph()
    tg.build_graph_from_vtk_surface(surface, scale_factor_to_nm)
    if verbose:
        print ('The graph has {} vertices and {} edges'.format(
                tg.graph.num_vertices(), tg.graph.num_edges()))

    # Find the vertex from triangle index:
    v = tg.graph.vertex(triangle_index)

    # Calculate shortest distances from v to other vertices:
    tg.graph.vp.dist_to_v = tg.graph.new_vertex_property("float")
    shortest_distance(tg.graph, source=v, target=None,
                      weights=tg.graph.ep.distance,
                      dist_map=tg.graph.vp.dist_to_v)

    # Transform the resulting graph to a surface with triangles:
    surface_with_dist_to_v = tg.graph_to_triangle_poly()
    return surface_with_dist_to_v


if __name__ == "__main__":
    rh = 4
    v_id = 379  # 89
    res = 20  # 10
    fold = ("/fs/pool/pool-ruben/Maria/curvature/synthetic_surfaces/plane/"
            "res{}_noise10/files4plotting/with_borders/".format(res))
    in_surface_vtp = "{}plane_half_size{}.VCTV_rh{}.vtp".format(fold, res, rh)
    out_surface_vtp = "{}plane_half_size{}.VCTV_rh{}_dist_to_v{}.vtp".format(
        fold, res, rh, v_id)

    in_surface = io.load_poly(in_surface_vtp)
    out_surface = calculate_distances_from_triangle(
        in_surface, scale_factor_to_nm=1, triangle_index=v_id, verbose=True)
    io.save_vtp(out_surface, out_surface_vtp)

    calculate_g_max_from_radius_hit(4)
    calculate_g_max_from_radius_hit(8)
