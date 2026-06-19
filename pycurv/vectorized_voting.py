"""
Vectorized implementations of voting loop operations using NumPy.

This module provides batch-optimized versions of the normal and curvature
vote collection algorithms, using NumPy's einsum and vectorized operations
to replace Python loops with efficient array operations.

Author: Performance optimization module
"""

import numpy as np
from numpy import einsum
import math


def collect_normal_votes_vectorized(v, neighbor_indices, neighbor_distances,
                                    xyz_array, normal_array, area_array,
                                    a_max, sigma):
    """
    Vectorized collection of normal votes for a vertex.

    Implements the same algorithm as TriangleGraph.collect_normal_votes()
    but uses NumPy batch operations instead of Python loops.

    Args:
        v: numpy.ndarray (3,) - coordinates of the center vertex
        neighbor_indices: numpy.ndarray (N,) - indices of neighbor vertices
        neighbor_distances: numpy.ndarray (N,) - geodesic distances to neighbors
        xyz_array: numpy.ndarray (V, 3) - xyz coordinates of all vertices
        normal_array: numpy.ndarray (V, 3) - normal vectors of all vertices
        area_array: numpy.ndarray (V,) - areas of all triangles
        a_max: float - maximum triangle area for weight normalization
        sigma: float - sigma parameter for exponential weighting

    Returns:
        tuple of (num_neighbors, V_v):
            - num_neighbors: int - number of neighbors processed
            - V_v: numpy.ndarray (3, 3) - weighted covariance matrix sum
    """
    n_neighbors = len(neighbor_indices)
    if n_neighbors == 0:
        return 0, np.zeros((3, 3))

    # Gather neighbor data (N, 3) and (N,)
    c_i = xyz_array[neighbor_indices]       # (N, 3) - neighbor coordinates
    n = normal_array[neighbor_indices]      # (N, 3) - neighbor normals
    a_i = area_array[neighbor_indices]      # (N,) - neighbor areas
    g_i = neighbor_distances                # (N,) - geodesic distances

    # Compute vectors from v to each neighbor
    vc_i = c_i - v                          # (N, 3)

    # Compute lengths of vc_i vectors
    # einsum('ij,ij->i', vc_i, vc_i) computes dot product of each row with itself
    vc_i_len = np.sqrt(einsum('ij,ij->i', vc_i, vc_i))  # (N,)

    # Avoid division by zero
    vc_i_len = np.maximum(vc_i_len, 1e-10)

    # Normalize vc_i vectors
    vc_i_norm = vc_i / vc_i_len[:, np.newaxis]  # (N, 3)

    # Compute cos(theta_i) = -dot(n, vc_i) / |vc_i|
    # einsum('ij,ij->i', n, vc_i) computes dot product row-wise
    cos_theta_i = -einsum('ij,ij->i', n, vc_i) / vc_i_len  # (N,)

    # Compute the normal votes: n_i = n + 2 * cos_theta_i * vc_i_norm
    n_i = n + 2 * cos_theta_i[:, np.newaxis] * vc_i_norm  # (N, 3)

    # Compute weights: w_i = (a_i / a_max) * exp(-g_i / sigma)
    w_i = (a_i / a_max) * np.exp(-g_i / sigma)  # (N,)

    # Compute weighted sum of outer products: V_v = sum(w_i * outer(n_i, n_i))
    # einsum('n,ni,nj->ij', w_i, n_i, n_i) computes:
    #   sum over n of: w_i[n] * n_i[n, i] * n_i[n, j]
    V_v = einsum('n,ni,nj->ij', w_i, n_i, n_i)  # (3, 3)

    return n_neighbors, V_v


def collect_curvature_votes_vectorized(v, n_v, neighbor_indices, neighbor_distances,
                                       xyz_array, normal_array, area_array,
                                       a_max, sigma, page_curvature_formula=False):
    """
    Vectorized collection of curvature votes for a vertex.

    Implements the same algorithm as SurfaceGraph.collect_curvature_votes()
    but uses NumPy batch operations instead of Python loops.

    Args:
        v: numpy.ndarray (3,) - coordinates of the center vertex
        n_v: numpy.ndarray (3,) - estimated normal at the center vertex
        neighbor_indices: numpy.ndarray (N,) - indices of neighbor vertices
        neighbor_distances: numpy.ndarray (N,) - geodesic distances to neighbors
        xyz_array: numpy.ndarray (V, 3) - xyz coordinates of all vertices
        normal_array: numpy.ndarray (V, 3) - estimated normal vectors (n_v)
        area_array: numpy.ndarray (V,) - areas of all triangles (or None)
        a_max: float - maximum triangle area (0.0 to disable area weighting)
        sigma: float - sigma parameter for exponential weighting
        page_curvature_formula: bool - use Page et al. curvature formula

    Returns:
        numpy.ndarray (3, 3) - the B_v matrix, or None if no valid neighbors
    """
    n_neighbors = len(neighbor_indices)
    if n_neighbors == 0:
        return None

    pi = math.pi

    # Gather neighbor data
    v_i = xyz_array[neighbor_indices]       # (N, 3) - neighbor coordinates
    n_v_i = normal_array[neighbor_indices]  # (N, 3) - neighbor estimated normals
    g_i = neighbor_distances                # (N,) - geodesic distances

    # Compute weights
    w_i = np.exp(-g_i / sigma)  # (N,)
    if a_max > 0 and area_array is not None:
        a_i = area_array[neighbor_indices]  # (N,)
        w_i = w_i * (a_i / a_max)

    # Compute vectors from v to each neighbor
    vv_i = v_i - v  # (N, 3)

    # Compute tangent directions t_i (projection onto tangent plane)
    # t_i = vv_i - dot(n_v, vv_i) * n_v
    dot_n_vv = einsum('j,ij->i', n_v, vv_i)  # (N,) - dot(n_v, vv_i) for each neighbor
    t_i = vv_i - dot_n_vv[:, np.newaxis] * n_v  # (N, 3)

    # Normalize t_i
    t_i_len = np.sqrt(einsum('ij,ij->i', t_i, t_i))  # (N,)

    # Filter out neighbors where t_i_len is too small (nearly aligned with normal)
    valid_mask = t_i_len > 1e-10
    if not np.any(valid_mask):
        return None

    # Apply mask to all arrays
    t_i = t_i[valid_mask]
    t_i_len = t_i_len[valid_mask]
    vv_i = vv_i[valid_mask]
    n_v_i = n_v_i[valid_mask]
    g_i = g_i[valid_mask]
    w_i = w_i[valid_mask]

    # Normalize t_i
    t_i = t_i / t_i_len[:, np.newaxis]  # (N, 3)

    # Compute p_i = cross(n_v, t_i) - perpendicular to arc plane
    p_i = np.cross(n_v, t_i)  # (N, 3)

    # Compute n_v_i_p = n_v_i - dot(p_i, n_v_i) * p_i - projection onto arc plane
    dot_p_n = einsum('ij,ij->i', p_i, n_v_i)  # (N,)
    n_v_i_p = n_v_i - dot_p_n[:, np.newaxis] * p_i  # (N, 3)

    # Compute |n_v_i_p|
    n_v_i_p_len = np.sqrt(einsum('ij,ij->i', n_v_i_p, n_v_i_p))  # (N,)

    # Filter out cases where n_v_i_p is too small
    valid_mask2 = n_v_i_p_len > 1e-10
    if not np.any(valid_mask2):
        return None

    t_i = t_i[valid_mask2]
    vv_i = vv_i[valid_mask2]
    g_i = g_i[valid_mask2]
    w_i = w_i[valid_mask2]
    n_v_i_p = n_v_i_p[valid_mask2]
    n_v_i_p_len = n_v_i_p_len[valid_mask2]

    # Compute cos(phi) = dot(n_v, n_v_i_p) / |n_v_i_p|
    cos_phi = einsum('j,ij->i', n_v, n_v_i_p) / n_v_i_p_len  # (N,)

    # Clamp cos_phi to [-1, 1] for numerical stability in arccos
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)  # (N,)

    # Compute kappa_i (normal curvature)
    if page_curvature_formula:
        # Page et al. formula: kappa_i = phi / s (arc length)
        kappa_i = phi / g_i
    else:
        # Tong and Tang formula
        vv_i_len = np.sqrt(einsum('ij,ij->i', vv_i, vv_i))  # (N,)
        kappa_i = np.abs(2 * np.cos((pi - phi) / 2) / vv_i_len)

        # Compute sign: -sign(dot(t_i, n_v_i_p))
        # Normalize n_v_i_p for sign computation
        n_v_i_p_norm = n_v_i_p / n_v_i_p_len[:, np.newaxis]
        dot_t_n = einsum('ij,ij->i', t_i, n_v_i_p_norm)  # (N,)
        kappa_i_sign = -np.sign(dot_t_n)
        kappa_i = kappa_i * kappa_i_sign

    # Compute B_v = sum(w_i * kappa_i * outer(t_i, t_i))
    # einsum('n,n,ni,nj->ij', w_i, kappa_i, t_i, t_i) computes:
    #   sum over n of: w_i[n] * kappa_i[n] * t_i[n, i] * t_i[n, j]
    B_v = einsum('n,n,ni,nj->ij', w_i, kappa_i, t_i, t_i)  # (3, 3)

    # Normalize weights to sum to 2*pi
    sum_w_i = np.sum(w_i)
    if sum_w_i > 0:
        factor = (2 * pi) / sum_w_i
        B_v = B_v * factor / (2 * pi)

    return B_v


def prepare_vertex_arrays(graph):
    """
    Prepare numpy arrays from graph vertex properties for vectorized operations.

    Args:
        graph: graph_tool Graph object with vertex properties

    Returns:
        dict with keys 'xyz', 'normal', 'area' mapping to numpy arrays
    """
    return {
        'xyz': graph.vp.xyz.get_2d_array([0, 1, 2]).T,      # (V, 3)
        'normal': graph.vp.normal.get_2d_array([0, 1, 2]).T,  # (V, 3)
        'area': np.array(graph.vp.area.a),                    # (V,)
    }


def prepare_estimated_normal_arrays(graph):
    """
    Prepare numpy arrays including estimated normals for curvature voting.

    Args:
        graph: graph_tool Graph object with vertex properties including n_v

    Returns:
        dict with keys 'xyz', 'n_v', 'area' mapping to numpy arrays
    """
    return {
        'xyz': graph.vp.xyz.get_2d_array([0, 1, 2]).T,  # (V, 3)
        'n_v': graph.vp.n_v.get_2d_array([0, 1, 2]).T,  # (V, 3)
        'area': np.array(graph.vp.area.a),              # (V,)
    }
