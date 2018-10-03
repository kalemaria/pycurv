import numpy as np
import math
import vtk

from surface_graphs import TriangleGraph


__author__ = 'kalemanov'

# CONSTANTS
SAMPLE_DST = 1
"""int: sampling distance used in find_2_distances.
"""


def find_1_distance(
        p0, normal, maxdist, tg_er, poly_er, verbose=False):
    """
    Given a point and a normal vector, finds the first intersection point with a
    membrane surface in the direction of the normal vector and measures
    the distance.

    Args:
        p0 (numpy.ndarray): 3D point coordinates
        normal (numpy.ndarray): 3D normal vector
        maxdist (int): maximal distance (nm) from p0 the membrane
        tg_er (TriangleGraph): graph of the target membrane surface
        poly_er (vtkPolyData): the target membrane surface
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        the distance and the vertex at the intersection: d1, v1
        or None, None in case no intersection was found
    """
    # Define a cellLocator to be able to compute intersections between lines
    # and the cER surface:
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(poly_er)
    locator.BuildLocator()
    tolerance = 0.001

    # Find a point pmax at distance maxdist from p0 in the normal direction:
    pmax = p0 + normal * maxdist

    # Find 1st intersection p1 on cER surface with the line from p0 to pmax
    # and the triangle containing it:
    # Outputs (we need only p1 and cell1_id):
    t = vtk.mutable(0)
    p1 = [0.0, 0.0, 0.0]  # x, y, z position of the first intersection
    pcoords = [0.0, 0.0, 0.0]
    sub_id = vtk.mutable(0)
    cell1_id = vtk.mutable(0)  # the triangle id containing p1
    locator.IntersectWithLine(p0, pmax, tolerance, t, p1, pcoords, sub_id,
                              cell1_id)
    # If there is no intersection, p1 stays like initialized:
    if p1 == [0.0, 0.0, 0.0]:
        if verbose:
            print("No intersection point found")
        return None, None

    # Get the vertex (at the triangle center) of cell1 from its id:
    v1 = tg_er.graph.vertex(cell1_id)

    if verbose:
        print("First intersection point found: ({}, {}, {})".format(
            p1[0], p1[1], p1[2]))
    p1 = np.array(p1)

    # Calculate the distance d1 from PM to cER:
    d1 = math.sqrt(np.dot(p1 - p0, p1 - p0))
    if d1 > maxdist:
        if verbose:
            print("d1 = {} > maxdist - discard the point".format(d1))
        return None, None
    elif verbose:
        print("d1 = {}".format(d1))

    # Return the distance and the vertex at the intersection point:
    return d1, v1


def find_2_distances(  # TODO refactor using find_1_distance
        p0, normal, maxdist, maxdist2, tg_er, poly_er, verbose=False):
    """
    Given a point and a normal vector, finds two intersection points with a
    double membrane surface in the direction of the normal vector and measures
    the two distances (to the first surface and between the two surfaces).

    Args:
        p0 (numpy.ndarray): 3D point coordinates
        normal (numpy.ndarray): 3D normal vector
        maxdist (int): maximal distance (nm) from p0 to first membrane
        maxdist2 (int): maximal distance (nm) from p0 to second membrane
        tg_er (TriangleGraph): graph of the target membrane surface
        poly_er (vtkPolyData): the target double membrane surface
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        the two distances and the vertices at the intersections: d1, d2, v1, v2
        or None, None, None, None in case less than two intersections were found
    """
    # Define a cellLocator to be able to compute intersections between lines
    # and the cER surface:
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(poly_er)
    locator.BuildLocator()
    tolerance = 0.001

    # functions used in the loop
    sqrt = math.sqrt
    dot = np.dot

    # find a point pmax at distance maxdist from p0 in the normal direction
    pmax = p0 + normal * maxdist

    # find 1st intersection p1 on cER surface with the line from p0 to pmax
    # and the triangle containing it
    # Outputs (we need only p1 and cell1_id):
    t = vtk.mutable(0)
    p1 = [0.0, 0.0, 0.0]  # x, y, z position of the first intersection
    pcoords = [0.0, 0.0, 0.0]
    sub_id = vtk.mutable(0)
    cell1_id = vtk.mutable(0)  # the triangle id containing p1
    locator.IntersectWithLine(p0, pmax, tolerance, t, p1, pcoords, sub_id,
                              cell1_id)
    # If there is no intersection, p1 stays like initialized:
    if p1 == [0.0, 0.0, 0.0]:
        if verbose:
            print("No intersection point found")
        return None, None, None, None

    # get the vertex (at the triangle center) of cell1 from its id:
    v1 = tg_er.graph.vertex(cell1_id)
    # check if p1 is on first cER membrane:
    # is the angle between the normal from v1 and normal from v > pi/2?
    # then it's on the second membrane - don't continue looking
    N1 = tg_er.graph.vp.N_v[v1]
    cos_angle1 = np.dot(normal, N1)
    if cos_angle1 < 0:  # angle1 > pi/2
        if verbose:
            print("First intersection point found on second membrane - "
                  "discard it")
        return None, None, None, None

    if verbose:
        print("First intersection point found: ({}, {}, {})".format(
            p1[0], p1[1], p1[2]))
    p1 = np.array(p1)

    # calculate the distance d1 from PM to cER:
    d1 = sqrt(dot(p1 - p0, p1 - p0))
    if d1 > maxdist:
        if verbose:
            print("d1 = {} > maxdist - discard the point".format(d1))
        return None, None, None, None
    elif verbose:
        print("distance from PM to cER d1 = {}".format(d1))

    t = vtk.mutable(0)
    p2 = [0.0, 0.0, 0.0]  # x, y, z position of the next intersection
    pcoords = [0.0, 0.0, 0.0]
    sub_id = vtk.mutable(0)
    cell2_id = vtk.mutable(0)  # the triangle id containing p2
    v2 = None
    maxth = maxdist2 - int(d1)
    for minth in range(SAMPLE_DST, maxth, SAMPLE_DST):
        # find two new points: at distance minth and maxth from p1 in the
        # normal direction
        p0_new = p1 + normal * minth
        pmax_new = p1 + normal * maxth

        # find 2nd intersection p2 on cER surface with the line from p0_new
        # to pmax_new and the triangle containing it
        locator.IntersectWithLine(p0_new, pmax_new, tolerance, t, p2,
                                  pcoords, sub_id, cell2_id)
        if p2 == [0.0, 0.0, 0.0]:  # no 2nd intersection - stop looking
            break
        else:
            # get the vertex (at the triangle center) of cell2 from its id:
            v2 = tg_er.graph.vertex(cell2_id)
            # check if p2 is on second cER membrane:
            # is the angle between the normals from v1 and from v2 < pi/2?
            # then we are still on the first membrane - continue looking
            N2 = tg_er.graph.vp.N_v[v2]
            cos_angle2 = np.dot(N1, N2)
            if cos_angle2 > 0:  # angle2 < pi/2
                # if verbose:
                #     print("Second intersection point found on the first "
                #           "membrane - continue looking")
                minth += 1
            else:  # otherwise we are on the second membrane - stop looking
                break
    # If there is no intersection, p2 stays like initialized:
    if p2 == [0.0, 0.0, 0.0]:
        if verbose:
            print("No second intersection point found - discard the first")
        return None, None, None, None

    if verbose:
        print("Second intersection point found: ({}, {}, {})".format(
            p2[0], p2[1], p2[2]))
    p2 = np.array(p2)

    # calculate the cER thickness d2:
    d2 = sqrt(dot(p2 - p1, p2 - p1))
    if verbose:
        print("cER thickness d2 = {}".format(d2))
    if d2 > maxth:
        if verbose:
            print("d2 = {} > maxth - discard the pair".format(d2))
        return None, None, None, None

    # Return the two distances and the vertices at the intersections:
    return d1, d2, v1, v2


def calculate_distances(
        tg_pm, tg_er, poly_er, maxdist, both_directions=False,
        reverse_direction=False, verbose=False):
    """
    Function to compute shortest distances between two membranes, here a plasma
    membrane (PM) and cortical ER (cER) using their surfaces.
    Adds a vertex property to cER graph:
    "PMdistance" with a distance from PM for the intersected triangles.

    Args:
        tg_pm (TriangleGraph): graph of PM surface with corrected normals
        tg_er (TriangleGraph): graph of cER surface
        poly_er (vtkPolyData): cER surface
        maxdist (int): maximal distance (nm) from PM to the cER membrane
        both_directions (boolean, optional): if True, look in both directions of
            each PM normal, otherwise only in the normal direction (default)
        reverse_direction (boolean, optional): if True, look in opposite
            direction of each PM normals (if both_directions True, will look
            in both directions)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        a lists of distances (nm) between the two membranes
    """
    print("maxdist = {} nm".format(maxdist))

    # Initialize the vertex property in cER graph and distances lists:
    tg_er.graph.vp.PMdistance = tg_er.graph.new_vertex_property(
        "float", vals=-1)
    d1s = []  # distances from PM to cER

    # For each vertex v in PM graph (represents triangle on PM surface):
    for v in tg_pm.graph.vertices():
        if verbose:
            print("I'm at vertex {}".format(int(v)))

        # Get its center coordinate as p0 and corrected normal vector normal:
        p0 = np.array(tg_pm.graph.vp.xyz[v])
        normal = np.array(tg_pm.graph.vp.N_v[v])

        # Look for distance d1 and intersected vertex v1 in cER graph:
        d1 = None
        v1 = None
        if both_directions:
            d1_sense, v1_sense = find_1_distance(
                p0, normal, maxdist, tg_er, poly_er, verbose)
            d1_sense_inverse, v1_sense_inverse = find_1_distance(
                p0, normal * -1, maxdist, tg_er, poly_er, verbose)
            # Find orientation:
            # if first ER membrane in "normal" direction is found
            if d1_sense is not None:
                # but not in opposite direction
                if d1_sense_inverse is None:
                    d1 = d1_sense
                    v1 = v1_sense
                # or the distance is smaller in "normal" direction,
                # take the "normal" direction points and distances
                elif d1_sense < d1_sense_inverse:
                    d1 = d1_sense
                    v1 = v1_sense
                # otherwise take the opposite direction
                else:
                    d1 = d1_sense_inverse
                    v1 = v1_sense_inverse
            # if first ER membrane in opposite direction is found but not in
            # "normal" direction, take the opposite direction data
            elif d1_sense_inverse is not None:
                d1 = d1_sense_inverse
                v1 = v1_sense_inverse
        elif reverse_direction:
            d1, v1 = find_1_distance(
                p0, normal * -1, maxdist, tg_er, poly_er, verbose)
        else:
            d1, v1 = find_1_distance(
                p0, normal, maxdist, tg_er, poly_er, verbose)

        if d1 is None:
            continue

        # Add d1 to the list:
        d1s.append(d1)

        # Fill out the v1 vertex property of cER graph "PMdistance" with d1:
        tg_er.graph.vp.PMdistance[v1] = d1

    return d1s


def calculate_distances_and_thicknesses(  # TODO add options both_directions and reverse_direction
        tg_pm, tg_er, poly_er, maxdist, maxdist2, verbose=False):
    """
    Function to compute shortest distances between two membranes, here a plasma
    membrane (PM) and cortical ER (cER) using their surfaces.
    Adds vertex properties to cER graph:
    "cERmembrane": 1 for the 1st intersected triangle and 2 for the 2nd one
    "PMdistance": d1 for the 1st intersected triangle
    "cERthickness": d2 for the 2nd intersected triangle

    Args:
        tg_pm (TriangleGraph): graph of PM surface with corrected normals
        tg_er (TriangleGraph): graph of cER surface
        poly_er (vtkPolyData): cER surface
        maxdist (int): maximal distance (nm) from PM to first cER membrane
        maxdist2 (int): maximal distance (nm) from PM to second cER membrane
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        two lists of distances:
        d1: distances (nm) from PM to first cER membrane
        d2: cER thicknesses (nm)
    """
    print("maxdist = {} nm".format(maxdist))
    print("maxdist2 = {} nm".format(maxdist2))

    # Initialize the vertex properties if cER graph and distances lists:
    tg_er.graph.vp.cERmembrane = tg_er.graph.new_vertex_property("int")
    tg_er.graph.vp.PMdistance = tg_er.graph.new_vertex_property(
        "float", vals=-1)
    tg_er.graph.vp.cERthickness = tg_er.graph.new_vertex_property(
        "float", vals=-1)
    d1s = []  # distances from PM to cER
    d2s = []  # distances between both cER membranes (cER thickness)

    # For each vertex v in PM graph (represents triangle on PM surface):
    for v in tg_pm.graph.vertices():
        if verbose:
            print("I'm at vertex {}".format(int(v)))

        # Get its center coordinate as p0 and corrected normal vector N_v
        p0 = np.array(tg_pm.graph.vp.xyz[v])
        N_v = np.array(tg_pm.graph.vp.N_v[v])

        # Look for distances in both directions of the PM normal
        d1_s, d2_s, v1_s, v2_s = find_2_distances(
            p0, N_v, maxdist, maxdist2, tg_er, poly_er, verbose)  # "sense"
        d1_si, d2_si, v1_si, v2_si = find_2_distances(
            p0, N_v * -1, maxdist, maxdist2, tg_er, poly_er, verbose)
        # "sense inverse"

        # Find orientation
        d1, d2 = None, None
        v1, v2 = None, None
        # if first ER membrane in "normal" direction is found
        if d1_s is not None:
            # and first ER membrane in opposite direction is not found
            if d1_si is None:
                d1, d2 = d1_s, d2_s
                v1, v2 = v1_s, v2_s
            # or the first distance is smaller in "normal" direction,
            # take the "normal" direction points and distances
            elif d1_s < d1_si:
                d1, d2 = d1_s, d2_s
                v1, v2 = v1_s, v2_s
            # otherwise take the opposite direction
            else:
                d1, d2 = d1_si, d2_si
                v1, v2 = v1_si, v2_si
        # if first ER membrane in opposite direction is found (but not in
        # "normal" direction), take the opposite direction points and distances
        elif d1_si is not None:
            d1, d2 = d1_si, d2_si
            v1, v2 = v1_si, v2_si

        if d1 is None:
            continue

        # Add d1 and d2 to lists:
        d1s.append(d1)
        d2s.append(d2)

        # Fill out the vertex property of cER graph "cERmembrane":
        # 1 for the 1st intersected triangle and 2 for the 2nd one
        tg_er.graph.vp.cERmembrane[v1] = 1
        tg_er.graph.vp.cERmembrane[v2] = 2

        # Fill out the vertex property of cER graph "PMdistance":
        # d1 for the 1st intersected triangle
        tg_er.graph.vp.PMdistance[v1] = d1

        # fill out the vertex property of cER graph  "cERthickness":
        # d2 for the 2nd intersected triangle
        tg_er.graph.vp.cERthickness[v2] = d2

    return d1s, d2s
