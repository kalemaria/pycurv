import numpy as np
import math
import vtk

from surface_graphs import TriangleGraph


__author__ = 'kalemanov'

# CONSTANTS
SAMPLE_DST = 1
"""int: sampling distance in nm used in find_2_distances.
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
        maxdist (float): maximal distance (nm) from p0 the membrane
        tg_er (TriangleGraph): graph of the target membrane surface
        poly_er (vtkPolyData): the target membrane surface
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        the distance, the vertex and the intersection point
        or None, None, None in case no intersection was found
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
        return None, None, None

    # Get the vertex (at the triangle center) of cell1 from its id:
    v1 = tg_er.graph.vertex(cell1_id)

    if verbose:
        print("Intersection point found: ({}, {}, {})".format(
            p1[0], p1[1], p1[2]))
    p1 = np.array(p1)

    # Calculate the distance d1 from PM to cER:
    d1 = math.sqrt(np.dot(p1 - p0, p1 - p0))
    if d1 > maxdist:
        if verbose:
            print("d1 = {} > maxdist - discard the point".format(d1))
        return None, None, None
    elif verbose:
        print("d1 = {}".format(d1))

    # Return the distance, the vertex and the intersection point:
    return d1, v1, p1


def find_2_distances(
        p0, normal, maxdist, maxthick, tg_er, poly_er, verbose=False):
    """
    Given a point and a normal vector, finds two intersection points with a
    double membrane surface in the direction of the normal vector and measures
    the two distances (to the first surface and between the two surfaces).

    Args:
        p0 (numpy.ndarray): 3D point coordinates
        normal (numpy.ndarray): 3D normal vector
        maxdist (float): maximal distance (nm) from p0 to first membrane
        maxthick (float): maximal thickness (nm) from first to second membrane
        tg_er (TriangleGraph): graph of the target membrane surface
        poly_er (vtkPolyData): the target double membrane surface
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        the two distances and the vertices at the intersections: d1, d2, v1, v2
        or None, None, None, None in case less than two intersections were found
    """
    d1, v1, p1 = find_1_distance(
        p0, normal, maxdist, tg_er, poly_er, verbose)

    if v1 is None:  # no first intersection point found - stop looking
        return None, None, None, None

    # check if v1 is on first cER membrane:
    # is the angle between the normal from v1 and normal from v > 80 degrees?
    # then it's on the second membrane or edge - don't continue looking
    normal1 = tg_er.graph.vp.normal[v1]
    cos_angle1 = np.dot(normal, normal1)
    if cos_angle1 < math.cos(math.radians(80)):  # angle1 > 80 degrees
        if verbose:
            print("First intersection point found on second membrane - "
                  "discard it")
        return None, None, None, None

    d2, v2, p2 = None, None, None
    # look for second intersection within maxthick from p1, starting at a small
    # distance minthick from p1, so do not find the same membrane again
    for minthick in range(SAMPLE_DST, int(math.ceil(maxthick)), SAMPLE_DST):
        p0_new = p1 + normal * minthick
        d2_minus_minthick, v2, p2 = find_1_distance(
            p0_new, normal, maxthick - minthick, tg_er, poly_er, verbose=False)
        if v2 is None:  # no 2nd intersection - stop looking
            break
        else:
            # check if p2 is on second cER membrane:
            # is the angle between the normals from v1 and from v2 < pi/2?
            normal2 = tg_er.graph.vp.normal[v2]
            cos_angle2 = np.dot(normal1, normal2)
            if cos_angle2 > math.cos(math.radians(100)):  # angle2 < 100 degrees
                # then we are still on the first membrane or edge - keep looking
                minthick += SAMPLE_DST
            else:  # otherwise we are on the second membrane - stop looking
                d2 = d2_minus_minthick + minthick
                break

    if v2 is None:
        if verbose:
            print("No second intersection point found - discard the first")
        return None, None, None, None

    if verbose:
        print("Second intersection point found: ({}, {}, {})".format(
            p2[0], p2[1], p2[2]))

    if d2 > maxthick:
        if verbose:
            print("d2 = {} > maxthick - discard the pair".format(d2))
        return None, None, None, None
    elif verbose:
        print("d2 = {}".format(d2))

    # Return the two distances and the vertices at the intersections:
    return d1, d2, v1, v2


def find_2_distances_2_surf(
        p0, normal, maxdist, maxthick, tg1, poly1, tg2, poly2, verbose=False):
    """
    Given a point and a normal vector, finds two intersection points with two
    membrane surfaces in the direction of the normal vector and measures
    the two distances (to the first surface and between the two surfaces).

    Args:
        p0 (numpy.ndarray): 3D point coordinates
        normal (numpy.ndarray): 3D normal vector
        maxdist (float): maximal distance (nm) from p0 to first membrane
        maxthick (float): maximal thickness (nm) from first to second membrane
        tg1 (TriangleGraph): graph of the first target membrane surface
        poly1 (vtkPolyData): the first target membrane surface
        tg2 (TriangleGraph): graph of the second target membrane surface
        poly2 (vtkPolyData): the second target membrane surface
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        the two distances and the vertices at the intersections: d1, d2, v1, v2
        or None, None, None, None in case less than two intersections were found
    """
    # look for first intersection within maxdist from p0
    d1, v1, p1 = find_1_distance(
        p0, normal, maxdist, tg1, poly1, verbose)

    if v1 is None:  # no first intersection point found - stop looking
        return None, None, None, None

    # look for second intersection within maxthick from p1
    d2, v2, p2 = find_1_distance(
        p1, normal, maxthick, tg2, poly2, verbose)

    if v2 is None:
        if verbose:
            print("No second intersection point found - discard the first")
        return None, None, None, None

    if verbose:
        print("Second intersection point found: ({}, {}, {})".format(
            p2[0], p2[1], p2[2]))

    if d2 > maxthick:
        if verbose:
            print("d2 = {} > maxthick - discard the pair".format(d2))
        return None, None, None, None
    elif verbose:
        print("d2 = {}".format(d2))

    # Return the two distances and the vertices at the intersections:
    return d1, d2, v1, v2


def calculate_distances(
        tg_pm, tg_er, poly_er, maxdist, offset=0, both_directions=False,
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
        maxdist (float): maximal distance (nm) from PM to the cER membrane
        offset (float, optional): positive or negative offset (nm, default 0)
            to add to the distances, depending on how the surfaces where
            generated and/or in order to account for membrane thickness
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
    maxdist -= offset  # because add the offset to the distances

    # Initialize the vertex property in cER graph and distances lists:
    tg_er.graph.vp.PMdistance = tg_er.graph.new_vertex_property(
        "float", vals=-1)
    d1s = []  # distances from PM to cER

    # For each vertex v in PM graph (represents triangle on PM surface):
    for v in tg_pm.graph.vertices():
        if verbose:
            print("I'm at vertex {}".format(int(v)))

        # Get its center coordinate as p0 and corrected normal vector N_v:
        p0 = np.array(tg_pm.graph.vp.xyz[v])
        normal = np.array(tg_pm.graph.vp.N_v[v])

        # Look for distance d1 and intersected vertex v1 in cER graph:
        d1 = None
        v1 = None
        if both_directions:
            d1_sense, v1_sense, _ = find_1_distance(
                p0, normal, maxdist, tg_er, poly_er, verbose)
            d1_sense_inverse, v1_sense_inverse, _ = find_1_distance(
                p0, normal * -1, maxdist, tg_er, poly_er, verbose)
            # Find orientation:
            # if first ER membrane in "normal" direction is found
            if d1_sense is not None:
                # but not in opposite direction
                # or the distance is smaller in "normal" direction,
                # take the "normal" direction points and distances
                if (d1_sense_inverse is None) or (d1_sense < d1_sense_inverse):
                    d1 = d1_sense
                    v1 = v1_sense
                # otherwise take the opposite direction
                else:
                    d1 = d1_sense_inverse
                    v1 = v1_sense_inverse
            # if first ER membrane in opposite direction is found (but not in
            # "normal" direction), take the opposite direction data
            elif d1_sense_inverse is not None:
                d1 = d1_sense_inverse
                v1 = v1_sense_inverse
        elif reverse_direction:
            d1, v1, _ = find_1_distance(
                p0, normal * -1, maxdist, tg_er, poly_er, verbose)
        else:
            d1, v1, _ = find_1_distance(
                p0, normal, maxdist, tg_er, poly_er, verbose)
        if d1 is None:
            continue

        # Correct d1 with the specified offset (default 0):
        d1 += offset

        # Add d1 to the list:
        d1s.append(d1)

        # Fill out the v1 vertex property of cER graph "PMdistance" with d1:
        tg_er.graph.vp.PMdistance[v1] = d1

    return d1s


def calculate_distances_and_thicknesses(
        tg_pm, tg_er, poly_er, maxdist, maxthick, offset=0,
        both_directions=True, reverse_direction=False, verbose=False):
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
        maxdist (float): maximal distance (nm) from PM to first cER membrane
        maxthick (float): maximal thickness (nm) from first to second cER membrane
        offset (float, optional): positive or negative offset (nm, default 0)
            to add to the distances, depending on how the surfaces where
            generated and/or in order to account for membrane thickness
        both_directions (boolean, optional): if True, look in both directions of
            each PM normal (default), otherwise only in the normal direction
        reverse_direction (boolean, optional): if True, look in opposite
            direction of each PM normals (if both_directions True, will look
            in both directions)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        two lists of distances:
        d1: distances (nm) from PM to first cER membrane
        d2: cER thicknesses (nm)
    """
    print("maxdist = {} nm".format(maxdist))
    print("maxthick = {} nm".format(maxthick))
    maxdist -= offset  # because add the offset to the distances
    maxthick -= offset  # because add the offset to the distances

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

        # Get its center coordinate as p0 and corrected normal vector N_v:
        p0 = np.array(tg_pm.graph.vp.xyz[v])
        normal = np.array(tg_pm.graph.vp.N_v[v])

        # Look for distance d1, d2 and intersected vertices v1, v2 in cER graph:
        d1, d2 = None, None
        v1, v2 = None, None
        if both_directions:
            d1_sense, d2_sense, v1_sense, v2_sense = find_2_distances(
                p0, normal, maxdist, maxthick, tg_er, poly_er, verbose)
            d1_sense_inverse, d2_sense_inverse, \
                v1_sense_inverse, v2_sense_inverse = find_2_distances(
                    p0, normal * -1, maxdist, maxthick, tg_er, poly_er, verbose)
            # Find orientation:
            # if first ER membrane in "normal" direction is found
            if d1_sense is not None:
                # and first ER membrane in opposite direction is not found
                # or the first distance is smaller in "normal" direction,
                # take the "normal" direction points and distances
                if (d1_sense_inverse is None) or (d1_sense < d1_sense_inverse):
                    d1, d2 = d1_sense, d2_sense
                    v1, v2 = v1_sense, v2_sense
                # otherwise take the opposite direction
                else:
                    d1, d2 = d1_sense_inverse, d2_sense_inverse
                    v1, v2 = v1_sense_inverse, v2_sense_inverse
            # if first ER membrane in opposite direction is found (but not in
            # "normal" direction), take the opposite direction data
            elif d1_sense_inverse is not None:
                d1, d2 = d1_sense_inverse, d2_sense_inverse
                v1, v2 = v1_sense_inverse, v2_sense_inverse
        elif reverse_direction:
            d1, d2, v1, v2 = find_2_distances(
                p0, normal * -1, maxdist, maxthick, tg_er, poly_er, verbose)
        else:
            d1, d2, v1, v2 = find_2_distances(
                p0, normal, maxdist, maxthick, tg_er, poly_er, verbose)
        if d1 is None:
            continue

        # Correct d1 and d2 with the specified offset (default 0):
        d1 += offset
        d2 += offset

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


def calculate_thicknesses(
        tg_pm, tg_er, poly_er, maxdist, maxthick, offset=0,
        both_directions=True, reverse_direction=False, verbose=False):
    """
    Function to compute cortical ER (cER) thickness, using plasma
    membrane (PM) surface normals (closer to cER) and two inner cER surfaces.
    Adds vertex properties to the second cER graph:
    "cERthickness": d2 for the 2nd intersected triangle

    Args:
        tg_pm (TriangleGraph): graph of PM surface with corrected normals
        tg_er (TriangleGraph): graph of inner cER surface
        poly_er (vtkPolyData): inner cER surface
        maxdist (float): maximal distance (nm) from PM to first cER membrane
        maxthick (float): maximal thickness (nm) from first to second cER
            membrane
        offset (float, optional): positive or negative offset (nm, default 0)
            to add to the thicknesses, depending on how the surfaces where
            generated and/or in order to account for membrane thickness
        both_directions (boolean, optional): if True, look in both directions of
            each PM normal (default), otherwise only in the normal direction
        reverse_direction (boolean, optional): if True, look in opposite
            direction of each PM normals (default=False; if both_directions
            True, will look in both directions)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        a lists of cER thicknesses (nm)
    """
    print("maxdist = {} nm".format(maxdist))
    print("maxthick = {} nm".format(maxthick))
    maxthick -= offset  # because add the offset to the distances

    # Initialize vertex properties of the cER graph and distances lists:
    tg_er.graph.vp.firstcER = tg_er.graph.new_vertex_property("float", vals=-1)
    tg_er.graph.vp.cERthickness = tg_er.graph.new_vertex_property(
        "float", vals=-1)
    d2s = []  # distances between both cER membranes (cER thickness)

    # For each vertex v in PM graph (represents triangle on PM surface):
    for v in tg_pm.graph.vertices():
        if verbose:
            print("I'm at vertex {}".format(int(v)))

        # Get its center coordinate as p0 and corrected normal vector N_v:
        p0 = np.array(tg_pm.graph.vp.xyz[v])
        normal = np.array(tg_pm.graph.vp.N_v[v])

        # Look for distance d1, d2 and intersected vertices v1, v2 in cER graph:
        d1, d2 = None, None
        v1, v2 = None, None
        if both_directions:
            d1_sense, d2_sense, v1_sense, v2_sense = find_2_distances(
                p0, normal, maxdist, maxthick, tg_er, poly_er, verbose)
            d1_sense_inverse, d2_sense_inverse, \
                v1_sense_inverse, v2_sense_inverse = find_2_distances(
                    p0, normal * -1, maxdist, maxthick, tg_er, poly_er, verbose)
            # Find orientation:
            # if first ER membrane in "normal" direction is found
            if d1_sense is not None:
                # and first ER membrane in opposite direction is not found
                # or the first distance is smaller in "normal" direction,
                # take the "normal" direction points and distances
                if (d1_sense_inverse is None) or (d1_sense < d1_sense_inverse):
                    d1, d2 = d1_sense, d2_sense
                    v1, v2 = v1_sense, v2_sense
                # otherwise take the opposite direction
                else:
                    d1, d2 = d1_sense_inverse, d2_sense_inverse
                    v1, v2 = v1_sense_inverse, v2_sense_inverse
            # if first ER membrane in opposite direction is found (but not in
            # "normal" direction), take the opposite direction data
            elif d1_sense_inverse is not None:
                d1, d2 = d1_sense_inverse, d2_sense_inverse
                v1, v2 = v1_sense_inverse, v2_sense_inverse
        elif reverse_direction:
            d1, d2, v1, v2 = find_2_distances(
                p0, normal * -1, maxdist, maxthick, tg_er, poly_er, verbose)
        else:
            d1, d2, v1, v2 = find_2_distances(
                p0, normal, maxdist, maxthick, tg_er, poly_er, verbose)

        if d2 is not None:
            # Correct d2 with the specified offset (default 0) and add to the
            # list:
            d2 += offset
            d2s.append(d2)

            # fill out the vertex property of the second cER graph
            # "cERthickness": d2
            tg_er.graph.vp.firstcER[v1] = d2
            tg_er.graph.vp.cERthickness[v2] = d2

    return d2s
