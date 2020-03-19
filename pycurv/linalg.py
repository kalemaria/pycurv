import numpy as np
import math
from scipy.linalg import expm

"""
Set of linear algebra and some basic math functions.

Author: Maria Salfer (Max Planck Institute for Biochemistry)
"""

__author__ = 'Maria Salfer'


def perpendicular_vector(iv):
    """
    Finds a unit vector perpendicular to a given vector.
    Implementation of algorithm of Ahmed Fasih https://math.stackexchange.com/
    questions/133177/finding-a-unit-vector-perpendicular-to-another-vector

    Args:
        iv (numpy.ndarray): input 3D vector

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
    m = 0
    for m in range(3):
        if iv[m] != 0:
            break
    if m == 2:
        n = 0
    else:
        n = m + 1
    ov[n] = iv[m]
    ov[m] = -iv[n]
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
    return expm(A * theta)


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


def dot_norm(p, pnorm, norm):
    """
    Makes the dot-product between the input point and the closest point normal.
    Both vectors are first normalized.

    Args:
        p (numpy.ndarray): the input point, must be float numpy array
        pnorm (numpy.ndarray): the point normal, must be float numpy array
        norm (numpy.ndarray): the closest point normal, must be float numpy
            array

    Returns:
        the dot-product between the input point and the closest point normal
        (float)
    """
    # Point and vector coordinates
    v = pnorm - p

    # Normalization
    mv = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if mv > 0:
        v /= mv
    else:
        return 0
    mnorm = math.sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2])
    if mnorm > 0:
        norm /= mnorm
    else:
        return 0

    return v[0]*norm[0] + v[1]*norm[1] + v[2]*norm[2]


def nice_acos(cos_theta):
    """
    Returns the angle in radians given a cosine of the angle without ValueError.

    Args:
        cos_theta (float): cosine of an angle

    Returns:
        angle in radians
    """
    try:
        theta = math.acos(cos_theta)
    except ValueError:
        if cos_theta > 1.0:
            cos_theta = 1.0
        elif cos_theta < -1.0:  # was 0 before!
            cos_theta = -1.0  # was 0 before!
        theta = math.acos(cos_theta)
    return theta


def nice_asin(sin_theta):
    """
    Returns the angle in radians given a sine of the angle without ValueError.

    Args:
        sin_theta (float): sine of an angle

    Returns:
        angle in radians
    """
    try:
        theta = math.asin(sin_theta)
    except ValueError:
        if sin_theta > 1.0:
            sin_theta = 1.0
        elif sin_theta < -1.0:
            sin_theta = -1.0
        theta = math.asin(sin_theta)
    return theta


def triangle_normal(ref_normal, a, b, c):
    """
    Calculate triangle normal using 3 triangle points a, b, c, given their
    coordinates (x, y, z).

    Args:
        ref_normal (numpy.ndarray): 3D reference normal vector to correct the
            vector orientation
        a (numpy.ndarray): 3D point a coordinates
        b (numpy.ndarray): 3D point b coordinates
        c (numpy.ndarray): 3D point c coordinates

    Returns:
        normal vector of the triangle abc (numpy.ndarray).
    """
    u = b - a
    v = c - a
    u_cross_v = np.cross(u, v)
    u_cross_v_len = math.sqrt(np.dot(u_cross_v, u_cross_v))
    normal1 = u_cross_v / u_cross_v_len
    # orient normal like ref_normal:
    normal2 = - normal1
    cos_angle1 = np.dot(ref_normal, normal1)
    cos_angle2 = np.dot(ref_normal, normal2)
    if cos_angle1 > cos_angle2:  # angle1 < angle2
        return normal1
    else:
        return normal2


def triangle_center(a, b, c):
    """
    Calculate triangle center using 3 triangle points a, b, c, given their
    coordinates (x, y, z).

    Args:
        a (numpy.ndarray): 3D point a coordinates
        b (numpy.ndarray): 3D point b coordinates
        c (numpy.ndarray): 3D point c coordinates

    Returns:
        center coordinates of the triangle abc (1x3 numpy.ndarray).
    """
    return (a + b + c) / 3.0


def triangle_area_cross_product(a, b, c):
    """
    Calculate triangle area using 3 triangle points a, b, c, given their
    coordinates (x, y, z), and cross product formula.

    Args:
        a (numpy.ndarray): 3D point a coordinates
        b (numpy.ndarray): 3D point b coordinates
        c (numpy.ndarray): 3D point c coordinates

    Returns:
        area of the triangle abc (float).
    """
    ab_x, ab_y, ab_z = b - a
    ac_x, ac_y, ac_z = c - a
    return 0.5 * math.sqrt((ab_y * ac_z - ab_z * ac_y) ** 2 +
                           (ab_z * ac_x - ab_x * ac_z) ** 2 +
                           (ab_x * ac_y - ab_y * ac_x) ** 2)


def triangle_area_heron(a, b, c):
    """
    Calculate triangle area using 3 triangle points a, b, c, given their
    coordinates (x, y, z), and Heron's formula.

    Args:
        a (numpy.ndarray): 3D point a coordinates
        b (numpy.ndarray): 3D point b coordinates
        c (numpy.ndarray): 3D point c coordinates

    Returns:
        area of the triangle abc (float).
    """
    ab = euclidean_distance(a, b)
    bc = euclidean_distance(b, c)
    ac = euclidean_distance(a, c)
    s = 0.5 * (ab + bc + ac)  # half perimeter
    return math.sqrt(s * (s - ab) * (s - bc) * (s - ac))


def euclidean_distance(a, b):
    """
    Calculates and returns the Euclidean distance between two voxels.

    Args:
        a (np.ndarray): first voxel coordinates in form of an array of
            integers of length 3 [x1, y1, z1]
        b (np.ndarray): second voxel coordinates in form of an array of
            integers of length 3 [x2, y2, z2]

    Returns:
        the Euclidean distance between two voxels (float).
    """
    sum_of_squared_differences = np.sum((a - b) ** 2)
    return math.sqrt(sum_of_squared_differences)
