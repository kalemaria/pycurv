import numpy as np
import scipy
# import pyfits
# import warnings
import os
import vtk
import math
from pyto.io.image_io import ImageIO

import pexceptions

"""
Set of functions and a class (TypesConverter) for reading and writing different
data types.

Authors: Maria Kalemanov and Antonio Martinez-Sanchez (Max Planck Institute for
Biochemistry)
"""

__author__ = 'martinez and kalemanov'


# CONSTANTS
MAX_DIST_SURF = 3
"""int: a constant determining the maximal distance in pixels of a point on the
surface from the segmentation mask, used in gen_surface method.
"""


def load_tomo(fname, mmap=False):
    """
    Loads a tomogram in MRC, EM or VTI format and converts it into a numpy
    format.

    Args:
        fname (str): full path to the tomogram, has to end with '.mrc', '.em' or
            '.vti'
        mmap (boolean, optional): if True (default False) a numpy.memmap object
            is loaded instead of numpy.ndarray, which means that data are not
            loaded completely to memory, this is useful only for very large
            tomograms. Only valid with formats MRC and EM. VERY IMPORTANT: This
            subclass of ndarray has some unpleasant interaction with some
            operations, because it does not quite fit properly as a subclass of
            numpy.ndarray

    Returns:
        numpy.ndarray or numpy.memmap object
    """
    # Input parsing
    stem, ext = os.path.splitext(fname)
    if mmap and (not (ext == '.mrc' or (ext == '.em'))):
        error_msg = ('mmap option is only valid for MRC or EM formats, current '
                     + ext)
        raise pexceptions.PySegInputError(expr='load_tomo', msg=error_msg)

    # if ext == '.fits':
    #     im_data = pyfits.getdata(fname).transpose()
    elif ext == '.mrc':
        image = ImageIO()
        if mmap:
            image.readMRC(fname, memmap=mmap)
        else:
            image.readMRC(fname)
        im_data = image.data
    elif ext == '.em':
        image = ImageIO()
        if mmap:
            image.readEM(fname, memmap=mmap)
        else:
            image.readEM(fname)
        im_data = image.data
    elif ext == '.vti':
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fname)
        reader.Update()
        im_data = vti_to_numpy(reader.GetOutput())
    else:
        error_msg = '%s is non valid format.' % ext
        raise pexceptions.PySegInputError(expr='load_tomo', msg=error_msg)

    # For avoiding 2D arrays
    if len(im_data.shape) == 2:
        im_data = np.reshape(im_data, (im_data.shape[0], im_data.shape[1], 1))

    return im_data


def vti_to_numpy(image, transpose=True):  # TODO Antonio, please check the function description, why transpose=True by default?
    """
    Converts VTK image data (that was read in from a VTI file) into a numpy
    format.

    Args:
        image (vtkImageData): input VTK image data, must be a scalar field
            (output of vtk.vtkXMLImageDataReader)
        transpose (boolean, optional): if True (default), the image is
            transposed (x and z axes are switched)

    Returns:
        numpy.ndarray with the image data
    """
    # Read tomogram data
    image.ComputeBounds()
    nx, ny, nz = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    if transpose:
        dout = np.zeros(shape=[nz, ny, nx],
                        dtype=TypesConverter.vtk_to_numpy(scalars))
        for i in range(scalars.GetNumberOfTuples()):
            [x, y, z] = image.GetPoint(i)
            dout[int(z), int(y), int(x)] = scalars.GetTuple1(i)
    else:
        dout = np.zeros(shape=[nx, ny, nz],
                        dtype=TypesConverter.vtk_to_numpy(scalars))
        for i in range(scalars.GetNumberOfTuples()):
            [x, y, z] = image.GetPoint(i)
            dout[int(x), int(y), int(z)] = scalars.GetTuple1(i)

    return dout


def save_numpy(array, fname):  # TODO Antonio, please check the function description
    """
    Saves a numpy array to a file in MRC, EM or VTI format.

    Args:
        array (numpy.ndarray): input array
        fname (str): full path to the tomogram, has to end with '.mrc', '.em' or
            '.vti'

    Returns:
        None
    """
    _, ext = os.path.splitext(fname)

    # Parse input array for fulfilling format requirements
    if (ext == '.mrc') or (ext == '.em'):
        if ((array.dtype != 'ubyte') and (array.dtype != 'int16') and
                (array.dtype != 'float32')):
            array = array.astype('float32')
        # if (len(array.shape) == 3) and (array.shape[2] == 1):
        #   array = np.reshape(array, (array.shape[0], array.shape[1]))

    if ext == '.vti':
        pname, fnameh = os.path.split(fname)
        save_vti(numpy_to_vti(array), fnameh, pname)
    # elif ext == '.fits':
    #     warnings.resetwarnings()
    #     warnings.filterwarnings('ignore', category=UserWarning, append=True)
    #     pyfits.writeto(fname, array, clobber=True, output_verify='silentfix')
    #     warnings.resetwarnings()
    #     warnings.filterwarnings('always', category=UserWarning, append=True)
    elif ext == '.mrc':
        img = ImageIO()
        # img.setData(np.transpose(array, (1, 0, 2)))
        img.setData(array)
        img.writeMRC(fname)
    elif ext == '.em':
        img = ImageIO()
        # img.setData(np.transpose(array, (1, 0, 2)))
        img.setData(array)
        img.writeEM(fname)
    else:
        error_msg = 'Format not valid %s.' % ext
        raise pexceptions.PySegInputError(expr='save_numpy', msg=error_msg)


def numpy_to_vti(array, offset=[0, 0, 0], spacing=[1, 1, 1]):  # TODO Antonio, please check the function description
    """
    Converts a numpy array into a VTK image data object.

    Args:
        array (numpy.ndarray): input numpy array
        offset (int [3], optional): the reading start positions in x, y and z
            dimensions, default [0, 0, 0]
        spacing (float [3], optional): the spacing (width, height, length) of
            the cubical cells that compose the data set, default [1, 1, 1]

    Returns:
        vtk.vtkImageData object
    """
    nx, ny, nz = array.shape
    image = vtk.vtkImageData()
    image.SetExtent(offset[0], nx+offset[0]-1, offset[1], ny+offset[1]-1,
                    offset[2], nz+offset[2]-1)
    image.SetSpacing(spacing)
    image.AllocateScalars(vtk.VTK_FLOAT, 1)
    scalars = image.GetPointData().GetScalars()

    for x in range(offset[0], nx):
        for y in range(offset[1], ny):
            for z in range(offset[2], nz):
                scalars.SetTuple1(image.ComputePointId([x, y, z]),
                                  float(array[x, y, z]))

    return image


def save_vti(image, fname, outputdir):  # TODO Antonio, please check the function description
    """
    Saves a VTK image data object into a VTI file.

    Args:
        image (vtk.vtkImageData):
        fname (str): output file name, should end with '.vti'
        outputdir (str): output directory

    Returns:
        None
    """
    writer = vtk.vtkXMLImageDataWriter()
    outputfile = outputdir + '/' + fname
    writer.SetFileName(outputfile)
    writer.SetInputData(image)
    if writer.Write() != 1:
        error_msg = 'Error writing the %s file on %s.' % (fname, outputdir)
        raise pexceptions.PySegInputError(expr='save_vti', msg=error_msg)


def save_vtp(poly, fname):  # TODO Antonio, please check the function description
    """
    Saves a VTK PolyData object into a VTP file.

    Args:
        poly (vtk.vtkPolyData): input VTK PolyData object
        fname (str): output file name

    Returns:
        None
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly)
    if writer.Write() != 1:
        error_msg = 'Error writing the file %s.' % fname
        raise pexceptions.PySegInputError(expr='save_vtp', msg=error_msg)


def load_poly(fname):  # TODO Antonio, please check the function description
    """
    Loads data from a VTK PolyData (VTP) file.

    Args:
        fname (str): input file name, should end with '.vtp'

    Returns:
        vtk.vtkPolyData object
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()


def gen_surface(tomo, lbl=1, mask=True, purge_ratio=1, field=False,
                mode_2d=False, verbose=False):  # TODO Antonio, please check the function description
    """
    Generates a VTK PolyData surface from a segmented tomogram.

    Args:
        tomo (numpy.ndarray or str): the input segmentation as numpy ndarray or
            the file name in MRC, EM or VTI format
        lbl (int, optional): label for the foreground, default 1
        mask (boolean, optional): if True (default), the input segmentation is
            used as mask for the surface
        purge_ratio (int, optional): if greater than 1 (default), then 1 every
            purge_ratio points of the segmentation are randomly deleted
        field (boolean, optional): if True (default False), additionally returns
            the polarity distance scalar field
        mode_2d (boolean, optional): needed for polarity distance calculation
            (if field is True), if True (default False), ...
        verbose (boolean, optional): if True (default False), prints out
            messages for checking the progress

    Returns:
        - output surface (vtk.vtkPolyData)
        - polarity distance scalar field (np.ndarray), if field is True
    """
    # Check input format
    if isinstance(tomo, str):
        fname, fext = os.path.splitext(tomo)
        # if fext == '.fits':
        #     tomo = pyfits.getdata(tomo)
        if fext == '.mrc':
            hold = ImageIO()
            hold.readMRC(file=tomo)
            tomo = hold.data
        elif fext == '.em':
            hold = ImageIO()
            hold.readEM(file=tomo)
            tomo = hold.data
        elif fext == '.vti':
            reader = vtk.vtkXMLImageDataReader()
            reader.SetFileName(tomo)
            reader.Update()
            tomo = vti_to_numpy(reader.GetOutput())
        else:
            error_msg = 'Format %s not readable.' % fext
            raise pexceptions.PySegInputError(expr='gen_surface', msg=error_msg)
    elif not isinstance(tomo, np.ndarray):
        error_msg = 'Input must be either a file name or a ndarray.'
        raise pexceptions.PySegInputError(expr='gen_surface', msg=error_msg)

    # Load file with the cloud of points
    nx, ny, nz = tomo.shape
    cloud = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cloud.SetPoints(points)

    if purge_ratio <= 1:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if tomo[x, y, z] == lbl:
                        points.InsertNextPoint(x, y, z)
    else:
        count = 0
        mx_value = purge_ratio - 1
        purge = np.random.randint(0, purge_ratio+1, nx*ny*nz)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if purge[count] == mx_value:
                        if tomo[x, y, z] == lbl:
                            points.InsertNextPoint(x, y, z)
                    count += 1

    if verbose:
        print 'Cloud of points loaded...'

    # Creating the isosurface
    surf = vtk.vtkSurfaceReconstructionFilter()
    # surf.SetSampleSpacing(2)
    surf.SetSampleSpacing(purge_ratio)
    # surf.SetNeighborhoodSize(10)
    surf.SetInputData(cloud)
    contf = vtk.vtkContourFilter()
    contf.SetInputConnection(surf.GetOutputPort())
    # if thick is None:
    contf.SetValue(0, 0)
    # else:
        # contf.SetValue(0, thick)

    # Sometimes the contouring algorithm can create a volume whose gradient
    # vector and ordering of polygon (using the right hand rule) are
    # inconsistent. vtkReverseSense cures    this problem.
    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(contf.GetOutputPort())
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    rsurf = reverse.GetOutput()

    if verbose:
        print 'Isosurfaces generated...'

    # Translate and scale to the proper positions
    cloud.ComputeBounds()
    rsurf.ComputeBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = cloud.GetBounds()
    rxmin, rxmax, rymin, rymax, rzmin, rzmax = rsurf.GetBounds()
    scale_x = (xmax-xmin) / (rxmax-rxmin)
    scale_y = (ymax-ymin) / (rymax-rymin)
    denom = rzmax - rzmin
    num = zmax - xmin
    if (denom == 0) or (num == 0):
        scale_z = 1
    else:
        scale_z = (zmax-zmin) / (rzmax-rzmin)
    transp = vtk.vtkTransform()
    transp.Translate(xmin, ymin, zmin)
    transp.Scale(scale_x, scale_y, scale_z)
    transp.Translate(-rxmin, -rymin, -rzmin)
    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetInputData(rsurf)
    tpd.SetTransform(transp)
    tpd.Update()
    tsurf = tpd.GetOutput()

    if verbose:
        print 'Rescaled and translated...'

    # Masking according to distance to the original segmentation
    if mask:
        tomod = scipy.ndimage.morphology.distance_transform_edt(
            np.invert(tomo == lbl))
        for i in range(tsurf.GetNumberOfCells()):

            # Check if all points which made up the polygon are in the mask
            points_cell = tsurf.GetCell(i).GetPoints()
            count = 0
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                if (tomod[int(round(x)), int(round(y)), int(round(z))]
                        > MAX_DIST_SURF):
                    count += 1

            if count > 0:
                tsurf.DeleteCell(i)

        # Release free memory
        tsurf.RemoveDeletedCells()

        if verbose:
            print 'Mask applied...'

    # Field distance
    if field:

        # Get normal attributes
        norm_flt = vtk.vtkPolyDataNormals()
        norm_flt.SetInputData(tsurf)
        norm_flt.ComputeCellNormalsOn()
        norm_flt.AutoOrientNormalsOn()
        norm_flt.ConsistencyOn()
        norm_flt.Update()
        tsurf = norm_flt.GetOutput()
        # for i in range(tsurf.GetPointData().GetNumberOfArrays()):
        #    array = tsurf.GetPointData().GetArray(i)
        #    if array.GetNumberOfComponents() == 3:
        #        break
        array = tsurf.GetCellData().GetNormals()

        # Build membrane mask
        tomoh = np.ones(shape=tomo.shape, dtype=np.bool)
        tomon = np.ones(shape=(tomo.shape[0], tomo.shape[1], tomo.shape[2], 3),
                        dtype=TypesConverter().vtk_to_numpy(array))
        # for i in range(tsurf.GetNumberOfCells()):
        #     points_cell = tsurf.GetCell(i).GetPoints()
        #     for j in range(0, points_cell.GetNumberOfPoints()):
        #         x, y, z = points_cell.GetPoint(j)
        #         # print x, y, z, array.GetTuple(j)
        #         x, y, z = int(round(x)), int(round(y)), int(round(z))
        #         tomo[x, y, z] = False
        #         tomon[x, y, z, :] = array.GetTuple(j)
        for i in range(tsurf.GetNumberOfCells()):
            points_cell = tsurf.GetCell(i).GetPoints()
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                # print x, y, z, array.GetTuple(j)
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                if tomo[x, y, z] == lbl:
                    tomoh[x, y, z] = False
                    tomon[x, y, z, :] = array.GetTuple(i)

        # Distance transform
        tomod, ids = scipy.ndimage.morphology.distance_transform_edt(
            tomoh, return_indices=True)

        # Compute polarity
        if mode_2d:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        i_x, i_y, i_z = (ids[0, x, y, z], ids[1, x, y, z],
                                         ids[2, x, y, z])
                        norm = tomon[i_x, i_y, i_z]
                        norm[2] = 0
                        pnorm = (i_x, i_y, 0)
                        p = (x, y, 0)
                        dprod = dot_norm(np.asarray(p, dtype=np.float),
                                         np.asarray(pnorm, dtype=np.float),
                                         np.asarray(norm, dtype=np.float))
                        tomod[x, y, z] = tomod[x, y, z] * np.sign(dprod)
        else:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        i_x, i_y, i_z = (ids[0, x, y, z], ids[1, x, y, z],
                                         ids[2, x, y, z])
                        hold_norm = tomon[i_x, i_y, i_z]
                        norm = hold_norm
                        # norm[0] = (-1) * hold_norm[1]
                        # norm[1] = hold_norm[0]
                        # norm[2] = hold_norm[2]
                        pnorm = (i_x, i_y, i_z)
                        p = (x, y, z)
                        dprod = dot_norm(np.asarray(pnorm, dtype=np.float),
                                         np.asarray(p, dtype=np.float),
                                         np.asarray(norm, dtype=np.float))
                        tomod[x, y, z] = tomod[x, y, z] * np.sign(dprod)

        if verbose:
            print 'Distance field generated...'

        return tsurf, tomod

    if verbose:
        print 'Finished!'

    return tsurf


def dot_norm(p, pnorm, norm):  # TODO Antonio, please check the function description
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


def write_values_to_file(values, filename):
    """
    Writes a list or numpy array of values into a file as a column.

    Args:
        values (list or numpy.ndarray): a list or 1D array of values
        filename (str): an output file name (with path)

    Returns:
        None
    """
    with open(filename, 'w') as f:
        for i in xrange(len(values)):
            f.write(str(values[i]) + '\n')


def merge_vtp_files(vtp_file_list, outfilename):
    """
    Merges a list of '.vtp' files to one file.

    Args:
        vtp_file_list (str list): a list of strings with paths and names of
            '.vtp' files
        outfilename (str): an output file name (with path)

    Returns:
        None
    """
    poly_list = []
    for vtp_file in vtp_file_list:
        poly = load_poly(vtp_file)
        poly_list.append(poly)
    merged_poly = append_polys(poly_list)
    save_vtp(merged_poly, outfilename)


def append_polys(poly_list):
    """
    Appends a list of VTK PolyData objects to one object.

    Args:
        poly_list (list): a list of vtk.vtkPolyData objects

    Returns:
        vtk.vtkPolyData object containing all input objects
    """
    appender = vtk.vtkAppendPolyData()
    for poly in poly_list:
        appender.AddInputData(poly)
    appender.Update()
    return appender.GetOutput()


def vtp_file_to_stl_file(infilename, outfilename):
    """
    Converts a '.vtp' file to an '.stl' file.

    Args:
        infilename (str): an input '.vtp' file name (with path)
        outfilename (str): an output '.stl' file name (with path)

    Returns:
        None
    """
    poly = load_poly(infilename)
    write_stl_file(poly, outfilename)


def write_stl_file(poly, outfilename):
    """
    Writs an '.stl' file from a VTK PolyData object.

    Args:
        poly (vtk.vtkPolyData): an input VTK PolyData object
        outfilename (str): an output '.stl' file name (with path)

    Returns:
        None
    """
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(outfilename)
    stlWriter.SetInputDataObject(poly)
    stlWriter.Update()
    stlWriter.Write()


def poly_array_to_volume(poly, array_name, scale_factor_to_nm, scale_x, scale_y,
                         scale_z, logfilename=None, mean=False, verbose=False):
    """
    Converts a triangle-cell data array of the given vtkPolyData to a 3D array
    of size like the underlying segmentation.

    Initializes a 3D matrix of size like the segmentation with zeros, calculates
    triangle centroid coordinates, transforms them from nanometers to voxels and
    puts the corresponding cell data value into the voxel.

    If more than one triangles map to the same voxel, takes the maximal or mean
    value. Optionally, logs such cases by writing out the voxel coordinates and
    the value list into a file.

    Args:
        poly (vtk.vtkPolyData): a vtkPolyData object with triangle-cells.
        array_name (str): name of the desired cell data array of the vtkPolyData
            object
        scale_factor_to_nm (float): pixel size in nanometers that was used for
            scaling the graph
        scale_x (int): x axis length in pixels of the segmentation
        scale_y (int): y axis length in pixels of the segmentation
        scale_z (int): z axis length in pixels of the segmentation
        logfilename (str, optional): specifies an output log file path (default
            None) for listing voxel coordinates with multiple values mapping to
            this voxel
        mean (boolean, optional): if True, takes the mean value in case multiple
            triangles map to the same voxel; if False (default), takes the
            maximal value
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out

    Returns:
        the 3D numpy.ndarray of size like the segmentation containing the cell
            data values at the corresponding coordinates
    """
    print ('Converting the vtkPolyData cell array %s to a 3D volume...'
           % array_name)
    # Find the array with the wanted name:
    array = None
    numberOfCellArrays = poly.GetCellData().GetNumberOfArrays()
    if numberOfCellArrays > 0:
        for i in range(numberOfCellArrays):
            if array_name == poly.GetCellData().GetArrayName(i):
                array = poly.GetCellData().GetArray(i)
                break
    else:
        print 'No cell arrays present in the PolyData!'
        return None

    # Check that the array was found and that it has 1 component values:
    if array is None:
        print 'Array %s was not found!' % array_name
        return None
    n_comp_array = array.GetNumberOfComponents()
    if n_comp_array != 1:
        print ('Array has %s components but 1 component is expected!'
               % n_comp_array)
        return None

    # Dictionary mapping voxel coordinates (for the volume returned later) to a
    # list of values falling within that voxel:
    voxel_to_values = {}

    # For each cell:
    for cell_id in xrange(poly.GetNumberOfCells()):
        # Get the cell i and check if it's a triangle:
        cell = poly.GetCell(cell_id)
        if isinstance(cell, vtk.vtkTriangle):

            # Get its vertex coordinates and calculate the centroid of the
            # triangle (because it is not saved as a vtkPolyData array):
            points_cell = cell.GetPoints()
            x_center = 0
            y_center = 0
            z_center = 0
            for j in range(0, 3):
                x, y, z = points_cell.GetPoint(j)
                x_center += x
                y_center += y
                z_center += z
            x_center /= 3
            y_center /= 3
            z_center /= 3

            # Get the array value assigned to the triangle cell:
            cell_value = array.GetTuple1(cell_id)

            # Calculate the corresponding voxel of the vertex and add the value
            # to the list keyed by the voxel in the dictionary:
            # Scaling the coordinates back from nm to voxels. (Without round
            # float coordinates are truncated to the next lowest integer.)
            voxel_x = int(round(x_center / scale_factor_to_nm))
            voxel_y = int(round(y_center / scale_factor_to_nm))
            voxel_z = int(round(z_center / scale_factor_to_nm))
            voxel = (voxel_x, voxel_y, voxel_z)
            if voxel in voxel_to_values:
                voxel_to_values[voxel].append(cell_value)
            else:
                voxel_to_values[voxel] = [cell_value]

            if verbose:
                print '\n(Triangle) cell number %s' % cell_id
                print 'centroid (%s, %s, %s)' % (x_center, y_center, z_center)
                print 'voxel (%s, %s, %s)' % (voxel[0], voxel[1], voxel[2])
                print '%s value = %s' % (array_name, cell_value)

        else:
            print ('\nOops, the cell number %s is not a vtkTriangle but a %s! '
                   'It will be ignored.' % (cell_id, cell.__class__.__name__))

    print '%s voxels mapped from %s cells' % (len(voxel_to_values),
                                              poly.GetNumberOfCells())

    # Initialize a 3D array scaled like the original segmentation, which will
    # hold in each voxel the maximal value among the corresponding vertex
    # coordinates in the graph and 0 in all other (background) voxels:
    volume = np.zeros((scale_x, scale_y, scale_z), dtype=np.float32)

    # Write the array and (if logfilename is given) write the cases with
    # multiple values into a log file:
    if logfilename is not None:
        f = open(logfilename, 'w')
    for voxel in voxel_to_values:
        value_list = voxel_to_values[voxel]
        # take the maximal or mean value from the list (the same if there is
        # only one value):
        if mean:
            final_value = sum(value_list) / float(len(value_list))
        else:
            final_value = max(value_list)
        volume[voxel[0], voxel[1], voxel[2]] = final_value

        if (logfilename is not None) and (len(value_list) > 1):
            line = '%s\t%s\t%s\t' % (voxel[0], voxel[1], voxel[2])
            for value in value_list:
                line += '%s\t' % value
            line = line[0:-1] + '\n'
            f.write(line)
    if logfilename is not None:
        f.close()

    return volume


class TypesConverter(object):
    """
    A static class for converting types between different libraries: numpy, VTK
    and graph-tool.

    In general if types do not match exactly, data are upcasted.
    """

    @staticmethod
    def vtk_to_numpy(din):
        """
        From a vtkDataArray object returns an equivalent numpy data type.

        Args:
            din (vtk.vtkDataArray): input vtkDataArray object

        Returns:
            numpy type
        """

        # Check that a type object is passed
        if not isinstance(din, vtk.vtkDataArray):
            error_msg = 'vtkDataArray object required as input.' % din  # TODO ask Antonio why "% din" if there is no "%s" - TypeError: not all arguments converted during string formatting
            raise pexceptions.PySegInputError(
                expr='vtk_to_numpy (TypesConverter)', msg=error_msg
            )

        if isinstance(din, vtk.vtkBitArray):
            return np.bool
        elif (isinstance(din, vtk.vtkIntArray) or
                isinstance(din, vtk.vtkTypeInt32Array)):
            return np.int
        elif isinstance(din, vtk.vtkTypeInt8Array):
            return np.int8
        elif isinstance(din, vtk.vtkTypeInt16Array):
            return np.int16
        elif isinstance(din, vtk.vtkTypeInt64Array):
            return np.int64
        elif isinstance(din, vtk.vtkTypeUInt8Array):
            return np.uint8
        elif isinstance(din, vtk.vtkTypeUInt16Array):
            return np.uint16
        elif isinstance(din, vtk.vtkTypeUInt32Array):
            return np.uint32
        elif isinstance(din, vtk.vtkTypeUInt64Array):
            return np.uint64
        elif (isinstance(din, vtk.vtkFloatArray) or
                isinstance(din, vtk.vtkTypeFloat32Array)):
            return np.float32
        elif (isinstance(din, vtk.vtkDoubleArray) or
                isinstance(din, vtk.vtkTypeFloat64Array)):
            return np.float64
        else:
            error_msg = 'VTK type not identified.' % din  # TODO the same
            raise pexceptions.PySegInputError(
                expr='numpy_to_vtk_array (TypesConverter)', msg=error_msg
            )

    @staticmethod
    def gt_to_vtk(din):
        """
        From the graph-tool property value type creates an equivalent
        vtkDataArray object.

        Args:
            din (str): graph-tool property value type

        Returns:
            vtkDataArray object
        """

        # Check that a string object is passed
        if not isinstance(din, str):
            error_msg = 'str object required as input.' % din  # TODO the same
            raise pexceptions.PySegInputError(expr='gt_to_vtk (TypesConverter)',
                                              msg=error_msg)

        if (din == 'bool') or (din == 'vector<bool>'):
            return vtk.vtkIntArray()  # was vtk.vtkBitArray()
        elif (din == 'int16_t') or (din == 'vector<int16_t>'):
            return vtk.vtkTypeInt16Array()
        elif (din == 'int32_t') or (din == 'vector<int32_t>'):
            return vtk.vtkIntArray()
        elif (din == 'int64_t') or (din == 'vector<int64_t>'):
            return vtk.vtkTypeInt64Array()
        elif (din == 'double') or (din == 'vector<double>'):
            return vtk.vtkFloatArray()
        else:
            error_msg = 'Graph-tool alias not identified.' % din  # TODO the same
            raise pexceptions.PySegInputError(expr='gt_to_vtk (TypesConverter)',
                                              msg=error_msg)

    @staticmethod
    def gt_to_numpy(din):
        """
        From the graph-tool property value type return an equivalent numpy data
        type.

        Args:
            din (str): graph-tool property value type

        Returns:
            numpy type
        """

        # Check that a string object is passed
        if not isinstance(din, str):
            error_msg = 'str object required as input.' % din  # TODO the same
            raise pexceptions.PySegInputError(
                expr='gt_to_numpy (TypesConverter)', msg=error_msg
            )

        if (din == 'bool') or (din == 'vector<bool>'):
            return np.bool
        elif (din == 'int16_t') or (din == 'vector<int16_t>'):
            return np.int16
        elif (din == 'int32_t') or (din == 'vector<int32_t>'):
            return np.int32
        elif (din == 'int64_t') or (din == 'vector<int64_t>'):
            return np.int64
        elif (din == 'double') or (din == 'vector<double>'):
            return np.float
        else:
            error_msg = 'Graph-tool alias not identified.' % din  # TODO the same
            raise pexceptions.PySegInputError(
                expr='gt_to_numpy (TypesConverter)', msg=error_msg
            )
