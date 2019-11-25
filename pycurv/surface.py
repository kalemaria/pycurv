import vtk

from . import pexceptions
from . import pycurv_io as io
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from .linalg import dot_norm
import time

"""
Set of functions for generating a single-layer, signed surface from a
membrane segmentation and postprocessing the surface, using the VTK library.

Authors: Maria Kalemanov and Antonio Martinez-Sanchez (Max Planck Institute for
Biochemistry)
"""

__author__ = 'martinez and kalemanov'


# CONSTANTS
MAX_DIST_SURF = 3
"""int: a constant determining the maximal distance in pixels of a point on the
surface from the segmentation mask, used in gen_isosurface and gen_surface
functions.
"""

THRESH_SIGMA1 = 0.699471735
"""float: when convolving a binary mask with a gaussian kernel with sigma 1,
values at the boundary with 0's become this value
"""


def reverse_sense_and_normals(vtk_algorithm_output):
    """
    Sometimes the contouring algorithm can create a volume whose gradient
    vector and ordering of polygon (using the right hand rule) are
    inconsistent. vtkReverseSense cures this problem.

    Args:
        vtk_algorithm_output (vtkAlgorithmOutput): output of a VTK algorithm,
            to get with: algorithm_instance.GetOutputPort()

    Returns:
        surface with reversed normals (vtk.vtkPolyData)
    """

    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(vtk_algorithm_output)
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    return reverse.GetOutput()


def gen_surface(tomo, lbl=1, mask=True, other_mask=None, purge_ratio=1,
                field=False, mode_2d=False, verbose=False):
    """
    Generates a VTK PolyData surface from a segmented tomogram.

    Args:
        tomo (numpy.ndarray or str): the input segmentation as numpy ndarray or
            the file name in MRC, EM or VTI format
        lbl (int, optional): label for the foreground, default 1
        mask (boolean, optional): if True (default), the input segmentation is
            used as mask for the surface
        other_mask (numpy.ndarray, optional): if given (default None), this
            segmentation is used as mask for the surface
        purge_ratio (int, optional): if greater than 1 (default 1), then 1 every
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
    # Read in the segmentation (if file is given) and check format
    if isinstance(tomo, str):
        tomo = io.load_tomo(tomo)
    elif not isinstance(tomo, np.ndarray):
        raise pexceptions.PySegInputError(
            expr='gen_surface',
            msg='Input must be either a file name or a ndarray.')

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
        print('Cloud of points loaded...')

    # Creating the isosurface
    surf = vtk.vtkSurfaceReconstructionFilter()
    # surf.SetSampleSpacing(2)
    surf.SetSampleSpacing(purge_ratio)
    # surf.SetNeighborhoodSize(10)
    surf.SetInputData(cloud)

    contf = vtk.vtkContourFilter()
    contf.SetInputConnection(surf.GetOutputPort())
    contf.SetValue(0, 0)

    rsurf = reverse_sense_and_normals(contf.GetOutputPort())

    if verbose:
        print('Isosurfaces generated...')

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
        print('Rescaled and translated...')

    # Masking according to distance to the original segmentation
    if mask:
        if other_mask is None:
            tomod = distance_transform_edt(np.invert(tomo == lbl))
        elif isinstance(other_mask, np.ndarray):
            tomod = distance_transform_edt(np.invert(other_mask == lbl))
        else:
            raise pexceptions.PySegInputError(
                expr='gen_surface', msg='Other mask must be a ndarray.')

        for i in range(tsurf.GetNumberOfCells()):

            # Check if all points which made up the polygon are in the mask
            points_cell = tsurf.GetCell(i).GetPoints()
            count = 0
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                if (tomod[int(round(x)), int(round(y)), int(round(z))] >
                        MAX_DIST_SURF):
                    count += 1

            if count > 0:
                tsurf.DeleteCell(i)

        # Release free memory
        tsurf.RemoveDeletedCells()

        if verbose:
            print('Mask applied...')

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
                        dtype=io.TypesConverter().vtk_to_numpy(array))
        # for i in range(tsurf.GetNumberOfCells()):
        #     points_cell = tsurf.GetCell(i).GetPoints()
        #     for j in range(0, points_cell.GetNumberOfPoints()):
        #         x, y, z = points_cell.GetPoint(j)
        #         # print(x, y, z, array.GetTuple(j))
        #         x, y, z = int(round(x)), int(round(y)), int(round(z))
        #         tomo[x, y, z] = False
        #         tomon[x, y, z, :] = array.GetTuple(j)
        for i in range(tsurf.GetNumberOfCells()):
            points_cell = tsurf.GetCell(i).GetPoints()
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                # print(x, y, z, array.GetTuple(j))
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                if tomo[x, y, z] == lbl:
                    tomoh[x, y, z] = False
                    tomon[x, y, z, :] = array.GetTuple(i)

        # Distance transform
        tomod, ids = distance_transform_edt(tomoh, return_indices=True)

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
            print('Distance field generated...')

        return tsurf, tomod

    if verbose:
        print('Finished!')

    return tsurf


def gen_isosurface(tomo, lbl, grow=0, sg=0, thr=1.0, mask=None):
    """
    Generates a isosurface using the Marching Cubes method.

    Args:
        tomo (str or numpy.ndarray): segmentation input file in one of the
            formats: '.mrc', '.em' or '.vti', or 3D array containing the
            segmentation
        lbl (int): the label to be considered (> 0)
        grow (int, optional): if > 0 the surface is grown by so many voxels
            (default 0 - no growing)
        sg (int, optional): sigma for gaussian smoothing in voxels (default 0 -
            no smoothing)
        thr (optional, float): thr for isosurface (default 1.0)
        mask (int or numpy.ndarray, optional): if given (default None), the
            surface will be masked with it: if integer, this label is extracted
            from the input segmentation to generate the binary mask, otherwise
            it has to be given as a numpy.ndarray with same dimensions as the
            input segmentation

    Returns:
        a surface (vtk.vtkPolyData)
    """
    # Read in the segmentation (if file is given) and check format
    if isinstance(tomo, str):
        tomo = io.load_tomo(tomo)
    elif not isinstance(tomo, np.ndarray):
        raise pexceptions.PySegInputError(
            expr='gen_isosurface',
            msg='Input must be either a file name or a ndarray.')

    # Binarize the segmentation
    data_type = tomo.dtype
    binary_seg = tomo == lbl

    # Growing
    if grow > 0:
        binary_seg = binary_dilation(binary_seg, iterations=grow)
        # 3x3 structuring element with connectivity 1 is used by default

    binary_seg = binary_seg.astype(data_type)

    # Smoothing
    if sg > 0:
        binary_seg = gaussian_filter(binary_seg.astype(np.float), sg)

    # Generate isosurface
    smoothed_seg_vti = io.numpy_to_vti(binary_seg)
    surfaces = vtk.vtkMarchingCubes()
    surfaces.SetInputData(smoothed_seg_vti)
    surfaces.ComputeNormalsOn()
    surfaces.ComputeGradientsOn()
    surfaces.SetValue(0, thr)
    surfaces.Update()

    surf = reverse_sense_and_normals(surfaces.GetOutputPort())

    # Apply the mask
    if mask is not None:
        if isinstance(mask, int):  # mask is a label inside the segmentation
            mask = (tomo == mask).astype(data_type)
        elif not isinstance(mask, np.ndarray):
            raise pexceptions.PySegInputError(
                expr='gen_isosurface',
                msg='Input mask must be either an integer or a ndarray.')
        dist_from_mask = distance_transform_edt(mask == 0)
        for i in range(surf.GetNumberOfCells()):
            # Check if all points which made up the polygon are in the mask
            points_cell = surf.GetCell(i).GetPoints()
            count = 0
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                if (dist_from_mask[
                        int(round(x)), int(round(y)), int(round(z))] >
                        MAX_DIST_SURF):
                    count += 1
            # Mark cells that are not completely in the mask for deletion
            if count > 0:
                surf.DeleteCell(i)
        # Delete
        surf.RemoveDeletedCells()

    return surf


def run_gen_surface(tomo, outfile_base, lbl=1, mask=True, other_mask=None,
                    save_input_as_vti=False, verbose=False, isosurface=False,
                    grow=0, sg=0, thr=1.0):
    """
    Generates a VTK PolyData triangle surface for objects in a segmented volume
    with a given label.

    Removes triangles with zero area, if any are present, from the resulting
    surface.

    Args:
        tomo (str or numpy.ndarray): segmentation input file in one of the
            formats: '.mrc', '.em' or '.vti', or 3D array containing the
            segmentation
        outfile_base (str): the path and filename without the ending for saving
            the surface (ending '.surface.vtp' will be added automatically)
        lbl (int, optional): the label to be considered, 0 will be ignored,
            default 1
        mask (boolean, optional): if True (default), a mask of the binary
            objects is applied on the resulting surface to reduce artifacts
            (in case isosurface=False)
        other_mask (numpy.ndarray, optional): if given (default None), this
            segmentation is used as mask for the surface
        save_input_as_vti (boolean, optional): if True (default False), the
            input is saved as a '.vti' file ('<outfile_base>.vti')
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out
        isosurface (boolean, optional): if True (default False), generate
            isosurface (good for filled segmentations) - last three parameters
            are used in this case
        grow (int, optional): if > 0 the surface is grown by so many voxels
            (default 0 - no growing)
        sg (int, optional): sigma for gaussian smoothing in voxels (default 0 -
            no smoothing)
        thr (optional, float): thr for isosurface (default 1.0)

    Returns:
        the triangle surface (vtk.PolyData)
    """
    t_begin = time.time()

    # Generating the surface (vtkPolyData object)
    if isosurface:
        surface = gen_isosurface(tomo, lbl, grow, sg, thr, mask=other_mask)
    else:
        surface = gen_surface(tomo, lbl, mask, other_mask, verbose=verbose)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Surface generation took: {} min {} s'.format(minutes, seconds))

    # Writing the vtkPolyData surface into a VTP file
    io.save_vtp(surface, outfile_base + '.surface.vtp')
    print('Surface was written to the file {}.surface.vtp'.format(outfile_base))

    if save_input_as_vti is True:
        # If input is a file name, read in the segmentation array from the file:
        if isinstance(tomo, str):
            tomo = io.load_tomo(tomo)
        elif not isinstance(tomo, np.ndarray):
            raise pexceptions.PySegInputError(
                expr='run_gen_surface',
                msg='Input must be either a file name or a ndarray.')

        # Save the segmentation as VTI for opening it in ParaView:
        io.save_numpy(tomo, outfile_base + '.vti')
        print('Input was saved as the file {}.vti'.format(outfile_base))

    return surface


def add_curvature_to_vtk_surface(surface, curvature_type, invert=True):
    """
    Adds curvatures (Gaussian, mean, maximum or minimum) calculated by VTK to
    each triangle vertex of a vtkPolyData surface.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        curvature_type (str): type of curvature to add: 'Gaussian', 'Mean',
            'Maximum' or 'Minimum'
        invert (boolean, optional): if True (default), VTK will calculate
            curvatures as for meshes with opposite pointing normals (their
            convention is outwards pointing normals, opposite from ours)

    Returns:
        the vtkPolyData surface with '<type>_Curvature' property added to each
        triangle vertex
    """
    if isinstance(surface, vtk.vtkPolyData):
        curvature_filter = vtk.vtkCurvatures()
        curvature_filter.SetInputData(surface)
        if curvature_type == "Gaussian":
            curvature_filter.SetCurvatureTypeToGaussian()
        elif curvature_type == "Mean":
            curvature_filter.SetCurvatureTypeToMean()
        elif curvature_type == "Maximum":
            curvature_filter.SetCurvatureTypeToMaximum()
        elif curvature_type == "Minimum":
            curvature_filter.SetCurvatureTypeToMinimum()
        else:
            raise pexceptions.PySegInputError(
                expr='add_curvature_to_vtk_surface',
                msg=("One of the following strings required as the second "
                     "input: 'Gaussian', 'Mean', 'Maximum' or 'Minimum'."))
        if invert:
            curvature_filter.InvertMeanCurvatureOn()  # default Off
        curvature_filter.Update()
        surface_curvature = curvature_filter.GetOutput()
        return surface_curvature
    else:
        raise pexceptions.PySegInputError(
            expr='add_curvature_to_vtk_surface',
            msg="A vtkPolyData object required as the first input.")
    # How to get the curvatures later, e.g. for point with ID 0:
    # point_data = surface_curvature.GetPointData()
    # curvatures = point_data.GetArray(n)
    # where n = 2 for Gaussian, 3 for Mean, 4 for Maximum or Minimum
    # curvature_point0 = curvatures.GetTuple1(0)


def add_point_normals_to_vtk_surface(surface, reverse_normals=False):
    """
    Adds a normal to each triangle vertex of a vtkPolyData surface.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        reverse_normals (boolean, optional): if True (default False), VTK will
            flip the normals (their convention is outwards pointing normals,
            opposite from ours)

    Returns:
        the vtkPolyData surface with '<type>_Curvature' property added to each
        triangle vertex
    """
    if isinstance(surface, vtk.vtkPolyData):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface)
        normals.ComputePointNormalsOn()
        if reverse_normals:
            normals.FlipNormalsOn()
        else:
            normals.FlipNormalsOff()
        normals.Update()
        surface_normals = normals.GetOutput()
        return surface_normals
    else:
        raise pexceptions.PySegInputError(
            expr='add_point_normals_to_vtk_surface',
            msg="A vtkPolyData object required as the first input.")


def rescale_surface(surface, scale):
    """
    Rescales the given vtkPolyData surface with a given scaling factor in each
    of the three dimensions.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        scale (tuple): a scaling factor in 3D (x, y, z)

    Returns:
        rescaled surface (vtk.vtkPolyData)
    """
    try:
        assert isinstance(surface, vtk.vtkPolyData)
    except AssertionError:
        raise pexceptions.PySegInputError(
            expr='rescale_surface',
            msg="A vtkPolyData object required as the first input.")
    transf = vtk.vtkTransform()
    transf.Scale(scale[0], scale[1], scale[2])
    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetInputData(surface)
    tpd.SetTransform(transf)
    tpd.Update()
    scaled_surface = tpd.GetOutput()
    return scaled_surface
