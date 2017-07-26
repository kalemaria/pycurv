import pexceptions
import pysurf_io as io
from scipy import ndimage
import numpy as np
import vtk
import time


def close_holes(infile, cube_size, iterations, outfile):
    """
    Closes small holes in a binary volume by using a binary closing operation.

    Args:
        infile (str): input 'MRC' file with a binary volume.
        cube_size (str): size of the cube used for the binary closing operation (dilation followed by erosion).
        iterations (int): number of closing iterations
        outfile (str): output 'MRC' file with the closed volume

    """
    tomo = io.load_tomo(infile)
    data_type = tomo.dtype  # dtype('uint8')
    tomo_closed = ndimage.binary_closing(tomo, structure=np.ones((cube_size, cube_size, cube_size)), iterations=iterations).astype(data_type)
    io.save_numpy(tomo_closed, outfile)


# Runs pysurf_io.gen_surface function, which generates a VTK PolyData triangle surface of objects in a segmented volume file with a given label.
# Parameters:
#   infile: (string) The segmentation input file in one of the formats: '.mrc', '.em', '.vti' or '.fits' or
#           (numpy ndarray) 3D array containing the segmentation
#   outfile_base: (string) The path and filename without the ending for saving the surface (ending '.surface.vtp' will be added automatically).
#   lbl: (integer>=1, optional) The label to be considered, 0 will be ignored (default 1).
#   mask: (boolean, optional) If True (default), a mask of the binary objects is applied on the resulting surface to reduce artifacts.
#   save_infile_as_vti: (boolean, optional) If True (default False), the input is saved as a .vti file (as <outfile_base.vti>).
# Returns the surface.
def run_gen_surface(input, outfile_base, lbl=1, mask=True, save_input_as_vti=False, verbose=False):
    t_begin = time.time()

    # Generating the surface (vtkPolyData object)
    surface = io.gen_surface(input, lbl=lbl, mask=mask, verbose=verbose)

    # Filter out triangles with area=0, if any are present
    surface = __filter_null_triangles(surface)

    t_end = time.time()
    duration = t_end - t_begin
    print 'Surface generation took: %s min %s s' % divmod(duration, 60)

    # Writing the vtkPolyData surface into a VTP file
    io.save_vtp(surface, outfile_base + '.surface.vtp')
    print 'Surface was written to the file %s.vtp' % outfile_base

    if save_input_as_vti is True:
        # If input is a file name, read in the segmentation array from the file:
        if isinstance(input, str):
            input = io.load_tomo(input)
        elif not isinstance(input, np.ndarray):
            error_msg = 'Input must be either a file name or a ndarray.'
            raise pexceptions.PySegInputError(expr='run_gen_surface', msg=error_msg)

        # Save the segmentation as VTI for opening it in ParaView:
        io.save_numpy(input, outfile_base + '.vti')
        print 'Input was saved as the file %s.vti' % outfile_base

    return surface


# For a given VTK PolyData surface, filters out triangles with area=0, if any are present. Returns the filtered surface.
def __filter_null_triangles(surface, verbose=False):
    if isinstance(surface, vtk.vtkPolyData):

        # Check numbers of cells (polygons or triangles) (and all points).
        print 'The surface has %s cells' % surface.GetNumberOfCells()
        if verbose:
            print '%s points' % surface.GetNumberOfPoints()

        null_area_triangles = 0
        for i in range(surface.GetNumberOfCells()):
            # Get the cell i and check if it's a triangle:
            cell = surface.GetCell(i)
            if isinstance(cell, vtk.vtkTriangle):
                # Get the 3 points which made up the triangular cell i:
                points_cell = cell.GetPoints()

                # Calculate the area of the triangle i;
                area = cell.TriangleArea(points_cell.GetPoint(0), points_cell.GetPoint(1), points_cell.GetPoint(2))
                if area <= 0:
                    if verbose:
                        print 'Triangle %s is marked for deletion, because its area is not > 0' % i
                    surface.DeleteCell(i)
                    null_area_triangles += 1

            else:
                print 'Oops, the cell number %s is not a triangle!' % i

        if null_area_triangles:
            surface.RemoveDeletedCells()
            print '%s triangles with area = 0 were removed, resulting in:' % null_area_triangles
            # Recheck numbers of cells (polygons or triangles):
            print '%s cells' % surface.GetNumberOfCells()

    else:
        print 'Error: Wrong input data type, \'surface\' has to be a vtkPolyData object.'
        exit(1)

    return surface
