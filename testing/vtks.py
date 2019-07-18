#!/usr/bin/env python
"""
Functions related to reading and writing VTK format files.

Authors:
    - Forrest Sheng Bao, 2012-2013  (forrest.bao@gmail.com)  http://fsbao.net
    - Arno Klein, 2012-2016  (arno@mindboggle.info)  http://binarybottle.com
    - Oliver Hinds, 2013 (ohinds@gmail.com)
    - Daniel Haehn, 2013 (daniel.haehn@childrens.harvard.edu)

Copyright 2016,  Mindboggle team (http://mindboggle.info), Apache v2.0 License

"""


def read_points(filename):
    """
    Load points of a VTK surface file.

    Not currently in use by Mindboggle.

    Parameters
    ----------
    filename : string
        path/filename of a VTK format file

    Returns
    -------
    points : list of lists of floats
        each element is a list of 3-D coordinates of a surface mesh vertex
    """
    import vtk

    Reader = vtk.vtkDataSetReader()
    Reader.SetFileName(filename)
    Reader.ReadAllScalarsOn()  # Activate the reading of all scalars
    Reader.Update()

    Data = Reader.GetOutput()
    points = [list(Data.GetPoint(point_id))
              for point_id in range(Data.GetNumberOfPoints())]

    return points


def read_scalars(filename, return_first=True, return_array=False):
    """
    Load all scalar lookup tables from a VTK file.

    Parameters
    ----------
    filename : string
        The path/filename of a VTK format file.
    return_first : bool
        Return only the first list of scalar values?
    return_array : bool (only if return_first)
        Return first list of scalars as a numpy array?

    Returns
    -------
    scalars : list of lists of integers or floats
        each element is a list of scalar values for the vertices of a mesh
    scalar_name(s) : list of strings
        each element is the name of a lookup table
    """
    import vtk
    if return_first and return_array:
        import numpy as np

    Reader = vtk.vtkDataSetReader()
    Reader.SetFileName(filename)
    Reader.ReadAllScalarsOn()  # Activate the reading of all scalars
    Reader.Update()
    Data = Reader.GetOutput()
    PointData = Data.GetPointData()

    scalars = []
    scalar_names = []
    if Reader.GetNumberOfScalarsInFile() > 0:
        for scalar_index in range(Reader.GetNumberOfScalarsInFile()):
            scalar_name = Reader.GetScalarsNameInFile(scalar_index)

            scalar_array = PointData.GetArray(scalar_name)
            scalar = [scalar_array.GetValue(i)
                      for i in range(scalar_array.GetDataSize())]
            scalars.append(scalar)
            scalar_names.append(scalar_name)
    else:
        # print("Scheisse! No scalars found!!! Reading 'curv' array anyway.")
        scalar_array = PointData.GetArray("curv")
        scalar = [scalar_array.GetValue(i)
                  for i in range(scalar_array.GetDataSize())]
        scalars.append(scalar)
        scalar_names.append("curv")

    if return_first:
        if scalars:
            scalars = scalars[0]
        if return_array:
            scalars = np.array(scalars)
        if scalar_names:
            scalar_names = scalar_names[0]
        else:
            scalar_names = ''

    return scalars, scalar_names
