import pysurf_io as io
import numpy as np
from skimage.measure import label, regionprops  # TODO namespace!?
import time
import gzip
from os import remove  # TODO namespace!?
from os.path import isfile  # TODO namespace!?
from graph_tool.all import load_graph  # TODO namespace!?


# Splits the segmentation in connected regions with at least the given size (number of voxels).
# Parameters:
#   infile: (string) The segmentation input file in one of the formats: '.mrc', '.em', '.vti' or '.fits'.
#   lbl: (integer>=1, optional) The label to be considered, 0 will be ignored (default 1).
#   close: (boolean, optional) If True, closes small holes in the segmentation first (default True).
#   close_cube_size: (integer, optional) If close is True, gives the size of the cube structuring element used for closing (default 5).
#   close_iter: (integer, optional) If close is True, gives the number of iterations the closing should be repeated (default 1).
#   min_region_size: (integer, optional) Gives the minimal number of voxels a region has to have to be considered (default 100).
# Returns:
#   regions: (a list of (N,3) ndarrays) Each item is a binary ndarray with the same shape as the segmentation but contains one region.
def split_segmentation(infile, lbl=1, close=True, close_cube_size=5, close_iter=1, min_region_size=100):
    # Load the segmentation numpy array from a file and get only the requested labels as 1 and the background as 0:
    seg = io.load_tomo(infile)
    assert(isinstance(seg, np.ndarray))
    data_type = seg.dtype
    binary_seg = (seg == lbl).astype(data_type)

    # If requested, close small holes in the segmentation:
    outfile = infile
    if close:
        outfile = "%s%s_closed_size%s_iter%s.mrc" % (infile[0:-4], lbl, close_cube_size, close_iter)
        if not isfile(outfile):
            from scipy import ndimage
            binary_seg = ndimage.binary_closing(binary_seg, structure=np.ones((close_cube_size, close_cube_size, close_cube_size)), iterations=close_iter).astype(data_type)
            # Write the closed binary segmentation into a file:
            io.save_numpy(binary_seg, outfile)
            print "Closed the binary segmentation and saved it into the file %s" % outfile
        else:  # the '.mrc' file already exists
            binary_seg = io.load_tomo(outfile)
            print "The closed binary segmentation was loaded from the file %s" % outfile

    # Label each connected region of the binary segmentation:
    label_seg = label(binary_seg)

    # Get only regions with at least the given size:
    regions = []
    for i, region in enumerate(regionprops(label_seg)):
        region_area = region.area
        if region_area >= min_region_size:
            print "%s. region has %s voxels and pass" % (i + 1, region_area)
            # Get the region coordinates and make an ndarray with same shape as the segmentation and 1 at those coordinates:
            region_ndarray = np.zeros(shape=tuple(seg.shape), dtype=data_type)
            region_coords = region.coords  # 2D array with 3 columns: x, y, z and number of rows corresponding to the number of voxels in the region
            for i in xrange(region_coords.shape[0]):  # iterate over the rows
                region_ndarray[region_coords[i, 0], region_coords[i, 1], region_coords[i, 2]] = 1
            regions.append(region_ndarray)
        else:
            print "%s. region has %s voxels and does NOT pass" % (i + 1, region_area)
    print "%s regions passed." % len(regions)
    return regions, outfile


# Converting vtkPolyData cell arrays from a '.vtp' file to 3-D volumes and saving them as '.mrc.gz' files.
# If mean=True (default False) in case multiple triangles map to the same voxel, takes the mean value, else the maximal value.
# If log_files=True (default False) writes the log files for such cases.
def vtp_arrays_to_mrc_volumes(all_file_base, all_surf_VV_vtp_file, pixel_size, scale_x, scale_y, scale_z, k, epsilon, eta, mean=False, log_files=False):
    array_name1 = "kappa_1"
    name1 = "max_curvature"
    array_name2 = "kappa_2"
    name2 = "min_curvature"
    array_name3 = "curvedness_VV"
    name3 = "curvedness"

    if mean == True:
        voxel_mean_str = ".voxel_mean"
    else:
        voxel_mean_str = ""
    mrcfilename1 = '%s.VV_k%s_epsilon%s_eta%s.%s.volume%s.mrc' % (all_file_base, k, epsilon, eta, name1, voxel_mean_str)
    mrcfilename2 = '%s.VV_k%s_epsilon%s_eta%s.%s.volume%s.mrc' % (all_file_base, k, epsilon, eta, name2, voxel_mean_str)
    mrcfilename3 = '%s.VV_k%s_epsilon%s_eta%s.%s.volume%s.mrc' % (all_file_base, k, epsilon, eta, name3, voxel_mean_str)
    if log_files:
        logfilename1 = mrcfilename1[0:-4] + '.log'
        logfilename2 = mrcfilename2[0:-4] + '.log'
        logfilename3 = mrcfilename3[0:-4] + '.log'
    else:
        logfilename1 = None
        logfilename2 = None
        logfilename3 = None

    # Load the vtkPolyData object from the '.vtp' file, calculate the volumes from arrays, write '.log' files, and save the volumes as '.mrc' files:
    poly = io.load_poly(all_surf_VV_vtp_file)
    volume1 = io.poly_array_to_volume(poly, array_name1, pixel_size, scale_x, scale_y, scale_z, logfilename=logfilename1, mean=mean)
    io.save_numpy(volume1, mrcfilename1)
    volume2 = io.poly_array_to_volume(poly, array_name2, pixel_size, scale_x, scale_y, scale_z, logfilename=logfilename2, mean=mean)
    io.save_numpy(volume2, mrcfilename2)
    volume3 = io.poly_array_to_volume(poly, array_name3, pixel_size, scale_x, scale_y, scale_z, logfilename=logfilename3, mean=mean)
    io.save_numpy(volume3, mrcfilename3)

    # Gunzip the '.mrc' files and delete the uncompressed files:
    for mrcfilename in [mrcfilename1, mrcfilename2, mrcfilename3]:
        with open(mrcfilename) as f_in, gzip.open(mrcfilename + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
        remove(mrcfilename)
        print 'Archive %s.gz was written' % mrcfilename