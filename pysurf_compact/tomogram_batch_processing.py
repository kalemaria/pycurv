import numpy as np
from skimage.measure import label, regionprops
import gzip
from os import remove
from os.path import isfile

import pysurf_io as io


def split_segmentation(infile, lbl=1, close=True, close_cube_size=5, close_iter=1, min_region_size=100):
    """
    Splits the segmentation in connected regions with at least the given size (number of voxels).

    Args:
        infile (str): the segmentation input file in one of the formats: '.mrc', '.em' or '.vti'.
        lbl (int, optional) the label to be considered, 0 will be ignored, default 1
        close (boolean, optional): if True (default), closes small holes in the segmentation first
        close_cube_size (int, optional): if close is True, gives the size of the cube structuring element used for closing, default 5
        close_iter (int, optional): if close is True, gives the number of iterations the closing should be repeated, default 1
        min_region_size (int, optional): gives the minimal number of voxels a region has to have in order to be considered, default 100

    Returns:
        a list of regions, where each item is a binary ndarray with the same shape as the segmentation but contains one region

    """
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
