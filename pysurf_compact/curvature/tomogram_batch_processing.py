import pysurf_io as io
import numpy as np
from skimage.measure import label, regionprops
import time
from run_gen_surface import run_gen_surface
from surface_graphs import TriangleGraph
from vector_voting import vector_voting
import gzip
from os import remove
from os.path import isfile
from graph_tool.all import load_graph


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


# Function for running all processing steps to estimate membrane curvature:
# 1. signed surface generation, 2. surface cleaning using a graph, 3. curvature calculation using a graph generated from the clean surface.
# fold          folder (string)
# tomo          tomogram (string)
# seg_file      membrane segmentation mask (string, may contain 'fold' and 'tomo')
# label         label to be considered in the membrane mask (int)
# pixel_size    pixel size in nm of the membrane mask (float)
# scale_x       size of the membrane mask in X dimension (int)
# scale_y       size of the membrane mask in Y dimension (int)
# scale_z       size of the membrane mask in Z dimension (int)
# k             parameter of Normal Vector Voting algorithm determining the neighborhood size, g_max = k * average weak triangle graph edge length (int)
def workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z, k):
    epsilon = 0
    eta = 0
    region_file_base = "%s%s_ERregion" % (fold, tomo)
    all_kappa_1_values = []
    all_kappa_2_values = []
    all_gauss_curvature_VV_values = []
    all_mean_curvature_VV_values = []
    all_shape_index_VV_values = []
    all_curvedness_VV_values = []
    region_surf_with_borders_files = []
    region_surf_VV_files = []
    all_file_base = "%s%s_ER" % (fold, tomo)
    all_kappa_1_file = "%s.VV_k%s_epsilon%s_eta%s.max_curvature.txt" % (all_file_base, k, epsilon, eta)
    all_kappa_2_file = "%s.VV_k%s_epsilon%s_eta%s.min_curvature.txt" % (all_file_base, k, epsilon, eta)
    all_gauss_curvature_VV_file = "%s.VV_k%s_epsilon%s_eta%s.gauss_curvature.txt" % (all_file_base, k, epsilon, eta)
    all_mean_curvature_VV_file = "%s.VV_k%s_epsilon%s_eta%s.mean_curvature.txt" % (all_file_base, k, epsilon, eta)
    all_shape_index_VV_file = "%s.VV_k%s_epsilon%s_eta%s.shape_index.txt" % (all_file_base, k, epsilon, eta)
    all_curvedness_VV_file = "%s.VV_k%s_epsilon%s_eta%s.curvedness.txt" % (all_file_base, k, epsilon, eta)
    all_surf_with_borders_vtp_file = '%s.cleaned_surface_with_borders_nm.vtp' % all_file_base
    all_surf_VV_vtp_file = '%s.cleaned_surface_nm.VV_k%s_epsilon%s_eta%s.vtp' % (all_file_base, k, epsilon, eta)

    # Split the segmentation into regions:
    regions, mask_file = split_segmentation(seg_file, lbl=label, close=True, close_cube_size=3, close_iter=1, min_region_size=100)

    for i, region in enumerate(regions):
        print "\n\nRegion %s" % (i + 1)

        # ***Part 1: surface generation***
        surf_file_base = "%s%s" % (region_file_base, i + 1)
        surf_file = surf_file_base + '.surface.vtp'  # region surface, output of run_gen_surface
        surf = None
        if not isfile(surf_file):
            print "\nGenerating a surface..."
            surf = run_gen_surface(region, surf_file_base, lbl=label, save_input_as_vti=True)

        # ***Part 2: surface cleaning***
        scale_factor_to_nm = pixel_size
        cleaned_scaled_surf_file = surf_file_base + '.cleaned_surface_with_borders_nm.vtp'
        cleaned_scaled_graph_file = surf_file_base + '.cleaned_triangle_graph_with_borders_nm.gt'

        if not isfile(cleaned_scaled_surf_file):  # the cleaned scaled surface .vtp file does not exist yet
            # if no cleaned scaled surface .vtp file was generated, then also no cleaned scaled graph .gt file was written
            # -> have to generate the scaled graph from the original surface and clean it
            if surf is None:  # this is the case if surface was generated in an earlier run
                surf = io.load_poly(surf_file)
                print 'A surface was loaded from the file %s' % surf_file

            print '\nBuilding the TriangleGraph from the vtkPolyData surface with curvatures...'
            tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
            scaled_surf = tg.build_graph_from_vtk_surface(surf, verbose=False)
            print 'The graph has %s vertices and %s edges' % (tg.graph.num_vertices(), tg.graph.num_edges())

            io.save_vtp(scaled_surf, surf_file[0:-4] + "_nm.vtp")
            print 'The surface scaled to nm was written into the file %s_nm.vtp' % surf_file[0:-4]

            print '\nFinding triangles that are 3 pixels to surface borders...'
            tg.find_vertices_near_border_fast(3 * scale_factor_to_nm, purge=True, verbose=False)
            print 'The graph has %s vertices and %s edges' % (tg.graph.num_vertices(), tg.graph.num_edges())

            print '\nFinding small connected components of the graph...'
            tg.find_small_connected_components(threshold=100, purge=True)

            if tg.graph.num_vertices() > 0:
                print 'The graph has %s vertices and %s edges and following properties' % (tg.graph.num_vertices(), tg.graph.num_edges())
                tg.graph.list_properties()
                tg.graph.save(cleaned_scaled_graph_file)
                print 'Cleaned and scaled graph with outer borders was written into the file %s' % cleaned_scaled_graph_file

                poly_triangles_filtered_with_borders = tg.graph_to_triangle_poly()
                io.save_vtp(poly_triangles_filtered_with_borders, cleaned_scaled_surf_file)
                print 'Cleaned and scaled surface with outer borders was written into the file %s' % cleaned_scaled_surf_file
                calculate_curvature = True
            else:
                print "Region %s was completely filtered out and will be omitted." % (i + 1)
                calculate_curvature = False

        else:  # this is the case if graph generation and cleaning was done in an earlier run and the cleaned scaled surface .vtp file exists
            calculate_curvature = True  # the graph has vertices for sure if the .vtp file was written

            if not isfile(cleaned_scaled_graph_file):  # the graph was not saved and has to be reversed-engineered from the surface
                surf = io.load_poly(cleaned_scaled_surf_file)
                print 'The cleaned and scaled surface with outer borders was loaded from the file %s' % cleaned_scaled_surf_file

                print '\nBuilding the triangle graph from the surface...'
                scale_factor_to_nm = 1  # because the surface is already scaled in nm
                tg = TriangleGraph(scale_factor_to_nm, scale_x, scale_y, scale_z)
                tg.build_graph_from_vtk_surface(surf, verbose=False)
                print 'The graph has %s vertices and %s edges' % (tg.graph.num_vertices(), tg.graph.num_edges())
                tg.graph.list_properties()
                tg.graph.save(cleaned_scaled_graph_file)
                print 'Cleaned and scaled graph with outer borders was written into the file %s' % cleaned_scaled_graph_file

            else:  # cleaned scaled graph can just be loaded from the found .gt file
                tg = TriangleGraph(1, scale_x, scale_y, scale_z)
                tg.graph = load_graph(cleaned_scaled_graph_file)
                print 'Cleaned and scaled graph with outer borders was loaded from the file %s' % cleaned_scaled_graph_file
                print 'The graph has %s vertices and %s edges' % (tg.graph.num_vertices(), tg.graph.num_edges())

        # ***Part 3: curvature calculation***
        if calculate_curvature:
            # Running the modified Vector Voting algorithm and saving the output:
            cleaned_scaled_graph_VV_file = surf_file_base + '.cleaned_triangle_graph_nm.VV_k%s_epsilon%s_eta%s.gt' % (k, epsilon, eta)
            cleaned_scaled_surf_VV_file = surf_file_base + '.cleaned_surface_nm.VV_k%s_epsilon%s_eta%s.vtp' % (k, epsilon, eta)
            kappa_1_file = "%s%s.VV_k%s_epsilon%s_eta%s.max_curvature.txt" % (region_file_base, i + 1, k, epsilon, eta)
            kappa_2_file = "%s%s.VV_k%s_epsilon%s_eta%s.min_curvature.txt" % (region_file_base, i + 1, k, epsilon, eta)
            gauss_curvature_VV_file = "%s%s.VV_k%s_epsilon%s_eta%s.gauss_curvature.txt" % (region_file_base, i + 1, k, epsilon, eta)
            mean_curvature_VV_file = "%s%s.VV_k%s_epsilon%s_eta%s.mean_curvature.txt" % (region_file_base, i + 1, k, epsilon, eta)
            shape_index_VV_file = "%s%s.VV_k%s_epsilon%s_eta%s.shape_index.txt" % (region_file_base, i + 1, k, epsilon, eta)
            curvedness_VV_file = "%s%s.VV_k%s_epsilon%s_eta%s.curvedness.txt" % (region_file_base, i + 1, k, epsilon, eta)

            surf_VV, tg_VV = vector_voting(tg, k, epsilon=epsilon, eta=eta, all_vertices=True, exclude_borders=True)  # does not calculate curvatures for triangles at borders
                                                                                                                      # and removes them in the end
            print 'The graph without outer borders and with VV curvatures has %s vertices and %s edges' % (tg.graph.num_vertices(), tg.graph.num_edges())
            tg_VV.graph.list_properties()
            tg.graph.save(cleaned_scaled_graph_VV_file)
            print 'The graph without outer borders and with VV curvatures was written into the file %s' % cleaned_scaled_graph_VV_file

            io.save_vtp(surf_VV, cleaned_scaled_surf_VV_file)
            print 'The surface without outer borders and with VV curvatures was written into the file %s' % cleaned_scaled_surf_VV_file

            # Making a list of all the region .vtp files
            region_surf_with_borders_files.append(cleaned_scaled_surf_file)
            region_surf_VV_files.append(cleaned_scaled_surf_VV_file)

            # Getting the VV curvatures from the output graph (without outer borders), and merging the respective values for all regions:
            kappa_1_values = tg_VV.get_vertex_property_array("kappa_1")
            all_kappa_1_values.extend(kappa_1_values.tolist())
            kappa_2_values = tg_VV.get_vertex_property_array("kappa_2")
            all_kappa_2_values.extend(kappa_2_values.tolist())
            gauss_curvature_VV_values = tg_VV.get_vertex_property_array("gauss_curvature_VV")
            all_gauss_curvature_VV_values.extend(gauss_curvature_VV_values.tolist())
            mean_curvature_VV_values = tg_VV.get_vertex_property_array("mean_curvature_VV")
            all_mean_curvature_VV_values.extend(mean_curvature_VV_values.tolist())
            shape_index_VV_values = tg_VV.get_vertex_property_array("shape_index_VV")
            all_shape_index_VV_values.extend(shape_index_VV_values.tolist())
            curvedness_VV_values = tg_VV.get_vertex_property_array("curvedness_VV")
            all_curvedness_VV_values.extend(curvedness_VV_values.tolist())

            # Writing all the region curvature values into files:
            io.write_values_to_file(kappa_1_values, kappa_1_file)
            io.write_values_to_file(kappa_2_values, kappa_2_file)
            io.write_values_to_file(gauss_curvature_VV_values, gauss_curvature_VV_file)
            io.write_values_to_file(mean_curvature_VV_values, mean_curvature_VV_file)
            io.write_values_to_file(shape_index_VV_values, shape_index_VV_file)
            io.write_values_to_file(curvedness_VV_values, curvedness_VV_file)
            print 'All the curvature values for the region were written into files'

    # Writing all the joint curvature values into files:
    io.write_values_to_file(all_kappa_1_values, all_kappa_1_file)
    io.write_values_to_file(all_kappa_2_values, all_kappa_2_file)
    io.write_values_to_file(all_gauss_curvature_VV_values, all_gauss_curvature_VV_file)
    io.write_values_to_file(all_mean_curvature_VV_values, all_mean_curvature_VV_file)
    io.write_values_to_file(all_shape_index_VV_values, all_shape_index_VV_file)
    io.write_values_to_file(all_curvedness_VV_values, all_curvedness_VV_file)
    print 'All the curvature values for the whole tomogram were written into files'

    # Merging all region '.vtp' files (once with outer borders before VV, if it has not been done yet, and once after VV):
    if not isfile(all_surf_with_borders_vtp_file):
        io.merge_vtp_files(region_surf_with_borders_files, all_surf_with_borders_vtp_file)
        print "Done merging all the found region cleaned and scaled surface with outer borders '.vtp' files into the file %s" % all_surf_with_borders_vtp_file
    io.merge_vtp_files(region_surf_VV_files, all_surf_VV_vtp_file)
    print "Done merging all the found region cleaned and scaled surface with curvatures '.vtp' files into the file %s" % all_surf_VV_vtp_file

    # Converting the '.vtp' tomogram files to '.stl' files:
    all_surf_with_borders_stl_file = all_surf_with_borders_vtp_file[0:-4] + '.stl'
    if not isfile(all_surf_with_borders_stl_file):
        io.vtp_file_to_stl_file(all_surf_with_borders_vtp_file, all_surf_with_borders_stl_file)
        print "The '.vtp' file %s was converted to .stl format" % all_surf_with_borders_vtp_file
    all_surf_VV_stl_file = all_surf_VV_vtp_file[0:-4] + '.stl'
    io.vtp_file_to_stl_file(all_surf_VV_vtp_file, all_surf_VV_stl_file)
    print "The '.vtp' file %s was converted to .stl format" % all_surf_VV_vtp_file

    # Converting vtkPolyData selected cell arrays from the '.vtp' file to 3-D volumes and saving them as '.mrc.gz' files.
    vtp_arrays_to_mrc_volumes(all_file_base, all_surf_VV_vtp_file, pixel_size, scale_x, scale_y, scale_z, k, epsilon, eta, log_files=True)  # maximal voxel value & .log files
    vtp_arrays_to_mrc_volumes(all_file_base, all_surf_VV_vtp_file, pixel_size, scale_x, scale_y, scale_z, k, epsilon, eta, mean=True)  # mean voxel value & no .log files


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


def main():
    t_begin = time.time()

    # TODO change those parameters for each tomogram & label:
    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/new_workflow/diffuseHtt97Q/"
    tomo = "t112"
    seg_file = "%s%s_final_ER1_vesicles2_notER3_NE4.Labels.mrc" % (fold, tomo)
    label = 1
    pixel_size = 2.526
    scale_x = 620
    scale_y = 620
    scale_z = 80
    k = 3

    workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z, k)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nTotal elapsed time: %s min %s s' % divmod(duration, 60)


if __name__ == "__main__":
    main()
