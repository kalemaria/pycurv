import sys
import os
import numpy as np
import pandas as pd
# from shutil import copyfile
from mindboggle.shapes.surface_shapes import curvature
from mindboggle.mio.vtks import read_points, read_scalars, rewrite_scalars

# Read in the arguments
surface_file = sys.argv[1]
neighborhood = sys.argv[2]
if len(sys.argv) > 3:
	method = int(sys.argv[3])
else:
	method = 0

# Process the arguments and make outputs names
ccode_path = os.environ['vtk_cpp_tools']
command = os.path.join(ccode_path, 'curvature', 'CurvatureMain')
basename = os.path.splitext(os.path.basename(surface_file))[0]
extended_basename = '{}.mindboggle_m{}_n{}'.format(basename, method, neighborhood)
stem = os.path.join(os.getcwd(), extended_basename)
mean_curv_file = '{}_mean_curv.vtk'.format(stem)
curvedness_file = '{}_curvedness.vtk'.format(stem)
# copyfile(surface_file, curvedness_file)
arguments = '-n {}'.format(neighborhood)
if method == 0 or method == 1:
	gauss_curv_file = '{}_gauss_curv.vtk'.format(stem)
	arguments = '{} -g {}'.format(arguments, gauss_curv_file)
if method == 0:
	max_curv_file = '{}_kappa1.vtk'.format(stem)
	min_curv_file = '{}_kappa2.vtk'.format(stem)
	min_dir_file = '{}_T2.txt'.format(stem)
	arguments = '{} -x {} -i {} -d {}'.format(
		arguments, max_curv_file, min_curv_file, min_dir_file)
verbose = True
curvatures_file = '{}_curvatures.csv'.format(stem)

# Run the method
default_mean_curv_file, _, _, _, _ = curvature(
	command, method, arguments, surface_file, verbose)
os.rename(default_mean_curv_file, mean_curv_file)  # rename mean curvature output file
os.remove(os.path.join(os.getcwd(), 'output.nipype'))  # remove unneeded output file

# Get the curvatures from VTK files
points = np.array(read_points(mean_curv_file))  # [[x1, y1, z1], [x2, y2, z2], ...]
xyz = points.T  # transposed: [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
mean_curv, _ = read_scalars(mean_curv_file, return_first=True, return_array=True)
# assert(xyz.shape[1] == mean_curv.size)
print('number of points: {}'.format(mean_curv.size))
if method == 0 or method == 1:
	gauss_curv, _ = read_scalars(gauss_curv_file, return_first=True, return_array=True)
if method == 0:
	max_curv, _ = read_scalars(max_curv_file, return_first=True, return_array=True)
	min_curv, _ = read_scalars(min_curv_file, return_first=True, return_array=True)
	curvedness = np.sqrt((max_curv ** 2 + min_curv ** 2) / 2)
	# assert(curvedness.size == max_curv.size)
	rewrite_scalars(surface_file, curvedness_file, curvedness, "curvedness")

# Write the curvatures to a CSV file
df = pd.DataFrame()
df['x'] = xyz[0]
df['y'] = xyz[1]
df['z'] = xyz[2]
df['mean_curvature'] = mean_curv
if method == 0 or method == 1:
	df['gauss_curvature'] = gauss_curv
if method == 0:
	df['kappa1'] = max_curv
	df['kappa2'] = min_curv
	df['curvedness'] = curvedness
df.to_csv(curvatures_file, sep=';')

# Example run:
#/home/jovyan/work/test_surfaces_mindboggle_output_n9# python ../scripts/run_mindboggle_curvature_command_line.py ../vtk_test_surfaces/torus_rr25_csr10.surface.vtk 9 

# Example command line run:
#/opt/vtk_cpp_tools/curvature/CurvatureMain -m 0 -n 2 -i torus_rr25_csr10.surface.mindboggle_n2_kappa2.vtk -x torus_rr25_csr10.surface.mindboggle_n2_kappa1.vtk -g torus_rr25_csr10.surface.mindboggle_n2_gauss_curv.vtk -d torus_rr25_csr10.surface.mindboggle_n2_T2.txt $DOCK/vtk_test_surfaces/torus_rr25_csr10.surface.vtk torus_rr25_csr10.surface.mindboggle_n2_mean_curvature.vtk

