import os
import numpy as np
import pandas as pd
from mindboggle.shapes.surface_shapes import curvature
from mindboggle.mio.vtks import read_points, read_scalars

"""
A function and a script running Mindboggle, getting the curvatures from
VTK files and writing them to a CSV file. Requires Mindboggle Docker container
(then do not forget to set: DOCK=/home/jovyan/work) or Mindboggle Python package
installed on your system.


Author: Maria Salfer (Max Planck Institute for Biochemistry)
"""


def run_mindboggle_curvature(surface_file, neighborhood, method=0):
    # Process the arguments and make outputs names
    ccode_path = os.environ['vtk_cpp_tools']
    command = os.path.join(ccode_path, 'curvature', 'CurvatureMain')
    basename = os.path.splitext(os.path.basename(surface_file))[0]
    extended_basename = '{}.mindboggle_m{}_n{}'.format(
        basename, method, neighborhood)
    stem = os.path.join(os.getcwd(), extended_basename)
    mean_curv_file = '{}_mean_curv.vtk'.format(stem)
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
    verbose = False
    curvatures_file = '{}_curvatures.csv'.format(stem)

    # Run the method
    default_mean_curv_file, _, _, _, _ = curvature(
        command, method, arguments, surface_file, verbose)
    # rename mean curvature output file
    os.rename(default_mean_curv_file, mean_curv_file)
    # remove unneeded output file
    os.remove(os.path.join(os.getcwd(), 'output.nipype'))

    # Get the curvatures from VTK files
    # [[x1, y1, z1], [x2, y2, z2], ...]
    points = np.array(read_points(mean_curv_file))
    xyz = points.T  # transposed: [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
    mean_curv, _ = read_scalars(
        mean_curv_file, return_first=True, return_array=True)
    assert(xyz.shape[1] == mean_curv.size)
    print('number of points: {}'.format(mean_curv.size))
    if method == 0 or method == 1:
        gauss_curv, _ = read_scalars(
            gauss_curv_file, return_first=True, return_array=True)
    if method == 0:
        max_curv, _ = read_scalars(
            max_curv_file, return_first=True, return_array=True)
        min_curv, _ = read_scalars(
            min_curv_file, return_first=True, return_array=True)

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
    df.to_csv(curvatures_file, sep=';')


if __name__ == '__main__':
    vtk_test_surfaces_fold = "/home/jovyan/work/vtk_test_surfaces"
    test_mindboggle_output_fold = "/home/jovyan/work/test_mindboggle_output"
    m = 0
    surfaces = ['smooth_torus_rr25_csr10.surface.vtk',
                'smooth_cylinder_r10_h25.surface.vtk',
                'smooth_sphere_r10.surface.vtk',
                'noisy_sphere_r10.surface.vtk',
                'noisy_torus_rr25_csr10.surface.vtk',
                'noisy_cylinder_r10_h25.surface.vtk']
    subfolds = ['smooth_torus', 'smooth_cylinder', 'smooth_sphere',
                'noisy_sphere', 'noisy_torus', 'noisy_cylinder']
    ns = list(range(11, 16, 1))  # range(2, 21, 2)
    for surface, subfold in zip(surfaces, subfolds):
        surface_file = os.path.join(vtk_test_surfaces_fold, surface)
        fold = os.path.join(test_mindboggle_output_fold, subfold)
        if not os.path.exists(fold):
            os.makedirs(fold)
        os.chdir(fold)
        for n in ns:
            run_mindboggle_curvature(surface_file, neighborhood=n, method=m)


