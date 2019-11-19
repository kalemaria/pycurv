import os
import numpy as np
import pandas as pd
# from mindboggle.mio.vtks import read_points, read_scalars
from scripts_running_mindboggle_and_freesurfer.vtks import read_points, read_scalars


def get_freesurfer_curvatures(mean_curv_file, max_curv_file, min_curv_file, curvatures_file):
    # Get the curvatures from VTK files
    points = np.array(read_points(mean_curv_file))  # [[x1, y1, z1], [x2, y2, z2], ...]
    xyz = points.T  # transposed: [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
    mean_curv, scalar_name = read_scalars(
        mean_curv_file, return_first=True, return_array=True)
    # print(scalar_name)
    try:
        assert(xyz.shape[1] == mean_curv.size)
    except AssertionError:
        print("number of points={} is not equal to number of scalars={}".format(
            xyz.shape[1], mean_curv.size))
    # print('number of points: {}'.format(mean_curv.size))
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
    df['kappa1'] = max_curv
    df['kappa2'] = min_curv
    df.to_csv(curvatures_file, sep=';')


if __name__ == '__main__':
    test_freesurfer_output_fold = (
        "/fs/pool/pool-ruben/Maria/workspace/github/"
        "my_tests_output/comparison_to_others/test_freesurfer_output")
    subfolds = ['smooth_sphere', 'noisy_sphere',
                'smooth_sphere', 'noisy_sphere',
                'smooth_torus', 'noisy_torus',
                'smooth_cylinder', 'noisy_cylinder'
                ]
    output_templates = [
        'smooth_sphere_r10.CURV.vtk', 'noisy_sphere_r10.CURV.vtk',
        'smooth_sphere_r20.CURV.vtk', 'noisy_sphere_r30.CURV.vtk',
        'smooth_torus_rr25_csr10.CURV.vtk', 'noisy_torus_rr25_csr10.CURV.vtk',
        'smooth_cylinder_r10_h25.CURV.vtk', 'noisy_cylinder_r10_h25.CURV.vtk'
        ]
    for subfold, output_template in zip(subfolds, output_templates):
        vtk_fold = os.path.join(test_freesurfer_output_fold, subfold, "surf")
        csv_fold = os.path.join(test_freesurfer_output_fold, subfold, "csv")
        if not os.path.exists(csv_fold):
            os.makedirs(csv_fold)
        mean_curv_file = os.path.join(vtk_fold, output_template.replace(
            "CURV", "H"))
        max_curv_file = os.path.join(vtk_fold, output_template.replace(
            "CURV", "K1"))
        min_curv_file = os.path.join(vtk_fold, output_template.replace(
            "CURV", "K2"))
        curvatures_file = os.path.join(csv_fold, output_template.replace(
            "CURV.vtk", "freesurfer_curvatures.csv"))
        get_freesurfer_curvatures(
            mean_curv_file, max_curv_file, min_curv_file, curvatures_file)
