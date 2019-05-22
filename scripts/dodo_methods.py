from pathlib2 import Path, PurePath
from curvature_calculation import (new_workflow,
                                   extract_curvatures_after_new_workflow)

"""
Runs for the curvature methods paper.
"""

RADIUS_HIT = 10
CONDITIONS = ["TCB"]


def task_calculate_cER_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/"
    pixel_size = 1.368
    radius_hit = RADIUS_HIT
    lbl = 2  # cER
    filled_lbl = 3  # cER lumen
    holes = 0
    min_component = 100

    for condition in CONDITIONS:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            seg_files = list(subfold_p.glob('**/*.mrc'))
            if len(seg_files) > 0:
                seg_file_p = seg_files[0]
                seg_file = str(seg_file_p)
                seg_filename = str(PurePath(seg_file_p).name)
                subfold_name = subfold_p.name
                date, _, lamella, tomo = subfold_name.split('_')
                base_filename = "{}_{}_{}_{}.cER".format(
                    condition, date, lamella, tomo)
                subfold = str(subfold_p) + '/'
                yield {'name': base_filename,
                       # 'verbosity': 2,
                       'actions': [
                           (new_workflow,  # NVV
                            [subfold, base_filename, pixel_size, radius_hit], {
                                'methods': ['VV'],
                                'page_curvature_formula': True,
                                'area2': False,
                                'seg_file': seg_filename,
                                'label': lbl,
                                'filled_label': filled_lbl,
                                'holes': holes,
                                'min_component': min_component,
                                'cores': 4
                            }),
                           (new_workflow,  # RVV
                            [subfold, base_filename, pixel_size, radius_hit], {
                                'methods': ['VV'],
                                'page_curvature_formula': False,
                                'area2': False,
                                'seg_file': seg_filename,
                                'label': lbl,
                                'filled_label': filled_lbl,
                                'holes': holes,
                                'min_component': min_component,
                                'cores': 4
                            }),
                           (new_workflow,  # SSVV
                            [subfold, base_filename, pixel_size, radius_hit], {
                                'methods': ['SSVV'],
                                'seg_file': seg_filename,
                                'label': lbl,
                                'filled_label': filled_lbl,
                                'holes': holes,
                                'min_component': min_component,
                                'cores': 4
                            }),
                           (new_workflow,  # AVV
                            [subfold, base_filename, pixel_size, radius_hit], {
                                'methods': ['VV'],
                                'page_curvature_formula': False,
                                'area2': True,
                                'seg_file': seg_filename,
                                'label': lbl,
                                'filled_label': filled_lbl,
                                'holes': holes,
                                'min_component': min_component,
                                'cores': 4
                            })
                        ],
                       'file_dep': [seg_file],
                       'targets': [
                           "{}{}.NVV_rh{}.vtp".format(
                               subfold, base_filename, radius_hit),
                           "{}{}.RVV_rh{}.vtp".format(
                               subfold, base_filename, radius_hit),
                           "{}{}.SSVV_rh{}.vtp".format(
                               subfold, base_filename, radius_hit),
                           "{}{}.AVV_rh{}.vtp".format(
                               subfold, base_filename, radius_hit)
                       ],
                       # force doit to always mark the task as up-to-date
                       # (unless target removed)
                       'uptodate': [True]
                       }
            else:
                print("No segmentation file was found.")


def task_extract_cER_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/"
    radius_hit = RADIUS_HIT
    max_nm_borders = 7

    for condition in CONDITIONS:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold_name = subfold_p.name
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}.cER".format(
                condition, date, lamella, tomo)
            subfold = str(subfold_p) + '/'
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (extract_curvatures_after_new_workflow,  # NVV
                        [subfold, base_filename, radius_hit], {
                            'methods': ['VV'],
                            'page_curvature_formula': True,
                            'area2': False,
                            'exclude_borders': max_nm_borders
                        }),
                       (extract_curvatures_after_new_workflow,  # RVV
                        [subfold, base_filename, radius_hit], {
                            'methods': ['VV'],
                            'page_curvature_formula': False,
                            'area2': False,
                            'exclude_borders': max_nm_borders
                        }),
                       (extract_curvatures_after_new_workflow,  # SSVV
                        [subfold, base_filename, radius_hit], {
                            'methods': ['SSVV'],
                            'exclude_borders': max_nm_borders
                        }),
                       (extract_curvatures_after_new_workflow,  # AVV
                        [subfold, base_filename, radius_hit], {
                            'methods': ['VV'],
                            'page_curvature_formula': False,
                            'area2': True,
                            'exclude_borders': max_nm_borders
                        })
                    ],
                   'file_dep': [
                       "{}{}.NVV_rh{}.vtp".format(
                           subfold, base_filename, radius_hit),
                       "{}{}.RVV_rh{}.vtp".format(
                           subfold, base_filename, radius_hit),
                       "{}{}.SSVV_rh{}.vtp".format(
                           subfold, base_filename, radius_hit),
                       "{}{}.AVV_rh{}.vtp".format(
                           subfold, base_filename, radius_hit)
                   ],
                   'targets': [
                       "{}{}.NVV_rh{}_excluding{}borders.csv".format(
                           subfold, base_filename, radius_hit, max_nm_borders),
                       "{}{}.RVV_rh{}_excluding{}borders.csv".format(
                           subfold, base_filename, radius_hit, max_nm_borders),
                       "{}{}.SSVV_rh{}_excluding{}borders.csv".format(
                           subfold, base_filename, radius_hit, max_nm_borders),
                       "{}{}.AVV_rh{}_excluding{}borders.csv".format(
                           subfold, base_filename, radius_hit, max_nm_borders)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }
