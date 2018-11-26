from pathlib2 import Path, PurePath
from curvature_calculation import (new_workflow, calculate_PM_curvatures,
                                   extract_curvatures_after_new_workflow)
from distances_calculation import (distances_and_thicknesses_calculation,
                                   extract_distances)
RADIUS_HIT = 10


def task_calculate_cER_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/"
    pixel_size = 1.368
    radius_hit = RADIUS_HIT
    methods = ["VV"]
    lbl = 2  # cER
    holes = 3
    min_component = 100

    for condition in ["TCB", "WT", "IST2", "SCS", "dTCB1", "dTCB2", "dTCB3"]:
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
                target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                    subfold, base_filename, radius_hit)
                yield {'name': base_filename,
                       # 'verbosity': 2,
                       'actions': [
                           (new_workflow,
                            [subfold, base_filename, pixel_size, radius_hit], {
                                'methods': methods,
                                'seg_file': seg_filename,
                                'label': lbl,
                                'holes': holes,
                                'min_component': min_component,
                                'cores': 4
                            })
                        ],
                       'file_dep': [seg_file],
                       'targets': [
                           "{}.gt".format(target_base),
                           "{}.vtp".format(target_base)
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
    methods = ["VV"]

    for condition in ["TCB", "WT", "IST2", "SCS", "dTCB1", "dTCB2", "dTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold_name = subfold_p.name
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}.cER".format(
                condition, date, lamella, tomo)
            subfold = str(subfold_p) + '/'
            target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                subfold, base_filename, radius_hit)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, radius_hit], {
                            'methods': methods,
                            'exclude_borders': 0
                        }),
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, radius_hit], {
                            'methods': methods,
                            'exclude_borders': 1
                        })
                    ],
                   'file_dep': [
                       "{}.gt".format(target_base),
                       "{}.vtp".format(target_base)
                   ],
                   'targets': [
                       "{}.csv".format(target_base),
                       "{}_excluding1borders.csv".format(target_base)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }


def task_calculate_distances():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/"
    radius_hit = RADIUS_HIT

    for condition in ["TCB", "WT", "IST2", "SCS", "dTCB1", "dTCB2", "dTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold_name = subfold_p.name
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}".format(
                condition, date, lamella, tomo)
            subfold = str(subfold_p) + '/'
            segmentation_file_p = list(subfold_p.glob('*.mrc'))[0].name
            segmentation_file = str(segmentation_file_p)
            target_base = "{}{}".format(subfold, base_filename)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (distances_and_thicknesses_calculation,
                        [subfold, segmentation_file, base_filename], {
                            'radius_hit': RADIUS_HIT
                        })
                    ],
                   'targets': [
                       "{}.PM.NVV_rh{}.gt".format(target_base, radius_hit),
                       "{}.cER.distancesFromPM.csv".format(target_base),
                       "{}.innercER.thicknesses.csv".format(target_base)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }


def task_extract_distances_without_borders():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/"

    for condition in ["TCB", "WT", "IST2", "SCS", "dTCB1", "dTCB2", "dTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold_name = subfold_p.name
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}".format(
                condition, date, lamella, tomo)
            distances_suffix = ".cER.distancesFromPM"
            thicknesses_suffix = ".innercER.thicknesses"
            subfold = str(subfold_p) + '/'
            target_base = "{}{}".format(
                subfold, base_filename)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (extract_distances,
                        [subfold, base_filename + distances_suffix], {
                            'name': 'PMdistance',
                            'exclude_borders': 1
                        }),
                       (extract_distances,
                        [subfold, base_filename + thicknesses_suffix], {
                            'name': 'cERthickness',
                            'exclude_borders': 1
                        })
                    ],
                   'file_dep': [
                       "{}.gt".format(target_base + distances_suffix),
                       "{}.gt".format(target_base + thicknesses_suffix)
                   ],
                   'targets': [
                       "{}_excluding1borders.csv".format(
                           target_base + distances_suffix),
                       "{}_excluding1borders.csv".format(
                           target_base + thicknesses_suffix)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }


def task_calculate_PM_curvatures():
    """
    Using lower surface and normals calculated by task_calculate_distances.
    Returns:
        None
    """
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/"
    radius_hit = RADIUS_HIT

    for condition in ["TCB", "WT", "IST2", "SCS", "dTCB1", "dTCB2", "dTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold_name = subfold_p.name
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}.PM".format(
                condition, date, lamella, tomo)
            subfold = str(subfold_p) + '/'
            gt_file_normals = "{}{}.NVV_rh{}.gt".format(
                subfold, base_filename, radius_hit)
            target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                subfold, base_filename, radius_hit)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (calculate_PM_curvatures,
                        [subfold, base_filename, radius_hit], {
                            'cores': 4
                        })
                    ],
                   'file_dep': [gt_file_normals],
                   'targets': [
                       "{}.gt".format(target_base),
                       "{}.vtp".format(target_base)
                   ],
                   # force doit to always mark the task as up-to-date
                   # (unless target removed)
                   'uptodate': [True]
                   }


def task_extract_PM_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/"
    radius_hit = RADIUS_HIT
    methods = ["VV"]

    for condition in ["TCB", "WT", "IST2", "SCS", "dTCB1", "dTCB2", "dTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold_name = subfold_p.name
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}.PM".format(
                condition, date, lamella, tomo)
            subfold = str(subfold_p) + '/'
            target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                subfold, base_filename, radius_hit)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, radius_hit], {
                            'methods': methods,
                            'exclude_borders': 0
                        }),
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, radius_hit], {
                            'methods': methods,
                            'exclude_borders': 1
                        })
                    ],
                   'file_dep': [
                       "{}.gt".format(target_base),
                       "{}.vtp".format(target_base)
                   ],
                   'targets': [
                       "{}.csv".format(target_base),
                       "{}_excluding1borders.csv".format(target_base)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }

# Note: to run one condition only, e.g. TCB: doit *:TCB*
