from pathlib import Path, PurePath
from pycurv_scripts import (
    new_workflow, calculate_PM_curvatures,
    extract_curvatures_after_new_workflow)
from pycurv_scripts import (
    distances_and_thicknesses_calculation, extract_distances)

"""
Runs for MCS paper (Collado et al. 2019, Developmental Cell).
- Estimates curvature of a cER surface extracted using the compartment
segmentation with radius_hit=10 nm and AVV.
- Calculates distances between a PM surface extracted using the compartment
segmentation and the closer cER surface and thicknesses between the two cER
sides (using corrected normals going from the PM). The same for mitochondrion-ER
and vacuole-nucleus.
- Estimates curvature of the PM surface with radius_hit=10 nm and AVV.

Run all unfinished tasks with X (integer) cores from terminal with: `doit -n X`.
Run all unfinished tasks for one condition only, e.g. TCB: `doit *:TCB*`.
Run a specific task for a specific segmentation, e.g.:
calculate_cER_curvatures:TCB_170824_l1_t3*
"""

RADIUS_HIT = 10
CONDITIONS = ["WT", "IST2", "SCS", "TCB", "dTCB1", "dTCB2", "dTCB3",
              "dTCB1_dTCB2", "WT_HS", "dTCB123", "dTCB123_HS", "dtether",
              "SMP", "C2d", "TCB3FULL"]


def task_calculate_cER_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/MCS_data/cER_curvature/"
    radius_hit = RADIUS_HIT
    methods = ["VV"]
    lbl = 2  # cER
    filled_lbl = 3  # cER lumen
    holes = 3
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
                date, microscope, lamella, tomo = subfold_name.split('_')
                if microscope == "POLARA":
                    if (condition == "SMP" or condition == "C2d" or
                            condition == "TCB3FULL"):
                        pixel_size = 1.4036
                    else:
                        pixel_size = 2.088
                else:  # "dTCB123", "dTCB3" or "dTCB1"
                    pixel_size = 1.368  # "TITAN"
                base_filename = "{}_{}_{}_{}.cER".format(
                    condition, date, lamella, tomo)
                subfold = str(subfold_p) + '/'
                target_base = "{}{}.AVV_rh{}".format(
                    subfold, base_filename, radius_hit)
                yield {'name': base_filename,
                       # 'verbosity': 2,
                       'actions': [
                           (new_workflow,
                            [base_filename, seg_filename, subfold, pixel_size,
                             radius_hit], {
                                'methods': methods,
                                'label': lbl,
                                'filled_label': filled_lbl,
                                'holes': holes,
                                'min_component': min_component,
                                'cores': 6
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
    base_fold = "/fs/pool/pool-ruben/Maria/MCS_data/cER_curvature/"
    radius_hit = RADIUS_HIT
    methods = ["VV"]

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
            target_base = "{}{}.AVV_rh{}".format(
                subfold, base_filename, radius_hit)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
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


def task_calculate_PMcER_distances():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/MCS_data/PM-ER_distances/"
    radius_hit = RADIUS_HIT

    for condition in CONDITIONS:
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
            distances_suffix = ".cER.distancesFromPM"
            thicknesses_suffix = ".innercER.thicknesses"
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
                       "{}{}.gt".format(target_base, distances_suffix),
                       "{}{}.gt".format(target_base, thicknesses_suffix),
                       "{}{}.csv".format(target_base, distances_suffix),
                       "{}{}.csv".format(target_base, thicknesses_suffix)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }


def task_extract_distances_without_borders():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/MCS_data/PM-ER_distances/"

    for condition in CONDITIONS:
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
                       "{}{}.gt".format(target_base, distances_suffix),
                       "{}{}.gt".format(target_base, thicknesses_suffix)
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
    Using lower surface and normals calculated by task_calculate_PMcER_distances.
    Returns:
        None
    """
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/MCS_data/PM-ER_distances/"
    radius_hit = RADIUS_HIT

    for condition in CONDITIONS:
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
            target_base = "{}{}.AVV_rh{}".format(
                subfold, base_filename, radius_hit)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (calculate_PM_curvatures,
                        [subfold, base_filename, radius_hit], {
                            'cores': 6
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
    base_fold = "/fs/pool/pool-ruben/Maria/MCS_data/PM-ER_distances/"
    radius_hit = RADIUS_HIT
    methods = ["VV"]

    for condition in CONDITIONS:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold_name = subfold_p.name
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}.PM".format(
                condition, date, lamella, tomo)
            subfold = str(subfold_p) + '/'
            target_base = "{}{}.AVV_rh{}".format(
                subfold, base_filename, radius_hit)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
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


def task_calculate_MitoER_distances():
    # constant parameters for all conditions and segmentations:
    fold = "/fs/pool/pool-ruben/Maria/MCS_data/ER-Mito_distances/"
    radius_hit = RADIUS_HIT
    mem1 = "Mito"
    mem2 = "ER"

    fold_p = Path(fold)
    # iterate over all subfolders
    for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
        subfold_name = subfold_p.name
        date, _, lamella, tomo = subfold_name.split('_')
        base_filename = "{}_{}_{}".format(date, lamella, tomo)
        subfold = str(subfold_p) + '/'
        segmentation_file_p = list(subfold_p.glob('*.mrc'))[0].name
        segmentation_file = str(segmentation_file_p)
        target_base = "{}{}".format(subfold, base_filename)
        distances_suffix = ".{}.distancesFrom{}".format(mem2, mem1)
        thicknesses_suffix = ".inner{}.thicknesses".format(mem2)
        yield {'name': base_filename,
               # 'verbosity': 2,
               'actions': [
                   (distances_and_thicknesses_calculation,
                    [subfold, segmentation_file, base_filename], {
                        'radius_hit': RADIUS_HIT,
                        'mem1': mem1,
                        'mem2': mem2
                    })
                ],
               'targets': [
                   "{}.{}.NVV_rh{}.gt".format(target_base, mem1, radius_hit),
                   "{}{}.gt".format(target_base, distances_suffix),
                   "{}{}.gt".format(target_base, thicknesses_suffix),
                   "{}{}.csv".format(target_base, distances_suffix),
                   "{}{}.csv".format(target_base, thicknesses_suffix)
               ],
               # force doit to always mark the task as up-to-date (unless
               # target removed)
               'uptodate': [True]
               }


def task_extract_MitoER_distances_without_borders():
    # constant parameters for all conditions and segmentations:
    fold = "/fs/pool/pool-ruben/Maria/MCS_data/ER-Mito_distances/"
    mem1 = "Mito"
    mem2 = "ER"

    fold_p = Path(fold)
    # iterate over all subfolders
    for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
        subfold_name = subfold_p.name
        date, _, lamella, tomo = subfold_name.split('_')
        base_filename = "{}_{}_{}".format(date, lamella, tomo)
        distances_suffix = ".{}.distancesFrom{}".format(mem2, mem1)
        thicknesses_suffix = ".inner{}.thicknesses".format(mem2)
        subfold = str(subfold_p) + '/'
        target_base = "{}{}".format(subfold, base_filename)
        yield {'name': base_filename,
               # 'verbosity': 2,
               'actions': [
                   (extract_distances,
                    [subfold, base_filename + distances_suffix], {
                        'name': '{}distance'.format(mem1),
                        'exclude_borders': 1
                    }),
                   (extract_distances,
                    [subfold, base_filename + thicknesses_suffix], {
                        'name': '{}thickness'.format(mem2),
                        'exclude_borders': 1
                    })
               ],
               'file_dep': [
                   "{}{}.gt".format(target_base, distances_suffix),
                   "{}{}.gt".format(target_base, thicknesses_suffix)
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


def task_calculate_VacNuc_distances():
    # constant parameters for all conditions and segmentations:
    fold = "/fs/pool/pool-ruben/Maria/MCS_data/Vac-Nuc_distances/"
    radius_hit = RADIUS_HIT
    mem1 = "Vac"
    mem2 = "Nuc"

    fold_p = Path(fold)
    # iterate over all subfolders
    for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
        subfold_name = subfold_p.name
        date, _, lamella, tomo = subfold_name.split('_')
        base_filename = "{}_{}_{}".format(date, lamella, tomo)
        subfold = str(subfold_p) + '/'
        segmentation_file_p = list(subfold_p.glob('*.mrc'))[0].name
        segmentation_file = str(segmentation_file_p)
        target_base = "{}{}".format(subfold, base_filename)
        distances_suffix = ".{}.distancesFrom{}".format(mem2, mem1)
        yield {'name': base_filename,
               # 'verbosity': 2,
               'actions': [
                   (distances_and_thicknesses_calculation,
                    [subfold, segmentation_file, base_filename], {
                        'radius_hit': RADIUS_HIT,
                        'mem1': mem1,
                        'mem2': mem2,
                        'lbl_between_mem1_mem2': 3,
                        'maxthick': 0
                    })
                ],
               'targets': [
                   "{}.{}.NVV_rh{}.gt".format(target_base, mem1, radius_hit),
                   "{}{}.gt".format(target_base, distances_suffix),
                   "{}{}.csv".format(target_base, distances_suffix),
               ],
               # force doit to always mark the task as up-to-date (unless
               # target removed)
               'uptodate': [True]
               }


def task_extract_VacNuc_distances_without_borders():
    # constant parameters for all conditions and segmentations:
    fold = "/fs/pool/pool-ruben/Maria/MCS_data/Vac-Nuc_distances/"
    mem1 = "Vac"
    mem2 = "Nuc"

    fold_p = Path(fold)
    # iterate over all subfolders
    for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
        subfold_name = subfold_p.name
        date, _, lamella, tomo = subfold_name.split('_')
        base_filename = "{}_{}_{}".format(date, lamella, tomo)
        distances_suffix = ".{}.distancesFrom{}".format(mem2, mem1)
        subfold = str(subfold_p) + '/'
        target_base = "{}{}".format(subfold, base_filename)
        yield {'name': base_filename,
               # 'verbosity': 2,
               'actions': [
                   (extract_distances,
                    [subfold, base_filename + distances_suffix], {
                        'name': '{}distance'.format(mem1),
                        'exclude_borders': 1
                    }),
               ],
               'file_dep': [
                   "{}{}.gt".format(target_base, distances_suffix),
               ],
               'targets': [
                   "{}_excluding1borders.csv".format(
                       target_base + distances_suffix),
               ],
               # force doit to always mark the task as up-to-date (unless
               # target removed)
               'uptodate': [True]
               }

def task_calculate_MitoNuc_distances():
    # constant parameters for all conditions and segmentations:
    fold = "/fs/pool/pool-ruben/Maria/MCS_data/Mito-Nuc_distances/"
    radius_hit = RADIUS_HIT
    mem1 = "Mito"
    mem2 = "Nuc"
    lbl_between_mem1_mem2 = 3
    pixel_size = 1.368

    fold_p = Path(fold)
    # iterate over all subfolders
    for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
        subfold_name = subfold_p.name
        date, _, lamella, tomo = subfold_name.split('_')
        base_filename = "{}_{}_{}".format(date, lamella, tomo)
        subfold = str(subfold_p) + '/'
        segmentation_file_p = list(subfold_p.glob('*.mrc'))[0].name
        segmentation_file = str(segmentation_file_p)
        target_base = "{}{}".format(subfold, base_filename)
        distances_suffix = ".{}.distancesFrom{}".format(mem2, mem1)
        yield {'name': base_filename,
               # 'verbosity': 2,
               'actions': [
                   (distances_and_thicknesses_calculation,
                    [subfold, segmentation_file, base_filename], {
                        'radius_hit': RADIUS_HIT,
                        'mem1': mem1,
                        'mem2': mem2,
                        'lbl_between_mem1_mem2': lbl_between_mem1_mem2,
                        'pixel_size': pixel_size,
                        'maxthick': 0
                    })
                ],
               'targets': [
                   "{}.{}.NVV_rh{}.gt".format(target_base, mem1, radius_hit),
                   "{}{}.gt".format(target_base, distances_suffix),
                   "{}{}.csv".format(target_base, distances_suffix),
               ],
               # force doit to always mark the task as up-to-date (unless
               # target removed)
               'uptodate': [True]
               }


def task_extract_MitoNuc_distances_without_borders():
    # constant parameters for all conditions and segmentations:
    fold = "/fs/pool/pool-ruben/Maria/MCS_data/Mito-Nuc_distances/"
    mem1 = "Mito"
    mem2 = "Nuc"

    fold_p = Path(fold)
    # iterate over all subfolders
    for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
        subfold_name = subfold_p.name
        date, _, lamella, tomo = subfold_name.split('_')
        base_filename = "{}_{}_{}".format(date, lamella, tomo)
        distances_suffix = ".{}.distancesFrom{}".format(mem2, mem1)
        subfold = str(subfold_p) + '/'
        target_base = "{}{}".format(subfold, base_filename)
        yield {'name': base_filename,
               # 'verbosity': 2,
               'actions': [
                   (extract_distances,
                    [subfold, base_filename + distances_suffix], {
                        'name': '{}distance'.format(mem1),
                        'exclude_borders': 1
                    }),
               ],
               'file_dep': [
                   "{}{}.gt".format(target_base, distances_suffix),
               ],
               'targets': [
                   "{}_excluding1borders.csv".format(
                       target_base + distances_suffix),
               ],
               # force doit to always mark the task as up-to-date (unless
               # target removed)
               'uptodate': [True]
               }

