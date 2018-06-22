from pathlib2 import Path, PurePath
from curvature_calculation import (new_workflow,
                                   extract_curvatures_after_new_workflow)
RADIUS_HIT = 10


def task_calculate_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/"
    pixel_size = 1.368
    radius_hit = RADIUS_HIT
    methods = ["VV"]
    lbl = 2  # cER
    holes = 3
    min_component = 100

    for condition in ["TCB", "SCS", "WT", "IST2", "DTCB1", "DTCB2", "DTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold = str(subfold_p)
            seg_files = list(subfold_p.glob('**/*labels*.mrc'))
            if len(seg_files) > 0:
                seg_file_p = seg_files[0]
                seg_file = str(seg_file_p)
                seg_filename = str(PurePath(seg_file_p).name)
                tomo = "{}{}{}".format(condition, subfold.split('_')[-2],
                                       subfold.split('_')[-1])
                base_filename = "{}_cER".format(tomo)
                subfold += '/'
                target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                    subfold, base_filename, radius_hit)
                yield {'name': tomo,
                       # 'verbosity': 2,
                       'actions': [
                           (new_workflow,
                            [subfold, base_filename, pixel_size, radius_hit], {
                                'methods': methods,
                                'seg_file': seg_filename,
                                'label': lbl,
                                'holes': holes,
                                'remove_small_components': min_component
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


def task_extract_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/"
    pixel_size = 1.368
    radius_hit = RADIUS_HIT
    methods = ["VV"]

    for condition in ["TCB", "SCS", "WT", "IST2", "DTCB1", "DTCB2", "DTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold = str(subfold_p)
            tomo = "{}{}{}".format(condition, subfold.split('_')[-2],
                                   subfold.split('_')[-1])
            base_filename = "{}_cER".format(tomo)
            subfold += '/'
            target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                subfold, base_filename, radius_hit)
            yield {'name': tomo,
                   # 'verbosity': 2,
                   'actions': [
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, pixel_size, radius_hit], {
                            'methods': methods,
                            'exclude_borders': 0
                        }),
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, pixel_size, radius_hit], {
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
