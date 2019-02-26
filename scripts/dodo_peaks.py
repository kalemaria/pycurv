from pathlib2 import PurePath
from curvature_calculation import (new_workflow,
                                   extract_curvatures_after_new_workflow)

"""
Runs for curvature paper.
Estimates curvature for a cER peak with two surface generation methods, two
curvature estimation methods and several RadiusHit parameter values.
"""

# constant parameters for all tasks:
METHODS = ["VV", "SSVV"]
RADIUS_HIT = [2, 5, 10, 15, 20]
BASE_FOLD = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature/"
CONDITION = "TCB"
SUBFOLD = "180830_TITAN_l2_t2peak"
SUBSUBFOLDS = ["unfilled", "filled"]


def task_calculate_cER_curvatures():
    # constant parameters for all subtasks:
    seg_filename = "t2_ny01_lbl.labels_FILLEDpeak.mrc"
    pixel_size = 1.368
    lbl = 2  # cER
    filled_lbl = [None, 3]  # not using or using the filled cER lumen
    holes = 3
    min_component = 50

    for i, subsubfold_name in enumerate(SUBSUBFOLDS):
        subfold_p = PurePath(BASE_FOLD, CONDITION, SUBFOLD)
        subsubfold_p = PurePath(str(subfold_p), subsubfold_name)
        subsubfold = str(subsubfold_p) + '/'
        seg_file_p = PurePath(subsubfold, seg_filename)
        seg_file = str(seg_file_p)
        date, _, lamella, tomo = subfold_p.name.split('_')
        base_filename = "{}_{}_{}_{}.cER".format(CONDITION, date, lamella, tomo)
        for radius_hit in RADIUS_HIT:
            target_base = "{}{}.SSVV_rh{}".format(
                subsubfold, base_filename, radius_hit)
            yield {'name': "{}_rh{}_{}".format(base_filename, radius_hit,
                                               subsubfold_name),
                   # base_filename+"_rh"+str(radius_hit)+"_filled"+str(filled_lbl[i])
                   # 'verbosity': 2,
                   'actions': [
                       (new_workflow,
                        [subsubfold, base_filename, pixel_size, radius_hit], {
                            'methods': METHODS,
                            'seg_file': seg_filename,
                            'label': lbl,
                            'filled_label': filled_lbl[i],
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


def task_extract_cER_curvatures():
    for subsubfold_name in SUBSUBFOLDS:
        subfold_p = PurePath(BASE_FOLD, CONDITION, SUBFOLD)
        subsubfold_p = PurePath(str(subfold_p), subsubfold_name)
        subsubfold = str(subsubfold_p) + '/'
        date, _, lamella, tomo = subfold_p.name.split('_')
        base_filename = "{}_{}_{}_{}.cER".format(CONDITION, date, lamella, tomo)
        for radius_hit in RADIUS_HIT:
            target_base = "{}{}.SSVV_rh{}".format(
                subsubfold, base_filename, radius_hit)
            yield {'name': "{}_rh{}_{}".format(base_filename, radius_hit,
                                               subsubfold_name),
                   # 'verbosity': 2,
                   'actions': [
                       (extract_curvatures_after_new_workflow,
                        [subsubfold, base_filename, radius_hit], {
                            'methods': METHODS,
                            'exclude_borders': 0
                        }),
                       (extract_curvatures_after_new_workflow,
                        [subsubfold, base_filename, radius_hit], {
                            'methods': METHODS,
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
