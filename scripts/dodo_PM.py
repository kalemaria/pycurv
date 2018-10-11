from pathlib2 import Path, PurePath
from curvature_calculation import (new_workflow,
                                   extract_curvatures_after_new_workflow)
from scripts import run_calculate_distances_and_thicknesses


def task_correct_normals():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/"
    pixel_size = 1.368
    radius_hit = 10
    methods = ["VV"]
    lbl = 1  # PM
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
                base_filename = "{}_PM".format(tomo)
                subfold += '/'
                PM_graph_file = "{}{}.NVV_rh{}_epsilon0_eta0.gt".format(
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
                                'remove_small_components': min_component,
                                'only_normals': True
                            })
                        ],
                       'file_dep': [seg_file],
                       'targets': [PM_graph_file],
                       # force doit to always mark the task as up-to-date
                       # (unless target removed)
                       'uptodate': [True]
                       }
            else:
                print("No segmentation file was found.")


def task_calculate_distances():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/"
    pixel_size = 1.368
    radius_hit = 10
    maxdist_voxels = 60
    maxthick_voxels = 60
    maxdist_nm = int(maxdist_voxels * pixel_size)
    maxthick_nm = int(maxthick_voxels * pixel_size)

    for condition in ["TCB", "SCS", "WT", "IST2", "DTCB1", "DTCB2", "DTCB3"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold = str(subfold_p)
            tomo = "{}{}{}".format(condition, subfold.split('_')[-2],
                                   subfold.split('_')[-1])
            subfold += '/'
            PM_graph_file = "{}{}_PM.NVV_rh{}_epsilon0_eta0.gt".format(
                    subfold, tomo, radius_hit)
            cER_base = "{}{}_cER.VV_area2_rh{}_epsilon0_eta0".format(
                subfold, tomo, radius_hit)
            cER_surf_file = "{}.vtp".format(cER_base)
            cER_graph_file = "{}.gt".format(cER_base)
            cER_surf_outfile = "{}.PMdist_maxdist{}_maxdist2{}.vtp".format(
                cER_base, maxdist_nm, maxthick_nm)
            cER_graph_outfile = "{}.PMdist_maxdist{}_maxdist2{}.gt".format(
                cER_base, maxdist_nm, maxthick_nm)
            distances_outfile = "{}.PMdist_maxdist{}_maxdist2{}.csv".format(
                cER_base, maxdist_nm, maxthick_nm)

            yield {'name': tomo,
                   'verbosity': 2,
                   'actions': [
                       (run_calculate_distances_and_thicknesses,
                        [PM_graph_file, cER_surf_file, cER_graph_file,
                            cER_surf_outfile, cER_graph_outfile,
                            distances_outfile, maxdist_nm, maxthick_nm],
                        {'verbose': False})
                    ],
                   'file_dep': [PM_graph_file, cER_surf_file, cER_graph_file],
                   'targets': [
                       cER_surf_outfile, cER_graph_outfile, distances_outfile],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }

# Note: to run one condition only, e.g. TCB: doit *:TCB*
