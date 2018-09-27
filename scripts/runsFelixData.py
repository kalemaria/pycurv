import time
import sys

from curvature_calculation import (
    new_workflow, extract_curvatures_after_new_workflow)

"""
Applying new_workflow and extract_curvatures_after_new_workflow on Felix' data.

Author: Maria Kalemanov (Max Planck Institute for Biochemistry),
date: 2018-05-29
"""

__author__ = 'kalemanov'


def main(tomo):
    t_begin = time.time()

    # parameters for all tomograms:
    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_method/"
    base_filename = "{}_ER".format(tomo)
    pixel_size = 2.526  # nm
    radius_hit = 10  # nm
    seg_file = ""
    lbl = 1
    cube_size = 3
    min_component = 100

    # parameters for each tomogram:
    if tomo == "t112":
        fold = "{}diffuseHtt97Q/".format(fold)
        seg_file = "{}_final_ER1_vesicles2_notER3_NE4.Labels.mrc".format(tomo)
    elif tomo == "t122":
        fold = "{}away_from_Htt97Q_IB/".format(fold)
        seg_file = "{}_final_ER1_vesicle2_notER3_NE4.Labels.mrc".format(tomo)
    elif tomo == "t158":
        fold = "{}diffuseHtt25Q/".format(fold)
        seg_file = "{}_final_ER1_NE2.Labels.mrc".format(tomo)
    elif tomo == "t166":
        fold = "{}Htt64Q_IB/".format(fold)
        seg_file = "{}_cleaned_ER.mrc".format(tomo)
    elif tomo == "t85":
        fold = "{}Htt97Q_IB_t85/".format(fold)
        seg_file = "{}_mask_membrane_final_ER.mrc".format(tomo)
    elif tomo == "t84":
        fold = "{}Htt97Q_IB_t84/".format(fold)
        seg_file = "{}_mask_membrane_final_ER.mrc".format(tomo)
    elif tomo == "t92":
        fold = "{}Htt97Q_IB_t92/".format(fold)
        seg_file = "{}_final_ER1_vesicles2_NE3.Labels.mrc".format(tomo)
    elif tomo == "t138":
        fold = "{}Htt97Q_IB_t138/".format(fold)
        pixel_size = 2.84  # nm
        seg_file = "{}_final_ER1_notInHttContact2.Labels.mrc".format(tomo)

    new_workflow(
        fold, base_filename, pixel_size, radius_hit,
        epsilon=0, eta=0, methods=['VV'],
        seg_file=seg_file, label=lbl, holes=cube_size,
        remove_wrong_borders=True, remove_small_components=min_component)

    for b in range(0, 2):
        print("\nExtracting curvatures for ER without {} nm from border".format(
            b))
        extract_curvatures_after_new_workflow(
            fold, base_filename, radius_hit,
            epsilon=0, eta=0, methods=['VV'], exclude_borders=b)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))


if __name__ == "__main__":
    tomo = sys.argv[1]
    main(tomo)
