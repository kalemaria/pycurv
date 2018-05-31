import time
import sys

from curvature_calculation import (
    new_workflow, extract_curvatures_after_new_workflow)


def main(tomo):
    t_begin = time.time()

    # parameters for all tomograms:
    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_methods/"
    base_filename = "{}_ER".format(tomo)
    pixel_size = 2.526  # nm
    scale_x = 1
    scale_y = 1
    scale_z = 1
    radius_hit = 10  # nm
    seg_file = ""
    lbl = 1
    cube_size = 3
    min_component = 100

    # parameters for each tomogram:
    if tomo == "t112":  # running on winzererfaehndl from 2018-05-29 13:21, finished on at 23:03
        fold = "{}diffuseHtt97Q/".format(fold)
        scale_x = 620
        scale_y = 620
        scale_z = 80
        seg_file = "{}_final_ER1_vesicles2_notER3_NE4.Labels.mrc".format(tomo)
    elif tomo == "t122":  # running on braeurosl from 2018-05-29 13:27, finished at 22:35
        fold = "{}away_from_Htt97Q_IB/".format(fold)
        scale_x = 618
        scale_y = 618
        scale_z = 91
        seg_file = "{}_final_ER1_vesicle2_notER3_NE4.Labels.mrc".format(tomo)
    elif tomo == "t158":  # running on hacker from 2018-05-29 13:30, finished on 05-30 12:25
        fold = "{}diffuseHtt25Q/".format(fold)
        scale_x = 590
        scale_y = 590
        scale_z = 141
        seg_file = "{}_final_ER1_NE2.Labels.mrc".format(tomo)
    elif tomo == "t166":  # running on augustiner from 2018-05-29 13:56, finished on 05-30 1:47
        fold = "{}Htt64Q_IB/".format(fold)
        scale_x = 590
        scale_y = 590
        scale_z = 96
        seg_file = "{}_cleaned_ER.mrc".format(tomo)
    elif tomo == "t85":  # running on kaefer from 2018-05-29 13:58, finished at 20:12
        fold = "{}Htt97Q_IB_t85/".format(fold)
        scale_x = 590
        scale_y = 590
        scale_z = 261
        seg_file = "{}_mask_membrane_final_ER.mrc".format(tomo)
    elif tomo == "t84":  # running on ochsenbraterei from 2018-05-29 14:02, finished at 19:51
        fold = "{}Htt97Q_IB_t84/".format(fold)
        scale_x = 590
        scale_y = 590
        scale_z = 171
        seg_file = "{}_mask_membrane_final_ER.mrc".format(tomo)
    elif tomo == "t92":  # running on loewenbraeu from 2018-05-29 14:05, finished at 20:21
        fold = "{}Htt97Q_IB_t92/".format(fold)
        scale_x = 590
        scale_y = 590
        scale_z = 266
        seg_file = "{}_final_ER1_vesicles2_NE3.Labels.mrc".format(tomo)

    new_workflow(
        fold, base_filename, pixel_size, scale_x, scale_y, scale_z,
        radius_hit, epsilon=0, eta=0, methods=['VV'],
        seg_file=seg_file, label=lbl, holes=cube_size,
        remove_wrong_borders=True, remove_small_components=min_component)

    for b in range(0, 2):
        print("\nExtracting curvatures for ER without {} nm from border".format(
            b))
        extract_curvatures_after_new_workflow(
            fold, base_filename, pixel_size, scale_x, scale_y, scale_z,
            radius_hit, epsilon=0, eta=0, methods=['VV'], exclude_borders=b)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nTotal elapsed time: %s min %s s' % divmod(duration, 60)


if __name__ == "__main__":
    tomo = sys.argv[1]
    main(tomo)
