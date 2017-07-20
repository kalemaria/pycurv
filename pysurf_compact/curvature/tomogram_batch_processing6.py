import time
from tomogram_batch_processing import workflow


def main():
    t_begin = time.time()

    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/new_workflow/Htt97Q_IB_t84_with_pyseg_compact/"  # TODO change to original path (remove "_with_pyseg_compact")
    tomo = "t84"
    seg_file = "%s%s_mask_membrane_final_ER.mrc" % (fold, tomo)
    label = 1
    pixel_size = 2.526
    scale_x = 590
    scale_y = 590
    scale_z = 171
    k = 5

    workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z, k)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nElapsed time: %s min %s s' % divmod(duration, 60)

if __name__ == "__main__":
    main()
