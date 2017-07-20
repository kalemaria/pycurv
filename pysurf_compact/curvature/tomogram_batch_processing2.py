import time
from tomogram_batch_processing import workflow


def main():
    t_begin = time.time()

    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/new_workflow/away_from_Htt97Q_IB/"
    tomo = "t122"
    seg_file = "%s%s_final_ER1_vesicle2_notER3_NE4.Labels.mrc" % (fold, tomo)
    label = 1
    pixel_size = 2.526
    scale_x = 618
    scale_y = 618
    scale_z = 91
    k = 5

    workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z, k)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nElapsed time: %s min %s s' % divmod(duration, 60)

if __name__ == "__main__":
    main()
