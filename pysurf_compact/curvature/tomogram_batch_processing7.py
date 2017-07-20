import time
from tomogram_batch_processing import workflow


def main():
    t_begin = time.time()

    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/new_workflow/Htt97Q_IB_t138/"
    tomo = "t138"
    seg_file = "%s%s_final_ER1_notInHttContact2.Labels.mrc" % (fold, tomo)
    label = 1
    pixel_size = 2.839
    scale_x = 928
    scale_y = 928
    scale_z = 300
    k = 5

    workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z, k)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nElapsed time: %s min %s s' % divmod(duration, 60)

if __name__ == "__main__":
    main()
