import time
from tomogram_batch_processing import workflow


def main():
    t_begin = time.time()

    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/new_workflow/Htt64Q_IB/"
    tomo = "t166"
    seg_file = "%s%s_cleaned_ER.mrc" % (fold, tomo)
    label = 1
    pixel_size = 2.526
    scale_x = 590
    scale_y = 590
    scale_z = 96
    k = 5

    workflow(fold, tomo, seg_file, label, pixel_size, scale_x, scale_y, scale_z, k)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nElapsed time: %s min %s s' % divmod(duration, 60)

if __name__ == "__main__":
    main()
