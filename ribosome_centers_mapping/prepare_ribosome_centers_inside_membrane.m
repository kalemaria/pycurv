%% Workflow for mapping ribosome centers onto membrane. 
%% One has to modify the input / output and parameters in each run-script.

% 1. Binning the particle coordinates in the motive list from template
% matching from bin 3 to bin 6, because membrane segmentation was done
% using bin 6 tomograms
run_bin_motl_twice

% 2. a) Creating a mask with a single voxel, which is approximately the 
% Sec61 center on the bin 6 membrane-bound ribosome template
create_sec61_center_mask_bin6

% 2. b) Mapping the Sec61 center binary mask at particle positions from
% the motive list to a volume of size like the bin 6 tomogram
run_visualise_sec61_centers_within_dist_to_membrane_bin6

% 2. c) Binarizes the non-binary mask resulting from the previous step (due 
% to rotations and interpolations in the volume)
run_region_maximas_to_1

% 3. Finding closest points inside the membrane mask from the Sec61 centers
% within a small allowed radius, discarding particles that are oriented
% with incorrect angle to the membrane, which might be false positives
run_find_closest_points_inside_membrane

% 4. Filtering the motive list to the particle centers that could be mapped 
% to the membrane in the previous step and visualising them in bin 3 volume
run_filter_motl_within_dist_to_points_and_visualise_particles
