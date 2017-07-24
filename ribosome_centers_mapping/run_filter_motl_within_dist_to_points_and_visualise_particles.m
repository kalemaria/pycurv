%% Runs filter_motl_within_dist_to_points.m with a motive list file, points mask file and a distance cutoff and saves the filtered motive list to a file.
%% Then runs visualise_particles_on_tomo.m with the filtered motive list.

clear all

tomograms = {'t85'};
motl_file = 'motl_with_mask_manual_within_18nm_to_membrane_final_bin3.em';
points_mask_file = 'sec61_centers_filtered_bin6.mrc';
dist = 7; %maximal distance between ribosome center and (original) Sec61 center in bin 6
%dist has to be determined so that the number of resulting particles is the same as the number of Sec61 center points in the mask
filtered_motl_file = 'motl_filtered.em';
binned_twice = 1; % 1 if motls are bin 3 and points masks bin 6
ribosome_mapping_file = 'mapped_ribosomes_filtered.mrc';

% tomogram sizes in bin 3:
tomo_x = 1180;
tomo_y = tomo_x;
tomo_z = [521];

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Filtering motive list...');
    motl = tom_emread([tomograms{i} '/' motl_file]); motl = motl.Value;
    points_mask = tom_mrcread([tomograms{i} '/' points_mask_file]); points_mask = points_mask.Value;
    [filtered_motl, distances] = filter_motl_within_dist_to_points(motl, points_mask, dist, binned_twice);
    tom_emwrite([tomograms{i} '/' filtered_motl_file], filtered_motl);
    figure;
    histogram(distances)
    
    disp('Mapping whole ribosomes with membrane');
    visualise_particles_on_tomo([tomograms{i} '/' filtered_motl_file], 'reference_with_membrane.em', tomo_x, tomo_y, tomo_z(i), [tomograms{i} '/' ribosome_mapping_file], 0);
end

disp('Finished!');