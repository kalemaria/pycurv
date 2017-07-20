%% Runs find_closest_points_inside_membrane_with_filtering for binary points and membrane masks and a given distance radius in voxels.

clear all

tomograms = {'t92'}; % TODO: change 't84', 't85'
method = 'etomo_cleaned_notcorr_Felix';
disp(['method ' method]);
handedness = 'right_handed';
radius = 4; % TODO: change?
disp(['radius ' num2str(radius)]);
points_mask_file = 'sec61_centers_membrane_bound_manual_binary_bin6.mrc'; % TODO: change (binary one!) t84: 'sec61_centers_membrane_bound_from_tomo_binary_bin6.mrc', t85: 'sec61_centers_within_18nm_to_membrane_final_binary_bin6.mrc'
membrane_mask_file = 'mask_membrane_bin6.mrc'; % TODO: change t85 & t84: 'mask_membrane_final_bin6.mrc'
membrane_points_mask_file = ['sec61_centers_inside_membrane_r' num2str(radius) '_bin6.mrc'];
filtered_points_mask_file = 'sec61_centers_filtered_bin6.mrc';

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Finding the closest Sec61 centers inside the membrane...');
    points_mask = tom_mrcread([tomograms{i} '/' method '/' handedness '/' points_mask_file]); points_mask = points_mask.Value;
    membrane_mask = tom_mrcread([tomograms{i} '/' method '/' membrane_mask_file]); membrane_mask = membrane_mask.Value;
    [membrane_points_mask, distances, filtered_points_mask] = find_closest_points_inside_membrane(points_mask, membrane_mask, radius);
    %tom_mrcwrite(membrane_points_mask, 'name', [tomograms{i} '/' method '/' handedness '/' membrane_points_mask_file]);
    tom_mrcwrite(filtered_points_mask, 'name', [tomograms{i} '/' method '/' handedness '/' filtered_points_mask_file]);
    figure;
    histogram(distances)
end

disp('Finished!');