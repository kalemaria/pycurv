%% Runs region_maximas_to_1.m for non-binary masks, making single voxels = 1 in each region.

clear all

tomograms = {'t84'}; % TODO: change 't85', 't92'
method = 'etomo_cleaned_notcorr_Felix';
disp(['method ' method]);
handedness = 'right_handed';
%dist = 18; %t85
%restriction = ['_within_' num2str(dist) 'nm_to_membrane_final']; %t85
input_file = 'sec61_centers_membrane_bound_from_tomo_bin6.mrc'; % TODO: change t85: ['sec61_centers' restriction '_bin6.mrc'], t92: 'sec61_centers_membrane_bound_manual_bin6.mrc'
output_file = 'sec61_centers_membrane_bound_from_tomo_binary_bin6.mrc'; % TODO: change t85: ['sec61_centers' restriction '_binary_bin6.mrc'], t92: 'sec61_centers_membrane_bound_manual_binary_bin6.mrc'

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Binarizing the Sec61 centers mask...');
    region_mask = tom_mrcread([tomograms{i} '/' method '/' handedness '/' input_file]); region_mask = region_mask.Value;
    binary_points_mask = region_maximas_to_1(region_mask);
    tom_mrcwrite(binary_points_mask, 'name', [tomograms{i} '/' method '/' handedness '/' output_file]);
end

disp('Finished!');