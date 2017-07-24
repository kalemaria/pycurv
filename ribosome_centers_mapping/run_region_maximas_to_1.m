%% Runs region_maximas_to_1.m for non-binary masks, making single voxels equal to 1 in each region.

clear all

tomograms = {'t85'};
dist = 18;
restriction = ['_within_' num2str(dist) 'nm_to_membrane_final'];
input_file = ['sec61_centers' restriction '_bin6.mrc'];
output_file = ['sec61_centers' restriction '_binary_bin6.mrc'];

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Binarizing the Sec61 centers mask...');
    region_mask = tom_mrcread([tomograms{i} '/' input_file]); region_mask = region_mask.Value;
    binary_points_mask = region_maximas_to_1(region_mask);
    tom_mrcwrite(binary_points_mask, 'name', [tomograms{i} '/' output_file]);
end

disp('Finished!');