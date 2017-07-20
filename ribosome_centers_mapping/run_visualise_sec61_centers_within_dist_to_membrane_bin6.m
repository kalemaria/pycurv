%% Runs visualise_binary_particles_on_tomo.m for motl files reduced to manually picked 
%% and restricted to some distance to the membrane particle lists from 
%% different tomograms reconstructed by Felix with etomo bin 3, pixel size 1.263 nm & binned to bin 6

clear all

tomograms = {'t84'}; % TODO: change 't85', 't92'
method = 'etomo_cleaned_notcorr_Felix';
disp(['method ' method]);
handedness = 'right_handed';
%dist = 18; %t85
%restriction = ['_within_' num2str(dist) 'nm_to_membrane_final']; %t85
motl_em_file = 'motl_membrane_bound_from_tomo_bin6.em'; % TODO: change t85: ['motl_with_mask_manual' restriction '_bin6.em'], t92: 'motl_membrane_bound_manual_bin6.em'
disp(['motl ' handedness '/' motl_em_file]);
tomo_x = 590;
tomo_y = tomo_x;
tomo_z = [171]; % TODO: change (same length as tomograms!) t85: 261, t92: 266
template = 'sec61_center_mask_bin6.mrc';
output_file = 'sec61_centers_membrane_bound_from_tomo_bin6.mrc'; % TODO: change t85: ['sec61_centers' restriction '_bin6.mrc'], t92: 'sec61_centers_membrane_bound_manual_bin6.mrc'

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Mapping Sec61 centers at place of ribosomes...');
    visualise_binary_particles_on_tomo([tomograms{i} '/' method '/' handedness '/' motl_em_file], template, tomo_x, tomo_y, tomo_z(i), [tomograms{i} '/' method '/' handedness '/' output_file]);
end

disp('Finished!');