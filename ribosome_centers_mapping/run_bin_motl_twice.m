%% Runs bin_motl_twice.m for motl files reduced to manually picked 
%% and restricted to some distance to the membrane particle lists from 
%% different tomograms reconstructed by Felix with etomo bin 3, pixel size 1.263 nm,
%% to make bin 6 motl files.

clear all

tomograms = {'t84'}; % TODO: change 't85', 't92'
method = 'etomo_cleaned_notcorr_Felix';
disp(['method ' method]);
handedness = 'right_handed';
%dist = 18; %t85
%restriction = ['_within_' num2str(dist) 'nm_to_membrane_final']; %t85
input_file = 'motl_membrane_bound_from_tomo.em'; % TODO: change t85: ['motl_with_mask_manual' restriction '.em'], t92: 'motl_membrane_bound_manual.em'
disp(['motive list: ' handedness '/' input_file]);
output_file = 'motl_membrane_bound_from_tomo_bin6.em'; % TODO: change t85: ['motl_with_mask_manual' restriction '_bin6.em'], t92: 'motl_membrane_bound_manual_bin6.em'

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Binning motive list twice...');
    motl = tom_emread([tomograms{i} '/' method '/' handedness '/' input_file]); motl = motl.Value;
    binned_motl = bin_motl_twice(motl);
    tom_emwrite([tomograms{i} '/' method '/' handedness '/' output_file], binned_motl);
end

disp('Finished!');