%% Runs bin_motl_twice.m for motl files reduced to manually picked 
%% and restricted to some distance to the membrane particle lists from 
%% different tomograms with etomo bin 3, pixel size 1.263 nm,
%% to make bin 6 motl files.

clear all

tomograms = {'t85'};
dist = 18;
restriction = ['_within_' num2str(dist) 'nm_to_membrane_final'];
input_file = ['motl_with_mask_manual' restriction '_bin3.em'];
output_file = ['motl_with_mask_manual' restriction '_bin6.em'];

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Binning motive list twice...');
    motl = tom_emread([tomograms{i} '/' input_file]); motl = motl.Value;
    binned_motl = bin_motl_twice(motl);
    tom_emwrite([tomograms{i} '/' output_file], binned_motl);
end

disp('Finished!');