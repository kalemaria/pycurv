%% Runs visualise_binary_particles_on_tomo.m for motl files reduced to manually picked 
%% and restricted to some distance to the membrane particles from different 
%% tomograms with etomo bin 3, pixel size 1.263 nm and binned to bin 6

clear all

tomograms = {'t85'};
dist = 18;
restriction = ['_within_' num2str(dist) 'nm_to_membrane_final'];
motl_em_file = ['motl_with_mask_manual' restriction '_bin6.em'];
tomo_x = 590;
tomo_y = tomo_x;
tomo_z = [261];
template = 'sec61_center_mask_bin6.mrc';
output_file = ['sec61_centers' restriction '_bin6.mrc'];

for i=1:length(tomograms)
    disp(['Tomogram ' tomograms{i}]);
    disp('Mapping Sec61 centers at place of ribosomes...');
    visualise_binary_particles_on_tomo([tomograms{i} '/' motl_em_file], template, tomo_x, tomo_y, tomo_z(i), [tomograms{i} '/' output_file]);
end

disp('Finished!');