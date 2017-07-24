function [] = visualise_particles_on_tomo(motl_file, template_file, tomo_x, tomo_y, tomo_z, mapped_particles_file, class)
% visualise_particles_on_tomo mappes a template structure at
% particle positions from a motive list to a tomographic volume.
%
%   motl_file               motive list EM file
%   template_file           template EM or MRC file
%   tomo_x                  x size of the original tomogram in voxels
%   tomo_y                  y size of the original tomogram in voxels
%   tomo_z                  z size of the original tomogram in voxels
%   mapped_particles_file   output EM or MRC file name
%   class                   class of particles in the MOTL
%

% Read in the motive list file.
motl = tom_emread(motl_file); motl = motl.Value;
if size(motl, 2) == 0
    disp(['Motive list ' motl_file ' is empty.'])
else
    % Read in the template file ('.em' or '.mrc').
    if strcmp(template_file(size(template_file, 2)-2:size(template_file, 2)), '.em') % if template_file ends with '.em'
        template = tom_emread(template_file);
    elseif strcmp(template_file(size(template_file, 2)-3:size(template_file, 2)), '.mrc') % else if template_file ends with '.mrc'
        template = tom_mrcread(template_file);
    else
        disp(['File format of ' template_file ' could not be recognized. Can read only .em or .mrc files.']);
    end
    template = template.Value;

    % Create the mapping of particles.
    mapped_particles = tom_classmask(motl, template, [tomo_x tomo_y tomo_z], class, 0);

    % Write the mapping of particles to a file ('.em' or '.mrc').
    if strcmp(mapped_particles_file(size(mapped_particles_file, 2)-2:size(mapped_particles_file, 2)), '.em') % if mapped_particles_file ends with '.em'
        tom_emwrite(mapped_particles_file, mapped_particles);
    elseif strcmp(mapped_particles_file(size(mapped_particles_file, 2)-3:size(mapped_particles_file, 2)), '.mrc') % else if mapped_particles_file ends with '.mrc'
        tom_mrcwrite(mapped_particles, 'name', mapped_particles_file);
    else
        disp(['File format of ' mapped_particles_file ' could not be recognized. Can write only .em or .mrc files.']);
    end
end