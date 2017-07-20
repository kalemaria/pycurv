function [] = visualise_binary_particles_on_tomo(motl_file, template_file, tomo_x, tomo_y, tomo_z, mapped_particles_file)

% Read in the motive list file.
motl = tom_emread(motl_file); motl = motl.Value;

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
mapped_particles = tom_classmask_unchanged_ref(motl, template, [tomo_x tomo_y tomo_z], 0, 0);

% Write the mapping of particles to a file ('.em' or '.mrc').
if strcmp(mapped_particles_file(size(mapped_particles_file, 2)-2:size(mapped_particles_file, 2)), '.em') % if mapped_particles_file ends with '.em'
    tom_emwrite(mapped_particles_file, mapped_particles);
elseif strcmp(mapped_particles_file(size(mapped_particles_file, 2)-3:size(mapped_particles_file, 2)), '.mrc') % else if mapped_particles_file ends with '.mrc'
    tom_mrcwrite(mapped_particles, 'name', mapped_particles_file);
else
    disp(['File format of ' mapped_particles_file ' could not be recognized. Can write only .em or .mrc files.']);
end