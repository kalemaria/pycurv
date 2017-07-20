% Creates a mask with a single voxel, which is approximately the Sec61 center 
% on the bin 6 membrane-bound ribosome template.

size = 23;
mask = zeros(size, size, size);
center_x = 11;
center_y = 13;
center_z = 17;
mask(center_x, center_y, center_z) = 1;
tom_mrcwrite(mask, 'name', 'sec61_center_mask_bin6.mrc');