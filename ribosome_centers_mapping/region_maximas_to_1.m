function binary_points_mask = region_maximas_to_1(region_mask)
%region_maximas_to_1 takes a grey-scale region mask, finds intensity
%maximas of each region and returns a binary mask where the maxima voxels
%are 1 and the rest 0.
% Input:
%   region_mask         grey-scale region mask
% Output:
%   binary_points_mask  binary mask with one foreground voxel per region    

binary_region_mask = region_mask > 0;
stats = regionprops('table', binary_region_mask, region_mask, 'MaxIntensity', 'PixelList');
voxel_lists = stats.PixelList;
max_intensities = stats.MaxIntensity;

binary_points_mask = zeros(size(region_mask,1), size(region_mask,2), size(region_mask,3)) > 0;
for ri = 1:size(stats, 1) % iterate over regions (ri = region index)
    disp(['Region number ' num2str(ri)]);
    region_voxels = voxel_lists(ri, 1); % 1x1 cell
    region_voxels = region_voxels{:};
    region_max_intensity = max_intensities(ri, 1);
    for vi = 1:size(region_voxels, 1) % iterate over voxels in the current region (vi = voxel index)
        %disp(['  Voxel number ' num2str(vi)]); %test
        % get voxel's coordinates (x, y, z):
        x = region_voxels(vi, 2);
        y = region_voxels(vi, 1); % I don't know why regionprops exchanges x and y coordinates...
        z = region_voxels(vi, 3);
        voxel_intensity = region_mask(x, y, z);
        if voxel_intensity == region_max_intensity
            disp(['    Maximal intensity is ' num2str(voxel_intensity) ' at voxel (' num2str(x) ', ' num2str(y) ', ' num2str(z) ')']);
            binary_points_mask(x, y, z) = 1;
            break;
        end
    end
end    
    
end

