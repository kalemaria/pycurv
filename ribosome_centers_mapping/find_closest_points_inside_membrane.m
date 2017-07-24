function [membrane_points_mask, distances, filtered_points_mask] = find_closest_points_inside_membrane(points_mask, membrane_mask, r)
%find_closest_points_inside_membrane finds closest voxels, within radius r, to each voxel in
%the point mask that are inside the membrane mask.
% Input:
%   points_mask             mask with single, disconnected voxels = 1, the rest = 0
%   membrane_mask           mask with membrane voxels = 1, others = 0
%   r                       radius in pixels
% Output:
%   membrane_points_mask    mask with single voxels inside membrane = 1, the rest = 0
%   distances               vector with the distances between the original
%                           voxels and the closest ones inside the membrane
%   filtered_points_mask    mask like the original points mask, only filtered
%                           to those voxels=1 having a near voxel inside membrane 

size_x = size(points_mask, 1);
size_y = size(points_mask, 2);
size_z = size(points_mask, 3);

membrane_points_mask = zeros(size_x, size_y, size_z);
distances = [];
filtered_points_mask = zeros(size_x, size_y, size_z);
% iterate over the foreground voxels in the points mask:
for x = 1:size_x
    for y = 1:size_y
        for z = 1:size_z
            if points_mask(x, y, z) == 1
                disp(['Point (' num2str(x) ', ' num2str(y) ', ' num2str(z) ')'])
                % if the voxel is inside the membrane, add it to the output masks:
                if membrane_mask(x, y, z) == 1
                    disp('    is inside membrane.')
                    membrane_points_mask(x, y, z) = 1;
                    distances(end+1) = 0;
                    filtered_points_mask(x, y, z) = 1;
                else % otherwise:
                    % make a mask of same size with only this voxel equal to 1:
                    voxel_mask = zeros(size_x, size_y, size_z);
                    voxel_mask(x, y, z) = 1;
                    % make a sphere-mask of same size with center at this
                    % voxel and radius equal to r (1 inside the sphere & 0 outside):
                    D = bwdist(voxel_mask); % volume with euclidean distances from this voxel
                    sphere_mask = D <= r;
                    % weight the sphere-mask by inverse distances:
                    weighted_sphere_mask = sphere_mask ./ (D+1);
                    % overlap the weighted sphere-mask with the membrane mask to get a "region mask"
                    % of nearby voxels inside the membrane weighted by "closiness" to the original voxel:
                    region_mask = weighted_sphere_mask .* single(membrane_mask);
                    % get the maximal intensity voxel from the region:
                    max_intensity = 0; % init.
                    central_voxel = []; % init.
                    % iterate over the relevant voxels in the region mask:
                    for i = (x-ceil(r)):(x+ceil(r))
                        for j = (y-ceil(r)):(y+ceil(r))
                            for k = (z-ceil(r)):(z+ceil(r))
                                voxel_intensity = region_mask(i, j, k);
                                if voxel_intensity > 0 % in foreground of the region
                                    % update the maximal intensity and the "central voxel", having
                                    % the highest intensity so far:
                                    if voxel_intensity > max_intensity
                                        max_intensity = voxel_intensity;
                                        central_voxel = [i, j, k];
                                    end
                                end
                            end
                        end
                    end
                    % if the current voxel had nearby voxels (distance <= r) inside the membrane, add the central
                    % voxel (with highest value) into the binary mask:
                    if max_intensity > 0
                        disp(['    is within ' num2str(r) ' voxels to membrane. Nearest membrane voxel is (' num2str(central_voxel(1)) ', ' num2str(central_voxel(2)) ', ' num2str(central_voxel(3)) ')'])
                        membrane_points_mask(central_voxel(1), central_voxel(2), central_voxel(3)) = 1;
                        distances(end+1) = 1/max_intensity - 1;
                        filtered_points_mask(x, y, z) = 1;
                    end
                end
            end
        end
    end
end
disp([num2str(size(distances, 2)) ' points are now inside the membrane.'])

end
