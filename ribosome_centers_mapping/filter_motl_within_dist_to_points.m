function [filtered_motl, distances] = filter_motl_within_dist_to_points(motl, points_mask, dist, binned_twice)
%filter_motl_within_dist_to_points filters a motive list to hold only particle centers within a given 
%distance (in voxels) to any foreground voxel from a given binary points mask.
% Input:
%   motl            motive list with particles
%   points_mask     mask with single, disconnected voxels = 1, the rest = 0
%   dist            maximal distance in voxels between the particle
%                   centers and any voxel=1 from the points mask (point)
% Output:
%   filtered_motl   motive list filtered to particles with a point
%                   within the given distance

size_x = size(points_mask, 1);
size_y = size(points_mask, 2);
size_z = size(points_mask, 3);

filtered_motl = zeros(size(motl, 1), size(motl, 2)); % init. the filtered motl of the biggest possible size
good_part = 0; % conting the "good" particles
distances = [];
for part = 1:size(motl, 2) % for each particle (column in motl),
    % get the coordinates if its center:
    x = motl(8, part);
    y = motl(9, part);
    z = motl(10, part);
    if binned_twice == 1 % if points mask is binned twice compared to the motl, bin the center coordinates twice
        x = ceil(x/2);
        y = ceil(y/2);
        z = ceil(z/2);
    end
    disp(['Particle ' num2str(part) ' at (binned twice) coordinates (' num2str(x) ', ' num2str(y) ', ' num2str(z) ')'])
    % make an euclidean distance matrix D from the particle center:
    particle_center_mask = zeros(size_x, size_y, size_z);
    particle_center_mask(x, y, z) = 1;
    D = bwdist(particle_center_mask); 
    % iterate over all relevant voxels in points_mask:
    min_dist = dist+1;
    for i = (x-ceil(dist)):(x+ceil(dist)) %1:size_x
        for j = (y-ceil(dist)):(y+ceil(dist)) %1:size_y
            for k = (z-ceil(dist)):(z+ceil(dist)) %1:size_z
                if points_mask(i, j, k) == 1 % if there is a point
                    if D(i, j, k) <= dist % and it's within dist to the particle center
                        % update the minimal distance seen so far:
                        min_dist = D(i, j, k);
                    end
                end
            end
        end
    end
    % if a voxel with distance <= dist was found, add this particle to the filtered particle list:
    if min_dist <= dist
        good_part = good_part + 1;
        filtered_motl(:, good_part) = motl(:, part);
        distances(end+1) = min_dist;
        disp(['  Point with distance ' num2str(min_dist) ' found'])
    end
end
filtered_motl = filtered_motl(:, 1:good_part); % cut the filtered motl to the correct size
disp(['Filtered ' num2str(part) ' particles to ' num2str(good_part)])

end

