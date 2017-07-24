function binned_motl = bin_motl_twice(motl)
%bin_motl_twice bins the particle coordinates in a motive list twice in each dimension and returns
%the binned motive list.
% Input:
%   motl        motive list to be binned twice
% Output:
%   binned_motl motive list binned twice 
binned_motl = motl;
for i = 1:size(motl, 2) % for each particle, bin twice x, y, z coordinates:
    binned_motl(8, i) = ceil(motl(8, i)/2);
    binned_motl(9, i) = ceil(motl(9, i)/2);
    binned_motl(10, i) = ceil(motl(10, i)/2);
end