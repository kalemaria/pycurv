function mask = tom_classmask_unchanged_ref(motl, r, dimxyz, class, bin)
% tom_classmask_unchanged_ref creates a mask from a MOTL using a binary 
% reference (modification of TOM function tom_classmask)
%   mask = tom_classmask_unchanged_ref(motl, r, dimxyz, class, bin);
%
%   motl            motive list
%   r               reference structure OR radius of pasted spheres
%   dimxyz          dimension in x,y,z (1x3 vector) of mask
%   class           class of particles in the MOTL
%   bin             binning factor for both reference and motl(8:10,:)
%
%   mask            output
%
%   Use as 1-0 mask to label a class of particles in a given tomographic
%   volume.
%
% last change 2016-07-14

indx =  find(motl(20,:) == class);
if ischar(r)
    ref = tom_emread(r); 
    ref = ref.Value;
    flag = 'ref';
elseif size(r,2) > 1 % if an already read in volume is given
    ref = r;
    flag = 'ref';
else
     ref = tom_sphere( [r+3 r+3 r+3].*2, r, 1);
     flag = 'sph';
end

ref = tom_bin(ref, bin);

clear mask
mask = zeros(dimxyz);
if min(min(min(motl(8:10,:)))) < 1 || max3d(motl(8:10,:)) > max(dimxyz)
    motl(8:10,:) = motl(8:10,:) / 2^bin + repmat(dimxyz', 1, size(motl,2)) / 2 +1;
end

for j = indx
    j_xyz = floor([motl(8,j) motl(9,j) motl(10,j)] - size(ref)./2 + 1);

    if strcmp(flag, 'ref')
        j_eul = [motl(17,j) motl(18,j) motl(19,j)];
        ref_rot = tom_rotate(ref , j_eul ,'linear');
        mask = tom_paste2(mask, ref_rot, j_xyz, 'max');

    elseif strcmp(flag, 'sph')
        mask = tom_paste2(mask, ref, j_xyz, 'max');

    else
        error('Radius for sphere or reference structure needed as input.')
    end
    display(char([flag ' pasted at xyz ' num2str(j_xyz) ' for particle ' num2str(motl(4,j)) ]))
end
