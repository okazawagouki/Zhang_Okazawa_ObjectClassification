function Lxy = rgb2Lxy(rgb, calib_file)

persistent cLxy cRGB pcalib_file

ims = size(rgb);
if size(rgb,3) > 1 % image mode (Lxy = [x y 3])
    rgb = reshape(rgb, [ims(1)*ims(2), ims(3)]);
end

%% load lookup table
if isempty(pcalib_file) || ~isequal(pcalib_file, calib_file)
    if strcmp(calib_file, 'sRGB')
        Lxy = sRGB2Lxy(rgb);
        return;
    end
    if ~exist(calib_file, 'file')
        error('calibration file does not exist: %s', calib_file);
    end
    S = load(calib_file);
    cLxy = S.Lxy;
    cRGB = S.digit;
    pcalib_file = calib_file;
end

%% construct lookup table

cXYZ = [cLxy(:,1) .* cLxy(:,2) ./ cLxy(:,3), ...
       cLxy(:,1), ...
       cLxy(:,1) .* (1 - cLxy(:,2) - cLxy(:,3)) ./ cLxy(:,3)];
cXYZ(cLxy(:,3)==0,:) = 0;


% Baseline XYZ
baseXYZ = mean(cXYZ(sum(cRGB,2)==0,:),1);

% lut
RGBdig = cell(1,3);
RGBlum = cell(1,3);
RGBxy = cell(1,3);
for n=1:3
    ind = find(cRGB(:,n) > 0 & sum(cRGB,2) == cRGB(:,n)); % only one digit is non zero.
    RGBdig{n} = [0; cRGB(ind,n)];
    if max(RGBdig{n}) ~= 255
        error('calibration was not performed using a full range (0-255)');
    end
    RGBlum{n} = [0; cLxy(ind,1) - baseXYZ(2)];
    [~, ind2] = max(cLxy(ind,1));
    RGBxy{n} = cLxy(ind(ind2), 2:3);
end

xlut = (0:255)';

Llut = cell(1,3);
for n=1:3
    Llut{n} = interp1(RGBdig{n}, RGBlum{n}, xlut, 'PCHIP');
end


%% main conversion

npt = size(rgb,1);

RGBlum = nan(npt, 3);
for n=1:3
    RGBlum(:,n) = Llut{n}(rgb(:,n)+1);
end

Alum2xyz=[RGBxy{1}(1)/RGBxy{1}(2), RGBxy{2}(1)/RGBxy{2}(2), RGBxy{3}(1)/RGBxy{3}(2); ...
            1, 1, 1 ; ...
            (1-sum(RGBxy{1}))/RGBxy{1}(2)  (1-sum(RGBxy{2}))/RGBxy{2}(2)  (1-sum(RGBxy{3}))/RGBxy{3}(2)];

XYZ = (Alum2xyz * RGBlum')' + ones(npt,1) * baseXYZ;


Lxy = [XYZ(:,2), XYZ(:,1)./sum(XYZ,2), XYZ(:,2)./sum(XYZ,2)];
Lxy(sum(XYZ,2)==0,:) = 0;


if ~isequal(size(Lxy), ims) % image mode
    Lxy = reshape(Lxy, ims);
end


end

function Lxy = sRGB2Lxy(RGB)
    SR = [.64 .33 .03];  %  X Y Z
    SG = [.3 .6 .1];
    SB = [.15 .06 .79];
    white = [.3127 .3290]; % D65
    wLum = 80;
    whiteXYZ = xyL2XYZ([white wLum]);


    A = [SR ;SG; SB]';

    % A * weight = white
    weight = A \ whiteXYZ';

    SR = SR * weight(1);
    SG = SG * weight(2);
    SB = SB * weight(3);


    RGB = double(RGB)/255;

    lRGB = zeros(size(RGB));
    ind = find(RGB(:)<=0.03928);
    lRGB(ind) = RGB(ind)/12.92;
    ind = find(RGB(:)>0.03928);
    lRGB(ind) = ((RGB(ind)+0.055)/1.055).^(2.4);

    XYZ = lRGB(:,1) * SR + lRGB(:,2) * SG + lRGB(:,3) * SB;
    xyL = XYZ2xyL(XYZ);
    Lxy = [xyL(:,3) xyL(:,1) xyL(:,2)];

end


function XYZ = xyL2XYZ(xyL)
    XYZ = [xyL(:,3) .* xyL(:,1) ./ xyL(:,2), ...
           xyL(:,3), ...
           xyL(:,3) .* (1 - xyL(:,1) - xyL(:,2)) ./ xyL(:,2)];
end

function xyL = XYZ2xyL(XYZ)
    xyL = zeros(size(XYZ));
    a = sum(XYZ, 2);
    ind = find(a ~= 0);

    xyL(ind,1) = XYZ(ind,1) ./ a(ind);
    xyL(ind,2) = XYZ(ind,2) ./ a(ind);
    xyL(ind,3) = XYZ(ind,2);
end






	
	