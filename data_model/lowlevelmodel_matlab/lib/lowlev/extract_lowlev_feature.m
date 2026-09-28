function feat = extract_lowlev_feature(img, alpha, bkg_col, calib_file)

if size(img,1) ~= size(img,2)
    error('image should be square');
end

if isempty(alpha)
    alpha = ones(size(img,1), size(img,2));
end
if isa(alpha, 'uint8')
    alpha = double(alpha)/255;
end

for n=1:3
    img(:,:,n) = uint8(double(img(:,:,n)) .* alpha + double(bkg_col(n)) .* (1 - alpha));
end

silhouette = alpha > .5;

%% color information
% See https://www.ncbi.nlm.nih.gov/pubmed/21569854

segment = 4;

Luv = rgb2Luv(img, calib_file);

segLuv = segment_img(Luv, segment);

feat.col.mean = [get_mean(Luv(:,:,1), alpha), get_mean(Luv(:,:,2), alpha), get_mean(Luv(:,:,3), alpha)];
feat.col.sd =   [get_sd(Luv(:,:,1), alpha),   get_sd(Luv(:,:,2), alpha),   get_sd(Luv(:,:,3), alpha)];
feat.col.skew = [get_skew(Luv(:,:,1), alpha), get_skew(Luv(:,:,2), alpha), get_skew(Luv(:,:,3), alpha)];
feat.col.kurt = [get_kurt(Luv(:,:,1), alpha), get_kurt(Luv(:,:,2), alpha), get_kurt(Luv(:,:,3), alpha)];

feat.col.segment_L_mean = cellfun(@(x)get_mean(x(:,:,1)), segLuv);
feat.col.segment_u_mean = cellfun(@(x)get_mean(x(:,:,2)), segLuv);
feat.col.segment_v_mean = cellfun(@(x)get_mean(x(:,:,3)), segLuv);

feat.col.segment_L_sd = cellfun(@(x)get_sd(x(:,:,1)), segLuv);
feat.col.segment_u_sd = cellfun(@(x)get_sd(x(:,:,2)), segLuv);
feat.col.segment_v_sd = cellfun(@(x)get_sd(x(:,:,3)), segLuv);

feat.col.segment_L_skew = cellfun(@(x)get_skew(x(:,:,1)), segLuv);
feat.col.segment_u_skew = cellfun(@(x)get_skew(x(:,:,2)), segLuv);
feat.col.segment_v_skew = cellfun(@(x)get_skew(x(:,:,3)), segLuv);

feat.col.segment_L_kurt = cellfun(@(x)get_kurt(x(:,:,1)), segLuv);
feat.col.segment_u_kurt = cellfun(@(x)get_kurt(x(:,:,2)), segLuv);
feat.col.segment_v_kurt = cellfun(@(x)get_kurt(x(:,:,3)), segLuv);

[feat.col_vector, feat.col_vector_info] = generate_vector(feat.col);

%% spatial frequency information
% See https://www.ncbi.nlm.nih.gov/pubmed/21569854
Nsc = 4;
Nor = 8;

[feat.sf_mag, feat.info.sf_mag_info] = get_sf_mag(Luv(:,:,1), Nsc, Nor);
% magnitude of each spatial frequency/orientation component

%% object shape info
% See https://www.ncbi.nlm.nih.gov/pubmed/28654965

feat.obj.abs_size = mean(alpha(:));
    % size of object occupied in an image. 1... filling all pixels, 0... empty

feat.obj.rectangle_fill = get_rectangle_fill(alpha);
    % size of object within a rectangular bounding box.
    % 1... rectangular object. 0.. empty

feat.obj.HV_aspect = get_HV_aspect(silhouette);
    % aspect ratio between horizontal and vertical axis of an object image
    % vertical/horizontal

[feat.contour, feat.norm_contour] = get_contour_coord(silhouette);

feat.obj.contour_dist = compute_contour_dist(feat.norm_contour, 0:5:360);
    % for each 10 degree, compute the distance of the counter from the
    % origin

[~, r] = cart2pol(feat.norm_contour(:,1), feat.norm_contour(:,2));
feat.obj.contour_var = std(r);
    % contour variance (standard deviation of the distance from the centroid of 
    % each object to each point on the object’s contour)
    
%% more object shape info

Nsc = 6;
[~, r] = cart2pol(feat.norm_contour(:,1), feat.norm_contour(:,2));
[feat.obj.contour_sf, feat.info.contour_sf_info] = get_contour_sf(r, Nsc);
    % spatial frequency of contour

[feat.obj.longitudinal_axis, feat.obj.elongation_factor] = get_longitudinal_axis(feat.norm_contour);
    % angle of axis that is most elongated (degree)
    % aspect ratio between the most elongated and its orthogonal axis

[feat.contour_vector, feat.contour_vector_info] = generate_vector(feat.obj);

end

function seg = segment_img(img, segment)
    block_rows = floor(size(img,1) / segment);
    block_cols = floor(size(img,2) / segment);

    seg = cell(segment);

    for i = 1:segment
        for j = 1:segment
            row_range = (i-1)*block_rows + 1 : i*block_rows;
            col_range = (j-1)*block_cols + 1 : j*block_cols;
            seg{i, j} = img(row_range, col_range, :);
        end
    end
end

function mn = get_mean(val, alpha)
    if ~exist('alpha', 'var')
        alpha = ones(size(val,1), size(val,2));
    end
    mn = sum(val(:) .* alpha(:)) / sum(alpha(:));
end

function sd = get_sd(val, alpha)
    if ~exist('alpha', 'var')
        alpha = ones(size(val,1), size(val,2));
    end
    mn = get_mean(val, alpha);
    sd = sqrt(sum((val(:) - mn).^2 .* alpha(:)) / sum(alpha(:)));
end

function sk = get_skew(val, alpha)
    if ~exist('alpha', 'var')
        alpha = ones(size(val,1), size(val,2));
    end
    mn = get_mean(val, alpha);
    sd = get_sd(val, alpha);
    m3 = sum((val(:) - mn).^3 .* alpha(:)) / sum(alpha(:));
    sk = m3 / (sd^3);
    if sd < 1e-10
        sk = 0;
    end
end

function kt = get_kurt(val, alpha)
    if ~exist('alpha', 'var')
        alpha = ones(size(val,1), size(val,2));
    end
    mn = get_mean(val, alpha);
    sd = get_sd(val, alpha);
    m4 = sum((val(:) - mn).^4 .* alpha(:)) / sum(alpha(:));
    kt = m4 / (sd^4);
    if sd < 1e-10
        kt = 0;
    end
end

function [mag, info] = get_sf_mag(img, Nsc, Nor)
    [pyr0,pind0] = buildSCFpyr(img, Nsc, Nor-1);
    nband = size(pind0,1);
    pyr0(pyrBandIndices(pind0,nband)) = ...
        real(pyrBand(pyr0,pind0,nband)) - mean2(real(pyrBand(pyr0,pind0,nband)));

    apyr0 = abs(pyr0);
    mag = zeros(size(pind0,1), 1);
    for nband = 1:size(pind0,1)
      indices = pyrBandIndices(pind0,nband);
      mag(nband) = mean2(apyr0(indices));
    end
    mag = reshape(mag(2:end-1), [Nor Nsc]); % orientation x scale
    
    if Nor == 4
        info = {'Vertical', 'Left oblique', 'Horizontal', 'Right oblique'};
    else
        info = [];
    end
end

function val = get_rectangle_fill(alpha)
        % define bounding box
    x1 = find(any(alpha>0,1), 1, 'first');
    x2 = find(any(alpha>0,1), 1, 'last');
    y1 = find(any(alpha>0,2), 1, 'first');
    y2 = find(any(alpha>0,2), 1, 'last');
    
    alpha = alpha(y1:y2, x1:x2); % clip img to bounding box
    val = mean(alpha(:));
end

function val = get_HV_aspect(alpha)
    x1 = find(any(alpha,1), 1, 'first');
    x2 = find(any(alpha,1), 1, 'last');
    y1 = find(any(alpha,2), 1, 'first');
    y2 = find(any(alpha,2), 1, 'last');
    val = (x2-x1)/(y2-y1); % Horizontal/Vertical
end

function [contour, ncontour] = get_contour_coord(alpha)
    alpha = fliplr(alpha');
    c = bwboundaries(alpha);
    clen = cellfun(@(x)size(x,1), c);
    [~, idx] = max(clen);
    contour = c{idx};
    contour = bsxfun(@minus, contour, mean(contour,1)); % centering
    ncontour = bsxfun(@rdivide, contour, size(alpha));
end

function dist = compute_contour_dist(contour, angle)
    [th, r] = cart2pol(contour(:,1), contour(:,2));
    th = th/pi*180;
    th(th < 0) = th(th < 0) + 360;
    dist = nan(length(angle)-1,1);
    for n=1:length(angle)-1
        ind = th >= angle(n) & th < angle(n+1);
        dist(n) = mean(r(ind));
    end
end



function [ang, elongation_factor] = get_longitudinal_axis(contour)
    [th, r] = cart2pol(contour(:,1), contour(:,2));
    th(th < 0) = th(th < 0) + 2 * pi;
    th(th > 2 * pi) = th(th > 2 * pi) - 2 * pi;
    
    alen = nan(180,1); % length along each angle
    for n=1:180
        [sang1, thidx1] = sort(abs(th - n/180*pi), 'ascend'); % find contour point within 1deg
        if sang1(1) > 5/180*pi % no point satisfies
            thidx1 = [];
        end
        
        [sang2, thidx2] = sort(abs(th - (n/180*pi + pi)), 'ascend'); % find contour point within 1deg
        if sang2(1) > 5/180*pi % no point satisfies
            thidx2 = [];
        end
        if ~isempty(thidx1) && ~isempty(thidx2) % point on both side
            alen(n) = max(r(thidx1(1))) + max(r(thidx2(1)));
        elseif ~isempty(thidx1) % one side
            thidx1 = thidx1(sang1 < 5/180*pi);
            if length(thidx1) > 1
                alen(n) = max(r(thidx1)) - min(r(thidx1));
            end
        elseif ~isempty(thidx2)
            thidx2 = thidx2(sang2 < 5/180*pi);
            if length(thidx2) > 1
                alen(n) = max(r(thidx2)) - min(r(thidx2));
            end
        end
    end
    [~, ang] = nanmax(alen);
    if ang == 180, ang = 0; end
    
    % get elongation factor
    [vx,vy] = pol2cart(ang/180*pi, 1);
    proj = contour * [vx;vy];
    Llen = max(proj) - min(proj);
    
    [vx,vy] = pol2cart((ang + 90)/180*pi, 1);
    proj = contour * [vx;vy];
    Slen = max(proj) - min(proj);
    elongation_factor = Llen/Slen;
end


function [val, sf_range] = get_contour_sf(contour, Nsc)
    sfp = fft(contour);
    sfp = abs(sfp(1:floor(end/2)));

    val = nan(Nsc,1);
    % 2-3, 4-7, 8-15, 16-32
    sf_range = nan(Nsc, 2);
    for n=1:Nsc
        st = 2^n;
        en = 2^(n+1);
        sf_range(n,:) = [st, en-1];
        val(n) = sum(sfp(st:en-1));
    end
end

function [vec, info] = generate_vector(in)
    fi = fieldnames(in);
    vec = [];
    info = {};
    for n=1:length(fi)
        val = in.(fi{n});
        val = val(:);
        vec = [vec; val]; %#ok<AGROW>
        info = [info; repmat(fi(n), length(val), 1)]; %#ok<AGROW>
    end
end



