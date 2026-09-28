function [feat, fname, feat_info] = extract_lowlev_feature_batch(folder, bkg_color, calib_file, type, grayscale)

if ~exist('type', 'var')
    type = 'struct'; % struct or matrix
end

pix_size = 600;
margin_size = 10;

if ischar(folder)
    fname = dir([folder '**/*.png']);
    fname = arrayfun(@(x)[x.folder '/' x.name], fname, 'uni', 0);
else
    fname = folder;
end

feat = cell(length(fname),1);

for n=1:length(fname)
    [img, ~, alpha] = imread(fname{n});
    if isempty(alpha)
        warning('%s does not have alpha', fname{n});
    end
    [img, alpha] = obj_image_resizing(img, alpha, [1 1] * pix_size, [1 1] * margin_size);
    if grayscale
        img = rgb2gray(img);
        img = cat(3, img, img, img);
    end
    feat{n} = extract_lowlev_feature(img, alpha, bkg_color, calib_file);
end

feat_info = {};
if isequal(type, 'struct')
    return;
end

feat_mat = [];

% color
feat_mat = [feat_mat, cell2mat(cellfun(@(x)x.col_vector, feat(:)', 'uni', 0))']; %#ok<*AGROW>
feat_info = [feat_info, feat{1}.col_vector_info(:)'];

% spatial frequency mag
[scale, ori] = meshgrid(1:size(feat{1}.sf_mag,2), 1:size(feat{1}.sf_mag,1));
sf_mag = cell2mat(cellfun(@(x)x.sf_mag(:)', feat, 'uni', 0));
sf_info = arrayfun(@(x,y)sprintf('SF mag: scale %d ori %d', x, y), scale(:), ori(:), 'uni', 0);
feat_mat = [feat_mat, sf_mag];
feat_info = [feat_info, sf_info(:)'];

% contour info
feat_mat = [feat_mat, cell2mat(cellfun(@(x)x.contour_vector, feat(:)', 'uni', 0))']; %#ok<*AGROW>
feat_info = [feat_info, feat{1}.contour_vector_info(:)'];

feat = feat_mat;

end
