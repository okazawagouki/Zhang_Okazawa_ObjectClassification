function output = feature_gistSF(fnames)
%FEATURE_GISTSF Gist + spatial-frequency features, dim = 512 + 32.
%   fnames: cell array of image paths.

    lowlevelmodel_setup();

    calib_file = fullfile(fileparts(mfilename('fullpath')), 'lib', 'calib_data_04-Jan-2020b.mat');
    bkg_color = [100 100 100];
    % Parameters:
    clear param
    param.imageSize = [256 256]; % it works also with non-square images
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;

    dim = 512+4*8;
    output = zeros(length(fnames),dim);
    wb = waitbar_text(0);
    for i = 1:length(fnames)
        waitbar_text(i/length(fnames), wb);
        img_url = fnames{i};
        img = imread(img_url);
        img = img(:,:,1);
        % [~, ~, alpha] = imread(img_url);
        % img = repmat(alpha, [1, 1, 3]);
        [gist, param] = LMgist(img, '', param);
        output(i,1:512) = zscore(gist(1,:));

        [img, ~, alpha] = imread(img_url);
        % img = repmat(alpha, [1, 1, 3]);
        feat = extract_lowlev_feature(img, alpha, bkg_color, calib_file);
        sf = zscore(feat.sf_mag,0,1);
        output(i,513:end) = reshape(sf,[32,1]);
    end
    waitbar_text('close', wb);

end
