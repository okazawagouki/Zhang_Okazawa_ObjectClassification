function output = feature_contour(fnames)
%FEATURE_CONTOUR Object shape / contour features, dim = 12 + 72.
%   fnames: cell array of image paths. Foreground/background is read from
%   the image alpha channel.

    lowlevelmodel_setup();

    calib_file = fullfile(fileparts(mfilename('fullpath')), 'lib', 'calib_data_04-Jan-2020b.mat');
    bkg_color = [100 100 100];

    %% object shape
    output = zeros(length(fnames),12+72);
    wb = waitbar_text(0);
    for i = 1:length(fnames)
        waitbar_text(i/length(fnames), wb);

        [img, ~, alpha] = imread(fnames{i});
        % img = repmat(alpha, [1, 1, 3]);
        feat = extract_lowlev_feature(img, alpha, bkg_color, calib_file);

        output(i,1) = feat.obj.abs_size;
        output(i,2) = feat.obj.rectangle_fill;
        output(i,3) = feat.obj.HV_aspect;
        output(i,4) = feat.obj.contour_var;
        output(i,5) = feat.obj.longitudinal_axis;
        output(i,6) = feat.obj.elongation_factor;
        output(i,7:12) = feat.obj.contour_sf;

        x = feat.obj.contour_dist;
        x(isnan(x)) = mean(x(~isnan(x)));
        output(i,13:end) = x;
    end
    waitbar_text('close', wb);
end
