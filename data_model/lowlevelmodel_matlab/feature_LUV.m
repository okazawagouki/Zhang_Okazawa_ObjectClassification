function output = feature_LUV(fnames)
%FEATURE_LUV Luminance & color (L*u*v*) features, dim = 192 (12 stats x 16 segments).
%   fnames: cell array of image paths. Foreground/background is read from
%   the image alpha channel (transparent pixels are filled with bkg_color).

    lowlevelmodel_setup();

    calib_file = fullfile(fileparts(mfilename('fullpath')), 'lib', 'calib_data_04-Jan-2020b.mat');
    bkg_color = [100 100 100];

    %% luminance & color
    output = zeros(length(fnames),192);
    wb = waitbar_text(0);
    for i = 1:length(fnames)
        waitbar_text(i/length(fnames), wb);

        [img, ~, alpha] = imread(fnames{i});
        % img = repmat(alpha, [1, 1, 3]);
        feat = extract_lowlev_feature(img, alpha, bkg_color, calib_file);

        field_names = {'segment_L_mean','segment_u_mean','segment_v_mean',...
            'segment_L_sd','segment_u_sd','segment_v_sd',...
            'segment_L_skew','segment_u_skew','segment_v_skew',...
            'segment_L_kurt','segment_u_kurt','segment_v_kurt',...
        };
        for j = 1:length(field_names)
            output(i,(j-1)*16+1:j*16) = zscore(reshape(feat.col.(field_names{j}),[16 1]));
        end
    end
    waitbar_text('close', wb);
end
