function output = feature_texture(fnames)
%FEATURE_TEXTURE Portilla-Simoncelli texture model features, dim = 740.
%   fnames: cell array of image paths. Output is z-scored column-wise
%   across images.

    lowlevelmodel_setup();

    %% just get texture parameter
    output = zeros(length(fnames),740);
    wb = waitbar_text(0);
    for i = 1:length(fnames)
        waitbar_text(i/length(fnames), wb);

        img = imread(fnames{i});
        % [img, ~, alpha] = imread(fnames{i});
        % img = repmat(alpha, [1, 1, 3]);

        img = double(img(:,:,1))/255;
        img = imresize(img, [256 256]);
        params = CtextureAnalysis(img, 4, 4, 7);
        vec = PSparam2vector(params);
        % its 740 parameter vector for one image
        output(i,:) = vec;

    end
    waitbar_text('close', wb);

    output = zscore(output);

end
