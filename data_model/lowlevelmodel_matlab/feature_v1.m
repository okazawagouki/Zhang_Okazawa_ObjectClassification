function output = feature_v1(fnames)
%FEATURE_V1 V1-like model features (Pinto et al., 2008), dim = 96*30*30.
%   fnames: cell array of image paths.
%
%   Optimized (bit-comparable to the original implementation up to float
%   round-off, ~1e-12): the 96 Gabor filters and the pooling window index
%   vectors are built once per call; images are processed in parfor when
%   the Parallel Computing Toolbox is available; the 17x17 average-pooling
%   loops are replaced by an integral-image computation.

    lowlevelmodel_setup();

    dim = 96*30*30;
    batchsize = length(fnames);
    output = zeros(batchsize, dim);

    % constant across images: filter bank and pooling windows
    gaborFilters = build_gabor_filters();
    [lr, rr] = pooling_windows();

    use_parfor = ~isempty(ver('parallel'));
    if use_parfor
        parfor i = 1:batchsize
            feat = v1_one(fnames{i}, gaborFilters, lr, rr);
            output(i,:) = reshape(feat, 1, dim);
        end
    else
        for i = 1:batchsize
            feat = v1_one(fnames{i}, gaborFilters, lr, rr);
            output(i,:) = reshape(feat, 1, dim);
        end
    end

end


function gaborFilters = build_gabor_filters()
% the 96 Gabor filters (6 frequencies x 16 orientations), 43x43 each.
% NOTE: the grid is the half-integer -21.5:21.5 (43 points, no integer
% center), exactly as in the original implementation — do NOT "simplify"
% it to -21:21, that shifts the filter phase.

    filterSize = [43 43];
    numOrientations = 16;
    numFrequencies = 6;
    frequencies = [1/2, 1/3, 1/4, 1/6, 1/11, 1/18];
    sigma = 9;

    [X, Y] = meshgrid(-(filterSize(2)/2):filterSize(2)/2, ...
                      -(filterSize(1)/2):filterSize(1)/2);
    G = exp(-(X.^2 + Y.^2) / (2 * sigma^2));

    gaborFilters = cell(1, numOrientations * numFrequencies);
    for f = 1:numFrequencies
        for o = 1:numOrientations
            orientation = (o - 1) * (2 * pi / numOrientations);
            S = sin(2 * pi * frequencies(f) * X .* cos(orientation) ...
                    + 2 * pi * frequencies(f) * Y .* sin(orientation));
            GF = G .* S;
            GF = GF - mean(GF);
            % NOTE: norm(GF) on a matrix is the 2-norm (largest singular
            % value) — keep it exactly as the original implementation
            GF = GF / norm(GF);
            gaborFilters{(f - 1) * numOrientations + o} = GF;
        end
    end
end


function [lr, rr] = pooling_windows()
% row/col window bounds of the 17x17 stride-5 average pooling on a 150x150
% map (centers 3, 8, ..., 148; borders clipped), as in the original loops.

    cs = 3 + (0:29) * 5;
    lr = max(cs - 8, 1);
    rr = min(cs + 8, 150);
end


function feat = v1_one(fname, gaborFilters, lr, rr)
% V1-like model for ONE image (Pinto et al., 2008).

%% Image preparation
img = imread(fname);
gray = im2gray(img);

[s1,~] = size(gray);
scale = 150/s1;
gray = imresize(gray, scale, 'bicubic');
gray = zscore(double(gray));

%% Local input divisive normalization
% Assume image's size is 150*150. Image is divided into 50*50 blocks
for i = 1:50
    for j = 1:50
        block = gray((i-1)*3+1:i*3,(j-1)*3+1:j*3);
        local_mean = mean(block);
        local_norm = norm(block);
        block = (block - local_mean);
        % we divided this value by the euclidean norm of the resulting
        % 9-dimensional vector (3 3 3 window) if the norm was greater than 1
        % (i.e., roughly speaking, the normalization was constrained
        % such that it could reduce responses, but not enhance them)
        if local_norm > 1
            block = block / local_norm;
        end
        gray((i-1)*3+1:i*3,(j-1)*3+1:j*3) = block;
    end
end

%% Linear filtering with the set of Gabor filters
im = zeros(96,150,150);
for i = 1:96
    im(i,:,:) = conv2(gray,gaborFilters{i},'same');
end

%% Thresholding and saturation
im(im<0) = 0;
im(im>0) = 1;

%% Local output divisive normalization
for i = 1:50
    for j = 1:50
        block = im(:,(i-1)*3+1:i*3,(j-1)*3+1:j*3);
        local_mean = mean(reshape(block,864,1));
        local_norm = norm(reshape(block,864,1));
        block = (block - local_mean);

        if local_norm > 1
            block = block / local_norm;
        end
        im(:,(i-1)*3+1:i*3,(j-1)*3+1:j*3) = block;
    end
end

%% Dimensionality reduction
% First, avgpooling 150x150 -> 30X30, with a 17x17 window (borders
% clipped). Vectorized with an integral image: window sum =
% II(rr,tr) - II(lr-1,tr) - II(rr,br-1) + II(lr-1,br-1), then divided by
% the (clipped) window size.
II = zeros(96,151,151);
% dim 2 = rows, dim 3 = cols (dim 1 is the 96 filters) — cumsum must run
% over the two SPATIAL dims only
II(:,2:end,2:end) = cumsum(cumsum(im,2),3);
S = II(:,rr+1,rr+1) - II(:,lr,rr+1) - II(:,rr+1,lr) + II(:,lr,lr);
cnt = reshape((rr - lr + 1)' * (rr - lr + 1), 1, 30, 30);  % 1x30x30 window sizes
feat = S ./ cnt;                        % 96x30x30

end
