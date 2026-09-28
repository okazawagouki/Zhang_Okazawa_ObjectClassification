function [img, alpha] = obj_image_resizing(img, alpha, target_size, margin)
% function [img, alpha] = obj_image_resizing(img, alpha, target_size, margin)
%  clip object image based on alpha and resize it to a requested size
%  target_size = [w, h] (pixel)
%  margin = [w, h] (pixel)

cdim = size(img,3);

ctarget_size = target_size - margin * 2; % remove margin
if any(ctarget_size < 0)
    error('margin size is too large');
end

if isempty(alpha)
    alpha = uint8(ones(size(img,1), size(img, 2)) * 255);
end

% clip
x_rng = [find(any(alpha,1), 1, 'first'), find(any(alpha,1), 1, 'last')];
y_rng = [find(any(alpha,2), 1, 'first'), find(any(alpha,2), 1, 'last')];
img = img(y_rng(1):y_rng(2), x_rng(1):x_rng(2), :);
alpha = alpha(y_rng(1):y_rng(2), x_rng(1):x_rng(2));

% rescale image
aspect = min(ctarget_size ./ [size(img, 2), size(img, 1)]);
img = imresize(img, aspect);
alpha = imresize(alpha, aspect);

% adjust margin size
if ctarget_size(1) > size(img,2)
    margin(1) = margin(1) + (ctarget_size(1) - size(img,2))/2;
end
if ctarget_size(2) > size(img,1)
    margin(2) = margin(2) + (ctarget_size(2) - size(img,1))/2;
end


% add margin
s1 = size(img,1);
img = cat(2, zeros(s1, floor(margin(1)),cdim), img, zeros(s1, ceil(margin(1)),cdim));
alpha = cat(2, zeros(s1, floor(margin(1))), alpha, zeros(s1, ceil(margin(1))));
s2 = size(img,2);
img = cat(1, zeros(floor(margin(2)), s2, cdim), img, zeros(ceil(margin(2)), s2, cdim));
alpha = cat(1, zeros(floor(margin(2)), s2), alpha, zeros(ceil(margin(2)), s2));

if size(img,1) == target_size(1) + 1
    img = img(2:end, :, :);
    alpha = alpha(2:end,:);
end
if size(img,2) == target_size(2) + 1
    img = img(:, 2:end, :);
    alpha = alpha(:, 2:end);
end

if size(img,1) ~= target_size(1) || size(img,2) ~= target_size(2)
    error('failed to generate image with the specified size.');
end








