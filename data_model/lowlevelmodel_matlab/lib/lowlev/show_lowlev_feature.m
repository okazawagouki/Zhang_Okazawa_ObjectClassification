function fh = show_lowlev_feature(img, alpha, bkg_color, feat)

fh = figure('pos', [100 100 700 400]);


subplot(2, 3, 1); % image
if isempty(alpha)
    alpha = ones(size(img,1), size(img,2));
else
    alpha = double(alpha)/255;
end
img(:,:,1) = uint8(double(img(:,:,1)) .* alpha + (1 - alpha) * bkg_color(1));
img(:,:,2) = uint8(double(img(:,:,2)) .* alpha + (1 - alpha) * bkg_color(2));
img(:,:,3) = uint8(double(img(:,:,3)) .* alpha + (1 - alpha) * bkg_color(3));

imshow(img);

subplot(2, 3, 2); % luminance info

txt = {sprintf('Luminance: mean %1.1f, SD %1.1f', feat.col.mean(1), feat.col.sd(1)), ...
       sprintf('Luminance: skew %1.1f, kurt %1.1f', feat.col.skew(1), feat.col.kurt(1)), ...
       sprintf('Luv u: mean %1.1f, SD %1.1f', feat.col.mean(2), feat.col.sd(2)), ...
       sprintf('Luv v: mean %1.1f, SD %1.1f', feat.col.mean(3), feat.col.sd(3))};

for n=1:length(txt)
    text(0, n, txt{n});
end
set(gca, 'ylim', [0 length(txt)+3], 'ydir', 'reverse', 'visible', 'off');

subplot(2, 3, 3); % spatial frequency info

sf_mag = feat.sf_mag;
scaling = 2.^ (2:2:(size(sf_mag,2)*2));
% scaling = [1 1 1 1];

pies(bsxfun(@rdivide, sf_mag, scaling));

subplot(2, 3, 4); % contour map
hold on;
plot(feat.norm_contour(:,1), feat.norm_contour(:,2), 'k');
axis equal;
set(gca, 'visible', 'off');

lax = feat.obj.longitudinal_axis/180*pi;
[x, y] = pol2cart([lax; lax + pi], [1 1] * max(abs(feat.norm_contour(:))) * 1.2);
plot(x, y, 'r', 'linew', 2);


subplot(2, 3, 5); % obj info

txt = {sprintf('Absolute size: %1.2f', feat.obj.abs_size), ...
       sprintf('Rectangle fill: %1.2f', feat.obj.rectangle_fill), ...
       sprintf('H/V aspect ratio: %1.2f', feat.obj.HV_aspect), ...
       sprintf('Contour variance: %1.4f', feat.obj.contour_var), ...
       sprintf('Longitudinal axis: %1.1f', feat.obj.longitudinal_axis), ...
       sprintf('Elongation factor: %1.2f', feat.obj.elongation_factor)};

for n=1:length(txt)
    text(0, n, txt{n});
end
set(gca, 'ylim', [0 length(txt)+3], 'ydir', 'reverse', 'visible', 'off');

subplot(2, 3, 6); % contour spatial frequency

nsf = length(feat.obj.contour_sf);

bar(1:nsf, feat.obj.contour_sf, 'facecolor', 'none', 'edgecolor', 'k');

format_panel(gca, 'xtick', 1:nsf, 'xticklabel', 1:nsf, 'xlabel', 'Spatial frequency (low to high)', ...
    'ylabel', 'Power');




end



function pies(c, offset)
    %
    % x = ori x scale
    % c = ori x scale
    % scale -> small to large
    % ori -> default upper left

    
    c = rot90([c;c],2);
    x = ones(size(c));
    bin = 1/size(x,2);
    si = (bin:bin:1);
    
    c = (c - min(c(:)))/(max(c(:)) - min(c(:)));
    col = cell(size(c));
    for i=1:length(c(:))
        col{i} = [1 1 1] - [0 1 1] * c(i);
    end

    hold on;
    for n=size(x,2):-1:1
        h = pie(x(:,n));
        for m = 1:2:length(h)
            set(h(m), 'Vertices', get(h(m), 'Vertices')*si(n));
            set(h(m), 'FaceColor', col{(m-1)/2+1,n});
        end
        for m = 2:2:length(h)
            set(h(m), 'Visible', 'off');
        end
    end

    if ~exist('offset', 'var')
        offset = 22.5 + 45;
    end
    view(offset, 90);
    set(gca, 'Visible', 'off');
    axis square;

end