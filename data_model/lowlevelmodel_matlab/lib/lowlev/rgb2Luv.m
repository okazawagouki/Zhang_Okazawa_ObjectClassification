function Luv = rgb2Luv(rgb, calib_file)

Lxy = rgb2Lxy(rgb, calib_file);

img_mode = size(Lxy,3) > 1;

if img_mode
    ims = size(Lxy);
    Lxy = reshape(Lxy, [ims(1)*ims(2), ims(3)]);
end

u = 4 * Lxy(:,2) ./ (-2 * Lxy(:,2) + 12 * Lxy(:,3) + 3);
v = 9 * Lxy(:,3) ./ (-2 * Lxy(:,2) + 12 * Lxy(:,3) + 3);
Luv = [Lxy(: ,1) u v];

if img_mode
    Luv = reshape(Luv, ims);
end
