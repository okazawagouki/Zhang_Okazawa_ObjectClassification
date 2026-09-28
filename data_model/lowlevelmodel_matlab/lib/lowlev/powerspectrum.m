function [sf nps] = powerspectrum(f, d, ns)
%% POWERSPECTRUM Get the normalized power spectrum of an image
%
%   [SF NPS] = powerspectrum(F) constructs the power spectrum of F by
%   averaging profiles of the power spectrum image centered at spatial 
%   frequency 0 mm^-1 and rotated by 1 degree increments. 
%
%   [SF NPS] = powerspectrum(F, D) takes as an additional parameter a two
%   element vector representing pixel size.
%
%   [SF NPS] = powerspectrum(F, D, NS) will do the above but take NS spatial
%   frequency samples.
%
%   Example:
%       f = imread('moon.tif');
%       [sf nps] = powerspectrum(f);
%       semilogy(sf, nps);

% Pixel dimensions
if nargin < 2
    d = [1 1];
end

% Adjust for nonsquare pixels
nd = 1./d;
di = nd/max(nd);

% Number of samples per line
if nargin < 3
    ns = 200;
end

% Compute the power spectrum
F = fft2(double(f));
F = fftshift(F);
S = abs(F).^2;

% Coordinates of DC term
p = size(S)/2;

% Sampling matrix
PS = zeros(360, ns);

% Sample along lines rotated about DC term
for theta = 1:360
    u = [cosd(theta) sind(theta)];
    q = p.*(1 + u).*di;
    PS(theta,:) = linesample(S, p, q, ns);
end

% Average and normalize
ps = mean(PS,1);
nps = ps/max(ps);

% Spatial frequencies
sfu = 1/(2*min(d)); % Nyquist criteria
sfi = sfu/ns;
sf = 0:sfi:sfu-sfi;

if nargout == 0
    semilogy(sf, nps);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = linesample(f, p, q, n, method)
%% LINESAMPLE Sample image intensities along a line 
%
%   V = linesample(F, P, Q) will sample the image F between points P and Q. 
%
%   V = linesample(F, P, Q, N) will do the above but take N number of evenly
%   spaced samples. 
%
%   V = linesample(..., METHOD) will do the above but use the specified
%   interpolation method. Bilinear (the default) and nearest neighbor
%   interpolation are currently supported.

% Calc number of samples, if necessary
if nargin < 4
    n = round(sqrt(sum((p-q).^2)));
end

% Calculate step size
step = (q-p)/n;

% Get points at which to sample
for d = 1:2 
    if p(d) == q(d)
        x{d} = p(d)*ones(1,n);
    else
        x{d} = p(d):step(d):q(d);
    end
end

% Sample 
if nargin < 5 || strcmp(method, 'bilinear')
    v = bilinear(f, x{1}(1:n), x{2}(1:n));
else 
    v = nearestneighbor(f, x{1}(1:n), x{2}(1:n));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function zi = bilinear(f, xi, yi)
%% Bilinear interpolation

d = size(f);

xu = ceil(xi);
xl = floor(xi);
yu = ceil(yi);
yl = floor(yi);

% Fix out of range subscripts
xu(xu > d(1)) = d(1);
xl(xl == 0) = 1;
yu(yu > d(2)) = d(2);
yl(yl == 0) = 1;

xuyu = sub2ind(d, xu, yu);
xlyu = sub2ind(d, xl, yu);
xuyl = sub2ind(d, xu, yl);
xlyl = sub2ind(d, xl, yl);

a = (xi-xl).*(f(xuyu)-f(xlyu))+f(xlyu);
b = (xi-xl).*(f(xuyl)-f(xlyl))+f(xlyl);

zi = (yi-yl).*(a-b)+b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function zi = nearestneighbor(f, xi, yi)
%% Nearest neighbor interpolation

d = size(f);

x = round(xi);
y = round(yi);

x(x > d(1)) = d(1);
x(x == 0) = 1;
y(y > d(2)) = d(2);
y(y == 0) = 1;

idx = sub2ind(size(f), x, y);
zi = f(idx);