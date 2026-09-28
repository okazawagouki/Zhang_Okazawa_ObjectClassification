function vec = PSparam2vector(PS)


%%%%%%%% marginal statistics and mean magnitude
vec1 = [PS.pixelStats([3;4])'; PS.pixelLPStats(:); PS.magMeans(:); PS.varianceHPR];


%%%%%%%% raw auto correlation
vec2 = reshape(PS.autoCorrReal, 7*7, 5);
vec2 = vec2(1:25,:); % remove duplicated num
vec2 = vec2(:);

%%%%%%%% cross correlation of magnitude
m1 = []; % auto mag
for n=1:4
    for m=1:4
        a = PS.autoCorrMag(:,:,n,m);
        m1 = [m1; a(1:25)']; %#ok<AGROW>
    end
end
vec5 = m1;

a = PS.cousinMagCorr(:,:,1:4); % cousin
m2 = [];
for n=1:4
    b = a(:,:,n);
    m2 = [m2; b([1 2 3 4 6 7 8 11 12 16])']; %#ok<AGROW>
end

m3 = PS.parentMagCorr(:,:,1:3); % parent
m3 = m3(:);

vec3 = [m2;m3];

%%%%%%% cross correlation of phase

a = PS.parentRealCorr(1:4,:,1:3);
vec4 = a(:);

vec = [vec1; vec2; vec3; vec4; vec5];
% uPS.marginal_stat = vec1;
% uPS.raw_auto_corr = vec2;
% uPS.auto_corr_mag = vec5;
% uPS.cross_corr_mag = vec3;
% uPS.cross_corr_phase = vec4;


    

