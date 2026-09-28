function p1 = average_param(p1, p2, ratio)

fi = fieldnames(p1);

for n=1:length(fi)
    p1.(fi{n}) = p1.(fi{n}) * ratio + p2.(fi{n}) * (1 - ratio);
end


%  pixelStats: [0.4341 0.0096 -1.1119 4.4824 0.0116 0.7036]
%       pixelLPStats: [5×2 double]
%       autoCorrReal: [7×7×5 double]
%        autoCorrMag: [7×7×4×4 double]
%           magMeans: [18×1 double]
%      cousinMagCorr: [4×4×5 double]
%      parentMagCorr: [4×4×4 double]
%     cousinRealCorr: [8×8×5 double]
%     parentRealCorr: [8×8×4 double]
%        varianceHPR: 1.1330e-04

