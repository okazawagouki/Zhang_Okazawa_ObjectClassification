function lowlevelmodel_setup()
%LOWLEVELMODEL_SETUP Add the lowlevelmodel_matlab libraries to the MATLAB path.
%   Call once before using feature_v1 / feature_LUV / feature_gistSF /
%   feature_texture / feature_contour. The feature functions also call this
%   automatically, so calling it from an entry script is optional.
%
%   Priority (highest first) mirrors the resolution of the original
%   function_matlab layout, so results are numerically identical:
%     lib/texture/matlabPyrTools  beats  lib/texture/textureSynth
%     (e.g. matlabPyrTools' buildSCFpyr.m wins over textureSynth's copy),
%   and the .m files in each matlabPyrTools beat the MEX binaries in its
%   MEX/ subfolder (matching the original, where histo.m etc. were used).
%   addpath() prepends, so paths are added in reverse priority order.

    here = fileparts(mfilename('fullpath'));

    % lowest priority: MEX fallbacks
    addpath(fullfile(here, 'lib', 'texture', 'textureSynth', 'MEX'));
    addpath(fullfile(here, 'lib', 'lowlev', 'matlabPyrTools', 'MEX'));
    addpath(fullfile(here, 'lib', 'texture', 'matlabPyrTools', 'MEX'));

    % textureSynth (Portilla-Simoncelli): below matlabPyrTools
    addpath(fullfile(here, 'lib', 'texture', 'textureSynth'));

    % pyramid toolbox: lowlev copy is a fallback for the texture copy
    % (identical .m files; the texture copy additionally ships range2.mexw64)
    addpath(fullfile(here, 'lib', 'lowlev', 'matlabPyrTools'));
    addpath(fullfile(here, 'lib', 'texture', 'matlabPyrTools'));

    % highest priority: the feature libraries themselves
    addpath(fullfile(here, 'lib', 'texture'));  % CtextureAnalysis.m, PSparam2vector.m
    addpath(fullfile(here, 'lib', 'gist'));     % LMgist.m
    addpath(fullfile(here, 'lib', 'lowlev'));   % extract_lowlev_feature.m, rgb2Luv.m, ...
    addpath(here);
end
