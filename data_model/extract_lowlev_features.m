% extract_lowlev_features.m
% Extract the low-level model features (v1 / LUV / gistSF / texture) for all
% task sets used by the main-figure notebooks, writing them directly to
% data_model/smalltask_features/ and data_model/largetask_features/ with the
% file naming the notebooks expect.
%
% Conventions match the released feature files:
%   small tasks: fnames are task-relative paths (task\rule\...\img.png),
%                category = rule class
%   large tasks: fnames carry the data_set\ prefix, category = rule class
%
% Run from anywhere (paths are resolved from this script's location):
%   matlab -batch "run('data_model/extract_lowlev_features.m')"
% Existing output files are skipped (delete them to re-extract).

here = fileparts(mfilename('fullpath'));      % .../data_model
root = fileparts(here);                       % repository root
addpath(fullfile(here, 'lowlevelmodel_matlab'));
lowlevelmodel_setup();

small_tasks = { ...
    'animate_vs_inanimate',            'animate_vs_inanimate_gen', ...
    'natural_vs_artificial',           'natural_vs_artificial_gen', ...
    'mammal_vs_reptile',               'mammal_vs_reptile_gen', ...
    'electronics_vs_tools',            'electronics_vs_tools_gen', ...
    'human_vs_monkey',                 'human_vs_monkey_gen', ...
    'monkey_vs_mammal',                'monkey_vs_mammal_gen', ...
    'animate_vs_inanimate_THINGS',     'animate_vs_inanimate_THINGS_gen', ...
    'natural_vs_artificial_THINGS',    'natural_vs_artificial_THINGS_gen', ...
    'mammal_vs_nonmammal_THINGS',      'mammal_vs_nonmammal_THINGS_gen', ...
    'electronics_vs_tools_THINGS',     'electronics_vs_tools_THINGS_gen', ...
    'outdoor_vs_indoor',               'outdoor_vs_indoor_gen', ...
    'big_vs_small',                    'big_vs_small_gen', ...
    'big_vs_small_Konkle',             'big_vs_small_Konkle_gen', ...
    'fire related_vs_water related',   'fire related_vs_water related_gen', ...
    'eastern_vs_western',              'eastern_vs_western_gen'};

large_tasks = { ...
    'Large_animate_vs_inanimate', ...
    'Large_mammal_vs_nonmammal', ...
    'Large_natural_vs_artificial'};

models = {'v1', 'LUV', 'gistSF', 'texture'};

funcs = struct( ...
    'v1',      @feature_v1, ...
    'LUV',     @feature_LUV, ...
    'gistSF',  @feature_gistSF, ...
    'texture', @feature_texture);

%% small tasks -> data_model/smalltask_features/<task>_<model>.mat
% folder layout: <root>\data_set\<task>\<rule>\...\<img>.png
for n = 1:numel(small_tasks)
    task_name = small_tasks{n};
    imgfiles = [dir(fullfile(root, 'data_set', task_name, '**', '*.png')); ...
                dir(fullfile(root, 'data_set', task_name, '**', '*.jpg'))];
    if isempty(imgfiles)
        fprintf('skip %s: no images found\n', task_name);
        continue
    end
    [category, fnames, imgpath] = parse_files(imgfiles, root, 5, false);
    extract_models(imgpath, category, fnames, task_name, ...
        fullfile(root, 'data_model', 'smalltask_features'), models, funcs);
end

%% large tasks -> data_model/largetask_features/<task>_<model>.mat
% folder layout: <root>\data_set\<task>\<anchor|stim>\<rule>\...\<img>.png
for n = 1:numel(large_tasks)
    task_name = large_tasks{n};
    imgfiles = dir(fullfile(root, 'data_set', task_name, '**', '*.png'));
    [category, fnames, imgpath] = parse_files(imgfiles, root, 6, true);
    extract_models(imgpath, category, fnames, task_name, ...
        fullfile(root, 'data_model', 'largetask_features'), models, funcs);
end

%% ------------------------------------------------------------------------
function [category, fnames, imgpath] = parse_files(imgfiles, root, rule_idx, with_data_set_prefix)
    category = cell(numel(imgfiles), 1);
    fnames = cell(numel(imgfiles), 1);
    imgpath = cell(numel(imgfiles), 1);
    for i = 1:numel(imgfiles)
        parts = split(strrep(imgfiles(i).folder, '/', '\'), '\');
        % parts: {drive, repo, 'data_set', task, [anchor|stim,] rule, ...}
        category{i} = parts{rule_idx};
        if with_data_set_prefix
            fnames{i} = strjoin([{'data_set', parts{4:end}}], '\');
        else
            fnames{i} = strjoin(parts(4:end), '\');
        end
        fnames{i} = fullfile(fnames{i}, imgfiles(i).name);
        imgpath{i} = fullfile(imgfiles(i).folder, imgfiles(i).name);
    end
end

function extract_models(imgpath, category, fnames, task_name, out_dir, model_names, funcs)
    for m = 1:numel(model_names)
        name = model_names{m};
        out_file = fullfile(out_dir, strcat(task_name, '_', name, '.mat'));
        if exist(out_file, 'file')
            continue
        end
        fprintf('Extracting %s / %s ...\n', task_name, name);
        output = funcs.(name)(imgpath);
        save(out_file, 'category', 'fnames', 'output');
        fprintf('  saved %s (%s)\n', out_file, mat2str(size(output)));
    end
end
