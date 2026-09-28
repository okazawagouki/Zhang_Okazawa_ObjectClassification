# Image classification battery test: monkey and human data

## Overview

This repository contains the analysis code and shared behavioral/model
data for:

> Zhang H, Zheng Z, Hu J, Wang Q, Xu M, Zhou Z, Li Z, Okazawa G (2026)
> **A battery of image classification challenges reveals shared and
> distinct object categorization behavior across monkeys, humans, and
> deep networks.** *eLife* **15**:RP111725.
> https://doi.org/10.7554/eLife.111725.1

Three rhesus monkeys and a cohort of human participants were trained and
tested on a battery of binary image-classification tasks — animate vs.
inanimate, natural vs. artificial, mammal vs. non-mammal, and more —
using a touchscreen "object drag" paradigm, and their generalization to
novel images was measured. 

## Notes

- For the tasks using the [THINGS image set](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792), you can identify the images from the file name given for each trial in the data.
- For the images used in the other tasks, please contact the authors (see below) to request them.
- This repository includes the behavioral data used in the main figures. Many additional trials were collected beyond these (e.g., trials during the training period, some other categorization tasks); if you are interested, please contact the authors.
- About 0.1% of trials are missing from the data due to incomplete data collection. See the Human data section of `Figure7.ipynb`.




## Code organization

| Path | Content |
|---|---|
| `Figure*.ipynb` | main-figure notebooks, one set per figure (run from the repo root; `FigureN.ipynb` reproduces Figure N) |
| `functions/` | shared helpers: data loading (`load_data.py`, `utils.py`), DDM sensitivity (`sensitivity.py`), model decoding (`decoding.py`), plotting theme (`plotting.py`) |
| `data_behavior/` | monkey/human behavioral session files and session indexes (`Monkey_sessions.xlsx`, `Human_sessions.xlsx`) |
| `data_model/` | model features (low-level, CNN/ViT, VLM, neural V1/V4/IT) + extraction scripts |
| `data_DDMfit/` | drift-diffusion model fit results, per subject and task |
| `results/` | notebook outputs (figure PDFs / CSVs), organized by figure |

- Environment: Python 3.12 — `pip install -r requirements.txt`.
- Run the notebooks from the repository root, in any order; each writes its
figures into `results/Figure1/` ... `results/Figure7/`. 
- Stimulus images are not included; the scripts in `data_model/` regenerate the features once images are supplied under `data_set/`.

## Dataset

- `data_behavior/` holds the raw session files (`*_fmt.mat`), one per subject/day/task, indexed by `Monkey_sessions.xlsx` and `Human_sessions.xlsx`. 
- Tasks are listed separately by learning phase — e.g. `animate_vs_inanimate` (training/learning sessions) vs. `animate_vs_inanimate_gen` (generalization sessions run after learning, on held-out images) are distinct rows. 
- Generalization sessions include both trials presenting trained and generalization images. You can distinguish them with the `trial_type` field (`main` = trained image, `gen` = generalization image).


### Monkey

| Task | Subjects | Trials | Figure |
|---|---:|---:|---:|
| Animate vs. Inanimate (`animate_vs_inanimate`) | 3 | 14,445 | 1C |
| Natural vs. Artificial (`natural_vs_artificial`) | 3 | 10,792 | 1H |
| Mammal vs. Non-mammal (`mammal_vs_reptile`) | 3 | 10,652 | 1H |
| Monkey vs. Mammal (`monkey_vs_mammal`) | 3 | 9,688 | 1H |
| Human vs. Monkey (`human_vs_monkey`) | 3 | 11,446 | 1H |
| Tool vs. Electronics (`electronics_vs_tools`) | 3 | 13,724 | 1H |
| Animate vs. Inanimate (generalization) (`animate_vs_inanimate_gen`) | 3 | 5,493 | 1I |
| Natural vs. Artificial (generalization) (`natural_vs_artificial_gen`) | 3 | 2,774 | 1I |
| Mammal vs. Non-mammal (generalization) (`mammal_vs_reptile_gen`) | 3 | 2,993 | 1I |
| Monkey vs. Mammal (generalization) (`monkey_vs_mammal_gen`) | 3 | 2,761 | 1I |
| Human vs. Monkey (generalization) (`human_vs_monkey_gen`) | 3 | 2,538 | 1I |
| Tool vs. Electronics (generalization) (`electronics_vs_tools_gen`) | 3 | 4,155 | 1I |
| Animate vs. Inanimate (THINGS) (`Large_animate_vs_inanimate`) | 3 | 26,347 | 2B |
| Mammal vs. Non-mammal (THINGS) (`Large_mammal_vs_nonmammal`) | 3 | 13,067 | 2B |
| Natural vs. Artificial (THINGS) (`Large_natural_vs_artificial`) | 3 | 14,720 | 2B |
| Animate vs. Inanimate (cartoon control) (`animate_vs_inanimate_cartoon`) | 3 | 4,811 | 2D |
| Animate vs. Inanimate (silhouette control) (`animate_vs_inanimate_silhouette`) | 3 | 5,283 | 2D |
| Mammal vs. Non-mammal (THINGS, texture-pixelated control) (`Large_mammal_vs_nonmammal_pixelate`) | 3 | 5,030 | 2D |
| Natural vs. Artificial (THINGS, grayscale control) (`Large_natural_vs_artificial_gray`) | 3 | 4,717 | 2D |
| Animate vs. Inanimate (texform, initial generalization test) (`animate_vs_inanimate_texform1`) | 3 | 2,888 | 3B |
| Animate vs. Inanimate (texform, training) (`animate_vs_inanimate_texform2`) | 3 | 19,425 | 3B |
| Animate vs. Inanimate (texform, generalization to novel texform) (`animate_vs_inanimate_texform3`) | 3 | 3,060 | 3B |
| Randomization control: category-level target association (training) (`randomization_test_L2`) | 3 | 12,513 | 3G |
| Randomization control: image-level target association (training only) (`randomization_test_L3`) | 3 | 14,149 | 3G |
| Randomization control: category-level target association (generalization) (`randomization_test_L2_gen`) | 3 | 5,592 | 3H |
| Outdoor vs. Indoor (`outdoor_vs_indoor_gen`) | 3 | 3,291 | 4B |
| Big vs. Small (THINGS) (`big_vs_small_gen`) | 3 | 2,916 | 4B |
| Big vs. Small (Konkle) (`big_vs_small_Konkle_gen`) | 3 | 3,040 | 4B |
| Fire- vs. Water-related (`fire related_vs_water related_gen`) | 3 | 3,185 | 4B |
| Western vs. Eastern culture (`eastern_vs_western_gen`) | 3 | 3,171 | 4B |
| Animate vs. Inanimate (outline control) (`animate_vs_inanimate_outline`) | 3 | 5,328 | 6G |
| Tool vs. Electronics (THINGS) (`electronics_vs_tools_THINGS_gen`) | 3 | 2,872 | 7A |

### Human

| Task | Subjects | Trials | Figure |
|---|---:|---:|---:|
| Animate vs. Inanimate (`animate_vs_inanimate`) | 7 | 10,998 | 5A |
| Natural vs. Artificial (`natural_vs_artificial`) | 7 | 11,300 | 5A |
| Mammal vs. Non-mammal (`mammal_vs_reptile`) | 7 | 11,100 | 5A |
| Monkey vs. Mammal (`monkey_vs_mammal`) | 7 | 11,209 | 5A |
| Human vs. Monkey (`human_vs_monkey`) | 7 | 10,500 | 5A |
| Tool vs. Electronics (`electronics_vs_tools`) | 7 | 11,200 | 5A |
| Animate vs. Inanimate (generalization) (`animate_vs_inanimate_gen`) | 9 | 2,624 | 7A |
| Natural vs. Artificial (generalization) (`natural_vs_artificial_gen`) | 9 | 2,382 | 7A |
| Mammal vs. Non-mammal (generalization) (`mammal_vs_reptile_gen`) | 9 | 2,177 | 7A |
| Monkey vs. Mammal (generalization) (`monkey_vs_mammal_gen`) | 9 | 1,979 | 7A |
| Human vs. Monkey (generalization) (`human_vs_monkey_gen`) | 9 | 1,743 | 7A |
| Tool vs. Electronics (generalization) (`electronics_vs_tools_gen`) | 9 | 2,441 | 7A |
| Animate vs. Inanimate (THINGS) (`Large_animate_vs_inanimate`) | 9 | 1,938 | 7A |
| Mammal vs. Non-mammal (THINGS) (`Large_mammal_vs_nonmammal`) | 9 | 1,976 | 7A |
| Natural vs. Artificial (THINGS) (`Large_natural_vs_artificial`) | 9 | 1,952 | 7A |
| Outdoor vs. Indoor (`outdoor_vs_indoor_gen`) | 9 | 2,316 | 4B |
| Big vs. Small (THINGS) (`big_vs_small_gen`) | 9 | 2,018 | 4B |
| Big vs. Small (Konkle) (`big_vs_small_Konkle_gen`) | 9 | 2,079 | 4B |
| Fire- vs. Water-related (`fire related_vs_water related_gen`) | 9 | 1,980 | 4B |
| Western vs. Eastern culture (`eastern_vs_western_gen`) | 9 | 2,010 | 4B |
| Tool vs. Electronics (THINGS) (`electronics_vs_tools_THINGS_gen`) | 9 | 1,944 | 7A |

## Data format

Each task/session is indexed in `Monkey_sessions.xlsx` / `Human_sessions.xlsx`
(sheet `Sessions`) and stored as one raw session file per row.

- **Session index** — columns `task`, `subject`, `filename`, and (Monkey only)
  `learned` (1 if the session met the learning criterion for that task).


### Data loader

`functions/load_data.py` provides loaders that
read the session index and trial files directly: `loadTrials` loads one
session file, already restricted to responded, two-target, non-correction-loop
trials, and `loadTrialsM` / `loadTrialsH` pool trials across sessions for a
given task and subject.

```python
from functions.load_data import session_table, loadTrialsM, loadTrialsH

# session_table() reads Monkey_sessions.xlsx / Human_sessions.xlsx
monkey_sessions = session_table('Monkey')
print(monkey_sessions[['task', 'subject']].drop_duplicates())

# Pick any (task, subject) pair from that table and load its trials.
# islearn=True (default) keeps only sessions that met the learning
# criterion for that task.
task, monkey = monkey_sessions.iloc[0][['task', 'subject']]
trials, session_files = loadTrialsM(task, monkey)

# Same pattern for human subjects
human_sessions = session_table('Human')
task, human = human_sessions.iloc[0][['task', 'subject']]
trials, session_files = loadTrialsH(task, human)

# Both loaders return trials, session_files:
print(len(trials), 'trials from', len(session_files), 'sessions')
t = trials[0]
print(t['img_url'], t['response'], t['targ_cor'], t['result'])
```

### Parameters

Main parameters

- `img_ID`: stimulus image ID in this task
- `img_url`: image path and name. The leading directory is the canonical task id (matching the session index); the directories below it keep the original stimulus-tree layout — category folders and the `_gen` markers that identify novel-image sets
- `category`: the task's two category names, in a fixed order (index 0/1 = category ID 1/2)
- `targ_category` / `targ_cat`: the two category names in target-position order (index 0 = target 1, index 1 = target 2). `targ_category` is used for the non-THINGS tasks, `targ_cat` for the THINGS/`Large_*` tasks
- `targ_cat_ID`: for each target position, the category ID (1 or 2, indexing into `category`) assigned to it.
- `targ_cor`: correct target (1 or 2; corresponding to `targ_category`/`targ_cat`)
- `response`: subject's choice (1 or 2; corresponding to `targ_category`/`targ_cat`)
- `result`: trial outcome, `CORRECT` or `WRONG` (whether `response` matched `targ_cor`)
- `rt`: reaction time (from stim on to choice)
- `trial_type`: `main` (trained image) or `gen` (generalization / novel image)

Other relevant parameters

- `hide_wrg_target`: whether the wrong target was hidden (a training-period manipulation). These trials existed in training sessions only, and the data loader skips them by default.
- `correction_loop_flg`: if subjects' choices are too biased, this will be turned on to show the other category until the bias is fixed. The data loader removes these trials by default.



## Contact

For questions or further inquiries about the dataset and code, please contact the first or corresponding author (zhangh2022@ion.ac.cn, okazawa@ion.ac.cn).

