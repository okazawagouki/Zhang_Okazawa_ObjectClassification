"""Shared DDM sensitivity loading for the Figure6 family of notebooks.

All loaders read ``data_DDMfit/{species}/{subject}_{task}_fit.mat`` and extract
the fitted per-image drift rate k. Sign convention (verified on the fit
files): the raw fit has k < 0 for images whose correct target is category 1
(~90-100% of images) and k > 0 for correct-target-2 images, so every loader
flips k (``k[utarg == 1] *= -1``) by default, giving

    positive k = drift toward the image's CORRECT category
    negative k = drift toward the WRONG category

Cross-subject averages should use these flipped, SIGNED values; taking abs()
would discard the wrong-direction information. ``flip=False`` returns the raw
fit values for the few analyses that need the raw convention (e.g. FigureS6
sorts images by raw k and re-orients the two categories afterwards).
"""

import glob
import os

import numpy as np
import pandas as pd
import scipy.io as scio

from functions.utils import index_in

MONKEYS = ["Judy", "Elsa", "Olaf"]

# Small-task display name -> DDM fit task name (file stem in data_DDMfit/).
TRAIN_TASK_MAP = {
    "Animate vs. Inanimate": "animate_vs_inanimate",
    "Natural vs. Artificial": "natural_vs_artificial",
    "Mammal vs. Non-mammal": "mammal_vs_reptile",
    "Tools vs. Electronics": "electronics_vs_tools",
    "Human vs. Monkey": "human_vs_monkey",
    "Monkey vs. Mammal": "monkey_vs_mammal",
}


def extract_image_basename(fname):
    """Image basename from a stored image path (accepting '\\' separators)."""
    return str(fname).strip().split("\\")[-1].strip()


def load_fit_file(path, flip=True):
    """Read one DDM fit file -> (utarg, uimg, k).

    k is the per-image drift rate; utarg the per-image correct-target flag.
    With flip=True, k[utarg == 1] is sign-flipped so that positive k = drift
    toward the image's CORRECT category (see module docstring).

    The shared fit files store
    fitresult with plain fields: uimg_ID (Nx1 cellstr), uimg_targ_cor
    (Nx1 double) and model_param.final (1x1 cell holding the k vector).
    """
    fr = scio.loadmat(path)["fitresult"]
    utarg = np.asarray(fr["uimg_targ_cor"][0, 0]).ravel().astype(float)
    uimg = np.array([str(np.asarray(x).ravel()[0])
                     for x in np.asarray(fr["uimg_ID"][0, 0]).ravel()],
                    dtype=object)
    k = np.asarray(fr["model_param"][0, 0]["final"][0, 0]).ravel().astype(float)
    if flip:
        k[utarg == 1] = k[utarg == 1] * -1
    return utarg, uimg, k


def load_subject_fit(name, task, species="human", flip=True):
    """(utarg, uimg, k) for one subject of one task."""
    path = os.path.join("data_DDMfit", species, f"{name}_{task}_fit.mat")
    return load_fit_file(path, flip=flip)


def _align_to_first(uimg_list, extra_lists):
    """Keep only images present in the first subject's set, reordered to it.

    uimg_list is updated in place; each array in extra_lists (one per subject)
    is filtered / reordered in the same way.
    """
    templ = uimg_list[0]
    for i, images in enumerate(uimg_list):
        idx = np.array([img in templ for img in images])
        images = images[idx]
        kept = [vals[i][idx] for vals in extra_lists]

        idx = index_in(images, templ)
        uimg_list[i] = images[idx]
        for vals, arr in zip(extra_lists, kept):
            vals[i] = arr[idx]


def load_human_fits(task, human_list, flip=True):
    """Per-subject (uimg, k) lists for a human task.

    Subjects without a fit file are skipped; every subject is aligned to the
    image set of the first subject that has a fit.
    """
    uimg_list, k_list = [], []
    for human in human_list:
        file = os.path.join("data_DDMfit", "human", f"{human}_{task}_fit.mat")
        if os.path.isfile(file):
            _, uimg, k = load_fit_file(file, flip=flip)
            uimg_list.append(uimg)
            k_list.append(k)

    _align_to_first(uimg_list, [k_list])
    return uimg_list, k_list


def load_monkey_fits(task, flip=True):
    """Per-monkey (uimg, k) lists.

    Monkeys without a fit file are skipped; image sets are NOT aligned across
    monkeys (the canonical Figure6 usage aligns humans only).
    """
    uimg_list, k_list = [], []
    for monkey in MONKEYS:
        file = os.path.join("data_DDMfit", "monkey", f"{monkey}_{task}_fit.mat")
        if os.path.isfile(file):
            _, uimg, k = load_fit_file(file, flip=flip)
            uimg_list.append(uimg)
            k_list.append(k)
    return uimg_list, k_list


def load_human_fits_full(task, human_list, flip=True):
    """load_human_fits, also returning the aligned per-subject utarg lists:
    (utarg_list, uimg_list, k_list)."""
    utarg_list, uimg_list, k_list = [], [], []
    for human in human_list:
        file = os.path.join("data_DDMfit", "human", f"{human}_{task}_fit.mat")
        if os.path.isfile(file):
            utarg, uimg, k = load_fit_file(file, flip=flip)
            utarg_list.append(utarg)
            uimg_list.append(uimg)
            k_list.append(k)

    _align_to_first(uimg_list, [utarg_list, k_list])
    return utarg_list, uimg_list, k_list


def load_monkey_fits_full(task, flip=True):
    """load_monkey_fits, also returning the per-monkey utarg lists:
    (utarg_list, uimg_list, k_list)."""
    utarg_list, uimg_list, k_list = [], [], []
    for monkey in MONKEYS:
        file = os.path.join("data_DDMfit", "monkey", f"{monkey}_{task}_fit.mat")
        if os.path.isfile(file):
            utarg, uimg, k = load_fit_file(file, flip=flip)
            utarg_list.append(utarg)
            uimg_list.append(uimg)
            k_list.append(k)
    return utarg_list, uimg_list, k_list


def mean_sensitivity(task, species="monkey", subjects=None, flip=True):
    """Cross-subject mean SIGNED k per image -> DataFrame(image, sensitivity).

    k is flipped per subject first (positive = toward the CORRECT category,
    see module docstring), then averaged per image across subjects. Images are
    matched by basename and unioned across subjects.

    subjects=None uses all MONKEYS for ``species="monkey"``, or every subject
    with a fit file for ``species="human"``. Returns None if no fit file could
    be loaded.
    """
    fit_task = TRAIN_TASK_MAP.get(task)
    if fit_task is None:
        return None

    if subjects is not None:
        root = os.path.join("data_DDMfit", species)
        paths = [os.path.join(root, f"{s}_{fit_task}_fit.mat") for s in subjects]
    elif species == "monkey":
        paths = [os.path.join("data_DDMfit", "monkey", f"{m}_{fit_task}_fit.mat")
                 for m in MONKEYS]
    else:
        paths = sorted(glob.glob(os.path.join("data_DDMfit", "human",
                                              f"*_{fit_task}_fit.mat")))

    sens_by_img = {}
    for path in paths:
        if not os.path.isfile(path):
            continue
        _, uimg, k = load_fit_file(path, flip=flip)
        for img, kk in zip(uimg, k):
            key = extract_image_basename(img)
            sens_by_img.setdefault(key, []).append(float(kk))

    if not sens_by_img:
        return None
    return pd.DataFrame({
        "image": list(sens_by_img),
        "sensitivity": [float(np.mean(v)) for v in sens_by_img.values()],
    })
