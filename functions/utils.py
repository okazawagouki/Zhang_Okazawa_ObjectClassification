import numpy as np


def index_in(a, b):
    index_dict = {value: idx for idx, value in enumerate(b)}
    return [index_dict.get(item, -1) for item in a]


def split_with_image(trials):
    """Group trials by image: (list of trial groups, sorted unique image names)."""
    images = [tr["img_url"].split("/")[-1] for tr in trials]
    uimages = np.sort(list(set(images)))
    idx = index_in(images,uimages)
    trials_image = []
    for i in range(len(uimages)):
        trials_image.append(np.array(trials)[np.array(idx)==i])
    return trials_image,uimages


def first_trials_by_image(trials):
    """(first trial of each unique image, sorted unique image names)."""
    trials_image, uimages = split_with_image(trials)
    return [trs[0] for trs in trials_image], uimages


def first_trial_accuracy(trials):
    """P(correct), binomial SE and per-image outcomes over each image's FIRST
    trial — the shared novel-image generalization metric."""
    first_trial, _ = first_trials_by_image(trials)
    correct = [tr["result"] == "CORRECT" for tr in first_trial]
    p = np.mean(correct)
    se = np.sqrt(p * (1 - p) / len(correct))
    return p, se, correct


def pairwise_euclid_nan(X: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distance between columns of X.

    For each column pair, only rows where neither column is NaN are used
    (same as MATLAB ``pairwise_euclid_nan``). Diagonal is 0; pairs with no
    overlapping valid rows remain NaN.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_rows, M); columns are compared.

    Returns
    -------
    D : np.ndarray
        Shape (M, M), symmetric.
    """
    X = np.asarray(X, dtype=float)
    M = X.shape[1]
    D = np.full((M, M), np.nan)
    np.fill_diagonal(D, 0.0)
    for i in range(M):
        xi = X[:, i]
        for j in range(i + 1, M):
            xj = X[:, j]
            v = ~np.isnan(xi) & ~np.isnan(xj)
            if np.any(v):
                d = np.sqrt(np.sum((xi[v] - xj[v]) ** 2))
                D[i, j] = d
                D[j, i] = d
    return D
