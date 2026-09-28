"""Shared model-feature loading and linear decoding pipeline.

The feature files in ``data_model/*_features/`` store three variables:
``fnames`` (image paths with '\\' separators), ``output`` ((n_images, n_dims)
feature matrix) and ``category`` (rule-class label per image). ``fnames`` /
``category`` are plain str arrays in some files and cell arrays in others;
``load_feature_mat`` normalizes both.
"""
import numpy as np
import scipy.io as scio
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from functions.load_data import unpack_cell


def load_feature_mat(path):
    """Read one feature file -> (X, category, fnames)."""
    feature = scio.loadmat(path)
    X = feature["output"]
    category = (unpack_cell(feature["category"])
                if feature["category"].ndim == 2 else feature["category"])
    fnames = (unpack_cell(feature["fnames"])
              if feature["fnames"].ndim == 2 else feature["fnames"])
    return X, category, fnames


def image_basenames(fnames):
    """Final path component of each stored image path ('a\\b\\img.png' -> 'img.png')."""
    return np.array([str(f).strip().split("\\")[-1] for f in fnames])


def pca_standardize(x_train, x_test, n_components=20):
    """PCA fitted on the training features, then z-scoring with the training
    statistics. PCA is skipped when the features already have at most
    ``n_components`` dimensions (the shipped PCA-20 feature files)."""
    if x_train.shape[1] > n_components:
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def fit_logistic(x_train, y_train):
    """Fit the shared logistic classifier (liblinear, fixed seed)."""
    model = LogisticRegression(random_state=0, fit_intercept=True,
                               solver="liblinear")
    return model.fit(x_train, y_train)
