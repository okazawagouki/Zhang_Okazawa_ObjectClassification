"""Regenerate the model features used by the main-figure notebooks.

Extracts CNN (torchvision) and VLM (HuggingFace) features for the small-task
and large-task image sets under ``data_set/`` and writes them to
``data_model/smalltask_features/`` and ``data_model/largetask_features/``
using the same file naming and variable layout (``fnames``, ``output``,
``category``) as the analysis notebooks expect:

    fnames   full image paths with '\\' separators (data_set\\<task>\\...)
    output   (n_images, n_dims) float32 feature matrix
    category rule-class label of each image (e.g. animate / inanimate)

Extraction conventions (matching the shipped feature files):
  - CNNs: torchvision pretrained (DEFAULT weights), avgpool activations
    (ViT: ``encoder``), batch of 100, flattened.
  - VLMs: facebook/dino-vitb16 (CLS token), openai/clip-vit-base-patch32
    and google/siglip2-base-patch32-256 (get_image_features), float32,
    no L2 normalization.

The low-level models (v1 / LUV / gistSF / texture) are extracted with
MATLAB — see ``extract_lowlev_features.m``.

Run from the repository root:
    python data_model/extract_features_all.py [--force]
Existing output files are skipped unless --force is given.
"""
import argparse
import math
import os
import sys

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import extract_feature as feat  # noqa: E402

SMALL_TASKS = [
    "animate_vs_inanimate", "natural_vs_artificial", "mammal_vs_reptile",
    "electronics_vs_tools", "human_vs_monkey", "monkey_vs_mammal",
    "animate_vs_inanimate_THINGS", "natural_vs_artificial_THINGS",
    "mammal_vs_nonmammal_THINGS", "electronics_vs_tools_THINGS",
    "outdoor_vs_indoor", "big_vs_small", "big_vs_small_Konkle",
    "fire related_vs_water related", "eastern_vs_western",
]
SMALL_TASKS_GEN = [t + "_gen" for t in SMALL_TASKS]
LARGE_TASKS = [
    "Large_animate_vs_inanimate", "Large_mammal_vs_nonmammal",
    "Large_natural_vs_artificial",
]

CNN_MODELS = ["alexnet", "vgg16", "resnet50", "vit_b_32"]
VLM_MODELS = {"dino_vitb": "facebook/dino-vitb16",
              "clip": "openai/clip-vit-base-patch32",
              "siglip2": "google/siglip2-base-patch32-256"}
# dino_vitb / siglip2 features are stored L2-normalized (the convention of
# the released feature files); clip and the CNNs are stored raw.
VLM_L2_NORMALIZE = {"dino_vitb": True, "clip": False, "siglip2": True}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def list_images(task_name):
    """All images of one task set, with rule-class labels.

    Returns (fnames, categories) where fnames use '\\' separators and carry
    the ``data_set\\<task>\\...`` prefix, matching the original feature files.
    """
    fnames, cats = [], []
    root = os.path.join("data_set", task_name)
    for path, _, files in os.walk(root):
        for filename in sorted(files):
            if not (filename.endswith(".png") or filename.endswith(".jpg")):
                continue
            rel = os.path.relpath(os.path.join(path, filename), "data_set")
            fnames.append(rel.replace("/", "\\").replace(os.sep, "\\"))
            # rule-class index within the FULL path (data_set\<task>\...):
            # large-task sets have an extra anchor/stim level, small tasks do
            # not — data_set\<task>\<anchor|stim>\<rule>\... vs
            # data_set\<task>\<rule>\... .
            parts = ("data_set/" + rel.replace("\\", "/")).split("/")
            if task_name in LARGE_TASKS:
                cats.append(parts[3])
            else:
                cats.append(parts[2])
    return fnames, cats


def extract_cnn(task_name, fnames, model_name):
    model, preprocess = feat.load_model(model_name)
    module_name = "encoder" if model_name == "vit_b_32" else "avgpool"
    imgpaths = [os.path.join("data_set", *f.split("\\")) for f in fnames]

    batchsize = 100
    niter = math.ceil(len(imgpaths) / batchsize)
    activations = []
    for i in trange(niter, desc=f"{task_name}/{model_name}"):
        lo, hi = i * batchsize, min((i + 1) * batchsize, len(imgpaths))
        with torch.no_grad():
            batch = feat.get_batch(imgpaths[lo:hi], preprocess)
            act = feat.get_activation(model, module_name, batch, device=DEVICE)
            activations.append(act.reshape(act.size(0), -1).numpy())
    return np.concatenate(activations, axis=0)


def _vlm_embedding(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
        else:
            vision_out = model(**inputs)
            outputs = vision_out.last_hidden_state[:, 0]
        outputs = outputs.float()
    return outputs.cpu().numpy().flatten()


def extract_vlm(task_name, fnames, hf_name, l2_normalize):
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_name, trust_remote_code=True).to(DEVICE)
    model.eval()
    imgpaths = [os.path.join("data_set", *f.split("\\")) for f in fnames]

    batchsize = 64
    niter = math.ceil(len(imgpaths) / batchsize)
    rows = []
    for i in trange(niter, desc=f"{task_name}/{hf_name}"):
        lo, hi = i * batchsize, min((i + 1) * batchsize, len(imgpaths))
        images = [Image.open(p).convert("RGB") for p in imgpaths[lo:hi]]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                outputs = model.get_image_features(**inputs)
                if hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
            else:
                outputs = model(**inputs).last_hidden_state[:, 0]
            outputs = outputs.float().cpu().numpy()
        rows.append(outputs)
    out = np.concatenate(rows, axis=0)
    if l2_normalize:
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out


def out_path(task_name, model_name):
    if task_name in LARGE_TASKS:
        return os.path.join("data_model", "largetask_features",
                            f"{task_name}_{model_name}.mat")
    return os.path.join("data_model", "smalltask_features",
                        f"{task_name}_{model_name}.mat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output files")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="restrict to these task set names")
    args = ap.parse_args()

    tasks = SMALL_TASKS + SMALL_TASKS_GEN + LARGE_TASKS
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    for task_name in tasks:
        if not os.path.isdir(os.path.join("data_set", task_name)):
            print(f"skip {task_name}: no data_set folder")
            continue
        fnames, cats = list_images(task_name)

        for model_name in CNN_MODELS:
            dst = out_path(task_name, model_name)
            if os.path.exists(dst) and not args.force:
                continue
            output = extract_cnn(task_name, fnames, model_name)
            scio.savemat(dst, {"fnames": fnames, "output": output,
                               "category": cats})
            print(f"saved {dst} {output.shape}")

        for model_name, hf_name in VLM_MODELS.items():
            dst = out_path(task_name, model_name)
            if os.path.exists(dst) and not args.force:
                continue
            output = extract_vlm(task_name, fnames, hf_name,
                                 VLM_L2_NORMALIZE[model_name])
            scio.savemat(dst, {"fnames": fnames, "output": output,
                               "category": cats})
            print(f"saved {dst} {output.shape}")


if __name__ == "__main__":
    main()
