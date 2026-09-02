# -*- coding: utf-8 -*-
"""
Prepack txt sensor samples into fast .npy cache files.

Expected sample shape in txt: (400, 16)
Stored sample shape in cache: (N, 16, 400), matching model input.

Default behavior packs:
  - experiments_5_fold/fold*/train_all_balanced
  - experiments_5_fold/fold*/val
  - split_data_new/test
  - split_data_new/val, if present
  - split_data_new/train, if it contains class-folder txt files
"""

import argparse
import json
import os
import re
from glob import glob

import numpy as np


NUM_ROWS = 400
NUM_CH = 16


def parse_label_from_prefix(path):
    base = os.path.basename(path)
    match = re.match(r"^\s*(\d+)_", base)
    if not match:
        raise ValueError(f"Cannot parse label from filename: {path}")
    return int(match.group(1))


def load_txt_sample(path):
    arr = np.loadtxt(path).astype(np.float32)
    if arr.shape != (NUM_ROWS, NUM_CH):
        raise ValueError(f"Unexpected shape at {path}: {arr.shape}, expected ({NUM_ROWS}, {NUM_CH})")
    return arr.T


def collect_class_folder(root):
    paths = []
    labels = []
    class_dirs = [
        d for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and d.isdigit()
    ]
    for cls in class_dirs:
        for path in sorted(glob(os.path.join(root, cls, "*.txt"))):
            paths.append(path)
            labels.append(int(cls))
    return paths, labels


def collect_flat_prefix(root):
    paths = []
    labels = []
    for path in sorted(glob(os.path.join(root, "*.txt"))):
        paths.append(path)
        labels.append(parse_label_from_prefix(path))
    return paths, labels


def infer_layout(root):
    class_paths, class_labels = collect_class_folder(root)
    if class_paths:
        return "class_folder", class_paths, class_labels

    flat_paths, flat_labels = collect_flat_prefix(root)
    if flat_paths:
        return "flat_prefix", flat_paths, flat_labels

    return "empty", [], []


def cache_dir_for(source_root, cache_root):
    rel = os.path.relpath(os.path.abspath(source_root), os.getcwd())
    return os.path.join(cache_root, rel)


def pack_dataset(source_root, cache_root, overwrite=False):
    source_root = os.path.normpath(source_root)
    if not os.path.isdir(source_root):
        print(f"[skip] missing: {source_root}")
        return False

    layout, paths, labels = infer_layout(source_root)
    if not paths:
        print(f"[skip] no txt samples: {source_root}")
        return False

    out_dir = cache_dir_for(source_root, cache_root)
    x_path = os.path.join(out_dir, "x.npy")
    y_path = os.path.join(out_dir, "y.npy")
    meta_path = os.path.join(out_dir, "meta.json")
    paths_path = os.path.join(out_dir, "paths.txt")

    if not overwrite and os.path.isfile(x_path) and os.path.isfile(y_path):
        print(f"[exists] {source_root} -> {out_dir}")
        return True

    os.makedirs(out_dir, exist_ok=True)
    x = np.empty((len(paths), NUM_CH, NUM_ROWS), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    for idx, path in enumerate(paths, start=1):
        x[idx - 1] = load_txt_sample(path)
        if idx == 1 or idx % 200 == 0 or idx == len(paths):
            print(f"[pack] {source_root}: {idx}/{len(paths)}")

    np.save(x_path, x)
    np.save(y_path, y)
    with open(paths_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(os.path.normpath(path) + "\n")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_root": source_root,
                "layout": layout,
                "num_samples": int(len(paths)),
                "x_shape": list(x.shape),
                "y_shape": list(y.shape),
                "dtype": "float32",
                "stored_sample_shape": [NUM_CH, NUM_ROWS],
            },
            f,
            indent=2,
        )

    print(f"[done] {source_root} -> {out_dir}")
    return True


def default_sources(folds_root, split_root):
    sources = []

    if os.path.isdir(folds_root):
        folds = [d for d in sorted(os.listdir(folds_root)) if d.startswith("fold")]
        for fold in folds:
            fold_root = os.path.join(folds_root, fold)
            sources.append(os.path.join(fold_root, "train_all_balanced"))
            sources.append(os.path.join(fold_root, "val"))

    if os.path.isdir(split_root):
        sources.append(os.path.join(split_root, "train"))
        sources.append(os.path.join(split_root, "val"))
        sources.append(os.path.join(split_root, "test"))

    return sources


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="npy_cache")
    parser.add_argument("--folds-root", default="experiments_5_fold")
    parser.add_argument("--split-root", default="split_data_new")
    parser.add_argument("--source", action="append", default=None, help="Specific dataset root to pack.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sources = args.source or default_sources(args.folds_root, args.split_root)
    os.makedirs(args.cache_root, exist_ok=True)

    for source in sources:
        pack_dataset(source, args.cache_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
