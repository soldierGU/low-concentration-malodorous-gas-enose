# -*- coding: utf-8 -*-
"""Train TimeMIL on the cached 5-fold split.

Aligned with the local comparison protocol:
  - data from npy_cache
  - 5-fold prepared split
  - Adam + CosineAnnealingLR
  - best checkpoint by validation macro-F1
  - final test acc, macro-F1, macro-recall, parameters, FLOPs, model size, CPU latency
"""

import argparse
import csv
import gc
import json
import os
import random
import time
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset

try:
    from thop import profile
except ImportError:
    profile = None

from model_for_compare.ref_model18_TimeMIL import Ref18_TimeMIL


NUM_ROWS = 400
NUM_CH = 16
NUM_CLASSES = 8
BATCH = 32
EPOCHS = 200
OUT_DIR = "compare_models_5fold_TimeMIL"
CACHE_ROOT = "npy_cache"
FOLDS_CACHE_ROOT = os.path.join(CACHE_ROOT, "experiments_5_fold")
TEST_CACHE_ROOT = os.path.join(CACHE_ROOT, "split_data_new", "test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CachedNpyDataset(Dataset):
    def __init__(self, root_dir):
        self.x = np.load(os.path.join(root_dir, "x.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(root_dir, "y.npy"), mmap_mode="r")
        if self.x.ndim != 3 or self.x.shape[1:] != (NUM_CH, NUM_ROWS):
            raise ValueError(f"Unexpected x.npy shape at {root_dir}: {self.x.shape}")
        if len(self.x) != len(self.y):
            raise ValueError(f"x/y length mismatch at {root_dir}: {len(self.x)} != {len(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.asarray(self.x[idx], dtype=np.float32).copy()
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_model(args):
    return Ref18_TimeMIL(
        num_channels=NUM_CH,
        num_classes=NUM_CLASSES,
        seq_length=NUM_ROWS,
        m_dim=args.m_dim,
        heads=args.heads,
        num_landmarks=args.num_landmarks,
        dropout=args.dropout,
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion, top_k=5):
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    topk = 0
    preds_all = []
    labels_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        _, p1 = torch.max(out, dim=1)
        _, pk = torch.topk(out, k=min(top_k, out.shape[1]), dim=1)

        bs = x.size(0)
        total += bs
        loss_sum += loss.item() * bs
        top1 += (p1 == y).sum().item()
        topk += (pk == y.view(-1, 1)).any(dim=1).sum().item()
        preds_all.extend(p1.detach().cpu().numpy())
        labels_all.extend(y.detach().cpu().numpy())

    avg_loss = loss_sum / total if total else 0.0
    acc1 = 100.0 * top1 / total if total else 0.0
    acck = 100.0 * topk / total if total else 0.0
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0) * 100.0
    rec = recall_score(labels_all, preds_all, average="macro", zero_division=0) * 100.0
    return avg_loss, acc1, acck, f1, rec


def train_one_fold(fold_idx, args):
    set_seeds(42 + fold_idx)
    fold_dir = os.path.join(FOLDS_CACHE_ROOT, f"fold{fold_idx}")
    train_cache_dir = os.path.join(fold_dir, "train_all_balanced")
    val_cache_dir = os.path.join(fold_dir, "val")
    save_dir = os.path.join(args.out_dir, "TimeMIL", f"fold{fold_idx}")
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(
        CachedNpyDataset(train_cache_dir),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CachedNpyDataset(val_cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    best_f1 = -1.0
    ckpt_path = os.path.join(save_dir, f"best_TimeMIL_fold{fold_idx}.pt")
    history_path = os.path.join(save_dir, f"history_TimeMIL_fold{fold_idx}.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc1", "val_loss", "val_acc1", "val_acc5", "val_f1", "val_recall"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        top1 = 0
        warmup = epoch <= args.warmup_epochs

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, warmup=warmup)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total += bs
            loss_sum += loss.item() * bs
            top1 += (out.argmax(dim=1) == y).sum().item()

        train_loss = loss_sum / total
        train_acc1 = 100.0 * top1 / total
        scheduler.step()
        val_loss, val_acc1, val_acc5, val_f1, val_recall = evaluate(model, val_loader, DEVICE, criterion)

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc1, val_loss, val_acc1, val_acc5, val_f1, val_recall])

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[TimeMIL F{fold_idx} Ep{epoch:03d}] "
                f"train_loss={train_loss:.4f}, train_acc={train_acc1:.2f}% | "
                f"val_acc={val_acc1:.2f}%, val_f1={val_f1:.2f}, val_recall={val_recall:.2f}",
                flush=True,
            )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    val_loss, val_acc1, val_acc5, val_f1, val_recall = evaluate(model, val_loader, DEVICE, criterion)
    summary = {
        "model": "TimeMIL",
        "fold": fold_idx,
        "m_dim": args.m_dim,
        "heads": args.heads,
        "num_landmarks": args.num_landmarks,
        "dropout": args.dropout,
        "warmup_epochs": args.warmup_epochs,
        "val_loss": val_loss,
        "val_acc1": val_acc1,
        "val_acc5": val_acc5,
        "val_f1": val_f1,
        "val_recall": val_recall,
        "best_ckpt": ckpt_path,
    }
    with open(os.path.join(save_dir, f"val_summary_TimeMIL_fold{fold_idx}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return summary


def model_from_summary(summary):
    return Ref18_TimeMIL(
        num_channels=NUM_CH,
        num_classes=NUM_CLASSES,
        seq_length=NUM_ROWS,
        m_dim=int(summary["m_dim"]),
        heads=int(summary["heads"]),
        num_landmarks=int(summary["num_landmarks"]),
        dropout=float(summary["dropout"]),
    )


@torch.no_grad()
def evaluate_test_for_folds(out_dir, batch_size):
    test_loader = DataLoader(
        CachedNpyDataset(TEST_CACHE_ROOT),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    criterion = nn.CrossEntropyLoss()
    fold_rows = []
    for fold_idx in range(1, 6):
        val_summary_path = os.path.join(out_dir, "TimeMIL", f"fold{fold_idx}", f"val_summary_TimeMIL_fold{fold_idx}.json")
        with open(val_summary_path, "r", encoding="utf-8") as f:
            val_summary = json.load(f)
        model = model_from_summary(val_summary)
        state = torch.load(val_summary["best_ckpt"], map_location="cpu")
        model.load_state_dict(state)
        model.to(DEVICE)
        test_loss, test_acc1, test_acc5, test_f1, test_recall = evaluate(model, test_loader, DEVICE, criterion)
        row = {
            "model": "TimeMIL",
            "fold": fold_idx,
            "test_loss": test_loss,
            "test_acc1": test_acc1,
            "test_acc5": test_acc5,
            "test_f1": test_f1,
            "test_recall": test_recall,
        }
        fold_rows.append(row)
        print(f"[TimeMIL Test F{fold_idx}] Acc={test_acc1:.2f}, F1={test_f1:.2f}, Recall={test_recall:.2f}", flush=True)
    return fold_rows


def checkpoint_size_mb(model):
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024.0 * 1024.0)


@torch.no_grad()
def profile_model(model):
    cpu_model = model.to("cpu").eval()
    dummy = torch.randn(1, NUM_CH, NUM_ROWS)
    params = sum(p.numel() for p in cpu_model.parameters())
    flops = None
    if profile is not None:
        flops, _ = profile(cpu_model, inputs=(dummy,), verbose=False)
        flops = int(flops)
    return params, flops, checkpoint_size_mb(cpu_model)


@torch.no_grad()
def measure_cpu_latency_ms(model, runs=100, warmup=20):
    cpu_model = model.to("cpu").eval()
    dummy = torch.randn(1, NUM_CH, NUM_ROWS)
    for _ in range(warmup):
        _ = cpu_model(dummy)
    start = time.perf_counter()
    for _ in range(runs):
        _ = cpu_model(dummy)
    return (time.perf_counter() - start) * 1000.0 / runs


def mean_std(rows, key):
    arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def write_outputs(out_dir, val_rows, test_rows, resource_row):
    val_csv = os.path.join(out_dir, "timemil_val_per_fold.csv")
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "fold", "m_dim", "heads", "num_landmarks", "dropout", "warmup_epochs",
            "val_loss", "val_acc1", "val_acc5", "val_f1", "val_recall", "best_ckpt",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    test_csv = os.path.join(out_dir, "timemil_test_per_fold.csv")
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "fold", "test_loss", "test_acc1", "test_acc5", "test_f1", "test_recall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    val_acc_m, val_acc_s = mean_std(val_rows, "val_acc1")
    val_f1_m, val_f1_s = mean_std(val_rows, "val_f1")
    val_rec_m, val_rec_s = mean_std(val_rows, "val_recall")
    test_acc_m, test_acc_s = mean_std(test_rows, "test_acc1")
    test_f1_m, test_f1_s = mean_std(test_rows, "test_f1")
    test_rec_m, test_rec_s = mean_std(test_rows, "test_recall")

    final = {
        "model": "TimeMIL",
        "val_acc_mean": val_acc_m,
        "val_acc_std": val_acc_s,
        "val_f1_mean": val_f1_m,
        "val_f1_std": val_f1_s,
        "val_recall_mean": val_rec_m,
        "val_recall_std": val_rec_s,
        "test_acc_mean": test_acc_m,
        "test_acc_std": test_acc_s,
        "test_f1_mean": test_f1_m,
        "test_f1_std": test_f1_s,
        "test_recall_mean": test_rec_m,
        "test_recall_std": test_rec_s,
        **resource_row,
    }

    with open(os.path.join(out_dir, "timemil_final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    final_csv = os.path.join(out_dir, "timemil_final_summary.csv")
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final.keys()))
        writer.writeheader()
        writer.writerow(final)
    return final, final_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--folds", default="1,2,3,4,5", help="Comma-separated fold ids to train when not using --skip-train.")
    parser.add_argument("--skip-train", action="store_true", help="Only re-run test/resource aggregation from existing checkpoints.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--m-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-landmarks", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(42)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Using device: {DEVICE}", flush=True)
    print(f"Using cached folds: {FOLDS_CACHE_ROOT}", flush=True)

    val_rows = []
    if not args.skip_train:
        fold_ids = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
        for fold_idx in fold_ids:
            val_rows.append(train_one_fold(fold_idx, args))
        if set(fold_ids) != {1, 2, 3, 4, 5}:
            print(f"Trained folds {fold_ids}. Re-run with --skip-train after all five folds exist to aggregate test/resource metrics.", flush=True)
            return
    else:
        for fold_idx in range(1, 6):
            path = os.path.join(args.out_dir, "TimeMIL", f"fold{fold_idx}", f"val_summary_TimeMIL_fold{fold_idx}.json")
            with open(path, "r", encoding="utf-8") as f:
                val_rows.append(json.load(f))

    test_rows = evaluate_test_for_folds(args.out_dir, args.batch_size)
    first_model = model_from_summary(val_rows[0])
    params, flops, model_size_mb = profile_model(first_model)
    latency_ms = measure_cpu_latency_ms(first_model)
    resource_row = {
        "parameters": int(params),
        "params_M": params / 1e6,
        "flops": int(flops) if flops is not None else None,
        "flops_M": (flops / 1e6) if flops is not None else None,
        "model_size_MB": model_size_mb,
        "cpu_latency_ms_per_sample": latency_ms,
    }

    final, final_csv = write_outputs(args.out_dir, val_rows, test_rows, resource_row)
    print(f"Final summary saved to: {final_csv}", flush=True)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
