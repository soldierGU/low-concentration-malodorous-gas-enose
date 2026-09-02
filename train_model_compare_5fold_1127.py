# -*- coding: utf-8 -*-
"""
5-fold cross-validation training for comparison models in `model_for_compare` package.

- Data structure (same as ablation):
    FOLDS_ROOT/
      fold1/
        train_all_balanced/0..7/*.txt
        val/*.txt               (flat, prefix label)
        test/*.txt ?? (若 test 单独放外面，可在 TEST_ROOT 修改)
      ...
      fold5/

- This script:
    1) Trains multiple models (from model_for_compare) using the same training strategy.
    2) Performs 5-fold CV: each fold has its own train/val.
    3) Saves per-fold best checkpoint (by val macro-F1).
    4) After training, evaluates each model on the unified test set and outputs mean±std.

Replace the MODEL_BUILDERS dict with your real models from `model_for_compare`.
"""

import os
import re
import csv
import json
import random
import time
from glob import glob
from thop import profile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score

# ========== 这里导入你要对比的模型类 ==========
# 示例：请根据实际代码替换 / 添加
#
# from model_for_compare import ModelA, ModelB, ModelC
#
# 然后在 MODEL_BUILDERS 中注册：
#
#   "ModelA": lambda: ModelA(num_channels=16, num_classes=8, ...),
#   "ModelB": lambda: ModelB(...),
#   ...
#
# 注意：所有模型的输入 shape 统一为 (B, 16, 400)
# ============================================
# TODO: 修改为你自己的模型构造函数
from model_for_compare import (
    # ExampleModel1,
    # ExampleModel2,
    # ExampleModel3,
    ResNet_18,
    DenseNet_121,
    GoogLeNet_1D,
    ref_model2,
    ref_model4,
    ref_model5,
    ref_model13_ModernTCN,
    ref_model14_TimesNetEnose,
    Transformer_compare,
    ref_model15_GFAMNet
)

# ----------------- 通用配置 -----------------
FOLDS_ROOT = "experiments_5_fold"     # 5折数据根目录
TEST_ROOT  = os.path.join("split_data_new/test")  # 统一测试集目录（若不在这里，请改）

OUT_DIR    = "compare_models_5fold_1218"   # 实验输出目录

NUM_ROWS = 400
NUM_CH   = 16
BATCH    = 32
EPOCHS   = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 在这里列出你要对比的模型 ==========
# 键：模型名字（写入csv、文件夹名）
# 值：无参 lambda，返回一个已构造好的 nn.Module
# Only GFAM-Net is enabled by default; uncomment another baseline only when reproducing that specific comparison.
MODEL_BUILDERS = {
    # 示例（请修改为你自己的）：
    # "ExampleModel1": lambda: ExampleModel1(num_channels=16, num_classes=8),
    # "ExampleModel2": lambda: ExampleModel2(in_ch=16, num_classes=8),
    # "ExampleModel3": lambda: ExampleModel3(n_channels=16, n_classes=8),
    # "ResNet18":lambda :ResNet_18.ResNet1D_18(num_channels=16,num_classes=8),
    # "DenseNet121":lambda :DenseNet_121.DenseNet1D_121(num_classes=8,num_channels=16),
    # "GoogLeNet":lambda :GoogLeNet_1D.GoogLeNet1D(num_channels=16,num_classes=8),
    # "ref2":lambda :ref_model2.OneDCNN_article1(num_channels=16, num_classes=8),
    # "ref4":lambda :ref_model4.MultiscaleCNN(num_classes=8,num_channels=16),
    # "ref5":lambda :ref_model5.OneD_DNR(num_classes=8),
   # "ModernTCN":lambda :ref_model13_ModernTCN.Ref13_ModernTCN_Classifier(num_classes=8, M=16, L=400),
   # "TimesNet": lambda :ref_model14_TimesNetEnose.Ref14_TimesNet_Enose(num_classes=8, C_in=16, seq_len=400),
   # "Transformer":lambda :Transformer_compare.TransformerEncoder(num_classes=8, C_in=16, T=400)
    "GFAM-Net":lambda :ref_model15_GFAMNet.Ref15_GFAMNet(num_classes=8, sensors=16, time_len=400)
}

# ----------------- 随机种子 -----------------3
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------- 数据读取 -----------------
def _safe_load_txt(p):
    """读取单个样本，默认 shape=(400,16)。"""
    arr = np.loadtxt(p).astype(np.float32)
    if arr.shape != (NUM_ROWS, NUM_CH):
        raise ValueError(f"Unexpected shape at {p}: {arr.shape}, expected ({NUM_ROWS},{NUM_CH})")
    return arr

def measure_inference_time(model, device, batch_size=32, runs=100):
    model.eval()
    dummy = torch.randn(batch_size, 16, 400).to(device)

    # warmup
    for _ in range(20):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs // batch_size):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    return total_time * 1000  # ms

def analyze_model(model, device):
    model = model.to(device)
    dummy = torch.randn(1, 16, 400).to(device)

    # 参数量
    num_params = sum(p.numel() for p in model.parameters())

    # MACs
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    macs = macs / 1e6  # M MACs
    params_m = params / 1e6  # M Params

    # inference time (per 100 samples)
    batch = 32
    dummy_batch = torch.randn(batch, 16, 400).to(device)
    model.eval()

    # warmup
    for _ in range(20):
        _ = model(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    import time
    start = time.time()
    iters = 100 // batch
    for _ in range(iters):
        _ = model(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000  # milliseconds

    return num_params, macs, params_m, elapsed_ms

def _parse_label_from_prefix(fname):
    """
    从文件名前缀解析标签：
      0_xxx.txt
      3_50ppm_yyy.txt
    取第一个 '_' 之前的数字为类别
    """
    base = os.path.basename(fname)
    m = re.match(r"^\s*(\d+)_", base)
    if not m:
        raise ValueError(f"Cannot parse label from filename: {fname}")
    return int(m.group(1))


class EnoDataset(Dataset):
    """简单 Dataset：paths 为 .txt 路径，labels 为 int(0..7)。"""
    def __init__(self, paths, labels):
        self.paths = list(paths)
        self.labels = list(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = _safe_load_txt(self.paths[idx])   # (400,16)
        x = np.transpose(x, (1, 0))           # -> (16,400)
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def load_train_folder_by_class(train_root):
    """
    train_all_balanced/ 结构:
      train_root/
        0/*.txt
        1/*.txt
        ...
        7/*.txt
    """
    paths, labels = [], []
    classes = sorted([d for d in os.listdir(train_root)
                      if os.path.isdir(os.path.join(train_root, d))])
    for cls in classes:
        cls_dir = os.path.join(train_root, cls)
        for p in sorted(glob(os.path.join(cls_dir, "*.txt"))):
            paths.append(p)
            labels.append(int(cls))
    return np.array(paths), np.array(labels)


def load_flat_with_prefix_label(flat_dir):
    """
    val/ 或 test/ 结构扁平:
      flat_dir/
        0_xxx.txt
        3_50ppm_yyy.txt
        ...
    标签通过文件名前缀解析。
    """
    paths, labels = [], []
    for p in sorted(glob(os.path.join(flat_dir, "*.txt"))):
        try:
            label = _parse_label_from_prefix(p)
            paths.append(p)
            labels.append(label)
        except ValueError:
            print(f"[WARN] skip file without label prefix: {p}")
    return np.array(paths), np.array(labels)


# ----------------- 评估 -----------------
@torch.no_grad()
def evaluate(model, loader, device, criterion, top_k=5):
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    topk = 0

    preds_all, labels_all = [], []

    for x, y in loader:
        x = x.to(device)   # (B, C, T)
        y = y.to(device)

        out = model(x)     # (B, num_classes)
        loss = criterion(out, y)

        _, p1 = torch.max(out, dim=1)            # top-1
        _, pk = torch.topk(out, k=top_k, dim=1)  # top-k

        correct_k = (pk == y.view(-1, 1)).any(dim=1).sum()

        bs = x.size(0)
        total += bs
        loss_sum += loss.item() * bs
        top1 += (p1 == y).sum().item()
        topk += correct_k.item()

        preds_all.extend(p1.detach().cpu().numpy())
        labels_all.extend(y.detach().cpu().numpy())

    avg_loss = loss_sum / total if total > 0 else 0.0
    acc1 = 100.0 * top1 / total if total > 0 else 0.0
    acck = 100.0 * topk / total if total > 0 else 0.0

    f1 = f1_score(labels_all, preds_all, average='macro') * 100.0
    rec = recall_score(labels_all, preds_all, average='macro') * 100.0

    return avg_loss, acc1, acck, f1, rec


# ----------------- 单折训练 -----------------
def train_one_fold(model_name, model_builder,
                   fold_idx, train_paths, train_labels,
                   val_paths, val_labels, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    set_seeds(42 + fold_idx)  # 不同fold略微变 seed

    train_loader = DataLoader(
        EnoDataset(train_paths, train_labels),
        batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        EnoDataset(val_paths, val_labels),
        batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
    )

    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_f1 = -1.0
    ckpt_path = os.path.join(save_dir, f"best_{model_name}_fold{fold_idx}.pt")

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        top1 = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total += bs
            loss_sum += loss.item() * bs
            top1 += (out.argmax(dim=1) == y).sum().item()

        train_loss = loss_sum / total if total > 0 else 0.0
        train_acc1 = 100.0 * top1 / total if total > 0 else 0.0

        scheduler.step()

        val_loss, val_acc1, val_acc5, val_f1, val_rec = evaluate(
            model, val_loader, device, criterion
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            # 只存 state_dict 到 CPU，避免占显存
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

        if ep == 1 or ep % 10 == 0:
            print(f"[{model_name}, F{fold_idx}, Ep{ep:03d}] "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc1:5.2f}% | "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc1:5.2f}%, "
                  f"val_f1={val_f1:5.2f}, val_rec={val_rec:5.2f}")

    # 训练结束，加载 best 模型再做一次 val 评估（方便记录）
    if os.path.isfile(ckpt_path):
        best_state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(best_state)
    val_loss, val_acc1, val_acc5, val_f1, val_rec = evaluate(
        model, val_loader, device, criterion
    )

    summary_path = os.path.join(save_dir, f"val_summary_{model_name}_fold{fold_idx}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "fold": fold_idx,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "val_f1": val_f1,
            "val_recall": val_rec,
            "best_ckpt": ckpt_path
        }, f, indent=2)

    return val_loss, val_acc1, val_acc5, val_f1, val_rec


# ----------------- 统一 test 阶段 -----------------
@torch.no_grad()
def test_all_models():
    print("\n====== Running Unified Test Evaluation for All Models ======\n")

    assert os.path.isdir(TEST_ROOT), f"Test dir not found: {TEST_ROOT}"
    test_paths, test_labels = load_flat_with_prefix_label(TEST_ROOT)
    test_loader = DataLoader(
        EnoDataset(test_paths, test_labels),
        batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
    )

    test_csv = os.path.join(OUT_DIR, "test_summary_results.csv")
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "acc_mean", "acc_std",
            "f1_mean", "f1_std",
            "rec_mean", "rec_std"
        ])

        for model_name, builder in MODEL_BUILDERS.items():
            print(f"[Test] Model = {model_name}")

            fold_acc, fold_f1, fold_rec = [], [], []

            for fold_idx in range(1, 6):
                ckpt_path = os.path.join(
                    OUT_DIR, model_name, f"fold{fold_idx}",
                    f"best_{model_name}_fold{fold_idx}.pt"
                )
                if not os.path.isfile(ckpt_path):
                    print(f"  [WARN] Missing ckpt: {ckpt_path}")
                    continue

                model = builder()
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)
                model.to(DEVICE)
                model.eval()

                criterion = nn.CrossEntropyLoss()
                loss, acc1, acc5, f1, rec = evaluate(
                    model, test_loader, DEVICE, criterion
                )
                print(f"   Fold {fold_idx}: Acc={acc1:.2f}, F1={f1:.2f}, Rec={rec:.2f}")
                fold_acc.append(acc1)
                fold_f1.append(f1)
                fold_rec.append(rec)

            acc_mu = float(np.mean(fold_acc))
            acc_sd = float(np.std(fold_acc))
            f1_mu  = float(np.mean(fold_f1))
            f1_sd  = float(np.std(fold_f1))
            rec_mu = float(np.mean(fold_rec))
            rec_sd = float(np.std(fold_rec))

            print(f" ==> {model_name}: "
                  f"Test Acc={acc_mu:.2f}±{acc_sd:.2f}, "
                  f"F1={f1_mu:.2f}±{f1_sd:.2f}, "
                  f"Rec={rec_mu:.2f}±{rec_sd:.2f}\n")

            w.writerow([
                model_name,
                f"{acc_mu:.4f}", f"{acc_sd:.4f}",
                f"{f1_mu:.4f}",  f"{f1_sd:.4f}",
                f"{rec_mu:.4f}", f"{rec_sd:.4f}",
            ])

    print(f"\nUnified Test Summary Saved to: {test_csv}\n")


# ----------------- 主流程 -----------------
def main():
    set_seeds(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    folds = [d for d in sorted(os.listdir(FOLDS_ROOT))
             if d.startswith("fold")]
    assert len(folds) == 5, f"Expect 5 folds under {FOLDS_ROOT}, got {folds}"

    # CSV paths
    per_fold_csv = os.path.join(OUT_DIR, "val_per_fold_results.csv")
    summary_csv  = os.path.join(OUT_DIR, "val_summary_results.csv")
    analysis_csv = os.path.join(OUT_DIR, "model_complexity.csv")

    # Create CSV headers if first time
    if not os.path.exists(per_fold_csv):
        with open(per_fold_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "model", "fold",
                "val_loss", "val_acc1", "val_acc5", "val_f1", "val_recall"
            ])

    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "model",
                "mean_acc1", "std_acc1",
                "mean_f1", "std_f1",
                "mean_recall", "std_recall"
            ])

    if not os.path.exists(analysis_csv):
        with open(analysis_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "model", "num_params", "macs(M)", "params(M)", "inference_100samples(ms)"
            ])

    print(f"[Compare] {len(MODEL_BUILDERS)} models to train on 5 folds.")

    for model_name, builder in MODEL_BUILDERS.items():
        print(f"\n==== Training Model: {model_name} ====")

        # --------------------------------------------------------
        # ★ 在这里插入模型复杂度统计（训练前）
        # --------------------------------------------------------
        model = builder()
        num_params, macs, params_m, inf_time = analyze_model(model, DEVICE)

        with open(analysis_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                model_name,
                num_params,
                macs,
                params_m,
                inf_time
            ])
        print(f"[Analysis] {model_name}: Params={num_params}, MACs={macs:.2f}M, "
              f"Params(M)={params_m:.2f}, Time(100samples)={inf_time:.2f} ms")
        # --------------------------------------------------------

        fold_metrics_acc1 = []
        fold_metrics_f1   = []
        fold_metrics_rec  = []

        for fold_idx, fold_name in enumerate(folds, start=1):
            fold_root = os.path.join(FOLDS_ROOT, fold_name)
            train_root = os.path.join(fold_root, "train_all_balanced")
            val_root   = os.path.join(fold_root, "val")

            tr_paths, tr_labels = load_train_folder_by_class(train_root)
            va_paths, va_labels = load_flat_with_prefix_label(val_root)

            cfg_dir = os.path.join(OUT_DIR, model_name, f"fold{fold_idx}")
            os.makedirs(cfg_dir, exist_ok=True)

            vloss, vacc1, vacc5, vf1, vrec = train_one_fold(
                model_name, builder,
                fold_idx,
                tr_paths, tr_labels,
                va_paths, va_labels,
                DEVICE, cfg_dir
            )

            # write per fold
            with open(per_fold_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    model_name, fold_idx,
                    f"{vloss:.4f}", f"{vacc1:.4f}",
                    f"{vacc5:.4f}", f"{vf1:.4f}", f"{vrec:.4f}"
                ])

            fold_metrics_acc1.append(vacc1)
            fold_metrics_f1.append(vf1)
            fold_metrics_rec.append(vrec)

        # summarize across folds
        acc_mu = np.mean(fold_metrics_acc1)
        acc_sd = np.std(fold_metrics_acc1)
        f1_mu  = np.mean(fold_metrics_f1)
        f1_sd  = np.std(fold_metrics_f1)
        rec_mu = np.mean(fold_metrics_rec)
        rec_sd = np.std(fold_metrics_rec)

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                model_name,
                f"{acc_mu:.4f}", f"{acc_sd:.4f}",
                f"{f1_mu:.4f}",  f"{f1_sd:.4f}",
                f"{rec_mu:.4f}", f"{rec_sd:.4f}",
            ])

        print(f"[Val Summary {model_name}] "
              f"acc1={acc_mu:.4f}±{acc_sd:.4f}, "
              f"f1={f1_mu:.4f}±{f1_sd:.4f}, "
              f"recall={rec_mu:.4f}±{rec_sd:.4f}")

if __name__ == "__main__":
    main()
    # 训练完统一在测试集上评估
    test_all_models()
