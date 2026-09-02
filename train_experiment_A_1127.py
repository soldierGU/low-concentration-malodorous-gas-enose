# -*- coding: utf-8 -*-
"""
Grid search on 5-fold augmented dataset (experiments_5_fold):
  - Train: foldk/train_all_balanced/ (class folders 0..7)
  - Val  : foldk/val/                (flat .txt, filename starts with '<cls>_')
Search space:
  - L (number of blocks) in {1,2,3,4}
  - ts_rate (TSDA shift rate) in {1,2,3,4}

Model:
  - MScale_TSDDA_Net_1127 from Network_1127_baseline_AP_TSDA_DA_L
  - widths is truncated to first L entries to control block count.

Training:
  - Adam, lr=1e-3
  - CosineAnnealingLR, T_max=200, eta_min=1e-6
  - Epochs = 200
  - Batch size = 32

Outputs:
  - OUT_DIR/per_config/L{L}_S{ts_rate}/fold{1..5}/val_summary_fold_*.json
  - OUT_DIR/grid_results_per_fold.csv   # 每一折的详细指标
  - OUT_DIR/grid_results_summary.csv    # 每个配置跨5折的均值±方差
"""

import os
import re
import csv
import json
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score

# ====== 你的新网络入口 ======
from Network_1127_baseline_AP_TSDA_DA_L import MScale_TSDDA_Net_1127


# ----------------- 路径与常量 -----------------
FOLDS_ROOT = "experiments_5_fold"           # 5折增强数据的根目录
OUT_DIR    = "tuning_L_tsrate_1127"         # 实验输出目录

NUM_ROWS = 400
NUM_CH   = 16
BATCH    = 32
EPOCHS   = 200

# 超参数搜索空间
L_LIST       = [1, 2, 3, 4]                 # 卷积块数
TS_RATE_LIST = [1, 2, 3, 4]                 # TSDA shift_rate

# 基础 widths（会按 L 截断）
WIDTHS_BASE = (64, 96, 128, 160, 192)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- 随机种子 -----------------
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------- 数据读取相关 -----------------
def _safe_load_txt(p):
    """读取单个样本，默认 shape=(400,16)，可以根据需要适当放宽检查。"""
    arr = np.loadtxt(p).astype(np.float32)
    if arr.shape != (NUM_ROWS, NUM_CH):
        raise ValueError(f"Unexpected shape at {p}: {arr.shape}, expected ({NUM_ROWS},{NUM_CH})")
    return arr


def _parse_label_from_prefix(fname):
    """
    从文件名前缀解析标签：
      支持形如:
        '0_xxx.txt'
        '3_50ppm_foo.txt'
      即只取第一个 '_' 前面的数字。
    """
    base = os.path.basename(fname)
    m = re.match(r"^\s*(\d+)_", base)
    if not m:
        raise ValueError(f"Cannot parse label from filename: {fname}")
    return int(m.group(1))


class EnoDataset(Dataset):
    """简单 Dataset：paths 为 .txt 路径，labels 为 int(0~7)。"""
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
    train_all_balanced/ 结构：
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
    val/ 结构假设为扁平：
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
            # 如果有不规范文件名，可以在这里选择打印警告或忽略
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


# ----------------- 模型构造 -----------------
def build_model(L, ts_rate):
    """
    构造 MScale_TSDDA_Net_1127，并根据 L 控制 block 数量。
    - 假设网络的初始化接口类似于你之前的 MScale_TSDA_Net_0925：
        num_channels, num_classes, stem_ch, widths, ks, dilations, ts_rate,
        ts_mode, ts_boundary, depth_k_tuple, depth_dynamic_tuple, per_channel_depth,
        use_antialias, use_tsda, baseline
    如有出入，你可以在这里按你的 Network_1127 实际参数名稍微调整。
    """
    assert 1 <= L <= len(WIDTHS_BASE), "L must be between 1 and 4 (inclusive)."
    widths = WIDTHS_BASE[:L]

    # 一个合理的 Depth-DA 配置示例：仅在最后一两个 block 开启 DDA
    # 你可以根据论文的最终设定调整 base_depth_k / depth_dynamic_tuple
    base_depth_k = (0, 1, 2, 3, 4)            # 对应最多4层
    depth_k_tuple = base_depth_k[:L]

    base_depth_dyn = (False, False, True, True, True)
    depth_dynamic_tuple = base_depth_dyn[:L]

    model = MScale_TSDDA_Net_1127(
        num_channels=16,
        num_classes=8,
        stem_ch=64,
        widths=widths,
        ks=(1, 3, 5),
        dilations=(1, 2, 3),        # 你原来的默认 dilation，如需更改可在此调整
        ts_rate=ts_rate,
        ts_mode='joint2d',
        ts_boundary='zpad',
        depth_k_tuple=depth_k_tuple,
        depth_dynamic_tuple=depth_dynamic_tuple,
        per_channel_depth=False,
        use_antialias=True,
        use_tsda=True,
        baseline=False,             # 如果你想跑 Baseline，可在外面改这个参数
    )
    return model


# ----------------- 单折训练 -----------------
def train_one_fold(L, ts_rate, fold_idx, train_paths, train_labels,
                   val_paths, val_labels, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    set_seeds(42 + fold_idx)  # 不同fold稍微偏移seed

    train_loader = DataLoader(
        EnoDataset(train_paths, train_labels),
        batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        EnoDataset(val_paths, val_labels),
        batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
    )

    model = build_model(L, ts_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # CosineAnnealingLR，从 1e-3 衰减到 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_f1 = -1.0
    best_state = None
    ckpt_path = os.path.join(save_dir, f"best_fold_{fold_idx}.pt")

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

        # 每个 epoch 结束调整一次学习率
        scheduler.step()

        val_loss, val_acc1, val_acc5, val_f1, val_rec = evaluate(
            model, val_loader, device, criterion
        )

        # 以 val_f1 作为 best model 的选择标准
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

        if ep == 1 or ep % 10 == 0:
            print(f"[L={L}, S={ts_rate}, F{fold_idx}, Ep{ep:03d}] "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc1:5.2f}% | "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc1:5.2f}%, "
                  f"val_f1={val_f1:5.2f}, val_rec={val_rec:5.2f}")

    # 用 best_state 做最终评估
    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_acc1, val_acc5, val_f1, val_rec = evaluate(
        model, val_loader, device, criterion
    )

    # 写单折的 summary
    summary_path = os.path.join(save_dir, f"val_summary_fold_{fold_idx}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "L": L,
            "ts_rate": ts_rate,
            "fold": fold_idx,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "val_f1": val_f1,
            "val_recall": val_rec,
            "best_ckpt": ckpt_path
        }, f, indent=2)

    return val_loss, val_acc1, val_acc5, val_f1, val_rec


# ----------------- 主流程 -----------------
def main():
    set_seeds(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = DEVICE
    print(f"Using device: {device}")

    # 列出 fold1..fold5
    folds = [d for d in sorted(os.listdir(FOLDS_ROOT))
             if d.startswith("fold")]
    assert len(folds) == 5, f"Expect 5 folds under {FOLDS_ROOT}, got {folds}"

    # 记录每一折的结果
    per_fold_csv = os.path.join(OUT_DIR, "grid_results_per_fold.csv")
    summary_csv  = os.path.join(OUT_DIR, "grid_results_summary.csv")

    need_header_per_fold = not os.path.exists(per_fold_csv)
    need_header_summary  = not os.path.exists(summary_csv)

    f_per = open(per_fold_csv, "a", newline="", encoding="utf-8")
    w_per = csv.writer(f_per)
    if need_header_per_fold:
        w_per.writerow(["L", "ts_rate", "fold",
                        "val_loss", "val_acc1", "val_acc5", "val_f1", "val_recall"])

    f_sum = open(summary_csv, "a", newline="", encoding="utf-8")
    w_sum = csv.writer(f_sum)
    if need_header_summary:
        w_sum.writerow(["L", "ts_rate",
                        "mean_acc1", "std_acc1",
                        "mean_f1", "std_f1",
                        "mean_recall", "std_recall"])

    total_cfg = len(L_LIST) * len(TS_RATE_LIST)
    print(f"[Grid] {len(L_LIST)} (L) × {len(TS_RATE_LIST)} (ts_rate) = {total_cfg} configs")

    for L in L_LIST:
        for ts_rate in TS_RATE_LIST:
            print(f"\n==== Config: L={L}, ts_rate={ts_rate} ====")

            fold_metrics_acc1 = []
            fold_metrics_f1   = []
            fold_metrics_rec  = []

            for fold_idx, fold_name in enumerate(folds, start=1):
                fold_root = os.path.join(FOLDS_ROOT, fold_name)
                train_root = os.path.join(fold_root, "train_all_balanced")  # 训练集根目录
                val_root   = os.path.join(fold_root, "val")

                # 训练集：class folder 0..7
                tr_paths, tr_labels = load_train_folder_by_class(train_root)
                # 验证集：flat + filename prefix label
                va_paths, va_labels = load_flat_with_prefix_label(val_root)

                # 为当前配置和fold建一个子目录
                cfg_dir = os.path.join(OUT_DIR, f"L{L}_S{ts_rate}", f"fold{fold_idx}")
                os.makedirs(cfg_dir, exist_ok=True)

                vloss, vacc1, vacc5, vf1, vrec = train_one_fold(
                    L, ts_rate, fold_idx,
                    tr_paths, tr_labels,
                    va_paths, va_labels,
                    device, cfg_dir
                )

                fold_metrics_acc1.append(vacc1)
                fold_metrics_f1.append(vf1)
                fold_metrics_rec.append(vrec)

                # 记录 per-fold 行
                w_per.writerow([L, ts_rate, fold_idx,
                                f"{vloss:.4f}", f"{vacc1:.4f}",
                                f"{vacc5:.4f}", f"{vf1:.4f}", f"{vrec:.4f}"])
                f_per.flush()

            # 汇总 5 折均值 & 方差
            acc_mu = float(np.mean(fold_metrics_acc1))
            acc_sd = float(np.std(fold_metrics_acc1))
            f1_mu  = float(np.mean(fold_metrics_f1))
            f1_sd  = float(np.std(fold_metrics_f1))
            rec_mu = float(np.mean(fold_metrics_rec))
            rec_sd = float(np.std(fold_metrics_rec))

            print(f"[Summary L={L}, ts_rate={ts_rate}] "
                  f"acc1={acc_mu:.4f}±{acc_sd:.4f}, "
                  f"f1={f1_mu:.4f}±{f1_sd:.4f}, "
                  f"recall={rec_mu:.4f}±{rec_sd:.4f}")

            w_sum.writerow([L, ts_rate,
                            f"{acc_mu:.4f}", f"{acc_sd:.4f}",
                            f"{f1_mu:.4f}",  f"{f1_sd:.4f}",
                            f"{rec_mu:.4f}", f"{rec_sd:.4f}"])
            f_sum.flush()

    f_per.close()
    f_sum.close()
    print("\n==> Done. Results saved to:")
    print("   -", per_fold_csv)
    print("   -", summary_csv)


if __name__ == "__main__":
    main()
