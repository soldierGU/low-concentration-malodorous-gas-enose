# -*- coding: utf-8 -*-
"""
Ablation experiments for MS-TS-DDA (L=3, ts_rate=4):
  - 3 switches: Anti-aliasing pooling (AA), TS-DA, Depth-DA (DA)
  - 6 representative combinations (B0 ~ B5)

Data:
  - Same 5-fold augmented dataset structure as train_experiment_A_1127.py:
      experiments_5_fold/
        fold1/
          train_all_balanced/0..7/*.txt
          val/*.txt
        ...
        fold5/...

Outputs:
  - OUT_DIR/per_fold_results.csv   # 每一折的详细指标
  - OUT_DIR/summary_results.csv    # 每个 ablation 组合跨5折的均值±方差
  - 每个组合的 best 模型与 fold summary json
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

# ====== 你的网络入口 ======
from Network_1127_baseline_AP_TSDA_DA_L import MScale_TSDDA_Net_1127


# ----------------- 路径与常量 -----------------
FOLDS_ROOT = "experiments_5_fold"                # 5折增强数据根目录
OUT_DIR    = "ablation_AA_TSDA_DA_1127_L3S4"     # 实验输出目录
CACHE_ROOT = "npy_cache"
USE_NPY_CACHE = True

NUM_ROWS = 400
NUM_CH   = 16
BATCH    = 32
EPOCHS   = 200

# 固定结构：最佳配置 L=3, ts_rate=4
FIXED_L       = 3
FIXED_TS_RATE = 4

# 基础 widths（根据 L 截断）
WIDTHS_BASE = (64, 96, 128, 160, 192)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- Ablation 组合定义 -----------------
"""
6 组开关组合（B0~B5）：
  - use_aa   : 是否使用 anti-aliasing pooling
  - use_tsda : 是否使用 TS-DA
  - use_da   : 是否使用 Depth-DA（跨层动态聚合）
"""
ABLATION_CONFIGS = [
    # name, use_aa, use_tsda, use_da
    ("B0_noAA_noTSDA_noDA", False, False, False),  # 纯基线
    ("B1_AA_only",          True,  False, False),  # 只 AA
    ("B2_TSDA_only",        False, True,  False),  # 只 TS-DA
    ("B3_DA_only",          False, False, True ),  # 只 DA
    ("B4_TSDA_DA",          False, True,  True ),  # TS-DA + DA
    ("B5_full_AA_TSDA_DA",  True,  True,  True ),  # 全部开启
    ("B6_AA_TSDA",          True,  True,  False),  # AA + TS-DA
    ("B7_AA_DA",            True,  False, True ),  # AA + Depth-DA
]

# Set to a set of config names to run only selected ablations.
# Keep as None for the public release so all ablation configs are reproducible.
RUN_ONLY_CONFIGS = None


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
    """读取单个样本，默认 shape=(400,16)。"""
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


class CachedNpyDataset(Dataset):
    """Dataset backed by cached x.npy/y.npy arrays with shape (N, 16, 400)."""
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        if self.x.ndim != 3 or self.x.shape[1:] != (NUM_CH, NUM_ROWS):
            raise ValueError(f"Unexpected cached x shape at {x_path}: {self.x.shape}")
        if len(self.x) != len(self.y):
            raise ValueError(f"Cached x/y length mismatch: {x_path}, {y_path}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.asarray(self.x[idx], dtype=np.float32)
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def cache_dir_for(source_root):
    rel = os.path.relpath(os.path.abspath(source_root), os.getcwd())
    return os.path.join(CACHE_ROOT, rel)


def load_dataset_from_root(source_root, is_train_root):
    if USE_NPY_CACHE:
        cached_dir = cache_dir_for(source_root)
        x_path = os.path.join(cached_dir, "x.npy")
        y_path = os.path.join(cached_dir, "y.npy")
        if os.path.isfile(x_path) and os.path.isfile(y_path):
            print(f"[cache] {source_root} -> {cached_dir}")
            return CachedNpyDataset(x_path, y_path)

    if is_train_root:
        paths, labels = load_train_folder_by_class(source_root)
    else:
        paths, labels = load_flat_with_prefix_label(source_root)
    print(f"[txt] {source_root}: {len(paths)} samples")
    return EnoDataset(paths, labels)


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
    val/ 结构扁平:
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


# ----------------- 模型构造：根据 ablation 配置切换开关 -----------------
def build_model_ablation(config_name, use_aa, use_tsda, use_da):
    """
    根据 ablation 设置构造 MScale_TSDDA_Net_1127：
      - L 固定为 3
      - ts_rate 固定为 4
      - use_aa / use_tsda / use_da 分别控制 AA / TS-DA / Depth-DA
    """
    L = FIXED_L
    ts_rate = FIXED_TS_RATE

    assert 1 <= L <= len(WIDTHS_BASE)
    widths = WIDTHS_BASE[:L]  # e.g. (64, 96, 128)

    # Depth-DA 的配置：
    #   - 若 use_da=False: depth_k 全 0，dynamic 全 False => 相当于关闭 DA
    #   - 若 use_da=True : 给一个你最终论文中采用的典型 DA 配置
    if use_da:
        # 可以根据你在论文中的最终设置略微调整
        depth_k_tuple = (0, 1, 2)          # 在较深层开启 depth-wise aggregation
        depth_dynamic_tuple = (False, False, True)
    else:
        depth_k_tuple = (0, 0, 0)
        depth_dynamic_tuple = (False, False, False)

    model = MScale_TSDDA_Net_1127(
        num_channels=16,
        num_classes=8,
        stem_ch=64,
        widths=widths,
        ks=(1, 3, 5),
        dilations=(1, 2, 3),
        ts_rate=ts_rate,
        ts_mode='joint2d',
        ts_boundary='zpad',
        depth_k_tuple=depth_k_tuple,
        depth_dynamic_tuple=depth_dynamic_tuple,
        per_channel_depth=False,
        use_antialias=use_aa,
        use_tsda=use_tsda,
        baseline=False,  # 这里用显式开关控制，不再依赖 baseline 标志
    )
    return model


# ----------------- 单折训练 -----------------
def train_one_fold(config_name, use_aa, use_tsda, use_da,
                   fold_idx, train_dataset, val_dataset, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    set_seeds(42 + fold_idx)  # 不同fold稍微偏移seed

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True
    )

    model = build_model_ablation(config_name, use_aa, use_tsda, use_da).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_f1 = -1.0
    best_state = None
    ckpt_path = os.path.join(save_dir, f"best_{config_name}_fold{fold_idx}.pt")
    history = []

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
        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc1": train_acc1,
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "val_f1": val_f1,
            "val_recall": val_rec,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

        if ep == 1 or ep % 10 == 0:
            print(f"[{config_name}, F{fold_idx}, Ep{ep:03d}] "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc1:5.2f}% | "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc1:5.2f}%, "
                  f"val_f1={val_f1:5.2f}, val_rec={val_rec:5.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_acc1, val_acc5, val_f1, val_rec = evaluate(
        model, val_loader, device, criterion
    )

    history_path = os.path.join(save_dir, f"history_{config_name}_fold{fold_idx}.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "train_acc1",
                "val_loss", "val_acc1", "val_acc5", "val_f1", "val_recall", "lr",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    summary_path = os.path.join(save_dir, f"val_summary_{config_name}_fold{fold_idx}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": config_name,
            "use_aa": bool(use_aa),
            "use_tsda": bool(use_tsda),
            "use_da": bool(use_da),
            "L": FIXED_L,
            "ts_rate": FIXED_TS_RATE,
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

    folds = [d for d in sorted(os.listdir(FOLDS_ROOT))
             if d.startswith("fold")]
    assert len(folds) == 5, f"Expect 5 folds under {FOLDS_ROOT}, got {folds}"

    per_fold_csv = os.path.join(OUT_DIR, "per_fold_results.csv")
    summary_csv  = os.path.join(OUT_DIR, "summary_results.csv")

    need_header_per = not os.path.exists(per_fold_csv)
    need_header_sum = not os.path.exists(summary_csv)

    f_per = open(per_fold_csv, "a", newline="", encoding="utf-8")
    w_per = csv.writer(f_per)
    if need_header_per:
        w_per.writerow([
            "config", "use_aa", "use_tsda", "use_da",
            "L", "ts_rate", "fold",
            "val_loss", "val_acc1", "val_acc5", "val_f1", "val_recall"
        ])

    f_sum = open(summary_csv, "a", newline="", encoding="utf-8")
    w_sum = csv.writer(f_sum)
    if need_header_sum:
        w_sum.writerow([
            "config", "use_aa", "use_tsda", "use_da",
            "L", "ts_rate",
            "mean_acc1", "std_acc1",
            "mean_f1", "std_f1",
            "mean_recall", "std_recall"
        ])

    configs_to_run = [
        cfg for cfg in ABLATION_CONFIGS
        if RUN_ONLY_CONFIGS is None or cfg[0] in RUN_ONLY_CONFIGS
    ]
    print(f"[Ablation] {len(configs_to_run)} configs on L={FIXED_L}, ts_rate={FIXED_TS_RATE}")

    for config_name, use_aa, use_tsda, use_da in configs_to_run:
        print(f"\n==== Config: {config_name} "
              f"(AA={use_aa}, TSDA={use_tsda}, DA={use_da}) ====")

        fold_metrics_acc1 = []
        fold_metrics_f1   = []
        fold_metrics_rec  = []

        for fold_idx, fold_name in enumerate(folds, start=1):
            fold_root = os.path.join(FOLDS_ROOT, fold_name)
            train_root = os.path.join(fold_root, "train_all_balanced")
            val_root   = os.path.join(fold_root, "val")

            train_dataset = load_dataset_from_root(train_root, is_train_root=True)
            val_dataset = load_dataset_from_root(val_root, is_train_root=False)

            cfg_dir = os.path.join(OUT_DIR, config_name, f"fold{fold_idx}")
            os.makedirs(cfg_dir, exist_ok=True)

            vloss, vacc1, vacc5, vf1, vrec = train_one_fold(
                config_name, use_aa, use_tsda, use_da,
                fold_idx,
                train_dataset,
                val_dataset,
                device, cfg_dir
            )

            fold_metrics_acc1.append(vacc1)
            fold_metrics_f1.append(vf1)
            fold_metrics_rec.append(vrec)

            w_per.writerow([
                config_name, int(use_aa), int(use_tsda), int(use_da),
                FIXED_L, FIXED_TS_RATE, fold_idx,
                f"{vloss:.4f}", f"{vacc1:.4f}",
                f"{vacc5:.4f}", f"{vf1:.4f}", f"{vrec:.4f}"
            ])
            f_per.flush()

        acc_mu = float(np.mean(fold_metrics_acc1))
        acc_sd = float(np.std(fold_metrics_acc1))
        f1_mu  = float(np.mean(fold_metrics_f1))
        f1_sd  = float(np.std(fold_metrics_f1))
        rec_mu = float(np.mean(fold_metrics_rec))
        rec_sd = float(np.std(fold_metrics_rec))

        print(f"[Summary {config_name}] "
              f"acc1={acc_mu:.4f}±{acc_sd:.4f}, "
              f"f1={f1_mu:.4f}±{f1_sd:.4f}, "
              f"recall={rec_mu:.4f}±{rec_sd:.4f}")

        w_sum.writerow([
            config_name, int(use_aa), int(use_tsda), int(use_da),
            FIXED_L, FIXED_TS_RATE,
            f"{acc_mu:.4f}", f"{acc_sd:.4f}",
            f"{f1_mu:.4f}",  f"{f1_sd:.4f}",
            f"{rec_mu:.4f}", f"{rec_sd:.4f}"
        ])
        f_sum.flush()

    f_per.close()
    f_sum.close()
    print("\n==> Done. Results saved to:")
    print("   -", per_fold_csv)
    print("   -", summary_csv)

@torch.no_grad()
def test_ablation_all():
    """
    Unified testing for all ablation configs.
    Loads each fold's best model, evaluates on the test set,
    and outputs average test Acc / F1 / Recall.
    """

    print("\n====== Running Unified Test Evaluation ======\n")

    # ---- 读取测试集 ----
    TEST_DIR = os.path.join("split_data_new/test")   # 你实际路径若不同，请改为你的 test 路径
    assert os.path.isdir(TEST_DIR), f"Test dir not found: {TEST_DIR}"

    test_dataset = load_dataset_from_root(TEST_DIR, is_train_root=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True
    )

    results = []

    configs_to_run = [
        cfg for cfg in ABLATION_CONFIGS
        if RUN_ONLY_CONFIGS is None or cfg[0] in RUN_ONLY_CONFIGS
    ]

    for config_name, use_aa, use_tsda, use_da in configs_to_run:
        print(f"[Test] Config = {config_name}")

        fold_acc = []
        fold_f1  = []
        fold_rec = []

        for fold_idx in range(1, 6):
            ckpt_path = os.path.join(
                OUT_DIR, config_name, f"fold{fold_idx}",
                f"best_{config_name}_fold{fold_idx}.pt"
            )
            if not os.path.isfile(ckpt_path):
                print(f"  WARNING: missing ckpt: {ckpt_path}")
                continue

            # 创建模型
            model = build_model_ablation(config_name, use_aa, use_tsda, use_da)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model.to(DEVICE)
            model.eval()

            # 评估
            criterion = nn.CrossEntropyLoss()
            loss, acc1, acc5, f1, rec = evaluate(model, test_loader, DEVICE, criterion)

            print(f"   Fold {fold_idx}: Acc={acc1:.2f}, F1={f1:.2f}, Rec={rec:.2f}")

            fold_acc.append(acc1)
            fold_f1.append(f1)
            fold_rec.append(rec)

        # 统一统计
        acc_mu = np.mean(fold_acc)
        acc_sd = np.std(fold_acc)
        f1_mu  = np.mean(fold_f1)
        f1_sd  = np.std(fold_f1)
        rec_mu = np.mean(fold_rec)
        rec_sd = np.std(fold_rec)

        print(f" ==> {config_name}: Test Acc={acc_mu:.2f}±{acc_sd:.2f}, "
              f"F1={f1_mu:.2f}±{f1_sd:.2f}, Rec={rec_mu:.2f}±{rec_sd:.2f}\n")

        results.append([
            config_name, use_aa, use_tsda, use_da,
            f"{acc_mu:.2f}", f"{acc_sd:.2f}",
            f"{f1_mu:.2f}",  f"{f1_sd:.2f}",
            f"{rec_mu:.2f}", f"{rec_sd:.2f}"
        ])

    # 保存 CSV
    test_csv = os.path.join(OUT_DIR, "test_summary_results.csv")
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "AA", "TSDA", "DA",
                    "acc_mean", "acc_std",
                    "f1_mean",  "f1_std",
                    "rec_mean", "rec_std"])
        w.writerows(results)

    print(f"\nUnified Test Summary Saved to: {test_csv}\n")

if __name__ == "__main__":
    main()
    # test_ablation_all()
