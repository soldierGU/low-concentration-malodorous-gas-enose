# ml_baselines.py
import os, re, json
from glob import glob
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from numpy.fft import rfft, rfftfreq
from datetime import datetime
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from scipy.stats import kurtosis
from sklearn.neighbors import KNeighborsClassifier
# ---------------- paths ----------------
NUM_ROWS = 400
NUM_CH   = 16

# 与深度学习实验共用相同的预划分 5 折目录。
FOLDS_ROOT = "experiments_5_fold"
TEST_DIR = "split_data_new/test"
OUT_DIR = "runs_ml_5fold"
NUM_FOLDS = 5

# ---------------- IO utils ----------------
def _safe_load_txt(p):
    arr = np.loadtxt(p).astype(np.float32)
    if arr.shape != (NUM_ROWS, NUM_CH):
        raise ValueError(f"bad shape at {p}: {arr.shape}")
    return arr

def _parse_label(fname):
    m = re.match(r"^\s*(\d+)_", os.path.basename(fname))
    if not m:
        raise ValueError(f"bad name: {fname}")
    return int(m.group(1))

def load_class_folder_train(train_root):
    """TRAIN_DIR: class 子目录下的 *.txt"""
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train dir not found: {train_root}")
    paths, labels = [], []
    classes = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root,d))])
    for cls in classes:
        for p in sorted(glob(os.path.join(train_root, cls, "*.txt"))):
            paths.append(p); labels.append(int(cls))
    return np.array(paths), np.array(labels)

def load_flat_folder(flat_dir):
    """VAL/TEST: 扁平目录（允许为空）"""
    if not os.path.isdir(flat_dir):
        print(f"[WARN] flat dir not found: {flat_dir} -> treated as empty")
        return np.array([]), np.array([])
    paths, labels = [], []
    for p in sorted(glob(os.path.join(flat_dir, "*.txt"))):
        try:
            y = _parse_label(p)
            paths.append(p); labels.append(y)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
    return np.array(paths), np.array(labels)

# ---------------- feature engineering ----------------
def _band_powers(x):
    # x: (T,)  baseline-removed
    X = rfft(x)
    freqs = rfftfreq(len(x), d=1.0)  # 采样间隔=1
    psd = (X.real**2 + X.imag**2)
    total = psd.sum() + 1e-12
    def band(a,b):
        m = (freqs>=a) & (freqs<b)
        return psd[m].sum()/total
    return np.array([band(0.0,0.05), band(0.05,0.15), band(0.15,0.5)], dtype=np.float32)

def features_one_signal(s):
    # s: (T,)
    baseline = s[:20].mean()
    steady   = s[-50:].mean()
    T        = len(s)
    ds       = np.diff(s)

    # --- 特征集合 ---
    max_val  = s.max()                         # 最大值
    auc      = np.trapz(s)                     # 面积
    std_all  = s.std()                         # 方差（标准差）
    kurt     = kurtosis(s)                     # 峰度
    max_rise = ds.max() if ds.size > 0 else 0  # 最大上升斜率

    # 频带功率（去基线后更稳）
    x  = s - baseline
    bp = _band_powers(x)  # -> (3,)

    # 合计 1+1+1+1+1+3 = 8 维
    return np.array([max_val, kurt], dtype=np.float32)

def extract_features(sample_2d):
    # sample_2d: (400,16)
    feats = [features_one_signal(sample_2d[:,ch]) for ch in range(NUM_CH)]
    return np.concatenate(feats, axis=0)  # (16 * F,)

def build_Xy(paths, labels):
    """允许空输入：返回空数组，防止 vstack 报错"""
    if paths.size == 0:
        return np.empty((0,)), np.empty((0,), dtype=int)
    X_list = [extract_features(_safe_load_txt(p)) for p in paths]
    X = np.vstack(X_list)
    y = labels.copy()
    return X, y

def corr_prune(X, thr=0.95):
    import pandas as pd
    df = pd.DataFrame(X)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thr)]
    keep = [i for i in range(X.shape[1]) if i not in set(drop_cols)]
    return X[:, keep], keep
# ---------------- main ----------------
def _scores(y_true, y_pred):
    return {
        "acc1": accuracy_score(y_true, y_pred) * 100,
        "F1": f1_score(y_true, y_pred, average="macro") * 100,
        "recall": recall_score(y_true, y_pred, average="macro") * 100,
    }


def _mean_std(values):
    return f"{np.mean(values):.2f}±{np.std(values):.2f}"


def main(selected_models=None):
    # 固定 TEST
    test_paths, test_labels = load_flat_folder(TEST_DIR)
    if test_paths.size == 0:
        raise RuntimeError(f"[ERR] TEST is empty -> {TEST_DIR}")

    X_test, y_test = build_Xy(test_paths, test_labels)
    print(f"[INFO] TEST: {len(test_paths)} files -> X_test {X_test.shape}, y_test {y_test.shape}")

    fold_dirs = [os.path.join(FOLDS_ROOT, f"fold{i}") for i in range(1, NUM_FOLDS + 1)]
    missing_folds = [path for path in fold_dirs if not os.path.isdir(path)]
    if missing_folds:
        raise FileNotFoundError(f"Missing prepared fold directories: {missing_folds}")

    # 模型集合
    models = {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("kbest", SelectKBest(mutual_info_classif, k=18 )),  # 先选120维，可调：80/100/160
            # ("pca", PCA(n_components=0.95, svd_solver="full")),  # 或先关掉试试
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
        ]),
        "RF": Pipeline([
            ("sfm", SelectFromModel(RandomForestClassifier(
                n_estimators=400, n_jobs=-1, class_weight="balanced_subsample", random_state=42
            ), threshold="median")),  # 重要度剪一半
            ("clf", RandomForestClassifier(
                n_estimators=600, max_depth=None, n_jobs=-1,
                class_weight="balanced_subsample", random_state=42))
        ]),

        "LogReg_LBFGS": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",  # 适合多分类
                C=1.0,  # 中等正则化
                max_iter=5000,  # 增加迭代次数
                tol=1e-3,  # 放宽收敛条件
                class_weight="balanced",
                random_state=42
            ))
        ]),

    }
    # KNN 模型集合（k=1,3,5,7）
    knn_models = {
        f"KNN_k{k}": Pipeline([
            ("scaler", StandardScaler()),
            # 如你不想做特征选择，可以去掉下面这一行
            ("kbest", SelectKBest(mutual_info_classif, k=20)),
            ("clf", KNeighborsClassifier(
                n_neighbors=k,
                metric="euclidean",  # 欧式距离
                weights="uniform",  # 均匀权重
                algorithm="auto",
                n_jobs=-1
            ))
        ]) for k in (1, 3, 5, 7)
    }
    models.update(knn_models)
    results = {}

    for name, pipe in models.items():
        if selected_models and name not in selected_models:
            print(f"跳过模型 {name}")
            continue
        print(f"\n=== {name}: prepared 5-fold evaluation ===")
        fold_metrics = {split: {metric: [] for metric in ("acc1", "F1", "recall")}
                        for split in ("validation", "test")}

        for fold, fold_root in enumerate(fold_dirs, start=1):
            train_dir = os.path.join(fold_root, "train_all_balanced")
            val_dir = os.path.join(fold_root, "val")
            train_paths, train_labels = load_class_folder_train(train_dir)
            val_paths, val_labels = load_flat_folder(val_dir)
            if train_paths.size == 0:
                raise RuntimeError(f"[ERR] TRAIN is empty -> {train_dir}")
            if val_paths.size == 0:
                raise RuntimeError(f"[ERR] VAL is empty -> {val_dir}")

            X_train, y_train = build_Xy(train_paths, train_labels)
            X_val, y_val = build_Xy(val_paths, val_labels)
            fold_model = clone(pipe)
            fold_model.fit(X_train, y_train)

            val_scores = _scores(y_val, fold_model.predict(X_val))
            test_scores = _scores(y_test, fold_model.predict(X_test))
            for metric in ("acc1", "F1", "recall"):
                fold_metrics["validation"][metric].append(val_scores[metric])
                fold_metrics["test"][metric].append(test_scores[metric])

            print(
                f" Fold {fold:02d} | "
                f"val acc {val_scores['acc1']:.2f}% F1 {val_scores['F1']:.2f} recall {val_scores['recall']:.2f} | "
                f"test acc {test_scores['acc1']:.2f}% F1 {test_scores['F1']:.2f} recall {test_scores['recall']:.2f}"
            )

        results[name] = {
            split: {metric: _mean_std(values) for metric, values in metrics.items()}
            for split, metrics in fold_metrics.items()
        }

    print("\n=== Classical ML baselines: prepared 5-fold summary ===")
    for name, metrics in results.items():
        val = metrics["validation"]
        test = metrics["test"]
        print(
            f"{name} | val: acc1 {val['acc1']} F1 {val['F1']} recall {val['recall']} | "
            f"test: acc1 {test['acc1']} F1 {test['F1']} recall {test['recall']}"
        )

    # 落盘
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(OUT_DIR, f"ml_5fold_summary_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 同步保存 CSV
    with open(os.path.join(OUT_DIR, f"ml_5fold_summary_{ts}.csv"), "w", encoding="utf-8") as f:
        f.write("model,val_acc1,val_F1,val_recall,test_acc1,test_F1,test_recall\n")
        for name, metrics in results.items():
            val = metrics["validation"]
            test = metrics["test"]
            f.write(
                f"{name},{val['acc1']},{val['F1']},{val['recall']},"
                f"{test['acc1']},{test['F1']},{test['recall']}\n"
            )

if __name__ == "__main__":
    main(selected_models=["SVM_RBF", "KNN_k1", "RF"])
