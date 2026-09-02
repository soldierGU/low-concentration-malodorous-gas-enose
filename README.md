# MS-TS-DDA

Code release for the MS-TS-DDA model and the comparison/training scripts used in the Measurement proof stage.

This repository accompanies the article **“An all-in-one electronic nose instrument for low-concentration malodorous gas measurement and classification.”** The article is currently at the proof stage; the final citation and DOI will be added after publication.

This repository intentionally excludes all private training data, cached arrays, checkpoints, logs, figures, and result tables.

## Main Files

- `Network_1127_baseline_AP_TSDA_DA_L.py`: MS-TS-DDA implementation. The main entry class is `MScale_TSDDA_Net_1127`.
- `train_ablation_1127.py`: 5-fold training/evaluation for the MS-TS-DDA ablation settings, including the full model setting `B5_full_AA_TSDA_DA`.
- `train_experiment_A_1127.py`: grid search for block count `L` and `ts_rate`.
- `train_model_compare_5fold_1127.py`: unified 5-fold training template for comparison models in `model_for_compare/`.
- `train_mptsnet_compare_5fold.py`, `train_smellnet_compare_5fold.py`, `train_timemil_compare_5fold.py`: comparison scripts for models that need special construction or training details.
- `ml_baseline.py`: classical machine-learning baselines evaluated on the same prepared 5-fold split.
- `model_for_compare/`: comparison model implementations.

## Expected Data Layout

Private datasets are not included. Place them locally with this layout when reproducing experiments:

```text
experiments_5_fold/
  fold1/
    train_all_balanced/
      0/*.txt
      ...
      7/*.txt
    val/*.txt
  ...
  fold5/
    train_all_balanced/
    val/

split_data_new/
  test/*.txt
```

Each sample is expected to be a text matrix with shape `(400, 16)`. Training loaders transpose it to `(16, 400)`.

## Environment

The original experiments were run with:

```text
Python environment: condaEcust
torch==2.6.0+cu126
numpy==1.25.1
scikit-learn==1.6.1
thop==2.0.14
```

Example setup:

```bash
conda create -n condaEcust python=3.10
conda activate condaEcust
python -m pip install -r requirements.txt
```

If CUDA-specific PyTorch wheels are needed, install PyTorch following the official wheel selector for your CUDA version.

## Example Commands

Run MS-TS-DDA ablation/full-model training:

```bash
python train_ablation_1127.py
```

Run the `L` and `ts_rate` grid search:

```bash
python train_experiment_A_1127.py
```

Run comparison model training:

```bash
python train_model_compare_5fold_1127.py
```

Run the classical machine-learning baselines on the same prepared folds:

```bash
python ml_baseline.py
```

The scripts write outputs to ignored directories such as `ablation_*`, `compare_models_5fold*`, and `npy_cache/`.

## Third-Party Notice

The TimeMIL comparison implementation is a local adaptation of the published architecture and public reference implementation. The upstream repository did not contain an explicit software license when checked on 2026-09-02. See [`model_for_compare/TIMEMIL_NOTICE.md`](model_for_compare/TIMEMIL_NOTICE.md) before reusing or redistributing that implementation.
