# MS-TS-DDA

Official code release for the MS-TS-DDA model and the comparison/training scripts accompanying our article in *Measurement*.

## Associated Paper

### [An all-in-one electronic nose instrument for low-concentration malodorous gas measurement and classification](https://doi.org/10.1016/j.measurement.2026.122903)

**Authors:** Chenlong Gu<sup>a,&#42;</sup>, Qianshen Wu<sup>b</sup>, Nan Wang<sup>a</sup>, Yuxuan Zhang<sup>c,d</sup>, Sebastian Bader<sup>d</sup>, Xiaofeng Ling<sup>a</sup>, Yongjing Wan<sup>a</sup>, Daqi Gao<sup>a,&#42;</sup>

<sup>a</sup> School of Information Science and Engineering, East China University of Science and Technology, Shanghai 200237, China<br>
<sup>b</sup> ParisTech Elite Institute of Technology, Shanghai Jiao Tong University, Shanghai 200240, China<br>
<sup>c</sup> College of Intelligent Science and Engineering, Beijing University of Agriculture, Beijing 102206, China<br>
<sup>d</sup> Department of Computer and Electrical Engineering, Mid Sweden University, Sundsvall 85170, Sweden

<sup>&#42;</sup> Corresponding authors.

**Journal:** *Measurement*<br>
**Status:** Published online<br>
**Year:** 2026<br>
**Article number:** 122903<br>
**DOI:** [10.1016/j.measurement.2026.122903](https://doi.org/10.1016/j.measurement.2026.122903)<br>
**Article page:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0263224126026138)

This repository intentionally excludes all private training data, cached arrays, checkpoints, logs, figures, and result tables.

## Citation

If you use this code in your research, please cite:

> Gu, C., Wu, Q., Wang, N., Zhang, Y., Bader, S., Ling, X., Wan, Y., and Gao, D. (2026). An all-in-one electronic nose instrument for low-concentration malodorous gas measurement and classification. *Measurement*, 122903. https://doi.org/10.1016/j.measurement.2026.122903

```bibtex
@article{Gu2026AllInOneENose,
  title   = {An all-in-one electronic nose instrument for low-concentration malodorous gas measurement and classification},
  author  = {Gu, Chenlong and Wu, Qianshen and Wang, Nan and Zhang, Yuxuan and Bader, Sebastian and Ling, Xiaofeng and Wan, Yongjing and Gao, Daqi},
  journal = {Measurement},
  year    = {2026},
  pages   = {122903},
  doi     = {10.1016/j.measurement.2026.122903},
  url     = {https://doi.org/10.1016/j.measurement.2026.122903}
}
```

## Main Files

- `Network_1127_baseline_AP_TSDA_DA_L.py`: MS-TS-DDA implementation. The main entry class is `MScale_TSDDA_Net_1127`.
- `train_ablation_1127.py`: 5-fold training/evaluation for the MS-TS-DDA ablation settings, including the full model setting `B5_full_AA_TSDA_DA`.
- `train_experiment_A_1127.py`: grid search for block count `L` and `ts_rate`.
- `train_model_compare_5fold_1127.py`: unified 5-fold training template for comparison models in `model_for_compare/`.
- `train_mptsnet_compare_5fold.py`, `train_smellnet_compare_5fold.py`: comparison scripts for models that need special construction or training details.
- `ml_baseline.py`: classical machine-learning baselines evaluated on the same prepared 5-fold split.
- `model_for_compare/`: comparison model implementations.

The TimeMIL comparison results are reported in the paper, but its adapted
implementation is not redistributed because the upstream repository did not
provide an explicit software license when this repository was released.

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

## License

Original code in this repository is released under the [MIT License](LICENSE).

Some comparison-model implementations are derived from or based on third-party
research projects and may be subject to separate terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

The adapted TimeMIL implementation is not redistributed because its upstream
repository did not provide an explicit software license at the time of this
release.
