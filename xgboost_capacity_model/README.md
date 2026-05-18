# XGBoost Capacity Model (EU)

Module to train an XGBoost regression model that predicts `electrical_capacity` for EU renewable power plants.

Folder structure
- `train_xgb_eu.py` - main preprocessing and training script (streaming, libsvm external memory training).
- `benchmark.py` - quick sample benchmark runner and full-run time estimator.
- `tune_hyperparams.py` - randomized grid-style hyperparameter tuning utility.
- `requirements.txt` - Python dependencies.
- `outputs/` - generated artifacts (models, metrics, feature importance, training curve, libsvm files).
- `logs/` - runtime logs.

Hardware notes
- Designed to use an NVIDIA CUDA GPU when available.
- Works on CPU if CUDA is unavailable; training falls back automatically.

Installation
1. Create and activate a Python environment (recommended Python 3.9+).
2. Install requirements:

```bash
pip install -r xgboost_capacity_model/requirements.txt
```

Quick benchmark

```bash
python xgboost_capacity_model/benchmark.py --sample_rows 300000 --num_boost_round 300
```

Full training

```bash
python xgboost_capacity_model/train_xgb_eu.py
```

Optional short commands

```bash
python xgboost_capacity_model/train_xgb_eu.py --sample_rows 1000 --num_boost_round 5 --output_dir xgboost_capacity_model/outputs/test_run
python xgboost_capacity_model/train_xgb_eu.py --output_dir xgboost_capacity_model/outputs/final_run_tuned
```

Hyperparameter tuning

```bash
python xgboost_capacity_model/tune_hyperparams.py --n_trials 12 --num_boost_round 600
```

Outputs explanation
- `train.libsvm` / `valid.libsvm`: SVMLight-format training and validation files used with XGBoost external memory.
- `xgb_model.json`: saved XGBoost model.
- `metrics.json`: RMSE, MAE, R2 and timing fields.
- `feature_importance.csv`: feature importance scores.
- `training_curve.csv`: iteration vs eval metric.
- `preprocessing_summary.json`: summary of columns, types and categorical cardinalities.

Troubleshooting
- CUDA not found: ensure NVIDIA drivers and CUDA toolkit are installed and that your `xgboost` build includes GPU support. The script falls back to CPU automatically when `--force_cpu` is used or CUDA is unavailable.
- Out-of-memory: reduce `--chunksize` and/or run the sample benchmark to validate pipeline. Consider increasing swap or using a machine with more RAM.
- Slow disk: XGBoost external-memory mode writes cache files; prefer fast NVMe storage.
