#!/usr/bin/env python3
"""
Train an XGBoost regression model on archive/renewable_power_plants_EU.csv

Features:
- Streaming two-pass preprocessing (chunked) to build consistent categorical encoders
- Writes SVMLight files for external-memory training
- GPU-optimized config with CPU fallback

Usage: see README or run `python train_xgb_eu.py --help`
"""
import os
import time
import json
import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import metrics

import xgboost as xgb


def setup_logger(log_path=None):
    logger = logging.getLogger("xgb_capacity")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def detect_device(force_cpu=False):
    if force_cpu:
        return "cpu", "hist"
    try:
        X_probe = np.array([[0.0], [1.0]], dtype=np.float32)
        y_probe = np.array([0.0, 1.0], dtype=np.float32)
        dprobe = xgb.DMatrix(X_probe, label=y_probe)
        xgb.train(
            {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 1,
                "eta": 1.0,
            },
            dprobe,
            num_boost_round=1,
            verbose_eval=False,
        )
        return "cuda", "hist"
    except Exception:
        return "cpu", "hist"


def safe_downcast_df(df):
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def gather_schema(input_csv, chunksize, drop_cols, target_col, sample_rows=None, logger=None):
    """First pass: scan file in chunks to gather categorical domains and columns summary."""
    if logger is None:
        logger = setup_logger()
    logger.info("Starting schema gathering pass")
    cats = defaultdict(set)
    numeric_cols = set()
    total_rows = 0
    kept_rows = 0
    sample_left = sample_rows
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        for c in drop_cols:
            if c in chunk.columns:
                chunk = chunk.drop(columns=[c])
        chunk = chunk[chunk[target_col].notna()]
        kept_rows += len(chunk)
        if "commissioning_date" in chunk.columns:
            chunk["commissioning_date"] = pd.to_datetime(chunk["commissioning_date"], errors="coerce")
            chunk["commissioning_year"] = chunk["commissioning_date"].dt.year
            chunk["commissioning_month"] = chunk["commissioning_date"].dt.month
            chunk = chunk.drop(columns=["commissioning_date"])
        for col in chunk.columns:
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(chunk[col]):
                numeric_cols.add(col)
            else:
                vals = chunk[col].dropna().unique()
                for v in vals:
                    cats[col].add(str(v))
        if sample_rows is not None:
            sample_left -= len(chunk)
            if sample_left <= 0:
                break
    schema = {
        "total_rows_scanned": total_rows,
        "kept_rows": kept_rows,
        "numeric_columns": sorted(list(numeric_cols)),
        "categorical_cardinalities": {k: len(v) for k, v in cats.items()},
    }
    cats_trimmed = {k: (list(v) if len(v) <= 1000 else []) for k, v in cats.items()}
    schema["categorical_values_preview"] = cats_trimmed
    logger.info("Schema gathering complete")
    return schema, cats


def build_mappings(cats, numeric_cols, logger=None):
    if logger is None:
        logger = setup_logger()
    mappings = {}
    for col, values in cats.items():
        sorted_vals = sorted(list(values))
        mapping = {"__UNKNOWN__": 0}
        idx = 1
        for v in sorted_vals:
            mapping[v] = idx
            idx += 1
        mappings[col] = mapping
    logger.info("Built categorical mappings for %d columns", len(mappings))
    return mappings


def values_to_libsvm(label, values):
    parts = []
    for i, v in enumerate(values, start=1):
        if v != 0.0 and not np.isnan(v):
            parts.append(f"{i}:{v}")
    return f"{label} " + " ".join(parts) + "\n"


def generate_libsvm_files(input_csv, output_dir, chunksize, target_col, drop_cols, mappings, numeric_cols,
                          sample_rows=None, seed=42, logger=None):
    """Second pass: transform chunks and stream write libsvm train/valid files."""
    if logger is None:
        logger = setup_logger()
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.libsvm")
    valid_path = os.path.join(output_dir, "valid.libsvm")
    valid_labels_path = os.path.join(output_dir, "valid_labels.txt")
    for p in [train_path, valid_path, valid_labels_path]:
        try:
            os.remove(p)
        except OSError:
            pass
    rng = __import__("random").Random(seed)
    cat_cols = sorted(list(mappings.keys()))
    num_cols = sorted(list(numeric_cols))
    feature_cols = num_cols + cat_cols
    logger.info("Feature columns count: %d", len(feature_cols))
    written = {"train": 0, "valid": 0}
    sample_left = sample_rows
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        for c in drop_cols:
            if c in chunk.columns:
                chunk = chunk.drop(columns=[c])
        chunk = chunk[chunk[target_col].notna()]
        if "commissioning_date" in chunk.columns:
            chunk["commissioning_date"] = pd.to_datetime(chunk["commissioning_date"], errors="coerce")
            chunk["commissioning_year"] = chunk["commissioning_date"].dt.year
            chunk["commissioning_month"] = chunk["commissioning_date"].dt.month
            chunk = chunk.drop(columns=["commissioning_date"])
        chunk = safe_downcast_df(chunk)
        for c in cat_cols:
            if c in chunk.columns:
                chunk[c] = chunk[c].fillna("__UNKNOWN__").astype(str)
                mapping = mappings[c]
                chunk[c] = chunk[c].map(lambda x: mapping.get(x, 0)).astype(float)
            else:
                chunk[c] = 0.0
        for c in num_cols:
            if c not in chunk.columns:
                chunk[c] = 0.0
        labels = chunk[target_col].to_numpy(dtype=np.float32, copy=False)
        feature_matrix = chunk[feature_cols].to_numpy(dtype=np.float32, copy=False)

        with open(train_path, "a") as ftrain, open(valid_path, "a") as fvalid, open(valid_labels_path, "a") as vlab:
            for i in range(len(chunk)):
                if sample_rows is not None:
                    if sample_left <= 0:
                        break
                    sample_left -= 1

                label = float(labels[i])
                line = values_to_libsvm(label, feature_matrix[i])

                if rng.random() < 0.8:
                    ftrain.write(line)
                    written["train"] += 1
                else:
                    fvalid.write(line)
                    vlab.write(str(label) + "\n")
                    written["valid"] += 1
        if sample_rows is not None and sample_left <= 0:
            break
    preprocessing_summary = {
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "written_counts": written,
    }
    with open(os.path.join(output_dir, "preprocessing_summary.json"), "w") as fh:
        json.dump(preprocessing_summary, fh, indent=2)
    logger.info("Wrote train.libsvm (%d) and valid.libsvm (%d)", written["train"], written["valid"]) 
    return train_path, valid_path, valid_labels_path, preprocessing_summary


def train_xgboost(train_path, valid_path, valid_labels_path, output_dir, num_boost_round, early_stopping_rounds,
                  device, tree_method, params_base, logger=None):
    if logger is None:
        logger = setup_logger()
    logger.info("Preparing DMatrix (will create binary caches if needed)")
    def ensure_binary(path, logger, suffix=".bin"):
        if path.endswith(".libsvm"):
            bin_path = path + suffix
            if not os.path.exists(bin_path):
                logger.info("Converting %s -> %s (this will parse the libsvm file)", path, bin_path)
                dm = xgb.DMatrix(path + "?format=libsvm")
                dm.save_binary(bin_path)
            return bin_path
        return path

    train_bin = ensure_binary(train_path, logger, suffix=".bin")
    valid_bin = ensure_binary(valid_path, logger, suffix=".bin")
    dtrain = xgb.DMatrix(train_bin)
    dvalid = xgb.DMatrix(valid_bin)
    params = params_base.copy()
    params.update({"tree_method": tree_method})
    if device == "cuda":
        params.update({"device": "cuda"})
    else:
        params.update({"device": "cpu"})
    evals_result = {}
    start = time.time()
    logger.info("Starting training: device=%s, tree_method=%s", device, tree_method)
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )
    train_time = time.time() - start
    logger.info("Training completed in %.2f sec", train_time)
    model_path = os.path.join(output_dir, "xgb_model.json")
    bst.save_model(model_path)
    iters = list(range(1, len(evals_result["valid"]["rmse"]) + 1))
    import csv
    with open(os.path.join(output_dir, "training_curve.csv"), "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["iteration", "valid_rmse"])
        for i, v in zip(iters, evals_result["valid"]["rmse"]):
            writer.writerow([i, v])
    fmap = bst.get_score(importance_type="gain")
    with open(os.path.join(output_dir, "feature_importance.csv"), "w") as fh:
        fh.write("feature,importance\n")
        for k, v in sorted(fmap.items(), key=lambda x: -x[1]):
            fh.write(f"{k},{v}\n")
    y_true = np.atleast_1d(np.loadtxt(valid_labels_path, dtype=float))
    y_pred = bst.predict(dvalid)
    rmse = float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    mae = float(metrics.mean_absolute_error(y_true, y_pred))
    r2 = float(metrics.r2_score(y_true, y_pred))
    metrics_out = {"rmse": rmse, "mae": mae, "r2": r2, "train_time_sec": train_time}
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    logger.info("Eval RMSE=%.4f MAE=%.4f R2=%.4f", rmse, mae, r2)
    return metrics_out, bst, evals_result


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost capacity model for EU data")
    parser.add_argument("--input", default="archive/renewable_power_plants_EU.csv")
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--num_boost_round", type=int, default=2000)
    parser.add_argument("--sample_rows", type=int, default=None,
                        help="If set, only process this many rows (for quick runs)")
    parser.add_argument("--output_dir", default="xgboost_capacity_model/outputs")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if GPU is available")
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(os.path.dirname(args.output_dir), "logs", "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)
    t0 = time.time()

    drop_cols = ["energy_source_level_3", "as_of_year", "municipality"]
    target_col = "electrical_capacity"

    t_pass1 = time.time()
    schema, cats = gather_schema(args.input, args.chunksize, drop_cols, target_col, sample_rows=args.sample_rows, logger=logger)
    t_pass1_time = time.time() - t_pass1

    mappings = build_mappings(cats, schema.get("numeric_columns", []), logger=logger)

    t_filegen = time.time()
    train_path, valid_path, valid_labels_path, prep_summary = generate_libsvm_files(
        args.input,
        args.output_dir,
        args.chunksize,
        target_col,
        drop_cols,
        mappings,
        schema.get("numeric_columns", []),
        sample_rows=args.sample_rows,
        seed=42,
        logger=logger,
    )
    t_filegen_time = time.time() - t_filegen

    device, tree_method = detect_device(args.force_cpu)
    params_base = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "lambda": 10.0,
        "alpha": 0.1,
        "seed": 42,
    }

    t_train = time.time()
    metrics_out, bst, evals_result = train_xgboost(
        train_path,
        valid_path,
        valid_labels_path,
        args.output_dir,
        args.num_boost_round,
        args.early_stopping_rounds,
        device,
        tree_method,
        params_base,
        logger=logger,
    )
    t_train_time = time.time() - t_train

    total_time = time.time() - t0
    metrics_out.update({"preprocess_time_sec": float(t_pass1_time + t_filegen_time), "total_time_sec": float(total_time)})
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_out, fh, indent=2)

    with open(os.path.join(args.output_dir, "preprocessing_summary.json"), "w") as fh:
        json.dump({"schema": schema, "mappings_sample": {k: list(v.keys())[:50] for k, v in mappings.items()},
                   "prep": prep_summary}, fh, indent=2)

    logger.info("All done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
