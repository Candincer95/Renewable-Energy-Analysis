#!/usr/bin/env python3
"""Hyperparameter tuning utility for XGBoost capacity model."""

import argparse
import csv
import itertools
import json
import logging
import os
import random
import time

import numpy as np
from sklearn import metrics
import xgboost as xgb


def setup_logger(log_path=None):
    logger = logging.getLogger("xgb_tuning")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
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
    return "cuda", "hist"


def ensure_binary(path, logger, suffix=".bin"):
    # For xgboost>=3.x, text loading is deprecated and external text memory is removed.
    if path.endswith(".libsvm"):
        bin_path = path + suffix
        if not os.path.exists(bin_path):
            logger.info("Converting %s -> %s", path, bin_path)
            dm = xgb.DMatrix(path + "?format=libsvm")
            dm.save_binary(bin_path)
        return bin_path
    return path


def sample_param_grid(n_trials, seed):
    grid = {
        "max_depth": [6, 8, 10],
        "eta": [0.03, 0.05, 0.08],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "lambda": [1.0, 5.0, 10.0],
        "alpha": [0.0, 0.1, 1.0],
    }
    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))
    rng = random.Random(seed)
    rng.shuffle(all_combos)
    picked = all_combos[: min(n_trials, len(all_combos))]
    params_list = []
    for combo in picked:
        params_list.append({k: v for k, v in zip(keys, combo)})
    return params_list


def main():
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters on prepared libsvm files")
    parser.add_argument("--train_libsvm", default="xgboost_capacity_model/outputs/full_run/train.libsvm")
    parser.add_argument("--valid_libsvm", default="xgboost_capacity_model/outputs/full_run/valid.libsvm")
    parser.add_argument("--valid_labels", default="xgboost_capacity_model/outputs/full_run/valid_labels.txt")
    parser.add_argument("--output_dir", default="xgboost_capacity_model/outputs/tuning")
    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--num_boost_round", type=int, default=800)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "tuning.log")
    logger = setup_logger(log_path)

    device, tree_method = detect_device(args.force_cpu)
    logger.info("Tuning start. device=%s tree_method=%s", device, tree_method)

    train_bin = ensure_binary(args.train_libsvm, logger)
    valid_bin = ensure_binary(args.valid_libsvm, logger)
    dtrain = xgb.DMatrix(train_bin)
    dvalid = xgb.DMatrix(valid_bin)
    y_true = np.loadtxt(args.valid_labels, dtype=float)

    base = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": args.seed,
        "tree_method": tree_method,
        "device": device,
    }

    param_list = sample_param_grid(args.n_trials, args.seed)
    results = []
    best = None

    for idx, trial in enumerate(param_list, start=1):
        params = base.copy()
        params.update(trial)
        trial_start = time.time()
        evals_result = {}

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=args.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )
        train_sec = time.time() - trial_start
        best_rmse = float(min(evals_result["valid"]["rmse"]))
        best_iter = int(booster.best_iteration)

        entry = {
            "trial": idx,
            "best_rmse": best_rmse,
            "best_iteration": best_iter,
            "train_time_sec": round(train_sec, 3),
            **trial,
        }
        results.append(entry)

        if best is None or best_rmse < best["best_rmse"]:
            best = entry.copy()
            best_model_path = os.path.join(args.output_dir, "xgb_model_tuned.json")
            booster.save_model(best_model_path)

        logger.info(
            "Trial %d/%d rmse=%.6f iter=%d depth=%s eta=%s subsample=%s colsample=%s",
            idx,
            len(param_list),
            best_rmse,
            best_iter,
            trial["max_depth"],
            trial["eta"],
            trial["subsample"],
            trial["colsample_bytree"],
        )

    # Recreate best model for detailed metrics
    final_params = base.copy()
    for key in ["max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight", "lambda", "alpha"]:
        final_params[key] = best[key]

    evals_result = {}
    final_booster = xgb.train(
        final_params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )
    y_pred = final_booster.predict(dvalid)
    best_metrics = {
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "r2": float(metrics.r2_score(y_true, y_pred)),
    }

    with open(os.path.join(args.output_dir, "tuning_results.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(args.output_dir, "best_params.json"), "w") as fh:
        json.dump({"best_trial": best, "metrics": best_metrics, "device": device}, fh, indent=2)

    logger.info("Tuning finished. Best RMSE=%.6f", best["best_rmse"])
    logger.info("Best params saved to %s", os.path.join(args.output_dir, "best_params.json"))


if __name__ == "__main__":
    main()
