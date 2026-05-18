#!/usr/bin/env python3
"""
Quick benchmark harness for xgboost_capacity_model.

Creates a small sample libsvm dataset and runs a short training to estimate full-run time.
"""
import argparse
import time
import os
import subprocess
import json


def count_rows(path):
    # Count lines (minus header) efficiently
    with open(path, "r") as fh:
        count = sum(1 for _ in fh) - 1
    return max(0, count)


def main():
    parser = argparse.ArgumentParser(description="Benchmark xgboost_capacity_model pipeline")
    parser.add_argument("--sample_rows", type=int, default=300000)
    parser.add_argument("--num_boost_round", type=int, default=300)
    parser.add_argument("--input", default="archive/renewable_power_plants_EU.csv")
    parser.add_argument("--output_dir", default="xgboost_capacity_model/outputs/bench")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cmd = [
        "python",
        "xgboost_capacity_model/train_xgb_eu.py",
        "--input",
        args.input,
        "--sample_rows",
        str(args.sample_rows),
        "--num_boost_round",
        str(args.num_boost_round),
        "--output_dir",
        args.output_dir,
    ]
    print("Running sample benchmark:", " ".join(cmd))
    t0 = time.time()
    subprocess.check_call(cmd)
    elapsed = time.time() - t0
    print(f"Sample run elapsed: {elapsed:.2f} sec")

    # Try to estimate full-run time by counting rows in input
    try:
        total_rows = count_rows(args.input)
        est_total = elapsed * (total_rows / args.sample_rows) if args.sample_rows > 0 else None
        print(f"Total rows in input (estimated): {total_rows}")
        if est_total:
            print(f"Estimated full-run time (linear scale): {est_total/3600:.2f} hours")
    except Exception as e:
        print("Could not estimate full-run time:", e)


if __name__ == "__main__":
    main()
