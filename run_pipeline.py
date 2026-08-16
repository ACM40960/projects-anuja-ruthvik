"""
run_pipeline.py — chains the ICU deterioration pipeline end-to-end.

Runs each stage as a subprocess, in dependency order, and stops immediately
if any stage exits non-zero. 03_leakage_audit.py runs twice: once in
--feature-only mode right after 02 (cheap, catches feature bugs before
spending BigQuery quota / time on 04 or model training), and once in
--label-only mode after 04 (needs horizon_labels.parquet to exist).
"""

import subprocess
import sys

SRC = "src"

PIPELINE = [
    f"{SRC}/01_data_extraction.py",
    f"{SRC}/02_preprocessing.py",
    [f"{SRC}/03_leakage_audit.py", "--feature-only"],
    f"{SRC}/04_prediction_horizons.py",
    f"{SRC}/05_horizon_sensitivity.py",
    [f"{SRC}/03_leakage_audit.py", "--label-only"],
    f"{SRC}/06_model_evaluation.py",
    f"{SRC}/07_patient_explanations.py",
    f"{SRC}/08_prepare_app.py",
]


def run_step(step) -> None:
    args = step if isinstance(step, list) else [step]
    label = " ".join(args)
    print(f"\n{'='*70}\nRUNNING {label}\n{'='*70}")
    result = subprocess.run([sys.executable, *args])
    if result.returncode != 0:
        print(f"\n{label} failed (exit code {result.returncode}). Stopping pipeline.")
        sys.exit(result.returncode)


def main():
    for step in PIPELINE:
        run_step(step)
    print("\nPipeline complete. Run `streamlit run app.py` to launch the dashboard.")


if __name__ == "__main__":
    main()
