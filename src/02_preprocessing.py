import json
import os
import numpy as np
import pandas as pd

DATA_DIR = "./data"
OBSERVATION_WINDOW_HOURS = 6
MIMIC_ROOT = "./mimic-iv-3.1"
HOSP_DIR = os.path.join(MIMIC_ROOT, "hosp")

# Physiologically plausible ranges — anything outside is dropped as an artifact
PLAUSIBLE_RANGES = {
    "heart_rate": (20, 250),
    "sbp": (40, 260),
    "dbp": (20, 200),
    "resp_rate": (4, 60),
    "spo2": (50, 100),
    "temperature_f": (85, 110),
    "gcs_eye": (1, 4),
    "gcs_verbal": (1, 5),
    "gcs_motor": (1, 6),
    "lactate": (0.1, 30),
    "wbc": (0.1, 100),
    "creatinine": (0.1, 20),
    "platelets": (1, 1500),
    "bicarbonate": (5, 50),
    "potassium": (1.5, 9),
    "sodium": (100, 180),
}


def verify_window_config() -> None:
    """Ensure Step 3 is consuming extracts created with the same 6h window."""
    path = f"{DATA_DIR}/window_config.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Re-run the fixed Step 2 extraction before preprocessing."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    extracted_obs = cfg.get("observation_window_hours")
    if extracted_obs != OBSERVATION_WINDOW_HOURS:
        raise RuntimeError(
            "Observation-window mismatch: Step 2 extracted "
            f"{extracted_obs}h but Step 3 is configured for "
            f"{OBSERVATION_WINDOW_HOURS}h."
        )

    print(
        f"Temporal configuration verified: Step 2 and Step 3 both use "
        f"0-{OBSERVATION_WINDOW_HOURS}h for time-varying predictors."
    )


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat, (lo, hi) in PLAUSIBLE_RANGES.items():
        mask = df["feature_name"] == feat
        out_of_range = mask & ((df["valuenum"] < lo) | (df["valuenum"] > hi))
        n_dropped = out_of_range.sum()
        if n_dropped:
            print(f"  dropping {n_dropped} implausible '{feat}' readings")
        df = df[~out_of_range]
    return df


def aggregate_readings(readings: pd.DataFrame, cohort_stay_ids) -> pd.DataFrame:
    """Turn long-format (stay_id, feature_name, charttime, valuenum) into
    one row per stay with mean/min/max/last/count per feature."""
    readings = readings.sort_values("charttime")

    agg = readings.groupby(["stay_id", "feature_name"])["valuenum"].agg(
        ["mean", "min", "max", "count"]
    )
    last = readings.groupby(["stay_id", "feature_name"])["valuenum"].last()
    agg["last"] = last
    agg = agg.reset_index()

    wide = agg.pivot(index="stay_id", columns="feature_name", values=["mean", "min", "max", "count", "last"])
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
    wide = wide.reindex(cohort_stay_ids)  # ensure every cohort stay has a row, even if all-missing
    return wide


def build_comorbidity_count(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return a leakage-safe placeholder comorbidity feature.

    MIMIC-IV diagnoses_icd is an admission-level coding table and does not provide
    a reliable event timestamp that lets us prove a diagnosis was known during
    the first 6 ICU hours. Counting all diagnoses from the hospitalization would
    leak information from after the observation window.

    Therefore n_diagnoses is deliberately set to 0 for this primary analysis.
    A future extension can use a comorbidity score derived only from information
    demonstrably available before or at ICU admission.
    """
    out = cohort[["stay_id"]].copy()
    out["n_diagnoses"] = 0
    print("  Leakage-safe comorbidity handling: n_diagnoses set to 0 for all stays")
    return out



def assert_observation_window(readings: pd.DataFrame, cohort: pd.DataFrame, *, key: str, name: str) -> None:
    """Abort if any raw measurement falls outside [intime, intime + 6h]."""
    required = {key, "charttime"}
    missing = required - set(readings.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}")

    mapping = cohort[[key, "intime"]].drop_duplicates()
    merged = readings.merge(mapping, on=key, how="left", validate="many_to_one")
    if merged["intime"].isna().any():
        raise RuntimeError(f"{name}: some readings could not be mapped to cohort intime")

    charttime = pd.to_datetime(merged["charttime"], utc=True)
    intime = pd.to_datetime(merged["intime"], utc=True)
    offsets = (charttime - intime).dt.total_seconds() / 3600.0

    min_offset = float(offsets.min()) if len(offsets) else float("nan")
    max_offset = float(offsets.max()) if len(offsets) else float("nan")
    violations = (offsets < 0) | (offsets > OBSERVATION_WINDOW_HOURS)

    print(f"  {name}: min={min_offset:.3f}h, max={max_offset:.3f}h, violations={int(violations.sum())}")

    if violations.any():
        bad = merged.loc[violations, [key, "charttime", "intime"]].head(10)
        raise RuntimeError(
            f"TEMPORAL LEAKAGE in {name}: {int(violations.sum())} readings outside "
            f"hours 0-{OBSERVATION_WINDOW_HOURS}. Sample:\n{bad.to_string(index=False)}"
        )

def main():
    verify_window_config()

    cohort = pd.read_parquet(f"{DATA_DIR}/cohort.parquet")
    vitals = pd.read_parquet(f"{DATA_DIR}/vitals_raw.parquet")
    labs = pd.read_parquet(f"{DATA_DIR}/labs_raw.parquet")
    demo = pd.read_parquet(f"{DATA_DIR}/demographics.parquet")

    # NOTE (fixed): preprocessing builds features for the ENTIRE base cohort
    # from 01 -- it deliberately does NOT read horizon_labels.parquet or filter
    # by any horizon's eligibility here. Two reasons:
    #   1. Feature engineering shouldn't need to know about label definitions
    #      at all -- keeping them decoupled means 02 and 04 can run in either
    #      order, and this file no longer silently depends on 04 having been
    #      run first with a specific PREDICTION_HORIZONS list.
    #   2. The previous version filtered to 24h-eligible stays HERE, before
    #      saving feature_matrix_raw.parquet. Since 05_horizon_sensitivity.py
    #      reuses that same shared file for the 36h/48h horizons too, any
    #      stay that was ineligible-for-24h-but-eligible-for-48h was silently
    #      dropped from the 48h analysis as well. Eligibility is horizon-
    #      specific, so filtering it must happen per-horizon, at the point
    #      each horizon's labels are actually joined in (06 for the primary
    #      24h model, 05 per-horizon for sensitivity) -- not baked into the
    #      one shared feature matrix.
    cohort["intime"] = pd.to_datetime(cohort["intime"], utc=True)
    vitals["charttime"] = pd.to_datetime(vitals["charttime"], utc=True)
    labs["charttime"] = pd.to_datetime(labs["charttime"], utc=True)

    print(f"Validating fixed {OBSERVATION_WINDOW_HOURS}-hour observation window before aggregation...")
    assert_observation_window(vitals, cohort, key="stay_id", name="Vitals")
    assert_observation_window(labs, cohort, key="hadm_id", name="Labs")

    # Join labs with cohort to get stay_id (labs are keyed by hadm_id in raw extraction)
    labs = labs.merge(cohort[["hadm_id", "stay_id"]], on="hadm_id", how="left")
    labs = labs[["stay_id", "itemid", "charttime", "valuenum", "feature_name"]]

    print("Clipping physiologically implausible outliers...")
    vitals = clip_outliers(vitals)
    labs = clip_outliers(labs)

    print("Aggregating vitals into per-stay features...")
    vitals_wide = aggregate_readings(vitals, cohort["stay_id"])
    print("Aggregating labs into per-stay features...")
    labs_wide = aggregate_readings(labs, cohort["stay_id"])

    combined = vitals_wide.join(labs_wide, how="outer")
    combined = combined.reset_index().rename(columns={"index": "stay_id"})

    print("Adding comorbidity count...")
    comorbidity = build_comorbidity_count(cohort)

    print("Merging demographics + shock index...")
    features = combined.merge(demo, on="stay_id", how="left")
    features = features.merge(comorbidity, on="stay_id", how="left")

    # Shock index = HR / SBP (last values) — left as NaN if either is missing;
    # imputed later in Step 4 alongside everything else, post-split.
    features["shock_index"] = features["heart_rate_last"] / features["sbp_last"].replace(0, np.nan)

    features["gender"] = (features["gender"] == "M").astype(int)  # 1=male, 0=female
    features = pd.get_dummies(features, columns=["first_careunit", "admission_type"], dummy_na=False)

    os.makedirs(DATA_DIR, exist_ok=True)

    # No labels merged in here, and no eligibility filtering -- this file
    # contains one row per stay in the FULL base cohort (all MIN_LOS_HOURS-
    # qualifying stays from 01). Whatever script needs a specific horizon's
    # labels (06 for the primary 24h model, 05 for the sensitivity sweep)
    # reads horizon_labels.parquet from 04 itself and does its own
    # horizon_hours==X, eligible==1 filter + inner join against this file.
    features.to_parquet(f"{DATA_DIR}/feature_matrix_raw.parquet", index=False)

    print(f"\nObservation-window lock: PASS -- all time-varying features were "
          f"validated within hours 0-{OBSERVATION_WINDOW_HOURS} before aggregation.")
    print(f"Feature matrix (pre-imputation, FULL base cohort, no label/horizon "
          f"filtering applied): {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"Saved to {DATA_DIR}/feature_matrix_raw.parquet")
    print("\nNote: this file has NO labels and NO horizon-eligibility filtering.")
    print("Run 04_prediction_horizons.py (any time before or after this script) to")
    print("produce horizon_labels.parquet, which 06_model_evaluation.py and")
    print("05_horizon_sensitivity.py each join against this feature matrix themselves,")
    print("filtered to their own horizon_hours/eligible criteria.")


if __name__ == "__main__":
    main()
