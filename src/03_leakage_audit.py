from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("./data")
OBSERVATION_WINDOW_HOURS = 6
PREDICTION_WINDOW_HOURS = 24
EPS = 1e-9


def _load(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_parquet(path)


def _timestamp_window_check(readings, cohort, *, key, name):
    if readings.empty:
        print(f"  FAIL: {name} has no rows")
        return False

    if key not in readings or "charttime" not in readings:
        print(f"  FAIL: {name} missing {key!r} or 'charttime'")
        return False

    mapping = cohort[[key, "intime"]].drop_duplicates()
    try:
        merged = readings.merge(mapping, on=key, how="left", validate="many_to_one")
    except Exception as exc:
        print(f"  FAIL: {name} could not be mapped one-to-one to cohort timing: {exc}")
        return False

    if merged["intime"].isna().any():
        n = int(merged["intime"].isna().sum())
        print(f"  FAIL: {n} {name} rows have no cohort intime")
        return False

    charttime = pd.to_datetime(merged["charttime"], utc=True)
    intime = pd.to_datetime(merged["intime"], utc=True)
    offsets = (charttime - intime).dt.total_seconds() / 3600.0

    before = offsets < -EPS
    after = offsets > OBSERVATION_WINDOW_HOURS + EPS
    bad = before | after

    print(f"  {name} min offset: {offsets.min():.6f} h")
    print(f"  {name} max offset: {offsets.max():.6f} h")
    print(f"  {name} before ICU admission: {int(before.sum()):,}")
    print(f"  {name} after {OBSERVATION_WINDOW_HOURS}h cutoff: {int(after.sum()):,}")

    if bad.any():
        cols = [key, "charttime", "intime"]
        print("  Sample violations:")
        print(merged.loc[bad, cols].head(10).to_string(index=False))
        return False

    print(f"  PASS: every {name.lower()} reading is inside hours 0-{OBSERVATION_WINDOW_HOURS}")
    return True


def check_raw_feature_timestamps():
    print("\n=== CHECK 1: RAW FEATURE TIMESTAMPS ===")
    cohort = _load("cohort.parquet")
    vitals = _load("vitals_raw.parquet")
    labs = _load("labs_raw.parquet")

    if "intime" not in cohort:
        print("  FAIL: cohort.parquet has no intime")
        return False

    vitals_ok = _timestamp_window_check(vitals, cohort, key="stay_id", name="Vitals")
    labs_ok = _timestamp_window_check(labs, cohort, key="hadm_id", name="Labs")
    return vitals_ok and labs_ok


def check_feature_matrix_columns():
    print("\n=== CHECK 2: FEATURE MATRIX CONTENT ===")
    X = _load("feature_matrix_raw.parquet")

    forbidden_exact = {
        "died_in_icu", "deteriorated_composite", "deathtime", "outtime",
        "dischtime", "hospital_expire_flag", "los_hours"
    }
    forbidden_keywords = (
        "death", "mortality", "deteriorat", "vasopressor", "ventilat",
        "discharge", "outtime", "expire", "los_hours"
    )

    exact_hits = sorted(set(X.columns) & forbidden_exact)
    keyword_hits = sorted(
        c for c in X.columns
        if any(k in c.lower() for k in forbidden_keywords)
    )
    suspicious = sorted(set(exact_hits + keyword_hits))

    if suspicious:
        print(f"  FAIL: outcome/post-outcome-looking predictors found: {suspicious}")
        return False

    id_cols = [c for c in ["stay_id", "subject_id", "hadm_id"] if c in X.columns]
    print(f"  Traceability identifiers present: {id_cols or 'none'}")
    print("  NOTE: identifiers are allowed in the saved matrix only if Step 4 drops them before modelling.")

    non_numeric = X.drop(columns=id_cols, errors="ignore").select_dtypes(exclude=[np.number, bool]).columns.tolist()
    if non_numeric:
        print(f"  FAIL: non-numeric predictors remain: {non_numeric}")
        return False

    print("  PASS: no obvious target/post-outcome columns in predictor matrix")
    return True


def check_labels_separate():
    print("\n=== CHECK 3: LABEL SEPARATION ===")
    X = _load("feature_matrix_raw.parquet")

    # NOTE (fixed): feature_matrix_raw.parquet no longer has a companion
    # labels_matrix.parquet -- 02_preprocessing.py doesn't touch labels at
    # all anymore (see its top-of-file note). Labels for any horizon live
    # only in horizon_labels.parquet from 04. This check now verifies BOTH
    # that (a) no label-ish column ever made it into the feature matrix, and
    # (b) the primary 24h horizon's eligible labels are actually available
    # and joinable against the feature matrix, which is what 06 does.
    leaked = [c for c in ["died_in_icu", "deteriorated_composite", "death_event"] if c in X.columns]
    if leaked:
        print(f"  FAIL: label columns present in feature matrix: {leaked}")
        return False

    horizon_path = DATA_DIR / "horizon_labels.parquet"
    if not horizon_path.exists():
        print(f"  SKIPPED: {horizon_path} not found -- this check needs 04_prediction_horizons.py")
        return None

    horizon_labels = _load("horizon_labels.parquet")
    needed = {"stay_id", "horizon_hours", "eligible", "death_event", "deteriorated_composite"}
    missing = needed - set(horizon_labels.columns)
    if missing:
        print(f"  FAIL: horizon_labels.parquet missing {sorted(missing)}")
        return False

    primary = horizon_labels[(horizon_labels["horizon_hours"] == 24) & (horizon_labels["eligible"] == 1)]
    if primary.empty:
        print("  FAIL: no eligible 24h-horizon rows found in horizon_labels.parquet")
        return False

    joined = X[["stay_id"]].merge(primary[["stay_id"]], on="stay_id", how="inner")
    print(f"  Feature matrix: {len(X):,} stays (full base cohort, no eligibility filter)")
    print(f"  24h-horizon eligible labels: {len(primary):,} stays")
    print(f"  Stays present in both (what 06_model_evaluation.py will actually train/test on): {len(joined):,}")
    if len(joined) == 0:
        print("  FAIL: zero overlap between feature matrix and 24h-eligible labels")
        return False

    print(f"  died_in_icu (death_event) prevalence, eligible 24h rows: {primary['death_event'].mean():.2%}")
    print(f"  deteriorated_composite prevalence, eligible 24h rows: {primary['deteriorated_composite'].mean():.2%}")
    print("  PASS: labels are physically separated from the feature matrix, "
          "and the 24h-eligible label set joins cleanly against it")
    return True


def check_one_stay_per_patient():
    print("\n=== CHECK 4: ONE STAY PER PATIENT ===")
    cohort = _load("cohort.parquet")
    if "subject_id" not in cohort:
        print("  FAIL: subject_id missing from cohort")
        return False
    dup = cohort["subject_id"].duplicated(keep=False)
    n_dup_patients = cohort.loc[dup, "subject_id"].nunique()
    print(f"  Cohort rows: {len(cohort):,}")
    print(f"  Unique patients: {cohort['subject_id'].nunique():,}")
    print(f"  Patients appearing more than once: {n_dup_patients:,}")
    if n_dup_patients:
        print("  FAIL: same patient can potentially cross train/test split")
        return False
    print("  PASS: exactly one cohort stay per patient")
    return True


def check_diagnosis_placeholder():
    print("\n=== CHECK 5: DIAGNOSIS/COMORBIDITY FEATURE ===")
    X = _load("feature_matrix_raw.parquet")
    if "n_diagnoses" not in X:
        print("  PASS: n_diagnoses is not used")
        return True
    vals = X["n_diagnoses"].dropna()
    unique = sorted(vals.unique().tolist())
    if len(unique) == 1 and unique[0] == 0:
        print("  PASS: n_diagnoses is a zero placeholder; full-hospitalization diagnoses are not used")
        return True
    print(f"  FAIL: n_diagnoses contains non-zero values (sample unique values: {unique[:10]})")
    print("  Admission-level diagnoses must not be counted unless availability within the observation window is proven.")
    return False


def check_label_window_timestamps():
    print("\n=== CHECK 6: LABEL EVENT TIMESTAMPS ===")
    path = DATA_DIR / "horizon_labels.parquet"
    if not path.exists():
        print(f"  SKIPPED: {path} not found -- this check needs 04_prediction_horizons.py")
        return None

    labels = _load("horizon_labels.parquet")
    required = {"prediction_start", "prediction_end", "death_time",
                "vasopressor_time", "ventilation_time", "deteriorated_composite", "eligible"}
    missing = required - set(labels.columns)
    if missing:
        print(f"  FAIL: horizon_labels.parquet missing columns {sorted(missing)}")
        return False

    start = pd.to_datetime(labels["prediction_start"], utc=True)
    end = pd.to_datetime(labels["prediction_end"], utc=True)

    all_ok = True
    for col in ["death_time", "vasopressor_time", "ventilation_time"]:
        event = pd.to_datetime(labels[col], utc=True)
        present = event.notna()
        violations = present & ((event <= start) | (event > end))
        n_bad = int(violations.sum())
        print(f"  {col}: {int(present.sum()):,} present, {n_bad} outside the "
              f"prediction window (before prediction_start or after prediction_end)")
        if n_bad:
            cols = ["stay_id", "horizon_hours", col, "prediction_start", "prediction_end"]
            print("  Sample violations:")
            print(labels.loc[violations, cols].head(10).to_string(index=False))
            all_ok = False

    # Cross-check: deteriorated_composite must agree with the raw events that
    # supposedly produced it, for every eligible row.
    any_event = labels[["death_time", "vasopressor_time", "ventilation_time"]].notna().any(axis=1)
    eligible = labels["eligible"].astype(bool)
    mismatch = (any_event != labels["deteriorated_composite"].astype(bool)) & eligible
    n_mismatch = int(mismatch.sum())
    print(f"  deteriorated_composite consistency: {n_mismatch} mismatches among eligible rows")
    if n_mismatch:
        cols = ["stay_id", "horizon_hours", "death_time", "vasopressor_time",
                "ventilation_time", "deteriorated_composite"]
        print("  Sample mismatches:")
        print(labels.loc[mismatch, cols].head(10).to_string(index=False))
        all_ok = False

    if all_ok:
        print(f"  PASS: all label events fall inside their prediction window, "
              f"across {labels['horizon_hours'].nunique()} horizon(s)")
    return all_ok


# ---------------------------------------------------------------------------
# Two independent groups, run at two different points in the chain:
#
#   FEATURE_SIDE_CHECKS -- need only 01 + 02's output. Run these
#   immediately after 02_preprocessing.py, BEFORE 04_prediction_horizons.py
#   even needs to exist. Catching a feature-engineering leak here means you
#   find out before spending BigQuery quota on label extraction or GPU/CPU
#   time on model training -- there's no reason to wait for the rest of the
#   chain to check the part that's already fully computed.
#
#   LABEL_SIDE_CHECKS -- need 04_prediction_horizons.py's horizon_labels.parquet
#   (Check 3 also needs 02's feature matrix, to verify the join). These
#   genuinely cannot run before 04, since there's nothing to audit yet --
#   they report SKIPPED (not FAIL) rather than a false failure when
#   horizon_labels.parquet doesn't exist, so running the audit right after
#   02 doesn't incorrectly report an overall FAIL for a step that simply
#   hasn't happened yet yet.
# ---------------------------------------------------------------------------
FEATURE_SIDE_CHECKS = {
    "Raw feature timestamps": check_raw_feature_timestamps,
    "Feature matrix content": check_feature_matrix_columns,
    "One stay per patient": check_one_stay_per_patient,
    "Diagnosis feature": check_diagnosis_placeholder,
}
LABEL_SIDE_CHECKS = {
    "Label separation": check_labels_separate,
    "Label event timestamps": check_label_window_timestamps,
}


def _run_checks(checks: dict) -> dict:
    results = {}
    for name, fn in checks.items():
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n{name}: FAIL with exception: {exc}")
            results[name] = False
    return results


def _print_summary(title: str, results: dict) -> bool:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for name, ok in results.items():
        status = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {status:7s}  {name}")
    # None (skipped) doesn't count as failure, but doesn't count as a pass either
    ran = {k: v for k, v in results.items() if v is not None}
    all_ok = all(ran.values()) if ran else True
    n_skipped = sum(1 for v in results.values() if v is None)
    print("=" * 72)
    if n_skipped:
        print(f"OVERALL (of checks that could run): {'PASS' if all_ok else 'FAIL'} "
              f"-- {n_skipped} check(s) skipped, pending a later stage")
    else:
        print("OVERALL:", "PASS" if all_ok else "FAIL — fix failed checks before proceeding")
    return all_ok


def check_all_feature_side() -> bool:
    """Run right after 02_preprocessing.py, before 04 needs to exist."""
    results = _run_checks(FEATURE_SIDE_CHECKS)
    return _print_summary("FEATURE-SIDE LEAKAGE AUDIT (run after 01+02, before 04)", results)


def check_all_label_side() -> bool:
    """Run after 04_prediction_horizons.py has produced horizon_labels.parquet."""
    results = _run_checks(LABEL_SIDE_CHECKS)
    return _print_summary("LABEL-SIDE LEAKAGE AUDIT (run after 04)", results)


def check_all() -> bool:
    """Full audit -- both groups. Label-side checks report SKIPPED (not FAIL)
    if horizon_labels.parquet doesn't exist yet, so this is safe to run at
    any point in the chain, including right after 02."""
    feature_results = _run_checks(FEATURE_SIDE_CHECKS)
    label_results = _run_checks(LABEL_SIDE_CHECKS)
    all_results = {**feature_results, **label_results}
    return _print_summary("FULL TEMPORAL/TARGET LEAKAGE AUDIT SUMMARY", all_results)


if __name__ == "__main__":
    import sys
    if "--feature-only" in sys.argv:
        ok = check_all_feature_side()
    elif "--label-only" in sys.argv:
        ok = check_all_label_side()
    else:
        ok = check_all()
    raise SystemExit(0 if ok else 1)
