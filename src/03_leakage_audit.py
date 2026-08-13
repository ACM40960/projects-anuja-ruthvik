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
        "dischtime", "hospital_expire_flag"
    }
    forbidden_keywords = (
        "death", "mortality", "deteriorat", "vasopressor", "ventilat",
        "discharge", "outtime", "expire"
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
    y = _load("labels_matrix.parquet")
    needed = {"stay_id", "died_in_icu", "deteriorated_composite"}
    missing = needed - set(y.columns)
    if missing:
        print(f"  FAIL: labels_matrix.parquet missing {sorted(missing)}")
        return False
    leaked = [c for c in ["died_in_icu", "deteriorated_composite"] if c in X.columns]
    if leaked:
        print(f"  FAIL: label columns present in feature matrix: {leaked}")
        return False
    print(f"  died_in_icu prevalence: {y['died_in_icu'].mean():.2%}")
    print(f"  deteriorated_composite prevalence: {y['deteriorated_composite'].mean():.2%}")
    print("  PASS: labels are physically separated from saved features")
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


def check_all():
    checks = {
        "Raw feature timestamps": check_raw_feature_timestamps,
        "Feature matrix content": check_feature_matrix_columns,
        "Label separation": check_labels_separate,
        "One stay per patient": check_one_stay_per_patient,
        "Diagnosis feature": check_diagnosis_placeholder,
    }
    results = {}
    for name, fn in checks.items():
        try:
            results[name] = bool(fn())
        except Exception as exc:
            print(f"\n{name}: FAIL with exception: {exc}")
            results[name] = False

    print("\n" + "=" * 72)
    print("STRICT TEMPORAL/TARGET LEAKAGE AUDIT SUMMARY")
    print("=" * 72)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 72)
    print("OVERALL:", "PASS" if all_ok else "FAIL — fix failed checks before final modelling")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if check_all() else 1)
