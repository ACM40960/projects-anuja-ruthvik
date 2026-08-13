import json
import os
import pandas as pd
from google.cloud import bigquery

# Configuration parameters
PROJECT_ID = "euphoric-quanta-505215-n6"    # Replace with your GCP project ID
OBSERVATION_WINDOW_HOURS = 6                # Observation window for feature extraction (hours)
PREDICTION_WINDOW_HOURS = 24                # Prediction window for outcome labels (hours)
MIN_LOS_HOURS = 12                          # Minimum ICU length of stay threshold
OUTPUT_DIR = "./data"                       # Output directory for extracted data

client = bigquery.Client(project=PROJECT_ID)


def run_query(sql: str) -> pd.DataFrame:
    """Run a BigQuery SQL query and return a pandas DataFrame."""
    return client.query(sql).to_dataframe()


def print_timestamp_sanity(name: str, readings: pd.DataFrame, cohort: pd.DataFrame, key: str) -> None:
    """Print actual min/max measurement offsets relative to ICU admission."""
    if readings.empty:
        print(f"  {name}: no readings returned")
        return
    cols = [key, "intime"]
    merged = readings.merge(cohort[cols].drop_duplicates(), on=key, how="left", validate="many_to_one")
    merged["charttime"] = pd.to_datetime(merged["charttime"], utc=True)
    merged["intime"] = pd.to_datetime(merged["intime"], utc=True)
    offsets = (merged["charttime"] - merged["intime"]).dt.total_seconds() / 3600.0
    print(f"  {name} timestamp offsets: min={offsets.min():.3f}h, max={offsets.max():.3f}h")
    violations = ((offsets < 0) | (offsets > OBSERVATION_WINDOW_HOURS)).sum()
    if violations:
        raise RuntimeError(f"TEMPORAL LEAKAGE: {violations} {name.lower()} readings fall outside hours 0-{OBSERVATION_WINDOW_HOURS}")


# Extract cohort: first ICU stay per patient with minimum length of stay
COHORT_SQL = f"""
WITH ranked_stays AS (
    SELECT
        subject_id, hadm_id, stay_id, first_careunit,
        intime, outtime,
        TIMESTAMP_DIFF(outtime, intime, HOUR) AS los_hours,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS stay_rank
    FROM `physionet-data.mimiciv_3_1_icu.icustays`
)
SELECT subject_id, hadm_id, stay_id, first_careunit, intime, outtime, los_hours
FROM ranked_stays
WHERE stay_rank = 1
  AND los_hours >= {MIN_LOS_HOURS}
"""


# Build label query: deterioration events within prediction window
def build_label_query(cohort_stay_ids_sql: str) -> str:
    return f"""
    WITH cohort AS ({cohort_stay_ids_sql}),
    mortality AS (
        SELECT c.stay_id,
               CASE WHEN a.deathtime IS NOT NULL
                         AND a.deathtime > TIMESTAMP_ADD(
                             c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR
                         )
                         AND a.deathtime <= TIMESTAMP_ADD(
                             c.intime,
                             INTERVAL {OBSERVATION_WINDOW_HOURS + PREDICTION_WINDOW_HOURS} HOUR
                         )
                    THEN 1 ELSE 0 END AS died_in_icu
        FROM cohort c
        JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
          ON c.hadm_id = a.hadm_id
    ),
    vasopressors AS (
        -- norepinephrine, epinephrine, dopamine, dobutamine, vasopressin, phenylephrine
        SELECT DISTINCT stay_id
        FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
        JOIN cohort c USING (stay_id)
        JOIN `physionet-data.mimiciv_3_1_icu.d_items` d ON ie.itemid = d.itemid
        WHERE LOWER(d.label) LIKE ANY (
              '%norepinephrine%', '%epinephrine%', '%dopamine%',
              '%dobutamine%', '%vasopressin%', '%phenylephrine%')
          AND ie.starttime > TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR)
          AND ie.starttime <= TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS + PREDICTION_WINDOW_HOURS} HOUR)
    ),
    ventilation AS (
        SELECT DISTINCT stay_id
        FROM `physionet-data.mimiciv_3_1_icu.procedureevents` pe
        JOIN cohort c USING (stay_id)
        JOIN `physionet-data.mimiciv_3_1_icu.d_items` d ON pe.itemid = d.itemid
        WHERE LOWER(d.label) LIKE '%invasive vent%'
          AND pe.starttime > TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR)
          AND pe.starttime <= TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS + PREDICTION_WINDOW_HOURS} HOUR)
    )
    SELECT
        c.stay_id,
        m.died_in_icu,
        CASE WHEN v.stay_id IS NOT NULL OR vent.stay_id IS NOT NULL OR m.died_in_icu = 1
             THEN 1 ELSE 0 END AS deteriorated_composite
    FROM cohort c
    LEFT JOIN mortality m USING (stay_id)
    LEFT JOIN vasopressors v USING (stay_id)
    LEFT JOIN ventilation vent USING (stay_id)
    """


# Extract vital signs from chartevents within observation window to prevent temporal leakage
VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "sbp",          # systolic BP (non-invasive)
    220180: "dbp",          # diastolic BP (non-invasive)
    220210: "resp_rate",
    220277: "spo2",
    223761: "temperature_f",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

def build_vitals_query(cohort_stay_ids_sql: str) -> str:
    itemid_list = ", ".join(str(i) for i in VITAL_ITEMIDS)
    return f"""
    WITH cohort AS ({cohort_stay_ids_sql})
    SELECT
        ce.stay_id, ce.itemid, ce.charttime, ce.valuenum
    FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
    JOIN cohort c USING (stay_id)
    WHERE ce.itemid IN ({itemid_list})
      AND ce.valuenum IS NOT NULL
      AND ce.charttime BETWEEN c.intime
          AND TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR)
    """


# Extract laboratory values from labevents within observation window
LAB_ITEMIDS = {
    50813: "lactate",
    51301: "wbc",
    50912: "creatinine",
    51265: "platelets",
    50882: "bicarbonate",
    50971: "potassium",
    50983: "sodium",
}

def build_labs_query(cohort_stay_ids_sql: str) -> str:
    itemid_list = ", ".join(str(i) for i in LAB_ITEMIDS)
    return f"""
    WITH cohort AS ({cohort_stay_ids_sql})
    SELECT
        le.hadm_id, le.itemid, le.charttime, le.valuenum
    FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
    JOIN cohort c USING (hadm_id)
    WHERE le.itemid IN ({itemid_list})
      AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN c.intime
          AND TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR)
    """


# Extract static demographic features
def build_demographics_query(cohort_stay_ids_sql: str) -> str:
    return f"""
    WITH cohort AS ({cohort_stay_ids_sql})
    SELECT
        c.stay_id, c.subject_id, c.first_careunit,
        p.gender, p.anchor_age AS age,
        a.admission_type
    FROM cohort c
    JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON c.subject_id = p.subject_id
    JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a ON c.hadm_id = a.hadm_id
    """


def main():
    print("Extracting cohort...")
    cohort = run_query(COHORT_SQL)
    print(f"  {len(cohort)} ICU stays in cohort")

    # Use cohort SQL as server-side subquery for consistency and efficiency
    cohort_sql = COHORT_SQL

    print("Extracting labels...")
    labels = run_query(build_label_query(cohort_sql))
    print(f"  Mortality rate: {labels['died_in_icu'].mean():.2%}")
    print(f"  Composite deterioration rate: {labels['deteriorated_composite'].mean():.2%}")

    print("Extracting vitals (chartevents)...")
    vitals = run_query(build_vitals_query(cohort_sql))
    vitals["feature_name"] = vitals["itemid"].map(VITAL_ITEMIDS)
    print(f"  {len(vitals)} vital sign readings")
    print_timestamp_sanity("Vitals", vitals, cohort, "stay_id")

    print("Extracting labs (labevents)...")
    labs = run_query(build_labs_query(cohort_sql))
    labs["feature_name"] = labs["itemid"].map(LAB_ITEMIDS)
    print(f"  {len(labs)} lab readings")
    print_timestamp_sanity("Labs", labs, cohort, "hadm_id")

    print("Extracting demographics...")
    demo = run_query(build_demographics_query(cohort_sql))

    # Save raw extracts for preprocessing pipeline
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cohort.to_parquet(f"{OUTPUT_DIR}/cohort.parquet", index=False)
    labels.to_parquet(f"{OUTPUT_DIR}/labels.parquet", index=False)
    vitals.to_parquet(f"{OUTPUT_DIR}/vitals_raw.parquet", index=False)
    labs.to_parquet(f"{OUTPUT_DIR}/labs_raw.parquet", index=False)
    demo.to_parquet(f"{OUTPUT_DIR}/demographics.parquet", index=False)

    # Store temporal design configuration for downstream validation
    window_metadata = {
        "observation_window_hours": OBSERVATION_WINDOW_HOURS,
        "prediction_window_hours": PREDICTION_WINDOW_HOURS,
        "prediction_start_hour": OBSERVATION_WINDOW_HOURS,
        "prediction_end_hour": OBSERVATION_WINDOW_HOURS + PREDICTION_WINDOW_HOURS,
    }
    with open(f"{OUTPUT_DIR}/window_config.json", "w", encoding="utf-8") as f:
        json.dump(window_metadata, f, indent=2)

    print(f"\nTemporal design: features=0-{OBSERVATION_WINDOW_HOURS}h | "
          f"outcomes=>{OBSERVATION_WINDOW_HOURS}-{OBSERVATION_WINDOW_HOURS + PREDICTION_WINDOW_HOURS}h")
    print(f"Saved temporal configuration to {OUTPUT_DIR}/window_config.json")
    print(f"\nDone. Raw extracts saved to {OUTPUT_DIR}/")
    print("Next: Step 3 — preprocessing (resampling, missingness handling, outlier removal)")


if __name__ == "__main__":
    main()
