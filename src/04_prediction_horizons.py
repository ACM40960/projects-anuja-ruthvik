import os
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "euphoric-quanta-505215-n6"
OBSERVATION_WINDOW_HOURS = 6
PREDICTION_HORIZONS = [24, 36, 48]
MIN_LOS_HOURS = 12
OUTPUT_DIR = "./data"

client = bigquery.Client(project=PROJECT_ID)


def run_query(sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()


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


def build_horizon_query(horizon_hours: int) -> str:
    end_hour = OBSERVATION_WINDOW_HOURS + horizon_hours
    return f"""
    WITH cohort AS ({COHORT_SQL}),

    mortality AS (
        SELECT
            c.stay_id,
            MIN(a.deathtime) AS death_time
        FROM cohort c
        JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
          ON c.hadm_id = a.hadm_id
        WHERE a.deathtime IS NOT NULL
          AND a.deathtime > TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR)
          AND a.deathtime <= TIMESTAMP_ADD(c.intime, INTERVAL {end_hour} HOUR)
          AND a.deathtime <= c.outtime
        GROUP BY c.stay_id
    ),

    vasopressors AS (
        SELECT
            c.stay_id,
            MIN(ie.starttime) AS vasopressor_time
        FROM cohort c
        JOIN `physionet-data.mimiciv_3_1_icu.inputevents` ie
          ON c.stay_id = ie.stay_id
        JOIN `physionet-data.mimiciv_3_1_icu.d_items` d
          ON ie.itemid = d.itemid
        WHERE REGEXP_CONTAINS(
            LOWER(d.label),
            r'norepinephrine|epinephrine|dopamine|dobutamine|vasopressin|phenylephrine'
        )
          AND ie.starttime > TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR)
          AND ie.starttime <= TIMESTAMP_ADD(c.intime, INTERVAL {end_hour} HOUR)
          AND ie.starttime <= c.outtime
        GROUP BY c.stay_id
    ),

    ventilation AS (
        SELECT
            c.stay_id,
            MIN(pe.starttime) AS ventilation_time
        FROM cohort c
        JOIN `physionet-data.mimiciv_3_1_icu.procedureevents` pe
          ON c.stay_id = pe.stay_id
        JOIN `physionet-data.mimiciv_3_1_icu.d_items` d
          ON pe.itemid = d.itemid
        WHERE REGEXP_CONTAINS(LOWER(d.label), r'invasive.*vent|ventilation.*invasive')
          AND pe.starttime > TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR)
          AND pe.starttime <= TIMESTAMP_ADD(c.intime, INTERVAL {end_hour} HOUR)
          AND pe.starttime <= c.outtime
        GROUP BY c.stay_id
    )

    SELECT
        c.subject_id,
        c.stay_id,
        c.intime,
        c.outtime,
        {horizon_hours} AS horizon_hours,
        TIMESTAMP_ADD(c.intime, INTERVAL {OBSERVATION_WINDOW_HOURS} HOUR) AS prediction_start,
        TIMESTAMP_ADD(c.intime, INTERVAL {end_hour} HOUR) AS prediction_end,
        m.death_time,
        v.vasopressor_time,
        vent.ventilation_time,
        CAST(m.death_time IS NOT NULL AS INT64) AS death_event,
        CAST(v.vasopressor_time IS NOT NULL AS INT64) AS vasopressor_event,
        CAST(vent.ventilation_time IS NOT NULL AS INT64) AS ventilation_event,
        CAST(
            m.death_time IS NOT NULL
            OR v.vasopressor_time IS NOT NULL
            OR vent.ventilation_time IS NOT NULL
            AS INT64
        ) AS deteriorated_composite,
        CAST(
            (m.death_time IS NOT NULL
             OR v.vasopressor_time IS NOT NULL
             OR vent.ventilation_time IS NOT NULL)
            OR c.outtime >= TIMESTAMP_ADD(c.intime, INTERVAL {end_hour} HOUR)
            AS INT64
        ) AS eligible
    FROM cohort c
    LEFT JOIN mortality m USING (stay_id)
    LEFT JOIN vasopressors v USING (stay_id)
    LEFT JOIN ventilation vent USING (stay_id)
    """


def validate_window(df: pd.DataFrame, horizon: int) -> None:
    """Programmatically verify all observed event times lie inside the horizon."""
    start = pd.to_datetime(df["prediction_start"], utc=True)
    end = pd.to_datetime(df["prediction_end"], utc=True)

    for col in ["death_time", "vasopressor_time", "ventilation_time"]:
        event = pd.to_datetime(df[col], utc=True)
        present = event.notna()
        violations = present & ((event <= start) | (event > end))
        n_bad = int(violations.sum())
        if n_bad:
            raise RuntimeError(
                f"Temporal label error for {horizon}h horizon: {n_bad} {col} events outside prediction window"
            )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []

    print(f"Fixed observation window: first {OBSERVATION_WINDOW_HOURS} hours")
    print(f"Prediction horizons: {PREDICTION_HORIZONS}\n")

    for horizon in PREDICTION_HORIZONS:
        print(f"Extracting {horizon}h horizon labels...")
        df = run_query(build_horizon_query(horizon))
        validate_window(df, horizon)

        eligible = df[df["eligible"] == 1]
        excluded = len(df) - len(eligible)
        positives = int(eligible["deteriorated_composite"].sum())
        rate = eligible["deteriorated_composite"].mean() if len(eligible) else float("nan")

        print(f"  Base cohort: {len(df):,}")
        print(f"  Eligible after censoring rule: {len(eligible):,}")
        print(f"  Excluded for insufficient follow-up without event: {excluded:,}")
        print(f"  Deterioration events: {positives:,} ({rate:.2%})\n")

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(f"{OUTPUT_DIR}/horizon_labels.parquet", index=False)


    print("Saved:")
    print(f"  {OUTPUT_DIR}/horizon_labels.parquet")
    print("\nStage 3 label extraction complete.")


if __name__ == "__main__":
    main()
