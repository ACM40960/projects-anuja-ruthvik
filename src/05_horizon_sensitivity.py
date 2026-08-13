import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
)
from xgboost import XGBClassifier

DATA_DIR = "./data"
RESULTS_DIR = "./results"
HORIZONS = [24, 36, 48]
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Best parameters from the primary 24h full-data run.
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit imputation only on training data; preserve missingness indicators."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    count_cols = [c for c in X_train.columns if c.endswith("_count")]
    value_cols = [c for c in X_train.columns if c not in count_cols]

    train_missing = {}
    test_missing = {}

    for col in value_cols:
        train_missing[f"{col}_missing"] = X_train[col].isna().astype(int)
        test_missing[f"{col}_missing"] = X_test[col].isna().astype(int)
        median = X_train[col].median()
        median = 0 if pd.isna(median) else median
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    for col in count_cols:
        X_train[col] = X_train[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    X_train = pd.concat([X_train, pd.DataFrame(train_missing, index=X_train.index)], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(test_missing, index=X_test.index)], axis=1)
    return X_train, X_test


def metrics_from_predictions(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    features = pd.read_parquet(f"{DATA_DIR}/feature_matrix_raw.parquet")
    labels = pd.read_parquet(f"{DATA_DIR}/horizon_labels.parquet")

    if "stay_id" not in features.columns:
        raise ValueError("feature_matrix_raw.parquet must contain stay_id for horizon label merging")

    results = []
    roc_payload = []

    plt.figure(figsize=(7, 6))

    for horizon in HORIZONS:
        print("\n" + "=" * 70)
        print(f"6h observation -> next {horizon}h deterioration")
        print("=" * 70)

        lab = labels[(labels["horizon_hours"] == horizon) & (labels["eligible"] == 1)].copy()
        df = features.merge(
            lab[["stay_id", "subject_id", "deteriorated_composite"]],
            on="stay_id",
            how="inner",
            validate="one_to_one",
        )

        y = df["deteriorated_composite"].astype(int)
        subject_ids = df["subject_id_y"] if "subject_id_y" in df.columns else df.get("subject_id")

        drop_cols = [c for c in [
            "stay_id", "subject_id", "subject_id_x", "subject_id_y", "hadm_id",
            "deteriorated_composite", "died_in_icu"
        ] if c in df.columns]
        X = df.drop(columns=drop_cols).select_dtypes(include=[np.number, bool]).astype(float)

        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        X_train = X.iloc[idx_train].copy()
        X_test = X.iloc[idx_test].copy()
        y_train = y.iloc[idx_train].copy()
        y_test = y.iloc[idx_test].copy()

        if subject_ids is not None:
            train_subjects = set(subject_ids.iloc[idx_train])
            test_subjects = set(subject_ids.iloc[idx_test])
            overlap = train_subjects & test_subjects
            if overlap:
                raise RuntimeError(f"Patient leakage detected for {horizon}h horizon: {len(overlap)} overlap")

        X_train, X_test = impute_train_test(X_train, X_test)

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(scale_pos_weight=pos_weight, **XGB_PARAMS)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        m = metrics_from_predictions(y_test, proba)

        row = {
            "observation_hours": 6,
            "horizon_hours": horizon,
            "cohort_n": len(df),
            "positive_rate": y.mean(),
            "train_n": len(X_train),
            "test_n": len(X_test),
            "scale_pos_weight": pos_weight,
            **m,
        }
        results.append(row)

        print(f"Cohort: {len(df):,} | positive rate: {y.mean():.2%}")
        print("TEST -> " + " | ".join(f"{k}={v:.3f}" for k, v in m.items()))

        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"6h -> {horizon}h (AUC={m['roc_auc']:.3f})")
        roc_payload.append((horizon, fpr, tpr))

    results_df = pd.DataFrame(results).sort_values("horizon_hours")
    results_df.to_csv(f"{RESULTS_DIR}/horizon_sensitivity_xgboost.csv", index=False)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost Sensitivity to Prediction Horizon\nFixed 6-hour Observation Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/horizon_sensitivity_roc.png", dpi=180)
    plt.close()

    print("\n" + "=" * 70)
    print("HORIZON SENSITIVITY SUMMARY")
    print("=" * 70)
    print(results_df[[
        "horizon_hours", "cohort_n", "positive_rate",
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ]].round(3).to_string(index=False))
    print(f"\nSaved {DATA_DIR}/horizon_sensitivity_xgboost.csv")
    print(f"Saved {DATA_DIR}/horizon_sensitivity_roc.png")


if __name__ == "__main__":
    main()
