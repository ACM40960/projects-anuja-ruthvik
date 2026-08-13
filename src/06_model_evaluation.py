import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve
)
from xgboost import XGBClassifier
import shap
import joblib

DATA_DIR = "./data"
RESULTS_DIR = "./results"
TARGET = "deteriorated_composite"   # or "died_in_icu"
TEST_SIZE = 0.2
N_TUNING_FOLDS = 5
CALIBRATION_CV_FOLDS = 3     # folds used inside CalibratedClassifierCV
CALIBRATION_METHOD = "isotonic"  # isotonic needs decent data volume (fine at 50k+ train rows); use "sigmoid" for smaller data
SHAP_SAMPLE_SIZE = 2000
SHAP_TOP_N = 15
SHAP_DEPENDENCE_FEATURES = [
    "sbp_min",
    "lactate_last",
]
RANDOM_STATE = 42


def load_raw_features():
    df = pd.read_parquet(f"{DATA_DIR}/feature_matrix_raw.parquet")
    labels = pd.read_parquet(f"{DATA_DIR}/labels_matrix.parquet")
    
    # Merge to get stay_id, subject_id for later patient-leakage checks
    df = df.merge(labels[["stay_id", "died_in_icu", "deteriorated_composite"]], on="stay_id", how="left")
    
    id_cols = ["stay_id", "subject_id", "hadm_id"]
    label_cols = ["died_in_icu", "deteriorated_composite"]
    drop_cols = [c for c in id_cols + label_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number, bool]).astype(float)
    y = df[TARGET].astype(int)
    return X, y, df


def impute_split(X_train, X_test, count_cols):
    """Add *_missing indicators and median-impute, fitting medians on the
    TRAINING split only (this is the leakage fix vs. the demo version)."""
    X_train, X_test = X_train.copy(), X_test.copy()
    value_cols = [c for c in X_train.columns if c not in count_cols]
    train_missing_cols, test_missing_cols = {}, {}

    for col in value_cols:
        train_missing_cols[f"{col}_missing"] = X_train[col].isna().astype(int)
        test_missing_cols[f"{col}_missing"] = X_test[col].isna().astype(int)
        median = X_train[col].median()
        median = 0 if pd.isna(median) else median
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    for col in count_cols:
        X_train[col] = X_train[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    X_train = pd.concat([X_train, pd.DataFrame(train_missing_cols, index=X_train.index)], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(test_missing_cols, index=X_test.index)], axis=1)

    return X_train, X_test


def build_search_spaces(pos_weight):
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))]),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [1, 5, 10, 20],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                           random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200, 400],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
            },
        ),
    }


def tune_and_fit(name, model, param_dist, X_train, y_train, cv):
    print(f"Tuning {name}...")
    search = RandomizedSearchCV(
        model, param_dist, n_iter=15, scoring="roc_auc",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  best CV ROC-AUC: {search.best_score_:.3f} | params: {search.best_params_}")
    return search.best_estimator_


def evaluate_on_test(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }, proba




def classify_feature_type(feature_name: str) -> str:
    """Classify SHAP features for interpretation/reporting."""
    if feature_name.endswith("_count"):
        return "measurement_frequency"
    if feature_name.endswith("_missing"):
        return "missingness_indicator"
    return "clinical_or_static_value"


def choose_dependence_features(columns):
    """Pick clinically interpretable dependence features with sensible fallbacks."""
    columns = list(columns)
    fallback_groups = {
        "sbp_min": ["sbp_min", "sbp_last", "sbp_mean", "dbp_min"],
        "resp_rate_mean": ["resp_rate_mean", "resp_rate_last", "resp_rate_max"],
        "lactate_last": ["lactate_last", "lactate_mean", "lactate_max", "lactate_min"],
        "gcs_verbal_last": ["gcs_verbal_last", "gcs_verbal_mean", "gcs_eye_last", "gcs_motor_last"],
    }
    chosen = []
    for preferred in SHAP_DEPENDENCE_FEATURES:
        for candidate in fallback_groups.get(preferred, [preferred]):
            if candidate in columns and candidate not in chosen:
                chosen.append(candidate)
                break
    return chosen


def save_global_shap_outputs(shap_values, X_for_plot, data_dir):
    """Save global SHAP ranking, bar plot, and dependence plots."""
    values = np.asarray(shap_values.values)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP matrix, got shape {values.shape}")

    importance = np.abs(values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_for_plot.columns,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    importance_df["feature_type"] = importance_df["feature"].map(classify_feature_type)
    
    top = importance_df.head(SHAP_TOP_N).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title(f"Top {SHAP_TOP_N} Global SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{data_dir}/shap_feature_importance_full.png", dpi=300, bbox_inches="tight")
    plt.close()

    dependence_features = choose_dependence_features(X_for_plot.columns)
    dep_rows = []
    for feature in dependence_features:
        feature_idx = X_for_plot.columns.get_loc(feature)
        feature_type = classify_feature_type(feature)
        dep_rows.append({
            "feature": feature,
            "feature_type": feature_type,
            "importance_rank": int(importance_df.loc[importance_df["feature"] == feature, "rank"].iloc[0]),
            "mean_abs_shap": float(importance_df.loc[importance_df["feature"] == feature, "mean_abs_shap"].iloc[0]),
        })
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            feature_idx,
            values,
            X_for_plot,
            feature_names=X_for_plot.columns,
            interaction_index=None,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(f"{data_dir}/shap_dependence_{feature}.png", dpi=300, bbox_inches="tight")
        plt.close()


    print(f"Saved {data_dir}/shap_feature_importance_full.png")
    if dependence_features:
        print("Saved SHAP dependence plots for: " + ", ".join(dependence_features))
    else:
        print("WARNING: No requested clinical dependence features were available.")

    # Explicitly surface monitoring-intensity / missingness features for interpretation.
    top_monitoring = importance_df[
        importance_df["feature_type"].isin(["measurement_frequency", "missingness_indicator"])
    ].head(10)
    if not top_monitoring.empty:
        print("\nTop monitoring/missingness-related SHAP features (interpret cautiously):")
        for _, row in top_monitoring.iterrows():
            print(f"  {int(row['rank']):2d}. {row['feature']}: mean|SHAP|={row['mean_abs_shap']:.4f}")


def main():
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, y, df = load_raw_features()
    count_cols = [c for c in X.columns if c.endswith("_count")]
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} raw features")
    print(f"Target '{TARGET}' overall positive rate: {y.mean():.2%}\n")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows ({y_train.mean():.2%} positive) | "
          f"Test: {len(X_test)} rows ({y_test.mean():.2%} positive)\n")

    # Verify no patient overlap between train and test splits
    if "subject_id" in df.columns:
        train_subjects = set(df.loc[idx_train, "subject_id"])
        test_subjects = set(df.loc[idx_test, "subject_id"])
        overlap = train_subjects & test_subjects
        assert not overlap, (
            f"PATIENT LEAKAGE: {len(overlap)} subject_id(s) appear in both "
            f"train and test: {sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )
        print(f"Patient-leakage check: OK -- {len(train_subjects)} train patients, "
              f"{len(test_subjects)} test patients, 0 overlap\n")
    else:
        print("Patient-leakage check: SKIPPED -- 'subject_id' not present in "
              "feature_matrix_raw.parquet (check step3 output)\n")

    X_test_raw = X_test.copy()

    print("Imputing (fit on train only)...")
    X_train, X_test = impute_split(X_train, X_test, count_cols)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"XGBoost scale_pos_weight (from train split): {pos_weight:.2f}\n")

    cv = StratifiedKFold(n_splits=N_TUNING_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search_spaces = build_search_spaces(pos_weight)

    fitted_models, results, probas = {}, {}, {}
    for name, (model, param_dist) in search_spaces.items():
        best_model = tune_and_fit(name, model, param_dist, X_train, y_train, cv)
        fitted_models[name] = best_model
        metrics, proba = evaluate_on_test(best_model, X_test, y_test)
        results[name] = metrics
        probas[name] = proba
        print(f"  TEST SET  -> " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        print()

    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{RESULTS_DIR}/model_comparison_full.csv")
    print("Saved results/model_comparison_full.csv")
    print(results_df.round(3))

    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (held-out test set)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/roc_curves_full.png")

    # Apply isotonic calibration to align predicted probabilities with observed frequencies
    print("Calibrating models (CalibratedClassifierCV, isotonic, "
          f"{CALIBRATION_CV_FOLDS}-fold internal CV on training data)...")

    calibrated_models, calibrated_results, calibrated_probas = {}, {}, {}
    for name, model in fitted_models.items():
        calibrated = CalibratedClassifierCV(
            estimator=model, method=CALIBRATION_METHOD, cv=CALIBRATION_CV_FOLDS
        )
        calibrated.fit(X_train, y_train)
        calibrated_models[name] = calibrated
        metrics, proba = evaluate_on_test(calibrated, X_test, y_test)
        calibrated_results[name] = metrics
        calibrated_probas[name] = proba
        print(f"  {name:20s} CALIBRATED -> " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    calibrated_results_df = pd.DataFrame(calibrated_results).T
    print("\nCalibrated model metrics:")
    print(calibrated_results_df.round(3))

    # Before/after calibration curves, side by side, for direct comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for name, proba in probas.items():
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        axes[0].plot(mean_pred, frac_pos, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("BEFORE calibration")
    axes[0].set_xlabel("Mean predicted probability"); axes[0].set_ylabel("Observed frequency")
    axes[0].legend()

    for name, proba in calibrated_probas.items():
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        axes[1].plot(mean_pred, frac_pos, marker="o", label=name)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("AFTER calibration (isotonic)")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/calibration_curves_full_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/calibration_curves_full_before_after.png")

    # Compute SHAP values on best-performing model using raw test set (uncalibrated for direct feature attribution)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"\nBest model by test ROC-AUC: {best_name}")

    sample_n = min(SHAP_SAMPLE_SIZE, len(X_test))
    X_shap = X_test.sample(n=sample_n, random_state=RANDOM_STATE)
    print(f"Running SHAP on a sample of {sample_n} test rows...")

    if isinstance(best_model, Pipeline):
        clf = best_model.named_steps["clf"]
        X_scaled = pd.DataFrame(best_model.named_steps["scale"].transform(X_shap), columns=X_shap.columns)
        explainer = shap.LinearExplainer(clf, X_scaled)
        shap_values = explainer(X_scaled)
        X_for_plot = X_scaled
    else:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer(X_shap)
        X_for_plot = X_shap

    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values, X_for_plot, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/shap_summary_full.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved results/shap_summary_full.png")

    # Generate global SHAP analysis (feature importance, dependence plots, measurement-frequency metrics)
    save_global_shap_outputs(shap_values, X_for_plot, RESULTS_DIR)

    # Package model artifacts for downstream use (steps 07-08)
    artifact_bundle = {
        "best_model_name": best_name,
        "underlying_model": best_model,
        "calibrated_model": calibrated_models[best_name],
        "X_test_imputed": X_test,
        "X_test_raw": X_test_raw,
        "y_test": y_test,
        "calibrated_probability": calibrated_probas[best_name],
    }
    joblib.dump(artifact_bundle, f"{DATA_DIR}/model_artifacts.joblib")
    print("Saved local model bundle to data/model_artifacts.joblib")

    print("\nDone. Primary model evaluation and global XAI outputs saved.")


if __name__ == "__main__":
    main()
