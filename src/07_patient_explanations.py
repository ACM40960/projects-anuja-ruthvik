from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
MODEL_BUNDLE = DATA_DIR / "model_artifacts.joblib"
CASE_TYPES = ("true_positive", "false_negative")
PREDICTION_THRESHOLD = 0.50
MAX_DISPLAY = 9

FEATURE_LABELS = {
    "sbp_min": "Minimum systolic blood pressure",
    "sbp_mean": "Mean systolic blood pressure",
    "dbp_min": "Minimum diastolic blood pressure",
    "resp_rate_mean": "Mean respiratory rate",
    "resp_rate_last": "Latest respiratory rate",
    "lactate_last": "Latest lactate",
    "lactate_mean": "Mean lactate",
    "lactate_count": "Lactate measurement count",
    "gcs_verbal_last": "Latest GCS verbal score",
    "gcs_eye_last": "Latest GCS eye score",
    "gcs_motor_last": "Latest GCS motor score",
    "spo2_min": "Minimum oxygen saturation",
    "spo2_last": "Latest oxygen saturation",
    "age": "Age",
    "shock_index": "Shock index",
}


def humanize_feature(feature: str) -> str:
    """Convert an engineered feature name into a concise display label."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    label = feature
    replacements = {
        "_mean": " mean",
        "_min": " minimum",
        "_max": " maximum",
        "_last": " latest",
        "_count": " measurement count",
        "_missing": " missingness indicator",
    }
    for suffix, replacement in replacements.items():
        if label.endswith(suffix):
            label = label[: -len(suffix)] + replacement
            break

    label = label.replace("first_careunit_", "Care unit: ")
    label = label.replace("admission_type_", "Admission type: ")
    return label.replace("_", " ").strip().capitalize()


def choose_cases(y_true: pd.Series, probability: np.ndarray) -> dict[str, object]:
    """Select the same representative TP/FN cases used in the earlier analysis."""
    cases = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=int),
            "probability": np.asarray(probability, dtype=float),
        },
        index=y_true.index,
    )
    cases["predicted"] = (cases["probability"] >= PREDICTION_THRESHOLD).astype(int)

    selected = {}
    tp = cases[cases["actual"].eq(1) & cases["predicted"].eq(1)]
    fn = cases[cases["actual"].eq(1) & cases["predicted"].eq(0)]

    if not tp.empty:
        selected["true_positive"] = tp["probability"].idxmax()
    if not fn.empty:
        selected["false_negative"] = fn["probability"].idxmin()
    return selected


def explain_case(model, x_row: pd.DataFrame) -> shap.Explanation:
    """Return a one-row SHAP explanation for the tuned underlying model."""
    if isinstance(model, Pipeline):
        classifier = model.named_steps["clf"]
        scaler = model.named_steps["scale"]
        transformed = pd.DataFrame(
            scaler.transform(x_row),
            columns=x_row.columns,
            index=x_row.index,
        )
        explanation = shap.LinearExplainer(classifier, transformed)(transformed)[0]
        plot_data = transformed.iloc[0].to_numpy()
    else:
        explanation = shap.TreeExplainer(model)(x_row)[0]
        plot_data = x_row.iloc[0].to_numpy()

    if np.asarray(explanation.values).ndim > 1:
        explanation = explanation[..., 1]

    return shap.Explanation(
        values=np.asarray(explanation.values),
        base_values=explanation.base_values,
        data=plot_data,
        feature_names=[humanize_feature(c) for c in x_row.columns],
    )


def main() -> None:
    if not MODEL_BUNDLE.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_BUNDLE}. Run src/06_model_evaluation.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_BUNDLE)

    model = bundle["underlying_model"]
    x_test = bundle["X_test_imputed"]
    y_test = bundle["y_test"]
    probability = bundle["calibrated_probability"]

    selected = choose_cases(y_test, probability)
    if not selected:
        raise RuntimeError("No true-positive or false-negative cases were available.")

    probability_series = pd.Series(
        np.asarray(probability, dtype=float),
        index=y_test.index,
    )

    for case_type in CASE_TYPES:
        if case_type not in selected:
            print(f"Skipping {case_type}: no eligible case.")
            continue

        row_index = selected[case_type]
        explanation = explain_case(model, x_test.loc[[row_index]])

        shap.plots.waterfall(explanation, max_display=MAX_DISPLAY, show=False)
        plt.title(f"Patient-level SHAP: {case_type.replace('_', ' ').title()}")
        plt.tight_layout()

        output = RESULTS_DIR / f"shap_waterfall_{case_type}.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

        risk = float(probability_series.loc[row_index])
        actual = int(y_test.loc[row_index])
        print(f"Saved {output} | calibrated risk={risk:.1%} | actual={actual}")

    print("Patient-level XAI complete.")


if __name__ == "__main__":
    main()
