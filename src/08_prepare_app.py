from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("./data")
MODEL_BUNDLE = DATA_DIR / "model_artifacts.joblib"


def main() -> None:
    if not MODEL_BUNDLE.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_BUNDLE}. Run src/06_model_evaluation.py first."
        )

    bundle = joblib.load(MODEL_BUNDLE)
    model = bundle["underlying_model"]
    x_test = bundle["X_test_imputed"].reset_index(drop=True).copy()
    x_raw = bundle["X_test_raw"].reset_index(drop=True).copy()
    probability = np.asarray(bundle["calibrated_probability"], dtype=float)

    display_ids = [f"ICU-{i:05d}" for i in range(1, len(x_test) + 1)]
    x_test.insert(0, "display_id", display_ids)
    x_raw.insert(0, "display_id", display_ids)

    monitor = pd.DataFrame(
        {
            "display_id": display_ids,
            "calibrated_probability": probability,
        }
    ).sort_values("calibrated_probability", ascending=False, ignore_index=True)

    x_test.to_parquet(DATA_DIR / "app_test_features_imputed.parquet", index=False)
    x_raw.to_parquet(DATA_DIR / "app_test_features_raw.parquet", index=False)
    monitor.to_csv(DATA_DIR / "app_patient_monitor.csv", index=False)
    joblib.dump(model, DATA_DIR / "app_underlying_model.joblib")

    print(f"Prepared {len(monitor):,} local de-identified app records.")
    print("Local app artifacts saved under data/. Do not publish this directory.")


if __name__ == "__main__":
    main()
