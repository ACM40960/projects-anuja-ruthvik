# ICU Deterioration Prediction: Key Findings Guide

A comprehensive guide to the key findings of the project, and the results of the ICU patient deterioration prediction system.

---

## Table of Contents

1. [Model Performance](#model-performance)
2. [Cohort Statistics](#cohort-statistics)
3. [Model Evaluation & Calibration](#model-evalution)
4. [Global Model Explanations](#global-model-explanations)
5. [Feature-Specific Dependence Analysis](#feature-analysis)
6. [Clinical Implications](#clinical-implications)
7. [Methodological Strengths](#methodological-strengths)

---

## Model Performance (24-hour Horizon)
- **Accuracy**: 75.1%
- **Sensitivity (Recall)**: 73.6% of deterioration cases identified
- **Specificity**: 75.4% of non-deterioration cases correct
- **ROC-AUC**: 0.825 (excellent discrimination)
- **Calibration**: Applied via isotonic regression

## Cohort Statistics
- **Total ICU Stays**: 46,982
- **Deterioration Cases**: 16,155 (34.4%)
- **Features**: ~210 per stay
- **Train/Test Split**: 80/20

For detailed results, model explanations, and sensitivity analysis, see the `results/` folder.

---

## Model Evaluation & Calibration

The system trains three models (Logistic Regression, Random Forest, XGBoost) and selects XGBoost as the primary model based on performance.

**Calibration**: Isotonic regression is applied to ensure probability predictions represent true risk. This is critical for clinical use.

```python
# Model training pseudocode
model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05)
model.fit(X_train, y_train)

# Calibration
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_train_proba, y_train)
y_calibrated = calibrator.predict(y_test_proba)
```

### Performance Evaluation Metrics

| Metric | Definition | Interpretation |
|:---|:---|:---|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Sensitivity** | TP / (TP + FN) | % of deterioration cases caught |
| **Specificity** | TN / (TN + FP) | % of non-deterioration cases correct |
| **ROC-AUC** | Area under ROC curve | Discrimination ability (0.5=random, 1.0=perfect) |
| **Precision** | TP / (TP + FP) | % of alerts that are valid |
| **F1-Score** | Harmonic mean of precision & recall | Balance between TP and false alarms |

---

## Global Model Explanations

Feature importance is determined using SHAP (SHapley Additive exPlanations), which calculates each feature's contribution to predictions across all patients.

**Top contributing features**:
- Lactate level (metabolic distress marker)
- GCS verbal score (neurologic status)
- Systolic blood pressure (hemodynamic status)
- Respiratory rate (breathing effort)
- Laboratory abnormalities

See `results/shap_feature_importance_full.png` and `results/shap_summary_full.png` for visualizations.

---

## Feature-Specific Dependence Analysis

Dependence plots show how individual feature values affect predictions:

- **High lactate** → Increased deterioration risk
- **Low blood pressure** → Increased deterioration risk
- **Low GCS (verbal)** → Increased deterioration risk
- **High respiratory rate** → Increased deterioration risk
- **Low platelet count** → Increased deterioration risk

See `results/shap_dependence_*.png` files for detailed dependence plots.

---

## Clinical Implications

1. **Early Prediction**: Deterioration signals are apparent in first 6 hours
2. **High Sensitivity**: Model catches 73.6% of deterioration cases
3. **Calibrated Risk**: Probability predictions represent true risk (post-calibration)
4. **Interpretable**: SHAP explanations enable clinical understanding
5. **Actionable**: Feature importance guides clinical assessment

## Methodological Strengths

1. **Rigorous Temporal Design**: Strict 6-hour feature window prevents leakage
2. **Multiple Events**: Captures different deterioration types
3. **Comprehensive Audit**: Leakage detection at feature and label levels
4. **Proper Calibration**: Ensures probabilities match reality
5. **Explainability**: SHAP provides transparent predictions

---

**Last Updated:** August 2026
**MIMIC-IV Version:** 3.1  
**Python Version:** 3.8+  
**Framework:** XGBoost 1.7+, SHAP 0.41+, scikit-learn 1.0+
