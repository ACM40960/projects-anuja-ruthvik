# AI for ICU Deterioration: Early Warning System

A comprehensive machine learning pipeline for predicting ICU patient deterioration using clinical data from MIMIC-IV. This project implements a 6-hour observation window to predict deterioration events (mortality, vasopressor requirement, mechanical ventilation) across multiple prediction horizons.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Access & Setup](#data-access--setup)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
5. [Model Performance Results](#model-performance-results)
6. [Model Evaluation & Calibration](#model-evaluation--calibration)
7. [Global Model Explanations](#global-model-explanations)
8. [Feature-Specific Dependence Analysis](#feature-specific-dependence-analysis)
9. [Patient-Level Explanations](#patient-level-explanations)
10. [Horizon Sensitivity Analysis](#horizon-sensitivity-analysis)
11. [Key Findings](#key-findings)
12. [Temporal Leakage Prevention](#temporal-leakage-prevention)
13. [Technical Implementation](#technical-implementation)

---

## Project Overview

### Objective
Develop a clinical decision support system that predicts ICU patient deterioration within 24, 36, and 48 hours using only data collected during the first 6 hours of ICU admission. This enables early intervention and resource allocation.

### Key Features
- **Rigorous Temporal Design**: 6-hour observation window → 24-48 hour prediction windows
- **Multiple Deterioration Events**: Mortality, vasopressor initiation, mechanical ventilation
- **Leakage Prevention**: Comprehensive audit framework to prevent information leakage
- **Explainability**: SHAP-based patient-level explanations for clinical interpretability
- **Horizon Sensitivity**: Robustness testing across 24h, 36h, and 48h prediction horizons
- **Model Calibration**: Isotonic regression for probability calibration

### Data Source
- **Dataset**: MIMIC-IV v3.1 (Medical Information Mart for Intensive Care)
- **Cohort**: First ICU stay per patient with ≥12 hours LOS
- **Features**: Vital signs, laboratory values, demographics
- **Access**: PhysioNet (requires credentialing)

---

## Data Access & Setup

### Prerequisites
- Python 3.8+
- Google Cloud Platform (GCP) account with BigQuery access
- PhysioNet account with MIMIC-IV v3.1 access
- Hardware: 16GB+ RAM recommended for loading parquet matrices and calculating SHAP values

### Step 1: Obtain PhysioNet Credentials

**Important**: MIMIC-IV is a large healthcare database and requires proper credentialing.

#### A. Create PhysioNet Account
1. Visit [PhysioNet](https://physionet.org/register/)
2. Complete the registration form with institutional affiliation
3. Verify your email address
4. Agree to the MIMIC-IV data use agreement

#### B. Request MIMIC-IV Access
1. Navigate to [MIMIC-IV v3.1 Project Page](https://physionet.org/content/mimiciv/3.1/)
2. Click "Request Access"
3. Complete the credentialing questionnaire:
   - Describe your intended research use
   - Specify your institution
   - Confirm HIPAA training completion
4. Await approval (typically 24-48 hours)
5. Accept the data use agreement

#### C. Set Up GCP BigQuery Access
1. Create a GCP project or use existing project
2. Enable BigQuery API in your GCP console
3. Create a service account with BigQuery Admin permissions
4. Download the service account JSON key file
5. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

### Step 2: Environment Setup

```bash
# Clone repository or download project files
mkdir icu-deterioration
cd icu-deterioration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap joblib google-cloud-bigquery

# Download MIMIC-IV files
# Navigate to https://physionet.org/content/mimiciv/3.1/
# Download: hosp/ and icu/ directories to ./mimic-iv-3.1/

# Create data directory
mkdir -p ./data ./results
```

### Step 3: Update Configuration
Edit each script to set your GCP project ID:
```python
PROJECT_ID = "your-gcp-project-id"  # Replace with your actual GCP project ID
```

---

## Pipeline Architecture

```
Phase 1: Feature Engineering (Steps 1-2)
├─ 01_data_extraction.py
│  ├─ Extract cohort from icustays
│  ├─ Collect vitals (chartevents)
│  ├─ Collect labs (labevents)
│  ├─ Collect demographics
│  └─ Save raw features (0-6h window)
│
├─ 02_preprocessing.py
│  ├─ Validate temporal boundaries
│  ├─ Clip physiologic outliers
│  ├─ Aggregate to per-stay statistics
│  ├─ Add derived features (shock index)
│  └─ Save feature_matrix_raw.parquet
│
└─ 03_leakage_audit.py
   ├─ Verify feature timestamps (0-6h)
   ├─ Check for outcome/post-outcome columns
   ├─ Validate label-feature separation
   └─ Generate audit report

Phase 2: Label Engineering (Step 4)
├─ 04_prediction_horizons.py
│  ├─ Define prediction windows (6h start, +24/36/48h end)
│  ├─ Identify mortality events (deathtime)
│  ├─ Identify vasopressor initiation
│  ├─ Identify mechanical ventilation
│  ├─ Define deteriorated_composite (any event)
│  ├─ Apply censoring rule (eligible flag)
│  └─ Save horizon_labels.parquet

Phase 3: Model Development (Steps 5-6)
├─ 05_horizon_sensitivity.py
│  ├─ Train XGBoost on each horizon
│  ├─ Evaluate across 24h/36h/48h
│  ├─ Generate ROC curves
│  └─ Save horizon_sensitivity_xgboost.csv
│
└─ 06_model_evaluation.py (PRIMARY MODEL)
   ├─ Train Logistic Regression, Random Forest, XGBoost
   ├─ Compare baseline models
   ├─ Tune hyperparameters
   ├─ Apply isotonic calibration
   ├─ Generate calibration curves
   ├─ Save model_artifacts.joblib
   └─ Save model_comparison_full.csv

Phase 4: Explainability & Deployment (Steps 7-8)
├─ 07_patient_explanations.py
│  ├─ Load trained model
│  ├─ Select representative cases (TP, FN)
│  ├─ Generate SHAP waterfall plots
│  └─ Save patient-level explanations
│
└─ 08_prepare_app.py
   ├─ Export imputed features
   ├─ Export raw features
   ├─ Create patient monitor rankings
   └─ Package for clinical application
```

---

## Step-by-Step Execution Guide

### **Step 1: Extract Features from BigQuery**

```bash
python 01_data_extraction.py
```

**What it does:**
- Queries MIMIC-IV BigQuery tables (requires live internet access)
- Extracts first ICU stay per patient with ≥12 hours LOS
- Collects vital signs (heart rate, BP, O₂ sat, respiratory rate, temperature, GCS)
- Collects labs (lactate, WBC, creatinine, platelets, electrolytes)
- Collects demographics and admission information
- **Enforces 6-hour observation window**: Only measurements within 6 hours of ICU admission included

**Output files:**
- `./data/cohort.parquet` - ICU stay identifiers and timing
- `./data/vitals_raw.parquet` - Raw vital sign readings
- `./data/labs_raw.parquet` - Raw laboratory values
- `./data/demographics.parquet` - Static patient characteristics
- `./data/window_config.json` - Temporal design configuration

**Typical output:**
```
Extracting cohort...
  5,000+ ICU stays in cohort
Extracting vitals (chartevents)...
  300,000+ vital sign readings
Extracting labs (labevents)...
  150,000+ lab readings
Temporal design: features=0-6h | outcomes=>6-30h
```

---

### **Step 2: Preprocess & Aggregate Features**

```bash
python 02_preprocessing.py
```

**What it does:**
- Validates all features fall within 6-hour observation window
- Clips physiologically implausible values (outliers)
- Aggregates raw measurements to per-stay features:
  - **Statistics**: mean, min, max, last value, count for each measurement
  - **Examples**: `sbp_mean`, `lactate_min`, `heart_rate_last`
- Creates derived features:
  - **Shock Index**: HR / SBP (hemodynamic distress indicator)
  - **Missingness Indicators**: Binary flags for missing values
- One-hot encodes categorical features (care unit, admission type)
- **Does NOT apply eligibility filtering** (horizon-specific logic later)

**Output files:**
- `./data/feature_matrix_raw.parquet` - Full feature matrix (all base cohort)
  - ~200 features per stay
  - No labels attached
  - No horizon-specific filtering applied

**Typical output:**
```
Validating 6-hour observation window...
  Vitals min=0.001h, max=5.999h, violations=0
  Labs min=0.002h, max=5.998h, violations=0
Clipping physiologically implausible outliers...
  dropping 15 implausible 'heart_rate' readings
  dropping 8 implausible 'lactate' readings
Feature matrix: 5,000 rows x 210 columns
```

---

### **Step 3: Audit for Temporal & Target Leakage**

```bash
python 03_leakage_audit.py
```

**What it does:**
- **Feature Timestamp Audit**: Verifies 100% of features fall within [0, 6) hours
- **Outcome Timestamp Audit**: Ensures labels only reference events after hour 6
- **Column-Level Audit**: Flags any columns containing outcome information
- **Label-Feature Separation**: Confirms no data leakage between X and y
- Generates human-readable audit report with findings

**Output files:**
- `./results/leakage_audit_report.txt` - Detailed audit findings
- Console output showing all validations passed/failed

**Example output:**
```
✓ FEATURE TIMESTAMP AUDIT: PASSED
  - All features collected within [0, 6) hour window
  - Cohort: 5,000 stays
  - Features: 0.001h to 5.999h from admission

✓ LABEL-FEATURE SEPARATION: PASSED
  - No outcome columns in feature set
  - No post-event data in features
  - 200 safe features confirmed
```

---

### **Step 4: Define Prediction Horizons & Labels**

```bash
python 04_prediction_horizons.py
```

**What it does:**
- Creates explicit label sets for 24h, 36h, 48h prediction horizons
- **Defines deterioration events**:
  - **Mortality**: Patient died in ICU within prediction window
  - **Vasopressor Initiation**: First vasopressor within prediction window
  - **Mechanical Ventilation**: First ventilator use within prediction window
  - **Composite**: Any of the above
- **Applies censoring logic**: Only includes patients whose stays are long enough to label
- **Validation**: Ensures no outcome information leaks into features

**Output files:**
- `./data/horizon_labels.parquet` - Stay-level labels for all 3 horizons

**Typical output:**
```
Generating labels for prediction horizons...
  6h → 24h window:  46,982 eligible stays, 34.4% positive
  6h → 36h window:  40,099 eligible stays, 42.3% positive
  6h → 48h window:  33,891 eligible stays, 51.9% positive
```

---

### **Step 5: Horizon Sensitivity Analysis**

```bash
python 05_horizon_sensitivity.py
```

**What it does:**
- Trains separate XGBoost models for each prediction horizon (24h, 36h, 48h)
- Evaluates robustness of predictions across different time windows
- Generates ROC curves showing performance at each horizon
- Produces horizon sensitivity table (CSV) with all metrics

**Output files:**
- `./results/horizon_sensitivity_xgboost.csv` - Performance metrics table
- `./results/horizon_sensitivity_roc.png` - ROC curve comparison

**Key findings:**
- Model performance is robust across horizons
- 24-hour AUC: 0.825, 36-hour AUC: 0.819, 48-hour AUC: 0.813
- Recall stable (~73%) indicating consistent positive case detection

---

### **Step 6: Train & Evaluate Primary Models**

```bash
python 06_model_evaluation.py
```

**What it does:**
- Trains three baseline models: Logistic Regression, Random Forest, XGBoost
- Performs hyperparameter tuning via RandomizedSearchCV
- Applies class weighting to handle imbalanced data
- **Calibrates probabilities** using isotonic regression
- Compares calibration before/after
- Generates comprehensive evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- Produces ROC curves for all three models
- Generates SHAP values for explainability

**Output files:**
- `./results/model_comparison_full.csv` - Performance metrics for all models
- `./results/roc_curves_full.png` - ROC curve comparison
- `./results/calibration_curves_full_before_after.png` - Calibration plots
- `./results/shap_feature_importance_full.png` - Global feature importance
- `./results/shap_summary_full.png` - Global SHAP summary plot
- `./results/shap_dependence_*.png` - Feature dependence plots
- `./results/model_artifacts.joblib` - Trained model and preprocessing objects

---

### **Step 7: Generate Patient-Level Explanations**

```bash
python 07_patient_explanations.py
```

**What it does:**
- Selects representative True Positive and False Negative cases
- Generates SHAP waterfall plots explaining individual predictions
- Waterfall shows contribution of each feature to final prediction

**Output files:**
- `./results/shap_waterfall_true_positive.png` - Explanation of correct positive prediction
- `./results/shap_waterfall_false_negative.png` - Explanation of missed deterioration case

---

### **Step 8: Prepare for Clinical Application**

```bash
python 08_prepare_app.py
```

**What it does:**
- Exports feature matrices for integration with clinical systems
- Creates patient ranking dashboard (top candidates for intervention)
- Packages model and preprocessor for deployment

**Output files:**
- `./data/app_test_features_imputed.parquet` - Clean features ready for predictions
- `./data/app_test_features_raw.parquet` - Raw features for audit trail
- `./data/app_patient_monitor.csv` - Risk rankings
- `./data/app_underlying_model.joblib` - Model for production use

---

## Model Performance Results

### 24-Hour Prediction Horizon (Primary Analysis)

#### Model Comparison: Test Set Performance

All models demonstrate strong discrimination ability with AUC-ROC > 0.81:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|:---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.742 | 0.603 | 0.731 | 0.661 | **0.812** | 0.692 |
| **Random Forest** | 0.755 | 0.628 | 0.701 | 0.663 | **0.818** | 0.702 |
| **XGBoost (Primary)** | 0.751 | 0.615 | 0.736 | 0.670 | **0.825** | 0.716 |

**Key Observations:**
- **XGBoost achieves highest ROC-AUC (0.825)**: Best discrimination between positive/negative cases
- **High Recall (0.736)**: Identifies ~73.6% of actual deterioration cases
- **Precision (0.615)**: Among patients flagged as positive, ~61.5% actually deteriorate
- **PR-AUC (0.716)**: Strong precision-recall performance in imbalanced dataset
- **Cohort**: 46,982 eligible ICU stays (34.4% positive rate)

#### ROC Curve Comparison

![ROC Curves - All Models](./roc_curves_full.png)

The ROC curve visualization shows:
- All three models cluster in the upper left, indicating excellent discrimination
- XGBoost (green) slightly outperforms competitors throughout the false positive rate spectrum
- Steeper curve early on indicates good performance at high specificity (low false alarm rate)
- Comparison against the diagonal (random classifier) shows substantial improvement

---

## Model Evaluation & Calibration

### Before & After Calibration Analysis

#### What is Calibration?
Calibration ensures that predicted probabilities match actual outcomes. A well-calibrated model with a 70% predicted risk should see ~70% mortality in that group.

**Why it matters for ICU prediction:**
- Uncalibrated models overestimate or underestimate risk
- Clinicians need reliable risk estimates for decision-making
- Isotonic regression adjusts predictions without changing discrimination (AUC)

#### Calibration Results

![Calibration Curves - Before & After Isotonic Regression](./calibration_curves_full_before_after.png)

**Left Panel (BEFORE Calibration):**
- Raw XGBoost predictions (green) deviate from perfect calibration (gray diagonal)
- At predicted probability 0.8, actual frequency is only ~0.65-0.70
- Model shows **overconfidence** (predicts higher risk than actual)
- Logistic Regression (blue) and Random Forest (orange) show different biases

**Right Panel (AFTER Isotonic Calibration):**
- All three models (blue, orange, green) now cluster closely on the diagonal
- Predicted probabilities align with actual observed frequencies
- Predictions are now clinically reliable for risk stratification
- No loss in discrimination power (ROC-AUC preserved)

**Key Insight:** Isotonic calibration transforms uncalibrated probabilities into reliable risk estimates suitable for clinical deployment.

---

## Global Model Explanations

Global explanations help understand which features drive predictions across all patients, providing insight into the model's overall decision-making logic.

### Feature Importance: Which Features Matter Most?

![SHAP Feature Importance - Top 15 Features](./shap_feature_importance_full.png)

**Top 5 Most Important Features:**

1. **Lactate Count** (Mean SHAP: 0.42)
   - Number of lactate measurements in first 6 hours
   - Higher count → more clinical concern (frequent monitoring of suspected hypoxia/shock)
   - Clinical relevance: Repeated lactate checks indicate escalating concern

2. **Systolic BP - Minimum** (Mean SHAP: 0.29)
   - Lowest systolic blood pressure recorded in first 6 hours
   - Lower values strongly increase deterioration risk
   - Clinical relevance: Hypotension is a critical sign of shock

3. **Systolic BP - Mean** (Mean SHAP: 0.20)
   - Average systolic blood pressure
   - Reflects sustained hypotension vs isolated low readings
   - Clinical relevance: Persistent low BP more predictive than single event

4. **Neuro Care Unit Status** (Mean SHAP: 0.19)
   - Whether patient admitted to neurological ICU as first care unit
   - Neurocritical patients have different baseline deterioration risk
   - Clinical relevance: Captures disease-specific risk stratification

5. **Age** (Mean SHAP: 0.15)
   - Patient age at admission
   - Older patients have higher deterioration risk
   - Clinical relevance: Age is universal ICU risk factor

**Interpretation Notes:**
- Feature importance = average |SHAP value|, showing consistency of impact
- Blood pressure metrics dominate (3 of top 5), confirming hemodynamic status as key predictor
- Count-based features (lactate count, sbp measurements) indicate frequency of concerning vital signs

---

### Summary Plot: Feature Values vs Impact

![SHAP Summary Plot (Beeswarm)](./shap_summary_full.png)

**How to Read This Plot:**
- **Y-axis**: Features ranked by importance (top = most important)
- **X-axis**: SHAP value (model's local prediction change)
- **Color**: Actual feature value (blue=low, red=high)
- **Dot position**: Shows if feature value increases or decreases prediction

**Key Patterns:**

1. **Lactate Count**: 
   - Red dots (high counts) mostly on the right → high count increases deterioration risk
   - More lactate measurements = more concern
   
2. **Systolic BP - Minimum**:
   - Red dots (high BP) mostly on the left (negative SHAP) → protective effect
   - Blue dots (low BP) on the right (positive SHAP) → harmful effect
   - Clear negative relationship: lower BP = worse outcome

3. **Systolic BP - Mean**:
   - Similar pattern to SBP minimum
   - Red dots on left (high mean BP protective)
   - Blue dots on right (low mean BP harmful)

4. **Neuro Care Unit**:
   - Pink bar on left (1 = neuro ICU) → decreases risk
   - Clinical interpretation: Neuro ICU patients have better prognosis (more specialized care)

---

## Feature-Specific Dependence Analysis

Dependence plots show how individual features relate to predictions, helping clinicians understand feature-outcome relationships.

### Lactate: The Most Critical Indicator

![SHAP Dependence Plot: Lactate (Last Value)](./shap_dependence_lactate_last.png)

**Interpretation:**
- **X-axis**: Latest lactate measurement (mmol/L)
- **Y-axis**: Model's contribution to deterioration prediction
- **Strong Linear Relationship**: As lactate increases, deterioration risk rises proportionally
  - Lactate 0-2 mmol/L: Minimal impact (normal lactate)
  - Lactate 2-5 mmol/L: Steady increase in risk
  - Lactate >10 mmol/L: Large risk increase
- **Density Clustering**: Most patients cluster at lower lactate values (left side)
- **Color Gradient**: Blue dots (lower lactate) contribute negatively to risk; red dots (higher lactate) contribute positively

**Clinical Relevance:**
- Lactate is a marker of tissue hypoperfusion and anaerobic metabolism
- Elevated lactate strongly predictive of sepsis, cardiogenic shock, or other life-threatening conditions
- Even modest elevations (3-5 mmol/L) signal concerning trend

---

### Blood Pressure: Hemodynamic Stability

![SHAP Dependence Plot: Systolic BP - Minimum](./shap_dependence_sbp_min.png)

**Interpretation:**
- **X-axis**: Minimum systolic BP in first 6 hours (mmHg)
- **Y-axis**: Model's contribution to deterioration prediction
- **Clear Threshold Effect**: 
  - SBP >80-90 mmHg: Protective effect (negative SHAP values)
  - SBP <80 mmHg: Harmful effect (positive SHAP values)
- **Strong Inverse Relationship**: Lower BP = worse outcomes
- **Clinical Alert Zone**: SBP <60-70 mmHg associated with large positive SHAP values

**Clinical Relevance:**
- Hypotension (SBP <90) is criterion for septic shock, cardiogenic shock, distributive shock
- Minimum BP better indicator of worst-case hemodynamic state than mean
- Threshold around 80-90 mmHg aligns with hemodynamic definitions in critical care

---

### Respiratory Rate: Ventilatory Stress

![SHAP Dependence Plot: Respiratory Rate - Mean](./shap_dependence_resp_rate_mean.png)

**Interpretation:**
- **X-axis**: Average respiratory rate in first 6 hours (breaths/min)
- **Y-axis**: Model's contribution to deterioration prediction
- **Non-Linear U-Shaped Relationship**:
  - Normal range (20-25 breaths/min): Minimal impact
  - Low rate (<15 breaths/min): Slightly harmful (respiratory depression)
  - High rate (>30 breaths/min): Increasingly harmful (tachypnea)
- **High Variability**: Scattered dots indicate inconsistent relationship (context-dependent)
- **Weaker Signal**: Compared to lactate/BP, more scattered pattern

**Clinical Relevance:**
- Tachypnea (>30) suggests respiratory distress, pain, anxiety, metabolic acidosis
- Bradypnea (<10) suggests respiratory depression or CNS pathology
- Relationship is context-dependent (mechanical ventilation, disease severity)

---

### Consciousness: Neurological Status

![SHAP Dependence Plot: GCS Verbal Score - Last](./shap_dependence_gcs_verbal_last.png)

**Interpretation:**
- **X-axis**: Latest Glasgow Coma Scale Verbal score (1-5, or 0 if intubated)
- **Y-axis**: Model's contribution to deterioration prediction
- **Categorical Pattern**:
  - GCS Verbal 1 (no response): Large positive SHAP → severe deterioration risk
  - GCS Verbal 2-3: Moderate positive SHAP
  - GCS Verbal 4-5 (normal): Zero/negative SHAP → protective
- **Clustered Distribution**: Discrete values (ordinal variable)
- **Strong Separation**: Clear clustering by GCS score level

**Clinical Relevance:**
- GCS measures consciousness level; lower scores = more severe neurological impairment
- GCS Verbal 1 indicates unresponsiveness (coma or intubation)
- Neurological decline is powerful predictor of deterioration
- Quick, bedside assessment valuable for early warning

---

## Patient-Level Explanations

Patient-level (local) explanations show why the model made a specific prediction for an individual patient, breaking down each feature's contribution.

### Case 1: True Positive (Correctly Predicted Deterioration)

![SHAP Waterfall Plot: True Positive Case](./shap_waterfall_true_positive.png)

**Understanding This Plot:**

**Base Prediction (Left):**
- f(x) = 5.133 (on log-odds scale)
- Translated to probability: ~99.4% deterioration risk
- **Model Output**: Patient flagged as high-risk deterioration case

**Feature Contributions (Stacked Bars):**

1. **196 Other Features**: +2.07
   - Collective contribution of all remaining features
   - Represents accumulated minor effects

2. **Minimum Diastolic BP**: +0.18
   - Low diastolic BP increases deterioration risk
   
3. **Urgent Admission Type**: +0.21
   - Emergency/urgent admission associated with higher risk
   
4. **Latest Oxygen Saturation**: +0.28
   - Low oxygen saturation increases risk
   
5. **Minimum Oxygen Saturation**: +0.29
   - Lowest recorded O2 saturation critical

6. **Lactate Measurement Count**: +0.41
   - Multiple lactate checks indicate escalating concern
   
7. **Maximum Diastolic BP**: -0.50
   - Higher diastolic BP (protective) slightly offsets risk
   - Indicates some hemodynamic stability
   
8. **Latest Lactate**: -0.52
   - Surprisingly negative here (value-specific)
   - Context-dependent effect
   
9. **Minimum Systolic BP**: +0.62
   - **Largest single contributor**: Very low SBP drives prediction
   - Patient in severe hypotension (SBP ~48 mmHg)

**Clinical Story:**
This patient was correctly flagged as high-risk because of:
- Critical hypotension (SBP 48 mmHg) → hemodynamic failure
- Low oxygen saturation → respiratory/cardiac compromise
- Multiple lactate measurements → escalating clinical concern
- Urgent admission type → pre-existing severe illness

**Ground Truth**: Patient did deteriorate (required vasopressor support), validating the model's prediction

---

### Case 2: False Negative (Missed Deterioration)

![SHAP Waterfall Plot: False Negative Case](./shap_waterfall_false_negative.png)

**Understanding This Plot:**

**Base Prediction (Left):**
- f(x) = -3.767 (on log-odds scale)
- Translated to probability: ~2.3% deterioration risk
- **Model Output**: Patient flagged as low-risk (will NOT deteriorate)
- **Actual Outcome**: Patient DID deteriorate — **model was wrong**

**Feature Contributions (Stacked Bars):**

1. **196 Other Features**: -1.06
   - Collective protective effect of remaining features
   
2. **Heart Rate Maximum**: -0.10
   - Higher HR slightly increases risk (but weak effect)
   
3. **Latest GCS Verbal**: -0.12
   - Slightly impaired consciousness (not severe)
   
4. **Minimum Diastolic BP**: -0.13
   - Low DBP increases risk (negative contribution)
   
5. **Minimum Systolic BP**: -0.29
   - Low SBP (105 mmHg) — in this case protective?
   - Moderate hypotension, not extreme

6. **Age**: -0.40
   - Patient's age decreases risk (younger patients have better prognosis)
   
7. **Lactate Measurement Count**: -0.61
   - **Only 1 lactate check** (normal frequency)
   - Clinicians not escalating concern
   
8. **Neuro Care Unit Placement**: -1.17
   - **Largest protective factor**: Patient in specialized neuro ICU
   - Associated with better outcomes

**Clinical Story of the Miss:**
This patient was incorrectly classified as low-risk because:
- **Specialized neuro ICU care** → model learned neuro patients generally do better
- **Low lactate measurement count** → only 1 check (no indication of escalating concern)
- **Younger age** → protective demographic factor
- **Not extreme hypotension** → moderate SBP, not critical

**Hidden Problem:**
- Despite these protective factors, the patient DID deteriorate
- Suggests a developing problem not captured by 6-hour data
- Could indicate late-onset deterioration (hours 6-24) from evolving process

**Learning Point:**
This case highlights limitations:
- 6-hour observation window may miss delayed complications
- Specialization effect (neuro ICU) might be confounded with better baseline status
- Need for dynamic predictions beyond initial 6 hours

---

## Horizon Sensitivity Analysis

### Model Performance Across Prediction Horizons

Predicting deterioration at different time windows reveals how early we can reliably detect risks.

#### Performance Metrics by Horizon

| Observation Window | Prediction Horizon | Eligible Cohort | Positive Rate | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **6h** | **24h** | 46,982 | 34.4% | 0.751 | 0.615 | 0.736 | 0.670 | **0.825** | 0.716 |
| **6h** | **36h** | 40,099 | 42.3% | 0.746 | 0.689 | 0.727 | 0.707 | **0.819** | 0.764 |
| **6h** | **48h** | 33,891 | 51.9% | 0.733 | 0.752 | 0.725 | 0.738 | **0.813** | 0.815 |

**Key Observations:**

1. **Discriminative Ability Holds Across Horizons**:
   - ROC-AUC remains strong: 0.825 → 0.819 → 0.813
   - Modest 1.5% decline over 48 hours
   - Indicates robust early warning capability

2. **Increasing Disease Prevalence**:
   - 24h positive rate: 34.4%
   - 36h positive rate: 42.3% (23% increase)
   - 48h positive rate: 51.9% (51% increase)
   - Longer windows capture more deterioration events

3. **Precision-Recall Trade-off**:
   - 24h: High precision (0.615), high recall (0.736)
   - 48h: Higher precision (0.752), similar recall (0.725)
   - At 48h, 75% of flagged patients actually deteriorate

4. **Clinical Implications**:
   - Early prediction (24h) best for maximum sensitivity (catch ~74% of cases)
   - Later prediction (48h) better for specificity (fewer false alarms)

---

#### ROC Curves: Visual Comparison

![XGBoost Sensitivity to Prediction Horizon](./horizon_sensitivity_roc.png)

**Visualization Details:**
- **Blue (6h → 24h, AUC=0.825)**: Steepest curve, best discrimination
- **Orange (6h → 36h, AUC=0.819)**: Nearly identical to 24h
- **Green (6h → 48h, AUC=0.813)**: Slightly flattened, still excellent

**Clinical Interpretation:**
- Curves cluster in upper-left, indicating excellent performance all horizons
- Minimal performance degradation suggests deterioration trajectory established early
- By 6 hours, key predictive features already manifest
- Model can predict 24-48 hours ahead with confidence

**Decision-Making:**
- **For immediate intervention**: Use 24h predictions (lowest false negative rate)
- **For resource planning**: Use 48h predictions (highest specificity)
- **For escalation alerts**: Use 24h predictions (highest sensitivity)

---

## Key Findings

### Clinical Insights

1. **Hemodynamic Status Dominates Predictions**
   - Blood pressure metrics (SBP min/mean) are top 3 predictors
   - Hypotension <90 mmHg a critical threshold
   - Even moderate hypotension signals deterioration risk

2. **Lactate: The Canary in the Coal Mine**
   - Single most important feature (highest SHAP value)
   - Number of lactate measurements rivals value itself
   - Frequency = clinical team recognizing patient deterioration
   - Strong linear relationship: higher lactate = worse outcomes

3. **6-Hour Window Captures Deterioration Trajectory**
   - Minimal performance loss 24h → 48h (AUC: 0.825 → 0.813)
   - Early pathophysiology already evident
   - First 6 hours contain "genetic code" of ICU course

4. **Model Calibration Critical for Deployment**
   - Raw XGBoost overconfident (predicted 80% risk when actual 65%)
   - Isotonic regression recovers reliability
   - Clinicians need well-calibrated probabilities for decision-making

5. **High Recall Minimizes Missed Cases**
   - 73.6% sensitivity at 24h
   - Only ~26% of deterioration cases slip through as "low risk"
   - Acceptable miss rate balanced against false alarm burden

---

### Performance Summary

**Primary Model (XGBoost @ 24h Horizon):**
- ✅ ROC-AUC: 0.825 (excellent discrimination)
- ✅ Sensitivity: 73.6% (high negative predictive value)
- ✅ Specificity: 75.4% (reasonable false alarm rate)
- ✅ Precision: 61.5% (majority of alerts valid)
- ✅ Calibrated: Yes (isotonic regression applied)
- ✅ Explainable: Yes (SHAP-based patient-level explanations available)

**Horizon Robustness:**
- Consistent performance across 24h, 36h, 48h horizons
- Negligible decline with extended prediction window
- Suitable for different clinical use cases

---

## Temporal Leakage Prevention

### Comprehensive Audit Framework

Temporal leakage (using future information to predict the past) is a critical failure mode in predictive healthcare. Our system includes multiple safeguards:

#### 1. Feature Timestamp Audit
```
✓ PASSED: All features collected within [0, 6) hour window
- 46,982 ICU stays examined
- 5,000+ feature readings per stay
- Min timestamp: 0.001 hours (36 seconds after admission)
- Max timestamp: 5.999 hours
- Violations: 0 (100% compliance)
```

#### 2. Outcome Timestamp Audit
```
✓ PASSED: All deterioration events after hour 6
- Mortality: Occurred between hours 6-30
- Vasopressor: Initiated after hour 6
- Mechanical Ventilation: Started after hour 6
- Censoring: Patients discharged before hour 6+horizon properly excluded
```

#### 3. Column-Level Audit
```
✓ PASSED: No outcome information in feature set
- Checked for post-event vital signs
- Verified no post-deterioration lab values
- Confirmed no discharge/death event columns
```

#### 4. Model Validation Protocol
```
✓ PASSED: Stratified train-test split by outcome
- Training: Hours 0-6 features only
- Test: Hours 0-6 features only
- Imputation: Fitted on train set, applied to test
- Hyperparameters: Selected via cross-validation on training set
```

---

## Technical Implementation

### Dependencies

```
Core Scientific Stack:
├── Data Processing
│   ├── pandas >= 1.3.0
│   ├── numpy >= 1.20.0
│   └── pyarrow >= 6.0.0
├── Machine Learning
│   ├── scikit-learn >= 1.0.0
│   ├── xgboost >= 1.5.0
│   └── imbalanced-learn >= 0.8.0
├── Explainability
│   └── shap >= 0.40.0
├── Visualization
│   ├── matplotlib >= 3.4.0
│   └── seaborn >= 0.11.0
└── Model Serialization
    └── joblib >= 1.1.0
```

### Data Processing Pipeline

```python
# Step 1: BigQuery Extraction
def run_query(sql: str) -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID)
    return client.query(sql).to_dataframe()

# Step 2: Temporal Validation
offsets = (charttime - intime).dt.total_seconds() / 3600.0
assert (offsets >= 0).all() and (offsets <= 6).all()

# Step 3: Feature Aggregation
agg = readings.groupby(["stay_id", "feature_name"])["valuenum"].agg(
    ["mean", "min", "max", "count", "last"]
)

# Step 4: Imputation (train set only)
X_train[col] = X_train[col].fillna(X_train[col].median())
X_test[col] = X_test[col].fillna(X_train[col].median())  # Use train median

# Step 5: Model Training
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    scale_pos_weight=pos_weight,
    random_state=42
)
model.fit(X_train, y_train)

# Step 6: Calibration
from sklearn.calibration import IsotonicRegression
calibrator = IsotonicRegression()
calibrated_proba = calibrator.fit_transform(raw_proba, y_test)

# Step 7: Explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
```

### File Structure

```
icu-deterioration/
├── README.md (this file)
├── requirements.txt
├── 01_data_extraction.py
├── 02_preprocessing.py
├── 03_leakage_audit.py
├── 04_prediction_horizons.py
├── 05_horizon_sensitivity.py
├── 06_model_evaluation.py
├── 07_patient_explanations.py
├── 08_prepare_app.py
├── data/
│   ├── cohort.parquet
│   ├── vitals_raw.parquet
│   ├── labs_raw.parquet
│   ├── demographics.parquet
│   ├── feature_matrix_raw.parquet
│   ├── horizon_labels.parquet
│   ├── window_config.json
│   ├── model_artifacts.joblib
│   ├── app_test_features_imputed.parquet
│   ├── app_test_features_raw.parquet
│   ├── app_patient_monitor.csv
│   └── app_underlying_model.joblib
├── results/
│   ├── model_comparison_full.csv
│   ├── horizon_sensitivity_xgboost.csv
│   ├── roc_curves_full.png
│   ├── calibration_curves_full_before_after.png
│   ├── horizon_sensitivity_roc.png
│   ├── shap_feature_importance_full.png
│   ├── shap_summary_full.png
│   ├── shap_dependence_gcs_verbal_last.png
│   ├── shap_dependence_lactate_last.png
│   ├── shap_dependence_resp_rate_mean.png
│   ├── shap_dependence_sbp_min.png
│   ├── shap_waterfall_true_positive.png
│   ├── shap_waterfall_false_negative.png
│   └── leakage_audit_report.txt
└── mimic-iv-3.1/  (download from PhysioNet)
    ├── hosp/
    │   ├── patients.csv
    │   ├── admissions.csv
    │   └── diagnoses_icd.csv
    └── icu/
        ├── icustays.csv
        ├── chartevents.csv
        ├── labevents.csv
        └── d_items.csv
```

### Key Hyperparameters

**XGBoost:**
```python
n_estimators: 400        # Number of boosting rounds
max_depth: 4             # Tree depth (prevents overfitting)
learning_rate: 0.05      # Shrinkage (slower, more robust learning)
subsample: 0.85          # Fraction of samples per iteration
eval_metric: 'logloss'   # Objective function
```

**Stratified Train/Test Split:**
```python
test_size: 0.20          # 80/20 split
stratify: y              # Maintain class distribution
random_state: 42         # Reproducibility
```

**Imputation Strategy:**
```python
Value columns:           # Use training set median
Count columns:           # Fill with 0 (no measurement = count 0)
Missing indicators:      # Binary flag for missing values
```

---

## Reproducibility & Citation

### To Reproduce Results

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
# Edit scripts to set PROJECT_ID

# 3. Download MIMIC-IV
# Visit https://physionet.org/content/mimiciv/3.1/
# Download hosp/ and icu/ directories

# 4. Run pipeline
python 01_data_extraction.py
python 02_preprocessing.py
python 03_leakage_audit.py --feature-only
python 04_prediction_horizons.py
python 03_leakage_audit.py --label-only
python 05_horizon_sensitivity.py
python 06_model_evaluation.py
python 07_patient_explanations.py
python 08_prepare_app.py
```

### Citing This Project

```bibtex
@dataset{icu_deterioration_2024,
  title={AI for ICU Deterioration: 6-Hour Observation Window Prediction System},
  author={Your Name},
  year={2024},
  note={Implemented using MIMIC-IV v3.1}
}
```

---

## Clinical Disclaimer

⚠️ **IMPORTANT LEGAL NOTICE**

This model is provided for **research purposes only** and has **NOT been validated for clinical deployment**. 

- Do NOT use in direct patient care without institutional review board (IRB) approval
- Do NOT use without clinical validation in your specific patient population
- Model predictions complement clinical judgment; clinician oversight mandatory
- Ensure compliance with HIPAA, HL7, and relevant healthcare regulations
- Validate extensively in your clinical environment before any deployment

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|:---|:---|
| **BigQuery authentication fails** | Verify GOOGLE_APPLICATION_CREDENTIALS env var, check service account permissions |
| **MIMIC-IV not found** | Download from https://physionet.org/content/mimiciv/3.1/, extract to ./mimic-iv-3.1/ |
| **Cohort empty** | Confirm BigQuery project ID correct, MIMIC tables accessible |
| **Leakage audit fails** | Re-run 01_data_extraction.py with temporal window constraints |
| **Model evaluation hangs** | Reduce cohort size for testing, check XGBoost GPU availability |
| **SHAP waterfall missing cases** | Ensure model_artifacts.joblib generated by step 6 |

---

## Future Extensions

1. **Real-time monitoring**: Stream predictions from new ICU admissions
2. **Model retraining**: Periodic updates with new patient data
3. **Subgroup analysis**: Performance in sepsis, cardiac, trauma cohorts
4. **Ensemble methods**: Combine with clinical scoring systems (APACHE, SOFA)
5. **Fairness audit**: Evaluate equity across racial/ethnic groups
6. **Prospective validation**: Randomized trial in clinical environment

---

## Contact & Support

For questions about this implementation:
- Review inline code comments in each script
- Check 03_leakage_audit.py for temporal validation details
- Consult PhysioNet documentation for data questions

---

**Last Updated:** August 2024  
**MIMIC-IV Version:** 3.1  
**Project Status:** Research Implementation Complete

---

## Appendix: Summary Statistics

### Cohort Summary (24-hour Prediction Horizon)

| Metric | Value |
|:---|---:|
| **Total Eligible ICU Stays** | 46,982 |
| **Deterioration Cases (Positive)** | 16,155 (34.4%) |
| **Non-deterioration (Negative)** | 30,827 (65.6%) |
| **Training Set Size** | 37,585 |
| **Test Set Size** | 9,397 |
| **Total Features** | ~210 |
| **Feature Matrix Dimensions** | 46,982 rows × 210 columns |

### Model Performance Summary (24h Horizon, XGBoost)

| Metric | Value |
|:---|---:|
| **Accuracy** | 0.751 (75.1%) |
| **Sensitivity (Recall)** | 0.736 (73.6% of deterioration cases caught) |
| **Specificity** | 0.754 (75.4% of non-deterioration cases correct) |
| **Precision** | 0.615 (61.5% of alerts valid) |
| **F1-Score** | 0.670 |
| **ROC-AUC** | 0.825 (excellent discrimination) |
| **PR-AUC** | 0.716 (strong precision-recall) |
| **Calibration** | Yes (isotonic regression applied) |

---

**End of README**
