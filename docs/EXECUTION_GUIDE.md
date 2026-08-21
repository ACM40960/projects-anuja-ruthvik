# Explainable AI for Early ICU Deterioration Prediction : Execution Guide

A detailed step-by-step guide for running the machine learning pipeline to predict ICU patient deterioration.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Pipeline Overview](#pipeline-overview)
4. [Detailed Step-by-Step Instructions](#detailed-step-by-step-instructions)
5. [Output Files & Verification](#output-files--verification)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap joblib google-cloud-bigquery

# 3. Configure credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# 4. Run pipeline (in order)
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

---

## Prerequisites & Setup

### Environment Requirements
- **Python**: 3.8 or higher
- **RAM**: 16GB+ recommended (for SHAP value calculations)
- **Storage**: 50GB+ for MIMIC-IV dataset and outputs
- **Internet**: Required for BigQuery access

### 1. PhysioNet Credentialing (Required)

**This step is mandatory and takes 24-48 hours**

1. Register at [PhysioNet](https://physionet.org/register/)
2. Complete registration with institutional affiliation
3. Navigate to [MIMIC-IV v3.1 Project Page](https://physionet.org/content/mimiciv/3.1/)
4. Click "Request Access" and complete the credentialing questionnaire:
   - Describe your intended research use
   - Verify institutional affiliation
   - Confirm HIPAA training completion
5. Await approval (typically 24-48 hours)
6. Accept the data use agreement once approved

### 2. Google Cloud Platform Setup

1. Create a GCP project (or use existing)
2. Enable BigQuery API:
   - Go to GCP Console → APIs & Services → Enable APIs
   - Search for "BigQuery API" and enable it
3. Create a service account:
   - Go to IAM & Admin → Service Accounts
   - Click "Create Service Account"
   - Grant "BigQuery Admin" role
   - Create JSON key file
4. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

### 3. Local Environment Setup

```bash
# Create project directory
mkdir icu-deterioration
cd icu-deterioration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap joblib google-cloud-bigquery

# Create required directories
mkdir -p ./data ./results
```

### 4. Update Configuration

Edit each Python script and update your GCP project ID:
```python
PROJECT_ID = "your-actual-gcp-project-id"  # Line typically near top of each script
```

### 5. Download MIMIC-IV Data

After PhysioNet approval:
1. Go to [MIMIC-IV v3.1 Download Page](https://physionet.org/content/mimiciv/3.1/)
2. Download the following directories:
   - `hosp/` (contains: patients.csv, admissions.csv, diagnoses_icd.csv)
   - `icu/` (contains: icustays.csv, chartevents.csv, labevents.csv, d_items.csv)
3. Extract to `./mimic-iv-3.1/` directory in your project folder

---

## Pipeline Overview

The pipeline consists of **8 sequential steps** organized into **4 phases**:

```
Phase 1: Feature Engineering (Steps 1-3)
  └─ Extract, preprocess, and audit features (0-6 hour window)

Phase 2: Label Engineering (Step 4)
  └─ Define deterioration outcomes (24/36/48 hour horizons)

Phase 3: Model Development (Steps 5-6)
  └─ Train models and evaluate performance

Phase 4: Explainability & Deployment (Steps 7-8)
  └─ Generate explanations and package for application
```

---

## Detailed Step-by-Step Instructions

### **Step 1: Extract Features from BigQuery**

**Command:**
```bash
python 01_data_extraction.py
```

**Purpose:**
- Query MIMIC-IV BigQuery tables (requires internet access)
- Extract first ICU stay per patient with ≥12 hours length of stay
- Collect vital signs and laboratory values from 0-6 hour window
- Collect demographic information

**What it extracts:**

| Data Type | Examples | Source |
|:---|:---|:---|
| **Vitals** | Heart rate, Blood pressure, O₂ saturation, Respiratory rate, Temperature, GCS | chartevents |
| **Labs** | Lactate, WBC, Creatinine, Platelets, Electrolytes (K, Na, Cl) | labevents |
| **Demographics** | Age, Gender, Admission type, Care unit | patients, admissions, icustays |
| **Temporal** | ICU admission time, measurements timestamps | icustays, chartevents |

**Key Features:**
- ✓ Enforces strict 6-hour observation window
- ✓ Filters for first ICU stay only
- ✓ Requires ≥12 hours ICU length of stay
- ✓ Handles missing values (imputed later)

**Output Files:**
```
./data/cohort.parquet                 # ICU stay identifiers and timing
./data/vitals_raw.parquet             # Raw vital sign readings
./data/labs_raw.parquet               # Raw laboratory values
./data/demographics.parquet           # Static patient characteristics
./data/window_config.json             # Temporal design configuration
```

**Expected Console Output:**
```
Extracting cohort...
  ✓ 5,000+ ICU stays identified
Extracting vitals (chartevents)...
  ✓ 300,000+ vital sign readings
Extracting labs (labevents)...
  ✓ 150,000+ lab readings
Temporal design: features=0-6h | outcomes=>6-30h
✓ Data extraction complete!
```

**Typical Runtime:** 10-30 minutes (depends on BigQuery load)

---

### **Step 2: Preprocess & Aggregate Features**

**Command:**
```bash
python 02_preprocessing.py
```

**Purpose:**
- Validate all features fall within 6-hour observation window
- Remove physiologically implausible values (outliers)
- Aggregate raw measurements to per-stay statistics
- Create derived features and handle missing values

**What it does:**

1. **Temporal Validation**
   - Verify all chartevents timestamps are within 0-6 hours of ICU admission
   - Verify all labevents timestamps are within 0-6 hours

2. **Outlier Clipping**
   - Heart rate: 20-200 bpm
   - Blood pressure: 40-300 mmHg
   - Temperature: 32-43°C
   - Oxygen saturation: 50-100%

3. **Feature Aggregation** (for each measurement type)
   - `mean` - Average value
   - `min` - Minimum value
   - `max` - Maximum value
   - `last` - Most recent value
   - `count` - Number of measurements (missingness indicator)

4. **Derived Features**
   - **Shock Index**: Heart Rate / Systolic BP (hemodynamic distress)
   - **Missing Indicators**: Binary flags for unmeasured values

5. **Categorical Encoding**
   - Care unit: One-hot encoded
   - Admission type: One-hot encoded

**Output Files:**
```
./data/feature_matrix_raw.parquet    # Full feature matrix (~210 features)
                                      # 46,982 rows × 210 columns
                                      # No labels attached yet
```

**Expected Console Output:**
```
Validating 6-hour observation window...
  ✓ 46,982 stays validated
Clipping physiologic outliers...
  ✓ Heart rate: 20-200 bpm
  ✓ Blood pressure: 40-300 mmHg
  ✓ Temperature: 32-43°C
Aggregating features to per-stay statistics...
  ✓ 210 features created
Handling missing values...
  ✓ Missing indicators added
✓ Preprocessing complete!
```

**Feature Count:** ~210 per ICU stay

**Typical Runtime:** 5-10 minutes

---

### **Step 3a: Audit Features for Data Leakage (Feature-Only)**

**Command:**
```bash
python 03_leakage_audit.py --feature-only
```

**Purpose:**
- Verify all features are from 0-6 hour observation window
- Check for any outcome-related columns
- Validate no post-outcome measurements included
- Generate audit report

**What it checks:**
- ✓ Feature timestamps within [0, 6] hours of ICU admission
- ✓ No mortality data in feature set
- ✓ No vasopressor data before prediction window
- ✓ No mechanical ventilation data before prediction window

**Output Files:**
```
./results/leakage_audit_report.txt   # Detailed audit report
```

**Expected Console Output:**
```
Auditing features for temporal integrity...
  ✓ All 210 features within 0-6h window
  ✓ No outcome contamination detected
Feature-only audit: PASSED ✓
```

**Typical Runtime:** 2-3 minutes

---

### **Step 4: Define Prediction Horizons & Labels**

**Command:**
```bash
python 04_prediction_horizons.py
```

**Purpose:**
- Define deterioration events across multiple prediction horizons
- Create binary outcome labels for 24, 36, and 48-hour predictions
- Ensure no information leakage between features and labels

**Deterioration Events Defined:**

| Event | Definition | Source |
|:---|:---|:---|
| **Mortality** | Patient death within horizon | deathtime |
| **Vasopressor** | Initiation of vasopressor drugs | medication events |
| **Mechanical Ventilation** | Initiation of mechanical ventilation | procedure events |
| **Composite** | Any of above three events | Combined logic |

**Prediction Horizons:**

```
Features: 0-6 hours after ICU admission
           |
           |--[6h gap]----|
                          |
                    Prediction Window:
                    24h horizon:  6-30h
                    36h horizon:  6-42h
                    48h horizon:  6-54h
```

**Output Files:**
```
./data/horizon_labels.parquet       # Labels for all three horizons
                                     # Columns: 24h_label, 36h_label, 48h_label
                                     # Also: eligible_24h, eligible_36h, eligible_48h
```

**Expected Console Output:**
```
Defining prediction horizons...
24-hour horizon:
  ✓ Mortality cases: 3,421
  ✓ Vasopressor cases: 5,890
  ✓ Ventilation cases: 4,244
  ✓ Composite (any event): 8,155 (17.4%)

36-hour horizon:
  ✓ Composite (any event): 10,203 (21.7%)

48-hour horizon:
  ✓ Composite (any event): 12,088 (25.7%)

✓ Horizon labels created!
```

**Typical Runtime:** 3-5 minutes

---

### **Step 3b: Audit Labels for Integrity (Label-Only)**

**Command:**
```bash
python 03_leakage_audit.py --label-only
```

**Purpose:**
- Verify outcome events occur after 6-hour feature window
- Check for proper temporal separation
- Validate label integrity

**What it checks:**
- ✓ All mortality events occur after 6h
- ✓ All vasopressor initiations occur after 6h
- ✓ All ventilation initiations occur after 6h
- ✓ Proper eligibility filters applied

**Expected Console Output:**
```
Auditing labels for temporal integrity...
  ✓ All outcomes after 6h window
  ✓ No feature-label overlap detected
Label-only audit: PASSED ✓
```

**Typical Runtime:** 2-3 minutes

---

### **Step 5: Horizon Sensitivity Analysis**

**Command:**
```bash
python 05_horizon_sensitivity.py
```

**Purpose:**
- Train XGBoost models for each prediction horizon (24h, 36h, 48h)
- Compare model performance across horizons
- Identify optimal prediction timing
- Generate sensitivity plots

**What it does:**

1. Train separate XGBoost models for:
   - 24-hour horizon (shortest prediction window)
   - 36-hour horizon (medium prediction window)
   - 48-hour horizon (longest prediction window)

2. Evaluate each model on test set:
   - ROC-AUC scores
   - Accuracy, Sensitivity, Specificity
   - Calibration

3. Compare predictions across horizons:
   - Which horizon achieves best performance?
   - How does sensitivity change with prediction window?

**Output Files:**
```
./results/horizon_sensitivity_xgboost.csv  # Performance metrics by horizon
./results/horizon_sensitivity_roc.png      # ROC curves for all three horizons
```

**Expected Console Output:**
```
Training XGBoost for 24-hour horizon...
  ✓ ROC-AUC: 0.825

Training XGBoost for 36-hour horizon...
  ✓ ROC-AUC: 0.821

Training XGBoost for 48-hour horizon...
  ✓ ROC-AUC: 0.815

✓ Sensitivity analysis complete!
```

**Typical Runtime:** 20-40 minutes

---

### **Step 6: Train & Calibrate Primary Model**

**Command:**
```bash
python 06_model_evaluation.py
```

**Purpose:**
- Train and compare multiple models (Logistic Regression, Random Forest, XGBoost)
- Select best performing model
- Apply isotonic regression for calibration
- Generate comprehensive evaluation plots

**Models Trained:**

| Model | Purpose | Notes |
|:---|:---|:---|
| **Logistic Regression** | Baseline interpretable model | Linear relationships |
| **Random Forest** | Ensemble baseline | Multiple decision trees |
| **XGBoost** | Primary production model | Gradient boosting, best performance |

**What it does:**

1. **Train-Test Split**: 80/20 stratified split
   - Training set: 37,585 stays (80%)
   - Test set: 9,397 stays (20%)

2. **Hyperparameter Tuning** (XGBoost):
   ```python
   n_estimators: 400        # Number of boosting rounds
   max_depth: 4             # Tree depth (prevents overfitting)
   learning_rate: 0.05      # Shrinkage parameter
   subsample: 0.85          # Fraction of samples per iteration
   ```

3. **Imputation Strategy**:
   - Value columns: Use training set median
   - Count columns: Fill with 0
   - Missing indicators: Binary flag

4. **Calibration**:
   - Apply isotonic regression on validation set
   - Ensures probability predictions are well-calibrated
   - Important for clinical use (probability = actual risk)

5. **Evaluation Metrics**:
   ```
   Accuracy:    % of correct predictions
   Sensitivity: % of deterioration cases caught
   Specificity: % of non-deterioration cases correct
   ROC-AUC:     Discrimination ability (0.5=random, 1.0=perfect)
   PR-AUC:      Precision-recall trade-off
   ```

**Output Files:**
```
./data/model_artifacts.joblib              # Trained XGBoost model + calibrator
./results/model_comparison_full.csv        # Performance metrics for all 3 models
./results/roc_curves_full.png              # ROC curves (before/after calibration)
./results/calibration_curves_full_before_after.png  # Calibration plots
```

**Expected Console Output:**
```
Training Logistic Regression...
  ✓ ROC-AUC: 0.768

Training Random Forest...
  ✓ ROC-AUC: 0.802

Training XGBoost...
  ✓ ROC-AUC: 0.825

Applying isotonic calibration...
  ✓ Calibration improved

Model Performance Summary (24-hour horizon):
  Accuracy:   0.751 (75.1%)
  Sensitivity: 0.736 (73.6%)
  Specificity: 0.754 (75.4%)
  ROC-AUC:    0.825

✓ Model evaluation complete!
```

**Expected Performance (24h Horizon):**
| Metric | Value |
|:---|:---|
| Accuracy | 75.1% |
| Sensitivity | 73.6% |
| Specificity | 75.4% |
| ROC-AUC | 0.825 |
| PR-AUC | 0.716 |

**Typical Runtime:** 30-60 minutes

---

### **Step 7: Generate Patient-Level Explanations**

**Command:**
```bash
python 07_patient_explanations.py
```

**Purpose:**
- Create SHAP-based explanations for individual predictions
- Highlight which features drove each patient's prediction
- Show true positive and false negative cases
- Enable clinical interpretability

**What it does:**

1. **Select Representative Cases**:
   - True Positive (TP): Model correctly predicted deterioration
   - False Negative (FN): Model missed deterioration

2. **Generate SHAP Explanations**:
   - Calculate SHAP values (feature contributions)
   - Create waterfall plots showing:
     - Base prediction (model average)
     - Each feature's positive/negative contribution
     - Final prediction

3. **Feature Importance Analysis**:
   - Global: Which features matter most across all predictions?
   - Local: Which features matter for this specific patient?

**Output Files:**
```
./results/shap_feature_importance_full.png          # Top 20 features globally
./results/shap_summary_full.png                     # SHAP summary plot
./results/shap_dependence_lactate_last.png          # Lactate impact on predictions
./results/shap_dependence_gcs_verbal_last.png       # GCS impact on predictions
./results/shap_dependence_sbp_min.png               # Blood pressure impact
./results/shap_dependence_resp_rate_mean.png        # Respiratory rate impact
./results/shap_waterfall_true_positive.png          # Example: Case predicted correctly
./results/shap_waterfall_false_negative.png         # Example: Case missed by model
```

**Expected Console Output:**
```
Loading trained model...
  ✓ Model loaded

Selecting representative cases...
  ✓ True positive case selected: stay_id=12345
  ✓ False negative case selected: stay_id=67890

Calculating SHAP values (this takes a while)...
  ✓ SHAP values computed

Generating explanation plots...
  ✓ Feature importance plot
  ✓ Summary plot
  ✓ Dependence plots (4 key features)
  ✓ Waterfall plots (TP and FN)

✓ Patient explanations complete!
```

**Key Insights from SHAP:**
- **Feature Importance**: Which measurements matter most for predictions?
- **Feature Dependence**: How do feature values affect predictions?
  - Low lactate → Lower risk
  - High lactate → Higher risk
- **Individual Predictions**: Why was THIS patient predicted to deteriorate?
  - Base model prediction: 35% risk
  - High lactate: +15% risk
  - Low GCS: +8% risk
  - Final prediction: 58% risk

**Typical Runtime:** 60-120 minutes (SHAP calculation is computationally intensive)

---

### **Step 8: Prepare for Clinical Application**

**Command:**
```bash
python 08_prepare_app.py
```

**Purpose:**
- Export trained model for deployment
- Package features and imputation parameters
- Create patient monitoring tool
- Prepare for real-time scoring

**What it does:**

1. **Export Model Artifacts**:
   - Trained XGBoost model
   - Isotonic calibration function
   - Feature scaling parameters

2. **Export Features**:
   - Imputed feature matrix (test set)
   - Raw feature matrix (test set)
   - Feature names and data types

3. **Create Patient Monitor**:
   - Rank patients by deterioration risk
   - Show top 20 highest-risk patients
   - Display key contributing features

**Output Files:**
```
./data/app_underlying_model.joblib          # Trained model + calibrator
./data/app_test_features_imputed.parquet    # Features for inference
./data/app_test_features_raw.parquet        # Raw unimputed features
./data/app_patient_monitor.csv              # Ranked patient risk list
```

**Expected Console Output:**
```
Exporting model artifacts...
  ✓ XGBoost model exported
  ✓ Calibration function exported

Exporting features...
  ✓ Imputed features exported (9,397 stays)
  ✓ Raw features exported
  ✓ Feature metadata exported

Creating patient monitor...
  Top 20 highest-risk patients:
    1. Stay 12345: 87% risk (Lactate=5.2, GCS=10)
    2. Stay 67890: 82% risk (Lactate=4.8, SBP=85)
    3. Stay 11111: 79% risk (Lactate=4.2, HR=125)
    ...

✓ App preparation complete!
Ready for clinical deployment.
```

**Typical Runtime:** 5-10 minutes

---

## Output Files & Verification

### File Structure After Complete Pipeline Run

```
icu-deterioration/
├── data/
│   ├── cohort.parquet                      # From step 1
│   ├── vitals_raw.parquet                  # From step 1
│   ├── labs_raw.parquet                    # From step 1
│   ├── demographics.parquet                # From step 1
│   ├── window_config.json                  # From step 1
│   ├── feature_matrix_raw.parquet          # From step 2
│   ├── horizon_labels.parquet              # From step 4
│   ├── model_artifacts.joblib              # From step 6
│   ├── app_underlying_model.joblib         # From step 8
│   ├── app_test_features_imputed.parquet   # From step 8
│   ├── app_test_features_raw.parquet       # From step 8
│   └── app_patient_monitor.csv             # From step 8
│
└── results/
    ├── leakage_audit_report.txt            # From step 3
    ├── model_comparison_full.csv           # From step 6
    ├── horizon_sensitivity_xgboost.csv     # From step 5
    ├── roc_curves_full.png                 # From step 6
    ├── calibration_curves_full_before_after.png  # From step 6
    ├── horizon_sensitivity_roc.png         # From step 5
    ├── shap_feature_importance_full.png    # From step 7
    ├── shap_summary_full.png               # From step 7
    ├── shap_dependence_gcs_verbal_last.png # From step 7
    ├── shap_dependence_lactate_last.png    # From step 7
    ├── shap_dependence_sbp_min.png         # From step 7
    ├── shap_dependence_resp_rate_mean.png  # From step 7
    ├── shap_waterfall_true_positive.png    # From step 7
    └── shap_waterfall_false_negative.png   # From step 7
```

### Verification Checklist

After running the pipeline, verify:

- [ ] **Step 1**: `./data/cohort.parquet` exists (~5000+ rows)
- [ ] **Step 2**: `./data/feature_matrix_raw.parquet` has ~210 columns
- [ ] **Step 3a**: `./results/leakage_audit_report.txt` shows "PASSED"
- [ ] **Step 4**: `./data/horizon_labels.parquet` has 24h/36h/48h labels
- [ ] **Step 3b**: Audit passes for labels
- [ ] **Step 5**: `./results/horizon_sensitivity_roc.png` shows 3 ROC curves
- [ ] **Step 6**: `./data/model_artifacts.joblib` exists (~10-50 MB)
- [ ] **Step 7**: All 8 SHAP PNG files exist in `./results/`
- [ ] **Step 8**: `./data/app_patient_monitor.csv` shows ranked patient risks

---

## Troubleshooting

### Common Issues & Solutions

#### BigQuery Authentication Fails
**Error:** `google.auth.exceptions.DefaultCredentialsError`
**Solutions:**
1. Verify environment variable is set:
   ```bash
   echo $GOOGLE_APPLICATION_CREDENTIALS
   ```
2. Confirm JSON key file exists at that path
3. Check service account has BigQuery Admin permissions
4. Re-authenticate:
   ```bash
   gcloud auth application-default login
   ```

#### MIMIC-IV Data Not Found
**Error:** `FileNotFoundError: No such file or directory: './mimic-iv-3.1/'`
**Solutions:**
1. Confirm you downloaded from PhysioNet
2. Extract to correct path: `./mimic-iv-3.1/`
3. Verify structure:
   ```
   ./mimic-iv-3.1/
   ├── hosp/
   │   ├── patients.csv
   │   ├── admissions.csv
   │   └── diagnoses_icd.csv
   └── icu/
       ├── icustays.csv
       ├── chartevents.csv
       └── labevents.csv
   ```

#### Cohort Empty After Data Extraction
**Error:** Empty parquet files after step 1
**Solutions:**
1. Verify GCP project ID is correct in script
2. Confirm MIMIC-IV tables exist in BigQuery:
   ```bash
   bq ls PROJECT_ID.physionet_data  # Should show MIMIC tables
   ```
3. Check BigQuery query permissions
4. Verify MIMIC-IV data loaded to BigQuery

#### Leakage Audit Fails
**Error:** `AssertionError: Feature leakage detected`
**Solutions:**
1. Re-run step 1 with strict 6-hour window enforcement
2. Check that no outcome data in feature set
3. Review `leakage_audit_report.txt` for details
4. Verify timestamps in raw data

#### Model Training Hangs or Runs Out of Memory
**Error:** Process killed, no memory error, or very slow
**Solutions:**
1. Check available RAM:
   ```bash
   free -h  # Linux/Mac
   # or Task Manager on Windows
   ```
2. Reduce cohort size (for testing):
   - Modify step 1: `LIMIT 5000` instead of full cohort
3. Reduce feature count:
   - Select top 50 features instead of 210
4. Check for GPU XGBoost:
   - If available, XGBoost can use GPU for 10-50x speedup

#### SHAP Calculation Takes Too Long (Step 7)
**Error:** Step 7 running for hours
**Solutions:**
1. This is expected (SHAP is computationally intensive)
2. For faster testing, reduce cohort:
   - Use first 1000 test samples instead of 9,397
3. Consider using `TreeExplainer` (faster) instead of `KernelExplainer`
4. If using GPU, enable SHAP GPU support

#### Missing Output Files
**Error:** Expected PNG/CSV files missing from `./results/`
**Solutions:**
1. Verify each step completed successfully
2. Check for error messages in console output
3. Manually run problematic step:
   ```bash
   python 06_model_evaluation.py  # if missing ROC curves
   ```

---

## Performance Expectations

### Typical Runtimes

| Step | Script | Runtime | Notes |
|:---|:---|:---:|:---|
| 1 | data_extraction.py | 10-30 min | Depends on BigQuery latency |
| 2 | preprocessing.py | 5-10 min | Fast, local processing |
| 3a | leakage_audit.py --feature-only | 2-3 min | Validation only |
| 4 | prediction_horizons.py | 3-5 min | Label engineering |
| 3b | leakage_audit.py --label-only | 2-3 min | Validation only |
| 5 | horizon_sensitivity.py | 20-40 min | 3 model trainings |
| 6 | model_evaluation.py | 30-60 min | Calibration + evaluation |
| 7 | patient_explanations.py | 60-120 min | SHAP is slow but essential |
| 8 | prepare_app.py | 5-10 min | Fast deployment prep |
| **TOTAL** | | **2-4 hours** | For full pipeline |

### System Requirements

| Resource | Minimum | Recommended |
|:---|:---|:---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 100 GB | 200+ GB |
| Internet | Required | Stable connection |
| GPU | Optional | Recommended for step 7 |

---

## Next Steps After Successful Pipeline Run

1. **Review Results**: Examine plots in `./results/`
2. **Validate Performance**: Check model metrics in CSV files
3. **Understand Predictions**: Study SHAP explanations
4. **Plan Deployment**: Prepare clinical validation study
5. **Archive Output**: Backup `./results/` and model files

---

## Support & Additional Resources

- **PhysioNet Docs**: https://physionet.org/content/mimiciv/3.1/
- **BigQuery Guide**: https://cloud.google.com/bigquery/docs
- **SHAP Documentation**: https://shap.readthedocs.io/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/

**Last Updated:** August 2026
**MIMIC-IV Version:** 3.1  
**Python Version:** 3.8+
