# Explainable AI for Early ICU Deterioration Prediction 


A machine learning pipeline for predicting ICU patient deterioration using clinical data from MIMIC-IV. This system uses a 6-hour observation window to predict deterioration events (mortality, vasopressor requirement, mechanical ventilation) across multiple prediction horizons.

---

## ⚠️ IMPORTANT: PhysioNet Data Confidentiality Notice

**This GitHub repository does NOT contain the MIMIC-IV dataset.** Due to strict confidentiality and data protection policies enforced by PhysioNet, **we cannot and will not provide the dataset in any format** (raw files, parquet, CSV, or any other form).

### Dataset Access Requirements

To use this pipeline, you MUST:

1. **Register at [PhysioNet](https://physionet.org/register/)** with institutional affiliation
2. **Request access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)**
3. **Complete credentialing** (typically 24-48 hours)
4. **Accept the MIMIC-IV Data Use Agreement**
5. **Download the data directly from PhysioNet** following their instructions

### What Can Be Shared

This repository contains:
- ✅ All Python scripts for data processing, modeling, and evaluation
- ✅ Model architecture and hyperparameter specifications
- ✅ Analysis code and visualization scripts
- ✅ Reproducibility guidance and documentation
- ✅ Results, plots, and performance metrics (in `results/` folder)

### What Cannot Be Shared

This repository does NOT contain:
- ❌ MIMIC-IV raw data files (patients.csv, admissions.csv, chartevents.csv, etc.)
- ❌ Processed patient data or feature matrices
- ❌ Any identifiable or de-identified patient records
- ❌ Any derived datasets containing patient information

**If you attempt to share this repository with MIMIC-IV data, you violate PhysioNet's data use agreement and federal HIPAA regulations.**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Setup & Prerequisites](#setup--prerequisites)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Execution Guide](#execution-guide)
7. [Clinical Disclaimer](#clinical-disclaimer)
8. [Support](#support)

---

## Quick Start

```bash
# 1. Clone and setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

# 3. Download MIMIC-IV from PhysioNet (REQUIRED)
# See "Setup & Prerequisites" section below

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

---

## Overview

### Objective
Predict ICU patient deterioration within 24, 36, and 48 hours using only data from the first 6 hours of ICU admission. This enables early clinical intervention and resource allocation.

### Key Features
- **Rigorous Temporal Design**: 6-hour observation window → 24-48 hour prediction windows
- **Multiple Deterioration Events**: Mortality, vasopressor initiation, mechanical ventilation
- **Leakage Prevention**: Comprehensive audit framework to prevent information leakage
- **Explainability**: SHAP-based patient-level explanations for clinical interpretability
- **Model Calibration**: Isotonic regression for probability calibration

### Data Source
- **Dataset**: MIMIC-IV v3.1 (PhysioNet)
- **Cohort**: First ICU stay per patient with ≥12 hours length of stay
- **Features**: Vital signs, laboratory values, demographics
- **Access**: Requires PhysioNet credentialing and direct download from PhysioNet

---

## Setup & Prerequisites

### Prerequisites
- Python 3.8+
- Google Cloud Platform (GCP) account with BigQuery access
- **PhysioNet account with MIMIC-IV v3.1 access** (REQUIRED)
- 16GB+ RAM recommended

### **CRITICAL: Obtaining MIMIC-IV Data**

**You must obtain the dataset directly from PhysioNet. We cannot provide it.**

#### Step 1: Create PhysioNet Account
1. Visit [PhysioNet Registration](https://physionet.org/register/)
2. Complete registration with institutional affiliation
3. Verify your email address

#### Step 2: Request MIMIC-IV Access
1. Go to [MIMIC-IV v3.1 Project Page](https://physionet.org/content/mimiciv/3.1/)
2. Click "Request Access"
3. Complete the credentialing questionnaire:
   - Describe your intended research use
   - Specify your institution
   - Confirm HIPAA training completion
4. **Wait for approval** (typically 24-48 hours)
5. Accept the data use agreement

#### Step 3: Download MIMIC-IV Files
After approval:
1. Download the following directories from PhysioNet:
   - `hosp/` directory (contains: patients.csv, admissions.csv, diagnoses_icd.csv)
   - `icu/` directory (contains: icustays.csv, chartevents.csv, labevents.csv, d_items.csv)
2. Extract to `./mimic-iv-3.1/` in your project directory
3. Verify file structure:
   ```
   ./mimic-iv-3.1/
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

### Environment Setup

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

# Create data directory
mkdir -p ./data ./results
```

### GCP BigQuery Configuration

1. Create a GCP project
2. Enable BigQuery API in your GCP console
3. Create a service account with BigQuery Admin permissions
4. Download the service account JSON key file
5. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
   ```

### Update Configuration

Edit each Python script and set your GCP project ID:
```python
PROJECT_ID = "your-gcp-project-id"  # Replace with your actual GCP project ID
```

---

## Pipeline Architecture

```
Phase 1: Feature Engineering
├─ 01_data_extraction.py
│  └─ Extract cohort, vitals, labs, demographics (0-6h window)
├─ 02_preprocessing.py
│  └─ Validate, clip outliers, aggregate to per-stay statistics
└─ 03_leakage_audit.py
   └─ Audit for temporal integrity and data leakage

Phase 2: Label Engineering
└─ 04_prediction_horizons.py
   └─ Define deterioration events across 24h/36h/48h horizons

Phase 3: Model Development
├─ 05_horizon_sensitivity.py
│  └─ Compare model performance across prediction horizons
└─ 06_model_evaluation.py
   └─ Train, calibrate, and evaluate primary models

Phase 4: Explainability & Deployment
├─ 07_patient_explanations.py
│  └─ Generate SHAP-based patient-level explanations
└─ 08_prepare_app.py
   └─ Package model and features for clinical application
```

---

## Execution Guide

For detailed step-by-step execution instructions, see **[EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md)**.

Quick reference:

| Step | Script | Purpose | Output |
|:---|:---|:---|:---|
| 1 | `01_data_extraction.py` | Extract features from BigQuery (0-6h window) | `cohort.parquet`, `vitals_raw.parquet`, `labs_raw.parquet` |
| 2 | `02_preprocessing.py` | Validate, clean, and aggregate features | `feature_matrix_raw.parquet` |
| 3 | `03_leakage_audit.py --feature-only` | Audit features for data leakage | `leakage_audit_report.txt` |
| 4 | `04_prediction_horizons.py` | Define outcomes for 24h/36h/48h horizons | `horizon_labels.parquet` |
| 5 | `03_leakage_audit.py --label-only` | Audit labels for integrity | Verification only |
| 6 | `05_horizon_sensitivity.py` | Test sensitivity across prediction horizons | Sensitivity plots |
| 7 | `06_model_evaluation.py` | Train and calibrate final model | `model_artifacts.joblib` |
| 8 | `07_patient_explanations.py` | Generate SHAP explanations | SHAP plots |
| 9 | `08_prepare_app.py` | Package for deployment | App-ready model |

**Full pipeline runtime:** 2-4 hours

---

## Technical Implementation

For detailed technical documentation, see **[TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md)**.

Key topics covered:
- System architecture and data flow
- Feature engineering and aggregation
- Model selection and hyperparameters
- Temporal design and leakage prevention
- ML implementation details
- Explainability and SHAP methodology
- Code examples

---

## Key Findings

For detailed key findings documentation, see **[KEY_FINDINGS.md](docs/KEY_FINDINGS.md)**.

Key topics covered:
- Model Performance
- Cohort Statistics
- Model Evaluation & Calibration
- Performance Evaluation Metrics
- Global Model Explanations
- Feature-Specific Dependence Analysis
- Clinical Implications
- Methodological Strengths

---

## File Structure

```
icu-deterioration/
├── README.md (this file)
├── docs/
│   ├── EXECUTION_GUIDE.md
│   ├── TECHNICAL_GUIDE.md
│   └── (additional documentation)
├── requirements.txt
├── 01_data_extraction.py
├── 02_preprocessing.py
├── 03_leakage_audit.py
├── 04_prediction_horizons.py
├── 05_horizon_sensitivity.py
├── 06_model_evaluation.py
├── 07_patient_explanations.py
├── 08_prepare_app.py
│
├── data/                    (generated after running pipeline)
│   ├── cohort.parquet
│   ├── feature_matrix_raw.parquet
│   ├── horizon_labels.parquet
│   ├── model_artifacts.joblib
│   └── app_*.parquet, app_*.joblib
│
├── results/                 (generated outputs, plots, metrics)
│   ├── leakage_audit_report.txt
│   ├── model_comparison_full.csv
│   ├── roc_curves_full.png
│   ├── calibration_curves_full_before_after.png
│   ├── horizon_sensitivity_roc.png
│   ├── shap_feature_importance_full.png
│   ├── shap_summary_full.png
│   ├── shap_dependence_*.png
│   ├── shap_waterfall_*.png
│   └── (additional analysis outputs)
│
└── mimic-iv-3.1/            (downloaded from PhysioNet - NOT included)
    ├── hosp/
    └── icu/
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

## Support

For questions and troubleshooting:
- Review inline code comments in each script
- Check [EXECUTION_GUIDE.md](docs/EXECUTION_GUIDE.md) for common issues and step-by-step instructions
- See [TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) for implementation details
- Consult [PhysioNet documentation](https://physionet.org/content/mimiciv/3.1/) for data questions

---

## Reproducibility & Citation

### To Reproduce Results

```bash
# 1. Obtain MIMIC-IV access from PhysioNet (REQUIRED)
# Visit https://physionet.org/content/mimiciv/3.1/

# 2. Download MIMIC-IV and extract to ./mimic-iv-3.1/

# 3. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
# Edit scripts to set PROJECT_ID

# 5. Run pipeline
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
  title={Explainable AI for Early ICU Deterioration Prediction},
  author={Anuja Thuraiyur Jayakumar (25203657), Ruthvik Gowda Bageri Manjunath (25205410)},
  year={2024},
  note={Implemented using MIMIC-IV v3.1, requires PhysioNet access}
}
```

---

## Data Use Compliance

**MIMIC-IV Data Use Agreement Requirements**

This code is provided to facilitate research using MIMIC-IV. Users of this code MUST:

1. ✅ Obtain PhysioNet credentials and MIMIC-IV access approval
2. ✅ Accept the MIMIC-IV Data Use Agreement
3. ✅ Use the dataset only for approved research purposes
4. ✅ NOT share de-identified or raw patient data
5. ✅ NOT include patient data in GitHub repositories
6. ✅ Comply with all HIPAA regulations
7. ✅ Acknowledge PhysioNet in publications

**Violation of these requirements may result in loss of data access and legal consequences.**

---

**Last Updated:** August 2026  
**MIMIC-IV Version:** 3.1  
**Status:** Research Implementation Complete

---

**End of README**
