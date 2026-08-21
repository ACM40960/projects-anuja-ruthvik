# ICU Deterioration Prediction: Technical Implementation Guide

A comprehensive guide to the technical architecture, algorithms, and implementation details of the ICU patient deterioration prediction system.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Feature Engineering](#feature-engineering)
3. [Model Development](#model-development)
4. [Critical Temporal Boundaries](#critical-temporal-boundaries)
5. [Architecture Summary](#architecture-summary)


---

## System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ MIMIC-IV Data (PhysioNet BigQuery)                          │
│ - Patients, admissions, ICU stays                           │
│ - Vital signs (chartevents)                                 │
│ - Lab results (labevents)                                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ 01: Data Extraction  │ ◄── 6-hour observation window
        │ (BigQuery queries)   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ 02: Preprocessing    │ ◄── Outlier clipping
        │ & Aggregation        │     Feature creation
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ 03a: Feature Audit   │ ◄── Temporal validation
        │ (Leakage detection)  │
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ 04: Label Engineering│ ◄── Outcome definition
        │ (24/36/48h horizons) │     Multiple events
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ 03b: Label Audit     │ ◄── Integrity validation
        │ (Leakage detection)  │
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ 05: Horizon Sensitivity│ ◄── Compare 24/36/48h
        │ Analysis (XGBoost)    │
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ 06: Model Development │ ◄── XGBoost, calibration
        │ & Evaluation          │     Isotonic regression
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ 07: Explainability    │ ◄── SHAP values
        │ (Patient-level)       │     Feature importance
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │ 08: Deployment Prep   │ ◄── Model packaging
        │ & Application         │     API readiness
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Clinical Application │
        │ (Decision Support)    │
        └──────────────────────┘
```

### Data Flow

```
Raw MIMIC-IV Data
       ↓
   [Step 1: Extract]
       ↓
  Raw Features (0-6h)
       ↓
   [Step 2: Preprocess]
       ↓
  Preprocessed Features (210 features)
       ↓
   [Step 3a: Feature Audit]
       ↓
  Validated Features (no leakage)
       ↓
           ┌────────────────────┐
           │ Feature Matrix     │
           │ 46,982 × 210       │
           └────────┬───────────┘
                    │
                    ├─→ [Step 4: Label Engineering]
                    │          ↓
                    │   Outcome Events
                    │   (Mortality, Vasopressor, Vent)
                    │          ↓
                    │   [Step 3b: Label Audit]
                    │          ↓
                    │   Validated Labels
                    │
                    ├─→ [Step 5: Horizon Sensitivity]
                    │   [Step 6: Model Development]
                    │          ↓
                    │   Trained Model + Calibrator
                    │
                    └─→ [Step 7: Explainability]
                        [Step 8: Deployment]
                             ↓
                        Production Model
```

---

## Feature Engineering

### Feature Categories

The system extracts and aggregates **~210 features** across multiple categories:

#### 1. Vital Signs (from chartevents)
```
Features extracted: 5-8 aggregations × 6 vital sign types

Vital Sign Types:
├─ Heart Rate (HR)
├─ Systolic Blood Pressure (SBP)
├─ Diastolic Blood Pressure (DBP)
├─ Mean Arterial Pressure (MAP)
├─ Respiratory Rate (RR)
├─ Temperature (Temp)
├─ Oxygen Saturation (SpO2)
└─ Glasgow Coma Scale (GCS)

Aggregations (per vital sign):
├─ Mean:   average value over 6 hours
├─ Min:    minimum value observed
├─ Max:    maximum value observed
├─ Last:   most recent measurement
├─ Count:  number of measurements (missingness)
├─ StdDev: standard deviation (variability)
└─ Slope:  trend over time
```

**Total vital features**: ~50-60

#### 2. Laboratory Values (from labevents)
```
Lab Types:
├─ Lactate          (metabolic distress marker)
├─ White Blood Cells (WBC) (infection marker)
├─ Creatinine       (kidney function)
├─ Platelets        (clotting marker)
├─ Hemoglobin       (anemia)
├─ Potassium (K)    (electrolyte)
├─ Sodium (Na)      (electrolyte)
├─ Chloride (Cl)    (electrolyte)
├─ Bicarbonate      (acid-base status)
├─ pH               (acid-base status)
├─ Glucose          (metabolic status)
├─ Albumin          (nutritional status)
└─ Bilirubin        (liver function)

Aggregations (per lab):
├─ Mean
├─ Min
├─ Max
├─ Last
└─ Count (# of measurements)
```

**Total lab features**: ~100-120

#### 3. Derived Features
```
Physiologic Ratios:
├─ Shock Index = Heart Rate / Systolic BP
│  (hemodynamic distress indicator)
├─ SOFA components (partial):
│  ├─ Respiratory: SpO2/FiO2 ratio (if ventilated)
│  ├─ Renal: Creatinine level
│  └─ Hematologic: Platelet count
└─ Base Excess (calculated from pH, HCO3)

Missingness Indicators:
├─ Has_HR (1 if HR measured, 0 if missing)
├─ Has_Lactate (1 if lactate measured, 0 if missing)
├─ Has_WBC (1 if WBC measured, 0 if missing)
├─ ... (binary flag for each feature type)
└─ Missing_Features_Count (total # missing)

Temporal Features:
├─ Hours_Since_Admission (0-6 hours)
├─ Measurement_Frequency (measurements per hour)
└─ Data_Completeness (% of expected data)
```

**Total derived features**: ~20-30

#### 4. Demographics & Admission Data
```
Static Patient Characteristics:
├─ Age (years)
├─ Gender (M/F)
├─ Ethnicity (one-hot encoded)
├─ BMI (if available)
└─ Comorbidities (selected ICD codes)

Admission Characteristics:
├─ Admission Type (Emergency, Urgent, Planned)
├─ Care Unit (MICU, SICU, CCU, etc.) - one-hot
├─ Primary Diagnosis (ICD code)
├─ Severity Score (APACHE estimated from early data)
└─ Prior Hospital Visits (count)
```

**Total demographic features**: ~20-30

### Feature Aggregation Strategy

All raw measurements within the 6-hour window are aggregated as follows:

```python
# Pseudocode for feature aggregation
def aggregate_features(measurements_df, feature_name):
    """
    Input: DataFrame with:
      - stay_id: ICU stay identifier
      - charttime: measurement timestamp
      - value: measured value
      - intime: ICU admission time
    
    Output: Dict of aggregated features
    """
    # Calculate time offsets from ICU admission
    offsets = (charttime - intime).total_seconds() / 3600  # Convert to hours
    
    # Verify 6-hour window (0 ≤ offset ≤ 6)
    assert (offsets >= 0).all() and (offsets <= 6).all()
    
    # Aggregate to per-stay statistics
    aggregated = {
        f'{feature_name}_mean': values.mean(),
        f'{feature_name}_min': values.min(),
        f'{feature_name}_max': values.max(),
        f'{feature_name}_last': values.iloc[-1],  # Most recent
        f'{feature_name}_count': values.count(),
        f'{feature_name}_stddev': values.std(),
    }
    
    return aggregated
```

### Outlier Clipping

Physiologically implausible values are clipped (winsorized) to defined ranges:

```python
# Outlier clipping ranges
outlier_ranges = {
    'Heart_Rate': (20, 200),           # bpm
    'SBP': (40, 300),                  # mmHg
    'DBP': (20, 200),                  # mmHg
    'MAP': (30, 250),                  # mmHg
    'RespRate': (5, 60),               # breaths/min
    'Temperature': (32, 43),           # Celsius
    'SpO2': (50, 100),                 # %
    'Lactate': (0, 10),                # mmol/L
    'WBC': (0.1, 30),                  # K/µL
    'Creatinine': (0.1, 8),            # mg/dL
    'Glucose': (40, 500),              # mg/dL
    'pH': (6.8, 7.8),                  # Units
}

# Clipping implementation
for feature, (lower, upper) in outlier_ranges.items():
    data[feature] = data[feature].clip(lower=lower, upper=upper)
```

### Missing Value Handling

**Strategy**: Multi-component approach
1. **During preprocessing**: Compute missingness indicators
2. **During training imputation**: Use median values (training set only)
3. **During prediction**: Standardized imputation approach

```python
# Step 1: Identify missing features
missing_indicators = {}
for feature in feature_list:
    missing_indicators[f'{feature}_missing'] = (data[feature].isna()).astype(int)

# Step 2: Train set: Compute median for each feature
train_medians = X_train.median()

# Step 3: Imputation strategy
# For value features (vitals, labs):
X_train[feature] = X_train[feature].fillna(train_medians[feature])
X_test[feature] = X_test[feature].fillna(train_medians[feature])

# For count features (measurement counts):
X_train[f'{feature}_count'] = X_train[f'{feature}_count'].fillna(0)
X_test[f'{feature}_count'] = X_test[f'{feature}_count'].fillna(0)

# For boolean features (has measurement):
X_train[f'{feature}_missing'] = X_train[f'{feature}_missing'].fillna(0)
X_test[f'{feature}_missing'] = X_test[f'{feature}_missing'].fillna(0)
```

---

## Model Development

### Model Selection & Architecture

#### 1. Logistic Regression (Baseline)
- **Purpose**: Interpretable baseline model
- **Pros**: Interpretable, fast, probabilistic outputs
- **Cons**: Assumes linear relationships
- **Use**: Performance benchmark

#### 2. Random Forest (Baseline)
- **Purpose**: Ensemble baseline, non-linear
- **Pros**: Handles non-linearity, feature importance
- **Cons**: Slower inference, less calibrated
- **Use**: Performance benchmark

#### 3. XGBoost (Primary)
- **Purpose**: Production model
- **Pros**: Best performance, fast inference, feature importance
- **Cons**: Requires hyperparameter tuning
- **Use**: Main prediction model


---

### Critical Temporal Boundaries

```
ICU Admission (time = 0h)
    │
    ├─ [0-6 hours]: OBSERVATION WINDOW
    │  └─ All features extracted from this period only
    │
    │
    ├─ [6 hours]: FEATURE CUTOFF
    │  └─ No features beyond this point
    │  └─ 6-hour gap prevents information leakage
    │
    │
    ├─ [6+ hours]: PREDICTION WINDOW START
    │
    ├──────────────────┤ 24-hour prediction window ├─────────────────┤
    6h                                             30h
    
    ├─────────────────────────┤ 36-hour prediction window ├─────────────────┤
    6h                                                       42h
    
    ├─────────────────────────────────────┤ 48-hour prediction window ├─────┤
    6h                                                                   54h
```
---

## Architecture Summary

```
┌──────────────────────────────────────────────────┐
│              PRODUCTION SYSTEM                   │
├──────────────────────────────────────────────────┤
│                                                  │
│  Input: Patient data (0-6h window)               │
│    - Vital signs                                │
│    - Lab results                                │
│    - Demographics                               │
│         │                                        │
│         ▼                                        │
│  ┌──────────────────────────┐                   │
│  │  Feature Engineering     │                   │
│  │  - Aggregation           │                   │
│  │  - Outlier clipping      │                   │
│  │  - Missing indicators    │                   │
│  │  (~210 features)         │                   │
│  └──────────────────────────┘                   │
│         │                                        │
│         ▼                                        │
│  ┌──────────────────────────┐                   │
│  │  Preprocessing           │                   │
│  │  - Imputation            │                   │
│  │  - Standardization       │                   │
│  └──────────────────────────┘                   │
│         │                                        │
│         ▼                                        │
│  ┌──────────────────────────┐                   │
│  │ XGBoost Model            │                   │
│  │  - 400 trees            │                   │
│  │  - max_depth=4          │                   │
│  │  - Raw probability      │                   │
│  └──────────────────────────┘                   │
│         │                                        │
│         ▼                                        │
│  ┌──────────────────────────┐                   │
│  │ Isotonic Calibration     │                   │
│  │  - Post-hoc adjustment   │                   │
│  │  - Calibrated probability│                   │
│  └──────────────────────────┘                   │
│         │                                        │
│         ▼                                        │
│  Output: Deterioration Risk Score (0-100%)     │
│         + Feature Explanations (SHAP)           │
│         + Confidence Metrics                    │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

**Last Updated:** August 2024  
**MIMIC-IV Version:** 3.1  
**Python Version:** 3.8+  
**Framework:** XGBoost 1.7+, SHAP 0.41+, scikit-learn 1.0+
