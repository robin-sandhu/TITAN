# ==============================================================================

# TITAN User Manual

**Developed by Robin Sandhu**

## A Standardized Framework for Clinical Prediction Model Development

# Version 1.0.0

# ==============================================================================

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Running TITAN](#running-titan)
5. [Input Data Requirements](#input-data-requirements)
6. [Output Files](#output-files)
7. [Configuration Options](#configuration-options)
8. [Command Line Usage](#command-line-usage)
9. [Interpreting Results](#interpreting-results)

---

## 1. Overview

TITAN is an automated machine learning pipeline designed for biomedical prediction modeling. It provides:

- **Automated target detection** using UMLS medical ontology
- **Robust preprocessing** with intelligent imputation
- **Calibrated Random Forest models** with isotonic calibration
- **Comprehensive validation** including bootstrap confidence intervals
- **SHAP explanations** for model interpretability
- **Decision Curve Analysis** for clinical utility assessment
- **Fairness metrics** for equity auditing
- **Publication-ready outputs** including PDF reports and CSV tables

---

## 2. Quick Start

### Command Line (Recommended)

```bash
python TITAN.py /path/to/your/dataset.csv
```

### Interactive Mode

```bash
python TITAN.py
```

Then enter your CSV file path when prompted.

---

## 3. Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Core Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Medical Ontology (Optional but Recommended)

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, shap, matplotlib; print('✓ Core dependencies OK')"
```

---

## 4. Running TITAN

### Method 1: Interactive Mode (Recommended for First Use)

```bash
python TITAN.py
```

1. TITAN will prompt for data file path(s)
2. Enter your CSV/TSV/Excel file path
3. Type `START` to begin analysis
4. TITAN auto-detects the target variable (you can override if needed)
5. Results are saved to `TITAN_INFINITY_OUTPUT/` folder

### Method 2: Programmatic Usage

```python
from TITAN import run_infinity_on_file
from pathlib import Path

result = run_infinity_on_file(
    filepath=Path("my_data.csv"),
    output_root=Path("output"),
    eda_level="full",
    interactive_target=False,
    session_config=None
)
```

---

## 5. Input Data Requirements

### File Formats Supported

- CSV (.csv)
- TSV (.tsv)
- Excel (.xlsx, .xls)

### Data Requirements

| Requirement     | Details                                   |
| --------------- | ----------------------------------------- |
| Minimum rows    | 100 (recommended: 500+)                   |
| Target variable | Binary outcome (0/1, Yes/No, True/False)  |
| Missing values  | Handled automatically via MICE imputation |
| Column names    | Avoid special characters; underscores OK  |

### Target Variable

TITAN auto-detects likely target columns using:

- Medical ontology terms (mortality, death, outcome, etc.)
- Column position (last column preferred)
- Unique value count (binary preferred)

You can override the auto-detection when prompted.

### Example Data Structure

```
patient_id, age, sex, bp_systolic, creatinine, mortality_30day
001, 65, M, 140, 1.2, 0
002, 72, F, 165, 1.8, 1
...
```

---

## 6. Output Files

### Directory Structure

```
TITAN_INFINITY_OUTPUT/
├── your_dataset_YYYYMMDD_HHMMSS/
│   ├── TITAN_REPORT.pdf          # Comprehensive PDF report
│   ├── TITAN_REPORT.txt          # Text summary
│   ├── TITAN_Compiled_Database.csv # All metrics in one file
│   ├── TABLE_Model_Performance.csv # Publication-ready table
│   ├── Hyperparameters_All.csv   # All model parameters
│   ├── audit_log.json            # Full audit trail
│   │
│   ├── charts/
│   │   ├── ROC_Curve.png
│   │   ├── Calibration_Plot.png
│   │   ├── FeatureImportance.png
│   │   ├── SHAP_Summary_TopFeatures.png
│   │   ├── DecisionCurveAnalysis.png
│   │   ├── PrecisionRecall_Curve.png
│   │   ├── CorrelationHeatmap.png
│   │   ├── NumericFeature_Histograms.png
│   │   ├── ViolinPlots_ByTarget.png
│   │   ├── ConfusionMatrix_*.png
│   │   └── ...
│   │
│   ├── tables/
│   │   ├── Schema_Dictionary.csv
│   │   ├── Descriptive_Statistics_Numeric.csv
│   │   ├── Missingness_By_Variable.csv
│   │   ├── Correlation_Matrix.csv
│   │   ├── Feature_Importance_Rankings.csv
│   │   ├── AUPRC_Summary.csv
│   │   ├── DCA_NetBenefit_Values.csv
│   │   ├── Subgroup_AUCs.csv
│   │   ├── Fairness_Metrics_By_Group.csv
│   │   └── CV_Repeated_Robustness.csv
│   │
│   └── models/
│       └── calibrated_rf_model.joblib
```

### Key Output Files Explained

| File                           | Description                         |
| ------------------------------ | ----------------------------------- |
| `TITAN_REPORT.pdf`             | Complete analysis with all charts   |
| `TABLE_Model_Performance.csv`  | Publication-ready performance table |
| `TITAN_Compiled_Database.csv`  | All metrics aggregated              |
| `ROC_Curve.png`                | Receiver Operating Characteristic   |
| `SHAP_Summary_TopFeatures.png` | Feature importance via SHAP         |
| `DecisionCurveAnalysis.png`    | Clinical decision curve             |
| `audit_log.json`               | Full reproducibility log            |

---

## 7. Configuration Options

### Environment Variables

| Variable          | Default | Description                                    |
| ----------------- | ------- | ---------------------------------------------- |
| `TITAN_ALLOW_PIP` | 0       | Set to 1 to allow auto-install of dependencies |

### Configurable Constants (in TITAN.py)

| Constant            | Default | Description                      |
| ------------------- | ------- | -------------------------------- |
| `RANDOM_STATE`      | 42      | Random seed for reproducibility  |
| `RF_ESTIMATORS`     | 300     | Number of trees in Random Forest |
| `RF_MAX_DEPTH`      | 12      | Maximum tree depth               |
| `EPV_BLOCK`         | 20.0    | Events-per-variable threshold    |
| `PAIRPLOT_SAMPLE_N` | 2000    | Samples for pairplot             |
| `HEATMAP_MAX_COLS`  | 40      | Max columns in heatmap           |

---

## 8. Command Line Usage

### Basic Usage

```bash
python TITAN.py
```

### With Data File

```bash
python TITAN.py /path/to/your/dataset.csv
```

### With External Validation

```bash
python TITAN.py --external=validation_data.csv
```

### Synthetic Data Test

```bash
python TITAN.py
# Then type: SYNTHETIC
# Then type: START
```

---

## 9. Interpreting Results

### AUC (Area Under ROC Curve)

| AUC Range | Interpretation            |
| --------- | ------------------------- |
| 0.90-1.00 | Excellent discrimination  |
| 0.80-0.90 | Good discrimination       |
| 0.70-0.80 | Acceptable discrimination |
| 0.60-0.70 | Poor discrimination       |
| 0.50-0.60 | No discrimination         |

### Calibration Slope

| Value | Interpretation                              |
| ----- | ------------------------------------------- |
| ~1.0  | Perfect calibration                         |
| <1.0  | Overfitting (predictions too extreme)       |
| >1.0  | Underfitting (predictions too conservative) |

### Brier Score

| Value | Interpretation |
| ----- | -------------- |
| 0.00  | Perfect        |
| <0.25 | Good           |
| >0.25 | Poor           |

### Decision Curve Analysis

- Model curve above "Treat All" and "Treat None" = clinical utility
- Higher net benefit = better clinical value
- Threshold range where model is useful indicates clinical applicability

### SHAP Values

- Positive SHAP = feature increases prediction
- Negative SHAP = feature decreases prediction
- Magnitude = importance of feature

---

## Support

For issues, see `TROUBLESHOOTING.md`

For methodology, see `REFERENCES.md`

For limitations, see `LIMITATIONS_AND_DISCLAIMERS.md`

---

© 2026 Robin Sandhu. TITAN is provided for research purposes only.
