# ==============================================================================

# TITAN Reproducibility Guide

**Developed by Robin Sandhu**

# Version 1.0.0

# ==============================================================================

## Overview

This document provides step-by-step instructions for reproducing TITAN analyses
and validating results. TITAN is designed with reproducibility as a core principle.

---

## 1. Environment Setup

### 1.1 System Requirements

- **Python**: 3.9, 3.10, or 3.11 (recommended: 3.10)
- **Operating System**: macOS, Linux, or Windows 10/11
- **RAM**: Minimum 8GB, recommended 16GB for large datasets
- **Disk**: 2GB free space for dependencies

### 1.2 Installation

```bash
# Create virtual environment (recommended)
python3 -m venv titan_env
source titan_env/bin/activate  # Linux/macOS
# titan_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install medical NER support
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### 1.3 Verify Installation

```bash
python3 -c "import TITAN; print('TITAN loaded successfully')"
```

---

## 2. Reproducibility Features

### 2.1 Random State Control

TITAN uses a fixed random state (`RANDOM_STATE = 42`) throughout:

- All bootstrap sampling
- Cross-validation fold generation
- Model training
- SHAP sampling

To reproduce results exactly, ensure:

1. Same Python version
2. Same package versions
3. Same input data (verified via SHA-256 hash)

### 2.2 Audit Trail

Every TITAN run produces an immutable JSON audit log containing:

- All parameters used
- Data characteristics (shape, types, missingness)
- Preprocessing decisions
- Model configuration
- Performance metrics
- Timestamps and version information

Location: `<output_dir>/audit_log.json`

### 2.3 Verification Keys

Each session generates a unique verification key:

- 16-character hex display key for session identification
- HMAC-SHA256 integrity hash for tamper detection
- Salt stored for verification

---

## 3. Step-by-Step Reproduction

### 3.1 Running TITAN

```python
from TITAN import run_full_audit

# Run analysis
results = run_full_audit(
    input_data="your_data.csv",
    target_column="outcome",
    output_dir="./TITAN_OUTPUT"
)
```

### 3.2 Verifying Results

1. **Check audit log**: Compare `audit_log.json` files
2. **Compare metrics**: Performance metrics should match exactly
3. **Verify data hash**: Input data SHA-256 should match
4. **Check versions**: Package versions should match

### 3.3 External Validation

For external validation on new data:

```python
from TITAN import load_model, evaluate_on_external_validation

# Load trained model
model = load_model("./TITAN_OUTPUT/TITAN_model.pkl")

# Evaluate on external data
ext_results = evaluate_on_external_validation(
    model_path="./TITAN_OUTPUT/TITAN_model.pkl",
    external_data_path="external_cohort.csv",
    target_column="outcome"
)
```

---

## 4. Known Sources of Non-Reproducibility

### 4.1 Floating Point Precision

Different CPU architectures may produce slightly different floating-point results.
Differences should be < 1e-10 and not affect conclusions.

### 4.2 Package Version Changes

Major version changes in scikit-learn or other packages may affect results.
Always record package versions from `pip freeze`.

### 4.3 Operating System Differences

Signal-based SHAP timeouts only work on Unix systems.
Windows uses graceful degradation (no timeout).

---

## 5. Docker Containerization

For maximum reproducibility, use the provided Dockerfile:

```bash
# Build container
docker build -t titan:latest .

# Run analysis
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output titan:latest \
    python3 -c "from TITAN import run_full_audit; run_full_audit('/data/input.csv', 'outcome', '/output')"
```

---

## 6. Reporting Reproducibility

When publishing results from TITAN, report:

1. **TITAN version**: Found in `audit_log.json`
2. **Python version**: `python3 --version`
3. **Key package versions**: scikit-learn, pandas, numpy, shap
4. **Random state**: Default is 42 unless modified
5. **Data characteristics**: Shape, missingness, class balance
6. **Verification key**: For audit trail linking

---

## 7. Troubleshooting

### Results differ between runs

1. Check Python version matches
2. Check package versions match
3. Verify input data hash matches
4. Ensure no data preprocessing outside TITAN

### Results differ between machines

1. Check CPU architecture (Intel vs ARM)
2. Check operating system
3. Use Docker for guaranteed reproducibility

---

## Contact

For reproducibility issues, please include:

- TITAN version
- Full `audit_log.json`
- Package versions (`pip freeze > versions.txt`)
- Input data characteristics (or data if shareable)
