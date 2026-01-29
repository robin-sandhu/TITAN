# TITAN

**Developed by Robin Sandhu**

## A Standardized Framework for Clinical Prediction Model Development

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

TITAN is a **human-in-the-loop copilot** for developing well-calibrated clinical prediction models. Unlike AutoML tools that chase AUC, TITAN enforces calibration at every step—because if your model says 30% risk, it should actually mean 30% risk in practice.

**Core principles:**

- Calibration first, discrimination second
- Reproducibility by design (fixed hyperparameters, full audit logs)
- Safety guardrails (EPV checks, leakage detection, fairness auditing)
- No magic—every decision is logged and justified

---

## Quick Start

### Installation

```bash
# Option A: Using Pip (Standard)
python3 -m venv titan_env
source titan_env/bin/activate
pip install -r requirements.txt

# Option B: Using Conda (Recommended for Data Science)
conda env create -f environment.yml
conda activate titan_env

# Verify installation
python3 -c "import TITAN; print('TITAN ready')"
```

### Basic Usage

```python
from TITAN import run_full_audit

results = run_full_audit(
    input_data="your_data.csv",
    target_column="outcome",
    output_dir="./output"
)
```

### Command Line

```bash
python3 TITAN.py /path/to/your/dataset.csv
```

---

## Project Structure

```
TITAN_FINAL_SUBMISSION/
├── TITAN.py                      # Core framework (8,400+ lines)
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment file
├── Dockerfile                    # Container support
│
├── MANUSCRIPT_DRAFT.md           # Academic paper draft
├── CODE_DOCUMENTATION.md         # Technical documentation
├── USER_MANUAL.md                # User guide
├── REPRODUCIBILITY.md            # Reproducibility guide
├── LIMITATIONS_AND_DISCLAIMERS.md # Important limitations
├── TROUBLESHOOTING.md            # Common issues & solutions
├── REFERENCES.md                 # Methodological references
│
├── test_titan.py                 # Unit tests
├── simple_test.csv               # Sample test data
│
└── install_macos.sh              # macOS installation script
```

---

## Key Features

| Feature                         | Description                                    |
| ------------------------------- | ---------------------------------------------- |
| **Random Forest + Calibration** | Isotonic/Platt calibration with nested CV      |
| **MICE Imputation**             | Multiple imputation with MAR mechanism testing |
| **EPV Guardrails**              | Automatic overfitting prevention               |
| **Bootstrap CI**                | Stratified bootstrap for all metrics           |
| **SHAP Explainability**         | Feature importance with timeout protection     |
| **Fairness Auditing**           | Subgroup analysis with disparity metrics       |
| **Decision Curve Analysis**     | Net benefit across threshold probabilities     |
| **Immutable Audit Log**         | HMAC-verified tamper-evident logging           |

---

## Documentation

- **[USER_MANUAL.md](USER_MANUAL.md)** - Getting started guide
- **[CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md)** - Technical reference
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** - Reproduction instructions
- **[LIMITATIONS_AND_DISCLAIMERS.md](LIMITATIONS_AND_DISCLAIMERS.md)** - Important limitations
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues
- **[REFERENCES.md](REFERENCES.md)** - Academic references

---

## ⚠️ Important Disclaimers

**TITAN IS NOT A MEDICAL DEVICE AND IS NOT APPROVED FOR CLINICAL USE.**

- Research and educational purposes only
- External validation required before any deployment
- See LIMITATIONS_AND_DISCLAIMERS.md for full details

---

## Citation

If you use TITAN in your research, please cite:

```
Sandhu, R. (2026). TITAN: A Standardized Framework for Clinical Prediction Model Development.
```

This repository is Zenodo-enabled. For machine-readable citation information, see the [CITATION.cff](CITATION.cff) file.

---

## License

MIT License - See LICENSE file for details.

---

## Contact

**Robin Sandhu**  
Developer and Maintainer
