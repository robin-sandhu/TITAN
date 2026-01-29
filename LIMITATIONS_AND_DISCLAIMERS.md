# ==============================================================================

# TITAN Limitations and Disclaimers

**Developed by Robin Sandhu**

# Version 1.0.0

# ==============================================================================

## IMPORTANT: READ BEFORE USE

This document outlines critical limitations, disclaimers, and appropriate use cases for TITAN. **Users must read and acknowledge these limitations before using this software for any purpose.**

---

## 1. NOT FOR CLINICAL USE

### ⚠️ CRITICAL DISCLAIMER

**TITAN IS NOT A MEDICAL DEVICE AND IS NOT APPROVED FOR CLINICAL USE.**

- This software is intended for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**
- Do NOT use TITAN outputs to make clinical decisions about patient care
- Do NOT use TITAN as a substitute for professional medical judgment
- Do NOT deploy TITAN models in clinical workflows without proper regulatory approval

### Regulatory Status

- TITAN has NOT been cleared or approved by the FDA (U.S. Food and Drug Administration)
- TITAN has NOT been CE marked for use in the European Union
- TITAN has NOT been approved by Health Canada, TGA, or any other regulatory body
- TITAN does NOT comply with IEC 62304 (Medical Device Software Lifecycle)

---

## 2. Statistical and Methodological Limitations

### 2.1 Sample Size Requirements

- **Minimum recommended:** 500 observations with ≥50 events
- **EPV (Events Per Variable):** Requires EPV ≥ 20 (Riley et al. 2019)
- Small samples may produce unstable, non-generalizable models
- Confidence intervals may be unreliable with <200 test samples

### 2.2 Missing Data

- TITAN uses Multiple Imputation by Chained Equations (MICE)
- Imputation assumes data is Missing At Random (MAR)
- If data is Missing Not At Random (MNAR), results may be biased
- High missingness (>40%) significantly degrades reliability

### 2.3 Model Type

- TITAN uses Random Forest with isotonic calibration
- Random Forest may not capture complex non-linear interactions optimally
- No automatic hyperparameter tuning beyond defaults
- Not suitable for time-to-event (survival) outcomes without modification

### 2.4 Calibration

- Isotonic calibration requires sufficient calibration set size
- Calibration may degrade on populations different from training data
- External validation is essential before any deployment

### 2.5 Feature Selection

- TITAN does not perform automatic feature selection
- Collinear features may inflate importance estimates
- Domain expertise required to select appropriate predictors

### 2.6 Temporal Validation

- TITAN does not enforce temporal train/test splits
- Models may not account for temporal drift
- Prospective validation strongly recommended

---

## 3. Data Quality Limitations

### 3.1 Data Leakage

- TITAN attempts to detect post-event diagnostic variables
- Cannot guarantee detection of all forms of data leakage
- User must verify no target leakage exists
- Improper feature engineering can invalidate results

### 3.2 Selection Bias

- Cannot correct for selection bias in the underlying data
- Results only valid for populations similar to training data
- Case-control sampling requires special handling (not implemented)

### 3.3 Measurement Error

- Assumes input data is accurately measured
- Does not account for inter-observer variability
- Laboratory assay variations not modeled

### 3.4 Label Quality

- Assumes target labels are accurate
- Outcome misclassification degrades performance
- ICD coding errors not detectable

---

## 4. Interpretability Limitations

### 4.1 SHAP Values

- SHAP values are model-specific, not causal
- High SHAP importance ≠ causal relationship
- Correlated features share importance (may underestimate individual effects)
- Computational timeout may limit SHAP to subset of features

### 4.2 Feature Importance

- Random Forest importance can be biased toward high-cardinality features
- Permutation importance adds randomness
- Do not interpret as causal effects

### 4.3 Decision Curve Analysis

- Net benefit estimates have uncertainty not shown in plots
- Assumes representative prevalence
- May not generalize to different clinical settings

---

## 5. Technical Limitations

### 5.1 Computational Resources

- Large datasets (>1M rows) may require significant memory
- SHAP computation is CPU-intensive
- No GPU acceleration implemented

### 5.2 File Formats

- Limited to CSV, TSV, Excel formats
- Does not support FHIR, HL7, or other healthcare standards
- Large files may be automatically sampled

### 5.3 Ontology Detection

- UMLS-based detection requires scispacy installation
- Not all medical concepts are recognized
- False positives/negatives in feature flagging possible

### 5.4 Platform Compatibility

- Tested on Python 3.8-3.11
- Some Unix-specific features (signal timeout) may not work on Windows

---

## 6. Ethical and Legal Considerations

### 6.1 Bias and Fairness

- Models may perpetuate biases present in training data
- Fairness metrics provided but do not guarantee equitable outcomes
- Disparate impact across demographic groups should be assessed
- Protected characteristics may be proxied by other variables

### 6.2 Privacy

- TITAN processes data locally (no cloud transmission)
- User responsible for data de-identification
- Audit logs may contain column names (review before sharing)

### 6.3 Intellectual Property

- User retains ownership of their data and results
- TITAN is provided under permissive license
- Third-party libraries have their own licenses

### 6.4 Liability

- Software provided "AS IS" without warranty
- Authors not liable for any damages from use
- User assumes all risk

---

## 7. Appropriate Use Cases

### ✅ APPROPRIATE

- Exploratory data analysis for research
- Hypothesis generation
- Educational demonstrations
- Preliminary model development
- Benchmarking against other methods
- Internal validation studies

### ❌ NOT APPROPRIATE

- Clinical decision support without regulatory approval
- Automated patient triage
- Treatment selection
- Diagnostic systems
- Any use affecting patient care without human oversight
- Regulatory submissions (without proper validation)

---

## 8. Recommendations Before Use

1. **Consult domain experts** - Statisticians, clinicians, and informaticists
2. **Validate externally** - Test on independent datasets
3. **Assess fairness** - Check for disparate impact
4. **Document limitations** - Be transparent in publications
5. **Seek regulatory guidance** - If considering clinical deployment
6. **Perform prospective validation** - Before any real-world use

---

## 9. Citation

If you use TITAN in published research, please cite:

```
Sandhu, R. (2026). TITAN: A Standardized Framework for Clinical Prediction Model Development.
Software available at: https://github.com/robin-sandhu/TITAN
```

And cite the methodological references in `REFERENCES.md`.

---

## 10. Acknowledgment

By using TITAN, you acknowledge that you have read, understood, and agree to these limitations and disclaimers.

---

**Document Version:** 1.0.0  
**Last Updated:** January 2026  
**Developer:** Robin Sandhu

© 2026 Robin Sandhu. All rights reserved.
