
# TITAN Troubleshooting Guide

**Developed by Robin Sandhu**

# Version 1.0.0

## Table of Contents

1. [Installation Issues](#1-installation-issues)
2. [Runtime Errors](#2-runtime-errors)
3. [Data Issues](#3-data-issues)
4. [Output Issues](#4-output-issues)
5. [Performance Issues](#5-performance-issues)

6. [Platform-Specific Issues](#7-platform-specific-issues)

---

## 1. Installation Issues

### 1.1 Missing Dependencies

**Error:**

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution:**

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install scikit-learn numpy pandas matplotlib seaborn shap
```

---

### 1.2 scispacy Installation Fails

**Error:**

```
ERROR: Could not find a version that satisfies the requirement scispacy
```

**Solution:**

1. Ensure you have Python 3.8+
2. Install in order:

```bash
pip install spacy>=3.0.0
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

**Note:** scispacy is optional. TITAN works without it but won't have medical ontology detection.

---

### 1.3 SHAP Installation Issues

**Error:**

```
ERROR: Failed building wheel for shap
```

**Solution (macOS):**

```bash
xcode-select --install
pip install shap
```

**Solution (Windows):**

1. Install Visual Studio Build Tools
2. Then: `pip install shap`

**Solution (Linux):**

```bash
sudo apt-get install build-essential
pip install shap
```

---

### 1.4 fpdf2 Font Issues

**Error:**

```
FPDF error: TTF font file not found
```

**Solution:**

```bash
pip install --upgrade fpdf2
```

TITAN uses built-in Helvetica font which doesn't require external files.

---

## 2. Runtime Errors

### 2.1 EPV Violation Error

**Error:**

```
ValueError: EPV violation: Only X.X events per variable (requires ≥20)
```

**Cause:** Too few events (positive cases) relative to features.

**Solutions:**

1. **Add more data** - Collect more samples with positive outcomes
2. **Reduce features** - Remove less important predictors
3. **Aggregate features** - Combine related variables
4. **Change threshold** (not recommended for research):
   - Edit `EPV_BLOCK` in TITAN.py (line ~202)

---

### 2.2 SHAP Timeout

**Warning:**

```
SHAP_TIMEOUT - computation exceeded 300s
```

**Cause:** SHAP is computationally expensive for large datasets.

**Solutions:**

1. This is handled gracefully - analysis continues without SHAP
2. To increase timeout, edit `SHAP_TIMEOUT_SECONDS` in TITAN.py
3. Reduce dataset size before analysis
4. Reduce number of features

---

### 2.3 Memory Error

**Error:**

```
MemoryError: Unable to allocate X GiB
```

**Solutions:**

1. **Close other applications**
2. **Use smaller dataset:**
   ```python
   df = df.sample(frac=0.5, random_state=42)
   ```
3. **Increase virtual memory:**
   - Windows: Increase page file size
   - macOS/Linux: Add swap space
4. **Use chunked processing** for very large files (>1GB)

---

### 2.4 Target Detection Fails

**Error:**

```
No suitable binary target column found
```

**Solutions:**

1. **Rename your target column** to include keywords:
   - `outcome`, `death`, `mortality`, `event`, `target`, `label`
2. **Ensure binary encoding:**
   - Values should be 0/1, True/False, Yes/No
3. **Move target to last column**
4. **Use interactive mode** to manually select target

---

### 2.5 Calibration Fails

**Warning:**

```
Calibration failed: not enough samples
```

**Cause:** Insufficient samples in calibration set.

**Solutions:**

1. Increase dataset size (minimum ~200 samples recommended)
2. This warning doesn't stop analysis - model proceeds without isotonic calibration

---

## 3. Data Issues

### 3.1 CSV Encoding Errors

**Error:**

```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**

1. **Convert file to UTF-8:**
   - Open in Excel → Save As → CSV UTF-8
2. **Specify encoding:**
   ```python
   df = pd.read_csv("file.csv", encoding='latin-1')
   ```
3. **Remove special characters** from data

---

### 3.2 Date Column Issues

**Warning:**

```
Could not parse dates in column 'date_of_birth'
```

**Solutions:**

1. Dates are auto-excluded from modeling (intentional)
2. If date is important, convert to numeric:
   - Days since baseline
   - Year extracted
   - Age calculated

---

### 3.3 High Cardinality Categorical

**Warning:**

```
Dropping column 'patient_name' - too many unique values
```

**Cause:** Columns with many unique text values are excluded.

**Solutions:**

1. This is expected behavior for IDs and names
2. If you need the variable, encode it first:
   - Group into categories
   - Use frequency encoding
   - Create binary indicators

---

### 3.4 All Values Missing

**Error:**

```
ValueError: Column 'lab_result' has 100% missing values
```

**Solutions:**

1. Remove the column from your dataset
2. Check data extraction process
3. TITAN automatically drops columns with >95% missing

---

### 3.5 Constant Column

**Warning:**

```
Dropping column 'study_site' - zero variance
```

**Cause:** Column has only one unique value.

**Solutions:**

1. This is expected - constant columns provide no information
2. If unexpected, check data filtering

---

## 4. Output Issues

### 4.1 PDF Not Generated

**Error:**

```
PDF_SKIPPED: fpdf2_not_installed
```

**Solution:**

```bash
pip install fpdf2
```

---

### 4.2 Charts Not Displaying

**Issue:** Charts saved but appear blank

**Solutions:**

1. Check `charts/` folder - files may exist but not open automatically
2. Update matplotlib:
   ```bash
   pip install --upgrade matplotlib
   ```
3. Try different backend:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   ```

---

### 4.3 SHAP Plot Missing

**Issue:** SHAP_Summary_TopFeatures.png not generated

**Cause:** SHAP computation timed out or failed

**Solutions:**

1. Check audit_log.json for SHAP_TIMEOUT or SHAP_FAILED entries
2. Reduce dataset size
3. Run with fewer features

---

### 4.4 Empty Tables

**Issue:** CSV tables are empty or have only headers

**Cause:** Usually insufficient data for that analysis

**Solutions:**

1. Check audit_log.json for specific failure reason
2. Ensure sufficient samples (500+ recommended)
3. Ensure binary target has both classes represented

---

## 5. Performance Issues

### 5.1 Analysis Takes Too Long

**Expected times:**
| Dataset Size | Expected Time |
|--------------|---------------|
| <1,000 rows | <2 minutes |
| 1,000-10,000 rows | 2-10 minutes |
| 10,000-100,000 rows | 10-60 minutes |
| >100,000 rows | May require sampling |

**Solutions to speed up:**

1. Reduce number of features
2. Sample large datasets:
   ```python
   df = df.sample(n=10000, random_state=42)
   ```
3. Skip SHAP (edit TITAN.py to set `SHAP_TIMEOUT_SECONDS = 1`)

---

### 5.2 High CPU Usage

**Cause:** Normal during model training and SHAP computation

**Solutions:**

1. Close unnecessary applications
2. Run overnight for large datasets
3. Consider cloud computing for very large analyses

---

## 7. Platform-Specific Issues

### 7.1 Windows: Signal Timeout Not Working

**Issue:** SHAP timeout doesn't work on Windows

**Cause:** `signal.alarm` is Unix-only

**Impact:** SHAP may run indefinitely on Windows

**Solution:** TITAN handles this gracefully - uses threading-based timeout as fallback

---

## Quick Diagnostic Commands

### Check Python Version

```bash
python --version  # Should be 3.8+
```

### Check Installed Packages

```bash
pip list | grep -E "numpy|pandas|sklearn|shap|matplotlib"
```

### Test Core Import

```bash
python -c "from TITAN import run_infinity_on_file; print('✓ TITAN imports OK')"
```

### Verify SHAP

```bash
python -c "import shap; print(f'SHAP version: {shap.__version__}')"
```

### Check tkinter (for GUI)

```bash
python -c "import tkinter; print('✓ tkinter OK')"
```

---

## Getting Help

If your issue isn't listed:

1. **Check audit_log.json** - Contains detailed error information
2. **Review console output** - Often has specific error messages
3. **Try synthetic data** - Tests if issue is data-specific:
   ```bash
   python TITAN.py
   # Enter: SYNTHETIC
   # Enter: START
   ```

---

**Document Version:** 1.0.0  
**Last Updated:** January 2026

© 2026 Robin Sandhu
