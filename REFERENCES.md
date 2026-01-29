# ==============================================================================

# TITAN Academic References

**Developed by Robin Sandhu**

# Version 1.0.0

# ==============================================================================

## Methodological Foundations

This document provides citations for the statistical and machine learning methods implemented in TITAN. Users publishing research using TITAN should cite relevant methodological papers.

---

## 1. Model Development and Validation

### Events Per Variable (EPV) Guidelines

> **Riley RD, Snell KI, Ensor J, et al.** (2019). Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes. _Statistics in Medicine_, 38(7):1276-1296.  
> DOI: [10.1002/sim.7992](https://doi.org/10.1002/sim.7992)

**Used in TITAN:** EPV threshold of 20 for model stability

### Sample Size for Prediction Models

> **Riley RD, Ensor J, Snell KIE, et al.** (2020). Calculating the sample size required for developing a clinical prediction model. _BMJ_, 368:m441.  
> DOI: [10.1136/bmj.m441](https://doi.org/10.1136/bmj.m441)

### TRIPOD Reporting Guidelines

> **Collins GS, Reitsma JB, Altman DG, Moons KGM.** (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. _Annals of Internal Medicine_, 162(1):55-63.  
> DOI: [10.7326/M14-0697](https://doi.org/10.7326/M14-0697)

---

## 2. Calibration Assessment

### Calibration Slope and Intercept

> **Van Calster B, McLernon DJ, van Smeden M, Bottolo L, Steyerberg EW.** (2019). Calibration: the Achilles heel of predictive analytics. _BMC Medicine_, 17(1):230.  
> DOI: [10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7)

**Used in TITAN:** Calibration slope/intercept via logistic regression

### Calibration Curves

> **Steyerberg EW, Vickers AJ, Cook NR, et al.** (2010). Assessing the performance of prediction models: a framework for traditional and novel measures. _Epidemiology_, 21(1):128-138.  
> DOI: [10.1097/EDE.0b013e3181c30fb2](https://doi.org/10.1097/EDE.0b013e3181c30fb2)

### Isotonic Calibration

> **Zadrozny B, Elkan C.** (2002). Transforming classifier scores into accurate multiclass probability estimates. _Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, 694-699.  
> DOI: [10.1145/775047.775151](https://doi.org/10.1145/775047.775151)

---

## 3. Decision Curve Analysis

### Original DCA Method

> **Vickers AJ, Elkin EB.** (2006). Decision curve analysis: a novel method for evaluating prediction models. _Medical Decision Making_, 26(6):565-574.  
> DOI: [10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361)

**Used in TITAN:** Net benefit calculation for clinical utility

### DCA Tutorial

> **Vickers AJ, van Calster B, Steyerberg EW.** (2019). Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests. _BMJ_, 352:i6.  
> DOI: [10.1136/bmj.i6](https://doi.org/10.1136/bmj.i6)

### DCA Extensions

> **Vickers AJ, Cronin AM, Elkin EB, Gonen M.** (2008). Extensions to decision curve analysis, a novel method for evaluating diagnostic tests, prediction models and molecular markers. _BMC Medical Informatics and Decision Making_, 8:53.  
> DOI: [10.1186/1472-6947-8-53](https://doi.org/10.1186/1472-6947-8-53)

---

## 4. Model Interpretability (SHAP)

### SHAP Framework

> **Lundberg SM, Lee SI.** (2017). A Unified Approach to Interpreting Model Predictions. _Advances in Neural Information Processing Systems_ (NeurIPS), 30:4765-4774.  
> URL: [https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

**Used in TITAN:** TreeExplainer for Random Forest interpretation

### SHAP for Trees

> **Lundberg SM, Erion G, Chen H, et al.** (2020). From local explanations to global understanding with explainable AI for trees. _Nature Machine Intelligence_, 2:56-67.  
> DOI: [10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)

### Interpretable ML Book (SHAP Chapter)

> **Molnar C.** (2022). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (2nd ed.). Chapter 9: SHAP.  
> URL: [https://christophm.github.io/interpretable-ml-book/shap.html](https://christophm.github.io/interpretable-ml-book/shap.html)

---

## 5. Random Forest

### Original Random Forest

> **Breiman L.** (2001). Random Forests. _Machine Learning_, 45(1):5-32.  
> DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

**Used in TITAN:** Primary classification algorithm

### Random Forest in Medicine

> **Goldstein BA, Polley EC, Briggs FBS.** (2011). Random forests for genetic association studies. _Statistical Applications in Genetics and Molecular Biology_, 10(1):32.  
> DOI: [10.2202/1544-6115.1691](https://doi.org/10.2202/1544-6115.1691)

---

## 6. Performance Metrics

### ROC Analysis

> **Hanley JA, McNeil BJ.** (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. _Radiology_, 143(1):29-36.  
> DOI: [10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)

### DeLong's Test for Comparing AUCs

> **DeLong ER, DeLong DM, Clarke-Pearson DL.** (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. _Biometrics_, 44(3):837-845.  
> DOI: [10.2307/2531595](https://doi.org/10.2307/2531595)

### Brier Score

> **Brier GW.** (1950). Verification of forecasts expressed in terms of probability. _Monthly Weather Review_, 78(1):1-3.  
> DOI: [10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](<https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2>)

### Precision-Recall Curves

> **Saito T, Rehmsmeier M.** (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. _PLoS ONE_, 10(3):e0118432.  
> DOI: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)

---

## 7. Bootstrap Confidence Intervals

### Bootstrap Methods

> **Efron B, Tibshirani RJ.** (1993). An Introduction to the Bootstrap. _Chapman & Hall/CRC_.  
> ISBN: 978-0412042317

**Used in TITAN:** Bootstrap CI for AUC (1000 iterations)

### Bootstrap for Medical Prediction

> **Steyerberg EW, Harrell FE Jr, Borsboom GJ, et al.** (2001). Internal validation of predictive models: efficiency of some procedures for logistic regression analysis. _Journal of Clinical Epidemiology_, 54(8):774-781.  
> DOI: [10.1016/S0895-4356(01)00341-9](<https://doi.org/10.1016/S0895-4356(01)00341-9>)

---

## 8. Missing Data

### Multiple Imputation

> **van Buuren S.** (2018). Flexible Imputation of Missing Data (2nd ed.). _Chapman & Hall/CRC_.  
> DOI: [10.1201/9780429492259](https://doi.org/10.1201/9780429492259)

### MICE Algorithm

> **van Buuren S, Groothuis-Oudshoorn K.** (2011). mice: Multivariate Imputation by Chained Equations in R. _Journal of Statistical Software_, 45(3):1-67.  
> DOI: [10.18637/jss.v045.i03](https://doi.org/10.18637/jss.v045.i03)

**Used in TITAN:** Iterative imputation for missing values

---

## 9. Fairness and Equity

### Fairness in ML

> **Mehrabi N, Morstatter F, Saxena N, Lerman K, Galstyan A.** (2021). A Survey on Bias and Fairness in Machine Learning. _ACM Computing Surveys_, 54(6):115.  
> DOI: [10.1145/3457607](https://doi.org/10.1145/3457607)

### Healthcare Algorithmic Fairness

> **Obermeyer Z, Powers B, Vogeli C, Mullainathan S.** (2019). Dissecting racial bias in an algorithm used to manage the health of populations. _Science_, 366(6464):447-453.  
> DOI: [10.1126/science.aax2342](https://doi.org/10.1126/science.aax2342)

---

## 10. Medical Ontology (UMLS)

### UMLS Overview

> **Bodenreider O.** (2004). The Unified Medical Language System (UMLS): integrating biomedical terminology. _Nucleic Acids Research_, 32(Database issue):D267-D270.  
> DOI: [10.1093/nar/gkh061](https://doi.org/10.1093/nar/gkh061)

**Used in TITAN:** Semantic type detection for medical features

### scispaCy

> **Neumann M, King D, Beltagy I, Ammar W.** (2019). ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing. _Proceedings of BioNLP 2019_, 319-327.  
> DOI: [10.18653/v1/W19-5034](https://doi.org/10.18653/v1/W19-5034)

---

## 11. Cross-Validation

### Repeated K-Fold CV

> **Kohavi R.** (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. _Proceedings of IJCAI_, 14(2):1137-1145.  
> URL: [https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)

### Nested Cross-Validation

> **Varma S, Simon R.** (2006). Bias in error estimation when using cross-validation for model selection. _BMC Bioinformatics_, 7:91.  
> DOI: [10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)

---

## 12. Clinical Prediction Model Development

### Steyerberg's Textbook

> **Steyerberg EW.** (2019). Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating (2nd ed.). _Springer_.  
> DOI: [10.1007/978-3-030-16399-0](https://doi.org/10.1007/978-3-030-16399-0)

### Prediction Model Development Guidelines

> **Moons KGM, Altman DG, Reitsma JB, et al.** (2015). Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): Explanation and Elaboration. _Annals of Internal Medicine_, 162(1):W1-W73.  
> DOI: [10.7326/M14-0698](https://doi.org/10.7326/M14-0698)

---

## 13. Software Dependencies

### scikit-learn

> **Pedregosa F, Varoquaux G, Gramfort A, et al.** (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research_, 12:2825-2830.  
> URL: [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

### NumPy

> **Harris CR, Millman KJ, van der Walt SJ, et al.** (2020). Array programming with NumPy. _Nature_, 585:357-362.  
> DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

### pandas

> **McKinney W.** (2010). Data Structures for Statistical Computing in Python. _Proceedings of the 9th Python in Science Conference_, 56-61.  
> DOI: [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)

### Matplotlib

> **Hunter JD.** (2007). Matplotlib: A 2D Graphics Environment. _Computing in Science & Engineering_, 9(3):90-95.  
> DOI: [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

### Seaborn

> **Waskom ML.** (2021). seaborn: statistical data visualization. _Journal of Open Source Software_, 6(60):3021.  
> DOI: [10.21105/joss.03021](https://doi.org/10.21105/joss.03021)

---

## Citation for TITAN

If you use TITAN in your research, please cite:

```bibtex
@software{sandhu2026titan,
  author = {Sandhu, Robin},
  title = {{TITAN}: A Standardized Framework for Clinical Prediction Model Development},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/REPOSITORY_URL}
}
```

---

**Document Version:** 1.0.0  
**Last Updated:** January 2026

© 2026 Robin Sandhu
