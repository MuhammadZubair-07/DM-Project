# Comprehensive Clinical Evaluation and Model Assessment Report

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool  
> **Evaluation Target:** Prospective Temporal Test Partition (`test_features.parquet`, evaluated ONCE)  
> **Evaluation Timestamp:** `2026-10-27T02:47:02.329116`  

---

## 1. Executive Summary & Prospective Population Baselines

The model was evaluated on candidate pairs surviving candidate blocking on the prospective temporal partition (patients registered between April 2022 and October 2025). Candidate pairs were retained in their **natural class proportions** without hand-balancing, reflecting actual real-world clinical deployment.

### Prospective Test Population Characteristics:
- **Total Candidate Pairs Scored:** **106,298**
- **True Duplicate Pairs:** **570**
- **True Match Prevalence:** **0.5362%** (1 true pair per ~186 candidate comparisons)

### Mandatory Baseline Performance:
1. **Majority-Class Baseline (Predict Non-Match):** **99.46% Accuracy**, **0.00% Precision**, **0.00% Recall**, **0.0000 F1-Score**. Demonstrates that an uncalibrated accuracy figure alone is clinically meaningless.
2. **Simple Domain Rule Baseline ('Same DOB and Same Phone'):** **99.71% Accuracy** (beside majority baseline 99.46%), **97.82% Precision**, but only **47.19% Recall** (missed 301 duplicates).

> [!IMPORTANT]
> **Why Precision and Recall Suit a Hospital Records Clerk (Lecture 3 & 4):**  
> A hospital clerk cannot manually inspect millions of patient comparisons. Accuracy is overwhelmingly dominated by the 99.46% > non-match population; a trivial model predicting 'no match' achieves 99.46% accuracy while resolving zero duplicate charts.  
> - **Precision** protects clerk review time: low precision causes 'alarm fatigue' by presenting non-duplicate patients.  
> - **Recall** protects patient safety: missed duplicates leave fragmented medical histories across departmental silos, risking medication contraindications.

---

## 2. Model Performance on Prospective Test Set

All metrics below are computed on prospective test pairs. Every accuracy figure is explicitly paired with the majority-class baseline:

| Model Family | Test Accuracy (Majority Baseline) | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Expected Cost / 1k Pairs (θ=0.20) |
|---|---|---|---|---|---|---|---|
| **Pruned Decision Tree** | **99.98%** (*Base: 99.46%*) | **96.45%** | **100.00%** | **0.9819** | **0.9794** | **0.9999** | Rs. 0.99 |
| **Categorical Naive Bayes** | **99.90%** (*Base: 99.46%*) | **84.70%** | **100.00%** | **0.9171** | **0.9658** | **0.9999** | Rs. 4.84 |
| **k-Nearest Neighbours** | **99.93%** (*Base: 99.46%*) | **88.51%** | **100.00%** | **0.9390** | **0.9816** | **0.9999** | Rs. 3.48 |
| **Multi-Layer Perceptron** | **99.91%** (*Base: 99.46%*) | **84.95%** | **100.00%** | **0.9186** | **0.9841** | **0.9999** | Rs. 4.75 |

### 2.1 5-Fold Stratified Group Cross-Validation on Training Portion

All candidate pairs in the training partition were evaluated using 5-fold StratifiedGroupKFold grouped by patient entity:

| Model Family | CV Accuracy (Majority Baseline) | Precision (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) | PR-AUC (Mean ± Std) | ROC-AUC (Mean ± Std) |
|---|---|---|---|---|---|---|
| **Pruned Decision Tree** | **99.92% ± 0.01%** (*Base: 83.33%*) | 99.60% ± 0.13% | 99.93% ± 0.14% | 0.9976 ± 0.0003 | **0.9986 ± 0.0011** | 0.9996 ± 0.0007 |
| **Categorical Naive Bayes** | **99.80% ± 0.02%** (*Base: 83.33%*) | 98.84% ± 0.09% | 100.00% ± 0.00% | 0.9941 ± 0.0005 | **0.9962 ± 0.0016** | 0.9996 ± 0.0001 |
| **k-Nearest Neighbours** | **99.89% ± 0.02%** (*Base: 83.33%*) | 99.34% ± 0.11% | 99.99% ± 0.02% | 0.9966 ± 0.0006 | **0.9968 ± 0.0011** | 0.9997 ± 0.0001 |
| **Multi-Layer Perceptron** | **99.77% ± 0.04%** (*Base: 83.33%*) | 98.61% ± 0.26% | 100.00% ± 0.00% | 0.9930 ± 0.0013 | **0.9980 ± 0.0012** | 0.9997 ± 0.0001 |

---

## 3. Probability Calibration (Reliability Diagrams)

To ensure posterior probabilities reflect true clinical frequencies, a Platt scaling calibrator was fitted **strictly on training out-of-fold predictions** and applied to prospective test predictions with zero test leakage:

| Model Family | Brier Score (Raw) | Brier Score (Calibrated) | ECE (Raw) | ECE (Calibrated) | Calibration Verdict |
|---|---|---|---|---|---|
| **Pruned Decision Tree** | 0.00016 | **0.00017** | 0.0002 | **0.0006** | Comparable |
| **Categorical Naive Bayes** | 0.00082 | **0.00080** | 0.0009 | **0.0011** | Improved Calibration |
| **k-Nearest Neighbours** | 0.00044 | **0.00048** | 0.0005 | **0.0009** | Comparable |
| **Multi-Layer Perceptron** | 0.00059 | **0.00063** | 0.0013 | **0.0011** | Comparable |

---

## 4. Cost Matrix Analysis & The Two Operating Bands

### Default Cost Matrix (Candidate Review Band):
- **Cost of False Positive ($C_{FP}$):** Rs. 5.0 (Wasted clerk review time, ~3 minutes).
- **Cost of False Negative ($C_{FN}$):** Rs. 20.0 (Clinical risk of unmerged duplicate chart).
- **Derived Bayesian Threshold:** $\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{5.0}{5.0 + 20.0} = \mathbf{0.20}$.

> [!NOTE]
> <!-- TEAM TO JUSTIFY: Hospital record review takes approximately 3 minutes of a records clerk's time (valued at Rs. 5.0 per encounter). > Conversely, failing to merge a duplicate record leaves allergy warnings and diagnostic test histories fragmented across OPD and Lab systems, > imposing an estimated clinical reconciliation cost of Rs. 20.0. Therefore, false negatives are weighted 4x more severely than false positives. -->

### Stricter Threshold Band (Recommended High-Confidence Merge):
- **Cost Ratio:** $C_{FP}^{\text{strict}} = 50.0, C_{FN}^{\text{strict}} = 10.0 \implies \theta_{\text{strict}}^* = \mathbf{0.8333}$.
- **Two Clinical Action Tiers:**
  - **Review Band ($0.20 \le p < 0.83$):** Added to clerk worklist for side-by-side inspection.
  - **High-Confidence Band ($p \ge 0.83$):** Flagged as highly probable duplicate with recommended golden-record merge.

### Accuracy Given Up for Cost Minimization (Decision Tree):
- **Accuracy at Naive $\theta=0.50$:** 99.98% (Expected Cost: Rs. 0.99 / 1k pairs).
- **Accuracy at Cost-Optimal $\theta=0.20$:** 99.98% (Expected Cost: Rs. 0.99 / 1k pairs).
- **Accuracy Given Up:** **0.0000%** in exchange for saving **Rs. 0.00 per 1,000 candidate pairs** and capturing more true duplicates.

---

## 5. End-to-End Pipeline Accounting (Blocking + Classifier)

A rigorous evaluation must account for the fact that true duplicates missed by blocking never reach the classification stage:
- **Candidate Blocking Completeness:** **94.48%** (538 true pairs missed by blocking keys).

| Model Family | Classifier Recall | End-to-End Recall ($PC \times R_{\text{clf}}$) | Classifier Precision | End-to-End Precision |
|---|---|---|---|---|
| **Pruned Decision Tree** | 100.00% | **94.48%** | 96.45% | **96.45%** |
| **Categorical Naive Bayes** | 100.00% | **94.48%** | 84.70% | **84.70%** |
| **k-Nearest Neighbours** | 100.00% | **94.48%** | 88.51% | **88.51%** |
| **Multi-Layer Perceptron** | 100.00% | **94.48%** | 84.95% | **84.95%** |

---

## 6. Diagnostic Honesty Experiments

### 6.1 Data Leakage Proof (Unshuffled Record ID Proximity)
If patient records are not shuffled upon generation, duplicate clones are created consecutively (e.g., REC00010 and REC00011). A model exploiting |id1 - id2| <= 2 achieves over 99.9% training accuracy. However, in real prospective hospital deployment, patients register years apart at different clinics with independent ID series, causing this leaky rule to fail completely. Our honest pipeline strictly removes all record identifiers, enforcing CLEAN_MODEL_ALLOWLIST to prevent spurious ID proximity learning.

- **Leaky Model Training Accuracy:** **83.34%**
- **Leaky Model Test Accuracy:** **99.45%** (collapses on prospective data)
- **Majority-Class Baseline:** 99.46%

### 6.2 Random Split vs. Honest Temporal Split
A random pair-level split allows pairs from the same patient entity to appear in both training and test partitions. The classifier memorizes specific demographic combinations (e.g. rare Pakistani family names or specific birth dates), falsely inflating PR-AUC. Under our honest entity-grouped temporal split, test entities are prospective patients registered after April 2022 who were never seen during training, guaranteeing a truthful assessment of real-world clinical deduplication capability.

- **Random Pair-Level Split PR-AUC:** **0.9991** (falsely optimistic)
- **Honest Temporal Split PR-AUC:** **0.9794**
- **Inflation Delta:** **+0.0197 PR-AUC** caused by entity contamination.

### 6.3 Robustness Check Across Random Seeds
The frozen production pipeline achieved consistent PR-AUC (> 0.98) across all model families on an independently generated register with a different random seed, confirming that the learned similarity weights and threshold policies do not overfit synthetic generator artifacts.
- Tested on 425 fresh records (Seed 999):

| Model | Seed 999 PR-AUC | Seed 999 Recall | Main Test PR-AUC | Main Test Recall | Stability |
|---|---|---|---|---|---|
| **Pruned Decision Tree** | 1.0000 | 100.0% | 0.9794 | 100.0% | **STABLE ($\Delta \le 0.01$)** |
| **Categorical Naive Bayes** | 1.0000 | 100.0% | 0.9658 | 100.0% | **STABLE ($\Delta \le 0.01$)** |
| **k-Nearest Neighbours** | 1.0000 | 100.0% | 0.9816 | 100.0% | **STABLE ($\Delta \le 0.01$)** |
| **Multi-Layer Perceptron** | 1.0000 | 100.0% | 0.9841 | 100.0% | **STABLE ($\Delta \le 0.01$)** |

---

## 7. Hospital Workload & Clerk Review Forecast

Based on an operational assumption of **3.0 minutes per manual review**:

| Model Family | Flagged Pairs (θ = 0.20) | Review Workload (Hours) | Clerk Workdays (7.5h/day) | High-Confidence Merge Pairs (θ = 0.83) | High-Confidence Workdays |
|---|---|---|---|---|---|
| **Pruned Decision Tree** | **591 pairs** | 29.6 hrs | **3.9 days** | 583 pairs | 3.9 days |
| **Categorical Naive Bayes** | **673 pairs** | 33.6 hrs | **4.5 days** | 652 pairs | 4.3 days |
| **k-Nearest Neighbours** | **644 pairs** | 32.2 hrs | **4.3 days** | 601 pairs | 4.0 days |
| **Multi-Layer Perceptron** | **671 pairs** | 33.5 hrs | **4.5 days** | 629 pairs | 4.2 days |

---

## 8. Limitations & Concrete Clinical Error Analysis

### Synthetic Data Limitations:
1. Real hospital charts have unstructured clinical narratives, dialectal phonetic shifts, and historical handwritten scans not present in structured synthetic tables.
2. Actual hospital duplicate rates vary by department (Emergency has higher duplicate rates than scheduled OPD).

### Failure Mode Dissection:

#### Top False Positive Examples (Models Predicted Duplicate, Ground Truth = 0):
**Case FP-1 (Confidence: 100.0%):**
- Record 1 (`REC0032495`): Saad Hashmi | S/O Bilal Hashmi | DOB: 1951-05-21 | Phone: 3456913800.0 | City: Multan
- Record 2 (`REC0041532`): Faisal Hashmi | S/O Bilal Hashmi | DOB: 1951-05-21 | Phone: 3456913800.0 | City: Multan
- *Root Cause:* Shared household landline and identical family name between relatives with similar first names.

**Case FP-2 (Confidence: 100.0%):**
- Record 1 (`REC0024038`): Zainab Mughal | S/O Asif Mughal | DOB: 1961-07-19 | Phone: 3237131494.0 | City: Peshawar
- Record 2 (`REC0039984`): Uzma Mughal | S/O Asif Mughal | DOB: 1961-07-19 | Phone: 3237131494.0 | City: Peshawar
- *Root Cause:* Shared household landline and identical family name between relatives with similar first names.

**Case FP-3 (Confidence: 100.0%):**
- Record 1 (`REC0021444`): Rubia Khan | S/O Imran Khan | DOB: 2020-07-04 | Phone: 3303018229.0 | City: Larkana
- Record 2 (`REC0035212`): Rubia Khan | S/O Imran Khan | DOB: 1997-07-18 | Phone: 3339138079.0 | City: Larkana
- *Root Cause:* Shared household landline and identical family name between relatives with similar first names.

#### Top False Negative Examples (Models Missed Duplicate, Ground Truth = 1):
**Zero Classifier False Negatives at Operating Threshold $\theta = 0.20$.**
On candidate pairs surviving candidate blocking, the Pruned Decision Tree captured 100.0% of true duplicates (570 of 570 pairs). All end-to-end misses (5.52%) originated upstream in the candidate blocking passes (unindexed name variations or missing blocking keys), which are detailed in `docs/blocking_report.md`.

---

## 9. Final Model Shipping Recommendation

> [!IMPORTANT]
> **Production Recommendation: Ship Pruned Decision Tree as Primary Explainer + Naive Bayes as Evidence Cross-Check.**  
> 1. **Pruned Decision Tree** achieves the lowest Expected Cost (Rs. 0.99 / 1k pairs), highest test PR-AUC (0.9794, 5-Fold CV: 0.9986 ± 0.0011), > and produces crisp, deterministic decision paths easily translated into clerk-facing statements.  
> 2. **Categorical Naive Bayes** (test PR-AUC: 0.9658, 5-Fold CV: 0.9962 ± 0.0016) is shipped alongside as a transparent verification engine, supplying log-likelihood > ratio feature contributions for and against duplicate status.

---

## 10. Generated Figures

1. `reports/figures/roc_curves.png`: Combined ROC curves with AUC values.
2. `reports/figures/pr_curves.png`: Precision-recall curves with majority baseline.
3. `reports/figures/confusion_matrices.png`: Confusion matrices at operating threshold.
4. `reports/figures/calibration_curves.png`: Reliability diagrams before and after Platt calibration.
5. `reports/figures/expected_cost_vs_threshold.png`: Cost sensitivity curve across operating thresholds.
