# Preprocessing Decisions and Experimental Evaluation Report

> **Course:** DSC306 Data Mining — COMSATS University Islamabad
> **Project:** Project 11 — Patient Record Deduplication Tool
> **Topic:** Rigorous Feature Preprocessing, Selection & Scaling (Sections 7.7 & 7.8)
> **Validation Protocol:** 5-Fold StratifiedGroupKFold (grouped by patient entity, random_state=42)

---

## 1. Executive Summary & Pre-Registered Decisions

In accordance with Lectures 3 and 4, preprocessing transformations determine model inductive biases, nearest-neighbour neighbourhoods, and explanation feasibility. To avoid circular evaluation and data leakage, **all transformers are fitted strictly on training folds inside the cross-validation loop**; validation and test sets are transformed without refitting.

### Pre-Registered Protocols in `DECISIONS.md`:
- **Progressive Sampling Stopping Criterion:** Doubling sample size declared converged when incremental PR-AUC gain $\Delta < 0.005$.
- **Evaluation Objective:** Average Precision (PR-AUC) as primary metric due to extreme positive-class skew.
- **Zero-Leakage Entity Grouping:** Pairs grouped by `min(entity(r1), entity(r2))` across 5 folds.

---

## 2. Rescaling Decisions per Model Family

### Empirical 5-Fold Cross-Validation Performance:

| Model Family | Unscaled PR-AUC | Scaled PR-AUC | $\Delta$ PR-AUC | Invariant? | Theoretical Justification (Lecture 4) |
|---|---|---|---|---|---|
| **Decision Tree** | 0.9982 $\pm$ 0.0009 | 0.9982 $\pm$ 0.0009 | +0.0000 | **YES** | Tree splits evaluate ordinal single-attribute inequalities ($x_j \le \theta$). Monotonic scaling preserves rank order exactly. |
| **$k$-Nearest Neighbours ($k$-NN)** | 0.9939 $\pm$ 0.0032 | **0.9975** $\pm$ 0.003 | **+0.0036** | **NO** | In Euclidean space, unscaled features ($[0, 100]$) dominate distance metrics, drowning out bounded $[0, 1]$ string similarities. |
| **Multi-Layer Perceptron (MLP)** | 0.9998 $\pm$ 0.0002 | **0.9999** $\pm$ 0.0001 | **+0.0001** | **NO** | Backpropagation gradients oscillate along wide-variance axes, causing slow or unstable convergence. |

### Heavy-Tailed Attributes Log Transformation:
As established in Lecture 4, unbounded count and elapsed-time features exhibit extreme positive skew. We apply natural log transforms $\ln(1 + |\Delta|)$ to `dob_diff_days_log`, `age_diff_log`, and `visit_count_diff_log`, compressing extreme outliers and converting multiplicative differences into additive scales.

---

## 3. Dimensionality Reduction (PCA) vs Feature Subset Selection

Evaluated on identical folds and the same common classifier (`LogisticRegression(max_iter=500)`):

| Strategy | Dimension ($k$) | 5-Fold PR-AUC | 5-Fold ROC-AUC | Runtime | Explainability for Records Clerk |
|---|---|---|---|---|---|
| **Full Features Baseline** | 58 features | 0.9999 | 1.0 | 12.929s | High (all 58 pairwise comparisons) |
| **PCA Reduction ($k=15$)** | 15 components | 0.9998 | 1.0 | 12.968s | **LOW (uninterpretable linear combinations)** |
| **Filter Selection (Correlation)** | 15 features | **0.9993** | **0.9999** | 13.785s | **HIGH (preserves direct clinical rules)** |
| **Greedy Forward Selection** | 6 features | 0.9986 | 0.9998 | 11.116s | **HIGHEST (minimal non-redundant set)** |

### Final Decision: Why We Ship Feature Selection
> [!IMPORTANT]
> **Verdict:** We strictly ship **Feature Selection** rather than PCA. While PCA captures 80% of variance with 15 components, > its features are linear combinations of 58 attributes (e.g. $0.23 \cdot \text{fn\_jw} - 0.18 \cdot \text{age} + \dots$) which cannot be translated into clerk-facing sentences. > Feature selection maintains high PR-AUC (0.9993) while keeping native, explainable attributes like `dob_exact`, `nid_exact`, and `phone_last7_equal`.

**Selected Filter Features:** `father_exact, dob_exact, dob_day_equal, dob_month_equal, dob_one_digit_edit, dob_diff_days_log, nid_hamming_le_1, nid_levenshtein...`

---

## 4. Discretization Strategies for Naive Bayes

Evaluated on `CategoricalNB(alpha=1.0)` across continuous similarity measures:

| Discretization Method | 5-Fold CV PR-AUC | 5-Fold CV ROC-AUC | Runtime | Behaviour on Skewed Clinical Similarities |
|---|---|---|---|---|
| **Equal-Width ($B=5$)** | 0.9923 $\pm$ 0.0015 | 0.9992 | 11.486s | Poor: creates empty intermediate bins when similarities cluster near 0.0 and 1.0. |
| **Equal-Frequency ($B=5$)** | 0.9986 $\pm$ 0.0004 | 0.9998 | 13.145s | Moderate: ties at boundary values (0.0 or 1.0) collapse bin edges. |
| **Supervised Entropy (MDLP)** | **0.9921** $\pm$ 0.0017 | **0.9992** | 13.822s | **Optimal:** places split thresholds exactly where class entropy $H(Y|X)$ is minimized. |

---

## 5. Missing-Value Treatments per Family

| Family | Evaluated Strategy | CV PR-AUC | Assessment & Implementation |
|---|---|---|---|
| **Decision Tree** | Sentinel (`-1.0`) + Presence Flags | **0.9982** | **Recommended:** Tree splits cleanly on presence indicators without blurring valid similarity scores. |
| **Decision Tree** | Mean Imputation | 0.9982 | Inferior: imputes synthetic similarity scores that misrepresent missing records. |
| **$k$-NN** | Mean Imputer + Scaler | **0.9952** | **Recommended:** Neutral distance imputation avoids geometric distortion. |
| **$k$-NN** | Sentinel + Scaler | 0.9952 | Inferior: sentinel values mapped to extreme distance poles disrupt k-NN queries. |

---

## 6. Progressive-Sampling Learning Curve

- **Pre-registered Convergence Threshold:** $\Delta < 0.005$ PR-AUC gain upon sample doubling.
- **Declared Convergence Sample Size:** **3,250 pairs**.

| Sample Size ($N$) | 5-Fold CV PR-AUC | Incremental Gain ($\Delta$) | Convergence Status |
|---|---|---|---|
| 1,625 | 0.9966 $\pm$ 0.0007 | +0.0000 | Active Growth |
| 3,250 | 0.9971 $\pm$ 0.0007 | +0.0006 | **CONVERGED** |
| 6,500 | 0.9963 $\pm$ 0.0012 | -0.0009 | **CONVERGED** |
| 13,000 | 0.9962 $\pm$ 0.0024 | -0.0001 | **CONVERGED** |
| 26,000 | 0.9974 $\pm$ 0.0008 | +0.0011 | **CONVERGED** |
| 51,984 | 0.9982 $\pm$ 0.0009 | +0.0009 | **CONVERGED** |

---

## 7. Generated Visualizations

The following figures were generated and saved in `reports/figures/`:
1. `reports/figures/rescaling_comparison.png`: Scaled vs unscaled bar charts across model families.
2. `reports/figures/pca_variance_explained.png`: Cumulative explained variance spectrum for PCA.
3. `reports/figures/pca_vs_selection.png`: Performance comparison between PCA and Feature Selection.
4. `reports/figures/discretization_comparison.png`: Performance of binning strategies on Naive Bayes.
5. `reports/figures/progressive_sampling_curve.png`: Progressive-sampling learning curve.
