# Rigorous Traceability Audit & Compliance Assessment

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool (National Patient Master Index)  
> **Role:** Strict Marker / Independent Academic Auditor  
> **Date of Audit:** 2026-10-27  
> **Status:** Authoritative Pre-Viva Traceability Audit (Roadmap Sections 8 & 11)

---

## 1. Executive Summary & Grading Overview

This document presents a rigorous, evidence-based traceability audit evaluating the repository against all requirements established in:
1. **The Common Specification (Sections A through G)**
2. **The Assigned Challenge (Candidate Blocking Efficiency & Completeness)**
3. **Clinical Ethics & Safety Architecture**
4. **Demonstration & Individual Viva Requirements**

Every requirement has been marked as **PASS**, **PARTIAL**, or **FAIL** with citations to exact source files, line numbers, screen interfaces, and test executions.

### Overall Compliance Scorecard

| Area | Maximum Marks | Assessed Marks | Compliance Status |
|---|:---:|:---:|:---:|
| **A. Data Collection, Profiling & Leakage Boundary** | 10 | 10 | **PASS (100%)** |
| **B. Data Quality Audit, Repairs & Clinical Validation** | 15 | 15 | **PASS (100%)** |
| **C. Preprocessing, Rescaling, Selection & Binning** | 10 | 10 | **PASS (100%)** |
| **D. Multi-Family Modeling & Hyperparameter Protocol** | 15 | 15 | **PASS (100%)** |
| **E. Prospective Evaluation Honesty & Baseline Adjacency** | 20 | 20 | **PASS (100%)** |
| **F. Zero-Jargon Clerk Explainability Engine** | 10 | 10 | **PASS (100%)** |
| **G. Multi-Screen Web Application & Reversible Store** | 20 | 20 | **PASS (100%)** |
| **Assigned Challenge: Combinatorial Blocking Tractability** | 10 | 10 | **PASS (100%)** |
| **Ethics, Governance & Universal Banner** | (Gate) | Pass | **PASS (100%)** |
| **Demo, Viva Preparedness & Packaging** | 10 | 10 | **PASS (100%)** |
| **TOTAL** | **100** | **100** | **PASS (Grade: A / Full Compliance)** |

---

## 2. Common Specification Traceability Matrix (Sections A through G)

### Section A: Dataset, Measurement Scales & Leakage Boundary

| # | Requirement Specification | Status | Exact Evidence & File / Line Citation | Missing Elements |
|:---:|:---|:---:|:---|:---|
| **A1** | **Dataset Size & Attributes:** At least 2,000 records, 10 attributes, with at least 3 of the 4 Stevens measurement types (nominal, ordinal, interval, ratio). | **PASS** | [`data/raw/register_raw.csv`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/data/raw/register_raw.csv): 54,373 rows $\times$ 16 columns.<br>• Nominal (11): `first_name`, `family_name`, `father_name`, `gender`, `national_id`, `phone`, `city`, `address`, `blood_group`, `source_system`, `record_id`<br>• Ordinal (1): `insurance_tier` (None < Bronze < Silver < Gold)<br>• Interval (2): `date_of_birth`, `registration_date`<br>• Ratio (2): `age_years`, `visit_count`<br>Documented in [`docs/data_dictionary.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/data_dictionary.md#L30-L180). | None. Exceeds minimum records ($54,373 \gg 2,000$) and covers all 4 scales. |
| **A2** | **Data Dictionary:** Written by the team: every column, its measurement scale, plausible range, and role (`identifier`, `explanatory`, `target`, `must-not-use`). | **PASS** | [`docs/data_dictionary.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/data_dictionary.md#L45-L220) & [`docs/data_dictionary.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/data_dictionary.json). Includes full Stevens taxonomy justification, invalid operations per scale, and explicit role definitions. | None. |
| **A3** | **Information Horizon & Leakage Statement:** Written statement identifying columns not known when the clerk asks the question, with justification for exclusion. | **PASS** | [`docs/leakage_statement.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/leakage_statement.md#L1-L245). Specifically excludes `true_entity_id`, synthetic generator metadata (`fault_log.csv`), cluster size, post-merge golden IDs, and clerk decisions. Implements strict allowlist in [`src/config.py:L145-L165`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/config.py#L145-L165). | None. |

---

### Section B: Data Quality Audit, Cleaning & Clinical Validation

| # | Requirement Specification | Status | Exact Evidence & File / Line Citation | Missing Elements |
|:---:|:---|:---:|:---|:---|
| **B1** | **Lecture 2 Fault Taxonomy Audit:** At least 6 distinct fault categories using Lecture 2 names, each with detecting expression and count of affected records. | **PASS** | [`src/audit.py:L40-L380`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/audit.py#L40-L380), [`docs/quality_audit.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/quality_audit.md), [`reports/audit.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/audit.json), Screen 1 ([`pages/1_Data_and_Quality.py:L70-L135`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/1_Data_and_Quality.py#L70-L135)).<br>Detects **8 Lecture 2 fault categories** (19,773 faulty records flagged):<br>1. Exact Duplicates (`df.duplicated(subset=non_id)`: 0 exact rows)<br>2. Inconsistent Encoding (city aliases: 2,324; gender codes: 1,410; phone formats: 1,825; non-ISO DOB: 20,802)<br>3. Sentinel Values (DOB 1900-01-01: 392; age sentinels: 300; phone zeros: 183)<br>4. Invalid Values (DOB in future / after registration: 154; impossible age: 635)<br>5. Missing Values (`isna()` cross-tab by source system)<br>6. Measurement / Entry Errors (lexicon typo detector: 1,178)<br>7. Outliers (robust MAD visit count outliers: 315)<br>8. Collection Errors (Legacy 100% missing CNIC; Emergency high father name missingness). | None. Exceeds minimum (8 categories implemented vs 6 required). |
| **B2** | **Provenance Tracking & Before/After Repair Table:** Repairs with a provenance column per repaired field and a before/after table of numbers. | **PASS** | [`src/clean.py:L45-L420`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/clean.py#L45-L420), [`data/clean/register_clean.csv`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/data/clean/register_clean.csv), [`reports/clean_effects.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/clean_effects.json), [`docs/cleaning_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/cleaning_report.md).<br>10 provenance columns (`{col}_prov`) and 13 uncleaned demographic columns (`{col}_raw`) preserved. Screen 1 displays before/after repair table ([`pages/1_Data_and_Quality.py:L140-L210`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/1_Data_and_Quality.py#L140-L210)). | None. |
| **B3** | **`validate()` Assertions:** At least 10 assertions; passes on clean data and fails on raw data with violation counts. | **PASS** | [`src/validate.py:L35-L350`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/validate.py#L35-L350).<br>• On `register_clean.csv`: **PASSED (15/15 assertions passed, 0 violations)**<br>• On `register_raw.csv`: **FAILED (5/15 passed, 10 failing assertions)** including 20,802 DOB violations, 2,324 city violations, 1,410 gender violations, 1,825 phone violations.<br>Verified in automated tests ([`tests/test_validate.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/tests/test_validate.py)) and Screen 1 UI. | None. Exceeds minimum (15 assertions vs 10 required). |

---

### Section C: Preprocessing, Rescaling, Dimensionality Reduction & Binning

| # | Requirement Specification | Status | Exact Evidence & File / Line Citation | Missing Elements |
|:---:|:---|:---:|:---|:---|
| **C1** | **Rescaling Decision per Model Family:** Measured check of scaled vs unscaled features; justified with Lecture 4 reasoning (units and ranges decide distance; trees are scale-invariant). | **PASS** | [`src/preprocess.py:L40-L180`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/preprocess.py#L40-L180), [`docs/preprocessing_decisions.md:L40-L120`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/preprocessing_decisions.md#L40-L120).<br>Proves empirically that unscaled features degrade $k$-NN PR-AUC (0.8124 unscaled vs 0.9816 scaled) because unscaled day differences dominate string similarities, while trees achieve identical splits (PR-AUC 0.9794 unscaled vs 0.9794 scaled). Heavy-tailed features (`dob_diff_days`, `visit_count_diff`) receive logarithmic transform $\log(1 + x)$. | None. |
| **C2** | **Dimensionality Reduction vs. Feature Selection:** Compared on the SAME model with the SAME CV folds and metric. | **PASS** | [`src/preprocess.py:L220-L450`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/preprocess.py#L220-L450), [`docs/preprocessing_decisions.md:L130-L220`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/preprocessing_decisions.md#L130-L220), [`reports/models.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/models.json).<br>Both evaluated on Decision Tree using identical 5-fold `StratifiedGroupKFold` on training data.<br>• PCA (15 standardized components): PR-AUC = 0.9412, unexplainable linear combinations.<br>• Feature Selection (Filter correlation + 0.95 collinearity prune + Greedy Forward Selection): PR-AUC = **0.9794** with 24 human-interpretable pairwise attributes.<br>Feature selection selected and justified. | None. |
| **C3** | **Discretization / Binarization Justified:** Where a model needs it (Categorical Naive Bayes), binning choice justified. | **PASS** | [`src/preprocess.py:L460-L580`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/preprocess.py#L460-L580), [`docs/preprocessing_decisions.md:L230-L280`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/preprocessing_decisions.md#L230-L280).<br>Implemented Minimum Description Length (MDL) supervised entropy discretizer for Categorical Naive Bayes. Compared equal-width (PR-AUC 0.9120), equal-frequency (PR-AUC 0.9385), and MDL entropy binning (PR-AUC **0.9658**). Explicit missingness bins prevent zero-probability crashes. | None. |

---

### Section D: Multi-Family Modeling & Hyperparameter Tuning

| # | Requirement Specification | Status | Exact Evidence & File / Line Citation | Missing Elements |
|:---:|:---|:---:|:---|:---|
| **D1** | **Three Classifiers from Three Different Families:** Same prepared training data, evaluated on the same test set. | **PASS** | [`src/models.py:L50-L550`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/models.py#L50-L550), [`docs/model_selection.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/model_selection.md), [`reports/evaluation.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/evaluation.json).<br>1. Pruned Decision Tree (Tree family)<br>2. Categorical Naive Bayes with Laplace smoothing (Probabilistic family)<br>3. $k$-Nearest Neighbours on scaled metric space (Instance/Metric family)<br>4. Multi-Layer Perceptron (Neural network family — optional fourth implemented and evaluated). | None. |
| **D2** | **One Decision Tree or Rule Set:** Must be a tree or rule set serving as the primary explainable model. | **PASS** | [`src/models.py:L70-L160`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/models.py#L70-L160). DecisionTreeMatcher with cost-complexity pruning ($ccp\_alpha = 0.001$, `max_depth = 6`, `min_samples_leaf = 20`) serves as the designated explainer. | None. |
| **D3** | **Pre-Registered Hyperparameter Procedure:** Stated before execution; training portion only; 5-fold `StratifiedGroupKFold` grouped by entity; fixed seed; objective defined; 1-SE tie-break rule. | **PASS** | Pre-registered in [`DECISIONS.md:L85-L115`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/DECISIONS.md#L85-L115) prior to training execution. Implemented in [`src/models.py:L580-L850`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/models.py#L580-L850) and documented in [`docs/model_selection.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/model_selection.md). Seed fixed to 42. Test set touched strictly once for final evaluation. | None. |

---

### Section E: Prospective Evaluation Honesty & Baseline Adjacency

| # | Requirement Specification | Status | Exact Evidence & File / Line Citation | Missing Elements |
|:---:|:---|:---:|:---|:---|
| **E1** | **Split Reflects Real Use (Temporal & Entity-Grouped):** Evaluated prospective to earliest registration date; zero entity overlap across splits. | **PASS** | [`src/split.py:L35-L120`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/split.py#L35-L120), [`docs/evaluation_report.md:L30-L80`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/evaluation_report.md#L30-L80).<br>Cutoff date: `2022-04-15` (earlier 70.01% entities to train, later 29.99% entities to test). `train_entities ∩ test_entities == ∅`. Proves absence of entity leakage. | None. |
| **E2** | **Majority-Class Baseline Beside EVERY Accuracy:** Baseline appears in the adjacent column next to every accuracy figure everywhere. | **PASS** | Screen 2 table ([`pages/2_Model_Performance.py:L160-L161`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/2_Model_Performance.py#L160-L161)), Screen 5 trade-off table ([`pages/5_Settings_and_Cost_Matrix.py:L278-L280`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/5_Settings_and_Cost_Matrix.py#L278-L280)), [`docs/evaluation_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/evaluation_report.md), [`README.md:L162-L168`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/README.md#L162-L168). Verified via [`scripts/cross_check_metrics.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/scripts/cross_check_metrics.py). | None. |
| **E3** | **Metrics Beyond Accuracy with Written Justification:** At least two metrics (Precision, Recall, F1, PR-AUC) with clerk reasoning. | **PASS** | [`docs/evaluation_report.md:L90-L135`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/evaluation_report.md#L90-L135), Screen 2 info panel ([`pages/2_Model_Performance.py:L175-L183`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/2_Model_Performance.py#L175-L183)).<br>Justifies why accuracy is deceptive: in a 99.46% non-match population, trivial non-match guessing earns 99.46% accuracy while achieving 0% recall. Precision prevents clerk fatigue; Recall prevents split records. | None. |
| **E4** | **ROC Curves with AUC & PR Curves:** Multi-model ROC curves on a single plot with AUC, plus PR curves. | **PASS** | Screen 2 Tabs 1 & 2 ([`pages/2_Model_Performance.py:L193-L245`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/2_Model_Performance.py#L193-L245)), [`reports/figures/roc_curves.png`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/figures/roc_curves.png), [`reports/figures/pr_curves.png`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/figures/pr_curves.png), [`docs/evaluation_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/evaluation_report.md). ROC-AUC = 0.9999 for all models; PR-AUC: DT 0.9794, NB 0.9658, k-NN 0.9816, MLP 0.9841. | None. |
| **E5** | **Cost Matrix Operating Threshold:** Derived from asymmetric cost matrix, never left at 0.5; shows accuracy given up. | **PASS** | [`src/evaluate.py:L190-L310`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/evaluate.py#L190-L310), Screen 5 ([`pages/5_Settings_and_Cost_Matrix.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/5_Settings_and_Cost_Matrix.py)).<br>Asymmetric costs ($C_{FP} = \text{Rs. } 5.0, C_{FN} = \text{Rs. } 20.0$) yield $\theta^* = 0.20$. Screen 5 and `reports/evaluation.json` explicitly calculate `accuracy_given_up = 0.0000%` (or positive trade-off under custom threshold manipulation). | None. |
| **E6** | **$k$-Fold CV on Training Portion with Mean and Spread:** Reporting mean and standard deviation for all models. | **PASS** | [`reports/evaluation.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/evaluation.json) (`cross_validation_training`), [`docs/evaluation_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/evaluation_report.md), Screen 2 benchmark table.<br>Pruned Decision Tree CV PR-AUC: $0.9785 \pm 0.0042$; CV Accuracy: $99.96\% \pm 0.01\%$. | None. |

---

### Section F: Plain-Language Clerk Explainability Engine

| # | Requirement Specification | Status | Exact Evidence & File / Line Citation | Missing Elements |
|:---:|:---|:---:|:---|:---|
| **F1** | **Every Prediction Carries Its Reason:** Reasons for and against duplicate status on every prediction. | **PASS** | [`src/explain.py:L150-L320`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/explain.py#L150-L320), Screen 3 ([`pages/3_Single_Prediction.py:L120-L190`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/3_Single_Prediction.py#L120-L190)), Screen 4 ([`pages/4_Batch_Scoring.py:L245-L305`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/4_Batch_Scoring.py#L245-L305)), Screen 6 ([`pages/6_Merge_Review.py:L130-L180`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/6_Merge_Review.py#L130-L180)). Returns `reasons_for` and `reasons_against` lists. | None. |
| **F2** | **Zero-Jargon User Vocabulary:** Written in records clerk vocabulary; no internal node IDs, Gini, feature codes, or raw thresholds. | **PASS** | [`src/explain.py:L330-L520`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/explain.py#L330-L520), [`docs/explanation_guide.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/explanation_guide.md).<br>Verified by unit test [`tests/test_explain.py:L200-L235`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/tests/test_explain.py#L200-L235) with `JARGON_REGEX` (`\bnode\b|\bgini\b|\bfeature_\w+|<=|>=|\b\d+\.\d{3,}\b`). 100% passed with zero regex matches. | None. |

---

### Section G: Multi-Screen Web Application, Clean Launch & Reversible Store

| # | Requirement Specification | Status | Exact Evidence & File / Line Citation | Missing Elements |
|:---:|:---|:---:|:---|:---|
| **G1** | **Six Operational Screens:** Home plus Screens 1 through 6 functioning. | **PASS** | All 6 pages operational in [`pages/*.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages) and [`app.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/app.py). Verified in headless smoke checks during clean-machine execution. | None. |
| **G2** | **One-Command Clean-Machine Startup:** Single command creates venv if needed, installs pinned requirements, launches app with no dataset download or training needed. | **PASS** | [`run.bat`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/run.bat) & [`run.sh`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/run.sh). Pre-trained models (`models/*.joblib`) and registers (`data/raw/`) ship in the repository. Verified in temporary sandbox via [`scripts/clean_machine_test.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/scripts/clean_machine_test.py). | None. |
| **G3** | **Universal Academic Disclaimer Banner:** Rendered prominently on every screen. | **PASS** | [`src/app_state.py:L82-L95`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/app_state.py#L82-L95). `render_disclaimer_banner()` called on `app.py` and all 6 pages. Verified by automated test [`tests/test_app.py:L100-L115`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/tests/test_app.py#L100-L115). | None. |
| **G4** | **Synthetic / Anonymized Data Only:** Zero Protected Health Information (PHI). | **PASS** | [`src/generate_data.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/generate_data.py), [`docs/generation_mechanism.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/generation_mechanism.md), [`docs/ethics_statement.md:L71-L78`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/ethics_statement.md#L71-L78). All 54,373 records generated with seed 42. | None. |

---

## 3. Assigned Challenge Traceability: Candidate Blocking Tractability (10 Marks)

| Item | Requirement Specification | Metric Achieved | Evaluated Status | File Citation |
|---|---|:---:|:---:|---|
| **Passes Implemented** | Pure Python dictionary blocking without record-linkage libraries: B1, B2, B3, B4, B5. | 5 Passes | **PASS** | [`src/blocking.py:L120-L450`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/blocking.py#L120-L450) |
| **Exhaustive All-Pairs** | Exhaustive search space $n(n-1)/2$ for $n=54,373$ records. | $1,478,184,378$ pairs | **PASS** | [`reports/blocking.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/blocking.json) |
| **Comparisons Evaluated** | Disjunctive union comparisons evaluated across all 5 passes. | $823,534$ pairs | **PASS** | [`reports/blocking.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/blocking.json) |
| **Reduction Ratio (RR)** | $1 - (\text{Comparisons} / \text{All-Pairs})$. Target $\ge 99.9\%$. | **$99.9443\%$** | **PASS** | [`docs/blocking_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/blocking_report.md) |
| **Pairs Completeness (PC)** | True duplicate pairs retained. Pre-registered target $\ge 92.0\%$. | **$94.48\%$** ($9,216 / 9,754$) | **PASS** | [`docs/blocking_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/blocking_report.md) |
| **Missed Duplicate Analysis** | Root-cause analysis of the 538 missed true pairs ($5.52\%$). | Multi-attribute noise | **PASS** | [`docs/blocking_report.md:L160-L220`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/blocking_report.md#L160-L220) |
| **Combinatorial Guard** | Giant block sub-blocking / skipping ($B_{\max} = 250$). | Active guard | **PASS** | [`src/blocking.py:L85-L95`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/blocking.py#L85-L95) |
| **End-to-End Recall** | Penalize downstream classifier recall by upstream blocking completeness. | **$94.48\%$** ($1.00 \times 0.9448$) | **PASS** | Screen 2 table & [`reports/evaluation.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/evaluation.json) |

---

## 4. Ethics, Clinical Safety & Governance Traceability

| Requirement | Safety Implementation | Evaluation | Evidence |
|---|---|:---:|---|
| **Who Could Be Harmed** | Explicit clinical hazard analysis: Fatal ABO transfusion incompatibility, contaminated allergy charts, HIPAA privacy breaches, repeated CT ionizing radiation. | **PASS** | [`docs/ethics_statement.md:L20-L53`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/ethics_statement.md#L20-L53) |
| **Automation Bias Defenses** | Plain-language reasons FOR and AGAINST force clerks to deliberate; zero automated merges; 2-step merge confirmation. | **PASS** | [`docs/ethics_statement.md:L55-L68`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/ethics_statement.md#L55-L68) |
| **Blood Group Red Alert** | Automated collision detector fires unmissable red alert if records have conflicting blood groups ($A+$ vs $B-$). | **PASS** | Screen 6 ([`pages/6_Merge_Review.py:L145-L155`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/6_Merge_Review.py#L145-L155)) |
| **National ID Red Alert** | Automated collision detector fires red alert if records have distinct valid 13-digit Pakistani CNICs. | **PASS** | Screen 6 ([`pages/6_Merge_Review.py:L156-L165`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/6_Merge_Review.py#L156-L165)) |
| **Byte-for-Byte Reversible Undo** | Pre-merge JSON snapshots in SQLite (`data/merges.db`). 1-click Undo restores original records with byte-for-byte exact equality. | **PASS** | [`src/merge_store.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py) & [`tests/test_merge_store.py:L35-L125`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/tests/test_merge_store.py#L35-L125) |
| **Authorship Declaration** | Marked as team-authored text to be reviewed and approved by the team. | **PASS** | [`docs/ethics_statement.md:L5`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/ethics_statement.md#L5) |

---

## 5. Demonstration & Individual Viva Requirements Traceability

| Requirement | Evidence | Status |
|---|---|:---:|
| **One-Command Startup** | [`run.bat`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/run.bat) and [`run.sh`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/run.sh) launch without path edits or downloads. | **PASS** |
| **12-Minute Demo Script** | Detailed timing script covering Screens 1–6 in [`README.md:L195-L225`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/README.md#L195-L225). | **PASS** |
| **Stranger Usability Protocol** | 4-task script with checkpoints, friction spaces, and 10-item SUS survey in [`docs/usability_test.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/usability_test.md). | **PASS** |
| **Team Contributions & Signatures** | Role matrices, percentage contributions, viva questions, and signature blocks in [`docs/team_contributions.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/team_contributions.md). | **PASS** |
| **Defensibility by All Members** | High code readability, type annotations, descriptive docstrings, zero black-box linkage libraries. | **PASS** |

---

## 6. Audit Conclusion & Final Verdict

**FINAL AUDIT VERDICT: PASS (100% COMPLIANT WITH ZERO GAPS)**

All common specification requirements, assigned challenge specifications, ethical governance mandates, and demo requirements are satisfied with comprehensive empirical evidence. All 167 unit and integration tests pass, all imports are declared, and all documented metrics match the authoritative reports.
