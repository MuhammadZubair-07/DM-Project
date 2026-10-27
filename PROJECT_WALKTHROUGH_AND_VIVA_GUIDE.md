# National Patient Master Index (NPMI) Deduplication Assistant
## Complete System Walkthrough, Screen-by-Screen Guide & Viva Defense Manual

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool  
> **Team:** 3 Students (Member A, Member B, Member C)  
> **Authoritative Metrics Source:** [`reports/evaluation.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/evaluation.json), [`reports/blocking.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/blocking.json), and [`reports/audit.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/audit.json). All numbers are mathematically verified and 100% reproducible.

---

## Table of Contents
1. [What This Project Is About & What It Does](#1-what-this-project-is-about--what-it-does)
2. [Complete Screen-by-Screen & Section-by-Section Guide](#2-complete-screen-by-screen--section-by-section-guide)
   - [Home Dashboard (`app.py`)](#home-dashboard-apppy)
   - [Screen 1: Data & Quality Audit (`pages/1_Data_and_Quality.py`)](#screen-1-data--quality-audit-pages1_data_and_qualitypy)
   - [Screen 2: Model Performance Benchmarks (`pages/2_Model_Performance.py`)](#screen-2-model-performance-benchmarks-pages2_model_performancepy)
   - [Screen 3: Single Pair Chart Investigation (`pages/3_Single_Prediction.py`)](#screen-3-single-pair-chart-investigation-pages3_single_predictionpy)
   - [Screen 4: High-Throughput Batch Scoring (`pages/4_Batch_Scoring.py`)](#screen-4-high-throughput-batch-scoring-pages4_batch_scoringpy)
   - [Screen 5: Hospital Cost Matrix & Settings (`pages/5_Settings_and_Cost_Matrix.py`)](#screen-5-hospital-cost-matrix--settings-pages5_settings_and_cost_matrixpy)
   - [Screen 6: Domain Merge Review Console (`pages/6_Merge_Review.py`)](#screen-6-domain-merge-review-console-pages6_merge_reviewpy)
3. [Member-by-Member Contribution & Viva Defense Guide](#3-member-by-member-contribution--viva-defense-guide)
   - [Member A: Data Engineering, Quality Audit & Validation](#member-a-data-engineering-quality-audit--validation)
   - [Member B: Candidate Blocking, Features, Models & Prospective Evaluation](#member-b-candidate-blocking-features-models--prospective-evaluation)
   - [Member C: Web Architecture, Reversible Merge Store & Usability](#member-c-web-architecture-reversible-merge-store--usability)
4. [Master Viva Questions & Defensible Answers for Each Screen](#4-master-viva-questions--defensible-answers-for-each-screen)

---

## 1. What This Project Is About & What It Does

### 1.1 The Clinical Problem
In modern hospital networks (teaching hospitals, trauma emergency units, diagnostic pathology labs, and outpatient clinics), electronic health records (EHRs) are frequently collected across fragmented legacy databases. When patients register:
- Names suffer from typographical errors, honorific variations, and phonetic transliterations (e.g., *Muhammad Tariq* vs. *Mohd. Tarique*).
- National Identity Cards (CNIC) are often uncollected, omitted, or mistyped.
- Birth dates suffer day-month transpositions or default sentinel dates (e.g. `1900-01-01`).
- Phone numbers change or use varying formats (`+92`, `0300`, dashes, spaces).

This causes the **same human being** to exist multiple times under distinct patient identification numbers (`record_id`), creating **duplicate medical charts**.

### 1.2 The Real-World Danger
1. **Fatal Blood Transfusion Incompatibilities:** If Patient A (Blood Group $O-$) and Patient B (Blood Group $AB+$) are erroneously merged, an emergency trauma team consulting the consolidated chart may infuse incompatible blood products, causing acute intravascular hemolysis and death.
2. **Fragmented & Split Clinical Histories:** When past encounters are unlinked, doctors miss critical chronic diagnoses (diabetes, kidney failure), life-threatening anaphylactic drug allergies (penicillin), and prior surgeries.
3. **Redundant Diagnostic Harm:** Patients undergo repeated ionizing radiation (redundant CT scans) and invasive biopsies simply because prior hospital records were not linked.
4. **Administrative & Financial Waste:** Millions of rupees in wasted laboratory tests and hundreds of hours spent by records clerks manually searching through paper charts.

### 1.3 What This Software Does
The **National Patient Master Index (NPMI) Deduplication Assistant** is a full-stack, clinically safe decision-support web portal built for **hospital medical records clerks, health informatics officers, and clinical auditors**:
1. **Ingests Dirty Patient Registers:** Audits 8 Lecture 2 fault categories and cleans demographic attributes while preserving uncleaned values in raw columns and logging audit transformations in 10 provenance columns.
2. **Solves the Combinatorial Blocking Challenge:** Reduces the quadratic search space from **1.48 billion comparisons** down to **823,534 candidate pairs** in **1.56 seconds** ($99.9443\%$ reduction ratio) using 5 from-scratch dictionary passes.
3. **Classifies Duplicate Candidates:** Applies 4 machine learning pipelines (Pruned Decision Tree, Categorical Naive Bayes, $k$-NN, MLP) evaluating 58 pairwise comparison features.
4. **Enforces Evaluation Honesty:** Places the **Majority-Class Baseline ($99.46\%$)** directly adjacent to every single accuracy figure, operating at an asymmetric, cost-optimal threshold ($\theta^* = 0.20$).
5. **Surfaces Zero-Jargon Plain Explanations:** Translates tree paths and log-likelihood ratios into clinical English (e.g., *"Same date of birth"*, *"Phone numbers differ"*), never exposing node IDs or Gini impurities.
6. **Executes 100% Reversible Golden Record Merges:** Automatically detects critical clinical hazards (Blood Group conflicts and CNIC collisions), generates consolidated Golden Records, and provides **1-click byte-for-byte exact Undo** via SQLite (`data/merges.db`).

---

## 2. Complete Screen-by-Screen & Section-by-Section Guide

The application consists of the Executive Home portal (`app.py`) and 6 dedicated functional screens located in `pages/`.

```
├── app.py                             # Executive Home Dashboard
├── pages/
│   ├── 1_Data_and_Quality.py          # Screen 1: Data Ingestion, Quality Audit & Validation
│   ├── 2_Model_Performance.py        # Screen 2: Prospective Benchmarks (Baseline Adjacent!) & Curves
│   ├── 3_Single_Prediction.py         # Screen 3: Single Pair Chart Investigation & Plain Explanations
│   ├── 4_Batch_Scoring.py             # Screen 4: High-Throughput Batch Scoring & Ranked Worklist
│   ├── 5_Settings_and_Cost_Matrix.py  # Screen 5: Cost Matrix Editor, Threshold Tuning & Live Simulation
│   └── 6_Merge_Review.py              # Screen 6: Domain Merge Console, Survivorship & Reversible Undo
```

---

### Home Dashboard (`app.py`)
- **Mandatory Academic Disclaimer Banner:** Rendered at the very top:  
  `⚠️ Student project. Not fit for clinical, financial or legal use. All data is synthetic.`  
  *(Enforces healthcare software governance standards, preventing accidental clinical deployment).*
- **Executive System Overview Card:** Outlines the core mission: matching fragmented patient encounters across hospital departments.
- **Workflow Architecture Diagram:** Shows the 6-stage end-to-end data mining pipeline:
  $$\text{Raw Register} \longrightarrow \text{Quality Audit} \longrightarrow \text{Candidate Blocking} \longrightarrow \text{Pairwise Features} \longrightarrow \text{Cost-Optimal Model} \longrightarrow \text{Reversible Merge}$$
- **Persona Quick-Navigation Matrix:** Directs users to the appropriate screen based on their role:
  - *Quality Auditor:* Screen 1 (Audit & Validation)
  - *Informatics Officer:* Screen 2 (Prospective Benchmarks)
  - *Intake Clerk:* Screen 3 (Single Pair Investigation) & Screen 6 (Merge Review)
  - *Department Supervisor:* Screen 4 (Batch Deduplication) & Screen 5 (Cost Settings)

---

### Screen 1: Data & Quality Audit (`pages/1_Data_and_Quality.py`)
*Built for: Clinical Data Auditors and Healthcare Ingestion Clerks.*

#### Section 1: Ingestion & Stevens Scale Data Dictionary
- **What it shows:** Allows uploading a CSV register or clicking **"Load demo_register.csv"** (~5,000 records).
- **Measurement Taxonomy:** Displays Stevens' 4-scale classification across all 16 register fields:
  - **Nominal (11):** `first_name`, `family_name`, `father_name`, `gender`, `national_id`, `phone`, `city`, `address`, `blood_group`, `source_system`, `record_id`
  - **Ordinal (1):** `insurance_tier` (`None` < `Bronze` < `Silver` < `Gold`)
  - **Interval (2):** `date_of_birth`, `registration_date` (calendar dates lacking an absolute physical zero)
  - **Ratio (2):** `age_years`, `visit_count` (possessing a true non-arbitrary zero point)

#### Section 2: Lecture 2 Data Quality Audit Table
- **What it shows:** An exhaustive audit detecting **8 distinct Lecture 2 fault categories** across **19,773 faulty records** ($36.37\%$ of the 54,373 register):
  1. *F1 Exact Duplicates:* `df.duplicated(subset=non_id_cols)` (0 exact duplicate rows).
  2. *F2 Inconsistent Encoding:* Unstandardized city aliases (2,324), non-ISO date formats (20,802), dirty gender codes (1,410), malformed phone formats (1,825).
  3. *F3 Sentinel Values:* Default DOB `1900-01-01` (392), sentinel age `-999` (300), phone zeros `0000000000` (183).
  4. *F4 Invalid Values:* Future DOBs or DOB > registration date (154), impossible ages > 110 (635).
  5. *F5 Missing Values:* Null rates cross-tabulated across source systems.
  6. *F6 Measurement / Entry Errors:* Lexicon lookup detecting typographical errors in names (1,178).
  7. *F7 Outliers:* Robust Median Absolute Deviation (MAD > 3.5) on `visit_count` (315 records).
  8. *F8 Collection Errors:* Structural omissions (Legacy system 100% missing CNIC).
- **Non-Random Missingness Proof:** Displays the Chi-Square contingency test ($\chi^2 = 13,842.19, p < 10^{-300}$), proving that missingness in `national_id` depends on `source_system`, confirming data is **Missing at Random (MAR)** rather than MCAR.

#### Section 3: Before/After Repair Table with Provenance Tracking
- **What it shows:** A side-by-side reconciliation table demonstrating the impact of the cleaning pipeline:
  - `city`: Reduced from 38 dirty variants down to exactly 13 canonical Pakistani cities.
  - `gender`: Resolved 1,410 non-standard codes down to `{M, F, U}`.
  - `dob`: Standardized 20,802 non-ISO strings into standard `YYYY-MM-DD`.
- **Provenance Architecture:** Preserves original values in 13 `{col}_raw` columns and records audit actions in 10 `{col}_prov` columns (`original`, `case_fixed`, `alias_mapped`, `sentinel_cleared`, `day_month_swapped`).

#### Section 4: Automated Clinical Validation Pipeline (`validate()`)
- **What it shows:** Runs 15 clinical assertions on both clean and raw registers:
  - **Clean Register (`register_clean.csv`):** **PASSED (15/15 assertions passed, 0 violations)**.
  - **Raw Register (`register_raw.csv`):** **FAILED (5/15 passed, 10 failing assertions)**, capturing 20,802 DOB errors, 2,324 city errors, 1,825 phone syntax errors, and 1,410 gender errors.

#### Section 5: Cleaned Dataset Export
- **What it shows:** A one-click download button exporting the standardized `register_clean.csv` complete with provenance tracking columns.

---

### Screen 2: Model Performance Benchmarks (`pages/2_Model_Performance.py`)
*Built for: Health Informatics Officers and Machine Learning Auditors.*

#### Section 1: Candidate Blocking Efficiency Summary
- **What it shows:** Displays 4 top-level KPI metric cards for the assigned challenge:
  - *Total All-Pairs Search Space:* **$1,478,184,378$ comparisons** ($n = 54,373$ records)
  - *Candidate Pairs Extracted:* **$823,534$ pairs**
  - *Reduction Ratio (RR):* **$99.9443\%$** of comparisons pruned
  - *Pairs Completeness (PC):* **$94.48\%$** ($9,216$ of $9,754$ true duplicates retained; 538 missed true pairs)
- **Detailed Blocking Pass Table:** Expander detailing all 5 passes (B1 Soundex+Year, B2 Fn3+YearMo, B3 Phone7, B4 CNIC exact, B5 Sorted-Neighbourhood $w=5$) and the multi-pass union runtime ($1.56$ seconds).

#### Section 2: Prospective Test Set Benchmark Table (MANDATORY BASELINE ADJACENCY!)
- **What it shows:** The primary benchmark table evaluated **strictly once** on the prospective test cohort ($N = 106,298$ candidate pairs, $570$ true duplicates, natural prevalence $0.5362\%$) at operating threshold $\theta^* = 0.20$:

| Classifier Model | Test Accuracy (%) | **Majority Baseline (%) (Adjacent)** | Precision (%) | Recall (%) | End-to-End Recall (%) | F1-Score | PR-AUC | ROC-AUC | Expected Cost (Rs./1k) | 5-Fold CV PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pruned Decision Tree** | **99.98%** | **99.46%** | **96.45%** | **100.0%** | **94.48%** | **0.9819** | **0.9794** | **0.9999** | **Rs. 0.99** | **0.9785 ± 0.0042** |
| **Categorical Naive Bayes** | 99.90% | 99.46% | 84.70% | 100.0% | 94.48% | 0.9171 | 0.9658 | 0.9999 | Rs. 4.84 | 0.9621 ± 0.0058 |
| **$k$-Nearest Neighbours** | 99.93% | 99.46% | 88.51% | 100.0% | 94.48% | 0.9390 | 0.9816 | 0.9999 | Rs. 3.48 | 0.9802 ± 0.0039 |
| **Multi-Layer Perceptron** | 99.91% | 99.46% | 84.95% | 100.0% | 94.48% | 0.9186 | 0.9841 | 0.9999 | Rs. 4.75 | 0.9829 ± 0.0035 |
| *Domain Rule Baseline ("Same DOB & Phone")* | 99.71% | 99.46% | 97.82% | 47.19% | 44.59% | 0.6367 | — | — | Rs. 60.58 | — |

- **Why Baseline Adjacency is Mandatory:** A blue informational banner explains that non-matches constitute **$99.46\%$** of pairs. A naive model predicting *'never a match'* achieves **$99.46\%$ accuracy** with **$0\%$ recall**. Placing the baseline beside accuracy proves the model generates genuine clinical lift.
- **End-to-End Recall:** Explicitly penalizes downstream model recall by upstream blocking losses:
  $$\text{End-to-End Recall} = \text{Blocking Completeness} \times \text{Classifier Recall} = 0.9448 \times 1.00 = \mathbf{94.48\%}$$

#### Section 3: Diagnostic Visualizations (Tabs 1 to 4)
- **Tab 1: ROC Curves (with AUC):** Multi-model ROC curves demonstrating near-perfect class separation ($\text{AUC} = 0.9999$).
- **Tab 2: Precision-Recall Curves:** Evaluates performance in the presence of severe class imbalance ($0.54\%$ positives).
- **Tab 3: Confusion Matrix at Threshold:** Interactive heatmap showing True Positives ($570$), False Positives ($21$), False Negatives ($0$), and True Negatives ($105,707$) at $\theta^* = 0.20$.
- **Tab 4: Probability Calibration Curves:** Reliability diagram showing that Platt-calibrated probabilities closely track the ideal $45^\circ$ diagonal.

#### Section 4: Leakage Verification & Proof of Integrity
- **What it shows:** An expander documenting the temporal entity split (cutoff `2022-04-15`), proving zero entity overlap between train and test (`train_entities ∩ test_entities == ∅`), and showing that `record_id` proximity yields zero predictive power on our shuffled register.

---

### Screen 3: Single Pair Chart Investigation (`pages/3_Single_Prediction.py`)
*Built for: Registration Clerks and Intake Supervisors investigating suspicious patient pairs.*

#### Section 1: Dual Patient Chart Comparison Form & Clinical Presets
- **What it shows:** Side-by-side demographic input forms for Record A and Record B covering all cleanable fields.
- **5 Benchmark Clinical Quick-Load Presets:**
  1. *🟢 Real-World Match (Typo & Missing CNIC):* Tests phonetic name variants and missing identifiers.
  2. *🟢 Exact Duplicate Record:* Tests identical demographic inputs.
  3. *🔴 Hard Negative (Twin Sisters):* Identical birth date, address, phone, and parents, but different first names and distinct CNICs.
  4. *🔴 Hard Negative (Father & Son):* Same full name, address, and phone, but differing birth dates and CNICs.
  5. *⚪ Clear Non-Match:* Completely distinct patients.

#### Section 2: Clinical Verdict & Cost-Derived Confidence Badge
- **What it shows:** Displays duplicate probability, calibrated confidence label, and visual verdict badge based on hospital cost bands:
  - $p \ge 0.8333$: **`Very Likely Duplicate (High Risk)`** (Purple Badge)
  - $0.20 \le p < 0.8333$: **`Worth Reviewing (Candidate Duplicate)`** (Orange Badge)
  - $p < 0.20$: **`Distinct Patients (Unlikely Duplicate)`** (Green Badge)

#### Section 3: Zero-Jargon Plain-Language Clinical Reasons
- **What it shows:** Translates model mechanics into clear English sentences:
  - **Reasons FOR Match (Green Bullets):** *"Same date of birth"*, *"Phone numbers match exact subscriber digits"*, *"First names are nearly identical (Muhammad / Mohammad)"*, *"Same residential address"*.
  - **Reasons AGAINST Match (Red Bullets):** *"First names differ"*, *"National ID is missing on one record so it could not be compared"*, *"Recorded birth years differ by 5 years"*.
  - **Zero Jargon Rule:** Zero internal node IDs, zero Gini impurity numbers, and zero mathematical inequalities ($<=$ or $>=$).

#### Section 4: Per-Field Agreement Flags
- **What it shows:** A side-by-side demographic table marking each attribute with 🟢 Agree, 🔴 Conflict, or ⚪ Unknown.

---

### Screen 4: High-Throughput Batch Scoring (`pages/4_Batch_Scoring.py`)
*Built for: Medical Records Supervisors processing bulk departmental patient registers.*

#### Section 1: Ingestion, Validation & Automated Leakage Guard
- **What it shows:** Ingests external departmental CSV registers or loads `sample_batch.csv` (500 records).
- **Automated Leakage Guard:** Scans uploaded files for forbidden columns (`true_entity_id`, `cluster_id`, `fault_log`). If detected, it displays a yellow warning and programmatically strips them before modeling.

#### Section 2: Pipeline Progress Bar & Execution
- **What it shows:** An animated progress bar executing 4 sequential stages in real time:
  1. *Cleaning & Standardization:* Standardizes dates, names, phones, and flags sentinels.
  2. *Multi-Pass Blocking (B1–B5):* Indexes candidate pairs across all 5 passes.
  3. *Feature Extraction:* Computes 58 comparison metrics per candidate pair.
  4. *Model Scoring & Platt Calibration:* Evaluates duplicate probabilities and synthesizes clinical reasons.

#### Section 3: Batch Blocking Savings Card
- **What it shows:** Reports comparisons performed vs. unblocked quadratic space, displaying reduction ratio ($99.5\%$) and execution runtime.

#### Section 4: Filterable Prioritized Duplicate Worklist
- **What it shows:** An interactive, ranked data table of duplicate candidates sorted by descending probability:
  - Columns: `Rank`, `Record A`, `Name A`, `Record B`, `Name B`, `Duplicate Probability`, `Verdict`, `Primary Reason`.
  - **Explicit Reason Rule:** Every single pair in the table contains an plain-English summary in the `Primary Reason` column.
  - Interactive filters: Filter by confidence band (`All`, `Very Likely`, `Worth Reviewing`, `Low Probability`) or search by patient name.

#### Section 5: Results Export & Handover
- **What it shows:** Exports scored candidate pairs with reasons to CSV for departmental records auditing or sends selected pairs directly to Screen 6 for merge review.

---

### Screen 5: Hospital Cost Matrix & Settings (`pages/5_Settings_and_Cost_Matrix.py`)
*Built for: Chief Medical Officers, Health Economists, and Clinic Administrators.*

#### Section 1: Active Model Selection
- **What it shows:** Switch between all 4 trained pipelines (Pruned Decision Tree, Categorical Naive Bayes, $k$-NN, MLP) with instant session state persistence across all screens.

#### Section 2: Hospital Misclassification Cost Matrix Editor
- **What it shows:** Configurable clinical costs with real-world units (Pakistani Rupees - PKR):
  - **Cost of False Alarm ($C_{FP}$):** Default = **Rs. 5.0** (wasted records clerk review time: 3 minutes @ Rs. 100/hour).
  - **Cost of Missed Duplicate ($C_{FN}$):** Default = **Rs. 20.0** (clinical safety risk: split medical history, redundant laboratory tests, missed allergies).
  - **Clinical Asymmetry Ratio:** Missing a duplicate is weighted **4× more hazardous** than clerk review time ($C_{FN} / C_{FP} = 4.0$).

#### Section 3: Operating Threshold Tuning Slider
- **What it shows:** Interactive threshold slider marked with the theoretical Bayesian cost-optimal threshold:
  $$\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{5.0}{5.0 + 20.0} = \mathbf{0.20}$$
  Includes quick-reset buttons: **"🎯 Reset to Cost-Optimal (0.20)"** and **"⚖️ Reset to Naive 0.50"**.

#### Section 4: LIVE Consequence Simulation (Prospective Test Cohort)
- **What it shows:** Dynamic metric cards recomputed in real time via `compute_cost_and_thresholds()` over all 106,298 test pairs:
  - *Expected Misclassification Cost per 1k Pairs:* Rs. 0.99 (showing delta vs $\theta=0.50$).
  - *Classifier Recall:* $100.0\%$ (showing duplicates saved vs $\theta=0.50$).
  - *Classifier Precision:* $96.45\%$.
  - *Worklist Queue Size:* Flagged candidate pairs requiring manual clerical inspection.
  - *Estimated Clerk Workload:* Total manual review hours and workdays (assuming 3.0 minutes per pair at 7.5 hours/day).

#### Section 5: Clinical Accuracy Trade-Off Table & Live Confusion Matrix
- **What it shows:** A side-by-side comparison table with **adjacent Majority Baseline ($99.46\%$)**, calculating the exact accuracy given up:
  $$\text{Accuracy Given Up} = \text{Accuracy}(\theta=0.50) - \text{Accuracy}(\theta^*=0.20) = \mathbf{0.000\%}$$
  Accompanied by a live 2x2 confusion matrix ($TP, FP, FN, TN$) updating dynamically as the slider moves.

---

### Screen 6: Domain Merge Review Console (`pages/6_Merge_Review.py`)
*Built for: Senior Medical Records Clerks executing human-in-the-loop patient merges.*

#### Section 1: Worklist Queue Selector & Navigation
- **What it shows:** Automatically loads pairs sent from Screen 4 or Screen 3, or allows selecting candidate pairs from the active worklist dropdown.

#### Section 2: Side-by-Side Demographic Review with Visual Cues
- **What it shows:** Side-by-side comparison of Record A and Record B across all 14 demographic attributes.
- **Triple-Encoding Accessibility:** Uses color, background fill, and explicit text tags:
  - 🟢 **Agree:** Attribute matches between charts.
  - 🔴 **Conflict:** Discrepancy detected (e.g. differing phone numbers or surnames).
  - ⚪ **Unknown:** Attribute is null/uncollected in one or both charts.

#### Section 3: Critical Clinical Hazard Red Alerts
- **What it shows:** Automated safety interlocks firing unmissable red alert boxes:
  - **Blood Group Conflict Red Alert:** Fires if both records contain differing valid blood groups ($A+$ vs $B-$), warning: `🚨 CRITICAL CLINICAL HAZARD: Recorded blood groups conflict! Merging these charts could lead to fatal transfusion errors.`
  - **National ID Collision Red Alert:** Fires if both records contain differing valid 13-digit Pakistani CNICs, warning: `🚨 IDENTITY COLLISION DETECTED: Records have distinct valid National IDs. Do NOT merge distinct citizens.`

#### Section 4: Golden Record Survivorship Rules Engine
- **What it shows:** Suggests a consolidated "Golden Record" surviving patient chart based on 3 clinical survivorship rules:
  1. *Sentinel Elimination:* Non-sentinel values always supersede sentinels (`1900-01-01`, `-999`).
  2. *Completeness:* Non-empty fields supersede nulls.
  3. *Recency:* Conflicting attributes default to the record with the more recent `registration_date`.
- **Clerk Override:** Radio buttons allow the clerk to select Record A, Record B, or type a custom edited value for any attribute.

#### Section 5: Two-Step Confirmation & Clerical Action Buttons
- **What it shows:** Three clear action buttons:
  - **`🔀 Merge Records`:** Opens a modal verification dialog requiring a mandatory clerk rationale note (e.g., *"Verified physical hospital intake form and patient phone contact"*).
  - **`❌ Reject (Distinct Citizens)`:** Permanently logs records as distinct individuals (e.g. twin sisters).
  - **`⏸️ Defer Review`:** Holds pair in queue pending additional physical chart retrieval.

#### Section 6: Reversible SQLite Merge Audit Log & Byte-for-Byte Undo
- **What it shows:** A complete historical audit log loaded from SQLite (`data/merges.db`) showing timestamp, merge ID, clerk note, and status.
- **1-Click Byte-for-Byte Exact Undo:** Clicking **"↩️ Undo Merge"** restores both original records into the active register with exact byte-for-byte serialization equality, updating the audit status to `'undone'`.

---

## 3. Member-by-Member Contribution & Viva Defense Guide

### Member A: Data Engineering, Quality Audit & Validation
- **Core Modules Owned:** [`src/generate_data.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/generate_data.py), [`src/audit.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/audit.py), [`src/clean.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/clean.py), [`src/validate.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/validate.py).
- **Core Documentation Owned:** [`docs/data_dictionary.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/data_dictionary.md), [`docs/quality_audit.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/quality_audit.md), [`docs/cleaning_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/cleaning_report.md), [`docs/generation_mechanism.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/generation_mechanism.md).
- **Web App Screens Owned:** Screen 1 (`1_Data_and_Quality.py`).

#### What Member A Did on Screen 1:
1. Built the ingestion pipeline and Stevens measurement taxonomy table.
2. Implemented all 8 Lecture 2 fault detectors flagging 19,773 faulty records.
3. Conducted the Chi-Square test ($\chi^2 = 13,842.19, p < 10^{-300}$) proving missingness in CNIC is Missing at Random (MAR) across source systems.
4. Engineered the before/after cleaning reconciliation table, creating 10 provenance columns (`{col}_prov`) and preserving 13 raw columns (`{col}_raw`).
5. Implemented 15 clinical assertions in `validate()`, proving it passes 15/15 on clean data and fails 10/15 on raw data.

#### Key Viva Questions for Member A & Model Defenses:
- **Q: Why is CNIC nominal and DOB interval?**  
  *Defense:* CNICs are symbolic identifiers; arithmetic means are invalid and numeric casting strips mandatory syntactic leading zeros. DOB has an arbitrary Gregorian epoch zero; you cannot compute ratios like "twice as old" on calendar dates alone.
- **Q: Why did you not "correct" patient name typos during data cleaning?**  
  *Defense:* Real hospital cleaning pipelines do not have an omniscient oracle. Correcting typos using ground truth is data leakage; matching noisy strings is the pairwise classifier's responsibility.
- **Q: How did you handle sentinels like 1900-01-01?**  
  *Defense:* Converted to null and set `dob_sentinel_flag = True`. This preserves the fact that a sentinel was submitted without letting the date distort continuous distance metrics.

---

### Member B: Candidate Blocking, Features, Models & Prospective Evaluation
- **Core Modules Owned:** [`src/blocking.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/blocking.py), [`src/features.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/features.py), [`src/preprocess.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/preprocess.py), [`src/models.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/models.py), [`src/evaluate.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/evaluate.py), [`src/explain.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/explain.py).
- **Core Documentation Owned:** [`docs/blocking_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/blocking_report.md), [`docs/feature_dictionary.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/feature_dictionary.md), [`docs/preprocessing_decisions.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/preprocessing_decisions.md), [`docs/model_selection.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/model_selection.md), [`docs/evaluation_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/evaluation_report.md), [`docs/explanation_guide.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/explanation_guide.md).
- **Web App Screens Owned:** Screen 2 (`2_Model_Performance.py`), Screen 5 (`5_Settings_and_Cost_Matrix.py`), algorithmic core of Screen 3.

#### What Member B Did on Screens 2 & 5:
1. Implemented the 5-pass blocking challenge from scratch in pure Python dictionaries, reducing 1.48B comparisons to 823k pairs ($99.9443\%$ reduction ratio, $94.48\%$ completeness).
2. Extracted 58 comparison features, writing character 2-gram Jaccard and character 3-gram TF-IDF cosine from scratch.
3. Built the prospective test benchmark table on Screen 2, ensuring the **Majority Baseline ($99.46\%$) is rendered directly adjacent to every accuracy figure**.
4. Built Screen 5's live simulation engine (`compute_cost_and_thresholds`), deriving the cost-optimal threshold $\theta^* = 0.20$ from asymmetric misclassification costs ($C_{FP}=5.0, C_{FN}=20.0$).
5. Engineered the zero-jargon explanation generator translating tree paths and Naive Bayes log-odds into plain English.

#### Key Viva Questions for Member B & Model Defenses:
- **Q: Why is 98% or 99% accuracy poor in this application?**  
  *Defense:* Non-matches represent $99.46\%$ of pairs. A useless model predicting "never a match" achieves $99.46\%$ accuracy while capturing zero duplicates ($0\%$ recall). Accuracy without an adjacent baseline is clinically meaningless.
- **Q: Why did you set the operating threshold to 0.20 instead of 0.50?**  
  *Defense:* Missing a duplicate risks split medical charts ($C_{FN}=\text{Rs. } 20.0$), which is 4× worse than clerk review time ($C_{FP}=\text{Rs. } 5.0$). Bayesian decision theory yields $\theta^* = \frac{5.0}{5.0 + 20.0} = 0.20$.
- **Q: What does a zero probability do to Naive Bayes, and how did you prevent it?**  
  *Defense:* An unobserved feature bin with probability 0 multiplies the entire posterior to zero. We applied Laplace smoothing ($\alpha=1.0$) to guarantee all probabilities remain positive.
- **Q: Why did Feature Selection outperform PCA?**  
  *Defense:* Feature selection won on PR-AUC ($0.9794$ vs $0.9412$). PCA's dense linear combinations destroy sparse pairwise match signals and cannot be explained to a hospital clerk.

---

### Member C: Web Architecture, Reversible Merge Store & Usability
- **Core Modules Owned:** [`app.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/app.py), [`pages/*.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages), [`src/app_state.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/app_state.py), [`src/merge_store.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py), [`scripts/*`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/scripts).
- **Core Documentation Owned:** [`README.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/README.md), [`docs/ethics_statement.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/ethics_statement.md), [`docs/usability_test.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/usability_test.md), [`docs/team_contributions.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/team_contributions.md), [`docs/traceability_audit.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/traceability_audit.md).
- **Web App Screens Owned:** Screen 6 (`6_Merge_Review.py`), Screen 4 (`4_Batch_Scoring.py`), Home (`app.py`), Screen 3 UI.

#### What Member C Did on Screens 6 & 4:
1. Designed the multi-page Streamlit portal with centralized session state, caching, and mandatory academic banners.
2. Built the Screen 6 merge review console with side-by-side triple-encoded Agree/Conflict tags.
3. Engineered automated Blood Group Conflict and CNIC Collision Red Alerts.
4. Built the SQLite Merge Store (`data/merges.db`), saving pre-merge JSON snapshots and proving **1-click byte-for-byte exact Undo**.
5. Engineered golden record survivorship rules (sentinel elimination, completeness, recency).
6. Created single-command clean-machine launchers (`run.bat`, `run.sh`) and verified sandbox reproducibility via `clean_machine_test.py`.

#### Key Viva Questions for Member C & Model Defenses:
- **Q: How does your merge store guarantee byte-for-byte exact restoration upon Undo?**  
  *Defense:* Before any merge is committed, raw unmerged JSON dictionaries of both records are stored in SQLite. Clicking Undo extracts the raw JSON and restores both records with proven byte-for-byte UTF-8 serialization equality.
- **Q: What is automation bias and how does Screen 6 combat it?**  
  *Defense:* The tendency for busy clerks to rubber-stamp AI suggestions. Combated by zero auto-merging, forced review of plain reasons for and against, 2-step confirmation dialogs, and red hazard alerts.
- **Q: Why does Screen 6 fire a red alert if blood groups conflict?**  
  *Defense:* Consolidating charts of patients with differing ABO blood groups causes fatal acute hemolytic transfusion reactions if an emergency team consults the merged chart.
- **Q: What happens if an examiner uploads a malformed CSV on Screen 4?**  
  *Defense:* The schema validator catches missing required columns and halts gracefully with `st.error()`, while the Leakage Guard strips any forbidden ground-truth columns.

---

## 4. Master Viva Questions & Defensible Answers for Each Screen

### Screen 1: Data & Quality Audit
1. **Q: Why is your audit divided into 8 categories?**  
   *A:* Following Lecture 2's data quality taxonomy (duplicate rows, inconsistent encoding, sentinel values, invalid values, missing values, entry errors, outliers, collection errors).
2. **Q: What is the Chi-Square test result?**  
   *A:* $\chi^2 = 13,842.19, p < 10^{-300}$, proving missingness in `national_id` is Missing at Random (MAR) across source systems.
3. **Q: Why are original values kept in `_raw` columns?**  
   *A:* Healthcare auditing requires full traceability; clerks must always be able to inspect what the patient originally submitted.

### Screen 2: Model Performance Benchmarks
4. **Q: What is your candidate Reduction Ratio?**  
   *A:* **$99.9443\%$** ($823,534$ comparisons out of $1,478,184,378$ all-pairs space).
5. **Q: What is Pairs Completeness?**  
   *A:* **$94.48\%$** ($9,216$ of $9,754$ true duplicates retained).
6. **Q: What is the Majority Baseline accuracy?**  
   *A:* **$99.46\%$**, displayed right beside our Decision Tree's **$99.98\%$** accuracy.
7. **Q: What is your primary model's PR-AUC?**  
   *A:* **$0.9794$** on prospective test pairs.

### Screen 3: Single Prediction
8. **Q: How does the model classify twin sisters?**  
   *A:* Correctly classifies them as **Distinct Patients** (Probability 0.08) because first names and CNICs differ despite identical birth dates, addresses, and parent names.
9. **Q: Are there any node numbers or Gini values in the explanation?**  
   *A:* Zero. Regex checks confirm zero machine learning jargon; all explanations use clinical English.

### Screen 4: Batch Scoring
10. **Q: How fast is candidate blocking during batch deduplication?**  
    *A:* Takes under 2 seconds for thousands of records because all 5 passes use pure Python hash dictionary lookups.
11. **Q: Does every pair in the worklist carry an explanation?**  
    *A:* Yes. The `Primary Reason` column provides an plain-language summary for every flagged pair.

### Screen 5: Settings & Cost Matrix
12. **Q: How is the operating threshold calculated?**  
    *A:* $\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{5.0}{5.0 + 20.0} = \mathbf{0.20}$.
13. **Q: What happens if you move the threshold to 0.50?**  
    *A:* Expected cost jumps from Rs. 0.99 up to Rs. 4.84, and 30 duplicate patients are missed, leaving them with split charts.

### Screen 6: Domain Merge Review
14. **Q: Can a clerk merge records automatically?**  
    *A:* No. Auto-merge is strictly prohibited to prevent catastrophic clinical errors.
15. **Q: What happens when you click Undo?**  
    *A:* Both original records are restored byte-for-byte from raw JSON snapshots into the active register, and the audit log records the undo timestamp.
