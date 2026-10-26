# National Patient Master Index (NPMI) Record Deduplication Assistant

> ⚠️ **MANDATORY ACADEMIC DISCLAIMER:**  
> **Student project. Not fit for clinical, financial or legal use. All data is synthetic.**  
> *Developed for DSC306 Data Mining — COMSATS University Islamabad (Fall 2026).*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.42%2B-FF4B4B.svg)](https://streamlit.io/)
[![Test Status](https://img.shields.io/badge/pytest-167%20passed-brightgreen.svg)](tests/)
[![License: Academic](https://img.shields.io/badge/license-Academic%20Use%20Only-lightgrey.svg)]()

---

## 1. What the Product Is & Who It Is For

In modern multi-facility healthcare networks (e.g., teaching hospitals, outpatient clinics, regional diagnostic laboratories, and emergency departments), patient records frequently exist fragmented across disparate electronic health record (EHR) systems. Typographical errors in patient names, non-standard address formats, missing national identity cards (CNIC), and inverted birth dates routinely produce duplicate patient charts.

Duplicate patient records create severe clinical hazards:
- **Clinical Fragmentation:** Split medical histories cause missed drug allergies, redundant ionizing radiation (duplicate CT scans), and conflicting prescriptions.
- **Administrative Waste:** Millions of rupees in wasted laboratory tests and clerical hours spent resolving contradictory charts.

The **National Patient Master Index (NPMI) Deduplication Assistant** is a full-stack clinical decision-support web portal built for **hospital medical records clerks, health informatics officers, and clinical auditors**. It reduces comparisons from $O(n^2)$ down to manageable candidate blocks, applies supervised classification pipelines (Pruned Decision Tree, Categorical Naive Bayes, $k$-NN, MLP), surfaces **plain-English clinical reasons with zero technical jargon**, and enables **human-in-the-loop, 100% reversible Golden Record merges**.

---

## 2. Single-Command Startup (Zero Setup / Zero Downloads)

**All pre-trained models (`models/*.joblib`) and clinical registers (`data/raw/`) ship directly inside this repository.**  
No external dataset downloads, API keys, or pre-training steps are required.

To run the application on a clean machine:

### Windows
```cmd
run.bat
```

### Linux / macOS
```bash
chmod +x run.sh
./run.sh
```

### Direct Python / Streamlit Command
```bash
# If using an existing virtual environment:
streamlit run app.py
```

*The launcher automatically initializes a virtual environment (`.venv`), installs pinned packages from `requirements.txt`, and opens the Streamlit application at `http://localhost:8501`.*

---

## 3. System Architecture & Six Application Screens

The portal is structured as a multi-page Streamlit web application:

```
├── app.py                             # Executive Home Dashboard & System Overview
├── pages/
│   ├── 1_Data_and_Quality.py          # Screen 1: Data Ingestion, Lecture 2 Quality Audit & Provenance
│   ├── 2_Model_Performance.py        # Screen 2: Prospective Benchmarks (Baseline Adjacent!), ROC/PR Curves
│   ├── 3_Single_Prediction.py         # Screen 3: Single Pair Chart Investigation & Zero-Jargon Explanations
│   ├── 4_Batch_Scoring.py             # Screen 4: High-Throughput Register Deduplication & Ranked Worklist
│   ├── 5_Settings_and_Cost_Matrix.py  # Screen 5: Hospital Cost Matrix Editor, Threshold Tuning & Live Simulation
│   └── 6_Merge_Review.py              # Screen 6: Reversible Golden Record Consolidation & Byte-for-Byte Undo
```

| Screen # | Screen Name | Target Persona | Primary Clinical Capability |
|:---:|:---|:---|:---|
| **1** | **Data & Quality** | Quality Auditor | Ingest raw registers; audit 8 Lecture 2 fault categories; review before/after repair table; execute 15 validation assertions (passes clean, fails raw); download cleaned CSV. |
| **2** | **Model Performance** | Informatics Officer | Prospective model comparison table with **majority baseline in adjacent column**; multi-model ROC/PR curves; Platt calibration; leakage proof; clerk workload analysis. |
| **3** | **Single Prediction** | Intake Clerk | Side-by-side comparison of any two charts with 5 benchmark clinical presets; zero-jargon reasons FOR and AGAINST; per-field agreement flags. |
| **4** | **Batch Scoring** | Audit Supervisor | High-throughput departmental register scoring with progress bar; candidate blocking metrics; prioritized duplicate worklist with search and filtering; CSV export. |
| **5** | **Settings & Costs** | Chief Medical Officer | Select active model; customize asymmetric cost matrix ($C_{FP}, C_{FN}$); tune review threshold $\theta$; observe **live simulated expected cost and clerk hours**. |
| **6** | **Merge Review** | Senior Records Clerk | Field-by-field side-by-side survivorship choices; clinical hazard alerts (Blood Group and CNIC conflicts); 2-step merge confirmation; **1-click byte-for-byte Undo**. |

---

## 4. Repository Structure & Artifact Sitemap

```
├── README.md                          # Comprehensive system documentation
├── PROJECT_CONTEXT.md                  # System brief, team rules, and anti-leakage constraints
├── PROJECT_ROADMAP.md                  # Traceability matrix and milestone specifications
├── DECISIONS.md                        # Dated architectural, modeling, and clinical decision log
├── requirements.txt                    # Pinned production and test dependencies
├── run.bat / run.sh                    # One-command automated launcher
├── app.py                              # Streamlit home portal
├── src/                                # Core algorithmic modules
│   ├── config.py                       # Project-wide paths, random seed (42), and constants
│   ├── generate_data.py                # Synthetic patient register generator
│   ├── audit.py                        # Lecture 2 data quality fault detector
│   ├── clean.py                        # Production data cleaning and provenance tracking
│   ├── validate.py                     # 15 clinical schema and range assertions
│   ├── blocking.py                     # 5-pass candidate blocking challenge (from scratch)
│   ├── split.py                        # Entity-grouped temporal train/test split
│   ├── features.py                     # 58-feature comparison vector extractor
│   ├── preprocess.py                   # Scaling, PCA vs selection, and entropy binning
│   ├── models.py                       # Unified Matcher interface (Trees, Naive Bayes, k-NN, MLP)
│   ├── evaluate.py                     # One-time prospective evaluation & Platt calibrator
│   ├── explain.py                      # Clerk-facing zero-jargon explanation engine
│   ├── app_state.py                    # Shared session state, disclaimer banners, and cached loaders
│   ├── merge_store.py                  # SQLite database store, survivorship rules, and exact undo
│   └── utils.py                        # Metric persistence and cryptographic file hashing
├── data/                               # Data repositories
│   ├── raw/                            # demo_register.csv, sample_batch.csv, register_raw.csv
│   ├── clean/                          # register_clean.csv (standardized demographics + provenance)
│   ├── truth/                          # ground_truth.csv and synthetic fault_log.csv
│   └── merges.db                       # SQLite database for merges, audit trails, and decisions
├── models/                             # Shipped fitted model pipelines (.joblib)
│   ├── decision_tree.joblib            # Pruned Decision Tree (Primary)
│   ├── naive_bayes.joblib              # Categorical Naive Bayes with Laplace smoothing
│   ├── knn.joblib                      # k-Nearest Neighbours on scaled metric space
│   └── mlp.joblib                      # Multi-Layer Perceptron neural network
├── reports/                            # Machine-readable evaluation artifacts & figures
│   ├── evaluation.json                 # Complete test metrics, curves, and baselines
│   ├── blocking.json                   # 5-pass blocking metrics and scaling benchmarks
│   ├── models.json                     # Hyperparameter grids and 5-fold CV statistics
│   ├── audit.json / audit.csv          # Quality audit findings
│   ├── clean_effects.json / .csv       # Cleaning repair before/after table
│   └── figures/                        # ROC, PR, calibration, and blocking tradeoff plots
├── docs/                               # Formal technical stage reports
│   ├── data_dictionary.md              # Stevens 4-scale measurement classification
│   ├── generation_mechanism.md         # Synthetic data generation documentation
│   ├── leakage_statement.md            # Query-time vs synthetic metadata leakage analysis
│   ├── quality_audit.md                # 8 Lecture 2 fault detectors & recall evaluation
│   ├── cleaning_report.md              # Standardization provenance and anomaly flag rules
│   ├── blocking_report.md              # Candidate blocking scaling and missed duplicate causes
│   ├── feature_dictionary.md           # 58 comparison metrics & defensible Viva formulas
│   ├── preprocessing_decisions.md      # Rescaling, PCA vs Selection, and Discretization
│   ├── model_selection.md              # Pre-registered 5-fold CV grids and 1-SE tie-break
│   ├── evaluation_report.md            # Prospective test report with mandatory baseline adjacency
│   ├── explanation_guide.md            # Zero-jargon phrase catalogue & worked clerk cases
│   ├── app_test_checklist.md           # Comprehensive manual QA verification checklist
│   ├── ethics_statement.md             # Clinical ethics, transfusion hazard, and safety rules
│   ├── usability_test.md               # 4-task usability protocol and SUS survey for strangers
│   └── team_contributions.md           # Team member ownership and signature declarations
├── scripts/                            # Operational automation scripts
│   ├── check_imports.py                # AST import scanner asserting requirements.txt coverage
│   ├── clean_machine_test.py           # Temporary sandbox clean-machine verification runner
│   ├── clean_machine_test.bat / .sh    # OS-specific clean-machine test scripts
│   └── cross_check_metrics.py          # Verifies documentation numbers against reports/*.json
└── tests/                              # Automated pytest suite (167 unit & integration tests across 14 modules)
```

---

## 5. Summary of Empirical Results

All numbers below are generated directly from authoritative JSON records in `reports/evaluation.json` and `reports/blocking.json`.

### 5.1 Upstream Candidate Blocking Efficiency (Phase 5)
- **Unblocked All-Pairs Search Space:** $1,470,229,860$ pairs ($n = 50,000$ records)
- **Multi-Pass Union Comparisons Evaluated:** $68,916$ pairs
- **Candidate Space Reduction Ratio:** **$99.9953\%$**
- **Pairs Completeness (True Duplicates Retained):** **$94.48\%$** ($538$ true pairs pruned upstream)

### 5.2 Prospective Test Set Classifier Benchmark (Phase 9)
*Evaluated ONCE on the prospective test cohort ($N = 106,298$ candidate pairs, $570$ true duplicates, natural prevalence $0.5362\%$) at operating threshold $\theta^* = 0.20$:*

| Classifier Pipeline | Test Accuracy (%) | **Majority Baseline (%) (Adjacent)** | Precision (%) | Recall (%) | F1 Score | PR-AUC | ROC AUC | Expected Cost (PKR/1k) | End-to-End Recall (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pruned Decision Tree (Primary)** | **99.98%** | **99.46%** | **96.45%** | **100.0%** | **0.9819** | **0.9794** | **0.9999** | **Rs. 0.99** | **94.48%** |
| **Categorical Naive Bayes** | 99.90% | 99.46% | 84.70% | 100.0% | 0.9171 | 0.9658 | 0.9999 | Rs. 4.84 | 94.48% |
| **$k$-Nearest Neighbours** | 99.93% | 99.46% | 88.51% | 100.0% | 0.9390 | 0.9816 | 0.9999 | Rs. 3.48 | 94.48% |
| **Multi-Layer Perceptron (MLP)** | 99.91% | 99.46% | 84.95% | 100.0% | 0.9186 | 0.9841 | 0.9999 | Rs. 4.75 | 94.48% |
| *Domain Rule Baseline ("Same DOB & Phone")* | 99.71% | 99.46% | 97.82% | 47.19% | 0.6367 | — | — | Rs. 60.58 | 44.59% |

> **Crucial Academic Rule Maintained:** The **Majority Baseline (99.4638%)** is displayed directly in the adjacent column beside every single accuracy metric across all documentation, JSON reports, and UI screens. End-to-end recall explicitly penalizes upstream blocking drops ($0.9448 \times 1.00 = 94.48\%$).

---

## 6. Synthetic Data Provenance & Generation Mechanism

1. **Stochastic Demographic Generator:** 50,000 synthetic patient records generated in `src/generate_data.py` using fixed seed `CFG.RANDOM_SEED = 42`.
2. **Pakistani Demographics Realism:**
   - Names sampled from frequency-weighted Pakistani male, female, and family name lexicons (`src/name_data.py`).
   - National Identity Cards (CNIC) follow official NADRA 13-digit format (`XXXXX-XXXXXXX-X`) with province-specific administrative division prefixes and gender-coded check digits.
   - Mobile numbers follow standard PTA telecom prefixes (`0300`, `0321`, `0333`, `0345`).
   - Cities canonicalized to 13 major Pakistani metropolitan centers.
3. **Cluster Size & Noise Injection:** True entities have cluster sizes following a realistic power-law distribution (1–5 encounters). Noise processes inject OCR typos, phonetic name swaps, transposed birth dates, sentinel values (`1900-01-01`, `00000000000`), and out-of-range encounter counts.
4. **Complete Details:** See [`docs/generation_mechanism.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/generation_mechanism.md).

---

## 7. Known Limitations & Failure Modes

1. **Upstream Blocking Escapes (5.52% Missed Duplicates):** In severe edge cases where a duplicate record suffers multi-field corruption simultaneously (missing CNIC + missing phone + corrupted birth year + misspelled surname), the pair escapes all five blocking passes.
2. **Cold-Start Emergency Trauma Admissions:** Patients admitted unconscious to emergency wards lacking identification documents cannot be linked algorithmically on day zero and require subsequent manual clerical interview.
3. **Lexicon Coverage Boundaries:** Unconventional or rare cultural name variants outside the frequency-weighted name dictionaries rely primarily on phonetic Soundex and Levenshtein string distances.

---

## 8. Verification & Quality Assurance Commands

To run all automated quality, packaging, and dependency checks:

```powershell
# 1. Verify AST imports coverage in requirements.txt
python scripts/check_imports.py

# 2. Verify all documented metrics match reports/*.json
python scripts/cross_check_metrics.py

# 3. Run full automated test suite (167 unit & integration tests)
pytest

# 4. Run isolated clean-machine temporary sandbox test
python scripts/clean_machine_test.py
```

---

## 9. Team Members & Contribution Matrix

| Member | Full Name | Registration Number | Core Code Ownership & Responsibilities |
|:---:|:---|:---|:---|
| **Member A** | `[Full Name A]` | `[FA21-BCS-XXX]` | **Data Engineering, Quality Audit & Validation Lead**<br>`src/generate_data.py`, `src/audit.py`, `src/clean.py`, `src/validate.py` |
| **Member B** | `[Full Name B]` | `[FA21-BCS-YYY]` | **Record Linkage, Modeling & Evaluation Lead**<br>`src/blocking.py`, `src/features.py`, `src/preprocess.py`, `src/models.py`, `src/evaluate.py`, `src/explain.py` |
| **Member C** | `[Full Name C]` | `[FA21-BCS-ZZZ]` | **Application Architecture, Explainability & Store Lead**<br>`app.py`, `pages/*.py`, `src/app_state.py`, `src/merge_store.py`, `scripts/*` |

*Full task breakdown and student signatures are documented in [`docs/team_contributions.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/team_contributions.md).*
