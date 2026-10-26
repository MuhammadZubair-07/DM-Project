# Team Contributions, Responsibilities & Code Ownership

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool (National Patient Master Index)  
> **Submission Date:** October 2026  
> **Instructions:** *Fill in individual student details, confirm task completion, and review git commit distributions prior to viva defense.*

---

## 1. Team Roster & Assigned Roles

| Member | Full Name | Registration Number | Primary Domain Role | Email / Contact |
|:---:|:---|:---|:---|:---|
| **Member A** | `[Full Name A]` | `[FA21-BCS-XXX]` | **Data Engineering, Audit & Validation Lead** | `studentA@comsats.edu.pk` |
| **Member B** | `[Full Name B]` | `[FA21-BCS-YYY]` | **Record Linkage, Modeling & Evaluation Lead** | `studentB@comsats.edu.pk` |
| **Member C** | `[Full Name C]` | `[FA21-BCS-ZZZ]` | **Application Architecture, Explainability & Store Lead** | `studentC@comsats.edu.pk` |

---

## 2. Phase-by-Phase Task Breakdown & Code Ownership

### Member A: Data Engineering, Generation, Audit & Cleaning
- **Phase 1: Synthetic Register Generation (`src/generate_data.py`)**
  - Synthesized 50,000 realistic patient records anchored to Pakistani demographic distributions (PBS census, NADRA CNIC formats, canonical city names).
  - Designed cluster generator with power-law duplicate distribution (1–5 duplicates per entity) and realistic temporal intake intervals.
  - Generated `data/raw/register_raw.csv`, `data/raw/demo_register.csv`, `data/raw/sample_batch.csv`, `data/truth/ground_truth.csv`, and `data/truth/fault_log.csv`.
- **Phase 2: Data Dictionary & Leakage Analysis (`docs/data_dictionary.md`, `docs/leakage_statement.md`)**
  - Documented Stevens' 4-scale measurement taxonomy for all 16 register fields.
  - Authored leakage statement identifying query-time vs synthetic metadata leakage hazards.
- **Phase 3: Data Quality Audit (`src/audit.py`, `docs/quality_audit.md`)**
  - Implemented 8 Lecture 2 fault detectors (`duplicate`, `inconsistent_encoding`, `sentinel_value`, `invalid_value`, `missing_value`, `measurement_error`, `outlier`, `collection_error`).
  - Flagged 19,773 dirty records and evaluated detector recall against the synthetic fault log.
- **Phase 4: Cleaning & Validation (`src/clean.py`, `src/validate.py`, `docs/cleaning_report.md`)**
  - Built production cleaning pipeline keeping original values in `_raw` columns and adding provenance columns (`_prov`).
  - Implemented 15 clinical assertions in `src/validate.py` that **pass 15/15 on clean and fail on raw**.
- **Viva Defense Specialization:** Data quality taxonomy, why name typos were not corrected during cleaning, proving absence of query-time leakage, non-random missingness.

---

### Member B: Candidate Blocking, Features, Models & Prospective Evaluation
- **Phase 5: Candidate Blocking Challenge (`src/blocking.py`, `docs/blocking_report.md`)**
  - Implemented from scratch using pure Python dictionaries: B1 (Soundex+Year), B2 (Fn3+YearMo), B3 (Phone7), B4 (CNIC exact), and B5 (Sorted-Neighbourhood Method $w=5$).
  - Achieved $99.9953\%$ candidate reduction ratio ($68,916$ pairs out of $1.47$ billion all-pairs) with $94.48\%$ pairs completeness.
- **Phase 6: Pair Comparison Vectors & Split (`src/split.py`, `src/pairs.py`, `src/features.py`)**
  - Built entity-grouped temporal train/test split guaranteeing zero entity overlap across splits.
  - Implemented 58 comparison features including from-scratch character $n$-gram Jaccard and TF-IDF Cosine.
- **Phase 7: Preprocessing & Rescaling (`src/preprocess.py`, `docs/preprocessing_decisions.md`)**
  - Compared PCA dimensionality reduction vs. Feature Selection on identical CV folds.
  - Built supervised entropy discretizer with minimum description length stopping for Naive Bayes.
- **Phase 8: Model Selection & Hyperparameters (`src/models.py`, `docs/model_selection.md`)**
  - Pre-registered 5-fold StratifiedGroupKFold on training data with 1-SE tie-break rule.
  - Trained Pruned Decision Tree, Categorical Naive Bayes, $k$-NN, and MLP.
- **Phase 9: Prospective Evaluation Protocol (`src/evaluate.py`, `docs/evaluation_report.md`)**
  - Touched test set ONCE ($106,298$ prospective candidate pairs, $570$ true duplicates).
  - Maintained **Majority Baseline (99.46%)** beside every accuracy figure.
  - Implemented cost matrices ($C_{FP}=5.0, C_{FN}=20.0 \implies \theta^* = 0.20$), ROC/PR curves, Platt probability calibration, and leakage proofs.
- **Phase 10: Explainability Engine (`src/explain.py`, `docs/explanation_guide.md`)**
  - Engineered zero-jargon explanation generator translating tree paths, Naive Bayes log-likelihood ratios, and $k$-NN neighbours into plain English.
- **Viva Defense Specialization:** Trade-off between reduction ratio and pairs completeness, prospective baseline interpretation, Platt calibration, cost-optimal threshold derivation.

---

### Member C: Application Architecture, Screens, Merge Store & Usability
- **Phase 11: App Foundation & Baseline Adjacency (`app.py`, `src/app_state.py`, Screens 1 & 2)**
  - Designed multi-page Streamlit portal with centralized session state and caching (`@st.cache_data`).
  - Enforced mandatory academic disclaimer banner across every single screen.
  - Built Screen 1 (Audit & Clean CSV Download) and Screen 2 (Prospective Model Performance with adjacent baseline).
- **Phase 12: Single Prediction, Batch Scoring & Settings (Screens 3, 4, 5)**
  - Implemented Screen 5 (Settings): dynamic model switcher, cost matrix editor, threshold slider, and live performance simulation.
  - Implemented Screen 3 (Single Prediction): two-chart comparison form with 5 quick-load benchmark clinical cases and plain-English reasons.
  - Implemented Screen 4 (Batch Scoring): high-throughput register deduplication with progress bar and filterable prioritized worklist.
- **Phase 13: Reversible Merge Store & Golden Record Review (`src/merge_store.py`, Screen 6)**
  - Implemented SQLite merge store (`data/merges.db`) with full JSON original snapshots.
  - Engineered golden record survivorship rules (recency, completeness, sentinel elimination).
  - Built Screen 6 console with side-by-side color/icon highlighting, critical hazard alerts (Blood Group and CNIC conflicts), 2-step merge confirmation, and 1-click **byte-for-byte exact Undo**.
- **Phase 14: Packaging, Hardening & Clean-Machine Test (`scripts/`, `README.md`)**
  - Built `scripts/check_imports.py` and `scripts/clean_machine_test.py` (`.bat`, `.sh`).
  - Authored `docs/ethics_statement.md` and `docs/usability_test.md`.
- **Viva Defense Specialization:** Golden record survivorship logic, reversible merge mechanics, preventing automation bias, end-to-end user workflow.

---

## 3. Estimated Work Distribution

| Project Domain | Member A | Member B | Member C |
|:---|:---:|:---:|:---:|
| **Data Generation & Quality Audit** | 80% | 15% | 5% |
| **Data Cleaning & Provenance Tracking** | 85% | 10% | 5% |
| **Candidate Blocking Challenge** | 10% | 80% | 10% |
| **Feature Extraction & Feature Selection** | 10% | 80% | 10% |
| **Model Training & Hyper-Parameter Tuning** | 5% | 85% | 10% |
| **Evaluation Metrics & Calibration** | 5% | 85% | 10% |
| **Clerk Explainability Engine** | 5% | 65% | 30% |
| **Streamlit Web Application (Screens 1–6)** | 10% | 15% | 75% |
| **SQLite Merge Store & Reversible Undo** | 5% | 15% | 80% |
| **Packaging, Documentation & Clean-Machine Testing** | 33% | 33% | 34% |
| **OVERALL PROJECT CONTRIBUTION** | **33.3%** | **33.3%** | **33.3%** |

---

## 4. Student Signature & Integrity Declaration

We hereby certify that the work documented in this repository represents the original collaborative effort of our team in accordance with COMSATS University Islamabad academic integrity regulations. All code, documentation, and synthetic data experiments were developed by the team.

| Member Name | Student ID | Signature | Date |
|---|---|---|---|
| `[Member A Name]` | `[FA21-BCS-XXX]` | _______________________ | 2026-10-27 |
| `[Member B Name]` | `[FA21-BCS-YYY]` | _______________________ | 2026-10-27 |
| `[Member C Name]` | `[FA21-BCS-ZZZ]` | _______________________ | 2026-10-27 |
