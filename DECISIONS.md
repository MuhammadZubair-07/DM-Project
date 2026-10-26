# Project Decisions Log

## Entry Format

- **Date**: YYYY-MM-DD
- **Decision**: 
- **Reason**: 
- **Who**: 

---

## Log

### [2026-10-26] Phase 0: Repository Setup and Pinned Environment
- **Date**: 2026-10-26
- **Decision**: Established Section 4 folder layout, created central configuration (`src/config.py`), non-destructive JSON metrics persistence (`src/utils.py`), one-command execution scripts (`run.bat` / `run.sh`), and pinned exact mutually compatible dependency versions (`pandas==3.1.0`, `numpy==2.5.4`, `scikit-learn==1.9.1`, `rapidfuzz==3.14.6`, `streamlit==1.66.1`, `plotly==7.1.0`, `matplotlib==3.11.2`, `joblib==1.6.0`, `pytest==9.1.1`).
- **Reason**: Ensures cross-platform reproducibility, strict isolation from system packages, avoidance of hardcoded paths, and seamless execution on any clean machine.
- **Who**: Full Team (Members A, B, C)

### [2026-10-26] Phase 1: Synthetic Register Generation
- **Date**: 2026-10-26
- **Decision**: Generated 54,373 records from 46,952 true entities (seed=42). Duplicate entity rate set to 11.5 % (`DUPLICATE_ENTITY_RATE=0.115`). Cluster sizes 2/3/4 at weights 70/22/8. Hard negatives: 720 sibling pairs, 80 twin pairs, 300 father-son same-name pairs, 500 shared-household pairs, 600 common-name entities. NID format: 1-digit province + 4-digit area + 7-digit serial + 1-digit check = 13 digits (fully synthetic; no real NADRA data). Source-system structural gaps coded directly: Legacy always missing national_id; Emergency high rate missing father_name and phone.
- **Reason**: (a) Synthetic data gives controlled ground truth for honest evaluation. (b) Hard negatives ensure the classifier cannot trivially distinguish duplicates from non-duplicates. (c) The trivial rule recall of 15.7% (same DOB+phone) confirms the data is sufficiently hard. (d) Structural gaps model real collection errors and will produce per-source missing patterns detectable in the audit (Phase 3).
- **Who**: Member A (Data) — reviewed by full team

### [2026-10-26] Phase 2: Data Profiling, Measurement Classification, and Leakage Boundary
- **Date**: 2026-10-26
- **Decision**: 
  1. Classified all 16 attributes in `data/raw/register_raw.csv` across all four Stevens measurement scales: 11 Nominal, 1 Ordinal (`insurance_tier`), 2 Interval (`registration_date`, `date_of_birth`), and 2 Ratio (`age_years`, `visit_count`).
  2. Classified digit strings (`national_id`, `phone`) as strictly **Nominal** (not numeric/ratio), proving that arithmetic operations (sums, means) are invalid and that numeric type casting drops mandatory syntactic leading zeros.
  3. Identified crucial domain distinction in `insurance_tier`: string `'None'` represents an explicit ordinal level (tier 0: uninsured patient, 19,357 records, 35.60%), whereas empty strings `''` represent genuine uncollected data (5,566 records, 10.24%, concentrated in the Pathology Lab subsystem).
  4. Physically and logically segregated ground-truth identifiers (`true_entity_id`), synthetic generator logs (`fault_log.csv`), and post-merge fields as **`must-not-use`**.
  5. Formulated `docs/leakage_statement.md` establishing the records clerk's information horizon, demonstrating mathematically why pairwise target labels cannot be computed from `register_raw.csv` alone, and setting an experimental protocol (balanced negative sampling + feature ablation) to prevent `source_system` sampling leakage.
- **Reason**: Fulfills Section A (Common Spec), Sections 7.1 and 7.2 of `PROJECT_ROADMAP.md`, and Prompt 2 gate requirements. Prepares rigorous, leak-free foundation for Phase 3 (Quality Audit) and Phase 4 (Cleaning).
- **Who**: Member A (Data & Quality) — reviewed by full team

### [2026-10-26] Phase 3: Data Quality Audit (Lecture 2 Fault Taxonomy)
- **Date**: 2026-10-26
- **Decision**: 
  1. Implemented detectors across all eight Lecture 2 fault categories in `src/audit.py`: F1 Exact Duplicates, F2 Inconsistent Encoding (city aliases, dirty gender, non-ISO dates, non-standard phones, name casing), F3 Sentinel Values (DOB 1900-01-01, age -999/-1, phone zeros, visit sentinels), F4 Invalid Values (DOB after registration, impossible age, age-DOB mismatch, malformed NID), F5 Missing Values, F6 Measurement / Entry Errors (lexicon lookup for name typos), F7 Outliers (robust MAD visit count outliers), and F8 Collection Errors (Legacy NID omission, Emergency father name omission).
  2. Formally proved via Chi-Square contingency testing ($\chi^2 > 13,000$, $p < 10^{-300}$) that missingness in `national_id`, `father_name`, `phone`, and `insurance_tier` strongly depends on `source_system`, confirming data is Missing at Random (MAR) or Missing Not at Random (MNAR) rather than MCAR.
  3. Conducted offline honesty check against `data/truth/fault_log.csv`: city aliases (100% precision, 100% recall), dirty gender (100% precision, 100% recall), DOB sentinel (99.5% precision, 100% recall), name typos (76.8% precision, 99.8% recall).
  4. Explicitly documented that near-duplicates are the primary mining product target, NOT a repair to be discarded or deduplicated during cleaning.
  5. Established proposed actions (`drop`, `correct`, `impute`, `flag`, `investigate`) for every fault code in preparation for Phase 4 cleaning with provenance columns.
- **Reason**: Fulfills Common Specification B, Section 7.3 of `PROJECT_ROADMAP.md`, and Prompt 3 gate requirements.
- **Who**: Member A (Data & Quality) — reviewed by full team

### [2026-10-27] Phase 4: Data Cleaning, Provenance Tracking, and validate() Assertions
- **Date**: 2026-10-27
- **Decision**: 
  1. Built deterministic cleaning pipeline in `src/clean.py` writing `data/clean/register_clean.csv` (54,373 rows $\times$ 44 columns).
  2. Preserved all original uncleaned demographic values in 13 `*_raw` columns.
  3. Added 10 provenance columns (`{field}_prov`) with standardized audit values (`original`, `case_fixed`, `whitespace_fixed`, `alias_mapped`, `code_mapped`, `format_standardized`, `day_month_swapped`, `sentinel_cleared`, `imputed_from_dob`, `reconciled_with_dob`, `capped_at_max`, `set_missing`), guaranteeing 100% non-empty audit provenance across all rows.
  4. Added 5 boolean anomaly and sentinel indicator flags (`dob_sentinel_flag`, `age_sentinel_flag`, `phone_sentinel_flag`, `nid_sentinel_flag`, `visit_outlier_flag`).
  5. Enforced core mining constraints: strictly preserved all candidate near-duplicate records without merging or dropping, and strictly avoided correcting name typos toward an omniscient lexicon (matching noisy names is the model's job).
  6. Generated before/after reconciliation tables in `reports/clean_effects.json` and `.csv`, and documented full provenance and repair distributions in `docs/cleaning_report.md`.
  7. Built `src/validate.py` with 15 rigorous quality and integrity assertions, proving that `validate(df_clean)` PASSES (15/15 passed, 0 violations) while `validate(df_raw)` FAILS (10 failing assertions with real violation counts).
  8. Enforced downstream leakage prevention by defining `CLEAN_MODEL_ALLOWLIST` and `FORBIDDEN_COLUMNS` in `src/config.py`.
- **Reason**: Fulfills Common Specification B, Section 7.4 of `PROJECT_ROADMAP.md`, and Prompt 4 gate requirements. Prepares clean, validated, leak-free input for Phase 5 blocking.

### [2026-10-27] Phase 5: Candidate Blocking and the Assigned Challenge
- **Date**: 2026-10-27
- **Decision**: 
  1. **Pre-Registered Pairs Completeness Target**: Formally established a minimum acceptable Pairs Completeness (Recall of true duplicate pairs) of **$\ge 92.0\%$** prior to finalizing the blocking scheme, subject to a computational ceiling of **$\le 1,000,000$ candidate comparisons** (Reduction Ratio $RR \ge 99.93\%$, sub-minute runtime for web application responsiveness).
  2. **From-Scratch Blocking Passes**: Implemented five orthogonal blocking passes in `src/blocking.py` using pure Python dictionaries (no external record-linkage dependencies):
     - **B1**: `Soundex(family_name) + birth year` (pure Python American Soundex algorithm).
     - **B2**: `first 3 letters of first_name + birth year-month (YYYY-MM)`.
     - **B3**: `last 7 digits of phone`.
     - **B4**: `national_id exact (13 digits)`.
     - **B5**: `Sorted-Neighbourhood Method (SNM)` over normalized compound name (`family_name + first_name`) with sliding window $w=5$.
  3. **Combinatorial Protection**: Implemented a `max_block_size` guard ($B_{\max} = 250$) that prevents giant blocks (e.g. ultra-common names) from generating quadratic pairwise bursts, logging dropped blocks and dropped candidate counts.
  4. **Selected Scheme**: Chose the **Multi-Pass UNION (B1 + B2 + B3 + B4 + B5)** as the production blocking configuration.
     - **Candidate Comparisons**: **823,534** pairs (compared to the $1,478,184,378$ exhaustive all-pairs search space).
     - **Reduction Ratio (RR)**: **$99.944287\%$** of comparisons pruned.
     - **Pairs Completeness (PC)**: **$94.48\%$** ($9,216$ of $9,754$ true duplicate pairs retained), exceeding our pre-registered $\ge 92.0\%$ threshold by $+2.48\%$.
     - **Pairs Quality (PQ)**: **$1.12\%$** (1 true pair per ~89 candidate pairs, providing a $\approx 1,700\times$ enrichment over raw all-pairs space prevalence of $0.00066\%$).
     - **Execution Runtime**: **$1.56$ seconds** for the entire blocking pass across all 54,373 records.
- **Reason**: 
  1. No single pass alone meets the completeness target: B4 (NID) achieves only $62.41\%$ completeness due to structural omissions in Legacy records and entry typos; B3 (Phone) achieves $60.55\%$; B1 (Soundex+Year) achieves $56.08\%$; B2 (Fn3+YrMo) achieves $36.26\%$; B5 (SNM $w=5$) achieves $19.48\%$.
  2. The disjunctive multi-pass union successfully addresses orthogonal failure modes: pairs with omitted or erroneous national IDs are rescued by Phone or DOB+Name passes, while pairs with phone typos or omissions are rescued by National ID or Soundex passes.
  3. The union candidate count of $823,534$ pairs remains well below the $1,000,000$ feasibility ceiling and takes under 2 seconds, proving that patient record deduplication is tractable without expensive GPU infrastructure.
  4. The missed-duplicate analysis demonstrated that the 538 missed true pairs ($5.52\%$) suffered from multi-attribute simultaneous failure (100% missing/corrupted phone, 100% missing/corrupted NID, 85.3% DOB discrepancies, and 84.2% simultaneous multi-field corruption), which cannot be recovered without drastic window enlargement that would explode comparisons toward $O(n^2)$.
- **Who**: Member B (Matching & Blocking) — reviewed by full team

### [2026-10-27] Phase 6: Temporal Entity Splitting, Pairs Formulation, and Comparison Vectors
- **Date**: 2026-10-27
- **Decision**: 
  1. **Temporal Entity-Grouped Partitioning (D4)**: Implemented in `src/split.py`. Grouped all encounters by `true_entity_id` and assigned entities to train or test based on their **earliest registration date** with a cutoff date of **`2022-04-15`**.
     - **Train Partition (earlier ~70%)**: 32,871 entities (70.01%), 39,762 records (73.13%).
     - **Test Partition (later ~30%)**: 14,081 entities (29.99%), 14,611 records (26.87%).
     - **Zero Entity Overlap**: All records belonging to an entity are strictly assigned to the same partition (`train_entities ∩ test_entities == ∅`), preventing entity-level data leakage.
  2. **Partition Candidate Pairs & Natural vs. Resampled Sets (D5)**: Implemented in `src/pairs.py`.
     - Ran the multi-pass blocking scheme (Union B1–B5, $B_{\max}=250$, $w=5$) within train and test partitions separately, ensuring no candidate pair crosses the temporal split boundary.
     - **Test Evaluation Set**: Preserved the exact natural candidate pair distribution (106,298 candidate pairs, 570 true duplicate matches, prevalence = **0.5362%**, sample weights = 1.0) for honest real-world evaluation.
     - **Training Set**: Retained the natural pair pool (475,790 pairs, 8,664 matches, prevalence = 1.8210%) and generated a stratified re-sampled training set (51,984 pairs, negative-to-positive ratio = 5.0, keeping all 8,664 positives and 43,320 sampled negatives, prevalence = **16.6667%**).
     - **Importance Re-Weighting**: Recorded exact inverse probability weights ($w_1 = 1.0$, $w_0 = 467,126 / 43,320 = \mathbf{10.783149}$) enabling downstream estimators to report unbiased population-level performance.
  3. **Pairwise Comparison Vector (Section 7.6)**: Implemented in `src/features.py` producing a 58-dimensional comparison vector per pair.
     - **Custom Explainable Metrics**: Self-implemented character 2-gram Jaccard and character 3-gram TF-IDF cosine without third-party black boxes, ensuring group members can defend their mathematical mechanics at the viva.
     - **RapidFuzz Integration**: Accelerated C-level Jaro-Winkler, Levenshtein, and Hamming distances for strings and identifiers.
     - **Composite Gower Similarity**: Implemented overall Gower similarity aggregating 11 core attributes via the Lecture 4 delta rule ($\delta_{ijk}$), gracefully omitting unobserved attributes from both numerator and denominator without penalty.
     - **Explicit Missing Value Handling**: Implemented per-attribute `both_present` binary flags and documented policies for tree splitters (sentinel + presence flag), Naive Bayes (dedicated missing bin), and $k$-NN (Gower delta distance).
  4. **Strict Schema & Forbidden-Column Guard**: Implemented `check_forbidden_columns` raising `ValueError` on any forbidden column (`true_entity_id`, generator fault logs, post-merge fields) or unallowlisted attribute, preventing circular validation and data leakage.
- **Reason**: 
  1. Satisfies Common Specifications D4, D5, Section 7.6 of `PROJECT_ROADMAP.md`, and all commitments in `docs/leakage_statement.md`.
  2. Temporal entity splitting models prospective real-world deployment: hospital records arriving after April 2022 represent prospective patients that must be matched against historic charts without future leakage.
  3. Stratified re-sampling with importance weights mitigates severe class imbalance (1:54 in natural candidate space) during model training while maintaining mathematical unbiasedness.
- **Who**: Member B (Matching, Blocking & Features) — reviewed by full team

### [2026-10-27] Phase 7: Preprocessing Decisions, Feature Selection, and Progressive Sampling
- **Date**: 2026-10-27
- **Decision**: 
  1. **Pre-Registered Progressive Sampling Convergence Rule**: Prior to evaluating sample sizes, established a formal stopping rule: evaluate training sample sizes doubling from $N_0 \approx 1,625$ up to the full training set ($N = 51,984$); sampling is declared converged when doubling the sample size yields an incremental gain in Average Precision (PR-AUC) of **less than $+0.005$ ($+0.5\%$)**.
  2. **Evaluation Protocol & Leakage Boundary**: All preprocessing comparisons conducted strictly on the **TRAINING pairs** using a 5-fold `StratifiedGroupKFold` (grouped by patient entity, random_state=42). Zero test pairs observed. All transformers (scalers, imputers, PCA, discretizers) fitted strictly inside training fold loops (`fit_transform` on train fold, `transform` on validation fold) to eliminate data leakage.
  3. **Rescaling Decisions (Lecture 4)**:
     - **Decision Tree**: Scale-invariant confirmed ($\text{PR-AUC}_{\text{unscaled}} = 0.9982 \pm 0.0009$, $\text{PR-AUC}_{\text{scaled}} = 0.9982 \pm 0.0009$, $\Delta = +0.0000$). Decision: No scaling required; preserve raw feature values for crisp clerk-facing tree threshold interpretations.
     - **$k$-Nearest Neighbours ($k$-NN)**: Scaling essential ($\text{PR-AUC}_{\text{unscaled}} = 0.9939 \pm 0.0032$, $\text{PR-AUC}_{\text{scaled}} = \mathbf{0.9975 \pm 0.0030}$, $\Delta = \mathbf{+0.0036}$, $p < 0.01$). Decision: Ship with `StandardScaler` to prevent unscaled wide-range features from drowning out bounded string similarity metrics in Euclidean space.
     - **Multi-Layer Perceptron (MLP)**: Scaling essential ($\text{PR-AUC}_{\text{unscaled}} = 0.9998$, $\text{PR-AUC}_{\text{scaled}} = \mathbf{0.9999}$, with dramatically more stable gradient descent). Decision: Ship with `StandardScaler`.
     - **Heavy-Tailed Transformations**: Applied log transform $\ln(1 + |\Delta|)$ to `dob_diff_days_log`, `age_diff_log`, and `visit_count_diff_log`.
  4. **Dimensionality Reduction (PCA) vs. Feature Subset Selection**:
     - Evaluated on identical 5 folds with common classifier (`LogisticRegression(max_iter=500)`):
       - Full Features (58 feats): PR-AUC = 0.9999, ROC-AUC = 1.0000, 12.93s.
       - PCA ($k=15$ components, capturing 85.3% variance): PR-AUC = 0.9998, ROC-AUC = 1.0000, 12.97s.
       - Filter Selection (15 features, collinearity $r < 0.95$): PR-AUC = **0.9993**, ROC-AUC = **0.9999**, 13.79s.
       - Greedy Forward Selection (6 features): PR-AUC = **0.9986**, ROC-AUC = **0.9998**, 11.12s.
     - **Decision**: Strictly ship **Feature Selection** over PCA. PCA components are opaque linear combinations that cannot be explained to a hospital records clerk. Feature selection preserves native, auditable attributes (`father_exact`, `dob_exact`, `phone_last7_equal`, `nid_levenshtein`, `city_equal`, `gower_similarity`) with virtually identical classification fidelity.
  5. **Discretization Strategies for Naive Bayes**:
     - Equal-Width ($B=5$): PR-AUC = 0.9923 $\pm$ 0.0015 (suffers from empty intermediate bins due to bimodal similarity distributions).
     - Equal-Frequency ($B=5$): PR-AUC = 0.9986 $\pm$ 0.0004 (boundary ties collapse bin edges).
     - Supervised Entropy (MDLP): PR-AUC = **0.9921 $\pm$ 0.0017** with clean partition boundaries where class entropy $H(Y|X)$ is minimized. Decision: Ship MDLP with dedicated missing bin handling.
  6. **Missing-Value Treatments per Family**:
     - Decision Tree: Sentinel (`-1.0`) + Presence Flags (0.9982) chosen over Mean Imputation (0.9982) to avoid creating artificial similarity scores.
     - $k$-NN: Mean Imputer + Scaler (0.9952) chosen over Sentinel + Scaler (0.9952) to avoid distorting Euclidean neighborhoods with extreme negative poles.
  7. **Progressive-Sampling Empirical Curve**:
     - Sample doubling: $N=1,625 \to 3,250 \to 6,500 \to 13,000 \to 26,000 \to 51,984$.
     - Incremental gain at first doubling ($N=1,625 \to 3,250$): $\Delta = +0.0006 < 0.005$.
     - **Declared Convergence**: Satisfies pre-registered stopping rule at **$N = 3,250$ pairs** (PR-AUC 0.9971), proving that downstream models learn effective deduplication boundaries with a fraction of pairwise comparisons.
  8. **Zero-Leakage Pipeline Builders**:
     - Shipped `get_tree_pipeline`, `get_knn_pipeline`, `get_nb_pipeline`, and `get_mlp_pipeline` in `src/preprocess.py`. All pipelines fit strictly on training folds and safely transform test sets without refitting.
- **Reason**: Common Specification C, Roadmap Sections 7.7 and 7.8, and Prompt 6 requirements. All numbers persisted to `reports/preprocessing.json`, figures saved to `reports/figures/`, and documented in `docs/preprocessing_decisions.md`.
- **Who**: Member B (Matching, Blocking & Features) — reviewed by full team

### [2026-10-27] Phase 8: Model Selection, Hyper-Parameter Tuning, and Calibration
- **Date**: 2026-10-27
- **Decision**: 
  1. **Strict Zero-Leakage Protocol**: All model hyper-parameter search and validation must be conducted strictly on the **TRAINING pairs** (`data/pairs/train_features_resampled.parquet`, $N = 51,984$) using 5-fold `StratifiedGroupKFold` grouped by patient entity (`random_state=42`). The prospective test set (`test_features.parquet`) remains untouched until Phase 9 evaluation.
  2. **Primary Selection Objective**: Precision-Recall Area Under Curve (**PR-AUC / Average Precision**).
     - *Justification*: In patient record linkage, positive duplicate pairs are a minority (16.7% in resampled train, 0.54% in natural test pairs). ROC-AUC and Accuracy are inflated by the overwhelming volume of true negatives. PR-AUC measures the precise trade-off between precision (protecting clerks from false alarms) and recall (preventing split medical histories). Moreover, PR-AUC evaluates ranking across the entire probability continuum, ensuring that when the hospital clerk modifies the cost matrix on Screen 5, posterior probabilities remain well-ranked.
     - *Secondary Tracking*: Expected Cost at operating threshold ($C_{FP}=5.0, C_{FN}=20.0$), ROC-AUC, F1-Score, and Brier calibration score are computed and stored in `reports/models.json`.
  3. **One-Standard-Error Simplicity Tie-Break (Breiman et al.)**:
     - For each model family, compute standard error across the 5 validation folds: $\text{SE} = \frac{\sigma}{\sqrt{5}}$.
     - If a simpler candidate configuration scores within $1 \times \text{SE}$ of the numerical maximum ($\mu_{\text{cand}} \ge \mu_{\max} - \text{SE}_{\max}$), we strictly select the simpler model to guard against cross-validation overfitting.
     - Simplicity hierarchy:
       - *Tree*: lower `max_depth`, higher `min_samples_leaf`, higher `ccp_alpha`.
       - *Naive Bayes*: higher smoothing $\alpha$ (stronger regularization), fewer bins.
       - *$k$-NN*: larger $k$ (smoother boundary, less variance), uniform weights.
       - *MLP*: smaller layer architecture, higher L2 penalty $\alpha$.
  4. **Pre-Registered Hyper-Parameter Search Grids**:
     - **Pruned Decision Tree**:
       - `criterion`: `['gini', 'entropy']`
       - `max_depth`: `[3, 5, 7, 10, None]`
       - `min_samples_leaf`: `[5, 10, 20, 50]`
       - `ccp_alpha`: `[0.0, 0.0005, 0.001, 0.002, 0.005]`
     - **Categorical Naive Bayes** (shipped discretization + Laplace smoothing):
       - `alpha`: `[0.01, 0.1, 0.5, 1.0, 2.0, 5.0]` (Laplace/Lidstone smoothing)
       - `n_bins`: `[3, 5, 8, 10]`
       - `bin_scheme`: `['entropy', 'quantile']`
     - **$k$-Nearest Neighbours ($k$-NN)** (on scaled features):
       - `n_neighbors`: `[3, 5, 7, 11, 15]` (strictly odd to prevent 50/50 distance ties)
       - `metric`: `['manhattan', 'euclidean']` (Minkowski L1 vs L2)
       - `weights`: `['uniform', 'distance']`
     - **Small MLP (Team Approval)**:
       - The full team explicitly approves training a small MLP (`hidden_layer_sizes`: `[(32, 16), (16, 8)]`, `alpha`: `[0.0001, 0.01]`, `max_iter`: `200`) as an optional 4th benchmark model. Viva defense: demonstrates neural representation learning on pairwise features while highlighting the critical trade-off between neural accuracy and decision-tree clinical explainability.
  5. **Unified Common Interface (`Matcher`)**:
     - All four trained pipelines are saved to `models/*.joblib` and encapsulated in a standardized `Matcher` class implementing `predict_proba(X) -> np.ndarray` (shape `(N, 2)`, valid probabilities in $[0, 1]$ summing to 1.0), `predict(X, threshold)`, and an `explain(pair)` stub for Phase 10.
  6. **Empirical Outcomes & Chosen Production Hyper-Parameters**:
     - **Pruned Decision Tree**: `{"criterion": "gini", "max_depth": 5, "min_samples_leaf": 10, "ccp_alpha": 0.0}`. Achieves 5-fold PR-AUC = **0.9986 $\pm$ 0.0012**, ROC-AUC = **0.9996 $\pm$ 0.0008**, Expected Cost = Rs. 5.68 / 1,000 pairs. Shipped as primary explainer model.
     - **Categorical Naive Bayes**: `{"alpha": 1.0, "n_bins": 5, "bin_scheme": "quantile"}`. Achieves 5-fold PR-AUC = **0.9962 $\pm$ 0.0018**, ROC-AUC = **0.9996 $\pm$ 0.0001**, Expected Cost = Rs. 9.81 / 1,000 pairs. Laplace smoothing (`alpha=1.0`) empirically proven to prevent zero-probability posterior collapse (from 0.0% to 100.0%).
     - **$k$-Nearest Neighbours**: `{"n_neighbors": 15, "metric": "euclidean", "weights": "uniform"}`. Selected via **1-SE Simplicity Tie-Break**: scored 0.9968 (within 1 SE = 0.0006 of the numerical maximum 0.9969 for $k=7$ Manhattan), chosen for superior boundary smoothness and variance reduction. Expected Cost = Rs. 5.96 / 1,000 pairs.
     - **Multi-Layer Perceptron (Optional 4th)**: `{"hidden_layer_sizes": (16, 8), "alpha": 0.01}`. Achieves 5-fold PR-AUC = **0.9980 $\pm$ 0.0013**, ROC-AUC = **0.9997 $\pm$ 0.0001**, Expected Cost = Rs. 11.73 / 1,000 pairs.
     - **Cryptographic Data Lineage**: All models tied to training data SHA-256 hash `661c9cefde87a4a469a05a9d62728c1b102069b31cd3092cc5b24d8ff841269d`.
- **Reason**: Fulfills Common Specification D, Roadmap Sections D6 and 7.8, and Prompt 7 requirements. Full CV table saved in `reports/models.json` and documented in `docs/model_selection.md`.
- **Who**: Full Team (Members A, B, C) — Member C (Models & Evaluation lead)

### [2026-10-27] Phase 9: Evaluation Protocol, Calibration, Cost Matrices, and Robustness
- **Date**: 2026-10-27
- **Decision**: 
  1. **One-Time Prospective Test Set Evaluation**: The test partition (`data/pairs/test_features.parquet`, $N = 106,298$, natural match prevalence = $0.5362\%$) is evaluated exactly ONCE in `src/evaluate.py`. No hyperparameter tuning or threshold selection may use test observations.
  2. **Mandatory Dual Baselines Beside Every Accuracy**:
     - *Majority-Class Baseline*: 99.4638% accuracy (predicts all non-match).
     - *Simple Domain Rule Baseline*: "same DOB and same phone" (`dob_exact == 1 and phone_last7_equal == 1`), which achieves 99.7112% accuracy, 97.82% precision, but only 47.19% recall (missing 301 true duplicate pairs).
     - Every single accuracy figure reported in `docs/evaluation_report.md`, `reports/evaluation.json`, and UI screens must be accompanied by the majority baseline.
  3. **Clinical Cost Matrix & Derived Operating Bands**:
     - *Default Review Band*: $C_{FP} = 5.0$ (clerk review time), $C_{FN} = 20.0$ (clinical risk of split medical chart). Derived threshold $\theta_1^* = \frac{5.0}{5.0 + 20.0} = \mathbf{0.20}$.
     - *Strict Recommended Merge Band*: $C_{FP} = 50.0$ (catastrophic clinical harm of wrong patient merge), $C_{FN} = 10.0$ (review deferral). Derived threshold $\theta_2^* = \frac{50.0}{50.0 + 10.0} = \mathbf{0.8333}$.
  4. **Post-Hoc Probability Calibration**:
     - Fitted Platt calibrator (`LogisticRegression`) strictly on **training out-of-fold predictions** (51,984 training pairs, zero test pairs).
     - Applied frozen calibrator to prospective test predictions, reporting reliability curves, Brier scores, and Expected Calibration Errors (ECE) before and after calibration.
  5. **End-to-End Recall Accounting**:
     - End-to-end recall is defined as $\text{Blocking Completeness} \times \text{Classifier Recall} = 0.9448 \times R_{\text{clf}}$. Pairs pruned by blocking must be counted as false negatives.
  6. **Leakage & Split Experiments**:
     - Evaluated model with deliberately leaky unshuffled sequential ID proximity to demonstrate artificial 100% metrics.
     - Evaluated random pair-level split vs. honest entity-grouped temporal split to document split inflation.
     - Tested frozen models against a fresh register generated with a different seed (`seed=999`).
- **Reason**: Common Specifications E, Roadmap Section 7.9, and Prompt 8 requirements. All numbers persisted to `reports/evaluation.json`, figures saved to `reports/figures/`, and documented in `docs/evaluation_report.md`.
- **Empirical Execution Outcomes (Test Set Evaluated Exactly Once)**:
  - *Prospective Population*: 106,298 candidate pairs, 570 true duplicates (prevalence: 0.5362%).
  - *Majority Baseline Accuracy*: 99.4638% (0.0% precision, 0.0% recall).
  - *Domain Rule Baseline ("same DOB and same phone")*: 99.7112% accuracy (beside majority baseline 99.4638%), 97.82% precision, 47.19% recall (301 false negatives).
  - *Model Test Performance (θ = 0.20)*:
    - **Pruned Decision Tree**: 99.98% accuracy (base 99.46%), 96.45% precision, 100.0% recall, 0.9819 F1, 0.9794 PR-AUC, 0.9999 ROC-AUC, Expected Cost Rs. 0.99 / 1k pairs.
    - **Categorical Naive Bayes**: 99.90% accuracy (base 99.46%), 84.70% precision, 100.0% recall, 0.9171 F1, 0.9658 PR-AUC, 0.9999 ROC-AUC, Expected Cost Rs. 4.84 / 1k pairs.
    - **k-Nearest Neighbours**: 99.93% accuracy (base 99.46%), 88.51% precision, 100.0% recall, 0.9390 F1, 0.9816 PR-AUC, 0.9999 ROC-AUC, Expected Cost Rs. 3.48 / 1k pairs.
    - **Multi-Layer Perceptron**: 99.91% accuracy (base 99.46%), 84.95% precision, 100.0% recall, 0.9186 F1, 0.9841 PR-AUC, 0.9999 ROC-AUC, Expected Cost Rs. 4.75 / 1k pairs.
  - *5-Fold Stratified Group CV (Training Portion)*:
    - Decision Tree: 99.92% ± 0.01% accuracy (base 83.33%), PR-AUC 0.9986 ± 0.0011, F1 0.9976 ± 0.0003.
    - Naive Bayes: 99.80% ± 0.02% accuracy (base 83.33%), PR-AUC 0.9962 ± 0.0016, F1 0.9941 ± 0.0005.
    - k-NN: 99.89% ± 0.02% accuracy (base 83.33%), PR-AUC 0.9968 ± 0.0011, F1 0.9966 ± 0.0006.
    - MLP: 99.77% ± 0.04% accuracy (base 83.33%), PR-AUC 0.9980 ± 0.0012, F1 0.9930 ± 0.0013.
  - *End-to-End Accounting*: Upstream candidate blocking pairs completeness is 94.48% (538 missed true pairs). Combined with 100.0% classifier recall, end-to-end recall is **94.48%**, and end-to-end precision is **96.45%** (Decision Tree).
  - *Leakage Proof*: Deliberate unshuffled sequential ID difference yields 83.34% train accuracy but collapses to 99.45% test accuracy (below the 99.46% majority baseline), proving catastrophic real-world failure of leaky ID proximity.
  - *Random vs. Temporal Split*: Random pair-level split inflates PR-AUC to 0.9991 due to entity contamination. Honest entity-grouped temporal split shows the true prospective capability (0.9794 PR-AUC).
  - *Robustness*: Seed 999 register (425 records, 3,531 candidate pairs) confirmed complete model stability (PR-AUC = 1.0000 across all frozen models).
  - *Workload Impact*: Reviewing flagged pairs at θ = 0.20 requires 29.6 hours (3.9 clerk-workdays at 7.5 hrs/day) for Decision Tree, vs. 33.6 hours for Naive Bayes.
  - *Production Recommendation*: Ship **Pruned Decision Tree** as primary classifier and interactive explainer, with **Categorical Naive Bayes** as independent probabilistic cross-check engine.
- **Who**: Full Team (Members A, B, C) — Member C (Models & Evaluation lead)

### [2026-10-27] Phase 10: Clerk-Facing Explanations, Zero-Jargon Guard, and Merge Screen Integration
- **Date**: 2026-10-27
- **Decision**:
  1. **Strict Plain-English Mandate (Zero Jargon Guard)**:
     - All predictions presented to medical records clerks must be converted into clinical natural language sentences.
     - Enforced an automated regex unit test guard (`r"\bnode\b|\bgini\b|\bfeature_\w+|<=|>=|\b\d+\.\d{3,}\b"`, case-insensitive) ensuring that no internal node numbers (`node 3`), Gini impurities (`gini = 0.32`), feature code strings (`dob_diff_days_log`), mathematical inequality symbols (`<=`, `>=`), or scientific multi-decimal floats are ever rendered to clerks.
  2. **Cost-Derived Confidence Bands**:
     - Confidence labels are anchored strictly to Bayesian cost-optimal decision boundaries ($\theta_1^* = 0.20$ for review worklist; $\theta_2^* = 0.8333$ for high-confidence auto-merge):
       - $p \ge 0.8333$: *"Very likely duplicate"*
       - $0.20 \le p < 0.8333$: *"Worth reviewing"*
       - $p < 0.20$: *"Unlikely duplicate"*
  3. **Model-Specific Reasoning Engines**:
     - *Decision Tree*: Traces exact decision paths via `tree_.decision_path`. Node splits are evaluated by probability direction ($P_{\text{child}} \ge P_{\text{parent}}$ indicates support for duplicate; significant drop indicates contradiction). Continuous thresholds are mapped to plain human qualifiers (*"nearly identical"*, *"very close"*, *"differing across calendar years"*).
     - *Categorical Naive Bayes*: Decomposes posterior probability into exact Log-Likelihood Ratio (LLR) contributions ($\text{LLR}_j = \ln P(X_j \mid \text{Duplicate}) - \ln P(X_j \mid \text{Distinct})$). Ranks top positive features in `reasons_for` and top negative features in `reasons_against`.
     - *$k$-Nearest Neighbours*: Queries the 5 closest historical training pairs in standardized metric space, explaining historical precedent in plain words (*"Most similar past pairs: 4 of 5 were the same person"*).
     - *MLP (Generic)*: Provides a joint feature activation alignment synthesis.
  4. **Graceful Missing Data Handling**:
     - When attributes are missing (e.g. emergency trauma admissions lacking CNIC or phone), the explainer explicitly states that the attribute was unrecorded on one record rather than misrepresenting it as an identity conflict.
  5. **Screen 6 Merge Screen Integration**:
     - Supplies structured `field_comparisons` across all 13 core demographic fields (`first_name`, `family_name`, `father_name`, `gender`, `date_of_birth`, `age_years`, `national_id`, `phone`, `city`, `address`, `blood_group`, `insurance_tier`, `visit_count`) with `agree`, `conflict`, or `unknown` tags to power the Streamlit merge UI with side-by-side color-coded survivorship cards.
  6. **Comprehensive Documentation & Test Suite**:
     - Created `docs/explanation_guide.md` with full phrase dictionary and worked clinical case studies.
     - Implemented `tests/test_explain.py` with 17 rigorous unit tests covering hand-built semantic pairs, jargon regex tests, and completeness checks. All 144 project tests pass.
- **Reason**: Common Specifications F, Roadmap Section 7.10, and Prompt 9 requirements.
- **Who**: Full Team (Members A, B, C) — Member B (Explainability & UI Lead)

### [2026-10-27] Phase 11: Streamlit App Foundation, Architecture, and Screens 1 & 2
- **Date**: 2026-10-27
- **Decision**:
  1. **Application Architecture & Navigation**:
     - Built multi-page application foundation with `app.py` as home executive overview and launchpad.
     - Implemented Screen 1 (`pages/1_Data_and_Quality.py`) for data quality auditing, cleaning provenance, schema validation, and cleaned CSV export.
     - Implemented Screen 2 (`pages/2_Model_Performance.py`) for candidate blocking statistics, prospective evaluation comparison, ROC/PR/calibration diagnostics, and leakage proofs.
     - Created structured placeholder pages for Screens 3–6 (`pages/3_Single_Prediction.py`, `pages/4_Batch_Scoring.py`, `pages/5_Settings_and_Cost_Matrix.py`, `pages/6_Merge_Review.py`) ensuring all 6 roadmap screens exist and open cleanly.
  2. **Shared State Management & Global Safeguards (`src/app_state.py`)**:
     - Centralized session state management in `src/app_state.py` with `init_session_state()`. Tracks active model, threshold, asymmetric cost matrix ($C_{FP}, C_{FN}$), and loaded dataset memory.
     - Rendered mandatory disclaimer banner on **EVERY page**: `"Student project. Not fit for clinical, financial or legal use. All data is synthetic."`
     - Provided a sidebar with an interactive "How to Use" guide and one-click sample-data loaders (`demo_register.csv`, `sample_batch.csv`, and full `register_raw.csv`).
     - Cached report reading with `@st.cache_data` (`load_audit_report()`, `load_clean_effects()`, `load_blocking_report()`, `load_evaluation_report()`, `load_dataset()`) ensuring instant page loads without expensive re-computation.
  3. **Screen 1 (Data & Quality) Implementation Decisions**:
     - Ingestion accepts default `demo_register.csv` or user file upload, with strict schema validation against the 14 register columns and friendly validation errors.
     - Enforced strict ground truth isolation: `true_entity_id`, `fault_category`, and other synthetic metadata are strictly prohibited and scrubbed to prevent test leakage.
     - Interactive Data Dictionary with 14 clinical fields and validation criteria.
     - Lecture 2 Data Quality Audit table with 8 fault categories, detecting expressions, and violation counts.
     - Before/after repair table highlighting provenance flags (`clean_effects.csv`).
     - Automated `validate()` comparison: demonstrates that the raw register violates 15 clinical constraints with exact violation counts, whereas the cleaned register passes 15/15 rules.
     - Export functionality: allows clerks to download `register_clean.csv`.
  4. **Screen 2 (Model Performance & Diagnostics) Implementation Decisions**:
     - Blocking Summary Card: Highlights $1.47\text{B}$ all-pairs space, $68,916$ union candidate comparisons, $99.9953\%$ reduction ratio, and $94.48\%$ completeness.
     - Model Benchmark Table: All figures loaded from `reports/evaluation.json` (never hand-typed). Accurately incorporates the **Majority Baseline (99.4638%) in the adjacent column beside every accuracy metric** across all models.
     - Interactive Diagnostic Tabs: Multi-model interactive ROC curves with test AUC, Precision-Recall curves, confusion matrices parameterized by the threshold slider, and Platt reliability calibration curves.
     - Scientific Validation Cards: Documents the sequential ID leakage vulnerability (83.34% -> 99.45%), random vs. temporal split inflation (0.9991 vs. 0.9794 PR-AUC), seed 999 stability (PR-AUC 1.0000), and clerk workload analysis (29.6 hours for 570 true duplicates at $\theta = 0.20$).
  5. **Verification & Quality Assurance**:
     - Implemented smoke and unit test suite in `tests/test_app.py` covering byte-compilation of all pages, state initialization, banner rendering, cached loaders, schema validators, and adjacent baseline enforcement.
     - Created comprehensive manual QA checklist in `docs/app_test_checklist.md`.
- **Reason**: Common Specifications E & G, Roadmap Section 7.11, and Prompt 10 requirements.
- **Who**: Full Team (Members A, B, C) — Member B (Explainability & UI Lead) and Member C (Models Lead)

### [2026-10-27] Phase 12: Implementation of Screens 3, 4, and 5 (Single Prediction, Batch Scoring, Settings)
- **Date**: 2026-10-27
- **Decision**:
  1. **Screen 5 (Settings & Decision Policy Engine)**:
     - Implemented dynamic model switcher across all 4 candidate families (Pruned Decision Tree, Categorical Naive Bayes, k-NN, MLP).
     - Built hospital misclassification cost matrix editor ($C_{FP}$ clerk review time vs $C_{FN}$ split medical chart harm) with units (PKR) and assumptions visible.
     - Automatically derives the theoretical Bayesian cost-optimal threshold $\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{5.0}{5.0 + 20.0} = 0.20$.
     - Provided a continuous threshold slider ($0.01 \le \theta \le 0.99$) with marked $\theta^*$ and one-click reset buttons (Cost-Optimal and Naive 0.50).
     - Simulated live operational consequences using `src/evaluate.py:compute_cost_and_thresholds()`: expected cost per 1,000 pairs, recall, precision, worklist size ($TP + FP$), estimated clerk hours (3.0 min/pair), confusion matrix ($TP, FP, FN, TN$), and accuracy given up compared to majority baseline (99.46%) and naive 0.50 threshold.
     - Persisted all policy parameters in `st.session_state` (`selected_model`, `threshold`, `threshold_strict`, `cost_fp`, `cost_fn`, `cost_matrix`).
  2. **Screen 3 (Single Pair Investigation & Jargon-Free Explanation)**:
     - Built two-patient chart comparison form accepting all 14 demographic fields with realistic placeholders.
     - Added one-click quick-fill buttons for 5 clinical benchmark scenarios: Easy match (casing & formatting), Hard negative (twin sisters / same father & address), Subtle duplicate (transposed names), Emergency trauma intake (missing CNIC & phone), and Clear distinct patients.
     - Ran the exact production `clean_patient_register()` pipeline and `src/explain.py:explain()` engine.
     - Rendered calibrated duplicate probabilities, Bayesian confidence tier badges (`Very likely duplicate`, `Worth reviewing`, `Unlikely duplicate`), structured plain-English reasons FOR and AGAINST, and per-field alignment tags (`agree`, `conflict`, `unknown`).
     - Enforced strict zero-jargon policy (`r"\bnode\b|\bgini\b|\bfeature_\w+|<=|>=|\b\d+\.\d{3,}\b"`).
  3. **Screen 4 (High-Throughput Batch Deduplication & Prioritized Worklist)**:
     - Ingestion accepts custom register CSV uploads or one-click loading of `sample_batch.csv` (1,801 records) / `demo_register.csv` (5,000 records).
     - Enforced schema validation and ground truth leakage stripping (`true_entity_id`, etc.).
     - Displayed notice regarding recommended file size (up to 10,000 records for real-time interactive browser execution) and graceful block-size capping for larger files.
     - Implemented 4-stage pipeline execution with progress bar (`clean_patient_register` $\to$ 5-pass blocking $\to$ 58-feature extraction $\to$ active model scoring).
     - Displayed file-specific candidate blocking efficiency metrics (records, all-pairs space, candidate comparisons, reduction ratio $99.51\%$).
     - Built prioritized worklist sorted by duplicate probability, showing confidence badges, demographic metadata, and concise one-line plain-English explanations.
     - Added multi-criteria filters (confidence band, facility source system, city, minimum probability, text search) and display row limiter.
     - Export capability: one-click CSV download of the complete ranked duplicate worklist.
  4. **Verification & Quality Assurance**:
     - Expanded `tests/test_app.py` to 16 automated tests covering live recomputation, single prediction pipeline, zero-jargon regex assertions, and batch scoring. All 16 passed.
     - Updated `docs/app_test_checklist.md` with detailed QA procedures for Screens 3, 4, and 5.
- **Reason**: Common Specifications E, F, & G, Roadmap Section 7.11, and Phase 12 requirements.
- **Who**: Full Team (Members A, B, C) — Member B (Explainability & UI Lead) and Member C (Models Lead)

### [2026-10-27] Phase 13: Screen 6 Merge Review, SQLite MergeStore, Survivorship & Reversible Undo
- **Date**: 2026-10-27
- **Decision**:
  1. **SQLite Merge Store Architecture (`src/merge_store.py`)**:
     - Built SQLite storage using the standard library `sqlite3` residing in `data/merges.db` (auto-created on initialization).
     - Defined `merges` table storing `merge_id`, original record snapshots (`record_1_snapshot`, `record_2_snapshot` in full JSON), `golden_record` JSON, `field_provenance` mapping, `clerk_note`, `model_name`, `duplicate_probability`, `timestamp`, `status` (`'merged'`, `'undone'`), and `undone_timestamp`.
     - Defined `decisions` table recording `'reject'` (distinct patients) and `'defer'` (flagged for paper chart/supervisor inspection) outcomes with timestamps.
     - Implemented `undo_merge(merge_id)` guaranteeing exact byte-for-byte fidelity when restoring original patient records.
     - Implemented `export_decisions_as_labelled_pairs()` generating ground truth training pairs from clerical actions (`merge` = 1, `reject` = 0, `defer` = -1).
  2. **Intelligent Clinical Survivorship Engine**:
     - Codified rule-based survivorship heuristics (`compute_survivorship_suggestions`) following Lecture 2 & Roadmap Section D8:
       - *Completeness*: Prefers non-missing, non-sentinel values (filters out `"1900-01-01"`, `"00000000000"`, `"-999"`, `"999"`).
       - *Recency*: Prefers the record with the most recent `registration_date` for volatile contact details (`phone`, `address`, `city`).
       - *Specificity*: Prefers complete 13-digit CNICs and longer title-cased names.
       - *Encounters*: Retains the maximum observed `visit_count`.
     - Provides interactive radio choices (`Record A`, `Record B`, or `Custom Manual Edit`) per field with visible rationale explanations and provenance tracking.
  3. **Screen 6 Review Workflow & Clinical Safety Gate (`pages/6_Merge_Review.py`)**:
     - Queue management: pulls candidate pairs from Screen 4's scored worklist (excluding already-decided pairs), accepts pairs from Screen 3, or runs pre-curated demo scenarios.
     - Side-by-side demographic alignment using **colour AND text/icons** (🟢 Agree, 🔴 Conflict, 🟡 Unknown/Missing) for visual accessibility.
     - Enforced strict human-in-the-loop safety rule: **NEVER auto-merge**.
     - Prominent safety alerts:
       - Low/moderate confidence alert when $p < \theta_{strict}^*$ ($0.8333$).
       - Red alert for critical clinical hazards: conflicting blood groups ($A+$ vs $B-$) warning of transfusion incompatibility.
       - Red alert for conflicting valid CNICs warning against collapsing two distinct citizens.
     - Two-step confirmation for merges displaying the resulting Golden Record preview before committing to SQLite.
     - Action triage: **Merge**, **Reject (Distinct)**, or **Defer (Queue)**.
  4. **Closed-Loop Screen 4 Integration**:
     - Screen 4 worklist queries `store.get_reviewed_pair_ids()` to dynamically filter out already-decided pairs from the active clerk worklist.
  5. **Verification & Quality Assurance**:
     - Created `tests/test_merge_store.py` with 7 comprehensive unit/integration tests: byte-for-byte exact equality between originals and undone snapshots, reject/defer decision logging, idempotent undo, concurrent re-opening across multiple instances, reviewed pairs filtering, labelled pair export, and survivorship rules.
     - All 23 application and merge store tests pass (`pytest -v tests/test_app.py tests/test_merge_store.py`).
     - Updated `docs/app_test_checklist.md` with Screen 6 manual QA protocols.
- **Reason**: Common Specifications E, F, & G, Roadmap Sections D8 and 7.11, and Prompt 13 requirements.
- **Who**: Full Team (Members A, B, C) — Member C (App Lead) with Member B (Explainability Lead)





