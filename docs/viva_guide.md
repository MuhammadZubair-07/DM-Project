# Comprehensive Viva Voce Defense Guide & Examiner Master Script

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool (National Patient Master Index)  
> **Team:** 3 Students (Member A, Member B, Member C)  
> **Date of Evaluation:** October 2026  
> **Authoritative Baseline Reference:** All metrics cited derive strictly from [`reports/evaluation.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/evaluation.json), [`reports/blocking.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/blocking.json), and [`reports/audit.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/audit.json). Zero numbers are invented.

---

## Table of Contents
1. [Member A Viva Guide: Data Engineering, Quality Audit & Validation (25 Questions)](#1-member-a-viva-guide-data-engineering-quality-audit--validation)
2. [Member B Viva Guide: Candidate Blocking, Features, Models & Prospective Evaluation (25 Questions)](#2-member-b-viva-guide-candidate-blocking-features-models--prospective-evaluation)
3. [Member C Viva Guide: Web Architecture, Reversible Merge Store & Usability (25 Questions)](#3-member-c-viva-guide-web-architecture-reversible-merge-store--usability)
4. [Cross-Examination Exercise (10 Questions Per Member Across Other Domains)](#4-cross-examination-exercise)
5. [Rehearsed 12-Minute Viva Demonstration Script & Pre-Flight Checklist](#5-rehearsed-12-minute-viva-demonstration-script)

---

## 1. Member A Viva Guide: Data Engineering, Quality Audit & Validation

*Domain: Synthetic generation, Stevens measurement scales, Lecture 2 fault taxonomy, data cleaning, provenance tracking, and `validate()` assertions.*

### Level 1: Fundamentals & Measurement Theory
#### Q1: Which attribute measurement scales are present in your patient register? Cite examples.
> **Model Answer:**  
> In [`docs/data_dictionary.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/data_dictionary.md), our 16 attributes span all four Stevens measurement scales:
> 1. **Nominal (11):** `first_name`, `family_name`, `father_name`, `gender`, `national_id`, `phone`, `city`, `address`, `blood_group`, `source_system`, `record_id`.
> 2. **Ordinal (1):** `insurance_tier` (strictly ordered: `None` < `Bronze` < `Silver` < `Gold`).
> 3. **Interval (2):** `date_of_birth`, `registration_date` (calendar dates have no natural absolute zero; 2020 is not "twice" 1010).
> 4. **Ratio (2):** `age_years`, `visit_count` (both possess a true non-arbitrary zero point permitting meaningful ratios, such as 4 visits being twice 2 visits).

#### Q2: *[Trick Question]* Why is `national_id` (CNIC) classified as Nominal if it consists entirely of 13 digits?
> **Model Answer:**  
> Because CNIC is a symbolic identifier, not a quantitative quantity. Calculating arithmetic means, sums, or variances on CNICs is mathematically nonsensical. Furthermore, treating it as a numeric integer strips essential leading zeros (e.g. `01234...`), destroying its legal syntax. It permits only equality ($=$ or $\neq$) and string distance operations, making it strictly **Nominal**.

#### Q3: *[Trick Question]* Why is `date_of_birth` an Interval scale rather than Ratio?
> **Model Answer:**  
> Calendar dates have an arbitrary zero point (the Gregorian epoch 0001-01-01). You cannot say that a person born in 2000 was born "twice as late" as someone born in the year 1000. However, differences between dates are meaningful (subtraction yields elapsed days), which defines an **Interval** scale. In contrast, `age_years` is **Ratio** because 0 years is a natural physical origin (birth).

#### Q4: How is uncollected data distinguished from an uninsured patient in `insurance_tier`?
> **Model Answer:**  
> As documented in [`DECISIONS.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/DECISIONS.md#L31), the string `'None'` represents an explicit ordinal level (Tier 0: uninsured patient, 19,357 records, 35.60%), whereas empty strings `''` represent genuine uncollected missing data (5,566 records, 10.24%, concentrated in the Pathology Lab subsystem).

#### Q5: What is the exact size of your raw register and how many true entities does it contain?
> **Model Answer:**  
> In [`data/raw/register_raw.csv`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/data/raw/register_raw.csv), there are exactly **54,373 records** synthesized from **46,952 unique true human entities** using seed `42` in [`src/generate_data.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/generate_data.py).

---

### Level 2: Synthetic Generation & Leakage Boundaries
#### Q6: Why did your team build a synthetic register rather than using real hospital data or UCI Record Linkage?
> **Model Answer:**  
> Three reasons: First, real hospital data contains Protected Health Information (PHI) subject to strict medical confidentiality. Second, the UCI Record Linkage dataset ships only pre-calculated comparison scores and match labels without raw names, phones, or addresses, making it impossible to demonstrate custom blocking or Stevens measurement scales. Third, our synthetic generator gives us controlled ground truth (`data/truth/ground_truth.csv`) to prospectively evaluate blocking completeness and model recall honestly.

#### Q7: What Pakistani demographic distributions did you incorporate?
> **Model Answer:**  
> In [`src/name_data.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/name_data.py), names are sampled from frequency-weighted Pakistani male, female, and family lexicons (e.g. Muhammad, Tariq, Khan). CNICs strictly replicate the NADRA 13-digit format (`XXXXX-XXXXXXX-X`) with valid provincial administrative division prefixes (e.g. `35201` for Lahore, `42101` for Karachi). Mobile numbers use standard PTA prefixes (`0300`, `0321`, `0333`, `0345`).

#### Q8: What hard negatives did you generate to prevent trivial classification?
> **Model Answer:**  
> As recorded in [`DECISIONS.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/DECISIONS.md#L22), we injected 5 classes of hard negatives: 720 sibling pairs (same surname, address, phone, nearby DOBs), 80 twin pairs (same DOB and address, different first names), 300 father-son pairs sharing full names, 500 shared households, and 600 ultra-common name collisions. Without hard negatives, a trivial model matching on phone or surname achieves falsely high accuracy.

#### Q9: What is the leakage boundary in your dataset? Which columns must never be seen by the model?
> **Model Answer:**  
> As defined in [`docs/leakage_statement.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/leakage_statement.md) and enforced by `CLEAN_MODEL_ALLOWLIST` in [`src/config.py:L145`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/config.py#L145), `true_entity_id`, synthetic generator corruption logs (`fault_log.csv`), cluster size, post-merge golden IDs, and clerk decisions are strictly forbidden. The model can only evaluate demographic fields available to a records clerk during patient intake.

#### Q10: How did you prove that `record_id` does not leak duplicate status?
> **Model Answer:**  
> In [`src/generate_data.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/generate_data.py), we shuffled the entire register before assigning sequential IDs (`REC00001`, `REC00002`). In [`src/evaluate.py:L388`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/evaluate.py#L388), our leakage experiment proved that on an unshuffled register, an index difference rule ($|id_1 - id_2| \le 2$) achieves 99.9% accuracy, whereas on our shuffled register it achieves 0.0% predictive power.

---

### Level 3: Quality Audit & Fault Taxonomy (Lecture 2)
#### Q11: Name all 8 Lecture 2 fault categories you implemented and give the detecting expression for each.
> **Model Answer:**  
> In [`src/audit.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/audit.py):
> 1. **F1 Duplicate Rows:** `df.duplicated(subset=non_id_cols)` (0 exact rows).
> 2. **F2 Inconsistent Encoding:** Distinct profile regex matching city aliases (`LHR`, `karachi`: 2,324), non-ISO dates (20,802), and dirty genders (`Male`, `1`: 1,410).
> 3. **F3 Sentinel Values:** Set membership checks on `1900-01-01` (392), age `-999` (300), phone `0000000000` (183).
> 4. **F4 Invalid Values:** Parsing checks for future DOBs or DOB > registration date (154), impossible ages (635).
> 5. **F5 Missing Values:** `isna()` counts across columns.
> 6. **F6 Measurement / Entry Errors:** Lexicon lookup detecting typographical errors in names (1,178).
> 7. **F7 Outliers:** Robust Median Absolute Deviation (MAD > 3.5) on `visit_count` (315).
> 8. **F8 Collection Errors:** Per-source system null rates (Legacy Hospital system 100% missing CNIC).

#### Q12: How many total dirty records did your audit detect?
> **Model Answer:**  
> Exactly **19,773 faulty records** out of 54,373 ($36.37\%$), documented in [`reports/audit.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/audit.json).

#### Q13: How did you prove that missingness is NOT Missing Completely at Random (MCAR)?
> **Model Answer:**  
> In [`src/audit.py:L320`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/audit.py#L320), we ran a Chi-Square Contingency Test cross-tabulating `source_system` against missingness in `national_id`. The test yielded $\chi^2 = 13,842.19$ with $p < 10^{-300}$. Because Legacy systems structurally never collected CNICs (100% missing), missingness depends on the collection system, proving it is **Missing at Random (MAR)** or **MNAR**, not MCAR.

#### Q14: *[Honesty Check]* Did you evaluate your fault detectors against your synthetic generator's fault log?
> **Model Answer:**  
> Yes. In [`DECISIONS.md:L42`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/DECISIONS.md#L42) and [`docs/quality_audit.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/quality_audit.md), our offline verification against `data/truth/fault_log.csv` demonstrated 100% recall on city aliases and dirty genders, 100% recall on DOB sentinels, and 99.8% recall (76.8% precision) on name typos.

#### Q15: Why are near-duplicates in the register NOT repaired or removed during the data quality stage?
> **Model Answer:**  
> Because detecting and resolving near-duplicate patient records is the **core machine learning product objective**. Dropping or arbitrarily merging them during data cleaning would destroy the very duplicate pairs the model is designed to detect and link.

---

### Level 4: Cleaning, Provenance & Validation Assertions
#### Q16: *[Trick Question]* Why did you NOT correct name typos during data cleaning?
> **Model Answer:**  
> Because in real hospital master indexes, a data cleaning pipeline does not have access to an omniscient oracle telling it what the patient's name was supposed to be. If we "corrected" Tariq to Tarique using ground truth, we would commit data leakage. Matching corrupted, noisy strings is the pairwise classifier's responsibility, not the cleaner's.

#### Q17: What are provenance columns and how many did you create?
> **Model Answer:**  
> In [`src/clean.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/clean.py), we created **10 provenance columns** (`{field}_prov`) alongside **13 uncleaned demographic columns** (`{field}_raw`). Each provenance column records the exact transform applied: `original`, `case_fixed`, `alias_mapped`, `code_mapped`, `format_standardized`, `sentinel_cleared`, or `imputed_from_dob`.

#### Q18: What did you do with sentinel dates like `1900-01-01`?
> **Model Answer:**  
> We converted them to null (`pd.NA`) so they would not distort numerical distance calculations, and flagged them in `dob_sentinel_flag = True`. This preserves the fact that a sentinel was submitted without letting the date skew age or interval metrics.

#### Q19: Explain the before/after numbers in your cleaning report for city and gender.
> **Model Answer:**  
> In [`reports/clean_effects.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/clean_effects.json):
> - `city`: Reduced from 38 distinct dirty variants (e.g. `LHR`, `karachi`, `ISB`) down to exactly 13 canonical Pakistani metropolitan cities.
> - `gender`: Resolved 1,410 non-standard codes (`Male`, `Female`, `1`, `2`) down to the canonical set `{M, F, U}`.

#### Q20: How many validation assertions are implemented in `src/validate.py`?
> **Model Answer:**  
> Exactly **15 clinical assertions**, including non-null unique IDs, canonical gender and city, DOB range and non-sentinel checks, age-DOB consistency within 1 year, PTA mobile phone syntax (`^03\d{9}$`), NADRA CNIC syntax (`^\d{13}$`), and absence of forbidden ground-truth columns.

---

### Level 5: Edge Cases, Defensibility & Viva Traps
#### Q21: What happens when `validate()` is run on `register_clean.csv` versus `register_raw.csv`?
> **Model Answer:**  
> In [`src/validate.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/validate.py):
> - On `register_clean.csv`: **PASSED (15/15 passed, 0 violations)**.
> - On `register_raw.csv`: **FAILED (5/15 passed, 10 failing assertions)**, capturing 20,802 DOB errors, 2,324 city errors, 1,825 phone syntax errors, and 1,410 gender errors.

#### Q22: Why did you cap visit counts at the 99.9th percentile rather than deleting those rows?
> **Model Answer:**  
> Following Lecture 2 guidelines on outliers: high visit counts (e.g. chronic dialysis patients visiting 3 times a week) represent legitimate clinical phenomena. Blindly deleting them deletes vulnerable patients. Capping at the documented cap (500 visits) prevents gradient distortion while retaining the patient in the master index.

#### Q23: How does your cleaning pipeline handle day-month swapped dates (e.g. 05/12/1985 vs 12/05/1985)?
> **Model Answer:**  
> If day $\le 12$, date parsing is inherently ambiguous. When registration date or age renders the parsed date impossible (e.g. DOB occurring after registration), [`src/clean.py:L215`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/clean.py#L215) swaps day and month, re-validates, and tags `dob_prov = 'day_month_swapped'`.

#### Q24: What is the difference between `register_raw.csv`, `demo_register.csv`, and `sample_batch.csv`?
> **Model Answer:**  
> `register_raw.csv` is the complete 54,373-record hospital database. `demo_register.csv` is a quick-loading ~5,000-record subset used for Screen 1 demonstration. `sample_batch.csv` is a 500-record batch ingestion file used to test Screen 4 batch deduplication.

#### Q25: If an examiner asks you to prove that no ground truth leaked into cleaning, how do you prove it?
> **Model Answer:**  
> I open [`src/clean.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/clean.py) and show that `df_clean` takes only demographic columns. We then run `validate(df_clean)`, which explicitly executes Assertion 14: scanning `df.columns` against `FORBIDDEN_COLUMNS` (`true_entity_id`, `cluster_id`, `fault_log`) and asserting 0 violations.

---

## 2. Member B Viva Guide: Candidate Blocking, Features, Models & Prospective Evaluation

*Domain: 5-pass blocking challenge, 58 pairwise comparison features, PCA vs Feature Selection, MDL binning, 4 model families, prospective test set evaluation, cost-optimal thresholds, and explainability.*

### Level 1: Blocking Challenge & Combinatorial Tractability
#### Q26: What is the unblocked all-pairs comparison space for your register, and why is pairwise classification without blocking intractable?
> **Model Answer:**  
> For $n = 54,373$ records, the exhaustive all-pairs search space is $\frac{n(n-1)}{2} = \mathbf{1,478,184,378}$ comparisons (approx. 1.48 billion pairs). At 1 millisecond per pair, exhaustive comparison would take over **17 days of non-stop computation**. Candidate blocking is mandatory to make the search space computationally tractable.

#### Q27: Name your 5 blocking passes. What libraries did you use?
> **Model Answer:**  
> In [`src/blocking.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/blocking.py), we implemented all 5 passes **from scratch using pure Python dictionaries (zero external record linkage libraries)**:
> - **B1:** `Soundex(family_name) + birth year` (custom American Soundex algorithm).
> - **B2:** `first 3 letters of first_name + birth year-month (YYYY-MM)`.
> - **B3:** `last 7 digits of phone` (local subscriber number ignoring telecom prefixes).
> - **B4:** `national_id exact` (13 digits).
> - **B5:** `Sorted-Neighbourhood Method (SNM)` over normalized compound name with sliding window $w=5$.

#### Q28: What is your candidate Reduction Ratio and Pairs Completeness? Cite exact numbers.
> **Model Answer:**  
> In [`reports/blocking.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/blocking.json) and [`docs/blocking_report.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/blocking_report.md):
> - **All-Pairs Space:** $1,478,184,378$ pairs.
> - **Comparisons Evaluated:** **823,534** candidate pairs.
> - **Reduction Ratio (RR):** **$99.9443\%$** of comparisons pruned.
> - **Pairs Completeness (PC):** **$94.48\%$** ($9,216$ of $9,754$ true duplicates retained), exceeding our pre-registered $\ge 92.0\%$ target.
> - **Runtime:** $1.56$ seconds.

#### Q29: What is your Pairs Quality, and why is it only ~1.12%?
> **Model Answer:**  
> Pairs Quality is $\frac{\text{True Duplicate Pairs Found}}{\text{Candidate Pairs Evaluated}} = \frac{9,216}{823,534} = \mathbf{1.12\%}$. In raw all-pairs space, true duplicates represent only $0.00066\%$ of pairs. Blocking enriches duplicate density by over **$1,700\times$**, filtering out $99.94\%$ of pure non-matches while retaining $94.48\%$ of all true duplicates.

#### Q30: Why did 538 true duplicate pairs escape blocking?
> **Model Answer:**  
> Our missed-duplicate analysis in [`docs/blocking_report.md:L160`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/blocking_report.md#L160) proved that all 538 missed pairs suffered from simultaneous multi-attribute corruption: 100% had missing/corrupted CNICs, 100% had missing/corrupted phones, 85.3% had mismatched birth dates, and 84.2% had misspelled surnames. When every key attribute is corrupted simultaneously, the pair cannot generate a common blocking key.

---

### Level 2: Comparison Vectors, Rescaling & Preprocessing
#### Q31: How many pairwise comparison features did you extract, and what similarity metrics did you use?
> **Model Answer:**  
> In [`src/features.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/features.py), we extract **58 pairwise comparison features** matching the attribute types in Lecture 4: Jaro-Winkler and Levenshtein for names, character 2-gram Jaccard, character 3-gram TF-IDF cosine, date component matches, day-month swap indicators, log-transformed absolute differences, CNIC Hamming distance, address token Jaccard, ordinal insurance differences, and composite Gower similarity.

#### Q32: *[Self-Implementation]* Did you use a library for character n-gram Jaccard and TF-IDF cosine?
> **Model Answer:**  
> No. In [`src/features.py:L45-L95`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/features.py#L45-L95), we wrote character 2-gram Jaccard ($\frac{|A \cap B|}{|A \cup B|}$) and character 3-gram TF-IDF cosine ($\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$) from scratch using pure NumPy and standard Python collections, ensuring complete defensibility at the viva.

#### Q33: How does the Lecture 4 delta rule ($\delta_{ijk}$) work in your composite Gower similarity?
> **Model Answer:**  
> Gower similarity is $S_{ij} = \frac{\sum_k \delta_{ijk} s_{ijk}}{\sum_k \delta_{ijk}}$. Under the delta rule, $\delta_{ijk} = 1$ only if attribute $k$ is known and valid in both records; if either record has attribute $k$ missing, $\delta_{ijk} = 0$. This removes the missing attribute from both numerator and denominator, preventing patients from being penalized for uncollected fields.

#### Q34: Why do you rescale features for k-NN but NOT for Decision Trees?
> **Model Answer:**  
> As taught in Lecture 4, decision trees make orthogonal, axis-parallel splits based purely on rank order; monotonic scaling changes nothing about the split point. In contrast, $k$-NN relies on Euclidean/Manhattan distance. If unscaled, an attribute with range $[0, 36500]$ (e.g. DOB difference in days) contributes $(36500)^2$ to distance, completely drowning out string similarities bounded in $[0, 1]$. Rescaling is mandatory for distance-based estimators.

#### Q35: Dimensionality reduction (PCA) vs Feature Subset Selection: Which won and why?
> **Model Answer:**  
> In [`docs/preprocessing_decisions.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/preprocessing_decisions.md), both were compared on Decision Tree using identical 5-fold `StratifiedGroupKFold`:
> - **PCA (15 components):** PR-AUC = $0.9412$. Linear combinations destroy sparse match indicators and cannot be explained to a clerk.
> - **Feature Selection (Filter correlation + 0.95 collinearity prune + Greedy Forward Selection):** PR-AUC = **$0.9794$** across 24 selected features. Feature selection won on both predictive performance and clerk interpretability.

---

### Level 3: Modeling, Pre-Registration & Discretization
#### Q36: *[Trick Question]* What does a zero probability do to Naive Bayes, and how did you prevent it?
> **Model Answer:**  
> In Naive Bayes, the posterior probability is proportional to $P(Y) \prod_i P(X_i | Y)$. If an unobserved feature bin has conditional probability $P(X_k | Y) = 0$, multiplying by zero completely zeroes out the entire posterior probability, regardless of overwhelming evidence from all other features. We applied **Laplace smoothing ($\alpha = 1.0$)**, initializing every bin count to 1 so probabilities remain strictly positive.

#### Q37: How did you discretize continuous similarity scores for Categorical Naive Bayes?
> **Model Answer:**  
> In [`src/preprocess.py:L480`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/preprocess.py#L480), we implemented Fayyad & Irani's **Minimum Description Length (MDL) supervised entropy discretizer**. It recursively chooses threshold boundaries that maximize class information gain, stopping when the message length required to encode the split exceeds the entropy reduction. It achieved PR-AUC of **$0.9658$**, outperforming equal-width ($0.9120$) and equal-frequency ($0.9385$) binning.

#### Q38: What four model families did you implement?
> **Model Answer:**  
> In [`src/models.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/models.py):
> 1. **Tree Family:** Pruned Decision Tree (`DecisionTreeMatcher`, $ccp\_alpha=0.001$, `max_depth=6`).
> 2. **Probabilistic Family:** Categorical Naive Bayes with Laplace smoothing (`NaiveBayesMatcher`).
> 3. **Metric / Instance Family:** $k$-Nearest Neighbours on standardized features (`KNNMatcher`, $k=15$, distance weighting).
> 4. **Neural Network Family:** Multi-Layer Perceptron (`MLPMatcher`, architecture 64-32, ReLU).

#### Q39: What was your pre-registered hyperparameter tuning procedure?
> **Model Answer:**  
> Pre-registered in [`DECISIONS.md:L85`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/DECISIONS.md#L85): 5-fold `StratifiedGroupKFold` grouped by entity on training pairs only, fixed seed 42, optimizing for PR-AUC / expected cost. When models were within one standard error (1-SE) of the best CV score, we selected the simpler model (lower depth / higher alpha) to prevent overfitting.

#### Q40: What was the temporal train/test split cutoff date, and why is a temporal split required?
> **Model Answer:**  
> Cutoff date: **`2022-04-15`** ([`src/split.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/split.py)). Entities with earliest registration on or before this date went to train (70.01% entities, 39,762 records); later entities went to test (29.99% entities, 14,611 records). In clinical deployment, records arrive sequentially over calendar time. Random splitting allows past and future encounters of the same entity to cross splits, causing artificial entity memorization.

---

### Level 4: Prospective Evaluation & Baseline Adjacency (The 20-Mark Section)
#### Q41: *[Trick Question]* Why is 98% or 99% accuracy poor in this application?
> **Model Answer:**  
> Because in our prospective candidate test population of $106,298$ pairs, true duplicates number only $570$ ($0.5362\%$). Non-matches constitute **$99.4638\%$** of the data. A useless "dumb" model that blindly predicts *'never a duplicate'* achieves **$99.46\%$ accuracy** while finding zero duplicates ($0\%$ recall). Accuracy is overwhelmingly dominated by the majority negative class; an accuracy figure without its baseline is meaningless.

#### Q42: What is the Majority-Class Baseline accuracy on your prospective test set?
> **Model Answer:**  
> Exactly **$99.4638\%$** ($99.46\%$), calculated as $\frac{106,298 - 570}{106,298} = \frac{105,728}{106,298}$. It is displayed directly beside every single accuracy figure in our benchmark tables and README.

#### Q43: What are the exact test metrics for your primary model (Pruned Decision Tree) at $\theta^* = 0.20$?
> **Model Answer:**  
> In [`reports/evaluation.json`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/reports/evaluation.json):
> - **Test Accuracy:** **$99.9802\%$** (beside Majority Baseline: **$99.4638\%$**)
> - **Precision:** **$96.4467\%$** ($96.45\%$)
> - **Classifier Recall:** **$100.0000\%$** ($100.0\%$, $570 / 570$ captured)
> - **End-to-End Recall:** **$94.4843\%$** ($94.48\%$, accounting for upstream blocking)
> - **F1-Score:** **$0.9819$**
> - **PR-AUC:** **$0.9794$**
> - **ROC-AUC:** **$0.9999$**
> - **Expected Cost:** **Rs. 0.99 per 1,000 pairs**

#### Q44: *[Trick Question]* Why is your operating threshold set to 0.20 instead of the standard 0.50?
> **Model Answer:**  
> In clinical healthcare, false negatives (missed duplicates leaving split medical records, missed allergies, and redundant CT scans) are far more harmful than false positives (clerk spending 3 minutes reviewing a false alarm). Our hospital cost matrix sets $C_{FP} = \text{Rs. } 5.0$ and $C_{FN} = \text{Rs. } 20.0$. By Bayesian decision theory:
> $$\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}} = \frac{5.0}{5.0 + 20.0} = \mathbf{0.20}$$
> Leaving the threshold at 0.50 naively assumes false alarms and split charts carry equal clinical harm, which is medically indefensible.

#### Q45: What is "accuracy given up", and what did your primary model give up at $\theta^* = 0.20$?
> **Model Answer:**  
> Accuracy given up is $\text{Acc}(\theta = 0.50) - \text{Acc}(\theta^* = 0.20)$. For our Pruned Decision Tree, operating at $\theta^* = 0.20$ captured $100\%$ of true duplicates with only 21 false alarms, giving up exactly **$0.0000\%$ accuracy** while cutting expected clinical misclassification costs from Rs. 4.84 down to **Rs. 0.99 per 1,000 pairs**.

---

### Level 5: Calibration, Explainability & Viva Defense
#### Q46: How did you calibrate predicted probabilities, and why is Platt calibration necessary?
> **Model Answer:**  
> Uncalibrated decision tree leaf frequencies or distance ratios often produce over-confident or distorted probabilities. We fitted a Platt calibrator (logistic sigmoid) on **5-fold training out-of-fold predictions**. On the test set, Brier score improved, aligning predicted probabilities with empirical match frequencies so that pairs flagged at $p \ge 0.20$ truly have $\ge 20\%$ duplicate probability.

#### Q47: How does your explanation engine generate plain-English reasons without exposing node IDs or Gini?
> **Model Answer:**  
> In [`src/explain.py:L330`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/explain.py#L330), we trace `clf.decision_path()`. Instead of printing "Node 4: feature_12 $\le$ 0.15", our feature-to-phrase dictionary translates feature indices and thresholds into clinical facts: "Same date of birth", "First names are nearly identical (Muhammad / Mohammad)", "National ID is missing on one record", or "Phone numbers differ".

#### Q48: How do you explain Naive Bayes predictions to a records clerk?
> **Model Answer:**  
> In [`src/explain.py:L480`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/explain.py#L480), we compute the log-likelihood ratio contribution for each feature: $\log \frac{P(X_i | Y=1)}{P(X_i | Y=0)}$. Features with positive LLRs become bullets in `reasons_for` (e.g. *"Matching subscriber phone digits (+2.8 log-odds)"*), while features with negative LLRs become bullets in `reasons_against` (e.g. *"Different cities (-1.9 log-odds)"*).

#### Q49: What is the Simple Domain Rule Baseline, and how did your models compare against it?
> **Model Answer:**  
> The rule baseline is "Same DOB and Same Phone". On the prospective test set, it achieves $99.71\%$ accuracy (beside baseline $99.46\%$) and $97.82\%$ precision, but suffers an unacceptable **$47.19\%$ recall** (misses more than half of all duplicates due to phone typos or uncollected numbers). Our Decision Tree captures **$100.0\%$ test recall** ($94.48\%$ end-to-end), proving machine learning is essential.

#### Q50: How do you prove that your test set was touched only once?
> **Model Answer:**  
> In [`DECISIONS.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/DECISIONS.md), hyperparameter grids, feature selection, discretization bins, and calibrators were all finalized and committed on training folds before `src/evaluate.py` was executed. The test features file (`data/pairs/test_features.parquet`) is read-only and evaluated strictly once in batch.

---

## 3. Member C Viva Guide: Web Architecture, Reversible Merge Store & Usability

*Domain: Streamlit multi-screen application, SQLite merge store, reversible byte-for-byte undo, survivorship rules, automation bias defense, clean-machine packaging, and clinical ethics governance.*

### Level 1: Application Architecture & State Management
#### Q51: How is session state shared across your multi-page Streamlit application?
> **Model Answer:**  
> In [`src/app_state.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/app_state.py), `init_session_state()` initializes global keys: `selected_model` (default `'decision_tree'`), `threshold` ($0.20$), `threshold_strict` ($0.8333$), `cost_matrix`, and `active_dataset`. Any adjustment made on Screen 5 (Settings) immediately propagates to Screens 2, 3, 4, and 6 without session reset.

#### Q52: What caching mechanisms did you implement to ensure responsive performance?
> **Model Answer:**  
> In [`src/app_state.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/app_state.py), we use `@st.cache_data` on file loaders (`load_demo_register`, `load_evaluation_metrics`, `load_blocking_metrics`) and `@st.cache_resource` on model loaders (`get_matcher`). Once loaded from disk, switching between pages takes under 50 milliseconds.

#### Q53: What does Screen 1 demonstrate to a clinical auditor?
> **Model Answer:**  
> Screen 1 ([`pages/1_Data_and_Quality.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/1_Data_and_Quality.py)) demonstrates data ingestion, the complete Lecture 2 fault audit table (8 categories, 19,773 records), the before/after repair table, the execution of `validate()` proving clean data passes 15/15 while raw data fails 10/15, and provides a direct download link for `register_clean.csv`.

#### Q54: What does Screen 2 demonstrate regarding evaluation honesty?
> **Model Answer:**  
> Screen 2 ([`pages/2_Model_Performance.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/2_Model_Performance.py)) displays the prospective model benchmark table where the **Majority-Class Baseline ($99.46\%$) is rendered directly in the adjacent column beside every accuracy figure**. It also renders multi-model ROC curves, PR curves, Platt calibration curves, and leakage proofs.

#### Q55: How does Screen 5 recompute performance outcomes live?
> **Model Answer:**  
> In [`pages/5_Settings_and_Cost_Matrix.py:L195`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/5_Settings_and_Cost_Matrix.py#L195), whenever the user adjusts the cost inputs ($C_{FP}, C_{FN}$) or the threshold slider, the page dynamically invokes `compute_cost_and_thresholds()` over prospective test predictions, updating expected cost, recall, precision, worklist size, clerk review hours, and the confusion matrix in real time.

---

### Level 2: SQLite Merge Store & Reversible Undo
#### Q56: *[Core Requirement]* How does your merge store guarantee byte-for-byte exact restoration upon Undo?
> **Model Answer:**  
> In [`src/merge_store.py:L110`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py#L110), before any merge is committed, the complete unmerged dictionaries of both records are serialized to JSON strings and stored in the SQLite database (`data/merges.db`). When the clerk clicks "Undo", [`undo_merge()`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py#L180) extracts the exact raw JSON snapshots and updates the status to `'undone'`. In [`tests/test_merge_store.py:L110-L122`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/tests/test_merge_store.py#L110-L122), we assert that the byte-level UTF-8 serialization of the restored records exactly equals the original records.

#### Q57: Where is the merge store physically located?
> **Model Answer:**  
> In an embedded SQLite database at [`data/merges.db`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/data/merges.db), created automatically if missing via standard library `sqlite3` without external database servers.

#### Q58: What database tables exist in `merges.db`?
> **Model Answer:**  
> In [`src/merge_store.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py):
> 1. `merges`: Stores `merge_id`, record IDs, full JSON snapshots of Record A, Record B, and Golden Record, field provenance JSON, clerk notes, model name, probability, status (`'merged'` or `'undone'`), and timestamps.
> 2. `decisions`: Stores `decision_id`, record IDs, decision type (`'reject'` or `'defer'`), clerk notes, and timestamps.

#### Q59: What is the clinical and technical difference between Reject and Defer on Screen 6?
> **Model Answer:**  
> - **Reject:** Formally declares that Record A and Record B belong to two distinct human beings (e.g. twin sisters). The pair is removed from active worklists and permanently logged as a negative pair.
> - **Defer:** A temporary hold when the clerk lacks sufficient physical documentation to decide. The pair remains logged in the audit trail but can be re-surfaced when additional intake forms arrive.

#### Q60: What golden record survivorship rules did you engineer?
> **Model Answer:**  
> In [`src/merge_store.py:L260`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py#L260), `compute_survivorship_suggestions()` enforces:
> 1. **Sentinel Elimination:** Non-sentinel values always supersede sentinels (`1900-01-01`, `-999`).
> 2. **Completeness:** Non-empty fields supersede nulls.
> 3. **Recency:** When both records contain valid conflicting data (e.g. phone or address), the value from the record with the more recent `registration_date` is selected by default.
> 4. **Human Override:** The clerk can override any field via radio buttons or text edits before confirming.

---

### Level 3: Usability, Ethics & Clinical Safety
#### Q61: What clinical disasters occur if two distinct patients are erroneously merged (False Positive)?
> **Model Answer:**  
> As documented in [`docs/ethics_statement.md:L25-L43`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/ethics_statement.md#L25-L43):
> 1. **Fatal ABO Transfusion Mismatch:** Infusing incompatible blood products due to mixed records causes acute intravascular hemolysis and death.
> 2. **Contaminated Medication/Allergy Charts:** Erasing documented penicillin anaphylaxis or withholding critical antibiotics.
> 3. **HIPAA / Privacy Violations:** Exposing psychiatric, reproductive, or infectious disease diagnoses to another citizen sharing the chart.

#### Q62: What clinical harms occur if duplicate records are left split (False Negative)?
> **Model Answer:**  
> Fragmented care: Patients receive redundant repeat ionizing radiation (duplicate CT scans), doctors miss prior surgical implants or baseline renal failure, and adverse drug-drug interactions go undetected.

#### Q63: *[Crucial Governance]* What is "automation bias", and how does your interface combat it?
> **Model Answer:**  
> Automation bias is the psychological tendency for busy hospital clerks to unthinkingly rubber-stamp algorithmic recommendations. We combat it through 4 strict UI defenses:
> 1. **Zero Auto-Merge:** The system can never merge records automatically.
> 2. **Forced Deliberation:** Explanations show concrete plain-English reasons **FOR** and **AGAINST**, forcing clerks to weigh conflicting evidence.
> 3. **Two-Step Confirmation Dialog:** Merging requires entering a clerk rationale note and confirming a modal warning.
> 4. **Red Alerts:** Conflicting blood groups or CNICs trigger unmissable red alert warnings.

#### Q64: How does Screen 6 handle conflicting blood groups ($A+$ vs $B-$)?
> **Model Answer:**  
> In [`pages/6_Merge_Review.py:L145`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/6_Merge_Review.py#L145), an automated clinical hazard detector compares blood group strings. If both are non-empty and differ, it renders a bold red warning: `🚨 CRITICAL CLINICAL HAZARD: Recorded blood groups conflict! Merging these charts could lead to fatal transfusion errors.`

#### Q65: How does Screen 6 handle conflicting valid National IDs?
> **Model Answer:**  
> In [`pages/6_Merge_Review.py:L156`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/6_Merge_Review.py#L156), if both records have valid 13-digit CNICs that differ, it renders a critical warning: `🚨 IDENTITY COLLISION DETECTED: Records have distinct valid National IDs. Do NOT merge distinct citizens.`

---

### Level 4: Usability Testing, Packaging & Clean-Machine Startup
#### Q66: What is in `docs/usability_test.md`, and how did you design it for a stranger?
> **Model Answer:**  
> In [`docs/usability_test.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/usability_test.md), we established an unassisted observer protocol with 4 tasks: (1) Find the worklist on Screen 4, (2) Compare a twin sister pair on Screen 3, (3) Update the cost matrix on Screen 5, and (4) Execute a merge and undo on Screen 6. It includes timed checkpoints, friction recording boxes, and the standard 10-item System Usability Scale (SUS) survey (industry target $> 68/100$).

#### Q67: How does your application achieve single-command startup on any clean machine?
> **Model Answer:**  
> Via [`run.bat`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/run.bat) (Windows) and [`run.sh`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/run.sh) (Linux/macOS). The script checks if `.venv` exists; if missing, it creates it using `python -m venv .venv`, activates it, installs pinned packages from `requirements.txt`, and launches `python -m streamlit run app.py`. All models (`models/*.joblib`) and data (`data/raw/`) ship inside the repo; zero external downloads are required.

#### Q68: What does `scripts/clean_machine_test.py` do?
> **Model Answer:**  
> In [`scripts/clean_machine_test.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/scripts/clean_machine_test.py), it creates an isolated sandbox virtualenv in `$TEMP`, installs `requirements.txt`, executes the 167-test pytest suite, runs headless smoke compilation across all 7 Streamlit pages, and purges the temp directory, guaranteeing cross-platform portability.

#### Q69: What is the total repository size, and how did you keep it reasonable?
> **Model Answer:**  
> The total non-virtualenv repository size is **72.93 MB** across 127 files. The largest file is `data/clean/register_clean.csv` (21.68 MB), followed by `models/knn.joblib` (13.23 MB). Pair comparison vectors are compressed using GZIP (`.csv.gz`) and Apache Parquet (`.parquet`), keeping the repository comfortably beneath GitHub's limits.

#### Q70: Why is the student project disclaimer required on every single screen?
> **Model Answer:**  
> Rendered via `render_disclaimer_banner()` in [`src/app_state.py:L82`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/app_state.py#L82): `"⚠️ Student project. Not fit for clinical, financial or legal use. All data is synthetic."` It is a mandatory medical informatics governance safeguard preventing accidental clinical use.

---

### Level 5: Synthesis, Viva Traps & System Resilience
#### Q71: What happens if an examiner uploads a malformed CSV with missing demographic columns on Screen 4?
> **Model Answer:**  
> In [`pages/4_Batch_Scoring.py:L155`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/4_Batch_Scoring.py#L155), the schema validator catches missing required columns (`first_name`, `family_name`, `gender`, `date_of_birth`), halts execution with `st.error()`, and displays a helpful list of missing columns rather than throwing an unhandled Python traceback.

#### Q72: What happens if an uploaded CSV contains a forbidden column like `true_entity_id`?
> **Model Answer:**  
> In [`pages/4_Batch_Scoring.py:L166`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/4_Batch_Scoring.py#L166), our automated Leakage Guard detects forbidden columns from `CFG.FORBIDDEN_COLUMNS`, displays a yellow warning `st.warning("Leakage Guard: Stripped forbidden ground-truth columns")`, and programmatically drops them before passing data to the model.

#### Q73: How does Screen 3 handle quick-load clinical presets?
> **Model Answer:**  
> In [`pages/3_Single_Prediction.py:L60-L100`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/pages/3_Single_Prediction.py#L60-L100), buttons load benchmark clinical cases directly into the session state form: (1) Exact Duplicate, (2) Real-World Match with Typos & Missing CNIC, (3) Hard Negative Twin Sisters, (4) Hard Negative Father & Son, and (5) Non-Match Different Citizens.

#### Q74: If an examiner asks to see the database schema of `data/merges.db`, where do you show it?
> **Model Answer:**  
> In [`src/merge_store.py:L40-L80`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/merge_store.py#L40-L80), showing the `CREATE TABLE IF NOT EXISTS merges` and `decisions` statements, complete with foreign keys, status constraints, and timestamping.

#### Q75: How do you demonstrate that an Undo operation updates the audit log without losing historical traceability?
> **Model Answer:**  
> When a merge is undone, the database row is **not deleted**. Instead, its `status` is updated from `'merged'` to `'undone'` and `undone_timestamp` is populated. Screen 6 displays the complete historical audit log showing both the original merge and its subsequent revocation.

---

## 4. Cross-Examination Exercise
*(10 Questions Every Member Must Be Able to Answer About the Other Members' Work)*

### Questions for Member A About Member B & C's Work
1. **[About B]** What is your candidate reduction ratio, and what does it mean? *(99.9443%; out of 1.48 billion pairs, blocking pruned over 99.94%, evaluating only 823,534 pairs).*
2. **[About B]** What is the Majority Baseline on the test set, and what was it? *(99.46%; the accuracy achieved by blindly guessing non-match on all pairs).*
3. **[About B]** What is the operating threshold and why was it chosen as 0.20? *(Derived from cost matrix: $5 / (5 + 20) = 0.20$; missing a duplicate is 4× worse than reviewing a false alarm).*
4. **[About B]** What are the four model families implemented? *(Decision Tree, Categorical Naive Bayes, $k$-NN, MLP).*
5. **[About B]** Why did Feature Selection beat PCA? *(PR-AUC 0.9794 vs 0.9412; preserved sparse pairwise signals and interpretability).*
6. **[About C]** Where are merges stored? *(SQLite database at `data/merges.db`).*
7. **[About C]** What happens when you click "Undo" on Screen 6? *(Restores both original records byte-for-byte from raw JSON snapshots).*
8. **[About C]** What happens if two records have conflicting blood groups on Screen 6? *(A critical red alert fires warning of fatal hemolytic transfusion risk).*
9. **[About C]** How do you start the application from scratch? *(`run.bat` on Windows or `./run.sh` on Linux/macOS).*
10. **[About C]** What is the mandatory disclaimer banner and where does it appear? *(Student project disclaimer rendered at the top of every screen).*

### Questions for Member B About Member A & C's Work
1. **[About A]** What measurement scales are in the register? *(All four: Nominal, Ordinal, Interval, Ratio).*
2. **[About A]** Why is CNIC nominal and DOB interval? *(CNIC arithmetic is invalid and numeric casting drops leading zeros; DOB has an arbitrary calendar epoch zero).*
3. **[About A]** Name 3 of the 8 Lecture 2 fault categories detected. *(Inconsistent encoding, sentinel values, invalid values).*
4. **[About A]** How many assertions are in `validate()` and what happens on clean vs raw? *(15 assertions; clean passes 15/15, raw fails 10/15).*
5. **[About A]** Why were name typos NOT corrected during cleaning? *(Correcting typos with ground truth is leakage; matching noisy names is the model's job).*
6. **[About C]** How does Screen 5 update metrics when you move the threshold slider? *(Runs live simulation over test predictions using `compute_cost_and_thresholds()`).*
7. **[About C]** What survivorship rules create the Golden Record? *(Sentinel elimination, completeness, and recency from latest registration date).*
8. **[About C]** What is automation bias and how does Screen 6 mitigate it? *(Clerks rubber-stamping AI; mitigated by 2-step confirmation and plain reasons for/against).*
9. **[About C]** What is the difference between Reject and Defer? *(Reject declares records permanently distinct; Defer pauses review for more evidence).*
10. **[About C]** What does `scripts/clean_machine_test.py` test? *(Installs in a clean temp sandbox, runs 167 tests, and verifies all 7 pages).*

### Questions for Member C About Member A & B's Work
1. **[About A]** How many total records and true entities are in the raw register? *(54,373 records across 46,952 true entities).*
2. **[About A]** What is in `docs/leakage_statement.md`? *(Information horizon excluding `true_entity_id`, generator flags, cluster sizes, and clerk actions).*
3. **[About A]** What are the `{col}_prov` columns? *(Provenance columns tracking the exact cleaning repair applied to each field).*
4. **[About A]** Why is missingness in CNIC not random? *(Chi-square test proved Legacy hospital system structurally never recorded CNICs).*
5. **[About A]** What hard negative patient pairs exist in the data? *(Siblings, twins, father-son same-name pairs, shared households).*
6. **[About B]** Name your 5 blocking passes. *(B1 Soundex+Year, B2 Fn3+YearMo, B3 Phone7, B4 CNIC exact, B5 Sorted-Neighbourhood).*
7. **[About B]** What was the blocking completeness? *(94.48%; 538 true duplicate pairs were missed).*
8. **[About B]** Why is accuracy with an adjacent baseline mandatory? *(In a 99.46% negative population, 99.46% accuracy is achieved by guessing non-match).*
9. **[About B]** What does Laplace smoothing do for Naive Bayes? *(Prevents zero conditional probabilities from zeroing out the entire posterior).*
10. **[About B]** What is end-to-end recall? *(Classifier recall multiplied by blocking completeness: $1.00 \times 0.9448 = 94.48\%$).*

---

## 5. Rehearsed 12-Minute Viva Demonstration Script

### Pre-Flight Checklist (Run 10 Minutes Before Entering Viva)
- [ ] Laptop connected to power; screen resolution set to 1080p; browser zoom at 100%.
- [ ] Run pre-check: `python scripts/check_imports.py` $\implies$ `[PASS]`.
- [ ] Run cross-check: `python scripts/cross_check_metrics.py` $\implies$ `[PASS]`.
- [ ] Run test suite: `pytest` $\implies$ `167 passed`.
- [ ] Launch application: `run.bat` (or `streamlit run app.py`) $\implies$ Opens `http://localhost:8501`.
- [ ] Open second browser tab with [`README.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/README.md) and [`docs/traceability_audit.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/traceability_audit.md).

---

### Minute-by-Minute Rehearsal Script

```
00:00 - 01:30 | 1. Introduction, Problem & Governance Banner (Speaker: Member C)
01:30 - 03:30 | 2. Screen 1: Data Engineering, Quality Audit & Validation (Speaker: Member A)
03:30 - 05:30 | 3. Screen 2: Candidate Blocking & Prospective Model Benchmark (Speaker: Member B)
05:30 - 07:00 | 4. Screen 3: Single Pair Investigation & Zero-Jargon Explanations (Speaker: Member B)
07:00 - 08:30 | 5. Screen 4: High-Throughput Batch Deduplication (Speaker: Member C)
08:30 - 10:00 | 6. Screen 5: Cost Matrix Optimization & Live Consequences (Speaker: Member B)
10:00 - 11:30 | 7. Screen 6: Reversible Golden Record Merge & Byte-for-Byte Undo (Speaker: Member C)
11:30 - 12:00 | 8. Limitations, Ethics & Closing Handover (All Members)
```

#### Minute 00:00 - 01:30: Executive Problem Brief & Governance Banner (Member C)
- **Action:** Open `http://localhost:8501` (Home).
- **Spoken Script:**
  > *"Respected examiners, duplicate patient records are a major clinical safety hazard in hospital networks. When departmental systems merge, typographical errors, uncollected CNICs, and format shifts fragment patient charts. This leads to fatal blood transfusion mismatches, lost allergy records, and millions in redundant laboratory tests.
  > 
  > Our solution is the National Patient Master Index Deduplication Assistant. Notice first our mandatory governance banner at the top of every screen: 'Student project. Not fit for clinical, financial or legal use. All data is synthetic.' Our architecture combines Lecture 2 data quality auditing, Lecture 4 candidate blocking, prospective multi-model evaluation with mandatory baseline pairing, zero-jargon explainability, and a 100% reversible SQLite merge store. I now hand over to Member A to demonstrate our data quality pipeline."*

#### Minute 01:30 - 03:30: Screen 1 — Quality Audit, Repairs & Validation (Member A)
- **Action:** Click **"1 Data and Quality"** $\implies$ Click **"Load demo_register.csv"**.
- **Spoken Script:**
  > *"On Screen 1, we ingest our patient register. Our dataset covers 54,373 records across all four Stevens measurement scales, including Nominal CNICs and Interval dates of birth.
  > 
  > Scrolling to Section 2, our Lecture 2 quality audit detects 8 distinct fault categories across 19,773 faulty records: inconsistent encodings like city aliases (2,324 records), sentinel values like DOB 1900-01-01 (392 records), and collection errors. Furthermore, our Chi-Square test ($\chi^2 > 13,000, p < 10^{-300}$) proves that missing CNICs are concentrated in the Legacy system, proving missingness is MAR, not MCAR.
  > 
  > In Section 3, we execute our 15 automated validation assertions. Notice the contrast: on our cleaned dataset, all 15 assertions pass with zero violations. But on the raw register, 10 assertions fail with thousands of violations. Finally, our cleaning pipeline preserves original values in 13 raw columns and tracks 10 provenance columns, downloadable right here."*

#### Minute 03:30 - 05:30: Screen 2 — Blocking Savings & Evaluation Honesty (Member B)
- **Action:** Click **"2 Model Performance"**.
- **Spoken Script:**
  > *"Moving to Screen 2, exhaustive pairwise comparison over 54,000 records requires 1.48 billion comparisons. To solve this, we implemented five orthogonal blocking passes from scratch using pure Python dictionaries: Soundex, name prefixes, phone subscriber digits, CNIC, and Sorted-Neighbourhood. Our multi-pass union prunes 99.9443% of comparisons down to 823,534 candidate pairs in 1.56 seconds, while retaining 94.48% of all true duplicates.
  > 
  > In Section 2, we evaluate our models prospectively on 106,298 test pairs. Look at our primary benchmark table: every accuracy figure is paired directly with the Majority-Class Baseline (99.46%). Because non-matches represent 99.46% of pairs, guessing 'non-match' achieves 99.46% accuracy with zero clinical utility. Our Pruned Decision Tree achieves 99.98% accuracy (beside 99.46% baseline), 96.45% precision, 100.0% recall, and 0.9794 PR-AUC. Furthermore, all models were trained using 5-fold StratifiedGroupKFold grouped by entity on a temporal split, guaranteeing zero entity leakage."*

#### Minute 05:30 - 07:00: Screen 3 — Single Prediction & Plain Reasons (Member B)
- **Action:** Click **"3 Single Prediction"** $\implies$ Click **"🔴 Hard Negative (Twin Sisters)"** $\implies$ Click **"🚀 Compare & Explain Pair"**.
- **Spoken Script:**
  > *"On Screen 3, a clerk can investigate two suspicious charts. We first load a hard negative: twin sisters sharing surname, father name, address, phone, and birth date. Notice our model correctly classifies them as 'Distinct Patients' (Probability 0.08, Unlikely Duplicate).
  > 
  > Look at the plain-language explanation below: there are zero node numbers, zero Gini values, and zero mathematical inequalities. The system explains in plain English: 'First names differ', 'National IDs differ'.
  > 
  > Now we click '🟢 Real-World Match (Typo & Missing CNIC)' and re-run. Despite an OCR typo in the first name and an uncollected CNIC, the system identifies the duplicate with 99.2% probability, highlighting matching phone subscriber digits, exact DOB, and high demographic agreement."*

#### Minute 07:00 - 08:30: Screen 4 — Batch Ingestion & Ranked Worklist (Member C)
- **Action:** Click **"4 Batch Scoring"** $\implies$ Click **"Load sample_batch.csv"** $\implies$ Click **"⚡ Run Deduplication Pipeline"**.
- **Spoken Script:**
  > *"Screen 4 handles high-throughput departmental register deduplication. We load a new 500-record intake batch. Notice our automated schema guard checks required columns and strips forbidden ground-truth fields if present.
  > 
  > When we click Run Deduplication, the progress bar executes cleaning, 5-pass blocking, 58-feature extraction, and model scoring in real time. Blocking achieves a 99.5% reduction. Below is the ranked worklist: pairs are sorted by duplicate probability, complete with demographic snippets and an explicit primary reason column for every pair. A supervisor can filter by confidence band or export to CSV."*

#### Minute 08:30 - 10:00: Screen 5 — Cost Matrix & Live Simulation (Member B)
- **Action:** Click **"5 Settings and Cost Matrix"**.
- **Spoken Script:**
  > *"On Screen 5, hospital leadership configures clinical misclassification costs. Missing a duplicate (C_FN = Rs. 20.0) risks split medical records, which is 4× worse than clerk review time (C_FP = Rs. 5.0).
  > 
  > Bayesian decision theory derives our optimal threshold: 5 / (5 + 20) = 0.20. Watch our live consequence simulation below as I move the threshold slider from 0.20 to 0.50. The page instantly recalculates over 106,000 test pairs: expected cost jumps from Rs. 0.99 up to Rs. 4.84, duplicate recall drops from 100% to 94.7%, and 30 duplicate patients are left with split charts. We click 'Reset to Cost-Optimal' to restore hospital policy alignment."*

#### Minute 10:00 - 11:30: Screen 6 — Reversible Merge & Byte-for-Byte Undo (Member C)
- **Action:** Click **"6 Merge Review"** $\implies$ Scroll to Side-by-Side comparison $\implies$ Click **"🔀 Merge Records"** $\implies$ Enter clerk note $\implies$ Confirm Step 2 $\implies$ Click **"↩️ Undo Merge"**.
- **Spoken Script:**
  > *"Screen 6 is our domain merge console. Notice the side-by-side comparison with green Agree and red Conflict tags. If records have conflicting blood groups or distinct valid CNICs, unmissable red alerts fire to prevent fatal transfusion errors.
  > 
  > Our golden record engine applies automated survivorship rules: non-sentinel values supersede sentinels, and the latest registration date resolves conflicts. The clerk can override any field via radio buttons.
  > 
  > To prevent automation bias, merging requires a two-step confirmation dialog with a mandatory clerk note. We click Confirm Merge: the golden record is written to SQLite and original JSON snapshots are saved.
  > 
  > Now, observe our core recovery feature: scrolling to Section 5, we click 'Undo Merge'. The merge status immediately updates to 'Undone', and both original records are restored byte-for-byte into the active register with zero data corruption."*

#### Minute 11:30 - 12:00: Limitations, Ethics & Closing Handover (All Members)
- **Spoken Script:**
  > *"To conclude honestly: our system has known limitations. Multi-attribute corrupted duplicates (5.5% of true pairs) escape blocking, and emergency trauma admissions without identification cannot be linked on day zero.
  > 
  > However, by enforcing strict human-in-the-loop review, plain-English explanations, automated blood group collision alerts, universal academic banners, and byte-for-byte exact reversible merges, our tool provides an accountable, clinically safe decision-support portal.
  > 
  > All 167 unit tests pass, and all code and models ship self-contained in this repository. We are ready for your individual questions."*

---

### Fallback Procedures If Something Goes Wrong During Viva
1. **If Streamlit freezes or browser disconnects:**
   - Press `Ctrl + C` in terminal.
   - Run: `streamlit run app.py` (takes 2 seconds to reload).
   - If port 8501 is busy: `streamlit run app.py --server.port 8502`.
2. **If `data/merges.db` is locked by another process:**
   - In terminal: `python -c "from src.merge_store import MergeStore; MergeStore().reset_db()"`
3. **If examiner asks to verify test numbers offline:**
   - Run: `python scripts/cross_check_metrics.py` (proves 100% alignment in 1.2 seconds).
4. **If examiner challenges code independence:**
   - Show [`src/blocking.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/blocking.py) and [`src/features.py`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/src/features.py) to prove zero third-party record linkage black boxes were used.
