# DSC306 Project 11: Patient Record Deduplication Tool: Complete Roadmap

Built from the semester brief and Lectures 2, 3 and 4. Lectures 5-14 were not available, so model and evaluation sections follow the brief's wording and standard practice. Re-check them against your slides when you have them.

---

## 0. How to use this document

1. Read sections 1-3 together as a team and settle the decisions in section 3. Record every decision in `DECISIONS.md`.
2. Work through the phases in section 6 in order. Each phase has an acceptance check; do not move on until it passes.
3. Use section 8 (traceability matrix) as the final audit. Every line of the brief maps to a file or screen.
4. Use `ANTIGRAVITY_PROMPTS.md` to drive the build.

---

## 1. The project at a glance

| Item | Detail |
|---|---|
| Product | Patient Record Deduplication Tool |
| User | Records clerk at a hospital that merged several departmental systems |
| Problem | The same patient exists several times with different spellings, formats and dates. Clinicians see partial histories and the hospital double-counts patients. |
| Product behaviour | Clerk runs the tool over the register, gets a worklist of probable duplicate pairs ranked by confidence, sees each pair side by side with agreeing and disagreeing fields highlighted, and can merge in one click, reversibly. |
| Mining task | Similarity-based matching and pairwise classification: is this pair the same person? |
| Assigned challenge (10 marks) | Making comparison tractable. Implement blocking, report comparisons performed versus all-pairs, and measure the true duplicates blocking misses. |
| Domain (sixth) screen | Side-by-side merge screen: fields marked agree/conflict, confidence score, Merge / Reject / Defer. |
| Deliverable | A working web app (Streamlit recommended) that a stranger can use unaided, started with one documented command. |
| Group | 3 students. Individual viva. |

**Marks (100):** Data quality 15, Pre-processing 10, Models 15, **Evaluation 20**, Explanation 10, **Application 20**, Challenge (blocking) 10, Demo and viva 10.

---

## 2. Non-negotiables

These carry marks or are automatic failures in the demo.

1. **Baseline beside every accuracy.** The majority-class baseline sits next to every accuracy figure, everywhere (report, screen 2, README). An accuracy alone earns nothing.
2. **Cost matrix drives the threshold.** The operating threshold is derived from an editable cost matrix, never left at 0.5. Show what accuracy you gave up to get it.
3. **Every prediction carries its reason**, written in the clerk's language, not the model's. ("Same date of birth and phone number; surnames sound alike", never "node 14, gini 0.31".)
4. **Honest split.** The data has dates, so the split is by time, and you say so. No entity may appear on both sides of the split.
5. **Leakage is found and excluded.** Document which columns would not be known when the clerk asks the question.
6. **One-command start on a clean machine.** No path edits, nothing outside `requirements.txt`, and the dataset ships with the project.
7. **On-screen disclaimer:** "Student project, not fit for clinical, financial or legal use." Show it on every page.
8. **Anonymised or synthetic data only.** Never a real person's record.
9. **Every member can defend their part at the viva.** Marked individually.

---

## 3. Key design decisions (recommendations)

Record your final choice for each in `DECISIONS.md`.

### D1. Data source: build your own synthetic register (recommended)
- The brief explicitly allows "a deliberately duplicated version of a register you build yourselves".
- Blocking needs raw names, dates, phones and addresses. As far as I recall, the UCI Record Linkage Comparison Patterns file contains only pre-computed comparison scores and a match label, with no raw fields. Check the file before relying on it. If so, it cannot demonstrate blocking and may fail the "3 of 4 measurement types" rule.
- You control the ground truth (a hidden entity ID), which makes labelled pairs and honest evaluation possible.
- Size: about 50,000 records, so the all-pairs figure is 50,000 x 49,999 / 2 = **1,249,975,000**. This reproduces the brief's "over a billion comparisons" example.
- Optional extra: use the UCI file only as an external sanity check of the pair classifier, if it is available and you have time.
- **Be honest in the report:** synthetic data is easier and cleaner than real hospital data. State this as a limitation.

### D2. Make the data hard enough to be meaningful
Include **hard negatives** (different people who look alike) or the classifier will be trivially accurate:
- Siblings and twins (same surname, address, phone, near dates of birth)
- Father and son with the same full name
- Same household sharing phone and address
- Very common names (for example Muhammad Ahmed) with different dates of birth

Include realistic Pakistani name variation: Muhammad / Mohammad / Mohd / M., Ahmed / Ahmad, Khan / Khaan, initials, swapped first and family name, honorifics, name order.

### D3. Keep ground truth out of the main file
- `register_raw.csv`: what the clerk's system holds, with shuffled `record_id`.
- `ground_truth.csv`: `record_id -> true_entity_id`. Used only to label training pairs and to evaluate. **Never a feature.**
- `fault_log.csv`: what the generator injected, used to check that the audit finds what is really there.
- Shuffle `record_id` so that duplicates are not adjacent (otherwise ID proximity leaks the label).

### D4. Split: entity-grouped, by time
- Assign each true entity to train or test by its **earliest registration date** (for example earlier 70% of the timeline to train, later 30% to test). All records of an entity go to the same side.
- Pairs are built only within a side, so no entity leaks.
- Cross-validation on the training portion uses **StratifiedGroupKFold** (grouped by entity).
- Run a deliberate experiment showing what a **random pair-level split** would falsely report. This is strong evidence of evaluation honesty.

### D5. What the test population is
The product scores pairs that survive blocking, so evaluate on **blocked candidate pairs with their natural class proportions**, not on a hand-balanced set. Report the true-match prevalence and the majority baseline for that population. You may train on a re-sampled set, but evaluation must reflect real use (re-weight if you stratify; Lecture 3).

### D6. Models: three families plus an optional fourth
| Family | Model | Why |
|---|---|---|
| Decision tree | Pruned tree | Required explainer. Decision path becomes plain sentences. |
| Naive Bayes | Categorical NB on binned similarities, Laplace smoothing | Posterior contributions per feature give a natural explanation. |
| Nearest neighbour | k-NN on scaled pair features | Explanation by showing the most similar labelled training pairs. |
| Optional | Small neural network (MLP) | Only if time allows. More to defend at the viva. |

### D7. Two operating bands from cost matrices
- Worklist threshold (is this pair worth a clerk's time?): FN costs more than FP, so a low threshold.
- Stricter "very likely duplicate" band: a false merge corrupts a medical record, so FP costs far more, giving a high threshold.
- Both bands are derived from editable cost matrices. For calibrated probabilities, flag when `p >= C_FP / (C_FP + C_FN)`. Verify calibration and fix it on training out-of-fold predictions, never on the test set.
- **You must write the justification for the cost values yourselves** (clerk review time versus the risk of a split medical history versus the harm of a wrong merge). Do not copy defaults from anyone.
- The product **never merges automatically**. A human always confirms.

### D8. Merge must be reversible
Never delete. Store a merge log (SQLite or JSON) with before-snapshots, survivorship choices and timestamps. Undo restores exactly.

### D9. Stack
Python 3.11+, pandas, numpy, scikit-learn, rapidfuzz (string similarity), Streamlit, plotly or matplotlib, joblib, pytest. Pin versions. Implement Jaccard and cosine yourselves so you can explain them (Lecture 4). Using a library for Jaro-Winkler and Levenshtein is fine, but you must be able to describe how they work.

### D10. Confirm with the instructor
- Synthetic register is acceptable (the brief says so for Project 11).
- Entity-grouped temporal split as the interpretation of "split by time".
- That the blocking report and pipeline evidence may live on screens 2 and 4.

---

## 4. Architecture and repository layout

```
dedup-tool/
  README.md                 one-command start, data provenance, results summary, limitations
  PROJECT_CONTEXT.md        shared brief for the AI agent
  DECISIONS.md              dated log of every decision and why
  requirements.txt          pinned
  run.sh / run.bat          one command: create venv, install, launch
  app.py                    Streamlit entry (Home page + disclaimer)
  pages/
    1_Data_and_Quality.py
    2_Model_Performance.py
    3_Single_Prediction.py
    4_Batch_Scoring.py
    5_Settings.py
    6_Merge_Review.py
  src/
    config.py  generate_data.py  profile.py  audit.py  clean.py  validate.py
    blocking.py  pairs.py  features.py  split.py  preprocess.py
    models.py  evaluate.py  explain.py  merge_store.py  utils.py
  data/
    raw/        register_raw.csv  demo_register.csv  sample_batch.csv
    truth/      ground_truth.csv  fault_log.csv        (not loaded by the app UI)
    clean/      register_clean.csv
    pairs/      labelled pairs (train / test)
  models/       trained models and metadata (.joblib)
  reports/      metrics.json  figures/  (every number in docs comes from here)
  docs/         data_dictionary.md  leakage_statement.md  quality_audit.md
                preprocessing_decisions.md  model_selection.md  evaluation_report.md
                blocking_report.md  explanation_guide.md  ethics_statement.md
                generation_mechanism.md  team_contributions.md
  tests/        test_validate.py  test_blocking.py  test_features.py  test_explain.py ...
```

Pipeline: generate register -> profile and audit -> clean (with provenance) -> validate -> block -> build candidate pairs -> compare fields into features -> split -> preprocess -> train 3 models -> evaluate -> explain -> serve in the app.

---

## 5. Team roles

| Member | Owns | Must defend at viva |
|---|---|---|
| A: Data | Generator, profiling, dictionary, audit, cleaning, validate(), leakage statement | Stages A and B, fault categories, provenance, why each repair |
| B: Models | Blocking, pairs, features, split, pre-processing, models, evaluation, explanation logic | Stages C, D, E, F, the blocking challenge |
| C: App | All six screens, merge store, README, run script, clean-machine test, usability test | Stage G, screen behaviour, reversible merge |

Rules: everyone reviews the others' work weekly; hold a **cross-teaching session** before the viva where each person explains another person's part. Each person should also be able to give a one-minute overview of the whole pipeline.

---

## 6. Phases

### Phase 0: Setup and decisions
- Repo, virtual environment, pinned requirements, folder layout, `PROJECT_CONTEXT.md`, `DECISIONS.md`.
- **Check:** `pytest` runs (even with one dummy test); fresh clone installs from `requirements.txt`.

### Phase 1: Build the register (Member A)
- Generate about 50,000 raw records plus a 5,000-record demo register and a 2,000-row sample batch file.
- About 8-15% of entities have 2-4 records (set and document your own rate). Inject corruptions per section 7.1. Include the hard negatives in D2.
- Document the mechanism in `docs/generation_mechanism.md`: rates, operators, seed.
- **Check:** at least 2,000 records, at least 10 attributes, all four measurement types present; a trivial rule such as "same DOB and same phone" misses a meaningful share of true pairs (otherwise your data is too easy).

### Phase 2: Profile and document (Member A)
- Shape, dtypes, per-column missing counts, min/max/unique (Lecture 2 lab).
- Classify every column N / O / I / R and write down the reasoning.
- Write the data dictionary and the leakage statement (7.2).
- **Check:** every column has type, range and role; at least one column is listed as "not known at query time".

### Phase 3: Quality audit (Member A)
- At least **six** distinct fault categories, using the Lecture 2 vocabulary, each with the detecting expression and the count of affected records (7.3).
- Compare found counts to `fault_log.csv` and report detector recall honestly.
- **Check:** audit report generated by a script, with counts that match the data.

### Phase 4: Clean and validate (Member A)
- Repair with a provenance column per repaired field. Keep `_raw` columns. Do not use ground truth in cleaning.
- Before/after table with a number for each repair.
- `validate()` with at least 10 assertions that **passes on clean and fails on raw** (7.4).
- **Check:** `pytest` proves both directions.

### Phase 5: Blocking (Member B). The assigned challenge
- Implement several blocking passes and their union (7.5). Cap oversized blocks.
- Report: all-pairs count, comparisons actually performed, reduction ratio, pairs completeness (true duplicates retained), pairs quality, block-size distribution, runtime, and a scaling table over 5k / 10k / 25k / 50k records.
- Analyse the duplicates blocking misses and say why.
- **Check:** `blocking_report.md` has all numbers from a script; the missed-duplicate analysis names concrete causes.

### Phase 6: Pairs, features and split (Member B)
- Candidate pairs from blocking, labelled from ground truth (evaluation and training only).
- Comparison features per pair (7.6). Handle missing values explicitly.
- Entity-grouped temporal split (D4).
- **Check:** assertion that no entity appears on both sides; feature list contains no forbidden column.

### Phase 7: Pre-processing decisions (Member B)
- Rescaling: decide and justify per model family (7.7).
- Compare **PCA versus feature subset selection** on the same model with identical folds.
- Discretization for naive Bayes and rules: compare equal-width, equal-frequency and supervised (entropy) binning; justify the one you ship.
- Stratified sampling and a progressive-sampling learning curve for training size.
- **Check:** `preprocessing_decisions.md` with measured comparisons, not just "we ran both".

### Phase 8: Models (Member B)
- Train the three families on identical prepared data and compare on the same test set.
- Hyper-parameters chosen by a stated, pre-registered procedure (7.8).
- **Check:** `model_selection.md` states the grid, objective, folds, seed and the chosen values.

### Phase 9: Evaluation (Member B). 20 marks, the biggest block
- Everything in 7.9. Touch the test set once.
- **Check:** `evaluation_report.md` has a baseline beside every accuracy, two extra metrics with justification, ROC with AUC, cost matrix and threshold, CV mean and spread, leakage proof, random-versus-temporal comparison, end-to-end recall including blocking.

### Phase 10: Explanation (Member B with C)
- Explanations for all three models, in plain language (7.10).
- **Check:** every prediction surfaced in the app has a reason; unit tests confirm no raw model jargon appears.

### Phase 11: Application (Member C, starting early with stubs)
- Foundation, then screens 1-5, then screen 6 (7.11). Wire to the first model early and swap later.
- **Check:** all six screens work end to end on the demo register.

### Phase 12: Documentation, packaging, hardening
- README, run script, clean-machine test, usability test with two people outside the class, ethics statement, team contributions.
- **Check:** a fresh machine or fresh virtual environment starts the app with one command and screens 1-6 work without editing anything.

### Phase 13: Final audit and viva rehearsal
- Walk the traceability matrix (section 8) line by line. Fix gaps. Mock viva (section 10).
- Freeze features. Only fix bugs.

---

## 7. Detailed specifications

### 7.1 Register columns and corruption operators

| Column | Meaning | Type (Lecture 2) | Plausible range | Role |
|---|---|---|---|---|
| record_id | Row ID (shuffled) | Nominal | unique | Identifier (never a feature) |
| source_system | Origin system (OPD, Lab, Radiology, Pharmacy, Emergency, Legacy) | Nominal | 6 values | Explanatory (use carefully, see leakage) |
| registration_date | When the record was created | Interval | within the register span | Used for splitting; excluded as a feature unless justified |
| first_name | Given name | Nominal | text | Explanatory |
| family_name | Surname | Nominal | text | Explanatory |
| father_name | Father or guardian name | Nominal | text, often missing | Explanatory |
| gender | M / F / U after cleaning | Nominal | dirty codes in raw | Explanatory |
| date_of_birth | DOB | Interval | 1925 to registration date | Explanatory |
| age_years | Age at registration | Ratio | 0-110 | Explanatory (consistency check against DOB) |
| national_id | Synthetic 13-digit ID | Nominal | 13 digits | Explanatory (synthetic, flagged as such) |
| phone | Mobile number | Nominal | 11 digits normalised | Explanatory |
| city | City | Nominal | canonical list | Explanatory |
| address | Free-text address | Nominal | text | Explanatory |
| blood_group | Blood group | Nominal | 8 values | Explanatory |
| insurance_tier | None < Basic < Standard < Premium | Ordinal | 4 levels | Explanatory |
| visit_count | Visits recorded | Ratio | 0-500 | Explanatory |
| true_entity_id | Ground truth (separate file only) | Nominal | n/a | **Forbidden** (evaluation only) |

**Corruption operators for duplicates (document rates):** name typo (insert/delete/substitute/transpose), spelling variant (Muhammad/Mohammad/Mohd), initial only, honorific added, first/family swap, case and whitespace noise, DOB day-month swap, DOB digit transposition, DOB missing or sentinel, phone format variants and digit error, national ID missing or one digit off, city alias (LHR, lahore), gender code variants, address abbreviations, missing father name, sentinel values (-999, 0, 1900-01-01, 0000000000), and unit or magnitude errors (age 199, visit_count 9999).

### 7.2 Leakage statement: columns not known when the clerk asks
- `true_entity_id` (ground truth)
- Any generator log or corruption flag
- Any "merged_into" or "canonical_id" produced after a merge
- Any clerk decision (merge, reject, defer)
- Cluster size or "number of duplicates of this record"
- `record_id` ordering (shuffled; assert no correlation with the label)
- Corruption artefacts that only duplicates carry (for example a trailing space only on corrupted copies). Test for this and remove it in cleaning.

Also describe: the `source_system` pair pattern (duplicates often cross systems). Real hospitals have this too, so it may be a legitimate feature, but only if negatives are sampled with the same cross-system rate as in real use. Decide, justify and show the effect with and without it.

### 7.3 Fault catalogue (use at least six; Lecture 2 names)

| # | Category | Example in this register | Detector (write as an expression) |
|---|---|---|---|
| F1 | Duplicate | Exact repeated rows (ignoring ID); near duplicates are the product's target and are reported by blocking, not "repaired" | `df.duplicated(subset=non_id_cols)` |
| F2 | Inconsistent encoding | city Lahore / LHR / lahore; gender M / Male / 1; three date formats; phone formats; name case and spacing | distinct-value profile; regex per format |
| F3 | Sentinel value | DOB 1900-01-01, age 0 or 999, phone 0000000000, visit_count -1 | value in a sentinel set, per column |
| F4 | Invalid value | month 13, DOB in the future, DOB after registration, ID not 13 digits, age inconsistent with DOB | date parse failure; range and cross-field checks |
| F5 | Missing value | blank phone, national_id, father_name; check whether missingness depends on `source_system` (not random) | `isna()` and a per-source cross-tab |
| F6 | Measurement (entry) error | name typos, transposed digits, dropped digit | name-lexicon lookup, rare-variant frequency, digit-count checks |
| F7 | Outlier | visit_count far above the 99.9th percentile (investigate, do not blindly delete; Lecture 2) | percentile or robust z with MAD |
| F8 | Collection error | a source system that never records a field | per-source null rate of 100% for a column |

### 7.4 Cleaning rules and `validate()`
- Normalise case and whitespace, standardise date formats to ISO, canonicalise city and gender, normalise phone to 11 digits, set sentinels and invalid values to missing and flag them.
- Do **not** "correct" name typos using ground truth. Matching them is the model's job.
- Provenance column for every repaired field (for example `city_prov` in {original, case_fixed, alias_mapped, set_missing}).
- Before/after table: number of distinct city strings, count of invalid DOBs, count of sentinels, count of malformed phones, and so on.

`validate(df)` assertions (at least 10; each returns a count of violations):
1. `record_id` unique and non-null
2. `gender` in {M, F, U}
3. `city` in the canonical list or null
4. DOB parses, is >= 1900-01-02 and <= registration_date
5. No DOB sentinel values
6. `age_years` in [0, 110] or null, and within 1 year of the age computed from DOB
7. `phone` matches `^03\d{9}$` or is null
8. `national_id` matches `^\d{13}$` or is null
9. Names are trimmed, single-spaced, no digits
10. `visit_count` is an integer >= 0 and <= the documented cap
11. `insurance_tier` in the ordered set
12. Provenance columns exist for every repaired field with allowed values
13. No exact duplicate rows (ignoring ID)
14. No forbidden column (ground truth, logs) in the modelling frame
15. `registration_date` within the register span

Show on screen 1 that it **passes on clean and fails on raw**, with violation counts.

### 7.5 Blocking (the assigned challenge)
Passes (union, then de-duplicate pairs):
- B1: Soundex(family_name) + birth year
- B2: first 3 letters of first_name + birth year-month
- B3: last 7 digits of phone
- B4: national_id exact
- B5: sorted-neighbourhood window on a normalised name key
- Optional stretch: q-gram or MinHash blocking

Controls: maximum block size (skip or sub-block giant blocks such as very common names), and missing keys simply do not generate a block.

Report for every scheme and for the union:
- All-pairs n(n-1)/2, comparisons performed, **reduction ratio** = 1 - comparisons / all-pairs
- **Pairs completeness** = true duplicate pairs retained / all true pairs (the cost of blocking in missed duplicates)
- **Pairs quality** = true pairs / candidate pairs
- Block-size histogram and largest block, runtime, scaling table (5k to 50k)
- Missed-duplicate analysis: for each missed true pair, which corruptions defeated every key. Present the chosen scheme and the target completeness you fixed **before** looking at the numbers.

### 7.6 Pair features (comparison vector)
Each feature is a per-field similarity (the Lecture 4 idea of choosing a measure by attribute type).
- **Names:** Jaro-Winkler, Levenshtein ratio, exact match, Soundex equality, initial equality, character 2-gram Jaccard, character 3-gram TF-IDF cosine on the full name, first/family swap flag
- **Date of birth:** exact, day equal, month equal, year equal, day-month swapped flag, one-digit edit flag, absolute difference in days (log-transformed)
- **Identifiers:** national ID exact, Hamming distance <= 1, phone exact, last-7 equal, phone edit distance
- **Other:** gender equal, city equal, address token Jaccard and cosine, blood group equal, insurance tier difference (ordinal, Lecture 4 formula), visit_count absolute difference (log)
- **Missingness:** per-field "both present" flags, plus an overall **Gower similarity with the delta rule** (Lecture 4) that skips missing fields
- Handling missing per model family: tree (sentinel value plus flag), naive Bayes (explicit "missing" bin), k-NN (Gower-style distance or imputation plus flag). State which and why.

### 7.7 Pre-processing decisions (justify per family)
| Family | Rescale? | Reason (Lecture 4: units and ranges decide who is "nearest") |
|---|---|---|
| Decision tree / rules | No | Splits are order-based and scale-invariant |
| k-NN | Yes (standardise or min-max after log of heavy-tailed features) | Distance-based; one feature must not dominate |
| Naive Bayes (categorical) | No scaling; **discretize** | Needs bins; choose binning by measured effect |
| Neural network (optional) | Yes | Gradient-based training |

Also:
- **PCA versus feature selection:** same model, same folds, same metric. Standardise before PCA (Lecture 3). Note PCA components are hard to explain, which matters for Stage F.
- **Selection:** filter (correlation with target; drop one of each pair correlated above about 0.95) and forward selection (Lecture 3's d(d+1)/2 cost).
- **Binarization:** one-hot for nominal fields such as source-system pair.
- **Sampling:** stratify to keep enough positives; re-weight metrics computed on any re-sampled set; plot a progressive-sampling learning curve.

### 7.8 Hyper-parameter procedure (state it before running)
- Training portion only, 5-fold StratifiedGroupKFold (grouped by entity), fixed seed.
- Objective: expected cost at the cost-optimal threshold (or PR-AUC), chosen in advance.
- Grids: tree (max_depth, min_samples_leaf, criterion, cost-complexity alpha), k-NN (k, L1 versus L2, weighting), naive Bayes (smoothing alpha, number and type of bins).
- Tie-break by the simplest model within one standard error of the best.
- The test set is used once, at the end.

### 7.9 Evaluation checklist (20 marks)
- Split as in D4, and say so in the report.
- Majority-class baseline **and** a simple domain rule baseline (for example "same DOB and same phone"), beside every accuracy.
- At least two metrics beyond accuracy with a written reason: **precision and recall** (plus F1 and precision-recall AUC). Reason: the clerk's time is wasted by false alarms, while missed duplicates leave patients with split histories; accuracy is dominated by non-matches.
- **ROC curve with AUC** for all three models, plus a precision-recall curve.
- Calibration check and fix (on training out-of-fold predictions).
- **Cost matrix** (editable), the derived threshold, and expected cost per 1,000 candidate pairs versus the 0.5 default. Show the accuracy you gave up.
- 5-fold (or 10-fold) grouped CV on the training portion, reporting **mean and standard deviation** for each metric.
- Confusion matrix at the operating threshold.
- **End-to-end recall** = blocking completeness x classifier recall, so blocking misses count as misses.
- **Leakage proof:** show performance with and without a leaky feature (for example `record_id` difference when IDs are not shuffled), then show the honest model.
- **Random versus temporal split:** show what the random split would have falsely reported.
- Limitations, including that the data is synthetic.
- Clerk workload view: worklist size and estimated review hours at the chosen threshold.

### 7.10 Explanation (user's language)
- **Tree:** convert each node test into a sentence using a feature-to-phrase dictionary, for example `dob_exact = 1` becomes "Same date of birth"; `fname_jw >= 0.9` becomes "First names are nearly identical (for example Muhammad / Mohammad)"; `nid_both_present = 0` becomes "National ID is missing on one record, so it could not be compared".
- **Naive Bayes:** rank features by log-likelihood-ratio contribution and describe the top ones, for and against.
- **k-NN:** show the nearest labelled training pairs and their outcomes in words.
- Include reasons **against** a match too ("Different blood groups", "National IDs differ").
- Highlight agreeing and disagreeing fields on the merge screen.
- Test that explanations never contain feature IDs, node numbers or Gini values.

### 7.11 Application: six screens
Global: disclaimer banner on every page, sample-data buttons, clear empty states and error messages for bad CSVs, help text, session state shared across pages, caching for speed, progress bars for long jobs. Cost matrix, threshold and model choice from screen 5 apply everywhere.

| # | Screen | Must do |
|---|---|---|
| 1 | Data and quality | Load raw data, show the data dictionary and audit report (fault, expression, count), before/after table, validate() results on raw versus clean, download the cleaned file |
| 2 | Model performance | Comparison table of the 3 models (accuracy **with baseline beside it**, precision, recall, F1, PR-AUC, expected cost, CV mean +/- sd), ROC curve with AUC, confusion matrix, blocking summary, threshold and cost view |
| 3 | Single prediction | Form for **two records** (A and B, with example pairs). Output: duplicate probability, confidence, verdict at the current threshold, explanation in plain language |
| 4 | Batch scoring | Upload a register CSV: validate columns, run cleaning, blocking and scoring, show blocking stats for this file, a ranked and filterable worklist (confidence, source, city, reason), download results |
| 5 | Settings | Choose model, move threshold, edit the cost matrix, and see expected cost, precision, recall, worklist size and confusion matrix update **live**; show the cost-optimal threshold |
| 6 | Merge review (domain) | Two candidate records side by side, field-by-field agree/conflict marking, confidence and reasons, survivorship choice per field, **Merge / Reject / Defer**, merge log, **Undo** |

Merge store: SQLite or JSON containing merge ID, both original snapshots, chosen golden record, field provenance, timestamp, status. Undo restores exactly. Reject and Defer decisions are recorded (optionally exported as new labelled pairs).

### 7.12 Ethics statement (include even though not mandatory for Project 11)
Who could be harmed: patients whose records are wrongly merged (clinical errors) or left split (missed history). What the interface does: no auto-merge, side-by-side review, confidence and reasons shown, reversible merges, audit log, on-screen disclaimer. Data is synthetic and anonymised.

---

## 8. Requirement traceability matrix

| Brief requirement | Satisfied by | Evidence |
|---|---|---|
| A1: at least 2,000 records, 10 attributes, 3 of 4 types | Generator, data dictionary | `docs/data_dictionary.md` |
| A2: data dictionary (type, range, role) | Phase 2 | `docs/data_dictionary.md` |
| A3: columns not known at query time | Leakage statement | `docs/leakage_statement.md` |
| B1: 6+ fault categories, expression, count | Audit | `docs/quality_audit.md`, screen 1 |
| B2: repairs with provenance column, before/after numbers | Clean | `register_clean.csv`, screen 1 |
| B3: validate() 10+ assertions, pass on clean, fail on raw | `validate.py` | tests, screen 1 |
| C1: rescaling decision per family | `preprocessing_decisions.md` | measured |
| C2: reduction or selection, compared on same model | Phase 7 | table in doc |
| C3: discretization or binarization justified | Phase 7 | doc |
| D1: 3 classifiers, 3 families, same data and test set | Phase 8 | `model_selection.md`, screen 2 |
| D2: one tree or rule set | Decision tree | same |
| D3: hyper-parameters by stated procedure | 7.8 | same |
| E1: split reflects use (by time) | D4 | `evaluation_report.md` |
| E2: baseline beside every accuracy | Everywhere | report, screens 2 and 5, README |
| E3: two extra metrics with reason | 7.9 | report |
| E4: ROC and AUC | Phase 9 | screen 2 |
| E5: cost matrix chooses threshold | Phase 9, screen 5 | screens 2 and 5 |
| E6: k-fold CV with spread | Phase 9 | report, screen 2 |
| F1: every prediction has its reason | `explain.py` | screens 3, 4, 6 |
| F2: reason in the user's language | tests | `explanation_guide.md` |
| G: six screens work | Phase 11 | demo |
| G: one-command clean-machine start, nothing missing | Phase 12 | README, tested run |
| Challenge: blocking, comparisons, missed duplicates | Phase 5 | `blocking_report.md`, screens 2 and 4 |
| Ethics: on-screen disclaimer, anonymised data | Every page | banner |
| Viva: each member answers on their part | Section 10 | rehearsal |

---

## 9. Pitfalls and honesty rules

- **No invented numbers.** Every figure in any document comes from a script and is stored in `reports/metrics.json`. Never type results by hand.
- Never tune on the test set. Pick thresholds on training out-of-fold predictions only.
- Never use `true_entity_id` or anything derived from it as a feature.
- Do not report accuracy alone, ever.
- Do not call it "99% accurate" if it is 99% accurate only because most pairs are non-matches. Show the baseline.
- Do not let duplicates in training and test share an entity.
- Report weaknesses (what the model cannot do) frankly; the brief rewards this.
- Do not over-fit the synthetic generator. Test on a freshly generated register with a different seed as a robustness check.
- Do not hide behind AI-written code. Everyone must be able to explain every module.

---

## 10. Viva preparation

Twelve-minute demo, eight minutes of questions. Practise answering these:

**Data (Member A)**
1. Which attribute types are in your data and why? Why is a national ID nominal and DOB interval?
2. Name your six faults. How did you detect each and what did you do?
3. Why did you not correct typos in names? Why keep `_raw` columns?
4. Which column would leak and how did you prove or exclude it?
5. Why is missingness in some fields not random?

**Models and evaluation (Member B)**
6. What does blocking do, and what does it cost you? Give your numbers.
7. Why is accuracy a poor measure here? What is the baseline?
8. How was the threshold chosen? What happens if the cost of a false merge doubles?
9. Why did you split by entity and time? What would a random split have shown?
10. Why did you choose this model family set? What does naive Bayes do with a zero probability?
11. Why standardise for k-NN but not for the tree?
12. PCA versus feature selection: which won and why?
13. What does the confidence number mean? Is it calibrated?
14. What does the system get wrong, and which cases will it never catch?

**Application (Member C)**
15. Walk through a merge and undo it. Where is the log?
16. What stops the clerk from merging the wrong pair by accident?
17. How does the one-command start work on a clean machine?
18. Who could be harmed by this tool, and what does the interface do about it?
19. How would a stranger know what to do on screen 3?

Run a **mock viva** where each person is questioned on someone else's part.

---

## 11. Final freeze checklist

- [ ] Fresh virtual environment: install from `requirements.txt` and run the one command; screens 1-6 work
- [ ] No hard-coded absolute paths; data and models ship with the repo
- [ ] All documents regenerated from the latest `reports/metrics.json`
- [ ] Baseline beside every accuracy figure in docs, app and README
- [ ] Disclaimer on every page; data is synthetic and says so
- [ ] Every prediction (screens 3, 4, 6) has a plain-language reason
- [ ] Undo restores a merge exactly
- [ ] Tests pass (`pytest`)
- [ ] `DECISIONS.md` complete; team contributions written
- [ ] Two people outside the class completed screen 3 and screen 6 without help
- [ ] Traceability matrix fully ticked
- [ ] Demo script rehearsed to under twelve minutes

---

## 12. Time note

The brief sets submission and viva for **Tuesday 27 October**. Confirm your real deadline with the instructor. If time ever gets short, the brief's own cut order is: drop the sixth domain screen first, then the third model, then reduce fault categories from six to three you can properly evidence. **Never cut the baseline, the cost matrix or the explanation.**
