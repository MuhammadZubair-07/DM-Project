# Antigravity Prompt Pack: DSC306 Project 11 (Patient Record Deduplication Tool)

## How to use this pack

1. Put `PROJECT_ROADMAP.md` in the repo root, along with the brief PDF and lecture PDFs if you want the agent to read them.
2. Do **Prompt 0A** first. It creates `PROJECT_CONTEXT.md`, which every later prompt tells the agent to read. If Antigravity has a place for standing rules or instructions, paste the same text there too.
3. Run the prompts **in order**, one per agent session or task. Each prompt is self-contained, ends with a stop-and-report step, and has a **Gate** you check yourselves before moving on.
4. After each prompt: run the tests, read the code that was written, commit, and update `DECISIONS.md`.
5. Ask the agent to produce a short plan first for big prompts (6, 9, 10, 12-14), review it, then let it proceed.
6. If any output looks too good, stop. Ask the agent to show the script and the numbers behind it.

Prompts in order: 0A context, 0B scaffold, 1 generator, 2 profile and dictionary, 3 audit, 4 clean and validate, 5 blocking, 6 pairs and features and split, 7 pre-processing, 8 models, 9 evaluation, 10 explanation, 11 app foundation and screens 1-2, 12 screens 3-5, 13 screen 6, 14 docs and packaging, 15 final audit, 16 viva prep.

---

## Prompt 0A: Project context (save as PROJECT_CONTEXT.md)

```text
Create a file PROJECT_CONTEXT.md in the repo root containing exactly the text between the markers below. Then create an empty DECISIONS.md with a heading and a dated-entry format (date, decision, reason, who). Do nothing else.

=== BEGIN PROJECT_CONTEXT ===
PROJECT: DSC306 Data Mining (COMSATS University Islamabad), semester project, Project 11, Patient Record Deduplication Tool. Group of 3 students. Everything must be defensible by each student at an individual viva.

PRODUCT: A web app (Streamlit) for a hospital records clerk. The clerk runs it over a merged patient register and gets a ranked worklist of probable duplicate pairs, each shown side by side with agreeing and disagreeing fields highlighted, with a one-click REVERSIBLE merge. The tool never merges automatically.
MINING TASK: similarity-based matching and pairwise classification (is this pair the same person?).
ASSIGNED CHALLENGE: make comparison tractable. Implement blocking; report comparisons performed vs all-pairs n(n-1)/2; measure true duplicates that blocking misses (pairs completeness), reduction ratio, pairs quality.
SIXTH (DOMAIN) SCREEN: side-by-side merge screen with agree/conflict marking, confidence, Merge / Reject / Defer, merge log, undo.

COMMON SPEC (all must be met):
A. Dataset of at least 2,000 records and 10 attributes with at least 3 of the 4 measurement types (nominal, ordinal, interval, ratio). Data dictionary written by us: every column, its type, plausible range, and role (identifier / explanatory / target / must-not-use). Written statement of columns not known at the moment the user asks the question (excluded).
B. Data quality audit: at least 6 distinct fault categories using Lecture 2 vocabulary (measurement error, collection error, noise, outlier, missing value, duplicate, inconsistent encoding, invalid value, sentinel value), each with the detecting expression and count of affected records. Repairs with a provenance column per repaired field and a before/after table of numbers. validate() with at least 10 assertions that passes on cleaned data and fails on raw.
C. Pre-processing: justified rescaling decision per model family; EITHER dimensionality reduction OR feature subset selection, with the two compared on the same model; discretization/binarization where a model needs it, binning choice justified.
D. At least 3 classifiers from 3 different families (decision tree, rule-based, nearest neighbour, naive Bayes, neural network), same prepared data, same test set. One must be a decision tree or rule set. Hyper-parameters chosen by a stated procedure.
E. Evaluation: split reflects real use (data has dates, so split BY TIME, and say so); majority-class baseline beside EVERY accuracy; at least two metrics beyond accuracy with a written reason; ROC curve with AUC; cost matrix with the operating threshold chosen from it (not 0.5); k-fold cross-validation on the training portion reporting mean AND spread.
F. Every prediction shown comes with its reason, written in the user's (clerk's) language, never model language like node IDs or gini values.
G. Six screens: 1 Data and quality; 2 Model performance (comparison table, ROC, confusion matrix, baseline beside every figure); 3 Single prediction (form -> prediction, confidence, explanation); 4 Batch scoring (upload CSV -> ranked filterable table -> download); 5 Settings (choose model, move threshold, edit cost matrix, live consequences); 6 Domain screen (merge review). Starts from a clean machine with ONE documented command; no path edits; nothing outside requirements.txt; ships its own dataset. On-screen statement on every page that this is a student project not fit for clinical, financial or legal use. Anonymised or synthetic data only.
Marks: data quality 15, pre-processing 10, models 15, evaluation honesty 20, explanation 10, application 20, assigned challenge 10, demo and viva 10.

COURSE VOCABULARY (Lectures 2-4): attribute types nominal/ordinal/interval/ratio and what operations each permits; discrete vs continuous vs binary (asymmetric binary needs Jaccard-style measures); noise vs artifact vs outlier; missing-value strategies (eliminate, impute, ignore, treat as own value) and the importance of WHY it is missing; duplicates and inconsistent values; sentinel values; aggregation; sampling (stratified, progressive); curse of dimensionality; PCA (standardise first; variance is not importance); feature selection (filter, wrapper, embedded; forward selection); feature creation; binarization (one-hot); discretization (equal width, equal frequency, supervised entropy); log transform; min-max, z-score, robust scaling; similarity vs dissimilarity; Minkowski L1/L2/Linf; metric properties; Mahalanobis; SMC vs Jaccard; cosine; correlation; combined mixed-type similarity with delta (Gower).

DECISIONS ALREADY TAKEN (change only by writing a dated entry in DECISIONS.md):
- Data: synthetic register of about 50,000 records we generate ourselves, with injected duplicates and hard negatives (siblings, twins, father/son same name, shared household phone/address, very common names). Pakistani naming variation. Ground truth in a SEPARATE file (data/truth/ground_truth.csv); never in the register, never a feature. record_id is shuffled.
- Split: entity-grouped and temporal: each true entity goes to train or test by its earliest registration date; pairs built only within a side. CV on train uses StratifiedGroupKFold grouped by entity.
- Evaluation population: candidate pairs surviving blocking, with natural class proportions.
- Models: decision tree (explainer), categorical naive Bayes on binned similarities with Laplace smoothing, k-NN. Optional fourth: small MLP.
- Threshold from an editable cost matrix; calibrated probabilities; two bands (worklist, very likely) both cost-derived. Never auto-merge. Merge store is reversible.
- Stack: Python 3.11+, pandas, numpy, scikit-learn, rapidfuzz, Streamlit, plotly/matplotlib, joblib, pytest, pinned in requirements.txt.

WORKING RULES FOR THE AGENT:
1. HONESTY: never invent or hard-code a metric or count. Every number in any document or screen must come from a script and be saved to reports/metrics.json (or a named JSON in reports/). Fixed random seeds. Report weaknesses frankly.
2. NO LEAKAGE: never use true_entity_id or anything derived from it as a feature or in cleaning. Touch the test set only for final evaluation. Choose thresholds and hyper-parameters on training data only.
3. Baseline (majority class) beside every accuracy, everywhere.
4. Readable code: small functions, docstrings, type hints, comments that explain WHY. Three students must be able to explain every module. No clever one-liners that hide logic.
5. Write pytest tests for each module. Run them and report results.
6. Do not silently change scope. If a requirement is ambiguous or a decision is needed, ask a concise question and wait.
7. At the end of every task output: (a) what you built, (b) files created/changed, (c) how to run it, (d) test results, (e) any numbers produced and where they are stored, (f) risks or open questions, (g) a five-line plain-English explanation a student could use at the viva, (h) a DECISIONS.md entry.
8. Keep everything relative-path and cross-platform (Windows, macOS, Linux). No hard-coded absolute paths.
9. Commit-sized steps. Do not refactor unrelated files.
=== END PROJECT_CONTEXT ===
```

**Gate:** `PROJECT_CONTEXT.md` and `DECISIONS.md` exist. Everyone has read the context file.

---

## Prompt 0B: Scaffold and environment

```text
Read PROJECT_CONTEXT.md and PROJECT_ROADMAP.md first.

Task: set up the repository.
1. Create the folder layout from section 4 of PROJECT_ROADMAP.md (src, pages, data/raw, data/truth, data/clean, data/pairs, models, reports/figures, docs, tests).
2. Create requirements.txt with pinned, mutually compatible versions of: pandas, numpy, scikit-learn, rapidfuzz, streamlit, plotly, matplotlib, joblib, pytest. Verify they install together in a fresh virtual environment on Python 3.11+. Do not add anything else without telling me why.
3. src/config.py: central constants (paths built with pathlib relative to the repo root, random seed, canonical city list placeholder, sentinel sets placeholder).
4. src/utils.py: helpers to save and load JSON metrics into reports/metrics.json (merge, never overwrite other keys).
5. A trivial passing test in tests/ proving imports work.
6. .gitignore (venv, caches, large temp files but NOT data/models we must ship).
7. A placeholder README.md stating the project name and the intended single start command.
8. run.sh and run.bat that: create a virtual environment if missing, install requirements, then launch the Streamlit app. Mark them as placeholders until the app exists.

Do not write any modelling code yet. Finish with the standard end-of-task report.
```

**Gate:** fresh clone, `pip install -r requirements.txt`, `pytest` passes.

---

## Prompt 1: Synthetic register generator (Member A)

```text
Read PROJECT_CONTEXT.md, PROJECT_ROADMAP.md (sections D1-D3, 7.1, 7.2) and DECISIONS.md.

Task: write src/generate_data.py that creates a realistic synthetic hospital patient register with injected duplicates, and the supporting files.

Requirements:
1. Entities: generate about 44,000 to 46,000 true people so that total records are about 50,000 after duplicates. About 10-12% of entities have 2-4 records (make the rate a config value and document it). Registration dates span 2015-01-01 to 2025-10-24; each record in an entity gets its own registration date and source_system (OPD, Lab, Radiology, Pharmacy, Emergency, Legacy). Each source system has its own conventions and gaps (for example Legacy uses dd/mm/yyyy and lacks national_id; Emergency often lacks father_name and phone).
2. Columns in register_raw.csv: record_id, source_system, registration_date, first_name, family_name, father_name, gender, date_of_birth, age_years, national_id (synthetic 13-digit, state clearly that it is synthetic), phone, city, address, blood_group, insurance_tier (ordinal None<Basic<Standard<Premium), visit_count. Include all four measurement types. Use curated Pakistani first-name, family-name and city lists written into the code or a data file (do not call external services). Include spelling-variant maps (Muhammad/Mohammad/Mohd/M., Ahmed/Ahmad, Khan/Khaan, etc.) and city aliases (Lahore/LHR/lahore).
3. Corruption operators applied to duplicate copies (and light noise on some singletons), each with a documented probability: name typos (insert, delete, substitute, transpose), spelling variants, initial only, honorific added, first/family swap, case and whitespace noise, DOB day-month swap, DOB digit transposition, DOB missing or sentinel (1900-01-01), phone format variants and a digit error, national_id missing or one digit off, city alias, gender code variants (M, Male, 1, etc.), address abbreviation, missing father_name, sentinel values (-999, 0, 0000000000), impossible values (age 199, visit_count 9999), mixed date formats.
4. Hard negatives: siblings and twins (same family name, address, phone, DOB within days or equal), father and son with identical full name, households sharing phone and address, very common names with different DOBs. Document their counts.
5. record_id must be SHUFFLED so that IDs carry no information about entity or order. Add a test that proves no relationship between record_id order and entity membership.
6. Outputs: data/raw/register_raw.csv (no ground truth), data/truth/ground_truth.csv (record_id, true_entity_id), data/truth/fault_log.csv (record_id, column, fault_type, original_value, injected_value), plus data/raw/demo_register.csv (about 5,000 records, built from a subset of whole entities) and data/raw/sample_batch.csv (about 2,000 records, whole entities) for the app.
7. Reproducible with a seed in config.py. Provide a CLI: python -m src.generate_data --seed 42.
8. Write docs/generation_mechanism.md describing every rate and operator, plainly stating that the data is synthetic and easier than real hospital data.
9. Sanity report (saved to reports/metrics.json under a key generation): record count, entity count, duplicate rate, cluster-size distribution, hard-negative counts, per-column missing rate, and how often a trivial rule "same DOB and same phone" would find true pairs (this must NOT be near-perfect; if it is, make corruption harder).

Do not clean anything yet. End with the standard end-of-task report.
```

**Gate:** at least 2,000 records, 10+ attributes, 4 types, trivial rule is clearly insufficient, ground truth is in a separate file.

---

## Prompt 2: Profile, data dictionary, leakage statement (Member A)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md and roadmap sections 7.1-7.2.

Task: src/profile.py plus two documents.
1. Profile register_raw.csv: shape, dtypes, per-column missing counts, min/max/unique values, top values per categorical column (the Lecture 2 lab). Save the results to reports/metrics.json (key profile) and a readable reports/profile.md.
2. Classify every column as nominal / ordinal / interval / ratio AND discrete / continuous / binary, with the Lecture 2 reasoning (does zero mean none? does doubling mean twice? is only order real? only identity?). Be careful with identifiers and digit-strings (national_id, phone are nominal, not numbers).
3. Write docs/data_dictionary.md: for each column the meaning, measurement type, plausible range, and role (identifier / explanatory / target / must-not-use). Include true_entity_id, fault logs and any derived label as must-not-use.
4. Write docs/leakage_statement.md: every column or fact that would NOT be known when the clerk asks "are these two records the same person?" and is therefore excluded. Include ground truth, fault logs, merge outcomes, clerk decisions, cluster size, record_id order, and any corruption artefact that only duplicates carry. Add a short section on source_system: explain why it might be a legitimate feature, how it could leak through the way pairs are sampled, and describe a test we will run with and without it.
5. Add a test that parses docs/data_dictionary.md or a machine-readable data_dictionary.json and confirms every column in register_raw.csv is covered.

End with the standard end-of-task report.
```

**Gate:** every column documented; leakage statement is specific, not generic.

---

## Prompt 3: Quality audit (Member A)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, docs/data_dictionary.md and roadmap section 7.3.

Task: src/audit.py producing a data quality audit of register_raw.csv.
1. Implement detectors for AT LEAST SIX distinct fault categories, named with Lecture 2 vocabulary: duplicate (exact), inconsistent encoding (city, gender, date format, phone format, name case/whitespace), sentinel value, invalid value (impossible dates, DOB after registration, malformed IDs, age inconsistent with DOB), missing value (and test whether missingness depends on source_system, which would make it not random), measurement/entry error (name typos via lexicon lookup and rare-variant frequency, transposed or dropped digits), outlier (visit_count, handled with robust statistics and flagged for investigation, not deleted), collection error (a source system that never records a field).
2. For each fault: category, affected columns, the exact detecting expression (store the pandas expression or regex as a readable string), and the COUNT of affected records. Produce a table saved as reports/audit.json and reports/audit.csv.
3. Compare detector output with data/truth/fault_log.csv and report detector recall and precision per fault type. This comparison is for our own honesty checks only; the detectors themselves must not read the fault log.
4. Near-duplicates are the product's target, not a repair: describe them separately in the audit and do NOT claim to deduplicate them here.
5. Write docs/quality_audit.md from the saved results (generated by script, not typed). For each fault include the Lecture 2 category, the example, the count, and the proposed action (drop, correct, impute, flag, or investigate) with the reason.
6. Tests for each detector using tiny hand-built frames.

End with the standard end-of-task report.
```

**Gate:** six or more faults, each with expression and count; counts come from the script.

---

## Prompt 4: Cleaning, provenance, validate() (Member A)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, docs/quality_audit.md and roadmap section 7.4.

Task: src/clean.py and src/validate.py.
1. Clean register_raw.csv into data/clean/register_clean.csv. Keep the original values in *_raw columns. Normalise case and whitespace, standardise dates to ISO, canonicalise city and gender, normalise phone to an 11-digit 03xxxxxxxxx form, set sentinels and impossible values to missing, flag them, and recompute or check age against DOB.
2. Do NOT use ground truth or the fault log in cleaning. Do NOT correct name typos toward a "true" name; matching them is the model's job. Do NOT remove near-duplicates.
3. For every repaired field add a provenance column (for example city_prov) with values such as original, case_fixed, alias_mapped, set_missing, reformatted. Every row must have a valid provenance value for each repaired field.
4. Produce a before/after table (reports/clean_effects.json and .csv): one row per repair with a NUMBER before and after (for example distinct city strings, invalid DOB count, sentinel count, malformed phone count, missing counts).
5. Write validate(df) in src/validate.py with AT LEAST 10 assertions. Each assertion has a name, a boolean result, and a violation count. Use the list in roadmap 7.4 as a starting point. Add the leakage guard (no forbidden column present).
6. Prove with tests that validate() PASSES on the clean data and FAILS on the raw data, and show how many assertions fail on raw.
7. Create a mapping of clean-frame columns that the models are allowed to use (an allowlist in config.py), so later code cannot pick up a forbidden column by accident.
8. Update docs/quality_audit.md or add docs/cleaning_report.md generated from the saved numbers.

End with the standard end-of-task report.
```

**Gate:** pytest proves pass-on-clean and fail-on-raw; before/after table has real numbers.

---

## Prompt 5: Blocking (Member B). The assigned challenge

```text
Read PROJECT_CONTEXT.md, DECISIONS.md and roadmap sections D5 and 7.5.

Task: src/blocking.py and docs/blocking_report.md.
1. Implement, from scratch with dictionaries (no record-linkage library), these blocking passes on the CLEANED register: B1 Soundex(family_name)+birth year; B2 first three letters of first_name + birth year-month; B3 last 7 digits of phone; B4 national_id exact; B5 sorted-neighbourhood over a normalised name key with a configurable window. Implement Soundex yourself or document the library used. Records with a missing key generate no block for that pass.
2. Candidate pairs are the de-duplicated union across passes. Provide per-pass and union functions. Add a max_block_size guard that skips or sub-blocks giant blocks (very common names) and records how many pairs were dropped by it.
3. Evaluate with ground truth (evaluation only): all-pairs count n(n-1)/2 (for 50,000 records this is 1,249,975,000), comparisons performed, reduction ratio = 1 - comparisons/all-pairs, pairs completeness (true duplicate pairs retained / all true pairs), pairs quality (true pairs / candidates), block-size distribution and the largest blocks, runtime. Produce this for each pass alone, for sensible combinations, and for the union.
4. Run a scaling experiment at 5k, 10k, 25k and 50k records showing candidate counts growing much slower than n squared.
5. Missed-duplicate analysis: for every true pair the union misses, record which corruptions defeated every key, and summarise the causes in a table.
6. Before looking at results, write into DECISIONS.md the target pairs completeness we will accept; then choose the scheme that meets it with the fewest comparisons, and justify. Save everything under reports/blocking.json and figures in reports/figures/.
7. docs/blocking_report.md generated from the saved numbers (no hand-typed figures), including a plain-English "what blocking costs us in missed duplicates" section.
8. Tests on tiny hand-built registers with known answers (including a missing-key case and an oversized block).

End with the standard end-of-task report.
```

**Gate:** you can state the all-pairs number, your comparison count, reduction ratio, completeness, and why the misses happen.

---

## Prompt 6: Pairs, features, split (Member B)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, docs/leakage_statement.md and roadmap sections D4, D5, 7.6.

Task: src/split.py, src/pairs.py, src/features.py.
1. Split: assign each true entity to train or test by its EARLIEST registration date (cutoff chosen so train is roughly the earlier 70%; record the cutoff date). All records of an entity go to the same side. Ground truth is used here and for labels only. Save the entity-to-side map. Assert no entity appears on both sides.
2. Pairs: within each side, run the chosen blocking scheme to produce candidate pairs; label them from ground truth (match / non-match). Report the match prevalence on each side. Keep the natural proportions for the evaluation set. For training, also build an optional re-sampled set (stratified, keeping all positives) and record the re-weighting needed to report population-level figures.
3. Features: implement the pair comparison vector in roadmap 7.6. Implement character n-gram Jaccard and TF-IDF cosine yourself (numpy/scikit-learn primitives are fine) so they can be explained; rapidfuzz is acceptable for Jaro-Winkler and Levenshtein. Include per-field both_present flags and an overall Gower similarity that skips missing fields using the delta rule. Handle missing values explicitly and document the treatment for each model family.
4. Forbidden-column guard: features.py must accept only allowlisted columns and raise an error otherwise. Test it.
5. Tests: tiny hand-built record pairs with known feature values (include Muhammad vs Mohammad, swapped day and month, missing national_id on one side, siblings, father and son).
6. Save labelled train and test pair files in data/pairs/ (compressed if large) and a feature dictionary doc docs/feature_dictionary.md listing each feature, its meaning in plain words, and the plain-language phrase we will later use in explanations.

End with the standard end-of-task report.
```

**Gate:** no entity overlap; feature dictionary has plain-language phrases; prevalence per side reported.

---

## Prompt 7: Pre-processing experiments (Member B)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, docs/feature_dictionary.md and roadmap sections 7.7 and 7.8.

Task: src/preprocess.py and docs/preprocessing_decisions.md. All comparisons use the TRAINING pairs only, StratifiedGroupKFold (5 folds, grouped by entity), fixed seed, identical folds across arms.
1. Rescaling decision per model family (decision tree, naive Bayes, k-NN, optional MLP): run a measured check of scaled versus unscaled where it matters, apply log transform to heavy-tailed features (for example DOB difference in days, visit_count difference), and justify each decision with Lecture 4 reasoning (units and ranges decide who is "nearest"; trees are scale-invariant).
2. Dimensionality reduction versus feature subset selection, compared on the SAME model with the SAME folds and metric: PCA (standardise first; choose number of components by cumulative variance and by CV) versus selection (filter by correlation with target plus dropping features correlated above 0.95; and greedy forward selection). Report performance, number of features, training time, and explainability. State which we ship and why (note PCA components are hard to explain in plain language).
3. Discretization for naive Bayes and rule-style models: compare equal-width, equal-frequency and supervised entropy-based binning on the same model; show the effect on bin sizes and on performance; justify the one we ship. Binarization: one-hot for nominal features such as the pair of source systems.
4. Missing-value treatment per family (sentinel plus flag for trees, explicit missing bin for naive Bayes, Gower-style distance or imputation plus flag for k-NN): compare at least two alternatives for one family and record the result.
5. Sampling: stratified sampling versus all negatives; a progressive-sampling learning curve (double the training size until the gain falls below a threshold we fix in DECISIONS.md before running).
6. Save every number to reports/preprocessing.json; produce the figures; generate docs/preprocessing_decisions.md from the saved numbers. Fit all transformers on training folds only inside the CV loop (no fitting on validation or test data). Provide a single sklearn Pipeline builder per model family for later prompts.
7. Tests that transformers fit on train only and that the pipeline can transform the test set without refitting.

End with the standard end-of-task report.
```

**Gate:** each decision is backed by a measured comparison; no leakage from fitting on all data.

---

## Prompt 8: Models and hyper-parameters (Member B)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, docs/preprocessing_decisions.md and roadmap sections D6 and 7.8.

Task: src/models.py and docs/model_selection.md.
1. FIRST write into DECISIONS.md the hyper-parameter procedure, before running anything: training pairs only, 5-fold StratifiedGroupKFold grouped by entity, fixed seed, objective (expected cost at the cost-optimal threshold, or PR-AUC; pick one and state why), grids, and the one-standard-error simplicity tie-break.
2. Models, all trained on the SAME prepared training data: (a) pruned decision tree (grid: max_depth, min_samples_leaf, criterion, ccp_alpha); (b) categorical naive Bayes on the shipped discretization with Laplace smoothing (grid: alpha, bin scheme and count). Include a small demonstration of the zero-probability problem and the effect of smoothing; (c) k-NN on scaled features (grid: k odd values, L1 versus L2, uniform versus distance weighting). Optional (d) small MLP only if the team approves in DECISIONS.md.
3. Save each fitted pipeline with joblib in models/ along with metadata (features used, preprocessing, chosen hyper-parameters, training date, seed, data hash).
4. Produce the CV table (mean and standard deviation for each metric, per model) and the chosen hyper-parameters; save to reports/models.json.
5. Do NOT evaluate on the test set in this task. Provide predict_proba for all models in a common interface (class Matcher with fit, predict_proba, explain stubs).
6. Tests: models load, produce probabilities in [0,1], and are deterministic for a fixed seed.
7. Generate docs/model_selection.md from saved results: the procedure, grids, chosen values, CV spread, and the reasoning for each family.

End with the standard end-of-task report.
```

**Gate:** procedure written before results; test set untouched.

---

## Prompt 9: Evaluation (Member B). 20 marks

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, docs/model_selection.md and roadmap section 7.9. This task carries the most marks. Be rigorous and honest.

Task: src/evaluate.py and docs/evaluation_report.md. The test set is used ONCE here.
1. Baselines on the test population: majority-class baseline accuracy (report the match prevalence) and a simple domain rule baseline ("same DOB and same phone"). Put the baseline beside EVERY accuracy figure in the report and in the saved JSON.
2. Metrics for all three models on the test pairs: accuracy (with baseline), precision, recall, F1, precision-recall AUC, ROC AUC. Write a short justification for why precision and recall suit a records clerk (false alarms waste review time; missed duplicates leave split medical histories; accuracy is dominated by non-matches).
3. ROC curves (all three on one figure) with AUC, precision-recall curves, and confusion matrices at the operating threshold.
4. Calibration: reliability curve per model; fit a calibrator (Platt or isotonic) on training out-of-fold predictions ONLY; show before and after.
5. Cost matrix: define a default cost matrix with a written justification placeholder for the team to complete (clearly marked TEAM TO JUSTIFY), derive the threshold (flag when p >= C_FP / (C_FP + C_FN) for calibrated probabilities) and also verify it by minimising expected cost on training out-of-fold predictions. Report expected cost per 1,000 candidate pairs at the cost-derived threshold versus 0.5, and state exactly how much accuracy was given up. Provide a function that recomputes everything for any user-edited cost matrix (the app will call it). Derive the second, stricter "very likely duplicate" band from a second cost ratio.
6. Cross-validation on the training portion: mean and standard deviation of each metric, per model.
7. End-to-end pipeline recall = blocking completeness x classifier recall, and end-to-end precision. Blocking misses must count as missed duplicates.
8. Leakage proof: demonstrate with and without a deliberately leaky feature (for example the record_id difference using an UNSHUFFLED copy of the data, or cluster size) to show how misleading accuracy becomes, then confirm the honest model excludes it.
9. Random versus temporal split: re-run one model under a random pair-level split and show what it would have falsely reported compared with the entity-grouped temporal split.
10. Robustness: regenerate a register with a different seed, run the frozen pipeline without retraining, and report the drop.
11. Workload view: worklist size and estimated clerk-hours at the chosen thresholds (state the assumed minutes per review in config.py and mark it as an assumption).
12. Limitations section: synthetic data, what each model cannot do, where it fails (include error analysis of false positives and false negatives with examples, using synthetic records only).
13. Save all numbers to reports/evaluation.json and figures to reports/figures/. Generate docs/evaluation_report.md from them (no typed numbers). Choose the model(s) to ship and justify using CV and test evidence together.
14. Tests: metric functions on tiny known examples; threshold function matches hand calculation.

End with the standard end-of-task report, including a list of anything that looks suspiciously good.
```

**Gate:** baseline everywhere; threshold from cost matrix; CV spread; leakage and random-split demonstrations present.

---

## Prompt 10: Explanation module (Member B with C)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, docs/feature_dictionary.md, the evaluation report and roadmap section 7.10.

Task: src/explain.py and docs/explanation_guide.md.
1. Implement explain(pair, model) returning a structured result: verdict, probability, confidence label, a list of reasons FOR a match and a list of reasons AGAINST, each as a plain-English sentence for a records clerk, plus per-field agree/conflict/unknown flags for the merge screen.
2. Decision tree: trace the path for the pair and convert each test into a sentence using the feature-to-phrase dictionary (examples: "Same date of birth", "First names are nearly identical (Muhammad / Mohammad)", "Phone numbers differ", "National ID is missing on one record so it could not be compared"). Never expose feature names, node numbers, gini or raw thresholds; convert thresholds into words ("very similar", "somewhat similar", "different").
3. Naive Bayes: compute each feature's log-likelihood-ratio contribution to the posterior and describe the top contributors for and against.
4. k-NN: retrieve the nearest labelled training pairs and describe them in words ("Most similar past pairs: 4 of 5 were the same person").
5. Include reasons against a match where relevant. Handle missing fields gracefully.
6. Confidence labels must be tied to the cost-derived bands (not arbitrary): for example Very likely duplicate, Worth reviewing, Unlikely.
7. Tests: for a set of hand-built pairs the explanations contain the expected phrases and NEVER contain jargon (regex for "node", "gini", "feature_", "<=", digits-with-many-decimals). Test that every explain() call returns at least one reason.
8. Write docs/explanation_guide.md: examples for each model and the phrase dictionary.
9. Ask me to review a sample of 20 explanations before finalising. Output them in a table with the pair, verdict and explanation.

End with the standard end-of-task report.
```

**Gate:** twenty sampled explanations read naturally to a non-technical person.

---

## Prompt 11: App foundation and screens 1-2 (Member C)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md and roadmap section 7.11.

Task: build the Streamlit app foundation and screens 1 and 2.
1. app.py (Home) plus pages/1_Data_and_Quality.py and pages/2_Model_Performance.py. Create placeholder pages for 3-6 so all six exist and open.
2. Shared module src/app_state.py for st.session_state: selected model, threshold, cost matrix, loaded data. A shared banner function shown on EVERY page: "Student project. Not fit for clinical, financial or legal use. All data is synthetic." Provide a short "how to use" help panel and sample-data buttons.
3. Screen 1: load the raw register (default demo_register.csv, upload allowed with schema validation and friendly errors); show the data dictionary; the audit report table (fault, category, expression, count); the before/after repair table; validate() results for raw versus clean showing it fails on raw and passes on clean with violation counts; a download button for the cleaned CSV. Never show ground truth.
4. Screen 2: comparison table for the three models (accuracy WITH the majority baseline in the adjacent column, precision, recall, F1, PR-AUC, ROC AUC, expected cost, CV mean +/- sd), ROC curves with AUC, confusion matrix for the selected model at the current threshold, a blocking summary card (all-pairs, comparisons, reduction ratio, pairs completeness), and the leakage and random-versus-temporal findings. All numbers read from reports/*.json, never recomputed by hand-typed values.
5. Use st.cache_data / st.cache_resource sensibly; page loads should feel fast. Plain, friendly language throughout; no jargon without a tooltip.
6. Write a smoke test that imports each page and a manual test checklist in docs/app_test_checklist.md.

End with the standard end-of-task report and the command to run the app.
```

**Gate:** all six pages open; screens 1-2 correct; baseline visible beside every accuracy.

---

## Prompt 12: Screens 3, 4, 5 (Member C)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, src/explain.py, src/evaluate.py and roadmap section 7.11.

Task: implement pages 3, 4 and 5.
Screen 5 Settings first (others depend on it): choose model (all three), move the threshold (slider with the cost-optimal value marked and a reset button), edit the cost matrix (cost of a false alarm and cost of a missed duplicate, with units explained and the assumption visible), and show LIVE: expected cost per 1,000 pairs, precision, recall, worklist size, estimated clerk hours, confusion matrix, and the accuracy given up compared with the baseline and with threshold 0.5. Use src/evaluate.py's recompute function. Persist settings in session state so all pages use them.

Screen 3 Single prediction: a form for Record A and Record B (all clean-able fields with realistic placeholders and a button that fills a few example pairs, including a hard negative like siblings and an easy match). Run the same cleaning and feature pipeline the models use, then show: duplicate probability, a confidence label tied to the cost-derived bands, the verdict at the current threshold, the plain-language reasons for and against, and field-by-field agreement. Handle missing or malformed inputs with helpful messages.

Screen 4 Batch scoring: upload a register CSV (sample_batch.csv offered). Validate columns, clean, run blocking, score all candidate pairs, and show: blocking statistics for THIS file (all-pairs, comparisons, reduction ratio), a ranked worklist (confidence, names, DOB, source systems, one-line reason), filters (confidence band, source system, city, minimum score), a sensible row limit for display, and a download of the full scored result as CSV. Use a progress bar. State the maximum recommended upload size and handle larger files gracefully. The worklist must respect the settings from screen 5.

Common: every prediction shows its reason; disclaimer banner; baseline is shown wherever accuracy is shown; no use of ground truth in the app. Add smoke tests and update docs/app_test_checklist.md.

End with the standard end-of-task report.
```

**Gate:** changing the cost matrix changes the threshold, screen 3 verdicts and screen 4 worklist.

---

## Prompt 13: Screen 6 Merge review and reversible merge (Member C)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, src/explain.py and roadmap sections D8 and 7.11.

Task: pages/6_Merge_Review.py and src/merge_store.py.
1. Merge store (SQLite via the standard library, file inside the project data folder, created automatically): tables for merges and decisions. Each merge stores a merge ID, both original record snapshots (full JSON), the chosen golden record, the per-field choice and provenance (which record each field came from), the clerk's note, timestamp, and status (merged, undone). Reject and Defer decisions are stored with timestamps.
2. Screen 6 workflow: takes the next pair from the screen 4 worklist (or a pair chosen from a list or sent from screen 3). Shows the two records side by side field by field, with agree / conflict / unknown highlighting (use colour AND text or icons, not colour alone), the confidence score, the confidence label and the plain-language reasons for and against.
3. Survivorship: default golden record suggestions (prefer non-missing, non-sentinel, most complete, most recent registration for contact details), with a per-field radio choice A or B and an optional edit.
4. Buttons: Merge (with a confirmation step showing exactly what will happen), Reject (record as not duplicates), Defer (put back in the queue). Never auto-merge. Warn when the model confidence is below the "very likely" band or when blood groups or national IDs conflict.
5. Merge log view with Undo that restores both original records EXACTLY. Provide a test that merges and then undoes and compares the restored data byte for byte with the originals.
6. Reject and Defer outcomes feed a "reviewed pairs" list that screen 4 uses to hide already-decided pairs. Optionally export decisions as labelled pairs.
7. Tests for the merge store (merge, undo, reject, defer, idempotent undo, concurrent re-open).
8. Update docs/app_test_checklist.md and add a short clerk-facing help text on the page.

End with the standard end-of-task report.
```

**Gate:** merge, then undo restores originals exactly; every action is logged.

---

## Prompt 14: Docs, packaging, clean-machine test (Member C with A and B)

```text
Read PROJECT_CONTEXT.md, DECISIONS.md and all of docs/.

Task: make the project shippable.
1. README.md: what the product is and who it is for; the ONE command to start (finalise run.sh and run.bat so a single command creates a virtual environment if needed, installs from requirements.txt, and launches the app; models and data ship in the repo so no training or download is required at start); Python version; folder map; where each stage's documents live; data provenance (synthetic, how it was generated, seed); a results summary generated from reports/*.json that includes the majority baseline beside every accuracy; known limitations; the student-project disclaimer; team members and who built what.
2. Confirm requirements.txt contains every import used anywhere. Write scripts/check_imports.py that scans the code and fails if an import is missing from requirements.txt.
3. scripts/clean_machine_test.sh and .bat: create a brand-new temporary virtual environment in a temp folder, install from requirements.txt, run pytest, start the app headless, request each page's health endpoint or run a smoke import, and report pass or fail. No absolute paths, no manual edits.
4. Ensure the repo size is reasonable and that every file the app needs is committed (data, models, reports JSON). List any large files and compress where useful.
5. docs/ethics_statement.md: who could be harmed if the product were wrong (patients wrongly merged, patients left with split histories, staff over-trusting the tool) and what the interface does to reduce that risk (no auto-merge, reasons shown, reversible merges, audit log, banner). Mark it as team-authored text to be edited by the team.
6. docs/team_contributions.md template for the team to fill in.
7. Regenerate all documents from the latest reports/*.json and verify no document contains a hand-typed metric that disagrees with the JSON (write a small script that cross-checks key numbers).
8. Add a usability test script docs/usability_test.md: tasks for a stranger (find the worklist, check a pair, change the cost matrix, merge and undo) with space to record where they got stuck.

End with the standard end-of-task report and the exact commands we will use for the demo.
```

**Gate:** clean-machine script passes on a different computer; README start command works with no edits.

---

## Prompt 15: Final audit against the brief

```text
Read PROJECT_CONTEXT.md, DECISIONS.md, PROJECT_ROADMAP.md (especially section 8 and section 11), the brief PDF if present, and the whole repository.

Task: act as a strict marker. Do not fix things silently; report first.
1. Build docs/traceability_audit.md: for every requirement in the common spec (A through G), the assigned challenge, the ethics requirement and the demo/viva requirement, state PASS, PARTIAL or FAIL, cite the exact file, line or screen as evidence, and what is missing.
2. Mechanical checks: run pytest; run scripts/check_imports.py; run the clean-machine test; run validate() on raw and clean and show the outcomes; grep the docs and app for any accuracy figure without an adjacent baseline; grep the code for true_entity_id used outside src/split.py, src/pairs.py, src/evaluate.py, src/generate_data.py and tests; grep explanation outputs for jargon; check every page shows the disclaimer; check no absolute paths; check that no metric appears in a doc that is not in reports/*.json.
3. Check for leakage routes: allowlist enforcement, record_id shuffling, entity overlap between train and test, any fitting on test data, thresholds tuned on test.
4. Check that screens 3, 4 and 6 never show a prediction without a reason; that screen 5 changes outcomes live; that merge undo restores exactly.
5. List the ten most likely viva questions per member given what is actually in the repo, flag any part of the code that is hard to explain or that you would struggle to defend, and recommend simplifications.
6. Produce a prioritised fix list ordered by marks at risk. Then WAIT for my approval before changing anything.
```

**Gate:** every FAIL and PARTIAL is resolved or consciously accepted and recorded in `DECISIONS.md`.

---

## Prompt 16: Viva preparation

```text
Read PROJECT_CONTEXT.md, PROJECT_ROADMAP.md section 10, DECISIONS.md and the repository.

Task: prepare us for the individual viva. We are three students: Member A built data (generator, audit, cleaning, validate), Member B built blocking, features, models, evaluation and explanation logic, Member C built the app, merge store and packaging.
1. Write docs/viva_guide.md with, for each member, 25 likely questions of increasing difficulty about THEIR part, each with a model answer that cites the actual file and numbers in this repository (read from reports/*.json; do not invent numbers). Include trick questions (why not 0.5? why is 98% accuracy poor here? what does a zero probability do to naive Bayes? why not clean the typos?).
2. Write a 12-minute demo script with timings: problem, data and quality (screen 1), models and honest evaluation (screen 2), a single hard pair (screen 3), a batch run showing blocking savings (screen 4), changing the cost matrix live (screen 5), a merge and an undo (screen 6), limitations and ethics. Include fallback steps if something breaks and a pre-demo checklist.
3. Create a cross-examination exercise: ten questions each member should be able to answer about the other two members' parts at a basic level.
4. Run a mock viva in this chat: ask me one question at a time, wait for my answer, grade it honestly, and give the model answer. Start with the member I name.
```
