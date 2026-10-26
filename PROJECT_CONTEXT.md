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
