# Data Quality Audit Report

> **Course:** DSC306 Data Mining — COMSATS University Islamabad
> **Project:** Project 11 — Patient Record Deduplication Tool
> **Topic:** Comprehensive Data Quality Audit (Lecture 2 & Roadmap Section 7.3)
> **Dataset Audited:** `data/raw/register_raw.csv` (54,373 records, 16 attributes)

---

## 1. Executive Summary & Audit Methodology

In accordance with Lecture 2 (*Data Quality, Preprocessing, and Noise Taxonomy*), we deployed detectors across **eight distinct fault categories**. Real hospital patient registers contain complex data quality issues resulting from distributed entry, acute clinical pressure, legacy software transitions, and typing mistakes.

### Key Principles of the Audit:
1. **Near-Duplicates are NOT Repaired Here:** Near-duplicate patient profiles represent our product's primary mining target. They are explicitly separated and resolved via candidate blocking and pairwise classification, never unilaterally collapsed in this audit.
2. **No Data Snooping:** Detectors operate strictly on observed attribute syntax and distributions. Ground-truth fault logs were inspected exclusively for post-hoc honesty validation.
3. **Lecture 2 Actions:** Every detected anomaly is mapped to an action (*drop*, *correct*, *impute*, *flag*, or *investigate*) with explicit reasoning.

---

## 2. Fault Catalogue Table

| Code | Category | Fault Description | Affected Columns | Expression / Rule | Count | % | Proposed Action |
|---|---|---|---|---|---|---|---|
| `F1.1` | **Duplicate** | Exact duplicate rows (ignoring record_id) | `source_system, registration_date...` | `df.duplicated(subset=[col for co...` | 0 | 0.0% | **drop** |
| `F2.1` | **Inconsistent encoding** | Non-canonical city names / aliases (LHR, lahore, etc.) | `city` | `(df['city'] != '') & (~df['city'...` | 2,324 | 4.27% | **correct** |
| `F2.2` | **Inconsistent encoding** | Dirty gender encodings (male, female, 1, 2, Ml, Fm) | `gender` | `(df['gender'] != '') & (~df['gen...` | 1,410 | 2.59% | **correct** |
| `F2.3` | **Inconsistent encoding** | Non-ISO date of birth formats (DD/MM/YYYY, DD-MM-YYYY) | `date_of_birth` | `(df['date_of_birth'] != '') & (~...` | 17,932 | 32.98% | **correct** |
| `F2.4` | **Inconsistent encoding** | Non-standard phone format (dashes, spaces, +92 country code) | `phone` | `(df['phone'] != '') & (~df['phon...` | 1,642 | 3.02% | **correct** |
| `F2.5` | **Inconsistent encoding** | Name casing and whitespace noise (extra spaces, lowercase) | `first_name, family_name` | `df['first_name'].str.contains(r'...` | 2,213 | 4.07% | **correct** |
| `F3.1` | **Sentinel value** | Sentinel date of birth (1900-01-01 legacy default) | `date_of_birth` | `df['date_of_birth'].isin(SENTINE...` | 394 | 0.72% | **flag** |
| `F3.2` | **Sentinel value** | Sentinel age values (-999, -1, 999, 199) | `age_years` | `df['age_years'].isin(['-999', '-...` | 300 | 0.55% | **flag** |
| `F3.3` | **Sentinel value** | Sentinel phone numbers (00000000000, 11111111111, etc.) | `phone` | `df['phone'].isin(SENTINEL_PHONES...` | 183 | 0.34% | **flag** |
| `F3.4` | **Sentinel value** | Sentinel visit counts (-999, -1) | `visit_count` | `df['visit_count'].isin(['-999', ...` | 218 | 0.4% | **flag** |
| `F4.1` | **Invalid value** | Impossible birth date occurring after registration date | `date_of_birth, registration_date` | `(pd.to_datetime(df['date_of_birt...` | 3,446 | 6.34% | **flag** |
| `F4.2` | **Invalid value** | Impossible age values outside lifespan (>110 or <0, non-sentinel) | `age_years` | `(pd.to_numeric(df['age_years']) ...` | 181 | 0.33% | **correct** |
| `F4.3` | **Invalid value** | Age inconsistent with date of birth (|calc_age - age_years| > 2 years) | `age_years, date_of_birth` | `abs(df['age_years'] - (df['regis...` | 2,352 | 4.33% | **investigate** |
| `F4.4` | **Invalid value** | Malformed national ID (length != 13 or non-digit characters) | `national_id` | `(df['national_id'] != '') & (~df...` | 0 | 0.0% | **flag** |
| `F5.father_name` | **Missing value** | Omitted / blank attribute (father_name) | `father_name` | `(df['father_name'].str.strip() =...` | 5,035 | 9.26% | **impute** |
| `F5.national_id` | **Missing value** | Omitted / blank attribute (national_id) | `national_id` | `(df['national_id'].str.strip() =...` | 4,095 | 7.53% | **impute** |
| `F5.phone` | **Missing value** | Omitted / blank attribute (phone) | `phone` | `(df['phone'].str.strip() == '')...` | 3,131 | 5.76% | **impute** |
| `F5.date_of_birth` | **Missing value** | Omitted / blank attribute (date_of_birth) | `date_of_birth` | `(df['date_of_birth'].str.strip()...` | 492 | 0.9% | **impute** |
| `F5.blood_group` | **Missing value** | Omitted / blank attribute (blood_group) | `blood_group` | `(df['blood_group'].str.strip() =...` | 4,844 | 8.91% | **impute** |
| `F5.insurance_tier` | **Missing value** | Omitted / blank attribute (insurance_tier) | `insurance_tier` | `(df['insurance_tier'].str.strip(...` | 5,566 | 10.24% | **impute** |
| `F6.1` | **Measurement (entry) error** | Name typographical errors / rare variants outside Pakistani lexicon | `first_name, family_name` | `~df['first_name'].str.lower().is...` | 4,632 | 8.52% | **investigate** |
| `F7.1` | **Outlier** | Extreme visit count far above robust MAD threshold (>100 / 9999) | `visit_count` | `(robust_z > 3.5) & (df['visit_co...` | 97 | 0.18% | **investigate** |
| `F8.1` | **Collection error** | Systematic omission of national_id in Legacy subsystem (100% missing) | `national_id, source_system` | `(df['source_system'] == 'Legacy'...` | 2,662 | 4.9% | **flag** |
| `F8.2` | **Collection error** | Systematic omission of father_name in Emergency subsystem (70% missing) | `father_name, source_system` | `(df['source_system'] == 'Emergen...` | 3,858 | 7.1% | **flag** |

---

## 3. Detailed Fault Analysis by Lecture 2 Category

### `F1.1`: Exact duplicate rows (ignoring record_id) (Duplicate)
- **Affected Attributes:** `source_system, registration_date, first_name, family_name, father_name, gender, date_of_birth, age_years, national_id, phone, city, address, blood_group, insurance_tier, visit_count`
- **Detecting Expression:** `df.duplicated(subset=[col for col in df.columns if col != 'record_id'], keep='first')`
- **Prevalence in Raw Register:** **0 records (0.0%)**
- **Proposed Action:** `drop`
- **Methodological Rationale:** True duplicates with identical attributes provide zero additional information and distort model training.

### `F2.1`: Non-canonical city names / aliases (LHR, lahore, etc.) (Inconsistent encoding)
- **Affected Attributes:** `city`
- **Detecting Expression:** `(df['city'] != '') & (~df['city'].isin(CANONICAL_CITIES))`
- **Prevalence in Raw Register:** **2,324 records (4.27%)**
- **Proposed Action:** `correct`
- **Methodological Rationale:** Standardize aliases (LHR -> Lahore) using canonical city dictionary to prevent false mismatches.

### `F2.2`: Dirty gender encodings (male, female, 1, 2, Ml, Fm) (Inconsistent encoding)
- **Affected Attributes:** `gender`
- **Detecting Expression:** `(df['gender'] != '') & (~df['gender'].isin(['M', 'F']))`
- **Prevalence in Raw Register:** **1,410 records (2.59%)**
- **Proposed Action:** `correct`
- **Methodological Rationale:** Map dirty codes to canonical {M, F} to enable symmetric binary agreement matching.

### `F2.3`: Non-ISO date of birth formats (DD/MM/YYYY, DD-MM-YYYY) (Inconsistent encoding)
- **Affected Attributes:** `date_of_birth`
- **Detecting Expression:** `(df['date_of_birth'] != '') & (~df['date_of_birth'].str.match(r'^\d{4}-\d{2}-\d{2}$'))`
- **Prevalence in Raw Register:** **17,932 records (32.98%)**
- **Proposed Action:** `correct`
- **Methodological Rationale:** Parse and reformat dates to ISO 8601 (YYYY-MM-DD) for uniform datetime parsing.

### `F2.4`: Non-standard phone format (dashes, spaces, +92 country code) (Inconsistent encoding)
- **Affected Attributes:** `phone`
- **Detecting Expression:** `(df['phone'] != '') & (~df['phone'].isin(SENTINEL_PHONES)) & (~df['phone'].str.match(r'^03\d{9}$'))`
- **Prevalence in Raw Register:** **1,642 records (3.02%)**
- **Proposed Action:** `correct`
- **Methodological Rationale:** Strip non-digits, replace +92 with 0, and format as uniform 11-digit mobile strings (03xxxxxxxxx).

### `F2.5`: Name casing and whitespace noise (extra spaces, lowercase) (Inconsistent encoding)
- **Affected Attributes:** `first_name, family_name`
- **Detecting Expression:** `df['first_name'].str.contains(r'^\s|\s$|\s{2,}') | ~df['first_name'].str.istitle()`
- **Prevalence in Raw Register:** **2,213 records (4.07%)**
- **Proposed Action:** `correct`
- **Methodological Rationale:** Strip leading/trailing whitespace, collapse internal spaces, and apply Title Case.

### `F3.1`: Sentinel date of birth (1900-01-01 legacy default) (Sentinel value)
- **Affected Attributes:** `date_of_birth`
- **Detecting Expression:** `df['date_of_birth'].isin(SENTINEL_DATES)`
- **Prevalence in Raw Register:** **394 records (0.72%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Replace sentinel with NaN and create boolean flag 'dob_sentinel_flag' to prevent artificial age calculations.

### `F3.2`: Sentinel age values (-999, -1, 999, 199) (Sentinel value)
- **Affected Attributes:** `age_years`
- **Detecting Expression:** `df['age_years'].isin(['-999', '-1', '999', '199'])`
- **Prevalence in Raw Register:** **300 records (0.55%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Replace sentinel with NaN, impute from registration_date - DOB if valid, and add provenance flag.

### `F3.3`: Sentinel phone numbers (00000000000, 11111111111, etc.) (Sentinel value)
- **Affected Attributes:** `phone`
- **Detecting Expression:** `df['phone'].isin(SENTINEL_PHONES)`
- **Prevalence in Raw Register:** **183 records (0.34%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Replace with NaN to avoid false-positive blocking and matching on dummy phone numbers.

### `F3.4`: Sentinel visit counts (-999, -1) (Sentinel value)
- **Affected Attributes:** `visit_count`
- **Detecting Expression:** `df['visit_count'].isin(['-999', '-1'])`
- **Prevalence in Raw Register:** **218 records (0.4%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Replace with NaN or median encounter count with sentinel provenance flag.

### `F4.1`: Impossible birth date occurring after registration date (Invalid value)
- **Affected Attributes:** `date_of_birth, registration_date`
- **Detecting Expression:** `(pd.to_datetime(df['date_of_birth']) > pd.to_datetime(df['registration_date']))`
- **Prevalence in Raw Register:** **3,446 records (6.34%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Flag invalid temporal sequence; attempt day-month swap inversion or set DOB to missing with flag.

### `F4.2`: Impossible age values outside lifespan (>110 or <0, non-sentinel) (Invalid value)
- **Affected Attributes:** `age_years`
- **Detecting Expression:** `(pd.to_numeric(df['age_years']) < 0) | (pd.to_numeric(df['age_years']) > 110)`
- **Prevalence in Raw Register:** **181 records (0.33%)**
- **Proposed Action:** `correct`
- **Methodological Rationale:** Recompute age from registration_date and date_of_birth if DOB is valid; otherwise set missing.

### `F4.3`: Age inconsistent with date of birth (|calc_age - age_years| > 2 years) (Invalid value)
- **Affected Attributes:** `age_years, date_of_birth`
- **Detecting Expression:** `abs(df['age_years'] - (df['registration_date'] - df['date_of_birth'])/365.25) > 2.0`
- **Prevalence in Raw Register:** **2,352 records (4.33%)**
- **Proposed Action:** `investigate`
- **Methodological Rationale:** Reconcile discrepancy using registration date and DOB, flagging the repaired field.

### `F4.4`: Malformed national ID (length != 13 or non-digit characters) (Invalid value)
- **Affected Attributes:** `national_id`
- **Detecting Expression:** `(df['national_id'] != '') & (~df['national_id'].str.match(r'^\d{13}$'))`
- **Prevalence in Raw Register:** **0 records (0.0%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Strip punctuation; if length is not exactly 13 digits, set to missing with a malformed flag.

### `F5.father_name`: Omitted / blank attribute (father_name) (Missing value)
- **Affected Attributes:** `father_name`
- **Detecting Expression:** `(df['father_name'].str.strip() == '')`
- **Prevalence in Raw Register:** **5,035 records (9.26%)**
- **Proposed Action:** `impute`
- **Methodological Rationale:** Handle missingness according to model family (explicit missing category, Gower delta rule).

### `F5.national_id`: Omitted / blank attribute (national_id) (Missing value)
- **Affected Attributes:** `national_id`
- **Detecting Expression:** `(df['national_id'].str.strip() == '')`
- **Prevalence in Raw Register:** **4,095 records (7.53%)**
- **Proposed Action:** `impute`
- **Methodological Rationale:** Handle missingness according to model family (explicit missing category, Gower delta rule).

### `F5.phone`: Omitted / blank attribute (phone) (Missing value)
- **Affected Attributes:** `phone`
- **Detecting Expression:** `(df['phone'].str.strip() == '')`
- **Prevalence in Raw Register:** **3,131 records (5.76%)**
- **Proposed Action:** `impute`
- **Methodological Rationale:** Handle missingness according to model family (explicit missing category, Gower delta rule).

### `F5.date_of_birth`: Omitted / blank attribute (date_of_birth) (Missing value)
- **Affected Attributes:** `date_of_birth`
- **Detecting Expression:** `(df['date_of_birth'].str.strip() == '')`
- **Prevalence in Raw Register:** **492 records (0.9%)**
- **Proposed Action:** `impute`
- **Methodological Rationale:** Handle missingness according to model family (explicit missing category, Gower delta rule).

### `F5.blood_group`: Omitted / blank attribute (blood_group) (Missing value)
- **Affected Attributes:** `blood_group`
- **Detecting Expression:** `(df['blood_group'].str.strip() == '')`
- **Prevalence in Raw Register:** **4,844 records (8.91%)**
- **Proposed Action:** `impute`
- **Methodological Rationale:** Handle missingness according to model family (explicit missing category, Gower delta rule).

### `F5.insurance_tier`: Omitted / blank attribute (insurance_tier) (Missing value)
- **Affected Attributes:** `insurance_tier`
- **Detecting Expression:** `(df['insurance_tier'].str.strip() == '')`
- **Prevalence in Raw Register:** **5,566 records (10.24%)**
- **Proposed Action:** `impute`
- **Methodological Rationale:** Handle missingness according to model family (explicit missing category, Gower delta rule).

### `F6.1`: Name typographical errors / rare variants outside Pakistani lexicon (Measurement (entry) error)
- **Affected Attributes:** `first_name, family_name`
- **Detecting Expression:** `~df['first_name'].str.lower().isin(first_lexicon) | ~df['family_name'].str.lower().isin(fam_lexicon)`
- **Prevalence in Raw Register:** **4,632 records (8.52%)**
- **Proposed Action:** `investigate`
- **Methodological Rationale:** DO NOT correct typos toward ground truth. Matching noisy names via edit distance is the model's job.

### `F7.1`: Extreme visit count far above robust MAD threshold (>100 / 9999) (Outlier)
- **Affected Attributes:** `visit_count`
- **Detecting Expression:** `(robust_z > 3.5) & (df['visit_count'] > 100)`
- **Prevalence in Raw Register:** **97 records (0.18%)**
- **Proposed Action:** `investigate`
- **Methodological Rationale:** Cap at plausible maximum (e.g. 100) or log-transform; do not blindly delete (Lecture 2).

### `F8.1`: Systematic omission of national_id in Legacy subsystem (100% missing) (Collection error)
- **Affected Attributes:** `national_id, source_system`
- **Detecting Expression:** `(df['source_system'] == 'Legacy') & (df['national_id'] == '')`
- **Prevalence in Raw Register:** **2,662 records (4.9%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Document structural gap; treat as systemic collection artifact rather than random missingness.

### `F8.2`: Systematic omission of father_name in Emergency subsystem (70% missing) (Collection error)
- **Affected Attributes:** `father_name, source_system`
- **Detecting Expression:** `(df['source_system'] == 'Emergency') & (df['father_name'] == '')`
- **Prevalence in Raw Register:** **3,858 records (7.1%)**
- **Proposed Action:** `flag`
- **Methodological Rationale:** Recognize emergency triage workflow; do not drop rows or assume missing completely at random.

---

## 4. Missingness Analysis & Source-System Dependency Proof

In Data Mining Lecture 2, understanding **why** data is missing dictates the appropriate remedy. We tested the null hypothesis ($H_0$) that missingness is independent of the originating `source_system` (Missing Completely at Random — MCAR) using Chi-Square Contingency Tests ($\chi^2$):

| Attribute | $\chi^2$ Statistic | p-value | Degrees of Freedom | Conclusion |
|---|---|---|---|---|
| `national_id` | 34,367.48 | `0.00e+00` | 5 | **REJECT $H_0$ (Missingness is NOT Random)** |
| `father_name` | 27,184.21 | `0.00e+00` | 5 | **REJECT $H_0$ (Missingness is NOT Random)** |
| `phone` | 13,915.45 | `0.00e+00` | 5 | **REJECT $H_0$ (Missingness is NOT Random)** |
| `insurance_tier` | 24,416.17 | `0.00e+00` | 5 | **REJECT $H_0$ (Missingness is NOT Random)** |

### Departmental Breakdown of Gaps:
- **`national_id` in Legacy System:** 100% missing (2,662 / 2,662 records). This is a textbook **Collection Error (F8)**: national ID numbers were not captured prior to the legacy database modernization.
- **`father_name` in Emergency Department:** 70.56% missing (3,858 / 5,468 records). In trauma triage, unaccompanied patients are stabilized without full family genealogical intake.
- **`phone` in Emergency Department:** 40.55% missing (2,242 / 5,468 records). Phone numbers are omitted during acute emergency admissions.
- **`insurance_tier` in Laboratory Subsystem:** 50.54% missing (5,566 / 11,012 records). Pathology sample slips do not track billing insurance tiers.

---

## 5. Detector Performance vs. Ground-Truth Fault Log (Honesty Check)

To evaluate our detectors honestly, we performed a post-hoc comparison against `data/truth/fault_log.csv`. The detectors achieved high recall across injected error categories:

| Fault Type | Ground Truth Injected | Detected by Audit | True Positives | Detector Precision | Detector Recall |
|---|---|---|---|---|---|
| `city_alias` | 2,324 | 2,324 | 2,324 | 100.0% | 100.0% |
| `gender_dirty` | 1,410 | 1,410 | 1,410 | 100.0% | 100.0% |
| `sentinel_dob` | 392 | 394 | 392 | 99.5% | 100.0% |
| `name_typos` | 3,563 | 4,632 | 3,555 | 76.8% | 99.8% |

> [!NOTE]
> Detector precision for name typos reflects that natural name variations (e.g. compound Pakistani names or unusual regional spellings) > also trigger the base lexicon test, which is expected behavior for an unsupervised dictionary filter.

---

## 6. Near-Duplicates are the Target, Not a Repair

In a deduplication project, **near-duplicate patient records must NEVER be removed or unified during preprocessing**. The records clerk's primary task is resolving ambiguous duplicate charts. Collapsing near-duplicates during data cleaning would destroy the evaluation dataset and preempt the machine learning pipeline.

Therefore, Phase 4 (Cleaning) will correct encodings, standardize formats, and isolate sentinels, but will **strictly preserve** all candidate duplicate records for candidate blocking (Phase 5) and pairwise matching (Phase 6).
