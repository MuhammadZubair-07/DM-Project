# Hospital Patient Register: Data Profile Report

> **Project:** DSC306 Data Mining — Project 11 (Patient Record Deduplication Tool)
> **Phase:** Phase 2 (Data Profiling & Measurement Classification — Lecture 2)
> **Analyzed Dataset:** `data/raw/register_raw.csv` (54,373 records, 16 attributes)

---

## 1. Executive Summary & Dataset Dimensions

- **Total Records ($n$):** 54,373
- **Total Attributes ($d$):** 16
- **Measurement Scales Represented:** All 4 Stevens types present (Nominal: 11, Ordinal: 1, Interval: 2, Ratio: 2)
- **Data Integrity Status:** Intentionally corrupted synthetic hospital register simulating departmental silos, legacy migrations, human typographical errors, and sentinels.

---

## 2. Column-by-Column Attribute Profile & Lecture 2 Classification

| # | Column Name | Measurement Type | Scale Property | Role | Missing Count | Missing % | Unique | Min / Max |
|---|---|---|---|---|---|---|---|---|
| 1 | **`record_id`** | Nominal | Discrete | `identifier` | 0 | 0.0% | 54,373 | `REC0000001` / `REC0054373` |
| 2 | **`source_system`** | Nominal | Discrete | `explanatory` | 0 | 0.0% | 6 | `Emergency` / `Radiology` |
| 3 | **`registration_date`** | Interval | Discrete | `temporal_split_key` | 0 | 0.0% | 3,950 | `2015-01-01` / `2025-10-24` |
| 4 | **`first_name`** | Nominal | Discrete | `explanatory` | 0 | 0.0% | 2,675 | `A.` / `zubair` |
| 5 | **`family_name`** | Nominal | Discrete | `explanatory` | 0 | 0.0% | 1,452 | `.` / `Zuunaid` |
| 6 | **`father_name`** | Nominal | Discrete | `explanatory` | 5,035 | 9.26% | 2,428 | `Abid Abbasi` / `Zubair Zaman` |
| 7 | **`gender`** | Nominal | Binary | `explanatory` | 0 | 0.0% | 14 | `1` / `male` |
| 8 | **`date_of_birth`** | Interval | Discrete | `explanatory` | 492 | 0.9% | 36,772 | `01-01-1947` / `9198-05-18` |
| 9 | **`age_years`** | Ratio | Discrete | `explanatory` | 0 | 0.0% | 207 | `-999.0` / `1823.0` |
| 10 | **`national_id`** | Nominal | Discrete | `explanatory` | 4,095 | 7.53% | 44,574 | `045673022112` / `956189129420` |
| 11 | **`phone`** | Nominal | Discrete | `explanatory` | 3,131 | 5.76% | 46,072 | `+92300184013` / `03961720762` |
| 12 | **`city`** | Nominal | Discrete | `explanatory` | 0 | 0.0% | 69 | `BWP` / `sukkur` |
| 13 | **`address`** | Nominal | Discrete | `explanatory` | 0 | 0.0% | 47,061 | `House 1, St ` / `House 999, S` |
| 14 | **`blood_group`** | Nominal | Discrete | `explanatory` | 4,844 | 8.91% | 8 | `A+` / `O-` |
| 15 | **`insurance_tier`** | Ordinal | Discrete | `explanatory` | 5,566 | 10.24% | 4 | `Basic` / `Standard` |
| 16 | **`visit_count`** | Ratio | Discrete | `explanatory` | 0 | 0.0% | 54 | `-999.0` / `9999.0` |

---

## 3. Lecture 2 Measurement Type Justifications

In Data Mining (Lecture 2), attribute classification dictates which mathematical operations, distance metrics, and preprocessing transformations are mathematically valid.

### `record_id` (Nominal — Discrete)
- **Role:** `identifier`
- **Meaning:** Unique synthetic alphanumeric record identifier (shuffled)
- **Plausible Domain:** REC0000001 to REC0054373 (unique string per row)
- **Lecture 2 Rationale:** Only identity (=, !=) is defined. Order is purely arbitrary; arithmetic operations (addition, averaging) are meaningless. Never used as a predictive feature.

### `source_system` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Hospital department or legacy subsystem that created the record
- **Plausible Domain:** {OPD, Lab, Radiology, Pharmacy, Emergency, Legacy}
- **Lecture 2 Rationale:** Categorical department labels with no intrinsic hierarchy or numeric scale. Only equality/frequency tests are meaningful. Used cautiously due to potential sampling leakage (see docs/leakage_statement.md).

### `registration_date` (Interval — Discrete)
- **Role:** `temporal_split_key`
- **Meaning:** Calendar date when the patient record was registered in the subsystem
- **Plausible Domain:** 2015-01-01 to 2025-10-24 (ISO or raw legacy formats)
- **Lecture 2 Rationale:** Calendar time has an arbitrary zero point (Gregorian origin). Differences between dates (elapsed days) are physically meaningful, but ratios (e.g. 2020 / 1010) are meaningless. Used for temporal train/test split.

### `first_name` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Patient given / first name (Pakistani naming convention)
- **Plausible Domain:** Common Pakistani given names and corrupted spelling variants
- **Lecture 2 Rationale:** Text categorical string. Equality and string similarity (Levenshtein, Jaro-Winkler) are defined, but no natural order or arithmetic exists.

### `family_name` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Patient family name, tribe, caste, or surname
- **Plausible Domain:** Common Pakistani family names (Khan, Ahmed, Ali, Malik, etc.)
- **Lecture 2 Rationale:** Text categorical string. Nominal identity; used for blocking and similarity matching.

### `father_name` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Father or guardian's full name (critical patronymic identifier in Pakistan)
- **Plausible Domain:** Text string, frequently omitted in Emergency registrations
- **Lecture 2 Rationale:** Text categorical string. Equality and string similarity are meaningful; no ordering or arithmetic.

### `gender` (Nominal — Binary)
- **Role:** `explanatory`
- **Meaning:** Administrative / biological gender code
- **Plausible Domain:** M, F (clean); {M, F, male, female, 1, 2, Ml, Fm} (dirty raw)
- **Lecture 2 Rationale:** Conceptually symmetric binary (M/F), but raw data contains 14 dirty nominal encodings. No hierarchy or magnitude exists between categories.

### `date_of_birth` (Interval — Discrete)
- **Role:** `explanatory`
- **Meaning:** Patient calendar birth date
- **Plausible Domain:** 1925-01-01 to 2025-10-24 (plus sentinels such as 1900-01-01)
- **Lecture 2 Rationale:** Calendar date on an interval scale. Differences measure elapsed age, but calendar year 0 is an arbitrary convention, making multiplication/ratio invalid.

### `age_years` (Ratio — Discrete)
- **Role:** `explanatory`
- **Meaning:** Chronological age in completed years at time of registration
- **Plausible Domain:** 0 to 110 (raw contains corrupt sentinels: -999, -1, 199, 999, 1823)
- **Lecture 2 Rationale:** True absolute zero exists (age 0 = newborn, absence of elapsed life years). Ratios are physically valid: a person of 40 is twice as old as a person of 20.

### `national_id` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Synthetic 13-digit Pakistani CNIC number (5-digit area + 7-digit serial + 1 check)
- **Plausible Domain:** 13-digit numeric string (synthetic; never real citizen data)
- **Lecture 2 Rationale:** Despite being composed of digits, national_id is NOMINAL. Digits serve purely as categorical labels. The leading digit represents province; adding two CNICs or calculating their average is completely nonsensical. Doubling a CNIC has no meaning.

### `phone` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Mobile contact phone number (Pakistani 11-digit mobile format: 03xxxxxxxxx)
- **Plausible Domain:** 11-digit string starting with 03 (plus formatting noise and sentinels)
- **Lecture 2 Rationale:** Phone numbers are NOMINAL digit-strings. Leading '0' is syntactically mandatory and would be destroyed if treated as a numeric scalar. Arithmetic addition and scaling have zero physical meaning.

### `city` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Patient residential city in Pakistan
- **Plausible Domain:** 16 canonical cities plus noisy abbreviations (LHR, KHI, ISB, etc.)
- **Lecture 2 Rationale:** Polytomous nominal categories representing geographic localities without ordering.

### `address` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Free-text street, house, sector, and area address line
- **Plausible Domain:** Free-text address strings (House ..., Street ..., Area)
- **Lecture 2 Rationale:** Unstructured text identifier with nominal properties. Evaluated via n-gram token overlap and edit distance rather than numeric arithmetic.

### `blood_group` (Nominal — Discrete)
- **Role:** `explanatory`
- **Meaning:** ABO and Rhesus blood antigen classification
- **Plausible Domain:** {A+, A-, B+, B-, AB+, AB-, O+, O-}
- **Lecture 2 Rationale:** Biological categories with no intrinsic mathematical ordering or magnitude.

### `insurance_tier` (Ordinal — Discrete)
- **Role:** `explanatory`
- **Meaning:** Patient hospital coverage benefit tier
- **Plausible Domain:** None < Basic < Standard < Premium (with missingness from Lab)
- **Lecture 2 Rationale:** Strictly ordered categories: None < Basic < Standard < Premium. Ranking is mathematically meaningful (<, >), but distance between levels is neither uniform nor measurable (Basic to Standard != Standard to Premium).

### `visit_count` (Ratio — Discrete)
- **Role:** `explanatory`
- **Meaning:** Cumulative hospital outpatient or inpatient visit count
- **Plausible Domain:** 0 to 500 (raw contains corrupt sentinels: -999, 9999)
- **Lecture 2 Rationale:** Count attribute with an absolute physical zero (0 visits = complete absence of encounters). Multiplicative ratios are valid: 10 visits is exactly twice as many encounters as 5 visits.

---

## 4. Missingness Analysis & Departmental Gaps

Missingness in healthcare data is rarely missing completely at random (MCAR). The profiling reveals structural collection patterns:

1. **`father_name` missingness (5,035 records, 9.26%):** Concentrated in **Emergency** registrations where acute triage prevents full family tree intake.
2. **`national_id` missingness (4,095 records, 7.53%):** Heavily concentrated in **Legacy** migration records where national identity cards were not captured, as well as Emergency patients.
3. **`phone` missingness (3,131 records, 5.76%):** Missing in Emergency cases and pediatric patients.
4. **`date_of_birth` missingness (492 records, 0.90%):** True missing DOBs, alongside an additional 392 records with corrupt sentinel date `1900-01-01`.
5. **`insurance_tier` distinction:** Exactly 5,566 records (10.24%) have missing data (empty string) due to subsystems like Lab that do not capture billing tiers, while 19,357 records (35.60%) explicitly carry the tier value `'None'` (uninsured patient).

---

## 5. Categorical Distributions (Top Values)

### Distribution: `source_system`

| Value | Frequency | Percentage |
|---|---|---|
| `OPD` | 19,076 | 35.08% |
| `Lab` | 11,012 | 20.25% |
| `Pharmacy` | 8,134 | 14.96% |
| `Radiology` | 8,021 | 14.75% |
| `Emergency` | 5,468 | 10.06% |
| `Legacy` | 2,662 | 4.9% |

### Distribution: `gender`

| Value | Frequency | Percentage |
|---|---|---|
| `M` | 26,924 | 49.52% |
| `F` | 26,039 | 47.89% |
| `female` | 142 | 0.26% |
| `1` | 129 | 0.24% |
| `male` | 128 | 0.24% |
| `2` | 124 | 0.23% |
| `Male` | 123 | 0.23% |
| `Fm` | 118 | 0.22% |
| `f` | 115 | 0.21% |
| `m` | 109 | 0.2% |
| `Female` | 107 | 0.2% |
| `Ml` | 107 | 0.2% |
| `FEMALE` | 106 | 0.19% |
| `MALE` | 102 | 0.19% |

### Distribution: `insurance_tier`

| Value | Frequency | Percentage |
|---|---|---|
| `None` | 19,357 | 35.6% |
| `Basic` | 14,659 | 26.96% |
| `Standard` | 9,832 | 18.08% |
| `(missing/empty)` | 5,566 | 10.24% |
| `Premium` | 4,959 | 9.12% |

### Distribution: `blood_group`

| Value | Frequency | Percentage |
|---|---|---|
| `B+` | 6,327 | 11.64% |
| `AB+` | 6,325 | 11.63% |
| `O-` | 6,245 | 11.49% |
| `A-` | 6,231 | 11.46% |
| `AB-` | 6,154 | 11.32% |
| `A+` | 6,091 | 11.2% |
| `O+` | 6,090 | 11.2% |
| `B-` | 6,066 | 11.16% |
| `(missing/empty)` | 4,844 | 8.91% |

---

## 6. Numerical Attribute Distributions & Anomaly Flags

### Numerical Summary: `age_years`

- **Mean ± Std:** 41.72 ± 78.95
- **Quartiles [Min, 25%, Median, 75%, Max]:** [-999.0, 16.0, 39.0, 62.0, 1823.0]
- **Negative Values Detected (Sentinels):** 146 records
- **Extreme / Impossible Outliers:** 321 records

### Numerical Summary: `visit_count`

- **Mean ± Std:** 40.58 ± 423.73
- **Quartiles [Min, 25%, Median, 75%, Max]:** [-999.0, 12.0, 25.0, 38.0, 9999.0]
- **Negative Values Detected (Sentinels):** 218 records
- **Extreme / Impossible Outliers:** 97 records

