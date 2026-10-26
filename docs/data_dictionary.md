# Data Dictionary: Hospital Patient Register

> **Course:** DSC306 Data Mining — COMSATS University Islamabad  
> **Project:** Project 11 — Patient Record Deduplication Tool  
> **Topic:** Data Dictionary, Measurement Types, Scale Properties, and Pipeline Roles (Lecture 2)  
> **Reference Files:** `data/raw/register_raw.csv`, `data/truth/ground_truth.csv`, `data/truth/fault_log.csv`

---

## 1. Overview & Lecture 2 Framework

In accordance with Lecture 2 (*Data Preprocessing, Attribute Types, and Measurement Scales* based on S.S. Stevens' 1946 taxonomy), every attribute in a mining pipeline must be formally categorized by:
1. **Measurement Type (Scale):** Nominal, Ordinal, Interval, or Ratio.
2. **Scale Property:** Discrete, Continuous, or Binary (symmetric vs asymmetric).
3. **Pipeline Role:** Identifier, Explanatory (Feature), Temporal Split Key, Target, or Must-Not-Use.

### The Four Measurement Scales (Stevens' Taxonomy)

| Scale | Permitted Operators | Central Tendency | Meaning of Zero | Meaning of Doubling | Prohibited Operations |
|---|---|---|---|---|---|
| **Nominal** | Equality ($=$, $\neq$) | Mode | None (label only) | None | Order ($<, >$), addition, subtraction, multiplication, mean |
| **Ordinal** | Equality ($=$, $\neq$), Order ($<, \le, >, \ge$) | Median, Percentiles | Arbitrary rank | No physical meaning | Distance, addition, subtraction, multiplication, mean |
| **Interval** | Equality, Order, Differences ($A - B$) | Mean, Std, Median | Arbitrary convention | Arbitrary ratio (no "twice as much") | Multiplication, division, true zero |
| **Ratio** | Equality, Order, Differences, Multiplication, Division | Geometric Mean, Harmonic Mean, Mean | Absolute natural zero (absence of quantity) | Valid physical ratio ("twice as much") | None |

> [!IMPORTANT]
> **Identifiers vs Numbers:** Digit strings such as `national_id` (13 digits) and `phone` (11 digits) are composed of numerical characters but are **strictly Nominal**. Adding two CNICs or averaging phone numbers is mathematically meaningless; leading zeros have positional syntax (e.g., province code in CNIC, trunk prefix in mobile numbers). Treating digit strings as continuous or ratio scalars destroys information.

---

## 2. Master Attribute Register

### 2.1 Raw Register Attributes (`data/raw/register_raw.csv`)

| # | Column Name | Meaning | Measurement Type | Scale Property | Plausible Domain Range | Pipeline Role |
|---|---|---|---|---|---|---|
| 1 | `record_id` | Unique synthetic record identifier | **Nominal** | Discrete | `REC0000001` to `REC0054373` | **Identifier** (Never a feature) |
| 2 | `source_system` | Originating hospital subsystem | **Nominal** | Discrete | `{OPD, Lab, Radiology, Pharmacy, Emergency, Legacy}` | **Explanatory** (Cautious; see leakage) |
| 3 | `registration_date` | Date record entered system | **Interval** | Discrete | `2015-01-01` to `2025-10-24` | **Temporal Split Key** (Metadata) |
| 4 | `first_name` | Patient given name | **Nominal** | Discrete | Common Pakistani given names | **Explanatory** |
| 5 | `family_name` | Patient family name / surname | **Nominal** | Discrete | Common Pakistani family names | **Explanatory** (Blocking key) |
| 6 | `father_name` | Father or guardian name | **Nominal** | Discrete | Common Pakistani male names | **Explanatory** (High missingness in Emergency) |
| 7 | `gender` | Biological / administrative sex | **Nominal** | Binary (in clean) | Clean: `{M, F}`; Raw: `{M, F, male, 1, ...}` | **Explanatory** |
| 8 | `date_of_birth` | Calendar birth date | **Interval** | Discrete | `1925-01-01` to `2025-10-24` | **Explanatory** (DOB diff in days) |
| 9 | `age_years` | Age in completed years | **Ratio** | Discrete | `0` to `110` (raw has -999, 199, 999) | **Explanatory** (Cross-check with DOB) |
| 10 | `national_id` | Synthetic 13-digit CNIC string | **Nominal** | Discrete | 13 digits: `PPPPP-SSSSSSS-C` | **Explanatory** (Digit string; NOT a number) |
| 11 | `phone` | Mobile contact number | **Nominal** | Discrete | 11 digits: `03xxxxxxxxx` | **Explanatory** (Digit string; NOT a number) |
| 12 | `city` | Patient residential city | **Nominal** | Discrete | 16 canonical Pakistani cities + aliases | **Explanatory** |
| 13 | `address` | Free-text residential address | **Nominal** | Discrete | Street, Sector, Area text | **Explanatory** |
| 14 | `blood_group` | ABO/Rh blood group | **Nominal** | Discrete | `{A+, A-, B+, B-, AB+, AB-, O+, O-}` | **Explanatory** |
| 15 | `insurance_tier` | Healthcare coverage level | **Ordinal** | Discrete | `None < Basic < Standard < Premium` | **Explanatory** |
| 16 | `visit_count` | Cumulative clinic visit count | **Ratio** | Discrete | `0` to `500` (raw has -999, 9999) | **Explanatory** |

---

### 2.2 External Metadata & Ground Truth Objects (`data/truth/`)

| Object / Field | File Source | Measurement Type | Scale Property | Pipeline Role | Leakage Prevention Enforcement |
|---|---|---|---|---|---|
| `true_entity_id` | `data/truth/ground_truth.csv` | **Nominal** | Discrete | **MUST-NOT-USE** | Never joined to raw register before model prediction; evaluation only. |
| `fault_type` | `data/truth/fault_log.csv` | **Nominal** | Discrete | **MUST-NOT-USE** | Internal generator diagnostic; strictly inaccessible to cleaning or matching. |
| `original_value` | `data/truth/fault_log.csv` | **Nominal / Mixed** | Discrete | **MUST-NOT-USE** | True pre-corruption state; strictly inaccessible to cleaning or matching. |
| `injected_value` | `data/truth/fault_log.csv` | **Nominal / Mixed** | Discrete | **MUST-NOT-USE** | Corruption artifact log; strictly inaccessible to cleaning or matching. |
| `is_duplicate` | Pairwise candidate dataset | **Nominal** | Asymmetric Binary | **Target** | Target label for pair classification ($y \in \{0, 1\}$); derived offline via truth. |
| `merged_into` | Application database / state | **Nominal** | Discrete | **MUST-NOT-USE** | Clerk action outcome; exists only downstream after user confirmation. |
| `cluster_size` | Theoretical entity graph | **Ratio** | Discrete | **MUST-NOT-USE** | Knowing cluster cardinality leaks duplicate frequency; prohibited. |

---

## 3. Detailed Column Specifications & Lecture 2 Rationale

### 1. `record_id`
- **Meaning:** Unique synthetic primary key assigned to each row of the register.
- **Data Type in Pandas:** `object` (string: `REC0000001` to `REC0054373`).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete.
- **Plausible Domain:** Exactly 54,373 distinct values formatted as `REC` followed by 7 decimal digits.
- **Role:** `identifier`.
- **Lecture 2 Rationale:**
  - *Does zero mean none?* No zero exists.
  - *Does doubling mean twice?* Multiplying record IDs is undefined.
  - *Is only order real?* No. Order is completely arbitrary. The `record_id` strings were explicitly **shuffled** after duplicate generation so that row position or ID magnitude carries zero information regarding entity grouping or creation sequence.
  - *Permitted operations:* Identity check only ($=$ or $\neq$). It must never be passed to a learning algorithm as a numerical feature.

---

### 2. `source_system`
- **Meaning:** Hospital department or IT software silo where the record originated:
  - `OPD`: Outpatient department (19,076 records, 35.08%).
  - `Lab`: Clinical Pathology Laboratory (11,012 records, 20.25%).
  - `Radiology`: Imaging department (8,021 records, 14.75%).
  - `Pharmacy`: In-hospital dispensing pharmacy (8,134 records, 14.96%).
  - `Emergency`: Acute trauma / emergency triage (5,468 records, 10.06%).
  - `Legacy`: Pre-2018 decommissioned paper-to-electronic register (2,662 records, 4.90%).
- **Data Type in Pandas:** `object` (string).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete (polytomous categorical, 6 classes).
- **Plausible Domain:** Exactly the set `{OPD, Lab, Radiology, Pharmacy, Emergency, Legacy}`.
- **Role:** `explanatory` (with strict leakage guardrails).
- **Lecture 2 Rationale:**
  - *Does zero mean none?* No concept of zero.
  - *Does doubling mean twice?* No arithmetic exists.
  - *Is order real?* No. OPD is neither "greater than" nor "less than" Emergency. Only identity and categorical equality exist.
  - *Leakage Consideration:* In real healthcare, duplicate patient charts frequently cross departments (e.g. an OPD record and a subsequent Emergency record for the same individual). However, if candidate pair generation is not sampled symmetrically, `source_system` differences can act as an artificial shortcut for duplicate status. See [`docs/leakage_statement.md`](file:///c:/Users/ZUBAIR/Desktop/DM%20MID%20project/docs/leakage_statement.md).

---

### 3. `registration_date`
- **Meaning:** Calendar date on which this specific encounter or patient profile was created in the system.
- **Data Type in Pandas:** `object` in raw CSV (mixed formats: `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`); standardized to ISO datetime in cleaning.
- **Measurement Scale:** **Interval**.
- **Scale Property:** Discrete (calendar day units).
- **Plausible Domain:** `2015-01-01` to `2025-10-24`.
- **Role:** `temporal_split_key` (Evaluation metadata).
- **Lecture 2 Rationale:**
  - *Does zero mean none?* No. Calendar time has an arbitrary origin (year 0 / AD 1 / Unix Epoch 1970). A date of 0 does not represent the absence of time.
  - *Does doubling mean twice?* No. Year 2020 is not "twice as old or twice as late" as year 1010.
  - *Are intervals meaningful?* **Yes.** The elapsed time between `2021-01-01` and `2021-01-11` (10 days) is identical in length to the elapsed time between `2024-05-10` and `2024-05-20` (10 days).
  - *Pipeline Role:* Used strictly to define the **temporal split** (early records $\to$ train, later records $\to$ test), ensuring our model simulates real prospective hospital deployment.

---

### 4. `first_name`
- **Meaning:** Patient given name (e.g., Muhammad, Fatima, Ali, Ayesha).
- **Data Type in Pandas:** `object` (string).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete.
- **Plausible Domain:** Pakistani first names, including corrupted forms (abbreviations `M.`, honorifics `Syed`, typos, transpositions).
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - *Does zero mean none?* No.
  - *Are names ordered?* Alphabetical collation is a lexicographic indexing convention, not a quantitative measurement.
  - *Pairwise Feature Formulation:* Transformed into continuous pairwise similarity features via Levenshtein distance, Jaro-Winkler similarity, and Soundex / Metaphone phonetics.

---

### 5. `family_name`
- **Meaning:** Patient surname, clan, caste, or tribal identifier (e.g., Khan, Ahmed, Malik, Chaudhry).
- **Data Type in Pandas:** `object` (string).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete.
- **Plausible Domain:** Common Pakistani family names, subject to typos, phonetics, and missingness.
- **Role:** `explanatory` (Primary candidate blocking key).
- **Lecture 2 Rationale:**
  - Nominal text categorical variable.
  - Used in candidate blocking: hashing soundex or exact standard token of `family_name` drastically reduces the $O(n^2)$ search space.

---

### 6. `father_name`
- **Meaning:** Father or legal guardian's name. In Pakistan, National ID registration and hospital records rely heavily on the patronymic string to disambiguate common names.
- **Data Type in Pandas:** `object` (string; 5,035 missing values = 9.26%).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete.
- **Plausible Domain:** Pakistani full male names.
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - Nominal text string.
  - Missingness is structural (not Missing Completely at Random): Emergency registrations frequently omit `father_name` because patients arrive in trauma or unaccompanied.

---

### 7. `gender`
- **Meaning:** Administrative sex assigned to the patient.
- **Data Type in Pandas:** `object` (in raw data: 14 distinct dirty tokens including `M`, `F`, `male`, `female`, `1`, `2`, `Ml`, `Fm`).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** **Binary** (in clean canonical representation) / Discrete Nominal (in raw representation).
- **Plausible Domain:** Clean domain: `{M, F}`.
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - *Binary distinction:* In clean form, gender is a **symmetric binary** attribute (neither Male nor Female is an "absence" or rare event). Symmetric matching measures (Simple Matching Coefficient) are appropriate for gender concordance.
  - *Raw state:* In raw data, collection errors and inconsistent encoding produced 14 variants, which must be mapped to `{M, F}` or flagged as invalid during cleaning (Lecture 2: Inconsistent Encoding).

---

### 8. `date_of_birth`
- **Meaning:** Calendar birth date of the patient.
- **Data Type in Pandas:** `object` in raw CSV (492 missing values = 0.90%; 392 sentinel values `1900-01-01`).
- **Measurement Scale:** **Interval**.
- **Scale Property:** Discrete (calendar day precision).
- **Plausible Domain:** `1925-01-01` to `2025-10-24` (raw contains sentinels like `1900-01-01` and future impossible dates).
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - Calendar date. Elapsed difference ($DOB_A - DOB_B$) produces an interval in days, which is a meaningful measure of age discrepancy between two candidate records.
  - Calendar zero is arbitrary.

---

### 9. `age_years`
- **Meaning:** Patient age in completed years at the time of registration.
- **Data Type in Pandas:** `int64` / numeric (raw contains corrupt sentinels `-999`, `-1`, and impossible values `199`, `999`, `1823`).
- **Measurement Scale:** **Ratio**.
- **Scale Property:** Discrete (recorded in integer years, though time is conceptually continuous).
- **Plausible Domain:** `0` to `110`.
- **Role:** `explanatory` (also used for deterministic cross-validation against `date_of_birth`).
- **Lecture 2 Rationale:**
  - *Does zero mean none?* **Yes.** Age 0 indicates a newborn infant (less than 1 full year elapsed since birth; zero complete years). Absolute zero physically exists.
  - *Does doubling mean twice?* **Yes.** A 60-year-old patient has lived exactly twice as many years as a 30-year-old patient.
  - *Permitted operations:* Addition, subtraction, multiplication, division, mean, standard deviation, geometric mean.
  - *Data Quality Check:* In Lecture 2 data audit, $|(registration\_date - DOB)/365.25 - age\_years| > 2$ reveals record corruption or transposition errors.

---

### 10. `national_id`
- **Meaning:** Synthetic 13-digit Pakistani Computerized National Identity Card (CNIC) number.
- **Data Type in Pandas:** `object` (string; 4,095 missing values = 7.53%).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete (13-character token).
- **Plausible Domain:** 13 numeric characters formatted as `PPPPP-SSSSSSS-C` (where P=Province/Area, S=Serial, C=Check digit).
- **Role:** `explanatory` (synthetic; flagged as synthetic throughout).
- **Lecture 2 Rationale:**
  - **CRITICAL LECTURE 2 DISTINCTION:** Although composed exclusively of decimal digits ($0-9$), `national_id` is **strictly Nominal**.
  - The leading digit represents administrative geography (e.g., 1=KPK, 2=FATA, 3=Punjab, 4=Sindh, 5=Balochistan, 6=Islamabad).
  - Adding, subtracting, or averaging two CNICs produces a meaningless number. Doubling a CNIC does not create "twice as much identity."
  - Treating CNIC as a continuous number would cause loss of leading zeros and impose a completely fictional spatial ordering.

---

### 11. `phone`
- **Meaning:** Primary mobile telephone contact number (Pakistani mobile format: 11 digits starting with `03`).
- **Data Type in Pandas:** `object` (string; 3,131 missing values = 5.76%; 183 sentinel entries `00000000000`).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete.
- **Plausible Domain:** 11-digit strings matching regex `^03[0-4][0-9]{8}$`.
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - Nominal digit string.
  - The leading `0` is a mandatory routing prefix. If stored as an integer, Python/Pandas drops the leading zero (`3001234567`), corrupting the phone syntax.
  - Addition and division of phone numbers are nonsensical operations.

---

### 12. `city`
- **Meaning:** Patient's primary city of residence.
- **Data Type in Pandas:** `object` (raw data contains 69 distinct strings due to case variations and 3-letter IATA-style abbreviations: `Lahore`, `lahore`, `LHR`, `Grw`, `HYD`).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete (polytomous categorical, 16 canonical targets).
- **Plausible Domain:** 16 canonical Pakistani urban centers.
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - Geographic category with no intrinsic mathematical ordering.

---

### 13. `address`
- **Meaning:** Free-text street address, house number, road, and neighborhood.
- **Data Type in Pandas:** `object` (string).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete.
- **Plausible Domain:** Text strings (e.g., `House 440, Street 33 Road, Clifton, Lahore`).
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - High-cardinality nominal text attribute.
  - Pairwise similarity is computed using token Jaccard similarity, q-gram overlap, and normalized Levenshtein distance.

---

### 14. `blood_group`
- **Meaning:** ABO and Rhesus factor blood classification.
- **Data Type in Pandas:** `object` (4,844 missing values = 8.91%).
- **Measurement Scale:** **Nominal**.
- **Scale Property:** Discrete (8 categories: `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`).
- **Plausible Domain:** Exactly the 8 serological blood groups.
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - Qualitative biological classification.
  - No hierarchy, magnitude, or numerical ordering exists among blood groups.

---

### 15. `insurance_tier`
- **Meaning:** Hospital health benefit and insurance coverage class.
- **Data Type in Pandas:** `object` in raw CSV (5,566 truly missing/empty records = 10.24%; 19,357 uninsured records with explicit string `'None'`).
- **Measurement Scale:** **Ordinal**.
- **Scale Property:** Discrete.
- **Plausible Domain:** `None < Basic < Standard < Premium`.
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - **THE CANONICAL ORDINAL ATTRIBUTE:** The categories exhibit a strict, immutable qualitative hierarchy:
    $$\text{None} < \text{Basic} < \text{Standard} < \text{Premium}$$
  - *Permitted operations:* Order comparisons ($<, \le, >, \ge$), rank correlation (Spearman's $\rho$, Kendall's $\tau$), median, percentiles.
  - *Prohibited operations:* Distance subtraction. The economic coverage difference between `None` and `Basic` cannot be assumed equal to the difference between `Standard` and `Premium`. Mean and standard deviation are mathematically impermissible unless integer scores ($0, 1, 2, 3$) are explicitly mapped as a subjective heuristic.
  - *Lecture 2 Missingness Insight:* Notice that string `'None'` is a valid ordinal level (uninsured patient), whereas the empty string `''` represents uncollected/missing data (e.g. from the Pathology Laboratory subsystem). Treating `'None'` as `NaN` would falsely inflate missingness from 10.24% to 45.84%.

---

### 16. `visit_count`
- **Meaning:** Total number of outpatient encounters or inpatient admissions recorded in this hospital profile.
- **Data Type in Pandas:** `int64` / numeric (raw data contains sentinels `-999` and extreme corruption outliers `9999`).
- **Measurement Scale:** **Ratio**.
- **Scale Property:** Discrete (non-negative integer count).
- **Plausible Domain:** `0` to `500` visits.
- **Role:** `explanatory`.
- **Lecture 2 Rationale:**
  - *Does zero mean none?* **Yes.** A patient with `visit_count = 0` has had zero prior hospital encounters (complete absence of visits).
  - *Does doubling mean twice?* **Yes.** 20 hospital visits represents exactly twice the clinical encounter volume of 10 hospital visits.
  - *Permitted operations:* All arithmetic operations, log transformations ($\log(1 + x)$ to handle high skew), mean, variance, geometric mean.

---

## 4. Ground Truth, Pairwise Targets, and Prohibited Objects

### 1. `true_entity_id`
- **File:** `data/truth/ground_truth.csv`
- **Measurement Type:** Nominal (String: `ENT0000001` to `ENT0046952`).
- **Role:** **MUST-NOT-USE** (Evaluation Only).
- **Why Prohibited:** `true_entity_id` represents the ground truth cluster assignment (the omniscient oracle). If any model, blocking rule, or cleaning function had access to `true_entity_id`, deduplication accuracy would reach 100% via trivial leakage. It is physically segregated into `data/truth/` and is strictly forbidden during candidate pair generation and model training.

### 2. Generator Fault Logs (`fault_type`, `original_value`, `injected_value`)
- **File:** `data/truth/fault_log.csv`
- **Measurement Type:** Nominal / Categorical.
- **Role:** **MUST-NOT-USE**.
- **Why Prohibited:** The fault log records the synthetic error generator's internal actions (e.g. "corrupted DOB on REC001294 via day-month swap"). Real hospital records clerks never receive an audit log stating which fields were mistyped.

### 3. Pair Target Label (`is_duplicate`)
- **Location:** Candidate pair evaluation datasets (`data/pairs/`).
- **Measurement Type:** **Nominal (Asymmetric Binary: $\{0, 1\}$)**.
- **Role:** **Target Variable**.
- **Properties:**
  - Positives ($y=1$) are extremely rare in all-pairs space ($\approx 9,754 / 1,478,184,378 \approx 0.00066\%$).
  - Even after efficient blocking, duplicates constitute a minority class ($\approx 5-15\%$).
  - Because it is an asymmetric binary target, **Accuracy is a dangerously misleading metric**. Majority class accuracy exceeds 90% simply by predicting "no match" for all pairs. Evaluation must focus on Precision, Recall, F1-Score, and ROC-AUC.

---

## 5. Summary Matrix: Traceability to Lecture 2

| Measurement Type | Count in Register | Columns | Key Permitted Operations | Prohibited Operations |
|---|---|---|---|---|
| **Nominal** | 11 | `record_id`, `source_system`, `first_name`, `family_name`, `father_name`, `gender`, `national_id`, `phone`, `city`, `address`, `blood_group` | Mode, frequency, contingency tables, string similarity | Ordering ($<, >$), arithmetic ($+, -, \times, \div$), mean |
| **Ordinal** | 1 | `insurance_tier` | Median, percentiles, ranking ($<, >$), rank correlation | Differences ($A - B$), arithmetic mean, standard deviation |
| **Interval** | 2 | `registration_date`, `date_of_birth` | Differences in days, mean of differences, standard deviation | Absolute zero, multiplication, division, ratios |
| **Ratio** | 2 | `age_years`, `visit_count` | Absolute zero, multiplication, division, mean, log scaling | None |

*Total columns profiled:* **16**. All four Stevens measurement types are rigorously demonstrated in compliance with DSC306 Lecture 2 standards.
