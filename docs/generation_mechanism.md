# Data Generation Mechanism

> **IMPORTANT HONESTY STATEMENT**
> All data described in this document is **entirely synthetic** (artificially computer-generated).
> No real patient records were used, referenced, or produced at any stage.
> The synthetic data is deliberately **easier and cleaner than real hospital data** because:
> - Names are drawn from fixed probability tables rather than free text with all the idiosyncrasies of handwritten or spoken registration.
> - Corruption operators are independent and memoryless; real clerical errors are correlated.
> - The register has no domain-specific organisational artefacts (e.g., clinic-specific abbreviations, departmental form templates).
> - Hard negatives are injected mechanically and are therefore more obvious than genuinely ambiguous real-world cases.
>
> This limitation must be stated in every evaluation report and on Screen 2 of the application.

---

## 1. Purpose

The generator (`src/generate_data.py`) produces a synthetic hospital patient register with:

- A controlled number of **duplicate records** per entity (same person registered multiple times with realistic corruptions).
- **Hard negatives** (different people who share many attributes), making pairwise classification non-trivial.
- Ground truth kept in a **separate file** that never touches the register.
- Shuffled `record_id` so that ID proximity carries no information about entity membership.

---

## 2. Configuration parameters

All rates are defined as module-level constants in `src/generate_data.py` and can be changed without touching any other file.  Changing them requires a dated entry in `DECISIONS.md`.

| Parameter | Value | Meaning |
|---|---|---|
| `TARGET_TOTAL_RECORDS` | 50,000 | Approximate register size |
| `DUPLICATE_ENTITY_RATE` | 0.115 (11.5 %) | Fraction of entities that have ≥ 2 records |
| `CLUSTER_SIZE_WEIGHTS` | [70, 22, 8] | Relative frequency of 2-, 3-, and 4-record clusters |
| `REG_DATE_START` | 2015-01-01 | Earliest possible registration date |
| `REG_DATE_END` | 2025-10-24 | Latest possible registration date |
| `RANDOM_SEED` | 42 (from `src/config.py`) | Global seed for reproducibility |

### Derived entity counts (seed 42)

- Approximate number of true entities: ~43,000–45,000 (computed to hit 50,000 total records)
- Mean cluster size for duplicated entities: ≈ 2.38

---

## 3. Entity generation

Each true entity gets a canonical (clean) record with the following fields generated independently at random:

| Field | Method |
|---|---|
| `first_name` | Weighted sample from 50 male or 50 female Pakistani given-name pool |
| `family_name` | Weighted sample from 50 Pakistani family-name pool |
| `father_name` | Random first name + same family name |
| `gender` | 50 % M / 50 % F (then corrupted in duplicates) |
| `date_of_birth` | Uniform random date giving age 0–90 at registration |
| `national_id` | Synthetic 13-digit ID: P(1) + area(4) + serial(7) + check(1) — **NOT a real NADRA number** |
| `phone` | Random 11-digit Pakistani mobile (`030x…` to `034x…`) |
| `city` | Weighted sample from 15 canonical Pakistani cities |
| `address` | Template: `House N, Street N TYPE, AREA, CITY` |
| `blood_group` | Uniform from {A+, A−, B+, B−, AB+, AB−, O+, O−} |
| `insurance_tier` | Weighted ordinal: None(40%), Basic(30%), Standard(20%), Premium(10%) |
| `visit_count` | Uniform integer 0–50 (outliers injected separately) |

---

## 4. Duplicate record creation

For each entity with cluster_size ≥ 2, a first (canonical) record is written cleanly.
Each additional record (the "duplicate copy") has the following corruption operators applied **independently** with the probabilities below.

### 4.1 Name corruption operators

| Operator | Probability | Fault category (Lecture 2) |
|---|---|---|
| Spelling variant (Muhammad → Mohammad) | 0.30 | Inconsistent encoding |
| Character-level typo (insert / delete / substitute / transpose) | 0.35 | Measurement / entry error |
| Initial only (first name → `F.`) | 0.10 | Collection error |
| Honorific prefix (add Mr., Haji, Dr., etc.) | 0.08 | Noise |
| First ↔ family name swap | 0.06 | Noise |
| Case / whitespace noise (ALL CAPS, trailing space) | 0.20 | Noise |
| Family name typo | 0.245 (= 0.35 × 0.70) | Measurement error |

### 4.2 Date of birth corruption operators

| Operator | Probability | Fault category |
|---|---|---|
| Replace with sentinel 1900-01-01 | 0.05 | Sentinel value |
| Set to null (missing) | 0.07 | Missing value |
| Day–month swap (only when day ≤ 12) | 0.15 | Invalid value |
| Digit transposition in year | 0.10 | Measurement error |
| Non-ISO date format (dd/mm/yyyy, mm-dd-yyyy) | 0.30 | Inconsistent encoding |

Legacy system **always** uses `dd/mm/yyyy` (collection error by design).

### 4.3 Phone corruption operators

| Operator | Probability | Fault category |
|---|---|---|
| Set to null | 0.12 | Missing value |
| Replace with sentinel 00000000000 | 0.03 | Sentinel value |
| One digit error | 0.15 | Measurement error |
| Format variant (+92, 0092, dashes, spaces) | 0.25 | Inconsistent encoding |

### 4.4 National ID corruption operators

| Operator | Probability | Fault category |
|---|---|---|
| Set to null | 0.20 | Missing value |
| One digit wrong | 0.08 | Measurement error |

### 4.5 Other field operators

| Field | Operator | Probability | Fault category |
|---|---|---|---|
| city | Replace canonical with alias (LHR, lahore) | 0.30 | Inconsistent encoding |
| gender | Replace M/F with dirty variant (Male, 1, male) | 0.20 | Inconsistent encoding |
| address | Abbreviate common words (Street→St, Road→Rd) | 0.25 | Noise |
| father_name | Set to null | 0.18 | Missing value |
| age_years | Replace with impossible value (199, 999, -1) | 0.04 | Invalid value / outlier |
| visit_count | Replace with sentinel (9999, -1, -999) | 0.04 | Sentinel value |

### 4.6 Source-system structural gaps (collection errors)

These gaps are **not random**; they model how each legacy system was designed.
This is the Lecture 2 concept of a collection error — the system was never designed to collect that field.

| Source system | Structural gap |
|---|---|
| Legacy | `national_id` always null; dates always `dd/mm/yyyy` |
| Emergency | `father_name` null ~70 % of the time; `phone` null ~40 % |
| Lab | `insurance_tier` null ~50 % of the time |
| Pharmacy | `blood_group` null ~60 % of the time |

### 4.7 Singleton noise

Singleton entities (cluster_size = 1) receive **one light noise operator** with probability 0.05.
This models normal data quality issues unrelated to duplication.

---

## 5. Hard negatives

Hard negatives are different people who look alike; they make the pairwise classifier non-trivial.

| Type | Injected count | Shared fields |
|---|---|---|
| Sibling pairs | ~720 pairs | family_name, address, phone; DOB within 1–5 years |
| Twin pairs | ~80 pairs | family_name, address, phone, same DOB |
| Father–son (same full name) | 300 pairs | first_name, family_name; DOB 20–35 years apart |
| Shared household | 500 pairs | family_name, phone, address; different DOBs and names |
| Very common names (Muhammad Ahmed / Fatima Khan) | 600 extra entities | first_name + family_name shared with many others |

---

## 6. record_id shuffling

1. Every record is first assigned a temporary placeholder ID (`TMP0000001`…).
2. A new random permutation of `REC0000001`…`REC{N}` is generated.
3. Placeholder IDs are replaced with the shuffled IDs.
4. The ground truth file is updated to match.

**Test:** `tests/test_generate_data.py::TestRecordIdShuffle::test_record_id_not_correlated_with_entity` verifies that the fraction of adjacent (by record_id sort) same-entity pairs is < 5 %.

---

## 7. Output files

| File | Purpose | Leakage risk |
|---|---|---|
| `data/raw/register_raw.csv` | The register the clerk sees | None (ground truth absent) |
| `data/truth/ground_truth.csv` | `record_id → true_entity_id` | **Never loaded by the UI**; training and evaluation only |
| `data/truth/fault_log.csv` | What was injected | Used only to measure audit recall; never a feature |
| `data/raw/demo_register.csv` | ~5,000-record demo for the app | Same structure as register |
| `data/raw/sample_batch.csv` | ~2,000-record sample for batch upload testing | Same structure |

---

## 8. Sanity checks

The generator reports these metrics in `reports/metrics.json["generation"]`:

- Total records, total entities, actual duplicate rate.
- Cluster-size distribution.
- Hard-negative counts.
- Per-column missing rates.
- **Trivial rule recall:** fraction of true duplicate pairs found by "same DOB AND same phone".
  - If this exceeds ~70 %, corruption is too easy and must be made harder.
  - If this is near 0 %, blocking by phone will miss almost all true pairs.

---

## 9. Reproducibility

Run `python -m src.generate_data --seed 42` to reproduce exactly.
Different seeds produce statistically similar but not identical registers.
A robustness check (different seed) can be used to verify that the pipeline is not over-fitted to seed 42.
