# Data Cleaning and Provenance Report

> **Course:** DSC306 Data Mining — COMSATS University Islamabad
> **Project:** Project 11 — Patient Record Deduplication Tool
> **Topic:** Data Cleaning, Standardization, and Provenance Tracking (Lecture 2 & Roadmap Section 7.4)
> **Processed Register:** `data/clean/register_clean.csv` (54,373 records)

---

## 1. Executive Summary & Cleaning Philosophy

In accordance with Lecture 2 (*Data Quality and Preprocessing*), raw hospital demographic registers require systematic harmonization before pairwise similarity comparison. However, deduplication imposes two strict constraints:

1. **Never Correct Name Typos to 'True' Names:** The model's explicit job is finding similar patients despite typos. Correcting names using an omniscient dictionary would destroy the realistic variability needed to train and evaluate classifiers.
2. **Never Remove Near-Duplicates:** Near-duplicate records are our primary mining target. Collapsing them in preprocessing would destroy the evaluation dataset.
3. **Transparent Audit Trail:** Every cleaned field preserves its original uncleaned value in a `*_raw` column and documents the exact cleaning step in a corresponding `_prov` (provenance) column.

---

## 2. Before and After Cleaning Comparison Table

| Repair Item | Description | Before Count | After Count | Net Improvement |
|---|---|---|---|---|
| **Distinct city values** | Total unique city strings in column | 69 | 15 | **-54** |
| **Non-canonical city records** | Records with city aliases or noisy abbreviations | 2,324 | 0 | **-2,324** |
| **Distinct gender values** | Total unique gender strings in column | 14 | 2 | **-12** |
| **Dirty gender encodings** | Records with non-canonical gender tokens (male, 1, Ml, etc.) | 1,410 | 0 | **-1,410** |
| **Non-ISO birth date formats** | DOB strings not formatted as standard YYYY-MM-DD | 17,932 | 0 | **-17,932** |
| **DOB sentinel values (1900-01-01)** | Default legacy sentinel birth dates | 394 | 2 | **-394** |
| **Chronologically invalid DOB (> reg_date)** | Birth date occurring after registration date | 3,446 | 0 | **-3,446** |
| **Phone sentinel values (00000000000)** | Dummy placeholder phone numbers | 183 | 0 | **-183** |
| **Non-standard phone format** | Phone numbers not formatted as 11-digit 03xxxxxxxxx | 1,642 | 0 | **-1,642** |
| **Age sentinel values (-999, -1, 999)** | Sentinel values representing unrecorded age | 300 | 0 | **-300** |
| **Impossible age values (<0 or >110)** | Ages outside biological human lifespan | 481 | 0 | **-481** |
| **Age vs DOB inconsistency (> 2 years)** | Discrepancy between age_years and registration_date - DOB | 2,352 | 0 | **-2,352** |
| **Name whitespace & casing irregularities** | Untrimmed spaces, multiple spaces, or non-Titlecase in names | 2,202 | 0 | **-2,202** |
| **Extreme visit count outliers (9999)** | Visit counts far above plausible maximum (capped at 100) | 97 | 0 | **-97** |
| **Visit count sentinel values (-999, -1)** | Negative placeholder encounter numbers | 218 | 0 | **-218** |

---

## 3. Provenance Audit Distributions

Every single row in `register_clean.csv` contains valid provenance values tracking how attributes were repaired:

### Provenance: `first_name_prov` (first_name)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 52,171 | 95.95% |
| `case_fixed` | 1,699 | 3.12% |
| `whitespace_fixed` | 503 | 0.93% |

### Provenance: `family_name_prov` (family_name)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 54,360 | 99.98% |
| `case_fixed` | 13 | 0.02% |

### Provenance: `father_name_prov` (father_name)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 54,373 | 100.0% |

### Provenance: `gender_prov` (gender)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 52,963 | 97.41% |
| `code_mapped` | 1,410 | 2.59% |

### Provenance: `city_prov` (city)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 52,049 | 95.73% |
| `alias_mapped` | 1,434 | 2.64% |
| `case_fixed` | 890 | 1.64% |

### Provenance: `date_of_birth_prov` (date_of_birth)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 33,569 | 61.74% |
| `format_standardized` | 16,787 | 30.87% |
| `set_missing` | 3,597 | 6.62% |
| `sentinel_cleared` | 394 | 0.72% |
| `day_month_swapped` | 26 | 0.05% |

### Provenance: `age_years_prov` (age_years)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 53,892 | 99.12% |
| `imputed_from_dob` | 239 | 0.44% |
| `set_missing` | 181 | 0.33% |
| `sentinel_cleared` | 61 | 0.11% |

### Provenance: `phone_prov` (phone)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 52,548 | 96.64% |
| `reformatted` | 1,231 | 2.26% |
| `set_missing` | 411 | 0.76% |
| `sentinel_cleared` | 183 | 0.34% |

### Provenance: `national_id_prov` (national_id)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 54,373 | 100.0% |

### Provenance: `visit_count_prov` (visit_count)

| Provenance Action | Record Count | Percentage |
|---|---|---|
| `original` | 54,058 | 99.42% |
| `sentinel_cleared` | 218 | 0.4% |
| `capped_at_max` | 97 | 0.18% |

---

## 4. Anomaly Indicator Flags

To preserve information about data quality without corrupting numerical distance metrics, cleared sentinels and capped outliers are flagged with boolean indicators:

- `dob_sentinel_flag`: **394 records** had legacy birth dates cleared (`1900-01-01`).
- `age_sentinel_flag`: **300 records** had placeholder ages cleared or imputed from DOB.
- `phone_sentinel_flag`: **183 records** had dummy phone numbers cleared (`00000000000`).
- `nid_sentinel_flag`: **0 records** had placeholder national IDs cleared.
- `visit_outlier_flag`: **97 records** had extreme encounter counts capped at 100.
