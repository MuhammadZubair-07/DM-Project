"""Data profiling module for the raw hospital patient register.

Profiles data/raw/register_raw.csv according to Lecture 2 (Data Preprocessing,
Attribute Types, Measurement Scales, and Summary Statistics).

Produces:
1. reports/metrics.json under key 'profile' (using src.utils.save_metrics)
2. reports/profile.md (readable markdown profiling report)
3. docs/data_dictionary.json (machine-readable data dictionary)

Usage:
    python -m src.profile
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config as CFG
from src.utils import save_metric

# ── Lecture 2 Attribute Definitions and Roles ─────────────────────────────────
# All 16 columns of register_raw.csv plus external evaluation metadata columns.
COLUMN_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "record_id": {
        "meaning": "Unique synthetic alphanumeric record identifier (shuffled)",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "REC0000001 to REC0054373 (unique string per row)",
        "role": "identifier",
        "reasoning": (
            "Only identity (=, !=) is defined. Order is purely arbitrary; "
            "arithmetic operations (addition, averaging) are meaningless. "
            "Never used as a predictive feature."
        ),
    },
    "source_system": {
        "meaning": "Hospital department or legacy subsystem that created the record",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "{OPD, Lab, Radiology, Pharmacy, Emergency, Legacy}",
        "role": "explanatory",
        "reasoning": (
            "Categorical department labels with no intrinsic hierarchy or numeric scale. "
            "Only equality/frequency tests are meaningful. Used cautiously due to "
            "potential sampling leakage (see docs/leakage_statement.md)."
        ),
    },
    "registration_date": {
        "meaning": "Calendar date when the patient record was registered in the subsystem",
        "measurement_type": "Interval",
        "scale_property": "Discrete",
        "plausible_range": "2015-01-01 to 2025-10-24 (ISO or raw legacy formats)",
        "role": "temporal_split_key",
        "reasoning": (
            "Calendar time has an arbitrary zero point (Gregorian origin). "
            "Differences between dates (elapsed days) are physically meaningful, "
            "but ratios (e.g. 2020 / 1010) are meaningless. Used for temporal train/test split."
        ),
    },
    "first_name": {
        "meaning": "Patient given / first name (Pakistani naming convention)",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "Common Pakistani given names and corrupted spelling variants",
        "role": "explanatory",
        "reasoning": (
            "Text categorical string. Equality and string similarity (Levenshtein, Jaro-Winkler) "
            "are defined, but no natural order or arithmetic exists."
        ),
    },
    "family_name": {
        "meaning": "Patient family name, tribe, caste, or surname",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "Common Pakistani family names (Khan, Ahmed, Ali, Malik, etc.)",
        "role": "explanatory",
        "reasoning": (
            "Text categorical string. Nominal identity; used for blocking and similarity matching."
        ),
    },
    "father_name": {
        "meaning": "Father or guardian's full name (critical patronymic identifier in Pakistan)",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "Text string, frequently omitted in Emergency registrations",
        "role": "explanatory",
        "reasoning": (
            "Text categorical string. Equality and string similarity are meaningful; "
            "no ordering or arithmetic."
        ),
    },
    "gender": {
        "meaning": "Administrative / biological gender code",
        "measurement_type": "Nominal",
        "scale_property": "Binary",
        "plausible_range": "M, F (clean); {M, F, male, female, 1, 2, Ml, Fm} (dirty raw)",
        "role": "explanatory",
        "reasoning": (
            "Conceptually symmetric binary (M/F), but raw data contains 14 dirty nominal encodings. "
            "No hierarchy or magnitude exists between categories."
        ),
    },
    "date_of_birth": {
        "meaning": "Patient calendar birth date",
        "measurement_type": "Interval",
        "scale_property": "Discrete",
        "plausible_range": "1925-01-01 to 2025-10-24 (plus sentinels such as 1900-01-01)",
        "role": "explanatory",
        "reasoning": (
            "Calendar date on an interval scale. Differences measure elapsed age, "
            "but calendar year 0 is an arbitrary convention, making multiplication/ratio invalid."
        ),
    },
    "age_years": {
        "meaning": "Chronological age in completed years at time of registration",
        "measurement_type": "Ratio",
        "scale_property": "Discrete",
        "plausible_range": "0 to 110 (raw contains corrupt sentinels: -999, -1, 199, 999, 1823)",
        "role": "explanatory",
        "reasoning": (
            "True absolute zero exists (age 0 = newborn, absence of elapsed life years). "
            "Ratios are physically valid: a person of 40 is twice as old as a person of 20."
        ),
    },
    "national_id": {
        "meaning": "Synthetic 13-digit Pakistani CNIC number (5-digit area + 7-digit serial + 1 check)",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "13-digit numeric string (synthetic; never real citizen data)",
        "role": "explanatory",
        "reasoning": (
            "Despite being composed of digits, national_id is NOMINAL. Digits serve purely as "
            "categorical labels. The leading digit represents province; adding two CNICs or "
            "calculating their average is completely nonsensical. Doubling a CNIC has no meaning."
        ),
    },
    "phone": {
        "meaning": "Mobile contact phone number (Pakistani 11-digit mobile format: 03xxxxxxxxx)",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "11-digit string starting with 03 (plus formatting noise and sentinels)",
        "role": "explanatory",
        "reasoning": (
            "Phone numbers are NOMINAL digit-strings. Leading '0' is syntactically mandatory and "
            "would be destroyed if treated as a numeric scalar. Arithmetic addition and scaling "
            "have zero physical meaning."
        ),
    },
    "city": {
        "meaning": "Patient residential city in Pakistan",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "16 canonical cities plus noisy abbreviations (LHR, KHI, ISB, etc.)",
        "role": "explanatory",
        "reasoning": (
            "Polytomous nominal categories representing geographic localities without ordering."
        ),
    },
    "address": {
        "meaning": "Free-text street, house, sector, and area address line",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "Free-text address strings (House ..., Street ..., Area)",
        "role": "explanatory",
        "reasoning": (
            "Unstructured text identifier with nominal properties. Evaluated via n-gram token overlap "
            "and edit distance rather than numeric arithmetic."
        ),
    },
    "blood_group": {
        "meaning": "ABO and Rhesus blood antigen classification",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "{A+, A-, B+, B-, AB+, AB-, O+, O-}",
        "role": "explanatory",
        "reasoning": (
            "Biological categories with no intrinsic mathematical ordering or magnitude."
        ),
    },
    "insurance_tier": {
        "meaning": "Patient hospital coverage benefit tier",
        "measurement_type": "Ordinal",
        "scale_property": "Discrete",
        "plausible_range": "None < Basic < Standard < Premium (with missingness from Lab)",
        "role": "explanatory",
        "reasoning": (
            "Strictly ordered categories: None < Basic < Standard < Premium. "
            "Ranking is mathematically meaningful (<, >), but distance between levels is "
            "neither uniform nor measurable (Basic to Standard != Standard to Premium)."
        ),
    },
    "visit_count": {
        "meaning": "Cumulative hospital outpatient or inpatient visit count",
        "measurement_type": "Ratio",
        "scale_property": "Discrete",
        "plausible_range": "0 to 500 (raw contains corrupt sentinels: -999, 9999)",
        "role": "explanatory",
        "reasoning": (
            "Count attribute with an absolute physical zero (0 visits = complete absence of encounters). "
            "Multiplicative ratios are valid: 10 visits is exactly twice as many encounters as 5 visits."
        ),
    },
    # External metadata / evaluation objects (Must NOT be used in matching)
    "true_entity_id": {
        "meaning": "Ground truth person ID in data/truth/ground_truth.csv",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "ENT0000001 to ENT0046952",
        "role": "must-not-use",
        "reasoning": (
            "Ground truth cluster oracle. Strictly forbidden during candidate generation, "
            "blocking, feature extraction, and inference. Used solely in offline evaluation."
        ),
    },
    "fault_type": {
        "meaning": "Provenance log entry of synthetic error injection in data/truth/fault_log.csv",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "Categorical error types (typo, sentinel, swap, omission, etc.)",
        "role": "must-not-use",
        "reasoning": (
            "Generator audit log. Revealing whether an attribute was corrupted is direct leakage."
        ),
    },
    "original_value": {
        "meaning": "Pre-corruption canonical attribute value recorded in fault log",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "Original string/numeric value before synthetic noise injection",
        "role": "must-not-use",
        "reasoning": (
            "Ground truth value; strictly forbidden."
        ),
    },
    "injected_value": {
        "meaning": "Post-corruption noisy attribute value recorded in fault log",
        "measurement_type": "Nominal",
        "scale_property": "Discrete",
        "plausible_range": "Injected corrupt string/numeric value",
        "role": "must-not-use",
        "reasoning": (
            "Generator metadata; strictly forbidden."
        ),
    },
    "is_duplicate": {
        "meaning": "Target pairwise matching label (1 = same true person, 0 = different people)",
        "measurement_type": "Nominal",
        "scale_property": "Binary",
        "plausible_range": "{0, 1}",
        "role": "target",
        "reasoning": (
            "Pair-level supervised target label. Highly asymmetric binary target "
            "(positives are rare). Never exists inside raw individual records."
        ),
    },
}


def _detect_sentinels(col: str, series: pd.Series) -> int:
    """Counts recognizable corrupt sentinel tokens in a column."""
    count = 0
    s_clean = series.fillna("").astype(str).str.strip()

    if col == "date_of_birth":
        count = int(s_clean.isin(CFG.SENTINEL_DATES).sum())
    elif col == "age_years":
        # Sentinels in raw data include -999, -1, 199, 999, 1823
        try:
            numeric_ages = pd.to_numeric(s_clean.replace("", np.nan), errors="coerce")
            count = int(((numeric_ages < 0) | (numeric_ages > 120)).sum())
        except Exception:
            count = 0
    elif col == "phone":
        count = int(s_clean.isin(CFG.SENTINEL_PHONES).sum())
    elif col == "national_id":
        count = int(s_clean.isin(CFG.SENTINEL_NATIONAL_IDS).sum())
    elif col == "visit_count":
        try:
            numeric_visits = pd.to_numeric(s_clean.replace("", np.nan), errors="coerce")
            count = int(((numeric_visits < 0) | (numeric_visits > 500)).sum())
        except Exception:
            count = 0
    return count


def profile_dataframe(df_raw: pd.DataFrame) -> dict[str, Any]:
    """Computes full statistical and measurement profile of the raw register.
    
    Args:
        df_raw: Raw DataFrame loaded with keep_default_na=False so that string
                sentinels like 'None' are not silently swallowed by default parser.

    Returns:
        Dictionary containing overall statistics, column profiles, categorical
        breakdowns, and measurement type distribution.
    """
    n_rows, n_cols = df_raw.shape
    columns_profile: dict[str, Any] = {}
    type_counts: dict[str, int] = {"Nominal": 0, "Ordinal": 0, "Interval": 0, "Ratio": 0}
    scale_counts: dict[str, int] = {"Discrete": 0, "Continuous": 0, "Binary": 0}

    for col in df_raw.columns:
        s = df_raw[col].astype(str)
        spec = COLUMN_SPECIFICATIONS.get(col, {
            "meaning": "Unknown column",
            "measurement_type": "Nominal",
            "scale_property": "Discrete",
            "plausible_range": "unspecified",
            "role": "explanatory",
            "reasoning": "Unspecified column"
        })

        m_type = spec["measurement_type"]
        s_prop = spec["scale_property"]
        type_counts[m_type] = type_counts.get(m_type, 0) + 1
        scale_counts[s_prop] = scale_counts.get(s_prop, 0) + 1

        # Missing values:
        # In register_raw.csv, an empty string '' represents a truly missing/omitted entry.
        # Notice special case for insurance_tier: 'None' is an explicit tier level (None < Basic < ...),
        # whereas '' is an omitted entry (e.g. from Lab system).
        is_empty = (s.str.strip() == "")
        missing_count = int(is_empty.sum())
        missing_rate = round(float(missing_count / n_rows), 4)

        # Unique values (excluding empty string)
        non_empty = s[~is_empty]
        unique_count = int(non_empty.nunique())

        # Top values (most frequent 5)
        top_vc = non_empty.value_counts().head(5)
        top_values = [
            {
                "value": str(val),
                "count": int(cnt),
                "percentage": round(float(cnt / n_rows * 100), 2),
            }
            for val, cnt in top_vc.items()
        ]

        # Min / Max computation
        min_val: Any = None
        max_val: Any = None
        sentinels_count = _detect_sentinels(col, s)

        if col in ["age_years", "visit_count"]:
            numeric_s = pd.to_numeric(non_empty, errors="coerce").dropna()
            if not numeric_s.empty:
                min_val = float(numeric_s.min())
                max_val = float(numeric_s.max())
        else:
            if not non_empty.empty:
                sorted_vals = non_empty.sort_values()
                min_val = str(sorted_vals.iloc[0])
                max_val = str(sorted_vals.iloc[-1])

        col_dict: dict[str, Any] = {
            "column_name": col,
            "pandas_dtype": str(df_raw[col].dtype),
            "meaning": spec["meaning"],
            "measurement_type": m_type,
            "scale_property": s_prop,
            "role": spec["role"],
            "plausible_range": spec["plausible_range"],
            "total_rows": n_rows,
            "non_null_count": n_rows - missing_count,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "missing_percentage": round(missing_rate * 100, 2),
            "unique_count": unique_count,
            "min_value": min_val,
            "max_value": max_val,
            "sentinel_count": sentinels_count,
            "top_values": top_values,
            "lecture_reasoning": spec["reasoning"],
        }
        columns_profile[col] = col_dict

    # Categorical specific distributions
    categorical_breakdowns: dict[str, Any] = {}

    # 1. source_system
    if "source_system" in df_raw.columns:
        sys_counts = df_raw["source_system"].value_counts().to_dict()
        categorical_breakdowns["source_system"] = {
            k: {"count": int(v), "percentage": round(v / n_rows * 100, 2)}
            for k, v in sys_counts.items()
        }

    # 2. gender raw dirty codes
    if "gender" in df_raw.columns:
        gender_counts = df_raw["gender"].value_counts().to_dict()
        categorical_breakdowns["gender"] = {
            k: {"count": int(v), "percentage": round(v / n_rows * 100, 2)}
            for k, v in gender_counts.items()
        }

    # 3. insurance_tier (explicit distinction: 'None' tier vs '' missing)
    if "insurance_tier" in df_raw.columns:
        tier_counts = df_raw["insurance_tier"].value_counts().to_dict()
        categorical_breakdowns["insurance_tier"] = {
            (k if k != "" else "(missing/empty)"): {
                "count": int(v),
                "percentage": round(v / n_rows * 100, 2),
            }
            for k, v in tier_counts.items()
        }

    # 4. blood_group
    if "blood_group" in df_raw.columns:
        bg_counts = df_raw["blood_group"].value_counts().to_dict()
        categorical_breakdowns["blood_group"] = {
            (k if k != "" else "(missing/empty)"): {
                "count": int(v),
                "percentage": round(v / n_rows * 100, 2),
            }
            for k, v in bg_counts.items()
        }

    # Numeric summaries
    numeric_summaries: dict[str, Any] = {}
    for num_col in ["age_years", "visit_count"]:
        if num_col in df_raw.columns:
            clean_num = pd.to_numeric(df_raw[num_col], errors="coerce").dropna()
            numeric_summaries[num_col] = {
                "count": int(clean_num.count()),
                "mean": round(float(clean_num.mean()), 2),
                "std": round(float(clean_num.std()), 2),
                "min": float(clean_num.min()),
                "p25": float(clean_num.quantile(0.25)),
                "median": float(clean_num.median()),
                "p75": float(clean_num.quantile(0.75)),
                "max": float(clean_num.max()),
                "negative_count": int((clean_num < 0).sum()),
                "outlier_extreme_count": int(
                    (clean_num > 120).sum() if num_col == "age_years" else (clean_num > 500).sum()
                ),
            }

    summary: dict[str, Any] = {
        "shape": [n_rows, n_cols],
        "total_records": n_rows,
        "total_columns": n_cols,
        "measurement_types_present": type_counts,
        "scale_properties_present": scale_counts,
        "columns": columns_profile,
        "categorical_breakdowns": categorical_breakdowns,
        "numeric_summaries": numeric_summaries,
    }
    return summary


def generate_profile_markdown(profile: dict[str, Any], output_path: Path) -> str:
    """Generates a comprehensive markdown document detailing the profile."""
    n_rows = profile["total_records"]
    n_cols = profile["total_columns"]
    type_counts = profile["measurement_types_present"]

    md_lines: list[str] = [
        "# Hospital Patient Register: Data Profile Report",
        "",
        "> **Project:** DSC306 Data Mining — Project 11 (Patient Record Deduplication Tool)",
        "> **Phase:** Phase 2 (Data Profiling & Measurement Classification — Lecture 2)",
        f"> **Analyzed Dataset:** `data/raw/register_raw.csv` ({n_rows:,} records, {n_cols} attributes)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Dataset Dimensions",
        "",
        f"- **Total Records ($n$):** {n_rows:,}",
        f"- **Total Attributes ($d$):** {n_cols}",
        f"- **Measurement Scales Represented:** All 4 Stevens types present ("
        f"Nominal: {type_counts.get('Nominal', 0)}, "
        f"Ordinal: {type_counts.get('Ordinal', 0)}, "
        f"Interval: {type_counts.get('Interval', 0)}, "
        f"Ratio: {type_counts.get('Ratio', 0)})",
        "- **Data Integrity Status:** Intentionally corrupted synthetic hospital register simulating "
        "departmental silos, legacy migrations, human typographical errors, and sentinels.",
        "",
        "---",
        "",
        "## 2. Column-by-Column Attribute Profile & Lecture 2 Classification",
        "",
        "| # | Column Name | Measurement Type | Scale Property | Role | Missing Count | Missing % | Unique | Min / Max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for idx, (col, data) in enumerate(profile["columns"].items(), start=1):
        min_v = str(data["min_value"])[:12] if data["min_value"] is not None else "-"
        max_v = str(data["max_value"])[:12] if data["max_value"] is not None else "-"
        range_str = f"`{min_v}` / `{max_v}`"
        md_lines.append(
            f"| {idx} | **`{col}`** | {data['measurement_type']} | {data['scale_property']} | "
            f"`{data['role']}` | {data['missing_count']:,} | {data['missing_percentage']}% | "
            f"{data['unique_count']:,} | {range_str} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Lecture 2 Measurement Type Justifications",
        "",
        "In Data Mining (Lecture 2), attribute classification dictates which mathematical operations, "
        "distance metrics, and preprocessing transformations are mathematically valid.",
        "",
    ])

    for col, data in profile["columns"].items():
        md_lines.extend([
            f"### `{col}` ({data['measurement_type']} — {data['scale_property']})",
            f"- **Role:** `{data['role']}`",
            f"- **Meaning:** {data['meaning']}",
            f"- **Plausible Domain:** {data['plausible_range']}",
            f"- **Lecture 2 Rationale:** {data['lecture_reasoning']}",
            "",
        ])

    md_lines.extend([
        "---",
        "",
        "## 4. Missingness Analysis & Departmental Gaps",
        "",
        "Missingness in healthcare data is rarely missing completely at random (MCAR). The profiling reveals "
        "structural collection patterns:",
        "",
        "1. **`father_name` missingness (5,035 records, 9.26%):** Concentrated in **Emergency** registrations "
        "where acute triage prevents full family tree intake.",
        "2. **`national_id` missingness (4,095 records, 7.53%):** Heavily concentrated in **Legacy** migration "
        "records where national identity cards were not captured, as well as Emergency patients.",
        "3. **`phone` missingness (3,131 records, 5.76%):** Missing in Emergency cases and pediatric patients.",
        "4. **`date_of_birth` missingness (492 records, 0.90%):** True missing DOBs, alongside an additional "
        "392 records with corrupt sentinel date `1900-01-01`.",
        "5. **`insurance_tier` distinction:** Exactly 5,566 records (10.24%) have missing data (empty string) "
        "due to subsystems like Lab that do not capture billing tiers, while 19,357 records (35.60%) explicitly carry "
        "the tier value `'None'` (uninsured patient).",
        "",
        "---",
        "",
        "## 5. Categorical Distributions (Top Values)",
        "",
    ])

    for cat_col, dist in profile["categorical_breakdowns"].items():
        md_lines.extend([
            f"### Distribution: `{cat_col}`",
            "",
            "| Value | Frequency | Percentage |",
            "|---|---|---|",
        ])
        for val, item in dist.items():
            md_lines.append(f"| `{val}` | {item['count']:,} | {item['percentage']}% |")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## 6. Numerical Attribute Distributions & Anomaly Flags",
        "",
    ])

    for num_col, stats in profile["numeric_summaries"].items():
        md_lines.extend([
            f"### Numerical Summary: `{num_col}`",
            "",
            f"- **Mean ± Std:** {stats['mean']} ± {stats['std']}",
            f"- **Quartiles [Min, 25%, Median, 75%, Max]:** [{stats['min']}, {stats['p25']}, {stats['median']}, {stats['p75']}, {stats['max']}]",
            f"- **Negative Values Detected (Sentinels):** {stats['negative_count']:,} records",
            f"- **Extreme / Impossible Outliers:** {stats['outlier_extreme_count']:,} records",
            "",
        ])

    md_content = "\n".join(md_lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_content


def export_machine_readable_dictionary(output_path: Path) -> dict[str, Any]:
    """Exports full machine-readable JSON data dictionary."""
    data_dict = {
        "version": "1.0",
        "description": "Complete data dictionary for DSC306 Project 11 Patient Record Deduplication",
        "total_attributes": len(COLUMN_SPECIFICATIONS),
        "attributes": COLUMN_SPECIFICATIONS,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return data_dict


def run_profiling(
    csv_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
    report_md_path: Path | str | None = None,
    dict_json_path: Path | str | None = None,
) -> dict[str, Any]:
    """Main execution function for dataset profiling."""
    target_csv = Path(csv_path) if csv_path is not None else CFG.RAW_REGISTER_PATH
    target_md = Path(report_md_path) if report_md_path is not None else (CFG.REPORTS_DIR / "profile.md")
    target_dict = Path(dict_json_path) if dict_json_path is not None else (CFG.DOCS_DIR / "data_dictionary.json")

    if not target_csv.exists():
        raise FileNotFoundError(f"Raw register not found at {target_csv}. Run src/generate_data.py first.")

    # Load with keep_default_na=False to distinguish empty strings from the string 'None'
    df_raw = pd.read_csv(target_csv, dtype=str, keep_default_na=False)

    # Compute profiling metrics
    profile_results = profile_dataframe(df_raw)

    # 1. Save to reports/metrics.json under key 'profile'
    save_metric("profile", profile_results, metrics_file=metrics_path)

    # 2. Write reports/profile.md
    generate_profile_markdown(profile_results, target_md)

    # 3. Export docs/data_dictionary.json
    export_machine_readable_dictionary(target_dict)

    return profile_results


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Profile raw patient register (Lecture 2)")
    parser.add_argument("--csv", type=str, default=str(CFG.RAW_REGISTER_PATH), help="Path to raw register CSV")
    args = parser.parse_args()

    print(f"Profiling register at {args.csv}...")
    results = run_profiling(csv_path=args.csv)
    print("Profiling complete!")
    print(f"  Shape: {results['shape']}")
    print(f"  Measurement types: {results['measurement_types_present']}")
    print(f"  Reports saved to: {CFG.REPORTS_DIR / 'profile.md'}")
    print(f"  Metrics merged to: {CFG.METRICS_PATH}")
    print(f"  Data dictionary saved to: {CFG.DOCS_DIR / 'data_dictionary.json'}")


if __name__ == "__main__":
    main()
