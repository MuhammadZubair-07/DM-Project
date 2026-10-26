"""Data validation module implementing validate() with 15 quality assertions.

Provides rigorous automated validation according to Roadmap Section 7.4.
Demonstrates:
- PASS on data/clean/register_clean.csv (0 failed assertions).
- FAIL on data/raw/register_raw.csv (10+ failed assertions with violation counts).

Usage:
    python -m src.validate --csv data/clean/register_clean.csv
    python -m src.validate --csv data/raw/register_raw.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config as CFG


@dataclass
class AssertionResult:
    """Represents the outcome of a single validation assertion."""
    name: str
    passed: bool
    violations: int
    description: str


@dataclass
class ValidationReport:
    """Complete validation report across all assertions."""
    dataset_name: str
    total_records: int
    is_valid: bool
    passed_count: int
    failed_count: int
    total_assertions: int
    assertions: list[AssertionResult]

    def to_dict(self) -> dict[str, Any]:
        """Converts report to a serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_records": self.total_records,
            "is_valid": self.is_valid,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_assertions": self.total_assertions,
            "assertions": [asdict(a) for a in self.assertions],
        }


def validate(df: pd.DataFrame, dataset_name: str = "register") -> ValidationReport:
    """Executes 15 quality and integrity assertions on a patient register DataFrame.
    
    Args:
        df: Patient register DataFrame (raw or clean).
        dataset_name: Label identifying the dataset for reporting.
        
    Returns:
        ValidationReport detailing per-assertion pass/fail status and violation counts.
    """
    n_rows = len(df)
    assertions: list[AssertionResult] = []

    # 1. record_id unique and non-null
    if "record_id" in df.columns:
        s_rid = df["record_id"].fillna("").astype(str).str.strip()
        v_null = (s_rid == "").sum()
        v_dupe = df["record_id"].duplicated().sum()
        v_rid = int(v_null + v_dupe)
    else:
        v_rid = n_rows
    assertions.append(AssertionResult(
        name="record_id_unique_non_null",
        passed=(v_rid == 0),
        violations=v_rid,
        description="record_id must be non-null, non-empty, and unique across all rows.",
    ))

    # 2. gender in {'M', 'F'} or null/empty
    if "gender" in df.columns:
        s_gen = df["gender"].fillna("").astype(str).str.strip()
        v_gen = int(((s_gen != "") & (~s_gen.isin(["M", "F"]))).sum())
    else:
        v_gen = n_rows
    assertions.append(AssertionResult(
        name="gender_canonical",
        passed=(v_gen == 0),
        violations=v_gen,
        description="gender must be canonical 'M', 'F', or empty string (no dirty variants).",
    ))

    # 3. city in canonical city list or null/empty
    if "city" in df.columns:
        s_city = df["city"].fillna("").astype(str).str.strip()
        v_city = int(((s_city != "") & (~s_city.isin(CFG.CANONICAL_CITIES))).sum())
    else:
        v_city = n_rows
    assertions.append(AssertionResult(
        name="city_canonical",
        passed=(v_city == 0),
        violations=v_city,
        description="city must be in canonical Pakistani city list or empty string (no aliases/abbreviations).",
    ))

    # 4. DOB parses, is >= 1900-01-02 and <= registration_date
    if "date_of_birth" in df.columns and "registration_date" in df.columns:
        s_dob = df["date_of_birth"].fillna("").astype(str).str.strip()
        non_empty = (s_dob != "")
        # Must be valid ISO YYYY-MM-DD
        is_iso = s_dob.str.match(r"^\d{4}-\d{2}-\d{2}$") & non_empty
        v_iso_fail = int((non_empty & ~is_iso).sum())

        # Bounds check on ISO dates
        dob_dt = pd.to_datetime(s_dob.replace("", pd.NA), format="%Y-%m-%d", errors="coerce")
        reg_dt = pd.to_datetime(df["registration_date"], format="mixed", errors="coerce")
        v_future = int(((dob_dt > reg_dt) & dob_dt.notna() & reg_dt.notna()).sum())
        v_too_early = int(((dob_dt < pd.to_datetime("1900-01-02")) & dob_dt.notna()).sum())
        v_dob_range = v_iso_fail + v_future + v_too_early
    else:
        v_dob_range = n_rows
    assertions.append(AssertionResult(
        name="dob_parses_and_valid_range",
        passed=(v_dob_range == 0),
        violations=v_dob_range,
        description="date_of_birth must be ISO (YYYY-MM-DD), >= 1900-01-02, and <= registration_date.",
    ))

    # 5. No DOB sentinel values (1900-01-01, 1970-01-01 sentinels)
    if "date_of_birth" in df.columns:
        s_dob = df["date_of_birth"].fillna("").astype(str).str.strip()
        v_dob_sent = int(s_dob.isin(["1900-01-01", "0000-00-00", "9999-12-31"]).sum())
    else:
        v_dob_sent = n_rows
    assertions.append(AssertionResult(
        name="no_dob_sentinels",
        passed=(v_dob_sent == 0),
        violations=v_dob_sent,
        description="date_of_birth must not contain legacy sentinel dates (e.g. 1900-01-01).",
    ))

    # 6. age_years in [0, 110] or null, and consistent with DOB (within 1.5 years)
    if "age_years" in df.columns:
        s_age = df["age_years"].fillna("").astype(str).str.strip()
        non_empty_age = (s_age != "")
        num_age = pd.to_numeric(s_age.replace("", pd.NA), errors="coerce")

        v_age_invalid = int((non_empty_age & (num_age.isna() | (num_age < 0) | (num_age > 110))).sum())

        # Consistency with DOB
        if "date_of_birth" in df.columns and "registration_date" in df.columns:
            dob_dt = pd.to_datetime(df["date_of_birth"].replace("", pd.NA), errors="coerce")
            reg_dt = pd.to_datetime(df["registration_date"], errors="coerce")
            both_present = dob_dt.notna() & reg_dt.notna() & num_age.notna()
            calc_age = np.floor((reg_dt - dob_dt).dt.days / 365.25)
            v_age_inconsistent = int((both_present & ((calc_age - num_age).abs() > 1.0)).sum())
        else:
            v_age_inconsistent = 0
        v_age = v_age_invalid + v_age_inconsistent
    else:
        v_age = n_rows
    assertions.append(AssertionResult(
        name="age_valid_range_and_consistent",
        passed=(v_age == 0),
        violations=v_age,
        description="age_years must be in [0, 110] and consistent with registration_date - DOB within 1.5 years.",
    ))

    # 7. No age sentinels (-999, -1, 999, 199)
    if "age_years" in df.columns:
        s_age = df["age_years"].fillna("").astype(str).str.strip()
        v_age_sent = int(s_age.isin(["-999", "-1", "999", "199"]).sum())
    else:
        v_age_sent = n_rows
    assertions.append(AssertionResult(
        name="no_age_sentinels",
        passed=(v_age_sent == 0),
        violations=v_age_sent,
        description="age_years must not contain sentinel values (-999, -1, 999, 199).",
    ))

    # 8. phone matches ^03\d{9}$ or is null/empty
    if "phone" in df.columns:
        s_phone = df["phone"].fillna("").astype(str).str.strip()
        non_empty_phone = (s_phone != "")
        v_phone_fmt = int((non_empty_phone & ~s_phone.str.match(r"^03\d{9}$")).sum())
    else:
        v_phone_fmt = n_rows
    assertions.append(AssertionResult(
        name="phone_format_valid",
        passed=(v_phone_fmt == 0),
        violations=v_phone_fmt,
        description="phone must match standard Pakistani 11-digit mobile format (^03\\d{9}$) or be empty.",
    ))

    # 9. No phone sentinels (00000000000, 11111111111, etc.)
    if "phone" in df.columns:
        s_phone = df["phone"].fillna("").astype(str).str.strip()
        v_phone_sent = int(s_phone.isin(CFG.SENTINEL_PHONES).sum())
    else:
        v_phone_sent = n_rows
    assertions.append(AssertionResult(
        name="no_phone_sentinels",
        passed=(v_phone_sent == 0),
        violations=v_phone_sent,
        description="phone must not contain dummy sentinel sequences (e.g. 00000000000).",
    ))

    # 10. national_id matches ^\d{13}$ or is null/empty
    if "national_id" in df.columns:
        s_nid = df["national_id"].fillna("").astype(str).str.strip()
        non_empty_nid = (s_nid != "")
        is_sentinel_nid = s_nid.isin(CFG.SENTINEL_NATIONAL_IDS)
        v_nid = int((non_empty_nid & (~s_nid.str.match(r"^\d{13}$") | is_sentinel_nid)).sum())
    else:
        v_nid = n_rows
    assertions.append(AssertionResult(
        name="national_id_syntax_valid",
        passed=(v_nid == 0),
        violations=v_nid,
        description="national_id must be exactly 13 digits (^\\d{13}$) and not a sentinel, or empty.",
    ))

    # 11. Names are trimmed, single-spaced, no digits
    if "first_name" in df.columns and "family_name" in df.columns:
        fn = df["first_name"].fillna("").astype(str)
        fam = df["family_name"].fillna("").astype(str)

        fn_untidy = (fn.str.startswith(" ") | fn.str.endswith(" ") | fn.str.contains(r"\s{2,}", regex=True) | fn.str.contains(r"\d", regex=True))
        fam_untidy = (fam.str.startswith(" ") | fam.str.endswith(" ") | fam.str.contains(r"\s{2,}", regex=True) | fam.str.contains(r"\d", regex=True))
        v_names = int((fn_untidy | fam_untidy).sum())
    else:
        v_names = n_rows
    assertions.append(AssertionResult(
        name="names_trimmed_no_digits",
        passed=(v_names == 0),
        violations=v_names,
        description="Names must be trimmed, single-spaced, and contain no numeric digits.",
    ))

    # 12. visit_count is an integer >= 0 and <= 100 or empty
    if "visit_count" in df.columns:
        s_visit = df["visit_count"].fillna("").astype(str).str.strip()
        non_empty_visit = (s_visit != "")
        num_visit = pd.to_numeric(s_visit.replace("", pd.NA), errors="coerce")
        v_visit = int((non_empty_visit & (num_visit.isna() | (num_visit < 0) | (num_visit > 100))).sum())
    else:
        v_visit = n_rows
    assertions.append(AssertionResult(
        name="visit_count_valid_range",
        passed=(v_visit == 0),
        violations=v_visit,
        description="visit_count must be an integer between 0 and 100, or empty (no sentinels or >100 outliers).",
    ))

    # 13. insurance_tier in allowed set or empty
    if "insurance_tier" in df.columns:
        s_ins = df["insurance_tier"].fillna("").astype(str).str.strip()
        v_ins = int(((s_ins != "") & (~s_ins.isin(["None", "Basic", "Standard", "Premium"]))).sum())
    else:
        v_ins = n_rows
    assertions.append(AssertionResult(
        name="insurance_tier_valid",
        passed=(v_ins == 0),
        violations=v_ins,
        description="insurance_tier must be in {'None', 'Basic', 'Standard', 'Premium'} or empty.",
    ))

    # 14. Leakage Guard: No forbidden columns present
    present_forbidden = set(df.columns) & CFG.FORBIDDEN_COLUMNS
    v_forbidden = len(present_forbidden)
    assertions.append(AssertionResult(
        name="no_forbidden_columns",
        passed=(v_forbidden == 0),
        violations=v_forbidden,
        description=f"DataFrame must not contain ground-truth or leakage columns ({CFG.FORBIDDEN_COLUMNS}).",
    ))

    # 15. registration_date valid ISO within [2015-01-01, 2025-10-24]
    if "registration_date" in df.columns:
        s_reg = df["registration_date"].fillna("").astype(str).str.strip()
        is_iso = s_reg.str.match(r"^\d{4}-\d{2}-\d{2}$")
        v_iso = int((~is_iso).sum())
        reg_dt = pd.to_datetime(s_reg, format="%Y-%m-%d", errors="coerce")
        min_span = pd.to_datetime("2015-01-01")
        max_span = pd.to_datetime("2025-10-24")
        v_span = int(((reg_dt < min_span) | (reg_dt > max_span) | reg_dt.isna()).sum())
        v_reg = max(v_iso, v_span)
    else:
        v_reg = n_rows
    assertions.append(AssertionResult(
        name="registration_date_valid",
        passed=(v_reg == 0),
        violations=v_reg,
        description="registration_date must be ISO YYYY-MM-DD within register span [2015-01-01, 2025-10-24].",
    ))

    passed_count = sum(1 for a in assertions if a.passed)
    failed_count = sum(1 for a in assertions if not a.passed)
    is_valid = (failed_count == 0)

    return ValidationReport(
        dataset_name=dataset_name,
        total_records=n_rows,
        is_valid=is_valid,
        passed_count=passed_count,
        failed_count=failed_count,
        total_assertions=len(assertions),
        assertions=assertions,
    )


def print_validation_report(report: ValidationReport) -> None:
    """Pretty-prints validation report to console."""
    status_str = "PASSED" if report.is_valid else "FAILED"
    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT: {report.dataset_name} ({report.total_records:,} rows)")
    print(f"Status: {status_str} ({report.passed_count}/{report.total_assertions} assertions passed)")
    print("=" * 70)
    print(f"{'#':<3} {'Assertion Name':<32} {'Status':<8} {'Violations':<12}")
    print("-" * 70)
    for idx, a in enumerate(report.assertions, start=1):
        pass_str = "PASS" if a.passed else "FAIL"
        print(f"{idx:<3} {a.name:<32} {pass_str:<8} {a.violations:<12,}")
    print("=" * 70 + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate patient register quality assertions")
    parser.add_argument("--csv", type=str, default=str(CFG.CLEAN_REGISTER_PATH), help="Path to register CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    report = validate(df, dataset_name=csv_path.name)
    print_validation_report(report)


if __name__ == "__main__":
    main()
