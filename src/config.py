"""Central configuration and constants for the Patient Record Deduplication Tool.

All file paths are defined relative to the repository root using pathlib to ensure
cross-platform compatibility across Windows, Linux, and macOS.
"""

from pathlib import Path

# Repository root directory (parent of src/)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Core subdirectories
SRC_DIR: Path = ROOT_DIR / "src"
PAGES_DIR: Path = ROOT_DIR / "pages"
DATA_DIR: Path = ROOT_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_TRUTH_DIR: Path = DATA_DIR / "truth"
DATA_CLEAN_DIR: Path = DATA_DIR / "clean"
DATA_PAIRS_DIR: Path = DATA_DIR / "pairs"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DOCS_DIR: Path = ROOT_DIR / "docs"
TESTS_DIR: Path = ROOT_DIR / "tests"

# Standard report and metric paths
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"

# Key data file paths
RAW_REGISTER_PATH: Path = DATA_RAW_DIR / "register_raw.csv"
DEMO_REGISTER_PATH: Path = DATA_RAW_DIR / "demo_register.csv"
SAMPLE_BATCH_PATH: Path = DATA_RAW_DIR / "sample_batch.csv"
GROUND_TRUTH_PATH: Path = DATA_TRUTH_DIR / "ground_truth.csv"
FAULT_LOG_PATH: Path = DATA_TRUTH_DIR / "fault_log.csv"
CLEAN_REGISTER_PATH: Path = DATA_CLEAN_DIR / "register_clean.csv"

# Global random seed for reproducible splits, sampling, and model training
RANDOM_SEED: int = 42

# Canonical city list placeholder (standardized Pakistani urban centers)
CANONICAL_CITIES: list[str] = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Rawalpindi",
    "Faisalabad",
    "Multan",
    "Peshawar",
    "Quetta",
    "Gujranwala",
    "Sialkot",
    "Hyderabad",
    "Bahawalpur",
    "Sargodha",
    "Sukkur",
    "Larkana",
]

# Sentinel value sets used in data audit and cleaning rules (Lecture 2)
# Sentinel dates typically injected or used as legacy defaults
SENTINEL_DATES: set[str] = {
    "1900-01-01",
    "1970-01-01",
    "2099-12-31",
    "0000-00-00",
    "9999-12-31",
}

# Sentinel numeric ages representing missingness or corrupt system entries
SENTINEL_AGES: set[int] = {
    -999,
    -1,
    999,
    199,
}

# Sentinel phone strings (e.g., all zeros, test patterns, repeated digits)
SENTINEL_PHONES: set[str] = {
    "00000000000",
    "03000000000",
    "03111111111",
    "03999999999",
    "11111111111",
    "99999999999",
}

# Sentinel national IDs (synthetic 13-digit placeholders)
SENTINEL_NATIONAL_IDS: set[str] = {
    "0000000000000",
    "1111111111111",
    "9999999999999",
}

# Generic missing / sentinel text tokens
SENTINEL_STRINGS: set[str] = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "nil",
    "unknown",
    "missing",
    "test",
    "xxx",
    "?",
}

# Sentinel visit counts indicating unrecorded or corrupted history
SENTINEL_VISIT_COUNTS: set[int] = {
    -1,
    -999,
    9999,
}

# ── Clean DataFrame Column Definitions & Allowlist (Phase 4) ─────────────────
# Columns in data/clean/register_clean.csv that downstream models and feature
# extractors are strictly permitted to consume. Prevents accidental ingestion of
# raw uncleaned columns, provenance metadata, or leaked ground truth.
CLEAN_MODEL_ALLOWLIST: list[str] = [
    "record_id",
    "source_system",
    "registration_date",
    "first_name",
    "family_name",
    "father_name",
    "gender",
    "date_of_birth",
    "age_years",
    "national_id",
    "phone",
    "city",
    "address",
    "blood_group",
    "insurance_tier",
    "visit_count",
]

# Provenance audit columns tracking how each attribute was repaired
CLEAN_PROVENANCE_COLUMNS: list[str] = [
    "first_name_prov",
    "family_name_prov",
    "father_name_prov",
    "gender_prov",
    "city_prov",
    "date_of_birth_prov",
    "age_years_prov",
    "phone_prov",
    "national_id_prov",
    "visit_count_prov",
]

# Quality anomaly and sentinel indicator flags
CLEAN_FLAG_COLUMNS: list[str] = [
    "dob_sentinel_flag",
    "age_sentinel_flag",
    "phone_sentinel_flag",
    "nid_sentinel_flag",
    "visit_outlier_flag",
]

# Strictly prohibited ground-truth and generator metadata columns (leakage guard)
FORBIDDEN_COLUMNS: set[str] = {
    "true_entity_id",
    "fault_type",
    "original_value",
    "injected_value",
    "is_duplicate",
    "merged_into",
    "canonical_id",
}

# Cleaning report paths
CLEAN_EFFECTS_PATH: Path = REPORTS_DIR / "clean_effects.json"
CLEAN_EFFECTS_CSV_PATH: Path = REPORTS_DIR / "clean_effects.csv"

# ── Clinical Cost Matrix and Threshold Constants (Phase 9) ───────────────────
# Asymmetric costs for candidate duplicate classification:
# Cost of investigating a false positive (wasted clerk review time, Rs. 5.0)
DEFAULT_COST_FP: float = 5.0
# Cost of a missed duplicate (split patient record, missing critical history, Rs. 20.0)
DEFAULT_COST_FN: float = 20.0
DEFAULT_COST_TP: float = 0.0
DEFAULT_COST_TN: float = 0.0

# Stricter cost ratio for automated / high-confidence recommended merge band:
# Erroneously merging two different patients causes catastrophic EHR corruption (Rs. 50.0)
STRICT_COST_FP: float = 50.0
STRICT_COST_FN: float = 10.0

# Operational assumption: average time spent by hospital clerk reviewing one flagged pair
# Stated in minutes (used to compute total workload and staffing requirements)
MINUTES_PER_CLERK_REVIEW: float = 3.0

