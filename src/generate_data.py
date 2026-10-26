"""Synthetic hospital patient register generator.

IMPORTANT: All data produced by this module is SYNTHETIC (artificially generated).
No real patient records are used or produced.  The data is intentionally easier
than real hospital data (see docs/generation_mechanism.md for details).

Pipeline
--------
1. Generate true entities (people) with clean canonical fields.
2. Clone entities that are selected as "duplicates" and apply corruption
   operators to each clone independently.
3. Inject hard negatives (siblings, twins, father/son, shared household).
4. Shuffle record_id so ID order carries no information about entity membership.
5. Write register_raw.csv (no ground truth), ground_truth.csv, fault_log.csv,
   demo_register.csv, sample_batch.csv.
6. Compute and save a sanity report under reports/metrics.json["generation"].

Usage
-----
    python -m src.generate_data              # uses seed from src/config.py
    python -m src.generate_data --seed 99   # override seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import string
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
# Adjust sys.path only when run as __main__ so imports work both ways
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as CFG
from src.name_data import (
    AREA_NAMES,
    BLOOD_GROUPS,
    CANONICAL_CITY_NAMES,
    CITY_DATA,
    FEMALE_FIRST_NAMES,
    GENDER_VARIANTS,
    HONORIFICS,
    INSURANCE_TIER_WEIGHTS,
    INSURANCE_TIERS,
    MALE_FIRST_NAMES,
    FAMILY_NAMES,
    NAME_SPELLING_VARIANTS,
    STREET_TYPES,
)
from src.utils import save_metrics

# ── Generation configuration constants ───────────────────────────────────────
# These values are all documented in docs/generation_mechanism.md.

# Target register size
TARGET_TOTAL_RECORDS: int = 50_000

# Fraction of true entities that have ≥2 records (are "duplicated entities").
# Approximately 10–12 % of entities → documented as a config value.
DUPLICATE_ENTITY_RATE: float = 0.115  # 11.5 % → documented

# Distribution of cluster sizes for duplicated entities.
# Weights for cluster sizes 2, 3, 4 respectively.
CLUSTER_SIZE_WEIGHTS: list[int] = [70, 22, 8]   # sums to 100

# Date range for registration_date
REG_DATE_START: date = date(2015, 1, 1)
REG_DATE_END:   date = date(2025, 10, 24)

# Source systems and their per-record sampling weights
SOURCE_SYSTEMS: list[str] = ["OPD", "Lab", "Radiology", "Pharmacy", "Emergency", "Legacy"]
SOURCE_SYSTEM_WEIGHTS: list[int] = [35, 20, 15, 15, 10, 5]

# ── Per-operator corruption probabilities ─────────────────────────────────────
# Each entry is documented in docs/generation_mechanism.md.
P_NAME_TYPO:              float = 0.35   # random character-level typo on a name token
P_SPELLING_VARIANT:       float = 0.30   # Muhammad → Mohammad, etc.
P_NAME_INITIAL_ONLY:      float = 0.10   # first name replaced by initial only
P_HONORIFIC_PREFIX:       float = 0.08   # add Mr./Haji/Dr. prefix
P_FIRST_FAMILY_SWAP:      float = 0.06   # swap first_name ↔ family_name
P_CASE_WHITESPACE_NOISE:  float = 0.20   # random case / extra spaces
P_DOB_DAY_MONTH_SWAP:     float = 0.15   # swap day and month if both ≤ 12
P_DOB_DIGIT_TRANSPOSE:    float = 0.10   # transpose two adjacent year/day digits
P_DOB_SENTINEL:           float = 0.05   # replace DOB with 1900-01-01
P_DOB_MISSING:            float = 0.07   # DOB set to null
P_PHONE_FORMAT_VARIANT:   float = 0.25   # phone formatting change
P_PHONE_DIGIT_ERROR:      float = 0.15   # one digit wrong in phone
P_PHONE_MISSING:          float = 0.12   # phone set to null
P_NID_MISSING:            float = 0.20   # national_id set to null
P_NID_DIGIT_ERROR:        float = 0.08   # one digit off in national_id
P_CITY_ALIAS:             float = 0.30   # use alias (LHR, lahore) instead of canonical
P_GENDER_VARIANT:         float = 0.20   # M → Male, 1, male, etc.
P_ADDRESS_ABBREVIATE:     float = 0.25   # abbreviate common words in address
P_FATHER_MISSING:         float = 0.18   # father_name set to null
P_AGE_SENTINEL:           float = 0.04   # age 199 (impossible)
P_VISIT_SENTINEL:         float = 0.04   # visit_count 9999
P_DATE_FORMAT_VARIANT:    float = 0.30   # dd/mm/yyyy or mm-dd-yyyy instead of ISO

# Singleton noise probabilities (light corruption on non-duplicate records)
P_SINGLETON_NOISE:        float = 0.05   # probability a singleton gets any noise

# Hard-negative configuration
HARD_NEG_SIBLING_PAIRS:     int = 800    # sibling / near-twin pairs
HARD_NEG_FATHER_SON_PAIRS:  int = 300    # father-son same full name
HARD_NEG_HOUSEHOLD_PAIRS:   int = 500    # shared household (phone + address)
HARD_NEG_COMMON_NAME_EXTRAS: int = 600   # very common names, different DOBs


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _weighted_choice(rng: random.Random, items: list[Any], weights: list[int]) -> Any:
    """Return one item sampled proportionally to weights."""
    total = sum(weights)
    r = rng.randint(1, total)
    running = 0
    for item, w in zip(items, weights):
        running += w
        if r <= running:
            return item
    return items[-1]


def _random_date(rng: random.Random, start: date, end: date) -> date:
    """Return a random date in [start, end]."""
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _generate_dob(rng: random.Random, reg_date: date) -> date:
    """Generate a plausible date of birth (age 0–90 at registration)."""
    max_age = 90
    min_age = 0
    dob_latest = reg_date - timedelta(days=min_age * 365)
    dob_earliest = reg_date - timedelta(days=max_age * 365)
    # Clamp to 1925
    dob_earliest = max(dob_earliest, date(1925, 1, 1))
    return _random_date(rng, dob_earliest, dob_latest)


def _age_from_dob(dob: date, reg_date: date) -> int:
    """Compute age in whole years at registration date."""
    age = reg_date.year - dob.year
    # Subtract 1 if birthday hasn't occurred yet in reg_date's year
    if (reg_date.month, reg_date.day) < (dob.month, dob.day):
        age -= 1
    return max(0, age)


def _generate_phone(rng: random.Random) -> str:
    """Generate a synthetic Pakistani mobile number (03xxxxxxxxx)."""
    # Prefixes: 030x, 031x, 032x, 033x, 034x
    prefix = rng.choice(["030", "031", "032", "033", "034"])
    suffix = "".join([str(rng.randint(0, 9)) for _ in range(8)])
    return prefix + suffix


def _generate_national_id(rng: random.Random) -> str:
    """Generate a synthetic 13-digit national ID (NADRA-style format, synthetic).

    WARNING: These IDs are entirely synthetic and do not correspond to any
    real NADRA records.
    Format: P AAAA SSSSSSS C  (1 + 4 + 7 + 1 = 13 digits)
      P = province digit (1-7)
      AAAA = 4-digit area code
      SSSSSSS = 7-digit serial
      C = 1-digit check
    """
    province = rng.randint(1, 7)             # 1 digit: province code
    area     = rng.randint(1000, 9999)       # 4 digits: area code
    serial   = rng.randint(1000000, 9999999) # 7 digits: serial number
    check    = rng.randint(0, 9)             # 1 digit: check digit
    raw = f"{province:01d}{area:04d}{serial:07d}{check:01d}"
    assert len(raw) == 13, f"NID length error: {len(raw)} for '{raw}'"
    return raw


def _generate_address(rng: random.Random, city: str) -> str:
    """Generate a synthetic free-text address string."""
    house_num = rng.randint(1, 999)
    street_num = rng.randint(1, 50)
    street_type = rng.choice(STREET_TYPES)
    area = rng.choice(AREA_NAMES)
    return f"House {house_num}, Street {street_num} {street_type}, {area}, {city}"


def _pick_name(rng: random.Random, name_pool: list[tuple[str, int]]) -> str:
    """Sample a name from a weighted pool."""
    names = [n for n, _ in name_pool]
    weights = [w for _, w in name_pool]
    return _weighted_choice(rng, names, weights)


def _build_name_weights(pool: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
    names  = [n for n, _ in pool]
    weights = [w for _, w in pool]
    return names, weights


# ═══════════════════════════════════════════════════════════════════════════════
# Entity generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_entity(
    rng: random.Random,
    entity_id: int,
    first_name_override: str | None = None,
    family_name_override: str | None = None,
    dob_override: date | None = None,
    phone_override: str | None = None,
    address_override: str | None = None,
) -> dict[str, Any]:
    """Generate one canonical (clean) entity record.

    Overrides are used when creating hard negatives that share certain fields.
    Returns a dict with all clean field values.
    """
    # ── gender ──────────────────────────────────────────────────────────────
    gender = rng.choice(["M", "F"])

    # ── name ────────────────────────────────────────────────────────────────
    if gender == "M":
        first_name = first_name_override or _pick_name(rng, MALE_FIRST_NAMES)
        father_first = _pick_name(rng, MALE_FIRST_NAMES)
    else:
        first_name = first_name_override or _pick_name(rng, FEMALE_FIRST_NAMES)
        father_first = _pick_name(rng, MALE_FIRST_NAMES)

    family_name = family_name_override or _pick_name(rng, FAMILY_NAMES)
    father_name = f"{father_first} {family_name}"

    # ── city / address ───────────────────────────────────────────────────────
    city_entry = _weighted_choice(rng, CITY_DATA, [w for _, _, w in CITY_DATA])
    city = city_entry[0]

    address = address_override or _generate_address(rng, city)
    phone   = phone_override   or _generate_phone(rng)

    # ── DOB and registration ─────────────────────────────────────────────────
    # We will assign registration_date(s) later per record; use a placeholder here
    dob = dob_override or _generate_dob(rng, REG_DATE_END)

    # ── other fields ─────────────────────────────────────────────────────────
    national_id  = _generate_national_id(rng)
    blood_group  = rng.choice(BLOOD_GROUPS)
    insurance_tier = _weighted_choice(rng, INSURANCE_TIERS, INSURANCE_TIER_WEIGHTS)
    visit_count  = rng.randint(0, 50)   # clean range; outliers injected later

    return {
        "_entity_id":    entity_id,
        "first_name":    first_name,
        "family_name":   family_name,
        "father_name":   father_name,
        "gender":        gender,
        "date_of_birth": dob,
        "national_id":   national_id,
        "phone":         phone,
        "city":          city,
        "address":       address,
        "blood_group":   blood_group,
        "insurance_tier": insurance_tier,
        "visit_count":   visit_count,
    }


def _generate_registration_dates(
    rng: random.Random,
    cluster_size: int,
    entity_id: int,
) -> list[date]:
    """Generate registration_date for each record in a cluster.

    Dates are spread across the register span; within a cluster the earliest
    date defines the entity's temporal split assignment.
    """
    dates = sorted(
        _random_date(rng, REG_DATE_START, REG_DATE_END)
        for _ in range(cluster_size)
    )
    return dates


def _assign_source_system(rng: random.Random) -> str:
    """Pick a source system weighted by real-world frequency."""
    return _weighted_choice(rng, SOURCE_SYSTEMS, SOURCE_SYSTEM_WEIGHTS)


# ═══════════════════════════════════════════════════════════════════════════════
# Corruption operators
# ═══════════════════════════════════════════════════════════════════════════════

def _typo_name(rng: random.Random, name: str) -> str:
    """Apply a single character-level typo (insert/delete/substitute/transpose)."""
    if len(name) < 2:
        return name
    op = rng.choice(["insert", "delete", "substitute", "transpose"])
    chars = list(name)
    pos = rng.randint(0, len(chars) - 1)

    if op == "insert":
        chars.insert(pos, rng.choice(string.ascii_lowercase))
    elif op == "delete" and len(chars) > 1:
        del chars[pos]
    elif op == "substitute":
        chars[pos] = rng.choice(string.ascii_lowercase)
    elif op == "transpose" and pos < len(chars) - 1:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

    result = "".join(chars)
    # Preserve original capitalisation on the first letter
    return result.capitalize()


def _apply_spelling_variant(rng: random.Random, name: str) -> str:
    """Replace a name token with a known spelling variant if one exists."""
    variants = NAME_SPELLING_VARIANTS.get(name)
    if variants:
        return rng.choice(variants)
    return name


def _corrupt_name_field(
    rng: random.Random,
    first: str,
    family: str,
    fault_list: list[dict],
    base_record_id: str,
) -> tuple[str, str]:
    """Apply name corruption operators; record each fault injected."""

    orig_first, orig_family = first, family

    # Spelling variant on first name
    if rng.random() < P_SPELLING_VARIANT:
        first = _apply_spelling_variant(rng, first)
        if first != orig_first:
            fault_list.append({
                "record_id": base_record_id,
                "column": "first_name",
                "fault_type": "inconsistent_encoding_spelling_variant",
                "original_value": orig_first,
                "injected_value": first,
            })
            orig_first = first

    # Typo on first name
    if rng.random() < P_NAME_TYPO:
        first = _typo_name(rng, first)
        if first != orig_first:
            fault_list.append({
                "record_id": base_record_id,
                "column": "first_name",
                "fault_type": "measurement_error_name_typo",
                "original_value": orig_first,
                "injected_value": first,
            })

    # Initial only
    if rng.random() < P_NAME_INITIAL_ONLY:
        orig = first
        first = first[0] + "."
        fault_list.append({
            "record_id": base_record_id,
            "column": "first_name",
            "fault_type": "collection_error_initial_only",
            "original_value": orig,
            "injected_value": first,
        })

    # Honorific prefix
    if rng.random() < P_HONORIFIC_PREFIX:
        orig = first
        first = rng.choice(HONORIFICS) + " " + first
        fault_list.append({
            "record_id": base_record_id,
            "column": "first_name",
            "fault_type": "noise_honorific_prefix",
            "original_value": orig,
            "injected_value": first,
        })

    # First / family swap
    if rng.random() < P_FIRST_FAMILY_SWAP:
        orig_f, orig_l = first, family
        first, family = family, first
        fault_list.append({
            "record_id": base_record_id,
            "column": "first_name,family_name",
            "fault_type": "noise_first_family_swap",
            "original_value": f"{orig_f}|{orig_l}",
            "injected_value": f"{first}|{family}",
        })

    # Case / whitespace noise
    if rng.random() < P_CASE_WHITESPACE_NOISE:
        choice = rng.choice(["upper", "lower", "extra_space"])
        orig = first
        if choice == "upper":
            first = first.upper()
        elif choice == "lower":
            first = first.lower()
        else:
            first = first + " "   # trailing space — removed during cleaning
        fault_list.append({
            "record_id": base_record_id,
            "column": "first_name",
            "fault_type": "noise_case_whitespace",
            "original_value": orig,
            "injected_value": repr(first),
        })

    # Typo on family name (independent roll)
    if rng.random() < P_NAME_TYPO * 0.7:
        orig = family
        family = _typo_name(rng, family)
        if family != orig:
            fault_list.append({
                "record_id": base_record_id,
                "column": "family_name",
                "fault_type": "measurement_error_name_typo",
                "original_value": orig,
                "injected_value": family,
            })

    return first, family


def _corrupt_dob(
    rng: random.Random,
    dob: date,
    fault_list: list[dict],
    record_id: str,
) -> date | str | None:
    """Apply DOB corruption; returns the (possibly corrupted) value."""

    orig = dob

    # Sentinel
    if rng.random() < P_DOB_SENTINEL:
        fault_list.append({
            "record_id": record_id,
            "column": "date_of_birth",
            "fault_type": "sentinel_value_dob",
            "original_value": str(orig),
            "injected_value": "1900-01-01",
        })
        return "1900-01-01"   # returned as string; will cause parse failure in validate

    # Missing
    if rng.random() < P_DOB_MISSING:
        fault_list.append({
            "record_id": record_id,
            "column": "date_of_birth",
            "fault_type": "missing_value_dob",
            "original_value": str(orig),
            "injected_value": "",
        })
        return None

    # Day-month swap
    if rng.random() < P_DOB_DAY_MONTH_SWAP and dob.day <= 12:
        swapped = date(dob.year, dob.day, dob.month)
        fault_list.append({
            "record_id": record_id,
            "column": "date_of_birth",
            "fault_type": "invalid_value_dob_day_month_swap",
            "original_value": str(orig),
            "injected_value": str(swapped),
        })
        dob = swapped

    # Digit transposition in year
    if rng.random() < P_DOB_DIGIT_TRANSPOSE:
        year_str = list(str(dob.year))
        pos = rng.randint(0, 2)
        year_str[pos], year_str[pos + 1] = year_str[pos + 1], year_str[pos]
        try:
            new_year = int("".join(year_str))
            new_dob = date(new_year, dob.month, dob.day)
            fault_list.append({
                "record_id": record_id,
                "column": "date_of_birth",
                "fault_type": "measurement_error_dob_digit_transpose",
                "original_value": str(orig),
                "injected_value": str(new_dob),
            })
            dob = new_dob
        except ValueError:
            pass  # ignore if transposition creates invalid date

    return dob


def _format_dob(rng: random.Random, dob: date | str | None, source_system: str) -> str | None:
    """Apply date format variants based on source system conventions.

    Legacy uses dd/mm/yyyy.  Emergency and Pharmacy sometimes use mm-dd-yyyy.
    Others use ISO (yyyy-mm-dd).
    """
    if dob is None:
        return None
    if isinstance(dob, str):   # already corrupted to a string sentinel
        return dob

    # Legacy always uses dd/mm/yyyy (collection error: non-standard format)
    if source_system == "Legacy":
        return dob.strftime("%d/%m/%Y")

    # Random format variant on any source
    if rng.random() < P_DATE_FORMAT_VARIANT:
        fmt = rng.choice(["%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"])
        return dob.strftime(fmt)

    return dob.isoformat()   # default ISO


def _corrupt_phone(
    rng: random.Random,
    phone: str,
    fault_list: list[dict],
    record_id: str,
) -> str | None:
    """Apply phone corruption operators."""
    orig = phone

    # Missing
    if rng.random() < P_PHONE_MISSING:
        fault_list.append({
            "record_id": record_id,
            "column": "phone",
            "fault_type": "missing_value_phone",
            "original_value": orig,
            "injected_value": "",
        })
        return None

    # Sentinel
    if rng.random() < 0.03:
        fault_list.append({
            "record_id": record_id,
            "column": "phone",
            "fault_type": "sentinel_value_phone",
            "original_value": orig,
            "injected_value": "00000000000",
        })
        return "00000000000"

    # One digit error
    if rng.random() < P_PHONE_DIGIT_ERROR:
        digits = list(phone)
        pos = rng.randint(2, len(digits) - 1)  # avoid changing the prefix "03"
        digits[pos] = str((int(digits[pos]) + rng.randint(1, 9)) % 10)
        corrupted = "".join(digits)
        fault_list.append({
            "record_id": record_id,
            "column": "phone",
            "fault_type": "measurement_error_phone_digit",
            "original_value": orig,
            "injected_value": corrupted,
        })
        phone = corrupted

    # Format variant
    if rng.random() < P_PHONE_FORMAT_VARIANT:
        orig_fmt = phone
        choice = rng.choice(["dashes", "spaces", "plus92", "0092"])
        if choice == "dashes" and len(phone) == 11:
            phone = f"{phone[:4]}-{phone[4:7]}-{phone[7:]}"
        elif choice == "spaces" and len(phone) == 11:
            phone = f"{phone[:4]} {phone[4:]}"
        elif choice == "plus92" and phone.startswith("0"):
            phone = "+92" + phone[1:]
        elif choice == "0092" and phone.startswith("0"):
            phone = "0092" + phone[1:]
        fault_list.append({
            "record_id": record_id,
            "column": "phone",
            "fault_type": "inconsistent_encoding_phone_format",
            "original_value": orig_fmt,
            "injected_value": phone,
        })

    return phone


def _corrupt_national_id(
    rng: random.Random,
    nid: str,
    fault_list: list[dict],
    record_id: str,
) -> str | None:
    """Apply national_id corruption."""
    orig = nid

    # Missing (Legacy system never records it)
    if rng.random() < P_NID_MISSING:
        fault_list.append({
            "record_id": record_id,
            "column": "national_id",
            "fault_type": "missing_value_national_id",
            "original_value": orig,
            "injected_value": "",
        })
        return None

    # One digit off
    if rng.random() < P_NID_DIGIT_ERROR:
        digits = list(nid)
        pos = rng.randint(0, len(digits) - 1)
        digits[pos] = str((int(digits[pos]) + rng.randint(1, 9)) % 10)
        corrupted = "".join(digits)
        fault_list.append({
            "record_id": record_id,
            "column": "national_id",
            "fault_type": "measurement_error_national_id_digit",
            "original_value": orig,
            "injected_value": corrupted,
        })
        return corrupted

    return nid


def _corrupt_city(
    rng: random.Random,
    city: str,
    fault_list: list[dict],
    record_id: str,
) -> str:
    """Replace canonical city with an alias."""
    orig = city
    if rng.random() < P_CITY_ALIAS:
        # Find aliases for this city
        for canon, aliases, _ in CITY_DATA:
            if canon == city and aliases:
                alias = rng.choice(aliases)
                fault_list.append({
                    "record_id": record_id,
                    "column": "city",
                    "fault_type": "inconsistent_encoding_city_alias",
                    "original_value": orig,
                    "injected_value": alias,
                })
                return alias
    return city


def _corrupt_gender(
    rng: random.Random,
    gender: str,
    fault_list: list[dict],
    record_id: str,
) -> str:
    """Replace canonical gender code with a dirty variant."""
    orig = gender
    if rng.random() < P_GENDER_VARIANT:
        variants = GENDER_VARIANTS.get(gender, [])
        if variants:
            dirty = rng.choice(variants)
            fault_list.append({
                "record_id": record_id,
                "column": "gender",
                "fault_type": "inconsistent_encoding_gender",
                "original_value": orig,
                "injected_value": dirty,
            })
            return dirty
    return gender


def _corrupt_address(
    rng: random.Random,
    address: str,
    fault_list: list[dict],
    record_id: str,
) -> str:
    """Abbreviate common words in the address."""
    orig = address
    if rng.random() < P_ADDRESS_ABBREVIATE:
        abbreviations = {
            "Street": "St", "Road": "Rd", "Avenue": "Ave",
            "House": "H#", "Block": "Blk",
        }
        for word, abbr in abbreviations.items():
            if word in address:
                address = address.replace(word, abbr, 1)
                break
        if address != orig:
            fault_list.append({
                "record_id": record_id,
                "column": "address",
                "fault_type": "noise_address_abbreviation",
                "original_value": orig,
                "injected_value": address,
            })
    return address


def _corrupt_age(
    rng: random.Random,
    age: int,
    fault_list: list[dict],
    record_id: str,
) -> int:
    """Inject an impossible age value (outlier / invalid value)."""
    orig = age
    if rng.random() < P_AGE_SENTINEL:
        bad_age = rng.choice([199, 999, -1, -999])
        fault_list.append({
            "record_id": record_id,
            "column": "age_years",
            "fault_type": "invalid_value_age_impossible",
            "original_value": str(orig),
            "injected_value": str(bad_age),
        })
        return bad_age
    return age


def _corrupt_visit_count(
    rng: random.Random,
    vc: int,
    fault_list: list[dict],
    record_id: str,
) -> int:
    """Inject a sentinel / outlier visit_count."""
    orig = vc
    if rng.random() < P_VISIT_SENTINEL:
        bad = rng.choice([9999, -1, -999])
        fault_list.append({
            "record_id": record_id,
            "column": "visit_count",
            "fault_type": "sentinel_value_visit_count",
            "original_value": str(orig),
            "injected_value": str(bad),
        })
        return bad
    return vc


def _apply_source_system_gaps(
    rng: random.Random,
    row: dict[str, Any],
    source: str,
    fault_list: list[dict],
) -> None:
    """Each source system has structural missingness patterns (collection errors).

    This is NOT random: it models how each departmental system was designed.
    """
    rid = row["record_id"]

    if source == "Legacy":
        # Legacy never stored national_id (collection error)
        if row.get("national_id") is not None:
            fault_list.append({
                "record_id": rid,
                "column": "national_id",
                "fault_type": "collection_error_legacy_no_nid",
                "original_value": row["national_id"],
                "injected_value": "",
            })
            row["national_id"] = None
        # Legacy uses dd/mm/yyyy (already handled in _format_dob)

    elif source == "Emergency":
        # Emergency often lacks father_name (collection error — not collected)
        if rng.random() < 0.70:
            if row.get("father_name") is not None:
                fault_list.append({
                    "record_id": rid,
                    "column": "father_name",
                    "fault_type": "collection_error_emergency_no_father",
                    "original_value": row["father_name"],
                    "injected_value": "",
                })
                row["father_name"] = None
        # Emergency often lacks phone
        if rng.random() < 0.40:
            if row.get("phone") is not None:
                fault_list.append({
                    "record_id": rid,
                    "column": "phone",
                    "fault_type": "collection_error_emergency_no_phone",
                    "original_value": row["phone"],
                    "injected_value": "",
                })
                row["phone"] = None

    elif source == "Lab":
        # Lab rarely records insurance_tier
        if rng.random() < 0.50:
            row["insurance_tier"] = None

    elif source == "Pharmacy":
        # Pharmacy rarely records blood_group
        if rng.random() < 0.60:
            row["blood_group"] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Record materialisation
# ═══════════════════════════════════════════════════════════════════════════════

def _materialise_record(
    rng: random.Random,
    entity: dict[str, Any],
    is_duplicate: bool,
    reg_date: date,
    source_system: str,
    placeholder_id: str,
    fault_list: list[dict],
    apply_corruption: bool = True,
) -> dict[str, Any]:
    """Turn one entity dict into a single register row.

    If is_duplicate=True and apply_corruption=True, corruption operators run.
    """
    rid = placeholder_id
    row: dict[str, Any] = {}

    row["record_id"]        = rid
    row["source_system"]    = source_system
    row["registration_date"] = reg_date.isoformat()  # may get format-corrupted later

    # --- names ---
    first  = entity["first_name"]
    family = entity["family_name"]
    father = entity["father_name"]

    if apply_corruption and is_duplicate:
        first, family = _corrupt_name_field(rng, first, family, fault_list, rid)

        # Father name: sometimes missing in duplicate
        if rng.random() < P_FATHER_MISSING:
            fault_list.append({
                "record_id": rid,
                "column": "father_name",
                "fault_type": "missing_value_father_name",
                "original_value": father,
                "injected_value": "",
            })
            father = None

    row["first_name"]  = first
    row["family_name"] = family
    row["father_name"] = father

    # --- gender ---
    gender = entity["gender"]
    if apply_corruption and is_duplicate:
        gender = _corrupt_gender(rng, gender, fault_list, rid)
    row["gender"] = gender

    # --- DOB ---
    dob = entity["date_of_birth"]
    corrupted_dob = dob
    if apply_corruption and is_duplicate:
        corrupted_dob = _corrupt_dob(rng, dob, fault_list, rid)
    # Format according to source system conventions
    row["date_of_birth"] = _format_dob(rng, corrupted_dob, source_system)

    # --- age_years ---
    if isinstance(corrupted_dob, date):
        age = _age_from_dob(corrupted_dob, reg_date)
    else:
        # DOB was corrupted to sentinel or null; compute from original
        age = _age_from_dob(dob, reg_date)
    if apply_corruption and is_duplicate:
        age = _corrupt_age(rng, age, fault_list, rid)
    row["age_years"] = age

    # --- identifiers ---
    nid = entity["national_id"]
    if apply_corruption and is_duplicate:
        nid = _corrupt_national_id(rng, nid, fault_list, rid)
    row["national_id"] = nid

    phone = entity["phone"]
    if apply_corruption and is_duplicate:
        phone = _corrupt_phone(rng, phone, fault_list, rid)
    row["phone"] = phone

    # --- location ---
    city = entity["city"]
    if apply_corruption and is_duplicate:
        city = _corrupt_city(rng, city, fault_list, rid)
    row["city"] = city

    address = entity["address"]
    if apply_corruption and is_duplicate:
        address = _corrupt_address(rng, address, fault_list, rid)
    row["address"] = address

    # --- other fields ---
    row["blood_group"]     = entity["blood_group"]
    row["insurance_tier"]  = entity["insurance_tier"]

    vc = entity["visit_count"]
    if apply_corruption and is_duplicate:
        vc = _corrupt_visit_count(rng, vc, fault_list, rid)
    row["visit_count"] = vc

    # Apply source-system structural gaps (collection errors)
    _apply_source_system_gaps(rng, row, source_system, fault_list)

    # Light singleton noise
    if not is_duplicate and rng.random() < P_SINGLETON_NOISE:
        _apply_singleton_noise(rng, row, fault_list)

    return row


def _apply_singleton_noise(
    rng: random.Random,
    row: dict[str, Any],
    fault_list: list[dict],
) -> None:
    """Apply a single light noise operator to a singleton record.

    This models normal data quality issues unrelated to duplication.
    """
    rid = row["record_id"]
    choice = rng.choice(["city_alias", "phone_format", "case"])

    if choice == "city_alias" and row.get("city"):
        row["city"] = _corrupt_city(rng, row["city"], fault_list, rid)
    elif choice == "phone_format" and row.get("phone"):
        row["phone"] = _corrupt_phone(rng, row["phone"], fault_list, rid)
    elif choice == "case" and row.get("first_name"):
        orig = row["first_name"]
        row["first_name"] = orig.upper() if rng.random() < 0.5 else orig.lower()
        fault_list.append({
            "record_id": rid,
            "column": "first_name",
            "fault_type": "noise_case_whitespace",
            "original_value": orig,
            "injected_value": row["first_name"],
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Hard-negative generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_sibling_pair(
    rng: random.Random,
    entity: dict[str, Any],
    entity_id_a: int,
    entity_id_b: int,
    twin: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create two sibling entities sharing family_name, address, phone.

    Twins share the same DOB; siblings have DOBs within 1-5 years of each other.
    """
    # Sibling B starts as a copy and gets a different first name and DOB
    b = dict(entity)
    b["_entity_id"] = entity_id_b

    # Same surname, address, phone — intentional hard negative
    # Different first name
    if entity["gender"] == "M":
        b["first_name"] = _pick_name(rng, MALE_FIRST_NAMES)
    else:
        b["first_name"] = _pick_name(rng, FEMALE_FIRST_NAMES)

    if twin:
        # Same DOB
        pass  # b keeps entity's DOB
    else:
        # DOB within 1–5 years (sibling)
        delta_days = rng.randint(365, 5 * 365)
        sign = rng.choice([-1, 1])
        dob_b = entity["date_of_birth"] + timedelta(days=sign * delta_days)
        # Clamp to valid range
        dob_b = max(dob_b, date(1925, 1, 2))
        dob_b = min(dob_b, date(2025, 1, 1))
        b["date_of_birth"] = dob_b

    a = dict(entity)
    a["_entity_id"] = entity_id_a
    return a, b


def _generate_father_son_pair(
    rng: random.Random,
    entity: dict[str, Any],
    entity_id_a: int,
    entity_id_b: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a father-son pair with identical first_name + family_name.

    DOBs differ by 20–35 years.
    """
    father = dict(entity)
    father["_entity_id"] = entity_id_a

    son = dict(entity)
    son["_entity_id"] = entity_id_b
    son["national_id"] = _generate_national_id(rng)
    son["phone"]       = _generate_phone(rng)

    # Son is 20–35 years younger
    age_gap = rng.randint(20 * 365, 35 * 365)
    son_dob = entity["date_of_birth"] + timedelta(days=age_gap)
    son_dob = min(son_dob, date(2025, 1, 1))
    son["date_of_birth"] = son_dob

    return father, son


# ═══════════════════════════════════════════════════════════════════════════════
# Main generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_register(seed: int) -> None:
    """Main entry point: generate the full synthetic patient register.

    Writes:
      data/raw/register_raw.csv
      data/raw/demo_register.csv
      data/raw/sample_batch.csv
      data/truth/ground_truth.csv
      data/truth/fault_log.csv
      reports/metrics.json  (key: "generation")
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    print(f"[generate_data] seed={seed}")

    # ── 1. Plan cluster sizes ────────────────────────────────────────────────
    # We want ~50,000 records total.
    # Let d = duplicate entity rate, and mean cluster size for duplicated = ~2.3
    # n_dup = n_entities * d,  n_single = n_entities * (1-d)
    # total_records ≈ n_single * 1 + n_dup * mean_cluster
    # Solving: n_entities ≈ TARGET / (1 - d + d * mean_cluster)
    mean_cluster = (2 * CLUSTER_SIZE_WEIGHTS[0] + 3 * CLUSTER_SIZE_WEIGHTS[1] + 4 * CLUSTER_SIZE_WEIGHTS[2]) / sum(CLUSTER_SIZE_WEIGHTS)
    n_entities = int(round(TARGET_TOTAL_RECORDS / (1 - DUPLICATE_ENTITY_RATE + DUPLICATE_ENTITY_RATE * mean_cluster)))
    print(f"[generate_data] planning {n_entities} entities, mean_cluster={mean_cluster:.2f}")

    # ── 2. Assign cluster sizes ──────────────────────────────────────────────
    n_dup_entities = int(round(n_entities * DUPLICATE_ENTITY_RATE))
    n_single_entities = n_entities - n_dup_entities

    cluster_sizes: list[int] = [1] * n_single_entities
    for _ in range(n_dup_entities):
        sz = _weighted_choice(rng, [2, 3, 4], CLUSTER_SIZE_WEIGHTS)
        cluster_sizes.append(sz)
    rng.shuffle(cluster_sizes)

    total_planned = sum(cluster_sizes)
    print(f"[generate_data] planned records: {total_planned}")

    # ── 3. Generate entities and records ─────────────────────────────────────
    all_rows: list[dict[str, Any]] = []
    ground_truth_rows: list[dict[str, str]] = []
    fault_log_rows: list[dict[str, str]] = []

    # entity_id counter starts at 1
    entity_id_counter = 0
    placeholder_id_counter = 0  # will be shuffled later

    hard_neg_counts: dict[str, int] = {
        "sibling_pairs": 0,
        "twin_pairs": 0,
        "father_son_pairs": 0,
        "household_pairs": 0,
        "common_name_entities": 0,
    }

    # We need a pool of extra entity IDs for hard negatives
    # Reserve space after the normal entities
    ENTITY_BASE = 1

    # Generate normal entities first
    normal_entity_rows: list[tuple[dict, list[dict]]] = []  # (entity_dict, records)

    for cluster_size in cluster_sizes:
        entity_id_counter += 1
        eid = entity_id_counter

        entity = _generate_entity(rng, eid)

        reg_dates = _generate_registration_dates(rng, cluster_size, eid)
        sources   = [_assign_source_system(rng) for _ in range(cluster_size)]

        for rec_idx, (reg_date, source) in enumerate(zip(reg_dates, sources)):
            placeholder_id_counter += 1
            pid = f"TMP{placeholder_id_counter:07d}"
            is_dup = (rec_idx > 0)   # first record is "canonical"; others are corrupted copies
            row = _materialise_record(
                rng, entity,
                is_duplicate=is_dup,
                reg_date=reg_date,
                source_system=source,
                placeholder_id=pid,
                fault_list=fault_log_rows,
                apply_corruption=True,
            )
            all_rows.append(row)
            ground_truth_rows.append({"record_id": pid, "true_entity_id": f"E{eid:07d}"})

    print(f"[generate_data] generated {len(all_rows)} normal records")

    # ── 4. Hard negatives ────────────────────────────────────────────────────
    # 4a. Sibling pairs
    for pair_idx in range(HARD_NEG_SIBLING_PAIRS):
        entity_id_counter += 1
        eid_a = entity_id_counter
        entity_id_counter += 1
        eid_b = entity_id_counter

        base_entity = _generate_entity(rng, eid_a)
        twin = (pair_idx < HARD_NEG_SIBLING_PAIRS // 10)  # first 10% are twins
        ent_a, ent_b = _generate_sibling_pair(rng, base_entity, eid_a, eid_b, twin=twin)

        for ent, eid in [(ent_a, eid_a), (ent_b, eid_b)]:
            cluster_size = rng.choice([1, 1, 2])  # some siblings have a duplicate record too
            reg_dates = _generate_registration_dates(rng, cluster_size, eid)
            sources   = [_assign_source_system(rng) for _ in range(cluster_size)]
            for rec_idx, (rd, src) in enumerate(zip(reg_dates, sources)):
                placeholder_id_counter += 1
                pid = f"TMP{placeholder_id_counter:07d}"
                row = _materialise_record(
                    rng, ent,
                    is_duplicate=(rec_idx > 0),
                    reg_date=rd,
                    source_system=src,
                    placeholder_id=pid,
                    fault_list=fault_log_rows,
                )
                all_rows.append(row)
                ground_truth_rows.append({"record_id": pid, "true_entity_id": f"E{eid:07d}"})

        if twin:
            hard_neg_counts["twin_pairs"] += 1
        else:
            hard_neg_counts["sibling_pairs"] += 1

    # 4b. Father-son pairs (same full name, different DOB)
    for _ in range(HARD_NEG_FATHER_SON_PAIRS):
        entity_id_counter += 1
        eid_a = entity_id_counter
        entity_id_counter += 1
        eid_b = entity_id_counter

        base_entity = _generate_entity(rng, eid_a)
        # Force male
        base_entity["gender"] = "M"
        father_ent, son_ent = _generate_father_son_pair(rng, base_entity, eid_a, eid_b)

        for ent, eid in [(father_ent, eid_a), (son_ent, eid_b)]:
            reg_dates = _generate_registration_dates(rng, 1, eid)
            sources   = [_assign_source_system(rng)]
            for rd, src in zip(reg_dates, sources):
                placeholder_id_counter += 1
                pid = f"TMP{placeholder_id_counter:07d}"
                row = _materialise_record(
                    rng, ent,
                    is_duplicate=False,
                    reg_date=rd,
                    source_system=src,
                    placeholder_id=pid,
                    fault_list=fault_log_rows,
                    apply_corruption=False,
                )
                all_rows.append(row)
                ground_truth_rows.append({"record_id": pid, "true_entity_id": f"E{eid:07d}"})

        hard_neg_counts["father_son_pairs"] += 1

    # 4c. Shared household (same phone + address, different people)
    for _ in range(HARD_NEG_HOUSEHOLD_PAIRS):
        entity_id_counter += 1
        eid_a = entity_id_counter
        entity_id_counter += 1
        eid_b = entity_id_counter

        # Generate one base for the shared address/phone
        base_entity = _generate_entity(rng, eid_a)
        shared_phone   = base_entity["phone"]
        shared_address = base_entity["address"]

        b_entity = _generate_entity(
            rng, eid_b,
            phone_override=shared_phone,
            address_override=shared_address,
            family_name_override=base_entity["family_name"],  # same household = same family name
        )

        for ent, eid in [(base_entity, eid_a), (b_entity, eid_b)]:
            reg_dates = _generate_registration_dates(rng, 1, eid)
            sources   = [_assign_source_system(rng)]
            for rd, src in zip(reg_dates, sources):
                placeholder_id_counter += 1
                pid = f"TMP{placeholder_id_counter:07d}"
                row = _materialise_record(
                    rng, ent,
                    is_duplicate=False,
                    reg_date=rd,
                    source_system=src,
                    placeholder_id=pid,
                    fault_list=fault_log_rows,
                    apply_corruption=False,
                )
                all_rows.append(row)
                ground_truth_rows.append({"record_id": pid, "true_entity_id": f"E{eid:07d}"})

        hard_neg_counts["household_pairs"] += 1

    # 4d. Very common names with different DOBs
    common_names_m = [("Muhammad", "Ahmed"), ("Muhammad", "Ali"), ("Muhammad", "Khan")]
    common_names_f = [("Fatima", "Khan"), ("Ayesha", "Ali")]
    for _ in range(HARD_NEG_COMMON_NAME_EXTRAS):
        entity_id_counter += 1
        eid = entity_id_counter
        if rng.random() < 0.6:
            fn, ln = rng.choice(common_names_m)
            gender = "M"
        else:
            fn, ln = rng.choice(common_names_f)
            gender = "F"

        entity = _generate_entity(rng, eid,
                                  first_name_override=fn,
                                  family_name_override=ln)
        entity["gender"] = gender

        reg_dates = _generate_registration_dates(rng, 1, eid)
        sources   = [_assign_source_system(rng)]
        for rd, src in zip(reg_dates, sources):
            placeholder_id_counter += 1
            pid = f"TMP{placeholder_id_counter:07d}"
            row = _materialise_record(
                rng, entity,
                is_duplicate=False,
                reg_date=rd,
                source_system=src,
                placeholder_id=pid,
                fault_list=fault_log_rows,
                apply_corruption=False,
            )
            all_rows.append(row)
            ground_truth_rows.append({"record_id": pid, "true_entity_id": f"E{eid:07d}"})

        hard_neg_counts["common_name_entities"] += 1

    print(f"[generate_data] total rows before shuffle: {len(all_rows)}")

    # ── 5. Shuffle record_id ─────────────────────────────────────────────────
    # Create new random record_ids and shuffle them so that ID proximity carries
    # no information about entity membership.
    n_records = len(all_rows)
    new_ids = [f"REC{i:07d}" for i in range(1, n_records + 1)]
    rng.shuffle(new_ids)

    # Build mapping from placeholder → real id
    placeholder_to_real: dict[str, str] = {}
    for row, new_id in zip(all_rows, new_ids):
        placeholder_to_real[row["record_id"]] = new_id
        row["record_id"] = new_id

    # Update ground truth with real IDs
    for gt_row in ground_truth_rows:
        gt_row["record_id"] = placeholder_to_real[gt_row["record_id"]]

    # Update fault log with real IDs
    for fl_row in fault_log_rows:
        fl_row["record_id"] = placeholder_to_real.get(fl_row["record_id"], fl_row["record_id"])

    # ── 6. Build DataFrames ──────────────────────────────────────────────────
    COLUMN_ORDER = [
        "record_id", "source_system", "registration_date",
        "first_name", "family_name", "father_name",
        "gender", "date_of_birth", "age_years", "national_id",
        "phone", "city", "address",
        "blood_group", "insurance_tier", "visit_count",
    ]

    df_register = pd.DataFrame(all_rows)[COLUMN_ORDER]
    df_gt       = pd.DataFrame(ground_truth_rows)[["record_id", "true_entity_id"]]
    df_fault    = pd.DataFrame(fault_log_rows) if fault_log_rows else pd.DataFrame(
        columns=["record_id", "column", "fault_type", "original_value", "injected_value"]
    )

    # Shuffle row order (so duplicates don't cluster)
    df_register = df_register.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── 7. Demo and sample subsets ───────────────────────────────────────────
    # Demo: ~5,000 records from whole entities
    demo_df, sample_df = _build_subsets(df_register, df_gt, seed)

    # ── 8. Write outputs ─────────────────────────────────────────────────────
    CFG.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    CFG.DATA_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reg_path = CFG.RAW_REGISTER_PATH
    df_register.to_csv(reg_path, index=False)
    print(f"[generate_data] wrote {len(df_register):,} rows -> {reg_path}")

    gt_path = CFG.GROUND_TRUTH_PATH
    df_gt.to_csv(gt_path, index=False)
    print(f"[generate_data] wrote {len(df_gt):,} rows -> {gt_path}")

    fault_path = CFG.FAULT_LOG_PATH
    df_fault.to_csv(fault_path, index=False)
    print(f"[generate_data] wrote {len(df_fault):,} fault rows -> {fault_path}")

    demo_path = CFG.DEMO_REGISTER_PATH
    demo_df.to_csv(demo_path, index=False)
    print(f"[generate_data] wrote {len(demo_df):,} rows -> {demo_path}")

    sample_path = CFG.SAMPLE_BATCH_PATH
    sample_df.to_csv(sample_path, index=False)
    print(f"[generate_data] wrote {len(sample_df):,} rows -> {sample_path}")

    # ── 9. Sanity report ─────────────────────────────────────────────────────
    report = _build_sanity_report(df_register, df_gt, hard_neg_counts, seed)
    save_metrics({"generation": report})
    print(f"[generate_data] sanity report -> {CFG.METRICS_PATH}")
    _print_sanity_summary(report)

    print("[generate_data] done.")


def _build_subsets(
    df_register: pd.DataFrame,
    df_gt: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build demo (~5,000 rows) and sample (~2,000 rows) subsets.

    Subsets contain WHOLE entities so entity grouping is preserved.
    """
    # Map record_id -> entity_id
    id_to_entity = df_gt.set_index("record_id")["true_entity_id"]
    df_register = df_register.copy()
    df_register["_entity_id"] = df_register["record_id"].map(id_to_entity)

    unique_entities = df_register["_entity_id"].unique().tolist()
    rng2 = random.Random(seed + 7)
    rng2.shuffle(unique_entities)

    # Demo: accumulate whole entities until ~5,000 rows
    demo_entities: list[str] = []
    demo_total = 0
    for eid in unique_entities:
        count = (df_register["_entity_id"] == eid).sum()
        if demo_total + count <= 5500:
            demo_entities.append(eid)
            demo_total += count
        if demo_total >= 4800:
            break

    demo_df = df_register[df_register["_entity_id"].isin(demo_entities)].drop(columns=["_entity_id"])

    # Sample: next entities until ~2,000 rows (non-overlapping with demo)
    remaining_entities = [e for e in unique_entities if e not in set(demo_entities)]
    sample_entities: list[str] = []
    sample_total = 0
    for eid in remaining_entities:
        count = (df_register["_entity_id"] == eid).sum()
        if sample_total + count <= 2200:
            sample_entities.append(eid)
            sample_total += count
        if sample_total >= 1800:
            break

    sample_df = df_register[df_register["_entity_id"].isin(sample_entities)].drop(columns=["_entity_id"])

    return demo_df.reset_index(drop=True), sample_df.reset_index(drop=True)


def _build_sanity_report(
    df: pd.DataFrame,
    df_gt: pd.DataFrame,
    hard_neg_counts: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """Compute sanity metrics; every number comes from data, never hard-coded."""

    total_records = len(df)
    total_entities = df_gt["true_entity_id"].nunique()
    duplicate_rate = round(1 - total_entities / total_records, 4)

    # Cluster-size distribution
    cluster_sizes = df_gt.groupby("true_entity_id").size().value_counts().sort_index()
    cluster_dist = {str(k): int(v) for k, v in cluster_sizes.items()}

    # Per-column missing rate
    missing_rates = {
        col: round(float(df[col].isna().mean()), 4)
        for col in df.columns
    }

    # Trivial rule coverage: same DOB and same phone → true pair?
    # Join register with ground truth to get entity for each record
    df_tagged = df.merge(df_gt, on="record_id")

    # Count true duplicate pairs
    dup_entities = df_gt.groupby("true_entity_id").filter(lambda x: len(x) > 1)
    true_dup_entity_set = set(dup_entities["true_entity_id"].unique())
    n_true_dup_entities = len(true_dup_entity_set)
    total_true_dup_pairs = int(
        dup_entities.groupby("true_entity_id")
        .size()
        .apply(lambda n: n * (n - 1) // 2)
        .sum()
    )

    # Trivial rule: join on dob + phone, then check if same entity
    dob_phone_pairs = (
        df_tagged[df_tagged["date_of_birth"].notna() & df_tagged["phone"].notna()]
        .merge(
            df_tagged[["record_id", "date_of_birth", "phone", "true_entity_id"]],
            on=["date_of_birth", "phone"],
            suffixes=("_a", "_b"),
        )
    )
    # Remove self-pairs and keep only a < b order to avoid double counting
    dob_phone_pairs = dob_phone_pairs[
        dob_phone_pairs["record_id_a"] < dob_phone_pairs["record_id_b"]
    ]
    trivial_candidates = len(dob_phone_pairs)
    trivial_true_pairs = int(
        (dob_phone_pairs["true_entity_id_a"] == dob_phone_pairs["true_entity_id_b"]).sum()
    )
    trivial_precision = round(trivial_true_pairs / trivial_candidates, 4) if trivial_candidates > 0 else 0.0
    trivial_recall = round(trivial_true_pairs / total_true_dup_pairs, 4) if total_true_dup_pairs > 0 else 0.0

    # All-pairs count
    n = total_records
    all_pairs = n * (n - 1) // 2

    report: dict[str, Any] = {
        "seed": seed,
        "total_records": total_records,
        "total_entities": total_entities,
        "duplicate_entity_rate_config": DUPLICATE_ENTITY_RATE,
        "duplicate_rate_actual": duplicate_rate,
        "n_true_duplicate_entities": n_true_dup_entities,
        "total_true_duplicate_pairs": total_true_dup_pairs,
        "cluster_size_distribution": cluster_dist,
        "hard_negatives": hard_neg_counts,
        "all_pairs_count": all_pairs,
        "per_column_missing_rate": missing_rates,
        "trivial_rule_same_dob_and_phone": {
            "candidates": trivial_candidates,
            "true_pairs_found": trivial_true_pairs,
            "precision": trivial_precision,
            "recall": trivial_recall,
            "note": (
                "A recall near 1.0 means corruption is too easy; "
                "a recall near 0.0 means blocking may miss many true pairs. "
                f"Actual recall={trivial_recall:.3f} — see generation_mechanism.md."
            ),
        },
        "source_system_distribution": {
            src: int((df["source_system"] == src).sum())
            for src in SOURCE_SYSTEMS
        },
    }
    return report


def _print_sanity_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the sanity report."""
    print("\n" + "=" * 60)
    print("GENERATION SANITY REPORT")
    print("=" * 60)
    print(f"  Total records    : {report['total_records']:,}")
    print(f"  Total entities   : {report['total_entities']:,}")
    print(f"  Duplicate rate   : {report['duplicate_rate_actual']:.1%}")
    print(f"  True dup pairs   : {report['total_true_duplicate_pairs']:,}")
    print(f"  All-pairs (n²/2) : {report['all_pairs_count']:,}")
    print(f"  Hard negatives   : {report['hard_negatives']}")
    tr = report["trivial_rule_same_dob_and_phone"]
    print(f"  Trivial rule (same DOB+phone):")
    print(f"    candidates={tr['candidates']:,}  true_found={tr['true_pairs_found']:,}"
          f"  precision={tr['precision']:.3f}  recall={tr['recall']:.3f}")
    print("  Per-column missing rates:")
    for col, rate in report["per_column_missing_rate"].items():
        if rate > 0:
            print(f"    {col:<20s}: {rate:.1%}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic hospital patient register."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CFG.RANDOM_SEED,
        help=f"Random seed (default: {CFG.RANDOM_SEED} from src/config.py)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_register(seed=args.seed)
