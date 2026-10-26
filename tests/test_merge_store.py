"""Unit and integration tests for the SQLite MergeStore and Survivorship Rules (Roadmap D8 & 7.11).

Verifies:
1. Merge and Undo byte-for-byte exact equality with original records.
2. Reject and Defer clerical decision recording with timestamps.
3. Idempotent Undo execution (calling undo multiple times is safe and consistent).
4. Concurrent re-opening across multiple MergeStore instances.
5. Reviewed pairs tracking (merged, rejected, deferred pairs, and removal on undo).
6. Labelled pair export for model retraining / audit.
7. Survivorship rule suggestions (completeness, recency, and sentinel elimination).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.merge_store import (
    MergeStore,
    compute_survivorship_suggestions,
)


@pytest.fixture
def temp_store(tmp_path: Path) -> MergeStore:
    """Creates a temporary isolated SQLite MergeStore."""
    db_file = tmp_path / "test_merges.db"
    return MergeStore(db_path=db_file)


# ── 1. Byte-for-Byte Exact Restoration Test ───────────────────────────────────

def test_merge_and_undo_byte_for_byte(temp_store: MergeStore) -> None:
    """Verifies that merging and undoing restores both original records byte-for-byte."""
    orig_a = {
        "record_id": "PAT_001_A",
        "first_name": "Muhammad",
        "family_name": "Tariq",
        "father_name": "Tariq Khan",
        "gender": "M",
        "date_of_birth": "1985-05-12",
        "age_years": "39",
        "national_id": "35201-1234567-1",
        "phone": "03001234567",
        "address": "123 Mall Road, Lahore",
        "city": "Lahore",
        "blood_group": "B+",
        "insurance_tier": "Gold",
        "visit_count": "5",
        "source_system": "Central Hospital",
        "registration_date": "2024-01-10",
    }

    orig_b = {
        "record_id": "PAT_002_B",
        "first_name": "Mohammad",
        "family_name": "Tariq",
        "father_name": "Tariq Khan",
        "gender": "M",
        "date_of_birth": "1985-05-12",
        "age_years": "39",
        "national_id": "3520112345671",
        "phone": "03001234567",
        "address": "123 Mall Rd., Lahore",
        "city": "Lahore",
        "blood_group": "B+",
        "insurance_tier": "Gold",
        "visit_count": "1",
        "source_system": "Outpatient Clinic",
        "registration_date": "2024-05-22",
    }

    golden = {
        "record_id": "GOLD_PAT_001_A_PAT_002_B",
        "first_name": "Muhammad",
        "family_name": "Tariq",
        "phone": "03001234567",
        "national_id": "35201-1234567-1",
    }
    provenance = {
        "first_name": "Record A",
        "family_name": "Record A",
        "phone": "Record B",
    }

    # Execute merge
    merge_id = temp_store.record_merge(
        record_1=orig_a,
        record_2=orig_b,
        golden_record=golden,
        field_provenance=provenance,
        clerk_note="Confirmed identity via telephone contact.",
        model_name="decision_tree",
        duplicate_probability=0.985,
    )

    assert merge_id.startswith("MRG-")
    merge_data = temp_store.get_merge(merge_id)
    assert merge_data is not None
    assert merge_data["status"] == "merged"
    assert merge_data["duplicate_probability"] == pytest.approx(0.985, 1e-4)

    # Undo merge
    restored_a, restored_b = temp_store.undo_merge(merge_id)

    # 1. Dictionary Equality
    assert restored_a == orig_a
    assert restored_b == orig_b

    # 2. Strict JSON Serialization Byte-for-Byte Equality
    orig_a_bytes = json.dumps(orig_a, sort_keys=True).encode("utf-8")
    restored_a_bytes = json.dumps(restored_a, sort_keys=True).encode("utf-8")
    assert orig_a_bytes == restored_a_bytes, "Restored Record A does not match original byte-for-byte!"

    orig_b_bytes = json.dumps(orig_b, sort_keys=True).encode("utf-8")
    restored_b_bytes = json.dumps(restored_b, sort_keys=True).encode("utf-8")
    assert orig_b_bytes == restored_b_bytes, "Restored Record B does not match original byte-for-byte!"

    # Verify status changed to undone
    updated_merge = temp_store.get_merge(merge_id)
    assert updated_merge is not None
    assert updated_merge["status"] == "undone"
    assert updated_merge["undone_timestamp"] is not None


# ── 2. Reject and Defer Decisions Test ────────────────────────────────────────

def test_record_decision_reject_and_defer(temp_store: MergeStore) -> None:
    """Verifies storing reject and defer decisions with timestamps and notes."""
    rec_a = {"record_id": "R1", "first_name": "Zainab"}
    rec_b = {"record_id": "R2", "first_name": "Fatima"}

    # Reject
    rej_id = temp_store.record_decision(
        record_1=rec_a,
        record_2=rec_b,
        decision="reject",
        clerk_note="Twin sisters sharing father and address.",
        model_name="decision_tree",
        duplicate_probability=0.15,
    )
    assert rej_id.startswith("DEC-REJ-")

    # Defer
    def_id = temp_store.record_decision(
        record_1=rec_a,
        record_2=rec_b,
        decision="defer",
        clerk_note="Deferred for supervisor review of paper chart.",
        model_name="decision_tree",
        duplicate_probability=0.65,
    )
    assert def_id.startswith("DEC-DEF-")

    # Invalid decision type
    with pytest.raises(ValueError):
        temp_store.record_decision(rec_a, rec_b, decision="invalid_action")

    decisions = temp_store.get_all_decisions()
    assert len(decisions) == 2
    types = [d["decision"] for d in decisions]
    assert "reject" in types
    assert "defer" in types


# ── 3. Idempotent Undo Test ───────────────────────────────────────────────────

def test_idempotent_undo(temp_store: MergeStore) -> None:
    """Verifies that calling undo multiple times is safe, idempotent, and consistent."""
    rec_a = {"record_id": "A", "val": 100}
    rec_b = {"record_id": "B", "val": 200}

    m_id = temp_store.record_merge(
        record_1=rec_a,
        record_2=rec_b,
        golden_record={"record_id": "AB"},
        field_provenance={},
    )

    # First undo
    res1_a, res1_b = temp_store.undo_merge(m_id)
    assert res1_a == rec_a
    assert res1_b == rec_b

    # Second undo (idempotent)
    res2_a, res2_b = temp_store.undo_merge(m_id)
    assert res2_a == rec_a
    assert res2_b == rec_b

    merge_entry = temp_store.get_merge(m_id)
    assert merge_entry is not None
    assert merge_entry["status"] == "undone"


# ── 4. Concurrent Re-Open Test ────────────────────────────────────────────────

def test_concurrent_reopen(tmp_path: Path) -> None:
    """Verifies ACID transactions across multiple MergeStore instances on the same file."""
    db_file = tmp_path / "shared_merges.db"

    store_1 = MergeStore(db_path=db_file)
    store_2 = MergeStore(db_path=db_file)

    rec_a = {"record_id": "R10", "name": "Patient 10"}
    rec_b = {"record_id": "R20", "name": "Patient 20"}

    # Write in store 1
    m_id = store_1.record_merge(
        record_1=rec_a,
        record_2=rec_b,
        golden_record={"record_id": "GOLD_10_20"},
        field_provenance={},
        clerk_note="Created by store 1",
    )

    # Read and undo in store 2
    merges_in_2 = store_2.get_all_merges()
    assert len(merges_in_2) == 1
    assert merges_in_2[0]["merge_id"] == m_id

    restored_a, restored_b = store_2.undo_merge(m_id)
    assert restored_a == rec_a
    assert restored_b == rec_b

    # Verify updated in store 1
    m_in_1 = store_1.get_merge(m_id)
    assert m_in_1 is not None
    assert m_in_1["status"] == "undone"


# ── 5. Reviewed Pairs Filtering Test ──────────────────────────────────────────

def test_reviewed_pairs_filtering(temp_store: MergeStore) -> None:
    """Verifies get_reviewed_pair_ids correctly tracks active merges and decisions."""
    rec_1 = {"record_id": "REC_A"}
    rec_2 = {"record_id": "REC_B"}
    rec_3 = {"record_id": "REC_C"}
    rec_4 = {"record_id": "REC_D"}

    # Merge REC_A and REC_B
    m_id = temp_store.record_merge(rec_1, rec_2, {"record_id": "G"}, {})
    # Reject REC_C and REC_D
    temp_store.record_decision(rec_3, rec_4, decision="reject")

    reviewed = temp_store.get_reviewed_pair_ids()
    assert ("REC_A", "REC_B") in reviewed
    assert ("REC_C", "REC_D") in reviewed

    # Undo merge on REC_A and REC_B
    temp_store.undo_merge(m_id)
    reviewed_after_undo = temp_store.get_reviewed_pair_ids()

    # Active merge is no longer reviewed (can be re-reviewed), but rejected remains
    assert ("REC_A", "REC_B") not in reviewed_after_undo
    assert ("REC_C", "REC_D") in reviewed_after_undo


# ── 6. Export Decisions as Labelled Pairs Test ─────────────────────────────────

def test_export_decisions_as_labelled_pairs(temp_store: MergeStore) -> None:
    """Verifies export_decisions_as_labelled_pairs produces valid training schemas."""
    r1 = {"record_id": "X1"}
    r2 = {"record_id": "X2"}
    r3 = {"record_id": "X3"}
    r4 = {"record_id": "X4"}

    temp_store.record_merge(r1, r2, {"record_id": "G1"}, {}, clerk_note="Merged pair", duplicate_probability=0.95)
    temp_store.record_decision(r3, r4, decision="reject", clerk_note="Distinct", duplicate_probability=0.05)

    df_export = temp_store.export_decisions_as_labelled_pairs()
    assert isinstance(df_export, pd.DataFrame)
    assert len(df_export) == 2

    expected_cols = {"record_id_1", "record_id_2", "label", "decision_type", "timestamp", "clerk_note"}
    assert expected_cols.issubset(df_export.columns)

    # Check labels
    merge_row = df_export[df_export["decision_type"] == "merge"].iloc[0]
    assert merge_row["label"] == 1

    reject_row = df_export[df_export["decision_type"] == "reject"].iloc[0]
    assert reject_row["label"] == 0


# ── 7. Survivorship Suggestions Rules Engine Test ─────────────────────────────

def test_survivorship_rules_completeness_and_recency() -> None:
    """Verifies rule-based golden record survivorship heuristics."""
    rec_a = {
        "record_id": "A",
        "first_name": "Tariq",
        "family_name": "Khan",
        "phone": "00000000000",  # Sentinel phone!
        "national_id": "35201-1234567-1",  # Complete CNIC
        "city": "Lahore",
        "address": "Old Address, Lahore",
        "visit_count": "2",
        "registration_date": "2023-01-01",  # Older registration
    }
    rec_b = {
        "record_id": "B",
        "first_name": "Tariq Mahmood",  # Longer, more specific name
        "family_name": "Khan",
        "phone": "03001234567",  # Valid phone
        "national_id": "",  # Missing CNIC
        "city": "Lahore",
        "address": "New Address, Model Town, Lahore",
        "visit_count": "5",  # Higher visit count
        "registration_date": "2024-06-15",  # Newer registration!
    }

    golden, choices, rationale = compute_survivorship_suggestions(rec_a, rec_b)

    # Phone: Record A had sentinel -> prefer Record B
    assert choices["phone"] == "Record B"
    assert golden["phone"] == "03001234567"

    # National ID: Record B was missing -> prefer Record A
    assert choices["national_id"] == "Record A"
    assert golden["national_id"] == "35201-1234567-1"

    # Address: Both valid, Record B is newer -> prefer Record B
    assert choices["address"] == "Record B"
    assert golden["address"] == "New Address, Model Town, Lahore"

    # Visit count: Prefer higher visit count (5)
    assert choices["visit_count"] == "Record B"
    assert golden["visit_count"] == "5"

    # First name: Record B has more complete name
    assert choices["first_name"] == "Record B"
    assert golden["first_name"] == "Tariq Mahmood"
