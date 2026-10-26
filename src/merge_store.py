"""SQLite Merge Store and Audit Log for Reversible Patient Record Merges.

Adheres to Roadmap Section 7.11 and Phase 13 requirements:
1. SQLite storage via the standard library `sqlite3` inside `data/merges.db`.
2. Tables:
   - `merges`: merge_id, record_1_snapshot (JSON), record_2_snapshot (JSON),
     golden_record (JSON), field_provenance (JSON), clerk_note, model_name,
     duplicate_probability, timestamp, status ('merged', 'undone'), undone_timestamp.
   - `decisions`: decision_id, record_id_1, record_id_2, record_1_snapshot,
     record_2_snapshot, decision ('reject', 'defer'), clerk_note, model_name,
     duplicate_probability, timestamp.
3. Byte-for-byte exact restoration upon `undo_merge()`.
4. Survivorship rule engine suggesting optimal golden record defaults.
5. Reviewed pairs registry filtering decided pairs out of Screen 4 worklists.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import config as CFG

DEFAULT_DB_PATH: Path = CFG.DATA_DIR / "merges.db"


class MergeStore:
    """Thread-safe SQLite store for clinical record merges and clerical decisions."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection with row factories."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables for merges and clerical review decisions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS merges (
                    merge_id TEXT PRIMARY KEY,
                    record_id_1 TEXT NOT NULL,
                    record_id_2 TEXT NOT NULL,
                    record_1_snapshot TEXT NOT NULL,
                    record_2_snapshot TEXT NOT NULL,
                    golden_record TEXT NOT NULL,
                    field_provenance TEXT NOT NULL,
                    clerk_note TEXT,
                    model_name TEXT,
                    duplicate_probability REAL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    undone_timestamp TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    record_id_1 TEXT NOT NULL,
                    record_id_2 TEXT NOT NULL,
                    record_1_snapshot TEXT NOT NULL,
                    record_2_snapshot TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    clerk_note TEXT,
                    model_name TEXT,
                    duplicate_probability REAL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            # Create indexes for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_merges_status ON merges(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_merges_r1_r2 ON merges(record_id_1, record_id_2)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_r1_r2 ON decisions(record_id_1, record_id_2)")
            conn.commit()

    def record_merge(
        self,
        record_1: dict[str, Any],
        record_2: dict[str, Any],
        golden_record: dict[str, Any],
        field_provenance: dict[str, str],
        clerk_note: str = "",
        model_name: str = "",
        duplicate_probability: float = 0.0,
    ) -> str:
        """Stores a reversible record merge with full JSON snapshots of both originals.
        
        Returns:
            merge_id (str)
        """
        r1_id = str(record_1.get("record_id", "REC1"))
        r2_id = str(record_2.get("record_id", "REC2"))
        merge_id = f"MRG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now().isoformat()

        # Exact JSON serialization
        r1_json = json.dumps(record_1, ensure_ascii=False, sort_keys=True)
        r2_json = json.dumps(record_2, ensure_ascii=False, sort_keys=True)
        golden_json = json.dumps(golden_record, ensure_ascii=False, sort_keys=True)
        prov_json = json.dumps(field_provenance, ensure_ascii=False, sort_keys=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO merges (
                    merge_id, record_id_1, record_id_2, record_1_snapshot,
                    record_2_snapshot, golden_record, field_provenance, clerk_note,
                    model_name, duplicate_probability, timestamp, status, undone_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    merge_id,
                    r1_id,
                    r2_id,
                    r1_json,
                    r2_json,
                    golden_json,
                    prov_json,
                    clerk_note,
                    model_name,
                    float(duplicate_probability),
                    now_iso,
                    "merged",
                    None,
                ),
            )
            conn.commit()

        return merge_id

    def undo_merge(self, merge_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restores both original records EXACTLY byte-for-byte from saved snapshots.
        
        Returns:
            (record_1, record_2) as original dicts.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM merges WHERE merge_id = ?", (merge_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError(f"Merge ID '{merge_id}' not found in store.")

            r1_dict = json.loads(row["record_1_snapshot"])
            r2_dict = json.loads(row["record_2_snapshot"])

            # Idempotently mark status as undone if not already undone
            if row["status"] != "undone":
                now_iso = datetime.now().isoformat()
                cursor.execute(
                    "UPDATE merges SET status = ?, undone_timestamp = ? WHERE merge_id = ?",
                    ("undone", now_iso, merge_id),
                )
                conn.commit()

        return r1_dict, r2_dict

    def record_decision(
        self,
        record_1: dict[str, Any],
        record_2: dict[str, Any],
        decision: str,
        clerk_note: str = "",
        model_name: str = "",
        duplicate_probability: float = 0.0,
    ) -> str:
        """Stores a clerical review decision ('reject' or 'defer') with full timestamp.
        
        Returns:
            decision_id (str)
        """
        if decision not in ["reject", "defer"]:
            raise ValueError(f"Decision must be 'reject' or 'defer', got: '{decision}'")

        r1_id = str(record_1.get("record_id", "REC1"))
        r2_id = str(record_2.get("record_id", "REC2"))
        decision_id = f"DEC-{decision.upper()[:3]}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now().isoformat()

        r1_json = json.dumps(record_1, ensure_ascii=False, sort_keys=True)
        r2_json = json.dumps(record_2, ensure_ascii=False, sort_keys=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO decisions (
                    decision_id, record_id_1, record_id_2, record_1_snapshot,
                    record_2_snapshot, decision, clerk_note, model_name,
                    duplicate_probability, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    r1_id,
                    r2_id,
                    r1_json,
                    r2_json,
                    decision,
                    clerk_note,
                    model_name,
                    float(duplicate_probability),
                    now_iso,
                ),
            )
            conn.commit()

        return decision_id

    def get_merge(self, merge_id: str) -> dict[str, Any] | None:
        """Fetches full merge record by merge_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM merges WHERE merge_id = ?", (merge_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["record_1_snapshot"] = json.loads(res["record_1_snapshot"])
            res["record_2_snapshot"] = json.loads(res["record_2_snapshot"])
            res["golden_record"] = json.loads(res["golden_record"])
            res["field_provenance"] = json.loads(res["field_provenance"])
            return res

    def get_all_merges(self, status: str | None = None) -> list[dict[str, Any]]:
        """Lists all merges sorted chronologically descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status is not None:
                cursor.execute("SELECT * FROM merges WHERE status = ? ORDER BY timestamp DESC", (status,))
            else:
                cursor.execute("SELECT * FROM merges ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["record_1_snapshot"] = json.loads(item["record_1_snapshot"])
                item["record_2_snapshot"] = json.loads(item["record_2_snapshot"])
                item["golden_record"] = json.loads(item["golden_record"])
                item["field_provenance"] = json.loads(item["field_provenance"])
                results.append(item)
            return results

    def get_all_decisions(self) -> list[dict[str, Any]]:
        """Lists all review decisions (reject / defer) sorted chronologically descending."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM decisions ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["record_1_snapshot"] = json.loads(item["record_1_snapshot"])
                item["record_2_snapshot"] = json.loads(item["record_2_snapshot"])
                results.append(item)
            return results

    def get_reviewed_pair_ids(self) -> set[tuple[str, str]]:
        """Returns set of canonical pair IDs (min, max) that have an active merge, reject, or deferral."""
        reviewed: set[tuple[str, str]] = set()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Active merges (not undone)
            cursor.execute("SELECT record_id_1, record_id_2 FROM merges WHERE status = 'merged'")
            for r1, r2 in cursor.fetchall():
                reviewed.add((min(str(r1), str(r2)), max(str(r1), str(r2))))

            # Decisions (reject or defer)
            cursor.execute("SELECT record_id_1, record_id_2 FROM decisions")
            for r1, r2 in cursor.fetchall():
                reviewed.add((min(str(r1), str(r2)), max(str(r1), str(r2))))

        return reviewed

    def export_decisions_as_labelled_pairs(self) -> pd.DataFrame:
        """Exports all merges and decisions as labelled pairs for retraining or audit."""
        rows: list[dict[str, Any]] = []

        # Merged pairs -> Label 1
        for m in self.get_all_merges(status="merged"):
            rows.append({
                "record_id_1": m["record_id_1"],
                "record_id_2": m["record_id_2"],
                "label": 1,
                "decision_type": "merge",
                "timestamp": m["timestamp"],
                "clerk_note": m.get("clerk_note", ""),
                "duplicate_probability": m.get("duplicate_probability", 0.0),
                "model_name": m.get("model_name", ""),
            })

        # Decisions (reject -> Label 0, defer -> Label -1)
        for d in self.get_all_decisions():
            lbl = 0 if d["decision"] == "reject" else -1
            rows.append({
                "record_id_1": d["record_id_1"],
                "record_id_2": d["record_id_2"],
                "label": lbl,
                "decision_type": d["decision"],
                "timestamp": d["timestamp"],
                "clerk_note": d.get("clerk_note", ""),
                "duplicate_probability": d.get("duplicate_probability", 0.0),
                "model_name": d.get("model_name", ""),
            })

        return pd.DataFrame(rows)


# ── Golden Record Survivorship Rules Engine ────────────────────────────────────

SENTINEL_VALUES: set[str] = {
    "", "1900-01-01", "1970-01-01", "2099-12-31", "0000-00-00",
    "00000000000", "-999", "-1", "999", "9999", "199", "none", "null", "nan"
}


def compute_survivorship_suggestions(
    rec_a: dict[str, Any],
    rec_b: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Generates rule-based golden record suggestions following Lecture 2 & Roadmap D8.
    
    Survivorship Heuristics:
    1. Completeness: Prefer non-missing, non-sentinel values.
    2. Recency: For contact details (phone, address, city), prefer the record with
       the most recent registration_date.
    3. Specificity: For names and identifiers, prefer the longer, fully formatted string.
    4. Clinical Encounters: For visit_count, prefer the maximum or non-zero value.
    5. Fallback: Default to Record A (primary incoming chart).
    
    Returns:
        (suggested_golden, default_choices, rationale_map)
        - default_choices: dict mapping field_name to 'Record A' or 'Record B'
    """
    fields = [
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

    reg_a = str(rec_a.get("registration_date", ""))
    reg_b = str(rec_b.get("registration_date", ""))
    # Recency check: Is Record B newer than Record A?
    b_is_more_recent = bool(reg_b and reg_b > reg_a)

    suggested_golden: dict[str, Any] = {
        "record_id": f"GOLD_{rec_a.get('record_id', 'A')}_{rec_b.get('record_id', 'B')}",
        "source_system": "Golden Patient Master Index (Merged)",
        "registration_date": max(reg_a, reg_b) if (reg_a or reg_b) else datetime.now().strftime("%Y-%m-%d"),
    }
    default_choices: dict[str, str] = {}
    rationale_map: dict[str, str] = {}

    for f in fields:
        val_a = str(rec_a.get(f, "") or "").strip()
        val_b = str(rec_b.get(f, "") or "").strip()

        is_a_sentinel = (val_a.lower() in SENTINEL_VALUES)
        is_b_sentinel = (val_b.lower() in SENTINEL_VALUES)

        # 1. One is sentinel/missing, the other is valid
        if is_a_sentinel and not is_b_sentinel:
            default_choices[f] = "Record B"
            suggested_golden[f] = val_b
            rationale_map[f] = "Record A was missing/sentinel; chose valid Record B."
            continue

        if is_b_sentinel and not is_a_sentinel:
            default_choices[f] = "Record A"
            suggested_golden[f] = val_a
            rationale_map[f] = "Record B was missing/sentinel; chose valid Record A."
            continue

        if is_a_sentinel and is_b_sentinel:
            default_choices[f] = "Record A"
            suggested_golden[f] = val_a
            rationale_map[f] = "Both records unrecorded/sentinel."
            continue

        # 2. Both valid: field-specific survivorship rules
        if f in ["phone", "address", "city"]:
            # Contact details: prefer most recent registration date
            if b_is_more_recent:
                default_choices[f] = "Record B"
                suggested_golden[f] = val_b
                rationale_map[f] = f"Record B has more recent registration ({reg_b} vs {reg_a})."
            else:
                default_choices[f] = "Record A"
                suggested_golden[f] = val_a
                rationale_map[f] = f"Record A has more recent or equal registration ({reg_a} vs {reg_b})."

        elif f in ["national_id"]:
            # Prefer 13-digit standard CNIC or longer value
            clean_nid_a = "".join(c for c in val_a if c.isdigit())
            clean_nid_b = "".join(c for c in val_b if c.isdigit())
            if len(clean_nid_b) == 13 and len(clean_nid_a) != 13:
                default_choices[f] = "Record B"
                suggested_golden[f] = val_b
                rationale_map[f] = "Record B has a complete 13-digit CNIC."
            elif len(clean_nid_a) == 13 and len(clean_nid_b) != 13:
                default_choices[f] = "Record A"
                suggested_golden[f] = val_a
                rationale_map[f] = "Record A has a complete 13-digit CNIC."
            elif val_a != val_b:
                default_choices[f] = "Record A"
                suggested_golden[f] = val_a
                rationale_map[f] = "Conflicting CNICs detected! Manual verification required."
            else:
                default_choices[f] = "Record A"
                suggested_golden[f] = val_a
                rationale_map[f] = "Both records agree on CNIC."

        elif f in ["first_name", "family_name", "father_name"]:
            # Prefer longer, more complete title-cased name
            if len(val_b) > len(val_a):
                default_choices[f] = "Record B"
                suggested_golden[f] = val_b
                rationale_map[f] = "Record B provides a more complete name."
            else:
                default_choices[f] = "Record A"
                suggested_golden[f] = val_a
                rationale_map[f] = "Record A provides equal or more complete name."

        elif f == "visit_count":
            # Encounters: take maximum observed visits
            try:
                cnt_a = int(val_a)
                cnt_b = int(val_b)
                if cnt_b > cnt_a:
                    default_choices[f] = "Record B"
                    suggested_golden[f] = str(cnt_b)
                    rationale_map[f] = f"Record B records higher cumulative encounter count ({cnt_b})."
                else:
                    default_choices[f] = "Record A"
                    suggested_golden[f] = str(cnt_a)
                    rationale_map[f] = f"Record A records equal or higher encounter count ({cnt_a})."
            except ValueError:
                default_choices[f] = "Record A"
                suggested_golden[f] = val_a
                rationale_map[f] = "Retained Record A encounter count."

        else:
            # Default to Record A
            default_choices[f] = "Record A"
            suggested_golden[f] = val_a
            rationale_map[f] = "Attributes agree or defaulted to primary chart."

    return suggested_golden, default_choices, rationale_map
