# =========================================================
# File        : frozen_hearingaid_holdout_protocol.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the frozen hearingaid holdout protocol module used by the reproducibility workflow.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - In-memory estimates, state objects, or helper values returned to calling code.
#
# Used in paper:
#   - Hearing-Aid fallback-safety extension reported in the manuscript and Supplementary Material.
#
# Main parameters:
#   - Frozen manuscript parameters and manifests; see README.md and documentation/CODE_GUIDE.md.
#
# Software:
#   - Python 3.x; dependencies listed in requirements.txt/environment.yml.
#
# Author      : Xin Lu
# Last update : 2026-09-25
# =========================================================
"""Frozen one-shot LOCATA hearing-aid holdout protocol for final R8-M.

This module freezes the winning development estimator before looking at the
hearing-aid holdout errors.  The holdout is the fourth LOCATA sensor
configuration and uses only Task 3/4, where the microphone platform is static.
No holdout result may alter the estimator, lambda, anchor, guard, or gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

RELEASE = "r8m-frozen-hearingaid-holdout-20260810"
PROTOCOL_ID = "DOA-R8M-FROZEN-HEARINGAID-HOLDOUT-20260810"
FINAL_METHOD = "full_adaptive_freedom"
FROZEN_ASSIGNMENT_STRENGTH = 1.5
FROZEN_ANCHOR_STRENGTH = 0.03
REQUIRED_TASKS = ("task3", "task4")
EXPECTED_CHANNEL_COUNT = 4
WINDOW_DURATION_S = 0.75
MAX_SELECTED_PER_RECORDING = 20
MIN_SELECTED_PER_RECORDING = 12
MIN_RECORDING_COUNT = 6
MIN_TOTAL_SELECTED = 72
MAXIMUM_OTHER_SOURCE_ACTIVITY_FRACTION = 1.0 / 15.0
THETA_MIN_DEG = -80.0
THETA_MAX_DEG = 80.0

# Holdout gates.  These are fixed before reading holdout errors.
MIN_OVERALL_GAIN_PCT = 0.0
MIN_TASK3_GAIN_PCT = 0.0
MIN_TASK4_GAIN_PCT = -2.0
MIN_NONDEGRADED_RECORDING_FRACTION = 0.70
MAX_NEW_GT45_ERRORS = 0
REQUIRE_NEW_GT10_NOT_EXCEED_REMOVED = True
BOOTSTRAP_DRAWS = 50000
BOOTSTRAP_SEED = 20260810

FROZEN_HASHES = {
    "code/sgf_amusic_route_b/causal_multiscale_hodge.py": "33953738b1eebaa9764b657f38a9b8774d854a5fed4da4d122dc1fa62d5d2d70",
    "code/sgf_amusic_route_b/mechanism_corrected_r8.py": "e1bbd03b892eb5e48fe93ced5987c0375af51e4fb95dbc86c6fdacd484ba8fc4",
    "code/sgf_amusic_route_b/mechanism_corrected_r8_structural_funnel.py": "5f30e9061598a21a102df8859e691e31e5a3ca228e4d5403ba87204e978d285f",
    "code/sgf_amusic_route_b/mechanism_corrected_r8_evidence_change.py": "c070109d52a9a03a319e09155ad2cc4197238aa01d1c6244e0351912e5a5e75f",
    "code/sgf_amusic_route_b/mechanism_corrected_r8_multimethod_evidence_change.py": "87e050400212f79145c32200f4b39eb3ee9823f87b5987ded1c9ee77f837ddf8",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_hashes(project_root: Path) -> dict:
    rows = []
    for rel, expected in FROZEN_HASHES.items():
        p = Path(project_root) / rel
        actual = sha256_file(p) if p.is_file() else "MISSING"
        rows.append({"path": rel, "expected_sha256": expected, "actual_sha256": actual,
                     "status": "PASS" if actual == expected else "FAIL"})
    return {"status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL", "files": rows}


def canonical_manifest_sha256(rows: Iterable[dict]) -> str:
    keep = [{
        "task": str(r["task"]),
        "recording_name": str(r["recording_name"]),
        "required_time_row_index": int(r["required_time_row_index"]),
        "active_source_index": int(r["active_source_index"]),
    } for r in rows]
    keep.sort(key=lambda x: (x["task"], x["recording_name"], x["required_time_row_index"], x["active_source_index"]))
    text = json.dumps(keep, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_time_spread_subset(eligible_indices: Sequence[int], timestamps: Sequence[float], row_ids: Sequence[int], maximum_count: int = MAX_SELECTED_PER_RECORDING):
    ordered = sorted((int(i) for i in eligible_indices), key=lambda i: (float(timestamps[i]), int(row_ids[i])))
    if len(ordered) > int(maximum_count):
        pos = np.unique(np.round(np.linspace(0, len(ordered)-1, int(maximum_count))).astype(int))
        selected = [ordered[int(p)] for p in pos]
    else:
        selected = ordered
    return selected
