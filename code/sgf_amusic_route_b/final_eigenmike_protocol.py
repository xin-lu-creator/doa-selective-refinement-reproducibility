# =========================================================
# File        : final_eigenmike_protocol.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the final eigenmike protocol module used by the reproducibility workflow.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - In-memory estimates, state objects, or helper values returned to calling code.
#
# Used in paper:
#   - RA-STR method construction, frozen evidence generation, or supporting analysis.
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
"""Final one-shot frozen cross-array validation protocol for R8.

The validation domain is the LOCATA final evaluation split recorded with the
32-channel Eigenmike.  A deterministic 12-channel near-equatorial sub-array is
selected from microphone geometry only.  No audio, source truth, baseline
error, or R8 output participates in channel selection.

This module contains only protocol and evaluation utilities.  It does not
modify the frozen R8 estimator or any of its parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

RELEASE = "r8-final-frozen-eigenmike-evaluation-20260806"
PROTOCOL_ID = "DOA-R8-FINAL-EIGENMIKE-EVAL-20260806"
ARRAY_NAME = "eigenmike"
SUBARRAY_CHANNEL_COUNT = 12
WINDOW_DURATION_S = 0.75
MAX_SELECTED_PER_RECORDING = 20
MIN_SELECTED_PER_RECORDING = 12
MIN_RECORDING_COUNT = 8
MIN_TOTAL_SELECTED = 96
REQUIRED_TASKS = ("task1", "task2", "task3", "task4")
MAXIMUM_OTHER_SOURCE_ACTIVITY_FRACTION = 1.0 / 15.0
THETA_MIN_DEG = -80.0
THETA_MAX_DEG = 80.0
PAIR_MIN_BASELINE_M = 0.05
MIN_RETAINED_PAIR_COUNT = 18

# Final TASLP go/no-go gates.  There is no intermediate "promising" outcome.
MIN_PRIMARY_RMSE_GAIN_PCT = 5.0
MAX_BOOTSTRAP_CI95_HIGH_DEG = 0.0
MIN_NONDEGRADED_RECORDING_FRACTION = 0.70
MAX_CATEGORY_DEGRADATION_PCT = 2.0
MAX_NEW_GT45_ERRORS = 0
REQUIRE_NEW_GT10_NOT_EXCEED_REMOVED = True
REQUIRE_P95_NOT_INCREASED = True
BOOTSTRAP_DRAWS = 50000
BOOTSTRAP_SEED = 20260806

FROZEN_R8_CORE_RELATIVE_PATH = "code/sgf_amusic_route_b/causal_multiscale_hodge.py"
FROZEN_R8_CORE_SHA256 = "33953738b1eebaa9764b657f38a9b8774d854a5fed4da4d122dc1fa62d5d2d70"


@dataclass(frozen=True)
class SubarraySelection:
    selected_indices: tuple[int, ...]
    selected_sector_indices: tuple[int, ...]
    selected_azimuth_deg: tuple[float, ...]
    selected_elevation_deg: tuple[float, ...]
    retained_pair_count: int
    horizontal_aperture_m: float
    minimum_selected_radius_m: float
    maximum_selected_abs_elevation_deg: float

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_r8_core(project_root: Path) -> dict:
    path = Path(project_root) / FROZEN_R8_CORE_RELATIVE_PATH
    actual = sha256_file(path) if path.is_file() else "MISSING"
    return {
        "path": str(path),
        "expected_sha256": FROZEN_R8_CORE_SHA256,
        "actual_sha256": actual,
        "status": "PASS" if actual == FROZEN_R8_CORE_SHA256 else "FAIL",
    }


def wrap_deg(angle: np.ndarray | float) -> np.ndarray:
    x = np.asarray(angle, dtype=float)
    return (x + 180.0) % 360.0 - 180.0


def _microphone_angles(positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(positions_m, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("Eigenmike positions must have shape M x 3")
    centered = pos - np.mean(pos, axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1)
    if np.any(radius <= 1.0e-9):
        raise ValueError("Degenerate Eigenmike microphone geometry")
    azimuth = np.rad2deg(np.arctan2(centered[:, 1], centered[:, 0]))
    elevation = np.rad2deg(np.arcsin(np.clip(centered[:, 2] / radius, -1.0, 1.0)))
    return radius, azimuth, elevation


def select_fixed_eigenmike_subarray(
    microphone_positions_m: np.ndarray,
    *,
    channel_count: int = SUBARRAY_CHANNEL_COUNT,
    pair_min_baseline_m: float = PAIR_MIN_BASELINE_M,
) -> SubarraySelection:
    """Select an azimuth-uniform near-equatorial sub-array from geometry only.

    The assignment is global and deterministic.  Twelve target azimuth sectors
    are matched one-to-one to physical microphones by a Hungarian assignment.
    The cost penalizes sector mismatch and elevation magnitude.  Channel index
    is used only as a deterministic infinitesimal tie-breaker.
    """
    pos = np.asarray(microphone_positions_m, dtype=float)
    if len(pos) < int(channel_count):
        raise ValueError(f"Need at least {channel_count} microphones, got {len(pos)}")
    radius, azimuth, elevation = _microphone_angles(pos)
    targets = -180.0 + (np.arange(int(channel_count), dtype=float) + 0.5) * (
        360.0 / float(channel_count)
    )
    angular = np.abs(wrap_deg(azimuth[None, :] - targets[:, None])) / 180.0
    elevation_penalty = np.abs(elevation[None, :]) / 90.0
    radius_penalty = (np.max(radius) - radius[None, :]) / max(np.max(radius), 1.0e-12)
    tie = np.arange(len(pos), dtype=float)[None, :] * 1.0e-10
    cost = angular * angular + 2.0 * elevation_penalty * elevation_penalty + 0.05 * radius_penalty + tie
    rows, cols = linear_sum_assignment(cost)
    order = np.argsort(rows)
    selected = np.asarray(cols[order], dtype=int)
    sectors = np.asarray(rows[order], dtype=int)

    chosen = pos[selected]
    horizontal = chosen[:, :2]
    D = np.linalg.norm(horizontal[:, None, :] - horizontal[None, :, :], axis=-1)
    horizontal_aperture = float(np.max(D)) if len(chosen) else 0.0
    D3 = np.linalg.norm(chosen[:, None, :] - chosen[None, :, :], axis=-1)
    retained_pairs = int(np.sum(np.triu(D3 >= float(pair_min_baseline_m), 1)))
    return SubarraySelection(
        selected_indices=tuple(int(i) for i in selected),
        selected_sector_indices=tuple(int(i) for i in sectors),
        selected_azimuth_deg=tuple(float(azimuth[i]) for i in selected),
        selected_elevation_deg=tuple(float(elevation[i]) for i in selected),
        retained_pair_count=retained_pairs,
        horizontal_aperture_m=horizontal_aperture,
        minimum_selected_radius_m=float(np.min(radius[selected])),
        maximum_selected_abs_elevation_deg=float(np.max(np.abs(elevation[selected]))),
    )


def same_channel_set(actual: Sequence[int], expected: Sequence[int]) -> bool:
    """Return True when two channel selections contain the same unique IDs.

    Eigenmike geometry files can differ at floating-point roundoff level across
    recordings.  When two microphones have effectively tied assignment costs,
    the Hungarian solver may swap their sector order even though the selected
    physical 12-channel set is unchanged.  Scientific freezing concerns the
    physical channel set; estimator input order is normalized separately to the
    first frozen order.
    """
    actual_tuple = tuple(int(x) for x in actual)
    expected_tuple = tuple(int(x) for x in expected)
    return (
        len(actual_tuple) == len(expected_tuple)
        and len(set(actual_tuple)) == len(actual_tuple)
        and len(set(expected_tuple)) == len(expected_tuple)
        and set(actual_tuple) == set(expected_tuple)
    )


def subarray_preflight_pass(selection: SubarraySelection) -> tuple[bool, dict]:
    gates = {
        "exactly_12_unique_channels": len(set(selection.selected_indices)) == SUBARRAY_CHANNEL_COUNT,
        "retained_pair_count_ge_18": selection.retained_pair_count >= MIN_RETAINED_PAIR_COUNT,
        "horizontal_aperture_ge_70mm": selection.horizontal_aperture_m >= 0.070,
        "maximum_abs_elevation_le_55deg": selection.maximum_selected_abs_elevation_deg <= 55.0,
    }
    return all(gates.values()), gates


def deterministic_time_spread_subset(
    eligible_indices: Sequence[int],
    timestamps_abs_s: Sequence[float],
    rows: Sequence[int],
    *,
    maximum_count: int = MAX_SELECTED_PER_RECORDING,
) -> tuple[list[int], dict]:
    ordered = sorted(
        (int(i) for i in eligible_indices),
        key=lambda i: (float(timestamps_abs_s[i]), int(rows[i])),
    )
    if len(ordered) > int(maximum_count):
        positions = np.unique(
            np.round(np.linspace(0, len(ordered) - 1, int(maximum_count))).astype(int)
        )
        selected = [ordered[int(p)] for p in positions]
    else:
        selected = ordered
    times = np.asarray([float(timestamps_abs_s[i]) for i in selected], dtype=float)
    spacings = np.diff(times) if len(times) > 1 else np.asarray([], dtype=float)
    return selected, {
        "eligible_windows": int(len(ordered)),
        "selected_windows": int(len(selected)),
        "maximum_selected": int(maximum_count),
        "selection_rule": "uniform_time_spread_among_preregistered_eligible_anchors",
        "minimum_selected_anchor_spacing_s": float(np.min(spacings)) if len(spacings) else float("nan"),
        "maximum_adjacent_acoustic_overlap_fraction": (
            float(np.max(np.maximum(0.0, 1.0 - spacings / WINDOW_DURATION_S)))
            if len(spacings) else float("nan")
        ),
        "window_independence_claimed": False,
        "inferential_unit": "recording",
        "truth_or_error_used_for_ranking": False,
    }


def canonical_manifest_bytes(rows: Iterable[dict]) -> bytes:
    keep = []
    for row in rows:
        keep.append({
            "task": str(row["task"]),
            "recording_name": str(row["recording_name"]),
            "required_time_row_index": int(row["required_time_row_index"]),
            "active_source_index": int(row["active_source_index"]),
        })
    keep.sort(key=lambda x: (
        x["task"], x["recording_name"], x["required_time_row_index"], x["active_source_index"]
    ))
    return (json.dumps(keep, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_manifest_sha256(rows: Iterable[dict]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(rows)).hexdigest()


def coverage_gate(status_rows: Sequence[dict], total_selected: int) -> tuple[bool, dict]:
    sufficient = [row for row in status_rows if bool(row.get("coverage_sufficient", False))]
    tasks = sorted({str(row.get("task")) for row in sufficient})
    task_counts = {task: sum(str(row.get("task")) == task for row in sufficient) for task in REQUIRED_TASKS}
    gates = {
        "at_least_8_recordings": len(sufficient) >= MIN_RECORDING_COUNT,
        "all_four_tasks_represented": all(task in tasks for task in REQUIRED_TASKS),
        "at_least_96_selected_windows": int(total_selected) >= MIN_TOTAL_SELECTED,
        "at_least_two_task1_recordings": task_counts.get("task1", 0) >= 2,
        "at_least_two_task3_recordings": task_counts.get("task3", 0) >= 2,
    }
    return all(gates.values()), {
        "gates": gates,
        "sufficient_recording_count": int(len(sufficient)),
        "represented_tasks": tasks,
        "task_recording_counts": task_counts,
        "total_selected_windows": int(total_selected),
    }


def classify_preflight_outcome(
    *, coverage_pass: bool, recording_error_count: int
) -> tuple[str, int, bool]:
    """Classify preflight without converting engineering failures to science.

    Returns ``(status, exit_code, scientific_final_decision)``.  Recording or
    geometry processing exceptions are engineering errors and must never write
    the binary TASLP decision.  Only a completed, error-free preflight whose
    preregistered coverage gates fail is a scientific preflight NO-GO.
    """
    if int(recording_error_count) > 0:
        return "FINAL_VALIDATION_PREFLIGHT_ENGINEERING_ERROR_NO_DECISION", 2, False
    if bool(coverage_pass):
        return "FINAL_VALIDATION_PREFLIGHT_PASS_RUN_AUTHORIZED", 0, False
    return "FINAL_VALIDATION_PREFLIGHT_NO_GO_ABANDON_TASLP", 3, True


def minimum_nondegraded_recordings(recording_count: int) -> int:
    return int(ceil(MIN_NONDEGRADED_RECORDING_FRACTION * int(recording_count) - 1.0e-12))
