# =========================================================
# File        : b0_r2_protocol.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the b0 r2 protocol module used by the reproducibility workflow.
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
from __future__ import annotations

"""Frozen deterministic inclusion and geometry audits for V52B B0-R2.

This module intentionally contains no DOA estimator and no truth-fitted correction.
It only defines the preregistered technical-silence gate, deterministic sampling,
and checks that the LOCATA scan geometry is derived from the released local
microphone coordinates by the already-frozen x-axis reflection only.
"""

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

HARD_SILENCE_FLOOR_DBFS = -80.0
HARD_SILENCE_FLOOR_RMS = 10.0 ** (HARD_SILENCE_FLOOR_DBFS / 20.0)
DEFAULT_CANDIDATE_POOL_WINDOWS = 32
DEFAULT_SELECTED_WINDOWS_PER_RECORDING = 8
MINIMUM_VALID_WINDOWS = 40
MINIMUM_RECORDINGS = 6
MINIMUM_WINDOWS_PER_RECORDING = 5
REQUIRED_TASKS = ("task1", "task3")


@dataclass(frozen=True)
class WindowGateDecision:
    frame_rms: float
    frame_rms_dbfs: float
    official_activity_available: bool
    exactly_one_source_active: bool
    energy_pass: bool
    eligible: bool
    active_source_index: int | None
    rejection_reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def frame_rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def amplitude_to_dbfs(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        return float("nan")
    if value == 0.0:
        return float("-inf")
    return float(20.0 * np.log10(value))


def preregistered_window_gate(
    frame: dict,
    *,
    hard_silence_floor_dbfs: float = HARD_SILENCE_FLOOR_DBFS,
) -> WindowGateDecision:
    """Apply the frozen metadata/activity and technical-silence gate.

    The gate never examines a DOA estimate or angular error.  Official LOCATA
    array-aligned VAD is required and exactly one source must be active.
    """
    rms = frame_rms(frame.get("audio", np.asarray([], dtype=float)))
    dbfs = amplitude_to_dbfs(rms)
    source_kind = str(frame.get("source_activity_source", ""))
    activity = frame.get("source_active")
    official = source_kind == "official_array_aligned_vad" and activity is not None
    active_index: int | None = None
    exactly_one = False
    if official:
        active = np.asarray(activity, dtype=bool).reshape(-1)
        ids = np.flatnonzero(active)
        exactly_one = len(ids) == 1
        if exactly_one:
            active_index = int(ids[0])
    energy_pass = bool(np.isfinite(rms) and rms > 10.0 ** (float(hard_silence_floor_dbfs) / 20.0))

    reasons: list[str] = []
    if not official:
        reasons.append("official_activity_unavailable")
    elif not exactly_one:
        reasons.append("not_exactly_one_active_source")
    if not np.isfinite(rms):
        reasons.append("nonfinite_frame_rms")
    elif not energy_pass:
        reasons.append("below_or_equal_hard_silence_floor")
    eligible = not reasons
    return WindowGateDecision(
        frame_rms=float(rms),
        frame_rms_dbfs=float(dbfs),
        official_activity_available=bool(official),
        exactly_one_source_active=bool(exactly_one),
        energy_pass=bool(energy_pass),
        eligible=bool(eligible),
        active_source_index=active_index,
        rejection_reason="accepted" if eligible else ";".join(reasons),
    )


def deterministic_uniform_subset(
    timestamps_s: Iterable[float],
    *,
    maximum_count: int = DEFAULT_SELECTED_WINDOWS_PER_RECORDING,
    minimum_spacing_s: float = 3.0,
) -> tuple[list[int], list[dict]]:
    """Select a deterministic, time-spread subset without using acoustic errors.

    Input order is preserved only through a stable timestamp sort. Duplicate
    timestamps are removed. A greedy minimum-spacing pass is followed by a
    uniform cap over the retained timeline so the first part of a recording is
    not overrepresented.
    """
    values = np.asarray(list(timestamps_s), dtype=float)
    if values.size == 0:
        return [], []
    order = np.argsort(values, kind="mergesort")
    unique: list[int] = []
    last_value: float | None = None
    for raw_index in order:
        value = float(values[int(raw_index)])
        if last_value is not None and abs(value - last_value) <= 1.0e-9:
            continue
        unique.append(int(raw_index))
        last_value = value

    spaced: list[int] = []
    last_selected: float | None = None
    for raw_index in unique:
        value = float(values[raw_index])
        if last_selected is None or value - last_selected >= float(minimum_spacing_s) - 1.0e-9:
            spaced.append(raw_index)
            last_selected = value

    # If a short recording cannot satisfy the spacing rule, preserve coverage
    # by using the unique official-active candidates rather than silently
    # lowering the preregistered minimum-windows requirement.
    pool = spaced if len(spaced) >= min(int(maximum_count), len(unique)) else unique
    if int(maximum_count) > 0 and len(pool) > int(maximum_count):
        positions = np.linspace(0, len(pool) - 1, int(maximum_count))
        ids = np.unique(np.round(positions).astype(int))
        pool = [pool[int(i)] for i in ids]

    audit = [
        {
            "selected_index": int(index),
            "timestamp_abs_s": float(values[index]),
            "selection_rank": int(rank),
            "minimum_spacing_s": float(minimum_spacing_s),
            "maximum_count": int(maximum_count),
            "selection_rule": "stable_unique_greedy_spacing_then_uniform_cap",
        }
        for rank, index in enumerate(pool)
    ]
    return pool, audit


def geometry_reference_audit(frame: dict, *, atol: float = 1.0e-12) -> dict:
    """Verify that no centroid or test-fitted translation was introduced."""
    raw = np.asarray(frame.get("microphone_positions_raw_local_m"), dtype=float)
    scan = np.asarray(frame.get("microphone_positions_m"), dtype=float)
    expected = raw.copy()
    valid_shape = bool(raw.ndim == 2 and raw.shape == scan.shape and raw.shape[1] == 3)
    if valid_shape:
        expected[:, 0] *= -1.0
        transform_error = float(np.max(np.abs(scan - expected))) if raw.size else 0.0
        raw_distances = np.linalg.norm(raw[:, None, :] - raw[None, :, :], axis=-1)
        scan_distances = np.linalg.norm(scan[:, None, :] - scan[None, :, :], axis=-1)
        distance_error = float(np.max(np.abs(raw_distances - scan_distances))) if raw.size else 0.0
        finite = bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(scan)))
    else:
        transform_error = float("inf")
        distance_error = float("inf")
        finite = False
    passed = bool(valid_shape and finite and transform_error <= atol and distance_error <= atol)
    centroid = np.mean(raw, axis=0) if valid_shape and len(raw) else np.full(3, np.nan)
    return {
        "status": "PASS" if passed else "FAIL",
        "reference_point_policy": "official_LOCATA_array_pose_plus_released_local_microphone_coordinates",
        "manual_reference_translation_applied": False,
        "truth_fitted_translation_applied": False,
        "microphone_centroid_recentered": False,
        "scan_transform": "local_x_reflection_only_for_LOCATA_azimuth_sign",
        "valid_shape": valid_shape,
        "finite_coordinates": finite,
        "maximum_transform_error_m": transform_error,
        "maximum_pairwise_distance_error_m": distance_error,
        "raw_microphone_centroid_x_m": float(centroid[0]),
        "raw_microphone_centroid_y_m": float(centroid[1]),
        "raw_microphone_centroid_z_m": float(centroid[2]),
    }
