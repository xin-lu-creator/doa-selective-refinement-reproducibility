# =========================================================
# File        : active_speech_gate.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the active speech gate module used by the reproducibility workflow.
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
"""Truth-free active-speech qualification for V52E-R5.

The gate addresses a protocol defect identified by V52D-R4: a single official
VAD sample at the causal window anchor and a broadband RMS floor do not ensure
that the 0.75 s analysis window contains enough target speech in the actual
350--3500 Hz localization band.  This module never reads a DOA estimate or
angular error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

_EPS = 1.0e-15


@dataclass(frozen=True)
class ActiveSpeechGateParameters:
    target_activity_duration_min_s: float = 0.10
    strict_activity_fraction: float = 0.50
    sensitivity_activity_fraction: float = 0.80
    localization_band_min_hz: float = 350.0
    localization_band_max_hz: float = 3500.0
    band_snr_margin_db: float = 6.0
    absolute_band_floor_dbfs: float = -80.0
    inactive_fraction_max: float = 0.05
    minimum_noise_windows: int = 8
    noise_grid_hop_s: float = 0.25
    lower_envelope_fraction: float = 0.20
    filter_order: int = 6

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActivityWindowStats:
    source_fractions: tuple[float, ...]
    target_fraction: float
    maximum_any_source_fraction: float
    maximum_other_source_fraction: float
    target_activity_samples: int
    target_activity_transitions: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActiveSpeechDecision:
    eligible: bool
    rejection_reason: str
    center_official_exactly_one_active: bool
    active_source_index: int | None
    target_activity_fraction: float
    target_activity_duration_s: float
    maximum_other_source_fraction: float
    band_rms: float
    band_rms_dbfs: float
    recording_noise_floor_dbfs: float
    band_snr_db: float
    target_activity_pass: bool
    band_floor_pass: bool
    band_snr_pass: bool
    relative_gate_required: bool
    strict_50pct_pass: bool
    sensitivity_80pct_pass: bool

    def as_dict(self) -> dict:
        return asdict(self)


def amplitude_to_dbfs(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        return float("nan")
    if value == 0.0:
        return float("-inf")
    return float(20.0 * np.log10(value))


def build_activity_index(geometry: dict) -> list[dict]:
    """Build cumulative official-VAD counts for O(log N) window queries."""
    series = geometry.get("source_activity")
    if series is None:
        return []
    result: list[dict] = []
    for item in series:
        times = np.asarray(item["t_abs_sec"], dtype=float).reshape(-1)
        active = np.asarray(item["active"], dtype=bool).reshape(-1)
        if len(times) != len(active) or len(times) == 0:
            raise ValueError("Malformed official source-activity series")
        if np.any(np.diff(times) < 0.0):
            order = np.argsort(times, kind="mergesort")
            times = times[order]
            active = active[order]
        cumulative = np.concatenate([[0], np.cumsum(active.astype(np.int64))])
        result.append({"times": times, "active": active, "cumulative": cumulative})
    return result


def activity_window_stats(
    activity_index: Sequence[dict],
    start_abs_s: float,
    end_abs_s: float,
    target_source_index: int | None,
) -> ActivityWindowStats:
    fractions: list[float] = []
    target_values = np.asarray([], dtype=bool)
    for source_index, item in enumerate(activity_index):
        times = np.asarray(item["times"], dtype=float)
        active = np.asarray(item["active"], dtype=bool)
        left = int(np.searchsorted(times, float(start_abs_s), side="left"))
        right = int(np.searchsorted(times, float(end_abs_s), side="right"))
        values = active[left:right]
        fraction = float(np.mean(values)) if len(values) else float("nan")
        fractions.append(fraction)
        if target_source_index is not None and source_index == int(target_source_index):
            target_values = values
    target_fraction = (
        float(np.mean(target_values)) if len(target_values) else float("nan")
    )
    finite = [value for value in fractions if np.isfinite(value)]
    maximum_any = max(finite) if finite else float("nan")
    other = [
        value for index, value in enumerate(fractions)
        if index != target_source_index and np.isfinite(value)
    ]
    maximum_other = max(other) if other else 0.0
    transitions = (
        int(np.sum(target_values[1:] != target_values[:-1]))
        if len(target_values) > 1 else 0
    )
    return ActivityWindowStats(
        source_fractions=tuple(float(value) for value in fractions),
        target_fraction=target_fraction,
        maximum_any_source_fraction=float(maximum_any),
        maximum_other_source_fraction=float(maximum_other),
        target_activity_samples=int(len(target_values)),
        target_activity_transitions=int(transitions),
    )


def resolve_shared_recording_audio(windows: Sequence[dict]) -> np.ndarray:
    """Recover the normalized full recording retained by window NumPy views."""
    if not windows:
        raise ValueError("No windows supplied")
    first = np.asarray(windows[0]["audio"], dtype=float)
    microphone_count = int(first.shape[0])
    required_samples = max(int(frame["audio_window_end_exclusive_sample"]) for frame in windows)
    candidate = windows[0]["audio"]
    seen: set[int] = set()
    while isinstance(candidate, np.ndarray) and id(candidate) not in seen:
        seen.add(id(candidate))
        array = np.asarray(candidate)
        options = []
        if array.ndim == 2 and array.shape[0] == microphone_count and array.shape[1] >= required_samples:
            options.append(array)
        if array.ndim == 2 and array.shape[1] == microphone_count and array.shape[0] >= required_samples:
            options.append(array.T)
        if array.ndim == 1 and array.size % microphone_count == 0:
            samples = array.size // microphone_count
            if samples >= required_samples:
                options.append(array.reshape(microphone_count, samples))
                options.append(array.reshape(samples, microphone_count).T)
        for option in options:
            start = int(windows[0]["audio_window_start_sample"])
            end = int(windows[0]["audio_window_end_exclusive_sample"])
            if option.shape[1] >= end and np.allclose(option[:, start:end], first, rtol=0.0, atol=0.0):
                return np.asarray(option, dtype=float)
        candidate = getattr(candidate, "base", None)
    raise RuntimeError("Could not recover shared full-recording audio from window views")


def band_energy_cumulative_from_audio(
    audio: np.ndarray,
    sample_rate_hz: float,
    *,
    frequency_min_hz: float = 350.0,
    frequency_max_hz: float = 3500.0,
    filter_order: int = 6,
) -> np.ndarray:
    """Return cumulative mean-square energy across microphones after band-pass.

    Only one length-N accumulator is retained, avoiding a full filtered copy for
    each microphone and keeping multi-process memory bounded.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 32:
        raise ValueError("audio must have shape (microphones, samples)")
    nyquist = 0.5 * float(sample_rate_hz)
    low = float(frequency_min_hz) / nyquist
    high = float(frequency_max_hz) / nyquist
    if not (0.0 < low < high < 1.0):
        raise ValueError("Invalid localization band")
    sos = butter(int(filter_order), [low, high], btype="bandpass", output="sos")
    sum_square = np.zeros(x.shape[1], dtype=np.float64)
    zi_base = sosfilt_zi(sos)
    for channel in x:
        values = np.asarray(channel, dtype=np.float64)
        zi = zi_base * float(values[0])
        filtered, _ = sosfilt(sos, values, zi=zi)
        sum_square += filtered * filtered
    mean_square = sum_square / float(x.shape[0])
    return np.concatenate([[0.0], np.cumsum(mean_square, dtype=np.float64)])


def window_band_rms(cumulative_mean_square: np.ndarray, start_sample: int, end_sample: int) -> float:
    cumulative = np.asarray(cumulative_mean_square, dtype=float)
    start = int(start_sample)
    end = int(end_sample)
    if start < 0 or end <= start or end >= len(cumulative):
        raise ValueError("Invalid window sample range")
    mean_square = float(cumulative[end] - cumulative[start]) / float(end - start)
    return float(np.sqrt(max(mean_square, 0.0)))


def estimate_inactive_noise_floor_dbfs(
    band_rms_dbfs: Sequence[float],
    maximum_activity_fractions: Sequence[float],
    *,
    inactive_fraction_max: float = 0.05,
    minimum_noise_windows: int = 8,
) -> tuple[float, dict]:
    levels = np.asarray(band_rms_dbfs, dtype=float)
    activities = np.asarray(maximum_activity_fractions, dtype=float)
    mask = (
        np.isfinite(levels)
        & np.isfinite(activities)
        & (activities <= float(inactive_fraction_max) + 1.0e-12)
    )
    selected = levels[mask]
    if len(selected) < int(minimum_noise_windows):
        raise RuntimeError(
            f"Only {len(selected)} inactive windows available for noise-floor estimation; "
            f"minimum is {minimum_noise_windows}"
        )
    floor = float(np.median(selected))
    mad = float(np.median(np.abs(selected - floor)))
    return floor, {
        "status": "PASS",
        "estimator": "median_band_rms_dbfs_over_official_inactive_windows",
        "inactive_fraction_max": float(inactive_fraction_max),
        "noise_windows": int(len(selected)),
        "noise_floor_dbfs": floor,
        "noise_floor_mad_db": mad,
        "noise_floor_p10_dbfs": float(np.quantile(selected, 0.10)),
        "noise_floor_p90_dbfs": float(np.quantile(selected, 0.90)),
    }



def estimate_full_recording_inactive_noise_floor_dbfs(
    cumulative_mean_square: np.ndarray,
    audio_t_abs_sec: Sequence[float],
    activity_index: Sequence[dict],
    sample_rate_hz: float,
    *,
    window_duration_s: float = 0.75,
    hop_duration_s: float = 0.25,
    inactive_fraction_max: float = 0.05,
    minimum_noise_windows: int = 8,
    lower_envelope_fraction: float = 0.20,
) -> tuple[float, dict, list[dict]]:
    """Estimate a truth-free recording-level 350--3500 Hz reference.

    Preferred mode uses the median level of at least ``minimum_noise_windows``
    fixed-grid windows that have complete official-VAD coverage and no source
    active for more than ``inactive_fraction_max`` of the window.  Some LOCATA
    recordings contain no such all-source-inactive intervals.  In that case a
    deterministic energy-only lower envelope is used: the median of the lowest
    ``lower_envelope_fraction`` of full-recording band-RMS windows, with at
    least ``minimum_noise_windows`` windows retained.  This fallback is not
    labelled an acoustic noise floor; it is a conservative band-level reference
    used only for the frozen 6 dB prominence gate.  It never reads DOA truth,
    estimates, angular errors, or required-time selections.
    """
    cumulative = np.asarray(cumulative_mean_square, dtype=float).reshape(-1)
    audio_times = np.asarray(audio_t_abs_sec, dtype=float).reshape(-1)
    sample_count = len(cumulative) - 1
    if sample_count < 1 or len(audio_times) < sample_count:
        raise ValueError("Band-energy accumulator and audio timestamps are inconsistent")
    audio_times = audio_times[:sample_count]
    if np.any(np.diff(audio_times) <= 0.0):
        raise ValueError("Audio timestamps must be strictly increasing")
    if not activity_index:
        raise ValueError("Official source-activity series are required")

    fs = float(sample_rate_hz)
    window_samples = max(1, int(round(float(window_duration_s) * fs)))
    hop_samples = max(1, int(round(float(hop_duration_s) * fs)))
    if window_samples >= sample_count:
        raise RuntimeError("Recording is shorter than the frozen reference-window duration")
    fraction = float(lower_envelope_fraction)
    if not (0.0 < fraction <= 0.5):
        raise ValueError("lower_envelope_fraction must be in (0, 0.5]")

    rows: list[dict] = []
    for candidate_index, start in enumerate(range(0, sample_count - window_samples + 1, hop_samples)):
        end = int(start + window_samples)
        start_abs = float(audio_times[start])
        end_abs = float(audio_times[end - 1])
        rms = window_band_rms(cumulative, start, end)
        level = amplitude_to_dbfs(rms)

        source_fractions: list[float] = []
        source_coverages: list[float] = []
        duration = max(end_abs - start_abs, 1.0 / fs)
        for item in activity_index:
            times = np.asarray(item["times"], dtype=float).reshape(-1)
            active = np.asarray(item["active"], dtype=bool).reshape(-1)
            left = int(np.searchsorted(times, start_abs, side="left"))
            right = int(np.searchsorted(times, end_abs, side="right"))
            values = active[left:right]
            source_fractions.append(float(np.mean(values)) if len(values) else float("nan"))
            if len(values):
                covered_start = max(start_abs, float(times[left]))
                covered_end = min(end_abs, float(times[right - 1]))
                source_coverages.append(max(0.0, covered_end - covered_start) / duration)
            else:
                source_coverages.append(0.0)

        finite_activity = [value for value in source_fractions if np.isfinite(value)]
        maximum_activity = max(finite_activity) if finite_activity else float("nan")
        minimum_coverage = min(source_coverages) if source_coverages else 0.0
        official_full_coverage = bool(minimum_coverage >= 0.95 - 1.0e-12)
        official_inactive = bool(
            official_full_coverage
            and np.isfinite(maximum_activity)
            and maximum_activity <= float(inactive_fraction_max) + 1.0e-12
        )
        rows.append({
            "noise_candidate_index": int(candidate_index),
            "audio_window_start_sample": int(start),
            "audio_window_end_exclusive_sample": int(end),
            "window_start_abs_s": start_abs,
            "window_end_abs_s": end_abs,
            "window_duration_s": float(window_duration_s),
            "grid_hop_s": float(hop_duration_s),
            "maximum_any_source_activity_fraction": float(maximum_activity),
            "minimum_official_vad_coverage_fraction": float(minimum_coverage),
            "official_vad_full_coverage_pass": official_full_coverage,
            "band_rms": float(rms),
            "band_rms_dbfs": float(level),
            "official_inactive_pass": official_inactive,
            "lower_envelope_selected": False,
            "reference_selected": False,
        })

    finite_rows = [row for row in rows if np.isfinite(float(row["band_rms_dbfs"]))]
    if len(finite_rows) < int(minimum_noise_windows):
        raise RuntimeError(
            f"Only {len(finite_rows)} finite full-recording reference windows are available; "
            f"minimum is {minimum_noise_windows}"
        )

    inactive_rows = [row for row in finite_rows if bool(row["official_inactive_pass"])]
    if len(inactive_rows) >= int(minimum_noise_windows):
        selected_rows = inactive_rows
        selected_levels = np.asarray([row["band_rms_dbfs"] for row in selected_rows], dtype=float)
        reference = float(np.median(selected_levels))
        reference_kind = "official_inactive_noise_floor"
        estimator = "median_band_rms_dbfs_over_full_recording_official_inactive_grid"
        fallback_used = False
        fallback_reason = ""
    else:
        ordered = sorted(finite_rows, key=lambda row: (float(row["band_rms_dbfs"]), int(row["noise_candidate_index"])))
        count = max(int(minimum_noise_windows), int(np.ceil(fraction * len(ordered))))
        count = min(count, len(ordered))
        selected_rows = ordered[:count]
        selected_levels = np.asarray([row["band_rms_dbfs"] for row in selected_rows], dtype=float)
        reference = float(np.median(selected_levels))
        reference_kind = "energy_lower_envelope_reference"
        estimator = "median_of_lowest_full_recording_band_rms_windows"
        fallback_used = True
        fallback_reason = (
            f"official_inactive_windows={len(inactive_rows)}<minimum={minimum_noise_windows}"
        )
        for row in selected_rows:
            row["lower_envelope_selected"] = True

    selected_ids = {int(row["noise_candidate_index"]) for row in selected_rows}
    for row in rows:
        row["reference_selected"] = int(row["noise_candidate_index"]) in selected_ids
        row["reference_kind"] = reference_kind

    mad = float(np.median(np.abs(selected_levels - reference)))
    audit = {
        "status": "PASS",
        "estimator": estimator,
        "reference_kind": reference_kind,
        "relative_gate_label": (
            "band_snr_db" if reference_kind == "official_inactive_noise_floor"
            else "band_level_margin_db"
        ),
        "candidate_source": "complete_synchronized_recording_fixed_grid_not_required_time",
        "noise_window_duration_s": float(window_duration_s),
        "noise_grid_hop_s": float(hop_duration_s),
        "candidate_windows_scanned": int(len(rows)),
        "finite_candidate_windows": int(len(finite_rows)),
        "official_inactive_candidates": int(len(inactive_rows)),
        "inactive_fraction_max": float(inactive_fraction_max),
        "noise_windows": int(len(selected_rows)),
        "reference_windows": int(len(selected_rows)),
        "noise_floor_dbfs": reference,
        "band_reference_dbfs": reference,
        "noise_floor_mad_db": mad,
        "reference_mad_db": mad,
        "noise_floor_p10_dbfs": float(np.quantile(selected_levels, 0.10)),
        "noise_floor_p90_dbfs": float(np.quantile(selected_levels, 0.90)),
        "lower_envelope_fraction": fraction,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "required_time_used_for_noise_floor": False,
        "physical_noise_floor_claimed": reference_kind == "official_inactive_noise_floor",
    }
    return reference, audit, rows


def qualify_active_speech_window(
    *,
    center_official_exactly_one_active: bool,
    active_source_index: int | None,
    activity_stats: ActivityWindowStats,
    band_rms: float,
    recording_noise_floor_dbfs: float,
    window_duration_s: float,
    relative_gate_required: bool,
    params: ActiveSpeechGateParameters = ActiveSpeechGateParameters(),
) -> ActiveSpeechDecision:
    """Qualify a speech-supported localization window without using DOA error.

    The primary evidence requirement is at least 100 ms of target activity in the
    0.75 s causal acoustic window.  The older 50% and 80% rules are retained as
    strict sensitivity subsets.  A 6 dB relative gate is enforced only when the
    recording reference is a genuine official-inactive noise floor.  When the
    recording has no such interval, the lower-energy envelope remains diagnostic
    and the absolute -80 dBFS band floor is the only hard energy gate.
    """
    band_dbfs = amplitude_to_dbfs(float(band_rms))
    snr = float(band_dbfs - recording_noise_floor_dbfs)
    target_fraction = float(activity_stats.target_fraction)
    target_duration_s = (
        target_fraction * float(window_duration_s)
        if np.isfinite(target_fraction)
        else float("nan")
    )
    target_pass = bool(
        np.isfinite(target_duration_s)
        and target_duration_s >= float(params.target_activity_duration_min_s) - 1.0e-12
    )
    floor_pass = bool(
        np.isfinite(band_dbfs)
        and band_dbfs > float(params.absolute_band_floor_dbfs)
    )
    snr_pass = bool(np.isfinite(snr) and snr >= float(params.band_snr_margin_db))
    reasons: list[str] = []
    if not center_official_exactly_one_active or active_source_index is None:
        reasons.append("center_not_exactly_one_official_active_source")
    if not target_pass:
        reasons.append("target_activity_duration_below_100ms")
    if not floor_pass:
        reasons.append("localization_band_below_absolute_floor")
    if bool(relative_gate_required) and not snr_pass:
        reasons.append("localization_band_snr_below_6db")
    eligible = len(reasons) == 0
    return ActiveSpeechDecision(
        eligible=eligible,
        rejection_reason="accepted" if eligible else ";".join(reasons),
        center_official_exactly_one_active=bool(center_official_exactly_one_active),
        active_source_index=(int(active_source_index) if active_source_index is not None else None),
        target_activity_fraction=target_fraction,
        target_activity_duration_s=float(target_duration_s),
        maximum_other_source_fraction=float(activity_stats.maximum_other_source_fraction),
        band_rms=float(band_rms),
        band_rms_dbfs=float(band_dbfs),
        recording_noise_floor_dbfs=float(recording_noise_floor_dbfs),
        band_snr_db=float(snr),
        target_activity_pass=target_pass,
        band_floor_pass=floor_pass,
        band_snr_pass=snr_pass,
        relative_gate_required=bool(relative_gate_required),
        strict_50pct_pass=bool(
            np.isfinite(target_fraction)
            and target_fraction >= float(params.strict_activity_fraction) - 1.0e-12
        ),
        sensitivity_80pct_pass=bool(
            np.isfinite(target_fraction)
            and target_fraction >= float(params.sensitivity_activity_fraction) - 1.0e-12
        ),
    )
