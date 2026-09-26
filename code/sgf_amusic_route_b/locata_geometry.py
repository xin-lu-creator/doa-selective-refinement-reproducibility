# =========================================================
# File        : locata_geometry.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the locata geometry module used by the reproducibility workflow.
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

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

from .broadband import conservative_aliasing_limit_hz
from .locata_time import (
    align_series_length,
    normalize_timestamp_units_for_rate,
    overlap_interval,
    parse_pose_file,
    read_timestamp_file,
    source_key_from_path,
)


@dataclass(frozen=True)
class LOCATAGeometryAudit:
    recording: str
    array_name: str
    usable: bool
    microphone_count: int
    rotation_orthogonality_error: float
    rotation_determinant: float
    aperture_m: float
    max_nearest_spacing_m: float
    aliasing_limit_hz: float
    channel_order: np.ndarray
    warnings: tuple[str, ...]
    diagnostics: dict


def _parse_pose(path: Path) -> dict:
    """Parse LOCATA pose metadata without destroying the shared system clock."""
    return parse_pose_file(Path(path))


def world_to_array_coordinates(
    point_world_xyz: np.ndarray,
    array_world_xyz: np.ndarray,
    rotation_array_to_world: np.ndarray | None,
) -> np.ndarray:
    """Transform a world point into the LOCATA array coordinate frame.

    LOCATA stores the array rotation matrix ``R`` in the array-to-world
    direction.  The official reference implementation therefore evaluates
    ``R.T @ (point - array_position)``.
    """
    relative = np.asarray(point_world_xyz, dtype=float) - np.asarray(array_world_xyz, dtype=float)
    if rotation_array_to_world is None:
        return relative
    return np.asarray(rotation_array_to_world, dtype=float).T @ relative


def local_azimuth_elevation_deg(relative_local_xyz: np.ndarray) -> tuple[float, float]:
    x, y, z = np.asarray(relative_local_xyz, dtype=float)
    # LOCATA defines azimuth zero on +y and positive angles towards -x.
    azimuth = np.rad2deg(np.arctan2(-x, y))
    elevation = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    return float(azimuth), float(elevation)


def wrap_azimuth_deg(angle_deg: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def internal_to_locata_azimuth_deg(angle_deg: np.ndarray | float) -> np.ndarray:
    """Map the generic scanner's +x-positive convention to LOCATA's -x-positive convention."""
    return wrap_azimuth_deg(-np.asarray(angle_deg, dtype=float))


def locata_to_internal_azimuth_deg(angle_deg: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`internal_to_locata_azimuth_deg` (the map is self-inverse)."""
    return wrap_azimuth_deg(-np.asarray(angle_deg, dtype=float))


def locata_scan_microphone_positions(microphone_positions_local_m: np.ndarray) -> np.ndarray:
    """Return scan-space positions whose generic steering grid is expressed in LOCATA azimuth.

    The generic broadband steering model uses direction ``(+sin(theta), +cos(theta))``.
    LOCATA uses ``(-sin(phi), +cos(phi))``. Reflecting the local x coordinate once at
    the real-LOCATA boundary makes every downstream MUSIC/JDML/SRP/tracking angle
    already use the official LOCATA sign, without changing the simulator convention.
    """
    positions = np.asarray(microphone_positions_local_m, dtype=float).copy()
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError("microphone_positions_local_m must have shape (M,2) or (M,3)")
    positions[:, 0] *= -1.0
    return positions


def _nearest_spacing(positions: np.ndarray) -> tuple[float, float]:
    pos = np.asarray(positions, dtype=float)
    if len(pos) < 2:
        return 0.0, 0.0
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    aperture = float(np.max(D))
    D[D <= 0] = np.inf
    nearest = np.min(D, axis=1)
    return aperture, float(np.max(nearest))


def validate_geometry(
    microphone_positions_local_m: np.ndarray,
    rotation_world_to_array: np.ndarray | None,
    *,
    recording: str = "",
    array_name: str = "unknown",
) -> LOCATAGeometryAudit:
    pos = np.asarray(microphone_positions_local_m, dtype=float)
    warnings: list[str] = []
    usable = bool(pos.ndim == 2 and pos.shape[0] >= 2 and pos.shape[1] == 3 and np.all(np.isfinite(pos)))
    if not usable:
        warnings.append("invalid_microphone_positions")
        pos = np.zeros((0, 3))
    if rotation_world_to_array is None:
        orth_error = float("nan")
        determinant = float("nan")
        warnings.append("rotation_missing")
    else:
        R = np.asarray(rotation_world_to_array, dtype=float)
        orth_error = float(np.linalg.norm(R @ R.T - np.eye(3), "fro"))
        determinant = float(np.linalg.det(R))
        if orth_error > 1e-3:
            warnings.append("rotation_not_orthogonal")
        if abs(determinant - 1.0) > 1e-3:
            warnings.append("rotation_determinant_not_one")
    aperture, nearest = _nearest_spacing(pos) if len(pos) else (0.0, 0.0)
    aliasing = conservative_aliasing_limit_hz(pos) if len(pos) else 0.0
    # Audio channels remain in their metadata/WAV order.  A dominant-axis
    # order is retained only as a diagnostic and must never silently reorder
    # audio relative to microphone coordinates.
    order = np.arange(len(pos), dtype=int) if len(pos) else np.empty(0, dtype=int)
    if len(pos):
        _, _, vh = np.linalg.svd(pos - np.mean(pos, axis=0), full_matrices=False)
        axis = vh[0]
        dominant_axis_order = np.argsort((pos - np.mean(pos, axis=0)) @ axis)
    else:
        dominant_axis_order = np.empty(0, dtype=int)
    if np.isfinite(aliasing) and aliasing > 0:
        warnings.append(f"adaptive_subarray_recommended_above_{int(round(aliasing))}Hz")
    return LOCATAGeometryAudit(
        recording=str(recording),
        array_name=str(array_name),
        usable=usable,
        microphone_count=int(len(pos)),
        rotation_orthogonality_error=orth_error,
        rotation_determinant=determinant,
        aperture_m=aperture,
        max_nearest_spacing_m=nearest,
        aliasing_limit_hz=float(aliasing),
        channel_order=np.asarray(order, dtype=int),
        warnings=tuple(warnings),
        diagnostics={
            "microphone_positions_local_m": pos,
            "dominant_axis_order": dominant_axis_order,
            "channel_order_semantics": "original_wav_metadata_order",
        },
    )


def _work_dir(recording: Path, array_name: str) -> Path:
    nested = recording / array_name
    return nested if nested.is_dir() else recording


def _find_one(work: Path, patterns: Sequence[str]) -> Path:
    for pattern in patterns:
        matches = sorted(work.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file matching {patterns} in {work}")


def infer_locata_task(recording: Path) -> str:
    text = str(recording).replace("\\", "/").lower()
    match = re.search(r"(?:^|/)task[_-]?(\d+)(?:/|$)", text)
    return f"task{match.group(1)}" if match else "unknown"


def _activity_time_axis(table: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """Construct an absolute timestamp axis from a headered activity table."""
    keys = {str(k).lower(): np.asarray(v) for k, v in table.items()}

    def find(*tokens: str) -> str | None:
        return next(
            (k for k in keys if any(token == k or token in k for token in tokens)),
            None,
        )

    year = find("year")
    month = find("month")
    day = find("day")
    hour = find("hour")
    minute = find("minute", "min")
    second = find("second", "sec")
    if all(item is not None for item in (year, month, day, hour, minute, second)):
        from datetime import datetime, timezone

        values = []
        for y, mo, d, h, mi, sec in zip(
            keys[year], keys[month], keys[day], keys[hour], keys[minute], keys[second]
        ):
            sec_float = float(sec)
            sec_int = int(np.floor(sec_float))
            micros = int(round((sec_float - sec_int) * 1_000_000.0))
            values.append(
                datetime(
                    int(y), int(mo), int(d), int(h), int(mi), sec_int, micros,
                    tzinfo=timezone.utc,
                ).timestamp()
            )
        return np.asarray(values, dtype=float), "calendar_ymdhms"

    explicit = next(
        (k for k in keys if "timestamp" in k or k in {"time", "t_sec", "seconds"}),
        None,
    )
    if explicit is not None:
        return np.asarray(keys[explicit], dtype=float), f"explicit_{explicit}"

    first = next(iter(keys))
    return np.asarray(keys[first], dtype=float), f"fallback_{first}"


def _source_index_from_name(path: Path) -> int:
    """Best-effort source index used only for deterministic VAD ordering."""
    name = path.stem.lower()
    matches = re.findall(r"(?:source|src|talker|spk|loudspeaker)[_-]?(\d+)", name)
    if matches:
        return int(matches[-1])
    numbers = re.findall(r"(\d+)", name)
    return int(numbers[-1]) if numbers else 10**9


def _vad_candidates(work: Path, array_name: str) -> tuple[list[Path], str]:
    """Return official array-aligned VAD files before source-side VAD files."""
    array = str(array_name).lower()
    all_txt = sorted(work.glob("*.txt"))
    array_side: list[Path] = []
    source_side: list[Path] = []
    activity_other: list[Path] = []
    for path in all_txt:
        name = path.name.lower()
        if "vad" in name and array in name:
            array_side.append(path)
        elif "vad" in name and ("source" in name or "src" in name):
            source_side.append(path)
        elif "activity" in name and ("source" in name or "src" in name):
            activity_other.append(path)

    def ordered(items: list[Path]) -> list[Path]:
        return sorted(
            set(items), key=lambda item: (_source_index_from_name(item), item.name.lower())
        )

    if array_side:
        return ordered(array_side), "official_array_aligned_vad"
    if source_side:
        return ordered(source_side), "official_source_side_vad_fallback"
    if activity_other:
        return ordered(activity_other), "official_activity_fallback"
    return [], "unavailable"


def _read_numeric_activity_rows(path: Path) -> np.ndarray | None:
    last_error: Exception | None = None
    for skiprows in (0, 1, 2):
        for delimiter in (None, "\t", ","):
            try:
                raw = np.loadtxt(path, dtype=float, delimiter=delimiter, skiprows=skiprows)
                raw = np.asarray(raw, dtype=float)
                if raw.size == 0:
                    continue
                if raw.ndim == 1:
                    raw = raw.reshape(-1, 1)
                return raw
            except Exception as exc:
                last_error = exc
    return None


def _read_activity_file(
    path: Path,
    *,
    source_kind: str,
    reference_t_abs_s: np.ndarray | None = None,
    reference_name: str = "",
) -> dict | None:
    """Read official activity labels on their synchronized reference timeline.

    LOCATA array-side VAD files commonly contain one activity value for every
    synchronized audio timestamp.  Such files must inherit the paired array
    audio timestamps; assigning a synthetic 120-Hz axis is incorrect.
    """
    raw = _read_numeric_activity_rows(path)
    if raw is None or raw.size == 0:
        return None

    if raw.shape[1] == 1:
        value = np.asarray(raw[:, 0], dtype=float)
        if reference_t_abs_s is None:
            # Retain an auditable legacy fallback for isolated unit tests and
            # nonstandard files, but the formal LOCATA loader always supplies
            # the paired synchronized timestamp stream.
            t_abs = np.arange(len(value), dtype=float) / 120.0
            parser_mode = "index_120hz_unpaired_fallback"
            length_diag = {
                "length_status": "unpaired_fallback",
                "length_difference": 0,
            }
        else:
            t_abs, length_diag = align_series_length(
                np.asarray(reference_t_abs_s, dtype=float),
                len(value),
                label=f"VAD {path.name}",
                maximum_trim=1,
            )
            n = min(len(t_abs), len(value))
            t_abs = t_abs[:n]
            value = value[:n]
            parser_mode = "paired_synchronized_timestamp_index"
    else:
        value = np.asarray(raw[:, -1], dtype=float)
        try:
            timestamp_series = read_timestamp_file(path)
            t_abs = np.asarray(timestamp_series.values_abs_s, dtype=float)
            parser_mode = f"embedded_{timestamp_series.parser_mode}"
        except Exception:
            if reference_t_abs_s is None:
                return None
            t_abs, length_diag = align_series_length(
                np.asarray(reference_t_abs_s, dtype=float),
                len(value),
                label=f"VAD {path.name}",
                maximum_trim=1,
            )
            parser_mode = "paired_synchronized_timestamp_fallback"
        n = min(len(t_abs), len(value))
        t_abs = t_abs[:n]
        value = value[:n]
        length_diag = {
            "length_status": "embedded_or_paired",
            "length_difference": int(len(t_abs) - len(value)),
        }

    if len(t_abs) == 0 or len(value) == 0:
        return None
    if len(t_abs) > 1 and not np.all(np.diff(t_abs) > 0):
        raise ValueError(f"Activity timestamps are not strictly increasing: {path}")
    active = np.asarray(value > 0.5, dtype=bool)
    return {
        "path": str(path),
        "file_name": path.name,
        "source_index": int(_source_index_from_name(path)),
        "source_key": source_key_from_path(path),
        "source_kind": str(source_kind),
        "reference_timestamp_file": str(reference_name),
        "t_abs_sec": np.asarray(t_abs, dtype=float),
        "t_sec": np.asarray(t_abs, dtype=float),
        "active": active,
        "parser_mode": parser_mode,
        "time_min_abs_s": float(t_abs[0]),
        "time_max_abs_s": float(t_abs[-1]),
        "active_fraction": float(np.mean(active)),
        **length_diag,
    }


def _load_activity_series(
    work: Path,
    source_count: int,
    *,
    array_name: str = "dicit",
    array_timestamps_abs_s: np.ndarray | None = None,
    array_timestamp_name: str = "",
    source_timestamp_map: dict[int, tuple[np.ndarray, str]] | None = None,
    return_source_kind: bool = False,
) -> list[dict] | None | tuple[list[dict] | None, str]:
    """Load activity labels on the correct synchronized audio timeline."""
    candidates, source_kind = _vad_candidates(work, array_name)
    if not candidates:
        return (None, "unavailable") if return_source_kind else None
    series: list[dict] = []
    source_timestamp_map = source_timestamp_map or {}
    for path in candidates:
        source_index = _source_index_from_name(path)
        if source_kind == "official_array_aligned_vad":
            reference = array_timestamps_abs_s
            reference_name = array_timestamp_name
        else:
            reference, reference_name = source_timestamp_map.get(
                source_index, (None, "")
            )
        item = _read_activity_file(
            path,
            source_kind=source_kind,
            reference_t_abs_s=reference,
            reference_name=reference_name,
        )
        if item is not None:
            series.append(item)
    if not series:
        return (None, "unavailable") if return_source_kind else None
    series.sort(
        key=lambda item: (item.get("source_index", 10**9), item.get("file_name", ""))
    )
    selected = series[: int(source_count)]
    return (selected, source_kind) if return_source_kind else selected


def interpolate_source_activity(geometry: dict, timestamps_s: Sequence[float]) -> list[np.ndarray] | None:
    activity = geometry.get("source_activity")
    if not activity:
        return None
    query = np.asarray(timestamps_s, dtype=float)
    outputs = []
    matrix = []
    for item in activity:
        t = np.asarray(item["t_sec"], dtype=float)
        a = np.asarray(item["active"], dtype=bool)
        right = np.searchsorted(t, query, side="left")
        right = np.clip(right, 0, len(t) - 1)
        left = np.clip(right - 1, 0, len(t) - 1)
        choose_left = np.abs(query - t[left]) <= np.abs(t[right] - query)
        ids = np.where(choose_left, left, right)
        matrix.append(a[ids])
    mat = np.asarray(matrix, dtype=bool).T
    return [row.copy() for row in mat]


def _find_optional(work: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(work.glob(pattern))
        if matches:
            return matches[0]
    return None


def _source_timestamp_map(work: Path) -> dict[int, tuple[np.ndarray, str]]:
    mapping: dict[int, tuple[np.ndarray, str]] = {}
    for path in sorted(work.glob("audio_source_timestamps*.txt")):
        try:
            series = read_timestamp_file(path)
        except Exception:
            continue
        mapping[_source_index_from_name(path)] = (
            np.asarray(series.values_abs_s, dtype=float),
            path.name,
        )
    return mapping


def _shift_pose_to_origin(pose: dict, origin_abs_s: float) -> dict:
    result = dict(pose)
    result["t_sec"] = np.asarray(result["t_abs_sec"], dtype=float) - float(origin_abs_s)
    return result


def _stream_diagnostic(name: str, values_abs_s: np.ndarray, origin_abs_s: float) -> dict:
    values = np.asarray(values_abs_s, dtype=float).reshape(-1)
    if len(values) == 0:
        return {
            "stream": name,
            "samples": 0,
            "start_abs_s": float("nan"),
            "end_abs_s": float("nan"),
            "start_relative_s": float("nan"),
            "end_relative_s": float("nan"),
        }
    return {
        "stream": name,
        "samples": int(len(values)),
        "start_abs_s": float(values[0]),
        "end_abs_s": float(values[-1]),
        "start_relative_s": float(values[0] - origin_abs_s),
        "end_relative_s": float(values[-1] - origin_abs_s),
    }


def load_recording_geometry(recording: Path, array_name: str = "dicit") -> dict:
    """Load one LOCATA recording on one shared synchronized time axis."""
    recording = Path(recording)
    work = _work_dir(recording, array_name)
    array_pose_path = _find_one(
        work,
        [
            f"position_array_{array_name}.txt",
            f"position_array_{array_name}_*.txt",
            "position_array*.txt",
        ],
    )
    source_paths = sorted(work.glob("position_source*.txt"))
    if not source_paths:
        raise FileNotFoundError(f"No source position metadata in {work}")
    audio_path = _find_one(
        work,
        [
            f"audio_array_{array_name}.wav",
            f"audio_array_{array_name}_*.wav",
            f"*{array_name}*.wav",
        ],
    )
    audio_timestamp_path = _find_one(
        work,
        [
            f"audio_array_timestamps_{array_name}.txt",
            f"audio_array_timestamps_{array_name}_*.txt",
            "audio_array_timestamps*.txt",
        ],
    )
    required_time_path = _find_optional(work, ["required_time.txt", "required*time*.txt"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        fs, audio = wavfile.read(str(audio_path))
    if audio.ndim != 2:
        raise ValueError("LOCATA array audio must be multichannel")

    audio_timestamp_series = read_timestamp_file(audio_timestamp_path)
    audio_t_abs, timestamp_unit_diag = normalize_timestamp_units_for_rate(
        audio_timestamp_series.values_abs_s,
        float(fs),
    )
    audio_t_abs, audio_length_diag = align_series_length(
        audio_t_abs,
        int(audio.shape[0]),
        label=f"Audio timestamp {audio_timestamp_path.name}",
        maximum_trim=1,
    )
    usable_audio_samples = min(int(audio.shape[0]), int(len(audio_t_abs)))
    audio_t_abs = np.asarray(audio_t_abs[:usable_audio_samples], dtype=float)
    if len(audio_t_abs) < 2:
        raise ValueError(f"Insufficient synchronized audio timestamps in {audio_timestamp_path}")
    common_origin_abs_s = float(audio_t_abs[0])

    arr = _shift_pose_to_origin(_parse_pose(array_pose_path), common_origin_abs_s)
    sources = [
        _shift_pose_to_origin(_parse_pose(path), common_origin_abs_s)
        for path in source_paths
    ]

    source_timestamp_map = _source_timestamp_map(work)
    source_activity, source_activity_source = _load_activity_series(
        work,
        len(sources),
        array_name=array_name,
        array_timestamps_abs_s=audio_t_abs,
        array_timestamp_name=audio_timestamp_path.name,
        source_timestamp_map=source_timestamp_map,
        return_source_kind=True,
    )
    if source_activity is not None:
        for item in source_activity:
            item["t_sec"] = (
                np.asarray(item["t_abs_sec"], dtype=float) - common_origin_abs_s
            )

    required_time = None
    if required_time_path is not None:
        required_series = read_timestamp_file(required_time_path, allow_equal=True)
        required_abs = np.asarray(required_series.values_abs_s, dtype=float)
        required_time = {
            "path": str(required_time_path),
            "file_name": required_time_path.name,
            "parser_mode": required_series.parser_mode,
            "t_abs_sec": required_abs,
            "t_sec": required_abs - common_origin_abs_s,
            "timestamp_audit": required_series.audit_dict(),
        }

    rotation0 = (
        None
        if arr.get("rotation") is None
        else np.asarray(arr["rotation"][0], dtype=float)
    )
    if arr.get("mic_positions") is not None:
        mic_world = np.asarray(arr["mic_positions"][0], dtype=float)
        mic_local = np.vstack(
            [
                world_to_array_coordinates(point, arr["xyz"][0], rotation0)
                for point in mic_world
            ]
        )
    else:
        mic_local = None
    if mic_local is not None and len(mic_local) != audio.shape[1]:
        raise ValueError(
            f"Microphone-coordinate count {len(mic_local)} does not match "
            f"audio channels {audio.shape[1]}"
        )

    audit = validate_geometry(
        np.zeros((audio.shape[1], 3)) if mic_local is None else mic_local,
        rotation0,
        recording=str(recording),
        array_name=array_name,
    )

    shared_clock_streams = [
        _stream_diagnostic("audio_array", audio_t_abs, common_origin_abs_s),
        _stream_diagnostic("array_pose", arr["t_abs_sec"], common_origin_abs_s),
    ]
    for index, source in enumerate(sources, start=1):
        shared_clock_streams.append(
            _stream_diagnostic(
                f"source_pose_{index}", source["t_abs_sec"], common_origin_abs_s
            )
        )
    if source_activity is not None:
        for index, item in enumerate(source_activity, start=1):
            shared_clock_streams.append(
                _stream_diagnostic(
                    f"source_activity_{index}",
                    item["t_abs_sec"],
                    common_origin_abs_s,
                )
            )
    if required_time is not None:
        shared_clock_streams.append(
            _stream_diagnostic(
                "required_time", required_time["t_abs_sec"], common_origin_abs_s
            )
        )

    required_overlap_inputs = [audio_t_abs, arr["t_abs_sec"]]
    required_overlap_inputs.extend(source["t_abs_sec"] for source in sources)
    if source_activity is not None:
        required_overlap_inputs.extend(item["t_abs_sec"] for item in source_activity)
    overlap_start, overlap_end, overlap_duration = overlap_interval(
        *required_overlap_inputs
    )
    shared_clock_status = "completed" if overlap_duration > 0 else "failed"
    if shared_clock_status != "completed":
        raise ValueError(
            "LOCATA synchronized streams have no common time overlap; "
            "check timestamp parsing and dataset integrity"
        )

    return {
        "recording": recording,
        "work_dir": work,
        "audio_path": audio_path,
        "audio_timestamp_path": audio_timestamp_path,
        "sample_rate_hz": int(fs),
        "audio_channels": int(audio.shape[1]),
        "audio_samples": int(audio.shape[0]),
        "usable_audio_samples": int(usable_audio_samples),
        "audio_t_abs_sec": audio_t_abs,
        "audio_t_sec": audio_t_abs - common_origin_abs_s,
        "common_origin_abs_s": common_origin_abs_s,
        "array_pose": arr,
        "source_poses": sources,
        "source_activity": source_activity,
        "source_activity_source": source_activity_source,
        "source_activity_diagnostics": []
        if source_activity is None
        else [
            {
                key: value
                for key, value in item.items()
                if key not in {"t_abs_sec", "t_sec", "active"}
            }
            for item in source_activity
        ],
        "required_time": required_time,
        "task": infer_locata_task(recording),
        "microphone_positions_local_m": mic_local,
        "audit": audit,
        "shared_clock_diagnostics": {
            "status": shared_clock_status,
            "common_origin_abs_s": common_origin_abs_s,
            "audio_timestamp_parser_mode": audio_timestamp_series.parser_mode,
            **timestamp_unit_diag,
            **audio_length_diag,
            "common_overlap_start_abs_s": float(overlap_start),
            "common_overlap_end_abs_s": float(overlap_end),
            "common_overlap_duration_s": float(overlap_duration),
            "streams": shared_clock_streams,
        },
    }


def interpolate_local_source_states(
    geometry: dict,
    timestamps_s: Sequence[float],
) -> list[dict[str, np.ndarray]]:
    """Interpolate source azimuth, elevation and range on the array-local frame.

    This is an estimator-independent forensic primitive.  It uses only LOCATA
    pose metadata and the already audited world-to-array coordinate chain.  The
    returned arrays preserve source order, allowing the official activity mask
    to select the active source without fitting any acoustic result.
    """
    arr = geometry["array_pose"]
    query = np.asarray(timestamps_s, dtype=float).reshape(-1)
    arr_xyz = interp1d(
        arr["t_sec"], arr["xyz"], axis=0, bounds_error=False,
        fill_value=(arr["xyz"][0], arr["xyz"][-1]),
    )(query)
    if arr.get("rotation") is None:
        rotations = [None] * len(query)
    else:
        # Nearest-pose sampling is intentionally identical to the formal truth
        # chain; SO(3) elements are never interpolated component-wise.
        ids = np.searchsorted(arr["t_sec"], query, side="left")
        ids = np.clip(ids, 0, len(arr["t_sec"]) - 1)
        rotations = [arr["rotation"][int(i)] for i in ids]

    output: list[dict[str, np.ndarray]] = []
    for qi, time in enumerate(query):
        azimuths: list[float] = []
        elevations: list[float] = []
        ranges: list[float] = []
        for source in geometry["source_poses"]:
            xyz = interp1d(
                source["t_sec"], source["xyz"], axis=0, bounds_error=False,
                fill_value=(source["xyz"][0], source["xyz"][-1]),
            )([time])[0]
            local = world_to_array_coordinates(xyz, arr_xyz[qi], rotations[qi])
            az, el = local_azimuth_elevation_deg(local)
            azimuths.append(float(az))
            elevations.append(float(el))
            ranges.append(float(np.linalg.norm(local)))
        output.append({
            "azimuth_deg": np.asarray(azimuths, dtype=float),
            "elevation_deg": np.asarray(elevations, dtype=float),
            "range_m": np.asarray(ranges, dtype=float),
        })
    return output


def interpolate_local_source_doas(geometry: dict, timestamps_s: Sequence[float]) -> list[np.ndarray]:
    """Backward-compatible azimuth-only wrapper."""
    return [state["azimuth_deg"] for state in interpolate_local_source_states(geometry, timestamps_s)]


def _circular_difference_scalar_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def summarize_source_motion_over_windows(
    geometry: dict,
    window_start_s: Sequence[float],
    window_end_s: Sequence[float],
    *,
    samples_per_window: int = 9,
) -> list[dict[str, np.ndarray]]:
    """Summarize metadata-only source motion inside each causal window.

    The summary is deliberately independent of the DOA estimator.  All window
    sample times are interpolated in one batch so dense required-time schedules
    remain inexpensive compared with acoustic processing.
    """
    starts = np.asarray(window_start_s, dtype=float).reshape(-1)
    ends = np.asarray(window_end_s, dtype=float).reshape(-1)
    if len(starts) != len(ends):
        raise ValueError("window start/end sequences must have equal length")
    if len(starts) == 0:
        return []
    count = max(int(samples_per_window), 3)
    fractions = np.linspace(0.0, 1.0, count)
    time_grid = starts[:, None] + (ends - starts)[:, None] * fractions[None, :]
    flat_states = interpolate_local_source_states(geometry, time_grid.reshape(-1))
    source_count = len(flat_states[0]["azimuth_deg"]) if flat_states else 0
    az = np.asarray([item["azimuth_deg"] for item in flat_states], dtype=float).reshape(
        len(starts), count, source_count
    )
    el = np.asarray([item["elevation_deg"] for item in flat_states], dtype=float).reshape(
        len(starts), count, source_count
    )
    rr = np.asarray([item["range_m"] for item in flat_states], dtype=float).reshape(
        len(starts), count, source_count
    )
    summaries: list[dict[str, np.ndarray]] = []
    for wi in range(len(starts)):
        az_w = az[wi]
        el_w = el[wi]
        rr_w = rr[wi]
        az_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(az_w), axis=0))
        az_mean = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(az_w)), axis=0)))
        projected = np.rad2deg(np.arcsin(np.clip(
            np.cos(np.deg2rad(el_w)) * np.sin(np.deg2rad(az_w)), -1.0, 1.0
        )))
        summaries.append({
            "azimuth_start_deg": az_w[0].copy(),
            "azimuth_mid_deg": az_w[len(az_w) // 2].copy(),
            "azimuth_end_deg": az_w[-1].copy(),
            "azimuth_circular_mean_deg": np.asarray(az_mean, dtype=float),
            "azimuth_excursion_deg": np.ptp(az_unwrapped, axis=0),
            "elevation_start_deg": el_w[0].copy(),
            "elevation_mid_deg": el_w[len(el_w) // 2].copy(),
            "elevation_end_deg": el_w[-1].copy(),
            "elevation_mean_deg": np.mean(el_w, axis=0),
            "elevation_excursion_deg": np.ptp(el_w, axis=0),
            "range_start_m": rr_w[0].copy(),
            "range_mid_m": rr_w[len(rr_w) // 2].copy(),
            "range_end_m": rr_w[-1].copy(),
            "range_mean_m": np.mean(rr_w, axis=0),
            "range_excursion_m": np.ptp(rr_w, axis=0),
            "cone_projected_azimuth_end_deg": projected[-1].copy(),
            "cone_projected_azimuth_mean_deg": np.mean(projected, axis=0),
            "endpoint_minus_window_mean_deg": np.asarray([
                _circular_difference_scalar_deg(ae, am)
                for ae, am in zip(az_w[-1], az_mean)
            ], dtype=float),
        })
    return summaries


def load_broadband_windows(
    recording: Path,
    *,
    array_name: str = "dicit",
    window_duration_s: float = 0.75,
    hop_duration_s: float = 0.25,
    max_windows: int = 0,
    window_anchor_mode: str = "fixed_hop_centered",
    activity_aware_max_windows: bool = False,
) -> tuple[list[dict], LOCATAGeometryAudit]:
    """Load synchronized LOCATA windows under an explicit anchor protocol.

    ``required_time_causal`` is the formal v5.0a2 protocol.  Every output row is
    anchored at one preserved row of ``required_time.txt`` and uses the causal
    interval ``[t_i-L, t_i]``.  The legacy ``fixed_hop_centered`` mode remains
    available only for backward-compatible diagnostics and unit tests.
    """
    geometry = load_recording_geometry(recording, array_name=array_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        fs, raw = wavfile.read(str(geometry["audio_path"]))
    usable_samples = min(int(len(raw)), int(geometry["usable_audio_samples"]))
    raw = raw[:usable_samples]
    audio = raw.astype(float)
    if np.issubdtype(raw.dtype, np.integer):
        audio /= max(float(np.iinfo(raw.dtype).max), 1.0)
    audio = audio.T

    length = max(32, int(round(float(window_duration_s) * fs)))
    hop = max(1, int(round(float(hop_duration_s) * fs)))
    audio_t_sec = np.asarray(geometry["audio_t_sec"], dtype=float)
    audio_t_abs = np.asarray(geometry["audio_t_abs_sec"], dtype=float)
    shared = geometry.get("shared_clock_diagnostics", {})
    origin_abs = float(geometry.get("common_origin_abs_s", 0.0))
    overlap_start_abs = float(shared.get("common_overlap_start_abs_s", audio_t_abs[0]))
    overlap_end_abs = float(shared.get("common_overlap_end_abs_s", audio_t_abs[-1]))

    mode = str(window_anchor_mode).strip().lower()
    anchor_diagnostics = {
        "window_anchor_mode": mode,
        "required_time_anchor_used": False,
        "window_duration_s": float(window_duration_s),
        "hop_duration_s": float(hop_duration_s),
        "requested_windows": 0,
        "accepted_required_windows_before_max_cap": 0,
        "accepted_required_windows": 0,
        "skipped_before_audio": 0,
        "skipped_after_audio": 0,
        "skipped_outside_shared_overlap": 0,
        "truncated_by_max_windows": 0,
        "activity_aware_max_windows": bool(activity_aware_max_windows),
        "active_required_windows_before_max_cap": 0,
        "activity_aware_selection_used": False,
    }

    if mode == "required_time_causal":
        required = geometry.get("required_time")
        if required is None:
            raise ValueError(
                "Formal v5.0a2 LOCATA evaluation requires required_time.txt; "
                "fixed-hop fallback is forbidden"
            )
        request_abs_all = np.asarray(required["t_abs_sec"], dtype=float).reshape(-1)
        request_rel_all = np.asarray(required["t_sec"], dtype=float).reshape(-1)
        anchor_diagnostics.update({
            "required_time_anchor_used": True,
            "required_time_file": str(required.get("file_name", "required_time.txt")),
            "required_time_parser_mode": str(required.get("parser_mode", "")),
            "required_time_timestamp_audit": dict(required.get("timestamp_audit", {})),
            "requested_windows": int(len(request_abs_all)),
        })
        starts_list: list[int] = []
        ends_list: list[int] = []
        anchors_list: list[int] = []
        centers_list: list[float] = []
        centers_abs_list: list[float] = []
        request_rows_list: list[int] = []
        sample_offset_list: list[float] = []
        for request_row, (request_abs, request_rel) in enumerate(
            zip(request_abs_all, request_rel_all)
        ):
            if request_abs < overlap_start_abs or request_abs > overlap_end_abs:
                anchor_diagnostics["skipped_outside_shared_overlap"] += 1
                continue
            if request_abs < audio_t_abs[0]:
                anchor_diagnostics["skipped_before_audio"] += 1
                continue
            if request_abs > audio_t_abs[-1]:
                anchor_diagnostics["skipped_after_audio"] += 1
                continue
            anchor_sample = int(np.searchsorted(audio_t_abs, request_abs, side="right") - 1)
            if anchor_sample < 0:
                anchor_diagnostics["skipped_before_audio"] += 1
                continue
            start_sample = int(anchor_sample - length + 1)
            end_exclusive = int(anchor_sample + 1)
            if start_sample < 0:
                anchor_diagnostics["skipped_before_audio"] += 1
                continue
            if end_exclusive > usable_samples:
                anchor_diagnostics["skipped_after_audio"] += 1
                continue
            starts_list.append(start_sample)
            ends_list.append(end_exclusive)
            anchors_list.append(anchor_sample)
            centers_list.append(float(request_rel))
            centers_abs_list.append(float(request_abs))
            request_rows_list.append(int(request_row))
            sample_offset_list.append(float(request_abs - audio_t_abs[anchor_sample]))
        anchor_diagnostics["accepted_required_windows_before_max_cap"] = int(len(starts_list))
        ids = np.arange(len(starts_list), dtype=int)
        if max_windows > 0 and activity_aware_max_windows and len(starts_list):
            pre_activities = interpolate_source_activity(
                geometry, np.asarray(centers_list, dtype=float)
            )
            if pre_activities is not None and len(pre_activities):
                active_matrix = np.asarray(pre_activities, dtype=bool)
                active_ids = np.flatnonzero(np.any(active_matrix, axis=1))
                anchor_diagnostics["active_required_windows_before_max_cap"] = int(len(active_ids))
                if len(active_ids) == 0:
                    raise ValueError(
                        "Required LOCATA Task 1/3 has official shared-clock activity "
                        "but no synchronized required-time window is active before smoke sampling"
                    )
                if len(active_ids) > int(max_windows):
                    relative = np.linspace(0, len(active_ids) - 1, int(max_windows)).astype(int)
                    ids = active_ids[relative]
                else:
                    ids = active_ids
                anchor_diagnostics["activity_aware_selection_used"] = True
        elif max_windows > 0 and len(starts_list) > max_windows:
            ids = np.linspace(0, len(starts_list) - 1, int(max_windows)).astype(int)
        if max_windows > 0 and len(ids) < len(starts_list):
            anchor_diagnostics["truncated_by_max_windows"] = int(len(starts_list) - len(ids))
            starts_list = [starts_list[int(i)] for i in ids]
            ends_list = [ends_list[int(i)] for i in ids]
            anchors_list = [anchors_list[int(i)] for i in ids]
            centers_list = [centers_list[int(i)] for i in ids]
            centers_abs_list = [centers_abs_list[int(i)] for i in ids]
            request_rows_list = [request_rows_list[int(i)] for i in ids]
            sample_offset_list = [sample_offset_list[int(i)] for i in ids]
        starts = np.asarray(starts_list, dtype=int)
        ends_exclusive = np.asarray(ends_list, dtype=int)
        anchor_indices = np.asarray(anchors_list, dtype=int)
        centers = np.asarray(centers_list, dtype=float)
        centers_abs = np.asarray(centers_abs_list, dtype=float)
        required_rows = np.asarray(request_rows_list, dtype=int)
        anchor_sample_offsets_s = np.asarray(sample_offset_list, dtype=float)
        anchor_diagnostics["accepted_required_windows"] = int(len(starts))
        if len(starts) == 0:
            raise ValueError(
                "required_time.txt was parsed, but no causal window remained after "
                f"audio/shared-overlap checks: {anchor_diagnostics}"
            )
    elif mode == "fixed_hop_centered":
        starts = np.arange(0, max(usable_samples - length + 1, 0), hop, dtype=int)
        anchor_indices = np.minimum(
            starts + int(round(0.5 * length)), usable_samples - 1
        )
        ends_exclusive = starts + length
        centers = audio_t_sec[anchor_indices] if len(anchor_indices) else np.asarray([], dtype=float)
        centers_abs = audio_t_abs[anchor_indices] if len(anchor_indices) else np.asarray([], dtype=float)
        within_overlap = (centers_abs >= overlap_start_abs) & (centers_abs <= overlap_end_abs)
        starts = starts[within_overlap]
        ends_exclusive = ends_exclusive[within_overlap]
        anchor_indices = anchor_indices[within_overlap]
        centers = centers[within_overlap]
        centers_abs = centers_abs[within_overlap]
        required_rows = np.full(len(starts), -1, dtype=int)
        anchor_sample_offsets_s = np.zeros(len(starts), dtype=float)
        if max_windows > 0 and len(starts) > max_windows:
            ids = np.linspace(0, len(starts) - 1, int(max_windows)).astype(int)
            starts = starts[ids]
            ends_exclusive = ends_exclusive[ids]
            anchor_indices = anchor_indices[ids]
            centers = centers[ids]
            centers_abs = centers_abs[ids]
            required_rows = required_rows[ids]
            anchor_sample_offsets_s = anchor_sample_offsets_s[ids]
    else:
        raise ValueError(
            "window_anchor_mode must be 'required_time_causal' or "
            "'fixed_hop_centered'"
        )

    truth_states = interpolate_local_source_states(geometry, centers)
    truths = [state["azimuth_deg"] for state in truth_states]
    window_start_times = audio_t_sec[starts]
    window_end_times = centers
    window_motion = summarize_source_motion_over_windows(
        geometry,
        window_start_times,
        window_end_times,
        samples_per_window=9,
    )
    activities = interpolate_source_activity(geometry, centers)
    activity_source_kind = str(geometry.get("source_activity_source", "unavailable"))
    activity_alignment_status = (
        "official_array_aligned_vad_shared_clock_ok"
        if activity_source_kind == "official_array_aligned_vad"
        else "activity_unavailable_or_fallback"
    )
    if activities is not None and len(activities):
        activity_matrix = np.asarray(activities, dtype=bool)
        any_active_windows = int(np.sum(np.any(activity_matrix, axis=1)))
        diagnostics = geometry.get("source_activity_diagnostics", [])
        maximum_active_fraction = max(
            [float(item.get("active_fraction", 0.0)) for item in diagnostics] or [0.0]
        )
        if maximum_active_fraction > 0.10 and any_active_windows == 0:
            task = str(geometry.get("task", "unknown")).lower()
            if task in {"task2", "task4", "task6"}:
                activity_alignment_status = "not_applicable_multisource_task_zero_overlap"
            elif task == "task5":
                activity_alignment_status = "diagnostic_task_zero_overlap"
            else:
                raise ValueError(
                    "Required LOCATA Task 1/3 has official shared-clock activity "
                    "but no synchronized analysis window is active"
                )

    positions = geometry["microphone_positions_local_m"]
    if positions is None:
        raise ValueError(
            "Real arbitrary-geometry broadband evaluation requires microphone coordinates"
        )
    required_audit = dict(anchor_diagnostics.get("required_time_timestamp_audit", {}))
    windows = []
    for index, (start, end_exclusive, anchor_sample, center, center_abs, truth, truth_state, motion, request_row, sample_offset) in enumerate(
        zip(
            starts,
            ends_exclusive,
            anchor_indices,
            centers,
            centers_abs,
            truths,
            truth_states,
            window_motion,
            required_rows,
            anchor_sample_offsets_s,
        )
    ):
        active = None if activities is None else np.asarray(activities[index], dtype=bool)
        windows.append(
            {
                "frame_index": int(index),
                "required_time_row_index": int(request_row),
                "timestamp": float(center),
                "timestamp_abs_s": float(center_abs),
                "required_time_abs_s": float(center_abs) if mode == "required_time_causal" else float("nan"),
                "audio_anchor_sample": int(anchor_sample),
                "audio_center_sample": int(anchor_sample),
                "audio_window_start_sample": int(start),
                "audio_window_end_exclusive_sample": int(end_exclusive),
                "anchor_sample_offset_s": float(sample_offset),
                "window_anchor_mode": mode,
                "required_time_anchor_used": bool(mode == "required_time_causal"),
                "required_time_file": str(anchor_diagnostics.get("required_time_file", "")),
                "required_time_parser_mode": str(anchor_diagnostics.get("required_time_parser_mode", "")),
                "required_time_timestamp_audit": required_audit,
                "window_anchor_diagnostics": dict(anchor_diagnostics),
                "audio": audio[:, int(start):int(end_exclusive)],
                "sample_rate_hz": int(fs),
                "microphone_positions_m": locata_scan_microphone_positions(positions),
                "microphone_positions_raw_local_m": np.asarray(positions),
                "steering_coordinate_transform": "local_x_reflection_for_locata_azimuth",
                "theta_true_deg": np.asarray(truth, dtype=float),
                "elevation_true_deg": np.asarray(truth_state["elevation_deg"], dtype=float),
                "source_range_true_m": np.asarray(truth_state["range_m"], dtype=float),
                "source_window_motion": {
                    key: np.asarray(value, dtype=float)
                    for key, value in motion.items()
                },
                "window_start_timestamp": float(audio_t_sec[int(start)]),
                "window_end_timestamp": float(center),
                "source_active": active,
                "source_activity_source": geometry.get(
                    "source_activity_source", "unavailable"
                ),
                "source_activity_diagnostics": geometry.get(
                    "source_activity_diagnostics", []
                ),
                "source_activity_alignment_status": activity_alignment_status,
                "shared_clock_status": shared.get("status", "unknown"),
                "shared_clock_overlap_duration_s": float(
                    shared.get("common_overlap_duration_s", float("nan"))
                ),
                "shared_clock_common_origin_abs_s": float(
                    shared.get("common_origin_abs_s", float("nan"))
                ),
                "audio_timestamp_file": Path(
                    geometry.get("audio_timestamp_path", "")
                ).name,
                "audio_timestamp_parser_mode": shared.get(
                    "audio_timestamp_parser_mode", ""
                ),
                "audio_timestamp_unit": shared.get("timestamp_unit", ""),
                "coordinate_convention": (
                    "LOCATA: azimuth 0 on +y; positive towards -x; "
                    "world-to-array uses R.T; generic steering uses one local-x reflection"
                ),
                "task": geometry.get("task", "unknown"),
            }
        )
    return windows, geometry["audit"]

def audit_locata_root(
    root: Path,
    output_dir: Path,
    *,
    array_name: str = "dicit",
) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings = sorted({p.parent for p in root.rglob(f"audio_array_{array_name}*.wav")}) if root.exists() else []
    rows = []
    errors = []
    for work in recordings:
        recording = work.parent if work.name == array_name else work
        try:
            geometry = load_recording_geometry(recording, array_name=array_name)
            audit = geometry["audit"]
            rows.append({
                "recording": str(recording),
                "array_name": array_name,
                "usable": audit.usable,
                "microphone_count": audit.microphone_count,
                "rotation_orthogonality_error": audit.rotation_orthogonality_error,
                "rotation_determinant": audit.rotation_determinant,
                "aperture_m": audit.aperture_m,
                "max_nearest_spacing_m": audit.max_nearest_spacing_m,
                "aliasing_limit_hz": audit.aliasing_limit_hz,
                "channel_order": json.dumps(audit.channel_order.tolist()),
                "channel_order_semantics": audit.diagnostics.get("channel_order_semantics", ""),
                "dominant_axis_order": json.dumps(np.asarray(audit.diagnostics.get("dominant_axis_order", [])).tolist()),
                "warnings": json.dumps(list(audit.warnings)),
                "shared_clock_status": geometry.get("shared_clock_diagnostics", {}).get("status", "unknown"),
                "audio_timestamp_parser_mode": geometry.get("shared_clock_diagnostics", {}).get("audio_timestamp_parser_mode", ""),
                "audio_timestamp_unit": geometry.get("shared_clock_diagnostics", {}).get("timestamp_unit", ""),
                "shared_clock_overlap_duration_s": geometry.get("shared_clock_diagnostics", {}).get("common_overlap_duration_s", float("nan")),
                "audio_timestamp_file": Path(geometry.get("audio_timestamp_path", "")).name,
            })
        except Exception as exc:  # auditable skip, not silent fallback
            errors.append({"recording": str(recording), "error": f"{type(exc).__name__}: {exc}"})
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "LOCATA_GEOMETRY_AUDIT.csv", index=False)
    status = {
        "root": str(root),
        "root_exists": root.exists(),
        "array_name": array_name,
        "recordings_discovered": len(recordings),
        "recordings_usable": int(table["usable"].sum()) if not table.empty else 0,
        "errors": errors,
        "status": (
            "completed" if rows and not errors
            else "completed_with_warnings" if rows and errors
            else "failed" if errors
            else "skipped"
        ),
    }
    (output_dir / "LOCATA_GEOMETRY_AUDIT.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return table, status
