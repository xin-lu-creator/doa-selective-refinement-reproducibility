# =========================================================
# File        : locata_time.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the locata time module used by the reproducibility workflow.
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
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimestampSeries:
    path: str
    values_abs_s: np.ndarray
    parser_mode: str
    monotone: bool
    median_step_s: float
    nondecreasing: bool
    duplicate_count: int
    negative_step_count: int
    zero_step_count: int
    minimum_positive_step_s: float
    maximum_step_s: float
    first_nonincreasing_index: int | None

    def audit_dict(self) -> dict:
        return {
            "row_count": int(len(self.values_abs_s)),
            "unique_timestamp_count": int(len(np.unique(self.values_abs_s))),
            "parser_mode": self.parser_mode,
            "strictly_increasing": bool(self.monotone),
            "nondecreasing": bool(self.nondecreasing),
            "duplicate_count": int(self.duplicate_count),
            "negative_step_count": int(self.negative_step_count),
            "zero_step_count": int(self.zero_step_count),
            "median_step_s": float(self.median_step_s),
            "minimum_positive_step_s": float(self.minimum_positive_step_s),
            "maximum_step_s": float(self.maximum_step_s),
            "first_nonincreasing_index": self.first_nonincreasing_index,
        }


def _first_numeric_line(path: Path, maximum_lines: int = 4) -> tuple[int, str | None, list[float]]:
    """Inspect a LOCATA text table without repeatedly reading a large file."""
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number in range(maximum_lines):
            line = stream.readline()
            if line == "":
                break
            stripped = line.strip()
            if not stripped:
                continue
            delimiter = "," if "," in stripped else None
            tokens = stripped.split(",") if delimiter == "," else stripped.split()
            try:
                values = [float(token) for token in tokens]
            except ValueError:
                continue
            if values:
                return line_number, delimiter, values
    raise ValueError(f"No numeric row found in LOCATA table {path}")


def _load_numeric_rows(path: Path) -> np.ndarray:
    """Load a moderate LOCATA text table with one deterministic parse pass."""
    skiprows, delimiter, _ = _first_numeric_line(Path(path))
    raw = np.loadtxt(
        path,
        dtype=float,
        delimiter=delimiter,
        skiprows=skiprows,
        ndmin=2,
    )
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        raise ValueError(f"Empty LOCATA table: {path}")
    return raw


def _calendar_rows_to_epoch_seconds(raw: np.ndarray) -> np.ndarray:
    """Vectorized UTC calendar conversion suitable for per-audio-sample files."""
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError("calendar table requires at least six columns")
    calendar = np.asarray(raw[:, :6], dtype=float)
    ymd = np.rint(calendar[:, :3]).astype(np.int64)
    unique_days, inverse = np.unique(ymd, axis=0, return_inverse=True)
    midnight_epoch = np.empty(len(unique_days), dtype=float)
    for index, (year, month, day) in enumerate(unique_days):
        midnight_epoch[index] = datetime(
            int(year), int(month), int(day), tzinfo=timezone.utc
        ).timestamp()
    seconds_of_day = (
        calendar[:, 3] * 3600.0
        + calendar[:, 4] * 60.0
        + calendar[:, 5]
    )
    return midnight_epoch[inverse] + seconds_of_day


def _read_calendar_timestamp_file_chunked(
    path: Path,
    *,
    skiprows: int,
    delimiter: str | None,
    chunksize: int = 500_000,
) -> np.ndarray:
    """Read huge six-column calendar timestamp files without a six-column peak copy."""
    sep = delimiter if delimiter is not None else r"\s+"
    chunks: list[np.ndarray] = []
    reader = pd.read_csv(
        path,
        sep=sep,
        header=None,
        skiprows=skiprows,
        usecols=list(range(6)),
        dtype=float,
        chunksize=int(chunksize),
        engine="c",
    )
    for chunk in reader:
        chunks.append(_calendar_rows_to_epoch_seconds(chunk.to_numpy(dtype=float)))
    if not chunks:
        raise ValueError(f"Empty LOCATA timestamp file: {path}")
    return np.concatenate(chunks)

def _looks_like_calendar(raw: np.ndarray) -> bool:
    return bool(
        raw.ndim == 2
        and raw.shape[1] >= 6
        and np.all(np.isfinite(raw[:, :6]))
        and np.all((raw[:, 0] >= 1900) & (raw[:, 0] <= 2100))
        and np.all((raw[:, 1] >= 1) & (raw[:, 1] <= 12))
        and np.all((raw[:, 2] >= 1) & (raw[:, 2] <= 31))
        and np.all((raw[:, 3] >= 0) & (raw[:, 3] < 24))
        and np.all((raw[:, 4] >= 0) & (raw[:, 4] < 60))
        and np.all((raw[:, 5] >= 0) & (raw[:, 5] < 61))
    )


def _choose_numeric_timestamp_column(raw: np.ndarray) -> tuple[np.ndarray, str]:
    """Choose a monotone numeric time column from a LOCATA timestamp table."""
    if raw.shape[1] == 1:
        return np.asarray(raw[:, 0], dtype=float), "numeric_single_column"
    candidates: list[tuple[float, int, np.ndarray]] = []
    for column in range(raw.shape[1]):
        values = np.asarray(raw[:, column], dtype=float)
        if not np.all(np.isfinite(values)) or len(values) < 2:
            continue
        diff = np.diff(values)
        positive_fraction = float(np.mean(diff > 0))
        nonnegative_fraction = float(np.mean(diff >= 0))
        span = float(values[-1] - values[0])
        if positive_fraction >= 0.95 and nonnegative_fraction >= 0.999 and span > 0:
            candidates.append((positive_fraction, column, values))
    if not candidates:
        raise ValueError("No monotone timestamp column found")
    # Prefer the earliest monotone column; LOCATA time fields precede payload columns.
    _, column, values = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return values, f"numeric_column_{column}"


@lru_cache(maxsize=256)
def _read_timestamp_file_cached(
    path_text: str,
    file_size: int,
    mtime_ns: int,
    allow_equal: bool,
) -> TimestampSeries:
    path = Path(path_text)
    skiprows, delimiter, first_values = _first_numeric_line(path)
    first = np.asarray(first_values, dtype=float).reshape(1, -1)
    if _looks_like_calendar(first):
        values = _read_calendar_timestamp_file_chunked(
            path, skiprows=skiprows, delimiter=delimiter
        )
        mode = "calendar_ymdhms"
    else:
        raw = np.loadtxt(
            path, dtype=float, delimiter=delimiter, skiprows=skiprows, ndmin=2
        )
        values, mode = _choose_numeric_timestamp_column(np.asarray(raw, dtype=float))
    values = np.asarray(values, dtype=float).reshape(-1)
    diff = np.diff(values)
    negative = np.flatnonzero(diff < 0)
    zero = np.flatnonzero(diff == 0)
    nonincreasing = np.flatnonzero(diff <= 0)
    positive = diff[diff > 0]
    strictly_increasing = bool(len(values) <= 1 or len(nonincreasing) == 0)
    nondecreasing = bool(len(values) <= 1 or len(negative) == 0)
    first_bad = int(nonincreasing[0] + 1) if len(nonincreasing) else None
    if len(negative):
        row = int(negative[0] + 1)
        raise ValueError(
            f"Timestamp series decreases at row {row}: {path}; "
            f"previous={values[row-1]:.9f}, current={values[row]:.9f}, "
            f"delta={diff[row-1]:.9g}"
        )
    if len(zero) and not bool(allow_equal):
        row = int(zero[0] + 1)
        raise ValueError(
            f"Timestamp series contains duplicate time at row {row}: {path}; "
            f"value={values[row]:.9f}. Use allow_equal=True only for streams "
            "whose official format permits repeated requested timestamps."
        )
    median_step = float(np.median(positive)) if len(positive) else float("nan")
    minimum_positive = float(np.min(positive)) if len(positive) else float("nan")
    maximum_step = float(np.max(diff)) if len(diff) else float("nan")
    return TimestampSeries(
        path=str(path),
        values_abs_s=values,
        parser_mode=mode,
        monotone=strictly_increasing,
        median_step_s=median_step,
        nondecreasing=nondecreasing,
        duplicate_count=int(len(zero)),
        negative_step_count=int(len(negative)),
        zero_step_count=int(len(zero)),
        minimum_positive_step_s=minimum_positive,
        maximum_step_s=maximum_step,
        first_nonincreasing_index=first_bad,
    )

def read_timestamp_file(path: Path, *, allow_equal: bool = False) -> TimestampSeries:
    """Read synchronized timestamps with explicit monotonicity policy.

    Audio, pose, and activity clocks remain strictly increasing. LOCATA's
    ``required_time.txt`` is a request schedule rather than a sampled signal;
    repeated requested timestamps are preserved and audited when
    ``allow_equal=True``. Negative time steps are never reordered or hidden.
    """
    path = Path(path)
    stat = path.stat()
    return _read_timestamp_file_cached(
        str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns), bool(allow_equal)
    )


def align_series_length(
    timestamps_abs_s: np.ndarray,
    expected_length: int,
    *,
    label: str,
    maximum_trim: int = 1,
) -> tuple[np.ndarray, dict]:
    """Validate a timestamp/label stream against its paired audio stream."""
    values = np.asarray(timestamps_abs_s, dtype=float).reshape(-1)
    difference = int(len(values) - int(expected_length))
    if difference == 0:
        return values, {"length_status": "exact", "length_difference": 0}
    if abs(difference) <= int(maximum_trim):
        n = min(len(values), int(expected_length))
        return values[:n], {
            "length_status": "trimmed_small_mismatch",
            "length_difference": difference,
            "trimmed_to": int(n),
        }
    raise ValueError(
        f"{label} length {len(values)} does not match paired audio length "
        f"{expected_length} (difference {difference})"
    )


def parse_pose_file(path: Path) -> dict:
    """Parse LOCATA pose metadata while preserving absolute synchronized time."""
    raw = _load_numeric_rows(Path(path))
    n_columns = raw.shape[1]
    if _looks_like_calendar(raw):
        t_abs = _calendar_rows_to_epoch_seconds(raw)
        offset = 6
        parser_mode = "calendar_ymdhms"
    else:
        t_abs, time_mode = _choose_numeric_timestamp_column(raw)
        # Official fallback tables use the first time column followed by xyz.
        offset = 1
        parser_mode = time_mode
    if n_columns < offset + 3:
        raise ValueError(f"Pose file has insufficient columns: {path}")
    xyz = np.asarray(raw[:, offset : offset + 3], dtype=float)
    rotation = None
    mic_positions = None
    # Calendar LOCATA pose layout: timestamp(6), position(3), ref-vector(3),
    # rotation(9), microphones(3*M).  Numeric fallback is retained for tests.
    if offset == 6 and n_columns >= 21:
        rotation = np.asarray(raw[:, 12:21], dtype=float).reshape(-1, 3, 3)
        if n_columns >= 24:
            n_mics = (n_columns - 21) // 3
            if n_mics > 0:
                mic_positions = np.asarray(
                    raw[:, 21 : 21 + 3 * n_mics], dtype=float
                ).reshape(-1, n_mics, 3)
    return {
        "path": str(path),
        "timestamp_parser_mode": parser_mode,
        "t_abs_sec": np.asarray(t_abs, dtype=float),
        "t_sec": np.asarray(t_abs, dtype=float),
        "xyz": xyz,
        "rotation": rotation,
        "mic_positions": mic_positions,
    }


def source_key_from_path(path: Path) -> str:
    name = Path(path).stem.lower()
    name = re.sub(r"^(?:vad_|audio_source_timestamps_|position_source_)", "", name)
    return name


def overlap_interval(*series: Iterable[float]) -> tuple[float, float, float]:
    arrays = [np.asarray(item, dtype=float).reshape(-1) for item in series]
    arrays = [item for item in arrays if len(item)]
    if not arrays:
        return float("nan"), float("nan"), 0.0
    start = max(float(item[0]) for item in arrays)
    end = min(float(item[-1]) for item in arrays)
    return start, end, max(0.0, end - start)


def normalize_timestamp_units_for_rate(
    values: np.ndarray,
    expected_rate_hz: float,
) -> tuple[np.ndarray, dict]:
    """Normalize common second/millisecond/microsecond/nanosecond time units.

    The scale is selected only when the median timestamp increment matches the
    expected sample period after applying a standard SI time conversion.
    """
    t = np.asarray(values, dtype=float).reshape(-1)
    if len(t) < 2 or expected_rate_hz <= 0:
        return t, {"timestamp_unit": "undetermined", "timestamp_scale": 1.0}
    step = float(np.median(np.diff(t)))
    expected = 1.0 / float(expected_rate_hz)
    candidates = [
        (1.0, "seconds"),
        (1e3, "milliseconds"),
        (1e6, "microseconds"),
        (1e9, "nanoseconds"),
    ]
    scored = []
    for scale, name in candidates:
        normalized_step = step / scale
        relative_error = abs(normalized_step - expected) / max(expected, 1e-15)
        scored.append((relative_error, scale, name, normalized_step))
    error, scale, name, normalized_step = min(scored, key=lambda item: item[0])
    if error <= 0.10:
        return t / scale, {
            "timestamp_unit": name,
            "timestamp_scale": float(scale),
            "timestamp_step_s": float(normalized_step),
            "timestamp_step_relative_error": float(error),
        }
    return t, {
        "timestamp_unit": "unrecognized",
        "timestamp_scale": 1.0,
        "timestamp_step_raw": step,
        "expected_step_s": expected,
        "best_relative_error": float(error),
    }
