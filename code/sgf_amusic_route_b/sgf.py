# =========================================================
# File        : sgf.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the sgf module used by the reproducibility workflow.
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
import numpy as np


@dataclass(frozen=True)
class SGFFeatureWeights:
    log_magnitude: float = 0.15
    slope_support: float = 0.15
    peak_curvature: float = 0.25
    prominence: float = 0.25
    peakness: float = 0.20

    def normalized(self) -> np.ndarray:
        w = np.asarray([
            self.log_magnitude,
            self.slope_support,
            self.peak_curvature,
            self.prominence,
            self.peakness,
        ], dtype=float)
        if np.any(w < 0) or np.sum(w) <= 0:
            raise ValueError("SGF feature weights must be nonnegative and not all zero")
        return w / np.sum(w)


def _odd_window(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = _odd_window(window)
    pad = window // 2
    xp = np.pad(np.asarray(x, dtype=float), pad, mode="edge")
    return np.convolve(xp, np.ones(window, dtype=float) / window, mode="valid")


def _robust_unit_interval(x: np.ndarray, lower_q: float = 0.02, upper_q: float = 0.98) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.quantile(x, lower_q))
    hi = float(np.quantile(x, upper_q))
    if hi <= lo + 1e-14:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def sgf_features(
    theta_grid_deg,
    spectrum,
    smoothing_window: int = 5,
    slope_window: int = 5,
    prominence_window: int = 11,
) -> dict[str, np.ndarray]:
    """Return five bounded, interpretable spectral-geometry features.

    The spectrum is first normalized to unit maximum. Derivatives are with
    respect to degrees because all algorithmic grid parameters are specified
    in degrees. The slope feature is a local average of |d log P / d theta|,
    so a sharp peak receives support from its two shoulders even though the
    derivative at the exact peak is approximately zero.
    """
    theta = np.asarray(theta_grid_deg, dtype=float)
    P = np.asarray(spectrum, dtype=float)
    if theta.ndim != 1 or P.ndim != 1 or len(theta) != len(P):
        raise ValueError("theta_grid_deg and spectrum must be one-dimensional and equal length")
    if len(theta) < 3:
        zeros = np.zeros_like(P)
        return {name: zeros.copy() for name in (
            "log_magnitude", "slope_support", "peak_curvature", "prominence", "peakness"
        )}
    if np.any(np.diff(theta) <= 0):
        raise ValueError("theta_grid_deg must be strictly increasing")

    P = np.maximum(P, 1e-14)
    P = P / np.max(P)
    logp = np.log(P)
    logp_s = _moving_average(logp, smoothing_window)
    d1 = np.gradient(logp_s, theta)
    d2 = np.gradient(d1, theta)
    local_background = _moving_average(logp_s, prominence_window)

    amplitude = _robust_unit_interval(logp_s)
    slope_support = _robust_unit_interval(_moving_average(np.abs(d1), slope_window))
    peak_curvature = _robust_unit_interval(np.maximum(-d2, 0.0))
    prominence_raw = np.maximum(logp_s - local_background, 0.0)
    prominence = _robust_unit_interval(prominence_raw)
    peakness = _robust_unit_interval(prominence_raw * np.maximum(-d2, 0.0))

    return {
        "log_magnitude": amplitude,
        "slope_support": slope_support,
        "peak_curvature": peak_curvature,
        "prominence": prominence,
        "peakness": peakness,
    }


def sgf_intensity(
    theta_grid_deg,
    spectrum,
    smoothing_window: int = 5,
    slope_window: int = 5,
    prominence_window: int = 11,
    weights: SGFFeatureWeights = SGFFeatureWeights(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = sgf_features(
        theta_grid_deg,
        spectrum,
        smoothing_window=smoothing_window,
        slope_window=slope_window,
        prominence_window=prominence_window,
    )
    F = np.column_stack([features[name] for name in (
        "log_magnitude", "slope_support", "peak_curvature", "prominence", "peakness"
    )])
    intensity = F @ weights.normalized()
    intensity = np.clip(intensity, 0.0, 1.0)
    return intensity, features
