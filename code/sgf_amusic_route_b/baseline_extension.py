# =========================================================
# File        : baseline_extension.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the baseline extension module used by the reproducibility workflow.
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
"""Three-tier baseline extension utilities for LOCATA Paper 1.

This module is deliberately isolated from the frozen SGF-AMUSIC estimator.  It
implements two publication baselines only:

* I-MUSIC: uniform incoherent sum of the raw narrowband MUSIC spectra.
* NormMUSIC: uniform incoherent sum after per-frequency peak normalization.
* GCC-PHAT/TDE: all-pair PHAT delays followed by a deterministic robust
  geometry-only azimuth fit.

No ground-truth direction is used by any estimator or pair/frequency selection
rule.  Truth is used only by the experiment driver after an estimate has been
produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from .broadband import CovarianceEvidence, _noise_projector, _positions_for_evidence, steering_matrix_geometry

SPEED_OF_SOUND = 343.0


@dataclass(frozen=True)
class GCCPHATParameters:
    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 3500.0
    theta_min_deg: float = -80.0
    theta_max_deg: float = 80.0
    coarse_step_deg: float = 1.0
    refine_radius_deg: float = 1.5
    pair_min_baseline_m: float = 0.05
    huber_delta_scale: float = 1.5
    confidence_floor: float = 0.05
    maximum_pairs: int = 0  # 0 means all admissible pairs.


def _normalize01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    lo = float(np.min(x)) if x.size else 0.0
    hi = float(np.max(x)) if x.size else 0.0
    if hi <= lo + 1e-15:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def music_family_spectra(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    theta_grid_deg: np.ndarray,
    num_sources: int = 1,
) -> dict[str, np.ndarray]:
    """Compute I-MUSIC and NormMUSIC on exactly the same evidence bank.

    The sensor subset and frequency bank are inherited from ``evidences`` and
    therefore match the frozen preprocessing.  The only difference between the
    two baselines is per-frequency spectrum normalization.
    """
    grid = np.asarray(theta_grid_deg, dtype=float)
    raw_rows: list[np.ndarray] = []
    norm_rows: list[np.ndarray] = []
    for item in evidences:
        projector = _noise_projector(item.covariance, int(num_sources))
        positions = _positions_for_evidence(item, microphone_positions_m)
        manifold = steering_matrix_geometry(positions, grid, float(item.frequency_hz))
        denominator = np.sum(manifold.conj() * (projector @ manifold), axis=0).real
        raw = 1.0 / np.maximum(denominator, 1e-12)
        raw_rows.append(raw)
        norm_rows.append(raw / max(float(np.max(raw)), 1e-12))
    if not raw_rows:
        raise ValueError("At least one covariance evidence item is required")
    raw_stack = np.asarray(raw_rows, dtype=float)
    norm_stack = np.asarray(norm_rows, dtype=float)
    imusic = np.mean(raw_stack, axis=0)
    normmusic = np.mean(norm_stack, axis=0)
    return {
        "grid_deg": grid,
        "imusic_spectrum": _normalize01(imusic),
        "normmusic_spectrum": _normalize01(normmusic),
        "per_frequency_raw": raw_stack,
        "per_frequency_normalized": norm_stack,
    }


def music_family_estimates(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    theta_grid_deg: np.ndarray,
    num_sources: int = 1,
) -> tuple[dict[str, float], dict]:
    if int(num_sources) != 1:
        raise NotImplementedError("The baseline extension is preregistered for single-source LOCATA Tasks 1 and 3")
    spectra = music_family_spectra(evidences, microphone_positions_m, theta_grid_deg, num_sources=1)
    grid = spectra["grid_deg"]
    estimates = {
        "imusic_uniform_raw": float(grid[int(np.argmax(spectra["imusic_spectrum"]))]),
        "normmusic_uniform": float(grid[int(np.argmax(spectra["normmusic_spectrum"]))]),
    }
    return estimates, spectra


def deterministic_pair_indices(
    microphone_positions_m: np.ndarray,
    *,
    minimum_baseline_m: float = 0.05,
    maximum_pairs: int = 0,
) -> np.ndarray:
    """Geometry-only deterministic microphone-pair bank.

    All admissible pairs are used by default.  If a cap is requested, pairs are
    selected by evenly sampling the descending baseline-length order.  This is
    deterministic and does not use audio or source truth.
    """
    pos = np.asarray(microphone_positions_m, dtype=float)
    rows: list[tuple[int, int, float]] = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            length = float(np.linalg.norm(pos[i] - pos[j]))
            if length >= float(minimum_baseline_m):
                rows.append((i, j, length))
    if not rows:
        raise ValueError("No microphone pair satisfies the minimum baseline")
    rows.sort(key=lambda item: (-item[2], item[0], item[1]))
    if int(maximum_pairs) > 0 and len(rows) > int(maximum_pairs):
        ids = np.linspace(0, len(rows) - 1, int(maximum_pairs)).round().astype(int)
        rows = [rows[int(k)] for k in np.unique(ids)]
    return np.asarray([[i, j] for i, j, _ in rows], dtype=int)


def _parabolic_peak(y: np.ndarray, index: int) -> float:
    """Sub-sample offset around a local maximum, clipped to one bin."""
    if index <= 0 or index >= len(y) - 1:
        return 0.0
    a, b, c = float(y[index - 1]), float(y[index]), float(y[index + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-15:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))


def gcc_phat_delay(
    signal_i: np.ndarray,
    signal_j: np.ndarray,
    sample_rate_hz: float,
    *,
    maximum_delay_s: float,
    frequency_min_hz: float,
    frequency_max_hz: float,
) -> tuple[float, dict]:
    """Estimate ``arrival_i - arrival_j`` with band-limited GCC-PHAT."""
    x = np.asarray(signal_i, dtype=float).reshape(-1)
    y = np.asarray(signal_j, dtype=float).reshape(-1)
    n = max(len(x), len(y))
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 32))))
    X = np.fft.rfft(x, n=nfft)
    Y = np.fft.rfft(y, n=nfft)
    frequency = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate_hz))
    mask = (frequency >= float(frequency_min_hz)) & (frequency <= float(frequency_max_hz))
    cross = X * np.conj(Y)
    phat = np.zeros_like(cross)
    phat[mask] = cross[mask] / np.maximum(np.abs(cross[mask]), 1e-15)
    correlation = np.fft.irfft(phat, n=nfft)
    maximum_lag = min(int(np.ceil(float(maximum_delay_s) * sample_rate_hz)) + 1, nfft // 2 - 1)
    lags = np.arange(-maximum_lag, maximum_lag + 1, dtype=int)
    values = np.concatenate([correlation[-maximum_lag:], correlation[: maximum_lag + 1]])
    peak_index = int(np.argmax(values))
    sub = _parabolic_peak(values, peak_index)
    lag_samples = float(lags[peak_index]) + sub
    absolute = np.abs(values)
    peak = float(values[peak_index])
    median = float(np.median(absolute))
    exclusion = max(1, int(round(0.00015 * sample_rate_hz)))
    keep = np.ones(len(values), dtype=bool)
    keep[max(0, peak_index - exclusion): min(len(values), peak_index + exclusion + 1)] = False
    second = float(np.max(values[keep])) if np.any(keep) else 0.0
    confidence = max(peak, 0.0) / max(median, 1e-12)
    return lag_samples / float(sample_rate_hz), {
        "lag_samples": lag_samples,
        "peak": peak,
        "peak_to_median": confidence,
        "peak_to_second": peak / max(second, 1e-12) if second > 0 else float("inf"),
        "nfft": int(nfft),
        "maximum_lag_samples": int(maximum_lag),
    }


def _circular_difference_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _predicted_pair_delays(positions: np.ndarray, pairs: np.ndarray, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(theta_deg))
    direction = np.asarray([np.sin(theta), np.cos(theta), 0.0], dtype=float)
    pos = np.asarray(positions, dtype=float)
    # Steering phase is +j 2pi f p.u/c, equivalent to arrival delay -p.u/c.
    arrival = -(pos @ direction) / SPEED_OF_SOUND
    return arrival[pairs[:, 0]] - arrival[pairs[:, 1]]


def _weighted_huber_cost(residual: np.ndarray, weight: np.ndarray, delta: float) -> float:
    r = np.asarray(residual, dtype=float)
    w = np.asarray(weight, dtype=float)
    a = np.abs(r)
    loss = np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    return float(np.sum(w * loss) / max(np.sum(w), 1e-12))


def gcc_phat_tde_estimate(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: GCCPHATParameters = GCCPHATParameters(),
) -> tuple[float, dict]:
    """All-pair GCC-PHAT followed by a robust far-field azimuth fit."""
    x = np.asarray(audio, dtype=float)
    pos = np.asarray(microphone_positions_m, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(pos):
        raise ValueError("audio must have shape (microphones, samples) matching positions")
    pairs = deterministic_pair_indices(
        pos,
        minimum_baseline_m=params.pair_min_baseline_m,
        maximum_pairs=params.maximum_pairs,
    )
    delays: list[float] = []
    weights: list[float] = []
    diagnostics: list[dict] = []
    baseline_lengths = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    maximum_baseline = max(float(np.max(baseline_lengths)), 1e-12)

    # Compute each channel FFT once, then evaluate all microphone pairs in a
    # single batched inverse FFT.  This is numerically equivalent to repeated
    # pairwise GCC-PHAT but avoids recomputing the same channel FFT for every
    # pair, which is important for the 6,559-row formal run.
    n = int(x.shape[1])
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 32))))
    spectra = np.fft.rfft(x, n=nfft, axis=1)
    frequency = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate_hz))
    frequency_mask = (
        (frequency >= float(params.frequency_min_hz))
        & (frequency <= float(params.frequency_max_hz))
    )
    cross = spectra[pairs[:, 0]] * np.conj(spectra[pairs[:, 1]])
    phat = np.zeros_like(cross)
    magnitude = np.abs(cross)
    phat[:, frequency_mask] = cross[:, frequency_mask] / np.maximum(
        magnitude[:, frequency_mask], 1e-15
    )
    correlations = np.fft.irfft(phat, n=nfft, axis=1)

    for pair_index, ((i, j), baseline) in enumerate(zip(pairs, baseline_lengths)):
        maximum_delay = float(baseline) / SPEED_OF_SOUND + 1.5 / float(sample_rate_hz)
        maximum_lag = min(
            int(np.ceil(maximum_delay * float(sample_rate_hz))) + 1,
            nfft // 2 - 1,
        )
        lags = np.arange(-maximum_lag, maximum_lag + 1, dtype=int)
        correlation = correlations[pair_index]
        values = np.concatenate([
            correlation[-maximum_lag:], correlation[: maximum_lag + 1]
        ])
        peak_index = int(np.argmax(values))
        sub = _parabolic_peak(values, peak_index)
        lag_samples = float(lags[peak_index]) + sub
        delay = lag_samples / float(sample_rate_hz)
        absolute = np.abs(values)
        peak = float(values[peak_index])
        median = float(np.median(absolute))
        exclusion = max(1, int(round(0.00015 * float(sample_rate_hz))))
        keep = np.ones(len(values), dtype=bool)
        keep[
            max(0, peak_index - exclusion): min(len(values), peak_index + exclusion + 1)
        ] = False
        second = float(np.max(values[keep])) if np.any(keep) else 0.0
        diag = {
            "lag_samples": lag_samples,
            "peak": peak,
            "peak_to_median": max(peak, 0.0) / max(median, 1e-12),
            "peak_to_second": peak / max(second, 1e-12) if second > 0 else float("inf"),
            "nfft": int(nfft),
            "maximum_lag_samples": int(maximum_lag),
        }
        confidence = np.log1p(max(float(diag["peak_to_median"]), 0.0))
        confidence = max(confidence, float(params.confidence_floor))
        geometry_weight = np.sqrt(max(float(baseline), 1e-12) / maximum_baseline)
        weight = confidence * geometry_weight
        delays.append(float(delay))
        weights.append(float(weight))
        diagnostics.append({
            "pair_index": int(pair_index),
            "microphone_i": int(i),
            "microphone_j": int(j),
            "baseline_m": float(baseline),
            "delay_s": float(delay),
            "weight": float(weight),
            **diag,
        })
    delays_array = np.asarray(delays, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    grid = np.arange(
        float(params.theta_min_deg),
        float(params.theta_max_deg) + 0.5 * float(params.coarse_step_deg),
        float(params.coarse_step_deg),
    )
    # Two-pass robust scale: a fixed two-sample preliminary Huber fit is
    # followed by a residual-MAD scale estimate.  Both passes are independent
    # of source truth and use the same frozen geometry-only pair bank.
    preliminary_delta = 2.0 / float(sample_rate_hz)
    preliminary_costs = np.asarray([
        _weighted_huber_cost(
            delays_array - _predicted_pair_delays(pos, pairs, theta),
            weight_array,
            preliminary_delta,
        )
        for theta in grid
    ], dtype=float)
    preliminary_theta = float(grid[int(np.argmin(preliminary_costs))])
    preliminary_residual = delays_array - _predicted_pair_delays(pos, pairs, preliminary_theta)
    residual_mad = float(np.median(np.abs(preliminary_residual)))
    delta = max(float(params.huber_delta_scale) * residual_mad, 0.25 / float(sample_rate_hz))
    costs = np.asarray([
        _weighted_huber_cost(delays_array - _predicted_pair_delays(pos, pairs, theta), weight_array, delta)
        for theta in grid
    ], dtype=float)
    coarse_theta = float(grid[int(np.argmin(costs))])
    lo = max(float(params.theta_min_deg), coarse_theta - float(params.refine_radius_deg))
    hi = min(float(params.theta_max_deg), coarse_theta + float(params.refine_radius_deg))
    result = minimize_scalar(
        lambda theta: _weighted_huber_cost(
            delays_array - _predicted_pair_delays(pos, pairs, float(theta)), weight_array, delta
        ),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-3},
    )
    theta_hat = float(result.x if result.success else coarse_theta)
    predicted = _predicted_pair_delays(pos, pairs, theta_hat)
    residual = delays_array - predicted
    return theta_hat, {
        "pair_count": int(len(pairs)),
        "pairs": pairs,
        "pair_diagnostics": diagnostics,
        "coarse_grid_deg": grid,
        "preliminary_coarse_cost": preliminary_costs,
        "preliminary_estimate_deg": preliminary_theta,
        "coarse_cost": costs,
        "coarse_estimate_deg": coarse_theta,
        "refined_estimate_deg": theta_hat,
        "huber_delta_s": float(delta),
        "weighted_rmse_s": float(np.sqrt(np.sum(weight_array * residual * residual) / max(np.sum(weight_array), 1e-12))),
        "median_abs_residual_s": float(np.median(np.abs(residual))),
        "optimizer_success": bool(result.success),
    }


def synthetic_plane_wave(
    positions_m: np.ndarray,
    theta_deg: float,
    *,
    sample_rate_hz: int = 16000,
    duration_s: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic fractional-delay synthetic signal for unit tests."""
    rng = np.random.default_rng(int(seed))
    n = int(round(float(duration_s) * sample_rate_hz))
    base = rng.normal(size=n)
    spectrum = np.fft.rfft(base)
    frequency = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    theta = np.deg2rad(float(theta_deg))
    direction = np.asarray([np.sin(theta), np.cos(theta), 0.0])
    arrival = -(np.asarray(positions_m, dtype=float) @ direction) / SPEED_OF_SOUND
    channels = []
    for delay in arrival:
        shifted = np.fft.irfft(spectrum * np.exp(-2j * np.pi * frequency * float(delay)), n=n)
        channels.append(shifted)
    return np.asarray(channels, dtype=float)
