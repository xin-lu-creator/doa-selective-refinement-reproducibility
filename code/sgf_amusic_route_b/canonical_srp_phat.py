# =========================================================
# File        : canonical_srp_phat.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the canonical srp phat module used by the reproducibility workflow.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - In-memory estimates, state objects, or helper values returned to calling code.
#
# Used in paper:
#   - SRP-PHAT comparison reported in the manuscript and Supplementary Material.
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
"""Canonical single-band SRP-PHAT formulation audit.

This module is intentionally narrow.  It evaluates the standard all-pair
band-limited SRP-PHAT objective on the same deterministic microphone-pair bank
and azimuth convention used by the frozen GCC-PHAT/TDE baseline.  It contains
no pair-quality weighting, robust per-pair standardization, Hodge weighting,
frequency-band renormalization, temporal context, Mandala, or source truth.

Two numerically related implementations are exposed:

* ``canonical_srp_phat_linear`` samples the circular GCC-PHAT at the predicted
  fractional delay using periodic linear interpolation.
* ``canonical_srp_phat_nearest`` samples the same GCC-PHAT at the nearest
  integer delay.  It is an audit control for delay-grid quantization, not a new
  scientific candidate.

A small direct frequency-domain reference is also provided for synthetic and
unit-test equivalence checks.  It is deliberately not used for the full LOCATA
run because direct pair-frequency-angle summation is unnecessarily expensive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .baseline_extension import (
    _parabolic_peak,
    _predicted_pair_delays,
    deterministic_pair_indices,
    gcc_phat_tde_estimate,
)

_EPS = 1.0e-15


@dataclass(frozen=True)
class CanonicalSRPParameters:
    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 3500.0
    theta_min_deg: float = -80.0
    theta_max_deg: float = 80.0
    theta_step_deg: float = 0.25
    pair_min_baseline_m: float = 0.05
    maximum_pairs: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalSRPResult:
    estimates_deg: dict[str, float]
    diagnostics: dict


def _normalize_shape(values: np.ndarray) -> np.ndarray:
    """Zero-center and unit-norm a spectrum for shape-only comparison."""
    x = np.asarray(values, dtype=float)
    centered = x - float(np.mean(x))
    norm = float(np.linalg.norm(centered))
    if norm <= _EPS:
        return np.zeros_like(centered)
    return centered / norm


def _peak_margin(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return 0.0
    top = np.partition(x, -2)[-2:]
    return float(np.max(top) - np.min(top))


def _estimate_from_spectrum(grid_deg: np.ndarray, spectrum: np.ndarray) -> float:
    grid = np.asarray(grid_deg, dtype=float)
    score = np.asarray(spectrum, dtype=float)
    index = int(np.argmax(score))
    if index <= 0 or index >= len(score) - 1:
        return float(grid[index])
    offset = _parabolic_peak(score, index)
    step = float(grid[1] - grid[0])
    return float(np.clip(grid[index] + offset * step, grid[0], grid[-1]))


def _interpolate_periodic_rows(
    correlations: np.ndarray,
    lag_samples: np.ndarray,
) -> np.ndarray:
    """Sample each periodic correlation row at pair-specific fractional lags."""
    corr = np.asarray(correlations, dtype=float)
    lag = np.asarray(lag_samples, dtype=float)
    if corr.ndim != 2 or lag.ndim != 2 or corr.shape[0] != lag.shape[0]:
        raise ValueError("correlations and lag_samples must be pair-by-* arrays")
    lower = np.floor(lag).astype(np.int64)
    fraction = lag - lower
    row = np.arange(corr.shape[0], dtype=np.int64)[:, None]
    i0 = np.mod(lower, corr.shape[1])
    i1 = np.mod(lower + 1, corr.shape[1])
    return (1.0 - fraction) * corr[row, i0] + fraction * corr[row, i1]


def _nearest_periodic_rows(
    correlations: np.ndarray,
    lag_samples: np.ndarray,
) -> np.ndarray:
    corr = np.asarray(correlations, dtype=float)
    lag = np.asarray(lag_samples, dtype=float)
    if corr.ndim != 2 or lag.ndim != 2 or corr.shape[0] != lag.shape[0]:
        raise ValueError("correlations and lag_samples must be pair-by-* arrays")
    row = np.arange(corr.shape[0], dtype=np.int64)[:, None]
    index = np.mod(np.rint(lag).astype(np.int64), corr.shape[1])
    return corr[row, index]


def _prepare_phat(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    params: CanonicalSRPParameters,
) -> dict:
    x = np.asarray(audio, dtype=float)
    positions = np.asarray(microphone_positions_m, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(positions):
        raise ValueError("audio must have shape (microphones, samples) matching positions")
    pairs = deterministic_pair_indices(
        positions,
        minimum_baseline_m=params.pair_min_baseline_m,
        maximum_pairs=params.maximum_pairs,
    )
    n = int(x.shape[1])
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 32))))
    spectra = np.fft.rfft(x, n=nfft, axis=1)
    frequency = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate_hz))
    mask = (
        (frequency >= float(params.frequency_min_hz))
        & (frequency <= float(params.frequency_max_hz))
    )
    cross = spectra[pairs[:, 0]] * np.conj(spectra[pairs[:, 1]])
    phat = np.zeros_like(cross)
    phat[:, mask] = cross[:, mask] / np.maximum(np.abs(cross[:, mask]), _EPS)
    correlations = np.fft.irfft(phat, n=nfft, axis=1).real
    return {
        "pairs": pairs,
        "nfft": int(nfft),
        "frequency_hz": frequency,
        "frequency_mask": mask,
        "phat": phat,
        "correlations": correlations,
    }


def direct_frequency_srp_spectrum(
    phat: np.ndarray,
    frequency_hz: np.ndarray,
    frequency_mask: np.ndarray,
    predicted_delays_s: np.ndarray,
    *,
    angle_chunk_size: int = 32,
) -> np.ndarray:
    """Direct standard SRP-PHAT frequency sum for small audit problems.

    The returned spectrum omits angle-independent constants.  For integer
    sample delays and a band that excludes DC/Nyquist, it is proportional to
    the corresponding IFFT-GCC lookup by exactly ``nfft/2``.
    """
    P = np.asarray(phat)
    frequency = np.asarray(frequency_hz, dtype=float)
    mask = np.asarray(frequency_mask, dtype=bool)
    delays = np.asarray(predicted_delays_s, dtype=float)
    if P.ndim != 2 or delays.ndim != 2 or P.shape[0] != delays.shape[0]:
        raise ValueError("phat and predicted_delays_s must be pair-by-* arrays")
    indices = np.flatnonzero(mask)
    score = np.zeros(delays.shape[1], dtype=float)
    for pair_index in range(P.shape[0]):
        coefficients = P[pair_index, indices]
        band_frequency = frequency[indices]
        for start in range(0, delays.shape[1], max(1, int(angle_chunk_size))):
            stop = min(delays.shape[1], start + max(1, int(angle_chunk_size)))
            phase = np.exp(
                1j * 2.0 * np.pi * band_frequency[:, None]
                * delays[pair_index, start:stop][None, :]
            )
            score[start:stop] += np.sum(
                np.real(coefficients[:, None] * phase), axis=0
            )
    return score


def canonical_srp_phat_estimates(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: CanonicalSRPParameters = CanonicalSRPParameters(),
) -> CanonicalSRPResult:
    """Return frozen GCC baseline and two canonical SRP-PHAT audit controls."""
    baseline, baseline_diagnostics = gcc_phat_tde_estimate(
        audio,
        sample_rate_hz,
        microphone_positions_m,
    )
    prepared = _prepare_phat(audio, sample_rate_hz, microphone_positions_m, params)
    pairs = prepared["pairs"]
    grid = np.arange(
        float(params.theta_min_deg),
        float(params.theta_max_deg) + 0.5 * float(params.theta_step_deg),
        float(params.theta_step_deg),
        dtype=float,
    )
    predicted_delays_s = np.column_stack([
        _predicted_pair_delays(microphone_positions_m, pairs, theta)
        for theta in grid
    ])
    predicted_lag_samples = predicted_delays_s * float(sample_rate_hz)
    linear_pair_support = _interpolate_periodic_rows(
        prepared["correlations"], predicted_lag_samples
    )
    nearest_pair_support = _nearest_periodic_rows(
        prepared["correlations"], predicted_lag_samples
    )
    # Canonical SRP-PHAT uses an unweighted sum over microphone pairs.  Division
    # by pair count would be angle-independent and therefore is omitted.
    linear_spectrum = np.sum(linear_pair_support, axis=0)
    nearest_spectrum = np.sum(nearest_pair_support, axis=0)
    linear_estimate = _estimate_from_spectrum(grid, linear_spectrum)
    nearest_estimate = _estimate_from_spectrum(grid, nearest_spectrum)
    linear_shape = _normalize_shape(linear_spectrum)
    nearest_shape = _normalize_shape(nearest_spectrum)
    shape_correlation = float(np.dot(linear_shape, nearest_shape))
    return CanonicalSRPResult(
        estimates_deg={
            "gcc_phat_tde_huber": float(baseline),
            "canonical_srp_phat_linear": float(linear_estimate),
            "canonical_srp_phat_nearest": float(nearest_estimate),
        },
        diagnostics={
            "parameters": params.as_dict(),
            "pair_count": int(len(pairs)),
            "nfft": int(prepared["nfft"]),
            "frequency_bin_count": int(np.sum(prepared["frequency_mask"])),
            "grid_count": int(len(grid)),
            "baseline": baseline_diagnostics,
            "linear_peak_margin": _peak_margin(linear_spectrum),
            "nearest_peak_margin": _peak_margin(nearest_spectrum),
            "linear_nearest_spectrum_shape_correlation": shape_correlation,
            "linear_nearest_estimate_delta_deg": float(
                abs((linear_estimate - nearest_estimate + 180.0) % 360.0 - 180.0)
            ),
            "pair_weights_used": False,
            "pairwise_standardization_used": False,
            "multiband_fusion_used": False,
            "hodge_used": False,
            "pair_mandala_used": False,
            "temporal_context_used": False,
            "truth_used": False,
        },
    )
