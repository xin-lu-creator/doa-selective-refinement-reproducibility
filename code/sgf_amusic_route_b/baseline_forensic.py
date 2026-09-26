# =========================================================
# File        : baseline_forensic.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the baseline forensic module used by the reproducibility workflow.
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
"""Baseline-dominance forensic primitives for V52D-R4.

This module does not propose a new DOA estimator.  It decomposes the frozen
GCC-PHAT/TDE-Huber endpoint into a preregistered 2^4 factorial over four
components: sub-sample peak refinement, confidence weighting, geometry
weighting, and adaptive Huber loss.  Every variant consumes one shared PHAT
pair bank, so differences cannot be attributed to hidden preprocessing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
from scipy.optimize import minimize_scalar

from .baseline_extension import (
    GCCPHATParameters,
    SPEED_OF_SOUND,
    _parabolic_peak,
    _predicted_pair_delays,
    _weighted_huber_cost,
    deterministic_pair_indices,
    gcc_phat_tde_estimate,
)

_EPS = 1.0e-15


@dataclass(frozen=True)
class BaselineForensicParameters:
    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 3500.0
    theta_min_deg: float = -80.0
    theta_max_deg: float = 80.0
    coarse_step_deg: float = 1.0
    refine_radius_deg: float = 1.5
    pair_min_baseline_m: float = 0.05
    confidence_floor: float = 0.05
    maximum_pairs: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _hash_array(value: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii"))
    h.update(str(tuple(x.shape)).encode("ascii"))
    h.update(x.tobytes())
    return h.hexdigest()


def method_name(subsample: bool, confidence: bool, geometry: bool, huber: bool) -> str:
    if subsample and confidence and geometry and huber:
        return "gcc_phat_tde_huber"
    return "gcc_%s_%s_%s_%s" % (
        "frac" if subsample else "int",
        "conf" if confidence else "equal",
        "geom" if geometry else "nogeom",
        "huber" if huber else "l2",
    )


def factorial_method_spec() -> list[dict]:
    rows: list[dict] = []
    for subsample in (False, True):
        for confidence in (False, True):
            for geometry in (False, True):
                for huber in (False, True):
                    rows.append({
                        "method": method_name(subsample, confidence, geometry, huber),
                        "subsample": bool(subsample),
                        "confidence_weight": bool(confidence),
                        "geometry_weight": bool(geometry),
                        "adaptive_huber": bool(huber),
                    })
    return rows


FACTORIAL_METHODS = tuple(row["method"] for row in factorial_method_spec())


def extract_shared_pair_bank(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: BaselineForensicParameters = BaselineForensicParameters(),
) -> dict:
    x = np.asarray(audio, dtype=float)
    positions = np.asarray(microphone_positions_m, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(positions):
        raise ValueError("audio must be microphone-by-sample and match geometry")
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

    baselines = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    max_baseline = max(float(np.max(baselines)), _EPS)
    integer_lags: list[float] = []
    fractional_lags: list[float] = []
    confidence_weights: list[float] = []
    geometry_weights: list[float] = []
    pair_rows: list[dict] = []
    for k, ((i, j), baseline) in enumerate(zip(pairs, baselines)):
        maximum_delay = float(baseline) / SPEED_OF_SOUND + 1.5 / float(sample_rate_hz)
        maximum_lag = min(
            int(np.ceil(maximum_delay * float(sample_rate_hz))) + 1,
            nfft // 2 - 1,
        )
        lags = np.arange(-maximum_lag, maximum_lag + 1, dtype=int)
        corr = correlations[k]
        values = np.concatenate([corr[-maximum_lag:], corr[: maximum_lag + 1]])
        peak_index = int(np.argmax(values))
        integer = float(lags[peak_index])
        fractional = integer + _parabolic_peak(values, peak_index)
        absolute = np.abs(values)
        peak = float(values[peak_index])
        median = float(np.median(absolute))
        peak_to_median = max(peak, 0.0) / max(median, 1.0e-12)
        conf = max(np.log1p(max(peak_to_median, 0.0)), float(params.confidence_floor))
        geom = np.sqrt(max(float(baseline), _EPS) / max_baseline)

        shifted = np.fft.fftshift(corr)
        shifted_lags = np.arange(-nfft // 2, nfft // 2, dtype=int)
        unrestricted_index = int(np.argmax(shifted))
        unrestricted_lag = int(shifted_lags[unrestricted_index])
        outside = abs(unrestricted_lag) > maximum_lag

        integer_lags.append(integer)
        fractional_lags.append(fractional)
        confidence_weights.append(float(conf))
        geometry_weights.append(float(geom))
        pair_rows.append({
            "pair_index": int(k),
            "microphone_i": int(i),
            "microphone_j": int(j),
            "baseline_m": float(baseline),
            "maximum_lag_samples": int(maximum_lag),
            "unrestricted_peak_lag_samples": int(unrestricted_lag),
            "unrestricted_peak_outside_physical_range": bool(outside),
            "integer_peak_lag_samples": float(integer),
            "fractional_peak_lag_samples": float(fractional),
            "subsample_offset_samples": float(fractional - integer),
            "peak": peak,
            "peak_to_median": float(peak_to_median),
            "confidence_weight": float(conf),
            "geometry_weight": float(geom),
            "full_weight": float(conf * geom),
        })

    return {
        "audio": x,
        "positions": positions,
        "pairs": pairs,
        "nfft": int(nfft),
        "frequency_hz": frequency,
        "frequency_mask": mask,
        "phat": phat,
        "correlations": correlations,
        "integer_delays_s": np.asarray(integer_lags, dtype=float) / float(sample_rate_hz),
        "fractional_delays_s": np.asarray(fractional_lags, dtype=float) / float(sample_rate_hz),
        "confidence_weights": np.asarray(confidence_weights, dtype=float),
        "geometry_weights": np.asarray(geometry_weights, dtype=float),
        "pair_rows": pair_rows,
        "fingerprints": {
            "audio_sha256": _hash_array(x),
            "microphone_positions_sha256": _hash_array(positions),
            "pair_indices_sha256": _hash_array(pairs),
            "phat_sha256": _hash_array(phat),
            "correlations_sha256": _hash_array(correlations),
        },
    }


def _l2_cost(residual: np.ndarray, weight: np.ndarray) -> float:
    r = np.asarray(residual, dtype=float)
    w = np.asarray(weight, dtype=float)
    return float(np.sum(w * r * r) / max(np.sum(w), _EPS))


def _fit_grid_and_refine(
    delays_s: np.ndarray,
    weights: np.ndarray,
    positions: np.ndarray,
    pairs: np.ndarray,
    sample_rate_hz: float,
    *,
    huber: bool,
    params: BaselineForensicParameters,
) -> tuple[float, dict]:
    grid = np.arange(
        float(params.theta_min_deg),
        float(params.theta_max_deg) + 0.5 * float(params.coarse_step_deg),
        float(params.coarse_step_deg),
    )
    if huber:
        preliminary_delta = 2.0 / float(sample_rate_hz)
        preliminary_costs = np.asarray([
            _weighted_huber_cost(
                delays_s - _predicted_pair_delays(positions, pairs, theta),
                weights,
                preliminary_delta,
            )
            for theta in grid
        ])
        preliminary_theta = float(grid[int(np.argmin(preliminary_costs))])
        preliminary_residual = delays_s - _predicted_pair_delays(
            positions, pairs, preliminary_theta
        )
        residual_mad = float(np.median(np.abs(preliminary_residual)))
        delta = max(1.5 * residual_mad, 0.25 / float(sample_rate_hz))
        cost_fn = lambda theta: _weighted_huber_cost(
            delays_s - _predicted_pair_delays(positions, pairs, float(theta)),
            weights,
            delta,
        )
    else:
        preliminary_theta = float("nan")
        delta = float("nan")
        preliminary_costs = np.asarray([], dtype=float)
        cost_fn = lambda theta: _l2_cost(
            delays_s - _predicted_pair_delays(positions, pairs, float(theta)), weights
        )
    costs = np.asarray([cost_fn(theta) for theta in grid], dtype=float)
    coarse = float(grid[int(np.argmin(costs))])
    lo = max(float(params.theta_min_deg), coarse - float(params.refine_radius_deg))
    hi = min(float(params.theta_max_deg), coarse + float(params.refine_radius_deg))
    result = minimize_scalar(
        cost_fn,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1.0e-3},
    )
    estimate = float(result.x if result.success else coarse)
    residual = delays_s - _predicted_pair_delays(positions, pairs, estimate)
    return estimate, {
        "preliminary_estimate_deg": preliminary_theta,
        "coarse_estimate_deg": coarse,
        "refined_estimate_deg": estimate,
        "huber_delta_s": float(delta),
        "weighted_rmse_s": float(
            np.sqrt(np.sum(weights * residual * residual) / max(np.sum(weights), _EPS))
        ),
        "median_abs_residual_s": float(np.median(np.abs(residual))),
        "optimizer_success": bool(result.success),
        "coarse_grid_deg": grid,
        "coarse_cost": costs,
        "preliminary_coarse_cost": preliminary_costs,
    }


def baseline_factorial_estimates(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: BaselineForensicParameters = BaselineForensicParameters(),
) -> tuple[dict[str, float], dict]:
    bank = extract_shared_pair_bank(
        audio, sample_rate_hz, microphone_positions_m, params=params
    )
    estimates: dict[str, float] = {}
    method_diagnostics: dict[str, dict] = {}
    for spec in factorial_method_spec():
        delays = (
            bank["fractional_delays_s"]
            if spec["subsample"]
            else bank["integer_delays_s"]
        )
        weights = np.ones(len(bank["pairs"]), dtype=float)
        if spec["confidence_weight"]:
            weights *= bank["confidence_weights"]
        if spec["geometry_weight"]:
            weights *= bank["geometry_weights"]
        estimate, diag = _fit_grid_and_refine(
            delays,
            weights,
            bank["positions"],
            bank["pairs"],
            sample_rate_hz,
            huber=spec["adaptive_huber"],
            params=params,
        )
        estimates[spec["method"]] = float(estimate)
        method_diagnostics[spec["method"]] = {
            **spec,
            **diag,
            "weight_sum": float(np.sum(weights)),
            "weight_min": float(np.min(weights)),
            "weight_median": float(np.median(weights)),
            "weight_max": float(np.max(weights)),
        }

    frozen_params = GCCPHATParameters(
        frequency_min_hz=params.frequency_min_hz,
        frequency_max_hz=params.frequency_max_hz,
        theta_min_deg=params.theta_min_deg,
        theta_max_deg=params.theta_max_deg,
        coarse_step_deg=params.coarse_step_deg,
        refine_radius_deg=params.refine_radius_deg,
        pair_min_baseline_m=params.pair_min_baseline_m,
        confidence_floor=params.confidence_floor,
        maximum_pairs=params.maximum_pairs,
    )
    frozen, frozen_diag = gcc_phat_tde_estimate(
        audio,
        sample_rate_hz,
        microphone_positions_m,
        params=frozen_params,
    )
    delta = abs(float(estimates["gcc_phat_tde_huber"]) - float(frozen))
    if delta > 2.0e-6:
        raise RuntimeError(
            f"Shared-bank frozen corner mismatch: {delta:.9g} deg"
        )
    estimates["gcc_phat_tde_huber"] = float(frozen)
    outside_fraction = float(np.mean([
        row["unrestricted_peak_outside_physical_range"] for row in bank["pair_rows"]
    ]))
    return estimates, {
        "parameters": params.as_dict(),
        "factorial_spec": factorial_method_spec(),
        "method_diagnostics": method_diagnostics,
        "frozen_baseline_diagnostics": frozen_diag,
        "frozen_corner_reproduction_delta_deg": float(delta),
        "pair_count": int(len(bank["pairs"])),
        "nfft": int(bank["nfft"]),
        "frequency_bin_count": int(np.sum(bank["frequency_mask"])),
        "unrestricted_peak_outside_physical_fraction": outside_fraction,
        "median_abs_subsample_offset_samples": float(np.median(np.abs([
            row["subsample_offset_samples"] for row in bank["pair_rows"]
        ]))),
        "fingerprints": dict(bank["fingerprints"]),
        "pair_rows": bank["pair_rows"],
        "shared_pair_bank": bank,
    }
