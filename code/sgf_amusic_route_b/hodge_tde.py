# =========================================================
# File        : hodge_tde.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the hodge tde module used by the reproducibility workflow.
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
"""Top-K GCC-PHAT peak reassignment by weighted graph-Hodge consistency.

V52B-R1/B0 is a deliberately small, truth-free Go/No-Go implementation.
It keeps the frozen GCC-PHAT/TDE + Huber estimator unchanged as the baseline,
retains up to K physically admissible GCC peaks per microphone pair, and uses
weighted projection onto the complete pair graph's gradient subspace to
reassign peaks jointly.  The final azimuth is fitted directly in edge space.

No Mandala weighting, coherence weighting, near-field range search, tracking,
or source truth is used in B0.  Those mechanisms are intentionally deferred
until the multi-peak Hodge core itself passes the preregistered smoke gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

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
class HodgeTDEParameters:
    """Frozen B0 parameters, fixed before any V52B-R1 LOCATA result."""

    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 3500.0
    theta_min_deg: float = -80.0
    theta_max_deg: float = 80.0
    coarse_step_deg: float = 1.0
    refine_radius_deg: float = 1.5
    pair_min_baseline_m: float = 0.05
    maximum_pairs: int = 0
    top_k: int = 3
    minimum_peak_separation_s: float = 0.00015
    amplitude_penalty: float = 0.20
    iterations: int = 4
    quality_peak_to_median_cap: float = 100.0
    quality_floor: float = 0.05
    huber_delta_scale: float = 1.345
    residual_scale_floor_samples: float = 0.25
    safe_cycle_improvement_fraction: float = 0.10
    safe_max_angle_change_deg: float = 15.0
    safe_min_selected_amplitude: float = 0.40
    safe_min_effective_pairs: float = 8.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HodgeTDEResult:
    estimates_deg: dict[str, float]
    diagnostics: dict


def _incidence_matrix(pairs: np.ndarray, microphone_count: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=int)
    B = np.zeros((len(pairs), int(microphone_count)), dtype=float)
    B[np.arange(len(pairs)), pairs[:, 0]] = 1.0
    B[np.arange(len(pairs)), pairs[:, 1]] = -1.0
    return B


def _weighted_potential(B: np.ndarray, delays_s: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Solve min ||W^(1/2)(delay-Bt)|| with gauge t[0]=0."""
    B = np.asarray(B, dtype=float)
    y = np.asarray(delays_s, dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float), _EPS)
    if B.ndim != 2 or len(y) != B.shape[0] or len(w) != B.shape[0]:
        raise ValueError("Incompatible Hodge inputs")
    design = B[:, 1:]
    root = np.sqrt(w)
    solution, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    t = np.zeros(B.shape[1], dtype=float)
    t[1:] = solution
    return t


def _weighted_cycle_ratio(
    B: np.ndarray,
    delays_s: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    t = _weighted_potential(B, delays_s, weights)
    residual = np.asarray(delays_s, dtype=float) - B @ t
    w = np.maximum(np.asarray(weights, dtype=float), _EPS)
    numerator = float(np.sum(w * residual * residual))
    denominator = float(np.sum(w * np.asarray(delays_s, dtype=float) ** 2))
    ratio = numerator / max(denominator, _EPS)
    return float(np.clip(ratio, 0.0, 1.0)), t, residual


def _robust_scale(residual_s: np.ndarray, sample_rate_hz: float, floor_samples: float) -> float:
    r = np.asarray(residual_s, dtype=float)
    center = float(np.median(r)) if len(r) else 0.0
    mad = float(np.median(np.abs(r - center))) if len(r) else 0.0
    return max(1.4826 * mad, float(floor_samples) / float(sample_rate_hz))


def _effective_count(weights: Sequence[float]) -> float:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(np.sum(w))
    if total <= _EPS:
        return 0.0
    return total * total / max(float(np.sum(w * w)), _EPS)


def _bounded_pair_quality(
    peak_to_median: float,
    baseline_m: float,
    maximum_baseline_m: float,
    params: HodgeTDEParameters,
) -> float:
    cap = max(float(params.quality_peak_to_median_cap), 1.0)
    confidence = np.log1p(max(float(peak_to_median), 0.0)) / np.log1p(cap)
    confidence = float(np.clip(confidence, params.quality_floor, 1.0))
    geometry = np.sqrt(max(float(baseline_m), _EPS) / max(float(maximum_baseline_m), _EPS))
    return float(np.clip(confidence * geometry, params.quality_floor * geometry, 1.0))


def _candidate_peaks(
    values: np.ndarray,
    lags: np.ndarray,
    sample_rate_hz: float,
    top_k: int,
    minimum_separation_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-size candidate delays, normalized amplitudes, validity."""
    y = np.asarray(values, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if len(y) != len(lags) or len(y) == 0:
        raise ValueError("Invalid GCC candidate grid")
    separation = max(1, int(round(float(minimum_separation_s) * float(sample_rate_hz))))
    local, _ = find_peaks(y, distance=separation)
    global_index = int(np.argmax(y))
    indices = [global_index]
    ranked = sorted((int(i) for i in local if int(i) != global_index), key=lambda i: (-y[i], i))
    for index in ranked:
        if all(abs(index - chosen) >= separation for chosen in indices):
            indices.append(index)
        if len(indices) >= int(top_k):
            break
    # If fewer local maxima exist, use next-largest well-separated samples.
    if len(indices) < int(top_k):
        ranked_all = np.argsort(-y, kind="mergesort")
        for raw in ranked_all:
            index = int(raw)
            if all(abs(index - chosen) >= separation for chosen in indices):
                indices.append(index)
            if len(indices) >= int(top_k):
                break
    count = min(len(indices), int(top_k))
    delays = np.full(int(top_k), np.nan, dtype=float)
    amplitudes = np.zeros(int(top_k), dtype=float)
    valid = np.zeros(int(top_k), dtype=bool)
    peak_value = max(float(y[global_index]), _EPS)
    for slot, index in enumerate(indices[:count]):
        sub = _parabolic_peak(y, index)
        delays[slot] = (float(lags[index]) + sub) / float(sample_rate_hz)
        amplitudes[slot] = float(np.clip(max(float(y[index]), 0.0) / peak_value, _EPS, 1.0))
        valid[slot] = True
    return delays, amplitudes, valid


def extract_gcc_peak_bank(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: HodgeTDEParameters = HodgeTDEParameters(),
) -> dict:
    """Compute all-pair plain-PHAT top-K candidate bank once per window."""
    x = np.asarray(audio, dtype=float)
    pos = np.asarray(microphone_positions_m, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(pos):
        raise ValueError("audio must have shape (microphones, samples) matching positions")
    pairs = deterministic_pair_indices(
        pos,
        minimum_baseline_m=params.pair_min_baseline_m,
        maximum_pairs=params.maximum_pairs,
    )
    baselines = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    maximum_baseline = max(float(np.max(baselines)), _EPS)
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
    correlation = np.fft.irfft(phat, n=nfft, axis=1)

    K = int(params.top_k)
    candidate_delays = np.full((len(pairs), K), np.nan, dtype=float)
    candidate_amplitudes = np.zeros((len(pairs), K), dtype=float)
    candidate_valid = np.zeros((len(pairs), K), dtype=bool)
    qualities = np.zeros(len(pairs), dtype=float)
    rows: list[dict] = []

    for pair_index, ((i, j), baseline) in enumerate(zip(pairs, baselines)):
        maximum_delay = float(baseline) / SPEED_OF_SOUND + 1.5 / float(sample_rate_hz)
        maximum_lag = min(
            int(np.ceil(maximum_delay * float(sample_rate_hz))) + 1,
            nfft // 2 - 1,
        )
        lags = np.arange(-maximum_lag, maximum_lag + 1, dtype=int)
        values = np.concatenate([
            correlation[pair_index, -maximum_lag:],
            correlation[pair_index, : maximum_lag + 1],
        ])
        delays, amplitudes, valid = _candidate_peaks(
            values,
            lags,
            sample_rate_hz,
            K,
            params.minimum_peak_separation_s,
        )
        candidate_delays[pair_index] = delays
        candidate_amplitudes[pair_index] = amplitudes
        candidate_valid[pair_index] = valid
        absolute = np.abs(values)
        global_index = int(np.argmax(values))
        peak = float(values[global_index])
        median = float(np.median(absolute))
        peak_to_median = max(peak, 0.0) / max(median, 1e-12)
        quality = _bounded_pair_quality(
            peak_to_median,
            float(baseline),
            maximum_baseline,
            params,
        )
        qualities[pair_index] = quality
        rows.append({
            "pair_index": int(pair_index),
            "microphone_i": int(i),
            "microphone_j": int(j),
            "baseline_m": float(baseline),
            "peak_to_median": float(peak_to_median),
            "bounded_quality": float(quality),
            "candidate_delays_s": delays.tolist(),
            "candidate_amplitudes": amplitudes.tolist(),
            "candidate_valid": valid.tolist(),
        })
    return {
        "pairs": pairs,
        "baselines_m": baselines,
        "candidate_delays_s": candidate_delays,
        "candidate_amplitudes": candidate_amplitudes,
        "candidate_valid": candidate_valid,
        "base_quality": qualities,
        "pair_rows": rows,
        "nfft": int(nfft),
    }


def reassign_topk_hodge(
    candidate_delays_s: np.ndarray,
    candidate_amplitudes: np.ndarray,
    candidate_valid: np.ndarray,
    pairs: np.ndarray,
    microphone_count: int,
    sample_rate_hz: float,
    base_quality: np.ndarray,
    *,
    initial_predicted_delays_s: np.ndarray | None = None,
    params: HodgeTDEParameters = HodgeTDEParameters(),
) -> dict:
    """Jointly reassign top-K pair peaks by Hodge consistency and IRLS."""
    delays = np.asarray(candidate_delays_s, dtype=float)
    amplitudes = np.asarray(candidate_amplitudes, dtype=float)
    valid = np.asarray(candidate_valid, dtype=bool)
    pairs = np.asarray(pairs, dtype=int)
    quality = np.clip(np.asarray(base_quality, dtype=float), _EPS, 1.0)
    if delays.shape != amplitudes.shape or delays.shape != valid.shape:
        raise ValueError("Candidate arrays must have identical shapes")
    if delays.shape[0] != len(pairs) or len(quality) != len(pairs):
        raise ValueError("Candidate bank and pair bank have incompatible sizes")
    if not np.all(valid[:, 0]):
        raise ValueError("Every pair must have a valid primary peak")

    B = _incidence_matrix(pairs, int(microphone_count))
    assignment = np.zeros(len(pairs), dtype=int)
    if initial_predicted_delays_s is not None:
        initial = np.asarray(initial_predicted_delays_s, dtype=float).reshape(-1)
        if len(initial) != len(pairs):
            raise ValueError("initial_predicted_delays_s has incompatible length")
        initial_scale = max(1.0 / float(sample_rate_hz), _EPS)
        initial_objective = np.full(delays.shape, np.inf, dtype=float)
        for edge in range(len(pairs)):
            for peak in range(delays.shape[1]):
                if valid[edge, peak]:
                    initial_objective[edge, peak] = (
                        abs(delays[edge, peak] - initial[edge]) / initial_scale
                        + float(params.amplitude_penalty)
                        * (-np.log(max(amplitudes[edge, peak], _EPS)))
                    )
        assignment = np.argmin(initial_objective, axis=1).astype(int)
    weights = quality.copy()
    iteration_rows: list[dict] = []

    top1 = delays[:, 0].copy()
    top1_cycle_ratio, _, top1_residual = _weighted_cycle_ratio(B, top1, weights)

    for iteration in range(int(params.iterations)):
        selected = delays[np.arange(len(pairs)), assignment]
        potential = _weighted_potential(B, selected, weights)
        predicted = B @ potential
        residual_before = selected - predicted
        scale = _robust_scale(
            residual_before,
            sample_rate_hz,
            params.residual_scale_floor_samples,
        )
        objective = np.full(delays.shape, np.inf, dtype=float)
        # Evaluate the candidate assignment objective explicitly by edge and peak.
        # This avoids any flattened-index ambiguity and keeps the implementation
        # aligned with the preregistered equation.
        for edge in range(len(pairs)):
            for peak in range(delays.shape[1]):
                if valid[edge, peak]:
                    objective[edge, peak] = (
                        abs(delays[edge, peak] - predicted[edge]) / max(scale, _EPS)
                        + float(params.amplitude_penalty)
                        * (-np.log(max(amplitudes[edge, peak], _EPS)))
                    )
        new_assignment = np.argmin(objective, axis=1).astype(int)
        selected = delays[np.arange(len(pairs)), new_assignment]
        potential_after = _weighted_potential(B, selected, weights)
        residual = selected - B @ potential_after
        scale_after = _robust_scale(
            residual,
            sample_rate_hz,
            params.residual_scale_floor_samples,
        )
        delta = float(params.huber_delta_scale) * scale_after
        huber = np.minimum(1.0, delta / np.maximum(np.abs(residual), _EPS))
        new_weights = quality * huber
        new_weights = np.maximum(new_weights, _EPS)
        new_weights /= max(float(np.max(new_weights)), _EPS)
        iteration_rows.append({
            "iteration": int(iteration + 1),
            "assignment_changes": int(np.sum(new_assignment != assignment)),
            "residual_scale_s": float(scale_after),
            "huber_delta_s": float(delta),
            "weighted_cycle_ratio": float(
                _weighted_cycle_ratio(B, selected, new_weights)[0]
            ),
            "effective_pair_count": float(_effective_count(new_weights)),
        })
        assignment = new_assignment
        weights = new_weights

    # Mandatory final P step after the final W update.
    selected = delays[np.arange(len(pairs)), assignment]
    final_cycle_ratio, final_potential, final_residual = _weighted_cycle_ratio(
        B,
        selected,
        weights,
    )
    # A safety gate must not credit an apparent cycle improvement that is caused
    # only by downweighting the inconsistent edges.  Therefore the top-1 and
    # final assignments are also compared under the same frozen base-quality
    # weights.  The common-weight ratio is the gate statistic; the final-IRLS
    # ratio remains a descriptive diagnostic.
    top1_cycle_ratio_common, _, _ = _weighted_cycle_ratio(B, top1, quality)
    final_cycle_ratio_common, _, _ = _weighted_cycle_ratio(B, selected, quality)
    selected_amplitudes = amplitudes[np.arange(len(pairs)), assignment]
    return {
        "assignment": assignment,
        "selected_delays_s": selected,
        "selected_amplitudes": selected_amplitudes,
        "weights": weights,
        "potential_s": final_potential,
        "cycle_residual_s": final_residual,
        "top1_cycle_residual_s": top1_residual,
        "top1_cycle_ratio": float(top1_cycle_ratio_common),
        "final_cycle_ratio": float(final_cycle_ratio_common),
        "final_cycle_ratio_irls_weights": float(final_cycle_ratio),
        "assignment_changes_from_top1": int(np.sum(assignment != 0)),
        "median_selected_amplitude": float(np.median(selected_amplitudes)),
        "effective_pair_count": float(_effective_count(weights)),
        "iterations": iteration_rows,
    }


def _edge_space_azimuth_fit(
    positions: np.ndarray,
    pairs: np.ndarray,
    delays_s: np.ndarray,
    weights: np.ndarray,
    sample_rate_hz: float,
    params: HodgeTDEParameters,
) -> tuple[float, dict]:
    grid = np.arange(
        float(params.theta_min_deg),
        float(params.theta_max_deg) + 0.5 * float(params.coarse_step_deg),
        float(params.coarse_step_deg),
    )
    preliminary_delta = 2.0 / float(sample_rate_hz)
    preliminary = np.asarray([
        _weighted_huber_cost(
            delays_s - _predicted_pair_delays(positions, pairs, theta),
            weights,
            preliminary_delta,
        )
        for theta in grid
    ])
    preliminary_theta = float(grid[int(np.argmin(preliminary))])
    preliminary_residual = delays_s - _predicted_pair_delays(
        positions,
        pairs,
        preliminary_theta,
    )
    scale = _robust_scale(
        preliminary_residual,
        sample_rate_hz,
        params.residual_scale_floor_samples,
    )
    delta = max(float(params.huber_delta_scale) * scale, params.residual_scale_floor_samples / sample_rate_hz)
    costs = np.asarray([
        _weighted_huber_cost(
            delays_s - _predicted_pair_delays(positions, pairs, theta),
            weights,
            delta,
        )
        for theta in grid
    ])
    coarse = float(grid[int(np.argmin(costs))])
    lo = max(float(params.theta_min_deg), coarse - float(params.refine_radius_deg))
    hi = min(float(params.theta_max_deg), coarse + float(params.refine_radius_deg))
    result = minimize_scalar(
        lambda theta: _weighted_huber_cost(
            delays_s - _predicted_pair_delays(positions, pairs, float(theta)),
            weights,
            delta,
        ),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-3},
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
    }


def hodge_tde_estimates(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: HodgeTDEParameters = HodgeTDEParameters(),
) -> HodgeTDEResult:
    """Return frozen GCC baseline, B0 Hodge candidate, and safe endpoint."""
    baseline_params = GCCPHATParameters(
        frequency_min_hz=params.frequency_min_hz,
        frequency_max_hz=params.frequency_max_hz,
        theta_min_deg=params.theta_min_deg,
        theta_max_deg=params.theta_max_deg,
        coarse_step_deg=params.coarse_step_deg,
        refine_radius_deg=params.refine_radius_deg,
        pair_min_baseline_m=params.pair_min_baseline_m,
        maximum_pairs=params.maximum_pairs,
    )
    baseline_theta, baseline_diag = gcc_phat_tde_estimate(
        audio,
        sample_rate_hz,
        microphone_positions_m,
        params=baseline_params,
    )
    bank = extract_gcc_peak_bank(
        audio,
        sample_rate_hz,
        microphone_positions_m,
        params=params,
    )
    baseline_predicted = _predicted_pair_delays(
        np.asarray(microphone_positions_m, dtype=float),
        bank["pairs"],
        baseline_theta,
    )
    reassigned = reassign_topk_hodge(
        bank["candidate_delays_s"],
        bank["candidate_amplitudes"],
        bank["candidate_valid"],
        bank["pairs"],
        len(np.asarray(microphone_positions_m)),
        sample_rate_hz,
        bank["base_quality"],
        initial_predicted_delays_s=baseline_predicted,
        params=params,
    )
    hodge_theta, hodge_fit = _edge_space_azimuth_fit(
        np.asarray(microphone_positions_m, dtype=float),
        bank["pairs"],
        reassigned["selected_delays_s"],
        reassigned["weights"],
        sample_rate_hz,
        params,
    )
    cycle_gain = (
        reassigned["top1_cycle_ratio"] - reassigned["final_cycle_ratio"]
    ) / max(reassigned["top1_cycle_ratio"], _EPS)
    angle_change = abs(float(hodge_theta) - float(baseline_theta))
    reasons: list[str] = []
    if reassigned["assignment_changes_from_top1"] < 1:
        reasons.append("no_peak_reassignment")
    if cycle_gain < float(params.safe_cycle_improvement_fraction):
        reasons.append("insufficient_cycle_reduction")
    if angle_change > float(params.safe_max_angle_change_deg):
        reasons.append("angle_change_too_large")
    if reassigned["median_selected_amplitude"] < float(params.safe_min_selected_amplitude):
        reasons.append("selected_peaks_too_weak")
    if reassigned["effective_pair_count"] < float(params.safe_min_effective_pairs):
        reasons.append("too_few_effective_pairs")
    safe_accept = len(reasons) == 0
    safe_theta = float(hodge_theta if safe_accept else baseline_theta)

    pair_rows = []
    for row, assignment, selected, amplitude, weight, residual in zip(
        bank["pair_rows"],
        reassigned["assignment"],
        reassigned["selected_delays_s"],
        reassigned["selected_amplitudes"],
        reassigned["weights"],
        reassigned["cycle_residual_s"],
    ):
        pair_rows.append({
            **row,
            "selected_peak_index": int(assignment),
            "selected_delay_s": float(selected),
            "selected_amplitude": float(amplitude),
            "final_weight": float(weight),
            "cycle_residual_s": float(residual),
        })

    return HodgeTDEResult(
        estimates_deg={
            "gcc_phat_tde_huber": float(baseline_theta),
            "gcc_topk_hodge_b0": float(hodge_theta),
            "gcc_topk_hodge_b0_safe": float(safe_theta),
        },
        diagnostics={
            "parameters": params.as_dict(),
            "pair_count": int(len(bank["pairs"])),
            "baseline": baseline_diag,
            "hodge_fit": hodge_fit,
            "top1_cycle_ratio": float(reassigned["top1_cycle_ratio"]),
            "final_cycle_ratio": float(reassigned["final_cycle_ratio"]),
            "relative_cycle_reduction": float(cycle_gain),
            "assignment_changes_from_top1": int(reassigned["assignment_changes_from_top1"]),
            "median_selected_amplitude": float(reassigned["median_selected_amplitude"]),
            "effective_pair_count": float(reassigned["effective_pair_count"]),
            "candidate_estimate_deg": float(hodge_theta),
            "baseline_estimate_deg": float(baseline_theta),
            "absolute_angle_change_deg": float(angle_change),
            "safe_accept": bool(safe_accept),
            "safe_fallback_reasons": reasons,
            "iterations": reassigned["iterations"],
            "pair_rows": pair_rows,
        },
    )
