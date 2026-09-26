# =========================================================
# File        : mst_tdoa_yamaoka2023.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the mst tdoa yamaoka2023 module used by the reproducibility workflow.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - In-memory estimates, state objects, or helper values returned to calling code.
#
# Used in paper:
#   - Adapted MST-TDE comparison reported in the manuscript and Supplementary Material.
#
# Main parameters:
#   - Preregistered MST-TDE adaptation; no post-result tuning.
#   - See documentation/CODE_GUIDE.md for fixed interface details.
#
# Software:
#   - Python 3.x; dependencies listed in requirements.txt/environment.yml.
#
# Author      : Xin Lu
# Last update : 2026-09-25
# =========================================================
"""Post-freeze Yamaoka-et-al.-2023 MST-TDE comparator for the DOA paper.

Scientific identity
-------------------
This module implements the core graph-TDOA mechanism of Yamaoka et al.,
"Minimum-Spanning-Tree-Based Time Delay Estimation Robust to Outliers",
IEEE Access 11 (2023), 121284--121294, DOI 10.1109/ACCESS.2023.3327011.
The registered setting is alpha=1: microphone-pair delays are the maxima of
band-limited GCC-PHAT functions, edge cost is minus the GCC-PHAT value at the
measured delay, an MST selects M-1 nonredundant pairs, and the full TDOA matrix
is reconstructed by sums along the unique tree paths.

Benchmark-interface adaptation
------------------------------
The source paper outputs a full TDOA matrix.  For the present LOCATA azimuth
benchmark only, that reconstructed matrix is mapped to the same 2-D far-field
azimuth convention used elsewhere in the frozen paper by an UNWEIGHTED
least-squares angle fit.  This downstream fit is an interface adaptation, not
claimed as a reproduction of every downstream localization choice in the
source paper.  It uses no source truth, no C/E0 output, no task label, and no
array label.

The only geometry-specific restriction is the physically admissible lag range
for each microphone pair (baseline/c plus 1.5 samples), matching the benchmark's
causal TDOA interface and fixed before any LOCATA comparator result is opened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar

SPEED_OF_SOUND = 343.0
_EPS = 1.0e-15


def _parabolic_peak(y: np.ndarray, index: int) -> float:
    """Sub-sample offset around a local maximum, clipped to one bin."""
    values = np.asarray(y, dtype=float)
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    a, b, c = float(values[index - 1]), float(values[index]), float(values[index + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1.0e-15:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))


def _predicted_pair_delays(positions: np.ndarray, pairs: np.ndarray, theta_deg: float) -> np.ndarray:
    """Far-field pair delays in the frozen paper's 2-D azimuth convention."""
    theta = np.deg2rad(float(theta_deg))
    direction = np.asarray([np.sin(theta), np.cos(theta), 0.0], dtype=float)
    pos = np.asarray(positions, dtype=float)
    arrival = -(pos @ direction) / SPEED_OF_SOUND
    pair_array = np.asarray(pairs, dtype=int)
    return arrival[pair_array[:, 0]] - arrival[pair_array[:, 1]]


@dataclass(frozen=True)
class MSTTDOAParameters:
    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 3500.0
    theta_min_deg: float = -80.0
    theta_max_deg: float = 80.0
    theta_step_deg: float = 1.0
    refine_radius_deg: float = 1.5
    alpha: int = 1
    physical_lag_margin_samples: float = 1.5

    def as_dict(self) -> dict:
        return asdict(self)


def _all_pair_indices(microphone_count: int) -> np.ndarray:
    return np.asarray(
        [(i, j) for i in range(int(microphone_count)) for j in range(i + 1, int(microphone_count))],
        dtype=int,
    )


def _pair_gcc_phat_measurements(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    params: MSTTDOAParameters,
) -> dict:
    if int(params.alpha) != 1:
        raise ValueError("The preregistered comparator uses alpha=1 only")
    x = np.asarray(audio, dtype=float)
    positions = np.asarray(microphone_positions_m, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(positions):
        raise ValueError("audio must have shape (microphones, samples) matching positions")
    pairs = _all_pair_indices(len(positions))
    if len(pairs) < 1:
        raise ValueError("At least two microphones are required")

    n = int(x.shape[1])
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 32))))
    spectra = np.fft.rfft(x, n=nfft, axis=1)
    freq = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate_hz))
    band = (freq >= float(params.frequency_min_hz)) & (freq <= float(params.frequency_max_hz))
    cross = spectra[pairs[:, 0]] * np.conj(spectra[pairs[:, 1]])
    phat = np.zeros_like(cross)
    phat[:, band] = cross[:, band] / np.maximum(np.abs(cross[:, band]), _EPS)
    correlations = np.fft.irfft(phat, n=nfft, axis=1).real

    delays = np.empty(len(pairs), dtype=float)
    reliabilities = np.empty(len(pairs), dtype=float)
    peak_lag_samples = np.empty(len(pairs), dtype=float)
    max_lag_samples = np.empty(len(pairs), dtype=int)
    for k, (i, j) in enumerate(pairs):
        baseline = float(np.linalg.norm(positions[i] - positions[j]))
        max_delay = baseline / SPEED_OF_SOUND + float(params.physical_lag_margin_samples) / float(sample_rate_hz)
        max_lag = min(int(np.ceil(max_delay * float(sample_rate_hz))) + 1, nfft // 2 - 1)
        lags = np.arange(-max_lag, max_lag + 1, dtype=int)
        corr = correlations[k]
        values = np.concatenate([corr[-max_lag:], corr[: max_lag + 1]])
        peak_index = int(np.argmax(values))
        sub = _parabolic_peak(values, peak_index)
        lag_samples = float(lags[peak_index]) + float(sub)
        # Yamaoka et al. Eq. (15), alpha=1: edge cost is negative GCC-PHAT
        # evaluated at the measured delay.  Linear interpolation at the
        # sub-sample peak is used only to evaluate that same peak value.
        p0 = float(values[peak_index])
        if sub > 0.0 and peak_index + 1 < len(values):
            p1 = float(values[peak_index + 1])
            peak_value = (1.0 - sub) * p0 + sub * p1
        elif sub < 0.0 and peak_index - 1 >= 0:
            p1 = float(values[peak_index - 1])
            peak_value = (1.0 + sub) * p0 + (-sub) * p1
        else:
            peak_value = p0
        delays[k] = lag_samples / float(sample_rate_hz)
        reliabilities[k] = peak_value
        peak_lag_samples[k] = lag_samples
        max_lag_samples[k] = max_lag

    return {
        "pairs": pairs,
        "delays_s": delays,
        "reliabilities": reliabilities,
        "costs": -reliabilities,
        "peak_lag_samples": peak_lag_samples,
        "max_lag_samples": max_lag_samples,
        "nfft": int(nfft),
        "frequency_bin_count": int(np.sum(band)),
    }


def _prim_mst(microphone_count: int, pairs: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Deterministic Prim MST; ties are broken lexicographically by (cost,i,j)."""
    m = int(microphone_count)
    pair_array = np.asarray(pairs, dtype=int)
    c = np.asarray(costs, dtype=float)
    if len(pair_array) != len(c):
        raise ValueError("pairs and costs length mismatch")
    matrix = np.full((m, m), np.inf, dtype=float)
    for (i, j), value in zip(pair_array, c):
        matrix[int(i), int(j)] = matrix[int(j), int(i)] = float(value)

    selected = {0}
    edges: list[tuple[int, int]] = []
    while len(selected) < m:
        choices: list[tuple[float, int, int]] = []
        for i in sorted(selected):
            for j in range(m):
                if j in selected or not np.isfinite(matrix[i, j]):
                    continue
                a, b = (i, j) if i < j else (j, i)
                choices.append((float(matrix[i, j]), int(a), int(b)))
        if not choices:
            raise RuntimeError("Signal graph is disconnected")
        _, a, b = min(choices)
        new_node = b if a in selected else a
        # In the rare case both endpoints' orientation was normalized above,
        # choose the endpoint not yet in the tree.
        if new_node in selected:
            new_node = a if b in selected else b
        selected.add(int(new_node))
        edges.append((int(a), int(b)))
    return np.asarray(edges, dtype=int)


def _tree_arrival_potentials(
    microphone_count: int,
    pairs: np.ndarray,
    delays_s: np.ndarray,
    mst_edges: np.ndarray,
) -> np.ndarray:
    """Recover relative arrival times from tree-edge delays tau_ij=t_i-t_j."""
    lookup: dict[tuple[int, int], float] = {}
    for (i, j), delay in zip(np.asarray(pairs, int), np.asarray(delays_s, float)):
        lookup[(int(i), int(j))] = float(delay)
        lookup[(int(j), int(i))] = -float(delay)
    adjacency: list[list[int]] = [[] for _ in range(int(microphone_count))]
    for i, j in np.asarray(mst_edges, int):
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))
    potential = np.full(int(microphone_count), np.nan, dtype=float)
    potential[0] = 0.0
    stack = [0]
    while stack:
        u = stack.pop()
        for v in sorted(adjacency[u]):
            if np.isfinite(potential[v]):
                continue
            # tau_uv = t_u - t_v  =>  t_v = t_u - tau_uv
            potential[v] = potential[u] - lookup[(u, v)]
            stack.append(v)
    if not np.all(np.isfinite(potential)):
        raise RuntimeError("MST traversal did not reach all microphones")
    return potential


def _fit_azimuth_unweighted(
    positions: np.ndarray,
    pairs: np.ndarray,
    reconstructed_delays_s: np.ndarray,
    params: MSTTDOAParameters,
) -> tuple[float, dict]:
    grid = np.arange(
        float(params.theta_min_deg),
        float(params.theta_max_deg) + 0.5 * float(params.theta_step_deg),
        float(params.theta_step_deg),
        dtype=float,
    )
    delays = np.asarray(reconstructed_delays_s, dtype=float)
    costs = np.asarray([
        np.mean((delays - _predicted_pair_delays(positions, pairs, float(theta))) ** 2)
        for theta in grid
    ], dtype=float)
    coarse = float(grid[int(np.argmin(costs))])
    lo = max(float(params.theta_min_deg), coarse - float(params.refine_radius_deg))
    hi = min(float(params.theta_max_deg), coarse + float(params.refine_radius_deg))
    result = minimize_scalar(
        lambda theta: float(np.mean((delays - _predicted_pair_delays(positions, pairs, float(theta))) ** 2)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1.0e-3},
    )
    estimate = float(result.x if result.success else coarse)
    residual = delays - _predicted_pair_delays(positions, pairs, estimate)
    return estimate, {
        "coarse_estimate_deg": coarse,
        "optimizer_success": bool(result.success),
        "unweighted_rmse_s": float(np.sqrt(np.mean(residual * residual))),
    }


def mst_tdoa_azimuth_estimate(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: MSTTDOAParameters = MSTTDOAParameters(),
) -> tuple[float, dict]:
    """Estimate azimuth with preregistered MST(alpha=1) TDOA comparator."""
    positions = np.asarray(microphone_positions_m, dtype=float)
    measured = _pair_gcc_phat_measurements(audio, sample_rate_hz, positions, params)
    pairs = measured["pairs"]
    mst_edges = _prim_mst(len(positions), pairs, measured["costs"])
    potential = _tree_arrival_potentials(len(positions), pairs, measured["delays_s"], mst_edges)
    reconstructed = potential[pairs[:, 0]] - potential[pairs[:, 1]]
    estimate, fit_diag = _fit_azimuth_unweighted(positions, pairs, reconstructed, params)

    pair_index = {tuple(map(int, p)): k for k, p in enumerate(pairs)}
    selected_indices = [pair_index[tuple(map(int, e))] for e in mst_edges]
    return estimate, {
        "parameters": params.as_dict(),
        "pair_count_full": int(len(pairs)),
        "tree_edge_count": int(len(mst_edges)),
        "mst_edges": mst_edges,
        "mst_edge_costs": np.asarray(measured["costs"])[selected_indices],
        "mst_edge_reliabilities": np.asarray(measured["reliabilities"])[selected_indices],
        "tree_arrival_potentials_s": potential,
        "reconstructed_pair_delays_s": reconstructed,
        "nfft": int(measured["nfft"]),
        "frequency_bin_count": int(measured["frequency_bin_count"]),
        "fit": fit_diag,
        "truth_used": False,
        "e0_or_c_used": False,
        "task_identity_used": False,
        "array_identity_used": False,
    }
