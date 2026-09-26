# =========================================================
# File        : causal_multiscale_hodge.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the causal multiscale hodge module used by the reproducibility workflow.
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
"""Causal multiscale persistence-constrained multi-hypothesis Hodge-TDE.

V52K-R8 responds to the V52J-R7 finding that single-window optimization
certainty is not acoustic correctness.  The new information is temporal but
strictly causal: three nested windows ending at the same output anchor are
analysed.  Each scale produces a small bank of discrete joint cycle/geometry
hypotheses.  A finite time-expanded graph then selects the minimum-description
path across scales.  The E0 path is an explicit null model; a Hodge correction
is emitted only when a non-null path wins after paying peak-change complexity
and is supported by at least two causal scales.

The dynamic program is exact on the generated state banks.  It is not a proof
that a selected acoustic path is direct.  A coherent reflection that persists
across every scale remains fundamentally indistinguishable without additional
information; this limitation is explicitly reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import log
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar

from .baseline_extension import _predicted_pair_delays
from .baseline_forensic import BaselineForensicParameters
from .certified_multihypothesis_hodge import (
    _coordinate_refine_assignment,
    _discrete_objective_from_indices,
    _unary_lower_bound,
    _weighted_cycle_matrix,
)
from .hodge_tde import HodgeTDEParameters, extract_gcc_peak_bank
from .r5_anchor_estimates import E0, r5_anchor_estimates

_EPS = 1.0e-12

R8_RAW = "gcc_topk_causal_multiscale_hodge_path_r8"
R8_SELECTIVE = "gcc_topk_causal_multiscale_hodge_path_r8_mdl_e0_fallback"


@dataclass(frozen=True)
class CausalMultiscaleHodgeParameters:
    """Frozen V52K-R8 foundation parameters.

    All scales end at the current output anchor; no future sample is used.
    The state-path decision is expressed relative to the E0 null path.  The
    peak-change cost is the uniform K-ary code length log(K), rather than a
    fitted classification threshold.
    """

    scales_s: tuple[float, ...] = (0.30, 0.50, 0.75)
    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 3500.0
    theta_min_deg: float = -80.0
    theta_max_deg: float = 80.0
    theta_step_deg: float = 2.0
    refine_radius_deg: float = 2.0
    pair_min_baseline_m: float = 0.05
    maximum_pairs: int = 0
    top_k: int = 3
    minimum_peak_separation_s: float = 0.00015
    peak_evidence_weight: float = 0.20
    cycle_weight: float = 1.0
    geometry_weight: float = 1.0
    quality_floor: float = 0.05
    coordinate_descent_passes: int = 7
    preselect_theta_count: int = 15
    states_per_scale: int = 7
    minimum_state_separation_deg: float = 4.0
    # Minimum-description penalties and persistence scales.
    peak_change_code_nats: float = log(3.0)
    delay_transition_scale_samples: float = 1.5
    direction_transition_scale_deg: float = 8.0
    direction_transition_weight: float = 0.25
    baseline_switch_code_nats: float = 0.5 * log(2.0)
    minimum_nonbaseline_scale_support: int = 2
    maximum_supported_direction_spread_deg: float = 8.0

    def as_dict(self) -> dict:
        out = asdict(self)
        out["scales_s"] = list(self.scales_s)
        return out

    def hodge_bank_parameters(self) -> HodgeTDEParameters:
        return HodgeTDEParameters(
            frequency_min_hz=self.frequency_min_hz,
            frequency_max_hz=self.frequency_max_hz,
            theta_min_deg=self.theta_min_deg,
            theta_max_deg=self.theta_max_deg,
            coarse_step_deg=max(self.theta_step_deg, 0.25),
            refine_radius_deg=self.refine_radius_deg,
            pair_min_baseline_m=self.pair_min_baseline_m,
            maximum_pairs=self.maximum_pairs,
            top_k=self.top_k,
            minimum_peak_separation_s=self.minimum_peak_separation_s,
            amplitude_penalty=self.peak_evidence_weight,
        )


@dataclass(frozen=True)
class CausalMultiscaleHodgeResult:
    estimates_deg: dict[str, float]
    diagnostics: dict


def _huber(value: np.ndarray | float, delta: float = 1.0) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    a = np.abs(x)
    return np.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def _circular_abs_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def _local_minima_indices(values: np.ndarray) -> list[int]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    out: list[int] = []
    for i in range(len(values)):
        left = values[i - 1] if i > 0 else np.inf
        right = values[i + 1] if i + 1 < len(values) else np.inf
        if values[i] <= left and values[i] <= right:
            out.append(int(i))
    return out


def _fixed_assignment_objective(
    theta_deg: float,
    *,
    indices: np.ndarray,
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    quality: np.ndarray,
    Q: np.ndarray,
    positions: np.ndarray,
    pairs: np.ndarray,
    sample_rate_hz: float,
    geometry_weight: float,
) -> float:
    geometry = (
        _predicted_pair_delays(positions, pairs, float(theta_deg))
        * float(sample_rate_hz)
    )
    b = float(geometry_weight) * quality * geometry
    constant = float(geometry_weight) * float(np.sum(quality * geometry * geometry))
    return float(
        _discrete_objective_from_indices(
            indices, candidates, peak_cost, Q, b, constant
        )
    )


def _refine_fixed_assignment(
    theta_deg: float,
    *,
    indices: np.ndarray,
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    quality: np.ndarray,
    Q: np.ndarray,
    positions: np.ndarray,
    pairs: np.ndarray,
    sample_rate_hz: float,
    params: CausalMultiscaleHodgeParameters,
) -> tuple[float, float]:
    lo = max(float(params.theta_min_deg), float(theta_deg) - float(params.refine_radius_deg))
    hi = min(float(params.theta_max_deg), float(theta_deg) + float(params.refine_radius_deg))

    def objective(theta: float) -> float:
        return _fixed_assignment_objective(
            theta,
            indices=indices,
            candidates=candidates,
            peak_cost=peak_cost,
            quality=quality,
            Q=Q,
            positions=positions,
            pairs=pairs,
            sample_rate_hz=sample_rate_hz,
            geometry_weight=params.geometry_weight,
        )

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1.0e-3},
    )
    refined = float(result.x if result.success else theta_deg)
    return refined, float(objective(refined))


def build_scale_state_bank(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    scale_s: float,
    params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
    forensic_params: BaselineForensicParameters = BaselineForensicParameters(),
) -> dict:
    """Generate a diverse discrete state bank for one causal scale."""
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 2:
        raise ValueError("audio must have shape channels x samples")
    length = int(round(float(scale_s) * float(sample_rate_hz)))
    if length < 32 or length > audio.shape[1]:
        raise ValueError(
            f"scale {scale_s}s requires {length} samples but audio has {audio.shape[1]}"
        )
    segment = audio[:, -length:]
    positions = np.asarray(microphone_positions_m, dtype=float)
    anchors, anchor_diag = r5_anchor_estimates(
        segment, sample_rate_hz, positions, params=forensic_params
    )
    e0_theta = float(anchors[E0])
    bank = extract_gcc_peak_bank(
        segment,
        sample_rate_hz,
        positions,
        params=params.hodge_bank_parameters(),
    )
    delays_s = np.asarray(bank["candidate_delays_s"], dtype=float)
    amplitudes = np.asarray(bank["candidate_amplitudes"], dtype=float)
    valid = np.asarray(bank["candidate_valid"], dtype=bool)
    pairs = np.asarray(bank["pairs"], dtype=int)
    quality = np.clip(
        np.asarray(bank["base_quality"], dtype=float).reshape(-1),
        float(params.quality_floor),
        1.0,
    )
    if not np.all(np.any(valid, axis=1)):
        raise RuntimeError("each pair must have at least one valid GCC candidate")
    candidates = delays_s * float(sample_rate_hz)
    peak_cost = np.zeros_like(candidates)
    peak_cost[valid] = float(params.peak_evidence_weight) * (
        -np.log(np.maximum(amplitudes[valid], _EPS))
    )
    cycle_matrix, graph_diag = _weighted_cycle_matrix(
        pairs, len(positions), quality
    )
    W = np.diag(quality)
    Q = float(params.cycle_weight) * cycle_matrix + float(params.geometry_weight) * W

    def solve_at_theta(theta: float) -> dict:
        geometry = (
            _predicted_pair_delays(positions, pairs, float(theta))
            * float(sample_rate_hz)
        )
        b = float(params.geometry_weight) * quality * geometry
        constant = float(params.geometry_weight) * float(
            np.sum(quality * geometry * geometry)
        )
        unary = peak_cost + float(params.geometry_weight) * quality[:, None] * (
            candidates - geometry[:, None]
        ) ** 2
        initial_indices = np.argmin(np.where(valid, unary, np.inf), axis=1)
        discrete = _coordinate_refine_assignment(
            initial_indices,
            candidates,
            peak_cost,
            valid,
            Q,
            b,
            constant,
            passes=params.coordinate_descent_passes,
        )
        refined_theta, refined_objective = _refine_fixed_assignment(
            float(theta),
            indices=discrete["indices"],
            candidates=candidates,
            peak_cost=peak_cost,
            quality=quality,
            Q=Q,
            positions=positions,
            pairs=pairs,
            sample_rate_hz=sample_rate_hz,
            params=params,
        )
        indices = np.asarray(discrete["indices"], dtype=int)
        rows = np.arange(len(indices))
        return {
            "theta_deg": float(refined_theta),
            "objective": float(refined_objective),
            "indices": indices,
            "selected_delays_samples": candidates[rows, indices].copy(),
            "selected_delays_s": delays_s[rows, indices].copy(),
            "selected_amplitudes": amplitudes[rows, indices].copy(),
            "coordinate_changes": int(discrete["coordinate_changes"]),
        }

    baseline = solve_at_theta(e0_theta)
    baseline["state_kind"] = "e0_null"
    baseline["is_baseline"] = True

    theta_grid = np.arange(
        float(params.theta_min_deg),
        float(params.theta_max_deg) + 0.5 * float(params.theta_step_deg),
        float(params.theta_step_deg),
    )
    geometry_bank = np.asarray([
        _predicted_pair_delays(positions, pairs, float(theta))
        * float(sample_rate_hz)
        for theta in theta_grid
    ])
    unary_bounds = np.asarray([
        _unary_lower_bound(
            candidates,
            peak_cost,
            valid,
            quality,
            geometry,
            params.geometry_weight,
        )
        for geometry in geometry_bank
    ])
    candidates_theta = set(
        int(i)
        for i in np.argsort(unary_bounds, kind="mergesort")[
            : int(params.preselect_theta_count)
        ]
    )
    candidates_theta.update(_local_minima_indices(unary_bounds))
    candidates_theta.add(int(np.argmin(np.abs(theta_grid - e0_theta))))
    solved_states = [solve_at_theta(float(theta_grid[i])) for i in sorted(candidates_theta)]
    solved_states.sort(key=lambda item: (item["objective"], item["theta_deg"]))

    diverse: list[dict] = []
    for state in solved_states:
        changed = int(np.sum(state["indices"] != baseline["indices"]))
        direction_delta = _circular_abs_deg(state["theta_deg"], baseline["theta_deg"])
        if changed == 0 and direction_delta < 0.5 * float(params.theta_step_deg):
            continue
        if any(
            _circular_abs_deg(state["theta_deg"], old["theta_deg"])
            < float(params.minimum_state_separation_deg)
            for old in diverse
        ):
            continue
        state = dict(state)
        state["state_kind"] = "joint_hodge"
        state["is_baseline"] = False
        diverse.append(state)
        if len(diverse) >= int(params.states_per_scale):
            break

    edge_count = max(len(pairs), 1)
    for state_id, state in enumerate([baseline] + diverse):
        changed = int(np.sum(state["indices"] != baseline["indices"]))
        spatial_delta_per_edge = (
            float(state["objective"]) - float(baseline["objective"])
        ) / float(edge_count)
        complexity_per_edge = (
            float(params.peak_change_code_nats) * float(changed) / float(edge_count)
        )
        state["state_id"] = int(state_id)
        state["changed_edges_from_e0"] = changed
        state["changed_edge_fraction_from_e0"] = float(changed / edge_count)
        state["spatial_delta_per_edge"] = float(spatial_delta_per_edge)
        state["complexity_per_edge"] = float(complexity_per_edge)
        state["node_cost"] = float(spatial_delta_per_edge + complexity_per_edge)
        state["geometry_residual_samples"] = (
            state["selected_delays_samples"]
            - _predicted_pair_delays(
                positions, pairs, float(state["theta_deg"])
            ) * float(sample_rate_hz)
        )

    return {
        "scale_s": float(scale_s),
        "e0_estimate_deg": e0_theta,
        "states": [baseline] + diverse,
        "pairs": pairs,
        "positions": positions,
        "sample_rate_hz": float(sample_rate_hz),
        "graph_diagnostics": graph_diag,
        "pair_rows": bank["pair_rows"],
        "anchor_diagnostics": anchor_diag,
    }


def _transition_cost(
    previous: dict,
    current: dict,
    *,
    params: CausalMultiscaleHodgeParameters,
) -> dict:
    prev_residual = np.asarray(previous["geometry_residual_samples"], dtype=float)
    curr_residual = np.asarray(current["geometry_residual_samples"], dtype=float)
    if prev_residual.shape != curr_residual.shape:
        raise ValueError("multiscale states must use an identical pair graph")
    residual_change = curr_residual - prev_residual
    delay_cost = float(
        np.mean(
            _huber(
                residual_change / max(float(params.delay_transition_scale_samples), _EPS)
            )
        )
    )
    direction_delta = _circular_abs_deg(
        current["theta_deg"], previous["theta_deg"]
    )
    direction_cost = float(params.direction_transition_weight) * float(
        _huber(
            direction_delta
            / max(float(params.direction_transition_scale_deg), _EPS)
        )
    )
    switch_cost = (
        float(params.baseline_switch_code_nats)
        if bool(previous["is_baseline"]) != bool(current["is_baseline"])
        else 0.0
    )
    return {
        "total": float(delay_cost + direction_cost + switch_cost),
        "delay_residual_persistence_cost": delay_cost,
        "direction_persistence_cost": direction_cost,
        "baseline_switch_cost": switch_cost,
        "direction_delta_deg": direction_delta,
        "median_abs_residual_change_samples": float(
            np.median(np.abs(residual_change))
        ),
    }


def solve_multiscale_state_path(
    scale_banks: Iterable[dict],
    *,
    params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
) -> dict:
    """Exactly enumerate the small time-expanded state graph.

    The number of scales is three and each layer contains at most one E0 state
    plus ``states_per_scale`` joint states, so exhaustive path enumeration is
    deterministic and exact (at most 8^3 = 512 paths under the frozen setup).
    """
    banks = list(scale_banks)
    if len(banks) != len(params.scales_s):
        raise ValueError("scale bank count must match the frozen scale count")
    path_rows: list[dict] = []
    state_ranges = [range(len(bank["states"])) for bank in banks]
    for path_index, indices in enumerate(product(*state_ranges)):
        states = [bank["states"][idx] for bank, idx in zip(banks, indices)]
        node_cost = float(sum(float(state["node_cost"]) for state in states))
        transitions = [
            _transition_cost(states[i - 1], states[i], params=params)
            for i in range(1, len(states))
        ]
        transition_cost = float(sum(row["total"] for row in transitions))
        total = node_cost + transition_cost
        nonbaseline = [state for state in states if not bool(state["is_baseline"])]
        final_nonbaseline = not bool(states[-1]["is_baseline"])
        supported = [
            state for state in nonbaseline
            if _circular_abs_deg(state["theta_deg"], states[-1]["theta_deg"])
            <= float(params.maximum_supported_direction_spread_deg)
        ] if final_nonbaseline else []
        support_count = int(len(supported))
        support_spread = (
            float(max(s["theta_deg"] for s in supported) - min(s["theta_deg"] for s in supported))
            if len(supported) >= 2 else 0.0
        )
        path_rows.append({
            "path_index": int(path_index),
            "state_indices": tuple(int(v) for v in indices),
            "states": states,
            "node_cost": node_cost,
            "transition_cost": transition_cost,
            "total_cost": float(total),
            "final_nonbaseline": bool(final_nonbaseline),
            "nonbaseline_scale_count": int(len(nonbaseline)),
            "supported_nonbaseline_scale_count": support_count,
            "supported_direction_spread_deg": support_spread,
            "transitions": transitions,
        })
    path_rows.sort(key=lambda row: (row["total_cost"], row["state_indices"]))
    best_any = path_rows[0]
    baseline_paths = [row for row in path_rows if not row["final_nonbaseline"]]
    nonbaseline_paths = [row for row in path_rows if row["final_nonbaseline"]]
    best_baseline = baseline_paths[0]
    all_e0_paths = [
        row for row in path_rows
        if all(bool(state["is_baseline"]) for state in row["states"])
    ]
    if len(all_e0_paths) != 1:
        raise RuntimeError("each multiscale bank must contain exactly one E0 null state")
    all_e0_null = all_e0_paths[0]
    best_nonbaseline = nonbaseline_paths[0] if nonbaseline_paths else None
    persistence_pass = bool(
        best_nonbaseline is not None
        and best_nonbaseline["supported_nonbaseline_scale_count"]
        >= int(params.minimum_nonbaseline_scale_support)
        and best_nonbaseline["supported_direction_spread_deg"]
        <= float(params.maximum_supported_direction_spread_deg)
    )
    model_selection_gain = (
        float(all_e0_null["total_cost"] - best_nonbaseline["total_cost"])
        if best_nonbaseline is not None else float("-inf")
    )
    adopt = bool(
        best_nonbaseline is not None
        and persistence_pass
        and model_selection_gain > 0.0
    )
    second_cost = float(path_rows[1]["total_cost"]) if len(path_rows) > 1 else float("inf")
    return {
        "adopt_joint_path": adopt,
        "persistence_pass": persistence_pass,
        "model_selection_gain": model_selection_gain,
        "best_path_margin": float(second_cost - float(best_any["total_cost"])),
        "best_any_path": best_any,
        "best_baseline_final_path": best_baseline,
        "all_e0_null_path": all_e0_null,
        "best_nonbaseline_final_path": best_nonbaseline,
        "path_count": int(len(path_rows)),
        "exact_path_optimization": True,
        "coherent_reflection_across_all_scales_excluded": False,
    }


def causal_multiscale_hodge_estimates(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
    forensic_params: BaselineForensicParameters = BaselineForensicParameters(),
) -> CausalMultiscaleHodgeResult:
    """Return long-scale E0, raw multiscale path, and selective endpoint."""
    audio = np.asarray(audio, dtype=float)
    longest_samples = int(round(max(params.scales_s) * float(sample_rate_hz)))
    if audio.ndim != 2 or audio.shape[1] < longest_samples:
        raise ValueError("audio does not contain the longest frozen causal scale")
    banks = [
        build_scale_state_bank(
            audio,
            sample_rate_hz,
            microphone_positions_m,
            scale_s=scale,
            params=params,
            forensic_params=forensic_params,
        )
        for scale in params.scales_s
    ]
    solved = solve_multiscale_state_path(banks, params=params)
    e0_long = float(banks[-1]["e0_estimate_deg"])
    best_nonbaseline = solved["best_nonbaseline_final_path"]
    raw = (
        float(best_nonbaseline["states"][-1]["theta_deg"])
        if best_nonbaseline is not None else e0_long
    )
    endpoint = float(raw if solved["adopt_joint_path"] else e0_long)

    scale_rows: list[dict] = []
    for scale_index, bank in enumerate(banks):
        for state in bank["states"]:
            scale_rows.append({
                "scale_index": int(scale_index),
                "scale_s": float(bank["scale_s"]),
                "state_id": int(state["state_id"]),
                "state_kind": str(state["state_kind"]),
                "is_baseline": bool(state["is_baseline"]),
                "theta_deg": float(state["theta_deg"]),
                "objective": float(state["objective"]),
                "node_cost": float(state["node_cost"]),
                "spatial_delta_per_edge": float(state["spatial_delta_per_edge"]),
                "complexity_per_edge": float(state["complexity_per_edge"]),
                "changed_edges_from_e0": int(state["changed_edges_from_e0"]),
                "changed_edge_fraction_from_e0": float(state["changed_edge_fraction_from_e0"]),
            })

    selected_path = (
        solved["best_nonbaseline_final_path"]
        if solved["adopt_joint_path"]
        else solved["all_e0_null_path"]
    )
    selected_pair_rows: list[dict] = []
    for scale_index, (bank, state) in enumerate(zip(banks, selected_path["states"])):
        for pair_row, selected_index, selected_delay in zip(
            bank["pair_rows"], state["indices"], state["selected_delays_s"]
        ):
            selected_pair_rows.append({
                "scale_index": int(scale_index),
                "scale_s": float(bank["scale_s"]),
                **pair_row,
                "selected_peak_index_r8": int(selected_index),
                "selected_delay_s_r8": float(selected_delay),
                "selected_path_state_id_r8": int(state["state_id"]),
                "selected_path_theta_deg_r8": float(state["theta_deg"]),
                "selected_path_state_kind_r8": str(state["state_kind"]),
            })

    def compact_path(path: dict | None) -> dict | None:
        if path is None:
            return None
        return {
            "state_indices": list(path["state_indices"]),
            "theta_values_deg": [float(s["theta_deg"]) for s in path["states"]],
            "state_kinds": [str(s["state_kind"]) for s in path["states"]],
            "node_cost": float(path["node_cost"]),
            "transition_cost": float(path["transition_cost"]),
            "total_cost": float(path["total_cost"]),
            "nonbaseline_scale_count": int(path["nonbaseline_scale_count"]),
            "supported_nonbaseline_scale_count": int(path["supported_nonbaseline_scale_count"]),
            "supported_direction_spread_deg": float(path["supported_direction_spread_deg"]),
        }

    diagnostics = {
        "parameters": params.as_dict(),
        "e0_estimate_deg": e0_long,
        "raw_multiscale_estimate_deg": raw,
        "selective_endpoint_deg": endpoint,
        "adopt_joint_path": bool(solved["adopt_joint_path"]),
        "persistence_pass": bool(solved["persistence_pass"]),
        "model_selection_gain": float(solved["model_selection_gain"]),
        "best_path_margin": float(solved["best_path_margin"]),
        "path_count": int(solved["path_count"]),
        "best_any_path": compact_path(solved["best_any_path"]),
        "best_baseline_final_path": compact_path(solved["best_baseline_final_path"]),
        "all_e0_null_path": compact_path(solved["all_e0_null_path"]),
        "best_nonbaseline_final_path": compact_path(solved["best_nonbaseline_final_path"]),
        "scale_state_rows": scale_rows,
        "selected_pair_rows": selected_pair_rows,
        "graph_diagnostics_by_scale": [bank["graph_diagnostics"] for bank in banks],
        "future_context_used": False,
        "truth_used_for_estimation": False,
        "historical_safe_gate_used": False,
        "exact_path_optimization": True,
        "coherent_reflection_across_all_scales_excluded": False,
    }
    return CausalMultiscaleHodgeResult(
        estimates_deg={
            E0: e0_long,
            R8_RAW: raw,
            R8_SELECTIVE: endpoint,
        },
        diagnostics=diagnostics,
    )
