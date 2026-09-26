# =========================================================
# File        : mechanism_corrected_r8.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the mechanism corrected r8 module used by the reproducibility workflow.
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
"""Mechanism-corrected R8 (R8-M) development implementation.

R8-M keeps the frozen V52K-R8 candidate generation, Hodge geometry term,
causal scales, transition cost, persistence requirement, and E0 null model.
Only two missing model-cost terms are added when the already-generated state
banks are rescored:

1. Geometry-aware reassignment description length.  Reassigning GCC peaks on
   long-baseline pairs is charged more because the physically admissible delay
   alphabet is larger.  A combinatorial term also charges selecting many pairs.
2. Uncertainty-adaptive E0 anchor.  A non-baseline state that moves far from E0
   is charged in proportion to a truth-free E0 reliability score derived from
   GCC peak ambiguity / peak salience.  The anchor weakens automatically when
   E0 itself is acoustically ambiguous.

No task identity, source truth, localization error, or future context enters
these costs.  This module is for mechanism-development only; it does not claim
independent validation on the already-seen LOCATA evaluation arrays.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import floor, lgamma, log
from typing import Iterable

import numpy as np

from .causal_multiscale_hodge import (
    CausalMultiscaleHodgeParameters,
    _circular_abs_deg,
    _huber,
    build_scale_state_bank,
    solve_multiscale_state_path,
)
from .baseline_forensic import BaselineForensicParameters
from .hodge_tde import SPEED_OF_SOUND
from .r5_anchor_estimates import E0

_EPS = 1.0e-12
R8M_SELECTIVE = "gcc_topk_causal_multiscale_hodge_r8m_mechanism_corrected"


@dataclass(frozen=True)
class MechanismCorrectedR8Parameters:
    """Only two R8-M development degrees of freedom."""

    assignment_strength: float = 0.0
    anchor_strength: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _log_comb(n: int, k: int) -> float:
    n = int(n)
    k = int(k)
    if k <= 0 or k >= n:
        return 0.0 if k in (0, n) else float("inf")
    return float(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0))


def feasible_delay_alphabet_size(
    baseline_m: float,
    sample_rate_hz: float,
    *,
    speed_of_sound_m_s: float = SPEED_OF_SOUND,
) -> int:
    """Number of integer sample-delay cells in the physical TDOA interval."""
    max_delay_samples = float(sample_rate_hz) * max(float(baseline_m), 0.0) / float(speed_of_sound_m_s)
    return int(2 * floor(max_delay_samples) + 1)


def geometry_assignment_description_per_edge(
    *,
    changed_mask: np.ndarray,
    baselines_m: np.ndarray,
    sample_rate_hz: float,
) -> float:
    """Normalized MDL charge for choosing which edges and delay cells change."""
    changed = np.asarray(changed_mask, dtype=bool).reshape(-1)
    baselines = np.asarray(baselines_m, dtype=float).reshape(-1)
    if len(changed) != len(baselines):
        raise ValueError("changed_mask and baselines_m must have equal length")
    edge_count = max(len(changed), 1)
    m = int(np.sum(changed))
    if m == 0:
        return 0.0
    alphabet_code = 0.0
    for baseline in baselines[changed]:
        cells = max(feasible_delay_alphabet_size(float(baseline), sample_rate_hz), 1)
        alphabet_code += log(float(cells))
    return float((_log_comb(edge_count, m) + alphabet_code) / float(edge_count))


def e0_uncertainty_ratio_from_bank(bank: dict) -> float:
    """Dimensionless ambiguity / salience ratio; higher means less reliable E0."""
    rows = list(bank.get("pair_rows", []))
    if not rows:
        return float("nan")
    ptm = np.asarray([float(row["peak_to_median"]) for row in rows], dtype=float)
    amp2 = []
    for row in rows:
        amps = np.asarray(row.get("candidate_amplitudes", []), dtype=float)
        valid = np.asarray(row.get("candidate_valid", []), dtype=bool)
        if len(amps) >= 2 and len(valid) >= 2 and bool(valid[1]) and np.isfinite(amps[1]):
            amp2.append(float(amps[1]))
        else:
            amp2.append(0.0)
    q10 = float(np.quantile(ptm[np.isfinite(ptm)], 0.10)) if np.any(np.isfinite(ptm)) else float("nan")
    amp2_mean = float(np.mean(np.asarray(amp2, dtype=float)))
    if not np.isfinite(q10) or q10 <= 0.0:
        return float("inf")
    return float(amp2_mean / max(q10, _EPS))


def e0_anchor_reliability(uncertainty_ratio: float) -> float:
    """Parameter-free monotone reliability transform used by the anchor cost.

    For the observed LOCATA regime u is typically below one.  -log(u) gives a
    compact dimensionless reliability scale without introducing another tuned
    threshold.  If u>=1, no baseline anchor is imposed.
    """
    u = float(uncertainty_ratio)
    if not np.isfinite(u):
        return 0.0
    u = max(u, _EPS)
    return float(max(0.0, -log(min(u, 1.0))))


def _augment_bank_costs(
    bank: dict,
    *,
    r8_params: CausalMultiscaleHodgeParameters,
    r8m_params: MechanismCorrectedR8Parameters,
) -> dict:
    out = deepcopy(bank)
    states = out["states"]
    if not states or not bool(states[0]["is_baseline"]):
        raise ValueError("R8 scale bank must begin with the E0 null state")
    baseline = states[0]
    base_indices = np.asarray(baseline["indices"], dtype=int)
    pairs = np.asarray(out["pairs"], dtype=int)
    positions = np.asarray(out["positions"], dtype=float)
    baselines = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    uncertainty = e0_uncertainty_ratio_from_bank(out)
    reliability = e0_anchor_reliability(uncertainty)
    e0_theta = float(baseline["theta_deg"])

    for state in states:
        indices = np.asarray(state["indices"], dtype=int)
        changed = indices != base_indices
        assignment_description = geometry_assignment_description_per_edge(
            changed_mask=changed,
            baselines_m=baselines,
            sample_rate_hz=float(out["sample_rate_hz"]),
        )
        direction_delta = _circular_abs_deg(float(state["theta_deg"]), e0_theta)
        direction_normalized = direction_delta / max(float(r8_params.direction_transition_scale_deg), _EPS)
        anchor_shape = float(_huber(direction_normalized, delta=1.0))
        assignment_extra = float(r8m_params.assignment_strength) * assignment_description
        anchor_extra = float(r8m_params.anchor_strength) * reliability * anchor_shape
        original_node = float(state["node_cost"])
        state["r8_node_cost_original"] = original_node
        state["r8m_assignment_description_per_edge"] = float(assignment_description)
        state["r8m_assignment_extra_cost"] = assignment_extra
        state["r8m_e0_uncertainty_ratio"] = float(uncertainty)
        state["r8m_e0_anchor_reliability"] = float(reliability)
        state["r8m_direction_delta_from_e0_deg"] = float(direction_delta)
        state["r8m_anchor_shape"] = anchor_shape
        state["r8m_anchor_extra_cost"] = anchor_extra
        state["node_cost"] = float(original_node + assignment_extra + anchor_extra)
    out["r8m_scale_diagnostics"] = {
        "e0_uncertainty_ratio": float(uncertainty),
        "e0_anchor_reliability": float(reliability),
        "median_pair_baseline_m": float(np.median(baselines)) if len(baselines) else float("nan"),
        "maximum_pair_baseline_m": float(np.max(baselines)) if len(baselines) else float("nan"),
    }
    return out


def prepare_r8m_state_banks(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    r8_params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
    forensic_params: BaselineForensicParameters = BaselineForensicParameters(),
) -> list[dict]:
    """Generate the frozen R8 state banks exactly once for later R8-M rescoring."""
    x = np.asarray(audio, dtype=float)
    return [
        build_scale_state_bank(
            x,
            sample_rate_hz,
            microphone_positions_m,
            scale_s=scale,
            params=r8_params,
            forensic_params=forensic_params,
        )
        for scale in r8_params.scales_s
    ]


def rescore_prepared_r8m_banks(
    scale_banks: Iterable[dict],
    *,
    r8m_params: MechanismCorrectedR8Parameters,
    r8_params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
) -> dict:
    banks = list(scale_banks)
    adjusted = [
        _augment_bank_costs(bank, r8_params=r8_params, r8m_params=r8m_params)
        for bank in banks
    ]
    solved = solve_multiscale_state_path(adjusted, params=r8_params)
    e0_long = float(adjusted[-1]["e0_estimate_deg"])
    best_nonbaseline = solved["best_nonbaseline_final_path"]
    raw = float(best_nonbaseline["states"][-1]["theta_deg"]) if best_nonbaseline is not None else e0_long
    endpoint = float(raw if solved["adopt_joint_path"] else e0_long)
    scale_diag = [dict(bank["r8m_scale_diagnostics"]) for bank in adjusted]
    selected = best_nonbaseline if solved["adopt_joint_path"] else solved["all_e0_null_path"]
    return {
        "estimate_deg": endpoint,
        "raw_estimate_deg": raw,
        "e0_estimate_deg": e0_long,
        "adopt_joint_path": bool(solved["adopt_joint_path"]),
        "persistence_pass": bool(solved["persistence_pass"]),
        "model_selection_gain": float(solved["model_selection_gain"]),
        "best_path_margin": float(solved["best_path_margin"]),
        "best_nonbaseline_support_count": int(
            best_nonbaseline["supported_nonbaseline_scale_count"] if best_nonbaseline is not None else 0
        ),
        "best_nonbaseline_direction_spread_deg": float(
            best_nonbaseline["supported_direction_spread_deg"] if best_nonbaseline is not None else np.nan
        ),
        "selected_theta_path_deg": [float(s["theta_deg"]) for s in selected["states"]],
        "scale_diagnostics": scale_diag,
        "parameters": r8m_params.as_dict(),
        "truth_used_for_estimation": False,
        "task_identity_used_for_estimation": False,
        "future_context_used": False,
    }


def r8m_parameter_grid() -> list[MechanismCorrectedR8Parameters]:
    """Small preregistered development grid; includes (0,0) as exact R8 audit."""
    assignment = (0.0, 0.03, 0.10, 0.30, 1.00)
    anchor = (0.0, 0.01, 0.03, 0.10, 0.30)
    return [MechanismCorrectedR8Parameters(a, b) for a in assignment for b in anchor]
