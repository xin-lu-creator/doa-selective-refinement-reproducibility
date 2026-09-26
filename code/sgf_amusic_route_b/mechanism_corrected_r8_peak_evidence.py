# =========================================================
# File        : mechanism_corrected_r8_peak_evidence.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the mechanism corrected r8 peak evidence module used by the reproducibility workflow.
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
"""Peak-evidence-corrected R8-M development implementation.

This module keeps the frozen R8 candidate/state bank and the already-tested
R8-M geometry-aware reassignment term plus uncertainty-adaptive E0 anchor.
It adds exactly one structural correction motivated by the Task-1 pair-level
forensics: a *one-sided evidence-debt surcharge* for reassignment decisions
that move a microphone pair from the E0 state's selected GCC peak to a weaker
GCC peak.

The surcharge is parameter-free once the existing assignment_strength is set:

    debt_e = max(0, log(A_E0,e / A_state,e))

and the mean debt across the changed pairs is added as a reassignment-quality
surcharge next to the geometry-aware assignment description before multiplying
by the existing assignment_strength.

Important design properties:
- selecting an equal/stronger peak than E0 adds zero evidence debt;
- selecting a weaker second/third peak adds a monotone evidence debt;
- no amplitude threshold, rank threshold, task label, truth, localization
  error, or future context is used;
- no third tuning parameter is introduced;
- assignment_strength=0 and anchor_strength=0 reproduces original frozen R8.

This is mechanism-development only on already-seen LOCATA evaluation arrays.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import log
from typing import Iterable

import numpy as np

from .causal_multiscale_hodge import (
    CausalMultiscaleHodgeParameters,
    _circular_abs_deg,
    _huber,
    solve_multiscale_state_path,
)
from .mechanism_corrected_r8 import (
    e0_anchor_reliability,
    e0_uncertainty_ratio_from_bank,
    geometry_assignment_description_per_edge,
    prepare_r8m_state_banks,
)

_EPS = 1.0e-12
R8M_PEAK_EVIDENCE_SELECTIVE = "gcc_topk_causal_multiscale_hodge_r8m_peak_evidence"


@dataclass(frozen=True)
class PeakEvidenceR8MParameters:
    """Same two degrees of freedom as the previous R8-M development stage."""

    assignment_strength: float = 0.0
    anchor_strength: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def peak_evidence_debt_mean_changed(
    *,
    changed_mask: np.ndarray,
    baseline_indices: np.ndarray,
    selected_indices: np.ndarray,
    candidate_amplitudes: np.ndarray,
) -> float:
    """Mean one-sided evidence loss across the pairs that are reassigned.

    Candidate amplitudes are the frozen GCC peak amplitudes normalized by each
    pair's primary peak.  For a changed edge e, only moves to a weaker peak than
    the E0 state are charged:

        max(0, log(A_E0 / A_state)).

    Moves to an equal or stronger peak carry no extra surcharge.  This is not a
    learned threshold and introduces no additional hyperparameter.
    """
    changed = np.asarray(changed_mask, dtype=bool).reshape(-1)
    base = np.asarray(baseline_indices, dtype=int).reshape(-1)
    selected = np.asarray(selected_indices, dtype=int).reshape(-1)
    amps = np.asarray(candidate_amplitudes, dtype=float)
    if amps.ndim != 2:
        raise ValueError("candidate_amplitudes must be a 2-D edge x peak array")
    if not (len(changed) == len(base) == len(selected) == amps.shape[0]):
        raise ValueError("changed mask, indices, and candidate amplitudes have incompatible shapes")
    changed_edges = np.flatnonzero(changed)
    if len(changed_edges) == 0:
        return 0.0
    debt = 0.0
    for edge in changed_edges:
        b = int(base[edge])
        s = int(selected[edge])
        if b < 0 or b >= amps.shape[1] or s < 0 or s >= amps.shape[1]:
            raise ValueError("peak index out of candidate-amplitude range")
        a_base = max(float(amps[edge, b]), _EPS)
        a_sel = max(float(amps[edge, s]), _EPS)
        debt += max(0.0, log(a_base / a_sel))
    return float(debt / float(len(changed_edges)))


def _augment_bank_costs_peak_evidence(
    bank: dict,
    *,
    r8_params: CausalMultiscaleHodgeParameters,
    params: PeakEvidenceR8MParameters,
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
    candidate_amplitudes = np.asarray(
        [row.get("candidate_amplitudes", []) for row in out.get("pair_rows", [])],
        dtype=float,
    )
    if candidate_amplitudes.shape[0] != len(base_indices):
        raise ValueError("pair_rows candidate amplitudes do not match R8 state graph")

    uncertainty = e0_uncertainty_ratio_from_bank(out)
    reliability = e0_anchor_reliability(uncertainty)
    e0_theta = float(baseline["theta_deg"])

    for state in states:
        indices = np.asarray(state["indices"], dtype=int)
        changed = indices != base_indices
        geometry_description = geometry_assignment_description_per_edge(
            changed_mask=changed,
            baselines_m=baselines,
            sample_rate_hz=float(out["sample_rate_hz"]),
        )
        evidence_debt = peak_evidence_debt_mean_changed(
            changed_mask=changed,
            baseline_indices=base_indices,
            selected_indices=indices,
            candidate_amplitudes=candidate_amplitudes,
        )
        combined_assignment = float(geometry_description + evidence_debt)

        direction_delta = _circular_abs_deg(float(state["theta_deg"]), e0_theta)
        direction_normalized = direction_delta / max(float(r8_params.direction_transition_scale_deg), _EPS)
        anchor_shape = float(_huber(direction_normalized, delta=1.0))
        assignment_extra = float(params.assignment_strength) * combined_assignment
        anchor_extra = float(params.anchor_strength) * reliability * anchor_shape
        original_node = float(state["node_cost"])

        state["r8_node_cost_original"] = original_node
        state["r8m_geometry_assignment_description_per_edge"] = float(geometry_description)
        state["r8m_peak_evidence_debt_mean_changed"] = float(evidence_debt)
        state["r8m_assignment_description_per_edge"] = combined_assignment
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


def rescore_prepared_r8m_peak_evidence_banks(
    scale_banks: Iterable[dict],
    *,
    r8m_params: PeakEvidenceR8MParameters,
    r8_params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
) -> dict:
    banks = list(scale_banks)
    adjusted = [
        _augment_bank_costs_peak_evidence(bank, r8_params=r8_params, params=r8m_params)
        for bank in banks
    ]
    solved = solve_multiscale_state_path(adjusted, params=r8_params)
    e0_long = float(adjusted[-1]["e0_estimate_deg"])
    best_nonbaseline = solved["best_nonbaseline_final_path"]
    raw = float(best_nonbaseline["states"][-1]["theta_deg"]) if best_nonbaseline is not None else e0_long
    endpoint = float(raw if solved["adopt_joint_path"] else e0_long)
    selected = best_nonbaseline if solved["adopt_joint_path"] else solved["all_e0_null_path"]

    selected_scale = selected["states"][-1]
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
        "selected_long_geometry_assignment_description_per_edge": float(
            selected_scale.get("r8m_geometry_assignment_description_per_edge", 0.0)
        ),
        "selected_long_peak_evidence_debt_mean_changed": float(
            selected_scale.get("r8m_peak_evidence_debt_mean_changed", 0.0)
        ),
        "selected_long_combined_assignment_description_per_edge": float(
            selected_scale.get("r8m_assignment_description_per_edge", 0.0)
        ),
        "scale_diagnostics": [dict(bank["r8m_scale_diagnostics"]) for bank in adjusted],
        "parameters": r8m_params.as_dict(),
        "truth_used_for_estimation": False,
        "task_identity_used_for_estimation": False,
        "future_context_used": False,
    }


def r8m_peak_evidence_parameter_grid() -> list[PeakEvidenceR8MParameters]:
    """Reuse the exact previous 25-point grid; no new search values are added."""
    assignment = (0.0, 0.03, 0.10, 0.30, 1.00)
    anchor = (0.0, 0.01, 0.03, 0.10, 0.30)
    return [PeakEvidenceR8MParameters(a, h) for a in assignment for h in anchor]
