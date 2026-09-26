# =========================================================
# File        : mechanism_corrected_r8_structural_funnel.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the mechanism corrected r8 structural funnel module used by the reproducibility workflow.
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
"""Final structural reliability funnel for mechanism-corrected R8-M.

This module compares four *predeclared* structures on the same frozen R8 state
banks.  The first three are the final mechanism hypotheses; the fourth is the
already-evaluated shared-log Peak-Evidence structure retained as a reference.

Let
    Cg : geometry-aware reassignment description length,
    D  : one-sided Peak-Evidence debt,
    u  : truth-free E0 uncertainty ratio,
    r  : max(0, -log(min(u,1))).

Variants
--------
1. adaptive_geometry_fixed_evidence
       C = lambda * (r * Cg + D)
   Reliability controls *model freedom* while weak alternative peaks always
   retain their observation-evidence cost.  This is the primary hypothesis.

2. reliability_prior_code
       C = lambda * (Cg + D) + r * I(refined state)
   Reliability is an additive MDL prior code for entering the refinement model.
   It is the most literal code-length interpretation: when E0 is reliable,
   selecting the more complex refinement model requires extra prior evidence.

3. full_adaptive_freedom
       C = lambda * r * (Cg + D)
   Reliability scales the complete reassignment description.  This is the
   direct model-freedom hypothesis suggested by the final Task-1 conflict.

4. shared_log_reference
       C = lambda * (Cg + r * D)
   Frozen reference matching the prior best Shared-log coupling structure.

The already-existing uncertainty-adaptive direction anchor is unchanged:
       anchor = 0.03 * r * Huber(delta_theta / scale).

No task identity, source truth, localization error, array label, fitted
threshold, or new learned feature enters the estimator.  Only lambda is varied
on a small fixed grid, and all variants reuse exactly the same state banks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
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
)
from .mechanism_corrected_r8_peak_evidence import peak_evidence_debt_mean_changed

_EPS = 1.0e-12
R8M_FINAL_STRUCTURAL_SELECTIVE = "gcc_topk_causal_multiscale_hodge_r8m_final_structural"
FROZEN_ANCHOR_STRENGTH = 0.03
FROZEN_BASE_ASSIGNMENT_STRENGTHS = (0.75, 1.00, 1.25, 1.50, 2.00)
STRUCTURAL_VARIANTS = (
    "adaptive_geometry_fixed_evidence",
    "reliability_prior_code",
    "full_adaptive_freedom",
    "shared_log_reference",
)


@dataclass(frozen=True)
class StructuralR8MParameters:
    base_assignment_strength: float = 1.0
    anchor_strength: float = FROZEN_ANCHOR_STRENGTH

    def as_dict(self) -> dict:
        return asdict(self)


def structural_assignment_cost(
    *,
    geometry_description: float,
    evidence_debt: float,
    reliability: float,
    is_refined_state: bool,
    base_assignment_strength: float,
    variant: str,
) -> tuple[float, float]:
    """Return (assignment_extra, reliability_prior_extra)."""
    if variant not in STRUCTURAL_VARIANTS:
        raise ValueError(f"Unknown structural variant: {variant}")
    g = max(float(geometry_description), 0.0)
    d = max(float(evidence_debt), 0.0)
    r = max(float(reliability), 0.0)
    lam = max(float(base_assignment_strength), 0.0)

    if variant == "adaptive_geometry_fixed_evidence":
        return float(lam * (r * g + d)), 0.0
    if variant == "reliability_prior_code":
        prior = float(r if is_refined_state else 0.0)
        return float(lam * (g + d)), prior
    if variant == "full_adaptive_freedom":
        return float(lam * r * (g + d)), 0.0
    # Frozen previous winner/reference.
    return float(lam * (g + r * d)), 0.0


def _augment_bank_costs_structural(
    bank: dict,
    *,
    structural_variant: str,
    r8_params: CausalMultiscaleHodgeParameters,
    params: StructuralR8MParameters,
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
        is_refined = not bool(state.get("is_baseline", False))
        assignment_extra, prior_extra = structural_assignment_cost(
            geometry_description=geometry_description,
            evidence_debt=evidence_debt,
            reliability=reliability,
            is_refined_state=is_refined,
            base_assignment_strength=float(params.base_assignment_strength),
            variant=structural_variant,
        )

        direction_delta = _circular_abs_deg(float(state["theta_deg"]), e0_theta)
        direction_normalized = direction_delta / max(float(r8_params.direction_transition_scale_deg), _EPS)
        anchor_shape = float(_huber(direction_normalized, delta=1.0))
        anchor_extra = float(params.anchor_strength) * reliability * anchor_shape
        original_node = float(state["node_cost"])

        state["r8_node_cost_original"] = original_node
        state["r8ms_structural_variant"] = structural_variant
        state["r8ms_geometry_assignment_description_per_edge"] = float(geometry_description)
        state["r8ms_peak_evidence_debt_mean_changed"] = float(evidence_debt)
        state["r8ms_e0_uncertainty_ratio"] = float(uncertainty)
        state["r8ms_reliability_factor"] = float(reliability)
        state["r8ms_assignment_extra_cost"] = float(assignment_extra)
        state["r8ms_reliability_prior_extra_cost"] = float(prior_extra)
        state["r8ms_direction_delta_from_e0_deg"] = float(direction_delta)
        state["r8ms_anchor_shape"] = float(anchor_shape)
        state["r8ms_anchor_extra_cost"] = float(anchor_extra)
        state["node_cost"] = float(original_node + assignment_extra + prior_extra + anchor_extra)

    out["r8ms_scale_diagnostics"] = {
        "structural_variant": structural_variant,
        "e0_uncertainty_ratio": float(uncertainty),
        "e0_anchor_reliability": float(reliability),
        "reliability_factor": float(reliability),
        "median_pair_baseline_m": float(np.median(baselines)) if len(baselines) else float("nan"),
        "maximum_pair_baseline_m": float(np.max(baselines)) if len(baselines) else float("nan"),
    }
    return out


def rescore_prepared_r8m_structural_banks(
    scale_banks: Iterable[dict],
    *,
    structural_variant: str,
    r8m_params: StructuralR8MParameters,
    r8_params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
) -> dict:
    banks = list(scale_banks)
    adjusted = [
        _augment_bank_costs_structural(
            bank,
            structural_variant=structural_variant,
            r8_params=r8_params,
            params=r8m_params,
        )
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
            selected_scale.get("r8ms_geometry_assignment_description_per_edge", 0.0)
        ),
        "selected_long_peak_evidence_debt_mean_changed": float(
            selected_scale.get("r8ms_peak_evidence_debt_mean_changed", 0.0)
        ),
        "selected_long_reliability_factor": float(selected_scale.get("r8ms_reliability_factor", 0.0)),
        "selected_long_assignment_extra_cost": float(selected_scale.get("r8ms_assignment_extra_cost", 0.0)),
        "selected_long_reliability_prior_extra_cost": float(
            selected_scale.get("r8ms_reliability_prior_extra_cost", 0.0)
        ),
        "scale_diagnostics": [dict(bank["r8ms_scale_diagnostics"]) for bank in adjusted],
        "structural_variant": structural_variant,
        "parameters": r8m_params.as_dict(),
        "truth_used_for_estimation": False,
        "task_identity_used_for_estimation": False,
        "future_context_used": False,
    }


def structural_parameter_grid() -> list[StructuralR8MParameters]:
    return [
        StructuralR8MParameters(float(a), FROZEN_ANCHOR_STRENGTH)
        for a in FROZEN_BASE_ASSIGNMENT_STRENGTHS
    ]
