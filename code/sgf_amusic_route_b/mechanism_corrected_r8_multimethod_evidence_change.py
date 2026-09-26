# =========================================================
# File        : mechanism_corrected_r8_multimethod_evidence_change.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the mechanism corrected r8 multimethod evidence change module used by the reproducibility workflow.
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
"""Parallel final validation of reliability structures with an evidence-change guard.

All methods reuse the same frozen R8 state banks and the same zero-parameter
admissibility rule:

    no long-scale pair-peak reassignment -> do not replace continuous E0.

The three final candidate structures are:

A. adaptive_geometry_fixed_evidence
       lambda * (r * Cg + D)
B. reliability_prior_code
       lambda * (Cg + D) + r * I(refined)
C. full_adaptive_freedom
       lambda * r * (Cg + D)

The prior shared-log structure is retained as a reference only:
D. shared_log_reference
       lambda * (Cg + r * D)

where Cg is the geometry-aware reassignment description, D is the one-sided
peak-evidence debt, and r is the frozen truth-free E0 reliability factor.

No task, array identity, source truth, localization error, future context, or
fitted threshold enters the estimator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .causal_multiscale_hodge import CausalMultiscaleHodgeParameters, solve_multiscale_state_path
from .mechanism_corrected_r8_evidence_change import _reassigned_pair_count
from .mechanism_corrected_r8_structural_funnel import (
    FROZEN_ANCHOR_STRENGTH,
    FROZEN_BASE_ASSIGNMENT_STRENGTHS,
    STRUCTURAL_VARIANTS,
    _augment_bank_costs_structural,
)

FINAL_CANDIDATE_VARIANTS = (
    "adaptive_geometry_fixed_evidence",
    "reliability_prior_code",
    "full_adaptive_freedom",
)
REFERENCE_VARIANTS = ("shared_log_reference",)
ALL_VALIDATION_VARIANTS = FINAL_CANDIDATE_VARIANTS + REFERENCE_VARIANTS
if tuple(STRUCTURAL_VARIANTS) != ALL_VALIDATION_VARIANTS:
    raise RuntimeError("Frozen structural-variant ordering changed")

FROZEN_ASSIGNMENT_STRENGTHS = tuple(float(x) for x in FROZEN_BASE_ASSIGNMENT_STRENGTHS)
FROZEN_ANCHOR_STRENGTH = float(FROZEN_ANCHOR_STRENGTH)
MULTIMETHOD_VARIANT_ID = "r8m_multimethod_evidence_change_validation"


@dataclass(frozen=True)
class MultiMethodParameters:
    base_assignment_strength: float = 1.5
    anchor_strength: float = FROZEN_ANCHOR_STRENGTH

    def as_dict(self) -> dict:
        return asdict(self)


def multimethod_parameter_grid() -> list[MultiMethodParameters]:
    return [MultiMethodParameters(a, FROZEN_ANCHOR_STRENGTH) for a in FROZEN_ASSIGNMENT_STRENGTHS]


def rescore_prepared_r8m_multimethod_banks(
    scale_banks: Iterable[dict],
    *,
    structural_variant: str,
    r8m_params: MultiMethodParameters,
    r8_params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
) -> dict:
    """Rescore one frozen state bank under one predeclared structure + guard."""
    if structural_variant not in ALL_VALIDATION_VARIANTS:
        raise ValueError(f"Unknown validation variant: {structural_variant}")
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
    best = solved["best_nonbaseline_final_path"]
    pre_guard_adopt = bool(solved["adopt_joint_path"] and best is not None)
    raw = float(best["states"][-1]["theta_deg"]) if best is not None else e0_long

    baseline_long = adjusted[-1]["states"][0]
    pre_guard_state = baseline_long
    pre_guard_reassigned = 0
    if best is not None:
        pre_guard_state = best["states"][-1]
        pre_guard_reassigned = _reassigned_pair_count(pre_guard_state, baseline_long)

    replacement_allowed = bool(pre_guard_reassigned > 0)
    guard_triggered = bool(pre_guard_adopt and not replacement_allowed)
    final_adopt = bool(pre_guard_adopt and replacement_allowed)
    endpoint = float(raw if final_adopt else e0_long)
    selected_path = best if final_adopt else solved["all_e0_null_path"]
    selected_scale = selected_path["states"][-1]

    return {
        "estimate_deg": endpoint,
        "raw_estimate_deg": raw,
        "e0_estimate_deg": e0_long,
        "adopt_joint_path": final_adopt,
        "pre_guard_adopt_joint_path": pre_guard_adopt,
        "replacement_allowed_by_reassignment": replacement_allowed,
        "no_reassignment_guard_triggered": guard_triggered,
        "pre_guard_long_reassigned_pair_count": int(pre_guard_reassigned),
        "selected_long_reassigned_pair_count": int(_reassigned_pair_count(selected_scale, baseline_long)),
        "persistence_pass": bool(solved["persistence_pass"]),
        "model_selection_gain": float(solved["model_selection_gain"]),
        "best_path_margin": float(solved["best_path_margin"]),
        "best_nonbaseline_support_count": int(
            best["supported_nonbaseline_scale_count"] if best is not None else 0
        ),
        "best_nonbaseline_direction_spread_deg": float(
            best["supported_direction_spread_deg"] if best is not None else np.nan
        ),
        "selected_theta_path_deg": [float(s["theta_deg"]) for s in selected_path["states"]],
        "pre_guard_selected_theta_path_deg": [float(s["theta_deg"]) for s in best["states"]] if best is not None else [],
        "pre_guard_long_geometry_assignment_description_per_edge": float(
            pre_guard_state.get("r8ms_geometry_assignment_description_per_edge", 0.0)
        ),
        "pre_guard_long_peak_evidence_debt_mean_changed": float(
            pre_guard_state.get("r8ms_peak_evidence_debt_mean_changed", 0.0)
        ),
        "selected_long_geometry_assignment_description_per_edge": float(
            selected_scale.get("r8ms_geometry_assignment_description_per_edge", 0.0)
        ),
        "selected_long_peak_evidence_debt_mean_changed": float(
            selected_scale.get("r8ms_peak_evidence_debt_mean_changed", 0.0)
        ),
        "selected_long_reliability_factor": float(selected_scale.get("r8ms_reliability_factor", 0.0)),
        "pre_guard_long_reliability_factor": float(pre_guard_state.get("r8ms_reliability_factor", 0.0)),
        "selected_long_assignment_extra_cost": float(selected_scale.get("r8ms_assignment_extra_cost", 0.0)),
        "selected_long_reliability_prior_extra_cost": float(
            selected_scale.get("r8ms_reliability_prior_extra_cost", 0.0)
        ),
        "scale_diagnostics": [dict(bank["r8ms_scale_diagnostics"]) for bank in adjusted],
        "structural_variant": structural_variant,
        "validation_variant_id": MULTIMETHOD_VARIANT_ID,
        "parameters": r8m_params.as_dict(),
        "truth_used_for_estimation": False,
        "task_identity_used_for_estimation": False,
        "array_label_used_for_estimation": False,
        "future_context_used": False,
    }
