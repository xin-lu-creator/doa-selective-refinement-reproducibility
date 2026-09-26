# =========================================================
# File        : mechanism_corrected_r8_evidence_change.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the mechanism corrected r8 evidence change module used by the reproducibility workflow.
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
"""Final R8-M evidence-change refinement.

This stage freezes the best previous structural hypothesis (full reliability-
adaptive freedom) and adds one zero-parameter structural admissibility rule:

    no pair-peak reassignment -> no replacement of E0.

The rule is not a numerical threshold.  A refinement may replace E0 only when
its selected long-scale state changes at least one frozen pair-level GCC peak
index relative to the E0 null state.  If the selected nonbaseline path keeps
exactly the same pair-peak assignment, the path is interpreted as a pure
quantization/path-grid alternative and E0 is retained.

The cost remains
    lambda * r(u) * (C_geometry + D_peak)
plus the already frozen uncertainty-adaptive anchor (0.03).

No task, array label, truth, localization error, or future context enters the
estimator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .causal_multiscale_hodge import CausalMultiscaleHodgeParameters, solve_multiscale_state_path
from .mechanism_corrected_r8_structural_funnel import (
    FROZEN_ANCHOR_STRENGTH,
    FROZEN_BASE_ASSIGNMENT_STRENGTHS,
    _augment_bank_costs_structural,
)

EVIDENCE_CHANGE_VARIANT = "full_adaptive_freedom_evidence_change"
FROZEN_ASSIGNMENT_STRENGTHS = tuple(float(x) for x in FROZEN_BASE_ASSIGNMENT_STRENGTHS)
FROZEN_EVIDENCE_CHANGE_ANCHOR = float(FROZEN_ANCHOR_STRENGTH)


@dataclass(frozen=True)
class EvidenceChangeParameters:
    base_assignment_strength: float = 1.5
    anchor_strength: float = FROZEN_EVIDENCE_CHANGE_ANCHOR

    def as_dict(self) -> dict:
        return asdict(self)


def evidence_change_parameter_grid() -> list[EvidenceChangeParameters]:
    return [EvidenceChangeParameters(a, FROZEN_EVIDENCE_CHANGE_ANCHOR) for a in FROZEN_ASSIGNMENT_STRENGTHS]


def _reassigned_pair_count(state: dict, baseline_state: dict) -> int:
    selected = np.asarray(state["indices"], dtype=int)
    baseline = np.asarray(baseline_state["indices"], dtype=int)
    if selected.shape != baseline.shape:
        raise ValueError("Selected and baseline pair-index vectors differ in shape")
    return int(np.sum(selected != baseline))


def rescore_prepared_r8m_evidence_change_banks(
    scale_banks: Iterable[dict],
    *,
    r8m_params: EvidenceChangeParameters,
    r8_params: CausalMultiscaleHodgeParameters = CausalMultiscaleHodgeParameters(),
) -> dict:
    """Rescore frozen R8 banks and enforce evidence-changing replacement.

    The nonbaseline path is solved exactly as in the previous full-adaptive
    structure.  Replacement is then admissible only if the *long-scale* state
    changes at least one pair peak index relative to the E0 null state.
    """
    banks = list(scale_banks)
    adjusted = [
        _augment_bank_costs_structural(
            bank,
            structural_variant="full_adaptive_freedom",
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

    pre_guard_reassigned = 0
    pre_guard_state = adjusted[-1]["states"][0]
    if best is not None:
        pre_guard_state = best["states"][-1]
        pre_guard_reassigned = _reassigned_pair_count(pre_guard_state, adjusted[-1]["states"][0])

    replacement_allowed = bool(pre_guard_reassigned > 0)
    guard_triggered = bool(pre_guard_adopt and not replacement_allowed)
    final_adopt = bool(pre_guard_adopt and replacement_allowed)
    endpoint = float(raw if final_adopt else e0_long)
    selected = best if final_adopt else solved["all_e0_null_path"]
    selected_scale = selected["states"][-1]

    return {
        "estimate_deg": endpoint,
        "raw_estimate_deg": raw,
        "e0_estimate_deg": e0_long,
        "adopt_joint_path": final_adopt,
        "pre_guard_adopt_joint_path": pre_guard_adopt,
        "replacement_allowed_by_reassignment": replacement_allowed,
        "no_reassignment_guard_triggered": guard_triggered,
        "pre_guard_long_reassigned_pair_count": int(pre_guard_reassigned),
        "selected_long_reassigned_pair_count": int(
            _reassigned_pair_count(selected_scale, adjusted[-1]["states"][0])
        ),
        "persistence_pass": bool(solved["persistence_pass"]),
        "model_selection_gain": float(solved["model_selection_gain"]),
        "best_path_margin": float(solved["best_path_margin"]),
        "best_nonbaseline_support_count": int(
            best["supported_nonbaseline_scale_count"] if best is not None else 0
        ),
        "best_nonbaseline_direction_spread_deg": float(
            best["supported_direction_spread_deg"] if best is not None else np.nan
        ),
        "selected_theta_path_deg": [float(s["theta_deg"]) for s in selected["states"]],
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
        "variant": EVIDENCE_CHANGE_VARIANT,
        "parameters": r8m_params.as_dict(),
        "truth_used_for_estimation": False,
        "task_identity_used_for_estimation": False,
        "array_label_used_for_estimation": False,
        "future_context_used": False,
    }
