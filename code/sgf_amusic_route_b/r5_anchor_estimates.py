# =========================================================
# File        : r5_anchor_estimates.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the r5 anchor estimates module used by the reproducibility workflow.
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
"""Exact targeted extraction of the two R5 GCC-Huber anchors.

This is a computational shortcut for two already-defined R4 factorial corners;
it does not introduce a new estimator.  Synthetic regression checks compare it
against ``baseline_factorial_estimates`` exactly.
"""
from __future__ import annotations

import numpy as np

from .baseline_extension import GCCPHATParameters, gcc_phat_tde_estimate
from .baseline_forensic import (
    BaselineForensicParameters,
    _fit_grid_and_refine,
    extract_shared_pair_bank,
)

E0 = "gcc_frac_equal_nogeom_huber"
E1 = "gcc_phat_tde_huber"


def r5_anchor_estimates(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: BaselineForensicParameters = BaselineForensicParameters(),
) -> tuple[dict[str, float], dict]:
    bank = extract_shared_pair_bank(
        audio, sample_rate_hz, microphone_positions_m, params=params
    )
    e0, e0_diag = _fit_grid_and_refine(
        bank["fractional_delays_s"],
        np.ones(len(bank["pairs"]), dtype=float),
        bank["positions"],
        bank["pairs"],
        sample_rate_hz,
        huber=True,
        params=params,
    )
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
    e1, e1_diag = gcc_phat_tde_estimate(
        audio, sample_rate_hz, microphone_positions_m, params=frozen_params
    )
    return {E0: float(e0), E1: float(e1)}, {
        "e0_diagnostics": e0_diag,
        "e1_diagnostics": e1_diag,
        "pair_count": int(len(bank["pairs"])),
        "fingerprints": dict(bank["fingerprints"]),
    }
