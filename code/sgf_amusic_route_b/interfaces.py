# =========================================================
# File        : interfaces.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the interfaces module used by the reproducibility workflow.
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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CovarianceEvidence:
    """One narrowband covariance observation and its reliability metadata.

    ``sensor_indices`` maps the covariance rows/columns to the original array
    geometry.  It is optional for backward compatibility; ``None`` means that
    all microphones are used in their original order.
    """

    covariance: np.ndarray
    frequency_hz: float | None = None
    weight: float = 1.0
    energy: float = 0.0
    eigen_gap: float = 0.0
    diffuseness: float = 1.0
    valid: bool = True
    snapshot_count: int = 0
    sensor_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAllocation:
    """Auditable representation of a candidate multi-source allocation."""

    angles_deg: np.ndarray
    basin_ids: tuple[int, ...] = field(default_factory=tuple)
    allocation_type: str = "unknown"
    scalar_dml: float = float("nan")
    residual_matrix: np.ndarray | None = None
    source_powers: np.ndarray | None = None
    incremental_evidence: np.ndarray | None = None
    bootstrap_support: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrequencyCluster:
    frequencies_hz: np.ndarray
    indices: np.ndarray
    distance_matrix: np.ndarray
    within_cluster_dispersion: float
    direct_path_score: float
    selected: bool


@dataclass(frozen=True)
class DOAEstimate:
    angles_deg: np.ndarray
    source_count: int
    confidence: np.ndarray
    selection_mode: str
    selected_frequencies_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    audit_status: str = "not_run"
    diagnostics: dict[str, Any] = field(default_factory=dict)
