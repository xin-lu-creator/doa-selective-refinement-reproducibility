# =========================================================
# File        : array_model.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the array model module used by the reproducibility workflow.
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

import numpy as np


def steering_matrix_ula(theta_deg, num_sensors: int, d_over_lambda: float = 0.5) -> np.ndarray:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    m = np.arange(int(num_sensors), dtype=float)[:, None]
    return np.exp(1j * 2.0 * np.pi * d_over_lambda * m * np.sin(theta)[None, :])


def sample_covariance(X: np.ndarray, diagonal_loading: float = 0.0) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("X must have shape (num_sensors, num_snapshots)")
    R = (X @ X.conj().T) / X.shape[1]
    R = 0.5 * (R + R.conj().T)
    if diagonal_loading > 0:
        R = R + float(diagonal_loading) * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
    return R


def eig_sorted(R: np.ndarray):
    values, vectors = np.linalg.eigh(np.asarray(R))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def music_spectrum_from_covariance(
    R: np.ndarray,
    num_sources: int,
    theta_grid_deg,
    d_over_lambda: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    R = np.asarray(R)
    M = R.shape[0]
    if not 0 < int(num_sources) < M:
        raise ValueError("num_sources must satisfy 0 < K < num_sensors")
    _, vectors = eig_sorted(R)
    En = vectors[:, int(num_sources):]
    A = steering_matrix_ula(theta_grid_deg, M, d_over_lambda)
    denom = np.sum(np.abs(En.conj().T @ A) ** 2, axis=0).real
    denom = np.maximum(denom, 1e-14)
    P = 1.0 / denom
    if normalize:
        P = P / (np.max(P) + 1e-14)
    return P
