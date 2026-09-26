# =========================================================
# File        : mandala_matrix.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the mandala matrix module used by the reproducibility workflow.
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

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import nnls

from .array_model import steering_matrix_ula


@dataclass(frozen=True)
class MatrixAuditParameters:
    """Matrix-lifted audit parameters for local/global JDML selection.

    The scalar concentrated-DML residual is the trace of the lifted residual
    matrix.  The audit retains the complete Hermitian residual structure and
    rejects a global solution that only improves the total trace by worsening
    a concentrated spatial mode, reducing residual diffuseness, or moving to
    an unsupported search boundary without sufficient evidence.
    """

    enabled: bool = True
    min_relative_trace_improvement: float = 5.0e-3
    boundary_relative_trace_improvement: float = 1.5e-2
    remote_relative_trace_improvement: float = 1.0e-2
    boundary_guard_deg: float = 1.0
    remote_from_basin_guard_deg: float = 3.0
    # Loewner ordering is recorded diagnostically but is not imposed exactly:
    # projectors onto different steering subspaces are rarely PSD-ordered.
    max_positive_mode_worsening_ratio: float = 1.0
    max_anisotropy_increase: float = 3.0e-2
    max_entropy_drop: float = 4.0e-2
    trace_tie_relative_tol: float = 2.5e-3
    reject_boundary_solutions: bool = True
    min_global_source_power_balance: float = 0.25
    min_global_to_local_balance_ratio: float = 1.0
    eps: float = 1.0e-12


MATRIX_AUDIT_DEFAULTS = MatrixAuditParameters()


def matrix_audit_parameter_dict(
    params: MatrixAuditParameters = MATRIX_AUDIT_DEFAULTS,
) -> dict:
    return asdict(params)


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=complex)
    return 0.5 * (x + x.conj().T)


def psd_square_root(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return the principal PSD square root of a Hermitian covariance matrix."""
    h = _hermitian(matrix)
    values, vectors = np.linalg.eigh(h)
    scale = max(float(np.max(np.abs(values))), 1.0)
    values = np.clip(values, 0.0, None)
    values[values < eps * scale] = 0.0
    return (vectors * np.sqrt(values)[None, :]) @ vectors.conj().T


def steering_projector(
    theta_deg: Sequence[float],
    num_sensors: int,
    d_over_lambda: float,
    *,
    rcond: float = 1e-12,
) -> np.ndarray:
    """Orthogonal projector onto the candidate steering subspace."""
    theta = np.asarray(theta_deg, dtype=float).reshape(-1)
    A = steering_matrix_ula(theta, int(num_sensors), float(d_over_lambda))
    gram = _hermitian(A.conj().T @ A)
    inv = np.linalg.pinv(gram, rcond=float(rcond), hermitian=True)
    P = A @ inv @ A.conj().T
    return _hermitian(P)


def residual_lift_matrix(
    R: np.ndarray,
    theta_deg: Sequence[float],
    d_over_lambda: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    r"""Lift the scalar DML residual into a matrix-valued residual.

    M_res(theta) = R^(1/2) (I - P_A(theta)) R^(1/2) / tr(R).

    Its trace equals the normalized concentrated-DML residual up to numerical
    roundoff, while its eigenstructure reveals unexplained spatial modes.
    """
    R_h = _hermitian(R)
    trace_R = max(float(np.trace(R_h).real), float(eps))
    root = psd_square_root(R_h, eps=eps)
    P = steering_projector(theta_deg, R_h.shape[0], d_over_lambda)
    M = root @ (np.eye(R_h.shape[0], dtype=complex) - P) @ root
    M = _hermitian(M) / trace_R
    # Finite-precision projection can create tiny negative eigenvalues. Project
    # back to the PSD cone without changing meaningful structure.
    values, vectors = np.linalg.eigh(M)
    values = np.clip(values, 0.0, None)
    return _hermitian((vectors * values[None, :]) @ vectors.conj().T)


def residual_matrix_metrics(matrix: np.ndarray, *, eps: float = 1e-12) -> dict:
    """Return trace, anisotropy and normalized spectral entropy diagnostics."""
    M = _hermitian(matrix)
    eig = np.clip(np.linalg.eigvalsh(M).real, 0.0, None)
    trace = float(np.sum(eig))
    if trace <= eps:
        return {
            "trace": trace,
            "lambda_max": 0.0,
            "lambda_max_ratio": 0.0,
            "spectral_entropy": 1.0,
            "effective_rank": float(len(eig)),
            "frobenius_norm": float(np.linalg.norm(M, "fro")),
            "eigenvalues": eig,
        }
    probabilities = eig / trace
    nz = probabilities[probabilities > eps]
    entropy = float(-np.sum(nz * np.log(nz)) / np.log(max(len(eig), 2)))
    return {
        "trace": trace,
        "lambda_max": float(eig[-1]),
        "lambda_max_ratio": float(eig[-1] / trace),
        "spectral_entropy": entropy,
        "effective_rank": float(np.exp(-np.sum(nz * np.log(nz)))),
        "frobenius_norm": float(np.linalg.norm(M, "fro")),
        "eigenvalues": eig,
    }


def source_covariance_metrics(
    R: np.ndarray,
    theta_deg: Sequence[float],
    d_over_lambda: float,
    *,
    eps: float = 1e-12,
) -> dict:
    """Estimate the de-traced source covariance supported by a DOA pair.

    A spurious remote angle often lowers the scalar DML trace by fitting sample
    noise while receiving almost no source power.  The 2x2 matrix estimate
    retains the source-to-source power and coherence structure and provides a
    physically interpretable balance test.
    """
    R_h = _hermitian(R)
    theta = np.asarray(theta_deg, dtype=float).reshape(-1)
    M = R_h.shape[0]
    K = len(theta)
    A = steering_matrix_ula(theta, M, d_over_lambda)
    gram = _hermitian(A.conj().T @ A)
    gram_inv = np.linalg.pinv(gram, rcond=eps, hermitian=True)
    A_dagger = gram_inv @ A.conj().T
    P = _hermitian(A @ gram_inv @ A.conj().T)
    residual_trace = max(float(np.trace((np.eye(M) - P) @ R_h).real), 0.0)
    noise_variance = residual_trace / max(M - K, 1)
    Q = A_dagger @ (R_h - noise_variance * np.eye(M)) @ A_dagger.conj().T
    Q = _hermitian(Q)
    eigenvalues = np.linalg.eigvalsh(Q).real
    diagonal_power = np.clip(np.diag(Q).real, 0.0, None)
    max_power = max(float(np.max(diagonal_power)), eps)
    power_balance = float(np.min(diagonal_power) / max_power)
    negative_mass = float(
        np.sum(np.clip(-eigenvalues, 0.0, None))
        / max(np.sum(np.abs(eigenvalues)), eps)
    )
    if K == 2:
        coherence = float(
            abs(Q[0, 1])
            / np.sqrt(max(float(Q[0, 0].real * Q[1, 1].real), eps))
        )
    else:
        coherence = float("nan")
    return {
        "matrix": Q,
        "noise_variance": float(noise_variance),
        "diagonal_power": diagonal_power,
        "power_balance": power_balance,
        "negative_eigenvalue_mass": negative_mass,
        "coherence": coherence,
        "trace": float(np.trace(Q).real),
        "eigenvalues": eigenvalues,
    }



def nonnegative_covariance_fit_metrics(
    R: np.ndarray,
    theta_deg: Sequence[float],
    d_over_lambda: float,
    *,
    eps: float = 1e-12,
) -> dict:
    r"""Fit a nonnegative diagonal source covariance plus white noise.

    The model is
        R ~= sum_k p_k a(theta_k)a(theta_k)^H + sigma^2 I,
    with p_k >= 0 and sigma^2 >= 0.  Unlike an unconstrained pseudoinverse
    estimate, this diagnostic remains interpretable for close or coherent
    candidate pairs and is therefore used as a Mandala local-allocation
    evidence measure.
    """
    R_h = _hermitian(R)
    theta = np.asarray(theta_deg, dtype=float).reshape(-1)
    M = int(R_h.shape[0])
    A = steering_matrix_ula(theta, M, float(d_over_lambda))
    atoms = [np.outer(A[:, k], A[:, k].conj()) for k in range(A.shape[1])]
    atoms.append(np.eye(M, dtype=complex))
    design = np.column_stack([atom.reshape(-1) for atom in atoms])
    target = R_h.reshape(-1)
    design_real = np.vstack([design.real, design.imag])
    target_real = np.concatenate([target.real, target.imag])
    coefficients, residual_norm = nnls(design_real, target_real)
    source_power = np.asarray(coefficients[:-1], dtype=float)
    noise_variance = float(coefficients[-1])
    max_power = max(float(np.max(source_power)) if source_power.size else 0.0, eps)
    power_balance = float(np.min(source_power) / max_power) if source_power.size else 0.0
    relative_fit_residual = float(residual_norm / max(np.linalg.norm(target_real), eps))
    total_source_power = float(np.sum(source_power))
    source_fraction = float(
        total_source_power / max(total_source_power + M * noise_variance, eps)
    )
    return {
        "source_power": source_power,
        "noise_variance": noise_variance,
        "power_balance": power_balance,
        "relative_fit_residual": relative_fit_residual,
        "source_fraction": source_fraction,
        "total_source_power": total_source_power,
    }

def distance_to_intervals(theta: float, intervals: Iterable[tuple[float, float]]) -> float:
    distances: list[float] = []
    for lo, hi in intervals:
        a, b = sorted((float(lo), float(hi)))
        if a <= float(theta) <= b:
            return 0.0
        distances.append(min(abs(float(theta) - a), abs(float(theta) - b)))
    return min(distances) if distances else float("inf")


def compare_residual_lifts(
    R: np.ndarray,
    local_theta: Sequence[float],
    global_theta: Sequence[float],
    *,
    d_over_lambda: float,
    theta_min: float,
    theta_max: float,
    intervals: Iterable[tuple[float, float]],
    params: MatrixAuditParameters = MATRIX_AUDIT_DEFAULTS,
) -> dict:
    """Compare local and global JDML solutions using Mandala matrix structure."""
    local = residual_lift_matrix(R, local_theta, d_over_lambda, eps=params.eps)
    global_ = residual_lift_matrix(R, global_theta, d_over_lambda, eps=params.eps)
    lm = residual_matrix_metrics(local, eps=params.eps)
    gm = residual_matrix_metrics(global_, eps=params.eps)
    local_source = source_covariance_metrics(
        R, local_theta, d_over_lambda, eps=params.eps
    )
    global_source = source_covariance_metrics(
        R, global_theta, d_over_lambda, eps=params.eps
    )
    delta = _hermitian(global_ - local)
    delta_eig = np.linalg.eigvalsh(delta).real
    positive_mode = max(float(delta_eig[-1]), 0.0)
    local_trace = max(float(lm["trace"]), params.eps)
    relative_improvement = float((lm["trace"] - gm["trace"]) / local_trace)
    mode_worsening_ratio = float(positive_mode / local_trace)
    anisotropy_increase = float(gm["lambda_max_ratio"] - lm["lambda_max_ratio"])
    entropy_drop = float(lm["spectral_entropy"] - gm["spectral_entropy"])

    global_theta_arr = np.sort(np.asarray(global_theta, dtype=float).reshape(-1))
    local_theta_arr = np.sort(np.asarray(local_theta, dtype=float).reshape(-1))
    boundary_distance = float(
        min(
            np.min(global_theta_arr - float(theta_min)),
            np.min(float(theta_max) - global_theta_arr),
        )
    )
    boundary_solution = bool(boundary_distance <= params.boundary_guard_deg + 1e-12)
    basin_distances = [distance_to_intervals(x, intervals) for x in global_theta_arr]
    max_basin_distance = float(max(basin_distances) if basin_distances else float("inf"))
    remote_solution = bool(max_basin_distance > params.remote_from_basin_guard_deg)
    displacement = float(np.max(np.abs(global_theta_arr - local_theta_arr)))

    required_improvement = float(params.min_relative_trace_improvement)
    if boundary_solution:
        required_improvement = max(
            required_improvement, float(params.boundary_relative_trace_improvement)
        )
    if remote_solution:
        required_improvement = max(
            required_improvement, float(params.remote_relative_trace_improvement)
        )

    trace_pass = bool(relative_improvement > required_improvement)
    mode_pass = bool(mode_worsening_ratio <= params.max_positive_mode_worsening_ratio)
    anisotropy_pass = bool(anisotropy_increase <= params.max_anisotropy_increase)
    entropy_pass = bool(entropy_drop <= params.max_entropy_drop)
    near_trace_tie = bool(relative_improvement <= params.trace_tie_relative_tol)
    local_balance = float(local_source["power_balance"])
    global_balance = float(global_source["power_balance"])
    balance_pass = bool(
        global_balance >= params.min_global_source_power_balance
        and global_balance
        >= params.min_global_to_local_balance_ratio * max(local_balance, params.eps)
    )
    boundary_pass = bool(
        not params.reject_boundary_solutions or not boundary_solution
    )
    accepted = bool(
        params.enabled
        and trace_pass
        and mode_pass
        and anisotropy_pass
        and entropy_pass
        and balance_pass
        and boundary_pass
        and not near_trace_tie
    )
    reasons: list[str] = []
    if not trace_pass:
        reasons.append("insufficient_relative_trace_improvement")
    if not mode_pass:
        reasons.append("positive_spatial_mode_worsening")
    if not anisotropy_pass:
        reasons.append("residual_anisotropy_increase")
    if not entropy_pass:
        reasons.append("residual_spectral_entropy_drop")
    if not balance_pass:
        reasons.append("insufficient_de_traced_source_power_balance")
    if not boundary_pass:
        reasons.append("boundary_solution_rejected")
    if near_trace_tie:
        reasons.append("trace_near_tie_local_preferred")
    if boundary_solution:
        reasons.append("global_solution_near_search_boundary")
    if remote_solution:
        reasons.append("global_solution_remote_from_candidate_basins")

    return {
        "accepted": accepted,
        "rejection_reasons": reasons,
        "relative_trace_improvement": relative_improvement,
        "required_relative_trace_improvement": required_improvement,
        "positive_mode_worsening": positive_mode,
        "positive_mode_worsening_ratio": mode_worsening_ratio,
        "anisotropy_increase": anisotropy_increase,
        "entropy_drop": entropy_drop,
        "boundary_solution": boundary_solution,
        "boundary_distance_deg": boundary_distance,
        "remote_solution": remote_solution,
        "max_distance_to_candidate_basin_deg": max_basin_distance,
        "max_local_global_displacement_deg": displacement,
        "local_trace": float(lm["trace"]),
        "global_trace": float(gm["trace"]),
        "local_source_power_balance": local_balance,
        "global_source_power_balance": global_balance,
        "source_balance_ratio": float(global_balance / max(local_balance, params.eps)),
        "local_source_negative_mass": float(local_source["negative_eigenvalue_mass"]),
        "global_source_negative_mass": float(global_source["negative_eigenvalue_mass"]),
        "local_source_coherence": float(local_source["coherence"]),
        "global_source_coherence": float(global_source["coherence"]),
        "local_source_covariance_trace": float(local_source["trace"]),
        "global_source_covariance_trace": float(global_source["trace"]),
        "local_lambda_max_ratio": float(lm["lambda_max_ratio"]),
        "global_lambda_max_ratio": float(gm["lambda_max_ratio"]),
        "local_spectral_entropy": float(lm["spectral_entropy"]),
        "global_spectral_entropy": float(gm["spectral_entropy"]),
        "local_effective_rank": float(lm["effective_rank"]),
        "global_effective_rank": float(gm["effective_rank"]),
        "delta_lambda_max": float(delta_eig[-1]),
        "delta_lambda_min": float(delta_eig[0]),
    }
