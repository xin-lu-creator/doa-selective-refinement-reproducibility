# =========================================================
# File        : frequency_mandala.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the frequency mandala module used by the reproducibility workflow.
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

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .mandala_matrix import residual_lift_matrix


@dataclass(frozen=True)
class FrequencyMandalaResult:
    distance_matrix: np.ndarray
    labels: np.ndarray
    selected_cluster: int
    selected_indices: np.ndarray
    cluster_scores: dict[int, float]
    cluster_dispersion: dict[int, float]


def _hermitian(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=complex)
    return 0.5 * (a + a.conj().T)


def regularize_hpd(matrix: np.ndarray, floor_relative: float = 1e-6) -> np.ndarray:
    """Project a covariance matrix onto a well-conditioned HPD cone."""
    h = _hermitian(matrix)
    values, vectors = np.linalg.eigh(h)
    scale = max(float(np.max(values.real)), float(np.trace(h).real / max(len(values), 1)), 1e-12)
    values = np.maximum(values.real, float(floor_relative) * scale)
    return _hermitian((vectors * values[None, :]) @ vectors.conj().T)


def matrix_log_hpd(matrix: np.ndarray, floor_relative: float = 1e-6) -> np.ndarray:
    h = regularize_hpd(matrix, floor_relative=floor_relative)
    values, vectors = np.linalg.eigh(h)
    return _hermitian((vectors * np.log(values)[None, :]) @ vectors.conj().T)




def principal_phase_focused_covariance(
    covariance: np.ndarray,
    *,
    trace_normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Remove the dominant frequency-dependent steering phase.

    For a single direct-path plane wave R ~= p aa^H + noise, the principal
    eigenvector has the phase of a.  Congruence by its conjugate phase maps
    the direct component close to an all-ones rank-one matrix.  Frequency
    clustering then reflects residual/multipath structure rather than the
    deterministic phase rotation caused by changing wavelength.
    """
    R = regularize_hpd(covariance)
    values, vectors = np.linalg.eigh(R)
    v = vectors[:, int(np.argmax(values.real))]
    phase = v / np.maximum(np.abs(v), float(eps))
    D = np.diag(np.conj(phase))
    focused = _hermitian(D @ R @ D.conj().T)
    if trace_normalize:
        focused = focused / max(float(np.trace(focused).real), float(eps))
    return focused

def log_euclidean_distance(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    floor_relative: float = 1e-6,
) -> float:
    delta = matrix_log_hpd(matrix_a, floor_relative) - matrix_log_hpd(matrix_b, floor_relative)
    return float(np.linalg.norm(delta, "fro"))


def frequency_distance_matrix(
    covariances: Sequence[np.ndarray],
    *,
    floor_relative: float = 1e-6,
) -> np.ndarray:
    covs = [regularize_hpd(x, floor_relative) for x in covariances]
    n = len(covs)
    if n == 0:
        raise ValueError("At least one covariance matrix is required")
    logs = [matrix_log_hpd(x, floor_relative) for x in covs]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(np.linalg.norm(logs[i] - logs[j], "fro"))
            D[i, j] = D[j, i] = value
    return D


def log_euclidean_mean(
    covariances: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    *,
    floor_relative: float = 1e-6,
) -> np.ndarray:
    covs = list(covariances)
    if not covs:
        raise ValueError("At least one covariance matrix is required")
    if weights is None:
        w = np.ones(len(covs), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(covs),):
            raise ValueError("weights must have one value per covariance")
    w = np.clip(w, 0.0, None)
    if not np.any(w > 0):
        raise ValueError("At least one positive weight is required")
    w /= np.sum(w)
    mean_log = sum(weight * matrix_log_hpd(cov, floor_relative) for weight, cov in zip(w, covs))
    values, vectors = np.linalg.eigh(_hermitian(mean_log))
    return regularize_hpd((vectors * np.exp(values)[None, :]) @ vectors.conj().T, floor_relative)




def cluster_precomputed_distance(
    distance_matrix: np.ndarray,
    *,
    qualities: Sequence[float] | None = None,
    num_clusters: int = 2,
    min_cluster_size: int = 2,
    quality_weight: float = 0.25,
) -> FrequencyMandalaResult:
    """Cluster a transparent precomputed frequency-distance matrix."""
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] == 0:
        raise ValueError("distance_matrix must be non-empty and square")
    if not np.allclose(D, D.T, atol=1e-10) or not np.allclose(np.diag(D), 0.0, atol=1e-10):
        raise ValueError("distance_matrix must be symmetric with zero diagonal")
    n = len(D)
    if n == 1:
        return FrequencyMandalaResult(D, np.ones(1, dtype=int), 1, np.array([0]), {1: 0.0}, {1: 0.0})
    k = max(1, min(int(num_clusters), n))
    tree = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(tree, t=k, criterion="maxclust").astype(int)
    if qualities is None:
        q = np.ones(n, dtype=float)
    else:
        q = np.asarray(qualities, dtype=float)
        if q.shape != (n,):
            raise ValueError("qualities must have one value per frequency")
        q = (q - np.min(q)) / max(float(np.max(q) - np.min(q)), 1e-12)
    scores: dict[int, float] = {}
    dispersions: dict[int, float] = {}
    valid = []
    for label in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == label)
        sub = D[np.ix_(ids, ids)]
        dispersion = 0.0 if len(ids) <= 1 else float(np.sum(sub) / (len(ids) * (len(ids) - 1)))
        dispersions[label] = dispersion
        size_reward = float(np.log1p(len(ids)) / np.log1p(n))
        scores[label] = dispersion - float(quality_weight) * float(np.mean(q[ids])) - 0.08 * size_reward
        if len(ids) >= int(min_cluster_size):
            valid.append(label)
    candidates = valid or sorted(scores)
    selected = min(candidates, key=lambda label: scores[label])
    ids = np.flatnonzero(labels == selected)
    return FrequencyMandalaResult(D, labels, int(selected), ids, scores, dispersions)

def cluster_frequency_covariances(
    covariances: Sequence[np.ndarray],
    *,
    qualities: Sequence[float] | None = None,
    num_clusters: int = 2,
    min_cluster_size: int = 2,
    quality_weight: float = 0.25,
    floor_relative: float = 1e-6,
) -> FrequencyMandalaResult:
    """Cluster frequency covariances and select the most coherent reliable group.

    The score rewards low within-cluster Log-Euclidean dispersion, high external
    quality (energy/eigengap/directness supplied by the caller), and adequate
    cluster size.  It is intentionally transparent for scientific ablation.
    """
    D = frequency_distance_matrix(covariances, floor_relative=floor_relative)
    n = len(D)
    if n == 1:
        return FrequencyMandalaResult(D, np.ones(1, dtype=int), 1, np.array([0]), {1: 0.0}, {1: 0.0})
    k = max(1, min(int(num_clusters), n))
    condensed = squareform(D, checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=k, criterion="maxclust").astype(int)
    if qualities is None:
        q = np.ones(n, dtype=float)
    else:
        q = np.asarray(qualities, dtype=float)
        if q.shape != (n,):
            raise ValueError("qualities must have one value per covariance")
        q_min, q_max = float(np.min(q)), float(np.max(q))
        q = (q - q_min) / max(q_max - q_min, 1e-12)

    scores: dict[int, float] = {}
    dispersions: dict[int, float] = {}
    valid_clusters: list[int] = []
    for label in sorted(set(labels.tolist())):
        ids = np.flatnonzero(labels == label)
        if len(ids) <= 1:
            dispersion = 0.0
        else:
            sub = D[np.ix_(ids, ids)]
            dispersion = float(np.sum(sub) / (len(ids) * (len(ids) - 1)))
        dispersions[label] = dispersion
        size_reward = float(np.log1p(len(ids)) / np.log1p(n))
        quality_reward = float(np.mean(q[ids]))
        # Lower score is better.
        scores[label] = dispersion - float(quality_weight) * quality_reward - 0.05 * size_reward
        if len(ids) >= int(min_cluster_size):
            valid_clusters.append(label)
    candidates = valid_clusters or sorted(scores)
    selected = min(candidates, key=lambda label: scores[label])
    ids = np.flatnonzero(labels == selected)
    return FrequencyMandalaResult(D, labels, int(selected), ids, scores, dispersions)


def wideband_matrix_residual_score(
    covariances: Sequence[np.ndarray],
    theta_deg: Sequence[float],
    d_over_lambda_by_frequency: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
) -> dict:
    """Aggregate matrix-lifted DML residuals over selected frequency bins."""
    covs = list(covariances)
    dvals = np.asarray(d_over_lambda_by_frequency, dtype=float)
    if len(covs) != len(dvals):
        raise ValueError("One d/lambda value is required per frequency covariance")
    if weights is None:
        w = np.ones(len(covs), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(covs),):
            raise ValueError("weights must have one value per frequency")
    w = np.clip(w, 0.0, None)
    w /= max(np.sum(w), 1e-12)
    lifts = [residual_lift_matrix(R, theta_deg, d) for R, d in zip(covs, dvals)]
    traces = np.asarray([float(np.trace(M).real) for M in lifts])
    mean_matrix = sum(weight * M for weight, M in zip(w, lifts))
    return {
        "score": float(np.dot(w, traces)),
        "per_frequency_trace": traces,
        "weighted_residual_matrix": _hermitian(mean_matrix),
        "frequency_consensus_std": float(np.sqrt(np.dot(w, (traces - np.dot(w, traces)) ** 2))),
    }


@dataclass(frozen=True)
class SoftFrequencyMandalaResult:
    """Continuous Frequency-Mandala reliability weights.

    No frequency is discarded solely because of its cluster label.  The
    structural weight is multiplied by the pre-existing physical quality
    score and is tempered until the requested effective number of bins is
    reached.
    """

    distance_matrix: np.ndarray
    structural_weights: np.ndarray
    combined_weights: np.ndarray
    robust_center_index: int
    effective_bin_count: float
    effective_bandwidth_hz: float
    selected_indices: np.ndarray
    diagnostics: dict


def effective_sample_size(weights: Sequence[float], eps: float = 1e-12) -> float:
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(np.sum(w))
    if total <= eps:
        return 0.0
    return float(total * total / max(float(np.sum(w * w)), eps))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.clip(np.asarray(weights, dtype=float)[order], 0.0, None)
    if not np.any(w > 0):
        return float(np.median(v))
    cumulative = np.cumsum(w)
    return float(v[int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])


def _weighted_bandwidth(frequencies_hz: np.ndarray, weights: np.ndarray, mass: float = 0.90) -> float:
    f = np.asarray(frequencies_hz, dtype=float)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if len(f) <= 1 or not np.any(w > 0):
        return 0.0
    order = np.argsort(f)
    f = f[order]
    w = w[order]
    w = w / np.sum(w)
    cdf = np.cumsum(w)
    tail = max((1.0 - float(mass)) / 2.0, 0.0)
    lo = float(np.interp(tail, cdf, f))
    hi = float(np.interp(1.0 - tail, cdf, f))
    return max(hi - lo, 0.0)


def soft_frequency_mandala_weights(
    covariances: Sequence[np.ndarray],
    frequencies_hz: Sequence[float],
    *,
    qualities: Sequence[float] | None = None,
    doa_peaks_deg: Sequence[float] | None = None,
    floor_relative: float = 1e-6,
    covariance_temperature: float = 1.25,
    doa_temperature_deg: float = 12.0,
    minimum_factor: float = 0.20,
    target_effective_bins: float = 12.0,
    minimum_retained_bins: int = 8,
    maximum_bins: int | None = None,
    minimum_effective_bandwidth_hz: float = 900.0,
) -> SoftFrequencyMandalaResult:
    """Compute soft, bandwidth-preserving Frequency-Mandala weights.

    The robust structural centre is the quality-weighted medoid of the
    Log-Euclidean covariance-distance matrix.  Frequencies far from that
    centre and far from the robust DOA consensus are attenuated, never hard
    deleted.  A bounded floor and adaptive temperature prevent the effective
    support from collapsing to a few bins.
    """
    covs = list(covariances)
    f = np.asarray(frequencies_hz, dtype=float)
    if len(covs) == 0 or f.shape != (len(covs),):
        raise ValueError("One frequency is required per covariance")
    n = len(covs)
    if qualities is None:
        q = np.ones(n, dtype=float)
    else:
        q = np.asarray(qualities, dtype=float)
        if q.shape != (n,):
            raise ValueError("qualities must have one value per covariance")
    q = np.clip(q, 0.0, None)
    if not np.any(q > 0):
        q = np.ones(n, dtype=float)
    qn = q / np.max(q)
    q_med = max(float(np.median(qn[qn > 0])) if np.any(qn > 0) else 1.0, 1e-6)
    # Keep low-energy bins from receiving zero weight while retaining quality
    # ordering.  Square root avoids over-concentration on a few bins.
    q_soft = np.sqrt(np.clip(qn / q_med, 0.05, 4.0))

    D = frequency_distance_matrix(covs, floor_relative=floor_relative)
    # Quality-weighted medoid: robust to one compact reflection cluster.
    qprob = q_soft / np.sum(q_soft)
    medoid_cost = D @ qprob
    centre = int(np.argmin(medoid_cost))
    d = D[:, centre]
    nonzero = d[d > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    z_cov = d / max(scale, 1e-12)

    if doa_peaks_deg is not None:
        peaks = np.asarray(doa_peaks_deg, dtype=float)
        if peaks.shape != (n,):
            raise ValueError("doa_peaks_deg must have one value per covariance")
        doa_center = _weighted_median(peaks, q_soft)
        z_doa = np.abs(peaks - doa_center) / max(float(doa_temperature_deg), 1e-12)
    else:
        peaks = np.full(n, np.nan)
        doa_center = float("nan")
        z_doa = np.zeros(n, dtype=float)

    cov_temp = max(float(covariance_temperature), 1e-6)
    floor = float(np.clip(minimum_factor, 0.0, 1.0))

    def make_weights(temp_multiplier: float) -> tuple[np.ndarray, np.ndarray]:
        structural = np.exp(-0.5 * (z_cov / (cov_temp * temp_multiplier)) ** 2)
        structural *= np.exp(-0.5 * (z_doa / temp_multiplier) ** 2)
        structural = floor + (1.0 - floor) * structural
        combined = q_soft * structural
        return structural, combined

    multiplier = 1.0
    structural, combined = make_weights(multiplier)
    target = min(max(float(target_effective_bins), 1.0), float(n))
    # Relax attenuation until adequate effective support and bandwidth remain.
    for _ in range(16):
        neff = effective_sample_size(combined)
        bandwidth = _weighted_bandwidth(f, combined)
        if neff >= target and bandwidth >= min(float(minimum_effective_bandwidth_hz), float(np.ptp(f))):
            break
        multiplier *= 1.25
        structural, combined = make_weights(multiplier)

    order = np.argsort(combined)[::-1]
    keep_count = max(int(minimum_retained_bins), int(np.ceil(effective_sample_size(combined))))
    if maximum_bins is not None:
        keep_count = min(keep_count, int(maximum_bins))
    keep_count = min(max(keep_count, 1), n)
    selected = np.sort(order[:keep_count])
    # Do not zero discarded weights in the returned full vector; callers may
    # use every reliable bin. selected_indices is an auditable compute budget.
    combined = np.clip(combined, 0.0, None)
    combined /= max(float(np.sum(combined)), 1e-12)
    return SoftFrequencyMandalaResult(
        distance_matrix=D,
        structural_weights=structural,
        combined_weights=combined,
        robust_center_index=centre,
        effective_bin_count=effective_sample_size(combined),
        effective_bandwidth_hz=_weighted_bandwidth(f, combined),
        selected_indices=selected,
        diagnostics={
            "quality_weight_normalized": q_soft.tolist(),
            "covariance_distance_to_medoid": d.tolist(),
            "covariance_scale": scale,
            "doa_peaks_deg": peaks.tolist(),
            "doa_consensus_center_deg": doa_center,
            "temperature_multiplier": multiplier,
            "minimum_factor": floor,
            "target_effective_bins": target,
            "retained_bin_count": int(len(selected)),
        },
    )
