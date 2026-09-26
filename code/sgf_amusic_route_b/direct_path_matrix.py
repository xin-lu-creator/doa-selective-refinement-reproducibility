# =========================================================
# File        : direct_path_matrix.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the direct path matrix module used by the reproducibility workflow.
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

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from .interfaces import CovarianceEvidence


def _hermitian(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=complex)
    return 0.5 * (a + a.conj().T)


def regularize_hpd(matrix: np.ndarray, floor_relative: float = 1e-6) -> np.ndarray:
    h = _hermitian(matrix)
    values, vectors = np.linalg.eigh(h)
    scale = max(float(np.max(np.abs(values))), float(np.trace(h).real / max(len(h), 1)), 1.0)
    floor = float(floor_relative) * scale
    values = np.maximum(values.real, floor)
    return _hermitian((vectors * values[None, :]) @ vectors.conj().T)


def matrix_power_hpd(matrix: np.ndarray, power: float, floor_relative: float = 1e-6) -> np.ndarray:
    h = regularize_hpd(matrix, floor_relative)
    values, vectors = np.linalg.eigh(h)
    return _hermitian((vectors * np.power(values, power)[None, :]) @ vectors.conj().T)


def matrix_log_hpd(matrix: np.ndarray, floor_relative: float = 1e-6) -> np.ndarray:
    h = regularize_hpd(matrix, floor_relative)
    values, vectors = np.linalg.eigh(h)
    return _hermitian((vectors * np.log(values)[None, :]) @ vectors.conj().T)


def matrix_exp_hermitian(matrix: np.ndarray) -> np.ndarray:
    h = _hermitian(matrix)
    values, vectors = np.linalg.eigh(h)
    return _hermitian((vectors * np.exp(values)[None, :]) @ vectors.conj().T)


@dataclass(frozen=True)
class DirectPathDecomposition:
    observed_covariance: np.ndarray
    direct_covariance: np.ndarray
    late_covariance: np.ndarray
    whitened_covariance: np.ndarray
    whitener: np.ndarray
    generalized_eigenvalues: np.ndarray
    directness_score: float
    prediction_residual_ratio: float
    diagnostics: dict
    direct_snapshots: np.ndarray | None = None
    late_snapshots: np.ndarray | None = None
    decomposition_usable: bool = True
    decomposition_utility: float = 0.0


def multichannel_late_prediction_decomposition(
    snapshots: np.ndarray,
    *,
    delay_frames: int = 2,
    prediction_order: int = 3,
    ridge_relative: float = 1e-3,
    floor_relative: float = 1e-6,
) -> DirectPathDecomposition:
    """WPE-inspired late-reverberation prediction without iterative variance updates.

    ``snapshots`` has shape ``(sensors, time_frames)``. Delayed multichannel
    observations predict the current frame. The residual covariance is used as
    a direct/early-path estimate; the predicted covariance is the late field.
    This deterministic variant is intentionally auditable and numerically safe.
    """
    y = np.asarray(snapshots, dtype=complex)
    if y.ndim != 2 or y.shape[1] < 4:
        raise ValueError("snapshots must have shape (sensors, frames) with at least four frames")
    m, t = y.shape
    delay = max(int(delay_frames), 1)
    order = max(int(prediction_order), 1)
    first = delay + order - 1
    observed = _hermitian(y @ y.conj().T / t)
    if t - first < max(4, m):
        late = regularize_hpd(0.25 * observed, floor_relative)
        direct = regularize_hpd(observed - 0.20 * late, floor_relative)
        whitener = matrix_power_hpd(late, -0.5, floor_relative)
        whitened = regularize_hpd(whitener @ direct @ whitener.conj().T, floor_relative)
        gev = np.linalg.eigvalsh(whitened)[::-1]
        score = float(max(gev[0] - np.median(gev[1:]) if len(gev) > 1 else gev[0], 0.0) / max(gev[0], 1e-12))
        return DirectPathDecomposition(
            observed, direct, late, whitened, whitener, gev, score, 1.0,
            {"fallback": True, "decomposition_usable": False, "decomposition_utility": -1.0},
            direct_snapshots=y.copy(), late_snapshots=np.zeros_like(y),
            decomposition_usable=False, decomposition_utility=-1.0,
        )

    targets = y[:, first:]
    regressors = []
    for lag in range(order):
        start = first - delay - lag
        regressors.append(y[:, start:start + targets.shape[1]])
    x = np.vstack(regressors)
    gram = _hermitian(x @ x.conj().T / x.shape[1])
    ridge = float(ridge_relative) * max(float(np.trace(gram).real / max(len(gram), 1)), 1e-12)
    cross = targets @ x.conj().T / x.shape[1]
    predictor = cross @ np.linalg.pinv(gram + ridge * np.eye(len(gram)), hermitian=True)
    predicted = predictor @ x
    residual = targets - predicted
    late = regularize_hpd(predicted @ predicted.conj().T / predicted.shape[1], floor_relative)
    direct = regularize_hpd(residual @ residual.conj().T / residual.shape[1], floor_relative)
    target_cov = regularize_hpd(targets @ targets.conj().T / targets.shape[1], floor_relative)
    # Keep the observed/direct scale comparable while suppressing prediction leakage.
    direct_scale = float(np.trace(target_cov).real / max(np.trace(direct).real, 1e-12))
    direct = regularize_hpd(direct * min(max(direct_scale, 0.25), 4.0), floor_relative)
    late = regularize_hpd(late, floor_relative)
    whitener = matrix_power_hpd(late, -0.5, floor_relative)
    whitened = regularize_hpd(whitener @ direct @ whitener.conj().T, floor_relative)
    gev = np.linalg.eigvalsh(whitened)[::-1]
    background = float(np.median(gev[1:])) if len(gev) > 1 else 1.0
    directness = float(np.clip((gev[0] - background) / max(gev[0], 1e-12), 0.0, 1.0))
    residual_ratio = float(np.trace(direct).real / max(np.trace(target_cov).real, 1e-12))
    observed_eig = np.sort(np.linalg.eigvalsh(target_cov).real)[::-1]
    direct_eig = np.sort(np.linalg.eigvalsh(direct).real)[::-1]
    observed_gap = float(max(observed_eig[0] - observed_eig[1], 0.0) / max(observed_eig[0], 1e-12)) if len(observed_eig) > 1 else 1.0
    direct_gap = float(max(direct_eig[0] - direct_eig[1], 0.0) / max(direct_eig[0], 1e-12)) if len(direct_eig) > 1 else 1.0
    explained_fraction = float(np.trace(late).real / max(np.trace(late).real + np.trace(direct).real, 1e-12))
    predictor_condition = float(np.linalg.cond(gram + ridge * np.eye(len(gram))))
    utility = float((direct_gap - observed_gap) + 0.35 * explained_fraction + 0.20 * directness - 0.05 * abs(residual_ratio - 0.75))
    usable = bool(
        np.isfinite(utility)
        and predictor_condition < 1e10
        and explained_fraction >= 0.02
        and residual_ratio < 1.10
    )
    return DirectPathDecomposition(
        observed_covariance=observed,
        direct_covariance=direct,
        late_covariance=late,
        whitened_covariance=whitened,
        whitener=whitener,
        generalized_eigenvalues=gev,
        directness_score=directness,
        prediction_residual_ratio=residual_ratio,
        diagnostics={
            "fallback": False,
            "delay_frames": delay,
            "prediction_order": order,
            "ridge": ridge,
            "target_frames": int(targets.shape[1]),
            "predictor_condition": predictor_condition,
            "observed_eigen_gap": observed_gap,
            "direct_eigen_gap": direct_gap,
            "directness_gain": direct_gap - observed_gap,
            "prediction_explained_fraction": explained_fraction,
            "decomposition_utility": utility,
            "decomposition_usable": usable,
        },
        direct_snapshots=residual,
        late_snapshots=predicted,
        decomposition_usable=usable,
        decomposition_utility=utility,
    )


def adaptive_multichannel_late_prediction_decomposition(
    snapshots: np.ndarray,
    *,
    candidate_parameters: Sequence[tuple[int, int, float]] = (
        (2, 3, 1e-3),
        (1, 2, 1e-3),
        (3, 3, 1e-2),
        (2, 5, 1e-2),
    ),
    floor_relative: float = 1e-6,
    minimum_utility: float = 0.01,
    minimum_explained_fraction: float = 0.03,
    maximum_predictor_condition: float = 1e9,
) -> DirectPathDecomposition:
    """Select an auditable WPE-style predictor from a small deterministic bank.

    The adaptive selector is intended for nonstationary real recordings.  It
    never accepts a predictor merely because it lowers residual energy; the
    direct covariance must improve spatial structure while the predictor stays
    numerically stable.
    """
    candidates: list[DirectPathDecomposition] = []
    for delay, order, ridge in candidate_parameters:
        try:
            candidates.append(multichannel_late_prediction_decomposition(
                snapshots, delay_frames=int(delay), prediction_order=int(order),
                ridge_relative=float(ridge), floor_relative=floor_relative,
            ))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        return multichannel_late_prediction_decomposition(
            snapshots, delay_frames=2, prediction_order=3, ridge_relative=1e-3,
            floor_relative=floor_relative,
        )
    selected = max(candidates, key=lambda item: float(item.decomposition_utility))
    diag = dict(selected.diagnostics)
    explained = float(diag.get("prediction_explained_fraction", 0.0))
    condition = float(diag.get("predictor_condition", np.inf))
    usable = bool(
        selected.decomposition_utility >= float(minimum_utility)
        and explained >= float(minimum_explained_fraction)
        and condition <= float(maximum_predictor_condition)
        and selected.prediction_residual_ratio < 1.05
    )
    return replace(
        selected,
        decomposition_usable=usable,
        diagnostics={
            **diag,
            "adaptive_prediction": True,
            "adaptive_candidate_count": len(candidates),
            "decomposition_usable": usable,
            "minimum_utility": float(minimum_utility),
            "minimum_explained_fraction": float(minimum_explained_fraction),
            "maximum_predictor_condition": float(maximum_predictor_condition),
        },
    )


def direct_path_evidence(evidences: Sequence[CovarianceEvidence]) -> list[CovarianceEvidence]:
    output: list[CovarianceEvidence] = []
    for item in evidences:
        meta = dict(item.metadata)
        direct = meta.get("direct_covariance")
        if direct is None:
            output.append(item)
            continue
        directness = float(meta.get("directness_score", 0.0))
        usable = bool(meta.get("decomposition_usable", True))
        if usable:
            chosen_covariance = np.asarray(direct, dtype=complex)
            new_weight = float(item.weight) * (0.25 + 0.75 * directness)
            mode = "direct_covariance"
        else:
            chosen_covariance = np.asarray(item.covariance, dtype=complex)
            new_weight = float(item.weight)
            mode = "observed_covariance_fallback"
        output.append(replace(
            item,
            covariance=chosen_covariance,
            weight=new_weight,
            eigen_gap=max(float(item.eigen_gap), directness) if usable else float(item.eigen_gap),
            diffuseness=min(float(item.diffuseness), 1.0 - directness) if usable else float(item.diffuseness),
            metadata={
                **meta,
                "base_observed_covariance": item.covariance,
                "direct_path_weight": new_weight,
                "direct_path_evidence_mode": mode,
            },
        ))
    return output


def log_euclidean_geometric_median(
    covariances: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
    floor_relative: float = 1e-6,
) -> np.ndarray:
    covs = list(covariances)
    if not covs:
        raise ValueError("At least one covariance is required")
    logs = [matrix_log_hpd(c, floor_relative) for c in covs]
    if weights is None:
        base = np.ones(len(logs), dtype=float)
    else:
        base = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if not np.any(base > 0):
        base = np.ones(len(logs), dtype=float)
    base /= np.sum(base)
    center = sum(w * x for w, x in zip(base, logs))
    for _ in range(int(max_iterations)):
        distances = np.asarray([np.linalg.norm(x - center, "fro") for x in logs], dtype=float)
        if np.min(distances) < tolerance:
            center_new = logs[int(np.argmin(distances))]
        else:
            w = base / np.maximum(distances, tolerance)
            w /= np.sum(w)
            center_new = sum(ww * x for ww, x in zip(w, logs))
        if np.linalg.norm(center_new - center, "fro") <= tolerance * max(np.linalg.norm(center, "fro"), 1.0):
            center = center_new
            break
        center = center_new
    return regularize_hpd(matrix_exp_hermitian(center), floor_relative)


def jbld_divergence(a: np.ndarray, b: np.ndarray, floor_relative: float = 1e-6) -> float:
    aa = regularize_hpd(a, floor_relative)
    bb = regularize_hpd(b, floor_relative)
    mid = regularize_hpd(0.5 * (aa + bb), floor_relative)
    _, ld_mid = np.linalg.slogdet(mid)
    _, ld_a = np.linalg.slogdet(aa)
    _, ld_b = np.linalg.slogdet(bb)
    return float(max(ld_mid - 0.5 * (ld_a + ld_b), 0.0))


def fit_nonnegative_covariance_model(
    covariance: np.ndarray,
    steering_matrix: np.ndarray,
    *,
    floor_relative: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    r = regularize_hpd(covariance, floor_relative)
    a = np.asarray(steering_matrix, dtype=complex)
    m, k = a.shape
    atoms = [np.outer(a[:, i], a[:, i].conj()) for i in range(k)] + [np.eye(m)]
    # Real-valued least squares on Hermitian entries.
    y = np.concatenate([r.real.reshape(-1), r.imag.reshape(-1)])
    x = np.column_stack([
        np.concatenate([atom.real.reshape(-1), atom.imag.reshape(-1)]) for atom in atoms
    ])
    coefficients, _ = nnls(x, y)
    model = sum(c * atom for c, atom in zip(coefficients, atoms))
    model = regularize_hpd(model, floor_relative)
    return coefficients[:-1], model, float(coefficients[-1])


def matrix_model_match_score(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    theta_deg: Sequence[float],
    steering_builder,
    *,
    robust: str = "median",
    floor_relative: float = 1e-6,
) -> dict:
    theta = np.sort(np.asarray(theta_deg, dtype=float).reshape(-1))
    scores, weights, powers = [], [], []
    for item in evidences:
        ids = np.arange(len(microphone_positions_m)) if item.sensor_indices is None else np.asarray(item.sensor_indices, dtype=int)
        positions = np.asarray(microphone_positions_m)[ids]
        a = steering_builder(positions, theta, float(item.frequency_hz))
        p, model, noise = fit_nonnegative_covariance_model(item.covariance, a, floor_relative=floor_relative)
        scores.append(jbld_divergence(item.covariance, model, floor_relative=floor_relative))
        weights.append(max(float(item.weight), 0.0))
        powers.append(np.r_[p, noise])
    score_arr = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not np.any(w > 0):
        w = np.ones_like(w)
    w /= np.sum(w)
    if robust == "median":
        order = np.argsort(score_arr)
        cdf = np.cumsum(w[order])
        score = float(score_arr[order[int(np.searchsorted(cdf, 0.5))]])
    else:
        center = float(np.dot(w, score_arr))
        scale = max(float(np.median(np.abs(score_arr - np.median(score_arr)))) * 1.4826, 1e-8)
        huber = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(score_arr - center), 1e-12))
        ww = w * huber
        ww /= np.sum(ww)
        score = float(np.dot(ww, score_arr))
    return {
        "score": score,
        "per_frequency_jbld": score_arr,
        "weights": w,
        "source_noise_powers": np.asarray(powers),
        "frequency_consensus_std": float(np.sqrt(np.dot(w, (score_arr - np.dot(w, score_arr)) ** 2))),
    }


def circular_difference_deg(estimate_deg: np.ndarray | float, truth_deg: np.ndarray | float) -> np.ndarray:
    e = np.asarray(estimate_deg, dtype=float)
    t = np.asarray(truth_deg, dtype=float)
    return (e - t + 180.0) % 360.0 - 180.0


@dataclass
class CircularMatrixTracker:
    process_variance_deg2: float = 4.0
    measurement_variance_deg2: float = 25.0
    innovation_gate_deg: float = 45.0
    angle_deg: float | None = None
    velocity_deg_s: float = 0.0
    covariance: np.ndarray | None = None
    last_time_s: float | None = None

    def update(self, measurement_deg: float, time_s: float, confidence: float = 1.0) -> tuple[float, dict]:
        z = float(measurement_deg)
        t = float(time_s)
        if self.angle_deg is None or self.last_time_s is None:
            self.angle_deg = z
            self.last_time_s = t
            self.covariance = np.diag([self.measurement_variance_deg2, 25.0])
            return z, {"initialized": True, "accepted": True, "innovation_deg": 0.0}
        dt = max(t - self.last_time_s, 1e-3)
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.process_variance_deg2 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
        state = np.array([self.angle_deg, self.velocity_deg_s])
        pred = f @ state
        p = f @ np.asarray(self.covariance) @ f.T + q
        innovation = float(circular_difference_deg(z, pred[0]))
        accepted = abs(innovation) <= self.innovation_gate_deg or confidence >= 0.85
        if accepted:
            r = self.measurement_variance_deg2 / max(float(confidence), 0.05)
            s = p[0, 0] + r
            k = p[:, 0] / s
            state_new = pred + k * innovation
            p_new = (np.eye(2) - np.outer(k, [1.0, 0.0])) @ p
        else:
            state_new, p_new = pred, p
        state_new[0] = (state_new[0] + 180.0) % 360.0 - 180.0
        self.angle_deg = float(state_new[0])
        self.velocity_deg_s = float(state_new[1])
        self.covariance = p_new
        self.last_time_s = t
        return self.angle_deg, {
            "initialized": False,
            "accepted": bool(accepted),
            "innovation_deg": innovation,
            "confidence": float(confidence),
            "posterior_variance_deg2": float(p_new[0, 0]),
        }


def circular_viterbi_smooth(
    candidate_angles_deg: Sequence[np.ndarray],
    emission_costs: Sequence[np.ndarray],
    timestamps_s: Sequence[float],
    *,
    transition_weight: float = 0.015,
    max_speed_deg_s: float = 180.0,
) -> tuple[np.ndarray, dict]:
    """Offline circular Viterbi path on a nonuniform required-time grid.

    Consecutive duplicate request times are collapsed before dynamic
    programming and expanded afterwards.  This prevents repeated rows from
    creating artificial transitions or multiplying the same observation cost.
    Positive transition intervals use their actual ``delta_t`` and are audited.
    """
    if not candidate_angles_deg or len(candidate_angles_deg) != len(emission_costs):
        raise ValueError("candidate and emission sequences must be non-empty and equal length")
    times = np.asarray(timestamps_s, dtype=float)
    if len(times) != len(candidate_angles_deg):
        raise ValueError("one timestamp is required per frame")
    if np.any(np.diff(times) < -1e-12):
        raise ValueError("Viterbi timestamps must be nondecreasing")
    angles_all = [np.asarray(x, dtype=float).reshape(-1) for x in candidate_angles_deg]
    costs_all = [np.asarray(x, dtype=float).reshape(-1) for x in emission_costs]
    if any(len(a) == 0 or len(a) != len(c) for a, c in zip(angles_all, costs_all)):
        raise ValueError("every frame requires matching non-empty candidates and costs")

    unique_indices: list[int] = []
    original_to_unique: list[int] = []
    for index, timestamp in enumerate(times):
        if not unique_indices or abs(float(timestamp - times[unique_indices[-1]])) > 1e-12:
            unique_indices.append(index)
        original_to_unique.append(len(unique_indices) - 1)
    unique_times = times[np.asarray(unique_indices, dtype=int)]
    angles = [angles_all[index] for index in unique_indices]
    costs = [costs_all[index] for index in unique_indices]

    dp = costs[0] - np.min(costs[0])
    backpointers: list[np.ndarray] = []
    transition_dt_s: list[float] = []
    for t in range(1, len(angles)):
        dt = float(unique_times[t] - unique_times[t - 1])
        if dt <= 0:
            raise ValueError("collapsed Viterbi timestamps must be strictly increasing")
        transition_dt_s.append(dt)
        delta = circular_difference_deg(angles[t][:, None], angles[t - 1][None, :])
        speed = np.abs(delta) / dt
        # Preserve the validated v4.2 cost while making the real time interval
        # explicit.  A later preregistered release may recalibrate this cost;
        # v5.0a2 only removes duplicate-time artefacts and adds diagnostics.
        transition = float(transition_weight) * (delta * delta) / dt
        transition = np.where(speed <= float(max_speed_deg_s), transition, transition + 1e6)
        total = transition + dp[None, :]
        parents = np.argmin(total, axis=1)
        dp = costs[t] - np.min(costs[t]) + total[np.arange(len(angles[t])), parents]
        backpointers.append(parents)
    state = int(np.argmin(dp))
    unique_path = [state]
    for parents in backpointers[::-1]:
        state = int(parents[state])
        unique_path.append(state)
    unique_path = unique_path[::-1]
    unique_estimates = np.asarray(
        [angles[t][unique_path[t]] for t in range(len(unique_path))], dtype=float
    )
    estimates = unique_estimates[np.asarray(original_to_unique, dtype=int)]
    dt_array = np.asarray(transition_dt_s, dtype=float)
    return estimates, {
        "path_indices_unique": unique_path,
        "original_to_unique_index": original_to_unique,
        "final_cost": float(np.min(dp)),
        "transition_weight": float(transition_weight),
        "max_speed_deg_s": float(max_speed_deg_s),
        "input_frame_count": int(len(times)),
        "unique_timestamp_count": int(len(unique_times)),
        "duplicate_timestamp_count": int(len(times) - len(unique_times)),
        "minimum_positive_dt_s": float(np.min(dt_array)) if len(dt_array) else float("nan"),
        "median_positive_dt_s": float(np.median(dt_array)) if len(dt_array) else float("nan"),
        "maximum_positive_dt_s": float(np.max(dt_array)) if len(dt_array) else float("nan"),
        "required_time_aware": True,
    }
