# =========================================================
# File        : certified_multihypothesis_hodge.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the certified multihypothesis hodge module used by the reproducibility workflow.
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
"""Grid-certified multi-hypothesis Hodge synchronization for TDOA localization.

V52J-R7 replaces the historical post-hoc safe gate by a single joint objective
that combines local GCC peak evidence, graph-cycle consistency, and far-field
array geometry.  For every frozen azimuth grid point, the one-hot peak
assignment problem is relaxed to a product of simplices.  A deterministic
Frank--Wolfe solver returns both a feasible relaxation point and a rigorous
Frank--Wolfe lower bound.  Rounded/coordinate-refined one-hot assignments give
feasible upper bounds.  Across the full grid this yields:

* a global discrete feasible upper bound;
* a rigorous lower bound for the grid-restricted discrete problem;
* an optimality gap;
* a near-optimal azimuth set whose diameter quantifies ambiguity.

The certificate concerns optimization/ambiguity on the frozen candidate bank
and azimuth grid.  It is not a proof that the correct acoustic path is present
in Top-K, nor a guarantee under propagation-model mismatch.  Those limits are
reported explicitly and are part of the method's stated applicability domain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import heapq

import numpy as np
from scipy.optimize import minimize_scalar

from .baseline_extension import _predicted_pair_delays
from .baseline_forensic import BaselineForensicParameters
from .hodge_tde import HodgeTDEParameters, extract_gcc_peak_bank
from .r5_anchor_estimates import E0, r5_anchor_estimates

_EPS = 1.0e-12

R7_RAW = "gcc_topk_joint_cycle_geometry_r7"
R7_CERTIFIED = "gcc_topk_joint_cycle_geometry_r7_certified_e0_fallback"


@dataclass(frozen=True)
class CertifiedMultiHypothesisParameters:
    """Frozen V52J-R7 foundation parameters.

    Delay residuals are normalized in samples, so cycle and geometry terms are
    dimensionless and portable across recordings with the same sample rate.
    The primary foundation setting deliberately uses equal cycle/geometry
    weights; no LOCATA error was used to tune these constants.
    """

    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 3500.0
    theta_min_deg: float = -80.0
    theta_max_deg: float = 80.0
    theta_step_deg: float = 2.0
    refine_radius_deg: float = 2.0
    pair_min_baseline_m: float = 0.05
    maximum_pairs: int = 0
    top_k: int = 3
    minimum_peak_separation_s: float = 0.00015
    peak_evidence_weight: float = 0.20
    cycle_weight: float = 1.0
    geometry_weight: float = 1.0
    quality_floor: float = 0.05
    frank_wolfe_max_iterations: int = 160
    frank_wolfe_gap_tolerance: float = 1.0e-7
    coordinate_descent_passes: int = 8
    initial_discrete_theta_count: int = 7
    branch_and_bound_theta_count: int = 5
    branch_and_bound_max_nodes: int = 128
    branch_and_bound_gap_tolerance: float = 1.0e-5
    # Certificate thresholds.  These are optimization-domain thresholds, not
    # fitted performance thresholds.
    certificate_relative_gap_max: float = 0.02
    certificate_absolute_gap_max: float = 0.05
    certificate_ambiguity_diameter_max_deg: float = 6.0
    certificate_integrality_min: float = 0.90
    near_optimal_objective_slack: float = 0.05

    def as_dict(self) -> dict:
        return asdict(self)

    def hodge_bank_parameters(self) -> HodgeTDEParameters:
        return HodgeTDEParameters(
            frequency_min_hz=self.frequency_min_hz,
            frequency_max_hz=self.frequency_max_hz,
            theta_min_deg=self.theta_min_deg,
            theta_max_deg=self.theta_max_deg,
            coarse_step_deg=max(self.theta_step_deg, 0.25),
            refine_radius_deg=self.refine_radius_deg,
            pair_min_baseline_m=self.pair_min_baseline_m,
            maximum_pairs=self.maximum_pairs,
            top_k=self.top_k,
            minimum_peak_separation_s=self.minimum_peak_separation_s,
            amplitude_penalty=self.peak_evidence_weight,
        )


@dataclass(frozen=True)
class CertifiedMultiHypothesisResult:
    estimates_deg: dict[str, float]
    diagnostics: dict


def _incidence_matrix(pairs: np.ndarray, microphone_count: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=int)
    B = np.zeros((len(pairs), int(microphone_count)), dtype=float)
    B[np.arange(len(pairs)), pairs[:, 0]] = 1.0
    B[np.arange(len(pairs)), pairs[:, 1]] = -1.0
    return B


def _weighted_cycle_matrix(
    pairs: np.ndarray,
    microphone_count: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return PSD matrix M with d.T M d = min_t ||W^(1/2)(d-Bt)||²."""
    B = _incidence_matrix(pairs, microphone_count)
    w = np.maximum(np.asarray(weights, dtype=float).reshape(-1), _EPS)
    if len(w) != len(B):
        raise ValueError("weights and pair bank have incompatible sizes")
    # Remove one gauge column.  pinv also handles a disconnected/admissible
    # pair graph; graph rank is reported and can be used as an applicability
    # diagnostic.
    A = B[:, 1:]
    W = np.diag(w)
    normal = A.T @ W @ A
    normal_pinv = np.linalg.pinv(normal, rcond=1.0e-12)
    M = W - W @ A @ normal_pinv @ A.T @ W
    M = 0.5 * (M + M.T)
    # Remove only tiny negative eigenvalues introduced by roundoff.
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    clipped = np.maximum(eigenvalues, 0.0)
    M = (eigenvectors * clipped) @ eigenvectors.T
    graph_rank = int(np.linalg.matrix_rank(B, tol=1.0e-10))
    component_count = int(microphone_count - graph_rank)
    cycle_rank = int(len(pairs) - graph_rank)
    return M, {
        "edge_count": int(len(pairs)),
        "microphone_count": int(microphone_count),
        "graph_rank": graph_rank,
        "component_count": component_count,
        "cycle_rank": cycle_rank,
        "minimum_cycle_matrix_eigenvalue": float(np.min(clipped)) if len(clipped) else 0.0,
        "maximum_cycle_matrix_eigenvalue": float(np.max(clipped)) if len(clipped) else 0.0,
    }


def _validate_candidates(
    candidate_delays_s: np.ndarray,
    candidate_amplitudes: np.ndarray,
    candidate_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delays = np.asarray(candidate_delays_s, dtype=float)
    amplitudes = np.asarray(candidate_amplitudes, dtype=float)
    valid = np.asarray(candidate_valid, dtype=bool)
    if delays.ndim != 2 or delays.shape != amplitudes.shape or delays.shape != valid.shape:
        raise ValueError("candidate delays, amplitudes, and validity must share E x K shape")
    if not np.all(np.any(valid, axis=1)):
        raise ValueError("every edge must have at least one valid candidate")
    if np.any(~np.isfinite(delays[valid])):
        raise ValueError("valid candidate delays must be finite")
    if np.any(amplitudes[valid] <= 0.0):
        raise ValueError("valid candidate amplitudes must be positive")
    return delays, amplitudes, valid


def _one_hot_from_indices(indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=int).reshape(-1)
    x = np.zeros(valid.shape, dtype=float)
    x[np.arange(len(indices)), indices] = 1.0
    if not np.all(valid[np.arange(len(indices)), indices]):
        raise ValueError("one-hot assignment selected an invalid candidate")
    return x


def _selected_delays(x: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    safe_candidates = np.where(np.isfinite(candidates), candidates, 0.0)
    return np.sum(np.asarray(x, dtype=float) * safe_candidates, axis=1)


def _objective_and_gradient(
    x: np.ndarray,
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    constant: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    d = _selected_delays(x, candidates)
    qd_minus_b = Q @ d - b
    objective = float(
        np.sum(x * peak_cost)
        + d @ Q @ d
        - 2.0 * b @ d
        + constant
    )
    safe_candidates = np.where(np.isfinite(candidates), candidates, 0.0)
    gradient = peak_cost + 2.0 * safe_candidates * qd_minus_b[:, None]
    return objective, gradient, d


def _frank_wolfe_relaxation(
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    valid: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    constant: float,
    *,
    max_iterations: int,
    gap_tolerance: float,
    initial_x: np.ndarray | None = None,
) -> dict:
    """Solve the convex product-simplex relaxation with a dual-gap bound."""
    E, _ = candidates.shape
    if initial_x is None:
        initial_indices = np.argmin(np.where(valid, peak_cost, np.inf), axis=1)
        x = _one_hot_from_indices(initial_indices, valid)
    else:
        x = np.asarray(initial_x, dtype=float).copy()
        if x.shape != candidates.shape:
            raise ValueError("initial_x has incompatible shape")
        x[~valid] = 0.0
        row_sums = np.sum(x, axis=1)
        if np.any(row_sums <= _EPS):
            raise ValueError("initial_x must put mass on a valid candidate for every edge")
        x /= row_sums[:, None]

    history: list[dict] = []
    lower_bound = -np.inf
    gap = np.inf
    for iteration in range(int(max_iterations)):
        objective, gradient, d = _objective_and_gradient(
            x, candidates, peak_cost, Q, b, constant
        )
        masked_gradient = np.where(valid, gradient, np.inf)
        s_indices = np.argmin(masked_gradient, axis=1)
        s = _one_hot_from_indices(s_indices, valid)
        gap = float(np.sum((x - s) * gradient))
        gap = max(gap, 0.0)
        lower_bound = float(objective - gap)
        if iteration == 0 or (iteration + 1) % 20 == 0 or gap <= gap_tolerance:
            history.append({
                "iteration": int(iteration + 1),
                "objective": float(objective),
                "frank_wolfe_gap": float(gap),
                "lower_bound": float(lower_bound),
            })
        if gap <= float(gap_tolerance):
            break
        direction_x = s - x
        direction_d = _selected_delays(direction_x, candidates)
        curvature = float(direction_d @ Q @ direction_d)
        if curvature <= _EPS:
            step = 1.0
        else:
            step = float(np.clip(gap / (2.0 * curvature), 0.0, 1.0))
        x += step * direction_x
        # Numerical cleanup preserves the product-simplex constraints.
        x[~valid] = 0.0
        x = np.maximum(x, 0.0)
        x /= np.sum(x, axis=1, keepdims=True)

    objective, gradient, d = _objective_and_gradient(
        x, candidates, peak_cost, Q, b, constant
    )
    masked_gradient = np.where(valid, gradient, np.inf)
    s_indices = np.argmin(masked_gradient, axis=1)
    s = _one_hot_from_indices(s_indices, valid)
    gap = max(float(np.sum((x - s) * gradient)), 0.0)
    lower_bound = float(objective - gap)
    return {
        "x": x,
        "relaxation_delays": d,
        "objective": float(objective),
        "lower_bound": lower_bound,
        "frank_wolfe_gap": float(gap),
        "iterations": int(iteration + 1),
        "history": history,
        "integrality_mean_max_mass": float(np.mean(np.max(x, axis=1))),
        "fractional_edge_fraction": float(np.mean(np.max(x, axis=1) < 1.0 - 1.0e-6)),
    }


def _discrete_objective_from_indices(
    indices: np.ndarray,
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    constant: float,
) -> float:
    rows = np.arange(len(indices))
    d = candidates[rows, indices]
    return float(
        np.sum(peak_cost[rows, indices])
        + d @ Q @ d
        - 2.0 * b @ d
        + constant
    )


def _coordinate_refine_assignment(
    initial_indices: np.ndarray,
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    valid: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    constant: float,
    *,
    passes: int,
) -> dict:
    """Deterministic one-edge coordinate descent for a feasible upper bound."""
    indices = np.asarray(initial_indices, dtype=int).copy()
    rows = np.arange(len(indices))
    if not np.all(valid[rows, indices]):
        raise ValueError("initial discrete assignment contains invalid candidates")
    d = candidates[rows, indices].copy()
    Qd = Q @ d
    changes = 0
    for _ in range(int(passes)):
        changed_this_pass = 0
        for edge in range(len(indices)):
            old = int(indices[edge])
            old_delay = float(d[edge])
            old_peak = float(peak_cost[edge, old])
            best = old
            best_delta = 0.0
            h_edge = float(Qd[edge] - b[edge])
            for candidate_index in np.flatnonzero(valid[edge]):
                candidate_index = int(candidate_index)
                if candidate_index == old:
                    continue
                delta_d = float(candidates[edge, candidate_index] - old_delay)
                delta = (
                    float(peak_cost[edge, candidate_index]) - old_peak
                    + 2.0 * delta_d * h_edge
                    + delta_d * delta_d * float(Q[edge, edge])
                )
                if delta < best_delta - 1.0e-12:
                    best_delta = float(delta)
                    best = candidate_index
            if best != old:
                new_delay = float(candidates[edge, best])
                delta_d = new_delay - old_delay
                indices[edge] = best
                d[edge] = new_delay
                Qd += Q[:, edge] * delta_d
                changes += 1
                changed_this_pass += 1
        if changed_this_pass == 0:
            break
    objective = _discrete_objective_from_indices(
        indices, candidates, peak_cost, Q, b, constant
    )
    return {
        "indices": indices,
        "selected_delays": d,
        "objective": float(objective),
        "coordinate_changes": int(changes),
    }



def _restricted_initial_x(x: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    y[~allowed] = 0.0
    row_sums = np.sum(y, axis=1)
    for edge in np.flatnonzero(row_sums <= _EPS):
        choices = np.flatnonzero(allowed[edge])
        if len(choices) == 0:
            raise ValueError("branch node removed every candidate from an edge")
        y[edge, int(choices[0])] = 1.0
    y /= np.sum(y, axis=1, keepdims=True)
    return y


def _branch_and_bound_discrete(
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    valid: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    constant: float,
    *,
    max_nodes: int,
    gap_tolerance: float,
    fw_max_iterations: int,
    fw_gap_tolerance: float,
    coordinate_passes: int,
    root_relaxation: dict | None = None,
    initial_incumbent: dict | None = None,
) -> dict:
    """Branch on fractional edges using convex-relaxation node bounds.

    The returned lower bound is the minimum bound of all open nodes (or the
    incumbent when the tree is exhausted).  Therefore it remains valid even
    when the node budget is reached.
    """
    if root_relaxation is None:
        root_relaxation = _frank_wolfe_relaxation(
            candidates, peak_cost, valid, Q, b, constant,
            max_iterations=fw_max_iterations,
            gap_tolerance=fw_gap_tolerance,
        )
    if initial_incumbent is None:
        initial_indices = np.argmax(root_relaxation["x"], axis=1).astype(int)
        initial_incumbent = _coordinate_refine_assignment(
            initial_indices, candidates, peak_cost, valid, Q, b, constant,
            passes=coordinate_passes,
        )
    incumbent = dict(initial_incumbent)
    upper = float(incumbent["objective"])
    counter = 0
    heap: list[tuple[float, int, np.ndarray, dict]] = []
    heapq.heappush(
        heap,
        (float(root_relaxation["lower_bound"]), counter, valid.copy(), root_relaxation),
    )
    nodes_solved = 0
    nodes_pruned = 0
    integral_nodes = 0
    while heap and nodes_solved < int(max_nodes):
        lower, _, allowed, relaxation = heapq.heappop(heap)
        if float(lower) >= upper - float(gap_tolerance):
            nodes_pruned += 1
            continue
        nodes_solved += 1
        x = relaxation["x"]
        max_mass = np.max(x, axis=1)
        branchable = np.flatnonzero((np.sum(allowed, axis=1) > 1) & (max_mass < 1.0 - 1.0e-8))
        if len(branchable) == 0:
            integral_nodes += 1
            indices = np.argmax(x, axis=1).astype(int)
            discrete = _coordinate_refine_assignment(
                indices, candidates, peak_cost, allowed, Q, b, constant,
                passes=coordinate_passes,
            )
            if discrete["objective"] < upper:
                incumbent = discrete
                upper = float(discrete["objective"])
            continue
        # Most fractional edge first; ties are deterministic by edge index.
        edge = int(branchable[np.argmin(max_mass[branchable])])
        choices = [int(k) for k in np.flatnonzero(allowed[edge])]
        choices.sort(key=lambda k: (-float(x[edge, k]), k))
        for candidate_index in choices:
            child_allowed = allowed.copy()
            child_allowed[edge, :] = False
            child_allowed[edge, candidate_index] = True
            child_initial = _restricted_initial_x(x, child_allowed)
            child_relaxation = _frank_wolfe_relaxation(
                candidates, peak_cost, child_allowed, Q, b, constant,
                max_iterations=fw_max_iterations,
                gap_tolerance=fw_gap_tolerance,
                initial_x=child_initial,
            )
            child_lower = float(child_relaxation["lower_bound"])
            rounded_indices = np.argmax(child_relaxation["x"], axis=1).astype(int)
            child_discrete = _coordinate_refine_assignment(
                rounded_indices, candidates, peak_cost, child_allowed, Q, b, constant,
                passes=coordinate_passes,
            )
            if child_discrete["objective"] < upper:
                incumbent = child_discrete
                upper = float(child_discrete["objective"])
            if child_lower < upper - float(gap_tolerance):
                counter += 1
                heapq.heappush(
                    heap,
                    (child_lower, counter, child_allowed, child_relaxation),
                )
            else:
                nodes_pruned += 1
        if heap and upper - float(heap[0][0]) <= float(gap_tolerance):
            break
    lower_bound = float(heap[0][0]) if heap else upper
    absolute_gap = max(upper - lower_bound, 0.0)
    return {
        "indices": incumbent["indices"],
        "selected_delays": incumbent["selected_delays"],
        "objective": upper,
        "lower_bound": lower_bound,
        "absolute_gap": float(absolute_gap),
        "exact_within_tolerance": bool(absolute_gap <= float(gap_tolerance)),
        "node_budget_exhausted": bool(heap and nodes_solved >= int(max_nodes)),
        "nodes_solved": int(nodes_solved),
        "nodes_pruned": int(nodes_pruned),
        "integral_nodes": int(integral_nodes),
        "open_nodes": int(len(heap)),
    }


def _unary_lower_bound(
    candidates: np.ndarray,
    peak_cost: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    geometry_delays: np.ndarray,
    geometry_weight: float,
) -> float:
    unary = peak_cost + float(geometry_weight) * weights[:, None] * (
        candidates - geometry_delays[:, None]
    ) ** 2
    return float(np.sum(np.min(np.where(valid, unary, np.inf), axis=1)))


def _theta_diameter(theta_values: Iterable[float]) -> float:
    values = np.asarray(list(theta_values), dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.max(values) - np.min(values))


def solve_certified_candidate_bank(
    candidate_delays_s: np.ndarray,
    candidate_amplitudes: np.ndarray,
    candidate_valid: np.ndarray,
    pairs: np.ndarray,
    microphone_positions_m: np.ndarray,
    sample_rate_hz: float,
    base_quality: np.ndarray,
    *,
    e0_estimate_deg: float,
    params: CertifiedMultiHypothesisParameters = CertifiedMultiHypothesisParameters(),
) -> dict:
    """Solve and certify the frozen-grid multi-hypothesis assignment problem."""
    delays_s, amplitudes, valid = _validate_candidates(
        candidate_delays_s, candidate_amplitudes, candidate_valid
    )
    pairs = np.asarray(pairs, dtype=int)
    positions = np.asarray(microphone_positions_m, dtype=float)
    quality = np.clip(np.asarray(base_quality, dtype=float).reshape(-1), params.quality_floor, 1.0)
    if len(pairs) != len(delays_s) or len(quality) != len(delays_s):
        raise ValueError("candidate bank, pair bank, and quality have incompatible sizes")

    # Delay normalization in samples.
    candidates = delays_s * float(sample_rate_hz)
    # Invalid slots carry zero in the algebra and are excluded by the validity mask.
    # This avoids the undefined product 0 * inf in simplex objectives.
    peak_cost = np.zeros(candidates.shape, dtype=float)
    peak_cost[valid] = float(params.peak_evidence_weight) * (
        -np.log(np.maximum(amplitudes[valid], _EPS))
    )
    cycle_matrix, graph_diag = _weighted_cycle_matrix(
        pairs, len(positions), quality
    )
    W = np.diag(quality)
    Q = float(params.cycle_weight) * cycle_matrix + float(params.geometry_weight) * W

    theta_grid = np.arange(
        float(params.theta_min_deg),
        float(params.theta_max_deg) + 0.5 * float(params.theta_step_deg),
        float(params.theta_step_deg),
    )
    geometry_bank = np.asarray([
        _predicted_pair_delays(positions, pairs, float(theta)) * float(sample_rate_hz)
        for theta in theta_grid
    ])
    unary_bounds = np.asarray([
        _unary_lower_bound(
            candidates, peak_cost, valid, quality, geometry,
            params.geometry_weight,
        )
        for geometry in geometry_bank
    ])

    # Establish a deterministic feasible upper bound before relaxation pruning.
    nearest_e0 = int(np.argmin(np.abs(theta_grid - float(e0_estimate_deg))))
    seed_order = list(np.argsort(unary_bounds, kind="mergesort")[: int(params.initial_discrete_theta_count)])
    if nearest_e0 not in seed_order:
        seed_order.append(nearest_e0)
    feasible_by_theta: dict[int, dict] = {}
    best_upper = np.inf
    best_theta_index = -1
    for theta_index in seed_order:
        geometry = geometry_bank[theta_index]
        b = float(params.geometry_weight) * quality * geometry
        constant = float(params.geometry_weight) * float(np.sum(quality * geometry * geometry))
        unary = peak_cost + float(params.geometry_weight) * quality[:, None] * (
            candidates - geometry[:, None]
        ) ** 2
        initial_indices = np.argmin(np.where(valid, unary, np.inf), axis=1)
        discrete = _coordinate_refine_assignment(
            initial_indices, candidates, peak_cost, valid, Q, b, constant,
            passes=params.coordinate_descent_passes,
        )
        feasible_by_theta[int(theta_index)] = discrete
        if discrete["objective"] < best_upper:
            best_upper = float(discrete["objective"])
            best_theta_index = int(theta_index)

    theta_rows: list[dict] = []
    relaxation_by_theta: dict[int, dict] = {}
    warm_x: np.ndarray | None = None
    for theta_index, (theta, geometry, unary_lb) in enumerate(
        zip(theta_grid, geometry_bank, unary_bounds)
    ):
        # Dropping the nonnegative cycle term is a rigorous lower bound.  If it
        # already exceeds the current feasible upper bound plus the ambiguity
        # slack, no full relaxation solve is required at this grid point.
        if float(unary_lb) > float(best_upper + params.near_optimal_objective_slack):
            theta_rows.append({
                "theta_deg": float(theta),
                "lower_bound": float(unary_lb),
                "lower_bound_kind": "unary_cycle_dropped",
                "relaxation_objective": float("nan"),
                "frank_wolfe_gap": float("nan"),
                "relaxation_integrality": float("nan"),
                "discrete_upper_bound": float(feasible_by_theta.get(theta_index, {}).get("objective", np.nan)),
                "relaxation_solved": False,
            })
            continue
        b = float(params.geometry_weight) * quality * geometry
        constant = float(params.geometry_weight) * float(np.sum(quality * geometry * geometry))
        relaxation = _frank_wolfe_relaxation(
            candidates, peak_cost, valid, Q, b, constant,
            max_iterations=params.frank_wolfe_max_iterations,
            gap_tolerance=params.frank_wolfe_gap_tolerance,
            initial_x=warm_x,
        )
        warm_x = relaxation["x"]
        relaxation_by_theta[int(theta_index)] = relaxation
        rounded_indices = np.argmax(relaxation["x"], axis=1).astype(int)
        rounded = _coordinate_refine_assignment(
            rounded_indices, candidates, peak_cost, valid, Q, b, constant,
            passes=params.coordinate_descent_passes,
        )
        existing = feasible_by_theta.get(theta_index)
        if existing is None or rounded["objective"] < existing["objective"]:
            feasible_by_theta[int(theta_index)] = rounded
        discrete = feasible_by_theta[int(theta_index)]
        if discrete["objective"] < best_upper:
            best_upper = float(discrete["objective"])
            best_theta_index = int(theta_index)
        theta_rows.append({
            "theta_deg": float(theta),
            "lower_bound": float(relaxation["lower_bound"]),
            "lower_bound_kind": "frank_wolfe_dual_gap",
            "relaxation_objective": float(relaxation["objective"]),
            "frank_wolfe_gap": float(relaxation["frank_wolfe_gap"]),
            "relaxation_integrality": float(relaxation["integrality_mean_max_mass"]),
            "discrete_upper_bound": float(discrete["objective"]),
            "relaxation_solved": True,
        })

    # Tighten the most competitive grid points with a bounded branch-and-bound
    # search.  Root Frank--Wolfe bounds remain valid if the node budget is hit;
    # the minimum open-node bound then certifies the residual uncertainty.
    competitive = [
        i for i, row in enumerate(theta_rows)
        if bool(row["relaxation_solved"])
        and float(row["lower_bound"]) <= float(best_upper + params.near_optimal_objective_slack)
    ]
    competitive.sort(key=lambda i: (float(theta_rows[i]["lower_bound"]), i))
    branch_rows: list[dict] = []
    branch_by_theta: dict[int, dict] = {}
    for theta_index in competitive[: int(params.branch_and_bound_theta_count)]:
        geometry = geometry_bank[theta_index]
        b = float(params.geometry_weight) * quality * geometry
        constant = float(params.geometry_weight) * float(np.sum(quality * geometry * geometry))
        branch = _branch_and_bound_discrete(
            candidates, peak_cost, valid, Q, b, constant,
            max_nodes=params.branch_and_bound_max_nodes,
            gap_tolerance=params.branch_and_bound_gap_tolerance,
            fw_max_iterations=params.frank_wolfe_max_iterations,
            fw_gap_tolerance=params.frank_wolfe_gap_tolerance,
            coordinate_passes=params.coordinate_descent_passes,
            root_relaxation=relaxation_by_theta[theta_index],
            initial_incumbent=feasible_by_theta[theta_index],
        )
        feasible_by_theta[theta_index] = {
            "indices": branch["indices"],
            "selected_delays": branch["selected_delays"],
            "objective": branch["objective"],
            "coordinate_changes": feasible_by_theta[theta_index].get("coordinate_changes", 0),
        }
        row = theta_rows[theta_index]
        row["lower_bound"] = float(max(float(row["lower_bound"]), branch["lower_bound"]))
        row["lower_bound_kind"] = "branch_and_bound_open_node_bound"
        row["discrete_upper_bound"] = float(branch["objective"])
        row["branch_and_bound_nodes_solved"] = int(branch["nodes_solved"])
        row["branch_and_bound_open_nodes"] = int(branch["open_nodes"])
        row["branch_and_bound_exact"] = bool(branch["exact_within_tolerance"])
        row["branch_and_bound_budget_exhausted"] = bool(branch["node_budget_exhausted"])
        branch_rows.append({"theta_deg": float(theta_grid[theta_index]), **branch})
        branch_by_theta[int(theta_index)] = branch
        if branch["objective"] < best_upper:
            best_upper = float(branch["objective"])
            best_theta_index = int(theta_index)

    # A newly improved upper bound can make additional grid points prunable;
    # existing lower bounds remain valid.  Locate the best feasible assignment.
    best_theta_index, best_discrete = min(
        feasible_by_theta.items(), key=lambda item: (item[1]["objective"], item[0])
    )
    best_upper = float(best_discrete["objective"])
    lower_bounds = np.asarray([row["lower_bound"] for row in theta_rows], dtype=float)
    global_lower = float(np.min(lower_bounds))
    absolute_gap = max(best_upper - global_lower, 0.0)
    relative_gap = absolute_gap / max(abs(best_upper), 1.0)
    near_threshold = best_upper + float(params.near_optimal_objective_slack)
    near_thetas = [
        float(row["theta_deg"]) for row in theta_rows
        if float(row["lower_bound"]) <= near_threshold
    ]
    ambiguity_diameter = _theta_diameter(near_thetas)
    best_relaxation = relaxation_by_theta.get(best_theta_index)
    best_branch = branch_by_theta.get(best_theta_index)
    if best_branch is not None and bool(best_branch["exact_within_tolerance"]):
        best_integrality = 1.0
    else:
        best_integrality = (
            float(best_relaxation["integrality_mean_max_mass"])
            if best_relaxation is not None else 1.0
        )

    reasons: list[str] = []
    gap_limit = max(
        float(params.certificate_absolute_gap_max),
        float(params.certificate_relative_gap_max) * max(abs(best_upper), 1.0),
    )
    if absolute_gap > gap_limit:
        reasons.append("grid_optimality_gap_not_closed")
    if ambiguity_diameter > float(params.certificate_ambiguity_diameter_max_deg):
        reasons.append("near_optimal_direction_set_too_wide")
    if best_integrality < float(params.certificate_integrality_min):
        reasons.append("best_relaxation_too_fractional")
    if graph_diag["component_count"] != 1:
        reasons.append("pair_graph_disconnected")
    if graph_diag["cycle_rank"] < 1:
        reasons.append("no_independent_cycle_redundancy")
    certified = len(reasons) == 0

    grid_theta = float(theta_grid[best_theta_index])
    selected_delays_s = best_discrete["selected_delays"] / float(sample_rate_hz)
    selected_indices = best_discrete["indices"]
    # Refine only the continuous direction for the already selected one-hot
    # assignment.  The certificate remains explicitly tied to the frozen grid.
    lo = max(float(params.theta_min_deg), grid_theta - float(params.refine_radius_deg))
    hi = min(float(params.theta_max_deg), grid_theta + float(params.refine_radius_deg))
    def fixed_assignment_geometry_cost(theta: float) -> float:
        predicted = _predicted_pair_delays(positions, pairs, float(theta)) * float(sample_rate_hz)
        residual = best_discrete["selected_delays"] - predicted
        return float(np.sum(quality * residual * residual))
    refinement = minimize_scalar(
        fixed_assignment_geometry_cost,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1.0e-3},
    )
    refined_theta = float(refinement.x if refinement.success else grid_theta)

    selected_amplitudes = amplitudes[np.arange(len(selected_indices)), selected_indices]
    selected_peak_costs = peak_cost[np.arange(len(selected_indices)), selected_indices]
    return {
        "grid_estimate_deg": grid_theta,
        "refined_estimate_deg": refined_theta,
        "selected_indices": selected_indices,
        "selected_delays_s": selected_delays_s,
        "selected_amplitudes": selected_amplitudes,
        "selected_peak_costs": selected_peak_costs,
        "certified": bool(certified),
        "certificate_reasons": reasons,
        "best_discrete_upper_bound": best_upper,
        "global_relaxation_lower_bound": global_lower,
        "absolute_optimality_gap": float(absolute_gap),
        "relative_optimality_gap": float(relative_gap),
        "near_optimal_theta_values_deg": near_thetas,
        "near_optimal_theta_diameter_deg": float(ambiguity_diameter),
        "best_relaxation_integrality": float(best_integrality),
        "grid_size": int(len(theta_grid)),
        "relaxation_solved_grid_points": int(sum(bool(row["relaxation_solved"]) for row in theta_rows)),
        "unary_pruned_grid_points": int(sum(not bool(row["relaxation_solved"]) for row in theta_rows)),
        "branch_and_bound_grid_points": int(len(branch_rows)),
        "branch_and_bound_rows": branch_rows,
        "graph_diagnostics": graph_diag,
        "theta_certificate_rows": theta_rows,
        "refinement_success": bool(refinement.success),
        "certificate_scope": "frozen_azimuth_grid_and_observed_topk_candidate_bank",
        "candidate_presence_guaranteed": False,
        "propagation_model_correctness_guaranteed": False,
    }


def certified_multihypothesis_hodge_estimates(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: CertifiedMultiHypothesisParameters = CertifiedMultiHypothesisParameters(),
    forensic_params: BaselineForensicParameters = BaselineForensicParameters(),
) -> CertifiedMultiHypothesisResult:
    """Return E0, raw joint estimate, and certificate-controlled E0 fallback."""
    anchors, anchor_diag = r5_anchor_estimates(
        audio, sample_rate_hz, microphone_positions_m, params=forensic_params
    )
    bank = extract_gcc_peak_bank(
        audio,
        sample_rate_hz,
        microphone_positions_m,
        params=params.hodge_bank_parameters(),
    )
    solved = solve_certified_candidate_bank(
        bank["candidate_delays_s"],
        bank["candidate_amplitudes"],
        bank["candidate_valid"],
        bank["pairs"],
        np.asarray(microphone_positions_m, dtype=float),
        sample_rate_hz,
        bank["base_quality"],
        e0_estimate_deg=float(anchors[E0]),
        params=params,
    )
    raw = float(solved["refined_estimate_deg"])
    endpoint = float(raw if solved["certified"] else anchors[E0])
    pair_rows: list[dict] = []
    for row, index, delay, amplitude, cost in zip(
        bank["pair_rows"],
        solved["selected_indices"],
        solved["selected_delays_s"],
        solved["selected_amplitudes"],
        solved["selected_peak_costs"],
    ):
        pair_rows.append({
            **row,
            "selected_peak_index_r7": int(index),
            "selected_delay_s_r7": float(delay),
            "selected_amplitude_r7": float(amplitude),
            "selected_peak_cost_r7": float(cost),
        })
    diagnostics = {
        "parameters": params.as_dict(),
        "fallback_anchor": E0,
        "e0_estimate_deg": float(anchors[E0]),
        "raw_joint_estimate_deg": raw,
        "certified_endpoint_deg": endpoint,
        "certificate_accept": bool(solved["certified"]),
        "certificate_reasons": list(solved["certificate_reasons"]),
        "certificate": {k: v for k, v in solved.items() if k not in {
            "selected_indices", "selected_delays_s", "selected_amplitudes",
            "selected_peak_costs", "theta_certificate_rows", "branch_and_bound_rows",
        }},
        "theta_certificate_rows": solved["theta_certificate_rows"],
        "pair_rows": pair_rows,
        "anchor_diagnostics": anchor_diag,
        "truth_used_for_estimation": False,
        "historical_safe_gate_used": False,
    }
    return CertifiedMultiHypothesisResult(
        estimates_deg={
            E0: float(anchors[E0]),
            R7_RAW: raw,
            R7_CERTIFIED: endpoint,
        },
        diagnostics=diagnostics,
    )
