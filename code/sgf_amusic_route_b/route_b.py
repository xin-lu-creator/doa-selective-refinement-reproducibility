# =========================================================
# File        : route_b.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the route b module used by the reproducibility workflow.
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

from dataclasses import asdict, dataclass, field
from typing import Iterable
import warnings

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .array_model import music_spectrum_from_covariance, sample_covariance
from .sgf import SGFFeatureWeights, sgf_intensity


ALGORITHM_VERSION = "SGF-AMUSIC-P1-RouteB-v2.0"


@dataclass(frozen=True)
class RouteBParameters:
    theta_min: float = -40.0
    theta_max: float = 40.0
    delta_c: float = 2.0
    delta_f: float = 0.1
    kappa: int = 4
    alpha: float = 0.6
    gamma: float = 1.0
    min_separation_deg: float = 2.0
    candidate_nms_deg: float = 2.0
    min_refine_radius_deg: float = 3.0
    max_refine_radius_deg: float = 5.0
    basin_margin_deg: float = 0.5
    rescue_sector_deg: float = 10.0
    rescue_relative_floor: float = 0.03
    fallback_global_fine_scan: bool = True
    smoothing_window: int = 5
    slope_window: int = 5
    prominence_window: int = 11
    d_over_lambda: float = 0.5
    feature_weights: SGFFeatureWeights = field(default_factory=SGFFeatureWeights)


ROUTE_B_DEFAULTS = RouteBParameters()


def parameter_dict(params: RouteBParameters = ROUTE_B_DEFAULTS) -> dict:
    out = asdict(params)
    return out


def _strict_grid(theta_min: float, theta_max: float, step: float) -> np.ndarray:
    if step <= 0 or theta_max <= theta_min:
        raise ValueError("Invalid grid limits or step")
    n = int(round((theta_max - theta_min) / step))
    return theta_min + step * np.arange(n + 1, dtype=float)


def _edge_peak_indices(score: np.ndarray) -> list[int]:
    edge: list[int] = []
    if len(score) == 1:
        return [0]
    if score[0] > score[1]:
        edge.append(0)
    if score[-1] > score[-2]:
        edge.append(len(score) - 1)
    return edge


def _detect_peak_basins(grid: np.ndarray, score: np.ndarray, params: RouteBParameters) -> list[dict]:
    """Detect one seed per one-dimensional peak basin.

    SciPy's local-maximum detector supplies one representative for a plateau.
    Width is measured at half prominence and then clamped to a finite radius,
    preventing a chain of adjacent high-valued atoms from turning into a
    nearly full-field refinement interval.
    """
    spacing = float(np.median(np.diff(grid)))
    distance_samples = max(1, int(np.ceil(params.candidate_nms_deg / spacing)))
    peaks, _ = find_peaks(score, distance=distance_samples, plateau_size=1)
    edge = _edge_peak_indices(score)
    if edge:
        peaks = np.unique(np.concatenate([peaks, np.asarray(edge, dtype=int)]))
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(score))], dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prominences, left_bases, right_bases = peak_prominences(score, peaks)
        widths, _, left_ips, right_ips = peak_widths(score, peaks, rel_height=0.5)
    basins: list[dict] = []
    for j, idx in enumerate(peaks):
        theta_peak = float(grid[idx])
        left_theta = float(np.interp(left_ips[j], np.arange(len(grid)), grid))
        right_theta = float(np.interp(right_ips[j], np.arange(len(grid)), grid))
        half_width = max(theta_peak - left_theta, right_theta - theta_peak)
        radius = float(np.clip(
            half_width + params.basin_margin_deg,
            params.min_refine_radius_deg,
            params.max_refine_radius_deg,
        ))
        prominence = float(prominences[j])
        rank_score = float(score[idx] + 0.5 * prominence)
        basins.append({
            "index": int(idx),
            "theta_peak": theta_peak,
            "peak_score": float(score[idx]),
            "prominence": prominence,
            "rank_score": rank_score,
            "left_base_theta": float(grid[int(left_bases[j])]),
            "right_base_theta": float(grid[int(right_bases[j])]),
            "half_prominence_width_deg": float(widths[j] * spacing),
            "radius_deg": radius,
            "source": "local_peak",
        })
    basins.sort(key=lambda item: item["rank_score"], reverse=True)
    return basins


def _far_enough(theta: float, selected: Iterable[dict], min_distance: float) -> bool:
    return all(abs(theta - float(item["theta_peak"])) >= min_distance - 1e-12 for item in selected)


def _sector_rescue_candidates(
    grid: np.ndarray,
    score: np.ndarray,
    selected: list[dict],
    limit: int,
    params: RouteBParameters,
) -> list[dict]:
    """Add at most one residual maximum per angular sector.

    This is only a coverage rescue. It allows a weak, spatially separated source
    to retain its own refinement interval without admitting many adjacent atoms
    from a strong broad lobe.
    """
    if len(selected) >= limit:
        return selected
    best = float(np.max(score))
    sector = max(params.rescue_sector_deg, 2.0 * params.delta_c)
    edges = np.arange(params.theta_min, params.theta_max + sector, sector)
    rescue: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (grid >= lo) & (grid < hi if hi < params.theta_max else grid <= hi)
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            continue
        idx = int(ids[np.argmax(score[ids])])
        theta = float(grid[idx])
        if score[idx] < params.rescue_relative_floor * best:
            continue
        if not _far_enough(theta, selected + rescue, params.rescue_sector_deg / 2.0):
            continue
        rescue.append({
            "index": idx,
            "theta_peak": theta,
            "peak_score": float(score[idx]),
            "prominence": 0.0,
            "rank_score": float(score[idx]),
            "left_base_theta": theta,
            "right_base_theta": theta,
            "half_prominence_width_deg": 0.0,
            "radius_deg": params.min_refine_radius_deg,
            "source": "sector_rescue",
        })
    rescue.sort(key=lambda item: item["rank_score"], reverse=True)
    return selected + rescue[: max(0, limit - len(selected))]


def _select_basins(
    grid: np.ndarray,
    score: np.ndarray,
    num_sources: int,
    params: RouteBParameters,
) -> list[dict]:
    limit = max(int(num_sources), int(params.kappa) * int(num_sources))
    basins = _detect_peak_basins(grid, score, params)
    selected: list[dict] = []
    for basin in basins:
        if _far_enough(basin["theta_peak"], selected, params.candidate_nms_deg):
            selected.append(basin)
        if len(selected) >= limit:
            break
    # Sector rescue is used only when the local-peak detector found fewer
    # than K distinct basins. It must not fill every unused top-L slot,
    # because doing so would unnecessarily widen the refinement field.
    if len(selected) < int(num_sources):
        selected = _sector_rescue_candidates(grid, score, selected, int(num_sources), params)
    selected.sort(key=lambda item: item["rank_score"], reverse=True)
    return selected[:limit]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((float(a), float(b)) for a, b in intervals)
    merged: list[list[float]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1] + 1e-12:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return [(float(lo), float(hi)) for lo, hi in merged]


def _candidate_intervals(selected: list[dict], params: RouteBParameters) -> list[tuple[float, float]]:
    raw = []
    for item in selected:
        radius = float(item["radius_deg"])
        center = float(item["theta_peak"])
        raw.append((
            max(params.theta_min, center - radius),
            min(params.theta_max, center + radius),
        ))
    return _merge_intervals(raw)


def _local_maxima_with_edges(score: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(score, plateau_size=1)
    edge = _edge_peak_indices(score)
    if edge:
        peaks = np.unique(np.concatenate([peaks, np.asarray(edge, dtype=int)]))
    if len(peaks) == 0:
        peaks = np.asarray([int(np.argmax(score))], dtype=int)
    return peaks


def _select_final_peaks(
    interval_results: list[dict],
    num_sources: int,
    min_separation_deg: float,
) -> list[float]:
    candidates: list[dict] = []
    best_per_interval: list[dict] = []
    for interval_id, result in enumerate(interval_results):
        grid = result["grid"]
        score = result["music"]
        peaks = _local_maxima_with_edges(score)
        local = [
            {"theta": float(grid[i]), "score": float(score[i]), "interval_id": interval_id}
            for i in peaks
        ]
        local.sort(key=lambda item: item["score"], reverse=True)
        candidates.extend(local)
        if local:
            best_per_interval.append(local[0])

    # Select globally from all fine-grid local maxima. A merged interval may
    # legitimately contain more than one close source, so forcing one peak per
    # interval would incorrectly promote noise peaks from unrelated intervals.
    selected: list[float] = []
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        theta = item["theta"]
        if all(abs(theta - x) >= min_separation_deg - 1e-12 for x in selected):
            selected.append(theta)
        if len(selected) >= num_sources:
            break
    return selected


def _global_fine_fallback(R: np.ndarray, num_sources: int, params: RouteBParameters) -> tuple[np.ndarray, np.ndarray]:
    grid = _strict_grid(params.theta_min, params.theta_max, params.delta_f)
    spectrum = music_spectrum_from_covariance(R, num_sources, grid, params.d_over_lambda)
    peaks = _local_maxima_with_edges(spectrum)
    ranked = sorted(peaks, key=lambda idx: spectrum[idx], reverse=True)
    selected: list[float] = []
    for idx in ranked:
        theta = float(grid[idx])
        if all(abs(theta - x) >= params.min_separation_deg - 1e-12 for x in selected):
            selected.append(theta)
        if len(selected) >= num_sources:
            break
    if len(selected) < num_sources:
        for idx in np.argsort(spectrum)[::-1]:
            theta = float(grid[idx])
            if all(abs(theta - x) >= params.min_separation_deg - 1e-12 for x in selected):
                selected.append(theta)
            if len(selected) >= num_sources:
                break
    return np.asarray(sorted(selected), dtype=float), spectrum


def sgf_amusic_from_covariance(
    R: np.ndarray,
    num_sources: int,
    params: RouteBParameters = ROUTE_B_DEFAULTS,
    alpha: float | None = None,
    return_details: bool = False,
):
    K = int(num_sources)
    a = params.alpha if alpha is None else float(alpha)
    coarse_grid = _strict_grid(params.theta_min, params.theta_max, params.delta_c)
    coarse_music = music_spectrum_from_covariance(R, K, coarse_grid, params.d_over_lambda)
    intensity, features = sgf_intensity(
        coarse_grid,
        coarse_music,
        smoothing_window=params.smoothing_window,
        slope_window=params.slope_window,
        prominence_window=params.prominence_window,
        weights=params.feature_weights,
    )
    coarse_score = coarse_music * (1.0 + a * intensity) ** params.gamma
    coarse_score /= np.max(coarse_score) + 1e-14

    selected_basins = _select_basins(coarse_grid, coarse_score, K, params)
    intervals = _candidate_intervals(selected_basins, params)
    interval_results: list[dict] = []
    for lo, hi in intervals:
        grid = _strict_grid(lo, hi, params.delta_f)
        # Fine spectra must remain on one common absolute scale. Normalizing
        # each interval independently would make unrelated interval maxima all
        # equal to one and corrupt the global final ranking.
        spectrum = music_spectrum_from_covariance(
            R, K, grid, params.d_over_lambda, normalize=False
        )
        interval_results.append({"lo": lo, "hi": hi, "grid": grid, "music": spectrum})

    selected = _select_final_peaks(interval_results, K, params.min_separation_deg)
    fallback_used = False
    fallback_spectrum = None
    if len(selected) < K and params.fallback_global_fine_scan:
        estimate, fallback_spectrum = _global_fine_fallback(R, K, params)
        fallback_used = True
    else:
        estimate = np.asarray(sorted(selected), dtype=float)

    if len(estimate) != K:
        raise RuntimeError(f"Estimator returned {len(estimate)} peaks for K={K}")

    if not return_details:
        return estimate

    total_width = float(sum(hi - lo for lo, hi in intervals))
    details = {
        "theta_hat": estimate,
        "algorithm_version": ALGORITHM_VERSION,
        "parameters": parameter_dict(params),
        "alpha_effective": a,
        "coarse_grid": coarse_grid,
        "coarse_music": coarse_music,
        "coarse_score": coarse_score,
        "coarse_intensity": intensity,
        "coarse_features": features,
        "selected_basins": selected_basins,
        "intervals": intervals,
        "interval_results": interval_results,
        "fallback_used": fallback_used,
        "fallback_spectrum": fallback_spectrum,
        "scan_count": int(len(coarse_grid) + sum(len(x["grid"]) for x in interval_results) + (0 if fallback_spectrum is None else len(fallback_spectrum))),
        "interval_count": int(len(intervals)),
        "total_refinement_width_deg": total_width,
    }
    return details


def sgf_amusic(
    X: np.ndarray,
    num_sources: int,
    params: RouteBParameters = ROUTE_B_DEFAULTS,
    alpha: float | None = None,
    return_details: bool = False,
):
    return sgf_amusic_from_covariance(
        sample_covariance(X),
        num_sources,
        params=params,
        alpha=alpha,
        return_details=return_details,
    )
