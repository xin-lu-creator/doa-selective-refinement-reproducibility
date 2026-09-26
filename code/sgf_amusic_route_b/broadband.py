# =========================================================
# File        : broadband.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the broadband module used by the reproducibility workflow.
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

from dataclasses import asdict, dataclass, replace
from typing import Sequence

import numpy as np
from scipy.signal import stft

from .frequency_mandala import (
    cluster_frequency_covariances,
    cluster_precomputed_distance,
    frequency_distance_matrix,
    principal_phase_focused_covariance,
    soft_frequency_mandala_weights,
)
from .interfaces import CovarianceEvidence
from .direct_path_matrix import (
    adaptive_multichannel_late_prediction_decomposition,
    multichannel_late_prediction_decomposition,
)
from .route_b import ROUTE_B_DEFAULTS, RouteBParameters, _candidate_intervals, _select_basins, _strict_grid


SPEED_OF_SOUND = 343.0
ALGORITHM_VERSION_BROADBAND = "LOCATA-Required-Time-Causal-Physical-Forensics-P1-v5.0a2"


@dataclass(frozen=True)
class BroadbandParameters:
    window_ms: float = 32.0
    overlap_fraction: float = 0.75
    nfft: int = 512
    covariance_duration_s: float = 0.75
    covariance_hop_s: float = 0.25
    frequency_min_hz: float = 350.0
    frequency_max_hz: float = 4000.0
    min_frequency_bins: int = 6
    max_frequency_bins: int = 48
    num_frequency_clusters: int = 3
    min_cluster_size: int = 3
    frequency_quality_weight: float = 0.35
    hpd_floor_relative: float = 1e-6
    coarse_step_deg: float = 1.0
    fine_step_deg: float = 0.2
    fine_radius_deg: float = 1.2
    candidate_halfwidth_deg: float = 5.0
    max_candidate_intervals: int = 4
    minimum_frequency_support: float = 0.60
    incremental_evidence_tol: float = 1.0e-3
    use_frequency_mandala: bool = True
    trace_normalize_for_clustering: bool = True
    phase_focus_for_clustering: bool = True
    doa_consensus_for_clustering: bool = True
    doa_consensus_weight: float = 0.75
    doa_consensus_scale_deg: float = 8.0
    doa_consensus_radius_deg: float = 10.0
    robust_consensus_override: bool = True
    # v4.1: soft Frequency Mandala.
    soft_frequency_mandala: bool = True
    soft_covariance_temperature: float = 1.25
    soft_doa_temperature_deg: float = 12.0
    soft_minimum_factor: float = 0.25
    soft_target_effective_bins: float = 14.0
    soft_minimum_effective_bandwidth_hz: float = 1000.0
    # v4.1: sensor-frequency adaptive subarray.
    adaptive_sensor_subarray: bool = True
    minimum_subarray_sensors: int = 4
    subarray_alias_margin: float = 0.92
    subarray_ambiguity_exclusion_deg: float = 12.0
    subarray_coherence_grid_step_deg: float = 4.0
    # Same-waveform narrowband baseline.
    narrowband_reference_hz: float = 1600.0
    # v4.2: direct/late covariance decomposition and GEVD diagnostics.
    direct_path_decomposition: bool = True
    late_prediction_delay_frames: int = 2
    late_prediction_order: int = 3
    late_prediction_ridge_relative: float = 1e-3
    direct_path_min_score: float = 0.05
    # Hard sensor-frequency ambiguity protection.
    enforce_projected_gap_hard: bool = True
    maximum_projected_gap_ratio: float = 1.0
    # Experimental K>1 direct-path covariance fitting is disabled until the
    # dedicated matrix evidence gate is validated under reverberation.
    experimental_multisource_direct_path_joint: bool = False
    # v4.3: fail-safe real-data and multi-source resolution gates.
    direct_path_safety_gate: bool = True
    multisource_resolution_gate: bool = True
    minimum_resolved_separation_deg: float = 2.0
    minimum_two_source_model_gain: float = 0.006
    minimum_two_source_frequency_support: float = 0.55
    minimum_two_source_power_balance: float = 0.15
    minimum_latent_two_source_power_balance: float = 0.15
    # v4.4: temporal/reflection-aware multi-source evidence.
    temporal_resolution_blocks: int = 3
    minimum_temporal_source_persistence: float = 0.67
    minimum_direct_source_fraction: float = 0.22
    minimum_latent_temporal_persistence: float = 0.34
    minimum_temporal_log_power_ratio_std: float = 0.10
    minimum_distinct_peak_family_support: float = 0.25
    # v4.4: adaptive real-data late-prediction bank.  The default remains
    # disabled to preserve the frozen single-source RT60 result.
    adaptive_late_prediction: bool = False
    adaptive_prediction_minimum_utility: float = 0.01
    adaptive_prediction_minimum_explained_fraction: float = 0.03
    adaptive_prediction_maximum_condition: float = 1.0e9
    # v4.5: coherent multi-frequency focused subspace evidence.
    focused_multifrequency_subspace: bool = True
    focused_reference_frequency_hz: float | None = None
    focused_grid_step_deg: float = 2.0
    focused_minimum_frequency_bins: int = 6
    focused_temporal_blocks: int = 4
    focused_temporal_contrast_mix: float = 0.35
    focused_average_family_weight: float = 4.0
    focused_temporal_family_weight: float = 4.0
    focused_joint_diagonalization_family_weight: float = 0.5
    per_frequency_joint_diagonalization_family_weight: float = 1.0
    per_frequency_family_weight: float = 0.25


BROADBAND_DEFAULTS = BroadbandParameters()

_SENSOR_SUBARRAY_CACHE: dict[tuple, tuple[np.ndarray, dict]] = {}



def broadband_parameter_dict(params: BroadbandParameters = BROADBAND_DEFAULTS) -> dict:
    return asdict(params)


def _hermitian(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=complex)
    return 0.5 * (a + a.conj().T)


def steering_matrix_geometry(
    microphone_positions_m: np.ndarray,
    azimuth_deg: Sequence[float] | np.ndarray,
    frequency_hz: float,
    *,
    speed_of_sound: float = SPEED_OF_SOUND,
    elevation_deg: float = 0.0,
) -> np.ndarray:
    """Far-field steering matrix for arbitrary 3-D microphone geometry."""
    pos = np.asarray(microphone_positions_m, dtype=float)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError("microphone_positions_m must have shape (M,2) or (M,3)")
    if pos.shape[1] == 2:
        pos = np.column_stack([pos, np.zeros(len(pos))])
    pos = pos - np.mean(pos, axis=0, keepdims=True)
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float).reshape(-1))
    el = np.deg2rad(float(elevation_deg))
    directions = np.stack(
        [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.full_like(az, np.sin(el))],
        axis=0,
    )
    phase = 2.0j * np.pi * float(frequency_hz) / float(speed_of_sound) * (pos @ directions)
    return np.exp(phase)


def steering_matrix_geometry_model(
    microphone_positions_m: np.ndarray,
    azimuth_deg: Sequence[float] | np.ndarray,
    frequency_hz: float,
    *,
    speed_of_sound: float = SPEED_OF_SOUND,
    elevation_deg: float = 0.0,
    source_distance_m: float = float("inf"),
    spherical_amplitude: bool = False,
) -> np.ndarray:
    """Auditable far/near-field steering family for v5.0 model forensics.

    ``source_distance_m=inf`` is exactly the existing far-field implementation.
    Finite distances use a source referenced to the microphone centroid and the
    differential path ``||r u-b_m||-r``.  The function is delivered disabled in
    v5.0a2 formal evaluation and is used only by independent gold-path tests;
    v5.0b may enable it after preregistration.
    """
    distance = float(source_distance_m)
    if not np.isfinite(distance):
        return steering_matrix_geometry(
            microphone_positions_m,
            azimuth_deg,
            frequency_hz,
            speed_of_sound=speed_of_sound,
            elevation_deg=elevation_deg,
        )
    if distance <= 0:
        raise ValueError("source_distance_m must be positive or infinity")
    pos = np.asarray(microphone_positions_m, dtype=float)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError("microphone_positions_m must have shape (M,2) or (M,3)")
    if pos.shape[1] == 2:
        pos = np.column_stack([pos, np.zeros(len(pos))])
    # Finite-range LOCATA truth is referenced to the official array pose
    # origin, so microphone coordinates remain in that audited local frame.
    # Re-centering is harmless only in the far field (global phase); at finite
    # range it changes the physical source centre and creates a deterministic
    # elevation/range bias when the microphone centroid is offset.
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float).reshape(-1))
    el = np.deg2rad(float(elevation_deg))
    directions = np.stack(
        [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.full_like(az, np.sin(el))],
        axis=1,
    )
    source = distance * directions
    paths = np.linalg.norm(source[None, :, :] - pos[:, None, :], axis=2)
    differential = paths - distance
    manifold = np.exp(
        -2.0j * np.pi * float(frequency_hz) / float(speed_of_sound) * differential
    )
    if spherical_amplitude:
        manifold = manifold * (distance / np.maximum(paths, 1e-12))
    return manifold


def conservative_aliasing_limit_hz(
    microphone_positions_m: np.ndarray,
    *,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> float:
    pos = np.asarray(microphone_positions_m, dtype=float)
    if len(pos) < 2:
        return float("inf")
    distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    distances[distances <= 0] = np.inf
    nearest = np.min(distances, axis=1)
    d = float(np.max(nearest))
    return float(speed_of_sound / (2.0 * d)) if d > 0 else float("inf")


def _dominant_axis_projection(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=float)
    centered = pos - np.mean(pos, axis=0, keepdims=True)
    if len(pos) <= 1:
        return np.arange(len(pos), dtype=float), np.arange(len(pos), dtype=int)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    order = np.argsort(projection)
    return projection, order


def manifold_mutual_coherence(
    positions_m: np.ndarray,
    frequency_hz: float,
    *,
    theta_min_deg: float = -80.0,
    theta_max_deg: float = 80.0,
    grid_step_deg: float = 4.0,
    exclusion_deg: float = 12.0,
) -> float:
    """Worst steering ambiguity for sufficiently separated directions."""
    grid = _strict_grid(theta_min_deg, theta_max_deg, grid_step_deg)
    A = steering_matrix_geometry(positions_m, grid, frequency_hz)
    A = A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
    coherence = np.abs(A.conj().T @ A)
    separation = np.abs(grid[:, None] - grid[None, :])
    mask = separation >= float(exclusion_deg)
    return float(np.max(coherence[mask])) if np.any(mask) else 1.0


def adaptive_sensor_indices(
    microphone_positions_m: np.ndarray,
    frequency_hz: float,
    *,
    minimum_sensors: int = 4,
    alias_margin: float = 0.92,
    ambiguity_exclusion_deg: float = 12.0,
    coherence_grid_step_deg: float = 4.0,
    enforce_projected_gap_hard: bool = True,
    maximum_projected_gap_ratio: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Select a frequency-dependent contiguous subarray on the dominant axis.

    Candidate subarrays must have adjacent projected gaps no larger than the
    half-wavelength bound (with margin).  The final score balances low
    manifold ambiguity, aperture, and sensor count.  This is auditable and
    deterministic; the original channel order is preserved in returned ids.
    """
    pos = np.asarray(microphone_positions_m, dtype=float)
    m = len(pos)
    if m < minimum_sensors:
        raise ValueError("Not enough sensors for adaptive subarray")
    cache_key = (
        tuple(np.round(pos.reshape(-1), 9).tolist()),
        round(float(frequency_hz), 6),
        int(minimum_sensors),
        round(float(alias_margin), 6),
        round(float(ambiguity_exclusion_deg), 6),
        round(float(coherence_grid_step_deg), 6),
        bool(enforce_projected_gap_hard),
        round(float(maximum_projected_gap_ratio), 6),
    )
    cached = _SENSOR_SUBARRAY_CACHE.get(cache_key)
    if cached is not None:
        ids_cached, diag_cached = cached
        return ids_cached.copy(), dict(diag_cached)
    projection, order = _dominant_axis_projection(pos)
    sorted_projection = projection[order]
    max_gap = float(alias_margin) * SPEED_OF_SOUND / (2.0 * max(float(frequency_hz), 1.0))
    gaps = np.diff(sorted_projection)
    split = np.flatnonzero(gaps > max_gap) + 1
    runs = np.split(order, split)
    candidates: list[np.ndarray] = []
    for run in runs:
        if len(run) < int(minimum_sensors):
            continue
        candidates.append(np.asarray(run, dtype=int))
        # Nested contiguous windows capture the dense centre of nonuniform arrays.
        for size in range(int(minimum_sensors), len(run)):
            starts = {0, len(run) - size, max((len(run) - size) // 2, 0)}
            for start in sorted(starts):
                candidates.append(np.asarray(run[start:start + size], dtype=int))
    # The full array is considered only when it satisfies the frequency-specific
    # projected-gap bound.  This prevents sparse large-aperture arrays from
    # reappearing at high frequency because of a misleading coarse coherence score.
    full_ids = np.arange(m, dtype=int)
    full_proj, _ = _dominant_axis_projection(pos)
    full_sorted = np.sort(full_proj)
    full_gap = float(np.max(np.diff(full_sorted))) if len(full_sorted) > 1 else 0.0
    if (not enforce_projected_gap_hard) or full_gap <= max_gap * float(maximum_projected_gap_ratio):
        candidates.append(full_ids)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for ids in candidates:
        ids = np.sort(np.unique(ids))
        if len(ids) >= int(minimum_sensors):
            unique[tuple(ids.tolist())] = ids
    full_aperture = max(float(np.ptp(sorted_projection)), 1e-12)
    rows = []
    for ids in unique.values():
        subset = pos[ids]
        mu = manifold_mutual_coherence(
            subset,
            frequency_hz,
            grid_step_deg=coherence_grid_step_deg,
            exclusion_deg=ambiguity_exclusion_deg,
        )
        sub_proj, sub_order = _dominant_axis_projection(subset)
        sorted_sub = np.sort(sub_proj)
        actual_max_gap = float(np.max(np.diff(sorted_sub))) if len(sorted_sub) > 1 else 0.0
        gap_ratio = actual_max_gap / max(max_gap, 1e-12)
        if enforce_projected_gap_hard and gap_ratio > float(maximum_projected_gap_ratio) + 1e-12:
            continue
        aperture_ratio = float(np.ptp(sub_proj)) / full_aperture if len(subset) > 1 else 0.0
        count_ratio = len(ids) / m
        # Ambiguity is primary; aperture and channel count break near ties.
        # A strong penalty prevents a sparse full array from winning merely by
        # aperture when it violates the frequency-specific adjacent-gap bound.
        score = (
            1.8 * (1.0 - mu)
            + 0.35 * aperture_ratio
            + 0.15 * count_ratio
            - 1.8 * max(gap_ratio - 1.0, 0.0)
        )
        physical_gap_ratio = actual_max_gap / max(SPEED_OF_SOUND / (2.0 * max(float(frequency_hz), 1.0)), 1e-12)
        rows.append((score, -mu, aperture_ratio, count_ratio, ids, mu, actual_max_gap, gap_ratio, physical_gap_ratio))
    if not rows:
        # Deterministic fail-safe: use the densest contiguous minimum-size window.
        windows = [order[i:i + int(minimum_sensors)] for i in range(0, m - int(minimum_sensors) + 1)]
        ids = min(windows, key=lambda w: float(np.max(np.diff(np.sort(projection[w])))))
        subset = pos[np.asarray(ids, dtype=int)]
        mu = manifold_mutual_coherence(subset, frequency_hz, grid_step_deg=coherence_grid_step_deg, exclusion_deg=ambiguity_exclusion_deg)
        actual = float(np.max(np.diff(np.sort(_dominant_axis_projection(subset)[0]))))
        physical = actual / max(SPEED_OF_SOUND / (2.0 * max(float(frequency_hz), 1.0)), 1e-12)
        rows.append((1.8 * (1.0 - mu), -mu, 0.0, len(ids) / m, np.asarray(ids, dtype=int), mu, actual, actual / max(max_gap, 1e-12), physical))
    best = max(rows, key=lambda row: row[:4])
    ids = np.asarray(best[4], dtype=int)
    diagnostics = {
        "sensor_count": int(len(ids)),
        "sensor_indices": ids.tolist(),
        "manifold_mutual_coherence": float(best[5]),
        "maximum_projected_gap_allowed_m": max_gap,
        "candidate_subarray_count": len(rows),
        "actual_max_projected_gap_m": float(best[6]),
        "projected_gap_ratio": float(best[7]),
        "physical_half_wavelength_gap_ratio": float(best[8]),
        "projected_gap_hard_constraint": bool(enforce_projected_gap_hard),
        "maximum_projected_gap_ratio": float(maximum_projected_gap_ratio),
    }
    _SENSOR_SUBARRAY_CACHE[cache_key] = (ids.copy(), dict(diagnostics))
    return ids, diagnostics


def _positions_for_evidence(item: CovarianceEvidence, microphone_positions_m: np.ndarray) -> np.ndarray:
    pos = np.asarray(microphone_positions_m, dtype=float)
    if item.sensor_indices is None:
        return pos
    ids = np.asarray(item.sensor_indices, dtype=int)
    return pos[ids]


def _full_covariance_for_clustering(item: CovarianceEvidence) -> np.ndarray:
    full = item.metadata.get("full_covariance") if item.metadata else None
    return np.asarray(full if full is not None else item.covariance, dtype=complex)


def extract_stft_covariances(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    *,
    params: BroadbandParameters = BROADBAND_DEFAULTS,
) -> tuple[list[CovarianceEvidence], dict]:
    """Extract auditable per-frequency covariance evidence.

    With adaptive subarrays enabled, the full-array global aliasing limit is
    not used as a hard frequency cutoff.  Each bin selects a suitable sensor
    subset and stores its mapping to the original array.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim != 2:
        raise ValueError("audio must have shape (sensors, samples)")
    positions = np.asarray(microphone_positions_m, dtype=float)
    if len(positions) != x.shape[0]:
        raise ValueError("microphone position count must match audio sensors")
    fs = float(sample_rate_hz)
    nperseg = max(32, int(round(params.window_ms * 1e-3 * fs)))
    noverlap = min(nperseg - 1, int(round(params.overlap_fraction * nperseg)))
    nfft = max(int(params.nfft), nperseg)
    spectra = []
    freqs = None
    times = None
    for channel in x:
        f, t, z = stft(
            channel,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary=None,
            padded=False,
        )
        spectra.append(z)
        freqs, times = f, t
    Z = np.stack(spectra, axis=0)
    alias_limit = conservative_aliasing_limit_hz(positions)
    configured_fmax = min(float(params.frequency_max_hz), 0.49 * fs)
    fmax = configured_fmax if params.adaptive_sensor_subarray else min(configured_fmax, 0.98 * alias_limit)
    valid_ids = np.flatnonzero((freqs >= params.frequency_min_hz) & (freqs <= fmax))
    if valid_ids.size == 0:
        raise RuntimeError("No STFT frequency bin remains after frequency limits")
    evidences: list[CovarianceEvidence] = []
    qualities = []
    for fi in valid_ids:
        Yfull = Z[:, fi, :]
        if Yfull.shape[1] < 3:
            continue
        Rfull = _hermitian(Yfull @ Yfull.conj().T / Yfull.shape[1])
        full_decomposition = None
        if params.direct_path_decomposition:
            if params.adaptive_late_prediction:
                full_decomposition = adaptive_multichannel_late_prediction_decomposition(
                    Yfull,
                    floor_relative=params.hpd_floor_relative,
                    minimum_utility=params.adaptive_prediction_minimum_utility,
                    minimum_explained_fraction=params.adaptive_prediction_minimum_explained_fraction,
                    maximum_predictor_condition=params.adaptive_prediction_maximum_condition,
                )
            else:
                full_decomposition = multichannel_late_prediction_decomposition(
                    Yfull,
                    delay_frames=params.late_prediction_delay_frames,
                    prediction_order=params.late_prediction_order,
                    ridge_relative=params.late_prediction_ridge_relative,
                    floor_relative=params.hpd_floor_relative,
                )
        if params.adaptive_sensor_subarray:
            ids, subdiag = adaptive_sensor_indices(
                positions,
                float(freqs[fi]),
                minimum_sensors=params.minimum_subarray_sensors,
                alias_margin=params.subarray_alias_margin,
                ambiguity_exclusion_deg=params.subarray_ambiguity_exclusion_deg,
                coherence_grid_step_deg=params.subarray_coherence_grid_step_deg,
                enforce_projected_gap_hard=params.enforce_projected_gap_hard,
                maximum_projected_gap_ratio=params.maximum_projected_gap_ratio,
            )
        else:
            ids = np.arange(x.shape[0], dtype=int)
            subdiag = {
                "sensor_count": int(len(ids)),
                "sensor_indices": ids.tolist(),
                "manifold_mutual_coherence": manifold_mutual_coherence(
                    positions, float(freqs[fi]),
                    grid_step_deg=params.subarray_coherence_grid_step_deg,
                    exclusion_deg=params.subarray_ambiguity_exclusion_deg,
                ),
            }
        R = _hermitian(Rfull[np.ix_(ids, ids)])
        if params.direct_path_decomposition:
            if params.adaptive_late_prediction:
                sub_decomposition = adaptive_multichannel_late_prediction_decomposition(
                    Yfull[np.asarray(ids, dtype=int)],
                    floor_relative=params.hpd_floor_relative,
                    minimum_utility=params.adaptive_prediction_minimum_utility,
                    minimum_explained_fraction=params.adaptive_prediction_minimum_explained_fraction,
                    maximum_predictor_condition=params.adaptive_prediction_maximum_condition,
                )
            else:
                sub_decomposition = multichannel_late_prediction_decomposition(
                    Yfull[np.asarray(ids, dtype=int)],
                    delay_frames=params.late_prediction_delay_frames,
                    prediction_order=params.late_prediction_order,
                    ridge_relative=params.late_prediction_ridge_relative,
                    floor_relative=params.hpd_floor_relative,
                )
        else:
            sub_decomposition = None
        eig = np.sort(np.linalg.eigvalsh(R).real)[::-1]
        energy = max(float(np.trace(R).real / R.shape[0]), 0.0)
        eigengap = float(max(eig[0] - eig[1], 0.0) / max(eig[0], 1e-12)) if len(eig) > 1 else 1.0
        diffuseness = float(np.clip(1.0 - eigengap, 0.0, 1.0))
        ambiguity_quality = max(1.0 - float(subdiag.get("manifold_mutual_coherence", 1.0)), 0.02)
        directness = float(sub_decomposition.directness_score) if sub_decomposition is not None else eigengap
        quality = np.log1p(energy) * (0.2 + 0.8 * eigengap) * np.sqrt(ambiguity_quality) * (0.35 + 0.65 * max(directness, params.direct_path_min_score))
        qualities.append(quality)
        evidences.append(CovarianceEvidence(
            covariance=R,
            frequency_hz=float(freqs[fi]),
            weight=float(quality),
            energy=energy,
            eigen_gap=eigengap,
            diffuseness=diffuseness,
            valid=True,
            snapshot_count=int(Yfull.shape[1]),
            sensor_indices=np.asarray(ids, dtype=int),
            metadata={
                **subdiag,
                "full_covariance": Rfull,
                "base_quality_weight": float(quality),
                "direct_covariance": None if sub_decomposition is None else sub_decomposition.direct_covariance,
                "late_covariance": None if sub_decomposition is None else sub_decomposition.late_covariance,
                "whitened_covariance": None if sub_decomposition is None else sub_decomposition.whitened_covariance,
                "whitener": None if sub_decomposition is None else sub_decomposition.whitener,
                "directness_score": float(directness),
                "prediction_residual_ratio": np.nan if sub_decomposition is None else float(sub_decomposition.prediction_residual_ratio),
                "direct_path_diagnostics": {} if sub_decomposition is None else sub_decomposition.diagnostics,
                "decomposition_usable": False if sub_decomposition is None else bool(sub_decomposition.decomposition_usable),
                "decomposition_utility": np.nan if sub_decomposition is None else float(sub_decomposition.decomposition_utility),
                "direct_snapshots": None if sub_decomposition is None else sub_decomposition.direct_snapshots,
                "late_snapshots": None if sub_decomposition is None else sub_decomposition.late_snapshots,
                "full_direct_covariance": None if full_decomposition is None else full_decomposition.direct_covariance,
                "full_late_covariance": None if full_decomposition is None else full_decomposition.late_covariance,
            },
        ))
    if len(evidences) < int(params.min_frequency_bins):
        raise RuntimeError(f"Only {len(evidences)} valid frequency bins; need at least {params.min_frequency_bins}")
    order = np.argsort(np.asarray(qualities))[::-1][: int(params.max_frequency_bins)]
    evidences = [evidences[int(i)] for i in np.sort(order)]
    return evidences, {
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "full_array_aliasing_limit_hz": alias_limit,
        "frequency_max_effective_hz": fmax,
        "adaptive_sensor_subarray": bool(params.adaptive_sensor_subarray),
        "stft_frame_count": int(Z.shape[2]),
        "selected_frequency_count_preweight": len(evidences),
        "time_grid_s": times,
    }


def _noise_projector(R: np.ndarray, num_sources: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(_hermitian(R))
    order = np.argsort(values.real)[::-1]
    vectors = vectors[:, order]
    En = vectors[:, int(num_sources):]
    return _hermitian(En @ En.conj().T)


def _per_frequency_music_peaks(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    num_sources: int,
    grid: np.ndarray,
) -> np.ndarray:
    peaks = []
    for item in evidences:
        Pn = _noise_projector(item.covariance, max(1, int(num_sources)))
        positions = _positions_for_evidence(item, microphone_positions_m)
        A = steering_matrix_geometry(positions, grid, float(item.frequency_hz))
        denom = np.sum(A.conj() * (Pn @ A), axis=0).real
        peaks.append(float(grid[int(np.argmin(denom))]))
    return np.asarray(peaks, dtype=float)


def select_frequency_cluster(
    evidences: Sequence[CovarianceEvidence],
    *,
    params: BroadbandParameters = BROADBAND_DEFAULTS,
    microphone_positions_m: np.ndarray | None = None,
    num_sources: int = 1,
    theta_grid_deg: np.ndarray | None = None,
) -> tuple[list[CovarianceEvidence], dict]:
    """Retained v4.0 hard-cluster baseline for scientific ablation."""
    ev = [item for item in evidences if item.valid]
    if len(ev) < params.min_frequency_bins:
        raise RuntimeError("Insufficient valid frequency evidence")
    if not params.use_frequency_mandala:
        weights = np.asarray([max(item.weight, 0.0) for item in ev])
        ids = np.argsort(weights)[::-1][: max(params.min_frequency_bins, min(len(ev), params.max_frequency_bins))]
        chosen = [ev[int(i)] for i in np.sort(ids)]
        return chosen, {"mode": "quality_only", "selected_indices": np.sort(ids).tolist()}
    covs = []
    for item in ev:
        R = _full_covariance_for_clustering(item)
        if params.phase_focus_for_clustering:
            R = principal_phase_focused_covariance(R, trace_normalize=params.trace_normalize_for_clustering)
        elif params.trace_normalize_for_clustering:
            R = R / max(float(np.trace(R).real), 1e-12)
        covs.append(R)
    qualities = [float(item.weight) for item in ev]
    grid = np.asarray(theta_grid_deg, dtype=float) if theta_grid_deg is not None else np.linspace(-80.0, 80.0, 161)
    if params.doa_consensus_for_clustering and microphone_positions_m is not None:
        peaks = _per_frequency_music_peaks(ev, microphone_positions_m, num_sources, grid)
        covariance_distance = frequency_distance_matrix(covs, floor_relative=params.hpd_floor_relative)
        nonzero = covariance_distance[covariance_distance > 0]
        scale = float(np.median(nonzero)) if nonzero.size else 1.0
        covariance_distance = covariance_distance / max(scale, 1e-12)
        angle_distance = np.abs(peaks[:, None] - peaks[None, :]) / max(params.doa_consensus_scale_deg, 1e-12)
        combined = covariance_distance + params.doa_consensus_weight * angle_distance
        np.fill_diagonal(combined, 0.0)
        result = cluster_precomputed_distance(
            combined,
            qualities=qualities,
            num_clusters=params.num_frequency_clusters,
            min_cluster_size=params.min_cluster_size,
            quality_weight=params.frequency_quality_weight,
        )
    else:
        peaks = np.full(len(ev), np.nan)
        result = cluster_frequency_covariances(
            covs,
            qualities=qualities,
            num_clusters=params.num_frequency_clusters,
            min_cluster_size=params.min_cluster_size,
            quality_weight=params.frequency_quality_weight,
            floor_relative=params.hpd_floor_relative,
        )
    ids = np.asarray(result.selected_indices, dtype=int)
    chosen = [ev[int(i)] for i in ids]
    return chosen, {
        "mode": "frequency_mandala",
        "selection_variant": "hard_cluster_v40",
        "selected_cluster": int(result.selected_cluster),
        "selected_indices": ids.tolist(),
        "labels": result.labels.tolist(),
        "cluster_scores": {str(k): float(v) for k, v in result.cluster_scores.items()},
        "cluster_dispersion": {str(k): float(v) for k, v in result.cluster_dispersion.items()},
        "distance_matrix": result.distance_matrix,
        "per_frequency_peak_deg": peaks.tolist(),
    }


def select_soft_frequency_evidence(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    num_sources: int,
    *,
    params: BroadbandParameters = BROADBAND_DEFAULTS,
    theta_grid_deg: np.ndarray | None = None,
) -> tuple[list[CovarianceEvidence], dict]:
    ev = [item for item in evidences if item.valid]
    if len(ev) < params.min_frequency_bins:
        raise RuntimeError("Insufficient valid frequency evidence")
    grid = np.asarray(theta_grid_deg, dtype=float) if theta_grid_deg is not None else np.linspace(-80.0, 80.0, 161)
    peaks = _per_frequency_music_peaks(ev, microphone_positions_m, num_sources, grid)
    covs = []
    for item in ev:
        R = _full_covariance_for_clustering(item)
        if params.phase_focus_for_clustering:
            R = principal_phase_focused_covariance(R, trace_normalize=params.trace_normalize_for_clustering)
        elif params.trace_normalize_for_clustering:
            R = R / max(float(np.trace(R).real), 1e-12)
        covs.append(R)
    result = soft_frequency_mandala_weights(
        covs,
        [float(item.frequency_hz) for item in ev],
        qualities=[float(item.weight) for item in ev],
        doa_peaks_deg=peaks if params.doa_consensus_for_clustering else None,
        floor_relative=params.hpd_floor_relative,
        covariance_temperature=params.soft_covariance_temperature,
        doa_temperature_deg=params.soft_doa_temperature_deg,
        minimum_factor=params.soft_minimum_factor,
        target_effective_bins=params.soft_target_effective_bins,
        minimum_retained_bins=params.min_frequency_bins,
        maximum_bins=params.max_frequency_bins,
        minimum_effective_bandwidth_hz=params.soft_minimum_effective_bandwidth_hz,
    )
    weighted = []
    for item, weight, structural in zip(ev, result.combined_weights, result.structural_weights):
        meta = dict(item.metadata)
        meta.update({
            "soft_mandala_weight": float(weight),
            "soft_mandala_structural_factor": float(structural),
        })
        weighted.append(replace(item, weight=float(weight), metadata=meta))
    return weighted, {
        "mode": "frequency_mandala_soft",
        "selected_indices": result.selected_indices.tolist(),
        "selected_frequencies_hz": [float(ev[i].frequency_hz) for i in result.selected_indices],
        "all_frequencies_hz": [float(item.frequency_hz) for item in ev],
        "frequency_weights": result.combined_weights.tolist(),
        "structural_weights": result.structural_weights.tolist(),
        "effective_frequency_count": float(result.effective_bin_count),
        "effective_bandwidth_hz": float(result.effective_bandwidth_hz),
        "robust_center_index": int(result.robust_center_index),
        "distance_matrix": result.distance_matrix,
        "per_frequency_peak_deg": peaks.tolist(),
        **result.diagnostics,
    }


def select_frequency_evidence(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    num_sources: int,
    *,
    mode: str,
    params: BroadbandParameters,
    theta_grid_deg: np.ndarray,
) -> tuple[list[CovarianceEvidence], dict]:
    if mode == "quality":
        return select_frequency_cluster(
            evidences,
            params=replace(params, use_frequency_mandala=False),
            microphone_positions_m=microphone_positions_m,
            num_sources=num_sources,
            theta_grid_deg=theta_grid_deg,
        )
    if mode == "hard":
        return select_frequency_cluster(
            evidences,
            params=replace(params, use_frequency_mandala=True),
            microphone_positions_m=microphone_positions_m,
            num_sources=num_sources,
            theta_grid_deg=theta_grid_deg,
        )
    if mode == "soft":
        return select_soft_frequency_evidence(
            evidences,
            microphone_positions_m,
            num_sources,
            params=params,
            theta_grid_deg=theta_grid_deg,
        )
    raise ValueError(f"Unknown frequency selection mode: {mode}")


def broadband_music_spectrum(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    theta_grid_deg: np.ndarray,
    num_sources: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(theta_grid_deg, dtype=float)
    spectra = []
    weights = []
    for item in evidences:
        Pn = _noise_projector(item.covariance, num_sources)
        positions = _positions_for_evidence(item, microphone_positions_m)
        A = steering_matrix_geometry(positions, grid, float(item.frequency_hz))
        denom = np.sum(A.conj() * (Pn @ A), axis=0).real
        p = 1.0 / np.maximum(denom, 1e-12)
        p /= max(float(np.max(p)), 1e-12)
        spectra.append(p)
        weights.append(max(float(item.weight), 0.0))
    w = np.asarray(weights, dtype=float)
    if not np.any(w > 0):
        w = np.ones_like(w)
    w /= np.sum(w)
    combined = np.dot(w, np.asarray(spectra))
    combined /= max(float(np.max(combined)), 1e-12)
    return combined, np.asarray(spectra)


def broadband_residual_score(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    theta_deg: Sequence[float],
) -> dict:
    theta = np.sort(np.asarray(theta_deg, dtype=float).reshape(-1))
    traces = []
    weights = []
    for item in evidences:
        R = _hermitian(item.covariance)
        positions = _positions_for_evidence(item, microphone_positions_m)
        A = steering_matrix_geometry(positions, theta, float(item.frequency_hz))
        P = A @ np.linalg.pinv(A.conj().T @ A, rcond=1e-10, hermitian=True) @ A.conj().T
        residual = float(np.trace((np.eye(R.shape[0]) - P) @ R).real / max(np.trace(R).real, 1e-12))
        traces.append(residual)
        weights.append(max(float(item.weight), 0.0))
    w = np.asarray(weights, dtype=float)
    if not np.any(w > 0):
        w = np.ones_like(w)
    w /= np.sum(w)
    traces = np.asarray(traces, dtype=float)
    score = float(np.dot(w, traces))
    return {
        "score": score,
        "per_frequency_trace": traces,
        "frequency_consensus_std": float(np.sqrt(np.dot(w, (traces - score) ** 2))),
        "weights": w,
    }


def broadband_incremental_evidence(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    theta_deg: Sequence[float],
    *,
    positive_tol: float = 1e-3,
) -> dict:
    theta = np.sort(np.asarray(theta_deg, dtype=float).reshape(-1))
    full = broadband_residual_score(evidences, microphone_positions_m, theta)
    increments = []
    per_frequency = []
    for k in range(len(theta)):
        reduced = np.delete(theta, k)
        if len(reduced) == 0:
            reduced_trace = np.ones(len(evidences))
            reduced_score = 1.0
        else:
            result = broadband_residual_score(evidences, microphone_positions_m, reduced)
            reduced_trace = result["per_frequency_trace"]
            reduced_score = result["score"]
        delta_f = reduced_trace - full["per_frequency_trace"]
        per_frequency.append(delta_f)
        increments.append(reduced_score - full["score"])
    per_frequency = np.asarray(per_frequency)
    support = np.mean(per_frequency > float(positive_tol), axis=1)
    return {
        "incremental_evidence": np.asarray(increments),
        "frequency_support": support,
        "per_frequency_increment": per_frequency,
        "full_score": float(full["score"]),
        "frequency_consensus_std": float(full["frequency_consensus_std"]),
    }


def _pair_grid_search(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    grid1: np.ndarray,
    grid2: np.ndarray,
    min_separation_deg: float,
) -> tuple[tuple[float, float], float, int]:
    best_theta = None
    best = float("inf")
    evaluations = 0
    for a in np.asarray(grid1, dtype=float):
        for b in np.asarray(grid2, dtype=float):
            if b - a < float(min_separation_deg) - 1e-12:
                continue
            value = broadband_residual_score(evidences, microphone_positions_m, (a, b))["score"]
            evaluations += 1
            if value < best:
                best = float(value)
                best_theta = (float(a), float(b))
    if best_theta is None:
        raise RuntimeError("No valid broadband angle pair")
    return best_theta, best, evaluations


def estimate_music_from_evidence(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    num_sources: int,
    *,
    route_params: RouteBParameters,
) -> tuple[np.ndarray, dict]:
    grid = _strict_grid(route_params.theta_min, route_params.theta_max, route_params.delta_f)
    spectrum, per_frequency = broadband_music_spectrum(evidences, microphone_positions_m, grid, num_sources)
    order = np.argsort(spectrum)[::-1]
    estimates: list[float] = []
    for idx in order:
        theta = float(grid[int(idx)])
        if all(abs(theta - old) >= route_params.min_separation_deg for old in estimates):
            estimates.append(theta)
        if len(estimates) >= int(num_sources):
            break
    return np.sort(np.asarray(estimates, dtype=float)), {
        "selected_frequencies_hz": [float(item.frequency_hz) for item in evidences],
        "frequency_weights": [float(item.weight) for item in evidences],
        "selected_sensor_indices": [
            None if item.sensor_indices is None else np.asarray(item.sensor_indices).tolist() for item in evidences
        ],
        "spectrum": spectrum,
        "per_frequency_spectrum": per_frequency,
        "grid": grid,
    }


def joint_dml_from_evidence(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    num_sources: int,
    *,
    route_params: RouteBParameters = ROUTE_B_DEFAULTS,
    broadband_params: BroadbandParameters = BROADBAND_DEFAULTS,
) -> dict:
    grid = _strict_grid(route_params.theta_min, route_params.theta_max, broadband_params.coarse_step_deg)
    spectrum, per_frequency_spectrum = broadband_music_spectrum(evidences, microphone_positions_m, grid, num_sources)
    if int(num_sources) == 1:
        coarse = float(grid[int(np.argmax(spectrum))])
        fine_grid = _strict_grid(
            max(route_params.theta_min, coarse - broadband_params.fine_radius_deg),
            min(route_params.theta_max, coarse + broadband_params.fine_radius_deg),
            broadband_params.fine_step_deg,
        )
        scores = [broadband_residual_score(evidences, microphone_positions_m, [x])["score"] for x in fine_grid]
        theta_hat = np.asarray([float(fine_grid[int(np.argmin(scores))])])
        evidence = broadband_incremental_evidence(
            evidences, microphone_positions_m, theta_hat,
            positive_tol=broadband_params.incremental_evidence_tol,
        )
        return {
            "theta_hat": theta_hat,
            "selection_mode": "broadband_single_source_joint_dml",
            "broadband_spectrum_grid_deg": grid,
            "broadband_spectrum": spectrum,
            "source_evidence": evidence,
            "final_residual": float(np.min(scores)),
            "pair_evaluations": int(len(scores)),
        }
    if int(num_sources) != 2:
        raise NotImplementedError("Broadband v4.1 currently supports K=1 or K=2")
    selected_basins = _select_basins(grid, spectrum, 2, route_params)
    intervals = _candidate_intervals(selected_basins, route_params)
    intervals = sorted(intervals, key=lambda x: x[0])[: int(broadband_params.max_candidate_intervals)]
    if not intervals:
        peak = float(grid[int(np.argmax(spectrum))])
        intervals = [(max(route_params.theta_min, peak - broadband_params.candidate_halfwidth_deg),
                      min(route_params.theta_max, peak + broadband_params.candidate_halfwidth_deg))]
    coarse_results = []
    for i, (lo1, hi1) in enumerate(intervals):
        g1 = _strict_grid(lo1, hi1, broadband_params.coarse_step_deg)
        for j in range(i, len(intervals)):
            lo2, hi2 = intervals[j]
            g2 = _strict_grid(lo2, hi2, broadband_params.coarse_step_deg)
            try:
                theta, score, evaluations = _pair_grid_search(
                    evidences, microphone_positions_m, g1, g2, route_params.min_separation_deg
                )
            except RuntimeError:
                continue
            evidence = broadband_incremental_evidence(
                evidences, microphone_positions_m, theta,
                positive_tol=broadband_params.incremental_evidence_tol,
            )
            coarse_results.append({
                "theta": theta,
                "score": score,
                "evaluations": evaluations,
                "basin_ids": (i, j),
                "same_basin": bool(i == j),
                "frequency_support": evidence["frequency_support"],
                "incremental_evidence": evidence["incremental_evidence"],
            })
    if not coarse_results:
        global_grid = _strict_grid(route_params.theta_min, route_params.theta_max, broadband_params.coarse_step_deg)
        theta, score, evaluations = _pair_grid_search(
            evidences, microphone_positions_m, global_grid, global_grid, route_params.min_separation_deg
        )
        coarse_results = [{"theta": theta, "score": score, "evaluations": evaluations,
                           "basin_ids": (-1, -1), "same_basin": False,
                           "frequency_support": np.zeros(2), "incremental_evidence": np.zeros(2)}]
    raw = min(coarse_results, key=lambda row: float(row["score"]))
    near_limit = float(raw["score"]) * 1.03 + 2e-3
    supported = [
        row for row in coarse_results
        if float(row["score"]) <= near_limit
        and float(np.min(row["frequency_support"])) >= broadband_params.minimum_frequency_support
    ]
    selected_coarse = max(
        supported,
        key=lambda row: (float(np.min(row["incremental_evidence"])), -float(row["score"])),
    ) if supported else raw
    coarse_theta = selected_coarse["theta"]
    g1 = _strict_grid(
        max(route_params.theta_min, coarse_theta[0] - broadband_params.fine_radius_deg),
        min(route_params.theta_max, coarse_theta[0] + broadband_params.fine_radius_deg),
        broadband_params.fine_step_deg,
    )
    g2 = _strict_grid(
        max(route_params.theta_min, coarse_theta[1] - broadband_params.fine_radius_deg),
        min(route_params.theta_max, coarse_theta[1] + broadband_params.fine_radius_deg),
        broadband_params.fine_step_deg,
    )
    theta, score, fine_evals = _pair_grid_search(
        evidences, microphone_positions_m, g1, g2, route_params.min_separation_deg
    )
    final_evidence = broadband_incremental_evidence(
        evidences, microphone_positions_m, theta,
        positive_tol=broadband_params.incremental_evidence_tol,
    )
    return {
        "theta_hat": np.asarray(theta, dtype=float),
        "selection_mode": "broadband_joint_dml",
        "confidence": np.clip(final_evidence["frequency_support"], 0.0, 1.0),
        "broadband_spectrum_grid_deg": grid,
        "broadband_spectrum": spectrum,
        "per_frequency_spectrum": per_frequency_spectrum,
        "selected_basins": selected_basins,
        "intervals": intervals,
        "coarse_allocation_results": coarse_results,
        "coarse_selected": selected_coarse,
        "final_residual": score,
        "source_evidence": final_evidence,
        "pair_evaluations": int(sum(row["evaluations"] for row in coarse_results) + fine_evals),
    }


def broadband_mandala_jdml(
    audio: np.ndarray,
    sample_rate_hz: float,
    microphone_positions_m: np.ndarray,
    num_sources: int,
    *,
    route_params: RouteBParameters = ROUTE_B_DEFAULTS,
    broadband_params: BroadbandParameters = BROADBAND_DEFAULTS,
    return_details: bool = False,
):
    """Backward-compatible wrapper; v4.1 uses soft Frequency Mandala."""
    all_evidence, stft_info = extract_stft_covariances(
        audio, sample_rate_hz, microphone_positions_m, params=broadband_params
    )
    grid = _strict_grid(route_params.theta_min, route_params.theta_max, broadband_params.coarse_step_deg)
    selected, frequency_info = select_frequency_evidence(
        all_evidence,
        microphone_positions_m,
        num_sources,
        mode="soft" if broadband_params.soft_frequency_mandala else "hard",
        params=broadband_params,
        theta_grid_deg=grid,
    )
    details = joint_dml_from_evidence(
        selected,
        microphone_positions_m,
        num_sources,
        route_params=route_params,
        broadband_params=broadband_params,
    )
    details.update({
        "algorithm_version": ALGORITHM_VERSION_BROADBAND,
        "selected_frequencies_hz": [float(x.frequency_hz) for x in selected],
        "frequency_weights": [float(x.weight) for x in selected],
        "stft_info": stft_info,
        "frequency_mandala": frequency_info,
    })
    return details if return_details else np.asarray(details["theta_hat"], dtype=float)


def srp_phat_spectrum(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    theta_grid_deg: np.ndarray,
) -> np.ndarray:
    """Frequency-weighted SRP-PHAT spectrum for arbitrary subarrays."""
    grid = np.asarray(theta_grid_deg, dtype=float)
    spectra = []
    weights = []
    for item in evidences:
        R = np.asarray(item.covariance, dtype=complex)
        phat = R / np.maximum(np.abs(R), 1e-12)
        np.fill_diagonal(phat, 0.0)
        positions = _positions_for_evidence(item, microphone_positions_m)
        A = steering_matrix_geometry(positions, grid, float(item.frequency_hz))
        score = np.real(np.sum(A.conj() * (phat @ A), axis=0))
        score -= np.min(score)
        score /= max(float(np.max(score)), 1e-12)
        spectra.append(score)
        weights.append(max(float(item.weight), 0.0))
    w = np.asarray(weights, dtype=float)
    if not np.any(w > 0):
        w = np.ones_like(w)
    w /= np.sum(w)
    combined = np.dot(w, np.asarray(spectra))
    combined -= np.min(combined)
    combined /= max(float(np.max(combined)), 1e-12)
    return combined


def srp_phat_estimate_from_evidence(
    evidences: Sequence[CovarianceEvidence],
    microphone_positions_m: np.ndarray,
    num_sources: int,
    *,
    route_params: RouteBParameters,
) -> tuple[np.ndarray, dict]:
    grid = _strict_grid(route_params.theta_min, route_params.theta_max, route_params.delta_f)
    spectrum = srp_phat_spectrum(evidences, microphone_positions_m, grid)
    order = np.argsort(spectrum)[::-1]
    estimates = []
    for idx in order:
        theta = float(grid[int(idx)])
        if all(abs(theta - old) >= route_params.min_separation_deg for old in estimates):
            estimates.append(theta)
        if len(estimates) >= int(num_sources):
            break
    return np.sort(np.asarray(estimates, dtype=float)), {
        "grid": grid,
        "spectrum": spectrum,
        "selected_frequencies_hz": [float(item.frequency_hz) for item in evidences],
    }


def front_back_ambiguity_score(
    positions_m: np.ndarray,
    frequency_hz: float,
    *,
    theta_grid_deg: np.ndarray | None = None,
) -> dict:
    """Measure ambiguity between directions separated by 180 degrees."""
    grid = np.asarray(theta_grid_deg if theta_grid_deg is not None else np.arange(-175.0, 180.0, 5.0), dtype=float)
    # For a linear azimuth array with steering proportional to sin(theta),
    # the front/back mirror is 180-theta, not theta+180.
    opposite = (180.0 - grid + 180.0) % 360.0 - 180.0
    a = steering_matrix_geometry(positions_m, grid, frequency_hz)
    b = steering_matrix_geometry(positions_m, opposite, frequency_hz)
    a /= np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-12)
    values = np.abs(np.sum(a.conj() * b, axis=0))
    return {
        "front_back_coherence_mean": float(np.mean(values)),
        "front_back_coherence_median": float(np.median(values)),
        "front_back_coherence_max": float(np.max(values)),
        "front_back_identifiable_fraction_0p95": float(np.mean(values < 0.95)),
    }
