# =========================================================
# File        : build_r8_final_eigenmike_manifest.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Build r8 final eigenmike manifest.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - Derived CSV/JSON evidence or evaluation summaries as defined by the CLI entry point.
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

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sgf_amusic_route_b.active_speech_gate import (
    ActiveSpeechGateParameters,
    activity_window_stats,
    band_energy_cumulative_from_audio,
    build_activity_index,
    estimate_full_recording_inactive_noise_floor_dbfs,
    qualify_active_speech_window,
    resolve_shared_recording_audio,
    window_band_rms,
)
from sgf_amusic_route_b.b0_r2_protocol import geometry_reference_audit, preregistered_window_gate
from sgf_amusic_route_b.final_eigenmike_protocol import (
    ARRAY_NAME,
    MAXIMUM_OTHER_SOURCE_ACTIVITY_FRACTION,
    MAX_SELECTED_PER_RECORDING,
    MIN_SELECTED_PER_RECORDING,
    PROTOCOL_ID,
    RELEASE,
    REQUIRED_TASKS,
    THETA_MAX_DEG,
    THETA_MIN_DEG,
    WINDOW_DURATION_S,
    canonical_manifest_sha256,
    classify_preflight_outcome,
    coverage_gate,
    deterministic_time_spread_subset,
    select_fixed_eigenmike_subarray,
    same_channel_set,
    subarray_preflight_pass,
    verify_frozen_r8_core,
)
from sgf_amusic_route_b.locata_geometry import (
    audit_locata_root,
    infer_locata_task,
    load_broadband_windows,
    load_recording_geometry,
)

PREFIX = "R8_FINAL_EIGENMIKE"


def _activity_bounds(frame: dict) -> tuple[float, float]:
    duration = float(frame["audio"].shape[1]) / float(frame["sample_rate_hz"])
    return float(frame["timestamp_abs_s"]) - duration, float(frame["timestamp_abs_s"])


def _recover_full_audio(windows: list[dict], geometry: dict) -> tuple[np.ndarray, str]:
    try:
        return resolve_shared_recording_audio(windows), "shared_numpy_view"
    except RuntimeError:
        from scipy.io import wavfile
        from scipy.io.wavfile import WavFileWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WavFileWarning)
            fs_raw, raw = wavfile.read(str(geometry["audio_path"]))
        if int(fs_raw) != int(windows[0]["sample_rate_hz"]):
            raise RuntimeError("Audio sample-rate mismatch during band-energy fallback")
        raw = raw[: int(geometry["usable_audio_samples"])]
        audio = raw.astype(float)
        if np.issubdtype(raw.dtype, np.integer):
            audio /= max(float(np.iinfo(raw.dtype).max), 1.0)
        return audio.T, "audited_audio_file_fallback"


def _eligible_domain(
    *,
    speech_eligible: bool,
    speech_reason: str,
    maximum_other_source_fraction: float,
    true_azimuth_deg: float,
) -> tuple[bool, str, bool, bool]:
    other_pass = bool(
        np.isfinite(maximum_other_source_fraction)
        and float(maximum_other_source_fraction)
        <= MAXIMUM_OTHER_SOURCE_ACTIVITY_FRACTION + 1.0e-12
    )
    truth_pass = bool(
        np.isfinite(true_azimuth_deg)
        and THETA_MIN_DEG - 1.0e-12
        <= float(true_azimuth_deg)
        <= THETA_MAX_DEG + 1.0e-12
    )
    reasons: list[str] = []
    if not speech_eligible:
        reasons.append(str(speech_reason) or "active_speech_gate_failed")
    if not other_pass:
        reasons.append("other_source_activity_exceeds_50ms")
    if not truth_pass:
        reasons.append("truth_outside_frozen_scan_domain")
    return bool(speech_eligible and other_pass and truth_pass), (
        "PASS" if not reasons else ";".join(reasons)
    ), other_pass, truth_pass


def prepare_recording(
    recording: Path,
    *,
    array_name: str,
    gate_params: ActiveSpeechGateParameters,
    expected_subarray_indices: tuple[int, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, tuple[int, ...], dict]:
    task = infer_locata_task(recording)
    geometry = load_recording_geometry(recording, array_name=array_name)
    windows, _ = load_broadband_windows(
        recording,
        array_name=array_name,
        window_duration_s=WINDOW_DURATION_S,
        max_windows=0,
        window_anchor_mode="required_time_causal",
        activity_aware_max_windows=False,
    )
    if not windows:
        raise RuntimeError(f"No required-time windows for {recording}")
    geo_audit = geometry_reference_audit(windows[0])
    if geo_audit["status"] != "PASS":
        raise RuntimeError(f"Geometry audit failed: {geo_audit}")

    raw_positions = np.asarray(geometry["microphone_positions_local_m"], dtype=float)
    selection = select_fixed_eigenmike_subarray(raw_positions)
    subarray_pass, subarray_gates = subarray_preflight_pass(selection)
    geometry_selected_indices = tuple(selection.selected_indices)
    if expected_subarray_indices is not None:
        if not same_channel_set(geometry_selected_indices, expected_subarray_indices):
            raise RuntimeError(
                "Geometry-only Eigenmike physical channel set changed across recordings: "
                f"expected {expected_subarray_indices}, got {geometry_selected_indices}"
            )
        # Normalize tied Hungarian sector-order permutations to the first frozen
        # recording order.  This changes no physical channel and uses no audio,
        # truth, error, or R8 output.
        selected_indices = tuple(expected_subarray_indices)
    else:
        selected_indices = geometry_selected_indices
    if not subarray_pass:
        raise RuntimeError(f"Frozen Eigenmike subarray preflight failed: {subarray_gates}")

    activity_index = build_activity_index(geometry)
    if not activity_index or str(geometry.get("source_activity_source", "")) != "official_array_aligned_vad":
        raise RuntimeError("Final validation requires official array-aligned Eigenmike VAD")

    full_audio, recovery_mode = _recover_full_audio(windows, geometry)
    full_audio = np.asarray(full_audio, dtype=float)[np.asarray(selected_indices, dtype=int)]
    band_cumulative = band_energy_cumulative_from_audio(
        full_audio,
        windows[0]["sample_rate_hz"],
        frequency_min_hz=gate_params.localization_band_min_hz,
        frequency_max_hz=gate_params.localization_band_max_hz,
        filter_order=gate_params.filter_order,
    )
    noise_floor, noise_audit, _ = estimate_full_recording_inactive_noise_floor_dbfs(
        band_cumulative,
        geometry["audio_t_abs_sec"],
        activity_index,
        windows[0]["sample_rate_hz"],
        window_duration_s=WINDOW_DURATION_S,
        hop_duration_s=gate_params.noise_grid_hop_s,
        inactive_fraction_max=gate_params.inactive_fraction_max,
        minimum_noise_windows=gate_params.minimum_noise_windows,
        lower_envelope_fraction=gate_params.lower_envelope_fraction,
    )

    qualification_rows: list[dict] = []
    eligible_indices: list[int] = []
    timestamps = [float(frame["timestamp_abs_s"]) for frame in windows]
    required_rows = [int(frame["required_time_row_index"]) for frame in windows]

    for index, frame in enumerate(windows):
        gate_frame = dict(frame)
        gate_frame["audio"] = np.asarray(frame["audio"], dtype=float)[
            np.asarray(selected_indices, dtype=int)
        ]
        center = preregistered_window_gate(gate_frame, hard_silence_floor_dbfs=-80.0)
        start_abs, end_abs = _activity_bounds(frame)
        stats = activity_window_stats(
            activity_index,
            start_abs,
            end_abs,
            center.active_source_index,
        )
        band_rms = window_band_rms(
            band_cumulative,
            int(frame["audio_window_start_sample"]),
            int(frame["audio_window_end_exclusive_sample"]),
        )
        speech = qualify_active_speech_window(
            center_official_exactly_one_active=bool(
                center.official_activity_available and center.exactly_one_source_active
            ),
            active_source_index=center.active_source_index,
            activity_stats=stats,
            band_rms=band_rms,
            recording_noise_floor_dbfs=noise_floor,
            window_duration_s=WINDOW_DURATION_S,
            relative_gate_required=bool(noise_audit["physical_noise_floor_claimed"]),
            params=gate_params,
        )
        source_index = int(center.active_source_index) if center.active_source_index is not None else -1
        truths = np.asarray(frame["theta_true_deg"], dtype=float).reshape(-1)
        elevations = np.asarray(frame.get("elevation_true_deg", []), dtype=float).reshape(-1)
        true_azimuth = (
            float(truths[source_index])
            if 0 <= source_index < len(truths) and np.isfinite(truths[source_index])
            else float("nan")
        )
        true_elevation = (
            float(elevations[source_index])
            if 0 <= source_index < len(elevations) and np.isfinite(elevations[source_index])
            else float("nan")
        )
        eligible, rejection, other_pass, truth_pass = _eligible_domain(
            speech_eligible=bool(speech.eligible),
            speech_reason=str(speech.rejection_reason),
            maximum_other_source_fraction=float(stats.maximum_other_source_fraction),
            true_azimuth_deg=true_azimuth,
        )
        if eligible:
            eligible_indices.append(index)
        qualification_rows.append({
            "release": RELEASE,
            "protocol_id": PROTOCOL_ID,
            "task": task,
            "recording": str(recording),
            "recording_name": recording.name,
            "required_time_row_index": int(frame["required_time_row_index"]),
            "timestamp_abs_s": float(frame["timestamp_abs_s"]),
            "active_source_index": source_index,
            "true_azimuth_deg_for_domain_check_only": true_azimuth,
            "true_elevation_deg_diagnostic_only": true_elevation,
            "center_exactly_one_source_active": bool(center.exactly_one_source_active),
            "target_activity_duration_s": float(speech.target_activity_duration_s),
            "target_activity_fraction_full_window": float(stats.target_fraction),
            "maximum_other_source_activity_fraction": float(stats.maximum_other_source_fraction),
            "other_source_activity_pass": bool(other_pass),
            "truth_in_frozen_scan_domain": bool(truth_pass),
            "band_rms_dbfs": float(speech.band_rms_dbfs),
            "recording_band_reference_dbfs": float(noise_floor),
            "band_reference_kind": str(noise_audit["reference_kind"]),
            "physical_noise_floor_claimed": bool(noise_audit["physical_noise_floor_claimed"]),
            "band_level_margin_db": float(speech.band_snr_db),
            "strict_50pct_pass": bool(speech.strict_50pct_pass),
            "sensitivity_80pct_pass": bool(speech.sensitivity_80pct_pass),
            "active_speech_eligible": bool(speech.eligible),
            "eligible_before_sampling": bool(eligible),
            "rejection_reason": str(rejection),
            "truth_used_for_estimation": False,
            "error_used_for_selection": False,
            "r8_output_used_for_selection": False,
        })

    selected_indices_in_windows, selection_summary = deterministic_time_spread_subset(
        eligible_indices,
        timestamps,
        required_rows,
        maximum_count=MAX_SELECTED_PER_RECORDING,
    )
    selected_set = set(selected_indices_in_windows)
    ranks = {index: rank for rank, index in enumerate(selected_indices_in_windows)}
    for index, row in enumerate(qualification_rows):
        row["selected_for_validation"] = bool(index in selected_set)
        row["selection_rank"] = int(ranks.get(index, -1))
        row["selection_rule"] = (
            "uniform_time_spread_among_preregistered_eligible_anchors"
            if index in selected_set else ""
        )

    manifest_rows = []
    for index in selected_indices_in_windows:
        row = qualification_rows[index]
        manifest_rows.append({
            "release": RELEASE,
            "protocol_id": PROTOCOL_ID,
            "source_protocol": "final_frozen_eigenmike_evaluation",
            "development_status": "independent_locata_evaluation_cross_array",
            "task": row["task"],
            "recording": row["recording"],
            "recording_name": row["recording_name"],
            "required_time_row_index": row["required_time_row_index"],
            "timestamp_abs_s": row["timestamp_abs_s"],
            "active_source_index": row["active_source_index"],
            "target_activity_duration_s": row["target_activity_duration_s"],
            "target_activity_fraction_full_window": row["target_activity_fraction_full_window"],
            "maximum_other_source_activity_fraction": row["maximum_other_source_activity_fraction"],
            "strict_50pct_pass": row["strict_50pct_pass"],
            "sensitivity_80pct_pass": row["sensitivity_80pct_pass"],
            "selection_rank": row["selection_rank"],
            "selection_rule": row["selection_rule"],
            "truth_used_for_estimation": False,
            "error_used_for_selection": False,
            "r8_output_used_for_selection": False,
            "window_independence_claimed": False,
        })

    sufficient = len(selected_indices_in_windows) >= MIN_SELECTED_PER_RECORDING
    status = {
        "release": RELEASE,
        "protocol_id": PROTOCOL_ID,
        "task": task,
        "recording": str(recording),
        "recording_name": recording.name,
        "status": "completed" if sufficient else "insufficient_preregistered_eligible_windows",
        "required_time_windows_scanned": int(len(windows)),
        "eligible_windows_before_sampling": int(len(eligible_indices)),
        "selected_windows": int(len(selected_indices_in_windows)),
        "minimum_selected_required": int(MIN_SELECTED_PER_RECORDING),
        "coverage_sufficient": bool(sufficient),
        "selection_summary": selection_summary,
        "noise_floor_audit": noise_audit,
        "band_energy_audio_recovery_mode": recovery_mode,
        "geometry_reference_audit": geo_audit,
        "official_vad_required": True,
        "single_active_anchor_required": True,
        "maximum_other_source_activity_fraction": MAXIMUM_OTHER_SOURCE_ACTIVITY_FRACTION,
        "frozen_scan_domain_deg": [THETA_MIN_DEG, THETA_MAX_DEG],
        "error_based_selection": False,
        "r8_output_based_selection": False,
    }
    selection_status = {
        "selection": selection.as_dict(),
        "geometry_solver_order_indices": list(geometry_selected_indices),
        "frozen_applied_order_indices": list(selected_indices),
        "order_normalized_only": bool(geometry_selected_indices != selected_indices),
        "physical_channel_set_unchanged": bool(
            expected_subarray_indices is None
            or same_channel_set(geometry_selected_indices, expected_subarray_indices)
        ),
        "gates": subarray_gates,
        "status": "PASS" if subarray_pass else "FAIL",
    }
    del windows, full_audio, band_cumulative
    gc.collect()
    return (
        pd.DataFrame(manifest_rows),
        pd.DataFrame(qualification_rows),
        status,
        selected_indices,
        selection_status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and freeze the final independent Eigenmike evaluation manifest"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    root_text = str(args.root.resolve()).replace("\\", "/").lower()
    if "eval" not in root_text:
        raise SystemExit(
            "LOCATA_EVAL_ROOT must point to the final LOCATA evaluation split; "
            "the path must contain 'eval' to prevent accidental development-set reuse."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    final_decision = args.project_root / "R8_FINAL_TASLP_DECISION.json"
    if final_decision.exists():
        raise SystemExit("Final decision already exists. Rerunning after result disclosure is forbidden.")

    core_audit = verify_frozen_r8_core(args.project_root)
    if core_audit["status"] != "PASS":
        raise SystemExit(f"Frozen R8 core hash mismatch: {core_audit}")

    audit_dir = args.output / "geometry_audit"
    audit, audit_status = audit_locata_root(args.root, audit_dir, array_name=ARRAY_NAME)
    if audit.empty:
        raise SystemExit("No Eigenmike recordings were discovered under LOCATA_EVAL_ROOT")
    usable = [Path(item) for item in audit.loc[audit["usable"].astype(bool), "recording"].tolist()]
    usable = [item for item in usable if infer_locata_task(item) in set(REQUIRED_TASKS)]
    if not usable:
        raise SystemExit("No usable Task 1-4 Eigenmike evaluation recordings were discovered")

    gate_params = ActiveSpeechGateParameters()
    all_manifest = []
    all_qualification = []
    statuses = []
    subarray_rows = []
    expected_indices: tuple[int, ...] | None = None
    errors = []
    for recording in sorted(usable, key=lambda p: (infer_locata_task(p), p.name, str(p))):
        try:
            manifest, qualification, status, selected_indices, selection_status = prepare_recording(
                recording,
                array_name=ARRAY_NAME,
                gate_params=gate_params,
                expected_subarray_indices=expected_indices,
            )
            if expected_indices is None:
                expected_indices = selected_indices
            all_manifest.append(manifest)
            all_qualification.append(qualification)
            statuses.append(status)
            subarray_rows.append({
                "task": infer_locata_task(recording),
                "recording_name": recording.name,
                "recording": str(recording),
                **selection_status,
            })
            print(
                f"[preflight {infer_locata_task(recording)}/{recording.name}] "
                f"eligible={status['eligible_windows_before_sampling']} "
                f"selected={status['selected_windows']} sufficient={status['coverage_sufficient']}",
                flush=True,
            )
        except Exception as exc:
            errors.append({
                "task": infer_locata_task(recording),
                "recording_name": recording.name,
                "recording": str(recording),
                "error": f"{type(exc).__name__}: {exc}",
            })

    manifest = pd.concat(all_manifest, ignore_index=True) if all_manifest else pd.DataFrame()
    qualification = pd.concat(all_qualification, ignore_index=True) if all_qualification else pd.DataFrame()
    status_table = pd.DataFrame(statuses)
    if len(manifest):
        sufficient_keys = {
            (str(row["task"]), str(row["recording_name"]))
            for row in statuses if bool(row.get("coverage_sufficient", False))
        }
        manifest = manifest[
            manifest.apply(
                lambda r: (str(r["task"]), str(r["recording_name"])) in sufficient_keys,
                axis=1,
            )
        ].copy()
        manifest = manifest.sort_values(
            ["task", "recording_name", "selection_rank", "required_time_row_index"]
        ).reset_index(drop=True)
    total_selected = int(len(manifest))
    coverage_pass, coverage_details = coverage_gate(statuses, total_selected)
    manifest_sha = canonical_manifest_sha256(manifest.to_dict("records")) if len(manifest) else "EMPTY"

    manifest_path = args.output / f"{PREFIX}_FROZEN_MANIFEST.csv"
    qualification_path = args.output / f"{PREFIX}_QUALIFICATION_LEDGER.csv"
    recording_status_path = args.output / f"{PREFIX}_RECORDING_STATUS.csv"
    manifest.to_csv(manifest_path, index=False)
    qualification.to_csv(qualification_path, index=False)
    status_table.to_csv(recording_status_path, index=False)
    (args.output / f"{PREFIX}_SUBARRAY_AUDIT.json").write_text(
        json.dumps(subarray_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    preflight_status, exit_code, write_scientific_final = classify_preflight_outcome(
        coverage_pass=coverage_pass,
        recording_error_count=len(errors),
    )
    lock_status = (
        "ENGINEERING_ERROR" if errors else ("FROZEN" if coverage_pass else "NO_GO")
    )
    lock = {
        "release": RELEASE,
        "protocol_id": PROTOCOL_ID,
        "status": lock_status,
        "manifest": str(manifest_path.resolve()),
        "manifest_canonical_sha256": manifest_sha,
        "manifest_rows": total_selected,
        "recording_count": int(len({(r["task"], r["recording_name"]) for r in manifest.to_dict("records")})),
        "array_name": ARRAY_NAME,
        "selected_channel_indices_zero_based": list(expected_indices or []),
        "frozen_r8_core_audit": core_audit,
        "coverage": coverage_details,
        "audit_status": audit_status,
        "recording_errors": errors,
        "truth_used_for_estimation": False,
        "truth_used_only_for_declared_scan_domain_eligibility": True,
        "error_used_for_selection": False,
        "r8_output_used_for_selection": False,
        "manual_recording_exclusion_permitted": False,
        "manual_window_exclusion_permitted": False,
    }
    preflight = {**lock, "status": preflight_status}
    preflight_path = args.output / f"{PREFIX}_PREFLIGHT_DECISION.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.project_root / "R8_FINAL_PREFLIGHT_STATUS.txt").write_text(preflight_status + "\n", encoding="utf-8")

    if errors:
        # Engineering failures are auditable but do not freeze a manifest and
        # never create the binary TASLP decision.  Absence of the lock allows a
        # corrected clean package to repeat Step 1 without scientific rerunning.
        engineering_path = args.output / f"{PREFIX}_ENGINEERING_ERROR.json"
        engineering_path.write_text(
            json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return exit_code

    lock_path = args.output / f"{PREFIX}_MANIFEST_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.project_root / "R8_FINAL_MANIFEST_LOCK.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if write_scientific_final:
        final = {
            **preflight,
            "status": "FINAL_TASLP_VALIDATION_PREFLIGHT_NO_GO_ABANDON_TASLP",
            "decision_rule": "completed_error_free_preflight_coverage_gate_failed",
            "action": "abandon_TASLP_route_and_stop_algorithm_development",
            "rerun_or_retuning_permitted": False,
        }
        final_text = json.dumps(final, indent=2, ensure_ascii=False) + "\n"
        (args.project_root / "R8_FINAL_TASLP_DECISION.json").write_text(
            final_text, encoding="utf-8"
        )
        (args.project_root / "R8_FINAL_TASLP_STATUS.txt").write_text(
            final["status"] + "\n", encoding="utf-8"
        )
        (args.project_root / "R8_FINAL_TASLP_REPORT.md").write_text(
            "# R8 Final Frozen Eigenmike Validation\n\n"
            f"Final status: `{final['status']}`\n\n"
            "The complete, error-free cross-array preflight did not pass the preregistered data-coverage gate. "
            "In accordance with the preregistered decision rule, this route is terminated and no further algorithm development is performed.\n",
            encoding="utf-8",
        )
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    return exit_code



if __name__ == "__main__":
    raise SystemExit(main())
