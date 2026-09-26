# =========================================================
# File        : run_r8m_final_multimethod_validation.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Run r8m final multimethod validation.
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
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sgf_amusic_route_b.baseline_extension import _circular_difference_deg
from sgf_amusic_route_b.baseline_forensic import BaselineForensicParameters
from sgf_amusic_route_b.causal_multiscale_hodge import CausalMultiscaleHodgeParameters, solve_multiscale_state_path
from sgf_amusic_route_b.locata_geometry import audit_locata_root, infer_locata_task, load_broadband_windows
from sgf_amusic_route_b.mechanism_corrected_r8 import prepare_r8m_state_banks
from sgf_amusic_route_b.mechanism_corrected_r8_multimethod_evidence_change import (
    ALL_VALIDATION_VARIANTS,
    FINAL_CANDIDATE_VARIANTS,
    MultiMethodParameters,
    multimethod_parameter_grid,
    rescore_prepared_r8m_multimethod_banks,
)

RELEASE = "r8m-final-multimethod-evidence-change-validation-20260810"
PROTOCOL_ID = "DOA-R8M-FINAL-MULTIMETHOD-EC-20260810"
PREFIX = "R8MMV"
ARRAYS = {
    "eigenmike": {"array_id": "eigenmike", "manifest_dir": "eigenmike"},
    "robothead": {"array_id": "benchmark2", "manifest_dir": "robothead"},
    "dicit": {"array_id": "dicit", "manifest_dir": "dicit"},
}


def canonical_manifest_sha256(frame: pd.DataFrame) -> str:
    rows = [{
        "task": str(r.task),
        "recording_name": str(r.recording_name),
        "required_time_row_index": int(r.required_time_row_index),
        "active_source_index": int(r.active_source_index),
    } for r in frame.itertuples()]
    rows.sort(key=lambda x: (x["task"], x["recording_name"], x["required_time_row_index"], x["active_source_index"]))
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _channels_from_lock(lock: dict) -> np.ndarray:
    values = lock.get("selected_channel_indices_zero_based", lock.get("native_channel_indices_zero_based"))
    if values is None:
        raise RuntimeError("Manifest lock does not contain frozen channel indices")
    return np.asarray([int(x) for x in values], dtype=int)


def _truth_for_frame(frame: dict, source_index: int) -> float:
    truth = np.asarray(frame["theta_true_deg"], dtype=float).reshape(-1)
    if source_index < 0 or source_index >= len(truth) or not np.isfinite(truth[source_index]):
        raise RuntimeError("Frozen manifest points to invalid source truth")
    return float(truth[source_index])


def _original_r8_from_banks(banks: list[dict], params: CausalMultiscaleHodgeParameters) -> dict:
    solved = solve_multiscale_state_path(banks, params=params)
    e0 = float(banks[-1]["e0_estimate_deg"])
    best = solved["best_nonbaseline_final_path"]
    raw = float(best["states"][-1]["theta_deg"]) if best is not None else e0
    selective = float(raw if solved["adopt_joint_path"] else e0)
    return {"e0": e0, "raw": raw, "selective": selective, "adopt": bool(solved["adopt_joint_path"]), "gain": float(solved["model_selection_gain"])}


def _phase_task_allowed(task: str, phase: str) -> bool:
    return task == "task1" if phase == "task1" else task in {"task2", "task3", "task4"}


def _load_variants_for_phase(phase: str, decision_path: Path | None) -> tuple[str, ...]:
    if phase == "task1":
        return tuple(ALL_VALIDATION_VARIANTS)
    if decision_path is None or not decision_path.is_file():
        raise RuntimeError("Remaining phase requires Task-1 decision JSON")
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    if not bool(payload.get("advance_to_full", False)):
        return tuple()
    # For unbiased LOAO and complete method comparison, once any final method
    # passes Task 1 we run all predeclared methods/strengths on remaining tasks.
    return tuple(ALL_VALIDATION_VARIANTS)


def evaluate_recording(
    recording: Path,
    manifest: pd.DataFrame,
    *, phase: str,
    array_label: str,
    array_id: str,
    channels: np.ndarray,
    variants: tuple[str, ...],
    grid: list[MultiMethodParameters],
    r8_params: CausalMultiscaleHodgeParameters,
    forensic_params: BaselineForensicParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    windows, _ = load_broadband_windows(
        recording,
        array_name=array_id,
        window_duration_s=max(r8_params.scales_s),
        max_windows=0,
        window_anchor_mode="required_time_causal",
        activity_aware_max_windows=False,
    )
    if not windows:
        raise RuntimeError(f"No windows in {recording}")
    first_channels = int(np.asarray(windows[0]["audio"]).shape[0])
    if np.any(channels < 0) or np.any(channels >= first_channels):
        raise RuntimeError(f"Frozen channel indices invalid for {recording}: M={first_channels}, channels={channels.tolist()}")

    window_map = {int(frame["required_time_row_index"]): frame for frame in windows}
    ordered = manifest.sort_values("selection_rank").copy()
    requested = ordered["required_time_row_index"].astype(int).tolist()
    missing = [row for row in requested if row not in window_map]
    if missing:
        raise RuntimeError(f"Frozen manifest rows missing from {recording}: {missing}")
    qmap = ordered.set_index("required_time_row_index")

    candidate_rows, diagnostic_rows = [], []
    for position, row_index in enumerate(requested, start=1):
        frame = window_map[row_index]
        q = qmap.loc[row_index]
        source_index = int(q["active_source_index"])
        truth = _truth_for_frame(frame, source_index)
        audio = np.asarray(frame["audio"], dtype=float)[channels]
        positions = np.asarray(frame["microphone_positions_m"], dtype=float)[channels]
        started = time.perf_counter()
        banks = prepare_r8m_state_banks(
            audio, frame["sample_rate_hz"], positions,
            r8_params=r8_params, forensic_params=forensic_params,
        )
        original = _original_r8_from_banks(banks, r8_params)
        common = {
            "release": RELEASE, "protocol_id": PROTOCOL_ID, "phase": phase,
            "development_kind": "final_multimethod_evidence_change_seen_arrays",
            "array_label": array_label, "array_id": array_id, "task": str(q["task"]),
            "recording_name": recording.name, "required_time_row_index": int(row_index),
            "selection_rank": int(q["selection_rank"]), "active_source_index": source_index,
            "true_azimuth_deg": truth,
            "task_identity_used_for_estimation": False, "truth_used_for_estimation": False,
            "array_label_used_for_estimation": False, "future_context_used": False,
        }
        long_diag = None
        for variant in variants:
            for p in grid:
                result = rescore_prepared_r8m_multimethod_banks(
                    banks, structural_variant=variant, r8m_params=p, r8_params=r8_params,
                )
                estimate = float(result["estimate_deg"])
                signed = float(_circular_difference_deg(estimate, truth))
                candidate_rows.append({
                    **common,
                    "coupling_variant": variant,
                    "eligible_final_method": bool(variant in FINAL_CANDIDATE_VARIANTS),
                    "assignment_strength": float(p.base_assignment_strength),
                    "anchor_strength": float(p.anchor_strength),
                    "parameter_id": f"a{p.base_assignment_strength:g}_h{p.anchor_strength:g}",
                    "candidate_id": f"{variant}__a{p.base_assignment_strength:g}_h{p.anchor_strength:g}",
                    "e0_estimate_deg": float(original["e0"]),
                    "original_r8_estimate_deg": float(original["selective"]),
                    "original_r8_adopt": bool(original["adopt"]),
                    "estimated_azimuth_deg": estimate,
                    "signed_error_deg": signed,
                    "absolute_error_deg": abs(signed),
                    "squared_error_deg2": signed * signed,
                    "r8m_adopt": bool(result["adopt_joint_path"]),
                    "r8m_pre_guard_adopt": bool(result["pre_guard_adopt_joint_path"]),
                    "r8m_no_reassignment_guard_triggered": bool(result["no_reassignment_guard_triggered"]),
                    "r8m_replacement_allowed_by_reassignment": bool(result["replacement_allowed_by_reassignment"]),
                    "r8m_pre_guard_long_reassigned_pair_count": int(result["pre_guard_long_reassigned_pair_count"]),
                    "r8m_selected_long_reassigned_pair_count": int(result["selected_long_reassigned_pair_count"]),
                    "r8m_model_selection_gain": float(result["model_selection_gain"]),
                    "r8m_best_path_margin": float(result["best_path_margin"]),
                    "r8m_best_nonbaseline_support_count": int(result["best_nonbaseline_support_count"]),
                    "r8m_best_nonbaseline_direction_spread_deg": float(result["best_nonbaseline_direction_spread_deg"]),
                    "r8m_selected_theta_path_json": json.dumps(result["selected_theta_path_deg"]),
                    "r8m_pre_guard_theta_path_json": json.dumps(result["pre_guard_selected_theta_path_deg"]),
                    "r8mmv_pre_guard_long_geometry_assignment_description_per_edge": float(result["pre_guard_long_geometry_assignment_description_per_edge"]),
                    "r8mmv_pre_guard_long_peak_evidence_debt_mean_changed": float(result["pre_guard_long_peak_evidence_debt_mean_changed"]),
                    "r8mmv_selected_long_geometry_assignment_description_per_edge": float(result["selected_long_geometry_assignment_description_per_edge"]),
                    "r8mmv_selected_long_peak_evidence_debt_mean_changed": float(result["selected_long_peak_evidence_debt_mean_changed"]),
                    "r8mmv_selected_long_reliability_factor": float(result["selected_long_reliability_factor"]),
                    "r8mmv_pre_guard_long_reliability_factor": float(result["pre_guard_long_reliability_factor"]),
                    "r8mmv_selected_long_assignment_extra_cost": float(result["selected_long_assignment_extra_cost"]),
                    "r8mmv_selected_long_reliability_prior_extra_cost": float(result["selected_long_reliability_prior_extra_cost"]),
                })
                if long_diag is None:
                    long_diag = result["scale_diagnostics"][-1]
        e0_signed = float(_circular_difference_deg(float(original["e0"]), truth))
        r8_signed = float(_circular_difference_deg(float(original["selective"]), truth))
        diagnostic_rows.append({
            **common,
            "e0_estimate_deg": float(original["e0"]), "e0_absolute_error_deg": abs(e0_signed),
            "original_r8_estimate_deg": float(original["selective"]), "original_r8_absolute_error_deg": abs(r8_signed),
            "original_r8_adopt": bool(original["adopt"]), "original_r8_model_selection_gain": float(original["gain"]),
            "long_scale_e0_uncertainty_ratio": float(long_diag["e0_uncertainty_ratio"]),
            "long_scale_e0_anchor_reliability": float(long_diag["e0_anchor_reliability"]),
            "median_pair_baseline_m": float(long_diag["median_pair_baseline_m"]),
            "maximum_pair_baseline_m": float(long_diag["maximum_pair_baseline_m"]),
        })
        elapsed = time.perf_counter() - started
        print(
            f"[{phase} {array_label} {q['task']}/{recording.name}] {position}/{len(requested)} "
            f"row={row_index} E0={original['e0']:.2f} R8adopt={original['adopt']} "
            f"u={long_diag['e0_uncertainty_ratio']:.3f} methods={len(variants)}x{len(grid)} time={elapsed:.2f}s",
            flush=True,
        )

    status = {
        "release": RELEASE, "protocol_id": PROTOCOL_ID, "phase": phase,
        "array_label": array_label, "array_id": array_id, "task": str(ordered.iloc[0]["task"]),
        "recording_name": recording.name, "recording": str(recording), "status": "completed",
        "windows": int(len(requested)), "coupling_variants": list(variants),
        "parameter_count_per_variant": int(len(grid)),
        "candidate_count_per_window": int(len(variants) * len(grid)),
        "frozen_channel_indices_zero_based": channels.tolist(),
        "task_identity_used_for_estimation": False, "truth_used_for_estimation": False,
    }
    return pd.DataFrame(candidate_rows), pd.DataFrame(diagnostic_rows), status


def main() -> int:
    p = argparse.ArgumentParser(description="Run final multi-method Evidence-Change validation")
    p.add_argument("--phase", choices=("task1", "remaining"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task1-decision", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    variants = _load_variants_for_phase(args.phase, args.task1_decision)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "remaining" and not variants:
        payload = {"release": RELEASE, "protocol_id": PROTOCOL_ID, "phase": args.phase, "status": "skipped_no_final_method_passed_task1", "total_unique_windows": 0, "development_only": True}
        (args.output / f"{PREFIX}_REMAINING_EXECUTION_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    grid = multimethod_parameter_grid()
    r8_params = CausalMultiscaleHodgeParameters()
    forensic_params = BaselineForensicParameters()
    per_recording = args.output / f"per_recording_{args.phase}"
    per_recording.mkdir(exist_ok=True)
    manifest_root = args.project_root / "evidence" / "r8m_frozen_manifests"
    all_candidates, all_diagnostics = [], []
    manifest_audit = {}

    for array_label, cfg in ARRAYS.items():
        sub = manifest_root / cfg["manifest_dir"]
        manifest = pd.read_csv(sub / "manifest.csv")
        lock = json.loads((sub / "manifest_lock.json").read_text(encoding="utf-8"))
        actual_sha = canonical_manifest_sha256(manifest)
        expected_sha = str(lock.get("manifest_canonical_sha256"))
        if actual_sha != expected_sha:
            raise SystemExit(f"{array_label} frozen manifest canonical SHA mismatch: {actual_sha} != {expected_sha}")
        if int(lock.get("manifest_rows", -1)) != len(manifest):
            raise SystemExit(f"{array_label} frozen manifest row count mismatch")
        channels = _channels_from_lock(lock)
        array_id = str(cfg["array_id"])
        if str(lock.get("array_name")) != array_id:
            raise SystemExit(f"{array_label} array ID mismatch")
        manifest = manifest[manifest["task"].astype(str).map(lambda x: _phase_task_allowed(x, args.phase))].copy()
        manifest_audit[array_label] = {"phase_manifest_rows": int(len(manifest)), "full_manifest_canonical_sha256": actual_sha, "array_id": array_id, "channels": channels.tolist()}
        if len(manifest) == 0:
            continue

        audit_dir = args.output / "geometry_audit" / array_label
        audit, _ = audit_locata_root(args.root, audit_dir, array_name=array_id)
        usable = audit.loc[audit["usable"].astype(bool), "recording"].tolist()
        recording_map = {(infer_locata_task(Path(item)), Path(item).name): Path(item) for item in usable}
        expected = sorted({(str(r.task), str(r.recording_name)) for r in manifest.itertuples()})
        missing = [key for key in expected if key not in recording_map]
        if missing:
            raise SystemExit(f"{array_label} frozen recordings missing: {missing}")

        for task, recording_name in expected:
            recording = recording_map[(task, recording_name)]
            subset = manifest[(manifest["task"].astype(str) == task) & (manifest["recording_name"].astype(str) == recording_name)].copy()
            stem = f"{array_label}_{task}_{recording_name}"
            cand_path = per_recording / f"{PREFIX}_{args.phase.upper()}_GRID_{stem}.csv"
            diag_path = per_recording / f"{PREFIX}_{args.phase.upper()}_DIAGNOSTICS_{stem}.csv"
            status_path = per_recording / f"{PREFIX}_{args.phase.upper()}_STATUS_{stem}.json"
            complete = all(x.is_file() for x in (cand_path, diag_path, status_path))
            if args.resume and complete:
                cand = pd.read_csv(cand_path)
                diag = pd.read_csv(diag_path)
                status = json.loads(status_path.read_text(encoding="utf-8"))
                expected_rows = len(subset) * len(grid) * len(variants)
                if len(cand) != expected_rows or len(diag) != len(subset) or int(status.get("windows", -1)) != len(subset):
                    complete = False
            if not (args.resume and complete):
                cand, diag, status = evaluate_recording(
                    recording, subset, phase=args.phase, array_label=array_label, array_id=array_id,
                    channels=channels, variants=variants, grid=grid,
                    r8_params=r8_params, forensic_params=forensic_params,
                )
                cand.to_csv(cand_path, index=False)
                diag.to_csv(diag_path, index=False)
                status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            all_candidates.append(cand)
            all_diagnostics.append(diag)

    candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    diagnostics = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
    up = args.phase.upper()
    candidates.to_csv(args.output / f"{PREFIX}_{up}_GRID_TRIALS.csv", index=False)
    diagnostics.to_csv(args.output / f"{PREFIX}_{up}_WINDOW_DIAGNOSTICS.csv", index=False)
    execution = {
        "release": RELEASE, "protocol_id": PROTOCOL_ID, "phase": args.phase, "status": "completed",
        "development_only": True, "coupling_variants": list(variants),
        "eligible_final_methods": list(FINAL_CANDIDATE_VARIANTS),
        "window_count_per_array": {name: int(np.sum(diagnostics.get("array_label", pd.Series(dtype=str)).astype(str) == name)) for name in ARRAYS},
        "total_unique_windows": int(len(diagnostics)),
        "parameter_grid": [x.as_dict() for x in grid], "parameter_count_per_variant": int(len(grid)),
        "candidate_count_per_window": int(len(variants) * len(grid)),
        "manifest_audit": manifest_audit,
        "task_identity_used_for_estimation": False, "truth_used_for_parameterized_estimation": False,
        "truth_used_later_for_development_evaluation": True, "all_three_arrays_already_seen": True,
        "independent_validation_claimed": False,
    }
    (args.output / f"{PREFIX}_{up}_EXECUTION_STATUS.json").write_text(json.dumps(execution, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(execution, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
