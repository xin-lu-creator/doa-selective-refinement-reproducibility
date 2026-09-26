# =========================================================
# File        : run_postfreeze_srpphat_identical_manifest.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Run postfreeze srpphat identical manifest.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - Derived CSV/JSON evidence or evaluation summaries as defined by the CLI entry point.
#
# Used in paper:
#   - SRP-PHAT comparison reported in the manuscript and Supplementary Material.
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
from sgf_amusic_route_b.canonical_srp_phat import CanonicalSRPParameters, canonical_srp_phat_estimates
from sgf_amusic_route_b.locata_geometry import audit_locata_root, infer_locata_task, load_broadband_windows


PROTOCOL_ID = "DOA-R8M-POSTFREEZE-SRP-AUDIT-20260814"
PROTOCOL_VERSION = 2
EXPECTED_ROWS_PER_ARRAY = 576
EXPECTED_TOTAL_ROWS = 1728
ARRAYS = {
    "dicit": {
        "array_name": "dicit",
        "manifest_dir": "dicit",
        "canonical_sha256": "fbde5a2dc106e475806077181d1bee6f1c2ccc9d592db42f033685f93b322b78",
    },
    "eigenmike": {
        "array_name": "eigenmike",
        "manifest_dir": "eigenmike",
        "canonical_sha256": "6318f35a246de138e43aa4a46fa5c5281efdae67c19f53f5b30c74dbe0f8c938",
    },
    "robothead": {
        "array_name": "benchmark2",
        "manifest_dir": "robothead",
        "canonical_sha256": "0de0b5d36deadee33a2000921c84757962fff12227508ae27cd972f0991ab641",
    },
}


def canonical_manifest_sha256(frame: pd.DataFrame) -> str:
    rows = [
        {
            "task": str(r.task),
            "recording_name": str(r.recording_name),
            "required_time_row_index": int(r.required_time_row_index),
            "active_source_index": int(r.active_source_index),
        }
        for r in frame.itertuples()
    ]
    rows.sort(
        key=lambda x: (
            x["task"],
            x["recording_name"],
            x["required_time_row_index"],
            x["active_source_index"],
        )
    )
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def channels_from_lock(lock: dict) -> np.ndarray:
    values = lock.get("selected_channel_indices_zero_based", lock.get("native_channel_indices_zero_based"))
    if values is None:
        raise SystemExit("Frozen manifest lock does not contain channel indices")
    channels = np.asarray([int(x) for x in values], dtype=int)
    if channels.ndim != 1 or len(channels) < 2 or len(set(channels.tolist())) != len(channels):
        raise SystemExit("Frozen channel list is invalid")
    return channels


def load_and_audit_manifests(project_root: Path) -> dict[str, dict]:
    manifest_root = project_root / "evidence" / "r8m_frozen_manifests"
    audited: dict[str, dict] = {}
    print("[manifest preflight] Array-specific frozen manifests", flush=True)
    for array_label, cfg in ARRAYS.items():
        folder = manifest_root / str(cfg["manifest_dir"])
        manifest_path = folder / "manifest.csv"
        lock_path = folder / "manifest_lock.json"
        if not manifest_path.is_file() or not lock_path.is_file():
            raise SystemExit(f"Missing frozen manifest/lock for {array_label}: {folder}")
        manifest = pd.read_csv(manifest_path)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if str(lock.get("status")) != "FROZEN":
            raise SystemExit(f"{array_label} manifest lock is not FROZEN")
        if len(manifest) != EXPECTED_ROWS_PER_ARRAY or int(lock.get("manifest_rows", -1)) != EXPECTED_ROWS_PER_ARRAY:
            raise SystemExit(
                f"{array_label} manifest row mismatch: csv={len(manifest)}, lock={lock.get('manifest_rows')}, "
                f"expected={EXPECTED_ROWS_PER_ARRAY}"
            )
        required = {"task", "recording_name", "required_time_row_index", "active_source_index", "selection_rank"}
        missing = sorted(required.difference(manifest.columns))
        if missing:
            raise SystemExit(f"{array_label} manifest missing columns: {missing}")
        duplicate_key = ["task", "recording_name", "required_time_row_index", "active_source_index"]
        if manifest.duplicated(duplicate_key).any():
            raise SystemExit(f"{array_label} frozen manifest has duplicate window keys")
        actual_sha = canonical_manifest_sha256(manifest)
        expected_sha = str(cfg["canonical_sha256"])
        lock_sha = str(lock.get("manifest_canonical_sha256"))
        if actual_sha != expected_sha or lock_sha != expected_sha:
            raise SystemExit(
                f"{array_label} canonical manifest SHA mismatch: actual={actual_sha}, lock={lock_sha}, expected={expected_sha}"
            )
        expected_array_name = str(cfg["array_name"])
        if str(lock.get("array_name")) != expected_array_name:
            raise SystemExit(
                f"{array_label} array ID mismatch: lock={lock.get('array_name')} expected={expected_array_name}"
            )
        channels = channels_from_lock(lock)
        audited[array_label] = {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "lock_path": lock_path,
            "canonical_sha256": actual_sha,
            "array_name": expected_array_name,
            "channels": channels,
        }
        print(
            f"  {array_label}: rows={len(manifest)} canonical_sha256=PASS channels={len(channels)} {channels.tolist()}",
            flush=True,
        )
    total = sum(len(v["manifest"]) for v in audited.values())
    if total != EXPECTED_TOTAL_ROWS:
        raise SystemExit(f"Frozen manifest total mismatch: expected {EXPECTED_TOTAL_ROWS}, found {total}")
    print(f"  TOTAL: {total} array-specific frozen windows -- PASS", flush=True)
    return audited


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="LOCATA final-evaluation root")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--manifest-preflight-only",
        action="store_true",
        help="Validate the three frozen array-specific manifests and channel locks without reading audio",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    audited = load_and_audit_manifests(project_root)
    if args.manifest_preflight_only:
        print("WP1 manifest/channel preflight: PASS", flush=True)
        return 0
    if args.root is None or args.output is None:
        parser.error("--root and --output are required unless --manifest-preflight-only is used")

    args.output.mkdir(parents=True, exist_ok=True)
    per = args.output / "per_recording"
    per.mkdir(exist_ok=True)
    params = CanonicalSRPParameters(theta_min_deg=-80, theta_max_deg=80, theta_step_deg=1.0)
    all_rows = []
    manifest_status = {}

    for array_label, info in audited.items():
        array_name = str(info["array_name"])
        manifest = info["manifest"]
        channels = np.asarray(info["channels"], dtype=int)
        audit, _ = audit_locata_root(args.root, args.output / f"geometry_audit_{array_label}", array_name=array_name)
        usable = audit.loc[audit.usable.astype(bool), "recording"].tolist()
        recmap = {(infer_locata_task(Path(x)), Path(x).name): Path(x) for x in usable}
        expected_recordings = sorted(
            {(str(r.task), str(r.recording_name)) for r in manifest.itertuples()}
        )
        missing_recordings = [key for key in expected_recordings if key not in recmap]
        if missing_recordings:
            raise SystemExit(f"Missing frozen recordings for {array_label}: {missing_recordings}")

        for task, recording_name in expected_recordings:
            sub = manifest[
                (manifest["task"].astype(str) == task)
                & (manifest["recording_name"].astype(str) == recording_name)
            ].copy()
            recording = recmap[(task, recording_name)]
            target = per / f"SRP_{array_label}_{task}_{recording_name}.csv"
            if args.resume and target.is_file():
                old = pd.read_csv(target)
                expected_keys = set(sub["required_time_row_index"].astype(int).tolist())
                old_keys = set(old.get("required_time_row_index", pd.Series(dtype=int)).astype(int).tolist())
                if (
                    len(old) == len(sub)
                    and expected_keys == old_keys
                    and "array_label" in old.columns
                    and set(old["array_label"].astype(str)) == {array_label}
                    and "manifest_canonical_sha256" in old.columns
                    and set(old["manifest_canonical_sha256"].astype(str)) == {str(info["canonical_sha256"])}
                ):
                    all_rows.append(old)
                    print(f"[resume {array_label}/{task}/{recording_name}] {len(old)} windows", flush=True)
                    continue

            windows, _ = load_broadband_windows(
                recording,
                array_name=array_name,
                window_duration_s=0.75,
                max_windows=0,
                window_anchor_mode="required_time_causal",
                activity_aware_max_windows=False,
            )
            window_map = {int(w["required_time_row_index"]): w for w in windows}
            rows = []
            for position, item in enumerate(sub.sort_values("selection_rank").itertuples(), 1):
                row_id = int(item.required_time_row_index)
                if row_id not in window_map:
                    raise SystemExit(f"Missing frozen row {array_label}/{task}/{recording_name}/{row_id}")
                frame = window_map[row_id]
                audio_all = np.asarray(frame["audio"], float)
                positions_all = np.asarray(frame["microphone_positions_m"], float)
                if np.any(channels < 0) or np.any(channels >= audio_all.shape[0]) or positions_all.shape[0] != audio_all.shape[0]:
                    raise SystemExit(
                        f"Frozen channels invalid for {array_label}/{task}/{recording_name}: "
                        f"M={audio_all.shape[0]}, channels={channels.tolist()}"
                    )
                audio = audio_all[channels]
                positions = positions_all[channels]
                source = int(item.active_source_index)
                truth_values = np.asarray(frame["theta_true_deg"], float).reshape(-1)
                if source < 0 or source >= len(truth_values) or not np.isfinite(truth_values[source]):
                    raise SystemExit(f"Invalid frozen source index {array_label}/{task}/{recording_name}/{row_id}")
                truth = float(truth_values[source])
                started = time.perf_counter()
                result = canonical_srp_phat_estimates(
                    audio,
                    frame["sample_rate_hz"],
                    positions,
                    params=params,
                )
                runtime = time.perf_counter() - started
                estimate = float(result.estimates_deg["canonical_srp_phat_linear"])
                signed = float(_circular_difference_deg(estimate, truth))
                rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "protocol_version": PROTOCOL_VERSION,
                        "array_label": array_label,
                        "array_name": array_name,
                        "task": task,
                        "recording_name": recording_name,
                        "required_time_row_index": row_id,
                        "selection_rank": int(item.selection_rank),
                        "active_source_index": source,
                        "true_azimuth_deg": truth,
                        "estimated_azimuth_deg": estimate,
                        "signed_error_deg": signed,
                        "absolute_error_deg": abs(signed),
                        "squared_error_deg2": signed * signed,
                        "runtime_s": runtime,
                        "truth_used_for_estimation": False,
                        "theta_min_deg": -80.0,
                        "theta_max_deg": 80.0,
                        "theta_step_deg": 1.0,
                        "frequency_min_hz": params.frequency_min_hz,
                        "frequency_max_hz": params.frequency_max_hz,
                        "pair_weights_used": False,
                        "postprocessing_used": False,
                        "frozen_channel_count": int(len(channels)),
                        "frozen_channel_indices_json": json.dumps(channels.tolist()),
                        "manifest_canonical_sha256": str(info["canonical_sha256"]),
                    }
                )
                print(
                    f"[SRP {array_label}/{task}/{recording_name}] {position}/{len(sub)} "
                    f"row={row_id} channels={len(channels)} time={runtime:.2f}s",
                    flush=True,
                )
            frame_rows = pd.DataFrame(rows)
            frame_rows.to_csv(target, index=False)
            all_rows.append(frame_rows)

        manifest_status[array_label] = {
            "rows": int(len(manifest)),
            "canonical_sha256": str(info["canonical_sha256"]),
            "array_name": array_name,
            "frozen_channel_indices_zero_based": channels.tolist(),
        }

    trials = pd.concat(all_rows, ignore_index=True).sort_values(
        ["array_label", "task", "recording_name", "required_time_row_index"], kind="mergesort"
    )
    unique_key = ["array_label", "task", "recording_name", "required_time_row_index", "active_source_index"]
    if len(trials) != EXPECTED_TOTAL_ROWS or trials.duplicated(unique_key).any():
        raise SystemExit(
            f"Output is not exactly {EXPECTED_TOTAL_ROWS} unique array-specific frozen windows: rows={len(trials)}"
        )
    per_array = trials.groupby("array_label").size().to_dict()
    if any(int(per_array.get(name, 0)) != EXPECTED_ROWS_PER_ARRAY for name in ARRAYS):
        raise SystemExit(f"Per-array output count mismatch: {per_array}")

    trials.to_csv(args.output / "WP1_SRP_PHAT_WINDOW_RESULTS.csv", index=False)
    status = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETED",
        "scientific_identity": "post-freeze external comparator audit on each array's own frozen manifest",
        "engineering_revision": "V2 fixes pre-performance manifest routing and frozen-channel application; no SRP performance outcome existed before this revision",
        "window_count": int(len(trials)),
        "per_array": {k: int(v) for k, v in per_array.items()},
        "manifest_audit": manifest_status,
        "configuration": params.as_dict(),
        "truth_used_for_estimation": False,
        "frozen_channels_applied": True,
    }
    (args.output / "WP1_SRP_PHAT_EXECUTION_STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
