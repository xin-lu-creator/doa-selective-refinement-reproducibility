# =========================================================
# File        : run_postfreeze_mst_tdoa_identical_manifest.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Run postfreeze mst tdoa identical manifest.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - Derived CSV/JSON evidence or evaluation summaries as defined by the CLI entry point.
#
# Used in paper:
#   - Adapted MST-TDE comparison reported in the manuscript and Supplementary Material.
#
# Main parameters:
#   - Preregistered MST-TDE adaptation; no post-result tuning.
#   - See documentation/CODE_GUIDE.md for fixed interface details.
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
from sgf_amusic_route_b.locata_geometry import audit_locata_root, infer_locata_task, load_broadband_windows
from sgf_amusic_route_b.mst_tdoa_yamaoka2023 import MSTTDOAParameters, mst_tdoa_azimuth_estimate

PROTOCOL_ID = "DOA-V5B-POSTFREEZE-MST-TDOA-YAMAOKA2023-20260913"
MANIFEST_SHA256 = {
    "dicit": "65708594bf37db1341159c9e05cde3f96374a92acc478fcbd658ccaa00198d14",
    "eigenmike": "2edcf43642b2fb4de25d1bbf903c851e610946e9abf488cfcd22ca0533102ad2",
    "robothead": "d0b7d7b8e323af7ffae0725ad12692e4d3ef35558a584610b78c5d021d869e55",
}
ARRAYS = {
    "dicit": {"array_name": "dicit", "lock_dir": "dicit"},
    "eigenmike": {"array_name": "eigenmike", "lock_dir": "eigenmike"},
    "robothead": {"array_name": "benchmark2", "lock_dir": "robothead"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _channels_from_lock(lock: dict) -> np.ndarray:
    values = lock.get("selected_channel_indices_zero_based", lock.get("native_channel_indices_zero_based"))
    if values is None:
        raise RuntimeError("Frozen manifest lock does not define channel indices")
    return np.asarray([int(v) for v in values], dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run preregistered post-freeze MST-TDOA nearest comparator")
    parser.add_argument("--root", type=Path, required=True, help="LOCATA final-evaluation root")
    parser.add_argument("--frozen-exact", type=Path, required=True, help="V2_WP1_EXACT_SRP_E0_C_WINDOWS.csv; used only to verify the frozen array-window key set")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    frozen = pd.read_csv(args.frozen_exact)
    key = ["array_label", "task", "recording_name", "required_time_row_index"]
    if len(frozen) != 1728 or frozen.duplicated(key).any():
        raise SystemExit("Frozen exact endpoint file is not 1,728 unique array-window rows")

    lock_root = args.project_root / "evidence" / "r8m_frozen_manifests"
    args.output.mkdir(parents=True, exist_ok=True)
    per = args.output / "per_recording"
    per.mkdir(exist_ok=True)
    params = MSTTDOAParameters()
    all_rows: list[pd.DataFrame] = []
    channel_audit = {}

    manifest_hashes = {}
    for array_label, cfg in ARRAYS.items():
        array_dir = lock_root / cfg["lock_dir"]
        manifest_path = array_dir / "manifest.csv"
        actual_hash = sha256(manifest_path)
        expected_hash = MANIFEST_SHA256[array_label]
        if actual_hash != expected_hash:
            raise SystemExit(f"Frozen {array_label} manifest hash mismatch: {actual_hash}")
        manifest = pd.read_csv(manifest_path)
        if len(manifest) != 576:
            raise SystemExit(f"Expected 576 frozen windows for {array_label}, found {len(manifest)}")
        local_key = ["task", "recording_name", "required_time_row_index"]
        if manifest.duplicated(local_key).any():
            raise SystemExit(f"Frozen {array_label} manifest contains duplicate array-window keys")
        expected_keys = frozen.loc[frozen["array_label"].astype(str) == array_label, local_key].drop_duplicates()
        actual_keys = manifest[local_key].drop_duplicates()
        key_check = actual_keys.merge(expected_keys, on=local_key, how="outer", indicator=True)
        if len(expected_keys) != 576 or not (key_check["_merge"] == "both").all():
            raise SystemExit(f"Frozen {array_label} manifest does not match the registered final-evidence key set")
        manifest_hashes[array_label] = actual_hash
        print(f"[preflight] {array_label}: 576/576 frozen keys match final evidence; manifest SHA-256 PASS", flush=True)

        lock = json.loads((array_dir / "manifest_lock.json").read_text(encoding="utf-8"))
        channels = _channels_from_lock(lock)
        channel_audit[array_label] = channels.tolist()
        array_name = cfg["array_name"]
        audit, _ = audit_locata_root(args.root, args.output / f"geometry_audit_{array_label}", array_name=array_name)
        usable = audit.loc[audit.usable.astype(bool), "recording"].tolist()
        recmap = {(infer_locata_task(Path(x)), Path(x).name): Path(x) for x in usable}

        for (task, recording_name), sub in manifest.groupby(["task", "recording_name"], sort=True):
            recording = recmap.get((str(task), str(recording_name)))
            if recording is None:
                raise SystemExit(f"Missing recording for {array_label}/{task}/{recording_name}")
            target = per / f"MST_{array_label}_{task}_{recording_name}.csv"
            if args.resume and target.is_file():
                old = pd.read_csv(target)
                if len(old) == len(sub):
                    all_rows.append(old)
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
                source = int(item.active_source_index)
                truth = float(np.asarray(frame["theta_true_deg"], float).reshape(-1)[source])
                audio_full = np.asarray(frame["audio"], float)
                positions_full = np.asarray(frame["microphone_positions_m"], float)
                if np.any(channels < 0) or np.any(channels >= audio_full.shape[0]):
                    raise SystemExit(f"Frozen channels invalid for {array_label}/{task}/{recording_name}")
                audio = audio_full[channels]
                positions = positions_full[channels]

                started = time.perf_counter()
                estimate, diag = mst_tdoa_azimuth_estimate(
                    audio, float(frame["sample_rate_hz"]), positions, params=params
                )
                runtime = float(time.perf_counter() - started)
                signed = float(_circular_difference_deg(estimate, truth))
                rows.append({
                    "protocol_id": PROTOCOL_ID,
                    "array_label": array_label,
                    "array_name": array_name,
                    "task": str(task),
                    "recording_name": str(recording_name),
                    "required_time_row_index": row_id,
                    "selection_rank": int(item.selection_rank),
                    "active_source_index": source,
                    "true_azimuth_deg": truth,
                    "estimated_azimuth_deg": float(estimate),
                    "signed_error_deg": signed,
                    "absolute_error_deg": abs(signed),
                    "squared_error_deg2": signed * signed,
                    "runtime_s": runtime,
                    "truth_used_for_estimation": False,
                    "alpha": int(params.alpha),
                    "frequency_min_hz": float(params.frequency_min_hz),
                    "frequency_max_hz": float(params.frequency_max_hz),
                    "theta_min_deg": float(params.theta_min_deg),
                    "theta_max_deg": float(params.theta_max_deg),
                    "theta_step_deg": float(params.theta_step_deg),
                    "frozen_channel_count": int(len(channels)),
                    "frozen_channel_indices_json": json.dumps(channels.tolist()),
                    "full_pair_count": int(diag["pair_count_full"]),
                    "mst_edge_count": int(diag["tree_edge_count"]),
                    "mst_edges_json": json.dumps(np.asarray(diag["mst_edges"], int).tolist()),
                })
                print(
                    f"[MST {array_label}/{task}/{recording_name}] {position}/{len(sub)} "
                    f"row={row_id} time={runtime:.2f}s",
                    flush=True,
                )
            frame_rows = pd.DataFrame(rows)
            frame_rows.to_csv(target, index=False)
            all_rows.append(frame_rows)

    trials = pd.concat(all_rows, ignore_index=True).sort_values(
        ["array_label", "task", "recording_name", "required_time_row_index"], kind="mergesort"
    )
    key = ["array_label", "task", "recording_name", "required_time_row_index"]
    if len(trials) != 1728 or trials.duplicated(key).any():
        raise SystemExit("Output is not exactly 1,728 unique array-window rows")
    output_csv = args.output / "V5B_MST_TDOA_WINDOW_RESULTS.csv"
    trials.to_csv(output_csv, index=False)
    status = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETED",
        "scientific_identity": "Yamaoka-2023 MST-TDE alpha=1 plus preregistered unweighted DOA interface",
        "window_count": int(len(trials)),
        "per_array": trials.groupby("array_label").size().to_dict(),
        "manifest_sha256_by_array": manifest_hashes,
        "configuration": params.as_dict(),
        "frozen_channels": channel_audit,
        "truth_used_for_estimation": False,
        "post_result_tuning_permitted": False,
    }
    (args.output / "V5B_MST_TDOA_EXECUTION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
