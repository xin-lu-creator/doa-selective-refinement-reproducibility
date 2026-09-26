# =========================================================
# File        : evaluate_v5b_mst_tdoa.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Evaluate v5b mst tdoa.
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
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARRAYS = ("dicit", "eigenmike", "robothead")
TASKS = ("task1", "task2", "task3", "task4")
KEY = ["array_label", "task", "recording_name", "required_time_row_index"]
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20_260_913
METHOD_ERROR_COLUMNS = {
    "E0": "e0_abs_deg",
    "C": "c_abs_deg",
    "SRP-PHAT": "srp_abs_deg",
    "MST-TDE": "mst_abs_deg",
}


def _recording_metrics(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    col = METHOD_ERROR_COLUMNS[method]
    rows = []
    for (array_label, task, recording), g in frame.groupby(["array_label", "task", "recording_name"], sort=True):
        error = g[col].to_numpy(float)
        rows.append({
            "method": method,
            "array_label": array_label,
            "task": task,
            "recording_name": recording,
            "rmse_deg": float(np.sqrt(np.mean(error * error))),
            "window_count": int(len(g)),
        })
    return pd.DataFrame(rows)


def _task_balanced_macro(recording: pd.DataFrame) -> float:
    table = recording.groupby(["array_label", "task"]).rmse_deg.mean().unstack().loc[list(ARRAYS), list(TASKS)]
    return float(table.to_numpy().mean())


def _method_summary(frame: pd.DataFrame, method: str) -> dict:
    col = METHOD_ERROR_COLUMNS[method]
    recording = _recording_metrics(frame, method)
    error = frame[col].to_numpy(float)
    array_macro = {}
    task_array = []
    for array_label in ARRAYS:
        r = recording[recording.array_label == array_label]
        task_macro = r.groupby("task").rmse_deg.mean().reindex(TASKS)
        array_macro[array_label] = float(task_macro.mean())
        for task, value in task_macro.items():
            task_array.append({"method": method, "array_label": array_label, "task": task, "recording_macro_rmse_deg": float(value)})
    return {
        "summary": {
            "method": method,
            "task_balanced_recording_macro_rmse_deg": _task_balanced_macro(recording),
            "pooled_p95_abs_error_deg": float(np.quantile(error, 0.95)),
            "pooled_p99_abs_error_deg": float(np.quantile(error, 0.99)),
            "pooled_gt10_count": int(np.sum(error > 10.0)),
            "pooled_gt45_count": int(np.sum(error > 45.0)),
            **{f"{a}_recording_macro_rmse_deg": array_macro[a] for a in ARRAYS},
        },
        "recording": recording,
        "array_task": pd.DataFrame(task_array),
    }


def _paired_bootstrap(recording_a: pd.DataFrame, recording_b: pd.DataFrame, *, candidate: str, reference: str) -> dict:
    # Arrays are sampled jointly within each task/recording identity; recording identities
    # are resampled within task so that the final estimand remains task-balanced.
    a = recording_a.rename(columns={"rmse_deg": "candidate_rmse"})[["array_label", "task", "recording_name", "candidate_rmse"]]
    b = recording_b.rename(columns={"rmse_deg": "reference_rmse"})[["array_label", "task", "recording_name", "reference_rmse"]]
    rec = a.merge(b, on=["array_label", "task", "recording_name"], validate="one_to_one")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    candidate_arrays = {arr: np.zeros(BOOTSTRAP_DRAWS, dtype=float) for arr in ARRAYS}
    reference_arrays = {arr: np.zeros(BOOTSTRAP_DRAWS, dtype=float) for arr in ARRAYS}
    for task in TASKS:
        recordings = sorted(rec.loc[rec.task == task, "recording_name"].unique())
        draw = rng.integers(0, len(recordings), size=(BOOTSTRAP_DRAWS, len(recordings)))
        for array_label in ARRAYS:
            g = rec[(rec.task == task) & (rec.array_label == array_label)].set_index("recording_name").reindex(recordings)
            candidate_arrays[array_label] += g.candidate_rmse.to_numpy()[draw].mean(axis=1) / len(TASKS)
            reference_arrays[array_label] += g.reference_rmse.to_numpy()[draw].mean(axis=1) / len(TASKS)
    delta = np.mean(np.vstack([candidate_arrays[a] for a in ARRAYS]), axis=0) - np.mean(np.vstack([reference_arrays[a] for a in ARRAYS]), axis=0)
    return {
        "comparison": f"{candidate}_minus_{reference}",
        "candidate": candidate,
        "reference": reference,
        "point_delta_deg": float(_task_balanced_macro(recording_a) - _task_balanced_macro(recording_b)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_mean_delta_deg": float(delta.mean()),
        "ci95_low_deg": float(np.quantile(delta, 0.025)),
        "ci95_high_deg": float(np.quantile(delta, 0.975)),
        "probability_candidate_better": float(np.mean(delta < 0.0)),
    }


def _tail_transitions(frame: pd.DataFrame, candidate_col: str, reference_col: str, label: str) -> dict:
    c = frame[candidate_col].to_numpy(float)
    r = frame[reference_col].to_numpy(float)
    return {
        "comparison": label,
        "new_gt10": int(np.sum((c > 10.0) & (r <= 10.0))),
        "removed_gt10": int(np.sum((c <= 10.0) & (r > 10.0))),
        "new_gt45": int(np.sum((c > 45.0) & (r <= 45.0))),
        "removed_gt45": int(np.sum((c <= 45.0) & (r > 45.0))),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mst-results", type=Path, required=True)
    p.add_argument("--frozen-e0-c-srp", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.frozen_e0_c_srp)
    mst = pd.read_csv(args.mst_results)
    if len(base) != 1728 or len(mst) != 1728:
        raise SystemExit(f"Expected 1,728 rows in both inputs, got base={len(base)}, mst={len(mst)}")
    if base.duplicated(KEY).any() or mst.duplicated(KEY).any():
        raise SystemExit("Duplicate identity rows detected")
    keep = KEY + ["true_azimuth_deg", "estimated_azimuth_deg", "absolute_error_deg", "runtime_s"]
    mst = mst[keep].rename(columns={
        "true_azimuth_deg": "mst_true_azimuth_deg",
        "estimated_azimuth_deg": "mst_estimate_deg",
        "absolute_error_deg": "mst_abs_deg",
        "runtime_s": "mst_runtime_s",
    })
    merged = base.merge(mst, on=KEY, how="inner", validate="one_to_one")
    if len(merged) != 1728:
        raise SystemExit("Exact identity merge did not retain 1,728 rows")
    if np.max(np.abs(merged.true_azimuth_deg.to_numpy(float) - merged.mst_true_azimuth_deg.to_numpy(float))) > 1e-9:
        raise SystemExit("Truth mismatch after exact identity merge")
    merged.to_csv(args.output / "V5B_EXACT_MST_E0_C_SRP_WINDOWS.csv", index=False)

    blocks = {m: _method_summary(merged, m) for m in METHOD_ERROR_COLUMNS}
    pd.DataFrame([blocks[m]["summary"] for m in METHOD_ERROR_COLUMNS]).to_csv(args.output / "V5B_METHOD_SUMMARY.csv", index=False)
    pd.concat([blocks[m]["array_task"] for m in METHOD_ERROR_COLUMNS], ignore_index=True).to_csv(args.output / "V5B_ARRAY_TASK_METHOD_SUMMARY.csv", index=False)
    pd.concat([blocks[m]["recording"] for m in METHOD_ERROR_COLUMNS], ignore_index=True).to_csv(args.output / "V5B_RECORDING_METRICS.csv", index=False)

    comparisons = []
    for candidate, reference in [("MST-TDE", "E0"), ("MST-TDE", "C"), ("MST-TDE", "SRP-PHAT")]:
        comparisons.append(_paired_bootstrap(blocks[candidate]["recording"], blocks[reference]["recording"], candidate=candidate, reference=reference))
    pd.DataFrame(comparisons).to_csv(args.output / "V5B_PAIRED_BOOTSTRAP.csv", index=False)

    tails = [
        _tail_transitions(merged, "mst_abs_deg", "e0_abs_deg", "MST-TDE_minus_E0"),
        _tail_transitions(merged, "mst_abs_deg", "c_abs_deg", "MST-TDE_minus_C"),
        _tail_transitions(merged, "mst_abs_deg", "srp_abs_deg", "MST-TDE_minus_SRP-PHAT"),
    ]
    pd.DataFrame(tails).to_csv(args.output / "V5B_TAIL_TRANSITIONS.csv", index=False)

    runtime = merged.groupby("array_label").mst_runtime_s.agg(
        count="count", mean_s="mean", median_s="median", p90_s=lambda x: float(np.quantile(x, 0.90))
    ).reset_index()
    overall = pd.DataFrame([{
        "array_label": "overall",
        "count": int(len(merged)),
        "mean_s": float(merged.mst_runtime_s.mean()),
        "median_s": float(merged.mst_runtime_s.median()),
        "p90_s": float(merged.mst_runtime_s.quantile(0.90)),
    }])
    pd.concat([runtime, overall], ignore_index=True).to_csv(args.output / "V5B_MST_RUNTIME_DESCRIPTIVE.csv", index=False)

    status = {
        "status": "COMPLETED",
        "window_count": int(len(merged)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interpretation": "post-freeze nearest-comparator audit; no comparator result may alter C",
    }
    (args.output / "V5B_EVALUATION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
