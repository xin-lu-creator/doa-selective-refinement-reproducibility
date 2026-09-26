# =========================================================
# File        : evaluate_r8m_final_multimethod_validation.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Evaluate r8m final multimethod validation.
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
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sgf_amusic_route_b.mechanism_corrected_r8_multimethod_evidence_change import (
    ALL_VALIDATION_VARIANTS,
    FINAL_CANDIDATE_VARIANTS,
    REFERENCE_VARIANTS,
    FROZEN_ASSIGNMENT_STRENGTHS,
)

PREFIX = "R8MMV"
ARRAYS = ("eigenmike", "robothead", "dicit")
TASKS = ("task1", "task2", "task3", "task4")
GRID = tuple(float(x) for x in FROZEN_ASSIGNMENT_STRENGTHS)


def rmse(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(a * a))) if len(a) else float("nan")


def pct_gain(baseline: float, candidate: float) -> float:
    return 100.0 * (float(baseline) - float(candidate)) / max(float(baseline), 1e-12)


def _threshold_counts(c_abs: pd.Series, e0_abs: pd.Series) -> dict:
    return {
        "new_gt10": int(np.sum((c_abs > 10.0) & (e0_abs <= 10.0))),
        "removed_gt10": int(np.sum((c_abs <= 10.0) & (e0_abs > 10.0))),
        "new_gt45": int(np.sum((c_abs > 45.0) & (e0_abs <= 45.0))),
        "removed_gt45": int(np.sum((c_abs <= 45.0) & (e0_abs > 45.0))),
    }


def build_task1_metrics(trials: pd.DataFrame, diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(trials["task"].astype(str)) != {"task1"}:
        raise RuntimeError("Task-1 trials contain non-Task-1 rows")
    base = diagnostics.set_index(["array_label", "recording_name", "required_time_row_index"])
    rec_rows, array_rows = [], []
    for keys, g in trials.groupby(["array_label", "coupling_variant", "parameter_id", "assignment_strength", "anchor_strength"], sort=True):
        array_label, variant, pid, a, h = keys
        rec_candidates, rec_e0, rec_nondeg = [], [], []
        for rec, rg in g.groupby("recording_name", sort=True):
            idx = pd.MultiIndex.from_arrays([
                [array_label] * len(rg), [rec] * len(rg), rg["required_time_row_index"].astype(int).tolist()
            ])
            e0_abs = base.loc[idx, "e0_absolute_error_deg"].to_numpy(float)
            cand_abs = rg["absolute_error_deg"].to_numpy(float)
            e0_rmse, cand_rmse = rmse(e0_abs), rmse(cand_abs)
            rec_candidates.append(cand_rmse); rec_e0.append(e0_rmse); rec_nondeg.append(cand_rmse <= e0_rmse + 1e-9)
            rec_rows.append({
                "array_label": array_label, "coupling_variant": variant, "parameter_id": pid,
                "assignment_strength": float(a), "anchor_strength": float(h), "recording_name": rec,
                "e0_rmse_deg": e0_rmse, "candidate_rmse_deg": cand_rmse,
                "gain_vs_e0_pct": pct_gain(e0_rmse, cand_rmse),
                "nondegraded": bool(cand_rmse <= e0_rmse + 1e-9), "window_count": int(len(rg)),
            })
        e0_macro, cand_macro = float(np.mean(rec_e0)), float(np.mean(rec_candidates))
        diag_idx = pd.MultiIndex.from_arrays([
            [array_label] * len(g), g["recording_name"].astype(str).tolist(), g["required_time_row_index"].astype(int).tolist()
        ])
        e0_abs_all = base.loc[diag_idx, "e0_absolute_error_deg"].astype(float).reset_index(drop=True)
        cand_abs_all = g["absolute_error_deg"].astype(float).reset_index(drop=True)
        counts = _threshold_counts(cand_abs_all, e0_abs_all)
        array_rows.append({
            "array_label": array_label, "coupling_variant": variant, "parameter_id": pid,
            "assignment_strength": float(a), "anchor_strength": float(h),
            "task1_recording_macro_rmse_deg": cand_macro,
            "e0_task1_recording_macro_rmse_deg": e0_macro,
            "task1_gain_vs_e0_pct": pct_gain(e0_macro, cand_macro),
            "task1_relative_change_pct": 100.0 * (cand_macro - e0_macro) / max(e0_macro, 1e-12),
            "recording_nondegraded_fraction": float(np.mean(rec_nondeg)),
            "adoption_fraction": float(np.mean(g["r8m_adopt"].astype(bool))),
            "adoption_count": int(np.sum(g["r8m_adopt"].astype(bool))),
            "no_reassignment_guard_fraction": float(np.mean(g["r8m_no_reassignment_guard_triggered"].astype(bool))),
            "no_reassignment_guard_count": int(np.sum(g["r8m_no_reassignment_guard_triggered"].astype(bool))),
            "window_count": int(len(g)), **counts,
        })
    return pd.DataFrame(rec_rows), pd.DataFrame(array_rows)


def task1_array_feasible(row: pd.Series) -> bool:
    label, gain = str(row["array_label"]), float(row["task1_gain_vs_e0_pct"])
    gain_ok = gain >= 5.0 if label == "eigenmike" else gain >= -2.0 if label in {"robothead", "dicit"} else False
    return bool(
        gain_ok
        and float(row["recording_nondegraded_fraction"]) >= 0.70
        and int(row["new_gt45"]) == 0
        and int(row["new_gt10"]) <= int(row["removed_gt10"])
    )


def _candidate_task1_table(array_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, pid, a, h), g in array_metrics.groupby(["coupling_variant", "parameter_id", "assignment_strength", "anchor_strength"], sort=True):
        if set(g["array_label"].astype(str)) != set(ARRAYS):
            continue
        feas = [task1_array_feasible(r) for _, r in g.iterrows()]
        gains = {str(r.array_label): float(r.task1_gain_vs_e0_pct) for r in g.itertuples()}
        rows.append({
            "coupling_variant": variant, "eligible_final_method": bool(variant in FINAL_CANDIDATE_VARIANTS),
            "reference_only": bool(variant in REFERENCE_VARIANTS),
            "parameter_id": pid, "assignment_strength": float(a), "anchor_strength": float(h),
            "all_arrays_task1_feasible": bool(all(feas)), "feasible_array_count": int(sum(feas)),
            "dicit_task1_gain_pct": gains.get("dicit", np.nan),
            "eigenmike_task1_gain_pct": gains.get("eigenmike", np.nan),
            "robothead_task1_gain_pct": gains.get("robothead", np.nan),
            "min_array_task1_gain_pct": float(g["task1_gain_vs_e0_pct"].min()),
            "mean_array_task1_gain_pct": float(g["task1_gain_vs_e0_pct"].mean()),
            "min_recording_nondegraded_fraction": float(g["recording_nondegraded_fraction"].min()),
            "total_new_gt45": int(g["new_gt45"].sum()),
            "total_new_gt10_minus_removed": int((g["new_gt10"] - g["removed_gt10"]).sum()),
            "total_no_reassignment_guard_count": int(g["no_reassignment_guard_count"].sum()),
        })
    return pd.DataFrame(rows)


def _adjacent_values(values: list[float]) -> bool:
    present = {float(x) for x in values}
    return any(GRID[i] in present and GRID[i + 1] in present for i in range(len(GRID) - 1))


def _longest_adjacent_run(values: list[float]) -> int:
    present = {float(x) for x in values}
    best = cur = 0
    for x in GRID:
        if x in present:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def task1_phase(results: Path) -> int:
    trials_path = results / f"{PREFIX}_TASK1_GRID_TRIALS.csv"
    diag_path = results / f"{PREFIX}_TASK1_WINDOW_DIAGNOSTICS.csv"
    if not trials_path.is_file() or not diag_path.is_file():
        raise SystemExit("Task1 outputs incomplete")
    trials, diagnostics = pd.read_csv(trials_path), pd.read_csv(diag_path)
    if len(diagnostics) != 780:
        raise RuntimeError(f"Expected 780 Task1 windows, got {len(diagnostics)}")
    expected_rows = 780 * len(ALL_VALIDATION_VARIANTS) * len(GRID)
    if len(trials) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} Task1 candidate rows, got {len(trials)}")
    rec, array = build_task1_metrics(trials, diagnostics)
    table = _candidate_task1_table(array)
    rec.to_csv(results / f"{PREFIX}_TASK1_RECORDING_METRICS.csv", index=False)
    array.to_csv(results / f"{PREFIX}_TASK1_ARRAY_METRICS.csv", index=False)
    table.to_csv(results / f"{PREFIX}_TASK1_CANDIDATE_TABLE.csv", index=False)

    method_rows = []
    for variant in ALL_VALIDATION_VARIANTS:
        vg = table[table["coupling_variant"] == variant].copy()
        feasible = sorted(vg.loc[vg["all_arrays_task1_feasible"].astype(bool), "assignment_strength"].astype(float).tolist())
        method_rows.append({
            "coupling_variant": variant,
            "eligible_final_method": bool(variant in FINAL_CANDIDATE_VARIANTS),
            "task1_feasible_strengths": json.dumps(feasible),
            "task1_feasible_count": len(feasible),
            "task1_has_adjacent_feasible_pair": _adjacent_values(feasible),
            "task1_longest_adjacent_feasible_run": _longest_adjacent_run(feasible),
            "minimum_task1_feasible_strength": min(feasible) if feasible else np.nan,
        })
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(results / f"{PREFIX}_TASK1_METHOD_SUMMARY.csv", index=False)

    eligible_passers = [r["coupling_variant"] for r in method_rows if r["eligible_final_method"] and r["task1_feasible_count"] > 0]
    advance = bool(eligible_passers)
    decision = {
        "status": "R8MMV_TASK1_ADVANCE_TO_FULL" if advance else "R8MMV_TASK1_STOP_NO_FINAL_METHOD_PASSED",
        "advance_to_full": advance,
        "eligible_final_methods_with_any_global_task1_feasible_strength": eligible_passers,
        "all_validation_methods": list(ALL_VALIDATION_VARIANTS),
        "reference_methods": list(REFERENCE_VARIANTS),
        "anchor_strength": 0.03,
        "assignment_strengths": list(GRID),
        "no_reassignment_no_replacement_rule": True,
        "full_phase_policy": "If any eligible method passes Task1, run all predeclared methods and strengths on Task2-4 so LOAO never depends on holdout-based Task1 filtering.",
        "task1_gate": {
            "dicit_gain": ">=-2%", "robothead_gain": ">=-2%", "eigenmike_gain": ">=+5%",
            "recording_nondegraded_fraction_each_array": ">=0.70", "new_gt45_each_array": "=0",
            "new_gt10_each_array": "<= removed_gt10",
        },
    }
    (results / f"{PREFIX}_TASK1_DECISION.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# R8-M Multi-Method Evidence-Change: Task 1 Funnel", "", f"Status: `{decision['status']}`", "", "Final candidate methods advancing to the full tasks:"]
    lines += [f"- `{x}`" for x in eligible_passers] if eligible_passers else ["- None"]
    lines += ["", "Note: D/Shared-log can be evaluated as a reference but is not eligible for final freezing."]
    (results / f"{PREFIX}_TASK1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


def build_full_recording_metrics(trials: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    base = diagnostics[["array_label", "task", "recording_name", "required_time_row_index", "e0_absolute_error_deg", "original_r8_absolute_error_deg", "original_r8_adopt"]].copy()
    rows = []
    for (array_label, task, rec), g in base.groupby(["array_label", "task", "recording_name"], sort=True):
        rows.append({"array_label": array_label, "task": task, "recording_name": rec, "coupling_variant": "E0", "parameter_id": "E0", "assignment_strength": np.nan, "anchor_strength": np.nan, "rmse_deg": rmse(g["e0_absolute_error_deg"]), "window_count": len(g)})
        rows.append({"array_label": array_label, "task": task, "recording_name": rec, "coupling_variant": "ORIGINAL_R8", "parameter_id": "ORIGINAL_R8", "assignment_strength": 0.0, "anchor_strength": 0.0, "rmse_deg": rmse(g["original_r8_absolute_error_deg"]), "window_count": len(g)})
    for keys, g in trials.groupby(["array_label", "task", "recording_name", "coupling_variant", "parameter_id", "assignment_strength", "anchor_strength"], sort=True):
        array_label, task, rec, variant, pid, a, h = keys
        rows.append({"array_label": array_label, "task": task, "recording_name": rec, "coupling_variant": variant, "parameter_id": pid, "assignment_strength": float(a), "anchor_strength": float(h), "rmse_deg": rmse(g["signed_error_deg"]), "window_count": len(g)})
    return pd.DataFrame(rows)


def summarize_full_grid(trials: pd.DataFrame, diagnostics: pd.DataFrame, rec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    e0_rec = rec[rec["parameter_id"] == "E0"].set_index(["array_label", "task", "recording_name"])["rmse_deg"]
    grid_rows, task_rows = [], []
    for keys, g in trials.groupby(["array_label", "coupling_variant", "parameter_id", "assignment_strength", "anchor_strength"], sort=True):
        array_label, variant, pid, a, h = keys
        rg = rec[(rec["array_label"] == array_label) & (rec["coupling_variant"] == variant) & (rec["parameter_id"] == pid)].copy()
        task_rmse, task_e0 = {}, {}
        for task in TASKS:
            vals = rg[rg["task"] == task]["rmse_deg"]
            e0_vals = rec[(rec["array_label"] == array_label) & (rec["parameter_id"] == "E0") & (rec["task"] == task)]["rmse_deg"]
            if len(vals) == 0 or len(e0_vals) == 0:
                raise RuntimeError(f"Missing full task metrics for {array_label}/{variant}/{pid}/{task}")
            task_rmse[task], task_e0[task] = float(vals.mean()), float(e0_vals.mean())
            task_rows.append({"array_label": array_label, "coupling_variant": variant, "parameter_id": pid, "assignment_strength": float(a), "anchor_strength": float(h), "task": task, "recording_macro_rmse_deg": task_rmse[task], "e0_recording_macro_rmse_deg": task_e0[task], "gain_vs_e0_pct": pct_gain(task_e0[task], task_rmse[task])})
        overall, overall_e0 = float(np.mean(list(task_rmse.values()))), float(np.mean(list(task_e0.values())))
        merged = rg.set_index(["array_label", "task", "recording_name"])["rmse_deg"].to_frame("candidate")
        merged["e0"] = e0_rec.loc[merged.index]
        nondeg = float(np.mean(merged["candidate"] <= merged["e0"] + 1e-9))
        diag_array = diagnostics[diagnostics["array_label"] == array_label].set_index(["task", "recording_name", "required_time_row_index"])
        cand = g.set_index(["task", "recording_name", "required_time_row_index"])
        e0_abs = diag_array.loc[cand.index, "e0_absolute_error_deg"].astype(float)
        c_abs = cand["absolute_error_deg"].astype(float)
        counts = _threshold_counts(c_abs.reset_index(drop=True), e0_abs.reset_index(drop=True))
        t23c, t23e = float(np.mean([task_rmse["task2"], task_rmse["task3"]])), float(np.mean([task_e0["task2"], task_e0["task3"]]))
        grid_rows.append({
            "array_label": array_label, "coupling_variant": variant, "parameter_id": pid,
            "assignment_strength": float(a), "anchor_strength": float(h),
            "task_balanced_recording_macro_rmse_deg": overall,
            "e0_task_balanced_recording_macro_rmse_deg": overall_e0,
            "overall_gain_vs_e0_pct": pct_gain(overall_e0, overall),
            "task1_relative_change_pct": 100.0 * (task_rmse["task1"] - task_e0["task1"]) / max(task_e0["task1"], 1e-12),
            "task4_relative_change_pct": 100.0 * (task_rmse["task4"] - task_e0["task4"]) / max(task_e0["task4"], 1e-12),
            "task23_gain_vs_e0_pct": pct_gain(t23e, t23c),
            "recording_nondegraded_fraction": nondeg, **counts,
            "adoption_fraction": float(np.mean(g["r8m_adopt"].astype(bool))),
            "adoption_count": int(np.sum(g["r8m_adopt"].astype(bool))),
            "no_reassignment_guard_fraction": float(np.mean(g["r8m_no_reassignment_guard_triggered"].astype(bool))),
            "no_reassignment_guard_count": int(np.sum(g["r8m_no_reassignment_guard_triggered"].astype(bool))),
        })
    return pd.DataFrame(grid_rows), pd.DataFrame(task_rows)


def full_safety_feasible(row: pd.Series) -> bool:
    return bool(
        float(row["overall_gain_vs_e0_pct"]) >= -1e-9
        and float(row["task1_relative_change_pct"]) <= 2.0 + 1e-9
        and float(row["task4_relative_change_pct"]) <= 2.0 + 1e-9
        and float(row["task23_gain_vs_e0_pct"]) >= -1e-9
        and float(row["recording_nondegraded_fraction"]) >= 0.70
        and int(row["new_gt45"]) == 0
        and int(row["new_gt10"]) <= int(row["removed_gt10"])
    )


def _task1_lookup(task1_array: pd.DataFrame, variant: str, array_label: str, pid: str) -> pd.Series:
    hit = task1_array[(task1_array["coupling_variant"] == variant) & (task1_array["array_label"] == array_label) & (task1_array["parameter_id"] == pid)]
    if len(hit) != 1:
        raise RuntimeError(f"Task1 metric not unique: {variant}/{array_label}/{pid}")
    return hit.iloc[0]


def select_on_training_arrays(grid: pd.DataFrame, task1_array: pd.DataFrame, variant: str, train_arrays: tuple[str, ...]) -> tuple[dict | None, pd.DataFrame]:
    rows = []
    vg = grid[grid["coupling_variant"] == variant]
    for (pid, a, h), g in vg.groupby(["parameter_id", "assignment_strength", "anchor_strength"], sort=True):
        tg = g[g["array_label"].isin(train_arrays)]
        if set(tg["array_label"].astype(str)) != set(train_arrays):
            continue
        full_each = [full_safety_feasible(r) for _, r in tg.iterrows()]
        t1_each = [task1_array_feasible(_task1_lookup(task1_array, variant, label, pid)) for label in train_arrays]
        rows.append({
            "coupling_variant": variant, "parameter_id": pid, "assignment_strength": float(a), "anchor_strength": float(h),
            "all_training_arrays_full_safety_feasible": bool(all(full_each)),
            "all_training_arrays_task1_hard_gate_feasible": bool(all(t1_each)),
            "all_training_requirements_feasible": bool(all(full_each) and all(t1_each)),
            "min_training_gain_pct": float(tg["overall_gain_vs_e0_pct"].min()),
            "mean_training_gain_pct": float(tg["overall_gain_vs_e0_pct"].mean()),
        })
    table = pd.DataFrame(rows)
    feasible = table[table["all_training_requirements_feasible"].astype(bool)].copy() if len(table) else pd.DataFrame()
    if not len(feasible):
        return None, table
    # Frozen selection rule: smallest feasible lambda, then robustness only as tie-breaker.
    best = feasible.sort_values(["assignment_strength", "min_training_gain_pct", "mean_training_gain_pct"], ascending=[True, False, False], kind="mergesort").iloc[0].to_dict()
    return best, table


def _global_stability_for_variant(grid: pd.DataFrame, task1_array: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    vg = grid[grid["coupling_variant"] == variant]
    for (pid, a, h), g in vg.groupby(["parameter_id", "assignment_strength", "anchor_strength"], sort=True):
        full_each = [full_safety_feasible(r) for _, r in g.iterrows()]
        t1_each = [task1_array_feasible(_task1_lookup(task1_array, variant, label, pid)) for label in ARRAYS]
        rows.append({
            "coupling_variant": variant, "parameter_id": pid, "assignment_strength": float(a), "anchor_strength": float(h),
            "all_three_arrays_full_safety_feasible": bool(all(full_each)),
            "all_three_arrays_task1_hard_gate_feasible": bool(all(t1_each)),
            "all_three_arrays_safety_feasible": bool(all(full_each) and all(t1_each)),
            "min_array_gain_pct": float(g["overall_gain_vs_e0_pct"].min()),
            "mean_array_gain_pct": float(g["overall_gain_vs_e0_pct"].mean()),
            "min_recording_nondegraded_fraction": float(g["recording_nondegraded_fraction"].min()),
            "total_new_gt45": int(g["new_gt45"].sum()),
            "total_new_gt10_minus_removed": int((g["new_gt10"] - g["removed_gt10"]).sum()),
        })
    return pd.DataFrame(rows)


def full_phase(results: Path) -> int:
    task1_decision_path = results / f"{PREFIX}_TASK1_DECISION.json"
    if not task1_decision_path.is_file():
        raise SystemExit("Task1 decision JSON missing")
    task1_decision = json.loads(task1_decision_path.read_text(encoding="utf-8"))
    if not bool(task1_decision.get("advance_to_full", False)):
        decision = {"status": "R8MMV_STOP_TASK1_FAILED", "ready_to_freeze_candidate": False, "development_only": True, "independent_validation_claimed": False}
        (results / f"{PREFIX}_FINAL_DEVELOPMENT_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(decision, indent=2)); return 0

    paths = [results / f"{PREFIX}_TASK1_GRID_TRIALS.csv", results / f"{PREFIX}_TASK1_WINDOW_DIAGNOSTICS.csv", results / f"{PREFIX}_REMAINING_GRID_TRIALS.csv", results / f"{PREFIX}_REMAINING_WINDOW_DIAGNOSTICS.csv"]
    if not all(x.is_file() for x in paths):
        raise SystemExit("Full-stage outputs incomplete")
    task1_trials, task1_diag, rem_trials, rem_diag = [pd.read_csv(x) for x in paths]
    task1_rec, task1_array = build_task1_metrics(task1_trials, task1_diag)
    trials = pd.concat([task1_trials, rem_trials], ignore_index=True)
    diagnostics = pd.concat([task1_diag, rem_diag], ignore_index=True).drop_duplicates(["array_label", "task", "recording_name", "required_time_row_index"], keep="first")
    if len(diagnostics) != 1728:
        raise RuntimeError(f"Expected 1728 unique windows, got {len(diagnostics)}")
    expected_rows = 1728 * len(ALL_VALIDATION_VARIANTS) * len(GRID)
    if len(trials) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} candidate rows, got {len(trials)}")

    rec = build_full_recording_metrics(trials, diagnostics)
    grid, task = summarize_full_grid(trials, diagnostics, rec)
    rec.to_csv(results / f"{PREFIX}_FULL_RECORDING_METRICS.csv", index=False)
    grid.to_csv(results / f"{PREFIX}_FULL_GRID_ARRAY_METRICS.csv", index=False)
    task.to_csv(results / f"{PREFIX}_FULL_GRID_TASK_METRICS.csv", index=False)

    loo_rows, selection_tables, global_tables, summary_rows = [], [], [], []
    for variant in ALL_VALIDATION_VARIANTS:
        for holdout in ARRAYS:
            train = tuple(x for x in ARRAYS if x != holdout)
            selected, table = select_on_training_arrays(grid, task1_array, variant, train)
            table["holdout_array"] = holdout; table["training_arrays"] = "+".join(train)
            selection_tables.append(table)
            if selected is None:
                loo_rows.append({"coupling_variant": variant, "holdout_array": holdout, "training_arrays": "+".join(train), "status": "NO_FEASIBLE_TRAINING_PARAMETER", "holdout_safety_feasible": False})
                continue
            hit = grid[(grid["coupling_variant"] == variant) & (grid["array_label"] == holdout) & (grid["parameter_id"] == selected["parameter_id"])]
            if len(hit) != 1:
                raise RuntimeError("Selected parameter not unique on holdout")
            row = hit.iloc[0]; t1row = _task1_lookup(task1_array, variant, holdout, str(selected["parameter_id"]))
            hold_full, hold_t1 = full_safety_feasible(row), task1_array_feasible(t1row)
            loo_rows.append({
                "coupling_variant": variant, "holdout_array": holdout, "training_arrays": "+".join(train),
                "status": "PARAMETER_SELECTED_ON_TRAINING_ONLY",
                "parameter_id": selected["parameter_id"], "assignment_strength": selected["assignment_strength"], "anchor_strength": selected["anchor_strength"],
                "training_task1_hard_gate_enforced": True,
                "training_min_gain_pct": selected["min_training_gain_pct"], "training_mean_gain_pct": selected["mean_training_gain_pct"],
                "holdout_overall_gain_pct": float(row["overall_gain_vs_e0_pct"]),
                "holdout_task1_relative_change_pct": float(row["task1_relative_change_pct"]),
                "holdout_task4_relative_change_pct": float(row["task4_relative_change_pct"]),
                "holdout_task23_gain_pct": float(row["task23_gain_vs_e0_pct"]),
                "holdout_recording_nondegraded_fraction": float(row["recording_nondegraded_fraction"]),
                "holdout_task1_recording_nondegraded_fraction": float(t1row["recording_nondegraded_fraction"]),
                "holdout_new_gt10": int(row["new_gt10"]), "holdout_removed_gt10": int(row["removed_gt10"]),
                "holdout_new_gt45": int(row["new_gt45"]), "holdout_removed_gt45": int(row["removed_gt45"]),
                "holdout_full_safety_feasible": bool(hold_full), "holdout_task1_hard_gate_feasible": bool(hold_t1),
                "holdout_safety_feasible": bool(hold_full and hold_t1),
            })

        stability = _global_stability_for_variant(grid, task1_array, variant)
        global_tables.append(stability)
        feasible = stability[stability["all_three_arrays_safety_feasible"].astype(bool)].copy()
        strengths = sorted(feasible["assignment_strength"].astype(float).tolist())
        adjacent = _adjacent_values(strengths); longest = _longest_adjacent_run(strengths)
        variant_loo = [r for r in loo_rows if r["coupling_variant"] == variant]
        loo_success = int(sum(bool(r.get("holdout_safety_feasible", False)) for r in variant_loo))
        best = None
        if len(feasible):
            best = feasible.sort_values(["assignment_strength", "min_array_gain_pct", "mean_array_gain_pct"], ascending=[True, False, False], kind="mergesort").iloc[0].to_dict()
        holdout_gains = [float(r.get("holdout_overall_gain_pct")) for r in variant_loo if r.get("holdout_overall_gain_pct") is not None]
        ready = bool(variant in FINAL_CANDIDATE_VARIANTS and loo_success == 3 and len(feasible) > 0 and adjacent)
        summary_rows.append({
            "coupling_variant": variant, "eligible_final_method": bool(variant in FINAL_CANDIDATE_VARIANTS),
            "reference_only": bool(variant in REFERENCE_VARIANTS),
            "loao_successful_holdouts": loo_success,
            "globally_safety_feasible_parameter_count": int(len(feasible)),
            "globally_safety_feasible_strengths": json.dumps(strengths),
            "has_adjacent_globally_feasible_parameter_pair": bool(adjacent),
            "longest_adjacent_globally_feasible_run": int(longest),
            "minimum_global_feasible_parameter": float(best["assignment_strength"]) if best else np.nan,
            "min_array_gain_at_minimum_feasible_pct": float(best["min_array_gain_pct"]) if best else np.nan,
            "mean_array_gain_at_minimum_feasible_pct": float(best["mean_array_gain_pct"]) if best else np.nan,
            "min_loao_holdout_overall_gain_pct": min(holdout_gains) if holdout_gains else np.nan,
            "mean_loao_holdout_overall_gain_pct": float(np.mean(holdout_gains)) if holdout_gains else np.nan,
            "ready_to_freeze_candidate": ready,
        })

    loo = pd.DataFrame(loo_rows); loo.to_csv(results / f"{PREFIX}_FULL_LEAVE_ONE_ARRAY_OUT.csv", index=False)
    pd.concat(selection_tables, ignore_index=True).to_csv(results / f"{PREFIX}_FULL_LOAO_TRAINING_SELECTIONS.csv", index=False)
    global_table = pd.concat(global_tables, ignore_index=True)
    global_table.to_csv(results / f"{PREFIX}_FULL_GLOBAL_STABILITY_GRID.csv", index=False)
    summary = pd.DataFrame(summary_rows)

    # Robustness-first method ranking. Performance breaks ties only after
    # LOAO and plateau width; no result-dependent formula creation is allowed.
    eligible = summary[summary["eligible_final_method"].astype(bool)].copy()
    eligible = eligible.sort_values(
        ["ready_to_freeze_candidate", "loao_successful_holdouts", "longest_adjacent_globally_feasible_run", "globally_safety_feasible_parameter_count", "min_loao_holdout_overall_gain_pct", "min_array_gain_at_minimum_feasible_pct", "mean_array_gain_at_minimum_feasible_pct", "minimum_global_feasible_parameter"],
        ascending=[False, False, False, False, False, False, False, True], kind="mergesort"
    )
    eligible["final_method_rank"] = np.arange(1, len(eligible) + 1)
    summary = summary.merge(eligible[["coupling_variant", "final_method_rank"]], on="coupling_variant", how="left")
    summary.to_csv(results / f"{PREFIX}_FINAL_METHOD_COMPARISON.csv", index=False)

    ready_methods = eligible[eligible["ready_to_freeze_candidate"].astype(bool)]["coupling_variant"].astype(str).tolist()
    recommended_method = ready_methods[0] if ready_methods else None
    recommended_strength = None
    if recommended_method is not None:
        r = eligible[eligible["coupling_variant"] == recommended_method].iloc[0]
        recommended_strength = float(r["minimum_global_feasible_parameter"])
    status = "R8MMV_MULTIMETHOD_READY_TO_FREEZE_CANDIDATE" if ready_methods else (
        "R8MMV_MULTIMETHOD_PROMISING_BUT_NOT_FREEZE" if int(eligible["loao_successful_holdouts"].max()) >= 2 else "R8MMV_MULTIMETHOD_NOT_STABLE_STOP_REVIEW"
    )
    decision = {
        "status": status,
        "ready_to_freeze_candidate": bool(ready_methods),
        "ready_eligible_methods_ranked": ready_methods,
        "recommended_method_if_ready": recommended_method,
        "recommended_assignment_strength_if_ready": recommended_strength,
        "anchor_strength": 0.03,
        "method_selection_rule": "Robustness first: LOAO 3/3 and adjacent global safety plateau are mandatory; wider plateau precedes gain, then smaller feasible lambda.",
        "reference_method_not_eligible_for_freeze": list(REFERENCE_VARIANTS),
        "no_reassignment_no_replacement_rule": True,
        "loao_training_selection_enforces_task1_hard_gate": True,
        "development_only": True,
        "independent_validation_claimed": False,
        "next_if_ready": "Freeze exact winning estimator/method/lambda before one later untouched holdout confirmation; do not tune on that holdout.",
    }
    (results / f"{PREFIX}_FINAL_DEVELOPMENT_DECISION.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# R8-M Final Multi-Method Evidence-Change Validation", "", f"Status: `{status}`", "",
        "## Method comparison", "",
    ]
    for _, r in eligible.iterrows():
        lines.append(
            f"- {int(r['final_method_rank'])}. `{r['coupling_variant']}`: LOAO {int(r['loao_successful_holdouts'])}/3, "
            f"global safe points {int(r['globally_safety_feasible_parameter_count'])}, longest adjacent plateau {int(r['longest_adjacent_globally_feasible_run'])}, "
            f"READY={'YES' if bool(r['ready_to_freeze_candidate']) else 'NO'}"
        )
    lines += ["", f"Recommended method: `{recommended_method}`", f"Recommended lambda: `{recommended_strength}`", "", "D/Shared-log is a reference only and is not eligible for freezing.", "This stage remains mechanism development on the three previously seen arrays; READY only authorizes one untouched holdout confirmation after freezing."]
    (results / f"{PREFIX}_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate final multi-method Evidence-Change validation")
    p.add_argument("--phase", choices=("task1", "full"), required=True)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    return task1_phase(args.results) if args.phase == "task1" else full_phase(args.results)


if __name__ == "__main__":
    raise SystemExit(main())
