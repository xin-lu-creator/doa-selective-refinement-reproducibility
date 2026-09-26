# =========================================================
# File        : evaluate_v2_hearingaid_task12.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Evaluate v2 hearingaid task12.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - Derived CSV/JSON evidence or evaluation summaries as defined by the CLI entry point.
#
# Used in paper:
#   - Hearing-Aid fallback-safety extension reported in the manuscript and Supplementary Material.
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


PROTOCOL_ID = "DOA-R8M-HA-TASK12-EXTENSION-20260814"
TASKS = ("task1", "task2")
BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 20_260_814
ZERO_EFFECT_TOL = 1e-12


def rmse(values) -> float:
    x = np.asarray(values, float)
    return float(np.sqrt(np.mean(x * x)))


def effect_direction(value: float, tol: float = ZERO_EFFECT_TOL) -> int:
    """Return -1/0/+1 for improvement/neutral/degradation.

    The primary delta is candidate minus E0, so negative is beneficial.  A
    neutral full-sample effect must never be reported as a leave-one-out
    direction reversal merely because a zero value satisfies ``>= 0``.
    """
    value = float(value)
    if value < -tol:
        return -1
    if value > tol:
        return 1
    return 0


def recording_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, recording), g in frame.groupby(["task", "recording_name"], sort=True):
        e0, candidate = rmse(g.e0_absolute_error_deg), rmse(g.absolute_error_deg)
        rows.append({
            "task": task, "recording_name": recording, "window_count": len(g),
            "e0_rmse_deg": e0, "candidate_rmse_deg": candidate,
            "delta_candidate_minus_e0_deg": candidate - e0,
            "nondegraded": bool(candidate <= e0 + 1e-12), "adoption_count": int(g.adopt.sum()),
        })
    return pd.DataFrame(rows)


def macro(rec: pd.DataFrame) -> tuple[float, float, float]:
    e0 = float(rec.groupby("task").e0_rmse_deg.mean().reindex(TASKS).mean())
    candidate = float(rec.groupby("task").candidate_rmse_deg.mean().reindex(TASKS).mean())
    return e0, candidate, candidate - e0


def bootstrap(rec: pd.DataFrame):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta = np.zeros(BOOTSTRAP_DRAWS)
    for task in TASKS:
        g = rec[rec.task == task].sort_values("recording_name")
        draw = rng.integers(0, len(g), size=(BOOTSTRAP_DRAWS, len(g)))
        delta += (
            g.candidate_rmse_deg.to_numpy()[draw].mean(axis=1)
            - g.e0_rmse_deg.to_numpy()[draw].mean(axis=1)
        ) / len(TASKS)
    return delta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.trials)
    if set(frame.task) != set(TASKS):
        raise SystemExit(f"Both Task 1 and Task 2 are required, found {sorted(set(frame.task))}")
    rec = recording_table(frame)
    rec.to_csv(args.output / "V2_HA12_RECORDING_METRICS.csv", index=False)
    e0, candidate, delta = macro(rec)
    draws = bootstrap(rec)
    ci_low, ci_high = float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
    adopted = frame.adopt.astype(bool)
    improved = adopted & (frame.absolute_error_deg < frame.e0_absolute_error_deg - 1e-12)
    worsened = adopted & (frame.absolute_error_deg > frame.e0_absolute_error_deg + 1e-12)
    adopted_clusters = int(frame.loc[adopted, ["task", "recording_name"]].drop_duplicates().shape[0])
    new45 = int(((frame.e0_absolute_error_deg <= 45) & (frame.absolute_error_deg > 45)).sum())
    removed45 = int(((frame.e0_absolute_error_deg > 45) & (frame.absolute_error_deg <= 45)).sum())
    new10 = int(((frame.e0_absolute_error_deg <= 10) & (frame.absolute_error_deg > 10)).sum())
    removed10 = int(((frame.e0_absolute_error_deg > 10) & (frame.absolute_error_deg <= 10)).sum())
    rec_counts = rec.groupby("task").size().to_dict()
    contributions = rec.copy()
    contributions["contribution_to_primary_delta_deg"] = contributions.apply(
        lambda r: r.delta_candidate_minus_e0_deg / (len(TASKS) * rec_counts[r.task]), axis=1
    )
    contributions["benefit_contribution_deg"] = -contributions.contribution_to_primary_delta_deg
    contributions.to_csv(args.output / "V2_HA12_CLUSTER_CONTRIBUTIONS.csv", index=False)
    primary_direction = effect_direction(delta)
    positive_benefit = contributions.loc[contributions.benefit_contribution_deg > ZERO_EFFECT_TOL, "benefit_contribution_deg"]
    total_benefit = float(positive_benefit.sum())
    max_share = (
        float(positive_benefit.max() / total_benefit)
        if total_benefit > ZERO_EFFECT_TOL else 0.0
    )
    loro_rows = []
    for task, recording in rec[["task", "recording_name"]].itertuples(index=False):
        kept = rec[~((rec.task == task) & (rec.recording_name == recording))]
        _, _, leave_delta = macro(kept)
        leave_direction = effect_direction(leave_delta)
        loro_rows.append({
            "deleted_task": task, "deleted_recording": recording,
            "delta_candidate_minus_e0_deg": leave_delta,
            "full_sample_direction": primary_direction,
            "leave_one_out_direction": leave_direction,
            "direction_reversal": bool(primary_direction != 0 and leave_direction != primary_direction),
        })
    loro = pd.DataFrame(loro_rows)
    loro.to_csv(args.output / "V2_HA12_LORO.csv", index=False)
    adoption_count = int(adopted.sum())
    improvement_rate = float(improved.sum() / adoption_count) if adoption_count else float("nan")
    nondegraded = float(rec.nondegraded.mean())
    strong = (
        adoption_count >= 15 and adopted_clusters >= 5 and delta < 0 and ci_high < 0
        and nondegraded >= .90 and new45 == 0 and not loro.direction_reversal.any()
        and max_share <= .25 and improvement_rate >= .60
    )
    signal = (
        delta < 0 and ci_low <= 0 <= ci_high and adoption_count >= 5 and adopted_clusters >= 3
        and new45 == 0 and nondegraded >= .90
    )
    fail = bool(new45 > 0 or ci_low > 0 or nondegraded < .70)
    status = "STRONG_EFFICACY_PASS" if strong else "EFFICACY_SIGNAL_WITH_SAFETY" if signal else "FAIL" if fail else "SAFETY_ONLY_PASS"
    decision = {
        "protocol_id": PROTOCOL_ID, "status": status,
        "scientific_identity": "outcome-unseen post-freeze cross-task extension within LOCATA",
        "window_count": int(len(frame)), "recording_cluster_count": int(len(rec)),
        "e0_task_balanced_recording_macro_rmse_deg": e0,
        "candidate_task_balanced_recording_macro_rmse_deg": candidate,
        "delta_candidate_minus_e0_deg": delta,
        "bootstrap_replicates": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "ci95_low_deg": ci_low, "ci95_high_deg": ci_high,
        "recording_nondegraded_fraction": nondegraded,
        "adoption_count": adoption_count, "adopted_cluster_count": adopted_clusters,
        "adopted_improved": int(improved.sum()), "adopted_worsened": int(worsened.sum()),
        "adopted_window_improvement_rate": improvement_rate,
        "new_gt10": new10, "removed_gt10": removed10, "new_gt45": new45, "removed_gt45": removed45,
        "primary_delta_direction": primary_direction,
        "zero_effect_tolerance_deg": ZERO_EFFECT_TOL,
        "loro_direction_reversal_count": int(loro.direction_reversal.sum()),
        "maximum_single_cluster_share_of_total_effect": max_share,
        "independent_corpus_claimed": False, "method_or_parameter_tuning_permitted_after_result": False,
    }
    (args.output / "V2_HA12_FINAL_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

