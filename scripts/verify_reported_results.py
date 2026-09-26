# =========================================================
# File        : verify_reported_results.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Recompute the principal reported statistics from bundled frozen outputs.
#
# Input       : Bundled final window-level CSV files under results/.
# Output      : PASS/FAIL checks printed to stdout; no manuscript data are modified.
# Used in paper: Reproducibility audit of reported RA-STR, E0, SRP-PHAT, MST-TDE, and Hearing-Aid results.
# Main parameters: 50,000 clustered bootstrap draws; frozen seeds 20260814 and 20260913.
# Software    : Python 3.x; numpy, pandas.
# Author      : Xin Lu
# Last update : 2026-09-26
# =========================================================
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARRAYS = ("dicit", "eigenmike", "robothead")
TASKS = ("task1", "task2", "task3", "task4")
KEY = ["array_label", "task", "recording_name", "required_time_row_index"]
DRAWS = 50_000

BASE = ROOT / "results" / "final" / "V2_WP1_EXACT_SRP_E0_C_WINDOWS.csv"
MST = ROOT / "results" / "final" / "V5B_MST_TDOA_WINDOW_RESULTS.csv"
HA12 = ROOT / "results" / "hearingaid" / "V2_HA12_TRIALS.csv"
HA34 = ROOT / "results" / "hearingaid" / "R8M_HA_TRIALS.csv"

EXPECTED = {
    "SRP-PHAT": 7.93409581491433,
    "E0": 7.673258513690722,
    "RA-STR": 7.124795394552808,
    "MST-TDE": 12.815470280422993,
    "C_minus_E0": -0.5484631191379137,
    "C_minus_SRP": -0.8093004203615219,
    "C_minus_MST": -5.690674885870185,
    "C_E0_CI": (-1.0285629442282154, -0.12245348382713724),
    "C_SRP_CI": (-1.973642128915255, 0.22903303257182106),
    "C_MST_CI": (-6.867335417595419, -4.590919169769933),
}


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def recording_table(frame: pd.DataFrame, method: str, col: str) -> pd.DataFrame:
    rows = []
    for (array_label, task, recording), g in frame.groupby(["array_label", "task", "recording_name"], sort=True):
        rows.append({"method": method, "array_label": array_label, "task": task, "recording_name": recording, "rmse_deg": rmse(g[col])})
    return pd.DataFrame(rows)


def macro(rec: pd.DataFrame) -> float:
    table = rec.groupby(["array_label", "task"]).rmse_deg.mean().unstack().loc[list(ARRAYS), list(TASKS)]
    return float(table.to_numpy().mean())


def bootstrap_three(rec: pd.DataFrame, candidate: str, reference: str, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    delta = np.zeros(DRAWS, dtype=float)
    piv = rec.pivot_table(index=["array_label", "task", "recording_name"], columns="method", values="rmse_deg").reset_index()
    for task in TASKS:
        task_rows = piv[piv.task == task]
        recordings = sorted(task_rows.recording_name.unique())
        draw = rng.integers(0, len(recordings), size=(DRAWS, len(recordings)))
        for array_label in ARRAYS:
            g = task_rows[task_rows.array_label == array_label].set_index("recording_name").reindex(recordings)
            c = g[candidate].to_numpy()[draw].mean(axis=1)
            r = g[reference].to_numpy()[draw].mean(axis=1)
            delta += (c - r) / (len(TASKS) * len(ARRAYS))
    return float(np.quantile(delta, .025)), float(np.quantile(delta, .975)), float(np.mean(delta < 0))


def bootstrap_two(rec_a: pd.DataFrame, rec_b: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    a = rec_a.rename(columns={"rmse_deg": "candidate_rmse"})[["array_label", "task", "recording_name", "candidate_rmse"]]
    b = rec_b.rename(columns={"rmse_deg": "reference_rmse"})[["array_label", "task", "recording_name", "reference_rmse"]]
    rec = a.merge(b, on=["array_label", "task", "recording_name"], validate="one_to_one")
    rng = np.random.default_rng(seed)
    ca = {arr: np.zeros(DRAWS) for arr in ARRAYS}
    rb = {arr: np.zeros(DRAWS) for arr in ARRAYS}
    for task in TASKS:
        recordings = sorted(rec.loc[rec.task == task, "recording_name"].unique())
        draw = rng.integers(0, len(recordings), size=(DRAWS, len(recordings)))
        for arr in ARRAYS:
            g = rec[(rec.task == task) & (rec.array_label == arr)].set_index("recording_name").reindex(recordings)
            ca[arr] += g.candidate_rmse.to_numpy()[draw].mean(axis=1) / len(TASKS)
            rb[arr] += g.reference_rmse.to_numpy()[draw].mean(axis=1) / len(TASKS)
    d = np.mean(np.vstack([ca[a] for a in ARRAYS]), axis=0) - np.mean(np.vstack([rb[a] for a in ARRAYS]), axis=0)
    return float(np.quantile(d, .025)), float(np.quantile(d, .975)), float(np.mean(d < 0))


def close(name: str, actual: float, expected: float, tol: float = 5e-9) -> None:
    if not np.isclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f"{name}: {actual} != {expected}")
    print(f"[PASS] {name}: {actual:.9f}")


def task_balanced_macro_ha(frame: pd.DataFrame, col: str) -> float:
    rows = []
    for (task, recording), g in frame.groupby(["task", "recording_name"], sort=True):
        rows.append({"task": task, "recording_name": recording, "rmse_deg": rmse(g[col])})
    rec = pd.DataFrame(rows)
    return float(rec.groupby("task").rmse_deg.mean().mean())


def main() -> int:
    base = pd.read_csv(BASE)
    mst = pd.read_csv(MST)
    if len(base) != 1728 or len(mst) != 1728:
        raise AssertionError(f"Expected 1,728 rows in both final files: base={len(base)}, mst={len(mst)}")
    if base.duplicated(KEY).any() or mst.duplicated(KEY).any():
        raise AssertionError("Duplicate frozen window keys detected")

    recs = {
        "SRP-PHAT": recording_table(base, "SRP-PHAT", "srp_abs_deg"),
        "E0": recording_table(base, "E0", "e0_abs_deg"),
        "RA-STR": recording_table(base, "RA-STR", "c_abs_deg"),
    }
    for m in ("SRP-PHAT", "E0", "RA-STR"):
        close(f"{m} macro RMSE", macro(recs[m]), EXPECTED[m])

    mst_keep = mst[KEY + ["true_azimuth_deg", "absolute_error_deg"]].rename(columns={"true_azimuth_deg": "mst_true", "absolute_error_deg": "mst_abs_deg"})
    merged = base.merge(mst_keep, on=KEY, validate="one_to_one")
    if np.max(np.abs(merged.true_azimuth_deg - merged.mst_true)) > 1e-9:
        raise AssertionError("MST truth mismatch")
    rec_mst = recording_table(merged, "MST-TDE", "mst_abs_deg")
    close("MST-TDE macro RMSE", macro(rec_mst), EXPECTED["MST-TDE"])

    close("RA-STR minus E0", macro(recs["RA-STR"]) - macro(recs["E0"]), EXPECTED["C_minus_E0"])
    close("RA-STR minus SRP-PHAT", macro(recs["RA-STR"]) - macro(recs["SRP-PHAT"]), EXPECTED["C_minus_SRP"])
    close("RA-STR minus MST-TDE", macro(recs["RA-STR"]) - macro(rec_mst), EXPECTED["C_minus_MST"])

    rec_all = pd.concat(recs.values(), ignore_index=True)
    ci = bootstrap_three(rec_all, "RA-STR", "E0", 20260814)
    close("RA-STR/E0 CI low", ci[0], EXPECTED["C_E0_CI"][0], 1e-7)
    close("RA-STR/E0 CI high", ci[1], EXPECTED["C_E0_CI"][1], 1e-7)
    ci = bootstrap_three(rec_all, "RA-STR", "SRP-PHAT", 20260814)
    close("RA-STR/SRP-PHAT CI low", ci[0], EXPECTED["C_SRP_CI"][0], 1e-7)
    close("RA-STR/SRP-PHAT CI high", ci[1], EXPECTED["C_SRP_CI"][1], 1e-7)
    # Existing MST evaluator is expressed MST-minus-C. Negate and reverse its interval for RA-STR-minus-MST-TDE.
    mst_ci = bootstrap_two(rec_mst, recs["RA-STR"], 20260913)
    c_mst_low, c_mst_high = -mst_ci[1], -mst_ci[0]
    close("RA-STR/MST-TDE CI low", c_mst_low, EXPECTED["C_MST_CI"][0], 1e-7)
    close("RA-STR/MST-TDE CI high", c_mst_high, EXPECTED["C_MST_CI"][1], 1e-7)

    adopted = base[base.c_adopt.astype(bool)]
    if len(adopted) != 68 or int((adopted.c_abs_deg < adopted.e0_abs_deg).sum()) != 40 or int((adopted.c_abs_deg > adopted.e0_abs_deg).sum()) != 28:
        raise AssertionError("RA-STR adoption counts differ from the manuscript")
    print("[PASS] RA-STR adoption: 68 total, 40 improved, 28 worsened")

    tails = {
        "SRP-PHAT": ("srp_abs_deg", 261, 4),
        "E0": ("e0_abs_deg", 254, 7),
        "RA-STR": ("c_abs_deg", 246, 4),
        "MST-TDE": ("mst_abs_deg", 480, 27),
    }
    for name, (col, gt10, gt45) in tails.items():
        frame = merged if name == "MST-TDE" else base
        x = frame[col].to_numpy(float)
        got = (int(np.sum(x > 10)), int(np.sum(x > 45)))
        if got != (gt10, gt45):
            raise AssertionError(f"{name} tail counts {got} != {(gt10, gt45)}")
        print(f"[PASS] {name} tails: >10={gt10}, >45={gt45}")

    ha12 = pd.read_csv(HA12)
    ha34 = pd.read_csv(HA34)
    if len(ha12) != 338 or int(ha12.adopt.astype(bool).sum()) != 0:
        raise AssertionError("Hearing-Aid Task 1/2 fallback mismatch")
    close("Hearing-Aid Task 1/2 macro RMSE", task_balanced_macro_ha(ha12, "absolute_error_deg"), 7.931129455557325, 1e-7)
    if len(ha34) != 200 or int(ha34.adopt.astype(bool).sum()) != 0:
        raise AssertionError("Hearing-Aid Task 3/4 fallback mismatch")
    print("[PASS] Hearing-Aid fallback: 338 Task1/2 + 200 Task3/4 windows, zero final adoption")

    # 29 task-recording identities; aggregate three array-specific recording RMSEs per identity.
    rec_c = recs["RA-STR"].rename(columns={"rmse_deg": "c"})
    rec_m = rec_mst.rename(columns={"rmse_deg": "mst"})
    d = rec_c.merge(rec_m, on=["array_label", "task", "recording_name"], validate="one_to_one")
    q = d.groupby(["task", "recording_name"])[["c", "mst"]].mean()
    lower = int((q.c < q.mst).sum())
    if len(q) != 29 or lower != 26:
        raise AssertionError(f"Directional cluster check differs: n={len(q)}, RA-STR lower={lower}")
    print("[PASS] RA-STR lower than adapted MST-TDE on 26/29 task-recording identities")

    print("\nALL REPORTED-VALUE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
