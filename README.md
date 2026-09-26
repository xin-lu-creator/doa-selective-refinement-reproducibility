# Reliability-Aware Selective TDOA Refinement (RA-STR)

Minimal reproducibility package for the manuscript **“Reliability-Aware Selective TDOA Refinement for Acoustic Localization.”**

Author: **Xin Lu**  
Release: `v1.0.0-submission`

## Contents

This repository contains only the materials needed to audit the reported results and rerun the final estimators:

- final RA-STR / explicit-null (`E0`) implementation and import closure;
- exhaustive SRP-PHAT comparator;
- adapted MST-TDE comparator;
- frozen LOCATA window manifests;
- final window-level derived results used to verify the reported numbers;
- Hearing-Aid fallback results;
- small final ablation, LOAO, coverage-matched, and sensitivity summaries.

Raw LOCATA audio is not redistributed.

## Nomenclature

The manuscript name **RA-STR** corresponds to the historical frozen identifier `C` in result files. The manuscript null estimator \(E_0\) corresponds to the historical identifier `E0`.

## Quick verification without raw audio

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_reported_results.py
```

The verifier recomputes the headline RMSE values, tail counts, adoption counts, paired bootstrap intervals, and Hearing-Aid fallback checks from the bundled frozen window-level outputs.

## Full LOCATA rerun

Obtain the LOCATA evaluation corpus from its official source. Set `LOCATA_EVAL_ROOT` in `USER_PATHS.bat` or pass the path directly to the Python entry points.

The principal entry points are:

- `code/run_r8m_final_multimethod_validation.py` — RA-STR / \(E_0\) development endpoint and final coupling evaluation.
- `code/run_postfreeze_srpphat_identical_manifest.py` — identical-window SRP-PHAT comparator.
- `code/run_postfreeze_mst_tdoa_identical_manifest.py` — adapted MST-TDE comparator.
- `code/build_v2_hearingaid_task12_manifest.py` and `code/run_v2_hearingaid_task12_once.py` — Hearing-Aid Task 1/2 fallback extension.

The three 576-window array-specific manifests are in `evidence/r8m_frozen_manifests/`.

## Citation

For the TASLP submission, cite the archived `v1.0.0-submission` release using Version DOI [10.5281/zenodo.22970637](https://doi.org/10.5281/zenodo.22970637). The Concept DOI for all versions is [10.5281/zenodo.22970636](https://doi.org/10.5281/zenodo.22970636).

## License

Original code and documentation in this repository are released under the MIT License. Third-party datasets remain subject to their original terms.
