@echo off
rem =========================================================
rem File        : USER_PATHS.bat
rem Project     : RA-STR / TASLP manuscript
rem Purpose     : Launch the USER PATHS reproducibility step.
rem
rem Input       :
rem   - Project-relative configuration and, when required, USER_PATHS.bat / V2_USER_PATHS.bat.
rem
rem Output      :
rem   - Outputs produced by the Python entry point launched by this batch file.
rem
rem Used in paper:
rem   - Reproducibility, evidence generation, comparator evaluation, or verification.
rem
rem Main parameters:
rem   - Frozen manuscript parameters and manifests; see README.md and documentation/CODE_GUIDE.md.
rem
rem Software:
rem   - Windows Command Prompt / PowerShell; invokes project-local Python.
rem
rem Author      : Xin Lu
rem Last update : 2026-09-25
rem =========================================================
rem =====================================================================
rem USER-EDITABLE THIRD-PARTY DATA PATHS
rem Raw LOCATA audio/annotations are NOT included in this archive.
rem Edit LOCATA_EVAL_ROOT before any full acoustic rerun.
rem =====================================================================
set "LOCATA_EVAL_ROOT=C:\PATH\TO\LOCATA\final_evaluation\eval"

rem Bundled frozen R8MMV derived result tables are sufficient for the
rem no-audio final evidence rebuild. Do not change this unless needed.
set "R8MMV_RESULTS_SOURCE=%~dp0data\frozen"
