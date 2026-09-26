@echo off
rem =========================================================
rem File        : 01_VERIFY_REPORTED_RESULTS.bat
rem Project     : RA-STR / TASLP manuscript
rem Purpose     : Recompute the principal manuscript statistics without raw audio.
rem Input       : Bundled frozen result CSV files.
rem Output      : PASS/FAIL messages on stdout.
rem Used in paper: Reproducibility audit of the reported results.
rem Main parameters: Frozen bootstrap seeds and 50,000 draws.
rem Software    : Windows Command Prompt / PowerShell; Python 3.x.
rem Author      : Xin Lu
rem Last update : 2026-09-26
rem =========================================================
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Run 00_SETUP.bat first.
  exit /b 2
)
".venv\Scripts\python.exe" "scripts\verify_reported_results.py"
exit /b %ERRORLEVEL%
