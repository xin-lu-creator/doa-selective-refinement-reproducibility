@echo off
rem =========================================================
rem File        : 00_SETUP.bat
rem Project     : RA-STR / TASLP manuscript
rem Purpose     : Create a local Python environment for this public release.
rem Input       : requirements.txt
rem Output      : .venv\
rem Used in paper: Reproducibility support only.
rem Main parameters: None.
rem Software    : Windows Command Prompt / PowerShell; Python 3.x.
rem Author      : Xin Lu
rem Last update : 2026-09-26
rem =========================================================
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if not exist ".venv\Scripts\python.exe" exit /b 2
".venv\Scripts\python.exe" -m pip install -r requirements.txt
exit /b %ERRORLEVEL%
