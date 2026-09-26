# =========================================================
# File        : frozen_hearingaid_task12_extension_protocol.py
# Project     : RA-STR / TASLP manuscript
# Purpose     : Provide the frozen hearingaid task12 extension protocol module used by the reproducibility workflow.
#
# Input       :
#   - Function/CLI arguments, frozen manifests, and project-relative data paths as documented in README.md.
#
# Output      :
#   - In-memory estimates, state objects, or helper values returned to calling code.
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
"""Frozen post-freeze Hearing-Aid Task 1/2 cross-task extension."""
from .frozen_hearingaid_holdout_protocol import *

RELEASE = "r8m-postfreeze-hearingaid-task12-extension-20260814"
PROTOCOL_ID = "DOA-R8M-HA-TASK12-EXTENSION-20260814"
REQUIRED_TASKS = ("task1", "task2")
EXPECTED_CENSUS_RECORDINGS = 26
BOOTSTRAP_SEED = 20260814

