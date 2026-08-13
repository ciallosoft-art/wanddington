"""Central active/legacy artifact paths.

The deployed project is GPT-native. Historical Claude artifacts remain frozen under
``workspace/results/sequential`` and are selected explicitly by the reproduction scripts.
Keeping the two roots here prevents a bare active command from silently consuming Claude-era
memory or overwriting paper artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_GPT_RESULT_DIR = Path(os.environ.get(
    "WADDINGTON_GPT_RESULT_DIR",
    REPO_ROOT / "workspace" / "results" / "gpt" / "gpt_5_6_luna_none",
))
ACTIVE_MEMORY_PATH = Path(os.environ.get(
    "WADDINGTON_MEMORY_PATH",
    ACTIVE_GPT_RESULT_DIR / "experience_memory.json",
))
ACTIVE_SKILL_LIBRARY_PATH = Path(os.environ.get(
    "WADDINGTON_SKILL_LIBRARY_PATH",
    ACTIVE_GPT_RESULT_DIR / "skill_library.json",
))

LEGACY_RESULT_DIR = REPO_ROOT / "workspace" / "results" / "sequential"
LEGACY_MEMORY_PATH = LEGACY_RESULT_DIR / "experience_memory.json"
LEGACY_SKILL_LIBRARY_PATH = LEGACY_RESULT_DIR / "skill_library.json"
