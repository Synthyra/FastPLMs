#!/usr/bin/env python3
"""Analyze ESMC-6B and ESMFold2 checkpoint geometry without running model inputs."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.analysis.esmc_weight_geometry import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
