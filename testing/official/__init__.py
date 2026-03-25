"""Helpers for loading official model implementations from submodules."""
import os
import sys

_ESM_SUBMODULE = os.path.join(os.path.dirname(__file__), "..", "..", "official", "esm")


def use_esm_submodule():
    """Load esm from official/esm submodule instead of pip.

    EvolutionaryScale's esm package conflicts with fair-esm (both use `import esm`).
    """
    if _ESM_SUBMODULE not in sys.path:
        sys.path.insert(0, _ESM_SUBMODULE)
