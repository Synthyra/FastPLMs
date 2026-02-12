import os
import sys
from pathlib import Path


def ensure_boltz_importable() -> None:
    candidates: list[Path] = []

    if "BOLTZ_SRC_DIR" in os.environ:
        env_override = os.environ["BOLTZ_SRC_DIR"]
        if len(env_override.strip()) > 0:
            candidates.append(Path(env_override.strip()))

    module_dir = Path(__file__).resolve().parent
    for parent in [module_dir, *module_dir.parents]:
        candidates.append(parent / "boltz" / "src")

    cwd_dir = Path.cwd().resolve()
    for parent in [cwd_dir, *cwd_dir.parents]:
        candidates.append(parent / "boltz" / "src")

    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return

    assert False, (
        "Could not locate cloned Boltz source directory ('boltz/src'). "
        "Set BOLTZ_SRC_DIR explicitly if needed."
    )
