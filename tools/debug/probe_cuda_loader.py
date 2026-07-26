"""Report CUDA library resolution and one required cuBLASLt symbol."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REQUIRED_SYMBOL = "cublasLtGroupedMatrixLayoutInit_internal"
TE_LIBRARY = Path(
    "/opt/venv/lib/python3.12/site-packages/transformer_engine/wheel_lib/libtransformer_engine.so"
)
CUDA_CUBLAS_LT = Path("/usr/local/cuda/lib64/libcublasLt.so.13")
PYTHON_CUBLAS_LT = Path("/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13")


def _output(*command: str) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _symbol_matches(path: Path) -> list[str]:
    output = _output("objdump", "-T", str(path))
    return [line.strip() for line in output.splitlines() if REQUIRED_SYMBOL in line]


def main() -> None:
    paths = (CUDA_CUBLAS_LT, PYTHON_CUBLAS_LT)
    result = {
        "required_symbol": REQUIRED_SYMBOL,
        "transformer_engine_ldd": _output("ldd", str(TE_LIBRARY)).splitlines(),
        "cublas_lt": {
            str(path): {
                "exists": path.exists(),
                "required_symbol_matches": _symbol_matches(path) if path.exists() else [],
            }
            for path in paths
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
