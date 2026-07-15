"""Fail-closed Transformer Engine import and FP8 capability probe."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version

import torch


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> int:
    report: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformer_engine": _package_version("transformer-engine"),
        "transformer_engine_cu12": _package_version("transformer-engine-cu12"),
        "transformer_engine_cu13": _package_version("transformer-engine-cu13"),
        "transformer_engine_torch": _package_version("transformer-engine-torch"),
    }
    try:
        import transformer_engine.pytorch as te

        try:
            result = te.is_fp8_available(return_reason=True)
        except TypeError:
            result = te.is_fp8_available()
        if isinstance(result, tuple):
            available = bool(result[0])
            reason = str(result[1]) if len(result) > 1 else ""
        else:
            available = bool(result)
            reason = ""
        report.update(fp8_available=available, reason=reason)
    except (ImportError, OSError, RuntimeError) as error:
        report.update(
            fp8_available=False,
            reason=f"{type(error).__name__}: {error}",
        )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["fp8_available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
