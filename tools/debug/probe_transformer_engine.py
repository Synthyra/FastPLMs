"""Fail-closed Transformer Engine import and FP8 capability probe."""

from __future__ import annotations

import json
import platform
import torch

from importlib.metadata import PackageNotFoundError, version


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
            fp8_status = te.is_fp8_available(return_reason=True)
        except TypeError:
            fp8_status = te.is_fp8_available()
        if isinstance(fp8_status, tuple):
            available = bool(fp8_status[0])
            reason = str(fp8_status[1]) if len(fp8_status) > 1 else ""
        else:
            available = bool(fp8_status)
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
