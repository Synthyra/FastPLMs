"""Run the manifest-declared H100 benchmark matrix outside pytest."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastplms.registry import ModelSpec, get_model_registry

from .regression import compare_reports
from .run import _load_model, _require_torch, environment_fingerprint, run_case

PADDING_LENGTHS = (1024, 512, 256, 128, 64, 64, 32, 32)
FIXED_SHAPES = (
    (1, 512, ()),
    (8, 1024, ()),
    (8, 1024, PADDING_LENGTHS),
)
EXHAUSTIVE_BATCH_SIZES = (1, 2, 4, 8)
EXHAUSTIVE_SEQUENCE_LENGTHS = (128, 256, 512, 1024)

SEQUENCE_FORWARD_PROFILE = "sequence_forward"
ESMFOLD2_REPRESENTATION_PROFILE = "esmfold2_representation"
STRUCTURE_STARTUP_PROFILE = "structure_startup"
STRUCTURE_DEDICATED_MODE = "structure"
ESMFOLD2_DEDICATED_MODE = "representation"


def benchmark_auto_class(spec: ModelSpec) -> str:
    """Select the manifest-advertised head measured for one architecture."""

    advertised = set(spec.auto_map)
    if (
        spec.family.id == "ankh"
        or spec.family.tokenizer_mode == "structure"
        or "AutoModelForMaskedLM" not in advertised
    ):
        selected = "AutoModel"
    else:
        selected = "AutoModelForMaskedLM"
    if selected not in advertised:
        raise ValueError(f"{spec.id} does not advertise required benchmark class {selected}")
    return selected


def benchmark_model_key(arguments: SimpleNamespace) -> tuple[str, str, str, str, str]:
    """Return the checkpoint identity that can share one in-memory model."""

    return (
        str(arguments.model),
        str(arguments.revision),
        str(arguments.auto_class),
        str(arguments.precision),
        str(arguments.bf16_execution),
    )


def _default_backend(spec: ModelSpec) -> str:
    if "sdpa" in spec.family.attention:
        return "sdpa"
    if not spec.family.attention:
        raise ValueError(f"{spec.id} does not declare a benchmark backend")
    return spec.family.attention[0]


def _arguments(
    spec: ModelSpec,
    *,
    backend: str,
    mode: str,
    batch_size: int,
    sequence_length: int,
    lengths: tuple[int, ...] = (),
    precision: str = "bf16",
    local_files_only: bool,
    profile: str,
    dedicated_mode: str | None = None,
    claim_eligible: bool | None = None,
    matrix_kind: str = "fixed",
) -> SimpleNamespace:
    if claim_eligible is None:
        claim_eligible = mode in {
            "compile",
            "steady",
            "projection",
            "esmc_projection",
        }
    return SimpleNamespace(
        model=spec.fast.repo_id,
        revision=spec.fast.revision,
        auto_class=benchmark_auto_class(spec),
        backend=backend,
        precision=precision,
        bf16_execution=spec.family.bf16_execution,
        mode=mode,
        batch_size=batch_size,
        sequence_length=sequence_length,
        lengths=lengths,
        local_files_only=local_files_only,
        seed=42,
        output=None,
        suite_profile=profile,
        dedicated_mode=dedicated_mode,
        claim_eligible=claim_eligible,
        matrix_kind=matrix_kind,
    )


def _representative_specs(family: str | None) -> list[ModelSpec]:
    registry = get_model_registry()
    specs = [
        spec
        for spec in registry.values()
        if (spec.is_deep_reference or spec.family.id == "esmfold2")
        and "benchmark" in spec.family.test_tiers
    ]
    if family is not None:
        specs = [spec for spec in specs if spec.family.id == family]
    if not specs:
        raise ValueError(f"No benchmark representative matches family={family!r}")
    return specs


def _axis(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def benchmark_cases(
    *,
    family: str | None,
    quick: bool,
    local_files_only: bool,
) -> Iterator[SimpleNamespace]:
    """Yield the fixed benchmark matrix derived from ``models.toml``."""

    specs = _representative_specs(family)

    if quick:
        spec = specs[0]
        if spec.family.id == "esmfold2":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="projection",
                batch_size=1,
                sequence_length=16,
                local_files_only=local_files_only,
                profile=ESMFOLD2_REPRESENTATION_PROFILE,
                dedicated_mode=ESMFOLD2_DEDICATED_MODE,
            )
            return
        if spec.family.tokenizer_mode == "structure":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="startup",
                batch_size=1,
                sequence_length=1,
                local_files_only=local_files_only,
                profile=STRUCTURE_STARTUP_PROFILE,
                dedicated_mode=STRUCTURE_DEDICATED_MODE,
                claim_eligible=False,
            )
            return
        yield _arguments(
            spec,
            backend=_default_backend(spec),
            mode="steady",
            batch_size=1,
            sequence_length=128,
            local_files_only=local_files_only,
            profile=SEQUENCE_FORWARD_PROFILE,
        )
        return

    for spec in specs:
        if spec.family.id == "esmfold2":
            for batch_size, sequence_length, lengths in FIXED_SHAPES:
                yield _arguments(
                    spec,
                    backend=_default_backend(spec),
                    mode="projection",
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    lengths=lengths,
                    precision="bf16",
                    local_files_only=local_files_only,
                    profile=ESMFOLD2_REPRESENTATION_PROFILE,
                    dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                )
            for precision in ("bf16", "fp8"):
                for backend in spec.family.attention:
                    for batch_size, sequence_length, lengths in FIXED_SHAPES:
                        yield _arguments(
                            spec,
                            backend=backend,
                            mode="esmc_projection",
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            lengths=lengths,
                            precision=precision,
                            local_files_only=local_files_only,
                            profile=ESMFOLD2_REPRESENTATION_PROFILE,
                            dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                        )
                yield _arguments(
                    spec,
                    backend=_default_backend(spec),
                    mode="esmfold2_embed",
                    batch_size=1,
                    sequence_length=512,
                    precision=precision,
                    local_files_only=local_files_only,
                    profile=ESMFOLD2_REPRESENTATION_PROFILE,
                    dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                    claim_eligible=False,
                )
            continue
        if spec.family.tokenizer_mode == "structure":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="startup",
                batch_size=1,
                sequence_length=1,
                local_files_only=local_files_only,
                profile=STRUCTURE_STARTUP_PROFILE,
                dedicated_mode=STRUCTURE_DEDICATED_MODE,
                claim_eligible=False,
            )
            continue

        yield _arguments(
            spec,
            backend=_default_backend(spec),
            mode="startup",
            batch_size=1,
            sequence_length=512,
            local_files_only=local_files_only,
            profile=SEQUENCE_FORWARD_PROFILE,
            claim_eligible=False,
        )
        yield _arguments(
            spec,
            backend=_default_backend(spec),
            mode="embed",
            batch_size=1,
            sequence_length=512,
            local_files_only=local_files_only,
            profile=SEQUENCE_FORWARD_PROFILE,
            claim_eligible=False,
        )
        for backend in spec.family.attention:
            yield _arguments(
                spec,
                backend=backend,
                mode="compile",
                batch_size=1,
                sequence_length=512,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )
            yield _arguments(
                spec,
                backend=backend,
                mode="steady",
                batch_size=1,
                sequence_length=512,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )
            yield _arguments(
                spec,
                backend=backend,
                mode="steady",
                batch_size=8,
                sequence_length=1024,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )
            yield _arguments(
                spec,
                backend=backend,
                mode="steady",
                batch_size=8,
                sequence_length=1024,
                lengths=PADDING_LENGTHS,
                local_files_only=local_files_only,
                profile=SEQUENCE_FORWARD_PROFILE,
            )


def exhaustive_benchmark_cases(
    *,
    family: str | None,
    batch_sizes: Iterable[int] = EXHAUSTIVE_BATCH_SIZES,
    sequence_lengths: Iterable[int] = EXHAUSTIVE_SEQUENCE_LENGTHS,
    local_files_only: bool,
) -> Iterator[SimpleNamespace]:
    """Yield a descriptive all-checkpoint sweep that is never claim-eligible."""

    batches = _axis(batch_sizes, "batch_sizes")
    lengths = _axis(sequence_lengths, "sequence_lengths")
    registry = get_model_registry()
    specs = [
        spec
        for spec in registry.values()
        if "benchmark" in spec.family.test_tiers and (family is None or spec.family.id == family)
    ]
    if not specs:
        raise ValueError(f"No exhaustive benchmark checkpoint matches family={family!r}")

    for spec in specs:
        if spec.family.id == "esmfold2":
            for batch_size in batches:
                for sequence_length in lengths:
                    yield _arguments(
                        spec,
                        backend=_default_backend(spec),
                        mode="projection",
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        local_files_only=local_files_only,
                        profile=ESMFOLD2_REPRESENTATION_PROFILE,
                        dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                        claim_eligible=False,
                        matrix_kind="exhaustive",
                    )
                    for precision in ("bf16", "fp8"):
                        for backend in spec.family.attention:
                            yield _arguments(
                                spec,
                                backend=backend,
                                mode="esmc_projection",
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                precision=precision,
                                local_files_only=local_files_only,
                                profile=ESMFOLD2_REPRESENTATION_PROFILE,
                                dedicated_mode=ESMFOLD2_DEDICATED_MODE,
                                claim_eligible=False,
                                matrix_kind="exhaustive",
                            )
            continue
        if spec.family.tokenizer_mode == "structure":
            yield _arguments(
                spec,
                backend=_default_backend(spec),
                mode="startup",
                batch_size=1,
                sequence_length=1,
                local_files_only=local_files_only,
                profile=STRUCTURE_STARTUP_PROFILE,
                dedicated_mode=STRUCTURE_DEDICATED_MODE,
                claim_eligible=False,
                matrix_kind="exhaustive",
            )
            continue
        for backend in spec.family.attention:
            for batch_size in batches:
                for sequence_length in lengths:
                    yield _arguments(
                        spec,
                        backend=backend,
                        mode="steady",
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        local_files_only=local_files_only,
                        profile=SEQUENCE_FORWARD_PROFILE,
                        claim_eligible=False,
                        matrix_kind="exhaustive",
                    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--family")
    matrix = parser.add_mutually_exclusive_group()
    matrix.add_argument("--quick", action="store_true")
    matrix.add_argument("--exhaustive", action="store_true")
    parser.add_argument(
        "--exhaustive-batch-sizes",
        nargs="+",
        type=int,
        default=EXHAUSTIVE_BATCH_SIZES,
    )
    parser.add_argument(
        "--exhaustive-sequence-lengths",
        nargs="+",
        type=int,
        default=EXHAUSTIVE_SEQUENCE_LENGTHS,
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    torch = _require_torch()
    report: dict[str, Any] = {
        "schema_version": 1,
        "environment": environment_fingerprint(torch),
        "matrix_kind": (
            "exhaustive" if arguments.exhaustive else "quick" if arguments.quick else "fixed"
        ),
        "claim_scope": (
            "descriptive_only"
            if arguments.exhaustive
            else "smoke_only"
            if arguments.quick
            else "validated_h100"
        ),
        "results": [],
    }
    cached_key: tuple[str, str, str, str, str] | None = None
    cached_model: Any | None = None
    if arguments.exhaustive:
        if arguments.baseline is not None:
            raise ValueError("Exhaustive sweeps are descriptive and cannot gate a baseline")
        cases = exhaustive_benchmark_cases(
            family=arguments.family,
            batch_sizes=arguments.exhaustive_batch_sizes,
            sequence_lengths=arguments.exhaustive_sequence_lengths,
            local_files_only=arguments.local_files_only,
        )
    else:
        cases = benchmark_cases(
            family=arguments.family,
            quick=arguments.quick,
            local_files_only=arguments.local_files_only,
        )
    for case in cases:
        key = benchmark_model_key(case)
        reused = cached_model is not None and key == cached_key
        load_ms: float | None = None
        if not reused:
            if cached_model is not None:
                del cached_model
                gc.collect()
                torch.cuda.empty_cache()
            cached_model, load_ms = _load_model(case, torch)
            cached_key = key
        result = run_case(
            case,
            model=cached_model,
            load_ms=load_ms,
            model_reused=reused,
        )
        result.update(
            {
                "suite_profile": case.suite_profile,
                "dedicated_mode": case.dedicated_mode,
                "claim_eligible": case.claim_eligible,
                "matrix_kind": case.matrix_kind,
            }
        )
        report["results"].append(result)
        _write_report(arguments.output, report)
        gc.collect()
        torch.cuda.empty_cache()

    if arguments.baseline is None:
        return 0
    gate = compare_reports(report, _load_report(arguments.baseline))
    gate_output = arguments.gate_output or arguments.output.with_suffix(".gate.json")
    _write_report(gate_output, gate.to_dict())
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
