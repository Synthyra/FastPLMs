"""Convert isolated official-reference results into compact golden bundles.

The converter never loads a model or imports an upstream package. It accepts
only the normalized, hash-checkable interchange formats written by the native
reference services, verifies their identity against ``models.toml``, and keeps
the minimum tensors needed for a routine candidate regression.

A sequence golden stores the official strict-FP32 outputs as ``output__*``. They
are the hardware-independent truth a candidate's FP32 run must match. The same
native run's official BF16 outputs are kept as ``official_bf16__*`` so a
candidate's BF16 error against that truth can be bounded by the official error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import torch

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from safetensors.torch import load_file

from fastplms.digests import file_sha256
from fastplms.json_files import compact_json
from fastplms.registry import ModelRegistry, ModelSpec, get_model_registry
from tools.goldens.bundle import (
    GoldenBundleRecord,
    GoldenError,
    validate_golden_bundle,
    write_golden_bundle,
)
from tools.tensor_digests import raw_tensor_sha256, tensor_bytes


NativeResultKind = Literal["sequence", "structure"]
_SEQUENCE_REQUIRED_TENSORS = frozenset(
    {"residue_mask", "output__last_hidden_state"}
)
_SEQUENCE_PRECISIONS = ("bf16", "fp32")
OFFICIAL_BF16_PREFIX = "official_bf16__"
_MAX_GOLDEN_TENSOR_BYTES = 64 * 1024 * 1024
_DPLM2_3B_GENERATION_LIMITATION = {
    "status": "official_unavailable",
    "public_method": "EsmForDPLM.generate",
    "exception_type": "TypeError",
    "reason": (
        "The checkpoint-selected EsmForDPLM sampler uses tokenizer.cls_token_id "
        "as bos_id, but the pinned DPLM2 tokenizer defines no cls_token_id."
    ),
}
_EXPECTED_GENERATION_LIMITATIONS = {
    "dplm2_3b": _DPLM2_3B_GENERATION_LIMITATION,
}


@dataclass(frozen=True, slots=True)
class NativeGoldenRecord:
    """Paths and immutable digests for one converted official result."""

    model_id: str
    metadata_path: Path
    tensors_path: Path
    bundle: GoldenBundleRecord

    def manifest_declaration(self, repository_root: Path) -> str:
        """Return the exact TOML declaration for files under ``tests/goldens``."""

        root = repository_root.resolve()
        try:
            metadata_path = self.metadata_path.resolve().relative_to(root).as_posix()
            tensors_path = self.tensors_path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise GoldenError("Golden output is outside the repository.") from error
        expected_metadata = f"tests/goldens/{self.model_id}.json"
        expected_tensors = f"tests/goldens/{self.model_id}.safetensors"
        if (metadata_path, tensors_path) != (expected_metadata, expected_tensors):
            raise GoldenError(
                "Only validated outputs under tests/goldens can be declared in models.toml."
            )
        metadata = f"{metadata_path}=sha256:{self.bundle.metadata_sha256}"
        tensors = f"{tensors_path}=sha256:{self.bundle.tensors_sha256}"
        return (
            "official_golden = { "
            f'metadata = "{metadata}", tensors = "{tensors}"'
            " }"
        )


@dataclass(frozen=True, slots=True)
class GoldenMatrixEntry:
    """Manifest-derived paths and readiness state for one check-tier golden."""

    model_id: str
    family: str
    kind: NativeResultKind
    reference_container: str
    request_path: Path
    native_result_path: Path
    native_ready: bool
    metadata_path: Path
    tensors_path: Path
    converted_ready: bool
    declared: bool

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON representation for remote orchestration."""

        return {
            "converted_ready": self.converted_ready,
            "declared": self.declared,
            "family": self.family,
            "kind": self.kind,
            "metadata_path": self.metadata_path.as_posix(),
            "model_id": self.model_id,
            "native_ready": self.native_ready,
            "native_result_path": self.native_result_path.as_posix(),
            "reference_container": self.reference_container,
            "request_path": self.request_path.as_posix(),
            "tensors_path": self.tensors_path.as_posix(),
        }


def check_tier_specs(registry: ModelRegistry) -> tuple[ModelSpec, ...]:
    """Return every checkpoint whose family declares the check tier."""

    return tuple(spec for spec in registry.values() if "check" in spec.family.test_tiers)


def _structure_result_paths(native_root: Path, spec: ModelSpec) -> tuple[Path, ...]:
    """Return the canonical structure bundle, preferring BF16-compute leaves."""

    root = native_root / "structure" / "results" / "reference" / spec.id
    paths: list[Path] = []
    if (root / "metadata.json").is_file() and (root / "bundle.safetensors").is_file():
        paths.append(root)
    if root.is_dir():
        paths.extend(
            path.parent
            for path in sorted(root.rglob("bundle.safetensors"))
            if path.parent != root and (path.parent / "metadata.json").is_file()
        )
    preferred = [path for path in paths if path.name in {"bf16", "bfloat16"}]
    if len(preferred) == 1:
        return tuple(preferred)
    return tuple(paths)


def _matrix_native_result_path(native_root: Path, spec: ModelSpec) -> Path:
    if spec.family.tokenizer_mode != "structure":
        return native_root / "results" / spec.id
    available = _structure_result_paths(native_root, spec)
    if len(available) == 1:
        return available[0]
    return native_root / "structure" / "results" / "reference" / spec.id


def golden_generation_matrix(
    registry: ModelRegistry,
    native_root: Path,
    output_root: Path,
) -> tuple[GoldenMatrixEntry, ...]:
    """Describe the generation path for every check-tier manifest entry."""

    entries: list[GoldenMatrixEntry] = []
    for spec in check_tier_specs(registry):
        kind: NativeResultKind = (
            "structure" if spec.family.tokenizer_mode == "structure" else "sequence"
        )
        if kind == "sequence":
            request_path = (
                native_root
                / "requests"
                / spec.family.reference_container
                / f"{spec.id}.json"
            )
            native_result_path = _matrix_native_result_path(native_root, spec)
            native_tensor_names = tuple(
                f"{precision}.safetensors" for precision in _SEQUENCE_PRECISIONS
            )
        else:
            request_path = (
                native_root
                / "structure"
                / "requests"
                / spec.family.reference_container
                / f"{spec.id}.json"
            )
            native_result_path = _matrix_native_result_path(native_root, spec)
            native_tensor_names = ("bundle.safetensors",)
        metadata_path = output_root / f"{spec.id}.json"
        tensors_path = output_root / f"{spec.id}.safetensors"
        entries.append(
            GoldenMatrixEntry(
                model_id=spec.id,
                family=spec.family.id,
                kind=kind,
                reference_container=spec.family.reference_container,
                request_path=request_path,
                native_result_path=native_result_path,
                native_ready=(native_result_path / "metadata.json").is_file()
                and all((native_result_path / name).is_file() for name in native_tensor_names),
                metadata_path=metadata_path,
                tensors_path=tensors_path,
                converted_ready=metadata_path.is_file() and tensors_path.is_file(),
                declared=spec.official_golden is not None,
            )
        )
    return tuple(entries)


def missing_check_golden_ids(registry: ModelRegistry) -> tuple[str, ...]:
    """Report undeclared check-tier goldens without treating partial work as complete."""

    return tuple(
        spec.id for spec in check_tier_specs(registry) if spec.official_golden is None
    )


def require_complete_check_goldens(registry: ModelRegistry) -> None:
    """Fail with the exact undeclared list instead of accepting partial coverage."""

    missing = missing_check_golden_ids(registry)
    if missing:
        raise GoldenError(
            "Check-tier official goldens are incomplete: " + ", ".join(missing) + "."
        )


def _tensor_set_fingerprint(tensors: Mapping[str, torch.Tensor]) -> str:
    # tensors: (...) one tensor per name, any shape
    if not tensors:
        raise GoldenError("A native golden input fingerprint requires tensors.")
    digest = hashlib.sha256()
    for name in sorted(tensors):
        T = tensors[name].detach().to(device="cpu").contiguous()  # (...)
        digest.update(
            compact_json(
                {"dtype": str(T.dtype), "name": name, "shape": list(T.shape)}
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(tensor_bytes(T))
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_compact(tensors: Mapping[str, torch.Tensor], *, model_id: str) -> None:
    # tensors: (...) one tensor per name, any shape
    size = sum(T.numel() * T.element_size() for T in tensors.values())
    if size > _MAX_GOLDEN_TENSOR_BYTES:
        raise GoldenError(
            f"{model_id}: compact golden tensors require {size} bytes, exceeding the "
            f"{_MAX_GOLDEN_TENSOR_BYTES}-byte limit. Add an explicit deterministic "
            "projection instead of committing a large fixture."
        )


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GoldenError(f"Unable to read native result metadata: {path}.") from error
    if not isinstance(value, dict):
        raise GoldenError(f"Native result metadata must be a JSON object: {path}.")
    return value


def _environment(raw: object, *, model_id: str) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise GoldenError(
            f"{model_id}: native result has no environment record; regenerate it in the "
            "pinned reference container."
        )
    environment: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise GoldenError(f"{model_id}: native environment contains an invalid key.")
        if isinstance(value, str):
            normalized = value
        elif value is None:
            normalized = "null"
        elif isinstance(value, (bool, int, float)):
            normalized = json.dumps(value, separators=(",", ":"))
        elif isinstance(value, (Mapping, list, tuple)):
            normalized = compact_json(value)
        else:
            raise GoldenError(
                f"{model_id}: native environment value {key!r} is not serializable."
            )
        if not normalized:
            raise GoldenError(f"{model_id}: native environment value {key!r} is empty.")
        environment[key] = normalized
    return environment


def _expected_files(spec: ModelSpec) -> list[dict[str, str]]:
    return [
        {"algorithm": item.algorithm, "digest": item.digest, "path": item.path}
        for item in spec.official.files
    ]


def _validate_identity(metadata: Mapping[str, Any], spec: ModelSpec) -> None:
    if metadata.get("schema_version") != 1:
        raise GoldenError(f"{spec.id}: unsupported native result schema.")
    if metadata.get("model_id") != spec.id:
        raise GoldenError(f"{spec.id}: native result model identity mismatch.")

    official = metadata.get("official")
    if isinstance(official, Mapping):
        repo_id = official.get("repo_id")
        revision = official.get("revision")
        files = official.get("files")
    else:
        repo_id = metadata.get("reference_repo_id")
        revision = metadata.get("reference_revision")
        files = metadata.get("reference_files")
    if repo_id != spec.official.repo_id or revision != spec.official.revision:
        raise GoldenError(f"{spec.id}: native official checkpoint identity mismatch.")
    if files != _expected_files(spec):
        raise GoldenError(
            f"{spec.id}: native official checkpoint file identities mismatch; regenerate "
            "the native result from the current manifest request."
        )


def _load_sequence_result(
    result_dir: Path,
    spec: ModelSpec,
    registry: ModelRegistry,
    official_bf16_golden_root: Path | None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, str],
    str,
    tuple[dict[str, str], ...],
    dict[str, str],
]:
    metadata_path = result_dir / "metadata.json"
    metadata = _read_metadata(metadata_path)
    _validate_identity(metadata, spec)
    if metadata.get("state_transform") != spec.family.state_transform:
        raise GoldenError(f"{spec.id}: native state transform mismatch.")
    expected_limitation = _EXPECTED_GENERATION_LIMITATIONS.get(spec.id)
    observed_limitation = metadata.get("generation_limitation")
    generation = metadata.get("generation")
    if expected_limitation is None:
        if observed_limitation is not None:
            raise GoldenError(
                f"{spec.id}: undeclared official generation limitation."
            )
        limitations: tuple[dict[str, str], ...] = ()
    else:
        if generation is not None:
            raise GoldenError(
                f"{spec.id}: native result cannot claim generation parity and an "
                "official generation limitation."
            )
        if observed_limitation != expected_limitation:
            raise GoldenError(
                f"{spec.id}: native official generation limitation mismatch."
            )
        limitations = (
            {"capability": "generation", **expected_limitation},
        )
    if (
        spec.family.id in {"dplm", "dplm2"}
        and expected_limitation is None
        and not isinstance(generation, Mapping)
    ):
        raise GoldenError(f"{spec.id}: native result omits required generation parity.")
    precision_keys = metadata.get("precision_tensor_keys")
    if not isinstance(precision_keys, Mapping):
        raise GoldenError(f"{spec.id}: native result has no precision tensor-key record.")
    fp32 = _load_precision_tensors(result_dir, spec, precision_keys, "fp32")  # values: (...)
    environment = _environment(metadata.get("environment"), model_id=spec.id)
    bf16_source_files: dict[str, str] = {}
    if "bf16" in precision_keys:
        if official_bf16_golden_root is not None:
            raise GoldenError(f"{spec.id}: native result already holds its own BF16 run.")
        bf16 = _load_precision_tensors(result_dir, spec, precision_keys, "bf16")  # values: (...)
    else:
        bf16, bf16_environment, bf16_source_files = _prior_official_bf16(  # values: (...)
            spec, registry, official_bf16_golden_root
        )
        environment["official_bf16_environment"] = bf16_environment
    input_names = sorted(name for name in fp32 if name.startswith("input__"))
    for name in (*input_names, "residue_mask"):
        if name not in bf16 or not torch.equal(fp32[name], bf16[name]):
            raise GoldenError(
                f"{spec.id}: native FP32 and BF16 runs used different inputs ({name})."
            )
    output_names = ["output__last_hidden_state"]
    if "output__logits" in fp32:
        output_names.append("output__logits")
    if ("output__logits" in bf16) != ("output__logits" in fp32):
        raise GoldenError(f"{spec.id}: native FP32 and BF16 output heads differ.")
    for name in output_names:
        if fp32[name].dtype != torch.float32:
            raise GoldenError(f"{spec.id}: native FP32 {name} is {fp32[name].dtype}.")
    selected = {name: fp32[name] for name in (*input_names, "residue_mask", *output_names)}
    for name in output_names:
        # The official BF16 output keeps its own dtype; it only sets the candidate's error budget.
        selected[OFFICIAL_BF16_PREFIX + name.removeprefix("output__")] = bf16[name]
    input_tensors = {
        name: selected[name]  # (...)
        for name in (*input_names, "residue_mask")
    }  # values: (...)
    return (  # (...) selected tensors by name, environment, input fingerprint, limitations, BF16 sources
        selected,
        environment,
        _tensor_set_fingerprint(input_tensors),
        limitations,
        bf16_source_files,
    )


def _prior_official_bf16(
    spec: ModelSpec,
    registry: ModelRegistry,
    golden_root: Path | None,
) -> tuple[dict[str, torch.Tensor], str, dict[str, str]]:
    """Take the official BF16 run from the golden ``models.toml`` currently pins.

    A reference whose pinned CUDA build does not exist on the recording platform runs in
    FP32 on the CPU. Its BF16 sample is then the official GPU run already in the declared
    golden, checked against the manifest digests before any tensor is used.
    """

    if golden_root is None:
        raise GoldenError(
            f"{spec.id}: native result has no BF16 run; pass --official-bf16-golden-root."
        )
    if spec.official_golden is None:
        raise GoldenError(f"{spec.id}: no declared golden can supply the official BF16 run.")
    metadata_path = golden_root / f"{spec.id}.json"
    tensors_path = golden_root / f"{spec.id}.safetensors"
    validate_golden_bundle(
        spec,
        registry,
        metadata_path=metadata_path,
        tensors_path=tensors_path,
        declaration=spec.official_golden,
    )
    prior = load_file(tensors_path, device="cpu")  # values: (...)
    if any(name.startswith(OFFICIAL_BF16_PREFIX) for name in prior):
        raise GoldenError(f"{spec.id}: the declared golden is already an FP32 golden.")
    environment = _read_metadata(metadata_path)["environment"]["details"]
    return (  # (...) the prior golden's tensors, its environment, and its file digests
        prior,
        compact_json(environment),
        {
            f"official_bf16/{spec.id}.json": file_sha256(metadata_path),
            f"official_bf16/{spec.id}.safetensors": file_sha256(tensors_path),
        },
    )


def _load_precision_tensors(
    result_dir: Path,
    spec: ModelSpec,
    precision_keys: Mapping[str, object],
    precision: str,
) -> dict[str, torch.Tensor]:
    """Load one native precision's tensors and check them against the metadata record."""

    label = precision.upper()
    tensors_path = result_dir / f"{precision}.safetensors"
    if not tensors_path.is_file():
        raise GoldenError(f"{spec.id}: native {label} result is missing: {tensors_path}.")
    try:
        tensors = load_file(tensors_path, device="cpu")  # values: (...)
    except Exception as error:
        raise GoldenError(f"{spec.id}: unable to load native {label} tensors.") from error
    if precision_keys.get(precision) != sorted(tensors):
        raise GoldenError(f"{spec.id}: native {label} tensor-key contract mismatch.")
    missing = sorted(_SEQUENCE_REQUIRED_TENSORS.difference(tensors))
    if missing or not any(name.startswith("input__") for name in tensors):
        raise GoldenError(
            f"{spec.id}: native {label} result omits required golden tensors: "
            f"{missing or ['input__*']}."
        )
    return tensors  # (...) one tensor per native name


def _load_structure_result(
    result_dir: Path,
    spec: ModelSpec,
) -> tuple[dict[str, torch.Tensor], dict[str, str], str]:
    metadata_path = result_dir / "metadata.json"
    tensors_path = result_dir / "bundle.safetensors"
    metadata = _read_metadata(metadata_path)
    _validate_identity(metadata, spec)
    if metadata.get("producer") != "reference":
        raise GoldenError(f"{spec.id}: only an official reference bundle can become a golden.")
    request_sha256 = metadata.get("request_sha256")
    if (
        not isinstance(request_sha256, str)
        or len(request_sha256) != 64
        or any(character not in "0123456789abcdef" for character in request_sha256)
    ):
        raise GoldenError(f"{spec.id}: structure request fingerprint is invalid.")
    if not tensors_path.is_file():
        raise GoldenError(f"{spec.id}: native structure bundle is missing: {tensors_path}.")
    try:
        tensors = load_file(tensors_path, device="cpu")  # values: (...)
    except Exception as error:
        raise GoldenError(f"{spec.id}: unable to load native structure tensors.") from error
    if metadata.get("tensor_keys") != sorted(tensors):
        raise GoldenError(f"{spec.id}: native structure tensor-key contract mismatch.")
    observed_hashes = {
        name: raw_tensor_sha256(T)  # T: (...)
        for name, T in sorted(tensors.items())
    }
    if metadata.get("tensor_hashes") != observed_hashes:
        raise GoldenError(f"{spec.id}: native structure tensor hash mismatch.")
    if not any(name.startswith("output__") for name in tensors):
        raise GoldenError(f"{spec.id}: native structure result contains no outputs.")
    return (  # (...) tensors by name, environment, request fingerprint
        dict(tensors),
        _environment(metadata.get("environment"), model_id=spec.id),
        request_sha256,
    )


def detect_native_result_kind(result_dir: Path) -> NativeResultKind:
    """Identify one normalized result directory from its immutable files."""

    # Either precision marks a sequence result, so a missing one is reported by name on load.
    has_sequence = any(
        (result_dir / f"{precision}.safetensors").is_file() for precision in _SEQUENCE_PRECISIONS
    )
    has_structure = (result_dir / "bundle.safetensors").is_file()
    if has_sequence and has_structure:
        raise GoldenError(f"Native result holds both sequence and structure tensors: {result_dir}.")
    if not has_sequence and not has_structure:
        raise GoldenError(f"Native result holds neither sequence nor structure tensors: {result_dir}.")
    return "sequence" if has_sequence else "structure"


def convert_native_result(
    spec: ModelSpec,
    registry: ModelRegistry,
    result_dir: Path,
    output_root: Path,
    *,
    generation_command: Sequence[str],
    replace: bool = False,
    official_bf16_golden_root: Path | None = None,
) -> NativeGoldenRecord:
    """Validate and convert one isolated official output without model loading.

    ``official_bf16_golden_root`` holds the currently declared golden, which supplies the
    official BF16 run when the native result recorded only FP32 on the CPU.
    """

    result_dir = result_dir.resolve()
    kind = detect_native_result_kind(result_dir)
    if kind == "sequence":
        if spec.family.tokenizer_mode == "structure":
            raise GoldenError(f"{spec.id}: sequence native result used for a structure model.")
        (
            tensors,
            environment,
            input_fingerprint,
            limitations,
            bf16_source_files,
        ) = _load_sequence_result(result_dir, spec, registry, official_bf16_golden_root)
    else:
        if spec.family.tokenizer_mode != "structure":
            raise GoldenError(f"{spec.id}: structure native result used for a sequence model.")
        tensors, environment, input_fingerprint = _load_structure_result(  # values: (...)
            result_dir, spec
        )
        limitations = ()
        bf16_source_files = {}
    _ensure_compact(tensors, model_id=spec.id)

    native_tensor_files = (
        tuple(
            name
            for name in (f"{precision}.safetensors" for precision in _SEQUENCE_PRECISIONS)
            if (result_dir / name).is_file()
        )
        if kind == "sequence"
        else ("bundle.safetensors",)
    )
    source_files = {
        "native/metadata.json": file_sha256(result_dir / "metadata.json"),
        **{f"native/{name}": file_sha256(result_dir / name) for name in native_tensor_files},
        **bf16_source_files,
    }

    output_root = output_root.resolve()
    metadata_path = output_root / f"{spec.id}.json"
    tensors_path = output_root / f"{spec.id}.safetensors"
    bundle = write_golden_bundle(
        spec,
        registry,
        tensors,
        metadata_path=metadata_path,
        tensors_path=tensors_path,
        generation_command=generation_command,
        environment=environment,
        input_fingerprint=input_fingerprint,
        source_files=source_files,
        limitations=limitations,
        replace=replace,
    )
    return NativeGoldenRecord(
        model_id=spec.id,
        metadata_path=metadata_path,
        tensors_path=tensors_path,
        bundle=bundle,
    )


def _find_native_result(native_root: Path, spec: ModelSpec) -> Path:
    sequence_path = native_root / "results" / spec.id
    if spec.family.tokenizer_mode == "structure":
        existing = list(_structure_result_paths(native_root, spec))
        detail = str(
            native_root / "structure" / "results" / "reference" / spec.id
        )
    else:
        existing = [sequence_path] if sequence_path.is_dir() else []
        detail = str(sequence_path)
    if len(existing) != 1:
        raise GoldenError(
            f"{spec.id}: expected exactly one normalized native result under: {detail}."
        )
    return existing[0]


def _canonical_generation_command(
    spec: ModelSpec,
    official_bf16_golden_root: Path | None,
) -> tuple[str, ...]:
    command = (
        "python",
        "-m",
        "tools.goldens",
        "--native-root",
        "artifacts/reference",
        "--output-root",
        "tests/goldens",
        "--model",
        spec.id,
    )
    if official_bf16_golden_root is None:
        return command
    return (*command, "--official-bf16-golden-root", official_bf16_golden_root.as_posix())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=Path("artifacts/reference"))
    parser.add_argument("--output-root", type=Path, default=Path("tests/goldens"))
    parser.add_argument("--model", action="append", dest="model_ids")
    parser.add_argument(
        "--native-result",
        type=Path,
        help="Explicit result directory; valid only with exactly one --model.",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--official-bf16-golden-root",
        type=Path,
        help="Directory with the declared goldens that supply BF16 for FP32-only native results.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Inspect manifest declaration completeness without converting results.",
    )
    parser.add_argument(
        "--report-missing",
        action="store_true",
        help="Print all undeclared check-tier model IDs after conversion.",
    )
    parser.add_argument(
        "--report-matrix",
        action="store_true",
        help="Print the manifest-wide generation and readiness matrix as JSON.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail after conversion unless every check-tier checkpoint is declared.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Convert selected results and print declarations without editing the manifest."""

    args = _parser().parse_args(argv)
    registry = get_model_registry()
    if args.status_only:
        if args.native_result is not None or args.model_ids or args.replace:
            raise GoldenError(
                "--status-only cannot be combined with conversion inputs or --replace."
            )
        if args.report_missing:
            print(
                json.dumps(
                    {"undeclared_check_goldens": missing_check_golden_ids(registry)}
                )
            )
        if args.report_matrix:
            print(
                json.dumps(
                    {
                        "check_golden_matrix": [
                            entry.as_dict()
                            for entry in golden_generation_matrix(
                                registry,
                                args.native_root,
                                args.output_root,
                            )
                        ]
                    },
                    sort_keys=True,
                )
            )
        if args.require_complete:
            require_complete_check_goldens(registry)
        return 0
    model_ids = args.model_ids or [spec.id for spec in check_tier_specs(registry)]
    unknown = sorted(set(model_ids).difference(registry))
    if unknown:
        raise GoldenError(f"Unknown model IDs: {unknown}.")
    if args.native_result is not None and len(model_ids) != 1:
        raise GoldenError("--native-result requires exactly one --model.")

    records: list[NativeGoldenRecord] = []
    for model_id in model_ids:
        spec = registry[model_id]
        result_dir = (
            args.native_result
            if args.native_result is not None
            else _find_native_result(args.native_root, spec)
        )
        records.append(
            convert_native_result(
                spec,
                registry,
                result_dir,
                args.output_root,
                generation_command=_canonical_generation_command(
                    spec, args.official_bf16_golden_root
                ),
                replace=args.replace,
                official_bf16_golden_root=args.official_bf16_golden_root,
            )
        )

    repository_root = Path.cwd()
    declarable = args.output_root.resolve() == (repository_root / "tests/goldens").resolve()
    for record in records:
        if declarable:
            print(f"[{record.model_id}] {record.manifest_declaration(repository_root)}")
        else:
            print(
                json.dumps(
                    {
                        "metadata_path": str(record.metadata_path),
                        "metadata_sha256": record.bundle.metadata_sha256,
                        "model_id": record.model_id,
                        "tensors_path": str(record.tensors_path),
                        "tensors_sha256": record.bundle.tensors_sha256,
                    },
                    sort_keys=True,
                )
            )
    if args.report_missing:
        print(json.dumps({"undeclared_check_goldens": missing_check_golden_ids(registry)}))
    if args.report_matrix:
        print(
            json.dumps(
                {
                    "check_golden_matrix": [
                        entry.as_dict()
                        for entry in golden_generation_matrix(
                            registry,
                            args.native_root,
                            args.output_root,
                        )
                    ]
                },
                sort_keys=True,
            )
        )
    if args.require_complete:
        require_complete_check_goldens(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
