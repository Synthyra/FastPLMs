"""CUDA-event benchmark runner for FastPLMs.

The primary steady-state path receives pre-tokenized tensors already resident
on the GPU. Startup, compilation, end-to-end embedding, the ESMFold2 learned
projection, and ESMC inference plus projection are measured separately so those
costs cannot be hidden in a single throughput number.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"
HOPPER_SM90_CAPABILITY = (9, 0)
_HOPPER_PRODUCT_PATTERN = re.compile(r"(?<![A-Z0-9])(GH200|H200|H100)(?![A-Z0-9])")


@dataclass(frozen=True)
class BenchmarkCase:
    """One model/backend/input-shape measurement."""

    model: str
    revision: str | None
    auto_class: str
    backend: str
    precision: str
    bf16_execution: str
    mode: str
    batch_size: int
    sequence_length: int
    lengths: tuple[int, ...]


@dataclass(frozen=True)
class MeasurementBlock:
    """Raw CUDA timings and derived throughputs for one measurement block."""

    samples_ms: tuple[float, ...]
    elapsed_ms: float
    forwards: int
    logical_tokens: int
    padded_tokens: int
    logical_tokens_per_second: float
    padded_tokens_per_second: float


def _require_torch(*, require_cuda: bool = True):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("The benchmark image must include PyTorch") from error
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("FastPLMs performance benchmarks require a CUDA GPU")
    return torch


def _version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _nvidia_smi() -> Mapping[str, str]:
    fields = (
        "name",
        "driver_version",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "memory.total",
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    return dict(zip(fields, values, strict=False))


def environment_fingerprint(torch: Any) -> dict[str, Any]:
    """Return the software and accelerator metadata needed to interpret results."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "transformers": _version("transformers"),
        "fastplms": _version("fastplms"),
        "transformer_engine": _version("transformer-engine"),
        "kernels": _version("kernels"),
        "kernels_data": _version("kernels-data"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "nvidia_smi": _nvidia_smi(),
    }


def validate_hopper_sm90_environment(environment: Mapping[str, Any]) -> None:
    """Require an allowed Hopper product for a release-claim benchmark matrix."""

    gpu = environment.get("gpu")
    if not isinstance(gpu, str) or _HOPPER_PRODUCT_PATTERN.search(gpu.upper()) is None:
        raise RuntimeError(
            f"Release-claim benchmarks require an NVIDIA H100, H200, or GH200 GPU; got {gpu!r}."
        )
    capability = environment.get("gpu_capability")
    if capability != list(HOPPER_SM90_CAPABILITY):
        raise RuntimeError(
            f"Release-claim benchmarks require compute capability 9.0; got {capability!r}."
        )


def _sequence(length: int, seed: int) -> str:
    generator = random.Random(seed)
    return "".join(generator.choice(CANONICAL_AAS) for _ in range(length))


def sequences_for_lengths(lengths: Sequence[int], *, special_tokens: int = 2) -> list[str]:
    """Create deterministic proteins whose tokenized lengths approach ``lengths``."""

    return [
        _sequence(max(1, length - special_tokens), 42 + index)
        for index, length in enumerate(lengths)
    ]


def _model_forward(model: Any, model_inputs: Mapping[str, Any]) -> Any:
    arguments = dict(model_inputs)
    parameters = inspect.signature(model.forward).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    optional = {
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
    }
    for name, value in optional.items():
        if accepts_kwargs or name in parameters:
            arguments[name] = value
    return model(**arguments)


def _uses_bf16_autocast(arguments: argparse.Namespace) -> bool:
    return arguments.precision == "bf16" and arguments.bf16_execution == "fp32_parameters_autocast"


def _resolve_bf16_execution(arguments: argparse.Namespace) -> str:
    """Resolve a registered checkpoint policy or validate a local-path override."""

    from fastplms.registry import get_model_registry

    explicit = getattr(arguments, "bf16_execution", None)
    matches = [
        spec
        for spec in get_model_registry().values()
        if arguments.model in {spec.fast.repo_id, spec.official.repo_id}
    ]
    policies = {spec.family.bf16_execution for spec in matches}
    if len(policies) > 1:
        raise ValueError(f"Checkpoint {arguments.model!r} has conflicting BF16 policies")
    if policies:
        manifest_policy = policies.pop()
        if explicit is not None and explicit != manifest_policy:
            raise ValueError(
                f"--bf16-execution={explicit!r} conflicts with the manifest policy "
                f"{manifest_policy!r} for {arguments.model!r}"
            )
        return manifest_policy
    if explicit is None:
        raise ValueError(
            "An unregistered local checkpoint requires an explicit --bf16-execution policy"
        )
    return explicit


def _benchmark_load_dtype(arguments: argparse.Namespace, torch: Any) -> Any:
    """Return the parameter-storage dtype declared by the manifest policy."""

    return torch.float32 if _uses_bf16_autocast(arguments) else torch.bfloat16


def _numeric_context(arguments: argparse.Namespace, torch: Any) -> Any:
    if _uses_bf16_autocast(arguments):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def prepare_inputs(
    model: Any,
    model_id: str | Path,
    lengths: Sequence[int],
    device: Any,
    *,
    revision: str | None,
    local_files_only: bool,
) -> tuple[dict[str, Any], int, int, list[str]]:
    """Prepare static device inputs and return biological/padded token counts."""

    torch = _require_torch(require_cuda=False)
    sequences = sequences_for_lengths(lengths)
    prep_tokens = getattr(getattr(model, "model", None), "prep_tokens", None)
    if prep_tokens is not None and hasattr(prep_tokens, "get_batch_kwargs"):
        batch = prep_tokens.get_batch_kwargs(sequences, device=device)
        model_inputs = {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
            "attention_mask": (batch["sequence_ids"] != -1).long(),
        }
    else:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": local_files_only,
            }
            if revision is not None:
                tokenizer_kwargs["revision"] = revision
            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        max_length = max(lengths)
        model_inputs = dict(
            tokenizer(
                sequences,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
        )
        model_inputs = {
            name: value.to(device, non_blocking=True) for name, value in model_inputs.items()
        }

    parameters = inspect.signature(model.forward).parameters
    if "sequence_id" in parameters and "sequence_id" not in model_inputs:
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("A model requiring sequence_id must expose an attention mask")
        model_inputs["sequence_id"] = attention_mask.to(dtype=torch.bool)

    if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            raise RuntimeError("An encoder-decoder benchmark requires decoder_start_token_id")
        batch_size = int(model_inputs["input_ids"].shape[0])
        model_inputs["decoder_input_ids"] = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            device=device,
            dtype=torch.long,
        )

    # Logical throughput counts biological residues. Attention masks also include
    # model-specific BOS, EOS, and other control tokens, so they cannot provide
    # this count. The deterministic input strings are the shared source of truth.
    logical_tokens = sum(len(sequence) for sequence in sequences)
    padded_tokens = int(model_inputs["input_ids"].numel())
    return model_inputs, logical_tokens, padded_tokens, sequences


def cuda_sample_ms(torch: Any, operation: Callable[[], Any]) -> float:
    """Time one GPU operation with CUDA events."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def warm_until_stable(
    torch: Any,
    operation: Callable[[], Any],
    *,
    window: int = 10,
    tolerance: float = 0.02,
    minimum_samples: int = 20,
    maximum_samples: int = 100,
) -> list[float]:
    """Warm until adjacent timing-window medians differ by less than ``tolerance``."""

    samples: list[float] = []
    for _ in range(maximum_samples):
        samples.append(cuda_sample_ms(torch, operation))
        if len(samples) < max(minimum_samples, 2 * window):
            continue
        previous = statistics.median(samples[-2 * window : -window])
        current = statistics.median(samples[-window:])
        if previous > 0.0 and abs(current - previous) / previous < tolerance:
            return samples
    raise RuntimeError(
        f"CUDA timings did not stabilize within {maximum_samples} warmup forwards; "
        "inspect clocks, thermals, and background workloads"
    )


def measure_blocks(
    torch: Any,
    operation: Callable[[], Any],
    *,
    logical_tokens_per_forward: int,
    padded_tokens_per_forward: int,
    blocks: int = 7,
    minimum_block_ms: float = 250.0,
    minimum_forwards: int = 5,
) -> list[MeasurementBlock]:
    """Collect raw samples in duration-bounded measurement blocks."""

    output: list[MeasurementBlock] = []
    for _ in range(blocks):
        samples: list[float] = []
        elapsed = 0.0
        while len(samples) < minimum_forwards or elapsed < minimum_block_ms:
            duration = cuda_sample_ms(torch, operation)
            samples.append(duration)
            elapsed += duration
        forwards = len(samples)
        elapsed_seconds = elapsed / 1000.0
        logical_tokens = logical_tokens_per_forward * forwards
        padded_tokens = padded_tokens_per_forward * forwards
        output.append(
            MeasurementBlock(
                samples_ms=tuple(samples),
                elapsed_ms=elapsed,
                forwards=forwards,
                logical_tokens=logical_tokens,
                padded_tokens=padded_tokens,
                logical_tokens_per_second=logical_tokens / elapsed_seconds,
                padded_tokens_per_second=padded_tokens / elapsed_seconds,
            )
        )
    return output


def _model_load_source(arguments: argparse.Namespace) -> tuple[str | Path, str | None]:
    """Return physical load coordinates without changing logical report identity."""

    return (
        getattr(arguments, "load_model", arguments.model),
        getattr(arguments, "load_revision", arguments.revision),
    )


def _load_model(arguments: argparse.Namespace, torch: Any) -> tuple[Any, float]:
    import transformers

    arguments.bf16_execution = _resolve_bf16_execution(arguments)
    try:
        auto_class = getattr(transformers, arguments.auto_class)
    except AttributeError as error:
        raise ValueError(f"Unknown Transformers AutoClass {arguments.auto_class!r}") from error
    load_model, load_revision = _model_load_source(arguments)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": arguments.local_files_only,
        "dtype": _benchmark_load_dtype(arguments, torch),
        "device_map": torch.device("cuda"),
        "attn_implementation": arguments.backend,
    }
    if load_revision is not None:
        load_kwargs["revision"] = load_revision
    if arguments.mode == "projection" and arguments.precision != "bf16":
        raise ValueError(
            "Learned projection consumes precomputed BF16 H; use "
            "--mode esmc_projection to compare BF16 and FP8 ESMC inference."
        )
    if arguments.mode == "esmfold2_embed" or arguments.precision != "bf16":
        load_kwargs["esmc_precision"] = arguments.precision
    esmc_load_model = getattr(arguments, "esmc_load_model", None)
    if arguments.mode in {"projection", "esmc_projection"} or esmc_load_model is not None:
        # Representation cases record the folding-core load and ESMC reload
        # separately. Only the end-to-end case reloads ESMC before measurement.
        load_kwargs["load_esmc"] = False

    torch.cuda.synchronize()
    start = time.perf_counter()
    model = auto_class.from_pretrained(load_model, **load_kwargs).eval()
    if arguments.mode == "esmfold2_embed" and esmc_load_model is not None:
        load_esmc = getattr(model, "load_esmc", None)
        if load_esmc is None:
            raise RuntimeError("Local ESMFold2 artifact loading requires model.load_esmc")
        load_esmc(
            str(esmc_load_model),
            precision=arguments.precision,
            device=torch.device("cuda"),
            local_files_only=True,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return model, elapsed_ms


def _select_backend(model: Any, backend: str) -> None:
    """Change one cached model to an advertised backend without reloading it."""

    config = getattr(model, "config", None)
    current = getattr(config, "_attn_implementation", None)
    if current is None:
        current = getattr(config, "attn_backend", None)
    if current == backend:
        return
    setter = getattr(model, "set_attn_implementation", None)
    if setter is None:
        raise RuntimeError(
            f"{type(model).__name__} cannot switch from {current!r} to {backend!r}; "
            "benchmark cases never reload a model to hide this contract gap."
        )
    setter(backend)


def _measure_embedding(
    torch: Any,
    model: Any,
    sequences: Sequence[str],
    batch_size: int,
    arguments: argparse.Namespace,
) -> tuple[float, Any]:
    from fastplms import embed_dataset

    torch.cuda.synchronize()
    start = time.perf_counter()
    with _numeric_context(arguments, torch):
        result = embed_dataset(model, sequences, batch_size=batch_size, pooling=("mean",))
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, result


def _esmfold2_residue_mask(torch: Any, lengths: Sequence[int]) -> Any:
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("ESMFold2 benchmark lengths must be positive")
    length_tensor = torch.tensor(lengths, device="cuda", dtype=torch.long)
    positions = torch.arange(max(lengths), device="cuda")
    return positions.unsqueeze(0) < length_tensor.unsqueeze(1)


def _prepare_esmfold2_inputs(torch: Any, lengths: Sequence[int]) -> tuple[dict[str, Any], int, int]:
    """Create deterministic, preallocated residue tensors for ESMC inference."""

    from fastplms.models.esmfold2.esmfold2_constants_esm3 import (
        SEQUENCE_PAD_TOKEN,
        SEQUENCE_STANDARD_AA_MAX_TOKEN,
        SEQUENCE_STANDARD_AA_MIN_TOKEN,
    )

    residue_mask = _esmfold2_residue_mask(torch, lengths)
    b = len(lengths)
    sequence_length = max(lengths)
    input_ids = torch.full(
        (b, sequence_length), SEQUENCE_PAD_TOKEN, device="cuda", dtype=torch.long
    )
    n_residue_tokens = SEQUENCE_STANDARD_AA_MAX_TOKEN - SEQUENCE_STANDARD_AA_MIN_TOKEN
    for batch_index, length in enumerate(lengths):
        residues = (
            torch.arange(length, device="cuda", dtype=torch.long) + batch_index
        ) % n_residue_tokens
        input_ids[batch_index, :length] = residues + SEQUENCE_STANDARD_AA_MIN_TOKEN
    model_inputs = {
        "input_ids": input_ids,
        "asym_id": torch.zeros_like(input_ids),
        "residue_index": torch.arange(sequence_length, device="cuda").expand(b, -1),
        "mol_type": torch.zeros_like(input_ids),
        "residue_mask": residue_mask,
    }
    return model_inputs, sum(lengths), b * sequence_length


def _run_esmfold2_esmc_projection(model: Any, model_inputs: Mapping[str, Any]) -> Any:
    """Run preallocated residues through ESMC and the learned sequence projection."""

    compute_hidden_states = getattr(model, "_compute_lm_hidden_states", None)
    project = getattr(model, "project_esmc_hidden_states", None)
    if compute_hidden_states is None or project is None:
        raise RuntimeError(
            "ESMC projection mode requires ESMFold2 hidden-state and projection APIs"
        )
    hidden_states = compute_hidden_states(
        model_inputs["input_ids"],
        model_inputs["asym_id"],
        model_inputs["residue_index"],
        model_inputs["mol_type"],
        model_inputs["residue_mask"],
    )
    return project(hidden_states, residue_mask=model_inputs["residue_mask"])


def _measure_projection(
    torch: Any,
    model: Any,
    lengths: Sequence[int],
) -> tuple[float, list[float], list[MeasurementBlock]]:
    project = getattr(model, "project_esmc_hidden_states", None)
    if project is None:
        raise RuntimeError("Projection mode requires model.project_esmc_hidden_states")
    residue_mask = _esmfold2_residue_mask(torch, lengths)
    batch_size = len(lengths)
    sequence_length = max(lengths)
    # H contains all 81 ESMC states and has shape (b, l, 81, 2560).
    H = torch.randn(
        (batch_size, sequence_length, 81, 2560),
        device="cuda",
        dtype=torch.bfloat16,
    )

    def operation() -> Any:
        with torch.inference_mode():
            return project(H, residue_mask=residue_mask)

    first_forward_ms = cuda_sample_ms(torch, operation)
    warmup = warm_until_stable(torch, operation)
    blocks = measure_blocks(
        torch,
        operation,
        logical_tokens_per_forward=sum(lengths),
        padded_tokens_per_forward=batch_size * sequence_length,
    )
    return first_forward_ms, warmup, blocks


def _measure_esmc_projection(
    torch: Any,
    model: Any,
    lengths: Sequence[int],
) -> tuple[float, list[float], list[MeasurementBlock]]:
    model_inputs, logical_tokens, padded_tokens = _prepare_esmfold2_inputs(torch, lengths)

    def operation() -> Any:
        with torch.inference_mode():
            return _run_esmfold2_esmc_projection(model, model_inputs)

    first_forward_ms = cuda_sample_ms(torch, operation)
    warmup = warm_until_stable(torch, operation)
    blocks = measure_blocks(
        torch,
        operation,
        logical_tokens_per_forward=logical_tokens,
        padded_tokens_per_forward=padded_tokens,
    )
    return first_forward_ms, warmup, blocks


def _precision_status_record(model: Any) -> dict[str, Any] | None:
    status = getattr(model, "esmc_precision_status", None)
    if status is None:
        return None
    if isinstance(status, Mapping):
        return dict(status)
    field_names = (
        "requested",
        "resolved",
        "reason",
        "device",
        "transformer_engine_version",
    )
    return {name: getattr(status, name) for name in field_names if hasattr(status, name)}


def run_case(
    arguments: argparse.Namespace,
    *,
    model: Any | None = None,
    load_ms: float | None = None,
    model_reused: bool = False,
) -> dict[str, Any]:
    """Execute one benchmark case and return a JSON-serializable record."""

    if arguments.mode == "projection" and arguments.precision != "bf16":
        raise ValueError(
            "Learned projection consumes precomputed BF16 H; use "
            "esmc_projection for BF16 versus FP8 comparisons."
        )
    arguments.bf16_execution = _resolve_bf16_execution(arguments)
    torch = _require_torch()
    torch.manual_seed(arguments.seed)
    telemetry_before = _nvidia_smi()
    if model is None:
        if load_ms is not None or model_reused:
            raise ValueError("load_ms and model_reused require a preloaded model")
        model, load_ms = _load_model(arguments, torch)
    else:
        _select_backend(model, arguments.backend)
    esmc_reload_ms: float | None = None
    esmc_precision_status: dict[str, Any] | None = None
    if arguments.mode == "esmc_projection":
        reload_esmc = getattr(model, "reload_esmc", None)
        if reload_esmc is None:
            raise RuntimeError("ESMC projection mode requires model.reload_esmc")
        status = getattr(model, "esmc_precision_status", None)
        resolved = getattr(status, "resolved", None)
        if resolved is None and isinstance(status, Mapping):
            resolved = status.get("resolved")
        if getattr(model, "_esmc", None) is None or resolved != arguments.precision:
            torch.cuda.synchronize()
            reload_start = time.perf_counter()
            esmc_load_model = getattr(arguments, "esmc_load_model", None)
            if esmc_load_model is not None and getattr(model, "_esmc", None) is None:
                load_esmc = getattr(model, "load_esmc", None)
                if load_esmc is None:
                    raise RuntimeError("Local ESMFold2 artifact loading requires model.load_esmc")
                load_esmc(
                    str(esmc_load_model),
                    precision=arguments.precision,
                    device=torch.device("cuda"),
                    local_files_only=True,
                )
            else:
                reload_esmc(
                    precision=arguments.precision,
                    device=torch.device("cuda"),
                    local_files_only=arguments.local_files_only,
                )
            torch.cuda.synchronize()
            esmc_reload_ms = (time.perf_counter() - reload_start) * 1000.0
            status = getattr(model, "esmc_precision_status", None)
            resolved = getattr(status, "resolved", None)
            if resolved is None and isinstance(status, Mapping):
                resolved = status.get("resolved")
        if resolved != arguments.precision:
            raise RuntimeError(
                f"Requested ESMC precision {arguments.precision!r}, resolved {resolved!r}"
            )
        esmc_precision_status = _precision_status_record(model)
    elif arguments.mode == "esmfold2_embed":
        esmc_precision_status = _precision_status_record(model)
        resolved = None if esmc_precision_status is None else esmc_precision_status.get("resolved")
        if resolved != arguments.precision:
            raise RuntimeError(
                f"Requested ESMC precision {arguments.precision!r}, resolved {resolved!r}"
            )
    if arguments.lengths:
        lengths = tuple(arguments.lengths)
        if arguments.batch_size != len(lengths):
            raise ValueError("--batch-size must equal the number of values passed to --lengths")
    else:
        lengths = (arguments.sequence_length,) * arguments.batch_size

    case = BenchmarkCase(
        model=arguments.model,
        revision=arguments.revision,
        auto_class=arguments.auto_class,
        backend=arguments.backend,
        precision=arguments.precision,
        bf16_execution=arguments.bf16_execution,
        mode=arguments.mode,
        batch_size=arguments.batch_size,
        sequence_length=max(lengths),
        lengths=lengths,
    )
    torch.cuda.reset_peak_memory_stats()
    compile_ms: float | None = None
    first_forward_ms: float | None = None
    embedding_ms: float | None = None
    warmup_samples: list[float] = []
    blocks: list[MeasurementBlock] = []
    if arguments.mode == "startup":
        pass
    elif arguments.mode == "projection":
        first_forward_ms, warmup_samples, blocks = _measure_projection(torch, model, lengths)
    elif arguments.mode == "esmc_projection":
        first_forward_ms, warmup_samples, blocks = _measure_esmc_projection(torch, model, lengths)
    elif arguments.mode == "esmfold2_embed":
        sequences = sequences_for_lengths(lengths, special_tokens=0)
        embedding_ms, _ = _measure_embedding(
            torch, model, sequences, arguments.batch_size, arguments
        )
    else:
        load_model, load_revision = _model_load_source(arguments)
        model_inputs, logical_tokens, padded_tokens, sequences = prepare_inputs(
            model,
            load_model,
            lengths,
            torch.device("cuda"),
            revision=load_revision,
            local_files_only=arguments.local_files_only,
        )

        def operation() -> Any:
            with torch.inference_mode(), _numeric_context(arguments, torch):
                return _model_forward(model, model_inputs)

        first_forward_ms = cuda_sample_ms(torch, operation)
        if arguments.mode == "compile":
            torch.cuda.synchronize()
            start = time.perf_counter()
            model = torch.compile(model)

            def operation() -> Any:
                with torch.inference_mode(), _numeric_context(arguments, torch):
                    return _model_forward(model, model_inputs)

            operation()
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - start) * 1000.0
        elif arguments.mode == "embed":
            embedding_ms, _ = _measure_embedding(
                torch, model, sequences, arguments.batch_size, arguments
            )

        if arguments.mode in {"steady", "compile"}:
            warmup_samples = warm_until_stable(torch, operation)
            blocks = measure_blocks(
                torch,
                operation,
                logical_tokens_per_forward=logical_tokens,
                padded_tokens_per_forward=padded_tokens,
            )

    torch.cuda.synchronize()
    samples = [sample for block in blocks for sample in block.samples_ms]
    latency = None
    if samples:
        latency = {
            "median_ms": statistics.median(samples),
            "p95_ms": statistics.quantiles(samples, n=100, method="inclusive")[94],
        }
    record = asdict(case)
    record.update(
        {
            "load_ms": load_ms,
            "model_reused": model_reused,
            "esmc_reload_ms": esmc_reload_ms,
            "esmc_precision_status": esmc_precision_status,
            "first_forward_ms": first_forward_ms,
            "compile_ms": compile_ms,
            "embedding_ms": embedding_ms,
            "warmup_samples_ms": warmup_samples,
            "blocks": [asdict(block) for block in blocks],
            "latency": latency,
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "telemetry_before": telemetry_before,
            "telemetry_after": _nvidia_smi(),
        }
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local artifact path or Hub model ID")
    parser.add_argument("--revision")
    parser.add_argument(
        "--auto-class",
        choices=("AutoModel", "AutoModelForMaskedLM", "AutoModelForSeq2SeqLM"),
        default="AutoModelForMaskedLM",
    )
    parser.add_argument(
        "--backend",
        default="sdpa",
        choices=(
            "eager",
            "sdpa",
            "flex_attention",
            "flash_attention_2",
            "flash_attention_3",
        ),
    )
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--bf16-execution",
        choices=("static_parameters", "fp32_parameters_autocast"),
        help=(
            "Required only for unregistered local checkpoints; registered IDs "
            "derive and validate this policy from models.toml."
        ),
    )
    parser.add_argument(
        "--mode",
        default="steady",
        choices=(
            "startup",
            "compile",
            "steady",
            "embed",
            "projection",
            "esmc_projection",
            "esmfold2_embed",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    torch = _require_torch()
    report = {
        "schema_version": 1,
        "environment": environment_fingerprint(torch),
        "results": [run_case(arguments)],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
