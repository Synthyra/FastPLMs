"""Execute pinned official models inside their native reference containers.

This module intentionally has no FastPLMs import. It invokes only an official
adapter, applies the independent compliance state transform, and writes a
normalized result that the candidate container can consume later.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from tests.parity.support.reference_adapters import (
    OfficialGenerationUnavailable,
    snapshot_path,
)
from tests.parity.support.semantic_config import transformed_semantic_config
from tests.parity.support.state_transforms import (
    transform_parameter_names,
    transform_preserves_aliases,
    transform_state,
)
from tools.remote.biohub_reference_environment import (
    validate_biohub_reference_environment_evidence,
)
from tools.remote.reference_source_attestation import (
    validate_reference_sources_evidence,
)

SCHEMA_VERSION = 1
_ADAPTER_PREFIX = "tests.parity.support.reference_adapters."
_BIOHUB_REFERENCE_FAMILIES = frozenset({"esm_plusplus", "esm3", "esmfold2"})
_BIOHUB_REFERENCE_SOURCE_NAMES = ("biohub-esm", "biohub-transformers")
_SPECIAL_TOKEN_FIELDS = (
    "pad_token_id",
    "bos_token_id",
    "cls_token_id",
    "eos_token_id",
    "mask_token_id",
    "unk_token_id",
)
_TOKENIZER_SETTINGS = (
    {"padding": True},
    {"padding": "max_length", "truncation": True, "max_length": 12},
    {"padding": True, "truncation": True, "max_length": 5},
)


def _tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _cuda_driver_version() -> str:
    """Read the exact host driver exposed to the reference container."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError("Native compliance requires the exact NVIDIA driver version.") from error
    versions = {
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    }
    if len(versions) != 1:
        raise RuntimeError(
            "Native compliance requires one unambiguous NVIDIA driver version."
        )
    return versions.pop()


def _environment_metadata() -> dict[str, object]:
    """Describe the isolated native environment without host-specific paths."""

    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if isinstance(name, str) and name:
            distributions[name.lower()] = distribution.version
    cuda_properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    uname = platform.uname()
    return {
        "cuda_device": cuda_properties.name if cuda_properties is not None else "unavailable",
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_properties is not None else None
        ),
        "cuda_total_memory": (
            int(cuda_properties.total_memory) if cuda_properties is not None else None
        ),
        "cuda_runtime": str(torch.version.cuda or "unavailable"),
        "cuda_driver": _cuda_driver_version(),
        "packages": json.dumps(distributions, separators=(",", ":"), sort_keys=True),
        "platform_machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "uname": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
    }


def _adapter_reference_sources(
    adapter: Any,
    request: Mapping[str, Any],
) -> dict[str, dict[str, object]] | None:
    """Read and validate an optional named official-source provenance hook."""

    hook = getattr(adapter, "reference_sources", None)
    required = request.get("family") in _BIOHUB_REFERENCE_FAMILIES
    if hook is None:
        if required:
            raise RuntimeError(
                f"{request.get('model_id')}: Biohub adapter omits source attestations."
            )
        return None
    if not callable(hook):
        raise RuntimeError("Official adapter reference-sources hook is not callable.")
    return validate_reference_sources_evidence(
        hook(),
        required_sources=_BIOHUB_REFERENCE_SOURCE_NAMES,
    )


def _adapter_reference_environment(
    adapter: Any,
    request: Mapping[str, Any],
) -> dict[str, object] | None:
    """Read and validate the locked runtime/image evidence for Biohub adapters."""

    hook = getattr(adapter, "reference_environment", None)
    required = request.get("family") in _BIOHUB_REFERENCE_FAMILIES
    if hook is None:
        if required:
            raise RuntimeError(
                f"{request.get('model_id')}: Biohub adapter omits reference environment."
            )
        return None
    if not callable(hook):
        raise RuntimeError("Official adapter reference-environment hook is not callable.")
    try:
        lock_root = Path(os.environ["FASTPLMS_BIOHUB_LOCK_ROOT"])
        contract = Path(os.environ["FASTPLMS_BIOHUB_LOCK_CONTRACT"])
    except KeyError as error:
        raise RuntimeError("Biohub lock validation environment is incomplete.") from error
    return validate_biohub_reference_environment_evidence(
        hook(),
        repository_root=lock_root,
        contract_path=contract,
    )


def _tokenizer_asset_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    files = request.get("tokenizer_files", [])
    if request["tokenizer_mode"] != "tokenizer":
        return {}
    if not files:
        # Some official implementations, including ESM3, define their exact
        # tokenizer vocabulary in pinned upstream source rather than checkpoint
        # assets. The source revision and behavior contract below cover that
        # case; checkpoint-backed tokenizers still hash every declared file.
        return {}
    snapshot = snapshot_path(request["reference_repo_id"], request["reference_revision"])
    result: dict[str, Any] = {}
    for relative_name in files:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe tokenizer asset path: {relative_name!r}")
        path = snapshot.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(f"Official tokenizer asset is missing: {path}")
        content = path.read_bytes()
        result[relative.as_posix()] = {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return result


def _state_contract(model: nn.Module, transform_name: str) -> dict[str, Any]:
    state = transform_state(transform_name, model.state_dict())
    tensors: dict[str, Any] = {}
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise TypeError(f"Official state entry {name!r} is not a tensor")
        tensors[name] = _tensor_digest(value)

    by_parameter: dict[int, set[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        mapped = transform_parameter_names(transform_name, name)
        by_parameter.setdefault(id(parameter), set()).update(mapped)
    aliases = (
        sorted(sorted(names) for names in by_parameter.values() if len(names) > 1)
        if transform_preserves_aliases(transform_name)
        else []
    )
    return {"tensors": tensors, "aliases": aliases}


def _normalize_tokenizer_error(message: str) -> str:
    """Remove a dependency-list difference between Transformers v4 and v5."""

    return message.replace(
        "python, numpy, pytorch or tensorflow object.",
        "python, numpy or pytorch object.",
    ).replace("python, numpy, or pytorch object.", "python, numpy or pytorch object.")


def _token_result(tokenizer: object, sequences: Sequence[str], options: Mapping[str, Any]) -> Any:
    try:
        encoded = tokenizer(sequences, return_tensors="pt", **options)
    except Exception as error:
        return [
            "error",
            type(error).__module__,
            type(error).__qualname__,
            _normalize_tokenizer_error(str(error)),
        ]
    normalized = {
        key: value.tolist() if torch.is_tensor(value) else value for key, value in encoded.items()
    }
    return ["ok", normalized]


def _tokenizer_contract(
    tokenizer: object,
    edge_sequences: Sequence[str],
    tokenizer_mode: str,
) -> dict[str, Any] | None:
    if tokenizer_mode != "tokenizer":
        return None
    return {
        "vocab": tokenizer.get_vocab(),
        "special_ids": {name: getattr(tokenizer, name, None) for name in _SPECIAL_TOKEN_FIELDS},
        "behavior": [
            {"options": options, "result": _token_result(tokenizer, edge_sequences, options)}
            for options in _TOKENIZER_SETTINGS
        ],
    }


def _to_device(values: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in values.items() if torch.is_tensor(value)}


def _prepare_dplm2_inputs(
    sequences: Sequence[str],
    tokenizer: object,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Construct aligned structure and amino-acid tracks for DPLM2."""

    vocabulary = tokenizer.get_vocab()
    required = {
        "<cls_aa>",
        "<eos_aa>",
        "<cls_struct>",
        "<eos_struct>",
        "<pad>",
    }
    required.update({residue for sequence in sequences for residue in sequence})
    missing = sorted(required.difference(vocabulary))
    if missing:
        raise RuntimeError(f"Official DPLM2 tokenizer omits input tokens: {missing}")
    structure_token_id = 50
    if structure_token_id >= len(vocabulary):
        raise RuntimeError("Official DPLM2 vocabulary omits structure token 50")
    track_length = max(map(len, sequences)) + 2
    pad_id = vocabulary["<pad>"]
    input_ids = torch.full(
        (len(sequences), 2 * track_length),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    residue_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_index, sequence in enumerate(sequences):
        residue_count = len(sequence)
        structure = [
            vocabulary["<cls_struct>"],
            *([structure_token_id] * residue_count),
            vocabulary["<eos_struct>"],
        ]
        amino_acids = [
            vocabulary["<cls_aa>"],
            *(vocabulary[residue] for residue in sequence),
            vocabulary["<eos_aa>"],
        ]
        input_ids[row_index, : len(structure)] = torch.tensor(structure, device=device)
        aa_start = track_length
        input_ids[row_index, aa_start : aa_start + len(amino_acids)] = torch.tensor(
            amino_acids,
            device=device,
        )
        residue_mask[row_index, 1 : 1 + residue_count] = True
        residue_mask[
            row_index,
            aa_start + 1 : aa_start + 1 + residue_count,
        ] = True
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_id).long(),
    }, residue_mask


def _prepare_inputs(
    request: Mapping[str, Any],
    tokenizer: object,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    sequences = request["sequences"]
    if request["tokenizer_mode"] == "sequence":
        prepared = dict(tokenizer.get_batch_kwargs(sequences, device=device))
        required = (
            "input_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "sequence_ids",
        )
        missing = [name for name in required if not torch.is_tensor(prepared.get(name))]
        if missing:
            raise RuntimeError(f"Official sequence adapter omits tensors: {missing}")
        # E1's preparer also returns labels and human-readable context records.
        # They are data-loader outputs, not arguments to the public inference
        # computation, and therefore do not belong in a tensor golden.
        inputs = {name: prepared[name] for name in required}
        residue_mask = inputs["sequence_ids"].ge(0)
        return inputs, residue_mask
    if request["family"] == "dplm2":
        return _prepare_dplm2_inputs(sequences, tokenizer, device)

    encoded = _to_device(
        tokenizer(sequences, return_tensors="pt", padding=True),
        device,
    )
    input_ids = encoded["input_ids"]
    residue_mask = encoded["attention_mask"].bool()
    for token_id in getattr(tokenizer, "all_special_ids", ()):
        residue_mask &= input_ids.ne(token_id)
    inputs = {
        name: value for name, value in encoded.items() if name in {"input_ids", "attention_mask"}
    }
    if request["architecture"] == "ESMC":
        inputs["sequence_id"] = encoded["attention_mask"].bool()
    return inputs, residue_mask


def _output_tensors(output: object) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    raw_hidden_states = getattr(output, "hidden_states", None)
    if torch.is_tensor(raw_hidden_states):
        hidden_states = tuple(raw_hidden_states)
    else:
        hidden_states = tuple(raw_hidden_states or ())
    if not hidden_states:
        raise RuntimeError("Official inference omitted hidden states")
    for index, value in enumerate(hidden_states):
        tensors[f"output__hidden_{index:04d}"] = value.detach().cpu().contiguous().clone()
    last_hidden = getattr(output, "last_hidden_state", None)
    if last_hidden is None:
        last_hidden = hidden_states[-1]
    tensors["output__last_hidden_state"] = last_hidden.detach().cpu().contiguous().clone()
    logits = getattr(output, "logits", None)
    if logits is not None:
        tensors["output__logits"] = logits.detach().cpu().contiguous().clone()
    return tensors


@contextlib.contextmanager
def _strict_fp32_matmul():
    """Disable TF32 locally while producing FP32 compliance outputs."""

    try:
        old_fp32_precision = torch.backends.fp32_precision
        old_matmul_precision = torch.backends.cuda.matmul.fp32_precision
        old_cudnn_precision = torch.backends.cudnn.fp32_precision
    except AttributeError:
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        return
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.fp32_precision = old_fp32_precision
        torch.backends.cuda.matmul.fp32_precision = old_matmul_precision
        torch.backends.cudnn.fp32_precision = old_cudnn_precision


def _inference_tensors(
    model: nn.Module,
    tokenizer: object,
    request: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    use_native_autocast = (
        request["family"] in {"dplm", "dplm2", "esm2", "esm3"} and dtype == torch.bfloat16
    )
    if use_native_autocast:
        # These pinned implementations use AMP for native mixed precision.
        # A static BF16 cast breaks their intentional FP32 softmax and rotary
        # operations before the following matrix reduction.
        model = model.to(device=device, dtype=torch.float32).eval()
    else:
        model = model.to(device=device, dtype=dtype).eval()
    torch.manual_seed(int(request["seed"]))
    inputs, residue_mask = _prepare_inputs(request, tokenizer, device)
    if dtype == torch.float32:
        numeric_context = _strict_fp32_matmul()
    elif use_native_autocast:
        numeric_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        numeric_context = contextlib.nullcontext()
    with torch.inference_mode(), numeric_context:
        output = model(**inputs, output_hidden_states=True)
    tensors = {
        f"input__{name}": value.detach().cpu().contiguous().clone()
        for name, value in inputs.items()
    }
    tensors["residue_mask"] = residue_mask.detach().cpu().contiguous().clone()
    tensors.update(_output_tensors(output))
    del output
    return tensors


def _ankh_generation_contract(
    adapter: Any,
    request: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run official ANKH generation from an explicit task prompt.

    ANKH is a T5 checkpoint. Its decoder input is task-specific, so this
    contract deliberately supplies a short decoder prompt instead of shifting
    or otherwise reusing the encoder source tokens.
    """

    load_seq2seq = getattr(adapter, "load_official_seq2seq", None)
    if not callable(load_seq2seq):
        raise RuntimeError("The ANKH reference adapter omits load_official_seq2seq().")
    generation_model, generation_tokenizer = load_seq2seq(
        reference_repo_id=request["reference_repo_id"],
        reference_revision=request["reference_revision"],
        device=device,
        dtype=torch.float32,
    )
    source_text = "M S T N P K"
    decoder_prompt_text = "A C"
    try:
        encoded = _to_device(
            generation_tokenizer(source_text, return_tensors="pt"),
            device,
        )
        prompt = _to_device(
            generation_tokenizer(
                decoder_prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ),
            device,
        )
        prompt_ids = prompt.get("input_ids")
        if not torch.is_tensor(prompt_ids) or prompt_ids.ndim != 2:
            raise RuntimeError("Official ANKH tokenizer returned invalid decoder prompt IDs.")
        decoder_start_token_id = getattr(
            generation_model.config,
            "decoder_start_token_id",
            None,
        )
        if not isinstance(decoder_start_token_id, int):
            raise RuntimeError("Official ANKH config omits decoder_start_token_id.")
        decoder_input_ids = torch.cat(
            (
                prompt_ids.new_full((prompt_ids.shape[0], 1), decoder_start_token_id),
                prompt_ids,
            ),
            dim=1,
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        kwargs = {
            "do_sample": False,
            "max_new_tokens": 4,
            "num_beams": 1,
            "use_cache": True,
        }
        torch.manual_seed(int(request["seed"]))
        torch.cuda.manual_seed_all(int(request["seed"]))
        with torch.inference_mode(), _strict_fp32_matmul():
            generated = generation_model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs,
            )
        if not torch.is_tensor(generated):
            raise RuntimeError("Official ANKH generation did not return output tokens.")
        decoder_fingerprint = _tensor_digest(decoder_input_ids)["sha256"]
        return {
            "interface": "T5ForConditionalGeneration.generate",
            "source_text": source_text,
            "input_ids": encoded["input_ids"].detach().cpu().tolist(),
            "attention_mask": encoded["attention_mask"].detach().cpu().tolist(),
            "decoder_prompt_text": decoder_prompt_text,
            "decoder_prompt_contract": "explicit-task-prompt",
            "decoder_input_ids": decoder_input_ids.detach().cpu().tolist(),
            "decoder_attention_mask": decoder_attention_mask.detach().cpu().tolist(),
            "decoder_input_fingerprint": decoder_fingerprint,
            "kwargs": kwargs,
            "output_tokens": generated.detach().cpu().tolist(),
            "seed": int(request["seed"]),
        }
    finally:
        del generation_model
        gc.collect()
        torch.cuda.empty_cache()


def _generation_contract(
    model: nn.Module | None,
    tokenizer: object,
    request: Mapping[str, Any],
    device: torch.device,
    *,
    adapter: Any | None = None,
) -> dict[str, Any] | None:
    """Run a deterministic official generation call required by the manifest."""

    family = request["family"]
    generation_policy = request.get("generation_policy", "required")
    if generation_policy == "not_applicable":
        return None
    if generation_policy not in {"required", "official_unavailable"}:
        raise RuntimeError(
            f"{request['model_id']}: unknown generation policy {generation_policy!r}."
        )
    if family == "ankh":
        if adapter is None:
            raise RuntimeError("ANKH generation requires the pinned official adapter.")
        return _ankh_generation_contract(adapter, request, device)
    if family not in {"dplm", "dplm2"}:
        raise RuntimeError(
            f"{request['model_id']}: generation is {generation_policy} but the "
            f"{family!r} adapter has no generation contract."
        )
    if model is None:
        raise RuntimeError(f"{request['model_id']}: official generation model is missing.")
    max_iter = 4
    if family == "dplm":
        encoded = tokenizer("ACDEFG", return_tensors="pt")
        input_tokens = encoded["input_ids"].to(device)
        kwargs: dict[str, Any] = {
            "max_iter": max_iter,
            "sampling_strategy": "argmax",
            "disable_resample": True,
        }
    else:
        vocabulary = tokenizer.get_vocab()
        required = (
            "<cls_struct>",
            "<eos_struct>",
            "<cls_aa>",
            "<eos_aa>",
            "A",
        )
        missing = sorted(name for name in required if name not in vocabulary)
        if missing:
            raise RuntimeError(f"Official DPLM2 tokenizer omits generation tokens: {missing}")
        structure = [vocabulary["<cls_struct>"], 50, 50, 50, 50, vocabulary["<eos_struct>"]]
        amino_acids = [vocabulary["<cls_aa>"], *([vocabulary["A"]] * 4), vocabulary["<eos_aa>"]]
        input_tokens = torch.tensor([structure + amino_acids], device=device)
        kwargs = {
            "max_iter": max_iter,
            "sampling_strategy": "argmax",
            "unmasking_strategy": "deterministic",
        }

    model = model.to(device=device, dtype=torch.float32).eval()
    torch.manual_seed(int(request["seed"]))
    torch.cuda.manual_seed_all(int(request["seed"]))
    with torch.inference_mode(), _strict_fp32_matmul():
        generated = model.generate(input_tokens=input_tokens, **kwargs)
    if isinstance(generated, Mapping):
        generated = generated.get("output_tokens")
    if not torch.is_tensor(generated):
        raise RuntimeError("Official DPLM generation did not return output tokens")
    return {
        "input_tokens": input_tokens.detach().cpu().tolist(),
        "kwargs": kwargs,
        "output_tokens": generated.detach().cpu().tolist(),
        "seed": int(request["seed"]),
    }


def _record_generation_contract(
    metadata: dict[str, Any],
    model: nn.Module | None,
    tokenizer: object,
    request: Mapping[str, Any],
    device: torch.device,
    *,
    adapter: Any,
) -> None:
    """Apply the manifest generation policy and fail closed on missing evidence."""

    policy = request.get("generation_policy", "required")
    try:
        generation = _generation_contract(
            model,
            tokenizer,
            request,
            device,
            adapter=adapter,
        )
    except OfficialGenerationUnavailable as error:
        metadata["generation_limitation"] = _validated_generation_limitation(
            request,
            error,
        )
        return

    if policy == "official_unavailable":
        raise RuntimeError(
            f"{request['model_id']}: official sampler executed despite an "
            "official_unavailable request."
        )
    if policy == "required":
        if not isinstance(generation, dict):
            raise RuntimeError(
                f"{request['model_id']}: required official generation evidence is missing."
            )
        metadata["generation"] = generation
        return
    if policy == "not_applicable":
        if generation is not None:
            raise RuntimeError(
                f"{request['model_id']}: not_applicable generation produced evidence."
            )
        return
    raise RuntimeError(f"{request['model_id']}: unknown generation policy {policy!r}.")


def _validated_generation_limitation(
    request: Mapping[str, Any],
    error: OfficialGenerationUnavailable,
) -> dict[str, str]:
    """Accept only the exact limitation declared by a native request."""

    limitation = error.as_record()
    expected = request.get("official_generation_limitation")
    if request.get("generation_policy", "required") != "official_unavailable":
        raise RuntimeError(
            f"{request['model_id']}: official generation is required; "
            "a public sampler failure cannot become a native result."
        ) from error
    if expected != limitation:
        raise RuntimeError(
            f"{request['model_id']}: official generation limitation differs "
            "from the manifest-derived request."
        ) from error
    return limitation


def run_request(request_path: Path, output_root: Path) -> Path:
    """Execute one official request and atomically publish its normalized result."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported native-reference schema in {request_path}")
    adapter_name = request.get("adapter")
    if not isinstance(adapter_name, str) or not adapter_name.startswith(_ADAPTER_PREFIX):
        raise ValueError(f"Invalid official adapter in {request_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Native BF16 compliance requires CUDA")

    adapter = importlib.import_module(adapter_name)
    load_kwargs: dict[str, Any] = {}
    if request.get("oracle_assets"):
        load_kwargs["oracle_assets"] = request["oracle_assets"]
    model, tokenizer = adapter.load_official_model(
        reference_repo_id=request["reference_repo_id"],
        reference_revision=request["reference_revision"],
        device=torch.device("cpu"),
        dtype=None,
        **load_kwargs,
    )
    reference_sources = _adapter_reference_sources(adapter, request)
    reference_environment = _adapter_reference_environment(adapter, request)
    core = getattr(model, "model", model)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": request["model_id"],
        "family": request["family"],
        "reference_repo_id": request["reference_repo_id"],
        "reference_revision": request["reference_revision"],
        "reference_files": request["reference_files"],
        "state_transform": request["state_transform"],
        "environment": _environment_metadata(),
        "semantic_config": transformed_semantic_config(core, request["state_transform"]),
        "state": _state_contract(core, request["state_transform"]),
        "tokenizer": _tokenizer_contract(
            tokenizer,
            request["edge_sequences"],
            request["tokenizer_mode"],
        ),
        "tokenizer_assets": _tokenizer_asset_contract(request),
    }
    if reference_sources is not None:
        metadata["reference_sources"] = reference_sources
    if reference_environment is not None:
        metadata["reference_environment"] = reference_environment

    device = torch.device("cuda")
    # ANKH's native encoder wrapper intentionally has no decoder. Defer its
    # generation contract until encoder inference is complete, then release the
    # encoder before loading the complete official T5 checkpoint.
    defer_generation = (
        request["family"] == "ankh" and request.get("generation_policy", "required") == "required"
    )
    if not defer_generation:
        _record_generation_contract(
            metadata,
            model,
            tokenizer,
            request,
            device,
            adapter=adapter,
        )
    precision_tensors: dict[str, dict[str, torch.Tensor]] = {}
    if request["deep_reference"]:
        precision_tensors["fp32"] = _inference_tensors(
            model, tokenizer, request, device, torch.float32
        )
    precision_tensors["bf16"] = _inference_tensors(
        model, tokenizer, request, device, torch.bfloat16
    )
    metadata["precision_tensor_keys"] = {
        precision: sorted(tensors) for precision, tensors in precision_tensors.items()
    }
    calibration_tensors: dict[str, dict[str, torch.Tensor]] = {}
    calibration_batches = request.get("calibration_batches", [])
    if calibration_batches:
        if request["family"] != "esm_plusplus":
            raise ValueError("Calibration batches are reserved for ESM++/ESMC requests")
        for batch in calibration_batches:
            kind = batch.get("kind")
            cases = batch.get("cases")
            if not isinstance(kind, str) or not isinstance(cases, list) or not cases:
                raise ValueError(f"{request['model_id']}: invalid ESMC calibration batch")
            sequences = [case["sequence"] for case in cases]
            calibration_request = {**request, "sequences": sequences}
            calibration_tensors[kind] = _inference_tensors(
                model,
                tokenizer,
                calibration_request,
                device,
                torch.bfloat16,
            )
        metadata["calibration_batches"] = calibration_batches
        metadata["calibration_tensor_keys"] = {
            kind: sorted(tensors) for kind, tensors in calibration_tensors.items()
        }

    if defer_generation:
        del model, core
        model = None
        core = None
        gc.collect()
        torch.cuda.empty_cache()
        _record_generation_contract(
            metadata,
            None,
            tokenizer,
            request,
            device,
            adapter=adapter,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / request["model_id"]
    if destination.exists():
        raise FileExistsError(f"Native reference result already exists: {destination}")
    temporary_path = Path(tempfile.mkdtemp(dir=output_root, prefix=".native-"))
    try:
        for precision, tensors in precision_tensors.items():
            save_file(tensors, temporary_path / f"{precision}.safetensors")
        calibration_root = temporary_path / "calibration"
        for kind, tensors in calibration_tensors.items():
            calibration_root.mkdir(parents=True, exist_ok=True)
            save_file(tensors, calibration_root / f"{kind}.safetensors")
        (temporary_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(destination)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    if model is not None:
        del model
    if core is not None:
        del core
    del precision_tensors, calibration_tensors
    gc.collect()
    torch.cuda.empty_cache()
    return destination


def _select_requests(
    request_dir: Path,
    model_ids: Sequence[str] | None,
    *,
    deep_only: bool = False,
) -> tuple[Path, ...]:
    """Select explicit request files without importing the FastPLMs manifest."""

    available = {path.stem: path for path in sorted(request_dir.glob("*.json"))}
    if not available:
        raise FileNotFoundError(f"No native reference requests in {request_dir}")
    if model_ids is not None:
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("Native reference model selections must be unique")
        unknown = sorted(set(model_ids).difference(available))
        if unknown:
            raise FileNotFoundError(
                f"Native reference requests are missing selected models: {unknown}"
            )
        candidates = (available[model_id] for model_id in model_ids)
    else:
        candidates = iter(available.values())

    selected: list[Path] = []
    for path in candidates:
        request = json.loads(path.read_text(encoding="utf-8"))
        if request.get("model_id") != path.stem:
            raise ValueError(f"Native reference request filename and model ID differ: {path}")
        if deep_only and request.get("deep_reference") is not True:
            continue
        selected.append(path)
    return tuple(selected)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        dest="model_ids",
        help="Run only this request ID; repeat to select multiple checkpoints.",
    )
    parser.add_argument(
        "--deep-only",
        action="store_true",
        help="Run only manifest-declared deep architecture representatives.",
    )
    arguments = parser.parse_args(argv)
    requests = _select_requests(
        arguments.request_dir,
        arguments.model_ids,
        deep_only=arguments.deep_only,
    )
    for request in requests:
        print(run_request(request, arguments.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
