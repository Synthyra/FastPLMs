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
import platform
import shutil
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
from tests.parity.support.state_transforms import (
    transform_parameter_names,
    transform_state,
)

SCHEMA_VERSION = 1
_ADAPTER_PREFIX = "tests.parity.support.reference_adapters."
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
_SEMANTIC_PATHS: dict[str, tuple[str, ...]] = {
    "vocab_size": (
        "config.vocab_size",
        "vocab_size",
        "alphabet_size",
        "embed.num_embeddings",
        "embeddings.word_embeddings.num_embeddings",
        "encoder.sequence_embed.num_embeddings",
    ),
    "d_model": (
        "config.hidden_size",
        "config.d_model",
        "hidden_size",
        "d_model",
        "embed_dim",
        "embed.embedding_dim",
        "embeddings.word_embeddings.embedding_dim",
        "encoder.sequence_embed.embedding_dim",
    ),
    "n_layers": (
        "config.num_hidden_layers",
        "config.num_layers",
        "config.n_layers",
        "num_layers",
        "transformer.blocks",
        "layers",
        "encoder.layer",
        "encoder.block",
    ),
    "n_heads": (
        "config.num_attention_heads",
        "config.num_heads",
        "config.n_heads",
        "attention_heads",
        "transformer.blocks.0.attn.n_heads",
        "layers.0.self_attn.num_heads",
        "encoder.layer.0.attention.self.num_attention_heads",
    ),
    "d_ff": ("config.intermediate_size", "config.d_ff"),
    "layer_norm_epsilon": ("config.layer_norm_eps", "config.layer_norm_epsilon"),
    "max_positions": ("config.max_position_embeddings",),
    "relative_buckets": ("config.relative_attention_num_buckets",),
    "relative_max_distance": ("config.relative_attention_max_distance",),
    "pad_token_id": ("config.pad_token_id", "padding_idx"),
    "bos_token_id": ("config.bos_token_id", "cls_idx"),
    "eos_token_id": ("config.eos_token_id", "eos_idx"),
    "mask_token_id": ("config.mask_token_id", "mask_idx"),
    "token_dropout": ("config.token_dropout", "token_dropout"),
}


def _attribute(root: object, path: str) -> Any:
    current = root
    for part in path.split("."):
        if part.isdigit() and hasattr(current, "__len__") and hasattr(current, "__getitem__"):
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def _semantic_config(model: nn.Module) -> dict[str, Any]:
    roots: list[object] = [model]
    if hasattr(model, "esm3"):
        roots.insert(0, model.esm3)
    result: dict[str, Any] = {}
    for semantic_name, paths in _SEMANTIC_PATHS.items():
        for root in roots:
            for path in paths:
                value = _attribute(root, path)
                if value is None:
                    continue
                if isinstance(value, (nn.ModuleList, list, tuple)):
                    value = len(value)
                if torch.is_tensor(value) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, (str, int, float, bool)):
                    result[semantic_name] = value
                    break
            if semantic_name in result:
                break
    required = {"vocab_size", "d_model", "n_layers", "n_heads"}
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"Official semantic configuration omits {missing}")
    return result


def _tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _environment_metadata() -> dict[str, str]:
    """Describe the isolated native environment without host-specific paths."""

    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if isinstance(name, str) and name:
            distributions[name.lower()] = distribution.version
    return {
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
        ),
        "cuda_runtime": str(torch.version.cuda or "unavailable"),
        "packages": json.dumps(distributions, separators=(",", ":"), sort_keys=True),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


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
    aliases = sorted(
        sorted(names) for names in by_parameter.values() if len(names) > 1
    )
    return {"tensors": tensors, "aliases": aliases}


def _normalize_tokenizer_error(message: str) -> str:
    """Remove a dependency-list difference between Transformers v4 and v5."""

    return message.replace(
        "python, numpy, pytorch or tensorflow object.",
        "python, numpy or pytorch object.",
    )


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
        key: value.tolist() if torch.is_tensor(value) else value
        for key, value in encoded.items()
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
        "special_ids": {
            name: getattr(tokenizer, name, None) for name in _SPECIAL_TOKEN_FIELDS
        },
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
        name: value
        for name, value in encoded.items()
        if name in {"input_ids", "attention_mask"}
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
        request["family"] in {"dplm", "dplm2", "esm2", "esm3"}
        and dtype == torch.bfloat16
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


def _generation_contract(
    model: nn.Module,
    tokenizer: object,
    request: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    """Run a deterministic official DPLM-family public generation call."""

    family = request["family"]
    if family not in {"dplm", "dplm2"}:
        return None
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
    core = getattr(model, "model", model)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": request["model_id"],
        "family": request["family"],
        "reference_repo_id": request["reference_repo_id"],
        "reference_revision": request["reference_revision"],
        "reference_files": request["reference_files"],
        "state_transform": request["state_transform"],
        "environment": _environment_metadata(),
        "semantic_config": _semantic_config(core),
        "state": _state_contract(core, request["state_transform"]),
        "tokenizer": _tokenizer_contract(
            tokenizer,
            request["edge_sequences"],
            request["tokenizer_mode"],
        ),
        "tokenizer_assets": _tokenizer_asset_contract(request),
    }

    device = torch.device("cuda")
    try:
        generation = _generation_contract(model, tokenizer, request, device)
    except OfficialGenerationUnavailable as error:
        metadata["generation_limitation"] = _validated_generation_limitation(
            request,
            error,
        )
    else:
        if request.get("generation_policy") == "official_unavailable":
            raise RuntimeError(
                f"{request['model_id']}: official sampler executed despite an "
                "official_unavailable request."
            )
        if generation is not None:
            metadata["generation"] = generation
    precision_tensors: dict[str, dict[str, torch.Tensor]] = {}
    if request["deep_reference"]:
        precision_tensors["fp32"] = _inference_tensors(
            model, tokenizer, request, device, torch.float32
        )
    precision_tensors["bf16"] = _inference_tensors(
        model, tokenizer, request, device, torch.bfloat16
    )
    metadata["precision_tensor_keys"] = {
        precision: sorted(tensors)
        for precision, tensors in precision_tensors.items()
    }

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / request["model_id"]
    if destination.exists():
        raise FileExistsError(f"Native reference result already exists: {destination}")
    temporary_path = Path(tempfile.mkdtemp(dir=output_root, prefix=".native-"))
    try:
        for precision, tensors in precision_tensors.items():
            save_file(tensors, temporary_path / f"{precision}.safetensors")
        (temporary_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(destination)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    del model, core, precision_tensors
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
            raise ValueError(
                f"Native reference request filename and model ID differ: {path}"
            )
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
