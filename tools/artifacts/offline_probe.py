"""Probe one local Hub artifact without importing the installed FastPLMs package."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _runtime_site_packages() -> tuple[Path, ...]:
    """Return dependency roots without processing editable-install ``.pth`` files."""

    paths: set[Path] = set()
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if path.name in {"site-packages", "dist-packages"} and path.is_dir():
            paths.add(path)
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _add_runtime_site_packages(paths: Iterable[Path]) -> None:
    """Expose installed dependencies passed explicitly to a ``python -I -S`` probe."""

    for path in paths:
        resolved = path.resolve()
        if not resolved.is_dir():
            raise RuntimeError(f"Runtime site-packages path does not exist: {resolved}")
        value = str(resolved)
        if value not in sys.path:
            sys.path.append(value)


def _require_artifact_isolation() -> None:
    """Reject artifact probes that can see an installed FastPLMs package."""

    if not sys.flags.isolated:
        raise RuntimeError("Artifact mode must run under python -I")
    if "fastplms" in sys.modules or importlib.util.find_spec("fastplms") is not None:
        raise RuntimeError("FastPLMs must be absent from sys.path in artifact mode")


def _tensor_digest(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_digest(tensor).encode())
    return digest.hexdigest()


def _normalize(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    import torch

    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _tensor_digest(value),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 6 or id(value) in seen:
        return f"<{type(value).__module__}.{type(value).__qualname__}>"
    seen.add(id(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item, depth=depth + 1, seen=seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, depth=depth + 1, seen=seen) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name), depth=depth + 1, seen=seen)
            for field in dataclasses.fields(value)
        }
    to_tuple = getattr(value, "to_tuple", None)
    if callable(to_tuple):
        return _normalize(to_tuple(), depth=depth + 1, seen=seen)
    values = getattr(value, "__dict__", None)
    if isinstance(values, dict):
        return {
            str(key): _normalize(item, depth=depth + 1, seen=seen)
            for key, item in sorted(values.items())
            if not str(key).startswith("_")
        }
    return repr(value)


def _output_digest(output: Any) -> str:
    encoded = json.dumps(
        _normalize(output),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _semantic_config(config: Any) -> dict[str, Any]:
    """Remove save-location metadata before configuration comparison."""

    values = config.to_dict()
    for name in (
        "_commit_hash",
        "_name_or_path",
        "fastplms_checkpoint_hash",
        "fastplms_checkpoint_repo_id",
        "fastplms_checkpoint_revision",
        "fastplms_model_id",
        "transformers_version",
    ):
        values.pop(name, None)
    return values


def _save_model_for_probe(model: Any, save_path: Path, implementation: str) -> None:
    """Save package source without invoking remote-code file collection.

    Artifact mode must exercise Transformers' normal remote-code save path.
    Package mode loads the direct FastPLMs class while retaining the artifact's
    ``auto_map`` metadata. Those classes intentionally report remote-code
    capability, but their direct package class has no AutoClass registration;
    asking Transformers to collect it would insert a null ``auto_map`` key.
    The temporary instance override is scoped only to this package comparison.
    """

    if implementation != "package":
        model.save_pretrained(save_path, safe_serialization=True)
        return

    model.__dict__["is_remote_code"] = lambda: False
    try:
        model.save_pretrained(save_path, safe_serialization=True)
    finally:
        model.__dict__.pop("is_remote_code", None)


def _load_class(implementation: str, auto_class: str, class_path: str) -> type:
    if implementation == "artifact":
        import transformers

        return getattr(transformers, auto_class)
    module_name, class_name = class_path.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module_name), class_name)


def _load_kwargs(
    family: str,
    bf16_execution: str,
    torch: Any,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    dtype = torch.float32 if bf16_execution == "fp32_parameters_autocast" else torch.bfloat16
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": dtype,
        "device_map": torch.device("cuda"),
    }
    if family == "esmfold2":
        kwargs["load_esmc"] = False
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def _tokenizer(artifact: Path, config: Any) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        artifact,
        config=config,
        local_files_only=True,
        trust_remote_code=True,
    )


def _exercise(
    model: Any,
    artifact: Path,
    family: str,
    bf16_execution: str,
    torch: Any,
) -> Any:
    sequence = "MSTNPKPQ"
    numeric_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16_execution == "fp32_parameters_autocast"
        else contextlib.nullcontext()
    )
    if family == "boltz2":
        with torch.inference_mode(), numeric_context:
            return model.predict_structure(
                sequence,
                recycling_steps=1,
                num_sampling_steps=2,
                diffusion_samples=1,
            )
    if family == "esmfold":
        with torch.inference_mode(), numeric_context:
            return model.fold_protein(sequence, return_pdb_string=False)
    if family == "esmfold2":
        # H follows Biohub's embedding-plus-80-block ordering and has shape
        # (b, l, 81, 2560). This exercises the advertised learned projection
        # without loading the separately pinned 6B ESMC checkpoint.
        hidden_states = torch.arange(
            2 * 81 * 2560,
            device="cuda",
            dtype=torch.bfloat16,
        ).reshape(1, 2, 81, 2560)
        residue_mask = torch.tensor([[True, False]], device="cuda")
        with torch.inference_mode(), numeric_context:
            return model.project_esmc_hidden_states(hidden_states, residue_mask)

    prep_tokens = getattr(getattr(model, "model", None), "prep_tokens", None)
    if family == "dplm2":
        tokenizer = _tokenizer(artifact, model.config)
        aa_sequence = f"{tokenizer.aa_cls_token}{sequence}{tokenizer.aa_eos_token}"
        encoded = tokenizer(
            [aa_sequence],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            name: value.to("cuda") for name, value in encoded.items() if torch.is_tensor(value)
        }
    elif prep_tokens is not None:
        batch = prep_tokens.get_batch_kwargs([sequence], device=torch.device("cuda"))
        inputs = dict(batch)
        # The raw-sequence preparer returns masked-LM training labels. Artifact
        # inference validates each advertised AutoClass without imposing those
        # token-level labels on unrelated sequence-classification heads.
        inputs.pop("labels", None)
        inputs["attention_mask"] = batch["sequence_ids"].ne(-1).long()
    else:
        encoded = _tokenizer(artifact, model.config)(
            [sequence],
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            name: value.to("cuda") for name, value in encoded.items() if torch.is_tensor(value)
        }
    if family == "esm_plusplus":
        inputs["sequence_id"] = inputs["attention_mask"].bool()
    if getattr(model.config, "is_encoder_decoder", False):
        inputs["decoder_input_ids"] = inputs["input_ids"]
        inputs["decoder_attention_mask"] = inputs["attention_mask"]
    with torch.inference_mode(), numeric_context:
        return model(**inputs)


def _load_saved_artifact(
    *,
    artifact: Path,
    family: str,
    bf16_execution: str,
    auto_class: str,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Load one saved directory exactly once in an isolated artifact process."""

    _require_artifact_isolation()
    auto_type = _load_class("artifact", auto_class, "")
    if auto_class == "AutoConfig":
        config = auto_type.from_pretrained(
            artifact,
            local_files_only=True,
            trust_remote_code=True,
        )
        semantic = _semantic_config(config)
        return {
            "config": hashlib.sha256(
                json.dumps(semantic, sort_keys=True, default=str).encode()
            ).hexdigest()
        }

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Offline artifact validation requires the H100 validation GPU")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = _load_model_exact(
        auto_type,
        artifact,
        trust_remote_code=True,
        **_load_kwargs(family, bf16_execution, torch, attn_implementation),
    ).eval()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output = _exercise(model, artifact, family, bf16_execution, torch)
    return {
        "state": _state_digest(model),
        "output": _output_digest(output),
    }


def _run_isolated_reload(
    *,
    artifact: Path,
    family: str,
    bf16_execution: str,
    auto_class: str,
    class_path: str,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Reload a saved remote-code directory without inheriting Python module state."""

    with tempfile.TemporaryDirectory(prefix="fastplms-isolated-reload-") as directory:
        isolation_root = Path(directory)
        output = isolation_root / "reload.json"
        command = [
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).resolve()),
            "--artifact",
            str(artifact.resolve()),
            "--family",
            family,
            "--bf16-execution",
            bf16_execution,
            "--auto-class",
            auto_class,
            "--class-path",
            class_path,
            "--implementation",
            "artifact",
            "--output",
            str(output),
            "--reload-only",
        ]
        for path in _runtime_site_packages():
            command.extend(("--runtime-site-package", str(path)))
        if attn_implementation is not None:
            command.extend(("--attn-implementation", attn_implementation))

        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["HF_HOME"] = (
            os.environ.get("HF_HOME", str(isolation_root / "hf-home"))
            if attn_implementation is not None
            else str(isolation_root / "hf-home")
        )
        environment["HF_MODULES_CACHE"] = str(isolation_root / "modules")
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            command,
            cwd=isolation_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stdout + completed.stderr).strip()
            raise RuntimeError(
                "Isolated saved-artifact reload failed" + (f":\n{details}" if details else ".")
            )
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Isolated saved-artifact reload produced invalid output") from error
        if not isinstance(result, dict):
            raise RuntimeError("Isolated saved-artifact reload output must be an object")
        return result


def _load_model_exact(auto_type: Any, artifact: Path, **kwargs: Any) -> Any:
    """Load a model while rejecting every incomplete weight-loading outcome."""

    loaded = auto_type.from_pretrained(
        artifact,
        output_loading_info=True,
        **kwargs,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Transformers did not return model loading diagnostics")
    model, loading_info = loaded
    if not isinstance(loading_info, dict):
        raise RuntimeError("Transformers returned invalid model loading diagnostics")
    failures = {
        name: loading_info.get(name)
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    if failures:
        raise RuntimeError(
            "Exact AutoModel weight loading failed: "
            + json.dumps(failures, sort_keys=True, default=str)
        )
    return model


def probe(
    *,
    artifact: Path,
    family: str,
    bf16_execution: str,
    auto_class: str,
    class_path: str,
    implementation: str,
    source_root: Path | None,
    reload_only: bool = False,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Load, infer, save, and reload one advertised class."""

    if implementation == "artifact":
        if source_root is not None:
            raise ValueError("Artifact mode must not receive a package source root")
        _require_artifact_isolation()
    else:
        if reload_only:
            raise ValueError("Reload-only mode is restricted to isolated artifacts")
        if source_root is None:
            raise ValueError("Package mode requires --source-root")
        sys.path.insert(0, str(source_root.resolve()))

    if reload_only:
        return _load_saved_artifact(
            artifact=artifact,
            family=family,
            bf16_execution=bf16_execution,
            auto_class=auto_class,
            attn_implementation=attn_implementation,
        )

    import torch

    auto_type = _load_class(implementation, auto_class, class_path)
    trust_remote_code = implementation == "artifact"
    if auto_class == "AutoConfig":
        config = auto_type.from_pretrained(
            artifact,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        first = _semantic_config(config)
        result = {
            "config": hashlib.sha256(
                json.dumps(first, sort_keys=True, default=str).encode()
            ).hexdigest()
        }
        with tempfile.TemporaryDirectory(prefix="fastplms-config-reload-") as directory:
            config.save_pretrained(directory)
            if implementation == "artifact":
                reloaded_result = _run_isolated_reload(
                    artifact=Path(directory),
                    family=family,
                    bf16_execution=bf16_execution,
                    auto_class=auto_class,
                    class_path=class_path,
                    attn_implementation=attn_implementation,
                )
                if reloaded_result != result:
                    raise AssertionError("Configuration changed across isolated save/reload")
            else:
                reloaded = auto_type.from_pretrained(
                    directory,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                if first != _semantic_config(reloaded):
                    raise AssertionError("Configuration changed across save/reload")
        return result

    if not torch.cuda.is_available():
        raise RuntimeError("Offline artifact validation requires the H100 validation GPU")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = _load_model_exact(
        auto_type,
        artifact,
        trust_remote_code=trust_remote_code,
        **_load_kwargs(family, bf16_execution, torch, attn_implementation),
    ).eval()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output = _exercise(model, artifact, family, bf16_execution, torch)
    result = {
        "state": _state_digest(model),
        "output": _output_digest(output),
    }

    with tempfile.TemporaryDirectory(prefix="fastplms-model-reload-") as directory:
        save_path = Path(directory)
        _save_model_for_probe(model, save_path, implementation)
        try:
            tokenizer = _tokenizer(artifact, model.config)
        except (OSError, ValueError):
            tokenizer = None
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
        del output, model
        torch.cuda.empty_cache()
        if implementation == "artifact":
            reloaded_result = _run_isolated_reload(
                artifact=save_path,
                family=family,
                bf16_execution=bf16_execution,
                auto_class=auto_class,
                class_path=class_path,
                attn_implementation=attn_implementation,
            )
            if reloaded_result != result:
                raise AssertionError("Model changed across isolated save/reload")
        else:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            reloaded = _load_model_exact(
                auto_type,
                save_path,
                trust_remote_code=False,
                **_load_kwargs(family, bf16_execution, torch, attn_implementation),
            ).eval()
            if _state_digest(reloaded) != result["state"]:
                raise AssertionError("State changed across save/reload")
            reloaded_output = _exercise(
                reloaded,
                save_path,
                family,
                bf16_execution,
                torch,
            )
            if _output_digest(reloaded_output) != result["output"]:
                raise AssertionError("Inference changed across save/reload")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--bf16-execution",
        required=True,
        choices=("static_parameters", "fp32_parameters_autocast"),
    )
    parser.add_argument("--auto-class", required=True)
    parser.add_argument("--class-path", required=True)
    parser.add_argument("--implementation", choices=("artifact", "package"), required=True)
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "flash_attention_3"),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reload-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--runtime-site-package",
        action="append",
        default=[],
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    _add_runtime_site_packages(arguments.runtime_site_package)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    result = probe(
        artifact=arguments.artifact.resolve(),
        family=arguments.family,
        bf16_execution=arguments.bf16_execution,
        auto_class=arguments.auto_class,
        class_path=arguments.class_path,
        implementation=arguments.implementation,
        source_root=arguments.source_root,
        reload_only=arguments.reload_only,
        attn_implementation=arguments.attn_implementation,
    )
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
