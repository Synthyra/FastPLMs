"""Probe one local Hub artifact without importing FastPLMs from outside it."""

from __future__ import annotations

import argparse
import builtins
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import importlib.abc
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast


_CPU_CONTRACT_MARKER = ".fastplms-cpu-contract.json"
_CPU_FORBIDDEN_READ_ROOTS = tuple(
    path.resolve()
    for path in (
        Path(__file__).resolve().parents[2] / "vendor" / "upstream",
        Path(__file__).resolve().parents[2] / ".git" / "modules",
        Path(__file__).resolve().parents[2] / "official",
    )
)


@dataclasses.dataclass(frozen=True)
class ProbeCase:
    """One advertised AutoClass contract within a checkpoint probe."""

    auto_class: str
    class_path: str
    expected_missing_key_prefixes: tuple[str, ...] = ()
    expected_unexpected_key_prefixes: tuple[str, ...] = ()


class _BlockExternalFastPLMs(importlib.abc.MetaPathFinder):
    """Prevent a probe from satisfying artifact imports from external source."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        del path, target
        if fullname == "fastplms" and fullname not in sys.modules:
            raise ModuleNotFoundError(
                "Artifact remote code attempted to import FastPLMs from outside its "
                "embedded runtime."
            )
        return None


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
    """Reject loaded FastPLMs state and guard against external source imports."""

    if not sys.flags.isolated:
        raise RuntimeError("Artifact mode must run under python -I")
    if "fastplms" in sys.modules:
        raise RuntimeError("FastPLMs must not be imported before artifact loading")
    if not any(isinstance(finder, _BlockExternalFastPLMs) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BlockExternalFastPLMs())


def _tensor_digest(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _matches_key_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _state_digest(
    model: Any,
    *,
    excluded_prefixes: Iterable[str] = (),
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if _matches_key_prefix(name, excluded_prefixes):
            continue
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
        "auto_map",
        "fastplms_checkpoint_hash",
        "fastplms_checkpoint_repo_id",
        "fastplms_checkpoint_revision",
        "fastplms_model_id",
        "fastplms_runtime_bundle_sha256",
        "fastplms_runtime_revision",
        "fastplms_source_tree_sha256",
        "fastplms_weights_revision",
        "transformers_version",
    ):
        values.pop(name, None)
    return values


def _save_model_for_probe(model: Any, save_path: Path, implementation: str) -> None:
    """Exercise the unmodified Transformers save path for every implementation."""

    del implementation
    model.save_pretrained(save_path, safe_serialization=True)


def _load_class(implementation: str, auto_class: str, class_path: str) -> type:
    if implementation == "artifact":
        import transformers

        return getattr(transformers, auto_class)
    module_name, class_name = class_path.rsplit(".", maxsplit=1)
    source_class = getattr(importlib.import_module(module_name), class_name)
    register = getattr(source_class, "register_for_auto_class", None)
    if not callable(register):
        raise RuntimeError(f"{class_path} cannot register for {auto_class}")
    register(auto_class)
    if getattr(source_class, "_auto_class", None) != auto_class:
        raise RuntimeError(f"{class_path} did not register for {auto_class}")
    config_class = getattr(source_class, "config_class", None)
    if auto_class != "AutoConfig" and config_class is not None:
        config_register = getattr(config_class, "register_for_auto_class", None)
        if not callable(config_register):
            raise RuntimeError(f"{class_path} config cannot register for AutoConfig")
        config_register("AutoConfig")
        if getattr(config_class, "_auto_class", None) != "AutoConfig":
            raise RuntimeError(f"{class_path} config did not register for AutoConfig")
    return source_class


def _assert_complete_saved_auto_map(
    save_path: Path,
    *,
    expected_auto_classes: set[str],
) -> None:
    """Require every advertised AutoClass to survive normal serialization."""

    try:
        config = json.loads((save_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Saved AutoClass config is missing or malformed") from error
    auto_map = config.get("auto_map") if isinstance(config, dict) else None
    if not isinstance(auto_map, dict) or set(auto_map) != expected_auto_classes:
        raise RuntimeError("Saved config does not preserve the complete advertised auto_map")
    invalid = {
        name: target
        for name, target in auto_map.items()
        if not isinstance(target, str) or not target or target.endswith(".None")
    }
    if invalid:
        raise RuntimeError(
            "Saved config contains null or invalid AutoClass targets: "
            + json.dumps(invalid, sort_keys=True)
        )


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
    # AutoModel intentionally exposes the encoder-only ANKH view while
    # retaining the official T5 ``is_encoder_decoder`` configuration value.
    # Decoder inputs belong only to models that actually allocate a decoder,
    # such as AutoModelForSeq2SeqLM.
    if getattr(model.config, "is_encoder_decoder", False) and hasattr(model, "decoder"):
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
    implementation: str,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Load one saved directory exactly once through its advertised AutoClass."""

    if implementation == "artifact":
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
        raise RuntimeError("Offline artifact validation requires a CUDA GPU")
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
    implementation: str,
    source_root: Path | None,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Reload a saved directory through its AutoClass in a fresh process."""

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
            implementation,
            "--output",
            str(output),
            "--reload-only",
        ]
        if source_root is not None:
            command.extend(("--source-root", str(source_root.resolve())))
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


def _load_model_exact(
    auto_type: Any,
    artifact: Path,
    *,
    expected_missing_key_prefixes: Iterable[str] = (),
    expected_unexpected_key_prefixes: Iterable[str] = (),
    **kwargs: Any,
) -> Any:
    """Load a model while rejecting every undeclared weight-loading outcome."""

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
    diagnostics: dict[str, list[Any]] = {}
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = loading_info.get(name, [])
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise RuntimeError(f"Transformers returned invalid {name} loading diagnostics")
        diagnostics[name] = sorted(values, key=repr)

    unexpected_missing = [
        key
        for key in diagnostics["missing_keys"]
        if not isinstance(key, str) or not _matches_key_prefix(key, expected_missing_key_prefixes)
    ]
    unexpected_checkpoint_keys = [
        key
        for key in diagnostics["unexpected_keys"]
        if not isinstance(key, str)
        or not _matches_key_prefix(key, expected_unexpected_key_prefixes)
    ]
    failures = {
        name: values
        for name, values in (
            ("missing_keys", unexpected_missing),
            ("unexpected_keys", unexpected_checkpoint_keys),
            ("mismatched_keys", diagnostics["mismatched_keys"]),
            ("error_msgs", diagnostics["error_msgs"]),
        )
        if values
    }
    if failures:
        raise RuntimeError(
            "Validated AutoModel weight loading failed: "
            + json.dumps(failures, sort_keys=True, default=str)
        )
    return model


def _prepare_probe_environment(
    implementation: str,
    source_root: Path | None,
    *,
    reload_only: bool,
) -> None:
    if implementation == "artifact":
        if source_root is not None:
            raise ValueError("Artifact mode must not receive a repository source root")
        _require_artifact_isolation()
        return
    if source_root is None:
        raise ValueError("Source mode requires --source-root")
    source_path = str(source_root.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


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
    expected_missing_key_prefixes: Iterable[str] = (),
    expected_unexpected_key_prefixes: Iterable[str] = (),
    _environment_prepared: bool = False,
) -> dict[str, Any]:
    """Load, infer, save, and reload one advertised class."""

    if not _environment_prepared:
        _prepare_probe_environment(
            implementation,
            source_root,
            reload_only=reload_only,
        )

    if reload_only:
        return _load_saved_artifact(
            artifact=artifact,
            family=family,
            bf16_execution=bf16_execution,
            auto_class=auto_class,
            implementation=implementation,
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
        expected_auto_classes = set(config.auto_map)
        first = _semantic_config(config)
        result = {
            "config": hashlib.sha256(
                json.dumps(first, sort_keys=True, default=str).encode()
            ).hexdigest()
        }
        with tempfile.TemporaryDirectory(prefix="fastplms-config-reload-") as directory:
            config.save_pretrained(directory)
            save_path = Path(directory)
            _assert_complete_saved_auto_map(
                save_path,
                expected_auto_classes=expected_auto_classes,
            )
            reloaded_result = _run_isolated_reload(
                artifact=save_path,
                family=family,
                bf16_execution=bf16_execution,
                auto_class=auto_class,
                class_path=class_path,
                implementation=implementation,
                source_root=source_root,
                attn_implementation=attn_implementation,
            )
            if reloaded_result != result:
                raise AssertionError("Configuration changed across isolated AutoClass save/reload")
        return result

    if not torch.cuda.is_available():
        raise RuntimeError("Offline artifact validation requires a CUDA GPU")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        **_load_kwargs(family, bf16_execution, torch, attn_implementation),
    }
    model = _load_model_exact(
        auto_type,
        artifact,
        expected_missing_key_prefixes=expected_missing_key_prefixes,
        expected_unexpected_key_prefixes=expected_unexpected_key_prefixes,
        **load_kwargs,
    ).eval()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output = _exercise(model, artifact, family, bf16_execution, torch)
    result = {
        "state": _state_digest(model),
        "pretrained_state": _state_digest(
            model,
            excluded_prefixes=expected_missing_key_prefixes,
        ),
        "output": _output_digest(output),
    }

    with tempfile.TemporaryDirectory(prefix="fastplms-model-reload-") as directory:
        save_path = Path(directory)
        expected_auto_classes = set(model.config.auto_map)
        _save_model_for_probe(model, save_path, implementation)
        _assert_complete_saved_auto_map(
            save_path,
            expected_auto_classes=expected_auto_classes,
        )
        try:
            tokenizer = _tokenizer(artifact, model.config)
        except (OSError, ValueError):
            tokenizer = None
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
        del output, model
        torch.cuda.empty_cache()
        torch.manual_seed(314159)
        torch.cuda.manual_seed_all(314159)
        independently_loaded = _load_model_exact(
            auto_type,
            artifact,
            expected_missing_key_prefixes=expected_missing_key_prefixes,
            expected_unexpected_key_prefixes=expected_unexpected_key_prefixes,
            **load_kwargs,
        ).eval()
        independently_loaded_state = _state_digest(
            independently_loaded,
            excluded_prefixes=expected_missing_key_prefixes,
        )
        del independently_loaded
        torch.cuda.empty_cache()
        if independently_loaded_state != result["pretrained_state"]:
            raise RuntimeError(
                "Pretrained AutoModel weights depend on the initialization seed; "
                "one or more shared/base weights were not loaded from the checkpoint"
            )
        reloaded_result = _run_isolated_reload(
            artifact=save_path,
            family=family,
            bf16_execution=bf16_execution,
            auto_class=auto_class,
            class_path=class_path,
            implementation=implementation,
            source_root=source_root,
            attn_implementation=attn_implementation,
        )
        expected_reload_result = {
            "state": result["state"],
            "output": result["output"],
        }
        if reloaded_result != expected_reload_result:
            raise AssertionError("Model changed across isolated AutoClass save/reload")
    return result


def _load_probe_cases(path: Path) -> tuple[ProbeCase, ...]:
    """Load a fail-closed batch description produced by the release test."""

    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid AutoClass probe case file: {path}") from error
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RuntimeError("AutoClass probe case file must contain a non-empty list")

    required = {
        "auto_class",
        "class_path",
        "expected_missing_key_prefixes",
        "expected_unexpected_key_prefixes",
    }
    cases: list[ProbeCase] = []
    names: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict) or set(raw_case) != required:
            raise RuntimeError(
                f"AutoClass probe case {index} must contain exactly {sorted(required)}"
            )
        auto_class = raw_case["auto_class"]
        class_path = raw_case["class_path"]
        missing = raw_case["expected_missing_key_prefixes"]
        unexpected = raw_case["expected_unexpected_key_prefixes"]
        if not isinstance(auto_class, str) or not auto_class:
            raise RuntimeError(f"AutoClass probe case {index} has an invalid auto_class")
        if auto_class in names:
            raise RuntimeError(f"Duplicate AutoClass probe case: {auto_class}")
        if not isinstance(class_path, str) or not class_path:
            raise RuntimeError(f"AutoClass probe case {index} has an invalid class_path")
        if (
            not isinstance(missing, list)
            or not all(isinstance(prefix, str) and prefix for prefix in missing)
            or not isinstance(unexpected, list)
            or not all(isinstance(prefix, str) and prefix for prefix in unexpected)
        ):
            raise RuntimeError(f"AutoClass probe case {index} has invalid key allowances")
        names.add(auto_class)
        cases.append(
            ProbeCase(
                auto_class=auto_class,
                class_path=class_path,
                expected_missing_key_prefixes=tuple(missing),
                expected_unexpected_key_prefixes=tuple(unexpected),
            )
        )
    return tuple(cases)


def _release_case_memory() -> None:
    """Release model objects and allocator cache between checkpoint views."""

    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def _assert_nested_cpu_output(actual: Any, expected: Any, torch: Any) -> None:
    """Compare tuple and ModelOutput views without flattening nested tensors."""

    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise AssertionError(
                f"Expected a tensor in tuple output, received {type(actual).__name__}."
            )
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError("Nested tuple/list output differs from ModelOutput.to_tuple().")
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_cpu_output(actual_item, expected_item, torch)
        return
    if actual != expected:
        raise AssertionError(
            f"Tuple output differs from ModelOutput.to_tuple(): {actual!r} != {expected!r}."
        )


def _cpu_sequence_inputs(
    family: str,
    auto_class: str,
    config: Any,
    torch: Any,
) -> tuple[dict[str, Any], bool]:
    """Return tiny tensor-only inputs and whether the advertised head owns a loss."""

    if family == "e1":
        input_ids = torch.tensor([[1, 5, 6, 2]], dtype=torch.long)
        inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "within_seq_position_ids": torch.arange(4).unsqueeze(0),
            "global_position_ids": torch.arange(4).unsqueeze(0),
            "sequence_ids": torch.zeros((1, 4), dtype=torch.long),
        }
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif family == "ankh":
        input_ids = torch.tensor([[2, 3, 1, 0], [4, 5, 1, 0]], dtype=torch.long)
        attention_mask = input_ids.ne(0)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    else:
        input_ids = torch.tensor(
            [[0, 3, 4, 2, 1], [0, 6, 2, 1, 1]],
            dtype=torch.long,
        )
        attention_mask = input_ids.ne(1)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if family == "esm3":
            inputs = {
                "sequence_tokens": input_ids,
                "sequence_id": attention_mask,
            }
        if family == "esm_plusplus":
            inputs["sequence_id"] = attention_mask

    has_loss = auto_class != "AutoModel"
    if auto_class == "AutoModelForSequenceClassification":
        inputs["labels"] = torch.arange(input_ids.shape[0], dtype=torch.long).remainder(
            int(config.num_labels)
        )
    elif auto_class == "AutoModelForTokenClassification":
        labels = input_ids.remainder(int(config.num_labels))
        inputs["labels"] = labels.masked_fill(~attention_mask, -100)
    elif auto_class == "AutoModelForSeq2SeqLM":
        decoder_input_ids = torch.tensor(
            [[0, 5, 1, 0], [0, 6, 1, 0]],
            dtype=torch.long,
        )
        decoder_attention_mask = decoder_input_ids.ne(0)
        inputs.update(
            {
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": decoder_input_ids.masked_fill(~decoder_attention_mask, -100),
                "use_cache": False,
            }
        )
    elif auto_class == "AutoModelForMaskedLM":
        inputs["labels"] = input_ids.masked_fill(~attention_mask, -100)
    else:
        has_loss = False
    return inputs, has_loss


def _cpu_primary_tensor(output: Any, torch: Any) -> Any:
    for name in ("last_hidden_state", "logits", "embeddings", "sequence_logits"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    for value in output.to_tuple():
        if torch.is_tensor(value):
            return value
    raise AssertionError("Advertised AutoClass output contains no tensor for backward().")


def _cpu_state_digest(model: Any) -> str:
    return _state_digest(model)


def _cpu_resize_and_setter_contract(model: Any) -> int:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "num_embeddings"):
        raise AssertionError("Advertised sequence AutoClass has no token input embeddings.")
    model.set_input_embeddings(input_embeddings)
    if model.get_input_embeddings() is not input_embeddings:
        raise AssertionError("set_input_embeddings() did not install the supplied module.")

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        model.set_output_embeddings(output_embeddings)
        if model.get_output_embeddings() is not output_embeddings:
            raise AssertionError("set_output_embeddings() did not install the supplied module.")

    resized_vocab = int(input_embeddings.num_embeddings) + 1
    resized = model.resize_token_embeddings(resized_vocab)
    if int(resized.num_embeddings) != resized_vocab:
        raise AssertionError("resize_token_embeddings() returned the wrong vocabulary size.")
    if int(model.get_input_embeddings().num_embeddings) != resized_vocab:
        raise AssertionError("Input embeddings did not retain the resized vocabulary.")
    resized_output = model.get_output_embeddings()
    output_size = getattr(resized_output, "out_features", resized_vocab)
    if resized_output is not None and int(output_size) != resized_vocab:
        raise AssertionError("Output embeddings did not retain the resized vocabulary.")
    return resized_vocab


def _probe_tiny_cpu_model(
    *,
    artifact: Path,
    family: str,
    auto_class: str,
    config: Any,
) -> dict[str, Any]:
    """Exercise one tiny remote AutoClass without checkpoint or accelerator access."""

    import torch

    if family == "boltz2":
        from torch import nn

        class _TinyBoltzCore(nn.Module):
            def __init__(self, width: int = 3) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.linspace(0.5, 1.0, width))

        module = sys.modules.get("fastplms.models.boltz.modeling_boltz2")
        if module is None:
            raise RuntimeError("The remote Boltz2 runtime module was not loaded.")
        dynamic_module: Any = module
        dynamic_module.Boltz2InferenceCore = _TinyBoltzCore

    auto_type: Any = _load_class("artifact", auto_class, "")
    model = auto_type.from_config(config, trust_remote_code=True).eval()
    if not model.is_remote_code() or model._auto_class != auto_class:
        raise AssertionError(f"{auto_class} did not retain its remote AutoClass registration.")
    expected_class = config.auto_map[auto_class].rsplit(".", maxsplit=1)[1]
    if type(model).__name__ != expected_class:
        raise AssertionError(
            f"{auto_class} dispatched {type(model).__name__}, expected {expected_class}."
        )

    if family in {"boltz2", "esmfold", "esmfold2"}:
        # Full public structure forwards use injected native-compatible cores in
        # tests/unit/test_structure_output_contracts.py. This isolated artifact
        # check owns remote Auto dispatch plus exact state persistence so those
        # behavior tests do not need a second copy of the runtime bundler.
        before = _cpu_state_digest(model)
        with tempfile.TemporaryDirectory(prefix="fastplms-cpu-structure-reload-") as directory:
            save_path = Path(directory)
            _save_model_for_probe(model, save_path, "artifact")
            reload_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": True,
            }
            if family == "esmfold2":
                reload_kwargs["load_esmc"] = False
            reloaded = _load_model_exact(auto_type, save_path, **reload_kwargs).eval()
            if _cpu_state_digest(reloaded) != before:
                raise AssertionError("Structure AutoClass state changed across save/reload.")
        return {
            "class": type(model).__name__,
            "state": before,
            "structure_forward_delegated": True,
        }

    inputs, has_loss = _cpu_sequence_inputs(family, auto_class, config, torch)
    output_flags = {
        "output_attentions": True,
        "output_hidden_states": True,
    }
    structured = model(**inputs, **output_flags, return_dict=True)
    tuple_output = model(**inputs, **output_flags, return_dict=False)
    if not hasattr(structured, "to_tuple") or not isinstance(tuple_output, tuple):
        raise AssertionError("Advertised AutoClass does not honor return_dict.")
    _assert_nested_cpu_output(tuple_output, structured.to_tuple(), torch)

    loss = getattr(structured, "loss", None)
    if has_loss:
        if loss is None or not torch.isfinite(loss):
            raise AssertionError("Advertised task AutoClass did not return a finite loss.")
        objective = loss
    else:
        objective = _cpu_primary_tensor(structured, torch).float().square().mean()
    objective.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Advertised AutoClass backward pass produced invalid gradients.")

    resized_vocab = _cpu_resize_and_setter_contract(model)
    reloaded_inputs = {name: value for name, value in inputs.items() if name != "labels"}
    with torch.inference_mode():
        expected_output = model(**reloaded_inputs, return_dict=True)
    expected_digest = _output_digest(expected_output)
    expected_state = _cpu_state_digest(model)
    resaved = False
    with tempfile.TemporaryDirectory(prefix="fastplms-cpu-autoclass-reload-") as directory:
        save_path = Path(directory) / "first"
        _save_model_for_probe(model, save_path, "artifact")
        reloaded = _load_model_exact(
            auto_type,
            save_path,
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
        if _cpu_state_digest(reloaded) != expected_state:
            raise AssertionError("AutoClass state changed across save/reload.")
        with torch.inference_mode():
            observed_output = reloaded(**reloaded_inputs, return_dict=True)
        if _output_digest(observed_output) != expected_digest:
            raise AssertionError("AutoClass output changed across save/reload.")
        if family in {"ankh", "dplm", "dplm2", "esm2", "esm3", "esm_plusplus"} and (
            auto_class == "AutoModel"
        ):
            resave_path = Path(directory) / "second"
            _save_model_for_probe(reloaded, resave_path, "artifact")
            resaved_model = _load_model_exact(
                auto_type,
                resave_path,
                local_files_only=True,
                trust_remote_code=True,
            ).eval()
            if _cpu_state_digest(resaved_model) != expected_state:
                raise AssertionError("Remote AutoClass state changed across save-resave.")
            with torch.inference_mode():
                resaved_output = resaved_model(**reloaded_inputs, return_dict=True)
            if _output_digest(resaved_output) != expected_digest:
                raise AssertionError("Remote AutoClass output changed across save-resave.")
            resaved = True

    return {
        "class": type(model).__name__,
        "loss": has_loss,
        "resized_vocab": resized_vocab,
        "state": expected_state,
        "tuple_fields": len(tuple_output),
        "resaved": resaved,
    }


def probe_tiny_cpu_many(
    *,
    artifact: Path,
    family: str,
    cases: Iterable[ProbeCase],
) -> dict[str, dict[str, Any]]:
    """Run every tiny family AutoClass in one isolated, offline CPU process."""

    marker_path = artifact / _CPU_CONTRACT_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "--tiny-cpu-contract requires an explicit non-release CPU test artifact marker."
        ) from error
    if marker != {
        "release_artifact": False,
        "schema_version": 1,
        "scope": "tests/cpu",
    }:
        raise RuntimeError("Invalid non-release CPU test artifact marker.")
    forbidden_release_files = [
        name
        for name in ("artifact-manifest.json", "provenance.json", "runtime-attestation.json")
        if (artifact / name).exists()
    ]
    if forbidden_release_files:
        raise RuntimeError(
            "Tiny CPU AutoClass probes refuse release-shaped artifacts: "
            + ", ".join(forbidden_release_files)
        )

    _prepare_probe_environment("artifact", None, reload_only=False)
    case_tuple = tuple(cases)
    config_cases = [case for case in case_tuple if case.auto_class == "AutoConfig"]
    if len(config_cases) != 1:
        raise ValueError("A tiny CPU AutoClass batch requires exactly one AutoConfig case.")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        artifact,
        local_files_only=True,
        trust_remote_code=True,
    )
    if getattr(config, "fastplms_cpu_contract_only", None) is not True:
        raise RuntimeError("Tiny CPU AutoClass config lacks its non-release identity.")
    expected_config_class = config.auto_map["AutoConfig"].rsplit(".", maxsplit=1)[1]
    if type(config).__name__ != expected_config_class:
        raise AssertionError(
            f"AutoConfig dispatched {type(config).__name__}, expected {expected_config_class}."
        )
    with tempfile.TemporaryDirectory(prefix="fastplms-cpu-config-reload-") as directory:
        config.save_pretrained(directory)
        reloaded_config = AutoConfig.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=True,
        )
        if _semantic_config(reloaded_config) != _semantic_config(config):
            raise AssertionError("Tiny remote AutoConfig changed across save/reload.")

    results: dict[str, dict[str, Any]] = {
        "AutoConfig": {
            "class": type(config).__name__,
            "config": hashlib.sha256(
                json.dumps(_semantic_config(config), sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
    }
    for case in case_tuple:
        if case.auto_class == "AutoConfig":
            continue
        if case.auto_class in results:
            raise ValueError(f"Duplicate AutoClass probe case: {case.auto_class}")
        try:
            # Some model constructors normalize or annotate the config. Give
            # every advertised view a fresh remote configuration instance.
            model_config = AutoConfig.from_pretrained(
                artifact,
                local_files_only=True,
                trust_remote_code=True,
            )
            results[case.auto_class] = _probe_tiny_cpu_model(
                artifact=artifact,
                family=family,
                auto_class=case.auto_class,
                config=model_config,
            )
        finally:
            _release_case_memory()
    return results


def _install_cpu_probe_hermetic_guards() -> None:
    """Deny network access and reference-source reads in a tiny isolated probe."""

    import socket

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access is forbidden in the tiny CPU AutoClass probe.")

    socket_module: Any = socket
    socket_type: Any = socket.socket
    socket_module.create_connection = blocked
    socket_module.getaddrinfo = blocked
    socket_type.connect = blocked
    socket_type.connect_ex = blocked
    socket_type.sendto = blocked
    if hasattr(socket.socket, "sendmsg"):
        socket_type.sendmsg = blocked

    import huggingface_hub
    import huggingface_hub._snapshot_download
    import huggingface_hub.file_download

    hub_module: Any = huggingface_hub
    file_download_module: Any = huggingface_hub.file_download
    snapshot_download_module: Any = huggingface_hub._snapshot_download
    hub_module.hf_hub_download = blocked
    hub_module.snapshot_download = blocked
    file_download_module.hf_hub_download = blocked
    file_download_module.http_get = blocked
    snapshot_download_module.snapshot_download = blocked

    def assert_portable_path(file: object) -> None:
        if isinstance(file, str):
            path_value = file
        elif isinstance(file, os.PathLike):
            path_value = os.fspath(file)
            if not isinstance(path_value, str):
                return
        else:
            return
        try:
            resolved = Path(path_value).resolve()
        except (OSError, TypeError, ValueError):
            return
        if any(resolved == root or root in resolved.parents for root in _CPU_FORBIDDEN_READ_ROOTS):
            raise RuntimeError(
                f"Tiny CPU AutoClass probes may not access submodule/reference path: {resolved}"
            )

    original_builtin_open = cast(Callable[..., Any], builtins.open)
    original_io_open = cast(Callable[..., Any], io.open)
    original_os_open = cast(Callable[..., int], os.open)

    def guarded_builtin_open(file: object, *args: Any, **kwargs: Any) -> Any:
        assert_portable_path(file)
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: object, *args: Any, **kwargs: Any) -> Any:
        assert_portable_path(file)
        return original_io_open(file, *args, **kwargs)

    def guarded_os_open(file: object, *args: Any, **kwargs: Any) -> int:
        assert_portable_path(file)
        return original_os_open(file, *args, **kwargs)

    builtins.open = guarded_builtin_open
    io.open = guarded_io_open
    os.open = guarded_os_open  # type: ignore[assignment]


def probe_many(
    *,
    artifact: Path,
    family: str,
    bf16_execution: str,
    cases: Iterable[ProbeCase],
    implementation: str,
    source_root: Path | None,
    attn_implementation: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Exercise all checkpoint AutoClasses sequentially in one process."""

    _prepare_probe_environment(implementation, source_root, reload_only=False)
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.auto_class in results:
            raise ValueError(f"Duplicate AutoClass probe case: {case.auto_class}")
        try:
            results[case.auto_class] = probe(
                artifact=artifact,
                family=family,
                bf16_execution=bf16_execution,
                auto_class=case.auto_class,
                class_path=case.class_path,
                implementation=implementation,
                source_root=source_root,
                attn_implementation=attn_implementation,
                expected_missing_key_prefixes=case.expected_missing_key_prefixes,
                expected_unexpected_key_prefixes=case.expected_unexpected_key_prefixes,
                _environment_prepared=True,
            )
        finally:
            _release_case_memory()
    if not results:
        raise ValueError("At least one AutoClass probe case is required")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--bf16-execution",
        required=True,
        choices=("static_parameters", "fp32_parameters_autocast"),
    )
    parser.add_argument("--auto-class")
    parser.add_argument("--class-path")
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--implementation", choices=("artifact", "source"), required=True)
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "flash_attention_3"),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reload-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--tiny-cpu-contract",
        action="store_true",
        help="Exercise a config-only tiny remote-code artifact on CPU",
    )
    parser.add_argument(
        "--expected-missing-key-prefix",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-unexpected-key-prefix",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--runtime-site-package",
        action="append",
        default=[],
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    _add_runtime_site_packages(arguments.runtime_site_package)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if arguments.cases_file is not None:
        if arguments.auto_class is not None or arguments.class_path is not None:
            parser.error("--cases-file cannot be combined with --auto-class/--class-path")
        if arguments.reload_only:
            parser.error("--cases-file cannot be combined with --reload-only")
        cases = _load_probe_cases(arguments.cases_file)
        if arguments.tiny_cpu_contract:
            if arguments.implementation != "artifact":
                parser.error("--tiny-cpu-contract requires --implementation artifact")
            if arguments.source_root is not None or arguments.attn_implementation is not None:
                parser.error(
                    "--tiny-cpu-contract cannot receive repository source or Flash backend options"
                )
            _install_cpu_probe_hermetic_guards()
            result = probe_tiny_cpu_many(
                artifact=arguments.artifact.resolve(),
                family=arguments.family,
                cases=cases,
            )
        else:
            result = probe_many(
                artifact=arguments.artifact.resolve(),
                family=arguments.family,
                bf16_execution=arguments.bf16_execution,
                cases=cases,
                implementation=arguments.implementation,
                source_root=arguments.source_root,
                attn_implementation=arguments.attn_implementation,
            )
    else:
        if arguments.tiny_cpu_contract:
            parser.error("--tiny-cpu-contract requires --cases-file")
        if arguments.auto_class is None or arguments.class_path is None:
            parser.error("--auto-class and --class-path are required without --cases-file")
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
            expected_missing_key_prefixes=arguments.expected_missing_key_prefix,
            expected_unexpected_key_prefixes=arguments.expected_unexpected_key_prefix,
        )
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
