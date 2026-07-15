"""Fresh-environment validation for every built local Hub artifact."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.artifacts.offline_probe import (
    _exercise,
    _load_kwargs,
    _load_model_exact,
    _run_isolated_reload,
    _runtime_site_packages,
    _save_model_for_probe,
    _semantic_config,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = tomllib.loads((ROOT / "src" / "fastplms" / "models.toml").read_text(encoding="utf-8"))
PROBE = ROOT / "tools" / "artifacts" / "offline_probe.py"


def _cases() -> list[Any]:
    cases: list[Any] = []
    families = MANIFEST["families"]
    for model in MANIFEST["models"]:
        family_id = model["family"]
        auto_map = model.get("auto_map", families[family_id]["auto_map"])
        repository_name = model["fast_repo"].split("/", maxsplit=1)[1]
        for auto_class, class_path in sorted(auto_map.items()):
            marks = [pytest.mark.artifact, pytest.mark.gpu, pytest.mark.slow]
            if model["size_category"] == "xlarge":
                marks.append(pytest.mark.large)
            cases.append(
                pytest.param(
                    model["id"],
                    family_id,
                    repository_name,
                    auto_class,
                    class_path,
                    id=f"{model['id']}-{auto_class}",
                    marks=marks,
                )
            )
    return cases


def _run_probe(
    *,
    artifact: Path,
    family: str,
    auto_class: str,
    class_path: str,
    implementation: str,
    output: Path,
    attn_implementation: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-I",
        "-S",
        str(PROBE),
        "--artifact",
        str(artifact),
        "--family",
        family,
        "--bf16-execution",
        MANIFEST["families"][family]["bf16_execution"],
        "--auto-class",
        auto_class,
        "--class-path",
        class_path,
        "--implementation",
        implementation,
        "--output",
        str(output),
    ]
    for path in _runtime_site_packages():
        command.extend(("--runtime-site-package", str(path)))
    if attn_implementation is not None:
        command.extend(("--attn-implementation", attn_implementation))
    if implementation == "package":
        command.extend(("--source-root", str(ROOT / "src")))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return subprocess.run(
        command,
        cwd=output.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.artifact
def test_generated_flash_artifacts_resolve_their_embedded_kernel_lock() -> None:
    """Remote code must not depend on a checkout or installed FastPLMs wheel."""

    expected = (ROOT / "kernels.lock").read_bytes()
    for repository_name in ("ESM2-8M", "ESMplusplus_small", "DPLM-150M"):
        package = ROOT / "dist" / "hub" / repository_name / "fastplms"
        lock = package / "kernels.lock"
        module_path = package / "attention" / "_kernel_lock.py"
        assert lock.read_bytes() == expected

        module_spec = importlib.util.spec_from_file_location(
            f"artifact_kernel_lock_{repository_name}",
            module_path,
        )
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        assert module._kernel_lock_path() == lock


@pytest.mark.artifact
def test_isolated_reload_rejects_incomplete_saved_remote_code(tmp_path: Path) -> None:
    """A fresh interpreter must load every source file from the saved directory."""

    artifact = tmp_path / "saved"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {"AutoConfig": "modeling_isolation.IsolationConfig"},
                "model_type": "fastplms-isolation-test",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "modeling_isolation.py").write_text(
        "from .support_config import IsolationConfig\n",
        encoding="utf-8",
    )
    support = artifact / "support_config.py"
    support.write_text(
        "from transformers import PretrainedConfig\n\n"
        "class IsolationConfig(PretrainedConfig):\n"
        "    model_type = 'fastplms-isolation-test'\n",
        encoding="utf-8",
    )

    complete = _run_isolated_reload(
        artifact=artifact,
        family="isolation-test",
        bf16_execution="static_parameters",
        auto_class="AutoConfig",
        class_path="unused.IsolationConfig",
    )
    assert set(complete) == {"config"}

    support.unlink()
    with pytest.raises(RuntimeError, match="Isolated saved-artifact reload failed"):
        _run_isolated_reload(
            artifact=artifact,
            family="isolation-test",
            bf16_execution="static_parameters",
            auto_class="AutoConfig",
            class_path="unused.IsolationConfig",
        )


def test_offline_probe_load_dtype_follows_manifest_execution_policy() -> None:
    torch = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        device=lambda value: value,
    )
    static = _load_kwargs("esm2", "static_parameters", torch)
    autocast = _load_kwargs("dplm", "fp32_parameters_autocast", torch)
    assert static["dtype"] == "bf16"
    assert autocast["dtype"] == "fp32"


def test_offline_probe_rejects_incomplete_weight_loading(tmp_path: Path) -> None:
    class AutoType:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            assert args == (tmp_path,)
            assert kwargs["output_loading_info"] is True
            return object(), {
                "missing_keys": ["encoder.layer.0.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    with pytest.raises(RuntimeError, match="Exact AutoModel weight loading failed"):
        _load_model_exact(AutoType, tmp_path, trust_remote_code=True)


def test_offline_probe_accepts_exact_weight_loading(tmp_path: Path) -> None:
    model = object()

    class AutoType:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            assert args == (tmp_path,)
            assert kwargs["output_loading_info"] is True
            return model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    assert _load_model_exact(AutoType, tmp_path, trust_remote_code=False) is model


def test_offline_probe_semantic_config_excludes_artifact_identity() -> None:
    config = SimpleNamespace(
        to_dict=lambda: {
            "hidden_size": 8,
            "fastplms_model_id": "toy",
            "fastplms_checkpoint_repo_id": "organization/toy",
            "fastplms_checkpoint_revision": "a" * 40,
            "fastplms_checkpoint_hash": "b" * 64,
        }
    )
    assert _semantic_config(config) == {"hidden_size": 8}


def test_structure_probe_runs_prediction_inside_bf16_autocast(tmp_path: Path) -> None:
    state = {"autocast_enabled": False}

    @contextlib.contextmanager
    def _autocast(*, device_type: str, dtype: str) -> Iterator[None]:
        assert device_type == "cuda"
        assert dtype == "bf16"
        state["autocast_enabled"] = True
        try:
            yield
        finally:
            state["autocast_enabled"] = False

    class _Model:
        def predict_structure(
            self,
            sequence: str,
            *,
            recycling_steps: int,
            num_sampling_steps: int,
            diffusion_samples: int,
        ) -> str:
            del self
            assert state["autocast_enabled"]
            assert (recycling_steps, num_sampling_steps, diffusion_samples) == (1, 2, 1)
            return sequence

    torch = SimpleNamespace(
        autocast=_autocast,
        bfloat16="bf16",
        inference_mode=contextlib.nullcontext,
    )

    output = _exercise(
        _Model(),
        tmp_path,
        "boltz2",
        "fp32_parameters_autocast",
        torch,
    )

    assert output == "MSTNPKPQ"
    assert not state["autocast_enabled"]


def test_package_probe_disables_remote_code_collection_only_while_saving(
    tmp_path: Path,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.observed: list[bool] = []

        @classmethod
        def is_remote_code(cls) -> bool:
            return True

        def save_pretrained(self, _path: Path, *, safe_serialization: bool) -> None:
            assert safe_serialization
            self.observed.append(self.is_remote_code())

    model = FakeModel()
    _save_model_for_probe(model, tmp_path, "package")
    assert model.observed == [False]
    assert model.is_remote_code()


@pytest.mark.parametrize(
    ("model_id", "family", "repository_name", "auto_class", "class_path"),
    _cases(),
)
def test_local_artifact_offline_autoclass_parity(
    model_id: str,
    family: str,
    repository_name: str,
    auto_class: str,
    class_path: str,
    tmp_path: Path,
) -> None:
    """Load offline, infer, save/reload, and match unchanged package source."""

    artifact = ROOT / "dist" / "hub" / repository_name
    assert artifact.is_dir(), f"Missing required built artifact for {model_id}: {artifact}"
    artifact_output = tmp_path / "artifact.json"
    package_output = tmp_path / "package.json"

    isolated = _run_probe(
        artifact=artifact,
        family=family,
        auto_class=auto_class,
        class_path=class_path,
        implementation="artifact",
        output=artifact_output,
    )
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr
    package = _run_probe(
        artifact=artifact,
        family=family,
        auto_class=auto_class,
        class_path=class_path,
        implementation="package",
        output=package_output,
    )
    assert package.returncode == 0, package.stdout + package.stderr
    assert json.loads(artifact_output.read_text(encoding="utf-8")) == json.loads(
        package_output.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("family", "repository_name", "class_path", "attn_implementation"),
    (
        (
            "esm2",
            "ESM2-8M",
            "fastplms.models.esm2.modeling_fastesm.FastEsmModel",
            "flash_attention_2",
        ),
        (
            "esm2",
            "ESM2-8M",
            "fastplms.models.esm2.modeling_fastesm.FastEsmModel",
            "flash_attention_3",
        ),
        (
            "esm_plusplus",
            "ESMplusplus_small",
            "fastplms.models.esm_plusplus.modeling_esm_plusplus.ESMplusplusModel",
            "flash_attention_2",
        ),
        (
            "esm_plusplus",
            "ESMplusplus_small",
            "fastplms.models.esm_plusplus.modeling_esm_plusplus.ESMplusplusModel",
            "flash_attention_3",
        ),
        (
            "dplm",
            "DPLM-150M",
            "fastplms.models.dplm.modeling_dplm.DPLMModel",
            "flash_attention_3",
        ),
    ),
)
@pytest.mark.artifact
@pytest.mark.gpu
@pytest.mark.slow
def test_local_artifact_locked_flash_backend(
    family: str,
    repository_name: str,
    class_path: str,
    attn_implementation: str,
    tmp_path: Path,
) -> None:
    """Exercise every advertised precompiled Flash backend in isolation."""

    artifact = ROOT / "dist" / "hub" / repository_name
    artifact_output = tmp_path / "artifact.json"
    package_output = tmp_path / "package.json"
    common = {
        "artifact": artifact,
        "family": family,
        "auto_class": "AutoModel",
        "class_path": class_path,
        "attn_implementation": attn_implementation,
    }
    isolated = _run_probe(
        **common,
        implementation="artifact",
        output=artifact_output,
    )
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr
    package = _run_probe(
        **common,
        implementation="package",
        output=package_output,
    )
    assert package.returncode == 0, package.stdout + package.stderr
    assert json.loads(artifact_output.read_text(encoding="utf-8")) == json.loads(
        package_output.read_text(encoding="utf-8")
    )
