"""Release contracts for intentionally scoped optional dependencies."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tools.remote.runtime_import_closure import (
    RuntimeImportClosureError,
    inspect_runtime_import_closure,
)

ROOT = Path(__file__).resolve().parents[2]


def test_uv_cpu_extra_is_explicit_locked_and_conflicts_with_cuda_extras() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    uv = project["tool"]["uv"]

    assert extras["cpu"] == ["torch==2.13.0"]
    assert uv["sources"]["torch"] == [
        {"index": "pytorch-cpu", "extra": "cpu"},
    ]
    assert uv["index"] == [
        {
            "name": "pytorch-cpu",
            "url": "https://download.pytorch.org/whl/cpu",
            "explicit": True,
        }
    ]
    assert project["tool"]["uv"]["conflicts"] == [
        [{"extra": "cpu"}, {"extra": "cueq"}],
        [{"extra": "cpu"}, {"extra": "fp8"}],
    ]
    assert "--all-extras" not in (ROOT / ".github/workflows/cpu-contracts.yml").read_text(
        encoding="utf-8"
    )

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    torch_packages = [package for package in lock["package"] if package["name"] == "torch"]
    cpu_torch = [
        package
        for package in torch_packages
        if package["source"]["registry"] == "https://download.pytorch.org/whl/cpu"
    ]
    assert {package["version"] for package in cpu_torch} == {"2.13.0", "2.13.0+cpu"}
    forbidden = ("cuda", "nvidia", "triton")
    assert all(
        not dependency["name"].startswith(forbidden)
        for package in cpu_torch
        for dependency in package.get("dependencies", [])
    )
    assert any(
        package["source"]["registry"] == "https://pypi.org/simple" for package in torch_packages
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "| `cpu` | CPU-only PyTorch 2.13 selection" in readme


def test_structure_extra_is_runtime_owned_or_documented_integration() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    structure = project["project"]["optional-dependencies"]["structure"]

    assert structure == [
        "accelerate>=1.10,<2",  # Transformers device_map in the 6B quick start.
        "biopython>=1.85,<2",
        "biotite>=1.4,<2",
        "brotli>=1.1,<2",
        "msgpack>=1.1,<2",
        "msgpack-numpy>=0.4.8,<1",
        "omegaconf>=2.3,<3",  # Explicit trusted Boltz Lightning import boundary.
        "rdkit>=2025.9,<2027",
        "scipy>=1.15,<2",
        "zstandard>=0.23,<1",
    ]


def test_binder_extra_is_bounded_locked_and_separate_from_structure() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert extras["binder"] == [
        "abnumber==0.4.4",
        "anarcii==2.0.8",
        "pandas>=3.0,<3.1",
        "pyarrow>=25,<26",
    ]
    binder_names = {
        requirement.partition("==")[0].partition(">=")[0] for requirement in extras["binder"]
    }
    structure_names = {
        requirement.partition("==")[0].partition(">=")[0] for requirement in extras["structure"]
    }
    assert binder_names.isdisjoint(structure_names)

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_versions = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in {"abnumber", "anarcii", "pandas", "pyarrow"}
    }
    assert locked_versions == {
        "abnumber": "0.4.4",
        "anarcii": "2.0.8",
        "pandas": "3.0.3",
        "pyarrow": "25.0.0",
    }

    for path in (ROOT / "README.md", ROOT / "docs" / "binder_design.md"):
        text = path.read_text(encoding="utf-8")
        assert "--extra binder" in text
        assert "--with abnumber" not in text


def test_cueq_backend_extra_is_version_aligned_cuda13_and_isolated() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert extras["cueq"] == [
        "cuequivariance==0.10.0; platform_system == 'Linux'",
        "cuequivariance-torch==0.10.0; platform_system == 'Linux'",
        "cuequivariance-ops-torch-cu13==0.10.0; platform_system == 'Linux'",
    ]
    assert not any("cuequivariance" in requirement for requirement in extras["structure"])

    source = (ROOT / "src/fastplms/models/esmfold2/modeling_esmfold2_common.py").read_text(
        encoding="utf-8"
    )
    assert 'find_spec("cuequivariance_ops_torch")' in source
    assert "'structure,cueq' extras" in source

    kernel_sources = [
        source,
        (ROOT / "src/fastplms/models/boltz/vb_layers_triangular_mult.py").read_text(
            encoding="utf-8"
        ),
        (ROOT / "src/fastplms/models/boltz/vb_tri_attn_primitives.py").read_text(encoding="utf-8"),
    ]
    for kernel_source in kernel_sources:
        assert "cuequivariance_torch.primitives" not in kernel_source
        assert 'find_spec("cuequivariance_ops_torch")' in kernel_source
        assert 'import_module("cuequivariance_torch")' in kernel_source
    assert "cue_module.triangle_multiplicative_update" in source
    assert "cueq.triangle_multiplicative_update" in kernel_sources[1]
    assert "cueq.triangle_attention" in kernel_sources[2]

    documentation = (ROOT / "docs/esmfold2.md").read_text(encoding="utf-8")
    for contract in (
        "fastplms[structure,cueq]",
        "cuequivariance-torch==0.10.0",
        "cuequivariance-ops-torch-cu13==0.10.0",
        "Linux",
        "CUDA 13",
        "NVIDIA Software License Agreement",
    ):
        assert contract in documentation


def test_reporting_extra_is_separate_from_training_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert extras["reporting"] == [
        "matplotlib>=3.10,<4",
        "scikit-learn>=1.7,<2",
        "scipy>=1.15,<2",
        "seaborn>=0.13,<1",
    ]
    train_names = {requirement.partition(">=")[0] for requirement in extras["train"]}
    reporting_names = {requirement.partition(">=")[0] for requirement in extras["reporting"]}
    assert train_names.isdisjoint(reporting_names)


def test_runtime_import_closure_rejects_undeclared_literal_dynamic_import(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runtime"
    source_root.mkdir()
    (source_root / "dynamic.py").write_text(
        'import importlib\nimportlib.import_module("undeclared_dynamic_dependency")\n',
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeImportClosureError,
        match="undeclared literal dynamic dependencies",
    ):
        inspect_runtime_import_closure(source_root, ROOT / "pyproject.toml")
