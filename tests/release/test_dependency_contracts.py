"""Release contracts for intentionally scoped optional dependencies."""

from __future__ import annotations

import pytest
from pathlib import Path

from tools.remote.runtime_import_closure import (
    RuntimeImportClosureError,
    inspect_runtime_import_closure,
)


ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements"


def _requirements(relative_path: str) -> list[str]:
    path = REQUIREMENTS / relative_path
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _package_name(requirement: str) -> str:
    name = requirement.partition(";")[0]
    for operator in ("==", ">="):
        name = name.partition(operator)[0]
    return name.strip()


def test_core_dependencies_are_direct_and_bounded() -> None:
    assert _requirements("core.in") == [
        "torch>=2.13,<2.14",
        "transformers>=5.13,<5.14",
        "huggingface-hub>=0.34,<2",
        "tokenizers>=0.22,<0.23",
        "safetensors>=0.5,<1",
        "numpy>=1.26,<3",
        "einops>=0.8,<1",
        "tqdm>=4.67,<5",
    ]


def test_cpu_validation_profile_is_explicit_and_cuda_free() -> None:
    assert _requirements("features/cpu.in") == ["torch==2.13.0"]
    assert _requirements("constraints/validation.txt") == [
        "torch==2.13.0",
        "transformers==5.13.0",
    ]
    assert _requirements("profiles/cpu-validation.in") == [
        "-r ../core.in",
        "-r ../features/cpu.in",
        "-r ../features/dev.in",
        "-r ../features/structure.in",
        "-r ../features/train.in",
    ]
    for profile in (REQUIREMENTS / "profiles").glob("*.in"):
        declarations = profile.read_text(encoding="utf-8")
        if "features/cpu.in" not in declarations:
            continue
        assert "features/cueq.in" not in declarations
        assert "features/fp8.in" not in declarations
    instructions = (REQUIREMENTS / "README.md").read_text(encoding="utf-8")
    assert "--torch-backend cpu" in instructions
    assert "requirements/constraints/validation.txt" in instructions


def test_structure_dependencies_are_runtime_owned_or_documented_integrations() -> None:
    assert _requirements("features/structure.in") == [
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


def test_binder_dependencies_are_bounded_and_separate_from_structure() -> None:
    binder = _requirements("features/binder.in")
    structure = _requirements("features/structure.in")
    assert binder == [
        "abnumber==0.4.4",
        "anarcii==2.0.8",
        "pandas>=3.0,<3.1",
        "pyarrow>=25,<26",
    ]
    assert {_package_name(item) for item in binder}.isdisjoint(
        {_package_name(item) for item in structure}
    )
    assert _requirements("profiles/binder.in") == [
        "-r ../core.in",
        "-r ../features/structure.in",
        "-r ../features/binder.in",
    ]


def test_cueq_dependencies_are_version_aligned_cuda13_and_isolated() -> None:
    cueq = _requirements("features/cueq.in")
    structure = _requirements("features/structure.in")
    assert cueq == [
        'cuequivariance==0.10.0; platform_system == "Linux"',
        'cuequivariance-torch==0.10.0; platform_system == "Linux"',
        'cuequivariance-ops-torch-cu13==0.10.0; platform_system == "Linux"',
    ]
    assert not any("cuequivariance" in requirement for requirement in structure)

    source = (ROOT / "src/fastplms/models/esmfold2/modeling_esmfold2_common.py").read_text(
        encoding="utf-8"
    )
    assert 'find_spec("cuequivariance_ops_torch")' in source

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
    assert _requirements("profiles/candidate-structure.in")[-2:] == [
        "-r ../features/cueq.in",
        "-r ../features/train.in",
    ]


def test_reporting_dependencies_are_separate_from_training_runtime() -> None:
    reporting = _requirements("features/reporting.in")
    training = _requirements("features/train.in")
    assert reporting == [
        "matplotlib>=3.10,<4",
        "scikit-learn>=1.7,<2",
        "scipy>=1.15,<2",
        "seaborn>=0.13,<1",
    ]
    assert {_package_name(item) for item in training}.isdisjoint(
        {_package_name(item) for item in reporting}
    )


def test_dependency_instructions_install_the_cpu_validation_profile() -> None:
    instructions = (REQUIREMENTS / "README.md").read_text(encoding="utf-8")

    assert "uv pip install" in instructions
    assert "-r requirements/profiles/cpu-validation.in" in instructions
    assert "-c requirements/constraints/validation.txt" in instructions


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
        inspect_runtime_import_closure(source_root, ROOT / "requirements")


def test_runtime_import_closure_rejects_optional_extra_as_core_import(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runtime"
    source_root.mkdir()
    (source_root / "unconditional.py").write_text("import pandas\n", encoding="utf-8")

    with pytest.raises(
        RuntimeImportClosureError,
        match="Unconditional import dependency scope mismatch",
    ):
        inspect_runtime_import_closure(source_root, ROOT / "requirements")


def test_runtime_import_closure_keeps_top_level_control_flow_import_time(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runtime"
    source_root.mkdir()
    (source_root / "conditional.py").write_text(
        "enabled = True\n"
        "if enabled:\n"
        "    import pandas\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeImportClosureError,
        match="Unconditional import dependency scope mismatch",
    ):
        inspect_runtime_import_closure(source_root, ROOT / "requirements")


def test_runtime_import_closure_records_guarded_dependency_intended_extra(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runtime"
    source_root.mkdir()
    (source_root / "guarded.py").write_text(
        "import importlib\n"
        "def load_kernel():\n"
        '    return importlib.import_module("kernels")\n',
        encoding="utf-8",
    )

    payload = inspect_runtime_import_closure(source_root, ROOT / "requirements")

    assert payload["feature_gated_dynamic_imports"] == [
        {
            "declared_scopes": ["extra:flash"],
            "kind": "dynamic",
            "line": 3,
            "module": "kernels",
            "required_scope": "extra:flash",
            "source": "guarded.py",
            "source_scope": "core",
        }
    ]


def test_runtime_import_closure_uses_manifest_scope_for_feature_module(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runtime"
    module_root = source_root / "models" / "feature"
    module_root.mkdir(parents=True)
    (source_root / "models.toml").write_text(
        "[families.feature]\n"
        'extra = "structure"\n'
        'runtime_paths = ["models/feature"]\n',
        encoding="utf-8",
    )
    (module_root / "module.py").write_text("import scipy\n", encoding="utf-8")

    payload = inspect_runtime_import_closure(source_root, ROOT / "requirements")

    assert payload["import_time_dependencies"] == [
        {
            "declared_scopes": ["extra:reporting", "extra:structure"],
            "kind": "static",
            "line": 1,
            "module": "scipy",
            "required_scope": "extra:structure",
            "source": "models/feature/module.py",
            "source_scope": "extra:structure",
        }
    ]


def test_runtime_import_closure_rejects_escaping_manifest_runtime_path(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runtime"
    source_root.mkdir()
    (source_root / "module.py").write_text("import torch\n", encoding="utf-8")
    (source_root / "models.toml").write_text(
        "[families.feature]\n"
        'extra = "structure"\n'
        'runtime_paths = ["../outside"]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeImportClosureError,
        match="non-portable runtime path",
    ):
        inspect_runtime_import_closure(source_root, ROOT / "requirements")


def test_runtime_import_closure_rejects_ambiguous_guarded_extra(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runtime"
    source_root.mkdir()
    (source_root / "guarded.py").write_text(
        "def load_accelerate():\n"
        "    import accelerate\n"
        "    return accelerate\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeImportClosureError,
        match="does not map to one intended dependency scope",
    ):
        inspect_runtime_import_closure(source_root, ROOT / "requirements")
