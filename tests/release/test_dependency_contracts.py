"""Release contracts for intentionally scoped optional dependencies."""

from __future__ import annotations

import pytest

from pathlib import Path
from packaging.requirements import Requirement
from packaging.version import Version
from tests.conftest import validation_pins

from tools.remote.runtime_import_closure import (
    RuntimeImportClosureError,
    inspect_runtime_import_closure,
)
from tools.typing_gate import BASELINE_MYPY_VERSION


ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements"


def _requirements(relative_path: str) -> list[str]:
    path = REQUIREMENTS / relative_path
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _declarations(relative_path: str) -> list[Requirement]:
    return [Requirement(requirement) for requirement in _requirements(relative_path)]


def _operators(requirement: Requirement) -> list[str]:
    return sorted(clause.operator for clause in requirement.specifier)


def test_every_range_is_a_floor_without_a_cap() -> None:
    """A range starts at the tested release and admits every later one; a pin stays exact.

    A cap is allowed only directly beneath a comment naming the incompatibility behind it.
    """
    declaration_files = [
        *sorted(REQUIREMENTS.glob("*.in")),
        *sorted((REQUIREMENTS / "features").glob("*.in")),
        REQUIREMENTS / "constraints" / "validation.txt",
    ]
    for path in declaration_files:
        explained = False
        for line in path.read_text(encoding="utf-8").splitlines():
            declaration = line.strip()
            if not declaration or declaration.startswith("#"):
                explained = declaration.startswith("#")
                continue
            requirement = Requirement(declaration)
            allowed = [[">="], ["=="], *([["<", ">="]] if explained else [])]
            assert _operators(requirement) in allowed, f"{path.name}: {requirement}"
            explained = False


def test_mypy_cap_holds_only_while_the_typing_baseline_needs_it() -> None:
    """dev.in caps mypy because tools/typing_gate.py compares only on its baseline's mypy.

    Recording the baseline on a mypy the cap excludes fails this test, which is the cue
    to move or drop the cap.
    """
    (mypy,) = [
        requirement
        for requirement in _declarations("features/dev.in")
        if requirement.name == "mypy"
    ]
    baseline = Version(BASELINE_MYPY_VERSION)
    assert mypy.specifier.contains(BASELINE_MYPY_VERSION)
    # A resolver must not reach the next mypy major while the baseline is on this one.
    assert not mypy.specifier.contains(f"{baseline.major + 1}.0")


def test_core_dependencies_are_direct_and_floored_at_the_validated_stack() -> None:
    core = {requirement.name: requirement for requirement in _declarations("core.in")}
    assert list(core) == [
        "torch",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "numpy",
        "einops",
        "tqdm",
    ]
    assert all(_operators(requirement) == [">="] for requirement in core.values())
    # The declared floor of each pinned package is the validated release line.
    for distribution, pinned in validation_pins().items():
        (floor,) = core[distribution].specifier
        assert Version(floor.version).release[:2] == Version(pinned).release[:2], distribution
        assert core[distribution].specifier.contains(pinned), distribution


def test_cpu_validation_profile_is_explicit_and_cuda_free() -> None:
    pins = validation_pins()
    assert list(pins) == ["torch", "transformers"]
    assert _requirements("features/cpu.in") == [f"torch=={pins['torch']}"]
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
    structure = _declarations("features/structure.in")
    assert [requirement.name for requirement in structure] == [
        "accelerate",  # Transformers device_map in the 6B quick start.
        "biopython",
        "biotite",
        "brotli",
        "msgpack",
        "msgpack-numpy",
        "omegaconf",  # Explicit trusted Boltz Lightning import boundary.
        "rdkit",
        "scipy",
        "zstandard",
    ]
    assert all(_operators(requirement) == [">="] for requirement in structure)


def test_binder_dependencies_are_pinned_or_floored_and_separate_from_structure() -> None:
    binder = _declarations("features/binder.in")
    structure = _declarations("features/structure.in")
    # The numbering tools stay exact; the table libraries follow the tested stack.
    assert {requirement.name: _operators(requirement) for requirement in binder} == {
        "abnumber": ["=="],
        "anarcii": ["=="],
        "pandas": [">="],
        "pyarrow": [">="],
    }
    assert {requirement.name for requirement in binder}.isdisjoint(
        {requirement.name for requirement in structure}
    )
    assert _requirements("profiles/binder.in") == [
        "-r ../core.in",
        "-r ../features/structure.in",
        "-r ../features/binder.in",
    ]


def test_cueq_dependencies_are_version_aligned_cuda13_and_isolated() -> None:
    cueq = _declarations("features/cueq.in")
    structure = _requirements("features/structure.in")
    assert [requirement.name for requirement in cueq] == [
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu13",
    ]
    # One exact release across the three packages, installed on Linux only.
    assert all(_operators(requirement) == ["=="] for requirement in cueq)
    assert len({str(requirement.specifier) for requirement in cueq}) == 1
    assert {str(requirement.marker) for requirement in cueq} == {'platform_system == "Linux"'}
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
    reporting = _declarations("features/reporting.in")
    training = _declarations("features/train.in")
    assert [requirement.name for requirement in reporting] == [
        "matplotlib",
        "scikit-learn",
        "scipy",
        "seaborn",
    ]
    assert all(_operators(requirement) == [">="] for requirement in reporting)
    assert {requirement.name for requirement in training}.isdisjoint(
        {requirement.name for requirement in reporting}
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
