"""Public validation must survive Python's optimized ``-O`` mode."""

from __future__ import annotations

import ast
import builtins
import os
import subprocess
import sys
import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _public_method(path: str, class_name: str, method_name: str) -> ast.FunctionDef:
    module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    child.name == method_name
                ):
                    if isinstance(child, ast.AsyncFunctionDef):
                        raise AssertionError(f"Unexpected async method {class_name}.{method_name}")
                    return child
    raise AssertionError(f"Missing public method {class_name}.{method_name}")


def _top_level_function(path: str, function_name: str) -> ast.FunctionDef:
    module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Missing public function {function_name}")


def _assert_no_runtime_asserts(function: ast.FunctionDef, label: str) -> None:
    runtime_asserts = [node for node in ast.walk(function) if isinstance(node, ast.Assert)]
    assert not runtime_asserts, f"{label} uses validation erased by python -O"


def test_boltz_public_boundaries_do_not_use_runtime_asserts() -> None:
    model_path = "src/fastplms/models/boltz/modeling_boltz2.py"
    config_method = _public_method(model_path, "Boltz2Config", "from_hyperparameters")
    _assert_no_runtime_asserts(config_method, "Boltz2Config.from_hyperparameters")
    for method_name in (
        "__init__",
        "from_boltz_checkpoint",
        "predict_structure",
        "save_as_cif",
    ):
        method = _public_method(model_path, "Boltz2Model", method_name)
        _assert_no_runtime_asserts(method, f"Boltz2Model.{method_name}")

    for function_name in ("_enforce_pairformer_v2", "_require_key"):
        function = _top_level_function(model_path, function_name)
        _assert_no_runtime_asserts(function, function_name)

    for function_name in ("_normalize_sequence", "build_boltz2_features"):
        function = _top_level_function(
            "src/fastplms/models/boltz/minimal_featurizer.py",
            function_name,
        )
        _assert_no_runtime_asserts(function, function_name)

    for function_name in ("_confidence_per_atom", "write_cif"):
        function = _top_level_function(
            "src/fastplms/models/boltz/cif_writer.py",
            function_name,
        )
        _assert_no_runtime_asserts(function, function_name)

    confidence_forward = _public_method(
        "src/fastplms/models/boltz/vb_modules_confidencev2.py",
        "ConfidenceModule",
        "forward",
    )
    _assert_no_runtime_asserts(confidence_forward, "ConfidenceModule.forward")

    indexing_matrix = _top_level_function(
        "src/fastplms/models/boltz/vb_modules_encodersv2.py",
        "get_indexing_matrix",
    )
    _assert_no_runtime_asserts(indexing_matrix, "get_indexing_matrix")
    for method_name in ("__init__", "compute"):
        schedule_method = _public_method(
            "src/fastplms/models/boltz/vb_potentials_schedules.py",
            "PiecewiseStepFunction",
            method_name,
        )
        _assert_no_runtime_asserts(
            schedule_method,
            f"PiecewiseStepFunction.{method_name}",
        )
    potential_method = _public_method(
        "src/fastplms/models/boltz/vb_potentials_potentials.py",
        "FlatBottomPotential",
        "compute_function",
    )
    _assert_no_runtime_asserts(
        potential_method,
        "FlatBottomPotential.compute_function",
    )


def test_boltz_encoder_schedule_and_potential_validation_survives_optimized_mode() -> None:
    script = r"""
import torch

from fastplms.models.boltz.vb_modules_encodersv2 import get_indexing_matrix
from fastplms.models.boltz.vb_potentials_potentials import FlatBottomPotential
from fastplms.models.boltz.vb_potentials_schedules import PiecewiseStepFunction


def must_raise(call, expected):
    try:
        call()
    except expected:
        return
    raise RuntimeError(f"Expected {expected.__name__}")


must_raise(
    lambda: get_indexing_matrix(1, 3, 6, torch.device("cpu")),
    ValueError,
)
must_raise(
    lambda: PiecewiseStepFunction((), (1.0,)),
    ValueError,
)
must_raise(
    lambda: FlatBottomPotential.compute_function(
        object(),
        value=torch.tensor([0.5]),
        k=torch.tensor(1.0),
        lower_bounds=torch.tensor([0.0]),
        upper_bounds=torch.tensor([1.0]),
        negation_mask=torch.tensor([False]),
    ),
    ValueError,
)
"""
    environment = os.environ.copy()
    source_root = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_esmfold2_public_boundaries_do_not_use_runtime_asserts() -> None:
    model_path = "src/fastplms/models/esmfold2/modeling_esmfold2.py"
    for method_name in (
        "_ensure_ttt_lm_head",
        "_ttt_tokenize",
        "_ttt_predict_logits",
        "_fold_protein_no_ttt",
        "fold_protein_ttt",
        "result_to_cif",
        "result_to_pdb",
    ):
        method = _public_method(model_path, "ESMFold2Model", method_name)
        _assert_no_runtime_asserts(method, f"ESMFold2Model.{method_name}")

    for path, class_name, method_name in (
        (
            "src/fastplms/models/esmfold2/esmfold2_protein_chain.py",
            "ProteinChain",
            "__post_init__",
        ),
        (
            "src/fastplms/models/esmfold2/esmfold2_protein_complex.py",
            "ProteinComplex",
            "__post_init__",
        ),
        (
            "src/fastplms/models/esmfold2/esmfold2_msa.py",
            "MSA",
            "__post_init__",
        ),
        (
            "src/fastplms/models/esmfold2/esmfold2_molecular_complex.py",
            "MolecularComplex",
            "__post_init__",
        ),
    ):
        method = _public_method(path, class_name, method_name)
        _assert_no_runtime_asserts(method, f"{class_name}.{method_name}")

    for method_name in (
        "from_atom37",
        "from_backbone_atom_coordinates",
        "from_mmcif",
        "chain_iterable_from_mmcif",
        "find_nonpolymer_contacts",
    ):
        method = _public_method(
            "src/fastplms/models/esmfold2/esmfold2_protein_chain.py",
            "ProteinChain",
            method_name,
        )
        _assert_no_runtime_asserts(method, f"ProteinChain.{method_name}")

    experimental_path = "src/fastplms/models/esmfold2/modeling_esmfold2_experimental.py"
    for method_name in ("_compute_lm_hidden_states", "result_to_cif", "result_to_pdb"):
        method = _public_method(
            experimental_path,
            "ESMFold2ExperimentalModel",
            method_name,
        )
        _assert_no_runtime_asserts(
            method,
            f"ESMFold2ExperimentalModel.{method_name}",
        )

    greedy = _top_level_function(
        "src/fastplms/models/esmfold2/esmfold2_msa_filter_sequences.py",
        "greedy_select_indices",
    )
    _assert_no_runtime_asserts(greedy, "greedy_select_indices")

    comparable = _public_method(
        "src/fastplms/models/esmfold2/esmfold2_protein_complex.py",
        "ProteinComplex",
        "_sanity_check_complexes_are_comparable",
    )
    _assert_no_runtime_asserts(
        comparable,
        "ProteinComplex._sanity_check_complexes_are_comparable",
    )
    table_validation = _top_level_function(
        "src/fastplms/models/esmfold2/esmfold2_molecular_complex.py",
        "_assert_table_lengths",
    )
    _assert_no_runtime_asserts(table_validation, "MolecularComplex table validation")


def test_attention_ankh_and_esmc_contracts_do_not_use_runtime_asserts() -> None:
    for class_name, method_names in (
        ("IndexFirstAxis", ("forward", "backward")),
        ("IndexPutFirstAxis", ("forward",)),
    ):
        for method_name in method_names:
            method = _public_method(
                "src/fastplms/attention/_core.py",
                class_name,
                method_name,
            )
            _assert_no_runtime_asserts(method, f"{class_name}.{method_name}")

    decoder_inputs = _public_method(
        "src/fastplms/models/ankh/modeling_ankh.py",
        "FastAnkhForConditionalGeneration",
        "_prepare_decoder_embedding_inputs",
    )
    _assert_no_runtime_asserts(
        decoder_inputs,
        "FastAnkhForConditionalGeneration._prepare_decoder_embedding_inputs",
    )

    for class_name, method_name in (
        ("RotaryEmbedding", "forward"),
        ("TransformerStack", "forward"),
    ):
        method = _public_method(
            "src/fastplms/models/esm_plusplus/modeling_esm_plusplus.py",
            class_name,
            method_name,
        )
        _assert_no_runtime_asserts(method, f"{class_name}.{method_name}")


def test_embedding_and_state_validation_contracts_do_not_use_runtime_asserts() -> None:
    for path, function_name in (
        ("src/fastplms/embeddings/runner.py", "embed_dataset"),
        ("src/fastplms/embeddings/storage.py", "load_sqlite_result"),
        ("tools/conversion/state_validation.py", "assert_model_parameters_fp32"),
        (
            "tools/conversion/state_validation.py",
            "assert_state_dict_floating_tensors_fp32",
        ),
        ("tools/conversion/state_validation.py", "assert_state_dict_equal"),
    ):
        function = _top_level_function(path, function_name)
        _assert_no_runtime_asserts(function, function_name)


_COMPOSITE_GUARD_PATHS = (
    "src/fastplms/models/ankh/modeling_ankh.py",
    "src/fastplms/models/dplm/modeling_dplm.py",
    "src/fastplms/models/dplm2/modeling_dplm2.py",
    "src/fastplms/models/e1/modeling_e1.py",
    "src/fastplms/models/esm2/modeling_fastesm.py",
    "src/fastplms/models/esm3/modeling_esm3.py",
    "src/fastplms/models/esm_plusplus/modeling_esm_plusplus.py",
    "src/fastplms/models/esmfold/modeling_fast_esmfold.py",
)


def _fastplms_import_guard(path: str) -> ast.Try:
    module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Try):
            continue
        if any(
            isinstance(child, ast.ImportFrom)
            and child.module is not None
            and child.module.startswith("fastplms")
            for child in node.body
        ):
            return node
    raise AssertionError(f"Missing FastPLMs import guard in {path}")


def _guard_required_names(guard: ast.Try) -> set[str]:
    return {
        alias.asname or alias.name
        for child in guard.body
        if isinstance(child, ast.ImportFrom)
        for alias in child.names
    }


def _execute_guard(guard: ast.Try, *, missing_name: str, shared: set[str]) -> None:
    original_import = builtins.__import__

    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("fastplms"):
            raise ModuleNotFoundError(
                f"synthetic missing dependency: {missing_name}",
                name=missing_name,
            )
        return original_import(name, globals, locals, fromlist, level)

    namespace = {name: object() for name in shared}
    namespace["__builtins__"] = {**vars(builtins), "__import__": unavailable}
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
        "<composite-import-guard>",
        "exec",
    )
    exec(code, namespace)


@pytest.mark.parametrize("path", _COMPOSITE_GUARD_PATHS)
def test_composite_guards_preserve_transitive_import_errors(path: str) -> None:
    guard = _fastplms_import_guard(path)
    with pytest.raises(ModuleNotFoundError) as captured:
        _execute_guard(
            guard,
            missing_name="synthetic_runtime_dependency",
            shared=_guard_required_names(guard),
        )
    assert captured.value.name == "synthetic_runtime_dependency"


@pytest.mark.parametrize("path", _COMPOSITE_GUARD_PATHS)
def test_composite_guards_require_every_predefined_shared_symbol(path: str) -> None:
    guard = _fastplms_import_guard(path)
    required = _guard_required_names(guard)
    missing_symbol = sorted(required)[0]
    with pytest.raises(ModuleNotFoundError) as captured:
        _execute_guard(
            guard,
            missing_name="fastplms",
            shared=required - {missing_symbol},
        )
    assert captured.value.name == "fastplms"


@pytest.mark.parametrize("path", _COMPOSITE_GUARD_PATHS)
def test_composite_guards_allow_complete_legacy_flat_context(path: str) -> None:
    guard = _fastplms_import_guard(path)
    _execute_guard(
        guard,
        missing_name="fastplms",
        shared=_guard_required_names(guard),
    )


def test_esmfold2_fallback_does_not_mask_transitive_import_errors() -> None:
    guard = _fastplms_import_guard("src/fastplms/models/esmfold2/modeling_esmfold2.py")
    handler = guard.handlers[0]
    assert isinstance(handler.type, ast.Name)
    assert handler.type.id == "ModuleNotFoundError"
    assert any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "fastplms"
            for comparator in node.comparators
        )
        for node in ast.walk(handler)
    )


def test_invalid_attention_ankh_and_esmc_inputs_fail_under_python_optimized_mode() -> None:
    script = r"""
import torch

from fastplms.attention._core import index_first_axis, index_put_first_axis
from fastplms.models.ankh.modeling_ankh import FastAnkhForConditionalGeneration
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    RotaryEmbedding,
    TransformerStack,
)


def must_raise(call, expected):
    try:
        call()
    except expected:
        return
    raise RuntimeError(f"Expected {expected.__name__}")


must_raise(
    lambda: index_first_axis(torch.ones(3), torch.tensor([0])),
    ValueError,
)
must_raise(
    lambda: index_first_axis(torch.ones(3, 2), torch.tensor([[0]])),
    ValueError,
)
must_raise(
    lambda: index_put_first_axis(torch.ones(1, 2), torch.tensor([[0]]), 3),
    ValueError,
)
must_raise(
    lambda: FastAnkhForConditionalGeneration._prepare_decoder_embedding_inputs(
        object(),
        batch_size=1,
        decoder_inputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
    ),
    ValueError,
)

rotary = RotaryEmbedding(8)
rotary._update_cos_sin_cache = lambda *args, **kwargs: None
must_raise(
    lambda: rotary(torch.randn(1, 3, 2, 8), torch.randn(1, 3, 2, 8)),
    RuntimeError,
)

stack = TransformerStack(16, 2, 1, attn_backend="sdpa")
must_raise(
    lambda: stack(
        torch.randn(2, 3, 16),
        attention_mask=torch.ones(2, 3, 1),
    ),
    ValueError,
)
"""
    environment = os.environ.copy()
    source_root = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_state_validation_failures_survive_python_optimized_mode() -> None:
    script = r"""
import torch

from tools.conversion.state_validation import (
    assert_model_parameters_fp32,
    assert_state_dict_equal,
    assert_state_dict_floating_tensors_fp32,
)


def must_raise(call):
    try:
        call()
    except AssertionError:
        return
    raise RuntimeError("Expected AssertionError")


must_raise(lambda: assert_model_parameters_fp32(torch.nn.Identity(), "empty"))
must_raise(
    lambda: assert_state_dict_floating_tensors_fp32(
        {"weight": torch.ones(1, dtype=torch.float16)},
        "half-state",
    )
)
must_raise(
    lambda: assert_state_dict_equal(
        {"weight": torch.ones(1)},
        {"weight": torch.zeros(1)},
        "mismatch",
    )
)
"""
    environment = os.environ.copy()
    source_root = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), source_root, environment.get("PYTHONPATH", "")) if value
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
