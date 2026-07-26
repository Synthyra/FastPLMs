"""Explicit runtime evidence for every family-level advertised AutoClass."""

from __future__ import annotations

import importlib
import pytest
from dataclasses import dataclass
from transformers import PretrainedConfig

from fastplms.registry import get_model_registry


@dataclass(frozen=True)
class AutoClassEvidence:
    symbol_path: str
    capabilities: frozenset[str]
    runtime_tests: tuple[str, ...]
    case_parameter: str | None = None
    limitations: tuple[str, ...] = ()


_CONFIG_TEST = (
    "tests/cpu/test_autoclass_evidence_matrix.py::"
    "test_every_advertised_config_round_trips_offline"
)
_SEQUENCE_TEST = (
    "tests/cpu/test_sequence_autoclass_contracts.py::"
    "test_esm2_advertised_models_forward_loss_backward_resize_and_reload"
)
_ESMC_TEST = (
    "tests/cpu/test_sequence_autoclass_contracts.py::"
    "test_esmc_public_models_forward_loss_backward_resize_and_reload"
)
_DPLM_TEST = (
    "tests/cpu/test_sequence_autoclass_contracts.py::"
    "test_dplm_advertised_models_forward_loss_backward_resize_and_reload"
)
_E1_TEST = (
    "tests/cpu/test_sequence_autoclass_contracts.py::"
    "test_e1_advertised_models_forward_loss_backward_resize_and_reload"
)
_ANKH_TEST = (
    "tests/cpu/test_ankh_contracts.py::test_ankh_encoder_views_backward_and_save_reload"
)
_BASE_CAPABILITIES = frozenset(
    {"backward", "forward", "resize", "return_dict", "save_reload", "tuple"}
)
_HEAD_CAPABILITIES = _BASE_CAPABILITIES | {"loss"}


def _config(symbol_path: str) -> AutoClassEvidence:
    return AutoClassEvidence(
        symbol_path=symbol_path,
        capabilities=frozenset({"construct", "serialize", "reload"}),
        runtime_tests=(_CONFIG_TEST,),
        case_parameter="symbol_path",
    )


def _sequence(
    symbol_path: str,
    runtime_test: str,
    *,
    loss: bool,
) -> AutoClassEvidence:
    return AutoClassEvidence(
        symbol_path=symbol_path,
        capabilities=_HEAD_CAPABILITIES if loss else _BASE_CAPABILITIES,
        runtime_tests=(runtime_test,),
        case_parameter="model_class",
        limitations=() if loss else ("supervised loss is not applicable to a base model",),
    )


AUTOCLASS_EVIDENCE: dict[tuple[str, str], AutoClassEvidence] = {
    ("esm2", "AutoConfig"): _config(
        "fastplms.models.esm2.modeling_fastesm.FastEsmConfig"
    ),
    ("esm2", "AutoModel"): _sequence(
        "fastplms.models.esm2.modeling_fastesm.FastEsmModel",
        _SEQUENCE_TEST,
        loss=False,
    ),
    ("esm2", "AutoModelForMaskedLM"): _sequence(
        "fastplms.models.esm2.modeling_fastesm.FastEsmForMaskedLM",
        _SEQUENCE_TEST,
        loss=True,
    ),
    ("esm2", "AutoModelForSequenceClassification"): _sequence(
        "fastplms.models.esm2.modeling_fastesm.FastEsmForSequenceClassification",
        _SEQUENCE_TEST,
        loss=True,
    ),
    ("esm2", "AutoModelForTokenClassification"): _sequence(
        "fastplms.models.esm2.modeling_fastesm.FastEsmForTokenClassification",
        _SEQUENCE_TEST,
        loss=True,
    ),
    ("esm_plusplus", "AutoConfig"): _config(
        "fastplms.models.esm_plusplus.modeling_esm_plusplus.ESMplusplusConfig"
    ),
    ("esm_plusplus", "AutoModel"): _sequence(
        "fastplms.models.esm_plusplus.modeling_esm_plusplus.ESMplusplusModel",
        _ESMC_TEST,
        loss=False,
    ),
    ("esm_plusplus", "AutoModelForMaskedLM"): _sequence(
        "fastplms.models.esm_plusplus.modeling_esm_plusplus.ESMplusplusForMaskedLM",
        _ESMC_TEST,
        loss=True,
    ),
    ("esm3", "AutoConfig"): _config(
        "fastplms.models.esm3.modeling_esm3.FastESM3Config"
    ),
    ("esm3", "AutoModel"): AutoClassEvidence(
        symbol_path="fastplms.models.esm3.modeling_esm3.FastESM3Model",
        capabilities=_BASE_CAPABILITIES | {"multimodal_generation"},
        runtime_tests=(
            "tests/cpu/test_generation_contracts.py::"
            "test_esm3_uses_hugging_face_initialization_and_only_retains_requested_states",
            "tests/cpu/test_generation_contracts.py::test_esm3_advertised_model_logits_backward",
            "tests/cpu/test_generation_contracts.py::test_esm3_loads_with_automodel",
            "tests/cpu/test_generation_contracts.py::"
            "test_esm3_resize_updates_sequence_input_and_output_embeddings",
            "tests/cpu/test_generation_contracts.py::test_esm3_sequence_only_forward",
        ),
        limitations=("supervised loss is not applicable to the ESM3 base model",),
    ),
    ("e1", "AutoConfig"): _config("fastplms.models.e1.modeling_e1.E1Config"),
    ("e1", "AutoModel"): _sequence(
        "fastplms.models.e1.modeling_e1.E1Model",
        _E1_TEST,
        loss=False,
    ),
    ("e1", "AutoModelForMaskedLM"): _sequence(
        "fastplms.models.e1.modeling_e1.E1ForMaskedLM",
        _E1_TEST,
        loss=True,
    ),
    ("e1", "AutoModelForSequenceClassification"): _sequence(
        "fastplms.models.e1.modeling_e1.E1ForSequenceClassification",
        _E1_TEST,
        loss=True,
    ),
    ("e1", "AutoModelForTokenClassification"): _sequence(
        "fastplms.models.e1.modeling_e1.E1ForTokenClassification",
        _E1_TEST,
        loss=True,
    ),
    ("dplm", "AutoConfig"): _config(
        "fastplms.models.dplm.modeling_dplm.DPLMConfig"
    ),
    ("dplm", "AutoModel"): _sequence(
        "fastplms.models.dplm.modeling_dplm.DPLMModel",
        _DPLM_TEST,
        loss=False,
    ),
    ("dplm", "AutoModelForMaskedLM"): _sequence(
        "fastplms.models.dplm.modeling_dplm.DPLMForMaskedLM",
        _DPLM_TEST,
        loss=True,
    ),
    ("dplm", "AutoModelForSequenceClassification"): _sequence(
        "fastplms.models.dplm.modeling_dplm.DPLMForSequenceClassification",
        _DPLM_TEST,
        loss=True,
    ),
    ("dplm", "AutoModelForTokenClassification"): _sequence(
        "fastplms.models.dplm.modeling_dplm.DPLMForTokenClassification",
        _DPLM_TEST,
        loss=True,
    ),
    ("dplm2", "AutoConfig"): _config(
        "fastplms.models.dplm2.modeling_dplm2.DPLM2Config"
    ),
    ("dplm2", "AutoModel"): _sequence(
        "fastplms.models.dplm2.modeling_dplm2.DPLM2Model",
        _DPLM_TEST,
        loss=False,
    ),
    ("dplm2", "AutoModelForMaskedLM"): _sequence(
        "fastplms.models.dplm2.modeling_dplm2.DPLM2ForMaskedLM",
        _DPLM_TEST,
        loss=True,
    ),
    ("dplm2", "AutoModelForSequenceClassification"): _sequence(
        "fastplms.models.dplm2.modeling_dplm2.DPLM2ForSequenceClassification",
        _DPLM_TEST,
        loss=True,
    ),
    ("dplm2", "AutoModelForTokenClassification"): _sequence(
        "fastplms.models.dplm2.modeling_dplm2.DPLM2ForTokenClassification",
        _DPLM_TEST,
        loss=True,
    ),
    ("ankh", "AutoConfig"): _config(
        "fastplms.models.ankh.modeling_ankh.FastAnkhConfig"
    ),
    ("ankh", "AutoModel"): _sequence(
        "fastplms.models.ankh.modeling_ankh.FastAnkhModel",
        _ANKH_TEST,
        loss=False,
    ),
    ("ankh", "AutoModelForMaskedLM"): _sequence(
        "fastplms.models.ankh.modeling_ankh.FastAnkhForMaskedLMExtension",
        _ANKH_TEST,
        loss=True,
    ),
    ("ankh", "AutoModelForSeq2SeqLM"): AutoClassEvidence(
        symbol_path=(
            "fastplms.models.ankh.modeling_ankh.FastAnkhForConditionalGeneration"
        ),
        capabilities=_HEAD_CAPABILITIES | {"encoder_decoder_state"},
        runtime_tests=(
            "tests/cpu/test_ankh_contracts.py::"
            "test_complete_t5_checkpoint_loads_clean_encoder_and_seq2seq_views",
            "tests/cpu/test_ankh_contracts.py::"
            "test_ankh_seq2seq_view_honors_tuple_output_and_resize",
            "tests/cpu/test_ankh_contracts.py::"
            "test_seq2seq_head_produces_finite_loss_and_gradients",
        ),
    ),
    ("ankh", "AutoModelForSequenceClassification"): _sequence(
        "fastplms.models.ankh.modeling_ankh.FastAnkhForSequenceClassification",
        _ANKH_TEST,
        loss=True,
    ),
    ("ankh", "AutoModelForTokenClassification"): _sequence(
        "fastplms.models.ankh.modeling_ankh.FastAnkhForTokenClassification",
        _ANKH_TEST,
        loss=True,
    ),
    ("boltz2", "AutoConfig"): _config(
        "fastplms.models.boltz.modeling_boltz2.Boltz2Config"
    ),
    ("boltz2", "AutoModel"): AutoClassEvidence(
        symbol_path="fastplms.models.boltz.modeling_boltz2.Boltz2Model",
        capabilities=frozenset(
            {"backward", "forward", "output_flags", "return_dict", "save_reload", "tuple"}
        ),
        runtime_tests=(
            "tests/cpu/test_structure_contracts.py::"
            "test_boltz_public_forward_honors_output_controls_backward_and_reload",
        ),
        limitations=(
            "token embedding resize and classifier-style supervised loss are not applicable "
            "to a structure pipeline",
        ),
    ),
    ("esmfold", "AutoConfig"): _config(
        "fastplms.models.esmfold.modeling_fast_esmfold.FastEsmFoldConfig"
    ),
    ("esmfold", "AutoModel"): AutoClassEvidence(
        symbol_path=(
            "fastplms.models.esmfold.modeling_fast_esmfold.FastEsmForProteinFolding"
        ),
        capabilities=frozenset(
            {
                "backward",
                "forward",
                "multimer_infer",
                "output_flags",
                "return_dict",
                "save_reload",
                "tuple",
            }
        ),
        runtime_tests=(
            "tests/cpu/test_structure_contracts.py::"
            "test_fast_esmfold_public_forward_honors_output_controls_and_backward",
            "tests/cpu/test_structure_contracts.py::"
            "test_fast_esmfold_tiny_model_saves_and_reloads_exact_state",
            "tests/cpu/test_structure_contracts.py::"
            "test_esmfold_infer_preserves_official_multimer_contract",
        ),
        limitations=(
            "the CPU contract injects the folding core because a complete ESMFold trunk is "
            "a release-candidate checkpoint contract",
            "token resize and classifier-style task loss are not applicable",
        ),
    ),
    ("esmfold2", "AutoConfig"): _config(
        "fastplms.models.esmfold2.configuration_esmfold2.ESMFold2Config"
    ),
    ("esmfold2", "AutoModel"): AutoClassEvidence(
        symbol_path="fastplms.models.esmfold2.modeling_esmfold2.ESMFold2Model",
        capabilities=frozenset(
            {"backward", "forward", "output_flags", "return_dict", "save_reload", "tuple"}
        ),
        runtime_tests=(
            "tests/cpu/test_structure_contracts.py::"
            "test_esmfold2_public_forward_honors_output_controls_and_sampler_overrides",
            "tests/cpu/test_structure_contracts.py::"
            "test_esmfold2_advertised_models_tiny_init_backward_and_save_reload",
        ),
        limitations=(
            "token resize and classifier-style task loss are not applicable to the "
            "structure pipeline",
        ),
    ),
}


def _load_symbol(path: str) -> type:
    module_name, separator, symbol_name = path.rpartition(".")
    if not separator:
        raise AssertionError(f"Invalid symbol path: {path}")
    symbol = getattr(importlib.import_module(module_name), symbol_name)
    assert isinstance(symbol, type), path
    return symbol


def _manifest_family_entries() -> dict[tuple[str, str], str]:
    registry = get_model_registry()
    return {
        (family_id, auto_class): symbol_path
        for family_id, family in registry.families.items()
        for auto_class, symbol_path in family.auto_map.items()
    }


_CONFIG_CASES = tuple(
    (family_id, evidence.symbol_path)
    for (family_id, auto_class), evidence in AUTOCLASS_EVIDENCE.items()
    if auto_class == "AutoConfig"
)


@pytest.mark.parametrize(("family_id", "symbol_path"), _CONFIG_CASES)
def test_every_advertised_config_round_trips_offline(
    family_id: str,
    symbol_path: str,
) -> None:
    config_class = _load_symbol(symbol_path)
    assert issubclass(config_class, PretrainedConfig), family_id
    config = config_class()
    serialized = config.to_dict()
    reloaded = config_class.from_dict(serialized)

    assert isinstance(reloaded, config_class)
    assert reloaded.to_dict() == serialized


def test_autoclass_runtime_evidence_matrix_exactly_matches_all_37_entries() -> None:
    manifest_entries = _manifest_family_entries()
    assert len(manifest_entries) == 37
    assert set(AUTOCLASS_EVIDENCE) == set(manifest_entries)
    for key, evidence in AUTOCLASS_EVIDENCE.items():
        assert evidence.symbol_path == manifest_entries[key]
        _load_symbol(evidence.symbol_path)
        assert evidence.capabilities
        assert evidence.runtime_tests


def _collected_base_node_id(item: pytest.Item) -> str:
    return item.nodeid.replace("\\", "/").partition("[")[0]


def _case_parameter_symbol(item: pytest.Item, parameter: str) -> str | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None or parameter not in callspec.params:
        return None
    value = callspec.params[parameter]
    if isinstance(value, str):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return None


def test_autoclass_runtime_evidence_targets_are_collected_cpu_tests(
    request: pytest.FixtureRequest,
) -> None:
    collected: dict[str, list[pytest.Item]] = {}
    for item in request.session.items:
        collected.setdefault(_collected_base_node_id(item), []).append(item)

    for evidence in AUTOCLASS_EVIDENCE.values():
        for node_id in evidence.runtime_tests:
            cases = collected.get(node_id, [])
            assert cases, f"Runtime evidence target was not collected: {node_id}"
            if evidence.case_parameter is None:
                continue
            observed_symbols = {
                symbol
                for item in cases
                if (
                    symbol := _case_parameter_symbol(item, evidence.case_parameter)
                )
                is not None
            }
            assert evidence.symbol_path in observed_symbols, (
                f"{node_id} was collected, but not the {evidence.case_parameter!r} case "
                f"for {evidence.symbol_path!r}; observed {sorted(observed_symbols)!r}"
            )


def test_model_specific_automap_overrides_have_cpu_runtime_evidence() -> None:
    registry = get_model_registry()
    experimental_path = (
        "fastplms.models.esmfold2.modeling_esmfold2_experimental."
        "ESMFold2ExperimentalModel"
    )
    overridden_paths = {
        symbol_path
        for spec in registry.values()
        for auto_class, symbol_path in spec.auto_map.items()
        if symbol_path != spec.family.auto_map[auto_class]
    }
    assert overridden_paths == {experimental_path}
    _load_symbol(experimental_path)
    module = importlib.import_module("tests.cpu.test_structure_contracts")
    for test_name in (
        "test_esmfold2_advertised_models_tiny_init_backward_and_save_reload",
        "test_esmfold2_public_forward_honors_output_controls_and_sampler_overrides",
    ):
        assert callable(getattr(module, test_name))
