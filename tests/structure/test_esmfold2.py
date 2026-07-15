"""ESMFold2 package behavior tests.

Official parity is exercised only through the isolated bundles in
``test_esmfold2_folding_compliance.py``.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusModel,
)
from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2 import (
    _load_fastplms_esmplusplus_for_esmfold2,
    _manifest_esmc_checkpoint_contract,
)
from fastplms.models.esmfold2.modeling_esmfold2_common import (
    compute_lm_hidden_states,
    maybe_apply_msa_column_masking,
    maybe_subsample_msa,
)
from fastplms.registry import ModelSpec, get_model_registry
from transformers.modeling_utils import PreTrainedModel

REGISTRY = get_model_registry()
ESMFOLD2_MODEL_KEYS = tuple(spec.id for spec in REGISTRY.by_family("esmfold2"))
TEST_SEQUENCE = "MSTNPKPQRKTKRNT"


def test_esmfold2_config_uses_fastplms_esmplusplus_defaults() -> None:
    config = ESMFold2Config()
    backbone = REGISTRY[REGISTRY.families["esmfold2"].backbone_model]

    assert config.esmc_id == backbone.fast.repo_id
    assert config.esmc_attn_backend is None
    assert config.lm_mask_pct == 0.0

    legacy = ESMFold2Config(esmc_attn_backend="flex")
    assert legacy.esmc_attn_backend == "flex_attention"


def test_esmfold2_config_normalizes_legacy_esmc_ids() -> None:
    backbone = REGISTRY[REGISTRY.families["esmfold2"].backbone_model]
    config = ESMFold2Config(esmc_id=backbone.official.repo_id)

    assert config.esmc_id == backbone.fast.repo_id


def test_esmfold2_esmc_source_uses_manifest_revision_and_file_identities() -> None:
    revision, files = _manifest_esmc_checkpoint_contract("biohub/ESMC-6B")
    expected = REGISTRY["esmc_6b"].fast

    assert revision == expected.revision
    assert files == {item.path: item.encoded for item in expected.files}


def test_esmfold2_rejects_other_remote_esmc_checkpoints() -> None:
    with pytest.raises(ValueError, match="is not the manifest-declared ESMFold2 backbone"):
        _manifest_esmc_checkpoint_contract("biohub/ESMC-300M")


def test_esmplusplus_sequence_id_masks_cross_chain_attention() -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )
    model = ESMplusplusModel(config).eval()
    input_ids = torch.tensor([[0, 3, 4, 2]], dtype=torch.long)
    sequence_id = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            sequence_id=sequence_id,
            output_attentions=True,
        )

    assert output.attentions is not None
    attention = output.attentions[0]
    torch.testing.assert_close(
        attention[:, :, :2, 2:],
        torch.zeros_like(attention[:, :, :2, 2:]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        attention[:, :, 2:, :2],
        torch.zeros_like(attention[:, :, 2:, :2]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu
def test_esmplusplus_flex_sequence_id_masks_run() -> None:
    device = torch.device("cuda")
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="flex_attention",
    )
    model = ESMplusplusModel(config).to(device=device).eval()
    input_ids = torch.tensor([[0, 3, 4, 2]], device=device, dtype=torch.long)
    sequence_id = torch.tensor([[0, 0, 1, 1]], device=device, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, sequence_id=sequence_id)

    assert output.last_hidden_state.shape == (1, 4, 64)


def test_esmplusplus_esmfold2_hidden_state_layout() -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=2,
        attn_backend="sdpa",
    )
    model = ESMplusplusModel(config).eval()
    input_ids = torch.tensor([[0, 3, 4, 2]], dtype=torch.long)

    with torch.no_grad():
        public_output = model(input_ids=input_ids, output_hidden_states=True)
        esmfold2_output = model(
            input_ids=input_ids,
            output_hidden_states=True,
            esmfold2_hidden_states=True,
        )

    assert public_output.hidden_states is not None
    assert esmfold2_output.hidden_states is not None
    assert len(public_output.hidden_states) == config.num_hidden_layers + 1
    assert len(esmfold2_output.hidden_states) == config.num_hidden_layers + 1
    torch.testing.assert_close(
        public_output.hidden_states[0],
        model.embed(input_ids),
        rtol=0.0,
        atol=0.0,
    )
    for public_state, esmfold2_state in zip(
        public_output.hidden_states,
        esmfold2_output.hidden_states,
        strict=True,
    ):
        torch.testing.assert_close(public_state, esmfold2_state, rtol=0.0, atol=0.0)


def test_esmplusplus_config_has_official_special_token_ids() -> None:
    config = ESMplusplusConfig()

    assert config.pad_token_id == 1
    assert config.mask_token_id == 32
    assert config.classifier_dropout == 0.1
    assert config.initializer_range == 0.02
    assert config.tie_word_embeddings is False


def test_esmplusplus_boolean_padding_keeps_eager_outputs_finite() -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="eager",
    )
    model = ESMplusplusModel(config).eval()
    input_ids = torch.tensor([[0, 3, 1, 1]], dtype=torch.long)
    sequence_id = torch.tensor([[True, True, False, False]])

    with torch.no_grad():
        output = model(input_ids=input_ids, sequence_id=sequence_id)

    assert torch.isfinite(output.last_hidden_state).all()


def test_esmplusplus_masked_lm_can_skip_logits() -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )
    model = ESMplusplusForMaskedLM(config).eval()
    input_ids = torch.tensor([[0, 3, 4, 2]], dtype=torch.long)

    with torch.no_grad():
        no_logits = model(input_ids=input_ids, compute_logits=False)
        with_logits = model(input_ids=input_ids, compute_logits=True)

    assert no_logits.logits is None
    assert with_logits.logits is not None
    assert with_logits.logits.shape == (1, 4, config.vocab_size)


def test_esmplusplus_auto_classes_share_exact_checkpoint_keys() -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )

    base_keys = set(ESMplusplusModel(config).state_dict())
    masked_lm_keys = set(ESMplusplusForMaskedLM(config).state_dict())

    assert base_keys == masked_lm_keys
    assert "sequence_head.0.weight" in base_keys


def test_esmfold2_loads_shared_esmplusplus_adapter(tmp_path) -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )
    ESMplusplusModel(config).save_pretrained(tmp_path)

    adapter = _load_fastplms_esmplusplus_for_esmfold2(
        esmc_model_path=str(tmp_path),
        attn_backend="sdpa",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[0, 3, 4, 2]], dtype=torch.long)
    sequence_id = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

    with torch.no_grad():
        output = adapter(
            input_ids=input_ids,
            sequence_id=sequence_id,
            output_hidden_states=True,
        )

    assert adapter.config.attn_backend == "sdpa"
    assert output.hidden_states.shape == (config.num_hidden_layers + 1, 1, 4, 16)


def test_esmfold2_forwards_manifest_revision_to_esmc_loaders(monkeypatch) -> None:
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    def load_config(cls, source: str, **kwargs):
        del cls
        calls.append(("config", source, kwargs))
        return config

    def load_model(cls, source: str, **kwargs):
        del cls
        calls.append(("model", source, kwargs))
        return ESMplusplusModel(config)

    monkeypatch.setattr(ESMplusplusConfig, "from_pretrained", classmethod(load_config))
    monkeypatch.setattr(ESMplusplusModel, "from_pretrained", classmethod(load_model))

    adapter = _load_fastplms_esmplusplus_for_esmfold2(
        esmc_model_path="Synthyra/ESMplusplus_6B",
        attn_backend="sdpa",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_revision = REGISTRY["esmc_6b"].fast.revision
    assert adapter.config is config
    assert calls[0] == (
        "config",
        "Synthyra/ESMplusplus_6B",
        {"revision": expected_revision},
    )
    assert calls[1][0:2] == ("model", "Synthyra/ESMplusplus_6B")
    assert calls[1][2]["revision"] == expected_revision
    assert calls[1][2]["config"] is config
    assert calls[1][2]["torch_dtype"] == torch.float32


def test_esmfold2_load_esmc_fp8_is_strict_when_unavailable(monkeypatch) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as esmfold2_module
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(
        esmfold2_module,
        "_te_fp8_capability",
        lambda _device: (False, "Transformer Engine reports FP8 unavailable."),
    )

    with pytest.raises(RuntimeError, match="Transformer Engine reports FP8 unavailable"):
        ESMFold2Model.load_esmc(model, "unused", precision="fp8")


def test_esmfold2_load_esmc_auto_selects_bf16_without_probing_fp8(monkeypatch) -> None:
    import fastplms.models.esmfold2.modeling_esmfold2 as esmfold2_module
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    class TinyAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.config = SimpleNamespace(hidden_size=16, num_hidden_layers=1)

    adapter = TinyAdapter()
    calls = {}

    def fake_loader(*, esmc_model_path, attn_backend, device, dtype):
        calls["loader"] = (esmc_model_path, attn_backend, device, dtype)
        return adapter.to(dtype=dtype)

    model = SimpleNamespace(
        config=SimpleNamespace(
            esmc_attn_backend="sdpa",
            lm_d_model=16,
            lm_num_layers=1,
        ),
        device=torch.device("cuda"),
    )
    monkeypatch.setattr(
        esmfold2_module,
        "_te_fp8_capability",
        lambda _device: pytest.fail("auto must not probe FP8 capability"),
    )
    monkeypatch.setattr(
        esmfold2_module,
        "_load_fastplms_esmplusplus_for_esmfold2",
        fake_loader,
    )
    monkeypatch.setattr(
        esmfold2_module,
        "_convert_esmc_attention_outputs_to_te",
        lambda _adapter: pytest.fail("auto must not convert ESMC to FP8"),
    )

    ESMFold2Model.load_esmc(model, "dummy-esm", precision="auto")

    expected_device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cuda")
    )
    assert calls["loader"] == (
        "dummy-esm",
        "sdpa",
        expected_device,
        torch.bfloat16,
    )
    assert model._esmc is adapter
    assert model._esmc_fp8 is False
    assert model._esmc_fp8_module_paths == ()
    assert model._esmc_precision_status.requested == "auto"
    assert model._esmc_precision_status.resolved == "bf16"
    assert "defaults to BF16" in model._esmc_precision_status.reason
    assert model._ttt_lm_head is None
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


@pytest.mark.parametrize("experimental", [False, True], ids=("released", "experimental"))
def test_esmfold2_from_pretrained_preserves_loading_info(monkeypatch, experimental) -> None:
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
        ESMFold2ExperimentalModel,
    )

    model_class = ESMFold2ExperimentalModel if experimental else ESMFold2Model
    config = SimpleNamespace(esmc_id="dummy-esmc", esmc_precision="auto")
    calls: list[tuple[str, str]] = []
    model = SimpleNamespace(
        config=config,
        load_esmc=lambda source, *, precision: calls.append((source, precision)),
    )
    loading_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }

    def fake_from_pretrained(cls, source, *args, **kwargs):
        del args
        assert cls is model_class
        assert source == "dummy-fold"
        assert kwargs["config"] is config
        assert kwargs["output_loading_info"] is True
        return model, loading_info

    monkeypatch.setattr(
        PreTrainedModel,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    loaded = model_class.from_pretrained(
        "dummy-fold",
        config=config,
        output_loading_info=True,
        esmc_precision="bf16",
    )

    assert loaded == (model, loading_info)
    assert calls == [("dummy-esmc", "bf16")]


def test_esmfold2_ttt_reloads_canonical_bf16_adapter() -> None:
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    class DummyModel:
        _ensure_ttt_bf16 = ESMFold2Model._ensure_ttt_bf16
        _ttt_get_trainable_modules = ESMFold2Model._ttt_get_trainable_modules

        def __init__(self) -> None:
            self._esmc = torch.nn.Linear(1, 1)
            self._esmc_fp8 = True
            self._esmc_precision_policy = "auto"
            self._esmc_precision_status = SimpleNamespace(
                device="cpu",
                transformer_engine_version=None,
            )
            self.config = SimpleNamespace(esmc_precision="auto")
            self.device = torch.device("cpu")
            self.reload_precision = None

        def reload_esmc(self, precision="auto", device=None) -> None:
            self.reload_precision = (precision, device)
            self._esmc = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
            self._esmc_fp8 = False

    model = DummyModel()
    trainable = model._ttt_get_trainable_modules()

    assert model.reload_precision == ("bf16", torch.device("cpu"))
    assert model._esmc_fp8 is False
    assert model.config.esmc_precision == "auto"
    assert model._esmc_precision_status.requested == "auto"
    assert model._esmc_precision_status.resolved == "bf16"
    assert trainable == [model._esmc]
    assert next(model._esmc.parameters()).dtype == torch.bfloat16


def test_compute_lm_hidden_states_pads_and_masks_non_special_tokens() -> None:
    class CapturingEsmc(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_ids = None
            self.sequence_id = None

        def forward(self, input_ids, sequence_id, output_hidden_states):
            assert output_hidden_states is True
            self.input_ids = input_ids.detach().clone()
            self.sequence_id = sequence_id.detach().clone()
            num_layers = 2
            hidden_size = 3
            hidden_states = torch.arange(
                num_layers * input_ids.numel() * hidden_size,
                dtype=torch.float32,
            ).reshape(num_layers, *input_ids.shape, hidden_size)
            return SimpleNamespace(hidden_states=hidden_states)

    esmc = CapturingEsmc()
    input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    asym_id = torch.tensor([[0, 0, 0]], dtype=torch.long)
    residue_index = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mol_type = torch.zeros_like(input_ids)
    token_mask = torch.ones_like(input_ids, dtype=torch.bool)

    result = compute_lm_hidden_states(
        esmc,
        input_ids,
        asym_id,
        residue_index,
        mol_type,
        token_mask,
        pad_to_multiple=8,
        lm_mask_pct=1.0,
        mask_token_id=32,
    )

    assert esmc.input_ids is not None
    assert esmc.sequence_id is not None
    assert esmc.input_ids.tolist() == [[0, 32, 32, 32, 2, 1, 1, 1]]
    assert esmc.sequence_id.tolist() == [[0, 0, 0, 0, 0, -1, -1, -1]]
    assert result.shape == (1, 3, 2, 3)


def test_msa_subsample_keeps_query_row() -> None:
    msa = torch.arange(20, dtype=torch.long).reshape(1, 5, 4)
    msa_attention_mask = torch.ones(1, 5, 4, dtype=torch.bool)
    has_deletion = torch.zeros(1, 5, 4, dtype=torch.bool)
    deletion_value = torch.zeros(1, 5, 4)

    torch.manual_seed(0)
    subsampled, mask, deletion, deletion_vals = maybe_subsample_msa(
        msa,
        msa_attention_mask,
        has_deletion,
        deletion_value,
        max_depth=3,
        enabled=True,
    )

    assert subsampled.shape == (1, 3, 4)
    assert torch.equal(subsampled[:, 0], msa[:, 0])
    assert mask is not None
    assert deletion is not None
    assert deletion_vals is not None
    assert torch.equal(mask[:, 0], msa_attention_mask[:, 0])
    assert torch.equal(deletion[:, 0], has_deletion[:, 0])
    assert torch.equal(deletion_vals[:, 0], deletion_value[:, 0])


def test_msa_column_masking_keeps_query_row() -> None:
    msa_attention_mask = torch.ones(2, 3, 4, dtype=torch.bool)

    masked = maybe_apply_msa_column_masking(msa_attention_mask, rate=1.0)

    assert masked is not None
    assert masked[:, 0, :].all()
    assert not masked[:, 1:, :].any()


def _esmfold2_spec(model_key: str) -> ModelSpec:
    spec = REGISTRY[model_key]
    assert spec.family.id == "esmfold2"
    return spec


def _load_fast_model(model_key: str) -> torch.nn.Module:
    spec = _esmfold2_spec(model_key)
    is_experimental = "experimental" in spec.id
    module_name = (
        "fastplms.models.esmfold2.modeling_esmfold2_experimental"
        if is_experimental
        else "fastplms.models.esmfold2.modeling_esmfold2"
    )
    class_name = "ESMFold2ExperimentalModel" if is_experimental else "ESMFold2Model"
    model_class = getattr(importlib.import_module(module_name), class_name)
    return (
        model_class.from_pretrained(
            spec.fast.repo_id,
            revision=spec.fast.revision,
            load_esmc=False,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )


def test_esmfold2_fold_protein_accepts_msa_path(tmp_path, monkeypatch) -> None:
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    captured = {}

    def fake_fold(self, input_value, **kwargs):
        del self
        captured["input"] = input_value
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(ESMFold2Model, "fold", fake_fold)
    msa_path = tmp_path / "query.a3m"
    msa_path.write_text(">query\nMSTN\n>hit\nMSTN\n", encoding="utf-8")
    model = object.__new__(ESMFold2Model)

    result = ESMFold2Model.fold_protein(
        model,
        "MSTN",
        msa_path=msa_path,
        msa_max_sequences=1,
        seed=7,
    )

    assert result == "ok"
    protein_input = captured["input"].sequences[0]
    assert protein_input.sequence == "MSTN"
    assert protein_input.msa is not None
    assert protein_input.msa.depth == 1
    assert captured["kwargs"]["seed"] == 7


def test_esmfold2_fold_protein_rejects_msa_query_mismatch(tmp_path, monkeypatch) -> None:
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    def fake_fold(self, input_value, **kwargs):
        del self, input_value, kwargs
        return "ok"

    monkeypatch.setattr(ESMFold2Model, "fold", fake_fold)
    msa_path = tmp_path / "query.a3m"
    msa_path.write_text(">query\nAAAA\n", encoding="utf-8")
    model = object.__new__(ESMFold2Model)

    with pytest.raises(AssertionError, match="MSA query does not match sequence"):
        ESMFold2Model.fold_protein(model, "MSTN", msa_path=msa_path)


def test_esmfold2_fold_protein_without_msa_preserves_single_sequence(monkeypatch) -> None:
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    captured = {}

    def fake_fold(self, input_value, **kwargs):
        del self, kwargs
        captured["input"] = input_value
        return "ok"

    monkeypatch.setattr(ESMFold2Model, "fold", fake_fold)
    model = object.__new__(ESMFold2Model)

    result = ESMFold2Model.fold_protein(model, "MSTN")

    assert result == "ok"
    protein_input = captured["input"].sequences[0]
    assert protein_input.sequence == "MSTN"
    assert protein_input.msa is None


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_key", ESMFOLD2_MODEL_KEYS)
def test_esmfold2_automodel_loads(model_key: str) -> None:
    model = _load_fast_model(model_key)

    assert callable(model.infer_protein)
    assert callable(model.fold)
    assert callable(model.fold_protein)
    assert callable(model.prepare_structure_input)
    assert callable(model.result_to_cif)
    assert callable(model.result_to_pdb)
    assert model.input_types.ProteinInput.__name__ == "ProteinInput"

    del model
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
def test_esmfold2_input_builder_complex_and_exports() -> None:
    model = _load_fast_model("esmfold2_fast")
    types = model.input_types
    msa = types.MSA.from_sequences([TEST_SEQUENCE, "MSTNPKPQRKTKRNS"])
    structure_input = types.StructurePredictionInput(
        sequences=[
            types.ProteinInput(id="A", sequence=TEST_SEQUENCE, msa=msa),
            types.DNAInput(id="B", sequence="ATGC"),
            types.LigandInput(id="L", smiles="O"),
        ]
    )

    features, chain_infos = model.prepare_structure_input(structure_input, seed=0)
    assert features["token_index"].shape[0] == 1
    assert features["token_index"].shape[1] > len(TEST_SEQUENCE)
    assert features["ref_pos"].shape[-1] == 3
    assert len(chain_infos) == 3

    result = model.fold_protein(
        TEST_SEQUENCE,
        num_loops=1,
        num_sampling_steps=1,
        num_diffusion_samples=1,
        seed=0,
    )
    cif = model.result_to_cif(result)
    pdb = model.result_to_pdb(result)
    assert "data_" in cif
    assert "ATOM" in pdb
    assert result.plddt.ndim == 1
    assert result.ptm is not None

    del model, features, result
    torch.cuda.empty_cache()
