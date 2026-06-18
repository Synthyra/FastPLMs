"""ESMFold2 AutoModel and parity tests."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModel

from fastplms.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusModel,
)
from fastplms.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.esmfold2.modeling_esmfold2 import (
    _load_fastplms_esmplusplus_for_esmfold2,
)
from fastplms.esmfold2.modeling_esmfold2_common import (
    compute_lm_hidden_states,
    maybe_apply_msa_column_masking,
    maybe_subsample_msa,
)
from fastplms.esmfold2.modeling_esmc import (
    _PyTorchLayerNormLinear,
    _PyTorchLayerNormMLP,
)
from testing.conftest import STRUCTURE_MODEL_REGISTRY

ESMFOLD2_MODEL_KEYS = ("esmfold2", "esmfold2_fast")
TEST_SEQUENCE = "MSTNPKPQRKTKRNT"
OUTPUT_TOLERANCES = {
    "distogram_logits": 0.0,
    "plddt": 1e-6,
    "pae": 0.0,
    "ptm": 0.0,
    "iptm": 0.0,
}


def test_esmfold2_config_uses_fastplms_esmplusplus_defaults() -> None:
    config = ESMFold2Config()

    assert config.esmc_id == "Synthyra/ESMplusplus_6B"
    assert config.esmc_attn_backend == "flex"
    assert config.lm_mask_pct == 0.0


def test_esmfold2_config_normalizes_legacy_esmc_ids() -> None:
    config = ESMFold2Config(esmc_id="biohub/ESMC-6B")

    assert config.esmc_id == "Synthyra/ESMplusplus_6B"


def test_esmc_pytorch_fallback_accepts_fp32_inputs_with_bf16_weights() -> None:
    ln_linear = _PyTorchLayerNormLinear(d_in=8, d_out=12).to(dtype=torch.bfloat16)
    ln_mlp = _PyTorchLayerNormMLP(hidden_size=8, ffn_hidden_size=16).to(dtype=torch.bfloat16)
    x = torch.randn(2, 4, 8, dtype=torch.float32)

    with torch.no_grad():
        linear_out = ln_linear(x)
        mlp_out = ln_mlp(x)

    assert linear_out.dtype == torch.bfloat16
    assert mlp_out.dtype == torch.bfloat16


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


def test_esmplusplus_flex_sequence_id_masks_run() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flex attention sequence_id regression.")
    device = torch.device("cuda")
    config = ESMplusplusConfig(
        vocab_size=16,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        attn_backend="flex",
    )
    model = ESMplusplusModel(config).to(device=device).eval()
    input_ids = torch.tensor([[0, 3, 4, 2]], device=device, dtype=torch.long)
    sequence_id = torch.tensor([[0, 0, 1, 1]], device=device, dtype=torch.long)

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, sequence_id=sequence_id)
    except (AssertionError, RuntimeError) as error:
        pytest.skip(f"Flex attention unavailable in this environment: {error}")

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
        esmfold2_output.hidden_states[0],
        model.embed(input_ids),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        esmfold2_output.hidden_states[-1],
        public_output.hidden_states[-1],
        rtol=0.0,
        atol=0.0,
    )


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


def test_esmfold2_load_esmc_fp8_requires_transformer_engine(monkeypatch) -> None:
    import fastplms.esmfold2.modeling_esmfold2 as esmfold2_module
    from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = SimpleNamespace()
    monkeypatch.setattr(esmfold2_module, "TE_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="requires transformer_engine"):
        ESMFold2Model.load_esmc(model, "unused", precision="fp8")


def test_esmfold2_load_esmc_fp8_converts_fastplms_adapter(monkeypatch) -> None:
    import fastplms.esmfold2.modeling_esmfold2 as esmfold2_module
    from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model

    class TinyAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.config = SimpleNamespace(hidden_size=16, num_hidden_layers=1)

    adapter = TinyAdapter()
    calls = {}

    def fake_loader(*, esmc_model_path, attn_backend, device, dtype):
        calls["loader"] = (esmc_model_path, attn_backend, device, dtype)
        return adapter.to(device=device, dtype=dtype)

    def fake_converter(module):
        calls["converter"] = module

    model = SimpleNamespace(
        config=SimpleNamespace(
            esmc_attn_backend="sdpa",
            lm_d_model=16,
            lm_num_layers=1,
        ),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(esmfold2_module, "TE_AVAILABLE", True)
    monkeypatch.setattr(
        esmfold2_module,
        "_load_fastplms_esmplusplus_for_esmfold2",
        fake_loader,
    )
    monkeypatch.setattr(
        esmfold2_module,
        "_convert_te_modules_to_fp8_inplace",
        fake_converter,
    )

    ESMFold2Model.load_esmc(model, "dummy-esm", precision="fp8")

    assert calls["loader"] == (
        "dummy-esm",
        "sdpa",
        torch.device("cpu"),
        torch.bfloat16,
    )
    assert calls["converter"] is adapter
    assert model._esmc is adapter
    assert model._esmc_fp8 is True
    assert model._ttt_lm_head is None
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_esmfold2_ttt_rejects_fp8_adapter() -> None:
    from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = SimpleNamespace(_esmc=torch.nn.Linear(1, 1), _esmc_fp8=True)

    with pytest.raises(RuntimeError, match="TTT is not supported with fp8"):
        ESMFold2Model._ttt_get_trainable_modules(model)


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


def _enable_deterministic_forward() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _load_official_model(model_key: str) -> torch.nn.Module:
    config = STRUCTURE_MODEL_REGISTRY[model_key]
    module = pytest.importorskip("transformers.models.esmfold2.modeling_esmfold2")
    official_cls = module.ESMFold2Model
    return (
        official_cls.from_pretrained(
            config["official_path"],
            load_esmc=False,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )


def _load_fast_model(model_key: str) -> torch.nn.Module:
    config = STRUCTURE_MODEL_REGISTRY[model_key]
    return (
        AutoModel.from_pretrained(
            config["fast_path"],
            trust_remote_code=True,
            load_esmc=False,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )


def _run_short_fold(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    common_module_name = (
        model.__class__.__module__.rsplit(".", 1)[0]
        + ".modeling_esmfold2_common"
    )
    common_module = importlib.import_module(common_module_name)
    with common_module._seed_context(0), torch.no_grad():
        return model.infer_protein(
            TEST_SEQUENCE,
            num_loops=1,
            num_sampling_steps=2,
            num_diffusion_samples=1,
        )


def test_esmfold2_fold_protein_accepts_msa_path(tmp_path, monkeypatch) -> None:
    from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model

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
    from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model

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
    from fastplms.esmfold2.modeling_esmfold2 import ESMFold2Model

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


def _aligned_rmsd(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    mask = atom_mask[0].bool() if atom_mask.ndim == 2 else atom_mask.bool()
    actual_coords = actual[0, mask].float()
    expected_coords = expected[0, mask].float()

    actual_centered = actual_coords - actual_coords.mean(dim=0, keepdim=True)
    expected_centered = expected_coords - expected_coords.mean(dim=0, keepdim=True)
    cov = actual_centered.T @ expected_centered
    u, _, vh = torch.linalg.svd(cov)
    det = torch.det(u @ vh)
    correction = torch.eye(3, device=actual.device, dtype=torch.float32)
    correction[2, 2] = torch.sign(det)
    rotation = u @ correction @ vh
    aligned = actual_centered @ rotation
    return torch.sqrt(torch.mean(torch.sum((aligned - expected_centered) ** 2, dim=-1)))


def _assert_forward_parity(model_key: str) -> None:
    _enable_deterministic_forward()
    official_model = _load_official_model(model_key)
    fast_model = _load_fast_model(model_key)

    official_output = _run_short_fold(official_model)
    fast_output = _run_short_fold(fast_model)

    for key, atol in OUTPUT_TOLERANCES.items():
        torch.testing.assert_close(
            fast_output[key],
            official_output[key],
            rtol=0.0,
            atol=atol,
            msg=f"ESMFold2 output mismatch: {key}",
        )

    rmsd = _aligned_rmsd(
        fast_output["sample_atom_coords"],
        official_output["sample_atom_coords"],
        official_output["atom_pad_mask"],
    )
    assert rmsd.item() < 1e-2, f"Aligned coordinate RMSD too high: {rmsd.item()}"

    del official_model, fast_model, official_output, fast_output
    torch.cuda.empty_cache()


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
@pytest.mark.parametrize("model_key", ESMFOLD2_MODEL_KEYS)
def test_esmfold2_weight_parity(model_key: str) -> None:
    official_model = _load_official_model(model_key)
    fast_model = _load_fast_model(model_key)

    official_state = official_model.state_dict()
    fast_state = fast_model.state_dict()
    assert official_state.keys() == fast_state.keys()

    for name, official_tensor in official_state.items():
        fast_tensor = fast_state[name]
        torch.testing.assert_close(
            fast_tensor,
            official_tensor,
            rtol=0.0,
            atol=0.0,
            msg=f"ESMFold2 parameter mismatch: {name}",
        )

    del official_model, fast_model
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_key", ESMFOLD2_MODEL_KEYS)
def test_esmfold2_forward_parity(model_key: str) -> None:
    env = os.environ.copy()
    with tempfile.TemporaryDirectory() as module_cache:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["HF_MODULES_CACHE"] = module_cache
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--esmfold2-forward-parity",
                model_key,
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    if result.returncode != 0 and "Skipped:" in result.stderr:
        pytest.skip(result.stderr.split("Skipped:", 1)[1].strip())
    assert result.returncode == 0, result.stdout + result.stderr


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


if __name__ == "__main__":
    assert len(sys.argv) == 3
    assert sys.argv[1] == "--esmfold2-forward-parity"
    _assert_forward_parity(sys.argv[2])
