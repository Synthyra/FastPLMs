"""Parts of the released ESMFold2 model at toy widths: pair transition, confidence head, MSA encoder, and precision helpers.

The confidence head is built from the toy configuration with its head enabled, so its pLDDT, PAE, PDE, resolved-atom, and
pTM outputs can be checked against their definitions. The precision helpers run against stand-in Transformer Engine modules.

Shapes: `b` batch, `l` tokens, `a` atoms, `s` diffusion samples, `bs` = `b * s`, `m` MSA rows.
"""

import contextlib
import importlib
import pytest
import torch

from types import SimpleNamespace
from tests.unit.tiny_families import TINY_CONFIDENCE_HEAD, tiny_esmfold2_config

from fastplms.models.esmfold2 import modeling_esmfold2 as model_module
from fastplms.models.esmfold2.modeling_esmfold2 import (
    ConfidenceHead,
    ESMCPrecisionStatus,
    MSAEncoder,
    MSAEncoderBlock,
    PairTransition,
)


TOKENS = 4
ATOMS = 8


def seeded(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture(autouse=True)
def fixed_weights():
    torch.manual_seed(2468)


def test_the_pair_transition_has_no_residual_and_is_chunked_along_rows_without_changing_values():
    transition = PairTransition(8, expansion_ratio=2).eval()
    pair = torch.randn(2, 7, 7, 8, generator=seeded(1))  # (b, l, l, d)

    whole = transition(pair)
    transition.set_chunk_size(3)
    chunked = transition(pair)

    assert transition._chunk_size == 3 and whole.shape == pair.shape
    torch.testing.assert_close(whole, transition.ffn(transition.norm(pair)))
    torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=1e-5)


def confidence_head() -> tuple[ConfidenceHead, object]:
    config = tiny_esmfold2_config(confidence_head=TINY_CONFIDENCE_HEAD)
    return ConfidenceHead(config).eval(), config


def confidence_inputs(config, samples: int = 1) -> dict[str, torch.Tensor]:
    d_inputs, d_pair = config.inputs.d_inputs, config.d_pair
    return {  # input name -> (b, ...) tensor
        "s_inputs": torch.randn(1, TOKENS, d_inputs, generator=seeded(2)),
        "z": torch.randn(1, TOKENS, TOKENS, d_pair, generator=seeded(3)),
        "x_pred": torch.randn(samples, ATOMS, 3, generator=seeded(4)) * 4.0,
        "distogram_atom_idx": torch.tensor([[0, 2, 4, 6]]),
        "token_attention_mask": torch.ones(1, TOKENS, dtype=torch.bool),
        "atom_to_token": torch.arange(ATOMS).div(2, rounding_mode="floor")[None],
        "atom_attention_mask": torch.ones(1, ATOMS, dtype=torch.bool),
        "asym_id": torch.tensor([[0, 0, 1, 1]]),
        "mol_type": torch.tensor([[0, 0, 0, 3]]),
    }


def test_the_confidence_head_scores_every_token_atom_and_pair_with_values_in_range():
    head, config = confidence_head()
    inputs = confidence_inputs(config)

    with torch.no_grad():
        scores = head(**inputs)

    assert scores["plddt_logits"].shape == (1, ATOMS, 4) and scores["plddt_per_atom"].shape == (1, ATOMS)
    assert scores["plddt"].shape == (1, TOKENS) and scores["plddt_ca"].shape == (1, TOKENS)
    assert scores["complex_plddt"].shape == scores["complex_iplddt"].shape == scores["ptm"].shape == scores["iptm"].shape == (1,)
    assert scores["pae_logits"].shape == (1, TOKENS, TOKENS, 4) and scores["pae"].shape == scores["pde"].shape == (1, TOKENS, TOKENS)
    assert scores["pde_logits"].shape == (1, TOKENS, TOKENS, 4) and scores["resolved_logits"].shape == (1, ATOMS, 2)
    assert scores["pair_chains_iptm"].shape == (1, 2, 2)
    for name in ("plddt", "plddt_per_atom", "plddt_ca", "ptm", "iptm", "complex_plddt"):
        assert scores[name].min() >= 0.0 and scores[name].max() <= 1.0
    assert scores["pae"].max() <= 32.0 and torch.isfinite(scores["pair_chains_iptm"]).all()


def test_a_head_with_zero_atom_weights_gives_every_atom_the_middle_confidence():
    head, config = confidence_head()
    inputs = confidence_inputs(config)

    with torch.no_grad():
        scores = head(**inputs)

    torch.testing.assert_close(scores["plddt"], torch.full((1, TOKENS), 0.5))
    torch.testing.assert_close(scores["plddt_per_atom"], torch.full((1, ATOMS), 0.5))
    torch.testing.assert_close(scores["complex_plddt"], torch.tensor([0.5]), atol=1e-4, rtol=1e-4)
    assert torch.count_nonzero(scores["resolved_logits"]) == 0


def test_samples_expand_the_batch_and_conditioning_terms_move_the_pair_scores():
    head, config = confidence_head()
    inputs = confidence_inputs(config, samples=2)
    shifted = {**inputs, "relative_position_encoding": torch.randn(1, TOKENS, TOKENS, config.d_pair, generator=seeded(5)),
               "token_bonds_encoding": torch.randn(1, TOKENS, TOKENS, config.d_pair, generator=seeded(6))}

    with torch.no_grad():
        flat = head(**inputs, num_diffusion_samples=2)
        stacked = head(**{**inputs, "x_pred": inputs["x_pred"][None]}, num_diffusion_samples=2)
        conditioned = head(**shifted, num_diffusion_samples=2)

    assert flat["pae"].shape == (2, TOKENS, TOKENS) and flat["plddt"].shape == (2, TOKENS)
    torch.testing.assert_close(stacked["pae"], flat["pae"])
    assert not torch.allclose(conditioned["pae"], flat["pae"])


def test_the_confidence_head_helpers_repeat_the_batch_and_flatten_the_sample_axis():
    batch = torch.arange(6).reshape(2, 3)  # (b, 3)

    assert ConfidenceHead._repeat_batch(batch, 1) is batch
    assert ConfidenceHead._repeat_batch(batch, 3).tolist() == [[0, 1, 2]] * 3 + [[3, 4, 5]] * 3
    four_dimensional = torch.arange(24).reshape(2, 3, 2, 2)  # (b, samples, n, c)
    assert ConfidenceHead._flatten_sample_axis(four_dimensional).shape == (6, 2, 2)
    three_dimensional = torch.arange(12).reshape(3, 2, 2)
    assert ConfidenceHead._flatten_sample_axis(three_dimensional) is three_dimensional


def test_the_confidence_head_forwards_chunk_sizes_and_backend_choices_to_its_trunk():
    head, _ = confidence_head()

    head.set_chunk_size(2)
    head.set_kernel_backend(None)

    assert [block.pair_transition._chunk_size for block in head.folding_trunk.blocks] == [2]
    assert [block._kernel_backend for block in head.folding_trunk.blocks] == [None]


def test_the_msa_encoder_returns_an_updated_pair_and_its_last_block_has_no_msa_update():
    encoder = MSAEncoder(d_msa=8, d_pair=8, d_inputs=10, d_hidden=2, n_layers=2, n_heads_msa=2, msa_head_width=4).eval()
    msa_one_hot = torch.nn.functional.one_hot(torch.randint(0, 33, (1, TOKENS, 3), generator=seeded(7)), 33).float()  # (b, l, m, 33)
    pair = torch.randn(1, TOKENS, TOKENS, 8, generator=seeded(8))
    mask = torch.ones(1, TOKENS, 3)  # (b, l, m)
    mask[:, -1] = 0.0

    with torch.no_grad():
        updated = encoder(pair, torch.randn(1, TOKENS, 10, generator=seeded(9)), msa_one_hot, torch.zeros(1, TOKENS, 3), torch.zeros(1, TOKENS, 3), mask)
    encoder.set_chunk_size(2)

    assert updated.shape == pair.shape and not torch.allclose(updated, pair)
    assert hasattr(encoder.blocks[0], "msa_transition") and not hasattr(encoder.blocks[1], "msa_transition")
    assert [block.outer_product_mean._chunk_size for block in encoder.blocks] == [2, 2]
    assert encoder.blocks[0].msa_transition._chunk_size == 2 and encoder.blocks[1].pair_transition._chunk_size == 2


def test_an_msa_block_updates_the_alignment_representation_unless_it_is_the_final_block():
    middle = MSAEncoderBlock(d_msa=8, d_pair=8, d_hidden=2, n_heads_msa=2, msa_head_width=4).eval()
    final = MSAEncoderBlock(d_msa=8, d_pair=8, d_hidden=2, n_heads_msa=2, msa_head_width=4, is_final_block=True).eval()
    rows = torch.randn(1, TOKENS, 3, 8, generator=seeded(10))  # (b, l, m, d_msa)
    pair = torch.randn(1, TOKENS, TOKENS, 8, generator=seeded(11))
    msa_mask = torch.ones(1, TOKENS, 3)
    pair_mask = torch.ones(1, TOKENS, TOKENS, dtype=torch.bool)

    with torch.no_grad():
        updated_rows, updated_pair = middle(rows, pair, msa_mask, pair_mask)
        same_rows, final_pair = final(rows, pair, msa_mask, pair_mask)

    assert updated_rows.shape == rows.shape and not torch.allclose(updated_rows, rows) and updated_pair.shape == pair.shape
    assert same_rows is rows and not torch.allclose(final_pair, pair)
    middle.set_chunk_size(2)
    assert middle.msa_transition._chunk_size == 2 and middle.outer_product_mean._chunk_size == 2


def test_a_precision_status_serializes_to_a_plain_dictionary():
    status = ESMCPrecisionStatus(requested="auto", resolved="bf16", reason="default", device="cpu", transformer_engine_version=None)

    assert status.as_dict() == {
        "requested": "auto", "resolved": "bf16", "reason": "default", "device": "cpu", "transformer_engine_version": None,
    }


def test_transformer_engine_is_loaded_lazily_and_its_absence_or_age_is_reported(monkeypatch):
    def missing(name):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="Transformer Engine could not be imported: ImportError"):
        model_module._load_transformer_engine()

    engine = SimpleNamespace(name="engine")
    old_recipe = SimpleNamespace()
    monkeypatch.setattr(importlib, "import_module", lambda name: engine if name.endswith("pytorch") else old_recipe)
    with pytest.raises(RuntimeError, match="does not expose Float8CurrentScaling"):
        model_module._load_transformer_engine()

    new_recipe = SimpleNamespace(Float8CurrentScaling=object)
    monkeypatch.setattr(importlib, "import_module", lambda name: engine if name.endswith("pytorch") else new_recipe)
    assert model_module._load_transformer_engine() == (engine, new_recipe)


def test_the_language_model_runs_in_bfloat16_autocast_on_cuda_and_fp8_adds_a_transformer_engine_context(monkeypatch):
    events = []

    class Recipe:
        Format = SimpleNamespace(HYBRID="hybrid")

        class Float8CurrentScaling:
            def __init__(self, **keywords):
                events.append(("recipe", keywords))

    @contextlib.contextmanager
    def engine_autocast(enabled, recipe):
        events.append(("engine", enabled, type(recipe).__name__))
        yield

    engine = SimpleNamespace(autocast=engine_autocast)
    monkeypatch.setattr(model_module, "_load_transformer_engine", lambda: (engine, Recipe))
    cuda = torch.device("cuda")

    with model_module._lm_precision_context("fp32", cuda):
        events.append("fp32 body")
    with model_module._lm_precision_context("bf16", torch.device("cpu")):
        events.append("cpu body")
    # torch.autocast warns that it disables itself only where CUDA is absent.
    def autocast_warning():
        return contextlib.nullcontext() if torch.cuda.is_available() else pytest.warns(UserWarning)

    with autocast_warning(), model_module._lm_precision_context("bf16", cuda):
        events.append("bf16 body")
    with autocast_warning(), model_module._lm_precision_context("fp8", cuda):
        events.append("fp8 body")

    assert events == [
        "fp32 body", "cpu body", "bf16 body",
        ("recipe", {"use_power_2_scales": False, "fp8_format": "hybrid"}), ("engine", True, "Float8CurrentScaling"), "fp8 body",
    ]


def test_the_adapter_hands_attention_changes_to_the_backbone_and_stacks_hidden_states():
    class Backbone(torch.nn.Module):
        config = SimpleNamespace(hidden_size=4)

        def __init__(self):
            super().__init__()
            self.changes = []

        def set_attn_implementation(self, implementation):
            self.changes.append(implementation)

        def forward(self, **keywords):
            self.keywords = keywords
            return SimpleNamespace(hidden_states=[torch.zeros(1, 2, 4), torch.ones(1, 2, 4)])

    backbone = Backbone()
    adapter = model_module._ESMFold2ESMplusplusAdapter(backbone)

    adapter.set_attn_implementation("sdpa")
    output = adapter(torch.zeros(1, 2, dtype=torch.long), output_hidden_states=True)

    assert backbone.changes == ["sdpa"] and adapter.config.hidden_size == 4
    assert output.hidden_states.shape == (2, 1, 2, 4) and backbone.keywords["esmfold2_hidden_states"] is True
