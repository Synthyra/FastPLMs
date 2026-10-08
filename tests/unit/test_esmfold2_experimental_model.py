"""The experimental ESMFold2 model at toy widths: confidence head, alignment encoder, recycling, and fold entry points.

Every module is built with a few units per axis, so a whole fold takes a fraction of a second on CPU. A miniature chemical
component dictionary stands in for the hash-pinned one and a tiny ESM++ backbone stands in for ESMC. Collaborators that
load checkpoints or compile kernels are replaced by recorders, so each test states what the experimental model itself
decides. The released model's counterparts are tested in `test_esmfold2_model_methods.py`.

Shapes: `b` batch, `bs` batch times diffusion samples, `l` tokens, `a` atoms, `m` alignment rows.
"""

import pytest
import torch

from tests.unit.synthetic_ccd import install_mini_ccd
from tests.unit.tiny_families import (
    TINY_CONFIDENCE_HEAD,
    TINY_MSA_ENCODER,
    tiny_esmfold2_backbone,
    tiny_esmfold2_config,
)
from torch import nn

from fastplms.models.esmfold2 import esmfold2_types
from fastplms.models.esmfold2 import modeling_esmfold2_experimental as experimental
from fastplms.models.esmfold2.esmfold2_molecular_complex import MolecularComplexResult
from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Output
from fastplms.models.esmfold2.modeling_esmfold2_common import MSA_CONDITIONING_INPUT_NAMES
from fastplms.models.esmfold2.modeling_esmfold2_experimental import (
    ConfidenceHead,
    ESMFold2ExperimentalModel,
    MSAEncoder,
    MSAEncoderBlock,
    _TransitionFFN,
)
from fastplms.models.esmfold2.protein_utils import prepare_protein_features


pytestmark = [
    pytest.mark.filterwarnings("ignore:CUDA is not available or torch_xla is imported:UserWarning"),
    pytest.mark.filterwarnings("ignore:No MSA provided for:UserWarning"),
]

FAST = {"num_loops": 0, "num_sampling_steps": 2}  # keeps a toy fold to a couple of denoising steps
BATCH = 2
LENGTH = 3
DEPTH = 4


def experimental_config(**overrides):
    return tiny_esmfold2_config("experimental", **overrides)


@pytest.fixture(autouse=True)
def toy_world(monkeypatch):
    install_mini_ccd(monkeypatch)
    torch.manual_seed(0)


@pytest.fixture
def model() -> ESMFold2ExperimentalModel:
    return ESMFold2ExperimentalModel(experimental_config(confidence_head=TINY_CONFIDENCE_HEAD)).eval()


@pytest.fixture
def head() -> ConfidenceHead:
    torch.manual_seed(1)
    config = experimental_config(confidence_head=TINY_CONFIDENCE_HEAD)
    confidence_head = ConfidenceHead(config).eval()
    nn.init.normal_(confidence_head.plddt_weight)  # the released initialization is all zeros, which scores every atom alike
    return confidence_head


def head_inputs(asym_id=(0, 0, 1), mol_type=(0, 0, 0), spacing=(0.0, 5.0, 11.0)) -> dict[str, torch.Tensor]:
    """Confidence-head inputs for three tokens of two atoms each.

    The representative atom of a token is its first atom and token `t` sits the t-th `spacing` angstroms along x, so the
    distance between two tokens is the gap between their spacings.
    """

    generator = torch.Generator().manual_seed(11)
    d_inputs = experimental_config().inputs.d_inputs
    coordinates = torch.zeros(1, 6, 3)  # (b, a, 3)
    coordinates[0, 0::2, 0] = torch.tensor(spacing)
    coordinates[0, 1::2, 0] = torch.tensor(spacing) + 1.0
    return {  # input name -> (b, ...) tensor
        "s_inputs": torch.randn(1, LENGTH, d_inputs, generator=generator),
        "z": torch.randn(1, LENGTH, LENGTH, 8, generator=generator),
        "x_pred": coordinates,
        "distogram_atom_idx": torch.tensor([[0, 2, 4]]),
        "token_attention_mask": torch.ones(1, LENGTH, dtype=torch.bool),
        "atom_to_token": torch.tensor([[0, 0, 1, 1, 2, 2]]),
        "atom_attention_mask": torch.ones(1, 6, dtype=torch.bool),
        "asym_id": torch.tensor([asym_id]),
        "mol_type": torch.tensor([mol_type]),
    }


def expected_tm(tokens: int, bins: int) -> float:
    """The TM-score of an error distribution that is uniform over `bins` bins spanning 0 to 32 angstroms."""

    d0 = 1.24 * (max(tokens, 19) - 15) ** (1 / 3) - 1.8
    centers = [(index + 0.5) * 32.0 / bins for index in range(bins)]
    return sum(1 / (1 + (center / d0) ** 2) for center in centers) / bins


def test_the_confidence_head_scores_each_atom_token_and_token_pair_of_every_sample(head):
    inputs = head_inputs()
    inputs["x_pred"] = torch.stack([inputs["x_pred"][0], inputs["x_pred"][0] * 4.0])[None]  # (b, samples, a, 3), second one stretched

    scores = head(**inputs, num_diffusion_samples=2)

    assert {name: tuple(value.shape) for name, value in scores.items()} == {
        "plddt_logits": (2, 6, 4),
        "plddt": (2, 3),
        "plddt_per_atom": (2, 6),
        "plddt_ca": (2, 3),
        "complex_plddt": (2,),
        "complex_iplddt": (2,),
        "pae_logits": (2, 3, 3, 4),
        "pae": (2, 3, 3),
        "ptm": (2,),
        "iptm": (2,),
        "pair_chains_iptm": (2, 2, 2),
    }
    per_atom = scores["plddt_per_atom"]
    torch.testing.assert_close(scores["plddt"], per_atom.reshape(2, 3, 2).mean(-1))
    torch.testing.assert_close(scores["plddt_ca"], per_atom[:, [0, 2, 4]])
    torch.testing.assert_close(scores["complex_plddt"], per_atom.mean(-1), atol=1e-5, rtol=1e-5)
    assert scores["complex_iplddt"][0] > 0 and scores["complex_iplddt"][1] == 0  # the stretched sample has no chain contact within 8 A


def test_interface_confidence_averages_atoms_of_tokens_near_another_chain_and_counts_ligand_tokens_twice(head):
    protein = head(**head_inputs())
    with_ligand = head(**head_inputs(mol_type=(0, 0, 3)))
    per_atom = protein["plddt_per_atom"][0]  # (a,)

    for scores, weights in ((protein, [0, 0, 1, 1, 1, 1]), (with_ligand, [0, 0, 1, 1, 2, 2])):
        weight = torch.tensor(weights, dtype=torch.float32)  # (a,): only tokens 1 and 2 touch the other chain within 8 A
        assert scores["complex_iplddt"][0].item() == pytest.approx(((per_atom * weight).sum() / weight.sum()).item(), rel=1e-4)


def test_atoms_marked_as_padding_do_not_count_toward_token_or_complex_confidence(head):
    inputs = head_inputs()
    inputs["atom_attention_mask"][0, 5] = False

    scores = head(**inputs)

    per_atom = scores["plddt_per_atom"][0]
    assert scores["plddt"][0, 2].item() == pytest.approx(per_atom[4].item(), rel=1e-5)
    assert scores["complex_plddt"][0].item() == pytest.approx(per_atom[:5].mean().item(), rel=1e-4)


def test_chain_pair_scores_leave_absent_chain_ids_at_zero_and_one_chain_has_no_interface_score(head):
    gapped = head(**head_inputs(asym_id=(0, 0, 2)))["pair_chains_iptm"][0]  # (c, c) with no token of chain 1
    single = head(**head_inputs(asym_id=(0, 0, 0)))

    assert gapped.shape == (3, 3) and gapped[1].eq(0).all() and gapped[:, 1].eq(0).all() and gapped[0, 2] > 0
    assert single["pair_chains_iptm"].shape == (1, 1, 1) and single["iptm"].item() == 0.0


def test_uniform_error_logits_give_the_midpoint_error_and_the_tm_score_of_the_bin_average(head):
    nn.init.zeros_(head.pae_head.weight)

    scores = head(**head_inputs())

    tm = expected_tm(tokens=3, bins=4)
    torch.testing.assert_close(scores["pae"], torch.full_like(scores["pae"], 16.0))
    assert scores["ptm"].item() == pytest.approx(tm, rel=1e-3) and scores["iptm"].item() == pytest.approx(tm, rel=1e-3)
    assert scores["pair_chains_iptm"][0, 0, 1].item() == pytest.approx(tm, rel=1e-3)
    assert scores["pair_chains_iptm"][0, 1, 1].item() == pytest.approx(tm, rel=1e-3)


def test_sample_axes_are_repeated_per_batch_entry_and_flattened_into_the_batch():
    values = torch.arange(6.0).reshape(2, 3)  # (b, 3)

    repeated = ConfidenceHead._repeat_batch(values, 2)  # (bs, 3)

    assert ConfidenceHead._repeat_batch(values, 1) is values
    assert repeated.tolist() == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]]
    assert ConfidenceHead._flatten_sample_axis(torch.zeros(2, 3, 5, 4)).shape == (6, 5, 4)
    assert ConfidenceHead._flatten_sample_axis(values) is values


def test_the_head_forwards_chunk_sizes_and_kernel_backends_to_its_trunk(head):
    head.set_chunk_size(2)
    head.set_kernel_backend(None)

    assert [block.pair_transition._chunk_size for block in head.folding_trunk.blocks] == [2]
    with pytest.raises(ValueError, match="backend must be one of"):
        head.set_kernel_backend("nonsense")


def test_the_transition_normalizes_its_input_before_the_gated_feed_forward():
    transition = _TransitionFFN(8).eval()
    states = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(2))  # (b, l, d)

    transitioned = transition(states)

    torch.testing.assert_close(transitioned, transition.ffn(transition.norm(states)))
    torch.testing.assert_close(transition(states + 5.0), transitioned, atol=1e-5, rtol=1e-5)  # normalization removes a constant shift


def alignment_block_inputs():
    generator = torch.Generator().manual_seed(3)
    return (
        torch.randn(BATCH, LENGTH, DEPTH, 8, generator=generator),  # (b, l, m, d_msa)
        torch.randn(BATCH, LENGTH, LENGTH, 8, generator=generator),  # (b, l, l, d_pair)
        torch.ones(BATCH, LENGTH, DEPTH),  # (b, l, m)
        torch.ones(BATCH, LENGTH, LENGTH),  # (b, l, l)
    )


def test_an_alignment_block_updates_only_the_batch_entries_whose_track_mask_is_on():
    block = MSAEncoderBlock(d_msa=8, d_pair=8, d_hidden=2, n_heads_msa=2, msa_head_width=4).eval()
    msa, pair, msa_mask, pair_mask = alignment_block_inputs()

    with torch.no_grad():
        ungated = block(msa, pair, msa_mask, pair_mask)
        gated = block(msa, pair, msa_mask, pair_mask, torch.tensor([False, True]))
        open_gate = block(msa, pair, msa_mask, pair_mask, torch.tensor([True, True]))

    assert ungated[0].shape == msa.shape and ungated[1].shape == pair.shape and not torch.equal(ungated[1], pair)
    assert torch.equal(gated[0][0], msa[0]) and torch.equal(gated[1][0], pair[0])
    torch.testing.assert_close(gated[0][1], ungated[0][1])
    torch.testing.assert_close(gated[1][1], ungated[1][1])
    torch.testing.assert_close(open_gate[1], ungated[1])


def test_an_alignment_block_gives_the_same_result_chunked_or_not():
    block = MSAEncoderBlock(d_msa=8, d_pair=8, d_hidden=2, n_heads_msa=2, msa_head_width=4).eval()
    msa, pair, msa_mask, pair_mask = alignment_block_inputs()

    with torch.no_grad():
        whole = block(msa, pair, msa_mask, pair_mask)
        block.set_chunk_size(2)
        chunked = block(msa, pair, msa_mask, pair_mask)

    assert block.outer_product_mean._chunk_size == 2 and block.tri_mul_out._engine._chunk_size == 2
    assert block.tri_mul_in._engine._chunk_size == 2
    torch.testing.assert_close(chunked[0], whole[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(chunked[1], whole[1], atol=1e-5, rtol=1e-5)


def test_the_alignment_encoder_adds_pair_features_only_for_entries_with_a_non_query_sequence():
    encoder = MSAEncoder(d_msa=8, d_pair=8, d_inputs=6, d_hidden=2, n_layers=2, n_heads_msa=2, msa_head_width=4).eval()
    generator = torch.Generator().manual_seed(4)
    pair = torch.randn(BATCH, LENGTH, LENGTH, 8, generator=generator)  # (b, l, l, d_pair)
    inputs = torch.randn(BATCH, LENGTH, 6, generator=generator)  # (b, l, d_inputs)
    rows = torch.nn.functional.one_hot(torch.randint(0, 33, (BATCH, LENGTH, DEPTH), generator=generator), 33).float()  # (b, l, m, 33)
    deletions = torch.zeros(BATCH, LENGTH, DEPTH)  # (b, l, m)
    mask = torch.ones(BATCH, LENGTH, DEPTH)
    mask[0, :, 1:] = 0.0  # the first entry keeps only its query row

    with torch.no_grad():
        mixed = encoder(pair, inputs, rows, deletions, deletions, mask)  # (b, l, l, d_pair)
        single_row = encoder(pair, inputs, rows[:, :, :1], deletions[:, :, :1], deletions[:, :, :1], mask[:, :, :1])
    encoder.set_chunk_size(2)

    assert mixed.shape == pair.shape and mixed[0].eq(0).all() and mixed[1].abs().max() > 0
    assert single_row.eq(0).all()
    assert [block.outer_product_mean._chunk_size for block in encoder.blocks] == [2, 2]


def test_a_new_model_reports_that_no_backbone_is_loaded(model):
    status = model.esmc_precision_status

    assert status.resolved == "unloaded" and status.requested == "auto" and status.device == "cpu"
    assert model._esmc is None and model._ttt_lm_head is None
    assert model.device == torch.device("cpu")


def test_loading_the_backbone_hands_the_request_to_the_installer(model, monkeypatch):
    requests = []
    monkeypatch.setattr(experimental, "_install_esmc_backbone", lambda target, path, **keywords: requests.append((target, path, keywords)))

    model.load_esmc("an/esmc")
    model.load_esmc("another/esmc", precision="fp32", device="cpu", local_files_only=True)

    assert requests[0] == (model, "an/esmc", {"precision": "auto", "device": None, "local_files_only": False})
    assert requests[1][2] == {"precision": "fp32", "device": "cpu", "local_files_only": True}


def test_reloading_the_backbone_drops_the_old_one_and_loads_the_same_source_again(model, monkeypatch):
    loads = []
    monkeypatch.setattr(model, "load_esmc", lambda *arguments, **keywords: loads.append((arguments, keywords)))
    model._esmc = nn.Linear(1, 1)
    model._ttt_lm_head = nn.Linear(1, 1)
    model._esmc_source = "an/esmc"
    model._esmc_local_files_only = True

    model.reload_esmc(precision="fp32", device="cpu")
    model.reload_esmc()
    model.reload_esmc(local_files_only=False)

    assert model._esmc is None and model._ttt_lm_head is None and model._esmc_fp8 is False
    assert loads[0] == (("an/esmc",), {"precision": "fp32", "device": "cpu", "local_files_only": True})
    assert loads[1][1]["precision"] == "auto" and loads[2][1]["local_files_only"] is False


def test_language_model_dropout_applies_at_inference_only_when_forced(model):
    states = torch.randn(1, 2, 2, 8, generator=torch.Generator().manual_seed(5))  # (b, l, n_states, d_lm)

    model.configure_lm_dropout(0.5)
    forced = [model.infer_protein("AC", lm_hidden_states=states, **FAST).last_hidden_state for _ in range(2)]
    model.configure_lm_dropout(0.5, force_lm_dropout_during_inference=False)
    quiet = [model.infer_protein("AC", lm_hidden_states=states, **FAST).last_hidden_state for _ in range(2)]

    assert model.config.lm_dropout == 0.5 and model.config.force_lm_dropout_during_inference is False
    assert not torch.equal(forced[0], forced[1]) and torch.equal(quiet[0], quiet[1])


def test_the_backbone_hidden_states_are_computed_for_every_token_and_the_backbone_can_be_missing(model):
    features = prepare_protein_features("AC")
    arguments = (features["input_ids"], features["asym_id"], features["residue_index"], features["mol_type"], features["token_attention_mask"])
    with pytest.raises(RuntimeError, match="require load_esmc=True"):
        model._compute_lm_hidden_states(*arguments)
    model._esmc = tiny_esmfold2_backbone()

    plain = model._compute_lm_hidden_states(*arguments)  # (b, l, n_states, d_lm)
    shown = model._compute_lm_hidden_states(*arguments, verbose=True)

    assert plain.shape == (1, 2, 2, 8) and torch.equal(plain, shown)


def test_an_fp8_backbone_pads_to_sixteen_and_is_reloaded_in_bf16_when_gradients_are_needed(model, monkeypatch):
    reloads = []
    paddings = []
    monkeypatch.setattr(experimental, "_reload_esmc_bf16_for_gradients", lambda target, *, reason: reloads.append((target, reason)))
    monkeypatch.setattr(
        experimental,
        "compute_lm_hidden_states",
        lambda backbone, *arguments, pad_to_multiple: paddings.append(pad_to_multiple) or "states",
    )
    model._esmc = nn.Identity()
    ids = torch.zeros(1, 2, dtype=torch.long)
    arguments = (ids, ids, ids, ids, ids.bool())

    assert model._compute_lm_hidden_states(*arguments) == "states"
    model._esmc_fp8 = True
    with torch.no_grad():
        model._compute_lm_hidden_states(*arguments)
    assert paddings == [None, 16] and reloads == []
    model._compute_lm_hidden_states(*arguments)

    assert len(reloads) == 1 and reloads[0][0] is model and "requires BF16" in reloads[0][1]


def test_the_backbone_changes_the_pair_representation_the_trunk_starts_from(model):
    without = model.infer_protein("AC", **FAST).last_hidden_state
    model._esmc = tiny_esmfold2_backbone()

    with_backbone = model.infer_protein("AC", **FAST).last_hidden_state

    assert with_backbone.shape == without.shape and not torch.allclose(with_backbone, without)


def test_runtime_backbones_are_left_out_of_the_saved_weights(model):
    model._esmc = tiny_esmfold2_backbone()
    model._ttt_lm_head = nn.Linear(2, 2)

    keys = list(model.state_dict())

    assert keys and not any(key.startswith(("_esmc.", "_ttt_lm_head.")) for key in keys)


def test_chunk_sizes_reach_every_trunk_the_confidence_head_and_the_alignment_encoder():
    config = experimental_config(
        confidence_head=TINY_CONFIDENCE_HEAD,
        msa_encoder=TINY_MSA_ENCODER,
        msa_conditioning=True,
        folding_trunk={"n_layers": 1, "n_heads": 2, "dropout": 0.0},
    )
    chunked = ESMFold2ExperimentalModel(config)

    chunked.set_chunk_size(2)

    assert [block.pair_transition._chunk_size for block in chunked.folding_trunk.blocks] == [2]
    assert [block.pair_transition._chunk_size for block in chunked.confidence_head.folding_trunk.blocks] == [2]
    assert [block.outer_product_mean._chunk_size for block in chunked.msa_encoder.blocks] == [2, 2]


def test_kernel_backends_that_are_not_installed_are_refused_before_any_module_changes(model):
    model.set_kernel_backend(None)

    assert model._kernel_backend is None
    with pytest.raises(RuntimeError, match="does not bundle"):
        model.set_kernel_backend("fused")
    with pytest.raises(ValueError, match="backend must be one of"):
        model.set_kernel_backend("nonsense")
    assert model._kernel_backend is None


def test_compiling_wraps_only_the_heavy_blocks_and_picks_dynamic_shapes_by_mode(monkeypatch):
    compiled_model = ESMFold2ExperimentalModel(
        experimental_config(confidence_head=TINY_CONFIDENCE_HEAD, msa_encoder=TINY_MSA_ENCODER, msa_conditioning=True)
    )
    compiled: list[tuple[str, bool]] = []

    def record(function, dynamic):
        compiled.append((type(function.__self__).__name__, dynamic))
        return function

    monkeypatch.setattr(torch, "compile", record)

    compiled_model.apply_torch_compile()
    per_pass = len(compiled)
    compiled_model.apply_torch_compile(mode="dynamic_seqlen")
    compiled_model.apply_torch_compile(mode="fixed_seqlen", dynamic=True)

    assert {name for name, _ in compiled} == {"PairUpdateBlock", "DiffusionTransformer", "DiffusionModule", "MSAEncoderBlock"}
    assert [{flag for _, flag in compiled[start : start + per_pass]} for start in (0, per_pass, 2 * per_pass)] == [{False}, {True}, {True}]


def test_a_protein_is_folded_to_per_token_outputs_with_confidence_and_the_input_features(model):
    output = model.infer_protein("AC", **FAST)

    assert isinstance(output, ESMFold2Output)
    assert output.sample_atom_coords.shape == (1, 32, 3) and output.distogram_logits.shape == (1, 2, 2, 8)
    assert output.plddt.shape == (1, 2) and output.pair_chains_iptm.shape == (1, 1, 1)
    assert output["res_type"].shape == (1, 2) and output["atom_to_token"].shape == (1, 32)
    with pytest.raises(ValueError, match="return_dict=False is invalid"):
        model.infer_protein("AC", return_dict=False)


def record_forward_keywords(candidate, monkeypatch) -> list[set[str]]:
    """Replace the forward pass of `candidate` with a recorder of the keyword names it receives."""

    seen: list[set[str]] = []

    def forward(**keywords):
        seen.append(set(keywords))
        return ESMFold2Output()

    monkeypatch.setattr(candidate, "forward", forward)
    return seen


def test_alignment_features_reach_the_forward_pass_only_for_a_model_conditioned_on_alignments(monkeypatch):
    plain = ESMFold2ExperimentalModel(experimental_config()).eval()
    conditioned = ESMFold2ExperimentalModel(experimental_config(msa_encoder=TINY_MSA_ENCODER, msa_conditioning=True)).eval()
    plain_seen = record_forward_keywords(plain, monkeypatch)
    conditioned_seen = record_forward_keywords(conditioned, monkeypatch)

    plain.infer_protein("AC")
    conditioned.infer_protein("AC")

    assert plain_seen[0].isdisjoint(MSA_CONDITIONING_INPUT_NAMES)
    assert set(MSA_CONDITIONING_INPUT_NAMES) <= conditioned_seen[0]


def test_early_exit_stops_recycling_once_the_pair_state_and_distogram_stop_changing(model):
    nn.init.zeros_(model.pair_loop_proj[1].weight)  # nothing the loop feeds back changes the pair state
    calls = []
    model.folding_trunk.register_forward_hook(lambda module, arguments, output: calls.append(output.shape))

    model.infer_protein("AC", num_loops=3, num_sampling_steps=1)
    full = len(calls)
    calls.clear()
    model.infer_protein("AC", num_loops=3, num_sampling_steps=1, early_exit=True)

    assert (full, len(calls)) == (4, 2)


def test_verbose_folding_reports_progress_for_each_stage(model, capsys):
    model.infer_protein("AC", num_loops=1, num_sampling_steps=1, verbose=True)

    progress = capsys.readouterr().err
    for stage in ("ESMFold2 features", "ESMFold2 recycling", "ESMFold2 confidence"):
        assert stage in progress


def test_the_model_exposes_the_input_schema_and_prepares_features_through_its_builder(model):
    request = esmfold2_types.StructurePredictionInput(sequences=[esmfold2_types.LigandInput(id="L", ccd=["EOH"])])

    features, chains = model.prepare_structure_input(request, seed=1)

    assert model.input_types is esmfold2_types and features["ref_pos"].shape == (1, 32, 3) and [chain.chain_id for chain in chains] == ["L"]
    assert model.input_builder is model.input_builder


def test_a_protein_is_folded_to_a_named_complex_and_exported_as_mmcif_or_pdb_text_and_files(model, tmp_path):
    folded = model.fold_protein("AC", chain_id="B", complex_id="named", seed=2, **FAST)
    typed = model.fold(
        esmfold2_types.StructurePredictionInput(sequences=[esmfold2_types.ProteinInput(id="A", sequence="AC")]), seed=2, **FAST
    )
    samples = model.fold_protein("AC", num_diffusion_samples=2, seed=2, **FAST)
    with pytest.raises(TypeError, match="one MolecularComplexResult at a time"):
        model.result_to_cif(samples)
    with pytest.raises(TypeError, match="one MolecularComplexResult at a time"):
        model.result_to_pdb(samples)

    cif = model.result_to_cif(folded)
    pdb = model.result_to_pdb(folded)
    model.save_as_cif(folded, tmp_path / "fold.cif")
    model.save_as_pdb(folded, tmp_path / "fold.pdb")
    direct_cif = model.infer_protein_as_cif("AC", chain_id="B", complex_id="named", seed=2, **FAST)
    direct_pdb = model.infer_protein_as_pdb("AC", chain_id="B", complex_id="named", seed=2, **FAST)

    assert isinstance(folded, MolecularComplexResult) and folded.complex.id == "named" and folded.complex.sequence == ["ALA", "CYS"]
    assert isinstance(typed, MolecularComplexResult) and folded.plddt.shape == (2,) and len(samples) == 2
    assert cif.startswith("data_named") and pdb.startswith("ATOM")
    assert (tmp_path / "fold.cif").read_text(encoding="utf-8") == cif and (tmp_path / "fold.pdb").read_text(encoding="utf-8") == pdb
    assert direct_cif == cif and direct_pdb == pdb
