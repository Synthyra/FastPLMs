"""The released ESMFold2 model at toy widths: folding entry points, language-model hooks, and test-time training logic.

The toy model has no trunk or atom blocks, so a whole fold takes a fraction of a second on CPU. A miniature chemical
component dictionary stands in for the hash-pinned one, and a tiny ESM++ backbone stands in for ESMC. Collaborators that
are too large to run (the training loop, the checkpoint loader) are replaced by recorders so each test states what the
model method itself decides.

Shapes: `b` batch, `l` tokens, `t` language-model tokens.
"""

import pytest
import torch

from types import SimpleNamespace
from tests.unit.synthetic_ccd import install_mini_ccd
from tests.unit.tiny_families import (
    TINY_CONFIDENCE_HEAD,
    TINY_MSA_ENCODER,
    tiny_esmc_config,
    tiny_esmfold2_backbone,
    tiny_esmfold2_config,
)
from torch import nn

from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from fastplms.models.esmfold2 import esmfold2_types
from fastplms.models.esmfold2.esmfold2_constants_esm3 import (
    SEQUENCE_BOS_TOKEN,
    SEQUENCE_EOS_TOKEN,
    SEQUENCE_MASK_TOKEN,
    SEQUENCE_PAD_TOKEN,
    SEQUENCE_STANDARD_AA_MAX_TOKEN,
    SEQUENCE_STANDARD_AA_MIN_TOKEN,
)
from fastplms.models.esmfold2.esmfold2_molecular_complex import MolecularComplexResult
from fastplms.models.esmfold2.esmfold2_msa import MSA
from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from fastplms.models.esmfold2.protein_utils import prepare_protein_features


pytestmark = [
    pytest.mark.filterwarnings("ignore:CUDA is not available or torch_xla is imported:UserWarning"),
    pytest.mark.filterwarnings("ignore:No MSA provided for:UserWarning"),
]

FAST = {"num_loops": 0, "num_sampling_steps": 2}  # keeps a toy fold to a couple of denoising steps


@pytest.fixture
def model(monkeypatch) -> ESMFold2Model:
    install_mini_ccd(monkeypatch)
    torch.manual_seed(0)
    return ESMFold2Model(tiny_esmfold2_config(confidence_head=TINY_CONFIDENCE_HEAD)).eval()


def stub_result(plddt: float | None) -> SimpleNamespace:
    return SimpleNamespace(plddt=None if plddt is None else torch.full((3,), plddt))


def test_a_new_model_reports_that_no_backbone_is_loaded(model):
    status = model.esmc_precision_status

    assert status.resolved == "unloaded" and status.requested == "auto" and status.device == "cpu"
    assert model._esmc is None and model._ttt_lm_head is None


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


def test_sequences_are_tokenized_with_boundary_tokens_padding_and_x_for_unknown_letters(model):
    tokens = model._ttt_tokenize(["AC", "ACD"])
    unknown = model._ttt_tokenize("A?")
    given = torch.tensor([[0, 5, 2]])

    assert tokens.shape == (2, 5) and tokens[0, 0] == SEQUENCE_BOS_TOKEN and tokens[0, 3] == SEQUENCE_EOS_TOKEN
    assert tokens[0, 4] == SEQUENCE_PAD_TOKEN and tokens[1, 4] == SEQUENCE_EOS_TOKEN
    assert unknown[0, 2] == model._ttt_tokenize("X")[0, 1] and model._ttt_tokenize(input_ids=given) is given
    with pytest.raises(ValueError, match="either seq or input_ids"):
        model._ttt_tokenize()
    with pytest.raises(ValueError, match="at least one protein sequence"):
        model._ttt_tokenize([])


def test_test_time_training_masks_pads_and_replaces_with_the_standard_amino_acid_tokens(model):
    ids = torch.tensor([[SEQUENCE_BOS_TOKEN, SEQUENCE_STANDARD_AA_MIN_TOKEN, SEQUENCE_STANDARD_AA_MAX_TOKEN - 1, SEQUENCE_EOS_TOKEN, SEQUENCE_PAD_TOKEN]])

    replacements = model._ttt_replacement_tokens(ids)
    ordinary = model._ttt_non_special_mask(ids)

    assert model._ttt_mask_token() == SEQUENCE_MASK_TOKEN and model._ttt_padding_token() == SEQUENCE_PAD_TOKEN
    assert replacements.dtype == ids.dtype and replacements.tolist() == list(range(SEQUENCE_STANDARD_AA_MIN_TOKEN, SEQUENCE_STANDARD_AA_MAX_TOKEN))
    assert ordinary.tolist() == [[False, True, True, False, False]]


def test_a_backbone_is_required_for_logits_and_the_masked_head_is_loaded_from_the_backbone_source(model, tmp_path):
    ids = model._ttt_tokenize("AC")
    ESMplusplusForMaskedLM(tiny_esmc_config(vocab_size=64)).save_pretrained(tmp_path / "esmc")
    with pytest.raises(RuntimeError, match="requires load_esmc=True"):
        model._ttt_predict_logits(ids)
    with pytest.raises(RuntimeError, match="requires load_esmc=True"):
        model._ensure_ttt_lm_head()
    with pytest.raises(TypeError, match="expects input_ids tensors"):
        model._ttt_predict_logits({"input_ids": ids})
    model._esmc = tiny_esmfold2_backbone()
    model._esmc_source = str(tmp_path / "esmc")

    logits = model._ttt_predict_logits(ids)  # (b, t, vocab)
    head = model._ttt_lm_head
    model._ensure_ttt_lm_head()

    assert logits.shape == (1, 4, 64) and torch.isfinite(logits).all()
    assert model._ttt_lm_head is head and not any(parameter.requires_grad for parameter in head.parameters())


def test_the_backbone_hidden_states_are_computed_for_every_token_and_the_backbone_can_be_missing(model):
    features = prepare_protein_features("AC")
    arguments = (features["input_ids"], features["asym_id"], features["residue_index"], features["mol_type"], features["token_attention_mask"])
    with pytest.raises(RuntimeError, match="requires load_esmc=True"):
        model._compute_lm_hidden_states(*arguments)
    model._esmc = tiny_esmfold2_backbone()

    plain = model._compute_lm_hidden_states(*arguments)  # (b, l, n_states, d_lm)
    shown = model._compute_lm_hidden_states(*arguments, verbose=True)
    masked = model._compute_lm_hidden_states(*arguments, lm_mask_pct=1.0)

    assert plain.shape == (1, 2, 2, 8) and torch.equal(plain, shown) and not torch.equal(plain, masked)


def test_the_backbone_changes_the_pair_representation_the_trunk_starts_from(model):
    without = model.infer_protein("AC", **FAST).last_hidden_state
    model._esmc = tiny_esmfold2_backbone()

    with_backbone = model.infer_protein("AC", **FAST, verbose=True).last_hidden_state

    assert with_backbone.shape == without.shape and not torch.allclose(with_backbone, without)


def test_a_protein_is_folded_to_per_token_outputs_and_the_output_mapping_form_is_required(model):
    output = model.infer_protein("AC", **FAST)

    assert output.sample_atom_coords.shape == (1, 32, 3) and output.distogram_logits.shape == (1, 2, 2, 8)
    assert output.plddt.shape == (1, 2) and output.atom_pad_mask.shape == (1, 32)
    with pytest.raises(ValueError, match="return_dict=False is invalid"):
        model.infer_protein("AC", return_dict=False)


def test_a_checkpoint_without_msa_conditioning_gets_no_msa_features_from_a_protein_sequence(model):
    assert model.config.msa_conditioning is False

    output = model.infer_protein("AC", **FAST)

    assert torch.isfinite(output.sample_atom_coords).all()


def test_the_model_exposes_the_input_schema_and_prepares_features_through_its_builder(model):
    request = esmfold2_types.StructurePredictionInput(sequences=[esmfold2_types.LigandInput(id="L", ccd=["EOH"])])

    features, chains = model.prepare_structure_input(request, seed=1)

    assert model.input_types is esmfold2_types and features["ref_pos"].shape == (1, 32, 3) and [chain.chain_id for chain in chains] == ["L"]
    assert model.input_builder is model.input_builder


def test_a_typed_request_is_folded_to_a_result_with_confidence_and_a_complex(model):
    request = esmfold2_types.StructurePredictionInput(sequences=[esmfold2_types.ProteinInput(id="A", sequence="AC")])

    with pytest.warns(UserWarning, match="No MSA provided"):
        folded = model.fold(request, num_loops=0, num_sampling_steps=2, seed=3, complex_id="typed")

    assert isinstance(folded, MolecularComplexResult) and folded.complex.id == "typed" and folded.complex.sequence == ["ALA", "CYS"]
    assert folded.plddt.shape == (2,) and folded.ptm is not None


def test_a_protein_fold_checks_that_an_alignment_matches_the_sequence_and_reads_it_from_a_file(model, tmp_path):
    alignment = tmp_path / "query.a3m"
    alignment.write_text(">query\nAC\n>hit\nA-\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at most one of msa or msa_path"):
        model.fold_protein("AC", msa=MSA.from_sequences(["AC"]), msa_path=alignment)
    with pytest.raises(ValueError, match="MSA query does not match sequence"):
        model.fold_protein("AG", msa=MSA.from_sequences(["AC"]))
    with pytest.raises(ValueError, match="trained without MSA conditioning"):
        model.fold_protein("AC", msa_path=alignment, **FAST)
    conditioned = ESMFold2Model(tiny_esmfold2_config(msa_encoder=TINY_MSA_ENCODER, msa_conditioning=True)).eval()

    from_file = conditioned.fold_protein("AC", msa_path=alignment, msa_max_sequences=2, **FAST)
    from_object = conditioned.fold_protein("AC", msa=MSA.from_sequences(["AC", "A-"]), **FAST)

    assert from_file.complex.sequence == ["ALA", "CYS"] and from_object.complex.sequence == ["ALA", "CYS"]


def test_a_folded_protein_is_exported_as_mmcif_or_pdb_text_and_files(model, tmp_path):
    folded = model.fold_protein("AC", seed=2, **FAST)
    with pytest.raises(TypeError, match="one MolecularComplexResult at a time"):
        model.result_to_cif([folded])
    with pytest.raises(TypeError, match="one MolecularComplexResult at a time"):
        model.result_to_pdb([folded])

    cif = model.result_to_cif(folded)
    pdb = model.result_to_pdb(folded)
    model.save_as_cif(folded, tmp_path / "fold.cif")
    model.save_as_pdb(folded, tmp_path / "fold.pdb")
    direct_cif = model.infer_protein_as_cif("AC", seed=2, **FAST)
    direct_pdb = model.infer_protein_as_pdb("AC", seed=2, **FAST)

    assert cif.startswith("data_pred") and pdb.startswith("ATOM")
    assert (tmp_path / "fold.cif").read_text(encoding="utf-8") == cif and (tmp_path / "fold.pdb").read_text(encoding="utf-8") == pdb
    assert direct_cif == cif and direct_pdb == pdb


def test_kernel_backends_that_are_not_installed_are_refused_before_any_module_changes(model):
    model.set_kernel_backend(None)

    assert model._kernel_backend is None
    with pytest.raises(RuntimeError, match="does not bundle"):
        model.set_kernel_backend("fused")
    with pytest.raises(ValueError, match="backend must be one of"):
        model.set_kernel_backend("nonsense")
    assert model._kernel_backend is None


def test_chunk_sizes_reach_the_trunks_the_confidence_head_and_the_msa_encoder():
    model = ESMFold2Model(tiny_esmfold2_config(confidence_head=TINY_CONFIDENCE_HEAD, msa_encoder=TINY_MSA_ENCODER, msa_conditioning=True))

    model.set_chunk_size(2)

    assert [block.pair_transition._chunk_size for block in model.confidence_head.folding_trunk.blocks] == [2]
    assert [block.outer_product_mean._chunk_size for block in model.msa_encoder.blocks] == [2, 2]


def test_compiling_wraps_the_heavy_blocks_without_running_them():
    model = ESMFold2Model(tiny_esmfold2_config(msa_encoder=TINY_MSA_ENCODER, msa_conditioning=True))
    block = model.msa_encoder.blocks[0]
    original = block.forward

    model.apply_torch_compile(mode="dynamic_seqlen")

    assert block.forward is not original and callable(block.forward)


def test_the_mean_confidence_of_a_result_needs_a_confidence_tensor_and_the_best_of_several_results_is_picked(model):
    low, high = stub_result(0.3), stub_result(0.8)

    assert ESMFold2Model._ttt_mean_plddt(high) == pytest.approx(0.8)
    assert model._ttt_select_result([low, high, stub_result(0.5)]) is high and model._ttt_select_result(low) is low
    with pytest.raises(RuntimeError, match="has no pLDDT tensor"):
        ESMFold2Model._ttt_mean_plddt(stub_result(None))
    with pytest.raises(RuntimeError, match="empty result list"):
        model._ttt_select_result([])


def test_a_training_step_evaluation_folds_in_eval_mode_and_restores_the_training_state(model, monkeypatch):
    seen = []

    def fold(sequence, **keywords):
        seen.append((sequence, keywords, model.training))
        return [stub_result(0.4), stub_result(0.6)]

    monkeypatch.setattr(model, "_fold_protein_no_ttt", fold)
    model.train()

    record, plddt = model._ttt_eval_step(3, 0.25, seq="AC", fold_kwargs={"chain_id": "B"})

    assert seen == [("AC", {"chain_id": "B"}, False)] and model.training is True
    assert plddt == pytest.approx(0.6) and (record["step"], record["loss"], record["plddt"]) == (3, 0.25, plddt)
    with pytest.raises(TypeError, match="protein-only and sequence-string only"):
        model._ttt_eval_step(0, 0.0, seq=["AC"], fold_kwargs={})


class TrainingRecorder:
    """Stands in for the training loop: records its call and returns canned per-step metrics."""

    def __init__(self, model: ESMFold2Model, step_results: list[tuple[int, float]]) -> None:
        self.calls: list[dict] = []
        self.resets = 0
        self.step_results = step_results
        model.ttt = self
        model.ttt_reset = self.reset
        model._ttt_initialized = True

    def __call__(self, **keywords):
        self.calls.append(keywords)
        return {
            "losses": [1.0, 0.5],
            "step_metrics": [{"step": step, "plddt": plddt, "result": stub_result(plddt)} for step, plddt in self.step_results],
        }

    def reset(self) -> None:
        self.resets += 1


def test_training_at_test_time_keeps_the_best_result_and_always_resets_the_model(model, monkeypatch):
    baseline = stub_result(0.5)
    monkeypatch.setattr(model, "_fold_protein_no_ttt", lambda sequence, **keywords: baseline)
    model._esmc = nn.Identity()
    recorder = TrainingRecorder(model, [(1, 0.4), (2, 0.9), (3, 0.7)])

    best = model.fold_protein_ttt("AC", seed=4, ttt_config={"steps": 3})

    assert best is not baseline and best.plddt[0] == pytest.approx(0.9)
    assert best.ttt_metrics == {"losses": [1.0, 0.5], "step_plddts": [0.5, 0.4, 0.9, 0.7], "baseline_plddt": 0.5, "best_plddt": pytest.approx(0.9), "best_step": 2}
    assert recorder.resets == 1 and recorder.calls[0]["seq"] == "AC" and recorder.calls[0]["fold_kwargs"]["seed"] == 4
    assert recorder.calls[0]["ttt_config"].eval_each_step is True


def test_training_that_never_beats_the_baseline_returns_the_baseline_and_a_missing_backbone_is_refused(model, monkeypatch):
    baseline = stub_result(0.9)
    monkeypatch.setattr(model, "_fold_protein_no_ttt", lambda sequence, **keywords: baseline)
    with pytest.raises(RuntimeError, match="requires load_esmc=True"):
        model.fold_protein_ttt("AC")
    model._esmc = nn.Identity()
    recorder = TrainingRecorder(model, [(1, 0.2)])

    best = model.fold_protein_ttt("AC")

    assert best is baseline and best.ttt_metrics["best_step"] == 0 and best.ttt_metrics["step_plddts"] == pytest.approx([0.9, 0.2])
    assert recorder.resets == 1


def test_fold_protein_runs_training_only_when_asked(model, monkeypatch):
    calls = []
    monkeypatch.setattr(model, "fold_protein_ttt", lambda **keywords: calls.append(("ttt", keywords["sequence"])) or "tuned")
    monkeypatch.setattr(model, "_fold_protein_no_ttt", lambda **keywords: calls.append(("plain", keywords["sequence"])) or "plain")

    assert model.fold_protein("AC", ttt=True) == "tuned" and model.fold_protein("AG") == "plain"
    assert calls == [("ttt", "AC"), ("plain", "AG")]


def test_the_backbone_is_the_only_thing_the_model_asks_the_mixin_to_train(model):
    model._esmc = nn.Identity()

    assert model._ttt_get_trainable_modules() == [model._esmc]


def test_more_recycling_loops_change_the_pair_representation(model):
    one_pass = model.infer_protein("AC", num_loops=0, num_sampling_steps=1).last_hidden_state
    recycled = model.infer_protein("AC", num_loops=2, num_sampling_steps=1).last_hidden_state

    assert one_pass.shape == recycled.shape == (1, 2, 2, 8)
    assert not torch.equal(one_pass, recycled)
