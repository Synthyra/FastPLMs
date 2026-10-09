"""The ESMFold2 input builder batches typed features, runs a model, and decodes the coordinates it returns.

The model is a small stand-in that returns tensors of the right shapes, so these tests cover the builder's own work:
batching, device placement, the keyword arguments passed to the model, and the decoding of each diffusion sample.

Shapes: `l` tokens, `a` atoms (padded to a multiple of 32), `s` diffusion samples.
"""

import pytest
import torch

from types import SimpleNamespace
from tests.unit.synthetic_ccd import install_mini_ccd

from fastplms.models.esmfold2 import esmfold2_processor as processor
from fastplms.models.esmfold2.esmfold2_msa import MSA
from fastplms.models.esmfold2.esmfold2_processor import ESMFold2InputBuilder
from fastplms.models.esmfold2.esmfold2_types import LigandInput, ProteinInput, StructurePredictionInput
from fastplms.models.esmfold2.modeling_esmfold2_common import MSA_CONDITIONING_INPUT_NAMES


TOKENS = 5  # alanine and glycine of chain A, then three ethanol atoms
ATOMS = 32  # 12 atoms padded to a multiple of 32


@pytest.fixture(autouse=True)
def mini_ccd(monkeypatch):
    return install_mini_ccd(monkeypatch)


class StandInModel:
    """Records the keyword arguments of each call and returns tensors of the shapes the builder decodes."""

    def __init__(self, msa_conditioning: bool = True, samples: int = 1) -> None:
        self.config = SimpleNamespace(msa_conditioning=msa_conditioning)
        self.device = torch.device("cpu")
        self.samples = samples
        self.calls: list[dict] = []

    def __call__(self, **features):
        self.calls.append(features)
        atoms = features["atom_attention_mask"].shape[-1]
        tokens = features["token_attention_mask"].shape[-1]
        samples = self.samples
        return {
            "sample_atom_coords": torch.arange(samples * atoms * 3, dtype=torch.float32).reshape(samples, atoms, 3),  # (s, a, 3)
            "plddt": torch.linspace(0.5, 0.9, tokens).repeat(samples, 1),  # (s, l)
            "ptm": torch.linspace(0.6, 0.7, samples),  # (s,)
            "iptm": torch.linspace(0.3, 0.4, samples),  # (s,)
            "pae": torch.zeros(samples, tokens, tokens),  # (s, l, l)
            "distogram_logits": torch.zeros(samples, tokens, tokens, 4),  # (s, l, l, 4)
        }


def request() -> StructurePredictionInput:
    return StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="AG", msa=MSA.from_sequences(["AG"])), LigandInput(id="L", ccd=["EOH"])]
    )


def test_features_get_a_batch_axis_and_move_to_the_requested_device_while_other_values_pass_through():
    features = {"tokens": torch.arange(3), "name": "kept", "flag": True}

    plain = processor._batch_features(features, None)
    placed = processor._batch_features(features, "cpu")

    assert plain["tokens"].shape == (1, 3) and plain["name"] == "kept" and plain["flag"] is True
    assert placed["tokens"].shape == (1, 3) and placed["tokens"].device.type == "cpu"


def test_only_the_sampler_settings_that_were_given_are_passed_on():
    assert processor._sampler_overrides(None, None, None) == {}
    assert processor._sampler_overrides(0.5, None, 100) == {"noise_scale": 0.5, "max_inference_sigma": 100}
    assert processor._sampler_overrides(1.0, 2.0, 3) == {"noise_scale": 1.0, "step_scale": 2.0, "max_inference_sigma": 3}


def test_the_builder_prepares_batched_features_and_chain_records_for_a_request():
    builder = ESMFold2InputBuilder()

    features, chains = builder.prepare_input(request(), seed=3)
    called, called_chains = builder(request(), seed=3, device="cpu")

    assert features["token_bonds"].shape == (1, TOKENS, TOKENS, 1) and features["ref_pos"].shape == (1, ATOMS, 3)
    assert [chain.chain_id for chain in chains] == ["A", "L"] and [chain.chain_id for chain in called_chains] == ["A", "L"]
    assert torch.equal(features["ref_pos"], called["ref_pos"]) and called["msa"].shape == (1, 1, TOKENS)


def test_the_builder_loads_the_dictionary_when_it_is_made(monkeypatch):
    loaded = []
    monkeypatch.setattr(processor, "load_ccd", lambda cache: loaded.append(cache))

    ESMFold2InputBuilder("a/cache/directory")
    ESMFold2InputBuilder()

    assert loaded == ["a/cache/directory", None]


def test_folding_runs_the_model_on_batched_features_and_decodes_one_complex():
    model = StandInModel()

    folded = ESMFold2InputBuilder().fold(model, request(), num_loops=2, num_sampling_steps=7, seed=1, complex_id="query")

    call = model.calls[0]
    assert call["num_loops"] == 2 and call["num_sampling_steps"] == 7 and call["num_diffusion_samples"] == 1
    assert call["early_exit"] is False and call["return_dict"] is True and call["verbose"] is False
    assert "noise_scale" not in call and call["msa"].shape[0] == 1
    assert folded.complex.id == "query" and folded.complex.sequence == ["ALA", "GLY", "EOH"]
    assert folded.complex.atom_positions.shape == (12, 3) and folded.plddt.shape == (TOKENS,)
    assert folded.ptm == pytest.approx(0.6) and folded.iptm == pytest.approx(0.3)
    assert folded.pae.shape == (TOKENS, TOKENS) and folded.distogram.shape == (TOKENS, TOKENS, 4)


def test_sampler_settings_and_the_verbose_flag_reach_the_model_and_show_two_progress_stages(capsys):
    model = StandInModel()

    folded = ESMFold2InputBuilder().fold(model, request(), noise_scale=0.25, step_scale=1.5, max_inference_sigma=80, early_exit=True, verbose=True)

    call = model.calls[0]
    assert (call["noise_scale"], call["step_scale"], call["max_inference_sigma"]) == (0.25, 1.5, 80)
    assert call["early_exit"] is True and call["verbose"] is True and folded.complex.sequence == ["ALA", "GLY", "EOH"]
    shown = capsys.readouterr().err
    assert "ESMFold2 typed features" in shown and "ESMFold2 decode" in shown


def test_several_diffusion_samples_decode_to_a_list_of_results():
    model = StandInModel(samples=3)

    results = ESMFold2InputBuilder().fold(model, request(), num_diffusion_samples=3)

    assert len(results) == 3 and [round(item.ptm, 3) for item in results] == [0.6, 0.65, 0.7]
    assert results[0].complex.atom_positions.tolist() != results[1].complex.atom_positions.tolist()


def test_a_checkpoint_trained_without_msa_conditioning_drops_the_msa_features_and_refuses_explicit_alignments():
    builder = ESMFold2InputBuilder()
    without_alignment = StructurePredictionInput(sequences=[LigandInput(id="L", ccd=["EOH"])])
    plain = StandInModel(msa_conditioning=False)
    conditioned = StandInModel(msa_conditioning=True)

    builder.fold(plain, without_alignment)
    builder.fold(conditioned, without_alignment)

    assert not set(MSA_CONDITIONING_INPUT_NAMES) & set(plain.calls[0]) and set(MSA_CONDITIONING_INPUT_NAMES) <= set(conditioned.calls[0])
    with pytest.raises(ValueError, match="rejects explicit MSAs"):
        builder.fold(StandInModel(msa_conditioning=False), request())
    with pytest.raises(RuntimeError, match="no Boolean msa_conditioning"):
        builder.fold(SimpleNamespace(config=SimpleNamespace(msa_conditioning=None), device="cpu"), request())
