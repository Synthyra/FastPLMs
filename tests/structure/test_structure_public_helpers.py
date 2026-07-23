"""CPU contracts for public structure convenience helpers."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from examples.binder_design_fastplms import compute_structure_losses
from fastplms.models.boltz import modeling_boltz2, vb_modules_confidencev2
from fastplms.models.boltz.minimal_featurizer import build_boltz2_features
from fastplms.models.boltz.vb_loss_diffusionv2 import smooth_lddt_loss
from fastplms.models.boltz.vb_modules_encodersv2 import get_indexing_matrix
from fastplms.models.boltz.vb_potentials_potentials import FlatBottomPotential
from fastplms.models.boltz.vb_potentials_schedules import PiecewiseStepFunction
from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmForProteinFolding
from fastplms.models.esmfold2.esmfold2_affine3d import Affine3D
from fastplms.models.esmfold2.esmfold2_predicted_aligned_error import tm_loss
from fastplms.models.esmfold2.protein_utils import prepare_protein_features
from fastplms.models.esmfold2.reproducibility import seed_context

pytestmark = pytest.mark.structure


class _FakeBoltz:
    def __init__(self, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.device = torch.device(device)
        self.config = SimpleNamespace(num_bins=64)
        self.core = SimpleNamespace(
            input_embedder=SimpleNamespace(
                atom_encoder=SimpleNamespace(atoms_per_window_queries=32)
            )
        )
        self.parameter = torch.nn.Parameter(torch.ones(1, dtype=dtype))

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self.parameter

    def _to_model_device(
        self,
        feats: dict[str, torch.Tensor],
        float_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        assert float_dtype == torch.float32
        return feats

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        return {
            "sample_atom_coords": torch.randn((1, 1, 3)),
            "plddt": torch.rand((1, 1)),
            "complex_plddt": torch.rand((1,)),
            "iptm": torch.rand((1,)),
            "ptm": torch.rand((1,)),
        }


def _fake_boltz_features(
    amino_acid_sequence: str,
    num_bins: int,
    atoms_per_window_queries: int,
) -> tuple[dict[str, torch.Tensor], SimpleNamespace]:
    assert num_bins == 64
    assert atoms_per_window_queries == 32
    # Exercise every ambient stream that the public helper promises to scope.
    random.random()
    np.random.random()
    return {
        "atom_pad_mask": torch.ones((1, 1)),
        "ref_pos": torch.randn((1, 1, 3)),
    }, SimpleNamespace(sequence=amino_acid_sequence)


def test_boltz_public_helper_is_seeded_and_restores_ambient_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(modeling_boltz2, "build_boltz2_features", _fake_boltz_features)
    model = _FakeBoltz()

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    expected_next = (random.random(), float(np.random.random()), torch.rand(1))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)

    first = modeling_boltz2.Boltz2Model.predict_structure(model, "ACD", seed=17)
    observed_next = (random.random(), float(np.random.random()), torch.rand(1))
    second = modeling_boltz2.Boltz2Model.predict_structure(model, "ACD", seed=17)

    assert observed_next[0] == expected_next[0]
    assert observed_next[1] == expected_next[1]
    torch.testing.assert_close(observed_next[2], expected_next[2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.sample_atom_coords,
        second.sample_atom_coords,
        rtol=0.0,
        atol=0.0,
    )
    assert first.seed == second.seed == 17


@pytest.mark.parametrize(
    "invalid_seed",
    (True, False, 1.0, "17", b"17", np.int64(17)),
)
def test_boltz_public_helper_rejects_coerced_seed_types_before_rng_mutation(
    monkeypatch: pytest.MonkeyPatch,
    invalid_seed: object,
) -> None:
    monkeypatch.setattr(
        modeling_boltz2,
        "build_boltz2_features",
        lambda *_args, **_kwargs: pytest.fail("feature preparation was reached"),
    )
    model = _FakeBoltz()

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    expected_next = (random.random(), float(np.random.random()), torch.rand(1))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)

    with pytest.raises(TypeError, match="seed must be an int or None"):
        modeling_boltz2.Boltz2Model.predict_structure(
            model,
            "ACD",
            seed=invalid_seed,  # type: ignore[arg-type]
        )

    observed_next = (random.random(), float(np.random.random()), torch.rand(1))
    assert observed_next[0] == expected_next[0]
    assert observed_next[1] == expected_next[1]
    torch.testing.assert_close(observed_next[2], expected_next[2], rtol=0.0, atol=0.0)


def test_boltz_public_helper_owns_cuda_bf16_autocast_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(modeling_boltz2, "build_boltz2_features", _fake_boltz_features)
    observed: list[tuple[str, torch.dtype]] = []

    @contextmanager
    def fake_autocast(*, device_type: str, dtype: torch.dtype) -> Iterator[None]:
        observed.append((device_type, dtype))
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    modeling_boltz2.Boltz2Model.predict_structure(_FakeBoltz(device="cuda"), "ACD", seed=3)

    assert observed == [("cuda", torch.bfloat16)]
    with pytest.raises(ValueError, match="requires FP32 parameter storage"):
        modeling_boltz2.Boltz2Model.predict_structure(
            _FakeBoltz(dtype=torch.bfloat16),
            "ACD",
            seed=3,
        )


def test_boltz_public_helper_rejects_non_finite_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(modeling_boltz2, "build_boltz2_features", _fake_boltz_features)
    model = _FakeBoltz()
    model.forward = lambda **_kwargs: {  # type: ignore[method-assign]
        "sample_atom_coords": torch.tensor([[[float("nan"), 0.0, 0.0]]])
    }

    with pytest.raises(RuntimeError, match="sample_atom_coords contains non-finite"):
        modeling_boltz2.Boltz2Model.predict_structure(model, "ACD", seed=3)


@pytest.mark.parametrize(
    ("k", "w", "h", "error_type", "message"),
    (
        (True, 4, 8, TypeError, "k must be an int"),
        (0, 4, 8, ValueError, "k must be positive"),
        (1, 3, 6, ValueError, "w must be even"),
        (1, 4, 7, ValueError, "h must be divisible"),
        (1, 4, 6, ValueError, "even number of half-window key blocks"),
    ),
)
def test_boltz_indexing_matrix_rejects_invalid_public_dimensions(
    k: object,
    w: int,
    h: int,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        get_indexing_matrix(  # type: ignore[arg-type]
            k,
            w,
            h,
            torch.device("cpu"),
        )


def test_boltz_piecewise_schedule_validates_and_owns_its_configuration() -> None:
    with pytest.raises(ValueError, match="at least one threshold"):
        PiecewiseStepFunction((), (1.0,))
    with pytest.raises(ValueError, match="exactly one more value"):
        PiecewiseStepFunction((0.5,), (1.0,))
    with pytest.raises(ValueError, match="strictly increasing"):
        PiecewiseStepFunction((0.5, 0.5), (1.0, 2.0, 3.0))

    thresholds = [0.5]
    values = [1.0, 2.0]
    schedule = PiecewiseStepFunction(thresholds, values)
    thresholds.clear()
    values.clear()
    assert schedule.compute(0.5) == 1.0
    assert schedule.compute(0.5001) == 2.0


@pytest.mark.parametrize(
    ("negation_mask", "error_type", "message"),
    (
        (torch.tensor([0]), TypeError, "must be a boolean tensor"),
        (torch.tensor([False, False]), ValueError, "broadcastable to value shape"),
        (torch.tensor([False]), ValueError, "at least one bound is infinite"),
    ),
)
def test_boltz_flat_bottom_potential_rejects_invalid_negation_masks(
    negation_mask: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        FlatBottomPotential.compute_function(
            object(),
            value=torch.tensor([0.5]),
            k=torch.tensor(1.0),
            lower_bounds=torch.tensor([0.0]),
            upper_bounds=torch.tensor([1.0]),
            negation_mask=negation_mask,
        )


def _assert_boltz_atom_confidence_mapping(device: torch.device) -> None:
    batch_size = 2
    multiplicity = 2
    token_count = 2
    slots_per_token = 3
    token_logits = torch.empty(
        batch_size * multiplicity,
        token_count,
        slots_per_token,
        1,
        device=device,
    )
    for batch_sample in range(batch_size * multiplicity):
        for token in range(token_count):
            for slot in range(slots_per_token):
                token_logits[batch_sample, token, slot, 0] = 100 * batch_sample + 10 * token + slot
    atom_to_token = torch.tensor(
        (
            ((0, 1), (1, 0), (0, 1), (0, 0)),
            ((1, 0), (1, 0), (0, 1), (0, 0)),
        ),
        dtype=torch.bool,
        device=device,
    )
    atom_pad_mask = torch.tensor(
        ((1, 1, 1, 0), (1, 1, 1, 0)),
        dtype=torch.bool,
        device=device,
    )

    atom_logits = vb_modules_confidencev2._token_slot_logits_to_atom_logits(
        token_logits,
        atom_to_token,
        atom_pad_mask,
        multiplicity=multiplicity,
    )

    expected = torch.tensor(
        (
            (10, 0, 11, 0),
            (110, 100, 111, 0),
            (200, 201, 210, 0),
            (300, 301, 310, 0),
        ),
        dtype=token_logits.dtype,
        device=device,
    )
    torch.testing.assert_close(atom_logits.squeeze(-1), expected, rtol=0.0, atol=0.0)


def test_boltz_atom_confidence_mapping_preserves_batch_multiplicity_and_atom_order() -> None:
    _assert_boltz_atom_confidence_mapping(torch.device("cpu"))


@pytest.mark.gpu
def test_boltz_atom_confidence_mapping_preserves_batch_multiplicity_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GH200 atom-confidence contract.")
    _assert_boltz_atom_confidence_mapping(torch.device("cuda"))


def test_boltz_atom_confidence_is_finite_for_short_uneven_batches_and_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size = 2
    multiplicity = 2
    token_count = 2
    atom_count = 4
    hidden_size = 4
    head = vb_modules_confidencev2.ConfidenceHeads(
        token_s=hidden_size,
        token_z=hidden_size,
        num_plddt_bins=4,
        num_pde_bins=4,
        num_pae_bins=4,
        token_level_confidence=False,
    )
    s = torch.randn(
        batch_size * multiplicity,
        token_count,
        hidden_size,
        requires_grad=True,
    )
    z = torch.randn(
        batch_size * multiplicity,
        token_count,
        token_count,
        hidden_size,
        requires_grad=True,
    )
    atom_to_token = torch.tensor(
        (
            ((1, 0), (0, 0), (0, 0), (0, 0)),
            ((1, 0), (0, 1), (0, 1), (0, 0)),
        ),
        dtype=torch.bool,
    )
    atom_pad_mask = torch.tensor(
        ((1, 0, 0, 0), (1, 1, 1, 0)),
        dtype=torch.bool,
    )
    feats = {
        "atom_to_token": atom_to_token,
        "atom_pad_mask": atom_pad_mask,
        "mol_type": torch.zeros((batch_size, token_count), dtype=torch.long),
        "asym_id": torch.tensor(((0, 0), (0, 1)), dtype=torch.long),
        "token_pad_mask": torch.tensor(((1, 0), (1, 1)), dtype=torch.float32),
    }
    zeros = torch.zeros(batch_size * multiplicity)
    monkeypatch.setattr(
        vb_modules_confidencev2,
        "compute_ptms",
        lambda *_args, **_kwargs: (zeros, zeros, zeros, zeros, {}),
    )

    output = head(
        s=s,
        z=z,
        x_pred=torch.randn(batch_size * multiplicity, atom_count, 3),
        d=torch.zeros(batch_size * multiplicity, token_count, token_count),
        feats=feats,
        multiplicity=multiplicity,
        pred_distogram_logits=torch.zeros(
            batch_size,
            token_count,
            token_count,
            64,
        ),
    )

    assert output["plddt_logits"].shape == (
        batch_size * multiplicity,
        atom_count,
        4,
    )
    assert output["resolved_logits"].shape == (
        batch_size * multiplicity,
        atom_count,
        2,
    )
    assert torch.isfinite(output["complex_pde"]).all()
    torch.testing.assert_close(
        output["complex_pde"][:multiplicity],
        torch.zeros(multiplicity),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(output["complex_plddt"]).all()
    assert torch.isfinite(output["complex_iplddt"]).all()
    assert not output["plddt_logits"][:multiplicity, 1:].any()
    assert not output["plddt_logits"][multiplicity:, 3:].any()

    (output["plddt"].mean() + output["pde"].mean()).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_esmfold_fold_single_uses_linker_masked_mean_plddt() -> None:
    class FakeESMFold:
        def infer(self, sequence: str) -> dict[str, torch.Tensor]:
            assert sequence == "AC:DE"
            return {
                "plddt": torch.full((1, 29, 37), 1.0),
                "mean_plddt": torch.tensor([87.5]),
                "ptm": torch.tensor([0.75]),
            }

    result = FastEsmForProteinFolding._fold_single(
        FakeESMFold(),
        "AC:DE",
        return_pdb_string=False,
    )

    assert result["plddt"] == 87.5
    assert result["ptm"] == 0.75
    assert "pdb_string" not in result


def test_boltz_real_features_flow_through_tiny_core_and_structure_loss() -> None:
    with seed_context(19):
        features, template = build_boltz2_features("ACDE")
        tiny_core = torch.nn.Linear(3, 3, bias=False)

    reference_positions = features["ref_pos"]
    predicted_positions = tiny_core(reference_positions)
    atom_mask = features["atom_pad_mask"].bool()
    loss = smooth_lddt_loss(
        predicted_positions,
        reference_positions,
        is_nucleotide=torch.zeros_like(atom_mask),
        coords_mask=atom_mask,
    )
    loss.backward()

    assert len(template.atom_names) == int(atom_mask.sum())
    assert reference_positions.shape[1] % 32 == 0
    assert torch.isfinite(loss)
    assert tiny_core.weight.grad is not None
    assert torch.isfinite(tiny_core.weight.grad).all()
    assert torch.count_nonzero(tiny_core.weight.grad) > 0


@pytest.mark.parametrize("invalid_seed", (True, 1.5, "19"))
def test_esmfold2_seed_context_rejects_non_integer_seeds_without_rng_mutation(
    invalid_seed: object,
) -> None:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    with (
        pytest.raises(TypeError, match="excluding bool"),
        seed_context(invalid_seed),  # type: ignore[arg-type]
    ):
        raise AssertionError("invalid seeds must fail before entering the context")

    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[0] == numpy_state[0]
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("raise_inside", (False, True))
def test_esmfold2_seed_context_restores_all_available_rng_streams(
    raise_inside: bool,
) -> None:
    random.seed(201)
    np.random.seed(202)
    torch.manual_seed(203)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(204)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def exercise() -> None:
        with seed_context(29):
            random.random()
            np.random.random()
            torch.rand(3)
            if torch.cuda.is_available():
                torch.rand(3, device="cuda")
            if raise_inside:
                raise RuntimeError("seed-context-test")

    if raise_inside:
        with pytest.raises(RuntimeError, match="seed-context-test"):
            exercise()
    else:
        exercise()

    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[0] == numpy_state[0]
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_state is not None:
        current_cuda_state = torch.cuda.get_rng_state_all()
        assert len(current_cuda_state) == len(cuda_state)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(current_cuda_state, cuda_state, strict=True)
        )


def test_esmfold2_real_features_flow_through_tiny_core_and_tm_loss() -> None:
    features = prepare_protein_features("ACDE")
    input_ids = features["input_ids"]
    sequence_length = input_ids.shape[1]

    with seed_context(23):
        embedding = torch.nn.Embedding(int(input_ids.max()) + 1, 8)
        pair_projection = torch.nn.Linear(8, 16)

    token_embeddings = embedding(input_ids)
    pair_features = token_embeddings.unsqueeze(2) + token_embeddings.unsqueeze(1)
    pae_logits = pair_projection(pair_features)
    target_frames = Affine3D.identity(
        (input_ids.shape[0], sequence_length),
        dtype=pae_logits.dtype,
        device=pae_logits.device,
    ).tensor
    loss = tm_loss(
        pae_logits,
        pred_affine=target_frames,
        targ_affine=target_frames,
        targ_mask=features["token_attention_mask"],
    )
    loss.backward()

    assert features["ref_pos"].shape[1] % 32 == 0
    assert int(features["atom_attention_mask"].sum()) > sequence_length
    assert torch.isfinite(loss)
    for parameter in (*embedding.parameters(), *pair_projection.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert (
        sum(
            int(torch.count_nonzero(parameter.grad))
            for parameter in (*embedding.parameters(), *pair_projection.parameters())
            if parameter.grad is not None
        )
        > 0
    )


def test_binder_structure_loss_is_finite_and_differentiable() -> None:
    with seed_context(29):
        distogram_logits = torch.randn((2, 16, 16, 128), requires_grad=True)

    losses = compute_structure_losses(distogram_logits, binder_length=12)
    losses["total_loss"].mean().backward()

    assert set(losses) == {
        "glob_loss",
        "inter_contact_loss",
        "intra_contact_loss",
        "total_loss",
    }
    assert all(loss.shape == (2,) for loss in losses.values())
    assert all(torch.isfinite(loss).all() for loss in losses.values())
    assert distogram_logits.grad is not None
    assert torch.isfinite(distogram_logits.grad).all()
    assert torch.count_nonzero(distogram_logits.grad) > 0
