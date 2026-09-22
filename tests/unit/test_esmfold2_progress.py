"""CPU coverage for ESMFold2 verbose progress reporting."""

from __future__ import annotations

import inspect
import pytest
import torch

from types import SimpleNamespace
from torch import Tensor, nn

from fastplms.models.esmfold2 import modeling_esmfold2 as standard
from fastplms.models.esmfold2 import modeling_esmfold2_common as common
from fastplms.models.esmfold2 import modeling_esmfold2_experimental as experimental


class _RecordingBar:
    def __init__(self, iterable=None, **kwargs) -> None:
        self.iterable = iterable
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def update(self, _count: int = 1) -> None:
        return None


class _RandomTrunk(nn.Module):
    def forward(self, value: Tensor, *, pair_attention_mask: Tensor) -> Tensor:
        del pair_attention_mask
        return value + torch.rand_like(value)


class _DummyDiffusion(nn.Module):
    def forward(self, *, x_noisy: Tensor, **_kwargs):
        return {
            "x_denoised": x_noisy,
            "token_repr": None,
            "atom_intermediates": None,
        }


def _tiny_standard_model() -> standard.ESMFold2Model:
    model = object.__new__(standard.ESMFold2Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        lm_encoder=SimpleNamespace(per_loop_lm_dropout=False, lm_dropout=0.0)
    )
    model.lm_encoder = None
    model.msa_encoder = None
    model.parcae_input_norm = nn.Identity()
    model.folding_trunk = _RandomTrunk()
    return model


def _run_standard_loop(model: standard.ESMFold2Model, verbose: bool) -> tuple[Tensor, Tensor]:
    shape = (1, 2, 2, 3)
    return (
        model._run_one_loop(
            z=torch.zeros(shape),
            z_init=torch.zeros(shape),
            lm_z=None,
            _msa_inputs=None,
            pair_mask=torch.ones(1, 2, 2),
            a=torch.ones(1, 1, 1, 3),
            b_mat=torch.eye(3),
            tok_mask=torch.ones(1, 2),
            total_steps=2,
            verbose=verbose,
        ),
        torch.get_rng_state(),
    )


@pytest.mark.cpu_contract
def test_public_forward_verbose_defaults_to_quiet() -> None:
    for model_class in (standard.ESMFold2Model, experimental.ESMFold2ExperimentalModel):
        for method_name in ("forward", "fold", "fold_protein"):
            parameter = inspect.signature(getattr(model_class, method_name)).parameters["verbose"]
            assert parameter.default is False


@pytest.mark.cpu_contract
@pytest.mark.parametrize(
    "model_class", (standard.ESMFold2Model, experimental.ESMFold2ExperimentalModel)
)
def test_high_level_fold_passes_verbose_to_typed_builder(model_class) -> None:
    seen: dict[str, object] = {}

    class StrictBuilder:
        def fold(self, model, input_value, *, verbose=False, **kwargs):
            seen.update(kwargs)
            seen["model"] = model
            seen["input"] = input_value
            seen["verbose"] = verbose
            return "result"

    model = object.__new__(model_class)
    nn.Module.__init__(model)
    builder = StrictBuilder()
    model._esmfold2_input_builder = builder
    input_value = object()

    assert model.fold(input_value, verbose=True) == "result"
    assert seen["model"] is model
    assert seen["input"] is input_value
    assert seen["verbose"] is True


@pytest.mark.cpu_contract
def test_standard_recycling_progress_does_not_change_output_or_rng(monkeypatch) -> None:
    bars: list[_RecordingBar] = []

    def record_tqdm(iterable=None, **kwargs):
        bar = _RecordingBar(iterable, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(standard, "tqdm", record_tqdm)
    model = _tiny_standard_model()

    torch.manual_seed(41)
    quiet, quiet_rng = _run_standard_loop(model, verbose=False)
    assert bars == []

    torch.manual_seed(41)
    verbose, verbose_rng = _run_standard_loop(model, verbose=True)
    assert torch.equal(quiet, verbose)
    assert torch.equal(quiet_rng, verbose_rng)
    assert len(bars) == 1
    assert bars[0].kwargs == {
        "total": 2,
        "desc": "ESMFold2 recycling",
        "unit": "loop",
    }


@pytest.mark.cpu_contract
def test_diffusion_progress_does_not_change_output_or_rng(monkeypatch) -> None:
    bars: list[_RecordingBar] = []

    def record_tqdm(iterable=None, **kwargs):
        bar = _RecordingBar(iterable, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(common, "tqdm", record_tqdm)
    head = object.__new__(common.DiffusionStructureHead)
    nn.Module.__init__(head)
    head.diffusion_module = _DummyDiffusion()
    head.inference_num_steps = 2
    head.sigma_data = 1.0
    head.gamma_0 = 0.0
    head.gamma_min = 0.0
    head.noise_scale = 0.0
    head.step_scale = 0.0
    head.inference_noise_schedule = lambda _steps, device: torch.tensor(
        [2.0, 1.0, 0.0], device=device
    )
    head._center_random_augmentation = lambda value, _mask, second_coords=None: (
        value,
        second_coords,
    )
    head._weighted_rigid_align = lambda _x, x_gt, _weights, _mask: x_gt

    arguments = {
        "z_trunk": torch.zeros(1, 1, 1, 1),
        "s_inputs": torch.zeros(1, 1, 1),
        "s_trunk": None,
        "relative_position_encoding": torch.zeros(1, 1, 1, 1),
        "ref_pos": torch.zeros(1, 1, 3),
        "ref_charge": torch.zeros(1, 1),
        "ref_mask": torch.ones(1, 1, dtype=torch.bool),
        "ref_element": torch.zeros(1, 1, 2),
        "ref_atom_name_chars": torch.zeros(1, 1, 2, dtype=torch.long),
        "ref_space_uid": torch.zeros(1, 1, dtype=torch.long),
        "tok_idx": torch.zeros(1, 1, dtype=torch.long),
        "asym_id": torch.zeros(1, 1, dtype=torch.long),
        "residue_index": torch.zeros(1, 1, dtype=torch.long),
        "entity_id": torch.zeros(1, 1, dtype=torch.long),
        "token_index": torch.zeros(1, 1, dtype=torch.long),
        "sym_id": torch.zeros(1, 1, dtype=torch.long),
        "num_sampling_steps": 2,
        "max_inference_sigma": None,
    }

    torch.manual_seed(53)
    quiet = head.sample(**arguments, verbose=False)["sample_atom_coords"]
    quiet_rng = torch.get_rng_state()
    assert bars == []

    torch.manual_seed(53)
    verbose = head.sample(**arguments, verbose=True)["sample_atom_coords"]
    verbose_rng = torch.get_rng_state()
    assert torch.equal(quiet, verbose)
    assert torch.equal(quiet_rng, verbose_rng)
    assert len(bars) == 1
    assert bars[0].kwargs == {
        "total": 2,
        "desc": "ESMFold2 diffusion",
        "unit": "step",
    }
