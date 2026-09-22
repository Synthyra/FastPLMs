"""Optional progress display preserves Boltz2 inference values and random draws."""

from __future__ import annotations

import inspect
import pytest
import torch

from torch import Tensor, nn

from fastplms.models.boltz import modeling_boltz2
from fastplms.models.boltz.modeling_boltz2 import (
    Boltz2Config,
    Boltz2InferenceCore,
    Boltz2Model,
    _default_steering_args,
)
from fastplms.models.boltz.vb_modules_diffusionv2 import AtomDiffusion


class _TinyDiffusion(AtomDiffusion):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.score_model = nn.Linear(1, 1)
        self.num_sampling_steps = 2
        self.gamma_min = 0.1
        self.gamma_0 = 0.1
        self.noise_scale = 1.0
        self.step_scale = 1.0
        self.step_scale_random = None
        self.alignment_reverse_diff = False

    def sample_schedule(self, num_sampling_steps: int) -> Tensor:
        return torch.linspace(1.0, 0.0, num_sampling_steps + 1)

    def preconditioned_network_forward(
        self, noised_atom_coords: Tensor, sigma: float, network_condition_kwargs: dict
    ) -> Tensor:
        return noised_atom_coords * 0.5


def _tiny_core() -> Boltz2InferenceCore:
    core = Boltz2InferenceCore.__new__(Boltz2InferenceCore)
    nn.Module.__init__(core)
    core.input_embedder = lambda feats: feats["embedding"]
    for name in ("s_init", "z_init_1", "z_init_2", "s_recycle", "z_recycle", "s_norm", "z_norm"):
        setattr(core, name, nn.Identity())
    core.rel_pos = lambda feats: torch.zeros(1, 2, 2, 2)
    core.token_bonds = lambda bonds: bonds.expand(-1, -1, -1, 2)
    core.contact_conditioning = lambda feats: torch.zeros(1, 2, 2, 2)
    core.msa_module = lambda z, *args, **kwargs: torch.zeros_like(z)
    core.pairformer_module = lambda s, z, **kwargs: (s + torch.rand_like(s), z)
    core.distogram_module = lambda z: z.unsqueeze(-2)
    core.diffusion_conditioning = lambda **kwargs: (None,) * 6
    core.structure_module = _TinyDiffusion()
    core.confidence_module = lambda **kwargs: {"plddt": torch.rand(1, 3)}
    core.bond_type_feature = False
    core.run_trunk_and_structure = True
    core.skip_run_structure = False
    core.confidence_prediction = True
    core.use_kernels = False
    core.steering_args = _default_steering_args()
    return core.eval()


def test_boltz_progress_preserves_outputs_and_rng(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    core = _tiny_core()
    monkeypatch.setattr(modeling_boltz2, "Boltz2InferenceCore", lambda **kwargs: core)
    model = Boltz2Model(Boltz2Config()).eval()
    feats = {
        "embedding": torch.ones(1, 2, 2),
        "token_pad_mask": torch.ones(1, 2),
        "token_bonds": torch.zeros(1, 2, 2, 1),
        "atom_pad_mask": torch.ones(1, 3),
    }
    torch.manual_seed(17)
    quiet = model(feats, recycling_steps=1, num_sampling_steps=2)
    quiet_rng = torch.random.get_rng_state()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""

    torch.manual_seed(17)
    visible = model(feats, recycling_steps=1, num_sampling_steps=2, verbose=True)
    assert torch.equal(torch.random.get_rng_state(), quiet_rng)
    assert visible.keys() == quiet.keys()
    for name in quiet:
        assert torch.equal(quiet[name], visible[name]), name
    captured = capsys.readouterr()
    for stage in ("Input embeddings", "Recycling", "Diffusion sampling", "Confidence"):
        assert f"Boltz2: {stage}" in captured.err


def test_boltz_public_progress_defaults_to_false() -> None:
    for method in (Boltz2Model.forward, Boltz2Model.predict_structure, AtomDiffusion.sample):
        assert inspect.signature(method).parameters["verbose"].default is False
