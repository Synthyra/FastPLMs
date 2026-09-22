"""CPU tests for opt-in ESMFold progress reporting."""

from __future__ import annotations

import inspect
import pytest
import torch

import fastplms.models.esmfold.modeling_fast_esmfold as esmfold_module

from types import SimpleNamespace
from typing import ClassVar
from torch import nn

from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmForProteinFolding


def _tiny_model() -> FastEsmForProteinFolding:
    model = object.__new__(FastEsmForProteinFolding)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
    )
    model.trunk = nn.Module()
    model.trunk.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.trunk.structure_module = nn.Identity()
    model.trunk.config = SimpleNamespace(max_recycles=2)
    model.esm = nn.Identity()
    model.distogram_head = nn.Identity()
    model.lm_head = nn.Identity()
    model.lddt_head = nn.Identity()
    model.ptm_head = nn.Identity()
    return model


def _patch_tiny_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_forward(
        self: FastEsmForProteinFolding,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
        del input_ids
        num_recycles = kwargs["num_recycles"]
        passes = 1 if num_recycles is None else int(num_recycles) + 1
        value = torch.ones(1)  # (1,)
        self.esm(value)
        for _ in range(passes):
            for block in self.trunk.blocks:
                block(value)
            self.trunk.structure_module(value)
        for name in ("distogram_head", "lm_head", "lddt_head", "ptm_head"):
            getattr(self, name)(value)
        return {
            "plddt": value.reshape(1, 1, 1),
            "s_s": value.reshape(1, 1, 1),
            "attentions": None,
        }

    monkeypatch.setattr(esmfold_module.EsmForProteinFolding, "forward", fake_forward)


class _RecordingProgress:
    instances: ClassVar[list[_RecordingProgress]] = []

    def __init__(self, *, total: int, desc: str, unit: str) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.updates = 0
        self.closed = False
        self.instances.append(self)

    def update(self, count: int = 1) -> None:
        self.updates += count

    def set_description(self, description: str) -> None:
        self.desc = description

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _RecordingProgress:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_forward_is_silent_by_default(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    forward_signature = inspect.signature(FastEsmForProteinFolding.forward)
    assert forward_signature.parameters["verbose"].default is False
    _patch_tiny_forward(monkeypatch)
    monkeypatch.setattr(
        esmfold_module,
        "tqdm",
        lambda **_kwargs: pytest.fail("default forward must not create a progress bar"),
    )

    output = _tiny_model().forward(torch.ones(1, 1), num_recycles=1)

    assert output.plddt is not None
    assert output.plddt.item() == 100.0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_verbose_forward_tracks_trunk_blocks_and_recycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_forward(monkeypatch)
    _RecordingProgress.instances.clear()
    monkeypatch.setattr(esmfold_module, "tqdm", _RecordingProgress)

    output = _tiny_model().forward(
        torch.ones(1, 1),
        num_recycles=1,
        verbose=True,
    )

    progress = _RecordingProgress.instances
    assert len(progress) == 1
    assert progress[0].total == 11
    assert progress[0].updates == 11
    assert progress[0].desc == "ESMFold confidence"
    assert progress[0].closed
    assert output.plddt is not None


def test_verbose_forward_preserves_output_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tiny_forward(monkeypatch)
    monkeypatch.setattr(esmfold_module, "tqdm", _RecordingProgress)

    silent_state_before = torch.random.get_rng_state()
    silent = _tiny_model().forward(torch.ones(1, 1), num_recycles=1)
    silent_state_after = torch.random.get_rng_state()
    verbose_state_before = torch.random.get_rng_state()
    verbose = _tiny_model().forward(torch.ones(1, 1), num_recycles=1, verbose=True)
    verbose_state_after = torch.random.get_rng_state()

    torch.testing.assert_close(silent.plddt, verbose.plddt)
    torch.testing.assert_close(silent.s_s, verbose.s_s)
    assert silent.plddt is not None and verbose.plddt is not None
    assert silent.plddt.dtype == verbose.plddt.dtype
    assert torch.equal(silent_state_before, verbose_state_before)
    assert torch.equal(silent_state_after, verbose_state_after)


def test_infer_forwards_verbose_to_public_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(FastEsmForProteinFolding)
    nn.Module.__init__(model)
    model.register_parameter("_anchor", nn.Parameter(torch.zeros(1)))
    seen: dict[str, object] = {}

    def fake_forward(*_args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        seen["verbose"] = kwargs["verbose"]
        shape = (1, 2, 37)
        return {
            "plddt": torch.ones(shape),
            "atom37_atom_exists": torch.ones(shape),
        }

    monkeypatch.setattr(model, "forward", fake_forward)

    output = model.infer("AC", verbose=True)

    assert seen["verbose"] is True
    assert output["mean_plddt"].item() == 1.0
