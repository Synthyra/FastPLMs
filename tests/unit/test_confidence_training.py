"""CPU checkpoint persistence checks for the confidence training loop."""

from __future__ import annotations

import sys
import types
import torch

from tools.confidence import training


def test_resume_allows_hardware_change_but_preserves_scientific_settings():
    settings = {"gpu": "NVIDIA H100", "seed": 17, "dataset_sha256": "a"}
    assert training._checkpoint_settings_match(settings, dict(settings, gpu="NVIDIA H200"))
    assert not training._checkpoint_settings_match(settings, dict(settings, seed=29))
    assert not training._checkpoint_settings_match(settings, dict(settings, dataset_sha256="b"))


def test_save_checkpoint_round_trips_training_state(monkeypatch, tmp_path) -> None:
    commit_calls: list[str] = []

    class FakeVolume:
        @classmethod
        def from_name(cls, name: str):
            return cls()

        def commit(self) -> None:
            commit_calls.append("committed")

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Volume=FakeVolume))
    monkeypatch.setattr(
        training,
        "get_model_spec",
        lambda model_id: types.SimpleNamespace(fast=types.SimpleNamespace(revision="revision")),
    )
    monkeypatch.setattr(training.torch.cuda, "get_rng_state_all", lambda: [])

    context = types.SimpleNamespace(head=torch.nn.Linear(3, 2), base_weight_sha256="base-hash")
    optimizer = torch.optim.AdamW(context.head.parameters(), lr=1e-4)
    input_tensor = torch.ones(1, 3)  # (1, 3)
    loss = context.head(input_tensor).sum()
    loss.backward()
    optimizer.step()
    torch.manual_seed(17)
    expected_rng = torch.get_rng_state().clone()
    expected_head = {
        key: value.detach().clone() for key, value in context.head.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()
    path = tmp_path / "last.pt"
    training._save_checkpoint(
        path,
        context,
        optimizer,
        {"setting": "value"},
        12,
        {"plddt_ce": 0.2, "pae_ce": 0.3},
        2,
        "esmfold2_300",
        ("rel", "bond"),
        123.5,
    )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["update"] == 12
    assert saved["best"] == {"plddt_ce": 0.2, "pae_ce": 0.3}
    assert saved["training_seconds"] == 123.5
    assert saved["settings"] == {"setting": "value"}
    assert saved["model_id"] == "esmfold2_300"
    assert saved["model_revision"] == "revision"
    assert saved["frozen_hashes"] == ("rel", "bond")
    assert saved["base_weight_sha256"] == "base-hash"
    assert saved["best_sha256"] is None
    assert saved["donor_validation_sha256"] is None
    assert torch.equal(saved["rng"], expected_rng)
    for key, value in expected_head.items():
        assert torch.equal(saved["head"][key], value)
    assert saved["optimizer"]["param_groups"] == expected_optimizer["param_groups"]
    assert commit_calls == ["committed"]
