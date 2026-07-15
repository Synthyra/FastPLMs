from __future__ import annotations

import inspect
from types import MethodType

import torch
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding

from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmForProteinFolding


def test_forward_uses_official_plddt_scale(monkeypatch) -> None:
    def forward(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"plddt": torch.tensor([0.75])}

    monkeypatch.setattr(EsmForProteinFolding, "forward", forward)
    model = FastEsmForProteinFolding.__new__(FastEsmForProteinFolding)
    torch.nn.Module.__init__(model)

    output = model.forward(torch.zeros(1, 1, dtype=torch.int64))

    assert output["plddt"].equal(torch.tensor([75.0]))


def test_infer_preserves_official_multimer_contract() -> None:
    assert list(inspect.signature(FastEsmForProteinFolding.infer).parameters) == [
        "self",
        "sequences",
        "residx",
        "masking_pattern",
        "num_recycles",
        "residue_index_offset",
        "chain_linker",
    ]
    model = FastEsmForProteinFolding.__new__(FastEsmForProteinFolding)
    torch.nn.Module.__init__(model)
    model.register_parameter(
        "device_anchor",
        torch.nn.Parameter(torch.empty(0), requires_grad=False),
    )
    observed: dict[str, object] = {}

    def forward(
        self: FastEsmForProteinFolding,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        observed.update(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        atom_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], 37)
        return {
            "aatype": input_ids,
            "atom37_atom_exists": atom_mask,
            "plddt": torch.full_like(atom_mask, 0.75),
        }

    model.forward = MethodType(forward, model)
    output = model.infer(
        "AC:DE",
        num_recycles=2,
        residue_index_offset=32,
        chain_linker="GG",
    )

    assert output["aatype"].shape == (1, 6)
    assert observed["attention_mask"].equal(torch.ones(1, 6, dtype=torch.int64))
    assert observed["position_ids"].equal(torch.tensor([[0, 1, 2, 3, 36, 37]], dtype=torch.int64))
    assert observed["masking_pattern"] is None
    assert observed["num_recycles"] == 2
    assert output["chain_index"].equal(torch.tensor([[0, 0, 0, 0, 1, 1]], dtype=torch.int64))
    assert output["atom37_atom_exists"][0, :, 0].equal(
        torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.float32)
    )
    assert output["mean_plddt"].equal(torch.tensor([0.75]))
