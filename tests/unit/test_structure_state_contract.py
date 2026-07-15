"""Unit contracts for compact exact structure-checkpoint metadata."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from tests.structure.support.state_contract import (
    exact_state_contract,
    semantic_config_contract,
    tensor_sha256,
    validate_exact_state_contract,
    validate_semantic_config_contract,
)


class SharedParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        parameter = nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
        self.left = parameter
        self.right = parameter
        self.register_buffer("counter", torch.tensor(3, dtype=torch.int64))


def test_exact_state_contract_covers_scalars_hashes_and_aliases() -> None:
    model = SharedParameterModel()

    contract = exact_state_contract(model)

    assert contract["tensors"]["counter"] == {
        "dtype": "int64",
        "shape": [],
        "sha256": tensor_sha256(model.counter),
    }
    assert contract["aliases"] == [["left", "right"]]
    validate_exact_state_contract(contract)


def test_exact_state_contract_applies_names_and_exclusions() -> None:
    model = SharedParameterModel()

    contract = exact_state_contract(
        model,
        name_transform=lambda name: (f"canonical.{name}",),
        excluded_prefixes=("counter",),
    )

    assert set(contract["tensors"]) == {"canonical.left", "canonical.right"}
    assert contract["aliases"] == [["canonical.left", "canonical.right"]]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract.__setitem__("sha256", "0" * 64),
        lambda contract: contract["tensors"]["counter"].__setitem__("shape", [-1]),
        lambda contract: contract["tensors"]["counter"].__setitem__("sha256", "not-a-hash"),
        lambda contract: contract["aliases"].append(["left", "missing"]),
    ],
)
def test_exact_state_contract_rejects_corruption(
    mutation: Callable[[dict[str, object]], None],
) -> None:
    contract = exact_state_contract(SharedParameterModel())
    mutation(contract)

    with pytest.raises(ValueError, match="Structure state"):
        validate_exact_state_contract(contract)


def test_semantic_config_contract_removes_packaging_fields_recursively() -> None:
    contract = semantic_config_contract(
        {
            "hidden_size": 8,
            "fastplms_model_id": "toy",
            "nested": {
                "architectures": ["PackagingOnly"],
                "depth": 2,
                "fastplms_checkpoint_hash": "a" * 64,
            },
            "dtype": torch.bfloat16,
        }
    )

    assert contract["fields"] == {
        "hidden_size": 8,
        "nested": {"depth": 2},
    }
    validate_semantic_config_contract(contract)


def test_semantic_config_contract_rejects_corruption() -> None:
    contract = semantic_config_contract({"hidden_size": 8})
    changed = copy.deepcopy(contract)
    changed["fields"]["hidden_size"] = 16

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_semantic_config_contract(changed)

    changed = copy.deepcopy(contract)
    changed["fields"]["auto_map"] = {"AutoModel": "remote.Model"}
    with pytest.raises(ValueError, match="packaging fields"):
        validate_semantic_config_contract(changed)
