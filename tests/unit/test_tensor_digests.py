"""The tensor digests keep the exact bytes that structure bundles, goldens, and validation reports already record."""

from __future__ import annotations

import hashlib
import pytest
import torch

from tests.structure.support import state_contract
from torch import Tensor

from tools.tensor_digests import raw_tensor_sha256, tensor_bytes, tensor_set_sha256, typed_tensor_sha256


def _vectors() -> dict[str, Tensor]:
    return {  # (...) float32_2x3 (2, 3), int64_2x2 (2, 2), bfloat16_4 (4,), bool_3 (3,), float32_transposed (3, 2), float32_empty (0, 3)
        "float32_2x3": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "int64_2x2": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        "bfloat16_4": torch.tensor([0.5, -1.25, 3.0, 2.0], dtype=torch.bfloat16),
        "bool_3": torch.tensor([True, False, True]),
        "float32_transposed": torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        "float32_empty": torch.zeros(0, 3),
    }


# Digests the structure-bundle, state-contract, and small-ESMFold2 implementations produced for these tensors
# before they shared one module.
RAW_DIGESTS = {
    "float32_2x3": "e2c0a71510b5394df7773b63fb5f54372b84c3564e67811bde7d665be227976d",
    "int64_2x2": "73e200e2b048c86d4e8c86b86bf62bbda84c7384e34e250b01aa30ab29d234a4",
    "bfloat16_4": "e5987fc32f0e247b9d11763899996e95bd68f338f29341f1bd888c3b63797a45",
    "bool_3": "85f90dfea1d8027e1463e5ca971a250110a20df0119d204a74220bc63516d15b",
    "float32_transposed": "0c9d0bb54e4f5a0121543129f106617549c7ff2b34c6842c5a2e19186c5a7914",
    "float32_empty": hashlib.sha256(b"").hexdigest(),
}
TYPED_DIGESTS = {
    "float32_2x3": "e20933188a476ce2418266391f2f875503b2dbea37a1f5a230c18b535bc495b5",
    "int64_2x2": "4c489611b77975a30d439cac14986e77f6542dd2b9021ec371cc8557a19038af",
    "bfloat16_4": "5daa7ea3bf586979e0265a31aa3b4eb2832d7499d4d2d6e24b6a2ff57918bda3",
    "bool_3": "d6a426a839235407c7f72158bcb3ecdec74a93342b3ceaf9e200f8994ebab8dc",
    "float32_transposed": "e3ed7467bcf5c4e04cc2b66bf5cf4cfa4721cd1edc4c7233090360f1413fd8a4",
    "float32_empty": "e4d7d5460f71e03636e7c49d61a9ccd4bc6910b15a8368367e4c0ca367ec744b",
}
SET_DIGEST = "9ea35c2dd86a8d142859a7c211bfc022b9abef5426d8d8e30e6df56d95345410"
SCALAR_RAW_DIGEST = "267ff33d242cf99619876d639456f362147845ca99fdd30d4195275ab833c807"


@pytest.mark.parametrize("name", sorted(RAW_DIGESTS))
def test_raw_digest_keeps_the_recorded_bytes(name: str) -> None:
    assert raw_tensor_sha256(_vectors()[name]) == RAW_DIGESTS[name]


@pytest.mark.parametrize("name", sorted(TYPED_DIGESTS))
def test_typed_digest_keeps_the_recorded_bytes(name: str) -> None:
    assert typed_tensor_sha256(_vectors()[name]) == TYPED_DIGESTS[name]


def test_set_digest_keeps_the_recorded_bytes_and_ignores_insertion_order() -> None:
    vectors = _vectors()
    state = {"b.weight": vectors["float32_2x3"], "a.bias": vectors["int64_2x2"], "c": vectors["bfloat16_4"]}
    assert tensor_set_sha256(state) == SET_DIGEST
    assert tensor_set_sha256(dict(reversed(list(state.items())))) == SET_DIGEST


def test_a_scalar_tensor_hashes_its_one_value() -> None:
    scalar = torch.tensor(1.5, dtype=torch.float16)
    assert raw_tensor_sha256(scalar) == SCALAR_RAW_DIGEST
    assert tensor_bytes(scalar) == tensor_bytes(scalar.reshape(1))


def test_value_bytes_are_those_of_the_contiguous_cpu_copy() -> None:
    transposed = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    assert tensor_bytes(transposed) == transposed.contiguous().numpy().tobytes()
    gradient_tensor = torch.ones(3, requires_grad=True)
    assert tensor_bytes(gradient_tensor) == torch.ones(3).numpy().tobytes()


def test_the_raw_digest_matches_across_dtypes_that_share_bytes_and_the_typed_digest_does_not() -> None:
    integers = torch.tensor([1, 2], dtype=torch.int32)
    reinterpreted = integers.view(torch.float32)
    assert raw_tensor_sha256(integers) == raw_tensor_sha256(reinterpreted)
    assert typed_tensor_sha256(integers) != typed_tensor_sha256(reinterpreted)
    assert typed_tensor_sha256(integers) != typed_tensor_sha256(integers.reshape(2, 1))


@pytest.mark.parametrize("name", sorted(RAW_DIGESTS))
def test_the_isolated_oracle_twin_hashes_each_tensor_as_the_host_module_does(name: str) -> None:
    tensor = _vectors()[name]
    assert state_contract.tensor_sha256(tensor) == raw_tensor_sha256(tensor) == RAW_DIGESTS[name]


def test_the_isolated_oracle_twin_hashes_a_tensor_set_as_the_host_module_does() -> None:
    vectors = _vectors()
    state = {"b.weight": vectors["float32_2x3"], "a.bias": vectors["int64_2x2"], "c": vectors["bfloat16_4"]}
    assert state_contract.tensor_set_sha256(state) == tensor_set_sha256(state) == SET_DIGEST
    assert state_contract.tensor_set_sha256({}) == tensor_set_sha256({})
