"""Unit contracts for the public binder-design example helpers."""

from __future__ import annotations

import sys
import pytest
import torch
from types import ModuleType
from typing import Any

from examples import binder_design_fastplms as binder


class _Position:
    def __init__(self, in_cdr: bool) -> None:
        self._in_cdr = in_cdr

    def is_in_cdr(self) -> bool:
        return self._in_cdr


class _Chain:
    def __init__(self, sequence: str, cdr_offsets: set[int]) -> None:
        self.seq = sequence
        self._cdr_offsets = cdr_offsets

    def __iter__(self):
        for offset, residue in enumerate(self.seq):
            yield _Position(offset in self._cdr_offsets), residue


def test_cdr_indices_use_public_abnumber_multiple_domain_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class PublicChain:
        @classmethod
        def multiple_domains(cls, sequence: str, **kwargs: Any) -> list[_Chain]:
            del cls
            observed.update(sequence=sequence, kwargs=kwargs)
            return [
                _Chain("AAACCC", {1, 4}),
                _Chain("GGGTTT", {0, 5}),
            ]

    abnumber = ModuleType("abnumber")
    abnumber.Chain = PublicChain  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "abnumber", abnumber)

    sequence = "AAACCCLINKGGGTTT"
    assert binder._cdr_indices(sequence) == [1, 4, 10, 15]
    assert observed == {
        "sequence": sequence,
        "kwargs": {
            "scheme": "chothia",
            "allowed_species": None,
            "use_anarcii": True,
        },
    }


def test_binder_helper_validation_uses_explicit_exceptions() -> None:
    with pytest.raises(ValueError, match="Unsupported fixed binder residue"):
        binder.build_initial_soft_sequence_logits("A?", batch_size=1)
    with pytest.raises(ValueError, match="Distogram logits must have shape"):
        binder.compute_distogram_iptm_proxy(
            torch.zeros(3, 3, 128),
            target_length=2,
            binder_sequence="AA",
            is_antibody=False,
        )
    with pytest.raises(ValueError, match="separated by one"):
        binder._binder_sequence_from_designed_sequence("missing-separator")
