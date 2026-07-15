"""Security contracts for ESMFold2 runtime assets and tensor payloads."""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import zstandard

from fastplms.models.esmfold2 import esmfold2_conformers as conformers
from fastplms.models.esmfold2.esmfold2_misc import deserialize_tensors


def _contract(payload: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        repository="biohub/ESMFold2",
        revision="1ebf0e3481a5184eb6171d40615c79e384b48796",
        path="ccd.pkl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        trust_kind="hash_pinned_pickle",
    )


def _install_contract(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> SimpleNamespace:
    contract = _contract(payload)
    registry = SimpleNamespace(runtime_assets={"esmfold2_ccd": contract})
    monkeypatch.setattr(conformers, "get_model_registry", lambda: registry)
    return contract


def test_ccd_local_asset_is_verified_before_pickle_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trusted_payload = pickle.dumps({"ALA": "fixture"})
    _install_contract(monkeypatch, trusted_payload)
    (tmp_path / "ccd.pkl").write_bytes(b"not-the-approved-pickle")

    def fail_if_loaded(_handle: object) -> object:
        raise AssertionError("pickle.load must not run before identity verification")

    monkeypatch.setattr(conformers.pickle, "load", fail_if_loaded)
    store = conformers._ChemicalComponentStore()
    with pytest.raises(ValueError, match=r"size mismatch|SHA256 mismatch"):
        store.load(tmp_path)


def test_ccd_hub_download_uses_manifest_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = pickle.dumps({})
    contract = _install_contract(monkeypatch, payload)
    asset = tmp_path / "ccd.pkl"
    asset.write_bytes(payload)
    monkeypatch.delenv("ESMCFOLD_CCD_PATH", raising=False)
    observed: dict[str, str] = {}

    def fake_download(**kwargs: str) -> str:
        observed.update(kwargs)
        return str(asset)

    monkeypatch.setattr(conformers, "hf_hub_download", fake_download)
    resolved = conformers._ChemicalComponentStore()._resolve_asset(None)

    assert resolved == asset
    assert observed == {
        "repo_id": contract.repository,
        "filename": contract.path,
        "revision": contract.revision,
    }


def test_ccd_verified_pickle_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = pickle.dumps({"ALA": "fixture"})
    _install_contract(monkeypatch, payload)
    (tmp_path / "ccd.pkl").write_bytes(payload)

    assert conformers._ChemicalComponentStore().load(tmp_path) == {"ALA": "fixture"}


def test_tensor_deserialization_rejects_arbitrary_pickle_globals() -> None:
    class UnsafePayload:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return eval, ("40 + 2",)

    buffer = io.BytesIO()
    torch.save(UnsafePayload(), buffer)
    compressed = zstandard.ZstdCompressor().compress(buffer.getvalue())

    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        deserialize_tensors(compressed)


def test_tensor_deserialization_accepts_tensor_mappings() -> None:
    buffer = io.BytesIO()
    expected = {"X": torch.arange(6).reshape(2, 3)}
    torch.save(expected, buffer)
    compressed = zstandard.ZstdCompressor().compress(buffer.getvalue())

    actual = deserialize_tensors(compressed)
    assert actual.keys() == expected.keys()
    assert torch.equal(actual["X"], expected["X"])
