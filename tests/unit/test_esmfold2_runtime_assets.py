"""Security contracts for ESMFold2 runtime assets and tensor payloads."""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import pytest
import torch
import zstandard

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

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


def test_ccd_path_replacement_after_hashing_cannot_change_loaded_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trusted_payload = pickle.dumps({"ALA": "trusted"})
    replacement_payload = pickle.dumps({"ALA": "replacement"})
    _install_contract(monkeypatch, trusted_payload)
    asset = tmp_path / "ccd.pkl"
    asset.write_bytes(trusted_payload)
    replacement = tmp_path / "replacement.pkl"
    replacement.write_bytes(replacement_payload)
    real_file_digest = conformers.file_digest

    def replace_path_after_hash(handle: BinaryIO, algorithm: str) -> Any:
        digest = real_file_digest(handle, algorithm)
        os.replace(replacement, asset)
        return digest

    monkeypatch.setattr(conformers, "file_digest", replace_path_after_hash)
    loaded = conformers._ChemicalComponentStore().load(tmp_path)

    assert loaded == {"ALA": "trusted"}
    assert pickle.loads(asset.read_bytes()) == {"ALA": "replacement"}


def test_ccd_in_place_mutation_after_hashing_cannot_change_loaded_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trusted_payload = pickle.dumps({"ALA": "trusted"})
    replacement_payload = pickle.dumps({"ALA": "replacement"})
    _install_contract(monkeypatch, trusted_payload)
    asset = tmp_path / "ccd.pkl"
    asset.write_bytes(trusted_payload)
    real_file_digest = conformers.file_digest

    def mutate_source_after_hash(handle: BinaryIO, algorithm: str) -> Any:
        digest = real_file_digest(handle, algorithm)
        asset.write_bytes(replacement_payload)
        return digest

    monkeypatch.setattr(conformers, "file_digest", mutate_source_after_hash)
    loaded = conformers._ChemicalComponentStore().load(tmp_path)

    assert loaded == {"ALA": "trusted"}
    assert pickle.loads(asset.read_bytes()) == {"ALA": "replacement"}


_XET_HASH = "49e6d925d799aeed5457c617c9d52257c0d283500d5ec4ff0a9db52cf2385e8e"


def _relative_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def _populate_hub_cache(
    hub_root: Path,
    contract: SimpleNamespace,
    payload: bytes,
    *,
    layout: str,
) -> Path:
    """Write the Hub cache layout that huggingface_hub produces and return the snapshot link.

    ``repository``: ``snapshots/<rev>/ccd.pkl -> blobs/<sha256>``, a regular file.
    ``shared``: the same links, with ``blobs/<sha256>`` itself a relative link to the
    cache-wide Xet store entry ``<hub>/blobs/<xet[:2]>/<xet>`` (huggingface_hub 1.32+).
    """

    repository_cache = hub_root / "models--biohub--ESMFold2"
    repository_blob = repository_cache / "blobs" / contract.sha256
    if layout == "repository":
        repository_blob.parent.mkdir(parents=True)
        repository_blob.write_bytes(payload)
    else:
        store_entry = hub_root / "blobs" / _XET_HASH[:2] / _XET_HASH
        store_entry.parent.mkdir(parents=True)
        store_entry.write_bytes(payload)
        _relative_symlink(repository_blob, store_entry)
    snapshot = repository_cache / "snapshots" / contract.revision / contract.path
    _relative_symlink(snapshot, repository_blob)
    return snapshot


def _use_hub_cache(monkeypatch: pytest.MonkeyPatch, hub_root: Path, snapshot: Path) -> None:
    monkeypatch.delenv("ESMCFOLD_CCD_PATH", raising=False)
    monkeypatch.setattr(conformers, "HF_HUB_CACHE", str(hub_root))
    monkeypatch.setattr(conformers, "hf_hub_download", lambda **_kwargs: str(snapshot))


@pytest.mark.parametrize("layout", ["repository", "shared"])
def test_ccd_loader_allows_manifest_owned_hub_snapshot_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    layout: str,
) -> None:
    payload = pickle.dumps({"ALA": "trusted"})
    contract = _install_contract(monkeypatch, payload)
    hub_root = tmp_path / "hub"
    snapshot = _populate_hub_cache(hub_root, contract, payload, layout=layout)
    _use_hub_cache(monkeypatch, hub_root, snapshot)

    assert conformers._ChemicalComponentStore().load() == {"ALA": "trusted"}


def test_ccd_loader_verifies_shared_store_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = pickle.dumps({"ALA": "trusted"})
    contract = _install_contract(monkeypatch, payload)
    hub_root = tmp_path / "hub"
    snapshot = _populate_hub_cache(hub_root, contract, payload, layout="shared")
    store_entry = hub_root / "blobs" / _XET_HASH[:2] / _XET_HASH
    store_entry.write_bytes(pickle.dumps({"ALA": "forged"}).ljust(len(payload), b"\0"))
    _use_hub_cache(monkeypatch, hub_root, snapshot)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        conformers._ChemicalComponentStore().load()


def _link_snapshot_past_repository_blob(hub_root: Path, snapshot: Path) -> None:
    """Point the snapshot straight at the store entry and unbind the pinned repository blob."""

    repository_blobs = snapshot.parent.parent.parent / "blobs"
    for link in repository_blobs.iterdir():
        link.unlink()
    snapshot.unlink()
    _relative_symlink(snapshot, hub_root / "blobs" / _XET_HASH[:2] / _XET_HASH)


def _misfile_store_entry(hub_root: Path, snapshot: Path) -> None:
    """Move the store entry under a prefix directory that does not match its hash."""

    misfiled = hub_root / "blobs" / "00" / _XET_HASH
    misfiled.parent.mkdir()
    os.replace(hub_root / "blobs" / _XET_HASH[:2] / _XET_HASH, misfiled)
    repository_blob = snapshot.parent.parent.parent / "blobs" / snapshot.readlink().name
    repository_blob.unlink()
    _relative_symlink(repository_blob, misfiled)


def _move_store_outside_cache(hub_root: Path, snapshot: Path) -> None:
    """Replace the store directory with a link to an identical store outside the cache root."""

    outside = hub_root.parent / "outside_store"
    os.replace(hub_root / "blobs", outside)
    (hub_root / "blobs").symlink_to(outside, target_is_directory=True)


@pytest.mark.parametrize(
    "tamper",
    [_link_snapshot_past_repository_blob, _misfile_store_entry, _move_store_outside_cache],
)
def test_ccd_loader_rejects_unbound_shared_store_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: Callable[[Path, Path], None],
) -> None:
    payload = pickle.dumps({"ALA": "trusted"})
    contract = _install_contract(monkeypatch, payload)
    hub_root = tmp_path / "hub"
    snapshot = _populate_hub_cache(hub_root, contract, payload, layout="shared")
    tamper(hub_root, snapshot)
    _use_hub_cache(monkeypatch, hub_root, snapshot)

    with pytest.raises(ValueError, match="escapes its repository blob cache"):
        conformers._ChemicalComponentStore().load()


def test_ccd_loader_rejects_hub_snapshot_link_outside_repo_blob_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = pickle.dumps({"ALA": "trusted"})
    contract = _install_contract(monkeypatch, payload)
    hub_root = tmp_path / "hub"
    repository_cache = hub_root / "models--biohub--ESMFold2"
    (repository_cache / "blobs").mkdir(parents=True)
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(payload)
    snapshot = repository_cache / "snapshots" / contract.revision / contract.path
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(outside)
    monkeypatch.delenv("ESMCFOLD_CCD_PATH", raising=False)
    monkeypatch.setattr(conformers, "HF_HUB_CACHE", str(hub_root))
    monkeypatch.setattr(conformers, "hf_hub_download", lambda **_kwargs: str(snapshot))

    with pytest.raises(ValueError, match="escapes its repository blob cache"):
        conformers._ChemicalComponentStore().load()


def test_ccd_loader_rejects_configured_cache_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = pickle.dumps({"ALA": "fixture"})
    _install_contract(monkeypatch, payload)
    target = tmp_path / "trusted.pkl"
    target.write_bytes(payload)
    (tmp_path / "ccd.pkl").symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        conformers._ChemicalComponentStore().load(tmp_path)


def test_ccd_loader_rejects_non_regular_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = pickle.dumps({"ALA": "fixture"})
    _install_contract(monkeypatch, payload)
    (tmp_path / "ccd.pkl").mkdir()

    with pytest.raises(ValueError, match="must be a regular file"):
        conformers._ChemicalComponentStore().load(tmp_path)


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
    expected = {"X": torch.arange(6).reshape(2, 3)}  # (n=2, d=3)
    torch.save(expected, buffer)
    compressed = zstandard.ZstdCompressor().compress(buffer.getvalue())

    actual = deserialize_tensors(compressed)  # actual["X"]: (n=2, d=3)
    assert actual.keys() == expected.keys()
    assert torch.equal(actual["X"], expected["X"])
