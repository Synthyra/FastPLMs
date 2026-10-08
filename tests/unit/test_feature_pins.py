"""Independent content pins survive unrelated appends and reject self-consistent rewrites."""

from __future__ import annotations

import hashlib
import json
import pickle
import pytest
import torch

from safetensors.torch import load_file, save_file

from fastplms.features import FeatureStore, StoredFeature


def build(root):
    feature = StoredFeature("pinned-fixture", "dense", 4, torch.float32)
    store = FeatureStore.open(root, feature)
    with store.segment("first") as writer:
        writer.append(["AC", "DE"], [torch.arange(4).float(), torch.ones(4)],
                      row_metadata=[{"id": "AC"}, {"id": "DE"}])
    with store.segment("second") as writer:
        writer.append(["FG"], [torch.full((4,), 2.0)], row_metadata=[{"id": "FG"}])
    return store


def test_pins_cover_requested_parts_not_index_or_other_segments(tmp_path):
    store = build(tmp_path)
    pins = store.content_pins(["DE", "AC", "DE"])
    assert set(pins) == {
        "feature.json", "segments/first/run.json", "segments/first/part-00000.safetensors",
        "segments/first/part-00000.rows.json.gz",
    }
    assert all(
        hashlib.sha256((store.directory / path).read_bytes()).hexdigest() == digest
        for path, digest in pins.items()
    )
    pinned = FeatureStore.read_only(store.directory, content_pins=pins)
    with store.segment("third") as writer:
        writer.append(["HI"], [torch.full((4,), 3.0)])
    # Index rebuilding changes its bytes but not the immutable selected features.
    store.reindex()
    torch.testing.assert_close(pinned.read(["DE"])[0], torch.ones(4), rtol=0, atol=0)
    assert pinned.content_pins(["DE"]) == pins
    with pytest.raises(ValueError, match="outside the pinned"):
        pinned.read(["FG"])
    reloaded = pickle.loads(pickle.dumps(pinned))
    torch.testing.assert_close(reloaded.read(["AC"])[0], torch.arange(4).float(), rtol=0, atol=0)


def test_rewriting_values_and_all_internal_checksums_still_fails_independent_pin(tmp_path):
    store = build(tmp_path)
    pins = store.content_pins(["AC"])
    pinned = FeatureStore.read_only(store.directory, content_pins=pins)
    part = store.directory / "segments/first/part-00000.safetensors"
    tensors = {name: tensor.clone() for name, tensor in load_file(str(part)).items()}
    tensors["values"][0, 0] += 10
    save_file(tensors, str(part))
    marker = part.with_name("run.json")
    payload = json.loads(marker.read_text())
    payload["parts"][0]["sha256"] = hashlib.sha256(part.read_bytes()).hexdigest()
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    marker.write_text(json.dumps(payload))
    # A self-consistent internal transaction checksum alone can accept the rewritten source.
    assert FeatureStore.read_only(store.directory).read(["AC"])[0][0].item() == 10
    with pytest.raises(ValueError, match="independent content pin"):
        pinned.read(["AC"])


@pytest.mark.parametrize("removed", [
    "segments/first/run.json", "segments/first/part-00000.safetensors",
    "segments/first/part-00000.rows.json.gz",
])
def test_incomplete_pin_closure_cannot_read_a_selected_row(tmp_path, removed):
    store = build(tmp_path)
    pins = store.content_pins(["AC"])
    del pins[removed]
    pinned = FeatureStore.read_only(store.directory, content_pins=pins)
    with pytest.raises(ValueError, match="outside the pinned"):
        pinned.row_metadata(["AC"])


@pytest.mark.parametrize("relative", [
    "../feature.json", "/feature.json", "segments/../run.json", "index.sqlite",
    "segments/first/arbitrary.py",
])
def test_pins_only_name_immutable_store_artifacts(tmp_path, relative):
    store = build(tmp_path)
    pins = store.content_pins([])
    pins[relative] = "a" * 64
    with pytest.raises(ValueError, match="immutable store files"):
        FeatureStore.read_only(store.directory, content_pins=pins)


def test_pinned_handle_rechecks_descriptor_after_open(tmp_path):
    store = build(tmp_path)
    pinned = FeatureStore.read_only(store.directory, content_pins=store.content_pins(["AC"]))
    descriptor = store.directory / "feature.json"
    descriptor.write_bytes(descriptor.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="independent content pin"):
        pinned.read(["AC"])
