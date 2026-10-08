"""Canonical token embeddings: CLS and EOS kept, l + 2 rows in every stream, pooled over all of them.

Every check runs on a tiny randomly initialized ESM++ on CPU. The reference is a plain forward pass on one
tokenized sequence, whose first row is CLS, whose next l rows are the residues and whose last row is EOS.

Symbols: l residues of a sequence; d hidden width; k top-k codes per token; n = l + 2 token rows of a sequence.
"""

from __future__ import annotations

import os
import threading
import time
import pytest
import torch

import fastplms.features.store as store_module

from pathlib import Path
from types import SimpleNamespace

from fastplms.embeddings import (
    HiddenTap,
    ReducedTap,
    ResidueVocabulary,
    SparseResidueTap,
    TapBatch,
    TokenTapExecutor,
    embed_into_features,
    embed_token_features,
    plan_token_batches,
    pool_token_rows,
)
from fastplms.embeddings.taps import plan_taps
from fastplms.features import (
    CSR,
    DENSE,
    RAGGED,
    RAGGED_TOPK,
    AsyncFeatureWriter,
    PackedBatch,
    FeatureStore,
    StoredFeature,
    TopKRow,
    sequence_digest,
)
from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel


N_LAYERS = 3  # hidden states 0..3; state 3 is the final normalized state
HIDDEN_SIZE = 32  # d
CODES = 6  # k, top-k codes per token
CODEBOOK = 16  # c
SEQUENCES = ["MKTAYIAKQR", "GG", "MSTNPKPQRKTKRNT", "ACD", "WYVFH", "ACDEFGHIKLMN"]


def tiny_model() -> ESMplusplusModel:
    torch.manual_seed(0)
    config = ESMplusplusConfig(
        hidden_size=HIDDEN_SIZE, num_attention_heads=4, num_hidden_layers=N_LAYERS, attn_backend="sdpa",
    )
    return ESMplusplusModel(config).eval().requires_grad_(False)


def topk_codes(batch: TapBatch) -> TopKRow:
    """The CODES largest hidden-state entries of every token row as fake SAE codes, packed (n, k)."""
    rows = batch.rows.gather(batch.X)  # (b, m, d) -> (n, d)
    retained = rows.topk(CODES, dim=-1)  # values, indices: (n, k)
    return TopKRow(retained.indices, retained.values.abs())


def pooled_codes(batch: TapBatch) -> torch.Tensor:
    """A (b, c) vector per sequence: the max over all token rows of codes scattered into CODEBOOK columns."""
    selection = batch.selection()
    rows = selection.gather(batch.X)[:, :CODEBOOK].abs()  # (n, c)
    pooled = torch.zeros(len(selection.counts), CODEBOOK)  # (b, c)
    pooled.scatter_reduce_(0, selection.owner.unsqueeze(1).expand_as(rows), rows, reduce="amax", include_self=True)
    pooled[:, ::2] = 0.0  # exact zeros, so the csr layout has something to compress
    return pooled  # (b, c)


def taps_and_features() -> tuple[list, dict[str, StoredFeature]]:
    identity = {"fixture": "token_embeddings"}
    taps = [
        HiddenTap("mid", 1, dtype=torch.bfloat16),
        HiddenTap("final", -1),
        HiddenTap("mean_var", -1, ("mean", "var"), dtype=torch.float32),
        SparseResidueTap("codes", 1, lambda batch: [], identity, codebook_size=HIDDEN_SIZE, sparse_count=CODES,
                         reduce_packed=topk_codes),
        ReducedTap("max_codes", 1, pooled_codes, identity),
    ]
    features = {
        "mid": StoredFeature("mid", RAGGED, HIDDEN_SIZE, torch.bfloat16),
        "final": StoredFeature("final", RAGGED, HIDDEN_SIZE, torch.float32),
        "mean_var": StoredFeature("mean_var", DENSE, 2 * HIDDEN_SIZE, torch.float32),
        "codes": StoredFeature("codes", RAGGED_TOPK, HIDDEN_SIZE, torch.float32, sparse_count=CODES),
        "max_codes": StoredFeature("max_codes", CSR, CODEBOOK, torch.float32),
    }
    return taps, features


def run(model, root: Path, sequences=SEQUENCES, **settings):
    taps, features = taps_and_features()
    options = {"max_sequences": 4, "max_tokens": 64, "window": 8, "part_bytes": 1 << 20, "segment_bytes": 1 << 22,
               "queue_bytes": 1 << 24, "workers": 2} | settings
    return embed_token_features(model, sequences, root, features, taps=taps, **options)


def reference(model, sequence: str, layer: int) -> torch.Tensor:
    """Hidden state `layer` of one sequence run alone: (l + 2, d), CLS first and EOS last."""
    encoded = model.tokenizer([sequence], return_tensors="pt")
    with torch.no_grad():
        states = model(**encoded, output_hidden_states=True).hidden_states  # n_layers + 1 of (1, n, d)
    return states[layer][0]  # (l + 2, d)


def stored(root: Path, name: str):
    _, features = taps_and_features()
    return FeatureStore.open(root, features[name])


def test_every_stream_stores_l_plus_two_rows_with_cls_first_and_eos_last(tmp_path):
    model = tiny_model()
    receipts = run(model, tmp_path)
    assert set(receipts) == {"mid", "final", "mean_var", "codes", "max_codes"}
    final = stored(tmp_path, "final")
    mid = stored(tmp_path, "mid")
    codes = stored(tmp_path, "codes")
    for sequence in SEQUENCES:
        expected = len(sequence) + 2  # l + 2
        rows = final.read([sequence])[0]  # (l + 2, d)
        assert rows.shape == (expected, HIDDEN_SIZE)
        torch.testing.assert_close(rows, reference(model, sequence, N_LAYERS), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            mid.read([sequence])[0].float(), reference(model, sequence, 1).bfloat16().float(), atol=2e-2, rtol=2e-2,
        )
        top = codes.read_topk([sequence])[0]
        assert top.indices.shape == top.values.shape == (expected, CODES)
        assert top.indices.dtype == torch.int32
        # The first row is the CLS token's and the last is EOS's: their codes come from their own states.
        want = reference(model, sequence, 1).topk(CODES, dim=-1)
        torch.testing.assert_close(top.values[0], want.values[0].abs(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(top.values[-1], want.values[-1].abs(), atol=1e-4, rtol=1e-4)
    assert final.residue_counts(SEQUENCES) == [len(sequence) + 2 for sequence in SEQUENCES]


def test_pooling_covers_all_l_plus_two_rows(tmp_path):
    model = tiny_model()
    run(model, tmp_path)
    pooled = stored(tmp_path, "mean_var").read(SEQUENCES)  # b of (2d,)
    for sequence, vector in zip(SEQUENCES, pooled, strict=True):
        rows = reference(model, sequence, N_LAYERS)  # (l + 2, d)
        torch.testing.assert_close(vector[:HIDDEN_SIZE], rows.mean(dim=0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(vector[HIDDEN_SIZE:], rows.var(dim=0, unbiased=False), atol=1e-5, rtol=1e-4)
        residues = rows[1:-1]  # l rows without CLS and EOS
        assert not torch.allclose(vector[:HIDDEN_SIZE], residues.mean(dim=0), atol=1e-6)
    sparse = stored(tmp_path, "max_codes").read_sparse(SEQUENCES)
    assert all(row.indices.dtype == torch.int32 and (row.values >= 0).all() for row in sparse)


def test_padding_never_changes_a_sequence_and_longest_first_batches_cover_each_once(tmp_path):
    model = tiny_model()
    taps, _ = taps_and_features()
    plan = plan_taps(taps, int(model.embedding_tap_state_count))
    executor = TokenTapExecutor(model, plan, vocabulary=ResidueVocabulary(model.tokenizer), max_residues=2046, dtype=None)
    with torch.inference_mode():
        together = executor.run_batch(SEQUENCES[:3], [sequence_digest(s) for s in SEQUENCES[:3]])
        alone = executor.run_batch(SEQUENCES[1:2], [sequence_digest(SEQUENCES[1])])
        together.wait(), alone.wait()
    assert together.rows == (12, 4, 17)  # l + 2 for each sequence, whatever the padding
    start = 12  # rows of the first sequence precede it in the packed (n, d) tensor
    torch.testing.assert_close(
        together.streams["final"]["values"][start:start + 4], alone.streams["final"]["values"], atol=1e-5, rtol=1e-5,
    )
    batches = list(plan_token_batches([len(s) for s in SEQUENCES], max_sequences=3, max_tokens=40, window=4))
    assert sorted(i for batch in batches for i in batch) == list(range(len(SEQUENCES)))
    for batch in batches:
        widest = max(len(SEQUENCES[i]) for i in batch) + 2
        assert len(batch) <= 3 and len(batch) * widest <= 40
    assert batches[0][0] == 2  # the longest of the first window leads it


def test_crop_keeps_the_first_residues_and_both_special_tokens(tmp_path):
    model = tiny_model()
    long = "MKTAYIAKQRMSTNPKPQRKTKRNT"
    run(model, tmp_path, [long], max_residues=7)
    rows = stored(tmp_path, "final").read([long])[0]
    assert rows.shape == (7 + 2, HIDDEN_SIZE)  # N-terminal crop to 7 residues, then CLS and EOS around them
    cropped = reference(model, long[:7], N_LAYERS)  # CLS + 7 residues + EOS
    torch.testing.assert_close(rows, cropped, atol=1e-4, rtol=1e-4)


def test_rerun_embeds_nothing_and_new_rows_append_without_touching_old_parts(tmp_path, monkeypatch):
    model = tiny_model()
    run(model, tmp_path, SEQUENCES[:3])
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.safetensors")}
    monkeypatch.setattr(model, "_embed_taps", lambda *a, **k: pytest.fail("A cache hit ran the model"))
    assert run(model, tmp_path, SEQUENCES[:3]) == {}
    monkeypatch.undo()
    receipts = run(model, tmp_path, SEQUENCES)
    assert all(sum(item.rows for item in group) == 3 for group in receipts.values())
    assert {path: path.read_bytes() for path in before} == before
    assert len(stored(tmp_path, "final")) == len(SEQUENCES)
    assert stored(tmp_path, "final").missing(SEQUENCES) == ()


def test_on_plan_reports_only_the_rows_this_run_will_embed(tmp_path):
    model = tiny_model()
    planned, ticks = [], []
    run(model, tmp_path, SEQUENCES[:2], on_plan=planned.append, progress=ticks.append)
    run(model, tmp_path, SEQUENCES, on_plan=planned.append, progress=ticks.append)  # a resume: two rows are held
    assert planned == [2, len(SEQUENCES) - 2]
    assert sum(ticks) == len(SEQUENCES)  # progress ticks the rows embedded, so a bar over `planned` fills exactly


def test_killed_run_resumes_and_matches_a_clean_run(tmp_path, monkeypatch):
    model = tiny_model()
    clean = tmp_path / "clean"
    run(model, clean, SEQUENCES)
    original = TokenTapExecutor.run_batch
    calls = []
    resumed = tmp_path / "resumed"

    def dying(self, sequences, digests):
        calls.append(len(sequences))
        if len(calls) == 3:
            # Let the writer finish the first batches' segments, as a real kill after minutes of work would.
            deadline = time.monotonic() + 30
            while len(list(resumed.glob("*/segments/*/run.json"))) < 5 and time.monotonic() < deadline:
                time.sleep(0.05)
            raise KeyboardInterrupt
        return original(self, sequences, digests)

    monkeypatch.setattr(TokenTapExecutor, "run_batch", dying)
    with pytest.raises(KeyboardInterrupt):
        run(model, resumed, SEQUENCES, max_sequences=2, max_tokens=64, segment_bytes=1)
    monkeypatch.undo()
    committed = len(stored(resumed, "final"))
    assert 0 < committed < len(SEQUENCES)  # whole segments only: some rows survive, none half-written
    for name in taps_and_features()[1]:
        stored(resumed, name).sweep()  # what a killed run left uncommitted
    run(model, resumed, SEQUENCES, max_sequences=2, max_tokens=64, segment_bytes=1)
    assert stored(resumed, "final").missing(SEQUENCES) == ()
    for name in ("final", "mean_var"):
        for sequence in SEQUENCES:
            torch.testing.assert_close(
                stored(resumed, name).read([sequence])[0], stored(clean, name).read([sequence])[0], atol=1e-5, rtol=1e-5,
            )


def test_a_kill_between_stream_commits_of_the_first_segment_still_resumes_without_a_sweep(tmp_path, monkeypatch):
    model = tiny_model()
    original = store_module.SegmentWriter.commit
    calls = []

    def dying(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 2:
            raise RuntimeError("killed between stream commits")  # the first stream's marker landed, the rest never will
        return original(self, *args, **kwargs)

    monkeypatch.setattr(store_module.SegmentWriter, "commit", dying)
    with pytest.raises(RuntimeError, match="killed between"):
        run(model, tmp_path, SEQUENCES)  # one segment holds every row
    monkeypatch.undo()
    names = list(taps_and_features()[1])
    held = {name: len(stored(tmp_path, name)) for name in names}
    assert sorted(held.values()) == [0] * (len(names) - 1) + [len(SEQUENCES)]  # exactly one stream committed
    receipts = run(model, tmp_path, SEQUENCES)  # no sweep: segment names follow what each stream still lacks
    for name in names:
        assert stored(tmp_path, name).missing(SEQUENCES) == ()
    assert next(name for name, count in held.items() if count) not in receipts  # the finished stream gained nothing


def test_a_schema_carrying_descriptor_needs_its_contract(tmp_path):
    model = tiny_model()
    taps, features = taps_and_features()
    for schema in ("feature_spec_v1", "feature_spec_v2"):
        tagged = {name: StoredFeature(spec.key, spec.layout, spec.width, spec.dtype, descriptor={"schema": schema},
                                      sparse_count=spec.sparse_count) for name, spec in features.items()}
        with pytest.raises(ValueError, match="requires its contract"):
            embed_token_features(model, SEQUENCES, tmp_path, tagged, taps=taps)
        with pytest.raises(ValueError, match="requires its contract"):
            embed_into_features(model, SEQUENCES, tmp_path, tagged, taps=taps, keep_special_tokens=True)
    assert not list(tmp_path.glob("*/segments"))  # nothing was opened for writing


def test_a_residue_only_contract_refuses_the_token_path(tmp_path):
    taps, features = taps_and_features()
    contract = SimpleNamespace(keep_special_tokens=False)
    with pytest.raises(ValueError, match="special tokens kept"):
        embed_token_features(tiny_model(), SEQUENCES, tmp_path, features, taps=taps, contract=contract)


def test_nonfinite_output_is_refused_and_leaves_nothing_visible(tmp_path):
    model = tiny_model()
    taps, features = taps_and_features()

    def bad(batch: TapBatch) -> torch.Tensor:
        return torch.full((batch.X.shape[0], CODEBOOK), float("nan"))  # (b, c)

    taps[-1] = ReducedTap("max_codes", 1, bad, {"fixture": "nan"})
    with pytest.raises(ValueError, match="non-finite"):
        embed_token_features(model, SEQUENCES, tmp_path, features, taps=taps, max_sequences=4, max_tokens=64,
                             window=8, part_bytes=1 << 20, segment_bytes=1 << 22, queue_bytes=1 << 24, workers=2)
    assert not list(tmp_path.glob("*/segments/*/run.json"))


def test_parts_and_directory_syncs_are_bounded_and_open_reads_no_part(tmp_path, monkeypatch):
    model = tiny_model()
    syncs = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda descriptor: (syncs.append(descriptor), real_fsync(descriptor))[1])
    receipts = run(model, tmp_path, SEQUENCES, part_bytes=3000, segment_bytes=1 << 22)
    parts = sum(len(item.parts) for group in receipts.values() for item in group)
    segments = sum(len(group) for group in receipts.values())
    # One flush for each part file and its row-identity sidecar, then the commit marker and the feature descriptor:
    # no flush per window or per row, and none for a re-read.
    assert parts > len(receipts)
    assert len(syncs) <= 2 * parts + 4 * segments + 2 * len(receipts)
    monkeypatch.setattr(store_module, "file_sha256", lambda path: pytest.fail("Opening re-read a part"))
    for spec in taps_and_features()[1].values():
        reopened = FeatureStore.open(tmp_path, spec, deep_verify=False)
        assert len(reopened) == len(SEQUENCES) and reopened.present_digests([sequence_digest(SEQUENCES[0])])


def test_a_failing_part_aborts_without_committing_and_without_hanging(tmp_path, monkeypatch):
    model = tiny_model()

    def refuse(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store_module.SegmentWriter, "append_packed", refuse)
    with pytest.raises(OSError, match="disk full"):
        run(model, tmp_path, SEQUENCES)
    assert not list(tmp_path.glob("*/segments/*/run.json"))




def test_submit_blocks_while_the_queue_is_full_then_everything_commits(tmp_path):
    _, features = taps_and_features()
    store = FeatureStore.open(tmp_path, features["final"])
    first, second = "ACDEF", "GHIKL"
    digests = [sequence_digest(first), sequence_digest(second)]
    gate = threading.Event()

    def batch(sequence: str, digest: str, wait) -> PackedBatch:
        rows = len(sequence) + 2  # l + 2
        return PackedBatch((sequence,), (digest,), (rows,), {"final": {"values": torch.ones(rows, HIDDEN_SIZE)}}, wait)

    one = batch(first, digests[0], lambda: gate.wait(30))
    writer = AsyncFeatureWriter(
        {"final": store}, fingerprint="queue", metadata={}, wanted={"final": frozenset(digests)},
        row_records=lambda stream, texts, keys: [{} for _ in texts], part_bytes=1 << 20, segment_bytes=1 << 22,
        queue_bytes=one.nbytes,  # one batch fills the queue
    )
    writer.submit(one)
    accepted = threading.Event()
    producer = threading.Thread(
        target=lambda: (writer.submit(batch(second, digests[1], lambda: None)), accepted.set()), daemon=True,
    )
    producer.start()
    assert not accepted.wait(0.3)  # the writer is stalled on the first batch, so the second waits
    gate.set()
    assert accepted.wait(30)
    producer.join(30)
    writer.close()
    assert len(FeatureStore.open(tmp_path, features["final"])) == 2


def test_an_aborted_writer_commits_nothing_and_refuses_more_batches(tmp_path):
    _, features = taps_and_features()
    store = FeatureStore.open(tmp_path, features["final"])
    writer = AsyncFeatureWriter(
        {"final": store}, fingerprint="aborted", metadata={}, wanted={"final": frozenset({"0" * 64})},
        row_records=lambda stream, texts, keys: [{} for _ in texts], part_bytes=1 << 20, segment_bytes=1 << 22,
        queue_bytes=1,
    )
    writer.abort()
    with pytest.raises(RuntimeError, match="aborted"):
        writer.submit(PackedBatch(("A",), ("0" * 64,), (3,), {"final": {"values": torch.ones(3, HIDDEN_SIZE)}}, lambda: None))
    assert not list(tmp_path.glob("*/segments/*/run.json"))


def test_an_abort_that_lands_before_the_commit_phase_commits_nothing(tmp_path, monkeypatch):
    _, features = taps_and_features()
    store = FeatureStore.open(tmp_path, features["final"])
    digest = sequence_digest("ACDEF")
    writer = AsyncFeatureWriter(
        {"final": store}, fingerprint="race", metadata={}, wanted={"final": frozenset({digest})},
        row_records=lambda stream, texts, keys: [{} for _ in texts], part_bytes=1 << 20, segment_bytes=1 << 22,
        queue_bytes=1 << 20,
    )
    writer.submit(PackedBatch(("ACDEF",), (digest,), (7,), {"final": {"values": torch.ones(7, HIDDEN_SIZE)}},
                              lambda: None))
    original = AsyncFeatureWriter._drain
    aborter = threading.Thread(target=writer.abort, daemon=True)

    def drain_then_abort(self):
        original(self)
        aborter.start()  # abort lands after the parts are written and before the first commit
        deadline = time.monotonic() + 30
        while not self._aborted and time.monotonic() < deadline:
            time.sleep(0.01)

    monkeypatch.setattr(AsyncFeatureWriter, "_drain", drain_then_abort)
    with pytest.raises(RuntimeError, match="aborted"):
        writer.close()
    aborter.join(30)
    assert not list(tmp_path.glob("*/segments/*/run.json"))  # the rows were written but never became visible


def test_pool_token_rows_matches_pooler_over_a_mask_of_every_token():
    from fastplms.embeddings import Pooler

    torch.manual_seed(1)
    X = torch.randn(3, 7, 8)  # (b, m, d)
    lengths = torch.tensor([7, 4, 2])
    mask = torch.arange(7)[None, :] < lengths[:, None]  # (b, m): every attended token, CLS and EOS included
    names = ("mean", "var", "std", "max", "norm")
    torch.testing.assert_close(pool_token_rows(X, mask, names), Pooler(names)(X, mask), atol=1e-6, rtol=1e-6)


def test_vocabulary_ids_equal_the_tokenizer_with_cls_and_eos():
    model = tiny_model()
    vocabulary = ResidueVocabulary(model.tokenizer)
    for sequence in SEQUENCES:
        assert vocabulary.encode(sequence).tolist() == model.tokenizer([sequence])["input_ids"][0]
    with pytest.raises(ValueError):
        vocabulary.encode("ACJD")  # J has no token
    with pytest.raises(ValueError, match="normalized"):
        vocabulary.encode("acd")
