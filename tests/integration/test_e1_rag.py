from __future__ import annotations

import io
import json
import subprocess
import tarfile
import time
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from email.utils import format_datetime

import pytest
import torch

from fastplms.embeddings import EmbeddingResult, load_sqlite_result
from fastplms.models.e1 import retrieval as e1_retrieval
from fastplms.models.e1.modeling_e1 import (
    E1_MSA_SAMPLING_SOURCE_REVISION,
    ColabFoldSearcher,
    ContextCache,
    ContextSpecification,
    E1Config,
    E1ForMaskedLM,
    E1ForSequenceClassification,
    E1ForTokenClassification,
    HomologueSearcher,
    _safe_extract_tar,
    get_msa_for_sequence,
    get_query_from_a3m,
    load_msa_dir,
    parse_msa,
    sample_context,
    sample_multiple_contexts,
)
from fastplms.registry import load_model_registry

E1_SAMPLING_GOLDEN_REVISION = "bfd2620a602248499f3d2583d85a7ecddf0b6e02"
E1_SAMPLING_GOLDEN_PROVENANCE = {
    "upstream_revision": E1_SAMPLING_GOLDEN_REVISION,
    "source_path": "src/E1/msa_sampling.py",
    "source_sha256": "9a2acc1932fe494613bbc8de0bea415c075f9e21eaa0caa8eb2b693410471e48",
    "generation_command": [
        "python",
        "tools/goldens/generate_e1_sampling.py",
        "--upstream-root",
        "vendor/upstream/e1",
        "--output",
        "artifacts/goldens/e1_sampling.json",
    ],
}
E1_SINGLE_CONTEXT_GOLDEN = {
    0: ("TCDFGHI,ACDEYGH,ACEFGHI", ["mid", "near", "gapped"]),
    3: ("ACDEYGH,TTTTTTTT,TCDFGHI", ["near", "far", "mid"]),
    11: ("ACDEYGH,TCDFGHI,ACEFGHI", ["near", "mid", "gapped"]),
}
E1_MULTIPLE_CONTEXT_GOLDEN = (
    ["TCDFGHI,TTTTTTTT", "ACDEYGH,TCDFGHI,ACEFGHI,ACDEFGHI"],
    [["mid", "far"], ["near", "mid", "gapped", "query"]],
)


def _write_tiny_a3m(path) -> None:
    path.write_text(
        ">query\nACDEFG\n>near\nACDEYG\n>far\nTTTTTT\n",
        encoding="utf-8",
    )


def _write_parity_a3m(path) -> None:
    path.write_text(
        ">query\nACDEFGHI\n>near\nACDEYGH-\n>gapped\nAC-EFGHI\n>mid\nTCD-FGHI\n>far\nTTTTTTTT\n",
        encoding="utf-8",
    )


def _tiny_e1_config() -> E1Config:
    return E1Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_num_sequences=8,
        max_num_positions_within_seq=64,
        max_num_positions_global=256,
        dtype="float32",
    )


def _tiny_e1_model(device: torch.device) -> E1ForMaskedLM:
    return E1ForMaskedLM(config=_tiny_e1_config()).eval().to(device)


def test_a3m_parsing_query_lookup_and_context_sampling(tmp_path) -> None:
    a3m_path = tmp_path / "query.a3m"
    _write_tiny_a3m(a3m_path)

    records = parse_msa(str(a3m_path))
    assert [record.id for record in records] == ["query", "near", "far"]
    assert get_query_from_a3m(str(a3m_path)) == "ACDEFG"

    msa_lookup = load_msa_dir(str(tmp_path))
    assert msa_lookup["ACDEFG"] == str(a3m_path)
    assert get_msa_for_sequence("ACDEYG", msa_lookup, min_identity=0.80) == str(a3m_path)

    context, ids = sample_context(
        msa_path=str(a3m_path),
        max_num_samples=1,
        max_token_length=32,
        max_query_similarity=0.1,
        min_query_similarity=0.0,
        seed=0,
        device=torch.device("cpu"),
    )
    assert context == "TTTTTT"
    assert ids == ["far"]


def test_context_sampling_matches_pinned_e1_golden(tmp_path) -> None:
    registry = load_model_registry()
    assert registry.upstreams["e1"].revision == E1_SAMPLING_GOLDEN_REVISION
    assert E1_MSA_SAMPLING_SOURCE_REVISION == E1_SAMPLING_GOLDEN_REVISION
    assert E1_SAMPLING_GOLDEN_PROVENANCE == {
        "upstream_revision": E1_SAMPLING_GOLDEN_REVISION,
        "source_path": "src/E1/msa_sampling.py",
        "source_sha256": "9a2acc1932fe494613bbc8de0bea415c075f9e21eaa0caa8eb2b693410471e48",
        "generation_command": [
            "python",
            "tools/goldens/generate_e1_sampling.py",
            "--upstream-root",
            "vendor/upstream/e1",
            "--output",
            "artifacts/goldens/e1_sampling.json",
        ],
    }

    a3m_path = tmp_path / "parity.a3m"
    _write_parity_a3m(a3m_path)

    kwargs = {
        "msa_path": str(a3m_path),
        "max_num_samples": 3,
        "max_token_length": 32,
        "max_query_similarity": 0.99,
        "min_query_similarity": 0.0,
        "neighbor_similarity_lower_bound": 0.8,
        "device": torch.device("cpu"),
    }
    for seed in (0, 3, 11):
        assert sample_context(seed=seed, **kwargs) == E1_SINGLE_CONTEXT_GOLDEN[seed]

    specs = [
        ContextSpecification(
            max_num_samples=3,
            max_token_length=16,
            max_query_similarity=0.99,
            min_query_similarity=0.0,
            neighbor_similarity_lower_bound=0.8,
        ),
        ContextSpecification(
            max_num_samples=4,
            max_token_length=32,
            max_query_similarity=1.0,
            min_query_similarity=0.2,
            neighbor_similarity_lower_bound=0.8,
        ),
    ]
    contexts, ids = sample_multiple_contexts(
        msa_path=str(a3m_path),
        context_specifications=specs,
        seed=7,
        device=torch.device("cpu"),
    )
    assert (contexts, ids) == E1_MULTIPLE_CONTEXT_GOLDEN


def test_context_cache_round_trip(tmp_path) -> None:
    cache = ContextCache(str(tmp_path), specs_hash="abc123", seed=7)
    assert cache.load("msa") is None
    cache.store("msa", {"ctx": "ACDEFG"})
    assert cache.load("msa") == {"ctx": "ACDEFG"}
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["source_revision"] == E1_SAMPLING_GOLDEN_REVISION
    assert payload["contexts"] == {"ctx": "ACDEFG"}


def test_context_cache_invalidates_when_msa_content_or_revision_changes(tmp_path) -> None:
    a3m_path = tmp_path / "query.a3m"
    _write_tiny_a3m(a3m_path)
    cache = ContextCache(str(tmp_path / "cache"), specs_hash="abc123", seed=7)
    cache.store(str(a3m_path), {"ctx": "ACDEFG"})
    assert cache.load(str(a3m_path)) == {"ctx": "ACDEFG"}

    a3m_path.write_text(">query\nTTTTTT\n", encoding="utf-8")
    assert cache.load(str(a3m_path)) is None
    cache.store(str(a3m_path), {"ctx": "TTTTTT"})
    assert (
        ContextCache(
            str(tmp_path / "cache"),
            specs_hash="abc123",
            seed=7,
            source_revision="different-revision",
        ).load(str(a3m_path))
        is None
    )


def test_safe_tar_extraction_blocks_traversal(tmp_path) -> None:
    tar_path = tmp_path / "bad.tar"
    payload = b"bad"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("../escape.a3m")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with tarfile.open(tar_path) as tar, pytest.raises(ValueError):
        _safe_extract_tar(tar, str(tmp_path / "out"))


@pytest.mark.parametrize("entry_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_safe_tar_extraction_rejects_links_and_devices(tmp_path, entry_type) -> None:
    tar_path = tmp_path / "unsafe.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("unsafe-entry")
        info.type = entry_type
        if entry_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
            info.linkname = "../escape.a3m"
        tar.addfile(info)

    with tarfile.open(tar_path) as tar, pytest.raises(ValueError):
        _safe_extract_tar(tar, str(tmp_path / "out"))


def test_public_e1_rag_methods_exist() -> None:
    model = _tiny_e1_model(torch.device("cpu"))
    methods = [
        model.search_homologues,
        model.batch_search_homologues,
        model.sample_msa_contexts,
        model.score_ppll,
        model.embed_with_msa,
        model.embed_dataset_with_msa,
    ]
    for method in methods:
        assert callable(method)
    assert callable(model.embed_dataset)


@pytest.mark.parametrize(
    "seq_id",
    ["../escape", "nested/query", r"..\escape", "/tmp/escape", r"C:\escape"],
)
def test_homologue_searchers_reject_seq_ids_outside_output_dir(tmp_path, seq_id) -> None:
    output_dir = str(tmp_path / "msas")
    with pytest.raises(ValueError, match="seq_id"):
        ColabFoldSearcher().search("ACDEFG", output_dir, seq_id=seq_id)
    with pytest.raises(ValueError, match="seq_id"):
        HomologueSearcher(target_db="target_db").search("ACDEFG", output_dir, seq_id=seq_id)


def test_e1_sequence_classifier_accepts_explicit_pooling_types() -> None:
    model = E1ForSequenceClassification(
        _tiny_e1_config(),
        pooling_types=["mean"],
    )
    assert model.pooler.names == ("mean",)
    assert model.classifier[0].in_features == model.config.hidden_size


def test_e1_token_classifier_exactly_consumes_official_encoder_width() -> None:
    """The extension head consumes the pinned encoder's unmodified H tensor."""

    config = _tiny_e1_config()
    config.num_labels = 3
    model = E1ForTokenClassification(config).eval()
    prepared = model.prep_tokens.get_batch_kwargs(["MSTNPKPQ"], device=torch.device("cpu"))
    inputs: dict[str, torch.Tensor] = {}
    for name in (
        "input_ids",
        "within_seq_position_ids",
        "global_position_ids",
        "sequence_ids",
    ):
        value = prepared[name]
        assert isinstance(value, torch.Tensor)
        inputs[name] = value

    with torch.inference_mode():
        encoder_output = model.model(**inputs)
        output = model(**inputs)
        expected_logits = model.classifier(encoder_output.last_hidden_state)

    assert model.classifier[0].in_features == config.hidden_size
    assert output.last_hidden_state.shape[-1] == config.hidden_size
    assert torch.equal(output.last_hidden_state, encoder_output.last_hidden_state)
    assert torch.equal(output.logits, expected_logits)


def test_colabfold_http_errors_are_contextual_and_not_retried(
    monkeypatch,
) -> None:
    searcher = ColabFoldSearcher(max_retries=1, base_delay=0.0, max_delay=0.0)
    headers = Message()
    headers["Content-Type"] = "text/plain"
    sleeps = []

    def fail_request(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://api.colabfold.com/missing",
            404,
            "Not Found",
            headers,
            io.BytesIO(b"missing"),
        )

    monkeypatch.setattr(e1_retrieval.urllib.request, "urlopen", fail_request)
    monkeypatch.setattr(e1_retrieval.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        searcher._request_with_retries("GET", "https://api.colabfold.com/missing")
    assert sleeps == []


def test_colabfold_retry_after_is_case_insensitive_date_aware_and_capped() -> None:
    searcher = ColabFoldSearcher(max_delay=5.0)
    assert searcher._retry_after_delay({"retry-after": "120"}, attempt=0) == 5.0
    retry_at = format_datetime(datetime.now(UTC) + timedelta(minutes=2))
    assert searcher._retry_after_delay({"Retry-After": retry_at}, attempt=0) == 5.0


def test_colabfold_request_respects_expired_deadline(monkeypatch) -> None:
    searcher = ColabFoldSearcher(max_retries=1, max_wait_time=1)

    def unexpected_request(*args, **kwargs):
        raise AssertionError("expired requests must not reach the network")

    monkeypatch.setattr(e1_retrieval.urllib.request, "urlopen", unexpected_request)
    with pytest.raises(TimeoutError, match="deadline"):
        searcher._request_with_retries(
            "GET",
            "https://api.colabfold.com/ticket/expired",
            deadline=time.monotonic() - 1.0,
        )


def test_mmseqs_searcher_subprocess_path_is_mockable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "target_db.dbtype").write_bytes(b"test-db")
    searcher = HomologueSearcher(target_db="target_db")
    calls = []

    identity = e1_retrieval._DockerImageIdentity(
        reference=e1_retrieval.DOCKER_IMAGE,
        repository=e1_retrieval.MMSEQS2_IMAGE_REPOSITORY,
        version=e1_retrieval.MMSEQS2_VERSION,
        manifest_digest=e1_retrieval.MMSEQS2_CPU_MANIFEST_DIGEST,
        image_id="sha256:" + "a" * 64,
        os="linux",
        architecture=e1_retrieval._docker_architecture(),
    )

    def fake_run(cmd, **kwargs):
        del kwargs
        calls.append(cmd)
        if "result2msa" in cmd:
            command_index = cmd.index("result2msa")
            output = tmp_path / cmd[command_index + 4]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(">query\nACDEFG\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(searcher, "_ensure_docker_image", lambda: identity)
    monkeypatch.setattr(searcher, "_run_docker_command", fake_run)

    a3m_path = searcher.search("ACDEFG", output_dir="msas", seq_id="query")

    assert a3m_path == "msas/query/query.a3m"
    assert any("createdb" in call for call in calls)
    assert any("search" in call for call in calls)
    assert any("result2msa" in call for call in calls)
    assert (tmp_path / "msas/query/search-provenance.json").is_file()


def test_colabfold_searcher_http_path_is_mockable(tmp_path, monkeypatch) -> None:
    searcher = ColabFoldSearcher(inter_request_delay=(0.0, 0.0))

    def fake_download(ticket_id: str, output_path: str) -> None:
        payload = b">query\nACDEFG\n"
        with tarfile.open(output_path, "w:gz") as tar:
            info = tarfile.TarInfo("uniref.a3m")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    monkeypatch.setattr(
        searcher,
        "_submit",
        lambda sequence, deadline=None: {"status": "RUNNING", "id": "ticket"},
    )
    monkeypatch.setattr(
        searcher,
        "_poll",
        lambda ticket_id, deadline=None: {"status": "COMPLETE"},
    )
    monkeypatch.setattr(
        searcher,
        "_download",
        lambda ticket_id, output_path, deadline=None: fake_download(ticket_id, output_path),
    )

    a3m_path = searcher.search("ACDEFG", str(tmp_path), seq_id="query")

    assert a3m_path.endswith("query.a3m")
    assert get_query_from_a3m(a3m_path) == "ACDEFG"


@pytest.mark.parametrize(
    ("searcher", "provider"),
    (
        (HomologueSearcher(target_db="target_db"), "mmseqs2"),
        (ColabFoldSearcher(inter_request_delay=(0.0, 0.0)), "colabfold"),
    ),
)
def test_batch_search_warns_on_partial_failure_without_sequence_leak(
    searcher,
    provider,
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    sensitive_sequence = "SENSITIVESEQUENCE"
    monkeypatch.chdir(tmp_path)

    def fail_search(sequence, output_dir, seq_id=None):
        raise RuntimeError(f"provider failure included {sequence}")

    monkeypatch.setattr(searcher, "search", fail_search)
    with caplog.at_level("WARNING", logger=e1_retrieval.__name__):
        result = searcher.batch_search(
            [sensitive_sequence],
            "results",
            seq_ids=["public-seq-id"],
            continue_on_error=True,
        )

    assert result == {}
    assert provider in caplog.text
    assert "public-seq-id" in caplog.text
    assert "RuntimeError" in caplog.text
    assert sensitive_sequence not in caplog.text


@pytest.mark.parametrize(
    "searcher",
    (
        HomologueSearcher(target_db="target_db"),
        ColabFoldSearcher(inter_request_delay=(0.0, 0.0)),
    ),
)
def test_batch_search_preserves_failure_when_continue_is_disabled(
    searcher,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fail_search(sequence, output_dir, seq_id=None):
        raise RuntimeError("provider failure")

    monkeypatch.setattr(searcher, "search", fail_search)
    with pytest.raises(RuntimeError, match="provider failure"):
        searcher.batch_search(
            ["ACDEFG"],
            "results",
            seq_ids=["query"],
            continue_on_error=False,
        )


@pytest.mark.gpu
def test_e1_score_ppll_with_tiny_synthetic_msa(tmp_path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _tiny_e1_model(device)
    a3m_path = tmp_path / "query.a3m"
    _write_tiny_a3m(a3m_path)

    scores = model.score_ppll(
        ["ACDEFG"],
        a3m_path=str(a3m_path),
        max_context_tokens=[64],
        similarity_thresholds=[1.0],
        min_query_similarity=0.0,
        progress=False,
    )

    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0

    per_context_scores = model.score_ppll(
        ["ACDEFG"],
        a3m_path=str(a3m_path),
        ensemble=False,
        max_context_tokens=[64, 128],
        similarity_thresholds=[1.0],
        min_query_similarity=0.0,
        progress=False,
    )
    assert len(per_context_scores) == 1
    assert len(per_context_scores[0]) == 2
    for score in per_context_scores[0]:
        assert 0.0 <= score <= 1.0


@pytest.mark.gpu
def test_e1_embed_with_msa_shapes(tmp_path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _tiny_e1_model(device)
    a3m_path = tmp_path / "query.a3m"
    _write_tiny_a3m(a3m_path)

    pooled = model.embed_with_msa(
        ["ACDEFG"],
        a3m_path=str(a3m_path),
        pooling_types=["mean"],
        progress=False,
    )
    matrix = model.embed_with_msa(
        ["ACDEFG"],
        a3m_path=str(a3m_path),
        matrix_embed=True,
        progress=False,
    )

    assert pooled.shape == (1, model.config.hidden_size)
    assert len(matrix) == 1
    assert matrix[0].shape == (6, model.config.hidden_size)


@pytest.mark.gpu
def test_e1_embed_dataset_with_msa_falls_back_without_msa(tmp_path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _tiny_e1_model(device)
    output = tmp_path / "e1-msa.sqlite"

    embeddings = model.embed_dataset_with_msa(
        ["ACDEFG", "ACDEFG"],
        msa_lookup={},
        batch_size=1,
        max_len=16,
        pooling_types=["mean"],
        progress=False,
        embed_dtype=torch.float32,
        output=output,
        format="sqlite",
    )

    assert isinstance(embeddings, EmbeddingResult)
    assert [(record.id, record.sequence) for record in embeddings] == [
        ("0", "ACDEFG"),
        ("1", "ACDEFG"),
    ]
    assert all(record.load_tensor().shape == (model.config.hidden_size,) for record in embeddings)
    assert embeddings.metadata["descriptor_index"] == "sqlite-records"
    assert embeddings.metadata["family_adapter"]["kind"] == "e1-msa-v1"
    reopened = load_sqlite_result(output)
    assert [record.sequence for record in reopened] == ["ACDEFG", "ACDEFG"]
