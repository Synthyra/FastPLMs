from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest
import torch

from fastplms.e1.modeling_e1 import (
    ColabFoldSearcher,
    ContextCache,
    ContextSpecification,
    E1Config,
    E1ForMaskedLM,
    HomologueSearcher,
    _safe_extract_tar,
    get_msa_for_sequence,
    get_query_from_a3m,
    load_msa_dir,
    parse_msa,
    sample_context,
    sample_multiple_contexts,
)


def _write_tiny_a3m(path) -> None:
    path.write_text(
        ">query\n"
        "ACDEFG\n"
        ">near\n"
        "ACDEYG\n"
        ">far\n"
        "TTTTTT\n",
        encoding="utf-8",
    )


def _write_parity_a3m(path) -> None:
    path.write_text(
        ">query\n"
        "ACDEFGHI\n"
        ">near\n"
        "ACDEYGH-\n"
        ">gapped\n"
        "AC-EFGHI\n"
        ">mid\n"
        "TCD-FGHI\n"
        ">far\n"
        "TTTTTTTT\n",
        encoding="utf-8",
    )


def _load_official_msa_sampling():
    official_src = Path(__file__).resolve().parents[1] / "official" / "e1" / "src"
    sys.path.insert(0, str(official_src))
    try:
        return pytest.importorskip("E1.msa_sampling")
    finally:
        sys.path.remove(str(official_src))


def _tiny_e1_model(device: torch.device) -> E1ForMaskedLM:
    config = E1Config(
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
    return E1ForMaskedLM(config=config).eval().to(device)


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


def test_context_sampling_matches_official_e1(tmp_path) -> None:
    official_msa_sampling = _load_official_msa_sampling()
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
        context, ids = sample_context(seed=seed, **kwargs)
        official_context, official_ids = official_msa_sampling.sample_context(seed=seed, **kwargs)
        assert context == official_context
        assert ids == official_ids

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
    official_specs = [
        official_msa_sampling.ContextSpecification(
            max_num_samples=spec.max_num_samples,
            max_token_length=spec.max_token_length,
            max_query_similarity=spec.max_query_similarity,
            min_query_similarity=spec.min_query_similarity,
            neighbor_similarity_lower_bound=spec.neighbor_similarity_lower_bound,
        )
        for spec in specs
    ]

    contexts, ids = sample_multiple_contexts(
        msa_path=str(a3m_path),
        context_specifications=specs,
        seed=7,
        device=torch.device("cpu"),
    )
    official_contexts, official_ids = official_msa_sampling.sample_multiple_contexts(
        msa_path=str(a3m_path),
        context_specifications=official_specs,
        seed=7,
        device=torch.device("cpu"),
    )
    assert contexts == official_contexts
    assert ids == official_ids


def test_context_cache_round_trip(tmp_path) -> None:
    cache = ContextCache(str(tmp_path), specs_hash="abc123", seed=7)
    assert cache.load("msa") is None
    cache.store("msa", {"ctx": "ACDEFG"})
    assert cache.load("msa") == {"ctx": "ACDEFG"}


def test_safe_tar_extraction_blocks_traversal(tmp_path) -> None:
    tar_path = tmp_path / "bad.tar"
    payload = b"bad"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo("../escape.a3m")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with tarfile.open(tar_path) as tar:
        with pytest.raises(ValueError):
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


def test_mmseqs_searcher_subprocess_path_is_mockable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    searcher = HomologueSearcher(target_db="target_db")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return None

    monkeypatch.setattr(searcher, "_ensure_docker_image", lambda: None)
    monkeypatch.setattr(searcher, "_run_docker_command", fake_run)

    a3m_path = searcher.search("ACDEFG", output_dir="msas", seq_id="query")

    assert a3m_path == "msas/query/query.a3m"
    assert any("createdb" in call for call in calls)
    assert any("search" in call for call in calls)
    assert any("result2msa" in call for call in calls)


def test_colabfold_searcher_http_path_is_mockable(tmp_path, monkeypatch) -> None:
    searcher = ColabFoldSearcher(inter_request_delay=(0.0, 0.0))

    def fake_download(ticket_id: str, output_path: str) -> None:
        payload = b">query\nACDEFG\n"
        with tarfile.open(output_path, "w:gz") as tar:
            info = tarfile.TarInfo("uniref.a3m")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    monkeypatch.setattr(searcher, "_submit", lambda sequence: {"status": "RUNNING", "id": "ticket"})
    monkeypatch.setattr(searcher, "_poll", lambda ticket_id: {"status": "COMPLETE"})
    monkeypatch.setattr(searcher, "_download", fake_download)

    a3m_path = searcher.search("ACDEFG", str(tmp_path), seq_id="query")

    assert a3m_path.endswith("query.a3m")
    assert get_query_from_a3m(a3m_path) == "ACDEFG"


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
def test_e1_embed_dataset_with_msa_falls_back_without_msa() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _tiny_e1_model(device)

    embeddings = model.embed_dataset_with_msa(
        ["ACDEFG"],
        msa_lookup={},
        batch_size=1,
        max_len=16,
        pooling_types=["mean"],
        progress=False,
    )

    assert set(embeddings) == {"ACDEFG"}
    assert embeddings["ACDEFG"].shape == (model.config.hidden_size,)
