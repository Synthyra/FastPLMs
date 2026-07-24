from __future__ import annotations

import ast
import concurrent.futures
import os
import re
import subprocess
import sys
import pytest
from pathlib import Path
from typing import Any, Self
from urllib.parse import unquote, urlsplit

from tools.artifacts.generate_docs import (
    CAPABILITY_EVIDENCE_SELECTORS,
    EMBEDDING_CAPABILITY_ROWS,
    GENERATION_CAPABILITY_ROWS,
    STRUCTURE_CAPABILITY_ROWS,
    attention_backend_evidence_keys,
    autoclass_evidence_keys,
    benchmark_autoclass_evidence_pairs,
    benchmark_backend_evidence,
    synchronize,
)
from tools.debug.check_notation import (
    iter_repository_files,
    scan_repository,
    violations_in_text,
)


ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_ROOTS = (
    ROOT / "AGENTS.md",
    ROOT / "CLAUDE.md",
    ROOT / "README.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "LICENSES",
    ROOT / "benchmarks" / "README.md",
    ROOT / "docker" / "README.md",
    ROOT / "docs",
    ROOT / "examples",
    ROOT / "model_cards",
    ROOT / "tools" / "debug" / "README.md",
    ROOT / "tools" / "remote" / "README.md",
    ROOT / "vendor" / "README.md",
)
FENCE_PATTERN = re.compile(
    r"^```(?P<language>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
UNBACKED_CLAIM_PATTERNS = (
    re.compile(
        r"\b(?:is|are|has been|have been)\s+"
        r"(?:fully\s+|exactly\s+)?equivalent\b",
        re.I,
    ),
    re.compile(
        r"\b(?:state[- ]of[- ]the[- ]art|outperforms?|"
        r"\d+(?:\.\d+)?\s*[x\u00d7]\s+(?:faster|speedup)|"
        r"\d+(?:\.\d+)?%\s+faster)\b",
        re.I,
    ),
)
LICENSE_FILE_PATTERN = re.compile(r"^LICEN[CS]E(?:[._-].*)?$", re.I)
MODEL_CARD_FILE_PATTERN = re.compile(r"^(?:MODEL_CARD|README)\.md$", re.I)
OFFLINE_EXAMPLES = (
    "artifact_loading.py",
    "embedding_and_retrieval.py",
    "attention_switching.py",
    "ankh_embeddings.py",
    "generation.py",
    "e1_rag.py",
    "ttt.py",
    "structure_preparation.py",
    "task_heads.py",
    "fine_tuning.py",
    "binder_design_fastplms.py",
)


def _markdown_files() -> tuple[Path, ...]:
    paths: list[Path] = []
    for candidate in MARKDOWN_ROOTS:
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(sorted(candidate.rglob("*.md")))
    return tuple(paths)


def _python_snippet(path: Path, marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    matches = [
        match.group("body")
        for match in FENCE_PATTERN.finditer(text)
        if match.group("language").lower() in {"python", "py"} and marker in match.group("body")
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one Python snippet containing {marker!r} in {path}, found {len(matches)}."
        )
    return matches[0]


def _local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    split = urlsplit(target)
    if split.scheme or split.netloc or not split.path:
        return None
    decoded = unquote(split.path)
    destination = ROOT / decoded.lstrip("/") if decoded.startswith("/") else source.parent / decoded
    resolved = destination.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise AssertionError(
            f"Documentation link escapes the repository: {source}: {raw_target}"
        ) from error
    return resolved


def test_shape_notation_detector_rejects_square_and_uppercase_dimensions() -> None:
    text = (
        "H: [" + "B, L, 81, 2560]\n"
        "Z: (" + "B, L, D)\n"
        "M: [" + "n_atoms]\n"
        "X: [" + "samples, atoms, 3]"
    )
    violations = list(violations_in_text(text, path=Path("example.md")))
    assert len(violations) == 4


def test_repository_documentation_uses_canonical_shape_notation() -> None:
    violations = scan_repository(ROOT)
    assert not violations, "\n" + "\n".join(violation.render(ROOT) for violation in violations)


def test_notation_inventory_includes_container_and_provenance_docs() -> None:
    paths = {path.relative_to(ROOT).as_posix() for path in iter_repository_files(ROOT)}
    assert {
        "LICENSES/README.md",
        "THIRD_PARTY_NOTICES.md",
        "docker/Dockerfile",
        "docker/docker-bake.hcl",
        "docker/compose.yaml",
        "vendor/README.md",
    }.issubset(paths)


def test_runtime_model_packages_do_not_embed_licenses_or_model_cards() -> None:
    model_root = ROOT / "src" / "fastplms" / "models"
    misplaced = sorted(
        path.relative_to(ROOT).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
        and (
            LICENSE_FILE_PATTERN.fullmatch(path.name)
            or MODEL_CARD_FILE_PATTERN.fullmatch(path.name)
        )
    )
    assert not misplaced, (
        "Runtime model packages must not embed license files or model cards; "
        "use LICENSES/ and model_cards/:\n" + "\n".join(misplaced)
    )


def test_manifest_generated_documentation_is_current() -> None:
    failures = synchronize(ROOT, check=True)
    assert not failures, "\n" + "\n".join(failures)


def test_generated_capability_evidence_covers_manifest() -> None:
    from fastplms.registry import load_model_registry

    registry = load_model_registry()
    text = (ROOT / "docs" / "generated" / "capability_evidence.md").read_text(encoding="utf-8")
    for family in registry.families.values():
        assert f"`{family.id}`" in text
        assert f"`{family.tokenizer_mode}`" in text
        for auto_class in family.auto_map:
            assert f"`{auto_class}`" in text
        for backend in family.attention:
            assert f"`{backend}`" in text
    for heading in (
        "Input, embedding, and storage contracts",
        "Generation and adaptation contracts",
        "Structure contracts",
    ):
        assert f"## {heading}" in text
    advertised_entries = sum(len(family.auto_map) for family in registry.families.values())
    assert text.count("[runnable AutoClass contract]") == advertised_entries
    assert "ANKH embeddings and generation" in text
    assert "pocket requests fail closed" in text


def test_autoclass_capability_evidence_matches_runtime_and_benchmark_selectors() -> None:
    from benchmarks.suite import benchmark_auto_class, benchmark_cases
    from fastplms.registry import get_model_registry
    from tests.cpu.test_autoclass_evidence_matrix import AUTOCLASS_EVIDENCE

    registry = get_model_registry()
    manifest_pairs = {
        (family.id, auto_class)
        for family in registry.families.values()
        for auto_class in family.auto_map
    }
    assert set(AUTOCLASS_EVIDENCE) == manifest_pairs

    specs_by_checkpoint = {
        (spec.fast.repo_id, spec.fast.revision): spec for spec in registry.values()
    }
    benchmark_pairs: set[tuple[str, str]] = set()
    for case in benchmark_cases(family=None, quick=False, local_files_only=True):
        if not case.claim_eligible:
            continue
        spec = specs_by_checkpoint[(case.model, case.revision)]
        assert case.auto_class == benchmark_auto_class(spec)
        benchmark_pairs.add((spec.family.id, case.auto_class))
    assert benchmark_autoclass_evidence_pairs(registry) == benchmark_pairs

    for family_id, auto_class in manifest_pairs:
        family = registry.families[family_id]
        evidence = set(autoclass_evidence_keys(registry, family_id, auto_class))
        assert {"cpu:autoclass-runtime", "artifact:checkpoint-autoclasses"}.issubset(evidence)
        assert not any(key.startswith("feature:") for key in evidence)
        assert ("benchmark:claim-eligible-primary-head" in evidence) == (
            (family_id, auto_class) in benchmark_pairs
        )
        if auto_class == "AutoConfig" or "Classification" in auto_class:
            assert not any(key.startswith("compliance:") for key in evidence)
            assert not any(key.startswith("benchmark:") for key in evidence)
        if family.id == "ankh" and auto_class == "AutoModelForSeq2SeqLM":
            assert "compliance:ankh-seq2seq" in evidence
            assert "benchmark:claim-eligible-primary-head" not in evidence


def test_backend_capability_evidence_uses_real_nightly_and_benchmark_scopes() -> None:
    from benchmarks.suite import benchmark_cases
    from fastplms.registry import get_model_registry

    registry = get_model_registry()
    advertised = {backend for family in registry.families.values() for backend in family.attention}
    benchmarked = {
        case.backend
        for case in benchmark_cases(family=None, quick=False, local_files_only=True)
        if case.claim_eligible
    }
    assert benchmark_backend_evidence(registry) == benchmarked

    for backend in advertised:
        evidence = set(attention_backend_evidence_keys(registry, backend))
        assert "cpu:attention-contracts" in evidence
        assert not any(key.startswith("feature:") for key in evidence)
        assert ("benchmark:claim-eligible-backends" in evidence) == (backend in benchmarked)
        has_sequence_family = any(
            family.tokenizer_mode != "structure" and backend in family.attention
            for family in registry.families.values()
        )
        has_current_gh200_execution = backend in {"eager", "sdpa", "flex_attention"}
        assert ("nightly:sequence-backends" in evidence) == (
            has_sequence_family and has_current_gh200_execution
        )
        assert ("historical:fa2-focused" in evidence) == (backend == "flash_attention_2")
        assert ("compliance:flash-unavailable-gh200" in evidence) == backend.startswith(
            "flash_attention_"
        )
        if backend.startswith("flash_attention_"):
            assert "compliance:deep-backends" not in evidence
            assert "benchmark:claim-eligible-backends" not in evidence


def test_capability_rows_fail_closed_against_invalid_tier_inheritance() -> None:
    rows = EMBEDDING_CAPABILITY_ROWS + GENERATION_CAPABILITY_ROWS + STRUCTURE_CAPABILITY_ROWS
    by_capability = {row.capability: row for row in rows}
    assert len(by_capability) == len(rows)
    for row in rows:
        assert row.evidence
        assert set(row.evidence).issubset(CAPABILITY_EVIDENCE_SELECTORS)

    peft = by_capability["Trainer/PEFT LoRA with immutable inputs and verified save/reload"]
    assert peft.evidence == ("cpu:peft", "nightly:peft")

    binder = by_capability["Atom-dense binder optimization and critic reporting"]
    assert not any(key.startswith("benchmark:") for key in binder.evidence)

    artifact = by_capability["Offline local artifact AutoClass loading"]
    assert artifact.evidence == (
        "cpu:artifact-example",
        "artifact:checkpoint-autoclasses",
    )

    mapping = by_capability["Ordered mapping or one-shot generator"]
    assert "tests/cpu/test_embedding_contracts.py" in mapping.example
    extended_poolers = by_capability["Max/norm/median/variance/CLS/PARTI pooling"]
    assert "tests/cpu/test_embedding_contracts.py" in extended_poolers.example
    for row in EMBEDDING_CAPABILITY_ROWS:
        if row.capability == "E1 raw-sequence and MSA-aware ordered embeddings":
            continue
        assert not any(key.startswith("feature:") for key in row.evidence)


def test_capability_evidence_selectors_resolve_to_their_declared_jobs() -> None:
    from tools.remote.run import SUITES

    compose = (ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    testing = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
    assert "tests/cpu" in testing and "cpu_contract" in testing

    commands = {
        name: tuple(suite.command)
        + tuple(part for command in suite.pre_commands for part in command)
        for name, suite in SUITES.items()
    }
    for selector in CAPABILITY_EVIDENCE_SELECTORS.values():
        for target in selector.targets:
            relative = target.split("::", maxsplit=1)[0]
            path = ROOT / relative
            assert path.exists(), f"Evidence selector target does not exist: {relative}"
            if selector.tier == "cpu_contract":
                assert relative.startswith("tests/cpu/")
            elif selector.tier in {"feature", "nightly", "compliance"}:
                assert relative in commands[selector.tier]
            elif selector.tier == "artifact":
                assert relative.startswith("tests/release/")
                assert "tests/release" in commands["artifact"]
            elif selector.tier == "structure":
                assert relative == "tests/structure" or relative.startswith("tests/structure/")
                assert "tests/structure" in commands["structure"]
            elif selector.tier == "benchmark":
                assert relative == "benchmarks/suite.py"
                assert "benchmark" in commands["benchmark"]
                assert 'entrypoint: ["python", "-m", "benchmarks.suite"]' in compose
            elif selector.tier == "historical":
                assert relative == "tools/remote/run.py"
                assert target.endswith("::_kernel_capability_preflight")
            else:
                raise AssertionError(f"Unvalidated evidence tier: {selector.tier}")


def test_generated_esmc_cards_state_mask_precedence_and_route_hopper_scope_to_docs() -> None:
    for name in ("esmc_small.md", "esmc_large.md", "esmc_6b.md"):
        text = (ROOT / "model_cards" / name).read_text(encoding="utf-8")
        assert "When `sequence_id` is supplied" in text
        assert "`attention_mask` is ignored" in text
        assert "/docs/attention_backends.md" in text
        assert "current exact GH200/aarch64" not in text
        assert "H100 environment" not in text

    attention_docs = " ".join(
        (ROOT / "docs/attention_backends.md").read_text(encoding="utf-8").split()
    )
    assert "exact GH200/aarch64 workstation" in attention_docs
    assert "H100 and H200" in attention_docs
    assert "not current release evidence" in attention_docs


def test_generated_cards_publish_canonical_state_commitments() -> None:
    from fastplms.registry import get_model_registry

    for spec in get_model_registry().values():
        if spec.canonical_state_sha256 is None:
            continue
        card = (ROOT / "model_cards" / f"{spec.id}.md").read_text(encoding="utf-8")
        assert f"Canonical transformed state SHA-256: `{spec.canonical_state_sha256}`" in card
        assert "Conversion equality attestation: recorded in `provenance.json`" in card


def test_curated_offline_examples_expose_executable_help() -> None:
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    def run_help(name: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        path = ROOT / "examples" / name
        return name, subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
                timeout=20,
            check=False,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(run_help, OFFLINE_EXAMPLES))

    for name, result in results:
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert "usage:" in result.stdout.lower()
    help_by_name = dict(results)
    structure_help = " ".join(help_by_name["structure_preparation.py"].stdout.split())
    assert "requires a full 48-block checkpoint" in structure_help


def test_routine_setup_avoids_parity_submodules_and_documents_manual_cpu_gate() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dependencies = readme.split("## Dependencies", maxsplit=1)[1].split(
        "## Quick start", maxsplit=1
    )[0]
    validation = readme.split("## Validation and reproducibility", maxsplit=1)[1]
    assert "git clone --recurse-submodules" not in dependencies
    assert "git submodule update --init --recursive" not in dependencies
    assert "Official reference repositories are not runtime" in dependencies
    assert "git submodule update --init --recursive" in validation
    assert "does not use GitHub Actions" in validation
    assert "tests/cpu" in validation

    remote = (ROOT / "tools" / "remote" / "README.md").read_text(encoding="utf-8")
    testing = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")
    assert "does not use GitHub Actions" in remote
    assert "no GitHub Actions workflows" in testing


def test_container_guide_runs_complete_candidate_and_compliance_workflows() -> None:
    text = (ROOT / "docker" / "README.md").read_text(encoding="utf-8")
    assert "candidate --load" in text
    assert "python -m pytest tests/unit tests/integration" in text
    assert '-m "not gpu and not slow and not structure"' in text
    assert "--suite compliance" in text
    assert "Building those images alone does not run\nparity" in text


def test_hub_quick_starts_follow_install_and_platform_contracts() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "docs" / "attention_backends.md",
        ROOT / "docs" / "embedding_api.md",
        ROOT / "docs" / "esmfold2.md",
        ROOT / "docs" / "finetuning.md",
        ROOT / "docs" / "migration.md",
        ROOT / "docs" / "models.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        loading = text.index(".from_pretrained(")
        prefix = text[:loading]
        assert "pip install" in prefix, path.relative_to(ROOT)
        assert re.search(r"Python 3\.11(?:-| through )3\.14", prefix), path.relative_to(ROOT)
        assert "PyTorch 2.13" in prefix, path.relative_to(ROOT)
        assert "Transformers 5.13" in prefix, path.relative_to(ROOT)


def test_esmfold2_fast_docs_do_not_claim_msa_conditioning() -> None:
    for path in (
        ROOT / "README.md",
        ROOT / "docs" / "migration.md",
        ROOT / "docs" / "models.md",
        ROOT / "examples" / "README.md",
    ):
        text = " ".join(path.read_text(encoding="utf-8").split())
        assert "24 folding blocks" in text, path.relative_to(ROOT)
        assert "48" in text and "optional MSA conditioning" in text, path.relative_to(ROOT)
        assert "reject MSA-derived inputs" in text, path.relative_to(ROOT)
        assert "https://biohub.ai/papers/esm_protein.pdf" in text, path.relative_to(ROOT)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = readme.split(
        "### ESMFold2 folding and learned representations",
        maxsplit=1,
    )[1].split("## Attention backends", maxsplit=1)[0]
    normalized_quick_start = " ".join(quick_start.split())
    assert '"Synthyra/ESMFold2-Fast"' in quick_start
    assert (
        "quick start below intentionally loads Fast and supplies no MSA" in normalized_quick_start
    )
    assert "Protein inputs can also carry an MSA" not in quick_start
    assert "Its ESMFold2 MSA branch requires one of the full checkpoints" in normalized_quick_start

    native_preparation = readme.split("### Native biological preparation", maxsplit=1)[1].split(
        "### Ordered embedding results", maxsplit=1
    )[0]
    normalized_native_preparation = " ".join(native_preparation.split())
    assert "Full ESMFold2 checkpoints additionally retain optional MSA conditioning" in (
        normalized_native_preparation
    )
    assert "ESMFold2 Fast checkpoints reject MSA-derived inputs" in normalized_native_preparation

    docs_index = " ".join((ROOT / "docs" / "README.md").read_text(encoding="utf-8").split())
    assert "the distinct full and Fast MSA contracts" in docs_index


def test_ankh_seq2seq_docs_describe_live_full_checkpoints() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "docs" / "artifacts.md",
        ROOT / "docs" / "embedding_api.md",
        ROOT / "docs" / "migration.md",
        ROOT / "docs" / "models.md",
        ROOT / "examples" / "README.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "legacy encoder-only" not in " ".join(text.split()), path.relative_to(ROOT)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    seq2seq_section = readme.split("ANKH selects the encoder final state", maxsplit=1)[1].split(
        "### Safetensors output", maxsplit=1
    )[0]
    assert '"Synthyra/ANKH_base"' in seq2seq_section
    assert "local_files_only=True" not in seq2seq_section

    cards = {
        "ankh_base.md": "Synthyra/ANKH_base",
        "ankh_large.md": "Synthyra/ANKH_large",
        "ankh2_large.md": "Synthyra/ANKH2_large",
        "ankh3_large.md": "Synthyra/ANKH3_large",
        "ankh3_xl.md": "Synthyra/ANKH3_xl",
    }
    for filename, repo_id in cards.items():
        path = ROOT / "model_cards" / filename
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "legacy encoder-only" not in normalized, path.relative_to(ROOT)
        assert repo_id in text, path.relative_to(ROOT)
        assert "contains the complete ANKH encoder-decoder checkpoint" in normalized
        seq2seq_loads = re.findall(
            r"AutoModelForSeq2SeqLM\.from_pretrained\((.*?)\)",
            text,
            flags=re.DOTALL,
        )
        assert seq2seq_loads, path.relative_to(ROOT)
        for call in seq2seq_loads:
            assert repo_id in call or "repo_id" in call, path.relative_to(ROOT)


def test_examples_readme_indexes_every_entry_point_and_states_coverage_boundaries() -> None:
    text = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    entry_points = {
        path.name
        for path in (ROOT / "examples").glob("*.py")
        if path.name not in {"__init__.py", "_runtime.py"}
    }
    assert entry_points
    assert not {name for name in entry_points if f"`{name}`" not in text}
    for required in (
        "## Embedding coverage matrix",
        "base weights + untrained task head",
        "LoRA is the demonstrated PEFT method",
        "arbitrary `Dataset.save_to_disk()` trees are not",
        "--device cpu|cuda[:index]",
        "--dtype float32|bfloat16",
    ):
        assert required in text


def test_generated_cards_put_installation_before_hub_quick_start() -> None:
    for path in sorted((ROOT / "model_cards").glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        assert text.index("## Install and platform requirements") < text.index("## Quick start")
        assert "resolve/main/requirements.txt" in text
        assert "fastplms @ git+" not in text
        assert "implementation itself is embedded in the model repository" in text

    for name in (
        "boltz2.md",
        "esmfold.md",
        "esmfold2.md",
        "esmfold2_fast.md",
        "esmfold2_experimental_cutoff2025.md",
        "esmfold2_experimental_fast_cutoff2025.md",
    ):
        text = (ROOT / "model_cards" / name).read_text(encoding="utf-8")
        assert "exact NVIDIA GH200 on Linux aarch64" in text
        assert "validated release target is Linux x86-64" not in text


def test_esmfold2_cards_match_checkpoint_specific_msa_contracts() -> None:
    generic_msa_claim = "typed interface also supports RNA, protein MSAs"
    for name in (
        "esmfold2_fast.md",
        "esmfold2_experimental_fast_cutoff2025.md",
    ):
        path = ROOT / "model_cards" / name
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "24-block" in normalized or "24 folding blocks" in normalized
        assert "trained without MSA conditioning" in normalized
        assert "ProteinInput.msa" in normalized
        assert "MSA-derived" in normalized and "reject" in normalized
        assert "multichain" in normalized and "multimolecule" in normalized
        assert "msa=None" in normalized
        assert generic_msa_claim not in normalized

    for name in (
        "esmfold2.md",
        "esmfold2_experimental_cutoff2025.md",
    ):
        path = ROOT / "model_cards" / name
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        assert "48-block" in normalized or "48 folding blocks" in normalized
        assert "optional MSA" in normalized


def test_esmc_cards_disclose_supported_divergence_without_fabricated_metrics() -> None:
    for name in ("esmc_small.md", "esmc_large.md", "esmc_6b.md"):
        text = (ROOT / "model_cards" / name).read_text(encoding="utf-8")
        assert "Supported, numerically divergent" in text
        assert "Pending release measurement" in text
        assert "SDPA is the default" in text


def test_esmc_release_operations_live_in_docs_not_model_cards() -> None:
    card_names = (
        "esmc_small.md",
        "esmc_large.md",
        "esmc_6b.md",
        "esmfold2.md",
        "esmfold2_fast.md",
        "esmfold2_experimental_cutoff2025.md",
        "esmfold2_experimental_fast_cutoff2025.md",
    )
    for name in card_names:
        text = (ROOT / "model_cards" / name).read_text(encoding="utf-8")
        assert "Locked oracle package compatibility exception" not in text
        assert "nvidia-cusparselt-cu13" not in text
        assert "/docs/attention_backends.md" in text
        assert "/docs/generated/capability_evidence.md" in text

    evidence = (ROOT / "docs/generated/capability_evidence.md").read_text(
        encoding="utf-8"
    )
    assert "Locked oracle package compatibility exception" in evidence
    assert "nvidia-cusparselt-cu13==0.8.1" in evidence


def test_dplm_cards_mark_apache_weights_redistributable() -> None:
    for name in (
        "dplm_150m.md",
        "dplm_650m.md",
        "dplm_3b.md",
        "dplm2_150m.md",
        "dplm2_650m.md",
        "dplm2_3b.md",
    ):
        text = (ROOT / "model_cards" / name).read_text(encoding="utf-8")
        assert 'license: "apache-2.0"' in text
        assert "Weight license status: `resolved`" in text
        assert "Redistributable: `true`" in text
        assert "/bytedance/dplm/blob/main/LICENSE" in text
        assert "/README.md#overview" in text


def test_esmfold2_cards_disclose_race_safe_pickle_boundary() -> None:
    for name in (
        "esmfold2.md",
        "esmfold2_fast.md",
        "esmfold2_experimental_cutoff2025.md",
        "esmfold2_experimental_fast_cutoff2025.md",
    ):
        text = (ROOT / "model_cards" / name).read_text(encoding="utf-8")
        assert "`cache_dir`" in text
        assert "symlinks are rejected" in text
        assert "private temporary snapshot" in text
        assert "path-replacement and in-place source-write races" in text


def test_artifact_docs_describe_private_verified_runtime_bridge() -> None:
    text = (ROOT / "docs" / "artifacts.md").read_text(encoding="utf-8")
    assert "loader-owned private `TemporaryDirectory`" in text
    assert "re-hashes the exact extracted inventory" in text
    assert "rejects symlinks, bytecode, non-file entries" in text
    assert "hash-named directory in the Transformers module cache" not in text


def test_finetuning_docs_pin_inputs_and_describe_verified_final_artifact() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "finetuning.md").read_text(encoding="utf-8")
    for text in (readme, guide):
        assert "--model-revision 185ecbd45665d050a8dae326d91886d330c5f9d0" in text
        assert "--classification-dataset-revision " in text
        assert "7e18f1b98859b0a3e3da283f63d0a153b774cf1f" in text
    for flag, revision in (
        (
            "--regression-train-dataset-revision",
            "f4a51e5e9f2c2a0185693f9fbcffc02d9dae08db",
        ),
        (
            "--regression-validation-dataset-revision",
            "826ccfb1488d52b7b361802fbde161373247d084",
        ),
        (
            "--regression-test-dataset-revision",
            "4e22f014745728fca2d9c10f2f2cfd5a29a4981c",
        ),
    ):
        assert flag in guide
        assert revision in guide
    assert "`ordered_rows_sha256`" in guide
    assert "`reload_verified: true`" in guide
    assert "first `min(2, len(test_dataset))` rows" in guide
    assert "remains on the original\nprepared Trainer" in guide
    assert "Before promotion, supplement" not in guide


def test_documentation_local_links_resolve() -> None:
    failures: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            target = _local_link_target(path, match.group("target"))
            if target is not None and not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: missing link target "
                    f"{target.relative_to(ROOT)}"
                )
    assert not failures, "\n" + "\n".join(failures)


def test_python_documentation_fences_compile() -> None:
    failures: list[str] = []
    count = 0
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in FENCE_PATTERN.finditer(text):
            if match.group("language").lower() not in {"python", "py"}:
                continue
            count += 1
            line = text.count("\n", 0, match.start("body")) + 1
            try:
                ast.parse(match.group("body"), filename=f"{path}:{line}")
            except SyntaxError as error:
                failures.append(f"{path.relative_to(ROOT)}:{line}: {error.msg}")
    assert count > 0, "No executable Python documentation snippets were found."
    assert not failures, "\n" + "\n".join(failures)


def test_readme_embedding_snippet_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastplms

    observed: dict[str, object] = {}

    def fake_embed_dataset(model: object, inputs: object, **kwargs: Any) -> object:
        observed.update(model=model, inputs=inputs, kwargs=kwargs)
        return object()

    monkeypatch.setattr(fastplms, "embed_dataset", fake_embed_dataset)
    namespace = {"model": object()}
    snippet = _python_snippet(ROOT / "README.md", "EmbeddingInput, embed_dataset")
    exec(compile(snippet, "README.md", "exec"), namespace)

    inputs = observed["inputs"]
    assert [record.id for record in inputs] == ["protein-a", "protein-a"]
    assert observed["kwargs"] == {
        "batch_size": 2,
        "pooling": ("mean", "std"),
        "output": "embeddings",
    }


def test_readme_automodel_snippet_executes_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    observed: dict[str, object] = {}

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> Self:
            observed.update(model_id=model_id, kwargs=kwargs)
            return cls()

        def eval(self) -> Self:
            return self

    monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)
    snippet = _python_snippet(ROOT / "README.md", 'attn_implementation="sdpa"')
    exec(compile(snippet, "README.md", "exec"), {})

    assert observed == {
        "model_id": "Synthyra/ESM2-150M",
        "kwargs": {
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        },
    }


def test_documentation_does_not_make_unbacked_equivalence_or_speed_claims() -> None:
    failures: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for pattern in UNBACKED_CLAIM_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: unbacked claim {match.group()!r}"
                )
    assert not failures, "\n" + "\n".join(failures)
