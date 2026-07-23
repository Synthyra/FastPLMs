"""Static container-boundary checks that do not require a Docker daemon."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from fastplms.registry import get_model_registry

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
BAKE_FILE = ROOT / "docker" / "docker-bake.hcl"
COMPOSE_FILE = ROOT / "docker" / "compose.yaml"
DOCKERIGNORE = ROOT / ".dockerignore"


def _stages(text: str) -> dict[str, str]:
    return {
        name: base
        for base, name in re.findall(
            r"^FROM\s+(\S+)\s+AS\s+(\S+)\s*$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    }


def _stage_section(text: str, stage: str) -> str:
    marker = re.compile(
        rf"^FROM\s+\S+\s+AS\s+{re.escape(stage)}\s*$",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    match = marker.search(text)
    assert match is not None, f"Missing Docker stage {stage!r}"
    next_stage = re.search(r"^FROM\s+", text[match.end() :], flags=re.MULTILINE)
    end = match.end() + next_stage.start() if next_stage is not None else len(text)
    return text[match.start() : end]


def test_manifest_reference_containers_are_build_targets() -> None:
    bake = BAKE_FILE.read_text(encoding="utf-8")
    targets = set(re.findall(r'^target\s+"([^"]+)"', bake, flags=re.MULTILINE))
    expected = {spec.family.reference_container for spec in get_model_registry().values()}
    assert expected.issubset(targets), f"Missing reference targets: {sorted(expected - targets)}"


def test_bake_defaults_to_native_host_platform_without_an_amd64_override() -> None:
    bake = BAKE_FILE.read_text(encoding="utf-8")
    common = bake.split('target "common" {', maxsplit=1)[1].split("\n}", maxsplit=1)[0]
    assert "platforms" not in common
    assert "linux/amd64" not in bake


def test_reference_stages_do_not_inherit_candidate_or_runtime_layers() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    stages = _stages(dockerfile)
    reference_stages = {name for name in stages if name.startswith("reference-")}
    assert reference_stages
    for stage in reference_stages:
        ancestry: list[str] = []
        current = stage
        while current in stages:
            base = stages[current]
            ancestry.append(base)
            if base not in stages:
                break
            current = base
        candidate_stages = {
            "source",
            "runtime",
            "candidate",
            "candidate-structure",
            "candidate-fp8",
        }
        assert not candidate_stages.intersection(ancestry), (
            f"{stage} inherits candidate/runtime layers: {ancestry}"
        )


def test_reference_stages_copy_notices_and_no_checkpoint_assets() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    for section in re.split(r"(?=^FROM\s+)", dockerfile, flags=re.MULTILINE):
        match = re.match(r"FROM\s+\S+\s+AS\s+(reference-\S+)", section)
        if match is None:
            continue
        stage = match.group(1)
        if stage in {"reference-protocol", "reference-esmfold2"}:
            # The protocol is copied into notice-bearing final stages; ESMFold2
            # inherits the notice-bearing Biohub reference stage.
            continue
        assert "THIRD_PARTY_NOTICES.md" in section, f"{stage} omits required notices"
        copied_sources = re.findall(r"^COPY\s+(\S+)", section, flags=re.MULTILINE)
        weight_suffixes = (".bin", ".ckpt", ".pt", ".pth", ".safetensors")
        assert not any(source.endswith(weight_suffixes) for source in copied_sources), (
            f"{stage} copies checkpoint weights"
        )


def test_reference_stages_copy_distribution_licenses_for_each_source_context() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    registry = get_model_registry()
    context_to_source = {
        source.id.replace("-", "_"): source.id for source in registry.upstreams.values()
    }
    checked: set[str] = set()
    for section in re.split(r"(?=^FROM\s+)", dockerfile, flags=re.MULTILINE):
        match = re.match(r"FROM\s+\S+\s+AS\s+(reference-\S+)", section)
        if match is None:
            continue
        for context in re.findall(r"--from=upstream_([a-z0-9_]+)", section):
            source_id = context_to_source[context]
            assert f"COPY LICENSES/{source_id} /licenses/{source_id}" in section, (
                f"{match.group(1)} omits distribution licenses for {source_id}"
            )
            checked.add(source_id)
    assert checked == set(registry.upstreams)


def test_compose_centralizes_gpu_and_ipc_configuration() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    assert "ipc: host" in compose
    assert "driver: nvidia" in compose
    assert "count: all" in compose
    assert "capabilities: [gpu]" in compose
    assert "HF_HOME: /cache/huggingface" in compose
    assert "TORCH_HOME: /cache/torch" in compose
    assert compose.count('CUBLAS_WORKSPACE_CONFIG: ":4096:8"') == 2
    for volume in ("hf", "torch", "xdg"):
        assert f"name: fastplms-{volume}-cache" in compose


def test_dependency_tool_caches_match_their_buildkit_mounts() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "UV_CACHE_DIR=/root/.cache/uv" in dockerfile
    assert "PIP_CACHE_DIR=/root/.cache/pip" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/uv" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile


def test_reference_esm2_installs_oracle_runtime_dependencies() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    section = dockerfile.split("FROM python312 AS reference-esm2", maxsplit=1)[1].split(
        "FROM python310-reference AS reference-boltz2", maxsplit=1
    )[0]
    assert "huggingface-hub==0.36.2" in section
    assert "numpy==1.26.4" in section


def test_reference_services_receive_only_the_exchange_and_cache_mounts() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    reference_anchor = compose.split("x-reference: &reference", maxsplit=1)[1].split(
        "services:", maxsplit=1
    )[0]
    assert "../artifacts/reference:/exchange" in reference_anchor
    assert "..:/workspace" not in reference_anchor
    for service in (
        "reference-ankh",
        "reference-biohub-esm",
        "reference-boltz2",
        "reference-dplm",
        "reference-e1",
        "reference-esm2",
        "reference-esmfold",
        "reference-esmfold2",
        "reference-protein-ttt",
    ):
        start = compose.index(f"  {service}:")
        section = compose[start : start + 240]
        assert "<<: *reference" in section


def test_compose_and_bake_reference_contexts_are_synchronized() -> None:
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    bake = BAKE_FILE.read_text(encoding="utf-8")
    expected = {
        "reference-ankh": {"upstream_ankh": "vendor/upstream/ankh"},
        "reference-biohub-esm": {
            "upstream_biohub_esm": "vendor/upstream/biohub-esm",
            "upstream_biohub_transformers": "vendor/upstream/biohub-transformers",
        },
        "reference-boltz2": {"upstream_boltz": "vendor/upstream/boltz"},
        "reference-dplm": {"upstream_dplm": "vendor/upstream/dplm"},
        "reference-e1": {"upstream_e1": "vendor/upstream/e1"},
        "reference-esm2": {"upstream_fair_esm": "vendor/upstream/fair-esm"},
        "reference-esmfold": {
            "upstream_fair_esm": "vendor/upstream/fair-esm",
            "upstream_openfold": "vendor/upstream/openfold",
        },
        "reference-esmfold2": {
            "upstream_biohub_esm": "vendor/upstream/biohub-esm",
            "upstream_biohub_transformers": "vendor/upstream/biohub-transformers",
        },
        "reference-protein-ttt": {
            "upstream_protein_ttt": "vendor/upstream/protein-ttt"
        },
    }

    for service, contexts in expected.items():
        compose_tail = compose.split(f"  {service}:\n", maxsplit=1)[1]
        next_service = re.search(r"\n  [^\s][^:\n]*:\n", compose_tail)
        compose_section = (
            compose_tail[: next_service.start()] if next_service is not None else compose_tail
        )
        bake_section = bake.split(f'target "{service}" {{', maxsplit=1)[1].split(
            "\n}", maxsplit=1
        )[0]
        assert "additional_contexts:" in compose_section
        assert "contexts = {" in bake_section
        for name, relative_path in contexts.items():
            assert f"{name}: ../{relative_path}" in compose_section
            assert re.search(
                rf"{re.escape(name)}\s*=\s*\"{re.escape(relative_path)}\"",
                bake_section,
            )


def test_reference_protocol_contains_the_isolated_esmfold2_bundle_producer() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    protocol = dockerfile.split("FROM scratch AS reference-protocol", maxsplit=1)[1].split(
        "FROM python310-reference AS reference-ankh",
        maxsplit=1,
    )[0]
    assert "tests/structure/support/esmfold2_bundle.py" in protocol


def test_reference_protocol_only_copies_existing_repository_paths() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    protocol = _stage_section(dockerfile, "reference-protocol")
    copied_sources = re.findall(r"^COPY\s+(\S+)\s+\S+\s*$", protocol, flags=re.MULTILINE)
    assert copied_sources
    missing = [source for source in copied_sources if not (ROOT / source).exists()]
    assert missing == []
    assert "COPY tools/remote/__init__.py" not in protocol
    assert "tests/parity/support/semantic_config.py" in protocol
    assert "src/fastplms" not in protocol


def test_candidate_dependencies_are_cached_before_project_source() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    metadata = _stage_section(dockerfile, "project-metadata")
    assert "COPY pyproject.toml uv.lock kernels.lock ./" in metadata
    assert "COPY src" not in metadata
    assert "COPY tests" not in metadata

    for dependencies, final in (
        ("source-dependencies", "source"),
        ("runtime-dependencies", "runtime"),
        ("candidate-dependencies", "candidate"),
        ("candidate-structure-dependencies", "candidate-structure"),
        ("candidate-fp8-dependencies", "candidate-fp8"),
    ):
        dependency_section = _stage_section(dockerfile, dependencies)
        assert "uv sync --frozen" in dependency_section
        assert "--no-install-project" in dependency_section
        assert "COPY src" not in dependency_section
        assert "COPY tests" not in dependency_section

        final_section = _stage_section(dockerfile, final)
        source_copy = final_section.index("COPY src ./src")
        project_sync = final_section.index("uv sync --frozen", source_copy)
        assert "--no-install-project" not in final_section[project_sync:]


def test_reference_protocol_and_legal_text_do_not_invalidate_dependency_layers() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    for stage in _stages(dockerfile):
        if not stage.startswith("reference-") or stage in {
            "reference-protocol",
            "reference-esmfold2",
        }:
            continue
        section = _stage_section(dockerfile, stage)
        last_install = section.rfind("pip install")
        if last_install < 0:
            continue
        for late_copy in (
            "COPY --from=reference-protocol",
            "COPY THIRD_PARTY_NOTICES.md",
            "COPY LICENSES/",
        ):
            if late_copy in section:
                assert section.index(late_copy) > last_install, (
                    f"{stage} copies {late_copy!r} before its dependency layer"
                )


def test_runtime_is_one_fail_closed_parameterized_stage() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    sections = {
        name: section
        for name, section in re.findall(
            r"^FROM\s+\S+\s+AS\s+(\S+)\s*$([\s\S]*?)(?=^FROM\s+|\Z)",
            dockerfile,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    }
    assert set(name for name in sections if name.startswith("runtime")) == {
        "runtime-dependencies",
        "runtime",
    }
    for stage in ("runtime-dependencies", "runtime"):
        runtime = sections[stage]
        assert "ARG FASTPLMS_RUNTIME_PROFILE=core" in runtime
        assert "core)" in runtime
        assert "esmfold2-fp8)" in runtime
        assert "--extra structure --extra fp8" in runtime
        assert "Unsupported FastPLMs runtime profile" in runtime
        assert "exit 64" in runtime
    assert "--no-install-project" in sections["runtime-dependencies"]
    assert "--no-install-project" not in sections["runtime"]

    bake = BAKE_FILE.read_text(encoding="utf-8")
    runtime_targets = {
        name: section
        for name, section in re.findall(
            r'^target\s+"([^"]+)"\s*\{([\s\S]*?)^\}',
            bake,
            flags=re.MULTILINE,
        )
        if name in {"runtime", "runtime-fp8"}
    }
    assert set(runtime_targets) == {"runtime", "runtime-fp8"}
    assert all('target   = "runtime"' in section for section in runtime_targets.values())
    assert 'FASTPLMS_RUNTIME_PROFILE = "core"' in runtime_targets["runtime"]
    assert (
        'FASTPLMS_RUNTIME_PROFILE = "esmfold2-fp8"'
        in runtime_targets["runtime-fp8"]
    )


def test_fp8_dependency_is_confined_to_fp8_container_targets() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    sections = {
        name: section
        for name, section in re.findall(
            r"^FROM\s+\S+\s+AS\s+(\S+)\s*$([\s\S]*?)(?=^FROM\s+|\Z)",
            dockerfile,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    }
    assert "--extra fp8" in sections["runtime"]
    assert "--extra fp8" in sections["candidate-fp8"]
    for stage in ("candidate", "candidate-structure", "candidate-artifact"):
        assert "--extra fp8" not in sections[stage]


def test_cueq_dependency_is_confined_to_structure_validation_targets() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    sections = {
        name: section
        for name, section in re.findall(
            r"^FROM\s+\S+\s+AS\s+(\S+)\s*$([\s\S]*?)(?=^FROM\s+|\Z)",
            dockerfile,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    }
    for stage in ("candidate-structure", "candidate-fp8"):
        assert "--extra cueq" in sections[stage]
    for stage in ("runtime", "candidate", "candidate-artifact"):
        assert "--extra cueq" not in sections[stage]


def test_kernel_lock_is_available_to_source_and_artifact_images() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY pyproject.toml uv.lock kernels.lock ./" in _stage_section(
        dockerfile, "project-metadata"
    )
    assert "COPY pyproject.toml uv.lock kernels.lock README.md LICENSE ./" in _stage_section(
        dockerfile, "candidate-artifact"
    )
    assert "!kernels.lock" in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()


def test_candidate_validation_extra_supports_transformers_device_map() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = project["project"]["optional-dependencies"]["dev"]
    assert "accelerate>=1.10,<2" in dev

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "--extra dev" in _stage_section(dockerfile, "candidate-dependencies")
    assert "--extra dev" in _stage_section(dockerfile, "candidate")
