"""Static container-boundary checks that do not require a Docker daemon."""

from __future__ import annotations

import re
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


def test_manifest_reference_containers_are_build_targets() -> None:
    bake = BAKE_FILE.read_text(encoding="utf-8")
    targets = set(re.findall(r'^target\s+"([^"]+)"', bake, flags=re.MULTILINE))
    expected = {spec.family.reference_container for spec in get_model_registry().values()}
    assert expected.issubset(targets), f"Missing reference targets: {sorted(expected - targets)}"


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
    assert "gpus: all" in compose
    assert "HF_HOME: /cache/huggingface" in compose
    assert "TORCH_HOME: /cache/torch" in compose
    for volume in ("hf", "torch", "xdg"):
        assert f"name: fastplms-{volume}-cache" in compose


def test_dependency_tool_caches_match_their_buildkit_mounts() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "UV_CACHE_DIR=/root/.cache/uv" in dockerfile
    assert "PIP_CACHE_DIR=/root/.cache/pip" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/uv" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile


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


def test_reference_protocol_contains_the_isolated_esmfold2_bundle_producer() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    protocol = dockerfile.split("FROM scratch AS reference-protocol", maxsplit=1)[1].split(
        "FROM python310-reference AS reference-ankh",
        maxsplit=1,
    )[0]
    assert "tests/structure/support/esmfold2_bundle.py" in protocol
    assert "src/fastplms" not in protocol


def test_derived_candidate_stages_sync_from_the_project_directory() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    for stage, next_stage in (
        ("candidate-structure", "candidate-fp8"),
        ("candidate-fp8", "candidate-artifact"),
    ):
        section = dockerfile.split(f"AS {stage}", maxsplit=1)[1].split(
            f" AS {next_stage}",
            maxsplit=1,
        )[0]
        assert section.index("WORKDIR /opt/fastplms") < section.index("uv sync --frozen")
        assert section.rindex("WORKDIR /workspace") > section.index("uv sync --frozen")


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
    assert set(name for name in sections if name.startswith("runtime")) == {"runtime"}
    runtime = sections["runtime"]
    assert "ARG FASTPLMS_RUNTIME_PROFILE=core" in runtime
    assert "core)" in runtime
    assert "esmfold2-fp8)" in runtime
    assert "--extra structure --extra fp8" in runtime
    assert "Unsupported FastPLMs runtime profile" in runtime
    assert "exit 64" in runtime

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


def test_kernel_lock_is_available_to_source_and_artifact_images() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert dockerfile.count("COPY pyproject.toml uv.lock kernels.lock README.md LICENSE ./") == 2
    assert "!kernels.lock" in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
