"""Repository-wide policy against source-built FlashAttention packages."""

from __future__ import annotations

import re
from pathlib import Path

from fastplms.registry import get_model_registry


ROOT = Path(__file__).resolve().parents[2]
_FLASH_PACKAGES = frozenset({"flash-attn", "flash_attn", "flashattention"})
_FLASH_BACKENDS = frozenset({"flash_attention_2", "flash_attention_3"})
_SOURCE_FLASH = re.compile(r"flash[-_]?attn|dao-ai(?:lab)?/flash-attention", re.IGNORECASE)
_INSTALL_OR_BUILD = re.compile(
    r"(?:^|\s)(?:pip(?:3)?\s+install|python(?:3)?\s+-m\s+pip\s+install|"
    r"uv(?:\s+pip)?\s+(?:add|install)|poetry\s+add|conda\s+install|mamba\s+install|"
    r"git\s+clone|python(?:3)?\s+setup\.py|cmake(?:\s|$)|ninja(?:\s|$)|make(?:\s|$))",
    re.IGNORECASE,
)


def _normalized_package(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[\s]", requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_dependency_contract_contains_no_flash_attn_distribution() -> None:
    requirement_root = ROOT / "requirements"
    dependency_files = sorted(requirement_root.rglob("*.in"))
    dependency_files.extend(sorted(requirement_root.rglob("*.txt")))
    requirements = [
        requirement
        for path in dependency_files
        for requirement in _requirements(path)
    ]

    assert not {
        requirement
        for requirement in requirements
        if _normalized_package(requirement) in _FLASH_PACKAGES
    }
    assert _requirements(requirement_root / "features" / "flash.in") == [
        "kernels>=0.15,<0.16"
    ]


def test_no_docker_script_or_documentation_command_builds_source_flash_attn() -> None:
    roots = (
        ROOT / "requirements",
        ROOT / "docker",
        ROOT / "tools",
        ROOT / "examples",
        ROOT / "benchmarks",
        ROOT / "docs",
    )
    files = [ROOT / "README.md"]
    allowed_suffixes = {
        ".bat",
        ".cmd",
        ".hcl",
        ".in",
        ".md",
        ".ps1",
        ".py",
        ".rst",
        ".sh",
        ".txt",
        ".yml",
        ".yaml",
    }
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and (path.suffix.lower() in allowed_suffixes or path.name == "Dockerfile")
        )

    violations: list[str] = []
    for path in sorted(set(files)):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if (
                path.suffix.lower() in {".in", ".txt"}
                and stripped
                and not stripped.startswith("#")
                and _normalized_package(stripped) in _FLASH_PACKAGES
            ):
                violations.append(f"{path.relative_to(ROOT)}:{number}: {stripped}")
            # Inspect a short logical-command window so shell and Docker line
            # continuations cannot hide a source package or repository.
            command = " ".join(lines[number - 1 : number + 3])
            if _SOURCE_FLASH.search(command) and _INSTALL_OR_BUILD.search(command):
                violations.append(f"{path.relative_to(ROOT)}:{number}: {stripped}")
    assert not violations, "Source FlashAttention install/build commands:\n  - " + "\n  - ".join(
        violations
    )


def test_fastplms_10_manifest_advertises_only_pinned_flash_backends() -> None:
    registry = get_model_registry()
    advertised = {
        family.id: sorted(set(family.attention).intersection(_FLASH_BACKENDS))
        for family in registry.families.values()
        if set(family.attention).intersection(_FLASH_BACKENDS)
    }
    assert advertised == {
        "dplm": ["flash_attention_3"],
        "esm2": ["flash_attention_2", "flash_attention_3"],
        "esm_plusplus": ["flash_attention_2", "flash_attention_3"],
    }
    assert {
        implementation: kernel.dtypes
        for implementation, kernel in registry.attention_kernels.items()
    } == {
        "flash_attention_2": ("bfloat16",),
        "flash_attention_3": ("bfloat16",),
    }
    for family_id, backends in advertised.items():
        for backend in backends:
            assert registry.supported_attention_dtypes(family_id, backend) == ("bfloat16",)

    documentation = (ROOT / "docs" / "attention_backends.md").read_text(encoding="utf-8")
    normalized_documentation = " ".join(documentation.split())
    assert "Both pinned FlashAttention kernels are BF16-only." in normalized_documentation
    assert (
        "Direct FP32 and FP16 calls raise before kernel loading."
        in normalized_documentation
    )
