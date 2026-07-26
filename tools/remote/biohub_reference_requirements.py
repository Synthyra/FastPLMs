"""Extract Biohub ESM dependencies without resolving its mutable Transformers URL."""

from __future__ import annotations

import argparse
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path


_PINNED_TRANSFORMERS_REQUIREMENT = (
    "transformers @ git+https://github.com/Biohub/transformers.git@main"
)
_SAFE_PEP508_SUBSET = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"
    r"(?:"
    r"(?:===|==|!=|~=|<=|>=|<|>)[A-Za-z0-9][A-Za-z0-9.*+!_-]*"
    r"(?:,(?:===|==|!=|~=|<=|>=|<|>)[A-Za-z0-9][A-Za-z0-9.*+!_-]*)*"
    r")?"
)


class BiohubReferenceRequirementsError(RuntimeError):
    """The pinned Biohub ESM dependency contract is missing or has drifted."""


def _canonical_name(requirement: str) -> str:
    name = re.split(r"[<>=!~ @\[]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def extract_biohub_reference_requirements(pyproject: Path) -> tuple[str, ...]:
    """Return every pinned Biohub ESM dependency except its mutable Transformers URL."""

    try:
        raw = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BiohubReferenceRequirementsError(
            f"Unable to read the pinned Biohub ESM pyproject: {pyproject}"
        ) from error
    project = raw.get("project")
    dependencies = project.get("dependencies") if isinstance(project, dict) else None
    if not isinstance(dependencies, list) or not all(
        isinstance(requirement, str) and requirement.strip() for requirement in dependencies
    ):
        raise BiohubReferenceRequirementsError(
            "Pinned Biohub ESM pyproject must declare a non-empty string dependency list."
        )

    transformer_requirements = [
        requirement
        for requirement in dependencies
        if _canonical_name(requirement) == "transformers"
    ]
    if transformer_requirements != [_PINNED_TRANSFORMERS_REQUIREMENT]:
        raise BiohubReferenceRequirementsError(
            "Pinned Biohub ESM must declare exactly its known mutable Transformers main URL; "
            f"received {transformer_requirements!r}."
        )

    filtered = tuple(
        requirement
        for requirement in dependencies
        if requirement != _PINNED_TRANSFORMERS_REQUIREMENT
    )
    unsafe_requirements = [
        requirement
        for requirement in filtered
        if _SAFE_PEP508_SUBSET.fullmatch(requirement) is None
    ]
    if unsafe_requirements:
        raise BiohubReferenceRequirementsError(
            "Biohub ESM contains a dependency outside the allowed PEP 508 subset: "
            f"{unsafe_requirements!r}."
        )
    return filtered


def write_biohub_reference_requirements(pyproject: Path, output: Path) -> tuple[str, ...]:
    """Write the filtered dependency set atomically for a reference-image build."""

    requirements = extract_biohub_reference_requirements(pyproject)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{requirement}\n" for requirement in requirements),
        encoding="utf-8",
    )
    temporary.replace(output)
    return requirements


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render the fail-closed Biohub ESM non-Transformers requirement file."""

    arguments = _parser().parse_args(argv)
    requirements = write_biohub_reference_requirements(arguments.pyproject, arguments.output)
    print(f"Wrote {len(requirements)} pinned Biohub ESM non-Transformers requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
