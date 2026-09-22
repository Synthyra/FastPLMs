"""Render shared manifest metadata for support tables and checkpoint cards."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable

from fastplms.registry import ModelFamily, ModelSpec
from tools.artifacts.doc_generation.capabilities import (
    AUTO_CLASS_STATUS,
)
from tools.artifacts.license_metadata import render_checkpoint_terms

GENERATED_MARKER = "<!-- Generated from src/fastplms/models.toml. Do not edit. -->"


def _code(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def _table_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _precision_contract(family: ModelFamily, spec: ModelSpec | None = None) -> str:
    experimental = set(family.experimental_precisions)
    return ", ".join(
        f"`{value}` (experimental)" if value in experimental else f"`{value}`"
        for value in family.precisions
        if spec is None
        or value != "fp8"
        or (spec.backbone_model is None and spec.id not in {"esmc_small", "esmc_large"})
    )


def _hub_license_label(family: ModelFamily) -> str:
    label = f"`{family.hub_license}`"
    if family.hub_license == "other":
        label += f" ({render_checkpoint_terms(family)})"
    return label


def _tokenizer_class_label(family: ModelFamily) -> str:
    if family.tokenizer_class is None:
        return "`n/a`"
    return f"`{family.tokenizer_class}`"


def _auto_class_status(family: ModelFamily, auto_class: str) -> str:
    """Describe whether an advertised entry point has trained checkpoint state."""

    if family.id == "ankh" and auto_class == "AutoModelForMaskedLM":
        return "FastPLMs extension"
    try:
        return AUTO_CLASS_STATUS[auto_class]
    except KeyError as error:
        raise ValueError(f"No model-card weight status is defined for {auto_class!r}.") from error


def _platform_requirements(family: ModelFamily) -> str:
    paragraphs = ["This model requires Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13."]
    if family.tokenizer_mode == "structure":
        paragraphs.extend(
            (
                "The artifact requirements include the structure dependencies.",
                "Validation runs in Docker on any compatible CUDA device. Record the "
                "container, hardware, precision, and inputs; no GPU product or "
                "workstation is required.",
            )
        )
    elif any(name.startswith("flash_attention_") for name in family.attention):
        paragraphs.append(
            "The artifact requirements include the FlashAttention loader dependency. "
            "FlashAttention also requires compatible CUDA hardware and BF16 execution."
        )
    else:
        paragraphs.append(
            "The CPU gate covers small offline tests. Published checkpoint throughput "
            "and parity require the documented device tier."
        )
    return "\n\n".join(
        textwrap.fill(
            paragraph,
            width=79,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for paragraph in paragraphs
    )


def _installation_section(spec: ModelSpec) -> str:
    return f"""\
## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \\
  "https://huggingface.co/{spec.fast.repo_id}/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository.
Transformers loads it through `trust_remote_code=True`.

{_platform_requirements(spec.family)}

The Hub quick start needs network access for the first download. For an
air-gapped run, build the manifest-pinned local artifact first and use the
offline example.

"""
