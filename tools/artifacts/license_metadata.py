"""Render validated Hugging Face model-card license metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlparse

from huggingface_hub import ModelCard

from fastplms.registry import HUB_LICENSE_IDENTIFIERS, ModelFamily

_HUB_LICENSE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")


def render_hub_license_yaml(family: ModelFamily) -> str:
    """Return deterministic YAML fields from the typed family contract."""

    return "\n".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in family.hub_license_metadata.items()
    )


def render_checkpoint_terms(family: ModelFamily) -> str:
    """Return precise Markdown for the checkpoint's governing terms."""

    if family.hub_license == "other":
        if family.hub_license_name is None or family.hub_license_link is None:
            raise ValueError("Custom Hub licenses require a name and link")
        return f"[{family.hub_license_name}]({family.hub_license_link})"
    return cast(str, family.checkpoint_license)


def validate_hub_license_metadata(metadata: Mapping[str, object]) -> dict[str, str]:
    """Validate and normalize one Hub license metadata mapping."""

    allowed_fields = {"license", "license_name", "license_link"}
    if not metadata or not set(metadata).issubset(allowed_fields):
        raise ValueError("Hub license metadata has missing or unknown fields")
    identifier = metadata.get("license")
    name = metadata.get("license_name")
    link = metadata.get("license_link")
    if not isinstance(identifier, str) or identifier not in HUB_LICENSE_IDENTIFIERS:
        raise ValueError("Model card has an unsupported Hugging Face license identifier")
    normalized = {"license": identifier}
    if identifier != "other":
        if name is not None or link is not None:
            raise ValueError("Standard Hub licenses may not define custom license fields")
        return normalized
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Custom Hub license is missing license_name")
    if _HUB_LICENSE_NAME_RE.fullmatch(name) is None:
        raise ValueError("Custom Hub license_name must be a lowercase Hub slug")
    if not isinstance(link, str) or not link.strip():
        raise ValueError("Custom Hub license is missing license_link")
    parsed_link = urlparse(link)
    if (
        parsed_link.scheme != "https"
        or not parsed_link.netloc
        or not parsed_link.path
        or parsed_link.username is not None
        or parsed_link.password is not None
    ):
        raise ValueError("Custom Hub license_link must be an absolute HTTPS URL")
    normalized["license_name"] = name
    normalized["license_link"] = link
    return normalized


def parse_hub_license_metadata(card_text: str) -> dict[str, str]:
    """Parse and validate a card's Hugging Face license fields."""

    try:
        data = ModelCard(card_text).data
    except Exception as error:
        raise ValueError(f"Invalid model-card metadata: {error}") from error
    metadata: dict[str, object] = {}
    for key in ("license", "license_name", "license_link"):
        value = getattr(data, key, None)
        if value is not None:
            metadata[key] = value
    return validate_hub_license_metadata(metadata)


__all__ = [
    "parse_hub_license_metadata",
    "render_checkpoint_terms",
    "render_hub_license_yaml",
    "validate_hub_license_metadata",
]
