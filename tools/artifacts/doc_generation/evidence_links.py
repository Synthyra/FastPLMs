"""Resolve declared evidence hyperlinks to immutable public dataset payloads."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from tools.artifacts.evidence_store import load_manifest


_LINK = re.compile(r"(?P<prefix>\]\()(?P<target>[^)\s]+)(?P<suffix>\))")
_FENCED_CODE = re.compile(r"(^```[^\n]*\n.*?^```[^\n]*$)", re.MULTILINE | re.DOTALL)
_GITHUB_SOURCE = "https://github.com/Synthyra/FastPLMs/blob/main/"


def _published_evidence_urls(root: Path) -> dict[str, str]:
    manifest = root / "evidence.toml"
    if not manifest.is_file():
        return {}
    store = load_manifest(manifest)
    if store.revision == "pending":
        return {}
    prefix = f"https://huggingface.co/datasets/{store.repository}/resolve/{store.revision}"
    return {entry.path: f"{prefix}/{quote(entry.path, safe='/')}" for entry in store.files}


def evidence_reference(relative: str, root: Path | None) -> str:
    """Render a recorded evidence path as a link once its dataset revision is pinned."""

    url = _published_evidence_urls(root).get(relative) if root is not None else None
    return f"[`{relative}`]({url})" if url is not None else f"`{relative}`"


def rewrite_evidence_links(markdown: str, *, root: Path, document_path: Path) -> str:
    """Rewrite only declared Markdown link targets, leaving code and other URLs intact."""

    published = _published_evidence_urls(root)
    if not published:
        return markdown
    root = root.resolve()
    document_path = document_path.resolve()

    def replace(match: re.Match[str]) -> str:
        target = match["target"]
        parsed = urlsplit(target)
        if target.startswith(_GITHUB_SOURCE):
            relative = unquote(urlsplit(target[len(_GITHUB_SOURCE) :]).path)
        else:
            if parsed.scheme or parsed.netloc or not parsed.path:
                return match[0]
            decoded = unquote(parsed.path)
            path = (
                root / decoded.lstrip("/")
                if decoded.startswith("/")
                else document_path.parent / decoded
            ).resolve()
            if not path.is_relative_to(root):
                return match[0]
            relative = path.relative_to(root).as_posix()
        url = published.get(relative)
        if url is None:
            return match[0]
        if parsed.query:
            url += "?" + parsed.query
        if parsed.fragment:
            url += "#" + parsed.fragment
        return match["prefix"] + url + match["suffix"]

    parts = _FENCED_CODE.split(markdown)
    return "".join(
        _LINK.sub(replace, part) if index % 2 == 0 else part for index, part in enumerate(parts)
    )
