"""Download and hash the native fair-esm parity-oracle assets.

This resolver reads only Meta's public fair-esm asset host. Files are written
transactionally to a caller-selected cache and the resulting URL, relative
path, byte size, and SHA-256 identity are emitted as JSON for manifest review.
It never uploads content or authenticates to a remote service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


_HOST = "dl.fbaipublicfiles.com"
_ROOT = f"https://{_HOST}/fair-esm"
_MODEL_NAMES = (
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
)


@dataclass(frozen=True)
class ResolvedAsset:
    model: str
    role: str
    path: str
    url: str
    sha256: str
    size: int


def _candidates() -> tuple[tuple[str, str, str, str], ...]:
    assets: list[tuple[str, str, str, str]] = []
    for model in _MODEL_NAMES:
        assets.extend(
            (
                (model, "weights", f"models/{model}.pt", f"{_ROOT}/models/{model}.pt"),
                (
                    model,
                    "contact_regression",
                    f"regression/{model}-contact-regression.pt",
                    f"{_ROOT}/regression/{model}-contact-regression.pt",
                ),
            )
        )
    assets.append(
        (
            "esmfold_3B_v1",
            "weights",
            "models/esmfold_3B_v1.pt",
            f"{_ROOT}/models/esmfold_3B_v1.pt",
        )
    )
    return tuple(assets)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _HOST:
        raise RuntimeError(f"Refusing non-fair-esm asset URL: {url}")


def _download_one(
    candidate: tuple[str, str, str, str],
    cache: Path,
) -> ResolvedAsset:
    model, role, relative_name, url = candidate
    _validate_url(url)
    relative = PurePosixPath(relative_name)
    target = cache.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    if not target.is_file():
        offset = partial.stat().st_size if partial.is_file() else 0
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={offset}-"} if offset else {},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            _validate_url(response.geturl())
            status = getattr(response, "status", 200)
            append = offset > 0 and status == 206
            if offset > 0 and not append:
                offset = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                while chunk := response.read(8 * 1024**2):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        partial.replace(target)
    return ResolvedAsset(
        model=model,
        role=role,
        path=relative.as_posix(),
        url=url,
        sha256=_sha256(target),
        size=target.stat().st_size,
    )


def resolve(cache: Path, jobs: int) -> list[ResolvedAsset]:
    """Resolve all supported ESM2 native oracle assets."""

    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        assets = list(pool.map(lambda item: _download_one(item, cache), _candidates()))
    return sorted(assets, key=lambda item: (item.model, item.role))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least one")
    document = [asdict(item) for item in resolve(args.cache, args.jobs)]
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
