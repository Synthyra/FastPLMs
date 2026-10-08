"""A record of which feature parts a full verification has passed, so a later reader can skip it.

Verifying a part hashes its bytes and loads its tensors, which costs one pass over the file. A store
of hundreds of gigabytes read by many processes would otherwise pay that pass in every process, on
every open. A receipt keeps the result for one feature directory: for each part, the digests the
commit marker states, and the size and modification time of the part file and of its row-identity
sidecar when they were verified. A reader trusts a part only when the marker states the same digests
and both files still have the recorded size and modification time. Any other part is verified in
full.

A receipt is a cache of one machine's verification, never a proof: a file rewritten to the same size
and the same modification time passes it. ``FeatureReader.verify`` on a reader built with
``trust_receipt=False`` hashes every byte and refreshes the receipt, and is the check for that case.
"""

from __future__ import annotations

import json
import os
import threading
import warnings

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from .digests import json_sha256
from .store import PART_TEMPLATE, SEGMENTS_DIRECTORY


SCHEMA = "feature_part_receipt_v1"


class PartReceipt:
    """The verified parts of one feature directory, read from and saved to one JSON file."""

    def __init__(
        self, path: str | Path, directory: Path, spec_payload: Mapping[str, Any],
        *, trust: bool = True,
    ) -> None:
        self.path = Path(path)
        self._directory = directory
        self._feature = spec_payload.get("key")
        self._descriptor_sha256 = json_sha256(spec_payload)
        self._trusted = self._load() if trust else {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, Any]]:
        """The recorded parts, or none without a receipt or when it names another descriptor."""
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            warnings.warn(
                f"Ignoring the unreadable verification receipt {self.path}: {error}",
                RuntimeWarning, stacklevel=3,
            )
            return {}
        if (not isinstance(document, dict) or document.get("schema") != SCHEMA
                or document.get("descriptor_sha256") != self._descriptor_sha256
                or not isinstance(document.get("parts"), dict)):
            return {}
        return document["parts"]

    def describe(self, segment: str, number: int, committed: Mapping[str, Any]) -> dict[str, Any]:
        """What verifying this part vouches for: the marker's digests and the files' stat."""
        base = self._directory / SEGMENTS_DIRECTORY / segment
        part = (base / PART_TEMPLATE.format(number)).stat()
        sidecar = committed.get("row_metadata")
        rows = None
        if isinstance(sidecar, dict):
            stat = (base / str(sidecar["file"])).stat()
            rows = {
                "file": sidecar["file"], "sha256": sidecar["sha256"],
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            }
        return {
            "sha256": committed["sha256"], "size": part.st_size, "mtime_ns": part.st_mtime_ns,
            "rows": rows,
        }

    def trusts(self, segment: str, number: int, described: Mapping[str, Any]) -> bool:
        """Whether an earlier verification passed these digests, sizes and modification times."""
        return self._trusted.get(f"{segment}/{number}") == described

    def record(self, segment: str, number: int, described: Mapping[str, Any]) -> None:
        with self._lock:
            self._pending[f"{segment}/{number}"] = dict(described)

    def save(self) -> None:
        """Merge the parts verified since the last save into the file, atomically.

        Concurrent savers can drop each other's entries, which only costs a later verification. A
        location that cannot be written warns, and this process keeps what it verified in memory.
        """
        with self._lock:
            if not self._pending:
                return
            document = {
                "schema": SCHEMA, "feature": self._feature,
                "descriptor_sha256": self._descriptor_sha256,
                "parts": {**self._load(), **self._pending},
            }
            owner = f"{os.getpid()}.{threading.get_ident()}"
            temporary = self.path.with_name(f"{self.path.name}.{owner}.writing")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
                temporary.replace(self.path)
            except OSError as error:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
                warnings.warn(
                    f"Could not save the verification receipt {self.path}: {error}. "
                    "Every open of this feature verifies its parts again.",
                    RuntimeWarning, stacklevel=2,
                )
            self._trusted = {**self._trusted, **self._pending}
            self._pending.clear()


__all__ = ["SCHEMA", "PartReceipt"]
