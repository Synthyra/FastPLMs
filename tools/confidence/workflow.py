"""Execute resumable pilot stages inside the Modal worker image."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time

import requests

from html.parser import HTMLParser
from pathlib import Path


ARCHIVES = {
    "rcsb": "1TEH73v9oxA1oYYnsPZntHqES_04vEz8P",
    "rcsb_multimer": "1aN9zUL4JokQc0L6AWVUNlsnftQBi8pjr",
    "cameo_val": "10fhgH7nnVA022nvN-v3bTg1Xor97t2Ne",
    "rcsb_multimer_val": "17meo4uBvvFfB2M-uor17KWwqQDYdSGFI",
}
ARCHIVE_COPIES = {"rcsb_multimer": "1K-yAbtbFvSYTQ2q8PGhO4d7LHrU3KvrG"}
EXCLUDED_TEMPLATE_DIRECTORIES = frozenset({"template.lmdb", "template_mapping.lmdb"})


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class DriveConfirmation(HTMLParser):
    """Read Google's public large-file download confirmation form."""

    def __init__(self) -> None:
        super().__init__()
        self.action = ""
        self.fields: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self.action = attributes.get("action") or ""
        if tag == "input" and attributes.get("name"):
            self.fields[attributes["name"]] = attributes.get("value") or ""


def download_archive(drive_id: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    with requests.Session() as session:
        response = session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": drive_id},
            timeout=60,
            stream=True,
        )
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", ""):
            response.close()
            raise ValueError("Expected the official large-file download confirmation form")
        form = DriveConfirmation()
        page = response.text
        form.feed(page)
        response.close()
        if form.action != "https://drive.usercontent.google.com/download":
            title = re.search(r"<title>(.*?)</title>", page, flags=re.DOTALL)
            raise ValueError(
                "Google Drive did not provide its public download form; page title: "
                + (title.group(1) if title else "untitled")
            )
        if form.fields.get("id") != drive_id:
            raise ValueError("Google Drive confirmation identifies a different archive")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with session.get(
            form.action, params=form.fields, headers=headers, timeout=(60, 180), stream=True
        ) as content:
            content.raise_for_status()
            if "text/html" in content.headers.get("Content-Type", ""):
                page = content.text
                title = re.search(r"<title>(.*?)</title>", page, flags=re.DOTALL)
                diagnostic = " ".join(re.sub(r"<[^>]+>", " ", page).split())
                raise ValueError(
                    "Google Drive returned an HTML error instead of the archive: "
                    + (title.group(1) if title else "untitled")
                    + "; "
                    + diagnostic[-1200:]
                )
            append = offset > 0 and content.status_code == 206
            if append and not content.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise ValueError("Google Drive resumed the archive at an unexpected offset")
            with partial.open("ab" if append else "wb") as handle:
                written = offset if append else 0
                next_report = written + 1024**3
                for chunk in content.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    written += len(chunk)
                    if written >= next_report:
                        print(
                            f"{destination.name}: {written / 1024**3:.1f} GiB downloaded",
                            flush=True,
                        )
                        next_report = written + 1024**3
    partial.replace(destination)


def prepare_data(
    root: Path,
    phase: str = "download",
    source: str = "cameo_val",
    drive_id: str | None = None,
    candidate_multiplier: int = 4,
) -> dict:
    from .data import SelectionSpec, prepare, safe_extract_archive, write_records_json

    if phase == "prune_templates":
        dataset_root = (root / "atlasfold" / "rcsb_multimer").resolve()
        removed = []
        for name in sorted(EXCLUDED_TEMPLATE_DIRECTORIES):
            target = (dataset_root / name).resolve()
            if target.parent != dataset_root:
                raise ValueError("Template directory escapes the extracted dataset")
            if target.is_dir():
                shutil.rmtree(target)
                removed.append(name)
        return {"status": "complete", "removed_template_directories": removed}

    if phase == "download":
        if source not in ARCHIVES:
            raise ValueError("Only the four approved experimental archives may be downloaded")
        selected_id = drive_id or ARCHIVES[source]
        if selected_id not in {ARCHIVES[source], ARCHIVE_COPIES.get(source)}:
            raise ValueError("Archive source must be official or the user-supplied copy")
        archives = root / "archives"
        archives.mkdir(parents=True, exist_ok=True)
        # Keep partial downloads from different Drive objects separate.
        suffix = "-user-copy" if selected_id != ARCHIVES[source] else ""
        archive = archives / f"{source}{suffix}.tar.zst"
        receipt = archives / f"{source}.json"
        if receipt.exists():
            return json.loads(receipt.read_text())
        if not archive.exists():
            download_archive(selected_id, archive)
        extracted = root / "atlasfold"
        extracted.mkdir(exist_ok=True)
        safe_extract_archive(archive, extracted, excluded_directories=EXCLUDED_TEMPLATE_DIRECTORIES)
        report = {
            "status": "downloaded",
            "source": source,
            "drive_id": selected_id,
            "official_drive_id": ARCHIVES[source],
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "manifests": [
                str(path.relative_to(extracted)) for path in extracted.rglob("manifest*.msgpack")
            ],
        }
        receipt.write_text(json.dumps(report, indent=2) + "\n")
        return report
    if phase == "quality":
        from .structure_metrics import audit_smoke_structures

        return audit_smoke_structures(root)
    if phase == "normalize":
        selected = prepare(
            root / "atlasfold",
            root / "data",
            train=SelectionSpec(split="train", count=1024, monomers=512, dimers=512),
            validation=SelectionSpec(split="validation", count=128, monomers=64, dimers=64),
            candidate_multiplier=candidate_multiplier,
            return_candidates=True,
        )
        empty_pools = [name for name, records in selected.items() if not records]
        if empty_pools:
            raise ValueError(f"No eligible records in {empty_pools}; inspect data/rejections.json")
        records = []
        for split, items in selected.items():
            for item in items:
                item["split"] = "pool" if split == "train" else split
                item["kind"] = "monomer" if len(item["chains"]) == 1 else "dimer"
                item["id"] = f"{item['source']}/{item['id']}"
                path = Path(item["structure_path"])
                item["structure_path"] = str(path.resolve().relative_to((root / "data").resolve()))
                records.append(item)
        write_records_json(records, root / "data/candidates.json")
        return {
            "status": "normalized",
            "counts": {key: len(value) for key, value in selected.items()},
        }
    if phase == "smoke":
        smoke_data = root / "smoke/data"
        candidates = prepare(
            root / "atlasfold",
            smoke_data,
            validation=SelectionSpec(split="validation", count=128, monomers=64, dimers=64),
            return_candidates=True,
        )["validation"]
        selected = []
        for chain_count in (1, 2):
            eligible = [record for record in candidates if len(record["chains"]) == chain_count]
            if not eligible:
                raise ValueError("The smoke panel needs a monomer and a dimer")
            item = min(
                eligible,
                key=lambda record: sum(len(chain["sequence"]) for chain in record["chains"]),
            )
            item["id"] = f"{item['source']}/{item['id']}"
            item["split"] = "train"
            item["kind"] = "monomer" if chain_count == 1 else "dimer"
            item["structure_path"] = str(
                Path(item["structure_path"]).resolve().relative_to(smoke_data.resolve())
            )
            selected.append(item)
        destination = smoke_data / "records.json"
        write_records_json(selected, destination)
        report = {
            "status": "verified",
            "records_sha256": sha256(destination),
            "scope": "two-target smoke check; no optimization or quality evaluation",
            "targets": [{"id": item["id"], "kind": item["kind"]} for item in selected],
        }
        (smoke_data / "split-report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if phase in {"inspect", "validation"}:
        selected = prepare(
            root / "atlasfold",
            root / "inspection",
            validation=SelectionSpec(split="validation", count=128, monomers=64, dimers=64),
            return_candidates=True,
        )
        records = selected["validation"]
        if not records:
            raise ValueError("No eligible records could be decoded from the validation archive")
        if phase == "validation":
            from .splits import chain_clusters, select_disjoint

            for record in records:
                record["split"] = "validation"
                record["id"] = f"{record['source']}/{record['id']}"
            clusters = chain_clusters(records, root / "inspection/sequence_exclusion")
            selected = select_disjoint(
                records,
                clusters,
                counts={"final_test": {}, "validation": {1: 64, 2: 64}, "train": {}},
            )
            return {
                "status": "verified",
                "targets": len(selected),
                "scope": "validation-only sequence and PDB exclusions; cross-split checks remain",
            }
        return {
            "status": "decoded",
            "targets": len(records),
            "strata": {
                str(n): sum(len(record["chains"]) == n for record in records) for n in (1, 2)
            },
            "examples": [
                {
                    "id": record["id"],
                    "chains": len(record["chains"]),
                    "residues": sum(len(chain["sequence"]) for chain in record["chains"]),
                }
                for record in records[:4]
            ],
        }
    if phase == "split":
        from .splits import finalize_splits

        return finalize_splits(root / "data")
    raise ValueError(f"Unknown data-preparation phase: {phase}")


def run_gpu_stage(root: Path, stage: str, model_id: str, **options) -> dict:
    from .training import benchmark, generate_caches, train_head, evaluate_head
    from .campaign import run_campaign
    from .packaging import package_head
    from .release import prepare_release

    functions = {
        "benchmark": benchmark,
        "cache": generate_caches,
        "train": train_head,
        "evaluate": evaluate_head,
        "campaign": run_campaign,
        "package": package_head,
        "release": prepare_release,
    }
    if stage not in functions:
        raise ValueError(f"Unknown GPU stage: {stage}")
    started = time.monotonic()
    report = functions[stage](root, model_id, **options)
    report["elapsed_seconds"] = time.monotonic() - started
    destination = root / model_id
    destination.mkdir(exist_ok=True)
    (destination / f"{stage}-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
