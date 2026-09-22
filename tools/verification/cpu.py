"""Verify selected repository contracts on one bounded, offline Modal CPU worker."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
import uuid

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def upstream_legal_files(root: Path) -> tuple[str, ...]:
    """Include the pinned upstream legal texts exercised by artifact release checks."""
    with (root / "src/fastplms/models.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    return tuple(
        f"{source['path']}/{name}"
        for source in manifest["upstreams"]
        for name in source["license_files"]
    )


def _write_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, help="New run directory; existing paths are refused."
    )
    args = parser.parse_args()
    output = args.output_root or ROOT / "artifacts/verification" / (
        f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    output = output.resolve()
    if output.exists():
        parser.error(f"Run directory already exists: {output}")

    from tools.execution.source import stage_source_snapshot
    from tools.gpu_evidence.source import SOURCE_DIRECTORIES, SOURCE_FILES, excluded_from_upload

    snapshot = stage_source_snapshot(
        ROOT,
        output / "source",
        directories=SOURCE_DIRECTORIES,
        files=(*SOURCE_FILES, *upstream_legal_files(ROOT)),
        exclude=excluded_from_upload,
    )
    report: dict[str, object] = {
        "status": "dispatching",
        "source": snapshot.to_dict(),
        "resources": {"cpu": 4, "memory_mib": 16384, "timeout_seconds": 1200, "gpu": None},
    }
    report_path = output / "report.json"
    _write_report(report_path, report)

    import modal

    from tools.verification.worker import verify

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "libgomp1")
        .uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    )
    for name in ("core.in", "features/structure.in", "features/dev.in", "features/train.in"):
        image = image.pip_install_from_requirements(str(snapshot.root / "requirements" / name))
    image = (
        image.uv_pip_install(
            "transformers==5.13.0",
            "wandb==0.18.7",
            "lmdb==1.7.3",
            "gdown==5.2.0",
            "python-dotenv==1.1.1",
            "gemmi==0.7.3",
            "pyarrow==21.0.0",
            "DockQ==2.1.3",
            "tmtools==0.2.0",
            "numpy==1.26.4",
            "scipy==1.16.3",
        )
        .env(
            {
                "PYTHONPATH": "/workspace/src:/workspace",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "4",
            }
        )
        .workdir("/workspace")
        .add_local_dir(str(snapshot.root), "/workspace")
    )
    app = modal.App("fastplms-cpu-verification")
    worker = app.function(
        image=image,
        cpu=4,
        memory=16384,
        timeout=1200,
        startup_timeout=600,
        max_containers=1,
        scaledown_window=2,
        include_source=False,
        block_network=True,
    )(verify)
    started = time.monotonic()
    try:
        with modal.enable_output(), app.run():
            report["modal_app_id"] = app.app_id
            _write_report(report_path, report)
            report.update(worker.remote())
    finally:
        if report["status"] == "dispatching":
            report["status"] = "dispatch_failed"
        report["wall_seconds"] = time.monotonic() - started
        _write_report(report_path, report)
    for batch in report.get("batches", []):
        name = batch["name"]
        for field, suffix in (
            ("stdout", "stdout.txt"),
            ("stderr", "stderr.txt"),
            ("junit_xml", "xml"),
        ):
            content = batch.get(field)
            if content is not None:
                (output / f"{name}.{suffix}").write_text(content, encoding="utf-8")
        print(f"{name}: {batch['status']}")
    print(f"Report: {report_path}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
