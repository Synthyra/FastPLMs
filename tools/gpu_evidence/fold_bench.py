"""ESMFold2 folding throughput and peak memory by protein length, on one worker.

Every series runs in its own process on the same GPU, so ratios between series
are within-run. The official implementation runs under its own interpreter with
its pinned Transformers fork; the FastPLMs series run from the working tree, and
one from the Git baseline tree so a working-tree change can be attributed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path

from fastplms.registry import get_model_registry

from .fold_worker import RESULT_PREFIX
from .source import BASELINE_WORKSPACE, WORKSPACE


REFERENCE_PYTHON = "/opt/reference/bin/python"
WORKER_SCRIPT = f"{WORKSPACE}/tools/gpu_evidence/fold_worker.py"
LENGTHS = (64, 128, 256, 384, 512, 768, 1024)
# Two lengths are enough to attribute a working-tree change against the baseline tree.
ATTRIBUTION_LENGTHS = (256, 768)
SMOKE_LENGTHS = (48, 96)
SERIES_TIMEOUT_SECONDS = 3_000


@dataclass(frozen=True)
class Series:
    """One curve: an implementation, a checkpoint, and an atom attention mode."""

    label: str
    model_id: str
    implementation: str  # "upstream", "fastplms", or "fastplms-baseline"
    atom_attention: str = "dense"
    lengths: tuple[int, ...] = LENGTHS
    compare_atom_attention: bool = False
    peak_allocations: bool = False
    unchunked: bool = False
    chunk_size: int | None = None


# ``dense`` series use every default, so their output is bitwise identical to the
# Git baseline. ``windowed-unchunked`` series add the two opt-in speed settings.
SERIES = (
    Series("upstream-esmfold2", "esmfold2", "upstream"),
    Series(
        "fastplms-baseline-esmfold2", "esmfold2", "fastplms-baseline", lengths=ATTRIBUTION_LENGTHS
    ),
    Series("fastplms-esmfold2-dense", "esmfold2", "fastplms", compare_atom_attention=True),
    Series(
        "fastplms-esmfold2-windowed-unchunked", "esmfold2", "fastplms", "windowed", unchunked=True
    ),
    Series("fastplms-esmfold2_600-dense", "esmfold2_600", "fastplms"),
    Series(
        "fastplms-esmfold2_600-windowed-unchunked",
        "esmfold2_600",
        "fastplms",
        "windowed",
        unchunked=True,
    ),
    Series("fastplms-esmfold2_300-dense", "esmfold2_300", "fastplms", compare_atom_attention=True),
    Series(
        "fastplms-esmfold2_300-windowed-unchunked",
        "esmfold2_300",
        "fastplms",
        "windowed",
        unchunked=True,
    ),
)
# The 2,048-residue extension. Each fold takes minutes, so every series makes a single
# instrumented pass there. The optimized series repeat 1,024 residues as an anchor
# against the main sweep, which ran on another worker. Short series run first, so a
# stage timeout would still leave their results in the output.
LONG_LENGTH = 2048
ANCHORED = (1024, LONG_LENGTH)
LONG_SERIES = (
    Series(
        "fastplms-esmfold2_300-windowed-unchunked",
        "esmfold2_300",
        "fastplms",
        "windowed",
        ANCHORED,
        unchunked=True,
    ),
    Series(
        "fastplms-esmfold2_600-windowed-unchunked",
        "esmfold2_600",
        "fastplms",
        "windowed",
        ANCHORED,
        unchunked=True,
    ),
    Series(
        "fastplms-esmfold2-windowed-unchunked",
        "esmfold2",
        "fastplms",
        "windowed",
        ANCHORED,
        unchunked=True,
    ),
    # In case unchunked pair updates exhaust memory at this length.
    Series(
        "fastplms-esmfold2-windowed-chunk512",
        "esmfold2",
        "fastplms",
        "windowed",
        (LONG_LENGTH,),
        chunk_size=512,
    ),
    Series("fastplms-esmfold2-dense", "esmfold2", "fastplms", lengths=(LONG_LENGTH,)),
    Series("upstream-esmfold2", "esmfold2", "upstream", lengths=(LONG_LENGTH,)),
)
LONG_SERIES_TIMEOUT_SECONDS = 3_300
# Attributes peak memory to model source lines, in both chunking modes.
PEAK_SERIES = (
    Series(
        "peak-esmfold2_300-default",
        "esmfold2_300",
        "fastplms",
        lengths=(512,),
        peak_allocations=True,
    ),
    Series(
        "peak-esmfold2_300-unchunked",
        "esmfold2_300",
        "fastplms",
        "windowed",
        (512,),
        unchunked=True,
        peak_allocations=True,
    ),
)
# The smoke run folds short proteins with every checkpoint of the sweep. It checks
# both environments end to end and leaves the weights in the cache volume, so the
# expensive worker does not pay for downloads.
SMOKE_SERIES = (
    Series("upstream-esmfold2", "esmfold2", "upstream", lengths=SMOKE_LENGTHS),
    Series("fastplms-baseline-esmfold2", "esmfold2", "fastplms-baseline", lengths=SMOKE_LENGTHS),
    Series("fastplms-esmfold2-windowed", "esmfold2", "fastplms", "windowed", SMOKE_LENGTHS),
    Series("fastplms-esmfold2_600-dense", "esmfold2_600", "fastplms", lengths=SMOKE_LENGTHS),
    Series("fastplms-esmfold2_300-windowed", "esmfold2_300", "fastplms", "windowed", SMOKE_LENGTHS),
)


def worker_command(series: Series, fold_settings: list[str]) -> tuple[list[str], dict[str, str]]:
    """Interpreter arguments and environment that run one series."""
    registry = get_model_registry()
    spec = registry.get(series.model_id)
    environment = dict(os.environ)
    if series.implementation == "upstream":
        interpreter, repo = REFERENCE_PYTHON, spec.official
        # The official environment must not see the candidate tree.
        environment["PYTHONPATH"] = ""
    else:
        interpreter, repo = sys.executable, spec.fast
        tree = BASELINE_WORKSPACE if series.implementation == "fastplms-baseline" else WORKSPACE
        environment["PYTHONPATH"] = f"{tree}/src"
    command = [
        interpreter,
        WORKER_SCRIPT,
        "--implementation",
        "upstream" if series.implementation == "upstream" else "fastplms",
        "--label",
        series.label,
        "--repo",
        repo.repo_id,
        "--revision",
        repo.revision,
        "--atom-attention",
        series.atom_attention,
        "--lengths",
        *(str(length) for length in series.lengths),
        *fold_settings,
    ]
    # Only the small checkpoints override the family AutoModel class.
    if dict(spec.auto_map_items).get("AutoModel", "").endswith("ESMFold2ExperimentalModel"):
        command.append("--experimental")
    if series.implementation == "upstream":
        backbone = registry.get(spec.backbone_model or spec.family.backbone_model).official
        command += ["--backbone-repo", backbone.repo_id, "--backbone-revision", backbone.revision]
    if series.compare_atom_attention:
        command.append("--compare-atom-attention")
    if series.unchunked:
        command.append("--unchunked")
    if series.peak_allocations:
        command.append("--peak-allocations")
    if series.chunk_size is not None:
        command += ["--chunk-size", str(series.chunk_size)]
    return command, environment


def run_series(
    series: Series, fold_settings: list[str], timeout_seconds: int = SERIES_TIMEOUT_SECONDS
) -> dict[str, object]:
    command, environment = worker_command(series, fold_settings)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
            cwd=Path(WORKSPACE),
        )
    except subprocess.TimeoutExpired:
        return {"label": series.label, "status": "timed_out"}
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return {"status": "ok", **json.loads(line.removeprefix(RESULT_PREFIX))}
    return {
        "label": series.label,
        "status": "failed",
        "exit_code": completed.returncode,
        "output_tail": (completed.stdout + completed.stderr)[-4_000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--long", action="store_true")
    modes.add_argument("--peak", action="store_true")
    args = parser.parse_args()

    if args.peak:
        results = [run_series(series, []) for series in PEAK_SERIES]
    elif args.long:
        fold_settings = ["--single-pass-from", str(LONG_LENGTH)]
        results = []
        for series in LONG_SERIES:
            results.append(run_series(series, fold_settings, LONG_SERIES_TIMEOUT_SECONDS))
            # Partial results survive in the output if a later series exhausts the stage.
            print("FOLD_BENCH_SERIES " + json.dumps(results[-1]), flush=True)
    else:
        fold_settings = (
            ["--num-loops", "1", "--num-sampling-steps", "8", "--repeats", "1"]
            if args.smoke
            else []
        )
        results = [
            run_series(series, fold_settings) for series in (SMOKE_SERIES if args.smoke else SERIES)
        ]
    print(json.dumps({"smoke": args.smoke, "long": args.long, "series": results}))
    if any(result["status"] != "ok" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
