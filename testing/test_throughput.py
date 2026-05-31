"""Pytest-based throughput benchmark across models, backends, batch sizes, and sequence lengths.

Wraps ThroughputChecker from throughput.py and saves structured results as JSON, CSV,
and a PNG plot to /workspace/ (or the current directory if /workspace/ does not exist).

Marked as `slow` + `gpu` because it runs compiled inference across many configurations.
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from testing.conftest import BACKENDS, FULL_MODEL_REGISTRY
from testing.throughput import ThroughputChecker, plot_results, save_structured_results


# Models to benchmark: one representative per family that uses a tokenizer.
# E1 is excluded because throughput.py uses model.tokenizer (tokenizer mode only).
THROUGHPUT_MODELS: Dict[str, str] = {
    "ESM2-8M": "Synthyra/ESM2-8M",
    "ESMplusplus_small": "Synthyra/ESMplusplus_small",
    "DPLM-150M": "Synthyra/DPLM-150M",
    "DPLM2-150M": "Synthyra/DPLM2-150M",
}

BATCH_SIZES = [2, 4, 8]
SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024]
MIN_LENGTH = 32

# Fewer timed batches than standalone for faster pytest runs
WARMUP_BATCHES = 5
TIMED_BATCHES = 25


def _get_output_dir() -> Path:
    workspace = Path("/workspace")
    if workspace.is_dir():
        return workspace
    return Path(".")


@pytest.mark.gpu
@pytest.mark.slow
def test_throughput_benchmark() -> None:
    """Benchmark tokens/sec across backends, batch sizes, and sequence lengths.

    Saves results as JSON, CSV, and PNG to the output directory.
    """
    assert torch.cuda.is_available(), "Throughput benchmark requires CUDA"

    checker = ThroughputChecker(
        warmup_batches=WARMUP_BATCHES,
        timed_batches=TIMED_BATCHES,
    )

    all_results: Dict[str, Dict] = {}

    for model_name, model_path in THROUGHPUT_MODELS.items():
        print(f"\n--- Benchmarking {model_name} ({model_path}) ---")
        results = checker.evaluate(
            model_path,
            BATCH_SIZES,
            min_length=MIN_LENGTH,
            sequence_lengths=SEQUENCE_LENGTHS,
            backends=list(BACKENDS),
        )
        all_results[model_path] = results

    output_dir = _get_output_dir()
    save_structured_results(all_results, str(output_dir))
    plot_results(all_results, str(output_dir / "throughput_comparison.png"))

    # Verify that results were collected
    json_path = output_dir / "throughput_results.json"
    assert json_path.exists(), "throughput_results.json was not created"
    with open(json_path) as f:
        rows = json.load(f)
    assert len(rows) > 0, "No throughput results collected"

    # Sanity check: at least one backend produced positive throughput
    max_throughput = max(r["tokens_per_sec"] for r in rows)
    assert max_throughput > 0, "All throughput measurements are zero"

    print(f"\nResults saved to {output_dir}")
    print(f"  JSON: {output_dir / 'throughput_results.json'}")
    print(f"  CSV:  {output_dir / 'throughput_results.csv'}")
    print(f"  PNG:  {output_dir / 'throughput_comparison.png'}")
