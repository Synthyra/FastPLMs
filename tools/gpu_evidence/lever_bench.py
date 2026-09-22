"""Compare inference latency between two source trees or between attention backends.

The ``trees`` mode runs the working tree against its Git baseline. The
``backends`` mode runs the working tree once per advertised attention backend.
All workers share one container and one GPU and alternate, so driver, clocks,
and thermals are shared. Models are randomly initialized at published
dimensions: latency does not depend on weight values, and no weights are
downloaded. Results are descriptive evidence for keeping or dropping a lever or
for ordering backends, not release benchmark claims.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any


TREES = {"baseline": Path("/baseline"), "candidate": Path("/workspace")}
ROUNDS = 3
WORKER_TIMEOUT_SECONDS = 900
# Length changes on every forward, as it does for length-bucketed embedding batches.
VARIED_LENGTHS = (128, 512, 192, 448, 256, 384, 160, 320)
# Residues per row of one right-padded batch of 8 x 512 positions. ``padded`` holds
# mixed-length proteins (2336 residues). ``full`` has no padding, which isolates the
# cost of the varlen gather and scatter when there is nothing to skip.
BATCH_ROW_LENGTHS = {
    "padded": (512, 448, 384, 320, 256, 192, 128, 96),
    "full": (512,) * 8,
}
PAD_TOKEN_ID = 1
Operation = Callable[[], object]


# One protein per forward, where kernel launches rather than arithmetic set the latency.
# Only tree comparisons use it, so the backend bench keeps its two batch kinds.
SINGLE_ROW_LENGTHS = {"single": (256,)}


def _masked_batch(torch: Any, device: str, smoke: bool, batch_kind: str) -> tuple[Any, Any]:
    rows = {**BATCH_ROW_LENGTHS, **SINGLE_ROW_LENGTHS}[batch_kind]
    row_lengths = torch.tensor(rows, device=device)  # (b,)
    if smoke:
        row_lengths = row_lengths // 16  # (b,)
    positions = torch.arange(int(row_lengths.max()), device=device)  # (l,)
    attention_mask = positions[None, :] < row_lengths[:, None]  # (b, l)
    residues = torch.randint(4, 24, attention_mask.shape, device=device)  # (b, l)
    input_ids = residues.masked_fill(~attention_mask, PAD_TOKEN_ID)  # (b, l)
    return input_ids, attention_mask


def _executable_backend(backend: str, smoke: bool) -> str:
    # The CPU smoke run checks wiring only, and FlashAttention needs CUDA.
    return "sdpa" if smoke and backend.startswith("flash_attention") else backend


def _esm2(
    torch: Any,
    device: str,
    smoke: bool,
    batch: int,
    lengths: tuple[int, ...],
    backend: str = "sdpa",
    batch_kind: str | None = None,
) -> Operation:
    from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel

    # ESM2-150M dimensions.
    hidden, layers, heads = (32, 2, 4) if smoke else (640, 30, 20)
    config = FastEsmConfig(
        vocab_size=33,
        pad_token_id=PAD_TOKEN_ID,
        mask_token_id=32,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        attn_backend=_executable_backend(backend, smoke),
    )
    model = FastEsmModel(config).to(device).eval()
    if batch_kind is not None:
        batches = [_masked_batch(torch, device, smoke, batch_kind)]
    else:
        batches = [
            (input_ids, torch.ones_like(input_ids))
            for input_ids in (
                torch.randint(4, 24, (batch, length), device=device)  # (b, l)
                for length in (lengths[:2] if smoke else lengths)
            )
        ]

    def operation() -> None:
        for input_ids, attention_mask in batches:
            model(input_ids=input_ids, attention_mask=attention_mask)

    return operation


def _esmpp_masked(torch: Any, device: str, smoke: bool, backend: str, batch_kind: str) -> Operation:
    from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
        ESMplusplusConfig,
        ESMplusplusModel,
    )

    # ESMC-300M dimensions.
    hidden, layers, heads = (32, 2, 4) if smoke else (960, 30, 15)
    config = ESMplusplusConfig(
        vocab_size=64,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        pad_token_id=PAD_TOKEN_ID,
        attn_backend=_executable_backend(backend, smoke),
    )
    model = ESMplusplusModel(config).to(device).eval()
    input_ids, attention_mask = _masked_batch(torch, device, smoke, batch_kind)  # both (b, l)

    def operation() -> None:
        model(input_ids=input_ids, attention_mask=attention_mask)

    return operation


def _dplm_masked(torch: Any, device: str, smoke: bool, backend: str, batch_kind: str) -> Operation:
    from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMModel

    # DPLM-150M dimensions.
    hidden, layers, heads = (32, 2, 4) if smoke else (640, 30, 20)
    config = DPLMConfig(
        vocab_size=33,
        pad_token_id=PAD_TOKEN_ID,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        position_embedding_type="rotary",
        attn_backend=_executable_backend(backend, smoke),
    )
    model = DPLMModel(config).to(device).eval()
    input_ids, attention_mask = _masked_batch(torch, device, smoke, batch_kind)  # both (b, l)

    def operation() -> None:
        model(input_ids=input_ids, attention_mask=attention_mask)

    return operation


def _dplm2_packed(torch: Any, device: str, smoke: bool) -> Operation:
    from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config, DPLM2Model

    # DPLM2-150M dimensions.
    hidden, layers, heads = (32, 2, 4) if smoke else (640, 30, 20)
    config = DPLM2Config(
        vocab_size=8229,
        pad_token_id=1,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        position_embedding_type="rotary",
    )
    model = DPLM2Model(config).to(device).eval()
    half = 16 if smoke else 256
    amino_acids = torch.randint(4, 24, (4, half), device=device)  # (b, l / 2)
    structure = torch.randint(40, 8000, (4, half), device=device)  # (b, l / 2)
    input_ids = torch.cat((amino_acids, structure), dim=-1)  # (b, l)
    # Packed layout: the amino-acid track, then the structure track.
    type_ids = torch.cat(  # (b, l)
        (
            torch.full_like(amino_acids, config.aa_type),
            torch.full_like(structure, config.struct_type),
        ),
        dim=-1,
    )
    attention_mask = torch.ones_like(input_ids)  # (b, l)

    def operation() -> None:
        model(input_ids=input_ids, attention_mask=attention_mask, type_ids=type_ids)

    return operation


def _e1_single_sequences(torch: Any, device: str, smoke: bool) -> Operation:
    from fastplms.models.e1.modeling_e1 import FAST_E1_ENCODER, E1Config

    # A 30-layer encoder at E1-small scale. Every third layer uses global positions.
    hidden, layers, heads = (32, 3, 4) if smoke else (768, 30, 12)
    config = E1Config(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=4 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads // 2,
        global_attention_every_n_layers=3,
        attn_backend="sdpa",
        dtype="float32",
    )
    model = FAST_E1_ENCODER(config).to(device).eval()
    batch, length = (2, 16) if smoke else (4, 512)
    positions = torch.arange(length, device=device).expand(batch, length)  # (b, l)
    inputs = {
        "input_ids": torch.randint(4, 24, (batch, length), device=device),  # (b, l)
        "within_seq_position_ids": positions,
        "global_position_ids": positions,
        # One protein per row and no retrieved context.
        "sequence_ids": torch.zeros(batch, length, dtype=torch.long, device=device),  # (b, l)
    }

    def operation() -> None:
        model(**inputs)

    return operation


def _runner_full_embeddings(torch: Any, device: str, smoke: bool) -> Operation:
    """Per-residue embedding of many short proteins, where host transfers are frequent."""
    from torch import nn

    from fastplms.embeddings import EmbeddingBatch, embed_dataset
    from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel

    # ESM2-35M dimensions.
    hidden, layers, heads = (32, 2, 4) if smoke else (480, 12, 20)
    config = FastEsmConfig(
        vocab_size=33,
        pad_token_id=PAD_TOKEN_ID,
        mask_token_id=32,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        attn_backend="sdpa",
    )

    class ResidueEmbedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = FastEsmModel(config)
            self.config = config

        def _embedding_batch(self, sequences: list[str]) -> Any:
            row_lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)  # (b,)
            positions = torch.arange(int(row_lengths.max()) + 2, device=device)  # (l,)
            # Position 0 is BOS and position r_i + 1 is EOS, so neither is a residue.
            residue_mask = (positions[None, :] >= 1) & (  # (b, l)
                positions[None, :] <= row_lengths[:, None]
            )
            attention_mask = positions[None, :] <= row_lengths[:, None] + 1  # (b, l)
            input_ids = torch.full_like(attention_mask, 5, dtype=torch.long)  # (b, l)
            input_ids = input_ids.masked_fill(~attention_mask, PAD_TOKEN_ID)  # (b, l)
            hidden_states = self.encoder(  # (b, l, d)
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            return EmbeddingBatch(X=hidden_states, residue_mask=residue_mask)

    model = ResidueEmbedder().to(device).eval()
    sequence_count = 8 if smoke else 256
    sequences = ["A" * (40 + (17 * index) % 160) for index in range(sequence_count)]

    def operation() -> None:
        embed_dataset(
            model,
            sequences,
            full_embeddings=True,
            batch_size=4 if smoke else 64,
            model_state_fingerprint="lever-bench",
        )

    return operation


WORKLOADS: dict[str, Callable[[Any, str, bool], Operation]] = {
    # Targets of the shared rotary-cache lever, at latency-bound and batched sizes.
    "esm2-varied-length-b1": lambda t, d, s: _esm2(t, d, s, 1, VARIED_LENGTHS),
    "esm2-varied-length-b8": lambda t, d, s: _esm2(t, d, s, 8, VARIED_LENGTHS),
    # Control: the length never changes, so the rotary lever cannot help here.
    "esm2-fixed-length-b8": lambda t, d, s: _esm2(t, d, s, 8, (320,) * len(VARIED_LENGTHS)),
    "dplm2-packed-b4": _dplm2_packed,
    "runner-full-embeddings-b64": _runner_full_embeddings,
    "e1-single-sequences-b4": _e1_single_sequences,
    "esmpp-single-l256-sdpa": lambda t, d, s: _esmpp_masked(t, d, s, "sdpa", "single"),
}
# The tree comparison defaults to the levers that change the default SDPA path.
DEFAULT_TREE_WORKLOADS = tuple(WORKLOADS)
# Masked-batch workloads per family and advertised backend (src/fastplms/models.toml).
MASKED_BACKENDS: dict[str, tuple[str, ...]] = {
    "esm2": ("sdpa", "flash_attention_2", "flash_attention_3"),
    "esmpp": ("sdpa", "flash_attention_2", "flash_attention_3"),
    "dplm": ("sdpa", "flash_attention_3"),
}
_MASKED_BUILDERS: dict[str, Callable[[Any, str, bool, str, str], Operation]] = {
    "esm2": lambda t, d, s, backend, batch_kind: _esm2(t, d, s, 8, (), backend, batch_kind),
    "esmpp": _esmpp_masked,
    "dplm": _dplm_masked,
}


def _masked_workload_name(family: str, batch_kind: str, backend: str) -> str:
    return f"{family}-{batch_kind}-b8-{backend}"


def _masked_workload(
    family: str, batch_kind: str, backend: str
) -> Callable[[Any, str, bool], Operation]:
    return lambda t, d, s: _MASKED_BUILDERS[family](t, d, s, backend, batch_kind)


for _family, _backends in MASKED_BACKENDS.items():
    for _batch_kind in BATCH_ROW_LENGTHS:
        for _backend in _backends:
            WORKLOADS[_masked_workload_name(_family, _batch_kind, _backend)] = _masked_workload(
                _family, _batch_kind, _backend
            )


def run_worker(workload: str, smoke: bool) -> dict[str, object]:
    """Measure one workload against whichever ``fastplms`` tree is on the path."""
    import torch

    import fastplms

    from benchmarks.run import measure_blocks, warm_until_stable

    device = "cpu" if smoke else "cuda"
    torch.manual_seed(0)
    operation = WORKLOADS[workload](torch, device, smoke)
    result: dict[str, object] = {"workload": workload, "fastplms_file": fastplms.__file__}
    with torch.inference_mode():
        if smoke:
            operation()
            return result
        # FP32 parameters under BF16 autocast: the default FastPLMs execution policy.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            warm_until_stable(torch, operation, tolerance=0.05, maximum_samples=200)
            blocks = measure_blocks(
                torch,
                operation,
                logical_tokens_per_forward=1,
                padded_tokens_per_forward=1,
                blocks=5,
                minimum_block_ms=1000.0,
            )
    samples = [sample for block in blocks for sample in block.samples_ms]
    result.update(median_ms=statistics.median(samples), sample_count=len(samples))
    return result


def _spawn_worker(tree: str, workload: str) -> dict[str, Any]:
    root = TREES[tree]
    environment = {**os.environ, "PYTHONPATH": f"{root / 'src'}{os.pathsep}/workspace"}
    completed = subprocess.run(
        [sys.executable, "-m", "tools.gpu_evidence.lever_bench", "--worker", workload],
        cwd="/workspace",
        env=environment,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{tree} worker failed for {workload}:\n{completed.stderr[-4000:]}")
    measured: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    # An A/B result is meaningless if both workers imported the same tree.
    if not Path(measured["fastplms_file"]).is_relative_to(root / "src"):
        raise RuntimeError(f"{tree} worker imported {measured['fastplms_file']}")
    return measured


def run_tree_driver(workloads: tuple[str, ...]) -> dict[str, object]:
    report: dict[str, object] = {}
    for workload in workloads:
        medians: dict[str, list[float]] = {tree: [] for tree in TREES}
        for round_index in range(ROUNDS):
            # Alternate which tree runs first so ordering effects cancel.
            order = list(TREES) if round_index % 2 == 0 else list(reversed(TREES))
            for tree in order:
                medians[tree].append(float(_spawn_worker(tree, workload)["median_ms"]))
        baseline = statistics.median(medians["baseline"])
        candidate = statistics.median(medians["candidate"])
        report[workload] = {
            "baseline_ms": medians["baseline"],
            "candidate_ms": medians["candidate"],
            "speedup": baseline / candidate,
            "per_round_speedup": [
                before / after
                for before, after in zip(medians["baseline"], medians["candidate"], strict=True)
            ],
        }
        print(json.dumps({workload: report[workload]}), flush=True)
    return report


def run_backend_driver() -> dict[str, object]:
    """Time every advertised backend of each family and masked batch in the working tree.

    A backend that cannot run on this GPU is recorded with its error. That outcome
    is evidence for an availability gate, so it must not abort the other backends.
    """
    report: dict[str, object] = {}
    cases = [
        (family, batch_kind, backends)
        for family, backends in MASKED_BACKENDS.items()
        for batch_kind in BATCH_ROW_LENGTHS
    ]
    for family, batch_kind, backends in cases:
        medians: dict[str, list[float]] = {backend: [] for backend in backends}
        failures: dict[str, str] = {}
        for round_index in range(ROUNDS):
            order = backends if round_index % 2 == 0 else tuple(reversed(backends))
            for backend in order:
                if backend in failures:
                    continue
                try:
                    workload = _masked_workload_name(family, batch_kind, backend)
                    measured = _spawn_worker("candidate", workload)
                except RuntimeError as error:
                    failures[backend] = str(error)[-1500:]
                    continue
                medians[backend].append(float(measured["median_ms"]))
        measured_ms = {
            backend: statistics.median(samples)
            for backend, samples in medians.items()
            if samples and backend not in failures
        }
        case = f"{family}-{batch_kind}"
        report[case] = {
            "median_ms": measured_ms,
            "per_round_ms": medians,
            "speedup_over_sdpa": {
                backend: measured_ms["sdpa"] / milliseconds
                for backend, milliseconds in measured_ms.items()
                if "sdpa" in measured_ms
            },
            "failures": failures,
        }
        print(json.dumps({case: report[case]}), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=tuple(WORKLOADS), default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run that only checks wiring")
    parser.add_argument("--mode", choices=("trees", "backends"), default="trees")
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS), default=None)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(run_worker(args.worker, args.smoke)))
        return 0
    if args.mode == "backends":
        print(json.dumps(run_backend_driver(), indent=2))
        return 0
    workloads = DEFAULT_TREE_WORKLOADS if args.workloads is None else tuple(args.workloads)
    print(json.dumps(run_tree_driver(workloads), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
