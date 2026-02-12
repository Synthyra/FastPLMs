import argparse
import pathlib
import random
import time
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import torch

from test_scripts.common import build_output_dir
from test_scripts.common import CANONICAL_AMINO_ACIDS
from test_scripts.common import load_model
from test_scripts.common import login_if_needed
from test_scripts.common import parse_int_list
from test_scripts.common import peak_memory_mb
from test_scripts.common import prepare_model_batch
from test_scripts.common import reset_peak_memory
from test_scripts.common import resolve_device
from test_scripts.common import resolve_dtype
from test_scripts.common import run_forward
from test_scripts.common import set_seed
from test_scripts.common import sync_cuda
from test_scripts.model_registry import get_model_specs
from test_scripts.reporting import write_csv
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary


matplotlib.use("Agg")


def _parse_backend_list(backends: str) -> List[str]:
    parsed: List[str] = []
    for chunk in backends.split(","):
        backend = chunk.strip()
        if len(backend) > 0:
            parsed.append(backend)
    assert len(parsed) > 0, "Expected at least one attention backend."
    return parsed


def _random_sequence(length: int, rng: random.Random) -> str:
    return "M" + "".join(rng.choices(CANONICAL_AMINO_ACIDS, k=length - 1))


def _build_sequence_batches(num_batches: int, batch_size: int, length: int, pad_min_ratio: float, seed: int) -> List[List[str]]:
    min_length = max(4, int(length * pad_min_ratio))
    min_length = min(min_length, length)
    rng = random.Random(seed)
    batches: List[List[str]] = []
    for _ in range(num_batches):
        lengths = [rng.randint(min_length, length) for _ in range(batch_size)]
        if batch_size > 1 and length > min_length:
            lengths[0] = length
            lengths[1] = rng.randint(min_length, length - 1)
        batch = [_random_sequence(seq_len, rng) for seq_len in lengths]
        batches.append(batch)
    return batches


def _batch_pad_fraction(spec, prepared_batch: Dict[str, torch.Tensor]) -> float:
    if spec.family == "e1":
        valid = (prepared_batch["sequence_ids"] != -1).float()
    else:
        valid = prepared_batch["attention_mask"].float()
    return float(1.0 - valid.mean().item())


def _plot_throughput(rows: List[Dict[str, object]], batch_sizes: List[int], output_path: pathlib.Path) -> None:
    plt.figure(figsize=(14, max(4, 4 * len(batch_sizes))))
    for index, batch_size in enumerate(batch_sizes):
        plt.subplot(len(batch_sizes), 1, index + 1)
        model_ids: List[str] = []
        for row in rows:
            if int(row["batch_size"]) == batch_size and str(row["series_label"]) not in model_ids:
                model_ids.append(str(row["series_label"]))

        for model_id in model_ids:
            x_lengths: List[int] = []
            y_toks_per_sec: List[float] = []
            for row in rows:
                if int(row["batch_size"]) == batch_size and str(row["series_label"]) == model_id and bool(row["pass"]):
                    x_lengths.append(int(row["sequence_length"]))
                    y_toks_per_sec.append(float(row["tokens_per_second"]))
            if len(x_lengths) > 0:
                paired = sorted(zip(x_lengths, y_toks_per_sec), key=lambda item: item[0])
                x_sorted = [pair[0] for pair in paired]
                y_sorted = [pair[1] for pair in paired]
                plt.plot(x_sorted, y_sorted, marker="o", label=model_id)

        plt.title(f"Throughput vs Sequence Length (batch_size={batch_size})")
        plt.xlabel("Sequence length")
        plt.ylabel("Tokens/second")
        plt.grid(True)
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_memory(rows: List[Dict[str, object]], batch_sizes: List[int], output_path: pathlib.Path) -> None:
    plt.figure(figsize=(14, max(4, 4 * len(batch_sizes))))
    for index, batch_size in enumerate(batch_sizes):
        plt.subplot(len(batch_sizes), 1, index + 1)
        model_ids: List[str] = []
        for row in rows:
            if int(row["batch_size"]) == batch_size and str(row["series_label"]) not in model_ids:
                model_ids.append(str(row["series_label"]))

        for model_id in model_ids:
            x_lengths: List[int] = []
            y_memory: List[float] = []
            for row in rows:
                if int(row["batch_size"]) == batch_size and str(row["series_label"]) == model_id and bool(row["pass"]):
                    x_lengths.append(int(row["sequence_length"]))
                    y_memory.append(float(row["peak_memory_mb"]))
            if len(x_lengths) > 0:
                paired = sorted(zip(x_lengths, y_memory), key=lambda item: item[0])
                x_sorted = [pair[0] for pair in paired]
                y_sorted = [pair[1] for pair in paired]
                plt.plot(x_sorted, y_sorted, marker="o", label=model_id)

        plt.title(f"Peak Memory vs Sequence Length (batch_size={batch_size})")
        plt.xlabel("Sequence length")
        plt.ylabel("Peak memory (MB)")
        plt.grid(True)
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_throughput_suite(args: argparse.Namespace) -> int:
    login_if_needed(args.token)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)
    output_dir = build_output_dir(args.output_dir, "throughput")

    lengths = parse_int_list(args.lengths)
    batch_sizes = parse_int_list(args.batch_sizes)
    attn_backends = _parse_backend_list(args.attn_backends)
    specs = get_model_specs(full_models=args.full_models, families=args.families)
    rows: List[Dict[str, object]] = []
    all_passed = True
    compile_model = args.compile_model or args.compare_attn

    if args.dry_run:
        for spec in specs:
            if args.compare_attn and spec.family in ["esm2", "esmplusplus"]:
                spec_backends = list(attn_backends)
            elif spec.family in ["esm2", "esmplusplus"]:
                spec_backends = [args.attn_backend]
            else:
                spec_backends = ["model_default"]

            for attn_backend in spec_backends:
                for length in lengths:
                    for batch_size in batch_sizes:
                        rows.append(
                            {
                                "model_key": spec.key,
                                "family": spec.family,
                                "repo_id": spec.repo_id,
                                "attn_backend": attn_backend,
                                "compiled_model": compile_model,
                                "series_label": f"{spec.repo_id}|{attn_backend}|compiled={compile_model}",
                                "sequence_length": length,
                                "batch_size": batch_size,
                                "latency_seconds": 0.0,
                                "tokens_per_second": 0.0,
                                "tokens_per_second_with_padding": 0.0,
                                "sequences_per_second": 0.0,
                                "pad_fraction": 0.0,
                                "peak_memory_mb": 0.0,
                                "pass": True,
                                "error": "",
                            }
                        )
        payload: Dict[str, object] = {
            "suite": "throughput",
            "all_passed": True,
            "device": str(device),
            "dtype": str(dtype),
            "lengths": lengths,
            "batch_sizes": batch_sizes,
            "full_models": args.full_models,
            "attn_backends": attn_backends,
            "compare_attn": args.compare_attn,
            "compile_model": compile_model,
            "pad_min_ratio": args.pad_min_ratio,
            "dry_run": True,
            "rows": rows,
        }
        write_json(output_dir / "metrics.json", payload)
        write_csv(output_dir / "metrics.csv", rows)
        summary_lines = [f"Suite: throughput (dry-run)", f"Benchmark points: {len(rows)}", f"Output directory: {output_dir}"]
        write_summary(output_dir / "summary.txt", summary_lines)
        print("\n".join(summary_lines))
        return 0

    for spec in specs:
        if args.compare_attn and spec.family in ["esm2", "esmplusplus"]:
            spec_backends = list(attn_backends)
        elif spec.family in ["esm2", "esmplusplus"]:
            spec_backends = [args.attn_backend]
        else:
            spec_backends = ["model_default"]

        for attn_backend in spec_backends:
            selected_backend = None if attn_backend == "model_default" else attn_backend
            print(
                f"[throughput] Testing {spec.repo_id} "
                f"(attn_backend={attn_backend}, compiled_model={compile_model}) on {device} with {dtype}"
            )
            try:
                model, tokenizer = load_model(
                    spec=spec,
                    task="base",
                    device=device,
                    dtype=dtype,
                    attn_backend=selected_backend,
                    compile_model=compile_model,
                )
            except Exception as exc:
                all_passed = False
                for length in lengths:
                    for batch_size in batch_sizes:
                        rows.append(
                            {
                                "model_key": spec.key,
                                "family": spec.family,
                                "repo_id": spec.repo_id,
                                "attn_backend": attn_backend,
                                "compiled_model": compile_model,
                                "series_label": f"{spec.repo_id}|{attn_backend}|compiled={compile_model}",
                                "sequence_length": length,
                                "batch_size": batch_size,
                                "latency_seconds": float("nan"),
                                "tokens_per_second": float("nan"),
                                "tokens_per_second_with_padding": float("nan"),
                                "sequences_per_second": float("nan"),
                                "pad_fraction": float("nan"),
                                "peak_memory_mb": float("nan"),
                                "pass": False,
                                "error": str(exc),
                            }
                        )
                continue

            for length in lengths:
                for batch_size in batch_sizes:
                    row: Dict[str, object] = {
                        "model_key": spec.key,
                        "family": spec.family,
                        "repo_id": spec.repo_id,
                        "attn_backend": attn_backend,
                        "compiled_model": compile_model,
                        "series_label": f"{spec.repo_id}|{attn_backend}|compiled={compile_model}",
                        "sequence_length": length,
                        "batch_size": batch_size,
                        "latency_seconds": float("nan"),
                        "tokens_per_second": float("nan"),
                        "tokens_per_second_with_padding": float("nan"),
                        "sequences_per_second": float("nan"),
                        "pad_fraction": float("nan"),
                        "peak_memory_mb": float("nan"),
                        "pass": False,
                        "error": "",
                    }
                    try:
                        batch_seed = args.seed + (length * 1000) + batch_size
                        sampled_sequence_length = length
                        if spec.family in ["esm2", "esmplusplus"]:
                            sampled_sequence_length = max(4, length - 2)
                        sequence_batches = _build_sequence_batches(
                            num_batches=args.num_batches,
                            batch_size=batch_size,
                            length=sampled_sequence_length,
                            pad_min_ratio=args.pad_min_ratio,
                            seed=batch_seed,
                        )
                        prepared_batches = []
                        batch_pad_fractions: List[float] = []
                        valid_token_count = 0
                        padded_token_count = 0
                        for sequence_batch in sequence_batches:
                            prepared = prepare_model_batch(
                                spec=spec,
                                model=model,
                                tokenizer=tokenizer,
                                sequence_batch=sequence_batch,
                                device=device,
                                pad_to_length=length,
                            )
                            prepared_batches.append(prepared)
                            batch_pad_fractions.append(_batch_pad_fraction(spec, prepared))
                            if spec.family == "e1":
                                valid_token_count += int((prepared["sequence_ids"] != -1).sum().item())
                            else:
                                valid_token_count += int(prepared["attention_mask"].sum().item())
                            padded_token_count += int(prepared["input_ids"].numel())

                        warmup_steps = args.warmup_steps
                        if compile_model:
                            warmup_steps = max(warmup_steps, 4)

                        with torch.no_grad():
                            for _ in range(warmup_steps):
                                _ = run_forward(
                                    spec=spec,
                                    model=model,
                                    batch=prepared_batches[0],
                                    output_hidden_states=False,
                                    output_attentions=False,
                                )

                            reset_peak_memory(device)
                            sync_cuda(device)
                            start = time.perf_counter()
                            for _ in range(args.timing_runs):
                                for prepared in prepared_batches:
                                    _ = run_forward(
                                        spec=spec,
                                        model=model,
                                        batch=prepared,
                                        output_hidden_states=False,
                                        output_attentions=False,
                                    )
                            sync_cuda(device)
                            elapsed = time.perf_counter() - start

                        batches_processed = args.timing_runs * len(prepared_batches)
                        valid_tokens_processed = args.timing_runs * valid_token_count
                        padded_tokens_processed = args.timing_runs * padded_token_count
                        sequences_processed = batches_processed * batch_size
                        latency_seconds = elapsed / batches_processed
                        row["latency_seconds"] = latency_seconds
                        row["tokens_per_second"] = valid_tokens_processed / elapsed
                        row["tokens_per_second_with_padding"] = padded_tokens_processed / elapsed
                        row["sequences_per_second"] = sequences_processed / elapsed
                        row["pad_fraction"] = sum(batch_pad_fractions) / len(batch_pad_fractions)
                        row["peak_memory_mb"] = peak_memory_mb(device)
                        row["pass"] = True
                    except Exception as exc:
                        row["error"] = str(exc)
                        row["pass"] = False
                        all_passed = False
                    rows.append(row)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload: Dict[str, object] = {
        "suite": "throughput",
        "all_passed": all_passed,
        "device": str(device),
        "dtype": str(dtype),
        "lengths": lengths,
        "batch_sizes": batch_sizes,
        "num_batches": args.num_batches,
        "warmup_steps": args.warmup_steps,
        "timing_runs": args.timing_runs,
        "full_models": args.full_models,
        "compare_attn": args.compare_attn,
        "attn_backend": args.attn_backend,
        "attn_backends": attn_backends,
        "compile_model": compile_model,
        "pad_min_ratio": args.pad_min_ratio,
        "rows": rows,
    }

    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "metrics.csv", rows)
    _plot_throughput(rows=rows, batch_sizes=batch_sizes, output_path=output_dir / "throughput_tokens_per_second.png")
    _plot_memory(rows=rows, batch_sizes=batch_sizes, output_path=output_dir / "throughput_peak_memory_mb.png")

    passed_count = 0
    for row in rows:
        if bool(row["pass"]):
            passed_count += 1
    summary_lines = [
        "Suite: throughput",
        f"Benchmark points: {len(rows)}",
        f"Points passed: {passed_count}",
        f"Points failed: {len(rows) - passed_count}",
        f"Output directory: {output_dir}",
    ]
    for row in rows:
        status = "PASS" if bool(row["pass"]) else "FAIL"
        summary_lines.append(
            f"{status} | {row['repo_id']} | backend={row['attn_backend']} | compiled={row['compiled_model']} | len={row['sequence_length']} | bs={row['batch_size']} | pad_fraction={row['pad_fraction']} | tok_s={row['tokens_per_second']} | mem_mb={row['peak_memory_mb']} | error={row['error']}"
        )
    write_summary(output_dir / "summary.txt", summary_lines)
    print("\n".join(summary_lines))

    if all_passed:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model throughput benchmarks.")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=str, default="64,128,256")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timing-runs", type=int, default=4)
    parser.add_argument("--attn-backend", type=str, default="flex", choices=["flex", "sdpa", "model_default"])
    parser.add_argument("--attn-backends", type=str, default="sdpa,flex")
    parser.add_argument("--compare-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pad-min-ratio", type=float, default=0.5)
    parser.add_argument("--full-models", action="store_true")
    parser.add_argument("--families", nargs="+", default=None, choices=["e1", "esm2", "esmplusplus"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_throughput_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())

