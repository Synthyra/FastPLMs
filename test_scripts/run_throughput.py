import argparse
import math
import pathlib
import random
import re
import time
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

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
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
    }
)


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


def _plot_flex_sdpa_delta(rows: List[Dict[str, object]], batch_sizes: List[int], metric_key: str, title_prefix: str, y_label: str, output_path: pathlib.Path) -> None:
    plt.figure(figsize=(14, max(4, 4 * len(batch_sizes))))
    for index, batch_size in enumerate(batch_sizes):
        plt.subplot(len(batch_sizes), 1, index + 1)
        model_ids: List[str] = []
        for row in rows:
            if int(row["batch_size"]) == batch_size and str(row["repo_id"]) not in model_ids:
                model_ids.append(str(row["repo_id"]))

        for model_id in model_ids:
            x_lengths: List[int] = []
            y_values: List[float] = []
            for row in rows:
                if int(row["batch_size"]) == batch_size and str(row["repo_id"]) == model_id:
                    x_lengths.append(int(row["sequence_length"]))
                    y_values.append(float(row[metric_key]))
            if len(x_lengths) > 0:
                paired = sorted(zip(x_lengths, y_values), key=lambda item: item[0])
                x_sorted = [pair[0] for pair in paired]
                y_sorted = [pair[1] for pair in paired]
                plt.plot(x_sorted, y_sorted, marker="o", label=model_id)

        plt.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0)
        plt.title(f"{title_prefix} (batch_size={batch_size})")
        plt.xlabel("Sequence length")
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _slugify_filename(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    assert len(slug) > 0, f"Failed to build filename slug from value: {value}"
    return slug


def _plot_publication_sdpa_vs_flex_per_model(rows: List[Dict[str, object]], output_dir: pathlib.Path) -> List[pathlib.Path]:
    publication_dir = output_dir / "publication_plots"
    publication_dir.mkdir(parents=True, exist_ok=True)

    model_rows_index: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if bool(row["pass"]) is False:
            continue
        backend = str(row["attn_backend"])
        if backend not in ["sdpa", "flex"]:
            continue
        if _is_finite(row["tokens_per_second"]) is False:
            continue
        if _is_finite(row["peak_memory_mb"]) is False:
            continue
        model_group_key = f"{row['repo_id']}|compiled={row['compiled_model']}"
        if model_group_key not in model_rows_index:
            model_rows_index[model_group_key] = []
        model_rows_index[model_group_key].append(row)

    generated_paths: List[pathlib.Path] = []
    for model_group_key in sorted(model_rows_index.keys()):
        model_rows = model_rows_index[model_group_key]
        has_sdpa = False
        has_flex = False
        for row in model_rows:
            backend = str(row["attn_backend"])
            if backend == "sdpa":
                has_sdpa = True
            if backend == "flex":
                has_flex = True
        if has_sdpa is False or has_flex is False:
            continue

        batch_sizes: List[int] = []
        for row in model_rows:
            batch_size = int(row["batch_size"])
            if batch_size not in batch_sizes:
                batch_sizes.append(batch_size)
        batch_sizes = sorted(batch_sizes)

        repo_id = str(model_rows[0]["repo_id"])
        compiled_model = bool(model_rows[0]["compiled_model"])
        figure, axes = plt.subplots(
            nrows=len(batch_sizes),
            ncols=2,
            figsize=(13, max(4.2, 3.8 * len(batch_sizes))),
            squeeze=False,
        )
        colors = {"sdpa": "#1f77b4", "flex": "#d62728"}
        markers = {"sdpa": "o", "flex": "s"}

        for row_index, batch_size in enumerate(batch_sizes):
            throughput_axis = axes[row_index][0]
            memory_axis = axes[row_index][1]

            for backend in ["sdpa", "flex"]:
                lengths: List[int] = []
                throughputs: List[float] = []
                memories: List[float] = []
                for row in model_rows:
                    if int(row["batch_size"]) == batch_size and str(row["attn_backend"]) == backend:
                        lengths.append(int(row["sequence_length"]))
                        throughputs.append(float(row["tokens_per_second"]))
                        memories.append(float(row["peak_memory_mb"]))
                if len(lengths) == 0:
                    continue
                paired = sorted(zip(lengths, throughputs, memories), key=lambda item: item[0])
                sorted_lengths = [pair[0] for pair in paired]
                sorted_throughputs = [pair[1] for pair in paired]
                sorted_memories = [pair[2] for pair in paired]
                label = backend.upper()
                throughput_axis.plot(
                    sorted_lengths,
                    sorted_throughputs,
                    marker=markers[backend],
                    color=colors[backend],
                    markersize=5.5,
                    linewidth=2.2,
                    label=label,
                )
                memory_axis.plot(
                    sorted_lengths,
                    sorted_memories,
                    marker=markers[backend],
                    color=colors[backend],
                    markersize=5.5,
                    linewidth=2.2,
                    label=label,
                )

            throughput_axis.set_title(f"Batch size {batch_size}: Throughput")
            throughput_axis.set_xlabel("Sequence length")
            throughput_axis.set_ylabel("Tokens/second")
            throughput_axis.grid(True, alpha=0.25)
            throughput_axis.spines["top"].set_visible(False)
            throughput_axis.spines["right"].set_visible(False)

            memory_axis.set_title(f"Batch size {batch_size}: Peak memory")
            memory_axis.set_xlabel("Sequence length")
            memory_axis.set_ylabel("Peak memory (MB)")
            memory_axis.grid(True, alpha=0.25)
            memory_axis.spines["top"].set_visible(False)
            memory_axis.spines["right"].set_visible(False)

            if row_index == 0:
                throughput_axis.legend(loc="upper left", frameon=False)
                memory_axis.legend(loc="upper left", frameon=False)

        figure.suptitle(f"{repo_id} | SDPA vs Flex | compiled={compiled_model}", y=0.995)
        figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        filename = (
            f"throughput_memory_sdpa_vs_flex_{_slugify_filename(repo_id)}"
            f"_compiled_{str(compiled_model).lower()}.png"
        )
        output_path = publication_dir / filename
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        generated_paths.append(output_path)

    return generated_paths


def _is_finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        return True
    return False


def _build_flex_sdpa_deltas(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    pair_index: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        if bool(row["pass"]) is False:
            continue
        family = str(row["family"])
        if family not in ["esm2", "esmplusplus"]:
            continue
        backend = str(row["attn_backend"])
        if backend not in ["sdpa", "flex"]:
            continue
        if _is_finite(row["tokens_per_second"]) is False:
            continue
        if _is_finite(row["peak_memory_mb"]) is False:
            continue
        pair_key = f"{row['model_key']}|{row['sequence_length']}|{row['batch_size']}|{row['compiled_model']}"
        if pair_key not in pair_index:
            pair_index[pair_key] = {}
        pair_index[pair_key][backend] = row

    delta_rows: List[Dict[str, object]] = []
    for pair_key in pair_index:
        pair = pair_index[pair_key]
        if "sdpa" not in pair or "flex" not in pair:
            continue
        sdpa_row = pair["sdpa"]
        flex_row = pair["flex"]
        sdpa_throughput = float(sdpa_row["tokens_per_second"])
        flex_throughput = float(flex_row["tokens_per_second"])
        sdpa_memory = float(sdpa_row["peak_memory_mb"])
        flex_memory = float(flex_row["peak_memory_mb"])
        assert sdpa_throughput > 0.0, f"Expected positive SDPA throughput for pair {pair_key}"
        throughput_ratio = flex_throughput / sdpa_throughput
        throughput_gain_percent = 100.0 * (throughput_ratio - 1.0)
        if sdpa_memory > 0.0:
            memory_ratio = flex_memory / sdpa_memory
            memory_reduction_percent = 100.0 * (1.0 - memory_ratio)
        else:
            memory_ratio = float("nan")
            memory_reduction_percent = float("nan")
        delta_rows.append(
            {
                "model_key": flex_row["model_key"],
                "family": flex_row["family"],
                "repo_id": flex_row["repo_id"],
                "compiled_model": flex_row["compiled_model"],
                "sequence_length": flex_row["sequence_length"],
                "batch_size": flex_row["batch_size"],
                "sdpa_tokens_per_second": sdpa_throughput,
                "flex_tokens_per_second": flex_throughput,
                "throughput_ratio_flex_over_sdpa": throughput_ratio,
                "throughput_gain_percent": throughput_gain_percent,
                "sdpa_peak_memory_mb": sdpa_memory,
                "flex_peak_memory_mb": flex_memory,
                "memory_ratio_flex_over_sdpa": memory_ratio,
                "memory_reduction_percent": memory_reduction_percent,
            }
        )
    return delta_rows


def _selected_backends(spec, args: argparse.Namespace, attn_backends: List[str]) -> List[str]:
    if args.compare_attn and spec.family in ["esm2", "esmplusplus"]:
        return list(attn_backends)
    if spec.family in ["esm2", "esmplusplus"]:
        return [args.attn_backend]
    return ["model_default"]


def _should_compile_model(spec, args: argparse.Namespace) -> bool:
    if spec.family == "e1":
        return False
    return bool(args.compile_model or args.compare_attn)


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
    total_points = 0
    for spec in specs:
        total_points += len(_selected_backends(spec=spec, args=args, attn_backends=attn_backends)) * len(lengths) * len(batch_sizes)
    progress = tqdm(total=total_points, desc="Throughput points", unit="point")

    if args.dry_run:
        for spec in specs:
            spec_backends = _selected_backends(spec=spec, args=args, attn_backends=attn_backends)

            for attn_backend in spec_backends:
                for length in lengths:
                    for batch_size in batch_sizes:
                        rows.append(
                            {
                                "model_key": spec.key,
                                "family": spec.family,
                                "repo_id": spec.repo_id,
                                "attn_backend": attn_backend,
                                "compiled_model": _should_compile_model(spec=spec, args=args),
                                "series_label": f"{spec.repo_id}|{attn_backend}|compiled={_should_compile_model(spec=spec, args=args)}",
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
                        progress.update(1)
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
            "compile_model": bool(args.compile_model or args.compare_attn),
            "pad_min_ratio": args.pad_min_ratio,
            "dry_run": True,
            "rows": rows,
        }
        write_json(output_dir / "metrics.json", payload)
        write_csv(output_dir / "metrics.csv", rows)
        summary_lines = [f"Suite: throughput (dry-run)", f"Benchmark points: {len(rows)}", f"Output directory: {output_dir}"]
        write_summary(output_dir / "summary.txt", summary_lines)
        print("\n".join(summary_lines))
        progress.close()
        return 0

    for spec in specs:
        spec_backends = _selected_backends(spec=spec, args=args, attn_backends=attn_backends)
        compile_model = _should_compile_model(spec=spec, args=args)

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
                        progress.update(1)
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

                        batches_processed = len(prepared_batches)
                        valid_tokens_processed = valid_token_count
                        padded_tokens_processed = padded_token_count
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
                    progress.update(1)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    progress.close()

    payload: Dict[str, object] = {
        "suite": "throughput",
        "all_passed": all_passed,
        "device": str(device),
        "dtype": str(dtype),
        "lengths": lengths,
        "batch_sizes": batch_sizes,
        "num_batches": args.num_batches,
        "warmup_steps": args.warmup_steps,
        "full_models": args.full_models,
        "compare_attn": args.compare_attn,
        "attn_backend": args.attn_backend,
        "attn_backends": attn_backends,
        "compile_model": bool(args.compile_model or args.compare_attn),
        "pad_min_ratio": args.pad_min_ratio,
        "rows": rows,
    }

    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "metrics.csv", rows)
    _plot_throughput(rows=rows, batch_sizes=batch_sizes, output_path=output_dir / "throughput_tokens_per_second.png")
    _plot_memory(rows=rows, batch_sizes=batch_sizes, output_path=output_dir / "throughput_peak_memory_mb.png")
    publication_plot_paths = _plot_publication_sdpa_vs_flex_per_model(rows=rows, output_dir=output_dir)
    delta_rows = _build_flex_sdpa_deltas(rows)
    write_csv(output_dir / "flex_vs_sdpa_deltas.csv", delta_rows)
    delta_payload: Dict[str, object] = {
        "suite": "throughput_flex_vs_sdpa",
        "rows": delta_rows,
    }
    write_json(output_dir / "flex_vs_sdpa_deltas.json", delta_payload)
    if len(delta_rows) > 0:
        _plot_flex_sdpa_delta(
            rows=delta_rows,
            batch_sizes=batch_sizes,
            metric_key="throughput_gain_percent",
            title_prefix="Flex Attention Throughput Gain vs SDPA",
            y_label="Throughput gain (%)",
            output_path=output_dir / "flex_vs_sdpa_throughput_gain_percent.png",
        )
        _plot_flex_sdpa_delta(
            rows=delta_rows,
            batch_sizes=batch_sizes,
            metric_key="memory_reduction_percent",
            title_prefix="Flex Attention Memory Reduction vs SDPA",
            y_label="Memory reduction (%)",
            output_path=output_dir / "flex_vs_sdpa_memory_reduction_percent.png",
        )

    passed_count = 0
    for row in rows:
        if bool(row["pass"]):
            passed_count += 1
    throughput_gain_values: List[float] = []
    memory_reduction_values: List[float] = []
    for row in delta_rows:
        throughput_gain_values.append(float(row["throughput_gain_percent"]))
        memory_reduction = float(row["memory_reduction_percent"])
        if math.isfinite(memory_reduction):
            memory_reduction_values.append(memory_reduction)

    if len(throughput_gain_values) > 0:
        mean_throughput_gain = sum(throughput_gain_values) / len(throughput_gain_values)
    else:
        mean_throughput_gain = float("nan")
    if len(memory_reduction_values) > 0:
        mean_memory_reduction = sum(memory_reduction_values) / len(memory_reduction_values)
    else:
        mean_memory_reduction = float("nan")
    summary_lines = [
        "Suite: throughput",
        f"Benchmark points: {len(rows)}",
        f"Points passed: {passed_count}",
        f"Points failed: {len(rows) - passed_count}",
        f"Flex-vs-SDPA pairs: {len(delta_rows)}",
        f"Publication plots generated: {len(publication_plot_paths)}",
        f"Publication plots directory: {output_dir / 'publication_plots'}",
        f"Mean throughput gain (%): {mean_throughput_gain}",
        f"Mean memory reduction (%): {mean_memory_reduction}",
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
    parser.add_argument("--lengths", type=str, default="64,128,256,512,1024,2048")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
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

