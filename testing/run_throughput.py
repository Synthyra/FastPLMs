import entrypoint_setup

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

from testing.common import (
    add_base_args,
    add_data_args,
    build_output_dir,
    ensure_dir,
    generate_sequences,
    load_model,
    LOAD_DTYPE,
    login_if_needed,
    maybe_tokenizer_for_embedding,
    resolve_device,
    resolve_runtime_dtype,
    set_seed,
)
from testing.model_registry import get_model_specs
from testing.reporting import write_csv, write_json, write_summary


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


def _parse_pad_fraction_list(values: str) -> List[float]:
    parsed: List[float] = []
    for chunk in values.split(","):
        value = float(chunk.strip())
        assert 0.0 <= value <= 1.0, f"Expected pad fraction in [0, 1], got {value}."
        parsed.append(value)
    assert len(parsed) > 0, "Expected at least one pad fraction."
    return parsed


def _random_sequence(length: int, rng: random.Random) -> str:
    return "M" + "".join(rng.choices(CANONICAL_AMINO_ACIDS, k=length - 1))


def _build_sequence_batches(
    num_batches: int,
    batch_size: int,
    length: int,
    padded_sequence_fraction: float,
    max_pad_fraction: float,
    seed: int,
) -> List[List[str]]:
    assert 0.0 <= padded_sequence_fraction <= 1.0, "Expected padded_sequence_fraction to be in [0, 1]."
    assert 0.0 <= max_pad_fraction <= 1.0, "Expected max_pad_fraction to be in [0, 1]."
    min_padded_length = max(4, int(math.ceil(length * (1.0 - max_pad_fraction))))
    min_padded_length = min(min_padded_length, length)
    can_pad = min_padded_length < length
    rng = random.Random(seed)
    batches: List[List[str]] = []
    for _ in range(num_batches):
        lengths: List[int] = []
        for _ in range(batch_size):
            if can_pad and rng.random() < padded_sequence_fraction:
                lengths.append(rng.randint(min_padded_length, length - 1))
            else:
                lengths.append(length)
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


def _plot_sdpa_vs_flex_per_model(rows: List[Dict[str, object]], output_dir: pathlib.Path) -> List[pathlib.Path]:
    per_model_dir = output_dir / "per_model_plots"
    per_model_dir.mkdir(parents=True, exist_ok=True)

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
        model_group_key = (
            f"{row['repo_id']}|pad={row['padded_sequence_fraction_setting']}"
        )
        if model_group_key not in model_rows_index:
            model_rows_index[model_group_key] = []
        model_rows_index[model_group_key].append(row)

    generated_paths: List[pathlib.Path] = []
    metric_configs = [
        ("tokens_per_second", "Tokens/second", "throughput"),
        ("peak_memory_mb", "Peak memory (MB)", "memory"),
    ]
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
        padded_sequence_fraction_setting = float(model_rows[0]["padded_sequence_fraction_setting"])
        colors = {"sdpa": "#1f77b4", "flex": "#d62728"}
        markers = {"sdpa": "o", "flex": "s"}

        for metric_key, y_label, filename_prefix in metric_configs:
            figure, axes = plt.subplots(
                nrows=len(batch_sizes),
                ncols=1,
                figsize=(7.5, max(3.8, 3.2 * len(batch_sizes))),
                squeeze=False,
            )
            for row_index, batch_size in enumerate(batch_sizes):
                axis = axes[row_index][0]
                for backend in ["sdpa", "flex"]:
                    lengths: List[int] = []
                    metric_values: List[float] = []
                    for row in model_rows:
                        if int(row["batch_size"]) == batch_size and str(row["attn_backend"]) == backend:
                            lengths.append(int(row["sequence_length"]))
                            metric_values.append(float(row[metric_key]))
                    if len(lengths) == 0:
                        continue
                    paired = sorted(zip(lengths, metric_values), key=lambda item: item[0])
                    sorted_lengths = [pair[0] for pair in paired]
                    sorted_values = [pair[1] for pair in paired]
                    axis.plot(
                        sorted_lengths,
                        sorted_values,
                        marker=markers[backend],
                        color=colors[backend],
                        markersize=5.5,
                        linewidth=2.2,
                        label=backend.upper(),
                    )
                axis.set_title(f"Batch size {batch_size}")
                axis.set_xlabel("Sequence length")
                axis.set_ylabel(y_label)
                axis.grid(True, alpha=0.25)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                if row_index == 0:
                    axis.legend(loc="upper left", frameon=False)

            figure.suptitle(
                (
                    f"{repo_id} | {filename_prefix.title()} | SDPA vs Flex | "
                    f"pad_setting={padded_sequence_fraction_setting}"
                ),
                y=0.995,
            )
            figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
            filename = (
                f"{filename_prefix}_sdpa_vs_flex_{_slugify_filename(repo_id)}"
                f"_pad_{_slugify_filename(str(padded_sequence_fraction_setting))}.png"
            )
            output_path = per_model_dir / filename
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
        if family not in ["esm2", "esmplusplus", "dplm", "dplm2"]:
            continue
        backend = str(row["attn_backend"])
        if backend not in ["sdpa", "flex"]:
            continue
        if _is_finite(row["tokens_per_second"]) is False:
            continue
        if _is_finite(row["peak_memory_mb"]) is False:
            continue
        pair_key = (
            f"{row['model_key']}|{row['sequence_length']}|{row['batch_size']}"
            f"|{row['padded_sequence_fraction_setting']}"
        )
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
        if sdpa_throughput > 0.0:
            throughput_ratio = flex_throughput / sdpa_throughput
            throughput_gain_percent = 100.0 * (throughput_ratio - 1.0)
        else:
            throughput_ratio = float("nan")
            throughput_gain_percent = float("nan")
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
                "padded_sequence_fraction_setting": flex_row["padded_sequence_fraction_setting"],
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


def _selected_backends(spec) -> List[str]:
    if spec.family in ["esm2", "esmplusplus", "dplm", "dplm2"]:
        return ["sdpa", "flex"]
    raise ValueError(f"Unexpected family for throughput compare: {spec.family}")


def run_throughput_suite(args: argparse.Namespace) -> int:
    login_if_needed(args.token)
    device = resolve_device(args.device)
    runtime_dtype = resolve_runtime_dtype()
    set_seed(args.seed)
    output_dir = build_output_dir(args.output_dir, "throughput")

    lengths = parse_int_list(args.lengths)
    batch_sizes = parse_int_list(args.batch_sizes)
    if args.pad_fractions is None:
        pad_fraction_settings = [args.padded_sequence_fraction]
    else:
        pad_fraction_settings = _parse_pad_fraction_list(args.pad_fractions)
    default_benchmark_families = ["esm2", "esmplusplus", "dplm", "dplm2"]
    if args.families is None:
        benchmark_families = default_benchmark_families
    else:
        benchmark_families = []
        for family in args.families:
            if family in default_benchmark_families:
                benchmark_families.append(family)
    assert len(benchmark_families) > 0, (
        "Throughput supports families: "
        f"{default_benchmark_families}, got {args.families}."
    )
    specs = get_model_specs(full_models=args.full_models, families=benchmark_families)
    assert len(specs) > 0, "Expected at least one model spec for throughput benchmarking."
    rows: List[Dict[str, object]] = []
    all_passed = True
    total_points = 0
    for spec in specs:
        total_points += (
            len(_selected_backends(spec=spec))
            * len(pad_fraction_settings)
            * len(lengths)
            * len(batch_sizes)
        )
    progress = tqdm(total=total_points, desc="Throughput points", unit="point")

    if args.dry_run:
        for spec in specs:
            spec_backends = _selected_backends(spec=spec)

            for attn_backend in spec_backends:
                for padded_sequence_fraction_setting in pad_fraction_settings:
                    for length in lengths:
                        for batch_size in batch_sizes:
                            rows.append(
                                {
                                    "model_key": spec.key,
                                    "family": spec.family,
                                    "repo_id": spec.repo_id,
                                    "attn_backend": attn_backend,
                                    "padded_sequence_fraction_setting": padded_sequence_fraction_setting,
                                    "flex_path_used": bool(attn_backend == "flex"),
                                    "series_label": (
                                        f"{spec.repo_id}|{attn_backend}|pad={padded_sequence_fraction_setting}"
                                    ),
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
            "load_dtype": str(LOAD_DTYPE),
            "runtime_dtype": str(runtime_dtype),
            "lengths": lengths,
            "batch_sizes": batch_sizes,
            "num_batches": args.num_batches,
            "warmup_steps": args.warmup_steps,
            "families": benchmark_families,
            "attn_backends": sorted(list({str(row["attn_backend"]) for row in rows})),
            "padded_sequence_fraction": args.padded_sequence_fraction,
            "pad_fraction_settings": pad_fraction_settings,
            "max_pad_fraction": args.max_pad_fraction,
            "dry_run": True,
            "rows": rows,
        }
        delta_rows = _build_flex_sdpa_deltas(rows)
        write_json(output_dir / "metrics.json", payload)
        write_csv(output_dir / "metrics.csv", rows)
        write_json(output_dir / "flex_vs_sdpa_deltas.json", {"suite": "throughput", "rows": delta_rows, "dry_run": True})
        summary_lines = [
            "Suite: throughput (dry-run)",
            f"Benchmark points: {len(rows)}",
            f"Flex-vs-SDPA pairs: {len(delta_rows)}",
            f"Output directory: {output_dir}",
        ]
        write_summary(output_dir / "summary.txt", summary_lines)
        print("\n".join(summary_lines))
        progress.close()
        return 0

    for spec in specs:
        spec_backends = _selected_backends(spec=spec)

        for attn_backend in spec_backends:
            selected_backend = None if attn_backend == "model_default" else attn_backend
            print(
                f"[throughput] Testing {spec.repo_id} "
                f"(attn_backend={attn_backend}) on {device} with runtime {runtime_dtype}"
            )
            try:
                model, tokenizer = load_model(
                    spec=spec,
                    task="base",
                    device=device,
                    runtime_dtype=runtime_dtype,
                    attn_backend=selected_backend,
                )
            except Exception as exc:
                all_passed = False
                for padded_sequence_fraction_setting in pad_fraction_settings:
                    for length in lengths:
                        for batch_size in batch_sizes:
                            rows.append(
                                {
                                    "model_key": spec.key,
                                    "family": spec.family,
                                    "repo_id": spec.repo_id,
                                    "attn_backend": attn_backend,
                                    "padded_sequence_fraction_setting": padded_sequence_fraction_setting,
                                    "flex_path_used": False,
                                    "series_label": (
                                        f"{spec.repo_id}|{attn_backend}|pad={padded_sequence_fraction_setting}"
                                    ),
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

            for padded_sequence_fraction_setting in pad_fraction_settings:
                for length in lengths:
                    for batch_size in batch_sizes:
                        row: Dict[str, object] = {
                            "model_key": spec.key,
                            "family": spec.family,
                            "repo_id": spec.repo_id,
                            "attn_backend": attn_backend,
                            "padded_sequence_fraction_setting": padded_sequence_fraction_setting,
                            "flex_path_used": False,
                            "series_label": (
                                f"{spec.repo_id}|{attn_backend}|pad={padded_sequence_fraction_setting}"
                            ),
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
                            pad_seed_offset = int(round(padded_sequence_fraction_setting * 10000))
                            batch_seed = args.seed + (pad_seed_offset * 1000000) + (length * 1000) + batch_size
                            sampled_sequence_length = length
                            if spec.family in ["esm2", "esmplusplus", "dplm", "dplm2"]:
                                sampled_sequence_length = max(4, length - 2)
                            sequence_batches = _build_sequence_batches(
                                num_batches=args.num_batches,
                                batch_size=batch_size,
                                length=sampled_sequence_length,
                                padded_sequence_fraction=padded_sequence_fraction_setting,
                                max_pad_fraction=args.max_pad_fraction,
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

                            warmup_steps = max(args.warmup_steps, 4)

                            with torch.no_grad():
                                for i in range(warmup_steps):
                                    _ = run_forward(
                                        spec=spec,
                                        model=model,
                                        batch=prepared_batches[i % len(prepared_batches)],
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
                            row["flex_path_used"] = bool(attn_backend == "flex")
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
        "load_dtype": str(LOAD_DTYPE),
        "runtime_dtype": str(runtime_dtype),
        "lengths": lengths,
        "batch_sizes": batch_sizes,
        "num_batches": args.num_batches,
        "warmup_steps": args.warmup_steps,
        "families": benchmark_families,
        "attn_backends": sorted(list({str(row["attn_backend"]) for row in rows})),
        "padded_sequence_fraction": args.padded_sequence_fraction,
        "pad_fraction_settings": pad_fraction_settings,
        "max_pad_fraction": args.max_pad_fraction,
        "rows": rows,
    }

    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "metrics.csv", rows)
    delta_rows = _build_flex_sdpa_deltas(rows)
    write_json(output_dir / "flex_vs_sdpa_deltas.json", {"suite": "throughput", "rows": delta_rows})
    per_model_plot_paths = _plot_sdpa_vs_flex_per_model(rows=rows, output_dir=output_dir)

    passed_count = 0
    for row in rows:
        if bool(row["pass"]):
            passed_count += 1
    summary_lines = [
        "Suite: throughput",
        f"Benchmark points: {len(rows)}",
        f"Points passed: {passed_count}",
        f"Points failed: {len(rows) - passed_count}",
        f"Flex-vs-SDPA pairs: {len(delta_rows)}",
        f"Per-model plots generated: {len(per_model_plot_paths)}",
        f"Per-model plots directory: {output_dir / 'per_model_plots'}",
        f"Output directory: {output_dir}",
    ]
    for row in rows:
        status = "PASS" if bool(row["pass"]) else "FAIL"
        summary_lines.append(
            (
                f"{status} | {row['repo_id']} | backend={row['attn_backend']} "
                f"| flex_path_used={row['flex_path_used']} | pad_setting={row['padded_sequence_fraction_setting']} "
                f"| len={row['sequence_length']} | bs={row['batch_size']} | pad_fraction={row['pad_fraction']} "
                f"| tok_s={row['tokens_per_second']} | mem_mb={row['peak_memory_mb']} | error={row['error']}"
            )
        )
    write_summary(output_dir / "summary.txt", summary_lines)
    print("\n".join(summary_lines))

    if all_passed:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model throughput benchmarks.")
    add_base_args(parser)
    parser.add_argument("--lengths", type=str, default="64,128,256,512,1024,2048")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--padded-sequence-fraction", type=float, default=0.3)
    parser.add_argument("--pad-fractions", type=str, default=None)
    parser.add_argument("--max-pad-fraction", type=float, default=0.5)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_throughput_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())

