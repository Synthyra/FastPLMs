import entrypoint_setup

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


SUPPORTED_BACKENDS = ("sdpa", "flex", "kernels_flash")


class ThroughputChecker:
    def __init__(
        self,
        warmup_batches: int = 10,
        timed_batches: int = 100,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_batches = warmup_batches
        self.timed_batches = timed_batches
        self.canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()
        return model

    def _generate_random_sequence(self, length: int) -> str:
        return "M" + "".join(random.choices(self.canonical_amino_acids, k=length - 1))

    def _generate_random_batch(self, batch_size: int, min_length: int, max_length: int) -> List[str]:
        max_length_example = self._generate_random_sequence(max_length)
        return [max_length_example] + [
            self._generate_random_sequence(random.randint(min_length, max_length))
            for _ in range(batch_size - 1)
        ]

    @torch.inference_mode()
    def _time(self, model: torch.nn.Module, tokenizer: object, batch_size: int, min_length: int, max_length: int) -> Tuple[float, int]:
        model = model.to(self.device).eval()
        set_seed(42)
        min_dynamic_warmup_batches = self.warmup_batches
        max_dynamic_warmup_batches = self.warmup_batches * 10
        stability_window = 3
        relative_stability_tolerance = 0.10

        def synchronize():
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        def run_one_batch() -> int:
            batch = self._generate_random_batch(batch_size, min_length, max_length)
            tokenized = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
            )
            input_ids = tokenized["input_ids"]
            if "attention_mask" in tokenized:
                nonpad_tokens_this = tokenized["attention_mask"].sum().item()
            else:
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is not None:
                    nonpad_tokens_this = (input_ids != pad_token_id).sum().item()
                else:
                    nonpad_tokens_this = input_ids.numel()
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            _ = model(**tokenized, output_hidden_states=True)
            return nonpad_tokens_this

        def time_batches(num_batches: int, message: str):
            processed_tokens = 0
            synchronize()
            start_time = time.time()
            for _ in tqdm(range(num_batches), desc=message, leave=False):
                processed_tokens += run_one_batch()
            synchronize()
            end_time = time.time()
            return end_time - start_time, processed_tokens

        # Compile first, then keep warming up until the compiled path stabilizes.
        model = torch.compile(model)
        warmup_latencies = []
        for warmup_idx in tqdm(range(max_dynamic_warmup_batches), desc="Warmup (dynamic)", leave=False):
            synchronize()
            warmup_start = time.time()
            _ = run_one_batch()
            synchronize()
            warmup_latency = time.time() - warmup_start
            warmup_latencies.append(warmup_latency)

            if warmup_idx + 1 < min_dynamic_warmup_batches:
                continue
            if len(warmup_latencies) < 2 * stability_window:
                continue

            previous_window = warmup_latencies[-2 * stability_window:-stability_window]
            current_window = warmup_latencies[-stability_window:]
            previous_mean = sum(previous_window) / stability_window
            current_mean = sum(current_window) / stability_window
            assert previous_mean > 0.0, "Warmup latency mean should be positive."
            relative_change = abs(current_mean - previous_mean) / previous_mean
            if relative_change <= relative_stability_tolerance:
                break

        time_taken, timed_tokens_sum = time_batches(self.timed_batches, "Timed")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return time_taken, timed_tokens_sum

    def evaluate(self, model_path: str, batch_sizes: List[int], min_length: int, sequence_lengths: List[int], backends: List[str]) -> Dict[str, Dict[Tuple[int, int], Dict[str, float]]]:
        results = {backend: {} for backend in backends}

        original_model = self._load_model(model_path)
        tokenizer = original_model.tokenizer

        for backend in backends:
            print(f"Benchmarking {model_path} with backend={backend}")
            try:
                backend_model = copy.deepcopy(original_model)
                backend_model.attn_backend = backend
            except AssertionError as error:
                print(f"Skipping backend '{backend}' for {model_path}: {error}")
                continue

            for bs in batch_sizes:
                for max_length in sequence_lengths:
                    model_copy = copy.deepcopy(backend_model)
                    time_taken, tokens = self._time(
                        model_copy,
                        tokenizer,
                        bs,
                        min_length,
                        max_length,
                    )
                    results[backend][(bs, max_length)] = {"time": time_taken, "tokens": tokens}

        original_model.cpu()
        del original_model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return results


def plot_results(all_results: Dict[str, Dict], output_path: str) -> None:
    sns.set_theme(style="whitegrid")
    plot_data = []

    for model_path, results in all_results.items():
        model_name = Path(model_path).name
        for backend in sorted(results.keys()):
            for (bs, max_length), entry in results[backend].items():
                time_taken = entry["time"]
                nonpad_tokens = entry["tokens"]
                tokens_per_sec = nonpad_tokens / time_taken if time_taken > 0 else 0.0
                plot_data.append(
                    {
                        "Model": model_name,
                        "Backend": backend,
                        "Batch": bs,
                        "SeqLen": max_length,
                        "TokensPerSec": tokens_per_sec,
                        "NonPadTokens": nonpad_tokens,
                        "Seconds": time_taken,
                    }
                )

    if not plot_data:
        return

    plot_df = pd.DataFrame(plot_data)
    sequence_lengths = sorted(plot_df["SeqLen"].dropna().unique().tolist())

    plot = sns.relplot(
        data=plot_df,
        x="SeqLen",
        y="TokensPerSec",
        hue="Backend",
        style="Batch",
        kind="line",
        marker="o",
        dashes=False,
        col="Model",
        col_wrap=1,
        height=4.5,
        aspect=1.5,
        facet_kws={"sharey": False},
    )
    plot.set_titles("{col_name}")
    plot.set(xticks=sequence_lengths)
    plot.set_axis_labels("Sequence length", "Non-pad tokens/s")
    plot.figure.suptitle("Throughput comparison by model")
    plot.tight_layout()
    plot.figure.subplots_adjust(top=0.93, right=0.95, bottom=0.06)
    plot.add_legend(title="Backend / Batch")
    plt.savefig(output_path, dpi=300)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # On Windows, use "%cd%" instead of "${PWD}" to get the current working directory:
    # docker run --gpus all -v "%cd%":/workspace fastplms python -m testing.throughput
    # On Linux/macOS, keep using ${PWD}:
    # docker run --gpus all -v ${PWD}:/workspace fastplms python -m testing.throughput
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument(
        "--model_paths",
        nargs="+",
        default=["Synthyra/ESM2-8M", "Synthyra/ESMplusplus_small"],
    )
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--sequence_lengths", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--backends", nargs="+", choices=SUPPORTED_BACKENDS, default=list(SUPPORTED_BACKENDS))
    parser.add_argument("--min_length", type=int, default=32)
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--timed_batches", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="throughput_comparison.png")
    args = parser.parse_args()

    if args.hf_token:
        from huggingface_hub import login

        login(token=args.hf_token)

    checker = ThroughputChecker(warmup_batches=args.warmup_batches, timed_batches=args.timed_batches)

    all_results = {}
    for model_path in args.model_paths:
        all_results[model_path] = checker.evaluate(
            model_path,
            args.batch_sizes,
            min_length=args.min_length,
            sequence_lengths=args.sequence_lengths,
            backends=args.backends,
        )

    plot_results(all_results, args.output_path)
