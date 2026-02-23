import time
import torch
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForMaskedLM
from tqdm.auto import tqdm


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ThroughputChecker:
    def __init__(
        self,
        warmup_batches: int = 10,
        timed_batches: int = 100,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_batches = warmup_batches
        self.timed_batches = timed_batches
        self.canonical_amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def _load_model(self, model_path: str):
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()
        return model

    def _generate_random_sequence(self, length: int) -> str:
        return 'M' + "".join(random.choices(self.canonical_amino_acids, k=length))
    
    def _generate_random_batch(self, batch_size: int, min_length: int, max_length: int) -> list[str]:
        return [self._generate_random_sequence(random.randint(min_length, max_length)) for _ in range(batch_size)]

    @torch.inference_mode()
    def _time(self, model, tokenizer, batch_size: int, min_length: int, max_length: int):
        set_seed(42)
        def time_batches(num_batches: int, message: str):
            start_time = time.time()
            for _ in tqdm(range(num_batches), desc=message, leave=False):
                batch = self._generate_random_batch(batch_size, min_length, max_length)
                tokenized = tokenizer(batch, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                _ = model(**tokenized, output_hidden_states=True)
            end_time = time.time()
            return end_time - start_time

        # Warmup and timing
        torch.compile(model)
        _ = time_batches(self.warmup_batches, "Warmup")
        torch.cuda.synchronize()
        time_taken = time_batches(self.timed_batches, "Timed")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return time_taken

    def evaluate(self, model_path: str, batch_sizes: list[int], min_length: int, sequence_lengths: List[int]):
        results = {"sdpa": {}, "flex": {}}
        
        # Benchmark SDPA
        model = self._load_model(model_path)
        tokenizer = model.tokenizer
        model.attn_backend = "sdpa"
        
        print(f"Benchmarking {model_path} with SDPA...")
        for bs in batch_sizes:
            for max_length in sequence_lengths:
                t = self._time(model, tokenizer, bs, min_length, max_length)
                results["sdpa"][(bs, max_length)] = t
        
        # Benchmark Flex
        model.attn_backend = "flex"
        print(f"Benchmarking {model_path} with Flex...")
        for bs in batch_sizes:
            for max_length in sequence_lengths:
                t = self._time(model, tokenizer, bs, min_length, max_length)
                results["flex"][(bs, max_length)] = t
        
        del model
        torch.cuda.empty_cache()
        return results


def plot_results(all_results: dict, output_path: str):
    sns.set_theme(style="whitegrid")
    plot_data = []
    
    for model_path, results in all_results.items():
        model_name = model_path.split("/")[-1]
        for backend in ["sdpa", "flex"]:
            for (bs, max_length), t in results[backend].items():
                plot_data.append({
                    "Model": f"{model_name} ({backend})",
                    "Combination": f"BS={bs}, L={max_length}",
                    "Time (s)": t,
                    "BS": bs,
                    "L": max_length
                })
    
    if not plot_data:
        return

    plt.figure(figsize=(12, 6))
    # Sort by BS then L for the x-axis labels
    combinations = sorted(list(set(d["Combination"] for d in plot_data)), 
                         key=lambda x: (int(x.split("=")[1].split(",")[0]), int(x.split("=")[2].split(",")[0])))
    
    sns.barplot(data=plot_data, x="Combination", y="Time (s)", hue="Model", order=combinations)
    plt.title("Model Throughput Comparison: SDPA vs Flex")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--model_paths", nargs="+", default=["Synthyra/ESM2-8M", "Synthyra/ESMplusplus_small"])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--sequence_lengths", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--timed_batches", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="throughput_comparison.png")
    args = parser.parse_args()

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    checker = ThroughputChecker(warmup_batches=args.warmup_batches, timed_batches=args.timed_batches)
    
    all_results = {}
    for path in args.model_paths:
        all_results[path] = checker.evaluate(path, args.batch_sizes, min_length=32, sequence_lengths=args.sequence_lengths)
    
    plot_results(all_results, args.output_path)
