"""Count parameters for all models in the registry.

Loads each model from HuggingFace and prints total / trainable parameter counts.
Run on a machine with enough VRAM or use --cpu to load on CPU (slow for large models).

Usage:
    py -m testing.count_parameters
    py -m testing.count_parameters --families esm2 ankh
    py -m testing.count_parameters --cpu
    py -m testing.count_parameters --include-structure
"""

import argparse
import json
import sys

import torch
from transformers import AutoModel, AutoModelForMaskedLM

SEQUENCE_MODELS = {
    # ESM2
    "esm2_8m": "Synthyra/ESM2-8M",
    "esm2_35m": "Synthyra/ESM2-35M",
    "esm2_150m": "Synthyra/ESM2-150M",
    "esm2_650m": "Synthyra/ESM2-650M",
    "esm2_3b": "Synthyra/ESM2-3B",
    # ESM++
    "esmc_small": "Synthyra/ESMplusplus_small",
    "esmc_large": "Synthyra/ESMplusplus_large",
    # E1
    "e1_150m": "Synthyra/Profluent-E1-150M",
    "e1_300m": "Synthyra/Profluent-E1-300M",
    "e1_600m": "Synthyra/Profluent-E1-600M",
    # DPLM
    "dplm_150m": "Synthyra/DPLM-150M",
    "dplm_650m": "Synthyra/DPLM-650M",
    "dplm_3b": "Synthyra/DPLM-3B",
    # DPLM2
    "dplm2_150m": "Synthyra/DPLM2-150M",
    "dplm2_650m": "Synthyra/DPLM2-650M",
    "dplm2_3b": "Synthyra/DPLM2-3B",
    # ANKH
    "ankh_base": "Synthyra/ANKH_base",
    "ankh_large": "Synthyra/ANKH_large",
    "ankh2_large": "Synthyra/ANKH2_large",
    "ankh3_large": "Synthyra/ANKH3_large",
    "ankh3_xl": "Synthyra/ANKH3_xl",
}

STRUCTURE_MODELS = {
    "boltz2": "Synthyra/Boltz2",
    "esmfold": "Synthyra/FastESMFold",
}

FAMILY_PREFIXES = {
    "esm2": "esm2_",
    "esmc": "esmc_",
    "esm++": "esmc_",
    "e1": "e1_",
    "dplm2": "dplm2_",
    "dplm": "dplm_",
    "ankh": "ankh",
    "boltz": "boltz",
    "esmfold": "esmfold",
}


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def count_params(repo_id: str, device: str) -> dict:
    model = AutoModelForMaskedLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        dtype=torch.float32,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Deduplicate tied parameters (e.g. shared embedding/lm_head)
    seen = set()
    unique_total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            unique_total += p.numel()

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "total": total,
        "trainable": trainable,
        "unique": unique_total,
        "total_str": _format_params(total),
        "unique_str": _format_params(unique_total),
    }


def count_structure_params(repo_id: str, device: str) -> dict:
    model = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=True,
        dtype=torch.float32,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    seen = set()
    unique_total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            unique_total += p.numel()

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "total": total,
        "trainable": trainable,
        "unique": unique_total,
        "total_str": _format_params(total),
        "unique_str": _format_params(unique_total),
    }


def main():
    parser = argparse.ArgumentParser(description="Count parameters for FastPLMs models")
    parser.add_argument(
        "--families",
        nargs="*",
        help="Filter to specific families (e.g. esm2 ankh dplm)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Load models on CPU instead of CUDA",
    )
    parser.add_argument(
        "--include-structure",
        action="store_true",
        help="Also count structure models (Boltz2, ESMFold)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    models_to_count = {}
    if args.families:
        for family in args.families:
            prefix = FAMILY_PREFIXES[family.lower()]
            for key, repo in SEQUENCE_MODELS.items():
                if key.startswith(prefix):
                    models_to_count[key] = repo
            if args.include_structure:
                for key, repo in STRUCTURE_MODELS.items():
                    if key.startswith(prefix):
                        models_to_count[key] = repo
    else:
        models_to_count.update(SEQUENCE_MODELS)
        if args.include_structure:
            models_to_count.update(STRUCTURE_MODELS)

    results = {}
    structure_keys = set(STRUCTURE_MODELS.keys())

    print(f"{'Model Key':<20} {'Repo ID':<35} {'Total':>12} {'Unique':>12}")
    print("-" * 83)

    for key, repo in models_to_count.items():
        try:
            if key in structure_keys:
                info = count_structure_params(repo, device)
            else:
                info = count_params(repo, device)
            results[key] = {"repo_id": repo, **info}
            print(f"{key:<20} {repo:<35} {info['total_str']:>12} {info['unique_str']:>12}")
        except Exception as e:
            print(f"{key:<20} {repo:<35} {'ERROR':>12} {str(e)[:30]}", file=sys.stderr)
            results[key] = {"repo_id": repo, "error": str(e)}

    if args.json:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
