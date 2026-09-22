"""Fold proteins of increasing length with one ESMFold2 implementation and report cost.

Runs as a plain script under either interpreter: the candidate environment
(``--implementation fastplms``) or the pinned official environment
(``--implementation upstream``). It therefore imports nothing from this
repository at module scope. One process loads one model, so peak memory and
timings never mix two implementations.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import torch

from collections.abc import Callable
from pathlib import Path
from typing import Any


# The model comes from one of two unrelated packages that share method names, so
# this script can only type it dynamically.
FoldingModel = Any

RESULT_PREFIX = "FOLD_BENCH_RESULT "
# Default of ``DiffusionStructureHead.sample`` in the official source and in FastPLMs.
OFFICIAL_MAX_INFERENCE_SIGMA = 256.0
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Well-characterized single-chain proteins for the dense-versus-windowed comparison.
REAL_PROTEINS = {
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "t4_lysozyme": (
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDS"
        "LDAVRRAAINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
    ),
    "gfp": (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEG"
        "YVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY"
        "QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    ),
}
# Phases are timed by wrapping these attributes when the model defines them.
PHASE_ATTRIBUTES = (
    ("language_model", "_compute_lm_hidden_states"),
    ("trunk_loop", "_run_one_loop"),
    ("coda", "parcae_coda"),
    ("diffusion_sampling", "structure_head.sample"),
    ("confidence", "confidence_head"),
)


def synthetic_sequence(length: int) -> str:
    """A fixed pseudo-random protein; fold cost depends on length, not on content."""
    generator = random.Random(length)
    return "".join(generator.choice(AMINO_ACIDS) for _ in range(length))


def _checkpoint_tensor_names(snapshot: Path) -> set[str]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        return set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    from safetensors import safe_open

    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def load_official_model(args: argparse.Namespace) -> FoldingModel:
    """Load the official model with the backbone revision its pinned source was written for.

    The official loader fetches the head of the backbone repository, whose tensor
    names have since changed; Transformers then initializes every unmatched weight
    at random and only warns. Pin the revision and refuse any unmatched weight.
    """
    from huggingface_hub import snapshot_download
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(args.repo, revision=args.revision, load_esmc=False)
    model = model.to("cuda").eval()
    backbone = snapshot_download(
        args.backbone_repo,
        revision=args.backbone_revision,
        allow_patterns=["*.json", "*.safetensors"],
    )
    model.load_esmc(backbone)
    checkpoint_names = _checkpoint_tensor_names(Path(backbone))
    missing = sorted(
        name
        for name in model._esmc.state_dict()
        if f"esmc.{name}" not in checkpoint_names and name not in checkpoint_names
    )
    if missing:
        raise RuntimeError(f"The official backbone left weights uninitialized: {missing[:8]}")
    return model


def load_model(args: argparse.Namespace) -> FoldingModel:
    if args.implementation == "upstream":
        return load_official_model(args)
    from fastplms.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from fastplms.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel

    model_class = ESMFold2ExperimentalModel if args.experimental else ESMFold2Model
    model = model_class.from_pretrained(args.repo, revision=args.revision)
    model = model.to("cuda").eval()
    if args.atom_attention != "dense":
        model.set_atom_attention(args.atom_attention)
    # Both implementations default to 64-row chunks in the pair-update blocks.
    if args.unchunked:
        model.set_chunk_size(None)
    elif args.chunk_size is not None:
        model.set_chunk_size(args.chunk_size)
    return model


def _resolve(model: FoldingModel, dotted_name: str) -> tuple[Any, str] | None:
    owner: Any = model
    *parents, leaf = dotted_name.split(".")
    for parent in parents:
        owner = getattr(owner, parent, None)
    if owner is None or getattr(owner, leaf, None) is None:
        return None
    return owner, leaf


class PhaseTimer:
    """Accumulates CUDA time per wrapped model phase for one fold."""

    def __init__(self, model: FoldingModel) -> None:
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._restore: list[tuple[Any, str, Any, bool]] = []
        for phase, dotted_name in PHASE_ATTRIBUTES:
            resolved = _resolve(model, dotted_name)
            if resolved is None:
                continue
            owner, leaf = resolved
            target = getattr(owner, leaf)
            is_module = isinstance(target, torch.nn.Module)
            original = target.forward if is_module else target
            wrapped = self._timed(phase, original)
            if is_module:
                target.forward = wrapped
                self._restore.append((target, "forward", original, True))
            else:
                setattr(owner, leaf, wrapped)
                self._restore.append((owner, leaf, original, False))

    def _timed(self, phase: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            stop = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            result = function(*args, **kwargs)
            stop.record()
            self._events.setdefault(phase, []).append((start, stop))
            return result

        return timed

    def close(self) -> dict[str, float]:
        torch.cuda.synchronize()
        for owner, leaf, _original, _is_module in self._restore:
            # Instance attributes shadowed the class attribute; removing them restores it.
            delattr(owner, leaf)
        return {
            phase: sum(start.elapsed_time(stop) for start, stop in events) / 1000.0
            for phase, events in self._events.items()
        }


def fold(model: FoldingModel, sequence: str, seed: int, **settings: Any) -> dict[str, Any]:
    torch.manual_seed(seed)
    with torch.no_grad():
        output: dict[str, Any] = model.infer_protein(sequence, **settings)
    return output


def timed_fold(model: FoldingModel, sequence: str, **settings: Any) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    fold(model, sequence, seed=0, **settings)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def measure_length(
    model: FoldingModel, length: int, repeats: int, settings: dict[str, Any], single_pass: bool
) -> dict[str, Any]:
    """Time folds of one length and record their peak memory and phase times.

    ``single_pass`` is for folds that take minutes: one fold is both timed and
    phase-instrumented, with no warm-up, whose cost would be a fraction of a percent.
    """
    sequence = synthetic_sequence(length)
    if not single_pass:
        # A shortened fold visits every kernel shape, so timed folds see a warm allocator.
        fold(model, sequence, seed=0, **{**settings, "num_loops": 1, "num_sampling_steps": 4})
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    static_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if single_pass:
        timer = PhaseTimer(model)
        seconds = [timed_fold(model, sequence, **settings)]
        phases = timer.close()
    else:
        seconds = [timed_fold(model, sequence, **settings) for _ in range(repeats)]
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if not single_pass:
        timer = PhaseTimer(model)
        fold(model, sequence, seed=0, **settings)
        phases = timer.close()
    return {
        "length": length,
        "status": "ok",
        "single_pass": single_pass,
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "residues_per_second": length / statistics.median(seconds),
        "static_bytes": static_bytes,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "phase_seconds": phases,
    }


def peak_allocations(
    model: FoldingModel, length: int, settings: dict[str, Any], limit: int = 25
) -> dict[str, Any]:
    """Which tensors are alive when one fold reaches its peak allocated memory.

    Replays the allocator trace of one fold, finds the moment of peak allocation,
    and groups the blocks alive then by the model source line that allocated them.
    """
    sequence = synthetic_sequence(length)
    fold(model, sequence, seed=0, **{**settings, "num_loops": 1, "num_sampling_steps": 4})
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(max_entries=4_000_000, stacks="python")
    base_bytes = torch.cuda.memory_allocated()
    fold(model, sequence, seed=0, **settings)
    torch.cuda.synchronize()
    events = torch.cuda.memory._snapshot()["device_traces"][0]
    torch.cuda.memory._record_memory_history(enabled=None)

    def source_line(frames: list[dict[str, Any]]) -> str:
        for frame in frames:
            if "esmfold2" in frame["filename"] or "esm_plusplus" in frame["filename"]:
                name = frame["filename"].rsplit("/", 1)[-1]
                return f"{name}:{frame['line']} {frame['name']}"
        return "outside the model source"

    live: dict[int, tuple[int, str]] = {}
    current = peak = 0
    peak_blocks: dict[int, tuple[int, str]] = {}
    for event in events:
        if event["action"] == "alloc":
            live[event["addr"]] = (event["size"], source_line(event["frames"]))
            current += event["size"]
            if current > peak:
                peak, peak_blocks = current, dict(live)
        elif event["action"] == "free_completed" and event["addr"] in live:
            current -= live.pop(event["addr"])[0]
    by_line: dict[str, int] = {}
    for size, line in peak_blocks.values():
        by_line[line] = by_line.get(line, 0) + size
    ranked = sorted(by_line.items(), key=lambda item: item[1], reverse=True)[:limit]
    return {
        "length": length,
        "bytes_before_fold": base_bytes,
        "traced_peak_bytes": peak,
        "pair_tensor_fp32_bytes": length * length * 256 * 4,
        "live_at_peak": [{"source": line, "bytes": size} for line, size in ranked],
    }


def representative_coordinates(
    model: FoldingModel, sequence: str, output: dict[str, Any]
) -> torch.Tensor:
    """Return one representative atom per residue: (l, 3) in FP32 on the CPU."""
    module_name = type(model).__module__.rsplit(".", 1)[0] + ".protein_utils"
    features = __import__(module_name, fromlist=["prepare_protein_features"])
    atom_index = features.prepare_protein_features(sequence)["distogram_atom_idx"].reshape(-1)  # (l,)
    coordinates = output["sample_atom_coords"].float().reshape(-1, 3)  # (samples * atoms, 3), first sample first
    representative: torch.Tensor = coordinates[atom_index.to(coordinates.device)].cpu()  # (l, 3)
    return representative  # (l, 3)


def aligned_rmsd(first: torch.Tensor, second: torch.Tensor) -> float:
    """Root-mean-square deviation after optimal rigid superposition (Kabsch)."""
    # first, second: (l, 3), one representative atom per residue.
    first = first.double() - first.double().mean(0)  # (l, 3)
    second = second.double() - second.double().mean(0)  # (l, 3)
    left, _, right = torch.linalg.svd(first.T @ second)  # left/right: (3, 3); singular values: (3,)
    reflection = torch.sign(torch.linalg.det(left @ right))  # ()
    correction = torch.diag(torch.tensor([1.0, 1.0, float(reflection)], dtype=torch.float64))  # (3, 3)
    rotated = first @ (left @ correction @ right)  # (l, 3)
    return float((rotated - second).pow(2).sum(-1).mean().sqrt())


def compare_atom_attention(model: FoldingModel, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Windowed against dense on real proteins, with seed-to-seed dense spread as the scale."""
    records = []
    for name, sequence in REAL_PROTEINS.items():
        folds: dict[str, dict[str, Any]] = {}
        for label, mode, seed in (
            ("dense_seed0", "dense", 0),
            ("dense_seed1", "dense", 1),
            ("windowed_seed0", "windowed", 0),
            ("windowed_seed1", "windowed", 1),
        ):
            model.set_atom_attention(mode)
            folds[label] = fold(model, sequence, seed=seed, **settings)
        coordinates = {
            label: representative_coordinates(model, sequence, output)
            for label, output in folds.items()
        }
        record: dict[str, Any] = {
            "protein": name,
            "length": len(sequence),
            "rmsd_dense_seed0_vs_windowed_seed0": aligned_rmsd(
                coordinates["dense_seed0"], coordinates["windowed_seed0"]
            ),
            "rmsd_dense_seed0_vs_dense_seed1": aligned_rmsd(
                coordinates["dense_seed0"], coordinates["dense_seed1"]
            ),
            "rmsd_windowed_seed0_vs_windowed_seed1": aligned_rmsd(
                coordinates["windowed_seed0"], coordinates["windowed_seed1"]
            ),
        }
        for label, output in folds.items():
            if output.get("plddt") is not None:
                record[f"mean_plddt_{label}"] = float(output["plddt"].float().mean())
        records.append(record)
    model.set_atom_attention("dense")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("fastplms", "upstream"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--backbone-repo", help="official backbone, for --implementation upstream")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--atom-attention", choices=("dense", "windowed"), default="dense")
    parser.add_argument("--compare-atom-attention", action="store_true")
    parser.add_argument("--unchunked", action="store_true")
    parser.add_argument("--peak-allocations", action="store_true")
    parser.add_argument("--chunk-size", type=int, help="pair-update chunk rows, if not the default")
    parser.add_argument(
        "--single-pass-from",
        type=int,
        help="from this length on, time one instrumented fold without a warm-up",
    )
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    args = parser.parse_args()

    settings: dict[str, Any] = {
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "num_diffusion_samples": args.num_diffusion_samples,
    }
    if args.implementation == "fastplms":
        # The official forward cannot change its sampler's noise cap, which drops the
        # schedule entries above it. FastPLMs forwards ``None`` (no cap) by default and
        # would run more denoising steps, so request the official schedule explicitly.
        settings["max_inference_sigma"] = OFFICIAL_MAX_INFERENCE_SIGMA
    load_started = time.perf_counter()
    model = load_model(args)
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "label": args.label,
        "implementation": args.implementation,
        "repo": args.repo,
        "revision": args.revision,
        "atom_attention": args.atom_attention,
        "chunk_size": None if args.unchunked else (args.chunk_size or "default"),
        "settings": settings,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "load_seconds": time.perf_counter() - load_started,
        "weights_bytes": torch.cuda.memory_allocated(),
        "lengths": [],
    }
    if args.peak_allocations:
        result["peak_allocations"] = [
            peak_allocations(model, length, settings) for length in args.lengths
        ]
        args.lengths = []
    for length in args.lengths:
        try:
            single_pass = args.single_pass_from is not None and length >= args.single_pass_from
            result["lengths"].append(
                measure_length(model, length, args.repeats, settings, single_pass)
            )
        except torch.OutOfMemoryError:
            # Longer proteins need more memory still, so the sweep ends here.
            result["lengths"].append({"length": length, "status": "out_of_memory"})
            torch.cuda.empty_cache()
            break
    if args.compare_atom_attention:
        result["atom_attention_comparison"] = compare_atom_attention(model, settings)
    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
