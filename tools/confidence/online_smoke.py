"""GPU smoke checks for online confidence training.

`throughput` measures folding, evaluation-head, and training-step time and peak memory by token
count; training steps run only up to the training token budget. `kernels` compares triangle kernel
backends and chunking on the same targets. `parity` checks the
fast atom mapping against the pilot's name-based mapping, the reconstructed head inputs against the
native confidence path of the same forward call, and whether a repeated call with the same seed
reproduces the samples. `overfit` trains on a few fixed rollouts to confirm that the losses fall and
that within-target sample ordering can be learned at all.
"""

from __future__ import annotations

import tempfile
import time
import numpy as np
import torch

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from .cache import _aligned_true_coordinates, load_folding_model
from .labels import compute_targets
from .online_training import (
    PAIR_MARGIN_EVALUATION,
    OnlineTrainingConfig,
    cross_entropy,
    head_output,
    pairwise_accuracy,
    sample_scores,
    structure,
    target_step,
)
from .rollouts import ATOM14_NAMES, chain_label, fold, use_fast_folding_kernels
from .training import HeadContext


Log = Callable[[str], None]
# (kernel backend, chunk size); the first is the setting the pilot and the parity check used.
KERNEL_SETTINGS: tuple[tuple[str | None, int | None], ...] = ((None, 32), ("cuequivariance", None), (None, None))
MIN_OVERFIT_PAIRS = 5


def closest_targets(targets: Sequence[Mapping[str, object]], token_sizes: Sequence[int], multi_chain_from: int) -> list[Mapping[str, object]]:
    """Pick one target per requested size, preferring complexes at and above `multi_chain_from` tokens."""
    chosen = []
    for size in token_sizes:
        pool = [target for target in targets if (int(target["num_chains"]) > 1) == (size >= multi_chain_from)]
        chosen.append(min(pool, key=lambda target: abs(int(target["num_tokens"]) - size)))
    return chosen


def _fast_model_and_head(model_id: str) -> tuple[torch.nn.Module, HeadContext]:
    """The frozen folding model on fast kernels and the donor head on default kernels, as v2 trains."""
    model = load_folding_model(model_id)
    use_fast_folding_kernels(model)
    return model, HeadContext(model_id)


@contextmanager
def _measured(record: dict[str, object], phase: str) -> Iterator[None]:
    """Record the wall time and peak allocated GPU memory of one phase under `phase` keys."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    yield
    torch.cuda.synchronize()
    record[f"{phase}_seconds"] = round(time.monotonic() - started, 2)
    record[f"{phase}_peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)


def throughput(
    model_id: str, pool_dir: Path, targets: Sequence[Mapping[str, object]], samples: int, max_training_tokens: int, log: Log
) -> list[dict[str, object]]:
    """Time folding, evaluation head passes, and (up to `max_training_tokens`) one training step per target."""
    model, context = _fast_model_and_head(model_id)
    config = OnlineTrainingConfig(model_id=model_id, samples_per_target=samples, targets_per_update=1)
    results = []
    for target in targets:
        result: dict[str, object] = {
            "target_id": target["target_id"],
            "num_tokens": int(target["num_tokens"]),
            "num_chains": int(target["num_chains"]),
        }
        with _measured(result, "fold"):
            rollout = fold(model, structure(pool_dir, target), samples, seed=17)
        with torch.no_grad(), _measured(result, "evaluation_head"):
            for sample in range(samples):
                head_output(context, rollout.head_inputs, rollout.x_pred, sample)
        result["sample_lddt"] = [round(item["lddt"], 4) for item in rollout.quality]
        result["sample_true_iptm"] = [round(item["true_iptm"], 4) for item in rollout.quality]
        if int(target["num_tokens"]) <= max_training_tokens:
            with _measured(result, "training_step"):
                losses = target_step(context, rollout, config)
            context.head.zero_grad(set_to_none=True)
            result |= {name: round(value, 4) for name, value in losses.items()}
        log(f"throughput {model_id}: {result}")
        results.append(result)
        del rollout
        torch.cuda.empty_cache()
    return results


def training_rate(model_id: str, pool_dir: Path, targets: Sequence[Mapping[str, object]], log: Log) -> dict[str, object]:
    """Seconds per training target, folding and stepping targets drawn the way training draws them.

    The first target warms up kernels and is left out of the mean.
    """
    model, context = _fast_model_and_head(model_id)
    config = OnlineTrainingConfig(model_id=model_id)
    seconds, tokens = [], []
    for index, target in enumerate(targets):
        torch.cuda.synchronize()
        started = time.monotonic()
        rollout = fold(model, structure(pool_dir, target), config.samples_per_target, index, config.num_loops, config.num_sampling_steps)
        target_step(context, rollout, config)
        context.head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        seconds.append(time.monotonic() - started)
        tokens.append(int(target["num_tokens"]))
        del rollout
    measured = np.array(seconds[1:])  # (targets - 1,)
    result: dict[str, object] = {
        "targets": len(measured),
        "num_sampling_steps": config.num_sampling_steps,
        "mean_seconds_per_target": float(measured.mean()),
        "standard_error_seconds": float(measured.std(ddof=1) / np.sqrt(len(measured))),
        "mean_tokens": float(np.mean(tokens[1:])),
        "seconds": [round(value, 2) for value in seconds],
        "tokens": tokens,
    }
    log(f"training rate {model_id}: {result['mean_seconds_per_target']:.2f} s per target (standard error {result['standard_error_seconds']:.2f}) at {result['mean_tokens']:.0f} mean tokens")
    return result


def kernels(model_id: str, pool_dir: Path, targets: Sequence[Mapping[str, object]], samples: int, log: Log) -> list[dict[str, object]]:
    """Time each kernel setting per target and measure how far it moves the numbers.

    Pair representations and sample quality are compared with the first setting's fold, a difference
    that also includes run-to-run variation. Head logits are compared on that one stored rollout, so
    they isolate the head kernels.
    """
    model = load_folding_model(model_id)
    context = HeadContext(model_id)
    config = OnlineTrainingConfig(model_id=model_id, samples_per_target=samples, targets_per_update=1)
    results = []
    for target in targets:
        reference: dict[str, torch.Tensor] = {}  # the first setting's head inputs and x_pred, on the CPU
        reference_logits: dict[str, torch.Tensor] = {}
        for backend, chunk_size in KERNEL_SETTINGS:
            for module in (model, context.head):
                module.set_kernel_backend(backend)
                module.set_chunk_size(chunk_size)
            result: dict[str, object] = {
                "target_id": target["target_id"],
                "num_tokens": int(target["num_tokens"]),
                "backend": backend or "reference",
                "chunk_size": chunk_size,
            }
            try:
                with _measured(result, "fold"):
                    rollout = fold(model, structure(pool_dir, target), samples, seed=17)
                with _measured(result, "training_step"):
                    target_step(context, rollout, config)
            except torch.OutOfMemoryError:
                result["out_of_memory"] = True
                log(f"kernels {model_id}: {result}")
                results.append(result)
                context.head.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            context.head.zero_grad(set_to_none=True)
            if not reference:
                # Stored on the CPU so later settings have the whole GPU.
                reference = {name: value.cpu() for name, value in rollout.head_inputs.items()} | {"x_pred": rollout.x_pred.cpu()}
            with torch.no_grad():
                inputs = {name: value.cuda() for name, value in reference.items() if name != "x_pred"}
                logits = head_output(context, inputs, reference["x_pred"].cuda(), 0)
            for name in ("plddt_logits", "pae_logits"):
                reference_logits.setdefault(name, logits[name].float().cpu())
            z = rollout.head_inputs["z"].cpu()  # (1, t, t, d_pair)
            result |= {
                "mean_sample_lddt": round(float(np.mean([item["lddt"] for item in rollout.quality])), 4),
                "z_relative_difference": float((z - reference["z"]).norm() / reference["z"].norm()),
                **{
                    f"stored_rollout_max_{name}_difference": float((logits[name].float().cpu() - reference_logits[name]).abs().max())
                    for name in ("plddt_logits", "pae_logits")
                },
            }
            log(f"kernels {model_id}: {result}")
            results.append(result)
            del rollout, inputs, logits
            torch.cuda.empty_cache()
    return results


def _pilot_structure_file(path: Path, sequences: Sequence[str], positions: np.ndarray) -> None:
    """Write the pilot's normalized structure format: atom14 names and 1-based residue numbers."""
    names = np.full((len(positions), 14), "", dtype="U4")  # (l, 14)
    chain_index, residue_index, offset = [], [], 0
    for chain, sequence in enumerate(sequences):
        for position, letter in enumerate(sequence):
            names[offset + position, : len(ATOM14_NAMES[letter])] = ATOM14_NAMES[letter]
        chain_index += [chain] * len(sequence)
        residue_index += list(range(1, len(sequence) + 1))
        offset += len(sequence)
    np.savez(path, coordinates=positions, atom_names=names, chain_index=np.array(chain_index), residue_index=np.array(residue_index))


@torch.no_grad()
def parity(model_id: str, pool_dir: Path, targets: Sequence[Mapping[str, object]], log: Log) -> list[dict[str, object]]:
    model, context = _fast_model_and_head(model_id)
    results = []
    from fastplms.models.esmfold2.esmfold2_input_builder import ProteinInput, StructurePredictionInput

    for target in targets:
        target_structure = structure(pool_dir, target)
        # The native head must score the very samples whose inputs `fold` reconstructs.
        original_head, original_enabled = model.confidence_head, model.config.confidence_head.enabled
        model.confidence_head, model.config.confidence_head.enabled = context.head, True
        try:
            rollout = fold(model, target_structure, 1, seed=17, native_confidence=True)
        finally:
            model.confidence_head, model.config.confidence_head.enabled = original_head, original_enabled
        repeat = fold(model, target_structure, 1, seed=17)
        inputs = StructurePredictionInput(
            sequences=[ProteinInput(id=chain_label(index), sequence=sequence) for index, sequence in enumerate(target_structure.sequences)]
        )
        features, chain_infos = model.prepare_structure_input(inputs, seed=17)
        features_cpu = {name: value.detach().cpu() for name, value in features.items()}
        record = {"chains": [{"id": chain_label(index), "sequence": sequence} for index, sequence in enumerate(target_structure.sequences)]}
        with tempfile.TemporaryDirectory() as work:
            path = Path(work, "structure.npz")
            _pilot_structure_file(path, target_structure.sequences, target_structure.positions)
            pilot_true, pilot_resolved = _aligned_true_coordinates(features_cpu, chain_infos, record, path, rollout.x_pred[0].cpu())
        ours = rollout.true_coords[0].cpu()  # (a, 3)
        atom_mask = features_cpu["atom_attention_mask"].reshape(-1).bool()
        our_resolved = torch.isfinite(ours).all(-1) & atom_mask
        both = our_resolved & pilot_resolved
        coordinate_difference = float((ours[both] - pilot_true[both]).abs().max()) if both.any() else 0.0
        pilot_targets = compute_targets(
            rollout.x_pred[0].cpu(),
            pilot_true,
            pilot_resolved,
            features_cpu["atom_to_token"].reshape(-1).long(),
            rollout.layout.backbone_indices,
            features_cpu["token_attention_mask"].reshape(-1).bool(),
        )

        native = rollout.native_confidence
        assert native is not None
        online = head_output(context, rollout.head_inputs, rollout.x_pred, 0)
        result = {
            "target_id": target["target_id"],
            "num_chains": int(target["num_chains"]),
            "resolved_atoms_equal": bool(torch.equal(our_resolved, pilot_resolved)),
            "max_true_coordinate_difference": coordinate_difference,
            "plddt_score_max_difference": float(
                (rollout.targets[0]["plddt_score"].cpu() - pilot_targets["plddt_score"]).abs().nan_to_num().max()
            ),
            "native_head_max_plddt_logit_difference": float((native["plddt_logits"].float() - online["plddt_logits"].float()).abs().max()),
            "native_head_max_pae_logit_difference": float((native["pae_logits"].float() - online["pae_logits"].float()).abs().max()),
            "repeat_call_max_coordinate_difference": float((repeat.x_pred - rollout.x_pred).abs().max()),
            "repeat_call_max_z_difference": float((repeat.head_inputs["z"] - rollout.head_inputs["z"]).abs().max()),
        }
        log(f"parity {model_id}: {result}")
        results.append(result)
    return results


def _training_metrics(context: HeadContext, rollouts: Sequence[object]) -> dict[str, float]:
    plddt_ce, pae_ce, correct, total, interface_correct, interface_total = [], [], 0, 0, 0, 0
    with torch.no_grad():
        for rollout in rollouts:
            scores = []
            for sample in range(rollout.x_pred.shape[0]):  # type: ignore[attr-defined]
                output = head_output(context, rollout.head_inputs, rollout.x_pred, sample)  # type: ignore[attr-defined]
                losses = cross_entropy(output, rollout.targets[sample], 1.0)  # type: ignore[attr-defined]
                plddt_ce.append(float(losses[0]))
                pae_ce.append(float(losses[1]))
                scores.append([float(score[0]) for score in sample_scores(output, rollout.targets[sample], rollout.head_inputs)])  # type: ignore[attr-defined]
            quality = rollout.quality  # type: ignore[attr-defined]
            pair = pairwise_accuracy([score[0] for score in scores], [item["lddt"] for item in quality], PAIR_MARGIN_EVALUATION)
            correct, total = correct + pair[0], total + pair[1]
            if rollout.num_chains > 1:  # type: ignore[attr-defined]
                pair = pairwise_accuracy([score[1] for score in scores], [item["true_iptm"] for item in quality], PAIR_MARGIN_EVALUATION)
                interface_correct, interface_total = interface_correct + pair[0], interface_total + pair[1]
    return {
        "plddt_ce": float(np.mean(plddt_ce)),
        "pae_ce": float(np.mean(pae_ce)),
        "plddt_pair_accuracy": correct / total if total else float("nan"),
        "plddt_pairs": float(total),
        "iptm_pair_accuracy": interface_correct / interface_total if interface_total else float("nan"),
        "iptm_pairs": float(interface_total),
        "pair_accuracy": (correct + interface_correct) / (total + interface_total) if total + interface_total else float("nan"),
        "pairs": float(total + interface_total),
    }


def overfit(model_id: str, pool_dir: Path, targets: Sequence[Mapping[str, object]], updates: int, log: Log) -> dict[str, object]:
    model, context = _fast_model_and_head(model_id)
    config = OnlineTrainingConfig(model_id=model_id, targets_per_update=len(targets), learning_rate=3e-4)
    rollouts = [fold(model, structure(pool_dir, target), config.samples_per_target, seed=100 + index) for index, target in enumerate(targets)]
    del model
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(context.head.parameters(), lr=config.learning_rate, weight_decay=0.0)
    context.head.train()
    initial = _training_metrics(context, rollouts)
    log(f"overfit {model_id} initial: {initial}")
    history = [{"update": 0, **initial}]
    for update in range(1, updates + 1):
        for rollout in rollouts:
            target_step(context, rollout, config)
        torch.nn.utils.clip_grad_norm_(context.head.parameters(), config.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if update % 50 == 0 or update == updates:
            metrics = _training_metrics(context, rollouts)
            history.append({"update": update, **metrics})
            log(f"overfit {model_id} update {update}: {metrics}")
    final = history[-1]
    return {
        "initial": initial,
        "final": final,
        "history": history,
        # Small targets rarely have pLDDT pairs beyond the margin, so ranking counts pLDDT and ipTM pairs together.
        "passed": final["plddt_ce"] < initial["plddt_ce"]
        and final["pae_ce"] < initial["pae_ce"]
        and final["pairs"] >= MIN_OVERFIT_PAIRS
        and final["pair_accuracy"] > 0.9,
    }
