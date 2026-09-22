"""Evaluate confidence heads on the untouched v2 test set.

Each test target is folded once with five diffusion samples at the inference defaults of three
recycling loops and 50 diffusion steps, and every candidate head scores the
same samples, so head comparisons do not depend on sampling noise. Structure quality uses all-atom
lDDT, TM-score from TM-align on C-alpha atoms, and DockQ over native interfaces for multi-chain
targets. Production `esmfold2` is evaluated separately on its own samples with its own head.
"""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import warnings
import numpy as np
import torch

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file

from .labels import _masked_cross_entropy
from .online_training import PAIR_MARGIN_EVALUATION, head_output, pairwise_accuracy, spearman, structure
from .ranking import expected_mean_plddt, expected_tm_scores
from .rollouts import ATOM14_NAMES, INFERENCE_LOOPS, INFERENCE_SAMPLING_STEPS, ONE_TO_THREE, Rollout, chain_label, fold


EVALUATION_SAMPLES = 5
EVALUATION_SEED_OFFSET = 1000
DOCKQ_MARGIN = 0.05
CALIBRATION_BINS = 10
DISORDER_BINS = 50  # residue pLDDT histogram resolution for the disorder AUROC
BOOTSTRAP_SAMPLES = 1000
LONG_STRATUM = "long"

Log = Callable[[str], None]


@dataclass(frozen=True)
class SampleStructures:
    """CPU inputs for TM-score and DockQ of one diffusion sample."""

    predicted_ca: np.ndarray  # (n_ca, 3) resolved C-alpha atoms in chain order
    true_ca: np.ndarray  # (n_ca, 3)
    ca_sequence: str
    predicted_pdb: str
    native_pdb: str
    chain_ids: tuple[str, ...]


def _pdb_text(coordinates: np.ndarray, atoms: Sequence[tuple[int, int, str, int, str]]) -> str:
    """Format atoms given as (atom index, chain, residue letter, residue number, atom name)."""
    lines = []
    for serial, (atom_index, chain, letter, residue_number, name) in enumerate(atoms, start=1):
        x, y, z = (float(value) for value in coordinates[atom_index])
        lines.append(
            f"ATOM  {serial:5d} {name:>4s} {ONE_TO_THREE[letter]:>3s} {chain_label(chain):>1s}{residue_number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}"
        )
    return "\n".join(lines) + "\nEND\n"


def sample_structures(rollout: Rollout, sequences: Sequence[str], sample: int) -> SampleStructures:
    predicted = rollout.x_pred[sample].cpu().numpy()  # (a, 3)
    true = rollout.true_coords[sample].cpu().numpy()  # (a, 3)
    resolved = np.isfinite(true).all(-1)  # (a,)
    starts = np.cumsum([0, *[len(sequence) for sequence in sequences]])
    true_index = rollout.layout.true_index.numpy()  # (a,)
    atoms, ca_atoms, ca_letters = [], [], []
    for atom_index in np.flatnonzero(resolved & (true_index >= 0)):
        residue, slot = divmod(int(true_index[atom_index]), 14)
        chain = int(np.searchsorted(starts, residue, side="right") - 1)
        letter = sequences[chain][residue - starts[chain]]
        name = ATOM14_NAMES[letter][slot]
        atoms.append((int(atom_index), chain, letter, residue - int(starts[chain]) + 1, name))
        if name == "CA":
            ca_atoms.append(int(atom_index))
            ca_letters.append(letter)
    return SampleStructures(
        predicted_ca=predicted[ca_atoms],
        true_ca=true[ca_atoms],
        ca_sequence="".join(ca_letters),
        predicted_pdb=_pdb_text(predicted, atoms),
        native_pdb=_pdb_text(true, atoms),
        chain_ids=tuple(chain_label(index) for index in range(len(sequences))),
    )


def structure_scores(structures: SampleStructures) -> dict[str, float | None]:
    """TM-score of C-alpha atoms and, for complexes, DockQ averaged over native interfaces."""
    from tmtools import tm_align

    # DockQ leaves its PDB readers open; the warning would repeat for every sample in worker logs.
    warnings.filterwarnings("ignore", category=ResourceWarning)

    alignment = tm_align(structures.predicted_ca, structures.true_ca, structures.ca_sequence, structures.ca_sequence)
    scores: dict[str, float | None] = {"tm_score": float(alignment.tm_norm_chain2), "dockq": None}
    # PDB chain ids hold one character, so DockQ is skipped for complexes above 26 chains.
    if 1 < len(structures.chain_ids) <= 26:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

        with tempfile.TemporaryDirectory() as work:
            model_path, native_path = Path(work, "model.pdb"), Path(work, "native.pdb")
            model_path.write_text(structures.predicted_pdb, encoding="ascii")
            native_path.write_text(structures.native_pdb, encoding="ascii")
            interfaces, total = run_on_all_native_interfaces(
                load_PDB(str(model_path)),
                load_PDB(str(native_path)),
                chain_map={chain: chain for chain in structures.chain_ids},
            )
        # DockQ returns the sum over native interfaces; report the mean.
        scores["dockq"] = float(total) / len(interfaces) if interfaces else None
    return scores


@torch.no_grad()
def head_sample_predictions(context: object, rollout: Rollout, sample: int) -> dict[str, object]:
    """Summaries and calibration inputs of one head on one sample."""
    output = head_output(context, rollout.head_inputs, rollout.x_pred, sample)  # type: ignore[arg-type]
    return _summaries(output["plddt_logits"], output["pae_logits"], rollout, sample)


def _summaries(plddt_logits: torch.Tensor, pae_logits: torch.Tensor, rollout: Rollout, sample: int) -> dict[str, object]:
    targets = rollout.targets[sample]
    inputs = rollout.head_inputs
    atom_mask = inputs["atom_attention_mask"].reshape(1, -1)  # (1, a)
    per_atom = (plddt_logits.float().softmax(-1) * ((torch.arange(50, device=plddt_logits.device) + 0.5) / 50)).sum(-1)[0]  # (a,)
    ptm, iptm = expected_tm_scores(pae_logits, inputs["asym_id"], inputs["token_attention_mask"])
    labeled = targets["plddt_mask"]  # (a,)
    atom_plddt, atom_lddt = per_atom[labeled], targets["plddt_score"][labeled].float()  # (n,), (n,)
    # Per-bin sums let bootstrap draws add samples instead of concatenating millions of atoms.
    bins = (atom_plddt * CALIBRATION_BINS).long().clamp(0, CALIBRATION_BINS - 1)  # (n,)
    bin_zeros = torch.zeros(CALIBRATION_BINS, device=atom_plddt.device)  # (bins,)
    # Residues whose C-alpha is missing from the experimental structure mark likely disorder. No head
    # receives pLDDT labels there, so low confidence on them has to generalize.
    ca_atoms = torch.cat(rollout.layout.chain_ca).to(per_atom.device)  # (r,)
    residue_plddt = per_atom[ca_atoms]  # (r,)
    unresolved = ~torch.isfinite(rollout.true_coords[sample][ca_atoms]).all(-1)  # (r,)
    residue_bins = (residue_plddt * DISORDER_BINS).long().clamp(0, DISORDER_BINS - 1)  # (r,)
    return {
        "mean_plddt": float(expected_mean_plddt(plddt_logits, atom_mask)[0]),
        "mean_plddt_labeled": float(expected_mean_plddt(plddt_logits, labeled[None])[0]),
        "ptm": float(ptm[0]),
        "iptm": float(iptm[0]),
        "plddt_ce": float(_masked_cross_entropy(plddt_logits[0], targets["plddt_target"], labeled)),
        "pae_ce": float(_masked_cross_entropy(pae_logits[0], targets["pae_target"], targets["pae_mask"])),
        "absolute_error_sum": float((atom_plddt - atom_lddt).abs().sum()),
        "calibration_count": torch.bincount(bins, minlength=CALIBRATION_BINS).tolist(),
        "calibration_predicted_sum": bin_zeros.clone().index_add_(0, bins, atom_plddt).tolist(),
        "calibration_true_sum": bin_zeros.clone().index_add_(0, bins, atom_lddt).tolist(),
        "resolved_residue_histogram": torch.bincount(residue_bins[~unresolved], minlength=DISORDER_BINS).tolist(),
        "unresolved_residue_histogram": torch.bincount(residue_bins[unresolved], minlength=DISORDER_BINS).tolist(),
        "resolved_residue_plddt_sum": float(residue_plddt[~unresolved].sum()),
        "unresolved_residue_plddt_sum": float(residue_plddt[unresolved].sum()),
    }


def fold_and_score(
    model: torch.nn.Module,
    pool_dir: Path,
    targets: Sequence[Mapping[str, object]],
    heads: Mapping[str, object] | None,
    output_path: Path,
    log: Log,
    workers: int = 32,
) -> list[dict[str, object]]:
    """Fold every target, score each sample with each head (or the model's own head), and add quality."""
    records: list[dict[str, object]] = []
    futures = []
    skipped: list[str] = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        for index, target in enumerate(targets):
            sequences = list(target["sequences"])  # type: ignore[arg-type]
            try:
                rollout = fold(
                    model,
                    structure(pool_dir, target),
                    EVALUATION_SAMPLES,
                    EVALUATION_SEED_OFFSET + index,
                    INFERENCE_LOOPS,
                    INFERENCE_SAMPLING_STEPS,
                    native_confidence=heads is None,
                )
            except torch.OutOfMemoryError:
                # The largest targets can exhaust the device; the gates then compare heads on the
                # targets every evaluation kept, rather than losing the whole run.
                skipped.append(str(target["target_id"]))
                log(f"skipped {target['target_id']} ({target['num_tokens']} tokens): out of memory")
                torch.cuda.empty_cache()
                continue
            for sample in range(EVALUATION_SAMPLES):
                if heads is None:
                    native = rollout.native_confidence or {}
                    predictions = {
                        "production": _summaries(native["plddt_logits"][sample : sample + 1], native["pae_logits"][sample : sample + 1], rollout, sample)
                    }
                else:
                    predictions = {name: head_sample_predictions(head, rollout, sample) for name, head in heads.items()}
                records.append(
                    {
                        "target_id": target["target_id"],
                        "stratum": target["stratum"],
                        "num_chains": len(sequences),
                        "num_tokens": int(target["num_tokens"]),
                        "sample": sample,
                        **{f"true_{name}": value for name, value in rollout.quality[sample].items()},
                        "predictions": predictions,
                    }
                )
                futures.append(executor.submit(structure_scores, sample_structures(rollout, sequences, sample)))
            if (index + 1) % 25 == 0:
                log(f"folded {index + 1}/{len(targets)} test targets")
        for record, future in zip(records, futures, strict=True):
            record.update(future.result())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records) + "\n", encoding="utf-8")
    output_path.with_name("skipped.json").write_text(json.dumps(skipped) + "\n", encoding="utf-8")
    if skipped:
        log(f"skipped {len(skipped)} of {len(targets)} targets that did not fit in memory")
    return records


def _metrics(records: Sequence[Mapping[str, object]], head: str) -> dict[str, float]:
    by_target: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_target[str(record["target_id"])].append(record)
    predictions = [record["predictions"][head] for record in records]  # type: ignore[index]
    counts = np.sum([values["calibration_count"] for values in predictions], axis=0)  # (bins,)
    predicted_sums = np.sum([values["calibration_predicted_sum"] for values in predictions], axis=0)  # (bins,)
    true_sums = np.sum([values["calibration_true_sum"] for values in predictions], axis=0)  # (bins,)
    # Atom-weighted mean over bins of |mean predicted pLDDT - mean true lDDT|; empty bins add zero.
    calibration = np.abs(predicted_sums - true_sums).sum() / counts.sum()
    resolved = np.sum([values["resolved_residue_histogram"] for values in predictions], axis=0)  # (bins,)
    unresolved = np.sum([values["unresolved_residue_histogram"] for values in predictions], axis=0)  # (bins,)
    resolved_total, unresolved_total = float(resolved.sum()), float(unresolved.sum())
    both = resolved_total > 0 and unresolved_total > 0
    # Probability that an unresolved residue has lower pLDDT than a resolved one; shared bins count half.
    resolved_above = resolved_total - np.cumsum(resolved)  # (bins,) resolved residues in higher bins
    disorder_auroc = float((unresolved * (resolved_above + 0.5 * resolved)).sum() / (unresolved_total * resolved_total)) if both else float("nan")
    below_half = DISORDER_BINS // 2  # bins under pLDDT 0.5
    complexes = [record for record in records if int(record["num_chains"]) > 1 and record["dockq"] is not None]
    plddt_correct = plddt_total = dockq_correct = dockq_total = 0
    regret, random_regret = [], []
    complexes_without_dockq = 0
    for samples in by_target.values():
        scores = [sample["predictions"][head] for sample in samples]  # type: ignore[index]
        correct, total = pairwise_accuracy([score["mean_plddt"] for score in scores], [sample["true_lddt"] for sample in samples], PAIR_MARGIN_EVALUATION)
        plddt_correct, plddt_total = plddt_correct + correct, plddt_total + total
        if int(samples[0]["num_chains"]) == 1:
            quality = [float(sample["true_lddt"]) for sample in samples]
            ranking = [score["mean_plddt"] for score in scores]  # ESMFold2 selects monomers by pLDDT
        elif all(sample["dockq"] is not None for sample in samples):
            quality = [float(sample["dockq"]) for sample in samples]
            ranking = [score["iptm"] for score in scores]  # and complexes by ipTM
            correct, total = pairwise_accuracy(ranking, quality, DOCKQ_MARGIN)
            dockq_correct, dockq_total = dockq_correct + correct, dockq_total + total
        else:
            # DockQ is undefined above 26 chains or without a native interface, so the complex has
            # no selection quality; every head and the reference exclude the same targets.
            complexes_without_dockq += 1
            continue
        regret.append(max(quality) - quality[int(np.argmax(ranking))])
        random_regret.append(max(quality) - float(np.mean(quality)))
    return {
        "targets": float(len(by_target)),
        "plddt_lddt_spearman": spearman([record["predictions"][head]["mean_plddt"] for record in records], [record["true_lddt"] for record in records]),  # type: ignore[index]
        "ptm_tm_spearman": spearman([record["predictions"][head]["ptm"] for record in records], [record["tm_score"] for record in records]),  # type: ignore[index]
        "iptm_dockq_spearman": spearman([record["predictions"][head]["iptm"] for record in complexes], [record["dockq"] for record in complexes]) if complexes else float("nan"),  # type: ignore[index]
        "atom_plddt_mae": float(sum(values["absolute_error_sum"] for values in predictions) / counts.sum()),
        "calibration_error_10bin": float(calibration),
        "plddt_ce": float(np.mean([values["plddt_ce"] for values in predictions])),
        "pae_ce": float(np.mean([values["pae_ce"] for values in predictions])),
        "within_target_plddt_accuracy": plddt_correct / plddt_total if plddt_total else float("nan"),
        "within_target_plddt_pairs": float(plddt_total),
        "within_target_iptm_dockq_accuracy": dockq_correct / dockq_total if dockq_total else float("nan"),
        "within_target_iptm_dockq_pairs": float(dockq_total),
        "top1_regret": float(np.mean(regret)),
        "random_selection_regret": float(np.mean(random_regret)),
        "complexes_without_dockq": float(complexes_without_dockq),
        "disorder_auroc": disorder_auroc,
        "resolved_residue_mean_plddt": sum(values["resolved_residue_plddt_sum"] for values in predictions) / resolved_total if resolved_total else float("nan"),
        "unresolved_residue_mean_plddt": sum(values["unresolved_residue_plddt_sum"] for values in predictions) / unresolved_total if unresolved_total else float("nan"),
        "resolved_fraction_below_50": float(resolved[:below_half].sum()) / resolved_total if resolved_total else float("nan"),
        "unresolved_fraction_below_50": float(unresolved[:below_half].sum()) / unresolved_total if unresolved_total else float("nan"),
        "unresolved_residues": unresolved_total,
    }


def bootstrap_records(by_target: Mapping[str, Sequence[Mapping[str, object]]], rng: np.random.Generator) -> list[dict[str, object]]:
    """Resample whole targets with replacement; each draw gets its own target id so repeats stay separate."""
    target_ids = sorted(by_target)
    return [
        {**record, "target_id": f"{target_ids[index]}#{draw}"}
        for draw, index in enumerate(rng.integers(len(target_ids), size=len(target_ids)))
        for record in by_target[target_ids[index]]
    ]


def summarize(records: Sequence[Mapping[str, object]], heads: Sequence[str]) -> dict[str, object]:
    """Standard-set estimates with target-bootstrap 95% intervals, plus per-stratum estimates.

    The long stratum lies beyond the training length, so it is reported only in `by_stratum`.
    """
    rng = np.random.default_rng(0)
    standard = [record for record in records if record["stratum"] != LONG_STRATUM]
    by_target: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in standard:
        by_target[str(record["target_id"])].append(record)
    strata = sorted({str(record["stratum"]) for record in records})
    summary: dict[str, object] = {}
    for head in heads:
        draws = defaultdict(list)
        for _ in range(BOOTSTRAP_SAMPLES):
            for name, value in _metrics(bootstrap_records(by_target, rng), head).items():
                draws[name].append(value)
        summary[head] = {
            "overall": _metrics(standard, head),
            "interval_95": {name: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))] for name, values in draws.items()},
            "by_stratum": {name: _metrics([record for record in records if record["stratum"] == name], head) for name in strata},
        }
    return summary


def load_heads(model_id: str, head_files: Mapping[str, Path | None]) -> dict[str, object]:
    """Instantiate one head per name; a `None` path keeps the pinned donor weights."""
    from .training import HeadContext

    heads = {}
    for name, path in head_files.items():
        context = HeadContext(model_id)
        if path is not None:
            context.head.load_state_dict(load_file(str(path)))
        context.head.eval()
        heads[name] = context
    return heads
