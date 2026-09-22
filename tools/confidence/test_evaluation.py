"""Evaluate confidence heads on the spent v2 test set or a validation dry run.

Each test target is folded once with five diffusion samples at the inference defaults of three
recycling loops and 50 diffusion steps, and every candidate head scores the
same samples, so head comparisons do not depend on sampling noise. Structure quality uses all-atom
lDDT, TM-score from TM-align on C-alpha atoms, and DockQ over native interfaces for multi-chain
targets. Production `esmfold2` is evaluated separately on its own samples with its own head.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import tempfile
import warnings
import numpy as np
import torch

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file

from .experiment_artifacts import write_new_json
from .labels import _masked_cross_entropy
from .online_training import head_output, structure
from .ranking import expected_mean_plddt, expected_tm_scores
from .rollouts import (
    ATOM14_NAMES,
    INFERENCE_LOOPS,
    INFERENCE_SAMPLING_STEPS,
    ONE_TO_THREE,
    Rollout,
    chain_label,
    fold,
)
from .v2_analysis import (
    BOOTSTRAP_SAMPLES as BOOTSTRAP_SAMPLES,
    CALIBRATION_BINS,
    DISORDER_BINS,
    DOCKQ_MARGIN as DOCKQ_MARGIN,
    LONG_STRATUM as LONG_STRATUM,
    PAIR_MARGIN_EVALUATION as PAIR_MARGIN_EVALUATION,
    bootstrap_records as bootstrap_records,
    pairwise_accuracy as pairwise_accuracy,
    sample_metrics,
    spearman as spearman,
    summarize as summarize,
)


EVALUATION_SAMPLES = 5
EVALUATION_SEED_OFFSET = 1000
_metrics = sample_metrics  # compatibility for callers of the earlier evaluation module

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
    # coordinates: (atoms, 3); indexing one atom produces its (3,) xyz vector.
    lines = []
    for serial, (atom_index, chain, letter, residue_number, name) in enumerate(atoms, start=1):
        x, y, z = (float(value) for value in coordinates[atom_index])
        lines.append(
            f"ATOM  {serial:5d} {name:>4s} {ONE_TO_THREE[letter]:>3s} "
            f"{chain_label(chain):>1s}{residue_number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}"
        )
    return "\n".join(lines) + "\nEND\n"


def sample_structures(rollout: Rollout, sequences: Sequence[str], sample: int) -> SampleStructures:
    predicted = rollout.x_pred[sample].cpu().numpy()  # (a, 3)
    true = rollout.true_coords[sample].cpu().numpy()  # (a, 3)
    resolved = np.isfinite(true).all(-1)  # (a,)
    starts = np.cumsum([0, *[len(sequence) for sequence in sequences]])  # (chains + 1,)
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

    alignment = tm_align(
        structures.predicted_ca,
        structures.true_ca,
        structures.ca_sequence,
        structures.ca_sequence,
    )
    scores: dict[str, float | None] = {
        "tm_score": float(alignment.tm_norm_chain2),
        "dockq": None,
    }
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


def _summaries(
    plddt_logits: torch.Tensor, pae_logits: torch.Tensor, rollout: Rollout, sample: int
) -> dict[str, object]:
    # plddt_logits: (1, a, 50); pae_logits: (1, t, t, 64).
    # a: padded atoms; t: padded tokens; n: labeled atoms; r: real residues.
    targets = rollout.targets[sample]
    inputs = rollout.head_inputs
    atom_mask = inputs["atom_attention_mask"].reshape(1, -1)  # (1, a)
    per_atom = (
        plddt_logits.float().softmax(-1)
        * ((torch.arange(50, device=plddt_logits.device) + 0.5) / 50)
    ).sum(-1)[0]  # (a,)
    ptm, iptm = expected_tm_scores(
        pae_logits, inputs["asym_id"], inputs["token_attention_mask"]
    )  # each (1,)
    labeled = targets["plddt_mask"]  # (a,)
    atom_plddt, atom_lddt = (
        per_atom[labeled],
        targets["plddt_score"][labeled].float(),
    )  # (n,), (n,)
    # Per-bin sums let bootstrap draws add samples instead of concatenating millions of atoms.
    bins = (atom_plddt * CALIBRATION_BINS).long().clamp(0, CALIBRATION_BINS - 1)  # (n,)
    bin_zeros = torch.zeros(CALIBRATION_BINS, device=atom_plddt.device)  # (bins,)
    # Missing experimental C-alpha atoms mark likely disorder. No head receives pLDDT labels
    # there, so low confidence on those residues has to generalize.
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
        "pae_ce": float(
            _masked_cross_entropy(pae_logits[0], targets["pae_target"], targets["pae_mask"])
        ),
        "absolute_error_sum": float((atom_plddt - atom_lddt).abs().sum()),
        "calibration_count": torch.bincount(bins, minlength=CALIBRATION_BINS).tolist(),
        "calibration_predicted_sum": bin_zeros.clone().index_add_(0, bins, atom_plddt).tolist(),
        "calibration_true_sum": bin_zeros.clone().index_add_(0, bins, atom_lddt).tolist(),
        "resolved_residue_histogram": torch.bincount(
            residue_bins[~unresolved], minlength=DISORDER_BINS
        ).tolist(),
        "unresolved_residue_histogram": torch.bincount(
            residue_bins[unresolved], minlength=DISORDER_BINS
        ).tolist(),
        "resolved_residue_plddt_sum": float(residue_plddt[~unresolved].sum()),
        "unresolved_residue_plddt_sum": float(residue_plddt[unresolved].sum()),
    }


def _write_partial_target(
    directory: Path, index: int, target_id: str, records: list[dict[str, object]]
) -> None:
    """Publish one complete target atomically without marking the evaluation complete."""
    path = directory / f"{index:06d}.json"
    if path.exists():
        raise FileExistsError(f"Partial target already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    write_new_json(
        temporary,
        {
            "status": "partial_evaluation",
            "target_index": index,
            "target_id": target_id,
            "skipped_out_of_memory": not records,
            "records": records,
        },
    )
    temporary.replace(path)


@dataclass
class _PendingTarget:
    target_id: str
    records: list[dict[str, object]]
    scores: list[Future[dict[str, float | None]]]

    def save(self, directory: Path, index: int) -> None:
        for record, future in zip(self.records, self.scores, strict=True):
            record.update(future.result())
        _write_partial_target(directory, index, self.target_id, self.records)


def _save_finished_targets(pending: dict[int, _PendingTarget], directory: Path) -> None:
    for index, target in list(pending.items()):
        if all(
            future.done() and not future.cancelled() and future.exception() is None
            for future in target.scores
        ):
            target.save(directory, index)
            del pending[index]


def fold_and_score(
    model: torch.nn.Module,
    pool_dir: Path,
    targets: Sequence[Mapping[str, object]],
    heads: Mapping[str, object] | None,
    output_path: Path,
    log: Log,
    workers: int = 32,
) -> list[dict[str, object]]:
    """Fold targets, score samples with candidate or native heads, and add structure quality."""
    skipped_path = output_path.with_name("skipped.json")
    for path in (output_path, skipped_path):
        if path.exists():
            raise FileExistsError(f"Evaluation output already exists: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = output_path.parent / "partial-records"
    partial_dir.mkdir(exist_ok=False)
    records: list[dict[str, object]] = []
    pending: dict[int, _PendingTarget] = {}
    skipped: list[str] = []
    rollout = None
    try:
        with ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            for index, target in enumerate(targets):
                _save_finished_targets(pending, partial_dir)
                sequences = list(target["sequences"])  # type: ignore[arg-type]
                try:
                    native_structure = structure(pool_dir, target)
                    positions = native_structure.positions  # (residues, 14, 3)
                    positions_identity = {
                        "sha256": hashlib.sha256(memoryview(positions).cast("B")).hexdigest(),
                        "shape": list(positions.shape),
                        "dtype": str(positions.dtype),
                    }
                    rollout = fold(
                        model,
                        native_structure,
                        EVALUATION_SAMPLES,
                        EVALUATION_SEED_OFFSET + index,
                        INFERENCE_LOOPS,
                        INFERENCE_SAMPLING_STEPS,
                        native_confidence=heads is None,
                    )
                except torch.OutOfMemoryError:
                    rollout = None
                    skipped.append(str(target["target_id"]))
                    log(
                        f"skipped {target['target_id']} ({target['num_tokens']} tokens): out of memory"
                    )
                    torch.cuda.empty_cache()
                    _write_partial_target(partial_dir, index, str(target["target_id"]), [])
                    continue
                target_records: list[dict[str, object]] = []
                futures: list[Future[dict[str, float | None]]] = []
                native = None
                try:
                    for sample in range(EVALUATION_SAMPLES):
                        if heads is None:
                            native = rollout.native_confidence or {}
                            predictions = {
                                "production": _summaries(
                                    native["plddt_logits"][sample : sample + 1],
                                    native["pae_logits"][sample : sample + 1],
                                    rollout,
                                    sample,
                                )
                            }
                        else:
                            predictions = {
                                name: head_sample_predictions(head, rollout, sample)
                                for name, head in heads.items()
                            }
                        record = {
                            "target_id": target["target_id"],
                            "stratum": target["stratum"],
                            "num_chains": len(sequences),
                            "num_tokens": int(target["num_tokens"]),
                            "sample": sample,
                            "target_positions": positions_identity,
                            **{
                                f"true_{name}": value
                                for name, value in rollout.quality[sample].items()
                            },
                            "predictions": predictions,
                        }
                        target_records.append(record)
                        records.append(record)
                        futures.append(
                            executor.submit(
                                structure_scores,
                                sample_structures(rollout, sequences, sample),
                            )
                        )
                finally:
                    # A previous target must not occupy GPU memory while the next fold runs.
                    rollout = None
                    native = None
                pending[index] = _PendingTarget(str(target["target_id"]), target_records, futures)
                _save_finished_targets(pending, partial_dir)
                if (index + 1) % 25 == 0:
                    log(f"folded {index + 1}/{len(targets)} test targets")
            for index in list(pending):
                pending[index].save(partial_dir, index)
                del pending[index]
    finally:
        rollout = None
        # Executor shutdown finishes submitted CPU scores even when a later target fails.
        _save_finished_targets(pending, partial_dir)
    write_new_json(output_path, records)
    write_new_json(skipped_path, skipped)
    if skipped:
        log(f"skipped {len(skipped)} of {len(targets)} targets that did not fit in memory")
    return records


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
