#!/usr/bin/env python3
"""Prepare ESMFold2 complexes or run seeded local structure helpers.

The ESMFold2 branch deliberately includes an MSA and therefore requires a full
48-block checkpoint, not an inference-optimized Fast checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def build_esmfold2_conditioned_complex(types: Any) -> Any:
    """Construct a supported full-variant multimolecule, MSA, and bond request."""
    protein = "MSTNPKPQRKTKRNT"
    msa = types.MSA.from_sequences([protein, "MSTNPKPQRKTKRNS"])
    return types.StructurePredictionInput(
        sequences=[
            types.ProteinInput(id="A", sequence=protein, msa=msa),
            types.ProteinInput(
                id="B",
                sequence="MKTIIALSYIFCLVFA",
                modifications=[types.Modification(position=0, ccd="MSE")],
            ),
            types.RNAInput(id="R", sequence="AUGC"),
            types.DNAInput(id="D", sequence="ATGC"),
            types.LigandInput(id="L", smiles="O"),
        ],
        covalent_bonds=[
            types.CovalentBond(
                chain_id1="B",
                res_idx1=0,
                atom_idx1=0,
                chain_id2="L",
                res_idx2=0,
                atom_idx2=0,
            )
        ],
    )


def prepare_esmfold2_complex(model: Any, seed: int) -> tuple[Any, Any]:
    types = model.input_types
    request = build_esmfold2_conditioned_complex(types)
    return model.prepare_structure_input(request, seed=seed)


def verify_esmfold2_pocket_rejection(model: Any, seed: int) -> str:
    """Show the explicit boundary inherited from the published feature pipeline."""
    types = model.input_types
    request = types.StructurePredictionInput(
        sequences=[
            types.ProteinInput(id="target", sequence="MSTNPKPQRKTKRNT"),
            types.ProteinInput(id="binder", sequence="MKTIIALSYIFCLVFA"),
        ],
        pocket=types.PocketConditioning(
            binder_chain_id="binder",
            contacts=[("target", 0)],
        ),
    )
    try:
        model.prepare_structure_input(request, seed=seed)
    except NotImplementedError as error:
        return str(error)
    raise RuntimeError("ESMFold2 unexpectedly accepted unsupported pocket conditioning.")


def verify_esmfold2_distogram_rejection(model: Any, seed: int) -> str:
    """Show that the schema cannot silently pass unused distance conditioning."""
    import numpy as np

    types = model.input_types
    request = types.StructurePredictionInput(
        sequences=[types.ProteinInput(id="target", sequence="MSTNPKPQRKTKRNT")],
        distogram_conditioning=[
            types.DistogramConditioning(
                chain_id="target",
                distogram=np.zeros((15, 15), dtype=np.float32),  # (l=15, l=15)
            )
        ],
    )
    try:
        model.prepare_structure_input(request, seed=seed)
    except NotImplementedError as error:
        return str(error)
    raise RuntimeError("ESMFold2 unexpectedly accepted unsupported distogram conditioning.")


def run_structure_helper(model: Any, family: str, sequence: str, seed: int) -> Any:
    if family == "boltz2":
        return model.predict_structure(
            amino_acid_sequence=sequence,
            recycling_steps=1,
            num_sampling_steps=8,
            diffusion_samples=1,
            seed=seed,
        )
    if family == "esmfold":
        return model.fold_protein(sequence)
    raise ValueError(f"No direct structure helper for {family!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("esmfold2", "esmfold", "boltz2"))
    parser.add_argument(
        "artifact",
        type=Path,
        help=(
            "Local artifact; the ESMFold2 MSA branch requires a full 48-block "
            "checkpoint and rejects Fast variants"
        ),
    )
    parser.add_argument(
        "--sequence",
        default="MSTNPKPQRKTKRNT",
        help="Protein sequence; ESMFold also accepts colon-delimited multimers",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact = arguments.artifact.expanduser().resolve()
    if not (artifact / "config.json").is_file():
        raise SystemExit(f"Not a local artifact: {artifact}")

    configure_offline()
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        artifact,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": arguments.device},
    ).eval()
    if arguments.family == "esmfold2":
        features, chain_info = prepare_esmfold2_complex(model, arguments.seed)
        print("feature-keys", sorted(features))
        print("chains", len(chain_info))
        pocket_contract = verify_esmfold2_pocket_rejection(model, arguments.seed)
        print("pocket-contract", pocket_contract)
        distogram_contract = verify_esmfold2_distogram_rejection(model, arguments.seed)
        print("distogram-contract", distogram_contract)
    else:
        result = run_structure_helper(
            model,
            arguments.family,
            arguments.sequence,
            arguments.seed,
        )
        print(type(result).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
