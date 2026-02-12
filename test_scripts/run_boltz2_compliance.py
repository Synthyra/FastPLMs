import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
from transformers import AutoModel

from boltz_automodel.cif_writer import write_cif
from boltz_automodel.get_boltz2_weights import BOLTZ2_CKPT_URL
from boltz_automodel.minimal_featurizer import build_boltz2_features
from boltz_automodel.minimal_structures import ProteinStructureTemplate
from test_scripts.common import build_output_dir
from test_scripts.common import autocast_context
from test_scripts.common import generate_sequences
from test_scripts.common import login_if_needed
from test_scripts.common import resolve_device
from test_scripts.common import resolve_dtype
from test_scripts.common import set_seed
from test_scripts.reporting import write_csv
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary


def _enforce_determinism() -> None:
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _download_checkpoint_if_needed(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        urllib.request.urlretrieve(BOLTZ2_CKPT_URL, str(checkpoint_path))  # noqa: S310
    return checkpoint_path


def _set_sequence_seed(seed: int, sequence_index: int) -> None:
    set_seed(seed + sequence_index)


def _to_device(feats: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for key in feats:
        value = feats[key]
        if value.is_floating_point():
            output[key] = value.to(device=device, dtype=dtype)
        else:
            output[key] = value.to(device=device)
    return output


def _clone_feats(feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for key in feats:
        output[key] = feats[key].clone()
    return output


def _summary_metric(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        return value
    return value.reshape(value.shape[0], -1)[:, 0]


def _extract_primary_plddt_vector(output: Dict[str, torch.Tensor], feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    assert "plddt" in output, "Missing pLDDT in model output."
    plddt = output["plddt"].detach().cpu()
    if plddt.ndim == 0:
        return plddt.reshape(1).float()
    if plddt.ndim >= 2:
        plddt = plddt[0]
    plddt = plddt.reshape(-1).float()

    token_mask = feats["token_pad_mask"][0].detach().cpu().reshape(-1) > 0
    atom_mask = feats["atom_pad_mask"][0].detach().cpu().reshape(-1) > 0
    if plddt.numel() == token_mask.numel():
        plddt = plddt[token_mask]
    elif plddt.numel() == atom_mask.numel():
        plddt = plddt[atom_mask]
    return plddt


def _compute_confidence_score(ptm: torch.Tensor, iptm: torch.Tensor, complex_plddt: torch.Tensor) -> torch.Tensor:
    if torch.allclose(iptm, torch.zeros_like(iptm)):
        return (4 * complex_plddt + ptm) / 5
    return (4 * complex_plddt + iptm) / 5


def _run_ours_forward(
    model,
    feats_ours: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    sequence_index: int,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad(), autocast_context(device=device, dtype=dtype):
        _set_sequence_seed(args.seed, sequence_index)
        return model.forward(
            feats=feats_ours,
            recycling_steps=args.recycling_steps,
            num_sampling_steps=args.num_sampling_steps,
            diffusion_samples=args.diffusion_samples,
            run_confidence_sequentially=args.run_confidence_sequentially,
        )


def _vector_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> Tuple[float, float, float]:
    delta = lhs.float() - rhs.float()
    abs_delta = torch.abs(delta)
    mae = float(abs_delta.mean().item())
    rmse = float(torch.sqrt(torch.mean(delta * delta)).item())
    max_abs = float(abs_delta.max().item())
    return mae, rmse, max_abs


def _kabsch_align_mobile_to_target(mobile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert mobile.ndim == 2 and target.ndim == 2, "Expected coordinate tensors with shape [N, 3]."
    assert mobile.shape == target.shape, "Coordinate tensors must have matching shapes."
    assert mobile.shape[1] == 3, "Coordinate tensors must have last dimension size 3."
    assert mobile.shape[0] > 0, "Expected at least one shared atom for alignment."

    mobile_32 = mobile.float()
    target_32 = target.float()
    if mobile_32.shape[0] < 3:
        mobile_centroid = mobile_32.mean(dim=0, keepdim=True)
        target_centroid = target_32.mean(dim=0, keepdim=True)
        return mobile_32 - mobile_centroid + target_centroid

    mobile_centroid = mobile_32.mean(dim=0, keepdim=True)
    target_centroid = target_32.mean(dim=0, keepdim=True)
    mobile_centered = mobile_32 - mobile_centroid
    target_centered = target_32 - target_centroid

    covariance = mobile_centered.transpose(0, 1).matmul(target_centered)
    u_mat, _, vh_mat = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.eye(3, dtype=mobile_32.dtype, device=mobile_32.device)
    det_sign = torch.det(vh_mat.transpose(0, 1).matmul(u_mat.transpose(0, 1))).item()
    if det_sign < 0:
        correction[2, 2] = -1.0
    rotation = vh_mat.transpose(0, 1).matmul(correction).matmul(u_mat.transpose(0, 1))
    return mobile_centered.matmul(rotation) + target_centroid


def _pairwise_distance_mae(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    assert lhs.ndim == 2 and rhs.ndim == 2, "Expected coordinate tensors with shape [N, 3]."
    assert lhs.shape == rhs.shape, "Coordinate tensors must have matching shapes."
    assert lhs.shape[1] == 3, "Coordinate tensors must have last dimension size 3."
    lhs_dist = torch.cdist(lhs.float(), lhs.float())
    rhs_dist = torch.cdist(rhs.float(), rhs.float())
    return float(torch.mean(torch.abs(lhs_dist - rhs_dist)).item())


def _write_single_chain_fasta(sequence: str, path: Path) -> None:
    text = f">A|protein|empty\n{sequence}\n"
    path.write_text(text, encoding="utf-8")


def _parse_pdb_atom_map(path: Path) -> Dict[Tuple[str, int, str], torch.Tensor]:
    atom_map: Dict[Tuple[str, int, str], torch.Tensor] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        chain_id = line[21:22].strip()
        residue_index = int(line[22:26])
        x_val = float(line[30:38])
        y_val = float(line[38:46])
        z_val = float(line[46:54])
        atom_map[(chain_id, residue_index, atom_name)] = torch.tensor(
            [x_val, y_val, z_val],
            dtype=torch.float32,
        )
    assert len(atom_map) > 0, f"No atoms parsed from reference PDB: {path}"
    return atom_map


def _build_ours_atom_map(
    sample_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_names: List[str],
    atom_residue_index: List[int],
    atom_chain_id: List[str],
) -> Dict[Tuple[str, int, str], torch.Tensor]:
    coords = sample_coords[0]
    valid_coords = coords[atom_mask > 0]
    assert valid_coords.shape[0] >= len(atom_names), (
        "Our model returned fewer valid atom coordinates than template atoms."
    )

    atom_map: Dict[Tuple[str, int, str], torch.Tensor] = {}
    for atom_idx in range(len(atom_names)):
        key = (
            atom_chain_id[atom_idx],
            atom_residue_index[atom_idx] + 1,
            atom_names[atom_idx],
        )
        atom_map[key] = valid_coords[atom_idx].float().cpu()
    return atom_map


def _build_reference_cif_tensors(
    template: ProteinStructureTemplate,
    atom_pad_mask: torch.Tensor,
    ref_atom_map: Dict[Tuple[str, int, str], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert atom_pad_mask.ndim == 1, "Expected atom pad mask with shape [atoms]."
    atom_slots = atom_pad_mask.shape[0]
    coords = torch.zeros((1, atom_slots, 3), dtype=torch.float32)
    ref_mask = atom_pad_mask.detach().cpu().float().clone()
    for atom_idx in range(template.num_atoms):
        key = (
            template.atom_chain_id[atom_idx],
            template.atom_residue_index[atom_idx] + 1,
            template.atom_names[atom_idx],
        )
        if key in ref_atom_map:
            coords[0, atom_idx] = ref_atom_map[key].float().cpu()
        else:
            ref_mask[atom_idx] = 0.0
    return coords, ref_mask


def _run_boltz_cli_reference(
    sequence: str,
    sequence_index: int,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[Tuple[str, int, str], torch.Tensor], torch.Tensor, Dict[str, float]]:
    with tempfile.TemporaryDirectory(prefix=f"boltz2_ref_{sequence_index}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        fasta_path = tmp_dir / f"seq_{sequence_index}.fasta"
        out_root = tmp_dir / "ref_out"
        _write_single_chain_fasta(sequence=sequence, path=fasta_path)

        accelerator = "gpu" if device.type == "cuda" else "cpu"
        command = [
            sys.executable,
            "-m",
            "boltz.main",
            "predict",
            str(fasta_path),
            "--out_dir",
            str(out_root),
            "--model",
            "boltz2",
            "--checkpoint",
            str(checkpoint_path),
            "--recycling_steps",
            str(args.recycling_steps),
            "--sampling_steps",
            str(args.num_sampling_steps),
            "--diffusion_samples",
            str(args.diffusion_samples),
            "--seed",
            str(args.seed + sequence_index),
            "--override",
            "--no_kernels",
            "--output_format",
            "pdb",
            "--num_workers",
            "0",
            "--devices",
            "1",
            "--accelerator",
            accelerator,
        ]

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(args.seed + sequence_index)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            stderr = completed.stderr[-4000:]
            stdout = completed.stdout[-4000:]
            raise RuntimeError(
                "pip boltz CLI prediction failed.\n"
                f"Command: {' '.join(command)}\n"
                f"STDOUT tail:\n{stdout}\n"
                f"STDERR tail:\n{stderr}"
            )

        results_root = out_root / f"boltz_results_{fasta_path.stem}" / "predictions"
        assert results_root.exists(), f"Reference predictions directory not found: {results_root}"

        pdb_candidates = sorted(results_root.rglob("*_model_0.pdb"))
        plddt_candidates = sorted(results_root.rglob("plddt_*_model_0.npz"))
        confidence_candidates = sorted(results_root.rglob("confidence_*_model_0.json"))

        assert len(pdb_candidates) > 0, f"No reference model_0 PDB found under {results_root}"
        assert len(plddt_candidates) > 0, f"No reference pLDDT npz found under {results_root}"
        assert len(confidence_candidates) > 0, f"No reference confidence json found under {results_root}"

        atom_map = _parse_pdb_atom_map(pdb_candidates[0])
        with np.load(plddt_candidates[0]) as handle:
            plddt = torch.tensor(handle["plddt"], dtype=torch.float32)
        confidence_summary = json.loads(confidence_candidates[0].read_text(encoding="utf-8"))
        for key in [
            "ptm",
            "iptm",
            "complex_plddt",
            "confidence_score",
        ]:
            assert key in confidence_summary, (
                f"Reference confidence summary missing key '{key}' in {confidence_candidates[0]}"
            )
        return atom_map, plddt, confidence_summary


def run_boltz2_compliance_suite(args: argparse.Namespace) -> int:
    if args.enforce_determinism:
        _enforce_determinism()
    login_if_needed(args.token)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)

    output_dir = build_output_dir(args.output_dir, "boltz2_compliance")
    checkpoint_path = _download_checkpoint_if_needed(Path(args.checkpoint_path))
    sequences = generate_sequences(
        num_sequences=args.num_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )

    model = AutoModel.from_pretrained(
        args.repo_id,
        trust_remote_code=True,
    )
    model = model.to(device=device, dtype=torch.float32).eval()

    rows: List[Dict[str, object]] = []
    overall_pass = True

    for sequence_index, sequence in enumerate(sequences):
        started = time.perf_counter()
        row: Dict[str, object] = {
            "sequence_index": sequence_index,
            "sequence": sequence,
            "sequence_seed": args.seed + sequence_index,
            "ours_dtype_effective": str(dtype),
            "coord_mae": float("nan"),
            "coord_rmse": float("nan"),
            "coord_max_abs": float("nan"),
            "coord_mae_aligned": float("nan"),
            "coord_rmse_aligned": float("nan"),
            "coord_max_abs_aligned": float("nan"),
            "pairwise_dist_mae": float("nan"),
            "plddt_mae": float("nan"),
            "ptm_abs_diff": float("nan"),
            "iptm_abs_diff": float("nan"),
            "complex_plddt_abs_diff": float("nan"),
            "confidence_score_abs_diff": float("nan"),
            "shared_atoms": 0,
            "ours_cif_path": "",
            "ref_cif_path": "",
            "pass": False,
            "seconds": 0.0,
            "error": "",
        }

        try:
            feats, template = build_boltz2_features(
                amino_acid_sequence=sequence,
                num_bins=model.config.num_bins,
                atoms_per_window_queries=model.core.input_embedder.atom_encoder.atoms_per_window_queries,
            )
            feats_ours = _to_device(_clone_feats(feats), device=device, dtype=torch.float32)

            try:
                out_ours = _run_ours_forward(
                    model=model,
                    feats_ours=feats_ours,
                    args=args,
                    device=device,
                    dtype=dtype,
                    sequence_index=sequence_index,
                )
            except RuntimeError as exc:
                error_text = str(exc)
                bf16_mismatch = "expected scalar type Float but found BFloat16" in error_text
                fp16_mismatch = "expected scalar type Float but found Half" in error_text
                if bf16_mismatch or fp16_mismatch:
                    out_ours = _run_ours_forward(
                        model=model,
                        feats_ours=feats_ours,
                        args=args,
                        device=device,
                        dtype=torch.float32,
                        sequence_index=sequence_index,
                    )
                    row["ours_dtype_effective"] = str(torch.float32)
                else:
                    raise

            ours_atom_map = _build_ours_atom_map(
                sample_coords=out_ours["sample_atom_coords"].detach().cpu(),
                atom_mask=feats_ours["atom_pad_mask"][0].detach().cpu(),
                atom_names=template.atom_names,
                atom_residue_index=template.atom_residue_index,
                atom_chain_id=template.atom_chain_id,
            )

            ref_atom_map, ref_plddt, ref_confidence = _run_boltz_cli_reference(
                sequence=sequence,
                sequence_index=sequence_index,
                checkpoint_path=checkpoint_path,
                args=args,
                device=device,
            )

            shared_keys = []
            for key in ours_atom_map:
                if key in ref_atom_map:
                    shared_keys.append(key)
            shared_keys.sort()
            assert len(shared_keys) > 0, "No overlapping atom keys between our output and pip boltz CLI output."
            row["shared_atoms"] = len(shared_keys)

            ours_coords_stack = torch.stack([ours_atom_map[key] for key in shared_keys], dim=0)
            ref_coords_stack = torch.stack([ref_atom_map[key] for key in shared_keys], dim=0)
            coord_mae, coord_rmse, coord_max_abs = _vector_metrics(ours_coords_stack, ref_coords_stack)
            row["coord_mae"] = coord_mae
            row["coord_rmse"] = coord_rmse
            row["coord_max_abs"] = coord_max_abs
            ours_coords_aligned = _kabsch_align_mobile_to_target(ours_coords_stack, ref_coords_stack)
            coord_mae_aligned, coord_rmse_aligned, coord_max_abs_aligned = _vector_metrics(
                ours_coords_aligned,
                ref_coords_stack,
            )
            row["coord_mae_aligned"] = coord_mae_aligned
            row["coord_rmse_aligned"] = coord_rmse_aligned
            row["coord_max_abs_aligned"] = coord_max_abs_aligned
            row["pairwise_dist_mae"] = _pairwise_distance_mae(ours_coords_stack, ref_coords_stack)

            ours_plddt = _extract_primary_plddt_vector(out_ours, feats_ours)
            ref_plddt = ref_plddt.float().cpu().reshape(-1)
            assert ours_plddt.numel() == ref_plddt.numel(), (
                f"pLDDT size mismatch (ours={ours_plddt.numel()}, ref={ref_plddt.numel()})."
            )
            row["plddt_mae"] = float(torch.mean(torch.abs(ours_plddt - ref_plddt)).item())

            ours_ptm = _summary_metric(out_ours["ptm"]).float().cpu()
            ours_iptm = _summary_metric(out_ours["iptm"]).float().cpu()
            ours_complex_plddt = _summary_metric(out_ours["complex_plddt"]).float().cpu()
            ours_confidence_score = _compute_confidence_score(
                ptm=ours_ptm,
                iptm=ours_iptm,
                complex_plddt=ours_complex_plddt,
            )

            ref_ptm = torch.tensor([float(ref_confidence["ptm"])], dtype=torch.float32)
            ref_iptm = torch.tensor([float(ref_confidence["iptm"])], dtype=torch.float32)
            ref_complex_plddt = torch.tensor(
                [float(ref_confidence["complex_plddt"])],
                dtype=torch.float32,
            )
            ref_confidence_score = torch.tensor(
                [float(ref_confidence["confidence_score"])],
                dtype=torch.float32,
            )

            row["ptm_abs_diff"] = float(torch.mean(torch.abs(ours_ptm - ref_ptm)).item())
            row["iptm_abs_diff"] = float(torch.mean(torch.abs(ours_iptm - ref_iptm)).item())
            row["complex_plddt_abs_diff"] = float(
                torch.mean(torch.abs(ours_complex_plddt - ref_complex_plddt)).item()
            )
            row["confidence_score_abs_diff"] = float(
                torch.mean(torch.abs(ours_confidence_score - ref_confidence_score)).item()
            )

            if args.write_cif_artifacts:
                structure_dir = output_dir / "structures" / f"seq_{sequence_index}"
                structure_dir.mkdir(parents=True, exist_ok=True)

                ours_cif_path = structure_dir / f"ours_seq{sequence_index}.cif"
                write_cif(
                    structure_template=template,
                    atom_coords=out_ours["sample_atom_coords"].detach().cpu(),
                    atom_mask=feats_ours["atom_pad_mask"][0].detach().cpu(),
                    output_path=str(ours_cif_path),
                    plddt=out_ours["plddt"].detach().cpu() if "plddt" in out_ours else None,
                    sample_index=0,
                )
                row["ours_cif_path"] = str(ours_cif_path)

                ref_coords_cif, ref_atom_mask_cif = _build_reference_cif_tensors(
                    template=template,
                    atom_pad_mask=feats_ours["atom_pad_mask"][0].detach().cpu(),
                    ref_atom_map=ref_atom_map,
                )
                ref_cif_path = structure_dir / f"ref_seq{sequence_index}.cif"
                write_cif(
                    structure_template=template,
                    atom_coords=ref_coords_cif,
                    atom_mask=ref_atom_mask_cif,
                    output_path=str(ref_cif_path),
                    plddt=ref_plddt,
                    sample_index=0,
                )
                row["ref_cif_path"] = str(ref_cif_path)

            if args.pass_coord_metric == "aligned":
                coord_mae_for_pass = row["coord_mae_aligned"]
                coord_rmse_for_pass = row["coord_rmse_aligned"]
                coord_max_for_pass = row["coord_max_abs_aligned"]
            else:
                coord_mae_for_pass = row["coord_mae"]
                coord_rmse_for_pass = row["coord_rmse"]
                coord_max_for_pass = row["coord_max_abs"]

            row["pass"] = bool(
                coord_mae_for_pass <= args.coord_mae_threshold
                and coord_rmse_for_pass <= args.coord_rmse_threshold
                and coord_max_for_pass <= args.coord_max_abs_threshold
                and row["plddt_mae"] <= args.plddt_mae_threshold
                and row["ptm_abs_diff"] <= args.summary_metric_abs_threshold
                and row["iptm_abs_diff"] <= args.summary_metric_abs_threshold
                and row["complex_plddt_abs_diff"] <= args.summary_metric_abs_threshold
                and row["confidence_score_abs_diff"] <= args.summary_metric_abs_threshold
            )
            if row["pass"] is False:
                overall_pass = False
        except Exception as exc:
            row["error"] = str(exc)
            overall_pass = False
        finally:
            row["seconds"] = round(time.perf_counter() - started, 4)
            rows.append(row)

    payload: Dict[str, object] = {
        "suite": "boltz2_compliance",
        "all_passed": overall_pass,
        "repo_id": args.repo_id,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "enforce_determinism": args.enforce_determinism,
        "pass_coord_metric": args.pass_coord_metric,
        "write_cif_artifacts": args.write_cif_artifacts,
        "num_sequences": args.num_sequences,
        "recycling_steps": args.recycling_steps,
        "num_sampling_steps": args.num_sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "rows": rows,
    }
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "metrics.csv", rows)

    passed_count = 0
    for row in rows:
        if bool(row["pass"]):
            passed_count += 1
    summary_lines = [
        "Suite: boltz2_compliance",
        f"Sequences tested: {len(rows)}",
        f"Sequences passed: {passed_count}",
        f"Sequences failed: {len(rows) - passed_count}",
        f"Output directory: {output_dir}",
        f"Device: {device}",
        f"Dtype: {dtype}",
        f"Pass coord metric: {args.pass_coord_metric}",
        f"Write CIF artifacts: {args.write_cif_artifacts}",
    ]
    for row in rows:
        status = "PASS" if bool(row["pass"]) else "FAIL"
        summary_lines.append(
            f"{status} | idx={row['sequence_index']} | seed={row['sequence_seed']} | "
            f"ours_dtype={row['ours_dtype_effective']} | "
            f"shared_atoms={row['shared_atoms']} | "
            f"coord_raw_mae={row['coord_mae']} | coord_raw_rmse={row['coord_rmse']} | "
            f"coord_raw_max={row['coord_max_abs']} | coord_aln_mae={row['coord_mae_aligned']} | "
            f"coord_aln_rmse={row['coord_rmse_aligned']} | coord_aln_max={row['coord_max_abs_aligned']} | "
            f"pairwise_dist_mae={row['pairwise_dist_mae']} | plddt_mae={row['plddt_mae']} | "
            f"ptm_diff={row['ptm_abs_diff']} | iptm_diff={row['iptm_abs_diff']} | "
            f"cplddt_diff={row['complex_plddt_abs_diff']} | "
            f"score_diff={row['confidence_score_abs_diff']} | "
            f"ours_cif={row['ours_cif_path']} | ref_cif={row['ref_cif_path']} | error={row['error']}"
        )
    write_summary(output_dir / "summary.txt", summary_lines)
    print("\n".join(summary_lines))

    if overall_pass:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Boltz2 compliance test against pip boltz CLI outputs.")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default="Synthyra/Boltz2")
    parser.add_argument("--checkpoint-path", type=str, default="boltz_automodel/weights/boltz2_conf.ckpt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-determinism", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-cif-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pass-coord-metric", type=str, default="aligned", choices=["raw", "aligned"])
    parser.add_argument("--num-sequences", type=int, default=3)
    parser.add_argument("--min-length", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=200)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--run-confidence-sequentially", action="store_true")
    parser.add_argument("--coord-mae-threshold", type=float, default=5e-3)
    parser.add_argument("--coord-rmse-threshold", type=float, default=5e-3)
    parser.add_argument("--coord-max-abs-threshold", type=float, default=5e-2)
    parser.add_argument("--plddt-mae-threshold", type=float, default=5e-3)
    parser.add_argument("--summary-metric-abs-threshold", type=float, default=5e-3)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_boltz2_compliance_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
