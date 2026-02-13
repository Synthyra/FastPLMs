import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

import biotite.structure as struc
import matplotlib
import numpy as np
import torch

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from transformers import AutoModel

from boltz_automodel.cif_writer import write_cif
from boltz_automodel.get_boltz2_weights import BOLTZ2_CKPT_URL
from boltz_automodel.minimal_featurizer import build_boltz2_features
from boltz_automodel.minimal_structures import ProteinStructureTemplate
from test_scripts.common import autocast_context
from test_scripts.common import build_output_dir
from test_scripts.common import login_if_needed
from test_scripts.common import resolve_device
from test_scripts.common import resolve_dtype
from test_scripts.common import set_seed
from test_scripts.reporting import write_csv
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary


matplotlib.use("Agg")
import matplotlib.pyplot as plt


assert "tm_score" in dir(struc), (
    "biotite.structure.tm_score is unavailable. Install biotite>=1.5.0 in the target environment."
)

TM_SCORE_FN = struc.tm_score
BOLTZ2_FIXED_RECYCLING_STEPS = 3
BOLTZ2_FIXED_SAMPLING_STEPS = 200
BOLTZ2_FIXED_DIFFUSION_SAMPLES = 20

SEQUENCE_OPTIONS = [
    "MDDADPEERNYDNMLKMLSDLNKDLEKLLEEMEKISVQATWMAYDMVVMRTNPTLAESMRRLEDAFVNCKEEMEKNWQELLHETKQRL",
    "MASLGHILVFCVGLLTMAKAESPKEHDPFTYDYQSLQIGGLVIAGILFILGILIVLSRRCRCKFNQQQRTGEPDEEEGTFRSSIRRLSTRRR",
    "MAVESRVTQEEIKKEPEKPIDREKTCPLLLRVFTTNNGRHHRMDEFSRGNVPSSELQIYTWMDATLKELTSLVKEVYPEARKKGTHFNFAIVFTDVKRPGYRVKEIGSTMSGRKGTDDSMTLQSQKFQIGDYLDIAITPPNRAPPPSGRMRPY",
]


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


def _detect_no_kernels_support() -> bool:
    command = [sys.executable, "-m", "boltz.main", "predict", "--help"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    return "--no_kernels" in combined_output


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
            recycling_steps=BOLTZ2_FIXED_RECYCLING_STEPS,
            num_sampling_steps=BOLTZ2_FIXED_SAMPLING_STEPS,
            diffusion_samples=BOLTZ2_FIXED_DIFFUSION_SAMPLES,
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
        atom_map[(chain_id, residue_index, atom_name)] = torch.tensor([x_val, y_val, z_val], dtype=torch.float32)
    assert len(atom_map) > 0, f"No atoms parsed from PDB: {path}"
    return atom_map


def _extract_model_id_from_name(filename: str) -> int:
    match = re.search(r"_model_(\d+)\.", filename)
    assert match is not None, f"Could not parse model id from filename: {filename}"
    return int(match.group(1))


def _map_paths_by_model(paths: List[Path]) -> Dict[int, Path]:
    path_map: Dict[int, Path] = {}
    for path in paths:
        model_id = _extract_model_id_from_name(path.name)
        assert model_id not in path_map, f"Found duplicate artifacts for model id {model_id}: {path}"
        path_map[model_id] = path
    return path_map


def _build_ours_atom_maps(
    sample_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_names: List[str],
    atom_residue_index: List[int],
    atom_chain_id: List[str],
) -> List[Dict[Tuple[str, int, str], torch.Tensor]]:
    coords = sample_coords.detach().cpu()
    if coords.ndim == 4:
        assert coords.shape[0] == 1, "Expected singleton batch dimension for sample coordinates."
        coords = coords[0]
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    assert coords.ndim == 3, f"Expected sample_atom_coords with 3 dimensions, got shape {coords.shape}."
    assert coords.shape[0] >= BOLTZ2_FIXED_DIFFUSION_SAMPLES, (
        f"Expected at least {BOLTZ2_FIXED_DIFFUSION_SAMPLES} samples, got {coords.shape[0]}."
    )

    atom_mask_bool = atom_mask.detach().cpu() > 0
    output: List[Dict[Tuple[str, int, str], torch.Tensor]] = []
    for sample_index in range(BOLTZ2_FIXED_DIFFUSION_SAMPLES):
        valid_coords = coords[sample_index][atom_mask_bool]
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
        output.append(atom_map)
    return output


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
    supports_no_kernels: bool,
) -> Tuple[List[Dict[Tuple[str, int, str], torch.Tensor]], List[torch.Tensor], List[Dict[str, float]]]:
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
            str(BOLTZ2_FIXED_RECYCLING_STEPS),
            "--sampling_steps",
            str(args.num_sampling_steps),
            "--diffusion_samples",
            str(BOLTZ2_FIXED_DIFFUSION_SAMPLES),
            "--seed",
            str(args.seed + sequence_index),
            "--override",
            "--output_format",
            "pdb",
            "--num_workers",
            "0",
            "--devices",
            "1",
            "--accelerator",
            accelerator,
        ]
        if supports_no_kernels:
            command.append("--no_kernels")

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(args.seed + sequence_index)
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
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

        pdb_candidates = sorted(results_root.rglob("*_model_*.pdb"))
        plddt_candidates = sorted(results_root.rglob("plddt_*_model_*.npz"))
        confidence_candidates = sorted(results_root.rglob("confidence_*_model_*.json"))

        assert len(pdb_candidates) > 0, f"No reference PDB artifacts found under {results_root}"
        assert len(plddt_candidates) > 0, f"No reference pLDDT npz artifacts found under {results_root}"
        assert len(confidence_candidates) > 0, f"No reference confidence json artifacts found under {results_root}"

        pdb_by_model = _map_paths_by_model(pdb_candidates)
        plddt_by_model = _map_paths_by_model(plddt_candidates)
        confidence_by_model = _map_paths_by_model(confidence_candidates)

        expected_model_ids = list(range(BOLTZ2_FIXED_DIFFUSION_SAMPLES))
        for model_id in expected_model_ids:
            assert model_id in pdb_by_model, f"Missing reference PDB for model {model_id}"
            assert model_id in plddt_by_model, f"Missing reference pLDDT for model {model_id}"
            assert model_id in confidence_by_model, f"Missing reference confidence JSON for model {model_id}"

        atom_maps: List[Dict[Tuple[str, int, str], torch.Tensor]] = []
        plddt_samples: List[torch.Tensor] = []
        confidence_summaries: List[Dict[str, float]] = []
        for model_id in expected_model_ids:
            atom_maps.append(_parse_pdb_atom_map(pdb_by_model[model_id]))

            with np.load(plddt_by_model[model_id]) as handle:
                assert "plddt" in handle.files, f"Missing 'plddt' array in {plddt_by_model[model_id]}"
                plddt_samples.append(torch.tensor(handle["plddt"], dtype=torch.float32))

            confidence_summary = json.loads(confidence_by_model[model_id].read_text(encoding="utf-8"))
            for key in ["ptm", "iptm", "complex_plddt", "confidence_score"]:
                assert key in confidence_summary, (
                    f"Reference confidence summary missing key '{key}' in {confidence_by_model[model_id]}"
                )
            confidence_summaries.append(confidence_summary)

        return atom_maps, plddt_samples, confidence_summaries


def _shared_ca_key_order(
    ours_atom_maps: List[Dict[Tuple[str, int, str], torch.Tensor]],
    ref_atom_maps: List[Dict[Tuple[str, int, str], torch.Tensor]],
) -> List[Tuple[str, int, str]]:
    shared_keys = {key for key in ours_atom_maps[0] if key[2] == "CA"}
    for atom_map in ours_atom_maps:
        ca_keys = {key for key in atom_map if key[2] == "CA"}
        shared_keys = shared_keys.intersection(ca_keys)
    for atom_map in ref_atom_maps:
        ca_keys = {key for key in atom_map if key[2] == "CA"}
        shared_keys = shared_keys.intersection(ca_keys)
    assert len(shared_keys) > 0, "No shared CA atoms found across all samples."
    ordered_keys = list(shared_keys)
    ordered_keys.sort()
    return ordered_keys


def _stack_coords_for_keys(
    atom_maps: List[Dict[Tuple[str, int, str], torch.Tensor]],
    ordered_keys: List[Tuple[str, int, str]],
) -> torch.Tensor:
    stacked_samples: List[torch.Tensor] = []
    for atom_map in atom_maps:
        coords: List[torch.Tensor] = []
        for key in ordered_keys:
            assert key in atom_map, f"Missing key {key} in atom map."
            coords.append(atom_map[key].float())
        stacked_samples.append(torch.stack(coords, dim=0))
    return torch.stack(stacked_samples, dim=0)


def _coords_to_ca_atom_array(coords: np.ndarray) -> struc.AtomArray:
    assert coords.ndim == 2, "Expected coordinate array with shape [N, 3]."
    assert coords.shape[1] == 3, "Expected coordinate array with shape [N, 3]."
    assert coords.shape[0] > 0, "Expected at least one CA atom for TM-score."
    assert np.all(np.isfinite(coords)), "Coordinate array for TM-score contains non-finite values."

    atom_count = coords.shape[0]
    array = struc.AtomArray(atom_count)
    array.coord = coords.astype(np.float32, copy=False)
    array.atom_name = np.full(atom_count, "CA")
    array.res_name = np.full(atom_count, "GLY")
    array.chain_id = np.full(atom_count, "A")
    array.res_id = np.arange(1, atom_count + 1, dtype=np.int32)
    array.element = np.full(atom_count, "C")
    return array


def _tm_score_from_coords(reference_coords: torch.Tensor, subject_coords: torch.Tensor) -> float:
    aligned_subject = _kabsch_align_mobile_to_target(subject_coords, reference_coords)
    reference_np = reference_coords.detach().cpu().numpy().astype(np.float64)
    aligned_np = aligned_subject.detach().cpu().numpy().astype(np.float64)
    reference_atom_array = _coords_to_ca_atom_array(reference_np)
    subject_atom_array = _coords_to_ca_atom_array(aligned_np)
    index_array = np.arange(reference_np.shape[0], dtype=np.int32)
    tm_value = float(
        TM_SCORE_FN(
            reference=reference_atom_array,
            subject=subject_atom_array,
            reference_indices=index_array,
            subject_indices=index_array,
            reference_length="shorter",
        )
    )
    assert np.isfinite(tm_value), "TM-score computation produced non-finite value."
    return tm_value


def _build_tm_matrix(reference_stack: torch.Tensor, subject_stack: torch.Tensor, symmetric: bool) -> np.ndarray:
    assert reference_stack.ndim == 3 and subject_stack.ndim == 3, "Expected stacks with shape [S, N, 3]."
    assert reference_stack.shape[1:] == subject_stack.shape[1:], "Reference and subject stacks must share atom layout."
    matrix = np.zeros((reference_stack.shape[0], subject_stack.shape[0]), dtype=np.float32)
    if symmetric:
        assert reference_stack.shape[0] == subject_stack.shape[0], "Symmetric matrix requires same sample count."
        for row_idx in range(reference_stack.shape[0]):
            for col_idx in range(row_idx, subject_stack.shape[0]):
                tm_value = _tm_score_from_coords(reference_stack[row_idx], subject_stack[col_idx])
                matrix[row_idx, col_idx] = tm_value
                matrix[col_idx, row_idx] = tm_value
        return matrix

    for row_idx in range(reference_stack.shape[0]):
        for col_idx in range(subject_stack.shape[0]):
            matrix[row_idx, col_idx] = _tm_score_from_coords(reference_stack[row_idx], subject_stack[col_idx])
    return matrix


def _write_tm_matrix_heatmap(path: Path, matrix: np.ndarray, title: str) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Column sample index")
    axis.set_ylabel("Row sample index")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_tm_matrix_artifacts(
    matrix_dir: Path,
    matrix_name: str,
    title: str,
    matrix: np.ndarray,
) -> Tuple[str, str, str]:
    csv_path = matrix_dir / f"{matrix_name}.csv"
    npy_path = matrix_dir / f"{matrix_name}.npy"
    png_path = matrix_dir / f"{matrix_name}.png"
    np.savetxt(csv_path, matrix, delimiter=",", fmt="%.6f")
    np.save(npy_path, matrix)
    _write_tm_matrix_heatmap(path=png_path, matrix=matrix, title=title)
    return str(csv_path), str(npy_path), str(png_path)


def _matrix_stats(matrix: np.ndarray) -> Dict[str, float]:
    flattened = matrix.reshape(-1)
    return {
        "mean": float(np.mean(flattened)),
        "median": float(np.median(flattened)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
    }


def run_boltz2_compliance_suite(args: argparse.Namespace) -> int:
    if args.enforce_determinism:
        _enforce_determinism()
    login_if_needed(args.token)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)
    output_dir = build_output_dir(args.output_dir, "boltz2_compliance")
    checkpoint_path = _download_checkpoint_if_needed(Path(args.checkpoint_path))
    sequences = SEQUENCE_OPTIONS
    supports_no_kernels = _detect_no_kernels_support()

    model = AutoModel.from_pretrained(args.repo_id, trust_remote_code=True)
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
            "num_ours_samples": 0,
            "num_ref_samples": 0,
            "shared_atoms": 0,
            "shared_ca_atoms": 0,
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
            "tm_cross_median": float("nan"),
            "tm_cross_mean": float("nan"),
            "tm_cross_min": float("nan"),
            "tm_cross_max": float("nan"),
            "tm_ref_within_median": float("nan"),
            "tm_ref_within_mean": float("nan"),
            "tm_ref_within_min": float("nan"),
            "tm_ref_within_max": float("nan"),
            "tm_ours_within_median": float("nan"),
            "tm_ours_within_mean": float("nan"),
            "tm_ours_within_min": float("nan"),
            "tm_ours_within_max": float("nan"),
            "tm_official_vs_ours_csv": "",
            "tm_official_vs_ours_npy": "",
            "tm_official_vs_ours_png": "",
            "tm_official_vs_official_csv": "",
            "tm_official_vs_official_npy": "",
            "tm_official_vs_official_png": "",
            "tm_ours_vs_ours_csv": "",
            "tm_ours_vs_ours_npy": "",
            "tm_ours_vs_ours_png": "",
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

            ours_atom_maps = _build_ours_atom_maps(
                sample_coords=out_ours["sample_atom_coords"],
                atom_mask=feats_ours["atom_pad_mask"][0],
                atom_names=template.atom_names,
                atom_residue_index=template.atom_residue_index,
                atom_chain_id=template.atom_chain_id,
            )
            ref_atom_maps, ref_plddt_samples, ref_confidence_samples = _run_boltz_cli_reference(
                sequence=sequence,
                sequence_index=sequence_index,
                checkpoint_path=checkpoint_path,
                args=args,
                device=device,
                supports_no_kernels=supports_no_kernels,
            )

            row["num_ours_samples"] = len(ours_atom_maps)
            row["num_ref_samples"] = len(ref_atom_maps)

            ours_atom_map_primary = ours_atom_maps[0]
            ref_atom_map_primary = ref_atom_maps[0]
            ref_plddt_primary = ref_plddt_samples[0].float().cpu().reshape(-1)
            ref_confidence_primary = ref_confidence_samples[0]

            shared_keys = []
            for key in ours_atom_map_primary:
                if key in ref_atom_map_primary:
                    shared_keys.append(key)
            shared_keys.sort()
            assert len(shared_keys) > 0, "No overlapping atom keys between our output and pip boltz CLI output."
            row["shared_atoms"] = len(shared_keys)

            shared_ca_keys = _shared_ca_key_order(ours_atom_maps=ours_atom_maps, ref_atom_maps=ref_atom_maps)
            row["shared_ca_atoms"] = len(shared_ca_keys)

            ours_coords_stack = torch.stack([ours_atom_map_primary[key] for key in shared_keys], dim=0)
            ref_coords_stack = torch.stack([ref_atom_map_primary[key] for key in shared_keys], dim=0)
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
            assert ours_plddt.numel() == ref_plddt_primary.numel(), (
                f"pLDDT size mismatch (ours={ours_plddt.numel()}, ref={ref_plddt_primary.numel()})."
            )
            row["plddt_mae"] = float(torch.mean(torch.abs(ours_plddt - ref_plddt_primary)).item())

            ours_ptm = _summary_metric(out_ours["ptm"]).float().cpu()
            ours_iptm = _summary_metric(out_ours["iptm"]).float().cpu()
            ours_complex_plddt = _summary_metric(out_ours["complex_plddt"]).float().cpu()
            ours_confidence_score = _compute_confidence_score(
                ptm=ours_ptm,
                iptm=ours_iptm,
                complex_plddt=ours_complex_plddt,
            )

            ref_ptm = torch.tensor([float(ref_confidence_primary["ptm"])], dtype=torch.float32)
            ref_iptm = torch.tensor([float(ref_confidence_primary["iptm"])], dtype=torch.float32)
            ref_complex_plddt = torch.tensor([float(ref_confidence_primary["complex_plddt"])], dtype=torch.float32)
            ref_confidence_score = torch.tensor([float(ref_confidence_primary["confidence_score"])], dtype=torch.float32)

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
                    ref_atom_map=ref_atom_map_primary,
                )
                ref_cif_path = structure_dir / f"ref_seq{sequence_index}.cif"
                write_cif(
                    structure_template=template,
                    atom_coords=ref_coords_cif,
                    atom_mask=ref_atom_mask_cif,
                    output_path=str(ref_cif_path),
                    plddt=ref_plddt_primary,
                    sample_index=0,
                )
                row["ref_cif_path"] = str(ref_cif_path)

            ours_ca_stack = _stack_coords_for_keys(atom_maps=ours_atom_maps, ordered_keys=shared_ca_keys)
            ref_ca_stack = _stack_coords_for_keys(atom_maps=ref_atom_maps, ordered_keys=shared_ca_keys)

            tm_official_vs_ours = _build_tm_matrix(
                reference_stack=ref_ca_stack,
                subject_stack=ours_ca_stack,
                symmetric=False,
            )
            tm_official_vs_official = _build_tm_matrix(
                reference_stack=ref_ca_stack,
                subject_stack=ref_ca_stack,
                symmetric=True,
            )
            tm_ours_vs_ours = _build_tm_matrix(
                reference_stack=ours_ca_stack,
                subject_stack=ours_ca_stack,
                symmetric=True,
            )

            matrix_dir = output_dir / "tm_matrices" / f"seq_{sequence_index}"
            matrix_dir.mkdir(parents=True, exist_ok=True)
            csv_path, npy_path, png_path = _write_tm_matrix_artifacts(
                matrix_dir=matrix_dir,
                matrix_name="official_vs_ours",
                title=f"Sequence {sequence_index}: official vs ours TM-score",
                matrix=tm_official_vs_ours,
            )
            row["tm_official_vs_ours_csv"] = csv_path
            row["tm_official_vs_ours_npy"] = npy_path
            row["tm_official_vs_ours_png"] = png_path

            csv_path, npy_path, png_path = _write_tm_matrix_artifacts(
                matrix_dir=matrix_dir,
                matrix_name="official_vs_official",
                title=f"Sequence {sequence_index}: official vs official TM-score",
                matrix=tm_official_vs_official,
            )
            row["tm_official_vs_official_csv"] = csv_path
            row["tm_official_vs_official_npy"] = npy_path
            row["tm_official_vs_official_png"] = png_path

            csv_path, npy_path, png_path = _write_tm_matrix_artifacts(
                matrix_dir=matrix_dir,
                matrix_name="ours_vs_ours",
                title=f"Sequence {sequence_index}: ours vs ours TM-score",
                matrix=tm_ours_vs_ours,
            )
            row["tm_ours_vs_ours_csv"] = csv_path
            row["tm_ours_vs_ours_npy"] = npy_path
            row["tm_ours_vs_ours_png"] = png_path

            cross_stats = _matrix_stats(tm_official_vs_ours)
            row["tm_cross_mean"] = cross_stats["mean"]
            row["tm_cross_median"] = cross_stats["median"]
            row["tm_cross_min"] = cross_stats["min"]
            row["tm_cross_max"] = cross_stats["max"]

            ref_within_stats = _matrix_stats(tm_official_vs_official)
            row["tm_ref_within_mean"] = ref_within_stats["mean"]
            row["tm_ref_within_median"] = ref_within_stats["median"]
            row["tm_ref_within_min"] = ref_within_stats["min"]
            row["tm_ref_within_max"] = ref_within_stats["max"]

            ours_within_stats = _matrix_stats(tm_ours_vs_ours)
            row["tm_ours_within_mean"] = ours_within_stats["mean"]
            row["tm_ours_within_median"] = ours_within_stats["median"]
            row["tm_ours_within_min"] = ours_within_stats["min"]
            row["tm_ours_within_max"] = ours_within_stats["max"]

            row["pass"] = bool(row["tm_cross_median"] >= args.tm_pass_threshold)
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
        "write_cif_artifacts": args.write_cif_artifacts,
        "num_sequences": len(sequences),
        "recycling_steps": BOLTZ2_FIXED_RECYCLING_STEPS,
        "num_sampling_steps": BOLTZ2_FIXED_SAMPLING_STEPS,
        "diffusion_samples": BOLTZ2_FIXED_DIFFUSION_SAMPLES,
        "tm_pass_threshold": args.tm_pass_threshold,
        "supports_no_kernels": supports_no_kernels,
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
        f"Recycling steps (fixed): {BOLTZ2_FIXED_RECYCLING_STEPS}",
        f"Diffusion samples (fixed): {BOLTZ2_FIXED_DIFFUSION_SAMPLES}",
        f"TM pass threshold: {args.tm_pass_threshold}",
        f"Write CIF artifacts: {args.write_cif_artifacts}",
        f"Reference CLI supports --no_kernels: {supports_no_kernels}",
    ]
    for row in rows:
        status = "PASS" if bool(row["pass"]) else "FAIL"
        summary_lines.append(
            f"{status} | idx={row['sequence_index']} | seed={row['sequence_seed']} | "
            f"ours_dtype={row['ours_dtype_effective']} | shared_atoms={row['shared_atoms']} | "
            f"shared_ca={row['shared_ca_atoms']} | tm_cross_median={row['tm_cross_median']} | "
            f"tm_ref_within_median={row['tm_ref_within_median']} | "
            f"tm_ours_within_median={row['tm_ours_within_median']} | "
            f"coord_aln_mae={row['coord_mae_aligned']} | plddt_mae={row['plddt_mae']} | "
            f"official_vs_ours_csv={row['tm_official_vs_ours_csv']} | "
            f"official_vs_official_csv={row['tm_official_vs_official_csv']} | "
            f"ours_vs_ours_csv={row['tm_ours_vs_ours_csv']} | "
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
    parser.add_argument("--run-confidence-sequentially", action="store_true")
    parser.add_argument("--coord-mae-threshold", type=float, default=5e-3)
    parser.add_argument("--coord-rmse-threshold", type=float, default=5e-3)
    parser.add_argument("--coord-max-abs-threshold", type=float, default=5e-2)
    parser.add_argument("--plddt-mae-threshold", type=float, default=5e-3)
    parser.add_argument("--summary-metric-abs-threshold", type=float, default=5e-3)
    parser.add_argument("--tm-pass-threshold", type=float, default=0.60)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_boltz2_compliance_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
