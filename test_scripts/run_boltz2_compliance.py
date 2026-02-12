import argparse
import inspect
import time
import urllib.request
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import torch
from transformers import AutoModel

from boltz_automodel.get_boltz2_weights import BOLTZ2_CKPT_URL
from boltz_automodel.minimal_featurizer import build_boltz2_features
from test_scripts.common import build_output_dir
from test_scripts.common import generate_sequences
from test_scripts.common import login_if_needed
from test_scripts.common import resolve_device
from test_scripts.common import resolve_dtype
from test_scripts.common import set_seed
from test_scripts.reporting import write_csv
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary


def _download_checkpoint_if_needed(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        urllib.request.urlretrieve(BOLTZ2_CKPT_URL, str(checkpoint_path))  # noqa: S310
    return checkpoint_path


def _set_sequence_seed(seed: int, sequence_index: int) -> None:
    per_sequence_seed = seed + sequence_index
    set_seed(per_sequence_seed)


def _to_device(feats: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for key, value in feats.items():
        if value.is_floating_point():
            output[key] = value.to(device=device, dtype=dtype)
        else:
            output[key] = value.to(device=device)
    return output


def _clone_feats(feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for key, value in feats.items():
        output[key] = value.clone()
    return output


def _vector_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> Tuple[float, float, float]:
    diff = (lhs.float() - rhs.float()).abs()
    mae = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((lhs.float() - rhs.float()) ** 2)).item())
    max_abs = float(diff.max().item())
    return mae, rmse, max_abs


def _summary_metric(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        return value
    return value.reshape(value.shape[0], -1)[:, 0]


def _compute_confidence_score(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    assert "complex_plddt" in output, "Missing complex_plddt in model output."
    assert "iptm" in output, "Missing iptm in model output."
    assert "ptm" in output, "Missing ptm in model output."
    complex_plddt = _summary_metric(output["complex_plddt"])
    iptm = _summary_metric(output["iptm"])
    ptm = _summary_metric(output["ptm"])
    if torch.allclose(iptm, torch.zeros_like(iptm)):
        return (4 * complex_plddt + ptm) / 5
    return (4 * complex_plddt + iptm) / 5


def _extract_primary_coordinates(output: Dict[str, torch.Tensor], atom_mask: torch.Tensor) -> torch.Tensor:
    assert "sample_atom_coords" in output, "Missing sample_atom_coords in model output."
    coords = output["sample_atom_coords"]
    assert coords.ndim == 3, "Expected sample_atom_coords with shape [samples, atoms, 3]."
    primary = coords[0]
    valid = atom_mask > 0
    return primary[valid]


def _to_plain_python(value):
    if isinstance(value, Mapping):
        output = {}
        for key in value:
            output[key] = _to_plain_python(value[key])
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain_python(item) for item in value]
    return value


def _filter_kwargs_for_callable(callable_obj, kwargs: Dict[str, object], excluded: List[str] | None = None) -> Dict[str, object]:
    signature = inspect.signature(callable_obj)
    allowed = set(signature.parameters.keys())
    if "self" in allowed:
        allowed.remove("self")
    if excluded is not None:
        for key in excluded:
            if key in allowed:
                allowed.remove(key)

    output: Dict[str, object] = {}
    for key in kwargs:
        if key in allowed:
            output[key] = kwargs[key]
    return output


def _strip_prefix_from_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for key in state_dict:
        if key.startswith(prefix):
            output[key[len(prefix):]] = state_dict[key]
        else:
            output[key] = state_dict[key]
    return output


def _choose_best_state_dict_for_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    target_keys = set(model.state_dict().keys())
    candidates = [
        state_dict,
        _strip_prefix_from_state_dict(state_dict, "model."),
        _strip_prefix_from_state_dict(state_dict, "module."),
    ]

    best = candidates[0]
    best_match = -1
    for candidate in candidates:
        match_count = 0
        for key in candidate:
            if key in target_keys:
                match_count += 1
        if match_count > best_match:
            best_match = match_count
            best = candidate
    return best


def _load_reference_model(checkpoint_path: str, device: torch.device):
    from boltz.model.models.boltz2 import Boltz2
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    from boltz.model.modules.diffusionv2 import DiffusionModule

    checkpoint = torch.load(checkpoint_path, map_location=str(device), weights_only=False)
    assert isinstance(checkpoint, dict), "Expected checkpoint to deserialize to a dictionary."
    assert "hyper_parameters" in checkpoint, "Checkpoint missing hyper_parameters."
    assert "state_dict" in checkpoint, "Checkpoint missing state_dict."

    hyper_parameters = _to_plain_python(checkpoint["hyper_parameters"])
    assert isinstance(hyper_parameters, dict), "Expected hyper_parameters to be dict-like."
    state_dict = checkpoint["state_dict"]
    assert isinstance(state_dict, dict), "Expected state_dict to be a dictionary."

    init_kwargs = _filter_kwargs_for_callable(Boltz2.__init__, hyper_parameters)
    if "use_kernels" in inspect.signature(Boltz2.__init__).parameters:
        init_kwargs["use_kernels"] = False

    if "diffusion_process_args" in init_kwargs:
        raw_diffusion_process_args = init_kwargs["diffusion_process_args"]
        assert isinstance(raw_diffusion_process_args, dict), (
            "Expected diffusion_process_args in hyper_parameters to be a dictionary."
        )
        init_kwargs["diffusion_process_args"] = _filter_kwargs_for_callable(
            AtomDiffusion.__init__,
            raw_diffusion_process_args,
            excluded=["score_model_args", "compile_score"],
        )

    if "score_model_args" in init_kwargs:
        raw_score_model_args = init_kwargs["score_model_args"]
        assert isinstance(raw_score_model_args, dict), (
            "Expected score_model_args in hyper_parameters to be a dictionary."
        )
        init_kwargs["score_model_args"] = _filter_kwargs_for_callable(
            DiffusionModule.__init__,
            raw_score_model_args,
        )

    model = Boltz2(**init_kwargs)
    best_state_dict = _choose_best_state_dict_for_model(model, state_dict)
    load_result = model.load_state_dict(best_state_dict, strict=False)
    assert len(load_result.unexpected_keys) == 0, (
        f"Unexpected keys when loading pip boltz reference checkpoint: {load_result.unexpected_keys[:10]}"
    )
    num_missing = len(load_result.missing_keys)
    assert num_missing < 32, (
        "Too many missing keys when loading pip boltz reference checkpoint. "
        f"Missing count: {num_missing}"
    )

    model = model.to(device).eval()
    return model


def run_boltz2_compliance_suite(args: argparse.Namespace) -> int:
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
        dtype=dtype,
    )
    model = model.to(device).eval()
    reference_model = _load_reference_model(str(checkpoint_path), device=device)

    rows: List[Dict[str, object]] = []
    overall_pass = True

    for sequence_index, sequence in enumerate(sequences):
        started = time.perf_counter()
        row: Dict[str, object] = {
            "sequence_index": sequence_index,
            "sequence": sequence,
            "coord_mae": float("nan"),
            "coord_rmse": float("nan"),
            "coord_max_abs": float("nan"),
            "plddt_mae": float("nan"),
            "ptm_abs_diff": float("nan"),
            "iptm_abs_diff": float("nan"),
            "complex_plddt_abs_diff": float("nan"),
            "confidence_score_abs_diff": float("nan"),
            "pass": False,
            "seconds": 0.0,
            "error": "",
        }

        try:
            feats, _ = build_boltz2_features(
                amino_acid_sequence=sequence,
                num_bins=model.config.num_bins,
                atoms_per_window_queries=model.core.input_embedder.atom_encoder.atoms_per_window_queries,
            )
            feats_ours = _to_device(_clone_feats(feats), device=device, dtype=dtype)
            feats_ref = _to_device(_clone_feats(feats), device=device, dtype=dtype)

            with torch.no_grad():
                _set_sequence_seed(args.seed, sequence_index)
                out_ours = model.forward(
                    feats=feats_ours,
                    recycling_steps=args.recycling_steps,
                    num_sampling_steps=args.num_sampling_steps,
                    diffusion_samples=args.diffusion_samples,
                    run_confidence_sequentially=args.run_confidence_sequentially,
                )
                _set_sequence_seed(args.seed, sequence_index)
                out_ref = reference_model.forward(
                    feats=feats_ref,
                    recycling_steps=args.recycling_steps,
                    num_sampling_steps=args.num_sampling_steps,
                    diffusion_samples=args.diffusion_samples,
                    run_confidence_sequentially=args.run_confidence_sequentially,
                )

            atom_mask = feats_ours["atom_pad_mask"][0]
            coords_ours = _extract_primary_coordinates(out_ours, atom_mask)
            coords_ref = _extract_primary_coordinates(out_ref, atom_mask)
            coord_mae, coord_rmse, coord_max_abs = _vector_metrics(coords_ours, coords_ref)
            row["coord_mae"] = coord_mae
            row["coord_rmse"] = coord_rmse
            row["coord_max_abs"] = coord_max_abs

            assert "plddt" in out_ours and "plddt" in out_ref, "Missing pLDDT output."
            plddt_ours = _summary_metric(out_ours["plddt"])
            plddt_ref = _summary_metric(out_ref["plddt"])
            plddt_mae = float(torch.mean(torch.abs(plddt_ours.float() - plddt_ref.float())).item())
            row["plddt_mae"] = plddt_mae

            ptm_ours = _summary_metric(out_ours["ptm"])
            ptm_ref = _summary_metric(out_ref["ptm"])
            iptm_ours = _summary_metric(out_ours["iptm"])
            iptm_ref = _summary_metric(out_ref["iptm"])
            cplddt_ours = _summary_metric(out_ours["complex_plddt"])
            cplddt_ref = _summary_metric(out_ref["complex_plddt"])
            score_ours = _compute_confidence_score(out_ours)
            score_ref = _compute_confidence_score(out_ref)

            row["ptm_abs_diff"] = float(torch.mean(torch.abs(ptm_ours.float() - ptm_ref.float())).item())
            row["iptm_abs_diff"] = float(torch.mean(torch.abs(iptm_ours.float() - iptm_ref.float())).item())
            row["complex_plddt_abs_diff"] = float(torch.mean(torch.abs(cplddt_ours.float() - cplddt_ref.float())).item())
            row["confidence_score_abs_diff"] = float(torch.mean(torch.abs(score_ours.float() - score_ref.float())).item())

            row["pass"] = bool(
                coord_mae <= args.coord_mae_threshold
                and coord_rmse <= args.coord_rmse_threshold
                and coord_max_abs <= args.coord_max_abs_threshold
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
    ]
    for row in rows:
        status = "PASS" if bool(row["pass"]) else "FAIL"
        summary_lines.append(
            f"{status} | idx={row['sequence_index']} | coord_mae={row['coord_mae']} | "
            f"coord_rmse={row['coord_rmse']} | coord_max={row['coord_max_abs']} | "
            f"plddt_mae={row['plddt_mae']} | ptm_diff={row['ptm_abs_diff']} | "
            f"iptm_diff={row['iptm_abs_diff']} | cplddt_diff={row['complex_plddt_abs_diff']} | "
            f"score_diff={row['confidence_score_abs_diff']} | error={row['error']}"
        )
    write_summary(output_dir / "summary.txt", summary_lines)
    print("\n".join(summary_lines))

    if overall_pass:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Boltz2 compliance test against pip boltz reference.")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default="Synthyra/Boltz2")
    parser.add_argument("--checkpoint-path", type=str, default="boltz_automodel/weights/boltz2_conf.ckpt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
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
