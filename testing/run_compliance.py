import entrypoint_setup

import argparse
import time
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from testing.common import (
    LOAD_DTYPE,
    add_base_args,
    add_data_args,
    build_output_dir,
    chunk_sequences,
    extract_official_state_dict,
    generate_sequences,
    load_official_model_for_compliance,
    login_if_needed,
    prepare_official_batch_for_compliance,
    resolve_device,
    set_seed,
)
from testing.model_registry import ModelSpec, get_model_specs
from testing.reporting import write_csv, write_json, write_summary


def _new_row(spec: ModelSpec) -> Dict[str, object]:
    return {
        "model_key": spec.key,
        "family": spec.family,
        "repo_id": spec.repo_id,
        "reference_repo_id": spec.reference_repo_id,
        "weight_parity_pass": False,
        "weight_overlap_param_count": 0,
        "weight_only_in_reference_count": 0,
        "weight_only_in_candidate_count": 0,
        "weight_shape_mismatch_count": 0,
        "weight_diff_param_count": 0,
        "weight_max_abs_diff": float("nan"),
        "weight_max_abs_diff_param": "",
        "output_parity_pass": False,
        "hidden_mse": float("nan"),
        "hidden_mean_abs": float("nan"),
        "hidden_max_abs": float("nan"),
        "logits_mse": float("nan"),
        "logits_mean_abs": float("nan"),
        "logits_max_abs": float("nan"),
        "argmax_accuracy": float("nan"),
        "overall_pass": False,
        "seconds": 0.0,
        "error_type": "",
        "error": "",
    }


def _state_dict_for_candidate(spec: ModelSpec, candidate_model) -> Dict[str, torch.Tensor]:
    state_dict = candidate_model.state_dict()
    if spec.family == "esm2":
        state_dict = {
            name: tensor
            for name, tensor in state_dict.items()
            if "position_embeddings" not in name and "position_ids" not in name
        }
    if spec.family == "dplm2":
        excluded = {"esm.contact_head.regression.weight", "esm.contact_head.regression.bias"}
        state_dict = {name: tensor for name, tensor in state_dict.items() if name not in excluded}
    return state_dict


def _compare_weights_allclose(
    reference_state: Dict[str, torch.Tensor],
    candidate_state: Dict[str, torch.Tensor],
    atol: float,
    rtol: float,
) -> Dict[str, object]:
    reference_keys = set(reference_state.keys())
    candidate_keys = set(candidate_state.keys())
    only_in_reference = sorted(reference_keys - candidate_keys)
    only_in_candidate = sorted(candidate_keys - reference_keys)
    overlap_keys = sorted(reference_keys & candidate_keys)
    shape_mismatches: List[str] = []
    differing_params: List[Dict[str, object]] = []
    max_abs_diff = 0.0
    max_abs_diff_param = ""

    for name in overlap_keys:
        reference_tensor = reference_state[name].detach().cpu().to(torch.float32)
        candidate_tensor = candidate_state[name].detach().cpu().to(torch.float32)
        if reference_tensor.shape != candidate_tensor.shape:
            shape_mismatches.append(name)
            continue
        if torch.allclose(reference_tensor, candidate_tensor, atol=atol, rtol=rtol):
            continue
        abs_diff = torch.abs(candidate_tensor - reference_tensor)
        param_max_abs = float(torch.max(abs_diff).item())
        param_mean_abs = float(torch.mean(abs_diff).item())
        differing_params.append({"name": name, "max_abs_diff": param_max_abs, "mean_abs_diff": param_mean_abs})
        if param_max_abs > max_abs_diff:
            max_abs_diff = param_max_abs
            max_abs_diff_param = name

    return {
        "overlap_param_count": len(overlap_keys),
        "only_in_reference_count": len(only_in_reference),
        "only_in_candidate_count": len(only_in_candidate),
        "shape_mismatch_count": len(shape_mismatches),
        "diff_param_count": len(differing_params),
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_param": max_abs_diff_param,
        "only_in_reference_sample": only_in_reference[:5],
        "only_in_candidate_sample": only_in_candidate[:5],
        "shape_mismatch_sample": shape_mismatches[:5],
        "diff_param_sample": differing_params[:5],
    }


def _masked_tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor, token_mask: torch.Tensor) -> Dict[str, float]:
    assert reference.ndim == 3, f"Expected 3D reference tensor, got {reference.ndim}D."
    assert candidate.ndim == 3, f"Expected 3D candidate tensor, got {candidate.ndim}D."
    assert reference.shape == candidate.shape, "Reference/candidate tensor shapes must match."
    assert token_mask.ndim == 2, f"Expected 2D token mask, got {token_mask.ndim}D."
    assert token_mask.shape == reference.shape[:2], "Token mask must match [batch, seq] shape."
    valid_positions = token_mask.bool()
    assert bool(valid_positions.any()), "Token mask has no valid positions."

    difference = candidate.float() - reference.float()
    valid_difference = difference[valid_positions]
    abs_difference = torch.abs(valid_difference)
    return {
        "mse": float(torch.mean(valid_difference * valid_difference).item()),
        "mean_abs": float(torch.mean(abs_difference).item()),
        "max_abs": float(torch.max(abs_difference).item()),
    }


def _run_output_parity(
    spec: ModelSpec,
    official_model,
    candidate_model,
    official_tokenizer,
    sequence_batches: List[List[str]],
    device: torch.device,
    output_atol: float,
    output_rtol: float,
) -> Dict[str, float]:
    hidden_mse_sum = 0.0
    hidden_mean_abs_sum = 0.0
    hidden_max_abs = 0.0
    logits_mse_sum = 0.0
    logits_mean_abs_sum = 0.0
    logits_max_abs = 0.0
    argmax_accuracy_sum = 0.0
    steps = 0

    for sequence_batch in tqdm(sequence_batches, desc=f"Output compare ({spec.key})", unit="batch", leave=False):
        batch_inputs = prepare_official_batch_for_compliance(
            spec=spec,
            sequence_batch=sequence_batch,
            tokenizer=official_tokenizer,
            device=device,
        )
        token_mask = batch_inputs["attention_mask"].bool()
        model_inputs = dict(batch_inputs)
        if spec.family == "e1":
            del model_inputs["attention_mask"]

        official_outputs = official_model(**model_inputs, output_hidden_states=True)
        candidate_outputs = candidate_model(**model_inputs, output_hidden_states=True)

        reference_hidden = official_outputs.hidden_states[-1]
        candidate_hidden = candidate_outputs.hidden_states[-1]
        reference_logits = official_outputs.logits
        candidate_logits = candidate_outputs.logits

        hidden_metrics = _masked_tensor_metrics(reference_hidden, candidate_hidden, token_mask)
        logits_metrics = _masked_tensor_metrics(reference_logits, candidate_logits, token_mask)
        hidden_mse_sum += hidden_metrics["mse"]
        hidden_mean_abs_sum += hidden_metrics["mean_abs"]
        hidden_max_abs = max(hidden_max_abs, hidden_metrics["max_abs"])
        logits_mse_sum += logits_metrics["mse"]
        logits_mean_abs_sum += logits_metrics["mean_abs"]
        logits_max_abs = max(logits_max_abs, logits_metrics["max_abs"])

        reference_hidden_valid = reference_hidden[token_mask].float()
        candidate_hidden_valid = candidate_hidden[token_mask].float()
        reference_logits_valid = reference_logits[token_mask].float()
        candidate_logits_valid = candidate_logits[token_mask].float()
        assert torch.allclose(reference_hidden_valid, candidate_hidden_valid, atol=output_atol, rtol=output_rtol), (
            f"Hidden state mismatch exceeds tolerance for {spec.repo_id}. "
            f"max_abs={hidden_metrics['max_abs']}, atol={output_atol}, rtol={output_rtol}."
        )
        assert torch.allclose(reference_logits_valid, candidate_logits_valid, atol=output_atol, rtol=output_rtol), (
            f"Logits mismatch exceeds tolerance for {spec.repo_id}. "
            f"max_abs={logits_metrics['max_abs']}, atol={output_atol}, rtol={output_rtol}."
        )

        reference_pred = torch.argmax(reference_logits, dim=-1)
        candidate_pred = torch.argmax(candidate_logits, dim=-1)
        argmax_accuracy_sum += float((reference_pred[token_mask] == candidate_pred[token_mask]).float().mean().item())
        steps += 1

    assert steps > 0, "Output parity requires at least one batch."
    return {
        "hidden_mse": hidden_mse_sum / steps,
        "hidden_mean_abs": hidden_mean_abs_sum / steps,
        "hidden_max_abs": hidden_max_abs,
        "logits_mse": logits_mse_sum / steps,
        "logits_mean_abs": logits_mean_abs_sum / steps,
        "logits_max_abs": logits_max_abs,
        "argmax_accuracy": argmax_accuracy_sum / steps,
    }


def _load_official_pair(spec: ModelSpec, device: torch.device):
    if spec.family == "dplm":
        from dplm_fastplms.load_official import load_official_model

        assert spec.reference_repo_id is not None, f"Missing reference repo id for {spec.key}."
        return load_official_model(reference_repo_id=spec.reference_repo_id, device=device, dtype=LOAD_DTYPE)

    if spec.family == "dplm2":
        from dplm2_fastplms.load_official import load_official_model

        assert spec.reference_repo_id is not None, f"Missing reference repo id for {spec.key}."
        return load_official_model(reference_repo_id=spec.reference_repo_id, device=device, dtype=LOAD_DTYPE)

    if spec.family == "esmplusplus":
        from esm_plusplus.load_official import load_official_model

        assert spec.reference_repo_id is not None, f"Missing reference repo id for {spec.key}."
        return load_official_model(reference_repo_id=spec.reference_repo_id, device=device, dtype=LOAD_DTYPE)

    return load_official_model_for_compliance(spec=spec, device=device, dtype=LOAD_DTYPE)


def _load_candidate_automodel(spec: ModelSpec, device: torch.device, force_download: bool):
    if spec.family in ["dplm", "dplm2"]:
        model_config = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=True)
        auto_model_reference = model_config.auto_map["AutoModelForMaskedLM"]
        resolved_model_class = get_class_from_dynamic_module(auto_model_reference, spec.repo_id)
        resolved_module = __import__(resolved_model_class.__module__, fromlist=["_module"])
        if "EsmForDPLM" in resolved_module.__dict__:
            resolved_module.__dict__["EsmForDPLM"].all_tied_weights_keys = {}
        if "EsmForDPLM2" in resolved_module.__dict__:
            resolved_module.__dict__["EsmForDPLM2"].all_tied_weights_keys = {}

    force_download_options = [force_download]
    if force_download is False:
        force_download_options.append(True)

    last_error = None
    for current_force_download in force_download_options:
        try:
            return AutoModelForMaskedLM.from_pretrained(
                spec.repo_id,
                trust_remote_code=True,
                torch_dtype=LOAD_DTYPE,
                force_download=current_force_download,
            ).to(device=device, dtype=LOAD_DTYPE).eval()
        except AttributeError as exc:
            if "all_tied_weights_keys" in str(exc) and current_force_download is False:
                last_error = exc
                continue
            raise
    assert last_error is not None, "Expected AutoModel retry to capture an AttributeError."
    raise last_error


def run_compliance_suite(args: argparse.Namespace) -> int:
    login_if_needed(args.token)
    device = resolve_device(args.device)
    set_seed(args.seed)
    output_dir = build_output_dir(args.output_dir, "compliance")
    specs = get_model_specs(full_models=args.full_models, families=args.families)
    sequences = generate_sequences(
        num_sequences=args.num_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )
    sequence_batches = chunk_sequences(sequences, args.batch_size)
    assert len(sequence_batches) > 0, "Expected at least one sequence batch."

    all_passed = True
    rows: List[Dict[str, object]] = []
    for spec in tqdm(specs, desc="Compliance models", unit="model"):
        row = _new_row(spec=spec)
        start = time.perf_counter()
        official_model = None
        candidate_model = None
        try:
            official_model, official_tokenizer = _load_official_pair(spec=spec, device=device)
            candidate_model = _load_candidate_automodel(spec=spec, device=device, force_download=args.force_download)

            reference_state = extract_official_state_dict(spec=spec, official_model=official_model)
            candidate_state = _state_dict_for_candidate(spec=spec, candidate_model=candidate_model)
            weight_metrics = _compare_weights_allclose(
                reference_state=reference_state,
                candidate_state=candidate_state,
                atol=args.weight_atol,
                rtol=args.weight_rtol,
            )
            row["weight_overlap_param_count"] = int(weight_metrics["overlap_param_count"])
            row["weight_only_in_reference_count"] = int(weight_metrics["only_in_reference_count"])
            row["weight_only_in_candidate_count"] = int(weight_metrics["only_in_candidate_count"])
            row["weight_shape_mismatch_count"] = int(weight_metrics["shape_mismatch_count"])
            row["weight_diff_param_count"] = int(weight_metrics["diff_param_count"])
            row["weight_max_abs_diff"] = float(weight_metrics["max_abs_diff"])
            row["weight_max_abs_diff_param"] = str(weight_metrics["max_abs_diff_param"])

            assert row["weight_overlap_param_count"] > 0, f"No overlapping parameters for {spec.repo_id}."
            assert row["weight_only_in_reference_count"] == 0, (
                f"Unexpected reference-only parameters for {spec.repo_id}: {weight_metrics['only_in_reference_sample']}"
            )
            assert row["weight_only_in_candidate_count"] == 0, (
                f"Unexpected candidate-only parameters for {spec.repo_id}: {weight_metrics['only_in_candidate_sample']}"
            )
            assert row["weight_shape_mismatch_count"] == 0, (
                f"Weight shape mismatches for {spec.repo_id}: {weight_metrics['shape_mismatch_sample']}"
            )
            assert row["weight_diff_param_count"] == 0, (
                f"Weight value mismatches for {spec.repo_id}: {weight_metrics['diff_param_sample']}"
            )
            row["weight_parity_pass"] = True

            output_metrics = _run_output_parity(
                spec=spec,
                official_model=official_model,
                candidate_model=candidate_model,
                official_tokenizer=official_tokenizer,
                sequence_batches=sequence_batches,
                device=device,
                output_atol=args.output_atol,
                output_rtol=args.output_rtol,
            )
            row["hidden_mse"] = float(output_metrics["hidden_mse"])
            row["hidden_mean_abs"] = float(output_metrics["hidden_mean_abs"])
            row["hidden_max_abs"] = float(output_metrics["hidden_max_abs"])
            row["logits_mse"] = float(output_metrics["logits_mse"])
            row["logits_mean_abs"] = float(output_metrics["logits_mean_abs"])
            row["logits_max_abs"] = float(output_metrics["logits_max_abs"])
            row["argmax_accuracy"] = float(output_metrics["argmax_accuracy"])
            row["output_parity_pass"] = True
            row["overall_pass"] = True
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            all_passed = False
        finally:
            row["seconds"] = round(time.perf_counter() - start, 4)
            rows.append(row)
            if official_model is not None:
                del official_model
            if candidate_model is not None:
                del candidate_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload: Dict[str, object] = {
        "suite": "compliance",
        "all_passed": all_passed,
        "device": str(device),
        "load_dtype": str(LOAD_DTYPE),
        "weight_atol": args.weight_atol,
        "weight_rtol": args.weight_rtol,
        "output_atol": args.output_atol,
        "output_rtol": args.output_rtol,
        "num_sequences": args.num_sequences,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "full_models": args.full_models,
        "rows": rows,
    }
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "metrics.csv", rows)

    passed_count = 0
    for row in rows:
        if bool(row["overall_pass"]):
            passed_count += 1
    summary_lines = [
        "Suite: compliance",
        f"Models tested: {len(rows)}",
        f"Models passed: {passed_count}",
        f"Models failed: {len(rows) - passed_count}",
        f"Output directory: {output_dir}",
    ]
    for row in rows:
        status = "PASS" if bool(row["overall_pass"]) else "FAIL"
        summary_lines.append(
            f"{status} | {row['repo_id']} | weights={row['weight_parity_pass']} "
            f"| outputs={row['output_parity_pass']} | hidden_max_abs={row['hidden_max_abs']} "
            f"| logits_max_abs={row['logits_max_abs']} | argmax={row['argmax_accuracy']} "
            f"| error_type={row['error_type']} | error={row['error']}"
        )
    write_summary(output_dir / "summary.txt", summary_lines)
    print("\n".join(summary_lines))
    if all_passed:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exp-style official-vs-AutoModel compliance checks.")
    add_base_args(parser, include_dry_run=False)
    add_data_args(parser, num_sequences_default=12, min_length_default=16, max_length_default=96, batch_size_default=2)
    parser.add_argument("--weight-atol", type=float, default=1e-6)
    parser.add_argument("--weight-rtol", type=float, default=1e-6)
    parser.add_argument("--output-atol", type=float, default=1e-4)
    parser.add_argument("--output-rtol", type=float, default=1e-4)
    parser.add_argument("--force-download", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_compliance_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
