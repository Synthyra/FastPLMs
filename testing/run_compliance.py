import argparse
import math
import time
import traceback
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from testing.common import add_base_args
from testing.common import add_data_args
from testing.common import build_output_dir
from testing.common import chunk_sequences
from testing.common import compare_model_state_dicts_fp32
from testing.common import generate_sequences
from testing.common import load_model
from testing.common import load_official_model_for_compliance
from testing.common import LOAD_DTYPE
from testing.common import login_if_needed
from testing.common import prepare_model_for_runtime
from testing.common import prepare_official_batch_for_compliance
from testing.common import resolve_device
from testing.common import resolve_runtime_dtype
from testing.common import set_seed
from testing.model_registry import ModelSpec
from testing.model_registry import get_model_specs
from testing.reporting import plot_bar
from testing.reporting import write_csv
from testing.reporting import write_json
from testing.reporting import write_summary


STRICT_HIDDEN_MSE_THRESHOLD = 1e-4
STRICT_HIDDEN_MEAN_ABS_THRESHOLD = 2e-3
STRICT_HIDDEN_MAX_ABS_THRESHOLD = 8e-2
STRICT_LOGITS_MSE_THRESHOLD = 1e-4
STRICT_LOGITS_MEAN_ABS_THRESHOLD = 2e-3
STRICT_LOGITS_MAX_ABS_THRESHOLD = 8e-2
ARGMAX_THRESHOLD = 0.99


def _tensor_diff_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    difference = candidate.float() - reference.float()
    abs_difference = torch.abs(difference)
    return {
        "mse": float(torch.mean(difference * difference).item()),
        "mean_abs": float(torch.mean(abs_difference).item()),
        "max_abs": float(torch.max(abs_difference).item()),
    }


def _masked_token_diff_metrics(reference: torch.Tensor, candidate: torch.Tensor, token_mask: torch.Tensor) -> Dict[str, float]:
    assert reference.ndim == 3, f"Expected 3D reference hidden states, got {reference.ndim}D."
    assert candidate.ndim == 3, f"Expected 3D candidate hidden states, got {candidate.ndim}D."
    assert token_mask.ndim == 2, f"Expected 2D token mask, got {token_mask.ndim}D."
    assert reference.shape == candidate.shape, "Reference/candidate shapes must match."
    assert reference.shape[:2] == token_mask.shape, "Token mask must match hidden state batch/sequence dims."
    valid_positions = token_mask.bool()
    assert bool(valid_positions.any()), "Expected at least one valid token position for masked diff metrics."
    difference = candidate.float() - reference.float()
    valid_difference = difference[valid_positions]
    valid_abs_difference = torch.abs(valid_difference)
    return {
        "mse": float(torch.mean(valid_difference * valid_difference).item()),
        "mean_abs": float(torch.mean(valid_abs_difference).item()),
        "max_abs": float(torch.max(valid_abs_difference).item()),
    }


def _masked_argmax_accuracy(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, token_mask: torch.Tensor) -> float:
    assert reference_logits.ndim == 3, f"Expected 3D reference logits, got {reference_logits.ndim}D."
    assert candidate_logits.ndim == 3, f"Expected 3D candidate logits, got {candidate_logits.ndim}D."
    assert token_mask.ndim == 2, f"Expected 2D token mask, got {token_mask.ndim}D."
    assert reference_logits.shape == candidate_logits.shape, "Reference/candidate logits shapes must match."
    assert reference_logits.shape[:2] == token_mask.shape, "Token mask must match logits batch/sequence dims."
    valid_positions = token_mask.bool()
    assert bool(valid_positions.any()), "Expected at least one valid token for argmax accuracy."
    reference_argmax = torch.argmax(reference_logits, dim=-1)
    candidate_argmax = torch.argmax(candidate_logits, dim=-1)
    return float((reference_argmax[valid_positions] == candidate_argmax[valid_positions]).float().mean().item())


def _reference_check(
    spec: ModelSpec,
    test_model,
    official_model,
    official_tokenizer,
    device: torch.device,
    sequences: List[str],
    batch_size: int,
    strict_reference: bool,
) -> Dict[str, object]:
    hidden_mse_sum = 0.0
    hidden_mean_abs_sum = 0.0
    hidden_max_abs = 0.0
    logits_mse_sum = 0.0
    logits_mean_abs_sum = 0.0
    logits_max_abs = 0.0
    argmax_acc_sum = 0.0
    steps = 0

    batches = chunk_sequences(sequences, batch_size)
    for sequence_batch in tqdm(batches, desc=f"Reference compare ({spec.key})", unit="batch", leave=False):
        inputs = prepare_official_batch_for_compliance(spec=spec, sequence_batch=sequence_batch, tokenizer=official_tokenizer, device=device)
        token_mask = inputs["attention_mask"]
        model_inputs = dict(inputs)
        if spec.family == "e1":
            del model_inputs["attention_mask"]
        official_outputs = official_model(**model_inputs, output_hidden_states=True)
        test_outputs = test_model(**model_inputs, output_hidden_states=True)
        hidden_metrics = _masked_token_diff_metrics(official_outputs.hidden_states[-1], test_outputs.hidden_states[-1], token_mask)
        logits_metrics = _masked_token_diff_metrics(official_outputs.logits, test_outputs.logits, token_mask)
        hidden_mse_sum += hidden_metrics["mse"]
        hidden_mean_abs_sum += hidden_metrics["mean_abs"]
        hidden_max_abs = max(hidden_max_abs, hidden_metrics["max_abs"])
        logits_mse_sum += logits_metrics["mse"]
        logits_mean_abs_sum += logits_metrics["mean_abs"]
        logits_max_abs = max(logits_max_abs, logits_metrics["max_abs"])
        argmax_acc_sum += _masked_argmax_accuracy(official_outputs.logits, test_outputs.logits, token_mask)
        steps += 1
    assert steps > 0, "Reference check requires at least one batch."

    mean_hidden_mse = hidden_mse_sum / steps
    mean_hidden_abs = hidden_mean_abs_sum / steps
    mean_logits_mse = logits_mse_sum / steps
    mean_logits_abs = logits_mean_abs_sum / steps
    mean_argmax_acc = argmax_acc_sum / steps
    if strict_reference:
        passed = (
            mean_hidden_mse <= STRICT_HIDDEN_MSE_THRESHOLD
            and mean_hidden_abs <= STRICT_HIDDEN_MEAN_ABS_THRESHOLD
            and hidden_max_abs <= STRICT_HIDDEN_MAX_ABS_THRESHOLD
            and mean_logits_mse <= STRICT_LOGITS_MSE_THRESHOLD
            and mean_logits_abs <= STRICT_LOGITS_MEAN_ABS_THRESHOLD
            and logits_max_abs <= STRICT_LOGITS_MAX_ABS_THRESHOLD
            and mean_argmax_acc >= ARGMAX_THRESHOLD
        )
    else:
        passed = mean_argmax_acc >= ARGMAX_THRESHOLD
    return {
        "reference_pass": passed,
        "reference_hidden_mse": mean_hidden_mse,
        "reference_hidden_mean_abs": mean_hidden_abs,
        "reference_hidden_max_abs": hidden_max_abs,
        "reference_logits_mse": mean_logits_mse,
        "reference_logits_mean_abs": mean_logits_abs,
        "reference_logits_max_abs": logits_max_abs,
        "reference_argmax_accuracy": mean_argmax_acc,
    }


def _reference_backends(spec: ModelSpec) -> List[str]:
    if spec.family in ["esm2", "esmplusplus"]:
        return ["sdpa", "flex"]
    return ["model_default"]


def _blank_weight_parity_metrics() -> Dict[str, object]:
    return {
        "match": False,
        "overlap_param_count": 0,
        "only_in_reference_count": 0,
        "only_in_candidate_count": 0,
        "shape_mismatch_count": 0,
        "diff_param_count": 0,
        "max_abs_diff": float("nan"),
        "max_abs_diff_param": "",
    }


def _apply_weight_parity_to_row(row: Dict[str, object], parity: Dict[str, object]) -> None:
    row["weight_parity_pass"] = bool(parity["match"])
    row["weight_parity_overlap_param_count"] = int(parity["overlap_param_count"])
    row["weight_parity_only_in_reference_count"] = int(parity["only_in_reference_count"])
    row["weight_parity_only_in_candidate_count"] = int(parity["only_in_candidate_count"])
    row["weight_parity_shape_mismatch_count"] = int(parity["shape_mismatch_count"])
    row["weight_parity_diff_param_count"] = int(parity["diff_param_count"])
    row["weight_parity_max_abs_diff"] = float(parity["max_abs_diff"])
    row["weight_parity_max_abs_diff_param"] = str(parity["max_abs_diff_param"])


def _new_row(spec: ModelSpec, backend: str) -> Dict[str, object]:
    return {
        "model_key": spec.key,
        "family": spec.family,
        "repo_id": spec.repo_id,
        "attn_backend": backend,
        "auto_load_pass": False,
        "weight_parity_pass": False,
        "weight_parity_overlap_param_count": 0,
        "weight_parity_only_in_reference_count": 0,
        "weight_parity_only_in_candidate_count": 0,
        "weight_parity_shape_mismatch_count": 0,
        "weight_parity_diff_param_count": 0,
        "weight_parity_max_abs_diff": float("nan"),
        "weight_parity_max_abs_diff_param": "",
        "reference_pass": False,
        "reference_hidden_mse": float("nan"),
        "reference_hidden_mean_abs": float("nan"),
        "reference_hidden_max_abs": float("nan"),
        "reference_logits_mse": float("nan"),
        "reference_logits_mean_abs": float("nan"),
        "reference_logits_max_abs": float("nan"),
        "reference_argmax_accuracy": float("nan"),
        "overall_pass": False,
        "seconds": 0.0,
        "error": "",
        "error_type": "",
        "traceback": "",
    }


def run_compliance_suite(args: argparse.Namespace) -> int:
    login_if_needed(args.token)
    device = resolve_device(args.device)
    runtime_dtype = resolve_runtime_dtype()
    set_seed(args.seed)
    output_dir = build_output_dir(args.output_dir, "compliance")

    specs = get_model_specs(full_models=args.full_models, families=args.families)
    sequences = generate_sequences(num_sequences=args.num_sequences, min_length=args.min_length, max_length=args.max_length, seed=args.seed)
    all_passed = True

    if args.dry_run:
        dry_rows: List[Dict[str, object]] = []
        for spec in specs:
            for backend in _reference_backends(spec):
                row = _new_row(spec=spec, backend=backend)
                row["auto_load_pass"] = True
                row["weight_parity_pass"] = True
                row["reference_pass"] = True
                row["overall_pass"] = True
                dry_rows.append(row)
        payload: Dict[str, object] = {
            "suite": "compliance",
            "all_passed": True,
            "device": str(device),
            "load_dtype": str(LOAD_DTYPE),
            "runtime_dtype": str(runtime_dtype),
            "full_models": args.full_models,
            "dry_run": True,
            "rows": dry_rows,
        }
        write_json(output_dir / "metrics.json", payload)
        write_csv(output_dir / "metrics.csv", dry_rows)
        summary_lines = [f"Suite: compliance (dry-run)", f"Models selected: {len(dry_rows)}", f"Output directory: {output_dir}"]
        for row in dry_rows:
            summary_lines.append(f"SELECTED | {row['repo_id']} | backend={row['attn_backend']}")
        write_summary(output_dir / "summary.txt", summary_lines)
        print("\n".join(summary_lines))
        return 0

    rows: List[Dict[str, object]] = []
    labels: List[str] = []
    hidden_max_abs_values: List[float] = []
    logits_max_abs_values: List[float] = []
    argmax_values: List[float] = []

    for spec in tqdm(specs, desc="Compliance models", unit="model"):
        official_model = None
        official_tokenizer = None
        official_runtime_ready = False
        weight_parity = None
        try:
            if args.skip_reference is False:
                official_model, official_tokenizer = load_official_model_for_compliance(spec=spec, device=device, dtype=LOAD_DTYPE)

            for backend in _reference_backends(spec):
                print(f"[compliance] Testing {spec.repo_id} backend={backend} on {device} with runtime {runtime_dtype}")
                start = time.perf_counter()
                selected_backend = None if backend == "model_default" else backend
                row = _new_row(spec=spec, backend=backend)
                test_model = None
                try:
                    test_model, _ = load_model(
                        spec=spec,
                        task="masked_lm",
                        device=device,
                        runtime_dtype=runtime_dtype,
                        attn_backend=selected_backend,
                        compile_model=False,
                        prepare_for_runtime=False,
                    )
                    row["auto_load_pass"] = True
                    if args.skip_reference:
                        row["weight_parity_pass"] = True
                        row["reference_pass"] = True
                    else:
                        assert official_model is not None, "Official model must be loaded when reference check is enabled."
                        assert official_tokenizer is not None, "Official tokenizer must be loaded when reference check is enabled."

                        if weight_parity is None:
                            weight_parity = compare_model_state_dicts_fp32(
                                reference_model=official_model,
                                candidate_model=test_model,
                                max_report=5,
                            )
                        _apply_weight_parity_to_row(row=row, parity=weight_parity)
                        if args.strict_reference and bool(weight_parity["match"]) is False:
                            raise AssertionError(
                                "Strict reference requires matching-name float32 weights for all families. "
                                f"overlap_param_count={weight_parity['overlap_param_count']} "
                                f"diff_param_count={weight_parity['diff_param_count']} "
                                f"shape_mismatch_count={weight_parity['shape_mismatch_count']} "
                                f"max_abs_diff={weight_parity['max_abs_diff']} "
                                f"max_abs_diff_param={weight_parity['max_abs_diff_param']} "
                                f"only_in_reference_count={weight_parity['only_in_reference_count']} "
                                f"only_in_candidate_count={weight_parity['only_in_candidate_count']}"
                            )

                        if official_runtime_ready is False:
                            official_model = prepare_model_for_runtime(
                                model=official_model,
                                device=device,
                                runtime_dtype=runtime_dtype,
                                compile_model=True,
                            )
                            official_runtime_ready = True
                        test_model = prepare_model_for_runtime(
                            model=test_model,
                            device=device,
                            runtime_dtype=runtime_dtype,
                            compile_model=True,
                        )
                        reference_metrics = _reference_check(
                            spec=spec,
                            test_model=test_model,
                            official_model=official_model,
                            official_tokenizer=official_tokenizer,
                            device=device,
                            sequences=sequences,
                            batch_size=args.batch_size,
                            strict_reference=args.strict_reference,
                        )
                        row["reference_pass"] = bool(reference_metrics["reference_pass"])
                        row["reference_hidden_mse"] = float(reference_metrics["reference_hidden_mse"])
                        row["reference_hidden_mean_abs"] = float(reference_metrics["reference_hidden_mean_abs"])
                        row["reference_hidden_max_abs"] = float(reference_metrics["reference_hidden_max_abs"])
                        row["reference_logits_mse"] = float(reference_metrics["reference_logits_mse"])
                        row["reference_logits_mean_abs"] = float(reference_metrics["reference_logits_mean_abs"])
                        row["reference_logits_max_abs"] = float(reference_metrics["reference_logits_max_abs"])
                        row["reference_argmax_accuracy"] = float(reference_metrics["reference_argmax_accuracy"])
                    row["overall_pass"] = bool(row["auto_load_pass"] and row["weight_parity_pass"] and row["reference_pass"])
                    if bool(row["overall_pass"]) is False:
                        all_passed = False
                except Exception as exc:
                    row["error"] = str(exc)
                    row["error_type"] = type(exc).__name__
                    row["traceback"] = traceback.format_exc()
                    if args.print_tracebacks:
                        print(f"[compliance] Exception while testing {spec.repo_id} backend={backend} ({row['error_type']}): {row['error']}")
                        print(row["traceback"])
                    all_passed = False
                finally:
                    row["seconds"] = round(time.perf_counter() - start, 4)
                    rows.append(row)
                    labels.append(f"{spec.key}|{backend}")
                    hidden_value = row["reference_hidden_max_abs"]
                    logits_value = row["reference_logits_max_abs"]
                    argmax_value = row["reference_argmax_accuracy"]
                    hidden_max_abs_values.append(0.0 if isinstance(hidden_value, float) and math.isnan(hidden_value) else float(hidden_value))
                    logits_max_abs_values.append(0.0 if isinstance(logits_value, float) and math.isnan(logits_value) else float(logits_value))
                    argmax_values.append(0.0 if isinstance(argmax_value, float) and math.isnan(argmax_value) else float(argmax_value))
                    if test_model is not None:
                        del test_model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
        except Exception as exc:
            fallback_weight_parity = _blank_weight_parity_metrics()
            if weight_parity is not None:
                fallback_weight_parity = weight_parity
            for backend in _reference_backends(spec):
                row = _new_row(spec=spec, backend=backend)
                _apply_weight_parity_to_row(row=row, parity=fallback_weight_parity)
                row["error"] = str(exc)
                row["error_type"] = type(exc).__name__
                row["traceback"] = traceback.format_exc()
                rows.append(row)
                labels.append(f"{spec.key}|{backend}")
                hidden_max_abs_values.append(0.0)
                logits_max_abs_values.append(0.0)
                argmax_values.append(0.0)
            all_passed = False
        finally:
            if official_model is not None:
                del official_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload: Dict[str, object] = {
        "suite": "compliance",
        "all_passed": all_passed,
        "device": str(device),
        "load_dtype": str(LOAD_DTYPE),
        "runtime_dtype": str(runtime_dtype),
        "num_sequences": args.num_sequences,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "full_models": args.full_models,
        "strict_reference": args.strict_reference,
        "rows": rows,
    }
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "metrics.csv", rows)
    plot_bar(output_dir / "reference_hidden_max_abs_nonpad.png", labels, hidden_max_abs_values, "Hidden-state max abs diff vs official (non-pad tokens)", "Max abs diff")
    plot_bar(output_dir / "reference_logits_max_abs_nonpad.png", labels, logits_max_abs_values, "Logits max abs diff vs official (non-pad tokens)", "Max abs diff")
    plot_bar(output_dir / "reference_argmax_accuracy_nonpad.png", labels, argmax_values, "Argmax accuracy vs official (non-pad tokens)", "Accuracy")

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
            f"{status} | {row['repo_id']} | backend={row['attn_backend']} | weight_parity={row['weight_parity_pass']} "
            f"| parity_overlap={row['weight_parity_overlap_param_count']} | parity_max_abs={row['weight_parity_max_abs_diff']} "
            f"| ref_hidden_max_abs={row['reference_hidden_max_abs']} | ref_logits_max_abs={row['reference_logits_max_abs']} "
            f"| argmax={row['reference_argmax_accuracy']} | error_type={row['error_type']} | error={row['error']}"
        )
    write_summary(output_dir / "summary.txt", summary_lines)
    print("\n".join(summary_lines))

    if all_passed:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model parity/compliance checks against official references.")
    add_base_args(parser)
    add_data_args(parser, num_sequences_default=12, min_length_default=16, max_length_default=96, batch_size_default=2)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--strict-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-tracebacks", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_compliance_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
