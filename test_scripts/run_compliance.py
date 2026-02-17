import argparse
import math
import time
import traceback
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from test_scripts.common import build_output_dir
from test_scripts.common import chunk_sequences
from test_scripts.common import compare_model_state_dicts_fp32
from test_scripts.common import generate_sequences
from test_scripts.common import load_model
from test_scripts.common import load_official_model_for_compliance
from test_scripts.common import login_if_needed
from test_scripts.common import prepare_official_batch_for_compliance
from test_scripts.common import resolve_device
from test_scripts.common import resolve_dtype
from test_scripts.common import set_seed
from test_scripts.model_registry import ModelSpec
from test_scripts.model_registry import get_model_specs
from test_scripts.reporting import plot_bar
from test_scripts.reporting import write_csv
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary


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
    hidden_mse_threshold: float,
    hidden_mean_abs_threshold: float,
    hidden_max_abs_threshold: float,
    logits_mse_threshold: float,
    logits_mean_abs_threshold: float,
    logits_max_abs_threshold: float,
    argmax_threshold: float,
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
            mean_hidden_mse <= hidden_mse_threshold
            and mean_hidden_abs <= hidden_mean_abs_threshold
            and hidden_max_abs <= hidden_max_abs_threshold
            and mean_logits_mse <= logits_mse_threshold
            and mean_logits_abs <= logits_mean_abs_threshold
            and logits_max_abs <= logits_max_abs_threshold
            and mean_argmax_acc >= argmax_threshold
        )
    else:
        passed = mean_argmax_acc >= argmax_threshold
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


def run_compliance_suite(args: argparse.Namespace) -> int:
    login_if_needed(args.token)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)
    output_dir = build_output_dir(args.output_dir, "compliance")

    specs = get_model_specs(full_models=args.full_models, families=args.families)
    sequences = generate_sequences(num_sequences=args.num_sequences, min_length=args.min_length, max_length=args.max_length, seed=args.seed)
    all_passed = True

    if args.dry_run:
        dry_rows: List[Dict[str, object]] = []
        for spec in specs:
            dry_rows.append({"model_key": spec.key, "family": spec.family, "repo_id": spec.repo_id, "overall_pass": True, "seconds": 0.0, "error": ""})
        payload: Dict[str, object] = {
            "suite": "compliance",
            "all_passed": True,
            "device": str(device),
            "dtype": str(dtype),
            "full_models": args.full_models,
            "dry_run": True,
            "rows": dry_rows,
        }
        write_json(output_dir / "metrics.json", payload)
        write_csv(output_dir / "metrics.csv", dry_rows)
        summary_lines = [f"Suite: compliance (dry-run)", f"Models selected: {len(dry_rows)}", f"Output directory: {output_dir}"]
        for row in dry_rows:
            summary_lines.append(f"SELECTED | {row['repo_id']}")
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
        try:
            if args.skip_reference is False:
                official_model, official_tokenizer = load_official_model_for_compliance(spec=spec, device=device, dtype=dtype)

            for backend in _reference_backends(spec):
                print(f"[compliance] Testing {spec.repo_id} backend={backend} on {device} with {dtype}")
                start = time.perf_counter()
                selected_backend = None if backend == "model_default" else backend
                test_model = None
                row: Dict[str, object] = {
                    "model_key": spec.key,
                    "family": spec.family,
                    "repo_id": spec.repo_id,
                    "attn_backend": backend,
                    "auto_load_pass": False,
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
                    "esmplusplus_fp32_parity": "",
                    "esmplusplus_fp32_parity_max_abs_diff": float("nan"),
                    "esmplusplus_fp32_parity_max_abs_diff_param": "",
                }
                try:
                    test_model, _ = load_model(spec=spec, task="masked_lm", device=device, dtype=dtype, attn_backend=selected_backend)
                    row["auto_load_pass"] = True
                    if args.skip_reference:
                        row["reference_pass"] = True
                    else:
                        assert official_model is not None, "Official model must be loaded when reference check is enabled."
                        assert official_tokenizer is not None, "Official tokenizer must be loaded when reference check is enabled."
                        if args.strict_reference and spec.family == "esmplusplus":
                            parity = compare_model_state_dicts_fp32(
                                reference_model=official_model,
                                candidate_model=test_model,
                                max_report=5,
                            )
                            row["esmplusplus_fp32_parity"] = bool(parity["match"])
                            row["esmplusplus_fp32_parity_max_abs_diff"] = float(parity["max_abs_diff"])
                            row["esmplusplus_fp32_parity_max_abs_diff_param"] = str(parity["max_abs_diff_param"])
                            if bool(parity["match"]) is False:
                                raise AssertionError(
                                    "Strict ESMplusplus reference requires float32-identical checkpoints. "
                                    f"diff_param_count={parity['diff_param_count']} "
                                    f"max_abs_diff={parity['max_abs_diff']} "
                                    f"max_abs_diff_param={parity['max_abs_diff_param']} "
                                    f"only_in_reference={parity['only_in_reference']} "
                                    f"only_in_candidate={parity['only_in_candidate']} "
                                    f"shape_mismatches={parity['shape_mismatches']} "
                                    f"diff_params_sample={parity['diff_params_sample']}"
                                )
                        reference_metrics = _reference_check(
                            spec=spec,
                            test_model=test_model,
                            official_model=official_model,
                            official_tokenizer=official_tokenizer,
                            device=device,
                            sequences=sequences,
                            batch_size=args.batch_size,
                            hidden_mse_threshold=args.hidden_mse_threshold,
                            hidden_mean_abs_threshold=args.hidden_mean_abs_threshold,
                            hidden_max_abs_threshold=args.hidden_max_abs_threshold,
                            logits_mse_threshold=args.logits_mse_threshold,
                            logits_mean_abs_threshold=args.logits_mean_abs_threshold,
                            logits_max_abs_threshold=args.logits_max_abs_threshold,
                            argmax_threshold=args.argmax_threshold,
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
                    row["overall_pass"] = bool(row["auto_load_pass"] and row["reference_pass"])
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
            for backend in _reference_backends(spec):
                row = {
                    "model_key": spec.key,
                    "family": spec.family,
                    "repo_id": spec.repo_id,
                    "attn_backend": backend,
                    "auto_load_pass": False,
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
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                "esmplusplus_fp32_parity": "",
                "esmplusplus_fp32_parity_max_abs_diff": float("nan"),
                "esmplusplus_fp32_parity_max_abs_diff_param": "",
                }
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
        "dtype": str(dtype),
        "num_sequences": args.num_sequences,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "full_models": args.full_models,
        "rows": rows,
        "strict_reference_note": (
            "For esmplusplus, strict_reference validates float32 state_dict equality against official ESMC before metric checks."
            if args.strict_reference
            else ""
        ),
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
    if args.strict_reference:
        summary_lines.append(
            "Strict reference note: esmplusplus requires float32 state_dict equality against official ESMC before metrics."
        )
    for row in rows:
        status = "PASS" if bool(row["overall_pass"]) else "FAIL"
        summary_lines.append(
            f"{status} | {row['repo_id']} | backend={row['attn_backend']} | ref_hidden_mse={row['reference_hidden_mse']} | ref_hidden_max_abs={row['reference_hidden_max_abs']} | ref_logits_mse={row['reference_logits_mse']} | ref_logits_max_abs={row['reference_logits_max_abs']} | argmax={row['reference_argmax_accuracy']} | error_type={row['error_type']} | error={row['error']}"
        )
    write_summary(output_dir / "summary.txt", summary_lines)
    print("\n".join(summary_lines))

    if all_passed:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model compliance and correctness checks.")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sequences", type=int, default=12)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--full-models", action="store_true")
    parser.add_argument("--families", nargs="+", default=None, choices=["e1", "esm2", "esmplusplus"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--attn-tolerance", type=float, default=5e-3)
    parser.add_argument("--check-attn-equivalence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-equivalence-mse-threshold", type=float, default=1e-4)
    parser.add_argument("--attn-equivalence-mean-abs-threshold", type=float, default=2e-3)
    parser.add_argument("--attn-equivalence-max-abs-threshold", type=float, default=8e-2)
    parser.add_argument("--e1-repeat-tolerance", type=float, default=1e-7)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--hidden-mse-threshold", type=float, default=1e-4)
    parser.add_argument("--hidden-mean-abs-threshold", type=float, default=2e-3)
    parser.add_argument("--hidden-max-abs-threshold", type=float, default=8e-2)
    parser.add_argument("--logits-mse-threshold", type=float, default=1e-4)
    parser.add_argument("--logits-mean-abs-threshold", type=float, default=2e-3)
    parser.add_argument("--logits-max-abs-threshold", type=float, default=8e-2)
    parser.add_argument("--argmax-threshold", type=float, default=0.99)
    parser.add_argument("--strict-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-tracebacks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_compliance_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())

