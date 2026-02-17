import argparse
import math
import time
from typing import Dict, List

import torch

from test_scripts.common import build_output_dir
from test_scripts.common import chunk_sequences
from test_scripts.common import generate_sequences
from test_scripts.common import load_model
from test_scripts.common import load_official_model_for_compliance
from test_scripts.common import login_if_needed
from test_scripts.common import prepare_official_batch_for_compliance
from test_scripts.common import prepare_model_batch
from test_scripts.common import resolve_device
from test_scripts.common import resolve_dtype
from test_scripts.common import run_forward
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


def _forward_contract_check(spec: ModelSpec, model, tokenizer, device: torch.device, sequences: List[str], batch_size: int, attn_tolerance: float) -> Dict[str, object]:
    batches = chunk_sequences(sequences, batch_size)
    assert len(batches) > 0, "Need at least one batch for contract checks."

    first_batch = prepare_model_batch(spec=spec, model=model, tokenizer=tokenizer, sequence_batch=batches[0], device=device)
    outputs_no_attn = run_forward(spec=spec, model=model, batch=first_batch, output_hidden_states=True, output_attentions=False)
    hidden_no_attn = outputs_no_attn.last_hidden_state

    assert hidden_no_attn.ndim == 3, "Expected 3D last_hidden_state."
    assert hidden_no_attn.shape[0] == len(batches[0]), "Batch dimension mismatch."
    finite_pass = bool(torch.isfinite(hidden_no_attn).all())

    attn_pass = True
    max_hidden_diff_attn = float("nan")
    if spec.family == "esmplusplus":
        outputs_with_attn = run_forward(spec=spec, model=model, batch=first_batch, output_hidden_states=True, output_attentions=True)
        hidden_with_attn = outputs_with_attn.last_hidden_state
        max_hidden_diff_attn = float(torch.max(torch.abs(hidden_no_attn - hidden_with_attn)).item())
        attn_pass = max_hidden_diff_attn <= attn_tolerance

    result: Dict[str, object] = {
        "forward_pass": True,
        "finite_pass": finite_pass,
        "attn_pass": attn_pass,
        "max_hidden_diff_attn": max_hidden_diff_attn,
    }
    return result


def _reference_check(
    spec: ModelSpec,
    test_model,
    device: torch.device,
    dtype: torch.dtype,
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
    official_model, official_tokenizer = load_official_model_for_compliance(spec=spec, device=device, dtype=dtype)

    hidden_mse_sum = 0.0
    hidden_mean_abs_sum = 0.0
    hidden_max_abs = 0.0
    logits_mse_sum = 0.0
    logits_mean_abs_sum = 0.0
    logits_max_abs = 0.0
    argmax_acc_sum = 0.0
    steps = 0

    for sequence_batch in chunk_sequences(sequences, batch_size):
        inputs = prepare_official_batch_for_compliance(spec=spec, sequence_batch=sequence_batch, tokenizer=official_tokenizer, device=device)
        official_outputs = official_model(**inputs, output_hidden_states=True)
        test_outputs = test_model(**inputs, output_hidden_states=True)
        hidden_metrics = _tensor_diff_metrics(official_outputs.hidden_states[-1], test_outputs.hidden_states[-1])
        logits_metrics = _tensor_diff_metrics(official_outputs.logits, test_outputs.logits)
        hidden_mse_sum += hidden_metrics["mse"]
        hidden_mean_abs_sum += hidden_metrics["mean_abs"]
        hidden_max_abs = max(hidden_max_abs, hidden_metrics["max_abs"])
        logits_mse_sum += logits_metrics["mse"]
        logits_mean_abs_sum += logits_metrics["mean_abs"]
        logits_max_abs = max(logits_max_abs, logits_metrics["max_abs"])
        ref_argmax = torch.argmax(official_outputs.logits, dim=-1)
        test_argmax = torch.argmax(test_outputs.logits, dim=-1)
        argmax_acc_sum += float((ref_argmax == test_argmax).float().mean().item())
        steps += 1
    assert steps > 0, "Reference check requires at least one batch."

    del official_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

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


def _e1_repeatability_check(spec: ModelSpec, model, tokenizer, device: torch.device, sequences: List[str], batch_size: int, repeat_tolerance: float) -> Dict[str, object]:
    assert spec.family == "e1", "Repeatability check is only for E1."
    batch = prepare_model_batch(spec=spec, model=model, tokenizer=tokenizer, sequence_batch=sequences[:batch_size], device=device)
    output_a = run_forward(spec=spec, model=model, batch=batch, output_hidden_states=True, output_attentions=False).last_hidden_state
    output_b = run_forward(spec=spec, model=model, batch=batch, output_hidden_states=True, output_attentions=False).last_hidden_state
    max_diff = float(torch.max(torch.abs(output_a - output_b)).item())
    passed = max_diff <= repeat_tolerance
    return {
        "e1_repeat_pass": passed,
        "e1_repeat_max_diff": max_diff,
    }


def _attn_equivalence_check(
    spec: ModelSpec,
    device: torch.device,
    dtype: torch.dtype,
    sequences: List[str],
    batch_size: int,
    mse_threshold: float,
    mean_abs_threshold: float,
    max_abs_threshold: float,
) -> Dict[str, object]:
    assert spec.family in ["esm2", "esmplusplus"], "Attention equivalence check is only for ESM2/ESM++."
    model_sdpa, tokenizer_sdpa = load_model(spec=spec, task="base", device=device, dtype=dtype, attn_backend="sdpa")
    model_flex, tokenizer_flex = load_model(spec=spec, task="base", device=device, dtype=dtype, attn_backend="flex")
    if tokenizer_sdpa is None:
        tokenizer = tokenizer_flex
    else:
        tokenizer = tokenizer_sdpa
    assert tokenizer is not None, "Tokenizer is required for attention backend equivalence check."

    mse_sum = 0.0
    mean_abs_sum = 0.0
    max_abs = 0.0
    steps = 0
    for sequence_batch in chunk_sequences(sequences, batch_size):
        prepared = prepare_model_batch(spec=spec, model=model_sdpa, tokenizer=tokenizer, sequence_batch=sequence_batch, device=device)
        output_sdpa = run_forward(spec=spec, model=model_sdpa, batch=prepared, output_hidden_states=True, output_attentions=False).last_hidden_state
        output_flex = run_forward(spec=spec, model=model_flex, batch=prepared, output_hidden_states=True, output_attentions=False).last_hidden_state
        diff_metrics = _tensor_diff_metrics(output_sdpa, output_flex)
        mse_sum += diff_metrics["mse"]
        mean_abs_sum += diff_metrics["mean_abs"]
        max_abs = max(max_abs, diff_metrics["max_abs"])
        steps += 1
    assert steps > 0, "Attention equivalence check requires at least one batch."

    del model_sdpa
    del model_flex
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_mse = mse_sum / steps
    mean_abs = mean_abs_sum / steps
    passed = mean_mse <= mse_threshold and mean_abs <= mean_abs_threshold and max_abs <= max_abs_threshold
    return {
        "attn_equiv_pass": passed,
        "attn_equiv_mse": mean_mse,
        "attn_equiv_mean_abs": mean_abs,
        "attn_equiv_max_abs": max_abs,
    }


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
    attn_diffs: List[float] = []

    for spec in specs:
        print(f"[compliance] Testing {spec.repo_id} on {device} with {dtype}")
        start = time.perf_counter()
        row: Dict[str, object] = {
            "model_key": spec.key,
            "family": spec.family,
            "repo_id": spec.repo_id,
            "auto_load_pass": False,
            "forward_pass": False,
            "finite_pass": False,
            "attn_pass": False,
            "max_hidden_diff_attn": float("nan"),
            "reference_pass": True,
            "reference_hidden_mse": float("nan"),
            "reference_hidden_mean_abs": float("nan"),
            "reference_hidden_max_abs": float("nan"),
            "reference_logits_mse": float("nan"),
            "reference_logits_mean_abs": float("nan"),
            "reference_logits_max_abs": float("nan"),
            "reference_argmax_accuracy": float("nan"),
            "e1_repeat_pass": True,
            "e1_repeat_max_diff": float("nan"),
            "attn_equiv_pass": True,
            "attn_equiv_mse": float("nan"),
            "attn_equiv_mean_abs": float("nan"),
            "attn_equiv_max_abs": float("nan"),
            "overall_pass": False,
            "seconds": 0.0,
            "error": "",
        }

        try:
            base_model, tokenizer = load_model(spec=spec, task="base", device=device, dtype=dtype)
            row["auto_load_pass"] = True

            contract = _forward_contract_check(
                spec=spec,
                model=base_model,
                tokenizer=tokenizer,
                device=device,
                sequences=sequences,
                batch_size=args.batch_size,
                attn_tolerance=args.attn_tolerance,
            )
            row["forward_pass"] = bool(contract["forward_pass"])
            row["finite_pass"] = bool(contract["finite_pass"])
            row["attn_pass"] = bool(contract["attn_pass"])
            row["max_hidden_diff_attn"] = float(contract["max_hidden_diff_attn"])

            if spec.family == "e1":
                repeat_metrics = _e1_repeatability_check(
                    spec=spec,
                    model=base_model,
                    tokenizer=tokenizer,
                    device=device,
                    sequences=sequences,
                    batch_size=args.batch_size,
                    repeat_tolerance=args.e1_repeat_tolerance,
                )
                row["e1_repeat_pass"] = bool(repeat_metrics["e1_repeat_pass"])
                row["e1_repeat_max_diff"] = float(repeat_metrics["e1_repeat_max_diff"])

            if args.check_attn_equivalence and spec.family in ["esm2", "esmplusplus"]:
                attn_metrics = _attn_equivalence_check(
                    spec=spec,
                    device=device,
                    dtype=dtype,
                    sequences=sequences,
                    batch_size=args.batch_size,
                    mse_threshold=args.attn_equivalence_mse_threshold,
                    mean_abs_threshold=args.attn_equivalence_mean_abs_threshold,
                    max_abs_threshold=args.attn_equivalence_max_abs_threshold,
                )
                row["attn_equiv_pass"] = bool(attn_metrics["attn_equiv_pass"])
                row["attn_equiv_mse"] = float(attn_metrics["attn_equiv_mse"])
                row["attn_equiv_mean_abs"] = float(attn_metrics["attn_equiv_mean_abs"])
                row["attn_equiv_max_abs"] = float(attn_metrics["attn_equiv_max_abs"])

            del base_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if args.skip_reference is False:
                row["reference_pass"] = False
                mlm_model, _ = load_model(spec=spec, task="masked_lm", device=device, dtype=dtype)
                reference_metrics = _reference_check(
                    spec=spec,
                    test_model=mlm_model,
                    device=device,
                    dtype=dtype,
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
                del mlm_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            row["overall_pass"] = bool(
                row["auto_load_pass"]
                and row["forward_pass"]
                and row["finite_pass"]
                and row["attn_pass"]
                and row["reference_pass"]
                and row["e1_repeat_pass"]
                and row["attn_equiv_pass"]
            )
            if bool(row["overall_pass"]) is False:
                all_passed = False
        except Exception as exc:
            row["error"] = str(exc)
            all_passed = False
        finally:
            row["seconds"] = round(time.perf_counter() - start, 4)
            rows.append(row)
            labels.append(spec.key)
            value = row["max_hidden_diff_attn"]
            if isinstance(value, float) and math.isnan(value):
                attn_diffs.append(0.0)
            else:
                attn_diffs.append(float(value))

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
    }
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "metrics.csv", rows)
    plot_bar(output_dir / "attention_max_diff.png", labels, attn_diffs, "Max Hidden-State Diff (attentions off vs on)", "Max abs diff")
    equivalence_rows: List[Dict[str, object]] = []
    for row in rows:
        if row["family"] in ["esm2", "esmplusplus"]:
            equivalence_rows.append(
                {
                    "model_key": row["model_key"],
                    "family": row["family"],
                    "repo_id": row["repo_id"],
                    "attn_equiv_pass": row["attn_equiv_pass"],
                    "attn_equiv_mse": row["attn_equiv_mse"],
                    "attn_equiv_mean_abs": row["attn_equiv_mean_abs"],
                    "attn_equiv_max_abs": row["attn_equiv_max_abs"],
                }
            )
    equivalence_payload: Dict[str, object] = {
        "suite": "compliance_attention_equivalence",
        "rows": equivalence_rows,
        "thresholds": {
            "mse": args.attn_equivalence_mse_threshold,
            "mean_abs": args.attn_equivalence_mean_abs_threshold,
            "max_abs": args.attn_equivalence_max_abs_threshold,
        },
    }
    write_json(output_dir / "attention_equivalence_metrics.json", equivalence_payload)
    write_csv(output_dir / "attention_equivalence_metrics.csv", equivalence_rows)

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
            f"{status} | {row['repo_id']} | attn_diff={row['max_hidden_diff_attn']} | attn_equiv_max_abs={row['attn_equiv_max_abs']} | ref_logits_mse={row['reference_logits_mse']} | ref_logits_max_abs={row['reference_logits_max_abs']} | argmax={row['reference_argmax_accuracy']} | error={row['error']}"
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_compliance_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())

