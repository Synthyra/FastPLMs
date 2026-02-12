import argparse
import pathlib
from typing import Dict, List

from test_scripts.common import ensure_dir
from test_scripts.common import now_timestamp
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary
from test_scripts.run_boltz2_compliance import run_boltz2_compliance_suite
from test_scripts.run_compliance import run_compliance_suite
from test_scripts.run_embedding import run_embedding_suite
from test_scripts.run_throughput import run_throughput_suite


def run_all_suites(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        root_dir = ensure_dir(pathlib.Path("test_scripts") / "results" / now_timestamp() / "run_all")
    else:
        root_dir = ensure_dir(pathlib.Path(args.output_dir))

    compliance_args = argparse.Namespace(
        token=args.token,
        device=args.device,
        dtype=args.compliance_dtype,
        seed=args.seed,
        num_sequences=args.compliance_num_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        batch_size=args.batch_size,
        full_models=args.full_models,
        families=args.families,
        output_dir=str(root_dir / "compliance"),
        attn_tolerance=args.attn_tolerance,
        e1_repeat_tolerance=args.e1_repeat_tolerance,
        skip_reference=args.skip_reference,
        esm2_hidden_mse_threshold=args.esm2_hidden_mse_threshold,
        esm2_logits_mse_threshold=args.esm2_logits_mse_threshold,
        esm2_argmax_threshold=args.esm2_argmax_threshold,
        strict_reference=args.strict_reference,
        dry_run=args.dry_run,
    )
    embedding_args = argparse.Namespace(
        token=args.token,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        num_sequences=args.embedding_num_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        batch_size=args.batch_size,
        full_models=args.full_models,
        families=args.families,
        output_dir=str(root_dir / "embedding"),
        dry_run=args.dry_run,
    )
    throughput_args = argparse.Namespace(
        token=args.token,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        lengths=args.lengths,
        batch_sizes=args.batch_sizes,
        num_batches=args.num_batches,
        warmup_steps=args.warmup_steps,
        timing_runs=args.timing_runs,
        attn_backend=args.attn_backend,
        attn_backends=args.attn_backends,
        compare_attn=args.compare_attn,
        compile_model=args.compile_model,
        pad_min_ratio=args.pad_min_ratio,
        full_models=args.full_models,
        families=args.families,
        output_dir=str(root_dir / "throughput"),
        dry_run=args.dry_run,
    )
    boltz2_compliance_args = argparse.Namespace(
        token=args.token,
        repo_id=args.boltz2_repo_id,
        checkpoint_path=args.boltz2_checkpoint_path,
        device=args.device,
        dtype=args.compliance_dtype,
        seed=args.seed,
        num_sequences=args.boltz2_num_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        enforce_determinism=True,
        write_cif_artifacts=True,
        pass_coord_metric="aligned",
        recycling_steps=args.boltz2_recycling_steps,
        num_sampling_steps=args.boltz2_sampling_steps,
        diffusion_samples=args.boltz2_diffusion_samples,
        run_confidence_sequentially=args.boltz2_run_confidence_sequentially,
        coord_mae_threshold=args.boltz2_coord_mae_threshold,
        coord_rmse_threshold=args.boltz2_coord_rmse_threshold,
        coord_max_abs_threshold=args.boltz2_coord_max_abs_threshold,
        plddt_mae_threshold=args.boltz2_plddt_mae_threshold,
        summary_metric_abs_threshold=args.boltz2_summary_metric_abs_threshold,
        tm_pass_threshold=args.boltz2_tm_pass_threshold,
        output_dir=str(root_dir / "boltz2_compliance"),
    )

    results: List[Dict[str, object]] = []

    print("[run_all] Running compliance suite...")
    compliance_rc = run_compliance_suite(compliance_args)
    results.append({"suite": "compliance", "exit_code": compliance_rc, "output_dir": str(root_dir / "compliance")})

    if args.skip_boltz2_compliance:
        print("[run_all] Skipping boltz2 compliance suite...")
        results.append({"suite": "boltz2_compliance", "exit_code": 0, "output_dir": str(root_dir / "boltz2_compliance")})
    else:
        print("[run_all] Running boltz2 compliance suite...")
        boltz2_compliance_rc = run_boltz2_compliance_suite(boltz2_compliance_args)
        results.append({"suite": "boltz2_compliance", "exit_code": boltz2_compliance_rc, "output_dir": str(root_dir / "boltz2_compliance")})

    print("[run_all] Running embedding suite...")
    embedding_rc = run_embedding_suite(embedding_args)
    results.append({"suite": "embedding", "exit_code": embedding_rc, "output_dir": str(root_dir / "embedding")})

    print("[run_all] Running throughput suite...")
    throughput_rc = run_throughput_suite(throughput_args)
    results.append({"suite": "throughput", "exit_code": throughput_rc, "output_dir": str(root_dir / "throughput")})

    payload: Dict[str, object] = {
        "suite": "run_all",
        "output_dir": str(root_dir),
        "results": results,
    }
    write_json(root_dir / "run_all_metrics.json", payload)

    all_success = True
    summary_lines = [f"Run-all output directory: {root_dir}"]
    for row in results:
        status = "PASS" if int(row["exit_code"]) == 0 else "FAIL"
        summary_lines.append(f"{status} | suite={row['suite']} | exit_code={row['exit_code']} | output_dir={row['output_dir']}")
        if int(row["exit_code"]) != 0:
            all_success = False

    write_summary(root_dir / "run_all_summary.txt", summary_lines)
    print("\n".join(summary_lines))

    if all_success:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run compliance, embedding, and throughput suites together.")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--compliance-dtype", type=str, default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-models", action="store_true")
    parser.add_argument("--families", nargs="+", default=None, choices=["e1", "esm2", "esmplusplus"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embedding-num-sequences", type=int, default=24)
    parser.add_argument("--compliance-num-sequences", type=int, default=12)
    parser.add_argument("--lengths", type=str, default="64,128,256")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timing-runs", type=int, default=4)
    parser.add_argument("--attn-tolerance", type=float, default=5e-3)
    parser.add_argument("--skip-boltz2-compliance", action="store_true")
    parser.add_argument("--boltz2-repo-id", type=str, default="Synthyra/Boltz2")
    parser.add_argument("--boltz2-checkpoint-path", type=str, default="boltz_automodel/weights/boltz2_conf.ckpt")
    parser.add_argument("--boltz2-num-sequences", type=int, default=3)
    parser.add_argument("--boltz2-recycling-steps", type=int, default=3)
    parser.add_argument("--boltz2-sampling-steps", type=int, default=200)
    parser.add_argument("--boltz2-diffusion-samples", type=int, default=200)
    parser.add_argument("--boltz2-run-confidence-sequentially", action="store_true")
    parser.add_argument("--boltz2-coord-mae-threshold", type=float, default=5e-3)
    parser.add_argument("--boltz2-coord-rmse-threshold", type=float, default=5e-3)
    parser.add_argument("--boltz2-coord-max-abs-threshold", type=float, default=5e-2)
    parser.add_argument("--boltz2-plddt-mae-threshold", type=float, default=5e-3)
    parser.add_argument("--boltz2-summary-metric-abs-threshold", type=float, default=5e-3)
    parser.add_argument("--boltz2-tm-pass-threshold", type=float, default=0.60)
    parser.add_argument("--e1-repeat-tolerance", type=float, default=1e-7)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--esm2-hidden-mse-threshold", type=float, default=1e-4)
    parser.add_argument("--esm2-logits-mse-threshold", type=float, default=1e-4)
    parser.add_argument("--esm2-argmax-threshold", type=float, default=0.99)
    parser.add_argument("--attn-backend", type=str, default="flex", choices=["flex", "sdpa", "model_default"])
    parser.add_argument("--attn-backends", type=str, default="sdpa,flex")
    parser.add_argument("--compare-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pad-min-ratio", type=float, default=0.5)
    parser.add_argument("--strict-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_all_suites(args)


if __name__ == "__main__":
    raise SystemExit(main())

