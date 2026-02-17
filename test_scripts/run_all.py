import argparse
import dataclasses
import json
import math
import pathlib
from typing import Dict, List

from tqdm.auto import tqdm

from test_scripts.common import ensure_dir
from test_scripts.common import now_timestamp
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary
from test_scripts.run_boltz2_compliance import run_boltz2_compliance_suite
from test_scripts.run_compliance import run_compliance_suite
from test_scripts.run_embedding import run_embedding_suite
from test_scripts.run_throughput import run_throughput_suite


DEFAULTS: Dict[str, object] = {
    "device": "auto",
    "dtype": "auto",
    "compliance_dtype": "float32",
    "seed": 42,
    "min_length": 16,
    "max_length": 96,
    "batch_size": 2,
    "embedding_num_sequences": 24,
    "compliance_num_sequences": 12,
    "lengths": "64,128,256",
    "batch_sizes": "1,2,4",
    "num_batches": 100,
    "warmup_steps": 100,
    "attn_tolerance": 5e-3,
    "check_attn_equivalence": True,
    "attn_equivalence_mse_threshold": 1e-4,
    "attn_equivalence_mean_abs_threshold": 2e-3,
    "attn_equivalence_max_abs_threshold": 8e-2,
    "e1_repeat_tolerance": 1e-7,
    "hidden_mse_threshold": 1e-4,
    "hidden_mean_abs_threshold": 2e-3,
    "hidden_max_abs_threshold": 8e-2,
    "logits_mse_threshold": 1e-4,
    "logits_mean_abs_threshold": 2e-3,
    "logits_max_abs_threshold": 8e-2,
    "argmax_threshold": 0.99,
    "attn_backend": "flex",
    "attn_backends": "sdpa,flex",
    "compare_attn": True,
    "compile_model": True,
    "pad_min_ratio": 0.5,
    "strict_reference": True,
    "print_tracebacks": True,
    "boltz2_repo_id": "Synthyra/Boltz2",
    "boltz2_checkpoint_path": "boltz_fastplms/weights/boltz2_conf.ckpt",
    "boltz2_num_sequences": 3,
    "boltz2_recycling_steps": 3,
    "boltz2_sampling_steps": 200,
    "boltz2_diffusion_samples": 200,
    "boltz2_coord_mae_threshold": 5e-3,
    "boltz2_coord_rmse_threshold": 5e-3,
    "boltz2_coord_max_abs_threshold": 5e-2,
    "boltz2_plddt_mae_threshold": 5e-3,
    "boltz2_summary_metric_abs_threshold": 5e-3,
    "boltz2_tm_pass_threshold": 0.60,
}


@dataclasses.dataclass(frozen=True)
class RunAllConfig:
    token: str | None
    device: str
    dtype: str
    compliance_dtype: str
    seed: int
    full_models: bool
    families: List[str] | None
    output_dir: str | None
    min_length: int
    max_length: int
    batch_size: int
    embedding_num_sequences: int
    compliance_num_sequences: int
    lengths: str
    batch_sizes: str
    num_batches: int
    warmup_steps: int
    attn_tolerance: float
    check_attn_equivalence: bool
    attn_equivalence_mse_threshold: float
    attn_equivalence_mean_abs_threshold: float
    attn_equivalence_max_abs_threshold: float
    skip_boltz2_compliance: bool
    boltz2_repo_id: str
    boltz2_checkpoint_path: str
    boltz2_num_sequences: int
    boltz2_recycling_steps: int
    boltz2_sampling_steps: int
    boltz2_diffusion_samples: int
    boltz2_run_confidence_sequentially: bool
    boltz2_coord_mae_threshold: float
    boltz2_coord_rmse_threshold: float
    boltz2_coord_max_abs_threshold: float
    boltz2_plddt_mae_threshold: float
    boltz2_summary_metric_abs_threshold: float
    boltz2_tm_pass_threshold: float
    e1_repeat_tolerance: float
    skip_reference: bool
    hidden_mse_threshold: float
    hidden_mean_abs_threshold: float
    hidden_max_abs_threshold: float
    logits_mse_threshold: float
    logits_mean_abs_threshold: float
    logits_max_abs_threshold: float
    argmax_threshold: float
    attn_backend: str
    attn_backends: str
    compare_attn: bool
    compile_model: bool
    pad_min_ratio: float
    strict_reference: bool
    print_tracebacks: bool
    dry_run: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunAllConfig":
        return cls(
            token=args.token,
            device=args.device,
            dtype=args.dtype,
            compliance_dtype=args.compliance_dtype,
            seed=args.seed,
            full_models=args.full_models,
            families=args.families,
            output_dir=args.output_dir,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
            embedding_num_sequences=args.embedding_num_sequences,
            compliance_num_sequences=args.compliance_num_sequences,
            lengths=args.lengths,
            batch_sizes=args.batch_sizes,
            num_batches=args.num_batches,
            warmup_steps=args.warmup_steps,
            attn_tolerance=args.attn_tolerance,
            check_attn_equivalence=args.check_attn_equivalence,
            attn_equivalence_mse_threshold=args.attn_equivalence_mse_threshold,
            attn_equivalence_mean_abs_threshold=args.attn_equivalence_mean_abs_threshold,
            attn_equivalence_max_abs_threshold=args.attn_equivalence_max_abs_threshold,
            skip_boltz2_compliance=args.skip_boltz2_compliance,
            boltz2_repo_id=args.boltz2_repo_id,
            boltz2_checkpoint_path=args.boltz2_checkpoint_path,
            boltz2_num_sequences=args.boltz2_num_sequences,
            boltz2_recycling_steps=args.boltz2_recycling_steps,
            boltz2_sampling_steps=args.boltz2_sampling_steps,
            boltz2_diffusion_samples=args.boltz2_diffusion_samples,
            boltz2_run_confidence_sequentially=args.boltz2_run_confidence_sequentially,
            boltz2_coord_mae_threshold=args.boltz2_coord_mae_threshold,
            boltz2_coord_rmse_threshold=args.boltz2_coord_rmse_threshold,
            boltz2_coord_max_abs_threshold=args.boltz2_coord_max_abs_threshold,
            boltz2_plddt_mae_threshold=args.boltz2_plddt_mae_threshold,
            boltz2_summary_metric_abs_threshold=args.boltz2_summary_metric_abs_threshold,
            boltz2_tm_pass_threshold=args.boltz2_tm_pass_threshold,
            e1_repeat_tolerance=args.e1_repeat_tolerance,
            skip_reference=args.skip_reference,
            hidden_mse_threshold=args.hidden_mse_threshold,
            hidden_mean_abs_threshold=args.hidden_mean_abs_threshold,
            hidden_max_abs_threshold=args.hidden_max_abs_threshold,
            logits_mse_threshold=args.logits_mse_threshold,
            logits_mean_abs_threshold=args.logits_mean_abs_threshold,
            logits_max_abs_threshold=args.logits_max_abs_threshold,
            argmax_threshold=args.argmax_threshold,
            attn_backend=args.attn_backend,
            attn_backends=args.attn_backends,
            compare_attn=args.compare_attn,
            compile_model=args.compile_model,
            pad_min_ratio=args.pad_min_ratio,
            strict_reference=args.strict_reference,
            print_tracebacks=args.print_tracebacks,
            dry_run=args.dry_run,
        )

    def resolve_root_dir(self) -> pathlib.Path:
        if self.output_dir is None:
            return ensure_dir(pathlib.Path("test_scripts") / "results" / now_timestamp() / "run_all")
        return ensure_dir(pathlib.Path(self.output_dir))

    def compliance_args(self, root_dir: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            token=self.token,
            device=self.device,
            dtype=self.compliance_dtype,
            seed=self.seed,
            num_sequences=self.compliance_num_sequences,
            min_length=self.min_length,
            max_length=self.max_length,
            batch_size=self.batch_size,
            full_models=self.full_models,
            families=self.families,
            output_dir=str(root_dir / "compliance"),
            attn_tolerance=self.attn_tolerance,
            check_attn_equivalence=self.check_attn_equivalence,
            attn_equivalence_mse_threshold=self.attn_equivalence_mse_threshold,
            attn_equivalence_mean_abs_threshold=self.attn_equivalence_mean_abs_threshold,
            attn_equivalence_max_abs_threshold=self.attn_equivalence_max_abs_threshold,
            e1_repeat_tolerance=self.e1_repeat_tolerance,
            skip_reference=self.skip_reference,
            hidden_mse_threshold=self.hidden_mse_threshold,
            hidden_mean_abs_threshold=self.hidden_mean_abs_threshold,
            hidden_max_abs_threshold=self.hidden_max_abs_threshold,
            logits_mse_threshold=self.logits_mse_threshold,
            logits_mean_abs_threshold=self.logits_mean_abs_threshold,
            logits_max_abs_threshold=self.logits_max_abs_threshold,
            argmax_threshold=self.argmax_threshold,
            strict_reference=self.strict_reference,
            print_tracebacks=self.print_tracebacks,
            dry_run=self.dry_run,
        )

    def embedding_args(self, root_dir: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            token=self.token,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
            num_sequences=self.embedding_num_sequences,
            min_length=self.min_length,
            max_length=self.max_length,
            batch_size=self.batch_size,
            full_models=self.full_models,
            families=self.families,
            output_dir=str(root_dir / "embedding"),
            dry_run=self.dry_run,
        )

    def throughput_args(self, root_dir: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            token=self.token,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
            lengths=self.lengths,
            batch_sizes=self.batch_sizes,
            num_batches=self.num_batches,
            warmup_steps=self.warmup_steps,
            attn_backend=self.attn_backend,
            attn_backends=self.attn_backends,
            compare_attn=self.compare_attn,
            compile_model=self.compile_model,
            pad_min_ratio=self.pad_min_ratio,
            full_models=self.full_models,
            families=self.families,
            output_dir=str(root_dir / "throughput"),
            dry_run=self.dry_run,
        )

    def boltz2_compliance_args(self, root_dir: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            token=self.token,
            repo_id=self.boltz2_repo_id,
            checkpoint_path=self.boltz2_checkpoint_path,
            device=self.device,
            dtype=self.compliance_dtype,
            seed=self.seed,
            num_sequences=self.boltz2_num_sequences,
            min_length=self.min_length,
            max_length=self.max_length,
            enforce_determinism=True,
            write_cif_artifacts=True,
            pass_coord_metric="aligned",
            recycling_steps=self.boltz2_recycling_steps,
            num_sampling_steps=self.boltz2_sampling_steps,
            diffusion_samples=self.boltz2_diffusion_samples,
            run_confidence_sequentially=self.boltz2_run_confidence_sequentially,
            coord_mae_threshold=self.boltz2_coord_mae_threshold,
            coord_rmse_threshold=self.boltz2_coord_rmse_threshold,
            coord_max_abs_threshold=self.boltz2_coord_max_abs_threshold,
            plddt_mae_threshold=self.boltz2_plddt_mae_threshold,
            summary_metric_abs_threshold=self.boltz2_summary_metric_abs_threshold,
            tm_pass_threshold=self.boltz2_tm_pass_threshold,
            output_dir=str(root_dir / "boltz2_compliance"),
        )


def _load_json_payload(path: pathlib.Path) -> Dict[str, object] | None:
    if path.exists() is False:
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), f"Expected JSON payload dict at: {path}"
    return payload


def _suite_headline_metrics(suite: str, suite_output_dir: pathlib.Path) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    payload = _load_json_payload(suite_output_dir / "metrics.json")
    if payload is None:
        metrics["headline"] = "metrics missing"
        return metrics

    if suite == "compliance":
        rows = payload["rows"]
        assert isinstance(rows, list), "Expected compliance rows list."
        total = len(rows)
        passed = 0
        for row in rows:
            if bool(row["overall_pass"]):
                passed += 1
        metrics["models_passed"] = passed
        metrics["models_total"] = total
        metrics["pass_rate"] = 0.0 if total == 0 else passed / total

        attn_payload = _load_json_payload(suite_output_dir / "attention_equivalence_metrics.json")
        if attn_payload is not None:
            attn_rows = attn_payload["rows"]
            assert isinstance(attn_rows, list), "Expected attention equivalence rows list."
            attn_total = len(attn_rows)
            attn_passed = 0
            for row in attn_rows:
                if bool(row["attn_equiv_pass"]):
                    attn_passed += 1
            metrics["attn_equiv_passed"] = attn_passed
            metrics["attn_equiv_total"] = attn_total
        metrics["headline"] = f"models {passed}/{total}"
        return metrics

    if suite == "embedding":
        rows = payload["rows"]
        assert isinstance(rows, list), "Expected embedding rows list."
        total = len(rows)
        passed = 0
        for row in rows:
            if bool(row["pass"]):
                passed += 1
        metrics["models_passed"] = passed
        metrics["models_total"] = total
        metrics["pass_rate"] = 0.0 if total == 0 else passed / total
        metrics["headline"] = f"roundtrip {passed}/{total}"
        return metrics

    if suite == "throughput":
        rows = payload["rows"]
        assert isinstance(rows, list), "Expected throughput rows list."
        total = len(rows)
        passed = 0
        for row in rows:
            if bool(row["pass"]):
                passed += 1
        metrics["points_passed"] = passed
        metrics["points_total"] = total

        delta_payload = _load_json_payload(suite_output_dir / "flex_vs_sdpa_deltas.json")
        if delta_payload is not None:
            delta_rows = delta_payload["rows"]
            assert isinstance(delta_rows, list), "Expected flex-vs-SDPA delta rows list."
            pair_count = len(delta_rows)
            metrics["flex_sdpa_pairs"] = pair_count
            gain_values: List[float] = []
            reduction_values: List[float] = []
            for row in delta_rows:
                gain_values.append(float(row["throughput_gain_percent"]))
                reduction = float(row["memory_reduction_percent"])
                if math.isfinite(reduction):
                    reduction_values.append(reduction)
            if len(gain_values) > 0:
                metrics["mean_throughput_gain_percent"] = sum(gain_values) / len(gain_values)
            if len(reduction_values) > 0:
                metrics["mean_memory_reduction_percent"] = sum(reduction_values) / len(reduction_values)

        metrics["headline"] = f"points {passed}/{total}"
        return metrics

    if suite == "boltz2_compliance":
        if "all_passed" in payload:
            metrics["all_passed"] = bool(payload["all_passed"])
            metrics["headline"] = "pass" if bool(payload["all_passed"]) else "fail"
        else:
            metrics["headline"] = "completed"
        return metrics

    metrics["headline"] = "completed"
    return metrics


def run_all_suites(args: argparse.Namespace) -> int:
    cfg = RunAllConfig.from_args(args)
    root_dir = cfg.resolve_root_dir()

    results: List[Dict[str, object]] = []
    suite_progress = tqdm(total=4, desc="Run-all suites", unit="suite")

    print("[run_all] Running compliance suite...")
    compliance_rc = run_compliance_suite(cfg.compliance_args(root_dir))
    compliance_metrics = _suite_headline_metrics("compliance", root_dir / "compliance")
    results.append({"suite": "compliance", "exit_code": compliance_rc, "output_dir": str(root_dir / "compliance"), "headline_metrics": compliance_metrics})
    suite_progress.update(1)

    if cfg.skip_boltz2_compliance:
        print("[run_all] Skipping boltz2 compliance suite...")
        results.append({"suite": "boltz2_compliance", "exit_code": 0, "output_dir": str(root_dir / "boltz2_compliance")})
        suite_progress.update(1)
    else:
        print("[run_all] Running boltz2 compliance suite...")
        boltz2_compliance_rc = run_boltz2_compliance_suite(cfg.boltz2_compliance_args(root_dir))
        boltz2_metrics = _suite_headline_metrics("boltz2_compliance", root_dir / "boltz2_compliance")
        results.append({"suite": "boltz2_compliance", "exit_code": boltz2_compliance_rc, "output_dir": str(root_dir / "boltz2_compliance"), "headline_metrics": boltz2_metrics})
        suite_progress.update(1)

    print("[run_all] Running embedding suite...")
    embedding_rc = run_embedding_suite(cfg.embedding_args(root_dir))
    embedding_metrics = _suite_headline_metrics("embedding", root_dir / "embedding")
    results.append({"suite": "embedding", "exit_code": embedding_rc, "output_dir": str(root_dir / "embedding"), "headline_metrics": embedding_metrics})
    suite_progress.update(1)

    print("[run_all] Running throughput suite...")
    throughput_rc = run_throughput_suite(cfg.throughput_args(root_dir))
    throughput_metrics = _suite_headline_metrics("throughput", root_dir / "throughput")
    results.append({"suite": "throughput", "exit_code": throughput_rc, "output_dir": str(root_dir / "throughput"), "headline_metrics": throughput_metrics})
    suite_progress.update(1)
    suite_progress.close()

    payload: Dict[str, object] = {
        "suite": "run_all",
        "output_dir": str(root_dir),
        "config": dataclasses.asdict(cfg),
        "results": results,
    }
    write_json(root_dir / "run_all_metrics.json", payload)

    all_success = True
    summary_lines = [f"Run-all output directory: {root_dir}"]
    for row in results:
        status = "PASS" if int(row["exit_code"]) == 0 else "FAIL"
        if "headline_metrics" in row:
            headline_metrics = row["headline_metrics"]
            headline_text = str(headline_metrics["headline"])
        else:
            headline_text = "n/a"
        summary_lines.append(
            f"{status} | suite={row['suite']} | exit_code={row['exit_code']} | headline={headline_text} | output_dir={row['output_dir']}"
        )
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
    parser.add_argument("--device", type=str, default=str(DEFAULTS["device"]))
    parser.add_argument("--dtype", type=str, default=str(DEFAULTS["dtype"]), choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--compliance-dtype", type=str, default=str(DEFAULTS["compliance_dtype"]), choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=int(DEFAULTS["seed"]))
    parser.add_argument("--full-models", action="store_true")
    parser.add_argument("--families", nargs="+", default=None, choices=["e1", "esm2", "esmplusplus"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--min-length", type=int, default=int(DEFAULTS["min_length"]))
    parser.add_argument("--max-length", type=int, default=int(DEFAULTS["max_length"]))
    parser.add_argument("--batch-size", type=int, default=int(DEFAULTS["batch_size"]))
    parser.add_argument("--embedding-num-sequences", type=int, default=int(DEFAULTS["embedding_num_sequences"]))
    parser.add_argument("--compliance-num-sequences", type=int, default=int(DEFAULTS["compliance_num_sequences"]))
    parser.add_argument("--lengths", type=str, default=str(DEFAULTS["lengths"]))
    parser.add_argument("--batch-sizes", type=str, default=str(DEFAULTS["batch_sizes"]))
    parser.add_argument("--num-batches", type=int, default=int(DEFAULTS["num_batches"]))
    parser.add_argument("--warmup-steps", type=int, default=int(DEFAULTS["warmup_steps"]))
    parser.add_argument("--attn-tolerance", type=float, default=float(DEFAULTS["attn_tolerance"]))
    parser.add_argument("--check-attn-equivalence", action=argparse.BooleanOptionalAction, default=bool(DEFAULTS["check_attn_equivalence"]))
    parser.add_argument("--attn-equivalence-mse-threshold", type=float, default=float(DEFAULTS["attn_equivalence_mse_threshold"]))
    parser.add_argument("--attn-equivalence-mean-abs-threshold", type=float, default=float(DEFAULTS["attn_equivalence_mean_abs_threshold"]))
    parser.add_argument("--attn-equivalence-max-abs-threshold", type=float, default=float(DEFAULTS["attn_equivalence_max_abs_threshold"]))
    parser.add_argument("--skip-boltz2-compliance", action="store_true")
    parser.add_argument("--boltz2-repo-id", type=str, default=str(DEFAULTS["boltz2_repo_id"]))
    parser.add_argument("--boltz2-checkpoint-path", type=str, default=str(DEFAULTS["boltz2_checkpoint_path"]))
    parser.add_argument("--boltz2-num-sequences", type=int, default=int(DEFAULTS["boltz2_num_sequences"]))
    parser.add_argument("--boltz2-recycling-steps", type=int, default=int(DEFAULTS["boltz2_recycling_steps"]))
    parser.add_argument("--boltz2-sampling-steps", type=int, default=int(DEFAULTS["boltz2_sampling_steps"]))
    parser.add_argument("--boltz2-diffusion-samples", type=int, default=int(DEFAULTS["boltz2_diffusion_samples"]))
    parser.add_argument("--boltz2-run-confidence-sequentially", action="store_true")
    parser.add_argument("--boltz2-coord-mae-threshold", type=float, default=float(DEFAULTS["boltz2_coord_mae_threshold"]))
    parser.add_argument("--boltz2-coord-rmse-threshold", type=float, default=float(DEFAULTS["boltz2_coord_rmse_threshold"]))
    parser.add_argument("--boltz2-coord-max-abs-threshold", type=float, default=float(DEFAULTS["boltz2_coord_max_abs_threshold"]))
    parser.add_argument("--boltz2-plddt-mae-threshold", type=float, default=float(DEFAULTS["boltz2_plddt_mae_threshold"]))
    parser.add_argument("--boltz2-summary-metric-abs-threshold", type=float, default=float(DEFAULTS["boltz2_summary_metric_abs_threshold"]))
    parser.add_argument("--boltz2-tm-pass-threshold", type=float, default=float(DEFAULTS["boltz2_tm_pass_threshold"]))
    parser.add_argument("--e1-repeat-tolerance", type=float, default=float(DEFAULTS["e1_repeat_tolerance"]))
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--hidden-mse-threshold", type=float, default=float(DEFAULTS["hidden_mse_threshold"]))
    parser.add_argument("--hidden-mean-abs-threshold", type=float, default=float(DEFAULTS["hidden_mean_abs_threshold"]))
    parser.add_argument("--hidden-max-abs-threshold", type=float, default=float(DEFAULTS["hidden_max_abs_threshold"]))
    parser.add_argument("--logits-mse-threshold", type=float, default=float(DEFAULTS["logits_mse_threshold"]))
    parser.add_argument("--logits-mean-abs-threshold", type=float, default=float(DEFAULTS["logits_mean_abs_threshold"]))
    parser.add_argument("--logits-max-abs-threshold", type=float, default=float(DEFAULTS["logits_max_abs_threshold"]))
    parser.add_argument("--argmax-threshold", type=float, default=float(DEFAULTS["argmax_threshold"]))
    parser.add_argument("--attn-backend", type=str, default=str(DEFAULTS["attn_backend"]), choices=["flex", "sdpa", "model_default"])
    parser.add_argument("--attn-backends", type=str, default=str(DEFAULTS["attn_backends"]))
    parser.add_argument("--compare-attn", action=argparse.BooleanOptionalAction, default=bool(DEFAULTS["compare_attn"]))
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=bool(DEFAULTS["compile_model"]))
    parser.add_argument("--pad-min-ratio", type=float, default=float(DEFAULTS["pad_min_ratio"]))
    parser.add_argument("--strict-reference", action=argparse.BooleanOptionalAction, default=bool(DEFAULTS["strict_reference"]))
    parser.add_argument("--print-tracebacks", action=argparse.BooleanOptionalAction, default=bool(DEFAULTS["print_tracebacks"]))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_all_suites(args)


if __name__ == "__main__":
    raise SystemExit(main())

