import argparse
import dataclasses
import json
import pathlib
from typing import Dict, List

from tqdm.auto import tqdm

from testing.common import add_base_args
from testing.common import ensure_dir
from testing.common import now_timestamp
from testing.reporting import write_json
from testing.reporting import write_summary
from testing.run_compliance import run_compliance_suite
from testing.run_embedding import run_embedding_suite
from testing.run_throughput import run_throughput_suite


DEFAULTS: Dict[str, object] = {
    "seed": 42,
    "min_length": 16,
    "max_length": 96,
    "batch_size": 2,
    "parity_num_sequences": 12,
    "embedding_num_sequences": 24,
    "lengths": "64,128,256",
    "batch_sizes": "1,2,4",
    "num_batches": 100,
    "warmup_steps": 100,
    "padded_sequence_fraction": 0.3,
    "pad_fractions": None,
    "max_pad_fraction": 0.5,
    "strict_reference": True,
    "print_tracebacks": True,
}


@dataclasses.dataclass(frozen=True)
class RunAllConfig:
    token: str | None
    device: str
    seed: int
    full_models: bool
    families: List[str] | None
    output_dir: str | None
    dry_run: bool
    min_length: int
    max_length: int
    batch_size: int
    parity_num_sequences: int
    embedding_num_sequences: int
    strict_reference: bool
    skip_reference: bool
    print_tracebacks: bool
    lengths: str
    batch_sizes: str
    num_batches: int
    warmup_steps: int
    padded_sequence_fraction: float
    pad_fractions: str | None
    max_pad_fraction: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunAllConfig":
        return cls(
            token=args.token,
            device=args.device,
            seed=args.seed,
            full_models=args.full_models,
            families=args.families,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
            parity_num_sequences=args.parity_num_sequences,
            embedding_num_sequences=args.embedding_num_sequences,
            strict_reference=args.strict_reference,
            skip_reference=args.skip_reference,
            print_tracebacks=args.print_tracebacks,
            lengths=args.lengths,
            batch_sizes=args.batch_sizes,
            num_batches=args.num_batches,
            warmup_steps=args.warmup_steps,
            padded_sequence_fraction=args.padded_sequence_fraction,
            pad_fractions=args.pad_fractions,
            max_pad_fraction=args.max_pad_fraction,
        )

    def resolve_root_dir(self) -> pathlib.Path:
        if self.output_dir is None:
            return ensure_dir(pathlib.Path("testing") / "results" / now_timestamp() / "run_all")
        return ensure_dir(pathlib.Path(self.output_dir))

    def compliance_args(self, root_dir: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            token=self.token,
            device=self.device,
            seed=self.seed,
            full_models=self.full_models,
            families=self.families,
            output_dir=str(root_dir / "compliance"),
            dry_run=self.dry_run,
            num_sequences=self.parity_num_sequences,
            min_length=self.min_length,
            max_length=self.max_length,
            batch_size=self.batch_size,
            skip_reference=self.skip_reference,
            strict_reference=self.strict_reference,
            print_tracebacks=self.print_tracebacks,
        )

    def embedding_args(self, root_dir: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            token=self.token,
            device=self.device,
            seed=self.seed,
            full_models=self.full_models,
            families=self.families,
            output_dir=str(root_dir / "embedding"),
            dry_run=self.dry_run,
            num_sequences=self.embedding_num_sequences,
            min_length=self.min_length,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

    def throughput_args(self, root_dir: pathlib.Path) -> argparse.Namespace:
        return argparse.Namespace(
            token=self.token,
            device=self.device,
            seed=self.seed,
            full_models=self.full_models,
            families=self.families,
            output_dir=str(root_dir / "throughput"),
            dry_run=self.dry_run,
            lengths=self.lengths,
            batch_sizes=self.batch_sizes,
            num_batches=self.num_batches,
            warmup_steps=self.warmup_steps,
            padded_sequence_fraction=self.padded_sequence_fraction,
            pad_fractions=self.pad_fractions,
            max_pad_fraction=self.max_pad_fraction,
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

    rows = payload["rows"]
    assert isinstance(rows, list), f"Expected list rows in {suite} metrics."

    if suite == "compliance":
        total = len(rows)
        passed = 0
        parity_passed = 0
        for row in rows:
            if bool(row["overall_pass"]):
                passed += 1
            if bool(row["weight_parity_pass"]):
                parity_passed += 1
        metrics["models_passed"] = passed
        metrics["models_total"] = total
        metrics["weight_parity_passed"] = parity_passed
        metrics["headline"] = f"models {passed}/{total}, parity {parity_passed}/{total}"
        return metrics

    if suite == "embedding":
        total = len(rows)
        passed = 0
        for row in rows:
            if bool(row["pass"]):
                passed += 1
        metrics["models_passed"] = passed
        metrics["models_total"] = total
        metrics["headline"] = f"contract {passed}/{total}"
        return metrics

    if suite == "throughput":
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
            metrics["flex_sdpa_pairs"] = len(delta_rows)
        metrics["headline"] = f"points {passed}/{total}"
        return metrics

    metrics["headline"] = "completed"
    return metrics


def run_all_suites(args: argparse.Namespace) -> int:
    cfg = RunAllConfig.from_args(args)
    root_dir = cfg.resolve_root_dir()

    results: List[Dict[str, object]] = []
    suite_progress = tqdm(total=3, desc="Run-all suites", unit="suite")

    print("[run_all] Running compliance/parity suite...")
    compliance_rc = run_compliance_suite(cfg.compliance_args(root_dir))
    compliance_metrics = _suite_headline_metrics("compliance", root_dir / "compliance")
    results.append({"suite": "compliance", "exit_code": compliance_rc, "output_dir": str(root_dir / "compliance"), "headline_metrics": compliance_metrics})
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
        "boltz2_note": "Boltz2 is intentionally separate. Run py -m testing.run_boltz2_compliance.",
    }
    write_json(root_dir / "run_all_metrics.json", payload)

    all_success = True
    summary_lines = [
        f"Run-all output directory: {root_dir}",
        "Boltz2 note: run py -m testing.run_boltz2_compliance separately.",
    ]
    for row in results:
        status = "PASS" if int(row["exit_code"]) == 0 else "FAIL"
        headline_metrics = row["headline_metrics"]
        headline_text = str(headline_metrics["headline"])
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
    parser = argparse.ArgumentParser(description="Run simplified parity, embedding, and throughput suites together.")
    add_base_args(parser)
    parser.add_argument("--min-length", type=int, default=int(DEFAULTS["min_length"]))
    parser.add_argument("--max-length", type=int, default=int(DEFAULTS["max_length"]))
    parser.add_argument("--batch-size", type=int, default=int(DEFAULTS["batch_size"]))
    parser.add_argument("--parity-num-sequences", type=int, default=int(DEFAULTS["parity_num_sequences"]))
    parser.add_argument("--embedding-num-sequences", type=int, default=int(DEFAULTS["embedding_num_sequences"]))
    parser.add_argument("--strict-reference", action=argparse.BooleanOptionalAction, default=bool(DEFAULTS["strict_reference"]))
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--print-tracebacks", action=argparse.BooleanOptionalAction, default=bool(DEFAULTS["print_tracebacks"]))
    parser.add_argument("--lengths", type=str, default=str(DEFAULTS["lengths"]))
    parser.add_argument("--batch-sizes", type=str, default=str(DEFAULTS["batch_sizes"]))
    parser.add_argument("--num-batches", type=int, default=int(DEFAULTS["num_batches"]))
    parser.add_argument("--warmup-steps", type=int, default=int(DEFAULTS["warmup_steps"]))
    parser.add_argument("--padded-sequence-fraction", type=float, default=float(DEFAULTS["padded_sequence_fraction"]))
    parser.add_argument("--pad-fractions", type=str, default=DEFAULTS["pad_fractions"])
    parser.add_argument("--max-pad-fraction", type=float, default=float(DEFAULTS["max_pad_fraction"]))
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_all_suites(args)


if __name__ == "__main__":
    raise SystemExit(main())
