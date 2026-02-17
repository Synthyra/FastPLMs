import argparse
import pathlib
import time
from typing import Dict, List

import torch

from test_scripts.common import build_output_dir
from test_scripts.common import ensure_dir
from test_scripts.common import generate_sequences
from test_scripts.common import load_model
from test_scripts.common import login_if_needed
from test_scripts.common import maybe_tokenizer_for_embedding
from test_scripts.common import resolve_device
from test_scripts.common import resolve_dtype
from test_scripts.common import set_seed
from test_scripts.model_registry import get_model_specs
from test_scripts.reporting import plot_bar
from test_scripts.reporting import write_csv
from test_scripts.reporting import write_json
from test_scripts.reporting import write_summary


def _validate_embedding_dict(embeddings: Dict[str, torch.Tensor], hidden_size: int, full_embeddings: bool) -> int:
    assert len(embeddings) > 0, "Embedding dictionary is empty."
    first_key = next(iter(embeddings))
    sample = embeddings[first_key]
    if full_embeddings:
        assert sample.ndim == 2, "Full embeddings must be rank-2 tensors."
        assert int(sample.shape[1]) == hidden_size, "Full embeddings must match hidden size."
        embedding_dim = int(sample.shape[1])
    else:
        assert sample.ndim == 1, "Pooled embeddings must be rank-1 tensors."
        embedding_dim = int(sample.shape[0])

    for sequence in embeddings:
        tensor = embeddings[sequence]
        if full_embeddings:
            assert tensor.ndim == 2, f"Expected full embedding rank-2 tensor for sequence {sequence}."
            assert int(tensor.shape[1]) == hidden_size, f"Full embedding hidden size mismatch for sequence {sequence}."
        else:
            assert tensor.ndim == 1, f"Expected pooled embedding rank-1 tensor for sequence {sequence}."
        assert torch.isfinite(tensor).all(), f"Found NaN/inf embeddings for sequence {sequence}."
    return embedding_dim


def _assert_embedding_dicts_match(expected: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor]) -> None:
    assert len(expected) == len(observed), "Embedding dictionary size mismatch."
    for sequence in expected:
        assert sequence in observed, f"Missing sequence in observed embeddings: {sequence}"
        expected_tensor = expected[sequence]
        observed_tensor = observed[sequence]
        assert expected_tensor.shape == observed_tensor.shape, f"Shape mismatch for sequence: {sequence}"
        assert expected_tensor.dtype == observed_tensor.dtype, f"Dtype mismatch for sequence: {sequence}"
        assert torch.equal(expected_tensor.cpu(), observed_tensor.cpu()), f"Tensor value mismatch for sequence: {sequence}"


def run_embedding_suite(args: argparse.Namespace) -> int:
    login_if_needed(args.token)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    set_seed(args.seed)

    output_dir = build_output_dir(args.output_dir, "embedding")
    ensure_dir(output_dir / "tmp")

    specs = get_model_specs(full_models=args.full_models, families=args.families)
    sequences = generate_sequences(num_sequences=args.num_sequences, min_length=args.min_length, max_length=args.max_length, seed=args.seed)
    expected_count = len(set(sequences))

    if args.dry_run:
        dry_rows: List[Dict[str, object]] = []
        for spec in specs:
            dry_rows.append({"model_key": spec.key, "family": spec.family, "repo_id": spec.repo_id, "pass": True, "seconds": 0.0, "error": ""})
        payload: Dict[str, object] = {
            "suite": "embedding",
            "all_passed": True,
            "device": str(device),
            "dtype": str(dtype),
            "full_models": args.full_models,
            "dry_run": True,
            "rows": dry_rows,
        }
        write_json(output_dir / "metrics.json", payload)
        write_csv(output_dir / "metrics.csv", dry_rows)
        summary_lines = [f"Suite: embedding (dry-run)", f"Models selected: {len(dry_rows)}", f"Output directory: {output_dir}"]
        for row in dry_rows:
            summary_lines.append(f"SELECTED | {row['repo_id']}")
        write_summary(output_dir / "summary.txt", summary_lines)
        print("\n".join(summary_lines))
        return 0

    rows: List[Dict[str, object]] = []
    total_time_values: List[float] = []
    labels: List[str] = []
    all_passed = True

    for spec in specs:
        print(f"[embedding] Testing {spec.repo_id} on {device} with {dtype}")
        model_start = time.perf_counter()
        row: Dict[str, object] = {
            "model_key": spec.key,
            "family": spec.family,
            "repo_id": spec.repo_id,
            "device": str(device),
            "dtype": str(dtype),
            "pass": False,
            "pooled_embedding_dim": -1,
            "full_embedding_dim": -1,
            "pooled_count": 0,
            "full_count": 0,
            "pooled_sql_count": 0,
            "full_sql_count": 0,
            "pooled_pth_roundtrip_pass": False,
            "full_pth_roundtrip_pass": False,
            "pooled_db_roundtrip_pass": False,
            "full_db_roundtrip_pass": False,
            "seconds": 0.0,
            "error": "",
        }

        tmp_dir = output_dir / "tmp" / spec.key
        ensure_dir(tmp_dir)
        pooled_sql_db_path = tmp_dir / "pooled_embeddings.db"
        full_sql_db_path = tmp_dir / "full_embeddings.db"
        pooled_save_path = tmp_dir / "pooled_embeddings.pth"
        full_save_path = tmp_dir / "full_embeddings.pth"
        artifact_paths = [pooled_sql_db_path, full_sql_db_path, pooled_save_path, full_save_path]
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                artifact_path.unlink()

        try:
            model, _ = load_model(spec=spec, task="base", device=device, dtype=dtype)
            tokenizer = maybe_tokenizer_for_embedding(spec, model)
            hidden_size = int(model.config.hidden_size)

            pooled_embeddings = model.embed_dataset(
                sequences=sequences,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_length,
                truncate=True,
                full_embeddings=False,
                embed_dtype=torch.float32,
                pooling_types=["mean", "cls"],
                sql=False,
                save=True,
                save_path=str(pooled_save_path),
            )
            pooled_dim = _validate_embedding_dict(embeddings=pooled_embeddings, hidden_size=hidden_size, full_embeddings=False)
            row["pooled_count"] = len(pooled_embeddings)
            row["pooled_embedding_dim"] = pooled_dim
            assert int(row["pooled_count"]) == expected_count, "Pooled embedding count mismatch."
            loaded_pooled_pth = model.load_embeddings_from_pth(str(pooled_save_path))
            _validate_embedding_dict(embeddings=loaded_pooled_pth, hidden_size=hidden_size, full_embeddings=False)
            _assert_embedding_dicts_match(expected=pooled_embeddings, observed=loaded_pooled_pth)
            row["pooled_pth_roundtrip_pass"] = True

            full_embeddings = model.embed_dataset(
                sequences=sequences,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_length,
                truncate=True,
                full_embeddings=True,
                embed_dtype=torch.float32,
                sql=False,
                save=True,
                save_path=str(full_save_path),
            )
            full_dim = _validate_embedding_dict(embeddings=full_embeddings, hidden_size=hidden_size, full_embeddings=True)
            row["full_count"] = len(full_embeddings)
            row["full_embedding_dim"] = full_dim
            assert int(row["full_count"]) == expected_count, "Full embedding count mismatch."
            loaded_full_pth = model.load_embeddings_from_pth(str(full_save_path))
            _validate_embedding_dict(embeddings=loaded_full_pth, hidden_size=hidden_size, full_embeddings=True)
            _assert_embedding_dicts_match(expected=full_embeddings, observed=loaded_full_pth)
            row["full_pth_roundtrip_pass"] = True

            _ = model.embed_dataset(
                sequences=sequences,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_length,
                truncate=True,
                full_embeddings=False,
                embed_dtype=torch.float32,
                pooling_types=["mean"],
                sql=True,
                save=False,
                sql_db_path=str(pooled_sql_db_path),
                save_path=str(pooled_save_path),
            )
            pooled_db_embeddings = model.load_embeddings_from_db(str(pooled_sql_db_path))
            row["pooled_sql_count"] = len(pooled_db_embeddings)
            assert int(row["pooled_sql_count"]) == expected_count, "Pooled SQL embedding count mismatch."
            _validate_embedding_dict(embeddings=pooled_db_embeddings, hidden_size=hidden_size, full_embeddings=False)
            _assert_embedding_dicts_match(expected=pooled_embeddings, observed=pooled_db_embeddings)
            row["pooled_db_roundtrip_pass"] = True

            _ = model.embed_dataset(
                sequences=sequences,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_length,
                truncate=True,
                full_embeddings=True,
                embed_dtype=torch.float32,
                pooling_types=["mean"],
                sql=True,
                save=False,
                sql_db_path=str(full_sql_db_path),
                save_path=str(full_save_path),
            )
            full_db_embeddings = model.load_embeddings_from_db(str(full_sql_db_path))
            row["full_sql_count"] = len(full_db_embeddings)
            assert int(row["full_sql_count"]) == expected_count, "Full SQL embedding count mismatch."
            _validate_embedding_dict(embeddings=full_db_embeddings, hidden_size=hidden_size, full_embeddings=True)
            _assert_embedding_dicts_match(expected=full_embeddings, observed=full_db_embeddings)
            row["full_db_roundtrip_pass"] = True

            row["pass"] = bool(
                row["pooled_pth_roundtrip_pass"]
                and row["full_pth_roundtrip_pass"]
                and row["pooled_db_roundtrip_pass"]
                and row["full_db_roundtrip_pass"]
            )
            assert bool(row["pass"]), "Embedding roundtrip check failed."
        except Exception as exc:
            row["error"] = str(exc)
            all_passed = False
        finally:
            model_end = time.perf_counter()
            row["seconds"] = round(model_end - model_start, 4)
            rows.append(row)
            labels.append(spec.key)
            total_time_values.append(float(row["seconds"]))
            if "model" in locals():
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    payload: Dict[str, object] = {
        "suite": "embedding",
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
    plot_bar(output_dir / "embedding_runtime_seconds.png", labels, total_time_values, "Embedding Suite Runtime by Model", "Seconds")

    passed_count = 0
    for row in rows:
        if bool(row["pass"]):
            passed_count += 1
    summary_lines = [
        f"Suite: embedding",
        f"Models tested: {len(rows)}",
        f"Models passed: {passed_count}",
        f"Models failed: {len(rows) - passed_count}",
        f"Output directory: {output_dir}",
    ]
    for row in rows:
        status = "PASS" if bool(row["pass"]) else "FAIL"
        summary_lines.append(f"{status} | {row['repo_id']} | seconds={row['seconds']} | error={row['error']}")
    write_summary(output_dir / "summary.txt", summary_lines)

    print("\n".join(summary_lines))
    if all_passed:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run embedding/mixin validation suite.")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sequences", type=int, default=24)
    parser.add_argument("--min-length", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--full-models", action="store_true")
    parser.add_argument("--families", nargs="+", default=None, choices=["e1", "esm2", "esmplusplus"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_embedding_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())

