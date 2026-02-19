import entrypoint_setup

import argparse
import time
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from testing.common import (
    add_base_args,
    add_data_args,
    build_output_dir,
    ensure_dir,
    generate_sequences,
    load_model,
    LOAD_DTYPE,
    login_if_needed,
    maybe_tokenizer_for_embedding,
    resolve_device,
    resolve_runtime_dtype,
    set_seed,
)
from testing.model_registry import get_model_specs
from testing.reporting import plot_bar, write_csv, write_json, write_summary


def _validate_embedding_dict(
    embeddings: Dict[str, torch.Tensor],
    hidden_size: int,
    full_embeddings: bool,
    expected_dtype: torch.dtype,
) -> int:
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
        assert tensor.dtype == expected_dtype, f"Unexpected embedding dtype for sequence {sequence}: {tensor.dtype}"
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


def _new_row(spec, device: torch.device, runtime_dtype: torch.dtype) -> Dict[str, object]:
    return {
        "model_key": spec.key,
        "family": spec.family,
        "repo_id": spec.repo_id,
        "device": str(device),
        "load_dtype": str(LOAD_DTYPE),
        "runtime_dtype": str(runtime_dtype),
        "pass": False,
        "pooled_embedding_dim": -1,
        "full_embedding_dim": -1,
        "pooled_count": 0,
        "full_count": 0,
        "pooled_sql_count": 0,
        "full_sql_count": 0,
        "dedup_contract_pass": False,
        "pooled_dtype_contract_pass": False,
        "full_dtype_contract_pass": False,
        "full_vs_pooled_contract_pass": False,
        "deterministic_repeat_pass": False,
        "pooled_pth_roundtrip_pass": False,
        "full_pth_roundtrip_pass": False,
        "pooled_db_roundtrip_pass": False,
        "full_db_roundtrip_pass": False,
        "seconds": 0.0,
        "error": "",
    }


def run_embedding_suite(args: argparse.Namespace) -> int:
    login_if_needed(args.token)
    device = resolve_device(args.device)
    runtime_dtype = resolve_runtime_dtype()
    set_seed(args.seed)

    output_dir = build_output_dir(args.output_dir, "embedding")
    ensure_dir(output_dir / "tmp")

    specs = get_model_specs(full_models=args.full_models, families=args.families)
    sequences = generate_sequences(num_sequences=args.num_sequences, min_length=args.min_length, max_length=args.max_length, seed=args.seed)
    expected_count = len(set(sequences))

    if args.dry_run:
        dry_rows: List[Dict[str, object]] = []
        for spec in specs:
            row = _new_row(spec=spec, device=device, runtime_dtype=runtime_dtype)
            row["pass"] = True
            row["dedup_contract_pass"] = True
            row["pooled_dtype_contract_pass"] = True
            row["full_dtype_contract_pass"] = True
            row["full_vs_pooled_contract_pass"] = True
            row["deterministic_repeat_pass"] = True
            row["pooled_pth_roundtrip_pass"] = True
            row["full_pth_roundtrip_pass"] = True
            row["pooled_db_roundtrip_pass"] = True
            row["full_db_roundtrip_pass"] = True
            dry_rows.append(row)
        payload: Dict[str, object] = {
            "suite": "embedding",
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

    for spec in tqdm(specs, desc="Embedding models", unit="model"):
        print(f"[embedding] Testing {spec.repo_id} on {device} with runtime {runtime_dtype}")
        model_start = time.perf_counter()
        row = _new_row(spec=spec, device=device, runtime_dtype=runtime_dtype)
        model = None

        tmp_dir = output_dir / "tmp" / spec.key
        ensure_dir(tmp_dir)
        pooled_sql_db_path = tmp_dir / "pooled_embeddings.db"
        full_sql_db_path = tmp_dir / "full_embeddings.db"
        pooled_save_path = tmp_dir / "pooled_embeddings.pth"
        full_save_path = tmp_dir / "full_embeddings.pth"
        deterministic_save_path = tmp_dir / "deterministic_pooled_embeddings.pth"
        artifact_paths = [pooled_sql_db_path, full_sql_db_path, pooled_save_path, full_save_path, deterministic_save_path]
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                artifact_path.unlink()

        try:
            model, _ = load_model(spec=spec, task="base", device=device, runtime_dtype=runtime_dtype)
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
            pooled_dim = _validate_embedding_dict(
                embeddings=pooled_embeddings,
                hidden_size=hidden_size,
                full_embeddings=False,
                expected_dtype=torch.float32,
            )
            row["pooled_count"] = len(pooled_embeddings)
            row["pooled_embedding_dim"] = pooled_dim
            row["pooled_dtype_contract_pass"] = True
            assert int(row["pooled_count"]) == expected_count, "Pooled embedding count mismatch after dedup/truncate."

            loaded_pooled_pth = model.load_embeddings_from_pth(str(pooled_save_path))
            _validate_embedding_dict(
                embeddings=loaded_pooled_pth,
                hidden_size=hidden_size,
                full_embeddings=False,
                expected_dtype=torch.float32,
            )
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
            full_dim = _validate_embedding_dict(
                embeddings=full_embeddings,
                hidden_size=hidden_size,
                full_embeddings=True,
                expected_dtype=torch.float32,
            )
            row["full_count"] = len(full_embeddings)
            row["full_embedding_dim"] = full_dim
            row["full_dtype_contract_pass"] = True
            assert int(row["full_count"]) == expected_count, "Full embedding count mismatch after dedup/truncate."

            loaded_full_pth = model.load_embeddings_from_pth(str(full_save_path))
            _validate_embedding_dict(
                embeddings=loaded_full_pth,
                hidden_size=hidden_size,
                full_embeddings=True,
                expected_dtype=torch.float32,
            )
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
                pooling_types=["mean", "cls"],
                sql=True,
                save=False,
                sql_db_path=str(pooled_sql_db_path),
                save_path=str(pooled_save_path),
            )
            pooled_db_embeddings = model.load_embeddings_from_db(str(pooled_sql_db_path))
            row["pooled_sql_count"] = len(pooled_db_embeddings)
            assert int(row["pooled_sql_count"]) == expected_count, "Pooled SQL embedding count mismatch."
            _validate_embedding_dict(
                embeddings=pooled_db_embeddings,
                hidden_size=hidden_size,
                full_embeddings=False,
                expected_dtype=torch.float32,
            )
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
            _validate_embedding_dict(
                embeddings=full_db_embeddings,
                hidden_size=hidden_size,
                full_embeddings=True,
                expected_dtype=torch.float32,
            )
            _assert_embedding_dicts_match(expected=full_embeddings, observed=full_db_embeddings)
            row["full_db_roundtrip_pass"] = True

            deterministic_embeddings = model.embed_dataset(
                sequences=sequences,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_length,
                truncate=True,
                full_embeddings=False,
                embed_dtype=torch.float32,
                pooling_types=["mean", "cls"],
                sql=False,
                save=False,
                save_path=str(deterministic_save_path),
            )
            _validate_embedding_dict(
                embeddings=deterministic_embeddings,
                hidden_size=hidden_size,
                full_embeddings=False,
                expected_dtype=torch.float32,
            )
            _assert_embedding_dicts_match(expected=pooled_embeddings, observed=deterministic_embeddings)
            row["deterministic_repeat_pass"] = True

            row["dedup_contract_pass"] = bool(
                int(row["pooled_count"]) == expected_count
                and int(row["full_count"]) == expected_count
                and int(row["pooled_sql_count"]) == expected_count
                and int(row["full_sql_count"]) == expected_count
            )
            assert bool(row["dedup_contract_pass"]), "Dedup contract failed."

            row["full_vs_pooled_contract_pass"] = bool(
                int(row["full_embedding_dim"]) == hidden_size and int(row["pooled_embedding_dim"]) == hidden_size * 2
            )
            assert bool(row["full_vs_pooled_contract_pass"]), "Full-vs-pooled dimension contract failed."

            row["pass"] = bool(
                row["dedup_contract_pass"]
                and row["pooled_dtype_contract_pass"]
                and row["full_dtype_contract_pass"]
                and row["full_vs_pooled_contract_pass"]
                and row["deterministic_repeat_pass"]
                and row["pooled_pth_roundtrip_pass"]
                and row["full_pth_roundtrip_pass"]
                and row["pooled_db_roundtrip_pass"]
                and row["full_db_roundtrip_pass"]
            )
            assert bool(row["pass"]), "Embedding mixin contract check failed."
        except Exception as exc:
            row["error"] = str(exc)
            all_passed = False
        finally:
            model_end = time.perf_counter()
            row["seconds"] = round(model_end - model_start, 4)
            rows.append(row)
            labels.append(spec.key)
            total_time_values.append(float(row["seconds"]))
            if model is not None:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    payload: Dict[str, object] = {
        "suite": "embedding",
        "all_passed": all_passed,
        "device": str(device),
        "load_dtype": str(LOAD_DTYPE),
        "runtime_dtype": str(runtime_dtype),
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
        "Suite: embedding",
        f"Models tested: {len(rows)}",
        f"Models passed: {passed_count}",
        f"Models failed: {len(rows) - passed_count}",
        f"Output directory: {output_dir}",
    ]
    for row in rows:
        status = "PASS" if bool(row["pass"]) else "FAIL"
        summary_lines.append(
            f"{status} | {row['repo_id']} | dedup={row['dedup_contract_pass']} "
            f"| shape_contract={row['full_vs_pooled_contract_pass']} | deterministic={row['deterministic_repeat_pass']} "
            f"| seconds={row['seconds']} | error={row['error']}"
        )
    write_summary(output_dir / "summary.txt", summary_lines)

    print("\n".join(summary_lines))
    if all_passed:
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run embedding mixin contract checks.")
    add_base_args(parser)
    add_data_args(parser, num_sequences_default=24, min_length_default=12, max_length_default=64, batch_size_default=4)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_embedding_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
