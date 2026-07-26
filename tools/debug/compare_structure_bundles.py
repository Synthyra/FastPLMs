"""Report tensor-level differences between two structure compliance bundles."""

from __future__ import annotations

import argparse
import torch
from collections.abc import Sequence
from pathlib import Path
from safetensors.torch import load_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument(
        "--prefix",
        default="",
        help="Only compare tensor keys beginning with this prefix.",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Only compare keys containing every supplied fragment.",
    )
    parser.add_argument(
        "--max-differences",
        type=int,
        default=1,
        help="Maximum number of differing values to display per tensor.",
    )
    parser.add_argument(
        "--largest-first",
        action="store_true",
        help="Display the largest absolute differences instead of index order.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compare safetensors bundles without loading package or upstream code."""

    arguments = _parser().parse_args(argv)
    actual = load_file(arguments.actual, device="cpu")  # values: (...)
    expected = load_file(arguments.expected, device="cpu")  # values: (...)
    keys = sorted(
        key
        for key in set(actual) | set(expected)
        if key.startswith(arguments.prefix)
        and all(fragment in key for fragment in arguments.contains)
    )
    print(
        "key\tactual_dtype\texpected_dtype\tactual_shape\texpected_shape\texact\t"
        "unequal_values\tmax_absolute_error\trelative_l2\tfirst_difference"
    )
    for key in keys:
        if key not in actual or key not in expected:
            present = "actual" if key in actual else "expected"
            print(f"{key}\tpresent_only_in_{present}")
            continue
        # r is the rank of this bundle entry; shapes vary by tensor key.
        X = actual[key]  # (...)
        X_ref = expected[key]  # (...)
        shape_matches = X.shape == X_ref.shape
        dtype_matches = X.dtype == X_ref.dtype
        exact = shape_matches and dtype_matches and torch.equal(X, X_ref)
        first_difference = ""
        if shape_matches and X.numel() > 0:
            unequal = torch.ne(X, X_ref)  # (...)
            unequal_count = int(unequal.sum().item())
            if unequal_count:
                indices = unequal.nonzero(as_tuple=False)  # (n_diff, r)
                if arguments.largest_first:
                    errors = (X.float() - X_ref.float()).abs()[unequal]  # (n_diff,)
                    order = torch.argsort(errors, descending=True)  # (n_diff,)
                    indices = indices[order]  # (n_diff, r)
                differences = []
                for raw_index in indices[: arguments.max_differences]:
                    # raw_index: (r,)
                    index = tuple(raw_index.tolist())
                    differences.append(f"{index}: {X[index].item()} != {X_ref[index].item()}")
                first_difference = "; ".join(differences)
            max_error = (X.float() - X_ref.float()).abs().max().item()
            error = f"{max_error:.9g}"
            difference_norm = torch.linalg.vector_norm(X.float() - X_ref.float())  # ()
            reference_norm = torch.linalg.vector_norm(X_ref.float()).clamp_min(
                torch.finfo(torch.float32).tiny
            )  # ()
            relative_l2 = f"{(difference_norm / reference_norm).item():.9g}"
        else:
            unequal_count = "n/a"
            error = "n/a"
            relative_l2 = "n/a"
        print(
            f"{key}\t{X.dtype}\t{X_ref.dtype}\t{tuple(X.shape)}\t"
            f"{tuple(X_ref.shape)}\t"
            f"{exact}\t{unequal_count}\t{error}\t{relative_l2}\t"
            f"{first_difference}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
