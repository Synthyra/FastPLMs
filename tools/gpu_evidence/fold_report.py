"""Record an ESMFold2 folding sweep as evidence and draw its throughput and memory figure.

``record`` copies the measurements of one ``fold-bench`` run, optionally extended
by ``fold-bench-long`` runs, into a tracked evidence file. ``plot`` draws the
figure and writes its caption from that file alone, so both can be regenerated
without a GPU. The figure carries only axes and a legend; every condition a
reader needs is in the caption.
"""

from __future__ import annotations

import argparse
import json

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .source import ROOT


EVIDENCE_PATH = ROOT / "docs/evidence/esmfold2/folding_cost.json"
FIGURE_PATH = ROOT / "docs/assets/esmfold2_folding_cost.png"
CAPTION_PATH = ROOT / "docs/assets/esmfold2_folding_cost_caption.md"
GIB = 1024**3
MIB = 1024**2
OFFICIAL = "upstream-esmfold2"
DEFAULTS = "fastplms-esmfold2-dense"
OPTIMIZED = "fastplms-esmfold2-windowed-unchunked"
# Figure curves in legend order: evidence label, legend text, color, line style, marker.
# ESMFold2-600 and ESMFold2-300 share one trunk and nearly coincide, so the first is
# drawn with a larger marker that stays visible beneath the second.
CURVES = (
    (OFFICIAL, "ESMFold2, official", "#6b7280", "-", ("o", 5.5)),
    (DEFAULTS, "ESMFold2, FastPLMs defaults", "#2563eb", (0, (4, 2)), ("o", 4.5)),
    (OPTIMIZED, "ESMFold2, FastPLMs optimized", "#2563eb", "-", ("o", 5.5)),
    (
        "fastplms-esmfold2_600-windowed-unchunked",
        "ESMFold2-600, FastPLMs optimized",
        "#d97706",
        "-",
        ("s", 8.0),
    ),
    (
        "fastplms-esmfold2_300-windowed-unchunked",
        "ESMFold2-300, FastPLMs optimized",
        "#059669",
        "-",
        ("^", 5.5),
    ),
)
# Where unchunked pair updates exhaust memory, the optimized curve continues with this
# series, drawn hollow.
OPTIMIZED_FALLBACK = "fastplms-esmfold2-windowed-chunk512"
# Series the caption may name although the figure does not draw as their own curve.
UNPLOTTED_LEGENDS = {
    OPTIMIZED_FALLBACK: "ESMFold2, FastPLMs windowed with 512-row chunks",
}
Point = dict[str, Any]


def _bench_series(run_directory: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Whether a run was a smoke run, and the series it printed.

    A stage prints one closing document. A long stage also prints each series as it
    finishes, which is what remains if the stage ran out of time.
    """
    output = (run_directory / "output.txt").read_text(encoding="utf-8")
    lines = output.splitlines()
    for line in reversed(lines):
        if line.startswith('{"smoke"'):
            document: dict[str, Any] = json.loads(line)
            return bool(document["smoke"]), list(document["series"])
    partial = [
        json.loads(line.removeprefix("FOLD_BENCH_SERIES "))
        for line in lines
        if line.startswith("FOLD_BENCH_SERIES ")
    ]
    if not partial:
        raise ValueError(f"No fold-bench result in {run_directory / 'output.txt'}.")
    return False, partial


def _environment(run_directory: Path) -> dict[str, Any]:
    receipt = json.loads((run_directory / "receipt.json").read_text(encoding="utf-8"))
    environment: dict[str, Any] = receipt["result"]["environment"]
    return environment


def record(run_directory: Path, extension_directories: list[Path]) -> dict[str, Any]:
    receipt = json.loads((run_directory / "receipt.json").read_text(encoding="utf-8"))
    smoke, series = _bench_series(run_directory)
    if smoke:
        raise ValueError("A smoke run uses shortened folds and is not evidence.")
    by_label = {entry["label"]: entry for entry in series}
    anchors: list[dict[str, Any]] = []
    for extension in extension_directories:
        if _environment(extension) != receipt["result"]["environment"]:
            raise ValueError(f"{extension.name} ran in a different environment than the sweep.")
        for entry in _bench_series(extension)[1]:
            base = by_label.get(entry["label"])
            if base is None or entry["status"] != "ok":
                series.append({**entry, "run": extension.name})
                by_label.setdefault(entry["label"], series[-1])
                continue
            measured = {point["length"]: point for point in base["lengths"]}
            for point in entry["lengths"]:
                if point["length"] not in measured:
                    base["lengths"].append({**point, "run": extension.name})
                elif point["status"] == "ok":
                    # A length both runs measured shows how far the two workers agree.
                    anchors.append(
                        {
                            "label": entry["label"],
                            "length": point["length"],
                            "sweep_seconds": measured[point["length"]]["median_seconds"],
                            "extension_seconds": point["median_seconds"],
                        }
                    )
    evidence = {
        "description": (
            "ESMFold2 single-chain folding cost by protein length. Each series ran in its "
            "own process on one worker, so ratios between series of one run are within-run. "
            "Sequences are fixed pseudo-random proteins; timings are medians of end-to-end "
            "infer_protein calls after a shortened warm-up fold at the same length. Points "
            "marked single_pass timed one instrumented fold without a warm-up."
        ),
        "run": run_directory.name,
        "extension_runs": [extension.name for extension in extension_directories],
        "cross_run_anchors": anchors,
        "source": receipt["source"],
        "baseline_revision": receipt["baseline_revision"],
        "environment": receipt["result"]["environment"],
        "series": series,
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    return evidence


def _measured(series: dict[str, Any]) -> list[Point]:
    points = [point for point in series["lengths"] if point["status"] == "ok"]
    return sorted(points, key=lambda point: point["length"])


def _series_by_label(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        series["label"]: series
        for series in evidence["series"]
        if series["status"] == "ok" and "lengths" in series
    }


def _fallback_points(by_label: dict[str, dict[str, Any]]) -> list[Point]:
    """Fallback measurements at lengths the optimized ESMFold2 series could not fold."""
    if OPTIMIZED not in by_label or OPTIMIZED_FALLBACK not in by_label:
        return []
    folded = {point["length"] for point in _measured(by_label[OPTIMIZED])}
    return [p for p in _measured(by_label[OPTIMIZED_FALLBACK]) if p["length"] not in folded]


def _ratios(
    numerator: dict[str, Any], denominator: dict[str, Any], value: Callable[[Point], float]
) -> dict[int, float]:
    """``numerator / denominator`` of one quantity at every length both series measured."""
    below = {point["length"]: point for point in _measured(denominator)}
    return {
        point["length"]: value(point) / value(below[point["length"]])
        for point in _measured(numerator)
        if point["length"] in below
    }


def caption(evidence: dict[str, Any]) -> str:
    """Everything the figure leaves out: conditions, definitions, and headline ratios."""
    by_label = _series_by_label(evidence)
    official, optimized, defaults = by_label[OFFICIAL], by_label[OPTIMIZED], by_label[DEFAULTS]
    settings = optimized["settings"]
    environment = evidence["environment"]
    speed = _ratios(optimized, official, lambda point: point["residues_per_second"])
    exact_speed = _ratios(defaults, official, lambda point: point["residues_per_second"])
    memory = _ratios(optimized, official, lambda point: point["peak_allocated_bytes"])
    shortest, longest = min(speed), max(speed)
    legends = {label: legend for label, legend, *_ in CURVES} | UNPLOTTED_LEGENDS
    out_of_memory = sorted(
        f"{legends.get(series['label'], series['label'])} at {point['length']:,} residues"
        for series in evidence["series"]
        for point in series.get("lengths", [])
        if point["status"] == "out_of_memory"
    )
    single_pass = sorted(
        {
            point["length"]
            for series in by_label.values()
            for point in _measured(series)
            if point.get("single_pass")
        }
    )
    sentences = [
        f"**ESMFold2 single-chain folding cost by protein length on one {environment['gpu']}.** "
        "Left: throughput in residues per second. Middle: peak allocated GPU memory. Right: "
        "working memory per residue, which is peak allocated memory above the loaded weights "
        "divided by the protein length.",
        f"Every fold uses {settings['num_loops']} trunk loops, "
        f"{settings['num_sampling_steps']} requested sampling steps under the official noise cap "
        f"of {settings['max_inference_sigma']:g} (which runs 35 of them), and "
        f"{settings['num_diffusion_samples']} diffusion sample, on one fixed pseudo-random "
        f"protein per length, with PyTorch {environment['torch']}, BF16 autocast over FP32 "
        "folding parameters, and a BF16 ESMC backbone.",
        "Times are medians of two or more end-to-end `infer_protein` calls after a shortened "
        "warm-up fold at the same length"
        + (
            f"; at {', '.join(f'{length:,}' for length in single_pass)} residues each series "
            "timed one fold without a warm-up."
            if single_pass
            else "."
        ),
        "Official is the pinned Biohub implementation on its PyTorch path, which is what it "
        "executes without flash-attn or its source-built Triton kernels.",
        "FastPLMs defaults changes no setting, and its output is bitwise identical to the "
        "previous FastPLMs revision; it adds the sampling pair-bias cache, a single layout of "
        "the right stream per chunked triangle update, and the early release of dead "
        "pair-sized tensors.",
        'FastPLMs optimized adds `set_atom_attention("windowed")` and `set_chunk_size(None)`.',
        f"Optimized ESMFold2 reaches {speed[shortest]:.2f}x the official throughput at "
        f"{shortest:,} residues and {speed[longest]:.2f}x at {longest:,}, for "
        f"{memory[longest]:.2f}x the official peak memory there; the bitwise-exact defaults "
        f"reach {min(exact_speed.values()):.2f}x to {max(exact_speed.values()):.2f}x.",
        "ESMFold2-600 and ESMFold2-300 share one 24-block trunk, so their curves nearly coincide.",
    ]
    for point in _fallback_points(by_label):
        official_failed = point["length"] not in {p["length"] for p in _measured(official)}
        exact = {p["length"]: p for p in _measured(defaults)}.get(point["length"])
        sentences.append(
            f"At {point['length']:,} residues unchunked pair updates exhaust this GPU, so the "
            "optimized ESMFold2 point there uses 512-row chunks and is drawn hollow: "
            f"{point['median_seconds']:.0f} s at a {point['peak_allocated_bytes'] / GIB:.0f} GiB "
            "peak"
            + (
                f", against {exact['median_seconds']:.0f} s for the bitwise-exact defaults"
                if exact
                else ""
            )
            + (
                ", while the official implementation runs out of memory."
                if official_failed
                else "."
            )
        )
    if out_of_memory:
        sentences.append(
            "These folds exhausted GPU memory, so their curves end earlier: "
            + "; ".join(out_of_memory)
            + "."
        )
    anchors = evidence.get("cross_run_anchors", [])
    if anchors:
        spread = max(
            abs(anchor["extension_seconds"] / anchor["sweep_seconds"] - 1) for anchor in anchors
        )
        sentences.append(
            "All series of a run ran in separate processes on one worker. The longest length "
            "comes from a second run, whose repeated measurements of shorter folds agree with "
            f"the first run within {spread:.1%}."
        )
    else:
        sentences.append("All series ran in separate processes on one worker in one run.")
    return " ".join(sentences) + "\n"


def plot() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    by_label = _series_by_label(evidence)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    # panel title, unit, value of one measured point
    panels: tuple[tuple[str, str, Callable[[Point], float]], ...] = (
        ("Throughput", "residues per second", lambda p: p["residues_per_second"]),
        ("Peak GPU memory", "GiB", lambda p: p["peak_allocated_bytes"] / GIB),
        (
            "Working memory per residue",
            "MiB per residue",
            lambda p: (p["peak_allocated_bytes"] - p["static_bytes"]) / MIB / p["length"],
        ),
    )
    for axis, (title, unit, value) in zip(axes, panels, strict=True):
        for label, legend, color, line_style, (marker, marker_size) in CURVES:
            series = by_label.get(label)
            if series is None:
                continue
            points = _measured(series)
            fallback = _fallback_points(by_label) if label == OPTIMIZED else []
            if fallback:
                axis.plot(
                    [points[-1]["length"], *(point["length"] for point in fallback)],
                    [value(points[-1]), *(value(point) for point in fallback)],
                    marker=marker,
                    markersize=marker_size + 1,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    markevery=slice(1, None),
                    linewidth=2.4,
                    color=color,
                )
            axis.plot(
                [point["length"] for point in points],
                [value(point) for point in points],
                marker=marker,
                markersize=marker_size,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.4 if line_style == "-" else 1.6,
                linestyle=line_style,
                color=color,
                label=legend,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        axis.set_xlabel("protein length (residues)")
        axis.set_ylabel(unit)
        axis.set_title(title)
        axis.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=300)
    plt.close(figure)
    CAPTION_PATH.write_text(caption(evidence), encoding="utf-8", newline="\n")
    return FIGURE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="copy fold-bench runs into evidence")
    record_parser.add_argument("run_directory", type=Path, help="a full fold-bench run")
    record_parser.add_argument(
        "extension_directories", type=Path, nargs="*", help="fold-bench-long runs that add lengths"
    )
    commands.add_parser("plot", help="draw the figure and write its caption from the evidence")
    args = parser.parse_args()
    if args.command == "record":
        record(args.run_directory, args.extension_directories)
        print(EVIDENCE_PATH.relative_to(ROOT).as_posix())
    else:
        print(plot().relative_to(ROOT).as_posix())
        print(CAPTION_PATH.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
