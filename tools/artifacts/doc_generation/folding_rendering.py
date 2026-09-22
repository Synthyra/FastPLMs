"""Render measured ESMFold2 folding costs and the settings they support."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from fastplms.registry import ModelSpec


FOLDING_COST_EVIDENCE = Path("docs/evidence/esmfold2/folding_cost.json")


FOLDING_COST_LENGTHS = (256, 1024, 2048)


def _folding_cost_cell(point: Mapping[str, object] | None) -> str:
    if point is None:
        return "not measured"
    if point.get("status") != "ok":
        return "out of memory"
    seconds = point["median_seconds"]
    assert isinstance(seconds, (int, float))
    return f"{seconds:.0f}" if seconds >= 10 else f"{seconds:.1f}"


def _folding_cost_table(spec: ModelSpec, evidence: Mapping[str, object]) -> str:
    """Rows of measured fold times for this checkpoint, or nothing when it was not measured."""

    series = evidence.get("series")
    assert isinstance(series, list)
    by_label: dict[str, dict[int, Mapping[str, object]]] = {}
    for entry in series:
        if entry.get("status") == "ok" and isinstance(entry.get("lengths"), list):
            by_label[entry["label"]] = {point["length"]: point for point in entry["lengths"]}
    defaults = by_label.get(f"fastplms-{spec.id}-dense")
    optimized = by_label.get(f"fastplms-{spec.id}-windowed-unchunked")
    if defaults is None or optimized is None:
        return ""
    # Where unchunked pair updates exhaust memory, the optimized fold uses 512-row chunks.
    fallback = by_label.get(f"fastplms-{spec.id}-windowed-chunk512", {})
    official = by_label.get("upstream-esmfold2") if spec.id == "esmfold2" else None
    header = "| Residues | FastPLMs defaults (s) | Optimized (s) |"
    divider = "| ---: | ---: | ---: |"
    if official is not None:
        header += " Official implementation (s) |"
        divider += " ---: |"
    rows = [header, divider]
    for length in FOLDING_COST_LENGTHS:
        best, note = optimized.get(length), ""
        if (best is None or best.get("status") != "ok") and fallback.get(length, {}).get(
            "status"
        ) == "ok":
            best, note = fallback[length], " with 512-row chunks"
        row = (
            f"| {length:,} | {_folding_cost_cell(defaults.get(length))} | "
            f"{_folding_cost_cell(best)}{note} |"
        )
        if official is not None:
            row += f" {_folding_cost_cell(official.get(length))} |"
        rows.append(row)
    environment = evidence["environment"]
    settings = evidence["series"][0]["settings"]
    assert isinstance(environment, Mapping) and isinstance(settings, Mapping)
    return (
        "\n".join(rows)
        + f"""

Measured on one {environment["gpu"]} with PyTorch {environment["torch"]}: one
fixed pseudo-random protein per length, {settings["num_loops"]} trunk loops,
{settings["num_sampling_steps"]} requested sampling steps under the official noise cap,
{settings["num_diffusion_samples"]} diffusion sample, BF16 autocast over FP32 folding
parameters, median of end-to-end folds. "Defaults" changes no setting.
"""
    )


def _esmfold2_folding_speed_section(spec: ModelSpec, root: Path | None) -> str:
    """Describe the opt-in folding speed settings, with this checkpoint's measured cost."""

    if spec.family.id != "esmfold2":
        return ""
    table = ""
    if root is not None and (root / FOLDING_COST_EVIDENCE).is_file():
        with (root / FOLDING_COST_EVIDENCE).open(encoding="utf-8") as handle:
            evidence = json.load(handle)
        if not isinstance(evidence, dict):
            raise ValueError(f"Folding evidence must be a JSON object: {FOLDING_COST_EVIDENCE}")
        table = _folding_cost_table(spec, evidence)
    return f"""\
## Folding speed settings

Two runtime settings trade memory or exactness for speed on long proteins. They
need no extra package and no compilation, and neither is stored in the
configuration.

```python
model.set_chunk_size(None)            # unchunked pair updates
model.set_atom_attention("windowed")  # the official flash-attn atom window, through PyTorch
```

`set_chunk_size(None)` removes the row chunking of the pair-update blocks, which
costs most of a long fold's time on a data-center GPU and saves little peak
memory; pass a chunk such as 512 when the unchunked fold does not fit.
`set_atom_attention("windowed")` restricts each atom to 64 real neighbors on
each side, as the official model does when flash-attn is installed. It needs
CUDA and changes numerical output, within sampling spread on the measured
panel. The
[ESMFold2 guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/esmfold2.md#measured-folding-cost)
records the conditions, the dense-versus-windowed comparison, and the figure.

{table}"""
