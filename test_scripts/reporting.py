import csv
import json
import pathlib
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")


def write_json(path: pathlib.Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: pathlib.Path, rows: List[Dict[str, object]]) -> None:
    if len(rows) == 0:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("")
        return

    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: pathlib.Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def plot_bar(path: pathlib.Path, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

