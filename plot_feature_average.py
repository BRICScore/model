"""
Plot the average value of a chosen feature for every data file in a directory.

Each file is JSON Lines: the first line holds metadata (person_id, activity, ...),
every following line holds one measurement window with numeric features.
The number of lines per file can vary.

Usage:
    python plot_feature_average.py              # uses FEATURE below
    python plot_feature_average.py bpm          # override the feature
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
DATA_DIR = Path(r"./data/features")  # directory to search
FILE_PATTERN = "*"          # e.g. "*.json" or "*.jsonl" to narrow the search
FEATURE = "bpm"             # default feature, can be overridden from the CLI
OUTPUT_FILE = None          # e.g. "bpm_average.png" to save instead of showing
# --------------------------------------------------------------------------


def read_file(path: Path, feature: str):
    """Return (metadata dict, list of feature values) for one file."""
    metadata = {}
    values = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skipping unparsable line {line_no} in {path.name}")
                continue
            if "person_id" in record:
                metadata = record
            elif feature in record and record[feature] is not None:
                values.append(float(record[feature]))
    return metadata, values


def main():
    feature = sys.argv[1] if len(sys.argv) > 1 else FEATURE

    files = sorted(p for p in DATA_DIR.glob(FILE_PATTERN) if p.is_file())
    if not files:
        sys.exit(f"No files matching '{FILE_PATTERN}' in {DATA_DIR}")

    labels, means, stds, counts = [], [], [], []
    for path in files:
        metadata, values = read_file(path, feature)
        if not values:
            print(f"{path.name}: no '{feature}' values, skipped")
            continue
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        label = path.stem
        details = [metadata.get(k) for k in ("activity", "condition") if metadata.get(k)]
        if details:
            label += "\n(" + ", ".join(details) + ")"

        labels.append(label)
        means.append(mean)
        stds.append(std)
        counts.append(len(values))
        print(f"{path.name}: mean {feature} = {mean:.6g}  (n={len(values)})")

    if not means:
        sys.exit(f"Feature '{feature}' was not found in any file.")

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(means) + 2), 5))
    bars = ax.bar(range(len(means)), means, yerr=stds, capsize=4,
                  color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{m:.3g}\nn={n}" for m, n in zip(means, counts)],
                 padding=3, fontsize=8)

    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, rotation=45 if len(means) > 6 else 0,
                       ha="right" if len(means) > 6 else "center", fontsize=8)
    ax.set_ylabel(f"average {feature}")
    ax.set_xlabel("file")
    ax.set_title(f"Average '{feature}' per file  (error bars = std dev)")
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.15)
    fig.tight_layout()

    if OUTPUT_FILE:
        fig.savefig(OUTPUT_FILE, dpi=150)
        print(f"Saved plot to {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
