"""Homoscedasticity check of the mean dependency length across the number of vertices."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fligner

# Output data config
DEPENDENCY_TREE_METRICS_DIR = Path("output/languages/").resolve()
OUTPUT_DIR = Path("output/").resolve()
OUTPUT_FILENAME = "homoscedasticity_p_values.csv"

# Input data config
METRICS_FILE_SUFFIX = "_dependency_tree_metrics.csv"

# Test config
MIN_GROUP_SIZE = 10


def get_metrics_paths() -> list[Path]:
    """List the dependency tree metrics files, one per language."""
    return sorted(DEPENDENCY_TREE_METRICS_DIR.glob(f"*{METRICS_FILE_SUFFIX}"))


def fligner_killeen_p_value(metrics: pd.DataFrame, *, log_scale: bool) -> float:
    """Fligner-Killeen p-value of the variance of the mean length across vertex counts.

    Sentences are grouped by their number of vertices ``n``, and only groups with at
    least ``MIN_GROUP_SIZE`` sentences are kept. When ``log_scale`` is enabled the test
    is run on ``log(d_avg)``, so trees with a zero mean length are excluded before the
    group sizes are counted.
    """
    data = metrics[["n", "d_avg"]].dropna()

    if log_scale:
        data = data[data["d_avg"] > 0].assign(d_avg=lambda df: np.log(df["d_avg"]))

    groups = [
        group["d_avg"].to_numpy()
        for _, group in data.groupby("n")
        if len(group) >= MIN_GROUP_SIZE
    ]
    if len(groups) < 2:
        return float("nan")

    return float(fligner(*groups).pvalue)


def main() -> None:
    rows = []
    for metrics_path in get_metrics_paths():
        language = metrics_path.name.replace(METRICS_FILE_SUFFIX, "").capitalize()
        metrics = pd.read_csv(metrics_path)

        p_value = fligner_killeen_p_value(metrics, log_scale=False)
        p_value_log = fligner_killeen_p_value(metrics, log_scale=True)
        rows.append(
            {
                "language": language,
                "p_value_linear": p_value,
                "p_value_log": p_value_log,
            }
        )
        print(
            f"{language}: p-value linear = {p_value:.4g}, "
            f"p-value log = {p_value_log:.4g}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"P-values saved to '{output_path}'.")


if __name__ == "__main__":
    main()
