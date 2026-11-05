"""Regression analysis of the mean dependency length of dependency trees."""

import argparse
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Output data config
DEPENDENCY_TREE_METRICS_DIR = Path("output/languages/").resolve()
OUTPUT_DIR = Path("output/plots/").resolve()

# Input data config
DEFAULT_LANGUAGE = "English"
METRICS_FILE_SUFFIX = "_dependency_tree_metrics.csv"

# Plot config
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 150
PLOT_STYLE = "whitegrid"
PLOT_CONTEXT = "notebook"
AVERAGE_COLOR = "#2ca02c"
RANDOM_ARRANGEMENT_COLOR = "#d62728"
LINEAR_LIMIT_PADDING = 0.05
LOG_LIMIT_PADDING = 1.05

# Regression config
MODEL_ENSEMBLE = {
    "model_0": lambda n: n / 3 + 1 / 3,
    "model_1": lambda n, b: (n / 2) ** b,
    "model_2": lambda n, a, b: a * n**b,
    "model_3": lambda n, a, c: a * math.exp(c * n),
    "model_4": lambda n, a: a * math.log(n),
    "model_1+": lambda n, b, d: (n / 2) ** b + d,
    "model_2+": lambda n, a, b, d: a * n**b + d,
    "model_3+": lambda n, a, c, d: a * math.exp(c * n) + d,
    "model_4+": lambda n, a, d: a * math.log(n) + d,
}


class ReferenceLine(NamedTuple):
    x: pd.Series
    y: pd.Series
    color: str
    label: str
    linestyle: str = "-"


def get_available_languages() -> list[str]:
    """List the languages for which dependency tree metrics are available."""
    if not DEPENDENCY_TREE_METRICS_DIR.is_dir():
        return []

    return sorted(
        {
            file_path.name.replace(METRICS_FILE_SUFFIX, "").capitalize()
            for file_path in DEPENDENCY_TREE_METRICS_DIR.glob(f"*{METRICS_FILE_SUFFIX}")
        }
    )


def load_language_metrics(language: str) -> pd.DataFrame:
    """Load the per-sentence dependency tree metrics of a given language."""
    metrics_path = (
        DEPENDENCY_TREE_METRICS_DIR / f"{language.lower()}{METRICS_FILE_SUFFIX}"
    )
    if not metrics_path.exists():
        available_languages = ", ".join(get_available_languages())
        raise FileNotFoundError(
            f"No dependency tree metrics found for language '{language}'. "
            f"Available languages: {available_languages}"
        )

    metrics = pd.read_csv(metrics_path)
    required_columns = {"n", "k2_avg", "d_avg", "q"}
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise ValueError(
            f"Metrics file '{metrics_path}' is missing columns: {sorted(missing_columns)}"
        )

    return metrics


def shared_y_limits(values: list[pd.Series], *, log_scale: bool) -> tuple[float, float]:
    """Compute the y-axis range covering all given values, so plots on one scale match."""
    combined = pd.concat(values).dropna()

    if log_scale:
        combined = combined[combined > 0]
        return combined.min() / LOG_LIMIT_PADDING, combined.max() * LOG_LIMIT_PADDING

    padding = (combined.max() - combined.min()) * LINEAR_LIMIT_PADDING
    return combined.min() - padding, combined.max() + padding


def plot_mean_length(
    data: pd.DataFrame,
    *,
    log_scale: bool = False,
    filename: str,
    ylabel: str = "mean dependency length",
    scatter_label: str = "sentences",
    scatter_size: float = 18,
    scatter_alpha: float = 0.35,
    lines: list[ReferenceLine] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> Path:
    """Plot mean dependency length against the number of vertices of a dependency tree.

    When ``log_scale`` is enabled both axes are plotted on a log scale, which is the
    equivalent of R's ``plot(x, y, log = "xy")``.

    ``lines`` adds reference lines (e.g. the random linear arrangement expectation),
    drawn with seaborn and shown in the legend. ``y_limits`` sets the y-axis range.
    """
    plot_data = data[["vertices", "mean_length"]].dropna().sort_values("vertices")

    if log_scale:
        plot_data = plot_data[
            (plot_data["vertices"] > 0) & (plot_data["mean_length"] > 0)
        ]

    sns.set_theme(style=PLOT_STYLE, context=PLOT_CONTEXT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    sns.scatterplot(
        data=plot_data,
        x="vertices",
        y="mean_length",
        ax=axis,
        s=scatter_size,
        alpha=scatter_alpha,
        edgecolor="none",
        label=scatter_label,
    )

    for line in lines or []:
        sns.lineplot(
            x=line.x,
            y=line.y,
            ax=axis,
            color=line.color,
            linewidth=2.5,
            linestyle=line.linestyle,
            label=line.label,
        )

    if log_scale:
        axis.set_xscale("log")
        axis.set_yscale("log")
    if y_limits is not None:
        axis.set_ylim(y_limits)

    axis.grid(True, which="major", axis="both", color="0.7", linewidth=0.9)
    if log_scale:
        axis.grid(True, which="minor", axis="both", color="0.88", linewidth=0.6)

    axis.set_xlabel("vertices")
    axis.set_ylabel(ylabel)
    axis.legend(loc="upper left", frameon=True, framealpha=0.95)
    figure.tight_layout()

    output_path = OUTPUT_DIR / filename
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return output_path


def plot_preliminary_results(language: str) -> list[Path]:
    """Generate the preliminary and aggregated mean length regression plots."""
    metrics = load_language_metrics(language)
    data = metrics.rename(columns={"n": "vertices", "d_avg": "mean_length"})

    aggregated_data = data.groupby("vertices")["mean_length"].mean().reset_index()
    random_arrangement = ReferenceLine(
        x=aggregated_data["vertices"],
        y=(aggregated_data["vertices"] + 1) / 3,
        color=RANDOM_ARRANGEMENT_COLOR,
        label="random linear arrangement (n + 1) / 3",
        linestyle="--",
    )

    linear_limits = shared_y_limits(
        [data["mean_length"], aggregated_data["mean_length"], random_arrangement.y],
        log_scale=False,
    )
    log_limits = shared_y_limits(
        [data["mean_length"], aggregated_data["mean_length"], random_arrangement.y],
        log_scale=True,
    )

    mean_length_path = plot_mean_length(
        data,
        log_scale=False,
        filename=f"{language.lower()}_mean_length.png",
        y_limits=linear_limits,
    )
    mean_length_log_path = plot_mean_length(
        data,
        log_scale=True,
        filename=f"{language.lower()}_mean_length_log.png",
        y_limits=log_limits,
    )

    aggregated_path = plot_mean_length(
        aggregated_data,
        log_scale=False,
        filename=f"{language.lower()}_mean_length_aggregated.png",
        ylabel="mean mean dependency length",
        scatter_label="mean per number of vertices",
        scatter_size=45,
        scatter_alpha=0.9,
        lines=[random_arrangement],
        y_limits=linear_limits,
    )
    aggregated_log_path = plot_mean_length(
        aggregated_data,
        log_scale=True,
        filename=f"{language.lower()}_mean_length_aggregated_log.png",
        ylabel="mean mean dependency length",
        scatter_label="mean per number of vertices",
        scatter_size=45,
        scatter_alpha=0.9,
        lines=[random_arrangement],
        y_limits=log_limits,
    )

    return [
        mean_length_path,
        mean_length_log_path,
        aggregated_path,
        aggregated_log_path,
    ]


def run_regression(language: str) -> None:
    metrics: pd.DataFrame = load_language_metrics(language)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate regression plots of the mean dependency length "
        "against the number of vertices of the dependency trees."
    )
    language_options = parser.add_mutually_exclusive_group()
    language_options.add_argument(
        "--language",
        "-l",
        choices=get_available_languages(),
        default=DEFAULT_LANGUAGE,
        help="Language to analyze (default: %(default)s).",
    )
    language_options.add_argument(
        "--all-languages",
        action="store_true",
        help="Generate plots and run regression for every available language.",
    )
    args = parser.parse_args(argv)

    languages = get_available_languages() if args.all_languages else [args.language]
    if args.all_languages and not languages:
        parser.error(f"No language metrics found in '{DEPENDENCY_TREE_METRICS_DIR}'.")

    for language in languages:
        output_paths = plot_preliminary_results(language)
        for path in output_paths:
            print(f"Plot saved to '{path}'.")

        run_regression(language)


if __name__ == "__main__":
    main()
