from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from preliminary_plots import (
    PLOT_CONTEXT,
    PLOT_STYLE,
    RANDOM_ARRANGEMENT_COLOR,
    SPREAD_COLOR,
    get_available_languages,
    load_language_metrics,
)
from scipy.optimize import least_squares
from scipy.stats import spearmanr

# Output data config
OUTPUT_DIR = Path("output/regression/").resolve()
PLOTS_DIR = Path("output/plots/best_fit/").resolve()
RESIDUAL_PLOTS_DIR = Path("output/plots/residual_variance/").resolve()

# Optimizer config
MAX_FUNCTION_EVALUATIONS = 5000
MIN_PREDICTION = 1e-12  # keeps log(f(n)) finite when a candidate f(n) is not positive
INVALID_RESIDUAL = 1e6  # replaces residuals that overflow during the search

# Diagnostics and plot config
SCHEMES = ("identity", "aggregate", "log")
MIN_GROUP_SIZE = 10  # sentences needed to estimate a residual variance for a length
BEST_FIT_COLOR = "#2ca02c"
IDENTITY_COLOR = "#1f77b4"
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 150


class Model(NamedTuple):
    function: Callable
    parameters: tuple[str, ...]


class FitData(NamedTuple):
    n: np.ndarray
    response: np.ndarray
    log_scale: bool


class FitResult(NamedTuple):
    params: dict[str, float]
    start: dict[str, float]
    rss: float
    s: float
    aic: float
    converged: bool


MODELS = {
    "0": Model(lambda n: (n + 1) / 3, ()),
    "1": Model(lambda n, b: (n / 2) ** b, ("b",)),
    "2": Model(lambda n, a, b: a * n**b, ("a", "b")),
    "3": Model(lambda n, a, c: a * np.exp(c * n), ("a", "c")),
    "4": Model(lambda n, a: a * np.log(n), ("a",)),
    "1+": Model(lambda n, b, d: (n / 2) ** b + d, ("b", "d")),
    "2+": Model(lambda n, a, b, d: a * n**b + d, ("a", "b", "d")),
    "3+": Model(lambda n, a, c, d: a * np.exp(c * n) + d, ("a", "c", "d")),
    "4+": Model(lambda n, a, d: a * np.log(n) + d, ("a", "d")),
}


def residuals(params: np.ndarray, model: Model, data: FitData) -> np.ndarray:
    predicted = model.function(data.n, *params)
    if data.log_scale:
        predicted = np.log(np.clip(predicted, MIN_PREDICTION, None))
    return np.nan_to_num(
        data.response - predicted,
        nan=INVALID_RESIDUAL,
        posinf=INVALID_RESIDUAL,
        neginf=-INVALID_RESIDUAL,
    )


def saturating_start(x: np.ndarray, y: np.ndarray) -> list[float]:
    d = 1.1 * float(np.quantile(y, 0.99))
    below_asymptote = y < d
    slope, intercept = np.polyfit(x[below_asymptote], np.log(d - y[below_asymptote]), 1)
    return [-float(np.exp(intercept)), float(slope), d]


def starting_points(
    name: str, n: np.ndarray, y: np.ndarray, fitted: dict[str, FitResult]
) -> list[list[float]]:
    log_n, log_y = np.log(n), np.log(y)

    if name == "1":
        log_half_n = np.log(n / 2)
        return [[float((log_half_n * log_y).sum() / (log_half_n**2).sum())]]
    if name == "2":
        b, log_a = np.polyfit(log_n, log_y, 1)
        return [[float(np.exp(log_a)), float(b)]]
    if name == "3":
        c, log_a = np.polyfit(n, log_y, 1)
        return [[float(np.exp(log_a)), float(c)]]
    if name == "4":
        return [[float(np.exp(np.mean(log_y - np.log(log_n))))]]
    if name == "1+":
        b = fitted["1"].params["b"]
        return [[b, 0.0], [b / 2, 1.0]]
    if name == "2+":
        a, b = fitted["2"].params["a"], fitted["2"].params["b"]
        return [[a, b, 0.0], [2 * a, b / 2, -a], saturating_start(log_n, y)]
    if name == "3+":
        a, c = fitted["3"].params["a"], fitted["3"].params["c"]
        return [[a, c, 0.0], saturating_start(n, y)]
    if name == "4+":
        a, d = np.polyfit(log_n, y, 1)
        return [[float(a), float(d)], [fitted["4"].params["a"], 0.0]]
    raise ValueError(f"Unknown model '{name}'")


def information_criteria(rss: float, sample_size: int, p: int) -> tuple[float, float]:
    s = float(np.sqrt(rss / (sample_size - p)))
    aic = float(
        sample_size * np.log(2 * np.pi)
        + sample_size * np.log(rss / sample_size)
        + sample_size
        + 2 * (p + 1)
    )
    return s, aic


def fit_model(
    name: str, data: FitData, y: np.ndarray, fitted: dict[str, FitResult]
) -> FitResult:
    model = MODELS[name]
    sample_size = len(data.n)

    if not model.parameters:
        rss = float(np.sum(residuals(np.array([]), model, data) ** 2))
        s, aic = information_criteria(rss, sample_size, 0)
        return FitResult({}, {}, rss, s, aic, True)

    best = None
    best_start = None
    for start in starting_points(name, data.n, y, fitted):
        solution = least_squares(
            residuals,
            np.array(start, dtype=float),
            args=(model, data),
            method="trf",
            x_scale="jac",
            max_nfev=MAX_FUNCTION_EVALUATIONS,
        )
        if best is None or solution.cost < best.cost:
            best, best_start = solution, start

    rss = float(np.sum(best.fun**2))
    s, aic = information_criteria(rss, sample_size, len(model.parameters))
    return FitResult(
        dict(zip(model.parameters, map(float, best.x))),
        dict(zip(model.parameters, best_start)),
        rss,
        s,
        aic,
        bool(best.status > 0),
    )


def build_fit_data(metrics: pd.DataFrame, scheme: str) -> tuple[FitData, np.ndarray]:
    if scheme == "aggregate":
        metrics = metrics.groupby("n")["d_avg"].mean().reset_index()

    n = metrics["n"].to_numpy(dtype=float)
    y = metrics["d_avg"].to_numpy(dtype=float)
    if scheme == "log":
        return FitData(n, np.log(y), log_scale=True), y
    return FitData(n, y, log_scale=False), y


def fit_all_models(metrics: pd.DataFrame, scheme: str) -> dict[str, FitResult]:
    data, y = build_fit_data(metrics, scheme)
    fitted: dict[str, FitResult] = {}
    for name in MODELS:
        fitted[name] = fit_model(name, data, y, fitted)
    return fitted


def model_selection_row(fits: dict[str, FitResult]) -> dict[str, dict[str, float]]:
    """AIC differences and Akaike weights of one language"""
    aic = np.array([fit.aic for fit in fits.values()])
    delta = aic - aic.min()
    weights = np.exp(-delta / 2)
    weights /= weights.sum()
    return {
        "delta": dict(zip(fits, map(float, delta))),
        "weight": dict(zip(fits, map(float, weights))),
    }


def best_model(fits: dict[str, FitResult]) -> str:
    return min(fits, key=lambda name: fits[name].aic)


def best_residuals(
    scheme: str, metrics: pd.DataFrame, fits: dict[str, FitResult]
) -> tuple[np.ndarray, np.ndarray]:
    """Sentence lengths and residuals of the best model, on the scale of its scheme"""
    name = best_model(fits)
    data, _ = build_fit_data(metrics, scheme)
    params = np.array(list(fits[name].params.values()))
    return data.n, residuals(params, MODELS[name], data)


def residual_diagnostics(
    language: str, scheme: str, metrics: pd.DataFrame, fits: dict[str, FitResult]
) -> dict:
    n, errors = best_residuals(scheme, metrics, fits)
    correlation = spearmanr(n, np.abs(errors))
    return {
        "language": language,
        "scheme": scheme,
        "best_model": best_model(fits),
        "spearman_rho": float(correlation.statistic),
        "p_value": float(correlation.pvalue),
    }


def plot_residual_variance(
    language: str, metrics: pd.DataFrame, fits_by_scheme: dict[str, dict[str, FitResult]]
) -> Path:
    sns.set_theme(style=PLOT_STYLE, context=PLOT_CONTEXT)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    for scheme, color in (("identity", IDENTITY_COLOR), ("log", BEST_FIT_COLOR)):
        n, errors = best_residuals(scheme, metrics, fits_by_scheme[scheme])
        groups = pd.Series(errors).groupby(n)
        variance = groups.var()[groups.count() >= MIN_GROUP_SIZE]
        variance = variance[variance > 0]
        name = best_model(fits_by_scheme[scheme])
        axis.plot(
            variance.index,
            variance.values,
            color=color,
            linewidth=2,
            label=f"{scheme} scale: model {name}",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("vertices")
    axis.set_ylabel("variance of the residuals")
    axis.legend(loc="upper left", frameon=True, framealpha=0.95)
    figure.tight_layout()

    output_dir = RESIDUAL_PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{language.lower()}_residual_variance.png"
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return output_path


def plot_best_fit(
    language: str, scheme: str, metrics: pd.DataFrame, fits: dict[str, FitResult]
) -> Path:
    """Plot the data with the best curve of a scheme and the null model, log-log"""
    n_values = np.arange(metrics["n"].min(), metrics["n"].max() + 1, dtype=float)
    means = metrics.groupby("n")["d_avg"].mean()

    sns.set_theme(style=PLOT_STYLE, context=PLOT_CONTEXT)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.scatter(
        metrics["n"], metrics["d_avg"], s=8, alpha=0.2, linewidths=0, label="sentences"
    )
    axis.plot(
        means.index, means.values, color=SPREAD_COLOR, linewidth=1.5, label="mean per n"
    )

    name = best_model(fits)
    axis.plot(
        n_values,
        MODELS[name].function(n_values, **fits[name].params),
        color=BEST_FIT_COLOR,
        linewidth=2.5,
        label=f"best {scheme} fit: model {name}",
    )

    axis.plot(
        n_values,
        MODELS["0"].function(n_values),
        color=RANDOM_ARRANGEMENT_COLOR,
        linewidth=2,
        linestyle="--",
        label="model 0: (n + 1) / 3",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    # Frame the data, not the null model, so the fit can be judged against the data
    axis.set_ylim(metrics["d_avg"].min() / 1.1, metrics["d_avg"].max() * 1.1)
    axis.set_xlabel("vertices")
    axis.set_ylabel("mean dependency length")
    axis.legend(loc="upper left", frameon=True, framealpha=0.95)
    figure.tight_layout()

    output_dir = PLOTS_DIR / scheme
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{language.lower()}_best_fit.png"
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return output_path


def write_tables(scheme: str, results: dict[str, dict[str, FitResult]]) -> None:
    """Write the s, AIC, delta, weight, parameter and starting value tables of a scheme."""
    output_dir = OUTPUT_DIR / scheme
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, list[dict]] = {key: [] for key in ("s", "aic", "delta", "weights")}
    params, starts, convergence = [], [], []

    for language, fits in results.items():
        selection = model_selection_row(fits)
        tables["s"].append({"language": language} | {m: f.s for m, f in fits.items()})
        tables["aic"].append({"language": language} | {m: f.aic for m, f in fits.items()})
        tables["delta"].append({"language": language} | selection["delta"])
        tables["weights"].append({"language": language} | selection["weight"])
        params.append(
            {"language": language}
            | {f"{m}_{p}": v for m, f in fits.items() for p, v in f.params.items()}
        )
        starts.append(
            {"language": language}
            | {f"{m}_{p}": v for m, f in fits.items() for p, v in f.start.items()}
        )
        convergence.append(
            {"language": language} | {m: f.converged for m, f in fits.items()}
        )

    for key, rows in tables.items():
        pd.DataFrame(rows).to_csv(output_dir / f"{key}.csv", index=False)
    pd.DataFrame(params).to_csv(output_dir / "params.csv", index=False)
    pd.DataFrame(starts).to_csv(output_dir / "initial_values.csv", index=False)
    pd.DataFrame(convergence).to_csv(output_dir / "converged.csv", index=False)


if __name__ == "__main__":
    results: dict[str, dict[str, dict[str, FitResult]]] = {s: {} for s in SCHEMES}
    diagnostics = []

    for language in get_available_languages():
        metrics = load_language_metrics(language)
        for scheme in SCHEMES:
            fits = fit_all_models(metrics, scheme)
            results[scheme][language] = fits
            diagnostics.append(residual_diagnostics(language, scheme, metrics, fits))
            plot_best_fit(language, scheme, metrics, fits)
        plot_residual_variance(
            language, metrics, {s: results[s][language] for s in SCHEMES}
        )
        winners = ", ".join(
            f"{scheme} = {best_model(results[scheme][language])}" for scheme in SCHEMES
        )
        print(f"{language}: best models: {winners}.")

    for scheme in SCHEMES:
        write_tables(scheme, results[scheme])
    pd.DataFrame(diagnostics).to_csv(OUTPUT_DIR / "residual_diagnostics.csv", index=False)
    print(f"Tables saved to '{OUTPUT_DIR}'.")
