from pathlib import Path

import pandas as pd

REGRESSION_DIR = Path("output/regression/").resolve()
P_VALUES_PATH = Path("output/homoscedasticity_p_values.csv").resolve()
OUTPUT_DIR = Path("output/latex/").resolve()

MODELS = ("0", "1", "2", "3", "4", "5", "1+", "2+", "3+", "4+", "5+")
PARAMETERS = {
    "1": ("b",),
    "2": ("a", "b"),
    "3": ("a", "c"),
    "4": ("a",),
    "5": ("a", "b", "c"),
    "1+": ("b", "d"),
    "2+": ("a", "b", "d"),
    "3+": ("a", "c", "d"),
    "4+": ("a", "d"),
    "5+": ("a", "b", "c", "d"),
}
SCHEMES = {"log": "log-log", "identity": "identity", "aggregate": "aggregated"}
METRICS = {
    "s": ("Residual standard error $s$", 4),
    "aic": ("AIC", 1),
    "delta": ("AIC differences $\\Delta$", 1),
    "weights": ("Akaike weights", 3),
}
LANGUAGES = (
    "arabic",
    "basque",
    "catalan",
    "chinese",
    "czech",
    "english",
    "greek",
    "hungarian",
    "italian",
    "turkish",
)
FIGURE_WIDTH = 0.40
UNDERFLOW_LABEL = "$<10^{-300}$"
MIN_WEIGHT = 0.001


def fixed(value: float, decimals: int) -> str:
    return f"{value + 0.0:.{decimals}f}"


def significant(value: float) -> str:
    text = f"{value:.3g}"
    if "e" not in text:
        return text
    if abs(value) >= 1:
        return f"{value:.0f}"
    mantissa, exponent = text.split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def metric_cell(metric: str, decimals: int, value: float, is_best: bool) -> str:
    if metric == "weights" and value < MIN_WEIGHT:
        return f"$<{MIN_WEIGHT}$"
    text = fixed(value, decimals)
    return f"$\\mathbf{{{text}}}$" if is_best else f"${text}$"


def table_header(caption: str, label: str, columns: str) -> list[str]:
    # The tables have many columns, so they are scaled to the width of the page
    return [
        "\\begin{table}[H]",
        "  \\centering",
        "  \\small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{columns}}}",
        "    \\toprule",
    ]


def write_metric_table(scheme: str, metric: str) -> None:
    title, decimals = METRICS[metric]
    table = pd.read_csv(REGRESSION_DIR / scheme / f"{metric}.csv", index_col="language")
    delta = pd.read_csv(REGRESSION_DIR / scheme / "delta.csv", index_col="language")

    lines = table_header(
        f"{title} of each model for the {SCHEMES[scheme]} fit. "
        "The best model of each language is in bold.",
        f"tab:{metric}_{scheme}",
        "l" + "r" * len(MODELS),
    )
    lines.append("    Language & " + " & ".join(MODELS) + " \\\\")
    lines.append("    \\midrule")
    for language, row in table.iterrows():
        best = delta.loc[language].idxmin()
        cells = [
            metric_cell(metric, decimals, row[model], model == best) for model in MODELS
        ]
        lines.append(f"    {language} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    (OUTPUT_DIR / f"{metric}_{scheme}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_parameter_table(scheme: str) -> None:
    table = pd.read_csv(REGRESSION_DIR / scheme / "params.csv", index_col="language")
    columns = [f"{model}_{p}" for model, ps in PARAMETERS.items() for p in ps]

    lines = table_header(
        f"Parameters giving the best fit of each model for the {SCHEMES[scheme]} fit. "
        "Model 0 has no parameters.",
        f"tab:params_{scheme}",
        "l" + "r" * len(columns),
    )
    spans = " & ".join(
        f"\\multicolumn{{{len(ps)}}}{{c}}{{{model}}}" for model, ps in PARAMETERS.items()
    )
    symbols = " & ".join(f"${p}$" for ps in PARAMETERS.values() for p in ps)
    lines += [f"    & {spans} \\\\", "    \\midrule", f"    Language & {symbols} \\\\"]
    lines.append("    \\midrule")
    for language, row in table.iterrows():
        cells = [f"${significant(row[column])}$" for column in columns]
        lines.append(f"    {language} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    (OUTPUT_DIR / f"params_{scheme}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_figure(name: str, caption: str, paths: dict[str, str]) -> None:
    lines = ["\\begin{figure}[H]", "  \\centering"]
    for position, (language, path) in enumerate(paths.items(), start=1):
        lines += [
            f"  \\begin{{subfigure}}[b]{{{FIGURE_WIDTH}\\textwidth}}",
            "    \\centering",
            f"    \\includegraphics[width=\\textwidth]{{{path}}}",
            f"    \\caption{{{language.capitalize()}}}",
            "  \\end{subfigure}",
        ]
        lines.append("  \\hfill" if position % 2 else "")
    lines += [f"  \\caption{{{caption}}}", f"  \\label{{fig:{name}}}", "\\end{figure}"]
    (OUTPUT_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_p_value(value: float) -> str:
    if value == 0:
        return UNDERFLOW_LABEL
    if value >= MIN_WEIGHT:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for scheme, label in SCHEMES.items():
        for metric in METRICS:
            write_metric_table(scheme, metric)
        write_parameter_table(scheme)
        write_figure(
            f"best_fit_{scheme}",
            f"Mean dependency length $\\langle d \\rangle$ versus sentence length $n$ with "
            f"the best model of the {label} fit (green), the mean per length (orange) and "
            "the null model (dashed red), in log-log scale.",
            {
                language: f"figs/best_fit/{scheme}/{language}_best_fit.png"
                for language in LANGUAGES
            },
        )

    write_figure(
        "residual_variance",
        "Variance of the residuals of the best model versus sentence length $n$, "
        "for the identity and log-log fits, in log-log scale. Lengths with fewer than "
        "10 sentences are not shown. A flat curve indicates homoscedasticity.",
        {
            language: f"figs/residual_variance/{language}_residual_variance.png"
            for language in LANGUAGES
        },
    )

    p_values = pd.read_csv(P_VALUES_PATH)
    for column in ("p_value_linear", "p_value_log"):
        p_values[column] = p_values[column].map(format_p_value)
    p_values.to_csv(OUTPUT_DIR / "homoscedasticity_p_values.csv", index=False)

    print(f"LaTeX files written to '{OUTPUT_DIR}'.")
