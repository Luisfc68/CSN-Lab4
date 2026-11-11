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
    # file name: (title, source CSV, column, decimals, criterion that picks the best model)
    "s": ("Residual standard error $s$", "aic", "s", 4, "aic"),
    "aic": ("AIC", "aic", "aic", 1, "aic"),
    "delta_aic": ("AIC differences $\\Delta$", "aic", "delta", 1, "aic"),
    "weight_aic": ("Akaike weights", "aic", "weight", 3, "aic"),
    "bic": ("BIC", "bic", "bic", 1, "bic"),
    "delta_bic": ("BIC differences $\\Delta$", "bic", "delta", 1, "bic"),
    "weight_bic": ("BIC weights", "bic", "weight", 3, "bic"),
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
    if metric == "weight" and value < MIN_WEIGHT:
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


def write_metric_table(scheme: str, name: str) -> None:
    title, source, column, decimals, criterion = METRICS[name]
    fits = pd.read_csv(REGRESSION_DIR / scheme / f"{source}.csv")
    table = fits.pivot(index="language", columns="model", values=column)
    scores = fits.pivot(index="language", columns="model", values=criterion)

    lines = table_header(
        f"{title} of each model for the {SCHEMES[scheme]} fit. "
        "The best model of each language is in bold.",
        f"tab:{name}_{scheme}",
        "l" + "r" * len(MODELS),
    )
    lines.append("    Language & " + " & ".join(MODELS) + " \\\\")
    lines.append("    \\midrule")
    for language, row in table.iterrows():
        best = scores.loc[language].idxmin()
        cells = [
            metric_cell(column, decimals, row[model], model == best) for model in MODELS
        ]
        lines.append(f"    {language} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    (OUTPUT_DIR / f"{name}_{scheme}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_parameter_table(
    scheme: str, source: str, models: tuple[str, ...], title: str, name: str
) -> None:
    values = pd.read_csv(REGRESSION_DIR / scheme / f"{source}.csv")
    table = pd.DataFrame(
        {
            f"{model}_{p}": values[values["model"] == model].set_index("language")[p]
            for model in models
            for p in PARAMETERS[model]
        }
    )
    columns = list(table.columns)

    lines = table_header(
        f"{title} for the {SCHEMES[scheme]} fit.",
        f"tab:{name}_{scheme}",
        "l" + "r" * len(columns),
    )
    spans = " & ".join(
        f"\\multicolumn{{{len(PARAMETERS[model])}}}{{c}}{{{model}}}" for model in models
    )
    symbols = " & ".join(f"${p}$" for model in models for p in PARAMETERS[model])
    lines += [f"    & {spans} \\\\", "    \\midrule", f"    Language & {symbols} \\\\"]
    lines.append("    \\midrule")
    for language, row in table.iterrows():
        cells = [f"${significant(row[column])}$" for column in columns]
        lines.append(f"    {language} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    (OUTPUT_DIR / f"{name}_{scheme}.tex").write_text(
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
        for name in METRICS:
            write_metric_table(scheme, name)
        write_parameter_table(
            scheme,
            "params",
            tuple(PARAMETERS),
            "Parameters giving the best fit of each model",
            "params",
        )
        write_parameter_table(
            scheme,
            "initial_values",
            ("1", "2", "3", "4", "5"),
            "Initial parameters of each nested model",
            "initial_nested",
        )
        write_parameter_table(
            scheme,
            "initial_values",
            ("1+", "2+", "3+", "4+", "5+"),
            "Initial parameters of each model with offset",
            "initial_offset",
        )
        write_figure(
            f"best_fit_{scheme}",
            f"Mean dependency length $\\langle d \\rangle$ versus sentence length $n$ with "
            f"the two best models by AIC (green) of the {label} fit, "
            "the mean per length (orange) and the null model (dashed red), in log-log scale.",
            {
                language: f"output/plots/best_fit/{scheme}/{language}_best_fit_aic_log.png"
                for language in LANGUAGES
            },
        )

    write_figure(
        "residual_variance",
        "Variance of the residuals of the best model versus sentence length $n$, "
        "for the identity and log-log fits, in log-log scale. Lengths with fewer than "
        "5 sentences are not shown. A flat curve indicates homoscedasticity.",
        {
            language: f"output/plots/residual_variance/{language}_residual_variance.png"
            for language in LANGUAGES
        },
    )

    p_values = pd.read_csv(P_VALUES_PATH)
    for column in ("p_value_linear", "p_value_log"):
        p_values[column] = p_values[column].map(format_p_value)
    p_values.to_csv(OUTPUT_DIR / "homoscedasticity_p_values.csv", index=False)

    print(f"LaTeX files written to '{OUTPUT_DIR}'.")
