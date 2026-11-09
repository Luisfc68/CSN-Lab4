from pathlib import Path

import numpy as np
import pandas as pd
from regression import (
    MODELS,
    SCHEMES,
    best_model,
    build_fit_data,
    fit_all_models,
    model_selection_row,
    residuals,
)
from preliminary_plots import get_available_languages, load_language_metrics
from scipy.optimize import least_squares

OUTPUT_DIR = Path("output/regression/validation/").resolve()

# Optimality check config
RESTARTS = 40
RESTART_RANGES = {"a": (-50, 50), "b": (-1, 2), "c": (-0.5, 0.2), "d": (-50, 50)}
RELATIVE_IMPROVEMENT = 1e-6

# Artificial data config: Basque sentence lengths, noise from the fitted log residuals
ARTIFICIAL_LANGUAGE = "Basque"
ARTIFICIAL_NOISE_SD = 0.18
REPLICATES = 25
AIC_TIE = 2  # models within this AIC difference are not really distinguishable
TRUE_PARAMETERS = {
    "1": (0.5,),
    "2": (0.75, 0.49),
    "3": (1.5, 0.036),
    "4": (1.02,),
    "1+": (0.5, 0.02),
    "2+": (1.0, 0.43, -0.37),
    "3+": (-4.0, -0.047, 4.8),
    "4+": (1.12, -0.24),
}


def check_optimality() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for language in get_available_languages():
        metrics = load_language_metrics(language)
        for scheme in SCHEMES:
            fits = fit_all_models(metrics, scheme)
            data, _ = build_fit_data(metrics, scheme)

            for name, model in MODELS.items():
                if not model.parameters:
                    continue

                reported = np.array(list(fits[name].params.values()))
                reported_rss = float(np.sum(residuals(reported, model, data) ** 2))
                best_rss = reported_rss
                for _ in range(RESTARTS):
                    start = [rng.uniform(*RESTART_RANGES[p]) for p in model.parameters]
                    solution = least_squares(
                        residuals,
                        start,
                        args=(model, data),
                        x_scale="jac",
                        max_nfev=3000,
                    )
                    best_rss = min(best_rss, float(np.sum(solution.fun**2)))

                rows.append(
                    {
                        "language": language,
                        "scheme": scheme,
                        "model": name,
                        "reported_rss": reported_rss,
                        "best_restart_rss": best_rss,
                        "local_optimum": bool(
                            best_rss < reported_rss * (1 - RELATIVE_IMPROVEMENT)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def check_artificial_data() -> pd.DataFrame:
    n = load_language_metrics(ARTIFICIAL_LANGUAGE)["n"].to_numpy(dtype=float)
    rows = []
    for name, truth in TRUE_PARAMETERS.items():
        selected = 0
        within_tie = 0
        errors = []
        for replicate in range(REPLICATES):
            noise = np.random.default_rng(1000 + replicate).normal(
                0, ARTIFICIAL_NOISE_SD, len(n)
            )
            y = MODELS[name].function(n, *truth) * np.exp(noise)
            fits = fit_all_models(
                pd.DataFrame({"n": n.astype(int), "d_avg": y}), "log"
            )
            selected += best_model(fits) == name
            within_tie += model_selection_row(fits)["delta"][name] < AIC_TIE
            estimated = np.array(list(fits[name].params.values()))
            errors.append(np.abs(estimated - truth) / np.abs(truth))

        rows.append(
            {
                "model": name,
                "replicates": REPLICATES,
                "times_selected": selected,
                "times_within_aic_tie": within_tie,
                "median_relative_error": dict(
                    zip(
                        MODELS[name].parameters,
                        np.round(np.median(errors, axis=0), 3).tolist(),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    optimality = check_optimality()
    artificial = check_artificial_data()

    optimality.to_csv(OUTPUT_DIR / "optimality.csv", index=False)
    artificial.to_csv(OUTPUT_DIR / "artificial_data.csv", index=False)

    local_optima = optimality[optimality["local_optimum"]]
    print(f"Fits checked: {len(optimality)}. Local optima: {len(local_optima)}.")
    if len(local_optima):
        print(local_optima.to_string(index=False))
    print(artificial.to_string(index=False))
