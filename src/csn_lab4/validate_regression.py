from pathlib import Path

import numpy as np
import pandas as pd
from regression import (
    MODELS,
    SCHEMES,
    best_model,
    build_fit_data,
    drop_small_groups,
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
ARTIFICIAL_NOISE_SD = 0.19
REPLICATES = 25
AIC_TIE = 2  # models within this AIC difference are not really distinguishable
# Parameters of the log fit of each model to Basque, after removing punctuation
TRUE_PARAMETERS = {
    "1": (0.4358,),
    "2": (0.8054, 0.4011),
    "3": (1.3594, 0.0352),
    "4": (0.9087,),
    "5": (0.7431, 0.4696, -0.0068),
    "1+": (0.4175, 0.068),
    "2+": (2.1613, 0.221, -1.5473),
    "3+": (-2.3814, -0.0859, 3.0651),
    "4+": (0.7429, 0.3584),
    "5+": (3.9758, 0.131, 0.0013, -3.4016),
}


def random_start(model, data, rng) -> list[float]:
    # On the log scheme a start with f(n) <= 0 has a flat error and the search cannot move
    while True:
        start = [rng.uniform(*RESTART_RANGES[p]) for p in model.parameters]
        if not data.log_scale or np.all(model.function(data.n, *start) > 0):
            return start


def check_optimality() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for language in get_available_languages():
        metrics = drop_small_groups(load_language_metrics(language))
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
                    solution = least_squares(
                        residuals,
                        random_start(model, data, rng),
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
    n = drop_small_groups(load_language_metrics(ARTIFICIAL_LANGUAGE))["n"].to_numpy(
        dtype=float
    )
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
