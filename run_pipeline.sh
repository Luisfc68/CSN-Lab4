#!/usr/bin/env bash
# Runs the full analysis pipeline for all languages, from the repository root.
set -euo pipefail

cd "$(dirname "$0")"
run() { uv run python "src/csn_lab4/$1" "${@:2}"; }

run preprocess.py
run check_validity.py
run preliminary_plots.py --all-languages
run homoscedasticity_check.py
run regression.py
run validate_regression.py
run latex_tables.py
