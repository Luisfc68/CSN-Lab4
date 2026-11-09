import csv
from pathlib import Path

DEPENDENCY_TREE_METRICS_DIR = Path("output/languages")
TOL = 1e-9  # Floating-point precision tolerance

if __name__ == "__main__":
    # Search for any metrics CSV file in the output directory
    for file_path in list(DEPENDENCY_TREE_METRICS_DIR.rglob("*.csv")):
        # Extract clean language name from filename
        lang_name = (
            file_path.stem.replace("_dependency_tree_metrics", "")
            .replace("_metrics", "")
            .capitalize()
        )
        print(f"Processing validity of dependency tree metrics of language {lang_name}")

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            n_validity_failures = 0

            for i, row in enumerate(reader, start=1):
                # 1. Cast strings from CSV to numeric types
                n = int(row["n"])
                k2_avg = float(row["k2_avg"])
                d_avg = float(row["d_avg"])
                q = int(row["q"])

                # 2. Check k2_avg theoretical bounds [Ferrer-i-Cancho, 2013]
                k2_lower = 4 - (6 / n)
                k2_upper = n - 1

                if k2_avg < (k2_lower - TOL) or k2_avg > (k2_upper + TOL):
                    n_validity_failures += 1
                    print(
                        f"    Row {i}: k2_avg ({k2_avg}) out of bounds [{k2_lower:.4f}, {k2_upper}]."
                    )

                # 3. Check d_avg theoretical bounds [Ferrer-i-Cancho, 2021]
                sum_d = (n - 1) * d_avg
                d_lower = 0.25 * (n * k2_avg + q)
                d_upper = 0.25 * (3 * ((n - 1) ** 2) + 1 - (n % 2))

                if sum_d < (d_lower - TOL) or sum_d > (d_upper + TOL):
                    n_validity_failures += 1
                    print(
                        f"    Row {i}: d_avg (sum_d = {sum_d}) out of bounds [{d_lower:.4f}, {d_upper:.4f}]."
                    )

            if n_validity_failures:
                print(
                    f"Language {lang_name} has {n_validity_failures} validity failures across the metrics gathered."
                )
            else:
                print(
                    f"Metrics gathered from language {lang_name} respect the theoretical bounds."
                )

            print()
