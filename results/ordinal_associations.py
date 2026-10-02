"""Trend tests and ordinal association estimates for the three comparisons in Figure 3."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import chi2


CSV_PATH = Path(__file__).with_name("survey.csv")
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 20261002

# Scores increase with anxiety, symptoms, psychological burden, or influence on future surgery.
SCORES = {
    "anxiety": {"A": 0, "B": 1, "C": 2, "D": 3},
    "reaction": {"A": 0, "B": 1, "C": 2},
    "burden": {"B": 0, "A": 1},
    "future_influence": {"A": 0, "B": 1, "C": 2},
}
COLUMNS = {"anxiety": 16, "reaction": 20, "burden": 23, "future_influence": 25}
COMPARISONS = (
    ("Anxiety vs postoperative reaction", "anxiety", "reaction"),
    ("Anxiety vs psychological burden", "anxiety", "burden"),
    ("Postoperative reaction vs future surgery influence", "reaction", "future_influence"),
)


def load_scores():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))[1:]
    if len(rows) != 66 or any(len(row) != 26 for row in rows):
        raise ValueError("Expected 66 complete questionnaire rows with 26 columns")
    return {
        name: np.array([SCORES[name][row[index].strip()[0]] for row in rows])
        for name, index in COLUMNS.items()
    }


def gamma(x, y):
    pair_products = (x[:, None] - x) * (y[:, None] - y)
    concordant = int((pair_products > 0).sum() // 2)
    discordant = int((pair_products < 0).sum() // 2)
    return (concordant - discordant) / (concordant + discordant)


def trend_test(x, y):
    correlation = np.corrcoef(x, y)[0, 1]
    chi_square = (len(x) - 1) * correlation**2
    return chi_square, chi2.sf(chi_square, df=1)


def bootstrap_ci(x, y, rng):
    n = len(x)
    estimates = [gamma(x[indices], y[indices])
                 for indices in (rng.integers(0, n, size=n) for _ in range(BOOTSTRAP_SAMPLES))]
    return np.percentile(estimates, [2.5, 97.5])


def main():
    scores = load_scores()
    rng = np.random.default_rng(RANDOM_SEED)
    for label, x_name, y_name in COMPARISONS:
        x, y = scores[x_name], scores[y_name]
        chi_square, p_value = trend_test(x, y)
        ci = bootstrap_ci(x, y, rng)
        print(f"{label}: n={len(x)}, linear-by-linear chi-square={chi_square:.3f}, "
              f"P={p_value:.4g}, gamma={gamma(x, y):.3f}, "
              f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")


if __name__ == "__main__":
    main()
