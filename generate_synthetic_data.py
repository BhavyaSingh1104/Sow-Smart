"""
generate_synthetic_data.py

Generates a synthetic historical rainfall + crop outcome dataset shaped
exactly like what you'll get from data.gov.in / IMD, so the rest of the
pipeline (features, model, recommendation engine) can be built and tested
right now.

WHEN YOU HAVE REAL DATA:
Replace this file's output (data/rainfall_history.csv) with the real
data.gov.in "Rainfall" catalog CSV, keeping the same column names:
    district, year, week_of_year, rainfall_mm, normal_rainfall_mm
No other file needs to change.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

DISTRICTS = [
    "Lucknow", "Kanpur", "Indore", "Bhopal", "Nagpur",
    "Jaipur", "Patna", "Varanasi", "Ludhiana", "Nashik",
]

CROPS = {
    "Wheat":   {"safe_window_days": 21, "duration_days": 145},
    "Maize":   {"safe_window_days": 14, "duration_days": 110},
    "Soybean": {"safe_window_days": 18, "duration_days": 100},
    "Cotton":  {"safe_window_days": 21, "duration_days": 160},
}

YEARS = list(range(2005, 2025))
WEEKS_IN_MONSOON = list(range(22, 40))  # roughly late May to Sept, ISO weeks


def generate_rainfall_history():
    """Simulates weekly rainfall per district per year during monsoon weeks."""
    rows = []
    for district in DISTRICTS:
        base_normal = np.random.uniform(35, 90)  # mm/week baseline, varies by district
        for year in YEARS:
            # Some years are drought years (El Nino-like), simulated randomly
            drought_year = np.random.random() < 0.22
            year_factor = np.random.uniform(0.35, 0.7) if drought_year else np.random.uniform(0.8, 1.3)
            for week in WEEKS_IN_MONSOON:
                normal = base_normal * (1 + 0.15 * np.sin((week - 22) / 17 * np.pi))
                actual = max(0, normal * year_factor * np.random.uniform(0.6, 1.4))
                rows.append({
                    "district": district,
                    "year": year,
                    "week_of_year": week,
                    "rainfall_mm": round(actual, 1),
                    "normal_rainfall_mm": round(normal, 1),
                })
    return pd.DataFrame(rows)


def generate_sowing_outcomes(rainfall_df):
    """
    Simulates historical sowing events and whether the crop failed,
    driven by the rainfall deficit in the weeks right after sowing.
    This is the label data your model will learn to predict.
    """
    rows = []
    for district in DISTRICTS:
        for year in YEARS:
            for crop, meta in CROPS.items():
                sow_week = np.random.choice(range(22, 30))
                window_weeks = max(1, meta["safe_window_days"] // 7)

                window = rainfall_df[
                    (rainfall_df.district == district)
                    & (rainfall_df.year == year)
                    & (rainfall_df.week_of_year >= sow_week)
                    & (rainfall_df.week_of_year < sow_week + window_weeks)
                ]
                if window.empty:
                    continue

                actual_total = window.rainfall_mm.sum()
                normal_total = window.normal_rainfall_mm.sum()
                deficit_pct = 1 - (actual_total / normal_total) if normal_total > 0 else 0

                # Failure probability rises sharply once deficit crosses ~35%
                fail_prob = 1 / (1 + np.exp(-8 * (deficit_pct - 0.35)))
                crop_failed = np.random.random() < fail_prob

                rows.append({
                    "district": district,
                    "year": year,
                    "crop": crop,
                    "sow_week": sow_week,
                    "rainfall_deficit_pct": round(deficit_pct * 100, 1),
                    "crop_failed": int(crop_failed),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rainfall_df = generate_rainfall_history()
    outcomes_df = generate_sowing_outcomes(rainfall_df)

    rainfall_df.to_csv("data/rainfall_history.csv", index=False)
    outcomes_df.to_csv("data/sowing_outcomes.csv", index=False)

    print(f"rainfall_history.csv  -> {len(rainfall_df)} rows")
    print(f"sowing_outcomes.csv   -> {len(outcomes_df)} rows")
    print(f"Crop failure rate in synthetic data: {outcomes_df.crop_failed.mean():.1%}")
