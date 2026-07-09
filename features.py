"""
features.py

Turns raw rainfall history + sowing outcomes into a model-ready feature table.
"""

import pandas as pd


def build_features(rainfall_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, outcome in outcomes_df.iterrows():
        district, year, sow_week = outcome.district, outcome.year, outcome.sow_week

        # Rainfall in the 4 weeks BEFORE sowing (soil moisture proxy)
        pre_window = rainfall_df[
            (rainfall_df.district == district)
            & (rainfall_df.year == year)
            & (rainfall_df.week_of_year >= sow_week - 4)
            & (rainfall_df.week_of_year < sow_week)
        ]
        pre_actual = pre_window.rainfall_mm.sum()
        pre_normal = pre_window.normal_rainfall_mm.sum()
        pre_deficit_pct = (1 - pre_actual / pre_normal) * 100 if pre_normal > 0 else 0

        # District's long-run average deficit (drought-proneness signal)
        district_hist = rainfall_df[
            (rainfall_df.district == district) & (rainfall_df.year < year)
        ]
        if not district_hist.empty and district_hist.normal_rainfall_mm.sum() > 0:
            district_avg_deficit_pct = (
                1 - district_hist.rainfall_mm.sum() / district_hist.normal_rainfall_mm.sum()
            ) * 100
        else:
            district_avg_deficit_pct = 0

        rows.append({
            "district": district,
            "year": year,
            "crop": outcome.crop,
            "sow_week": sow_week,
            "pre_sowing_deficit_pct": round(pre_deficit_pct, 1),
            "district_avg_deficit_pct": round(district_avg_deficit_pct, 1),
            "crop_failed": outcome.crop_failed,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    rainfall_df = pd.read_csv("data/rainfall_history.csv")
    outcomes_df = pd.read_csv("data/sowing_outcomes.csv")

    feature_df = build_features(rainfall_df, outcomes_df)
    feature_df.to_csv("data/features.csv", index=False)
    print(f"features.csv -> {len(feature_df)} rows, columns: {list(feature_df.columns)}")
