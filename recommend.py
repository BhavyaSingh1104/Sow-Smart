"""
recommend.py

Turns a model risk score into the actual advice a district manager or
farmer would act on: sow now / delay / switch variety / avoid window,
plus a variety suggestion and a plain-language reason.
"""

import pandas as pd

VARIETY_CATALOG = pd.read_csv("data/variety_catalog.csv")

RISK_TIERS = [
    (25, "sow_now", "Low risk. Rainfall outlook supports sowing as planned."),
    (50, "delay_7_10_days", "Moderate risk. Delay sowing by 7-10 days and re-check the forecast."),
    (75, "switch_variety", "High risk. Standard variety is unlikely to survive this window."),
    (100, "avoid_window", "Severe risk. Recommend against sowing in this window entirely."),
]


def get_recommendation(
    crop: str,
    risk_score: float,
    pre_sowing_deficit_pct: float,
    days_to_window_close: int = 14,
):
    """
    crop: crop name, must match variety_catalog.csv
    risk_score: model output, 0-100 (probability of failure * 100)
    pre_sowing_deficit_pct: top feature driving the score, used for the "why"
    days_to_window_close: days left in the agronomically safe sowing window
    """
    for threshold, action, reason in RISK_TIERS:
        if risk_score <= threshold:
            break

    result = {
        "risk_score": round(risk_score, 1),
        "action": action,
        "reason": reason,
        "top_driver": f"Pre-sowing rainfall deficit of {pre_sowing_deficit_pct:.0f}% vs normal",
        "variety_suggestion": None,
    }

    if action in ("switch_variety", "avoid_window"):
        options = VARIETY_CATALOG[
            (VARIETY_CATALOG.crop == crop) & (VARIETY_CATALOG.drought_tolerant == "Yes")
        ]
        # Prefer a variety short enough to still fit before the window closes
        fitting = options[options.duration_days <= days_to_window_close * 20]  # rough buffer
        chosen = fitting.iloc[0] if not fitting.empty else (options.iloc[0] if not options.empty else None)

        if chosen is not None:
            standard = VARIETY_CATALOG[
                (VARIETY_CATALOG.crop == crop) & (VARIETY_CATALOG.drought_tolerant == "No")
            ]
            standard_name = standard.iloc[0].variety_name if not standard.empty else "standard variety"
            result["variety_suggestion"] = (
                f"Switch from {standard_name} to {chosen.variety_name} "
                f"({chosen.duration_days}-day drought-tolerant variety)"
            )

    return result


if __name__ == "__main__":
    # Quick smoke test with a few example scenarios
    examples = [
        {"crop": "Wheat", "risk_score": 15, "pre_sowing_deficit_pct": 8},
        {"crop": "Maize", "risk_score": 42, "pre_sowing_deficit_pct": 28},
        {"crop": "Soybean", "risk_score": 68, "pre_sowing_deficit_pct": 45},
        {"crop": "Cotton", "risk_score": 90, "pre_sowing_deficit_pct": 60},
    ]
    for ex in examples:
        rec = get_recommendation(**ex)
        print(f"\n{ex['crop']} | risk={ex['risk_score']}")
        for k, v in rec.items():
            print(f"  {k}: {v}")
