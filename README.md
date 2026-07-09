# 🌾 Sow Smart — sowing risk advisory for seed companies

**Predicts crop-failure risk from delayed monsoon rainfall, and turns it into a decision:
sow now, delay, switch to a drought-tolerant variety, or avoid the window entirely.**

> Built to solve a real problem: my father's seed company loses money every season crops
> are sown before a delayed or weak monsoon, and there's no early-warning system to catch
> it in time. Sow Smart is that early-warning system.

`XGBoost` `FastAPI` `scikit-learn` `pandas` — trained on IMD-shaped rainfall data, ROC-AUC 0.77

**[→ Try the live dashboard](./dashboard.html)** — pick a district and crop, watch the risk
gauge and recommendation update in real time.

### What makes this different from a typical "weather + ML" student project
- **It ends in a decision, not a number.** Most rainfall-prediction projects stop at a
  probability. This one converts that probability into a specific, actionable recommendation
  — including *which* drought-tolerant seed variety to switch to.
- **It's explainable.** Every recommendation ships with the top contributing factor
  (e.g. "pre-sowing rainfall deficit of 42% vs normal"), not just a black-box score.
- **It's grounded in a real business, not a Kaggle dataset.** The problem, the variety
  catalog, and the decision thresholds all came from an actual seed company's operations.

---

## Full documentation below


## Why this exists

Most "AI weather" student projects stop at "here's a rainfall prediction." Sow Smart goes
one step further — it converts that prediction into a business decision, with an
explainable "why," because that's what an actual seed company would need to act on it.

## Architecture

```
IMD rainfall data ─┐
                    ├─> preprocessing ─> XGBoost risk model ─> recommendation engine ─> dashboard
Farmer/company input┘
```

1. **Data** (`src/generate_synthetic_data.py`) — generates a synthetic but realistic
   district-level weekly rainfall dataset and historical sowing outcomes, shaped exactly
   like the real data.gov.in "Rainfall" catalog. Swap in the real CSV when you have it —
   see "Using real data" below.
2. **Features** (`src/features.py`) — builds the feature table: pre-sowing rainfall
   deficit, district's historical drought-proneness, sow week, crop, district.
3. **Model** (`src/train_model.py`) — trains an XGBoost classifier to predict crop
   failure probability. Chosen over a black-box deep model because it's fast, handles
   the class imbalance (~25% failure rate) well with `scale_pos_weight`, and gives
   feature importances for explainability.
4. **Recommendation engine** (`src/recommend.py`) — converts the risk score into a risk
   tier (0-25 sow now / 26-50 delay / 51-75 switch variety / 76-100 avoid), and for the
   two higher tiers, looks up a drought-tolerant variety from `data/variety_catalog.csv`.
5. **API** (`api/main.py`) — FastAPI backend exposing `/predict`, `/districts`, `/crops`.
6. **Dashboard** (`dashboard.html`) — standalone HTML dashboard with a live-updating risk
   gauge and a field-level overview table. Currently reads from a static demo snapshot;
   point it at the live API for production use (see below).

## Model performance (on synthetic data)

- ROC-AUC: 0.77
- Top feature: pre-sowing rainfall deficit (44% of model importance) — matches domain
  intuition, which is worth calling out explicitly in an interview.

Run `python3 src/train_model.py` to see the full classification report and confusion matrix.

## Running it yourself

```bash
pip install -r requirements.txt
python3 src/generate_synthetic_data.py
python3 src/features.py
python3 src/train_model.py
uvicorn api.main:app --reload --port 8000
```

Then open `dashboard.html` in a browser (currently uses a static data snapshot — see
"Connecting the live dashboard" below to wire it to the running API instead).

## Using real data

Replace `data/rainfall_history.csv` with a real export from:
- **data.gov.in** → search "Rainfall" catalog (district-wise + IMD gridded data)
- **IMD's CDSP portal** (cdsp.imdpune.gov.in) → gridded rainfall/temperature downloads

Keep the same columns (`district, year, week_of_year, rainfall_mm, normal_rainfall_mm`)
and every downstream script works unchanged. For `sowing_outcomes.csv`, replace with your
own company's historical sowing dates and outcomes if available — that's the single
biggest upgrade you can make to this project, since it's proprietary data almost nobody
else building a similar project will have access to.

## Connecting the live dashboard

`dashboard.html` currently embeds a static JSON snapshot for demo purposes (no backend
required to view it). To make it call the live FastAPI backend instead, replace the
`const DATA = [...]` line with a `fetch('http://localhost:8000/predict', ...)` call per
district/crop combination, using the `/districts` and `/crops` endpoints to populate
the dropdowns dynamically.

## Honest limitations (worth stating in interviews, not hiding)

- Trained on synthetic data until real IMD/company data is substituted — say this
  upfront, it's more credible than pretending otherwise.
- Variety catalog uses illustrative durations for real Indian seed varieties (HD 2967,
  HD 3086, JS 335, etc.) — verify exact figures against your company's actual catalogue
  before treating this as agronomic advice.
- The "current season outlook" in the API currently uses historical district averages
  as a stand-in for a live seasonal forecast. Swapping in IMD's actual seasonal forecast
  API would be the natural next step for a production version.

## Stack

Python (pandas, scikit-learn, XGBoost, FastAPI) · vanilla HTML/CSS/JS dashboard
