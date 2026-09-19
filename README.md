# GridWise AI

## Energy intelligence from forecast to decision

**GridWise AI** is an end-to-end energy analytics platform that forecasts short-horizon electricity demand and converts the forecast into a constrained operating schedule. It is designed to demonstrate the full workflow expected in applied data science and industrial analytics: data acquisition, leakage-aware feature engineering, chronological validation, model benchmarking, mathematical optimization, visualization, testing, and continuous integration.

The project is intentionally honest about its scope. It uses a public household-energy dataset as a reproducible proxy for industrial telemetry, while the architecture is designed around problems that transfer to buildings, factories, and energy operations.

[![CI](https://github.com/hossiendehghan989/gridwise-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/hossiendehghan989/gridwise-ai/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why the project is technically interesting

A demand forecast is not yet an operational solution. An operator also needs to know what action is feasible, what constraints it obeys, and what trade-off it creates. GridWise AI therefore connects two different technical layers:

- **Forecasting:** estimate future appliance demand from environmental, calendar, and historical-demand signals.
- **Optimization:** schedule a fixed amount of flexible load while respecting a per-period power cap and balancing peak demand with illustrative price and carbon signals.

The design makes the business logic visible. A reviewer can trace each result from the input data to the model metrics and then to the recommended schedule.

## Validated results

The current benchmark uses the final 20% of the chronological dataset as an out-of-sample holdout. No random shuffling is used.

| Model | MAE (Wh) | RMSE (Wh) | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Naive: previous reading | 26.5 | 66.2 | 0.46 | 21.9% |
| **Gradient Boosting** | **29.2** | **62.0** | **0.53** | **26.8%** |
| Naive: rolling mean | 34.1 | 75.1 | 0.31 | 28.9% |
| Extra Trees | 39.4 | 67.7 | 0.44 | 40.9% |
| Naive: same hour yesterday | 61.0 | 118.4 | -0.71 | 61.9% |
| Random Forest | 63.1 | 96.1 | -0.13 | 69.8% |

The important result is not that one model wins every metric. The benchmark shows that the naive previous-reading baseline has lower MAE, while Gradient Boosting has the best RMSE and R². This is a useful operational finding: model selection depends on whether the business values average absolute error, peak-risk sensitivity, or explained variance.

In the included 12-period scheduling scenario, the optimizer preserves the 1,200 Wh flexible-energy budget and reduces the combined demand peak by 100 Wh under a 300 Wh per-period cap.

## Architecture

```text
                     ┌─────────────────────┐
                     │ UCI energy dataset  │
                     └──────────┬──────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │ Validation and feature layer  │
                │ calendar · weather · lags     │
                └──────────────┬────────────────┘
                               │
                               ▼
             ┌────────────────────────────────────┐
             │ Chronological model benchmark      │
             │ naive · gradient boosting · trees  │
             └──────────────┬─────────────────────┘
                            │ forecast
                            ▼
        ┌────────────────────────────────────────────┐
        │ Linear-program scheduler                   │
        │ energy budget · power cap · peak · price   │
        │ carbon signal                              │
        └─────────────────┬──────────────────────────┘
                          │
                          ▼
             ┌─────────────────────────────────┐
             │ Streamlit decision dashboard   │
             │ metrics · charts · scenario     │
             └─────────────────────────────────┘
```

## Methodology

### Data and target

The target is `Appliances`, the measured appliance energy consumption in watt-hours. The public UCI dataset contains indoor temperatures, indoor humidity, outdoor weather variables, lighting consumption, and timestamps at ten-minute intervals. The project aggregates no data artificially; it uses the published measurements directly and downloads them through `download_data.py`.

### Feature engineering

The feature layer contains calendar indicators, cyclical encodings for hour and day of week, indoor and outdoor environmental measurements, and historical demand variables. Historical features are shifted before rolling statistics are calculated. This prevents the current target from entering its own predictors.

### Evaluation design

The data is ordered by time. The first 80% is used for training and the final 20% is reserved for evaluation. Every model and baseline is measured on the same holdout. This design is more appropriate for forecasting than a random split because it reflects the direction in which a production system would make predictions.

The repository reports MAE, RMSE, R², and MAPE. No single metric is treated as universally correct. MAE is easy to communicate, RMSE penalizes large misses, and R² describes explained variance relative to a constant baseline.

## Research extension

The repository also includes a deeper experimental layer in [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md). It adds expanding-window walk-forward evaluation, quantile prediction intervals, feature ablation, and scheduling sensitivity analysis. The current experiments produced 80.8% empirical coverage for a nominal 80% interval with a mean width of 87.4 Wh. Removing historical lag features increased holdout RMSE from 62.0 Wh to 143.2 Wh, while removing weather variables reduced it to 61.0 Wh on this dataset. These findings are reported as empirical observations, not universal claims.

Run the research experiments with:

```bash
python research_experiment.py
```

The command regenerates walk-forward, feature-ablation, prediction-interval, and scheduling-sensitivity artifacts.

### Optimization formulation

Let `x_t` be flexible load scheduled at period `t`, and let `z` represent the resulting peak. The optimizer solves a linear program that minimizes a weighted objective containing the peak variable and normalized price and carbon signals.

Subject to:

```text
0 ≤ x_t ≤ maximum_power_per_period
Σ x_t = required_flexible_energy
forecast_demand_t + x_t ≤ z
```

The formulation guarantees that the flexible-energy budget is conserved. It also exposes the trade-off between peak reduction and lower-cost or lower-carbon periods. The dashboard uses illustrative signals to make the trade-off visible; a production deployment would replace them with live tariffs and grid-intensity data.

## Run locally

```bash
git clone https://github.com/hossiendehghan989/gridwise-ai.git
cd gridwise-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
pytest -q
python validate.py
streamlit run dashboard.py
```

The dashboard displays the model benchmark, out-of-sample actual-versus-forecast behavior, and an interactive scheduling scenario. The `validate.py` script prints the benchmark table and verifies that the optimized energy budget is conserved.

## Project structure

```text
.
├── .github/workflows/ci.yml       # automated tests on push and pull request
├── dashboard.py                   # Streamlit decision interface
├── download_data.py               # reproducible public-data download
├── validate.py                    # benchmark and optimization validation
├── requirements.txt
├── src/energy_optimizer.py        # features, models, metrics, optimizer
├── tests/test_energy_optimizer.py
├── LICENSE
└── data/                          # local downloaded data, ignored by Git
```

## Resume-ready description

> Built **GridWise AI**, an end-to-end energy intelligence platform that benchmarks leakage-aware demand forecasts on a chronological holdout and converts predictions into a constrained linear-program schedule. Implemented Python, scikit-learn, SciPy, Streamlit, automated tests, and GitHub Actions; achieved 62.0 Wh RMSE and 0.53 R² with Gradient Boosting while preserving a fixed flexible-energy budget.

## Honest limitations and next steps

This repository is a serious portfolio implementation, not a claim of production readiness. The dataset represents a home rather than a factory, and the price and carbon vectors in the dashboard are illustrative. A production extension would add real tariff and carbon feeds, equipment-level constraints, probabilistic prediction intervals, a model registry, drift monitoring, an API, database persistence, and deployment infrastructure.

## Data source

[1]: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction "UCI Appliances Energy Prediction dataset"

The data is provided by the UCI Machine Learning Repository [1].

## License

MIT License. See [LICENSE](LICENSE).
