# GridWise AI

## Forecasting and optimization for smarter energy operations

**GridWise AI** is an end-to-end energy intelligence project that forecasts household electricity demand and optimizes flexible load placement. It combines supervised machine learning with constrained linear optimization to demonstrate how an industrial engineer can move from raw operational data to a measurable decision-support solution.

The project is designed as a portfolio artifact: it is reproducible, testable, explainable, and deployable as an interactive dashboard.

## Why this project matters

Energy systems do not only need accurate forecasts. They also need decisions that respect operational constraints. GridWise AI addresses both problems in one workflow:

1. It learns demand patterns from environmental, calendar, and appliance measurements.
2. It evaluates predictions on a chronological out-of-sample holdout.
3. It shifts a fixed amount of flexible demand into lower-load periods while respecting a power cap.
4. It visualizes the forecast and the operational effect of the optimized schedule.

## Technical highlights

| Layer | Implementation |
| --- | --- |
| **Data** | UCI Appliances Energy Prediction dataset with indoor climate, outdoor weather, lights, and appliance energy measurements. |
| **Feature engineering** | Calendar variables, weekend flags, and cyclical hour/day-of-week encodings. |
| **Forecasting** | Histogram Gradient Boosting regression with chronological validation. |
| **Optimization** | Linear programming with fixed flexible energy, per-period power bounds, and demand-aware scheduling. |
| **Product layer** | Streamlit dashboard with forecast metrics, time-series visualization, and peak-reduction scenario. |
| **Quality** | Automated tests for feature construction and schedule feasibility. |

## Architecture

```text
UCI energy data
      │
      ▼
Feature engineering ──► Gradient-boosting forecast ──► Out-of-sample metrics
      │                                                     │
      └──────────────────────► Linear-program schedule ◄────┘
                                      │
                                      ▼
                              Streamlit decision dashboard
```

## Quick start

```bash
git clone https://github.com/hossiendehghan989/gridwise-ai.git
cd gridwise-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
pytest -q
streamlit run dashboard.py
```

The dashboard opens locally and reports mean absolute error, root mean squared error, R², forecast behavior, baseline peak demand, optimized peak demand, and the amount of flexible energy shifted.

On the downloaded UCI holdout, the current model reached **30.5 Wh MAE**, **62.8 Wh RMSE**, and **0.52 R²**. In the included 12-period scenario, the optimizer reduced the combined demand peak by **100 Wh** while preserving the flexible-energy budget.

## Reproducibility

The repository does not commit the downloaded dataset. Running `download_data.py` retrieves the public dataset from the UCI Machine Learning Repository and extracts it into `data/`. The training procedure uses a chronological split rather than a random split to avoid leaking future information into the past.

## Resume-ready impact statement

> Built **GridWise AI**, an end-to-end energy intelligence platform that combines chronological demand forecasting with constrained linear-program optimization; delivered a tested Streamlit dashboard that translates model output into peak-reduction decisions.

## Project structure

```text
.
├── dashboard.py
├── download_data.py
├── requirements.txt
├── src/
│   └── energy_optimizer.py
├── tests/
│   └── test_energy_optimizer.py
└── data/                 # downloaded locally, not committed
```

## Limitations and next steps

This first version uses a public household-energy dataset rather than live industrial telemetry. A production version would add a proper data-ingestion layer, probabilistic forecasts, electricity-price and carbon-intensity feeds, equipment constraints, model monitoring, and a database-backed API.

## License

MIT License. See `LICENSE` for details.

## Data source

[1]: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction "UCI Appliances Energy Prediction dataset"

The dataset is provided by the UCI Machine Learning Repository [1].
