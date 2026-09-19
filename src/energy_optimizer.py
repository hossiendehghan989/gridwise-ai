"""GridWise AI: forecasting benchmarks and constrained energy scheduling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ForecastResult:
    model: object
    features: list[str]
    metrics: dict[str, float]
    predictions: pd.Series
    actual: pd.Series
    train_size: int
    test_size: int


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and chronologically sort the UCI energy dataset."""
    df = pd.read_csv(path)
    required = {"date", "Appliances"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Create leakage-aware calendar, weather, and historical demand features."""
    data = df.copy()
    data["hour"] = data["date"].dt.hour
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["appliances_lag_1"] = data["Appliances"].shift(1)
    data["appliances_lag_24"] = data["Appliances"].shift(24)
    data["appliances_roll_6"] = data["Appliances"].shift(1).rolling(6).mean()
    features = [
        "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
        "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
        "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
        "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos", "appliances_lag_1", "appliances_lag_24", "appliances_roll_6",
    ]
    data = data.dropna(subset=features + ["Appliances"])
    return data[features], data["Appliances"], features


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return {
        "mae_wh": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse_wh": float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        "r2": float(r2_score(actual_values, predicted_values)),
        "mape_pct": float((np.abs((actual_values - predicted_values) / np.maximum(actual_values, 1))).mean() * 100),
    }


def train_forecaster(df: pd.DataFrame, test_fraction: float = 0.2) -> ForecastResult:
    """Train the primary model using a chronological holdout."""
    X, y, features = build_features(df)
    split = int(len(X) * (1 - test_fraction))
    model = HistGradientBoostingRegressor(
        max_iter=350, learning_rate=0.055, max_leaf_nodes=31,
        l2_regularization=0.4, random_state=42
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    dates = df.loc[X.index, "date"]
    actual = y.iloc[split:].copy()
    actual.index = dates.iloc[split:]
    pred = pd.Series(model.predict(X.iloc[split:]), index=dates.iloc[split:], name="forecast")
    return ForecastResult(model, features, _metrics(actual, pred), pred, actual, split, len(actual))


def benchmark_models(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, ForecastResult]:
    """Compare naive baselines and tree models on exactly the same time holdout."""
    X, y, features = build_features(df)
    split = int(len(X) * (1 - test_fraction))
    dates = df.loc[X.index, "date"]
    actual = y.iloc[split:].copy()
    actual.index = dates.iloc[split:]
    models: dict[str, Callable[[], object]] = {
        "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=350, learning_rate=0.055, max_leaf_nodes=31, l2_regularization=0.4, random_state=42),
        "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    rows: list[dict[str, float | str]] = []
    for name, factory in models.items():
        model = factory()
        model.fit(X.iloc[:split], y.iloc[:split])
        pred = pd.Series(model.predict(X.iloc[split:]), index=dates.iloc[split:])
        rows.append({"model": name, **_metrics(actual, pred)})
    raw = df.set_index("date")["Appliances"].loc[dates]
    for name, series in {
        "Naive · previous reading": raw.shift(1).loc[dates.iloc[split:]],
        "Naive · same hour yesterday": raw.shift(24).loc[dates.iloc[split:]],
        "Naive · rolling mean": raw.shift(1).rolling(6).mean().loc[dates.iloc[split:]],
    }.items():
        aligned = series.reindex(actual.index).dropna()
        rows.append({"model": name, **_metrics(actual.loc[aligned.index], aligned)})
    result = train_forecaster(df, test_fraction)
    table = pd.DataFrame(rows).sort_values("mae_wh").reset_index(drop=True)
    return table, result


def optimize_schedule(
    predicted_demand: np.ndarray,
    flexible_energy_wh: float = 1200,
    max_power_wh: float = 300,
    price: np.ndarray | None = None,
    carbon_intensity: np.ndarray | None = None,
    price_weight: float = 0.25,
    carbon_weight: float = 0.15,
) -> dict[str, np.ndarray | float]:
    """Minimize peak plus price/carbon cost under energy and power constraints."""
    demand = np.asarray(predicted_demand, dtype=float)
    if demand.ndim != 1 or len(demand) < 2 or np.any(~np.isfinite(demand)):
        raise ValueError("predicted_demand must be a finite one-dimensional array with at least 2 periods")
    n = len(demand)
    price = np.ones(n) if price is None else np.asarray(price, dtype=float)
    carbon = np.ones(n) if carbon_intensity is None else np.asarray(carbon_intensity, dtype=float)
    if len(price) != n or len(carbon) != n:
        raise ValueError("price and carbon_intensity must match the demand horizon")
    if flexible_energy_wh <= 0 or max_power_wh <= 0 or flexible_energy_wh > n * max_power_wh:
        raise ValueError("flexible_energy_wh must fit within the horizon and power cap")

    def normalize(values: np.ndarray) -> np.ndarray:
        span = values.max() - values.min()
        return (values - values.min()) / span if span > 0 else np.zeros_like(values)

    # x = flexible load per period; z = total-demand peak auxiliary variable.
    objective = np.r_[price_weight * normalize(price) + carbon_weight * normalize(carbon), 1.0]
    a_ub = np.zeros((n, n + 1))
    for i in range(n):
        a_ub[i, i] = 1.0
        a_ub[i, -1] = -1.0
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=-demand,
        A_eq=np.r_[np.ones(n), 0.0][None, :],
        b_eq=[flexible_energy_wh],
        bounds=[(0, max_power_wh)] * n + [(0, None)],
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Schedule optimization failed: {result.message}")
    baseline = np.full(n, flexible_energy_wh / n)
    optimized = result.x[:n]
    baseline_total = demand + baseline
    optimized_total = demand + optimized
    return {
        "baseline": baseline,
        "optimized": optimized,
        "baseline_total": baseline_total,
        "optimized_total": optimized_total,
        "baseline_peak_wh": float(baseline_total.max()),
        "optimized_peak_wh": float(optimized_total.max()),
        "peak_reduction_wh": float(baseline_total.max() - optimized_total.max()),
        "energy_shifted_wh": float(np.abs(optimized - baseline).sum() / 2),
        "baseline_cost_index": float(np.dot(baseline, price)),
        "optimized_cost_index": float(np.dot(optimized, price)),
        "baseline_carbon_index": float(np.dot(baseline, carbon)),
        "optimized_carbon_index": float(np.dot(optimized, carbon)),
    }
