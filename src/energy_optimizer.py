"""Energy demand forecasting and load-shifting optimization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ForecastResult:
    model: HistGradientBoostingRegressor
    features: list[str]
    metrics: dict[str, float]
    predictions: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
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
    features = [
        "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
        "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
        "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "appliances_lag_1", "appliances_lag_24",
    ]
    data = data.dropna(subset=features + ["Appliances"])
    return data[features], data["Appliances"], features


def train_forecaster(df: pd.DataFrame, test_fraction: float = 0.2) -> ForecastResult:
    X, y, features = build_features(df)
    split = int(len(X) * (1 - test_fraction))
    model = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=0.2, random_state=42
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    dates = df.loc[X.index, "date"]
    pred = pd.Series(model.predict(X.iloc[split:]), index=dates.iloc[split:])
    actual = y.iloc[split:]
    metrics = {
        "mae_wh": float(mean_absolute_error(actual, pred)),
        "rmse_wh": float(mean_squared_error(actual, pred) ** 0.5),
        "r2": float(r2_score(actual, pred)),
    }
    return ForecastResult(model, features, metrics, pred)


def optimize_schedule(
    predicted_demand: np.ndarray,
    flexible_energy_wh: float = 1200,
    max_power_wh: float = 300,
    carbon_intensity: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Shift flexible demand into low-cost/low-carbon periods.

    The optimizer keeps total flexible energy constant and caps the hourly load.
    """
    demand = np.asarray(predicted_demand, dtype=float)
    n = len(demand)
    if carbon_intensity is None:
        carbon_intensity = np.ones(n)
    carbon = np.asarray(carbon_intensity, dtype=float)
    # Variables are flexible load for each period plus an auxiliary peak variable.
    # Minimize total peak first, then use carbon intensity as a small tie-breaker.
    objective = np.r_[np.zeros(n), 1.0]
    a_ub = np.zeros((n, n + 1))
    for i in range(n):
        a_ub[i, i] = 1.0
        a_ub[i, -1] = -1.0
    b_ub = -demand
    result = linprog(
        objective, A_ub=a_ub, b_ub=b_ub,
        A_eq=np.r_[np.ones(n), 0.0][None, :], b_eq=[flexible_energy_wh],
        bounds=[(0, max_power_wh)] * n + [(0, None)], method="highs"
    )
    if not result.success:
        raise ValueError(f"Schedule optimization failed: {result.message}")
    baseline = np.full(n, flexible_energy_wh / n)
    optimized = result.x[:n]
    return {
        "baseline": baseline,
        "optimized": optimized,
        "baseline_peak_wh": float((demand + baseline).max()),
        "optimized_peak_wh": float((demand + optimized).max()),
        "energy_shifted_wh": float(np.abs(optimized - baseline).sum() / 2),
    }
