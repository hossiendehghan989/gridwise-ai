"""Research utilities for robust evaluation and uncertainty quantification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .energy_optimizer import build_features, optimize_schedule


@dataclass
class IntervalResult:
    predictions: pd.DataFrame
    coverage: float
    mean_width_wh: float


def walk_forward_evaluation(
    df: pd.DataFrame,
    folds: int = 4,
    min_train_fraction: float = 0.5,
) -> pd.DataFrame:
    """Evaluate the model on expanding-window, strictly future holdouts."""
    X, y, _ = build_features(df)
    n = len(X)
    first_test = max(int(n * min_train_fraction), 1)
    remaining = n - first_test
    fold_size = max(1, remaining // folds)
    rows = []
    for fold in range(folds):
        train_end = first_test + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.055, max_leaf_nodes=31,
            l2_regularization=0.4, random_state=42,
        )
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        pred = model.predict(X.iloc[train_end:test_end])
        actual = y.iloc[train_end:test_end]
        rows.append({
            "fold": fold + 1,
            "train_observations": train_end,
            "test_observations": test_end - train_end,
            "mae_wh": mean_absolute_error(actual, pred),
            "rmse_wh": mean_squared_error(actual, pred) ** 0.5,
        })
    return pd.DataFrame(rows)


def quantile_prediction_interval(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> IntervalResult:
    """Estimate a conditional prediction interval using quantile boosting."""
    X, y, _ = build_features(df)
    split = int(len(X) * (1 - test_fraction))
    dates = df.loc[X.index, "date"].iloc[split:]
    predictions = {}
    for name, quantile in (("lower", lower_quantile), ("median", 0.5), ("upper", upper_quantile)):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_iter=300,
            learning_rate=0.055, max_leaf_nodes=31, l2_regularization=0.4, random_state=42,
        )
        model.fit(X.iloc[:split], y.iloc[:split])
        predictions[name] = model.predict(X.iloc[split:])
    frame = pd.DataFrame(predictions, index=dates)
    frame["actual"] = y.iloc[split:].to_numpy()
    frame["covered"] = frame["actual"].between(frame["lower"], frame["upper"])
    return IntervalResult(
        frame,
        float(frame["covered"].mean()),
        float((frame["upper"] - frame["lower"]).mean()),
    )


def feature_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Measure how much historical and weather features contribute."""
    X, y, features = build_features(df)
    split = int(len(X) * 0.8)
    groups = {
        "full feature set": features,
        "without historical lags": [f for f in features if "lag" not in f and "roll" not in f],
        "without weather variables": [f for f in features if not any(k in f for k in ("T", "RH", "Press", "Wind", "Visibility", "Tdew"))],
        "calendar only": [f for f in features if f in {"hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"}],
    }
    rows = []
    for name, selected in groups.items():
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.055, max_leaf_nodes=31, l2_regularization=0.4, random_state=42)
        model.fit(X[selected].iloc[:split], y.iloc[:split])
        pred = model.predict(X[selected].iloc[split:])
        actual = y.iloc[split:]
        rows.append({
            "feature_set": name,
            "features": len(selected),
            "mae_wh": mean_absolute_error(actual, pred),
            "rmse_wh": mean_squared_error(actual, pred) ** 0.5,
        })
    return pd.DataFrame(rows).sort_values("rmse_wh").reset_index(drop=True)


def scheduling_sensitivity(predicted_demand: np.ndarray) -> pd.DataFrame:
    """Quantify how scheduling outcomes change across objective weights."""
    rows = []
    for price_weight, carbon_weight in ((0, 0), (0.25, 0), (0, 0.25), (0.25, 0.15), (0.5, 0.5)):
        n = len(predicted_demand)
        price = np.linspace(0.4, 1.2, n)
        carbon = np.linspace(0.9, 0.45, n)
        plan = optimize_schedule(
            predicted_demand, flexible_energy_wh=1200, max_power_wh=300,
            price=price, carbon_intensity=carbon,
            price_weight=price_weight, carbon_weight=carbon_weight,
        )
        rows.append({
            "price_weight": price_weight,
            "carbon_weight": carbon_weight,
            "peak_reduction_wh": plan["peak_reduction_wh"],
            "energy_shifted_wh": plan["energy_shifted_wh"],
            "cost_index": plan["optimized_cost_index"],
            "carbon_index": plan["optimized_carbon_index"],
        })
    return pd.DataFrame(rows)
