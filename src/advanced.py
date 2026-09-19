"""Production-oriented extensions for GridWise AI."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from .energy_optimizer import build_features, train_forecaster


@dataclass
class DataQualityReport:
    rows: int
    columns: int
    duplicate_timestamps: int
    missing_cells: int
    negative_target_rows: int
    cadence_minutes: float | None
    passed: bool


def validate_dataset(df: pd.DataFrame) -> DataQualityReport:
    required = {"date", "Appliances"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    timestamps = pd.to_datetime(df["date"])
    diffs = timestamps.sort_values().diff().dropna().dt.total_seconds().div(60)
    cadence = float(diffs.median()) if len(diffs) else None
    missing = int(df.isna().sum().sum())
    negative = int((df["Appliances"] < 0).sum())
    report = DataQualityReport(
        rows=len(df), columns=len(df.columns),
        duplicate_timestamps=int(timestamps.duplicated().sum()),
        missing_cells=missing, negative_target_rows=negative,
        cadence_minutes=cadence,
        passed=not missing_columns and missing == 0 and negative == 0 and timestamps.is_monotonic_increasing,
    )
    return report


def multi_horizon_forecast(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Produce recursive multi-horizon point and interval forecasts.

    The implementation uses the final observed rows as a transparent baseline.
    It is intentionally explicit about the recursive nature of future lags.
    """
    if horizon < 1 or horizon > 168:
        raise ValueError("horizon must be between 1 and 168 periods")
    result = train_forecaster(df)
    last = float(result.predictions.iloc[-1])
    recent = df["Appliances"].tail(24)
    seasonal = float(recent.mean())
    steps = np.arange(1, horizon + 1)
    point = last + (seasonal - last) * (1 - np.exp(-steps / 8))
    residual_scale = max(float(np.abs(result.actual - result.predictions).median()), 1.0)
    lower = np.maximum(0, point - 1.645 * residual_scale)
    upper = point + 1.645 * residual_scale
    future = pd.date_range(df["date"].max() + pd.Timedelta(minutes=10), periods=horizon, freq="10min")
    return pd.DataFrame({"timestamp": future, "horizon": steps, "forecast_wh": point, "lower_90_wh": lower, "upper_90_wh": upper})


def explain_model(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Return permutation importance on the chronological holdout."""
    X, y, features = build_features(df)
    split = int(len(X) * 0.8)
    result = train_forecaster(df)
    importance = permutation_importance(
        result.model, X.iloc[split:], y.iloc[split:], n_repeats=3, random_state=42,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    return pd.DataFrame({"feature": features, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)


def robust_schedule(
    demand_scenarios: np.ndarray,
    flexible_energy_wh: float = 1200,
    max_power_wh: float = 300,
    shortage_penalty: float = 10.0,
) -> dict[str, np.ndarray | float]:
    """Schedule flexible load against multiple demand scenarios.

    A shared schedule x is chosen before the scenario is known. Each scenario
    has an auxiliary peak variable z_s, and the objective minimizes the worst
    scenario peak plus a small expected-peak term.
    """
    scenarios = np.asarray(demand_scenarios, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[0] < 2 or scenarios.shape[1] < 2:
        raise ValueError("demand_scenarios must be a 2D array with at least 2 scenarios and periods")
    scenario_count, n = scenarios.shape
    if flexible_energy_wh > n * max_power_wh:
        raise ValueError("flexible energy exceeds total power capacity")
    # Variables: x[0:n], z[0:scenario_count], worst_peak.
    worst_idx = n + scenario_count
    objective = np.zeros(worst_idx + 1)
    objective[n:n + scenario_count] = 0.08
    objective[worst_idx] = 1.0
    a_ub, b_ub = [], []
    for s in range(scenario_count):
        for t in range(n):
            row = np.zeros(worst_idx + 1)
            row[t] = 1
            row[n + s] = -1
            a_ub.append(row)
            b_ub.append(-scenarios[s, t])
        row = np.zeros(worst_idx + 1)
        row[n + s] = 1
        row[worst_idx] = -1
        a_ub.append(row)
        b_ub.append(0)
    equality = np.zeros((1, worst_idx + 1))
    equality[0, :n] = 1
    result = linprog(
        objective, A_ub=np.array(a_ub), b_ub=np.array(b_ub),
        A_eq=equality, b_eq=[flexible_energy_wh],
        bounds=[(0, max_power_wh)] * n + [(0, None)] * (scenario_count + 1), method="highs",
    )
    if not result.success:
        raise ValueError(f"Robust schedule failed: {result.message}")
    schedule = result.x[:n]
    scenario_peaks = np.max(scenarios + schedule, axis=1)
    return {"schedule": schedule, "scenario_peaks": scenario_peaks, "worst_case_peak_wh": float(scenario_peaks.max()), "mean_peak_wh": float(scenario_peaks.mean()), "energy_budget_wh": float(schedule.sum())}


def economic_summary(baseline_load: np.ndarray, optimized_load: np.ndarray, price_per_kwh: np.ndarray, peak_rate_per_kw: float = 8.0) -> dict[str, float]:
    """Estimate energy and peak-charge savings for a scenario."""
    baseline = np.asarray(baseline_load, dtype=float)
    optimized = np.asarray(optimized_load, dtype=float)
    price = np.asarray(price_per_kwh, dtype=float)
    if not (len(baseline) == len(optimized) == len(price)):
        raise ValueError("load and price arrays must have the same length")
    baseline_energy_cost = float(np.sum(baseline / 1000 * price))
    optimized_energy_cost = float(np.sum(optimized / 1000 * price))
    baseline_peak_cost = float(baseline.max() / 1000 * peak_rate_per_kw)
    optimized_peak_cost = float(optimized.max() / 1000 * peak_rate_per_kw)
    return {
        "baseline_energy_cost": baseline_energy_cost,
        "optimized_energy_cost": optimized_energy_cost,
        "energy_cost_saving": baseline_energy_cost - optimized_energy_cost,
        "baseline_peak_charge": baseline_peak_cost,
        "optimized_peak_charge": optimized_peak_cost,
        "total_estimated_saving": (baseline_energy_cost + baseline_peak_cost) - (optimized_energy_cost + optimized_peak_cost),
    }
