import numpy as np
import pandas as pd

from src.energy_optimizer import build_features, optimize_schedule


def sample_frame(n=48):
    dates = pd.date_range("2016-01-01", periods=n, freq="h")
    data = {"date": dates, "Appliances": np.linspace(50, 250, n), "lights": np.ones(n)}
    for i in range(1, 10):
        data[f"T{i}"] = np.full(n, 20.0)
        data[f"RH_{i}"] = np.full(n, 45.0)
    data["T_out"] = 10.0
    data["Press_mm_hg"] = 750.0
    data["Windspeed"] = 2.0
    data["Visibility"] = 10.0
    data["Tdewpoint"] = 8.0
    return pd.DataFrame(data)


def test_feature_builder_has_cyclical_time_features():
    X, y, features = build_features(sample_frame())
    assert len(X) == len(y) == 24
    assert "hour_sin" in features
    assert "dow_cos" in features
    assert "appliances_lag_24" in features


def test_optimizer_preserves_flexible_energy_and_caps_load():
    plan = optimize_schedule(np.array([100, 200, 300, 150], dtype=float), flexible_energy_wh=400, max_power_wh=200)
    assert np.isclose(plan["optimized"].sum(), 400)
    assert plan["optimized"].max() <= 200.0001
    assert plan["energy_shifted_wh"] > 0
