import numpy as np
import pandas as pd
import pytest

from src.advanced import conformal_interval, explain_model, multi_horizon_forecast, validate_dataset
from src.energy_optimizer import benchmark_models, load_dataset, optimize_schedule, train_forecaster
from tests.test_energy_optimizer import sample_frame


def test_validation_rejects_missing_columns_and_reports_bad_rows():
    with pytest.raises(ValueError, match="Missing required"):
        validate_dataset(pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2, freq="h")}))
    frame = sample_frame(4)
    frame.loc[1, "Appliances"] = -1
    frame.loc[2, "T1"] = np.nan
    report = validate_dataset(frame)
    assert not report.passed
    assert report.negative_target_rows == 1
    assert report.missing_cells == 1


def test_forecast_and_conformal_interval_have_expected_shapes():
    frame = sample_frame(96)
    forecast = multi_horizon_forecast(frame, horizon=4)
    assert list(forecast.columns) == ["timestamp", "horizon", "forecast_wh", "lower_90_wh", "upper_90_wh"]
    interval = conformal_interval(frame)
    assert len(interval) > 0
    assert interval.attrs["coverage_target"] == 0.9
    assert (interval["lower"] <= interval["upper"]).all()


def test_feature_explanation_returns_ranked_features():
    result = explain_model(sample_frame(96), top_n=5)
    assert len(result) == 5
    assert result["importance_mean"].notna().all()


def test_forecasting_benchmarks_and_loader(tmp_path):
    frame = sample_frame(96)
    path = tmp_path / "energy.csv"
    frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert len(loaded) == len(frame)
    forecast = train_forecaster(frame)
    table, _ = benchmark_models(frame)
    assert forecast.test_size > 0
    assert {"model", "mae_wh", "rmse_wh"}.issubset(table.columns)


def test_optimizer_rejects_bad_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        optimize_schedule(np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="match"):
        optimize_schedule(np.array([1, 2]), price=np.array([1]))
    with pytest.raises(ValueError, match="finite"):
        optimize_schedule(np.array([1, np.nan]))
