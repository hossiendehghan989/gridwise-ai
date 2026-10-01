import numpy as np

from src.research import feature_ablation, quantile_prediction_interval, scheduling_sensitivity, walk_forward_evaluation
from tests.test_energy_optimizer import sample_frame


def test_walk_forward_evaluation_is_chronological():
    result = walk_forward_evaluation(sample_frame(96), folds=3, min_train_fraction=0.5)
    assert len(result) == 3
    assert (result["train_observations"].diff().dropna() > 0).all()
    assert (result["test_observations"] > 0).all()


def test_quantile_interval_returns_ordered_predictions():
    result = quantile_prediction_interval(sample_frame(96), test_fraction=0.2)
    assert len(result.predictions) > 0
    assert (result.predictions["lower"] <= result.predictions["median"]).all()
    assert (result.predictions["median"] <= result.predictions["upper"]).all()
    assert 0 <= result.coverage <= 1


def test_feature_ablation_reports_all_feature_groups():
    result = feature_ablation(sample_frame(96))
    assert set(result["feature_set"]) == {
        "full feature set",
        "without historical lags",
        "without weather variables",
        "calendar only",
    }
    assert result["rmse_wh"].notna().all()


def test_scheduling_sensitivity_has_five_scenarios():
    result = scheduling_sensitivity(np.array([100, 120, 140, 160, 180, 200], dtype=float))
    assert len(result) == 5
    assert np.all(result["energy_shifted_wh"] >= 0)
