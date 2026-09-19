import numpy as np
import pandas as pd

from src.advanced import economic_summary, robust_schedule, validate_dataset


def test_data_quality_report_passes_for_clean_data():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="h"), "Appliances": [1, 2, 3]})
    report = validate_dataset(df)
    assert report.passed
    assert report.duplicate_timestamps == 0


def test_robust_schedule_preserves_budget_across_scenarios():
    scenarios = np.array([[100, 200, 150], [120, 240, 170], [90, 180, 140]], dtype=float)
    result = robust_schedule(scenarios, flexible_energy_wh=300, max_power_wh=150)
    assert np.isclose(result["energy_budget_wh"], 300)
    assert len(result["scenario_peaks"]) == 3


def test_economic_summary_returns_savings_fields():
    result = economic_summary(np.array([100, 200]), np.array([150, 150]), np.array([1.0, 2.0]))
    assert "total_estimated_saving" in result
    assert result["optimized_peak_charge"] < result["baseline_peak_charge"]
