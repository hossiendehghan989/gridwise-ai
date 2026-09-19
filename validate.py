from pathlib import Path

import numpy as np

from src.energy_optimizer import benchmark_models, load_dataset, optimize_schedule


df = load_dataset(Path("data/energydata_complete.csv"))
benchmark, result = benchmark_models(df)
forecast = result.predictions.tail(12).to_numpy()
price = np.linspace(0.45, 1.2, 12)
carbon = np.linspace(0.9, 0.45, 12)
plan = optimize_schedule(forecast, flexible_energy_wh=1200, max_power_wh=300, price=price, carbon_intensity=carbon)
print(f"rows={len(df)}")
print("benchmark:")
print(benchmark.to_string(index=False))
print(f"primary_metrics={result.metrics}")
print(f"peak_reduction_wh={plan['peak_reduction_wh']:.2f}")
print(f"energy_budget_error_wh={abs(plan['optimized'].sum() - 1200):.8f}")
