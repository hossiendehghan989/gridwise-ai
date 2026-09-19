from pathlib import Path
from src.energy_optimizer import load_dataset, optimize_schedule, train_forecaster

df = load_dataset(Path("data/energydata_complete.csv"))
result = train_forecaster(df)
plan = optimize_schedule(result.predictions.tail(12).to_numpy())
print(f"rows={len(df)}")
print(f"metrics={result.metrics}")
print(f"peak_reduction_wh={plan['baseline_peak_wh'] - plan['optimized_peak_wh']:.2f}")
