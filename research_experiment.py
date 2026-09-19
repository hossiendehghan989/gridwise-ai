from pathlib import Path

from src.energy_optimizer import load_dataset, train_forecaster
from src.research import feature_ablation, quantile_prediction_interval, scheduling_sensitivity, walk_forward_evaluation

root = Path(__file__).parent
artifacts = root / "artifacts"
artifacts.mkdir(exist_ok=True)
df = load_dataset(root / "data/energydata_complete.csv")
walk_forward_evaluation(df).to_csv(artifacts / "walk_forward.csv", index=False)
feature_ablation(df).to_csv(artifacts / "feature_ablation.csv", index=False)
interval = quantile_prediction_interval(df)
interval.predictions.to_csv(artifacts / "prediction_intervals.csv")
scheduling_sensitivity(train_forecaster(df).predictions.tail(12).to_numpy()).to_csv(artifacts / "scheduling_sensitivity.csv", index=False)
print(f"coverage={interval.coverage:.4f}")
print(f"mean_interval_width_wh={interval.mean_width_wh:.2f}")
print("saved artifacts to", artifacts)
