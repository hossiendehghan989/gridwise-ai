from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.energy_optimizer import benchmark_models, load_dataset, optimize_schedule

st.set_page_config(page_title="GridWise AI | Energy Intelligence", page_icon="⚡", layout="wide")
st.title("GridWise AI")
st.caption("Forecasting, benchmarking, and constrained scheduling for smarter energy operations")

path = Path("data/energydata_complete.csv")
if not path.exists():
    st.error("Dataset not found. Run `python download_data.py` first.")
    st.stop()

with st.spinner("Training models on the chronological holdout..."):
    df = load_dataset(path)
    benchmark, result = benchmark_models(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best model", benchmark.iloc[0]["model"])
m2.metric("MAE", f"{result.metrics['mae_wh']:.1f} Wh")
m3.metric("RMSE", f"{result.metrics['rmse_wh']:.1f} Wh")
m4.metric("R²", f"{result.metrics['r2']:.2f}")

st.subheader("Model benchmark")
st.dataframe(
    benchmark.style.format({"mae_wh": "{:.1f}", "rmse_wh": "{:.1f}", "r2": "{:.3f}", "mape_pct": "{:.1f}%"}),
    use_container_width=True,
    hide_index=True,
)
st.caption("All models are evaluated on the same chronological out-of-sample holdout.")

st.subheader("Out-of-sample demand forecast")
plot_df = pd.DataFrame({"actual": result.actual, "forecast": result.predictions}).tail(500)
st.line_chart(plot_df)

st.subheader("Constrained flexible-load scheduling")
c1, c2, c3 = st.columns(3)
horizon = c1.slider("Planning horizon", 6, 24, 12)
flexible_energy = c2.slider("Flexible energy (Wh)", 300, 3000, 1200, step=100)
max_power = c3.slider("Per-period power cap (Wh)", 100, 600, 300, step=50)
demand = result.predictions.tail(horizon).to_numpy()
price = pd.Series([0.8, 0.7, 0.6, 0.5, 0.45, 0.5, 0.65, 0.9, 1.1, 1.2, 1.0, 0.85, 0.75, 0.65, 0.55, 0.5, 0.6, 0.8, 1.0, 1.15, 1.2, 1.0, 0.9, 0.8]).head(horizon).to_numpy()
carbon = pd.Series([0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.85, 0.75, 0.65, 0.55, 0.5, 0.45, 0.5, 0.6, 0.7]).head(horizon).to_numpy()
plan = optimize_schedule(demand, flexible_energy, max_power, price, carbon)

p1, p2, p3 = st.columns(3)
p1.metric("Peak reduction", f"{plan['peak_reduction_wh']:.0f} Wh")
p2.metric("Energy shifted", f"{plan['energy_shifted_wh']:.0f} Wh")
p3.metric("Optimized peak", f"{plan['optimized_peak_wh']:.0f} Wh")

fig, ax = plt.subplots(figsize=(11, 4))
periods = range(1, horizon + 1)
ax.plot(periods, demand, label="Forecast demand", linewidth=2, color="#334155")
ax.plot(periods, plan["baseline_total"], label="Baseline total", linestyle="--", color="#94a3b8")
ax.plot(periods, plan["optimized_total"], label="Optimized total", linewidth=2.5, color="#0f766e")
ax.set_xlabel("Planning period")
ax.set_ylabel("Energy (Wh)")
ax.set_xticks(list(periods))
ax.grid(alpha=0.2)
ax.legend(ncol=3, frameon=False)
st.pyplot(fig)
st.caption("The optimizer preserves the flexible-energy budget while considering peak demand, illustrative price, and carbon signals.")
