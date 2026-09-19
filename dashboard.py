from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.energy_optimizer import load_dataset, optimize_schedule, train_forecaster

st.set_page_config(page_title="GridWise AI", page_icon="⚡", layout="wide")
st.title("GridWise AI")
st.caption("Forecasting and optimization for smarter energy operations")

path = Path("data/energydata_complete.csv")
if not path.exists():
    st.error("Dataset not found. Run `python download_data.py` first.")
    st.stop()

df = load_dataset(path)
result = train_forecaster(df)
col1, col2, col3 = st.columns(3)
col1.metric("Forecast MAE", f"{result.metrics['mae_wh']:.0f} Wh")
col2.metric("Forecast RMSE", f"{result.metrics['rmse_wh']:.0f} Wh")
col3.metric("R²", f"{result.metrics['r2']:.2f}")

st.subheader("Out-of-sample demand forecast")
plot_df = pd.DataFrame({"actual": df.set_index("date")["Appliances"], "forecast": result.predictions}).dropna().tail(500)
st.line_chart(plot_df)

st.subheader("Flexible-load scheduling")
horizon = st.slider("Planning horizon (hours)", 6, 24, 12)
demand = result.predictions.tail(horizon).to_numpy()
plan = optimize_schedule(demand)
left, right = st.columns(2)
left.metric("Baseline peak", f"{plan['baseline_peak_wh']:.0f} Wh")
right.metric("Optimized peak", f"{plan['optimized_peak_wh']:.0f} Wh", delta=f"-{plan['baseline_peak_wh'] - plan['optimized_peak_wh']:.0f} Wh")
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(demand, label="Forecast demand", linewidth=2)
ax.plot(demand + plan["baseline"], label="Baseline total", linestyle="--")
ax.plot(demand + plan["optimized"], label="Optimized total", linewidth=2)
ax.set_xlabel("Planning hour")
ax.set_ylabel("Energy (Wh)")
ax.legend(ncol=3, frameon=False)
ax.grid(alpha=0.2)
st.pyplot(fig)
st.caption(f"Flexible energy shifted: {plan['energy_shifted_wh']:.0f} Wh")
