# GridWise AI: Technical Research Report

## Abstract

GridWise AI studies a two-stage decision problem in which short-horizon electricity demand forecasts are converted into a feasible flexible-load schedule. The central methodological question is not whether a machine-learning model can reduce prediction error in isolation. It is whether a forecasting-and-optimization pipeline can provide a stable, interpretable, and constraint-respecting basis for operational decisions.

The system uses a public household-energy dataset as a reproducible proxy for industrial telemetry. It evaluates models with chronological holdouts and expanding-window validation, quantifies forecast uncertainty through quantile regression, studies feature contributions through ablation, and formulates scheduling as a linear program. The resulting analysis makes explicit where the system performs well, where simple baselines remain competitive, and which assumptions prevent a production interpretation.

## 1. Research question and contribution

The project addresses the following research question:

> Under realistic temporal validation, can a compact forecasting model and a constrained scheduling formulation produce operationally useful recommendations without violating the energy budget or per-period power limit?

The contribution is an integrated, reproducible pipeline with four properties. First, the evaluation protocol respects temporal direction. Second, the model is compared with strong naive baselines rather than judged against an arbitrary benchmark. Third, the forecast is represented as an interval rather than only a point estimate. Fourth, the optimization layer exposes the trade-off between peak, cost, and carbon objectives.

This is a methodological portfolio study, not a claim of deployment readiness or causal inference.

## 2. Data-generating context

The data source is the UCI Appliances Energy Prediction dataset [1]. It contains appliance energy consumption, lighting consumption, indoor temperature and humidity, and outdoor weather measurements sampled at ten-minute intervals. The target is `Appliances`, measured in watt-hours.

The dataset is appropriate for reproducible development because it contains exogenous environmental variables and a temporal target. It is not representative of an industrial plant by default. A factory introduces additional structure, including production schedules, equipment states, maintenance events, tariff contracts, and non-stationary operating regimes. Therefore, any industrial claim must be treated as a transfer hypothesis rather than as a result established by this dataset.

## 3. Forecasting formulation

Let \(y_t\) denote appliance demand at time \(t\), \(x_t\) denote contemporaneous measured covariates, and \(h_t\) denote historical information available before time \(t\). The forecasting task is:

\[
\hat{y}_{t+1} = f(x_t, h_t; \theta),
\]

where \(f\) is a gradient-boosting regression function and \(\theta\) is learned only from observations earlier than the evaluation period.

The feature vector includes calendar encodings, environmental measurements, one-period and 24-period lags, and a rolling mean calculated after shifting the target. The shift is important. Without it, the current target could enter the rolling statistic and produce optimistic but invalid estimates.

## 4. Evaluation protocol

The primary split uses the first 80% of valid chronological observations for training and the final 20% for testing. The project also implements expanding-window walk-forward evaluation. In fold \(k\), the model is trained on all observations before the fold's test interval and evaluated only on that future interval:

\[
\mathcal{D}_{train}^{(k)} = \{1, \ldots, t_k\}, \qquad
\mathcal{D}_{test}^{(k)} = \{t_k+1, \ldots, t_{k+1}\}.
\]

This design tests whether performance is stable across time rather than dependent on one arbitrary cutoff. The reported metrics are mean absolute error, root mean squared error, coefficient of determination, and mean absolute percentage error. Each metric answers a different question; no metric is treated as a sufficient description of forecast quality.

## 5. Benchmark interpretation

The point-holdout benchmark currently shows that the previous-reading baseline has the best MAE, while Gradient Boosting has the best RMSE and R². This result is scientifically useful rather than embarrassing. It indicates that the supervised model improves the treatment of larger deviations but does not dominate a highly local persistence baseline on every loss function.

A responsible deployment decision would therefore depend on the operational loss. If every watt-hour of average absolute error has equal value, persistence may be preferred. If large misses create disproportionate peak or procurement risk, the lower RMSE of Gradient Boosting may be more relevant. The choice should be made with a domain-specific cost function, not by reporting the most favorable metric.

## 6. Uncertainty quantification

The research extension fits lower, median, and upper quantile Gradient Boosting models. For a nominal 80% interval, the lower and upper models target the 0.10 and 0.90 conditional quantiles:

\[
\hat{q}_{0.10}(x_t) \leq y_{t+1} \leq \hat{q}_{0.90}(x_t).
\]

The interval is evaluated with empirical coverage and mean width. Coverage measures the fraction of observations contained by the interval. Width measures sharpness, with narrower intervals preferred only when coverage remains adequate. These two quantities must be interpreted together because an interval can achieve high coverage by being uninformatively wide.

The implementation is a useful baseline for uncertainty estimation, not a calibrated probabilistic forecast. A production study should add conformal calibration, conditional coverage diagnostics, and separate evaluation across seasons and operating regimes.

## 7. Feature ablation

The ablation experiment compares the full feature set with versions that remove historical lags, remove weather variables, or retain only calendar variables. The purpose is not to claim causality. Removing a feature group changes the information set and may also change the model's approximation problem. The result is therefore interpreted as predictive contribution under the selected model class.

A strong finding would be stable degradation when a meaningful information group is removed. An unstable result would suggest redundancy, non-stationarity, or insufficient sample size. The experiment is stored as a CSV artifact so that the conclusion can be inspected rather than inferred from code alone.

## 8. Scheduling model

Let \(u_t\) be flexible load assigned to period \(t\), \(d_t\) be predicted inflexible demand, and \(z\) be the resulting peak. The linear program is:

\[
\min_{u,z} \quad z + \lambda_p \sum_t p_t u_t + \lambda_c \sum_t c_t u_t
\]

subject to:

\[
0 \leq u_t \leq \bar{u}_t,
\]

\[
\sum_t u_t = E_{flex},
\]

\[
d_t + u_t \leq z \quad \forall t.
\]

Here, \(p_t\) is an electricity-price signal, \(c_t\) is a carbon-intensity signal, \(E_{flex}\) is the required flexible-energy budget, and \(\lambda_p\) and \(\lambda_c\) control the trade-off. The formulation is deliberately linear so that feasibility and optimality are transparent.

The dashboard currently uses illustrative price and carbon vectors. They are not market observations. Replacing them with a real tariff, grid-emissions signal, or industrial operating cost is a necessary step before drawing economic conclusions.

## 9. Sensitivity analysis

A single optimized schedule can hide objective sensitivity. The research runner evaluates multiple price and carbon weights. The resulting table shows whether peak reduction is robust or whether it is purchased by shifting load into expensive or carbon-intensive periods.

The current experiment produced a stable 100 Wh peak reduction across the tested weight settings. The amount of shifted energy varied from approximately 542 Wh to 571 Wh, which shows that the objective weights changed the allocation pattern more than the final peak in this small scenario.

## 10. Observed experimental findings

The four expanding-window folds produced RMSE values of 76.8 Wh, 67.6 Wh, 56.8 Wh, and 66.3 Wh. The improvement from the first to the third fold suggests that the model benefits from additional history, although the final fold demonstrates that performance is not monotonic and may vary with the operating regime.

The ablation result is especially informative. Removing historical lags increased RMSE from 62.0 Wh to 143.2 Wh, while retaining only calendar variables produced 82.7 Wh RMSE. Surprisingly, removing weather variables reduced RMSE to 61.0 Wh on this holdout. This should not be interpreted as evidence that weather is irrelevant in general. It suggests that the available weather features may be redundant, noisy, or misaligned with the target at the selected resolution. The result justifies testing feature availability, lag alignment, and regime-specific behavior before expanding the feature set.

The nominal 80% quantile interval achieved 80.8% empirical coverage with a mean width of 87.4 Wh. This is close to the nominal target on the current holdout, but it is not sufficient evidence of calibrated conditional coverage. Segment-level coverage and conformal calibration remain necessary.

For a serious operational study, sensitivity should be extended to uncertainty in demand, power caps, flexible-energy requirements, tariff structures, and equipment availability. Scenario-based or robust optimization would then be more appropriate than a deterministic point forecast.

## 11. Reproducibility

Run the complete experiment with:

```bash
python download_data.py
pytest -q
python research_experiment.py
```

The experiment writes the following artifacts:

- `artifacts/walk_forward.csv`
- `artifacts/feature_ablation.csv`
- `artifacts/prediction_intervals.csv`
- `artifacts/scheduling_sensitivity.csv`

The repository excludes the downloaded dataset and generated artifacts from version control by default. This keeps the repository lightweight while preserving a deterministic route to regeneration.

## 12. Threats to validity

**External validity.** Household data cannot establish performance in factories or energy markets. Transfer requires new data and revalidation.

**Temporal validity.** A chronological split reduces leakage but does not guarantee stability under structural change. Walk-forward variation should be inspected before deployment.

**Metric validity.** Aggregate error metrics do not measure peak risk, economic loss, comfort, equipment wear, or constraint violations. A production scorecard must include those outcomes.

**Uncertainty validity.** Quantile regression is not automatically calibrated. Coverage must be evaluated by time segment and operating regime.

**Optimization validity.** The schedule is only as meaningful as its demand forecast, flexible-load assumption, and price/carbon signals. A mathematically optimal schedule can still be operationally irrelevant if the model omits a real constraint.

## 13. Future research

The next scientifically meaningful extensions are conformal prediction for finite-sample coverage, probabilistic multi-step forecasting, hierarchical models for multiple buildings, mixed-integer equipment constraints, robust optimization under forecast uncertainty, and online monitoring for concept drift. These extensions should be evaluated with pre-registered metrics and scenario definitions rather than added only to increase model complexity.

## References

[1]: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction "UCI Appliances Energy Prediction dataset"
