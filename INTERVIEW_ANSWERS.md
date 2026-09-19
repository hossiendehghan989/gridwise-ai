# Interview Answer Matrix

This document maps the project implementation to the questions a technical reviewer is likely to ask.

## Why can a naive baseline beat a machine-learning model on MAE?

The benchmark measures the previous-reading baseline alongside Gradient Boosting on the same chronological holdout. The baseline reaches lower MAE because appliance demand has strong short-term persistence. Gradient Boosting reaches lower RMSE and higher R², which indicates better treatment of larger deviations and more explained variance. The project therefore treats model choice as an operational loss-design problem rather than assuming that a complex model must win every metric.

## Why did removing weather variables improve the result?

The ablation runner measures feature groups explicitly. Removing weather variables slightly improves the current holdout, while removing historical lags causes a large degradation. The defensible interpretation is that the available weather measurements are redundant, noisy, or misaligned at this sampling resolution. The result is dataset-specific and is not treated as evidence that weather is irrelevant in general.

## How is leakage prevented?

The split is chronological. Lag features are shifted before they are used, and rolling statistics are computed from prior observations. All models use the same future holdout. The data-quality layer also checks timestamp ordering and required fields.

## Why use walk-forward validation?

The expanding-window evaluation simulates repeated retraining and future prediction. It exposes temporal instability that a single holdout can hide and produces fold-level errors that can be inspected by time regime.

## Are the intervals calibrated?

The project includes both quantile intervals and split-conformal intervals. Conformal calibration uses a separate chronological calibration segment to estimate a residual radius and evaluates coverage on a later test segment. This supports a marginal finite-sample coverage claim under exchangeability assumptions. It does not guarantee conditional coverage for every hour, season, or operating regime; the report states this limitation explicitly.

## How does robust optimization handle forecast error?

The robust scheduler receives multiple demand scenarios and selects one shared schedule before the realized scenario is known. It minimizes a worst-case peak auxiliary variable while preserving the flexible-energy budget and power cap. Scenario quality is part of the uncertainty model, so production scenarios must come from calibrated forecasts rather than arbitrary perturbations.

## What changes with industrial data?

The forecast features, operational constraints, tariff model, and validation design must be rebuilt around the process. Industrial extensions include equipment states, production plans, maintenance, ramp limits, minimum up/down times, battery state, and process-quality constraints. The household experiment is a reproducible proxy, not evidence of industrial transfer without revalidation.

## How are cost and ROI calculated?

The economic module computes energy cost and peak charges before and after scheduling. `annualized_roi` then returns net annual saving, first-year ROI, and simple payback from implementation and operating costs. The calculation remains a transparent scenario estimator; real deployment requires verified tariff rules and a complete cost ledger.

## How is the API secured?

The service validates request schemas, exposes a health endpoint, supports a configurable `GRIDWISE_API_KEY`, and has a smoke test. When the key is configured, forecast and quality routes require the `X-API-Key` header. Public deployment still requires TLS, rate limiting, structured logs, secret management, dependency scanning, and authorization appropriate to the environment.

## When is the model retrained?

`retraining_decision` implements a deterministic policy. Retraining review is triggered when current MAE exceeds a configurable multiple of baseline MAE or when a drift score exceeds its threshold. A production workflow would run this policy on scheduled batches, register a challenger model, compare it with the champion, require approval, and support rollback.
