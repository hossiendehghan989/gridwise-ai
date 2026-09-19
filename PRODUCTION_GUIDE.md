# Production Guide

## Service surface

The repository now exposes a small FastAPI service:

- `GET /health` checks service and dataset availability.
- `GET /quality` returns the data-quality report.
- `POST /forecast` accepts `{ "horizon": 24 }` and returns a multi-horizon forecast with a 90% interval.

Run locally with:

```bash
uvicorn api:app --reload --port 8000
```

Build the container with:

```bash
docker compose up --build
```

## Data validation

`validate_dataset` checks required columns, missing cells, duplicate timestamps, negative targets, and timestamp ordering. These checks are deliberately small and explicit. A production pipeline should add schema versioning, range constraints for each sensor, freshness checks, and quarantine handling for invalid batches.

## Robust scheduling

`robust_schedule` chooses one schedule against multiple demand scenarios. The schedule is selected before the realized scenario is known, which is more conservative than optimizing separately for each forecast. The objective minimizes a worst-case peak auxiliary variable and a small mean-peak term. This is a transparent scenario-based approximation; distributionally robust optimization would require a defined uncertainty set and a stronger statistical justification.

## Explainability

`explain_model` computes permutation importance on the chronological holdout. Importance is predictive, not causal. A feature with high permutation importance is useful to the fitted model under the evaluation distribution; this does not prove that changing the feature would change demand.

## Economics

`economic_summary` estimates energy cost and demand-charge changes from a scenario. The calculation is intentionally a transparent approximation. It is not a tariff engine. Real deployment requires interval-specific tariffs, taxes, fixed charges, demand-charge windows, market settlement rules, and a verified unit convention.

## Operational safeguards

Before deployment, add authentication, rate limiting, structured logs, request IDs, model/version metadata, data drift monitoring, and a human approval step for schedules that control physical equipment. The current API is a research and demonstration service and must not be connected directly to actuators without these safeguards.
