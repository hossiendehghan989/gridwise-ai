# GridWise AI Roadmap

The roadmap is organized around making the research prototype more reproducible, realistic, and useful for energy-operation experiments.

## Now

- Keep the public benchmark reproducible.
- Improve documentation and onboarding.
- Make assumptions and limitations visible.
- Encourage focused contributions around tests, evaluation, and energy-domain examples.

## Next

- [ ] Add live or user-supplied tariff and carbon-intensity adapters.
- [ ] Add probabilistic forecasting with calibration and coverage monitoring.
- [ ] Add equipment-level constraints and operational feasibility checks.
- [ ] Add experiment artifacts and a small model-comparison report.
- [ ] Add a scheduled retraining example with drift thresholds.

## Later

- [ ] Explore mixed-integer scheduling for discrete equipment decisions.
- [ ] Add persistent storage for forecasts, schedules, and evaluation runs.
- [ ] Add production observability and deployment examples.
- [ ] Validate the workflow against an industrial or building-energy dataset.

## Design principles

1. **Forecasting must serve a decision.**
2. **Every result needs an evaluation context.**
3. **Constraints and assumptions are part of the model.**
4. **A reproducible research prototype should be honest about production gaps.**
