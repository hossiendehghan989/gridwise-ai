# Contributing to GridWise AI

Thank you for helping improve an open-source energy decision-support project.

## Good first contributions

- Improve README examples or documentation
- Add a focused unit test
- Add a benchmark or baseline with a clear evaluation protocol
- Improve dataset validation or error messages
- Add an energy-domain example with explicit assumptions
- Improve reproducibility, typing, or CI

## Before opening a pull request

```bash
python download_data.py
pytest -q
python validate.py
```

If your change touches research experiments, also run:

```bash
python research_experiment.py
```

## Pull request expectations

Please include:

1. **Problem:** what user or engineering problem does this solve?
2. **Approach:** what changed and why?
3. **Evidence:** tests, benchmark results, or screenshots where relevant.
4. **Limitations:** what remains illustrative or out of scope?

Keep pull requests focused. Avoid mixing refactors, unrelated formatting changes, and new features in one change.

## Reproducibility standards

- Preserve chronological evaluation for forecasting experiments.
- Do not introduce target leakage through current-period features.
- Report the baseline and the evaluation split with every new model result.
- Keep illustrative signals clearly labeled.
- Add or update tests for behavior changes.

## Code style

Prefer small, readable functions with explicit inputs and outputs. Use the existing project conventions and avoid adding dependencies unless they are necessary and documented in `requirements.txt`.

## Code of conduct

Be respectful, specific, and constructive. Technical disagreement is welcome; personal attacks are not.
