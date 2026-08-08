# Contributing

Contributions should preserve the repository as a reproducible research
artifact. Prefer small, reviewable changes with clear experiment implications.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Checks

Run these before opening a pull request:

```bash
ruff check .
black --check .
pytest tests/ -m "not gpu"
```

GPU-dependent changes should include the exact experiment command, hardware
details, and output paths.

## Experiment Reporting

For new results, include:

- Git commit hash.
- Full command line.
- Dataset split and filters.
- Model identifier and precision.
- Hardware and CUDA details.
- Relevant output files, especially `aggregate.json`.
- Any deviation from the default protocol.

Do not commit raw datasets, model weights, or generated outputs unless their
licenses explicitly allow redistribution.
