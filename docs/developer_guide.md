# Developer Guide

## Project Structure
- `core/` – Processing engines and utilities.
- `cli/` – CLI entry point.
- `ui/` – PyQt/Tk GUIs.
- `backend/` – Scripts for automation (sample generation, benchmarks).
- `docs/` – Project documentation.
- `tests/` – Unit and integration tests.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests & Coverage
```bash
coverage run --source=core,cli,ui,backend -m unittest discover
coverage xml -o benchmarks/results/coverage.xml
```

## Benchmarks
```bash
python backend/run_benchmarks.py --videos-enabled
```
Outputs summary JSON and restored media under `benchmarks/results/`.

## Code Style
- Python 3.11.
- Keep functions documented with brief docstrings if complex.
- Use logging (`core.logger.setup_logging`) for runtime feedback.
- GUI: PyQt5 preferred, Tk fallback must remain functional.

## Configuration
- Default config: `config/config.yaml`.
- Use `config.load_config` to load and merge overrides.
- `ImageWatermarkRemover.from_config` / `VideoWatermarkRemover.from_config` pull defaults.

## Adding Tests
- Place new tests under `tests/`.
- CLI/UI tests rely on sample assets; ensure `backend/generate_samples.py` runs first.

## Packaging
PyInstaller spec to be created in Phase 10. When packaging, ensure:
- Data files (config, assets) referenced via relative paths.
- Entry points call `cli.main` or `ui.main_window.run_gui`.
