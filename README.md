# Watermark Remover Suite

Watermark Remover Suite is a layered toolkit for removing watermarks from images and videos while preserving as much of the original content as possible. The project aims to provide both command-line and graphical interfaces, along with automation around packaging, signing, benchmarking, and release management.

## Project Goals
- Provide a reliable inpainting-based watermark removal engine for images and videos.
- Offer both CLI and PyQt-based GUI front ends.
- Support batch processing, configuration-driven workflows, and reproducible benchmarks.
- Package the application with PyInstaller and produce Windows installers ready for signing.
- Document validation, benchmarking, and release steps for auditing.

## Repository Layout
- `core/` &mdash; Image and video processing engines, shared utilities.
- `cli/` &mdash; Command-line entry points and argument parsing.
- `ui/` &mdash; GUI application code and related assets.
- `config/` &mdash; YAML configuration files and loader utilities.
- `backend/` &mdash; Automation helpers, packaging, and integration scripts.
- `assets/` &mdash; Icons and other static resources for the GUI.
- `benchmarks/` &mdash; Benchmark datasets, harnesses, and result summaries.
- `docs/` &mdash; Documentation, design notes, and verification reports.
- `installers/` &mdash; Inno Setup scripts and packaged installers.
- `sample_inputs/` &mdash; Sample media used for offline testing.
- `tests/` &mdash; Automated test suites and fixtures.

## Sample Assets
Synthetic demo files live under `sample_inputs/`. Regenerate them any time with:

```powershell
python backend/generate_samples.py
```

More details are available in `docs/simulation_assets.md`.

## Testing & Benchmarks
Run the automated test suite with coverage:

```powershell
coverage run --source=core,cli,ui,backend -m unittest discover
coverage xml -o benchmarks/results/coverage.xml
```

Generate benchmark timings on the sample assets:

```powershell
python backend/run_benchmarks.py --videos-enabled --log-level INFO
```

See `docs/testing_strategy.md` for the full validation workflow.

## Packaging
Build the standalone executable with PyInstaller:

```powershell
pyinstaller backend\pyinstaller.spec --distpath dist --workpath build
dist\WatermarkRemoverSuite\WatermarkRemoverSuite.exe --help
```

Refer to `docs/packaging.md` for additional details.

## Getting Started
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Development Workflow
1. Implement or update features within the appropriate module directory.
2. Add or update tests inside `tests/` and run them inside the virtual environment.
3. Update documentation in `docs/` and `README.md` when user-facing behaviour changes.
4. Build distributables with PyInstaller once functionality is validated.
5. Track changes through Git commits and ensure automation scripts remain reproducible.

## License
See the [LICENSE](LICENSE) file for licensing details.

## Status
This repository is currently in the early scaffolding stage (Phase 0). Subsequent phases will implement the core processing engine, interfaces, automation, and release tooling.
