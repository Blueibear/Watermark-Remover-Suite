# Testing & Validation Strategy

## Test Suites
- **Unit Tests** – `python -m unittest discover`
  - Covers core processing, CLI, GUI initialization, sample generation, and benchmark harnesses.
- **Coverage Analysis** – `coverage run --source=core,cli,ui,backend -m unittest discover`
  - Reports stored to `benchmarks/results/coverage.xml` via `coverage xml`.

## Sample Data
- Generated via `python backend/generate_samples.py`.
- Stored under `sample_inputs/images` and `sample_inputs/videos`.

## Benchmark Harness
- `python backend/run_benchmarks.py` processes sample assets and records timings in `benchmarks/results/summary.json`.
- Invoke with `--videos-enabled` to include video benchmarks.

## GUI Checks
- PyQt5 UI smoke tests run in offscreen mode.
- Tkinter fallback test ensures environments without Qt still initialize.

## CI Recommendations
1. `python -m pip install -r requirements.txt`
2. `python backend/generate_samples.py`
3. `coverage run --source=core,cli,ui,backend -m unittest discover`
4. `coverage xml -o benchmarks/results/coverage.xml`
5. `python backend/run_benchmarks.py --videos-enabled --log-level INFO`
