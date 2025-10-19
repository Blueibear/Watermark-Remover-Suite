# Watermark Remover Suite

Watermark Remover Suite is a layered toolkit for removing watermarks from images and videos while preserving as much of the original content as possible. The project aims to provide both command-line and graphical interfaces, along with automation around packaging, signing, benchmarking, and release management.

## Project Goals
- Provide a reliable inpainting-based watermark removal engine for images and videos.
- Offer both CLI and PyQt-based GUI front ends.
- Support batch processing, configuration-driven workflows, and reproducible benchmarks.
- Package the application with PyInstaller and produce Windows installers ready for signing.
- Document validation, benchmarking, and release steps for auditing.

## Repository Layout
- `watermark_remover/` &mdash; New MVP package with CLI, pipeline, models, and QC helpers.
- `core/` &mdash; Legacy image and video processing engines, shared utilities.
- `cli/` &mdash; Legacy command-line entry points and argument parsing.
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

## CLI (MVP Stub)
The MVP CLI is exposed via the `wmr` entry point:

```powershell
wmr image input.jpg --out output.jpg --method telea --mask auto
wmr video input.mp4 --out output.mp4 --method telea --window 48 --overlap 12
```

Use `--method lama` after placing `lama.onnx` inside `~/.wmr/models/`. Stable Diffusion integration is reserved for future work.

## Packaging
Build the standalone executable with PyInstaller:

```powershell
pyinstaller backend\pyinstaller.spec --distpath dist --workpath build
dist\WatermarkRemoverSuite\WatermarkRemoverSuite.exe --help
```

Refer to `docs/packaging.md` for additional details.

## Signing
Mock signing workflow:

```powershell
python backend\sign_release.py --input dist\WatermarkRemoverSuite\WatermarkRemoverSuite.exe `
    --output dist\WatermarkRemoverSuite_signed.exe --log signature_verification.log
Get-FileHash dist\WatermarkRemoverSuite_signed.exe -Algorithm SHA256 >> signature_verification.log
```

See `docs/signing.md` for guidance on replacing this with real code signing.

## Release Publishing
Mock GitHub release automation:

```powershell
python backend\publish_release.py --repo Watermark-Remover-Suite `
    --tag v1.0.0 --title "Initial Stable Release" `
    --notes verification_reports\final_summary.md `
    --artifacts dist\WatermarkRemoverSuite_signed.exe installers\build\WatermarkRemoverSuite_Setup.exe `
    --output release_confirmation.json --log github_upload_log.txt
```

Refer to `docs/release_workflow.md` for details and future replacement with real publishing.

## Post-Release Verification
Mock integrity checks:

```powershell
python backend\verify_release.py `
    --artifacts dist\WatermarkRemoverSuite_signed.exe installers\build\WatermarkRemoverSuite_Setup.exe `
    --checksums installers\build\SHA256SUMS.txt `
    --log verification_reports\hash_verification.log
```

See `docs/verification_workflow.md` for guidance on extending these checks.

## Getting Started
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```
Optional extras:
- `pip install -e .[onx]` to enable the LaMa ONNX backend (requires `onnxruntime`).
- `pip install -e .[sd]` to prepare Stable Diffusion dependencies (integration pending).

The legacy `requirements.txt` remains for backwards compatibility with earlier automation scripts.

Prepare model cache directories with:

```powershell
python -m watermark_remover.models.download_models --all
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
