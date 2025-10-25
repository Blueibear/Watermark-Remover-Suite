# Watermark Remover Suite

Watermark Remover Suite is a layered toolkit for removing watermarks from images and videos while preserving as much of the original content as possible. The project aims to provide both command-line and graphical interfaces, along with automation around packaging, signing, benchmarking, and release management.

---

## 📌 Project Goals

- Provide a reliable inpainting-based watermark removal engine for images and videos.
- Offer both CLI and PyQt-based GUI front ends.
- Support batch processing, configuration-driven workflows, and reproducible benchmarks.
- Package the application with PyInstaller and produce Windows installers ready for signing.
- Document validation, benchmarking, and release steps for auditing.

---

## 📁 Repository Layout

```
watermark_remover/   → MVP package with CLI, pipeline, models, and QC helpers
core/                → Legacy image/video engines and utilities
cli/                 → Legacy command-line interface
ui/                  → GUI application and assets
config/              → YAML config files
backend/             → Automation scripts (build, sign, release, verify)
assets/              → GUI icons and static resources
benchmarks/          → Benchmark harnesses and datasets
docs/                → Project documentation and technical notes
installers/          → Inno Setup scripts and packaged installers
sample_inputs/       → Example input media
tests/               → Unit tests and regression fixtures
```

---

## 🧪 Sample Assets

Synthetic demo files live under `sample_inputs/`.  
Regenerate them any time with:

```bash
python backend/generate_samples.py
```

More details: [`docs/simulation_assets.md`](docs/simulation_assets.md)

---

## ⚙️ CLI (MVP Stub)

The MVP CLI is exposed via the `wmr` entry point:

```bash
wmr image input.jpg --out output.jpg --method telea --mask auto
wmr video input.mp4 --out output.mp4 --method telea --window 48 --overlap 12
```

**Available methods:**
- `telea` — Fast OpenCV-based inpainting (default, CPU-only)
- `lama` — LaMa ONNX model (requires `lama.onnx` in `~/.wmr/models/` and `pip install -e .[onx]`)
- `sd` — Stable Diffusion inpainting (requires `pip install -e .[sd]`, downloads model on first use)
- `noop` — No-op pass-through (for testing)

---

## 📦 Packaging

Build the standalone executable with PyInstaller:

```bash
pyinstaller backend\pyinstaller.spec --distpath dist --workpath build
dist\WatermarkRemoverSuite\WatermarkRemoverSuite.exe --help
```

More details: [`docs/packaging.md`](docs/packaging.md)

---

## 🔏 Signing

Digitally sign the main executable using a self-signed or real code signing certificate:

```powershell
python backend\sign_release.py `
  --input dist\WatermarkRemoverSuite\WatermarkRemoverSuite.exe `
  --output dist\WatermarkRemoverSuite_signed.exe `
  --log signature_verification.log
```

Environment variables required:
- `CODESIGN_PFX_PATH` — path to your `.pfx` certificate
- `CODESIGN_PFX_PASSWORD` — password for the certificate

Hash output is saved in `installers\build\SHA256SUMS.txt`.

More details: [`docs/signing.md`](docs/signing.md)

---

## 🚀 Release Publishing

Release artifacts are published to GitHub automatically using the Releases API:

```powershell
python backend\publish_release.py `
  --tag v0.2.1 `
  --title "October Stable Release" `
  --notes verification_reports\final_summary.md
```

This publishes:
- `WatermarkRemoverSuite_signed.exe`
- `WatermarkRemoverSuite_Setup.exe`

...to the release for the specified tag.

Environment variable required:
- `GITHUB_TOKEN` — GitHub personal access token with `repo` scope

Logs:
- `release_confirmation.json`
- `github_upload_log.txt`

More details: [`docs/release_workflow.md`](docs/release_workflow.md)

---

## 🛡️ Post-Release Verification

Verify public release assets directly from GitHub:

```powershell
python backend\verify_release.py
```

This:
- Downloads assets from the GitHub release
- Recomputes their SHA256 hashes
- Compares them against `installers/build/SHA256SUMS.txt`
- Fails if any mismatch or corruption is detected

More details: [`docs/verification_workflow.md`](docs/verification_workflow.md)

---

## 🚀 Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Optional extras:

```bash
pip install -e .[onx]   # LaMa ONNX backend (requires onnxruntime)
pip install -e .[sd]    # Stable Diffusion inpainting (diffusers, transformers, accelerate)
pip install -e .[gui]   # PySide6 alternative GUI
pip install -e .[develop]  # Development tools (pytest, black, ruff, mypy)
```

Legacy `requirements.txt` available for early-phase automation compatibility.

Prepare model cache directories:

```bash
python -m watermark_remover.models.download_models --all
```

---

## 🔧 Development Workflow

- Update or add features within the correct module
- Add/modify tests under `tests/`
- Keep documentation and README.md current with behavior
- Use PyInstaller to build production-ready apps
- Ensure `sign_release.py`, `publish_release.py`, and `verify_release.py` remain reproducible

---

## ⚖️ License

See [LICENSE](LICENSE) for full licensing information.

---

## 📈 Status

This repository has moved beyond scaffolding (Phase 0) and now includes:
- Working MVP CLI/GUI
- Real code signing and hashing
- Verified GitHub publishing automation
- Binary integrity verification tooling

Next phases will continue expanding model integration, UX polish, and CI pipelines.