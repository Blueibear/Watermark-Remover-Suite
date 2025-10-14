# Watermark Remover Suite – User Guide

## Overview
Watermark Remover Suite provides CLI and GUI workflows for removing watermarks from images and videos.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Command-Line Interface
Run `python -m cli.main --help` or `watermark-remover --help` (after packaging).

### Basic commands
```bash
python -m cli.main image -i sample_inputs/images/sample_01_watermarked.png -o out.png
python -m cli.main video -i sample_inputs/videos/sample_01_watermarked.mp4 -o out.mp4
python -m cli.main batch -m manifests/sample_batch.yaml
```
- `--config` overrides the default config file.
- `--log-level DEBUG` and `--log-file` adjust logging.

### Batch Manifest Format
```yaml
- type: image
  input: sample_inputs/images/sample_01_watermarked.png
  output: outputs/sample_01_restored.png
  mask: sample_inputs/images/sample_01_mask.png
- type: video
  input: sample_inputs/videos/sample_01_watermarked.mp4
  output: outputs/sample_01_restored.mp4
  auto_mask:
    threshold: 240
```

## GUI
Launch via:
```bash
python -c "from ui.main_window import run_gui; run_gui()"
```
Fallback Tk GUI:
```bash
python -c "from ui.fallback import run_fallback_gui; run_fallback_gui()"
```
- Image tab: select input image, optional mask, choose output, click “Process Image”.
- Video tab: similar flow for videos.
- Preview panel shows before/after snapshots when processing images.

## Sample Assets
Generate synthetic assets for quick testing:
```bash
python backend/generate_samples.py
```

## Benchmarks
```bash
python backend/run_benchmarks.py --videos-enabled
```
Results stored under `benchmarks/results/summary.json`.

## Logging
Configured via `config/config.yaml`. CLI allows overrides using `--log-level` and `--log-file`.
