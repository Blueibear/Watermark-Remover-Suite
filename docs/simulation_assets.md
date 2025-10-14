# Simulation & Offline Assets

Phase 6 produces synthetic media to validate the Watermark Remover Suite without relying on third-party datasets.

## Generation Script
- Entry point: `backend/generate_samples.py`
- Dependencies: NumPy, OpenCV, MoviePy
- Outputs:
  - `sample_inputs/images/sample_XX_base.png`
  - `sample_inputs/images/sample_XX_watermarked.png`
  - `sample_inputs/images/sample_XX_mask.png`
  - `sample_inputs/videos/sample_XX_watermarked.mp4`

Run with defaults:

```bash
python backend/generate_samples.py
```

### Options
- `--image-count N` – number of images (default 3)
- `--video-count N` – number of videos (default 2)
- `--videos/--no-videos` – toggle video creation
- `--images-dir PATH` / `--videos-dir PATH` – override destination folders

## Usage
- Sample images are used in automated tests and CLI demonstrations.
- Videos are short (approx. 3 seconds) RGB clips with deterministic watermark placement for reproducible benchmarking.

## Regeneration Checklist
1. Delete old assets if large changes are required.
2. Run `python backend/generate_samples.py`.
3. Re-run tests: `python -m unittest tests.test_sample_generation`.
