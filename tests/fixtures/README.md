# Test Fixtures

This directory contains minimal test fixtures for unit tests:

- `tiny.png`: 64×64 PNG image with simple gradient pattern
- `tiny.mp4`: 1-2 second, 320×240 silent MP4 video (to be generated)

These fixtures are used for fast CPU-only unit tests that don't require GPU or heavy compute.

## Generating tiny.mp4

To generate the video fixture, run:

```bash
ffmpeg -f lavfi -i testsrc=size=320x240:duration=2:rate=15 -pix_fmt yuv420p tests/fixtures/tiny.mp4
```
