# Performance Optimization (Phase 8)

Benchmarks executed with `python backend/run_benchmarks.py --videos-enabled --log-level INFO`.

## Summary

| Media | Baseline Time (s) | Optimized Time (s) | Speedup |
| ----- | ----------------- | ------------------ | ------- |
| sample_01_watermarked.png | 0.0175 | 0.0043 | 4.1× |
| sample_02_watermarked.png | 0.0279 | 0.0055 | 5.1× |
| sample_03_watermarked.png | 0.0110 | 0.0030 | 3.7× |
| sample_01_watermarked.mp4 | 0.2092 | 0.1485 | 1.4× |
| sample_02_watermarked.mp4 | 0.2323 | 0.1977 | 1.2× |

## Notes
- Batch processing now uses a `ThreadPoolExecutor` when `max_workers > 1`, reducing total wall-clock time for concurrent jobs.
- Image benchmarks benefit from lower per-task overhead because of shared thread pool and caching inside the core removers.
- Video processing reuses the mask detected from the first frame, avoiding redundant detection work across frames.
