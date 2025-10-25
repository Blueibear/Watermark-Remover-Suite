# Stable Diffusion Integration Guide

## Overview

The Watermark Remover Suite now supports Stable Diffusion-based inpainting for high-quality watermark removal. This method uses deep learning to intelligently reconstruct masked regions while maintaining photorealistic quality.

## Installation

Install the required dependencies:

```bash
pip install -e .[sd]
```

This installs:
- `diffusers[torch]>=0.30` — Hugging Face Diffusers library
- `transformers>=4.43` — Transformer models
- `accelerate>=0.33` — Training and inference optimizations
- `xformers>=0.0.27` — Memory-efficient attention (Linux only)

## Usage

### Command Line

```bash
# Image processing with SD
wmr image input.jpg --out output.jpg --method sd --mask auto

# Video processing with SD
wmr video input.mp4 --out output.mp4 --method sd --window 48 --overlap 12
```

### Model Download

On first use, the Stable Diffusion model will be automatically downloaded from Hugging Face Hub:
- Model: `stabilityai/stable-diffusion-2-inpainting`
- Size: ~5GB
- Location: Cached in `~/.cache/huggingface/`

## Configuration

The SD inpainting engine uses the following default parameters:

- **Model**: `stabilityai/stable-diffusion-2-inpainting`
- **Inference steps**: 50 (higher = better quality, slower)
- **Guidance scale**: 7.5 (controls how closely to follow the prompt)
- **Prompt**: "high quality, photorealistic, natural lighting, detailed"
- **Negative prompt**: "watermark, text, logo, lowres, blurry, distorted"

## Hardware Requirements

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (CUDA support)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for models

### Minimum
- **GPU**: NVIDIA GPU with 4GB+ VRAM (with memory optimizations)
- **RAM**: 8GB+ system RAM
- **CPU**: Works on CPU but ~10-50x slower

### Device Selection

The integration automatically selects the best available device:
1. CUDA (NVIDIA GPU) — preferred
2. MPS (Apple Silicon) — macOS only
3. CPU — fallback

## Performance Comparison

| Method | Speed | Quality | GPU Required | Model Size |
|--------|-------|---------|--------------|------------|
| telea  | Fast  | Basic   | No           | 0 MB       |
| lama   | Medium| Good    | Optional     | ~50 MB     |
| sd     | Slow  | Excellent| Recommended | ~5 GB      |

### Approximate Processing Times (1920x1080 image)

- **telea**: <1 second (CPU)
- **lama**: 2-5 seconds (GPU) / 10-30 seconds (CPU)
- **sd**: 10-30 seconds (GPU) / 3-10 minutes (CPU)

## Memory Optimizations

The SD integration includes several memory optimizations:

1. **xformers** — Memory-efficient attention (automatic on Linux with GPU)
2. **torch.float16** — Half-precision inference on GPU (automatic)
3. **Sequential CPU offload** — For low-memory systems (automatic on CPU)

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors on GPU:

1. Close other GPU-intensive applications
2. Reduce image resolution before processing
3. Try using the `lama` method instead

### Slow Performance

- Ensure you have CUDA installed for GPU acceleration
- Check GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`
- First run will be slower due to model download

### Model Download Issues

If model download fails:
- Check internet connection
- Verify access to huggingface.co
- Try manual download: `huggingface-cli download stabilityai/stable-diffusion-2-inpainting`

## Advanced Usage

### Custom Model

To use a different Stable Diffusion inpainting model, modify the `_SD_MODEL_ID` in `watermark_remover/core/pipeline.py`:

```python
_SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"  # SD 1.5 based
```

### Adjusting Quality/Speed

Edit the parameters in `pipeline.py` `_inpaint()` function:

```python
return sd_run(
    img,
    mask,
    model_id=_SD_MODEL_ID,
    device="auto",
    seed=seed,
    num_inference_steps=30,  # Reduce for faster processing
    guidance_scale=7.5,
)
```

## Implementation Details

The SD integration follows the same modular architecture as LaMa:

- **Module**: `watermark_remover/core/inpaint_sd.py`
- **Main function**: `inpaint_sd(img_bgr, mask_u8, ...)`
- **Caching**: Model is loaded once and reused across frames
- **Deterministic**: Supports seeding for reproducible results

### Key Features

1. **PIL/OpenCV Compatibility** — Automatic conversion between formats
2. **Mask Handling** — Supports both binary and grayscale masks
3. **Device Agnostic** — Automatic device selection
4. **Memory Efficient** — Includes optimization flags
5. **Video Support** — Per-frame seeding for temporal consistency

## Comparison with LaMa

| Feature | LaMa ONNX | Stable Diffusion |
|---------|-----------|------------------|
| Quality | Good for textures | Excellent for all |
| Speed   | Fast | Slower |
| Memory  | Low (~2GB) | High (~6GB) |
| Setup   | Manual model download | Automatic |
| Customization | Limited | High (prompts, models) |

## Best Practices

1. **Use SD for final output** — When quality is critical
2. **Use LaMa for iteration** — When speed matters
3. **Start with telea** — For quick previews
4. **Process in batches** — SD model loads once, processes multiple images
5. **Use consistent seeds** — For reproducible results

## References

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Stable Diffusion 2.0 Inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
- [Model Card](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/README.md)
