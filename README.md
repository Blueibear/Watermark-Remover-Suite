# Watermark Remover Suite

A powerful toolkit for removing watermarks from images and videos using state-of-the-art inpainting methods. Features multiple backends including OpenCV, LaMa ONNX, and Stable Diffusion, with optional temporal guidance for seamless video processing.

[![CI](https://github.com/Blueibear/Watermark-Remover-Suite/workflows/CI/badge.svg)](https://github.com/Blueibear/Watermark-Remover-Suite/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚖️ Legal & Ethical Disclaimer

**IMPORTANT: This tool is provided for legitimate purposes only.**

✅ **Appropriate Uses:**
- Removing watermarks from **your own content**
- Processing **licensed content** with explicit permission
- Research, education, and academic purposes
- Restoring personal photos or videos

❌ **Prohibited Uses:**
- Removing watermarks to violate copyright or intellectual property rights
- Processing content without proper authorization
- Commercial use of others' copyrighted material
- Any use that violates local laws or regulations

**By using this software, you agree to:**
1. Comply with all applicable copyright and intellectual property laws
2. Only process content you own or have explicit permission to modify
3. Take full responsibility for your use of this tool
4. Respect the rights of content creators and copyright holders

The developers of this software assume **no liability** for misuse or legal violations.

---

## Features

- **🖼️ Image Watermark Removal**: Remove watermarks from JPG, PNG, and other image formats
- **🎥 Video Watermark Removal**: Process videos with temporal consistency and audio preservation
- **🤖 Multiple Inpainting Methods**:
  - **Telea** (OpenCV, fast CPU baseline)
  - **LaMa ONNX** (high-quality, GPU-accelerated)
  - **Stable Diffusion** (cutting-edge, requires GPU)
- **🎯 Automatic Mask Detection**: Smart watermark region detection
- **🌊 Optical Flow Guidance**: Temporal blending for seamless video transitions
- **✅ Quality Control**: Automated QC with per-frame validation and retry logic
- **⚡ GPU Acceleration**: CUDA support for LaMa and Stable Diffusion backends
- **📦 Batch Processing**: Process multiple files efficiently
- **🛠️ Configurable Pipeline**: Extensive customization options

---

## System Requirements

### Minimum (CPU-only baseline)
- **Python**: 3.11 or 3.12
- **OS**: Windows, macOS, or Linux
- **RAM**: 4 GB
- **Disk Space**: 2 GB
- **Dependencies**: FFmpeg (for video processing)

### Recommended (GPU-accelerated)
- **Python**: 3.11 or 3.12
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11
- **GPU**: NVIDIA GPU with 6+ GB VRAM (for LaMa/SD backends)
- **CUDA**: 12.1 or higher
- **RAM**: 16 GB
- **Disk Space**: 10 GB (including models)

---

## 🚀 Quick Start (CPU Baseline)

Get started in 2 minutes with the CPU-only baseline:

```bash
# 1. Clone repository
git clone https://github.com/Blueibear/Watermark-Remover-Suite.git
cd Watermark-Remover-Suite

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install package
pip install -e .

# 4. Process an image (automatic watermark detection)
wmr image input.jpg --out output.jpg --method telea

# 5. Process a video (with temporal guidance)
wmr video input.mp4 --out output.mp4 --method telea --temporal-guidance
```

**That's it!** The baseline OpenCV Telea method works out-of-the-box without GPU or model downloads.

---

## Installation

### 1. Basic Installation (CPU-only)

```bash
# Clone repository
git clone https://github.com/Blueibear/Watermark-Remover-Suite.git
cd Watermark-Remover-Suite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core package
pip install --upgrade pip
pip install -e .
```

### 2. Install System Dependencies

**FFmpeg** (required for video processing):

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

### 3. Optional: GPU Acceleration

#### Option A: LaMa ONNX (Recommended for quality + speed)

```bash
# Install ONNX runtime with GPU support
pip install -e .[onx]

# Download LaMa ONNX model
python -m watermark_remover.models.download_models --model lama-onnx
# Note: LaMa ONNX requires manual conversion (see model docs)
```

#### Option B: Stable Diffusion (Cutting-edge quality, requires 8+ GB VRAM)

```bash
# Install with SD dependencies
pip install -e .[sd]

# Stable Diffusion will auto-download on first use
# Default model: runwayml/stable-diffusion-inpainting
```

#### Option C: RAFT Optical Flow (Enhanced temporal blending)

```bash
# Download RAFT model weights
python -m watermark_remover.models.download_models --model raft-things

# Alternative: Use RAFT-Sintel for artistic/animated content
python -m watermark_remover.models.download_models --model raft-sintel
```

### 4. Install Optional Features

```bash
# GUI (PySide6-based interface)
pip install -e .[gui]

# Development tools (pytest, ruff, black, mypy)
pip install -e .[develop]

# All optional features
pip install -e .[sd,onx,gui,develop,models]
```

### 5. Verify Installation

```bash
# Check CLI is available
wmr --help

# List available models
python -m watermark_remover.models.download_models --list

# Run smoke tests
pytest tests/test_smoke_video.py -v
```

---

## Usage

### Command-Line Interface (CLI)

#### Image Processing

**Basic usage (auto-detect watermark):**
```bash
wmr image input.jpg --out output.jpg
```

**With manual mask:**
```bash
wmr image input.jpg --out output.jpg --mask manual --mask-path watermark_mask.png
```

**With different inpainting methods:**
```bash
# Fast CPU baseline (default)
wmr image input.jpg --out output.jpg --method telea

# High-quality LaMa ONNX (requires model download)
wmr image input.jpg --out output.jpg --method lama

# Stable Diffusion (requires GPU)
wmr image input.jpg --out output.jpg --method sd
```

**Advanced options:**
```bash
wmr image input.jpg --out output.jpg \
  --mask auto \
  --dilate 7 \
  --method telea \
  --seed 1234
```

#### Video Processing

**Basic usage:**
```bash
wmr video input.mp4 --out output.mp4 --method telea
```

**With temporal guidance (optical flow blending):**
```bash
wmr video input.mp4 --out output.mp4 \
  --method telea \
  --temporal-guidance \
  --window 48 \
  --overlap 12
```

**With quality control and retry:**
```bash
wmr video input.mp4 --out output.mp4 \
  --method lama \
  --qc "warped_ssim>=0.92" \
  --retry-dilate 3
```

**GPU-accelerated with Stable Diffusion:**
```bash
wmr video input.mp4 --out output.mp4 \
  --method sd \
  --temporal-guidance \
  --window 24 \
  --overlap 6
```

#### Command Reference

**Common Arguments (image & video):**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input` | path | - | Input file path (required) |
| `--out` | path | - | Output file path (required) |
| `--mask` | choice | `auto` | Mask mode: `auto` or `manual` |
| `--mask-path` | path | - | Path to manual mask (if `--mask manual`) |
| `--dilate` | int | `5` | Mask dilation iterations |
| `--method` | choice | `telea` | Inpainting: `telea`, `lama`, `sd`, `noop` |
| `--seed` | int | `1234` | Random seed for reproducibility |

**Video-Specific Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--window` | int | `48` | Chunk size (frames per batch) |
| `--overlap` | int | `12` | Overlap between chunks |
| `--temporal-guidance` | flag | `False` | Enable optical flow blending |
| `--qc` | expr | - | Quality check expression (e.g., `warped_ssim>=0.92`) |
| `--retry-dilate` | int | `0` | Dilate mask on QC failure |

**Inpainting Methods:**
- **`telea`**: Fast CPU baseline using OpenCV Telea algorithm (works everywhere)
- **`lama`**: High-quality LaMa ONNX (requires model download, GPU recommended)
- **`sd`**: Stable Diffusion inpainting (requires GPU, 8+ GB VRAM)
- **`noop`**: No-op (copy input, useful for testing)

**Mask Modes:**
- **`auto`**: Automatically detect watermark in bottom-right corner
- **`manual`**: Use custom binary mask from `--mask-path`

### Graphical User Interface (GUI)

```bash
# Launch GUI (requires [gui] optional dependency)
python -m ui.main_window
```

---

## 📊 Performance Notes

### Processing Speed

**Image Processing:**
| Method | GPU | Typical Speed | Quality |
|--------|-----|---------------|---------|
| `telea` | No | ~0.5s/image | Good |
| `lama` | Yes | ~2s/image | Excellent |
| `sd` | Yes | ~5-10s/image | Outstanding |

**Video Processing (1080p):**
| Method | GPU | Temporal | Typical Speed | Quality |
|--------|-----|----------|---------------|---------|
| `telea` | No | No | ~2 FPS | Good |
| `telea` | No | Yes | ~1.5 FPS | Very Good |
| `lama` | Yes | Yes | ~5-8 FPS | Excellent |
| `sd` | Yes | Yes | ~1-2 FPS | Outstanding |

*Benchmarks on Intel i7-10700K + NVIDIA RTX 3080*

### Memory Requirements

| Method | CPU RAM | GPU VRAM | Notes |
|--------|---------|----------|-------|
| `telea` | ~2 GB | - | CPU-only |
| `lama` | ~4 GB | ~4 GB | GPU recommended |
| `sd` | ~8 GB | ~8 GB | GPU required |

### Tips for Best Performance

1. **Start with Telea**: Fast baseline, works everywhere, no setup
2. **Use LaMa for batch jobs**: Best quality/speed trade-off with GPU
3. **Reserve SD for critical tasks**: Highest quality but slowest
4. **Enable temporal guidance for videos**: Smoother transitions, worth the 20% slowdown
5. **Adjust chunk size**: Larger `--window` = faster but more VRAM

---

## 🎨 Examples

### Automatic Watermark Detection

```bash
# Auto-detect watermark in bottom-right corner
wmr image branded_photo.jpg --out clean_photo.jpg --method telea
```

### Manual Mask for Complex Watermarks

```bash
# Create mask (white = remove, black = keep)
# Use any image editor (GIMP, Photoshop, Paint.NET)

# Process with manual mask
wmr image input.jpg --out output.jpg \
  --mask manual \
  --mask-path custom_mask.png \
  --method lama
```

### Video with Temporal Consistency

```bash
# Process video with flow-based blending
wmr video interview.mp4 --out interview_clean.mp4 \
  --method telea \
  --temporal-guidance \
  --window 48 \
  --overlap 12
```

### Quality-Controlled Video Processing

```bash
# Retry failed frames with dilated mask
wmr video concert.mp4 --out concert_clean.mp4 \
  --method lama \
  --qc "warped_ssim>=0.92" \
  --retry-dilate 5 \
  --temporal-guidance
```

---

## 🔧 Advanced Configuration

### Model Download Management

```bash
# List all available models and their status
python -m watermark_remover.models.download_models --list

# Download specific model
python -m watermark_remover.models.download_models --model raft-things

# Download all automatic models
python -m watermark_remover.models.download_models --all

# Force re-download
python -m watermark_remover.models.download_models --model raft-things --force
```

### Custom YAML Configuration

Create `my_config.yaml`:

```yaml
# Image processing defaults
image:
  method: lama
  dilate: 7
  seed: 42

# Video processing defaults
video:
  method: telea
  window: 64
  overlap: 16
  temporal_guidance: true
  qc_expr: "warped_ssim>=0.90"

# Model paths
models:
  raft: ~/.wmr/models/raft-things.pth
  lama: ~/.wmr/models/lama.onnx
  sd_model: runwayml/stable-diffusion-inpainting
```

Load configuration:
```bash
wmr image input.jpg --out output.jpg --config my_config.yaml
```

---

## 🐛 Troubleshooting

### Watermark not fully removed
- **Increase dilation**: Try `--dilate 10` or higher
- **Use manual mask**: Create precise mask in image editor
- **Try different method**: LaMa often handles complex watermarks better

### Video has flickering/artifacts
- **Enable temporal guidance**: Add `--temporal-guidance`
- **Increase overlap**: Try `--overlap 16` or `--overlap 20`
- **Check QC threshold**: Lower `--qc` threshold if too many frames fail

### Out of Memory (GPU)
- **Reduce chunk size**: Lower `--window` to 24 or 16
- **Use CPU offloading**: For SD, it auto-enables on low VRAM
- **Switch to Telea**: CPU-only fallback

### Installation issues
- **Python version**: Ensure Python 3.11 or 3.12 (`python --version`)
- **CUDA mismatch**: Reinstall PyTorch with correct CUDA version
- **FFmpeg missing**: Install system FFmpeg package

### Model download fails
- **Check network**: Some models require manual download
- **Verify disk space**: Models can be 100+ MB
- **Use manual instructions**: See `--list` output for manual steps

---

## 📝 Supported File Formats

**Images:**
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)
- WebP (`.webp`)

**Videos:**
- MP4 (`.mp4`) - recommended
- AVI (`.avi`)
- MOV (`.mov`)
- MKV (`.mkv`)
- WebM (`.webm`)

**Masks:**
- PNG (`.png`) - recommended for transparency
- JPEG (`.jpg`)
- BMP (`.bmp`)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run smoke tests only
pytest tests/test_smoke_video.py -v

# Run MVP package tests
pytest tests/test_watermark_remover_mvp.py -v

# Run with coverage
pytest tests/ --cov=watermark_remover --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Lint code (`ruff check .` and `black .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

**Reminder**: This tool is provided for legitimate purposes only. Users are responsible for compliance with copyright and intellectual property laws.

---

## 🙏 Acknowledgments

- **OpenCV**: Fast CPU inpainting baseline
- **LaMa** (Samsung Research): High-quality large mask inpainting
- **RAFT** (Princeton Vision Lab): Optical flow estimation
- **Stable Diffusion** (Runway/Stability AI): Generative inpainting
- **LPIPS**: Perceptual quality metrics

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/Blueibear/Watermark-Remover-Suite/issues)
- **Documentation**: See `docs/` folder
- **Discussions**: [GitHub Discussions](https://github.com/Blueibear/Watermark-Remover-Suite/discussions)