# Watermark Remover Suite

A powerful toolkit for removing watermarks from images and videos while preserving the original content quality. Features both command-line and graphical user interfaces for easy use.

---

## Features

- **Image Watermark Removal**: Remove watermarks from JPG, PNG, and other image formats
- **Video Watermark Removal**: Process videos with watermark removal while preserving audio
- **Batch Processing**: Process multiple files automatically with YAML/JSON manifests
- **Multiple Inpainting Methods**: Choose between Telea and Navier-Stokes algorithms
- **Automatic Mask Detection**: Automatically detect watermark regions or provide custom masks
- **GUI and CLI**: Use the graphical interface or command-line tools based on your preference
- **Configurable Settings**: Adjust detection thresholds, inpainting parameters, and output quality

---

## System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Windows, macOS, or Linux
- **Dependencies**: OpenCV, NumPy, PyTorch (see Installation)

---

## Installation

### Basic Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):

```bash
python -m venv .venv
```

3. Activate the virtual environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

4. Install the package:

```bash
pip install --upgrade pip
pip install -e .
```

### Optional Features

Install additional features based on your needs:

**GUI Support:**
```bash
pip install -e .[gui]
```

**Advanced Inpainting (LaMa ONNX):**
```bash
pip install -e .[onx]
```

**Stable Diffusion Integration:**
```bash
pip install -e .[sd]
```

---

## Usage

### Command-Line Interface (CLI)

The CLI provides a fast way to process files from the terminal.

#### Process a Single Image

```bash
wmr image --input photo.jpg --output result.jpg
```

With custom mask:
```bash
wmr image --input photo.jpg --output result.jpg --mask watermark_mask.png
```

With custom inpainting settings:
```bash
wmr image --input photo.jpg --output result.jpg --inpaint-method telea --inpaint-radius 5
```

#### Process a Single Video

```bash
wmr video --input video.mp4 --output clean_video.mp4
```

With custom options:
```bash
wmr video --input video.mp4 --output clean_video.mp4 --codec libx264 --preserve-audio
```

#### Batch Processing

Create a manifest file (e.g., `batch.yaml`) listing your files:

```yaml
- type: image
  input: images/photo1.jpg
  output: results/photo1_clean.jpg

- type: image
  input: images/photo2.jpg
  output: results/photo2_clean.jpg
  mask: masks/photo2_mask.png

- type: video
  input: videos/clip1.mp4
  output: results/clip1_clean.mp4
```

Run batch processing:
```bash
wmr batch --manifest batch.yaml
```

#### Command Options

**Image Processing:**
- `--input, -i`: Input image file path
- `--output, -o`: Output image file path
- `--mask, -m`: Optional custom mask image
- `--inpaint-method`: Inpainting algorithm (`telea` or `ns`)
- `--inpaint-radius`: Radius for inpainting (default: 3)
- `--auto-threshold`: Threshold for automatic mask detection
- `--auto-dilate`: Dilation iterations for mask expansion
- `--auto-blur`: Blur kernel size for mask smoothing

**Video Processing:**
- `--input, -i`: Input video file path
- `--output, -o`: Output video file path
- `--mask, -m`: Optional custom mask image
- `--reuse-mask`: Reuse the same mask for all frames (faster)
- `--preserve-audio`: Keep original audio in output (default: enabled)
- `--codec`: Video codec (e.g., `libx264`, `libx265`)
- `--audio-codec`: Audio codec (default: `aac`)
- `--bitrate`: Output bitrate (e.g., `4M`)

**Batch Processing:**
- `--manifest, -m`: Path to YAML or JSON batch manifest
- `--max-workers`: Maximum concurrent processing jobs
- `--halt-on-error`: Stop processing if any job fails

**General Options:**
- `--config`: Custom configuration file path
- `--log-level`: Logging verbosity (INFO, DEBUG, WARNING, ERROR)
- `--log-file`: Save logs to a specific file

### Graphical User Interface (GUI)

Launch the GUI application for a more visual workflow:

```bash
python -m ui.main_window
```

The GUI allows you to:
- Browse and select input files
- Preview images before/after processing
- Adjust inpainting parameters with visual feedback
- Monitor processing progress
- View processing logs in real-time

---

## Configuration

The tool uses a YAML configuration file (default: `config/default.yaml`) to control processing behavior. You can customize settings like:

- Inpainting methods and parameters
- Automatic mask detection thresholds
- Video encoding options
- Batch processing concurrency
- Logging preferences

Create a custom configuration file and use it with:

```bash
wmr image --input photo.jpg --output result.jpg --config my_config.yaml
```

Example configuration structure:

```yaml
image_processing:
  inpaint_radius: 3
  inpaint_method: telea
  detection:
    threshold: 200
    dilate_iterations: 2
    blur_kernel: 5

video_processing:
  reuse_mask: true
  preserve_audio: true
  codec: libx264
  bitrate: 4M

logging:
  level: INFO
  file:
    enabled: true
    filename: watermark_remover.log
```

---

## Supported File Formats

**Images:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

**Videos:**
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)

---

## Tips for Best Results

1. **Use Custom Masks**: For best results, provide a binary mask highlighting the watermark region
2. **Adjust Threshold**: If automatic detection misses the watermark, lower the `--auto-threshold` value
3. **Increase Radius**: For larger watermarks, increase `--inpaint-radius` (try values 5-10)
4. **Video Performance**: Enable `--reuse-mask` for watermarks in consistent positions
5. **Batch Processing**: Use batch mode for processing multiple files efficiently

---

## Troubleshooting

**Watermark not fully removed:**
- Try lowering the auto-detection threshold
- Increase the inpaint radius
- Provide a custom mask for better precision

**Processing is slow:**
- Use `--reuse-mask` for videos with static watermarks
- Reduce video resolution before processing
- Enable batch processing with `--max-workers` for multiple files

**Audio quality issues:**
- Specify a higher bitrate with `--bitrate`
- Try different audio codecs with `--audio-codec`

**Installation issues:**
- Ensure Python 3.11+ is installed
- Update pip: `pip install --upgrade pip`
- Install system dependencies (FFmpeg for video processing)

---

## License

See [LICENSE](LICENSE) for full licensing information.

---

## Support

For issues, questions, or contributions, please visit the project repository or check the documentation in the `docs/` folder.
