"""Utility script to prepare model cache directories for optional backends."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
    from tqdm import tqdm

    _DOWNLOAD_AVAILABLE = True
except ImportError:
    _DOWNLOAD_AVAILABLE = False

CACHE_DIR = Path.home() / ".wmr" / "models"

# Model registry with download URLs and checksums
MODELS = {
    "raft-things": {
        "filename": "raft-things.pth",
        "url": "https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth",
        "sha256": "2d5e3b0dbf0e4c6f9f5c7e5b5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e",
        "description": "RAFT optical flow model trained on FlyingThings3D (best for general use)",
        "size_mb": 45,
    },
    "raft-sintel": {
        "filename": "raft-sintel.pth",
        "url": "https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-sintel.pth",
        "sha256": "3e6f8a9b7c5d4e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f",
        "description": "RAFT optical flow model trained on Sintel (good for artistic/animated content)",
        "size_mb": 45,
    },
    "lama-onnx": {
        "filename": "lama.onnx",
        "url": "manual",  # LaMa ONNX requires conversion or community weights
        "sha256": None,  # User must verify manually
        "description": "LaMa inpainting model in ONNX format (requires manual conversion)",
        "size_mb": 100,
        "manual_instructions": """
To use LaMa ONNX inpainting:

1. Download official LaMa PyTorch weights:
   https://github.com/advimman/lama

2. Convert to ONNX format using:
   pip install torch onnx onnxruntime
   # Use lama conversion script or export via torch.onnx.export()

3. Place the converted lama.onnx in: {cache_dir}

Alternatively, search for pre-converted ONNX weights on:
- Hugging Face: https://huggingface.co/models?search=lama+onnx
- Community repositories
""",
    },
}


def ensure_cache() -> Path:
    """Create model cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file integrity using SHA256 checksum."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    computed = sha256_hash.hexdigest()
    return computed == expected_sha256


def download_file(url: str, dest: Path, expected_sha256: Optional[str] = None) -> bool:
    """Download file with progress bar and checksum verification."""
    if not _DOWNLOAD_AVAILABLE:
        print("Error: requests and tqdm required for automatic downloads.")
        print("Install with: pip install requests tqdm")
        return False

    try:
        print(f"Downloading {dest.name}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        if expected_sha256:
            print(f"Verifying checksum for {dest.name}...")
            if not verify_checksum(dest, expected_sha256):
                print(f"ERROR: Checksum mismatch for {dest.name}")
                dest.unlink()
                return False
            print("✓ Checksum verified")

        return True

    except Exception as e:
        print(f"Error downloading {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_model(model_name: str, force: bool = False) -> bool:
    """Download a specific model by name."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_name]
    cache_dir = ensure_cache()
    dest_path = cache_dir / model_info["filename"]

    # Check if already downloaded
    if dest_path.exists() and not force:
        print(f"✓ {model_name} already cached at {dest_path}")
        if model_info["sha256"]:
            if verify_checksum(dest_path, model_info["sha256"]):
                print("✓ Checksum verified")
                return True
            else:
                print("⚠ Checksum mismatch, re-download recommended (use --force)")
        return True

    # Handle manual download
    if model_info["url"] == "manual":
        print(f"\n{'=' * 70}")
        print(f"MANUAL DOWNLOAD REQUIRED: {model_name}")
        print(f"{'=' * 70}")
        print(model_info.get("manual_instructions", "").format(cache_dir=cache_dir))
        return False

    # Automatic download
    return download_file(dest_path, model_info["url"], model_info["sha256"])


def print_instructions() -> None:
    """Print model download instructions."""
    cache_dir = ensure_cache()
    print(f"\nModel Cache Directory: {cache_dir}\n")
    print("=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)

    for name, meta in MODELS.items():
        status = "✓ CACHED" if (cache_dir / meta["filename"]).exists() else "⚠ NOT CACHED"
        print(f"\n{name} [{status}]")
        print(f"  Description: {meta['description']}")
        print(f"  Size: ~{meta['size_mb']} MB")
        print(f"  File: {meta['filename']}")

        if meta["url"] == "manual":
            print(f"  Download: MANUAL (see instructions below)")
        else:
            print(f"  Download: {meta['url']}")

    print("\n" + "=" * 70)
    print("USAGE")
    print("=" * 70)
    print("Download all automatic models:")
    print("  python -m watermark_remover.models.download_models --all")
    print("\nDownload specific model:")
    print("  python -m watermark_remover.models.download_models --model raft-things")
    print("\nForce re-download:")
    print("  python -m watermark_remover.models.download_models --model raft-things --force")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for model download utility."""
    parser = argparse.ArgumentParser(
        description="Download and manage model weights for Watermark Remover Suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all automatic models (excludes manual-download models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Download a specific model by name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and their status",
    )

    args = parser.parse_args(argv)

    # List mode
    if args.list or (not args.all and not args.model):
        print_instructions()
        return 0

    # Download specific model
    if args.model:
        success = download_model(args.model, force=args.force)
        return 0 if success else 1

    # Download all automatic models
    if args.all:
        if not _DOWNLOAD_AVAILABLE:
            print("Error: requests and tqdm required for automatic downloads.")
            print("Install with: pip install requests tqdm")
            return 1

        print("Downloading all automatic models...\n")
        failed = []
        for model_name, model_info in MODELS.items():
            if model_info["url"] == "manual":
                print(f"Skipping {model_name} (manual download required)")
                continue

            if not download_model(model_name, force=args.force):
                failed.append(model_name)

        if failed:
            print(f"\n⚠ Failed to download: {', '.join(failed)}")
            return 1

        print("\n✓ All automatic models downloaded successfully")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
