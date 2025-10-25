"""Create synthetic test samples for watermark removal demonstration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_sample_image_with_watermark(
    output_path: Path,
    width: int = 640,
    height: int = 480,
    watermark_text: str = "SAMPLE",
) -> None:
    """Create a synthetic image with a watermark."""
    # Create gradient background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        img[i, :] = [int(255 * i / height), 50, int(255 * (1 - i / height))]

    # Add some random noise for texture
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add watermark text in bottom-right corner
    watermark_position = (width - 180, height - 40)
    cv2.putText(
        img,
        watermark_text,
        watermark_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Add semi-transparent watermark overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (width - 200, height - 60), (width - 10, height - 10), (200, 200, 200), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Re-add text on top
    cv2.putText(
        img,
        watermark_text,
        watermark_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), img)
    print(f"Created sample image: {output_path}")


def create_sample_mask(
    output_path: Path,
    width: int = 640,
    height: int = 480,
) -> None:
    """Create a binary mask for the watermark region."""
    mask = np.zeros((height, width), dtype=np.uint8)
    # Mark bottom-right corner where watermark is
    mask[height - 70 :, width - 210 :] = 255
    cv2.imwrite(str(output_path), mask)
    print(f"Created sample mask: {output_path}")


def create_sample_video_with_watermark(
    output_path: Path,
    width: int = 320,
    height: int = 240,
    num_frames: int = 30,
    fps: float = 10.0,
) -> None:
    """Create a synthetic video with moving content and static watermark."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Create animated gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        offset = int(255 * frame_idx / num_frames)

        for i in range(height):
            frame[i, :] = [
                (offset + int(128 * i / height)) % 256,
                100,
                (255 - offset + int(128 * i / height)) % 256,
            ]

        # Add moving circle
        circle_x = int(width * (0.2 + 0.6 * frame_idx / num_frames))
        circle_y = height // 2
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 0), -1)

        # Add static watermark
        watermark_position = (width - 90, height - 30)
        cv2.putText(
            frame,
            "WM",
            watermark_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Add watermark background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (width - 110, height - 50),
            (width - 5, height - 5),
            (200, 200, 200),
            -1,
        )
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Re-add text
        cv2.putText(
            frame,
            "WM",
            watermark_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"Created sample video: {output_path} ({num_frames} frames)")


def main() -> None:
    """Generate all sample test fixtures."""
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)

    # Create sample image
    create_sample_image_with_watermark(fixtures_dir / "sample_image.png")

    # Create sample mask
    create_sample_mask(fixtures_dir / "sample_mask.png")

    # Create sample video (short clip)
    create_sample_video_with_watermark(fixtures_dir / "sample_video.mp4", num_frames=30)

    print("\n✓ All test fixtures created successfully!")
    print(f"Location: {fixtures_dir}")
    print("\nUsage examples:")
    print(f"  wmr image {fixtures_dir}/sample_image.png --out output.png --method telea")
    print(
        f"  wmr image {fixtures_dir}/sample_image.png --out output.png --mask manual --mask-path {fixtures_dir}/sample_mask.png"
    )
    print(f"  wmr video {fixtures_dir}/sample_video.mp4 --out output.mp4 --method telea")


if __name__ == "__main__":
    main()
