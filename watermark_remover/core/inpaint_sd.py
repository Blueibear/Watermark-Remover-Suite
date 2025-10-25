"""Stable Diffusion inpainting runner for the MVP pipeline."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image

    _DIFFUSERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StableDiffusionInpaintPipeline = None  # type: ignore
    torch = None  # type: ignore
    Image = None  # type: ignore
    _DIFFUSERS_AVAILABLE = False


class StableDiffusionInpainter:
    """Minimal Stable Diffusion inpainting wrapper."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-inpainting",
        device: str = "auto",
        dtype: str = "auto",
    ):
        if not _DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "diffusers, transformers, and torch are required for SD inpainting. "
                "Install with: pip install -e .[sd]"
            )

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Determine dtype
        if dtype == "auto":
            if self.device == "cuda":
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        # Load the pipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not available, continue without it

        # Enable CPU offload for low memory systems
        if self.device == "cpu":
            try:
                self.pipe.enable_sequential_cpu_offload()
            except Exception:
                pass

    def inpaint(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        prompt: str = "high quality, photorealistic, natural lighting, detailed",
        negative_prompt: str = "watermark, text, logo, lowres, blurry, distorted",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Inpaint using Stable Diffusion.

        Args:
            img_bgr: Input image in BGR format (OpenCV format)
            mask_u8: Binary mask in uint8 format (255 = inpaint region)
            prompt: Positive prompt for guidance
            negative_prompt: Negative prompt to avoid artifacts
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility

        Returns:
            Inpainted image in BGR format
        """
        # Convert BGR to RGB PIL image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Convert mask to PIL (ensure it's single channel)
        if mask_u8.ndim == 3:
            mask_gray = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_u8
        pil_mask = Image.fromarray(mask_gray)

        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Run inpainting
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        # Convert result back to BGR numpy array
        result_rgb = np.array(result)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        return result_bgr


# Global cache for the SD pipeline to avoid reloading
_SD_CACHE: Optional[StableDiffusionInpainter] = None


def inpaint_sd(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    model_id: str = "stabilityai/stable-diffusion-2-inpainting",
    device: str = "auto",
    seed: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> np.ndarray:
    """
    Inpaint using Stable Diffusion with caching.

    Args:
        img_bgr: Input image in BGR format
        mask_u8: Binary mask (255 = inpaint region)
        model_id: Hugging Face model ID
        device: "auto", "cuda", "mps", or "cpu"
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance

    Returns:
        Inpainted image in BGR format
    """
    global _SD_CACHE

    # Initialize or reuse the cached pipeline
    if _SD_CACHE is None:
        _SD_CACHE = StableDiffusionInpainter(model_id=model_id, device=device)

    return _SD_CACHE.inpaint(
        img_bgr,
        mask_u8,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
