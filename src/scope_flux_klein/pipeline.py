"""FLUX Klein Realtime pipeline -- continuous image generation.

Generates images in real-time using the FLUX.2-klein-4B model.
Each call to __call__() checks if parameters changed since the last
call. If nothing changed (and seed is fixed), the cached frame is
returned instantly. Otherwise a fresh inference pass runs.
In video mode, the input frame conditions the generation via img2img.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import FluxKleinConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class FluxKleinPipeline(Pipeline):
    """Real-time image generation using FLUX.2-klein-4B.

    Supports two modes:
    - Text mode: Generates images purely from text prompts.
    - Video mode: Uses input video/canvas frames as conditioning for img2img.
      Connect a camera OR the drawing canvas as the video source to draw
      and have FLUX transform your sketch in real-time.

    Called continuously during streaming. Uses a Krea-style feedback loop:
    the first frame generates from scratch, then each subsequent frame
    "edits" the previous output via img2img with low strength for speed.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return FluxKleinConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        from .model.inference import FluxKleinModel

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Read load-time parameters
        num_inference_steps = kwargs.get("num_inference_steps", 2)
        enable_cpu_offload = kwargs.get("enable_cpu_offload", False)

        # Load the FLUX model
        self.model = FluxKleinModel(
            device=self.device,
            enable_cpu_offload=enable_cpu_offload,
            num_inference_steps=num_inference_steps,
        )

        # Cached output for feedback loop. When feedback_strength > 0,
        # each frame "edits" the previous output via img2img instead of
        # generating from scratch — faster and produces smooth transitions.
        self._prev_output: torch.Tensor | None = None

    def prepare(self, **kwargs) -> Requirements | None:
        """Declare input requirements based on current mode.

        Video mode needs one input frame for img2img conditioning.
        Text mode needs no input frames.
        """
        video = kwargs.get("video")
        if video:
            return Requirements(input_size=1)
        return None

    def __call__(self, **kwargs) -> dict:
        """Generate an image from the current prompt and optional input frame.

        Called continuously during streaming. In text mode with feedback_strength > 0,
        implements a Krea-style feedback loop: the first frame generates from scratch,
        then each subsequent frame "edits" the previous output via img2img. This is
        faster and produces smooth continuous output.
        """
        # Diagnostic: log all kwargs so we can see what Scope sends
        logger.info(
            "__call__ kwargs: %s",
            {k: (type(v).__name__, v) if not isinstance(v, (torch.Tensor, list)) else type(v).__name__ for k, v in kwargs.items()},
        )

        # Read runtime parameters
        prompt = kwargs.get("prompt", "")
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        output_width = kwargs.get("output_width", 384)
        output_height = kwargs.get("output_height", 384)
        strength = kwargs.get("strength", 0.7)
        feedback_strength = kwargs.get("feedback_strength", 0.4)
        seed = kwargs.get("seed", -1)
        video = kwargs.get("video")

        # --- Video mode: always use input frame as img2img source ---
        # Check video FIRST — video mode should work even with empty prompt
        if video is not None and len(video) > 0:
            result = self._generate_img2img(
                prompt=prompt if prompt.strip() else "high quality image",
                input_frames=video,
                strength=strength,
                width=output_width,
                height=output_height,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        # --- Text mode: need a prompt ---
        elif not prompt.strip():
            # No prompt and no video — return cached or black
            if self._prev_output is not None:
                return {"video": self._prev_output}
            return {"video": torch.zeros(1, output_height, output_width, 3)}

        # --- Text mode with feedback loop (Krea-style) ---
        elif self._prev_output is not None and feedback_strength > 0:
            # Convert previous output tensor back to PIL for img2img
            prev_pil = self.model._thwc_tensor_to_pil(self._prev_output)
            result = self.model.image_to_image(
                prompt=prompt,
                image=prev_pil,
                strength=feedback_strength,
                width=output_width,
                height=output_height,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        # --- Text mode, first frame or feedback disabled ---
        else:
            result = self._generate_text(
                prompt=prompt,
                width=output_width,
                height=output_height,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        # Cache result for feedback loop
        self._prev_output = result.detach().clone()

        clamped = result.clamp(0, 1)
        logger.info(
            "Output: shape=%s dtype=%s device=%s min=%.4f max=%.4f mean=%.4f",
            clamped.shape, clamped.dtype, clamped.device,
            clamped.min().item(), clamped.max().item(), clamped.mean().item(),
        )
        return {"video": clamped}

    def _generate_text(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance_scale: float,
        seed: int,
    ) -> torch.Tensor:
        """Generate an image from text prompt only."""
        return self.model.text_to_image(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def _generate_img2img(
        self,
        prompt: str,
        input_frames: list,
        strength: float,
        width: int,
        height: int,
        guidance_scale: float,
        seed: int,
    ) -> torch.Tensor:
        """Generate an image conditioned on an input frame."""
        # Extract the first input frame: (1, H, W, C) in [0, 255]
        frame = input_frames[0].squeeze(0)  # (H, W, C)

        # Convert to PIL Image
        frame_np = frame.cpu().numpy()
        if frame_np.max() > 1.0:
            frame_np = frame_np.clip(0, 255).astype(np.uint8)
        else:
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

        pil_image = Image.fromarray(frame_np, mode="RGB")

        return self.model.image_to_image(
            prompt=prompt,
            image=pil_image,
            strength=strength,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            seed=seed,
        )
