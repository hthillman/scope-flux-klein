"""FLUX Klein Realtime pipeline -- continuous image generation.

Generates images in real-time using the FLUX.2-klein-4B model.
Each call to __call__() runs a full inference pass (2 steps by default),
producing a fresh image from the current prompt. In video mode,
the input frame conditions the generation via img2img.
"""

from __future__ import annotations

import logging
import threading
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

    Called continuously during streaming. Each invocation runs a full
    FLUX inference pass and returns a single-frame output.
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

        # Store previous output for temporal continuity
        self._prev_output: torch.Tensor | None = None

        # Skip-if-busy: prevent queue overflow by returning cached frame
        # when the previous generation hasn't finished yet
        self._busy_lock = threading.Lock()

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

        Called once per frame during streaming. Reads the prompt from kwargs
        and generates a new image using FLUX.2-klein-4B.
        """
        # Read runtime parameters
        prompt = kwargs.get("prompt", "")
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        output_width = kwargs.get("output_width", 384)
        output_height = kwargs.get("output_height", 384)
        strength = kwargs.get("strength", 0.7)
        seed = kwargs.get("seed", -1)
        video = kwargs.get("video")

        # If prompt is empty, return previous output or black frame
        if not prompt.strip():
            if self._prev_output is not None:
                return {"video": self._prev_output}
            black = torch.zeros(1, output_height, output_width, 3)
            return {"video": black}

        # Skip-if-busy: if still generating the previous frame, return
        # the cached output immediately to prevent output queue overflow
        if not self._busy_lock.acquire(blocking=False):
            if self._prev_output is not None:
                return {"video": self._prev_output}
            black = torch.zeros(1, output_height, output_width, 3)
            return {"video": black}

        try:
            # Dispatch based on mode
            if video is not None and len(video) > 0:
                result = self._generate_img2img(
                    prompt=prompt,
                    input_frames=video,
                    strength=strength,
                    width=output_width,
                    height=output_height,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
            else:
                result = self._generate_text(
                    prompt=prompt,
                    width=output_width,
                    height=output_height,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )

            # Store for temporal continuity
            self._prev_output = result.detach().clone()

            return {"video": result.clamp(0, 1)}
        finally:
            self._busy_lock.release()

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
