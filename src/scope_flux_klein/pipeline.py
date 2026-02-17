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
        num_inference_steps = kwargs.get("num_inference_steps", 4)
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
        # Scope sends "prompts" (plural) — extract the first prompt string
        prompts = kwargs.get("prompts")
        if prompts and len(prompts) > 0:
            prompt = prompts[0] if isinstance(prompts[0], str) else str(prompts[0])
        else:
            prompt = kwargs.get("prompt", "")
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        output_width = kwargs.get("output_width", 384)
        output_height = kwargs.get("output_height", 384)
        feedback_strength = kwargs.get("feedback_strength")
        if feedback_strength is None:
            feedback_strength = 0.5
        seed = kwargs.get("seed", -1)
        video = kwargs.get("video")

        # --- Video mode: use input frame for generation ---
        # Check video FIRST — video mode should work even with empty prompt
        if video is not None and len(video) > 0:
            effective_prompt = prompt if prompt.strip() else "high quality image"

            # After first frame, use partial denoising for speed (Krea trick).
            # Blend previous Flux output with new video frame so the model
            # refines an image that already has prompt influence, rather than
            # denoising the raw camera feed (which ignores the prompt).
            if self._prev_output is not None and feedback_strength > 0:
                print(
                    f"[FLUX-KLEIN] BRANCH: video refine_frame "
                    f"(strength={feedback_strength})",
                    flush=True,
                )
                # Blend in tensor space: 50% prev output + 50% video frame
                # Both are THWC [0,1] — blend then convert to BCHW [-1,1]
                video_thwc = self._video_frame_to_thwc(video, output_width, output_height)
                prev_thwc = self._prev_output
                if prev_thwc.shape[1:3] != video_thwc.shape[1:3]:
                    prev_thwc = torch.nn.functional.interpolate(
                        prev_thwc.permute(0, 3, 1, 2),
                        size=(video_thwc.shape[1], video_thwc.shape[2]),
                        mode="bilinear", align_corners=False,
                    ).permute(0, 2, 3, 1)
                blended_thwc = 0.5 * prev_thwc + 0.5 * video_thwc

                # Convert THWC [0,1] → BCHW [-1,1] for direct VAE input
                blended_bchw = blended_thwc.squeeze(0).permute(2, 0, 1).unsqueeze(0)
                blended_bchw = (blended_bchw * 2.0 - 1.0).to(torch.float16)

                result = self.model.refine_frame(
                    prompt=effective_prompt,
                    previous_image=blended_bchw,
                    width=output_width,
                    height=output_height,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    strength=feedback_strength,
                )
            else:
                print("[FLUX-KLEIN] BRANCH: video img2img (full)", flush=True)
                result = self._generate_img2img(
                    prompt=effective_prompt,
                    input_frames=video,
                    width=output_width,
                    height=output_height,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )

        # --- Text mode: need a prompt ---
        elif not prompt.strip():
            print("[FLUX-KLEIN] BRANCH: no prompt, returning cached/black", flush=True)
            # No prompt and no video — return cached or black
            if self._prev_output is not None:
                return {"video": self._prev_output}
            return {"video": torch.zeros(1, output_height, output_width, 3)}

        # --- Text mode with feedback loop (Krea-style partial denoise) ---
        elif self._prev_output is not None and feedback_strength > 0:
            print(f"[FLUX-KLEIN] BRANCH: refine_frame (strength={feedback_strength})", flush=True)
            # Convert prev THWC [0,1] → BCHW [-1,1] for direct VAE input
            prev_bchw = self._prev_output.squeeze(0).permute(2, 0, 1).unsqueeze(0)
            prev_bchw = (prev_bchw * 2.0 - 1.0).to(torch.float16)
            result = self.model.refine_frame(
                prompt=prompt,
                previous_image=prev_bchw,
                width=output_width,
                height=output_height,
                guidance_scale=guidance_scale,
                seed=seed,
                strength=feedback_strength,
            )

        # --- Text mode, first frame or feedback disabled ---
        else:
            print("[FLUX-KLEIN] BRANCH: text_to_image (full inference)", flush=True)
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

    @staticmethod
    def _video_frame_to_thwc(
        video: list, width: int, height: int,
    ) -> torch.Tensor:
        """Convert the first Scope video frame tensor to THWC [0,1].

        Returns:
            Tensor of shape (1, H, W, 3) in [0, 1] range, float32.
        """
        frame = video[0].squeeze(0).float()  # (H, W, C)
        if frame.max() > 1.0:
            frame = frame / 255.0
        frame = frame.clamp(0, 1).unsqueeze(0)  # (1, H, W, C)
        # Resize if needed
        if frame.shape[1] != height or frame.shape[2] != width:
            frame = torch.nn.functional.interpolate(
                frame.permute(0, 3, 1, 2),
                size=(height, width),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        return frame

    @staticmethod
    def _video_frame_to_pil(video: list) -> Image.Image:
        """Convert the first Scope video frame tensor to a PIL Image."""
        frame = video[0].squeeze(0)  # (H, W, C)
        frame_np = frame.cpu().numpy()
        if frame_np.max() > 1.0:
            frame_np = frame_np.clip(0, 255).astype(np.uint8)
        else:
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(frame_np, mode="RGB")

    def _generate_img2img(
        self,
        prompt: str,
        input_frames: list,
        width: int,
        height: int,
        guidance_scale: float,
        seed: int,
    ) -> torch.Tensor:
        """Generate an image conditioned on an input frame.

        Flux2KleinPipeline uses the image as conditioning via joint attention
        — no strength parameter. The image guides generation but full
        inference runs every time.
        """
        pil_image = self._video_frame_to_pil(input_frames)

        return self.model.image_to_image(
            prompt=prompt,
            image=pil_image,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            seed=seed,
        )
