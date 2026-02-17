"""FLUX.2-klein-4B model loading and inference wrapper."""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline

logger = logging.getLogger(__name__)


def _patch_sageattention():
    """Stub out sageattention to prevent broken C extension from loading.

    Scope's container has sageattention compiled against a newer glibc
    than available (GLIBCXX_3.4.32). We inject dummy modules into
    sys.modules so diffusers' eager import check finds our harmless
    stub instead of the broken native extension.
    """
    if "sageattention" in sys.modules:
        return  # Already loaded (or stubbed), don't override

    stub = ModuleType("sageattention")
    fused_stub = ModuleType("sageattention._fused")

    # Add the attributes that diffusers.models.attention_dispatch tries to import
    for attr in [
        "sageattn",
        "sageattn_qk_int8_pv_fp8_cuda",
        "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "sageattn_qk_int8_pv_fp16_cuda",
        "sageattn_qk_int8_pv_fp16_triton",
        "sageattn_varlen",
    ]:
        setattr(stub, attr, None)

    stub._fused = fused_stub
    sys.modules["sageattention"] = stub
    sys.modules["sageattention._fused"] = fused_stub


class FluxKleinModel:
    """Wrapper around Flux2KleinPipeline for text-to-image and image-to-image.

    Handles model loading, CPU offload configuration, seed management,
    and conversion between PIL Images and torch tensors in THWC format.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        device: torch.device | None = None,
        enable_cpu_offload: bool = True,
        num_inference_steps: int = 4,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_inference_steps = num_inference_steps

        logger.info("Loading FLUX.2-klein-4B from %s", model_id)

        # Must patch before importing diffusers to avoid broken sageattention C extension
        _patch_sageattention()

        try:
            from diffusers import Flux2KleinPipeline
        except ImportError:
            raise ImportError(
                "Flux2KleinPipeline not found in your version of diffusers. "
                "This model requires diffusers >= 0.37.0.dev0. Install from git: "
                "pip install git+https://github.com/huggingface/diffusers.git"
            )

        self.pipe: Flux2KleinPipeline = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

        if enable_cpu_offload:
            logger.info("Enabling model CPU offload")
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        logger.info("FLUX.2-klein-4B loaded successfully")

    def _make_generator(self, seed: int) -> torch.Generator | None:
        """Create a torch Generator for the given seed, or None for random."""
        if seed < 0:
            return None
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return gen

    @staticmethod
    def _snap_to_multiple(value: int, multiple: int = 16) -> int:
        """Round a dimension to the nearest multiple (FLUX requires multiples of 16)."""
        return max(multiple, (value // multiple) * multiple)

    @torch.no_grad()
    def text_to_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 1.0,
        seed: int = -1,
    ) -> torch.Tensor:
        """Generate an image from a text prompt.

        Returns:
            Tensor of shape (1, H, W, 3) in [0, 1] range, float32.
        """
        width = self._snap_to_multiple(width)
        height = self._snap_to_multiple(height)
        generator = self._make_generator(seed)

        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )

        return self._pil_to_thwc_tensor(result.images[0])

    @torch.no_grad()
    def image_to_image(
        self,
        prompt: str,
        image: Image.Image,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 1.0,
        seed: int = -1,
    ) -> torch.Tensor:
        """Generate an image conditioned on an input image and prompt.

        Flux2KleinPipeline uses the image as conditioning via joint attention
        concatenation — there is no strength parameter. The image guides the
        generation but full inference always runs.

        Returns:
            Tensor of shape (1, H, W, 3) in [0, 1] range, float32.
        """
        width = self._snap_to_multiple(width)
        height = self._snap_to_multiple(height)

        # Resize input image to match output dimensions
        image = image.resize((width, height), Image.LANCZOS)

        generator = self._make_generator(seed)

        result = self.pipe(
            prompt=prompt,
            image=image,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )

        return self._pil_to_thwc_tensor(result.images[0])

    @staticmethod
    def _thwc_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a THWC tensor in [0, 1] range back to a PIL Image.

        Args:
            tensor: Shape (1, H, W, 3), float32, range [0, 1].
        """
        arr = (tensor.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _pil_to_thwc_tensor(image: Image.Image) -> torch.Tensor:
        """Convert a PIL Image to a THWC tensor in [0, 1] range.

        Returns:
            Tensor of shape (1, H, W, 3), float32, range [0, 1].
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)
        return tensor
