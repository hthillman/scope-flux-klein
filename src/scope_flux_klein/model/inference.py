"""FLUX.2-klein-4B model loading and inference wrapper."""

from __future__ import annotations

import logging
import sys
import time
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
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

        # Prompt embedding cache — Qwen3 encoding is expensive, reuse across frames
        self._cached_prompt_str: str | None = None
        self._cached_prompt_embeds: torch.Tensor | None = None
        self._cached_text_ids: torch.Tensor | None = None

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

    @torch.no_grad()
    def refine_frame(
        self,
        prompt: str,
        previous_image: Image.Image,
        width: int = 384,
        height: int = 384,
        guidance_scale: float = 1.0,
        seed: int = -1,
        strength: float = 0.3,
    ) -> torch.Tensor:
        """Partial-denoise the previous output for fast feedback (Krea trick).

        Instead of generating from scratch every frame, this:
        1. Encodes previous output to latents via VAE
        2. Adds noise at a partial timestep (controlled by strength)
        3. Denoises for only the remaining steps
        4. Decodes back to pixels

        With strength=0.3 and 4 steps, only ~1-2 transformer passes run
        instead of 4, roughly doubling FPS on subsequent frames.

        Returns:
            Tensor of shape (1, H, W, 3) in [0, 1] range, float32.
        """
        width = self._snap_to_multiple(width)
        height = self._snap_to_multiple(height)
        device = self.pipe._execution_device
        dtype = self.pipe.transformer.dtype
        generator = self._make_generator(seed)

        t0 = time.perf_counter()

        # --- 1. Cache prompt embeddings (only re-encode when prompt changes) ---
        if prompt != self._cached_prompt_str:
            self._cached_prompt_embeds, self._cached_text_ids = (
                self.pipe.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                )
            )
            self._cached_prompt_str = prompt

        prompt_embeds = self._cached_prompt_embeds
        text_ids = self._cached_text_ids

        # --- 2. Encode previous image to latents ---
        # Resize and preprocess to tensor in [-1, 1]
        previous_image = previous_image.resize((width, height), Image.LANCZOS)
        image_tensor = self.pipe.image_processor.preprocess(
            previous_image, height=height, width=width,
        )
        image_tensor = image_tensor.to(device=device, dtype=dtype)

        # VAE encode → raw latents (B, 32, H/8, W/8)
        image_latents = self.pipe.vae.encode(image_tensor).latent_dist.mode()

        # Patchify: (B, 32, H/8, W/8) → (B, 128, H/16, W/16)
        image_latents = self.pipe._patchify_latents(image_latents)

        # BN normalize
        bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
        bn_std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1).to(device, dtype)
            + self.pipe.vae.config.batch_norm_eps
        )
        image_latents = (image_latents - bn_mean) / bn_std

        # Prepare position IDs from patchified spatial shape
        latent_ids = self.pipe._prepare_latent_ids(image_latents).to(device)

        # Pack: (B, 128, H', W') → (B, H'*W', 128)
        clean_latents = self.pipe._pack_latents(image_latents)

        # --- 3. Set up truncated schedule ---
        self.pipe.scheduler.set_timesteps(
            self.num_inference_steps, device=device,
        )
        all_timesteps = self.pipe.scheduler.timesteps

        init_timestep = min(
            int(self.num_inference_steps * strength), self.num_inference_steps,
        )
        t_start = max(self.num_inference_steps - init_timestep, 0)
        timesteps = all_timesteps[t_start:]
        num_steps = len(timesteps)

        if hasattr(self.pipe.scheduler, "set_begin_index"):
            self.pipe.scheduler.set_begin_index(t_start)

        if num_steps == 0:
            # strength ~0: no denoising needed, return input as-is
            return self._pil_to_thwc_tensor(previous_image)

        # --- 4. Add noise at starting sigma ---
        start_sigma = (
            timesteps[0].float() / self.pipe.scheduler.config.num_train_timesteps
        )
        start_sigma = start_sigma.view(-1, 1, 1).to(device=device, dtype=dtype)

        noise = randn_tensor(
            clean_latents.shape, generator=generator, device=device, dtype=dtype,
        )
        latents = start_sigma * noise + (1.0 - start_sigma) * clean_latents

        # --- 5. Run transformer for remaining steps ---
        do_cfg = guidance_scale > 1.0
        image_seq_len = latents.shape[1]

        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            with self.pipe.transformer.cache_context("cond"):
                noise_pred = self.pipe.transformer(
                    hidden_states=latents.to(dtype),
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

            # Trim to image sequence length (transformer may output extra tokens)
            noise_pred = noise_pred[:, :image_seq_len]

            if do_cfg:
                with self.pipe.transformer.cache_context("uncond"):
                    neg_pred = self.pipe.transformer(
                        hidden_states=latents.to(dtype),
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,  # empty prompt TODO
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
                neg_pred = neg_pred[:, :image_seq_len]
                noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

            # Euler step
            latents = self.pipe.scheduler.step(
                noise_pred, t, latents, return_dict=False,
            )[0]

        # --- 6. Decode latents back to image ---
        # Unpack: (B, seq, 128) → (B, 128, H', W')
        latents = self.pipe._unpack_latents_with_ids(latents, latent_ids)

        # BN denormalize
        latents = latents * bn_std + bn_mean

        # Unpatchify: (B, 128, H', W') → (B, 32, H/8, W/8)
        latents = self.pipe._unpatchify_latents(latents)

        # VAE decode: (B, 32, H/8, W/8) → (B, 3, H, W)
        image = self.pipe.vae.decode(latents.to(dtype), return_dict=False)[0]

        # Postprocess to PIL then to THWC tensor
        pil_images = self.pipe.image_processor.postprocess(image, output_type="pil")

        elapsed = time.perf_counter() - t0
        print(
            f"[FLUX-KLEIN] refine_frame: {num_steps} steps, "
            f"{elapsed:.3f}s ({1/elapsed:.1f} FPS potential)",
            flush=True,
        )

        return self._pil_to_thwc_tensor(pil_images[0])

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
