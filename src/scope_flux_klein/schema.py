"""Configuration schema for FLUX Klein Realtime pipeline."""

from pydantic import Field

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class FluxKleinConfig(BasePipelineConfig):
    """Real-time image generation using FLUX.2-klein-4B."""

    pipeline_id = "flux-klein"
    pipeline_name = "FLUX Klein Realtime"
    pipeline_description = (
        "Real-time text-to-image and image-to-image generation using "
        "Black Forest Labs' FLUX.2-klein-4B model. Generates images in "
        "sub-second time with 4-step inference."
    )

    supports_prompts = True
    estimated_vram_gb = 13.0

    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(height=384, width=384, noise_scale=0.7),
    }

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="black-forest-labs/FLUX.2-klein-4B",
            files=[
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model-00001-of-00002.safetensors",
                "text_encoder/model-00002-of-00002.safetensors",
                "text_encoder/model.safetensors.index.json",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "transformer/config.json",
                "transformer/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        ),
    ]

    # --- Load-time parameters (require pipeline reload) ---

    num_inference_steps: int = Field(
        default=2,
        ge=1,
        le=8,
        description=(
            "Number of denoising steps per generation. Fewer steps = faster "
            "but lower quality. FLUX Klein is optimized for 2-4 steps."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Inference Steps",
            is_load_param=True,
        ),
    )

    enable_cpu_offload: bool = Field(
        default=False,
        description=(
            "Offload model components to CPU when not in use to save VRAM. "
            "Reduces per-frame speed significantly. Enable only if GPU has "
            "less than 16GB VRAM."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="CPU Offload",
            is_load_param=True,
        ),
    )

    # --- Runtime parameters (adjustable during streaming) ---

    feedback_strength: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Strength for feedback loop. When > 0, each frame edits the "
            "previous output instead of generating from scratch (Krea-style). "
            "Lower = faster, smoother transitions. 0 = full regeneration each frame."
        ),
        json_schema_extra=ui_field_config(order=9, label="Feedback Strength"),
    )

    guidance_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description=(
            "Classifier-free guidance scale. Higher values follow the prompt "
            "more closely. FLUX Klein works best at 1.0."
        ),
        json_schema_extra=ui_field_config(order=10, label="Guidance Scale"),
    )

    output_width: int = Field(
        default=384,
        ge=256,
        le=1024,
        description="Output image width in pixels. Lower = faster generation. Snapped to multiple of 16.",
        json_schema_extra=ui_field_config(order=11, label="Width"),
    )

    output_height: int = Field(
        default=384,
        ge=256,
        le=1024,
        description="Output image height in pixels. Lower = faster generation. Snapped to multiple of 16.",
        json_schema_extra=ui_field_config(order=12, label="Height"),
    )

    strength: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "How much to transform the input image in video (img2img) mode. "
            "0.0 = no change, 1.0 = fully regenerate ignoring input."
        ),
        json_schema_extra=ui_field_config(
            order=20,
            label="Strength",
            modes=["video"],
        ),
    )

    seed: int = Field(
        default=-1,
        ge=-1,
        le=2147483647,
        description="Random seed for reproducible generation. -1 = random seed each frame.",
        json_schema_extra=ui_field_config(order=30, label="Seed"),
    )
