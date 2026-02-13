# scope-flux-klein

Real-time image generation plugin for [Daydream Scope](https://github.com/daydreamlive/scope) using Black Forest Labs' [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model.

Generates images continuously as you type, similar to Krea's realtime mode. Supports both text-to-image and image-to-image (video input) generation with 4-step inference for sub-second generation.

## Requirements

- NVIDIA GPU with 8GB+ VRAM (13GB+ recommended without CPU offload)
- CUDA support

## Installation

**From local path** (for development):

1. Open Scope and go to **Settings > Plugins**
2. Click **Browse** and select the `scope-flux-klein` folder
3. Click **Install**

**From Git URL**:

```
git+https://github.com/YOUR_USERNAME/scope-flux-klein.git
```

## Parameters

### Load-time (require pipeline reload)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| Inference Steps | int | 4 | 1-8 | Denoising steps per generation. Optimized for 4. |
| CPU Offload | bool | True | — | Offload model to CPU when idle to save VRAM. |

### Runtime (adjustable during streaming)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| Guidance Scale | float | 1.0 | 0.0-5.0 | How closely to follow the prompt. |
| Width | int | 1024 | 512-1024 | Output width (snapped to multiple of 16). |
| Height | int | 1024 | 512-1024 | Output height (snapped to multiple of 16). |
| Strength | float | 0.7 | 0.0-1.0 | Img2img transformation strength (video mode only). |
| Seed | int | -1 | -1 to 2^31 | Random seed. -1 = random each frame. |

## Development

Edit code, then click **Reload** next to the plugin in Settings. No reinstall needed.
