# scope-flux-klein

Real-time image generation plugin for [Daydream Scope](https://github.com/daydreamlive/scope) using Black Forest Labs' [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) model.

Uses a Krea-style feedback loop: the first frame generates from scratch, then each subsequent frame "edits" the previous output via img2img. This is faster than full regeneration and produces smooth, continuous output as you adjust prompts and sliders.

## Requirements

- NVIDIA GPU with 13GB+ VRAM (or 8GB+ with CPU offload enabled)
- CUDA support

## Installation

**From local path** (for development):

1. Open Scope and go to **Settings > Plugins**
2. Click **Browse** and select the `scope-flux-klein` folder
3. Click **Install**

**From Git URL**:

```
git+https://github.com/hthillman/scope-flux-klein.git
```

## Usage

### Text mode (default)

1. Select **FLUX Klein Realtime** from the pipeline list
2. Type a prompt — the first frame generates from scratch, then output continuously refines
3. Adjust **Feedback Strength** to control how much each frame changes:
   - `0.3-0.4` = smooth, subtle evolution (recommended for realtime)
   - `0.6-0.8` = more dramatic changes per frame
   - `0.0` = disable feedback loop, generate from scratch each time (slower)

### Video mode (sketch input)

1. Switch to **Video** mode
2. Select a video source:
   - **File** — load a sketch/image from any drawing app
   - **Camera** — point at paper/whiteboard for physical sketching
3. Type a prompt describing what you want FLUX to generate from the input
4. **Strength** slider controls how much FLUX transforms vs. preserves the input

## Parameters

### Load-time (require pipeline reload)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Inference Steps | 2 | 1-8 | Denoising steps per generation. 2 for speed, 4 for quality. |
| CPU Offload | Off | — | Offload model to CPU to save VRAM. Slower per-frame. Enable if GPU has <16GB. |

### Runtime (adjustable during streaming)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Feedback Strength | 0.4 | 0.0-1.0 | How much to edit each frame vs. regenerate. 0 = full regen. |
| Guidance Scale | 1.0 | 0.0-5.0 | How closely to follow the prompt. |
| Width | 384 | 256-1024 | Output width (snapped to multiple of 16). |
| Height | 384 | 256-1024 | Output height (snapped to multiple of 16). |
| Strength | 0.7 | 0.0-1.0 | Img2img transformation strength (video mode only). |
| Seed | -1 | -1 to 2^31 | Random seed. -1 = random each frame. |

## Development

Edit code, then click **Reload** next to the plugin in Settings. No reinstall needed.
