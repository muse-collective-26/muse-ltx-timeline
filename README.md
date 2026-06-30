# Muse Collective LTX Timeline

**Standalone LTX 2.3 AV Director + Infinite Sampler for ComfyUI**

Built by [Muse Collective](https://musecollective.co.uk) — no external custom node dependencies required.

![Muse Collective LTX Timeline](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange?style=flat-square)
![LTX 2.3](https://img.shields.io/badge/LTX-2.3%20AV-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What it does

A full timeline-based node for generating long-form AI video with LTX 2.3 AV — chunked generation, reference-frame continuity, per-segment prompts, and mixed audio — all from a single node with a built-in visual timeline editor.

### Key features

- **Visual timeline editor** — drag-and-drop images, audio, video segments directly onto the timeline
- **Infinite chunking** — generate videos of any length by splitting into overlapping chunks with carry-frame latent locking for seamless transitions
- **Per-segment prompting** — different prompts for different parts of the video using PromptRelay temporal attention
- **Audio generation** — LTX generates ambient/sfx audio from `[SOUNDS]` tags in prompts
- **Custom audio mixing** — drop speech or music onto the AUDIO track; gets mixed with generated audio
- **BG Audio track** — separate background ambience track with volume control
- **IC-LoRA Video** — reference video for identity/style consistency
- **Retake mode** — replace a section of an existing video with a new generation
- **Stage 1 + Stage 2** — dual-pass sampling with spatial upscaler
- **Three clean toggles** — Gen Audio / Custom Audio / Motion Guide

---

## Nodes included

| Node | Description |
|------|-------------|
| `Muse Collective LTX Timeline V1` | Main director + infinite sampler |
| `LTXInfiniteDirectorSamplerV7` | Previous chunked sampler (no timeline UI) |

---

## Requirements

- ComfyUI (latest)
- LTX 2.3 AV model + Audio VAE + spatial upscaler
- Python packages: `av`, `torchaudio`, `soundfile`

---

## Installation

### Via ComfyUI Manager
Search for **Muse Collective LTX Timeline** and click Install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/musecollective/muse-ltx-timeline
```
Then restart ComfyUI.

---

## Model setup

You'll need these models in your ComfyUI models folder:

| Model | Path |
|-------|------|
| LTX 2.3 AV diffusion model | `models/diffusion_models/` |
| LTX Audio VAE | `models/vae/` |
| LTX Video VAE | `models/vae/` |
| LTX CLIP text encoder | `models/text_encoders/` |
| LTX spatial upscaler | `models/upscale_models/` |

---

## Quick start

1. Add the **Muse Collective LTX Timeline V1** node
2. Connect: model, clip, audio_vae, vae, spatial_upscaler
3. Drop images onto the **MAIN** track
4. Write per-segment prompts and a global prompt
5. Add `[SOUNDS]` tags to prompts for generated audio
6. Hit Queue

---

## Timeline tracks

| Track | Purpose |
|-------|---------|
| **MAIN** | Image/video/text segments — defines what the model sees at each point |
| **AUDIO** | Speech or music clips — mixed with generated audio |
| **BG AUDIO** | Background ambience — sits under everything with volume control |
| **IC-LoRA Video** | Reference video for IC-LoRA consistency |

---

## Audio modes

The three toolbar toggles control audio behaviour:

| Toggle | Effect |
|--------|--------|
| **Gen Audio** | LTX generates ambient audio from `[SOUNDS]` prompts |
| **Custom Audio** | Audio from the AUDIO timeline track is used |
| **Motion Guide** | Motion segments guide the generation |

Both Gen Audio and Custom Audio can be on simultaneously — they get mixed together.

---

## Prompt format

Use tags inside segment prompts:

```
A grey SUV drives down a country road. [sounds] Engine noise, tyres on tarmac.
[camera] The camera follows from behind.
```

---

## Chunking settings

| Setting | Description |
|---------|-------------|
| `chunk_duration_seconds` | Length of each generated chunk |
| `auto_chunk_threshold` | Videos longer than this get chunked automatically |
| `carry_frames` | Reference frames locked from previous chunk (default 73 ≈ 3s) |
| `carry_strength` | How strongly carry frames influence the next chunk |
| `crossfade_frames` | Blend frames between chunks to smooth transitions |

---

## Credits

- Timeline UI derived from [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) by WhatDreamsCost
- PromptRelay temporal attention system by WhatDreamsCost
- LTX 2.3 model by [Lightricks](https://github.com/Lightricks/LTX-Video)
- Built and extended by [Muse Collective](https://musecollective.co.uk)

---

## License

MIT — free to use, modify and distribute.
