# Muse Collective LTX Timeline

**Standalone LTX 2.3 AV Director + Infinite Sampler for ComfyUI**

Built by [Muse Collective](https://musecollective.co.uk) — generate long-form AI video with lipsync, custom audio, background ambience, and per-segment prompts from a single node with a built-in visual timeline editor.

![Muse Collective LTX Timeline](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange?style=flat-square)
![LTX 2.3](https://img.shields.io/badge/LTX-2.3%20AV-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What it does

A full timeline-based node for generating long-form AI video with LTX 2.3 AV — chunked generation, reference-frame continuity, per-segment prompts, lipsync, and layered audio — all from a single node.

### Key features

- **Visual timeline editor** — drag-and-drop images, audio, and video segments directly onto the timeline
- **Infinite chunking** — generate 90 seconds+ by splitting into overlapping chunks with carry-frame latent locking for seamless transitions
- **Per-segment prompting** — different prompts for different parts of the video using PromptRelay temporal attention masking
- **Lipsync** — sync mouth movements to a custom speech audio file using the LTX talking head LoRA
- **Three audio layers** — generated ambient audio, custom speech/music, and a separate BG ambience track
- **IC-LoRA support** — reference video for camera motion or style consistency
- **Retake mode** — replace a section of an existing video with a new generation
- **Dual-pass sampling** — Stage 1 generation + Stage 2 spatial upscaler pass
- **Color matching** — automatic color correction between chunks

---

## Nodes included

| Node | Description |
|------|-------------|
| `MuseDirectorSamplerV1` | Main director + infinite sampler with timeline UI |
| `LTXInfiniteDirectorSamplerV7` | Previous chunked sampler (no timeline UI) |

---

## Requirements

### ComfyUI custom nodes (install via Manager)

- **WhatDreamsCost-ComfyUI** — required for DirectorGuide processing
- **ComfyUI-VideoHelperSuite** — video preview and saving

### Python packages

```bash
pip install av torchaudio soundfile
```

---

## Installation

### Via ComfyUI Manager
Search for **Muse Collective LTX Timeline** and click Install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/muse-collective-26/muse-ltx-timeline
```

Restart ComfyUI after installing.

---

## Model setup

Download all models and place them in the correct subfolders inside your ComfyUI `models/` directory.

### LTX 2.3 Diffusion Model
Download: [ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors)
Place in: `models/diffusion_models/`

### Audio VAE
Download: [ltx-2.3-22b-distilled_audio_vae.safetensors](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-distilled_audio_vae.safetensors)
Place in: `models/vae/`

### Video VAE
Download: [ltx-2.3-22b-distilled_video_vae.safetensors](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-distilled_video_vae.safetensors)
Place in: `models/vae/`

### Text Encoder (CLIP)
Download: [gemma_3_12B_it_fp4_mixed.safetensors](https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors)
Place in: `models/text_encoders/`

### Spatial Upscaler
Download: [ltx-2.3-spatial-upscaler-x2-1.1.safetensors](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors)
Place in: `models/latent_upscale_models/`

### Talking Head LoRA (required for Lipsync)
Download: [LTX-2.3-22b-AV-LoRA-talking-head-v1.safetensors](https://huggingface.co/Lightricks/LTX-Video/resolve/main/LTX-2.3-22b-AV-LoRA-talking-head-v1.safetensors)
Place in: `models/loras/LTX2.3/`

---

## Example workflow

An example workflow is included in the repo: **`Muse-Director-V1.json`**

This workflow is configured for a 90-second talking head video at 704×1280 with:
- Lipsync to a custom speech audio file
- BG audio track for ambience
- Talking head LoRA loaded
- Base model wired for optional ambient audio generation

Load it in ComfyUI via **Load** and swap in your own image and audio files.

---

## Node connections

| Input | Type | Description |
|-------|------|-------------|
| `model` | MODEL | LTX model with LoRA applied (use talking head LoRA if using lipsync) |
| `clip` | CLIP | LTX text encoder |
| `audio_vae` | VAE | LTX Audio VAE |
| `vae` | VAE | LTX Video VAE |
| `spatial_upscaler` | LATENT_UPSCALE_MODEL | LTX spatial upscaler |
| `bg_audio` | AUDIO (optional) | Background ambience — mixed under everything |
| `base_model` | MODEL (optional) | Base model without LoRA — wire the UNETLoader output directly here for ambient audio generation without the talking head LoRA influencing it |

---

## Timeline tracks

| Track | Purpose |
|-------|---------|
| **MAIN** | Image/video segments — the reference frame(s) for generation |
| **AUDIO** | Speech or music clips for Custom Audio mode |
| **BG AUDIO** | Background ambience — layered under everything, volume controlled by `bg_volume` |
| **MOTION** | Reference video segments for IC-LoRA motion guidance |

---

## Audio toggles

Three buttons in the node toolbar control audio behaviour. They can be used individually or in combination.

---

### Gen Audio (ON only)

LTX generates ambient sound and atmosphere from `[SOUNDS]` tags in your prompts. No audio file needed — the model creates sound that matches the scene description.

```
[SOUNDS] Busy coffee shop, cups clinking, espresso machine, background chatter
```

---

### Custom Audio (ON only)

Plays the audio file(s) from the **AUDIO** timeline track directly in the output. The audio is passed through as-is — no generation, no lipsync. Use this for background music, a pre-recorded voice-over, or any audio you want to play over the video without affecting the visuals.

---

### Custom Audio + Lipsync (Custom Audio ON, Lipsync ON, Gen Audio OFF)

Drives the character's lip movements to match the speech in your custom audio file.

**Requirements:**
- A speech audio file on the **AUDIO** timeline track
- **Custom Audio** button ON
- **Lipsync** button ON
- **Talking head LoRA** loaded into the model input

The model will sync mouth movements to the speech. The original audio file plays in the output — it is not re-generated.

---

### Gen Audio + Custom Audio (both ON, Lipsync OFF)

Generates ambient audio from `[SOUNDS]` prompts AND plays the custom audio file. Both are mixed together in the output. Use this for generated atmosphere layered under a pre-recorded music track or narration that does not require lipsync.

---

### BG Audio track

The BG AUDIO track is independent of all three toggle buttons. Drop any audio file onto it and it will be mixed under the main audio output at the volume level set by `bg_volume`. Works with any combination of the other audio modes.

---

## Prompt format

Use uppercase tags inside segment and global prompts:

```
A woman sits at a podcast desk, talking confidently to camera.
[SPEECH] Right. I'm going to tell you something most people in this space won't admit.
[SOUNDS] Quiet studio, soft air conditioning hum, distant city traffic
```

| Tag | Effect |
|-----|--------|
| `[SPEECH]` | The words the character should say — used with lipsync or generated audio |
| `[SOUNDS]` | Ambient sounds and atmosphere — used when Gen Audio is ON |

> **Important:** Tags must be uppercase. `[speech]` and `[sounds]` (lowercase) are ignored by the model.

---

## Chunking settings

| Setting | Description |
|---------|-------------|
| `chunk_duration_seconds` | Length of each generated chunk (default 30s) |
| `auto_chunk_threshold` | Videos longer than this are split into chunks automatically |
| `carry_frames` | Frames locked from the previous chunk as a reference (default 73 ≈ 3s at 24fps) |
| `carry_strength` | How strongly carry frames anchor the next chunk (default 1.0) |
| `crossfade_frames` | Blend frames at chunk boundaries to smooth transitions |

---

## Recommended settings

| Use case | Resolution | Chunk | Steps S1 / S2 | Denoise S2 |
|----------|-----------|-------|---------------|------------|
| Talking head 9:16 | 704×1280 | 30s | 8 / 4 | 0.42 |
| Short clip 16:9 | 960×544 | 10s | 8 / 4 | 0.42 |
| Long form 9:16 | 704×1280 | 30s | 8 / 4 | 0.42 |

---

## Credits

- Timeline UI and PromptRelay system derived from [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) by WhatDreamsCost
- LTX 2.3 AV model by [Lightricks](https://github.com/Lightricks/LTX-Video)
- Built and extended by [Muse Collective](https://musecollective.co.uk)

---

## License

MIT — free to use, modify and distribute.
