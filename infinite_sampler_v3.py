"""
LTXInfiniteDirectorSampler — one-node long-form LTX 2.3 generation.

Widget order mirrors LTXDirector exactly so the JS timeline editor maps
correctly. Chunking params follow after LTXDirector's last widget.

Use end_second to set total video length (e.g. 85.0).
chunk_duration_seconds controls how long each generation chunk is (~17s).
auto_chunk_threshold: if total duration <= this value, runs single-pass.
"""

import logging
import os

import torch
import folder_paths
import comfy.model_management as mm

from comfy_extras.nodes_custom_sampler import (
    CFGGuider,
    KSamplerSelect,
    BasicScheduler,
    RandomNoise,
    SamplerCustomAdvanced,
)
from comfy_extras.nodes_lt import (
    LTXVConditioning,
    LTXVConcatAVLatent,
    LTXVSeparateAVLatent,
)
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WhatDreamsCost Director nodes — already loaded by ComfyUI at startup,
# so we grab them from the global node registry (no re-import needed).
# ---------------------------------------------------------------------------

_DIRECTOR_CLS = None
_DIRECTOR_GUIDE_CLS = None
_DIRECTOR_CROP_CLS = None


def _load_wdc():
    global _DIRECTOR_CLS, _DIRECTOR_GUIDE_CLS, _DIRECTOR_CROP_CLS
    if _DIRECTOR_CLS is not None:
        return
    import nodes as _nodes
    mapping = _nodes.NODE_CLASS_MAPPINGS
    _DIRECTOR_CLS = mapping["LTXDirector"]
    _DIRECTOR_GUIDE_CLS = mapping["LTXDirectorGuide"]
    _DIRECTOR_CROP_CLS = mapping["LTXDirectorCropGuides"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_out_conditioning(conditioning):
    c = []
    for t in conditioning:
        d = t[1].copy()
        pooled = d.get("pooled_output", None)
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        lyrics = d.get("conditioning_lyrics", None)
        if lyrics is not None:
            d["conditioning_lyrics"] = torch.zeros_like(lyrics)
        c.append([torch.zeros_like(t[0]), d])
    return c


def _inject_carry(vae, video_latent, carry_frames, strength):
    import comfy.utils
    lat = video_latent["samples"]
    _, _, T, H_l, W_l = lat.shape
    _, w_s, h_s = vae.downscale_index_formula
    tgt_w, tgt_h = int(W_l * w_s), int(H_l * h_s)

    frames = carry_frames
    if frames.shape[1] != tgt_h or frames.shape[2] != tgt_w:
        frames = (
            comfy.utils.common_upscale(
                frames.movedim(-1, 1), tgt_w, tgt_h, "lanczos", "disabled"
            ).movedim(1, -1).clamp(0, 1)
        )

    enc = vae.encode(frames[:, :, :, :3])
    n_carry = min(enc.shape[2], T)
    lat2 = lat.clone()
    lat2[:, :, :n_carry] = enc[:, :, :n_carry].to(lat2.device, lat2.dtype)

    b = lat2.shape[0]
    mask = video_latent.get("noise_mask", None)
    if mask is None:
        mask = torch.ones((b, 1, T, 1, 1), dtype=torch.float32, device=lat2.device)
    else:
        mask = mask.clone()
    mask[:, :, :n_carry] = 1.0 - strength
    return {"samples": lat2, "noise_mask": mask}


def _save_chunk_mp4(frames, fps, path):
    import av as _av
    frames_u8 = (frames.cpu().float().clamp(0, 1) * 255).byte().numpy()
    H, W = int(frames_u8.shape[1]), int(frames_u8.shape[2])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=int(fps))
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for f in frames_u8:
            avf = _av.VideoFrame.from_ndarray(f, format="rgb24")
            for pkt in stream.encode(avf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    log.info("[LTXInfiniteDirectorSampler] Saved: %s", path)


def _unpack(result):
    return result.args if hasattr(result, "args") else result


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class LTXInfiniteDirectorSamplerV3:
    """
    Full-pipeline LTX 2.3 chunked sampler with the full LTXDirector timeline UI.

    Widget order mirrors LTXDirector exactly (start_second → end_second → … →
    img_compression) so the JS timeline editor maps correctly. Chunking and
    sampling params follow after.

    Set end_second = total video length (e.g. 85.0).
    Set chunk_duration_seconds to ~17 (= 409 LTX frames, 8n+1 rule).
    Videos saved to output/LTXInfinite/chunk_000.mp4, chunk_001.mp4 …
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                # ── model loaders ───────────────────────────────────────────
                "model":      ("MODEL",),
                "clip":       ("CLIP",),
                "audio_vae":  ("VAE",),

                # ── LTXDirector-compatible time range widgets ────────────────
                # (same names + order as LTXDirector so the JS editor maps correctly)
                "start_second":     ("FLOAT", {"default": 0.0,  "min": 0.0,    "max": 3600.0, "step": 0.01}),
                "end_second":       ("FLOAT", {"default": 85.0, "min": 0.0,    "max": 3600.0, "step": 0.01,
                                               "tooltip": "Total video length in seconds."}),
                "duration_seconds": ("FLOAT", {"default": 85.0, "min": 0.0,    "max": 3600.0, "step": 0.01}),
                "start_frame":      ("INT",   {"default": 0,    "min": 0,       "max": 86400}),
                "end_frame":        ("INT",   {"default": 2040, "min": 0,       "max": 86400}),
                "duration_frames":  ("INT",   {"default": 2040, "min": 0,       "max": 86400}),

                # ── timeline data (hidden by JS editor) ──────────────────────
                "timeline_data":    ("STRING", {"default": "{}"}),
                "local_prompts":    ("STRING", {"default": ""}),
                "segment_lengths":  ("STRING", {"default": ""}),

                # ── prompts ─────────────────────────────────────────────────
                "global_prompt":    ("STRING", {"multiline": True, "default": ""}),
                "guide_strength":   ("STRING", {"default": ""}),
                "epsilon":          ("FLOAT",  {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),

                # ── Director display settings ────────────────────────────────
                "frame_rate":    ("FLOAT", {"default": 24.0, "min": 1.0,  "max": 120.0, "step": 0.01}),
                "display_mode":  (["seconds", "frames"], {"default": "seconds"}),
                "custom_width":  ("INT",   {"default": 720,  "min": 64,  "max": 4096, "step": 32}),
                "custom_height": ("INT",   {"default": 1280, "min": 64,  "max": 4096, "step": 32}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "crop", "pad"],
                                  {"default": "maintain aspect ratio"}),
                "divisible_by":  ("INT",   {"default": 32, "min": 1, "max": 256}),
                "img_compression": ("INT", {"default": 0,  "min": 0, "max": 51}),

                # ── audio / motion flags (hidden by JS) ──────────────────────
                "use_custom_audio":  ("BOOLEAN", {"default": False}),
                "inpaint_audio":     ("BOOLEAN", {"default": True}),
                "use_custom_motion": ("BOOLEAN", {"default": True}),
                "override_audio":    ("BOOLEAN", {"default": False}),

                # ── video VAE + upscaler ─────────────────────────────────────
                "vae":              ("VAE",                {"tooltip": "Video VAE for carry-frame encoding and decode."}),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL", {"tooltip": "Wire a Load Latent Upscale Model node here."}),

                # ── chunking ────────────────────────────────────────────────
                "chunk_duration_seconds": ("FLOAT", {"default": 17.0, "min": 2.0, "max": 120.0, "step": 0.1,
                                                     "tooltip": "~17s = 409 LTX frames (8n+1 rule)."}),
                "auto_chunk_threshold":   ("FLOAT", {"default": 20.0, "min": 0.0, "max": 3600.0, "step": 0.5,
                                                     "tooltip": "Run single-pass if total ≤ this; chunk if above."}),
                "carry_frames":     ("INT",   {"default": 9,    "min": 1,    "max": 64,   "step": 1}),
                "carry_strength":   ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "crossfade_frames": ("INT",   {"default": 24,   "min": 0,    "max": 120,  "step": 1,
                                              "tooltip": "Pixel frames to cross-fade at each chunk boundary (24 = 1s dissolve)."}),

                # ── IC-LoRA ──────────────────────────────────────────────────
                "ic_lora_name":     (["None"] + loras, {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # ── sampling ────────────────────────────────────────────────
                "stage1_steps":   ("INT",   {"default": 8,    "min": 1, "max": 50}),
                "stage2_steps":   ("INT",   {"default": 4,    "min": 1, "max": 50}),
                "stage2_denoise": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg":            ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed":           ("INT",   {"default": 1000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "filename_prefix": ("STRING", {"default": "LTXInfinite/chunk"}),

                # ── background audio ─────────────────────────────────────────
                "bg_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                        "tooltip": "Volume for background audio. 1.0 = same level as lip sync audio."}),
            },
            "optional": {
                "bg_audio": ("AUDIO", {"tooltip": "Optional background/music track mixed into output. Does not drive lip sync."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("last_chunk_frames", "audio")
    FUNCTION = "execute"
    CATEGORY = "LTXInfiniteDirector"
    DESCRIPTION = (
        "Full-pipeline LTX 2.3 chunked generation. "
        "Set end_second = total video length. Chunks auto-saved as MP4."
    )

    def execute(
        self,
        model, clip, audio_vae,
        start_second, end_second, duration_seconds,
        start_frame, end_frame, duration_frames,
        timeline_data, local_prompts, segment_lengths,
        global_prompt, guide_strength, epsilon,
        frame_rate, display_mode, custom_width, custom_height,
        resize_method, divisible_by, img_compression,
        use_custom_audio, inpaint_audio, use_custom_motion, override_audio,
        vae, spatial_upscaler,
        chunk_duration_seconds, auto_chunk_threshold,
        carry_frames, carry_strength, crossfade_frames,
        ic_lora_name, ic_lora_strength,
        stage1_steps, stage2_steps, stage2_denoise, cfg,
        seed, filename_prefix,
        bg_volume=1.0, bg_audio=None,
    ):
        _load_wdc()
        Director = _DIRECTOR_CLS
        DirectorGuide = _DIRECTOR_GUIDE_CLS
        DirectorCrop = _DIRECTOR_CROP_CLS

        # Coerce values that can get corrupted by the JS schema remapper
        if not isinstance(ic_lora_name, str):
            ic_lora_name = "None"
        stage1_steps = max(1, int(stage1_steps))
        stage2_steps = max(1, int(stage2_steps))

        total_duration = end_second - start_second

        # ── single-pass vs chunked ────────────────────────────────────────
        use_chunks = (auto_chunk_threshold <= 0.0) or (total_duration > auto_chunk_threshold)
        effective_chunk = chunk_duration_seconds if use_chunks else total_duration
        mode = "chunked" if use_chunks else "single-pass"
        log.info("[LTXInfiniteDirectorSampler] %s mode — %.1fs total", mode, total_duration)

        # ── build chunk list ──────────────────────────────────────────────
        chunks = []
        t = start_second
        while t < end_second - 0.01:
            end = min(t + effective_chunk, end_second)
            chunks.append((t, end))
            t = end
        log.info("[LTXInfiniteDirectorSampler] %d chunk(s)", len(chunks))

        output_dir = folder_paths.get_output_directory()

        # Pre-load bg audio segments so we can build per-chunk slices (same trim as speech)
        _bg_tracks = []  # list of (waveform_tensor [C, S], src_sr, seg_start_frames, seg_length_frames, seg_trim_start_frames, vol)
        try:
            import json as _json_pre
            import soundfile as _sf_pre
            import torchaudio as _ta_pre
            _td_pre = _json_pre.loads(timeline_data) if timeline_data and timeline_data.strip() not in ("", "{}") else {}
            _tl_bg_vol_pre = float(_td_pre.get("bgAudioVolume", 1.0))
            import folder_paths as _fp_pre
            for _bseg in _td_pre.get("bgAudioSegments", []):
                _af = _bseg.get("audioFile", "")
                if not _af:
                    continue
                _ap = None
                for _base in [_fp_pre.get_input_directory(), os.path.join(_fp_pre.get_input_directory(), "whatdreamscost")]:
                    _c = os.path.join(_base, os.path.basename(_af))
                    if os.path.exists(_c):
                        _ap = _c
                        break
                if not _ap:
                    log.warning("[LTXInfiniteDirectorSamplerV3] BG pre-load: file not found: %s", _af)
                    continue
                try:
                    _data, _sr = _sf_pre.read(_ap, dtype="float32", always_2d=True)
                    _bg_w = torch.from_numpy(_data.T)  # [C, S]
                    _bg_tracks.append((
                        _bg_w, _sr,
                        float(_bseg.get("start", 0)),
                        float(_bseg.get("length", 1)),
                        float(_bseg.get("trimStart", 0)),
                        _tl_bg_vol_pre,
                    ))
                    log.info("[LTXInfiniteDirectorSamplerV3] BG pre-loaded: %s (%d samples @ %d Hz)", _af, _bg_w.shape[-1], _sr)
                except Exception as _exc:
                    log.warning("[LTXInfiniteDirectorSamplerV3] BG pre-load failed %s: %s", _af, _exc)
        except Exception as _exc:
            log.warning("[LTXInfiniteDirectorSamplerV3] BG pre-parse failed: %s", _exc)

        carry = None
        all_frames = []
        all_waveforms = []
        all_bg_waveforms = []  # per-chunk bg slices, same trim as speech
        audio_sample_rate = 44100

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_dur = chunk_end - chunk_start
            chunk_seed = seed
            s2_seed = seed - 1

            log.info("[LTXInfiniteDirectorSampler] Chunk %d/%d  %.2f→%.2f s  seed=%d",
                     chunk_idx + 1, len(chunks), chunk_start, chunk_end, chunk_seed)

            # 1. LTXDirector for this time window
            fr = int(frame_rate)
            s_fr = int(chunk_start * frame_rate)
            e_fr = int(chunk_end * frame_rate)
            s1_w = max(divisible_by, (custom_width  // 2 // divisible_by) * divisible_by)
            s1_h = max(divisible_by, (custom_height // 2 // divisible_by) * divisible_by)

            # Pre-create latent at exact s1_w×s1_h so PromptRelay uses our dimensions
            # instead of auto-calculating from the guide image (which varies per chunk length).
            import math as _math
            n_chunk_frames = e_fr - s_fr
            ltxv_len = int(_math.ceil((n_chunk_frames - 1) / 8.0) * 8) + 1
            latent_t = ((ltxv_len - 1) // 8) + 1
            pre_latent = {"samples": torch.zeros(
                [1, 128, latent_t, s1_h // 32, s1_w // 32],
                device=mm.intermediate_device(),
            )}

            dir_out = _unpack(Director.execute(
                model, clip,
                chunk_start, chunk_end, chunk_dur,
                s_fr, e_fr, e_fr - s_fr,
                timeline_data, local_prompts, segment_lengths,
                global_prompt, guide_strength, epsilon,
                frame_rate, display_mode,
                s1_w, s1_h, resize_method,
                divisible_by, img_compression, audio_vae, pre_latent,
                use_custom_audio, inpaint_audio, use_custom_motion, override_audio,
            ))
            d_model, positive, video_latent, audio_latent, guide_data, motion_guide_data, d_frame_rate, combined_audio = dir_out

            # 2. Inject carry frames
            n_prev_px = carry.shape[0] if carry is not None else 0
            n_prev = (n_prev_px - 1) // 8 + 1 if n_prev_px > 0 else 0  # pixel→latent
            if carry is not None:
                video_latent = _inject_carry(vae, video_latent, carry, carry_strength)

            # 3. Stage 1 — euler / linear_quadratic / 8 steps
            zero_neg = _zero_out_conditioning(positive)
            pos1, neg1 = _unpack(LTXVConditioning.execute(positive, zero_neg, frame_rate))

            pos1, neg1, lat1, model_lora, _ = _unpack(DirectorGuide.execute(
                pos1, neg1, vae, video_latent, guide_data,
                motion_guide_data=motion_guide_data,
                model=d_model,
                ic_lora_name=ic_lora_name,
                ic_lora_strength=ic_lora_strength,
            ))
            # DirectorCrop goes AFTER sampling — before sampling it clears the guide
            # conditioning before the sampler sees it, and also the lat includes guide
            # frames that must stay attached during sampling so the model can attend to them.
            guider1  = _unpack(CFGGuider.execute(model_lora, pos1, neg1, cfg))[0]
            sampler  = _unpack(KSamplerSelect.execute("euler"))[0]
            sigmas1  = _unpack(BasicScheduler.execute(model_lora, "linear_quadratic", stage1_steps, 1.0))[0]
            noise1   = _unpack(RandomNoise.execute(chunk_seed))[0]
            has_audio = "samples" in audio_latent
            if has_audio:
                av1 = _unpack(LTXVConcatAVLatent.execute(lat1, audio_latent))[0]
            else:
                av1 = lat1
            out1, _  = _unpack(SamplerCustomAdvanced.execute(noise1, guider1, sampler, sigmas1, av1))
            if has_audio:
                out1_vid, aud1 = _unpack(LTXVSeparateAVLatent.execute(out1))
            else:
                out1_vid = out1
                aud1 = {}
            # Crop appended guide frames from Stage 1 output.
            # Save pos1_c/neg1_c — Stage 2 reuses them (guide metadata already stripped).
            pos1_c, neg1_c, vid1 = DirectorCrop().execute(pos1, neg1, out1_vid)

            # 4. Spatial upscale
            vid_up = _unpack(LTXVLatentUpsampler.execute(vid1, spatial_upscaler, vae))[0]

            # 5. Stage 2 — refine the upscaled latent (vid_up resolution).
            # Use Stage 1's post-crop conditioning (guide metadata cleared) as the
            # starting point, matching the WDC reference workflow where Stage 2
            # receives Stage 1's output pos/neg directly. DirectorGuide re-encodes
            # guide images at vid_up's spatial resolution automatically.
            pos2, neg2, lat2, _, _ = _unpack(DirectorGuide.execute(
                pos1_c, neg1_c, vae, vid_up, guide_data,
                motion_guide_data=motion_guide_data,
                model=model_lora,
                ic_lora_name="None",
                ic_lora_strength=1.0,
            ))
            guider2  = _unpack(CFGGuider.execute(model_lora, pos2, neg2, cfg))[0]
            sigmas2  = _unpack(BasicScheduler.execute(model_lora, "linear_quadratic", stage2_steps, stage2_denoise))[0]
            noise2   = _unpack(RandomNoise.execute(s2_seed))[0]
            if has_audio:
                av2 = _unpack(LTXVConcatAVLatent.execute(lat2, aud1))[0]
            else:
                av2 = lat2
            # Stage2: no carry mask — let Stage2 freely regenerate audio-conditioned lip movement.
            # Stage1 carry injection already primed the latent with visual context; Stage2 at 0.42
            # denoise naturally blends carry appearance with correct audio-driven lip positions.
            out2, _  = _unpack(SamplerCustomAdvanced.execute(noise2, guider2, sampler, sigmas2, av2))
            if has_audio:
                out2_nosemask = {k: v for k, v in out2.items() if k != "noise_mask"}
                out2_vid, _ = _unpack(LTXVSeparateAVLatent.execute(out2_nosemask))
            else:
                out2_vid = out2
            # Crop appended guide frames from Stage 2 output
            _, _, vid_final = DirectorCrop().execute(pos2, neg2, out2_vid)

            # 6. VAE decode
            lat_shape = list(vid_final["samples"].shape)
            frames = vae.decode(vid_final["samples"])
            log.info("[LTXInfiniteDirectorSampler] latent %s → decoded %s", lat_shape, list(frames.shape))

            # Normalise to [T, H, W, C] — LTX VAE may return [B, T, H, W, C] (5D)
            # or [T, H, W, C] (4D) or [B, H, W, C] (4D single-image batch).
            if frames.ndim == 5:
                frames = frames.squeeze(0)   # [B, T, H, W, C] → [T, H, W, C]
            if frames.ndim == 3:
                frames = frames.unsqueeze(0) # [H, W, C] → [1, H, W, C]
            # frames is now [T, H, W, C]

            # WAN-style pixel carry: force-replace first carry_frames pixels with exact
            # pixels from the previous chunk — pixel-perfect continuity at carry boundary.
            if chunk_idx > 0 and carry is not None:
                n_replace = min(carry.shape[0], frames.shape[0])
                carry_src = carry[:n_replace].to(frames.device, frames.dtype)
                if carry_src.shape[1] != frames.shape[1] or carry_src.shape[2] != frames.shape[2]:
                    import comfy.utils as _cu
                    carry_src = _cu.common_upscale(
                        carry_src.movedim(-1, 1), frames.shape[2], frames.shape[1], "lanczos", "disabled"
                    ).movedim(1, -1).clamp(0, 1)
                frames = frames.clone()
                frames[:n_replace] = carry_src
                log.info("[LTXInfiniteDirectorSamplerV3] Pixel-replaced %d carry frames (WAN-style, chunk %d)", n_replace, chunk_idx)

            # 7. Save chunk
            out_path = os.path.join(output_dir, f"{filename_prefix}_{chunk_idx:03d}.mp4")
            try:
                _save_chunk_mp4(frames, frame_rate, out_path)
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSampler] Save failed for chunk %d: %s", chunk_idx, exc)

            # 8. Extract carry for next chunk, accumulate all frames
            n = min(carry_frames, frames.shape[0])
            carry = frames[-n:]
            # Normalize spatial size to match chunk 0 (PromptRelay may pick different res per chunk length)
            if all_frames and (frames.shape[1] != all_frames[0].shape[1] or frames.shape[2] != all_frames[0].shape[2]):
                import comfy.utils as _cu
                frames = _cu.common_upscale(
                    frames.movedim(-1, 1), all_frames[0].shape[2], all_frames[0].shape[1], "lanczos", "disabled"
                ).movedim(1, -1).clamp(0, 1)
                log.info("[LTXInfiniteDirectorSampler] Resized chunk %d frames %s→%s to match chunk 0",
                         chunk_idx, (frames.shape[1], frames.shape[2]), (all_frames[0].shape[1], all_frames[0].shape[2]))
            # Trim carry frames from start of chunk 2+ — they are pixel-copies of chunk N end, already in output
            if chunk_idx > 0 and n > 0:
                frames = frames[n:]
                log.info("[LTXInfiniteDirectorSampler] Trimmed %d carry frames from chunk %d output (prevents duplicate frames)", n, chunk_idx)
            all_frames.append(frames)
            if isinstance(combined_audio, dict) and "waveform" in combined_audio:
                waveform = combined_audio["waveform"]
                audio_sample_rate = combined_audio.get("sample_rate", 44100)
                # Trim audio to match trimmed video — keep audio/video in sync
                if chunk_idx > 0 and n > 0:
                    trim_samples = int(n * audio_sample_rate / frame_rate)
                    waveform = waveform[:, :, trim_samples:]
                    log.info("[LTXInfiniteDirectorSampler] Trimmed %d audio samples to match %d carry frames", trim_samples, n)
                all_waveforms.append(waveform)

            # Build per-chunk bg audio slice (same time window + same trim as speech audio)
            # so bg stays in sync with speech across chunk boundaries.
            if _bg_tracks:
                target_sr = audio_sample_rate
                chunk_bg_total = waveform.shape[-1] if (isinstance(combined_audio, dict) and "waveform" in combined_audio) else int((chunk_end - chunk_start) * target_sr)
                bg_chunk_out = torch.zeros((2, chunk_bg_total), dtype=torch.float32)
                for (_bg_raw, _bg_sr, _seg_start_fr, _seg_len_fr, _seg_trim_fr, _vol) in _bg_tracks:
                    # Resample to target sr if needed
                    _bg_w = _bg_raw
                    if _bg_sr != target_sr:
                        import torchaudio as _ta2
                        _bg_w = _ta2.functional.resample(_bg_w.unsqueeze(0), _bg_sr, target_sr).squeeze(0)
                    # Same slicing logic as _build_combined_audio: offset into segment for this chunk's start
                    _offset_fr = max(0.0, s_fr - _seg_start_fr)
                    _eff_trim_fr = _seg_trim_fr + _offset_fr
                    _eff_len_fr = max(1.0, _seg_len_fr - _offset_fr)
                    _dst_start_fr = max(0.0, _seg_start_fr - s_fr)
                    _src_start = int(_eff_trim_fr / frame_rate * target_sr)
                    _src_end = _src_start + int(_eff_len_fr / frame_rate * target_sr)
                    _src_end = min(_src_end, _bg_w.shape[-1])
                    _dst_start = int(_dst_start_fr / frame_rate * target_sr)
                    if _src_end > _src_start and _dst_start < chunk_bg_total:
                        _clip = _bg_w[:, _src_start:_src_end]
                        # Apply same trim-from-start as speech audio
                        if chunk_idx > 0 and n > 0:
                            _bg_trim = int(n * target_sr / frame_rate)
                            _clip = _clip[:, _bg_trim:]
                        _avail = min(_clip.shape[-1], chunk_bg_total - _dst_start)
                        if _clip.shape[0] == 1:
                            _clip = _clip.expand(2, -1)
                        elif _clip.shape[0] > 2:
                            _clip = _clip[:2, :]
                        bg_chunk_out[:, _dst_start:_dst_start + _avail] += _clip[:, :_avail] * _vol
                all_bg_waveforms.append(bg_chunk_out.unsqueeze(0))  # [1, C, S]

            # 9. Purge VRAM
            mm.soft_empty_cache()

        if not all_frames:
            all_frames = [torch.zeros((1, custom_height, custom_width, 3))]

        if all_waveforms:
            full_audio = {"waveform": torch.cat(all_waveforms, dim=2), "sample_rate": audio_sample_rate}
        else:
            full_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}

        # Mix bg audio — assembled from per-chunk slices (same trim as speech) so bg stays in sync.
        if all_bg_waveforms:
            try:
                full_bg_w = torch.cat(all_bg_waveforms, dim=2)  # [1, C, total_S]
                lip_w = full_audio["waveform"]
                lip_len = lip_w.shape[-1]
                bg_len = full_bg_w.shape[-1]
                if bg_len > lip_len:
                    full_bg_w = full_bg_w[..., :lip_len]
                elif bg_len < lip_len:
                    full_bg_w = torch.nn.functional.pad(full_bg_w, (0, lip_len - bg_len))
                if full_bg_w.shape[1] != lip_w.shape[1]:
                    if full_bg_w.shape[1] == 1:
                        full_bg_w = full_bg_w.expand(-1, lip_w.shape[1], -1)
                    else:
                        full_bg_w = full_bg_w[:, :lip_w.shape[1], :]
                # vol already baked in during per-chunk accumulation
                mixed = lip_w + full_bg_w.to(lip_w.device, lip_w.dtype)
                full_audio = {"waveform": mixed, "sample_rate": audio_sample_rate}
                log.info("[LTXInfiniteDirectorSamplerV3] Mixed BG audio from timeline (per-chunk, %d chunks)", len(all_bg_waveforms))
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSamplerV3] BG audio timeline mix error: %s", exc)

        # Mix in background audio if provided via wire input
        if bg_audio is not None and "waveform" in bg_audio:
            try:
                lip_w  = full_audio["waveform"]          # [B, C, S]
                bg_w   = bg_audio["waveform"].clone()
                bg_sr  = bg_audio.get("sample_rate", audio_sample_rate)

                # Resample bg to match lip sync sample rate if needed
                if bg_sr != audio_sample_rate:
                    import torchaudio
                    bg_w = torchaudio.functional.resample(bg_w, bg_sr, audio_sample_rate)

                lip_len = lip_w.shape[-1]
                bg_len  = bg_w.shape[-1]

                # Trim or pad bg to match lip sync length
                if bg_len > lip_len:
                    bg_w = bg_w[..., :lip_len]
                elif bg_len < lip_len:
                    pad = lip_len - bg_len
                    bg_w = torch.nn.functional.pad(bg_w, (0, pad))

                # Match channels (mono bg → stereo lip sync or vice versa)
                if bg_w.shape[1] != lip_w.shape[1]:
                    if bg_w.shape[1] == 1:
                        bg_w = bg_w.expand(-1, lip_w.shape[1], -1)
                    else:
                        bg_w = bg_w[:, :lip_w.shape[1], :]

                mixed = lip_w + bg_w.to(lip_w.device, lip_w.dtype) * bg_volume
                full_audio = {"waveform": mixed, "sample_rate": audio_sample_rate}
                log.info("[LTXInfiniteDirectorSamplerV3] BG audio mixed at volume %.2f", bg_volume)
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSamplerV3] BG audio mix failed: %s", exc)

        n_cf_global = int(crossfade_frames) if crossfade_frames is not None else 24
        if len(all_frames) > 1 and n_cf_global > 0:
            result_chunks = [all_frames[0]]
            for i in range(1, len(all_frames)):
                prev = result_chunks[-1]
                curr = all_frames[i]
                n_cf = min(n_cf_global, prev.shape[0], curr.shape[0])
                if n_cf > 0:
                    alphas = torch.linspace(0.0, 1.0, n_cf + 2, device=prev.device)[1:-1]
                    blended = prev.clone()
                    for j in range(n_cf):
                        a = float(alphas[j])
                        blended[-n_cf + j] = ((1.0 - a) * prev[-n_cf + j] + a * curr[j]).clamp(0.0, 1.0)
                    result_chunks[-1] = blended
                    result_chunks.append(curr[n_cf:])
                else:
                    result_chunks.append(curr)
            log.info("[LTXInfiniteDirectorSamplerV3] Cross-faded %d chunk boundaries (%d frames each)", len(all_frames) - 1, n_cf_global)
            full_video = torch.cat(result_chunks, dim=0)
        else:
            full_video = torch.cat(all_frames, dim=0)

        # Save full concatenated video
        full_path = os.path.join(output_dir, f"{filename_prefix}_full.mp4")
        try:
            _save_chunk_mp4(full_video, frame_rate, full_path)
        except Exception as exc:
            log.warning("[LTXInfiniteDirectorSampler] Full video save failed: %s", exc)

        return (full_video, full_audio)


NODE_CLASS_MAPPINGS = {
    "LTXInfiniteDirectorSamplerV3": LTXInfiniteDirectorSamplerV3,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXInfiniteDirectorSamplerV3": "LTX Infinite Director Sampler V3 (BG Audio)",
}
