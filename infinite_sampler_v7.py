"""
LTXInfiniteDirectorSamplerV7 — Reference-frame latent extension.

Uses the same architecture as Kijai's NativeLooping and the YouTube LTX extend workflow:
the last N pixel frames from chunk N are VAE-encoded and written into the START of
chunk N+1's latent with noise_mask=0 (frozen). The model then generates only the NEW
frames (noise_mask=1) in a single pass, with the locked reference frames providing
hard temporal continuity — not a soft guide hint.

This is fundamentally different from V4 (which re-generated the overlap) and V5/V6
(which used append_keyframe / replace_latent_frames on a single frame). Here the
reference region is multi-frame, full-quality, and completely frozen.

carry_frames controls how many pixel frames are locked as reference (default 73 ≈ 3s).
"""

import json
import logging
import os
import math

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
    log.info("[LTXInfiniteDirectorSamplerV7] Saved: %s", path)


def _unpack(result):
    return result.args if hasattr(result, "args") else result


class LTXInfiniteDirectorSamplerV7:
    """
    WAN-style true overlap carry for LTX 2.3.

    Each chunk (except the first) extends its start window back by carry_frames so the
    model generates the overlap region with the correct audio conditioning. After
    generation the overlap is trimmed, leaving only new content. No latent injection,
    no noise masks, no pixel replacement — the model produces continuity naturally.

    Widget layout is identical to V3 so existing workflow JSON works unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model":      ("MODEL",),
                "clip":       ("CLIP",),
                "audio_vae":  ("VAE",),
                "start_second":     ("FLOAT", {"default": 0.0,  "min": 0.0,    "max": 3600.0, "step": 0.01}),
                "end_second":       ("FLOAT", {"default": 85.0, "min": 0.0,    "max": 3600.0, "step": 0.01,
                                               "tooltip": "Total video length in seconds."}),
                "duration_seconds": ("FLOAT", {"default": 85.0, "min": 0.0,    "max": 3600.0, "step": 0.01}),
                "start_frame":      ("INT",   {"default": 0,    "min": 0,       "max": 86400}),
                "end_frame":        ("INT",   {"default": 2040, "min": 0,       "max": 86400}),
                "duration_frames":  ("INT",   {"default": 2040, "min": 0,       "max": 86400}),
                "timeline_data":    ("STRING", {"default": "{}"}),
                "local_prompts":    ("STRING", {"default": ""}),
                "segment_lengths":  ("STRING", {"default": ""}),
                "global_prompt":    ("STRING", {"multiline": True, "default": ""}),
                "guide_strength":   ("STRING", {"default": ""}),
                "epsilon":          ("FLOAT",  {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "frame_rate":    ("FLOAT", {"default": 24.0, "min": 1.0,  "max": 120.0, "step": 0.01}),
                "display_mode":  (["seconds", "frames"], {"default": "seconds"}),
                "custom_width":  ("INT",   {"default": 720,  "min": 64,  "max": 4096, "step": 32}),
                "custom_height": ("INT",   {"default": 1280, "min": 64,  "max": 4096, "step": 32}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "crop", "pad"],
                                  {"default": "maintain aspect ratio"}),
                "divisible_by":  ("INT",   {"default": 32, "min": 1, "max": 256}),
                "img_compression": ("INT", {"default": 0,  "min": 0, "max": 51}),
                "generate_audio":    ("BOOLEAN", {"default": True,  "tooltip": "LTX generates audio from [SOUNDS] prompts. Can be combined with BG Audio."}),
                "custom_audio_on":   ("BOOLEAN", {"default": False, "tooltip": "Use audio file from the timeline AUDIO track. Can be combined with Generate Audio."}),
                "motion_guide_on":   ("BOOLEAN", {"default": True,  "tooltip": "Use motion guide segments from the timeline."}),
                "vae":              ("VAE",),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL",),
                "chunk_duration_seconds": ("FLOAT", {"default": 17.0, "min": 2.0, "max": 120.0, "step": 0.1}),
                "auto_chunk_threshold":   ("FLOAT", {"default": 20.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "carry_frames":     ("INT",   {"default": 73,   "min": 1,    "max": 240,  "step": 1,
                                               "tooltip": "Reference frames from previous chunk to lock at the start of each new chunk. More = stronger continuity, bigger latent. 73 ≈ 3s at 24fps."}),
                "carry_strength":   ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.01,
                                               "tooltip": "Unused in V4 (kept for workflow compatibility)."}),
                "crossfade_frames": ("INT",   {"default": 0,    "min": 0,    "max": 120,  "step": 1}),
                "ic_lora_name":     (["None"] + loras, {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "stage1_steps":   ("INT",   {"default": 8,    "min": 1, "max": 50}),
                "stage2_steps":   ("INT",   {"default": 4,    "min": 1, "max": 50}),
                "stage2_denoise": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg":            ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed":           ("INT",   {"default": 1000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "filename_prefix": ("STRING", {"default": "muse"}),
                "bg_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "bg_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE")
    RETURN_NAMES = ("last_chunk_frames", "audio", "stage1_frames")
    FUNCTION = "execute"
    CATEGORY = "LTXInfiniteDirector"
    DESCRIPTION = (
        "Reference-frame latent extension for LTX 2.3. The last carry_frames pixel frames "
        "from chunk N are VAE-encoded and locked (noise_mask=0) at the start of chunk N+1's "
        "latent. The model generates only the new region in a single pass, anchored to real "
        "prior content — same architecture as Kijai NativeLooping and LTX extend workflows."
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
        generate_audio, custom_audio_on, motion_guide_on,
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

        if not isinstance(ic_lora_name, str):
            ic_lora_name = "None"
        stage1_steps = max(1, int(stage1_steps))
        stage2_steps = max(1, int(stage2_steps))

        total_duration = end_second - start_second
        use_chunks = (auto_chunk_threshold <= 0.0) or (total_duration > auto_chunk_threshold)
        effective_chunk = chunk_duration_seconds if use_chunks else total_duration
        mode = "chunked" if use_chunks else "single-pass"
        log.info("[LTXInfiniteDirectorSamplerV7] %s mode — %.1fs total", mode, total_duration)

        chunks = []
        t = start_second
        while t < end_second - 0.01:
            end = min(t + effective_chunk, end_second)
            chunks.append((t, end))
            t = end
        log.info("[LTXInfiniteDirectorSamplerV7] %d chunk(s), carry_frames=%d (WAN-style overlap)", len(chunks), carry_frames)

        output_dir = folder_paths.get_output_directory()

        # Find next free counter so files never overwrite (same pattern as VHS_VideoCombine)
        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_full.mp4")):
            counter += 1

        # Pre-load bg audio for per-chunk slicing (keeps bg in sync with speech across trims)
        _bg_tracks = []
        try:
            import json as _json_pre
            import soundfile as _sf_pre
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
                    continue
                try:
                    _data, _sr = _sf_pre.read(_ap, dtype="float32", always_2d=True)
                    _bg_tracks.append((
                        torch.from_numpy(_data.T),  # [C, S]
                        _sr,
                        float(_bseg.get("start", 0)),
                        float(_bseg.get("length", 1)),
                        float(_bseg.get("trimStart", 0)),
                        _tl_bg_vol_pre,
                    ))
                    log.info("[LTXInfiniteDirectorSamplerV7] BG pre-loaded: %s", _af)
                except Exception as _exc:
                    log.warning("[LTXInfiniteDirectorSamplerV7] BG pre-load failed: %s", _exc)
        except Exception as _exc:
            log.warning("[LTXInfiniteDirectorSamplerV7] BG pre-parse failed: %s", _exc)

        all_frames = []
        all_s1_frames = []
        all_waveforms = []
        all_bg_waveforms = []
        audio_sample_rate = 44100
        live_pixel_frames = None  # [T, H, W, C] — last carry_frames decoded frames from chunk N
        color_ref_mean = None  # chunk 1 per-channel mean (color reference for all chunks)
        color_ref_std  = None  # chunk 1 per-channel std

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_seed = seed
            s2_seed = seed - 1

            # ── Reference-frame extension: extend start back by ref pixel frames ──
            # For chunk 2+: the last carry_frames decoded pixels from chunk N are
            # frozen into the START of this chunk's latent (noise_mask=0). The model
            # only generates the new region (noise_mask=1), anchored to real prior content.
            raw_s_fr = int(chunk_start * frame_rate)
            e_fr = int(chunk_end * frame_rate)

            if chunk_idx > 0 and live_pixel_frames is not None:
                ref_pixel_count = min(carry_frames, live_pixel_frames.shape[0], raw_s_fr)
            else:
                ref_pixel_count = 0

            # overlap_frames alias for trim/audio logic below
            overlap_frames = ref_pixel_count
            s_fr = raw_s_fr - ref_pixel_count
            gen_start = s_fr / frame_rate

            log.info(
                "[LTXInfiniteDirectorSamplerV7] Chunk %d/%d  %.2f→%.2f s  (ref=%d px → %.3fs locked)  seed=%d",
                chunk_idx + 1, len(chunks), chunk_start, chunk_end, ref_pixel_count, gen_start, chunk_seed,
            )

            s1_w = max(divisible_by, (custom_width  // 2 // divisible_by) * divisible_by)
            s1_h = max(divisible_by, (custom_height // 2 // divisible_by) * divisible_by)

            n_chunk_frames = e_fr - s_fr
            ltxv_len = int(math.ceil((n_chunk_frames - 1) / 8.0) * 8) + 1
            latent_t = ((ltxv_len - 1) // 8) + 1
            pre_latent = {"samples": torch.zeros(
                [1, 128, latent_t, s1_h // 32, s1_w // 32],
                device=mm.intermediate_device(),
            )}


            chunk_dur = e_fr / frame_rate - gen_start  # extended duration

            dir_out = _unpack(Director.execute(
                model, clip,
                gen_start, chunk_end, chunk_dur,
                s_fr, e_fr, e_fr - s_fr,
                timeline_data, local_prompts, segment_lengths,
                global_prompt, guide_strength, epsilon,
                frame_rate, display_mode,
                s1_w, s1_h, resize_method,
                divisible_by, img_compression, audio_vae, pre_latent,
                custom_audio_on, generate_audio, motion_guide_on, False,
            ))
            d_model, positive, video_latent, audio_latent, guide_data, motion_guide_data, d_frame_rate, combined_audio = dir_out

            # Audio mask logic:
            # generate_audio=False, use_custom_audio=False → silence
            # generate_audio=True,  use_custom_audio=False → all-ones mask, LTX generates everything from [SOUNDS]
            # generate_audio=False, use_custom_audio=True  → Director mask: 0=preserve speech, gaps=silence
            # generate_audio=True,  use_custom_audio=True  → Director mask: 0=preserve speech, 1=LTX generates in gaps
            # The Director already handles the last two cases correctly (inpaint_audio=generate_audio).
            # Only override when generate_audio=True but no custom audio — Director returns all-zeros in that case.
            if generate_audio and "samples" in audio_latent:
                # Always force all-ones mask when generating — we want LTX to generate
                # ambient/sfx audio for the full duration regardless of custom audio.
                # When custom_audio_on is also set, the custom waveform gets mixed on top
                # after decode, so speech + generated ambience play simultaneously.
                s = audio_latent["samples"]
                ones = torch.ones(s.shape[0], s.shape[2], s.shape[3], dtype=torch.float32, device=s.device)
                audio_latent = {**audio_latent, "noise_mask": ones}
                log.info("[LTXInfiniteDirectorSamplerV7] generate_audio=True — all-ones mask, LTX generates full-duration audio from [SOUNDS] prompts.")

            # ── Stage 1 ──────────────────────────────────────────────────────────
            zero_neg = _zero_out_conditioning(positive)
            pos1, neg1 = _unpack(LTXVConditioning.execute(positive, zero_neg, frame_rate))

            pos1, neg1, lat1, model_lora, _ = _unpack(DirectorGuide.execute(
                pos1, neg1, vae, video_latent, guide_data,
                motion_guide_data=motion_guide_data,
                model=d_model,
                ic_lora_name=ic_lora_name,
                ic_lora_strength=ic_lora_strength,
            ))
            # ── Reference-frame lock (post-DirectorGuide, must happen after resize) ──
            # VAE-encode the last ref_pixel_count pixel frames at Stage1 resolution and
            # write them into lat1 at position 0 with noise_mask=0 (frozen). The sampler
            # will not touch these frames — they stay as-is, providing hard continuity.
            ref_t = 0
            if live_pixel_frames is not None and ref_pixel_count > 0:
                try:
                    import comfy.utils as _cu
                    ref_px = live_pixel_frames[-ref_pixel_count:]  # [T, H, W, C]
                    # Downscale to Stage1 resolution
                    ref_s1 = _cu.common_upscale(
                        ref_px.movedim(-1, 1),  # [T, C, H, W]
                        s1_w, s1_h, "bilinear", "disabled"
                    ).movedim(1, -1)            # [T, s1_h, s1_w, C]
                    ref_lat = vae.encode(ref_s1[:, :, :, :3])   # [1, 128, ref_t, h, w]
                    ref_t = ref_lat.shape[2]
                    lat1_samples = lat1["samples"].clone()
                    cap = min(ref_t, lat1_samples.shape[2])
                    lat1_samples[:, :, :cap] = ref_lat[:, :, :cap].to(lat1_samples.device, lat1_samples.dtype)
                    # noise_mask: 0=frozen, 1=generate
                    ref_mask = torch.ones(
                        [1, 1, lat1_samples.shape[2], 1, 1],
                        device=lat1_samples.device, dtype=torch.float32,
                    )
                    ref_mask[:, :, :cap] = 0.0
                    lat1 = {**lat1, "samples": lat1_samples, "noise_mask": ref_mask}
                    log.info(
                        "[LTXInfiniteDirectorSamplerV7] Reference lock S1: %d px → %d latent frames frozen",
                        ref_pixel_count, cap,
                    )
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV7] Reference lock S1 failed: %s", exc)

            guider1 = _unpack(CFGGuider.execute(model_lora, pos1, neg1, cfg))[0]
            sampler  = _unpack(KSamplerSelect.execute("euler"))[0]
            sigmas1  = _unpack(BasicScheduler.execute(model_lora, "linear_quadratic", stage1_steps, 1.0))[0]
            noise1   = _unpack(RandomNoise.execute(chunk_seed))[0]
            has_audio = "samples" in audio_latent
            if has_audio:
                av1 = _unpack(LTXVConcatAVLatent.execute(lat1, audio_latent))[0]
            else:
                av1 = lat1
            out1, _ = _unpack(SamplerCustomAdvanced.execute(noise1, guider1, sampler, sigmas1, av1))
            if has_audio:
                out1_vid, aud1 = _unpack(LTXVSeparateAVLatent.execute(out1))
            else:
                out1_vid = out1
                aud1 = {}
            pos1_c, neg1_c, vid1 = DirectorCrop().execute(pos1, neg1, out1_vid)

            # ── Stage1 decode for comparison output ──────────────────────────────
            try:
                s1_decoded = vae.decode(vid1["samples"])
                if s1_decoded.ndim == 5:
                    s1_decoded = s1_decoded.squeeze(0)
                if s1_decoded.ndim == 3:
                    s1_decoded = s1_decoded.unsqueeze(0)
                if overlap_frames > 0:
                    s1_decoded = s1_decoded[overlap_frames:]
                nominal_s1 = int(round((chunk_end - chunk_start) * frame_rate))
                if s1_decoded.shape[0] > nominal_s1:
                    s1_decoded = s1_decoded[:nominal_s1]
                all_s1_frames.append(s1_decoded.cpu())
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSamplerV7] Stage1 decode failed: %s", exc)

            # ── Stage 2 ──────────────────────────────────────────────────────────
            vid_up = _unpack(LTXVLatentUpsampler.execute(vid1, spatial_upscaler, vae))[0]

            pos2, neg2, lat2, _, _ = _unpack(DirectorGuide.execute(
                pos1_c, neg1_c, vae, vid_up, guide_data,
                motion_guide_data=motion_guide_data,
                model=model_lora,
                ic_lora_name="None",
                ic_lora_strength=1.0,
            ))
            # ── Reference-frame lock Stage2 ──────────────────────────────────────
            # Apply the same frozen mask to Stage2 so DirectorGuide's Appended Keyframe
            # Guidance cannot override the reference region at full resolution.
            if live_pixel_frames is not None and ref_pixel_count > 0 and ref_t > 0:
                try:
                    import comfy.utils as _cu2
                    s2_h = lat2["samples"].shape[3] * 32
                    s2_w = lat2["samples"].shape[4] * 32
                    ref_px2 = live_pixel_frames[-ref_pixel_count:]
                    ref_s2 = _cu2.common_upscale(
                        ref_px2.movedim(-1, 1), s2_w, s2_h, "bilinear", "disabled"
                    ).movedim(1, -1)
                    ref_lat2 = vae.encode(ref_s2[:, :, :, :3])
                    cap2 = min(ref_lat2.shape[2], lat2["samples"].shape[2])
                    lat2_samples = lat2["samples"].clone()
                    lat2_samples[:, :, :cap2] = ref_lat2[:, :, :cap2].to(lat2_samples.device, lat2_samples.dtype)
                    ref_mask2 = torch.ones(
                        [1, 1, lat2_samples.shape[2], 1, 1],
                        device=lat2_samples.device, dtype=torch.float32,
                    )
                    ref_mask2[:, :, :cap2] = 0.0
                    lat2 = {**lat2, "samples": lat2_samples, "noise_mask": ref_mask2}
                    log.info("[LTXInfiniteDirectorSamplerV7] Reference lock S2: %d latent frames frozen", cap2)
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV7] Reference lock S2 failed: %s", exc)

            guider2  = _unpack(CFGGuider.execute(model_lora, pos2, neg2, cfg))[0]
            sigmas2  = _unpack(BasicScheduler.execute(model_lora, "linear_quadratic", stage2_steps, stage2_denoise))[0]
            noise2   = _unpack(RandomNoise.execute(s2_seed))[0]
            if has_audio:
                av2 = _unpack(LTXVConcatAVLatent.execute(lat2, aud1))[0]
            else:
                av2 = lat2
            out2, _ = _unpack(SamplerCustomAdvanced.execute(noise2, guider2, sampler, sigmas2, av2))
            if has_audio:
                out2_nosemask = {k: v for k, v in out2.items() if k != "noise_mask"}
                out2_vid, _ = _unpack(LTXVSeparateAVLatent.execute(out2_nosemask))
            else:
                out2_vid = out2
            _, _, vid_final = DirectorCrop().execute(pos2, neg2, out2_vid)

            # ── VAE decode ───────────────────────────────────────────────────────
            lat_shape = list(vid_final["samples"].shape)
            frames = vae.decode(vid_final["samples"])
            log.info("[LTXInfiniteDirectorSamplerV7] latent %s → decoded %s", lat_shape, list(frames.shape))

            if frames.ndim == 5:
                frames = frames.squeeze(0)
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            # frames: [T, H, W, C]

            # ── Save per-chunk MP4 (pre-trim, for debugging) ─────────────────────
            out_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_{chunk_idx:03d}.mp4")
            try:
                _save_chunk_mp4(frames, frame_rate, out_path)
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSamplerV7] Chunk save failed: %s", exc)

            # ── Trim reference region from decoded output ─────────────────────────
            # The first ref_pixel_count frames are the locked reference from chunk N.
            # They're already in all_frames — discard them, keep only new content.
            if overlap_frames > 0:
                frames = frames[overlap_frames:]
                log.info(
                    "[LTXInfiniteDirectorSamplerV7] Trimmed %d reference frames from chunk %d",
                    overlap_frames, chunk_idx,
                )

            # Normalize spatial size to chunk 0 if resolution drifted
            if all_frames and (frames.shape[1] != all_frames[0].shape[1] or frames.shape[2] != all_frames[0].shape[2]):
                import comfy.utils as _cu
                frames = _cu.common_upscale(
                    frames.movedim(-1, 1), all_frames[0].shape[2], all_frames[0].shape[1], "lanczos", "disabled"
                ).movedim(1, -1).clamp(0, 1)

            # Cap to nominal chunk frame count — ltxv_len rounding can add 1 extra frame
            # that would duplicate at the boundary with the next chunk.
            nominal_frames = int(round((chunk_end - chunk_start) * frame_rate))
            if frames.shape[0] > nominal_frames:
                frames = frames[:nominal_frames]

            # ── Color match: normalise each chunk's mean+std to chunk 1 ────────
            # MKL-style per-channel stats transfer. Chunk 1 is the color reference
            # (generated directly from the reference image). All subsequent chunks
            # are matched to it so lighting/exposure stays consistent across the
            # full video regardless of what the model generated independently.
            if chunk_idx == 0:
                # Store chunk 1 stats as the color reference for all future chunks.
                _ref_frames = frames.float()
                color_ref_mean = _ref_frames.mean(dim=(0, 1, 2))   # [C]
                color_ref_std  = _ref_frames.std(dim=(0, 1, 2)).clamp(min=1e-5)
            elif all_frames:
                try:
                    src = frames.float()
                    src_mean = src.mean(dim=(0, 1, 2))
                    src_std  = src.std(dim=(0, 1, 2)).clamp(min=1e-5)
                    # Shift and scale each channel to match chunk 1 stats.
                    corrected = (src - src_mean) / src_std * color_ref_std + color_ref_mean
                    corrected = corrected.clamp(0.0, 1.0)
                    # Blend: full correction at start, taper to 50% over 1 second
                    # so the match feels natural rather than a hard jump.
                    n_blend = min(int(frame_rate), frames.shape[0])
                    blend = torch.linspace(1.0, 0.5, n_blend, device=frames.device)
                    corrected[:n_blend] = (
                        corrected[:n_blend] * blend[:, None, None, None]
                        + src[:n_blend] * (1.0 - blend[:, None, None, None])
                    )
                    frames = corrected.to(all_frames[-1].dtype)
                    log.info(
                        "[LTXInfiniteDirectorSamplerV7] Color match chunk %d: "
                        "mean %s→%s std %s→%s",
                        chunk_idx + 1,
                        [f"{v:.3f}" for v in src_mean.tolist()],
                        [f"{v:.3f}" for v in color_ref_mean.tolist()],
                        [f"{v:.3f}" for v in src_std.tolist()],
                        [f"{v:.3f}" for v in color_ref_std.tolist()],
                    )
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV7] Color match failed: %s", exc)

            all_frames.append(frames)

            # Store last carry_frames frames as reference for next chunk
            cat_so_far = torch.cat(all_frames, dim=0)
            live_pixel_frames = cat_so_far[-carry_frames:].clone().cpu()

            # ── Audio source selection ────────────────────────────────────────────
            # Combinations:
            #   generate only  → decoded audio latent
            #   custom only    → combined_audio from Director (the audio file waveform)
            #   both           → decoded + custom mixed
            #   neither        → silence (combined_audio is already zeros from Director)
            #   any + bg_audio wire → mixed in below after chunk loop
            decoded_wav = None
            if generate_audio and has_audio and isinstance(aud1, dict) and "samples" in aud1:
                try:
                    # audio_vae is a ComfyUI VAE wrapper around AudioVAE.
                    # The inner model's decode(latents) → waveform [B, C, T] where C=1 (mono).
                    # We access via first_stage_model which is the raw AudioVAE instance.
                    inner_vae = audio_vae.first_stage_model
                    aud_samples = aud1["samples"].cpu().float()
                    decoded_wav = inner_vae.decode(aud_samples)  # [B, 1, T]
                    # Expand mono to stereo
                    if decoded_wav.shape[1] == 1:
                        decoded_wav = decoded_wav.expand(-1, 2, -1)
                    decoded_wav = decoded_wav.cpu().float()
                    audio_sr = getattr(inner_vae, "output_sample_rate", 44100)
                    # Resample to 44100 Hz so it matches custom audio and VHS_VideoCombine
                    if audio_sr != 44100:
                        import torchaudio
                        decoded_wav = torchaudio.functional.resample(decoded_wav, audio_sr, 44100)
                        audio_sr = 44100
                    log.info("[LTXInfiniteDirectorSamplerV7] Decoded generated audio: shape %s sr=%d", list(decoded_wav.shape), audio_sr)
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV7] Audio decode failed: %s", exc)
                    decoded_wav = None

            custom_wav = combined_audio.get("waveform") if (custom_audio_on and isinstance(combined_audio, dict)) else None
            audio_sr = combined_audio.get("sample_rate", 44100) if isinstance(combined_audio, dict) else 44100
            # audio_sr stays at 44100 — decoded_wav already resampled to 44100 in decode block

            if decoded_wav is not None and custom_wav is not None:
                # Both: mix generated + custom, matching lengths
                min_len = min(decoded_wav.shape[-1], custom_wav.shape[-1])
                mixed = decoded_wav[..., :min_len] + custom_wav[..., :min_len]
                combined_audio = {"waveform": mixed, "sample_rate": audio_sr}
                log.info("[LTXInfiniteDirectorSamplerV7] Audio: mixed generated + custom")
            elif decoded_wav is not None:
                combined_audio = {"waveform": decoded_wav, "sample_rate": audio_sr}
                log.info("[LTXInfiniteDirectorSamplerV7] Audio: generated only")
            elif custom_wav is not None:
                combined_audio = {"waveform": custom_wav, "sample_rate": audio_sr}
                log.info("[LTXInfiniteDirectorSamplerV7] Audio: custom only")
            else:
                # Neither — silence (combined_audio already contains zeros from Director)
                log.info("[LTXInfiniteDirectorSamplerV7] Audio: silence (no generate or custom)")

            # ── Audio: trim overlap region to match video ────────────────────────
            if isinstance(combined_audio, dict) and "waveform" in combined_audio:
                waveform = combined_audio["waveform"]
                audio_sample_rate = combined_audio.get("sample_rate", 44100)
                if overlap_frames > 0:
                    trim_samples = int(overlap_frames * audio_sample_rate / frame_rate)
                    waveform = waveform[:, :, trim_samples:]
                    log.info(
                        "[LTXInfiniteDirectorSamplerV7] Trimmed %d audio samples (overlap %d frames)",
                        trim_samples, overlap_frames,
                    )
                # Cap to nominal chunk duration — ltxv_len rounding can add ~1 frame
                # of audio that overlaps with the next chunk, causing a stutter.
                nominal_samples = int((chunk_end - chunk_start) * audio_sample_rate)
                if waveform.shape[-1] > nominal_samples:
                    waveform = waveform[:, :, :nominal_samples]
                all_waveforms.append(waveform)

            # ── Per-chunk bg audio (same slice + trim as speech) ─────────────────
            if _bg_tracks:
                target_sr = audio_sample_rate
                chunk_speech_len = waveform.shape[-1] if (isinstance(combined_audio, dict) and "waveform" in combined_audio) else int((chunk_end - gen_start) * target_sr)
                bg_chunk_out = torch.zeros((2, chunk_speech_len), dtype=torch.float32)
                for (_bg_raw, _bg_sr, _seg_start_fr, _seg_len_fr, _seg_trim_fr, _vol) in _bg_tracks:
                    _bg_w = _bg_raw
                    if _bg_sr != target_sr:
                        import torchaudio as _ta2
                        _bg_w = _ta2.functional.resample(_bg_w.unsqueeze(0), _bg_sr, target_sr).squeeze(0)
                    _offset_fr = max(0.0, s_fr - _seg_start_fr)
                    _eff_trim_fr = _seg_trim_fr + _offset_fr
                    _eff_len_fr = max(1.0, _seg_len_fr - _offset_fr)
                    _dst_start_fr = max(0.0, _seg_start_fr - s_fr)
                    _src_start = int(_eff_trim_fr / frame_rate * target_sr)
                    _src_end = min(_src_start + int(_eff_len_fr / frame_rate * target_sr), _bg_w.shape[-1])
                    _dst_start = int(_dst_start_fr / frame_rate * target_sr)
                    if _src_end > _src_start and _dst_start < chunk_speech_len:
                        _clip = _bg_w[:, _src_start:_src_end]
                        if overlap_frames > 0:
                            _bg_trim = int(overlap_frames * target_sr / frame_rate)
                            _clip = _clip[:, _bg_trim:]
                        _avail = min(_clip.shape[-1], chunk_speech_len - _dst_start)
                        if _clip.shape[0] == 1:
                            _clip = _clip.expand(2, -1)
                        elif _clip.shape[0] > 2:
                            _clip = _clip[:2, :]
                        bg_chunk_out[:, _dst_start:_dst_start + _avail] += _clip[:, :_avail] * _vol
                all_bg_waveforms.append(bg_chunk_out.unsqueeze(0))

            mm.soft_empty_cache()

        if not all_frames:
            all_frames = [torch.zeros((1, custom_height, custom_width, 3))]

        if all_waveforms:
            full_audio = {"waveform": torch.cat(all_waveforms, dim=2), "sample_rate": audio_sample_rate}
        else:
            full_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}

        # Mix per-chunk bg audio
        if all_bg_waveforms:
            try:
                full_bg_w = torch.cat(all_bg_waveforms, dim=2)
                lip_w = full_audio["waveform"]
                lip_len = lip_w.shape[-1]
                if full_bg_w.shape[-1] > lip_len:
                    full_bg_w = full_bg_w[..., :lip_len]
                elif full_bg_w.shape[-1] < lip_len:
                    full_bg_w = torch.nn.functional.pad(full_bg_w, (0, lip_len - full_bg_w.shape[-1]))
                if full_bg_w.shape[1] != lip_w.shape[1]:
                    full_bg_w = full_bg_w.expand(-1, lip_w.shape[1], -1) if full_bg_w.shape[1] == 1 else full_bg_w[:, :lip_w.shape[1], :]
                full_audio = {"waveform": lip_w + full_bg_w.to(lip_w.device, lip_w.dtype), "sample_rate": audio_sample_rate}
                log.info("[LTXInfiniteDirectorSamplerV7] Mixed BG audio (per-chunk, %d chunks)", len(all_bg_waveforms))
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSamplerV7] BG audio mix failed: %s", exc)

        # Mix wire-input bg audio
        if bg_audio is not None and "waveform" in bg_audio:
            try:
                lip_w = full_audio["waveform"]
                bg_w  = bg_audio["waveform"].clone()
                bg_sr = bg_audio.get("sample_rate", audio_sample_rate)
                if bg_sr != audio_sample_rate:
                    import torchaudio
                    bg_w = torchaudio.functional.resample(bg_w, bg_sr, audio_sample_rate)
                lip_len = lip_w.shape[-1]
                if bg_w.shape[-1] > lip_len:
                    bg_w = bg_w[..., :lip_len]
                elif bg_w.shape[-1] < lip_len:
                    bg_w = torch.nn.functional.pad(bg_w, (0, lip_len - bg_w.shape[-1]))
                if bg_w.shape[1] != lip_w.shape[1]:
                    bg_w = bg_w.expand(-1, lip_w.shape[1], -1) if bg_w.shape[1] == 1 else bg_w[:, :lip_w.shape[1], :]
                full_audio = {"waveform": lip_w + bg_w.to(lip_w.device, lip_w.dtype) * bg_volume, "sample_rate": audio_sample_rate}
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSamplerV7] Wire BG audio mix failed: %s", exc)

        # Optional crossfade between chunks (keep at 0 for talking head)
        n_cf = int(crossfade_frames) if crossfade_frames else 0
        if len(all_frames) > 1 and n_cf > 0:
            result = [all_frames[0]]
            for i in range(1, len(all_frames)):
                prev, curr = result[-1], all_frames[i]
                n = min(n_cf, prev.shape[0], curr.shape[0])
                if n > 0:
                    alphas = torch.linspace(0.0, 1.0, n + 2, device=prev.device)[1:-1]
                    blended = prev.clone()
                    for j in range(n):
                        a = float(alphas[j])
                        blended[-n + j] = ((1 - a) * prev[-n + j] + a * curr[j]).clamp(0, 1)
                    result[-1] = blended
                    result.append(curr[n:])
                else:
                    result.append(curr)
            full_video = torch.cat(result, dim=0)
        else:
            full_video = torch.cat(all_frames, dim=0)

        full_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_full.mp4")
        try:
            _save_chunk_mp4(full_video, frame_rate, full_path)
        except Exception as exc:
            log.warning("[LTXInfiniteDirectorSamplerV7] Full video save failed: %s", exc)

        s1_video = torch.cat(all_s1_frames, dim=0) if all_s1_frames else full_video
        return (full_video, full_audio, s1_video)


NODE_CLASS_MAPPINGS = {
    "LTXInfiniteDirectorSamplerV7": LTXInfiniteDirectorSamplerV7,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXInfiniteDirectorSamplerV7": "LTX Infinite Director Sampler V7 (Ref-Frame Lock)",
}
