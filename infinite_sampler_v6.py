"""
LTXInfiniteDirectorSamplerV6 — Hard pixel lock at chunk boundaries.

Builds on V4's WAN-style overlap carry, adding a true pixel-level anchor at the
start of every chunk using LTXVAddGuide.replace_latent_frames(). The last decoded
pixel frame of chunk N is VAE-encoded at Stage1 resolution and locked into latent
position 0 of chunk N+1 via noise_mask=0.0. This is the same mechanism used by
Kijai's NativeLooping workflow — a hard constraint that prevents the boundary snap
caused by the WhatDreamsCost Appended Keyframe Guidance mechanism.

Director timeline is preserved for prompts, audio, and image guides.
"""

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
    LTXVAddGuide,
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
    log.info("[LTXInfiniteDirectorSamplerV6] Saved: %s", path)


def _unpack(result):
    return result.args if hasattr(result, "args") else result


class LTXInfiniteDirectorSamplerV6:
    """
    Hard pixel lock at chunk boundaries for LTX 2.3.

    V4 WAN-style overlap + LTXVAddGuide.replace_latent_frames() pixel anchor at position 0.
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
                "use_custom_audio":  ("BOOLEAN", {"default": False}),
                "inpaint_audio":     ("BOOLEAN", {"default": True}),
                "use_custom_motion": ("BOOLEAN", {"default": True}),
                "override_audio":    ("BOOLEAN", {"default": False}),
                "vae":              ("VAE",),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL",),
                "chunk_duration_seconds": ("FLOAT", {"default": 17.0, "min": 2.0, "max": 120.0, "step": 0.1}),
                "auto_chunk_threshold":   ("FLOAT", {"default": 20.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "carry_frames":     ("INT",   {"default": 9,    "min": 1,    "max": 64,   "step": 1,
                                               "tooltip": "Overlap frames. Model generates these on both sides of every boundary; only the new side is kept."}),
                "carry_strength":   ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.01,
                                               "tooltip": "Unused in V4 (kept for workflow compatibility)."}),
                "crossfade_frames": ("INT",   {"default": 0,    "min": 0,    "max": 120,  "step": 1}),
                "ic_lora_name":     (["None"] + loras, {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "live_frame_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                                   "tooltip": "Strength of pixel-level lock at chunk boundaries (1.0 = hard freeze, 0.0 = disabled)."}),
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

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("last_chunk_frames", "audio")
    FUNCTION = "execute"
    CATEGORY = "LTXInfiniteDirector"
    DESCRIPTION = (
        "V4 WAN-style overlap + hard pixel lock at chunk boundaries via LTXVAddGuide.replace_latent_frames(). "
        "The last decoded pixel frame of each chunk is VAE-encoded at Stage1 resolution and locked into "
        "latent position 0 of the next chunk (noise_mask=0.0), preventing the boundary snap caused by "
        "WhatDreamsCost Appended Keyframe Guidance."
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
        live_frame_strength,
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
        log.info("[LTXInfiniteDirectorSamplerV6] %s mode — %.1fs total", mode, total_duration)

        chunks = []
        t = start_second
        while t < end_second - 0.01:
            end = min(t + effective_chunk, end_second)
            chunks.append((t, end))
            t = end
        log.info("[LTXInfiniteDirectorSamplerV6] %d chunk(s), carry_frames=%d (WAN-style overlap)", len(chunks), carry_frames)

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
                    log.info("[LTXInfiniteDirectorSamplerV6] BG pre-loaded: %s", _af)
                except Exception as _exc:
                    log.warning("[LTXInfiniteDirectorSamplerV6] BG pre-load failed: %s", _exc)
        except Exception as _exc:
            log.warning("[LTXInfiniteDirectorSamplerV6] BG pre-parse failed: %s", _exc)

        all_frames = []
        all_waveforms = []
        all_bg_waveforms = []
        audio_sample_rate = 44100
        carry_px = None   # unused — kept for future reference
        carry_lat = None  # last N latent frames from Stage1 output of previous chunk
        live_pixel_frame = None  # last decoded pixel frame of previous chunk [1, H, W, C]
        color_ref_mean = None  # chunk 1 per-channel mean (color reference for all chunks)
        color_ref_std  = None  # chunk 1 per-channel std

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_seed = seed
            s2_seed = seed - 1

            # ── WAN-style overlap: extend start back by carry_frames for chunk 2+ ──
            # The model generates carry_frames extra frames at the start, conditioned on
            # the correct audio for that time window. We trim them from the output.
            raw_s_fr = int(chunk_start * frame_rate)
            e_fr = int(chunk_end * frame_rate)

            if chunk_idx > 0:
                overlap_frames = min(carry_frames, raw_s_fr)  # can't go before frame 0
            else:
                overlap_frames = 0

            s_fr = raw_s_fr - overlap_frames
            gen_start = s_fr / frame_rate

            log.info(
                "[LTXInfiniteDirectorSamplerV6] Chunk %d/%d  %.2f→%.2f s  (overlap=%d fr back to %.3fs)  seed=%d",
                chunk_idx + 1, len(chunks), chunk_start, chunk_end, overlap_frames, gen_start, chunk_seed,
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
                use_custom_audio, inpaint_audio, use_custom_motion, override_audio,
            ))
            d_model, positive, video_latent, audio_latent, guide_data, motion_guide_data, d_frame_rate, combined_audio = dir_out

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
            # ── Carry injection into Stage1 latent (post-DirectorGuide) ─────────────
            # Must happen here — DirectorGuide may resize/reinitialise the latent.
            # Stage1 noise_mask=0.0 on carry positions tells the sampler to keep them,
            # providing visual continuity. Stage2 has NO mask so lips move freely.
            if carry_lat is not None and overlap_frames > 0:
                try:
                    n_lat_carry = min(carry_lat.shape[2], lat1["samples"].shape[2])
                    cl = carry_lat[:, :, :n_lat_carry]
                    tgt_h = lat1["samples"].shape[3]
                    tgt_w = lat1["samples"].shape[4]
                    if cl.shape[3] != tgt_h or cl.shape[4] != tgt_w:
                        import comfy.utils as _cu
                        cl_flat = cl.reshape(n_lat_carry, 128, cl.shape[3], cl.shape[4])
                        cl_flat = _cu.common_upscale(cl_flat, tgt_w, tgt_h, "bilinear", "disabled")
                        cl = cl_flat.reshape(1, 128, n_lat_carry, tgt_h, tgt_w)
                    lat1["samples"][:, :, :n_lat_carry] = cl.to(lat1["samples"].device, lat1["samples"].dtype)
                    # Stage1 noise mask: 0.0 = frozen (sampler won't denoise these frames)
                    existing_mask = lat1.get("noise_mask", None)
                    if existing_mask is not None and not (hasattr(existing_mask, "is_nested") and existing_mask.is_nested):
                        s1_mask = existing_mask.clone()
                    else:
                        s1_mask = torch.ones(
                            [1, 1, lat1["samples"].shape[2], lat1["samples"].shape[3], lat1["samples"].shape[4]],
                            device=lat1["samples"].device,
                        )
                    s1_mask[:, :, :n_lat_carry] = 0.0
                    lat1 = {**lat1, "noise_mask": s1_mask}
                    log.info(
                        "[LTXInfiniteDirectorSamplerV6] Carry injected: %d latent frames locked in Stage1, Stage2 free",
                        n_lat_carry,
                    )
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV6] Carry injection failed: %s", exc)

            # ── Pixel lock: hard-freeze position 0 to last decoded frame of chunk N ─
            # Uses LTXVAddGuide.replace_latent_frames() — same mechanism as Kijai's
            # NativeLooping. Encodes the live pixel frame at Stage1 resolution and sets
            # noise_mask=0.0 at latent position 0, preventing the Appended Keyframe
            # Guidance boundary snap.
            if live_pixel_frame is not None and chunk_idx > 0 and live_frame_strength > 0.0:
                try:
                    import comfy.utils as _cu
                    # Downscale live frame to Stage1 spatial resolution
                    live_s1 = _cu.common_upscale(
                        live_pixel_frame.movedim(-1, 1),  # [1, C, H, W]
                        s1_w, s1_h, "bilinear", "disabled"
                    ).movedim(1, -1)  # [1, s1_h, s1_w, C]
                    # VAE encode → [1, 128, 1, s1_h//32, s1_w//32]
                    live_lat = vae.encode(live_s1[:, :, :, :3])
                    # Ensure noise_mask exists
                    if "noise_mask" not in lat1:
                        lat1 = {**lat1, "noise_mask": torch.ones(
                            [1, 1, lat1["samples"].shape[2], lat1["samples"].shape[3], lat1["samples"].shape[4]],
                            device=lat1["samples"].device,
                        )}
                    lat1_samples, lat1_mask = LTXVAddGuide.replace_latent_frames(
                        lat1["samples"], lat1["noise_mask"],
                        live_lat.to(lat1["samples"].device, lat1["samples"].dtype),
                        latent_idx=0, strength=live_frame_strength,
                    )
                    lat1 = {**lat1, "samples": lat1_samples, "noise_mask": lat1_mask}
                    log.info("[LTXInfiniteDirectorSamplerV6] Pixel lock applied at latent pos 0 (strength=%.2f)", live_frame_strength)
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV6] Pixel lock failed: %s", exc)

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

            # Store Stage1 latent tail for next chunk's carry injection (half-res, no re-encode needed)
            n_lat_carry_store = (carry_frames - 1) // 8 + 1
            carry_lat = vid1["samples"][:, :, -n_lat_carry_store:].clone().cpu()

            # ── Stage 2 ──────────────────────────────────────────────────────────
            vid_up = _unpack(LTXVLatentUpsampler.execute(vid1, spatial_upscaler, vae))[0]

            pos2, neg2, lat2, _, _ = _unpack(DirectorGuide.execute(
                pos1_c, neg1_c, vae, vid_up, guide_data,
                motion_guide_data=motion_guide_data,
                model=model_lora,
                ic_lora_name="None",
                ic_lora_strength=1.0,
            ))
            # ── Pixel lock Stage 2: same mechanism, full-resolution ───────────────
            # Stage 2 DirectorGuide also runs Appended Keyframe Guidance, which at
            # 0.42 denoise is strong enough to override the Stage 1 pixel lock.
            # Apply replace_latent_frames at position 0 for Stage 2 as well.
            if live_pixel_frame is not None and chunk_idx > 0 and live_frame_strength > 0.0:
                try:
                    import comfy.utils as _cu2
                    s2_h = lat2["samples"].shape[3] * 32
                    s2_w = lat2["samples"].shape[4] * 32
                    live_s2 = _cu2.common_upscale(
                        live_pixel_frame.movedim(-1, 1),
                        s2_w, s2_h, "bilinear", "disabled"
                    ).movedim(1, -1)
                    live_lat2 = vae.encode(live_s2[:, :, :, :3])
                    if "noise_mask" not in lat2:
                        lat2 = {**lat2, "noise_mask": torch.ones(
                            [1, 1, lat2["samples"].shape[2], lat2["samples"].shape[3], lat2["samples"].shape[4]],
                            device=lat2["samples"].device,
                        )}
                    lat2_samples, lat2_mask = LTXVAddGuide.replace_latent_frames(
                        lat2["samples"], lat2["noise_mask"],
                        live_lat2.to(lat2["samples"].device, lat2["samples"].dtype),
                        latent_idx=0, strength=live_frame_strength,
                    )
                    lat2 = {**lat2, "samples": lat2_samples, "noise_mask": lat2_mask}
                    log.info("[LTXInfiniteDirectorSamplerV6] Pixel lock Stage2 applied at latent pos 0 (strength=%.2f)", live_frame_strength)
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV6] Pixel lock Stage2 failed: %s", exc)

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
            log.info("[LTXInfiniteDirectorSamplerV6] latent %s → decoded %s", lat_shape, list(frames.shape))

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
                log.warning("[LTXInfiniteDirectorSamplerV6] Chunk save failed: %s", exc)

            # ── WAN-style trim: discard the overlap frames from the start ─────────
            # For chunk 2+: the first overlap_frames were generated as a bridge to
            # connect with chunk N. They aren't new content — trim them.
            if overlap_frames > 0:
                frames = frames[overlap_frames:]
                log.info(
                    "[LTXInfiniteDirectorSamplerV6] Trimmed %d overlap frames from chunk %d (WAN-style)",
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
                        "[LTXInfiniteDirectorSamplerV6] Color match chunk %d: "
                        "mean %s→%s std %s→%s",
                        chunk_idx + 1,
                        [f"{v:.3f}" for v in src_mean.tolist()],
                        [f"{v:.3f}" for v in color_ref_mean.tolist()],
                        [f"{v:.3f}" for v in src_std.tolist()],
                        [f"{v:.3f}" for v in color_ref_std.tolist()],
                    )
                except Exception as exc:
                    log.warning("[LTXInfiniteDirectorSamplerV6] Color match failed: %s", exc)

            all_frames.append(frames)
            # Store last pixel frame for next chunk's pixel lock
            live_pixel_frame = frames[-1:].clone().cpu()  # [1, H, W, C]

            # ── Audio: trim overlap region to match video ────────────────────────
            if isinstance(combined_audio, dict) and "waveform" in combined_audio:
                waveform = combined_audio["waveform"]
                audio_sample_rate = combined_audio.get("sample_rate", 44100)
                if overlap_frames > 0:
                    trim_samples = int(overlap_frames * audio_sample_rate / frame_rate)
                    waveform = waveform[:, :, trim_samples:]
                    log.info(
                        "[LTXInfiniteDirectorSamplerV6] Trimmed %d audio samples (overlap %d frames)",
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
                log.info("[LTXInfiniteDirectorSamplerV6] Mixed BG audio (per-chunk, %d chunks)", len(all_bg_waveforms))
            except Exception as exc:
                log.warning("[LTXInfiniteDirectorSamplerV6] BG audio mix failed: %s", exc)

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
                log.warning("[LTXInfiniteDirectorSamplerV6] Wire BG audio mix failed: %s", exc)

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
            log.warning("[LTXInfiniteDirectorSamplerV6] Full video save failed: %s", exc)

        return (full_video, full_audio)


NODE_CLASS_MAPPINGS = {
    "LTXInfiniteDirectorSamplerV6": LTXInfiniteDirectorSamplerV6,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXInfiniteDirectorSamplerV6": "LTX Infinite Director Sampler V6 (Pixel Lock)",
}
