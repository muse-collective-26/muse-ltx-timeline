"""
Muse Collective LTX Timeline V1
================================
Fully standalone LTX 2.3 AV Director + Infinite Sampler.
No WhatDreamsCost dependency — all logic is owned by Muse Collective.

Architecture mirrors V7 (reference-frame latent extension for chunk continuity)
with a clean Muse-branded timeline UI and three audio toggles:
  generate_audio  — LTX generates ambient/sfx from [SOUNDS] prompts
  custom_audio_on — use audio file(s) from the AUDIO timeline track
  motion_guide_on — use motion guide segments from the timeline

Both audio modes can be active simultaneously; their waveforms are mixed.
"""

import json
import logging
import math
import os
import base64
import io as _io

import numpy as np
import torch
import torch.nn.functional as F
import av
from PIL import Image

import folder_paths
import comfy.model_management as mm
import node_helpers

from comfy_extras.nodes_custom_sampler import (
    CFGGuider, KSamplerSelect, BasicScheduler, RandomNoise, SamplerCustomAdvanced,
)
from comfy_extras.nodes_lt import (
    LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent,
)
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

from .muse_prompt_relay import (
    get_raw_tokenizer, map_token_indices, build_segments,
    create_mask_fn, distribute_segment_lengths, convert_to_latent_lengths,
)
from .muse_patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)


# ── Media helpers ─────────────────────────────────────────────────────────────

def _load_image_tensor(seg: dict) -> torch.Tensor:
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?") or b64_str.startswith("/api/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _load_video_tensor(seg: dict, frame_rate: float) -> torch.Tensor:
    file_path = _resolve_path(seg.get("imageFile", ""))
    if not file_path or not os.path.exists(file_path):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    trim_start_frames = float(seg.get("trimStart", 0))
    length_frames = float(seg.get("length", 1))
    start_sec = trim_start_frames / frame_rate
    frames = []
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if stream.time_base:
                seek_pts = int((max(0, start_sec - 0.5)) / float(stream.time_base))
            else:
                seek_pts = int((max(0, start_sec - 0.5)) * av.time_base)
            container.seek(seek_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None and stream.time_base:
                    frame_time = float(frame.pts * stream.time_base)
                if frame_time is None:
                    frame_time = 0.0
                if frame_time < start_sec - 0.01:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= int(length_frames):
                    break
    except Exception as exc:
        log.warning("[MuseDirector] Video extract error: %s", exc)
    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)


def _resolve_path(rel: str) -> str:
    """Try input dir, then muse subdir, then whatdreamscost subdir."""
    if not rel:
        return ""
    input_dir = folder_paths.get_input_directory()
    for base in [input_dir,
                 os.path.join(input_dir, "muse"),
                 os.path.join(input_dir, "whatdreamscost")]:
        p = os.path.join(base, os.path.basename(rel))
        if os.path.exists(p):
            return p
    p = os.path.join(input_dir, rel)
    return p if os.path.exists(p) else ""


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int,
                  method: str, divisible_by: int) -> torch.Tensor:
    def snap(val, div):
        return max(div, (val // div) * div)
    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)
    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor
    t_nchw = tensor.permute(0, 3, 1, 2)
    if method == "stretch to fit":
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)
    elif method == "maintain aspect ratio":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        resized = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    elif method in ("pad", "pad green"):
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_l = (tw - new_w) // 2
        pad_t = (th - new_h) // 2
        if method == "pad green":
            resized = torch.zeros((N, C, th, tw), dtype=t_nchw.dtype, device=t_nchw.device)
            resized[:, 0, :, :] = 102 / 255.0
            resized[:, 1, :, :] = 1.0
            resized[:, :, pad_t:pad_t+new_h, pad_l:pad_l+new_w] = inner
        else:
            resized = F.pad(inner, (pad_l, tw - new_w - pad_l, pad_t, th - new_h - pad_t), value=0)
    elif method == "crop":
        ratio = max(tw / W, th / H)
        new_w = int(W * ratio)
        new_h = int(H * ratio)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner[:, :, top:top+th, left:left+tw]
    else:
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    if crf == 0:
        return tensor
    N, H, W, C = tensor.shape
    h = (H // 2) * 2
    w = (W // 2) * 2
    tensor_bytes = (tensor[:, :h, :w, :] * 255.0).byte().cpu().numpy()
    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        for i in range(N):
            frame = av.VideoFrame.from_ndarray(tensor_bytes[i], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()
        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = [f.to_ndarray(format="rgb24") for f in container_r.decode(video=0)]
        container_r.close()
        if not decoded:
            return tensor
        decoded_np = np.stack(decoded).astype(np.float32) / 255.0
        out = tensor.clone()
        dec_N = min(N, len(decoded))
        out[:dec_N, :h, :w] = torch.from_numpy(decoded_np[:dec_N]).to(tensor.device, tensor.dtype)
        return out
    except Exception as exc:
        log.warning("[MuseDirector] img_compression failed: %s", exc)
        return tensor


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _build_combined_audio(timeline_data_str: str, start_frame: int,
                          duration_frames: int, frame_rate: float) -> dict:
    """Load and mix audio segments from timeline JSON into a single waveform."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32),
                   "sample_rate": target_sr}
    if not timeline_data_str:
        return empty_audio
    try:
        data = json.loads(timeline_data_str)
        audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio
    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        af = seg.get("audioFile", "")
        if af:
            file_path = _resolve_path(af)
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        if not buffer and seg.get("audioB64"):
            b64 = seg["audioB64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                buffer = _io.BytesIO(base64.b64decode(b64))
            except Exception:
                pass
        if not buffer:
            continue
        try:
            clip_frames = []
            with av.open(buffer) as container:
                if not container.streams.audio:
                    continue
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
                for frame in container.decode(stream):
                    for rf in resampler.resample(frame):
                        clip_frames.append(torch.from_numpy(rf.to_ndarray()))
                for rf in resampler.resample(None):
                    clip_frames.append(torch.from_numpy(rf.to_ndarray()))
            if not clip_frames:
                continue
            waveform = torch.cat(clip_frames, dim=1)  # [2, samples]
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))
            if start_frames + length_frames <= start_frame:
                continue
            offset = max(0, start_frame - start_frames)
            trim_start_frames += offset
            length_frames = max(1, length_frames - offset)
            start_frames = max(0, start_frames - start_frame)
            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = min(start_sample_src + length_samples, waveform.shape[1])
            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0:
                continue
            clip_waveform = waveform[:, start_sample_src:end_sample_src]
            start_sample_dst = int(start_frames / frame_rate * target_sr)
            if start_sample_dst >= out_waveform.shape[1]:
                continue
            end_sample_dst = start_sample_dst + actual_length
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
            if actual_length <= 0:
                continue
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform
        except Exception as exc:
            log.warning("[MuseDirector] Audio segment error: %s", exc)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


# ── Conditioning ──────────────────────────────────────────────────────────────

def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    locals_list = [p.strip() for p in local_prompts.split("|")]
    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        log.info("[MuseDirector] No local segments — using global prompt exclusively.")
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(global_prompt))
        return model.clone(), conditioning
    for i, p in enumerate(locals_list):
        if not p:
            locals_list[i] = global_prompt.strip() or "video"
    arch, patch_size, temporal_stride = detect_model_type(model)
    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])
    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)
    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)
    log.info("[MuseDirector] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[MuseDirector] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
    log.info("[MuseDirector] Latent: %d frames, %d tokens/frame, segments: %s",
             latent_frames, tokens_per_frame, effective_lengths)
    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)
    patched = model.clone()
    apply_patches(patched, arch, mask_fn)
    return patched, conditioning


# ── Keyframe / guide logic ────────────────────────────────────────────────────

def _get_guide_attention_entries(conditioning):
    for item in conditioning:
        entries = item[1].get("guide_attention_entries", None)
        if entries is not None:
            return entries
    return []


def _set_guide_attention_entries(conditioning, entries):
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": entries}
    )


def _append_guide_attention_entry(conditioning, pre_filter_count, latent_shape,
                                   attention_strength=1.0, attention_mask=None):
    entries = [*_get_guide_attention_entries(conditioning)]
    entries.append({
        "pre_filter_count": int(pre_filter_count),
        "strength": float(attention_strength),
        "pixel_mask": attention_mask,
        "latent_shape": list(latent_shape),
    })
    return _set_guide_attention_entries(conditioning, entries)


def _build_guide_data(tdata: dict, start_frame: int, duration_frames: int,
                      frame_rate: float, custom_width: int, custom_height: int,
                      resize_method: str, divisible_by: int, img_compression: int,
                      guide_strength_str: str):
    """Parse image/video segments from timeline_data into guide_data dict."""
    guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": frame_rate}
    derived_w, derived_h = custom_width, custom_height

    img_segs = [
        s for s in tdata.get("segments", [])
        if s.get("type", "image") in ("image", "video")
        and (s.get("imageFile") or s.get("imageB64"))
        and int(s.get("start", 0)) < start_frame + duration_frames
        and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
    ]
    img_segs.sort(key=lambda s: s["start"])

    strengths = []
    if guide_strength_str.strip():
        strengths = [float(x.strip()) for x in guide_strength_str.split(",") if x.strip()]

    def snap(val, div):
        return max(div, (val // div) * div)

    for idx, seg in enumerate(img_segs):
        seg_start = int(seg.get("start", 0))
        offset = max(0, start_frame - seg_start)
        if seg.get("type") == "video":
            if offset > 0:
                seg["trimStart"] = float(seg.get("trimStart", 0)) + offset
                seg["length"] = max(1, int(seg.get("length", 1)) - offset)
            tensor = _load_video_tensor(seg, float(frame_rate))
        else:
            tensor = _load_image_tensor(seg)

        src_h, src_w = tensor.shape[1], tensor.shape[2]
        if custom_width > 0 and custom_height > 0:
            tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
        elif custom_width > 0:
            tgt_w = snap(custom_width, divisible_by)
            tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        elif custom_height > 0:
            tgt_h = snap(custom_height, divisible_by)
            tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        else:
            tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)

        if img_compression > 0:
            tensor = _compress_image(tensor, img_compression)

        if idx == 0:
            derived_h = tensor.shape[1]
            derived_w = tensor.shape[2]

        insert_frame = max(0, seg_start - start_frame)
        if seg.get("isEndFrame"):
            insert_frame = max(0, seg_start + int(seg.get("length", 1)) - 1 - start_frame)
        strength = strengths[idx] if idx < len(strengths) else 1.0
        guide_data["images"].append(tensor)
        guide_data["insert_frames"].append(insert_frame)
        guide_data["strengths"].append(float(strength))

    if not guide_data["images"]:
        src_w = derived_w if derived_w > 0 else 768
        src_h = derived_h if derived_h > 0 else 512
        tensor = torch.zeros((1, src_h, src_w, 3), dtype=torch.float32)
        if custom_width > 0 and custom_height > 0:
            tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
        elif custom_width > 0:
            tgt_w = snap(custom_width, divisible_by)
            tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        elif custom_height > 0:
            tgt_h = snap(custom_height, divisible_by)
            tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        else:
            tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)
        guide_data["images"].append(tensor)
        guide_data["insert_frames"].append(0)
        guide_data["strengths"].append(0.0)
        derived_w = tensor.shape[2]
        derived_h = tensor.shape[1]

    return guide_data, derived_w, derived_h


def _apply_guide(pos, neg, vae, video_latent, guide_data, model):
    """Apply appended keyframe guidance (replicates LTXDirectorGuide behaviour)."""
    from comfy_extras.nodes_lt import LTXVAddGuide

    images = guide_data.get("images", [])
    insert_frames = guide_data.get("insert_frames", [])
    strengths = guide_data.get("strengths", [])
    frame_rate = guide_data.get("frame_rate", 24.0)

    if not images:
        return pos, neg, video_latent, model

    lat = video_latent
    pos_out, neg_out = pos, neg

    for img_tensor, insert_frame, strength in zip(images, insert_frames, strengths):
        if strength == 0.0:
            continue
        try:
            result = LTXVAddGuide.add_guide(
                positive=pos_out,
                negative=neg_out,
                vae=vae,
                image=img_tensor,
                latent=lat,
                frame_idx=insert_frame,
                strength=float(strength),
                noise_augmentation=0.0,
            )
            if result and len(result) >= 3:
                pos_out, neg_out, lat = result[0], result[1], result[2]
        except Exception as exc:
            log.warning("[MuseDirector] Guide application failed: %s", exc)

    return pos_out, neg_out, lat, model


def _crop_conditioning(pos, neg, latent):
    """Trim conditioning to match latent frame count (replicates LTXDirectorCrop)."""
    try:
        from comfy_extras.nodes_lt import LTXVCropGuides
        result = LTXVCropGuides.crop(pos, neg, latent)
        if result and len(result) >= 3:
            return result[0], result[1], result[2]
    except Exception:
        pass
    return pos, neg, latent


# ── Audio latent builder ──────────────────────────────────────────────────────

def _build_audio_latent(audio_vae, audio_out, ltxv_length, frame_rate,
                        custom_audio_on, generate_audio):
    """Encode audio waveform into latent space and build noise mask."""
    if audio_vae is None:
        return {}
    inner = getattr(audio_vae, "first_stage_model", audio_vae)
    z_channels = audio_vae.latent_channels
    audio_freq = inner.latent_frequency_bins
    num_audio_latents = inner.num_of_latents_from_frames(ltxv_length, float(frame_rate))

    if not custom_audio_on:
        # Generate from scratch — empty latent, all-ones mask set later
        audio_latents = torch.zeros(
            (1, z_channels, num_audio_latents, audio_freq),
            device=mm.intermediate_device(),
        )
        return {"samples": audio_latents, "type": "audio"}

    # Encode custom audio waveform
    waveform = audio_out["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if hasattr(audio_vae, "first_stage_model"):
        latent_samples = audio_vae.encode(waveform.movedim(1, -1))
    else:
        latent_samples = audio_vae.encode({
            "waveform": waveform,
            "sample_rate": audio_out["sample_rate"],
        })
    if latent_samples.numel() == 0:
        raise ValueError("Encoded audio latent is empty.")

    B, C, F_len, H_len = latent_samples.shape
    # All-ones gap mask: entire duration is available for generation
    # (V7's override logic will force all-ones anyway when generate_audio=True)
    gap_mask = torch.ones((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)

    log.info("[MuseDirector] Generated custom audio latent with dynamic noise mask.")
    return {"samples": latent_samples, "type": "audio", "noise_mask": gap_mask}


# ── Chunk save ────────────────────────────────────────────────────────────────

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
    log.info("[MuseDirector] Saved: %s", path)


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


def _unpack(result):
    return result.args if hasattr(result, "args") else result


# ── Node ──────────────────────────────────────────────────────────────────────

class MuseDirectorSamplerV1:
    """
    Muse Collective LTX Timeline V1

    Fully standalone LTX 2.3 AV Director + Infinite Sampler.
    No WhatDreamsCost dependency.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model":            ("MODEL",),
                "clip":             ("CLIP",),
                "audio_vae":        ("VAE",),
                "vae":              ("VAE",),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL",),

                # Timeline hidden widgets (managed by JS)
                "start_second":     ("FLOAT",  {"default": 0.0,   "min": 0.0, "max": 3600.0, "step": 0.01}),
                "end_second":       ("FLOAT",  {"default": 10.0,  "min": 0.0, "max": 3600.0, "step": 0.01}),
                "duration_seconds": ("FLOAT",  {"default": 10.0,  "min": 0.0, "max": 3600.0, "step": 0.01}),
                "start_frame":      ("INT",    {"default": 0,     "min": 0,   "max": 86400}),
                "end_frame":        ("INT",    {"default": 240,   "min": 0,   "max": 86400}),
                "duration_frames":  ("INT",    {"default": 240,   "min": 1,   "max": 86400}),
                "timeline_data":    ("STRING", {"default": "{}"}),
                "local_prompts":    ("STRING", {"default": ""}),
                "segment_lengths":  ("STRING", {"default": ""}),
                "global_prompt":    ("STRING", {"multiline": True, "default": ""}),
                "guide_strength":   ("STRING", {"default": ""}),
                "epsilon":          ("FLOAT",  {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),

                # Generation settings
                "frame_rate":    ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "display_mode":  (["seconds", "frames"], {"default": "seconds"}),
                "custom_width":  ("INT",   {"default": 960,  "min": 64,  "max": 4096, "step": 32}),
                "custom_height": ("INT",   {"default": 544,  "min": 64,  "max": 4096, "step": 32}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "crop", "pad"],
                                  {"default": "maintain aspect ratio"}),
                "divisible_by":  ("INT",   {"default": 32, "min": 1, "max": 256}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 51}),

                # Audio toggles
                "generate_audio":   ("BOOLEAN", {"default": True,
                                                  "tooltip": "LTX generates ambient/sfx audio from [SOUNDS] prompts."}),
                "custom_audio_on":  ("BOOLEAN", {"default": False,
                                                  "tooltip": "Use audio file(s) from the AUDIO timeline track."}),
                "motion_guide_on":  ("BOOLEAN", {"default": True,
                                                  "tooltip": "Use motion guide segments from the timeline."}),

                # Chunking
                "chunk_duration_seconds": ("FLOAT", {"default": 10.0, "min": 2.0, "max": 120.0, "step": 0.1}),
                "auto_chunk_threshold":   ("FLOAT", {"default": 10.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "carry_frames":     ("INT",   {"default": 73,   "min": 1,   "max": 240, "step": 1,
                                               "tooltip": "Reference frames from previous chunk locked at chunk start. 73 ≈ 3s at 24fps."}),
                "carry_strength":   ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "crossfade_frames": ("INT",   {"default": 0,    "min": 0,   "max": 120, "step": 1}),

                # IC-LoRA
                "ic_lora_name":     (["None"] + loras, {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # Sampling
                "stage1_steps":   ("INT",   {"default": 8,    "min": 1, "max": 50}),
                "stage2_steps":   ("INT",   {"default": 4,    "min": 1, "max": 50}),
                "stage2_denoise": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg":            ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed":           ("INT",   {"default": 42,   "min": 0,  "max": 0xFFFFFFFFFFFFFFFF}),
                "control_after_generate": (["fixed", "increment", "decrement", "randomize"],
                                           {"default": "fixed"}),
                "filename_prefix": ("STRING", {"default": "muse"}),
                "bg_volume":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),

                # Timeline UI placeholder (hidden by JS)
                "timeline_ui": ("STRING", {"default": ""}),
            },
            "optional": {
                "bg_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE")
    RETURN_NAMES = ("last_chunk_frames", "audio", "stage1_frames")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"
    DESCRIPTION = (
        "Muse Collective LTX Timeline V1 — fully standalone LTX 2.3 AV Director + Infinite Sampler. "
        "Reference-frame latent extension for seamless multi-chunk generation. "
        "No WhatDreamsCost dependency."
    )

    def execute(
        self,
        model, clip, audio_vae, vae, spatial_upscaler,
        start_second, end_second, duration_seconds,
        start_frame, end_frame, duration_frames,
        timeline_data, local_prompts, segment_lengths,
        global_prompt, guide_strength, epsilon,
        frame_rate, display_mode, custom_width, custom_height,
        resize_method, divisible_by, img_compression,
        generate_audio, custom_audio_on, motion_guide_on,
        chunk_duration_seconds, auto_chunk_threshold,
        carry_frames, carry_strength, crossfade_frames,
        ic_lora_name, ic_lora_strength,
        stage1_steps, stage2_steps, stage2_denoise, cfg,
        seed, control_after_generate, filename_prefix,
        bg_volume=1.0, bg_audio=None, timeline_ui="",
    ):
        if not isinstance(ic_lora_name, str):
            ic_lora_name = "None"
        stage1_steps = max(1, int(stage1_steps))
        stage2_steps = max(1, int(stage2_steps))

        # Parse timeline
        try:
            tdata = json.loads(timeline_data) if timeline_data and timeline_data.strip() not in ("", "{}") else {}
        except Exception:
            tdata = {}

        # Sync global_prompt from timeline if not connected
        if not global_prompt:
            global_prompt = tdata.get("global_prompt", "")

        log.info("[MuseDirector] global_prompt: %r", global_prompt)

        total_duration = end_second - start_second
        use_chunks = (auto_chunk_threshold <= 0.0) or (total_duration > auto_chunk_threshold)
        effective_chunk = chunk_duration_seconds if use_chunks else total_duration
        mode = "chunked" if use_chunks else "single-pass"
        log.info("[MuseDirector] %s mode — %.1fs total", mode, total_duration)

        chunks = []
        t = start_second
        while t < end_second - 0.01:
            end = min(t + effective_chunk, end_second)
            chunks.append((t, end))
            t = end
        log.info("[MuseDirector] %d chunk(s), carry_frames=%d", len(chunks), carry_frames)

        output_dir = folder_paths.get_output_directory()
        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_full.mp4")):
            counter += 1

        # Pre-load bg audio segments from timeline
        _bg_tracks = []
        try:
            _tl_bg_vol = float(tdata.get("bgAudioVolume", 1.0))
            for _bseg in tdata.get("bgAudioSegments", []):
                _af = _bseg.get("audioFile", "")
                if not _af:
                    continue
                _ap = _resolve_path(_af)
                if not _ap:
                    continue
                try:
                    import soundfile as _sf
                    _data, _sr = _sf.read(_ap, dtype="float32", always_2d=True)
                    _bg_tracks.append((
                        torch.from_numpy(_data.T),
                        _sr,
                        float(_bseg.get("start", 0)),
                        float(_bseg.get("length", 1)),
                        float(_bseg.get("trimStart", 0)),
                        _tl_bg_vol,
                    ))
                    log.info("[MuseDirector] BG pre-loaded: %s", _af)
                except Exception as exc:
                    log.warning("[MuseDirector] BG pre-load failed: %s", exc)
        except Exception as exc:
            log.warning("[MuseDirector] BG pre-parse failed: %s", exc)

        all_frames = []
        all_s1_frames = []
        all_waveforms = []
        all_bg_waveforms = []
        audio_sample_rate = 44100
        live_pixel_frames = None
        color_ref_mean = None
        color_ref_std = None

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_seed = seed
            s2_seed = seed - 1

            raw_s_fr = int(chunk_start * frame_rate)
            e_fr = int(chunk_end * frame_rate)

            if chunk_idx > 0 and live_pixel_frames is not None:
                ref_pixel_count = min(carry_frames, live_pixel_frames.shape[0], raw_s_fr)
            else:
                ref_pixel_count = 0

            overlap_frames = ref_pixel_count
            s_fr = raw_s_fr - ref_pixel_count
            gen_start = s_fr / frame_rate

            log.info(
                "[MuseDirector] Chunk %d/%d  %.2f→%.2f s  (ref=%d px → %.3fs locked)  seed=%d",
                chunk_idx + 1, len(chunks), chunk_start, chunk_end,
                ref_pixel_count, gen_start, chunk_seed,
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

            chunk_dur = e_fr / frame_rate - gen_start

            # ── Build guide data ─────────────────────────────────────────────
            chunk_s_frame = s_fr
            chunk_dur_frames = e_fr - s_fr
            guide_data, derived_w, derived_h = _build_guide_data(
                tdata, chunk_s_frame, chunk_dur_frames,
                frame_rate, s1_w, s1_h, resize_method, divisible_by,
                img_compression, guide_strength,
            )
            guide_data["timeline_data"] = timeline_data
            guide_data["start_frame"] = chunk_s_frame
            guide_data["duration_frames"] = chunk_dur_frames
            guide_data["resize_method"] = resize_method

            # ── Build conditioning ───────────────────────────────────────────
            patched_model, positive = _encode_relay(
                model, clip, pre_latent,
                global_prompt, local_prompts, segment_lengths, epsilon,
            )

            # ── Build audio waveform + latent ────────────────────────────────
            combined_audio = _build_combined_audio(
                timeline_data, chunk_s_frame, ltxv_len, float(frame_rate),
            )
            audio_latent = _build_audio_latent(
                audio_vae, combined_audio, ltxv_len, frame_rate,
                custom_audio_on, generate_audio,
            )

            # Force all-ones mask when generate_audio is on
            if generate_audio and "samples" in audio_latent:
                s = audio_latent["samples"]
                ones = torch.ones(s.shape[0], s.shape[2], s.shape[3],
                                  dtype=torch.float32, device=s.device)
                audio_latent = {**audio_latent, "noise_mask": ones}
                log.info("[MuseDirector] generate_audio=True — all-ones mask.")

            # ── Stage 1 ─────────────────────────────────────────────────────
            zero_neg = _zero_out_conditioning(positive)
            pos1, neg1 = _unpack(LTXVConditioning.execute(positive, zero_neg, frame_rate))

            # Apply image guides
            pos1, neg1, lat1 = pre_latent, None, pre_latent
            try:
                pos1, neg1, lat1, patched_model = _apply_guide(
                    positive, zero_neg, vae, pre_latent, guide_data, patched_model,
                )
                pos1, neg1 = _unpack(LTXVConditioning.execute(pos1, neg1, frame_rate))
            except Exception as exc:
                log.warning("[MuseDirector] Guide application failed, using plain conditioning: %s", exc)
                pos1, neg1 = _unpack(LTXVConditioning.execute(positive, zero_neg, frame_rate))
                lat1 = pre_latent

            # Reference-frame lock
            ref_t = 0
            if live_pixel_frames is not None and ref_pixel_count > 0:
                try:
                    import comfy.utils as _cu
                    ref_px = live_pixel_frames[-ref_pixel_count:]
                    ref_s1 = _cu.common_upscale(
                        ref_px.movedim(-1, 1), s1_w, s1_h, "bilinear", "disabled"
                    ).movedim(1, -1)
                    ref_lat = vae.encode(ref_s1[:, :, :, :3])
                    ref_t = ref_lat.shape[2]
                    lat1_samples = lat1["samples"].clone()
                    cap = min(ref_t, lat1_samples.shape[2])
                    lat1_samples[:, :, :cap] = ref_lat[:, :, :cap].to(lat1_samples.device, lat1_samples.dtype)
                    ref_mask = torch.ones(
                        [1, 1, lat1_samples.shape[2], 1, 1],
                        device=lat1_samples.device, dtype=torch.float32,
                    )
                    ref_mask[:, :, :cap] = 0.0
                    lat1 = {**lat1, "samples": lat1_samples, "noise_mask": ref_mask}
                    log.info("[MuseDirector] Reference lock S1: %d px → %d latent frames frozen",
                             ref_pixel_count, cap)
                except Exception as exc:
                    log.warning("[MuseDirector] Reference lock S1 failed: %s", exc)

            guider1 = _unpack(CFGGuider.execute(patched_model, pos1, neg1, cfg))[0]
            sampler  = _unpack(KSamplerSelect.execute("euler"))[0]
            sigmas1  = _unpack(BasicScheduler.execute(patched_model, "linear_quadratic", stage1_steps, 1.0))[0]
            noise1   = _unpack(RandomNoise.execute(chunk_seed))[0]
            has_audio = "samples" in audio_latent
            if has_audio:
                av1_lat = _unpack(LTXVConcatAVLatent.execute(lat1, audio_latent))[0]
            else:
                av1_lat = lat1
            out1, _ = _unpack(SamplerCustomAdvanced.execute(noise1, guider1, sampler, sigmas1, av1_lat))
            if has_audio:
                out1_vid, aud1 = _unpack(LTXVSeparateAVLatent.execute(out1))
            else:
                out1_vid = out1
                aud1 = {}

            pos1_c, neg1_c, vid1 = _crop_conditioning(pos1, neg1, out1_vid)

            # Stage1 decode for comparison
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
                log.warning("[MuseDirector] Stage1 decode failed: %s", exc)

            # ── Stage 2 ─────────────────────────────────────────────────────
            vid_up = _unpack(LTXVLatentUpsampler.execute(vid1, spatial_upscaler, vae))[0]

            try:
                pos2, neg2, lat2, _, = _apply_guide(
                    pos1_c, neg1_c, vae, vid_up, guide_data, patched_model,
                )[:4]
                pos2, neg2 = _unpack(LTXVConditioning.execute(pos2, neg2, frame_rate))
            except Exception as exc:
                log.warning("[MuseDirector] Stage2 guide failed: %s", exc)
                pos2, neg2 = _unpack(LTXVConditioning.execute(pos1_c, neg1_c, frame_rate))
                lat2 = vid_up

            # Reference-frame lock Stage2
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
                    log.info("[MuseDirector] Reference lock S2: %d latent frames frozen", cap2)
                except Exception as exc:
                    log.warning("[MuseDirector] Reference lock S2 failed: %s", exc)

            guider2  = _unpack(CFGGuider.execute(patched_model, pos2, neg2, cfg))[0]
            sigmas2  = _unpack(BasicScheduler.execute(patched_model, "linear_quadratic",
                                                       stage2_steps, stage2_denoise))[0]
            noise2   = _unpack(RandomNoise.execute(s2_seed))[0]
            if has_audio:
                av2_lat = _unpack(LTXVConcatAVLatent.execute(lat2, aud1))[0]
            else:
                av2_lat = lat2
            out2, _ = _unpack(SamplerCustomAdvanced.execute(noise2, guider2, sampler, sigmas2, av2_lat))
            if has_audio:
                out2_nosemask = {k: v for k, v in out2.items() if k != "noise_mask"}
                out2_vid, _ = _unpack(LTXVSeparateAVLatent.execute(out2_nosemask))
            else:
                out2_vid = out2
            _, _, vid_final = _crop_conditioning(pos2, neg2, out2_vid)

            # ── VAE decode ───────────────────────────────────────────────────
            lat_shape = list(vid_final["samples"].shape)
            frames = vae.decode(vid_final["samples"])
            log.info("[MuseDirector] latent %s → decoded %s", lat_shape, list(frames.shape))
            if frames.ndim == 5:
                frames = frames.squeeze(0)
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)

            out_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_{chunk_idx:03d}.mp4")
            try:
                _save_chunk_mp4(frames, frame_rate, out_path)
            except Exception as exc:
                log.warning("[MuseDirector] Chunk save failed: %s", exc)

            if overlap_frames > 0:
                frames = frames[overlap_frames:]

            if all_frames and (frames.shape[1] != all_frames[0].shape[1] or frames.shape[2] != all_frames[0].shape[2]):
                import comfy.utils as _cu
                frames = _cu.common_upscale(
                    frames.movedim(-1, 1), all_frames[0].shape[2], all_frames[0].shape[1],
                    "lanczos", "disabled"
                ).movedim(1, -1).clamp(0, 1)

            nominal_frames = int(round((chunk_end - chunk_start) * frame_rate))
            if frames.shape[0] > nominal_frames:
                frames = frames[:nominal_frames]

            # Color match
            if chunk_idx == 0:
                _ref_frames = frames.float()
                color_ref_mean = _ref_frames.mean(dim=(0, 1, 2))
                color_ref_std  = _ref_frames.std(dim=(0, 1, 2)).clamp(min=1e-5)
            elif all_frames:
                try:
                    src = frames.float()
                    src_mean = src.mean(dim=(0, 1, 2))
                    src_std  = src.std(dim=(0, 1, 2)).clamp(min=1e-5)
                    corrected = (src - src_mean) / src_std * color_ref_std + color_ref_mean
                    corrected = corrected.clamp(0.0, 1.0)
                    n_blend = min(int(frame_rate), frames.shape[0])
                    blend = torch.linspace(1.0, 0.5, n_blend, device=frames.device)
                    corrected[:n_blend] = (
                        corrected[:n_blend] * blend[:, None, None, None]
                        + src[:n_blend] * (1.0 - blend[:, None, None, None])
                    )
                    frames = corrected.to(all_frames[-1].dtype)
                except Exception as exc:
                    log.warning("[MuseDirector] Color match failed: %s", exc)

            all_frames.append(frames)
            cat_so_far = torch.cat(all_frames, dim=0)
            live_pixel_frames = cat_so_far[-carry_frames:].clone().cpu()

            # ── Audio source selection ────────────────────────────────────────
            decoded_wav = None
            if generate_audio and has_audio and isinstance(aud1, dict) and "samples" in aud1:
                try:
                    inner_vae = audio_vae.first_stage_model
                    aud_samples = aud1["samples"].cpu().float()
                    decoded_wav = inner_vae.decode(aud_samples)
                    if decoded_wav.shape[1] == 1:
                        decoded_wav = decoded_wav.expand(-1, 2, -1)
                    decoded_wav = decoded_wav.cpu().float()
                    audio_sr = getattr(inner_vae, "output_sample_rate", 44100)
                    if audio_sr != 44100:
                        import torchaudio
                        decoded_wav = torchaudio.functional.resample(decoded_wav, audio_sr, 44100)
                        audio_sr = 44100
                    log.info("[MuseDirector] Decoded generated audio: shape %s sr=%d",
                             list(decoded_wav.shape), audio_sr)
                except Exception as exc:
                    log.warning("[MuseDirector] Audio decode failed: %s", exc)
                    decoded_wav = None

            custom_wav = combined_audio.get("waveform") if (custom_audio_on and isinstance(combined_audio, dict)) else None
            audio_sr = combined_audio.get("sample_rate", 44100) if isinstance(combined_audio, dict) else 44100

            if decoded_wav is not None and custom_wav is not None:
                min_len = min(decoded_wav.shape[-1], custom_wav.shape[-1])
                mixed = decoded_wav[..., :min_len] + custom_wav[..., :min_len]
                combined_audio = {"waveform": mixed, "sample_rate": audio_sr}
                log.info("[MuseDirector] Audio: mixed generated + custom")
            elif decoded_wav is not None:
                combined_audio = {"waveform": decoded_wav, "sample_rate": audio_sr}
                log.info("[MuseDirector] Audio: generated only")
            elif custom_wav is not None:
                combined_audio = {"waveform": custom_wav, "sample_rate": audio_sr}
                log.info("[MuseDirector] Audio: custom only")
            else:
                log.info("[MuseDirector] Audio: silence")

            # Trim + cap audio
            if isinstance(combined_audio, dict) and "waveform" in combined_audio:
                waveform = combined_audio["waveform"]
                audio_sample_rate = combined_audio.get("sample_rate", 44100)
                if overlap_frames > 0:
                    trim_samples = int(overlap_frames * audio_sample_rate / frame_rate)
                    waveform = waveform[:, :, trim_samples:]
                nominal_samples = int((chunk_end - chunk_start) * audio_sample_rate)
                if waveform.shape[-1] > nominal_samples:
                    waveform = waveform[:, :, :nominal_samples]
                all_waveforms.append(waveform)

            # Per-chunk bg audio
            if _bg_tracks:
                chunk_speech_len = waveform.shape[-1] if (isinstance(combined_audio, dict) and "waveform" in combined_audio) else int((chunk_end - gen_start) * audio_sample_rate)
                bg_chunk_out = torch.zeros((2, chunk_speech_len), dtype=torch.float32)
                for (_bg_raw, _bg_sr, _seg_start_fr, _seg_len_fr, _seg_trim_fr, _vol) in _bg_tracks:
                    _bg_w = _bg_raw
                    if _bg_sr != audio_sample_rate:
                        import torchaudio as _ta2
                        _bg_w = _ta2.functional.resample(_bg_w.unsqueeze(0), _bg_sr, audio_sample_rate).squeeze(0)
                    _offset_fr = max(0.0, s_fr - _seg_start_fr)
                    _eff_trim_fr = _seg_trim_fr + _offset_fr
                    _eff_len_fr = max(1.0, _seg_len_fr - _offset_fr)
                    _dst_start_fr = max(0.0, _seg_start_fr - s_fr)
                    _src_start = int(_eff_trim_fr / frame_rate * audio_sample_rate)
                    _src_end = min(_src_start + int(_eff_len_fr / frame_rate * audio_sample_rate), _bg_w.shape[-1])
                    _dst_start = int(_dst_start_fr / frame_rate * audio_sample_rate)
                    if _src_end > _src_start and _dst_start < chunk_speech_len:
                        _clip = _bg_w[:, _src_start:_src_end]
                        if overlap_frames > 0:
                            _bg_trim = int(overlap_frames * audio_sample_rate / frame_rate)
                            _clip = _clip[:, _bg_trim:]
                        _avail = min(_clip.shape[-1], chunk_speech_len - _dst_start)
                        if _clip.shape[0] == 1:
                            _clip = _clip.expand(2, -1)
                        elif _clip.shape[0] > 2:
                            _clip = _clip[:2, :]
                        bg_chunk_out[:, _dst_start:_dst_start + _avail] += _clip[:, :_avail] * _vol
                all_bg_waveforms.append(bg_chunk_out.unsqueeze(0))

            mm.soft_empty_cache()

        # ── Assemble output ────────────────────────────────────────────────────
        if not all_frames:
            all_frames = [torch.zeros((1, custom_height, custom_width, 3))]

        if all_waveforms:
            full_audio = {"waveform": torch.cat(all_waveforms, dim=2), "sample_rate": audio_sample_rate}
        else:
            full_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}

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
            except Exception as exc:
                log.warning("[MuseDirector] BG audio mix failed: %s", exc)

        if bg_audio is not None and "waveform" in bg_audio:
            try:
                lip_w = full_audio["waveform"]
                bg_w = bg_audio["waveform"].clone()
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
                full_audio = {"waveform": lip_w + bg_w.to(lip_w.device, lip_w.dtype) * bg_volume,
                              "sample_rate": audio_sample_rate}
            except Exception as exc:
                log.warning("[MuseDirector] Wire BG audio mix failed: %s", exc)

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
            log.warning("[MuseDirector] Full video save failed: %s", exc)

        s1_video = torch.cat(all_s1_frames, dim=0) if all_s1_frames else full_video
        return (full_video, full_audio, s1_video)


NODE_CLASS_MAPPINGS = {
    "MuseDirectorSamplerV1": MuseDirectorSamplerV1,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseDirectorSamplerV1": "Muse Collective LTX Timeline V1",
}
