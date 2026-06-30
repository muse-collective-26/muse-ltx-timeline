"""
Muse Collective — MuseGuide
============================
Standalone replacement for LTXDirectorGuide / LTXDirectorCropGuides.
No WhatDreamsCost dependency — all logic is owned by Muse Collective.

Handles:
  - Image keyframe guidance (reference image locked to latent start)
  - Motion video segments via IC-LoRA
  - Retake mode (preserve base video, regenerate a region)
  - Conditioning crop after sampling

Architecture derived from WhatDreamsCost LTXDirectorGuide with permission-compatible
reimplementation using only ComfyUI core (comfy_extras.nodes_lt.LTXVAddGuide).
"""

import json
import logging
import math
import os

import av
import numpy as np
import torch
import torch.nn.functional as F

import comfy
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
from comfy_extras import nodes_lt

log = logging.getLogger(__name__)


# ── Image resize helper ───────────────────────────────────────────────────────

def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int,
                  method: str, divisible_by: int = 1) -> torch.Tensor:
    """Resize [N, H, W, C] float32 tensor to target dimensions."""
    def snap(val, div):
        div = max(1, int(div))
        return max(div, (val // div) * div)

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)

    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor

    t = tensor.permute(0, 3, 1, 2)  # NHWC → NCHW

    if method == "stretch to fit":
        r = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)

    elif method in ("maintain aspect ratio",):
        ratio = min(tw / W, th / H)
        nw = snap(int(W * ratio), divisible_by)
        nh = snap(int(H * ratio), divisible_by)
        r = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)

    elif method in ("pad", "pad green"):
        ratio = min(tw / W, th / H)
        nw = snap(int(W * ratio), divisible_by)
        nh = snap(int(H * ratio), divisible_by)
        inner = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        pl = (tw - nw) // 2
        pt = (th - nh) // 2
        if method == "pad green":
            r = torch.zeros((N, C, th, tw), dtype=t.dtype, device=t.device)
            r[:, 0] = 102 / 255.0
            r[:, 1] = 1.0
            r[:, :, pt:pt + nh, pl:pl + nw] = inner
        else:
            r = F.pad(inner, (pl, tw - nw - pl, pt, th - nh - pt), value=0)

    elif method == "crop":
        ratio = max(tw / W, th / H)
        nw, nh = int(W * ratio), int(H * ratio)
        inner = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        l = (nw - tw) // 2
        top = (nh - th) // 2
        r = inner[:, :, top:top + th, l:l + tw]

    else:
        r = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)

    return r.permute(0, 2, 3, 1)  # NCHW → NHWC


# ── Latent helpers ────────────────────────────────────────────────────────────

def _clone_noise_mask(latent, latent_image):
    if "noise_mask" in latent and latent["noise_mask"] is not None:
        return latent["noise_mask"].clone()
    B, _, F, _, _ = latent_image.shape
    return torch.ones((B, 1, F, 1, 1), dtype=torch.float32, device=latent_image.device)


def _resize_latent_spatial(latent_image, noise_mask, w, h, method):
    b, c, f, lh, lw = latent_image.shape
    if lw == w and lh == h:
        return latent_image, noise_mask
    lat4 = latent_image.permute(0, 2, 1, 3, 4).reshape(b * f, c, lh, lw)
    lat4 = comfy.utils.common_upscale(lat4, w, h, method, "disabled")
    latent_image = lat4.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)
    if noise_mask is not None and (noise_mask.shape[-1] > 1 or noise_mask.shape[-2] > 1):
        mb, mc, mf, mh, mw = noise_mask.shape
        m4 = noise_mask.permute(0, 2, 1, 3, 4).reshape(mb * mf, mc, mh, mw)
        m4 = comfy.utils.common_upscale(m4, w, h, method, "disabled")
        noise_mask = m4.reshape(mb, mf, mc, h, w).permute(0, 2, 1, 3, 4)
    return latent_image, noise_mask


def _ceil_to_multiple(value, multiple):
    multiple = max(1, int(multiple))
    return int(math.ceil(value / multiple) * multiple)


def _snap_latent_to_downscale(latent_image, noise_mask, downscale_factor, method):
    factor = int(max(1, round(float(downscale_factor))))
    if factor <= 1:
        return latent_image, noise_mask
    _, _, _, h, w = latent_image.shape
    nw = _ceil_to_multiple(w, factor)
    nh = _ceil_to_multiple(h, factor)
    if nw == w and nh == h:
        return latent_image, noise_mask
    log.warning("[MuseGuide] Auto-snapping latent %sx%s → %sx%s for IC-LoRA downscale %s", w, h, nw, nh, factor)
    return _resize_latent_spatial(latent_image, noise_mask, nw, nh, method)


def _dilate_latent(latent: dict, horizontal_scale: int, vertical_scale: int) -> dict:
    if horizontal_scale == 1 and vertical_scale == 1:
        return latent
    samples = latent["samples"]
    mask = latent.get("noise_mask", None)
    ds = samples.shape[:3] + (samples.shape[3] * vertical_scale, samples.shape[4] * horizontal_scale)
    dilated = torch.zeros(ds, device=samples.device, dtype=samples.dtype)
    dilated[..., ::vertical_scale, ::horizontal_scale] = samples
    dms = (dilated.shape[0], 1, dilated.shape[2], dilated.shape[3], dilated.shape[4])
    dmask = torch.full(dms, -1.0, device=samples.device, dtype=samples.dtype)
    dmask[..., ::vertical_scale, ::horizontal_scale] = (mask if mask is not None else 1.0)
    return {"samples": dilated, "noise_mask": dmask}


# ── IC-LoRA helpers ───────────────────────────────────────────────────────────

def _load_lora_model_only(model, ic_lora_name, strength):
    lora_path = folder_paths.get_full_path_or_raise("loras", ic_lora_name)
    lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
    try:
        downscale = float(metadata["reference_downscale_factor"])
    except Exception:
        downscale = 1.0
        log.warning("[MuseGuide] Could not read reference_downscale_factor from %s, using 1.0", ic_lora_name)
    if strength == 0:
        return model, downscale
    model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
    return model_lora, downscale


def _encode_video_iclora(vae, latent_w, latent_h, images, scale_factors,
                          latent_downscale_factor, crop, use_tiled, tile_size, tile_overlap,
                          resize_method="crop"):
    tsf, wsf, hsf = scale_factors
    keep = ((images.shape[0] - 1) // tsf) * tsf + 1
    images = images[:keep]
    tw = max(8, int(latent_w * wsf / latent_downscale_factor))
    th = max(8, int(latent_h * hsf / latent_downscale_factor))
    if resize_method == "maintain aspect ratio":
        resize_method = "pad"
    pixels = _resize_image(images, tw, th, resize_method, divisible_by=1)
    enc = pixels[:, :, :, :3]
    if use_tiled:
        guide_lat = vae.encode_tiled(enc, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
    else:
        guide_lat = vae.encode(enc)
    return pixels, guide_lat


# ── Guide attention helpers ───────────────────────────────────────────────────

def _get_guide_attention_entries(conditioning):
    for item in conditioning:
        entries = item[1].get("guide_attention_entries")
        if entries is not None:
            return entries
    return []


def _set_guide_attention_entries(conditioning, entries):
    return node_helpers.conditioning_set_values(conditioning, {"guide_attention_entries": entries})


def _append_guide_attention_entry(conditioning, pre_filter_count, latent_shape, attention_strength=1.0):
    entries = [*_get_guide_attention_entries(conditioning)]
    entries.append({
        "pre_filter_count": int(pre_filter_count),
        "strength": float(attention_strength),
        "pixel_mask": None,
        "latent_shape": list(latent_shape),
    })
    return _set_guide_attention_entries(conditioning, entries)


# ── Video loading ─────────────────────────────────────────────────────────────

class _ResampleGuideFrames:
    @staticmethod
    def execute(images, source_fps, target_fps, target_num_frames, mode):
        if images is None:
            return images
        n = int(images.shape[0])
        target_num_frames = int(target_num_frames)
        if n <= 1:
            return images.repeat(target_num_frames, 1, 1, 1) if target_num_frames > 1 else images
        source_fps = float(max(0.001, source_fps))
        target_fps = float(max(0.001, target_fps))
        if target_num_frames <= 0:
            duration = (n - 1) / source_fps
            target_num_frames = max(1, int(round(duration * target_fps)) + 1)
        if target_num_frames == n and abs(target_fps - source_fps) < 1e-6:
            return images
        positions = torch.linspace(0, n - 1, target_num_frames, device=images.device, dtype=torch.float32)
        if mode == "nearest":
            idx = torch.round(positions).long().clamp(0, n - 1)
            return images.index_select(0, idx)
        idx0 = torch.floor(positions).long().clamp(0, n - 1)
        idx1 = torch.ceil(positions).long().clamp(0, n - 1)
        alpha = (positions - idx0.to(positions.dtype)).view(-1, 1, 1, 1)
        f0 = images.index_select(0, idx0).float()
        f1 = images.index_select(0, idx1).float()
        return (f0 * (1 - alpha) + f1 * alpha).to(images.dtype)


def _resolve_video_path(video_file):
    if os.path.isabs(str(video_file)) and os.path.exists(str(video_file)):
        return str(video_file)
    input_dir = folder_paths.get_input_directory()
    candidate = os.path.join(input_dir, str(video_file))
    if os.path.exists(candidate):
        return candidate
    try:
        annotated = folder_paths.get_annotated_filepath(str(video_file))
        if annotated and os.path.exists(annotated):
            return annotated
    except Exception:
        pass
    raise FileNotFoundError(f"[MuseGuide] Could not find video: {video_file}")


def _load_motion_video_frames(video_file, trim_start_frames, length_frames, director_fps, resample_mode="nearest"):
    path = _resolve_video_path(video_file)
    target_fps = max(1.0, float(director_fps))
    start_s = max(0.0, float(trim_start_frames) / target_fps)
    dur_s = max(0.0, float(length_frames) / target_fps)
    end_s = start_s + dur_s if dur_s > 0 else None
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    try:
        source_fps = float(stream.average_rate) if stream.average_rate else float(stream.base_rate)
    except Exception:
        source_fps = target_fps
    if source_fps <= 0:
        source_fps = target_fps
    if start_s > 0:
        try:
            seek_pts = int(max(0, start_s - 0.5) / float(stream.time_base)) if stream.time_base else int(max(0, start_s - 0.5) * av.time_base)
            container.seek(seek_pts, stream=stream, backward=True)
        except Exception as e:
            log.warning("[MuseGuide] Seek failed: %s", e)
    frames = []
    decoded = 0
    for frame in container.decode(stream):
        t = frame.time if frame.time is not None else (
            float(frame.pts * stream.time_base) if frame.pts is not None and stream.time_base else float(decoded / source_fps)
        )
        decoded += 1
        if t < start_s - 0.01:
            continue
        if end_s is not None and t >= end_s:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if not frames:
        raise ValueError(f"[MuseGuide] No frames decoded from: {video_file}")
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    images = torch.from_numpy(frames_np)
    target_count = max(1, int(round(float(length_frames))))
    return _ResampleGuideFrames.execute(images, source_fps, target_fps, target_count, resample_mode)


# ── Conditioning crop helpers ─────────────────────────────────────────────────

def _cond_get_value(conditioning, key, default=None):
    for item in conditioning:
        val = item[1].get(key)
        if val is not None:
            return val
    return default


def _get_exact_crop_count(conditioning):
    val = _cond_get_value(conditioning, "nghtdrp_guide_crop_latent_frames")
    if val is not None:
        try:
            return max(0, int(val))
        except Exception:
            return 0
    kf = _cond_get_value(conditioning, "keyframe_idxs")
    if kf is None:
        return 0
    try:
        return int(torch.unique(kf[:, 0, :, 0]).shape[0])
    except Exception:
        return 0


def _get_noise_mask_for_crop(latent):
    img = latent["samples"]
    mask = latent.get("noise_mask")
    if mask is None:
        B, _, F, _, _ = img.shape
        return torch.ones((B, 1, F, 1, 1), dtype=torch.float32, device=img.device)
    return mask.clone()


# ── Main node ─────────────────────────────────────────────────────────────────

class MuseGuide:
    """
    Muse Collective replacement for LTXDirectorGuide.
    Encodes reference images and motion video segments as keyframes into the latent.
    Supports IC-LoRA motion guidance and retake mode.
    No WhatDreamsCost dependency.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras") or ["put_ic_lora_in_ComfyUI_models_loras"]
        return {
            "required": {
                "positive":   ("CONDITIONING",),
                "negative":   ("CONDITIONING",),
                "vae":        ("VAE",),
                "latent":     ("LATENT",),
                "guide_data": ("GUIDE_DATA",),
            },
            "optional": {
                "motion_guide_data":      ("MOTION_GUIDE_DATA",),
                "model":                  ("MODEL",),
                "ic_lora_name":           (["None"] + loras, {"default": "None"}),
                "ic_lora_strength":       ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "scale_by":               ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "upscale_method":         (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {"default": "bicubic"}),
                "image_attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop":                   (["disabled", "center"], {"default": "center"}),
                "auto_snap_ic_grid":      ("BOOLEAN", {"default": True}),
                "use_tiled_encode":       ("BOOLEAN", {"default": False}),
                "tile_size":              ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "tile_overlap":           ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
                "retake_mode":            ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MODEL", "FLOAT")
    RETURN_NAMES = ("positive", "negative", "latent", "model", "latent_downscale_factor")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"

    @classmethod
    def execute(cls, positive, negative, vae, latent, guide_data,
                motion_guide_data=None, model=None, ic_lora_name="None",
                ic_lora_strength=1.0, scale_by=1.0, upscale_method="bicubic",
                image_attention_strength=1.0, crop="center", auto_snap_ic_grid=True,
                use_tiled_encode=False, tile_size=256, tile_overlap=64, retake_mode=False):

        motion_segments = (motion_guide_data or {}).get("segments", []) if motion_guide_data else []
        image_guides_count = len(guide_data.get("images", [])) if guide_data else 0
        log.info("[MuseGuide] execute — image_guides: %d, motion_segments: %d, ic_lora: %s",
                 image_guides_count, len(motion_segments), ic_lora_name)

        # Resolve resize method
        active_resize = (guide_data or {}).get("resize_method") or \
                        (motion_guide_data or {}).get("resize_method") or \
                        ("crop" if crop == "center" else "stretch to fit")

        # IC-LoRA
        latent_downscale_factor = 1.0
        if model is not None and ic_lora_name != "None":
            model, latent_downscale_factor = _load_lora_model_only(model, ic_lora_name, ic_lora_strength)

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        noise_mask = _clone_noise_mask(latent, latent_image)

        # scale_by is for IC-LoRA reference video encoding only — never resize the generation latent here
        if auto_snap_ic_grid and model is not None and ic_lora_name != "None":
            latent_image, noise_mask = _snap_latent_to_downscale(latent_image, noise_mask, latent_downscale_factor, upscale_method)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        initial_latent_length = int(latent_length)
        time_scale_factor = scale_factors[0]
        ltxv_length = (latent_length - 1) * time_scale_factor + 1
        is_lora_active = model is not None and ic_lora_name != "None"

        # Parse timeline data for retake detection
        timeline_data_str = (guide_data or {}).get("timeline_data", "{}")
        try:
            tdata = json.loads(timeline_data_str)
        except Exception:
            tdata = {}

        is_retake_active = bool(retake_mode) or tdata.get("retakeMode", False)
        is_empty_latent = latent_image.abs().max().item() < 1e-5

        # ── Retake Mode ───────────────────────────────────────────────────────
        if is_retake_active:
            log.info("[MuseGuide] Retake mode active. is_empty_latent: %s", is_empty_latent)
            target_pix_w = latent_width * 32
            target_pix_h = latent_height * 32

            retake_start   = int(tdata.get("retakeStart", 0))
            retake_len     = int(tdata.get("retakeLength", 0))
            retake_strength = float(tdata.get("retakeStrength", 1.0))
            start_frame    = int((guide_data or {}).get("start_frame", 0))
            relative_start = max(0, retake_start - start_frame)

            l_start = relative_start // time_scale_factor
            l_end   = int(math.ceil((relative_start + retake_len) / time_scale_factor))
            l_start = min(l_start, latent_length)
            l_end   = min(l_end, latent_length)

            need_base_video = not (not is_empty_latent and l_start == 0 and l_end >= latent_length)

            retake_vid_info = tdata.get("retakeVideo") or {}
            video_file = retake_vid_info.get("imageFile", "") if isinstance(retake_vid_info, dict) else ""
            director_fps = float((motion_guide_data or {}).get("frame_rate", (guide_data or {}).get("frame_rate", 24)))

            if need_base_video:
                if not video_file:
                    raise ValueError(
                        "[MuseGuide] Retake mode is active but no base video has been selected on the timeline."
                    )
                try:
                    video_frames = _load_motion_video_frames(video_file, start_frame, ltxv_length, director_fps)
                    retake_resize = active_resize if active_resize != "maintain aspect ratio" else "pad"
                    pixels = _resize_image(video_frames, target_pix_w, target_pix_h, retake_resize, divisible_by=1)
                    keep = ((pixels.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
                    enc_src = pixels[:keep, :, :, :3]
                    if use_tiled_encode:
                        base_latent = vae.encode_tiled(enc_src, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap)
                    else:
                        base_latent = vae.encode(enc_src)
                    base_latent = base_latent.to(device=latent_image.device, dtype=latent_image.dtype)
                    paste_len = min(base_latent.shape[2], latent_length)
                    if is_empty_latent:
                        latent_image[:, :, :paste_len] = base_latent[:, :, :paste_len]
                    else:
                        if l_start > 0:
                            latent_image[:, :, :l_start] = base_latent[:, :, :l_start]
                        if l_end < paste_len:
                            latent_image[:, :, l_end:paste_len] = base_latent[:, :, l_end:paste_len]
                except Exception as exc:
                    log.warning("[MuseGuide] Base video load/encode failed: %s", exc)

            noise_mask = torch.zeros_like(noise_mask)
            if l_end > l_start:
                noise_mask[:, :, l_start:l_end] = retake_strength

            crop_frames = max(0, int(latent_image.shape[2]) - initial_latent_length)
            positive = node_helpers.conditioning_set_values(positive, {"nghtdrp_guide_crop_latent_frames": crop_frames})
            negative = node_helpers.conditioning_set_values(negative, {"nghtdrp_guide_crop_latent_frames": crop_frames})
            return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, model, float(latent_downscale_factor))

        # ── Standard Keyframe Guidance ────────────────────────────────────────
        images       = (guide_data or {}).get("images", [])
        insert_frames = (guide_data or {}).get("insert_frames", [])
        strengths    = (guide_data or {}).get("strengths", [])
        director_fps = float((motion_guide_data or {}).get("frame_rate", (guide_data or {}).get("frame_rate", 24)))
        segments     = (motion_guide_data or {}).get("segments", [])

        if images or segments:
            log.info("[MuseGuide] Appended Keyframe Guidance. is_lora_active: %s", is_lora_active)

            # A. Image guides
            for idx, img_tensor in enumerate(images):
                f_idx    = insert_frames[idx] if idx < len(insert_frames) else 0
                strength = float(strengths[idx] if idx < len(strengths) else 1.0)
                if strength <= 0.0:
                    continue

                B_img, H_img, W_img, C_img = img_tensor.shape
                tpw = int(latent_width * 32)
                tph = int(latent_height * 32)
                if tpw != W_img or tph != H_img:
                    nchw = img_tensor.permute(0, 3, 1, 2)
                    nchw = comfy.utils.common_upscale(nchw, tpw, tph, upscale_method, "disabled")
                    img_tensor = nchw.permute(0, 2, 3, 1)

                image_pixels, guide_latent = nodes_lt.LTXVAddGuide.encode(vae, latent_width, latent_height, img_tensor, scale_factors)
                frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(positive, latent_length, len(image_pixels), int(f_idx), scale_factors)

                if latent_idx >= latent_length:
                    continue

                max_f = latent_length - latent_idx
                if guide_latent.shape[2] > max_f:
                    guide_latent = guide_latent[:, :, :max_f]

                tokens_added   = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
                guide_orig_shape = list(guide_latent.shape[2:])

                positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                    positive, negative, frame_idx, latent_image, noise_mask, guide_latent, strength, scale_factors
                )
                if is_lora_active:
                    positive = _append_guide_attention_entry(positive, tokens_added, guide_orig_shape, image_attention_strength)
                    negative = _append_guide_attention_entry(negative, tokens_added, guide_orig_shape, image_attention_strength)

            # B. Motion video segments
            for seg in segments:
                try:
                    video_file = seg.get("videoFile")
                    if not video_file:
                        continue

                    seg_start      = int(seg.get("start", 0))
                    seg_length     = int(seg.get("length", 1))
                    trim_start     = int(seg.get("trimStart", 0))
                    vid_strength   = float(seg.get("videoStrength", 1.0))
                    vid_attn_str   = float(seg.get("videoAttentionStrength", 0.65))

                    if seg_length <= 0 or vid_strength <= 0.0:
                        continue

                    video_frames = _load_motion_video_frames(video_file, trim_start, seg_length, director_fps, seg.get("resampleMode", "nearest"))
                    keep = ((video_frames.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
                    video_frames = video_frames[:keep]
                    causal_fix = int(seg_start) == 0 or keep == 1
                    encode_frames = video_frames if causal_fix else torch.cat([video_frames[:1], video_frames], dim=0)

                    _, guide_latent = _encode_video_iclora(
                        vae, latent_width, latent_height, encode_frames, scale_factors,
                        latent_downscale_factor, crop, use_tiled_encode, tile_size, tile_overlap,
                        resize_method=active_resize,
                    )
                    if not causal_fix:
                        guide_latent = guide_latent[:, :, 1:]

                    latent_idx = (seg_start + time_scale_factor - 1) // time_scale_factor if seg_start > 0 else 0
                    if latent_idx >= latent_length:
                        continue

                    if seg_start > 0 and guide_latent.shape[2] > 1:
                        guide_latent = guide_latent[:, :, 1:]
                        seg_start  += time_scale_factor
                        latent_idx += 1
                        if latent_idx >= latent_length:
                            continue

                    max_f = latent_length - latent_idx
                    if guide_latent.shape[2] > max_f:
                        guide_latent = guide_latent[:, :, :max_f]

                    guide_orig_shape = list(guide_latent.shape[2:])
                    Bg, Cg, Fg, Hg, Wg = guide_latent.shape
                    guide_mask = torch.ones((Bg, 1, Fg, Hg, Wg), device=guide_latent.device, dtype=guide_latent.dtype)

                    if seg_start > 0:
                        for i, s in enumerate([0.25, 0.65]):
                            if i < Fg:
                                guide_mask[:, :, i] = 1.0 + vid_strength * (1.0 - s)

                    ldf = int(max(1, round(float(latent_downscale_factor))))
                    if ldf > 1:
                        dilated = _dilate_latent({"samples": guide_latent, "noise_mask": guide_mask}, ldf, ldf)
                        guide_mask   = dilated["noise_mask"]
                        guide_latent = dilated["samples"]

                    tokens_added = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
                    positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                        positive, negative, seg_start, latent_image, noise_mask,
                        guide_latent, vid_strength, scale_factors,
                        guide_mask=guide_mask,
                        latent_downscale_factor=float(latent_downscale_factor),
                        causal_fix=causal_fix,
                    )
                    if is_lora_active:
                        positive = _append_guide_attention_entry(positive, tokens_added, guide_orig_shape, vid_attn_str)
                        negative = _append_guide_attention_entry(negative, tokens_added, guide_orig_shape, vid_attn_str)

                except Exception as exc:
                    raise RuntimeError(f"[MuseGuide] Motion segment failed for {seg}: {exc}") from exc

        else:
            log.info("[MuseGuide] No guides present — passing through.")

        crop_frames = max(0, int(latent_image.shape[2]) - initial_latent_length)
        positive = node_helpers.conditioning_set_values(positive, {"nghtdrp_guide_crop_latent_frames": crop_frames})
        negative = node_helpers.conditioning_set_values(negative, {"nghtdrp_guide_crop_latent_frames": crop_frames})
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, model, float(latent_downscale_factor))


# ── Crop node ─────────────────────────────────────────────────────────────────

class MuseCropGuides:
    """
    Muse Collective replacement for LTXDirectorCropGuides.
    Trims the appended guide keyframes from the latent after sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent":   ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"

    def execute(self, positive, negative, latent):
        latent_image = latent["samples"].clone()
        noise_mask   = _get_noise_mask_for_crop(latent)
        crop_frames  = _get_exact_crop_count(positive)

        if crop_frames > 0:
            crop_frames = min(crop_frames, max(0, latent_image.shape[2] - 1))
            latent_image = latent_image[:, :, :-crop_frames]
            noise_mask   = noise_mask[:, :, :-crop_frames]

        clear = {
            "keyframe_idxs": None,
            "guide_attention_entries": None,
            "nghtdrp_guide_crop_latent_frames": None,
        }
        positive = node_helpers.conditioning_set_values(positive, clear)
        negative = node_helpers.conditioning_set_values(negative, clear)
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MuseGuide":      MuseGuide,
    "MuseCropGuides": MuseCropGuides,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseGuide":      "Muse Guide",
    "MuseCropGuides": "Muse Crop Guides",
}
