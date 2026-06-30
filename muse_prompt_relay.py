"""
Muse Collective — local copy of PromptRelay core logic.
Frozen from WhatDreamsCost-ComfyUI prompt_relay.py so MuseDirectorV1 has no WDC dependency.
"""
import logging
import math
import torch

log = logging.getLogger(__name__)


def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame
    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)
    return offset


def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames, is_audio=False):
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq
    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        if is_audio:
            sigma_val = seg.get("sigma_audio", seg["sigma"])
            window_val = seg.get("window_audio", seg["window"])
            strength_val = seg.get("strength_audio", 1.0)
        else:
            sigma_val = seg["sigma"]
            window_val = seg["window"]
            strength_val = seg.get("strength", 1.0)
        cost = strength_val * (torch.relu(d - window_val) ** 2) / (2 * sigma_val ** 2)
        offset[:, local] = cost.to(offset.dtype)
    return offset


def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1

    def mask_fn(Lq, Lk, dtype, device, transformer_options):
        if Lq == Lk:
            return None
        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None
        grid_sizes = transformer_options.get("grid_sizes", None)
        attn_type = transformer_options.get("promptrelay_attn_type", "attn2")
        is_audio = (attn_type == "audio_attn2")
        if is_audio:
            mode = "scaled"
            video_lq = -1
        else:
            if grid_sizes is not None:
                video_tpf = int(grid_sizes[1]) * int(grid_sizes[2])
            else:
                if Lq % latent_frames == 0:
                    video_tpf = Lq // latent_frames
                else:
                    video_tpf = fallback_tokens_per_frame
            video_lq = latent_frames * video_tpf
            if Lk == video_lq or Lk < max_token_idx:
                return None
            mode = "video" if Lq == video_lq else "scaled"
        key = (Lq, Lk, mode, device)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames, is_audio=is_audio)
            log.info(
                "[MusePromptRelay] Built penalty matrix (%s): Lq=%d, Lk=%d, nonzero=%d/%d",
                mode, Lq, Lk, (cost > 0).sum().item(), cost.numel(),
            )
            cache[key] = -cost
        return cache[key].to(dtype)

    return mask_fn


def build_segments(token_ranges, segment_lengths, epsilon=1e-3, relay_options=None):
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448
    opts = relay_options or {}
    v_strength = opts.get("video_strength", 1.0)
    v_window_scale = opts.get("video_window_scale", 1.0)
    a_epsilon = opts.get("audio_epsilon")
    a_strength = opts.get("audio_strength", 1.0)
    a_window_scale = opts.get("audio_window_scale", 1.0)
    if a_epsilon is not None and 0 < a_epsilon < 1:
        sigma_audio = 1.0 / math.log(1.0 / a_epsilon)
    else:
        sigma_audio = sigma
    q_token_idx = []
    frame_cursor = 0
    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            continue
        midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 2, 0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": max(base_window * v_window_scale, 0.0),
            "sigma": sigma,
            "strength": v_strength,
            "window_audio": max(base_window * a_window_scale, 0.0),
            "sigma_audio": sigma_audio,
            "strength_audio": a_strength,
        })
        frame_cursor += L
    return q_token_idx


def get_raw_tokenizer(clip):
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer
    raise RuntimeError(
        f"Could not find raw tokenizer on CLIP object. "
        f"Known attributes: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    if not has_eos:
        try:
            test_res = raw_tokenizer("test")
            if isinstance(test_res, dict) and "input_ids" in test_res:
                ids = test_res["input_ids"]
            elif hasattr(test_res, "input_ids"):
                ids = test_res.input_ids
            elif isinstance(test_res, list):
                ids = test_res
            else:
                ids = []
            if ids:
                eos_id = getattr(raw_tokenizer, "eos_token_id", None)
                if eos_id is not None and ids[-1] == eos_id:
                    has_eos = True
                elif ids[-1] == 1:
                    has_eos = True
        except Exception:
            pass
    eos_adj = 1 if has_eos else 0
    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt
    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len
    return full_prompt, token_ranges


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = specified_lengths
    else:
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments
    effective = []
    cursor = 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective


def convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)
    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames
    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1
    return result
