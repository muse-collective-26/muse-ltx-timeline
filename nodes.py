"""
LTX Infinite Director — custom nodes for chunked LTX 2.3 generation.

Brings WanInfinite's carry-frame architecture to LTX Director:

  LTXVInjectCarryFrames  — encodes the last N decoded frames from the previous chunk
                            and writes them directly into position 0 of the new chunk's
                            video latent, then freezes those frames via the noise_mask.
                            Pure latent operation — does NOT touch conditioning.

  LTXVExtractCarryFrames — slices the last N pixel frames from decoded video output,
                            ready to pass to the next subgraph.
"""

import logging

import torch
import comfy.utils
import comfy_extras.nodes_lt as nodes_lt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1: Inject carry frames into a chunk's video latent
# ---------------------------------------------------------------------------

class LTXVInjectCarryFrames:
    """
    Encodes carry frames from the previous chunk and writes them into position 0
    of the current chunk's video latent. Sets noise_mask = 0 for those frames so
    the KSampler preserves them exactly, producing a near-seamless seam.

    Pure latent-space operation — does not modify conditioning.
    Wire video_latent (from LTXDirector) → this node → Stage #1 latent input.

    For chunk 1, leave carry_frames disconnected: the node passes the latent through
    unchanged and the reference image wired in LTXDirectorGuide acts as anchor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "The video_latent output from LTXDirector for this chunk. "
                            "The carry frames will be encoded and written into position 0."
                        )
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "How strongly carry frames anchor the seam. "
                            "1.0 = fully frozen (exact pixel match, hard seam). "
                            "0.85–0.95 allows the sampler a little freedom for a softer join."
                        ),
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 29,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": (
                            "H.264 CRF pre-processing applied to carry frames before VAE encode "
                            "(matches the LTXVPreprocess step used by LTXVAddGuideAdvanced). "
                            "Higher = more blur/compression. 0 = no processing."
                        ),
                    },
                ),
            },
            "optional": {
                "carry_frames": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Last N decoded pixel frames from the previous chunk. "
                            "Leave disconnected for chunk 1 — node passes latent through unchanged."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "LTXInfiniteDirector"
    DESCRIPTION = (
        "Encodes carry frames from the previous chunk and writes them into "
        "position 0 of the current chunk's video latent. Freezes those frames "
        "via noise_mask so KSampler preserves them. Leave carry_frames "
        "disconnected on the first subgraph."
    )

    def execute(self, vae, latent, strength=1.0, crf=29, carry_frames=None):
        if carry_frames is None:
            log.info("[LTXVInjectCarryFrames] No carry frames — chunk 1 pass-through.")
            return (latent,)

        scale_factors = vae.downscale_index_formula
        latent_samples = latent["samples"]
        _, _, latent_length, latent_height, latent_width = latent_samples.shape

        # Resize carry frames to match the Stage-1 latent's pixel dimensions
        _, width_scale_factor, height_scale_factor = scale_factors
        target_w = latent_width * width_scale_factor
        target_h = latent_height * height_scale_factor

        frames = carry_frames  # [N, H, W, 3]
        if frames.shape[1] != target_h or frames.shape[2] != target_w:
            frames = (
                comfy.utils.common_upscale(
                    frames.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
                )
                .movedim(1, -1)
                .clamp(0, 1)
            )

        # Apply CRF pre-processing (matches LTXVPreprocess used by standard guide nodes)
        if crf > 0:
            frames = nodes_lt.LTXVPreprocess().execute(frames, crf)[0]

        # VAE-encode carry frames → LTX latent [1, C, T_carry, H_lat, W_lat]
        encode_pixels = frames[:, :, :, :3]
        t = vae.encode(encode_pixels)
        carry_latent_frames = min(t.shape[2], latent_length)

        latent_image = latent_samples.clone()

        # Overwrite the start of this chunk's latent with the carry-frame encoding
        latent_image[:, :, :carry_latent_frames] = t[:, :, :carry_latent_frames].to(
            device=latent_image.device, dtype=latent_image.dtype
        )

        # Build noise mask: carry-frame region = (1 - strength) = frozen/soft-frozen,
        # rest of chunk = 1.0 = fully denoised by KSampler.
        b = latent_image.shape[0]
        noise_mask = latent.get("noise_mask", None)
        if noise_mask is None:
            noise_mask = torch.ones(
                (b, 1, latent_length, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )
        else:
            noise_mask = noise_mask.clone()

        noise_mask[:, :, :carry_latent_frames] = 1.0 - strength

        log.info(
            "[LTXVInjectCarryFrames] Wrote %d carry frames (latent frames: %d/%d). "
            "Noise mask for carry region = %.2f",
            frames.shape[0],
            carry_latent_frames,
            latent_length,
            1.0 - strength,
        )

        return ({"samples": latent_image, "noise_mask": noise_mask},)


# ---------------------------------------------------------------------------
# Node 2: Extract carry frames from decoded chunk output
# ---------------------------------------------------------------------------

class LTXVExtractCarryFrames:
    """
    Extracts the last N decoded pixel frames from a chunk's VAEDecode output
    to pass as carry_frames to the next chunk's LTXVInjectCarryFrames node.

    Connect the IMAGE output of your VAEDecode (or the SaveVideo decoder node)
    to this node. Wire carry_frames to the next subgraph.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_frames": (
                    "INT",
                    {
                        "default": 9,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Number of frames to extract from the end of the decoded output. "
                            "9 = 1 LTX latent frame at 8× temporal compression. "
                            "Must match num_frames on the next chunk's LTXVInjectCarryFrames."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("carry_frames",)
    FUNCTION = "execute"
    CATEGORY = "LTXInfiniteDirector"
    DESCRIPTION = (
        "Extracts the last N decoded frames from a chunk's output. "
        "Wire carry_frames into the next chunk's LTXVInjectCarryFrames."
    )

    def execute(self, images, num_frames):
        n = images.shape[0]
        num_frames = min(num_frames, n)
        carry = images[-num_frames:]
        log.info(
            "[LTXVExtractCarryFrames] Extracted %d carry frames from %d decoded frames.",
            num_frames,
            n,
        )
        return (carry,)


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LTXVInjectCarryFrames": LTXVInjectCarryFrames,
    "LTXVExtractCarryFrames": LTXVExtractCarryFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVInjectCarryFrames": "LTX Inject Carry Frames",
    "LTXVExtractCarryFrames": "LTX Extract Carry Frames",
}
