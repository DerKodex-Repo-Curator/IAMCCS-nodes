# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional R49 REF2VA delivery using the accepted Stage 1 latent and video plate."""

import math


DEFAULT_SIGMAS = "0.25, 0.12, 0"
PROFILES = {
    "up_to_12gb": (22, 5, 320, 320, 128, 0.5),
    "over_12_under_24gb": (39, 22, 512, 384, 160, 0.5),
    "24gb": (73, 22, 768, 512, 192, 0.75),
    "over_24gb": (124, 39, 960, 544, 256, 1.0),
}
PROMPT = (
    "Refine <Video 1>. Preserve the subject, facial identity, framing, lighting, "
    "camera movement and timing. Restore natural fine detail."
)


def schema():
    return {
        "h3_plate_refine_profile": (["off", *PROFILES], {
            "default": "off", "display_name": "H3 Plate Refine VRAM profile",
        }),
        "h3_plate_refine_sigmas": ("STRING", {
            "default": DEFAULT_SIGMAS, "display_name": "H3 Plate Refine sigmas",
            "tooltip": "Explicit decreasing sigmas, ending in zero. Used with Euler.",
        }),
    }


def configure(plan, settings):
    plan.pop("h3_plate_refine", None)
    profile = str(settings.get("h3_plate_refine_profile", "off"))
    if (not plan.get("upscale_enabled") or plan.get("upscale_mode") != "h3_fast_latent_2pass"
            or profile == "off"):
        return
    if profile not in PROFILES:
        raise ValueError(f"Unknown H3 Plate Refine VRAM profile: {profile}")
    if plan.get("task_mode") == "v2va_face_swap":
        raise ValueError("H3 Plate Refine requires a full-frame native H3 latent.")
    try:
        sigmas = [float(part.strip()) for part in str(settings.get("h3_plate_refine_sigmas", DEFAULT_SIGMAS)).split(",")]
    except ValueError as error:
        raise ValueError("H3 Plate Refine sigmas must be comma-separated numbers.") from error
    if (not 2 <= len(sigmas) <= 21 or sigmas[-1] != 0
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in sigmas)
            or any(a <= b for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError("H3 Plate Refine needs decreasing sigmas in [0, 1], ending in 0.")
    plan["h3_plate_refine"] = {"profile": profile, "sigmas": sigmas}


class _RecordingClipTarget:
    """Keep the Windows & Tiles recorder on this clone, including for ClipProj."""

    def __init__(self, clip):
        self.clip = clip

    def __getattr__(self, name):
        return getattr(self.clip, name)

    def clone(self):
        return _RecordingClipTarget(self.clip.clone())


def window_node():
    import nodes

    node = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3TemporalWindow")
    if node is None or not any(item.id == "plate_scale" for item in node.define_schema().inputs):
        raise RuntimeError("H3 Plate Refine requires MiniMax H3 Temporal Window & Spatial Tiles 1.0.8 or compatible.")
    return node


def refine(model, clip, video_vae, native_latent, native_frames, delivery_latent, config, seed, prompt):
    import logging
    import psutil
    import torch
    import nodes
    import comfy.model_management as mm
    from comfy_extras.nodes_custom_sampler import BasicGuider, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift
    from .iamccs_minimax_h3_fast_latent_2pass import _result_item, _split_av, _concat_av
    from .iamccs_minimax_h3_pixel_refine_variant import _provider

    plugin = window_node()
    common = _provider("common")
    video, audio = _split_av(delivery_latent)
    _, _, latent_frames, height_tokens, width_tokens = video["samples"].shape
    frames = common.latents_to_frames(latent_frames)
    height, width = height_tokens * 16, width_tokens * 16
    window, overlap, tile_w, tile_h, tile_overlap, scale = PROFILES[config["profile"]]
    if width < height:
        tile_w, tile_h = tile_h, tile_w

    # A visible segment may have lost its technical prefix. Decode the complete
    # Stage 1 latent in that case so frame zero matches the refine latent.
    if int(native_frames.shape[0]) == frames:
        plate = native_frames
    else:
        estimated = frames * int(native_frames.shape[1]) * int(native_frames.shape[2]) * 3 * 4
        if estimated > psutil.virtual_memory().available // 2:
            raise MemoryError("H3 Plate Refine cannot hold the full technical plate in available system RAM.")
        plate = nodes.VAEDecode().decode(vae=video_vae, samples=native_latent)[0]
    if int(plate.shape[0]) != frames:
        raise ValueError(f"H3 Plate Refine needs {frames} aligned plate frames, got {plate.shape[0]}.")

    model = _result_item(MiniMaxH3SigmaShift.execute(model=model, shift_video=1.0, shift_audio=1.0))
    patched = plugin.execute(
        model=model, clip=_RecordingClipTarget(clip), plate=plate, vae=video_vae,
        temporal_windows=True, window_frames=window, overlap_frames=overlap,
        short_final_window=True, min_final_window_frames=22, carry_overlap=True,
        spatial_tiles=True, tile_width=tile_w, tile_height=tile_h, tile_overlap=tile_overlap,
        crop_ref_images_per_tile=False, tile_audio="keep_input", color_match=False,
        tile_carry="own_tile", plate_scale=scale,
    )
    conditioning = _result_item(MiniMaxH3ReferenceToVideo.execute(
        clip=_result_item(patched, 1), prompt=prompt or PROMPT,
        width=width, height=height, length=frames, vae=video_vae,
        ref_videos={"ref_video_0": plate},
    ))
    mm.unload_all_models()
    mm.soft_empty_cache()
    report = (
        f"H3 PLATE REFINE | {config['profile']} | {width}x{height} | "
        f"{window}/{overlap} frames | {tile_w}x{tile_h}/{tile_overlap} px | "
        f"plate_scale={scale} | Euler {config['sigmas']} | seed={seed}"
    )
    logging.getLogger(__name__).info("%s", report)
    result = _result_item(SamplerCustomAdvanced.execute(
        noise=_result_item(RandomNoise.execute(noise_seed=seed)),
        guider=_result_item(BasicGuider.execute(model=_result_item(patched), conditioning=conditioning)),
        sampler=_result_item(KSamplerSelect.execute(sampler_name="euler")),
        sigmas=torch.tensor(config["sigmas"], dtype=torch.float32), latent_image=delivery_latent,
    ))
    refined, _ = _split_av(result)
    if refined["samples"].shape != video["samples"].shape:
        raise RuntimeError("H3 Plate Refine changed the video latent extent.")
    return _concat_av(refined, audio), report
