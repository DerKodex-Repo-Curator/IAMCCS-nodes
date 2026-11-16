# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional native-stage video decoder; conditioning always keeps the input VAE."""
import logging


def schema():
    return {"h3_video_decoder": (["native", "sol"], {
        "default": "native", "display_name": "Native-stage video decoder",
        "tooltip": "Native uses the connected H3 VAE. Sol automatically loads an installed Sol-H3 decoder for native-stage decoding only. Encoding, image guides, audio and later refine/upscale VAEs stay unchanged. Sol processes the whole segment and can require more activation memory.",
    })}


def configure(plan, settings, vae_names):
    mode = str(settings.get("h3_video_decoder", "native"))
    if mode not in {"native", "sol"}:
        raise ValueError(f"Unknown H3 video decoder: {mode}")
    name = ""
    if mode == "sol":
        candidates = [n for n in vae_names if "sol" in n.lower() and "h3" in n.lower()
                      and n.lower().endswith(".safetensors")]
        preferred = [n for n in candidates if "int8" in n.lower() and "convrot" in n.lower()]
        candidates = preferred or candidates
        if len(candidates) != 1:
            raise ValueError("Sol-H3 decoder: install one matching Sol-H3 VAE in a registered VAE folder. "
                             "One INT8 ConvRot variant is preferred; ambiguous candidates must be removed from the selection path.")
        name = candidates[0]
    plan["video_decoder"] = {"mode": mode, "model_name": name,
                             "scope": "native_stage_decode_only", "conditioning_vae": "connected_original"}


def select_video_decoder(native_vae, plan):
    config = plan.get("video_decoder") or {}
    mode = config.get("mode", "native")
    if mode == "native":
        return native_vae
    if mode != "sol":
        raise ValueError(f"Unknown H3 video decoder: {mode}")
    # Optional Comfy runtime dependency; pure planning/tests need no model loaders.
    import nodes

    decoder = nodes.VAELoader().load_vae(vae_name=config["model_name"])[0]
    if decoder.latent_channels != 24 or not getattr(decoder.first_stage_model, "noncausal", False):
        raise ValueError("Selected Sol-H3 VAE is not the supported noncausal 24-channel decoder.")
    logging.getLogger(__name__).info(
        "H3 video decoder | mode=sol | model=%s | native-stage decode only | original encoder/guides/audio preserved",
        config["model_name"],
    )
    return decoder
