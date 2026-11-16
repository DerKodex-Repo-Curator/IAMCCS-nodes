"""Optional file-based finishing for a completed R49 delivery master."""

from fractions import Fraction
from hashlib import sha256
import json
import logging
from math import isfinite
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile

LOG = logging.getLogger("IAMCCS.H3.FinalFinish")
FIELDS = {
    "h3_final_relay_enabled": ("BOOLEAN", {"default": False, "display_name": "Enable RelayVSR"}),
    "h3_final_relay_detail": ("FLOAT", {"default": .45, "min": 0.0, "max": 1.0, "step": .01,
        "display_name": "RelayVSR Detail", "tooltip": "0 is bicubic; 1 applies the full RelayVSR correction."}),
    "h3_final_dlss_enabled": ("BOOLEAN", {"default": False, "display_name": "DLSS5 · 1×"}),
    "h3_final_dlss_style": (["Default", "Natural", "Cinematic"], {"default": "Default", "display_name": "DLSS5 Style"}),
    "h3_final_dlss_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": .01, "display_name": "DLSS5 Intensity"}),
    "h3_final_dlss_tone": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": .01, "display_name": "DLSS5 Local Tone"}),
    "h3_final_dlss_structure": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": .01, "display_name": "DLSS5 Local Structure"}),
    "h3_final_dlss_skin": ("FLOAT", {"default": 2.0, "min": -1.0, "max": 2.0, "step": .01, "display_name": "DLSS5 Skin Structure"}),
    "h3_final_dlss_skin_mask": ("BOOLEAN", {"default": True, "display_name": "DLSS5 Automatic Skin Mask"}),
    "h3_final_reca_enabled": ("BOOLEAN", {"default": False, "display_name": "ReCaVSR"}),
    "h3_final_reca_scale": (["1.5", "2", "4"], {"default": "2", "display_name": "ReCaVSR Scale"}),
    "h3_final_reca_color_fix": (["none", "adain", "wavelet"], {"default": "none", "display_name": "ReCaVSR Colour Match"}),
    "h3_final_reca_kv_offload": ("BOOLEAN", {"default": False, "display_name": "ReCaVSR Attention Cache in RAM"}),
    "h3_vsr_per_segment": ("BOOLEAN", {"default": False, "display_name": "VSR as Second Pass per Segment"}),
    "h3_final_relay_scale": (["2", "4"], {"default": "2", "display_name": "RelayVSR Scale"}),
}


def configuration(settings):
    values = {}
    for field, (kind, spec) in FIELDS.items():
        value = settings.get(field, spec["default"])
        if kind == "FLOAT":
            try:
                number = float(value)
            except (ValueError, TypeError):
                number = spec["default"]
            value = min(spec["max"], max(spec["min"], number)) if isfinite(number) else spec["default"]
        elif kind == "BOOLEAN":
            value = value is True
        elif value not in kind:
            value = spec["default"]
        values[field] = value
    return values


def finishing_enabled(settings):
    return any(settings.get(key) is True for key in (
        "h3_final_relay_enabled", "h3_final_reca_enabled", "h3_final_dlss_enabled",
    ))


def is_segment(path):
    # All per-segment R49 encoders publish segment_NNNN or a sidecar index.
    if re.fullmatch(r"segment_\d+", path.stem, re.IGNORECASE):
        return True
    sidecar = path.with_suffix(path.suffix + ".iamccs.json")
    if sidecar.is_file():
        generation = json.loads(sidecar.read_text(encoding="utf-8")).get("generation") or {}
        return "segment_index" in generation and "segment_files" not in generation
    return False


def is_segmented_vsr_master(path):
    sidecar = path.with_suffix(path.suffix + ".iamccs.json")
    if not sidecar.is_file():
        return False
    generation = json.loads(sidecar.read_text(encoding="utf-8")).get("generation") or {}
    return generation.get("vsr_segmented") is True


def _node(name):
    import nodes
    node = nodes.NODE_CLASS_MAPPINGS.get(name)
    if node is None:
        raise RuntimeError(f"R49 Final Finish requires the ComfyUI node {name}. Install or enable its provider.")
    return node


def _relay_models(scale=2):
    import folder_paths
    def select(category, decoder):
        matches = [name for name in folder_paths.get_filename_list(category)
                   if "relayvsr" in name.lower() and re.search(rf"(?:^|[_ /-])(?:x{scale}|{scale}x)(?:[_ .-]|$)", name.lower())
                   and ("flash" in name.lower()) == decoder]
        if not matches:
            raise RuntimeError(f"R49 Final Finish: install RelayVSR ×{scale} {'Flash decoder' if decoder else 'model'} in {category}.")
        return min(matches, key=lambda name: (
            "convrot" not in name.lower() if not decoder else "fp32" not in name.lower(),
            "int8" not in name.lower() if not decoder else False, name.lower()))
    return select("diffusion_models", False), select("vae", True)


def _reca_model():
    import folder_paths

    _node("ReCaVSRLoader")
    _node("ReCaVSRUpscale")
    candidates = []
    for name in folder_paths.get_filename_list("recavsr"):
        stem = Path(name).stem.lower()
        if Path(name).suffix.lower() != ".safetensors" or not ("recavsr" in stem or stem == "transformer"):
            continue
        path = Path(folder_paths.get_full_path_or_raise("recavsr", name))
        if all(path.with_name(companion).is_file() for companion in ("prompt.safetensors", "flashdecoder.safetensors")):
            candidates.append(name)
    if not candidates:
        raise RuntimeError("R49 Final Finish: install a ReCaVSR transformer with prompt.safetensors and flashdecoder.safetensors beside it in models/recavsr.")
    return min(candidates, key=lambda name: ("int8" not in name.lower(), name.lower()))


def _stream_spec(path, *, hash_audio=False):
    import av
    with av.open(str(path)) as media:
        video = media.streams.video[0]
        width, height = video.width, video.height
        rate = video.average_rate
        if not rate or width < 2 or height < 2:
            raise ValueError("R49 Final Finish needs a video with valid dimensions and frame rate.")
        if video.codec_context.color_trc in (16, 18):
            raise ValueError("R49 Final Finish currently expects an SDR master.")
        audio = tuple(media.streams.audio)
        if not hash_audio:
            return width, height, rate
        hashes = {stream.index: sha256() for stream in audio}
        for packet in media.demux(*audio) if audio else ():
            if packet.size:
                hashes[packet.stream.index].update(bytes(packet))
        return width, height, rate, tuple(item.hexdigest() for item in hashes.values())


def _ffmpeg():
    from .iamccs_minimax_h3_shotboard import _find_ffmpeg
    executable = _find_ffmpeg()
    if not executable:
        raise RuntimeError("R49 Final Finish requires FFmpeg.")
    return executable


def _encoder_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _count_frames(path):
    import av
    with av.open(str(path)) as media:
        stream = media.streams.video[0]
        first = None
        count = 0
        for frame in media.decode(stream):
            if frame.pts is None:
                raise ValueError("R49 Final Finish needs timestamped frames.")
            stamp = frame.pts * frame.time_base
            if first is None:
                first = stamp
            if abs(stamp - first - Fraction(count, stream.average_rate)) > frame.time_base:
                raise ValueError("R49 Final Finish currently expects a constant-frame-rate master.")
            count += 1
        return count


def _relay_window(output_width, output_height):
    from comfy.ldm.wan.model_vsr import RELAY_KEYFRAME_INTERVAL as period

    intervals = max(1, min(4, (2 * 1024**3 // (output_width * output_height * 3 * 4) - 1) // period))
    return period * intervals + 1, period + 1 if intervals > 1 else 1


def _relay(source, output, detail, model_names, scale):
    import av
    import numpy as np
    import torch
    import comfy.model_management as mm

    width, height, fps = _stream_spec(source)
    output_width, output_height = width * scale, height * scale
    model = _node("UNETLoader")().load_unet(model_names[0], "default")[0]
    model = _node("ModelAttentionBackend").execute(model, "comfy kitchen attention")[0]
    vae = _node("VAELoader")().load_vae(model_names[1])[0]
    noise = _node("RandomNoise").execute(42)[0]
    sampler = _node("KSamplerSelect").execute("euler")[0]
    sigmas = _node("BasicScheduler").execute(model, "simple", 1, 1.0)[0]
    conditioning = _node("RelayVSRConditioning")
    guider = _node("BasicGuider")
    sample = _node("SamplerCustomAdvanced")
    decode = _node("VSRDecode")

    # Budget output IMAGE storage and leave one native Relay interval of context.
    batch, overlap = _relay_window(output_width, output_height)
    fps_string = f"{fps.numerator}/{fps.denominator}"
    command = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{output_width}x{output_height}",
        "-framerate", fps_string, "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-threads", "4",
        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-movflags", "+faststart", str(output)]
    LOG.info("RelayVSR ×%d | %dx%d → %dx%d | detail=%.2f | window=%d overlap=%d",
             scale, width, height, output_width, output_height, detail, batch, overlap)

    def write_window(buffer, skip, sink):
        mm.throw_exception_if_processing_interrupted()
        images = torch.from_numpy(np.stack(buffer)).float().div_(255)
        positive, latent = conditioning.execute(model, images).result
        guide = guider.execute(model, positive)[0]
        sampled = sample.execute(noise, guide, sampler, sigmas, latent)[0]
        # The Flash decoder needs its own activation space at large canvases.
        if output_width * output_height >= 1920 * 1080:
            mm.unload_all_models()
            mm.soft_empty_cache()
        frames = decode.execute(vae, sampled, images, detail)[0]
        if tuple(frames.shape[:3]) != (len(buffer), output_height, output_width):
            raise RuntimeError("RelayVSR returned unexpected frame geometry.")
        for frame in frames[skip:]:
            mm.throw_exception_if_processing_interrupted()
            pixels = frame.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1).mul_(255).round_().to(torch.uint8)
            sink.write(pixels.contiguous().numpy().tobytes())
        return len(buffer) - skip

    with TemporaryFile() as errors, av.open(str(source)) as media:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors, creationflags=_encoder_flags()) as encoder:
            written, skip, pending = 0, 0, []
            try:
                for frame in media.decode(video=0):
                    pending.append(frame.to_ndarray(format="rgb24"))
                    if len(pending) == batch:
                        written += write_window(pending, skip, encoder.stdin)
                        pending, skip = pending[-overlap:], overlap
                if len(pending) > skip:
                    written += write_window(pending, skip, encoder.stdin)
                encoder.stdin.close()
                if encoder.wait() or not written:
                    raise RuntimeError("RelayVSR could not encode the output.")
            except BaseException as error:
                if encoder.poll() is None:
                    encoder.terminate()
                if isinstance(error, (RuntimeError, BrokenPipeError)):
                    errors.seek(0)
                    message = errors.read().decode("utf-8", "replace")[-1600:].strip()
                    if message:
                        raise RuntimeError(f"RelayVSR encoder: {message}") from error
                raise
    return written


def _reca(source, output, settings, work, model_name, scale):
    from comfy_api.latest import InputImpl

    # The provider otherwise decodes all audio to RAM and re-encodes it.
    # Feed it a video-only stream and copy the original audio back at mux time.
    silent = work / "reca_input.mp4"
    ffmpeg = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    subprocess.run([*ffmpeg, "-i", str(source), "-map", "0:v:0", "-c:v", "copy", str(silent)],
                   check=True, creationflags=_encoder_flags())
    model = _node("ReCaVSRLoader").execute(transformer=model_name)[0]
    video = _node("ReCaVSRUpscale").execute(
        model=model, video=InputImpl.VideoFromFile(str(silent)), scale=scale,
        color_fix=settings["h3_final_reca_color_fix"], seed=42, decoder="flashdecoder",
        kv_offload=settings["h3_final_reca_kv_offload"], encoder="h264",
    )[0]
    raw = video.get_stream_source()
    raw_path = Path(raw) if isinstance(raw, str) else None
    try:
        if raw_path is None:
            raw_path = work / "reca_video.mp4"
            with raw_path.open("wb") as destination:
                raw.seek(0)
                shutil.copyfileobj(raw, destination)
        subprocess.run([*ffmpeg, "-i", str(raw_path), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                        "-map_metadata", "1", "-c", "copy", "-movflags", "+faststart", str(output)],
                       check=True, creationflags=_encoder_flags())
    finally:
        # ReCaVSR returns an owned temporary file, outside this work directory.
        import folder_paths

        if raw_path is not None and raw_path.resolve().parent == (Path(folder_paths.get_temp_directory()) / "recavsr").resolve():
            raw_path.unlink(missing_ok=True)


def _dlss(source, output, settings, work, audio_source, fps):
    config = _node("DLSS5Settings").execute(
        upscaling_mode="1x (DLAA / native)", nr_preset="Default",
        nr_style=settings["h3_final_dlss_style"], nr_intensity=settings["h3_final_dlss_intensity"],
        local_tone_strength=settings["h3_final_dlss_tone"], local_structure_strength=settings["h3_final_dlss_structure"],
        skin_structure_strength=settings["h3_final_dlss_skin"], automatic_mask=settings["h3_final_dlss_skin_mask"],
        dlss_model_preset="Default", motion="auto", scene_change_threshold=.24, warmup_frames=0, runtime_dir="")[0]
    raw, written = _node("DLSS5EnhanceVideoFile").execute(
        video_path=str(source), settings=config, codec="H.264", container="MP4", quality="Max",
        filename_prefix="DLSS5", output_directory=str(work), max_frames=0, copy_audio=False,
        verify_neural_rendering=True).result
    # The plugin's intermediate MKV rounds to milliseconds. Restore the
    # validated source CFR clock by stream-copying into an exact timescale.
    timescale = fps.numerator * 512
    ticks = fps.denominator * 512
    clocked = work / "dlss_clock.mp4"
    prefix = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    subprocess.run([*prefix, "-i", raw, "-i", str(audio_source), "-map", "0:v:0", "-map", "1:a?",
                    "-map_metadata", "1", "-c", "copy", "-video_track_timescale", str(timescale), str(clocked)],
                   check=True, creationflags=_encoder_flags())
    setts = f"setts=pts=round(PTS/{ticks})*{ticks}:dts=round(DTS/{ticks})*{ticks}:duration={ticks}"
    subprocess.run([*prefix, "-i", str(clocked), "-map", "0", "-c", "copy", "-bsf:v", setts,
                    "-video_track_timescale", str(timescale), "-movflags", "+faststart", str(output)],
                   check=True, creationflags=_encoder_flags())
    return int(written)


def finish_master(source, settings, *, relay_scale=None):
    import comfy.model_management as mm
    import folder_paths

    source = Path(source).resolve(strict=True)
    root = Path(folder_paths.get_output_directory()).resolve()
    if not source.is_relative_to(root):
        raise ValueError("R49 Final Finish expects a master inside the ComfyUI output directory.")
    settings = configuration(settings)
    relay, reca, dlss = (settings[key] for key in ("h3_final_relay_enabled", "h3_final_reca_enabled", "h3_final_dlss_enabled"))
    if not finishing_enabled(settings):
        return source
    if relay and reca:
        raise ValueError("R49 Final Finish: choose RelayVSR or ReCaVSR. DLSS5 can follow either.")
    if relay_scale is None:
        relay_scale = int(settings["h3_final_relay_scale"])
    if relay and relay_scale not in (2, 4):
        raise ValueError("RelayVSR scale must be 2× or 4×.")
    scale = float(settings["h3_final_reca_scale"]) if reca else (relay_scale if relay else 1)
    relay_models = _relay_models(relay_scale) if relay else None
    reca_model = _reca_model() if reca else None
    if dlss:
        _node("DLSS5Settings")
        _node("DLSS5EnhanceVideoFile")
    width, height, fps, audio = _stream_spec(source, hash_audio=True)
    frames = _count_frames(source)
    if frames < 1:
        raise ValueError("R49 Final Finish received an empty master.")
    st = source.stat()
    key = sha256(json.dumps([1, st.st_size, st.st_mtime_ns, settings, relay_models, reca_model, scale],
                           sort_keys=True).encode()).hexdigest()[:12]
    suffix = "_".join(effect for effect, on in ((f"relay_x{scale:g}", relay), (f"reca_x{scale:g}", reca), ("dlss5", dlss)) if on)
    output = source.with_name(f"{source.stem}_{suffix}_{key}.mp4")
    target = (int(width * scale + .5), int(height * scale + .5))
    if reca and any(dimension % 2 for dimension in target):
        raise ValueError(f"ReCaVSR output {target[0]}×{target[1]} must have even dimensions for H.264. Choose another scale.")
    expected = (*target, fps, audio)

    def verify(file):
        actual = _stream_spec(file, hash_audio=True)
        actual_frames = _count_frames(file)
        if actual != expected or actual_frames != frames:
            raise RuntimeError(
                f"R49 Final Finish failed verification: {actual[0]}×{actual[1]} @ {actual[2]}, "
                f"{actual_frames}/{frames} frames, audio match={actual[3] == audio}; "
                f"expected {expected[0]}×{expected[1]} @ {fps}.")

    if output.is_file():
        verify(output)
        return output
    mm.unload_all_models()
    mm.soft_empty_cache()
    with TemporaryDirectory(prefix="r49_finish_", dir=source.parent, ignore_cleanup_errors=True) as temp:
        work = Path(temp)
        current = source
        if relay:
            current = work / "relay.mp4"
            if _relay(source, current, settings["h3_final_relay_detail"], relay_models, relay_scale) != frames:
                raise RuntimeError("RelayVSR did not write every source frame.")
        elif reca:
            current = work / "reca.mp4"
            _reca(source, current, settings, work, reca_model, scale)
        if dlss:
            mm.unload_all_models()
            mm.soft_empty_cache()
            processed = work / "dlss.mp4"
            if _dlss(current, processed, settings, work, source, fps) != frames:
                raise RuntimeError("DLSS5 did not write every source frame.")
            current = processed
        verify(current)
        current.replace(output)
    LOG.info("R49 Final Finish complete | %s | frames=%d | audio=stream copy", output, frames)
    return output
