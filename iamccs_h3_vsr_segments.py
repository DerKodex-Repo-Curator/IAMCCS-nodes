# SPDX-License-Identifier: GPL-3.0-or-later
"""File-backed VSR checkpoints for the R49 native delivery branch."""
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from . import iamccs_h3_final_finish as finish


def enabled(plan):
    settings = plan.get("final_finish") or {}
    return bool(settings.get("h3_vsr_per_segment") and not plan.get("upscale_enabled")
                and (settings.get("h3_final_relay_enabled") or settings.get("h3_final_reca_enabled")))


def _ffmpeg(*args):
    subprocess.run([finish._ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                    *map(str, args)], check=True, creationflags=finish._encoder_flags())


def _join_video(paths, counts, output, skip=0, *, delivery=False):
    """Prepare exact frame ranges without accumulating frames or decoding audio."""
    start = skip
    inputs, filters, labels = [], [], []
    for path, count in zip(paths, counts, strict=True):
        if start >= count:
            start -= count
            continue
        index = len(labels)
        inputs.extend(("-i", path))
        label = f"v{index}"
        filters.append(f"[{index}:v]trim=start_frame={start}:end_frame={count},setpts=PTS-STARTPTS[{label}]")
        labels.append(f"[{label}]")
        start = 0
    if not labels:
        raise ValueError("VSR needs at least one source frame.")
    filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[v]")
    fps = finish._stream_spec(paths[0])[2]
    filters.append(f"[v]settb=expr={fps.denominator}/{fps.numerator},setpts=N[clocked]")
    quality = ("-preset", "fast", "-crf", 14) if delivery else ("-preset", "ultrafast", "-qp", 0)
    _ffmpeg(*inputs, "-filter_complex", ";".join(filters), "-map", "[clocked]",
            "-r", f"{fps.numerator}/{fps.denominator}", "-fps_mode", "cfr", "-c:v", "libx264",
            *quality, "-pix_fmt", "yuv420p", output)


def upscale_segment(native_paths, output, settings, previous_outputs=()):
    """Run VSR through this segment, retaining only its authored frame range."""
    from .iamccs_minimax_h3_shotboard import _read_segment_frame_count

    settings = finish.configuration(settings)
    counts = [_read_segment_frame_count(path) for path in native_paths]
    prior_frames = sum(counts[:-1])
    relay = settings.get("h3_final_relay_enabled") is True
    reca = settings.get("h3_final_reca_enabled") is True
    if relay == reca:
        raise ValueError("Choose exactly one VSR for the segment pass.")
    width, height, fps = finish._stream_spec(native_paths[-1])
    scale = int(settings["h3_final_relay_scale"]) if relay else float(settings["h3_final_reca_scale"])
    skip = 0
    repair = 0
    if relay and prior_frames:
        batch, overlap = finish._relay_window(int(width * scale), int(height * scale))
        stride = batch - overlap
        skip = max(0, (prior_frames // stride - 1) * stride)
        repair = (prior_frames - batch) % stride if prior_frames >= batch else prior_frames
    elif prior_frames:
        # ReCaVSR pads its last incomplete 21-frame prefix or 8-frame body block.
        repair = (prior_frames - 21) % 8 if prior_frames >= 21 else prior_frames
    # ReCaVSR reconstructs its recurrent state by replaying the native prefix.
    # No model state is kept across H3 jobs or serialized into Saved Take.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="vsr_segment_", dir=output.parent, ignore_cleanup_errors=True) as directory:
        work = Path(directory)
        source = work / "input.mp4"
        _join_video(native_paths, counts, source, skip)
        if finish._count_frames(source) != sum(counts) - skip:
            raise RuntimeError("VSR source preparation changed the authored frame count.")
        processed = finish.finish_master(source, {**settings, "h3_final_dlss_enabled": False})
        first = prior_frames - skip
        staged = work / "delivery.mp4"
        if first:
            _join_video([processed], [first + counts[-1]], staged, first)
        else:
            processed.replace(staged)
        expected = (int(width * scale + .5), int(height * scale + .5), fps)
        if finish._stream_spec(staged) != expected or finish._count_frames(staged) != counts[-1]:
            raise RuntimeError("VSR segment output changed the authored frame range or canvas.")
        end = first
        replacements = []
        for part, (previous, count) in enumerate(zip(reversed(previous_outputs), reversed(counts[:-1]))):
            if not repair:
                break
            take = min(repair, count, end)
            if not take:
                break
            tail, corrected = work / f"tail_{part}.mp4", work / f"previous_{part}.mp4"
            _join_video([processed], [end], tail, end - take)
            _join_video([previous, tail], [count - take, take], corrected)
            if finish._count_frames(corrected) != count:
                raise RuntimeError("VSR tail repair changed the preceding segment length.")
            replacements.append((corrected, previous))
            repair -= take
            end -= count
        for corrected, previous in replacements:
            corrected.replace(previous)
        staged.replace(output)
    return counts[-1]


def assemble(segments, counts, native_master, output):
    """Use the exact audio timeline already joined by R49."""
    with TemporaryDirectory(prefix="vsr_assembly_", dir=output.parent) as directory:
        video = Path(directory) / "joined.mp4"
        staged = Path(directory) / "final.mp4"
        _join_video(segments, counts, video, delivery=True)
        _ffmpeg("-i", video, "-i", native_master, "-map", "0:v:0", "-map", "1:a?",
                "-c", "copy", "-movflags", "+faststart", staged)
        native = finish._stream_spec(native_master, hash_audio=True)
        delivered = finish._stream_spec(staged, hash_audio=True)
        if delivered[2:] != native[2:] or finish._count_frames(staged) != finish._count_frames(native_master):
            raise RuntimeError("VSR master does not match the native frame clock and audio.")
        staged.replace(output)
