# SPDX-License-Identifier: GPL-3.0-or-later
"""R49 managed output paths, final film provenance and bounded segment cleanup."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

ROOT_NAME = "IAMCCS_R49_MINIMAX_BACKEND"
SEGMENTS = "Segments"
FILMS = "Films"


def managed_prompt(prompt):
    return isinstance(prompt, dict) and any(
        isinstance(node, dict) and (
            node.get("class_type") == "IAMCCS_MiniMaxH3UniversalDeliverySourceR49"
            or (node.get("class_type") == "IAMCCS_MiniMaxH3UniversalNativeCheckpointR42"
                and str((node.get("inputs") or {}).get("filename_prefix", "")).startswith(ROOT_NAME + "/"))
        )
        for node in prompt.values()
    )


def output_root():
    import folder_paths

    return Path(folder_paths.get_output_directory()).resolve()


def _plain(path, root):
    """Reject symlinks, junctions and other reparse points below the output root."""
    for part in (path, *(p for p in path.parents if p != root and p.is_relative_to(root))):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return False
    return path.is_relative_to(root) and path.resolve() == path


def _managed_path(*parts):
    root = output_root()
    path = root.joinpath(ROOT_NAME, *parts)
    if not _plain(path, root):
        raise ValueError("R49 output path must remain inside the ComfyUI output directory.")
    return path


def segment_directory(prompt, legacy, *parts):
    return _managed_path(SEGMENTS, *parts) if managed_prompt(prompt) else legacy


def active_segment_directory(legacy, *parts):
    from .iamccs_minimax_h3_shotboard import _current_prompt

    try:
        prompt, _, _, _ = _current_prompt()
    except RuntimeError:
        return legacy
    return segment_directory(prompt, legacy, *parts)


def continuous_segment_directory():
    return _managed_path(SEGMENTS, "ContinuousAV")


def context_file(legacy):
    # Read and write the same path across automatically queued R49 chunks.
    folder = active_segment_directory(legacy.parent, "Context", legacy.parent.name)
    return folder / legacy.name


def _read_json(path):
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def _write_json(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        staged.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _json_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value


def publish_film(source, prompt, extra_pnginfo, plan, provenance_source=None):
    """Publish a stream copy with the same metadata tags as ComfyUI Save Video."""
    from comfy_api.latest import InputImpl

    root = output_root()
    source = Path(source).resolve(strict=True)
    if not _plain(source, root):
        raise ValueError("R49 final film must originate inside the ComfyUI output directory.")
    if source.is_relative_to(_managed_path(FILMS)):
        return source
    origin = Path(provenance_source or source).resolve(strict=True)
    if not _plain(origin, root):
        raise ValueError("R49 film metadata source must be inside the ComfyUI output directory.")
    segments = _managed_path(SEGMENTS)
    relative = source.relative_to(segments) if source.is_relative_to(segments) else source.relative_to(root)
    destination = _managed_path(FILMS, *relative.parts)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("R49 final film publishing expects an MP4 source.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = source.stat()
    fingerprint = [source.relative_to(root).as_posix(), info.st_size, info.st_mtime_ns]
    suffix = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()[:12]
    destination = destination.with_name(f"{destination.stem}_{suffix}.mp4")
    metadata_path = destination.with_suffix(".mp4.iamccs.json")
    existing = _read_json(metadata_path)
    if destination.is_file() and existing.get("source_fingerprint") == fingerprint:
        return destination

    metadata = _read_json(origin.with_suffix(".mp4.iamccs.json"))
    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    if isinstance(workflow, str):
        try:
            workflow = json.loads(workflow)
        except ValueError:
            workflow = None
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        workflow = _read_json(origin.with_suffix(".mp4.source_workflow.json"))
    native_tags = {}
    if workflow:
        workflow = _json_values(workflow)
        native_tags["workflow"] = workflow
        workflow_path = destination.with_suffix(".mp4.source_workflow.json")
        _write_json(workflow_path, workflow)
        metadata["source_workflow_file"] = workflow_path.name
    if isinstance(prompt, dict) and prompt:
        # ComfyUI adds execution fields such as is_changed=[NaN]. Preserve
        # authored inputs and discard runtime cache data for JSON.parse on drop.
        authored_prompt = _json_values({
            node_id: {key: value for key, value in node.items() if key in ("class_type", "inputs", "_meta")}
            for node_id, node in prompt.items() if isinstance(node, dict)
        })
        native_tags["prompt"] = authored_prompt
        prompt_path = destination.with_suffix(".mp4.source_prompt.json")
        _write_json(prompt_path, authored_prompt)
        metadata["source_prompt_file"] = prompt_path.name
    if not native_tags:
        raise ValueError("R49 final film needs a source workflow or API prompt for reimport.")
    generation = metadata.get("generation")
    generation = dict(generation) if isinstance(generation, dict) else {}
    generation.pop("segment_index", None)
    generation["shotplan"] = plan
    metadata["delivery_source"] = origin.relative_to(root).as_posix()
    metadata.update(schema="iamccs.minimax_h3.film", generation=generation, source_fingerprint=fingerprint)
    _write_json(metadata_path, _json_values(metadata))
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="r49_publish_") as temporary:
        staged = Path(temporary) / "film.mp4"
        InputImpl.VideoFromFile(str(source)).save_to(str(staged), metadata=native_tags)
        staged.replace(destination)
    return destination


def archived_continuation_master(chain):
    """Return the durable, complete prefix recorded by a Saved Take."""
    path = chain.get("master_path")
    if chain.get("archived_master") is not True or not isinstance(path, str) or not path:
        return None
    root = output_root()
    films = _managed_path(FILMS)
    movie = root / path
    sidecar = movie.with_suffix(".mp4.iamccs.json")
    if (movie.suffix.lower() == ".mp4" and _plain(movie, films) and _plain(sidecar, films)
            and movie.is_file() and sidecar.is_file()):
        return movie
    return None


def _saved_take_dependencies(folder):
    root = output_root()
    library = root / "IAMCCS" / "MiniMaxH3" / "CONTINUATION"
    if not _plain(library, root):
        return set()
    protected = set()
    for sidecar in library.rglob("*.chain.json"):
        checkpoint = sidecar.with_suffix("").with_suffix(".safetensors")
        if not (_plain(sidecar, library) and _plain(checkpoint, library) and checkpoint.is_file()):
            continue
        try:
            chain = _read_json(sidecar)
        except (OSError, ValueError):
            continue
        if chain.get("schema") != "iamccs.minimax_h3.continuation_chain.v1" or archived_continuation_master(chain):
            continue
        for key in ("segments", "soft_segments", "master_path"):
            paths = chain.get(key)
            for value in paths if isinstance(paths, list) else [paths]:
                if isinstance(value, str) and value:
                    movie = root / value
                    if movie.suffix.lower() == ".mp4" and _plain(movie, folder) and movie.is_file():
                        protected.add(movie)
    return protected


def segment_inventory():
    folder = _managed_path(SEGMENTS)
    films = _managed_path(FILMS)
    complete_runs = {}
    delivered = {}
    for sidecar in films.rglob("*.mp4.iamccs.json"):
        movie = sidecar.with_suffix("").with_suffix("")
        if not _plain(sidecar, films) or not _plain(movie, films) or not movie.is_file():
            continue
        try:
            metadata = _read_json(sidecar)
            archived_at = movie.stat().st_mtime_ns
        except (OSError, ValueError):
            continue
        if metadata.get("schema") != "iamccs.minimax_h3.film":
            continue
        source = metadata.get("delivery_source")
        if isinstance(source, str):
            delivered[source] = max(archived_at, delivered.get(source, 0))
        generation = metadata.get("generation")
        run = generation.get("render_id") if isinstance(generation, dict) else None
        if isinstance(run, str) and run:
            complete_runs[run] = max(archived_at, complete_runs.get(run, 0))
    entries = {}
    protected = set()
    for directory, subdirs, files in os.walk(folder, followlinks=False):
        directory = Path(directory)
        subdirs[:] = [name for name in subdirs if _plain(directory / name, folder)]
        for name in files:
            path = directory / name
            if not _plain(path, folder):
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                entries[path] = (info.st_size, info.st_mtime_ns)
                if path.name.lower() == "final_film.mp4" or re.search(r"_full\.mp4$", name, re.IGNORECASE):
                    try:
                        sidecar = path.with_suffix(".mp4.iamccs.json")
                        data = _read_json(sidecar) if _plain(sidecar, folder) else {}
                        generation = data.get("generation")
                        run = generation.get("render_id") if isinstance(generation, dict) else None
                    except (OSError, ValueError):
                        run = None
                    relative = path.relative_to(folder.parents[1]).as_posix()
                    archived_at = delivered.get(relative, 0)
                    if not name.lower().endswith("_run_and_gun_full.mp4") and isinstance(run, str):
                        archived_at = max(archived_at, complete_runs.get(run, 0))
                    if info.st_mtime_ns > archived_at:
                        protected.add(path)
    lineage = _saved_take_dependencies(folder)
    for movie in protected | lineage:
        entries.pop(movie, None)
        for suffix in (".iamccs.json", ".source_workflow.json", ".source_prompt.json"):
            entries.pop(movie.with_suffix(movie.suffix + suffix), None)
    key = hashlib.sha256(json.dumps(sorted((p.relative_to(folder).as_posix(), *value)
                                          for p, value in entries.items())).encode()).hexdigest()
    return entries, {"files": len(entries), "bytes": sum(size for size, _ in entries.values()),
                     "protected_masters": len(protected), "protected_lineage": len(lineage - protected), "token": key}


def register_routes(routes, web, prompt_server):
    @routes.get("/api/iamccs/h3/r49-segments")
    async def inspect_segments(request):
        try:
            _, inventory = segment_inventory()
        except (OSError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=409)
        return web.json_response(inventory, headers={"Cache-Control": "no-store"})

    @routes.post("/api/iamccs/h3/r49-segments/purge")
    async def purge_segments(request):
        try:
            data = await request.json()
            queue = prompt_server.prompt_queue
            if queue.get_tasks_remaining():
                raise ValueError("Wait until the ComfyUI queue is empty before purging segments.")
            entries, inventory = segment_inventory()
            if not isinstance(data, dict) or data.get("token") != inventory["token"]:
                raise ValueError("The segment list changed. Review the current count and try again.")
        except (OSError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=409)
        folder = _managed_path(SEGMENTS)
        deleted = reclaimed = 0
        skipped = []
        for path, fingerprint in entries.items():
            try:
                if not _plain(path, folder):
                    raise ValueError("Path changed")
                current = path.lstat()
                if (current.st_size, current.st_mtime_ns) != fingerprint:
                    raise ValueError("File changed")
                path.unlink()
                deleted += 1
                reclaimed += fingerprint[0]
            except (OSError, ValueError) as exc:
                skipped.append({"file": path.relative_to(folder).as_posix(), "error": str(exc)})
        return web.json_response({"deleted": deleted, "bytes": reclaimed, "skipped": skipped})
