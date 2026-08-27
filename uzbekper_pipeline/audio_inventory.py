"""Audio validation + inventory for the UzbekPER Modal pipeline (Task 3).

The 2026-08-25 Vast run treated "file exists and >1KB" as fetched. That let
HTML error pages and truncated parquet chunks masquerade as audio and poisoned
inference. This module makes validation structural:

* :func:`probe_audio` parses the RIFF/WAVE header AND the data chunk, then
  hashes content — raising :class:`AudioProbeError` on anything unreadable.
* :func:`build_inventory` walks ready-manifest rows tolerating per-record
  failures with a status field (present / missing / corrupt).
* :func:`summarize_inventory` aggregates counts and present-audio duration.

Stdlib only (wave, hashlib, struct) so it runs on the VPS and inside CPU-only
Modal images without extra wheels.
"""
from __future__ import annotations

import hashlib
import os
import struct
import wave
from typing import Any


class AudioProbeError(RuntimeError):
    """Raised when a file is not decodable PCM WAV audio."""


_WAV_MIN_BYTES = 44  # canonical bare-bones header size


def probe_audio(path: str) -> dict[str, Any]:
    """Return {'sample_rate','channels','duration_seconds','sha256'} for a WAV.

    Raises:
        FileNotFoundError: path does not exist.
        AudioProbeError: file is empty, not RIFF/WAVE, or has a truncated /
            undecodable data chunk.
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    if len(raw) < _WAV_MIN_BYTES:
        raise AudioProbeError(f"{path}: too small to be WAV ({len(raw)} bytes)")

    # Structural check: RIFF header + WAVE form tag.
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioProbeError(f"{path}: not a RIFF/WAVE file")

    # Walk chunks to find fmt + data.
    pos = 12
    fmt = None
    data_off = None
    data_size = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            if pos + 8 + 16 > len(raw):
                raise AudioProbeError(f"{path}: truncated fmt chunk")
            fmt = raw[pos + 8:pos + 8 + 16]
        elif cid == b"data":
            data_off = pos + 8
            data_size = size
            # The data chunk header must be fully present and its declared
            # body must fit in the file -- otherwise the file is truncated.
            if pos + 8 > len(raw) or data_off + data_size > len(raw):
                raise AudioProbeError(f"{path}: truncated data chunk")
        else:
            # Unknown chunk header must still fit within the file.
            if pos + 8 + size > len(raw):
                raise AudioProbeError(f"{path}: truncated chunk {cid!r}")
        pos += 8 + size + (size & 1)  # chunks are word-aligned

    if fmt is None or data_off is None:
        raise AudioProbeError(f"{path}: missing fmt or data chunk")

    audio_format, channels, rate, _bps, block_align, bits = struct.unpack(
        "<HHIIHH", fmt
    )
    if audio_format != 1:  # PCM
        raise AudioProbeError(f"{path}: non-PCM format {audio_format}")
    if channels < 1 or rate < 1:
        raise AudioProbeError(f"{path}: degenerate fmt chunk")

    bytes_per_frame = max(1, block_align)
    total_frames = data_size // bytes_per_frame

    # Truncation guard: declared data size must fit within the actual file.
    if data_off + data_size > len(raw):
        raise AudioProbeError(f"{path}: truncated data chunk")

    duration = total_frames / float(rate)

    return {
        "sample_rate": rate,
        "channels": channels,
        "duration_seconds": round(duration, 3),
        "bits_per_sample": bits,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_inventory(
    rows: list[dict[str, Any]],
    audio_root: str,
    relpath_key: str = "clip_id",
) -> list[dict[str, Any]]:
    """Walk manifest rows and classify each clip's local file.

    Returns one record per input row (same order as sorted by clip_id), each::

        {"clip_id", "status", ...probe fields...}   status in
        {"present","missing","corrupt"}

    Never raises for individual bad files — corruption is data, not an error.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        cid = str(row.get(relpath_key) or "")
        path = os.path.join(audio_root, cid)
        rec: dict[str, Any] = {"clip_id": cid}
        if not os.path.isfile(path):
            rec["status"] = "missing"
        else:
            try:
                info = probe_audio(path)
                rec.update(info)
                rec["status"] = "present"
            except FileNotFoundError:
                rec["status"] = "missing"
            except AudioProbeError as exc:
                rec["status"] = "corrupt"
                rec["error_kind"] = (
                    "not_wav" if "not a RIFF" in str(exc)
                    else "truncated" if "truncated" in str(exc)
                    else "undecodable"
                )
        out.append(rec)
    out.sort(key=lambda r: r["clip_id"])
    return out


def summarize_inventory(inv: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts and total duration of PRESENT clips only."""
    present = [r for r in inv if r.get("status") == "present"]
    return {
        "present": len(present),
        "missing": sum(1 for r in inv if r.get("status") == "missing"),
        "corrupt": sum(1 for r in inv if r.get("status") == "corrupt"),
        "total_present_seconds": round(
            sum(r.get("duration_seconds", 0.0) for r in present), 2
        ),
    }
