"""Cloudflare Workers AI speech-to-text (Deepgram Nova-3) -> ElevenLabs Scribe-shaped JSON.

The rest of video-use (pack_transcripts.py, render.py, timeline_view.py) reads the
Scribe response layout: a top-level 'words' list whose entries have
'type' ('word' | 'spacing' | 'audio_event'), 'text', 'start', 'end' and
'speaker_id'. This module calls Nova-3 and converts its output to that layout,
so nothing downstream has to change.

Differences from Scribe:
  * No audio-event tags - (laughter), (applause) never appear.
  * Speaker ids are only consistent inside one request. Long files are split into
    chunks, so with --num-speakers > 1 speaker labels may swap between chunks.

Env (or .env):
    CLOUDFLARE_ACCOUNT_ID
    CLOUDFLARE_API_TOKEN     (token needs "Workers AI" read/edit permission)
    CLOUDFLARE_STT_MODEL     optional, default @cf/deepgram/nova-3
"""

from __future__ import annotations

import base64
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import requests

DEFAULT_MODEL = "@cf/deepgram/nova-3"
API_BASE = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
CHUNK_SECONDS = 600  # target chunk length for long takes
SNAP_WINDOW = 30.0   # look this far either side of a boundary for a silence to cut at


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _silence_midpoints(path: Path) -> list[float]:
    """Midpoints of silences >= 0.3s, found with ffmpeg silencedetect."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def plan_chunks(path: Path, chunk_seconds: float = CHUNK_SECONDS) -> list[tuple[float, float]]:
    """(start, end) windows covering the file, cut at silences where possible."""
    total = _wav_duration(path)
    if total <= chunk_seconds * 1.2:
        return [(0.0, total)]
    mids = _silence_midpoints(path)
    cuts: list[float] = []
    target = chunk_seconds
    while target < total - chunk_seconds * 0.2:
        near = [m for m in mids if abs(m - target) <= SNAP_WINDOW and m > (cuts[-1] if cuts else 0)]
        cut = min(near, key=lambda m: abs(m - target)) if near else target
        cuts.append(cut)
        target = cut + chunk_seconds
    edges = [0.0, *cuts, total]
    return list(zip(edges[:-1], edges[1:]))


def _slice(src: Path, dest: Path, start: float, end: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _call_nova(audio: Path, account: str, token: str, model: str,
               language: str | None, diarize: bool) -> dict:
    body: dict = {
        "audio": {"body": base64.b64encode(audio.read_bytes()).decode(), "contentType": "audio/wav"},
        "punctuate": True,
        "filler_words": True,   # keep "um"/"uh" so the editor can cut them
        "diarize": diarize,
    }
    if language:
        body["language"] = language
    else:
        body["detect_language"] = True
    resp = requests.post(
        API_BASE.format(account=account, model=model),
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=1800,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Cloudflare Workers AI returned {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    if not data.get("success", True):
        raise RuntimeError(f"Cloudflare Workers AI error: {data.get('errors')}")
    return data.get("result", data)


def nova_to_scribe(result: dict, offset: float = 0.0) -> list[dict]:
    """Convert one Nova-3 result into Scribe-style word / spacing entries."""
    try:
        alt = result["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        return []
    out: list[dict] = []
    prev_end: float | None = None
    for w in alt.get("words", []):
        start = float(w["start"]) + offset
        end = float(w["end"]) + offset
        if prev_end is not None and start > prev_end:
            out.append({"text": " ", "start": prev_end, "end": start, "type": "spacing",
                        "speaker_id": None})
        spk = w.get("speaker")
        out.append({
            "text": w.get("punctuated_word") or w.get("word", ""),
            "start": start,
            "end": end,
            "type": "word",
            "speaker_id": f"speaker_{spk}" if spk is not None else "speaker_0",
            "logprob": None,
        })
        prev_end = end
    return out


def transcribe_audio(audio: Path, account: str, token: str, model: str = DEFAULT_MODEL,
                     language: str | None = None, num_speakers: int | None = None) -> dict:
    """Transcribe a mono 16 kHz wav, chunking long files. Returns a Scribe-shaped dict."""
    diarize = num_speakers is None or num_speakers > 1
    windows = plan_chunks(audio)
    words: list[dict] = []
    detected = language
    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, end) in enumerate(windows):
            if len(windows) == 1:
                piece = audio
            else:
                piece = Path(tmp) / f"chunk{i:03d}.wav"
                _slice(audio, piece, start, end)
            result = _call_nova(piece, account, token, model, language, diarize)
            if not detected:
                detected = ((result.get("results", {}).get("channels") or [{}])[0]
                            .get("detected_language"))
            words.extend(nova_to_scribe(result, offset=start))
    text = " ".join(w["text"] for w in words if w["type"] == "word")
    return {
        "language_code": detected or "unknown",
        "language_probability": None,
        "text": text,
        "words": words,
        "provider": f"cloudflare:{model}",
    }
