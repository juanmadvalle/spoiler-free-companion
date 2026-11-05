"""Subtitle files (.srt / .vtt) as timed passages, so a film or series can be followed by episode and minute.

Subtitles must come from the user: this project ships none. Each file is one episode (or the whole film). Lines are
grouped into short passages that remember when they start and end, and a viewer at minute T has seen only the
passages that END at or before T, so a line that is still on screen or about to be said is never included.
"""

import html
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 400  # shorter than book passages: finer time resolution
MAX_SPAN = 60.0  # seconds covered by one passage
MAX_GAP = 12.0  # a longer silence starts a new passage
MAX_CUES = 20000

_TIME = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")
_AD = re.compile(r"opensubtitles|subtitles? (?:by|downloaded|ripped)|sync(?:ed)? (?:and|&) correct|www\.\w[\w.-]*\.\w+"
                 r"|addic7ed|subscene|yify", re.I)
_EPISODE = re.compile(r"[sS](\d{1,2})[ ._-]?[eE](\d{1,3})|(\d{1,2})x(\d{2})")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _seconds(match: re.Match) -> float:
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000


def decode(data: bytes) -> str:
    """Subtitle files come in UTF-8 (with or without BOM) or in older Windows/Latin-1 encodings."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_cues(text: str) -> list[Cue]:
    """Cues of an SRT or WebVTT text, in time order, with markup, ads and empty lines removed."""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿").strip()):
        lines = block.split("\n")
        at = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if at is None:  # WEBVTT header, NOTE, STYLE or a damaged block
            continue
        start_text, _, end_text = lines[at].partition("-->")
        start, end = _TIME.search(start_text), _TIME.search(end_text)
        if not start or not end:
            continue
        body = " ".join(html.unescape(_TAGS.sub("", line)).strip() for line in lines[at + 1:])
        body = re.sub(r"\s+", " ", body.replace("♪", " ")).strip(" -–—")
        if not body or _AD.search(body):
            continue
        begin, finish = _seconds(start), _seconds(end)
        cues.append(Cue(begin, max(finish, begin), body))
        if len(cues) > MAX_CUES:
            raise ValueError("Too many subtitle lines; is this the right file?")
    cues.sort(key=lambda cue: cue.start)
    return cues


def passages(cues: list[Cue]) -> list[tuple[str, float, float]]:
    """Group consecutive cues into (text, start, end) passages bounded by size, duration and silence."""
    out, parts, begin, last_end = [], [], 0.0, 0.0
    for cue in cues:
        size = sum(len(p) + 1 for p in parts)
        if parts and (size + len(cue.text) > MAX_CHARS or cue.start - begin > MAX_SPAN or cue.start - last_end > MAX_GAP):
            out.append((" ".join(parts), begin, last_end))
            parts = []
        if not parts:
            begin = cue.start
        parts.append(cue.text)
        last_end = max(last_end, cue.end) if len(parts) > 1 else cue.end
    if parts:
        out.append((" ".join(parts), begin, last_end))
    return out


def natural_key(name: str) -> list:
    """Sort 'ep2' before 'ep10'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def episode_title(name: str, number: int) -> str:
    """'S01E02' from the file name when it has one, else the cleaned name, else 'Episode N'."""
    stem = Path(name).stem
    found = _EPISODE.search(stem)
    if found:
        season, episode = (found.group(1), found.group(2)) if found.group(1) else (found.group(3), found.group(4))
        return f"S{int(season):02d}E{int(episode):02d}"
    cleaned = re.sub(r"[._]+", " ", stem).strip()
    return cleaned or f"Episode {number}"


def parse_clock(value: str) -> float:
    """'90', '12:30', '1:02:03' or '1h02m03s' as seconds."""
    value = value.strip().lower()
    found = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s?)?", value)
    if found and any(found.groups()):
        hours, minutes, seconds = (int(g or 0) for g in found.groups())
        return hours * 3600 + minutes * 60 + seconds
    parts = value.split(":")
    if 1 <= len(parts) <= 3 and all(p.isdigit() for p in parts):
        total = 0
        for part in parts:
            total = total * 60 + int(part)
        return float(total)
    raise ValueError(f"Cannot read '{value}' as a time; use 12:30, 1:02:03 or 90")


def format_clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
