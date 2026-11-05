"""Turn an EPUB or plain-text book into ordered, position-tagged chunks."""

import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

MAX_CHUNK_CHARS = 700
MIN_CHAPTER_CHARS = 200


@dataclass(frozen=True)
class Chapter:
    title: str
    paragraphs: list[str]
    times: list[tuple[float, float]] | None = None  # (start, end) seconds per paragraph, for subtitles


class _TextExtractor(HTMLParser):
    BLOCKS = {"p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
    HEADINGS = {"h1", "h2", "h3"}

    def __init__(self):
        super().__init__()
        self.paragraphs, self.heading = [], ""
        self._buffer, self._in_heading, self._skip = [], False, 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self.BLOCKS:
            self._flush()
            self._in_heading = tag in self.HEADINGS and not self.heading

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCKS:
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buffer.append(data)

    def _flush(self):
        text = " ".join("".join(self._buffer).split())
        self._buffer = []
        if text:
            if self._in_heading:
                self.heading = text
            self.paragraphs.append(text)
        self._in_heading = False

    def close(self):
        super().close()
        self._flush()


def read_epub(path: Path) -> list[Chapter]:
    """Chapters follow the spine (reading order); empty files are dropped."""
    with zipfile.ZipFile(path) as book:
        container = ElementTree.fromstring(book.read("META-INF/container.xml"))
        rootfile = next(e.get("full-path") for e in container.iter() if e.tag.endswith("rootfile"))
        package = ElementTree.fromstring(book.read(rootfile))
        manifest = {e.get("id"): e.get("href") for e in package.iter() if e.tag.endswith("}item")}
        spine = [e.get("idref") for e in package.iter() if e.tag.endswith("}itemref")]
        base = PurePosixPath(rootfile).parent
        chapters = []
        for number, idref in enumerate(spine, start=1):
            href = manifest.get(idref)
            if not href:
                continue
            html = book.read(str(base / href).lstrip("./")).decode("utf-8", errors="replace")
            parser = _TextExtractor()
            parser.feed(html)
            parser.close()
            if sum(len(p) for p in parser.paragraphs) >= MIN_CHAPTER_CHARS:
                chapters.append(Chapter(parser.heading or f"Section {number}", parser.paragraphs))
    return chapters


_HEADING = re.compile(
    r"^\s*(?:(?:CHAPTER|Chapter|ADVENTURE|Adventure|BOOK|Book|PART|Part"
    r"|CAP[IÍ]TULO|Cap[ií]tulo|CAPITULO|Capitulo|TRATADO|Tratado|LIBRO|Libro|PARTE|Parte"
    r"|AVENTURA|Aventura|ACTO|Acto|JORNADA|Jornada)\s+[\w.\-]+.*"
    # "Letter 1" / "Carta 2" are chapters; "Carta de don Quijote a Dulcinea" is a letter inside one.
    r"|(?:LETTER|Letter|CARTA|Carta)\s+(?:[IVXLC]+\b|\d+\b|[Pp]rimera\b|[Ss]egunda\b|[Tt]ercera\b).*"
    r"|(?:PREFACE|Preface|INTRODUCTION|Introduction|FOREWORD|Foreword|AFTERWORD|Afterword|CONTENTS|Contents"
    r"|NOTES|Notes|FOOTNOTES|APPENDIX|Appendix"
    r"|PR[OÓ]LOGO|Pr[oó]logo|PREFACIO|Prefacio|INTRODUCCI[OÓ]N|Introducci[oó]n|[IÍ]NDICE|[IÍ]ndice"
    r"|NOTAS|Notas|AP[EÉ]NDICE|Ap[eé]ndice|DEDICATORIA|Dedicatoria|AGRADECIMIENTOS|Agradecimientos"
    r"|EP[IÍ]LOGO|Ep[ií]logo)\b.*"
    r"|(?:PRIMERA|Primera|SEGUNDA|Segunda|TERCERA|Tercera|CUARTA|Cuarta)\s+(?:PARTE|parte)\b.*"
    r"|[IVXLC]+\.\s*[A-ZÁÉÍÓÚÑ].*)\s*$")
# A numbered chapter line may carry its whole title ("Capítulo IV. De lo que le sucedió ..."), so it may be long.
_NUMBERED = re.compile(
    r"^\s*(?:chapter|cap[ií]tulo|tratado|parte|libro|carta|book|part|letter)\s+"
    r"(?:[ivxlcdm]+|\d+|primer[oa]|segund[oa]|tercer[oa]|cuart[oa]|quint[oa]|sext[oa]|s[eé]ptim[oa]|octav[oa]"
    r"|noven[oa]|d[eé]cim[oa]|[uú]ltim[oa])\b", re.I)
# Paratext often discusses the whole plot, so it never counts as text the reader has read.
# (An unnumbered prologue is treated the same way: editions often use it for an editor's introduction.)
_PARATEXT = re.compile(r"^\s*(?:preface|introduction|foreword|afterword|contents|notes|footnotes|appendix"
                       r"|pr[oó]logo|prefacio|introducci[oó]n|[ií]ndice|notas|ap[eé]ndice|dedicatoria|agradecimientos"
                       r"|(?:primera|segunda|tercera|cuarta)\s+parte)\b",  # a part's title page and dedications
                       re.I)
FRONT_MATTER = "Beginning"


def read_text(path: Path) -> list[Chapter]:
    """Plain text (e.g. Project Gutenberg): strip license boilerplate, split on headings.

    Text before the first heading (title page, contents, illustrations) and prefaces,
    introductions and notes are dropped: they are not story text and may reveal the plot.
    """
    data = path.read_bytes()
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError:  # older Spanish and French texts are often Latin-1/Windows-1252
        raw = data.decode("cp1252", errors="replace")
    raw = raw.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF", raw)
    raw = raw[start.end() if start else 0:end.start() if end else len(raw)]
    sections, title, paragraphs = [], FRONT_MATTER, []
    for block in re.split(r"\n\s*\n", raw):
        text = " ".join(block.split())
        if not text:
            continue
        if len(text) < (250 if _NUMBERED.match(text) else 80) and _HEADING.match(text):
            sections.append((title, paragraphs))
            title, paragraphs = text, []
        else:
            paragraphs.append(text)
    sections.append((title, paragraphs))
    headed = any(t != FRONT_MATTER for t, _ in sections)
    return [Chapter(t, p) for t, p in sections
            if sum(len(x) for x in p) >= MIN_CHAPTER_CHARS and not _PARATEXT.match(t)
            and not (headed and t == FRONT_MATTER)]


def read_subtitles(path: Path, title: str | None = None, number: int = 1) -> Chapter:
    """One subtitle file as one timed chapter (an episode, or the whole film)."""
    from . import subtitles

    grouped = subtitles.passages(subtitles.parse_cues(subtitles.decode(path.read_bytes())))
    if not grouped:
        raise ValueError(f"No subtitle lines found in {path.name}")
    return Chapter(title or subtitles.episode_title(path.name, number), [g[0] for g in grouped],
                   [(g[1], g[2]) for g in grouped])


def read_book(path: Path) -> list[Chapter]:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return read_epub(path)
    if suffix in (".txt", ".text"):
        return read_text(path)
    if suffix in (".srt", ".vtt"):
        return [read_subtitles(path)]
    raise ValueError(f"Unsupported format: {suffix} (use .epub, .txt, .srt or .vtt)")


_SENTENCE = re.compile(r"""(?<=[.!?])["”’']?\s+""")


def _pieces(paragraph: str, limit: int) -> list[str]:
    """A paragraph longer than the limit is cut at sentence ends (never mid-sentence unless one is huge)."""
    if len(paragraph) <= limit:
        return [paragraph]
    pieces, current = [], ""
    for sentence in _SENTENCE.split(paragraph):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    return pieces + [current] if current else pieces


def chunk(paragraphs: list[str], limit: int | None = None) -> list[str]:
    """Pack paragraphs (split at sentence ends when too long) into passages of at most `limit` characters.

    Passages never cross a chapter, so reading position stays exact.
    """
    limit = limit or MAX_CHUNK_CHARS
    chunks, current = [], ""
    for paragraph in paragraphs:
        for piece in _pieces(paragraph, limit):
            if current and len(current) + len(piece) + 1 > limit:
                chunks.append(current)
                current = ""
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks
