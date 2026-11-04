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
    r"^\s*(?:(?:CHAPTER|Chapter|ADVENTURE|Adventure|BOOK|Book|PART|Part)\s+[\w.\-]+.*"
    r"|[IVXLC]+\.\s*[A-Z].*)\s*$")


def _is_contents(title: str, paragraphs: list[str]) -> bool:
    """A table of contents lists every later title, so it must not count as story text."""
    return title == "Beginning" and "contents" in " ".join(paragraphs)[:300].lower()


def read_text(path: Path) -> list[Chapter]:
    """Plain text (e.g. Project Gutenberg): strip license boilerplate, split on headings."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF", raw)
    raw = raw[start.end() if start else 0:end.start() if end else len(raw)]
    chapters, title, paragraphs = [], "Beginning", []
    for block in re.split(r"\n\s*\n", raw):
        text = " ".join(block.split())
        if not text:
            continue
        if len(text) < 80 and _HEADING.match(text):
            if sum(len(p) for p in paragraphs) >= MIN_CHAPTER_CHARS and not _is_contents(title, paragraphs):
                chapters.append(Chapter(title, paragraphs))
            title, paragraphs = text, []
        else:
            paragraphs.append(text)
    if sum(len(p) for p in paragraphs) >= MIN_CHAPTER_CHARS:
        chapters.append(Chapter(title, paragraphs))
    return chapters


def read_book(path: Path) -> list[Chapter]:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return read_epub(path)
    if suffix in (".txt", ".text"):
        return read_text(path)
    raise ValueError(f"Unsupported format: {suffix} (use .epub or .txt)")


def chunk(paragraphs: list[str]) -> list[str]:
    """Merge short consecutive paragraphs; never split one across a chapter."""
    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current} {paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks
