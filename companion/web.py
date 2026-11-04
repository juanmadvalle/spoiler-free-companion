"""Local web interface. Binds to 127.0.0.1 only; every API call needs a per-run token and a local Host header.

The page and the JSON API are served by the standard library. Nothing leaves this computer except the few
retrieved passages sent to the LLM you configured, exactly as in the command line.
"""

import hmac
import json
import secrets
import tempfile
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import keystore, llm
from .agent import Session, run_agent
from .answer import ask, default_answerer, extractive, recap
from .embed import default_embedder
from .ingest import read_book
from .store import Library

PAGE = Path(__file__).parent / "static" / "index.html"
MAX_JSON = 64 * 1024
MAX_BOOK = 60 * 1024 * 1024
ALLOWED_SUFFIXES = (".epub", ".txt")


class App:
    """Shared state: library file, embedder, token and background jobs."""

    def __init__(self, db: Path, embedder=None, token: str | None = None):
        self.db, self.embedder = Path(db), embedder
        self.token = token or secrets.token_urlsafe(24)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def library(self) -> Library:
        return Library(self.db, embedder=self.embedder)

    # ---- API operations (each returns a JSON-able dict) -------------------------------------------------
    def state(self) -> dict:
        library = self.library()
        try:
            books = [{"title": t, "chapters": len(library.chapters(t)), "semantic": library.has_vectors(t)}
                     for t in library.works()]
        finally:
            library.close()
        config = llm.settings()
        stored = keystore.load()
        return {"books": books, "embeddings_available": self.embedder is not None,
                "llm": {"configured": default_answerer() is not extractive,
                        "provider": "anthropic" if keystore.anthropic_key() else ("compatible" if config else None),
                        "model": config.model if config else None,
                        "stored_provider": stored.get("provider"), "key": keystore.mask(stored.get("api_key", ""))}}

    def chapters(self, title: str) -> list[dict]:
        library = self.library()
        try:
            return [{"number": n, "title": t} for n, t in library.chapters(title)]
        finally:
            library.close()

    @staticmethod
    def horizon(library: Library, title: str, body: dict) -> int:
        """The reader's position: an exact passage ordinal (from a located phrase) or 'finished chapter N'."""
        if body.get("ordinal") is not None:
            ordinal = int(body["ordinal"])
            top = library.db.execute("SELECT MAX(ordinal) FROM chunks WHERE work_id = ?",
                                     (library._work(title),)).fetchone()[0] or 0
            if not 0 <= ordinal <= top:
                raise ValueError("Position out of range")
            return ordinal
        chapter = int(body.get("chapter", 0))
        return 0 if chapter <= 0 else library.horizon_after_chapter(title, chapter)

    def locate(self, body: dict) -> dict:
        library = self.library()
        try:
            ordinal, found = library.locate(str(body["title"]), str(body["phrase"]))
            return {"ordinal": ordinal, "chapter": found.chapter, "chapter_title": found.chapter_title,
                    "excerpt": found.text[:240]}
        finally:
            library.close()

    def answer(self, body: dict, recap_only: bool = False) -> dict:
        title = str(body["title"])
        library = self.library()
        try:
            horizon = self.horizon(library, title, body)
            answerer = default_answerer()
            if recap_only:
                text, passages = recap(library, title, horizon, answerer)
            elif body.get("agent"):
                session = Session(library, title, horizon)
                text, passages = run_agent(session, str(body["question"])), []
                return {"answer": text, "passages": [], "tools": session.log, "mode": "agent", "ordinal": horizon}
            else:
                text, passages = ask(library, title, str(body["question"]), horizon, answerer)
            mode = "passages" if answerer is extractive else "llm"
            return {"answer": text, "mode": mode, "ordinal": horizon,
                    "passages": [{"ordinal": p.ordinal, "chapter": p.chapter, "chapter_title": p.chapter_title,
                                  "text": p.text} for p in passages]}
        finally:
            library.close()

    def ingest(self, filename: str, data: bytes, title: str, embed: bool) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError("Only .epub and .txt files are supported")
        with tempfile.TemporaryDirectory() as folder:  # the uploaded name is never used as a path
            path = Path(folder) / f"upload{suffix}"
            path.write_bytes(data)
            try:
                chapters = read_book(path)
            except ValueError:
                raise
            except Exception:  # corrupt zip, bad XML, missing parts: all mean "not a readable book"
                raise ValueError("Could not read this file as an EPUB or text book") from None
        if not chapters:
            raise ValueError("No readable chapters found")
        title = (title or Path(filename).stem).strip()[:200] or "Untitled"
        library = self.library()
        try:
            passages = library.add_work(title, chapters)
        finally:
            library.close()
        result = {"title": title, "chapters": len(chapters), "passages": passages}
        if embed and self.embedder is not None:
            result["job"] = self._start_embedding(title)
        return result

    def _start_embedding(self, title: str) -> str:
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {"state": "running", "done": 0, "total": 0, "title": title}

        def work():
            library = self.library()
            try:
                library.embed_work(title, progress=lambda done, total: self.jobs[job_id].update(done=done, total=total))
                self.jobs[job_id]["state"] = "done"
            except Exception as exc:  # reported to the page, never crashes the server
                self.jobs[job_id].update(state="error", error=str(exc))
            finally:
                library.close()

        threading.Thread(target=work, daemon=True).start()
        return job_id

    def delete(self, title: str) -> dict:
        library = self.library()
        try:
            library.remove_work(title)
        finally:
            library.close()
        return {"deleted": title}

    def save_settings(self, body: dict) -> dict:
        provider = str(body.get("provider", "gemini"))
        keystore.save(provider, str(body.get("api_key", "")), str(body.get("base_url", "")), str(body.get("model", "")))
        return self.state()

    def models(self) -> dict:
        return {"models": llm.list_models()}


def make_handler(app: App, port_getter):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SpoilerFreeCompanion"
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the terminal quiet; nothing sensitive is logged anyway
            pass

        # ---- helpers ---------------------------------------------------------------------------------------
        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload):
            self._send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

        def _local_host(self) -> bool:
            """Rejects DNS-rebinding: only the exact local origins are served."""
            port = port_getter()
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _authorised(self) -> bool:
            return hmac.compare_digest(self.headers.get("X-Token", ""), app.token)

        def _body(self, limit: int) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            if length > limit:
                raise ValueError("Request too large")
            return self.rfile.read(length)

        def _guard(self) -> bool:
            if not self._local_host():
                self._json(403, {"error": "Forbidden host"})
                return False
            return True

        # ---- routes ----------------------------------------------------------------------------------------
        def do_GET(self):
            if not self._guard():
                return
            url = urlparse(self.path)
            if url.path == "/":
                if not hmac.compare_digest(parse_qs(url.query).get("t", [""])[0], app.token):
                    return self._send(403, "Open the link printed in the terminal (it contains a one-time token)."
                                      .encode(), "text/plain; charset=utf-8")
                nonce = secrets.token_urlsafe(12)
                html = PAGE.read_text(encoding="utf-8").replace("__TOKEN__", app.token).replace("__NONCE__", nonce)
                csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                       "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                return self._send(200, html.encode(), "text/html; charset=utf-8", {"Content-Security-Policy": csp})
            if url.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not url.path.startswith("/api/"):
                return self._json(404, {"error": "Not found"})
            if not self._authorised():
                return self._json(403, {"error": "Bad token"})
            self._dispatch("GET", url, None)

        def do_POST(self):
            if not self._guard():
                return
            url = urlparse(self.path)
            if not url.path.startswith("/api/") or not self._authorised():
                return self._json(403, {"error": "Bad token"})
            try:
                if url.path == "/api/ingest":
                    data = self._body(MAX_BOOK)
                    return self._dispatch("POST", url, data)
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    return self._json(415, {"error": "JSON required"})  # also blocks simple cross-site form posts
                body = json.loads(self._body(MAX_JSON) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("JSON object expected")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            self._dispatch("POST", url, body)

        def _dispatch(self, method: str, url, body):
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path
            try:
                if method == "GET" and path == "/api/state":
                    result = app.state()
                elif method == "GET" and path == "/api/chapters":
                    result = app.chapters(query["title"])
                elif method == "GET" and path == "/api/models":
                    result = app.models()
                elif method == "GET" and path.startswith("/api/jobs/"):
                    result = app.jobs.get(path.rsplit("/", 1)[-1])
                    if result is None:
                        return self._json(404, {"error": "Unknown job"})
                elif method == "POST" and path == "/api/ingest":
                    result = app.ingest(query.get("filename", ""), body, query.get("title", ""),
                                        query.get("embed") == "1")
                elif method == "POST" and path == "/api/locate":
                    result = app.locate(body)
                elif method == "POST" and path == "/api/ask":
                    result = app.answer(body)
                elif method == "POST" and path == "/api/recap":
                    result = app.answer(body, recap_only=True)
                elif method == "POST" and path == "/api/delete":
                    result = app.delete(str(body["title"]))
                elif method == "POST" and path == "/api/settings":
                    result = app.save_settings(body)
                elif method == "POST" and path == "/api/settings/clear":
                    keystore.clear()
                    result = app.state()
                else:
                    return self._json(404, {"error": "Not found"})
            except KeyError as exc:
                return self._json(400, {"error": f"Missing or unknown: {exc}"})
            except LookupError as exc:
                return self._json(404, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                return self._json(400, {"error": str(exc)})
            except (RuntimeError, ImportError) as exc:  # LLM or provider problems
                return self._json(502, {"error": str(exc)})
            except Exception:  # last resort: always answer, never leak internals
                return self._json(500, {"error": "Internal error"})
            self._json(200, result)

    return Handler


def serve(db: Path, port: int = 8765, open_browser: bool = True, embedder=None) -> None:
    app = App(db, embedder=embedder if embedder is not None else default_embedder())
    server = ThreadingHTTPServer(("127.0.0.1", port), None)
    server.RequestHandlerClass = make_handler(app, lambda: server.server_address[1])
    url = f"http://127.0.0.1:{server.server_address[1]}/?t={app.token}"
    print(f"Spoiler-free companion running at {url}\nPress Ctrl+C to stop. Only this computer can reach it.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
