"""Local web server: access control, spoiler-safe answers over HTTP, ingestion and key handling."""
import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from companion import web
from tests.test_companion import CHAPTERS, SECRETS, make_epub
from tests.test_keystore import SECRET as FAKE_KEY  # the one deliberately fake key lives in test_keystore

TOKEN = "test-token-123"


class WebCase(unittest.TestCase):
    """Server, token and isolated config shared by the web tests."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        env = patch.dict(os.environ, {"COMPANION_CONFIG": str(folder / "config.json"), "COMPANION_NO_KEYRING": "1",
                                      "HOME": str(folder), "USERPROFILE": str(folder)}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.app = web.App(folder / "lib.db", token=TOKEN)
        library = self.app.library()
        library.add_work("S", CHAPTERS)
        library.close()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), None)
        self.server.RequestHandlerClass = web.make_handler(self.app, lambda: self.server.server_address[1])
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method, path, body=None, token=TOKEN, host=None, content_type="application/json", raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        headers = {"Host": host or f"127.0.0.1:{self.port}"}
        if token is not None:
            headers["X-Token"] = token
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if payload is not None:
            headers["Content-Type"] = content_type
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, dict(response.getheaders()), data

    def api(self, method, path, body=None, **kwargs):
        status, _, data = self.call(method, path, body, **kwargs)
        return status, json.loads(data)


class WebTests(WebCase):
    # ---- access control -------------------------------------------------------------------------------------
    def test_server_listens_on_loopback_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_page_needs_the_token_and_sets_a_strict_policy(self):
        self.assertEqual(self.call("GET", "/", token=None)[0], 403)
        self.assertEqual(self.call("GET", "/?t=wrong", token=None)[0], 403)
        status, headers, data = self.call("GET", f"/?t={TOKEN}", token=None)
        self.assertEqual(status, 200)
        html = data.decode()
        self.assertIn(f'const TOKEN = "{TOKEN}"', html)
        self.assertNotIn("__NONCE__", html)
        policy = headers["Content-Security-Policy"]
        self.assertIn("default-src 'none'", policy)
        self.assertIn("frame-ancestors 'none'", policy)
        self.assertNotIn("unsafe-inline", policy)
        self.assertNotIn("http://", html.replace("http://localhost:11434", ""))  # no external resources

    def test_api_rejects_missing_or_wrong_token_and_foreign_hosts(self):
        self.assertEqual(self.api("GET", "/api/state", token=None)[0], 403)
        self.assertEqual(self.api("GET", "/api/state", token="nope")[0], 403)
        self.assertEqual(self.api("GET", "/api/state", host="evil.example.com")[0], 403)  # DNS rebinding
        self.assertEqual(self.api("GET", "/api/state", host=f"127.0.0.1:{self.port + 1}")[0], 403)
        self.assertEqual(self.api("GET", "/api/state")[0], 200)
        self.assertEqual(self.api("GET", "/api/state", host=f"localhost:{self.port}")[0], 200)

    def test_non_ascii_tokens_get_a_403_instead_of_crashing_the_handler(self):
        for token in ("é", "café-token", "☃"):
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            connection.putrequest("GET", "/api/state", skip_host=True)
            connection.putheader("Host", f"127.0.0.1:{self.port}")
            connection.putheader("X-Token", token.encode("latin-1", "replace").decode("latin-1"))
            connection.endheaders()
            self.assertEqual(connection.getresponse().status, 403, token)
            connection.close()
        for query in ("%C3%A9", "t%C3%A9", "%E2%98%83"):
            status, _, _ = self.call("GET", f"/?t={query}", token=None)
            self.assertEqual(status, 403, query)
        self.assertEqual(self.api("GET", "/api/state")[0], 200)  # the server is still alive

    def test_negative_or_invalid_content_length_is_rejected_without_hanging(self):
        for value in ("-1", "-100", "abc"):
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            connection.putrequest("POST", "/api/ask", skip_host=True)
            connection.putheader("Host", f"127.0.0.1:{self.port}")
            connection.putheader("X-Token", TOKEN)
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", value)
            connection.endheaders()
            self.assertEqual(connection.getresponse().status, 400, value)  # a timeout here would mean it hangs
            connection.close()
        self.assertEqual(self.api("GET", "/api/state")[0], 200)

    def test_cross_site_style_posts_are_refused(self):
        status, _ = self.api("POST", "/api/ask", raw=b"title=S&question=x", content_type="application/x-www-form-urlencoded")
        self.assertEqual(status, 415)
        self.assertEqual(self.api("POST", "/api/ask", {"title": "S"}, token=None)[0], 403)

    def test_request_size_limits(self):
        with patch.object(web, "MAX_JSON", 10):
            self.assertEqual(self.api("POST", "/api/locate", {"title": "S", "phrase": "x" * 50})[0], 400)
        with patch.object(web, "MAX_BOOK", 10):
            status, _ = self.api("POST", "/api/ingest?filename=a.txt", raw=b"x" * 50, content_type="application/octet-stream")
            self.assertEqual(status, 400)

    # ---- spoiler safety over HTTP ---------------------------------------------------------------------------
    def test_answers_never_contain_text_after_the_reader_position(self):
        for chapter in range(0, len(CHAPTERS) + 1):
            future = SECRETS[chapter:]
            question = "what is the " + " and the ".join(f"{s}ite crystal" for s in SECRETS)
            for path in ("/api/ask", "/api/recap"):
                status, data = self.api("POST", path, {"title": "S", "chapter": chapter, "question": question})
                self.assertEqual(status, 200, data)
                blob = json.dumps(data)
                for secret in future:
                    self.assertNotIn(secret + "ite crystal found", blob.replace(question, ""))
                    self.assertNotIn(f"chapter {SECRETS.index(secret) + 1} the", blob)
                for passage in data["passages"]:
                    self.assertLessEqual(passage["chapter"], max(chapter, 0))

    def test_a_forged_ordinal_cannot_exceed_the_book_and_future_questions_are_refused(self):
        self.assertEqual(self.api("POST", "/api/ask", {"title": "S", "ordinal": 10**6, "question": "crystal"})[0], 400)
        self.assertEqual(self.api("POST", "/api/ask", {"title": "S", "ordinal": -1, "question": "crystal"})[0], 400)
        status, data = self.api("POST", "/api/ask", {"title": "S", "chapter": 1, "question": "How does it end?"})
        self.assertEqual(status, 200)
        self.assertEqual(data["passages"], [])
        self.assertIn("not established", data["answer"])

    def test_characters_and_cards_over_http_respect_the_position(self):
        # CHAPTERS plant a secret word per chapter; names found at chapter 1 must not include later ones
        status, early = self.api("POST", "/api/characters", {"title": "S", "chapter": 1})
        self.assertEqual(status, 200)
        blob = json.dumps(early)
        for secret in SECRETS[1:]:
            self.assertNotIn(secret, blob.lower())
        self.assertEqual(self.api("POST", "/api/characters", {"title": "S", "ordinal": 10**6})[0], 400)
        self.assertEqual(self.api("POST", "/api/card", {"title": "S", "chapter": 1})[0], 400)  # no name given
        self.assertEqual(self.api("POST", "/api/card", {"title": "S", "chapter": 1, "name": "Nobody"})[0], 404)
        if early:
            status, card = self.api("POST", "/api/card", {"title": "S", "chapter": 1, "name": early[0]["name"]})
            self.assertEqual(status, 200)
            self.assertLessEqual(card["last"]["chapter"], 1)

    def test_unknown_book_and_chapter(self):
        self.assertEqual(self.api("POST", "/api/ask", {"title": "Nope", "chapter": 1, "question": "x"})[0], 400)
        self.assertEqual(self.api("POST", "/api/ask", {"title": "S", "chapter": 99, "question": "crystal"})[0], 400)

    def test_agent_mode_without_an_llm_is_a_clear_error_not_a_crash(self):
        status, data = self.api("POST", "/api/ask", {"title": "S", "chapter": 1, "question": "crystal", "agent": True})
        self.assertEqual(status, 502)
        self.assertIn("No LLM configured", data["error"])

    def test_locate_returns_a_position(self):
        status, data = self.api("POST", "/api/locate", {"title": "S", "phrase": "Marlowe found the bravoite crystal near the harbour"})
        self.assertEqual((status, data["chapter"]), (200, 2))
        status, _ = self.api("POST", "/api/locate", {"title": "S", "phrase": "zebra pineapple unrelated words"})
        self.assertEqual(status, 404)

    # ---- library --------------------------------------------------------------------------------------------
    def test_ingest_list_chapters_and_delete(self):
        epub_path = Path(self.temp.name) / "b.epub"
        body = "Some long sentence about nothing in particular. " * 8
        make_epub(epub_path, [("One", body + "first"), ("Two", body + "second")])
        status, data = self.api("POST", "/api/ingest?filename=../../evil.epub&title=My%20Book",
                                raw=epub_path.read_bytes(), content_type="application/octet-stream")
        self.assertEqual((status, data["title"], data["chapters"]), (200, "My Book", 2))
        state = self.api("GET", "/api/state")[1]
        self.assertEqual({b["title"] for b in state["books"]}, {"S", "My Book"})
        self.assertEqual([c["title"] for c in self.api("GET", "/api/chapters?title=My%20Book")[1]], ["One", "Two"])
        self.assertEqual(self.api("POST", "/api/delete", {"title": "My Book"})[0], 200)
        self.assertEqual({b["title"] for b in self.api("GET", "/api/state")[1]["books"]}, {"S"})
        self.assertEqual(self.api("POST", "/api/delete", {"title": "My Book"})[0], 400)

    def test_epub_with_namespaces_and_heading_titles_like_a_browser_made_one(self):
        """Same structure as the EPUB built in the browser check: namespaced container/package, h1 titles."""
        from zipfile import ZIP_STORED, ZipFile

        path = Path(self.temp.name) / "ns.epub"
        body = "<p>" + "The tide rose over the old pier while the gulls wheeled above the grey water. " * 6 + "</p>"
        item = '<item id="{0}" href="{0}.xhtml" media-type="application/xhtml+xml"/>'
        with ZipFile(path, "w", ZIP_STORED) as z:
            z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                       'version="1.0"><rootfiles><rootfile full-path="OEBPS/book.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
            z.writestr("OEBPS/book.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><manifest>'
                       + item.format("c0") + item.format("c1") + '</manifest><spine><itemref idref="c1"/><itemref idref="c0"/></spine></package>')
            z.writestr("OEBPS/c0.xhtml", f'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Second in file order</h1>{body}</body></html>')
            z.writestr("OEBPS/c1.xhtml", f'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body><h1>First in reading order</h1>{body}</body></html>')
        status, data = self.api("POST", "/api/ingest?filename=ns.epub&title=NS", raw=path.read_bytes(), content_type="application/octet-stream")
        self.assertEqual((status, data["chapters"]), (200, 2))
        titles = [c["title"] for c in self.api("GET", "/api/chapters?title=NS")[1]]
        self.assertEqual(titles, ["First in reading order", "Second in file order"])  # follows the spine, not file order

    def test_ingest_rejects_other_formats_and_garbage(self):
        self.assertEqual(self.api("POST", "/api/ingest?filename=a.exe", raw=b"MZ", content_type="application/octet-stream")[0], 400)
        self.assertEqual(self.api("POST", "/api/ingest?filename=a.txt", raw=b"too short", content_type="application/octet-stream")[0], 400)
        self.assertEqual(self.api("POST", "/api/ingest?filename=a.epub", raw=b"not a zip", content_type="application/octet-stream")[0], 400)

    # ---- the user's key -------------------------------------------------------------------------------------
    def test_key_is_stored_locally_and_never_sent_back(self):
        status, data = self.api("POST", "/api/settings", {"provider": "gemini", "api_key": FAKE_KEY, "model": "m"})
        self.assertEqual(status, 200)
        self.assertEqual(data["llm"]["stored_provider"], "gemini")
        self.assertNotIn(FAKE_KEY, json.dumps(data))
        self.assertNotIn(FAKE_KEY, json.dumps(self.api("GET", "/api/state")[1]))
        self.assertTrue(data["llm"]["configured"])
        self.assertIn(FAKE_KEY, (Path(self.temp.name) / "config.json").read_text())  # on disk, outside the repo
        self.assertTrue(Path(self.temp.name, "config.json").resolve().is_relative_to(Path(self.temp.name).resolve()))
        cleared = self.api("POST", "/api/settings/clear", {})[1]
        self.assertFalse(cleared["llm"]["configured"])
        self.assertEqual(self.api("POST", "/api/settings", {"provider": "gemini", "api_key": " "})[0], 400)
        self.assertEqual(self.api("POST", "/api/settings", {"provider": "bogus", "api_key": "x"})[0], 400)


if __name__ == "__main__":
    unittest.main()
