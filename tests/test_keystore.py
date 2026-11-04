"""The user's own key stays on their machine: stored outside the repo, masked on display, env wins."""
import io
import os
import re
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from companion import keystore, llm
from companion.cli import main

REPO = Path(__file__).resolve().parent.parent
REAL_HOME = Path.home()  # captured before tests clear the environment
SECRET = "AIzaSyFAKEKEYFORTESTSONLY0123456789abcd"
KEY_PATTERNS = [re.compile(p) for p in (r"AIza[0-9A-Za-z_\-]{35}", r"sk-ant-[0-9A-Za-z_\-]{20,}", r"sk-[0-9A-Za-z]{32,}")]


class KeystoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "cfg" / "config.json"
        patcher = patch.dict(os.environ, {"COMPANION_CONFIG": str(self.config), "COMPANION_NO_KEYRING": "1"}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_load_clear_round_trip_and_masking(self):
        self.assertEqual(keystore.load(), {})
        where = keystore.save("gemini", SECRET, model="m1")
        self.assertEqual(where, str(self.config))
        self.assertEqual(keystore.load(), {"provider": "gemini", "model": "m1", "api_key": SECRET})
        self.assertNotIn(SECRET, keystore.mask(SECRET))
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        self.assertTrue(keystore.clear())
        self.assertFalse(keystore.clear())
        self.assertEqual(keystore.load(), {})

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            keystore.save("nope", SECRET)
        with self.assertRaises(ValueError):
            keystore.save("gemini", "   ")
        keystore.save("custom", "", base_url="http://localhost:11434/v1")  # local servers need no key

    def test_stored_key_is_used_and_environment_wins(self):
        keystore.save("gemini", SECRET)
        self.assertEqual(llm.settings().key, SECRET)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}):
            self.assertEqual(llm.settings().key, "from-env")

    def test_anthropic_provider_is_routed_separately(self):
        keystore.save("anthropic", "ant-key")
        self.assertEqual(keystore.anthropic_key(), "ant-key")
        self.assertIsNone(llm.settings())  # not offered to the OpenAI-compatible path
        keystore.save("gemini", SECRET)
        self.assertIsNone(keystore.anthropic_key())

    def test_cli_key_commands_never_print_the_secret(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch.object(sys, "stdin", io.StringIO(SECRET + "\n")):
            self.assertEqual(main(["key", "set", "--stdin"]), 0)
            self.assertEqual(main(["key", "status"]), 0)
        self.assertNotIn(SECRET, out.getvalue() + err.getvalue())
        self.assertIn("provider=gemini", out.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["key", "clear"]), 0)
        self.assertEqual(keystore.load(), {})
        with redirect_stderr(io.StringIO()), patch.object(sys, "stdin", io.StringIO("\n")):
            self.assertEqual(main(["key", "set", "--stdin"]), 1)

    def test_default_config_is_in_the_home_directory_not_the_repo(self):
        with patch.dict(os.environ, {"COMPANION_CONFIG": "", "USERPROFILE": str(REAL_HOME), "HOME": str(REAL_HOME)}):
            path = keystore.config_path().resolve()
        self.assertTrue(path.is_relative_to(REAL_HOME.resolve()))
        self.assertFalse(path.is_relative_to(REPO))

    def test_no_api_key_is_committed_anywhere_in_the_repository(self):
        skipped = {".git", "data", "__pycache__", ".venv"}
        for path in REPO.rglob("*"):
            if path.is_file() and not skipped & set(path.relative_to(REPO).parts) and path.suffix in {
                    ".py", ".md", ".toml", ".json", ".txt", ".cfg", ".ini", ".yml", ".yaml", ""}:
                if path.name == "test_keystore.py":
                    continue  # holds the deliberately fake key above
                text = path.read_text(encoding="utf-8", errors="ignore")
                for pattern in KEY_PATTERNS:
                    self.assertIsNone(pattern.search(text), f"possible API key in {path.relative_to(REPO)}")


if __name__ == "__main__":
    unittest.main()
