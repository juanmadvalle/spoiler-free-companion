"""COMPANION_HOME moves the library, the saved key and the model cache together; specific settings still win."""
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from companion import embed, keystore, paths
from companion.cli import main
from companion.store import Library


def env(**values):
    """Replace the environment with only these values, so nothing from the machine leaks into a test."""
    return patch.dict(os.environ, {"USERPROFILE": os.environ.get("USERPROFILE", ""), "HOME": os.environ.get("HOME", ""), **values},
                      clear=True)


class PathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "profile"
        self.data = self.root / "dev" / "personal"
        self.home.mkdir()
        patcher = patch("companion.paths.Path.home", return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_without_the_variable_everything_stays_under_the_profile(self):
        with env():
            self.assertEqual(paths.default_db(), self.home / ".spoiler-free-companion.db")
            self.assertEqual(paths.config_file(), self.home / ".spoiler-free-companion" / "config.json")
            self.assertEqual(paths.model_cache(), self.home / ".cache" / "spoiler-free-companion" / "models")

    def test_one_variable_moves_all_three_locations(self):
        with env(COMPANION_HOME=str(self.data)):
            self.assertEqual(paths.default_db(), self.data / ".spoiler-free-companion.db")
            self.assertEqual(paths.config_file(), self.data / ".spoiler-free-companion" / "config.json")
            self.assertEqual(paths.model_cache(), self.data / ".cache" / "spoiler-free-companion" / "models")

    def test_more_specific_settings_win(self):
        with env(COMPANION_HOME=str(self.data), COMPANION_CONFIG=str(self.root / "other.json"),
                 COMPANION_MODEL_DIR=str(self.root / "models")):
            self.assertEqual(paths.config_file(), self.root / "other.json")
            self.assertEqual(paths.model_cache(), self.root / "models")
            self.assertEqual(paths.default_db(), self.data / ".spoiler-free-companion.db")

    def test_models_already_downloaded_under_the_profile_are_reused(self):
        legacy = self.home / ".cache" / "spoiler-free-companion" / "models"
        legacy.mkdir(parents=True)
        with env(COMPANION_HOME=str(self.data)):
            self.assertEqual(paths.model_cache(), legacy)  # no 500 MB download just because the data moved
            moved = self.data / ".cache" / "spoiler-free-companion" / "models"
            moved.mkdir(parents=True)
            self.assertEqual(paths.model_cache(), moved)  # once the models are moved, the new place is used

    def test_a_tilde_in_the_variable_is_expanded(self):
        with env(COMPANION_HOME="~"):
            self.assertEqual(str(paths.data_home()), os.path.expanduser("~"))

    def test_the_embedders_use_the_configured_cache(self):
        seen = {}
        fake = types.ModuleType("fastembed")
        fake.TextEmbedding = lambda name, cache_dir: seen.update(cache=cache_dir) or types.SimpleNamespace(
            embed=lambda texts: [[1.0, 0.0]])
        with env(COMPANION_HOME=str(self.data)), patch.dict(sys.modules, {"fastembed": fake}):
            embed.FastEmbedder("any-model").embed(["x"])
        expected = self.data / ".cache" / "spoiler-free-companion" / "models"
        self.assertEqual(Path(seen["cache"]), expected)
        self.assertTrue(expected.is_dir())  # created where COMPANION_HOME says, not under the profile

    def test_the_saved_key_follows_the_variable(self):
        with env(COMPANION_HOME=str(self.data), COMPANION_NO_KEYRING="1"):
            where = keystore.save("gemini", "fake-key-for-the-test-0123456789")
            self.assertEqual(Path(where), self.data / ".spoiler-free-companion" / "config.json")
            self.assertEqual(keystore.load()["api_key"], "fake-key-for-the-test-0123456789")
            self.assertFalse((self.home / ".spoiler-free-companion").exists())  # nothing written under the profile

    def test_the_command_line_uses_the_library_in_the_configured_home(self):
        self.data.mkdir(parents=True)
        with env(COMPANION_HOME=str(self.data), COMPANION_NO_KEYRING="1"):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(main(["--no-embeddings", "list"]), 0)
            self.assertTrue((self.data / ".spoiler-free-companion.db").exists())
            self.assertFalse((self.home / ".spoiler-free-companion.db").exists())
            library = Library(self.data / ".spoiler-free-companion.db")
            self.assertEqual(library.works(), [])
            library.close()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--db", str(self.root / "explicit.db"), "--no-embeddings", "list"]), 0)
            self.assertTrue((self.root / "explicit.db").exists())  # --db still wins


if __name__ == "__main__":
    unittest.main()
