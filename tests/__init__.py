"""Tests never touch the developer's real settings or keys, and never call a live model.

This runs before any test module, so every test starts with no stored API key and no key in the environment.
Tests that need a key set one explicitly (usually a fake) with their own environment patch.
"""
import os
import tempfile

os.environ["COMPANION_CONFIG"] = os.path.join(tempfile.gettempdir(), "companion-tests-no-config", "config.json")
os.environ["COMPANION_NO_KEYRING"] = "1"
os.environ["COMPANION_HOME"] = os.path.join(tempfile.gettempdir(), "companion-tests-home")  # never the real data folder
for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "COMPANION_LLM_API_KEY", "COMPANION_LLM_BASE_URL"):
    os.environ.pop(name, None)
