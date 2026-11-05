"""Where the user's own API key lives: their machine, never the repository.

Order of precedence everywhere: environment variable, then what `companion key set` stored.
The secret goes to the operating system's credential store when the optional `keyring` package is
installed (Windows Credential Manager, macOS Keychain, ...); otherwise to a config file in the user's
home directory with owner-only permissions. The file is plain text, so prefer `keyring` or an
environment variable on a shared computer.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

SERVICE = "spoiler-free-companion"
PROVIDERS = ("gemini", "anthropic", "custom")


def config_path() -> Path:
    return Path(os.environ.get("COMPANION_CONFIG") or Path.home() / ".spoiler-free-companion" / "config.json")


@lru_cache(maxsize=1)
def _probe():
    """The keyring module if this machine has a usable backend. Probed once per process: it can be slow or prompt."""
    try:
        import keyring

        keyring.get_password(SERVICE, "probe")  # fails on machines without a usable backend
        return keyring
    except Exception:
        return None


def _keyring():
    return None if os.environ.get("COMPANION_NO_KEYRING") else _probe()


_secret: dict = {}  # the key read from the credential store, kept until it is saved or cleared


def load() -> dict:
    """Stored settings with the secret merged in; empty when nothing is stored."""
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError):  # RuntimeError: no home directory can be determined
        return {}
    if not isinstance(data, dict):
        return {}
    if "api_key" not in data and (ring := _keyring()):
        if "value" not in _secret:
            _secret["value"] = ring.get_password(SERVICE, "api_key")
        if _secret["value"]:
            data["api_key"] = _secret["value"]
    return data


def save(provider: str, api_key: str, base_url: str = "", model: str = "") -> str:
    """Store the settings; returns where the secret went ('keyring' or the file path)."""
    if provider not in PROVIDERS:
        raise ValueError(f"Provider must be one of {', '.join(PROVIDERS)}")
    api_key = api_key.strip()
    if not api_key and provider != "custom":
        raise ValueError("The key is empty")
    data = {"provider": provider, **({"base_url": base_url} if base_url else {}), **({"model": model} if model else {})}
    ring = _keyring()
    if ring and api_key:
        ring.set_password(SERVICE, "api_key", api_key)
        where = "the system credential store"
    else:
        data["api_key"] = api_key
        where = str(config_path())
    _secret.clear()
    _write_atomically(config_path(), json.dumps(data, indent=2))
    return where


def _write_atomically(path: Path, text: str) -> None:
    """Write beside the target, then swap it in: a crash or full disk never leaves a half-written key file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # owner-only where the OS supports it
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        path.chmod(0o600)
    except OSError:
        pass


def clear() -> bool:
    removed = False
    _secret.clear()
    if (ring := _keyring()):
        try:
            ring.delete_password(SERVICE, "api_key")
            removed = True
        except Exception:
            pass
    try:
        config_path().unlink()
        removed = True
    except FileNotFoundError:
        pass
    return removed


def mask(key: str) -> str:
    return "(none)" if not key else (key[:4] + "..." + key[-2:] if len(key) > 10 else "***")


def anthropic_key() -> str | None:
    """ANTHROPIC_API_KEY, or the stored key when the stored provider is anthropic."""
    stored = load()
    return os.environ.get("ANTHROPIC_API_KEY") or (stored.get("api_key") if stored.get("provider") == "anthropic" else None) or None
