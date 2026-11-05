"""Where the companion keeps its data on this computer.

By default everything lives under your user profile. Set COMPANION_HOME to keep it elsewhere (for example
C:\\dev\\personal): the library, the saved key and the model cache then use the same names under that folder.
A more specific setting always wins: --db, COMPANION_CONFIG and COMPANION_MODEL_DIR.
"""

import os
from pathlib import Path


def data_home() -> Path:
    """COMPANION_HOME when set, otherwise the user's profile folder."""
    configured = os.environ.get("COMPANION_HOME")
    return Path(configured).expanduser() if configured else Path.home()


def default_db() -> Path:
    return data_home() / ".spoiler-free-companion.db"


def config_file() -> Path:
    return Path(os.environ.get("COMPANION_CONFIG") or data_home() / ".spoiler-free-companion" / "config.json")


def model_cache() -> Path:
    """Where embedding models are stored.

    If COMPANION_HOME was just introduced and has no cache yet, models already downloaded under the profile folder
    are reused instead of being downloaded again.
    """
    if os.environ.get("COMPANION_MODEL_DIR"):
        return Path(os.environ["COMPANION_MODEL_DIR"])
    wanted = data_home() / ".cache" / "spoiler-free-companion" / "models"
    if os.environ.get("COMPANION_HOME") and not wanted.exists():
        try:
            legacy = Path.home() / ".cache" / "spoiler-free-companion" / "models"
        except RuntimeError:  # no profile folder can be determined
            return wanted
        if legacy.exists():
            return legacy
    return wanted
