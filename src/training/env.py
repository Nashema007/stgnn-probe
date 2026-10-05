"""Environment helpers for training entry points."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _strip_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_dotenv_value(key: str, dotenv_path: Path | None = None) -> str | None:
    """Read a single key from a simple .env file without mutating os.environ."""
    path = dotenv_path or Path.cwd() / ".env"
    if not path.exists():
        return None

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, raw_value = stripped.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name.removeprefix("export ").strip()
        if name == key:
            value = _strip_env_value(raw_value)
            return value or None
    return None


def get_env_or_dotenv(key: str) -> str | None:
    """Resolve a key from process env, then .env in cwd, then repo-root .env."""
    value = os.getenv(key)
    if value:
        return value

    cwd_dotenv = Path.cwd() / ".env"
    value = read_dotenv_value(key, cwd_dotenv)
    if value:
        return value

    repo_dotenv = _REPO_ROOT / ".env"
    if repo_dotenv != cwd_dotenv:
        return read_dotenv_value(key, repo_dotenv)
    return None


def resolve_wandb_entity(explicit_entity: str | None = None) -> str | None:
    """Resolve W&B entity from explicit config/CLI, environment, then .env."""
    return explicit_entity or get_env_or_dotenv("WANDB_ENTITY")
