"""Validate provider keys for the room voice assistant."""

from dataclasses import dataclass
from typing import Mapping
import os


@dataclass(frozen=True)
class ProviderKeys:
    openai: str
    groq: str
    cartesia: str


def load_provider_keys(environ: Mapping[str, str] | None = None) -> ProviderKeys:
    """Read provider keys and fail with actionable variable names when required."""
    environ = os.environ if environ is None else environ
    keys = ProviderKeys(
        openai=(environ.get("OPENAI_API_KEY") or "").strip(),
        groq=(environ.get("GROQ_API_KEY") or "").strip(),
        cartesia=(environ.get("CARTESIA_API_KEY") or "").strip(),
    )
    missing = []
    if not (keys.openai or keys.groq):
        missing.append("GROQ_API_KEY or OPENAI_API_KEY")
    if not keys.cartesia:
        missing.append("CARTESIA_API_KEY")
    if missing:
        raise RuntimeError("Configure the following provider key(s) in .env: " + ", ".join(missing))
    return keys

