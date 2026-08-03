"""
config.py
Central configuration for the RAG Document Q&A system.

Loads defaults from config.yaml and allows override via environment
variables (.env). This keeps secrets/host-specific values out of
version control while keeping tunable RAG parameters in a readable
YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file into os.environ if present

BASE_DIR = Path(__file__).resolve().parent
CONFIG_YAML_PATH = BASE_DIR / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dict. Returns {} if the file is missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


@dataclass
class Settings:
    """
    Strongly-typed application settings.

    Precedence: environment variable > config.yaml value > dataclass default.
    """

    # --- Paths ---
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    chroma_dir: Path = BASE_DIR / "chroma_db"
    logs_dir: Path = BASE_DIR / "logs"

    # --- Ollama / LLM ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    temperature: float = 0.2

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k: int = 4

    # --- Chroma ---
    collection_name: str = "documents"

    # --- Logging ---
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # Ensure required directories exist at startup.
        for d in (self.data_dir, self.chroma_dir, self.logs_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


def _coerce(value: str, default: Any) -> Any:
    """Coerce an environment-variable string to match the type of `default`."""
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def load_settings() -> Settings:
    """
    Build a Settings object by merging config.yaml with environment
    variable overrides. Environment variables use the pattern
    RAG_<FIELD_NAME_UPPER>, e.g. RAG_CHUNK_SIZE, RAG_OLLAMA_MODEL.
    """
    yaml_data = _load_yaml(CONFIG_YAML_PATH)
    flat: Dict[str, Any] = {}

    # Flatten the (possibly nested) YAML into constructor kwargs matching
    # the Settings dataclass field names.
    for section in ("paths", "ollama", "embeddings", "chunking", "retrieval", "chroma", "logging"):
        section_data = yaml_data.get(section, {}) or {}
        flat.update(section_data)

    defaults = Settings()
    kwargs: Dict[str, Any] = {}

    for field_name in defaults.__dataclass_fields__:
        default_value = getattr(defaults, field_name)
        value = flat.get(field_name, default_value)

        env_key = f"RAG_{field_name.upper()}"
        if env_key in os.environ:
            value = _coerce(os.environ[env_key], default_value)

        # Path fields must be resolved relative to BASE_DIR if given as strings.
        if field_name in ("data_dir", "chroma_dir", "logs_dir") and isinstance(value, str):
            value = BASE_DIR / value

        kwargs[field_name] = value

    return Settings(**kwargs)


settings = load_settings()
