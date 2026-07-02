"""
Loads run configuration from a YAML file (e.g. configs/baseline.yaml).
Falls back to hardcoded defaults if the file is missing or a key isn't set,
so the app still runs even without a config file present.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Fallback defaults -- used if the config file or a specific key is missing.
DEFAULTS = {
    "run": {"name": "baseline_v1", "output_dir": "results"},
    "models": {"embedding_model": "all-MiniLM-L6-v2", "llm_model": "llama3.2"},
    "retrieval": {"top_k": 5},
    "chunking": {"chunk_size": 400, "chunk_overlap": 50},
    "index": {"collection_name": "pdf_chunks", "persist_directory": "./data/chroma_db"},
    "prompt": {"version": "v1"},
    "evaluation": {"questions_file": "evaluation/sample_questions.json", "max_questions": 5},
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "baseline.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Loads the YAML config file and merges it over the defaults.
    Missing sections/keys fall back to DEFAULTS, so a partial or
    missing config file never crashes the app.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning("Config file not found at %s -- using built-in defaults.", config_path)
        return DEFAULTS

    with open(config_path, "r") as f:
        loaded = yaml.safe_load(f) or {}

    # Shallow merge: for each top-level section, fill in missing keys from DEFAULTS.
    merged = {}
    for section, default_values in DEFAULTS.items():
        section_values = loaded.get(section, {}) or {}
        merged[section] = {**default_values, **section_values}

    return merged