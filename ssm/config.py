import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path("~/.ssm").expanduser()
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS = {
    "model_path": None,
    "n_ctx": 16384,
    "n_threads": -1,       # -1 = auto (all CPU cores)
    "n_batch": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 1024,
    "rag_db_path": str(CONFIG_DIR / "index"),
    "rag_n_results": 5,
    "chunk_size": 60,
    "chunk_overlap": 10,
}

# Curated model menu — all fit in 64 GB RAM
MODELS = {
    "qwen7b": {
        "repo":  "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file":  "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "size":  "~5 GB",
        "desc":  "Fast. Good for quick completions.",
    },
    "qwen32b": {
        "repo":  "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "file":  "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "size":  "~20 GB",
        "desc":  "Recommended. Beats GPT-4 on coding benchmarks.",
    },
    "deepseek32b": {
        "repo":  "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "file":  "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "size":  "~20 GB",
        "desc":  "Strong reasoning. Great for debugging and architecture.",
    },
    "qwen72b": {
        "repo":  "bartowski/Qwen2.5-72B-Instruct-GGUF",
        "file":  "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "size":  "~43 GB",
        "desc":  "Near-frontier quality. Slower (~1.5 tok/s).",
    },
}


def load() -> dict:
    if CONFIG_FILE.exists():
        data = json.loads(CONFIG_FILE.read_text())
        return {**DEFAULTS, **data}
    return dict(DEFAULTS)


def save(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get(key: str):
    return load().get(key)


def set(key: str, value) -> None:
    config = load()
    config[key] = value
    save(config)
