from dataclasses import dataclass
import os
from typing import Literal, cast


LLMBackend = Literal["ollama", "vllm"]


@dataclass(frozen=True)
class LLMConfig:
    backend: LLMBackend
    base_url: str
    model: str
    timeout_sec: float
    api_key: str | None = None
    keep_alive: str | None = None


@dataclass(frozen=True)
class Config:
    llm: LLMConfig


@dataclass(frozen=True)
class GenerationOptions:
    num_ctx: int = 4096
    num_predict: int = 128
    temperature: float = 0.2
    top_p: float = 1.0


def require_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise ValueError(
            f"Missing required environment variable: {name}"
        )

    return value.strip()


def optional_env(name: str) -> str | None:
    value = os.getenv(name)

    if value is None:
        return None

    value = value.strip()
    return value or None


def load_llm_config() -> LLMConfig:
    raw_backend = os.getenv(
        "LLM_BACKEND",
        "ollama",
    ).strip().lower()

    if raw_backend not in {"ollama", "vllm"}:
        raise ValueError(
            "LLM_BACKEND must be 'ollama' or 'vllm', "
            f"got: {raw_backend}"
        )

    backend = cast(LLMBackend, raw_backend)

    if backend == "ollama":
        host = require_env("OLLAMA_HOST")
        port = int(os.getenv("OLLAMA_PORT", "11434"))

        return LLMConfig(
            backend="ollama",
            base_url=f"http://{host}:{port}",
            model=require_env("OLLAMA_MODEL"),
            timeout_sec=float(
                os.getenv("OLLAMA_TIMEOUT_SEC", "120")
            ),
            keep_alive=os.getenv(
                "OLLAMA_KEEP_ALIVE",
                "30m",
            ).strip(),
        )

    return LLMConfig(
        backend="vllm",
        base_url=require_env("VLLM_BASE_URL").rstrip("/"),
        model=require_env("VLLM_MODEL"),
        timeout_sec=float(
            os.getenv("VLLM_TIMEOUT_SEC", "120")
        ),
        api_key=optional_env("VLLM_API_KEY"),
    )


def load_env() -> Config:
    from dotenv import load_dotenv

    load_dotenv()

    return Config(
        llm=load_llm_config(),
    )
