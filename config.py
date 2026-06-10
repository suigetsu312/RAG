from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    llm_host: str
    llm_port: int
    llm_model: str
    llm_timeout_sec: float


def require_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")

    return value


def load_env() -> Config:
    from dotenv import load_dotenv

    load_dotenv()

    return Config(
        llm_host=require_env("LLM_HOST"),
        llm_port=int(require_env("LLM_PORT")),
        llm_model=require_env("LLM_MODEL"),
        llm_timeout_sec=float(os.getenv("LLM_TIMEOUT_SEC", "120")),
    )