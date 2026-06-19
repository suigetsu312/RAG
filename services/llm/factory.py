from logging import Logger

from config import Config, GenerationOptions
from .base import LLMService
from .ollama import OllamaLLMService
from .vllm import VLLMLLMService


def create_llm_service(
    config: Config,
    logger: Logger | None = None,
    default_options: GenerationOptions | None = None,
) -> LLMService:
    if config.llm.backend == "ollama":
        return OllamaLLMService(
            config=config.llm,
            logger=logger,
            default_options=default_options,
        )

    if config.llm.backend == "vllm":
        return VLLMLLMService(
            config=config.llm,
            logger=logger,
            default_options=default_options,
        )

    raise ValueError(
        f"Unsupported LLM backend: {config.llm.backend}"
    )