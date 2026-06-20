from __future__ import annotations

from config import GenerationOptions
from rag.generators.base import AnswerGenerator
from rag.generators.result import GenerationResult
from services.llm import LLMService


class LLMAnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        llm_service: LLMService,
        options: GenerationOptions | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._options = options

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationResult:
        result = self._llm_service.chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            options=self._options,
        )

        return GenerationResult(
            content=result.content,
            metadata={
                "latency_ms": result.latency_ms,
                "total_duration_ms": result.total_duration_ms,
                "load_duration_ms": result.load_duration_ms,
                "prompt_eval_duration_ms": (
                    result.prompt_eval_duration_ms
                ),
                "eval_duration_ms": result.eval_duration_ms,
                "generation_tokens_per_sec": (
                    result.generation_tokens_per_sec
                ),
            },
        )
