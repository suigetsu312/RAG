from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from config import GenerationOptions
from rag.generators import LLMAnswerGenerator
from rag.generators.result import GenerationResult
from services.llm import LLMService


def test_generate_forwards_prompts_and_options() -> None:
    llm_service = Mock(spec=LLMService)
    options = Mock(spec=GenerationOptions)

    llm_service.chat.return_value = SimpleNamespace(
        content="RAG generated answer",
        latency_ms=120.0,
        total_duration_ms=110.0,
        load_duration_ms=10.0,
        prompt_eval_duration_ms=20.0,
        eval_duration_ms=80.0,
        generation_tokens_per_sec=25.0,
    )

    generator = LLMAnswerGenerator(
        llm_service=llm_service,
        options=options,
    )

    answer = generator.generate(
        system_prompt="system instructions",
        user_prompt="context and question",
    )

    assert answer == GenerationResult(
        content="RAG generated answer",
        metadata={
            "latency_ms": 120.0,
            "total_duration_ms": 110.0,
            "load_duration_ms": 10.0,
            "prompt_eval_duration_ms": 20.0,
            "eval_duration_ms": 80.0,
            "generation_tokens_per_sec": 25.0,
        },
    )

    llm_service.chat.assert_called_once_with(
        prompt="context and question",
        system_prompt="system instructions",
        options=options,
    )


def test_generate_allows_default_llm_options() -> None:
    llm_service = Mock(spec=LLMService)

    llm_service.chat.return_value = SimpleNamespace(
        content="answer",
        latency_ms=100.0,
        total_duration_ms=None,
        load_duration_ms=None,
        prompt_eval_duration_ms=None,
        eval_duration_ms=None,
        generation_tokens_per_sec=None,
    )

    generator = LLMAnswerGenerator(
        llm_service=llm_service,
    )

    answer = generator.generate(
        system_prompt="system",
        user_prompt="user",
    )

    assert answer.content == "answer"

    llm_service.chat.assert_called_once_with(
        prompt="user",
        system_prompt="system",
        options=None,
    )
