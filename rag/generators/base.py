from __future__ import annotations

from typing import Protocol

from rag.generators.result import GenerationResult


class AnswerGenerator(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationResult:
        ...
