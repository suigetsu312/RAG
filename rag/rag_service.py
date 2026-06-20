from __future__ import annotations

from time import perf_counter

from rag.document import RAGResult, RetrievedChunk
from rag.generators import AnswerGenerator
from rag.metrics import RAGTimings, RetrievalTimings
from rag.prompts import PromptBuilder
from rag.retrieval_pipeline import RetrievalResult
from rag.retrieval_runtime import (
    RetrievalComponentManager,
)


class RAGService:
    NO_CONTEXT_ANSWER = (
        "目前索引中找不到足以回答此問題的相關內容。"
    )

    INSUFFICIENT_CONTEXT_ANSWER = (
        "找不到與問題足夠相關的文件內容，"
        "因此無法根據目前的知識庫回答。"
    )

    def __init__(
        self,
        retrieval_components: RetrievalComponentManager,
        prompt_builder: PromptBuilder,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._retrieval_components = (
            retrieval_components
        )
        self._prompt_builder = prompt_builder
        self._answer_generator = answer_generator

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        components = (
            self._retrieval_components.snapshot()
        )

        return components.pipeline.retrieve(
            query=query,
            top_k=top_k,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResult:
        total_start = perf_counter()
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError(
                "question must not be empty"
            )

        components = (
            self._retrieval_components.snapshot()
        )

        retrieval_result = (
            components.pipeline.retrieve(
                query=normalized_question,
                top_k=top_k,
            )
        )

        retrieved_chunks = (
            retrieval_result.retrieved_chunks
        )
        retrieval_timings = (
            retrieval_result.timings
        )

        if not retrieved_chunks:
            return self._create_rejected_result(
                answer=self.NO_CONTEXT_ANSWER,
                retrieved_chunks=[],
                retrieval_timings=(
                    retrieval_timings
                ),
                total_start=total_start,
            )

        if not components.relevance_policy.is_relevant(
            retrieved_chunks
        ):
            return self._create_rejected_result(
                answer=(
                    self.INSUFFICIENT_CONTEXT_ANSWER
                ),
                retrieved_chunks=retrieved_chunks,
                retrieval_timings=(
                    retrieval_timings
                ),
                total_start=total_start,
            )

        prompt_start = perf_counter()

        prompt = self._prompt_builder.build(
            question=normalized_question,
            retrieved_chunks=retrieved_chunks,
        )

        prompt_build_ms = self._elapsed_ms(
            prompt_start
        )

        generation_start = perf_counter()

        answer = self._answer_generator.generate(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )

        generation_ms = self._elapsed_ms(
            generation_start
        )

        normalized_answer = answer.content.strip()

        if not normalized_answer:
            raise RuntimeError(
                "answer generator returned "
                "an empty answer"
            )

        return RAGResult(
            answer=normalized_answer,
            retrieved_chunks=retrieved_chunks,
            timings=RAGTimings(
                query_embedding_ms=(
                    retrieval_timings
                    .query_embedding_ms
                ),
                retrieval_ms=(
                    retrieval_timings.retrieval_ms
                    + retrieval_timings.rerank_ms
                ),
                prompt_build_ms=prompt_build_ms,
                generation_ms=generation_ms,
                total_ms=self._elapsed_ms(
                    total_start
                ),
            ),
            generation_metadata=answer.metadata,
        )

    def _create_rejected_result(
        self,
        *,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        retrieval_timings: RetrievalTimings,
        total_start: float,
    ) -> RAGResult:
        return RAGResult(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            timings=RAGTimings(
                query_embedding_ms=(
                    retrieval_timings
                    .query_embedding_ms
                ),
                retrieval_ms=(
                    retrieval_timings.retrieval_ms
                    + retrieval_timings.rerank_ms
                ),
                prompt_build_ms=0.0,
                generation_ms=0.0,
                total_ms=self._elapsed_ms(
                    total_start
                ),
            ),
        )

    @staticmethod
    def _elapsed_ms(
        started_at: float,
    ) -> float:
        return (
            perf_counter() - started_at
        ) * 1000.0
