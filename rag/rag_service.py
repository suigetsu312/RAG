from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from rag.document import RAGResult, RetrievedChunk
from rag.embeddings import EmbeddingService
from rag.generators import AnswerGenerator
from rag.metrics import RAGTimings, RetrievalTimings
from rag.prompts import PromptBuilder
from rag.vector_stores import VectorStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    retrieved_chunks: list[RetrievedChunk]
    timings: RetrievalTimings


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
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        prompt_builder: PromptBuilder,
        answer_generator: AnswerGenerator,
        min_relevance_score: float = 0.60,
    ) -> None:
        if not -1.0 <= min_relevance_score <= 1.0:
            raise ValueError(
                "min_relevance_score must be between -1.0 and 1.0"
            )

        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._prompt_builder = prompt_builder
        self._answer_generator = answer_generator
        self._min_relevance_score = min_relevance_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        total_start = perf_counter()

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query must not be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        embedding_start = perf_counter()

        query_result = self._embedding_service.embed(
            normalized_query
        )

        query_embedding_ms = self._elapsed_ms(
            embedding_start
        )

        retrieval_start = perf_counter()

        retrieved_chunks = self._vector_store.search(
            query_embedding=query_result.embedding,
            top_k=top_k,
        )

        retrieval_ms = self._elapsed_ms(
            retrieval_start
        )

        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            timings=RetrievalTimings(
                query_embedding_ms=query_embedding_ms,
                retrieval_ms=retrieval_ms,
                total_ms=self._elapsed_ms(total_start),
            ),
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

        retrieval_result = self.retrieve(
            query=normalized_question,
            top_k=top_k,
        )

        retrieved_chunks = retrieval_result.retrieved_chunks
        retrieval_timings = retrieval_result.timings

        if not retrieved_chunks:
            return RAGResult(
                answer=self.NO_CONTEXT_ANSWER,
                retrieved_chunks=[],
                timings=RAGTimings(
                    query_embedding_ms=(
                        retrieval_timings.query_embedding_ms
                    ),
                    retrieval_ms=(
                        retrieval_timings.retrieval_ms
                    ),
                    prompt_build_ms=0.0,
                    generation_ms=0.0,
                    total_ms=self._elapsed_ms(total_start),
                ),
            )

        if (
            retrieved_chunks[0].score
            < self._min_relevance_score
        ):
            return RAGResult(
                answer=self.INSUFFICIENT_CONTEXT_ANSWER,
                retrieved_chunks=retrieved_chunks,
                timings=RAGTimings(
                    query_embedding_ms=(
                        retrieval_timings.query_embedding_ms
                    ),
                    retrieval_ms=(
                        retrieval_timings.retrieval_ms
                    ),
                    prompt_build_ms=0.0,
                    generation_ms=0.0,
                    total_ms=self._elapsed_ms(total_start),
                ),
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
                "answer generator returned an empty answer"
            )

        return RAGResult(
            answer=normalized_answer,
            retrieved_chunks=retrieved_chunks,
            timings=RAGTimings(
                query_embedding_ms=(
                    retrieval_timings.query_embedding_ms
                ),
                retrieval_ms=(
                    retrieval_timings.retrieval_ms
                ),
                prompt_build_ms=prompt_build_ms,
                generation_ms=generation_ms,
                total_ms=self._elapsed_ms(total_start),
            ),
            generation_metadata=answer.metadata,
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (
            perf_counter() - started_at
        ) * 1000.0
