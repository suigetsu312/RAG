from __future__ import annotations

from time import perf_counter

from rag.document import RAGResult
from rag.embeddings.base import EmbeddingService
from rag.generators.base import AnswerGenerator
from rag.metrics import RAGTimings
from rag.prompts import PromptBuilder
from rag.vector_stores.base import VectorStore


class RAGService:
    NO_CONTEXT_ANSWER = (
        "目前索引中找不到足以回答此問題的相關內容。"
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        prompt_builder: PromptBuilder,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._prompt_builder = prompt_builder
        self._answer_generator = answer_generator

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

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        embedding_start = perf_counter()
        query_result = self._embedding_service.embed(
            normalized_question
        )

        query_embedding_ms = (
            perf_counter() - embedding_start
        ) * 1000.0

        retrieval_start = perf_counter()
        retrieved_chunks = self._vector_store.search(
            query_embedding=query_result.embedding,
            top_k=top_k,
        )
        retrieval_ms = (
            perf_counter() - retrieval_start
        ) * 1000.0

        if not retrieved_chunks:
            total_ms = (perf_counter() - total_start) * 1000.0

            return RAGResult(
                answer=self.NO_CONTEXT_ANSWER,
                retrieved_chunks=[],
                timings=RAGTimings(
                    query_embedding_ms=query_embedding_ms,
                    retrieval_ms=retrieval_ms,
                    prompt_build_ms=0.0,
                    generation_ms=0.0,
                    total_ms=total_ms,
                ),
            )

        prompt_start = perf_counter()
        prompt = self._prompt_builder.build(
            question=normalized_question,
            retrieved_chunks=retrieved_chunks,
        )
        prompt_build_ms = (
            perf_counter() - prompt_start
        ) * 1000.0

        generation_start = perf_counter()
        answer = self._answer_generator.generate(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )
        generation_ms = (
            perf_counter() - generation_start
        ) * 1000.0

        normalized_answer = answer.content.strip()

        if not normalized_answer:
            raise RuntimeError(
                "answer generator returned an empty answer"
            )

        total_ms = (perf_counter() - total_start) * 1000.0

        return RAGResult(
            answer=normalized_answer,
            retrieved_chunks=retrieved_chunks,
            timings=RAGTimings(
                query_embedding_ms=query_embedding_ms,
                retrieval_ms=retrieval_ms,
                prompt_build_ms=prompt_build_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
            generation_metadata=answer.metadata
        )
