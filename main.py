import logging
from pathlib import Path

from config import GenerationOptions, load_env
from rag.embeddings import LocalEmbeddingService
from rag.generators import LLMAnswerGenerator
from rag.indexing_service import IndexingResult, IndexingService
from rag.prompts import PromptBuilder
from rag.rag_service import RAGService
from rag.runtime import create_chunker, create_document_loader
from rag.vector_stores import FAISSVectorStore
from services.llm import LLMService, create_llm_service

SHORT_ANSWER = GenerationOptions(
    num_ctx=2048,
    num_predict=128,
    temperature=0.2,
    top_p=1.0,
)

RAG_ANSWER = GenerationOptions(
    num_ctx=4096,
    num_predict=256,
    temperature=0.2,
    top_p=1.0,
)

LONG_EXPLANATION = GenerationOptions(
    # vLLM 的 context 上限由 server --max-model-len 控制。
    # 目前 server 設為 4096。
    num_ctx=4096,
    num_predict=1024,
    temperature=0.2,
    top_p=1.0,
)


def build_rag_service(
    *,
    llm_service: LLMService,
    documents_directory: str | Path,
    embedding_model: str,
    embedding_device: str,
    embedding_batch_size: int,
) -> tuple[IndexingResult, RAGService]:
    embedding_service = LocalEmbeddingService(
        model_name=embedding_model,
        device=embedding_device,
        batch_size=embedding_batch_size,
    )

    indexing_service = IndexingService(
        loader=create_document_loader(),
        chunker=create_chunker(),
        embedding_service=embedding_service,
        vector_store_factory=FAISSVectorStore,
    )

    indexing_result = indexing_service.build(
        documents_directory
    )

    generation_options = GenerationOptions(
        num_ctx=4096,
        num_predict=1024,
        temperature=0.2,
    )

    answer_generator = LLMAnswerGenerator(
        llm_service=llm_service,
        options=generation_options,
    )

    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=indexing_result.vector_store,
        prompt_builder=PromptBuilder(),
        answer_generator=answer_generator,
    )

    return indexing_result, rag_service


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def main() -> None:
    setup_logging()

    config = load_env()

    # 保留你目前原本的 LLM 建立程式。
    llm = create_llm_service(config)

    indexing_result, rag_service = build_rag_service(
        llm_service=llm,
        documents_directory="data/documents",
        embedding_model=config.embedding.model,
        embedding_device=config.embedding.device,
        embedding_batch_size=config.embedding.batch_size,
    )

    print("=== Indexing ===")
    print(
        f"documents: {indexing_result.document_count}"
    )
    print(
        f"chunks: {indexing_result.chunk_count}"
    )
    print(
        "embedding dimension: "
        f"{indexing_result.embedding_dimension}"
    )
    indexing_timings = indexing_result.timings
    print("indexing timings:")
    print(
        "  document load: "
        f"{indexing_timings.document_load_ms:.2f} ms"
    )
    print(
        f"  chunking: {indexing_timings.chunking_ms:.2f} ms"
    )
    print(
        f"  embedding: {indexing_timings.embedding_ms:.2f} ms"
    )
    print(
        "  vector store add: "
        f"{indexing_timings.vector_store_add_ms:.2f} ms"
    )
    print(f"  total: {indexing_timings.total_ms:.2f} ms")

    question = "Transformer 的主要機制是什麼？"

    result = rag_service.ask(
        question=question,
        top_k=3,
    )

    print()
    print("=== Question ===")
    print(question)

    print()
    print("=== Answer ===")
    print(result.answer)

    print()
    print("=== RAG Timings ===")
    rag_timings = result.timings
    print(
        "query embedding: "
        f"{rag_timings.query_embedding_ms:.2f} ms"
    )
    print(f"retrieval: {rag_timings.retrieval_ms:.2f} ms")
    print(
        f"prompt build: {rag_timings.prompt_build_ms:.2f} ms"
    )
    print(f"generation: {rag_timings.generation_ms:.2f} ms")
    print(f"total: {rag_timings.total_ms:.2f} ms")

    generation_metadata = result.generation_metadata
    if generation_metadata:
        print()
        print("=== LLM Generation Metrics ===")

        for label, key, unit in (
            ("latency", "latency_ms", " ms"),
            ("total duration", "total_duration_ms", " ms"),
            ("load duration", "load_duration_ms", " ms"),
            (
                "prompt eval duration",
                "prompt_eval_duration_ms",
                " ms",
            ),
            ("eval duration", "eval_duration_ms", " ms"),
            (
                "generation tokens per second",
                "generation_tokens_per_sec",
                "",
            ),
        ):
            value = generation_metadata.get(key)
            if value is not None:
                print(f"{label}: {value:.2f}{unit}")

    print()
    print("=== Retrieved Sources ===")

    for rank, retrieved in enumerate(
        result.retrieved_chunks,
        start=1,
    ):
        chunk = retrieved.chunk

        print(
            f"{rank}. "
            f"score={retrieved.score:.4f} "
            f"source={chunk.source} "
            f"chunk_id={chunk.id}"
        )


if __name__ == "__main__":
    main()
