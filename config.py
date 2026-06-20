import os
from dataclasses import dataclass
from typing import Literal, cast


LLMBackend = Literal["ollama", "vllm"]

RelevancePolicyKind = Literal[
    "disabled",
    "threshold",
]
RetrieverKind = Literal[
    "dense",
    "hybrid",
]

RerankerKind = Literal[
    "none",
    "cross_encoder",
]


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    retriever: RetrieverKind
    reranker: RerankerKind
    relevance_policy: RelevancePolicyKind

    dense_candidate_k: int
    sparse_candidate_k: int
    rerank_candidate_k: int
    rrf_k: int

    min_relevance_score: float

    reranker_model: str
    reranker_device: str
    reranker_batch_size: int
    reranker_max_length: int


@dataclass(frozen=True)
class LLMConfig:
    backend: LLMBackend
    base_url: str
    model: str
    timeout_sec: float
    api_key: str | None = None
    keep_alive: str | None = None


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    model: str
    device: str
    batch_size: int


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig


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


def load_retrieval_config() -> RetrievalConfig:
    retriever = os.getenv(
        "RAG_RETRIEVER",
        "dense",
    ).strip()

    reranker = os.getenv(
        "RAG_RERANKER",
        "none",
    ).strip()

    relevance_policy = os.getenv(
        "RAG_RELEVANCE_POLICY",
        "threshold",
    ).strip()

    if retriever not in {"dense", "hybrid"}:
        raise ValueError(
            f"Unsupported RAG_RETRIEVER: {retriever}"
        )

    if reranker not in {
        "none",
        "cross_encoder",
    }:
        raise ValueError(
            f"Unsupported RAG_RERANKER: {reranker}"
        )

    if relevance_policy not in {
        "disabled",
        "threshold",
    }:
        raise ValueError(
            "Unsupported RAG_RELEVANCE_POLICY: "
            f"{relevance_policy}"
        )

    return RetrievalConfig(
        retriever=cast(RetrieverKind, retriever),
        reranker=cast(RerankerKind, reranker),
        relevance_policy=cast(
            RelevancePolicyKind,
            relevance_policy,
        ),
        dense_candidate_k=_get_positive_int(
            "RAG_DENSE_CANDIDATE_K",
            20,
        ),
        sparse_candidate_k=_get_positive_int(
            "RAG_SPARSE_CANDIDATE_K",
            20,
        ),
        rerank_candidate_k=_get_positive_int(
            "RAG_RERANK_CANDIDATE_K",
            20,
        ),
        rrf_k=_get_positive_int(
            "RAG_RRF_K",
            60,
        ),
        min_relevance_score=_get_float(
            "RAG_MIN_RELEVANCE_SCORE",
            0.60,
        ),
        reranker_model=os.getenv(
            "RAG_RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ).strip(),
        reranker_device=os.getenv(
            "RAG_RERANKER_DEVICE",
            "cpu",
        ).strip(),
        reranker_batch_size=_get_positive_int(
            "RAG_RERANKER_BATCH_SIZE",
            8,
        ),
        reranker_max_length=_get_positive_int(
            "RAG_RERANKER_MAX_LENGTH",
            512,
        ),
    )


def load_env() -> Config:
    from dotenv import load_dotenv

    load_dotenv()

    embedding_batch_size = int(
        os.getenv("EMBEDDING_BATCH_SIZE", "32")
    )

    if embedding_batch_size <= 0:
        raise ValueError(
            "EMBEDDING_BATCH_SIZE must be greater than 0"
        )

    retrieval = load_retrieval_config()

    return Config(
        llm=load_llm_config(),
        embedding=EmbeddingConfig(
            backend=os.getenv(
                "EMBEDDING_BACKEND",
                "local",
            ).strip(),
            model=os.getenv(
                "EMBEDDING_MODEL",
                "BAAI/bge-m3",
            ).strip(),
            device=os.getenv(
                "EMBEDDING_DEVICE",
                "cuda",
            ).strip(),
            batch_size=embedding_batch_size,
        ),
        retrieval=retrieval,
    )


def _get_positive_int(
    name: str,
    default: int,
) -> int:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be an integer, got: {raw_value}"
        ) from error

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than 0, got: {value}"
        )

    return value


def _get_float(
    name: str,
    default: float,
) -> float:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a float, got: {raw_value}"
        ) from error
