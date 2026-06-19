import logging

from config import GenerationOptions, load_env
from services.llm import create_llm_service


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

    logger = logging.getLogger("pipeline")
    config = load_env()

    logger.info(
        "Application started | llm_backend=%s | model=%s | "
        "base_url=%s",
        config.llm.backend,
        config.llm.model,
        config.llm.base_url,
    )

    try:
        with create_llm_service(
            config=config,
            logger=logging.getLogger("pipeline.llm"),
            default_options=RAG_ANSWER,
        ) as llm:
            result = llm.chat(
                prompt="介紹一下 Transformer 架構。",
                system_prompt=(
                    "你是技術回答器。"
                    "使用繁體中文直接回答，不展開無關背景。"
                ),
                options=LONG_EXPLANATION,
            )

        print("\n=== Response ===\n")
        print(result.content)

        print("\n=== Metrics ===\n")
        print(f"backend: {result.backend}")
        print(f"model: {result.model}")
        print(f"request_id: {result.request_id}")
        print(f"finish_reason: {result.finish_reason}")
        print(f"latency_ms: {result.latency_ms:.2f}")
        print(f"prompt_tokens: {result.prompt_tokens}")
        print(f"completion_tokens: {result.completion_tokens}")
        print(f"total_tokens: {result.total_tokens}")
        print(
            "end_to_end_tokens_per_sec: "
            f"{result.end_to_end_tokens_per_sec}"
        )
        print(
            "generation_tokens_per_sec: "
            f"{result.generation_tokens_per_sec}"
        )

    except Exception:
        logger.exception("Application failed")
        raise


if __name__ == "__main__":
    main()
