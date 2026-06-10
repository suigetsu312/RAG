import logging

from config import load_env
from llm_service import LLMService


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()

    config = load_env()

    llm = LLMService(
        config=config,
        logger=logging.getLogger("pipeline.llm"),
    )

    result = llm.chat(
        prompt="what is the focal loss formula which is used for object detection?",
        system_prompt="你是computer vision expert，回答要精簡且正確。",
    )

    print(result)


if __name__ == "__main__":
    main()