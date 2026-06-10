import logging
import time
from logging import Logger

import requests

from config import Config
from dataclasses import dataclass

@dataclass(frozen=True)
class LLMResult:
    content: str
    latency_ms: float
    total_duration_ms: float | None
    load_duration_ms: float | None
    prompt_eval_count: int | None
    prompt_eval_duration_ms: float | None
    eval_count: int | None
    eval_duration_ms: float | None
    tokens_per_sec: float | None

def ns_to_ms(ns: int | None) -> float | None:
    if ns is None:
        return None
    return ns / 1_000_000


def tokens_per_sec(eval_count: int | None, eval_duration_ns: int | None) -> float | None:
    if not eval_count or not eval_duration_ns:
        return None

    sec = eval_duration_ns / 1_000_000_000
    if sec <= 0:
        return None

    return eval_count / sec

class LLMService:
    def __init__(
        self,
        config: Config,
        logger: Logger | None = None,
    ):
        self._config = config
        self._base_url = f"http://{config.llm_host}:{config.llm_port}"
        self._logger = logger or logging.getLogger(__name__)

    def chat(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        payload = {
            "model": self._config.llm_model,
            "messages": messages,
            "stream": False,
        }

        url = f"{self._base_url}/api/chat"
        start = time.perf_counter()

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._config.llm_timeout_sec,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            response.raise_for_status()

            data = response.json()
            eval_count = data.get("eval_count")
            eval_duration_ns = data.get("eval_duration")

            result = LLMResult(
                content=data["message"]["content"],
                latency_ms=latency_ms,
                total_duration_ms=ns_to_ms(data.get("total_duration")),
                load_duration_ms=ns_to_ms(data.get("load_duration")),
                prompt_eval_count=data.get("prompt_eval_count"),
                prompt_eval_duration_ms=ns_to_ms(data.get("prompt_eval_duration")),
                eval_count=eval_count,
                eval_duration_ms=ns_to_ms(eval_duration_ns),
                tokens_per_sec=tokens_per_sec(eval_count, eval_duration_ns),
            )

            self._logger.info(
                "Ollama chat completed | model=%s | latency_ms=%.2f | total_ms=%s | load_ms=%s | prompt_tokens=%s | output_tokens=%s | tps=%s",
                self._config.llm_model,
                result.latency_ms,
                result.total_duration_ms,
                result.load_duration_ms,
                result.prompt_eval_count,
                result.eval_count,
                result.tokens_per_sec,
            )
            
            return result
        except requests.Timeout:
            latency_ms = (time.perf_counter() - start) * 1000

            self._logger.error(
                "Ollama chat timeout | model=%s | latency_ms=%.2f | timeout_sec=%.2f",
                self._config.llm_model,
                latency_ms,
                self._config.llm_timeout_sec,
            )
            raise

        except requests.RequestException:
            latency_ms = (time.perf_counter() - start) * 1000

            self._logger.exception(
                "Ollama chat request failed | model=%s | latency_ms=%.2f",
                self._config.llm_model,
                latency_ms,
            )
            raise