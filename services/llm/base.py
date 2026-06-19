from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import time
from logging import Logger
from typing import Any, Mapping

import requests

from config import GenerationOptions, LLMConfig


@dataclass(frozen=True)
class LLMResult:
    content: str
    backend: str
    model: str
    latency_ms: float

    request_id: str | None = None
    finish_reason: str | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # completion_tokens / 完整 HTTP latency。
    # 包含 prefill、decode、網路與序列化成本。
    end_to_end_tokens_per_sec: float | None = None

    # Ollama 能提供的 server-side timing。
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    generation_tokens_per_sec: float | None = None


class LLMService(ABC):
    def __init__(
        self,
        config: LLMConfig,
        logger: Logger | None = None,
        default_options: GenerationOptions | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(
            self.__class__.__module__
        )
        self._default_options = default_options or GenerationOptions()
        self._session = session or requests.Session()

    @abstractmethod
    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        options: GenerationOptions | None = None,
    ) -> LLMResult:
        raise NotImplementedError

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "LLMService":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    @staticmethod
    def build_messages(
        prompt: str,
        system_prompt: str | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    def log_request_start(
        self,
        *,
        backend: str,
        prompt: str,
        system_prompt: str | None,
        options: GenerationOptions,
    ) -> None:
        self._logger.info(
            "%s chat started | model=%s | prompt_chars=%d | "
            "system_chars=%d | num_ctx=%d | max_tokens=%d | "
            "temperature=%.3f | top_p=%.3f",
            backend,
            self._config.model,
            len(prompt),
            len(system_prompt) if system_prompt else 0,
            options.num_ctx,
            options.num_predict,
            options.temperature,
            options.top_p,
        )

    def post_json(
        self,
        *,
        backend: str,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], float, Mapping[str, str]]:
        start = time.perf_counter()

        try:
            response = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_sec,
            )

            latency_ms = (time.perf_counter() - start) * 1000.0

            try:
                response.raise_for_status()
            except requests.HTTPError:
                response_body = response.text[:1000]

                self._logger.error(
                    "%s HTTP error | model=%s | status=%d | "
                    "latency_ms=%.2f | response=%r",
                    backend,
                    self._config.model,
                    response.status_code,
                    latency_ms,
                    response_body,
                )
                raise

            try:
                data = response.json()
            except ValueError:
                self._logger.error(
                    "%s invalid JSON response | model=%s | "
                    "status=%d | latency_ms=%.2f | response=%r",
                    backend,
                    self._config.model,
                    response.status_code,
                    latency_ms,
                    response.text[:1000],
                )
                raise

            if not isinstance(data, dict):
                raise ValueError(
                    f"{backend} returned a non-object JSON response"
                )

            return data, latency_ms, response.headers

        except requests.Timeout:
            latency_ms = (time.perf_counter() - start) * 1000.0

            self._logger.error(
                "%s chat timeout | model=%s | latency_ms=%.2f | "
                "timeout_sec=%.2f",
                backend,
                self._config.model,
                latency_ms,
                self._config.timeout_sec,
            )
            raise

        except requests.RequestException:
            latency_ms = (time.perf_counter() - start) * 1000.0

            self._logger.exception(
                "%s request failed | model=%s | latency_ms=%.2f",
                backend,
                self._config.model,
                latency_ms,
            )
            raise

    @staticmethod
    def optional_int(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return value

        return None

    @staticmethod
    def optional_str(value: object) -> str | None:
        if isinstance(value, str):
            return value

        return None

    @staticmethod
    def calculate_tokens_per_sec(
        token_count: int | None,
        duration_ms: float | None,
    ) -> float | None:
        if token_count is None or duration_ms is None:
            return None

        if token_count <= 0 or duration_ms <= 0:
            return None

        return token_count / (duration_ms / 1000.0)

    @staticmethod
    def ns_to_ms(value_ns: int | None) -> float | None:
        if value_ns is None:
            return None

        return value_ns / 1_000_000.0