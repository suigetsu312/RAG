from logging import Logger
from typing import Any

import requests

from config import GenerationOptions, LLMConfig
from .base import LLMResult, LLMService


class VLLMLLMService(LLMService):
    def __init__(
        self,
        config: LLMConfig,
        logger: Logger | None = None,
        default_options: GenerationOptions | None = None,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(
            config=config,
            logger=logger,
            default_options=default_options,
            session=session,
        )

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        options: GenerationOptions | None = None,
    ) -> LLMResult:
        generation_options = options or self._default_options

        self.log_request_start(
            backend="vllm",
            prompt=prompt,
            system_prompt=system_prompt,
            options=generation_options,
        )

        payload = {
            "model": self._config.model,
            "messages": self.build_messages(
                prompt,
                system_prompt,
            ),
            "stream": False,
            "max_tokens": generation_options.num_predict,
            "temperature": generation_options.temperature,
            "top_p": generation_options.top_p,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

        headers = {
            "Content-Type": "application/json",
        }

        if self._config.api_key:
            headers["Authorization"] = (
                f"Bearer {self._config.api_key}"
            )

        data, latency_ms, response_headers = self.post_json(
            backend="vllm",
            url=(
                f"{self._config.base_url.rstrip('/')}"
                "/chat/completions"
            ),
            payload=payload,
            headers=headers,
        )

        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            raise ValueError(
                "vLLM response does not contain choices"
            )

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            raise ValueError(
                "vLLM response contains an invalid choice"
            )

        message = first_choice.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "vLLM response does not contain a valid message"
            )

        content = message.get("content")

        if not isinstance(content, str):
            reasoning_content = message.get("reasoning_content")

            if isinstance(reasoning_content, str):
                content = reasoning_content
            else:
                raise ValueError(
                    "vLLM response does not contain text content"
                )

        usage = data.get("usage")

        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = self.optional_int(
            usage.get("prompt_tokens")
        )
        completion_tokens = self.optional_int(
            usage.get("completion_tokens")
        )
        total_tokens = self.optional_int(
            usage.get("total_tokens")
        )

        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens

        request_id = self.optional_str(data.get("id"))

        if request_id is None:
            request_id = response_headers.get("x-request-id")

        finish_reason = self.optional_str(
            first_choice.get("finish_reason")
        )

        end_to_end_tps = self.calculate_tokens_per_sec(
            completion_tokens,
            latency_ms,
        )

        result = LLMResult(
            content=content,
            backend="vllm",
            model=self._config.model,
            latency_ms=latency_ms,
            request_id=request_id,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            end_to_end_tokens_per_sec=end_to_end_tps,

            # 標準非串流 OpenAI-compatible response
            # 沒有 Ollama 對應的 server-side duration。
            generation_tokens_per_sec=None,
        )

        self._logger.info(
            "vLLM chat completed | model=%s | request_id=%s | "
            "latency_ms=%.2f | prompt_tokens=%s | "
            "completion_tokens=%s | total_tokens=%s | "
            "e2e_tps=%s | finish_reason=%s",
            result.model,
            result.request_id,
            result.latency_ms,
            result.prompt_tokens,
            result.completion_tokens,
            result.total_tokens,
            result.end_to_end_tokens_per_sec,
            result.finish_reason,
        )

        return result