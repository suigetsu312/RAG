from logging import Logger

import requests

from config import GenerationOptions, LLMConfig
from .base import LLMResult, LLMService


class OllamaLLMService(LLMService):
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
            backend="ollama",
            prompt=prompt,
            system_prompt=system_prompt,
            options=generation_options,
        )

        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": self.build_messages(
                prompt,
                system_prompt,
            ),
            "stream": False,
            "options": {
                "num_ctx": generation_options.num_ctx,
                "num_predict": generation_options.num_predict,
                "temperature": generation_options.temperature,
                "top_p": generation_options.top_p,
            },
        }

        if self._config.keep_alive:
            payload["keep_alive"] = self._config.keep_alive

        data, latency_ms, _ = self.post_json(
            backend="ollama",
            url=f"{self._config.base_url.rstrip('/')}/api/chat",
            payload=payload,
        )

        message = data.get("message")

        if not isinstance(message, dict):
            raise ValueError(
                "Ollama response does not contain a valid message"
            )

        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError(
                "Ollama response does not contain text content"
            )

        prompt_tokens = self.optional_int(
            data.get("prompt_eval_count")
        )
        completion_tokens = self.optional_int(
            data.get("eval_count")
        )

        total_tokens = None

        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        total_duration_ms = self.ns_to_ms(
            self.optional_int(data.get("total_duration"))
        )
        load_duration_ms = self.ns_to_ms(
            self.optional_int(data.get("load_duration"))
        )
        prompt_eval_duration_ms = self.ns_to_ms(
            self.optional_int(data.get("prompt_eval_duration"))
        )
        eval_duration_ms = self.ns_to_ms(
            self.optional_int(data.get("eval_duration"))
        )

        generation_tps = self.calculate_tokens_per_sec(
            completion_tokens,
            eval_duration_ms,
        )
        end_to_end_tps = self.calculate_tokens_per_sec(
            completion_tokens,
            latency_ms,
        )

        result = LLMResult(
            content=content,
            backend="ollama",
            model=self._config.model,
            latency_ms=latency_ms,
            finish_reason=self.optional_str(
                data.get("done_reason")
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            end_to_end_tokens_per_sec=end_to_end_tps,
            total_duration_ms=total_duration_ms,
            load_duration_ms=load_duration_ms,
            prompt_eval_duration_ms=prompt_eval_duration_ms,
            eval_duration_ms=eval_duration_ms,
            generation_tokens_per_sec=generation_tps,
        )

        self._logger.info(
            "Ollama chat completed | model=%s | latency_ms=%.2f | "
            "total_ms=%s | load_ms=%s | prompt_eval_ms=%s | "
            "eval_ms=%s | prompt_tokens=%s | completion_tokens=%s | "
            "total_tokens=%s | generation_tps=%s | e2e_tps=%s | "
            "finish_reason=%s",
            result.model,
            result.latency_ms,
            result.total_duration_ms,
            result.load_duration_ms,
            result.prompt_eval_duration_ms,
            result.eval_duration_ms,
            result.prompt_tokens,
            result.completion_tokens,
            result.total_tokens,
            result.generation_tokens_per_sec,
            result.end_to_end_tokens_per_sec,
            result.finish_reason,
        )

        return result