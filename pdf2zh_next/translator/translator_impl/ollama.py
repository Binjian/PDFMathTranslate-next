import logging

import ollama
from babeldoc.utils.atomic_integer import AtomicInteger
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


def _parse_timeout(value: str | None) -> float | None:
    try:
        t = float(value or "300")
    except (TypeError, ValueError):
        return 300.0
    return t if t > 0 else None


def _parse_think(value: str | None):
    v = str(value or "").strip().lower()
    if v in ("", "auto", "none", "default"):
        return None
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("low", "medium", "high"):
        return v
    return False


class OllamaTranslator(BaseTranslator):
    # https://github.com/ollama/ollama
    name = "ollama"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.options = {
            "temperature": 0,
            "num_predict": settings.translate_engine_settings.num_predict,
        }  # 随机采样可能会打断公式标记

        engine = settings.translate_engine_settings
        timeout = _parse_timeout(getattr(engine, "ollama_timeout", "300"))
        self.think = _parse_think(getattr(engine, "ollama_think", "false"))

        self.client = ollama.Client(
            host=engine.ollama_host,
            timeout=timeout,
            trust_env=False,
        )
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("num_predict", self.options["num_predict"])
        self.model = engine.ollama_model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.add_cache_impact_parameters("think", self.think)
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @staticmethod
    def _remove_cot_content(content: str) -> str:
        import re
        return re.sub(r"^<think>.+?</think>", "", content, count=1, flags=re.DOTALL).strip()

    @retry(
        retry=retry_if_exception_type(ollama.ResponseError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        if (max_token := len(text) * 5) > self.options["num_predict"]:
            self.options["num_predict"] = max_token
        chat_kwargs = {
            "model": self.model,
            "options": self.options,
            "messages": self.prompt(text),
        }
        if self.think is not None:
            chat_kwargs["think"] = self.think
        try:
            response = self.client.chat(**chat_kwargs)
        except ollama.ResponseError as exc:
            if "think" not in chat_kwargs or "think" not in str(exc).lower():
                raise
            logger.warning("Ollama server rejected think option; retrying without it.")
            chat_kwargs.pop("think")
            self.think = None
            response = self.client.chat(**chat_kwargs)
        self.token_count.inc(response.prompt_eval_count + response.eval_count)
        self.prompt_token_count.inc(response.prompt_eval_count)
        self.completion_token_count.inc(response.eval_count)
        message = response.message.content.strip()
        message = self._remove_cot_content(message)
        return message

    @retry(
        retry=retry_if_exception_type(ollama.ResponseError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        if (max_token := len(text) * 5) > self.options["num_predict"]:
            self.options["num_predict"] = max_token

        chat_kwargs = {
            "model": self.model,
            "options": self.options,
            "messages": [{"role": "user", "content": text}],
        }
        if self.think is not None:
            chat_kwargs["think"] = self.think
        try:
            response = self.client.chat(**chat_kwargs)
        except ollama.ResponseError as exc:
            if "think" not in chat_kwargs or "think" not in str(exc).lower():
                raise
            logger.warning("Ollama server rejected think option; retrying without it.")
            chat_kwargs.pop("think")
            self.think = None
            response = self.client.chat(**chat_kwargs)
        self.token_count.inc(response.prompt_eval_count + response.eval_count)
        self.prompt_token_count.inc(response.prompt_eval_count)
        self.completion_token_count.inc(response.eval_count)
        message = response.message.content.strip()
        message = self._remove_cot_content(message)
        return message
