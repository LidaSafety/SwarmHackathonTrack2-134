"""LLM backends. Claude goes through the Anthropic SDK; other providers through their
OpenAI-compatible endpoints (OpenAI, Gemini, OpenRouter, Ollama, vLLM)."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field

import anthropic
import openai

from .config import ModelSpec


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    refusal: bool = False
    error: str | None = None
    rl_retries: int = 0
    extra: dict = field(default_factory=dict)  # backend-specific per-call data (e.g. local activation probes)

    def cost(self, spec: ModelSpec) -> float:
        # cache reads billed at ~0.1x input
        billed_in = self.input_tokens + 0.1 * self.cache_read_tokens
        return (billed_in * spec.price_in + self.output_tokens * spec.price_out) / 1e6

    def budget_cost(self, spec: ModelSpec) -> float:
        """For the spending cap: the provider's billed cost when reported, never below the price-table estimate."""
        return max(self.cost(spec), self.extra.get("billed_usd", 0.0))


RATE_LIMITED = "rate_limited"


class FatalBackendError(RuntimeError):
    """Misconfiguration (bad key, no access, unknown model): abort instead of logging garbage runs."""


class Backend:
    async def complete(self, system: str, user: str) -> LLMResult:  # pragma: no cover
        raise NotImplementedError


class AnthropicBackend(Backend):
    _client: anthropic.AsyncAnthropic | None = None

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        if AnthropicBackend._client is None:
            AnthropicBackend._client = anthropic.AsyncAnthropic(max_retries=6)
        self.client = AnthropicBackend._client

    async def complete(self, system: str, user: str) -> LLMResult:
        kwargs = {}
        if self.spec.effort:
            kwargs["output_config"] = {"effort": self.spec.effort}
        try:
            resp = await self.client.messages.create(
                model=self.spec.model,
                max_tokens=self.spec.max_tokens,
                # system prompt is identical across rounds for an agent -> cacheable prefix
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                **kwargs,
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
            raise FatalBackendError(f"{self.spec.name}: {e.message}") from e
        except anthropic.RateLimitError:
            return LLMResult("", error=RATE_LIMITED)
        except anthropic.BadRequestError as e:
            return LLMResult("", error=f"bad_request: {e.message}")
        except anthropic.APIStatusError as e:
            return LLMResult("", error=f"api_status_{e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return LLMResult("", error=f"connection: {e}")
        u = resp.usage
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(
            text,
            input_tokens=u.input_tokens + (u.cache_creation_input_tokens or 0),
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            refusal=resp.stop_reason == "refusal",
        )


class OpenAICompatBackend(Backend):
    _clients: dict[tuple, openai.AsyncOpenAI] = {}

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        key = os.environ.get(spec.api_key_env or "OPENAI_API_KEY", "not-needed")
        ck = (spec.base_url, key)
        if ck not in self._clients:
            self._clients[ck] = openai.AsyncOpenAI(base_url=spec.base_url, api_key=key, max_retries=6, timeout=300)
        self.client = self._clients[ck]

    async def complete(self, system: str, user: str) -> LLMResult:
        res = await self._complete(system, user)
        if not res.error and not res.refusal and not res.text.strip():
            # reasoning models sometimes spend the whole budget thinking; one retry
            retry = await self._complete(system, user)
            retry.input_tokens += res.input_tokens
            retry.output_tokens += res.output_tokens
            if "billed_usd" in res.extra:
                retry.extra["billed_usd"] = retry.extra.get("billed_usd", 0.0) + res.extra["billed_usd"]
            retry.extra["empty_retry_finish"] = res.extra.get("finish_reason")
            res = retry
        return res

    async def _complete(self, system: str, user: str) -> LLMResult:
        try:
            resp = await self.client.chat.completions.create(
                model=self.spec.model,
                max_completion_tokens=self.spec.max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body=self.spec.extra_body or None,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise FatalBackendError(f"{self.spec.name}: {e.message}") from e
        except openai.RateLimitError:
            return LLMResult("", error=RATE_LIMITED)
        except openai.APIStatusError as e:
            return LLMResult("", error=f"api_status_{e.status_code}: {e.message}")
        except openai.APIConnectionError as e:
            return LLMResult("", error=f"connection: {e}")
        choice = resp.choices[0]
        u = resp.usage
        extra = {"finish_reason": choice.finish_reason}
        if u is not None:
            ux = u.model_extra or {}
            if ux.get("cost") is not None:  # OpenRouter: billed USD for this call
                extra["billed_usd"] = float(ux["cost"])
            details = u.completion_tokens_details
            if details is not None and getattr(details, "reasoning_tokens", None) is not None:
                extra["reasoning_tok"] = details.reasoning_tokens
        return LLMResult(
            choice.message.content or "",
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
            refusal=bool(getattr(choice.message, "refusal", None)) or choice.finish_reason == "content_filter",
            extra=extra,
        )


def make_backend(spec: ModelSpec) -> Backend | None:
    if spec.backend == "anthropic":
        return AnthropicBackend(spec)
    if spec.backend == "openai":
        return OpenAICompatBackend(spec)
    if spec.backend == "mock":
        return None  # handled by swarmlab.mock
    raise ValueError(f"unknown backend {spec.backend!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_response(text: str) -> tuple[dict | None, str | None]:
    """Extract the first JSON object from a model reply. Returns (obj, error)."""
    text = _THINK.sub("", text)
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for c in candidates:
        start = c.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(c)):
                ch = c[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(c[start : i + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(obj, dict):
                            return obj, None
                        break
            start = c.find("{", start + 1)
    return None, "could not parse a JSON object from your reply"


class RateGate:
    """Global concurrency cap per backend kind."""

    def __init__(self, limits: dict[str, int]):
        self.sems = {k: asyncio.Semaphore(v) for k, v in limits.items()}

    def __call__(self, backend: str) -> asyncio.Semaphore:
        return self.sems.setdefault(backend, asyncio.Semaphore(8))
