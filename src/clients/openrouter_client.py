"""Unified LLM client via OpenRouter — single API key for all models."""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from src.utils.logging import get_logger

logger = get_logger("openrouter")

# Approximate cost per 1M tokens for common models (input/output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "google/gemini-2.5-pro": (1.25, 10.0),
    "openai/gpt-4.1": (2.0, 8.0),
    "deepseek/deepseek-r1": (0.55, 2.19),
    "x-ai/grok-3": (3.0, 15.0),
}


class OpenRouterClient:
    """Async client for OpenRouter API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/pred-mark-clat",
                },
                timeout=120.0,
            )
        return self._client

    async def complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> tuple[str, float, int]:
        """
        Send a chat completion request.

        Returns: (response_text, estimated_cost_usd, total_tokens)
        """
        async with self._semaphore:
            client = await self._get_client()

            payload: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if response_format:
                payload["response_format"] = response_format

            start = time.monotonic()
            try:
                resp = await client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("rate_limited", model=model)
                    await asyncio.sleep(5)
                    resp = await client.post("/chat/completions", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    raise
            duration_ms = int((time.monotonic() - start) * 1000)

            # Extract response
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"No choices in response from {model}")

            text = choices[0].get("message", {}).get("content", "")

            # Extract usage
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Estimate cost
            cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

            logger.debug(
                "llm_complete",
                model=model,
                tokens=total_tokens,
                cost=f"${cost:.4f}",
                duration_ms=duration_ms,
            )

            return text, cost, total_tokens

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on known model pricing."""
        pricing = MODEL_PRICING.get(model, (5.0, 15.0))  # default conservative
        input_cost = (prompt_tokens / 1_000_000) * pricing[0]
        output_cost = (completion_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
