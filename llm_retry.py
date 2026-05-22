"""Shared retry wrapper for Anthropic API calls with exponential backoff.

Used by assembler, verifier, classifier, synthetic_generator, and refixer.
Handles RateLimitError, OverloadedError, and transient network errors.
"""
from __future__ import annotations

import asyncio
import logging
import random

import anthropic

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_DELAY = 2.0
MAX_DELAY = 120.0

_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)

# Module-level semaphore for global Anthropic concurrency control.
# Prevents flooding the API when running 2800 tasks concurrently.
_api_semaphore: asyncio.Semaphore | None = None


def set_api_concurrency(max_concurrent: int) -> None:
    """Configure the global API concurrency limit.

    Call once at pipeline startup. Default is 10 concurrent API calls.
    """
    global _api_semaphore
    _api_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info("API concurrency set to %d", max_concurrent)


def _get_semaphore() -> asyncio.Semaphore:
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(10)
    return _api_semaphore


async def call_anthropic(
    client: anthropic.AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    messages: list[dict],
    system: str | None = None,
    stage: str = "unknown",
) -> anthropic.types.Message:
    """Make an Anthropic API call with exponential backoff and concurrency control.

    Args:
        client: AsyncAnthropic client instance
        model: Model name
        max_tokens: Max tokens for response
        messages: Conversation messages
        system: Optional system prompt
        stage: Label for logging/tracking (e.g., "verifier", "assembler")

    Returns:
        The API response Message

    Raises:
        The last exception after all retries are exhausted
    """
    sem = _get_semaphore()
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system

    last_exc = None
    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                response = await client.messages.create(**kwargs)
                return response
            except _RETRYABLE_ERRORS as e:
                last_exc = e
                delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                logger.warning(
                    "[%s] API error (attempt %d/%d): %s. Retrying in %.1fs...",
                    stage, attempt + 1, MAX_RETRIES, type(e).__name__, delay,
                )
                await asyncio.sleep(delay)

    logger.error("[%s] All %d retries exhausted", stage, MAX_RETRIES)
    raise last_exc
