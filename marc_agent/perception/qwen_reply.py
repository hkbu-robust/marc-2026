"""Normalize Qwen / vLLM replies so parsers can find a finite candidate ID."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:|+-]+")

LENGTH_RETRY_TOKENS = 128


def strip_thinking(text: str) -> str:
    """Drop think-trace blocks and leftover think tags."""
    cleaned = _THINK_BLOCK.sub(" ", text or "")
    cleaned = _THINK_TAG.sub(" ", cleaned)
    return cleaned.strip()


def reply_tokens(text: str) -> tuple[str, ...]:
    """Left-to-right identifier-like tokens from a cleaned reply."""
    return tuple(_TOKEN_RE.findall(strip_thinking(text)))


def choice_text(payload: Mapping[str, Any]) -> tuple[str, Optional[str]]:
    """Return ``(text, finish_reason)`` from an OpenAI-style chat payload.

    Prefers ``message.content``; uses ``reasoning_content`` when content is empty.
    """
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("vLLM response did not contain choices[0].message") from exc
    finish = choice.get("finish_reason")
    finish_reason = str(finish) if finish is not None else None
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if isinstance(content, str) and content.strip():
        return content, finish_reason
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning, finish_reason
    if isinstance(content, str):
        return content, finish_reason
    raise ValueError("vLLM message content must be text")


class ReplyParseError(ValueError):
    """Parse failed; ``raw_output`` is the verifier text that was rejected."""

    def __init__(self, cause: BaseException, raw_output: Optional[str] = None):
        super().__init__(str(cause))
        self.raw_output = raw_output


def last_parsed(text: str, parse_token: Callable[[str], Any]) -> Any:
    """Parse each token left to right; the last success wins.

    ``parse_token`` must raise ``ValueError`` when a token is not a valid id.
    """
    tokens = reply_tokens(text)
    if not tokens:
        raise ValueError("output is empty")
    last = None
    last_error: Optional[BaseException] = None
    found = False
    for token in tokens:
        try:
            last = parse_token(token)
            found = True
        except ValueError as exc:
            last_error = exc
    if found:
        return last
    if last_error is not None:
        raise last_error
    raise ValueError("output did not contain a candidate ID")


def call_verifier(verifier, board, parse: Callable[[str], Any]):
    """Call ``verifier``; if a truncated reply fails parse, retry once longer."""
    raw = verifier(board)
    try:
        return parse(raw), raw
    except ValueError as exc:
        if getattr(verifier, "last_finish_reason", None) != "length":
            raise ReplyParseError(exc, raw) from exc
        retry = getattr(verifier, "retry_longer", None)
        if not callable(retry):
            raise ReplyParseError(exc, raw) from exc
        raw = retry(board)
        try:
            return parse(raw), raw
        except ValueError as retry_exc:
            raise ReplyParseError(retry_exc, raw) from retry_exc
