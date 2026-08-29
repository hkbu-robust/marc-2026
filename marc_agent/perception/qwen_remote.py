"""Remote vLLM adapter for Stage 1 Qwen SAR verification."""

from __future__ import annotations

import base64
import io
import json
import logging
import socket
import time
from dataclasses import dataclass
from typing import Callable, Mapping, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marc_agent.perception.qwen_reply import LENGTH_RETRY_TOKENS, choice_text
from marc_agent.perception.qwen_runtime import QwenRemoteConfig

if TYPE_CHECKING:
    from marc_agent.perception.sar_selection import CandidateBoard

log = logging.getLogger("marc_agent.qwen_remote")


class QwenRemoteError(RuntimeError):
    """Base class for failures talking to the configured vLLM server."""


class QwenRemoteTimeout(QwenRemoteError):
    pass


class QwenRemoteAuthenticationError(QwenRemoteError):
    pass


class QwenRemoteServerError(QwenRemoteError):
    pass


class QwenRemoteConnectionError(QwenRemoteError):
    pass


class QwenRemoteResponseError(QwenRemoteError):
    pass


HttpPost = Callable[[str, bytes, Mapping[str, str], float], bytes]
HttpGet = Callable[[str, Mapping[str, str], float], bytes]


@dataclass(frozen=True)
class QwenRemoteMetrics:
    request_bytes: int
    image_count: int
    latency_ms: float


def _post(url: str, body: bytes, headers: Mapping[str, str], timeout_s: float) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise QwenRemoteAuthenticationError(
                f"vLLM rejected the configured API key (HTTP {exc.code})"
            ) from exc
        raise QwenRemoteServerError(f"vLLM returned HTTP {exc.code}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise QwenRemoteTimeout(
            f"vLLM did not respond within {timeout_s:.1f}s"
        ) from exc
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise QwenRemoteTimeout(
                f"vLLM did not respond within {timeout_s:.1f}s"
            ) from exc
        raise QwenRemoteConnectionError(f"Could not reach vLLM: {exc.reason}") from exc
    except OSError as exc:
        raise QwenRemoteConnectionError(f"Could not reach vLLM: {exc}") from exc


def _get(url: str, headers: Mapping[str, str], timeout_s: float) -> bytes:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise QwenRemoteAuthenticationError(
                f"vLLM rejected the configured API key (HTTP {exc.code})"
            ) from exc
        raise QwenRemoteServerError(f"vLLM returned HTTP {exc.code}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise QwenRemoteTimeout(
            f"vLLM did not respond within {timeout_s:.1f}s"
        ) from exc
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise QwenRemoteTimeout(
                f"vLLM did not respond within {timeout_s:.1f}s"
            ) from exc
        raise QwenRemoteConnectionError(f"Could not reach vLLM: {exc.reason}") from exc
    except OSError as exc:
        raise QwenRemoteConnectionError(f"Could not reach vLLM: {exc}") from exc


def check_remote_health(
    config: QwenRemoteConfig,
    *,
    get: HttpGet = _get,
) -> tuple[str, ...]:
    """Verify vLLM authentication and that the configured served model exists."""
    raw = get(
        config.models_url,
        {"Authorization": f"Bearer {config.api_key}"},
        config.timeout_s,
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
        model_ids = tuple(str(item["id"]) for item in payload["data"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise QwenRemoteResponseError(
            "vLLM /models response did not contain data[].id"
        ) from exc
    if config.model_name not in model_ids:
        raise QwenRemoteResponseError(
            f"configured model {config.model_name!r} is not served; "
            f"available={','.join(model_ids) or 'none'}"
        )
    return model_ids


def image_data_url(image: object) -> str:
    """Encode one PIL-compatible image as a lossless PNG data URL."""
    try:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG", compress_level=1)
    except (AttributeError, OSError, ValueError) as exc:
        raise QwenRemoteResponseError("Candidate board contains an invalid image") from exc
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_remote_payload(
    board: "CandidateBoard",
    config: QwenRemoteConfig,
    *,
    max_tokens: int | None = None,
) -> bytes:
    content = [
        {
            "type": "image_url",
            "image_url": {"url": image_data_url(image)},
        }
        for image in board.images
    ]
    content.append({"type": "text", "text": board.prompt})
    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": int(max_tokens if max_tokens is not None else config.max_new_tokens),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class RemoteQwenSarVerifier:
    """Send one candidate board to an OpenAI-compatible vLLM endpoint."""

    method = "qwen_remote"

    def __init__(
        self,
        config: QwenRemoteConfig | None = None,
        *,
        post: HttpPost = _post,
    ):
        self.config = config or QwenRemoteConfig.load()
        self._post = post
        self.last_metrics: QwenRemoteMetrics | None = None
        self.budget_timeout_s: float | None = None
        self.last_finish_reason: str | None = None

    def set_budget_timeout(self, timeout_s: float) -> None:
        self.budget_timeout_s = float(timeout_s)

    def retry_longer(self, board: "CandidateBoard") -> str:
        """Re-post with more tokens after a truncated reply failed to parse."""
        return self._complete(
            board,
            max_tokens=max(LENGTH_RETRY_TOKENS, int(self.config.max_new_tokens)),
        )

    def __call__(self, board: "CandidateBoard") -> str:
        return self._complete(board)

    def _timeout_s(self) -> tuple[float, int]:
        timeout_s = min(
            float(self.config.timeout_s),
            float(self.config.wall_timeout_s),
        )
        budget = self.budget_timeout_s
        if budget is not None:
            timeout_s = min(timeout_s, float(budget))
        retry_ok = budget is None or float(budget) >= 2.0 * timeout_s
        max_attempts = 2 if retry_ok else 1
        return timeout_s, max_attempts

    def _complete(
        self,
        board: "CandidateBoard",
        *,
        max_tokens: int | None = None,
    ) -> str:
        body = build_remote_payload(board, self.config, max_tokens=max_tokens)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        timeout_s, max_attempts = self._timeout_s()
        started = time.perf_counter()
        raw = b""
        latency_ms = 0.0
        try:
            for attempt in range(1, max_attempts + 1):
                log.info(
                    "[qwen-remote] dispatch endpoint=%s model=%s images=%d "
                    "candidates=%d request=%dB deadline=%.1fs budget=%s "
                    "max_tokens=%d attempt=%d/%d",
                    self.config.chat_completions_url,
                    self.config.model_name,
                    len(board.images),
                    len(board.candidate_ids),
                    len(body),
                    timeout_s,
                    (
                        "none"
                        if self.budget_timeout_s is None
                        else f"{float(self.budget_timeout_s):.1f}s"
                    ),
                    int(max_tokens if max_tokens is not None else self.config.max_new_tokens),
                    attempt,
                    max_attempts,
                )
                try:
                    raw = self._post(
                        self.config.chat_completions_url,
                        body,
                        headers,
                        timeout_s,
                    )
                    break
                except (QwenRemoteTimeout, QwenRemoteConnectionError) as exc:
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    kind = (
                        "timeout"
                        if isinstance(exc, QwenRemoteTimeout)
                        else "failed"
                    )
                    log.warning(
                        "[qwen-remote] %s endpoint=%s model=%s "
                        "latency=%.0fms error=%s detail=%s attempt=%d/%d",
                        kind,
                        self.config.chat_completions_url,
                        self.config.model_name,
                        latency_ms,
                        type(exc).__name__,
                        exc,
                        attempt,
                        max_attempts,
                    )
                    if attempt >= max_attempts:
                        raise
                except Exception as exc:
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    log.warning(
                        "[qwen-remote] failed endpoint=%s model=%s latency=%.0fms "
                        "error=%s detail=%s attempt=%d/%d",
                        self.config.chat_completions_url,
                        self.config.model_name,
                        latency_ms,
                        type(exc).__name__,
                        exc,
                        attempt,
                        max_attempts,
                    )
                    raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.last_metrics = QwenRemoteMetrics(
                request_bytes=len(body),
                image_count=len(board.images),
                latency_ms=latency_ms,
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
            text, finish_reason = choice_text(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.last_finish_reason = None
            raise QwenRemoteResponseError(
                "vLLM response did not contain choices[0].message.content"
            ) from exc
        self.last_finish_reason = finish_reason
        preview = " ".join(text.split())
        log.info(
            "[qwen-remote] response status=ok model=%s images=%d request=%dB "
            "response=%dB latency=%.0fms finish_reason=%s text=%s",
            self.config.model_name,
            len(board.images),
            len(body),
            len(raw),
            latency_ms,
            finish_reason or "none",
            preview,
        )
        return text
