"""Tests for the bounded OpenAI-compatible Qwen vLLM adapter."""

from __future__ import annotations

import base64
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from marc_agent.perception.qwen_remote import (
    QwenRemoteAuthenticationError,
    QwenRemoteConnectionError,
    QwenRemoteResponseError,
    QwenRemoteTimeout,
    RemoteQwenSarVerifier,
    build_remote_payload,
    check_remote_health,
)
from marc_agent.perception.qwen_runtime import QwenRemoteConfig
from marc_agent.perception.sar_selection import CandidateBoard
from marc_agent.types import CctvDetectionRound, CctvFrame


def _config() -> QwenRemoteConfig:
    return QwenRemoteConfig(
        base_url="http://workstation:8000/v1",
        api_key="test-secret",
        model_name="qwen3-vl-8b",
        timeout_s=5.0,
        max_new_tokens=8,
    )


def _board() -> CandidateBoard:
    image = Image.new("RGB", (32, 24), color=(20, 40, 60))
    return CandidateBoard(
        images=(image,),
        prompt="Reply exactly rig_a:person_prone:0|emergency",
        evidence=(),
    )


class QwenRemoteTests(unittest.TestCase):
    def test_all_camera_scene_description_is_cached_and_used_for_selection(self):
        prompts = []
        max_tokens = []

        def post(_url, body, _headers, _timeout):
            payload = json.loads(body)
            prompt = payload["messages"][0]["content"][-1]["text"]
            prompts.append(prompt)
            max_tokens.append(payload["max_tokens"])
            output = (
                "rig_a: a park with a seated person; rig_b: an empty road"
                if "Inspect every supplied camera" in prompt
                else "rig_a:person_prone:0|emergency"
            )
            return json.dumps(
                {"choices": [{"message": {"content": output}}]}
            ).encode("utf-8")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        scene = CctvDetectionRound(
            frames={
                "rig_b": CctvFrame(
                    "rig_b", 32, 24, Image.new("RGB", (32, 24), "blue")
                ),
                "rig_a": CctvFrame(
                    "rig_a", 32, 24, Image.new("RGB", (32, 24), "green")
                ),
            }
        )

        verifier.prepare_scene(scene, "Find the collapsed person in the park.")
        self.assertEqual("rig_a:person_prone:0|emergency", verifier(_board()))
        verifier.prepare_scene(scene, "Find the collapsed person in the park.")
        self.assertEqual("rig_a:person_prone:0|emergency", verifier(_board()))

        self.assertEqual(3, len(prompts))
        self.assertIn("IMAGE_1=rig_a, IMAGE_2=rig_b", prompts[0])
        self.assertIn("Mission: Find the collapsed person", prompts[0])
        self.assertIn("Cached all-camera scene reference", prompts[1])
        self.assertIn("rig_b: an empty road", prompts[1])
        self.assertEqual(prompts[-1], verifier.last_prompt)
        self.assertEqual(192, max_tokens[0])

    def test_verifier_rejects_unbatched_multimodal_requests(self):
        image = Image.new("RGB", (8, 8))
        board = CandidateBoard(
            images=(image,) * 13,
            prompt="choose",
            evidence=(),
        )
        verifier = RemoteQwenSarVerifier(_config(), post=lambda *_args: b"")
        with self.assertRaisesRegex(QwenRemoteResponseError, "batch it to at most 6"):
            verifier(board)

    def test_payload_contains_lossless_png_image_and_deterministic_generation(self):
        image = Image.new("RGB", (32, 24), color=(20, 40, 60))
        board = CandidateBoard(
            images=(image,),
            prompt="Reply exactly rig_a:person_prone:0|emergency",
            evidence=(),
        )
        payload = json.loads(build_remote_payload(board, _config()))
        self.assertEqual("qwen3-vl-8b", payload["model"])
        self.assertEqual(0, payload["temperature"])
        self.assertFalse(payload["stream"])
        self.assertEqual({"enable_thinking": False}, payload["chat_template_kwargs"])
        content = payload["messages"][0]["content"]
        url = content[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        raw = base64.b64decode(url.split(",", 1)[1])
        self.assertTrue(raw.startswith(b"\x89PNG"))
        decoded = Image.open(io.BytesIO(raw)).convert("RGB")
        self.assertEqual(image.size, decoded.size)
        self.assertEqual(image.tobytes(), decoded.tobytes())
        self.assertEqual("text", content[-1]["type"])

    def test_verifier_posts_with_auth_and_returns_message_text(self):
        captured = {}

        def post(url, body, headers, timeout):
            captured.update(url=url, body=body, headers=headers, timeout=timeout)
            return json.dumps(
                {"choices": [{"message": {"content": "rig_a:person_prone:0|emergency"}}]}
            ).encode("utf-8")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO") as logged:
            self.assertEqual("rig_a:person_prone:0|emergency", verifier(_board()))
        self.assertIn("text=rig_a:person_prone:0|emergency", "\n".join(logged.output))
        self.assertEqual(
            "http://workstation:8000/v1/chat/completions", captured["url"]
        )
        self.assertEqual("Bearer test-secret", captured["headers"]["Authorization"])
        self.assertEqual(5.0, captured["timeout"])
        self.assertEqual(1, verifier.last_metrics.image_count)
        self.assertGreater(verifier.last_metrics.request_bytes, 100)

    def test_malformed_response_is_classified(self):
        verifier = RemoteQwenSarVerifier(_config(), post=lambda *_args: b"{}")
        with self.assertRaises(QwenRemoteResponseError):
            verifier(_board())

    def test_connection_failure_is_logged_without_api_key(self):
        def fail(*_args):
            raise QwenRemoteConnectionError("connection refused")

        verifier = RemoteQwenSarVerifier(_config(), post=fail)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO") as captured:
            with self.assertRaises(QwenRemoteConnectionError):
                verifier(_board())
        output = "\n".join(captured.output)
        self.assertIn("dispatch endpoint=", output)
        self.assertIn("error=QwenRemoteConnectionError", output)
        self.assertIn("attempt=1/2", output)
        self.assertIn("attempt=2/2", output)
        self.assertNotIn("test-secret", output)

    def test_connection_retries_once_then_returns(self):
        calls = []

        def post(*_args):
            calls.append(1)
            if len(calls) == 1:
                raise QwenRemoteConnectionError("connection refused")
            return json.dumps(
                {"choices": [{"message": {"content": "rig_a:person_prone:0|emergency"}}]}
            ).encode("utf-8")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO"):
            self.assertEqual("rig_a:person_prone:0|emergency", verifier(_board()))
        self.assertEqual(2, len(calls))

    def test_health_check_requires_configured_served_model(self):
        response = json.dumps({"data": [{"id": "qwen3-vl-8b"}]}).encode()
        captured = {}

        def get(url, headers, timeout):
            captured.update(url=url, headers=headers, timeout=timeout)
            return response

        self.assertEqual(
            ("qwen3-vl-8b",),
            check_remote_health(_config(), get=get),
        )
        self.assertEqual("http://workstation:8000/v1/models", captured["url"])
        self.assertNotIn("test-secret", captured["url"])

        missing = json.dumps({"data": [{"id": "different-model"}]}).encode()
        with self.assertRaisesRegex(QwenRemoteResponseError, "not served"):
            check_remote_health(_config(), get=lambda *_args: missing)

    def test_real_http_transport_against_fake_vllm_server(self):
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                captured["path"] = self.path
                captured["authorization"] = self.headers["Authorization"]
                captured["payload"] = json.loads(self.rfile.read(length))
                response = json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "rig_a:person_prone:0|emergency"}}
                        ]
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = QwenRemoteConfig(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="test-secret",
                model_name="qwen3-vl-8b",
                timeout_s=2.0,
                max_new_tokens=8,
            )
            output = RemoteQwenSarVerifier(config)(_board())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

        self.assertEqual("rig_a:person_prone:0|emergency", output)
        self.assertEqual("/v1/chat/completions", captured["path"])
        self.assertEqual("Bearer test-secret", captured["authorization"])
        self.assertEqual("qwen3-vl-8b", captured["payload"]["model"])

    def test_timeout_retries_once_then_returns(self):
        calls = []

        def post(*_args):
            calls.append(1)
            if len(calls) == 1:
                raise QwenRemoteTimeout("vLLM did not respond within 5.0s")
            return json.dumps(
                {"choices": [{"message": {"content": "rig_a:person_prone:0|emergency"}}]}
            ).encode("utf-8")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO") as captured:
            self.assertEqual("rig_a:person_prone:0|emergency", verifier(_board()))
        self.assertEqual(2, len(calls))
        output = "\n".join(captured.output)
        self.assertIn("deadline=5.0s", output)
        self.assertIn("budget=none", output)
        self.assertIn("attempt=1/2", output)
        self.assertIn("attempt=2/2", output)

    def test_wall_timeout_caps_deadline_and_retries_when_budget_allows(self):
        calls = []

        def post(_url, _body, _headers, timeout):
            calls.append(timeout)
            raise QwenRemoteTimeout(f"vLLM did not respond within {timeout:.1f}s")

        config = QwenRemoteConfig(
            base_url="http://workstation:8000/v1",
            api_key="test-secret",
            model_name="qwen3-vl-8b",
            timeout_s=90.0,
            max_new_tokens=8,
        )
        verifier = RemoteQwenSarVerifier(config, post=post)
        verifier.set_budget_timeout(27.9)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO") as captured:
            with self.assertRaises(QwenRemoteTimeout):
                verifier(_board())
        self.assertEqual([12.0, 12.0], calls)
        output = "\n".join(captured.output)
        self.assertIn("deadline=12.0s", output)
        self.assertIn("budget=27.9s", output)
        self.assertIn("attempt=1/2", output)
        self.assertIn("attempt=2/2", output)

    def test_timeout_retry_exhausted_raises(self):
        calls = []

        def post(*_args):
            calls.append(1)
            raise QwenRemoteTimeout("vLLM did not respond within 5.0s")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO"):
            with self.assertRaises(QwenRemoteTimeout):
                verifier(_board())
        self.assertEqual(2, len(calls))

    def test_auth_error_is_not_retried(self):
        calls = []

        def post(*_args):
            calls.append(1)
            raise QwenRemoteAuthenticationError("vLLM rejected the configured API key (HTTP 401)")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO"):
            with self.assertRaises(QwenRemoteAuthenticationError):
                verifier(_board())
        self.assertEqual(1, len(calls))

    def test_empty_content_falls_back_to_reasoning_content(self):
        def post(*_args):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_content": "P01",
                            },
                        }
                    ]
                }
            ).encode("utf-8")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        with self.assertLogs("marc_agent.qwen_remote", level="INFO") as logged:
            self.assertEqual("P01", verifier(_board()))
        self.assertEqual("stop", verifier.last_finish_reason)
        self.assertIn("finish_reason=stop", "\n".join(logged.output))

    def test_length_finish_retries_longer_after_parse_fail(self):
        from marc_agent.perception.qwen_reply import call_verifier
        from marc_agent.perception.sar_selection import parse_verifier_output

        calls = []

        def post(url, body, headers, timeout):
            payload = json.loads(body)
            calls.append(payload["max_tokens"])
            if payload["max_tokens"] < 128:
                return json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": "<think>not done"},
                            }
                        ]
                    }
                ).encode("utf-8")
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "P02"},
                        }
                    ]
                }
            ).encode("utf-8")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        ids = ("rig_a:person_bending:0", "rig_a:person_walking:0")
        with self.assertLogs("marc_agent.qwen_remote", level="INFO"):
            parsed, raw = call_verifier(
                verifier,
                _board(),
                lambda text: parse_verifier_output(text, ids),
            )
        self.assertEqual(("rig_a:person_walking:0", None), parsed)
        self.assertEqual("P02", raw)
        self.assertEqual([8, 128], calls)


if __name__ == "__main__":
    unittest.main()
