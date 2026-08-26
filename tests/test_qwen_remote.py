"""Tests for the bounded OpenAI-compatible Qwen vLLM adapter."""

from __future__ import annotations

import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from marc_agent.perception.qwen_remote import (
    QwenRemoteResponseError,
    RemoteQwenSarVerifier,
    build_remote_payload,
)
from marc_agent.perception.qwen_runtime import QwenRemoteConfig
from marc_agent.perception.sar_selection import CandidateBoard


def _config() -> QwenRemoteConfig:
    return QwenRemoteConfig(
        base_url="http://workstation:8000/v1",
        api_key="test-secret",
        model_name="qwen3-vl-8b",
        timeout_s=5.0,
        max_new_tokens=8,
        jpeg_quality=80,
    )


def _board() -> CandidateBoard:
    image = Image.new("RGB", (32, 24), color=(20, 40, 60))
    return CandidateBoard(
        images=(image,),
        prompt="Reply exactly rig_a:person:0|emergency",
        evidence=(),
    )


class QwenRemoteTests(unittest.TestCase):
    def test_payload_contains_base64_image_and_deterministic_generation(self):
        payload = json.loads(build_remote_payload(_board(), _config()))
        self.assertEqual("qwen3-vl-8b", payload["model"])
        self.assertEqual(0, payload["temperature"])
        self.assertFalse(payload["stream"])
        content = payload["messages"][0]["content"]
        url = content[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        self.assertTrue(base64.b64decode(url.split(",", 1)[1]).startswith(b"\xff\xd8"))
        self.assertEqual("text", content[-1]["type"])

    def test_verifier_posts_with_auth_and_returns_message_text(self):
        captured = {}

        def post(url, body, headers, timeout):
            captured.update(url=url, body=body, headers=headers, timeout=timeout)
            return json.dumps(
                {"choices": [{"message": {"content": "rig_a:person:0|emergency"}}]}
            ).encode("utf-8")

        verifier = RemoteQwenSarVerifier(_config(), post=post)
        self.assertEqual("rig_a:person:0|emergency", verifier(_board()))
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
                            {"message": {"content": "rig_a:person:0|emergency"}}
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
                jpeg_quality=80,
            )
            output = RemoteQwenSarVerifier(config)(_board())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

        self.assertEqual("rig_a:person:0|emergency", output)
        self.assertEqual("/v1/chat/completions", captured["path"])
        self.assertEqual("Bearer test-secret", captured["authorization"])
        self.assertEqual("qwen3-vl-8b", captured["payload"]["model"])


if __name__ == "__main__":
    unittest.main()
