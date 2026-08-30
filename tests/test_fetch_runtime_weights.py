"""Skip-if-complete behaviour for build-time weight fetch (no network)."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_FETCH = Path(__file__).resolve().parents[1] / "models" / "runtime" / "fetch_runtime_weights.py"
_SPEC = importlib.util.spec_from_file_location("fetch_runtime_weights", _FETCH)
_FETCH_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_FETCH_MOD)

from marc_agent.graspgenx_policy import gripper_assets_dir, gripper_name, has_checkpoint

QWEN_REVISION = _FETCH_MOD.QWEN_REVISION
YOLO11X_SHA256 = _FETCH_MOD.YOLO11X_SHA256
fetch_all = _FETCH_MOD.fetch_all
fetch_graspgenx = _FETCH_MOD.fetch_graspgenx
fetch_grippers = _FETCH_MOD.fetch_grippers
fetch_qwen = _FETCH_MOD.fetch_qwen
fetch_yolo11x = _FETCH_MOD.fetch_yolo11x


class FetchRuntimeWeightsTests(unittest.TestCase):
    def test_qwen_pin_matches_contract(self):
        contract = json.loads(
            Path("models/runtime/stage1-sar-qwen/runtime_contract.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(QWEN_REVISION, contract["model"]["revision"])

    def test_incomplete_qwen_calls_downloader(self):
        def fake_snapshot(*_args, **kwargs):
            dest = Path(kwargs["local_dir"])
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                (dest / name).write_text("{}", encoding="utf-8")
            (dest / "model.safetensors").write_bytes(b"weights")

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            status = fetch_qwen(dest, snapshot_download=fake_snapshot)
        self.assertTrue(status.startswith("fetched qwen"))

    def test_skips_complete_qwen_snapshot(self):
        downloader = Mock(side_effect=AssertionError("must not download"))
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                (model_dir / name).write_text("{}", encoding="utf-8")
            shard = "model-00001-of-00001.safetensors"
            (model_dir / shard).write_bytes(b"weights")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"w": shard}}), encoding="utf-8"
            )
            status = fetch_qwen(model_dir, snapshot_download=downloader)
        self.assertTrue(status.startswith("skip qwen"))
        downloader.assert_not_called()

    def test_skips_complete_graspgenx_checkpoint(self):
        downloader = Mock(side_effect=AssertionError("must not download"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gen", "dis"):
                folder = root / name
                folder.mkdir()
                (folder / "epoch_1.pth").write_bytes(b"ckpt")
            status = fetch_graspgenx(root, snapshot_download=downloader)
        self.assertTrue(status.startswith("skip graspgenx"))
        downloader.assert_not_called()

    def test_skips_complete_gripper_tree(self):
        downloader = Mock(side_effect=AssertionError("must not download"))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "grippers"
            (dest / "gripper_descriptions" / "assets" / "x_grippers" / "robotiq_2f_140").mkdir(
                parents=True
            )
            status = fetch_grippers(dest, snapshot_download=downloader)
        self.assertTrue(status.startswith("skip grippers"))
        downloader.assert_not_called()

    def test_skips_existing_yolo11x_with_matching_sha(self):
        downloader = Mock(side_effect=AssertionError("must not download"))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "yolo11x.pt"
            dest.write_bytes(b"x" * 2_000_000)
            with patch.object(_FETCH_MOD, "_sha256", return_value=YOLO11X_SHA256):
                status = fetch_yolo11x(dest, hub_download=downloader)
        self.assertTrue(status.startswith("skip yolo11x"))
        downloader.assert_not_called()

    def test_wrong_yolo11x_sha_redownloads(self):
        calls = []

        def fake_hub(*, repo_id, filename, local_dir):
            calls.append(repo_id)
            path = Path(local_dir) / filename
            path.write_bytes(b"y" * 2_000_000)
            return str(path)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "yolo11x.pt"
            dest.write_bytes(b"x" * 2_000_000)
            with patch.object(
                _FETCH_MOD,
                "_sha256",
                side_effect=["deadbeef" * 8, YOLO11X_SHA256],
            ):
                status = fetch_yolo11x(dest, hub_download=fake_hub)
        self.assertTrue(status.startswith("fetched yolo11x"))
        self.assertEqual(["Ultralytics/YOLO11"], calls)

    def test_gen_only_checkpoint_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gen").mkdir()
            (root / "gen" / "epoch_1.pth").write_bytes(b"ckpt")
            (root / "dis").mkdir()
            self.assertFalse(has_checkpoint(str(root)))

    def test_fetch_all_skips_every_complete_artifact(self):
        downloader = Mock(side_effect=AssertionError("must not download"))
        hub = Mock(side_effect=AssertionError("must not download"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen = root / "qwen"
            qwen.mkdir()
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                (qwen / name).write_text("{}", encoding="utf-8")
            (qwen / "model.safetensors").write_bytes(b"weights")
            grasp = root / "graspgenx"
            for name in ("gen", "dis"):
                folder = grasp / name
                folder.mkdir(parents=True)
                (folder / "epoch_1.pth").write_bytes(b"ckpt")
            grippers = root / "grippers"
            (grippers / "assets" / "x_grippers").mkdir(parents=True)
            yolo = root / "yolo11x.pt"
            yolo.write_bytes(b"x" * 2_000_000)
            with patch.object(_FETCH_MOD, "yolo11x_ready", return_value=True):
                lines = fetch_all(
                    paths={
                        "qwen": qwen,
                        "graspgenx": grasp,
                        "grippers": grippers,
                        "yolo11x": yolo,
                    },
                    snapshot_download=downloader,
                    hub_download=hub,
                    bake_qwen=True,
                )
        self.assertTrue(all(item.startswith("skip ") for item in lines))
        downloader.assert_not_called()
        hub.assert_not_called()

    def test_include_qwen_defaults_to_local(self):
        self.assertTrue(_FETCH_MOD.include_qwen({}))
        self.assertTrue(_FETCH_MOD.include_qwen({"MARC_QWEN_BACKEND": "local"}))
        self.assertFalse(_FETCH_MOD.include_qwen({"MARC_QWEN_BACKEND": "remote"}))
        self.assertFalse(_FETCH_MOD.include_qwen({"MARC_QWEN_BACKEND": "REMOTE"}))

    def test_fetch_all_skips_qwen_when_backend_is_not_local(self):
        downloader = Mock(side_effect=AssertionError("must not download"))
        hub = Mock(side_effect=AssertionError("must not download"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen = root / "qwen"
            qwen.mkdir()
            grasp = root / "graspgenx"
            for name in ("gen", "dis"):
                folder = grasp / name
                folder.mkdir(parents=True)
                (folder / "epoch_1.pth").write_bytes(b"ckpt")
            grippers = root / "grippers"
            (grippers / "assets" / "x_grippers").mkdir(parents=True)
            yolo = root / "yolo11x.pt"
            yolo.write_bytes(b"x" * 2_000_000)
            with patch.dict(os.environ, {"MARC_QWEN_BACKEND": "remote"}, clear=False):
                with patch.object(_FETCH_MOD, "yolo11x_ready", return_value=True):
                    lines = fetch_all(
                        paths={
                            "qwen": qwen,
                            "graspgenx": grasp,
                            "grippers": grippers,
                            "yolo11x": yolo,
                        },
                        snapshot_download=downloader,
                        hub_download=hub,
                    )
        self.assertEqual("skip qwen (MARC_QWEN_BACKEND is not local)", lines[0])
        self.assertTrue(all(item.startswith("skip ") for item in lines))
        downloader.assert_not_called()
        hub.assert_not_called()

    def test_fast_build_does_not_fetch_optional_runtime_weights(self):
        downloader = Mock(side_effect=AssertionError("must not download GraspGenX"))
        hub = Mock(side_effect=AssertionError("must not download YOLO"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(_FETCH_MOD, "yolo11x_ready", return_value=True):
                lines = fetch_all(
                    paths={
                        "qwen": root / "qwen",
                        "graspgenx": root / "graspgenx",
                        "grippers": root / "grippers",
                        "yolo11x": root / "yolo11x.pt",
                    },
                    snapshot_download=downloader,
                    hub_download=hub,
                    bake_qwen=False,
                )
        self.assertEqual(["skip qwen (MARC_QWEN_BACKEND is not local)"], lines)
        downloader.assert_not_called()
        hub.assert_not_called()

    def test_remote_compose_passes_backend_as_build_arg(self):
        text = Path("demo/docker-compose.qwen-remote.yml").read_text(encoding="utf-8")
        self.assertIn("MARC_QWEN_BACKEND: remote", text)
        self.assertIn("args:", text)

    def test_root_compose_pins_local_qwen(self):
        text = Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('MARC_SAR_DETECTOR: "qwen"', text)
        self.assertIn('MARC_QWEN_BACKEND: "local"', text)

    def test_demo_compose_uses_host_qwen(self):
        text = Path("demo/docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('MARC_SAR_DETECTOR: "qwen-remote"', text)
        self.assertIn("MARC_QWEN_BACKEND: remote", text)
        self.assertIn("http://127.0.0.1:8000/v1", text)
        self.assertIn("qwen3-vl-8b", text)
        self.assertIn("runtime: nvidia", text)
        self.assertIn('NVIDIA_VISIBLE_DEVICES: "all"', text)
        self.assertIn('NVIDIA_DRIVER_CAPABILITIES: "compute,utility"', text)
        self.assertIn('MARC_CCTV_YOLO_DEVICE: "0"', text)
        self.assertIn('MARC_YOLO_DEVICE: "0"', text)

    def test_agent_image_uses_blackwell_capable_torch(self):
        text = Path("demo/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("torch==2.7.1 torchvision==0.22.1", text)
        self.assertIn("https://download.pytorch.org/whl/cu128", text)
        self.assertNotIn("https://download.pytorch.org/whl/cu121", text)

    def test_local_mount_skips_bake_but_runs_local_backend(self):
        text = Path("demo/docker-compose.qwen-local.yml").read_text(encoding="utf-8")
        self.assertIn("build:", text)
        self.assertIn("MARC_QWEN_BACKEND: remote", text)
        self.assertIn('MARC_QWEN_BACKEND: "local"', text)

    def test_default_gripper_is_robotiq_2f_140(self):
        with patch.dict(os.environ):
            os.environ.pop("MARC_GRASPGENX_GRIPPER", None)
            self.assertEqual("robotiq_2f_140", gripper_name())

    def test_gripper_assets_dir_finds_nested_x_grippers(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "gripper_descriptions" / "assets" / "x_grippers"
            nested.mkdir(parents=True)
            self.assertEqual(
                str(Path(tmp) / "gripper_descriptions" / "assets"),
                gripper_assets_dir(tmp),
            )
            self.assertFalse(has_checkpoint(tmp))

    def test_429_retries_then_succeeds(self):
        calls = []
        slept = []

        def flaky(**_kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise _http_429(retry_after="2")
            return "ok"

        result = _FETCH_MOD._hub_call(
            "grippers",
            flaky,
            sleeper=slept.append,
            delays=(5, 9, 15),
        )
        self.assertEqual("ok", result)
        self.assertEqual(3, len(calls))
        self.assertEqual([2.0, 2.0], slept)

    def test_404_does_not_retry(self):
        calls = []

        def missing(**_kwargs):
            calls.append(1)
            exc = Exception("404 Client Error: Not Found")
            exc.status_code = 404
            raise exc

        with self.assertRaises(Exception) as caught:
            _FETCH_MOD._hub_call(
                "qwen",
                missing,
                sleeper=lambda _dt: None,
                delays=(30, 60),
            )
        self.assertEqual(404, getattr(caught.exception, "status_code"))
        self.assertEqual(1, len(calls))

    def test_exhausted_429_reraises(self):
        calls = []

        def always_429(**_kwargs):
            calls.append(1)
            raise _http_429()

        with self.assertRaises(Exception) as caught:
            _FETCH_MOD._hub_call(
                "grippers",
                always_429,
                sleeper=lambda _dt: None,
                delays=(1, 1),
            )
        self.assertEqual(429, _FETCH_MOD._http_status(caught.exception))
        self.assertEqual(3, len(calls))

    def test_fetch_grippers_retries_rate_limit(self):
        calls = []

        def flaky(*_args, **kwargs):
            calls.append(kwargs["repo_id"])
            if len(calls) == 1:
                raise _http_429()
            dest = Path(kwargs["local_dir"])
            (dest / "x_grippers").mkdir()
            return str(dest)

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "grippers"
            with patch.object(_FETCH_MOD.time, "sleep") as slept:
                status = fetch_grippers(dest, snapshot_download=flaky)
        self.assertTrue(status.startswith("fetched grippers"))
        self.assertEqual(2, len(calls))
        slept.assert_called()


class AgentDockerfileTests(unittest.TestCase):
    def test_fast_build_does_not_install_graspgenx(self):
        text = Path("demo/Dockerfile").read_text(encoding="utf-8")
        active = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertIn(
            'pip3 install --upgrade "packaging>=26.2" "setuptools>=64,<78"',
            active,
        )
        self.assertNotIn("/tmp/GraspGenX", active)
        self.assertNotIn("from graspgenx", active)
        self.assertNotIn(
            "pip3 install --no-deps --no-build-isolation \\\n"
            "        git+https://github.com/NVlabs/GraspGenX.git",
            text,
        )
        self.assertIn("ARG MARC_QWEN_BACKEND=local", text)
        self.assertIn("MARC_QWEN_BACKEND=\"$MARC_QWEN_BACKEND\"", text)
        self.assertIn("fetch_runtime_weights.py", text)


def _http_429(retry_after=None):
    exc = Exception("429 Client Error: Too Many Requests")
    exc.status_code = 429
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    exc.response = SimpleNamespace(status_code=429, headers=headers)
    return exc


if __name__ == "__main__":
    unittest.main()
