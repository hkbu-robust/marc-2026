from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_auto_start_delay_patch_is_idempotent():
    path = Path(__file__).parents[1] / "simulation-platform/patches/patch_runtime_auto_start_delay.py"
    spec = spec_from_file_location("auto_start_delay_patch", path)
    patch = module_from_spec(spec)
    spec.loader.exec_module(patch)
    source = (
        "class Manager:\n"
        "    def __init__(self):\n"
        + patch.INIT_OLD
        + "    def tick(self):\n"
        + patch.WAIT_OLD
        + "    def set(self, wait_register):\n"
        + patch.SET_OLD
        + "    def reset(self):\n"
        + patch.RESET_OLD
    )
    result = patch.patch_source(source)
    assert patch.MARKER in result
    assert "self._auto_registered_at = None" in result
    assert "self.sim_ctx.play()" in result
    assert patch.patch_source(result) == result
