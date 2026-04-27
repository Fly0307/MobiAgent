import json
import shutil
import sys
import types
import uuid
from pathlib import Path


def install_test_stubs() -> None:
    if "openai" not in sys.modules:
        openai_module = types.ModuleType("openai")

        class OpenAI:  # pragma: no cover - simple import stub
            def __init__(self, *args, **kwargs):
                pass

        openai_module.OpenAI = OpenAI
        sys.modules["openai"] = openai_module

    if "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")

    if "PIL" not in sys.modules:
        pil_module = types.ModuleType("PIL")
        image_module = types.ModuleType("PIL.Image")
        draw_module = types.ModuleType("PIL.ImageDraw")
        font_module = types.ModuleType("PIL.ImageFont")

        class _DummyImage:  # pragma: no cover - simple import stub
            size = (100, 100)

            def convert(self, *args, **kwargs):
                return self

            def crop(self, *args, **kwargs):
                return self

            def resize(self, *args, **kwargs):
                return self

            def getdata(self):
                return [0] * 64

        def _open(*args, **kwargs):
            return _DummyImage()

        image_module.Image = _DummyImage
        image_module.open = _open
        draw_module.Draw = lambda *args, **kwargs: object()
        font_module.FreeTypeFont = object
        font_module.load_default = lambda *args, **kwargs: object()

        pil_module.Image = image_module
        pil_module.ImageDraw = draw_module
        pil_module.ImageFont = font_module

        sys.modules["PIL"] = pil_module
        sys.modules["PIL.Image"] = image_module
        sys.modules["PIL.ImageDraw"] = draw_module
        sys.modules["PIL.ImageFont"] = font_module

    if "auto_explore.adapters.device" not in sys.modules:
        device_module = types.ModuleType("auto_explore.adapters.device")

        class _BaseDevice:  # pragma: no cover - simple import stub
            def __init__(self, *args, **kwargs):
                self._last_click = None

            def start_app(self, *args, **kwargs):
                return None

            def click(self, *args, **kwargs):
                self._last_click = args
                return None

            def dump_hierarchy(self):
                return "<root />"

            def swipe(self, *args, **kwargs):
                return None

        def _json_or_empty(content):
            try:
                return json.loads(content)
            except Exception:
                return {}

        device_module.AndroidDevice = _BaseDevice
        device_module.HarmonyDevice = _BaseDevice
        device_module.get_screenshot = lambda *args, **kwargs: ""
        device_module.robust_json_loads = _json_or_empty
        device_module.call_model_with_validation_retry = lambda *args, **kwargs: {}
        device_module.compute_swipe_positions = lambda *args, **kwargs: ((0, 0), (10, 10))
        device_module.convert_qwen3_coordinates_to_absolute = (
            lambda value, *args, **kwargs: value
        )
        device_module.validate_decider_response = lambda *args, **kwargs: True
        sys.modules["auto_explore.adapters.device"] = device_module

    if "auto_explore.adapters.snapshot_init" not in sys.modules:
        snapshot_module = types.ModuleType("auto_explore.adapters.snapshot_init")

        def load_snapshot_manager_client():
            class SnapshotManagerClient:  # pragma: no cover - simple import stub
                def __init__(self, *args, **kwargs):
                    pass

            return SnapshotManagerClient

        snapshot_module.load_snapshot_manager_client = load_snapshot_manager_client
        sys.modules["auto_explore.adapters.snapshot_init"] = snapshot_module


def make_workspace_tempdir(name: str) -> Path:
    root = Path.cwd() / "auto_explore" / "tests" / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_workspace_tempdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
