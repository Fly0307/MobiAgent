import concurrent.futures
import json
import logging
import os
import shutil
import textwrap
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image, ImageDraw, ImageFont

from auto_explore.core.settings import _ANNOTATION_EXECUTOR


_PENDING_ARTIFACT_FUTURES: set[concurrent.futures.Future] = set()
_PENDING_ARTIFACT_FUTURES_LOCK = threading.Lock()


def _register_artifact_future(future: concurrent.futures.Future) -> concurrent.futures.Future:
    with _PENDING_ARTIFACT_FUTURES_LOCK:
        _PENDING_ARTIFACT_FUTURES.add(future)

    def _cleanup(done_future: concurrent.futures.Future) -> None:
        with _PENDING_ARTIFACT_FUTURES_LOCK:
            _PENDING_ARTIFACT_FUTURES.discard(done_future)

    future.add_done_callback(_cleanup)
    return future


def submit_artifact_task(func, *args):
    return _register_artifact_future(_ANNOTATION_EXECUTOR.submit(func, *args))


def flush_artifact_tasks(timeout: Optional[float] = None) -> None:
    with _PENDING_ARTIFACT_FUTURES_LOCK:
        futures = list(_PENDING_ARTIFACT_FUTURES)
    if futures:
        concurrent.futures.wait(futures, timeout=timeout)


def _cv2_imread_unicode(path: str):
    """cv2.imread that handles non-ASCII paths on Windows."""
    import numpy as np

    try:
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _cv2_imwrite_unicode(path: str, img) -> bool:
    """cv2.imwrite that handles non-ASCII paths on Windows."""
    try:
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False


def _load_font(size: int = 36) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("msyh.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _record_artifact_io(metrics, start_time: float) -> None:
    if metrics is not None:
        metrics.record_artifact_io(time.perf_counter() - start_time)


def _run_annotation_safe(
    action: str,
    action_record: Dict[str, Any],
    screenshot_file: str,
    output_dir: str,
    step_index: int,
    metrics=None,
) -> None:
    """Safe wrapper for action annotation work."""
    start = time.perf_counter()
    try:
        annotate_action_visuals(action, action_record, screenshot_file, output_dir, step_index)
    except Exception as e:
        logging.warning(f"[annotation] step={step_index}: {e}")
    finally:
        _record_artifact_io(metrics, start)


def annotate_action_visuals(
    action: str,
    action_record: Dict[str, Any],
    screenshot_file: str,
    data_dir: str,
    step_index: int,
) -> None:
    """Generate visualization images for the current action."""
    try:
        img = Image.open(screenshot_file)
    except Exception as e:
        logging.warning(f"Failed to open screenshot for annotation: {e}")
        return

    draw = ImageDraw.Draw(img)
    font = _load_font()
    text = f"{action.upper()} {action_record.get('source_task', '')}".strip()
    text = textwrap.fill(text, width=24)

    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
    except Exception:
        text_width = 0

    text_x = max(0, (img.width - text_width) // 2)
    draw.text((text_x, 0), text, fill="red", font=font)

    highlighted_path = os.path.join(data_dir, f"{step_index}_highlighted.jpg")
    img.save(highlighted_path)

    if action in {"click", "click_input"}:
        bounds = action_record.get("bounds")
        if bounds and len(bounds) == 4:
            x1, y1, x2, y2 = bounds
            img_bounds = Image.open(highlighted_path)
            draw_bounds = ImageDraw.Draw(img_bounds)
            draw_bounds.rectangle([x1, y1, x2, y2], outline="red", width=5)
            bounds_path = os.path.join(data_dir, f"{step_index}_bounds.jpg")
            img_bounds.save(bounds_path)

            cv2image = _cv2_imread_unicode(bounds_path)
            if cv2image is not None:
                x = action_record.get("position_x")
                y = action_record.get("position_y")
                if x is not None and y is not None:
                    cv2.circle(cv2image, (int(x), int(y)), 12, (0, 255, 0), -1)
                    click_point_path = os.path.join(data_dir, f"{step_index}_click_point.jpg")
                    _cv2_imwrite_unicode(click_point_path, cv2image)

    if action == "swipe":
        sx = action_record.get("press_position_x")
        sy = action_record.get("press_position_y")
        ex = action_record.get("release_position_x")
        ey = action_record.get("release_position_y")
        if None not in (sx, sy, ex, ey):
            cv2image = _cv2_imread_unicode(highlighted_path)
            if cv2image is not None:
                cv2.arrowedLine(
                    cv2image,
                    (int(sx), int(sy)),
                    (int(ex), int(ey)),
                    (255, 0, 0),
                    6,
                    tipLength=0.3,
                )
                swipe_path = os.path.join(data_dir, f"{step_index}_swipe.jpg")
                _cv2_imwrite_unicode(swipe_path, cv2image)


def _write_hierarchy_safe(
    hierarchy: Any,
    device_type: str,
    data_dir: str,
    step_index: int,
    metrics=None,
) -> None:
    """Write hierarchy snapshots safely."""
    start = time.perf_counter()
    try:
        if device_type == "Android":
            path = os.path.join(data_dir, f"{step_index}.xml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(hierarchy))
            return

        path = os.path.join(data_dir, f"{step_index}.json")
        try:
            obj = json.loads(hierarchy) if isinstance(hierarchy, str) else hierarchy
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(hierarchy))
    except Exception as e:
        logging.warning(f"[hierarchy write] step={step_index}: {e}")
    finally:
        _record_artifact_io(metrics, start)


def save_hierarchy(
    device,
    device_type: str,
    data_dir: str,
    step_index: int,
    *,
    async_enabled: bool = True,
    metrics=None,
) -> None:
    """Persist the current UI hierarchy."""
    try:
        hierarchy = device.dump_hierarchy()
    except Exception as e:
        logging.error(f"Dump hierarchy failed: {e}")
        hierarchy = "<hierarchy_dump_failed/>" if device_type == "Android" else {}

    if async_enabled:
        submit_artifact_task(_write_hierarchy_safe, hierarchy, device_type, data_dir, step_index, metrics)
    else:
        _write_hierarchy_safe(hierarchy, device_type, data_dir, step_index, metrics)


def get_current_screenshot_path(device_type: str) -> str:
    name = "screenshot-Android.jpg" if device_type == "Android" else "screenshot-Harmony.jpg"
    return os.path.join(os.getcwd(), name)


def save_raw_screenshot(data_dir: str, step_index: int, device_type: str, *, metrics=None) -> str:
    start = time.perf_counter()
    src = get_current_screenshot_path(device_type)
    dst = os.path.join(data_dir, f"{step_index}.jpg")
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        wf.write(rf.read())
    _record_artifact_io(metrics, start)
    return dst


def save_named_raw_screenshot(data_dir: str, filename: str, device_type: str, *, metrics=None) -> str:
    start = time.perf_counter()
    src = get_current_screenshot_path(device_type)
    dst = os.path.join(data_dir, filename)
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        wf.write(rf.read())
    _record_artifact_io(metrics, start)
    return dst


def _compute_task_description(
    actions: List[Dict[str, Any]],
    app_name: str,
    task_description: Optional[str] = None,
    step_words: Optional[Dict[int, str]] = None,
) -> str:
    step_words = step_words or {}
    step_tasks: List[str] = []
    for item in actions:
        step_task = str(item.get("source_task", "")).strip()
        if step_task:
            step_tasks.append(step_task)

    if step_tasks:
        step_desc_parts = []
        for idx, step_task in enumerate(step_tasks, 1):
            step_label = step_words.get(idx, f"Step {idx}")
            step_desc_parts.append(f"{step_label}: {step_task}")
        return f"Open {app_name}, " + "; ".join(step_desc_parts)
    return task_description or f"Open {app_name}"


def persist_outputs(
    output_dir: str,
    app_name: str,
    actions: List[Dict[str, Any]],
    reacts: List[Dict[str, Any]],
    task_description: Optional[str] = None,
) -> None:
    step_words = {
        1: "Step 1",
        2: "Step 2",
        3: "Step 3",
        4: "Step 4",
        5: "Step 5",
        6: "Step 6",
        7: "Step 7",
        8: "Step 8",
        9: "Step 9",
        10: "Step 10",
    }

    computed_task_description = _compute_task_description(
        actions=actions,
        app_name=app_name,
        task_description=task_description,
        step_words=step_words,
    )

    normalized_actions: List[Dict[str, Any]] = []
    for idx, item in enumerate(actions, 1):
        normalized = dict(item)
        normalized["action_index"] = idx
        normalized.pop("source_task", None)
        normalized_actions.append(normalized)

    from datetime import datetime

    now = datetime.now()
    execution_timestamp = {
        "date": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "time": now.strftime("%H:%M:%S"),
    }

    normalized_reacts: List[Dict[str, Any]] = []
    for idx, item in enumerate(reacts, 1):
        normalized = dict(item)
        normalized["action_index"] = idx
        normalized.pop("source_task", None)
        normalized_reacts.append(normalized)

    payload = {
        "app_name": app_name,
        "task_type": "auto_search",
        "old_task_description": None,
        "task_description": computed_task_description,
        "execution_timestamp": execution_timestamp,
        "action_count": len(normalized_actions),
        "actions": normalized_actions,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, "react.json"), "w", encoding="utf-8") as f:
        json.dump(normalized_reacts, f, ensure_ascii=False, indent=4)


def persist_step_output(
    output_dir: str,
    app_name: str,
    action_record: Dict[str, Any],
    react_item: Dict[str, Any],
) -> None:
    persist_outputs(
        output_dir,
        app_name,
        [action_record],
        [react_item],
        task_description=action_record.get("source_task"),
    )


def _persist_step_output_safe(
    output_dir: str,
    app_name: str,
    action_record: Dict[str, Any],
    react_item: Dict[str, Any],
    metrics=None,
) -> None:
    """Safe wrapper for per-step JSON persistence."""
    start = time.perf_counter()
    try:
        persist_step_output(output_dir, app_name, action_record, react_item)
    except Exception as e:
        logging.warning(f"[persist step] {e}")
    finally:
        _record_artifact_io(metrics, start)


def _persist_outputs_safe(
    output_dir: str,
    app_name: str,
    actions: List[Dict[str, Any]],
    reacts: List[Dict[str, Any]],
    metrics=None,
) -> None:
    """Safe wrapper for final path JSON persistence."""
    start = time.perf_counter()
    try:
        persist_outputs(output_dir, app_name, actions, reacts)
    except Exception as e:
        logging.warning(f"[persist outputs] {e}")
    finally:
        _record_artifact_io(metrics, start)


def copy_step_artifacts_to_path(steps_dir: str, path_dir: str, step_indices: List[int], *, metrics=None) -> None:
    """Copy step artifacts into a finalized path directory."""
    start = time.perf_counter()
    os.makedirs(path_dir, exist_ok=True)
    for new_idx, step_index in enumerate(step_indices, 1):
        step_dir = os.path.join(steps_dir, f"step_{step_index:04d}")
        if not os.path.isdir(step_dir):
            logging.warning(f"Step dir not found for path copy: {step_dir}")
            continue
        old_prefix = f"{step_index}"
        new_prefix = f"{new_idx}"
        for name in os.listdir(step_dir):
            src = os.path.join(step_dir, name)
            if not os.path.isfile(src):
                continue
            if name.startswith(old_prefix + ".") or name.startswith(old_prefix + "_"):
                renamed = new_prefix + name[len(old_prefix) :]
            else:
                renamed = name
            dst = os.path.join(path_dir, renamed)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logging.warning(f"Failed to copy {src} -> {dst}: {e}")
    _record_artifact_io(metrics, start)


def save_named_hierarchy(
    device,
    device_type: str,
    data_dir: str,
    base_name: str,
    *,
    async_enabled: bool = True,
    metrics=None,
) -> None:
    step_index = str(base_name)
    save_hierarchy(
        device,
        device_type,
        data_dir,
        step_index,
        async_enabled=async_enabled,
        metrics=metrics,
    )


def write_trace_meta(output_dir: str, payload: Dict[str, Any], *, metrics=None) -> None:
    start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "trace_meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _record_artifact_io(metrics, start)


__all__ = [
    "_compute_task_description",
    "_cv2_imread_unicode",
    "_cv2_imwrite_unicode",
    "_load_font",
    "_persist_outputs_safe",
    "_persist_step_output_safe",
    "_run_annotation_safe",
    "_write_hierarchy_safe",
    "annotate_action_visuals",
    "copy_step_artifacts_to_path",
    "flush_artifact_tasks",
    "get_current_screenshot_path",
    "persist_outputs",
    "persist_step_output",
    "save_hierarchy",
    "save_named_hierarchy",
    "save_named_raw_screenshot",
    "save_raw_screenshot",
    "submit_artifact_task",
    "write_trace_meta",
]
