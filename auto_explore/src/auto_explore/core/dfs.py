import difflib
import json
import logging
import os
import queue
import threading
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from auto_explore.adapters.device import convert_qwen3_coordinates_to_absolute, get_screenshot
from auto_explore.core import settings
from auto_explore.core.artifacts import (
    _persist_outputs_safe,
    _persist_step_output_safe,
    _compute_task_description,
    copy_step_artifacts_to_path,
    save_hierarchy,
    save_named_hierarchy,
    save_named_raw_screenshot,
    save_raw_screenshot,
    submit_artifact_task,
    write_trace_meta,
)
from auto_explore.core.decider import WaitActionSkip, execute_decider_one_step
from auto_explore.core.explorer import (
    ExplorerCache,
    ScreenStateCache,
    _capture_screen,
    _compute_adaptive_similarity_threshold,
    _deduplicate_candidates,
    call_explorer_model,
    get_hierarchy_text,
)
from auto_explore.core.fingerprints import (
    _detect_countdown,
    _hamming_distance_hex,
    _hierarchy_fingerprint,
    _normalize_hierarchy_text,
    _simple_verify,
    _stable_text_fingerprint,
    _triple_verify,
    compute_fingerprints,
)
from auto_explore.core.navigation import (
    _find_dismissible_element,
    _get_current_screen_size,
    _is_app_in_foreground,
    navigate_back,
    perform_backtrack_action,
    replay_action_record,
)
from auto_explore.core.prompting import append_done_to_path
from auto_explore.core.ui_collect import enqueue_ui_collect_if_new


def _build_explorer_trace(metrics, *, current_depth: int, stage: str, screenshot_started: float, screenshot_finished: float):
    return {
        "phase": "explorer",
        "current_depth": current_depth,
        "stage": stage,
        "T0": metrics.relative_time(screenshot_started),
        "T1": metrics.relative_time(screenshot_finished),
    }


_PROGRESS_STRONG = "strong_progress"
_PROGRESS_WEAK = "weak_progress"
_PROGRESS_NONE = "no_progress"
_PARTIAL_PATH_MIN_LENGTH = 3
_WEAK_PROGRESS_TASK_KEYWORDS = (
    "search",
    "搜索",
    "查找",
    "tab",
    "标签",
    "导航",
    "首页",
    "我的",
    "发现",
    "切换",
)


def _current_screenshot_path(device_type: str) -> str:
    return "screenshot-Android.jpg" if device_type == "Android" else "screenshot-Harmony.jpg"


def _sequence_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(
        None,
        _normalize_hierarchy_text(a),
        _normalize_hierarchy_text(b),
    ).ratio()


def _extract_ui_state_markers(hierarchy_text: str) -> Dict[str, set]:
    markers = {
        "selected": set(),
        "focused": set(),
        "editable_texts": set(),
        "hint_texts": set(),
    }
    if not hierarchy_text:
        return markers

    def _marker(text: str, desc: str, res_id: str, cls: str) -> str:
        for value in (text, desc, res_id, cls):
            if value:
                return str(value).strip()
        return ""

    def _process_attrs(attrs: Dict[str, object]) -> None:
        text = str(attrs.get("text", "") or attrs.get("content", "")).strip()
        desc = str(attrs.get("content-desc", "") or attrs.get("contentDescription", "")).strip()
        res_id = str(attrs.get("resource-id", "") or attrs.get("id", "")).strip()
        cls = str(attrs.get("class", "") or attrs.get("className", "")).strip()
        hint = str(attrs.get("hint", "") or attrs.get("placeholder", "")).strip()
        marker = _marker(text, desc, res_id, cls)
        selected = str(attrs.get("selected", "false")).lower() in {"true", "1"}
        focused = str(attrs.get("focused", "false")).lower() in {"true", "1"}
        clickable = str(attrs.get("clickable", "false")).lower() in {"true", "1"}
        editable = (
            str(attrs.get("editable", "false")).lower() in {"true", "1"}
            or "edittext" in cls.lower()
            or "search" in res_id.lower()
        )
        if selected and marker:
            markers["selected"].add(marker)
        if focused and marker:
            markers["focused"].add(marker)
        if editable or (clickable and ("search" in marker.lower() or "搜索" in marker)):
            if text:
                markers["editable_texts"].add(text)
            if hint:
                markers["hint_texts"].add(hint)

    if hierarchy_text.lstrip().startswith("<"):
        try:
            root = ET.fromstring(hierarchy_text)
        except Exception:
            return markers
        for node in root.iter():
            _process_attrs(node.attrib)
        return markers

    try:
        payload = json.loads(hierarchy_text)
    except Exception:
        return markers

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            attrs = node.get("attributes") if isinstance(node.get("attributes"), dict) else node
            if isinstance(attrs, dict):
                _process_attrs(attrs)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return markers


def _is_weak_progress_task(task: str, action_type: str) -> bool:
    if action_type in {"input", "click_input"}:
        return True
    task_lower = task.lower()
    return any(keyword in task or keyword in task_lower for keyword in _WEAK_PROGRESS_TASK_KEYWORDS)


def _states_equivalent(
    pre_hierarchy_text: str,
    pre_struct_fp: str,
    pre_dhash: str,
    post_hierarchy_text: str,
    post_struct_fp: str,
    post_dhash: str,
) -> bool:
    if _simple_verify(pre_hierarchy_text, post_hierarchy_text):
        return True
    if pre_struct_fp and post_struct_fp and pre_struct_fp == post_struct_fp:
        if pre_dhash and post_dhash:
            return _hamming_distance_hex(pre_dhash, post_dhash) <= 3
        return True
    if pre_dhash and post_dhash:
        return _hamming_distance_hex(pre_dhash, post_dhash) <= 3
    return False


def _classify_progress(
    *,
    task: str,
    action_record: Dict[str, object],
    pre_hierarchy_text: str,
    post_hierarchy_text: str,
    pre_struct_fp: str,
    post_struct_fp: str,
    pre_dhash: str,
    post_dhash: str,
    adaptive_threshold: float,
) -> Tuple[str, Dict[str, object]]:
    similarity = _sequence_similarity(pre_hierarchy_text, post_hierarchy_text)
    stable_text_changed = _stable_text_fingerprint(pre_hierarchy_text) != _stable_text_fingerprint(post_hierarchy_text)
    struct_changed = pre_struct_fp != post_struct_fp
    dhash_distance = _hamming_distance_hex(pre_dhash, post_dhash) if pre_dhash and post_dhash else 0
    pre_markers = _extract_ui_state_markers(pre_hierarchy_text)
    post_markers = _extract_ui_state_markers(post_hierarchy_text)
    selected_changed = pre_markers["selected"] != post_markers["selected"]
    focused_changed = pre_markers["focused"] != post_markers["focused"]
    editable_text_changed = pre_markers["editable_texts"] != post_markers["editable_texts"]
    hint_changed = pre_markers["hint_texts"] != post_markers["hint_texts"]
    action_type = str(action_record.get("type", "")).lower()

    details = {
        "similarity": similarity,
        "adaptive_threshold": adaptive_threshold,
        "stable_text_changed": stable_text_changed,
        "struct_changed": struct_changed,
        "dhash_distance": dhash_distance,
        "selected_changed": selected_changed,
        "focused_changed": focused_changed,
        "editable_text_changed": editable_text_changed,
        "hint_changed": hint_changed,
    }

    if similarity < adaptive_threshold or stable_text_changed or (struct_changed and dhash_distance > 3):
        return _PROGRESS_STRONG, details

    weak_signal = selected_changed or focused_changed or editable_text_changed or hint_changed
    if weak_signal and _is_weak_progress_task(task, action_type):
        return _PROGRESS_WEAK, details

    return _PROGRESS_NONE, details


def _save_partial_trace(
    *,
    reason: str,
    app_name: str,
    device,
    device_type: str,
    partial_paths_dir: str,
    partial_path_counter: List[int],
    steps_dir: str,
    actions: List[Dict[str, object]],
    reacts: List[Dict[str, object]],
    terminal_depth: int,
    runtime,
) -> None:
    if len(actions) < _PARTIAL_PATH_MIN_LENGTH:
        return

    partial_path_counter[0] += 1
    trace_id = partial_path_counter[0]
    trace_dir = os.path.join(partial_paths_dir, f"path_{trace_id:04d}")
    metrics = runtime.metrics
    step_indices = [item.get("action_index") for item in actions if item.get("action_index")]
    copy_step_artifacts_to_path(steps_dir, trace_dir, step_indices, metrics=metrics)

    try:
        get_screenshot(device, device_type)
        save_named_raw_screenshot(trace_dir, "terminal.jpg", device_type, metrics=metrics)
        save_named_hierarchy(
            device,
            device_type,
            trace_dir,
            "terminal",
            async_enabled=runtime.features.async_artifact_io,
            metrics=metrics,
        )
    except Exception as exc:
        logging.warning("Failed to save partial terminal artifacts for %s: %s", trace_dir, exc)

    if runtime.features.async_artifact_io:
        submit_artifact_task(
            _persist_outputs_safe,
            trace_dir,
            app_name,
            list(actions),
            list(reacts),
            metrics,
        )
    else:
        _persist_outputs_safe(
            trace_dir,
            app_name,
            list(actions),
            list(reacts),
            metrics,
        )

    write_trace_meta(
        trace_dir,
        {
            "status": "partial",
            "saved_reason": reason,
            "path_length": len(actions),
            "terminal_depth": terminal_depth,
        },
        metrics=metrics,
    )
    if hasattr(metrics, "record_saved_trace"):
        metrics.record_saved_trace("partial")


def explore_dfs(
    *,
    app_name: str,
    depth_limit: int,
    breadth: int,
    current_depth: int,
    decider_client: OpenAI,
    decider_model: str,
    explorer_client: OpenAI,
    explorer_model: str,
    device,
    device_type: str,
    use_qwen3: bool,
    allow_hierarchy_text_decider: bool,
    data_dir: str,
    actions: List[Dict[str, object]],
    reacts: List[Dict[str, object]],
    step_counter: List[int],
    path_counter: List[int],
    partial_path_counter: List[int],
    page_counter: List[int],
    steps_dir: str,
    paths_dir: str,
    partial_paths_dir: str,
    enable_ui_semantic_collect: bool,
    ui_pages_dir: str,
    page_registry: Dict[str, Dict[str, object]],
    collect_queue: Optional["queue.Queue[Dict[str, object]]"],
    queue_lock: Optional[threading.Lock],
    index_lock: Optional[threading.Lock],
    index_path: Optional[str],
    ui_collect_async: bool,
    ui_collect_queue_size: int,
    runtime,
    path_actions: Optional[List[Dict[str, object]]] = None,
    path_reacts: Optional[List[Dict[str, object]]] = None,
    visited_tasks: Optional[Dict[str, set]] = None,
    explorer_cache: Optional["ExplorerCache"] = None,
    screen_cache: Optional["ScreenStateCache"] = None,
    popup_dismiss_max_attempts: int = 2,
) -> None:
    """DFS exploration that executes ranked candidates and backtracks afterward."""
    if current_depth >= depth_limit:
        return

    metrics = runtime.metrics
    capture_started = metrics.now()
    screenshot_b64, hierarchy_text = _capture_screen(device, device_type, screen_cache)
    capture_finished = metrics.now()

    do_collect = (
        enable_ui_semantic_collect
        and collect_queue is not None
        and queue_lock is not None
        and index_lock is not None
        and bool(index_path)
    )

    if do_collect:
        assert collect_queue is not None and queue_lock is not None and index_lock is not None and index_path
        enqueue_ui_collect_if_new(
            app_name=app_name,
            device_type=device_type,
            current_depth=current_depth,
            hierarchy_text=hierarchy_text,
            ui_pages_dir=ui_pages_dir,
            page_registry=page_registry,
            page_counter=page_counter,
            collect_queue=collect_queue,
            queue_lock=queue_lock,
            index_lock=index_lock,
            index_path=index_path,
            ui_collect_queue_size=ui_collect_queue_size,
        )

    if runtime.features.popup_auto_dismiss:
        for rule_attempt in range(popup_dismiss_max_attempts):
            dismiss_bounds = _find_dismissible_element(hierarchy_text)
            if dismiss_bounds is None:
                break
            x = (dismiss_bounds[0] + dismiss_bounds[2]) // 2
            y = (dismiss_bounds[1] + dismiss_bounds[3]) // 2
            device.click(x, y)
            time.sleep(settings.DEVICE_WAIT_TIME)
            capture_started = metrics.now()
            screenshot_b64, hierarchy_text = _capture_screen(device, device_type, screen_cache)
            capture_finished = metrics.now()
            logging.info("\033[93m[RuleDismiss] clicked (%d,%d) attempt %d\033[0m", x, y, rule_attempt + 1)

    action_history = list(path_actions) if path_actions else list(actions)
    if visited_tasks is None:
        visited_tasks = {}
    no_progress_task_counts: Dict[str, int] = {}
    suppressed_no_progress_tasks: set[str] = set()

    current_page_fp, struct_fp, _ = compute_fingerprints(
        hierarchy_text,
        concurrent_mode=runtime.features.concurrent_fingerprint,
    )
    if runtime.features.already_explored_filter:
        already_explored_list = list(visited_tasks.get(current_page_fp, set()))
    else:
        already_explored_list = []

    cached_candidates = None
    if explorer_cache is not None and runtime.features.explorer_cache:
        cached_candidates = explorer_cache.get(struct_fp, current_depth, breadth, already_explored_list)

    popup_info: Optional[Dict[str, object]] = None
    if cached_candidates is not None:
        if runtime.features.already_explored_filter:
            candidates = [
                candidate
                for candidate in cached_candidates
                if candidate.get("single_step_task", "") not in visited_tasks.get(current_page_fp, set())
            ]
        else:
            candidates = list(cached_candidates)
        logging.info(f"Depth={current_depth}, Explorer cache hit, {len(candidates)} candidates after filter")
    else:
        try:
            candidates, popup_info = call_explorer_model(
                explorer_client,
                explorer_model,
                screenshot_b64,
                hierarchy_text,
                current_depth,
                breadth,
                action_history,
                already_explored=already_explored_list,
                metrics=metrics,
                trace_meta=_build_explorer_trace(
                    metrics,
                    current_depth=current_depth,
                    stage="initial",
                    screenshot_started=capture_started,
                    screenshot_finished=capture_finished,
                ),
            )
        except Exception as exc:
            reason = "explorer_empty" if "No valid candidates returned" in str(exc) else "branch_failure"
            logging.error("Explorer failed at depth %d: %s", current_depth, exc)
            _save_partial_trace(
                reason=reason,
                app_name=app_name,
                device=device,
                device_type=device_type,
                partial_paths_dir=partial_paths_dir,
                partial_path_counter=partial_path_counter,
                steps_dir=steps_dir,
                actions=list(path_actions) if path_actions else [],
                reacts=list(path_reacts) if path_reacts else [],
                terminal_depth=current_depth,
                runtime=runtime,
            )
            return
        if explorer_cache is not None and runtime.features.explorer_cache:
            explorer_cache.put(struct_fp, current_depth, breadth, already_explored_list, candidates)

    if suppressed_no_progress_tasks:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("single_step_task", "") not in suppressed_no_progress_tasks
        ]

    if runtime.features.popup_auto_dismiss:
        for popup_attempt in range(popup_dismiss_max_attempts):
            if not (popup_info and popup_info.get("detected")):
                break
            close_pt = popup_info.get("close_point")
            if close_pt:
                size = _get_current_screen_size(device_type)
                if size:
                    img_w, img_h = size
                    abs_pt = (
                        convert_qwen3_coordinates_to_absolute(close_pt, img_w, img_h, is_bbox=False)
                        if use_qwen3
                        else close_pt
                    )
                    device.click(int(abs_pt[0]), int(abs_pt[1]))
                    logging.info(
                        f"\033[93m[PopupDismiss] clicked close at {abs_pt} (attempt {popup_attempt + 1})\033[0m"
                    )
                else:
                    navigate_back(device, device_type)
                    logging.info(
                        f"\033[93m[PopupDismiss] no screen size, pressed back (attempt {popup_attempt + 1})\033[0m"
                    )
            else:
                h1 = hierarchy_text
                time.sleep(1.0)
                if screen_cache is not None:
                    _, h2 = screen_cache.capture(device, device_type, force=True)
                else:
                    h2 = get_hierarchy_text(device)
                remaining = _detect_countdown(h1, h2)
                if remaining is not None:
                    logging.info(
                        "\033[93m[PopupDismiss] Countdown ad detected, waiting %ds... (attempt %d)\033[0m",
                        remaining,
                        popup_attempt + 1,
                    )
                    time.sleep(remaining + 0.5)
                else:
                    navigate_back(device, device_type)
                    logging.info(
                        f"\033[93m[PopupDismiss] no close_point, no countdown -> pressed back (attempt {popup_attempt + 1})\033[0m"
                    )
            time.sleep(settings.DEVICE_WAIT_TIME)
            capture_started = metrics.now()
            screenshot_b64, hierarchy_text = _capture_screen(device, device_type, screen_cache)
            capture_finished = metrics.now()
            try:
                candidates, popup_info = call_explorer_model(
                    explorer_client,
                    explorer_model,
                    screenshot_b64,
                    hierarchy_text,
                    current_depth,
                    breadth,
                    action_history,
                    already_explored=already_explored_list,
                    metrics=metrics,
                    trace_meta=_build_explorer_trace(
                        metrics,
                        current_depth=current_depth,
                        stage="popup_redetect",
                        screenshot_started=capture_started,
                        screenshot_finished=capture_finished,
                    ),
                )
            except Exception as exc:
                reason = "explorer_empty" if "No valid candidates returned" in str(exc) else "branch_failure"
                logging.error("Explorer popup redetect failed at depth %d: %s", current_depth, exc)
                _save_partial_trace(
                    reason=reason,
                    app_name=app_name,
                    device=device,
                    device_type=device_type,
                    partial_paths_dir=partial_paths_dir,
                    partial_path_counter=partial_path_counter,
                    steps_dir=steps_dir,
                    actions=list(path_actions) if path_actions else [],
                    reacts=list(path_reacts) if path_reacts else [],
                    terminal_depth=current_depth,
                    runtime=runtime,
                )
                return

    if runtime.features.candidate_dedup:
        candidates = _deduplicate_candidates(candidates, already_explored=already_explored_list)
    if suppressed_no_progress_tasks:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("single_step_task", "") not in suppressed_no_progress_tasks
        ]
    if not candidates:
        _save_partial_trace(
            reason="explorer_empty",
            app_name=app_name,
            device=device,
            device_type=device_type,
            partial_paths_dir=partial_paths_dir,
            partial_path_counter=partial_path_counter,
            steps_dir=steps_dir,
            actions=list(path_actions) if path_actions else [],
            reacts=list(path_reacts) if path_reacts else [],
            terminal_depth=current_depth,
            runtime=runtime,
        )
        return
    base_hierarchy_text = hierarchy_text

    logging.info(f"Depth={current_depth}, got {len(candidates)} candidates")
    for candidate in candidates:
        logging.info(
            "\033[96m[Depth %d] candidate rank=%s task=%s reason=%s\033[0m",
            current_depth,
            candidate.get("rank"),
            candidate.get("single_step_task"),
            candidate.get("reason"),
        )

    cand_idx = 0
    while cand_idx < len(candidates):
        if cand_idx > 0:
            current_hierarchy_text = get_hierarchy_text(device)
            similarity = difflib.SequenceMatcher(
                None,
                _normalize_hierarchy_text(base_hierarchy_text),
                _normalize_hierarchy_text(current_hierarchy_text),
            ).ratio()
            page_change_threshold = _compute_adaptive_similarity_threshold(base_hierarchy_text)
            if similarity < page_change_threshold:
                remaining = max(breadth - cand_idx, 0)
                if remaining == 0:
                    break
                logging.info(
                    "\033[93mPage changed (similarity=%.3f < threshold=%.3f). Regenerating %d candidates.\033[0m",
                    similarity,
                    page_change_threshold,
                    remaining,
                )
                capture_started = metrics.now()
                screenshot_b64 = get_screenshot(device, device_type)
                capture_finished = metrics.now()
                base_hierarchy_text = current_hierarchy_text
                if do_collect:
                    assert collect_queue is not None and queue_lock is not None and index_lock is not None and index_path
                    enqueue_ui_collect_if_new(
                        app_name=app_name,
                        device_type=device_type,
                        current_depth=current_depth,
                        hierarchy_text=current_hierarchy_text,
                        ui_pages_dir=ui_pages_dir,
                        page_registry=page_registry,
                        page_counter=page_counter,
                        collect_queue=collect_queue,
                        queue_lock=queue_lock,
                        index_lock=index_lock,
                        index_path=index_path,
                        ui_collect_queue_size=ui_collect_queue_size,
                    )
                new_page_fp = _hierarchy_fingerprint(base_hierarchy_text)
                if runtime.features.already_explored_filter:
                    new_already_explored = list(visited_tasks.get(new_page_fp, set()))
                else:
                    new_already_explored = []
                no_progress_task_counts = {}
                suppressed_no_progress_tasks = set()
                try:
                    new_candidates, _ = call_explorer_model(
                        explorer_client,
                        explorer_model,
                        screenshot_b64,
                        base_hierarchy_text,
                        current_depth,
                        remaining,
                        list(path_actions) if path_actions else list(actions),
                        already_explored=new_already_explored,
                        metrics=metrics,
                        trace_meta=_build_explorer_trace(
                            metrics,
                            current_depth=current_depth,
                            stage="regenerate",
                            screenshot_started=capture_started,
                            screenshot_finished=capture_finished,
                        ),
                    )
                except Exception as exc:
                    reason = "explorer_empty" if "No valid candidates returned" in str(exc) else "branch_failure"
                    logging.error("Explorer regenerate failed at depth %d: %s", current_depth, exc)
                    _save_partial_trace(
                        reason=reason,
                        app_name=app_name,
                        device=device,
                        device_type=device_type,
                        partial_paths_dir=partial_paths_dir,
                        partial_path_counter=partial_path_counter,
                        steps_dir=steps_dir,
                        actions=list(path_actions) if path_actions else list(actions),
                        reacts=list(path_reacts) if path_reacts else list(reacts),
                        terminal_depth=current_depth,
                        runtime=runtime,
                    )
                    break
                if runtime.features.candidate_dedup:
                    new_candidates = _deduplicate_candidates(new_candidates, already_explored=new_already_explored)
                if suppressed_no_progress_tasks:
                    new_candidates = [
                        item
                        for item in new_candidates
                        if item.get("single_step_task", "") not in suppressed_no_progress_tasks
                    ]
                candidates = candidates[:cand_idx] + new_candidates

        candidate = candidates[cand_idx]
        task = candidate["single_step_task"]
        step_counter[0] += 1
        step_idx = step_counter[0]
        step_output_dir = os.path.join(steps_dir, f"step_{step_idx:04d}")
        os.makedirs(step_output_dir, exist_ok=True)

        current_path_actions = list(path_actions) if path_actions else []
        current_path_reacts = list(path_reacts) if path_reacts else []
        decider_history = [
            str(item.get("source_task", "")).strip()
            for item in current_path_actions[-10:]
            if str(item.get("source_task", "")).strip()
        ]

        logging.info(
            "\033[96m[Depth %d] execute rank=%s task=%s reason=%s\033[0m",
            current_depth,
            candidate.get("rank"),
            task,
            candidate.get("reason"),
        )

        action_record = None
        backtrack_action_record = None
        progress_state = _PROGRESS_NONE
        backtrack_duration_override: Optional[float] = None
        skip_heavy_backtrack = False
        current_path_actions_for_recovery = current_path_actions
        pre_hierarchy_text = get_hierarchy_text(device)
        get_screenshot(device, device_type)
        pre_screenshot_path = _current_screenshot_path(device_type)
        _, pre_struct_fp, pre_dhash = compute_fingerprints(
            pre_hierarchy_text,
            screenshot_path=pre_screenshot_path,
            concurrent_mode=runtime.features.concurrent_fingerprint,
        )
        step_started = time.perf_counter()
        try:
            step_result = execute_decider_one_step(
                decider_client=decider_client,
                decider_model=decider_model,
                device=device,
                device_type=device_type,
                app_name=app_name,
                step_task=task,
                use_qwen3=use_qwen3,
                allow_hierarchy_text_decider=allow_hierarchy_text_decider,
                output_dir=step_output_dir,
                step_index=step_idx,
                runtime=runtime,
                history=decider_history,
            )

            action_record = step_result["action_record"]
            backtrack_action_record = action_record
            react_item = step_result["react_item"]
            post_hierarchy_text = step_result.get("post_hierarchy_text", "")
            if do_collect:
                assert collect_queue is not None and queue_lock is not None and index_lock is not None and index_path
                _ = get_screenshot(device, device_type)
                enqueue_ui_collect_if_new(
                    app_name=app_name,
                    device_type=device_type,
                    current_depth=current_depth + 1,
                    hierarchy_text=post_hierarchy_text,
                    ui_pages_dir=ui_pages_dir,
                    page_registry=page_registry,
                    page_counter=page_counter,
                    collect_queue=collect_queue,
                    queue_lock=queue_lock,
                    index_lock=index_lock,
                    index_path=index_path,
                    ui_collect_queue_size=ui_collect_queue_size,
                )

            actions.append(action_record)
            reacts.append(react_item)
            current_path_actions.append(action_record)
            current_path_reacts.append(react_item)
            current_path_actions_for_recovery = current_path_actions

            if screen_cache is not None:
                screen_cache.invalidate()

            if runtime.features.async_artifact_io:
                submit_artifact_task(
                    _persist_step_output_safe,
                    step_output_dir,
                    app_name,
                    dict(action_record),
                    dict(react_item),
                    metrics,
                )
            else:
                _persist_step_output_safe(
                    step_output_dir,
                    app_name,
                    dict(action_record),
                    dict(react_item),
                    metrics,
                )
            metrics.record_step_duration(time.perf_counter() - step_started)

            get_screenshot(device, device_type)
            post_screenshot_path = _current_screenshot_path(device_type)
            _, post_struct_fp, post_dhash = compute_fingerprints(
                post_hierarchy_text,
                screenshot_path=post_screenshot_path,
                concurrent_mode=runtime.features.concurrent_fingerprint,
            )
            progress_threshold = _compute_adaptive_similarity_threshold(pre_hierarchy_text)
            progress_state, progress_details = _classify_progress(
                task=task,
                action_record=action_record,
                pre_hierarchy_text=pre_hierarchy_text,
                post_hierarchy_text=post_hierarchy_text,
                pre_struct_fp=pre_struct_fp,
                post_struct_fp=post_struct_fp,
                pre_dhash=pre_dhash,
                post_dhash=post_dhash,
                adaptive_threshold=progress_threshold,
            )

            if progress_state != _PROGRESS_NONE and runtime.features.already_explored_filter:
                if current_page_fp not in visited_tasks:
                    visited_tasks[current_page_fp] = set()
                visited_tasks[current_page_fp].add(task)

            should_recurse = progress_state == _PROGRESS_STRONG or (
                progress_state == _PROGRESS_WEAK and _is_weak_progress_task(task, str(action_record.get("type", "")).lower())
            )

            if progress_state == _PROGRESS_NONE:
                no_progress_task_counts[task] = no_progress_task_counts.get(task, 0) + 1
                if no_progress_task_counts[task] >= 2:
                    suppressed_no_progress_tasks.add(task)
                logging.warning(
                    "\033[93m[Depth %d] No effective page transition after task='%s' "
                    "(similarity=%.3f >= threshold=%.3f, stable_changed=%s, struct_changed=%s, dhash=%s). "
                    "Stopping branch expansion.\033[0m",
                    current_depth,
                    task,
                    progress_details["similarity"],
                    progress_details["adaptive_threshold"],
                    progress_details["stable_text_changed"],
                    progress_details["struct_changed"],
                    progress_details["dhash_distance"],
                )
                _save_partial_trace(
                    reason="no_progress",
                    app_name=app_name,
                    device=device,
                    device_type=device_type,
                    partial_paths_dir=partial_paths_dir,
                    partial_path_counter=partial_path_counter,
                    steps_dir=steps_dir,
                    actions=current_path_actions,
                    reacts=current_path_reacts,
                    terminal_depth=current_depth + 1,
                    runtime=runtime,
                )

                light_recovery_started = time.perf_counter()
                action_type = str(action_record.get("type", "")).lower()
                if action_type in {"click_input", "input"}:
                    navigate_back(device, device_type)
                    light_back_hierarchy = get_hierarchy_text(device)
                    get_screenshot(device, device_type)
                    light_back_screenshot_path = _current_screenshot_path(device_type)
                    _, light_back_struct_fp, light_back_dhash = compute_fingerprints(
                        light_back_hierarchy,
                        screenshot_path=light_back_screenshot_path,
                        concurrent_mode=runtime.features.concurrent_fingerprint,
                    )
                    skip_heavy_backtrack = _states_equivalent(
                        pre_hierarchy_text,
                        pre_struct_fp,
                        pre_dhash,
                        light_back_hierarchy,
                        light_back_struct_fp,
                        light_back_dhash,
                    )
                    if not skip_heavy_backtrack:
                        backtrack_action_record = {"type": "click", "source_task": task}
                else:
                    skip_heavy_backtrack = _states_equivalent(
                        pre_hierarchy_text,
                        pre_struct_fp,
                        pre_dhash,
                        post_hierarchy_text,
                        post_struct_fp,
                        post_dhash,
                    )
                    if not skip_heavy_backtrack:
                        navigate_back(device, device_type)
                        light_back_hierarchy = get_hierarchy_text(device)
                        get_screenshot(device, device_type)
                        light_back_screenshot_path = _current_screenshot_path(device_type)
                        _, light_back_struct_fp, light_back_dhash = compute_fingerprints(
                            light_back_hierarchy,
                            screenshot_path=light_back_screenshot_path,
                            concurrent_mode=runtime.features.concurrent_fingerprint,
                        )
                        skip_heavy_backtrack = _states_equivalent(
                            pre_hierarchy_text,
                            pre_struct_fp,
                            pre_dhash,
                            light_back_hierarchy,
                            light_back_struct_fp,
                            light_back_dhash,
                        )
                backtrack_duration_override = time.perf_counter() - light_recovery_started

            if progress_state == _PROGRESS_STRONG and current_depth + 1 >= depth_limit:
                path_counter[0] += 1
                path_id = path_counter[0]
                path_output_dir = os.path.join(paths_dir, f"path_{path_id:04d}")
                step_indices = [item.get("action_index") for item in current_path_actions if item.get("action_index")]
                copy_step_artifacts_to_path(steps_dir, path_output_dir, step_indices, metrics=metrics)
                path_task_description = _compute_task_description(actions=current_path_actions, app_name=app_name)
                full_path_actions, full_path_reacts = append_done_to_path(
                    current_path_actions,
                    current_path_reacts,
                    path_task_description,
                )
                done_index = len(full_path_actions)
                try:
                    get_screenshot(device, device_type)
                    save_raw_screenshot(path_output_dir, done_index, device_type, metrics=metrics)
                    save_hierarchy(
                        device,
                        device_type,
                        path_output_dir,
                        done_index,
                        async_enabled=runtime.features.async_artifact_io,
                        metrics=metrics,
                    )
                except Exception as e:
                    logging.warning(f"Failed to save done artifacts for path {path_id:04d}: {e}")
                if runtime.features.async_artifact_io:
                    submit_artifact_task(
                        _persist_outputs_safe,
                        path_output_dir,
                        app_name,
                        list(full_path_actions),
                        list(full_path_reacts),
                        metrics,
                    )
                else:
                    _persist_outputs_safe(
                        path_output_dir,
                        app_name,
                        list(full_path_actions),
                        list(full_path_reacts),
                        metrics,
                    )
                if hasattr(metrics, "record_saved_trace"):
                    metrics.record_saved_trace("complete")
            elif should_recurse:
                explore_dfs(
                    app_name=app_name,
                    depth_limit=depth_limit,
                    breadth=breadth,
                    current_depth=current_depth + 1,
                    decider_client=decider_client,
                    decider_model=decider_model,
                    explorer_client=explorer_client,
                    explorer_model=explorer_model,
                    device=device,
                    device_type=device_type,
                    use_qwen3=use_qwen3,
                    allow_hierarchy_text_decider=allow_hierarchy_text_decider,
                    data_dir=data_dir,
                    actions=actions,
                    reacts=reacts,
                    step_counter=step_counter,
                    path_counter=path_counter,
                    partial_path_counter=partial_path_counter,
                    page_counter=page_counter,
                    steps_dir=steps_dir,
                    paths_dir=paths_dir,
                    partial_paths_dir=partial_paths_dir,
                    enable_ui_semantic_collect=enable_ui_semantic_collect,
                    ui_pages_dir=ui_pages_dir,
                    page_registry=page_registry,
                    collect_queue=collect_queue,
                    queue_lock=queue_lock,
                    index_lock=index_lock,
                    index_path=index_path,
                    ui_collect_async=ui_collect_async,
                    ui_collect_queue_size=ui_collect_queue_size,
                    runtime=runtime,
                    path_actions=current_path_actions,
                    path_reacts=current_path_reacts,
                    visited_tasks=visited_tasks,
                    explorer_cache=explorer_cache,
                    screen_cache=screen_cache,
                    popup_dismiss_max_attempts=popup_dismiss_max_attempts,
                )

        except WaitActionSkip:
            step_counter[0] -= 1
            try:
                import shutil

                shutil.rmtree(step_output_dir)
            except Exception:
                pass
            logging.info(
                "\033[93m[Depth %d] Wait action - candidate '%s' skipped, not recorded.\033[0m",
                current_depth,
                task,
            )
            cand_idx += 1
            continue
        except Exception as e:
            logging.error(f"Failed to execute candidate at depth {current_depth}: {e}")
            _save_partial_trace(
                reason="branch_failure",
                app_name=app_name,
                device=device,
                device_type=device_type,
                partial_paths_dir=partial_paths_dir,
                partial_path_counter=partial_path_counter,
                steps_dir=steps_dir,
                actions=current_path_actions_for_recovery,
                reacts=current_path_reacts,
                terminal_depth=current_depth,
                runtime=runtime,
            )

        if action_record is None:
            cand_idx += 1
            continue

        if skip_heavy_backtrack:
            metrics.record_backtrack(
                verified_without_recovery=True,
                recovery_attempts=0,
                recovery_succeeded=False,
                duration_sec=backtrack_duration_override or 0.0,
            )
            cand_idx += 1
            continue

        backtrack_started = time.perf_counter()
        app_was_restarted = False
        perform_backtrack_action(device, device_type, backtrack_action_record)

        post_back_hierarchy = get_hierarchy_text(device)
        if not _is_app_in_foreground(device, device_type, app_name, post_back_hierarchy):
            logging.warning("\033[91mBacktrack left app unexpectedly. Restarting app: %s\033[0m", app_name)
            device.start_app(app_name)
            time.sleep(settings.DEVICE_WAIT_TIME * 2)
            post_back_hierarchy = get_hierarchy_text(device)
            app_was_restarted = True

        post_back_screenshot_path = _current_screenshot_path(device_type)
        get_screenshot(device, device_type)

        if app_was_restarted:
            verified = _is_app_in_foreground(device, device_type, app_name, post_back_hierarchy)
            if verified:
                logging.info("\033[92mBacktrack verified via app restart (dynamic content expected to differ).\033[0m")
            else:
                logging.warning("\033[91mApp did not return to foreground after restart.\033[0m")
        elif runtime.features.triple_verify:
            verified = _triple_verify(
                pre_hierarchy_text,
                pre_struct_fp,
                pre_dhash,
                post_back_hierarchy,
                post_back_screenshot_path,
                concurrent_mode=runtime.features.concurrent_fingerprint,
            )
        else:
            verified = _simple_verify(pre_hierarchy_text, post_back_hierarchy)

        recovery_attempts = 0
        recovered = False
        if not verified and runtime.features.replay_recovery:
            logging.warning("\033[93mBacktrack verification failed. Attempting full path replay.\033[0m")
            for attempt in range(2):
                recovery_attempts += 1
                device.start_app(app_name)
                time.sleep(settings.DEVICE_WAIT_TIME * 2)
                replay_ok = True
                for replay_act in current_path_actions_for_recovery[:-1]:
                    if not replay_action_record(device, replay_act):
                        replay_ok = False
                        break
                    time.sleep(settings.DEVICE_WAIT_TIME)
                if not replay_ok:
                    logging.warning("\033[93mPath replay action failed on attempt %d.\033[0m", attempt + 1)
                    continue
                replay_hierarchy = get_hierarchy_text(device)
                get_screenshot(device, device_type)
                if not current_path_actions_for_recovery[:-1]:
                    replay_recovered = _is_app_in_foreground(device, device_type, app_name, replay_hierarchy)
                    if replay_recovered:
                        recovered = True
                        logging.info("\033[92mFull path replay recovery succeeded (first step, app in foreground).\033[0m")
                        break
                    continue
                if runtime.features.triple_verify:
                    replay_verified = _triple_verify(
                        pre_hierarchy_text,
                        pre_struct_fp,
                        pre_dhash,
                        replay_hierarchy,
                        post_back_screenshot_path,
                        concurrent_mode=runtime.features.concurrent_fingerprint,
                    )
                else:
                    replay_verified = _simple_verify(pre_hierarchy_text, replay_hierarchy)
                if replay_verified:
                    recovered = True
                    logging.info("\033[92mFull path replay recovery succeeded on attempt %d.\033[0m", attempt + 1)
                    break

        metrics.record_backtrack(
            verified_without_recovery=verified,
            recovery_attempts=recovery_attempts,
            recovery_succeeded=recovered,
            duration_sec=time.perf_counter() - backtrack_started,
        )

        if not verified and not recovered:
            logging.error(
                "\033[91mBacktrack recovery failed after verification. Skipping remaining candidates at this depth.\033[0m"
            )
            break

        cand_idx += 1


def init_decider_client(service_ip: str, decider_port: int, base_url: str = "", api_key: str = "") -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key or "0", base_url=base_url)
    return OpenAI(api_key="0", base_url=f"http://{service_ip}:{decider_port}/v1")


def init_explorer_client(base_url: str, api_key: str) -> OpenAI:
    logging.info(f"Initializing explorer client with base_url={base_url}")
    logging.info(f"API Key is set: {bool(api_key)}")
    return OpenAI(api_key=api_key, base_url=base_url)


__all__ = ["explore_dfs", "init_decider_client", "init_explorer_client"]
