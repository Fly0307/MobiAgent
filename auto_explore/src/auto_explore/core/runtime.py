from __future__ import annotations

import json
import math
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


FEATURE_FLAG_DEFAULTS: Dict[str, bool] = {
    "triple_verify": True,
    "replay_recovery": True,
    "candidate_dedup": True,
    "already_explored_filter": True,
    "explorer_cache": True,
    "screen_cache": True,
    "popup_auto_dismiss": True,
    "async_artifact_io": True,
    "concurrent_fingerprint": True,
    "hierarchy_text_decider": True,
}


def _flag_cli_name(flag_name: str) -> str:
    return f"--enable_{flag_name}"


def add_feature_flag_args(parser) -> None:
    for flag_name, default_enabled in FEATURE_FLAG_DEFAULTS.items():
        parser.add_argument(
            _flag_cli_name(flag_name),
            choices=["on", "off"],
            default="on" if default_enabled else "off",
            help=f"Enable runtime optimization: {flag_name}",
        )


@dataclass(frozen=True)
class FeatureFlags:
    triple_verify: bool = True
    replay_recovery: bool = True
    candidate_dedup: bool = True
    already_explored_filter: bool = True
    explorer_cache: bool = True
    screen_cache: bool = True
    popup_auto_dismiss: bool = True
    async_artifact_io: bool = True
    concurrent_fingerprint: bool = True
    hierarchy_text_decider: bool = True

    @classmethod
    def from_namespace(cls, args) -> "FeatureFlags":
        kwargs = {}
        for flag_name, default_enabled in FEATURE_FLAG_DEFAULTS.items():
            raw_value = getattr(args, f"enable_{flag_name}", None)
            if raw_value is None:
                kwargs[flag_name] = default_enabled
            else:
                kwargs[flag_name] = raw_value == "on"
        return cls(**kwargs)

    def to_cli_args(self) -> List[str]:
        args: List[str] = []
        for flag_name, enabled in asdict(self).items():
            args.extend([_flag_cli_name(flag_name), "on" if enabled else "off"])
        return args

    def with_overrides(self, **overrides: bool) -> "FeatureFlags":
        payload = asdict(self)
        payload.update(overrides)
        return FeatureFlags(**payload)


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return float(ordered[index])


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


@dataclass
class RuntimeContext:
    features: FeatureFlags
    metrics: "MetricsCollector"


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_start = time.perf_counter()
        self._run_end: Optional[float] = None
        self._timing_traces: List[Dict[str, Any]] = []
        self._step_durations: List[float] = []
        self._backtrack_durations: List[float] = []
        self._explorer_call_count = 0
        self._explorer_time_sec_total = 0.0
        self._explorer_cache_requests = 0
        self._explorer_cache_hits = 0
        self._screen_cache_requests = 0
        self._screen_cache_hits = 0
        self._decider_call_count = 0
        self._decider_time_sec_total = 0.0
        self._backtrack_count = 0
        self._backtrack_verify_successes = 0
        self._backtrack_recovery_sequence_count = 0
        self._backtrack_recovery_attempt_count = 0
        self._backtrack_recovery_success_count = 0
        self._artifact_io_time_sec_total = 0.0
        self._partial_path_count = 0
        self._complete_path_count = 0

    def finish_run(self) -> None:
        with self._lock:
            self._run_end = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter()

    def relative_time(self, timestamp: Optional[float] = None) -> float:
        raw = self.now() if timestamp is None else timestamp
        return float(raw - self._run_start)

    def record_timing_trace(self, trace: Mapping[str, Any]) -> None:
        with self._lock:
            self._timing_traces.append(dict(trace))

    def record_step_duration(self, duration_sec: float) -> None:
        with self._lock:
            self._step_durations.append(float(duration_sec))

    def record_explorer_cache_lookup(self, hit: bool) -> None:
        with self._lock:
            self._explorer_cache_requests += 1
            if hit:
                self._explorer_cache_hits += 1

    def record_screen_cache_lookup(self, hit: bool) -> None:
        with self._lock:
            self._screen_cache_requests += 1
            if hit:
                self._screen_cache_hits += 1

    def record_explorer_call(self, duration_sec: float) -> None:
        with self._lock:
            self._explorer_call_count += 1
            self._explorer_time_sec_total += float(duration_sec)

    def record_decider_call(self, duration_sec: float) -> None:
        with self._lock:
            self._decider_call_count += 1
            self._decider_time_sec_total += float(duration_sec)

    def record_backtrack(
        self,
        *,
        verified_without_recovery: bool,
        recovery_attempts: int,
        recovery_succeeded: bool,
        duration_sec: float,
    ) -> None:
        with self._lock:
            self._backtrack_count += 1
            self._backtrack_durations.append(float(duration_sec))
            if verified_without_recovery:
                self._backtrack_verify_successes += 1
            if recovery_attempts > 0:
                self._backtrack_recovery_sequence_count += 1
                self._backtrack_recovery_attempt_count += int(recovery_attempts)
                if recovery_succeeded:
                    self._backtrack_recovery_success_count += 1

    def record_artifact_io(self, duration_sec: float) -> None:
        with self._lock:
            self._artifact_io_time_sec_total += float(duration_sec)

    def record_saved_trace(self, status: str) -> None:
        with self._lock:
            if status == "partial":
                self._partial_path_count += 1
            elif status == "complete":
                self._complete_path_count += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            run_end = self._run_end if self._run_end is not None else time.perf_counter()
            step_durations = list(self._step_durations)
            backtrack_durations = list(self._backtrack_durations)
            explorer_call_count = self._explorer_call_count
            explorer_time_sec_total = self._explorer_time_sec_total
            explorer_cache_requests = self._explorer_cache_requests
            explorer_cache_hits = self._explorer_cache_hits
            screen_cache_requests = self._screen_cache_requests
            screen_cache_hits = self._screen_cache_hits
            decider_call_count = self._decider_call_count
            decider_time_sec_total = self._decider_time_sec_total
            backtrack_count = self._backtrack_count
            backtrack_verify_successes = self._backtrack_verify_successes
            backtrack_recovery_sequence_count = self._backtrack_recovery_sequence_count
            backtrack_recovery_attempt_count = self._backtrack_recovery_attempt_count
            backtrack_recovery_success_count = self._backtrack_recovery_success_count
            artifact_io_time_sec_total = self._artifact_io_time_sec_total
            partial_path_count = self._partial_path_count
            complete_path_count = self._complete_path_count
            timing_traces = list(self._timing_traces)

        final_backtrack_successes = backtrack_verify_successes + backtrack_recovery_success_count
        return {
            "run_wall_time_sec": float(run_end - self._run_start),
            "step_count": len(step_durations),
            "avg_step_time_sec": _safe_mean(step_durations),
            "p50_step_time_sec": _percentile(step_durations, 50.0),
            "p90_step_time_sec": _percentile(step_durations, 90.0),
            "explorer_call_count": explorer_call_count,
            "explorer_time_sec_total": float(explorer_time_sec_total),
            "avg_explorer_time_sec": _safe_mean([explorer_time_sec_total / explorer_call_count]) if explorer_call_count else 0.0,
            "decider_call_count": decider_call_count,
            "decider_time_sec_total": float(decider_time_sec_total),
            "avg_decider_time_sec": _safe_mean([decider_time_sec_total / decider_call_count]) if decider_call_count else 0.0,
            "backtrack_count": backtrack_count,
            "backtrack_verify_success_rate": _safe_rate(backtrack_verify_successes, backtrack_count),
            "backtrack_recovery_sequence_count": backtrack_recovery_sequence_count,
            "backtrack_recovery_attempt_count": backtrack_recovery_attempt_count,
            "backtrack_recovery_success_rate": _safe_rate(
                backtrack_recovery_success_count,
                backtrack_recovery_sequence_count,
            ),
            "final_backtrack_success_rate": _safe_rate(final_backtrack_successes, backtrack_count),
            "avg_backtrack_time_sec": _safe_mean(backtrack_durations),
            "cache_hit_rate": _safe_rate(explorer_cache_hits, explorer_cache_requests),
            "explorer_cache_hit_rate": _safe_rate(explorer_cache_hits, explorer_cache_requests),
            "screen_cache_hit_rate": _safe_rate(screen_cache_hits, screen_cache_requests),
            "artifact_io_time_sec_total": float(artifact_io_time_sec_total),
            "partial_path_count": partial_path_count,
            "complete_path_count": complete_path_count,
            "total_saved_trace_count": partial_path_count + complete_path_count,
            "timing_traces": timing_traces,
        }


def build_metrics_payload(
    *,
    metrics: MetricsCollector,
    experiment_tag: str,
    data_dir: str,
    feature_flags: FeatureFlags,
    configured_depth_limit: int | None = None,
    configured_breadth: int | None = None,
) -> Dict[str, Any]:
    payload = metrics.snapshot()
    payload["experiment_tag"] = experiment_tag
    payload["data_dir"] = data_dir
    payload["feature_flags"] = asdict(feature_flags)
    payload["configured_depth_limit"] = configured_depth_limit
    payload["configured_breadth"] = configured_breadth
    return payload


def write_metrics_payload(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_metric_runs(metric_payloads: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    runs = [dict(item) for item in metric_payloads]
    if not runs:
        return {"runs": [], "aggregate": {}}

    numeric_keys = [
        key
        for key, value in runs[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    aggregate: Dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(run.get(key, 0.0)) for run in runs]
        aggregate[key] = {
            "mean": _safe_mean(values),
            "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return {"runs": runs, "aggregate": aggregate}


def build_relative_delta(
    *,
    baseline_summary: Mapping[str, Any],
    experiment_summary: Mapping[str, Any],
) -> Dict[str, float]:
    baseline_agg = dict(baseline_summary.get("aggregate", {}))
    experiment_agg = dict(experiment_summary.get("aggregate", {}))
    deltas: Dict[str, float] = {}
    for key, baseline_stats in baseline_agg.items():
        exp_stats = experiment_agg.get(key)
        if not isinstance(baseline_stats, Mapping) or not isinstance(exp_stats, Mapping):
            continue
        baseline_mean = float(baseline_stats.get("mean", 0.0))
        exp_mean = float(exp_stats.get("mean", 0.0))
        if baseline_mean == 0.0:
            deltas[key] = 0.0
        else:
            deltas[key] = (exp_mean - baseline_mean) / baseline_mean
    return deltas
