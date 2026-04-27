import argparse
import codecs
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from auto_explore.core.runtime import (
    FeatureFlags,
    build_relative_delta,
    summarize_metric_runs,
)


@dataclass(frozen=True)
class AblationExperiment:
    name: str
    feature_flags: FeatureFlags
    enable_ui_semantic_collect: bool


def default_experiments(base_flags: Optional[FeatureFlags] = None) -> List[AblationExperiment]:
    base = base_flags or FeatureFlags()
    experiments = [
        AblationExperiment("E0_full", base, False),
        AblationExperiment("E1_no_triple_verify", base.with_overrides(triple_verify=False), False),
        AblationExperiment("E2_no_replay_recovery", base.with_overrides(replay_recovery=False), False),
        AblationExperiment("E3_no_candidate_dedup", base.with_overrides(candidate_dedup=False), False),
        AblationExperiment("E4_no_already_explored_filter", base.with_overrides(already_explored_filter=False), False),
        AblationExperiment("E5_no_explorer_cache", base.with_overrides(explorer_cache=False), False),
        AblationExperiment("E6_no_screen_cache", base.with_overrides(screen_cache=False), False),
        AblationExperiment("E7_no_popup_auto_dismiss", base.with_overrides(popup_auto_dismiss=False), False),
        AblationExperiment("E8_no_async_artifact_io", base.with_overrides(async_artifact_io=False), False),
        AblationExperiment("E9_no_concurrent_fingerprint", base.with_overrides(concurrent_fingerprint=False), False),
        AblationExperiment("E10_no_hierarchy_text_decider", base.with_overrides(hierarchy_text_decider=False), False),
        AblationExperiment("U0_ui_collect_off", base, False),
        AblationExperiment("U1_ui_collect_on", base, True),
    ]
    return experiments


def build_auto_search_command(
    *,
    python_executable: str,
    base_args: List[str],
    experiment: AblationExperiment,
    output_dir: Path,
    run_index: int,
) -> List[str]:
    run_dir = output_dir / experiment.name / f"run_{run_index:03d}"
    data_dir = run_dir / "data"
    metrics_path = run_dir / "metrics.json"
    return [
        python_executable,
        "-m",
        "auto_explore.cli.auto_search",
        *base_args,
        "--data_dir",
        str(data_dir),
        "--metrics_output_path",
        str(metrics_path),
        "--experiment_tag",
        experiment.name,
        "--enable_ui_semantic_collect",
        "on" if experiment.enable_ui_semantic_collect else "off",
        *experiment.feature_flags.to_cli_args(),
    ]


def _load_manifest(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _base_args_from_manifest(manifest: Mapping[str, Any]) -> List[str]:
    if "base_args" in manifest:
        base_args = manifest.get("base_args", [])
        if not isinstance(base_args, list):
            raise ValueError("manifest.base_args must be a list")
        return [str(item) for item in base_args]

    required = ["app_name", "depth", "breadth"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"manifest missing required keys: {', '.join(missing)}")

    args = [
        "--app_name",
        str(manifest["app_name"]),
        "--depth",
        str(manifest["depth"]),
        "--breadth",
        str(manifest["breadth"]),
    ]
    extra_args = manifest.get("extra_args", [])
    if not isinstance(extra_args, list):
        raise ValueError("manifest.extra_args must be a list")
    args.extend(str(item) for item in extra_args)
    return args


def _experiment_filter(experiments: Iterable[AblationExperiment], names: Optional[List[str]]) -> List[AblationExperiment]:
    all_experiments = list(experiments)
    if not names:
        return all_experiments
    name_set = set(names)
    return [exp for exp in all_experiments if exp.name in name_set]


def _read_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def run_ablation(
    *,
    base_args: List[str],
    output_root: Path,
    repeats: int,
    experiments: List[AblationExperiment],
    python_executable: str,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    per_experiment_summary: Dict[str, Any] = {}

    for experiment in experiments:
        metrics_payloads: List[Mapping[str, Any]] = []
        experiment_root = output_root / experiment.name
        experiment_root.mkdir(parents=True, exist_ok=True)
        for run_index in range(1, repeats + 1):
            run_dir = experiment_root / f"run_{run_index:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "runner.log"
            cmd = build_auto_search_command(
                python_executable=python_executable,
                base_args=base_args,
                experiment=experiment,
                output_dir=output_root,
                run_index=run_index,
            )
            with log_path.open("wb") as log_file:
                log_file.write(codecs.BOM_UTF8)
                log_file.flush()
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).resolve().parents[4]),
                    env=_build_subprocess_env(),
                )
            metrics_path = run_dir / "metrics.json"
            metrics_payloads.append(_read_metrics(metrics_path))

        summary = summarize_metric_runs(metrics_payloads)
        summary_payload = {
            "experiment": experiment.name,
            "feature_flags": experiment.feature_flags.to_cli_args(),
            "enable_ui_semantic_collect": experiment.enable_ui_semantic_collect,
            **summary,
        }
        per_experiment_summary[experiment.name] = summary_payload
        _write_json(experiment_root / "summary.json", summary_payload)

    baseline_summary = per_experiment_summary.get("E0_full")
    ablation_metrics: Dict[str, Any] = {}
    for experiment_name, summary in per_experiment_summary.items():
        entry = dict(summary)
        if baseline_summary is not None:
            entry["relative_delta_vs_E0_full"] = build_relative_delta(
                baseline_summary=baseline_summary,
                experiment_summary=summary,
            )
        ablation_metrics[experiment_name] = entry

    payload = {"experiments": ablation_metrics}
    _write_json(output_root / "ablation_metrics.json", payload)
    return payload


def _build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[4]
    src_root = repo_root / "auto_explore" / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_root) if not existing else f"{src_root}{os.pathsep}{existing}"
    # Force child Python processes to emit UTF-8 so runner.log stays readable with Chinese content.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run auto_explore ablation experiments")
    parser.add_argument("--manifest", type=str, default="", help="JSON manifest describing the benchmark configuration")
    parser.add_argument("--output-root", type=str, default="", help="Root directory to store ablation outputs")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated runs per experiment")
    parser.add_argument("--python-executable", type=str, default=sys.executable, help="Python executable used for child runs")
    parser.add_argument("--experiment", action="append", default=[], help="Run only selected experiment names")
    parser.add_argument(
        "base_args",
        nargs=argparse.REMAINDER,
        help="Direct auto_search args when --manifest is not provided. Use -- --app_name App --depth 2 --breadth 2 ...",
    )
    return parser


def _resolve_base_args(args: argparse.Namespace) -> List[str]:
    if args.manifest:
        return _base_args_from_manifest(_load_manifest(args.manifest))
    if args.base_args and args.base_args[0] == "--":
        return args.base_args[1:]
    return args.base_args


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    base_args = _resolve_base_args(args)
    if not base_args:
        raise ValueError("Please provide --manifest or direct auto_search args after --")

    output_root = Path(args.output_root) if args.output_root else Path("results") / "ablation"
    experiments = _experiment_filter(default_experiments(), args.experiment or None)
    payload = run_ablation(
        base_args=base_args,
        output_root=output_root,
        repeats=args.repeats,
        experiments=experiments,
        python_executable=args.python_executable,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
