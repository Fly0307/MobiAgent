from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _src_root() -> Path:
    return _repo_root() / "auto_explore" / "src"


if str(_src_root()) not in sys.path:
    sys.path.insert(0, str(_src_root()))

from auto_explore.cli.ablation_runner import default_experiments


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test every ablation experiment entry one by one")
    parser.add_argument("--manifest", type=str, default="", help="Benchmark manifest forwarded to ablation_runner")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Root directory for smoke-test outputs. Defaults to results/ablation_script_test",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repeated runs per experiment")
    parser.add_argument("--python-executable", type=str, default=sys.executable, help="Python executable for child runs")
    parser.add_argument(
        "--stop-on-error",
        choices=["on", "off"],
        default="on",
        help="Stop immediately when one experiment invocation fails",
    )
    parser.add_argument(
        "base_args",
        nargs=argparse.REMAINDER,
        help="Direct auto_search args when --manifest is not provided. Use -- --app_name App --depth 1 --breadth 1 ...",
    )
    return parser


def _resolve_base_args(args: argparse.Namespace) -> List[str]:
    if args.base_args and args.base_args[0] == "--":
        return args.base_args[1:]
    return list(args.base_args)


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    src_root = _src_root()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_root) if not existing else f"{src_root}{os.pathsep}{existing}"
    return env


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    base_args = _resolve_base_args(args)
    if not args.manifest and not base_args:
        raise ValueError("Please provide --manifest or direct auto_search args after --")

    repo_root = _repo_root()
    output_root = Path(args.output_root) if args.output_root else repo_root / "results" / "ablation_script_test"
    output_root.mkdir(parents=True, exist_ok=True)
    env = _build_env()

    failures: List[str] = []
    for experiment in default_experiments():
        experiment_root = output_root / experiment.name
        experiment_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_executable,
            "-m",
            "auto_explore.cli.ablation_runner",
            "--output-root",
            str(experiment_root),
            "--repeats",
            str(args.repeats),
            "--experiment",
            experiment.name,
        ]
        if args.manifest:
            cmd.extend(["--manifest", args.manifest])
        elif base_args:
            cmd.append("--")
            cmd.extend(base_args)

        print(f"[smoke-test] running {experiment.name}")
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
        )
        if result.returncode != 0:
            failures.append(experiment.name)
            print(f"[smoke-test] FAILED: {experiment.name} exit_code={result.returncode}")
            if args.stop_on_error == "on":
                raise SystemExit(result.returncode)
        else:
            print(f"[smoke-test] passed: {experiment.name}")

    if failures:
        print(f"[smoke-test] failures: {', '.join(failures)}")
        raise SystemExit(1)

    print("[smoke-test] all ablation experiment entries completed successfully")


if __name__ == "__main__":
    main()
