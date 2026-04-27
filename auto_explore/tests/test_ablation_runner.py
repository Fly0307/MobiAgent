import json
from pathlib import Path
from unittest import TestCase, mock

from auto_explore.cli.ablation_runner import (
    _build_subprocess_env,
    build_auto_search_command,
    default_experiments,
    run_ablation,
)
from auto_explore.core.runtime import FeatureFlags
from test_support import make_workspace_tempdir, remove_workspace_tempdir


class AblationRunnerTests(TestCase):
    def test_default_experiments_cover_main_matrix_and_ui_collect(self):
        names = [item.name for item in default_experiments()]
        self.assertEqual(
            names,
            [
                "E0_full",
                "E1_no_triple_verify",
                "E2_no_replay_recovery",
                "E3_no_candidate_dedup",
                "E4_no_already_explored_filter",
                "E5_no_explorer_cache",
                "E6_no_screen_cache",
                "E7_no_popup_auto_dismiss",
                "E8_no_async_artifact_io",
                "E9_no_concurrent_fingerprint",
                "E10_no_hierarchy_text_decider",
                "U0_ui_collect_off",
                "U1_ui_collect_on",
            ],
        )

    def test_build_auto_search_command_includes_metrics_and_feature_flags(self):
        experiment = default_experiments(FeatureFlags())[1]
        cmd = build_auto_search_command(
            python_executable="python",
            base_args=["--app_name", "DemoApp", "--depth", "1", "--breadth", "1"],
            experiment=experiment,
            output_dir=Path("/tmp/ablation"),
            run_index=2,
        )

        self.assertEqual(cmd[:4], ["python", "-m", "auto_explore.cli.auto_search", "--app_name"])
        self.assertIn("--metrics_output_path", cmd)
        self.assertIn("--experiment_tag", cmd)
        self.assertIn("--enable_triple_verify", cmd)
        self.assertEqual(cmd[cmd.index("--enable_triple_verify") + 1], "off")
        self.assertIn("--enable_ui_semantic_collect", cmd)

    def test_run_ablation_writes_summary_and_ablation_metrics(self):
        tmpdir = make_workspace_tempdir("ablation_runner")
        try:
            output_root = tmpdir / "results"
            experiments = default_experiments()[:2]

            def fake_run(cmd, check, stdout, stderr, cwd, env):
                metrics_path = Path(cmd[cmd.index("--metrics_output_path") + 1])
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "run_wall_time_sec": 10.0 if "E0_full" in str(metrics_path) else 12.0,
                    "step_count": 4,
                    "avg_step_time_sec": 2.5,
                    "p50_step_time_sec": 2.0,
                    "p90_step_time_sec": 3.0,
                    "explorer_call_count": 2,
                    "explorer_time_sec_total": 3.0,
                    "avg_explorer_time_sec": 1.5,
                    "decider_call_count": 2,
                    "decider_time_sec_total": 4.0,
                    "avg_decider_time_sec": 2.0,
                    "backtrack_count": 2,
                    "backtrack_verify_success_rate": 0.5,
                    "backtrack_recovery_sequence_count": 1,
                    "backtrack_recovery_attempt_count": 1,
                    "backtrack_recovery_success_rate": 1.0,
                    "final_backtrack_success_rate": 1.0,
                    "avg_backtrack_time_sec": 1.2,
                    "cache_hit_rate": 0.5,
                    "explorer_cache_hit_rate": 0.5,
                    "screen_cache_hit_rate": 0.0,
                    "artifact_io_time_sec_total": 0.8,
                    "timing_traces": [],
                }
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")
                return mock.Mock(returncode=0)

            with mock.patch("auto_explore.cli.ablation_runner.subprocess.run", side_effect=fake_run):
                payload = run_ablation(
                    base_args=["--app_name", "DemoApp", "--depth", "1", "--breadth", "1"],
                    output_root=output_root,
                    repeats=2,
                    experiments=experiments,
                    python_executable="python",
                )

            self.assertTrue((output_root / "E0_full" / "summary.json").exists())
            self.assertTrue((output_root / "E1_no_triple_verify" / "summary.json").exists())
            self.assertTrue((output_root / "ablation_metrics.json").exists())
            experiments_payload = payload["experiments"]
            self.assertIn("E0_full", experiments_payload)
            self.assertIn("E1_no_triple_verify", experiments_payload)
            self.assertIn("relative_delta_vs_E0_full", experiments_payload["E1_no_triple_verify"])
        finally:
            remove_workspace_tempdir(tmpdir)

    def test_build_subprocess_env_forces_utf8_output(self):
        env = _build_subprocess_env()
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertIn("auto_explore\\src", env["PYTHONPATH"].replace("/", "\\"))
