import json
from pathlib import Path
from unittest import TestCase, mock

from test_support import install_test_stubs, make_workspace_tempdir, remove_workspace_tempdir

install_test_stubs()

from auto_explore.cli import auto_search
from auto_explore.core.runtime import FeatureFlags


class AutoSearchMetricsTests(TestCase):
    def test_parse_args_exposes_feature_flags_with_defaults_enabled(self):
        args = auto_search.parse_args(
            [
                "--app_name",
                "DemoApp",
                "--depth",
                "2",
                "--breadth",
                "2",
                "--openrouter_api_key",
                "dummy-key",
            ]
        )

        flags = FeatureFlags.from_namespace(args)
        self.assertTrue(flags.triple_verify)
        self.assertTrue(flags.replay_recovery)
        self.assertTrue(flags.candidate_dedup)
        self.assertTrue(flags.already_explored_filter)
        self.assertTrue(flags.explorer_cache)
        self.assertTrue(flags.screen_cache)
        self.assertTrue(flags.popup_auto_dismiss)
        self.assertTrue(flags.async_artifact_io)
        self.assertTrue(flags.concurrent_fingerprint)
        self.assertTrue(flags.hierarchy_text_decider)

    def test_run_writes_metrics_json_with_expected_fields(self):
        tmpdir = make_workspace_tempdir("auto_search_metrics")
        try:
            data_dir = tmpdir / "data"
            metrics_path = tmpdir / "metrics.json"
            args = auto_search.parse_args(
                [
                    "--app_name",
                    "DemoApp",
                    "--depth",
                    "1",
                    "--breadth",
                    "1",
                    "--device",
                    "Android",
                    "--openrouter_api_key",
                    "dummy-key",
                    "--data_dir",
                    str(data_dir),
                    "--metrics_output_path",
                    str(metrics_path),
                    "--experiment_tag",
                    "unit-test",
                    "--enable_explorer_cache",
                    "off",
                ]
            )

            fake_device = mock.Mock()
            fake_device.start_app = mock.Mock()

            with mock.patch("auto_explore.cli.auto_search.AndroidDevice", return_value=fake_device), mock.patch(
                "auto_explore.cli.auto_search.init_decider_client", return_value=object()
            ), mock.patch(
                "auto_explore.cli.auto_search.init_explorer_client", return_value=object()
            ), mock.patch(
                "auto_explore.cli.auto_search.explore_dfs"
            ) as mocked_explore_dfs, mock.patch(
                "auto_explore.cli.auto_search.flush_artifact_tasks"
            ):
                payload = auto_search.run(args)

            self.assertTrue(metrics_path.exists())
            saved_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["experiment_tag"], "unit-test")
            self.assertEqual(saved_payload["data_dir"], str(data_dir))
            self.assertIn("run_wall_time_sec", saved_payload)
            self.assertIn("step_count", saved_payload)
            self.assertIn("timing_traces", saved_payload)
            self.assertIn("partial_path_count", saved_payload)
            self.assertIn("complete_path_count", saved_payload)
            self.assertIn("total_saved_trace_count", saved_payload)
            self.assertEqual(saved_payload["configured_depth_limit"], 1)
            self.assertEqual(saved_payload["configured_breadth"], 1)
            self.assertEqual(payload["feature_flags"]["explorer_cache"], False)
            mocked_explore_dfs.assert_called_once()
        finally:
            remove_workspace_tempdir(tmpdir)
