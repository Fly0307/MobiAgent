import sys
from pathlib import Path
from unittest import TestCase


class AutoExploreLayoutTests(TestCase):
    def test_auto_explore_top_level_layout_exists(self):
        required = [
            Path("auto_explore/README.md"),
            Path("auto_explore/configs/simulators.example.json"),
            Path("auto_explore/scripts/run_single.sh"),
            Path("auto_explore/scripts/run_parallel.sh"),
            Path("auto_explore/scripts/run_single.bat"),
            Path("auto_explore/scripts/run_eval.bat"),
            Path("auto_explore/scripts/run_ablation.bat"),
            Path("auto_explore/scripts/run_ablation_experiment.bat"),
            Path("auto_explore/scripts/run_ablation_all.bat"),
            Path("auto_explore/scripts/test_ablation_scripts.py"),
            Path("auto_explore/scripts/test_ablation_scripts.bat"),
            Path("auto_explore/eval/README.md"),
            Path("auto_explore/eval/legacy_text_system_prompt.md"),
            Path("auto_explore/eval/legacy_text_user_prompt.md"),
            Path("auto_explore/eval/path_multimodal_system_prompt.md"),
            Path("auto_explore/eval/path_multimodal_user_prompt.md"),
            Path("auto_explore/src/auto_explore/__init__.py"),
            Path("auto_explore/src/auto_explore/eval/__init__.py"),
            Path("auto_explore/src/auto_explore/cli/__init__.py"),
            Path("auto_explore/src/auto_explore/core/__init__.py"),
            Path("auto_explore/src/auto_explore/adapters/__init__.py"),
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing: {path}")

    def test_docs_reference_new_auto_explore_paths(self):
        content = Path("docs/auto-search.md").read_text(encoding="utf-8")
        self.assertIn("auto_explore/", content)
        self.assertNotIn("runner/mobiagent/auto-search.py", content)

    def test_import_auto_explore_adds_repo_root_to_sys_path(self):
        import auto_explore

        repo_root = str(Path(__file__).resolve().parents[2])
        self.assertIn(repo_root, sys.path)

    def test_run_scripts_do_not_depend_on_pwd_for_pythonpath(self):
        single = Path("auto_explore/scripts/run_single.sh").read_text(encoding="utf-8")
        parallel = Path("auto_explore/scripts/run_parallel.sh").read_text(encoding="utf-8")
        eval_script = Path("auto_explore/scripts/run_eval.bat").read_text(encoding="utf-8")
        ablation = Path("auto_explore/scripts/run_ablation.bat").read_text(encoding="utf-8")
        ablation_single = Path("auto_explore/scripts/run_ablation_experiment.bat").read_text(encoding="utf-8")
        ablation_test = Path("auto_explore/scripts/test_ablation_scripts.bat").read_text(encoding="utf-8")

        self.assertNotIn('${PWD}/auto_explore/src', single)
        self.assertNotIn('${PWD}/auto_explore/src', parallel)
        self.assertNotIn('%CD%\\auto_explore\\src', eval_script)
        self.assertNotIn('%CD%\\auto_explore\\src', ablation)
        self.assertNotIn('%CD%\\auto_explore\\src', ablation_single)
        self.assertNotIn('%CD%\\auto_explore\\src', ablation_test)

    def test_ablation_script_wrappers_route_to_expected_entrypoints(self):
        single_experiment = Path("auto_explore/scripts/run_ablation_experiment.bat").read_text(encoding="utf-8")
        run_all = Path("auto_explore/scripts/run_ablation_all.bat").read_text(encoding="utf-8")
        smoke_test = Path("auto_explore/scripts/test_ablation_scripts.py").read_text(encoding="utf-8")

        self.assertIn("--experiment", single_experiment)
        self.assertIn("run_ablation.bat", run_all)
        self.assertIn("default_experiments()", smoke_test)

    def test_auto_search_cli_is_trimmed_to_orchestration_layer(self):
        content = Path("auto_explore/src/auto_explore/cli/auto_search.py").read_text(encoding="utf-8")
        self.assertNotIn("def explore_dfs(", content)
        self.assertNotIn("def call_explorer_model(", content)
        self.assertNotIn("def execute_decider_one_step(", content)

    def test_eval_package_exists(self):
        content = Path("auto_explore/eval/README.md").read_text(encoding="utf-8")
        self.assertIn("trajectory_completeness_score", content)
        self.assertIn("reasoning_quality_score", content)
        self.assertIn("path_multimodal", content)
        self.assertIn("image_coherence_score", content)

    def test_eval_script_targets_eval_cli(self):
        content = Path("auto_explore/scripts/run_eval.bat").read_text(encoding="utf-8")
        self.assertIn("auto_explore.eval.cli", content)
        self.assertIn("AUTO_EXPLORE_EVAL_INPUT_PATH", content)
        self.assertIn("AUTO_EXPLORE_EVAL_JUDGE_MODE", content)
