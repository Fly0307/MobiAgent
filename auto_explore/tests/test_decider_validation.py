from types import SimpleNamespace
from unittest import TestCase, mock

from test_support import install_test_stubs

install_test_stubs()

from auto_explore.core.decider import execute_decider_one_step, sanitize_decider_response


class _Metrics:
    def now(self):
        return 0.0

    def relative_time(self, value=None):
        return 0.0 if value is None else value

    def record_decider_call(self, *_args, **_kwargs):
        return None

    def record_timing_trace(self, *_args, **_kwargs):
        return None

    def record_step_duration(self, *_args, **_kwargs):
        return None

    def record_backtrack(self, *_args, **_kwargs):
        return None

    def record_artifact_io(self, *_args, **_kwargs):
        return None


class DeciderResponseSanitizationTests(TestCase):
    def test_sanitize_accepts_valid_click_bbox(self):
        response = sanitize_decider_response(
            {"action": "click", "parameters": {"bbox": [126, "895", 258, 928]}, "reasoning": "ok"}
        )
        self.assertEqual(response["parameters"]["bbox"], [126, 895, 258, 928])

    def test_sanitize_rejects_non_numeric_bbox_value(self):
        with self.assertRaisesRegex(ValueError, "Decider invalid bbox"):
            sanitize_decider_response(
                {"action": "click", "parameters": {"bbox": [126, "8PS", 258, 928]}, "reasoning": "bad"}
            )

    def test_sanitize_rejects_wrong_bbox_length(self):
        with self.assertRaisesRegex(ValueError, "Decider invalid bbox"):
            sanitize_decider_response(
                {"action": "click", "parameters": {"bbox": [0, 10, 20]}, "reasoning": "bad"}
            )

    def test_sanitize_rejects_click_input_without_text(self):
        with self.assertRaisesRegex(ValueError, "Decider invalid action schema"):
            sanitize_decider_response(
                {"action": "click_input", "parameters": {"bbox": [0, 0, 10, 10]}, "reasoning": "bad"}
            )

    def test_sanitize_rejects_swipe_invalid_coords(self):
        with self.assertRaisesRegex(ValueError, "Decider invalid coords"):
            sanitize_decider_response(
                {
                    "action": "swipe",
                    "parameters": {"start_coords": [0, "left"], "end_coords": [10, 20]},
                    "reasoning": "bad",
                }
            )


class DeciderRetryTests(TestCase):
    def test_execute_decider_retries_after_invalid_bbox_and_succeeds(self):
        runtime = SimpleNamespace(
            metrics=_Metrics(),
            features=SimpleNamespace(hierarchy_text_decider=False, async_artifact_io=False),
        )
        fake_device = mock.Mock()
        fake_device.click = mock.Mock()

        responses = [
            {"reasoning": "bad", "action": "click", "parameters": {"bbox": [126, "8PS", 258, 928]}},
            {"reasoning": "ok", "action": "click", "parameters": {"bbox": [10, 20, 30, 40]}},
        ]

        with mock.patch("auto_explore.core.decider.get_screenshot", return_value="img-b64"), mock.patch(
            "auto_explore.core.decider.get_hierarchy_text", return_value="<root />"
        ), mock.patch(
            "auto_explore.core.decider.build_auto_decider_messages", return_value=[{"role": "user", "content": []}]
        ), mock.patch(
            "auto_explore.core.decider.call_model_with_validation_retry",
            side_effect=responses,
        ) as mocked_call, mock.patch(
            "auto_explore.core.decider.save_raw_screenshot", return_value="dummy.jpg"
        ), mock.patch(
            "auto_explore.core.decider.save_hierarchy"
        ), mock.patch(
            "auto_explore.core.decider.refine_bbox_with_hierarchy", return_value=[10, 20, 30, 40]
        ), mock.patch(
            "auto_explore.core.decider._run_annotation_safe"
        ):
            result = execute_decider_one_step(
                decider_client=object(),
                decider_model="demo-model",
                device=fake_device,
                device_type="Android",
                app_name="DemoApp",
                step_task="click target",
                use_qwen3=False,
                allow_hierarchy_text_decider=False,
                output_dir="unused",
                step_index=1,
                runtime=runtime,
                history=["step 1", "step 2"],
            )

        self.assertEqual(mocked_call.call_count, 2)
        self.assertEqual(result["decider_response"]["parameters"]["bbox"], [10, 20, 30, 40])
        fake_device.click.assert_called_once_with(20, 30)
