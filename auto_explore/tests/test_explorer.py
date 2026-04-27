from types import SimpleNamespace
from unittest import TestCase, mock

from test_support import install_test_stubs

install_test_stubs()

from auto_explore.core.explorer import _parse_explorer_response_content, call_explorer_model
from auto_explore.core.prompting import build_explorer_prompt


class _FakeCompletions:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self._content = content
        self._finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                    finish_reason=self._finish_reason,
                )
            ]
        )


class _FakeChat:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.completions = _FakeCompletions(content, finish_reason=finish_reason)


class _FakeClient:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.chat = _FakeChat(content, finish_reason=finish_reason)


class ExplorerParsingTests(TestCase):
    def test_parse_explorer_response_accepts_plain_json(self):
        parsed = _parse_explorer_response_content(
            '{"candidates":[{"rank":1,"single_step_task":"点击搜索框","reason":"入口明显"}]}',
            finish_reason="stop",
            attempt=1,
        )
        self.assertEqual(parsed["candidates"][0]["single_step_task"], "点击搜索框")

    def test_parse_explorer_response_accepts_fenced_json(self):
        parsed = _parse_explorer_response_content(
            '```json\n{"candidates":[{"rank":1,"single_step_task":"点击我的","reason":"入口明显"}]}\n```',
            finish_reason="stop",
            attempt=1,
        )
        self.assertEqual(parsed["candidates"][0]["single_step_task"], "点击我的")

    def test_parse_explorer_response_accepts_json_with_prefix_suffix(self):
        parsed = _parse_explorer_response_content(
            '下面是结果：{"candidates":[{"rank":1,"single_step_task":"点击购物车","reason":"高频入口"}]} 谢谢',
            finish_reason="stop",
            attempt=1,
        )
        self.assertEqual(parsed["candidates"][0]["single_step_task"], "点击购物车")

    def test_parse_explorer_response_reports_truncation(self):
        with self.assertRaisesRegex(ValueError, "疑似截断"):
            _parse_explorer_response_content(
                '{"candidates":[{"rank":1,"single_step_task":"点击我的淘宝","reason":"入口明显"}',
                finish_reason="length",
                attempt=2,
            )

    def test_call_explorer_model_parses_fenced_json_candidates(self):
        client = _FakeClient(
            '```json\n{"candidates":[{"rank":1,"single_step_task":"点击我的淘宝","reason":"高频入口"}]}\n```'
        )

        candidates, popup_info = call_explorer_model(
            explorer_client=client,
            explorer_model="demo-model",
            screenshot_b64="abc",
            hierarchy_text="<root />",
            depth=0,
            breadth=3,
            action_history=[],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["single_step_task"], "点击我的淘宝")
        self.assertIsNone(popup_info)

    def test_call_explorer_model_rejects_non_list_candidates(self):
        client = _FakeClient('{"candidates":{"rank":1}}')

        with mock.patch("auto_explore.core.explorer.time.sleep"), self.assertRaisesRegex(RuntimeError, "candidates"):
            call_explorer_model(
                explorer_client=client,
                explorer_model="demo-model",
                screenshot_b64="abc",
                hierarchy_text="<root />",
                depth=0,
                breadth=3,
                action_history=[],
            )

    def test_call_explorer_model_uses_zero_temperature_by_default(self):
        client = _FakeClient('{"candidates":[{"rank":1,"single_step_task":"tap search","reason":"stable"}]}')

        candidates, popup_info = call_explorer_model(
            explorer_client=client,
            explorer_model="demo-model",
            screenshot_b64="abc",
            hierarchy_text="<root />",
            depth=0,
            breadth=3,
            action_history=[],
        )

        self.assertEqual(len(candidates), 1)
        self.assertIsNone(popup_info)
        self.assertEqual(client.chat.completions.calls[0]["temperature"], 0.0)


class ExplorerPromptTests(TestCase):
    def test_build_explorer_prompt_enforces_strict_json_contract(self):
        prompt = build_explorer_prompt(
            depth=0,
            breadth=3,
            hierarchy_text="<root />",
            action_history=[],
        )

        self.assertIn("只输出一个 JSON object", prompt)
        self.assertIn("不要输出任何额外文本、解释、前后缀、markdown", prompt)
        self.assertIn("reason 必须短句化", prompt)
        self.assertIn('例如 {"candidates": []}', prompt)
