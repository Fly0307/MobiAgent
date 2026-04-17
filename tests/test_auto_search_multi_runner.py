import json
from pathlib import Path
from unittest import TestCase, mock

from runner.mobiagent.auto_search_multi_runner import (
    SimulatorConfig,
    build_auto_search_command,
    init_simulators,
    load_simulators_from_json,
)


class AutoSearchMultiRunnerTests(TestCase):
    def test_load_simulators_from_json_extracts_backend_and_adb_endpoint(self):
        config = {
            "simulators": [
                {
                    "name": "模拟器1",
                    "backend_url": "http://123.60.91.241:9000",
                    "adb_endpoint": "123.60.91.241:8000",
                },
                {
                    "name": "模拟器2",
                    "backend_url": "http://123.60.91.241:9001",
                    "adb_endpoint": "123.60.91.241:8080",
                },
            ]
        }
        simulators = load_simulators_from_json(json.dumps(config, ensure_ascii=False))

        self.assertEqual([sim.name for sim in simulators], ['模拟器1', '模拟器2'])
        self.assertEqual(simulators[0].backend_url, 'http://123.60.91.241:9000')
        self.assertEqual(simulators[0].adb_endpoint, '123.60.91.241:8000')
        self.assertEqual(simulators[1].backend_url, 'http://123.60.91.241:9001')
        self.assertEqual(simulators[1].adb_endpoint, '123.60.91.241:8080')

    def test_load_simulators_from_json_requires_simulators_key(self):
        with self.assertRaisesRegex(ValueError, "simulators"):
            load_simulators_from_json("{}")

    def test_init_simulators_runs_all_tasks_in_order(self):
        calls = []

        class FakeClient:
            def __init__(self, base_url, device):
                calls.append(('init', base_url, device))
                self.base_url = base_url
                self.device = device

            def health_check(self):
                calls.append(('health', self.base_url, self.device))
                return True

            def ensure_initialized(self):
                calls.append(('ensure_initialized', self.base_url, self.device))
                return True

            def load_snapshot_via_task_init(self, task_name):
                calls.append(('load_task', self.base_url, self.device, task_name))
                return True

        simulators = [
            SimulatorConfig(name='模拟器1', backend_url='http://127.0.0.1:9000', adb_endpoint='127.0.0.1:8000'),
            SimulatorConfig(name='模拟器2', backend_url='http://127.0.0.1:9001', adb_endpoint='127.0.0.1:8080'),
        ]

        with mock.patch('runner.mobiagent.auto_search_multi_runner.SnapshotManagerClient', FakeClient):
            init_simulators(simulators, ['TaskA', 'TaskB'])

        self.assertEqual(calls, [
            ('init', 'http://127.0.0.1:9000', '127.0.0.1:8000'),
            ('health', 'http://127.0.0.1:9000', '127.0.0.1:8000'),
            ('ensure_initialized', 'http://127.0.0.1:9000', '127.0.0.1:8000'),
            ('load_task', 'http://127.0.0.1:9000', '127.0.0.1:8000', 'TaskA'),
            ('load_task', 'http://127.0.0.1:9000', '127.0.0.1:8000', 'TaskB'),
            ('init', 'http://127.0.0.1:9001', '127.0.0.1:8080'),
            ('health', 'http://127.0.0.1:9001', '127.0.0.1:8080'),
            ('ensure_initialized', 'http://127.0.0.1:9001', '127.0.0.1:8080'),
            ('load_task', 'http://127.0.0.1:9001', '127.0.0.1:8080', 'TaskA'),
            ('load_task', 'http://127.0.0.1:9001', '127.0.0.1:8080', 'TaskB'),
        ])

    def test_build_auto_search_command_includes_adb_endpoint_and_output_dir(self):
        simulator = SimulatorConfig(name='模拟器1', backend_url='http://127.0.0.1:9000', adb_endpoint='127.0.0.1:8000')

        cmd = build_auto_search_command(
            simulator=simulator,
            app_name='微博',
            depth=2,
            breadth=2,
            auto_search_args=['--device', 'Android', '--decider_base_url', 'http://decider/v1'],
            output_root=Path('/tmp/auto-search-test'),
        )

        self.assertEqual(cmd[:4], ['python', '-m', 'runner.mobiagent.auto-search', '--app_name'])
        self.assertIn('--adb_endpoint', cmd)
        self.assertEqual(cmd[cmd.index('--adb_endpoint') + 1], '127.0.0.1:8000')
        self.assertIn('--data_dir', cmd)
        self.assertEqual(cmd[cmd.index('--data_dir') + 1], '/tmp/auto-search-test/模拟器1')
