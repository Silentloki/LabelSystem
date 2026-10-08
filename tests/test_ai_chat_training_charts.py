"""Real image files reach the injected API boundary; response is simulated."""
from pathlib import Path

from PIL import Image
from test_ai_chat_selection import Fixtures
import test_ai_training_resources as resources_test
from test_ai_training_analysis import Fixtures as AnalysisFixtures
from Utils.AIWorkAgent import run_agent
from Utils.AIWorkspace import read_json
from Utils.AIResultBrowser import visual_rows


class TrainingChartTests(Fixtures):
    training_folder = resources_test.TrainingResourceTests.training_folder
    tools = resources_test.TrainingResourceTests.tools

    def setUp(self):
        super().setUp()
        self.folder = self.training_folder('train35')
        self.resources = {'r1': {'kind': 'directory', 'name': 'train35', 'path': str(self.folder)}}
        for name in ['BoxPR_curve.png', 'confusion_matrix.png']:
            Image.new('RGB', (80, 80), 'white').save(self.folder / name)

    def report(self):
        report = AnalysisFixtures.response()
        report['next_experiment']['change'] = None
        report['chart_readings'] = [
            {'evidence': 'C.plot_pr', 'readable': True, 'names': ['划痕'], 'values': [.7]},
            {'evidence': 'C.plot_confusion', 'readable': True, 'names': ['划痕', 'background'],
             'row_axis': 'predicted', 'column_axis': 'true', 'matrix': [[8, 2], [3, 0]]}]
        return report

    def test_images_are_attached_and_report_counts_computed_without_extra_confirmation(self):
        calls = []
        def request(*args, **kwargs):
            calls.append((args[3], args[4]))
            if len(calls) == 1:
                self.assertFalse(args[4])
                return {'action': 'tool', 'tool': 'training_runs', 'args': {'resource': 'r1'}}, {}
            self.assertEqual(len(args[4]), 2)
            self.assertTrue(all(Path(path).is_file() for path in args[4]))
            return {'action': 'reply', 'message': '分析完成', 'analysis': self.report()}, {}
        tools = self.tools()
        payload = run_agent(self.root, self.paths, {}, {}, [], '分析训练结果', ('all', None, None),
                            ('mock', 'https://invalid.test', 'mock'), tools.context, request=request)
        self.assertEqual(payload['work']['status'], 'completed')
        self.assertEqual(payload['requests'], 2)
        self.assertIn('检对 8，漏检 3', payload['result']['message'])
        self.assertIn('额外误报', payload['result']['message'])
        result = payload['work']['steps'][0]['result']
        self.assertTrue((Path(result['output']) / 'report.html').exists())
        self.assertEqual(len(visual_rows(tools.store, result['artifact_id'])[1]), 2)
        log = read_json(tools.store.location('runs', payload['run_id']) / 'run.json')
        self.assertEqual(len(log['model_responses'][1]['image_inputs']), 2)

    def test_missing_chart_readings_cannot_be_reported_as_success(self):
        from Utils.AIChatTraining import complete
        tools = self.tools()
        tools.execute('training_runs', {})
        with self.assertRaises(ValueError):
            complete(tools.training_analysis, AnalysisFixtures.response())
        report = self.report()
        report['next_experiment']['action'] = '请使用 --save-class-metrics'
        with self.assertRaisesRegex(ValueError, '未提供'):
            complete(tools.training_analysis, report)
        report = self.report()
        report['summary']['text'] = '分类损失从 1.233 上升至 1.189。'
        with self.assertRaisesRegex(ValueError, '方向写反'):
            complete(tools.training_analysis, report)

    def test_changed_chart_is_rejected_before_sending(self):
        from Utils.AIChatTraining import image_paths
        tools = self.tools()
        result = tools.execute('training_runs', {})
        Path(tools.training_analysis['images'][0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '已改变'):
            image_paths(tools.training_analysis)

    def test_record_only_does_not_prepare_images(self):
        tools = self.tools()
        result = tools.execute('training_runs', {'analyze': False})
        self.assertEqual(result['charts'], [])
        self.assertIsNone(tools.training_analysis)
        with self.assertRaisesRegex(ValueError, '没有已保存的图表'):
            tools.execute('view_result', {'artifact_id': result['artifact_id']})
