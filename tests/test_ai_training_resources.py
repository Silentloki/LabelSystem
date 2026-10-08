"""Training resource binding and recovery from the observed real API failure."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_ai_chat_selection import Fixtures
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkAgent import run_agent
from Utils.AIResources import resource_catalog
from Utils.AIWorkspace import read_json


class TrainingResourceTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.folder = self.training_folder('train35')
        self.resources = {'r1': {'kind': 'directory', 'name': 'train35', 'path': str(self.folder)}}

    def training_folder(self, name):
        folder = self.root / name
        folder.mkdir(parents=True)
        (folder / 'results.csv').write_text(
            'epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),train/box_loss,val/box_loss\n'
            '1,2,0.5,0.4,0.5,0.3,1.1,1.3\n2,4,0.8,0.7,0.8,0.6,0.7,1.0\n', encoding='utf-8')
        (folder / 'args.yaml').write_text('task: detect\nepochs: 3\nimgsz: 640\npatience: 100\n', encoding='utf-8')
        (folder / 'data.yaml').write_text('names: [划痕]\n', encoding='utf-8')
        return folder

    def tools(self):
        return WorkTools(self.root, self.paths, context={'resources': self.resources,
                         'current_paths': [], 'viewing_artifact': 'unrelated-image-preview'})

    def test_omitted_resource_reads_only_registered_training_run_even_in_preview(self):
        tools = self.tools()
        before = (self.folder / 'results.csv').read_bytes()
        result = tools.execute('training_runs', {})
        self.assertEqual(result['count'], 1)
        run = result['runs'][0]
        self.assertEqual(run['resource'], 'r1')
        self.assertEqual(run['name'], 'train35')
        self.assertEqual(run['context']['names'], ['划痕'])
        self.assertEqual(run['best']['epoch'], 2)
        self.assertEqual(run['epochs_recorded'], 2)
        self.assertTrue(run['trend'])
        self.assertEqual(before, (self.folder / 'results.csv').read_bytes())

    def test_artifact_id_error_points_to_existing_resource_instead_of_requesting_readd(self):
        with self.assertRaisesRegex(ValueError, 'r1：train35') as failure:
            self.tools().execute('training_runs', {'resource': '20260923_bad_artifact'})
        self.assertIn('无需用户重复添加', str(failure.exception))

    def test_multiple_training_directories_require_selection_and_project_is_explicit(self):
        second = self.training_folder('train36')
        self.resources['r2'] = {'kind': 'directory', 'name': 'train36', 'path': str(second)}
        tools = self.tools()
        with self.assertRaisesRegex(ValueError, '多个训练'):
            tools.execute('training_runs', {})
        self.assertEqual(tools.execute('training_runs', {'resource': 'r2'})['runs'][0]['name'], 'train36')
        with patch('Utils.AITrainingAnalysis.discover_runs', return_value=[]), self.assertRaisesRegex(ValueError, '当前工程'):
            tools.execute('training_runs', {'resource': 'project'})

    def test_empty_or_broken_runs_do_not_create_successful_zero_record_artifact(self):
        tools = self.tools()
        (self.folder / 'results.csv').write_text('invalid csv', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '读取失败'):
            tools.execute('training_runs', {'resource': 'r1'})
        (self.folder / 'results.csv').unlink()
        with self.assertRaisesRegex(ValueError, '未找到'):
            tools.execute('training_runs', {'resource': 'r1'})
        self.assertEqual(tools.store.list_entries(), [])

    def test_collection_counts_only_successfully_read_runs(self):
        parent = self.root / 'collection'
        good = self.training_folder('collection/trainA')
        bad = self.training_folder('collection/detect/trainB')
        (bad / 'results.csv').write_text('epoch\nnot-a-number\n', encoding='utf-8')
        self.resources = {'r1': {'kind': 'directory', 'name': 'collection', 'path': str(parent)}}
        result = self.tools().execute('training_runs', {})
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(Path(result['runs'][0]['folder']), good.resolve())

    def test_model_receives_typed_resources_and_can_correct_wrong_id(self):
        from test_ai_training_analysis import Fixtures as AnalysisFixtures
        report = AnalysisFixtures.response()
        report['next_experiment']['change'] = None
        prompts = []
        replies = iter([
            {'action': 'tool', 'tool': 'training_runs', 'args': {'resource': '20260923_artifact_id'}},
            {'action': 'tool', 'tool': 'training_runs', 'args': {'resource': 'r1'}},
            {'action': 'reply', 'message': '分析完成', 'analysis': report},
        ])
        def request(*args, **kwargs):
            prompts.append(args[3])
            return next(replies), {'usage': {'total_tokens': 10}}
        tools = self.tools()
        payload = run_agent(self.root, self.paths, {}, {}, [], '分析这次训练结果', ('artifact', 'preview', None),
                            ('mock-key', 'https://invalid.test', 'mock'), tools.context, request=request)
        self.assertIn('"role": "training_results"', prompts[0])
        self.assertIn('"latest_resource_id": "r1"', prompts[0])
        self.assertIn('r1：train35', prompts[1])
        self.assertIn('"epochs_recorded": 2', prompts[2])
        self.assertEqual(payload['work']['status'], 'completed')
        record = read_json(tools.store.location('runs', payload['run_id']) / 'run.json')
        self.assertEqual(record['resources']['r1']['role'], 'training_results')

    def test_resource_registration_is_visible_in_both_ui_and_conversation(self):
        from PyQt5.QtWidgets import QApplication
        from Utils.test import TestWindow
        app = QApplication.instance() or QApplication([])
        window = TestWindow(str(self.root))
        try:
            window.show_ai_chat()
            dock = window.ai_chat_dock
            key = dock.register_resource(str(self.folder), 'directory')
            self.assertEqual(key, 'r1')
            self.assertIn('训练结果', dock.resource_label.text())
            self.assertIn('train35', dock.history[-1]['content'])
            self.assertEqual(resource_catalog(dock.work_context()['resources'])['r1']['training_runs'], 1)
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
