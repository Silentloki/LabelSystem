"""Saved inference analysis, including the reported r3 -> training misroute."""
import json
import unittest
from unittest.mock import patch

import torch
from test_ai_chat_selection import Fixtures, rect
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkAgent import run_agent
from Utils.AIWorkspace import Workspace, file_sha, read_json, write_json


class PredictionAnalysisTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.store = Workspace(self.root)
        self.resources = {}
        for key in ('r1', 'r3'):
            path = self.root / key / 'best.pt'
            path.parent.mkdir()
            path.write_bytes(key.encode())  # Never loaded as a model.
            self.resources[key] = {'kind': 'model', 'name': 'best.pt', 'path': str(path)}
        self.first = self.batch('r1')
        self.third = self.batch('r3')

    def batch(self, model):
        path = self.resources[model]['path']
        key, folder, manifest = self.store.create('candidates', '模型推理', {
            'model': 'best.pt', 'model_path': path, 'model_resource': model,
            'model_sha256': file_sha(path), 'conf': .5, 'imgsz': 1280,
            'class_filter': 'project_classes'})
        write_json(folder / 'candidates.json', [
            {'path': self.paths[0], 'document': {'annotations': [
                {**rect('脏污', .1, .2), 'confidence': .6},
                {**rect('脏污', .2, .3), 'confidence': .8}, rect('破损', .2, .3)]}, 'ignored_classes': 2},
            {'path': self.paths[1], 'document': {'annotations': []}, 'ignored_classes': 1}])
        self.store.finish(folder, manifest, count=2, failed=1)
        return key

    def tools(self, current=None):
        return WorkTools(self.root, self.paths, context={'resources': self.resources,
                         'viewing_artifact': current, 'current_paths': self.paths})

    def test_same_name_models_resolve_by_content_and_summary_is_read_only(self):
        before = {str(p): file_sha(p) for p in self.root.rglob('*') if p.is_file()}
        tools = self.tools(self.first)
        result = tools.execute('prediction_results', {'model': 'r3', 'limit': 1})
        self.assertEqual(result['candidate_id'], self.third)
        self.assertEqual(result['summary'], {'successful_images': 2, 'failed_images': 1,
            'images_with_predictions': 1, 'images_without_predictions': 1, 'predictions': 3,
            'ignored_predictions': 3, 'legacy_unmapped_images': 0})
        self.assertEqual(result['parameters']['imgsz'], 1280)
        classes = {row['label']: row for row in result['classes']}
        self.assertAlmostEqual(classes['脏污']['confidence_mean'], .7)
        self.assertEqual(classes['脏污']['images'], 1)
        self.assertEqual(classes['破损']['confidence_missing'], 1)
        self.assertIsNone(classes['破损']['confidence_mean'])
        self.assertTrue(result['has_more'])
        page = tools.execute('prediction_results', {'candidate_id': self.third, 'offset': 1})
        self.assertEqual(page['images'][0]['predictions'], 0)
        self.assertEqual(page['summary'], result['summary'])
        self.assertFalse(result['image_pixels_sent'])
        self.assertEqual(before, {str(p): file_sha(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(tools.effects, [])

    def test_ambiguous_batches_require_choice_but_matching_current_is_used(self):
        another = self.batch('r3')
        result = self.tools().execute('prediction_results', {'model': 'r3', 'limit': 1})
        self.assertEqual(result['status'], 'selection_required')
        self.assertEqual(result['matching_batches'], 2)
        self.assertTrue(result['has_more'])
        self.assertEqual(self.tools(self.third).execute('prediction_results', {'model': 'r3'})['candidate_id'], self.third)
        self.assertEqual(self.tools().execute('prediction_results', {'candidate_id': another})['candidate_id'], another)
        with self.assertRaises(ValueError):
            self.tools().execute('prediction_results', {'candidate_id': self.first, 'model': 'r3'})

    def test_replaced_missing_weights_do_not_rewrite_or_misidentify_old_batch(self):
        from pathlib import Path
        path = Path(self.resources['r3']['path'])
        old = file_sha(path)
        path.write_bytes(b'a different version')
        self.assertEqual(self.tools(self.third).execute('prediction_results', {'model': 'r3'})['status'], 'not_found')
        path.unlink()
        with self.assertRaisesRegex(ValueError, '已不可用'):
            self.tools().execute('prediction_results', {'model': 'r3'})
        result = self.tools(self.third).execute('prediction_results', {'candidate_id': 'current'})
        self.assertEqual(result['model']['sha256'], old)

    def test_legacy_model_identity_from_saved_plan_and_deleted_results(self):
        folder, manifest = self.store.artifact(self.third)
        original = dict(manifest['metadata'])
        plan_id, plan_folder, plan = self.store.create('prediction_plan', '旧方案', original)
        self.store.finish(plan_folder, plan)
        manifest['metadata'] = {'plan_id': plan_id, 'model_sha256': original['model_sha256']}
        write_json(folder / 'manifest.json', manifest)
        self.assertEqual(self.tools().execute('prediction_results', {'model': 'r3'})['candidate_id'], self.third)
        manifest['status'] = 'deleted'
        write_json(folder / 'manifest.json', manifest)
        self.assertEqual(self.tools().execute('prediction_results', {'model': 'r3'})['status'], 'not_found')
        with self.assertRaises(ValueError):
            self.tools().execute('prediction_results', {'candidate_id': self.third})

    def test_no_current_wrong_type_and_invalid_paging_do_not_broaden_scope(self):
        with self.assertRaisesRegex(ValueError, '当前未打开'):
            self.tools().execute('prediction_results', {'candidate_id': 'current'})
        key, folder, manifest = self.store.create('crops', '切图')
        self.store.finish(folder, manifest)
        with self.assertRaises(ValueError):
            self.tools(key).execute('prediction_results', {})
        for args in ({'limit': 0}, {'offset': -1}, {'candidate_id': '../outside'}, {'model': 'unknown'}):
            with self.assertRaises(ValueError):
                self.tools().execute('prediction_results', args)

    def test_empty_predictions_and_unknown_failure_count_are_not_good_samples(self):
        folder, manifest = self.store.artifact(self.third)
        manifest.pop('failed')
        write_json(folder / 'manifest.json', manifest)
        write_json(folder / 'candidates.json', [{'path': self.paths[0], 'document': {'annotations': []}}])
        result = self.tools(self.third).execute('prediction_results', {})
        self.assertEqual(result['summary']['images_without_predictions'], 1)
        self.assertIsNone(result['summary']['failed_images'])
        self.assertEqual(result['classes'], [])
        self.assertIn('无预测不等于良品', result['note'])

    def test_observed_wrong_tool_can_recover_to_actual_prediction_evidence(self):
        prompts = []
        replies = iter([
            {'action': 'tool', 'tool': 'training_runs', 'args': {'resource': 'r3'}},
            {'action': 'tool', 'tool': 'prediction_results', 'args': {'model': 'r3'}},
            {'action': 'reply', 'message': '读取了保存的预测统计，未进行看图判断。'}])
        def request(*args, **kwargs):
            prompts.append(args[3])
            self.assertEqual(args[4], [])
            return next(replies), {}
        with patch('Utils.AITrainingAnalysis.discover_runs', side_effect=AssertionError('must not scan training')):
            payload = run_agent(self.root, self.paths, {}, {}, [], '分析r3的推理结果',
                ('artifact', self.third, None), ('mock', 'https://invalid.test', 'mock'),
                self.tools(self.third).context, request=request)
        self.assertEqual(payload['work']['status'], 'completed')
        self.assertEqual([s['tool'] for s in payload['work']['steps']], ['prediction_results'])
        self.assertIn('该资源是推理模型', prompts[1])
        injected = json.loads(prompts[2].split('\n工作上下文：\n')[1])
        result = injected['tool_results_this_turn'][-1]['result']
        self.assertEqual(result['candidate_id'], self.third)
        self.assertEqual(result['summary']['predictions'], 3)
        self.assertEqual(result['parameters']['imgsz'], 1280)
        self.assertIn('prediction_results', prompts[0])


if __name__ == '__main__':
    unittest.main()
