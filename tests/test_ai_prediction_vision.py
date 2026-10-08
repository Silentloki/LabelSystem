"""Image evidence reaches the real serializer; only HTTP responses are stubbed."""
import copy
import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from test_ai_chat_selection import Fixtures, rect
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkAgent import run_agent
from Utils.AIWorkspace import Workspace, file_sha, read_json, write_json
from Utils.AIContextInspector import load_inspection
from Utils import AIPredictionVision as vision


def response(value):
    class Response(io.BytesIO):
        status = 200
    return Response(json.dumps({'choices': [{'message': {'content': json.dumps(value, ensure_ascii=False)},
                                           'finish_reason': 'stop'}]}, ensure_ascii=False).encode())


class PredictionVisionTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.store = Workspace(self.root)
        self.candidate_id, self.folder, self.manifest = self.store.create('candidates', '模型推理测试',
            {'model': 'best.pt', 'model_sha256': 'f' * 64, 'conf': .2, 'imgsz': 1280})
        self.rows = []
        for index, path in enumerate(self.paths[:4]):
            annotations = [] if index == 1 else [{**rect('脏污', .15, .2), 'confidence': .3 + index * .1}]
            self.rows.append({'path': path, 'image_sha256': file_sha(self.root / path),
                'document': {'image_width': 100, 'image_height': 200, 'annotations': annotations},
                'source_document': {'image_width': 100, 'image_height': 200, 'annotations': [rect('破损', .7, .8)]},
                'ignored_classes': 0})
        write_json(self.folder / 'candidates.json', self.rows)
        self.store.finish(self.folder, self.manifest, count=4, failed=0)
        self.context = {'viewing_artifact': self.candidate_id, 'selected_result_paths': [self.paths[0]],
                        'current_paths': [], 'current_result_count': 4, 'max_calls': 4}

    def tools(self):
        return WorkTools(self.root, self.paths, context=self.context)

    def prepare(self, tools=None, **kwargs):
        tools = tools or self.tools()
        result = tools.execute('inspect_predictions', kwargs)
        return tools, result

    def report(self, evidence):
        return {'summary': '仅就附送样本作观察，细节需核对。',
                'images': [{'image_id': item['image_id'], 'readable': True,
                    'suspected_issue': 'uncertain', 'finding': '测试图缺少实际缺陷纹理，无法确认预测正确。',
                    'recommendation': '请人工查看对应区域。'} for item in evidence['images']],
                'next_steps': ['核对疑点，不据此改动标注。']}

    def test_selected_only_prepares_original_overlay_and_details_without_source_changes(self):
        before = self.hashes()
        record_hash = file_sha(self.folder / 'candidates.json')
        tools, result = self.prepare(scope='selected')
        self.assertEqual(result['viewed_images'], 1)
        self.assertEqual(result['attachment_count'], 3)
        bundle = tools.prediction_analysis
        evidence = bundle['evidence']
        self.assertEqual(evidence['images'][0]['source'], self.paths[0])
        self.assertEqual(evidence['images'][0]['reference_annotations'], 0)
        raw, overlay, details = [Image.open(path).convert('RGB') for path in vision.image_paths(bundle)]
        self.assertEqual(raw.size, (100, 200))
        self.assertNotEqual(raw.tobytes(), overlay.tobytes())
        self.assertEqual(details.width, 1024)
        for image in (raw, overlay, details):
            image.close()
        self.assertEqual(before, self.hashes())
        self.assertEqual(record_hash, file_sha(self.folder / 'candidates.json'))
        self.assertEqual(tools.effects, [])

    def test_balanced_sample_discloses_unseen_images_and_reference_is_opt_in(self):
        tools, result = self.prepare()
        self.assertEqual(result['viewed_images'], 3)
        selected = tools.prediction_analysis['evidence']['images']
        self.assertIn(self.paths[1], [row['source'] for row in selected])
        self.assertEqual(result['not_viewed_images'], 1)
        self.assertTrue(all(row['reference_annotations'] == 0 for row in selected))
        tools, _ = self.prepare(scope='selected', with_reference=True)
        self.assertEqual(tools.prediction_analysis['evidence']['images'][0]['reference_annotations'], 1)

    def test_scope_errors_do_not_silently_sample_or_fallback(self):
        self.context['selected_result_paths'] = []
        with self.assertRaisesRegex(ValueError, '没有可查看'):
            self.prepare(scope='selected')
        self.context['selected_result_paths'] = self.paths[:4]
        self.assertEqual(self.prepare(scope='selected')[1]['viewed_images'], 4)
        for args in ({'scope': 'images', 'image_ids': ['images/not-in-batch.png']},
                     {'image_ids': self.paths[:1]}, {'limit': 11}, {'with_reference': 'true'}):
            with self.assertRaises(ValueError):
                self.prepare(**args)
        self.context['viewing_artifact'] = 'other'
        with self.assertRaisesRegex(ValueError, '选中图片不属于'):
            self.prepare(candidate_id=self.candidate_id, scope='selected')

    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows path identity')
    def test_windows_record_backslashes_match_ui_forward_slashes_without_rewriting_history(self):
        for row in self.rows:
            row['path'] = row['path'].replace('/', '\\')
        write_json(self.folder / 'candidates.json', self.rows)
        record_hash = file_sha(self.folder / 'candidates.json')
        before = self.hashes()
        self.context['selected_result_paths'] = self.paths[:2]
        for args in ({'scope': 'selected'}, {'scope': 'images', 'image_ids': self.paths[:2]},
                     {'scope': 'images', 'image_ids': [p.upper() for p in self.paths[:2]]}):
            tools, result = self.prepare(**args)
            self.assertEqual(result['viewed_images'], 2)
            self.assertEqual([r['source'] for r in tools.prediction_analysis['evidence']['images']],
                             [r['path'] for r in self.rows[:2]])
            self.assertTrue(vision.image_paths(tools.prediction_analysis))
        self.assertEqual(record_hash, file_sha(self.folder / 'candidates.json'))
        self.assertEqual(before, self.hashes())

    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows path identity')
    def test_equivalent_selected_paths_deduplicate_but_ambiguous_records_are_rejected(self):
        path = self.paths[0]
        tools, result = self.prepare(scope='images', image_ids=[path, path.replace('/', '\\'), path.upper()])
        self.assertEqual(result['viewed_images'], 1)
        duplicate = copy.deepcopy(self.rows[0])
        duplicate['path'] = path.replace('/', '\\')
        write_json(self.folder / 'candidates.json', self.rows + [duplicate])
        with self.assertRaisesRegex(ValueError, '重复图片引用'):
            self.prepare(scope='selected')

    def test_path_matching_never_guesses_by_basename_or_accepts_outside_project(self):
        for path in (Path(self.paths[0]).name, '../outside.png', str((self.root / self.paths[0]).resolve())):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.prepare(scope='images', image_ids=[path])

    def test_five_and_ten_explicit_images_ignore_old_small_sample_limit_eleven_rejected(self):
        for index in range(4, 11):
            path = f'images/limit{index}.png'
            Image.new('RGB', (100, 200), (index, 40, 80)).save(self.root / path)
            self.paths.append(path)
            row = copy.deepcopy(self.rows[0])
            row.update(path=path, image_sha256=file_sha(self.root / path))
            self.rows.append(row)
        write_json(self.folder / 'candidates.json', self.rows)
        manifest = read_json(self.folder / 'manifest.json')
        manifest['count'] = 11
        write_json(self.folder / 'manifest.json', manifest)
        paths = [r['path'] for r in self.rows]
        before = self.hashes()
        for count in (5, 10):
            self.context['selected_result_paths'] = paths[:count]
            for kwargs in ({'scope': 'selected'}, {'scope': 'selected', 'limit': 4},
                           {'scope': 'images', 'image_ids': paths[:count], 'limit': 3}):
                tools, result = self.prepare(**kwargs)
                self.assertEqual(result['viewed_images'], count)
                self.assertEqual([r['source'] for r in tools.prediction_analysis['evidence']['images']], paths[:count])
        self.context['selected_result_paths'] = paths
        for kwargs in ({'scope': 'selected'}, {'scope': 'images', 'image_ids': paths}):
            with self.assertRaisesRegex(ValueError, '最多看 10 张'):
                self.prepare(**kwargs)
        self.assertEqual(before, self.hashes())

    def test_changed_source_and_missing_fingerprint_block_visual_evidence(self):
        (self.root / self.paths[0]).write_bytes(b'changed')
        tools = self.tools()
        with self.assertRaisesRegex(ValueError, '原图已变化'):
            self.prepare(tools, scope='selected')
        self.assertIsNone(tools.prediction_analysis)
        self.rows[1].pop('image_sha256')
        write_json(self.folder / 'candidates.json', self.rows)
        with self.assertRaisesRegex(ValueError, '缺少指纹'):
            self.prepare(scope='images', image_ids=[self.paths[1]])

    def test_mutated_attachment_or_deleted_batch_cannot_be_sent(self):
        tools, _ = self.prepare(scope='selected')
        bundle = tools.prediction_analysis
        Path(vision.image_paths(bundle)[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '附件已改变'):
            vision.image_paths(bundle)
        tools, _ = self.prepare(scope='selected')
        manifest = read_json(self.folder / 'manifest.json')
        manifest['status'] = 'deleted'
        write_json(self.folder / 'manifest.json', manifest)
        with self.assertRaises(ValueError):
            vision.image_paths(tools.prediction_analysis)

    def test_polygon_and_no_prediction_evidence_keep_shape_and_image_identity(self):
        self.rows[0]['document']['annotations'] = [{'type': 'polygon', 'lable': '脏污', 'confidence': .6,
            'points': [{'x': .1, 'y': .1}, {'x': .8, 'y': .2}, {'x': .4, 'y': .7}]}]
        write_json(self.folder / 'candidates.json', self.rows)
        tools, result = self.prepare(scope='images', image_ids=self.paths[:2])
        self.assertEqual(result['attachment_count'], 5)
        evidence = tools.prediction_analysis['evidence']
        self.assertEqual(evidence['images'][0]['regions'][0]['label'], '脏污')
        self.assertEqual(evidence['images'][1]['predictions'], 0)
        overlay = Image.open(vision.image_paths(tools.prediction_analysis)[1]).convert('RGB')
        self.assertEqual(overlay.getpixel((40, 137)), (0, 215, 136))
        overlay.close()

    def test_report_requires_exact_attached_images_and_explicit_unreadability(self):
        tools, _ = self.prepare(scope='selected')
        bundle = tools.prediction_analysis
        report = self.report(bundle['evidence'])
        bad = copy.deepcopy(report)
        bad['images'][0]['image_id'] = 'I99'
        with self.assertRaises(ValueError):
            vision.complete(bundle, bad)
        bad = copy.deepcopy(report)
        bad['images'][0].update(readable=False, suspected_issue='possible_miss')
        with self.assertRaises(ValueError):
            vision.complete(bundle, bad)
        self.assertFalse((bundle['folder'] / 'analysis.json').exists())
        report['images'][0]['readable'] = False
        text = vision.complete(bundle, report)
        self.assertIn('1/4', text)
        self.assertIn('未查看 3', text)
        self.assertIn('尚未人工确认', text)
        self.assertEqual(read_json(bundle['folder'] / 'status.json')['readable_images'], 0)

    def test_real_serialized_request_contains_visual_attachments_and_report(self):
        sent = []
        def http(request, **kwargs):
            payload = json.loads(request.data)
            sent.append(payload)
            if len(sent) == 1:
                return response({'action': 'tool', 'tool': 'inspect_predictions',
                                 'args': {'candidate_id': 'current', 'scope': 'selected'}})
            content = payload['messages'][1]['content']
            evidence = json.loads(content[0]['text'].split('\n完整看图证据索引：\n')[1])
            self.assertEqual(len(content) - 1, 3)
            self.assertTrue(all(item['image_url']['url'].startswith('data:image/png;base64,') for item in content[1:]))
            return response({'action': 'reply', 'message': '完成', 'prediction_analysis': self.report(evidence)})
        before = self.hashes()
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=http):
            payload = run_agent(self.root, self.paths, {}, {}, [], '看看选中图片有没有预测问题',
                ('artifact', self.candidate_id, None), ('sk-test', 'https://example.invalid', 'vision-model'), self.context)
        self.assertEqual(payload['work']['status'], 'completed')
        self.assertEqual(len(sent), 2)
        self.assertIn('1/4', payload['result']['message'])
        inspection = load_inspection(self.root, payload['run_id'])
        self.assertEqual(len(inspection['requests'][1]['transports'][0]['images']), 3)
        evidence_file = next(self.store.location('runs', payload['run_id']).glob('prediction_vision/*/evidence.json'))
        evidence = read_json(evidence_file)
        self.assertEqual([row['sha256'] for row in inspection['requests'][1]['transports'][0]['images']],
                         [row['sha256'] for row in evidence['attachments']])
        self.assertEqual(read_json(evidence_file.with_name('status.json'))['status'], 'completed')
        self.assertEqual(before, self.hashes())

    def test_non_visual_provider_failure_does_not_fallback_to_fake_text_analysis(self):
        calls = []
        def http(request, **kwargs):
            calls.append(json.loads(request.data))
            if len(calls) == 1:
                return response({'action': 'tool', 'tool': 'inspect_predictions', 'args': {'scope': 'selected'}})
            raise urllib.error.HTTPError('https://example.invalid', 400, 'images unsupported', {}, io.BytesIO(b'image input unsupported'))
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=http):
            payload = run_agent(self.root, self.paths, {}, {}, [], '看图分析', ('artifact', self.candidate_id, None),
                ('sk-test', 'https://example.invalid', 'text-only'), self.context)
        self.assertNotEqual(payload['work']['status'], 'completed')
        self.assertEqual(len(calls), 2)
        self.assertFalse(list(self.store.location('runs', payload['run_id']).glob('prediction_vision/*/analysis.json')))
        status = next(self.store.location('runs', payload['run_id']).glob('prediction_vision/*/status.json'))
        self.assertEqual(read_json(status)['status'], 'not_completed')


if __name__ == '__main__':
    unittest.main()
