"""Selected-photo actions: isolated project data, stubbed HTTP, real Qt widgets."""
import copy
import json
import pickle
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from PyQt5.QtCore import Qt, QItemSelectionModel
from PyQt5.QtWidgets import QApplication, QComboBox, QDialogButtonBox

from test_ai_chat_selection import Fixtures, rect
from test_ai_prediction_vision import response
from Utils.AIWorkspace import Workspace, file_sha, read_json, write_json
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkAgent import run_agent
from Utils.AIResultBrowser import visual_rows
from Utils.AIResultActions import target_rows, ui_target
from Utils.AIResultStorage import delete_result, deletion_preview
from Utils.ProjectContext import ProjectContext
from Utils import AISelectedAnalysis as vision


class ActionFixtures(Fixtures):
    def setUp(self):
        super().setUp()
        self.store = Workspace(self.root)
        self.aid, self.folder, manifest = self.store.create('candidates', '测试预测',
            {'model': 'best.pt', 'model_sha256': 'a' * 64, 'imgsz': 1280, 'conf': .2})
        rows = []
        for p in self.paths[:4]:
            doc = copy.deepcopy(self.documents[Path(p).stem])
            rows.append({'path': p, 'image_sha256': file_sha(self.root / p),
                         'document': doc, 'source_document': doc})
        write_json(self.folder / 'candidates.json', rows)
        write_json(self.folder / 'errors.json', [])
        self.store.finish(self.folder, manifest, count=4, predicted_annotations=6, failed=0)
        self.context = {'viewing_artifact': self.aid, 'selected_result_paths': self.paths[:2],
                        'selected_paths': [self.paths[3]], 'current_paths': [], 'max_calls': 8,
                        'interaction_target': {'artifact_id': self.aid, 'images': self.paths[:2],
                                               'mode': 'selected', 'count': 2, 'invalid': False}}

    def tools(self, message='把这些图加入问题样本集'):
        return WorkTools(self.root, self.paths, context={**copy.deepcopy(self.context), 'user_request': message})

    @staticmethod
    def report(page):
        return [{'image_id': r['image_id'], 'readable': True,
                 'finding': '测试图无真实纹理，无法确认缺陷。', 'recommendation': '人工复核。'} for r in page['images']]


class ResultActionTests(ActionFixtures):
    def test_error_set_reopens_and_deletion_only_removes_references(self):
        before = self.hashes()
        prediction_hash = file_sha(self.folder / 'candidates.json')
        tools = self.tools()
        result = tools.execute('add_error_set', {})
        folder, _ = self.store.artifact(result['artifact_id'])
        self.assertEqual({p.name for p in folder.iterdir()}, {'manifest.json', 'references.json'})
        _, rows = visual_rows(Workspace(self.root), result['artifact_id'])
        self.assertEqual([r['source'] for r in rows], self.paths[:2])
        self.assertTrue(all(r['source_artifact'] == self.aid for r in rows))
        self.assertTrue(self.tools().execute('add_error_set', {})['reused_result'])
        delete_result(self.store, result['artifact_id'], deletion_preview(self.store, result['artifact_id'])['token'])
        self.assertEqual(before, self.hashes())
        self.assertEqual(prediction_hash, file_sha(self.folder / 'candidates.json'))

    def test_changed_image_changed_result_and_deleted_parent_are_unavailable(self):
        result = self.tools().execute('add_error_set', {})
        rows = read_json(self.folder / 'candidates.json')
        changed = copy.deepcopy(rows)
        changed[0]['document']['annotations'] = []
        write_json(self.folder / 'candidates.json', changed)
        with self.assertRaisesRegex(ValueError, '已变化'):
            visual_rows(self.store, result['artifact_id'])
        write_json(self.folder / 'candidates.json', rows)
        manifest = read_json(self.folder / 'manifest.json')
        manifest['status'] = 'deleted'
        write_json(self.folder / 'manifest.json', manifest)
        with self.assertRaises(ValueError):
            visual_rows(self.store, result['artifact_id'])

    def test_feedback_from_user_is_bound_atomic_and_not_annotation_edit(self):
        before = self.hashes()
        tools = self.tools('把这些图记为漏检')
        self.assertEqual(tools.execute('record_feedback', {'category': 'false_negative'})['count'], 2)
        tools.execute('record_feedback', {'category': 'false_negative'})
        records = ProjectContext(self.root).feedback()
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r['artifact'] == self.aid and r['author'] == 'human' and
                            r['provenance']['via'] == 'chat_user_instruction' for r in records))
        self.assertEqual(before, self.hashes())

    def test_conflicting_second_image_rolls_back_first(self):
        context = ProjectContext(self.root)
        context.add_feedback(self.paths[1], 'correct', artifact_id=self.aid)
        with self.assertRaisesRegex(ValueError, '相反'):
            self.tools('把这些图记为漏检').execute('record_feedback', {'category': 'false_negative'})
        self.assertEqual(context.feedback('false_negative'), [])
        self.assertEqual(len(context.feedback('correct')), 1)

    def test_analysis_negation_condition_and_ai_category_cannot_be_human_feedback(self):
        for message in ['分析有没有漏检', '不要把这些图记为漏检', '如果有漏检就记录',
                        '把这些图记为疑似漏检', '把这些图记为误检', '能否记为漏检？']:
            with self.subTest(message=message), self.assertRaises(ValueError):
                self.tools(message).execute('record_feedback', {'category': 'false_negative'})
        self.assertEqual(ProjectContext(self.root).feedback(), [])
        with self.assertRaises(ValueError):
            self.tools('分析这些图的共同原因').execute('add_error_set', {})

    def test_empty_invalid_cross_batch_and_hidden_original_never_fallback(self):
        for target in [{'artifact_id': self.aid, 'images': []},
                       {'artifact_id': self.aid, 'images': self.paths[:1], 'invalid': True},
                       {'artifact_id': 'other', 'images': self.paths[:1]},
                       {'artifact_id': self.aid, 'images': self.paths[4:]}]:
            self.context['interaction_target'] = target
            with self.assertRaises(ValueError):
                self.tools().execute('add_error_set', {})
        self.assertFalse(any(r['kind'] == 'error_set' for r in self.store.list_entries()))

    def test_original_and_generated_results_keep_their_own_sources(self):
        self.context.update(viewing_artifact=None, interaction_target={'artifact_id': None, 'images': self.paths[:1]})
        result = self.tools().execute('add_error_set', {})
        self.assertIsNone(visual_rows(self.store, result['artifact_id'])[1][0]['source_artifact'])
        crop = WorkTools(self.root, self.paths[:4]).execute('crop', {'scope': 'all', 'mode': 'tiles', 'tile_size': 128, 'overlap': 0})
        _, rows = visual_rows(self.store, crop['artifact_id'])
        selected = [Path(rows[0]['image']).relative_to(self.root).as_posix()]
        self.context.update(viewing_artifact=crop['artifact_id'], interaction_target={'artifact_id': crop['artifact_id'], 'images': selected})
        tools = self.tools()
        result = tools.execute('add_error_set', {})
        saved = visual_rows(self.store, result['artifact_id'])[1][0]
        self.assertEqual(saved['image'], rows[0]['image'])
        self.assertEqual(saved['source_artifact'], crop['artifact_id'])

    def test_feedback_in_error_set_resolves_original_prediction_batch(self):
        result = self.tools().execute('add_error_set', {})
        self.context.update(viewing_artifact=result['artifact_id'],
                            interaction_target={'artifact_id': result['artifact_id'], 'images': self.paths[:1]})
        self.tools('把这张记为漏检').execute('record_feedback', {'category': 'false_negative'})
        self.assertEqual(ProjectContext(self.root).feedback()[0]['artifact'], self.aid)

    def test_selected_visual_evidence_and_complete_report(self):
        before = self.hashes()
        tools = self.tools('分析这些图')
        page = tools.execute('inspect_selection', {})
        self.assertEqual([r['image'] for r in page['images']], self.paths[:2])
        self.assertTrue(vision.image_paths(tools.selection_analysis))
        with self.assertRaisesRegex(ValueError, '覆盖'):
            vision.complete(tools.selection_analysis, {'summary': 'test', 'images': self.report(page)[:1]})
        answer = vision.complete(tools.selection_analysis, {'summary': '没有足够纹理判断。', 'images': self.report(page)})
        self.assertIn('2/2', answer)
        self.assertEqual(ProjectContext(self.root).feedback(), [])
        self.assertEqual(before, self.hashes())

    def test_twenty_selected_images_are_covered_page_by_page_over_actual_serializer(self):
        paths = []
        for index in range(20):
            path = f'images/extra{index}.png'
            Image.new('RGB', (100, 200), (index, 80, 90)).save(self.root / path)
            write_json(self.root / 'jsons' / f'extra{index}.json', {'image_width': 100, 'image_height': 200, 'annotations': []})
            paths.append(path)
        all_paths = self.paths + paths
        (self.root / 'datafile.dat').write_bytes(pickle.dumps(all_paths))
        context = {'interaction_target': {'artifact_id': None, 'images': paths, 'count': 20, 'mode': 'selected'},
                   'selected_paths': paths, 'current_paths': all_paths, 'max_calls': 8}
        calls, attached = [], []
        def http(request, **kwargs):
            body = json.loads(request.data)
            content = body['messages'][-1]['content']
            prompt = next(v['text'] for v in content if v['type'] == 'text') if isinstance(content, list) else content
            images = [v for v in content if v['type'] == 'image_url'] if isinstance(content, list) else []
            calls.append(body)
            if len(calls) == 1:
                self.assertIn('interaction_target', prompt)
                return response({'action': 'tool', 'tool': 'inspect_selection', 'args': {}})
            page = json.loads(prompt.split('本批证据：')[1].split('\n此前已通过校验')[0])
            self.assertEqual(len(page['images']), 10)
            self.assertEqual(len(images), 20)
            self.assertGreaterEqual(body['max_tokens'], 7500)
            import base64, hashlib
            for image, evidence in zip(images, page['attachments']):
                raw = base64.b64decode(image['image_url']['url'].split(',')[1])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), evidence['sha256'])
            attached.extend(row['image'] for row in page['images'])
            report = self.report(page)
            if page['next_offset'] < 20:
                return response({'action': 'tool', 'tool': 'inspect_selection',
                                 'args': {'offset': page['next_offset'], 'previous_report': report}})
            return response({'action': 'reply', 'message': '完成',
                             'selection_analysis': {'summary': '20张合成图的共同观察，仅验证链路。', 'images': report}})
        with patch('urllib.request.urlopen', side_effect=http):
            result = run_agent(self.root, all_paths, {}, {}, [], '分析这20张的共同原因',
                               ('all', None, None), ('fake', 'https://example.invalid', 'test'), context=context)
        self.assertEqual(result['work']['status'], 'completed', result)
        self.assertEqual(len(calls), 3)
        self.assertEqual(attached, paths)
        self.assertIn('20/20', result['result']['message'])

    def test_visual_partial_budget_and_stale_attachment_are_truthful(self):
        tools = self.tools()
        page = tools.execute('inspect_selection', {})
        path = Path(vision.image_paths(tools.selection_analysis)[0])
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '附件已改变'):
            vision.image_paths(tools.selection_analysis)
        note = vision.mark_unfinished(tools.selection_analysis, 'limit_reached')
        self.assertIn('0/2', note)

    def test_missing_annotation_invalidates_original_problem_set(self):
        self.context.update(viewing_artifact=None, interaction_target={'artifact_id': None, 'images': self.paths[:1]})
        result = self.tools().execute('add_error_set', {})
        (self.root / 'jsons' / 'small.json').unlink()
        with self.assertRaisesRegex(ValueError, '标注文件状态已变化'):
            visual_rows(self.store, result['artifact_id'])

    def test_partial_analysis_is_reported_as_partial_not_full_success(self):
        for index in range(6):
            path = f'images/partial{index}.png'
            Image.new('RGB', (100, 200), (index, 40, 80)).save(self.root / path)
            self.paths.append(path)
        context = {'viewing_artifact': None, 'interaction_target': {'artifact_id': None, 'images': self.paths},
                   'selected_paths': self.paths, 'max_calls': 3}
        def request(*args, **kwargs):
            prompt = args[3]
            if not args[4]:
                return {'action': 'tool', 'tool': 'inspect_selection', 'args': {}}, {}
            page = json.loads(prompt.split('本批证据：')[1].split('\n此前已通过校验')[0])
            return {'action': 'reply', 'message': 'test', 'selection_analysis': {
                'summary': '只分析了前十张。', 'images': self.report(page)}}, {}
        result = run_agent(self.root, self.paths, {}, {}, [], '分析这些图', ('all', None, None),
                           ('fake', 'https://example.invalid', 'test'), context=context, request=request)
        self.assertEqual(result['work']['status'], 'partial')
        self.assertIn('未查看 1 张', result['result']['message'])

    def test_agent_records_only_actual_message_and_preserves_target_in_resume(self):
        calls = []
        def request(*args, **kwargs):
            calls.append(args[3])
            if len(calls) == 1:
                return {'action': 'tool', 'tool': 'record_feedback', 'args': {'category': 'false_negative'}}, {}
            return {'action': 'reply', 'message': '已记录两张。'}, {}
        result = run_agent(self.root, self.paths, {}, {}, [], '把这些图记为漏检', ('artifact', self.aid, None),
                           ('fake', 'https://example.invalid', 'test'), context=self.context, request=request)
        self.assertEqual(result['work']['status'], 'completed')
        self.assertEqual(result['work']['continuation']['interaction_target'], self.context['interaction_target'])
        self.assertTrue(all(r['note'] == '把这些图记为漏检' for r in ProjectContext(self.root).feedback()))

    def test_page_skip_and_source_change_reject_subsequent_evidence(self):
        tools = self.tools()
        page = tools.execute('inspect_selection', {})
        self.assertEqual(page['source_batches'][0]['parameters']['imgsz'], 1280)
        with self.assertRaises(ValueError):
            tools.execute('inspect_selection', {'offset': 99, 'previous_report': self.report(page)})
        rows = read_json(self.folder / 'candidates.json')
        rows[0]['document']['annotations'] = []
        write_json(self.folder / 'candidates.json', rows)
        with self.assertRaisesRegex(ValueError, '已变化'):
            vision.image_paths(tools.selection_analysis)


class ResultActionUITests(ActionFixtures):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from Utils.test import TestWindow
        self.window = TestWindow(str(self.root))
        self.browser = self.window.result_browser
        self.browser.open(self.aid)
        self.window.show_ai_chat()
        self.chat = self.window.ai_chat_dock
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def test_multiselect_current_fallback_and_failed_selection(self):
        self.browser.list.clearSelection()
        self.browser.list.setCurrentRow(2, QItemSelectionModel.NoUpdate)
        self.chat.refresh_scope()
        self.assertIn('当前查看 1 张', self.chat.scope_label.text())
        self.assertEqual(self.chat.work_context()['selected_result_paths'], [self.paths[2]])
        self.browser.list.item(0).setSelected(True)
        self.browser.list.item(3).setSelected(True)
        self.assertEqual(self.chat.work_context()['selected_result_paths'], [self.paths[0], self.paths[3]])
        self.assertIn('已选中 2 张', self.chat.scope_label.text())
        self.browser.rows[3] = {'name': 'failed.png', 'error': 'failed'}
        self.assertTrue(ui_target(self.window)['invalid'])
        self.assertEqual(self.chat.work_context()['selected_result_paths'], [])

    def test_context_menu_preserves_selection_and_error_set_reopens(self):
        self.browser.list.clearSelection()
        self.browser.list.item(0).setSelected(True)
        self.browser.list.item(2).setSelected(True)
        self.browser.ask_ai()
        self.assertEqual(self.chat.work_context()['selected_result_paths'], [self.paths[0], self.paths[2]])
        tools = WorkTools(self.root, self.paths, context={**self.chat.work_context(), 'user_request': '加入error set'})
        result = tools.execute('add_error_set', {})
        self.assertTrue(self.browser.open(result['artifact_id']))
        self.assertEqual(self.browser.list.count(), 2)
        self.assertTrue(self.browser.view.scene().items())
        self.assertFalse(self.browser.controls.isVisible())

    def test_error_set_manual_feedback_uses_prediction_source(self):
        from Utils.ProjectContextDialog import record_feedback
        result = self.tools().execute('add_error_set', {})
        def save(dialog):
            combo = dialog.findChild(QComboBox)
            combo.setCurrentIndex(combo.findData('false_negative'))
            self.assertEqual(combo.currentData(), 'false_negative')
            dialog.findChild(QDialogButtonBox).accepted.emit()
            return 1
        with patch('Utils.ProjectContextDialog.QDialog.exec_', save):
            record_feedback(self.window, self.paths[:2], result['artifact_id'])
        self.assertTrue(all(r['artifact'] == self.aid for r in ProjectContext(self.root).feedback()))
        self.assertEqual(len(ProjectContext(self.root).feedback()), 2)


if __name__ == '__main__':
    unittest.main()
