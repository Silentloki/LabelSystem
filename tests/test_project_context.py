"""Project context integration uses isolated files and mocked model requests only."""
import copy
import json
import os
import shutil
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from test_ai_chat_selection import Fixtures
from Utils.ProjectContext import ProjectContext
from Utils.AIWorkspace import Workspace, write_json, file_sha
from Utils.AIWorkTools import WorkTools


class ContextTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.context = ProjectContext(self.root)

    def candidate(self):
        store = Workspace(self.root)
        aid, folder, manifest = store.create('candidates', '测试推理批次', {'model': 'test.pt', 'conf': .5})
        rows = []
        for path in self.paths[:2]:
            doc = copy.deepcopy(self.documents[Path(path).stem])
            rows.append({'path': path, 'image_sha256': file_sha(self.root / path),
                         'source_document': doc, 'document': doc, 'can_adopt': False})
        write_json(folder / 'candidates.json', rows)
        write_json(folder / 'errors.json', [])
        store.finish(folder, manifest, count=2, failed=0, predicted_annotations=3)
        return aid

    def test_registry_preserves_legacy_aliases_and_shares_across_conversations(self):
        model = self.root / 'example.pt'
        model.write_bytes(b'fixture model, never loaded')
        pid = self.context.register_resource(model, 'model')
        merged = self.context.merge_resources({'r1': {'kind': 'directory', 'path': str(self.root / 'images'), 'name': 'images'}})
        self.assertIn(pid, merged)
        self.assertEqual(merged['r1']['kind'], 'directory')
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(self.context.merge_resources(merged)), 2)
        self.assertIn(pid, ProjectContext(self.root).merge_resources({}))

    def test_project_move_preserves_relative_resource_and_feedback(self):
        model = self.root / 'model.pt'
        model.write_bytes(b'model')
        pid = self.context.set_model(model, 'preferred_model')
        self.context.add_feedback(self.paths[0], 'annotation_question')
        destination = self.root / 'moved_project'
        destination.mkdir()
        for name in ('images', 'jsons', 'ai_workbench'):
            shutil.copytree(self.root / name, destination / name)
        for name in ('model.pt', 'datafile.dat', 'flagfile.dat', 'label.txt'):
            shutil.copy2(self.root / name, destination / name)
        moved = ProjectContext(destination)
        self.assertEqual(Path(moved.resources()[pid]['path']), (destination / 'model.pt').resolve())
        self.assertFalse(moved.feedback()[0]['stale'])
        self.assertEqual(moved.settings()['preferred_model']['resource_id'], pid)

    def test_dataset_split_missing_test_and_yaml_refresh(self):
        folder = self.root / 'dataset'
        (folder / 'images/train').mkdir(parents=True)
        (folder / 'images/val').mkdir()
        config = folder / 'data.yaml'
        config.write_text('train: images/train\nval: images/val\nnames: [defect]\n', encoding='utf-8')
        state = self.context.inventory()
        dataset = state['datasets'][0]
        self.assertTrue(dataset['split_status']['train'][0]['available'])
        self.assertEqual(dataset['splits']['test'], [])
        self.assertEqual(state['summary']['images'], 5)
        config.write_text('train: missing\nnames: [changed]\n', encoding='utf-8')
        updated = self.context.inventory()['datasets'][0]
        self.assertEqual(updated['classes'], ['changed'])
        self.assertFalse(updated['split_status']['train'][0]['available'])

    def test_result_lineage_deleted_and_missing_exports(self):
        aid = self.candidate()
        store = Workspace(self.root)
        _, folder, manifest = store.create('export', '示例导出', {'source_artifact': aid})
        result = store.finish(folder, manifest, count=2, source_artifact=aid)
        child = self.context.artifact(result['artifact_id'])
        self.assertEqual(child['source_artifact'], aid)
        parent_folder, parent = store.artifact(aid)
        parent['status'] = 'deleted'
        write_json(parent_folder / 'manifest.json', parent)
        self.assertFalse(self.context.artifact(result['artifact_id'])['available'])

    def test_human_prediction_feedback_is_scoped_deduplicated_and_selectable(self):
        aid = self.candidate()
        before = self.hashes()
        fid = self.context.add_feedback(self.paths[0], 'false_positive', '人工确认', aid)
        self.assertEqual(self.context.add_feedback(self.paths[0], 'false_positive', '补充说明', aid), fid)
        other = self.candidate()
        self.context.add_feedback(self.paths[0], 'false_negative', '另一模型漏检', other)
        result = self.context.query('feedback', category='false_positive', day='today', artifact_id=aid)
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['batches'][0]['marked_images'], 1)
        tools = WorkTools(self.root, self.paths)
        chosen = tools.execute('select_feedback', {'category': 'false_positive', 'artifact_id': aid})
        self.assertEqual(chosen['count'], 1)
        self.assertEqual(tools.scope('last'), [self.paths[0]])
        self.assertEqual(before, self.hashes())
        self.context.resolve_feedback(fid)
        self.assertEqual(self.context.query('feedback', category='false_positive')['total'], 0)

    def test_feedback_requires_prediction_batch_and_rejects_conflicting_judgment(self):
        with self.assertRaises(ValueError):
            self.context.add_feedback(self.paths[0], 'false_positive')
        aid = self.candidate()
        self.context.add_feedback(self.paths[0], 'false_positive', artifact_id=aid)
        with self.assertRaises(ValueError):
            self.context.add_feedback(self.paths[0], 'correct', artifact_id=aid)
        with self.assertRaises(ValueError):
            WorkTools(self.root, self.paths).execute('select_feedback', {'category': 'false_positive'})
        with self.assertRaises(ValueError):
            self.context.add_feedback('../outside.png', 'image_quality')

    def test_changed_prediction_image_or_deleted_parent_is_not_reused(self):
        aid = self.candidate()
        self.context.add_feedback(self.paths[0], 'false_positive', artifact_id=aid)
        (self.root / self.paths[0]).write_bytes(b'changed image')
        self.assertTrue(self.context.feedback()[0]['stale'])
        with self.assertRaises(ValueError):
            WorkTools(self.root, self.paths).execute('select_feedback', {'category': 'false_positive', 'artifact_id': aid})

    def test_json_reformat_keeps_feedback_but_actual_change_invalidates(self):
        self.context.add_feedback(self.paths[0], 'annotation_question')
        path = self.root / 'jsons/small.json'
        doc = json.loads(path.read_text(encoding='utf-8'))
        path.write_text(json.dumps(doc, indent=4, sort_keys=True), encoding='utf-8')
        self.assertFalse(self.context.feedback()[0]['stale'])
        doc['annotations'] = []
        write_json(path, doc)
        self.assertTrue(self.context.feedback()[0]['stale'])
        self.context.add_feedback(self.paths[0], 'annotation_question', '重新记录')
        self.assertEqual(len(self.context.feedback()), 2)

    def test_main_model_is_separate_and_replaced_weights_require_reselection(self):
        model = self.root / 'model.pt'
        model.write_bytes(b'model1')
        pid = self.context.set_model(model)
        self.assertNotIn('preferred_model', self.context.settings())
        self.context.set_model(model, 'preferred_model')
        model.write_bytes(b'changed weights')
        state = self.context.inventory()['summary']['model_roles']
        self.assertTrue(state['preferred_model']['changed_since_selected'])
        tools = WorkTools(self.root, self.paths)
        with self.assertRaisesRegex(ValueError, '替换'):
            tools.resource(pid, 'model')

    def test_date_pagination_and_partial_counts(self):
        for path in self.paths[:3]:
            self.context.add_feedback(path, 'image_quality')
        rows = self.context.query('feedback', day='today', limit=2)
        self.assertEqual(rows['total'], 3)
        self.assertEqual(rows['next_offset'], 2)
        self.assertEqual(len(self.context.query('feedback', offset=2, limit=2)['items']), 1)
        self.assertEqual(self.context.query('feedback', day='yesterday')['total'], 0)
        with self.assertRaises(ValueError):
            self.context.query('feedback', day='not-a-date')

    def test_corrupt_context_is_not_replaced(self):
        self.context.path.parent.mkdir(parents=True)
        self.context.path.write_bytes(b'not sqlite')
        with self.assertRaises(sqlite3.DatabaseError):
            self.context.resources()
        self.assertEqual(self.context.path.read_bytes(), b'not sqlite')

    def test_agent_can_query_and_select_human_batch_without_paths_in_message(self):
        from Utils.AIWorkAgent import run_agent
        aid = self.candidate()
        self.context.add_feedback(self.paths[0], 'false_positive', artifact_id=aid)
        prompts = []
        actions = iter([
            {'action': 'tool', 'tool': 'project_context', 'args': {'topic': 'feedback', 'category': 'false_positive', 'day': 'today'}},
            {'action': 'tool', 'tool': 'select_feedback', 'args': {'category': 'false_positive', 'artifact_id': aid}},
            {'action': 'reply', 'message': '已找到今天人工记录误检的图片。'}])
        def request(*args, **kwargs):
            prompts.append(args[3])
            return next(actions), {}
        result = run_agent(self.root, self.paths, {}, {}, [], '找出今天记过误检的图片', ('all', None, None),
                           ('fake', 'invalid', 'mock'), request=request)
        self.assertEqual(result['work']['status'], 'completed')
        self.assertEqual(result['work']['effects'][0]['paths'], [self.paths[0]])
        self.assertIn('project_state', prompts[0])
        self.assertIn('human', prompts[1])

    def test_empty_problem_query_never_falls_back_to_previous_result(self):
        aid = self.candidate()
        tools = WorkTools(self.root, self.paths)
        tools.last_artifact = aid
        tools.last_scope = self.paths
        tools.export_result = aid
        result = tools.execute('select_feedback', {'category': 'false_positive', 'artifact_id': aid})
        self.assertEqual(result['count'], 0)
        self.assertEqual(tools.scope('last'), [])
        self.assertIsNone(tools.last_artifact)
        self.assertIsNone(tools.export_result)

    def test_processed_image_feedback_preserves_result_scope(self):
        tools = WorkTools(self.root, self.paths, context={'current_paths': self.paths[:1]})
        result = tools.execute('crop', {'mode': 'tiles', 'tile_size': 128})
        from Utils.AIResultBrowser import visual_rows
        _, rows = visual_rows(tools.store, result['artifact_id'])
        image = Path(rows[0]['image']).relative_to(self.root.resolve()).as_posix()
        self.context.add_feedback(image, 'annotation_question', artifact_id=result['artifact_id'])
        subset = tools.execute('select_feedback', {'category': 'annotation_question', 'artifact_id': result['artifact_id']})
        self.assertEqual(subset['kind'], 'selection')
        self.assertEqual(subset['source_artifact'], result['artifact_id'])
        self.assertEqual(subset['count'], 1)
        lineage = self.context.query('lineage', artifact_id=result['artifact_id'])
        self.assertEqual(lineage['items'][0]['source_image'], self.paths[0])
        folder, manifest = tools.store.artifact(result['artifact_id'])
        manifest['status'] = 'deleted'
        write_json(folder / 'manifest.json', manifest)
        self.assertTrue(self.context.feedback()[0]['stale'])

    def test_experiment_dataset_model_links_cache_and_comparable_rankings(self):
        from Utils.AITrainingAnalysis import RECORD
        dataset = self.root / 'dataset'
        dataset.mkdir()
        write_json(dataset / 'data.yaml', {'names': ['defect'], 'train': 'images/train', 'val': 'images/val'})
        for name, score, fingerprint in [('one', .4, 'same'), ('two', .6, 'same'), ('different', .9, 'different')]:
            folder = self.root / 'runs' / 'detect' / name
            (folder / 'weights').mkdir(parents=True)
            csv = folder / 'results.csv'
            csv.write_text('epoch,metrics/mAP50-95(B)\n1,' + str(score) + '\n', encoding='utf-8')
            write_json(folder / 'args.yaml', {'task': 'detect', 'data': str((dataset / 'data.yaml').resolve())})
            write_json(folder / RECORD, {'status': 'completed', 'validation_fingerprint': fingerprint,
                'names': ['defect'], 'evaluation_config': {'task': 'detect', 'imgsz': 640},
                'results_sha256': file_sha(csv), 'started_at': '2026-09-28 12:00:00'})
            (folder / 'weights/best.pt').write_bytes(b'fake model')
        result = self.context.inventory()
        self.assertEqual(len(result['experiments']), 3)
        self.assertTrue(all(r['dataset_id'] == result['datasets'][0]['id'] for r in result['experiments']))
        self.assertEqual(len(result['comparisons']), 1)
        self.assertEqual(result['comparisons'][0]['value'], .6)
        self.assertEqual(sum('produced_by' in r for r in result['resources']), 3)
        with patch('Utils.AITrainingAnalysis.load_run', side_effect=AssertionError('should use metadata cache')):
            self.context.inventory()
        csv = self.root / 'runs/detect/two/results.csv'
        csv.write_text('epoch,metrics/mAP50-95(B)\n1,0.7\n2,0.8\n', encoding='utf-8')
        updated = self.context.inventory()
        changed = next(r for r in updated['experiments'] if r['title'] == 'two')
        self.assertIsNone(changed['comparison_group'])
        self.assertEqual(changed['best'][changed['primary_metric']], .8)

    def test_overview_does_not_send_bulk_paths_or_annotation_documents(self):
        store = Workspace(self.root)
        _, folder, manifest = store.create('prediction_plan', '预测方案', {
            'paths': self.paths, 'overrides': {'private': {'annotations': ['do not send']}},
            'model': 'fixture.pt', 'conf': .5})
        store.finish(folder, manifest, count=5)
        value = self.context.query()
        metadata = value['recent_results'][0]['parameters']
        self.assertNotIn('paths', metadata)
        self.assertNotIn('overrides', metadata)
        self.assertEqual(value['recent_results'][0]['source_count'], 5)


class ContextUITests(Fixtures):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from Utils.test import TestWindow
        self.window = TestWindow(str(self.root))

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def test_resource_added_in_new_chat_is_available_when_returning_to_old_chat(self):
        self.window.show_ai_chat()
        chat = self.window.ai_chat_dock
        old_id = chat.conversation['id']
        chat.clear_history()
        model = self.root / 'example.pt'
        model.write_bytes(b'fake')
        chat.register_resource(model, 'model')
        chat.activate_conversation(chat.conversation_store.read(old_id))
        self.assertIn(str(model.resolve()), [r['path'] for r in chat.resources.values()])
        self.assertEqual(chat.project_context_button.text(), '项目状态')
        self.assertNotIn('ai_review_action', vars(self.window))

    def test_loading_model_registers_actual_selection_without_setting_project_preference(self):
        model = self.root / 'test.pt'
        model.write_bytes(b'fixture, no actual model')
        fake = object()
        with patch('Utils.test.QFileDialog.getOpenFileName', return_value=(str(model), '')):
            with patch('Utils.test.YOLO', return_value=fake):
                self.window.load_model()
        store = ProjectContext(self.root)
        rid = self.window.current_inference_resource_id
        self.assertEqual(store.settings()['last_loaded_inference']['resource_id'], rid)
        self.assertNotIn('preferred_model', store.settings())
        self.window.show_ai_chat()
        self.assertEqual(self.window.ai_chat_dock.work_context()['current_inference_resource_id'], rid)

    def test_context_dialog_loads_without_api_and_keeps_originals(self):
        from Utils.ProjectContextDialog import ProjectContextDialog
        before = self.hashes()
        ProjectContext(self.root).add_feedback(self.paths[0], 'image_quality', '人工备注')
        dialog = ProjectContextDialog(self.window)
        dialog.show()
        try:
            for _ in range(500):
                if not dialog.worker.isRunning():
                    break
                QTest.qWait(10)
            self.assertFalse(dialog.worker.isRunning())
            self.app.processEvents()
            self.assertEqual(dialog.data['summary']['images'], 5)
            self.assertEqual(dialog.feedback_table.rowCount(), 1)
            self.assertEqual(before, self.hashes())
        finally:
            dialog.worker.requestInterruption()
            dialog.worker.wait(5000)
            dialog.close()
            dialog.deleteLater()
            self.app.processEvents()

    def test_human_dialog_saves_choice_to_selected_prediction_only(self):
        from Utils.ProjectContextDialog import record_feedback
        from PyQt5.QtWidgets import QComboBox, QPlainTextEdit, QDialogButtonBox
        aid = ContextTests.candidate(self)
        before = self.hashes()
        def submit(dialog):
            choice = dialog.findChild(QComboBox)
            choice.setCurrentIndex(choice.findData('false_positive'))
            dialog.findChild(QPlainTextEdit).setPlainText('人工确认反光误检')
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
            return dialog.result()
        with patch('Utils.ProjectContextDialog.QDialog.exec_', submit):
            record_feedback(self.window, [self.paths[0]], aid)
        records = ProjectContext(self.root).feedback()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['artifact'], aid)
        self.assertEqual(records[0]['category'], 'false_positive')
        self.assertEqual(before, self.hashes())


if __name__ == '__main__':
    unittest.main()
