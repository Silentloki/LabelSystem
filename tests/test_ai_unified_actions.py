"""Unified operations on isolated projects; no network or model execution."""
import copy
import pickle
from pathlib import Path
from unittest.mock import patch

from test_ai_result_actions import ActionFixtures
from Utils.AIArtifactExport import export_rows
from Utils.AIResultBrowser import visual_rows
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkAgent import run_agent
from Utils.AIWorkspace import file_sha, read_json, write_json, read_project_list
from PyQt5.QtWidgets import QApplication


class UnifiedActionTests(ActionFixtures):
    def originals(self, paths=None):
        return WorkTools(self.root, self.paths, context={'selected_paths': paths or self.paths[:2],
            'current_paths': self.paths[:4], 'classes': ['脏污', '破损']})

    def tx(self, tools, status):
        result = tools.execute('set_sample_status', {'status': status})
        return result['pending']['transaction_id']

    def test_good_previews_exact_selection_and_restores_all_bytes(self):
        write_json(self.root / 'sample_tree.json', {'groups': {'脏污': ['A']},
            'assignments': {self.paths[0]: {'脏污': 'A'}, self.paths[2]: {'脏污': 'A'}}})
        originals = {p: p.read_bytes() for p in [self.root / 'flagfile.dat', self.root / 'sample_tree.json',
                    self.root / 'jsons/small.json', self.root / 'jsons/large.json']}
        before, candidates = self.hashes(), file_sha(self.folder / 'candidates.json')
        tools = self.tools('把这些改为良品')
        tid = self.tx(tools, 'good')
        self.assertEqual(before, self.hashes())
        self.assertTrue(all(p.read_bytes() == data for p, data in originals.items()))
        transaction = read_json(self.store.location('transactions', tid) / 'transaction.json')
        self.assertEqual(transaction['metadata']['affected_images'], self.paths[:2])
        self.assertIn('2 处', transaction['metadata']['changes'][0]['action'])
        done = self.store.commit(tid)
        self.assertEqual(done['set_flags'], {p: 2 for p in self.paths[:2]})
        self.assertEqual(read_project_list(self.root / 'flagfile.dat'), [2, 2, 1, 1, 0])
        self.assertEqual(read_json(self.root / 'jsons/small.json')['annotations'], [])
        self.assertEqual(read_json(self.root / 'sample_tree.json')['assignments'], {self.paths[2]: {'脏污': 'A'}})
        undo = tools.execute('undo', {'transaction_id': tid})['pending']
        self.store.commit(undo['transaction_id'])
        self.assertTrue(all(p.read_bytes() == data for p, data in originals.items()))
        self.assertEqual(before, self.hashes())
        self.assertEqual(candidates, file_sha(self.folder / 'candidates.json'))

    def test_overkill_and_state_only_clear_are_recoverable(self):
        tools = self.originals()
        self.store.commit(self.tx(tools, 'overkill'))
        self.assertEqual(read_project_list(self.root / 'flagfile.dat')[:2], [3, 3])
        before = self.hashes()
        tid = self.tx(tools, 'clear_good')
        record = read_json(self.store.location('transactions', tid) / 'transaction.json')
        self.assertEqual([r['path'] for r in record['entries']], ['flagfile.dat'])
        self.store.commit(tid)
        self.assertEqual(read_project_list(self.root / 'flagfile.dat')[:2], [0, 0])
        undo = self.store.prepare_undo(tid)
        done = self.store.commit(undo['transaction_id'])
        self.assertEqual(done['affected_images'], self.paths[:2])
        self.assertEqual(done['set_flags'], {p: 3 for p in self.paths[:2]})
        self.assertEqual(before, self.hashes())

    def test_missing_annotation_good_undo_removes_only_created_annotation(self):
        tools = self.originals([self.paths[4]])
        tid = self.tx(tools, 'good')
        self.store.commit(tid)
        self.assertTrue((self.root / 'jsons/missing.json').exists())
        self.store.commit(self.store.prepare_undo(tid)['transaction_id'])
        self.assertFalse((self.root / 'jsons/missing.json').exists())
        self.assertEqual(read_project_list(self.root / 'flagfile.dat')[-1], 0)

    def test_list_flags_or_annotations_changed_after_preview_reject_commit(self):
        for target in ['datafile.dat', 'flagfile.dat', 'jsons/small.json']:
            with self.subTest(target=target):
                path = self.root / target
                original = path.read_bytes()
                before = self.hashes()
                tid = self.tx(self.originals(), 'good')
                if target.endswith('.json'):
                    changed = read_json(path)
                    changed['annotations'] = []
                    write_json(path, changed)
                else:
                    path.write_bytes(original + b' ')
                with self.assertRaises(ValueError):
                    self.store.commit(tid)
                path.write_bytes(original)
                self.assertEqual(self.hashes(), before)

    def test_clear_skips_other_statuses_and_empty_selection_cannot_expand(self):
        self.assertEqual(self.originals().execute('set_sample_status', {'status': 'clear_good'})['count'], 0)
        tools = self.originals()
        tools.context['selected_paths'] = []
        with self.assertRaises(ValueError):
            self.tx(tools, 'good')
        self.context['selected_result_paths'] = []
        with self.assertRaises(ValueError):
            self.tx(self.tools(), 'good')

    def test_windows_saved_paths_match_without_using_hidden_selection(self):
        values = read_json(self.folder / 'candidates.json')
        for row in values:
            row['path'] = row['path'].replace('/', '\\')
        write_json(self.folder / 'candidates.json', values)
        tid = self.tx(self.tools(), 'good')
        done = self.store.commit(tid)
        self.assertEqual(done['affected_images'], self.paths[:2])
        self.assertEqual(read_project_list(self.root / 'flagfile.dat')[3], 1)

    def test_legacy_project_list_paths_and_result_reference_use_exact_identity(self):
        (self.root / 'datafile.dat').write_bytes(pickle.dumps([p.replace('/', '\\') for p in self.paths]))
        self.context['selected_result_paths'] = self.paths[:1]
        tid = self.tx(self.tools(), 'good')
        self.assertEqual(self.store.commit(tid)['affected_images'], self.paths[:1])
        result = self.tools().execute('similar', {'scope': 'all'})
        self.assertEqual(result['count'], 4)
        self.context['selected_result_paths'] = self.paths[:2]
        with self.assertRaisesRegex(ValueError, '参考照片'):
            self.tools().execute('similar', {'scope': 'all'})

    def test_prediction_selection_stats_and_exports_keep_prediction_semantics(self):
        tools = self.tools()
        selected = tools.execute('select', {'scope': 'selected', 'title': '第一张', 'criteria': '首张',
                                'code': "result = [r['id'] for r in records[:1]]"})
        _, rows = visual_rows(self.store, selected['artifact_id'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['predictions'], 2)
        stats = tools.execute('stats', {'scope': 'last'})
        self.assertEqual(stats['annotation_basis'], 'saved_predictions')
        self.assertEqual(stats['summary']['total_images'], 1)
        self.assertEqual(tools.execute('export', {'scope': 'last', 'content': 'predictions'})['count'], 1)
        self.assertEqual(tools.execute('export', {'scope': 'artifact:' + selected['artifact_id'], 'content': 'images'})['count'], 1)
        for content in ['labels', 'annotated', 'dataset']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                tools.execute('export', {'scope': 'artifact:' + selected['artifact_id'], 'content': content})
        with self.assertRaises(ValueError):
            tools.execute('crop', {'scope': 'artifact:' + selected['artifact_id'], 'mode': 'tiles'})

    def test_problem_set_can_filter_export_and_modify_corresponding_originals(self):
        tools = self.tools()
        problems = tools.execute('add_error_set', {})
        self.assertEqual(tools.execute('stats', {'scope': 'last'})['summary']['total_images'], 2)
        selected = tools.execute('select', {'scope': 'last', 'title': '首张', 'criteria': '首张',
            'code': "result = [r['id'] for r in records[:1]]"})
        self.assertEqual(tools.execute('export', {'scope': 'last', 'content': 'preview'})['count'], 1)
        result = tools.execute('relabel', {'scope': 'artifact:' + selected['artifact_id'], 'old': '脏污', 'new': '破损'})
        self.assertEqual(result['images'], 1)
        self.store.commit(result['pending']['transaction_id'])
        self.assertEqual(read_json(self.root / 'jsons/small.json')['annotations'][0]['lable'], '破损')
        self.assertEqual(read_json(self.folder / 'candidates.json')[0]['document']['annotations'][0]['lable'], '脏污')

    def test_original_problem_set_can_export_annotations(self):
        self.context.update(viewing_artifact=None, selected_paths=self.paths[:2],
            interaction_target={'artifact_id': None, 'images': self.paths[:2]})
        tools = self.tools()
        tools.execute('add_error_set', {})
        self.assertEqual(tools.execute('export', {'scope': 'last', 'content': 'annotated'})['count'], 2)

    def test_unannotated_original_in_problem_set_does_not_become_negative(self):
        self.context.update(viewing_artifact=None, selected_paths=[self.paths[4]],
            interaction_target={'artifact_id': None, 'images': [self.paths[4]]})
        tools = self.tools()
        tools.execute('add_error_set', {})
        self.assertEqual(tools.execute('export', {'scope': 'last', 'content': 'images'})['count'], 1)
        with self.assertRaisesRegex(ValueError, '未保存标注'):
            tools.execute('export', {'scope': 'last', 'content': 'annotated'})

    def test_crop_chain_preserves_lineage_and_requires_explicit_original_lookup(self):
        tools = self.originals([self.paths[0]])
        first = tools.execute('crop', {'scope': 'selected', 'mode': 'tiles', 'tile_size': 128})
        second = tools.execute('crop', {'scope': 'last', 'mode': 'tiles', 'tile_size': 64})
        rows, _ = export_rows(tools, second['artifact_id'])
        self.assertTrue(all(r['original_source'] == self.paths[0] for r in rows))
        self.assertEqual(len({r['split_group'] for r in rows}), 1)
        with self.assertRaisesRegex(ValueError, '派生'):
            tools.execute('set_sample_status', {'scope': 'last', 'status': 'good'})
        group = tools.execute('original_images', {'scope': 'last'})
        self.assertEqual(group['count'], 1)
        result = tools.execute('set_sample_status', {'scope': 'last', 'status': 'good'})
        self.assertEqual(result['images'], 1)

    def test_changed_prediction_image_rejected_before_export_or_modification(self):
        with (self.root / self.paths[0]).open('ab') as stream:
            stream.write(b'changed')
        for name, args in [('set_sample_status', {'status': 'good'}),
                           ('export', {'scope': 'selected', 'content': 'images'})]:
            with self.subTest(tool=name), self.assertRaisesRegex(ValueError, '已变化'):
                self.tools().execute(name, args)

    def test_agent_good_request_produces_preview_without_writing_feedback(self):
        from Utils.ProjectContext import ProjectContext
        before = self.hashes()
        def request(*args, **kwargs):
            self.assertIn('set_sample_status', args[3])
            return {'action': 'tool', 'tool': 'set_sample_status',
                    'args': {'status': 'good', 'scope': 'selected'}}, {}
        result = run_agent(self.root, self.paths, {}, {}, [], '把这个改为良品',
            ('artifact', self.aid, None), ('fake', 'https://example.invalid', 'test'),
            context=self.context, request=request)
        self.assertIsNotNone(result['work']['pending'])
        self.assertEqual(result['work']['pending']['kind'], 'transaction')
        self.assertEqual(ProjectContext(self.root).feedback(), [])
        self.assertEqual(before, self.hashes())

    def test_capabilities_distinguish_predictions_and_project_status(self):
        actions = self.tools().execute('available_actions', {})['actions']
        self.assertTrue(actions['set_sample_status'])
        self.assertTrue(actions['export_predictions_or_preview'])
        self.assertFalse(actions['export_dataset_or_annotations'])
        self.assertFalse(actions['crop'])


class UnifiedActionUITests(ActionFixtures):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from Utils.test import TestWindow
        self.window = TestWindow(str(self.root))
        self.window.show_ai_chat()
        self.chat = self.window.ai_chat_dock
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def commit_and_refresh(self, pending):
        from Utils.AIWorkDialogs import ensure_canvas_matches
        ensure_canvas_matches(self.window, pending)
        result = self.store.commit(pending['transaction_id'])
        self.chat.apply_local_result({'local_work': result})
        self.app.processEvents()

    def test_good_clear_and_restore_update_real_ui_flags_and_canvas(self):
        self.window.listWidget.setCurrentRow(0)
        tools = WorkTools(self.root, self.paths, context=self.chat.work_context())
        result = tools.execute('set_sample_status', {'status': 'good'})
        self.assertTrue(self.window.graphicsView.imageItem.annotations)
        self.commit_and_refresh(result['pending'])
        self.assertEqual(self.window.flag[0], 2)
        self.assertEqual(self.window.graphicsView.imageItem.annotations, [])
        tools = WorkTools(self.root, self.paths, context=self.chat.work_context())
        clear = tools.execute('set_sample_status', {'status': 'clear_good'})
        self.commit_and_refresh(clear['pending'])
        self.assertEqual(self.window.flag[0], 0)
        undo = self.store.prepare_undo(clear['pending']['transaction_id'])
        self.commit_and_refresh({'kind': 'transaction', **undo})
        self.assertEqual(self.window.flag[0], 2)

    def test_unsaved_other_canvas_is_not_discarded_by_status_refresh(self):
        from Utils.AIWorkDialogs import ensure_canvas_matches
        self.window.listWidget.setCurrentRow(0)
        tools = WorkTools(self.root, self.paths, context={'selected_paths': [self.paths[1]]})
        pending = tools.execute('set_sample_status', {'status': 'good'})['pending']
        self.window.graphicsView.imageItem.remove_annotations()
        with self.assertRaisesRegex(ValueError, '另一张图片'):
            ensure_canvas_matches(self.window, pending)

