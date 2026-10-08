"""Conversation persistence and UI continuity, using disposable local projects."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PyQt5.QtWidgets import QApplication

from test_ai_chat_selection import Fixtures, AREA_CODE
from Utils.AIConversations import ConversationStore, load_groups, result_card
from Utils.AIWorkspace import Workspace, write_json, read_json
from Utils.AIResultStorage import deletion_preview, delete_result, set_result_hidden


class ConversationStoreTests(Fixtures):
    def test_search_schema_and_project_relative_resources(self):
        store = ConversationStore(self.root)
        conversation = store.new()
        conversation['messages'] = [{'id': 'message1', 'role': 'user', 'text': '分析train30并检查脏污'}]
        conversation['api_key'] = 'should-not-persist'
        conversation['draft'] = 'current-secret'
        resource = {'r1': {'path': str(self.root / 'images'), 'name': 'images', 'kind': 'directory'}}
        conversation['resources'] = store.encode_resources(resource)
        store.save(conversation, 'current-secret')
        raw = store.path(conversation['id']).read_text(encoding='utf-8')
        self.assertNotIn('should-not-persist', raw)
        self.assertNotIn('current-secret', raw)
        self.assertNotIn(str(self.root).replace('\\', '\\\\'), raw)
        self.assertEqual(store.decode_resources(store.current()['resources']), resource)
        match = store.list('脏污')
        self.assertEqual(match[0]['anchor'], 'message1')
        self.assertEqual(store.list('unmatched'), [])
        with self.assertRaises(ValueError):
            store.read('../images')

    def test_old_runs_become_explicitly_reconstructed_work_history_without_execution(self):
        workspace = Workspace(self.root)
        aid, folder, manifest = workspace.create('crops', '旧切图')
        workspace.finish(folder, manifest, count=2, source_count=2)
        write_json(workspace.location('runs', 'old-run') / 'run.json', {
            'id': 'old-run', 'kind': 'agent', 'title': '对话工作任务', 'created_at': '2026-09-23 12:00',
            'request': '切图', 'status': 'completed', 'steps': [{'result': {'artifact_id': aid, 'kind': 'crops', 'count': 2, 'source_count': 2}}]})
        store = ConversationStore(self.root)
        with patch('Utils.AIWorkAgent.run_agent', side_effect=AssertionError('must not execute')):
            value = store.current()
            again = store.current()
        self.assertIn('由旧记录整理', value['title'])
        self.assertIn('未保存完整聊天', value['messages'][0]['text'])
        self.assertEqual(value['messages'][1]['text'], '切图')
        self.assertEqual(value['messages'][2]['cards'][0]['id'], aid)
        self.assertEqual(value['id'], again['id'])


class ConversationUITests(Fixtures):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from Utils.test import TestWindow
        self.window = TestWindow(str(self.root))
        self.window.show_ai_chat()
        self.dock = self.window.ai_chat_dock

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def reopen(self):
        from Utils.test import TestWindow
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        with patch('Utils.AIAugment.request_qwen_json', side_effect=AssertionError('must not call API')):
            self.window = TestWindow(str(self.root))
            self.window.show_ai_chat()
            self.dock = self.window.ai_chat_dock
            self.app.processEvents()

    def artifact(self):
        store = Workspace(self.root)
        aid, folder, manifest = store.create('candidates', '预测结果')
        write_json(folder / 'candidates.json', [])
        write_json(folder / 'errors.json', [])
        result = store.finish(folder, manifest, count=0, failed=0, predicted_annotations=0)
        return store, aid, folder, result

    def test_independent_window_minimize_reopen_preserves_chat_and_project_close_hides_it(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QDockWidget
        self.assertTrue(self.dock.isWindow())
        self.assertTrue(self.dock.windowFlags() & Qt.WindowMinimizeButtonHint)
        self.assertFalse(self.window.findChildren(QDockWidget))
        self.dock.say('你', '保留这段对话')
        self.dock.input.setPlainText('尚未发送的草稿')
        cid = self.dock.conversation['id']
        self.dock.showMinimized()
        self.app.processEvents()
        self.assertTrue(self.dock.isMinimized())
        self.window.show_ai_chat()
        self.app.processEvents()
        self.assertIs(self.window.ai_chat_dock, self.dock)
        self.assertFalse(self.dock.isMinimized())
        self.assertTrue(self.dock.isVisible())
        self.dock.close()
        self.assertFalse(self.dock.isVisible())
        self.window.show_ai_chat()
        self.assertEqual(self.dock.conversation['id'], cid)
        self.assertEqual(self.dock.input.toPlainText(), '尚未发送的草稿')
        self.assertIn('保留这段对话', self.dock.transcript.toPlainText())
        self.window.close()
        self.assertFalse(self.dock.isVisible())
        self.assertEqual(self.dock.conversation_store.current()['draft'], '尚未发送的草稿')

    def test_result_subset_is_visible_and_restored_as_processed_images(self):
        from Utils.AIWorkTools import WorkTools
        original = self.hashes()
        tools = WorkTools(self.root, self.paths, context={'current_paths': self.paths[1:3]})
        crop = tools.execute('crop', {'mode': 'tiles', 'tile_size': 64})
        selected = tools.execute('select', {'scope': 'artifact:' + crop['artifact_id'],
            'title': '两张切图', 'criteria': '前两张结果图片', 'code': 'result = [r["id"] for r in records[:2]]'})
        self.assertTrue(self.window.result_browser.open(selected['artifact_id']))
        self.assertEqual(self.window.result_browser.list.count(), 2)
        self.assertEqual(self.dock.work_context()['current_result_count'], 2)
        self.dock.say('助手', '两张切图', [result_card(selected)])
        self.reopen()
        self.assertEqual(self.window.current_tree_filter, ('artifact', selected['artifact_id'], None))
        self.assertEqual(self.window.result_browser.list.count(), 2)
        self.assertTrue(all('crop_' in row['name'] for row in self.window.result_browser.rows))
        self.assertEqual(original, self.hashes())

    def test_result_card_opens_actual_dataset_folder_without_switching_view(self):
        from Utils.AIWorkTools import WorkTools
        tools = WorkTools(self.root, self.paths, context={'current_paths': self.paths[1:3]})
        result = tools.execute('export', {})
        before = self.window.current_tree_filter
        with patch('Utils.AIChatHistory.QDesktopServices.openUrl', return_value=True) as opened:
            self.dock.open_chat_result('folder', result['artifact_id'])
        self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), Path(result['dataset_dir']))
        self.assertEqual(self.window.current_tree_filter, before)

    def test_processed_multiselection_exports_only_selected_images_and_opens_public_folder(self):
        from Utils.AIWorkTools import WorkTools
        tools = WorkTools(self.root, self.paths, context={'current_paths': self.paths[1:3]})
        crop = tools.execute('crop', {'mode': 'tiles', 'tile_size': 64})
        browser = self.window.result_browser
        self.assertTrue(browser.open(crop['artifact_id']))
        browser.list.clearSelection()
        browser.list.item(1).setSelected(True)
        browser.list.item(3).setSelected(True)
        context = self.dock.work_context()
        self.assertEqual(len(context['selected_result_paths']), 2)
        self.assertEqual(context['selected_paths'], [])
        self.assertIn('选中 2 张', self.dock.scope_label.text())
        result = WorkTools(self.root, self.paths, context=context).execute('export', {'scope': 'selected', 'content': 'images'})
        self.assertEqual(result['count'], 2)
        self.dock.say('助手', '只导出两张图片。', [result_card(result)])
        with patch('Utils.AIChatHistory.QDesktopServices.openUrl', return_value=True) as opened:
            self.dock.open_chat_result('folder', result['artifact_id'])
        self.assertEqual(Path(opened.call_args.args[0].toLocalFile()), Path(result['export_dir']))
        self.reopen()
        self.assertIn('只导出两张图片', self.dock.transcript.toPlainText())

    def test_reopen_restores_chat_draft_resources_groups_and_context_without_key(self):
        gid = self.window.upsert_temporary_selection('大面积脏污', '>5%', self.paths[1:3], AREA_CODE)
        self.dock.api_key.setText('fixture-private-key')
        self.dock.say('你', '筛选大面积脏污')
        self.dock.history = [{'role': 'user', 'content': '筛选大面积脏污'}, {'role': 'assistant', 'content': '已找到2张'}]
        self.dock.say('助手', '已找到2张', [result_card({'group_id': gid, 'count': 2, 'title': '大面积脏污'})])
        self.dock.register_resource(self.root / 'images', 'directory')
        self.dock.input.setPlainText('接下来切图')
        cid = self.dock.conversation['id']
        self.reopen()
        self.assertEqual(self.dock.conversation['id'], cid)
        self.assertIn('已找到2张', self.dock.transcript.toPlainText())
        self.assertEqual(self.dock.input.toPlainText(), '接下来切图')
        self.assertEqual(self.dock.api_key.text(), '')
        self.assertEqual(self.window.temporary_selections[gid]['paths'], self.paths[1:3])
        self.assertEqual(self.dock.work_context()['current_paths'], self.paths[1:3])
        self.assertEqual(self.dock.resources['r1']['path'], str(self.root / 'images'))
        self.assertIsNone(self.dock.worker)
        for p in (self.root / 'ai_workbench/conversations').glob('*.json'):
            self.assertNotIn('fixture-private-key', p.read_text(encoding='utf-8'))

    def test_new_conversation_preserves_history_and_search_selects_old_message(self):
        self.dock.say('你', '查看train30的训练表现')
        first = self.dock.conversation['id']
        self.dock.clear_history()
        second = self.dock.conversation['id']
        self.assertNotEqual(first, second)
        self.assertEqual(self.dock.history, [])
        self.assertEqual(self.dock.transcript.toPlainText(), '')
        self.dock.history_search.setText('train30')
        self.dock.show_conversations()
        self.assertEqual(self.dock.conversation_list.count(), 1)
        self.dock.open_conversation(self.dock.conversation_list.item(0))
        self.assertEqual(self.dock.conversation['id'], first)
        self.assertIn('train30', self.dock.transcript.toPlainText())
        self.assertEqual(len(self.dock.conversation_store.list()), 2)

    def test_hidden_deleted_and_missing_results_are_reflected_in_old_chat(self):
        store, aid, folder, result = self.artifact()
        self.dock.say('助手', '推理已完成', [result_card(result)])
        self.assertEqual(self.dock.result_states()[('artifact', aid)], ('可查看', True))
        self.window.result_browser.open(aid)
        self.window.result_browser.hide(aid)
        self.assertEqual(self.dock.result_states()[('artifact', aid)], ('已隐藏', True))
        self.dock.open_chat_result('artifact', aid)
        self.assertEqual(self.window.result_browser.active_id, aid)
        plan = deletion_preview(store, aid)
        delete_result(store, aid, plan['token'])
        self.assertEqual(self.dock.result_states()[('artifact', aid)], ('结果已删除', False))
        self.reopen()
        self.assertTrue(self.dock.missing_view)
        self.dock.api_key.setText('fake-key')
        self.dock.input.setPlainText('处理刚才的结果')
        self.dock.submit()
        self.assertIsNone(self.dock.worker)
        self.assertIn('重新选择', self.dock.scope_label.text())

    def test_interrupted_task_restores_as_interrupted_and_never_resumes(self):
        self.dock.say('你', '修改这批标注')
        self.dock.start_task('等待确认')
        self.dock.update_task('等待确认', 'awaiting_confirmation')
        self.reopen()
        self.assertIsNone(self.dock.pending_operation)
        self.assertIsNone(self.dock.continuation)
        self.assertIn('未继续执行', self.dock.transcript.toPlainText())
        self.assertEqual(self.dock.conversation['messages'][-1]['status'], 'interrupted')

    def test_removed_source_keeps_empty_group_without_falling_back_to_all(self):
        gid = self.window.upsert_temporary_selection('一张图', 'fixture', [self.paths[0]])
        groups = load_groups(self.root, self.paths[1:])
        self.assertIn(gid, groups)
        self.assertEqual(groups[gid]['paths'], [])

    def test_credentials_remain_redacted_after_changing_key(self):
        self.dock.api_key.setText('old-private-key')
        self.dock.input.setPlainText('draft old-private-key')
        self.dock.history = [{'role': 'user', 'content': 'old-private-key'}]
        self.dock.save_conversation()
        self.dock.api_key.setText('new-private-key')
        self.dock.save_conversation()
        raw = self.dock.conversation_store.path(self.dock.conversation['id']).read_text(encoding='utf-8')
        self.assertNotIn('old-private-key', raw)
        self.assertNotIn('new-private-key', raw)

    def test_corrupt_chat_is_not_overwritten_and_warning_is_visible(self):
        original = self.dock.conversation_store.path(self.dock.conversation['id'])
        # Close first so the valid in-memory record does not intentionally save
        # over the fixture corruption during the close event.
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        original.write_text('{broken', encoding='utf-8')
        from Utils.test import TestWindow
        self.window = TestWindow(str(self.root))
        self.window.show_ai_chat()
        self.dock = self.window.ai_chat_dock
        self.assertEqual(original.read_text(encoding='utf-8'), '{broken')
        self.assertIn('原文件保留', self.dock.save_error.text())
        self.assertNotEqual(self.dock.conversation_store.path(self.dock.conversation['id']), original)


if __name__ == '__main__':
    unittest.main()
