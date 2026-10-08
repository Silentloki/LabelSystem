"""Result subsets, repeat prevention, request budget and transcript positioning."""
import copy
import unittest
from pathlib import Path

from test_ai_workbench import WorkFixtures
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkAgent import run_agent
from Utils.AIWorkspace import read_json, file_sha
from Utils.AIResultBrowser import visual_rows
from Utils.AIResultStorage import deletion_preview, delete_result

CODE = "matches = [r for r in records if any(a['label'] == '脏污' for a in r['annotations'])]\nranked = sorted(matches, key=lambda r: max(a['area_ratio'] for a in r['annotations'] if a['label'] == '脏污'), reverse=True)\nresult = [r['id'] for r in ranked[:3]]"
ARGS = {'title': '处理结果最大脏污3张', 'criteria': '最大脏污面积占比前三', 'code': CODE}


class ResultSelectionTests(WorkFixtures):
    def test_select_result_reopen_subset_export_and_delete_preserve_source(self):
        self.context['current_paths'] = self.paths[:4]
        crop = self.call('crop', mode='tiles', tile_size=64)
        self.context.update(viewing_artifact=crop['artifact_id'], current_paths=[])
        self.tools = WorkTools(self.root, self.paths, context=self.context)
        protected = {p: file_sha(p) for p in Path(crop['output']).rglob('*') if p.is_file()}
        originals = self.hashes()
        selected = self.call('select', **ARGS)
        self.assertEqual(selected['kind'], 'selection')
        self.assertEqual(selected['count'], 3)
        self.assertEqual(self.tools.effects, [])
        again = self.call('select', **ARGS)
        self.assertEqual(again['artifact_id'], selected['artifact_id'])
        self.assertTrue(again['reused_result'])
        _, rows = visual_rows(self.store, selected['artifact_id'])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all('/images/crop_' in r['image'].replace('\\', '/') for r in rows))
        self.context['viewing_artifact'] = selected['artifact_id']
        self.tools = WorkTools(self.root, self.paths, context=self.context)
        self.assertEqual(self.call('stats')['summary']['total_images'], 3)
        subset = self.call('select', title='这组三张中前一张', criteria='第一张', code='result = [records[0]["id"]]')
        value = read_json(Path(subset['output']) / 'selection.json')
        self.assertEqual(value['source_artifact'], crop['artifact_id'])
        exported = self.call('export', scope='last', format='native')
        self.assertEqual(exported['count'], 1)
        plan = deletion_preview(self.store, selected['artifact_id'])
        delete_result(self.store, selected['artifact_id'], plan['token'])
        self.assertEqual(protected, {p: file_sha(p) for p in protected})
        self.assertEqual(originals, self.hashes())
        self.assertEqual(len(visual_rows(self.store, subset['artifact_id'])[1]), 1)
        empty = self.call('select', scope='artifact:' + subset['artifact_id'], title='空筛选', criteria='无匹配', code='result = []')
        self.assertEqual(empty['count'], 0)
        with self.assertRaisesRegex(ValueError, '没有匹配'):
            self.call('export', scope='last')
        plan = deletion_preview(self.store, crop['artifact_id'])
        delete_result(self.store, crop['artifact_id'], plan['token'])
        with self.assertRaises(ValueError):
            visual_rows(self.store, subset['artifact_id'])

    def test_original_duplicate_selects_reuse_one_group_across_turns(self):
        self.context['current_paths'] = self.paths[:4]
        first = self.call('select', **ARGS)
        second = self.call('select', **ARGS)
        self.assertEqual(first['group_id'], second['group_id'])
        self.assertEqual(len(self.tools.effects), 1)
        other = WorkTools(self.root, self.paths, groups=self.tools.groups, context=self.context)
        third = other.execute('select', ARGS)
        self.assertEqual(first['group_id'], third['group_id'])
        self.assertEqual(other.effects, [])

    def test_repeated_model_selection_stops_early_with_one_result(self):
        self.context.update(max_calls=32, current_paths=self.paths[:4])
        result = run_agent(self.root, self.paths, {}, {}, [], '找三张最大脏污', 'all', ('fake', 'url', 'model'),
                           context=self.context, request=lambda *a, **kw: (
                               {'action': 'tool', 'tool': 'select', 'args': copy.deepcopy(ARGS)}, {}))
        self.assertEqual(result['requests'], 3)
        self.assertEqual(len(result['work']['effects']), 1)
        self.assertEqual(len(result['work']['steps']), 1)
        self.assertEqual(result['work']['status'], 'partial')
        self.assertIn('重复', result['result']['message'])

    def test_more_than_eight_requests_supported_without_repeating_writes(self):
        self.context.update(max_calls=12, current_paths=self.paths[:4])
        calls = []
        def request(*args, **kwargs):
            calls.append(1)
            return ({'action': 'reply', 'message': '完成'} if len(calls) == 10 else
                    {'action': 'tool', 'tool': 'stats', 'args': {}}, {})
        result = run_agent(self.root, self.paths, {}, {}, [], '多步统计', 'all', ('fake', 'url', 'model'),
                           context=self.context, request=request)
        self.assertEqual(result['requests'], 10)
        self.assertEqual(result['work']['status'], 'completed')


class TranscriptPositionTests(unittest.TestCase):
    def test_messages_follow_bottom_and_preserve_reading_position(self):
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtTest import QTest
        from Utils.AIChatView import ConversationTranscript
        app = QApplication.instance() or QApplication([])
        view = ConversationTranscript()
        view.resize(620, 400)
        view.show()
        messages = [{'id': str(i), 'role': 'assistant', 'created_at': '2026-09-24 15:48',
                     'text': '这是一条多行历史消息。\n' * 4} for i in range(35)]
        try:
            view.render(messages, {}, bottom=True)
            QTest.qWait(80)
            bar = view.verticalScrollBar()
            self.assertGreater(bar.maximum(), 0)
            self.assertEqual(bar.value(), bar.maximum())
            bar.setValue(bar.maximum() // 2)
            view.remember_scroll()
            reading = bar.value()
            messages[-1]['text'] += '更新进度\n' * 5
            view.render(messages, {})
            view.render(messages, {})
            QTest.qWait(80)
            self.assertEqual(bar.value(), reading)
            messages.append({'id': 'new', 'role': 'user', 'created_at': '2026-09-24 15:50', 'text': '继续'})
            view.render(messages, {}, bottom=True)
            view.render(messages, {})
            QTest.qWait(80)
            self.assertEqual(bar.value(), bar.maximum())
        finally:
            view.close()
            view.deleteLater()
            app.processEvents()
