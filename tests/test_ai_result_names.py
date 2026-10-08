"""Readable artifact paths keep legacy references and safe unique writes."""
from datetime import datetime
from unittest.mock import patch

from test_ai_chat_selection import Fixtures
from Utils.AIWorkspace import Workspace, write_json


class ResultNameTests(Fixtures):
    def test_readable_names_collision_suffix_and_path_guards(self):
        store = Workspace(self.root)
        with patch('Utils.AIWorkspace.datetime') as clock:
            clock.now.return_value = datetime(2026, 9, 24, 16, 30, 5)
            first, folder, manifest = store.create('dataset', '样本导出', {'format': 'native'})
            (folder / 'keep.txt').write_text('first result', encoding='utf-8')
            second, _, _ = store.create('dataset', '样本导出', {'format': 'native'})
            crop, _, _ = store.create('crops', '网格切图', {'mode': 'tiles', 'tile_size': 1280})
            yolo, _, _ = store.create('dataset', 'YOLO数据集', {'format': 'yolo'})
            self.assertEqual(first, '图片与标注_2026-09-24_16-30-05')
            self.assertEqual(second, first + '_2')
            self.assertTrue(crop.startswith('切图_1280×1280_'))
            self.assertTrue(yolo.startswith('YOLO数据集_'))
            self.assertEqual((folder / 'keep.txt').read_text(encoding='utf-8'), 'first result')
            store.finish(folder, manifest, count=1)
            self.assertEqual(store.artifact(first)[0], folder)
        for name in ('../images', '切图/../images', 'C:\\outside', '切图:文件', '结果.记录'):
            with self.assertRaises(ValueError):
                store.location('artifacts', name)

    def test_legacy_ids_work_and_history_sorts_by_creation_time_not_name(self):
        store = Workspace(self.root)
        old_id = '20260924_151330_046905_c749f928'
        old_folder = store.location('artifacts', old_id)
        old_folder.mkdir(parents=True)
        write_json(old_folder / 'manifest.json', {'id': old_id, 'kind': 'crops', 'title': '旧切图',
            'status': 'completed', 'created_at': '2099-01-01 12:00:00'})
        store.create('dataset', '图片与标注', {'format': 'native'})
        self.assertEqual(store.artifact(old_id)[0], old_folder)
        self.assertEqual(store.list_entries(limit=1)[0]['id'], old_id)
