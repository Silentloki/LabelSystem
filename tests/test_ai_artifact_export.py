"""Generated-image export, cross-turn routing and original-data protection."""
import copy
from collections import defaultdict
from pathlib import Path

from PIL import Image
from test_ai_workbench import WorkFixtures
from Utils.AIConversations import result_card
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkAgent import run_agent
from Utils.AIWorkspace import read_json, write_json, file_sha
from Utils.AIExportStorage import result_file


class ArtifactExportTests(WorkFixtures):
    def setUp(self):
        super().setUp()
        for i, path in enumerate(self.paths):
            Image.new('RGB', (100, 200), (40 * i, 120, 200)).save(self.root / path)
        self.context['current_paths'] = self.paths[:4]
        self.crop = self.call('crop', mode='tiles', tile_size=64, overlap=0)
        self.folder = Path(self.crop['output'])
        self.index = read_json(self.folder / 'sources.json')
        self.context.update(viewing_artifact=self.crop['artifact_id'], current_paths=[],
                            current_result_count=self.crop['count'])
        self.tools = WorkTools(self.root, self.paths, context=self.context)
        self.originals = self.hashes()

    def test_current_result_yolo_and_native_use_tiles_and_their_own_labels(self):
        source_hashes = {p: file_sha(p) for p in self.folder.rglob('*') if p.is_file()}
        yolo = self.call('export', scope='current_result')
        native = self.call('export', format='native')
        self.assertEqual(yolo['count'], self.crop['count'])
        self.assertEqual(native['count'], self.crop['count'])
        self.assertEqual(yolo['classes'], self.context['classes'])
        self.assertIsNone(native['splits'])
        self.assertNotIn('训练集', result_card(native)['summary'])
        groups = defaultdict(set)
        for result in (yolo, native):
            root = Path(result['output'])
            rows = read_json(root / 'sources.json')
            manifest = read_json(root / 'manifest.json')
            for row in rows:
                output_image = result_file(self.store, root, manifest, row, 'image')
                with Image.open(output_image) as image:
                    self.assertEqual(image.size, (64, 64))
                self.assertEqual(file_sha(output_image), file_sha(self.root / row['source']))
                expected = read_json(self.folder / 'jsons' / (Path(row['source']).stem + '.json'))
                self.assertEqual(read_json(result_file(self.store, root, manifest, row, 'json')), expected)
                if result is yolo:
                    groups[row['original_source']].add(row['split'])
            self.assertTrue(any(not row['document']['annotations'] for row in rows))
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(self.originals, self.hashes())
        self.assertEqual(source_hashes, {p: file_sha(p) for p in source_hashes})
        from Utils.AnnotationImporter import inspect_yolo_dataset
        inspected = inspect_yolo_dataset(yolo['dataset_dir'])
        self.assertFalse(inspected['errors'])
        self.assertEqual(inspected['summary']['ready_records'], self.crop['count'])

    def test_logged_cross_turn_view_then_two_exports_completes_without_api(self):
        actions = iter([
            {'action': 'tool', 'tool': 'view_result', 'args': {'artifact_id': self.crop['artifact_id']}},
            {'action': 'tool', 'tool': 'export', 'args': {'scope': 'current_result', 'format': 'yolo'}},
            {'action': 'tool', 'tool': 'export', 'args': {'scope': 'last', 'format': 'native'}},
            {'action': 'reply', 'message': '两种导出已完成。'}])
        result = run_agent(self.root, self.paths, {}, {}, [], '将切图分别导出为数据和数据集',
                           ('artifact', self.crop['artifact_id'], None), ('fake', 'url', 'model'),
                           context=self.context, request=lambda *a, **kw: (next(actions), {}))
        self.assertEqual(result['work']['status'], 'completed')
        steps = result['work']['steps']
        self.assertEqual(steps[0]['result']['source_count'], 4)
        self.assertEqual([step['result']['count'] for step in steps[1:]], [self.crop['count']] * 2)
        self.assertEqual(self.originals, self.hashes())

    def test_select_crop_last_exports_crop_not_previous_original_selection(self):
        self.context.update(viewing_artifact=None, current_paths=self.paths[:4])
        tools = WorkTools(self.root, self.paths, context=self.context)
        tools.execute('select', {'code': 'result = [r["id"] for r in records]',
                                 'title': '原图分组', 'criteria': '所有当前原图'})
        crop = tools.execute('crop', {'scope': 'last', 'mode': 'tiles', 'tile_size': 64})
        for format in ('native', 'yolo'):
            result = tools.execute('export', {'scope': 'last', 'format': format})
            self.assertEqual(result['count'], crop['count'])
            self.assertEqual(result['source_artifact'], crop['artifact_id'])

    def test_missing_deleted_and_escaping_results_never_fall_back_to_originals(self):
        index_path = self.folder / 'sources.json'
        original = index_path.read_bytes()
        malformed = copy.deepcopy(self.index)
        malformed[0]['image'] = '../../../../images/small.png'
        write_json(index_path, malformed)
        with self.assertRaises(ValueError):
            self.call('export', scope='current_result')
        index_path.write_bytes(original)
        annotation = self.folder / self.index[0]['json']
        saved = annotation.read_bytes()
        annotation.unlink()
        with self.assertRaises(OSError):
            self.call('export', scope='current_result')
        annotation.write_bytes(saved)
        manifest_path = self.folder / 'manifest.json'
        manifest = read_json(manifest_path)
        manifest['status'] = 'deleted'
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, '已删除'):
            self.call('export', scope='current_result')
        self.assertEqual(self.originals, self.hashes())

    def test_export_support_does_not_allow_mutating_originals_from_result_view(self):
        with self.assertRaisesRegex(ValueError, '派生图片'):
            self.call('relabel', old='脏污', new='改名')
        with self.assertRaisesRegex(ValueError, '预览处理产物'):
            self.call('export', scope='selected')
        # Recropping is supported, but must operate on the tiles themselves.
        recropped = self.call('crop', mode='tiles', tile_size=64, overlap=0)
        self.assertEqual(recropped['count'], self.crop['count'])
        sources = read_json(Path(recropped['output']) / 'sources.json')
        self.assertTrue(all('/images/crop_' in row['source'] for row in sources))
        self.assertEqual(self.originals, self.hashes())
