"""Public export layout, scope/content isolation and owned-file deletion."""
import copy
import shutil
import stat
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from test_ai_workbench import WorkFixtures
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkspace import read_json, write_json, file_sha, Workspace
from Utils.AIExportStorage import export_root, result_file
from Utils.AIResultStorage import deletion_preview, delete_result
from Utils.AIResultBrowser import visual_rows
from Utils.AIWorkAgent import run_agent


class ExportLayoutTests(WorkFixtures):
    def setUp(self):
        super().setUp()
        self.context['current_paths'] = self.paths[:4]

    def test_dataset_and_annotated_layout_and_reexport(self):
        before = self.hashes()
        for fmt, category in [('native', 'annotated'), ('yolo', 'dataset'), ('yolo_detect', 'dataset')]:
            with self.subTest(format=fmt):
                args = {'format': fmt, 'scope': 'current'}
                if fmt == 'yolo_detect':
                    args['train_ratio'] = 1
                result = self.call('export', **args)
                output = Path(result['export_dir'])
                if category == 'dataset':
                    self.assertEqual(output.parent, self.root)
                    self.assertRegex(output.name, r'^dataset_AI_\d{4}-\d{2}-\d{2}_')
                    self.assertTrue((output / 'data.yaml').is_file())
                    self.assertTrue((output / 'labels').is_dir())
                else:
                    self.assertEqual(output.parent, self.root / 'exports' / 'annotated')
                    self.assertEqual({p.name for p in output.iterdir()}, {'images', 'jsons'})
                folder, manifest = self.tools.store.artifact(result['artifact_id'])
                self.assertFalse(any(p.suffix == '.png' for p in folder.rglob('*')))
                self.assertEqual(len(visual_rows(self.tools.store, result['artifact_id'])[1]), result['count'])
                again = self.call('export', scope='artifact:' + result['artifact_id'], content='annotated')
                self.assertEqual(again['count'], result['count'])
                self.assertNotEqual(again['export_dir'], result['export_dir'])
        self.assertEqual(before, self.hashes())

    def test_images_only_accepts_unannotated_and_labels_only_does_not_copy_images(self):
        before = self.hashes()
        images = self.call('export', scope='all', content='images')
        target = Path(images['export_dir'])
        self.assertEqual(target.parent, self.root / 'exports/images')
        self.assertEqual(len(list(target.iterdir())), len(self.paths))
        self.assertTrue(all(p.suffix == '.png' for p in target.iterdir()))
        labels = self.call('export', content='labels')
        target = Path(labels['export_dir'])
        self.assertEqual(target.parent, self.root / 'exports/labels')
        self.assertTrue(all(p.suffix == '.json' for p in target.iterdir()))
        self.assertEqual(read_json(target / 'small.json'), self.documents['small'])
        self.assertEqual(before, self.hashes())

    def test_selected_tiles_are_not_expanded_and_empty_or_stale_selection_fails(self):
        crop = self.call('crop', mode='tiles', tile_size=64, overlap=0)
        _, rows = visual_rows(self.tools.store, crop['artifact_id'])
        selected = [Path(r['image']).relative_to(self.root).as_posix() for r in rows[1:4]]
        context = {**self.context, 'viewing_artifact': crop['artifact_id'], 'selected_result_paths': selected}
        tools = WorkTools(self.root, self.paths, context=context)
        for content in ('images', 'labels', 'annotated', 'dataset'):
            result = tools.execute('export', {'scope': 'selected', 'content': content})
            self.assertEqual(result['count'], 3)
        for invalid in ([], self.paths[:1], selected + selected):
            tools.context['selected_result_paths'] = invalid
            with self.assertRaises(ValueError):
                tools.execute('export', {'scope': 'selected', 'content': 'images'})

    def test_predictions_keep_confidence_and_never_become_training_labels(self):
        before = self.hashes()
        aid, folder, manifest = self.tools.store.create('candidates', '测试推理')
        doc = copy.deepcopy(self.documents['large'])
        doc['annotations'][0]['confidence'] = .87
        write_json(folder / 'candidates.json', [{'path': self.paths[1], 'document': doc,
                    'source_document': self.documents['large'], 'image_sha256': file_sha(self.root / self.paths[1])}])
        write_json(folder / 'errors.json', [])
        self.tools.store.finish(folder, manifest, count=1)
        for content in ('predictions', 'preview', 'images'):
            result = self.call('export', scope='artifact:' + aid, content=content)
            target = Path(result['export_dir'])
            self.assertEqual(len(list(target.iterdir())), 1)
            if content == 'predictions':
                exported = read_json(next(target.iterdir()))
                self.assertFalse(exported['adopted'])
                self.assertEqual(exported['document'], doc)
            if content == 'preview':
                with Image.open(next(target.iterdir())) as image:
                    self.assertEqual(image.size, (100, 200))
                self.assertNotEqual(file_sha(next(target.iterdir())), file_sha(self.root / self.paths[1]))
        for content in ('dataset', 'annotated', 'labels'):
            with self.assertRaises(ValueError):
                self.call('export', scope='artifact:' + aid, content=content)
        self.assertEqual(before, self.hashes())

    def test_delete_covers_public_batch_and_keeps_sources_and_other_exports(self):
        before = self.hashes()
        result = self.call('export', content='dataset')
        other = self.call('export', content='images')
        public = Path(result['export_dir'])
        saved = {p: file_sha(p) for p in Path(other['export_dir']).rglob('*') if p.is_file()}
        plan = deletion_preview(self.tools.store, result['artifact_id'])
        self.assertEqual(plan['export_folder'], str(public))
        self.assertTrue(any(f['path'].startswith('@export/images/') for f in plan['files']))
        (public / 'added.txt').write_text('new')
        with self.assertRaisesRegex(ValueError, '确认期间'):
            delete_result(self.tools.store, result['artifact_id'], plan['token'])
        plan = deletion_preview(self.tools.store, result['artifact_id'])
        delete_result(self.tools.store, result['artifact_id'], plan['token'])
        self.assertFalse(public.exists())
        self.assertEqual(before, self.hashes())
        self.assertEqual(saved, {p: file_sha(p) for p in saved})

    def test_tampered_public_path_rejected_and_project_move_keeps_references(self):
        result = self.call('export', format='native')
        folder, manifest = self.tools.store.artifact(result['artifact_id'])
        altered = {**manifest, 'export_relative': 'images'}
        write_json(folder / 'manifest.json', altered)
        with self.assertRaises(ValueError):
            deletion_preview(self.tools.store, result['artifact_id'])
        write_json(folder / 'manifest.json', manifest)
        moved = self.base / (self.root.name + '_moved')
        try:
            shutil.copytree(self.root, moved)
            _, rows = visual_rows(Workspace(moved), result['artifact_id'])
            self.assertTrue(all(Path(row['image']).is_relative_to(moved) for row in rows))
            self.assertTrue(all(Path(row['image']).exists() for row in rows))
        finally:
            moved.resolve().relative_to(self.base.resolve())
            if moved.exists():
                shutil.rmtree(moved)

    def test_agent_export_content_and_scope_are_independent(self):
        context = {**self.context, 'selected_paths': self.paths[:1]}
        replies = iter([
            {'action': 'tool', 'tool': 'export', 'args': {'scope': 'selected', 'content': 'images'}},
            {'action': 'tool', 'tool': 'export', 'args': {'scope': 'current', 'content': 'labels'}},
            {'action': 'reply', 'message': '已分别导出。'}])
        result = run_agent(self.root, self.paths, {}, {}, [], '选中图片只导图片，当前范围只导标注',
            ('all', None, None), ('fake', 'url', 'model'), context=context,
            request=lambda *a, **k: (next(replies), {}))
        self.assertEqual(result['work']['status'], 'completed')
        self.assertEqual([s['result']['count'] for s in result['work']['steps']], [1, 4])

    def test_public_reparse_point_and_existing_batch_are_never_overwritten(self):
        from datetime import datetime
        public = self.root / 'dataset_AI_2026-09-24_16-30-05'
        public.mkdir()
        protected = public / 'keep.txt'
        protected.write_text('existing')
        with patch('Utils.AIWorkspace.datetime') as clock:
            clock.now.return_value = datetime(2026, 9, 24, 16, 30, 5)
            with self.assertRaisesRegex(ValueError, '未覆盖'):
                self.call('export')
            result = self.call('export')
            self.assertTrue(result['export_dir'].endswith('_2'))
        self.assertEqual(protected.read_text(), 'existing')
        original = Path.lstat
        def junction(path, *args, **kwargs):
            if path == Path(result['export_dir']) / 'images':
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'lstat', junction), self.assertRaisesRegex(ValueError, '目录联接'):
            deletion_preview(self.store, result['artifact_id'])

    def test_failed_copy_does_not_publish_partial_export(self):
        before = self.hashes()
        with patch('Utils.AIDataExport.shutil.copyfile', side_effect=OSError('fixture disk failure')):
            with self.assertRaises(OSError):
                self.call('export', content='images')
        self.assertFalse((self.root / 'exports').exists())
        self.assertEqual(self.store.list_entries()[0]['status'], 'failed')
        self.assertEqual(before, self.hashes())
