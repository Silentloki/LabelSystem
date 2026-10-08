"""Parity with the existing desktop dataset path, including negatives/subclasses."""
import csv
import random
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from test_ai_chat_selection import Fixtures, rect
from Utils.AIWorkTools import WorkTools
from Utils.AIWorkspace import read_json, write_json
from Utils.Task import Worker, build_subsample_stratified_split


class DatasetExportTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.paths = []
        self.flags = []
        self.assignments = {}
        self.names = ['破损', '脏污']
        for i in range(27):
            path = f'images/export_{i:02d}.png'
            self.paths.append(path)
            self.flags.append(1 if i < 20 else 2 if i < 23 else 3 if i < 26 else 0)
            Image.new('RGB', (30, 40), (i, 100, 200)).save(self.root / path)
            if i < 20:
                name = self.names[i % 2]
                write_json(self.root / 'jsons' / (Path(path).stem + '.json'), {
                    'image_width': 30, 'image_height': 40, 'annotations': [rect(name, .6, .7)]})
                self.assignments[path] = {name: '小类' if i < 6 else '大类'}
        self.context = {'classes': self.names, 'flags': dict(zip(self.paths, self.flags)),
                        'sample_assignments': self.assignments,
                        'sample_groups': {n: ['小类', '大类'] for n in self.names}}
        self.tools = WorkTools(self.root, self.paths, context=self.context)

    def test_default_matches_desktop_split_names_labels_and_metadata(self):
        before = self.hashes()
        result = self.tools.execute('export', {'scope': 'all'})
        root = Path(result['dataset_dir'])
        self.assertTrue(root.name.startswith('dataset_AI_'))
        self.assertEqual(root.parent, self.root)
        self.assertEqual(result['count'], 26)
        self.assertEqual(result['skipped_unreviewed'], 1)
        labels = {i: {self.names[i % 2]} for i in range(20)}
        # Exercise the existing desktop call, with the same shuffle sequence.
        with patch('Utils.Task.random', random.Random(42)):
            train, val, _ = build_subsample_stratified_split(
                self.flags, 1, self.paths, self.names, self.assignments, list(range(20, 26)), labels)
        rows = read_json(Path(result['output']) / 'sources.json')
        self.assertEqual({r['source'] for r in rows if r['split'] == 'train'}, {self.paths[i] for i in train})
        self.assertEqual({r['source'] for r in rows if r['split'] == 'val'}, {self.paths[i] for i in val})
        self.assertEqual(result['splits'], {'train': len(train), 'val': len(val)})
        # Also run the actual old writer using that split and compare all outputs.
        worker = Worker(str(self.root), self.paths, self.flags, self.names, 1,
                        good=list(range(20, 26)), sample_groups=self.context['sample_groups'],
                        split_indices=(train, val), sample_assignments=self.assignments)
        worker.run()
        manual = self.root / 'dataset'
        for file in manual.rglob('*'):
            if file.is_file():
                self.assertEqual(file.read_bytes(), (root / file.relative_to(manual)).read_bytes(), str(file))
        from Utils.AnnotationImporter import inspect_yolo_dataset
        inspected = inspect_yolo_dataset(root)
        self.assertFalse(inspected['errors'])
        self.assertEqual(inspected['summary']['ready_records'], 26)
        self.assertEqual(before, self.hashes())
        with (root / 'subclass_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 20)
        from Utils.AIResultBrowser import visual_rows
        self.assertEqual(len(visual_rows(self.tools.store, result['artifact_id'])[1]), 26)

    def test_repeated_export_preserves_previous_dataset_and_scope(self):
        self.context['current_paths'] = self.paths[:3]
        first = self.tools.execute('export', {})
        yaml = Path(first['data_yaml']).read_bytes()
        second = self.tools.execute('export', {})
        self.assertNotEqual(first['dataset_dir'], second['dataset_dir'])
        self.assertEqual(Path(first['data_yaml']).read_bytes(), yaml)
        self.assertEqual(second['count'], 3)

    def test_original_json_requires_explicit_native(self):
        self.context['current_paths'] = self.paths[:2]
        result = self.tools.execute('export', {'format': 'native'})
        self.assertIsNone(result['data_yaml'])
        self.assertTrue((Path(result['output']) / 'jsons').is_dir())

    def test_unreviewed_only_and_name_collisions_rejected(self):
        self.context['current_paths'] = self.paths[-1:]
        with self.assertRaisesRegex(ValueError, '没有已标注'):
            self.tools.execute('export', {})
        duplicate = 'images/export_00.jpg'
        Image.new('RGB', (30, 40)).save(self.root / duplicate)
        self.paths.append(duplicate)
        tools = WorkTools(self.root, self.paths, context=self.context)
        with self.assertRaisesRegex(ValueError, '同名'):
            tools.execute('export', {'scope': 'all'})

    def test_polygons_preserve_vertices_in_default_dataset(self):
        path = self.root / 'jsons/export_00.json'
        doc = read_json(path)
        doc['annotations'] = [{'type': 'polygon', 'lable': self.names[0],
                               'points': [{'x': .1, 'y': .1}, {'x': .8, 'y': .2}, {'x': .4, 'y': .9}]}]
        write_json(path, doc)
        self.context['current_paths'] = self.paths[:1]
        result = self.tools.execute('export', {})
        label = next((Path(result['dataset_dir']) / 'labels').rglob('*.txt'))
        self.assertEqual(label.read_text().split(), ['0', '0.1', '0.1', '0.8', '0.2', '0.4', '0.9'])
