"""Result views use actual Qt widgets/files; no API or local inference needed."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QWheelEvent
from PyQt5.QtWidgets import QApplication, QGraphicsSimpleTextItem, QGraphicsRectItem, QMessageBox

from test_ai_chat_selection import Fixtures, rect
from Utils.AIWorkspace import Workspace, write_json, file_sha, read_json
from Utils.AIWorkTools import WorkTools
from Utils.AIResultBrowser import visual_rows


class ResultZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from Utils.AIResultBrowser import ResultView
        self.view = ResultView()
        self.view.resize(800, 600)
        pixmap = QPixmap(2296, 1748)
        pixmap.fill(Qt.gray)
        self.view.scene().addPixmap(pixmap)
        self.view.image_rect = QRectF(pixmap.rect())
        self.view.scene().setSceneRect(self.view.image_rect)
        self.view.show()
        self.app.processEvents()
        self.view.fit()
        self.app.processEvents()

    def tearDown(self):
        self.view.close()
        self.view.deleteLater()
        self.app.processEvents()

    def wheel(self, angle=0, pixels=0):
        point = self.view.viewport().rect().center()
        event = QWheelEvent(QPointF(point), QPointF(self.view.viewport().mapToGlobal(point)),
                            QPoint(0, pixels), QPoint(0, angle), Qt.NoButton, Qt.NoModifier,
                            Qt.NoScrollPhase, False)
        self.app.sendEvent(self.view.viewport(), event)
        self.app.processEvents()

    def test_wheel_enlarges_beyond_fit_despite_scrollbar_resize_and_can_shrink(self):
        initial = self.view.transform().m11()
        self.wheel(120)
        enlarged = self.view.transform().m11()
        self.assertGreater(enlarged, initial * 1.15)
        self.assertTrue(self.view.verticalScrollBar().maximum() > 0)
        self.wheel(120)
        self.assertAlmostEqual(self.view.transform().m11(), enlarged * 1.2)
        self.wheel(-120)
        self.assertAlmostEqual(self.view.transform().m11(), enlarged)

    def test_manual_zoom_survives_resize_and_fit_restores_automatic_fitting(self):
        self.wheel(120)
        zoom = self.view.transform().m11()
        self.view.resize(900, 700)
        self.app.processEvents()
        self.assertAlmostEqual(self.view.transform().m11(), zoom)
        self.view.fit()
        self.app.processEvents()
        fitted = self.view.transform().m11()
        self.view.resize(650, 500)
        self.app.processEvents()
        self.assertLess(self.view.transform().m11(), fitted)

    def test_zero_delta_does_not_shrink_and_pixel_wheel_can_enlarge(self):
        initial = self.view.transform().m11()
        self.wheel()
        self.assertAlmostEqual(self.view.transform().m11(), initial)
        self.wheel(pixels=20)
        self.assertGreater(self.view.transform().m11(), initial)


class ResultBrowserTests(Fixtures):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from Utils.test import TestWindow
        self.window = TestWindow(str(self.root))
        self.app.processEvents()
        self.browser = self.window.result_browser
        self.store = Workspace(self.root)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def candidates(self):
        aid, folder, manifest = self.store.create('candidates', '本地模型预标注候选', {'model': 'best.pt'})
        rows = []
        for i, rel in enumerate(self.paths[:3]):
            doc = copy.deepcopy(self.documents[Path(rel).stem])
            predicted = rect('脏污', .4, .5)
            predicted['confidence'] = .875
            rows.append({'path': rel, 'image_sha256': file_sha(self.root / rel), 'source_document': doc,
                         'document': {**doc, 'annotations': [predicted] if i == 1 else []}, 'can_adopt': False})
        write_json(folder / 'candidates.json', rows)
        write_json(folder / 'errors.json', [{'path': self.paths[3], 'error': 'fixture inference failure'}])
        return self.store.finish(folder, manifest, count=3, failed=1, predicted_annotations=1, adoptable=0, protected=3)

    def test_real_completion_opens_result_and_never_changes_original_canvas(self):
        result = self.candidates()
        before = self.hashes()
        paths, flags = list(self.window.relative_paths), list(self.window.flag)
        canvas = self.window.graphicsView.imageItem
        original = [a.to_dict() for a in canvas.annotations]
        self.window.show_ai_chat()
        self.window.ai_chat_dock.apply_local_result({'local_work': result, 'kind': 'preannotate', 'elapsed': 1})
        self.assertEqual(self.window.current_tree_filter, ('artifact', result['artifact_id'], None))
        self.assertEqual(self.browser.list.count(), 4)
        self.assertIn('1处预测', self.browser.caption.text())
        self.assertIn('预测 1 处', self.browser.heading.text())
        texts = [i.text() for i in self.browser.view.scene().items() if isinstance(i, QGraphicsSimpleTextItem)]
        self.assertEqual(texts, ['脏污 87.5%'])
        self.assertFalse(self.window.rect_action.isEnabled())
        self.assertEqual([a.to_dict() for a in canvas.annotations], original)
        self.assertEqual(paths, self.window.relative_paths)
        self.assertEqual(flags, self.window.flag)
        self.assertEqual(before, self.hashes())

    def test_prediction_filters_original_overlay_and_failure_are_distinct(self):
        self.browser.open(self.candidates()['artifact_id'])
        self.browser.filter.setCurrentIndex(1)
        self.assertEqual(self.browser.list.count(), 1)
        self.browser.original.setChecked(True)
        labels = [i.text() for i in self.browser.view.scene().items() if isinstance(i, QGraphicsSimpleTextItem)]
        self.assertTrue(any(t.startswith('原标注') for t in labels))
        self.browser.overlay.setChecked(False)
        self.assertTrue(all(t.text().startswith('原标注') for t in self.browser.view.scene().items() if isinstance(t, QGraphicsSimpleTextItem)))
        self.browser.filter.setCurrentIndex(2)
        self.assertEqual(self.browser.list.count(), 2)
        self.assertIn('不代表已确认为良品', self.browser.caption.text())
        self.browser.filter.setCurrentIndex(3)
        self.assertEqual(self.browser.list.count(), 1)
        self.assertIn('fixture inference failure', self.browser.caption.text())
        self.assertEqual(len(self.browser.view.scene().items()), 0)

    def test_same_filename_results_show_version_and_source_in_tree(self):
        ids = []
        for index in range(2):
            result = self.candidates()
            aid = result['artifact_id']
            folder, manifest = self.store.artifact(aid)
            manifest['metadata'].update(model_path=f'D:/models/exp{index}/weights/best.pt',
                                        model_sha256=str(index + 1) * 64)
            write_json(folder / 'manifest.json', manifest)
            self.browser.register(aid)
            ids.append(aid)
        self.window.rebuild_sample_tree()
        for index, aid in enumerate(ids):
            item = self.window.find_sample_tree_item(('artifact', aid, None))
            self.assertIn(str(index + 1) * 8, item.text(0))
            self.assertIn(f'exp{index}/best.pt', item.text(0))
            self.assertIn(f'D:/models/exp{index}/weights/best.pt', item.toolTip(0))
            self.assertIn(str(index + 1) * 64, item.toolTip(0))

    def test_filtered_other_classes_are_normal_and_do_not_count_as_project_predictions(self):
        result = self.candidates()
        path = Path(result['output']) / 'candidates.json'
        rows = read_json(path)
        rows[0]['ignored_classes'] = 3
        write_json(path, rows)
        self.browser.open(result['artifact_id'])
        self.browser.list.setCurrentRow(0)
        self.assertIn('过滤 3 处', self.browser.heading.text())
        self.assertIn('过滤 3 处', self.browser.caption.text())
        self.assertIn('项目类别预测 0 处', self.browser.caption.text())
        self.assertNotIn('不完整', self.browser.heading.text())
        self.assertNotIn('重新推理', self.browser.caption.text())
        self.assertIn('无预测', self.browser.list.item(0).text())
        self.browser.filter.setCurrentIndex(2)
        self.assertEqual(self.browser.list.count(), 2)

    def test_generated_images_auto_open_and_return_to_editing_preserves_categories(self):
        candidate_id = self.candidates()['artifact_id']
        self.browser.open(candidate_id)
        self.browser.overlay.setChecked(False)
        self.browser.original.setChecked(True)
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        tools = WorkTools(self.root, self.paths, context={'current_paths': self.paths[:2]})
        result = tools.execute('crop', {'mode': 'tiles', 'tile_size': 1280})
        before = self.hashes()
        categories = list(self.window.graphicsView.imageItem.existing_categories)
        dock.start_revision = self.window.temporary_selection_revision
        dock.pending_message = '切图'
        dock.apply_work_result({'work': {'steps': tools.steps, 'effects': [], 'pending': None, 'status': 'completed'},
                               'result': {'message': '完成'}, 'usage': {}, 'requests': 1, 'elapsed': 1, 'run_id': 'fixture'})
        self.assertEqual(self.browser.active_id, result['artifact_id'])
        self.assertIn('1280×1280', self.browser.caption.text())
        self.assertEqual(self.browser.list.count(), 2)
        self.assertTrue(self.browser.heading.isHidden())
        self.assertTrue(self.browser.controls.isHidden())
        labels = [i.text() for i in self.browser.view.scene().items() if isinstance(i, QGraphicsSimpleTextItem)]
        self.assertTrue(labels)
        self.assertTrue(all(not text.startswith('原标注') for text in labels))
        with patch('Utils.ProjectContextDialog.record_feedback') as record:
            self.browser.feedback_action.trigger()
            self.assertEqual(record.call_args.args[2], result['artifact_id'])
            self.assertEqual(len(record.call_args.args[1]), 1)
        self.browser.list.setCurrentRow(1)
        self.assertEqual(before, self.hashes())
        self.browser.open(candidate_id)
        self.assertFalse(self.browser.heading.isHidden())
        self.assertFalse(self.browser.controls.isHidden())
        self.assertFalse(self.browser.original.isChecked())
        self.assertFalse(self.browser.overlay.isChecked())
        self.browser.overlay.setChecked(True)
        self.assertTrue(any(isinstance(i, QGraphicsSimpleTextItem) for i in self.browser.view.scene().items()))
        self.window.on_sample_tree_item_clicked(self.window.find_sample_tree_item(('all', None, None)), 0)
        self.assertIsNone(self.browser.active_id)
        self.assertTrue(self.window.rect_action.isEnabled())
        self.assertEqual(self.window.graphicsView.imageItem.existing_categories, categories)
        self.assertEqual(len(self.window.relative_paths), 5)

    def test_existing_results_restore_on_reopen_and_remove_only_hides_entry(self):
        result = self.candidates()
        aid = result['artifact_id']
        from Utils.test import TestWindow
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.window = TestWindow(str(self.root))
        self.browser = self.window.result_browser
        self.assertIn(aid, self.browser.entries)
        self.assertIsNotNone(self.window.find_sample_tree_item(('artifact', aid, None)))
        self.browser.open(aid)
        self.browser.remove(aid)
        self.assertNotIn(aid, self.browser.entries)
        self.assertEqual(self.window.current_tree_filter[0], 'all')
        self.assertTrue((Path(result['output']) / 'candidates.json').exists())
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.window = TestWindow(str(self.root))
        self.browser = self.window.result_browser
        self.assertNotIn(aid, self.browser.entries)
        self.assertTrue(self.browser.open(aid))
        self.assertFalse(read_json(Path(result['output']) / 'manifest.json')['hidden'])

    def test_delete_dialog_cancel_keeps_files_and_active_preview(self):
        result = self.candidates()
        aid = result['artifact_id']
        self.browser.open(aid)
        with patch.object(QMessageBox, 'exec_', return_value=0):
            self.assertFalse(self.browser.delete(aid))
        self.assertEqual(self.browser.active_id, aid)
        self.store.artifact(aid)
        self.assertTrue((Path(result['output']) / 'candidates.json').exists())

    def test_delete_prediction_removes_only_result_and_history_disables_reuse(self):
        result = self.candidates()
        aid = result['artifact_id']
        before = self.hashes()
        self.browser.open(aid)
        def confirm(box):
            self.assertEqual(box.windowTitle(), '删除处理结果')
            self.assertIn('预测框、类别、置信度', box.informativeText())
            self.assertIn('原图片不会删除', box.informativeText())
            self.assertEqual(box.defaultButton().text(), '取消')
            next(b for b in box.buttons() if box.buttonRole(b) == QMessageBox.DestructiveRole).click()
        with patch.object(QMessageBox, 'exec_', confirm):
            self.assertTrue(self.browser.delete(aid))
        self.assertEqual(before, self.hashes())
        self.assertNotIn(aid, self.browser.entries)
        self.assertIsNone(self.browser.active_id)
        self.assertTrue(self.window.rect_action.isEnabled())
        self.assertEqual(self.browser.rows, [])
        self.assertEqual(len(self.browser.view.scene().items()), 0)
        self.assertFalse((Path(result['output']) / 'candidates.json').exists())
        self.window.show_ai_chat()
        from Utils.AIWorkDialogs import WorkHistoryDialog
        dialog = WorkHistoryDialog(self.window.ai_chat_dock)
        self.assertIn('结果已删除', dialog.summary.toPlainText())
        self.assertFalse(dialog.use_button.isEnabled())
        self.assertFalse(dialog.delete_button.isEnabled())
        dialog.close()

    def test_hidden_result_can_be_restored_or_deleted_from_history(self):
        result = self.candidates()
        aid = result['artifact_id']
        self.browser.open(aid)
        self.browser.hide(aid)
        self.window.show_ai_chat()
        from Utils.AIWorkDialogs import WorkHistoryDialog
        dialog = WorkHistoryDialog(self.window.ai_chat_dock)
        self.assertIn('已隐藏', dialog.table.item(0, 3).text())
        self.assertEqual(dialog.use_button.text(), '重新显示隐藏结果')
        self.assertTrue(dialog.delete_button.isEnabled())
        dialog.use_entry()
        self.assertEqual(self.browser.active_id, aid)
        dialog.close()

    def test_delete_crops_confirmation_describes_real_generated_files(self):
        result = WorkTools(self.root, self.paths, context={'current_paths': self.paths[:2]}).execute(
            'crop', {'mode': 'tiles', 'tile_size': 1280})
        aid = result['artifact_id']
        before = self.hashes()
        self.browser.open(aid)
        def confirm(box):
            self.assertIn('2 个切图/裁剪图片文件', box.informativeText())
            self.assertIn('对应标注', box.informativeText())
            next(b for b in box.buttons() if box.buttonRole(b) == QMessageBox.DestructiveRole).click()
        with patch.object(QMessageBox, 'exec_', confirm):
            self.assertTrue(self.browser.delete(aid))
        self.assertEqual(before, self.hashes())
        self.assertEqual([p.name for p in Path(result['output']).iterdir()], ['manifest.json'])

    def test_view_result_tool_opens_saved_output_without_api_or_inference(self):
        result = self.candidates()
        tools = WorkTools(self.root, self.paths)
        opened = tools.execute('view_result', {'artifact_id': result['artifact_id']})
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        dock.start_revision = self.window.temporary_selection_revision
        dock.pending_message = '查看刚才推理结果'
        with patch('Utils.AIAugment.request_qwen_json', side_effect=AssertionError('must not request API')):
            dock.apply_work_result({'work': {'steps': tools.steps, 'effects': [], 'pending': None, 'status': 'completed'},
                                   'result': {'message': '打开结果'}, 'usage': {}, 'requests': 1, 'elapsed': 1, 'run_id': 'fixture'})
        self.assertEqual(self.browser.active_id, opened['artifact_id'])
        context = dock.work_context()
        self.assertEqual(context['current_paths'], [])
        self.assertEqual(context['selected_paths'], [])
        self.assertIsNone(context['current_image'])
        with self.assertRaisesRegex(ValueError, '模型候选需先核对采用'):
            WorkTools(self.root, self.paths, context=context).execute('crop', {'mode': 'tiles'})

    def test_changed_or_missing_sources_and_corrupt_result_do_not_show_stale_boxes(self):
        result = self.candidates()
        self.browser.open(result['artifact_id'])
        (self.root / self.paths[1]).write_bytes(b'changed')
        self.browser.show_row(self.browser.list.currentRow())
        self.assertIn('已经改变', self.browser.caption.text())
        self.assertEqual(len(self.browser.view.scene().items()), 0)
        write_json(Path(result['output']) / 'candidates.json', [{'bad': True}])
        self.browser.open(result['artifact_id'])
        self.assertIn('无法读取', self.browser.heading.text())
        self.assertEqual(self.browser.list.count(), 0)

    def test_result_paths_cannot_escape_artifact(self):
        tools = WorkTools(self.root, self.paths, context={'current_paths': self.paths[:1]})
        result = tools.execute('crop', {'label': '脏污'})
        folder = Path(result['output'])
        write_json(folder / 'sources.json', [{'image': '../../../../images/small.png'}])
        with self.assertRaises(ValueError):
            visual_rows(self.store, result['artifact_id'])


if __name__ == '__main__':
    unittest.main()
