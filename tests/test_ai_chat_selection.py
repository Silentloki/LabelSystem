"""Real local selection/UI tests; provider responses are simulated, no paid calls."""
import copy
import hashlib
import json
import os
import pickle
import shutil
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Utils.AIChatSelection import annotation_features, project_snapshot, process_request, make_prompt
from Utils.AISelectionCode import SelectionProgram, SelectionCodeError


AREA_CODE = """matches = [r for r in records if any(a['label'] == '脏污' and a['area_ratio'] > 0.05 for a in r['annotations'])]
ranked = sorted(matches, key=lambda r: max(a['area_ratio'] for a in r['annotations'] if a['label'] == '脏污'), reverse=True)
result = [r['id'] for r in ranked]
"""


def rect(label, x, y):
    return {"type": "rect", "lable": label, "points": [{"x": 0, "y": 0}, {"x": x, "y": y}]}


class Fixtures(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.base = Path(__file__).resolve().parent / ".tmp_ai_chat"
        self.root = self.base / uuid.uuid4().hex
        (self.root / "images").mkdir(parents=True)
        (self.root / "jsons").mkdir()
        self.paths = [f"images/{name}.png" for name in ("small", "large", "largest", "other", "missing")]
        for path in self.paths:
            Image.new("RGB", (100, 200), (240, 242, 244)).save(self.root / path)
        self.documents = {}
        for name, annotations in (
            ("small", [rect("脏污", .1, .1), rect("脏污", .1, .1)]),
            ("large", [rect("脏污", .4, .4)]),
            ("largest", [rect("脏污", .7, .7), rect("破损", .9, .9)]),
            ("other", [rect("破损", .8, .8)]),
        ):
            doc = {"image_width": 100, "image_height": 200, "annotations": annotations}
            (self.root / "jsons" / (name + ".json")).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            self.documents[name] = doc
        (self.root / "label.txt").write_text("脏污\n破损\n", encoding="utf-8")
        (self.root / "datafile.dat").write_bytes(pickle.dumps(self.paths))
        (self.root / "flagfile.dat").write_bytes(pickle.dumps([1, 1, 1, 1, 0]))
        (self.root / "config.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        target = self.root.resolve()
        target.relative_to(self.base.resolve())
        shutil.rmtree(target)

    def snapshot(self):
        return project_snapshot(self.root, self.paths)

    def result(self, code=AREA_CODE, group_id=None):
        return {"action": "select", "message": "按最大脏污面积占比筛选。", "title": "大面积脏污",
                "criteria": "单个脏污标注面积占整图比例 > 5%，按该比例降序", "code": code, "group_id": group_id}

    def hashes(self):
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ("images", "jsons") for p in (self.root / folder).iterdir()}


class SelectionTests(Fixtures):
    def test_geometry_polygon_rect_and_pixels(self):
        feature = annotation_features(rect("脏污", .4, .5), 100, 200)
        self.assertAlmostEqual(feature["area_ratio"], .2)
        self.assertAlmostEqual(feature["area_pixels"], 4000)
        polygon = {"type": "polygon", "label": "脏污", "points": [
            {"x": 0, "y": 0}, {"x": .8, "y": 0}, {"x": 0, "y": .5}]}
        self.assertAlmostEqual(annotation_features(polygon, 100, 200)["area_ratio"], .2)
        polygon["points"].reverse()
        self.assertAlmostEqual(annotation_features(polygon, 100, 200)["area_ratio"], .2)

    def test_invalid_annotations_exclude_image_and_report_warning(self):
        path = self.root / "jsons" / "small.json"
        path.write_text('{"annotations":[{}]}', encoding="utf-8")
        snapshot = self.snapshot()
        self.assertEqual(snapshot["summary"]["skipped_images"], 2)
        self.assertNotIn("images/small.png", [r["id"] for r in snapshot["records"]])

    def test_read_and_execute_does_not_write_files_or_upload_records(self):
        before = self.hashes()
        snapshot = self.snapshot()
        prompt = make_prompt(snapshot, {}, [], "挑选大面积脏污", ("all", None, None))
        self.assertNotIn("images/large.png", prompt)
        self.assertNotIn(str(self.root), prompt)
        self.assertEqual(SelectionProgram().run(AREA_CODE, snapshot["records"], {}),
                         ["images/largest.png", "images/large.png"])
        self.assertEqual(before, self.hashes())
        self.assertEqual(snapshot["summary"]["classes"]["脏污"]["images"], 3)
        self.assertEqual(snapshot["summary"]["classes"]["脏污"]["annotations"], 4)

    def test_followup_top_one_uses_group_and_duplicate_ids_deduplicate(self):
        snapshot = self.snapshot()
        code = "matches = [r for r in records if r['id'] in groups['g']]\n" + AREA_CODE.replace(
            "r for r in records", "r for r in matches").replace("for r in ranked]", "for r in ranked[:1]]")
        self.assertEqual(SelectionProgram().run(code, snapshot["records"], {"g": ["images/large.png", "images/largest.png"]}),
                         ["images/largest.png"])
        self.assertEqual(SelectionProgram().run("result = ['images/large.png', 'images/large.png']", snapshot["records"], {}),
                         ["images/large.png"])

    def test_relative_area_default_top_twenty_percent_and_empty(self):
        code = AREA_CODE.replace(" and a['area_ratio'] > 0.05", "").replace(
            "for r in ranked]", "for r in ranked[:max(1, (len(ranked) + 4) // 5)]]")
        self.assertEqual(SelectionProgram().run(code, self.snapshot()["records"], {}), ["images/largest.png"])
        self.assertEqual(SelectionProgram().run(AREA_CODE.replace("脏污", "不存在"), self.snapshot()["records"], {}), [])

    def test_arbitrary_io_unknown_ids_and_excessive_work_are_rejected(self):
        records = self.snapshot()["records"]
        for code in ("import os\nresult=[]", "result = open('file')", "result = records.__class__",
                     "result = __import__('os').system('echo unsafe')", "result = ['not-in-project']",
                     "result = ['x'] * 1000000000", "while True: pass", "result = 2 ** 99999999"):
            with self.subTest(code=code), self.assertRaises(SelectionCodeError):
                SelectionProgram().run(code, records, {})
        with self.assertRaises(SelectionCodeError):
            SelectionProgram(max_steps=10).run(AREA_CODE, records, {})
        with self.assertRaises(SelectionCodeError):
            SelectionProgram(cancelled=lambda: True).run(AREA_CODE, records, {})

    def test_snapshot_overrides_current_unsaved_annotation_without_writes(self):
        before = self.hashes()
        doc = copy.deepcopy(self.documents["small"])
        doc["annotations"] = [rect("脏污", .9, .9)]
        snapshot = project_snapshot(self.root, self.paths, {"images/small.png": doc})
        self.assertEqual(SelectionProgram().run(AREA_CODE, snapshot["records"], {})[0], "images/small.png")
        self.assertEqual(before, self.hashes())

    def test_corrupt_json_cannot_be_masked_by_empty_canvas_override(self):
        (self.root / "jsons" / "small.json").write_text("{broken", encoding="utf-8")
        snapshot = project_snapshot(self.root, self.paths, {"images/small.png": {
            "image_width": 100, "image_height": 200, "annotations": []}})
        self.assertNotIn("images/small.png", [r["id"] for r in snapshot["records"]])

    def test_request_uses_actual_execution_not_model_count(self):
        def request(*args, **kwargs):
            self.assertEqual(args[-1], [])
            self.assertEqual(kwargs["retries"], 1)
            return self.result(), {"usage": {"total_tokens": 50}, "choices": [{"finish_reason": "stop"}]}
        payload = process_request(self.snapshot(), {}, [], "筛选", None,
                                  ("fake", "https://invalid.test", "fake"), request=request)
        self.assertEqual(payload["result"]["paths"], ["images/largest.png", "images/large.png"])
        self.assertEqual(payload["usage"]["total_tokens"], 50)

    def test_provider_truncation_missing_target_and_cancellation_do_not_execute(self):
        for result, response, cancelled in (
            (self.result(), {"choices": [{"finish_reason": "length"}]}, lambda: False),
            (self.result(group_id="gone"), {}, lambda: False),
            ({"action": "remove", "message": "移除", "group_id": None}, {}, lambda: False),
            (self.result(), {}, lambda: True),
        ):
            with self.assertRaises(ValueError):
                process_request(self.snapshot(), {}, [], "筛选", None, ("fake", "url", "model"),
                                cancelled=cancelled, request=lambda *a, **kw: (result, response))


class SelectionQtTests(Fixtures):
    @classmethod
    def setUpClass(cls):
        import torch  # Windows: load before Qt.
        from PyQt5.QtWidgets import QApplication
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

    def wait(self):
        deadline = time.monotonic() + 12
        while self.window.ai_chat_dock.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(self.window.ai_chat_dock.worker)

    def test_virtual_group_navigation_update_remove_does_not_create_category(self):
        window = self.window
        before = list(window.graphicsView.imageItem.existing_categories)
        group = window.upsert_temporary_selection("大面积脏污", "占比 > 5%", ["images/largest.png", "images/large.png"], AREA_CODE)
        self.assertEqual(window.listWidget.count(), 2)
        self.assertEqual(window.listWidget.currentItem().toolTip(), "images/largest.png")
        self.assertEqual(len(window.graphicsView.imageItem.annotations), 2)
        self.assertIsNotNone(window.find_sample_tree_item(("temporary", group, None)))
        window.upsert_temporary_selection("最大的1张", "前1张", ["images/largest.png"], AREA_CODE, group)
        self.assertEqual(len(window.temporary_selections), 1)
        self.assertEqual(window.listWidget.count(), 1)
        window.persist_project_state()
        state = json.loads((self.root / "sample_tree.json").read_text(encoding="utf-8"))
        self.assertNotIn(group, json.dumps(state))
        self.assertEqual(before, window.graphicsView.imageItem.existing_categories)
        self.assertEqual((self.root / "label.txt").read_text(encoding="utf-8"), "脏污\n破损\n")
        window.remove_temporary_selection(group)
        self.assertEqual(window.listWidget.count(), 5)
        self.assertFalse(window.temporary_selections)
        self.assertTrue((self.root / "images" / "largest.png").exists())
        for name, doc in self.documents.items():
            self.assertEqual(json.loads((self.root / "jsons" / (name + ".json")).read_text(encoding="utf-8")), doc)

    def test_empty_group_and_removed_image_never_falls_back_to_all(self):
        group = self.window.upsert_temporary_selection("空结果", "无匹配", [])
        self.assertEqual(self.window.listWidget.count(), 0)
        self.assertTrue(self.window.graphicsView.imageItem.pixmap().isNull())
        self.window.upsert_temporary_selection("临时", "条件", ["images/large.png"], "", group)
        self.window.relative_paths.remove("images/large.png")
        self.assertEqual(self.window.get_paths_for_current_tree_filter(), [])
        with self.assertRaises(ValueError):
            self.window.upsert_temporary_selection("无效", "", ["outside"])

    def test_full_chat_worker_roundtrip_and_followup(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        dock.api_key.setText("fake-test-key")
        dock.input.setPlainText("把大面积脏污挑出来")
        with patch("Utils.AIAugment.request_qwen_json", side_effect=[
                (self.result(), {"usage": {"total_tokens": 50}}), ({"action": "reply", "message": "筛选完成。"}, {})]):
            dock.submit()
            self.wait()
        self.assertEqual(self.window.listWidget.count(), 2)
        group = next(iter(self.window.temporary_selections))
        followup = self.result("result = groups['" + group + "'][:1]", group)
        followup["criteria"] = "原分组前1张"
        dock.input.setPlainText("只看最大的1张")
        with patch("Utils.AIAugment.request_qwen_json", side_effect=[
                (followup, {}), ({"action": "reply", "message": "筛选完成。"}, {})]):
            dock.submit()
            self.wait()
        self.assertEqual(self.window.listWidget.count(), 1)
        self.assertEqual(len(dock.history), 4)
        self.assertNotIn("fake-test-key", json.dumps(dock.details))
        self.assertIn("1 张", dock.transcript.toPlainText())

    def test_cancelled_and_outdated_results_cannot_apply(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        dock.stopped = True
        dock.apply_result({"result": self.result()})
        self.assertFalse(self.window.temporary_selections)
        dock.stopped = False
        dock.start_revision = 0
        self.window.upsert_temporary_selection("手动", "", [])
        dock.apply_result({"result": self.result()})
        self.assertEqual(len(self.window.temporary_selections), 1)
        self.assertIn("发生变化", dock.transcript.toPlainText())


if __name__ == "__main__":
    unittest.main()
