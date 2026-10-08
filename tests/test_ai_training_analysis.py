"""Contract and UI tests. Requests and training are mocked, never paid calls."""
import copy
import json
import os
import pickle
import shutil
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Utils import AITrainingAnalysis as service


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parent / ".tmp_training_analysis"
        self.root = self.base / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.folder = self.root / "experiment"
        self.folder.mkdir()
        (self.root / "data.yaml").write_text("names: [defect]\n", encoding="utf-8")
        (self.root / "initial.pt").write_bytes(b"test only, not a real model")
        self.args = {"task": "detect", "data": str(self.root / "data.yaml"), "epochs": 3, "imgsz": 640, "batch": 2}
        # JSON is a valid YAML subset.
        (self.folder / "args.yaml").write_text(json.dumps(self.args), encoding="utf-8")
        self.csv_text = ("epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50-95(B),train/box_loss,val/box_loss\n"
                         "1,2,0.5,0.4,0.3,1.1,1.3\n2,4,0.8,0.7,0.6,0.7,1.0\n3,6,0.7,0.65,0.55,0.6,1.1\n")
        (self.folder / "results.csv").write_text(self.csv_text, encoding="utf-8")
        self.api = ("test-secret-training-only", "https://invalid.test", "mock-model")

    def tearDown(self):
        target = self.root.resolve()
        target.relative_to(self.base.resolve())
        shutil.rmtree(target)

    def metadata(self):
        value = {"status": "completed", "project": str(self.root.resolve()),
                 "validation_fingerprint": "same-validation-content", "names": {"0": "defect"},
                 "evaluation_config": {"task": "detect", "imgsz": 640, "conf": .001},
                 "ui_config": {"data": str(self.root / "data.yaml"), "weight": str(self.root / "initial.pt"),
                               "task": "目标检测", "epochs": 3, "imgsz": 640, "batch": 2},
                 "results_sha256": service.file_hash(self.folder / "results.csv")}
        service.write_json(self.folder / service.RECORD, value)
        return value

    @staticmethod
    def response():
        return {"summary": {"text": "测试响应：第2轮主指标最高。", "evidence": ["C.best"]},
                "findings": [{"text": "测试假设，需验证。", "evidence": ["C.trend"]}],
                "comparison": {"text": "没有可比实验。", "evidence": ["comparison"]},
                "next_experiment": {"title": "测试实验", "hypothesis": "测试假设", "action": "一次改动",
                                    "keep_fixed": "同一验证集", "observe": "验证指标", "stop": "无改善则停止",
                                    "cost": "增加计算", "evidence": ["C.best"],
                                    "change": {"parameter": "epochs", "old": 3, "new": 4}}}

    def request(self, key, endpoint, model, prompt, paths, **kwargs):
        self.assertEqual(paths, [])
        self.assertEqual(kwargs["retries"], 1)
        self.assertEqual(kwargs["max_tokens"], 3500)
        self.assertNotIn(key, prompt)
        return self.response(), {"usage": {"total_tokens": 100}}


class TrainingAnalysisTests(Fixtures):
    def chart_setup(self):
        from Utils import TrainingReport
        (self.folder / "data.yaml").write_text("names: [划痕, 凹坑]\n", encoding="utf-8")
        for name in ("BoxPR_curve.png", "confusion_matrix.png"):
            (self.folder / name).write_bytes(b"placeholder: requests in these tests are mocked")
        readings = [
            {"evidence": "C.plot_pr", "readable": True, "names": ["划痕", "凹坑"], "values": [.59, .63]},
            {"evidence": "C.plot_confusion", "readable": True, "names": ["划痕", "凹坑", "background"],
             "row_axis": "predicted", "column_axis": "true", "matrix": [[75, 4, 25], [3, 36, 27], [72, 16, 0]]},
        ]
        current = service.load_run(self.folder)
        return TrainingReport, current, readings

    def test_generic_class_report_counts_follow_matrix_axes_and_use_dataset_naming(self):
        view, current, readings = self.chart_setup()
        validated = view.validate_readings(readings, service.prepare_evidence(current)["facts"])
        rows = view.class_rows(current, validated)
        self.assertEqual([r["name"] for r in rows], ["划痕", "凹坑"])
        self.assertEqual((rows[0]["instances"], rows[0]["correct"], rows[0]["missed"], rows[0]["wrong_class"], rows[0]["background_false"]),
                         (150, 75, 72, 3, 25))
        self.assertEqual((rows[1]["instances"], rows[1]["correct"], rows[1]["missed"]), (56, 36, 16))
        self.assertIn("experiment", current["context"]["display_name"])
        self.assertNotIn("待填写", view.render_html(current, None, None))

    def test_chart_axes_proportions_and_unknown_class_are_rejected(self):
        view, current, readings = self.chart_setup()
        facts = service.prepare_evidence(current)["facts"]
        for mutate in (lambda r: r[1].update(row_axis="true"),
                       lambda r: r[1]["matrix"][0].__setitem__(0, .75),
                       lambda r: r[0]["names"].__setitem__(0, "编造缺陷")):
            modified = copy.deepcopy(readings)
            mutate(modified)
            with self.assertRaises(ValueError):
                view.validate_readings(modified, facts)

    def test_class_evidence_cannot_be_invented_from_aggregate_metrics(self):
        view, current, readings = self.chart_setup()
        response = self.response()
        response["chart_readings"] = readings
        response["class_findings"] = [{"name": "划痕", "text": "关注漏检。", "evidence": ["C.best"]}]
        with self.assertRaises(ValueError):
            service.validate_report(response, service.prepare_evidence(current), self.args)

    def test_only_two_charts_sent_once_and_cached_by_content(self):
        view, current, readings = self.chart_setup()
        calls = []
        def request(key, endpoint, model, prompt, paths, **kwargs):
            calls.append(paths)
            self.assertEqual([Path(p).name for p in paths], ["BoxPR_curve.png", "confusion_matrix.png"])
            response = self.response()
            response["chart_readings"] = readings
            response["class_findings"] = [{"name": "划痕", "text": "先查看漏检样例。", "evidence": ["C.plot_confusion"]}]
            return response, {"usage": {"total_tokens": 125}}
        result = service.analyze(current, None, self.api, request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(calls), 1)
        rendered = view.render_html(current, None, result)
        self.assertIn("检对 75", rendered)
        self.assertNotIn("comparison_status", rendered)
        self.assertIsNotNone(service.latest_report(self.folder, current=current))
        (self.folder / "BoxPR_curve.png").write_bytes(b"changed chart")
        self.assertIsNone(service.latest_report(self.folder, current=service.load_run(self.folder)))

    def test_local_directions_and_possible_early_stop(self):
        from Utils.TrainingReport import training_relations
        current = service.load_run(self.folder)
        current["best"]["epoch"] = 202
        current["last"]["epoch"] = 302
        current["best"]["val/cls_loss"] = 1.20722
        current["last"]["val/cls_loss"] = 1.18765
        current["args"]["patience"] = 100
        facts = training_relations(current)
        self.assertEqual(facts["best_to_last"]["val/cls_loss"]["direction"], "下降")
        self.assertTrue(facts["early_stop_possible"])

    def test_external_paths_identify_batch_without_guessing_part(self):
        from Utils import TrainingReport
        args = {"data": "/server/datasets/outer_surface_v4/data.yaml", "task": "detect"}
        context, charts, hashes = TrainingReport.inputs(self.folder, args, {}, service.file_hash)
        self.assertEqual(context["display_name"], "outer_surface_v4 · experiment")
        context, charts, hashes = TrainingReport.inputs(self.folder, {}, {}, service.file_hash)
        self.assertEqual(context["display_name"], "experiment")

    def test_unreadable_chart_is_not_filled_with_guessed_numbers(self):
        view, current, readings = self.chart_setup()
        readings = [{"evidence": r["evidence"], "readable": False, "reason": "读不清"} for r in readings]
        checked = view.validate_readings(readings, service.prepare_evidence(current)["facts"])
        rows = view.class_rows(current, checked)
        self.assertNotIn("correct", rows[0])
        self.assertIn("暂不能判断", view.class_message(rows[0]))

    def test_definitive_overfitting_claim_is_rejected(self):
        current = service.load_run(self.folder)
        value = self.response()
        value["summary"]["text"] = "表明训练已过拟合"
        with self.assertRaises(ValueError):
            service.validate_report(value, service.prepare_evidence(current), self.args)

    def test_empty_class_support_and_invalid_structured_values_are_not_good_results(self):
        from Utils import TrainingReport
        self.assertIn("暂不能评估", TrainingReport.class_message({"instances": 0, "correct": 0, "background_false": 4}))
        current = service.load_run(self.folder)
        current["record"]["per_class"] = [{"Class": "defect", "mAP50": "bad", "Box-P": float('nan')}]
        current["record"]["confusion_counts"] = {"matrix": [[1]]}
        rows = TrainingReport.class_rows(current)
        self.assertIsNone(rows[0]["ap50"])
        self.assertNotIn("correct", rows[0])

    def test_real_local_calculations_and_unknown_comparison(self):
        current = service.load_run(self.folder)
        self.assertEqual(current["best"]["epoch"], 2)
        self.assertEqual(current["last"]["epoch"], 3)
        self.assertEqual(current["elapsed"], 6)
        other = copy.deepcopy(current)
        other["folder"] += "_another"
        comparison = service.compare_runs(current, other)
        self.assertEqual(comparison["status"], "unknown")
        self.assertEqual(comparison["deltas"], {})
        self.assertTrue(any("按类" in v for v in current["warnings"]))

    def test_missing_values_are_not_zero_and_masks_use_mask_metric(self):
        path = self.folder / "results.csv"
        path.write_text("epoch,metrics/mAP50-95(B),metrics/mAP50-95(M)\n1,0.9,nan\n2,0.5,0.6\n", encoding="utf-8")
        current = service.load_run(self.folder)
        self.assertIsNone(current["rows"][0]["metrics/mAP50-95(M)"])
        self.args["task"] = "segment"
        (self.folder / "args.yaml").write_text(json.dumps(self.args), encoding="utf-8")
        self.assertEqual(service.load_run(self.folder)["best"]["epoch"], 2)

    def test_rejects_duplicate_epochs_and_out_of_range_scores(self):
        path = self.folder / "results.csv"
        for content in ("epoch,metrics/mAP50-95(B)\n1,0.3\n1,0.4\n", "epoch,metrics/mAP50-95(B)\n1,34\n"):
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                service.load_run(self.folder)

    def test_changed_records_do_not_reuse_dataset_identity(self):
        self.metadata()
        with (self.folder / "results.csv").open("a", encoding="utf-8") as stream:
            stream.write("4,8,0.8,0.75,0.65,0.5,1.0\n")
        current = service.load_run(self.folder)
        self.assertEqual(current["record"], {})
        self.assertTrue(any("发生变化" in v for v in current["warnings"]))

    def test_running_experiment_not_analyzed(self):
        value = self.metadata()
        value["status"] = "running"
        service.write_json(self.folder / service.RECORD, value)
        with self.assertRaisesRegex(ValueError, "尚未"):
            service.load_run(self.folder)

    def test_comparison_requires_content_and_evaluation_match(self):
        self.metadata()
        current = service.load_run(self.folder)
        previous = copy.deepcopy(current)
        previous["folder"] += "_previous"
        previous["best"]["metrics/mAP50-95(B)"] = .4
        self.assertEqual(service.compare_runs(current, previous)["deltas"]["metrics/mAP50-95(B)"], .2)
        previous["record"]["evaluation_config"]["imgsz"] = 960
        self.assertEqual(service.compare_runs(current, previous)["status"], "different")
        previous["record"]["evaluation_config"]["imgsz"] = 640
        previous["record"]["validation_fingerprint"] = "changed"
        self.assertEqual(service.compare_runs(current, previous)["deltas"], {})

    def test_single_request_persistence_cache_and_no_source_changes(self):
        current = service.load_run(self.folder)
        result = service.analyze(current, None, self.api, self.request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["request_count"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 100)
        self.assertEqual(service.load_run(self.folder)["hashes"], current["hashes"])
        cached = service.latest_report(self.folder, current=current)
        self.assertEqual(cached["path"], result["path"])
        self.assertNotIn(self.api[0], Path(result["path"]).read_text(encoding="utf-8"))

    def test_failure_is_not_a_report_and_redacts_key(self):
        def failure(*args, **kwargs):
            raise RuntimeError("denied " + self.api[0])
        current = service.load_run(self.folder)
        result = service.analyze(current, None, self.api, failure)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("result", result)
        self.assertIsNone(service.latest_report(self.folder, current=current))
        self.assertNotIn(self.api[0], Path(result["path"]).read_text(encoding="utf-8"))

    def test_cancel_before_request_does_not_call_provider(self):
        def request(*args, **kwargs):
            self.fail("cancelled work must not request")
        result = service.analyze(service.load_run(self.folder), None, self.api, request, cancelled=lambda: True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["request_count"], 0)

    def test_rejects_unknown_evidence_and_unsafe_changes(self):
        current = service.load_run(self.folder)
        evidence = service.prepare_evidence(current)
        value = self.response()
        value["summary"]["evidence"] = ["C.imaginary"]
        with self.assertRaises(ValueError):
            service.validate_report(value, evidence, self.args)
        for change in ({"parameter": "lr0", "old": .01, "new": .1},
                       {"parameter": "epochs", "old": 300, "new": 400},
                       {"parameter": "imgsz", "old": 640, "new": 999},
                       {"parameter": "batch", "old": 2, "new": -1}):
            value = self.response()
            value["next_experiment"]["change"] = change
            with self.assertRaises(ValueError):
                service.validate_report(value, evidence, self.args)

    def test_truncated_or_refused_provider_output_is_not_success(self):
        for raw in ({"choices": [{"finish_reason": "length"}]},
                    {"choices": [{"message": {"refusal": "declined"}}]}):
            result = service.analyze(service.load_run(self.folder), None, self.api,
                                     lambda *args, **kwargs: (self.response(), raw))
            self.assertEqual(result["status"], "failed")

    def test_restore_baseline_only_for_same_project_and_unchanged_source(self):
        self.metadata()
        current = service.load_run(self.folder)
        result = service.analyze(current, None, self.api, self.request)
        config = service.application_config(current, result, self.root)
        self.assertEqual((config["epochs"], config["imgsz"], config["batch"]), (4, 640, 2))
        with self.assertRaises(ValueError):
            service.application_config(current, result, self.root / "other")
        current["hashes"]["results.csv"] = "changed"
        with self.assertRaises(ValueError):
            service.application_config(current, result, self.root)

    def test_discovery_does_not_pick_another_project(self):
        other = self.root / "other_project"
        other.mkdir()
        self.assertIn(self.folder.resolve(), service.discover_runs(self.root, self.root))
        self.assertEqual(service.discover_runs(other, self.root), [])

    def test_training_recorder_saves_actual_validation_and_detects_changed_inputs(self):
        from Utils.TrainingRunRecord import TrainingRunRecorder
        self.metadata()
        image, label = self.root / "image.png", self.root / "image.txt"
        image.write_bytes(b"test image bytes")
        label.write_text("0 .5 .5 .2 .2", encoding="utf-8")
        loader = SimpleNamespace(dataset=SimpleNamespace(im_files=[str(image)], label_files=[str(label)], labels=[]))
        validator = SimpleNamespace(args=SimpleNamespace(task="detect", imgsz=640, plots=True),
                                    metrics=SimpleNamespace(summary=lambda: [{"Class": "defect", "mAP50-95": .6}]),
                                    confusion_matrix=SimpleNamespace(matrix=[[1., 2.], [3., 0.]]))
        trainer = SimpleNamespace(save_dir=self.folder, data={"names": {0: "defect"}}, train_loader=loader,
                                  test_loader=loader, validator=validator, metrics={"metrics/mAP50-95(B)": .6})
        recorder = TrainingRunRecorder(self.root, {})
        recorder.start(trainer)
        self.assertEqual(service.read_json(self.folder / service.RECORD)["status"], "running")
        recorder.finish(trainer)
        recorded = service.load_run(self.folder)["record"]
        self.assertEqual(recorded["per_class"][0]["Class"], "defect")
        self.assertTrue(recorded["validation_fingerprint"])
        self.assertEqual(recorded["confusion_counts"]["matrix"], [[1, 2], [3, 0]])
        label.write_text("0 .5 .5 .3 .3\n", encoding="utf-8")
        recorder.finish(trainer)
        self.assertNotIn("validation_fingerprint", service.read_json(self.folder / service.RECORD))


class TrainingAnalysisQtTests(Fixtures):
    @classmethod
    def setUpClass(cls):
        import torch  # Windows DLL loading must precede Qt.
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from PyQt5.QtWidgets import QWidget
        from Utils.AITrainingAnalysisDialog import AITrainingAnalysisDialog
        self.owner = QWidget()
        self.owner.project_dir = str(self.root)
        self.owner.latest_training_run = str(self.folder)
        self.applied = []
        self.owner.apply_training_analysis_config = self.applied.append
        self.metadata()
        self.dialog = AITrainingAnalysisDialog(self.owner)
        self.wait_worker()

    def wait_worker(self):
        deadline = time.monotonic() + 10
        while self.dialog.running() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertFalse(self.dialog.running())

    def tearDown(self):
        if self.dialog.running():
            self.dialog.stop()
            self.dialog.worker.wait(10000)
        self.dialog.hide()
        self.owner.deleteLater()
        self.app.processEvents()
        super().tearDown()

    def test_button_request_saved_reopen_and_confirm_apply_without_training(self):
        from PyQt5.QtWidgets import QMessageBox
        self.assertIsNotNone(self.dialog.current)
        self.dialog.api_key.setText(self.api[0])
        with patch("Utils.AITrainingAnalysisDialog.request_qwen_json", side_effect=self.request) as request:
            self.dialog.analyze_button.click()
            self.wait_worker()
            self.assertEqual(request.call_count, 1)
            self.assertTrue(self.dialog.apply_button.isEnabled())
            self.dialog.refresh.click()
            self.wait_worker()
            self.assertEqual(request.call_count, 1)
            self.assertIn("没有再次调用", self.dialog.status.text())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.dialog.apply_button.click()
        self.assertEqual(self.applied, [])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.dialog.apply_button.click()
        self.assertEqual(self.applied[0]["epochs"], 4)
        self.assertEqual((self.folder / "results.csv").read_text(encoding="utf-8"), self.csv_text)

    def test_training_page_has_entry_and_config_application_does_not_start_training(self):
        from Utils.test import TestWindow
        (self.root / "images").mkdir()
        (self.root / "jsons").mkdir()
        (self.root / "label.txt").write_text("defect\n", encoding="utf-8")
        (self.root / "datafile.dat").write_bytes(pickle.dumps([]))
        (self.root / "flagfile.dat").write_bytes(pickle.dumps([]))
        service.write_json(self.root / "config.json", {})
        window = TestWindow(str(self.root))
        try:
            self.assertEqual(window.ai_training_button.text(), "AI 分析训练结果")
            config = service.read_json(self.folder / service.RECORD)["ui_config"]
            config["epochs"] = 4
            window.apply_training_analysis_config(config)
            self.assertEqual(window.epochEdit.text(), "4")
            self.assertIsNone(window.train_thread)
            self.assertEqual(window.comboBox_2.currentText(), config["weight"])
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
