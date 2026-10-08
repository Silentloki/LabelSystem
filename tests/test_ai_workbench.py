"""Local filesystem, geometry, transaction, tool composition and Qt contracts.

Provider responses / predictor in unit tests are simulated. Files and UI are
real disposable fixtures. Production data and API credentials are never used.
"""
import copy
import json
import os
import pickle
import shutil
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch  # Windows DLL initialization before Qt.
from PIL import Image

from test_ai_chat_selection import Fixtures, AREA_CODE, rect
from Utils.AIWorkspace import Workspace, json_bytes, read_json, write_json, document_token
from Utils.AIWorkTools import WorkTools, TOOL_SPECS, clipped_annotations
from Utils.AIWorkAgent import run_agent
from Utils.AICandidates import generate_candidates, prepare_adoption


class WorkFixtures(Fixtures):
    def setUp(self):
        super().setUp()
        self.context = {"current_paths": self.paths, "selected_paths": [self.paths[0]], "current_image": self.paths[0],
                        "classes": ["脏污", "破损"], "flags": dict(zip(self.paths, [1, 1, 1, 1, 0])),
                        "resources": {}, "max_calls": 4}
        self.tools = WorkTools(self.root, self.paths, context=self.context)
        self.store = self.tools.store

    def call(self, tool, **args):
        return self.tools.execute(tool, args)

    def model_resource(self):
        file = self.root / "mock_model.pt"
        file.write_bytes(b"unit-test predictor placeholder, not a loadable model")
        self.context["resources"]["m1"] = {"path": str(file), "kind": "model", "name": file.name}
        return file

    def fake_predictor(self):
        return SimpleNamespace(names={0: "脏污"}, task="detect", predict=lambda **kwargs: [SimpleNamespace(
            boxes=SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([.9]),
                                  xyxyn=torch.tensor([[0., 0., .4, .4]])), masks=None)])

    def candidates(self):
        self.model_resource()
        plan = self.call("preannotate", model="m1")["pending"]
        return generate_candidates(self.root, plan["artifact_id"], predictor=self.fake_predictor())


class TransactionTests(WorkFixtures):
    def test_preview_commit_and_undo_roundtrip_exact_source_bytes(self):
        originals = self.hashes()
        old_labels = (self.root / "label.txt").read_bytes()
        preview = self.call("relabel", old="脏污", new="污点", scope="selected")
        self.assertEqual(preview["images"], 1)
        self.assertEqual(preview["annotations"], 2)
        self.assertEqual(originals, self.hashes())
        tid = preview["pending"]["transaction_id"]
        result = self.store.commit(tid)
        self.assertEqual(result["status"], "committed")
        doc = read_json(self.root / "jsons/small.json")
        self.assertEqual([a["lable"] for a in doc["annotations"]], ["污点", "污点"])
        undo = self.store.prepare_undo(tid)
        self.store.commit(undo["transaction_id"])
        self.assertEqual(originals, self.hashes())
        self.assertEqual(old_labels, (self.root / "label.txt").read_bytes())
        with self.assertRaises(ValueError):
            self.store.commit(tid)

    def test_concurrent_changes_block_commit_and_later_changes_block_undo(self):
        pending = self.call("relabel", old="脏污", new="污点", scope="selected")["pending"]
        path = self.root / "jsons/small.json"
        doc = read_json(path)
        doc["annotations"][0]["lable"] = "人工修改"
        write_json(path, doc)
        with self.assertRaises(ValueError):
            self.store.commit(pending["transaction_id"])
        self.assertEqual(read_json(path), doc)
        pending = self.call("relabel", old="脏污", new="污点", scope="selected")["pending"]
        self.store.commit(pending["transaction_id"])
        newdoc = read_json(path)
        newdoc["annotations"][0]["lable"] = "后续修改"
        write_json(path, newdoc)
        with self.assertRaises(ValueError):
            self.store.prepare_undo(pending["transaction_id"])
        self.assertEqual(read_json(path), newdoc)

    def test_ui_float_reserialization_does_not_prevent_recovery(self):
        pending = self.call("relabel", old="脏污", new="污点", scope="selected")["pending"]
        self.store.commit(pending["transaction_id"])
        path = self.root / "jsons/small.json"
        doc = read_json(path)
        for ann in doc["annotations"]:
            for point in ann["points"]:
                point["x"] = float(point["x"])
                point["y"] = float(point["y"])
        write_json(path, doc)
        undo = self.store.prepare_undo(pending["transaction_id"])
        self.store.commit(undo["transaction_id"])
        self.assertEqual(read_json(path), self.documents["small"])

    def test_failed_second_write_rolls_back_first_write(self):
        self.context["current_paths"] = self.paths[:3]
        before = self.hashes()
        pending = self.call("relabel", old="脏污", new="污点")["pending"]
        import Utils.AIWorkspace as workspace
        actual = workspace.atomic_bytes
        failed = []

        def write(path, data):
            if Path(path).name == "large.json" and not failed:
                failed.append(True)
                raise OSError("test write failure")
            return actual(path, data)

        with patch("Utils.AIWorkspace.atomic_bytes", side_effect=write), self.assertRaises(OSError):
            self.store.commit(pending["transaction_id"])
        self.assertEqual(before, self.hashes())
        record = read_json(self.store.location("transactions", pending["transaction_id"]) / "transaction.json")
        self.assertEqual(record["status"], "rolled_back")

    def test_interrupted_transaction_can_prepare_recovery_without_overwriting_new_edits(self):
        import base64
        self.context["current_paths"] = self.paths[:3]
        before = self.hashes()
        pending = self.call("relabel", old="脏污", new="污点")["pending"]
        record_path = self.store.location("transactions", pending["transaction_id"]) / "transaction.json"
        record = read_json(record_path)
        record["status"] = "applying"
        first = record["entries"][0]
        self.store.source(first["path"]).write_bytes(base64.b64decode(first["after"]))
        write_json(record_path, record)
        undo = self.store.prepare_undo(pending["transaction_id"])
        self.store.commit(undo["transaction_id"])
        self.assertEqual(before, self.hashes())

    def test_path_escape_and_arbitrary_source_write_rejected(self):
        for path in ("../elsewhere.json", "C:/other.json"):
            with self.assertRaises(ValueError):
                self.store.source(path)
        with self.assertRaises(ValueError):
            self.store.prepare_transaction("invalid", {"config.json": b"{}"})
        with self.assertRaises(ValueError):
            self.store.location("artifacts", "../../bad")


class DataToolTests(WorkFixtures):
    def test_registry_all_tools_have_implementation_and_validate_arguments(self):
        for name in TOOL_SPECS:
            self.assertTrue(callable(getattr(self.tools, "tool_" + name)))
        with self.assertRaises(ValueError):
            self.call("delete_files", scope="all")
        with self.assertRaises(ValueError):
            self.call("export", output="C:/outside")

    def test_scope_default_is_current_and_selection_can_feed_export(self):
        self.context["current_paths"] = self.paths[1:3]
        result = self.call("select", code=AREA_CODE, title="大面积", criteria="占比>5%")
        self.assertEqual(result["count"], 2)
        output = self.call("export", scope="last", format="native")
        self.assertEqual(output["count"], 2)
        self.assertEqual(len(list((Path(output["export_dir"]) / "images").iterdir())), 2)
        self.assertEqual(read_json(Path(output["output"]) / "sources.json")[0]["source"], "images/largest.png")

    def test_yolo_split_has_no_byte_duplicate_leakage_and_is_importable(self):
        self.context["current_paths"] = self.paths[:4]
        Image.new("RGB", (100, 200), (10, 20, 30)).save(self.root / self.paths[0])
        Image.new("RGB", (100, 200), (90, 80, 70)).save(self.root / self.paths[1])
        before = self.hashes()
        result = self.call("export", format="yolo_detect", train_ratio=.5)
        from Utils.AnnotationImporter import inspect_yolo_dataset
        inspection = inspect_yolo_dataset(result['dataset_dir'])
        self.assertEqual(inspection["summary"]["ready_records"], 4)
        self.assertFalse(inspection["errors"])
        rows = read_json(Path(result["output"]) / "sources.json")
        hashes = {}
        for row in rows:
            hashes.setdefault(row["image_hash"], set()).add(row["split"])
        self.assertTrue(all(len(v) == 1 for v in hashes.values()))
        self.assertEqual(before, self.hashes())

    def test_missing_labels_not_exported_as_negative_and_failed_output_not_completed(self):
        with self.assertRaises(ValueError):
            self.call("export", scope="all", format="native")
        self.context["current_paths"] = self.paths[:3]
        with self.assertRaises(ValueError):
            self.call("export", format="yolo_segment")
        entries = self.store.list_entries()
        self.assertTrue(entries)
        self.assertEqual(entries[0]["status"], "failed")

    def test_crop_preserves_other_labels_and_clips_polygon(self):
        self.context["current_paths"] = [self.paths[2]]
        doc = read_json(self.root / "jsons/largest.json")
        doc["annotations"].append({"type": "polygon", "lable": "破损", "points": [
            {"x": .5, "y": .2}, {"x": .9, "y": .2}, {"x": .9, "y": .8}, {"x": .5, "y": .8}]})
        write_json(self.root / "jsons/largest.json", doc)
        before = self.hashes()
        result = self.call("crop", label="脏污", padding=0)
        self.assertEqual(result["count"], 1)
        cropped = read_json(Path(result["output"]) / "jsons/crop_000001.json")
        self.assertEqual((cropped["image_width"], cropped["image_height"]), (70, 140))
        self.assertEqual(len(cropped["annotations"]), 3)
        for ann in cropped["annotations"]:
            for point in ann["points"]:
                self.assertTrue(0 <= point["x"] <= 1 and 0 <= point["y"] <= 1)
        self.assertEqual(before, self.hashes())

    def test_tile_count_and_concave_polygon_disconnected_clip(self):
        self.context["current_paths"] = [self.paths[2]]
        result = self.call("crop", mode="tiles", tile_size=100, overlap=0)
        self.assertEqual(result["count"], 2)
        # Inverted U: clipping away the top connector leaves two components.
        points = [(0.1, .1), (.9, .1), (.9, .9), (.7, .9), (.7, .3), (.3, .3), (.3, .9), (.1, .9)]
        doc = {"image_width": 100, "image_height": 100, "annotations": [{"type": "polygon", "lable": "脏污",
                 "points": [{"x": x, "y": y} for x, y in points]}]}
        self.assertEqual(len(clipped_annotations(doc, (0, 50, 100, 100))), 2)

    def test_native_conversion_skips_ambiguous_stems_and_preserves_sources(self):
        self.context["resources"]["r1"] = {"path": str(self.root), "kind": "directory", "name": "source"}
        (self.root / "images/sub").mkdir()
        shutil.copyfile(self.root / self.paths[0], self.root / "images/sub/small.png")
        result = self.call("convert_dataset", resource="r1", format="native")
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["skipped"], 3)  # two ambiguous small images + missing JSON

    def test_duplicates_and_similarity_are_explicit_candidates(self):
        result = self.call("duplicates")
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["duplicate_sets"], 1)
        similar = self.call("similar", top_n=2)
        self.assertEqual(similar["count"], 2)
        self.assertIn("不是", similar["criteria"])

    def test_versions_include_unlabeled_images_and_find_real_changes(self):
        before = self.call("snapshot", scope="all", title="修改前")
        self.assertEqual(before["count"], 5)
        snapshot = read_json(Path(before["output"]) / "snapshot.json")
        self.assertIsNone(snapshot[self.paths[4]]["annotation_hash"])
        doc = read_json(self.root / "jsons/large.json")
        doc["annotations"][0]["lable"] = "污点"
        write_json(self.root / "jsons/large.json", doc)
        result = self.call("compare", before=before["artifact_id"])
        self.assertEqual(result["annotation_changed"], 1)
        self.assertEqual(result["image_changed"], 0)
        self.assertEqual(self.tools.last_scope, [self.paths[1]])

    def test_subset_snapshot_does_not_report_other_project_images_as_new(self):
        before = self.call("snapshot", scope="selected")
        result = self.call("compare", before=before["artifact_id"])
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["removed"], 0)

    def test_training_variants_and_reusable_workflow_do_not_start_training(self):
        self.context["training_config"] = {"data": "dataset/data.yaml", "weight": "local.pt", "task": "目标检测", "epochs": 100, "batch": 8, "imgsz": 640}
        plan = self.call("training_plan", parameter="imgsz", values=[640, 960])
        self.assertEqual([c["imgsz"] for c in plan["configs"]], [640, 960])
        self.tools.steps.clear()
        self.call("select", code=AREA_CODE, title="大面积", criteria="占比>5%")
        self.call("stats", scope="last")
        workflow = self.call("save_workflow", title="大面积样本统计")
        rerun = self.call("run_workflow", workflow_id=workflow["workflow_id"])
        self.assertEqual(rerun["completed_steps"], 2)
        self.assertEqual(self.tools.last_scope, [self.paths[2], self.paths[1]])


class CandidateTests(WorkFixtures):
    def test_same_named_model_results_keep_their_recorded_versions(self):
        from Utils.AIModelIdentity import model_label, recorded_model
        manifests = []
        before = self.hashes()
        for index, experiment in enumerate(('exp17', 'exp18', 'exp18')):
            path = self.root / experiment / 'weights/best.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'fixture model version {index}'.encode())
            self.context['resources']['m1'] = {'path': str(path), 'kind': 'model', 'name': 'best.pt'}
            pending = self.call('preannotate', model='m1')['pending']
            result = generate_candidates(self.root, pending['artifact_id'], predictor=self.fake_predictor())
            _, manifest = self.store.artifact(result['artifact_id'])
            self.assertEqual(pending['model'], model_label(manifest['metadata']))
            self.assertIn(experiment + '/best.pt', manifest['title'])
            manifests.append(manifest)
        self.assertEqual(len({m['title'] for m in manifests}), 3)
        # Simulate older saved results: identity comes from their saved plan,
        # even after weights have changed or disappeared.
        path.unlink()
        old = copy.deepcopy(manifests[1])
        del old['metadata']['model_path']
        del old['metadata']['model_resource']
        recovered = recorded_model(self.store, old)
        self.assertEqual(recovered, manifests[1]['metadata'])
        self.assertNotEqual(recovered['model_sha256'], manifests[2]['metadata']['model_sha256'])
        self.assertNotIn('model_path', old['metadata'])
        _, plan = self.store.artifact(old['metadata']['plan_id'])
        plan['metadata']['model_sha256'] = 'wrong-version'
        write_json(self.store.location('artifacts', plan['id']) / 'manifest.json', plan)
        self.assertNotIn('model_path', recorded_model(self.store, old))
        self.assertEqual(before, self.hashes())

    def test_prediction_candidates_protect_existing_and_adoption_is_recoverable(self):
        before = self.hashes()
        flag_bytes = (self.root / "flagfile.dat").read_bytes()
        result = self.candidates()
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["adoptable"], 1)
        self.assertEqual(result["protected"], 4)
        self.assertEqual(before, self.hashes())
        with self.assertRaises(ValueError):
            prepare_adoption(self.root, result["artifact_id"], [self.paths[0]])
        pending = prepare_adoption(self.root, result["artifact_id"], [self.paths[4]])
        self.store.commit(pending["transaction_id"])
        self.assertEqual(pickle.loads((self.root / "flagfile.dat").read_bytes()), [1, 1, 1, 1, 1])
        self.assertEqual(read_json(self.root / "jsons/missing.json")["annotations"][0]["lable"], "脏污")
        undo = self.store.prepare_undo(pending["transaction_id"])
        self.store.commit(undo["transaction_id"])
        self.assertEqual(before, self.hashes())
        self.assertEqual(flag_bytes, (self.root / "flagfile.dat").read_bytes())

    def test_changed_image_order_blocks_adoption_and_undo(self):
        result = self.candidates()
        pending = prepare_adoption(self.root, result["artifact_id"], [self.paths[4]])
        path_file = self.root / "datafile.dat"
        original = path_file.read_bytes()
        path_file.write_bytes(pickle.dumps(list(reversed(self.paths))))
        with self.assertRaisesRegex(ValueError, "样本列表"):
            self.store.commit(pending["transaction_id"])
        self.assertFalse((self.root / "jsons/missing.json").exists())
        path_file.write_bytes(original)
        self.store.commit(pending["transaction_id"])
        undo = self.store.prepare_undo(pending["transaction_id"])
        path_file.write_bytes(pickle.dumps(list(reversed(self.paths))))
        with self.assertRaisesRegex(ValueError, "样本列表"):
            self.store.prepare_undo(pending["transaction_id"])
        with self.assertRaisesRegex(ValueError, "样本列表"):
            self.store.commit(undo["transaction_id"])
        self.assertTrue((self.root / "jsons/missing.json").exists())

    def test_candidates_cannot_overwrite_later_human_work(self):
        result = self.candidates()
        write_json(self.root / "jsons/missing.json", self.documents["large"])
        with self.assertRaises(ValueError):
            prepare_adoption(self.root, result["artifact_id"], [self.paths[4]])

    def test_evaluation_uses_existing_ground_truth_and_skips_unlabeled(self):
        result = self.candidates()
        evaluation = self.call("evaluate", candidate_id=result["artifact_id"])
        self.assertEqual(evaluation["images"], 4)
        self.assertEqual(evaluation["skipped"], 1)
        self.assertEqual(evaluation["missed"], 5)
        self.assertEqual(evaluation["extra_predictions"], 3)

    def test_changed_model_or_unrelated_names_rejected(self):
        file = self.model_resource()
        plan = self.call("preannotate", model="m1")["pending"]
        file.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            generate_candidates(self.root, plan["artifact_id"], predictor=self.fake_predictor())
        plan = self.call("preannotate", model="m1")["pending"]
        predictor = self.fake_predictor()
        predictor.names = {0: "person"}
        with self.assertRaisesRegex(ValueError, '类别不匹配'):
            generate_candidates(self.root, plan["artifact_id"], predictor=predictor)

    def test_project_class_filter_preserved_with_model_size_and_explicit_mapping(self):
        self.model_resource()
        predictor = self.fake_predictor()
        predictor.names = {0: '破损', 1: '条形脏污'}
        predictor.overrides = {'imgsz': 1280}
        response = SimpleNamespace(boxes=SimpleNamespace(cls=torch.tensor([0, 1]), conf=torch.tensor([.9, .8]),
            xyxyn=torch.tensor([[0., 0., .4, .4], [.5, .5, .8, .8]])), masks=None)
        with patch.object(predictor, 'predict', return_value=[response]) as predict:
            plan = self.call('preannotate', model='m1')['pending']
            self.assertEqual(plan['imgsz'], '跟随模型训练尺寸')
            result = generate_candidates(self.root, plan['artifact_id'], predictor=predictor)
            self.assertTrue(all(call.kwargs['imgsz'] == 1280 for call in predict.call_args_list))
            rows = read_json(Path(result['output']) / 'candidates.json')
            self.assertEqual(result['predicted_annotations'], 5)
            self.assertTrue(all(r['ignored_classes'] == 1 for r in rows))
            self.assertTrue(all([a['lable'] for a in r['document']['annotations']] == ['破损'] for r in rows))
            self.assertTrue(rows[-1]['can_adopt'])
            prepare_adoption(self.root, result['artifact_id'], [self.paths[4]])
            self.call('evaluate', candidate_id=result['artifact_id'])
            predict.reset_mock()
            plan = self.call('preannotate', model='m1', imgsz=960, class_map={'条形脏污': '脏污'})['pending']
            result = generate_candidates(self.root, plan['artifact_id'], predictor=predictor)
            self.assertTrue(all(call.kwargs['imgsz'] == 960 for call in predict.call_args_list))
            rows = read_json(Path(result['output']) / 'candidates.json')
            self.assertEqual([a['lable'] for a in rows[-1]['document']['annotations']], ['破损', '脏污'])
            self.assertTrue(rows[-1]['can_adopt'])
            prepare_adoption(self.root, result['artifact_id'], [self.paths[4]])
            (self.root / 'label.txt').write_text('破损\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '项目类别变化'):
                prepare_adoption(self.root, result['artifact_id'], [self.paths[4]])


class AgentTests(WorkFixtures):
    def test_direct_crop_action_is_normalized_and_only_processes_current_group(self):
        self.context["current_paths"] = self.paths[:2]
        for action in ({"action": "crop", "args": {"mode": "defect", "label": "脏污"}},
                       {"action": "crop", "mode": "tiles", "tile_size": 1280}):
            with self.subTest(action=action):
                before = self.hashes()
                payload, calls = self.run_responses([action, {"action": "reply", "message": "已裁剪。"}])
                self.assertEqual(payload["work"]["status"], "completed")
                step = payload["work"]["steps"][0]
                self.assertEqual(step["tool"], "crop")
                self.assertEqual(step["result"]["source_count"], 2)
                folder = Path(step["result"]["output"])
                rows = read_json(folder / "sources.json")
                self.assertEqual({r["source"] for r in rows}, set(self.paths[:2]))
                if action.get("mode") == "tiles":
                    for row in rows:
                        with Image.open(folder / row["image"]) as image:
                            self.assertEqual(image.size, (1280, 1280))
                            self.assertEqual(image.getpixel((1279, 1279)), (0, 0, 0))
                        document = read_json(folder / row["json"])
                        ann = document["annotations"][0]
                        original = self.documents[Path(row["source"]).stem]["annotations"][0]
                        self.assertAlmostEqual(ann["points"][1]["x"] * 1280, original["points"][1]["x"] * 100)
                        self.assertAlmostEqual(ann["points"][1]["y"] * 1280, original["points"][1]["y"] * 200)
                self.assertEqual(before, self.hashes())

    def test_legacy_selection_is_intermediate_then_crop_executes(self):
        payload, calls = self.run_responses([
            {"action": "select", "title": "大面积", "criteria": ">5%", "code": AREA_CODE, "message": "先筛选再切图"},
            {"action": "tool", "tool": "crop", "args": {"scope": "last", "mode": "tiles", "tile_size": 1280}},
            {"action": "reply", "message": "已完成切图。"},
        ])
        self.assertEqual(len(calls), 3)
        self.assertEqual([s["tool"] for s in payload["work"]["steps"]], ["select", "crop"])
        self.assertEqual(payload["work"]["steps"][1]["result"]["source_count"], 2)

    def test_unknown_protocol_is_recorded_and_feedback_allows_correction(self):
        self.context["current_paths"] = self.paths[:1]
        bad = {"action": "tile_images", "message": "test-secret"}
        payload, calls = self.run_responses([bad,
            {"action": "tool", "tool": "crop", "args": {"mode": "tiles", "tile_size": 1280}},
            {"action": "reply", "message": "完成。"}])
        self.assertIn("不支持的 action", calls[1])
        self.assertEqual(payload["work"]["status"], "completed")
        record = read_json(self.store.location("runs", payload["run_id"]) / "run.json")
        self.assertEqual(record["model_responses"][0]["parsed"]["action"], "tile_images")
        self.assertNotIn("test-secret", json.dumps(record))

    def test_invalid_protocol_stops_after_two_failures_without_executing(self):
        payload, calls = self.run_responses([{"action": "unknown"}, {"action": ["crop"]}])
        self.assertEqual(len(calls), 2)
        self.assertEqual(payload["work"]["status"], "failed")
        self.assertEqual(payload["work"]["steps"], [])

    def test_malformed_json_is_repaired_with_usage_and_response_preserved(self):
        from Utils.AIAugment import AIJsonParseError
        exc = AIJsonParseError("bad", response_json={"usage": {"total_tokens": 8},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"action":test-secret'}}]})
        with patch("Utils.AIAugment.request_qwen_json", side_effect=[exc,
                ({"action": "tool", "tool": "stats", "args": {"scope": "selected"}}, {"usage": {"total_tokens": 9}}),
                ({"action": "reply", "message": "已统计。"}, {})]) as request:
            payload = run_agent(self.root, self.paths, {}, {}, [], "统计选中图片", ("all", None, None),
                                ("test-secret", "https://invalid.test", "mock"), self.context)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(payload["usage"]["total_tokens"], 17)
        self.assertEqual(payload["work"]["status"], "completed")
        record = read_json(self.store.location("runs", payload["run_id"]) / "run.json")
        self.assertIn('[API Key 已隐藏]', record["model_responses"][0]["content"])

    def run_responses(self, results, **extra):
        calls = []
        def request(*args, **kwargs):
            self.assertEqual(kwargs["retries"], 1)
            self.assertEqual(args[-1], [])
            calls.append(args[3])
            return results[len(calls) - 1], {"usage": {"total_tokens": 10}}
        payload = run_agent(self.root, self.paths, {}, {}, [], "完成工作", ("all", None, None),
                            ("test-secret", "https://invalid.test", "mock"), self.context, request=request, **extra)
        return payload, calls

    def test_agent_combines_real_select_export_and_observes_result(self):
        before = self.hashes()
        payload, calls = self.run_responses([
            {"action": "tool", "tool": "select", "args": {"code": AREA_CODE, "title": "大面积", "criteria": "占比>5%"}},
            {"action": "tool", "tool": "export", "args": {"scope": "last"}},
            {"action": "reply", "message": "已筛选并导出。"},
        ])
        self.assertEqual(len(calls), 3)
        self.assertEqual(payload["usage"]["total_tokens"], 30)
        self.assertEqual(payload["work"]["status"], "completed")
        exported = payload["work"]["steps"][1]["result"]
        self.assertEqual(exported["count"], 2)
        self.assertIn(exported["artifact_id"], calls[2])
        self.assertNotIn("test-secret", (self.store.location("runs", payload["run_id"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(before, self.hashes())

    def test_pending_stops_loop_and_never_commits_original_files(self):
        self.context["current_paths"] = self.paths[:3]
        before = self.hashes()
        payload, calls = self.run_responses([{"action": "tool", "tool": "relabel", "args": {"old": "脏污", "new": "污点"}}])
        self.assertEqual(len(calls), 1)
        self.assertEqual(payload["work"]["status"], "awaiting_confirmation")
        self.assertTrue(payload["work"]["pending"]["transaction_id"])
        self.assertEqual(before, self.hashes())

    def test_error_feedback_supports_correction_and_budget_limit_is_honest(self):
        payload, calls = self.run_responses([
            {"action": "tool", "tool": "missing_tool", "args": {}},
            {"action": "tool", "tool": "stats", "args": {"scope": "selected"}},
            {"action": "reply", "message": "统计完成。"},
        ])
        self.assertIn("不支持", calls[1])
        self.assertEqual(payload["work"]["steps"][0]["result"]["summary"]["total_images"], 1)
        self.context["max_calls"] = 1
        payload, calls = self.run_responses([{"action": "tool", "tool": "stats", "args": {}}])
        self.assertEqual(payload["work"]["status"], "limit_reached")
        self.assertIn("上限", payload["result"]["message"])

    def test_repeated_output_tool_is_not_executed_twice(self):
        self.context["current_paths"] = self.paths[:3]
        call = {"action": "tool", "tool": "export", "args": {}}
        payload, calls = self.run_responses([call, call, {"action": "reply", "message": "完成。"}])
        self.assertEqual(len(payload["work"]["steps"]), 1)
        self.assertEqual(len(self.store.list_entries()), 1)
        self.assertTrue(payload["work"]["observations"][1]["result"]["reused_same_turn"])

    def test_failed_tool_then_false_success_reply_does_not_mark_completed(self):
        payload, _ = self.run_responses([
            {"action": "tool", "tool": "export", "args": {"output": "C:/outside"}},
            {"action": "reply", "message": "导出成功"},
        ])
        self.assertEqual(payload["work"]["status"], "failed")
        self.assertNotIn("导出成功", payload["result"]["message"])

    def test_stop_after_completed_output_preserves_artifact_and_stops_next_call(self):
        self.context["current_paths"] = self.paths[:3]
        state = {"stopped": False}
        original = WorkTools.execute

        def execute(instance, name, args):
            result = original(instance, name, args)
            state["stopped"] = True
            return result

        with patch.object(WorkTools, "execute", execute):
            payload, calls = self.run_responses([{"action": "tool", "tool": "export", "args": {}}], cancelled=lambda: state["stopped"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(payload["work"]["status"], "stopped")
        self.assertEqual(self.store.list_entries()[0]["status"], "completed")


class WorkQtTests(Fixtures):
    @classmethod
    def setUpClass(cls):
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
        deadline = time.monotonic() + 15
        while (self.window.ai_chat_dock.worker is not None or self.window.ai_chat_dock.resume_queued) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(self.window.ai_chat_dock.worker)
        self.assertIsNone(self.window.ai_chat_dock.resume_queued)

    def test_pending_commit_reload_undo_and_unsaved_canvas_guard(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        context = dock.work_context()
        context["current_paths"] = self.paths[:3]
        self.window.sample_groups["脏污"] = ["小面积"]
        self.window.sample_assignments[self.paths[0]] = {"脏污": "小面积"}
        self.window.persist_project_state()
        tools = WorkTools(self.root, self.paths, context=context)
        pending = tools.execute("relabel", {"old": "脏污", "new": "污点"})["pending"]
        from Utils.AIWorkDialogs import ensure_canvas_matches
        canvas = self.window.graphicsView.imageItem
        canvas.annotations[0].category = "临时人工改动"
        with self.assertRaises(ValueError):
            ensure_canvas_matches(self.window, pending)
        canvas.annotations[0].category = "脏污"
        dock.launch_local(pending)
        self.wait()
        self.assertEqual([a.category for a in canvas.annotations], ["污点", "污点"])
        self.assertNotIn(self.paths[0], self.window.sample_assignments)
        self.assertTrue(self.window.centralWidget().isEnabled())
        undo = tools.store.prepare_undo(pending["transaction_id"])
        dock.launch_local({"kind": "transaction", **undo})
        self.wait()
        self.assertEqual([a.category for a in canvas.annotations], ["脏污", "脏污"])
        self.assertEqual(self.window.sample_assignments[self.paths[0]], {"脏污": "小面积"})

    def test_composed_chat_applies_group_and_persists_export(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        dock.api_key.setText("fake-test-key")
        dock.input.setPlainText("筛选大面积脏污并导出")
        responses = [
            ({"action": "tool", "tool": "select", "args": {"code": AREA_CODE, "title": "大面积", "criteria": ">5%"}}, {}),
            ({"action": "tool", "tool": "export", "args": {"scope": "last"}}, {}),
            ({"action": "reply", "message": "完成。"}, {}),
        ]
        with patch("Utils.AIAugment.request_qwen_json", side_effect=responses):
            dock.submit()
            self.assertFalse(self.window.centralWidget().isEnabled())
            self.wait()
        self.assertEqual(self.window.listWidget.count(), 2)
        self.assertTrue(self.window.centralWidget().isEnabled())
        entries = Workspace(self.root).list_entries()
        self.assertTrue(any(e["kind"] == "dataset" and e["status"] == "completed" for e in entries))

    def test_followup_crop_and_1280_tiles_use_the_displayed_selection(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        dock.api_key.setText("fake-test-key")
        dock.input.setPlainText("挑选出大面积的脏污")
        with patch("Utils.AIAugment.request_qwen_json", side_effect=[
                ({"action": "select", "message": "筛选大面积", "title": "大面积脏污", "criteria": ">5%", "code": AREA_CODE}, {}),
                ({"action": "reply", "message": "已筛选。"}, {})]):
            dock.submit()
            self.wait()
        selected = set(dock.work_context()["current_paths"])
        self.assertEqual(selected, {self.paths[1], self.paths[2]})
        for message, args in [("裁出这组脏污", {"mode": "defect", "label": "脏污"}),
                              ("把这组图片切成1280*1280", {"mode": "tiles", "tile_size": 1280})]:
            # Generated output now becomes the current view. Explicitly return
            # to the source group when requesting a second source operation.
            group_id = next(iter(self.window.temporary_selections))
            self.window.on_sample_tree_item_clicked(self.window.find_sample_tree_item(("temporary", group_id, None)), 0)
            dock.input.setPlainText(message)
            with patch("Utils.AIAugment.request_qwen_json", side_effect=[
                    ({"action": "tool", "tool": "crop", "args": args}, {}),
                    ({"action": "reply", "message": "已切图。"}, {})]) as request:
                dock.submit()
                self.wait()
            self.assertIn('"current_count": 2', request.call_args_list[0].args[3])
            result = dock.details[-1]["work"]["steps"][0]["result"]
            self.assertEqual(result["source_count"], 2)
            folder = Path(result["output"])
            self.assertEqual({r["source"] for r in read_json(folder / "sources.json")}, selected)
            self.assertEqual(len(self.window.temporary_selections), 1)
            self.assertEqual(self.window.current_tree_filter[0], "artifact")
            self.assertEqual(self.window.result_browser.list.count(), result["count"])
            if args["mode"] == "tiles":
                for image_path in (folder / "images").glob("*.png"):
                    with Image.open(image_path) as image:
                        self.assertEqual(image.size, (1280, 1280))

    def test_confirmed_change_resumes_remaining_work_with_same_request_budget(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        dock.api_key.setText("fake-test-key")
        dock.input.setPlainText("把选中的图片的脏污改为污点，再导出这张图片和标签")
        responses = [
            ({"action": "tool", "tool": "relabel", "args": {"scope": "selected", "old": "脏污", "new": "污点"}}, {}),
            ({"action": "tool", "tool": "export", "args": {"scope": "selected"}}, {}),
            ({"action": "reply", "message": "已修改并导出。"}, {}),
        ]
        with patch("Utils.AIAugment.request_qwen_json", side_effect=responses) as request:
            dock.submit()
            self.wait()
            self.assertEqual(request.call_count, 1)
            self.assertIsNotNone(dock.pending_operation)
            run_id = dock.pending_operation["run_id"]
            dock.launch_local(dock.pending_operation)
            self.wait()
            self.assertEqual(request.call_count, 3)
        store = Workspace(self.root)
        record = read_json(store.location("runs", run_id) / "run.json")
        self.assertEqual(record["requests"], 3)
        self.assertEqual(record["status"], "completed")
        exports = [e for e in store.list_entries() if e["kind"] == "dataset"]
        self.assertEqual(len(exports), 1)
        rows = read_json(store.location("artifacts", exports[0]["id"]) / "sources.json")
        self.assertEqual(rows[0]["document"]["annotations"][0]["lable"], "污点")

    def test_candidate_adoption_updates_canvas_and_flag_and_can_restore_missing_json(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        model_path = self.root / "fake.pt"
        model_path.write_bytes(b"mock local predictor")
        ctx = dock.work_context()
        ctx["resources"] = {"m": {"path": str(model_path), "kind": "model", "name": "fake.pt"}}
        tools = WorkTools(self.root, self.paths, context=ctx)
        plan = tools.execute("preannotate", {"model": "m"})["pending"]
        predictor = SimpleNamespace(names={0: "脏污"}, task="detect", predict=lambda **kwargs: [SimpleNamespace(
            boxes=SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([.9]), xyxyn=torch.tensor([[.1, .1, .5, .5]])), masks=None)])
        result = generate_candidates(self.root, plan["artifact_id"], predictor=predictor)
        self.window.listWidget.setCurrentRow(4)
        dock.adopt_candidates(result["artifact_id"], [self.paths[4]])
        self.wait()
        self.assertEqual(self.window.flag[4], 1)
        self.assertEqual(len(self.window.graphicsView.imageItem.annotations), 1)
        tid = next(e["id"] for e in tools.store.list_entries("transactions") if e["status"] == "committed")
        undo = tools.store.prepare_undo(tid)
        dock.launch_local({"kind": "transaction", **undo})
        self.wait()
        self.assertEqual(self.window.flag[4], 0)
        self.assertEqual(len(self.window.graphicsView.imageItem.annotations), 0)
        self.assertFalse((self.root / "jsons/missing.json").exists())

    def test_saved_workflow_continues_after_confirmed_step_without_replaying_it(self):
        self.window.show_ai_chat()
        dock = self.window.ai_chat_dock
        dock.api_key.setText("fake-test-key")
        store = Workspace(self.root)
        workflow_id = "test_saved_workflow"
        write_json(store.location("workflows", workflow_id) / "workflow.json", {
            "id": workflow_id, "kind": "workflow", "title": "修改后导出", "status": "saved", "created_at": "2026-09-22",
            "steps": [{"tool": "relabel", "args": {"scope": "selected", "old": "脏污", "new": "污点"}},
                      {"tool": "export", "args": {"scope": "selected"}}]})
        dock.input.setPlainText("运行已保存的修改后导出")
        responses = [({"action": "tool", "tool": "run_workflow", "args": {"workflow_id": workflow_id}}, {}),
                     ({"action": "reply", "message": "保存的步骤已运行完成。"}, {})]
        with patch("Utils.AIAugment.request_qwen_json", side_effect=responses) as request:
            dock.submit()
            self.wait()
            self.assertEqual(len(dock.continuation["workflow_tail"]), 1)
            dock.launch_local(dock.pending_operation)
            self.wait()
            self.assertEqual(request.call_count, 2)
        self.assertEqual(len([e for e in store.list_entries() if e["kind"] == "dataset"]), 1)
        self.assertEqual(len([e for e in store.list_entries("transactions") if e["status"] == "committed"]), 1)


if __name__ == "__main__":
    unittest.main()
