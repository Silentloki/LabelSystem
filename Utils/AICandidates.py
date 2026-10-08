"""Local-model candidate generation and explicit adoption of empty annotations."""
import copy
import pickle
from pathlib import Path

from Utils.AIChatSelection import annotation_features
from Utils.AIWorkspace import Workspace, document_token, file_sha, json_bytes, read_json, read_project_list, sha, write_json
from Utils.AIWorkTools import WorkTools, bounds, class_name, numeric
from Utils.AIModelIdentity import model_label


def generate_candidates(project, plan_id, cancelled=lambda: False, progress=lambda text: None, predictor=None):
    store = Workspace(project)
    try:
        return _generate_candidates(store, project, plan_id, cancelled, progress, predictor)
    except Exception as exc:
        for folder, manifest in store.open_artifacts.values():
            manifest.update(status="stopped" if cancelled() else "failed", error=str(exc)[:1500])
            write_json(folder / "manifest.json", manifest)
        raise


def _generate_candidates(store, project, plan_id, cancelled, progress, predictor):
    _, plan = store.artifact(plan_id, "prediction_plan")
    params = plan["metadata"]
    model_path = Path(params["model_path"])
    if file_sha(model_path) != params["model_sha256"]:
        raise ValueError("模型文件已改变，请重新提出预标注请求。")
    if predictor is None:
        from ultralytics import YOLO
        predictor = YOLO(str(model_path))
    if getattr(predictor, "task", "detect") not in {"detect", "segment"}:
        raise ValueError("当前候选入口支持检测与分割模型。")
    names = predictor.names
    names = {i: str(v) for i, v in enumerate(names)} if isinstance(names, list) else {int(k): str(v) for k, v in names.items()}
    mapping = params["class_map"]
    classes = set(params["classes"])
    if any(v not in classes for v in mapping.values()):
        raise ValueError("映射后的类别必须是工程已有类别。")
    if not any(mapping.get(name, name) in classes for name in names.values()):
        raise ValueError("模型类别与工程类别不匹配，请选择项目模型或明确提供类别映射。")
    imgsz = params.get('imgsz')
    if imgsz is None:
        imgsz = getattr(predictor, 'overrides', {}).get('imgsz') or 640
    imgsz = ([numeric(size, 32, 4096, '模型输入尺寸', True) for size in imgsz]
             if isinstance(imgsz, (list, tuple)) else numeric(imgsz, 32, 4096, '模型输入尺寸', True))
    tools = WorkTools(project, params["paths"], overrides=params.get("overrides"), cancelled=cancelled, progress=progress)
    entry_id, folder, manifest = store.create("candidates", "模型推理 · " + model_label(params),
                                             {"plan_id": plan_id, "model": model_path.name,
                                              "model_path": str(model_path), "model_resource": params.get('model_resource'),
                                              "model_sha256": params["model_sha256"], "conf": params["conf"],
                                              "imgsz": imgsz, "imgsz_source": 'explicit' if params.get('imgsz') is not None else 'model',
                                              "model_classes": names, "project_classes": sorted(classes),
                                              "class_map": mapping, "class_filter": "project_classes"})
    rows, errors = [], []
    for index, relative in enumerate(params["paths"]):
        tools.check()
        progress(f"本地模型预测 {index + 1}/{len(params['paths'])}：{Path(relative).name}")
        try:
            image = store.source(relative)
            before_hash = file_sha(image)
            annotation_path = store.source("jsons/" + image.stem + ".json")
            original_bytes = annotation_path.read_bytes() if annotation_path.exists() else None
            original = tools.document(relative, missing_ok=True)
            results = predictor.predict(source=str(image), conf=params["conf"], imgsz=imgsz,
                                        save=False, verbose=False)
            tools.check()
            if len(results) != 1:
                raise ValueError("本地模型未返回对应单图结果。")
            result = results[0]
            boxes = result.boxes
            anns = []
            ignored = 0
            if boxes is not None:
                ids = boxes.cls.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                rectangles = boxes.xyxyn.cpu().tolist()
                polygons = result.masks.xyn if getattr(result, "masks", None) is not None else None
                for i, (idx, score, box) in enumerate(zip(ids, scores, rectangles)):
                    model_name = names[int(idx)]
                    label = mapping.get(model_name, model_name)
                    if label not in classes:
                        ignored += 1
                        continue
                    if polygons is not None:
                        ann = {"type": "polygon", "lable": label, "points": [
                            {"x": min(1., max(0., float(x))), "y": min(1., max(0., float(y)))} for x, y in polygons[i]]}
                    else:
                        x0, y0, x1, y1 = [min(1., max(0., float(v))) for v in box]
                        ann = {"type": "rect", "lable": label, "points": [{"x": x0, "y": y0}, {"x": x1, "y": y1}]}
                    ann["confidence"] = float(score)
                    annotation_features(ann, original["image_width"], original["image_height"])
                    anns.append(ann)
            current_bytes = annotation_path.read_bytes() if annotation_path.exists() else None
            if file_sha(image) != before_hash or document_token(annotation_path, current_bytes) != document_token(annotation_path, original_bytes):
                raise ValueError("预测期间源数据发生变化，此图结果未采用。")
            rows.append({"path": relative, "image_sha256": before_hash,
                         "before_token": document_token(annotation_path, original_bytes),
                         "source_document": original, "had_annotation_file": original_bytes is not None,
                         "document": {"image_width": original["image_width"], "image_height": original["image_height"], "annotations": anns},
                         "can_adopt": not original["annotations"] and bool(anns),
                         "ignored_classes": ignored})
        except Exception as exc:
            if cancelled():
                raise ValueError("预标注已停止，未修改正式标注。") from None
            errors.append({"path": relative, "error": str(exc)[:500]})
    write_json(folder / "candidates.json", rows)
    write_json(folder / "errors.json", errors)
    result = store.finish(folder, manifest, count=len(rows), failed=len(errors),
                          predicted_annotations=sum(len(r["document"]["annotations"]) for r in rows),
                          ignored_predictions=sum(r['ignored_classes'] for r in rows),
                          adoptable=sum(r["can_adopt"] for r in rows),
                          protected=sum(bool(r["source_document"]["annotations"]) for r in rows),
                          note="按项目类别保留预测，其他类别已过滤；候选不等于正确标注，已有正式标注不覆盖，无预测图片不自动标为良品。")
    return result


def prepare_adoption(project, candidate_id, paths, previous_flags=None):
    store = Workspace(project)
    folder, _ = store.artifact(candidate_id, "candidates")
    rows = {row["path"]: row for row in read_json(folder / "candidates.json")}
    classes = set((store.project / 'label.txt').read_text(encoding='utf-8-sig').splitlines())
    if not paths or any(path not in rows for path in paths):
        raise ValueError("请选择该候选结果内的图片。")
    changes, previews, image_hashes, expected_tokens = {}, [], {}, {}
    for relative in dict.fromkeys(paths):
        row = rows[relative]
        if (row.get('unmapped_classes') or
                any(class_name(ann) not in classes for ann in row['document']['annotations'])):
            raise ValueError('候选存在未映射类别或项目类别变化，请明确类别映射并重新预测：' + relative)
        if not row["can_adopt"]:
            raise ValueError("该图片已有标注或没有可采用的预测：" + relative)
        image = store.source(relative)
        annotation = store.source("jsons/" + image.stem + ".json")
        current = annotation.read_bytes() if annotation.exists() else None
        if file_sha(image) != row["image_sha256"] or document_token(annotation, current) != row["before_token"]:
            raise ValueError("候选生成后源数据改变，请重新预测：" + relative)
        if current is not None and read_json(annotation).get("annotations"):
            raise ValueError("已有人工标注，不能覆盖：" + relative)
        doc = copy.deepcopy(row["document"])
        for ann in doc["annotations"]:
            # The annotation canvas round-trips the native shape fields only.
            # Keep confidence in the candidate record, not the formal JSON.
            ann.pop("confidence", None)
            annotation_features(ann, doc["image_width"], doc["image_height"])
        changes["jsons/" + image.stem + ".json"] = json_bytes(doc)
        expected_tokens["jsons/" + image.stem + ".json"] = row["before_token"]
        image_hashes[relative] = row["image_sha256"]
        previews.append({"image": relative, "annotations": len(doc["annotations"]), "action": "采用已查看的候选标注"})
    path_file, flag_file = store.project / "datafile.dat", store.project / "flagfile.dat"
    initial_paths, initial_flags = path_file.read_bytes(), flag_file.read_bytes()
    image_order, flags = read_project_list(path_file, initial_paths), read_project_list(flag_file, initial_flags)
    if len(image_order) != len(flags) or any(type(p) is not str for p in image_order) or any(type(f) is not int for f in flags):
        raise ValueError("工程图片与状态列表不对应，不能采用候选。")
    previous_flags = {p: flags[image_order.index(p)] for p in paths}
    for p in paths:
        flags[image_order.index(p)] = 1
    changes["flagfile.dat"] = pickle.dumps(flags)
    expected_tokens["flagfile.dat"] = document_token(flag_file, initial_flags)
    return store.prepare_transaction("采用本地模型候选", changes,
                                     {"candidate_id": candidate_id, "changes": previews, "affected_images": list(paths),
                                      "set_flags": {p: 1 for p in paths}, "previous_flags": previous_flags,
                                      "image_hashes": image_hashes, "expected_tokens": expected_tokens,
                                      "state_hashes": {"datafile.dat": sha(initial_paths)}})


def candidate_evaluation(tools, candidate_id, iou=.5):
    from Utils.AIWorkTools import numeric
    threshold = numeric(iou, .01, 1, "匹配IoU")
    folder, _ = tools.store.artifact(candidate_id, "candidates")
    rows = read_json(folder / "candidates.json")
    if any(row.get('unmapped_classes') for row in rows):
        raise ValueError('候选存在未映射类别，不能直接按项目大类评估漏检；请明确类别映射后重新预测。')
    output, missing, extra, skipped = [], [], [], []

    def overlap(a, b):
        ax0, ay0, ax1, ay1 = bounds(a)
        bx0, by0, bx1, by1 = bounds(b)
        intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
        return intersection / max(1e-15, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection)

    for row in rows:
        tools.check()
        rel = row["path"]
        if rel not in tools.known or not row["had_annotation_file"]:
            skipped.append(rel)
            continue
        gt = row["source_document"]["annotations"]
        if not gt and tools.context.get("flags", {}).get(rel) not in (2, 3):
            skipped.append(rel)
            continue
        annotation = tools.store.source("jsons/" + Path(rel).stem + ".json")
        if (document_token(annotation, annotation.read_bytes() if annotation.exists() else None) != row["before_token"]
                or file_sha(tools.store.source(rel)) != row["image_sha256"]):
            raise ValueError("预测后的原标注/图片发生变化，不能混合版本评估。")
        predictions = row["document"]["annotations"]
        candidates = sorted([(overlap(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(predictions)
                             if class_name(g) == class_name(p) and overlap(g, p) >= threshold], reverse=True)
        used_gt, used_pred = set(), set()
        for score, i, j in candidates:
            if i not in used_gt and j not in used_pred:
                used_gt.add(i)
                used_pred.add(j)
        missed, false = len(gt) - len(used_gt), len(predictions) - len(used_pred)
        output.append({"path": rel, "matched": len(used_gt), "missed": missed, "extra_predictions": false})
        if missed:
            missing.append(rel)
        if false:
            extra.append(rel)
    tools.add_group("模型漏检候选", f"依据已有标注，同类框IoU ≥ {threshold} 贪心一对一匹配后未匹配标注", missing)
    tools.add_group("模型额外预测", f"同类框IoU ≥ {threshold} 贪心匹配后未匹配预测；需人工核对", extra)
    entry_id, output_dir, manifest = tools.store.create("evaluation", "模型预测与标注对照", {"candidate_id": candidate_id, "iou": threshold})
    write_json(output_dir / "evaluation.json", {"rows": output, "skipped": skipped})
    return tools.store.finish(output_dir, manifest, images=len(output), skipped=len(skipped),
                              missed=sum(r["missed"] for r in output), extra_predictions=sum(r["extra_predictions"] for r in output),
                              note="基于现有标注的框IoU对照，不是官方mAP评测；不证明原标注一定正确。")
