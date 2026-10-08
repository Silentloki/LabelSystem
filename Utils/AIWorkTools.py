"""Domain tools for the visual-work assistant. Tools never execute shell code."""
import copy
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from Utils.AIChatSelection import annotation_features, project_snapshot
from Utils.AISelectionCode import SelectionProgram
from Utils.AIWorkspace import Workspace, document_token, file_sha, json_bytes, read_json, sha, stamp, write_json
from Utils.AIResultBrowser import VISUAL_KINDS
from Utils.AIResources import resource_catalog, resource_choices, training_folders
from Utils.AIModelIdentity import model_label


# The same registry drives the prompt and rejects unknown / misspelled arguments.
TOOL_SPECS = {
    "available_actions": ("查询当前照片范围能执行的操作和限制。范围不支持时先核对能力，不用记录问题冒充正式修改。", {"scope?": "默认selected"}),
    "set_sample_status": ("将工程照片设为完全良品/过杀品，或清除其良品状态。良品/过杀品会清除缺陷标注与子样本归类；先展示具体预览，确认后可恢复。", {
        "status": "good=完全良品|overkill=过杀品|clear_good=清除良品/过杀品状态",
        "scope?": "默认selected；支持对应工程原图的推理结果和问题集；派生图须先明确定位来源原图"}),
    "original_images": ("仅当用户明确要处理结果的来源原图时，定位这些工程原图并建立临时组，后续用last。不会采用预测标注。", {"scope?": "默认selected"}),
    "inspect_selection": ("实际查看本次选中/当前照片及对应结果，分析共同问题。每批最多10张照片，超过10张按序翻页，不要求用户重新选择。", {
        "offset?": "首批0；后续必须使用返回的next_offset，不能跳过图片",
        "previous_report?": "翻页前提交上一批的逐图报告列表，每项image_id/readable/finding/recommendation；首批省略"}),
    "record_feedback": ("按用户明确的‘记为漏检/误检/图片质量问题/标注待核对/正常’指令记录本次选中照片。仅保存用户判断，不接受AI推测。", {
        "category": "false_positive|false_negative|image_quality|annotation_question|correct"}),
    "add_error_set": ("按用户明确要求将本次选中照片加入问题样本集（error set），保存来源引用并展示，不复制图片或修改正式标注。", {
        "title?": "名称，默认问题样本集；每次选择保存为独立集合，相同请求复用"}),
    "project_context": ("查询项目状态地图：数据集划分、关联模型、实验、处理批次及人工问题记录。只读，不运行模型。", {
        "topic?": "overview|datasets|resources|experiments|results|lineage|feedback，默认overview；lineage按图片查询指定批次的来源关系",
        "day?": "YYYY-MM-DD|today|yesterday；results/experiments按开始日期，feedback按人工记录日期",
        "category?": "仅feedback：false_positive|false_negative|image_quality|annotation_question|correct",
        "artifact_id?": "限定具体处理/推理批次", "offset?": "分页偏移，默认0", "limit?": "1~20，默认20"}),
    "select_feedback": ("将仍有效的人工问题记录变为可继续处理的图片范围；误检漏检须指定具体推理批次。", {
        "category": "false_positive|false_negative|image_quality|annotation_question|correct",
        "artifact_id?": "推理或处理批次ID", "day?": "人工记录日期YYYY-MM-DD|today|yesterday"}),
    "select": ("根据图片自己的标注筛选。原图生成临时分组，处理结果生成引用快照，相同筛选复用结果。", {"code": "Python筛选程序", "title": "名称", "criteria": "精确条件", "group_id?": "仅更新原图分组时的ID", "scope?": "默认current；处理结果可用current_result或artifact:ID；明确从全部原图重筛才用all"}),
    "stats": ("统计指定范围的各类图片数、标注数和面积。", {"scope?": "范围"}),
    "export": ("导出原图或已完成的切图/数据产物。导出为数据用native（图片+JSON标注，不划分）；导出为数据集用yolo（train/val、data.yaml和子分类统计）。两种都要时分别调用。", {"scope?": "all=全部原图、current=当前范围、selected=选中图片（支持结果多选）、group:ID=筛选组、current_result=当前处理结果、artifact:ID=指定结果、last=最近范围", "content?": "dataset=训练数据集|annotated=图片和JSON标注|images=仅图片|labels=仅JSON标注|predictions=预测记录|preview=带预测框图片；默认按format", "format?": "yolo（默认，与软件一致）|native|yolo_detect|yolo_segment", "train_ratio?": "仅显式yolo_detect/segment可定制比例；默认yolo复用软件分层规则", "seed?": "整数随机种子"}),
    "crop": ("围绕指定类别标注裁剪，或网格切图；同步所有相交标注。", {"scope?": "范围", "mode?": "defect|tiles", "label?": "defect模式的类别", "padding?": "四周边距占目标框比例0~2，默认0.2", "tile_size?": "切图边长像素", "overlap?": "切图重叠比例0~0.8"}),
    "convert_dataset": ("整理用户添加的外部YOLO目录/LabelSystem原生目录，输出独立可导入数据。不会猜测不匹配文件。", {"resource": "已添加的目录ID", "format": "yolo|native"}),
    "duplicates": ("按图片文件SHA256找完全相同文件，创建临时分组。", {"scope?": "范围"}),
    "similar": ("按灰度感知哈希找外观相似候选；不是缺陷语义识别。", {"scope?": "范围", "reference?": "current或工程图片ID", "top_n?": "1~100，默认20"}),
    "relabel": ("准备批量修改标注类别的具体预览，等待用户确认。", {"scope?": "范围", "old": "原类别", "new": "目标类别"}),
    "undo": ("准备恢复某次已完成修改；后续发生变化则拒绝覆盖。", {"transaction_id": "修改记录ID"}),
    "snapshot": ("保存数据版本指纹与标注内容，不复制原图。", {"scope?": "范围", "title?": "版本名称"}),
    "compare": ("比较两个已保存版本，或版本与当前工程。", {"before": "快照产物ID", "after?": "快照产物ID或current"}),
    "preannotate": ("用本地.pt模型生成候选标注，按项目类别过滤，不写原标注。先展示模型、数量和参数，用户启动后运行。", {"scope?": "范围", "model": "已添加模型资源ID", "conf?": "0~1，默认0.5", "imgsz?": "可选输入尺寸；用户未指定时省略，沿用模型训练尺寸，不自行填640", "class_map?": "用户明确指定的模型名称到现有工程类别的映射；不属于项目类别且未映射的预测正常过滤"}),
    "evaluate": ("将候选预测与原标注按同类框IoU匹配，整理漏检/额外预测图片。不是AI标注审核。", {"candidate_id": "候选产物ID", "iou?": "0~1，默认0.5"}),
    "prediction_results": ("分析已保存的模型推理结果：定位批次，读取实际模型/参数、预测与无预测图片数、类别和置信度统计；不需要results.csv，不重新推理，不代表看图。", {"candidate_id?": "候选推理批次ID或current；省略时优先匹配当前批次，多批不猜选", "model?": "模型资源编号，如r3；按内容指纹关联已有推理批次", "offset?": "图片或待选批次分页偏移，默认0", "limit?": "每页1~20，默认20"}),
    "inspect_predictions": ("实际看图分析已保存推理结果。准备原图、预测框/置信度与局部图，下一轮请求自动附图；不重跑模型，不修改标注。", {"candidate_id?": "候选批次ID或current", "model?": "模型资源编号如r3", "scope?": "sample默认抽样；selected只看当前结果选中项；images明确图片列表", "image_ids?": "scope=images时该批图片相对路径列表，不是文件名猜测", "limit?": "仅控制sample抽样数，1~10，默认3；selected/images自动完整读取指定的最多10张，不受较小limit限制；超过10张分批，不静默截取", "with_reference?": "默认false；用户要求与原标注对照时true，红色显示推理时标注参考"}),
    "training_runs": ("读取训练指标并准备PR图和混淆矩阵，下一次请求直接完成逐类分析，不再询问是否看图。多个结果时先选择一个。", {"resource?": "resources中的目录ID，如r1；project明确读取当前工程。不可填产物ID", "analyze?": "默认true直接分析；仅列出训练记录时用false"}),
    "training_compare": ("读取两个结果目录，对比已有指标和数据可比性。", {"current": "目录资源ID", "previous": "目录资源ID"}),
    "training_plan": ("记录多个训练配置，不自动启动；可选一项填入训练页。", {"parameter": "imgsz|epochs|batch", "values": "最多5个正整数"}),
    "history": ("读取本工程的产物、操作与修改记录。", {}),
    "view_result": ("在软件左侧临时结果中打开已生成的推理、裁剪、导出或格式整理图片。不会重新推理。", {"artifact_id": "产物ID或last"}),
    "save_workflow": ("保存一段已成功执行的工具序列，供下次复用。", {"title": "工具名称"}),
    "run_workflow": ("在当前范围复用已保存步骤；原标注修改仍需具体预览。", {"workflow_id": "已保存工具ID"}),
}


def numeric(value, low, high, name, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name}必须在 {low}~{high} 之间。")
    if integer and int(value) != value:
        raise ValueError(name + "必须是整数。")
    return int(value) if integer else float(value)


def class_name(annotation):
    return annotation.get("lable") or annotation.get("label") or annotation.get("category")


def bounds(annotation):
    points = annotation["points"]
    return (min(p["x"] for p in points), min(p["y"] for p in points),
            max(p["x"] for p in points), max(p["y"] for p in points))


def clipped_annotations(document, box):
    """Rectangles are analytic; polygon clipping uses pixel masks (<=1 px rounding).

    Raster clipping preserves disconnected pieces of concave polygons, unlike
    returning a single bridged polygon from a simple rectangle clip algorithm.
    """
    import cv2
    import numpy as np
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    iw, ih = document["image_width"], document["image_height"]
    output = []
    for ann in document["annotations"]:
        label = class_name(ann)
        if ann["type"] == "rect":
            x0, y0, x1, y1 = bounds(ann)
            x0, y0 = max(0, x0 * iw - left), max(0, y0 * ih - top)
            x1, y1 = min(width, x1 * iw - left), min(height, y1 * ih - top)
            if x1 > x0 and y1 > y0:
                output.append({"type": "rect", "lable": label, "points": [
                    {"x": x0 / width, "y": y0 / height}, {"x": x1 / width, "y": y1 / height}]})
        else:
            mask = np.zeros((height + 1, width + 1), dtype=np.uint8)
            pts = np.array([[round(p["x"] * iw - left), round(p["y"] * ih - top)] for p in ann["points"]], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) >= 3 and cv2.contourArea(contour) > 0:
                    output.append({"type": "polygon", "lable": label, "points": [
                        {"x": min(width, max(0, int(p[0]))) / width,
                         "y": min(height, max(0, int(p[1]))) / height} for p in contour[:, 0, :]]})
    return output


class WorkTools:
    def __init__(self, project, paths, groups=None, overrides=None, context=None,
                 cancelled=lambda: False, progress=lambda text: None):
        self.store = Workspace(project)
        self.paths = list(paths)
        self.known = set(paths)
        self.groups = copy.deepcopy(groups or {})
        self.overrides = overrides or {}
        self.context = context or {}
        self.resources = self.context.get("resources", {})
        self.cancelled, self.progress = cancelled, progress
        self.effects = []
        self.last_scope = None
        self.last_artifact = None
        self.export_result = self.context.get("viewing_artifact")
        self.steps = []
        self.training_analysis = None
        self.prediction_analysis = None
        self.selection_analysis = None
        self.run_id = None

    def tool_inspect_selection(self, offset=0, previous_report=None):
        from Utils.AISelectedAnalysis import prepare
        return prepare(self, offset, previous_report)

    def tool_available_actions(self, scope='selected'):
        from Utils.AIImageOperations import available_actions
        return available_actions(self, scope)

    def original_scope(self, scope='selected', trace_sources=False):
        from Utils.AIImageOperations import original_scope
        return original_scope(self, scope, trace_sources)

    def tool_set_sample_status(self, status, scope='selected'):
        from Utils.AIImageOperations import set_sample_status
        return set_sample_status(self, status, scope)

    def tool_original_images(self, scope='selected'):
        paths = self.original_scope(scope, trace_sources=True)
        result = self.add_group('结果对应的工程原图', '明确定位来源原图；沿用正式标注，不采用预测', paths)
        result['note'] = '已定位来源工程原图；用last继续操作，按工程正式标注处理。'
        return result

    def tool_record_feedback(self, category):
        from Utils.AIResultActions import record_feedback
        return record_feedback(self, category)

    def tool_add_error_set(self, title='问题样本集'):
        from Utils.AIResultActions import add_error_set
        return add_error_set(self, title)

    def tool_project_context(self, topic='overview', day=None, category=None, artifact_id=None, offset=0, limit=20):
        from Utils.ProjectContext import ProjectContext
        value = ProjectContext(self.store.project).query(topic, day, category, artifact_id, offset, limit,
            paths=self.paths, flags=[self.context.get('flags', {}).get(p, 0) for p in self.paths]
                if 'flags' in self.context else None, classes=self.context.get('classes'), cancelled=self.cancelled)
        self.resources = ProjectContext(self.store.project).merge_resources(self.resources)
        return value

    def tool_select_feedback(self, category, artifact_id=None, day=None):
        from Utils.ProjectContext import ProjectContext, ISSUES
        if category in {'false_positive', 'false_negative', 'correct'} and not artifact_id:
            raise ValueError('请先查询人工问题记录，并明确具体推理批次；不能跨模型混合误检或漏检。')
        rows = ProjectContext(self.store.project).feedback(category, day, artifact_id)
        if any(row['stale'] for row in rows):
            raise ValueError('问题记录包含已变化或删除的图片/结果，请先在项目状态中核对，不能悄悄跳过。')
        images = list(dict.fromkeys(row['image'] for row in rows))
        title = '人工记录 · ' + ISSUES[category]
        if not images:
            self.last_scope = []
            self.last_artifact = None
            self.export_result = None
            return {'kind': 'selection', 'title': title, 'count': 0, 'note': '没有符合条件的有效人工记录；不回退其他图片范围。'}
        if artifact_id:
            _, manifest = self.store.artifact(artifact_id)
            if manifest['kind'] in {'crops', 'dataset', 'converted', 'selection', 'error_set'}:
                from Utils.AIResultSelection import create_selection, selection_source
                parent = selection_source(self.store, artifact_id)[0] if manifest['kind'] == 'selection' else artifact_id
                return create_selection(self, parent, images, title, '人工记录筛选')
            if manifest['kind'] != 'candidates':
                raise ValueError('此类记录不支持作为图片处理范围。')
        if any(path not in self.known for path in images):
            raise ValueError('问题图片不在原图工程范围，请选择它所属的具体处理结果。')
        result = self.add_group(title, '人工记录：' + (artifact_id or '原图') + '；日期：' + (day or '全部'), images)
        result['source_artifact'] = artifact_id
        result['scope_note'] = '来自该推理批次人工记录的原图片，可使用last继续处理；不采用预测标注'
        return result

    def check(self):
        if self.cancelled():
            raise ValueError("操作已停止；未完成产物不会作为成功结果使用。")

    def scope(self, name="current"):
        if name in {"current", "selected"} and self.context.get("viewing_artifact"):
            raise ValueError("当前正在预览处理产物，不能把它当作工程原图片操作；请先选择左侧原样本分组。")
        if name == "all":
            values = self.paths
        elif name == "current":
            values = self.context.get("current_paths", self.paths)
        elif name == "selected":
            values = self.context.get("selected_paths", [])
        elif name == "last":
            if self.last_scope is None:
                raise ValueError("尚无上一步样本结果。")
            values = self.last_scope
        elif isinstance(name, str) and name.startswith("group:") and name[6:] in self.groups:
            values = self.groups[name[6:]]["paths"]
        else:
            raise ValueError("无效样本范围；请选择当前视图、已选图片或已有临时分组。")
        if any(p not in self.known for p in values):
            raise ValueError("范围中的图片已不在工程中。")
        return list(dict.fromkeys(values))

    def resource(self, name, kind=None):
        if isinstance(name, str) and name.startswith('p') and name not in self.resources:
            from Utils.ProjectContext import ProjectContext
            resource = ProjectContext(self.store.project).resources().get(name)
            if resource:
                self.resources[name] = resource
        if not isinstance(name, str) or name not in self.resources:
            choices = resource_choices(self.resources, kind)
            if choices:
                raise ValueError(f"外部资源ID无效：{str(name)[:100]}。外部资源已添加，可选：{choices}。请用r1这类资源ID，不要填产物ID或文件路径；无需用户重复添加。")
            raise ValueError("尚未添加所需类型的外部资源，请用“添加目录/模型”选择目录或模型。")
        resource = self.resources[name]
        if kind and resource["kind"] != kind:
            raise ValueError("资源类型不匹配。")
        path = Path(resource["path"]).resolve()
        if not path.exists():
            raise ValueError("所选资源已不存在。")
        if kind == 'model':
            from Utils.ProjectContext import ProjectContext
            project_context = ProjectContext(self.store.project)
            canonical = project_context.resource_id(path, 'model')
            preferred = project_context.settings().get('preferred_model', {})
            if preferred.get('resource_id') == canonical and preferred.get('sha256') != file_sha(path):
                raise ValueError('项目主用模型文件已被替换，请在项目状态中重新选择确认，不能沿用旧版本身份。')
        return path

    def document(self, relative, missing_ok=False):
        if relative not in self.known:
            raise ValueError("未知工程图片。")
        image = self.store.source(relative)
        if not image.is_file():
            raise ValueError("图片不存在：" + relative)
        source = self.store.source("jsons/" + image.stem + ".json")
        original = read_json(source) if source.exists() else None
        if original is not None and (not isinstance(original, dict) or not isinstance(original.get("annotations"), list)):
            raise ValueError("标注文件无效：" + relative)
        doc = copy.deepcopy(self.overrides.get(relative, original))
        if doc is None:
            if not missing_ok and self.context.get("flags", {}).get(relative) not in (2, 3):
                raise ValueError("缺少标注，不能当作负样本：" + relative)
            doc = {"annotations": []}
        with Image.open(image) as im:
            width, height = im.size
        doc["image_width"], doc["image_height"] = width, height
        for annotation in doc["annotations"]:
            annotation_features(annotation, width, height)
        return doc

    def add_group(self, title, criteria, paths, code="", group_id=None):
        import uuid
        if group_id and group_id not in self.groups:
            raise ValueError("要更新的临时分组不存在。")
        if not group_id:
            for existing_id, group in self.groups.items():
                if group['title'] == title and group['criteria'] == criteria and group['paths'] == paths:
                    self.last_scope = paths
                    return {"group_id": existing_id, "title": title, "criteria": criteria,
                            "count": len(paths), "reused_result": True}
        group_id = group_id or uuid.uuid4().hex
        self.groups[group_id] = {"title": title, "criteria": criteria, "paths": paths, "code": code}
        self.effects.append({"kind": "group", "group_id": group_id, **self.groups[group_id]})
        self.last_scope = paths
        return {"group_id": group_id, "title": title, "criteria": criteria, "count": len(paths)}

    def execute(self, name, args):
        self.check()
        if name not in TOOL_SPECS or not isinstance(args, dict):
            raise ValueError("不支持的工具调用。")
        spec = TOOL_SPECS[name][1]
        allowed = {key.rstrip("?") for key in spec}
        required = {key for key in spec if not key.endswith("?")}
        if set(args) - allowed or required - set(args):
            raise ValueError("工具参数不匹配：" + name)
        self.progress("正在执行：" + TOOL_SPECS[name][0])
        args = copy.deepcopy(args)
        for field in ("candidate_id", "before", "after", "artifact_id"):
            if args.get(field) == "last":
                if self.last_artifact is None:
                    raise ValueError("尚无上一步产物。")
                args[field] = self.last_artifact
        opened = set(self.store.open_artifacts)
        try:
            result = getattr(self, "tool_" + name)(**args)
        except Exception as exc:
            for entry_id in set(self.store.open_artifacts) - opened:
                folder, manifest = self.store.open_artifacts.pop(entry_id)
                manifest.update(status="stopped" if self.cancelled() else "failed", error=str(exc)[:2000])
                write_json(folder / "manifest.json", manifest)
            raise
        if result.get("artifact_id"):
            self.last_artifact = result["artifact_id"]
        self.steps.append({"tool": name, "args": copy.deepcopy(args), "result": result})
        return result

    def tool_select(self, code, title, criteria, group_id=None, scope="current"):
        from Utils.AIChatSelection import validate_response
        validate_response({"action": "select", "code": code, "title": title, "criteria": criteria,
                           "message": criteria, "group_id": group_id}, self.groups)
        source = self.export_scope(scope, allow_predictions=True)
        if source.get('artifact_id'):
            if group_id:
                raise ValueError('结果图片筛选不能覆盖原图分组。')
            from Utils.AIArtifactExport import export_rows
            from Utils.AIResultSelection import create_selection
            rows, context = export_rows(self, source['artifact_id'], source.get('selected'), allow_predictions=True)
            records = self.result_records(rows)
            paths = SelectionProgram(self.cancelled).run(code, records, {})
            result = create_selection(self, context['base_artifact'], paths, title, criteria)
            result['failed_images'] = context.get('failed_images', 0)
            result['annotation_basis'] = 'saved_predictions' if any(r.get('prediction') for r in rows) else 'saved_annotations'
            return result
        snapshot = project_snapshot(self.store.project, self.scope(scope), self.overrides, self.cancelled, self.progress)
        paths = SelectionProgram(self.cancelled).run(code, snapshot["records"], {k: v["paths"] for k, v in self.groups.items()})
        result = self.add_group(title, criteria, paths, code, group_id)
        result["skipped_images"] = len(snapshot["warnings"])
        return result

    def tool_stats(self, scope="current"):
        source = self.export_scope(scope, allow_predictions=True)
        if source.get('artifact_id'):
            from Utils.AIArtifactExport import export_rows
            rows, context = export_rows(self, source['artifact_id'], source.get('selected'), allow_predictions=True)
            records = self.result_records(rows)
            counts = Counter(a['label'] for row in records for a in row['annotations'])
            return {'summary': {'total_images': len(records), 'classes': {
                name: {'annotations': count, 'images': sum(any(a['label'] == name for a in r['annotations']) for r in records)}
                for name, count in counts.items()}}, 'skipped': [],
                'failed_images': context.get('failed_images', 0),
                'annotation_basis': 'saved_predictions' if any(r.get('prediction') for r in rows) else 'saved_annotations',
                'note': '预测统计不代表人工确认；无预测不等于良品。' if any(r.get('prediction') for r in rows) else ''}
        paths = self.scope(scope)
        snapshot = project_snapshot(self.store.project, paths, self.overrides, self.cancelled, self.progress)
        return {"summary": snapshot["summary"], "skipped": snapshot["warnings"][:20]}

    def result_records(self, rows):
        records = []
        for row in rows:
            self.check()
            doc = row['document']
            width, height = doc['image_width'], doc['image_height']
            records.append({'id': row['source'], 'filename': Path(row['source']).name,
                            'width': width, 'height': height,
                            'annotations': [annotation_features(a, width, height) for a in doc['annotations']]})
        return records

    def _rows(self, paths):
        rows = []
        for relative in paths:
            self.check()
            expected_image = file_sha(self.store.source(relative))
            doc = self.document(relative)
            if file_sha(self.store.source(relative)) != expected_image:
                raise ValueError("读取期间图片发生变化：" + relative)
            rows.append({"source": relative, "document": doc, "image_hash": expected_image})
        return rows

    def _new_dataset(self, kind, title, rows, metadata=None):
        if not rows:
            raise ValueError("样本范围为空。")
        entry_id, folder, manifest = self.store.create(kind, title, metadata)
        (folder / "images").mkdir()
        (folder / "jsons").mkdir()
        for i, row in enumerate(rows):
            self.check()
            image = self.store.source(row["source"])
            stem = f"{i + 1:06d}_" + image.stem
            target = folder / "images" / (stem + image.suffix.lower())
            shutil.copyfile(image, target)
            if file_sha(target) != row["image_hash"]:
                raise ValueError("复制期间原图片发生变化，产物未完成。")
            write_json(folder / "jsons" / (stem + ".json"), row["document"])
            row["image"] = "images/" + target.name
            row["json"] = "jsons/" + stem + ".json"
        return folder, manifest

    def export_scope(self, scope="current", allow_predictions=False):
        # Resolve the exact visible objects. Each operation separately decides
        # whether it can read predictions or needs project-original samples.
        aid = None
        if scope == 'selected' and self.context.get('viewing_artifact'):
            aid = self.context['viewing_artifact']
            if not self.context.get('selected_result_paths'):
                raise ValueError('正在预览处理产物，请先选中需要导出的结果图片。')
        elif scope == "current_result" or (scope == "current" and self.export_result):
            aid = self.export_result
            if not aid:
                raise ValueError("尚未选择处理结果，请先查看结果或指定artifact:产物ID。")
        elif isinstance(scope, str) and scope.startswith("artifact:"):
            aid = scope[len("artifact:"):]
        elif scope == "last":
            for step in reversed(self.steps):
                if step['result'].get('group_id'):
                    break
                if step['tool'] == 'select' and not step['result'].get('artifact_id'):
                    raise ValueError('上一步结果筛选没有匹配图片，不能回退到未筛选范围。')
                if step['tool'] in {'crop', 'convert_dataset', 'view_result', 'select', 'add_error_set', 'select_feedback'} and step['result'].get('artifact_id'):
                    aid = step['result'].get('artifact_id')
                    break
            if not aid and self.last_scope is None:
                aid = self.last_artifact or self.export_result
        if aid:
            _, manifest = self.store.artifact(aid)
            allowed = {"crops", "converted", "dataset", "selection", "error_set"}
            if allow_predictions:
                allowed.add('candidates')
            if manifest["kind"] not in allowed:
                raise ValueError("此结果不含可直接导出的图片与正式结果标注；模型候选需先核对采用。")
            source = {"artifact_id": aid}
            if scope == 'selected':
                source['selected'] = list(self.context['selected_result_paths'])
            return source
        return {"paths": self.scope(scope)}

    def tool_export(self, scope="current", format="yolo", train_ratio=.8, seed=42, content=None):
        if content not in {None, 'dataset', 'images', 'labels', 'annotated', 'predictions', 'preview'}:
            raise ValueError('未知导出内容。')
        if content in {'images', 'labels', 'predictions', 'preview'}:
            from Utils.AIDataExport import export_data
            return export_data(self, scope, content)
        if content == 'annotated':
            format = 'native'
        elif content == 'dataset' and format == 'native':
            raise ValueError('数据集请选择YOLO格式；图片加标注请选择annotated。')
        if format not in {"yolo", "native", "yolo_detect", "yolo_segment"}:
            raise ValueError("未知导出格式。")
        ratio = numeric(train_ratio, 0, 1, "训练比例")
        seed = numeric(seed, 0, 2147483647, "随机种子", True)
        source = self.export_scope(scope)
        rows, input_context = None, None
        if source.get("artifact_id"):
            from Utils.AIArtifactExport import export_rows
            rows, input_context = export_rows(self, source["artifact_id"], source.get('selected'))
            paths = [row["source"] for row in rows]
        else:
            paths = source["paths"]
        if format == "yolo":
            if ratio != .8:
                raise ValueError("软件一致导出使用既有分层规则；自定义比例请明确使用yolo_detect或yolo_segment。")
            from Utils.AIDatasetExport import export_dataset
            return export_dataset(self, paths, seed, rows=rows, input_context=input_context)
        rows = self._rows(paths) if rows is None else rows
        folder, manifest = self._new_dataset("dataset", "图片与JSON标注" if format == "native" else "YOLO数据集（AI）", rows,
                                             {"format": format, "train_ratio": ratio, "seed": seed,
                                              "source_artifact": source.get("artifact_id")})
        names = sorted({class_name(a) for row in rows for a in row["document"]["annotations"]})
        splits = {"train": 0, "val": 0}
        if format != "native":
            if not names:
                raise ValueError("没有缺陷类别，无法生成可训练的YOLO数据集。")
            # Exact duplicate source images stay together across train/val.
            hashes = sorted(set(row.get("split_group", row["image_hash"]) for row in rows))
            random.Random(seed).shuffle(hashes)
            count = round(len(hashes) * ratio)
            if 0 < ratio < 1:
                if len(hashes) < 2:
                    raise ValueError("只有一组独立图片，无法可靠划分训练集与验证集。")
                count = max(1, min(len(hashes) - 1, count))
            train = set(hashes[:count])
            for row in rows:
                self.check()
                split = "train" if row.get("split_group", row["image_hash"]) in train else "val"
                splits[split] += 1
                old = folder / row["image"]
                dest = folder / "yolo" / "images" / split / old.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, dest)
                lines = []
                for ann in row["document"]["annotations"]:
                    idx = names.index(class_name(ann))
                    if format == "yolo_detect":
                        x0, y0, x1, y1 = bounds(ann)
                        vals = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
                    else:
                        if ann["type"] != "polygon":
                            raise ValueError("分割导出需要真实多边形，不能将矩形自动当作精确分割。")
                        vals = [v for p in ann["points"] for v in (p["x"], p["y"])]
                    lines.append(str(idx) + " " + " ".join(f"{v:.10g}" for v in vals))
                label = folder / "yolo" / "labels" / split / (old.stem + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                row["split"] = split
            # Use the conventional block layout also understood by the app's
            # deliberately small YAML importer (not a whole JSON document).
            (folder / "yolo" / "data.yaml").write_text(
                "train: images/train\nval: images/val\n"
                "names: " + json.dumps(names, ensure_ascii=False) + "\n"
                "nc: " + str(len(names)) + "\n", encoding="utf-8")
        write_json(folder / "sources.json", rows)
        return self.store.finish_export(folder, manifest, count=len(rows), classes=names, splits=splits if format != "native" else None,
                                 format=format, source_artifact=source.get("artifact_id"),
                                 data_yaml=str(folder / "yolo/data.yaml") if format != "native" else None,
                                 note="已导出图片与JSON标注，不划分训练集/验证集。" if format == "native" else
                                      "同源切片或相同图片保持在同一侧；划分不保证每个小样本类别在两边都有。")

    def tool_crop(self, scope="current", mode="defect", label=None, padding=.2, tile_size=640, overlap=.2):
        if mode not in {"defect", "tiles"}:
            raise ValueError("未知裁剪方式。")
        padding = numeric(padding, 0, 2, "边距比例")
        tile_size = numeric(tile_size, 16, 8192, "切图尺寸", True)
        overlap = numeric(overlap, 0, .8, "重叠比例")
        if mode == "defect" and not isinstance(label, str):
            raise ValueError("请指定要围绕裁剪的缺陷类别。")
        source = self.export_scope(scope)
        if source.get('artifact_id'):
            from Utils.AIArtifactExport import export_rows
            rows, _ = export_rows(self, source['artifact_id'], source.get('selected'))
        else:
            rows = self._rows(source['paths'])
        if not rows:
            raise ValueError('此范围没有可切图的图片。')
        entry_id, folder, manifest = self.store.create("crops", "缺陷裁剪" if mode == "defect" else "网格切图",
                                                     {"mode": mode, "label": label, "padding": padding,
                                                      "tile_size": tile_size, "overlap": overlap})
        (folder / "images").mkdir()
        (folder / "jsons").mkdir()
        outputs = []
        for row in rows:
            self.check()
            doc = row["document"]
            width, height = doc["image_width"], doc["image_height"]
            boxes = []
            if mode == "defect":
                for ann in doc["annotations"]:
                    if class_name(ann) == label:
                        x0, y0, x1, y1 = bounds(ann)
                        dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
                        boxes.append((max(0, math.floor((x0 - dx) * width)), max(0, math.floor((y0 - dy) * height)),
                                      min(width, math.ceil((x1 + dx) * width)), min(height, math.ceil((y1 + dy) * height))))
            else:
                step = max(1, round(tile_size * (1 - overlap)))
                xs = list(range(0, max(1, width - tile_size + 1), step))
                ys = list(range(0, max(1, height - tile_size + 1), step))
                xs = sorted(set(xs + [max(0, width - tile_size)]))
                ys = sorted(set(ys + [max(0, height - tile_size)]))
                if len(xs) * len(ys) > 10000:
                    raise ValueError("单图切片超过10000张，请增加尺寸或降低重叠。")
                # PIL pads beyond the image with black. Keep exact requested
                # dimensions for small images; normalize labels to the tile.
                boxes = [(x, y, x + tile_size, y + tile_size) for y in ys for x in xs]
            with Image.open(self.store.source(row["source"])) as image:
                for box in dict.fromkeys(boxes):
                    self.check()
                    if len(outputs) >= 20000:
                        raise ValueError("本次切片超过20000张，请缩小范围。")
                    stem = f"crop_{len(outputs) + 1:06d}"
                    cropped = image.crop(box)
                    cropped.save(folder / "images" / (stem + ".png"))
                    annotations = clipped_annotations(doc, box)
                    out_doc = {"image_width": cropped.width, "image_height": cropped.height, "annotations": annotations}
                    for ann in annotations:
                        annotation_features(ann, cropped.width, cropped.height)
                    write_json(folder / "jsons" / (stem + ".json"), out_doc)
                    outputs.append({"source": row["source"], "image_hash": row["image_hash"], "box_pixels": box,
                                    "original_source": row.get('original_source', row['source']),
                                    "split_group": row.get('split_group') or row['image_hash'],
                                    "image": "images/" + stem + ".png", "json": "jsons/" + stem + ".json"})
            if file_sha(self.store.source(row["source"])) != row["image_hash"]:
                raise ValueError("裁剪期间原图改变，产物未完成。")
        write_json(folder / "sources.json", outputs)
        return self.store.finish(folder, manifest, count=len(outputs), source_count=len(rows),
                                 note=(f"切图尺寸 {tile_size}×{tile_size}；设定重叠 {overlap:.0%}，末端贴边可能增加重叠；不足尺寸右下补黑。" if mode == "tiles" else "")
                                      + "同步所有相交类别；多边形裁剪按像素栅格处理，边界有约1像素取整误差。")

    def tool_relabel(self, old, new, scope="current"):
        if not all(isinstance(v, str) and v.strip() and len(v) <= 100 and "\n" not in v and "\r" not in v for v in (old, new)):
            raise ValueError("类别名称无效。")
        old, new = old.strip(), new.strip()
        if old == new:
            raise ValueError("新旧类别相同，无需修改。")
        changes, preview, affected, guards, expected_tokens = {}, [], [], {}, {}
        for relative in self.original_scope(scope):
            self.check()
            annotation_source = self.store.source("jsons/" + Path(relative).stem + ".json")
            initial = annotation_source.read_bytes() if annotation_source.exists() else None
            expected = document_token(annotation_source, initial)
            doc = self.document(relative, missing_ok=True)
            count = 0
            for ann in doc["annotations"]:
                if class_name(ann) == old:
                    for key in ("lable", "label", "category"):
                        if key in ann:
                            ann[key] = new
                    count += 1
            if count:
                path = "jsons/" + Path(relative).stem + ".json"
                changes[path] = json_bytes(doc)
                preview.append({"image": relative, "old": old, "new": new, "annotations": count})
                affected.append(relative)
                guards[relative] = file_sha(self.store.source(relative))
                expected_tokens[path] = expected
        if not changes:
            raise ValueError("范围内没有该类别标注，没有修改。")
        labels = self.store.project / "label.txt"
        label_bytes = labels.read_bytes() if labels.exists() else None
        names = label_bytes.decode("utf-8-sig").splitlines() if label_bytes is not None else []
        if new not in names:
            expected_tokens["label.txt"] = document_token(labels, label_bytes)
            changes["label.txt"] = ("\n".join(names + [new]) + "\n").encode("utf-8")
        # Category-specific sample assignments are part of the recoverable
        # change too. Otherwise the UI's cleanup would discard them forever.
        tree_path = self.store.project / "sample_tree.json"
        if tree_path.exists():
            tree_bytes = tree_path.read_bytes()
            tree = json.loads(tree_bytes.decode("utf-8-sig"))
            tree.setdefault("groups", {}).setdefault(new, [])
            assignments = tree.setdefault("assignments", {})
            for relative in affected:
                if relative in assignments:
                    assignments[relative].pop(old, None)
                    if not assignments[relative]:
                        assignments.pop(relative)
            changes["sample_tree.json"] = json_bytes(tree)
            expected_tokens["sample_tree.json"] = document_token(tree_path, tree_bytes)
        pending = self.store.prepare_transaction(f"将 {old} 改为 {new}", changes,
            {"changes": preview, "affected_images": affected, "image_hashes": guards, "expected_tokens": expected_tokens,
             "canvas_inputs": {p: self.overrides[p] for p in affected if p in self.overrides}})
        return {"pending": {"kind": "transaction", **pending}, "images": len(affected),
                "annotations": sum(p["annotations"] for p in preview)}

    def tool_undo(self, transaction_id):
        return {"pending": {"kind": "transaction", **self.store.prepare_undo(transaction_id)}}

    def tool_snapshot(self, scope="current", title="数据版本"):
        paths = self.original_scope(scope)
        entries = self.version_entries(paths)
        entry_id, folder, manifest = self.store.create("snapshot", str(title)[:100],
                                                     {"scope": scope, "full_project": set(paths) == self.known})
        write_json(folder / "snapshot.json", entries)
        return self.store.finish(folder, manifest, count=len(entries))

    def version_entries(self, paths):
        entries = {}
        for relative in paths:
            self.check()
            source = self.store.source("jsons/" + Path(relative).stem + ".json")
            data = source.read_bytes() if source.exists() else None
            doc = None
            error = None
            try:
                doc = self.overrides.get(relative, json.loads(data.decode("utf-8-sig")) if data is not None else None)
                effective = json_bytes(doc) if doc is not None else None
            except (ValueError, UnicodeError) as exc:
                error = str(exc)
                effective = data
            entries[relative] = {"image_hash": file_sha(self.store.source(relative)),
                                 "annotation_hash": document_token(source, effective), "document": doc,
                                 "status_flag": self.context.get("flags", {}).get(relative), "error": error}
        return entries

    def tool_compare(self, before, after="current"):
        folder, baseline = self.store.artifact(before, "snapshot")
        left = read_json(folder / "snapshot.json")
        if after == "current":
            paths = self.paths if baseline.get("metadata", {}).get("full_project", True) else [p for p in left if p in self.known]
            right = self.version_entries(paths)
        else:
            folder2, _ = self.store.artifact(after, "snapshot")
            right = read_json(folder2 / "snapshot.json")
        added, removed = sorted(right.keys() - left.keys()), sorted(left.keys() - right.keys())
        images = [p for p in sorted(left.keys() & right.keys()) if left[p]["image_hash"] != right[p]["image_hash"]]
        labels = [p for p in sorted(left.keys() & right.keys()) if left[p]["annotation_hash"] != right[p]["annotation_hash"]
                  or left[p].get("status_flag") != right[p].get("status_flag")]
        entry_id, output, manifest = self.store.create("comparison", "数据版本对比", {"before": before, "after": after})
        write_json(output / "changes.json", {"added": added, "removed": removed, "image_changed": images, "annotation_changed": labels})
        present = [p for p in dict.fromkeys(added + images + labels) if p in self.known]
        if present:
            self.add_group("版本变化样本", "新增、图片或标注发生变化的当前样本", present)
        return self.store.finish(output, manifest, added=len(added), removed=len(removed),
                                 image_changed=len(images), annotation_changed=len(labels))

    def tool_duplicates(self, scope="current"):
        buckets = defaultdict(list)
        for relative in self.original_scope(scope):
            self.check()
            buckets[file_sha(self.store.source(relative))].append(relative)
        duplicates = [items for items in buckets.values() if len(items) > 1]
        result = self.add_group("完全重复图片", "按原图片文件 SHA256 相同分组；未删除任何图片", [p for b in duplicates for p in b])
        entry_id, folder, manifest = self.store.create("duplicates", "完全重复图片")
        write_json(folder / "duplicates.json", duplicates)
        return {**self.store.finish(folder, manifest, duplicate_sets=len(duplicates)), **result}

    def tool_similar(self, scope="current", reference="current", top_n=20):
        top_n = numeric(top_n, 1, 100, "候选数量", True)
        if reference == 'current' and self.context.get('viewing_artifact'):
            selected = self.original_scope('selected')
            if len(selected) != 1:
                raise ValueError('查找相似图片须明确一张参考照片，请只选中一张或指定reference。')
            reference = selected[0]
        elif reference == 'current':
            reference = self.context.get('current_image')
        elif isinstance(reference, str):
            reference = {self.store.source_key(p): p for p in self.paths}.get(self.store.source_key(reference))
        if reference not in self.known:
            raise ValueError("请先打开作为参考的图片。")

        def fingerprint(relative):
            with Image.open(self.store.source(relative)) as image:
                small = image.convert("L").resize((9, 8))
                values = [small.getpixel((x, y)) for y in range(8) for x in range(9)]
            return sum((values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))

        target = fingerprint(reference)
        ranking = []
        for relative in self.original_scope(scope):
            self.check()
            if relative != reference:
                ranking.append(((target ^ fingerprint(relative)).bit_count(), relative))
        ranking.sort()
        result = self.add_group("外观相似候选", f"灰度差分哈希距离最小的前 {top_n} 张；不是脏污语义判定", [p for _, p in ranking[:top_n]])
        return {**result, "metric": "64位dHash汉明距离", "distances": [d for d, _ in ranking[:top_n]]}

    def tool_convert_dataset(self, resource, format):
        from Utils.AnnotationImporter import inspect_yolo_dataset, parse_json_annotation_file, IMAGE_EXTENSIONS
        source = self.resource(resource, "directory")
        # Do not follow resource-directory links to unrelated locations.
        for branch in (source / "images", source / "labels", source / "jsons"):
            if branch.exists():
                branch.resolve().relative_to(source)
                for path in branch.rglob("*"):
                    self.check()
                    path.resolve().relative_to(source)
        if (source / "data.yaml").exists():
            (source / "data.yaml").resolve().relative_to(source)
        records, issues = [], []
        if format == "yolo":
            inspection = inspect_yolo_dataset(source)
            if inspection["fatal_errors"]:
                raise ValueError("目录检查失败：" + json.dumps(inspection["fatal_errors"], ensure_ascii=False)[:1500])
            for row in inspection["records"]:
                if row["can_import"] and row["label_state"] != "missing":
                    records.append((Path(row["source_image_path"]), row["annotation_json"]))
                else:
                    issues.append({"path": row["source_image_path"], "errors": row["errors"], "state": row["label_state"]})
        elif format == "native":
            images = [p for p in (source / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
            stems = Counter(p.stem.casefold() for p in images)
            for image in sorted(images):
                self.check()
                if stems[image.stem.casefold()] > 1:
                    issues.append({"path": str(image), "errors": ["同名图片对应关系不明确，已跳过。"]})
                    continue
                image.resolve().relative_to(source)
                annotation = (source / "jsons" / (image.stem + ".json")).resolve()
                annotation.relative_to(source)
                parsed = parse_json_annotation_file(annotation)
                if parsed["can_import"]:
                    records.append((image, parsed["annotation_json"]))
                else:
                    issues.append({"path": str(image), "errors": parsed["errors"]})
        else:
            raise ValueError("支持YOLO目录或含images/jsons的原生目录。")
        if not records:
            raise ValueError("没有可对应的有效图片与标注；不会猜测文件对应关系。")
        entry_id, folder, manifest = self.store.create("converted", "整理外部数据", {"resource": resource, "format": format})
        (folder / "images").mkdir()
        (folder / "jsons").mkdir()
        sources = []
        for i, (image, doc) in enumerate(records):
            self.check()
            image.resolve().relative_to(source)
            stem = f"{i + 1:06d}_" + image.stem
            with Image.open(image) as im:
                doc["image_width"], doc["image_height"] = im.size
            for ann in doc["annotations"]:
                annotation_features(ann, doc["image_width"], doc["image_height"])
            expected = file_sha(image)
            target = folder / "images" / (stem + image.suffix.lower())
            shutil.copyfile(image, target)
            if file_sha(target) != expected:
                raise ValueError("整理期间源图片变化，产物未完成。")
            write_json(folder / "jsons" / (stem + ".json"), doc)
            sources.append({"source": str(image), "image": "images/" + target.name, "sha256": expected})
        write_json(folder / "sources.json", sources)
        write_json(folder / "issues.json", issues)
        return self.store.finish(folder, manifest, count=len(records), skipped=len(issues),
                                 note="按原有对应关系整理，未自动猜测错名标签。原目录未修改。")

    def tool_preannotate(self, model, scope="current", conf=.5, imgsz=None, class_map=None):
        weights = self.resource(model, "model")
        if weights.suffix.lower() != ".pt" or not weights.is_file():
            raise ValueError("请选择本地.pt模型。")
        conf = numeric(conf, .001, 1, "置信度")
        imgsz = numeric(imgsz, 32, 4096, "输入尺寸", True) if imgsz is not None else None
        if class_map is not None and (not isinstance(class_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in class_map.items())):
            raise ValueError("类别映射必须是名称对应表。")
        paths = self.original_scope(scope)
        if not paths:
            raise ValueError("范围内没有图片。")
        entry_id, folder, manifest = self.store.create("prediction_plan", "本地模型候选预标注",
            {"model_resource": model, "model_path": str(weights), "model_sha256": file_sha(weights),
             "conf": conf, "imgsz": imgsz, "class_map": class_map or {}, "paths": paths,
             "classes": self.context.get("classes", []), "overrides": self.overrides})
        self.store.finish(folder, manifest, count=len(paths))
        return {"pending": {"kind": "preannotate", "artifact_id": entry_id, "title": "运行本地模型，生成候选标注",
                            "model": model_label(manifest['metadata']), "model_path": str(weights),
                            "count": len(paths), "conf": conf, "imgsz": imgsz if imgsz is not None else "跟随模型训练尺寸",
                            "note": "只生成候选，已有正式标注不会被覆盖。"}}

    def tool_training_runs(self, resource=None, analyze=True):
        self.training_analysis = None
        self.prediction_analysis = None
        if resource in self.resources and self.resources[resource].get('kind') == 'model':
            raise ValueError('该资源是推理模型，不是训练结果目录。分析它已完成的推理结果请调用 '
                             'prediction_results(model="' + resource + '")；不要改查当前工程 results.csv。'
                             '若用户明确要求训练曲线，需另选关联的训练结果目录。')
        from Utils.AITrainingAnalysis import discover_runs, load_run
        from Utils.TrainingReport import training_relations
        if resource is None:
            candidates = [key for key, value in resource_catalog(self.resources).items()
                          if value.get('role') == 'training_results']
            if len(candidates) > 1:
                raise ValueError("已有多个训练结果目录，请明确resource，不能猜选：" + resource_choices(self.resources, 'directory'))
            resource = candidates[0] if candidates else 'project'
        folders = (discover_runs(self.store.project, Path(__file__).resolve().parents[1]) if resource == 'project'
                   else training_folders(self.resource(resource, "directory")))
        if not folders:
            raise ValueError("未找到可读取的训练记录（results.csv）。检查范围：" +
                             ('当前工程' if resource == 'project' else resource + '：' + self.resources[resource]['name']) +
                             "。已添加目录：" + (resource_choices(self.resources, 'directory') or '无') +
                             "。目录资源与本次生成的产物ID不同；请选对已有资源或核对其中的results.csv。")
        if type(analyze) is not bool:
            raise ValueError('analyze必须是布尔值。')
        results, failures, loaded = [], [], []
        for folder in folders[:40]:
            self.check()
            try:
                run = load_run(folder)
                loaded.append(run)
                results.append({"resource": resource, "folder": str(folder), "name": run["name"], "best": run["best"], "last": run["last"],
                                "args": run["args"], "elapsed": run["elapsed"], "warnings": run["warnings"],
                                "context": run["context"], "primary_metric": run["primary"],
                                "epochs_recorded": len(run["rows"]), "trend": run["trend"],
                                "relations": training_relations(run),
                                "per_class": run['record'].get('per_class'),
                                "chart_files": [c['filename'] for c in run['charts']],
                                "validation_fingerprint": run["record"].get("validation_fingerprint")})
            except Exception as exc:
                failures.append({"folder": str(folder), "error": str(exc)})
        if not results:
            raise ValueError("训练目录已定位，但记录读取失败：" + json.dumps(failures, ensure_ascii=False)[:1400])
        entry_id, output, manifest = self.store.create("training_records", "训练记录", {"resource": resource})
        write_json(output / "training.json", results)
        write_json(output / "errors.json", failures)
        charts = []
        if analyze and len(loaded) == 1:
            from Utils.AIChatTraining import prepare
            self.training_analysis = prepare(output, loaded[0])
            charts = self.training_analysis['charts']
        return {**self.store.finish(output, manifest, count=len(results), failed=len(failures)), "runs": results,
                "errors": failures, "charts": charts,
                "note": ('已读取训练指标并准备图表；原图可在左侧临时结果核对。' if charts else
                         '训练指标已读取；没有可用图表时依据现有记录分析。' if len(loaded) == 1 else
                         '已列出训练记录；逐类分析请指定一个训练结果目录。')}

    def tool_evaluate(self, candidate_id, iou=.5):
        from Utils.AICandidates import candidate_evaluation
        return candidate_evaluation(self, candidate_id, iou)

    def tool_prediction_results(self, candidate_id=None, model=None, offset=0, limit=20):
        from Utils.AIPredictionAnalysis import prediction_results
        return prediction_results(self, candidate_id, model, offset, limit)

    def tool_inspect_predictions(self, candidate_id=None, model=None, scope='sample', image_ids=None, limit=None, with_reference=False):
        from Utils.AIPredictionVision import prepare
        return prepare(self, candidate_id, model, scope, image_ids, limit, with_reference)

    def tool_training_compare(self, current, previous):
        from Utils.AITrainingAnalysis import load_run, compare_runs
        left = load_run(self.resource(current, "directory"))
        right = load_run(self.resource(previous, "directory"))
        result = compare_runs(left, right)
        entry_id, folder, manifest = self.store.create("training_comparison", "训练记录对比")
        write_json(folder / "comparison.json", result)
        return {**self.store.finish(folder, manifest), "comparison": result}

    def tool_training_plan(self, parameter, values):
        if parameter not in {"imgsz", "epochs", "batch"} or not isinstance(values, list) or not 1 <= len(values) <= 5:
            raise ValueError("每次选择一个参数，最多5个实验值。")
        limits = {"imgsz": (32, 4096), "epochs": (1, 10000), "batch": (1, 1024)}
        values = [numeric(v, *limits[parameter], parameter, True) for v in values]
        config = copy.deepcopy(self.context.get("training_config", {}))
        if not all(config.get(k) for k in ("data", "weight", "task", "epochs", "imgsz", "batch")):
            raise ValueError("请先在训练页填写一份完整基础配置。")
        variants = [{**config, parameter: v} for v in dict.fromkeys(values)]
        entry_id, folder, manifest = self.store.create("training_plan", "训练实验配置", {"parameter": parameter})
        write_json(folder / "configs.json", variants)
        return self.store.finish(folder, manifest, count=len(variants), configs=variants,
                                 note="尚未启动训练。在工作记录中选择配置填入训练页后，由用户启动。")

    def tool_history(self):
        return {kind: self.store.list_entries(kind) for kind in ("artifacts", "transactions", "runs", "workflows")}

    def tool_view_result(self, artifact_id):
        _, manifest = self.store.artifact(artifact_id)
        if manifest["kind"] not in VISUAL_KINDS:
            raise ValueError("此产物没有可展示的图片。")
        if manifest['kind'] == 'training_records':
            from Utils.AIResultBrowser import visual_rows
            visual_rows(self.store, artifact_id)
        self.export_result = artifact_id
        return {"artifact_id": artifact_id, "kind": manifest["kind"], "title": manifest["title"],
                **{key: manifest[key] for key in ("source_count", "splits", "format") if key in manifest},
                "count": manifest.get("count", 0), "note": "已在软件左侧临时结果中打开图片，不需要重新生成。"}

    def tool_save_workflow(self, title):
        from Utils.AIWorkspace import identifier
        reusable = [step for step in self.steps if step["tool"] not in {"history", "save_workflow", "run_workflow", "undo"}
                    and not step["result"].get("pending")]
        if not reusable:
            reusable = self.context.get("reusable_steps", [])
        if not reusable:
            raise ValueError("当前还没有可以保存的成功步骤。")
        if any(step["tool"] not in TOOL_SPECS or step["tool"] in {"save_workflow", "run_workflow", "undo"} for step in reusable):
            raise ValueError("这组步骤不适合直接复用。")
        steps = []
        for step in reusable[:8]:
            args = copy.deepcopy(step["args"])
            if "scope" in args and (args["scope"] == "selected" or args["scope"].startswith("group:")):
                args["scope"] = "current"
            if step["tool"] == "select":
                args.pop("group_id", None)
                if "groups[" in args.get("code", ""):
                    raise ValueError("筛选代码引用了临时分组，请先改成对当前工程的可复用条件。")
            steps.append({"tool": step["tool"], "args": args})
        entry_id = identifier()
        folder = self.store.location("workflows", entry_id)
        bindings = {step["args"][key]: self.resources[step["args"][key]] for step in steps for key in ("resource", "model", "current", "previous")
                    if step["args"].get(key) in self.resources}
        write_json(folder / "workflow.json", {"id": entry_id, "kind": "workflow", "title": str(title)[:100],
                                                "created_at": stamp(), "status": "saved", "steps": steps, "resource_bindings": bindings})
        return {"workflow_id": entry_id, "steps": steps, "title": str(title)[:100]}

    def tool_run_workflow(self, workflow_id):
        workflow = read_json(self.store.location("workflows", workflow_id) / "workflow.json")
        if not 1 <= len(workflow["steps"]) <= 8 or any(s["tool"] in {"run_workflow", "save_workflow", "undo"} for s in workflow["steps"]):
            raise ValueError("保存步骤无效或包含递归调用。")
        for key, bound in workflow.get("resource_bindings", {}).items():
            current = self.resources.get(key)
            if not current or Path(current["path"]).resolve() != Path(bound["path"]).resolve():
                raise ValueError("常用步骤的外部资源未绑定到原目录/模型，请重新添加并核对：" + bound["name"])
        results = []
        for step in workflow["steps"]:
            result = self.execute(step["tool"], step["args"])
            results.append(result)
            if result.get("pending"):
                return {"pending": result["pending"], "completed_steps": len(results) - 1,
                        "remaining_steps": workflow["steps"][len(results):]}
        return {"workflow_id": workflow_id, "completed_steps": len(results), "results": results}
