"""Review candidates, durable work records and local background operations."""
import base64
import json
import html
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal, QPointF, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QPolygonF, QDesktopServices
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLabel, QPushButton, QSplitter,
                             QAbstractItemView, QMessageBox, QInputDialog, QTextBrowser)

from Utils.AIWorkspace import Workspace, read_json, write_json, identifier, stamp
from Utils.AIResultBrowser import VISUAL_KINDS
from Utils.AIResultStorage import MANAGED_KINDS
from Utils.AIModelIdentity import model_label, recorded_model


class LocalWorkWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, project, operation, parent=None):
        super().__init__(parent)
        self.project, self.operation = project, operation

    def run(self):
        import time
        store = Workspace(self.project)
        run_id = identifier()
        record_path = store.location("runs", run_id) / "run.json"
        record = {"id": run_id, "kind": "local_operation", "title": self.operation.get("title", "采用已查看的模型候选"),
                  "created_at": stamp(), "status": "running", "operation": self.operation, "requests": 0}
        started = time.monotonic()
        try:
            write_json(record_path, record)
            kind = self.operation["kind"]
            if kind == "transaction":
                self.progress.emit("正在保存已确认的修改与恢复记录…")
                result = store.commit(self.operation["transaction_id"], self.isInterruptionRequested)
            elif kind == "preannotate":
                from Utils.AICandidates import generate_candidates
                result = generate_candidates(self.project, self.operation["artifact_id"], self.isInterruptionRequested, self.progress.emit)
            elif kind == "adopt":
                from Utils.AICandidates import prepare_adoption
                pending = prepare_adoption(self.project, self.operation["artifact_id"], self.operation["paths"], self.operation["previous_flags"])
                result = store.commit(pending["transaction_id"], self.isInterruptionRequested)
            else:
                raise ValueError("未知本地操作。")
            record.update(status="completed", result=result, finished_at=stamp(), elapsed=round(time.monotonic() - started, 2))
            try:
                write_json(record_path, record)
            except OSError as exc:
                self.progress.emit("操作已完成，但工作记录保存失败：" + str(exc))
            # Even if stop arrives during atomic commit, report committed work.
            self.completed.emit({"local_work": result, "kind": kind, "elapsed": record["elapsed"]})
        except Exception as exc:
            record.update(status="stopped" if self.isInterruptionRequested() else "failed", error=str(exc), finished_at=stamp())
            try:
                write_json(record_path, record)
            except OSError:
                pass
            self.failed.emit(str(exc))


def visual_document(doc):
    if not doc:
        return []
    return [(ann["type"], ann.get("lable") or ann.get("label") or ann.get("category"),
             [(round(float(p["x"]), 9), round(float(p["y"]), 9)) for p in ann["points"]])
            for ann in doc.get("annotations", [])]


def ensure_canvas_matches(host, operation):
    """A pending transaction must not discard edits still only on the canvas."""
    item = host.listWidget.currentItem()
    if not item:
        return
    rel = item.toolTip()
    canvas = host.graphicsView.imageItem
    current = {"annotations": [a.to_dict() for a in canvas.annotations]}
    store = Workspace(host.project_dir)
    if operation["kind"] == "transaction":
        transaction = read_json(store.location("transactions", operation["transaction_id"]) / "transaction.json")
        if rel not in transaction.get("metadata", {}).get("affected_images", []):
            target = store.source('jsons/' + Path(rel).stem + '.json')
            before = read_json(target) if target.exists() else None
            if visual_document(current) != visual_document(before):
                raise ValueError('当前另一张图片有未保存标注，请先保存，避免修改后刷新界面丢失画布内容。')
            return
        overrides = transaction.get("metadata", {}).get("canvas_inputs", {})
        if rel in overrides:
            before = overrides[rel]
        else:
            target = "jsons/" + Path(rel).stem + ".json"
            entry = next((e for e in transaction["entries"] if e["path"] == target), None)
            if entry is not None:
                before = json.loads(base64.b64decode(entry["before"]).decode("utf-8-sig")) if entry["before"] else None
            else:
                before = read_json(store.source(target)) if store.source(target).exists() else None
        if visual_document(current) != visual_document(before):
            raise ValueError("当前画布在方案生成后有改动，请重新生成方案，避免覆盖尚未保存的标注。")
    elif operation["kind"] == "adopt" and rel in operation["paths"] and current["annotations"]:
        raise ValueError("当前图片已有画布标注，不能采用候选覆盖；请先处理这张图片。")
    elif operation["kind"] == "preannotate":
        _, plan = store.artifact(operation["artifact_id"], "prediction_plan")
        if rel in plan["metadata"]["paths"]:
            before = plan["metadata"].get("overrides", {}).get(rel)
            if before is not None and visual_document(before) != visual_document(current):
                raise ValueError("当前标注在方案生成后改变，请重新生成预标注方案。")


class CandidateDialog(QDialog):
    def __init__(self, project, artifact_id, parent=None):
        super().__init__(parent)
        self.store = Workspace(project)
        self.artifact_id = artifact_id
        folder, manifest = self.store.artifact(artifact_id, "candidates")
        self.rows = read_json(folder / "candidates.json")
        for row in self.rows:
            if row.get('unmapped_classes'):
                row['can_adopt'] = False
        self.chosen = []
        self.setWindowTitle("查看本地模型候选标注")
        self.resize(1150, 780)
        layout = QVBoxLayout(self)
        note = QLabel(f"完成 {manifest['count']} 张，失败 {manifest['failed']} 张。绿色为模型候选；红色为已有标注。\n按项目类别保留预测，其他类别已过滤。仅可采用原标注为空且有项目类别预测的图片；已有人工标注不会被覆盖。")
        note.setWordWrap(True)
        layout.addWidget(note)
        split = QSplitter()
        self.table = QTableWidget(len(self.rows), 4)
        self.table.setHorizontalHeaderLabels(["采用", "图片", "预测数", "原标注"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnWidth(0, 48)
        for i, row in enumerate(self.rows):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable if row["can_adopt"] else Qt.NoItemFlags)
            if row["can_adopt"]:
                check.setCheckState(Qt.Unchecked)
            self.table.setItem(i, 0, check)
            self.table.setItem(i, 1, QTableWidgetItem(row["path"]))
            self.table.setItem(i, 2, QTableWidgetItem(str(len(row["document"]["annotations"]))))
            self.table.setItem(i, 3, QTableWidgetItem("类别待映射" if row.get('unmapped_classes') else "保护已有标注" if row["source_document"]["annotations"] else "空"))
        self.preview = QLabel("选择一张图片查看")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(400, 380)
        split.addWidget(self.table)
        split.addWidget(self.preview)
        split.setSizes([400, 700])
        layout.addWidget(split, 1)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        buttons = QHBoxLayout()
        select_all = QPushButton("全选可采用项")
        accept = QPushButton("采用勾选候选")
        close = QPushButton("关闭")
        buttons.addWidget(select_all)
        buttons.addStretch()
        buttons.addWidget(accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        select_all.clicked.connect(lambda: [self.table.item(i, 0).setCheckState(Qt.Checked) for i, r in enumerate(self.rows) if r["can_adopt"]])
        accept.clicked.connect(self.adopt)
        close.clicked.connect(self.reject)
        self.table.currentCellChanged.connect(lambda row, *unused: self.show_row(row))
        if self.rows:
            self.table.selectRow(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "table"):
            QTimer.singleShot(0, lambda: self.show_row(self.table.currentRow()))

    def show_row(self, index):
        if not 0 <= index < len(self.rows):
            return
        row = self.rows[index]
        pix = QPixmap(str(self.store.source(row["path"])))
        if pix.isNull():
            self.preview.setText("无法读取原图片")
            return
        pix = pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(pix)
        labels = []
        for doc, color in ((row["source_document"], QColor("#ef4444")), (row["document"], QColor("#16a34a"))):
            painter.setPen(QPen(color, 2))
            for ann in doc["annotations"]:
                points = [QPointF(p["x"] * pix.width(), p["y"] * pix.height()) for p in ann["points"]]
                if ann["type"] == "rect":
                    from PyQt5.QtCore import QRectF
                    painter.drawRect(QRectF(points[0], points[1]).normalized())
                else:
                    painter.drawPolygon(QPolygonF(points))
                name = ann.get("lable") or ann.get("label") or ann.get("category")
                score = ann.get("confidence")
                label = str(name) + (f" {score:.2f}" if score is not None else "")
                painter.drawText(points[0], label)
                if score is not None:
                    labels.append(label)
        painter.end()
        self.preview.setPixmap(pix)
        self.caption.setText(row["path"] + "\n候选：" + ("、".join(labels) if labels else "无预测；不会自动标为良品"))
        if row.get('ignored_classes'):
            self.caption.setText(self.caption.text() + f"\n按项目类别过滤 {row['ignored_classes']} 处其他类别预测。")
        elif row.get('unmapped_classes'):
            self.caption.setText(self.caption.text() + '\n类别待映射：' + '、'.join(row['unmapped_classes']))

    def adopt(self):
        paths = [row["path"] for i, row in enumerate(self.rows) if row["can_adopt"] and self.table.item(i, 0).checkState() == Qt.Checked]
        if not paths:
            QMessageBox.information(self, "未选择", "请先核对并勾选要采用的候选。")
            return
        self.chosen = paths
        self.accept()


class WorkHistoryDialog(QDialog):
    KINDS = {"error_set": "问题样本集", "selection": "结果筛选", "candidates": "模型候选", "prediction_plan": "预标注方案", "dataset": "数据集", "crops": "裁剪/切图",
             "snapshot": "数据版本", "comparison": "版本对比", "annotation_change": "标注修改", "agent": "对话任务",
             "workflow": "常用步骤", "duplicates": "重复图片", "converted": "格式整理", "evaluation": "预测对照",
             "training_records": "训练记录", "training_comparison": "训练对比", "training_plan": "实验配置", "local_operation": "本地操作"}
    STATES = {"completed": "完成", "committed": "已保存修改", "prepared": "待执行", "running": "进行中",
              "applying": "保存中（可恢复）", "recovery_required": "待恢复", "failed": "失败", "stopped": "已停止",
              "partial": "部分完成", "cancelled": "已取消", "discarded": "已取消", "rolled_back": "已恢复",
              "awaiting_confirmation": "待确认", "limit_reached": "请求次数用完", "saved": "已保存",
              "deleted": "结果已删除", "deleting": "清理未完成", "delete_failed": "清理未完成"}
    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        self.store = Workspace(dock.host.project_dir)
        self.setWindowTitle("工作记录与产物")
        self.resize(920, 650)
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["时间", "任务 / 产物", "类型", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 175)
        self.table.setColumnWidth(2, 120)
        self.table.setColumnWidth(3, 130)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.entries = []
        layout.addWidget(self.table)
        self.summary = QTextBrowser()
        layout.addWidget(QLabel("结果与说明"))
        layout.addWidget(self.summary, 1)
        row = QHBoxLayout()
        open_button = QPushButton("打开记录目录")
        self.use_button = QPushButton("使用所选结果")
        self.delete_button = QPushButton("删除处理结果…")
        row.addWidget(open_button)
        row.addWidget(self.use_button)
        row.addWidget(self.delete_button)
        layout.addLayout(row)
        self.table.currentCellChanged.connect(self.show_entry)
        open_button.clicked.connect(self.open_folder)
        self.use_button.clicked.connect(self.use_entry)
        self.delete_button.clicked.connect(self.delete_entry)
        self.reload_entries()

    def reload_entries(self):
        selected = self.selected()
        selected_id = selected[1]["id"] if selected else None
        self.entries = [(kind, value) for kind in ("artifacts", "transactions", "runs", "workflows")
                        for value in self.store.list_entries(kind, 60)]
        self.entries.sort(key=lambda e: (e[1].get('created_at') or '', e[1]['id']), reverse=True)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.entries))
        selected_row = 0
        for row, (_, item) in enumerate(self.entries):
            if item["id"] == selected_id:
                selected_row = row
            for col, key in enumerate(("created_at", "title", "kind", "status")):
                value = str(item.get(key) or "")
                if key == 'title' and item.get('kind') == 'candidates':
                    try:
                        _, manifest = self.store.artifact(item['id'], 'candidates')
                        value = '模型推理 · ' + model_label(recorded_model(self.store, manifest))
                    except (OSError, ValueError, KeyError):
                        pass
                value = self.KINDS.get(value, value) if key == "kind" else self.STATES.get(value, value) if key == "status" else value
                if key == "status" and item.get("hidden") and item["status"] == "completed":
                    value += "（已隐藏）"
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.blockSignals(False)
        if self.entries:
            self.table.selectRow(selected_row)
            self.show_entry()
        else:
            self.use_button.setEnabled(False)
            self.delete_button.setEnabled(False)

    def delete_entry(self):
        entry = self.selected()
        if entry and entry[0] == "artifacts":
            self.dock.host.result_browser.delete(entry[1]["id"])
            self.reload_entries()

    def selected(self):
        row = self.table.currentRow()
        return self.entries[row] if 0 <= row < len(self.entries) else None

    def show_entry(self, *unused):
        entry = self.selected()
        if not entry:
            return
        kind, item = entry
        names = {"artifacts": "manifest.json", "runs": "run.json", "transactions": "transaction.json", "workflows": "workflow.json"}
        self.use_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        try:
            value = read_json(self.store.location(kind, item["id"]) / names[kind])
            if kind == "transactions":
                value = {k: v for k, v in value.items() if k != "entries"}
            self.summary.setHtml(self.summary_html(value))
            unavailable = value.get("status") in {"deleted", "deleting", "delete_failed"}
            self.use_button.setEnabled(not unavailable)
            self.delete_button.setEnabled(kind == "artifacts" and value.get("kind") in MANAGED_KINDS and
                                          value.get("status") in {"completed", "deleting", "delete_failed"})
            self.delete_button.setText("重试清理剩余文件…" if value.get("status") in {"deleting", "delete_failed"} else "删除处理结果…")
            title = "在临时结果中查看图片" if item["kind"] in VISUAL_KINDS else "选取配置填入训练页" if item["kind"] == "training_plan" else ("继续查看待执行方案" if item["status"] == "prepared" else "准备恢复修改") if kind == "transactions" else "在对话中引用此记录"
            self.use_button.setText("结果不可用" if unavailable else "重新显示隐藏结果" if value.get("hidden") and item["kind"] in VISUAL_KINDS else title)
            if item['kind'] == 'export' and not unavailable:
                self.use_button.setText('打开导出文件夹')
        except Exception as exc:
            self.summary.setPlainText("记录无法读取：" + str(exc))

    def summary_html(self, value):
        rows = ["任务：" + str(value.get("title", "")), "状态：" + self.STATES.get(value.get("status"), str(value.get("status"))),
                "时间：" + str(value.get("created_at", ""))]
        if value.get("hidden") and value.get("status") == "completed":
            rows.append("此结果已隐藏，文件仍保留。点击“重新显示隐藏结果”可恢复左侧入口。")
        if value.get("status") in {"deleting", "delete_failed"}:
            rows.append("结果清理未完成，部分文件可能已删除，不能继续预览或采用。可重试清理剩余文件。")
        if value.get("deletion_error"):
            rows.append("清理失败原因：" + str(value["deletion_error"]))
        labels = {"count": "图片/项目数量", "source_count": "原图片数量", "failed": "失败图片", "skipped": "跳过数量",
                  "adoptable": "可采用候选", "protected": "已有标注受保护", "predicted_annotations": "模型候选标注数",
                  "added": "新增图片", "removed": "移除图片", "image_changed": "图片内容改变", "annotation_changed": "标注或状态改变",
                  "requests": "模型请求次数", "elapsed": "耗时（秒）", "duplicate_sets": "完全重复组数",
                  "missed": "未匹配标注数", "extra_predictions": "额外预测数"}
        for key, label in labels.items():
            if key in value:
                rows.append(label + "：" + str(value[key]))
        if value.get("splits"):
            rows.append(f"训练集：{value['splits'].get('train', 0)} 张；验证集：{value['splits'].get('val', 0)} 张")
        if value.get("classes"):
            rows.append("类别：" + "、".join(value["classes"]))
        params = value.get("metadata", {})
        if value.get('kind') == 'candidates':
            params = recorded_model(self.store, value)
            rows.append('模型版本：' + model_label(params))
            if params.get('model_path'):
                rows.append('模型来源：' + params['model_path'])
            if params.get('model_sha256'):
                rows.append('模型完整指纹：' + params['model_sha256'])
        for key, label in (("model", "本地模型"), ("conf", "置信度"), ("imgsz", "输入尺寸"), ("label", "裁剪目标"),
                           ("padding", "裁剪边距比例"), ("tile_size", "切图尺寸"), ("overlap", "切图重叠比例")):
            if key in params and params[key] is not None:
                rows.append(label + "：" + str(params[key]))
        for change in params.get("changes", []):
            rows.append(str(change.get("image", change.get("path", ""))) + "：" +
                        (f"{change['old']} → {change['new']}，{change['annotations']} 处" if "old" in change else str(change.get("action", "修改"))))
        for step in value.get("steps", []):
            from Utils.AIWorkTools import TOOL_SPECS
            description = TOOL_SPECS.get(step.get("tool"), (step.get("tool", ""), {}))[0]
            rows.append("步骤：" + description)
        if value.get("note"):
            rows.append(value["note"])
        if value.get("error"):
            rows.append("未完成原因：" + value["error"])
        if value.get("data_yaml"):
            rows.append("数据集配置：" + value["data_yaml"])
        return "".join("<p>" + html.escape(str(line)) + "</p>" for line in rows)

    def open_folder(self):
        entry = self.selected()
        if entry:
            if entry[0] == 'artifacts' and entry[1]['status'] == 'completed':
                self.dock.open_chat_result('folder', entry[1]['id'])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.location(entry[0], entry[1]["id"]))))

    def use_entry(self):
        entry = self.selected()
        if not entry:
            return
        kind, item = entry
        try:
            if kind == "artifacts":
                state = read_json(self.store.location(kind, item["id"]) / "manifest.json").get("status")
                if state in {"deleted", "deleting", "delete_failed"}:
                    raise ValueError("处理结果已删除或清理未完成，不能使用。")
            if item["kind"] in VISUAL_KINDS and item["status"] == "completed":
                if self.dock.host.result_browser.open(item["id"]):
                    self.accept()
            elif item['kind'] == 'export' and item['status'] == 'completed':
                self.dock.open_chat_result('folder', item['id'])
            elif item["kind"] == "training_plan" and item["status"] == "completed":
                folder, _ = self.store.artifact(item["id"], "training_plan")
                configs = read_json(folder / "configs.json")
                labels = [f"方案{i + 1}：epochs={c['epochs']} / imgsz={c['imgsz']} / batch={c['batch']}" for i, c in enumerate(configs)]
                label, ok = QInputDialog.getItem(self, "选取配置", "仅填写参数，不自动启动训练", labels, 0, False)
                if ok:
                    self.dock.host.apply_training_analysis_config(configs[labels.index(label)])
                    self.dock.say("助手", "已将所选实验配置填入训练页，尚未启动训练。")
                    self.accept()
            elif kind == "transactions":
                if item["status"] == "prepared":
                    record = read_json(self.store.location(kind, item["id"]) / "transaction.json")
                    pending = {"transaction_id": item["id"], "title": record["title"], "files": len(record["entries"]),
                               "changes": record["metadata"].get("changes", [])}
                else:
                    pending = self.store.prepare_undo(item["id"])
                self.dock.set_pending({"kind": "transaction", **pending})
                self.accept()
            else:
                self.dock.input.setPlainText("使用这条工作记录 " + item["id"] + "：")
                self.accept()
        except Exception as exc:
            QMessageBox.warning(self, "无法使用", str(exc))
