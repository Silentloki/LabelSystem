"""Visual artifact adapters and an isolated, read-only result view in the editor.

The original annotation canvas/list stay intact behind two stacked widgets.
Result navigation never invokes the editor's autosave or changes project paths.
"""
from pathlib import Path

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, QSignalBlocker
from PyQt5.QtGui import QColor, QPen, QPolygonF, QPixmap, QFont, QPainter
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QCheckBox, QPushButton, QListWidget, QListWidgetItem,
                             QStackedWidget, QGraphicsView, QGraphicsScene,
                             QGraphicsItem, QTreeWidgetItem, QToolButton, QMessageBox, QAbstractItemView, QAction)

from Utils.AIWorkspace import Workspace, read_json, file_sha
from Utils.AIResultStorage import VISUAL_KINDS, set_result_hidden, deletion_preview, delete_result
from Utils.AIModelIdentity import model_label, model_details, recorded_model


def artifact_file(folder, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("结果文件必须位于本次产物目录中。")
    path = (folder / relative).resolve()
    path.relative_to(folder.resolve())
    return path


def visual_rows(store, artifact_id):
    folder, manifest = store.artifact(artifact_id)
    kind = manifest["kind"]
    if kind not in VISUAL_KINDS:
        raise ValueError("此记录没有可展示的图片结果。")
    if kind == 'error_set':
        from Utils.AIResultActions import error_set_rows
        return manifest, error_set_rows(store, artifact_id)
    if kind == 'selection':
        from Utils.AIResultSelection import selection_source
        parent, images = selection_source(store, artifact_id)
        _, parent_rows = visual_rows(store, parent)
        by_path = {store.source_key(str(Path(row['image']).relative_to(store.project))): row for row in parent_rows if row.get('image')}
        if any(store.source_key(path) not in by_path for path in images):
            raise ValueError('筛选引用的结果图片已缺失。')
        return manifest, [by_path[store.source_key(path)] for path in images]
    if kind == 'training_records':
        chart_index = artifact_file(folder, 'charts.json')
        charts = read_json(chart_index) if chart_index.exists() else []
        if not charts:
            raise ValueError('此训练记录没有已保存的图表，请让助手重新读取训练目录。')
        rows = [{'image': str(artifact_file(folder, chart['image'])), 'name': chart['name'],
                 'source': chart['source'], 'hash': chart['sha256'],
                 'document': {'annotations': []}, 'chart': True} for chart in charts]
    elif kind == "candidates":
        rows = [{"image": str(store.source(r["path"])), "name": Path(r["path"]).name,
                 "source": r["path"], "document": r["document"],
                 "original": r["source_document"], "hash": r["image_sha256"],
                 "ignored_classes": r.get('ignored_classes', 0), "unmapped_classes": r.get('unmapped_classes', []),
                 "predictions": len(r["document"]["annotations"])}
                for r in read_json(artifact_file(folder, "candidates.json"))]
        for error in read_json(artifact_file(folder, "errors.json")):
            rows.append({"name": Path(error["path"]).name, "source": error["path"],
                         "error": error["error"], "predictions": None})
    else:
        rows = []
        for r in read_json(artifact_file(folder, "sources.json")):
            from Utils.AIExportStorage import result_file
            image = result_file(store, folder, manifest, r, 'image')
            annotation = result_file(store, folder, manifest, r, 'json')
            rows.append({"image": str(image), "name": image.name, "source": r.get("source", ""),
                         "annotation": str(annotation), "original": None,
                         "original_source": r.get('original_source', r.get('source')),
                         "split_group": r.get('split_group') or (r.get('image_hash') if kind == 'crops' else None)})
    return manifest, rows


class ResultView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#202733"))
        self.image_rect = QRectF()
        self._auto_fit = True
        self._fitting = False

    def fit(self):
        self._auto_fit = True
        if not self.image_rect.isEmpty() and not self._fitting:
            self._fitting = True
            try:
                self.fitInView(self.image_rect, Qt.KeepAspectRatio)
            finally:
                self._fitting = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta and not self.image_rect.isEmpty():
            # Scrollbars appearing during scale() resize the viewport. Disable
            # automatic fit first so that resize does not undo this zoom.
            self._auto_fit = False
            factor = 1.2 if delta > 0 else 1 / 1.2
            self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self._fitting:
            self.fit()


class ResultBrowser:
    def __init__(self, host):
        self.host, self.store = host, Workspace(host.project_dir)
        self.entries, self.rows, self.active_id = {}, [], None
        self.action_states, self.right_states = None, None
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setObjectName("aiResultList")
        self.list_stack = QStackedWidget()
        host.left_panel.layout().replaceWidget(host.listWidget, self.list_stack)
        self.list_stack.addWidget(host.listWidget)
        self.list_stack.addWidget(self.list)
        self.page = QWidget()
        self.page.setObjectName("aiResultPage")
        layout = QVBoxLayout(self.page)
        layout.setContentsMargins(4, 0, 4, 0)
        self.heading = QLabel()
        self.heading.setWordWrap(True)
        self.heading.setTextFormat(Qt.PlainText)
        layout.addWidget(self.heading)
        self.controls = QWidget()
        bar = QHBoxLayout(self.controls)
        bar.setContentsMargins(0, 0, 0, 0)
        self.filter = QComboBox()
        self.filter.addItems(["全部图片", "有预测", "无预测", "推理失败"])
        self.original = QCheckBox("对照原标注（红色）")
        self.overlay = QCheckBox("显示结果标注（绿色）")
        self.overlay.setChecked(True)
        fit = QPushButton("适应窗口")
        self.adopt = QPushButton("核对并采用候选")
        self.feedback_button = QPushButton("记录问题")
        self.feedback_button.setToolTip("记录选中图片的人工判断，关联当前批次，不修改标注。")
        for w in (self.filter, self.original, self.overlay, fit, self.adopt, self.feedback_button):
            bar.addWidget(w)
        layout.addWidget(self.controls)
        self.view = ResultView()
        self.view.setToolTip("只读预览：滚轮缩放，拖动平移；右键可适应窗口或记录问题。")
        self.fit_action = QAction("适应窗口", self.page)
        self.fit_action.triggered.connect(self.view.fit)
        self.feedback_action = QAction("记录问题", self.page)
        self.feedback_action.triggered.connect(self.record_feedback)
        self.view.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.view.addActions([self.fit_action, self.feedback_action])
        self.list.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.list.addAction(self.feedback_action)
        self.ask_ai_action = QAction("让 AI 处理选中图片…", self.page)
        self.ask_ai_action.triggered.connect(self.ask_ai)
        self.list.addAction(self.ask_ai_action)
        self.view.addAction(self.ask_ai_action)
        layout.addWidget(self.view, 1)
        self.caption = QLabel()
        self.caption.setTextFormat(Qt.PlainText)
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.view_stack = QStackedWidget()
        host.horizontalLayout_2.replaceWidget(host.graphicsView, self.view_stack)
        self.view_stack.addWidget(host.graphicsView)
        self.view_stack.addWidget(self.page)
        host.horizontalLayout_2.setStretch(1, 4)
        self.list.currentRowChanged.connect(self.show_row)
        self.list.currentRowChanged.connect(lambda _: self.host.notify_ai_context_changed())
        self.list.itemSelectionChanged.connect(self.host.notify_ai_context_changed)
        self.filter.currentIndexChanged.connect(self.populate)
        self.original.toggled.connect(lambda: self.show_row(self.list.currentRow()))
        self.overlay.toggled.connect(lambda: self.show_row(self.list.currentRow()))
        fit.clicked.connect(self.view.fit)
        self.adopt.clicked.connect(self.adopt_candidates)
        self.feedback_button.clicked.connect(self.record_feedback)
        # Restore existing artifacts without any API calls or new inference.
        for entry in self.store.list_entries(limit=100):
            if entry["kind"] in VISUAL_KINDS and entry["status"] == "completed" and not entry.get("hidden"):
                self.register(entry["id"])

    def register(self, artifact_id):
        try:
            _, manifest = self.store.artifact(artifact_id)
            if manifest["kind"] not in VISUAL_KINDS:
                return False
            if manifest['kind'] == 'training_records':
                visual_rows(self.store, artifact_id)
            self.entries[artifact_id] = manifest
            return True
        except (ValueError, OSError, KeyError):
            return False

    def title(self, manifest):
        title = ("模型推理 · " + model_label(recorded_model(self.store, manifest))
                 if manifest["kind"] == "candidates" else manifest["title"])
        return title + " · " + manifest.get("created_at", "")[5:19]

    def add_tree_nodes(self, root):
        for artifact_id, manifest in sorted(self.entries.items(), key=lambda row: (row[1].get('created_at', ''), row[0]), reverse=True):
            item = QTreeWidgetItem([f"{self.title(manifest)} ({manifest.get('count', 0)})"])
            item.setData(0, Qt.UserRole, ("artifact", artifact_id, None))
            item.setToolTip(0, "点击查看处理后的图片与标注；不增加正式类别。\n" + self.title(manifest))
            if manifest['kind'] == 'candidates':
                item.setToolTip(0, self.title(manifest) + '\n' + model_details(recorded_model(self.store, manifest)))
            root.addChild(item)
            close = QToolButton(self.host.sampleTree)
            close.setText("×")
            close.setFixedWidth(25)
            close.setToolTip("隐藏结果：保留文件，可从“工作记录与产物”重新打开")
            close.clicked.connect(lambda _, aid=artifact_id: self.hide(aid))
            self.host.sampleTree.setItemWidget(item, 1, close)

    def open(self, artifact_id):
        if not self.register(artifact_id):
            return False
        try:
            self.entries[artifact_id] = set_result_hidden(self.store, artifact_id, False)
        except (ValueError, OSError) as exc:
            self.entries.pop(artifact_id, None)
            QMessageBox.warning(self.host, "无法打开结果", str(exc))
            return False
        self.active_id = artifact_id
        self.host.timer1.stop()
        self.host.current_tree_filter = ("artifact", artifact_id, None)
        self.host.tabWidget.setCurrentWidget(self.host.labelPage)
        self.list_stack.setCurrentWidget(self.list)
        self.view_stack.setCurrentWidget(self.page)
        self.set_editor_visible(False)
        self.host.rebuild_sample_tree()
        def reveal_entry():
            if self.active_id == artifact_id:
                item = self.host.find_sample_tree_item(("artifact", artifact_id, None))
                if item is not None:
                    self.host.sampleTree.scrollToItem(item)
        QTimer.singleShot(0, reveal_entry)
        try:
            manifest, self.rows = visual_rows(self.store, artifact_id)
            candidates = manifest["kind"] == "candidates"
            self.heading.setToolTip(model_details(recorded_model(self.store, manifest)) if candidates else '')
            self.heading.setVisible(candidates)
            self.controls.setVisible(candidates)
            self.feedback_action.setVisible(manifest['kind'] != 'training_records')
            self.filter.setVisible(candidates)
            self.original.setVisible(candidates)
            self.overlay.setVisible(manifest['kind'] != 'training_records')
            self.adopt.setVisible(candidates)
            self.feedback_button.setVisible(manifest['kind'] != 'training_records')
            self.original.blockSignals(True)
            self.original.setChecked(False)
            self.original.blockSignals(False)
            self.heading.setText(self.title(manifest) + "\n" +
                (f"完成 {manifest['count']} 张 · 预测 {manifest['predicted_annotations']} 处 · 失败 {manifest['failed']} 张。绿色为模型预测，可切换红色原标注作对照。"
                 if candidates else f"训练图表 {len(self.rows)} 张，可核对PR曲线及混淆矩阵。" if manifest['kind'] == 'training_records'
                 else f"处理结果 {len(self.rows)} 张，显示生成后的图片及同步标注。") +
                "\n只读预览 · 滚轮缩放、拖动平移；切换左侧正式分类可继续标注。")
            if candidates:
                metadata = manifest.get('metadata', {})
                self.heading.setText(self.heading.text() + f"\n置信度：{metadata.get('conf', '未记录')} · 输入尺寸：{metadata.get('imgsz', '未记录')}")
                ignored = sum(row.get('ignored_classes', 0) for row in self.rows)
                unmapped = sorted({name for row in self.rows for name in row.get('unmapped_classes', [])})
                if ignored:
                    self.heading.setText(self.heading.text() + f' · 按项目类别过滤 {ignored} 处其他类别预测。')
                if unmapped:
                    self.heading.setText(self.heading.text() + ' · 类别待映射（已显示）：' + '、'.join(unmapped))
            self.filter.blockSignals(True)
            self.filter.setCurrentIndex(0)
            self.filter.blockSignals(False)
            self.populate()
        except Exception as exc:
            self.rows = []
            self.populate()
            self.controls.hide()
            self.heading.show()
            self.feedback_action.setVisible(False)
            self.heading.setText("结果无法读取：" + str(exc))
        return True

    def leave(self):
        self.active_id = None
        self.list_stack.setCurrentWidget(self.host.listWidget)
        self.view_stack.setCurrentWidget(self.host.graphicsView)
        self.set_editor_visible(True)

    def record_feedback(self):
        from Utils.ProjectContextDialog import record_feedback
        images = []
        for item in self.list.selectedItems():
            index = item.data(Qt.UserRole)
            if isinstance(index, int) and 0 <= index < len(self.rows):
                row = self.rows[index]
                if row.get('image') and not row.get('chart') and not row.get('error'):
                    images.append(Path(row['image']).relative_to(self.store.project).as_posix())
        record_feedback(self.host, images, self.active_id)

    def ask_ai(self):
        # Creating a chat may restore its previous view; preserve this explicit
        # user selection when opening from the result viewer.
        aid = self.active_id
        selected = [item.data(Qt.UserRole) for item in self.list.selectedItems()]
        current = self.list.currentItem().data(Qt.UserRole) if self.list.currentItem() else None
        self.host.show_ai_chat()
        if self.active_id != aid and aid:
            self.open(aid)
        from PyQt5.QtCore import QItemSelectionModel
        self.list.clearSelection()
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == current:
                self.list.setCurrentItem(item, QItemSelectionModel.NoUpdate)
            item.setSelected(item.data(Qt.UserRole) in selected)
        self.host.ai_chat_dock.refresh_scope()
        self.host.ai_chat_dock.input.setFocus()

    def set_editor_visible(self, visible):
        self.host.horizontalLayout_2.setStretch(2, 1 if visible else 0)
        actions = [self.host.rect_action, self.host.poly_action, self.host.batch_action,
                   self.host.run_action, self.host.ai_augment_action,
                   self.host.loadimg_action, self.host.import_annotations_action,
                   self.host.import_dataset_action, self.host.export_action]
        widgets = [self.host.verticalLayout.itemAt(i).widget() for i in range(self.host.verticalLayout.count())]
        widgets = [w for w in widgets if w is not None]
        if not visible and self.action_states is None:
            self.action_states = [(a, a.isEnabled()) for a in actions]
            self.right_states = [(w, not w.isHidden()) for w in widgets]
            for a in actions:
                a.setEnabled(False)
            for w in widgets:
                w.hide()
        elif visible and self.action_states is not None:
            for a, enabled in self.action_states:
                a.setEnabled(enabled)
            for w, shown in self.right_states:
                w.setVisible(shown)
            self.action_states = self.right_states = None

    def populate(self, *unused):
        self.list.blockSignals(True)
        self.list.clear()
        selected = 0
        found_prediction = False
        for index, row in enumerate(self.rows):
            mode = self.filter.currentIndex() if not self.filter.isHidden() else 0
            n = row.get("predictions")
            if (mode == 1 and not n) or (mode == 2 and (n != 0 or row.get("error"))) or (mode == 3 and not row.get("error")):
                continue
            suffix = " · 失败" if row.get("error") else (f" · {n}处预测" if n else " · 无预测") if n is not None else ""
            item = QListWidgetItem((suffix.strip(" ·") + " · " if suffix else "") + row["name"])
            item.setData(Qt.UserRole, index)
            item.setToolTip(row["name"] + suffix + "\n来源：" + str(row.get("source", "")))
            self.list.addItem(item)
            if n and not found_prediction:
                selected, found_prediction = self.list.count() - 1, True
        self.list.blockSignals(False)
        if self.list.count():
            self.list.setCurrentRow(selected)
        else:
            self.show_row(-1)

    def show_row(self, index):
        scene = self.view.scene()
        scene.clear()
        self.view.image_rect = QRectF()
        self.view.resetTransform()
        if index < 0 or not self.list.item(index):
            self.caption.setText("当前条件下没有图片。")
            return
        row_index = self.list.item(index).data(Qt.UserRole)
        if not isinstance(row_index, int) or not 0 <= row_index < len(self.rows):
            self.caption.setText("结果已更新，请重新选择图片。")
            return
        row = self.rows[row_index]
        try:
            if row.get("error"):
                raise ValueError("本张推理失败：" + row["error"])
            if row.get("hash") and file_sha(row["image"]) != row["hash"]:
                raise ValueError("原图片在推理后已经改变，不能显示旧预测叠加。")
            pix = QPixmap(row["image"])
            if pix.isNull():
                raise ValueError("图片文件缺失或无法读取。")
            scene.addPixmap(pix)
            self.view.image_rect = QRectF(pix.rect())
            doc = row.get("document") or read_json(row["annotation"])
            candidates = self.entries[self.active_id]['kind'] == 'candidates'
            if candidates and self.original.isChecked() and row.get("original"):
                self.draw(row["original"], pix, "#ff6565", True)
            if not candidates or self.overlay.isChecked():
                self.draw(doc, pix, "#34e88e", False)
            scene.setSceneRect(self.view.image_rect)
            self.view.fit()
            n = len(doc.get("annotations", []))
            self.caption.setText(f"{row['name']} · {pix.width()}×{pix.height()} · " +
                (f"{n}处预测" if n else "未检出目标（不代表已确认为良品）") if "predictions" in row else
                f"{row['name']} · {pix.width()}×{pix.height()} · {n}处结果标注\n来源：{row.get('source', '')}")
            if row.get('chart'):
                self.caption.setText(f"{row['name']} · {pix.width()}×{pix.height()} · 训练图表原图\n来源：{row['source']}")
            if row.get('ignored_classes'):
                self.caption.setText(f"{row['name']} · 项目类别预测 {n} 处；已过滤 {row['ignored_classes']} 处其他类别预测。")
            elif row.get('unmapped_classes'):
                self.caption.setText(self.caption.text() + '\n类别待映射：' + '、'.join(row['unmapped_classes']) + '；已完整显示，不能直接采用。')
        except Exception as exc:
            scene.clear()
            self.caption.setText(row["name"] + "\n" + str(exc))

    def draw(self, doc, pix, color, dashed):
        pen = QPen(QColor(color), 2)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.DashLine)
        for ann in doc.get("annotations", []):
            points = [QPointF(p["x"] * pix.width(), p["y"] * pix.height()) for p in ann["points"]]
            if ann["type"] == "rect":
                self.view.scene().addRect(QRectF(points[0], points[1]).normalized(), pen)
            else:
                self.view.scene().addPolygon(QPolygonF(points), pen)
            name = str(ann.get("lable") or ann.get("label") or ann.get("category", ""))
            if "confidence" in ann:
                name += f" {ann['confidence']:.1%}"
            text = self.view.scene().addSimpleText(("原标注 " if dashed else "") + name, QFont("Microsoft YaHei UI", 10))
            text.setBrush(QColor(color))
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(min(p.x() for p in points), max(0, min(p.y() for p in points) - 22))

    def _remove_entry(self, artifact_id):
        self.entries.pop(artifact_id, None)
        if self.active_id == artifact_id:
            current = self.host.get_item_relative_path(self.host.listWidget.currentItem())
            selected = {self.host.get_item_relative_path(item) for item in self.host.listWidget.selectedItems()}
            self.leave()
            self.rows = []
            self.list.clear()
            self.view.scene().clear()
            self.host.current_tree_filter = ("all", None, None)
            # Refreshing the original list normally autosaves its previous
            # item. Hiding/deleting a preview must not rewrite formal JSON or
            # discard an unsaved original canvas.
            blocker = QSignalBlocker(self.host.listWidget)
            self.host.refresh_image_list(preserve_display=True)
            item = self.host._find_list_item_by_rel_path(current)
            if item is not None:
                self.host.listWidget.setCurrentItem(item)
            for index in range(self.host.listWidget.count()):
                item = self.host.listWidget.item(index)
                item.setSelected(self.host.get_item_relative_path(item) in selected)
            del blocker
        self.host.rebuild_sample_tree()

    def hide(self, artifact_id):
        try:
            set_result_hidden(self.store, artifact_id, True)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self.host, "无法隐藏结果", str(exc))
            return False
        self._remove_entry(artifact_id)
        self.host.notify_ai_context_changed()
        return True

    def remove(self, artifact_id):
        """Compatibility for older callers: removing an entry only hides it."""
        return self.hide(artifact_id)

    def delete(self, artifact_id):
        dock = getattr(self.host, "ai_chat_dock", None)
        if dock is not None and dock.running():
            QMessageBox.information(self.host, "任务进行中", "请等待当前任务结束，再删除处理结果。")
            return False
        try:
            plan = deletion_preview(self.store, artifact_id)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self.host, "无法删除处理结果", str(exc))
            return False
        manifest = plan["manifest"]
        images = sum(Path(f["path"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                     for f in plan["files"])
        contents = {
            "candidates": "本次保存的预测框、类别、置信度及相关记录。推理引用的原图片不会删除。",
            "selection": "本次筛选的引用清单及记录。来源切图、导出图片和对应标注不会删除。",
            "error_set": "本次问题样本集的引用清单和说明。来源图片、预测批次、正式标注和人工问题记录不会删除。",
            "crops": f"本次生成的 {images} 个切图/裁剪图片文件、对应标注及处理记录。",
            "dataset": f"本次导出的数据集及预览文件，包含 {images} 个图片文件、标签、配置和相关记录。",
            "export": "本批次主动导出的文件及相关记录。",
            "converted": f"本次格式整理生成的 {images} 个图片文件、对应标注及处理记录。",
            "training_records": f"本次保存的 {images} 张训练图表副本、分析报告及相关记录。外部训练目录不会删除。",
        }
        box = QMessageBox(self.host)
        box.setWindowTitle("删除处理结果")
        box.setIcon(QMessageBox.Warning)
        box.setTextFormat(Qt.PlainText)
        box.setText("删除“" + self.title(manifest) + "”？")
        box.setInformativeText(contents[manifest["kind"]] +
            f"\n\n共删除 {len(plan['files'])} 个文件，约 {plan['bytes'] / 1024 / 1024:.2f} MB。" +
            "\n保留项目原始图片、正式标注及修改恢复记录；其他批次结果不受影响。" +
            "\n工作记录将标记为“已删除”，无法重新打开。此操作不能撤销。")
        box.setDetailedText("删除范围：" + plan["folder"] +
                            ('\n导出目录：' + plan['export_folder'] if plan.get('export_folder') else '') +
                            "\n\n" + "\n".join(f["path"] for f in plan["files"]))
        confirm = box.addButton("删除处理结果", QMessageBox.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec_()
        if box.clickedButton() != confirm:
            return False
        try:
            delete_result(self.store, artifact_id, plan["token"])
        except (ValueError, OSError) as exc:
            # A partial deletion is not a valid preview. Its tombstone remains
            # in history, where the user can inspect and retry the cleanup.
            try:
                state = read_json(Path(plan["folder"]) / "manifest.json").get("status")
            except (ValueError, OSError):
                state = "delete_failed"
            if state in {"deleting", "delete_failed"}:
                self._remove_entry(artifact_id)
                self.host.notify_ai_context_changed()
            QMessageBox.warning(self.host, "删除未完成", str(exc) + "\n请在工作记录中检查剩余文件，必要时重试清理。")
            return False
        self._remove_entry(artifact_id)
        self.host.notify_ai_context_changed()
        return True

    def adopt_candidates(self):
        from Utils.AIWorkDialogs import CandidateDialog
        dialog = CandidateDialog(self.host.project_dir, self.active_id, self.host)
        if dialog.exec_() == dialog.Accepted:
            self.host.show_ai_chat()
            self.host.ai_chat_dock.adopt_candidates(self.active_id, dialog.chosen)
