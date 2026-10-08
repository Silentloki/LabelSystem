"""Route-2 work assistant: bounded tools, previews and confirmation/resume UI."""
import copy
import os
import time
import sqlite3
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QLineEdit, QPlainTextEdit, QPushButton,
                             QLabel, QDialog, QSpinBox, QFileDialog, QMenu, QMessageBox, QDialogButtonBox,
                             QFrame, QStackedWidget, QListWidget)

from Utils.AIAugment import DEFAULT_QWEN_URL, DEFAULT_QWEN_MODEL
from Utils.AIWorkAgent import run_agent, MAX_REQUESTS, DEFAULT_REQUESTS
from Utils.AIChatHistory import ConversationMixin
from Utils.AIChatView import CHAT_STYLE, ChatInput, ConversationTranscript
from Utils.AIConversations import result_card


class ChatWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, project, paths, overrides, groups, history, message, current_filter, api, parent=None, context=None):
        super().__init__(parent)
        self.args = (project, paths, overrides, groups, history, message, current_filter, api)
        self.context = context or {}

    def run(self):
        started = time.monotonic()
        project, paths, overrides, groups, history, message, current_filter, api = self.args
        try:
            payload = run_agent(project, paths, overrides, groups, history, message, current_filter, api,
                                self.context, self.isInterruptionRequested, self.progress.emit)
            payload["elapsed"] = time.monotonic() - started
            self.completed.emit(payload)
        except Exception as exc:
            # Never echo the credential, including in provider error responses.
            self.failed.emit(str(exc).replace(api[0], "[API Key 已隐藏]")[:2000])
        finally:
            self.args = None


class AIChatWindow(ConversationMixin, QMainWindow):
    def __init__(self, host):
        super().__init__(host, Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint |
                         Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("AI 助手")
        self.setObjectName("ai_chat_window")
        self.host = host
        self.worker = None
        self.history = []
        self.details = []
        self.stopped = False
        self.resources = {}
        self.pending_operation = None
        self.reusable_steps = []
        self.ui_lock_states = None
        self.continuation = None
        self.scheduled_resume = None
        self.resume_queued = None
        body = QWidget()
        body.setObjectName("aiChatBody")
        body.setStyleSheet(CHAT_STYLE)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(9)
        header = QHBoxLayout()
        name = QLabel("AI 助手")
        name.setObjectName("chatTitle")
        header.addWidget(name, 1)
        self.history_button = QPushButton("历史")
        self.history_button.setCheckable(True)
        self.clear_button = QPushButton("新对话")
        self.settings_button = QPushButton("设置")
        self.settings_button.setCheckable(True)
        for button in (self.history_button, self.clear_button, self.settings_button):
            button.setObjectName("chatQuiet")
            header.addWidget(button)
        layout.addLayout(header)
        self.conversation_title = QLabel("新对话")
        self.conversation_title.setWordWrap(True)
        self.conversation_title.setObjectName("chatMuted")
        layout.addWidget(self.conversation_title)
        self.settings_panel = QGroupBox("连接设置")
        form = QFormLayout(self.settings_panel)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("填写 API Key")
        self.endpoint = QLineEdit(DEFAULT_QWEN_URL)
        self.endpoint.setCursorPosition(0)
        self.model = QLineEdit(DEFAULT_QWEN_MODEL)
        form.addRow("API Key", self.api_key)
        form.addRow("接口地址", self.endpoint)
        form.addRow("模型", self.model)
        self.max_calls = QSpinBox()
        self.max_calls.setRange(1, MAX_REQUESTS)
        self.max_calls.setValue(DEFAULT_REQUESTS)
        self.max_calls.setToolTip("每次发言最多请求模型的次数；每次请求都可能计费。工具结果返回后，模型可以选择下一步。")
        form.addRow("最多请求", self.max_calls)
        disclosure = QLabel("Key 只在本窗口内存中使用。发送统计、近期对话和工具结果；看图分析会发送选中样本及结果图，训练分析附带统计图。按实际请求计费。")
        disclosure.setWordWrap(True)
        disclosure.setObjectName("chatMuted")
        form.addRow(disclosure)
        self.settings_panel.hide()
        layout.addWidget(self.settings_panel)
        self.key_hint = QPushButton("填写 API Key，开始工作 →")
        self.key_hint.setObjectName("chatQuiet")
        layout.addWidget(self.key_hint)
        self.pages = QStackedWidget()
        self.chat_page = QWidget()
        chat_layout = QVBoxLayout(self.chat_page)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        nav = QHBoxLayout()
        project_name = QLabel(os.path.basename(os.path.normpath(self.host.project_dir)))
        project_name.setObjectName("chatMuted")
        nav.addWidget(project_name, 1)
        self.overview_button = QPushButton("工作概览")
        self.overview_button.setObjectName("chatQuiet")
        nav.addWidget(self.overview_button)
        chat_layout.addLayout(nav)
        self.overview_panel = QWidget()
        overview_layout = QVBoxLayout(self.overview_panel)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        self.overview_list = QListWidget()
        self.overview_list.setWordWrap(True)
        self.overview_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.overview_list.setMaximumHeight(170)
        overview_layout.addWidget(self.overview_list)
        row = QHBoxLayout()
        self.work_history_button = QPushButton("全部产物与旧记录")
        row.addWidget(self.work_history_button)
        overview_layout.addLayout(row)
        self.overview_panel.hide()
        chat_layout.addWidget(self.overview_panel)
        self.transcript = ConversationTranscript()
        chat_layout.addWidget(self.transcript, 1)
        self.pages.addWidget(self.chat_page)
        self.history_page = QWidget()
        history_layout = QVBoxLayout(self.history_page)
        history_layout.setContentsMargins(0, 6, 0, 0)
        history_layout.addWidget(QLabel("项目对话"))
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索对话名称或消息内容")
        history_layout.addWidget(self.history_search)
        self.conversation_list = QListWidget()
        self.conversation_list.setWordWrap(True)
        self.conversation_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        history_layout.addWidget(self.conversation_list, 1)
        back = QPushButton("返回当前对话")
        history_layout.addWidget(back)
        self.pages.addWidget(self.history_page)
        layout.addWidget(self.pages, 1)
        self.pending_panel = QFrame()
        self.pending_panel.setObjectName("chatPending")
        pending_layout = QVBoxLayout(self.pending_panel)
        self.pending_label = QLabel()
        self.pending_label.setWordWrap(True)
        pending_layout.addWidget(self.pending_label)
        pending_row = QHBoxLayout()
        self.approve_button = QPushButton("查看方案并执行")
        self.discard_button = QPushButton("取消方案")
        pending_row.addWidget(self.approve_button)
        pending_row.addWidget(self.discard_button)
        pending_layout.addLayout(pending_row)
        self.pending_panel.hide()
        layout.addWidget(self.pending_panel)
        self.scope_label = QLabel()
        self.scope_label.setObjectName("chatScope")
        self.scope_label.setWordWrap(True)
        layout.addWidget(self.scope_label)
        self.resource_label = QLabel()
        self.resource_label.setObjectName("chatMuted")
        self.resource_label.setWordWrap(True)
        layout.addWidget(self.resource_label)
        composer = QFrame()
        composer.setObjectName("chatComposer")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(10, 8, 10, 8)
        self.input = ChatInput()
        self.input.setObjectName("chatInput")
        self.input.setPlaceholderText("选中照片后，告诉我你想做什么…\n例如：分析这些图 / 记为漏检 / 加入问题样本集")
        self.input.setFixedHeight(104)
        composer_layout.addWidget(self.input)
        row = QHBoxLayout()
        self.resource_button = QPushButton("＋ 添加目录 / 模型")
        self.resource_button.setObjectName("chatQuiet")
        row.addWidget(self.resource_button)
        self.project_context_button = QPushButton("项目状态")
        self.project_context_button.setObjectName("chatQuiet")
        self.project_context_button.clicked.connect(self.host.show_project_context)
        row.addWidget(self.project_context_button)
        row.addStretch(1)
        self.send = QPushButton("发送")
        self.send.setObjectName("chatSend")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.hide()
        row.addWidget(self.stop_button)
        row.addWidget(self.send)
        composer_layout.addLayout(row)
        layout.addWidget(composer)
        self.status = QLabel("对话随项目保存 · Ctrl+Enter 发送")
        self.status.setObjectName("chatMuted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.save_error = QLabel()
        self.save_error.setWordWrap(True)
        self.save_error.setStyleSheet("color: #B45309;")
        self.save_error.hide()
        layout.addWidget(self.save_error)
        self.setCentralWidget(body)
        self.setMinimumWidth(560)
        available = host.screen().availableGeometry()
        self.resize(min(760, available.width() - 40), min(960, available.height() - 60))
        self.move(available.x() + (available.width() - self.width()) // 2,
                  available.y() + (available.height() - self.height()) // 2)
        self.send.clicked.connect(self.submit)
        self.stop_button.clicked.connect(self.stop)
        self.clear_button.clicked.connect(self.clear_history)
        self.resource_button.clicked.connect(self.add_resource)
        self.work_history_button.clicked.connect(self.show_work_history)
        self.approve_button.clicked.connect(self.approve_pending)
        self.discard_button.clicked.connect(self.discard_pending)
        self.input.submitted.connect(self.submit)
        self.settings_button.toggled.connect(self.settings_panel.setVisible)
        self.key_hint.clicked.connect(lambda: self.settings_button.setChecked(True))
        self.api_key.textChanged.connect(lambda text: self.key_hint.setVisible(not bool(text.strip())))
        self.history_button.clicked.connect(self.show_conversations)
        back.clicked.connect(self.show_conversations)
        self.history_search.textChanged.connect(self.populate_conversations)
        self.conversation_list.itemClicked.connect(self.open_conversation)
        self.overview_button.clicked.connect(self.show_overview)
        self.overview_list.itemClicked.connect(self.jump_to_message)
        self.transcript.action.connect(self.open_chat_result)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(400)
        self.save_timer.timeout.connect(self.save_conversation)
        self.input.textChanged.connect(lambda: self.save_timer.start())
        self.init_conversation()
        self.host.sampleTree.currentItemChanged.connect(lambda: QTimer.singleShot(0, self.refresh_scope))
        self.host.listWidget.itemSelectionChanged.connect(self.refresh_scope)
        self.host.listWidget.currentRowChanged.connect(lambda _: self.refresh_scope())

    def running(self):
        return bool(self.resume_queued) or (self.worker is not None and self.worker.isRunning())

    def submit(self, resume=None):
        if not isinstance(resume, dict):
            resume = None
        if self.running():
            return
        message = resume["request"] if resume else self.input.toPlainText().strip()
        if not message:
            return
        if len(message) > 6000:
            self.status.setText("本次消息过长，请控制在 6000 字以内。")
            return
        api = (self.api_key.text().strip(), self.endpoint.text().strip(), self.model.text().strip())
        if not all(api):
            self.status.setText("请先填写 API Key、接口地址和模型。")
            self.settings_button.setChecked(True)
            return
        if self.missing_view:
            self.status.setText("上次处理对象已不可用，请先在左侧选择这次要处理的图片。")
            return
        if not self.host.relative_paths:
            self.status.setText("请先导入图片。")
            return
        # Current canvas participates without forcing a disk write. Reuse only
        # a loaded canvas corresponding to this exact image.
        overrides = {}
        item = self.host.listWidget.currentItem()
        if item and self.host.current_tree_filter[0] != "artifact":
            canvas = self.host.graphicsView.imageItem
            size = canvas.pixmap().size()
            expected_path = self.host.resolve_image_path(item.toolTip())
            canvas_path = getattr(canvas, "path", "")
            if (not size.isEmpty() and canvas_path and expected_path
                    and os.path.normcase(os.path.abspath(canvas_path)) == os.path.normcase(os.path.abspath(expected_path))):
                overrides[item.toolTip()] = {"image_width": size.width(), "image_height": size.height(),
                                             "annotations": [a.to_dict() for a in canvas.annotations]}
        self.stopped = False
        self.pages.setCurrentWidget(self.chat_page)
        self.history_button.setChecked(False)
        self.pending_message = message
        self.start_revision = self.host.temporary_selection_revision
        context = self.work_context()
        context['_inspection_secrets'] = tuple(self.session_secrets)
        if resume:
            context["resume"] = resume
        self.worker = ChatWorker(self.host.project_dir, list(self.host.relative_paths), overrides,
                                 copy.deepcopy(self.host.temporary_selections), copy.deepcopy(self.history),
                                 message, self.host.current_tree_filter, api, self, context=context)
        self.worker.progress.connect(self.on_progress)
        self.worker.completed.connect(self.apply_result)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished)
        if not resume:
            self.say("你", message)
            self.input.clear()
            self.start_task()
        else:
            self.update_task("已完成确认的操作，正在继续后续步骤…", "running")
        self.settings_button.setChecked(False)
        self.send.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.history_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_button.show()
        self.set_pending(None)
        self.lock_project(True)
        self.worker.start()

    def on_progress(self, text):
        if not self.stopped:
            self.status.setText(text)
            self.update_task(text)

    def stop(self):
        if self.resume_queued:
            self.resume_queued = None
            self.stopped = True
            self.status.setText("已停止后续步骤，已完成结果保留。")
        if self.worker is not None and self.worker.isRunning():
            self.stopped = True
            self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.status.setText("已请求停止；当前请求或图片处理结束后停止后续步骤。已开始保存的修改会完整结束或恢复。")

    def apply_result(self, payload):
        if "local_work" in payload:
            self.apply_local_result(payload)
            return
        if "work" in payload:
            self.apply_work_result(payload)
            return
        if self.stopped:
            return
        result = payload["result"]
        self.details.append(copy.deepcopy(payload))
        if self.host.temporary_selection_revision != self.start_revision and result["action"] != "reply":
            self.failed("等待期间临时分组发生变化，本次结果未应用；请重新发送。")
            return
        try:
            if result["action"] == "select":
                group_id = self.host.upsert_temporary_selection(
                    result["title"], result["criteria"], result["paths"], result["code"], result.get("group_id"))
                count = len(self.host.temporary_selections[group_id]["paths"])
                text = f"已展示“{result['title']}”：{count} 张。\n条件：{result['criteria']}"
                if not count:
                    text += "\n没有符合条件的图片，可以继续调整条件。"
                self.details[-1]["applied_group_id"] = group_id
                self.reusable_steps = [{"tool": "select", "args": {k: result[k] for k in ("title", "criteria", "code")}, "result": {"group_id": group_id}}]
                # Stable ID plus executed code are preserved in group context.
            elif result["action"] == "remove":
                title = self.host.temporary_selections[result["group_id"]]["title"]
                self.host.remove_temporary_selection(result["group_id"])
                text = f"已移除临时分组“{title}”，原图片和标注仍保留。"
            else:
                text = result["message"]
            summary = payload["summary"]
            if result["action"] == "select":
                text += f"\n本次读取 {summary['readable_images']}/{summary['total_images']} 张的标注。"
                if summary["skipped_images"]:
                    text += f" {summary['skipped_images']} 张缺少或无法读取有效数据，未参与筛选；详情见执行记录。"
                text += "\n这是本次筛选快照；修改标注后可让我重新筛选。"
            cards = []
            if result["action"] == "select":
                cards = [result_card({"group_id": group_id, "title": result["title"], "count": count})]
            self.say("助手", text, cards)
            self.history.extend([{"role": "user", "content": self.pending_message}, {"role": "assistant", "content": text}])
            self.history = self.history[-12:]
            tokens = payload["usage"].get("total_tokens", "未提供")
            self.status.setText(f"完成 · {payload['elapsed']:.1f} 秒 · tokens：{tokens}")
            self.update_task("已完成", "completed", payload)
            self.save_conversation()
        except Exception as exc:
            self.failed(str(exc))

    def failed(self, message):
        if not self.stopped:
            self.say("助手", "本次未完成：" + message)
            if getattr(self, "pending_message", None):
                self.history.extend([{"role": "user", "content": self.pending_message},
                                     {"role": "assistant", "content": "本次未完成：" + message}])
                self.history = self.history[-12:]
            self.status.setText("未应用结果，可以修改需求后重试。")
            self.update_task("本次未完成 · 已完成结果保留", "failed")
            self.save_conversation()

    def finished(self):
        if self.stopped:
            self.say("助手", "本次已停止；此前完成的产物仍可在工作记录中查看。")
            self.status.setText("已停止")
            self.update_task("已停止 · 已完成结果保留", "stopped")
        self.send.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.history_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_button.hide()
        self.lock_project(False)
        self.approve_button.setEnabled(True)
        self.discard_button.setEnabled(True)
        worker = self.worker
        self.worker = None
        if worker:
            worker.deleteLater()
        continuation = self.scheduled_resume
        self.scheduled_resume = None
        if continuation and not self.stopped:
            self.resume_queued = continuation
            QTimer.singleShot(0, self.resume_next)
        self.refresh_scope()
        self.save_conversation()

    def resume_next(self):
        continuation = self.resume_queued
        self.resume_queued = None
        if continuation and not self.stopped:
            self.submit(continuation)

    def work_context(self):
        def number(edit):
            try:
                return int(edit.text())
            except ValueError:
                return None
        from Utils.AIResultActions import ui_target
        target = ui_target(self.host)
        viewing = target['artifact_id']
        target_paths = [] if target['invalid'] else target['images']
        return {"current_paths": self.host.get_paths_for_current_tree_filter(),
                "interaction_target": target,
                "viewing_artifact": viewing,
                "current_result_count": self.host.result_browser.entries.get(viewing, {}).get("count", 0) if viewing else 0,
                "selected_paths": [] if viewing else target_paths,
                "selected_result_paths": target_paths if viewing else [],
                "current_image": self.host.listWidget.currentItem().toolTip() if self.host.listWidget.currentItem() and not viewing else None,
                "classes": list(self.host.graphicsView.imageItem.existing_categories),
                "flags": dict(zip(self.host.relative_paths, self.host.flag)),
                "sample_assignments": copy.deepcopy(self.host.sample_assignments),
                "sample_groups": copy.deepcopy(self.host.sample_groups),
                "resources": copy.deepcopy(self.resources), "max_calls": self.max_calls.value(),
                "current_inference_resource_id": getattr(self.host, 'current_inference_resource_id', None),
                "reusable_steps": copy.deepcopy(self.reusable_steps),
                "training_config": {"data": self.host.dataEdit.text().strip(), "weight": self.host.comboBox_2.currentText(),
                                    "task": self.host.comboBox.currentText(), "epochs": number(self.host.epochEdit),
                                    "imgsz": number(self.host.sizeEdit), "batch": number(self.host.batchEdit)}}

    def lock_project(self, locked):
        widgets = (self.host.centralWidget(), self.host.toolBar, self.host.menuBar())
        if locked and self.ui_lock_states is None:
            self.ui_lock_states = [w.isEnabled() for w in widgets]
            for widget in widgets:
                widget.setEnabled(False)
        elif not locked and self.ui_lock_states is not None:
            for widget, state in zip(widgets, self.ui_lock_states):
                widget.setEnabled(state)
            self.ui_lock_states = None
        for button in (self.resource_button, self.work_history_button):
            button.setEnabled(not locked)

    def add_resource(self):
        menu = QMenu(self)
        directory = menu.addAction("添加数据或训练结果目录")
        model = menu.addAction("添加本地模型 (.pt)")
        action = menu.exec_(self.resource_button.mapToGlobal(self.resource_button.rect().bottomLeft()))
        if action == directory:
            path = QFileDialog.getExistingDirectory(self, "添加需要助手读取的目录")
            kind = "directory"
        elif action == model:
            path, _ = QFileDialog.getOpenFileName(self, "添加训练好的本地模型", "", "模型 (*.pt)")
            kind = "model"
        else:
            return
        if path:
            self.register_resource(path, kind)

    def register_resource(self, path, kind):
        from pathlib import Path
        from Utils.AIResources import resource_catalog
        number = 1
        while 'r' + str(number) in self.resources:
            number += 1
        key = 'r' + str(number)
        self.resources[key] = {"path": str(Path(path).resolve()), "kind": kind, "name": Path(path).name}
        catalog = resource_catalog(self.resources)
        self.resource_label.setText("外部资源：" + "；".join(
            f"{k} {v['name']}" + ("（训练结果）" if v.get('role') == 'training_results' else "")
            for k, v in catalog.items()))
        info = catalog[key]
        self.resources[key]["role"] = info.get("role")
        from Utils.ProjectContext import ProjectContext
        try:
            ProjectContext(self.host.project_dir).register_resource(path, kind, info.get('role'))
        except (OSError, ValueError, sqlite3.Error) as exc:
            self.status.setText("项目资源尚未保存：" + str(exc))
        from Utils.AIModelIdentity import model_source
        display_name = model_source(path) if kind == 'model' else Path(path).name
        text = f"已添加 {key}：{display_name}。" + ("已识别训练结果，可以直接说‘分析这次训练结果’。"
                if info.get('role') == 'training_results' else "可以在对话中指定使用它。")
        self.say("助手", text)
        self.history.append({"role": "assistant", "content": text})
        self.history = self.history[-12:]
        self.refresh_resource_label()
        self.save_conversation()
        return key

    def show_work_history(self):
        from Utils.AIWorkDialogs import WorkHistoryDialog
        WorkHistoryDialog(self).exec_()

    def set_pending(self, pending):
        previous = self.pending_operation
        if previous and pending is None:
            from Utils.AIWorkspace import Workspace, read_json, write_json
            store = Workspace(self.host.project_dir)
            if previous.get("run_id"):
                path = store.location("runs", previous["run_id"]) / "run.json"
                record = read_json(path)
                if record["status"] == "awaiting_confirmation":
                    record["status"] = "cancelled"
                    write_json(path, record)
            if previous.get("transaction_id"):
                path = store.location("transactions", previous["transaction_id"]) / "transaction.json"
                record = read_json(path)
                if record["status"] == "prepared":
                    record["status"] = "discarded"
                    write_json(path, record)
        self.pending_operation = pending
        if pending is None:
            self.continuation = None
        self.approve_button.setVisible(bool(pending))
        self.discard_button.setVisible(bool(pending))
        self.pending_panel.setVisible(bool(pending))
        if pending:
            count = pending.get("count", pending.get("files", 0))
            self.pending_label.setText(f"待执行：{pending['title']}（{count} 项）。请查看具体方案。")
        else:
            self.pending_label.clear()

    def discard_pending(self):
        if self.running() or not self.pending_operation:
            return
        self.set_pending(None)
        self.update_task("方案已取消 · 未执行这项修改", "cancelled")
        self.say("助手", "这项方案已取消，此前完成的结果保留。")
        self.history.append({"role": "assistant", "content": "用户已取消待确认方案，不继续执行。"})
        self.history = self.history[-12:]
        self.save_conversation()

    def closeEvent(self, event):
        self.save_timer.stop()
        if not self.save_conversation():
            event.ignore()
            return
        super().closeEvent(event)

    def approve_pending(self):
        if self.running() or not self.pending_operation:
            return
        from Utils.AIWorkspace import Workspace, read_json
        pending = self.pending_operation
        detail = pending
        if pending["kind"] == "transaction":
            store = Workspace(self.host.project_dir)
            transaction = read_json(store.location("transactions", pending["transaction_id"]) / "transaction.json")
            detail = {"title": transaction["title"], "files": len(transaction["entries"]),
                      "changes": transaction["metadata"].get("changes", []),
                      "note": "保存修改前内容；源文件有新变化时拒绝覆盖。"}
        dialog = QDialog(self)
        dialog.setWindowTitle("执行前查看具体方案")
        dialog.resize(760, 570)
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        lines = [detail["title"], ""]
        if pending["kind"] == "transaction":
            lines.append(f"涉及 {len(detail['changes'])} 项改动；会保存修改前内容，供后续恢复。")
            for change in detail["changes"]:
                text = change.get("image", change.get("path", "")) + "："
                if "old" in change:
                    text += f"{change['old']} → {change['new']}（{change['annotations']} 处标注）"
                else:
                    text += change.get("action", "修改")
                lines.append(text)
            lines.append("\n源文件或当前画布在预览后有新改动时，将停止执行，避免覆盖。")
        else:
            lines.extend(["本地模型：" + detail["model"], f"图片数量：{detail['count']} 张",
                          f"置信度：{detail['conf']}；输入尺寸：{detail['imgsz']}", detail["note"]])
            if detail.get('model_path'):
                lines.append('模型来源：' + detail['model_path'])
        view.setPlainText("\n".join(lines))
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("执行此方案")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            self.launch_local(pending)

    def launch_local(self, operation):
        if self.worker is not None:
            return
        from Utils.AIWorkDialogs import LocalWorkWorker, ensure_canvas_matches
        try:
            ensure_canvas_matches(self.host, operation)
        except Exception as exc:
            self.failed(str(exc))
            return
        self.stopped = False
        self.worker = LocalWorkWorker(self.host.project_dir, operation, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.completed.connect(self.apply_result)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished)
        self.send.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.approve_button.setEnabled(False)
        self.discard_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_button.show()
        self.history_button.setEnabled(False)
        if self.current_task_id is None or self.pending_operation is None:
            self.start_task("正在执行已确认的本地操作…")
        else:
            self.update_task("正在执行已确认的本地操作…", "running")
        self.lock_project(True)
        self.worker.start()

    def adopt_candidates(self, artifact_id, paths):
        self.launch_local({"kind": "adopt", "artifact_id": artifact_id, "paths": paths,
                           "previous_flags": {p: self.host.flag[self.host.get_global_index_from_path(p)] for p in paths}})

    def apply_local_result(self, payload):
        from Utils.AIWorkspace import Workspace, read_json, write_json, stamp
        result = payload["local_work"]
        was_stopped = self.stopped
        self.details.append(payload)
        if result.get("status") == "committed":
            self.stopped = False
            self.host.reload_after_ai_changes(result)
            text = f"已完成：{result['title']}。修改前的内容已保存，可以从工作概览中的全部产物与旧记录准备恢复。"
        else:
            shown = self.host.result_browser.open(result["artifact_id"])
            text = (f"本地模型候选已生成：完成 {result['count']} 张，失败 {result['failed']} 张；"
                    f"预测 {result['predicted_annotations']} 处。" +
                    ("已在左侧“临时结果”打开，可逐张查看预测框、类别和置信度，并切换原标注对照。" if shown else "请在工作记录查看结果。"))
        card = result_card(result)
        self.say("助手", text, [card] if card else None)
        self.update_task("本地操作已完成", "completed")
        self.history.append({"role": "assistant", "content": text})
        self.history = self.history[-12:]
        continuation = copy.deepcopy(self.continuation)
        if continuation:
            # Replace the pending observation with the authoritative local result
            # before asking the model what remains to do. Same overall budget.
            continuation["observations"][-1]["status"] = "completed"
            continuation["observations"][-1]["result"] = result
            for step in continuation["steps"]:
                if step["result"].get("pending"):
                    step["result"] = result
            continuation["elapsed"] = continuation.get("elapsed", 0) + payload.get("elapsed", 0)
            if result.get("artifact_id"):
                continuation["last_artifact"] = result["artifact_id"]
            if not was_stopped and (continuation["requests"] < continuation["max_calls"] or continuation.get("workflow_tail")):
                self.scheduled_resume = continuation
        if self.pending_operation and self.pending_operation.get("run_id"):
            store = Workspace(self.host.project_dir)
            path = store.location("runs", self.pending_operation["run_id"]) / "run.json"
            record = read_json(path)
            record.update(status="completed", local_result=result, finished_at=stamp())
            write_json(path, record)
        self.set_pending(None)
        self.status.setText("本地操作完成（未请求千问）")
        self.save_conversation()

    def apply_work_result(self, payload):
        self.details.append(copy.deepcopy(payload))
        work = payload["work"]
        if not self.stopped:
            if self.host.temporary_selection_revision != self.start_revision and work["effects"]:
                self.failed("执行期间临时分组改变，未应用新的视图；已生成产物保留在工作记录中。")
                return
            for effect in work["effects"]:
                if effect["kind"] == "group":
                    group_id = effect["group_id"]
                    # New IDs were created in the worker; register once then update.
                    if group_id not in self.host.temporary_selections:
                        self.host.temporary_selections[group_id] = {"paths": [], "title": effect["title"], "criteria": effect["criteria"]}
                    self.host.upsert_temporary_selection(effect["title"], effect["criteria"], effect["paths"], effect.get("code", ""), group_id)
                elif effect["kind"] == "remove_group":
                    self.host.remove_temporary_selection(effect["group_id"])
            pending = work.get("pending")
            if pending:
                pending = {**pending, "run_id": payload["run_id"]}
            self.set_pending(pending)
            if pending:
                self.continuation = copy.deepcopy(work["continuation"])
        cards = []
        seen = set()
        from Utils.AIResultBrowser import VISUAL_KINDS
        for step in work["steps"]:
            result = step["result"]
            card = result_card(result)
            if card and (card["type"], card["id"]) not in seen:
                cards.append(card)
                seen.add((card["type"], card["id"]))
            if (not self.stopped and result.get("artifact_id") and
                    (result.get("kind") in VISUAL_KINDS or step["tool"] == "view_result")):
                self.host.result_browser.open(result["artifact_id"])
        text = payload["result"].get("message") or ("方案已准备好，请核对后执行。" if work.get("pending") else "本次处理结果如下。")
        self.say("助手", text, cards)
        context_text = text + "\n" + "\n".join(c["title"] + "：" + c["summary"] for c in cards)
        self.history.extend([{"role": "user", "content": self.pending_message}, {"role": "assistant", "content": context_text}])
        self.history = self.history[-12:]
        self.reusable_steps = [s for s in work["steps"] if not s["result"].get("pending")]
        tokens = payload["usage"].get("total_tokens", "未提供")
        status = {"completed": "完成", "partial": "部分完成", "failed": "失败", "stopped": "已停止",
                  "awaiting_confirmation": "待确认", "limit_reached": "达到请求上限"}.get(work["status"], work["status"])
        self.status.setText(f"{status} · Ctrl+Enter 继续发送")
        self.update_task(status, work["status"], payload)
        self.refresh_scope()
        self.save_conversation()
