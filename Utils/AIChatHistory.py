"""Persistence, history navigation and result actions for the chat dock."""
import copy
import sqlite3
from pathlib import Path

from PyQt5.QtCore import Qt, QSize, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QListWidgetItem, QFrame, QVBoxLayout, QLabel

from Utils.AIConversations import ConversationStore
from Utils.AIChatView import display_time
from Utils.AIWorkspace import identifier, stamp, read_json


class ConversationMixin:
    def init_conversation(self):
        self.conversation_store = ConversationStore(self.host.project_dir)
        self.current_task_id = None
        self.missing_view = False
        self.view_notice = ""
        self.session_secrets = set()
        self.restore_warning = ""
        try:
            value = self.conversation_store.current()
        except (OSError, ValueError, TypeError) as exc:
            value = self.conversation_store.new()
            self.restore_warning = "旧对话暂时无法读取，原文件保留：" + str(exc)
        self.opening_from_result = bool(self.host.result_browser.active_id)
        self.activate_conversation(value)
        self.opening_from_result = False

    def save_conversation(self):
        if not hasattr(self, "conversation"):
            return True
        value = self.conversation
        value["context"] = copy.deepcopy(self.history[-12:])
        value["resources"] = self.conversation_store.encode_resources(self.resources)
        value["draft"] = self.input.toPlainText()
        if not self.missing_view:
            value["view"] = list(self.host.current_tree_filter)
        try:
            from Utils.AIWorkAgent import redact
            secret = self.api_key.text().strip()
            if secret:
                self.session_secrets.add(secret)
            clean = value
            for key in sorted(self.session_secrets, key=len, reverse=True):
                clean = redact(clean, key)
            self.conversation_store.save(clean)
            self.save_error.setText(self.restore_warning)
            self.save_error.setVisible(bool(self.restore_warning))
            return True
        except (ValueError, OSError, TypeError) as exc:
            self.save_error.setText("对话尚未保存，请检查项目目录权限或磁盘空间。" + str(exc))
            self.save_error.show()
            return False

    def activate_conversation(self, value, anchor=None):
        self.conversation = value
        self.current_task_id = None
        self.missing_view = False
        self.view_notice = ""
        self.history = copy.deepcopy(value.get("context", []))[-12:]
        self.resources = self.conversation_store.decode_resources(value.get("resources", {}))
        try:
            from Utils.ProjectContext import ProjectContext
            self.resources = ProjectContext(self.host.project_dir).merge_resources(self.resources)
        except (OSError, ValueError, sqlite3.Error) as exc:
            self.restore_warning = "项目资源暂时无法读取，当前对话资源保留：" + str(exc)
        self.details = []
        self.reusable_steps = []
        self.input.blockSignals(True)
        self.input.setPlainText(value.get("draft", ""))
        self.input.blockSignals(False)
        for message in value["messages"]:
            if message.get("status") in {"running", "awaiting_confirmation"}:
                message.update(status="interrupted", text="上次工作未继续执行。已完成的结果保留；待确认方案请在全部产物中重新核对。")
        view = value.get("view")
        if isinstance(view, list) and len(view) == 3 and not getattr(self, 'opening_from_result', False):
            if view[0] == "temporary" and view[1] in self.host.temporary_selections:
                self.host.current_tree_filter = tuple(view)
                self.host.refresh_image_list(preferred_row=0)
            elif view[0] == "artifact":
                try:
                    _, manifest = self.conversation_store.workspace.artifact(view[1])
                    if manifest.get("hidden"):
                        self.missing_view = True
                        self.view_notice = "上次结果已隐藏 · 点击结果卡片恢复，或在左侧选择图片"
                    elif not self.host.result_browser.open(view[1]):
                        self.missing_view = True
                except (ValueError, OSError):
                    self.missing_view = True
            elif view[0] == "temporary":
                self.missing_view = True
            # Formal filters are intentionally taken from the visible editor;
            # merely opening a chat must not switch the underlying image.
        self.conversation_title.setText(value.get("title", "新对话"))
        self.refresh_resource_label()
        self.refresh_scope()
        self.transcript.visible_count = 80
        self.render_conversation(anchor=anchor, bottom=True)
        self.pages.setCurrentWidget(self.chat_page)
        self.history_button.setChecked(False)
        self.status.setText("历史已恢复 · 可以接着聊" if value["messages"] else "对话随项目保存")
        self.save_conversation()

    def say(self, who, text, cards=None, run_id=None):
        from Utils.AIWorkAgent import redact
        text = redact(str(text), self.api_key.text().strip())
        role = "user" if who == "你" else "assistant"
        message = {"id": identifier(), "role": role, "text": text, "created_at": stamp()}
        if cards:
            message["cards"] = cards
        if run_id:
            message["run_id"] = run_id
        self.conversation["messages"].append(message)
        if role == "user" and self.conversation["title"] == "新对话":
            self.conversation["title"] = text.replace("\n", " ")[:32]
            self.conversation_title.setText(self.conversation["title"])
        self.conversation["updated_at"] = stamp()
        self.render_conversation(bottom=True if role == 'user' else None)
        self.save_conversation()
        return message["id"]

    def start_task(self, text="正在理解需求…"):
        message = {"id": identifier(), "role": "task", "text": text, "created_at": stamp(), "status": "running"}
        self.current_task_id = message["id"]
        self.conversation["messages"].append(message)
        self.render_conversation()
        self.save_conversation()

    def update_task(self, text, status=None, payload=None):
        message = next((m for m in self.conversation["messages"] if m["id"] == self.current_task_id), None)
        if message is None:
            return
        message["text"] = text
        if status:
            message["status"] = status
        if payload:
            if payload.get("run_id"):
                message["run_id"] = payload["run_id"]
            message["metrics"] = (f"{payload.get('elapsed', 0):.1f} 秒 · {payload.get('requests', 0)} 次请求" +
                                  (f" · {payload['usage']['total_tokens']:,} tokens" if isinstance(payload.get('usage', {}).get('total_tokens'), int) else ""))
        self.transcript.update_task(message["id"], text)
        if status or payload:
            self.render_conversation()
            self.save_conversation()

    def result_states(self):
        result = {}
        store = self.conversation_store.workspace
        for message in self.conversation["messages"]:
            for card in message.get("cards", []):
                key = (card["type"], card["id"])
                if key in result:
                    continue
                if card["type"] == "group":
                    exists = card["id"] in self.host.temporary_selections
                    result[key] = ("筛选快照" if exists else "分组已移除", exists)
                else:
                    try:
                        manifest = read_json(store.location("artifacts", card["id"]) / "manifest.json")
                        state = manifest.get("status")
                        if state == "completed":
                            if manifest.get('export_category'):
                                from Utils.AIExportStorage import export_root
                                if not export_root(store, manifest).is_dir():
                                    raise ValueError('导出文件夹已缺失。')
                            if manifest.get('kind') == 'selection':
                                from Utils.AIResultSelection import selection_source
                                selection_source(store, card['id'])
                            if manifest.get('kind') == 'error_set':
                                for parent in manifest.get('metadata', {}).get('source_artifacts', []):
                                    store.artifact(parent)
                            result[key] = ("已隐藏" if manifest.get("hidden") else "可查看", True)
                        else:
                            result[key] = ({"deleted": "结果已删除", "delete_failed": "清理未完成", "deleting": "清理未完成"}.get(state, "结果未完成"), False)
                    except (OSError, ValueError, TypeError):
                        result[key] = ("结果文件缺失", False)
        return result

    def render_conversation(self, anchor=None, bottom=None):
        self.transcript.render(self.conversation["messages"], self.result_states(), anchor, bottom)

    def refresh_resource_label(self):
        from Utils.AIModelIdentity import model_source
        names = []
        for key, r in self.resources.items():
            available = Path(r["path"]).exists()
            kind = "训练结果" if r.get("role") == "training_results" else "模型" if r.get("kind") == "model" else "目录"
            name = model_source(r['path']) if r.get('kind') == 'model' else r.get('name', '资源')
            names.append(key + ' ' + name + "（" + (kind if available else "不可用") + "）")
        self.resource_label.setText("已添加 · " + "  /  ".join(names[:3]) + (f" 等 {len(names)} 项" if len(names) > 3 else ""))
        self.resource_label.setVisible(bool(names))
        self.resource_label.setToolTip("\n".join(r["path"] for r in self.resources.values()))

    def refresh_scope(self):
        if self.missing_view:
            self.scope_label.setText(self.view_notice or "上次处理对象已不可用 · 请先在左侧重新选择")
            return
        from Utils.AIResultActions import ui_target
        target = ui_target(self.host)
        kind, key, child = self.host.current_tree_filter
        if kind == "artifact":
            manifest = self.host.result_browser.entries.get(key, {})
            text = manifest.get("title", "处理结果") + f" · {manifest.get('count', 0)} 项"
        else:
            count = self.host.listWidget.count()
            name = (self.host.temporary_selections.get(key, {}).get("title", "筛选结果") if kind == "temporary"
                    else child if kind == "child" else key if kind in {"tag", "status"} else "全部原图")
            if kind == "status":
                name = {"complete_good": "完全良品", "overkill_good": "过杀品", "bad": "坏品", "unlabeled": "未标注"}.get(key, key)
            text = f"{name} · {count} 张"
        if target['invalid']:
            choice = '选中项含不可用图片，请重新选择'
        elif target['images']:
            choice = (f"已选中 {len(target['images'])} 张" if target['mode'] == 'selected'
                      else '当前查看 1 张')
            if len(target['images']) == 1:
                choice += ' · ' + Path(target['images'][0]).name
        else:
            choice = '尚未选择照片'
        self.scope_label.setText(choice + '\n来源 · ' + text)
        self.scope_label.setToolTip('“这张/这些图”按上述对象处理；Ctrl/Shift可多选，无选中项时使用正在查看的那张。明确说“整批/这组”才使用整个范围。\n' +
                                   '\n'.join(target['images'][:30]))

    def clear_history(self):
        # Kept as a compatibility method; the UI action is now New conversation.
        if not self.can_switch_conversation() or not self.save_conversation():
            return
        value = self.conversation_store.new()
        value["resources"] = self.conversation_store.encode_resources(self.resources)
        self.activate_conversation(value)
        self.input.setFocus()

    def can_switch_conversation(self):
        if self.running():
            return False
        if self.pending_operation:
            self.status.setText("当前还有待确认方案，请执行或取消后再切换对话。")
            return False
        return True

    def show_conversations(self):
        if not self.can_switch_conversation():
            self.history_button.setChecked(False)
            return
        if self.pages.currentWidget() == self.history_page:
            self.pages.setCurrentWidget(self.chat_page)
            self.history_button.setChecked(False)
            return
        if not self.save_conversation():
            return
        self.populate_conversations()
        self.pages.setCurrentWidget(self.history_page)
        self.history_button.setChecked(True)

    def populate_conversations(self):
        self.conversation_list.clear()
        for row in self.conversation_store.list(self.history_search.text().strip()):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, row)
            item.setToolTip(row["title"] + "\n" + row["snippet"][:500])
            item.setSizeHint(QSize(0, 156))
            self.conversation_list.addItem(item)
            frame = QFrame()
            frame.setObjectName("chatHistoryCard")
            frame.setProperty("current", row["id"] == self.conversation["id"])
            frame.setAttribute(Qt.WA_TransparentForMouseEvents)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(5)
            title = QLabel(row["title"])
            title.setTextFormat(Qt.PlainText)
            title.setWordWrap(True)
            title.setStyleSheet("font-size: 12pt; font-weight: 600;")
            layout.addWidget(title)
            date = QLabel("最近更新 · " + display_time(row["updated_at"]))
            date.setWordWrap(True)
            date.setObjectName("chatMuted")
            layout.addWidget(date)
            snippet = QLabel(row["snippet"].replace("\n", " ")[:48])
            snippet.setTextFormat(Qt.PlainText)
            snippet.setWordWrap(True)
            snippet.setObjectName("chatMuted")
            layout.addWidget(snippet)
            self.conversation_list.setItemWidget(item, frame)

    def open_conversation(self, item):
        if not self.can_switch_conversation() or not self.save_conversation():
            return
        row = item.data(Qt.UserRole)
        try:
            self.activate_conversation(self.conversation_store.read(row["id"]), row.get("anchor"))
        except (OSError, ValueError, TypeError) as exc:
            self.status.setText("无法读取这段对话：" + str(exc))

    def show_overview(self):
        self.overview_list.clear()
        for message in self.conversation["messages"]:
            if message.get("role") == "user" or message.get("cards"):
                title = message["text"].replace("\n", " ")[:65] if message["role"] == "user" else " / ".join(c["title"] for c in message["cards"])
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, message["id"])
                self.overview_list.addItem(item)
        self.overview_panel.setVisible(not self.overview_panel.isVisible())

    def jump_to_message(self, item):
        self.overview_panel.hide()
        self.render_conversation(anchor=item.data(Qt.UserRole), bottom=False)

    def open_chat_result(self, kind, entry_id):
        if kind == 'context':
            from Utils.AIContextInspectorDialog import ContextInspectorDialog
            ContextInspectorDialog(self.host.project_dir, entry_id, self).exec_()
            return
        if kind == 'folder':
            try:
                folder, manifest = self.conversation_store.workspace.artifact(entry_id)
                if manifest.get('export_category'):
                    from Utils.AIExportStorage import export_root
                    folder = export_root(self.conversation_store.workspace, manifest)
                    if not folder.is_dir():
                        raise ValueError('导出文件夹已缺失。')
                elif manifest['kind'] == 'dataset':
                    from Utils.AIResultBrowser import artifact_file
                    for name in ('dataset_AI', 'yolo'):
                        candidate = artifact_file(folder, name)
                        if candidate.is_dir():
                            folder = candidate
                            break
                if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
                    self.status.setText('无法打开文件夹：' + str(folder))
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))
            return
        if self.running():
            self.status.setText("任务进行中，请结束后再切换处理对象。")
            return
        if kind == "group" and entry_id in self.host.temporary_selections:
            self.host.current_tree_filter = ("temporary", entry_id, None)
            self.host.refresh_image_list()
            self.missing_view = False
        elif kind == "artifact":
            from Utils.AIResultBrowser import VISUAL_KINDS
            try:
                _, manifest = self.conversation_store.workspace.artifact(entry_id)
                if manifest["kind"] in VISUAL_KINDS:
                    self.host.result_browser.open(entry_id)
                    self.missing_view = False
                else:
                    self.show_work_history()
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))
        else:
            self.status.setText("这组结果已移除，请重新筛选。")
        self.refresh_scope()
        self.render_conversation(bottom=False)
        self.save_conversation()
