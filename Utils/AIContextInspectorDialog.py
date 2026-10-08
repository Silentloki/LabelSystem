"""Read-only UI for saved context evidence; opening it never queries an AI."""
import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QSplitter, QTabWidget, QPlainTextEdit, QTreeWidget, QTreeWidgetItem,
    QPushButton, QFileDialog, QMessageBox)

from Utils.AIContextInspector import load_inspection, share_copy, sanitize
from Utils.AIWorkspace import write_json


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


class ContextInspectorDialog(QDialog):
    def __init__(self, project, run_id, parent=None):
        super().__init__(parent)
        self.project, self.run_id = project, run_id
        self.setWindowTitle('上下文检查 · Context Inspector')
        self.resize(1140, 780)
        self.setMinimumSize(760, 520)
        layout = QVBoxLayout(self)
        note = QLabel('按请求查看当时的记录。准备快照与接口提交分开；工具结果只有进入后续请求，才构成发送证据。')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel()
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        split = QSplitter()
        self.requests = QListWidget()
        self.requests.setMinimumWidth(200)
        split.addWidget(self.requests)
        self.tabs = QTabWidget()
        self.overview = self.text_tab('请求概况')
        fields = QSplitter(Qt.Vertical)
        self.fields = QTreeWidget()
        self.fields.setHeaderLabels(['字段', '来源', '快照概况'])
        self.fields.setColumnWidth(0, 210)
        self.fields.setColumnWidth(1, 300)
        self.field_value = QPlainTextEdit()
        self.field_value.setReadOnly(True)
        fields.addWidget(self.fields)
        fields.addWidget(self.field_value)
        fields.setSizes([310, 250])
        self.tabs.addTab(fields, '注入字段')
        self.tools = self.text_tab('工具查询')
        self.raw = self.text_tab('原始请求')
        self.local = self.text_tab('本地参考')
        split.addWidget(self.tabs)
        split.setSizes([230, 870])
        layout.addWidget(split, 1)
        footer = QHBoxLayout()
        refresh = QPushButton('刷新已保存记录')
        self.export_button = QPushButton('导出脱敏记录…')
        close = QPushButton('关闭')
        footer.addWidget(refresh)
        footer.addWidget(self.export_button)
        footer.addStretch()
        footer.addWidget(close)
        layout.addLayout(footer)
        refresh.clicked.connect(self.reload)
        self.export_button.clicked.connect(self.export)
        close.clicked.connect(self.accept)
        self.requests.currentRowChanged.connect(self.show_request)
        self.fields.currentItemChanged.connect(self.show_field)
        self.reload()

    def text_tab(self, title):
        widget = QPlainTextEdit()
        widget.setReadOnly(True)
        self.tabs.addTab(widget, title)
        return widget

    def reload(self):
        selected = self.requests.currentRow()
        try:
            self.record = load_inspection(self.project, self.run_id)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.record = {'missing': True, 'note': '无法读取上下文记录，原文件保留：' + str(exc)}
        self.requests.blockSignals(True)
        self.requests.clear()
        for request in self.record.get('requests', []):
            transports = request.get('transports', [])
            state = ('接口已返回' if transports[-1]['status'] == 'received' else '接口提交未成功') if transports else '仅准备快照'
            item = QListWidgetItem(f"第 {request['number']} 次请求 · {state}\n{request['created_at']}")
            item.setData(Qt.UserRole, request)
            self.requests.addItem(item)
        self.requests.blockSignals(False)
        self.export_button.setEnabled(not self.record.get('missing'))
        warnings = self.record.get('read_warnings', []) + self.record.get('index', {}).get('warnings', [])
        self.status.setText(self.record.get('note') or f"任务 {self.run_id} · {len(self.record['requests'])} 轮快照 · {len(self.record['events'])} 条工具/协议记录" +
                            ('\n' + '\n'.join(warnings) if warnings else ''))
        if self.requests.count():
            self.requests.setCurrentRow(min(max(selected, 0), self.requests.count() - 1))
        else:
            self.show_request(-1)

    def show_request(self, index):
        self.fields.clear()
        self.field_value.clear()
        if index < 0 or self.requests.item(index) is None:
            for widget in (self.overview, self.tools, self.raw, self.local):
                widget.setPlainText(self.record.get('note', '尚未保存请求快照。'))
            return
        request = self.requests.item(index).data(Qt.UserRole)
        public = request['injected_context']
        transports = request.get('transports', [])
        lines = [f"第 {request['number']} 次请求 · {request['created_at']}",
                 '请求状态：' + request.get('status', '未知'),
                 '接口提交证据：' + (f'{len(transports)} 次提交尝试' if transports else '未捕获；只有程序准备快照，不能断言已发给AI'),
                 '当前视图：' + pretty(public.get('current_view')),
                 f"原图范围 {public.get('current_count', 0)} 张 · 当前结果 {public.get('current_result_count', 0)} 张 · 选中 {public.get('selected_count', 0)} 张",
                 '本次请求：' + str(public.get('request', '')),
                 '\n缺失或注意项：', *request.get('warnings', []),
                 '\n注入时省略/截断：', *(request.get('omissions') or ['未检测到本轮工具结果或传入历史的额外截断。']),
                 '固定规则：历史最多12条；近期索引最多40条；工具结果列表最多20项、字符串最多2000字符、深度超过6层摘要化。',
                 '\n图片附件：' + (pretty(transports[-1].get('images', [])) if transports else '未捕获提交附件；准备的路径见原始请求页。'),
                 '\n' + request['evidence_note']]
        if request.get('error'):
            lines += ['\n请求错误：' + request['error']]
        self.overview.setPlainText('\n'.join(lines))
        for key, value in public.items():
            brief = f'{len(value)} 项' if isinstance(value, (list, dict)) else str(value)[:100]
            item = QTreeWidgetItem([key, request.get('field_sources', {}).get(key, '请求构建器'), brief])
            item.setData(0, Qt.UserRole, value)
            self.fields.addTopLevelItem(item)
        if self.fields.topLevelItemCount():
            self.fields.setCurrentItem(self.fields.topLevelItem(0))
        events = [event for event in self.record.get('events', []) if event.get('after_request') == request['number']]
        later = [r for r in self.record['requests'] if r['number'] > request['number']]
        future = '后续请求尚未捕获；不能称为已将这些执行结果发给AI。' if events and not later else '后续是否包含完整结果，请对照对应请求的 tool_results_this_turn。'
        self.tools.setPlainText('本次请求已经携带的工具结果（实际注入摘要）：\n' + pretty(public.get('tool_results_this_turn', [])) +
            '\n\n本次回复之后的本地执行记录（不是本次已发送内容）：\n' + pretty(events) + '\n' + future +
            '\n\n经确认继续的本地步骤（不单独构成AI请求）：\n' +
            pretty([event for event in self.record.get('events', []) if event.get('after_request') is None]))
        self.raw.setPlainText('接口最终JSON请求体（密钥隐藏，图片base64替换为字节身份信息）：\n' +
            (pretty(transports) if transports else '没有捕获HTTP提交，不能提供实际请求体。') +
            '\n\n程序准备的完整提示文本：\n' + request['prompt'] +
            '\n\n准备的附件路径：\n' + pretty(request.get('requested_images', [])) +
            '\n\n本次返回记录：\n' + pretty(request.get('response')))
        self.local.setPlainText('本次构建请求时程序持有的参考信息。这里出现不代表字段已发送；发送内容以注入字段及接口请求体为准。\n\n' + pretty(request['known_local']))

    def show_field(self, current, previous=None):
        self.field_value.setPlainText(pretty(current.data(0, Qt.UserRole)) if current is not None else '')

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, '导出脱敏上下文记录', f'context_{self.run_id}.json', 'JSON (*.json)')
        if not path:
            return
        try:
            result = share_copy(self.record, self.project)
            write_json(path, result)
            self.status.setText('已导出脱敏记录。图片内容未导出；文件名、类别和业务文字仍保留，请分享前检查。')
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, '导出失败', sanitize(str(exc)))
