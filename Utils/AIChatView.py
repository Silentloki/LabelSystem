"""Conversation widgets: readable messages, inline progress, result actions."""
import html
import os
import re
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFrame, QPlainTextEdit, QSizePolicy, QLayout)


CHAT_STYLE = """
QWidget#aiChatBody { background: #F8FAFC; color: #263244; }
QWidget#aiChatBody QWidget { font-size: 12pt; }
QWidget#aiChatBody QLabel { border: none; padding: 0; background: transparent; }
QWidget#aiChatBody QLabel#chatTitle { font-size: 16pt; font-weight: 600; color: #15243A; }
QWidget#aiChatBody QLabel#chatMuted { color: #66758B; font-size: 11pt; }
QWidget#aiChatBody QLabel#chatScope { color: #5D718C; font-size: 11pt; padding: 3px 0; }
QWidget#aiChatBody QPushButton { font-size: 12pt; padding: 7px 12px; min-height: 24px; border-radius: 7px; background: white; border: 1px solid #DFE5ED; }
QWidget#aiChatBody QPushButton:hover { background: #EEF4FF; border-color: #B8CDEF; }
QWidget#aiChatBody QPushButton:disabled { color: #9CA8B8; background: #F3F5F8; }
QWidget#aiChatBody QPushButton#chatQuiet { border: none; background: transparent; color: #66758B; }
QWidget#aiChatBody QPushButton#chatSend { background: #336AE8; border: none; color: white; padding: 6px 20px; }
QWidget#aiChatBody QPushButton#chatSend:disabled { background: #C4D2ED; }
QScrollArea#chatTranscript, QWidget#chatMessages { border: none; background: #F8FAFC; }
QWidget#aiChatBody QStackedWidget { border: none; background: transparent; }
QFrame#chatUser { background: #EAF0FC; border: none; border-radius: 13px; }
QFrame#chatAssistant { background: transparent; border: none; }
QFrame#chatTask { background: #F0F4FA; border: 1px solid #E3E9F2; border-radius: 9px; }
QFrame#chatTaskDone { background: transparent; border: none; }
QFrame#chatResult { background: white; border: 1px solid #E1E7F0; border-radius: 9px; }
QFrame#chatHistoryCard { background: white; border: 1px solid #E1E7F0; border-radius: 9px; }
QFrame#chatHistoryCard[current="true"] { background: #F0F5FF; border-color: #BFD1F1; }
QFrame#chatComposer { background: white; border: 1px solid #D7E0EE; border-radius: 12px; }
QWidget#aiChatBody QPlainTextEdit#chatInput { background: transparent; border: none; padding: 2px; font-size: 12pt; }
QFrame#chatPending { background: #FFF8E8; border: 1px solid #ECDAB4; border-radius: 9px; }
QWidget#aiChatBody QLineEdit { padding: 6px; border-radius: 6px; }
QWidget#aiChatBody QListWidget { border: none; background: transparent; padding: 0; }
QWidget#aiChatBody QListWidget::item { padding: 10px; margin: 3px 0; border-radius: 8px; }
"""


def display_time(value):
    """Keep the date visible even when a conversation spans multiple days."""
    if not value:
        return "时间未记录"
    try:
        return datetime.fromisoformat(str(value)).strftime("%Y年%m月%d日 %H:%M")
    except ValueError:
        return str(value)


def formatted_text(text):
    # Escape first: model/user text cannot create links, images or actions.
    text = html.escape(text)
    text = re.sub(r"\*\*([^\n*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`\n]+)`", r"<span style='color:#536783'>\1</span>", text)
    text = re.sub(r"(?m)^#{1,4}\s+(.+)$", r"<b>\1</b>", text)
    text = re.sub(r"(?m)^[-*]\s+", "• ", text)
    return "<div style='line-height:145%'>" + text.replace("\n", "<br>") + "</div>"


class ChatInput(QPlainTextEdit):
    submitted = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
            self.submitted.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class ConversationTranscript(QScrollArea):
    action = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setObjectName("chatTranscript")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content.setObjectName("chatMessages")
        self.layout = QVBoxLayout(self.content)
        self.layout.setContentsMargins(2, 10, 2, 16)
        self.layout.setSpacing(14)
        self.layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.setWidget(self.content)
        self.messages, self.states, self.widgets, self.task_labels = [], {}, {}, {}
        self.visible_count = 80
        self.follow_bottom = True
        self.scroll_anchor = None
        self.saved_scroll = 0
        self.rendering = False
        self.position_timer = QTimer(self)
        self.position_timer.setSingleShot(True)
        self.position_timer.timeout.connect(self.position_messages)
        self.verticalScrollBar().rangeChanged.connect(self.schedule_position)
        self.verticalScrollBar().actionTriggered.connect(self.user_scroll)
        self.verticalScrollBar().sliderMoved.connect(self.user_scroll)

    def schedule_position(self, *unused):
        if not self.rendering:
            self.position_timer.start(0)

    def position_messages(self):
        if self.scroll_anchor and self.scroll_anchor in self.widgets:
            self.ensureWidgetVisible(self.widgets[self.scroll_anchor], 0, 15)
            self.saved_scroll = self.verticalScrollBar().value()
        elif self.follow_bottom:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        else:
            self.verticalScrollBar().setValue(self.saved_scroll)

    def user_scroll(self, *unused):
        self.position_timer.stop()
        self.scroll_anchor = None
        self.follow_bottom = False
        QTimer.singleShot(0, self.remember_scroll)

    def remember_scroll(self):
        bar = self.verticalScrollBar()
        self.saved_scroll = bar.value()
        self.follow_bottom = bar.maximum() - bar.value() <= 4

    def wheelEvent(self, event):
        self.user_scroll()
        super().wheelEvent(event)

    def toPlainText(self):
        return "\n".join(m["text"] + "\n" + "\n".join(c.get("title", "") + " " + c.get("summary", "")
                            for c in m.get("cards", [])) for m in self.messages)

    def clear(self):
        self.render([], {})

    def _label(self, text, muted=False):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        label.setTextFormat(Qt.PlainText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if muted:
            label.setObjectName("chatMuted")
        return label

    def render(self, messages, states, anchor=None, bottom=None):
        scroll = self.saved_scroll if self.position_timer.isActive() and not self.follow_bottom else self.verticalScrollBar().value()
        self.rendering = True
        self.position_timer.stop()
        self.scroll_anchor = anchor
        if anchor:
            self.follow_bottom = False
        elif bottom is True:
            self.follow_bottom = True
        if not self.follow_bottom:
            self.saved_scroll = scroll
        self.messages, self.states = messages, states
        self.widgets, self.task_labels = {}, {}
        if anchor:
            index = next((i for i, m in enumerate(messages) if m["id"] == anchor), len(messages))
            self.visible_count = max(self.visible_count, len(messages) - index)
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        if len(messages) > self.visible_count:
            older = QPushButton("加载更早的消息")
            older.setObjectName("chatQuiet")
            older.clicked.connect(self.show_older)
            self.layout.addWidget(older)
        if not messages:
            empty = self._label("从一句需求开始\n\n筛选图片、整理数据、查看训练结果。\n处理过程和结果会保存在这段对话里。")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: #66758B; padding: 35px 12px; font-size: 12pt;")
            self.layout.addWidget(empty)
        for message in messages[-self.visible_count:]:
            role = message.get("role", "assistant")
            frame = QFrame()
            frame.setObjectName("chatUser" if role == "user" else "chatTask" if role == "task" else "chatAssistant")
            box = QVBoxLayout(frame)
            box.setContentsMargins(13, 10, 13, 11)
            box.setSpacing(8)
            box.addWidget(self._label(("你" if role == "user" else "处理进度" if role == "task" else "AI 助手") +
                                      "  ·  " + display_time(message.get("created_at")), True))
            if role == "task":
                if message.get("status") != "running":
                    frame.setObjectName("chatTaskDone")
                    box.setContentsMargins(13, 0, 13, 0)
                text = self._label(message["text"] +
                                   (" · " + message["metrics"].split(" · ")[0] if message.get("metrics") else ""), True)
                text.setToolTip(message.get("metrics", ""))
                self.task_labels[message["id"]] = text
                box.addWidget(text)
                if message.get('run_id') and os.environ.get('LABELSYSTEM_CONTEXT_DEBUG') == '1':
                    inspect = QPushButton('查看本次上下文')
                    inspect.setObjectName('chatQuiet')
                    inspect.clicked.connect(lambda _, rid=message['run_id']: self.action.emit('context', rid))
                    box.addWidget(inspect, 0, Qt.AlignLeft)
                self.layout.addWidget(frame)
                self.widgets[message["id"]] = frame
                continue
            text = self._label(message["text"])
            text.setTextFormat(Qt.RichText)
            text.setText(formatted_text(message["text"]))
            text.setStyleSheet("font-size: 12pt; color: #273549;")
            box.addWidget(text)
            if message.get("metrics"):
                box.addWidget(self._label(message["metrics"], True))
            for card in message.get("cards", []):
                result = QFrame()
                result.setObjectName("chatResult")
                card_box = QVBoxLayout(result)
                card_box.setContentsMargins(11, 10, 11, 10)
                title = self._label(card["title"])
                title.setStyleSheet("font-weight: 600; font-size: 12pt;")
                card_box.addWidget(title)
                card_box.addWidget(self._label(card["summary"], True))
                state = states.get((card["type"], card["id"]), ("结果不可用", False))
                row = QHBoxLayout()
                row.addWidget(self._label(state[0], True), 1)
                if card['type'] == 'artifact' and card.get('kind') in {'crops', 'dataset', 'converted', 'training_records', 'export'}:
                    folder_button = QPushButton('打开文件夹')
                    folder_button.setEnabled(state[1])
                    folder_button.clicked.connect(lambda _, cid=card['id']: self.action.emit('folder', cid))
                    row.addWidget(folder_button)
                open_button = QPushButton("重新显示" if state[0] == "已隐藏" else "查看结果")
                open_button.setVisible(card.get('kind') != 'export')
                open_button.setEnabled(state[1])
                open_button.clicked.connect(lambda _, typ=card["type"], cid=card["id"]: self.action.emit(typ, cid))
                row.addWidget(open_button)
                card_box.addLayout(row)
                box.addWidget(result)
            if role == "user":
                frame.setMaximumWidth(640)
                wrapper = QWidget()
                row = QHBoxLayout(wrapper)
                row.setContentsMargins(36, 0, 0, 0)
                row.addStretch(1)
                row.addWidget(frame, 9)
                self.layout.addWidget(wrapper)
            else:
                self.layout.addWidget(frame)
            self.widgets[message["id"]] = frame
        self.layout.addStretch(1)
        self.layout.activate()
        self.rendering = False
        self.schedule_position()

    def update_task(self, mid, text):
        if mid in self.task_labels:
            self.task_labels[mid].setText(text)

    def show_older(self):
        self.visible_count += 80
        anchor = self.messages[max(0, len(self.messages) - self.visible_count)]["id"]
        self.render(self.messages, self.states, anchor=anchor, bottom=False)
