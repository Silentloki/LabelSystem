"""Human-readable project inventory and explicit human issue recording."""
import html
import json
import sqlite3
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QTextBrowser,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QComboBox, QPlainTextEdit, QDialogButtonBox, QMessageBox)

from Utils.ProjectContext import ProjectContext, ISSUES


class ContextWorker(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, host, parent):
        super().__init__(parent)
        self.project = host.project_dir
        self.paths, self.flags = list(host.relative_paths), list(host.flag)
        self.classes = list(host.graphicsView.imageItem.existing_categories)

    def run(self):
        try:
            store = ProjectContext(self.project)
            value = store.inventory(self.paths, self.flags, self.classes, self.isInterruptionRequested)
            value['feedback'] = store.feedback(status='all')
            self.ready.emit(value)
        except Exception as exc:
            self.failed.emit(str(exc))


def display_location(value):
    return value.get('project_relative', value.get('path', '未知'))


class ProjectContextDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.store = ProjectContext(host.project_dir)
        self.worker = None
        self.data = {}
        self.setWindowTitle('项目状态')
        self.resize(920, 650)
        layout = QVBoxLayout(self)
        self.status = QLabel('正在整理项目记录…')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.overview = QTextBrowser()
        self.tabs.addTab(self.overview, '项目概况')
        self.datasets = QTextBrowser()
        self.tabs.addTab(self.datasets, '数据集')
        self.resources = self.table(['名称', '类型', '状态', '位置'], '项目资源')
        self.experiments = self.table(['实验', '状态', '主指标', '最高轮次指标', '可比性'], '实验')
        self.results = self.table(['处理批次', '时间', '图片数', '状态', '来源批次'], '处理记录')
        self.feedback_table = self.table(['类型', '图片', '推理/处理批次', '记录时间', '状态', '备注'], '人工记录')
        row = QHBoxLayout()
        self.refresh_button = QPushButton('刷新')
        self.prefer_button = QPushButton('将选中模型设为项目主用')
        self.view_button = QPushButton('查看选中处理结果')
        self.resolve_button = QPushButton('所选问题已处理')
        for button in (self.refresh_button, self.prefer_button, self.view_button, self.resolve_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.refresh_button.clicked.connect(self.refresh)
        self.prefer_button.clicked.connect(self.set_preferred)
        self.view_button.clicked.connect(self.view_result)
        self.resolve_button.clicked.connect(self.resolve_issue)
        self.tabs.currentChanged.connect(self.update_buttons)
        self.refresh()

    def table(self, headers, title):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(table, title)
        return table

    def update_buttons(self, *unused):
        current = self.tabs.currentWidget()
        busy = self.worker is not None and self.worker.isRunning()
        self.prefer_button.setVisible(current is self.resources)
        self.view_button.setVisible(current is self.results)
        self.resolve_button.setVisible(current is self.feedback_table)
        for button in (self.refresh_button, self.prefer_button, self.view_button, self.resolve_button):
            button.setEnabled(not busy)

    def refresh(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.status.setText('正在读取项目记录，不运行模型…')
        self.worker = ContextWorker(self.host, self)
        self.worker.ready.connect(self.populate)
        self.worker.failed.connect(lambda text: self.status.setText('项目状态未能更新：' + text))
        self.worker.finished.connect(self.update_buttons)
        self.worker.start()
        self.update_buttons()

    @staticmethod
    def fill(table, rows):
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value if value is not None else '未知'))
                item.setToolTip(item.text())
                table.setItem(i, j, item)

    def populate(self, data):
        self.data = data
        summary = data['summary']
        esc = lambda value: html.escape(str(value))
        states = summary['sample_states']
        resources = {r['id']: r for r in data['resources']}
        roles = summary['model_roles']
        def model_name(role):
            value = roles.get(role, {})
            resource = resources.get(value.get('resource_id'))
            return (resource['name'] + ('（不可用）' if not value.get('available') else
                    '（文件已变化，请重新选择）' if value.get('changed_since_selected') else '')) if resource else '未指定'
        self.overview.setHtml(
            f"<h3>{esc(summary['project_name'])}</h3><p>图片 {summary['images']} 张 · 类别：{esc('、'.join(summary['classes']))}</p>"
            f"<p>坏品 {states['bad']} · 完全良品 {states['good']} · 过杀品 {states['overkill']} · 未标注 {states['unlabeled']}</p>"
            f"<p>原图：images/　正式标注：jsons/</p><p>独立 Mask 目录：{esc('、'.join(summary['locations']['masks']) or '未发现')}</p>"
            f"<p>{esc(summary['mask_note'])}</p><p>项目主用模型：{esc(model_name('preferred_model'))}</p>"
            f"<p>本地模型上次加载：{esc(model_name('last_loaded_inference'))}（不自动加载或运行）</p>"
            f"<p>已识别 {summary['datasets']} 个数据集、{summary['experiments']} 次实验、{summary['results']} 个处理批次。</p>"
            "<p>数据来自工程文件和实际工作记录。人工问题独立保存，不更改样本类别或正式标注；未标记不等于已确认正常。</p>"
            + ''.join('<p>' + esc(w) + '</p>' for w in summary['warnings']))
        parts = []
        for dataset in data['datasets']:
            parts.append(f"<h3>{esc(dataset['title'])}</h3><p>{esc(display_location(dataset['config']))}</p>"
                         f"<p>类别：{esc(json.dumps(dataset['classes'], ensure_ascii=False))}</p>")
            for name, locations in dataset['split_status'].items():
                parts.append('<p>' + name + '：' + ('；'.join(esc(display_location(v['location'])) +
                    ('（存在）' if v['available'] else '（缺失）') for v in locations) or '未声明') + '</p>')
        self.datasets.setHtml(''.join(parts) or '尚未发现 data.yaml；不会推断 train/val/test 划分。')
        self.fill(self.resources, [[r['name'], '模型' if r['kind'] == 'model' else '目录',
            '可用' if r['available'] else '缺失', r['path']] for r in data['resources']])
        group_labels = {g['group']: f'可比组 {i + 1}' for i, g in enumerate(data['comparisons'])}
        highest = {key for g in data['comparisons'] for key in g['highest']}
        self.fill(self.experiments, [[r['title'], r['status'], r['primary_metric'],
            (r['best'] or {}).get(r['primary_metric']), group_labels.get(r['comparison_group'], '无已确认可比实验') +
            (' · 组内指标最高' if r['id'] in highest else '')]
            for r in data['experiments']])
        self.fill(self.results, [[r['title'], r['created_at'], r['count'],
            ('已隐藏' if r['hidden'] else '可用') if r['available'] else r['status'], r['source_artifact'] or '见批次记录']
            for r in data['results']])
        self.fill(self.feedback_table, [[r['category_name'], r['image'], r['artifact'] or '原图', r['created_at'],
            '来源已变化/不可用' if r['stale'] else '已处理' if r['status'] == 'resolved' else '已记录正常' if r['category'] == 'correct' else '待处理',
            r['note']] for r in data['feedback']])
        self.status.setText('项目状态更新于 ' + summary['as_of'][:19] + ' · 人工记录按实际记录日期查询')

    def selected(self, table, key):
        index = table.currentRow()
        rows = self.data.get(key, [])
        if not 0 <= index < len(rows):
            raise ValueError('请先选中一条记录。')
        return rows[index]

    def set_preferred(self):
        try:
            row = self.selected(self.resources, 'resources')
            if row['kind'] != 'model':
                raise ValueError('请选择模型资源。')
            self.store.set_model(row['path'], 'preferred_model')
            self.refresh()
        except (OSError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(self, '未能设置主用模型', str(exc))

    def view_result(self):
        try:
            row = self.selected(self.results, 'results')
            if not self.host.result_browser.open(row['artifact_id']):
                raise ValueError('此记录没有可查看图片，或来源已经不可用。')
            self.host.tabWidget.setCurrentWidget(self.host.labelPage)
            self.accept()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '无法查看', str(exc))

    def resolve_issue(self):
        try:
            row = self.selected(self.feedback_table, 'feedback')
            self.store.resolve_feedback(row['id'])
            self.refresh()
        except (OSError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(self, '未能更新记录', str(exc))

    def done(self, result):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.status.setText('正在结束读取，请稍后关闭。')
            return
        super().done(result)


def record_feedback(host, images, artifact_id=None):
    if not images:
        QMessageBox.information(host, '记录问题', '请先选中图片。')
        return
    dialog = QDialog(host)
    dialog.setWindowTitle('人工记录问题')
    dialog.resize(460, 280)
    layout = QVBoxLayout(dialog)
    label = QLabel(f'已选 {len(images)} 张图片。记录你的判断，不修改图片或标注。')
    label.setWordWrap(True)
    layout.addWidget(label)
    choices = QComboBox()
    prediction = False
    source_artifacts = {image: artifact_id for image in images}
    if artifact_id:
        try:
            store = ProjectContext(host.project_dir).store
            kind = store.artifact(artifact_id)[1]['kind']
            if kind == 'error_set':
                from Utils.AIResultActions import error_set_rows
                source_artifacts = {Path(row['image']).relative_to(store.project).as_posix(): row['source_artifact']
                                    for row in error_set_rows(store, artifact_id)}
                if any(image not in source_artifacts for image in images):
                    raise ValueError('选中图片不在问题样本集中。')
                prediction = all(source_artifacts[image] and store.artifact(source_artifacts[image])[1]['kind'] == 'candidates'
                                 for image in images)
            else:
                prediction = kind == 'candidates'
        except (ValueError, OSError) as exc:
            QMessageBox.warning(host, '无法记录', str(exc))
            return
    for key, title in ISSUES.items():
        if prediction or key in {'image_quality', 'annotation_question'}:
            choices.addItem(title, key)
    layout.addWidget(choices)
    note = QPlainTextEdit()
    note.setPlaceholderText('备注（选填），例如：边缘反光被识别为破损')
    layout.addWidget(note)
    buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Save).setText('保存')
    buttons.button(QDialogButtonBox.Cancel).setText('取消')
    layout.addWidget(buttons)
    buttons.rejected.connect(dialog.reject)
    def save():
        count = 0
        try:
            store = ProjectContext(host.project_dir)
            with store.db() as db:
                for image in images:
                    store.add_feedback(image, choices.currentData(), note.toPlainText().strip(), source_artifacts[image], _db=db)
            count = len(images)
        except (OSError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(dialog, '未全部保存', f'已保存 {count} 张，剩余未完成：{exc}')
            return
        host.statusBar().showMessage(f'已记录 {count} 张图片的人工判断，可在“项目状态”查看。', 12000)
        dialog.accept()
    buttons.accepted.connect(save)
    dialog.exec_()
