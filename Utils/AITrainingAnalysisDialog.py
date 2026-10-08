"""Modeless training analysis UI. All file parsing and requests run off the UI thread."""
import html
import json
from pathlib import Path

from PyQt5.QtCore import QThread, QUrl, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

from Utils import AITrainingAnalysis as service
from Utils import TrainingReport as report_view
from Utils.AIAugment import DEFAULT_QWEN_MODEL, DEFAULT_QWEN_URL, request_qwen_json


def escaped(value):
    return html.escape(str(value)).replace("\n", "<br>")


def fact_name(key):
    if key == "comparison":
        return "实验对比条件与配置差异"
    prefix, field = key.split(".", 1)
    names = {"config": "训练参数", "progress": "训练进度与耗时", "best": "最高分轮次指标",
             "last": "最后一轮指标", "trend": "分段平均趋势", "limits": "证据范围",
             "dataset": "数据集记录", "per_class": "按类别验证指标", "final": "最终模型验证指标",
             "context": "数据批次与类别", "relations": "程序计算的变化及早停条件",
             "plot_pr": "逐类 PR 图", "plot_confusion": "混淆矩阵原图", "confusion": "原始混淆矩阵"}
    return ("本轮 · " if prefix == "C" else "上轮 · ") + names.get(field, field)


def metric_name(key):
    return key.replace("metrics/", "").replace("precision", "精确率").replace("recall", "召回率").replace("(B)", "（检测框）").replace("(M)", "（分割）")


class AnalysisWorker(QThread):
    prepared = pyqtSignal(object, object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, mode, current, previous=None, api=None, parent=None, request=None):
        super().__init__(parent)
        self.mode, self.current, self.previous, self.api = mode, current, previous, api
        self.request = request or request_qwen_json

    def run(self):
        try:
            if self.mode == "load":
                current = service.load_run(self.current)
                previous = service.load_run(self.previous) if self.previous else None
                service.compare_runs(current, previous)
                self.prepared.emit(current, previous)
            else:
                # Re-read immediately before sending, so cached UI values cannot analyze changed files.
                current = service.load_run(self.current["folder"])
                previous = service.load_run(self.previous["folder"]) if self.previous else None
                if current["hashes"] != self.current["hashes"] or (previous and previous["hashes"] != self.previous["hashes"]):
                    raise ValueError("训练文件已变化，请重新读取结果后分析")
                self.completed.emit(service.analyze(current, previous, self.api, self.request,
                                                    cancelled=self.isInterruptionRequested))
        except Exception as exc:
            key = self.api[0] if self.api else ""
            message = str(exc).replace(key, "***") if key else str(exc)
            self.failed.emit(message)


class AITrainingAnalysisDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.worker = None
        self.current = self.previous = self.report = None
        self.setWindowTitle("AI 训练结果分析")
        self.resize(1080, 800)
        self.setModal(False)
        layout = QVBoxLayout(self)
        hint = QLabel("看清分析的是哪批数据、各类缺陷检得怎样，再决定下一步做什么。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        selectors = QFormLayout()
        self.run_combo, self.previous_combo = QComboBox(), QComboBox()
        self.run_combo.setMinimumContentsLength(30)
        self.previous_combo.setMinimumContentsLength(30)
        self.browse = QPushButton("选择结果目录")
        self.browse_previous = QPushButton("选择对比目录")
        self.refresh = QPushButton("重新读取")
        row = QHBoxLayout()
        row.addWidget(self.run_combo, 1)
        row.addWidget(self.browse)
        row.addWidget(self.refresh)
        selectors.addRow("本次实验", row)
        row = QHBoxLayout()
        row.addWidget(self.previous_combo, 1)
        row.addWidget(self.browse_previous)
        selectors.addRow("对比实验", row)
        layout.addLayout(selectors)
        self.overview = QLabel("请选择已经完成的训练结果。")
        self.overview.setWordWrap(True)
        self.overview.setMinimumHeight(44)
        layout.addWidget(self.overview)
        self.tabs = QTabWidget()
        self.result_view = QTextBrowser()
        self.result_view.setOpenExternalLinks(False)
        self.result_view.setOpenLinks(False)
        self.result_view.anchorClicked.connect(self.show_evidence)
        self.tabs.addTab(self.result_view, "分析结果")
        self.evidence_view = QPlainTextEdit()
        self.evidence_view.setReadOnly(True)
        self.tabs.addTab(self.evidence_view, "训练记录与依据")
        self.chart_view = QTextBrowser()
        self.chart_view.setOpenExternalLinks(False)
        self.tabs.addTab(self.chart_view, "原始图表")
        layout.addWidget(self.tabs, 1)
        self.settings = QGroupBox("模型设置（API Key 只在本次窗口内使用）")
        form = QFormLayout(self.settings)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("填写千问 API Key")
        self.endpoint = QLineEdit(DEFAULT_QWEN_URL)
        self.model = QLineEdit(DEFAULT_QWEN_MODEL)
        form.addRow("API Key", self.api_key)
        form.addRow("接口地址", self.endpoint)
        form.addRow("模型", self.model)
        layout.addWidget(self.settings)
        self.status = QLabel("读取记录和查看已保存报告不调用 AI。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.analyze_button = QPushButton("生成分析（1 次请求）")
        self.stop_button = QPushButton("停止等待")
        self.stop_button.setEnabled(False)
        self.apply_button = QPushButton("应用建议到下一轮配置")
        self.apply_button.setEnabled(False)
        hide_button = QPushButton("收起窗口")
        for button in (self.analyze_button, self.stop_button, self.apply_button):
            buttons.addWidget(button)
        buttons.addStretch()
        buttons.addWidget(hide_button)
        layout.addLayout(buttons)
        self.run_combo.currentIndexChanged.connect(self.load_selection)
        self.previous_combo.currentIndexChanged.connect(self.load_selection)
        self.browse.clicked.connect(lambda: self.choose_folder(False))
        self.browse_previous.clicked.connect(lambda: self.choose_folder(True))
        self.refresh.clicked.connect(self.load_selection)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop)
        self.apply_button.clicked.connect(self.apply_suggestion)
        hide_button.clicked.connect(self.hide)
        self._populate()

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def show_evidence(self, url):
        if url.scheme() != "evidence":
            return
        if ".plot_" in url.path():
            self.tabs.setCurrentWidget(self.chart_view)
            self.chart_view.scrollToAnchor(url.path())
            return
        self.tabs.setCurrentWidget(self.evidence_view)
        cursor = self.evidence_view.textCursor()
        cursor.movePosition(cursor.Start)
        self.evidence_view.setTextCursor(cursor)
        self.evidence_view.find(fact_name(url.path()))

    def _populate(self):
        root = Path(__file__).resolve().parent.parent
        runs = service.discover_runs(self.owner.project_dir, root)
        latest = getattr(self.owner, "latest_training_run", None)
        if latest and (Path(latest) / "results.csv").is_file():
            runs = list(dict.fromkeys([Path(latest).resolve(), *runs]))
        self.run_combo.blockSignals(True)
        self.previous_combo.blockSignals(True)
        self.run_combo.clear()
        self.previous_combo.clear()
        self.previous_combo.addItem("不对比，先分析本轮", None)
        for path in runs:
            self.run_combo.addItem(str(path), str(path))
            self.previous_combo.addItem(str(path), str(path))
        self.run_combo.blockSignals(False)
        self.previous_combo.blockSignals(False)
        self.load_selection()

    def reopen(self, folder=None):
        self.show()
        self.raise_()
        self.activateWindow()
        if folder and not self.running():
            path = str(Path(folder).resolve())
            index = self.run_combo.findData(path)
            if index < 0:
                self.run_combo.addItem(path, path)
                index = self.run_combo.count() - 1
            if self.run_combo.currentIndex() != index:
                self.run_combo.setCurrentIndex(index)

    def choose_folder(self, previous):
        path = QFileDialog.getExistingDirectory(self, "选择含 results.csv 的训练结果目录", self.owner.project_dir)
        if not path:
            return
        combo = self.previous_combo if previous else self.run_combo
        index = combo.findData(path)
        if index < 0:
            combo.addItem(path, path)
            index = combo.count() - 1
        combo.setCurrentIndex(index)

    def set_busy(self, busy):
        for widget in (self.run_combo, self.previous_combo, self.browse, self.browse_previous,
                       self.refresh, self.settings):
            widget.setEnabled(not busy)
        self.analyze_button.setEnabled(not busy and self.current is not None)
        self.stop_button.setEnabled(busy and self.worker.mode == "analyze")
        self.apply_button.setEnabled(False)
        if not busy:
            self.update_apply_state()

    def launch(self, worker):
        self.worker = worker
        worker.prepared.connect(self.on_prepared)
        worker.completed.connect(self.on_completed)
        worker.failed.connect(self.on_failed)
        worker.finished.connect(lambda: self.set_busy(False))
        self.set_busy(True)
        worker.start()

    def load_selection(self, *_):
        if self.running():
            return
        self.current = self.previous = self.report = None
        self.apply_button.setEnabled(False)
        self.result_view.clear()
        self.evidence_view.clear()
        self.chart_view.clear()
        folder = self.run_combo.currentData()
        if not folder:
            self.analyze_button.setEnabled(False)
            self.overview.setText("没有找到当前工程的训练记录。训练完成后打开，或选择已有的结果目录。")
            return
        self.status.setText("正在读取训练记录，不调用 AI…")
        self.launch(AnalysisWorker("load", folder, self.previous_combo.currentData(), parent=self))

    def on_prepared(self, current, previous):
        self.current, self.previous = current, previous
        evidence = service.prepare_evidence(current, previous)
        self.evidence_view.setPlainText("\n\n".join(
            fact_name(key) + "\n来源：" + value["source"] + "\n" + json.dumps(value["value"], ensure_ascii=False, indent=2)
            for key, value in evidence["facts"].items()))
        chart_html = []
        for prefix, run in (("C", current), ("P", previous)):
            if run:
                for chart in run["charts"]:
                    key = prefix + "." + chart["key"]
                    chart_html.append('<a name="' + key + '"></a><h3>' + escaped(fact_name(key))
                                      + '</h3><p>' + escaped(chart["filename"]) + '</p><img width="850" src="'
                                      + escaped(QUrl.fromLocalFile(chart["path"]).toString()) + '">')
        self.chart_view.setHtml("".join(chart_html) or "没有需要读取的统计图，逐类数据见训练记录。")
        self.overview.setText(f"已读取 {current['name']}：{len(current['rows'])} 轮训练记录，{len(current['charts'])} 张类别统计图。\n"
                              + ("已选择对比实验。" if previous else "本次分析这批数据的缺陷表现。"))
        self.report = service.latest_report(current["folder"], current=current, previous=previous)
        if self.report:
            self.model.setText(self.report["model"])
        self.render()
        chart_count = len(current["charts"]) + (len(previous["charts"]) if previous else 0)
        self.status.setText("已打开保存的分析，没有再次调用。" if self.report else
                            f"将发送训练摘要和 {chart_count} 张类别统计图；点击生成后请求一次，不上传原始训练图片。")

    def start_analysis(self):
        if self.running() or self.current is None:
            return
        thread = getattr(self.owner, "train_thread", None)
        if thread is not None and thread.isRunning():
            QMessageBox.information(self, "训练进行中", "请等待本次训练结束后再分析结果。")
            return
        key, endpoint, model = self.api_key.text().strip(), self.endpoint.text().strip(), self.model.text().strip()
        if not key or not endpoint or not model:
            QMessageBox.information(self, "填写模型设置", "请填写 API Key、接口地址和模型。")
            return
        self.report = None
        self.render()
        self.status.setText("正在分析：本次只请求一次，最多等待 120 秒。")
        self.launch(AnalysisWorker("analyze", self.current, self.previous, (key, endpoint, model), parent=self))

    def stop(self):
        if self.running():
            self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.status.setText("已请求停止；正在发出的请求需等待返回或超时，已发送的请求可能仍计费。")

    def on_completed(self, report):
        self.report = report
        self.render()
        usage = report.get("usage", {}).get("total_tokens")
        usage_text = f"{usage:,} tokens" if isinstance(usage, int) else "接口未返回用量"
        if report["status"] == "completed":
            self.status.setText(f"分析已保存；用时 {report['elapsed_seconds']} 秒，{usage_text}。再次打开无需重新调用。")
        elif report["status"] == "cancelled":
            self.status.setText(f"本次已停止，{usage_text}；未生成可用分析。")
        else:
            self.status.setText(f"分析失败，{usage_text}；错误记录已保存，没有生成替代结论。")

    def on_failed(self, message):
        self.status.setText(message)

    def render(self):
        if not self.current:
            return
        self.result_view.setHtml(report_view.render_html(self.current, self.previous, self.report))
        self.analyze_button.setText("重新分析（1 次请求）" if self.report else "生成分析（1 次请求）")
        self.update_apply_state()

    def update_apply_state(self):
        self.apply_button.setEnabled(False)
        if self.running() or not self.current or not self.report:
            return
        try:
            service.application_config(self.current, self.report, self.owner.project_dir)
        except (ValueError, KeyError, TypeError) as exc:
            self.apply_button.setToolTip(str(exc))
        else:
            self.apply_button.setToolTip("确认后还原本轮基础配置并应用这一项改动；不会自动开始训练。")
            self.apply_button.setEnabled(True)

    def apply_suggestion(self):
        if self.running() or not self.report:
            return
        thread = getattr(self.owner, "train_thread", None)
        if thread is not None and thread.isRunning():
            QMessageBox.information(self, "训练进行中", "训练结束后才能更改下一轮配置。")
            return
        try:
            fresh = service.load_run(self.current["folder"])
            config = service.application_config(fresh, self.report, self.owner.project_dir)
            change = self.report["result"]["next_experiment"]["change"]
            text = (f"将训练页还原为本轮的基础配置，并修改 {service.PARAMETERS[change['parameter']]}："
                    f"{change['old']} → {change['new']}。\n\n"
                    f"数据集：{config['data']}\n初始权重：{config['weight']}\n任务：{config['task']}\n"
                    f"轮次：{config['epochs']}；尺寸：{config['imgsz']}；Batch：{config['batch']}\n\n"
                    "确认后只填写配置，由你点击“开始训练”。")
            if QMessageBox.question(self, "应用下一轮配置", text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            self.owner.apply_training_analysis_config(config)
            self.status.setText("建议已应用到训练页。请检查配置后自行开始训练。")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "无法应用建议", str(exc))

    def closeEvent(self, event):
        # Keep the worker and in-memory key alive while a request finishes.
        self.hide()
        event.ignore()
