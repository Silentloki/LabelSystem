from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import filecmp
import shutil
import pickle
import sqlite3
from PyQt5.QtCore import QTimer
from LabelItem import ImageItem
from PyQt5.QtCore import QThreadPool
from Utils import Task
from Utils.AIAugment import AIAugmentDialog
from Utils.AITrainingAnalysisDialog import AITrainingAnalysisDialog
import pyqtgraph as pg
from Utils.yoloTool_fixed import YOLOTrainThread
from Utils.yoloTool_fixed import DetectionThread
from ultralytics import YOLO
import cv2
from UI.WorkWindow import Ui_MainWindow
import json
from Utils.tsl import ChartApp
from Utils.GlobalVar import sqlite_db
import numpy as np
import sys
import os

def get_real_path(relative_path):
    """完美兼容 PyInstaller v6+ 打包环境和源码运行的寻路函数"""
    if hasattr(sys, '_MEIPASS'):
        # 打包后，资源文件全在 _internal 里，_MEIPASS 自动精确指向它
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        # 源码运行时，直接用当前项目根目录
        return os.path.join(os.path.abspath("."), relative_path)

# 直接调用函数，绝对路径死死锁住
ICON_GOOD = get_real_path(os.path.join("res", "good.png"))
ICON_LABEL = get_real_path(os.path.join("res", "label.png"))

# 工作界面
class TestWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, project_dir):
        super(TestWindow, self).__init__()
        self.setupUi(self)
        self.graphicsView.setProject(project_dir)
        self.detection_thread = None
        self.good = 0
        self.bad = 0
        self.deflag = False
        self.class_filter = []
        self.threadPool = QThreadPool.globalInstance()
        self.threadPool.setMaxThreadCount(4)
        self.progressBar.setValue(0)
        self.project_dir = project_dir
        self.image_dir = os.path.join(project_dir, "images")
        self.json_dir = os.path.join(project_dir, "jsons")
        self.data_file = os.path.join(project_dir, "datafile.dat")
        self.flag_file = os.path.join(project_dir, "flagfile.dat")
        self.lable_file = os.path.join(project_dir, "label.txt")
        self.sample_tree_file = os.path.join(project_dir, "sample_tree.json")
        self.data_yaml = os.path.join(project_dir, "dataset/data.yaml")
        self.image_paths = []
        self.relative_paths = []
        self.flag = []
        self.sample_groups = {}
        self.sample_assignments = {}
        # Project-local reference groups, separate from categories/assignments.
        self.temporary_selections = {}
        self.temporary_selection_revision = 0
        self.ai_chat_dock = None
        self.result_browser = None
        self.current_tree_filter = ("all", None, None)
        self.sample_tree_categories_snapshot = []
        self.image_label_cache = {}
        self.path_index_map = {}
        self.mode = True
        self.count = 0
        self.is_good = False
        self.import_good_flag = 0
        self.scene = QGraphicsScene()
        self.checkboxes = {}
        # 自动遍历使用同一个可控定时器，避免重复点击后留下多个后台定时器。
        self.timer1 = QTimer(self)
        self.timer1.setInterval(50)
        self.timer1.timeout.connect(self.process_next_item)
        self.current_index = 0
        self.total_items = 0

        # ``Ui_MainWindow`` only defines the original three file-menu actions.
        # Keep the generated UI untouched and extend the menu at runtime instead.
        self.import_annotations_action = QAction("导入标注", self)
        self.import_annotations_action.setObjectName("import_annotations_action")
        self.import_dataset_action = QAction("导入数据集", self)
        self.import_dataset_action.setObjectName("import_dataset_action")
        self.menu.insertAction(self.export_action, self.import_annotations_action)
        self.menu.insertAction(self.export_action, self.import_dataset_action)
        self.project_context_action = QAction("项目状态", self)
        self.project_context_action.triggered.connect(self.show_project_context)
        self.menu.addSeparator()
        self.menu.addAction(self.project_context_action)

        # 工具栏额外 Action
        self.batch_action = QAction("清除良品状态", self)
        self.toolBar.addAction(self.batch_action)
        self.run_action = QAction("自动遍历", self)
        self.toolBar.addAction(self.run_action)
        self.ai_augment_action = QAction("AI数据增强", self)
        self.toolBar.addAction(self.ai_augment_action)
        self.ai_chat_action = QAction("AI 助手", self)
        self.ai_chat_action.setObjectName("ai_chat_action")
        self.toolBar.addAction(self.ai_chat_action)
        self.ai_chat_action.triggered.connect(self.show_ai_chat)
        self.ai_training_dialog = None
        self.latest_training_run = None
        self.train_thread = None
        self.ai_training_button = QPushButton("AI 分析训练结果", self.trainPage)
        self.ai_training_button.setObjectName("ai_training_button")
        self.verticalLayout_6.addWidget(self.ai_training_button)
        self.ai_training_button.clicked.connect(self.show_ai_training_analysis)
        self.export_action.setText("导入完全良品")
        self.projectCheck = QPushButton("项目体检", self.labelPage)
        self.projectCheck.setObjectName("projectCheck")
        self.verticalLayout.insertWidget(2, self.projectCheck)

        # --- 信号连接 ---
        self.batch_action.triggered.connect(self.set_norm)
        self.run_action.triggered.connect(self.batch_run)
        self.ai_augment_action.triggered.connect(self.show_ai_augment_dialog)
        self.export_action.triggered.connect(self.import_good)

        self.read_action.triggered.connect(lambda: self.graphicsView.setMode(
            ImageItem.LabelablePixmapItem.AnnotationMode.NONE))
        self.rect_action.triggered.connect(lambda: self.graphicsView.setMode(
            ImageItem.LabelablePixmapItem.AnnotationMode.RECTANGLE))
        self.poly_action.triggered.connect(lambda: self.graphicsView.setMode(
            ImageItem.LabelablePixmapItem.AnnotationMode.POLYGON))

        self.loadimg_action.triggered.connect(self.import_image_directory)
        self.import_annotations_action.triggered.connect(self.import_json_annotations)
        self.import_dataset_action.triggered.connect(self.import_yolo_dataset)
        self.pushButton.clicked.connect(self.generate)
        self.listWidget.currentItemChanged.connect(self.show_image)
        self.graphicsView.sendUp.connect(self.setPage)
        self.graphicsView.sendNext.connect(self.setPage)
        self.train_btn.clicked.connect(self.train)
        self.autoLabel.clicked.connect(self.auto)
        self.choosePath.clicked.connect(self.setYaml)
        self.tabWidget.currentChanged.connect(self.scanUpadte)

        # 推理界面按钮
        self.open_action.triggered.connect(self.openProject)
        self.model_btn.clicked.connect(self.load_model)
        self.image_btn.clicked.connect(lambda: self.start_detection('image'))
        self.folder_btn.clicked.connect(lambda: self.start_detection('folder'))
        self.video_btn.clicked.connect(lambda: self.start_detection('video'))

        # 标注引擎交互信号
        self.graphicsView.imageItem.signal_proxy.send_go.connect(self.update_dektop)
        self.graphicsView.imageItem.signal_proxy.annotation_updated.connect(self.update_realtime_stats)
        self.graphicsView.imageItem.signal_proxy.annotation_deleted.connect(self.on_annotation_deleted)
        self.statics.clicked.connect(self.staticData)
        self.projectCheck.clicked.connect(self.show_project_health_report)

        # 初始化数据加载
        self.load_existing_project()
        self.setup_sample_tree_panel()
        self.initUI()
        from Utils.AIConversations import load_groups
        try:
            self.temporary_selections = load_groups(self.project_dir, self.relative_paths)
        except (OSError, ValueError, TypeError) as exc:
            self.statusBar().showMessage("筛选分组未能恢复，原记录已保留：" + str(exc), 15000)
        from Utils.AIResultBrowser import ResultBrowser
        self.result_browser = ResultBrowser(self)
        self.rebuild_sample_tree()

        # 列表右键菜单
        self.listWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listWidget.customContextMenuRequested.connect(self.show_context_menu)
        # --- 新增：开启 Windows 风格多选模式 ---
        self.listWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setup_top_navigation()

    # ================= 业务核心方法 =================

    def setup_top_navigation(self):
        """Rearrange only the two header rows, retaining the existing pages/actions."""
        self.tabWidget.tabBar().hide()
        self.page_navigation = QWidget(self.centralwidget)
        self.page_navigation.setObjectName("page_navigation")
        navigation_layout = QHBoxLayout(self.page_navigation)
        navigation_layout.setContentsMargins(6, 4, 6, 4)
        navigation_layout.setSpacing(4)
        self.page_button_group = QButtonGroup(self)
        self.page_button_group.setExclusive(True)
        self.page_buttons = []
        for index, title in enumerate(("标注", "训练", "推理")):
            self.tabWidget.setTabText(index, title)
            button = QToolButton(self.page_navigation)
            button.setText(title)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.page_button_group.addButton(button, index)
            button.clicked.connect(lambda checked=False, page=index: self.tabWidget.setCurrentIndex(page))
            navigation_layout.addWidget(button)
            self.page_buttons.append(button)
        navigation_layout.addStretch(1)
        self.toolBar.removeAction(self.ai_chat_action)
        self.ai_chat_button = QToolButton(self.page_navigation)
        self.ai_chat_button.setObjectName("header_ai_chat")
        self.ai_chat_button.setDefaultAction(self.ai_chat_action)
        self.ai_chat_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        navigation_layout.addWidget(self.ai_chat_button)
        self.page_navigation.setStyleSheet("""
            QWidget#page_navigation { background: #FFFFFF; border-bottom: 1px solid #E5E7EB; }
            QWidget#page_navigation QToolButton {
                padding: 6px 16px; border: 1px solid transparent; border-radius: 6px;
                background: transparent; color: #4B5563;
            }
            QWidget#page_navigation QToolButton:hover { background: #EFF6FF; color: #1D4ED8; }
            QWidget#page_navigation QToolButton:checked {
                background: #E8EFFF; border-color: #B9CDF9; color: #1D4ED8;
            }
            QWidget#page_navigation QToolButton#header_ai_chat {
                background: #F2F6FF; border-color: #C2D1EF; color: #1D4ED8;
            }
        """)

        # Embed the original toolbar below navigation, with all six actions exposed.
        self.removeToolBar(self.toolBar)
        self.toolBar.setParent(self.centralwidget)
        self.toolBar.setMovable(False)
        self.toolBar.setFloatable(False)
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toolBar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toolBar.setStyleSheet("""
            QToolBar { padding: 4px 6px; spacing: 4px; border-bottom: 1px solid #E5E7EB; }
            QToolBar QToolButton { padding: 6px 12px; border-radius: 6px; }
        """)
        self.verticalLayout_3.insertWidget(0, self.page_navigation)
        self.verticalLayout_3.insertWidget(1, self.toolBar)
        self.verticalLayout_3.setSpacing(0)
        self.annotation_mode_group = QActionGroup(self)
        self.annotation_mode_group.setExclusive(True)
        modes = ImageItem.LabelablePixmapItem.AnnotationMode
        for action, mode in ((self.read_action, modes.NONE), (self.rect_action, modes.RECTANGLE),
                             (self.poly_action, modes.POLYGON)):
            action.setCheckable(True)
            self.annotation_mode_group.addAction(action)
            action.setChecked(self.graphicsView.imageItem.current_mode == mode)
        self.tabWidget.currentChanged.connect(self.sync_top_navigation)
        self.sync_top_navigation(self.tabWidget.currentIndex())

    def sync_top_navigation(self, index):
        if 0 <= index < len(self.page_buttons):
            self.page_buttons[index].setChecked(True)
        self.toolBar.setVisible(self.tabWidget.currentWidget() is self.labelPage)

    def show_ai_chat(self):
        from Utils.AIChatDialog import AIChatWindow
        if self.ai_chat_dock is None:
            self.ai_chat_dock = AIChatWindow(self)
        if self.ai_chat_dock.isMinimized():
            self.ai_chat_dock.setWindowState(self.ai_chat_dock.windowState() & ~Qt.WindowMinimized)
        self.ai_chat_dock.show()
        self.ai_chat_dock.raise_()
        self.ai_chat_dock.activateWindow()

    def show_project_context(self):
        from Utils.ProjectContextDialog import ProjectContextDialog
        ProjectContextDialog(self).exec_()

    def record_project_model(self, path):
        from Utils.ProjectContext import ProjectContext
        self.current_inference_resource_id = None
        try:
            self.current_inference_resource_id = ProjectContext(self.project_dir).set_model(path)
        except (OSError, ValueError, sqlite3.Error) as exc:
            self.statusBar().showMessage("模型已加载，但项目记录未保存：" + str(exc), 15000)

    def upsert_temporary_selection(self, title, criteria, paths, code="", group_id=None):
        """Public UI tool: accept only existing project references, never labels."""
        import uuid
        if group_id is not None and group_id not in self.temporary_selections:
            raise ValueError("临时分组已移除，请重新筛选。")
        known = set(self.relative_paths)
        if any(path not in known for path in paths):
            raise ValueError("工程图片列表已变化，请重新筛选。")
        group_id = group_id or uuid.uuid4().hex
        self.temporary_selections[group_id] = {
            "title": title, "criteria": criteria, "paths": list(dict.fromkeys(paths)), "code": code}
        self.temporary_selection_revision += 1
        if self.result_browser:
            self.result_browser.leave()
        self.current_tree_filter = ("temporary", group_id, None)
        self.tabWidget.setCurrentWidget(self.labelPage)
        self.refresh_image_list(preferred_row=0)
        self.save_temporary_selections()
        return group_id

    def remove_temporary_selection(self, group_id):
        if group_id not in self.temporary_selections:
            return
        del self.temporary_selections[group_id]
        self.temporary_selection_revision += 1
        if self.current_tree_filter == ("temporary", group_id, None):
            self.current_tree_filter = ("all", None, None)
            self.refresh_image_list(preferred_row=0)
        else:
            self.rebuild_sample_tree()
        self.save_temporary_selections()

    def save_temporary_selections(self):
        from Utils.AIConversations import save_groups
        try:
            save_groups(self.project_dir, self.temporary_selections)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage("筛选分组尚未保存：" + str(exc), 15000)
        self.notify_ai_context_changed()

    def notify_ai_context_changed(self):
        chat = self.ai_chat_dock
        if chat is not None and hasattr(chat, "conversation"):
            chat.refresh_scope()
            chat.render_conversation(bottom=False)
            chat.save_conversation()

    def reload_after_ai_changes(self, result):
        """Reload committed documents without autosaving the stale canvas back."""
        if self.result_browser and self.result_browser.active_id:
            self.result_browser.leave()
            self.current_tree_filter = ("all", None, None)
        current = self.get_item_relative_path(self.listWidget.currentItem())
        for path, flag in result.get("set_flags", {}).items():
            index = self.get_global_index_from_path(path)
            if index >= 0:
                self.flag[index] = flag
        if os.path.exists(self.lable_file):
            with open(self.lable_file, encoding="utf-8-sig") as stream:
                self.graphicsView.imageItem.initCatories([line.strip() for line in stream if line.strip()])
        canvas = self.graphicsView.imageItem
        blocker = QSignalBlocker(canvas.signal_proxy)
        canvas.cancel_current_annotation()
        canvas.remove_annotations()
        canvas.setPixmap(QPixmap())
        canvas.setPath("")
        del blocker
        self.invalidate_image_label_cache()
        self.load_sample_tree_state()
        self.cleanup_invalid_assignments()
        self._refresh_after_external_import(current)

    def show_temporary_selection_details(self, group_id):
        group = self.temporary_selections.get(group_id)
        if group:
            box = QMessageBox(self)
            box.setWindowTitle(group["title"])
            box.setText(group["criteria"] + "\n\n本次筛选快照；修改标注后可通过对话重新筛选。\n分组随项目保存，只引用原图片，不复制图片。")
            box.setDetailedText(group.get("code", ""))
            box.exec_()

    def show_ai_augment_dialog(self):
        if not os.path.exists(os.path.join(self.project_dir, "dataset", "data.yaml")):
            QMessageBox.information(self, "提示", "请先点击“生成数据集”，生成 dataset/data.yaml 后再进行 AI 数据增强。")
            return
        dialog = AIAugmentDialog(self.project_dir, self)
        dialog.exec_()

    def sync_categories_from_data(self):
        """核心修复：从JSON文件中自动找回标签定义"""
        if not os.path.exists(self.json_dir):
            return False
        self.invalidate_image_label_cache()
        categories = list(self.graphicsView.imageItem.existing_categories)
        known_tags = set(categories)
        for rel_path in self.relative_paths:
            for tag in sorted(self.get_image_labels(rel_path)):
                tag = str(tag).strip()
                if tag and tag not in known_tags:
                    categories.append(tag)
                    known_tags.add(tag)
        changed = categories != self.graphicsView.imageItem.existing_categories
        if changed:
            self.graphicsView.imageItem.initCatories(categories)
        return changed

    def setup_sample_tree_panel(self):
        self.left_panel = QWidget(self.labelPage)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.sampleTree = QTreeWidget(self.left_panel)
        self.sampleTree.setHeaderHidden(True)
        self.sampleTree.setColumnCount(2)
        self.sampleTree.setIndentation(16)
        self.sampleTree.header().setStretchLastSection(False)
        self.sampleTree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.sampleTree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.sampleTree.setColumnWidth(1, 30)
        self.sampleTree.setObjectName("sampleTree")
        self.sampleTree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sampleTree.customContextMenuRequested.connect(self.show_sample_tree_context_menu)
        self.sampleTree.itemClicked.connect(self.on_sample_tree_item_clicked)

        left_layout.addWidget(self.sampleTree, 2)
        left_layout.addWidget(self.listWidget, 5)

        self.horizontalLayout_2.removeWidget(self.listWidget)
        self.horizontalLayout_2.insertWidget(0, self.left_panel)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 4)
        self.horizontalLayout_2.setStretch(2, 1)

        self.rebuild_sample_tree()

    def load_sample_tree_state(self):
        self.sample_groups = {}
        self.sample_assignments = {}
        if not os.path.exists(self.sample_tree_file):
            return
        try:
            with open(self.sample_tree_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.sample_groups = data.get('groups', {})
            self.sample_assignments = data.get('assignments', {})
        except Exception:
            self.sample_groups = {}
            self.sample_assignments = {}

    def ensure_sample_tree_integrity(self):
        valid_tags = set(self.graphicsView.imageItem.existing_categories)
        cleaned_groups = {}
        for tag in self.graphicsView.imageItem.existing_categories:
            children = self.sample_groups.get(tag, [])
            cleaned_children = []
            seen = set()
            for child in children:
                child_name = str(child).strip()
                if child_name and child_name not in seen:
                    cleaned_children.append(child_name)
                    seen.add(child_name)
            cleaned_groups[tag] = cleaned_children
        self.sample_groups = cleaned_groups

        valid_paths = set(self.relative_paths)
        cleaned_assignments = {}
        for rel_path, assignment in self.sample_assignments.items():
            if rel_path not in valid_paths or not isinstance(assignment, dict):
                continue
            per_image = {}
            for tag, child in assignment.items():
                if tag in valid_tags and child in self.sample_groups.get(tag, []):
                    per_image[tag] = child
            if per_image:
                cleaned_assignments[rel_path] = per_image
        self.sample_assignments = cleaned_assignments

    def get_status_counts(self):
        return {
            "complete_good": self.flag.count(2),
            "overkill_good": self.flag.count(3),
            "bad": self.flag.count(1),
            "unlabeled": self.flag.count(0),
        }

    def get_status_paths(self, status):
        status_flag = {"complete_good": 2, "overkill_good": 3, "bad": 1, "unlabeled": 0}.get(status)
        if status_flag is None:
            return []
        return [
            rel_path
            for i, rel_path in enumerate(self.relative_paths)
            if i < len(self.flag) and self.flag[i] == status_flag
        ]

    def rebuild_sample_tree(self):
        if not hasattr(self, 'sampleTree'):
            return
        current_key = self.current_tree_filter
        self.sampleTree.blockSignals(True)
        self.sampleTree.clear()

        all_item = QTreeWidgetItem([f"全部样本 ({len(self.relative_paths)})"])
        all_item.setData(0, Qt.UserRole, ("all", None, None))
        self.sampleTree.addTopLevelItem(all_item)

        status_counts = self.get_status_counts()
        status_item = QTreeWidgetItem(["样本状态"])
        status_item.setData(0, Qt.UserRole, ("status_root", None, None))
        self.sampleTree.addTopLevelItem(status_item)
        for status_key, status_name in [
            ("complete_good", "完全良品"),
            ("overkill_good", "过杀品"),
            ("bad", "坏品"),
            ("unlabeled", "未标注"),
        ]:
            child_item = QTreeWidgetItem([f"{status_name} ({status_counts[status_key]})"])
            child_item.setData(0, Qt.UserRole, ("status", status_key, None))
            status_item.addChild(child_item)
        status_item.setExpanded(True)

        for tag in self.graphicsView.imageItem.existing_categories:
            parent_item = QTreeWidgetItem([tag])
            parent_item.setData(0, Qt.UserRole, ("tag", tag, None))
            self.sampleTree.addTopLevelItem(parent_item)
            for child in self.sample_groups.get(tag, []):
                child_item = QTreeWidgetItem([child])
                child_item.setData(0, Qt.UserRole, ("child", tag, child))
                parent_item.addChild(child_item)
            parent_item.setExpanded(True)

        if self.temporary_selections or (self.result_browser and self.result_browser.entries):
            temporary_root = QTreeWidgetItem(["临时结果"])
            temporary_root.setToolTip(0, "筛选分组与处理结果，不属于正式类别；处理产物可在重开工程后查看")
            temporary_root.setData(0, Qt.UserRole, ("temporary_root", None, None))
            self.sampleTree.addTopLevelItem(temporary_root)
            known = set(self.relative_paths)
            for group_id, group in self.temporary_selections.items():
                count = sum(path in known for path in group["paths"])
                item = QTreeWidgetItem([f"{group['title']} ({count})"])
                item.setData(0, Qt.UserRole, ("temporary", group_id, None))
                item.setToolTip(0, group["criteria"] + "\n临时快照，右键可查看条件或移除。")
                temporary_root.addChild(item)
                button = QToolButton(self.sampleTree)
                button.setText("×")
                button.setToolTip("移除此临时分组，保留原图片和标注")
                button.setFixedWidth(25)
                button.setStyleSheet("QToolButton { border: none; padding: 0; min-height: 24px; background: transparent; }")
                button.clicked.connect(lambda _, gid=group_id: self.remove_temporary_selection(gid))
                self.sampleTree.setItemWidget(item, 1, button)
            if self.result_browser:
                self.result_browser.add_tree_nodes(temporary_root)
            temporary_root.setExpanded(True)

        self.sampleTree.blockSignals(False)
        self.select_sample_tree_item(current_key)

    def select_sample_tree_item(self, key):
        if not hasattr(self, 'sampleTree'):
            return
        target = self.find_sample_tree_item(key)
        if target is not None:
            parent = target.parent()
            if parent is not None:
                parent.setExpanded(True)
        if target is None and self.sampleTree.invisibleRootItem().childCount() > 0:
            target = self.sampleTree.invisibleRootItem().child(0)
            key = target.data(0, Qt.UserRole)
        if target:
            self.sampleTree.setCurrentItem(target)
            self.current_tree_filter = key
        self.sample_tree_categories_snapshot = list(self.graphicsView.imageItem.existing_categories)

    def find_sample_tree_item(self, key):
        if not hasattr(self, 'sampleTree') or not key:
            return None
        root = self.sampleTree.invisibleRootItem()
        for i in range(root.childCount()):
            top_item = root.child(i)
            if top_item.data(0, Qt.UserRole) == key:
                return top_item
            for j in range(top_item.childCount()):
                child_item = top_item.child(j)
                if child_item.data(0, Qt.UserRole) == key:
                    return child_item
        return None

    def update_current_tree_item_count(self, count):
        if not hasattr(self, 'sampleTree'):
            return
        item = self.find_sample_tree_item(self.current_tree_filter)
        if not item:
            return
        filter_type, tag, child = self.current_tree_filter
        if filter_type == "tag" and tag:
            item.setText(0, f"{tag} ({count})")
        elif filter_type == "child" and child:
            item.setText(0, f"{child} ({count})")

    def sync_sample_tree_categories(self):
        categories = list(self.graphicsView.imageItem.existing_categories)
        if categories == self.sample_tree_categories_snapshot:
            return
        self.ensure_sample_tree_integrity()
        self.persist_project_state()
        self.rebuild_sample_tree()

    def on_sample_tree_item_clicked(self, item, _column):
        key = item.data(0, Qt.UserRole)
        if not key or key[0] in ("status_root", "temporary_root"):
            return
        if self.ai_chat_dock is not None:
            self.ai_chat_dock.missing_view = False
        if key[0] == "artifact":
            self.result_browser.open(key[1])
            return
        if self.result_browser:
            self.result_browser.leave()
        self.current_tree_filter = key
        self.refresh_image_list()

    def show_sample_tree_context_menu(self, pos):
        if not hasattr(self, 'sampleTree'):
            return
        item = self.sampleTree.itemAt(pos)
        menu = QMenu(self)
        if item is None:
            add_defect_action = QAction("新增缺陷", self)
            add_defect_action.triggered.connect(self.add_project_tag)
            menu.addAction(add_defect_action)
            menu.exec_(self.sampleTree.mapToGlobal(pos))
            return

        key = item.data(0, Qt.UserRole)
        if not key:
            return
        self.sampleTree.setCurrentItem(item)

        if key[0] == "artifact":
            view_action = menu.addAction("查看图片结果")
            hide_action = menu.addAction("隐藏结果")
            delete_action = menu.addAction("删除处理结果…")
            action = menu.exec_(self.sampleTree.mapToGlobal(pos))
            if action == view_action:
                self.result_browser.open(key[1])
            elif action == hide_action:
                self.result_browser.hide(key[1])
            elif action == delete_action:
                self.result_browser.delete(key[1])
            return
        elif key[0] == "temporary":
            details_action = menu.addAction("查看筛选条件与代码")
            remove_action = menu.addAction("移除临时分组（保留原图）")
            action = menu.exec_(self.sampleTree.mapToGlobal(pos))
            if action == details_action:
                self.show_temporary_selection_details(key[1])
            elif action == remove_action:
                self.remove_temporary_selection(key[1])
            return
        elif key[0] == "temporary_root":
            return
        elif key[0] == "all":
            add_defect_action = QAction("新增缺陷", self)
            add_defect_action.triggered.connect(self.add_project_tag)
            menu.addAction(add_defect_action)
        elif key[0] in ["status_root", "status"]:
            pass
        elif key[0] == "tag":
            add_action = QAction("新增子样本分类", self)
            rename_tag_action = QAction("重命名缺陷", self)
            delete_tag_action = QAction("删除缺陷", self)
            add_action.triggered.connect(lambda _, tag=key[1]: self.add_subcategory(tag))
            rename_tag_action.triggered.connect(lambda _, tag=key[1]: self.rename_project_tag(tag))
            delete_tag_action.triggered.connect(lambda _, tag=key[1]: self.delete_project_tag(tag))
            menu.addAction(add_action)
            menu.addAction(rename_tag_action)
            menu.addAction(delete_tag_action)
        elif key[0] == "child":
            rename_action = QAction("重命名子样本分类", self)
            delete_action = QAction("删除子样本分类", self)
            promote_action = QAction("升级为新缺陷", self)
            export_action = QAction("导出该分类样本", self)
            rename_action.triggered.connect(lambda _, tag=key[1], child=key[2]: self.rename_subcategory(tag, child))
            delete_action.triggered.connect(lambda _, tag=key[1], child=key[2]: self.delete_subcategory(tag, child))
            promote_action.triggered.connect(lambda _, tag=key[1], child=key[2]: self.promote_subcategory_to_defect(tag, child))
            export_action.triggered.connect(lambda _, tag=key[1], child=key[2]: self.export_subcategory_samples(tag, child))
            menu.addAction(rename_action)
            menu.addAction(delete_action)
            menu.addAction(promote_action)
            menu.addSeparator()
            menu.addAction(export_action)

        if not menu.isEmpty():
            menu.exec_(self.sampleTree.mapToGlobal(pos))

    def add_subcategory(self, tag):
        name = self.prompt_tag_name("新增子样本分类", f"请为【{tag}】输入子样本分类名称:")
        if not name:
            return
        children = self.sample_groups.setdefault(tag, [])
        if name in children:
            QMessageBox.information(self, "提示", "子样本分类名称已存在")
            return
        children.append(name)
        self.persist_project_state()
        self.rebuild_sample_tree()

    def rename_subcategory(self, tag, child):
        new_name = self.prompt_tag_name("重命名子样本分类", "请输入新的子样本分类名称:", child)
        if not new_name or new_name == child:
            return
        children = self.sample_groups.get(tag, [])
        if new_name in children:
            QMessageBox.information(self, "提示", "新的子样本分类名称已存在")
            return
        self.sample_groups[tag] = [new_name if x == child else x for x in children]
        for rel_path, assignment in self.sample_assignments.items():
            if assignment.get(tag) == child:
                assignment[tag] = new_name
        self.persist_project_state()
        self.rebuild_sample_tree()
        if self.current_tree_filter == ("child", tag, child):
            self.current_tree_filter = ("child", tag, new_name)
            self.refresh_image_list()

    def delete_subcategory(self, tag, child):
        used_count = sum(1 for assignment in self.sample_assignments.values() if assignment.get(tag) == child)
        if used_count > 0:
            reply = QMessageBox.question(
                self,
                "删除确认",
                f"删除子样本分类【{child}】会清除其图片归类关系，是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        self.sample_groups[tag] = [x for x in self.sample_groups.get(tag, []) if x != child]
        for rel_path, assignment in list(self.sample_assignments.items()):
            if assignment.get(tag) == child:
                assignment.pop(tag, None)
            if not assignment:
                self.sample_assignments.pop(rel_path, None)
        self.persist_project_state()
        if self.current_tree_filter == ("child", tag, child):
            self.current_tree_filter = ("tag", tag, None)
        self.rebuild_sample_tree()
        self.refresh_image_list()

    def promote_subcategory_to_defect(self, tag, child):
        if child not in self.sample_groups.get(tag, []):
            QMessageBox.information(self, "提示", "子样本分类不存在")
            return

        new_name = self.prompt_tag_name("升级为新缺陷", "请输入新缺陷名称:", child)
        if not new_name:
            return
        if new_name == tag:
            QMessageBox.information(self, "提示", "新缺陷名称不能和原缺陷相同")
            return
        if new_name in self.graphicsView.imageItem.existing_categories:
            QMessageBox.information(self, "提示", "缺陷名称已存在")
            return

        self.save_current_image_annotations()
        paths = self.get_paths_for_subcategory(tag, child)
        assigned_count = sum(
            1 for assignment in self.sample_assignments.values()
            if assignment.get(tag) == child
        )
        skipped_count = max(0, assigned_count - len(paths))
        confirm_lines = [
            f"将子样本分类【{child}】升级为新的缺陷【{new_name}】。",
            f"会把 {len(paths)} 张已归类样本中的标注从【{tag}】改为【{new_name}】。",
            "升级后会从原缺陷下移除该子分类。",
        ]
        if skipped_count:
            confirm_lines.append(f"另有 {skipped_count} 条归类记录因图片中没有【{tag}】标注，将只清除归类关系。")
        if not paths:
            confirm_lines.append("当前没有可迁移的标注样本，只会创建新缺陷并移除该子分类。")

        reply = QMessageBox.question(
            self,
            "升级确认",
            "\n".join(confirm_lines) + "\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        changed_files, changed_annotations = self.replace_tag_in_jsons_for_paths(paths, tag, new_name)
        self.invalidate_image_label_cache()

        self.graphicsView.imageItem.existing_categories.append(new_name)
        self.sample_groups.setdefault(new_name, [])
        self.sample_groups[tag] = [x for x in self.sample_groups.get(tag, []) if x != child]

        for rel_path, assignment in list(self.sample_assignments.items()):
            if assignment.get(tag) == child:
                assignment.pop(tag, None)
            if not assignment:
                self.sample_assignments.pop(rel_path, None)

        current_item = self.listWidget.currentItem()
        current_rel_path = self.get_item_relative_path(current_item)
        if current_rel_path in set(paths):
            for ann in self.graphicsView.imageItem.annotations:
                if ann.category == tag:
                    ann.set_category(new_name)

        self.persist_project_state()
        self.set_fliter_Widget()
        self.current_tree_filter = ("tag", new_name, None)
        self.rebuild_sample_tree()
        self.refresh_image_list()
        self.update_realtime_stats()
        QMessageBox.information(
            self,
            "升级完成",
            f"已创建新缺陷【{new_name}】。\n"
            f"更新 JSON 文件: {changed_files} 个\n"
            f"更新标注: {changed_annotations} 个"
        )

    def get_item_relative_path(self, item):
        return item.toolTip() if item else None

    def resolve_image_path(self, rel_path):
        if not rel_path:
            return None
        return os.path.normpath(os.path.join(self.project_dir, rel_path))

    def annotation_json_path_for_rel_path(self, rel_path):
        if not rel_path:
            return None
        image_name = os.path.splitext(os.path.basename(rel_path))[0]
        return os.path.join(self.json_dir, image_name + '.json')

    def remove_image_path_cache(self, image_path):
        if not image_path:
            return
        target = os.path.normcase(os.path.abspath(image_path))
        self.image_paths = [
            path for path in self.image_paths
            if os.path.normcase(os.path.abspath(path)) != target
        ]

    def get_global_index_from_path(self, rel_path):
        return self.path_index_map.get(rel_path, -1)

    def rebuild_path_index(self):
        self.path_index_map = {
            rel_path: index
            for index, rel_path in enumerate(self.relative_paths)
        }

    def get_item_global_index(self, item):
        return self.get_global_index_from_path(self.get_item_relative_path(item))

    def get_paths_for_tag(self, tag):
        matched = []
        for rel_path in self.relative_paths:
            if tag in self.get_image_labels(rel_path):
                matched.append(rel_path)
        return matched

    def get_paths_for_subcategory(self, tag, child):
        return [
            rel_path
            for rel_path, assignment in self.sample_assignments.items()
            if assignment.get(tag) == child and tag in self.get_image_labels(rel_path)
        ]

    def get_paths_for_current_tree_filter(self):
        filter_type, tag, child = self.current_tree_filter
        if filter_type == "artifact":
            # Result images are not source samples. Do not silently process all
            # originals when the user refers to visible processed images.
            return []
        if filter_type == "temporary":
            known = set(self.relative_paths)
            return [p for p in self.temporary_selections.get(tag, {}).get("paths", []) if p in known]
        if filter_type == "status" and tag:
            return self.get_status_paths(tag)
        if filter_type == "tag" and tag:
            self.cleanup_invalid_assignments()
            return self.get_paths_for_tag(tag)
        if filter_type == "child" and tag and child:
            self.cleanup_invalid_assignments()
            return self.get_paths_for_subcategory(tag, child)
        return list(self.relative_paths)

    def refresh_image_list(self, preferred_row=None, preserve_display=False):
        if self.result_browser and self.current_tree_filter[0] == "artifact":
            self.result_browser.open(self.current_tree_filter[1])
            return
        if self.result_browser:
            self.result_browser.leave()
        self.rebuild_sample_tree()
        paths = self.get_paths_for_current_tree_filter()
        self.update_current_tree_item_count(len(paths))
        blocker = QSignalBlocker(self.listWidget) if preserve_display else None
        self.populate_listwidget(paths)
        if blocker is not None:
            del blocker
        self.set_fliter_Widget()
        if preferred_row is not None and 0 <= preferred_row < self.listWidget.count():
            self.listWidget.setCurrentRow(preferred_row)
        if not paths and self.current_tree_filter[0] == "temporary":
            # An empty selection must not keep showing the previous image.
            canvas = self.graphicsView.imageItem
            signal_blocker = QSignalBlocker(canvas.signal_proxy)
            canvas.remove_annotations()
            canvas.setPixmap(QPixmap())
            canvas.setPath("")
            del signal_blocker
            self.update_realtime_stats()

    def make_unique_export_dir(self, base_dir):
        if not os.path.exists(base_dir):
            return base_dir
        index = 1
        while True:
            candidate = f"{base_dir}_{index}"
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def safe_export_name(self, name):
        invalid_chars = '<>:"/\\|?*'
        cleaned = ''.join('_' if ch in invalid_chars else ch for ch in str(name)).strip()
        return cleaned or "未命名"

    def export_subcategory_samples(self, tag, child):
        paths = self.get_paths_for_subcategory(tag, child)
        if not paths:
            QMessageBox.information(self, "提示", f"子样本分类【{child}】下暂无可导出的图片。")
            return

        root_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not root_dir:
            return

        export_name = f"{self.safe_export_name(tag)}_{self.safe_export_name(child)}_样本"
        export_dir = self.make_unique_export_dir(os.path.join(root_dir, export_name))
        image_out = os.path.join(export_dir, "images")
        json_out = os.path.join(export_dir, "jsons")
        os.makedirs(image_out, exist_ok=True)
        os.makedirs(json_out, exist_ok=True)

        image_count = 0
        json_count = 0
        missing_images = []
        missing_jsons = []

        for rel_path in paths:
            image_path = os.path.join(self.project_dir, rel_path)
            image_name = os.path.basename(rel_path)
            json_name = os.path.splitext(image_name)[0] + ".json"
            json_path = os.path.join(self.json_dir, json_name)

            if os.path.exists(image_path):
                shutil.copy2(image_path, os.path.join(image_out, image_name))
                image_count += 1
            else:
                missing_images.append(rel_path)

            if os.path.exists(json_path):
                shutil.copy2(json_path, os.path.join(json_out, json_name))
                json_count += 1
            else:
                missing_jsons.append(json_name)

        info_path = os.path.join(export_dir, "export_info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("子样本分类导出信息\n")
            f.write(f"工程目录: {self.project_dir}\n")
            f.write(f"缺陷类别: {tag}\n")
            f.write(f"子样本分类: {child}\n")
            f.write(f"导出图片数量: {image_count}\n")
            f.write(f"导出JSON数量: {json_count}\n")
            f.write(f"缺失图片数量: {len(missing_images)}\n")
            f.write(f"缺失JSON数量: {len(missing_jsons)}\n")
            if missing_images:
                f.write("\n缺失图片:\n")
                for item in missing_images:
                    f.write(f"- {item}\n")
            if missing_jsons:
                f.write("\n缺失JSON:\n")
                for item in missing_jsons:
                    f.write(f"- {item}\n")

        QMessageBox.information(
            self,
            "导出完成",
            f"已导出子样本分类【{tag}/{child}】\n"
            f"图片: {image_count} 张\n"
            f"JSON: {json_count} 个\n"
            f"目录: {export_dir}"
        )

    def get_image_labels(self, rel_path):
        if not rel_path:
            return set()
        if rel_path in self.image_label_cache:
            return set(self.image_label_cache[rel_path])
        json_path = os.path.join(self.project_dir, "jsons", os.path.splitext(os.path.basename(rel_path))[0] + ".json")
        if not os.path.exists(json_path):
            self.image_label_cache[rel_path] = set()
            return set()
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            labels = {
                ann.get('lable') or ann.get('label') or ann.get('category')
                for ann in data.get("annotations", [])
                if ann.get('lable') or ann.get('label') or ann.get('category')
            }
            self.image_label_cache[rel_path] = set(labels)
            return labels
        except Exception:
            self.image_label_cache[rel_path] = set()
            return set()

    def invalidate_image_label_cache(self, rel_path=None):
        if rel_path:
            self.image_label_cache.pop(rel_path, None)
        else:
            self.image_label_cache.clear()

    def get_current_annotation_labels(self):
        return {ann.category for ann in self.graphicsView.imageItem.annotations if ann.category}

    def cleanup_invalid_assignments_for_path(self, rel_path, labels=None):
        if not rel_path or rel_path not in self.sample_assignments:
            return False
        labels = set(labels) if labels is not None else self.get_image_labels(rel_path)
        assignment = self.sample_assignments.get(rel_path, {})
        original_keys = set(assignment.keys())
        for tag in list(assignment.keys()):
            if tag not in labels:
                assignment.pop(tag, None)
        if assignment:
            self.sample_assignments[rel_path] = assignment
        else:
            self.sample_assignments.pop(rel_path, None)
        return set(assignment.keys()) != original_keys

    def cleanup_invalid_assignments(self):
        changed = False
        for rel_path in list(self.sample_assignments.keys()):
            if self.cleanup_invalid_assignments_for_path(rel_path):
                changed = True
        if changed:
            self.persist_project_state()
        return changed

    def batch_run(self):
        """切换自动遍历：首次点击开始，再次点击立即停止。"""
        if self.timer1.isActive():
            self.stop_batch_run()
            return

        if self.sync_categories_from_data():
            self.ensure_sample_tree_integrity()
            self.persist_project_state()
            self.rebuild_sample_tree()

        if self.listWidget.count() == 0:
            QMessageBox.information(self, "自动遍历", "当前列表没有可遍历的图片。")
            return

        self.listWidget.setCurrentRow(-1)
        self.current_index = 0
        self.total_items = self.listWidget.count()
        self.run_action.setText("停止遍历")
        self.timer1.start()

    def stop_batch_run(self):
        """停止自动遍历，并恢复工具栏动作名称。"""
        if self.timer1.isActive():
            self.timer1.stop()
        self.run_action.setText("自动遍历")

    def process_next_item(self):
        if not self.timer1.isActive():
            return
        current_count = self.listWidget.count()
        if self.current_index < min(self.total_items, current_count):
            self.listWidget.setCurrentRow(self.current_index)
            self.current_index += 1
        else:
            if current_count > 0:
                current_row = self.listWidget.currentRow()
                if current_row < 0 or current_row >= current_count:
                    self.listWidget.setCurrentRow(current_count - 1)
            self.stop_batch_run()

    def import_good(self):
        """导入完全良品"""
        self.is_good = True
        self.import_good_flag = 2
        self.import_image_directory()
        self.is_good = False
        self.import_good_flag = 0

    # ================= 标注 / YOLO 数据集导入 =================

    @staticmethod
    def _import_value(source, name, default=None):
        """同时兼容导入服务返回的 dataclass 和 dict。"""
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    @staticmethod
    def _import_messages(value):
        def format_message(item):
            if isinstance(item, dict):
                message = str(item.get('message') or item.get('code') or '未知导入问题').strip()
                path = item.get('path')
                line = item.get('line')
                location = ''
                if path:
                    location = os.path.basename(str(path)) or str(path)
                if line is not None:
                    location = f"{location} 第 {line} 行".strip()
                return f"{location}：{message}" if location else message
            return str(item).strip()

        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return [message for item in value if (message := format_message(item))]
        except TypeError:
            message = format_message(value)
            return [message] if message else []

    @staticmethod
    def _annotation_stem(path):
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def _normalise_import_key(value):
        return os.path.normcase(str(value)).lower()

    def _get_annotation_importer(self):
        """延迟导入，避免程序启动时因可选导入服务而失败。"""
        try:
            from Utils.AnnotationImporter import AnnotationImporter
            return AnnotationImporter
        except Exception as exc:
            QMessageBox.critical(
                self,
                "导入功能不可用",
                f"无法加载标注导入服务：{exc}"
            )
            return None

    def _flush_current_annotation_before_import(self):
        """先落盘当前画布，避免刷新列表时覆盖刚导入的标注。"""
        item = self.listWidget.currentItem()
        if not item:
            return

        try:
            self.graphicsView.imageItem.cancel_current_annotation()
            self.save_current_image_annotations()
        except Exception as exc:
            raise RuntimeError(f"保存当前图片标注失败：{exc}") from exc

        index = self.get_item_global_index(item)
        if 0 <= index < len(self.flag) and self.flag[index] not in [2, 3]:
            if self.graphicsView.imageItem.annotations:
                self.flag[index] = 1
            else:
                self.flag[index] = 0
        self.persist_project_state()

    def _project_image_stem_map(self):
        result = {}
        for rel_path in self.relative_paths:
            key = self._normalise_import_key(self._annotation_stem(rel_path))
            result.setdefault(key, []).append(rel_path)
        return result

    def _existing_project_import_keys(self):
        """读取磁盘中的残留图片/JSON，不能只依赖 datafile.dat。"""
        image_stems = set()
        image_names = set()
        json_stems = set()
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        if os.path.isdir(self.image_dir):
            for entry in os.scandir(self.image_dir):
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in image_exts:
                    image_names.add(self._normalise_import_key(entry.name))
                    image_stems.add(self._normalise_import_key(self._annotation_stem(entry.name)))
        if os.path.isdir(self.json_dir):
            for entry in os.scandir(self.json_dir):
                if entry.is_file() and entry.name.lower().endswith('.json'):
                    json_stems.add(self._normalise_import_key(self._annotation_stem(entry.name)))
        return image_stems, image_names, json_stems

    @staticmethod
    def _annotation_labels(json_data):
        if not isinstance(json_data, dict):
            return []
        labels = []
        seen = set()
        for annotation in json_data.get('annotations', []):
            if not isinstance(annotation, dict):
                continue
            label = annotation.get('lable') or annotation.get('label') or annotation.get('category')
            label = str(label).strip() if label is not None else ''
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    @staticmethod
    def _status_for_import(status, json_data, empty_default=0):
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None
        if status in [0, 1, 2, 3]:
            return status
        if isinstance(json_data, dict) and json_data.get('annotations'):
            return 1
        return empty_default

    def _add_imported_categories(self, categories):
        current = list(self.graphicsView.imageItem.existing_categories)
        known = set(current)
        for category in categories:
            category = str(category).strip()
            if category and category not in known:
                current.append(category)
                known.add(category)
        if current != self.graphicsView.imageItem.existing_categories:
            self.graphicsView.imageItem.initCatories(current)

    def _write_annotation_json(self, target_path, json_data):
        """用同目录临时文件写入，写入异常时不破坏已有 JSON。"""
        parent = os.path.dirname(target_path)
        os.makedirs(parent, exist_ok=True)
        temporary_path = target_path + '.importing'
        try:
            with open(temporary_path, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, indent=2, ensure_ascii=False)
            os.replace(temporary_path, target_path)
        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    def _backup_annotation_jsons(self, json_paths):
        """覆盖前创建一次带时间戳的备份目录，返回目录和失败信息。"""
        existing_paths = [path for path in json_paths if os.path.isfile(path)]
        if not existing_paths:
            return None, []

        timestamp = QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss_zzz')
        backup_dir = os.path.join(self.json_dir, '_import_backups', timestamp)
        failures = []
        try:
            os.makedirs(backup_dir, exist_ok=False)
        except FileExistsError:
            backup_dir = backup_dir + '_1'
            os.makedirs(backup_dir, exist_ok=False)

        for source_path in existing_paths:
            try:
                shutil.copy2(source_path, os.path.join(backup_dir, os.path.basename(source_path)))
            except OSError as exc:
                failures.append(f"{os.path.basename(source_path)}：{exc}")
        return backup_dir, failures

    def _choose_json_overwrite_policy(self, existing_count):
        if not existing_count:
            return 'overwrite'
        reply = QMessageBox.question(
            self,
            "发现已有标注",
            f"匹配到 {existing_count} 个已有 JSON 标注。\n\n"
            "选择“是”将覆盖这些标注，并在覆盖前自动备份；\n"
            "选择“否”将跳过已有标注；选择“取消”将终止导入。",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            return 'overwrite'
        if reply == QMessageBox.No:
            return 'skip'
        return 'cancel'

    def _find_list_item_by_rel_path(self, rel_path):
        if not rel_path:
            return None
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            if self.get_item_relative_path(item) == rel_path:
                return item
        return None

    def _refresh_after_external_import(self, preferred_rel_path=None):
        """刷新树、列表、统计，并仅按新 JSON 重新加载一次当前图片。"""
        self.invalidate_image_label_cache()
        self.rebuild_path_index()
        self.total = len(self.relative_paths)
        self.ensure_sample_tree_integrity()
        self.persist_project_state()

        # 清空旧画布后阻断列表信号，防止 clear() 触发 show_image() 回写旧标注。
        self.graphicsView.imageItem.cancel_current_annotation()
        self.graphicsView.imageItem.remove_annotations()
        blocker = QSignalBlocker(self.listWidget)
        self.refresh_image_list(preserve_display=True)
        target_item = self._find_list_item_by_rel_path(preferred_rel_path)
        if target_item is None and self.listWidget.count() > 0:
            target_item = self.listWidget.item(0)
        if target_item is not None:
            self.listWidget.setCurrentItem(target_item)
        del blocker

        # 这里直接调用，pre=None，保证不会把旧画布内容保存回 JSON。
        self.show_image(target_item, None)

    def _show_import_summary(self, title, lines, details=None, warning=False):
        message_lines = [str(line) for line in lines if str(line).strip()]
        detail_lines = self._import_messages(details)
        if detail_lines:
            shown = detail_lines[:20]
            message_lines.extend(['', '详情：'] + [f"- {line}" for line in shown])
            if len(detail_lines) > len(shown):
                message_lines.append(f"- 其余 {len(detail_lines) - len(shown)} 项未显示")
        message = '\n'.join(message_lines)
        if warning:
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def import_json_annotations(self):
        """将外部 JSON 标注按同名图片导入当前工程。"""
        if not self.relative_paths:
            QMessageBox.information(self, "导入标注", "当前工程还没有图片，请先导入图片。")
            return

        source_dir = QFileDialog.getExistingDirectory(self, "选择 JSON 标注目录")
        if not source_dir:
            return

        importer = self._get_annotation_importer()
        if importer is None:
            return

        try:
            self._flush_current_annotation_before_import()
        except RuntimeError as exc:
            QMessageBox.critical(self, "导入标注失败", str(exc))
            return

        json_files = []
        for root, directories, files in os.walk(source_dir):
            # 避免用户选择项目 jsons 时把本软件的历史备份再次导入。
            directories[:] = [
                directory for directory in directories
                if directory != '_import_backups' and not directory.startswith('.')
            ]
            for filename in files:
                if filename.lower().endswith('.json'):
                    json_files.append(os.path.join(root, filename))
        json_files.sort(key=lambda path: os.path.normcase(path))

        if not json_files:
            QMessageBox.information(self, "导入标注", "所选目录中没有 JSON 文件。")
            return

        project_stems = self._project_image_stem_map()
        source_by_stem = {}
        duplicate_source_stems = set()
        for source_path in json_files:
            key = self._normalise_import_key(self._annotation_stem(source_path))
            if key in source_by_stem:
                duplicate_source_stems.add(key)
            else:
                source_by_stem[key] = source_path

        skipped_no_image = 0
        skipped_ambiguous = 0
        skipped_duplicates = 0
        parse_errors = []
        candidates = []
        for key, source_path in source_by_stem.items():
            if key in duplicate_source_stems:
                skipped_duplicates += 1
                continue
            matched_paths = project_stems.get(key, [])
            if not matched_paths:
                skipped_no_image += 1
                continue
            if len(matched_paths) != 1:
                skipped_ambiguous += 1
                continue
            try:
                json_data = importer.parse_json_annotation_file(source_path)
            except Exception as exc:
                parse_errors.append(f"{os.path.basename(source_path)}：{exc}")
                continue
            if not isinstance(json_data, dict):
                parse_errors.append(f"{os.path.basename(source_path)}：导入服务未返回有效 JSON")
                continue
            if not isinstance(json_data.get('annotations', []), list):
                parse_errors.append(f"{os.path.basename(source_path)}：annotations 必须是列表")
                continue
            rel_path = matched_paths[0]
            candidates.append({
                'source_path': source_path,
                'rel_path': rel_path,
                'target_path': self.annotation_json_path_for_rel_path(rel_path),
                'json_data': json_data,
            })

        if not candidates:
            self._show_import_summary(
                "导入标注未完成",
                [
                    "没有找到可导入的同名 JSON 标注。",
                    f"未匹配图片：{skipped_no_image}",
                    f"图片名歧义：{skipped_ambiguous}",
                    f"重复 JSON 名：{skipped_duplicates}",
                    f"格式错误：{len(parse_errors)}",
                ],
                parse_errors,
                warning=True
            )
            return

        existing_candidates = [item for item in candidates if os.path.isfile(item['target_path'])]
        policy = self._choose_json_overwrite_policy(len(existing_candidates))
        if policy == 'cancel':
            return
        if policy == 'skip':
            candidate_targets = {item['target_path'] for item in existing_candidates}
            candidates = [item for item in candidates if item['target_path'] not in candidate_targets]

        if not candidates:
            self._show_import_summary(
                "导入标注未完成",
                ["所有同名标注都已跳过。"],
                warning=True
            )
            return

        backup_dir = None
        backup_errors = []
        if policy == 'overwrite':
            overwrite_paths = [item['target_path'] for item in candidates if os.path.isfile(item['target_path'])]
            try:
                backup_dir, backup_errors = self._backup_annotation_jsons(overwrite_paths)
            except OSError as exc:
                QMessageBox.critical(self, "导入标注失败", f"创建覆盖备份失败：{exc}")
                return
            if backup_errors:
                self._show_import_summary(
                    "导入标注已取消",
                    ["部分旧标注无法备份，为保护原数据，本次没有覆盖。"],
                    backup_errors,
                    warning=True
                )
                return

        imported = 0
        write_errors = []
        labels = []
        preferred_rel_path = self.get_item_relative_path(self.listWidget.currentItem())
        for item in candidates:
            try:
                self._write_annotation_json(item['target_path'], item['json_data'])
                rel_path = item['rel_path']
                index = self.get_global_index_from_path(rel_path)
                if 0 <= index < len(self.flag):
                    self.flag[index] = self._status_for_import(None, item['json_data'])
                self.sample_assignments.pop(rel_path, None)
                labels.extend(self._annotation_labels(item['json_data']))
                imported += 1
                if preferred_rel_path is None:
                    preferred_rel_path = rel_path
            except Exception as exc:
                write_errors.append(f"{os.path.basename(item['source_path'])}：{exc}")

        if imported:
            self._add_imported_categories(labels)
            self._refresh_after_external_import(preferred_rel_path)

        summary = [
            f"已导入标注：{imported}",
            f"未匹配图片：{skipped_no_image}",
            f"图片名歧义：{skipped_ambiguous}",
            f"重复 JSON 名：{skipped_duplicates}",
            f"格式错误：{len(parse_errors)}",
            f"写入失败：{len(write_errors)}",
        ]
        if backup_dir:
            summary.append(f"旧标注备份：{backup_dir}")
        self._show_import_summary(
            "导入标注完成" if imported else "导入标注未完成",
            summary,
            parse_errors + write_errors,
            warning=not imported or bool(parse_errors or write_errors)
        )

    def _inspection_category_names(self, inspection):
        for key in ('categories', 'class_names', 'names'):
            value = self._import_value(inspection, key, None)
            if value is None:
                continue
            if isinstance(value, dict):
                try:
                    return [str(value[index]).strip() for index in sorted(value, key=lambda item: int(item))]
                except (TypeError, ValueError):
                    return [str(item).strip() for item in value.values()]
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value]
        return []

    def _copy_dataset_entry(self, source_path, target_image_path, target_json_path, json_data):
        """先复制到临时文件，再写入目标，避免留下半成品。"""
        image_temp_path = target_image_path + '.importing'
        json_temp_path = target_json_path + '.importing'
        committed_paths = []
        os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(target_json_path), exist_ok=True)
        try:
            shutil.copy2(source_path, image_temp_path)
            with open(json_temp_path, 'w', encoding='utf-8') as file:
                json.dump(json_data, file, indent=2, ensure_ascii=False)
            os.replace(image_temp_path, target_image_path)
            committed_paths.append(target_image_path)
            os.replace(json_temp_path, target_json_path)
        except Exception:
            # 预检已保证目标不存在；若第二次 replace 失败，也回滚刚移动的图片。
            for path in (image_temp_path, json_temp_path, *committed_paths):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            raise

    @staticmethod
    def _dataset_json_matches(target_json_path, expected_json):
        """只把内容完全相同的 JSON 视为上次中断导入留下的文件。"""
        try:
            with open(target_json_path, 'r', encoding='utf-8-sig') as file:
                return json.load(file) == expected_json
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False

    def _is_recoverable_dataset_entry(
            self, source_path, target_image_path, target_json_path, expected_json):
        """确认未登记文件确实是本次同一条数据集导入的遗留物。"""
        if not (os.path.isfile(target_image_path) and os.path.isfile(target_json_path)):
            return False
        try:
            image_matches = filecmp.cmp(source_path, target_image_path, shallow=False)
        except OSError:
            return False
        return image_matches and self._dataset_json_matches(target_json_path, expected_json)

    def import_yolo_dataset(self):
        """导入包含 images、labels、data.yaml 的 YOLO 数据集。"""
        dataset_root = QFileDialog.getExistingDirectory(self, "选择 YOLO 数据集目录")
        if not dataset_root:
            return

        importer = self._get_annotation_importer()
        if importer is None:
            return

        try:
            inspection = importer.inspect_yolo_dataset(dataset_root)
        except Exception as exc:
            QMessageBox.critical(self, "导入数据集失败", f"数据集检查失败：{exc}")
            return

        global_errors = self._import_messages(self._import_value(inspection, 'errors', []))
        global_warnings = self._import_messages(self._import_value(inspection, 'warnings', []))
        fatal_errors = self._import_messages(self._import_value(inspection, 'fatal_errors', []))
        entries = (
            self._import_value(inspection, 'entries', None)
            or self._import_value(inspection, 'records', [])
            or []
        )
        if fatal_errors:
            self._show_import_summary(
                "导入数据集未完成",
                ["数据集目录结构或 data.yaml 无效，未写入任何文件。"],
                fatal_errors + global_warnings,
                warning=True
            )
            return
        if not entries:
            self._show_import_summary(
                "导入数据集未完成",
                ["没有发现可导入的图片。"],
                global_errors + global_warnings,
                warning=True
            )
            return

        try:
            self._flush_current_annotation_before_import()
        except RuntimeError as exc:
            QMessageBox.critical(self, "导入数据集失败", str(exc))
            return

        registered_stems = set(self._project_image_stem_map().keys())
        registered_names = {
            self._normalise_import_key(os.path.basename(rel_path))
            for rel_path in self.relative_paths
        }
        existing_stems, existing_names, existing_json_stems = self._existing_project_import_keys()
        existing_stems.update(registered_stems)
        source_stem_counts = {}
        source_name_counts = {}
        prepared_entries = []
        skipped_invalid = []

        for entry in entries:
            entry_errors = self._import_messages(self._import_value(entry, 'errors', []))
            entry_can_import = self._import_value(entry, 'can_import', True)
            image_path = self._import_value(entry, 'image_path', None)
            image_path = os.path.normpath(str(image_path)) if image_path else ''
            image_name = os.path.basename(image_path)
            stem = self._import_value(entry, 'stem', None) or self._annotation_stem(image_name)
            json_data = self._import_value(entry, 'json_data', None)
            if entry_can_import is False or entry_errors or not image_path or not os.path.isfile(image_path):
                reason = '; '.join(entry_errors) if entry_errors else '数据集检查未通过'
                if not image_path or not os.path.isfile(image_path):
                    reason = '图片文件不存在'
                skipped_invalid.append(f"{image_name or '未知图片'}：{reason}")
                continue
            if not isinstance(json_data, dict) or not isinstance(json_data.get('annotations', []), list):
                skipped_invalid.append(f"{image_name}：缺少有效的转换标注数据")
                continue
            stem_key = self._normalise_import_key(stem)
            name_key = self._normalise_import_key(image_name)
            source_stem_counts[stem_key] = source_stem_counts.get(stem_key, 0) + 1
            source_name_counts[name_key] = source_name_counts.get(name_key, 0) + 1
            prepared_entries.append({
                'entry': entry,
                'image_path': image_path,
                'image_name': image_name,
                'stem': str(stem),
                'stem_key': stem_key,
                'name_key': name_key,
                'json_data': json_data,
            })

        source_duplicate_keys = {
            key for key, count in source_stem_counts.items() if count > 1
        }
        source_duplicate_names = {
            key for key, count in source_name_counts.items() if count > 1
        }
        skipped_duplicates = 0
        skipped_existing = 0
        importable_entries = []
        entry_warnings = []
        for item in prepared_entries:
            entry_warnings.extend(self._import_messages(self._import_value(item['entry'], 'warnings', [])))
            if item['stem_key'] in source_duplicate_keys or item['name_key'] in source_duplicate_names:
                skipped_duplicates += 1
                continue
            target_image_path = os.path.join(self.image_dir, item['image_name'])
            target_json_path = os.path.join(self.json_dir, item['stem'] + '.json')
            registered_conflict = (
                item['stem_key'] in registered_stems
                or item['name_key'] in registered_names
            )
            disk_conflict = (
                item['stem_key'] in existing_stems
                or item['stem_key'] in existing_json_stems
                or item['name_key'] in existing_names
                or os.path.exists(target_image_path)
                or os.path.exists(target_json_path)
            )
            if disk_conflict:
                # 兼容旧版首次导入失败：文件已经复制，但 datafile.dat 仍是空 dict。
                # 只有图片字节和 JSON 语义都与本次来源完全相同，才安全登记而不覆盖。
                if not registered_conflict and self._is_recoverable_dataset_entry(
                        item['image_path'], target_image_path, target_json_path, item['json_data']):
                    item['recover_existing'] = True
                    importable_entries.append(item)
                    continue
                skipped_existing += 1
                continue
            importable_entries.append(item)

        if not importable_entries:
            self._show_import_summary(
                "导入数据集未完成",
                [
                    "没有可安全导入的新样本。",
                    f"已有样本冲突：{skipped_existing}",
                    f"数据集内重名：{skipped_duplicates}",
                    f"无效样本：{len(skipped_invalid)}",
                ],
                global_errors + global_warnings + skipped_invalid + entry_warnings,
                warning=True
            )
            return

        imported = 0
        recovered = 0
        copy_errors = []
        labels = self._inspection_category_names(inspection)
        preferred_rel_path = self.get_item_relative_path(self.listWidget.currentItem())
        progress = QProgressDialog("正在导入数据集…", "", 0, len(importable_entries), self)
        progress.setWindowTitle("导入数据集")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setValue(0)
        try:
            for number, item in enumerate(importable_entries, start=1):
                recovering = bool(item.get('recover_existing'))
                progress.setLabelText(
                    f"正在{'恢复' if recovering else '导入'} "
                    f"({number}/{len(importable_entries)})：{item['image_name']}"
                )
                QApplication.processEvents()
                target_image_path = os.path.join(self.image_dir, item['image_name'])
                target_json_path = os.path.join(self.json_dir, item['stem'] + '.json')
                old_relative_count = len(self.relative_paths)
                old_image_count = len(self.image_paths)
                old_flag_count = len(self.flag)
                old_label_count = len(labels)
                copied_now = False
                try:
                    if not recovering:
                        self._copy_dataset_entry(
                            item['image_path'], target_image_path, target_json_path, item['json_data']
                        )
                        copied_now = True
                    rel_path = os.path.relpath(target_image_path, self.project_dir)
                    self.relative_paths.append(rel_path)
                    self.image_paths.append(target_image_path)
                    status = self._status_for_import(
                        self._import_value(item['entry'], 'status', None),
                        item['json_data']
                    )
                    self.flag.append(status)
                    labels.extend(self._annotation_labels(item['json_data']))
                    imported += 1
                    if recovering:
                        recovered += 1
                    if preferred_rel_path is None:
                        preferred_rel_path = rel_path
                except Exception as exc:
                    # 若复制后登记工程状态失败，回滚本轮新写入文件和内存状态，避免再次留下孤儿文件。
                    del self.relative_paths[old_relative_count:]
                    del self.image_paths[old_image_count:]
                    del self.flag[old_flag_count:]
                    del labels[old_label_count:]
                    if copied_now:
                        for path in (target_image_path, target_json_path):
                            if os.path.isfile(path):
                                try:
                                    os.remove(path)
                                except OSError:
                                    pass
                    copy_errors.append(f"{item['image_name']}：{exc}")
                finally:
                    progress.setValue(number)
                    QApplication.processEvents()
        finally:
            progress.close()

        if imported:
            self._add_imported_categories(labels)
            self._refresh_after_external_import(preferred_rel_path)

        self._show_import_summary(
            "导入数据集完成" if imported else "导入数据集未完成",
            [
                f"已导入样本：{imported}",
                f"已有样本冲突：{skipped_existing}",
                f"数据集内重名：{skipped_duplicates}",
                f"无效样本：{len(skipped_invalid)}",
                f"已恢复上次中断样本：{recovered}",
                f"处理失败：{len(copy_errors)}",
            ],
            global_errors + global_warnings + skipped_invalid + entry_warnings + copy_errors,
            warning=not imported or bool(global_errors or skipped_invalid or copy_errors)
        )

    def set_batch_good(self):
        num = 180
        num = num - self.flag.count(2)
        if num <= 0: return
        index = 0
        while num and index < len(self.flag):
            if self.flag[index] == 0:
                self.set_good(self.listWidget.item(index), 2)
                num -= 1
            index += 1

    def set_norm(self):
        """清除良品状态"""
        for i in range(len(self.flag)):
            if self.flag[i] in [2, 3]:
                self.flag[i] = 0
        self.refresh_image_list()
        self.update_realtime_stats()

    def clear_json_annotations_for_item(self, item):
        rel_path = self.get_item_relative_path(item)
        if not rel_path:
            return
        json_path = os.path.join(self.json_dir, os.path.splitext(os.path.basename(rel_path))[0] + ".json")
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['annotations'] = []
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.invalidate_image_label_cache(rel_path)
        except Exception:
            pass

    def set_good(self, item, good_flag=2):
        """单张设为完全良品或过杀品"""
        index = self.get_item_global_index(item)
        if index < 0: return
        rel_path = self.get_item_relative_path(item)
        if item == self.listWidget.currentItem():
            self.graphicsView.imageItem.remove_annotations()
        self.clear_json_annotations_for_item(item)
        self.flag[index] = good_flag
        self.sample_assignments.pop(rel_path, None)
        item.setIcon(QIcon(ICON_GOOD))
        self.persist_project_state()
        if index + 1 < self.listWidget.count():
            self.listWidget.setCurrentRow(index + 1)
        self.refresh_image_list()
        self.update_realtime_stats()

    def delete_item(self, item):
        """物理删除与UI更新强绑定"""
        rel_path = self.get_item_relative_path(item)
        global_index = self.get_item_global_index(item)
        row = self.listWidget.row(item)
        p_img = self.resolve_image_path(rel_path)
        p_js = self.annotation_json_path_for_rel_path(rel_path)
        try:
            if p_img and os.path.exists(p_img):
                os.remove(p_img)
            if p_js and os.path.exists(p_js):
                os.remove(p_js)
            if rel_path in self.relative_paths:
                self.relative_paths.remove(rel_path)
            self.remove_image_path_cache(p_img)
            self.invalidate_image_label_cache(rel_path)
        except Exception as e:
            print(f"删除失败: {e}")
        finally:
            if 0 <= global_index < len(self.flag): self.flag.pop(global_index)
            self.sample_assignments.pop(rel_path, None)
            self.rebuild_path_index()
            self.total = len(self.relative_paths)
            self.persist_project_state()
            self.refresh_image_list()
            if row >= 0 and self.listWidget.count() > 0:
                self.listWidget.setCurrentRow(min(max(row, 0), self.listWidget.count() - 1))
            self.update_realtime_stats()

    def show_image(self, cur, pre):
        """展示选中图片"""
        if pre:
            pre_rel_path = self.get_item_relative_path(pre)
            p_idx = self.get_global_index_from_path(pre_rel_path)
            p_path = self.annotation_json_path_for_rel_path(pre_rel_path)
            pre_image_path = self.resolve_image_path(pre_rel_path)
            if 0 <= p_idx < len(self.flag) and pre_image_path and os.path.exists(pre_image_path):
                if len(self.graphicsView.imageItem.annotations) > 0:
                    self.flag[p_idx] = 1
                    pre.setIcon(QIcon(ICON_LABEL))
                elif self.flag[p_idx] not in [2, 3]:
                    self.flag[p_idx] = 0
                    pre.setIcon(QIcon())
                self.graphicsView.imageItem.save_annotations(p_path)
                self.invalidate_image_label_cache(pre_rel_path)

        self.scene.clear()
        if cur:
            path = self.resolve_image_path(cur.toolTip())
            if path and os.path.exists(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    self.graphicsView.imageItem.setPath(path)
                    self.graphicsView.imageItem.remove_annotations()
                    self.graphicsView.imageItem.setPixmap(pix)
                    self.graphicsView.onCenter()
                    c_name = os.path.splitext(os.path.basename(cur.toolTip()))[0]
                    cur_idx = self.get_item_global_index(cur)
                    if not (0 <= cur_idx < len(self.flag) and self.flag[cur_idx] in [2, 3]):
                        self.graphicsView.imageItem.load_annotations(os.path.join(self.json_dir, c_name + '.json'))
        self.update_realtime_stats()

    def update_realtime_stats(self):
        """实时统计更新"""
        self.sync_sample_tree_categories()
        anns = self.graphicsView.imageItem.annotations
        counts = {}
        for a in anns:
            n = a.category if a.category else "未命名"
            counts[n] = counts.get(n, 0) + 1
        self.bad = self.flag.count(1)
        self.complete_good = self.flag.count(2)
        self.overkill_good = self.flag.count(3)
        self.good = self.complete_good + self.overkill_good
        lb = self.bad + self.good
        self.rebuild_sample_tree()
        self.textBrowser.clear()
        self.textBrowser.append(
            f"图像总计: {self.total}\n已处理: {lb}\n坏品: {self.bad}\n完全良品: {self.complete_good}\n过杀品: {self.overkill_good}\n未标注: {self.total - lb}\n\n---当前图片详情---\n")
        if not counts: self.textBrowser.append("无标注内容")
        for k, v in counts.items(): self.textBrowser.append(f"{k}: {v}")
        # --- 新增这行：每次更新统计时，顺便重绘右侧面板，让新标签立刻显现 ---
        self.set_fliter_Widget()
    def scanUpadte(self, i):
        """切换标签页更新预览图"""
        if i == 1:
            if os.path.exists(self.data_yaml):
                train_root = os.path.join(self.project_dir, "dataset/images/train")
                val_root = os.path.join(self.project_dir, "dataset/images/val")
                self.trainPath = []
                self.valPath = []
                if os.path.exists(train_root):
                    for p in os.listdir(train_root): self.trainPath.append(os.path.join(train_root, p))
                if os.path.exists(val_root):
                    for p in os.listdir(val_root): self.valPath.append(os.path.join(val_root, p))
                if self.trainPath:
                    self.trainImg.setImage(self.trainPath[0])
                    bn = os.path.splitext(os.path.basename(self.trainPath[0]))[0]
                    self.trainImg.imageItem.load_annotations(os.path.join(self.json_dir, bn + '.json'))
                if self.valPath:
                    self.valImg.setImage(self.valPath[0])
                    bn = os.path.splitext(os.path.basename(self.valPath[0]))[0]
                    self.valImg.imageItem.load_annotations(os.path.join(self.json_dir, bn + '.json'))

    def staticData(self):
        """统计分析图表"""
        lables = {}
        subcategories = {}
        for img in self.image_paths:
            bn = os.path.splitext(os.path.basename(img))[0]
            jp = os.path.join(self.json_dir, bn + '.json')
            if os.path.exists(jp):
                try:
                    with open(jp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for a in data.get("annotations", []):
                            tag = a.get('lable') or a.get('label')
                            if tag: lables[tag] = lables.get(tag, 0) + 1
                except:
                    continue
        for rel_path, assignment in self.sample_assignments.items():
            for tag, child in assignment.items():
                key = f"{tag}/{child}"
                subcategories[key] = subcategories.get(key, 0) + 1
        if lables:
            self.chart = ChartApp()
            dimension_data = {
                '标签维度统计': {
                    'data': lables,
                    'title': '标签维度统计',
                    'xLabel': '标签名称',
                    'yLabel': '标注数量',
                }
            }
            if subcategories:
                dimension_data['子样本分类统计'] = {
                    'data': subcategories,
                    'title': '子样本分类统计',
                    'xLabel': '子样本分类',
                    'yLabel': '图片数量',
                }
            self.chart.set_dimension_data(dimension_data)
            self.chart.show()
        else:
            QMessageBox.information(self, "提示", "暂无有效标注数据")

    def format_bytes(self, size):
        value = float(size)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if value < 1024 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024

    def count_files_by_suffix(self, folder, suffixes):
        if not os.path.exists(folder):
            return 0
        suffixes = {x.lower() for x in suffixes}
        count = 0
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in suffixes:
                        count += 1
        except Exception:
            return 0
        return count

    def get_project_disk_info(self):
        try:
            usage = shutil.disk_usage(os.path.abspath(self.project_dir))
            return self.format_bytes(usage.free)
        except Exception:
            return "无法读取"

    def get_folder_size(self, folder):
        total = 0
        if not os.path.exists(folder):
            return total
        for root, _, files in os.walk(folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    total += os.path.getsize(file_path)
                except OSError:
                    continue
        return total

    def build_project_health_report(self):
        image_count = self.count_files_by_suffix(self.image_dir, ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])
        json_count = self.count_files_by_suffix(self.json_dir, ['.json'])
        managed_count = len(self.relative_paths)
        status_counts = self.get_status_counts()
        tag_names = list(self.graphicsView.imageItem.existing_categories)
        free_space = self.get_project_disk_info()
        images_size = self.format_bytes(self.get_folder_size(self.image_dir))
        dataset_dir = os.path.join(self.project_dir, "dataset")
        dataset_exists = os.path.exists(dataset_dir)

        lines = [
            "项目体检报告",
            "",
            "【基本信息】",
            f"工程路径：{self.project_dir}",
            f"工程记录图片数：{managed_count}",
            f"images 目录图片数：{image_count}",
            f"JSON 标注文件数：{json_count}",
            f"已有 dataset 目录：{'是，重新生成数据集会覆盖旧 dataset' if dataset_exists else '否'}",
            "",
            "【样本状态】",
            f"坏品：{status_counts['bad']}",
            f"完全良品：{status_counts['complete_good']}",
            f"过杀品：{status_counts['overkill_good']}",
            f"未标注：{status_counts['unlabeled']}",
            "",
            "【标签类别】",
            f"类别数量：{len(tag_names)}",
        ]
        if tag_names:
            for i, tag in enumerate(tag_names, 1):
                lines.append(f"{i}. {tag}")
        else:
            lines.append("暂无标签类别")

        lines.extend([
            "",
            "【磁盘空间】",
            f"工程所在磁盘剩余空间：{free_space}",
            f"本项目工程 images 所占空间：{images_size}",
            "",
            "【提示】",
        ])
        if managed_count != image_count:
            lines.append("- 工程记录图片数与 images 目录图片数不一致，建议复查是否有图片未导入或被手动移动。")
        if json_count == 0:
            lines.append("- 当前未发现 JSON 标注文件。")
        if not tag_names:
            lines.append("- 当前未发现标签类别。")
        if dataset_exists:
            lines.append("- 当前工程已有 dataset 目录，重新生成数据集会删除并重建该目录。")
        if len(lines) >= 2 and lines[-1] == "【提示】":
            lines.append("暂无明显提醒。")
        return "\n".join(lines)

    def show_project_health_report(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("项目体检")
        dialog.setMinimumSize(760, 600)
        layout = QVBoxLayout(dialog)
        report = QTextBrowser(dialog)
        report.setPlainText(self.build_project_health_report())
        close_btn = QPushButton("关闭", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(report)
        layout.addWidget(close_btn)
        dialog.exec_()

    def set_fliter_Widget(self):
        """左侧树已经接管样本管理，右侧不再显示旧筛选工具栏"""
        if self.fliter.layout():
            old = self.fliter.layout()
            while old.count():
                it = old.takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
            QWidget().setLayout(old)
        self.fliter.hide()

    def current_image_json_path(self):
        item = self.listWidget.currentItem()
        if not item:
            return None
        return self.annotation_json_path_for_rel_path(item.toolTip())

    def save_current_image_annotations(self):
        json_path = self.current_image_json_path()
        if json_path:
            self.graphicsView.imageItem.save_annotations(json_path)
            item = self.listWidget.currentItem()
            rel_path = self.get_item_relative_path(item)
            self.invalidate_image_label_cache(rel_path)
            if self.cleanup_invalid_assignments_for_path(rel_path, self.get_current_annotation_labels()):
                self.persist_project_state()

    def on_annotation_deleted(self):
        self.save_current_image_annotations()
        item = self.listWidget.currentItem()
        if not item:
            return
        idx = self.get_item_global_index(item)
        if 0 <= idx < len(self.flag) and self.flag[idx] not in [2, 3]:
            if self.graphicsView.imageItem.annotations:
                self.flag[idx] = 1
                item.setIcon(QIcon(ICON_LABEL))
            else:
                self.flag[idx] = 0
                item.setIcon(QIcon())
            self.persist_project_state()

    def show_tag_context_menu(self, pos):
        if not hasattr(self, 'tag_list_widget'):
            return
        item = self.tag_list_widget.itemAt(pos)
        menu = QMenu(self)
        add_action = QAction("新增标签", self)
        add_action.triggered.connect(self.add_project_tag)
        menu.addAction(add_action)

        if item is not None:
            self.tag_list_widget.setCurrentItem(item)
            rename_action = QAction("修改标签", self)
            delete_action = QAction("删除标签", self)
            rename_action.triggered.connect(self.rename_project_tag)
            delete_action.triggered.connect(self.delete_project_tag)
            menu.addAction(rename_action)
            menu.addAction(delete_action)

        menu.exec_(self.tag_list_widget.mapToGlobal(pos))

    def prompt_tag_name(self, title, label, text=""):
        value, ok = QInputDialog.getText(self, title, label, text=text)
        if not ok:
            return None
        value = value.strip()
        if not value:
            QMessageBox.information(self, "提示", "标签名称不能为空")
            return None
        return value

    def add_project_tag(self):
        new_name = self.prompt_tag_name("新增标签", "请输入标签名称:")
        if not new_name:
            return
        if new_name in self.graphicsView.imageItem.existing_categories:
            QMessageBox.information(self, "提示", "标签名称已存在")
            return
        self.graphicsView.imageItem.existing_categories.append(new_name)
        self.persist_project_state()
        self.set_fliter_Widget()
        self.rebuild_sample_tree()

    def rename_project_tag(self, old_name=None):
        if old_name is None:
            current_item = self.sampleTree.currentItem() if hasattr(self, 'sampleTree') else None
            if not current_item:
                QMessageBox.information(self, "提示", "请先选择一个缺陷")
                return
            key = current_item.data(0, Qt.UserRole)
            if not key or key[0] != "tag":
                QMessageBox.information(self, "提示", "请先选择一个缺陷")
                return
            old_name = key[1]
        new_name = self.prompt_tag_name("修改标签", "请输入新的标签名称:", old_name)
        if not new_name or new_name == old_name:
            return
        if new_name in self.graphicsView.imageItem.existing_categories:
            QMessageBox.information(self, "提示", "新的标签名称已存在")
            return

        self.save_current_image_annotations()
        self.replace_tag_in_jsons(old_name, new_name)
        self.invalidate_image_label_cache()

        categories = self.graphicsView.imageItem.existing_categories
        idx = categories.index(old_name)
        categories[idx] = new_name
        self.sample_groups[new_name] = self.sample_groups.pop(old_name, [])
        for assignment in self.sample_assignments.values():
            if old_name in assignment:
                assignment[new_name] = assignment.pop(old_name)
        for ann in self.graphicsView.imageItem.annotations:
            if ann.category == old_name:
                ann.set_category(new_name)
        self.persist_project_state()
        self.set_fliter_Widget()
        if self.current_tree_filter == ("tag", old_name, None):
            self.current_tree_filter = ("tag", new_name, None)
        elif self.current_tree_filter[0] == "child" and self.current_tree_filter[1] == old_name:
            self.current_tree_filter = ("child", new_name, self.current_tree_filter[2])
        self.rebuild_sample_tree()
        self.refresh_image_list()
        self.update_realtime_stats()

    def delete_project_tag(self, tag_name=None):
        if tag_name is None:
            current_item = self.sampleTree.currentItem() if hasattr(self, 'sampleTree') else None
            if not current_item:
                QMessageBox.information(self, "提示", "请先选择一个缺陷")
                return
            key = current_item.data(0, Qt.UserRole)
            if not key or key[0] != "tag":
                QMessageBox.information(self, "提示", "请先选择一个缺陷")
                return
            tag_name = key[1]
        self.save_current_image_annotations()
        used_count = self.count_tag_usage(tag_name)

        if used_count > 0:
            reply = QMessageBox.question(
                self,
                "删除确认",
                f"删除标签【{tag_name}】将会删除该标签对应的所有标注，是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            self.delete_tag_from_jsons(tag_name)
            self.invalidate_image_label_cache()

        self.graphicsView.imageItem.existing_categories = [
            tag for tag in self.graphicsView.imageItem.existing_categories if tag != tag_name
        ]
        self.sample_groups.pop(tag_name, None)
        for rel_path, assignment in list(self.sample_assignments.items()):
            assignment.pop(tag_name, None)
            if not assignment:
                self.sample_assignments.pop(rel_path, None)
        remaining_annotations = []
        scene = self.graphicsView.imageItem.scene()
        for ann in self.graphicsView.imageItem.annotations:
            if ann.category == tag_name:
                if scene:
                    scene.removeItem(ann)
            else:
                remaining_annotations.append(ann)
        self.graphicsView.imageItem.annotations = remaining_annotations
        self.persist_project_state()
        self.set_fliter_Widget()
        if self.current_tree_filter == ("tag", tag_name, None) or (
            self.current_tree_filter[0] == "child" and self.current_tree_filter[1] == tag_name
        ):
            self.current_tree_filter = ("all", None, None)
        self.rebuild_sample_tree()
        self.refresh_image_list()
        self.update_realtime_stats()

    def count_tag_usage(self, tag_name):
        count = 0
        if not os.path.exists(self.json_dir):
            return count
        for file_name in os.listdir(self.json_dir):
            if not file_name.endswith('.json'):
                continue
            json_path = os.path.join(self.json_dir, file_name)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                count += sum(1 for ann in data.get('annotations', []) if (ann.get('lable') or ann.get('label')) == tag_name)
            except Exception:
                continue
        return count

    def replace_tag_in_jsons(self, old_name, new_name):
        if not os.path.exists(self.json_dir):
            return
        for file_name in os.listdir(self.json_dir):
            if not file_name.endswith('.json'):
                continue
            json_path = os.path.join(self.json_dir, file_name)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                changed = False
                for ann in data.get('annotations', []):
                    if ann.get('lable') == old_name:
                        ann['lable'] = new_name
                        changed = True
                    elif ann.get('label') == old_name:
                        ann['label'] = new_name
                        changed = True
                if changed:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                continue

    def replace_tag_in_jsons_for_paths(self, rel_paths, old_name, new_name):
        if not os.path.exists(self.json_dir):
            return 0, 0
        changed_files = 0
        changed_annotations = 0
        seen = set()
        for rel_path in rel_paths:
            if rel_path in seen:
                continue
            seen.add(rel_path)
            json_path = os.path.join(self.json_dir, os.path.splitext(os.path.basename(rel_path))[0] + ".json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                changed = False
                for ann in data.get('annotations', []):
                    ann_changed = False
                    if ann.get('lable') == old_name:
                        ann['lable'] = new_name
                        ann_changed = True
                    if ann.get('label') == old_name:
                        ann['label'] = new_name
                        ann_changed = True
                    if ann_changed:
                        changed = True
                        changed_annotations += 1
                if changed:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    changed_files += 1
            except Exception:
                continue
        return changed_files, changed_annotations

    def delete_tag_from_jsons(self, tag_name):
        if not os.path.exists(self.json_dir):
            return
        for file_name in os.listdir(self.json_dir):
            if not file_name.endswith('.json'):
                continue
            json_path = os.path.join(self.json_dir, file_name)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                annotations = data.get('annotations', [])
                filtered = [ann for ann in annotations if (ann.get('lable') or ann.get('label')) != tag_name]
                if len(filtered) != len(annotations):
                    data['annotations'] = filtered
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                continue

    def fliter_images(self):
        """按标签过滤列表"""
        sel = self.get_selected()
        if not sel: return
        res = []
        for f in self.relative_paths:
            bn = os.path.splitext(os.path.basename(f))[0];
            jp = os.path.join(self.json_dir, bn + ".json")
            try:
                with open(jp, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                    tags = [a.get('lable') or a.get('label') for a in data.get('annotations', [])]
                    if set(tags) & set(sel): res.append(f)
            except:
                continue
        self.listWidget.clear()
        for p in res:
            it = QListWidgetItem(os.path.basename(p));
            it.setToolTip(p);
            self.listWidget.addItem(it)
        if res: self.listWidget.setCurrentRow(0)

    def fliter_index(self, content):
        temp = []
        selected = set(content)
        for i, filename in enumerate(self.relative_paths):
            if self.get_image_labels(filename) & selected:
                temp.append(i)
        return temp

    def count_train_detection_annotations(self, train_indices, selected_tags):
        rect_counts = {tag: 0 for tag in selected_tags}
        polygon_counts = {tag: 0 for tag in selected_tags}
        selected = set(selected_tags)
        for x in train_indices:
            if x >= len(self.relative_paths):
                continue
            rel_path = self.relative_paths[x]
            json_path = os.path.join(
                self.json_dir,
                os.path.splitext(os.path.basename(rel_path))[0] + ".json"
            )
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            for annotation in data.get('annotations', []):
                label_name = annotation.get('lable') or annotation.get('label')
                if label_name not in selected:
                    continue
                ann_type = annotation.get('type')
                if ann_type == 'rect' and len(annotation.get('points', [])) >= 2:
                    rect_counts[label_name] += 1
                elif ann_type == 'polygon':
                    polygon_counts[label_name] += 1
        return rect_counts, polygon_counts

    def show_detection_augment_dialog(self, selected_tags, rect_counts, polygon_counts):
        dialog = QDialog(self)
        dialog.setWindowTitle("检测增强数量设置")
        dialog.setMinimumSize(520, 420)
        main_lay = QVBoxLayout(dialog)

        desc = QLabel("旋转+平移增强只处理矩形检测框；多边形分割标注不会参与本次增强。增强数量表示每类新增多少张增强图。")
        desc.setWordWrap(True)
        main_lay.addWidget(desc)

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.addWidget(QLabel("缺陷类别"), 0, 0)
        grid.addWidget(QLabel("训练集检测框数量"), 0, 1)
        grid.addWidget(QLabel("新增增强图数量"), 0, 2)
        grid.addWidget(QLabel("提示"), 0, 3)

        spinboxes = {}
        for row, tag in enumerate(selected_tags, start=1):
            rect_count = rect_counts.get(tag, 0)
            polygon_count = polygon_counts.get(tag, 0)
            tag_label = QLabel(tag)
            count_label = QLabel(str(rect_count))
            spin = QSpinBox()
            spin.setRange(0, 100000)
            spin.setEnabled(rect_count > 0)
            note = ""
            if rect_count <= 0:
                note = "无矩形框，不能增强"
            elif polygon_count > 0:
                note = f"另有 {polygon_count} 个多边形已跳过"
            note_label = QLabel(note)
            note_label.setWordWrap(True)

            grid.addWidget(tag_label, row, 0)
            grid.addWidget(count_label, row, 1)
            grid.addWidget(spin, row, 2)
            grid.addWidget(note_label, row, 3)
            spinboxes[tag] = spin

        scroll.setWidget(container)
        main_lay.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        main_lay.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return {
            str(selected_tags.index(tag)): spin.value()
            for tag, spin in spinboxes.items()
            if spin.isEnabled() and spin.value() > 0
        }

    def generate(self):
        """生成数据集核心逻辑"""
        self.progressBar.setValue(0)
        if self.sync_categories_from_data():
            self.ensure_sample_tree_integrity()
            self.persist_project_state()
            self.rebuild_sample_tree()
        dialog = QDialog(self);
        dialog.setWindowTitle("导出数据集标签选择");
        dialog.setMinimumSize(400, 500)
        main_lay = QVBoxLayout(dialog)
        scroll = QScrollArea();
        scroll.setWidgetResizable(True)
        container = QWidget();
        container_lay = QVBoxLayout(container)

        self.temp_checkboxes = {}
        cb_all = QCheckBox("全选缺陷");
        container_lay.addWidget(cb_all)
        for tag in self.graphicsView.imageItem.existing_categories:
            cb = QCheckBox(tag);
            container_lay.addWidget(cb);
            self.temp_checkboxes[tag] = cb

        updating_all = False

        def set_all_categories(state):
            nonlocal updating_all
            if updating_all:
                return
            updating_all = True
            checked = state == Qt.Checked
            for checkbox in self.temp_checkboxes.values():
                checkbox.setChecked(checked)
            updating_all = False

        def update_select_all_state():
            nonlocal updating_all
            if updating_all:
                return
            states = [checkbox.isChecked() for checkbox in self.temp_checkboxes.values()]
            if states and all(states):
                state = Qt.Checked
            elif any(states):
                state = Qt.PartiallyChecked
            else:
                state = Qt.Unchecked
            updating_all = True
            cb_all.setCheckState(state)
            updating_all = False

        cb_all.stateChanged.connect(set_all_categories)
        for checkbox in self.temp_checkboxes.values():
            checkbox.stateChanged.connect(update_select_all_state)
        if self.temp_checkboxes:
            cb_all.setChecked(True)
        else:
            cb_all.setEnabled(False)

        container_lay.addSpacing(12)
        container_lay.addWidget(QLabel("良品样本（作为无标注负样本加入数据集）"))
        cb_complete_good = QCheckBox(f"完全良品 ({self.flag.count(2)})")
        cb_overkill_good = QCheckBox(f"过杀品 ({self.flag.count(3)})")
        cb_complete_good.setChecked(True)
        cb_overkill_good.setChecked(True)
        container_lay.addWidget(cb_complete_good)
        container_lay.addWidget(cb_overkill_good)

        container_lay.addSpacing(12)
        cb_detection_aug = QCheckBox("生成后执行旋转+平移检测增强（输出 dataset_aug）")
        cb_detection_aug.setToolTip("只处理矩形检测框；多边形分割标签不参与本次增强")
        container_lay.addWidget(cb_detection_aug)

        submit = QPushButton("确认生成数据集")

        def on_submit():
            has_selected_defect = any(
                checkbox.isChecked() for checkbox in self.temp_checkboxes.values()
            )
            has_selected_good = (
                cb_complete_good.isChecked() and self.flag.count(2) > 0
            ) or (
                cb_overkill_good.isChecked() and self.flag.count(3) > 0
            )
            if not has_selected_defect and not has_selected_good:
                QMessageBox.warning(dialog, "提示", "请至少选择一个有样本的缺陷或良品类别。")
                return
            dialog.accept()

        submit.clicked.connect(on_submit);
        main_lay.addWidget(scroll);
        main_lay.addWidget(submit)
        scroll.setWidget(container)

        if dialog.exec_() == QDialog.Accepted:
            selected_tags = [t for t, c in self.temp_checkboxes.items() if c.isChecked()]
            mode = 1 if selected_tags and len(selected_tags) == len(self.temp_checkboxes) else 2
            flag_to_use = self.flag if mode == 1 else self.fliter_index(selected_tags)
            good_indices = []
            if cb_complete_good.isChecked():
                good_indices.extend(np.where(np.array(self.flag) == 2)[0].tolist())
            if cb_overkill_good.isChecked():
                good_indices.extend(np.where(np.array(self.flag) == 3)[0].tolist())
            good_to_use = sorted(set(good_indices))
            candidate_indices = Task.resolve_bad_indices(flag_to_use, mode)
            image_labels_by_index = {
                index: self.get_image_labels(self.relative_paths[index])
                for index in candidate_indices
                if 0 <= index < len(self.relative_paths)
            }
            try:
                train_indices, val_indices, _ = Task.build_subsample_stratified_split(
                    flag_to_use,
                    mode,
                    self.relative_paths,
                    selected_tags,
                    self.sample_assignments,
                    good_to_use,
                    image_labels_by_index
                )
            except RuntimeError as e:
                QMessageBox.critical(self, "数据集划分失败", str(e))
                return
            augment_config = None
            if cb_detection_aug.isChecked():
                rect_counts, polygon_counts = self.count_train_detection_annotations(train_indices, selected_tags)
                target_counts = self.show_detection_augment_dialog(selected_tags, rect_counts, polygon_counts)
                if target_counts is None:
                    return
                if target_counts:
                    augment_config = {
                        "target_counts": target_counts,
                        "output_root": os.path.join(self.project_dir, "dataset_aug")
                    }
                else:
                    QMessageBox.information(self, "提示", "未填写增强数量，本次只生成原始 dataset。")
            worker = Task.Worker(
                self.project_dir,
                self.relative_paths,
                flag_to_use,
                selected_tags,
                mode,
                good_to_use,
                self.sample_groups,
                (train_indices, val_indices),
                augment_config,
                self.sample_assignments
            )
            worker.workSingals.sendValue.connect(self.progressBar.setValue)
            self.threadPool.start(worker)

    # ================= 辅助/基础逻辑 =================

    def import_image_directory(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if d:
            os.makedirs(self.image_dir, exist_ok=True);
            os.makedirs(self.json_dir, exist_ok=True)
            self.copy_images(d);
            self.save_image_paths();
            self.listWidget.setCurrentRow(0)

    def copy_images(self, src):
        self.count = 0;
        valid = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        if self.image_paths: self.mode = False
        for f in os.listdir(src):
            s_p = os.path.join(src, f)
            if not os.path.isfile(s_p):
                continue
            if os.path.splitext(f)[1].lower() in valid:
                d_p = os.path.join(self.image_dir, f)
                if d_p not in self.image_paths:
                    shutil.copy2(s_p, d_p);
                    self.count += 1;
                    self.image_paths.append(d_p)

    def save_image_paths(self):
        self.relative_paths = [os.path.relpath(p, self.project_dir) for p in self.image_paths]
        self.rebuild_path_index()
        self.total = len(self.relative_paths)
        if self.mode:
            self.flag = [self.import_good_flag if self.is_good else 0 for _ in range(self.total)]
        else:
            self.flag.extend([self.import_good_flag if self.is_good else 0 for _ in range(self.count)])
        self.update_realtime_stats()
        self.persist_project_state()
        self.ensure_sample_tree_integrity()
        self.rebuild_sample_tree()
        self.refresh_image_list()

    @staticmethod
    def _normalise_loaded_relative_paths(raw_paths):
        """将历史工程的图片记录迁移为当前统一使用的 list[str]。"""
        repaired = not isinstance(raw_paths, list)
        if isinstance(raw_paths, dict):
            # 最早的新建工程错误写入了空 dict；若极少数旧工程以路径为 key，保留这些 key。
            candidates = list(raw_paths.keys())
        elif isinstance(raw_paths, (list, tuple, set)):
            candidates = list(raw_paths)
        elif isinstance(raw_paths, (str, os.PathLike)):
            candidates = [raw_paths]
        else:
            candidates = []

        relative_paths = []
        for value in candidates:
            if isinstance(value, os.PathLike):
                value = os.fspath(value)
            if isinstance(value, str) and value.strip():
                relative_paths.append(value)
            else:
                repaired = True
        return relative_paths, repaired

    @staticmethod
    def _normalise_loaded_flags(raw_flags, relative_paths):
        """将历史状态文件迁移为与图片列表等长的 0--3 状态列表。"""
        repaired = not isinstance(raw_flags, list)
        if isinstance(raw_flags, dict):
            values = [raw_flags.get(path, 0) for path in relative_paths]
        elif isinstance(raw_flags, (list, tuple)):
            values = list(raw_flags)
        else:
            values = []

        if len(values) != len(relative_paths):
            repaired = True
        values = values[:len(relative_paths)]
        while len(values) < len(relative_paths):
            values.append(0)

        flags = []
        for value in values:
            try:
                flag = int(value)
            except (TypeError, ValueError):
                flag = 0
                repaired = True
            if flag not in [0, 1, 2, 3]:
                flag = 0
                repaired = True
            flags.append(flag)
        return flags, repaired

    def load_existing_project(self):
        missing_sample_tree_file = not os.path.exists(self.sample_tree_file)
        state_repaired = False
        if os.path.exists(self.lable_file):
            with open(self.lable_file, 'r+', encoding='utf-8') as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
                self.graphicsView.imageItem.initCatories(lines)
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    raw_paths = pickle.load(f)
            except Exception:
                raw_paths = []
                state_repaired = True
            self.relative_paths, paths_repaired = self._normalise_loaded_relative_paths(raw_paths)
            state_repaired = state_repaired or paths_repaired
        if os.path.exists(self.flag_file):
            try:
                with open(self.flag_file, 'rb') as f:
                    raw_flags = pickle.load(f)
            except Exception:
                raw_flags = []
                state_repaired = True
            self.flag, flags_repaired = self._normalise_loaded_flags(raw_flags, self.relative_paths)
            state_repaired = state_repaired or flags_repaired
        else:
            self.flag, flags_repaired = self._normalise_loaded_flags([], self.relative_paths)
            state_repaired = state_repaired or flags_repaired
        self.rebuild_path_index()
        self.total = len(self.relative_paths)
        self.image_paths = [os.path.join(self.project_dir, p) for p in self.relative_paths]
        categories_changed = self.sync_categories_from_data()
        self.load_sample_tree_state()
        self.ensure_sample_tree_integrity()
        if missing_sample_tree_file or categories_changed or state_repaired:
            self.persist_project_state()
        self.populate_listwidget(self.relative_paths)
        if self.listWidget.count() > 0: self.listWidget.setCurrentRow(0)

    def persist_project_state(self):
        try:
            with open(self.lable_file, 'w', encoding='utf-8') as f:
                for l in self.graphicsView.imageItem.existing_categories:
                    f.write(l + '\n')
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.relative_paths, f)
            with open(self.flag_file, 'wb') as f:
                pickle.dump(self.flag, f)
            with open(self.sample_tree_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'groups': self.sample_groups,
                    'assignments': self.sample_assignments
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"save project state failed: {e}")

    def populate_listwidget(self, paths):
        self.listWidget.clear()
        for p in paths:
            it = QListWidgetItem(os.path.basename(p));
            it.setToolTip(p)
            global_index = self.get_global_index_from_path(p)
            if 0 <= global_index < len(self.flag):
                if self.flag[global_index] == 1:
                    it.setIcon(QIcon(ICON_LABEL))
                elif self.flag[global_index] in [2, 3]:
                    it.setIcon(QIcon(ICON_GOOD))
            self.listWidget.addItem(it)

    def initUI(self):
        self.trainProgress.setValue(0);
        self.epochs = [];
        self.loss_values = [];
        self.map50_values = [];
        self.precision_values = [];
        self.recall_values = []
        self.showMaximized();
        self.tabWidget.setCurrentIndex(0);
        self.plot_widgets = []
        self.nameEdit.setText(self.project_dir);
        self.dataEdit.setText(self.data_yaml)
        self.epochEdit.setText("100");
        self.sizeEdit.setText("640");
        self.batchEdit.setText("32")
        titles = ["损失", "mAP50-95", "精确率", "召回率"];
        colors = ['r', 'g', 'b', 'm']
        for i, title in enumerate(titles):
            plot = pg.PlotWidget(title=title);
            plot.setBackground('w')
            curve = plot.plot(pen=colors[i]);
            self.plot_widgets.append({'widget': plot, 'curve': curve})
            self.gridLayout.addWidget(plot, i // 2, i % 2)
        self.set_fliter_Widget()

        # --- 添加这一行，调用动态加载模型的方法 ---
        self.load_dynamic_models()

    def load_dynamic_models(self):
        """动态读取 pt 文件夹下的所有模型，刷新到下拉框"""
        import sys
        self.comboBox_2.clear()  # 清空 UI 里写死的旧模型名字

        # 获取真实的根目录（兼容打包环境）
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        pt_dir = os.path.join(base_dir, 'pt')
        if os.path.exists(pt_dir):
            models = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
            if models:
                self.comboBox_2.addItems(models)
            else:
                self.comboBox_2.addItem("未找到pt模型")
        else:
            self.comboBox_2.addItem("未找到pt文件夹")

    # ================= YOLO/辅助功能 =================

    def openProject(self):
        from UI.Project import ProjectManager
        self.w = ProjectManager();
        self.w.show();
        self.close()

    def setYaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择YAML", "", "YAML (*.yaml)")
        if p: self.data_yaml = p; self.dataEdit.setText(p)

    def auto(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "Model (*.pt)")
        if p:
            self.model = YOLO(p)
            self.record_project_model(p)
            self.detection_thread = DetectionThread('folder', self.image_dir, self.model, self.class_filter, True,
                                                    self.json_dir)
            self.graphicsView.imageItem.initCatories(list(self.model.names.values()))
            self.detection_thread.end_signal.connect(
                lambda: self.listWidget.setCurrentRow(self.listWidget.currentRow() + 1))
            self.detection_thread.start()

    def show_ai_training_analysis(self):
        if self.train_thread is not None and self.train_thread.isRunning():
            QMessageBox.information(self, "训练进行中", "训练结束后再分析，避免读取尚未完整的结果。")
            return
        if self.ai_training_dialog is None:
            self.ai_training_dialog = AITrainingAnalysisDialog(self)
        self.ai_training_dialog.reopen(self.latest_training_run)

    def remember_training_result(self, folder):
        self.latest_training_run = os.path.abspath(folder)
        self.logText.append("训练结果目录：" + self.latest_training_run)

    def apply_training_analysis_config(self, config):
        self.data_yaml = config["data"]
        self.dataEdit.setText(config["data"])
        index = self.comboBox_2.findText(config["weight"])
        if index < 0:
            self.comboBox_2.addItem(config["weight"])
            index = self.comboBox_2.count() - 1
        self.comboBox_2.setCurrentIndex(index)
        self.comboBox.setCurrentText(config["task"])
        self.epochEdit.setText(str(config["epochs"]))
        self.sizeEdit.setText(str(config["imgsz"]))
        self.batchEdit.setText(str(config["batch"]))
        self.tabWidget.setCurrentWidget(self.trainPage)

    def train(self):
        if self.train_thread is not None and self.train_thread.isRunning():
            return
        self.logText.clear()
        self.trainProgress.setValue(0)

        try:
            data_yaml = os.path.abspath(self.dataEdit.text().strip() or self.data_yaml)
            model_name = self.comboBox_2.currentText().strip()
            epoch = int(self.epochEdit.text())
            img_sz = int(self.sizeEdit.text())
            batch = int(self.batchEdit.text())
            if min(epoch, img_sz, batch) <= 0:
                raise ValueError("训练轮次、输入尺寸和 Batch 必须是正整数")

            if not os.path.exists(data_yaml):
                raise FileNotFoundError(f"Data yaml not found: {data_yaml}")
            if not model_name or "链壘鍒?" in model_name:
                raise FileNotFoundError("No available model weight found in pt directory.")

            self.data_yaml = data_yaml
            self.logText.append(f"Data: {data_yaml}")
            self.logText.append(f"Weight: {model_name}")
            self.logText.append(f"Params: imgsz={img_sz}, batch={batch}, epoch={epoch}")

            self.train_thread = YOLOTrainThread(
                data=data_yaml,
                task=self.comboBox.currentText(),
                weight=model_name,
                epoch=epoch,
                img_sz=img_sz,
                batch=batch,
                name=self.project_dir
            )
            self.train_thread.epoch_progress.connect(self.update_progress)
            self.train_thread.train_finished.connect(self.on_train_finished)
            self.train_thread.result_directory.connect(self.remember_training_result)
            self.train_thread.record_warning.connect(self.logText.append)
            self.train_thread.finished.connect(lambda: self.train_btn.setEnabled(True))
            self.train_btn.setEnabled(False)
            self.epochs.clear()
            self.loss_values.clear()
            self.map50_values.clear()
            self.precision_values.clear()
            self.recall_values.clear()
            for plot in self.plot_widgets:
                plot['curve'].setData([], [])
            self.train_thread.start()
            self.logText.append("Training thread started.")
        except Exception as e:
            self.logText.append(f"Training start failed: {e}")
            QMessageBox.critical(self, "训练启动失败", str(e))

    def update_progress(self, epoch, metrics):
        self.trainProgress.setValue(int((epoch + 1) / self.train_thread.epoch * 100))
        self.logText.append(
            f"Epoch {epoch + 1}: loss={metrics['loss']:.4f}, "
            f"mAP50-95={metrics['map50-95']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
        )
        QMetaObject.invokeMethod(self, "_update_chart", Qt.QueuedConnection, Q_ARG(int, epoch),
                                 Q_ARG(float, metrics['loss']), Q_ARG(float, metrics['map50-95']),
                                 Q_ARG(float, metrics['precision']), Q_ARG(float, metrics['recall']))

    def on_train_finished(self, message):
        if message == "success":
            self.trainProgress.setValue(100)
            self.logText.append("Training finished successfully.")
            QMessageBox.information(self, "训练完成", "模型训练已完成。")
        else:
            self.logText.append(f"Training failed: {message}")
            QMessageBox.critical(self, "训练失败", message)

    def load_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "Model (*.pt)")
        if p:
            self.model = YOLO(p)
            self.info_browser.append(f"✅ 加载成功: {os.path.basename(p)}")
            self.record_project_model(p)

    def start_detection(self, t):
        if t == 'image':
            p = QFileDialog.getOpenFileName(self, "选图", "", "Img (*.jpg *.png)")[0]
        elif t == 'folder':
            p = QFileDialog.getExistingDirectory(self, "选文件夹")
        if p:
            self.detection_thread = DetectionThread(t, p, self.model, self.class_filter)
            self.detection_thread.result_signal.connect(self.update_result);
            self.detection_thread.start()

    def update_result(self, o, r, c):
        self.left_label.setPixmap(self.cv2_to_pixmap(o));
        self.right_label.setPixmap(self.cv2_to_pixmap(r))
        self.info_browser.setText("📊 统计:\n" + "\n".join([f"{k}: {v}" for k, v in c.items()]))

    def cv2_to_pixmap(self, f):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB);
        h, w, ch = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))

    def update_dektop(self, str):
        self.set_fliter_Widget()

    def assign_items_to_subcategory(self, items, tag, child):
        if not items:
            return
        invalid_items = []
        changed = False
        current_filter_child = None
        if self.current_tree_filter[0] == "child" and self.current_tree_filter[1] == tag:
            current_filter_child = self.current_tree_filter[2]
        should_advance_in_child_view = (
            len(items) == 1
            and current_filter_child is not None
            and current_filter_child != child
        )
        target_row = self.listWidget.row(items[0]) if should_advance_in_child_view else None
        refresh_current_child_view = current_filter_child is not None
        for item in items:
            rel_path = self.get_item_relative_path(item)
            if not rel_path:
                continue
            if tag not in self.get_image_labels(rel_path):
                invalid_items.append(item.text())
                continue
            assignment = self.sample_assignments.setdefault(rel_path, {})
            if assignment.get(tag) == child:
                continue
            assignment[tag] = child
            changed = True
        if invalid_items:
            QMessageBox.information(
                self,
                "提示",
                f"有 {len(invalid_items)} 张图片不包含缺陷【{tag}】，已跳过归类。"
            )
        if not changed:
            return
        self.persist_project_state()
        if refresh_current_child_view:
            self.refresh_image_list(
                preferred_row=target_row,
                preserve_display=should_advance_in_child_view
            )

    def get_common_labels_for_items(self, items):
        common_labels = None
        for item in items:
            rel_path = self.get_item_relative_path(item)
            labels = self.get_image_labels(rel_path)
            if common_labels is None:
                common_labels = set(labels)
            else:
                common_labels &= labels
        return common_labels or set()

    def clear_items_subcategory(self, items):
        if not items:
            return
        changed_tags = set()
        for item in items:
            rel_path = self.get_item_relative_path(item)
            if rel_path in self.sample_assignments:
                changed_tags.update(self.sample_assignments.get(rel_path, {}).keys())
                self.sample_assignments.pop(rel_path, None)
        if not changed_tags:
            return
        self.persist_project_state()
        if self.current_tree_filter[0] == "child" and self.current_tree_filter[1] in changed_tags:
            self.refresh_image_list()

    def show_context_menu(self, pos):
        # 1. 获取所有当前选中的 item 列表
        selected_items = self.listWidget.selectedItems()
        if not selected_items:
            return

        m = QMenu()
        # 动态修改菜单文字，提示是单选还是多选
        complete_good_text = "设为完全良品 (批量)" if len(selected_items) > 1 else "设为完全良品"
        overkill_good_text = "设为过杀品 (批量)" if len(selected_items) > 1 else "设为过杀品"
        d_text = "删除图片 (批量)" if len(selected_items) > 1 else "删除图片"

        complete_good_action = QAction(complete_good_text, self)
        overkill_good_action = QAction(overkill_good_text, self)
        d = QAction(d_text, self)
        m.addAction(complete_good_action)
        m.addAction(overkill_good_action)
        m.addAction(d)

        classify_menu = m.addMenu("归类到子样本分类")
        has_subcategory = False
        allowed_tags = self.get_common_labels_for_items(selected_items)
        for tag in self.graphicsView.imageItem.existing_categories:
            if tag not in allowed_tags:
                continue
            children = self.sample_groups.get(tag, [])
            if not children:
                continue
            has_subcategory = True
            tag_menu = classify_menu.addMenu(tag)
            for child in children:
                action = QAction(child, self)
                action.triggered.connect(
                    lambda _, current_tag=tag, current_child=child: self.assign_items_to_subcategory(
                        selected_items, current_tag, current_child
                    )
                )
                tag_menu.addAction(action)
        if not has_subcategory:
            classify_menu.setEnabled(False)
            if not allowed_tags:
                classify_menu.setTitle("归类到子样本分类（当前图片无可归类缺陷）")
            else:
                classify_menu.setTitle("归类到子样本分类（无可用子分类）")

        clear_classify = QAction("清除子样本归类", self)
        m.addAction(clear_classify)
        m.addSeparator()
        feedback_action = m.addAction("记录图片问题…")

        action = m.exec_(self.listWidget.mapToGlobal(pos))

        if action == complete_good_action:
            # 批量设为完全良品
            for item in selected_items:
                self.set_good(item, 2)

        elif action == overkill_good_action:
            # 批量设为过杀品
            for item in selected_items:
                self.set_good(item, 3)

        elif action == d:
            # 批量删除：为防止索引错乱，必须倒序删除！
            # 先把选中的 item 按行号从大到小排序
            items_to_delete = sorted(selected_items, key=lambda x: self.listWidget.row(x), reverse=True)
            for item in items_to_delete:
                self.delete_item(item)
        elif action == clear_classify:
            self.clear_items_subcategory(selected_items)
        elif action == feedback_action:
            from Utils.ProjectContextDialog import record_feedback
            record_feedback(self, [self.get_item_relative_path(item) for item in selected_items])

    def generate_data_dir(self):
        cats = self.graphicsView.imageItem.existing_categories
        for c in cats:
            tags = self.fliter_index([c])
            w = Task.Data_Worker(self.project_dir, self.relative_paths, tags, c, cats)
            self.threadPool.start(w)

    @pyqtSlot(int, float, float, float, float)
    def _update_chart(self, epoch, loss, map50, precision, recall):
        self.epochs.append(epoch);
        self.loss_values.append(loss);
        self.map50_values.append(map50)
        self.precision_values.append(precision);
        self.recall_values.append(recall)
        self._update_single_plot(0, self.loss_values);
        self._update_single_plot(1, self.map50_values)
        self._update_single_plot(2, self.precision_values);
        self._update_single_plot(3, self.recall_values)

    def _update_single_plot(self, index, y_data):
        self.plot_widgets[index]['curve'].setData(self.epochs, y_data)
        plot = self.plot_widgets[index]['widget']
        if len(self.epochs) > 1:
            plot.setXRange(0, self.epochs[-1] + 10);
            plot.setYRange(min(y_data) * 0.9, max(y_data) * 1.1)

    def select_all(self):
        for item in self.checkboxes.values():
            item.setCheckState(Qt.Checked)
        self.update_result_label()

    def deselect_all(self):
        for item in self.checkboxes.values():
            item.setCheckState(Qt.Unchecked)
        self.update_result_label()

    def update_result_label(self):
        sel = self.get_selected()
        t = "已选择: " + (", ".join(sel[:3]) + (f" 等{len(sel)}项" if len(sel) > 3 else "") if sel else "无")
        self.result_label.setText(t)

    def get_selected(self):
        return [t for t, item in self.checkboxes.items() if item.checkState() == Qt.Checked]

    def setPage(self, offset):
        idx = self.listWidget.currentRow() + offset
        if 0 <= idx < self.listWidget.count(): self.listWidget.setCurrentRow(idx)

    def closeEvent(self, event):
        """关闭保存逻辑"""
        chat = getattr(self, "ai_chat_dock", None)
        if chat is not None and chat.running():
            chat.stop()
            self.statusBar().showMessage("正在停止 AI 助手，请等待当前请求结束后再关闭工程。", 15000)
            event.ignore()
            return
        analysis = getattr(self, "ai_training_dialog", None)
        if analysis is not None and analysis.running():
            analysis.stop()
            self.statusBar().showMessage("正在结束训练结果分析，请等待当前请求返回或超时后再关闭工程。", 15000)
            event.ignore()
            return
        training = getattr(self, "train_thread", None)
        if training is not None and training.isRunning():
            self.statusBar().showMessage("训练正在运行，请训练结束后再关闭工程。", 15000)
            event.ignore()
            return
        if chat is not None:
            chat.save_timer.stop()
            if not chat.save_conversation():
                self.statusBar().showMessage("对话尚未保存，请检查项目目录后再关闭。", 15000)
                event.ignore()
                return
            chat.hide()
        try:
            self.persist_project_state()
        except:
            pass
