import sys
from PyQt5.QtWidgets import (QMainWindow, QSplitter, QListWidget, QPushButton,
                             QVBoxLayout, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
import json
import shutil
import pickle
from datetime import datetime
from Utils.test import TestWindow
from PyQt5.QtCore import QSettings
from Utils.GlobalVar import sqlite_db



sys.path.append('pt')


#工程创建和信息填写

class ProjectManager(QWidget):
    REQUIRED_PROJECT_DIRS = ("images", "jsons")
    REQUIRED_PROJECT_FILES = ("datafile.dat", "flagfile.dat", "label.txt")

    def __init__(self):
        super().__init__()
        self.init_ui()
        # self.init_data()


    def init_ui(self):
        self.setFixedSize(600, 480)
        self.gridLayout = QGridLayout()
        self.btn_create = QPushButton("新建工程")
        self.btn_import = QPushButton("选择工程文件夹")
        self.btn_open = QPushButton("打开选中工程")
        self.btn_delete = QPushButton("删除工程")
        self.btn_exit = QPushButton("退出程序")
        self.label = QLabel("最近打开工程：")
        self.project_list = QListWidget()
        self.gridLayout.addWidget(self.label, 0, 0, 1, 5)
        self.gridLayout.addWidget(self.project_list, 1, 0, 9, 5)

        self.gridLayout.addWidget(self.btn_create, 1, 5, 1, 1)
        self.gridLayout.addWidget(self.btn_import, 3, 5, 1, 1)
        self.gridLayout.addWidget(self.btn_open, 4, 5, 1, 1)
        self.gridLayout.addWidget(self.btn_delete, 6, 5, 1, 1)
        self.gridLayout.addWidget(self.btn_exit, 8, 5, 1, 1)
        self.setLayout(self.gridLayout)
        self.update_project_list()

        # 连接信号与槽
        self.btn_create.clicked.connect(self.create_project)
        self.btn_import.clicked.connect(self.choose_and_open_project)
        self.btn_open.clicked.connect(lambda : self.open_selected_project(self.project_list.currentItem()))
        self.btn_delete.clicked.connect(self.delete_project)
        self.btn_exit.clicked.connect(self.close_application)
        self.project_list.itemDoubleClicked.connect(self.open_selected_project)

    def create_project(self):

        # 打开配置窗口
        self.config_window = ProjectInfoDlg()
        self.config_window.sendUpdate.connect(self.update_project_list)
        self.config_window.sendClose.connect(self.close)
        self.config_window.show()

    def update_project_list(self):
        """更新左侧工程列表"""
        self.setting = QSettings("recnet.ini", QSettings.IniFormat)
        self.recent_projects = self.normalize_recent_projects(self.setting.value("recent/path", []))
        self.project_list.clear()
        if self.recent_projects:
            for path in self.recent_projects:
                self.project_list.addItem(path)

    def normalize_recent_projects(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple)):
            items = []
            for item in value:
                if isinstance(item, str) and "," in item:
                    items.extend(item.split(","))
                else:
                    items.append(item)
        else:
            items = [value]

        projects = []
        seen = set()
        for item in items:
            path = str(item).strip()
            if not path or path == "@Invalid()":
                continue
            norm = os.path.abspath(os.path.normpath(path))
            key = os.path.normcase(norm)
            if key in seen:
                continue
            projects.append(norm)
            seen.add(key)
        return projects

    def remember_project(self, project_path):
        project_path = os.path.abspath(os.path.normpath(project_path))
        projects = self.normalize_recent_projects(self.setting.value("recent/path", []))
        project_key = os.path.normcase(project_path)
        projects = [p for p in projects if os.path.normcase(p) != project_key]
        projects.insert(0, project_path)
        self.setting.setValue("recent/path", projects)
        self.setting.sync()
        self.recent_projects = projects
        self.update_project_list()

    def forget_project(self, project_path):
        project_path = os.path.abspath(os.path.normpath(str(project_path).strip()))
        projects = self.normalize_recent_projects(self.setting.value("recent/path", []))
        project_key = os.path.normcase(project_path)
        next_projects = [p for p in projects if os.path.normcase(p) != project_key]
        if len(next_projects) == len(projects):
            return False
        self.setting.setValue("recent/path", next_projects)
        self.setting.sync()
        self.recent_projects = next_projects
        self.update_project_list()
        return True

    def get_project_missing_entries(self, project_path):
        missing = []
        for name in self.REQUIRED_PROJECT_DIRS:
            if not os.path.isdir(os.path.join(project_path, name)):
                missing.append(name + "/")
        for name in self.REQUIRED_PROJECT_FILES:
            if not os.path.isfile(os.path.join(project_path, name)):
                missing.append(name)
        return missing

    def choose_and_open_project(self):
        start_dir = os.getcwd()
        for path in getattr(self, "recent_projects", []):
            if os.path.isdir(path):
                start_dir = path
                break
        project_path = QFileDialog.getExistingDirectory(self, "选择已有工程文件夹", start_dir)
        if project_path:
            self.open_project_path(project_path)

    def open_selected_project(self, item):
        """双击打开选中的工程"""
        if item is None:
            QMessageBox.information(self, "提示", "请先选择一个工程，或点击“选择工程文件夹”。")
            return
        self.open_project_path(item.text())

    def open_project_path(self, selected_path):
        project_path = os.path.abspath(os.path.normpath(str(selected_path).strip()))
        if not os.path.isdir(project_path):
            message = f"工程文件夹不存在：\n{project_path}"
            if self.forget_project(project_path):
                message += "\n\n已自动从最近打开工程列表移除。"
            QMessageBox.critical(self, "打开失败", message)
            return
        missing = self.get_project_missing_entries(project_path)
        if missing:
            QMessageBox.critical(
                self,
                "打开失败",
                "这不是完整的标注工程文件夹，缺少：\n"
                + "\n".join(missing)
                + "\n\n请复制完整工程文件夹，不要只复制 images/jsons。"
            )
            return
        self.remember_project(project_path)
        self.workspace = TestWindow(project_path)
        self.workspace.show()
        self.close()

    def confirm_delete_project(self, project_path):
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("确认删除项目？")
        dialog.setText(
            "将删除此路径下的整个项目：\n\n"
            f"{project_path}\n\n"
            "该操作会删除项目目录内的所有文件，且不可恢复。"
        )
        cancel_button = dialog.addButton("取消", QMessageBox.RejectRole)
        delete_button = dialog.addButton("确认删除", QMessageBox.DestructiveRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec_()
        return dialog.clickedButton() == delete_button


    def delete_project(self):
        if self.project_list.count():
            current_item = self.project_list.currentItem()
            if current_item is None:
                QMessageBox.information(self, "提示", "请先选择一个要删除的工程。")
                return

            project_path = current_item.text()
            if project_path:
                if not os.path.isdir(project_path):
                    self.forget_project(project_path)
                    QMessageBox.information(
                        self,
                        "已移除",
                        f"工程文件夹不存在，已从最近打开工程列表移除：\n{project_path}"
                    )
                    return
                if self.confirm_delete_project(project_path):
                    try:
                        delete_sql = "DELETE FROM projects_table WHERE name = :name"
                        params = {'name': project_path}
                        affected_rows = sqlite_db.execute_command(delete_sql, params)
                        delete_sql = f"drop table {project_path}"
                        affected_rows = sqlite_db.execute_command(delete_sql)


                        shutil.rmtree(project_path)  # 递归删除文件夹
                        self.forget_project(project_path)
                        QMessageBox.information(self, "删除完成", f"已删除工程：\n{project_path}")
                    except Exception as e:
                        QMessageBox.critical(self, "删除失败", str(e))
        else:
            QMessageBox.information(self, "提示", "当前没有可删除的工程。")

    def close_application(self):
        """关闭程序功能"""
        reply = QMessageBox.question(self, '退出确认', "确定要退出程序吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()


class ProjectInfoDlg(QDialog):

    sendUpdate = pyqtSignal()
    sendClose = pyqtSignal()

    def __init__(self):
        super(ProjectInfoDlg, self).__init__()
        self.initUI()


    def initUI(self):
        self.setFixedSize(600, 480)
        self.gridLayout = QGridLayout()
        self.labels = [
            QLabel('工程名称：'),
            QLabel('工程描述：'),
            QLabel('任务类型：'),
            QLabel('标注类型：')
        ]
        self.lineEdits = [
            QLineEdit(),
            QLineEdit(),
        ]
        self.comboxs = [
            QComboBox(),
            QComboBox()
        ]
        self.taskDir = {
            '图像分类': '文本标注',
            '图像检测': '矩形标注',
            '图像分割': '多边形标注',
            '异常检测': '文本标注'
        }
        self.comboxs[0].addItems(self.taskDir.keys())
        self.comboxs[1].addItems(self.taskDir.values())
        self.comboxs[0].setCurrentIndex(1)
        self.comboxs[1].setCurrentIndex(1)
        self.btns = [
            QPushButton("确定"),
            QPushButton("取消")
        ]
        for i, button in enumerate(self.labels):
            self.gridLayout.addWidget(button, 1 + 2*i, 0, 1, 1)
        for i, lineEdit in enumerate(self.lineEdits):
            self.gridLayout.addWidget(lineEdit, 1 + 2*i, 1, 1, 4)
            for i, combox in enumerate(self.comboxs):
                self.gridLayout.addWidget(combox, 5 + 2 * i, 1, 1, 4)
        for i, btn in enumerate(self.btns):
            self.gridLayout.addWidget(btn, 8, 0 + 3*i, 1, 2)
        self.setLayout(self.gridLayout)

        self.btns[1].clicked.connect(self.closeWindow)
        self.btns[0].clicked.connect(self.checkInfo)
        self.name = ""
        self.desc = ""


    def closeWindow(self):
        self.close()

    def crateWorkWindow(self, project_dir):
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)
            # 生成配置文件
            config = {
                "name": os.path.basename(project_dir),
                "created": datetime.now().isoformat(),
                "datafile": "datafile.dat",
                "flagfile": "flagfile.dat"
            }
            with open(os.path.join(project_dir, "config.json"), 'w') as f:
                json.dump(config, f)
            # 创建数据文件并初始化
            datafile_path = os.path.join(project_dir, config["datafile"])
            flagpath = os.path.join(project_dir, config["flagfile"])
            try:
                # TestWindow 使用列表保存图片路径和状态；新工程也必须从空列表开始。
                # 旧版这里写入 {}，首次“导入数据集”时会因 dict 没有 append() 而失败。
                initial_paths = []
                initial_flags = []
                with open(datafile_path, 'wb') as f:
                    pickle.dump(initial_paths, f)  # 使用pickle序列化二进制数据
                with open(flagpath, 'wb') as f:
                    pickle.dump(initial_flags, f)  # 使用pickle序列化二进制数据
            except Exception as e:
                QMessageBox.critical(self, "错误", f"数据文件创建失败: {str(e)}")
                shutil.rmtree(project_dir)  # 回滚：删除已创建的工程目录
                return
            self.workspace = TestWindow(project_dir)

            self.workspace.show()
            self.sendUpdate.emit()
            self.sendClose.emit()
            self.close()

        else:
            QMessageBox.information(self, "提示", "工程已存在！")


    def checkInfo(self):
        if self.name == "":
            self.name = self.lineEdits[0].text()
            self.desc = self.lineEdits[1].text()
        task = self.comboxs[0].currentText()
        label = self.comboxs[1].currentText()
        if self.name == "" or self.desc == "":
            QMessageBox.information(self, '提示', '工程信息不完整！')
            return

        insert_sql = "INSERT INTO projects_table (name, created_at) VALUES (:name, :created_at)"
        params = {
            'name': self.name,
            'created_at': datetime.now().isoformat()
        }
        user_id = sqlite_db.execute_command(insert_sql, params, return_id=True)

        create_table_sql = f"""
               CREATE TABLE IF NOT EXISTS {self.name} (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   image_path TEXT NOT NULL UNIQUE,      -- 图像路径
                   label_path TEXT NOT NULL UNIQUE,      -- 标注路径
                   label_type INTEGER DEFAULT 0    -- 标注类型（INT）
               )
               """
        sqlite_db.execute_command(create_table_sql)


        self.setting = QSettings("recnet.ini", QSettings.IniFormat)
        self.recent_works = self.setting.value('recent/path')
        if self.recent_works:
            self.recent_works.insert(0, self.name)
        else:
            self.recent_works = [self.name]
        self.setting.setValue('recent/path', self.recent_works)
        self.setting.sync()
        self.crateWorkWindow(self.name)

    def setSysArg(self, name, desc):
        self.name = name
        self.desc = desc


