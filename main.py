import multiprocessing
import os
import sys

import torch  # noqa: F401  # Keep torch imported early for packaged runtime stability.
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from UI.Project import ProjectManager
import PyQt5.QtSvg  # noqa: F401  # Ensure QtSvg is collected when packaging.


def get_resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def load_app_stylesheet(app):
    qss_path = get_resource_path(os.path.join("res", "app.qss"))
    if not os.path.exists(qss_path):
        return
    with open(qss_path, "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())


if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    load_app_stylesheet(app)

    window = ProjectManager()
    window.show()
    sys.exit(app.exec_())
