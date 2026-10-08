import json
import os
import sys

import cv2
import numpy as np
import torch.cuda
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from Utils.TrainingRunRecord import TrainingRunRecorder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class YOLOTrainThread(QThread):
    epoch_progress = pyqtSignal(int, dict)
    train_finished = pyqtSignal(str)
    result_directory = pyqtSignal(str)
    record_warning = pyqtSignal(str)

    def __init__(
        self,
        cfg="yolov8n.yaml",
        data=None,
        task="目标检测",
        weight="yolov8n.pt",
        img_sz=640,
        batch=32,
        epoch=100,
        name="",
    ):
        super().__init__()
        self.data = data
        self.cfg = "yolov8n-seg.yaml" if task == "图像分割" else cfg
        default_weight = "yolov8n-seg.pt" if task == "图像分割" and weight == "yolov8n.pt" else weight
        self.weight = self._resolve_weight_path(default_weight)
        self.img_sz = img_sz
        self.name = name
        self.batch = batch
        self.epoch = epoch
        self._last_emitted_epoch = -1
        self.recorder = TrainingRunRecorder(name or os.getcwd(), {
            "data": os.path.abspath(data), "weight": os.path.abspath(self.weight), "task": task,
            "epochs": epoch, "imgsz": img_sz, "batch": batch,
        })

        self.model = YOLO(self.cfg).load(self.weight)
        self.model.add_callback("on_fit_epoch_end", self._emit_metrics)
        self.model.add_callback("on_pretrain_routine_end", self._record_start)
        self.model.add_callback("on_train_end", self._record_end)

    def _record_start(self, trainer):
        try:
            self.recorder.start(trainer)
        except Exception as exc:
            self.record_warning.emit("训练记录保存不完整：" + str(exc))

    def _record_end(self, trainer):
        try:
            self.recorder.finish(trainer)
        except Exception as exc:
            self.record_warning.emit("训练结果元数据保存失败：" + str(exc))
        self.result_directory.emit(str(trainer.save_dir))

    def _resolve_weight_path(self, weight):
        if not weight:
            raise FileNotFoundError("No model weight selected.")

        if os.path.isabs(weight) and os.path.exists(weight):
            return weight

        candidates = []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.join(project_root, "pt", weight))
        candidates.append(os.path.join(os.getcwd(), "pt", weight))
        candidates.append(os.path.join(os.getcwd(), weight))

        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            candidates.append(os.path.join(exe_dir, "pt", weight))
            candidates.append(os.path.join(exe_dir, weight))
            if hasattr(sys, "_MEIPASS"):
                candidates.append(os.path.join(sys._MEIPASS, "pt", weight))
                candidates.append(os.path.join(sys._MEIPASS, weight))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Model weight not found: {weight}")

    def _safe_float(self, value):
        if value is None:
            return 0.0
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=float).reshape(-1)
            return float(arr.mean()) if arr.size else 0.0
        if isinstance(value, np.ndarray):
            arr = value.astype(float).reshape(-1)
            return float(arr.mean()) if arr.size else 0.0
        return float(value)

    def run(self):
        null_stream = None
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            if sys.stdout is None or sys.stderr is None:
                null_stream = open(os.devnull, "w", encoding="utf-8")
                if sys.stdout is None:
                    sys.stdout = null_stream
                if sys.stderr is None:
                    sys.stderr = null_stream

            self.model.train(
                data=self.data,
                epochs=self.epoch,
                imgsz=self.img_sz,
                batch=self.batch,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                amp=True,
                project="runs",
                name=self.name,
                workers=0,
                single_cls=False,
                verbose=False,
            )
            try:
                self.model.export(format="onnx", opset=14)
            except Exception as exc:
                self.record_warning.emit("训练已完成，但 ONNX 导出失败：" + str(exc))
            self.train_finished.emit("success")
        except Exception as e:
            try:
                self.recorder.fail(str(e))
            except Exception:
                pass
            self.train_finished.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if null_stream is not None:
                null_stream.close()

    def _emit_metrics(self, trainer):
        if trainer.epoch == self._last_emitted_epoch:
            return
        self._last_emitted_epoch = trainer.epoch
        metric_source = getattr(trainer, "metrics", {}) or {}
        map_key = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in metric_source else "metrics/mAP50-95(M)"
        precision_key = "metrics/precision(B)" if "metrics/precision(B)" in metric_source else "metrics/precision(M)"
        recall_key = "metrics/recall(B)" if "metrics/recall(B)" in metric_source else "metrics/recall(M)"
        metrics = {
            "loss": self._safe_float(getattr(trainer, "tloss", None)),
            "map50-95": self._safe_float(metric_source.get(map_key, 0.0)),
            "precision": self._safe_float(metric_source.get(precision_key, 0.0)),
            "recall": self._safe_float(metric_source.get(recall_key, 0.0)),
        }
        self.epoch_progress.emit(trainer.epoch, metrics)


class DetectionThread(QThread):
    result_signal = pyqtSignal(np.ndarray, np.ndarray, dict)
    end_signal = pyqtSignal()

    def __init__(self, source_type, source_path, model, class_filter, flag=False, save_dir=None):
        super().__init__()
        self.flag = flag
        self.source_type = source_type
        self.source_path = source_path
        self.model = model
        self.class_filter = class_filter
        self._running = True
        self.save_dir = save_dir

    def stop(self):
        self._running = False

    def run(self):
        if self.source_type == "image":
            self._process_image()
        elif self.source_type == "folder":
            self._process_folder()
        elif self.source_type == "video":
            self._process_video()

    def _process_image(self):
        frame = cv2.imread(self.source_path)
        self._detect_and_emit(frame)

    def _process_folder(self):
        for filename in os.listdir(self.source_path):
            if not self._running:
                break
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                if self.flag:
                    self.autoFilename = filename
                path = os.path.join(self.source_path, filename)
                frame = cv2.imread(path)
                self._detect_and_emit(frame)

    def _process_video(self):
        cap = cv2.VideoCapture(self.source_path)
        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self._detect_and_emit(frame)
                QThread.msleep(30)
            else:
                break
        cap.release()

    def _detect_and_emit(self, frame):
        results = self.model.predict(frame, verbose=False, conf=0.5)
        if self.flag is False:
            boxes = self._filter_boxes(results[0].boxes)
            result_img = results[0].plot()
            counts = self._count_objects(boxes)
            self.result_signal.emit(frame, result_img, counts)
        else:
            data = {
                "image_width": frame.shape[1],
                "image_height": frame.shape[0],
            }
            annotations = []
            for result in json.loads(results[0].to_json(normalize=True)):
                annotation = {
                    "color": "#ff0000",
                    "selected": False,
                    "type": "rect",
                    "points": [
                        {"x": result["box"]["x1"], "y": result["box"]["y1"]},
                        {"x": result["box"]["x2"], "y": result["box"]["y2"]},
                    ],
                    "lable": result["name"],
                }
                annotations.append(annotation)
            data["annotations"] = annotations
            file_name = os.path.join(self.save_dir, self.autoFilename)
            file_name = os.path.splitext(file_name)[0] + ".json"
            with open(file_name, "w") as f:
                json.dump(data, f, indent=2)
            self.end_signal.emit()

    def _filter_boxes(self, boxes):
        if self.class_filter:
            return [box for box in boxes if self.model.names[int(box.cls)] in self.class_filter]
        return boxes

    def _count_objects(self, boxes):
        counts = {}
        for box in boxes:
            cls_name = self.model.names[int(box.cls)]
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts
