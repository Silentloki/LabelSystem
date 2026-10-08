"""Capture the inputs and final validation of a real Ultralytics training run."""
import math
import time
from pathlib import Path

from Utils.AITrainingAnalysis import RECORD, EVAL_KEYS, digest, file_hash, now, register_run, write_json


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class TrainingRunRecorder:
    def __init__(self, project, ui_config):
        self.project = str(Path(project).resolve())
        self.ui_config = ui_config
        self.folder = None
        self.file_states = {}
        self.record = {}

    def _snapshot(self, loader):
        dataset = loader.dataset
        images = list(dataset.im_files)
        labels = list(dataset.label_files)
        if not images or len(images) != len(labels):
            raise ValueError("无法获取实际训练/验证文件列表")
        pairs = []
        for image, label in zip(images, labels):
            row = []
            for filename in (image, label):
                path = Path(filename)
                if path.is_file():
                    before = path.stat()
                    value = file_hash(path)
                    after = path.stat()
                    state = (after.st_size, after.st_mtime_ns)
                    if (before.st_size, before.st_mtime_ns) != state:
                        raise ValueError("记录数据指纹时文件发生变化")
                    self.file_states[str(path)] = state
                else:
                    value = "missing"
                    self.file_states[str(path)] = None
                row.append(value)
            pairs.append(row)
        # Include the actual parsed labels, so stale label caches cannot silently compare as equal.
        loaded_labels = []
        for label in getattr(dataset, "labels", []):
            loaded_labels.append(plain({key: label.get(key) for key in
                                        ("cls", "bboxes", "segments", "normalized", "bbox_format")}))
        return digest({"files": sorted(pairs), "loaded_labels": loaded_labels}), len(images)

    def start(self, trainer):
        self.folder = Path(trainer.save_dir).resolve()
        start = time.monotonic()
        self.record = {"version": 1, "project": self.project, "status": "running", "started_at": now(),
                       "ui_config": self.ui_config, "names": plain(trainer.data.get("names", {})), "warnings": []}
        try:
            import ultralytics
            self.record["ultralytics_version"] = ultralytics.__version__
            for key, loader in (("train", trainer.train_loader), ("validation", trainer.test_loader)):
                fingerprint, count = self._snapshot(loader)
                self.record[key + "_fingerprint"] = fingerprint
                self.record[key + "_images"] = count
        except Exception as exc:
            self.record.pop("validation_fingerprint", None)
            self.record["warnings"].append("数据指纹不完整，不能确认实验可比性：" + str(exc)[:200])
        self.record["fingerprint_seconds"] = round(time.monotonic() - start, 2)
        self.save()
        register_run(self.project, self.folder)

    def save(self):
        if self.folder is not None:
            write_json(self.folder / RECORD, self.record)

    def finish(self, trainer):
        if self.folder is None:
            self.start(trainer)
            # A late snapshot cannot establish what was used throughout training.
            self.record.pop("validation_fingerprint", None)
            self.record["warnings"].append("训练前未成功记录输入，完成后不补认验证集版本。")
        self.record.update(status="completed", finished_at=now())
        for filename, old in self.file_states.items():
            path = Path(filename)
            current = (path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else None
            if current != old:
                self.record.pop("validation_fingerprint", None)
                self.record["warnings"].append("训练期间数据文件有变化，不能确认实验可比性。")
                break
        validator = trainer.validator
        self.record["evaluation_config"] = {key: plain(getattr(validator.args, key, None)) for key in EVAL_KEYS}
        self.record["evaluation_config"]["ultralytics_version"] = self.record.get("ultralytics_version")
        self.record["final_metrics"] = plain(trainer.metrics)
        try:
            self.record["per_class"] = plain(validator.metrics.summary())
        except Exception:
            self.record["warnings"].append("当前训练库未提供可导出的按类验证指标。")
        confusion = getattr(validator, "confusion_matrix", None)
        if confusion is not None and hasattr(confusion, "matrix") and getattr(validator.args, "plots", False):
            names = self.record["names"]
            ordered = [names[k] for k in sorted(names, key=lambda k: int(k))] if isinstance(names, dict) else list(names)
            matrix = plain(confusion.matrix)
            if len(matrix) == len(ordered) + 1:
                self.record["confusion_counts"] = {"names": ordered + ["background"],
                    "matrix": [[int(v) for v in row] for row in matrix],
                    "row_axis": "predicted", "column_axis": "true"}
        csv_path = self.folder / "results.csv"
        if csv_path.is_file():
            self.record["results_sha256"] = file_hash(csv_path)
        self.save()

    def fail(self, message):
        if self.folder is not None and self.record.get("status") != "completed":
            self.record.update(status="failed", finished_at=now())
            self.record["warnings"].append("训练未正常完成：" + str(message)[:300])
            self.save()
