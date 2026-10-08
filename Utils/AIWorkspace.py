"""Durable work artifacts and compare-before-write annotation transactions.

No model-generated paths are executable. Sources are explicit project paths or
UI-registered resources; writes are new artifacts or reviewed transactions.
"""
import base64
import hashlib
import json
import os
import io
import pickle
import re
import time
import uuid
from datetime import datetime
from pathlib import Path


def stamp():
    return datetime.now().isoformat(sep=" ", timespec="milliseconds")


def identifier():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f_") + uuid.uuid4().hex[:8]


def artifact_name(kind, title, metadata):
    """Readable Windows-safe result directory name; timestamps are still unique."""
    if kind == 'dataset':
        label = '图片与标注' if metadata.get('format') == 'native' else 'YOLO数据集'
    elif kind == 'crops':
        if metadata.get('mode') == 'tiles' and metadata.get('tile_size'):
            size = metadata['tile_size']
            label = f'切图_{size}×{size}'
        else:
            label = '缺陷裁剪' + ('_' + str(metadata['label']) if metadata.get('label') else '')
    elif kind == 'selection':
        label = '筛选_' + title
    else:
        label = title or kind
    label = re.sub(r'[^A-Za-z0-9_\-\u4e00-\u9fff×]+', '_', label).strip('_-')[:32] or '处理结果'
    return label + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_project_list(path, data=None):
    class DataOnlyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            raise ValueError("工程列表只允许普通数据。")
    value = DataOnlyUnpickler(io.BytesIO(Path(path).read_bytes() if data is None else data)).load()
    if not isinstance(value, list) or any(type(v) not in (str, int) for v in value):
        raise ValueError("工程图片/状态列表格式无效。")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def write_json(path, value):
    atomic_bytes(path, json_bytes(value))


def document_token(path, data):
    """UI autosave may reformat JSON; compare meaning as well as exact backups."""
    if data is None:
        return None
    try:
        if Path(path).suffix.lower() == ".json":
            def canonical(value):
                if isinstance(value, dict):
                    return {k: canonical(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [canonical(v) for v in value]
                if type(value) is float:
                    return int(value) if value.is_integer() else round(value, 12)
                return value
            return sha(json.dumps(canonical(json.loads(data.decode("utf-8-sig"))), sort_keys=True,
                                  ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if Path(path).name == "label.txt":
            return sha(json.dumps(data.decode("utf-8-sig").splitlines(), ensure_ascii=False).encode("utf-8"))
    except (ValueError, UnicodeError):
        pass
    return sha(data)


class Workspace:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.root = (self.project / "ai_workbench").resolve()
        self.root.relative_to(self.project)
        self.open_artifacts = {}

    def source(self, relative):
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("只接受当前工程的相对路径。")
        path = (self.project / relative).resolve()
        path.relative_to(self.project)
        return path

    def source_key(self, relative):
        """Compare project-relative references using this host's path rules.

        UI references use forward slashes; historical Windows records may use
        backslashes or different case. Keep the original record unchanged and
        never fall back to a basename or a path outside this project.
        """
        return os.path.normcase(str(self.source(relative)))

    def location(self, kind, entry_id):
        if kind not in {"runs", "artifacts", "transactions", "workflows"}:
            raise ValueError("未知记录类型。")
        pattern = r"[A-Za-z0-9_\-\u4e00-\u9fff×]{1,100}" if kind == 'artifacts' else r"[A-Za-z0-9_-]{1,100}"
        if not isinstance(entry_id, str) or not re.fullmatch(pattern, entry_id):
            raise ValueError("无效记录 ID。")
        path = (self.root / kind / entry_id).resolve()
        path.relative_to(self.root)
        return path

    def create(self, kind, title, metadata=None):
        base = artifact_name(kind, title, metadata or {})
        suffix = 1
        while True:
            entry_id = base if suffix == 1 else f'{base}_{suffix}'
            folder = self.location("artifacts", entry_id)
            try:
                folder.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
        manifest = {"id": entry_id, "kind": kind, "title": title, "created_at": stamp(),
                    "status": "running", "metadata": metadata or {}}
        write_json(folder / "manifest.json", manifest)
        self.open_artifacts[entry_id] = (folder, manifest)
        return entry_id, folder, manifest

    def finish(self, folder, manifest, **result):
        manifest.update(result, status="completed", finished_at=stamp())
        write_json(folder / "manifest.json", manifest)
        self.open_artifacts.pop(manifest["id"], None)
        return {"artifact_id": manifest["id"], "kind": manifest["kind"], "title": manifest["title"],
                "output": str(folder), **result}

    def finish_export(self, folder, manifest, **result):
        from Utils.AIExportStorage import publish_dataset
        result = publish_dataset(self, folder, manifest, result)
        return self.finish(folder, manifest, **result)

    def artifact(self, entry_id, kind=None):
        folder = self.location("artifacts", entry_id)
        value = read_json(folder / "manifest.json")
        if value.get("status") in {"deleted", "deleting", "delete_failed"}:
            raise ValueError("处理结果已删除或清理未完成，请在工作记录中查看状态；不能重新打开或采用。")
        if value.get("status") != "completed" or (kind and value.get("kind") != kind):
            raise ValueError("产物尚未完成或类型不匹配。")
        return folder, value

    def list_entries(self, kind="artifacts", limit=40):
        directory = self.root / kind
        names = {"artifacts": "manifest.json", "transactions": "transaction.json",
                 "runs": "run.json", "workflows": "workflow.json"}
        result = []
        if directory.is_dir():
            for folder in sorted(directory.iterdir(), key=lambda p: p.name, reverse=True):
                try:
                    folder.resolve().relative_to(self.root)
                    value = read_json(folder / names[kind])
                    result.append({k: value.get(k) for k in ("id", "kind", "title", "created_at", "status", "hidden")})
                except (OSError, ValueError, KeyError):
                    continue
        return sorted(result, key=lambda row: (row.get('created_at') or '', row.get('id') or ''), reverse=True)[:limit]

    def prepare_transaction(self, title, changes, metadata=None):
        """changes: {project_relative_file: bytes or None}. No original is written."""
        if not changes:
            raise ValueError("没有需要修改的文件。")
        entry_id = identifier()
        folder = self.location("transactions", entry_id)
        folder.mkdir(parents=True, exist_ok=False)
        entries = []
        for relative, after in changes.items():
            target = self.source(relative)
            # This layer intentionally has no general arbitrary-file write tool.
            if not ((target.parent == self.project / "jsons" and target.suffix.lower() == ".json")
                    or target in {self.project / "label.txt", self.project / "sample_tree.json", self.project / "flagfile.dat"}):
                raise ValueError("修改仅限工程标注与类别文件。")
            before = target.read_bytes() if target.exists() else None
            expected = (metadata or {}).get("expected_tokens", {})
            if relative in expected and document_token(target, before) != expected[relative]:
                raise ValueError("生成预览期间源文件发生变化：" + relative)
            entries.append({"path": relative, "before": base64.b64encode(before).decode() if before is not None else None,
                            "after": base64.b64encode(after).decode() if after is not None else None,
                            "before_token": document_token(target, before), "after_token": document_token(target, after)})
        value = {"id": entry_id, "kind": "annotation_change", "title": title, "created_at": stamp(),
                 "status": "prepared", "metadata": metadata or {}, "entries": entries}
        write_json(folder / "transaction.json", value)
        return {"transaction_id": entry_id, "title": title, "files": len(entries),
                "changes": (metadata or {}).get("changes", []), "status": "prepared"}

    def commit(self, entry_id, cancelled=lambda: False):
        folder = self.location("transactions", entry_id)
        value = read_json(folder / "transaction.json")
        if value["status"] != "prepared":
            raise ValueError("此操作已执行或失效，不能重复执行。")
        if cancelled():
            raise ValueError("操作已停止。")
        guards = value.get("metadata", {}).get("image_hashes", {})
        for relative, expected in value.get("metadata", {}).get("state_hashes", {}).items():
            if file_sha(self.source(relative)) != expected:
                raise ValueError("工程样本列表发生变化，请重新生成方案。")
        for relative, expected in guards.items():
            if file_sha(self.source(relative)) != expected:
                raise ValueError("图片已变化，请重新生成方案：" + relative)
        for entry in value["entries"]:
            target = self.source(entry["path"])
            current = target.read_bytes() if target.exists() else None
            if document_token(target, current) != entry["before_token"]:
                raise ValueError("文件已被修改，本次不会覆盖：" + entry["path"])
        value["status"] = "applying"
        value["applied"] = []
        write_json(folder / "transaction.json", value)
        try:
            # Once commit begins, finish or roll back the transaction as a unit.
            for entry in value["entries"]:
                target = self.source(entry["path"])
                current = target.read_bytes() if target.exists() else None
                if document_token(target, current) != entry["before_token"]:
                    raise ValueError("提交时文件发生变化：" + entry["path"])
                if entry["after"] is None:
                    target.unlink(missing_ok=True)
                else:
                    atomic_bytes(target, base64.b64decode(entry["after"]))
                value["applied"].append(entry["path"])
                write_json(folder / "transaction.json", value)
            value.update(status="committed", finished_at=stamp())
            write_json(folder / "transaction.json", value)
        except Exception:
            conflicts = []
            for entry in reversed(value["entries"]):
                if entry["path"] not in value["applied"]:
                    continue
                target = self.source(entry["path"])
                current = target.read_bytes() if target.exists() else None
                if document_token(target, current) != entry["after_token"]:
                    conflicts.append(entry["path"])
                    continue
                if entry["before"] is None:
                    target.unlink(missing_ok=True)
                else:
                    atomic_bytes(target, base64.b64decode(entry["before"]))
            value.update(status="recovery_required" if conflicts else "rolled_back", conflicts=conflicts)
            write_json(folder / "transaction.json", value)
            raise
        return {"transaction_id": entry_id, "status": "committed", "title": value["title"],
                "changed_paths": [e["path"] for e in value["entries"]],
                "affected_images": value.get("metadata", {}).get("affected_images", []),
                "set_flags": value.get("metadata", {}).get("set_flags", {})}

    def prepare_undo(self, entry_id):
        value = read_json(self.location("transactions", entry_id) / "transaction.json")
        if value["status"] not in {"committed", "applying", "recovery_required"}:
            raise ValueError("只能恢复已完成的修改。")
        state_hashes = value.get("metadata", {}).get("state_hashes", {})
        for relative, expected in state_hashes.items():
            if file_sha(self.source(relative)) != expected:
                raise ValueError("工程样本列表发生变化，不能恢复旧的图片状态。")
        changes, expected_tokens = {}, {}
        for entry in value["entries"]:
            target = self.source(entry["path"])
            data = target.read_bytes() if target.exists() else None
            actual = document_token(target, data)
            if value["status"] != "committed" and actual == entry["before_token"]:
                continue
            if actual != entry["after_token"]:
                raise ValueError("后续已有新修改，不能覆盖恢复：" + entry["path"])
            changes[entry["path"]] = base64.b64decode(entry["before"]) if entry["before"] is not None else None
            expected_tokens[entry["path"]] = actual
        return self.prepare_transaction("恢复：" + value["title"], changes,
                                        {"undo_of": entry_id, "affected_images": [p for p in value.get("metadata", {}).get("affected_images", [])
                                                                                if "jsons/" + Path(p).stem + ".json" in changes or 'flagfile.dat' in changes],
                                         "expected_tokens": expected_tokens,
                                         "state_hashes": state_hashes,
                                         "set_flags": value.get("metadata", {}).get("previous_flags", {}),
                                         "previous_flags": value.get("metadata", {}).get("set_flags", {}),
                                         "changes": [{"path": e["path"], "action": "恢复修改前内容"} for e in value["entries"]]})
