"""Hide or delete owned result files; never follow source references or links."""
import stat
from pathlib import Path

from Utils.AIWorkspace import read_json, write_json, json_bytes, sha, stamp


VISUAL_KINDS = {"candidates", "crops", "dataset", "converted", "training_records", "selection", "error_set"}
MANAGED_KINDS = VISUAL_KINDS | {'export'}
DELETION_STATES = {"completed", "deleting", "delete_failed"}


def _checked_path(path, root):
    path.relative_to(root)
    for part in (path, *path.parents):
        if part == root:
            break
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("结果目录含符号链接或目录联接，不能自动清理。")
    if path.resolve() != path:
        raise ValueError("结果路径发生变化，不能自动清理。")
    return path.lstat()


def _result(store, artifact_id):
    # Validate the ID, then use the exact project-owned location, not a resolved
    # alias or any path supplied in a source index / manifest.
    store.location("artifacts", artifact_id)
    folder = store.project / "ai_workbench" / "artifacts" / artifact_id
    _checked_path(folder, store.project)
    _checked_path(folder / "manifest.json", store.project)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("id") != artifact_id or manifest.get("kind") not in MANAGED_KINDS:
        raise ValueError("不是可管理的图片处理结果。")
    return folder, manifest


def set_result_hidden(store, artifact_id, hidden):
    folder, manifest = _result(store, artifact_id)
    if manifest.get("status") != "completed":
        raise ValueError("此结果已删除或不可用，请在工作记录中查看状态。")
    if bool(manifest.get("hidden")) != bool(hidden):
        manifest["hidden"] = bool(hidden)
        write_json(folder / "manifest.json", manifest)
    return manifest


def deletion_preview(store, artifact_id):
    folder, manifest = _result(store, artifact_id)
    if manifest.get("status") not in DELETION_STATES:
        raise ValueError("只可删除已完成的处理结果；已删除或仍在生成的结果不能删除。")
    files, directories, signature = [], [], []
    pending = [folder]
    while pending:
        directory = pending.pop()
        _checked_path(directory, store.project)
        for child in sorted(directory.iterdir()):
            info = _checked_path(child, store.project)
            relative = child.relative_to(folder).as_posix()
            signature.append((relative, info.st_size, info.st_mtime_ns, info.st_ino))
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
                pending.append(child)
            elif stat.S_ISREG(info.st_mode):
                if relative != "manifest.json":
                    files.append({"path": relative, "size": info.st_size})
            else:
                raise ValueError("结果目录含非普通文件，不能自动清理。")
    public = None
    if manifest.get('export_category'):
        from Utils.AIExportStorage import export_root
        public = export_root(store, manifest)
        if public.exists():
            pending = [public]
            directories.append('@export')
            info = _checked_path(public, store.project)
            signature.append(('@export', info.st_size, info.st_mtime_ns, info.st_ino))
            while pending:
                directory = pending.pop()
                _checked_path(directory, store.project)
                for child in sorted(directory.iterdir()):
                    info = _checked_path(child, store.project)
                    relative = '@export/' + child.relative_to(public).as_posix()
                    signature.append((relative, info.st_size, info.st_mtime_ns, info.st_ino))
                    if stat.S_ISDIR(info.st_mode):
                        directories.append(relative)
                        pending.append(child)
                    elif stat.S_ISREG(info.st_mode):
                        files.append({'path': relative, 'size': info.st_size})
                    else:
                        raise ValueError('导出目录含非普通文件，不能自动清理。')
    return {"artifact_id": artifact_id, "folder": str(folder), "manifest": manifest,
            'export_folder': str(public) if public else None,
            "files": files, "directories": directories,
            "bytes": sum(f["size"] for f in files),
            "token": sha(json_bytes([manifest, sorted(signature)]))}


def delete_result(store, artifact_id, expected_token):
    plan = deletion_preview(store, artifact_id)
    if plan["token"] != expected_token:
        raise ValueError("结果文件在确认期间发生变化，请重新查看删除清单。")
    folder = Path(plan["folder"])
    manifest = plan["manifest"]
    manifest.update(status="deleting", hidden=True)
    write_json(folder / "manifest.json", manifest)
    try:
        def owned_path(relative):
            if relative == '@export' or relative.startswith('@export/'):
                return Path(plan['export_folder']) / relative[len('@export/'):]
            return folder / relative
        for item in plan["files"]:
            path = owned_path(item['path'])
            _checked_path(path, store.project)
            path.unlink()
        for relative in sorted(plan["directories"], key=lambda p: len(Path(p).parts), reverse=True):
            directory = owned_path(relative)
            _checked_path(directory, store.project)
            directory.rmdir()
        if {p.name for p in folder.iterdir()} != {"manifest.json"}:
            raise ValueError("删除期间有新文件写入，请重新查看剩余文件后清理。")
    except (OSError, ValueError) as exc:
        manifest.update(status="delete_failed", deletion_error=str(exc))
        write_json(folder / "manifest.json", manifest)
        raise
    # Keep a small tombstone so history never offers a deleted output as usable.
    deleted = {k: manifest[k] for k in ("id", "kind", "title", "created_at")}
    deleted.update(status="deleted", hidden=True, deleted_at=stamp(),
                   note="处理结果文件已删除，不能重新打开；原始图片、正式标注和修改恢复记录保留。")
    write_json(folder / "manifest.json", deleted)
    return deleted
