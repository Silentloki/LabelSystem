"""Project-local conversations and source-reference groups (no credentials)."""
import copy
import re
from pathlib import Path

from Utils.AIWorkspace import Workspace, identifier, stamp, read_json, write_json


def load_groups(project, paths):
    file = Workspace(project).root / "selections.json"
    if not file.exists():
        return {}
    value = read_json(file)
    if not isinstance(value, dict):
        raise ValueError("筛选分组记录格式无效。")
    known = set(paths)
    groups = {}
    for gid, group in value.items():
        if (isinstance(gid, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", gid)
                and isinstance(group, dict) and isinstance(group.get("paths"), list)
                and all(isinstance(p, str) for p in group["paths"])):
            groups[gid] = {"title": str(group.get("title", "筛选结果")),
                           "criteria": str(group.get("criteria", "")), "code": str(group.get("code", "")),
                           "paths": [p for p in group["paths"] if p in known]}
    return groups


def save_groups(project, groups):
    write_json(Workspace(project).root / "selections.json", groups)


class ConversationStore:
    def __init__(self, project):
        self.workspace = Workspace(project)
        self.root = self.workspace.root / "conversations"
        self.root.resolve().relative_to(self.workspace.project)

    def path(self, cid):
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", cid):
            raise ValueError("对话编号无效。")
        path = self.root / (cid + ".json")
        path.resolve().relative_to(self.root.resolve())
        return path

    def new(self):
        return {"version": 1, "id": identifier(), "title": "新对话", "created_at": stamp(),
                "updated_at": stamp(), "messages": [], "context": [], "resources": {}, "draft": "", "view": None}

    def read(self, cid):
        value = read_json(self.path(cid))
        if not isinstance(value, dict) or value.get("id") != cid or not isinstance(value.get("messages"), list):
            raise ValueError("对话记录损坏，原文件已保留。")
        if any(not isinstance(value.get(k), str) for k in ("title", "created_at", "updated_at")):
            raise ValueError("对话索引无效，原文件已保留。")
        for row in value["messages"]:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not isinstance(row.get("id"), str):
                raise ValueError("对话消息格式无效，原文件已保留。")
            if not isinstance(row.get("cards", []), list) or any(
                    not isinstance(c, dict) or any(not isinstance(c.get(k), str) for k in ("type", "id", "title", "summary"))
                    for c in row.get("cards", [])):
                raise ValueError("对话结果索引无效，原文件已保留。")
        if not isinstance(value.get("resources", {}), dict) or not isinstance(value.get("context", []), list):
            raise ValueError("对话上下文无效，原文件已保留。")
        for resource in value.get("resources", {}).values():
            if not isinstance(resource, dict) or not isinstance(resource.get("path", resource.get("project_relative")), str):
                raise ValueError("对话资源记录无效，原文件已保留。")
        return value

    def current(self):
        path = self.root / "current.json"
        if path.exists():
            cid = read_json(path).get("id")
            return self.read(cid)
        existing = self.list()
        if existing:
            return self.read(existing[0]["id"])
        legacy = self.import_legacy()
        if legacy:
            return legacy
        return self.new()

    def save(self, conversation, secret=""):
        fields = {"version", "id", "title", "created_at", "updated_at", "messages", "context", "resources", "draft", "view"}
        value = copy.deepcopy({k: v for k, v in conversation.items() if k in fields})
        # Credentials/settings never enter this schema. Also scrub any pasted
        # current Key from message text, paths or model replies before writing.
        from Utils.AIWorkAgent import redact
        value = redact(value, secret)
        write_json(self.path(value["id"]), value)
        write_json(self.root / "current.json", {"id": value["id"]})

    def import_legacy(self):
        """Older versions saved execution evidence, not complete transcripts."""
        entries = self.workspace.list_entries("runs", limit=1000)
        records = []
        for entry in reversed(entries):
            if entry.get("kind") != "agent":
                continue
            try:
                record = read_json(self.workspace.location("runs", entry["id"]) / "run.json")
                if record.get("request"):
                    records.append(record)
            except (OSError, ValueError):
                continue
        if not records:
            return None
        value = self.new()
        value["title"] = "以前的工作（由旧记录整理）"
        value["messages"].append({"id": identifier(), "role": "assistant", "created_at": stamp(),
                                  "text": "旧版本未保存完整聊天。这里按最近1000条执行记录中可读取的对话任务整理工作过程，原记录保留。"})
        states = {"completed": "已完成", "failed": "未完成", "partial": "部分完成", "stopped": "已停止",
                  "running": "未结束", "awaiting_confirmation": "停在待确认，未自动继续", "limit_reached": "达到请求上限"}
        for record in records:
            when = record.get("created_at", stamp())
            value["messages"].append({"id": identifier(), "role": "user", "text": record["request"], "created_at": when})
            cards, seen = [], set()
            for step in record.get("steps", []):
                card = result_card(step.get("result", {}))
                if card and (card["type"], card["id"]) not in seen:
                    cards.append(card)
                    seen.add((card["type"], card["id"]))
            if record.get("local_result"):
                card = result_card(record["local_result"])
                if card:
                    cards.append(card)
            text = "旧执行记录 · " + states.get(record.get("status"), "请查看记录")
            value["messages"].append({"id": identifier(), "role": "assistant", "text": text, "created_at": when,
                                      "cards": cards, "run_id": record["id"]})
        self.save(value)
        return value

    def list(self, query=""):
        results = []
        if not self.root.exists():
            return results
        for path in self.root.glob("*.json"):
            if path.name == "current.json":
                continue
            try:
                value = self.read(path.stem)
            except (OSError, ValueError, TypeError):
                continue
            match = next((m for m in value["messages"] if query.casefold() in m["text"].casefold()), None) if query else None
            if query and query.casefold() not in value["title"].casefold() and match is None:
                continue
            results.append({"id": value["id"], "title": value["title"], "updated_at": value["updated_at"],
                            "snippet": (match or (value["messages"][-1] if value["messages"] else {})).get("text", "还没有消息"),
                            "anchor": match["id"] if match else None})
        return sorted(results, key=lambda row: row["updated_at"], reverse=True)

    def encode_resources(self, resources):
        result = copy.deepcopy(resources)
        for resource in result.values():
            try:
                relative = Path(resource["path"]).resolve().relative_to(self.workspace.project)
            except ValueError:
                continue
            resource["project_relative"] = relative.as_posix()
            resource.pop("path", None)
        return result

    def decode_resources(self, resources):
        result = copy.deepcopy(resources)
        for resource in result.values():
            relative = resource.pop("project_relative", None)
            if relative is not None:
                resource["path"] = str(self.workspace.source(relative))
        return result


def result_card(result):
    """Small human-facing summary; paths / protocol fields stay in details."""
    aid = result.get("artifact_id")
    gid = result.get("group_id")
    if not aid and not gid:
        return None
    kind = result.get("kind")
    title = result.get("title", "处理结果")
    count = result.get("count", 0)
    if kind == "candidates":
        title = "模型推理结果"
        text = f"完成 {count} 张 · 预测 {result.get('predicted_annotations', 0)} 处 · 失败 {result.get('failed', 0)} 张"
    elif kind == "crops":
        text = (f"{result['source_count']} 张原图 → " if 'source_count' in result else '') + f"{count} 张结果图片 · 标注已同步"
    elif kind == "dataset":
        splits = result.get("splits")
        text = (f"训练集 {splits.get('train', 0)} 张 · 验证集 {splits.get('val', 0)} 张" if splits
                else f"已导出 {count} 张图片及对应标注")
        if result.get("skipped"):
            text += f" · 跳过 {result['skipped']} 张"
    elif kind == 'export':
        text = f"已导出 {count} 项 · {title} · 不划分训练集/验证集"
    elif kind == "training_records":
        text = "训练指标与图表记录，可打开核对。"
    elif kind == "selection":
        text = f"筛选快照 · {count} 张处理结果 · 未复制图片"
    elif kind == 'error_set':
        text = f"问题样本集 · {count} 张 · 保留来源引用，未复制图片"
    elif gid:
        text = f"筛选快照 · {count} 张原图 · 未复制图片"
    else:
        text = f"已保存 {count} 项结果"
    return {"type": "artifact" if aid else "group", "id": aid or gid, "kind": kind,
            "title": title, "summary": text}
