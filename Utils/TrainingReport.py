"""Business-facing training report: class evidence, local arithmetic and readable HTML."""
import html
import json
import math
from pathlib import Path, PurePosixPath

import yaml


CONTEXT_FILE = "ai_training_analysis/context.json"


def names_list(value):
    if isinstance(value, dict):
        try:
            value = [value[k] for k in sorted(value, key=lambda k: int(k))]
        except (ValueError, TypeError):
            return []
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        return []
    return list(dict.fromkeys(v.strip() for v in value if v.lower() not in ("background", "all classes")))


def inputs(folder, args, record, file_hash):
    folder = Path(folder)
    context_path = folder / CONTEXT_FILE
    context = json.loads(context_path.read_text(encoding="utf-8-sig")) if context_path.is_file() else {}
    part = context.get("label", context.get("part", ""))
    if not isinstance(part, str) or len(part) > 100:
        raise ValueError("检测部位名称无效")
    names = names_list(record.get("names", []))
    source = "训练时保存的类别" if names else ""
    hashes = {CONTEXT_FILE: file_hash(context_path)} if context_path.is_file() else {}
    # Never infer category names or the part from the currently open, possibly unrelated project.
    candidates = [folder / "data.yaml"]
    data = args.get("data")
    if isinstance(data, str) and Path(data).is_absolute():
        candidates.append(Path(data))
    for path in candidates:
        if names:
            break
        if path.is_file() and path.stat().st_size < 1024 * 1024:
            value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            names = names_list(value.get("names", [])) if isinstance(value, dict) else []
            if names:
                source = "当前可读取的 data.yaml（非历史快照）"
                hashes["class_definitions"] = file_hash(path)
    data_path = PurePosixPath(str(args.get("data", "")).replace("\\", "/"))
    dataset_name = data_path.stem if data_path.stem not in ("data", "dataset", "") else ""
    if not dataset_name:
        dataset_name = next((p.name for p in data_path.parents if p.name.lower() not in
                             ("", "dataset", "datasets", "data", "train", "val")), "")
    label = part.strip() or (f"{dataset_name} · {folder.name}" if dataset_name and dataset_name != folder.name else folder.name)
    charts = []
    # Two task-specific charts, no source photographs or contact sheets.
    # Class counts are object/box-level even for segmentation; don't mix mask AP into the same table.
    pr_name = "BoxPR_curve.png"
    for key, name, kind in (("plot_pr", pr_name, "ap50"),
                            ("plot_confusion", "confusion_matrix.png", "confusion_counts")):
        if (kind == "ap50" and record.get("per_class")) or (kind == "confusion_counts" and record.get("confusion_counts")):
            continue
        path = folder / name
        if path.is_file() and path.stat().st_size <= 12 * 1024 * 1024:
            charts.append({"key": key, "path": str(path), "filename": name, "kind": kind})
            hashes[name] = file_hash(path)
    return {"display_name": label, "user_label": part.strip(), "run_name": folder.name,
            "dataset_name": dataset_name, "names": names, "names_source": source}, charts, hashes


def training_relations(run):
    best, last = run["best"], run["last"]
    changes = {}
    if best:
        for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss", run["primary"]):
            a, b = best.get(key), last.get(key)
            if a is not None and b is not None:
                changes[key] = {"from": a, "to": b, "delta": round(b - a, 7),
                                "direction": "上升" if b > a else "下降" if b < a else "不变"}
    patience = run["args"].get("patience")
    gap = last["epoch"] - best["epoch"] if best else None
    plausible = isinstance(patience, (int, float)) and patience > 0 and gap is not None and gap >= patience
    return {"best_to_last": changes, "epochs_since_best": gap, "patience": patience,
            "early_stop_possible": plausible,
            "interpretation": ("最高分后经过的轮次符合 patience 早停的可能条件；未保存日志，不能确认原因，也不能判定异常中断。"
                               if plausible else "轮次数量本身不能证明训练异常中断或过拟合。")}


def validate_readings(readings, facts):
    if not isinstance(readings, list) or len(readings) > 4:
        raise ValueError("图表读取格式无效")
    clean, seen = [], set()
    for item in readings:
        if not isinstance(item, dict):
            raise ValueError("图表读取条目无效")
        ref = item.get("evidence")
        fact = facts.get(ref, {})
        kind = fact.get("value", {}).get("kind") if isinstance(fact.get("value"), dict) else None
        if kind not in ("ap50", "confusion_counts") or ref in seen:
            raise ValueError("引用了未发送的图表或重复读取")
        seen.add(ref)
        if item.get("readable") is False:
            clean.append({"evidence": ref, "readable": False, "reason": str(item.get("reason", "图中文字读不清"))[:200]})
            continue
        if item.get("readable") is not True:
            raise ValueError("图表未说明是否可读")
        names = item.get("names")
        if not isinstance(names, list) or not names or len(names) > 100 or len(names) != len(set(map(str, names))):
            raise ValueError("图表类别列表无效")
        if any(not isinstance(v, str) or not v.strip() or len(v) > 100 for v in names):
            raise ValueError("图表类别名称无效")
        row = {"evidence": ref, "readable": True, "kind": kind, "names": names}
        if kind == "ap50":
            values = item.get("values")
            if any(v.lower() in ("background", "all classes") for v in names):
                raise ValueError("背景或总体平均不能作为缺陷类别")
            if not isinstance(values, list) or len(values) != len(names) or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("PR 图逐类 AP50 数值无效")
            row["values"] = values
        else:
            if names[-1] != "background" or any(v.lower() in ("background", "all classes") for v in names[:-1]):
                raise ValueError("混淆矩阵必须明确最后一行/列为 background")
            if item.get("row_axis") != "predicted" or item.get("column_axis") != "true":
                raise ValueError("混淆矩阵方向未确认，不能计算漏检或误报")
            matrix = item.get("matrix")
            size = len(names)
            if not isinstance(matrix, list) or len(matrix) != size or any(
                    not isinstance(r, list) or len(r) != size or any(type(v) is not int or v < 0 for v in r)
                    for r in matrix):
                raise ValueError("混淆矩阵必须是原始计数，不能把归一化比例当数量")
            if matrix[-1][-1] != 0:
                raise ValueError("检测混淆矩阵背景/背景格应为空或 0，请核对图表")
            row.update(matrix=matrix, row_axis="predicted", column_axis="true")
        expected = facts.get(ref.split(".")[0] + ".context", {}).get("value", {}).get("names", [])
        actual = names[:-1] if kind == "confusion_counts" else names
        if expected and not set(actual).issubset(expected):
            raise ValueError("图表类别与已有类别定义不一致，请核对目录中的文件是否来自同次实验")
        clean.append(row)
    # Chart order may differ, class identity may not.
    for prefix in ("C", "P"):
        groups = [set(r["names"][:-1] if r["kind"] == "confusion_counts" else r["names"])
                  for r in clean if r["readable"] and r["evidence"].startswith(prefix + ".")]
        if len(groups) > 1 and groups[0] != groups[1]:
            raise ValueError("两张图的类别不一致，不能合并成逐类结论")
    return clean


def class_rows(run, readings=()):
    rows = {name: {"name": name, "sources": []} for name in run.get("context", {}).get("names", [])}
    per_class = run["record"].get("per_class", [])
    for item in per_class if isinstance(per_class, list) else []:
        if not isinstance(item, dict):
            continue
        name = item.get("Class")
        if not isinstance(name, str):
            continue
        row = rows.setdefault(name, {"name": name, "sources": []})
        def metric(key):
            value = item.get(key)
            return value if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 else None
        row.update({"ap50": metric("mAP50"), "precision": metric("Box-P"),
                    "recall": metric("Box-R"), "instances": item.get("Instances"), "ap_source": "结构化验证记录"})
        row["sources"].append("C.per_class")
    count_reading = run["record"].get("confusion_counts")
    all_readings = list(readings)
    if count_reading:
        try:
            checked = validate_readings([dict(count_reading, readable=True, evidence="C.confusion")],
                                        {"C.confusion": {"value": {"kind": "confusion_counts"}}})
            all_readings = checked + all_readings
        except (TypeError, ValueError):
            pass
    for item in all_readings:
        if not item.get("readable") or not item["evidence"].startswith("C."):
            continue
        if item["kind"] == "ap50":
            for name, value in zip(item["names"], item["values"]):
                row = rows.setdefault(name, {"name": name, "sources": []})
                if row.get("ap50") is None:
                    row.update(ap50=value, ap_source="PR 图读数（近似）")
                    row["sources"].append(item["evidence"])
        else:
            names, matrix = item["names"][:-1], item["matrix"]
            for i, name in enumerate(names):
                row = rows.setdefault(name, {"name": name, "sources": []})
                total = sum(r[i] for r in matrix)
                row.update(instances=total, correct=matrix[i][i], missed=matrix[-1][i],
                           wrong_class=sum(matrix[j][i] for j in range(len(names)) if j != i),
                           background_false=matrix[i][-1],
                           confused_with={n: matrix[j][i] for j, n in enumerate(names) if j != i and matrix[j][i]},
                           count_source="结构化混淆矩阵" if item["evidence"] == "C.confusion" else "混淆矩阵图读数")
                row["sources"].append(item["evidence"])
    return list(rows.values())


def class_message(row):
    if "correct" in row:
        if row["instances"] == 0:
            return f"验证集没有这类目标，暂不能评估检出能力；另有 {row['background_false']} 个额外误报。"
        confused = "、".join(f"{name} {number} 个" for name, number in row["confused_with"].items())
        wrong = "错认成 " + confused if confused else "没有错认成其他缺陷"
        return f"实际 {row['instances']} 个：检对 {row['correct']}，漏检 {row['missed']}；{wrong}。另有 {row['background_false']} 个额外误报。"
    if row.get("precision") is not None and row.get("recall") is not None:
        return f"验证记录：检出结果的精确率 {row['precision']:.1%}，真实目标的召回率 {row['recall']:.1%}。"
    if row.get("ap50") is not None:
        return "已读到该类综合检测分数，缺少检对、漏检和误报的数量。"
    return "缺少该类验证数值，暂不能判断表现。"


def render_html(run, previous, report):
    esc = lambda text: html.escape(str(text)).replace("\n", "<br>")
    result = report.get("result", {}) if report and report.get("status") == "completed" else {}
    readings = result.get("chart_readings", [])
    rows = class_rows(run, readings)
    part = run.get("context", {}).get("display_name") or run["name"]
    output = ['<style>body{font-size:14px;color:#202938}h2,h3{color:#185b83}td,th{padding:9px}p{line-height:145%}small{color:#687585}</style>']
    output.append("<h2>" + esc(part) + " · 训练结果</h2>")
    output.append("<p>" + (f"检查 {len(rows)} 类缺陷：" + esc("、".join(r["name"] for r in rows)) if rows else
                           "缺陷类别：生成分析时从本轮类别记录及图表读取。") + "</p>")
    if result:
        output.append("<p><b>整体情况：</b>" + esc(result["summary"]["text"]) + "</p>")
    output.append("<h3>每种缺陷的结果</h3>")
    if rows:
        output.append('<table width="100%" border="1" cellspacing="0" cellpadding="8" style="border-color:#d5dfe9"><tr bgcolor="#eef4fa"><th>缺陷</th><th>验证结果怎么理解</th><th>检测分数 AP50</th></tr>')
        for row in rows:
            score = (f"{row['ap50']:.3f}" if row.get("ap50") is not None else "未提供")
            output.append("<tr><td><b>" + esc(row["name"]) + "</b></td><td>" + esc(class_message(row)) + "</td><td>" + score + "</td></tr>")
        output.append("</table><p><small>AP50 是综合检测分数，越高越好，不等同“识别正确的百分比”。数量按本轮混淆矩阵的评估条件统计，不能直接当成生产线表现。</small></p>")
        sources = list(dict.fromkeys(source for row in rows for source in row["sources"]))
        source_names = {"C.plot_pr": "查看 PR 原图", "C.plot_confusion": "查看混淆矩阵原图",
                        "C.per_class": "查看按类记录", "C.confusion": "查看矩阵计数"}
        output.append("<p><small>" + ("含图表读数，可点开原图核对。 " if any("plot" in s for s in sources) else "")
                      + " · ".join('<a href="evidence:' + esc(s) + '">' + source_names.get(s, "查看依据") + '</a>' for s in sources) + "</small></p>")
    else:
        output.append("<p>尚未取得逐类结果。点击生成分析后，将读取选定的 PR 图和混淆矩阵。</p>")
    if result:
        if result.get("class_findings"):
            output.append("<h3>各类下一步关注什么</h3>")
            for item in result["class_findings"]:
                output.append("<p><b>" + esc(item["name"]) + "：</b>" + esc(item["text"]) + "</p>")
        if result["findings"]:
            output.append("<h3>还需要留意</h3>")
            for item in result["findings"]:
                output.append("<p>" + esc(item["text"]) + "</p>")
        if previous:
            output.append("<h3>相比上次</h3><p>" + esc(result["comparison"]["text"]) + "</p>")
        experiment = result["next_experiment"]
        output.append("<h3>下一步先做：" + esc(experiment["title"]) + "</h3>")
        for key, label in (("action", "怎么做"), ("hypothesis", "想验证什么"), ("observe", "看什么结果"),
                           ("keep_fixed", "其余保持"), ("stop", "何时停止"), ("cost", "需要的投入")):
            output.append("<p><b>" + label + "：</b>" + esc(experiment[key]) + "</p>")
        output.append("<p><small>分析时间：" + esc(report.get("created_at", "")) + "；详细训练指标和证据范围见“训练记录与依据”。</small></p>")
    elif report and report.get("error"):
        output.append("<p>本次分析未完成：" + esc(report["error"]) + "</p>")
    else:
        output.append("<p>尚未生成新版分析。读取记录和填写部位不调用 AI。</p>")
    return "".join(output)
