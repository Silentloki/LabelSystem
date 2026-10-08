import base64
import csv
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
DEFAULT_QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_QWEN_MODEL = "qwen3-vl-plus"


class AIJsonParseError(RuntimeError):
    def __init__(self, message, raw_text="", response_json=None):
        super().__init__(message)
        self.raw_text = raw_text
        self.response_json = response_json

AUGMENT_METHOD_SPECS = {
    "roi_photometric": {
        "label_change": "keep_all_source_labels",
        "extra_target_boxes_per_image": 0,
        "description": "复制源图为新训练图，只在目标缺陷 ROI 内做亮度、对比度、灰度和轻微噪声扰动；标签文件整体复制。",
    },
    "copy_paste_soft_blend": {
        "label_change": "append_one_target_box",
        "extra_target_boxes_per_image": 1,
        "description": "默认禁用；只有显式 allow_append_target_boxes=true 时才复制目标缺陷 patch 并追加 1 个目标类新框。",
    },
    "keep_original": {
        "label_change": "none",
        "extra_target_boxes_per_image": 0,
        "description": "不生成增强图。",
    },
}

METHOD_ALIASES = {
    "roi_photo": "roi_photometric",
    "roi_photometric": "roi_photometric",
    "copy_paste": "copy_paste_soft_blend",
    "copy_paste_soft_blend": "copy_paste_soft_blend",
    "keep": "keep_original",
    "keep_original": "keep_original",
}


def safe_name(value):
    value = str(value).strip() or "unknown"
    return re.sub(r'[<>:"/\\|?*\s]+', "_", value)


def normalize_method_name(value):
    return METHOD_ALIASES.get(str(value or "").strip(), "")


def normalize_roi_strength(value, default="medium"):
    value = str(value or default).strip().lower()
    if value in {"low", "medium", "high"}:
        return value
    if value in {"保守", "轻", "轻量", "weak"}:
        return "low"
    if value in {"强", "heavy", "strong"}:
        return "high"
    return default


def method_contract_text():
    lines = [
        "本地增强算子合同如下，计划必须严格按这些真实副作用计算：",
    ]
    for method, spec in AUGMENT_METHOD_SPECS.items():
        lines.append(
            f"- {method}: {spec['description']} "
            f"label_change={spec['label_change']}, "
            f"extra_target_boxes_per_image={spec['extra_target_boxes_per_image']}"
        )
    lines.extend(
        [
            "注意：每生成 1 张增强训练图，源标签中的所有框都会作为新训练实例进入 train。",
            "因此 target_new_images 不是 target_box_delta。",
            "默认禁止 copy_paste_soft_blend 追加新目标框；除非计划显式给出 allow_append_target_boxes=true，否则本地会自动降级为 roi_photometric。",
            "不要输出 method=mixed；必须输出 method_mix 明细。",
        ]
    )
    return "\n".join(lines)


def split_count_by_weights(total, weighted_methods):
    total = max(0, int(total))
    methods = [(normalize_method_name(name), max(0.0, float(weight))) for name, weight in weighted_methods]
    methods = [(name, weight) for name, weight in methods if name and name != "keep_original" and weight > 0]
    if total <= 0 or not methods:
        return []
    weight_sum = sum(weight for _, weight in methods)
    raw = []
    assigned = 0
    for name, weight in methods:
        value = int(total * weight / weight_sum)
        raw.append({"method": name, "images": value})
        assigned += value
    remainder = total - assigned
    index = 0
    while remainder > 0 and raw:
        raw[index % len(raw)]["images"] += 1
        remainder -= 1
        index += 1
    return [item for item in raw if item["images"] > 0]


def normalize_method_mix(item, target_new):
    raw_mix = item.get("method_mix") or item.get("methods") or []
    normalized = []
    allow_append_boxes = bool(item.get("allow_append_target_boxes", False) or item.get("allow_copy_paste_append", False))
    if isinstance(raw_mix, list):
        for entry in raw_mix:
            if not isinstance(entry, dict):
                continue
            method = normalize_method_name(entry.get("method"))
            if method == "copy_paste_soft_blend" and not allow_append_boxes:
                method = "roi_photometric"
            if method not in AUGMENT_METHOD_SPECS or method == "keep_original":
                continue
            try:
                images = int(entry.get("images", entry.get("target_new_images", 0)))
            except (TypeError, ValueError):
                images = 0
            if images > 0:
                normalized.append({"method": method, "images": images})

    if not normalized and target_new > 0:
        method = str(item.get("method", "")).strip()
        allow_copy_paste = bool(item.get("allow_copy_paste", method in {"mixed", "copy_paste_soft_blend"}))
        if method == "mixed":
            if allow_copy_paste and allow_append_boxes:
                normalized = split_count_by_weights(target_new, [("copy_paste_soft_blend", 0.45), ("roi_photometric", 0.55)])
            else:
                normalized = [{"method": "roi_photometric", "images": target_new}]
        else:
            method = normalize_method_name(method)
            if method == "copy_paste_soft_blend" and not allow_append_boxes:
                method = "roi_photometric"
            if method in AUGMENT_METHOD_SPECS and method != "keep_original":
                normalized = [{"method": method, "images": target_new}]

    total = sum(int(entry["images"]) for entry in normalized)
    if target_new <= 0:
        return []
    if total <= 0:
        return [{"method": "roi_photometric", "images": target_new}]
    if total != target_new:
        scale = target_new / total
        adjusted = []
        for entry in normalized:
            images = int(round(entry["images"] * scale))
            adjusted.append({"method": entry["method"], "images": max(0, images)})
        while sum(entry["images"] for entry in adjusted) < target_new and adjusted:
            index = sum(entry["images"] for entry in adjusted) % len(adjusted)
            adjusted[index]["images"] += 1
        while sum(entry["images"] for entry in adjusted) > target_new and adjusted:
            largest = max(range(len(adjusted)), key=lambda idx: adjusted[idx]["images"])
            if adjusted[largest]["images"] <= 0:
                break
            adjusted[largest]["images"] -= 1
        normalized = [entry for entry in adjusted if entry["images"] > 0]
    return normalized


def method_mix_text(item):
    mix = item.get("method_mix") or []
    if not mix:
        return item.get("method", "")
    return ", ".join(f"{entry['method']}:{entry['images']}" for entry in mix)


def expected_target_delta_for_mix(stat, method_mix):
    current_instances = int(stat.get("instances", 0))
    current_images = int(stat.get("image_count", 0))
    avg_target_boxes = (current_instances / current_images) if current_images else 0.0
    total = 0.0
    for entry in method_mix or []:
        method = normalize_method_name(entry.get("method"))
        images = max(0, int(entry.get("images", 0)))
        spec = AUGMENT_METHOD_SPECS.get(method, AUGMENT_METHOD_SPECS["roi_photometric"])
        total += images * avg_target_boxes
        total += int(spec.get("extra_target_boxes_per_image", 0)) * images
    return round(total, 3)


def unique_output_dir(project_dir):
    base = Path(project_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = base / f"dataset_ai_aug_{stamp}"
    index = 1
    while root.exists():
        root = base / f"dataset_ai_aug_{stamp}_{index}"
        index += 1
    return root


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_data_yaml_names(data_yaml_path):
    path = Path(data_yaml_path)
    if not path.exists():
        return []
    lines = read_text(path).splitlines()
    names = {}
    in_names = False
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("names:"):
            in_names = True
            inline = stripped[len("names:"):].strip()
            if inline.startswith("[") and inline.endswith("]"):
                values = [item.strip().strip("'\"") for item in inline[1:-1].split(",")]
                return [v for v in values if v]
            continue
        if in_names:
            if not raw.startswith((" ", "\t")):
                break
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            try:
                idx = int(key.strip().strip("'\""))
            except ValueError:
                continue
            names[idx] = value.strip().strip("'\"")
    return [names[i] for i in sorted(names)]


def write_data_yaml_from_names(dataset_dir, names):
    lines = ["train: images/train", "val: images/val", "", f"nc: {len(names)}", "names:"]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    (Path(dataset_dir) / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_image_for_label(images_train_dir, label_path):
    stem = Path(label_path).stem
    for ext in IMAGE_EXTS:
        image_path = Path(images_train_dir) / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = parts[0]
    try:
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(values) == 4:
        cx, cy, w, h = values
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return {"class_id": cls_id, "kind": "box", "points": points}
    if len(values) >= 6 and len(values) % 2 == 0:
        points = list(zip(values[0::2], values[1::2]))
        return {"class_id": cls_id, "kind": "polygon", "points": points}
    return None


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def format_label_line(class_id, points, kind):
    points = [(clamp01(x), clamp01(y)) for x, y in points]
    if kind == "box":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return f"{class_id} {cx:.6f} {cy:.6f} {(x2 - x1):.6f} {(y2 - y1):.6f}"
    values = []
    for x, y in points:
        values.append(f"{x:.6f}")
        values.append(f"{y:.6f}")
    return f"{class_id} " + " ".join(values)


def points_to_pixels(points, width, height):
    return np.array(
        [[int(round(clamp01(x) * width)), int(round(clamp01(y) * height))] for x, y in points],
        dtype=np.int32,
    )


def pixel_points_to_norm(points, width, height):
    return [(clamp01(x / width), clamp01(y / height)) for x, y in points]


def polygon_bbox_pixels(points_px, width, height, pad=4):
    x, y, w, h = cv2.boundingRect(points_px)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_from_points(points_px, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points_px) >= 3:
        cv2.fillPoly(mask, [points_px], 255)
    return mask


def read_label_file(label_path):
    if not Path(label_path).exists():
        return []
    return [line.strip() for line in read_text(label_path).splitlines() if line.strip()]


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def collect_dataset_records(dataset_dir, class_names):
    dataset_dir = Path(dataset_dir)
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    records = []
    class_stats = {
        str(idx): {"class_id": str(idx), "class_name": name, "instances": 0, "images": set(), "area_ratios": []}
        for idx, name in enumerate(class_names)
    }

    for label_path in sorted(labels_train.glob("*.txt")):
        image_path = find_image_for_label(images_train, label_path)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        lines = read_label_file(label_path)
        for line_index, line in enumerate(lines):
            parsed = parse_label_line(line)
            if not parsed or parsed["class_id"] not in class_stats:
                continue
            points = parsed["points"]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            area_ratio = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            stat = class_stats[parsed["class_id"]]
            stat["instances"] += 1
            stat["images"].add(str(image_path))
            stat["area_ratios"].append(area_ratio)
            records.append(
                {
                    "class_id": parsed["class_id"],
                    "class_name": class_names[int(parsed["class_id"])],
                    "image_path": image_path,
                    "label_path": label_path,
                    "line_index": line_index,
                    "line": line,
                    "parsed": parsed,
                    "width": width,
                    "height": height,
                    "area_ratio": area_ratio,
                }
            )

    for stat in class_stats.values():
        stat["image_count"] = len(stat.pop("images"))
        areas = stat.pop("area_ratios")
        stat["mean_area_ratio"] = float(np.mean(areas)) if areas else 0.0
        stat["min_area_ratio"] = float(np.min(areas)) if areas else 0.0
        stat["max_area_ratio"] = float(np.max(areas)) if areas else 0.0
    return records, class_stats


def create_overlay_evidence(record, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    points_px = points_to_pixels(record["parsed"]["points"], width, height)
    bbox = polygon_bbox_pixels(points_px, width, height, pad=14)
    if bbox is None:
        return None
    mask = mask_from_points(points_px, width, height)
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0) * 0.35
    overlay = (overlay * (1.0 - alpha[:, :, None]) + color * alpha[:, :, None]).astype(np.uint8)
    cv2.polylines(overlay, [points_px], True, (0, 0, 255), 2)
    x1, y1, x2, y2 = bbox
    crop = overlay[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    stem = f"cls{int(record['class_id']):03d}_{safe_name(Path(record['image_path']).stem)}_{record['line_index']:04d}"
    full_path = output_dir / f"{stem}_overlay.jpg"
    crop_path = output_dir / f"{stem}_crop.jpg"
    cv2.imwrite(str(full_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"overlay": full_path, "crop": crop_path}


def build_evidence_package(dataset_dir, output_root, class_names, selected_class_ids, max_samples_per_class=3):
    records, class_stats = collect_dataset_records(dataset_dir, class_names)
    selected = set(str(v) for v in selected_class_ids)
    evidence_dir = Path(output_root) / "review" / "evidence"
    evidence_rows = []
    per_class = defaultdict(list)
    for record in records:
        if record["class_id"] in selected:
            per_class[record["class_id"]].append(record)

    image_paths_for_ai = []
    random.seed(42)
    for class_id in selected:
        class_records = sorted(per_class.get(class_id, []), key=lambda r: r["area_ratio"])
        if not class_records:
            continue
        picks = []
        picks.append(class_records[0])
        picks.append(class_records[len(class_records) // 2])
        picks.append(class_records[-1])
        random.shuffle(class_records)
        picks.extend(class_records[:max(0, max_samples_per_class - len(picks))])
        unique = []
        seen = set()
        for record in picks:
            key = (record["image_path"], record["line_index"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(record)
            if len(unique) >= max_samples_per_class:
                break
        class_dir = evidence_dir / f"cls{int(class_id):03d}"
        for record in unique:
            paths = create_overlay_evidence(record, class_dir)
            if not paths:
                continue
            image_paths_for_ai.append(paths["crop"])
            evidence_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[int(class_id)],
                    "image": str(record["image_path"]),
                    "overlay": str(paths["overlay"]),
                    "crop": str(paths["crop"]),
                    "area_ratio": f"{record['area_ratio']:.6f}",
                }
            )

    with open(Path(output_root) / "ai_aug_evidence_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "image", "overlay", "crop", "area_ratio"])
        writer.writeheader()
        writer.writerows(evidence_rows)
    return records, class_stats, image_paths_for_ai


def build_full_evidence_package(dataset_dir, output_root, class_names, selected_class_ids):
    records, class_stats = collect_dataset_records(dataset_dir, class_names)
    selected = set(str(v) for v in selected_class_ids)
    evidence_dir = Path(output_root) / "review" / "evidence"
    evidence_rows = []
    image_paths_for_ai = []

    for record in records:
        if record["class_id"] not in selected:
            continue
        class_dir = evidence_dir / f"cls{int(record['class_id']):03d}"
        paths = create_overlay_evidence(record, class_dir)
        if not paths:
            continue
        image_paths_for_ai.append(paths["crop"])
        evidence_rows.append(
            {
                "class_id": record["class_id"],
                "class_name": record["class_name"],
                "image": str(record["image_path"]),
                "overlay": str(paths["overlay"]),
                "crop": str(paths["crop"]),
                "area_ratio": f"{record['area_ratio']:.6f}",
            }
        )

    with open(Path(output_root) / "ai_aug_evidence_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "image", "overlay", "crop", "area_ratio"])
        writer.writeheader()
        writer.writerows(evidence_rows)
    return records, class_stats, image_paths_for_ai


def image_to_data_url(path):
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def extract_json_object(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("AI 返回内容不是有效 JSON")


def robust_extract_json_object(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    last_error = None
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        for payload in (candidate, repaired):
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                last_error = exc
    raise AIJsonParseError(f"AI 返回内容不是有效 JSON: {last_error}", raw_text=text)


def build_ai_prompt(class_stats, selected_class_ids):
    selected_ids = [str(v) for v in selected_class_ids]
    selected_stats = {cid: class_stats[cid] for cid in selected_ids if cid in class_stats}
    global_stats = {
        cid: {
            "class_name": stat["class_name"],
            "instances": stat["instances"],
            "image_count": stat["image_count"],
            "mean_area_ratio": round(stat["mean_area_ratio"], 6),
        }
        for cid, stat in class_stats.items()
    }
    return (
        "你是工业缺陷数据增强策略专家。请只分析用户选中的缺陷类别和随附的缺陷证据图，"
        "不要建议增强未选中的类别。你需要从全局类别分布出发，为每个选中缺陷给出专属增强方案和建议新增图片数量。\n\n"
        "增强只允许程序支持的可控方法：\n"
        "1. copy_paste_soft_blend：复制已有缺陷 patch，软边融合到同图相近材质区域，标签必须同步新增。\n"
        "2. roi_photometric：只在缺陷 mask/ROI 内做亮度、对比度、灰度、轻微噪声扰动，标签不变。\n"
        "3. mixed：两者结合，优先 copy-paste，失败时退回 ROI 扰动。\n"
        "4. keep_original：不增强。\n\n"
        "请输出严格 JSON，不要 Markdown。格式如下：\n"
        "{\n"
        "  \"classes\": [\n"
        "    {\n"
        "      \"class_id\": \"0\",\n"
        "      \"class_name\": \"缺陷名\",\n"
        "      \"target_new_images\": 10,\n"
        "      \"method\": \"mixed\",\n"
        "      \"allow_copy_paste\": true,\n"
        "      \"risk\": \"low|medium|high\",\n"
        "      \"rationale\": \"为什么这样增强、为什么是这个数量\",\n"
        "      \"notes\": \"必须保留共现标注，增强后只进新数据集 train\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"全局类别统计：{json.dumps(global_stats, ensure_ascii=False)}\n"
        f"选中类别统计：{json.dumps(selected_stats, ensure_ascii=False)}\n"
        "数量建议要克制：小样本可以补到中位类别附近，不要盲目补到最大类；如果样本太少或风险高，数量要保守。"
    )


def request_qwen_json(api_key, base_url, model, prompt, image_paths=None, timeout=120, retries=3, max_tokens=None):
    content = [{"type": "text", "text": prompt}]
    for image_path in (image_paths or []):
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出严格 JSON，用中文写 rationale，不要输出 Markdown。"},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    payload["messages"][0]["content"] = "只输出严格 JSON 对象，不要 Markdown，不要代码块；字符串中的双引号必须转义。"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(1, retries + 1):
        from Utils.AIContextInspector import capture_transport, capture_transport_outcome
        capture_transport(data, base_url, image_paths or [], attempt)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="ignore")
            capture_transport_outcome('received', {'http_status': getattr(response, 'status', None)})
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            capture_transport_outcome('http_error', {'http_status': exc.code, 'message': detail[:1000]})
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"千问 API HTTP {exc.code}: {detail[:1000]}")
            last_error = f"HTTP {exc.code}: {detail[:500]}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = str(exc)
            capture_transport_outcome('connection_error', last_error)

        if attempt < retries:
            time.sleep(min(8, attempt * 2))
    else:
        raise RuntimeError(f"千问 API 连接失败，已重试 {retries} 次: {last_error}")

    response_json = json.loads(body)
    content_text = response_json["choices"][0]["message"]["content"]
    if isinstance(content_text, list):
        content_text = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content_text
        )
    try:
        return robust_extract_json_object(content_text), response_json
    except AIJsonParseError as exc:
        exc.response_json = response_json
        raise


def request_qwen_plan(api_key, base_url, model, prompt, image_paths, timeout=120):
    return request_qwen_json(api_key, base_url, model, prompt, image_paths[:24], timeout=timeout)


def chunked(items, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield start // chunk_size + 1, items[start:start + chunk_size]


def build_batch_analysis_prompt(class_stats, selected_class_ids, batch_index, batch_count):
    selected_ids = [str(v) for v in selected_class_ids]
    selected_stats = {cid: class_stats[cid] for cid in selected_ids if cid in class_stats}
    return (
        "你是工业缺陷数据增强策略专家。现在进行全量分批视觉分析。"
        f"这是第 {batch_index}/{batch_count} 批证据图，所有图片都是用户选中缺陷类别在训练集中的缺陷 crop/标注叠加图。\n"
        "请仔细观察本批图片中的缺陷形态、纹理、位置倾向、背景材质、可能的增强风险。"
        "不要给最终新增数量，先只给本批视觉观察总结。\n\n"
        "只允许输出严格 JSON，格式如下：\n"
        "{\n"
        "  \"batch_index\": 1,\n"
        "  \"class_summaries\": [\n"
        "    {\n"
        "      \"class_id\": \"0\",\n"
        "      \"class_name\": \"缺陷名\",\n"
        "      \"observed_count_estimate\": 10,\n"
        "      \"visual_patterns\": \"本批看到的主要形态/纹理/尺寸变化\",\n"
        "      \"augmentation_suggestions\": \"适合的增强方向\",\n"
        "      \"risks\": \"容易生成假样本或误增强的点\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"选中类别统计：{json.dumps(selected_stats, ensure_ascii=False)}"
    )


def build_final_plan_prompt(class_stats, selected_class_ids, batch_summaries):
    selected_ids = [str(v) for v in selected_class_ids]
    selected_stats = {cid: class_stats[cid] for cid in selected_ids if cid in class_stats}
    global_stats = {
        cid: {
            "class_name": stat["class_name"],
            "instances": stat["instances"],
            "image_count": stat["image_count"],
            "mean_area_ratio": round(stat["mean_area_ratio"], 6),
        }
        for cid, stat in class_stats.items()
    }
    return (
        "你是工业缺陷数据增强策略专家。你已经看完选中缺陷类别训练集的全量分批证据图。"
        "现在请结合全局类别统计、选中类别统计、所有批次视觉总结，给出最终每类专属增强方案和建议新增图片数量。\n\n"
        "增强只允许程序支持的可控方法：\n"
        "1. copy_paste_soft_blend：复制已有缺陷 patch，软边融合到同图相近材质区域，标签必须同步新增。\n"
        "2. roi_photometric：只在缺陷 mask/ROI 内做亮度、对比度、灰度、轻微噪声扰动，标签不变。\n"
        "3. mixed：两者结合，优先 copy-paste，失败时退回 ROI 扰动。\n"
        "4. keep_original：不增强。\n\n"
        "请输出严格 JSON，不要 Markdown。格式如下：\n"
        "{\n"
        "  \"classes\": [\n"
        "    {\n"
        "      \"class_id\": \"0\",\n"
        "      \"class_name\": \"缺陷名\",\n"
        "      \"target_new_images\": 10,\n"
        "      \"method\": \"mixed\",\n"
        "      \"allow_copy_paste\": true,\n"
        "      \"risk\": \"low|medium|high\",\n"
        "      \"rationale\": \"为什么这样增强、为什么是这个数量\",\n"
        "      \"notes\": \"必须保留共现标注，增强后只进新数据集 train\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "数量建议要从全局考虑：不要盲目补到最大类；类别已经足够多时只做轻量增强或不增强；"
        "少样本、高风险类别要保守。\n"
        f"全局类别统计：{json.dumps(global_stats, ensure_ascii=False)}\n"
        f"选中类别统计：{json.dumps(selected_stats, ensure_ascii=False)}\n"
        f"分批视觉总结：{json.dumps(batch_summaries, ensure_ascii=False)}"
    )


def build_stage1_diagnosis_prompt(class_stats, selected_class_ids, batch_index, batch_count):
    selected_ids = [str(v) for v in selected_class_ids]
    selected_stats = {cid: class_stats[cid] for cid in selected_ids if cid in class_stats}
    return (
        "你是工业缺陷数据增强策略专家。现在进行第 1 阶段：数据集视觉诊断。"
        f"这是第 {batch_index}/{batch_count} 批证据图，所有图片都是用户选中缺陷类别在训练集中的缺陷 crop/标注叠加图。\n"
        "请观察缺陷形态、纹理、位置倾向、背景材质、疑似错标、类别边界模糊、子形态/子分类线索和增强风险。"
        "本阶段禁止给最终新增数量，禁止输出可执行增强计划，只做诊断。\n\n"
        "只允许输出严格 JSON，格式如下：\n"
        "{\n"
        "  \"batch_index\": 1,\n"
        "  \"class_summaries\": [\n"
        "    {\n"
        "      \"class_id\": \"0\",\n"
        "      \"class_name\": \"缺陷名\",\n"
        "      \"observed_count_estimate\": 10,\n"
        "      \"visual_patterns\": \"本批看到的主要形态、纹理、尺寸、位置变化\",\n"
        "      \"subtype_hints\": \"即使当前类别未升级为子分类，也要按肉眼形态拆出子形态线索\",\n"
        "      \"confusable_with\": [\"容易混淆的类别 id 或类别名\"],\n"
        "      \"suspect_label_notes\": \"疑似错标、边界模糊、框不准的观察，没有则写空字符串\",\n"
        "      \"safe_augmentation_directions\": \"安全的增强方向\",\n"
        "      \"unsafe_augmentation_directions\": \"容易生成假样本或放大错标的增强方向\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"选中类别统计：{json.dumps(selected_stats, ensure_ascii=False)}"
    )


def build_stage2_plan_prompt(class_stats, selected_class_ids, diagnosis):
    selected_ids = [str(v) for v in selected_class_ids]
    selected_stats = {cid: class_stats[cid] for cid in selected_ids if cid in class_stats}
    global_stats = {
        cid: {
            "class_name": stat["class_name"],
            "instances": stat["instances"],
            "image_count": stat["image_count"],
            "mean_area_ratio": round(stat["mean_area_ratio"], 6),
        }
        for cid, stat in class_stats.items()
    }
    return (
        "你是工业缺陷数据增强策略专家。现在进行第 2 阶段：生成可执行增强计划。"
        "请结合全局类别统计、选中类别统计和第 1 阶段诊断，给出每类专属增强计划。\n\n"
        f"{method_contract_text()}\n\n"
        "必须同时规划 target_new_images 和 expected_target_box_delta。"
        "expected_target_box_delta 是增强结果写入 train 后，目标类训练框数量预计增加多少；它不是图片数。"
        "默认不允许使用 copy_paste_soft_blend 追加新目标框；如果确实要用，必须显式输出 allow_append_target_boxes=true，"
        "并把每张图额外追加 1 个目标框计入 expected_target_box_delta。未显式允许时，本地会自动改为 roi_photometric。"
        "如果类别存在子形态或未升级子分类，也必须在 subtype_plan 中说明各子形态如何覆盖。"
        "策略必须细化到可执行约束：ROI 光度扰动强度、禁用源图特征、标注风险处理、每种子形态适合/禁止的方法。"
        "不要输出 method=mixed；必须输出 method_mix。\n\n"
        "请输出严格 JSON，不要 Markdown。格式如下：\n"
        "{\n"
        "  \"classes\": [\n"
        "    {\n"
        "      \"class_id\": \"0\",\n"
        "      \"class_name\": \"缺陷名\",\n"
        "      \"target_new_images\": 10,\n"
        "      \"expected_target_box_delta\": 10,\n"
        "      \"method_mix\": [\n"
        "        {\"method\": \"roi_photometric\", \"images\": 6, \"expected_target_box_delta\": 6},\n"
        "        {\"method\": \"roi_photometric\", \"images\": 4, \"expected_target_box_delta\": 4}\n"
        "      ],\n"
      "      \"allow_append_target_boxes\": false,\n"
        "      \"roi_photometric_strength\": \"low|medium|high\",\n"
        "      \"forbidden_methods\": [\"不应使用的方法\"],\n"
        "      \"subtype_plan\": \"按视觉子形态/子分类说明覆盖思路\",\n"
        "      \"method_constraints\": \"每种方法的适用边界，例如ROI只允许轻微低频亮度/对比度变化，不允许制造新纹理\",\n"
        "      \"source_filter_rules\": \"哪些源图/源标注应禁用，例如框过大、疑似错标、边界不清、背景占比过高\",\n"
        "      \"label_risk_policy\": \"遇到框不准、类别混淆、共现缺陷时如何处理\",\n"
        "      \"risk\": \"low|medium|high\",\n"
        "      \"rationale\": \"为什么这样增强、为什么是这个数量、为什么这些方法安全\",\n"
        "      \"notes\": \"必须保留共现标注，增强后只进新数据集 train\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "数量建议要从全局考虑：不要盲目补到最多类；类别已经足够多时只做轻量增强或不增强；"
        "少样本、高风险类别要保守。\n"
        f"全局类别统计：{json.dumps(global_stats, ensure_ascii=False)}\n"
        f"选中类别统计：{json.dumps(selected_stats, ensure_ascii=False)}\n"
        f"第 1 阶段诊断：{json.dumps(diagnosis, ensure_ascii=False)}"
    )


def request_qwen_batch_with_split(
    api_key,
    base_url,
    model,
    class_stats,
    selected_class_ids,
    batch_paths,
    batch_label,
    batch_count,
    progress_callback=None,
    min_batch_size=4,
):
    prompt = build_stage1_diagnosis_prompt(class_stats, selected_class_ids, batch_label, batch_count)
    try:
        summary, response = request_qwen_json(api_key, base_url, model, prompt, batch_paths)
        return [summary], [{"stage": "batch", "batch_index": batch_label, "response": response}]
    except Exception as exc:
        if len(batch_paths) <= min_batch_size:
            fallback = {
                "batch_index": batch_label,
                "parse_failed": True,
                "error": str(exc),
                "class_summaries": [],
            }
            response_payload = {
                "stage": "batch",
                "batch_index": batch_label,
                "error": str(exc),
                "image_count": len(batch_paths),
            }
            if isinstance(exc, AIJsonParseError):
                response_payload["raw_text"] = exc.raw_text
                response_payload["response"] = exc.response_json
            if progress_callback:
                progress_callback(f"第 {batch_label} 批解析失败，已记录并继续；图片 {len(batch_paths)} 张。")
            return [fallback], [response_payload]
        mid = len(batch_paths) // 2
        left = batch_paths[:mid]
        right = batch_paths[mid:]
        if progress_callback:
            progress_callback(
                f"第 {batch_label} 批连接失败，自动拆成 {len(left)} + {len(right)} 张重试；原因：{exc}"
            )
        left_summaries, left_responses = request_qwen_batch_with_split(
            api_key,
            base_url,
            model,
            class_stats,
            selected_class_ids,
            left,
            f"{batch_label}.1",
            batch_count,
            progress_callback,
            min_batch_size,
        )
        right_summaries, right_responses = request_qwen_batch_with_split(
            api_key,
            base_url,
            model,
            class_stats,
            selected_class_ids,
            right,
            f"{batch_label}.2",
            batch_count,
            progress_callback,
            min_batch_size,
        )
        return left_summaries + right_summaries, left_responses + right_responses


def request_qwen_diagnosis(api_key, base_url, model, class_stats, selected_class_ids, image_paths, batch_size, progress_callback=None):
    image_paths = list(image_paths)
    batch_size = max(1, int(batch_size))
    batches = list(chunked(image_paths, batch_size))
    batch_summaries = []
    raw_responses = []

    for batch_index, batch_paths in batches:
        if progress_callback:
            progress_callback(f"正在调用千问进行第 1 阶段诊断 {batch_index}/{len(batches)}，图片 {len(batch_paths)} 张...")
        summaries, responses = request_qwen_batch_with_split(
            api_key,
            base_url,
            model,
            class_stats,
            selected_class_ids,
            batch_paths,
            batch_index,
            len(batches),
            progress_callback,
        )
        batch_summaries.extend(summaries)
        raw_responses.extend(responses)
    return {"batch_summaries": batch_summaries}, {"stage": "diagnosis", "raw_responses": raw_responses}


def request_qwen_plan_from_diagnosis(api_key, base_url, model, class_stats, selected_class_ids, diagnosis, progress_callback=None):
    if progress_callback:
        progress_callback("正在调用千问进行第 2 阶段增强计划...")
    prompt = build_stage2_plan_prompt(class_stats, selected_class_ids, diagnosis)
    plan, response = request_qwen_json(api_key, base_url, model, prompt, [])
    return plan, {"stage": "plan", "response": response}


def request_qwen_full_batch_plan(api_key, base_url, model, class_stats, selected_class_ids, image_paths, batch_size, progress_callback=None):
    image_paths = list(image_paths)
    batch_size = max(1, int(batch_size))
    batches = list(chunked(image_paths, batch_size))
    batch_summaries = []
    raw_responses = []

    for batch_index, batch_paths in batches:
        if progress_callback:
            progress_callback(f"正在调用千问分析第 {batch_index}/{len(batches)} 批，图片 {len(batch_paths)} 张...")
        summaries, responses = request_qwen_batch_with_split(
            api_key,
            base_url,
            model,
            class_stats,
            selected_class_ids,
            batch_paths,
            batch_index,
            len(batches),
            progress_callback,
        )
        batch_summaries.extend(summaries)
        raw_responses.extend(responses)

    if progress_callback:
        progress_callback("正在汇总所有批次，生成最终增强方案...")
    final_prompt = build_final_plan_prompt(class_stats, selected_class_ids, batch_summaries)
    final_plan, final_response = request_qwen_json(api_key, base_url, model, final_prompt, [])
    raw_responses.append({"stage": "final", "response": final_response})
    return final_plan, {"batch_summaries": batch_summaries, "raw_responses": raw_responses}


def local_fallback_plan(class_stats, selected_class_ids):
    selected = [class_stats[str(cid)] for cid in selected_class_ids if str(cid) in class_stats]
    nonzero = [stat["instances"] for stat in class_stats.values() if stat["instances"] > 0]
    target_baseline = int(np.median(nonzero)) if nonzero else 0
    classes = []
    for stat in selected:
        current = int(stat["instances"])
        if current <= 0:
            target_new = 0
            method = "keep_original"
            risk = "high"
        else:
            target_new = max(0, min(target_baseline - current, max(10, current)))
            method = "roi_photometric"
            risk = "medium" if current < 5 else "low"
        classes.append(
            {
                "class_id": stat["class_id"],
                "class_name": stat["class_name"],
                "target_new_images": int(target_new),
                "expected_target_box_delta": int(target_new),
                "method_mix": [{"method": method, "images": int(target_new)}] if target_new > 0 else [],
                "method": method,
                "allow_copy_paste": False,
                "allow_append_target_boxes": False,
                "roi_photometric_strength": "low" if risk == "high" else "medium",
                "method_constraints": "本地兜底：只做 ROI 光度扰动，不追加新框。",
                "source_filter_rules": "本地兜底未做 AI 源图筛选；如原标注可疑，需人工复核。",
                "label_risk_policy": "保留原标签，不改变类别和框数量。",
                "risk": risk,
                "rationale": "本地兜底方案：按训练集类别中位数保守补齐，未经过千问视觉分析。",
                "notes": "AI 调用失败或未填写 API Key 时生成的兜底方案。",
            }
        )
    return {"classes": classes, "source": "local_fallback"}


def normalize_ai_plan(plan, class_stats, selected_class_ids):
    selected = {str(v) for v in selected_class_ids}
    stats_by_name = {stat["class_name"]: stat["class_id"] for stat in class_stats.values()}
    normalized = []
    for item in plan.get("classes", []):
        class_id = str(item.get("class_id", "")).strip()
        if class_id not in selected:
            class_name = str(item.get("class_name", "")).strip()
            class_id = stats_by_name.get(class_name, class_id)
        if class_id not in selected or class_id not in class_stats:
            continue
        method = str(item.get("method", "mixed")).strip()
        if method not in {"mixed", "copy_paste_soft_blend", "roi_photometric", "keep_original"}:
            method = "mixed"
        try:
            target_new = int(item.get("target_new_images", 0))
        except (TypeError, ValueError):
            target_new = 0
        target_new = max(0, min(target_new, 10000))
        method_mix = normalize_method_mix(item, target_new)
        target_new = sum(entry["images"] for entry in method_mix)
        method_summary = method_mix_text({"method_mix": method_mix}) or method
        try:
            ai_declared_expected_target_box_delta = float(item.get("expected_target_box_delta", target_new))
        except (TypeError, ValueError):
            ai_declared_expected_target_box_delta = float(target_new)
        expected_target_box_delta = expected_target_delta_for_mix(class_stats[class_id], method_mix)
        normalized.append(
            {
                "class_id": class_id,
                "class_name": class_stats[class_id]["class_name"],
                "current_instances": int(class_stats[class_id]["instances"]),
                "current_images": int(class_stats[class_id]["image_count"]),
                "target_new_images": target_new,
                "expected_target_box_delta": expected_target_box_delta,
                "ai_declared_expected_target_box_delta": ai_declared_expected_target_box_delta,
                "method_mix": method_mix,
                "method": method_summary,
                "allow_copy_paste": bool(item.get("allow_copy_paste", method in {"mixed", "copy_paste_soft_blend"})),
                "allow_append_target_boxes": bool(item.get("allow_append_target_boxes", False) or item.get("allow_copy_paste_append", False)),
                "roi_photometric_strength": normalize_roi_strength(
                    item.get("roi_photometric_strength") or item.get("roi_strength"),
                    "low" if str(item.get("risk", "")).lower() == "high" else "medium",
                ),
                "forbidden_methods": item.get("forbidden_methods", []),
                "subtype_plan": str(item.get("subtype_plan", "")),
                "method_constraints": str(item.get("method_constraints", "")),
                "source_filter_rules": str(item.get("source_filter_rules", "")),
                "label_risk_policy": str(item.get("label_risk_policy", "")),
                "risk": str(item.get("risk", "medium")),
                "rationale": str(item.get("rationale", "")),
                "notes": str(item.get("notes", "")),
            }
        )
    return {"classes": normalized}


def simulate_plan_effect(class_stats, plan):
    max_existing = max([int(stat.get("instances", 0)) for stat in class_stats.values()] or [0])
    per_class = []
    warnings = []
    total_new_images = 0
    total_expected_target_boxes = 0.0
    for item in plan.get("classes", []):
        class_id = str(item["class_id"])
        stat = class_stats.get(class_id, {})
        current_instances = int(stat.get("instances", 0))
        current_images = int(stat.get("image_count", 0))
        avg_target_boxes = (current_instances / current_images) if current_images else 0.0
        planned_images = 0
        expected_target_delta = 0.0
        extra_target_boxes = 0
        method_rows = []
        if current_instances <= 0 and item.get("target_new_images", 0) > 0:
            warnings.append(
                {
                    "class_id": class_id,
                    "level": "block",
                    "message": "该类没有训练源样本，不能执行增强生成。",
                }
            )
        for entry in item.get("method_mix", []):
            method = normalize_method_name(entry.get("method"))
            images = max(0, int(entry.get("images", 0)))
            spec = AUGMENT_METHOD_SPECS.get(method, AUGMENT_METHOD_SPECS["roi_photometric"])
            extra = int(spec.get("extra_target_boxes_per_image", 0)) * images
            target_delta = images * avg_target_boxes + extra
            planned_images += images
            expected_target_delta += target_delta
            extra_target_boxes += extra
            method_rows.append(
                {
                    "method": method,
                    "images": images,
                    "expected_target_box_delta": round(target_delta, 3),
                    "extra_target_boxes": extra,
                    "label_change": spec.get("label_change", ""),
                }
            )
        expected_after = current_instances + expected_target_delta
        if extra_target_boxes > 0:
            warnings.append(
                {
                    "class_id": class_id,
                    "level": "review",
                    "message": f"copy-paste 将额外追加 {extra_target_boxes} 个目标框，需确认不是把图片数误当实例数。",
                }
            )
        if max_existing and expected_after > max_existing * 1.25:
            warnings.append(
                {
                    "class_id": class_id,
                    "level": "review",
                    "message": f"模拟后目标框约 {expected_after:.1f}，超过当前最大类 125%，可能过度补偿。",
                }
            )
        if str(item.get("risk", "")).lower() == "high" and extra_target_boxes > 0:
            warnings.append(
                {
                    "class_id": class_id,
                    "level": "review",
                    "message": "高风险类别使用了会追加目标框的 copy-paste，需要 AI/人工复核。",
                }
            )
        row = {
            "class_id": class_id,
            "class_name": item.get("class_name", ""),
            "current_instances": current_instances,
            "current_images": current_images,
            "planned_new_images": planned_images,
            "avg_target_boxes_per_source_image": round(avg_target_boxes, 4),
            "expected_target_box_delta": round(expected_target_delta, 3),
            "expected_instances_after": round(expected_after, 3),
            "ai_declared_expected_target_box_delta": item.get(
                "ai_declared_expected_target_box_delta",
                item.get("expected_target_box_delta"),
            ),
            "method_rows": method_rows,
        }
        per_class.append(row)
        total_new_images += planned_images
        total_expected_target_boxes += expected_target_delta
    return {
        "per_class": per_class,
        "warnings": warnings,
        "total_new_images": total_new_images,
        "total_expected_target_box_delta": round(total_expected_target_boxes, 3),
    }


def build_stage3_review_prompt(class_stats, plan, simulation):
    return (
        "你是工业缺陷数据增强策略审查员。现在进行第 3 阶段：执行前模拟复核。\n"
        "请重点检查：新增图片数和新增目标框数是否混淆、copy-paste 是否导致实例膨胀、少数类是否过度补偿、"
        "高混淆类别是否会放大错标、未升级子分类/子形态是否被覆盖。\n"
        "只输出严格 JSON，不要 Markdown。格式如下：\n"
        "{\n"
        "  \"approved\": true,\n"
        "  \"blocking_issues\": [\"必须先修正的问题\"],\n"
        "  \"review_warnings\": [\"可以执行但需要注意的问题\"],\n"
        "  \"recommended_adjustments\": \"如需修改计划，请简述；不需要则写空字符串\"\n"
        "}\n\n"
        f"增强算子合同：{method_contract_text()}\n"
        f"当前计划：{json.dumps(plan, ensure_ascii=False)}\n"
        f"本地模拟结果：{json.dumps(simulation, ensure_ascii=False)}\n"
        f"类别统计：{json.dumps(class_stats, ensure_ascii=False)}"
    )


def request_qwen_simulation_review(api_key, base_url, model, class_stats, plan, simulation):
    prompt = build_stage3_review_prompt(class_stats, plan, simulation)
    review, response = request_qwen_json(api_key, base_url, model, prompt, [])
    return review, {"stage": "simulation_review", "response": response}


def write_simulation_report(output_root, simulation, ai_review=None):
    output_root = Path(output_root)
    payload = {"simulation": simulation, "ai_review": ai_review or {}}
    (output_root / "ai_aug_simulation_review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# AI 数据增强执行前模拟复核",
        "",
        f"- planned_new_images: {simulation.get('total_new_images', 0)}",
        f"- expected_target_box_delta: {simulation.get('total_expected_target_box_delta', 0)}",
        "",
        "## Per Class",
        "",
        "| class_id | class_name | current_boxes | planned_images | expected_box_delta | expected_after | methods |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in simulation.get("per_class", []):
        methods = ", ".join(f"{m['method']}:{m['images']}" for m in row.get("method_rows", []))
        lines.append(
            f"| {row['class_id']} | {row.get('class_name', '')} | {row.get('current_instances', 0)} | "
            f"{row.get('planned_new_images', 0)} | {row.get('expected_target_box_delta', 0)} | "
            f"{row.get('expected_instances_after', 0)} | {methods} |"
        )
    lines.extend(["", "## Warnings", ""])
    for warning in simulation.get("warnings", []):
        lines.append(f"- [{warning.get('level', '')}] class {warning.get('class_id', '')}: {warning.get('message', '')}")
    if ai_review:
        lines.extend(["", "## AI Advisory Review", "", f"```json\n{json.dumps(ai_review, ensure_ascii=False, indent=2)}\n```"])
    (output_root / "ai_aug_simulation_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_text(value, limit=180):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        text = "；".join(str(v) for v in value if str(v).strip())
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit - 3] + "..."
    return text


def append_unique_text(bucket, key, value):
    text = compact_text(value)
    if text and text not in bucket[key]:
        bucket[key].append(text)


def build_diagnosis_summary_rows(diagnosis, class_stats=None, selected_class_ids=None):
    class_stats = class_stats or {}
    selected = {str(v) for v in (selected_class_ids or [])}
    rows_by_class = {}
    parse_failures = []

    for batch in (diagnosis or {}).get("batch_summaries", []):
        if not isinstance(batch, dict):
            continue
        batch_index = str(batch.get("batch_index", ""))
        if batch.get("parse_failed"):
            parse_failures.append(f"{batch_index}: {compact_text(batch.get('error'), 120)}")
        summaries = batch.get("class_summaries", [])
        if isinstance(summaries, dict):
            summaries = list(summaries.values())
        if not isinstance(summaries, list):
            continue
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            class_id = str(summary.get("class_id", "")).strip()
            class_name = str(summary.get("class_name", "")).strip()
            if not class_id and class_name:
                for stat in class_stats.values():
                    if stat.get("class_name") == class_name:
                        class_id = str(stat.get("class_id"))
                        break
            if not class_id:
                class_id = "unknown"
            stat = class_stats.get(class_id, {})
            row = rows_by_class.setdefault(
                class_id,
                {
                    "class_id": class_id,
                    "class_name": class_name or stat.get("class_name", ""),
                    "batch_count": 0,
                    "observed_count_estimate": 0,
                    "visual_patterns": [],
                    "subtype_hints": [],
                    "confusable_with": [],
                    "suspect_label_notes": [],
                    "safe_augmentation_directions": [],
                    "unsafe_augmentation_directions": [],
                },
            )
            row["batch_count"] += 1
            try:
                row["observed_count_estimate"] += int(float(summary.get("observed_count_estimate", 0)))
            except (TypeError, ValueError):
                pass
            append_unique_text(row, "visual_patterns", summary.get("visual_patterns"))
            append_unique_text(row, "subtype_hints", summary.get("subtype_hints"))
            append_unique_text(row, "confusable_with", summary.get("confusable_with"))
            append_unique_text(row, "suspect_label_notes", summary.get("suspect_label_notes"))
            append_unique_text(row, "safe_augmentation_directions", summary.get("safe_augmentation_directions"))
            append_unique_text(row, "unsafe_augmentation_directions", summary.get("unsafe_augmentation_directions") or summary.get("risks"))

    if selected:
        for class_id in sorted(selected, key=lambda value: int(value) if value.isdigit() else 999999):
            if class_id in rows_by_class:
                continue
            stat = class_stats.get(class_id, {})
            rows_by_class[class_id] = {
                "class_id": class_id,
                "class_name": stat.get("class_name", ""),
                "batch_count": 0,
                "observed_count_estimate": 0,
                "visual_patterns": ["未得到可解析的 AI 诊断摘要"],
                "subtype_hints": [],
                "confusable_with": [],
                "suspect_label_notes": [],
                "safe_augmentation_directions": [],
                "unsafe_augmentation_directions": parse_failures[:3],
            }

    rows = list(rows_by_class.values())
    rows.sort(key=lambda row: int(row["class_id"]) if str(row["class_id"]).isdigit() else 999999)
    for row in rows:
        for key in [
            "visual_patterns",
            "subtype_hints",
            "confusable_with",
            "suspect_label_notes",
            "safe_augmentation_directions",
            "unsafe_augmentation_directions",
        ]:
            row[key] = compact_text(row.get(key, []), 220)
    return rows


def write_diagnosis_report(output_root, diagnosis, class_stats=None, selected_class_ids=None):
    output_root = Path(output_root)
    rows = build_diagnosis_summary_rows(diagnosis, class_stats, selected_class_ids)
    (output_root / "ai_aug_stage1_diagnosis_summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# AI 数据增强阶段1诊断摘要",
        "",
        "| class_id | class_name | batches | observed | visual_patterns | subtype_hints | confusable_with | suspect_label_notes | unsafe_directions |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('class_id', '')} | {row.get('class_name', '')} | {row.get('batch_count', 0)} | "
            f"{row.get('observed_count_estimate', 0)} | {row.get('visual_patterns', '')} | "
            f"{row.get('subtype_hints', '')} | {row.get('confusable_with', '')} | "
            f"{row.get('suspect_label_notes', '')} | {row.get('unsafe_augmentation_directions', '')} |"
        )
    (output_root / "ai_aug_stage1_diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def copy_dataset_to_output(dataset_dir, output_root, class_names):
    dataset_dir = Path(dataset_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for item in dataset_dir.iterdir():
        target = output_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    write_data_yaml_from_names(output_root, class_names)
    for folder in ["review/pending/images", "review/pending/labels", "review/accepted/images", "review/accepted/labels", "review/rejected", "review/flagged", "review/uncovered"]:
        (output_root / folder).mkdir(parents=True, exist_ok=True)


def choose_paste_offset(points_px, width, height, patch_w, patch_h):
    min_x = int(np.min(points_px[:, 0]))
    min_y = int(np.min(points_px[:, 1]))
    for _ in range(30):
        max_shift_x = max(12, int(width * 0.18))
        max_shift_y = max(12, int(height * 0.18))
        dx = random.randint(-max_shift_x, max_shift_x)
        dy = random.randint(-max_shift_y, max_shift_y)
        if abs(dx) + abs(dy) < 10:
            continue
        new_min_x = min_x + dx
        new_min_y = min_y + dy
        if new_min_x < 0 or new_min_y < 0:
            continue
        if new_min_x + patch_w >= width or new_min_y + patch_h >= height:
            continue
        return dx, dy
    return None


def apply_roi_photometric(image, mask, strength="medium"):
    output = image.copy().astype(np.float32)
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return image.copy()
    if strength == "low":
        alpha = random.uniform(0.92, 1.08)
        beta = random.uniform(-8, 8)
        noise_sigma = random.uniform(1, 3)
    elif strength == "high":
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-16, 16)
        noise_sigma = random.uniform(2, 7)
    else:
        alpha = random.uniform(0.88, 1.12)
        beta = random.uniform(-12, 12)
        noise_sigma = random.uniform(1, 5)
    noise = np.random.normal(0, noise_sigma, output.shape).astype(np.float32)
    changed = output * alpha + beta + noise
    output[mask_bool] = changed[mask_bool]
    return np.clip(output, 0, 255).astype(np.uint8)


def augment_roi_photometric(record, out_image_path, out_label_path, strength="medium"):
    image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return False, "image_read_failed"
    height, width = image.shape[:2]
    points_px = points_to_pixels(record["parsed"]["points"], width, height)
    mask = mask_from_points(points_px, width, height)
    if cv2.countNonZero(mask) <= 0:
        return False, "empty_mask"
    augmented = apply_roi_photometric(image, mask, normalize_roi_strength(strength))
    save_image(out_image_path, augmented)
    shutil.copy2(record["label_path"], out_label_path)
    return True, "roi_photometric"


def augment_copy_paste(record, out_image_path, out_label_path):
    image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return False, "image_read_failed"
    height, width = image.shape[:2]
    points_px = points_to_pixels(record["parsed"]["points"], width, height)
    bbox = polygon_bbox_pixels(points_px, width, height, pad=6)
    if bbox is None:
        return False, "bad_bbox"
    mask = mask_from_points(points_px, width, height)
    x1, y1, x2, y2 = bbox
    patch = image[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2].copy()
    if patch.size == 0 or cv2.countNonZero(patch_mask) <= 0:
        return False, "empty_patch"
    offset = choose_paste_offset(points_px, width, height, x2 - x1, y2 - y1)
    if offset is None:
        return False, "no_valid_offset"
    dx, dy = offset
    target_x1 = x1 + dx
    target_y1 = y1 + dy
    target_x2 = target_x1 + (x2 - x1)
    target_y2 = target_y1 + (y2 - y1)
    if target_x1 < 0 or target_y1 < 0 or target_x2 > width or target_y2 > height:
        return False, "target_out_of_bounds"

    alpha = cv2.GaussianBlur(patch_mask, (0, 0), sigmaX=3).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    target = image[target_y1:target_y2, target_x1:target_x2].astype(np.float32)
    blended = target * (1.0 - alpha[:, :, None]) + patch.astype(np.float32) * alpha[:, :, None]
    output = image.copy()
    output[target_y1:target_y2, target_x1:target_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    new_points_px = points_px + np.array([[dx, dy]], dtype=np.int32)
    new_points = pixel_points_to_norm(new_points_px.tolist(), width, height)
    new_line = format_label_line(record["class_id"], new_points, record["parsed"]["kind"])
    lines = read_label_file(record["label_path"])
    lines.append(new_line)
    save_image(out_image_path, output)
    Path(out_label_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_label_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, "copy_paste_soft_blend"


def build_candidate_name(class_id, class_name, serial):
    return f"ai_aug_cls{int(class_id):03d}_{serial:06d}"


def generate_augmented_candidates(dataset_dir, output_root, class_names, plan, progress_callback=None):
    dataset_dir = Path(dataset_dir)
    output_root = Path(output_root)
    records, _ = collect_dataset_records(dataset_dir, class_names)
    by_class = defaultdict(list)
    for record in records:
        by_class[record["class_id"]].append(record)

    random.seed(42)
    np.random.seed(42)
    pending_images = output_root / "review" / "pending" / "images"
    pending_labels = output_root / "review" / "pending" / "labels"
    pending_images.mkdir(parents=True, exist_ok=True)
    pending_labels.mkdir(parents=True, exist_ok=True)

    total_target = sum(
        int(entry.get("images", 0))
        for item in plan.get("classes", [])
        for entry in (item.get("method_mix") or [])
    )
    generated_rows = []
    generated_total = 0
    serial = 1
    skipped = Counter()

    for item in plan.get("classes", []):
        class_id = str(item["class_id"])
        method_mix = item.get("method_mix") or normalize_method_mix(item, int(item.get("target_new_images", 0)))
        class_target = sum(int(entry.get("images", 0)) for entry in method_mix)
        if class_target <= 0:
            continue
        candidates = by_class.get(class_id, [])
        if not candidates:
            skipped[(class_id, "no_source")] += class_target
            continue
        roi_strength = normalize_roi_strength(item.get("roi_photometric_strength", "medium"))
        for mix_entry in method_mix:
            method = normalize_method_name(mix_entry.get("method"))
            target = int(mix_entry.get("images", 0))
            if method not in {"roi_photometric", "copy_paste_soft_blend"} or target <= 0:
                continue
            max_attempts = max(20, target * 20)
            attempts = 0
            class_generated = 0
            while class_generated < target and attempts < max_attempts:
                attempts += 1
                record = random.choice(candidates)
                stem = build_candidate_name(class_id, item["class_name"], serial)
                out_image = pending_images / f"{stem}.png"
                out_label = pending_labels / f"{stem}.txt"
                if method == "roi_photometric":
                    ok, used_method = augment_roi_photometric(record, out_image, out_label, roi_strength)
                else:
                    ok, used_method = augment_copy_paste(record, out_image, out_label)
                if not ok:
                    skipped[(class_id, used_method)] += 1
                    continue
                generated_rows.append(
                    {
                        "accepted": "0",
                        "class_id": class_id,
                        "class_name": item["class_name"],
                        "planned_method": method,
                        "method": used_method,
                        "roi_photometric_strength": roi_strength if method == "roi_photometric" else "",
                        "source_image": str(record["image_path"]),
                        "source_label": str(record["label_path"]),
                        "candidate_image": str(out_image),
                        "candidate_label": str(out_label),
                        "target_dataset_image": f"images/train/{out_image.name}",
                        "target_dataset_label": f"labels/train/{out_label.name}",
                    }
                )
                class_generated += 1
                generated_total += 1
                serial += 1
                if progress_callback and total_target:
                    progress_callback(int(generated_total / total_target * 100))

    manifest_path = output_root / "review" / "pending" / "manifest.csv"
    write_candidate_manifest(manifest_path, generated_rows)
    with open(output_root / "ai_aug_generation_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_total": generated_total,
                "planned_total": total_target,
                "skipped": {f"{k[0]}:{k[1]}": v for k, v in skipped.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if progress_callback:
        progress_callback(100)
    return generated_rows


def write_candidate_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "accepted",
        "class_id",
        "class_name",
        "planned_method",
        "method",
        "roi_photometric_strength",
        "source_image",
        "source_label",
        "candidate_image",
        "candidate_label",
        "target_dataset_image",
        "target_dataset_label",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def accept_candidates(output_root, rows):
    output_root = Path(output_root)
    accepted = []
    for row in rows:
        image_src = Path(row["candidate_image"])
        label_src = Path(row["candidate_label"])
        if not image_src.exists() or not label_src.exists():
            continue
        image_dst = output_root / "images" / "train" / image_src.name
        label_dst = output_root / "labels" / "train" / label_src.name
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)
        accepted_row = dict(row)
        accepted_row["accepted"] = "1"
        accepted.append(accepted_row)
        shutil.copy2(image_src, output_root / "review" / "accepted" / "images" / image_src.name)
        shutil.copy2(label_src, output_root / "review" / "accepted" / "labels" / label_src.name)
    write_candidate_manifest(output_root / "review" / "accepted" / "manifest.csv", accepted)
    pending_manifest = output_root / "review" / "pending" / "manifest.csv"
    if pending_manifest.exists():
        all_rows = []
        with open(pending_manifest, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if any(row.get("candidate_image") == accepted_row["candidate_image"] for accepted_row in accepted):
                    row["accepted"] = "1"
                all_rows.append(row)
        write_candidate_manifest(pending_manifest, all_rows)
    return accepted


def write_plan_report(output_root, class_stats, plan):
    output_root = Path(output_root)
    plan_path = output_root / "ai_aug_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    lines = [
        "# AI 数据增强方案",
        "",
        "## 规则",
        "- 只处理用户选中的缺陷类别。",
        "- 只从原 dataset 的 train 中取图。",
        "- 增强候选先进入 review/pending，随后自动写入新数据集 images/train 和 labels/train。",
        "- review/accepted 会保留已写入 train 的增强样本记录。",
        "- val 保持原样，不增强。",
        "",
        "## 类别统计与计划",
        "",
        "| class_id | 缺陷 | 当前实例 | 当前图片 | 建议新增 | 方法 | 风险 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in plan.get("classes", []):
        stat = class_stats.get(str(item["class_id"]), {})
        lines.append(
            f"| {item['class_id']} | {item['class_name']} | {stat.get('instances', 0)} | "
            f"{stat.get('image_count', 0)} | {item.get('target_new_images', 0)} | "
            f"{item.get('method', '')} | {item.get('risk', '')} |"
        )
    lines.append("")
    lines.append("## AI 理由")
    for item in plan.get("classes", []):
        lines.append("")
        lines.append(f"### {item['class_id']} {item['class_name']}")
        lines.append(f"- 方法: {item.get('method', '')}")
        lines.append(f"- 建议新增: {item.get('target_new_images', 0)}")
        lines.append(f"- 风险: {item.get('risk', '')}")
        lines.append(f"- 理由: {item.get('rationale', '')}")
        if item.get("notes"):
            lines.append(f"- 备注: {item.get('notes')}")
    (output_root / "ai_aug_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan_report_v2(output_root, class_stats, plan):
    output_root = Path(output_root)
    plan_path = output_root / "ai_aug_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    lines = [
        "# AI 数据增强计划",
        "",
        "## 流程",
        "- 1. AI 分析原始训练图并生成可执行增强策略。",
        "- 2. 本地执行增强、硬检查、可选 AI 疑点筛查，然后写入新数据集 train。",
        "- val 保持原样，不增强。",
        "",
        "## 类别统计和计划",
        "",
        "| class_id | 缺陷 | 当前框 | 当前图片 | 计划新增图 | 预计新增目标框 | 方法明细 | ROI强度 | 风险 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in plan.get("classes", []):
        stat = class_stats.get(str(item["class_id"]), {})
        lines.append(
            f"| {item['class_id']} | {item['class_name']} | {stat.get('instances', 0)} | "
            f"{stat.get('image_count', 0)} | {item.get('target_new_images', 0)} | "
            f"{item.get('expected_target_box_delta', '')} | {method_mix_text(item)} | "
            f"{item.get('roi_photometric_strength', '')} | {item.get('risk', '')} |"
        )
    lines.append("")
    lines.append("## AI 说明")
    for item in plan.get("classes", []):
        lines.append("")
        lines.append(f"### {item['class_id']} {item['class_name']}")
        lines.append(f"- 方法明细: {method_mix_text(item)}")
        lines.append(f"- 新增图片: {item.get('target_new_images', 0)}")
        lines.append(f"- AI 预计新增目标框: {item.get('expected_target_box_delta', '')}")
        lines.append(f"- ROI 光度扰动强度: {item.get('roi_photometric_strength', '')}")
        lines.append(f"- 风险: {item.get('risk', '')}")
        if item.get("subtype_plan"):
            lines.append(f"- 子形态/子分类计划: {item.get('subtype_plan')}")
        if item.get("method_constraints"):
            lines.append(f"- 方法约束: {item.get('method_constraints')}")
        if item.get("source_filter_rules"):
            lines.append(f"- 源图/标注筛选规则: {item.get('source_filter_rules')}")
        if item.get("label_risk_policy"):
            lines.append(f"- 标注风险处理: {item.get('label_risk_policy')}")
        lines.append(f"- 理由: {item.get('rationale', '')}")
        if item.get("notes"):
            lines.append(f"- 备注: {item.get('notes')}")
    (output_root / "ai_aug_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def local_quality_review(output_root, rows):
    output_root = Path(output_root)
    warnings = []
    rejected = []
    by_class = Counter()
    by_method = Counter()
    for row in rows:
        image_path = Path(row["candidate_image"])
        label_path = Path(row["candidate_label"])
        by_class[row.get("class_id", "")] += 1
        by_method[row.get("method", "")] += 1
        if not image_path.exists() or not label_path.exists():
            rejected.append(str(image_path))
            warnings.append(f"missing candidate file: {image_path}")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append(str(image_path))
            warnings.append(f"unreadable candidate image: {image_path}")
            continue
        candidate_lines = read_label_file(label_path)
        source_lines = read_label_file(row.get("source_label", ""))
        method = normalize_method_name(row.get("method"))
        expected_label_count = len(source_lines) + (1 if method == "copy_paste_soft_blend" else 0)
        if source_lines and len(candidate_lines) != expected_label_count:
            rejected.append(str(image_path))
            warnings.append(
                f"label count violates method contract: {image_path} expected {expected_label_count}, got {len(candidate_lines)}"
            )
            continue
        for line in candidate_lines:
            parsed = parse_label_line(line)
            if not parsed:
                rejected.append(str(image_path))
                warnings.append(f"bad label line: {label_path}")
                break
            xs = [p[0] for p in parsed["points"]]
            ys = [p[1] for p in parsed["points"]]
            if min(xs) < 0 or max(xs) > 1 or min(ys) < 0 or max(ys) > 1:
                rejected.append(str(image_path))
                warnings.append(f"out of bounds label: {label_path}")
                break
    return {
        "approved": len(rejected) == 0,
        "source": "local_quality_gate",
        "warnings": warnings,
        "rejected_candidate_images": sorted(set(rejected)),
        "generated_by_class": dict(by_class),
        "generated_by_method": dict(by_method),
        "total_candidates": len(rows),
    }


def build_stage4_quality_prompt(plan, simulation, local_review, rows, batch_index=1, batch_count=1, start_index=0):
    sample_rows = []
    for offset, row in enumerate(rows, 1):
        sample_index = start_index + offset
        sample_rows.append(
            {
                "sample_index": sample_index,
                "class_id": row.get("class_id"),
                "class_name": row.get("class_name"),
                "method": row.get("method"),
                "candidate_image": Path(row.get("candidate_image", "")).name,
                "source_image": Path(row.get("source_image", "")).name,
            }
        )
    return (
        "你是工业缺陷增强数据的可疑样本筛查员，不是最终裁判。"
        f"现在检查第 {batch_index}/{batch_count} 批增强候选图。\n"
        "随附图片按 sample_rows 顺序排列。必须为 sample_rows 中每一张候选图输出一条 per_sample_findings，"
        "candidate_image 必须使用 sample_rows 里的文件名，不能漏图、不能改名。"
        "请只按固定 reason_code 标记样本："
        "fake_texture、label_mismatch、wrong_class_risk、copy_paste_artifact、box_count_violation、strategy_violation、source_label_suspect。"
        "decision=pass 时 reason_code 和 evidence 可以为空；decision=warning/reject 时必须指出具体哪里不对。"
        "不能因为整体策略不完美就否定整批；必须逐图判断。\n"
        "只输出严格 JSON，不要 Markdown。格式如下：\n"
        "{\n"
        "  \"approved\": true,\n"
        "  \"review_warnings\": [\"风险说明\"],\n"
        "  \"rejected_sample_indices\": [],\n"
        "  \"rejected_candidate_images\": [],\n"
        "  \"per_sample_findings\": [\n"
        "    {\"sample_index\": 1, \"candidate_image\": \"文件名.png\", \"decision\": \"pass|warning|reject\", \"reason_code\": \"\", \"evidence\": \"\"}\n"
        "  ],\n"
        "  \"rationale\": \"整体质检判断\"\n"
        "}\n\n"
        f"计划：{json.dumps(plan, ensure_ascii=False)}\n"
        f"执行前模拟：{json.dumps(simulation, ensure_ascii=False)}\n"
        f"本地硬检查：{json.dumps(local_review, ensure_ascii=False)}\n"
        f"sample_rows：{json.dumps(sample_rows, ensure_ascii=False)}"
    )


def request_qwen_quality_review(api_key, base_url, model, plan, simulation, local_review, rows):
    batch_size = 24
    rows = list(rows)
    all_names = [Path(row["candidate_image"]).name for row in rows]
    valid_names = set(all_names)
    row_by_name = {Path(row["candidate_image"]).name: row for row in rows}
    merged = {
        "approved": True,
        "review_warnings": [],
        "rejected_sample_indices": [],
        "rejected_candidate_images": [],
        "per_sample_findings": [],
        "by_candidate": {},
        "covered_candidate_images": [],
        "uncovered_candidate_images": [],
        "duplicate_candidate_images": [],
        "unknown_candidate_images": [],
        "coverage_complete": False,
        "candidate_count": len(rows),
        "rationale": "",
    }
    raw_responses = []
    duplicate_names = []
    unknown_names = []

    def merge_finding(name, finding):
        name = Path(str(name or "")).name
        if not name:
            return
        if name not in valid_names:
            unknown_names.append(name)
            return
        decision = str(finding.get("decision", "pass")).strip().lower()
        if decision not in {"pass", "warning", "reject"}:
            decision = "warning"
        normalized = {
            "sample_index": finding.get("sample_index"),
            "candidate_image": name,
            "decision": decision,
            "reason_code": str(finding.get("reason_code", "")),
            "evidence": str(finding.get("evidence", "")),
        }
        existing = merged["by_candidate"].get(name)
        if existing:
            duplicate_names.append(name)
            severity = {"pass": 0, "warning": 1, "reject": 2}
            if severity[decision] < severity.get(existing.get("decision", "pass"), 0):
                return
        merged["by_candidate"][name] = normalized

    batches = list(chunked(rows, batch_size))
    for batch_index, batch_rows in batches:
        start_index = (batch_index - 1) * batch_size
        image_paths = [row["candidate_image"] for row in batch_rows if Path(row["candidate_image"]).exists()]
        prompt = build_stage4_quality_prompt(
            plan,
            simulation,
            local_review,
            batch_rows,
            batch_index=batch_index,
            batch_count=len(batches),
            start_index=start_index,
        )
        review, response = request_qwen_json(api_key, base_url, model, prompt, image_paths)
        raw_responses.append({"batch_index": batch_index, "response": response, "review": review})
        merged["approved"] = bool(merged["approved"] and review.get("approved", True))
        merged["review_warnings"].extend(review.get("review_warnings", []) or [])
        merged["rejected_candidate_images"].extend(review.get("rejected_candidate_images", []) or [])
        for finding in review.get("per_sample_findings", []) or []:
            if not isinstance(finding, dict):
                continue
            candidate_name = finding.get("candidate_image")
            if not candidate_name:
                try:
                    candidate_name = Path(rows[int(finding.get("sample_index")) - 1]["candidate_image"]).name
                except (TypeError, ValueError, IndexError):
                    candidate_name = ""
            merge_finding(candidate_name, finding)
        for index in review.get("rejected_sample_indices", []) or []:
            try:
                numeric_index = int(index)
                candidate_name = Path(rows[numeric_index - 1]["candidate_image"]).name
            except (TypeError, ValueError):
                continue
            except IndexError:
                continue
            merged["rejected_sample_indices"].append(numeric_index)
            merge_finding(
                candidate_name,
                {
                    "sample_index": numeric_index,
                    "candidate_image": candidate_name,
                    "decision": "reject",
                    "reason_code": "ai_rejected_sample_index",
                    "evidence": "AI returned this sample index in rejected_sample_indices.",
                },
            )
        for candidate_name in review.get("rejected_candidate_images", []) or []:
            name = Path(str(candidate_name)).name
            merge_finding(
                name,
                {
                    "sample_index": None,
                    "candidate_image": name,
                    "decision": "reject",
                    "reason_code": "ai_rejected_candidate_image",
                    "evidence": "AI returned this file in rejected_candidate_images.",
                },
            )
        if review.get("rationale"):
            merged["rationale"] += ("\n" if merged["rationale"] else "") + str(review.get("rationale"))
    merged["review_warnings"] = list(dict.fromkeys(str(v) for v in merged["review_warnings"]))
    merged["rejected_candidate_images"] = list(dict.fromkeys(str(v) for v in merged["rejected_candidate_images"]))
    merged["rejected_sample_indices"] = sorted(set(merged["rejected_sample_indices"]))
    covered = sorted(name for name in merged["by_candidate"] if name in valid_names)
    uncovered = sorted(name for name in all_names if name not in merged["by_candidate"])
    merged["per_sample_findings"] = [merged["by_candidate"][name] for name in covered]
    merged["covered_candidate_images"] = covered
    merged["uncovered_candidate_images"] = uncovered
    merged["duplicate_candidate_images"] = sorted(set(duplicate_names))
    merged["unknown_candidate_images"] = sorted(set(unknown_names))
    merged["coverage_complete"] = len(uncovered) == 0
    if uncovered:
        merged["approved"] = False
        merged["review_warnings"].append(
            f"AI质检覆盖不完整，未覆盖 {len(uncovered)} 张候选图；未覆盖图不得自动写入 train。"
        )
    if duplicate_names:
        merged["review_warnings"].append(f"AI质检返回了重复文件名：{', '.join(sorted(set(duplicate_names))[:20])}")
    if unknown_names:
        merged["review_warnings"].append(f"AI质检返回了未知文件名：{', '.join(sorted(set(unknown_names))[:20])}")
    return merged, {"stage": "quality_review", "raw_responses": raw_responses}


def split_rows_by_quality(rows, quality_review):
    rejected_names = set()
    for value in quality_review.get("rejected_candidate_images", []) or []:
        rejected_names.add(Path(str(value)).name)
    for finding in quality_review.get("per_sample_findings", []) or []:
        if not isinstance(finding, dict):
            continue
        decision = str(finding.get("decision", "")).strip().lower()
        if decision not in {"warning", "reject"}:
            continue
        candidate_image = finding.get("candidate_image")
        if candidate_image:
            rejected_names.add(Path(str(candidate_image)).name)
            continue
        index = finding.get("sample_index")
        try:
            row = rows[int(index) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        rejected_names.add(Path(row["candidate_image"]).name)
    for index in quality_review.get("rejected_sample_indices", []) or []:
        try:
            row = rows[int(index) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        rejected_names.add(Path(row["candidate_image"]).name)
    accepted = []
    rejected = []
    for row in rows:
        if Path(row["candidate_image"]).name in rejected_names or str(row["candidate_image"]) in rejected_names:
            rejected.append(row)
        else:
            accepted.append(row)
    return accepted, rejected


def split_rows_by_ai_coverage(rows, ai_review):
    by_candidate = ai_review.get("by_candidate", {}) if isinstance(ai_review, dict) else {}
    accepted = []
    blocked = []
    flagged = []
    uncovered = []
    for row in rows:
        name = Path(row["candidate_image"]).name
        finding = by_candidate.get(name)
        if not finding:
            uncovered.append(row)
            blocked.append(row)
            continue
        decision = str(finding.get("decision", "")).strip().lower()
        if decision == "pass":
            accepted.append(row)
        else:
            flagged.append(row)
            blocked.append(row)
    return accepted, blocked, flagged, uncovered


def write_quality_report(
    output_root,
    local_review,
    ai_review,
    accepted_rows,
    rejected_rows,
    ai_flagged_rows=None,
    ai_uncovered_rows=None,
    ai_quality_mode="required_full_coverage",
):
    output_root = Path(output_root)
    ai_flagged_rows = ai_flagged_rows or []
    ai_uncovered_rows = ai_uncovered_rows or []
    payload = {
        "local_review": local_review,
        "ai_review": ai_review or {},
        "ai_quality_mode": ai_quality_mode,
        "accepted_count": len(accepted_rows),
        "not_written_count": len(rejected_rows),
        "local_hard_rejected_count": len(local_review.get("rejected_candidate_images", []) or []),
        "ai_flagged_count": len(ai_flagged_rows),
        "ai_uncovered_count": len(ai_uncovered_rows),
        "not_written_rows": rejected_rows,
        "ai_flagged_rows": ai_flagged_rows,
        "ai_uncovered_rows": ai_uncovered_rows,
    }
    (output_root / "ai_aug_quality_review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if rejected_rows:
        write_candidate_manifest(output_root / "review" / "rejected" / "manifest.csv", rejected_rows)
    if ai_flagged_rows:
        write_candidate_manifest(output_root / "review" / "flagged" / "manifest.csv", ai_flagged_rows)
    if ai_uncovered_rows:
        write_candidate_manifest(output_root / "review" / "uncovered" / "manifest.csv", ai_uncovered_rows)
    lines = [
        "# AI 数据增强执行后质检",
        "",
        f"- candidates: {local_review.get('total_candidates', 0)}",
        f"- accepted: {len(accepted_rows)}",
        f"- not_written: {len(rejected_rows)}",
        f"- local_hard_rejected: {len(local_review.get('rejected_candidate_images', []) or [])}",
        f"- ai_flagged: {len(ai_flagged_rows)}",
        f"- ai_uncovered: {len(ai_uncovered_rows)}",
        f"- ai_quality_mode: {ai_quality_mode}",
        "",
        "## Local Gate",
        "",
        f"```json\n{json.dumps(local_review, ensure_ascii=False, indent=2)}\n```",
    ]
    if ai_review:
        lines.extend(["", "## AI Review", "", f"```json\n{json.dumps(ai_review, ensure_ascii=False, indent=2)}\n```"])
    (output_root / "ai_aug_quality_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class AIAugmentDialog(QDialog):
    def __init__(self, project_dir, parent=None):
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self.dataset_dir = self.project_dir / "dataset"
        self.output_root = None
        self.class_names = load_data_yaml_names(self.dataset_dir / "data.yaml")
        self.class_stats = {}
        self.plan = None
        self.selected = []
        self.image_paths_for_ai = []
        self.diagnosis = None
        self.diagnosis_rows = []
        self.simulation = None
        self.simulation_ai_review = None
        self.quality_review = None
        self.generated_rows = []
        self.table_mode = "plan"
        self.setWindowTitle("AI 数据增强")
        self.setMinimumSize(980, 720)
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        tip = QLabel("流程：选择缺陷 -> AI 分析全量训练图并生成精细策略 -> 人工确认数量/方法 -> 本地执行增强和硬检查 -> 写入新数据集 train；val 不增强。")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        api_box = QGroupBox("千问 API")
        api_layout = QGridLayout(api_box)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.base_url_edit = QLineEdit(DEFAULT_QWEN_URL)
        self.model_edit = QLineEdit(DEFAULT_QWEN_MODEL)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 50)
        self.batch_size_spin.setValue(12)
        api_layout.addWidget(QLabel("API Key"), 0, 0)
        api_layout.addWidget(self.api_key_edit, 0, 1)
        api_layout.addWidget(QLabel("Base URL"), 1, 0)
        api_layout.addWidget(self.base_url_edit, 1, 1)
        api_layout.addWidget(QLabel("Model"), 2, 0)
        api_layout.addWidget(self.model_edit, 2, 1)
        api_layout.addWidget(QLabel("每批分析图片数"), 3, 0)
        api_layout.addWidget(self.batch_size_spin, 3, 1)
        layout.addWidget(api_box)

        class_box = QGroupBox("选择要 AI 协助增强的缺陷")
        class_layout = QVBoxLayout(class_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder_lay = QVBoxLayout(holder)
        self.class_checks = {}
        if not self.class_names:
            holder_lay.addWidget(QLabel("未找到 dataset/data.yaml，请先生成数据集。"))
        for idx, name in enumerate(self.class_names):
            cb = QCheckBox(f"{idx}: {name}")
            holder_lay.addWidget(cb)
            self.class_checks[str(idx)] = cb
        scroll.setWidget(holder)
        class_layout.addWidget(scroll)
        layout.addWidget(class_box, 1)

        action_row = QHBoxLayout()
        self.analyze_btn = QPushButton("1 AI分析+生成策略")
        self.generate_btn = QPushButton("2 执行增强")
        self.ai_quality_check = QCheckBox("AI质检全量覆盖后写入train")
        self.ai_quality_check.setChecked(True)
        self.ai_quality_check.setEnabled(False)
        self.plan_btn = QPushButton("2 AI生成增强计划")
        self.simulate_btn = QPushButton("3 本地模拟+AI复核")
        self.plan_btn.setEnabled(False)
        self.simulate_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.plan_btn.setVisible(False)
        self.simulate_btn.setVisible(False)
        action_row.addWidget(self.analyze_btn)
        action_row.addWidget(self.generate_btn)
        action_row.addWidget(self.ai_quality_check)
        layout.addLayout(action_row)

        self.plan_table = QTableWidget(0, 8, self)
        self.plan_table.setHorizontalHeaderLabels(["启用", "缺陷", "当前框", "当前图片", "新增图片", "预计新增框", "方法明细", "风险/理由"])
        layout.addWidget(self.plan_table, 2)

        self.log = QTextBrowser()
        layout.addWidget(self.log, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.analyze_btn.clicked.connect(self.analyze_and_plan)
        self.plan_btn.clicked.connect(self.generate_plan)
        self.simulate_btn.clicked.connect(self.simulate_and_review)
        self.generate_btn.clicked.connect(self.execute_augmentation)

    def selected_class_ids(self):
        return [cid for cid, cb in self.class_checks.items() if cb.isChecked()]

    def log_line(self, text):
        self.log.append(str(text))

    def api_settings(self):
        return (
            self.api_key_edit.text().strip(),
            self.base_url_edit.text().strip() or DEFAULT_QWEN_URL,
            self.model_edit.text().strip() or DEFAULT_QWEN_MODEL,
        )

    def analyze_and_plan(self):
        self.stage1_diagnose()
        if self.output_root and self.diagnosis:
            self.generate_plan()

    def stage1_diagnose(self):
        if not self.dataset_dir.exists():
            QMessageBox.warning(self, "提示", "未找到 dataset，请先生成数据集。")
            return
        selected = self.selected_class_ids()
        if not selected:
            QMessageBox.information(self, "提示", "请至少选择一个缺陷类别。")
            return
        self.selected = selected
        self.output_root = unique_output_dir(self.project_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_line(f"阶段1输出目录：{self.output_root}")
        self.log_line("阶段1：正在构建全量证据包...")
        _, self.class_stats, self.image_paths_for_ai = build_full_evidence_package(
            self.dataset_dir,
            self.output_root,
            self.class_names,
            selected,
        )
        self.log_line(f"证据图数量：{len(self.image_paths_for_ai)} 张；批大小：{self.batch_size_spin.value()}")
        if not self.image_paths_for_ai:
            QMessageBox.warning(self, "提示", "选中类别在训练集中没有可分析的标注图片。")
            return
        api_key, base_url, model = self.api_settings()
        if api_key:
            try:
                self.diagnosis, raw_response = request_qwen_diagnosis(
                    api_key,
                    base_url,
                    model,
                    self.class_stats,
                    selected,
                    self.image_paths_for_ai,
                    self.batch_size_spin.value(),
                    self.log_line,
                )
                with open(self.output_root / "ai_aug_stage1_diagnosis_raw.json", "w", encoding="utf-8") as f:
                    json.dump(raw_response, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                QMessageBox.warning(self, "AI 诊断失败", f"{exc}\n\n将使用本地诊断占位，方便继续流程。")
                self.diagnosis = {"source": "local_fallback", "batch_summaries": []}
        else:
            QMessageBox.information(self, "提示", "未填写 API Key，将使用本地诊断占位，方便流程调试。")
            self.diagnosis = {"source": "local_fallback", "batch_summaries": []}
        (self.output_root / "ai_aug_stage1_diagnosis.json").write_text(
            json.dumps(self.diagnosis, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.diagnosis_rows = write_diagnosis_report(
            self.output_root,
            self.diagnosis,
            self.class_stats,
            selected,
        )
        self.populate_diagnosis_table(self.diagnosis_rows)
        self.plan_btn.setEnabled(True)
        self.simulate_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.log_line(f"阶段1完成：诊断摘要已显示，报告已保存：{self.output_root / 'ai_aug_stage1_diagnosis_report.md'}")

    def generate_plan(self):
        if not self.output_root or not self.diagnosis:
            QMessageBox.information(self, "提示", "请先执行阶段1：AI诊断数据集。")
            return
        api_key, base_url, model = self.api_settings()
        if api_key:
            try:
                raw_plan, raw_response = request_qwen_plan_from_diagnosis(
                    api_key,
                    base_url,
                    model,
                    self.class_stats,
                    self.selected,
                    self.diagnosis,
                    self.log_line,
                )
                with open(self.output_root / "ai_aug_stage2_plan_raw.json", "w", encoding="utf-8") as f:
                    json.dump(raw_response, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                QMessageBox.warning(self, "AI 计划失败", f"{exc}\n\n将使用本地兜底计划，方便继续流程。")
                raw_plan = local_fallback_plan(self.class_stats, self.selected)
        else:
            raw_plan = local_fallback_plan(self.class_stats, self.selected)
        self.plan = normalize_ai_plan(raw_plan, self.class_stats, self.selected)
        write_plan_report_v2(self.output_root, self.class_stats, self.plan)
        self.populate_plan_table()
        self.simulate_btn.setEnabled(False)
        self.generate_btn.setEnabled(True)
        self.log_line("AI分析+策略完成：请在表格中确认/修改新增数量，然后执行增强。")

    def simulate_and_review(self):
        if not self.plan or not self.output_root:
            QMessageBox.information(self, "提示", "请先执行阶段2：AI生成增强计划。")
            return
        self.plan = self.collect_confirmed_plan()
        write_plan_report_v2(self.output_root, self.class_stats, self.plan)
        self.simulation = simulate_plan_effect(self.class_stats, self.plan)
        api_key, base_url, model = self.api_settings()
        self.simulation_ai_review = None
        if api_key:
            try:
                self.simulation_ai_review, raw_response = request_qwen_simulation_review(
                    api_key,
                    base_url,
                    model,
                    self.class_stats,
                    self.plan,
                    self.simulation,
                )
                with open(self.output_root / "ai_aug_stage3_review_raw.json", "w", encoding="utf-8") as f:
                    json.dump(raw_response, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                QMessageBox.warning(self, "AI 复核失败", f"{exc}\n\n已保留本地模拟结果，可继续执行增强。")
        write_simulation_report(self.output_root, self.simulation, self.simulation_ai_review)
        self.populate_simulation_table()
        self.generate_btn.setEnabled(True)
        self.log_line(f"模拟完成：本地模拟和 AI 复核结果已显示，报告已保存：{self.output_root / 'ai_aug_simulation_review.md'}")

    def analyze(self):
        if not self.dataset_dir.exists():
            QMessageBox.warning(self, "提示", "未找到 dataset，请先点击“生成数据集”。")
            return
        selected = self.selected_class_ids()
        if not selected:
            QMessageBox.information(self, "提示", "请至少选择一个缺陷类别。")
            return
        self.output_root = unique_output_dir(self.project_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_line(f"输出目录：{self.output_root}")
        self.log_line("正在生成选中缺陷训练集全量证据包...")
        _, self.class_stats, image_paths = build_full_evidence_package(
            self.dataset_dir,
            self.output_root,
            self.class_names,
            selected,
        )
        self.log_line(f"证据图数量：{len(image_paths)} 张；每批分析图片数：{self.batch_size_spin.value()}")
        if not image_paths:
            QMessageBox.warning(self, "提示", "选中类别在训练集中没有可分析的标注图片。")
            return
        raw_plan = None
        api_key = self.api_key_edit.text().strip()
        if api_key:
            try:
                self.log_line(f"正在调用千问模型：{self.model_edit.text().strip() or DEFAULT_QWEN_MODEL}")
                raw_plan, raw_response = request_qwen_full_batch_plan(
                    api_key,
                    self.base_url_edit.text().strip() or DEFAULT_QWEN_URL,
                    self.model_edit.text().strip() or DEFAULT_QWEN_MODEL,
                    self.class_stats,
                    selected,
                    image_paths,
                    self.batch_size_spin.value(),
                    self.log_line,
                )
                with open(self.output_root / "ai_aug_qwen_raw_response.json", "w", encoding="utf-8") as f:
                    json.dump(raw_response, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                QMessageBox.warning(self, "AI 分析失败", f"{exc}\n\n将生成本地兜底方案，建议仅用于调试。")
                raw_plan = local_fallback_plan(self.class_stats, selected)
        else:
            QMessageBox.information(self, "提示", "未填写 API Key，将生成本地兜底方案，建议仅用于界面和流程调试。")
            raw_plan = local_fallback_plan(self.class_stats, selected)

        self.plan = normalize_ai_plan(raw_plan, self.class_stats, selected)
        write_plan_report_v2(self.output_root, self.class_stats, self.plan)
        self.populate_plan_table()
        self.generate_btn.setEnabled(True)
        self.log_line("方案已生成，请在表格中确认/修改新增数量。")

    def populate_plan_table(self):
        self.table_mode = "plan"
        rows = self.plan.get("classes", []) if self.plan else []
        self.plan_table.clear()
        self.plan_table.setColumnCount(8)
        self.plan_table.setHorizontalHeaderLabels(["启用", "缺陷", "当前框", "当前图片", "新增图片", "预计新增框", "方法明细", "风险/理由"])
        self.plan_table.setRowCount(len(rows))
        for row_idx, item in enumerate(rows):
            enabled = QCheckBox()
            enabled.setChecked(item.get("target_new_images", 0) > 0)
            holder = QWidget()
            box = QHBoxLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)
            box.setAlignment(Qt.AlignCenter)
            box.addWidget(enabled)
            holder.check = enabled
            self.plan_table.setCellWidget(row_idx, 0, holder)
            self.plan_table.setItem(row_idx, 1, QTableWidgetItem(f"{item['class_id']}: {item['class_name']}"))
            self.plan_table.setItem(row_idx, 2, QTableWidgetItem(str(item.get("current_instances", 0))))
            self.plan_table.setItem(row_idx, 3, QTableWidgetItem(str(item.get("current_images", 0))))
            spin = QSpinBox()
            spin.setRange(0, 10000)
            spin.setValue(int(item.get("target_new_images", 0)))
            self.plan_table.setCellWidget(row_idx, 4, spin)
            self.plan_table.setItem(row_idx, 5, QTableWidgetItem(str(item.get("expected_target_box_delta", ""))))
            self.plan_table.setItem(row_idx, 6, QTableWidgetItem(method_mix_text(item)))
            detail = (
                f"{item.get('risk', '')} | ROI:{item.get('roi_photometric_strength', '')} | "
                f"{item.get('rationale', '')} | {item.get('method_constraints', '')}"
            )
            self.plan_table.setItem(row_idx, 7, QTableWidgetItem(detail))
        self.plan_table.resizeColumnsToContents()

    def populate_diagnosis_table(self, rows):
        self.table_mode = "diagnosis"
        rows = rows or []
        self.plan_table.clear()
        self.plan_table.setColumnCount(8)
        self.plan_table.setHorizontalHeaderLabels(["阶段", "缺陷", "批次/观察", "主要形态", "子形态", "易混淆", "疑似标注", "风险/安全方向"])
        self.plan_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            self.plan_table.setItem(row_idx, 0, QTableWidgetItem("阶段1诊断"))
            self.plan_table.setItem(row_idx, 1, QTableWidgetItem(f"{row.get('class_id', '')}: {row.get('class_name', '')}"))
            observed = f"{row.get('batch_count', 0)} 批 / 观察约 {row.get('observed_count_estimate', 0)}"
            self.plan_table.setItem(row_idx, 2, QTableWidgetItem(observed))
            self.plan_table.setItem(row_idx, 3, QTableWidgetItem(row.get("visual_patterns", "")))
            self.plan_table.setItem(row_idx, 4, QTableWidgetItem(row.get("subtype_hints", "")))
            self.plan_table.setItem(row_idx, 5, QTableWidgetItem(row.get("confusable_with", "")))
            self.plan_table.setItem(row_idx, 6, QTableWidgetItem(row.get("suspect_label_notes", "")))
            risk_text = f"安全: {row.get('safe_augmentation_directions', '')} | 风险: {row.get('unsafe_augmentation_directions', '')}"
            self.plan_table.setItem(row_idx, 7, QTableWidgetItem(risk_text))
        self.plan_table.resizeColumnsToContents()

    def populate_simulation_table(self):
        self.table_mode = "simulation"
        simulation = self.simulation or {}
        rows = simulation.get("per_class", [])
        warnings_by_class = defaultdict(list)
        for warning in simulation.get("warnings", []):
            warnings_by_class[str(warning.get("class_id", ""))].append(
                f"{warning.get('level', '')}: {warning.get('message', '')}"
            )
        ai_review_text = ""
        if isinstance(self.simulation_ai_review, dict):
            approved = self.simulation_ai_review.get("approved")
            blocking = compact_text(self.simulation_ai_review.get("blocking_issues", []), 120)
            review_warnings = compact_text(self.simulation_ai_review.get("review_warnings", []), 120)
            ai_review_text = f"AI approved={approved}; block={blocking}; warn={review_warnings}"

        self.plan_table.clear()
        self.plan_table.setColumnCount(8)
        self.plan_table.setHorizontalHeaderLabels(["阶段", "缺陷", "当前框", "新增图片", "预计新增框", "预计后框数", "方法明细", "告警/AI复核"])
        self.plan_table.setRowCount(len(rows) + 1)
        for row_idx, row in enumerate(rows):
            class_id = str(row.get("class_id", ""))
            methods = ", ".join(f"{m.get('method', '')}:{m.get('images', 0)}" for m in row.get("method_rows", []))
            warnings = compact_text(warnings_by_class.get(class_id, []), 220)
            self.plan_table.setItem(row_idx, 0, QTableWidgetItem("执行前模拟"))
            self.plan_table.setItem(row_idx, 1, QTableWidgetItem(f"{class_id}: {row.get('class_name', '')}"))
            self.plan_table.setItem(row_idx, 2, QTableWidgetItem(str(row.get("current_instances", 0))))
            self.plan_table.setItem(row_idx, 3, QTableWidgetItem(str(row.get("planned_new_images", 0))))
            self.plan_table.setItem(row_idx, 4, QTableWidgetItem(str(row.get("expected_target_box_delta", 0))))
            self.plan_table.setItem(row_idx, 5, QTableWidgetItem(str(row.get("expected_instances_after", 0))))
            self.plan_table.setItem(row_idx, 6, QTableWidgetItem(methods))
            self.plan_table.setItem(row_idx, 7, QTableWidgetItem(warnings))

        summary_idx = len(rows)
        self.plan_table.setItem(summary_idx, 0, QTableWidgetItem("合计"))
        self.plan_table.setItem(summary_idx, 1, QTableWidgetItem(""))
        self.plan_table.setItem(summary_idx, 2, QTableWidgetItem(str(sum(int(row.get("current_instances", 0)) for row in rows))))
        self.plan_table.setItem(summary_idx, 3, QTableWidgetItem(str(simulation.get("total_new_images", 0))))
        self.plan_table.setItem(summary_idx, 4, QTableWidgetItem(str(simulation.get("total_expected_target_box_delta", 0))))
        self.plan_table.setItem(summary_idx, 5, QTableWidgetItem(""))
        self.plan_table.setItem(summary_idx, 6, QTableWidgetItem(""))
        self.plan_table.setItem(summary_idx, 7, QTableWidgetItem(ai_review_text))
        self.plan_table.resizeColumnsToContents()

    def collect_confirmed_plan(self):
        if self.table_mode != "plan":
            return self.plan or {"classes": []}
        rows = []
        for row_idx, item in enumerate(self.plan.get("classes", [])):
            holder = self.plan_table.cellWidget(row_idx, 0)
            spin = self.plan_table.cellWidget(row_idx, 4)
            next_item = dict(item)
            enabled = bool(holder and hasattr(holder, "check") and holder.check.isChecked())
            next_item["target_new_images"] = int(spin.value()) if enabled and spin else 0
            next_item["method_mix"] = normalize_method_mix(next_item, next_item["target_new_images"])
            next_item["method"] = method_mix_text(next_item)
            next_item["expected_target_box_delta"] = expected_target_delta_for_mix(
                self.class_stats.get(str(next_item.get("class_id")), {}),
                next_item["method_mix"],
            )
            rows.append(next_item)
        return {"classes": rows}

    def execute_augmentation(self):
        if not self.plan or not self.output_root:
            QMessageBox.information(self, "提示", "请先执行 AI分析+生成策略。")
            return
        confirmed_plan = self.collect_confirmed_plan()
        if sum(int(item.get("target_new_images", 0)) for item in confirmed_plan.get("classes", [])) <= 0:
            QMessageBox.information(self, "提示", "计划新增数量为 0，没有需要生成的候选。")
            return
        self.plan = confirmed_plan
        self.simulation = simulate_plan_effect(self.class_stats, self.plan)
        write_simulation_report(self.output_root, self.simulation, None)
        api_key, base_url, model = self.api_settings()
        if not api_key:
            QMessageBox.warning(self, "提示", "AI质检是写入 train 的前置条件，请填写 API Key 后再执行增强。")
            return
        try:
            self.log_line("执行增强：正在复制原 dataset 到新数据集目录...")
            copy_dataset_to_output(self.dataset_dir, self.output_root, self.class_names)
            write_plan_report_v2(self.output_root, self.class_stats, self.plan)
            self.log_line("执行增强：正在按 method_mix 生成增强候选...")
            self.generated_rows = generate_augmented_candidates(
                self.dataset_dir,
                self.output_root,
                self.class_names,
                self.plan,
                lambda v: self.log_line(f"生成进度：{v}%") if v in {25, 50, 75, 100} else None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "生成失败", str(exc))
            return
        if not self.generated_rows:
            QMessageBox.warning(self, "提示", "没有生成任何增强候选，请检查选中类别、标签或方案。")
            return
        local_review = local_quality_review(self.output_root, self.generated_rows)
        ai_review = None
        ai_flagged_rows = []
        ai_uncovered_rows = []
        try:
            self.log_line("AI质检：正在分批检查全部候选图；覆盖不完整的候选不会写入 train。")
            ai_review, raw_response = request_qwen_quality_review(
                api_key,
                base_url,
                model,
                self.plan,
                self.simulation,
                local_review,
                self.generated_rows,
            )
            with open(self.output_root / "ai_aug_stage4_quality_raw.json", "w", encoding="utf-8") as f:
                json.dump(raw_response, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            ai_review = {
                "approved": False,
                "coverage_complete": False,
                "candidate_count": len(self.generated_rows),
                "covered_candidate_images": [],
                "uncovered_candidate_images": [Path(row["candidate_image"]).name for row in self.generated_rows],
                "by_candidate": {},
                "review_warnings": [f"AI质检调用失败：{exc}"],
                "rationale": "AI质检失败，全部候选视为未覆盖，不自动写入 train。",
            }
            QMessageBox.warning(self, "AI 质检失败", f"{exc}\n\n全部候选已保留在 review/pending；未被 AI 覆盖，不会自动写入 train。")

        local_accepted_rows, local_rejected_rows = split_rows_by_quality(self.generated_rows, local_review)
        ai_pass_rows, ai_blocked_rows, ai_flagged_rows, ai_uncovered_rows = split_rows_by_ai_coverage(self.generated_rows, ai_review)
        local_pass_names = {Path(row["candidate_image"]).name for row in local_accepted_rows}
        ai_pass_names = {Path(row["candidate_image"]).name for row in ai_pass_rows}
        accepted_names = local_pass_names & ai_pass_names
        accepted_rows = [
            row for row in self.generated_rows
            if Path(row["candidate_image"]).name in accepted_names
        ]
        rejected_rows = [
            row for row in self.generated_rows
            if Path(row["candidate_image"]).name not in accepted_names
        ]
        accepted = accept_candidates(self.output_root, accepted_rows)
        write_quality_report(
            self.output_root,
            local_review,
            ai_review,
            accepted,
            rejected_rows,
            ai_flagged_rows=ai_flagged_rows,
            ai_uncovered_rows=ai_uncovered_rows,
            ai_quality_mode="required_full_coverage",
        )
        self.populate_simulation_table()
        self.log_line(
            f"执行完成：候选 {len(self.generated_rows)} 张，写入 train {len(accepted)} 张，"
            f"本地硬拒绝 {len(local_rejected_rows)} 张，AI疑点 {len(ai_flagged_rows)} 张，"
            f"AI未覆盖 {len(ai_uncovered_rows)} 张。"
        )
        QMessageBox.information(
            self,
            "完成",
            f"AI 增强完成：\n{self.output_root}\n\n"
            f"候选：{len(self.generated_rows)} 张\n"
            f"写入 train：{len(accepted)} 张\n"
            f"本地硬拒绝：{len(local_rejected_rows)} 张\n"
            f"AI疑点/拒绝：{len(ai_flagged_rows)} 张\n"
            f"AI未覆盖：{len(ai_uncovered_rows)} 张\n"
            f"data.yaml：{self.output_root / 'data.yaml'}"
        )

    def stage4_execute_quality(self):
        self.execute_augmentation()

    def generate_candidates(self):
        self.execute_augmentation()
        return
        if not self.plan or not self.output_root:
            return
        confirmed_plan = self.collect_confirmed_plan()
        if sum(int(item.get("target_new_images", 0)) for item in confirmed_plan.get("classes", [])) <= 0:
            QMessageBox.information(self, "提示", "新增数量为 0，没有需要生成的候选。")
            return
        try:
            self.log_line("正在复制原 dataset 到新数据集目录...")
            copy_dataset_to_output(self.dataset_dir, self.output_root, self.class_names)
            self.plan = confirmed_plan
            write_plan_report_v2(self.output_root, self.class_stats, self.plan)
            self.log_line("正在生成增强候选...")
            self.generated_rows = generate_augmented_candidates(
                self.dataset_dir,
                self.output_root,
                self.class_names,
                self.plan,
                lambda v: self.log_line(f"生成进度：{v}%") if v in {25, 50, 75, 100} else None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "生成失败", str(exc))
            return
        if not self.generated_rows:
            QMessageBox.warning(self, "提示", "没有生成任何增强候选，请检查选中类别、标签或方案。")
            return
        self.log_line(f"候选已生成：{len(self.generated_rows)} 张，正在自动写入新数据集 train。")
        accepted = accept_candidates(self.output_root, self.generated_rows)
        self.log_line(f"已写入 train：{len(accepted)} 张。")
        QMessageBox.information(
            self,
            "完成",
            f"新数据集已生成：\n{self.output_root}\n\n"
            f"生成增强数据：{len(self.generated_rows)} 张\n"
            f"已写入 train：{len(accepted)} 张\n"
            f"data.yaml：{self.output_root / 'data.yaml'}"
        )
