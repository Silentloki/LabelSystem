import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
AUGMENT_SPLITS = {"train"}

PATCH_PAD = 6
SURFACE_CONTEXT_PAD = 12
MIN_BOX_SIZE = 3.0
MAX_OVERLAP_IOU = 0.35
MAX_PLACEMENT_TRIES = 20
MAX_ATTEMPTS_PER_CLASS_FACTOR = 20
SURFACE_MEAN_DIFF_MAX = 35.0
SURFACE_STD_DIFF_MAX = 35.0
WHITE_FOREGROUND_MIN_RATIO = 0.60
WHITE_FOREGROUND_RATIO_TOLERANCE = 0.05


def class_sort_key(value):
    value = str(value)
    if value.isdigit():
        return 0, int(value)
    return 1, value


def parse_split_and_inner(labels_dir, label_path):
    rel_path = label_path.relative_to(labels_dir)
    parts = rel_path.parts
    if parts and parts[0] in {"train", "val"}:
        split = parts[0]
        inner = Path(*parts[1:]) if len(parts) > 1 else Path(rel_path.name)
        has_explicit_split = True
    else:
        split = "train"
        inner = rel_path
        has_explicit_split = False
    return split, inner, has_explicit_split


def find_image_for_label(images_dir, labels_dir, label_path):
    split, inner_label, has_explicit_split = parse_split_and_inner(labels_dir, label_path)
    if has_explicit_split:
        image_base = (images_dir / split / inner_label).with_suffix("")
    else:
        image_base = (images_dir / inner_label).with_suffix("")

    for ext in IMAGE_EXTS:
        image_path = image_base.with_suffix(ext)
        if image_path.exists():
            return image_path, split, inner_label
    return None, split, inner_label


def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = parts[0]
                x_center, y_center, box_w, box_h = map(float, parts[1:5])
            except ValueError:
                continue
            x_center *= img_w
            y_center *= img_h
            box_w *= img_w
            box_h *= img_h
            x1 = x_center - box_w / 2.0
            y1 = y_center - box_h / 2.0
            x2 = x_center + box_w / 2.0
            y2 = y_center + box_h / 2.0
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def collect_train_detection_counts(label_files, labels_dir):
    counts = Counter()
    train_label_files = []
    skipped_segmentation_lines = 0

    for label_path in label_files:
        split, _, _ = parse_split_and_inner(labels_dir, label_path)
        if split != "train":
            continue
        train_label_files.append(label_path)
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    counts[parts[0]] += 1
                elif len(parts) > 5:
                    skipped_segmentation_lines += 1

    return counts, train_label_files, skipped_segmentation_lines


def load_class_name_map(data_yaml_path):
    if not data_yaml_path.exists():
        return {}

    text = data_yaml_path.read_text(encoding="utf-8", errors="ignore")
    name_map = {}
    in_names_block = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("names:"):
            in_names_block = True
            after_colon = stripped[len("names:"):].strip()
            if after_colon.startswith("[") and after_colon.endswith("]"):
                values = [item.strip().strip("'\"") for item in after_colon[1:-1].split(",")]
                for idx, value in enumerate(values):
                    if value:
                        name_map[str(idx)] = value
                return name_map
            continue
        if in_names_block:
            if raw_line.startswith(" ") or raw_line.startswith("\t"):
                if ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")
                if key and value:
                    name_map[key] = value
                continue
            break
    return name_map


def class_display_text(cls_id, class_name_map):
    cls_name = class_name_map.get(str(cls_id))
    if cls_name:
        return f"{cls_id}({cls_name})"
    return str(cls_id)


def save_image(image_path, image, gray_bmp=True):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif suffix == ".bmp":
        if gray_bmp and image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(image_path), image)
    else:
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def build_augmented_paths(output_images_dir, output_labels_dir, cls_id, inner_label, image_suffix, serial_index):
    safe_stem = str(inner_label.with_suffix("")).replace("\\", "__").replace("/", "__")
    aug_image_name = f"cls{cls_id}_{safe_stem}_aug_{serial_index:05d}{image_suffix}"
    aug_label_name = f"cls{cls_id}_{safe_stem}_aug_{serial_index:05d}.txt"
    return output_images_dir / aug_image_name, output_labels_dir / aug_label_name


def yolo_line_from_box(box, img_w, img_h):
    cls_id, x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    x_center = x1 + box_w / 2.0
    y_center = y1 + box_h / 2.0
    return (
        f"{cls_id} "
        f"{x_center / img_w:.6f} "
        f"{y_center / img_h:.6f} "
        f"{box_w / img_w:.6f} "
        f"{box_h / img_h:.6f}"
    )


def write_yolo_boxes(label_path, boxes, img_w, img_h):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [yolo_line_from_box(box, img_w, img_h) for box in boxes]
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def box_center(box):
    _, x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_iou(box_a, box_b):
    _, ax1, ay1, ax2, ay2 = box_a
    _, bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def max_overlap_with_boxes(candidate_box, other_boxes):
    if not other_boxes:
        return 0.0
    return max(box_iou(candidate_box, other_box) for other_box in other_boxes)


def extract_defect_patch(image, box):
    _, x1, y1, x2, y2 = box
    img_h, img_w = image.shape[:2]
    box_x1 = int(np.floor(x1))
    box_y1 = int(np.floor(y1))
    box_x2 = int(np.ceil(x2))
    box_y2 = int(np.ceil(y2))
    pad_x1 = max(0, box_x1 - PATCH_PAD)
    pad_y1 = max(0, box_y1 - PATCH_PAD)
    pad_x2 = min(img_w, box_x2 + PATCH_PAD)
    pad_y2 = min(img_h, box_y2 + PATCH_PAD)
    patch = image[pad_y1:pad_y2, pad_x1:pad_x2].copy()
    mask = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
    local_x1 = box_x1 - pad_x1
    local_y1 = box_y1 - pad_y1
    local_x2 = local_x1 + (box_x2 - box_x1)
    local_y2 = local_y1 + (box_y2 - box_y1)
    mask[local_y1:local_y2, local_x1:local_x2] = 255
    return patch, mask


def extract_surface_stats(image, box):
    _, x1, y1, x2, y2 = box
    img_h, img_w = image.shape[:2]
    box_x1 = int(np.floor(x1))
    box_y1 = int(np.floor(y1))
    box_x2 = int(np.ceil(x2))
    box_y2 = int(np.ceil(y2))
    pad_x1 = max(0, box_x1 - SURFACE_CONTEXT_PAD)
    pad_y1 = max(0, box_y1 - SURFACE_CONTEXT_PAD)
    pad_x2 = min(img_w, box_x2 + SURFACE_CONTEXT_PAD)
    pad_y2 = min(img_h, box_y2 + SURFACE_CONTEXT_PAD)
    roi = image[pad_y1:pad_y2, pad_x1:pad_x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def build_white_foreground_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def white_foreground_ratio(mask, box):
    _, x1, y1, x2, y2 = box
    img_h, img_w = mask.shape[:2]
    box_x1 = max(0, int(np.floor(x1)))
    box_y1 = max(0, int(np.floor(y1)))
    box_x2 = min(img_w, int(np.ceil(x2)))
    box_y2 = min(img_h, int(np.ceil(y2)))
    if box_x2 <= box_x1 or box_y2 <= box_y1:
        return 0.0
    roi = mask[box_y1:box_y2, box_x1:box_x2]
    if roi.size == 0:
        return 0.0
    return float(cv2.countNonZero(roi)) / float(roi.size)


def remove_original_defect(image, box):
    _, x1, y1, x2, y2 = box
    inpainted = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    box_x1 = int(np.floor(x1))
    box_y1 = int(np.floor(y1))
    box_x2 = int(np.ceil(x2))
    box_y2 = int(np.ceil(y2))
    mask[box_y1:box_y2, box_x1:box_x2] = 255
    return cv2.inpaint(inpainted, mask, 3, cv2.INPAINT_TELEA)


def rotate_patch_if_needed(patch, mask, mode, rotate_range):
    if mode not in {"rotate", "both"}:
        return patch, mask
    angle = random.uniform(*rotate_range)
    h, w = patch.shape[:2]
    diagonal = int(np.ceil(np.sqrt(w ** 2 + h ** 2)))
    pad_w = (diagonal - w) // 2
    pad_h = (diagonal - h) // 2
    padded_patch = cv2.copyMakeBorder(patch, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    padded_mask = cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    center = diagonal // 2, diagonal // 2
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_patch = cv2.warpAffine(
        padded_patch,
        matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated_mask = cv2.warpAffine(
        padded_mask,
        matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_patch, rotated_mask


def crop_patch_to_mask(patch, mask):
    if cv2.countNonZero(mask) < 10:
        return None, None
    x, y, w, h = cv2.boundingRect(mask)
    return patch[y:y + h, x:x + w], mask[y:y + h, x:x + w]


def choose_paste_center(base_box, patch_w, patch_h, img_w, img_h, mode, translate_ratio):
    base_cx, base_cy = box_center(base_box)
    min_cx = patch_w / 2.0 + 1
    max_cx = img_w - patch_w / 2.0 - 1
    min_cy = patch_h / 2.0 + 1
    max_cy = img_h - patch_h / 2.0 - 1
    if min_cx > max_cx or min_cy > max_cy:
        return None
    if mode == "rotate":
        cx = min(max(base_cx, min_cx), max_cx)
        cy = min(max(base_cy, min_cy), max_cy)
        return cx, cy
    shift_x = random.uniform(-translate_ratio, translate_ratio) * img_w
    shift_y = random.uniform(-translate_ratio, translate_ratio) * img_h
    cx = min(max(base_cx + shift_x, min_cx), max_cx)
    cy = min(max(base_cy + shift_y, min_cy), max_cy)
    return cx, cy


def build_box_from_center(cls_id, center_x, center_y, box_w, box_h):
    x1 = center_x - box_w / 2.0
    y1 = center_y - box_h / 2.0
    x2 = x1 + box_w
    y2 = y1 + box_h
    return cls_id, x1, y1, x2, y2


def candidate_matches_surface(image, candidate_box, reference_surface_stats):
    if reference_surface_stats is None:
        return True
    candidate_surface_stats = extract_surface_stats(image, candidate_box)
    if candidate_surface_stats is None:
        return False
    ref_mean, ref_std = reference_surface_stats
    cand_mean, cand_std = candidate_surface_stats
    return (
        abs(cand_mean - ref_mean) <= SURFACE_MEAN_DIFF_MAX
        and abs(cand_std - ref_std) <= SURFACE_STD_DIFF_MAX
    )


def candidate_on_white_foreground(foreground_mask, candidate_box, reference_ratio):
    candidate_ratio = white_foreground_ratio(foreground_mask, candidate_box)
    min_ratio = max(WHITE_FOREGROUND_MIN_RATIO, reference_ratio - WHITE_FOREGROUND_RATIO_TOLERANCE)
    return candidate_ratio >= min_ratio


def augment_one_defect(image, boxes, candidate_indices, mode, rotate_range, translate_ratio, foreground_mask=None):
    target_index = random.choice(candidate_indices)
    target_box = boxes[target_index]
    cls_id = target_box[0]
    patch, mask = extract_defect_patch(image, target_box)
    patch, mask = rotate_patch_if_needed(patch, mask, mode, rotate_range)
    patch, mask = crop_patch_to_mask(patch, mask)
    if patch is None:
        return None, None
    patch_h, patch_w = patch.shape[:2]
    if patch_h < MIN_BOX_SIZE or patch_w < MIN_BOX_SIZE:
        return None, None
    if foreground_mask is None:
        foreground_mask = build_white_foreground_mask(image)
    reference_foreground_ratio = white_foreground_ratio(foreground_mask, target_box)
    reference_surface_stats = extract_surface_stats(image, target_box)
    cleaned_image = remove_original_defect(image, target_box)
    other_boxes = [box for idx, box in enumerate(boxes) if idx != target_index]

    candidate_center = None
    candidate_box = None
    for _ in range(MAX_PLACEMENT_TRIES):
        center = choose_paste_center(target_box, patch_w, patch_h, image.shape[1], image.shape[0], mode, translate_ratio)
        if center is None:
            return None, None
        cx, cy = center
        candidate_box = build_box_from_center(cls_id, cx, cy, patch_w, patch_h)
        if (
            max_overlap_with_boxes(candidate_box, other_boxes) <= MAX_OVERLAP_IOU
            and candidate_on_white_foreground(foreground_mask, candidate_box, reference_foreground_ratio)
            and candidate_matches_surface(cleaned_image, candidate_box, reference_surface_stats)
        ):
            candidate_center = center
            break
    if candidate_center is None:
        return None, None

    center_int = int(round(candidate_center[0])), int(round(candidate_center[1]))
    try:
        blended = cv2.seamlessClone(patch, cleaned_image, mask, center_int, cv2.NORMAL_CLONE)
    except cv2.error:
        return None, None
    final_boxes = list(other_boxes)
    final_boxes.append(candidate_box)
    return blended, final_boxes


def build_source_pool(label_files, images_dir, labels_dir, selected_classes):
    pool = defaultdict(list)
    for label_path in label_files:
        image_path, split, inner_label = find_image_for_label(images_dir, labels_dir, label_path)
        if image_path is None or split not in AUGMENT_SPLITS:
            continue
        class_to_indices = defaultdict(list)
        valid_box_count = 0
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_to_indices[parts[0]].append(valid_box_count)
                valid_box_count += 1
        if valid_box_count == 0:
            continue
        for cls_id in selected_classes:
            if class_to_indices.get(cls_id):
                pool[cls_id].append(
                    {
                        "image_path": image_path,
                        "label_path": label_path,
                        "inner_label": inner_label,
                        "indices": class_to_indices[cls_id],
                    }
                )
    return pool


def load_sample_payload(sample):
    if "image" in sample and "boxes" in sample:
        return sample
    image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return None
    img_h, img_w = image.shape[:2]
    boxes = load_yolo_boxes(sample["label_path"], img_w, img_h)
    if not boxes:
        return None
    sample["image"] = image
    sample["img_h"] = img_h
    sample["img_w"] = img_w
    sample["boxes"] = boxes
    sample["foreground_mask"] = build_white_foreground_mask(image)
    return sample


def write_report(output_root, input_root, class_name_map, train_counts, target_counts, generated_counts, skipped_lines, skipped_attempts):
    lines = [
        "rotate + translate detection augmentation report",
        "================================================",
        f"input_dataset: {input_root}",
        f"output_dataset: {output_root}",
        "mode: both",
        "rotate_range: -180.0 to 180.0",
        "translate_ratio: 0.15",
        "seed: 42",
        "source_split: train",
        "label_format: YOLO detection only (class x_center y_center width height)",
        f"skipped_segmentation_lines: {skipped_lines}",
        "",
        "per_class:",
    ]
    total = 0
    for cls_id in sorted(target_counts, key=class_sort_key):
        generated = generated_counts.get(cls_id, 0)
        total += generated
        lines.append(
            f"- {class_display_text(cls_id, class_name_map)}: "
            f"train_boxes={train_counts.get(cls_id, 0)}, "
            f"planned_generated_images={target_counts[cls_id]}, "
            f"actual_generated_images={generated}"
        )
    lines.extend(
        [
            "",
            f"total_generated_images: {total}",
            f"skipped_failed_augment_attempts: {skipped_attempts}",
            "",
            "outputs:",
            f"- data_yaml: {output_root / 'data.yaml'}",
            f"- train_images: {output_root / 'images' / 'train'}",
            f"- train_labels: {output_root / 'labels' / 'train'}",
            f"- report: {output_root / 'augmentation_report.txt'}",
        ]
    )
    (output_root / "augmentation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def augment_dataset(input_root, output_root, target_counts, progress_callback=None):
    input_root = Path(input_root)
    output_root = Path(output_root)
    target_counts = {str(k): int(v) for k, v in (target_counts or {}).items() if int(v) > 0}
    if not target_counts:
        return {"generated": {}, "skipped_attempts": 0}
    if not (input_root / "images").exists() or not (input_root / "labels").exists():
        raise RuntimeError(f"dataset images/labels not found: {input_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(input_root, output_root)

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    output_images_dir = images_dir / "train"
    output_labels_dir = labels_dir / "train"
    data_yaml_path = output_root / "data.yaml"

    random.seed(42)
    np.random.seed(42)

    label_files = sorted(labels_dir.rglob("*.txt"))
    train_counts, train_label_files, skipped_segmentation_lines = collect_train_detection_counts(label_files, labels_dir)
    selected_classes = sorted(target_counts, key=class_sort_key)
    class_name_map = load_class_name_map(data_yaml_path)
    source_pool = build_source_pool(train_label_files, images_dir, labels_dir, selected_classes)

    generated_counts = Counter({cls_id: 0 for cls_id in selected_classes})
    skipped_failed_augment = 0
    serial_index = 1
    total_target = sum(target_counts.values())
    generated_total = 0

    for cls_id in selected_classes:
        candidates = source_pool.get(cls_id, [])
        if not candidates:
            continue
        target = target_counts[cls_id]
        max_attempts = target * MAX_ATTEMPTS_PER_CLASS_FACTOR
        attempts = 0
        while generated_counts[cls_id] < target and attempts < max_attempts:
            attempts += 1
            sample = load_sample_payload(random.choice(candidates))
            if sample is None:
                skipped_failed_augment += 1
                continue
            augmented_image, augmented_boxes = augment_one_defect(
                sample["image"],
                sample["boxes"],
                sample["indices"],
                "both",
                (-180.0, 180.0),
                0.15,
                sample.get("foreground_mask"),
            )
            if augmented_image is None or not augmented_boxes:
                skipped_failed_augment += 1
                continue

            out_image_path, out_label_path = build_augmented_paths(
                output_images_dir,
                output_labels_dir,
                cls_id,
                sample["inner_label"],
                sample["image_path"].suffix,
                serial_index,
            )
            save_image(out_image_path, augmented_image, gray_bmp=True)
            write_yolo_boxes(out_label_path, augmented_boxes, sample["img_w"], sample["img_h"])
            generated_counts[cls_id] += 1
            generated_total += 1
            serial_index += 1
            if progress_callback and total_target > 0:
                progress_callback(int((generated_total / total_target) * 100))

    write_report(
        output_root,
        input_root,
        class_name_map,
        train_counts,
        target_counts,
        generated_counts,
        skipped_segmentation_lines,
        skipped_failed_augment,
    )
    if progress_callback:
        progress_callback(100)
    return {
        "generated": dict(generated_counts),
        "skipped_attempts": skipped_failed_augment,
        "skipped_segmentation_lines": skipped_segmentation_lines,
    }
