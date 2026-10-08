import random
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal
import os
import shutil
import numpy as np
import json
import csv
from collections import defaultdict
from Utils import DetectionAugment


class WorkerSignals(QObject):
    sendValue = pyqtSignal(int)


def build_dataset_split(flag, mode, good=None):
    good = good if good is not None else []
    train = []
    val = []
    count = 0

    if mode == 2:
        bad_indices = list(flag)
        random.shuffle(bad_indices)
        badcount = len(bad_indices)
        num = int(badcount * 0.8)
        train = bad_indices[:num]
        val = bad_indices[num:]

        selected_good = list(good)
        random.shuffle(selected_good)
        good_count = len(selected_good)
        num_good = int(good_count * 0.66)
        train.extend(selected_good[:num_good])
        val.extend(selected_good[num_good:])
        count = badcount + good_count

    elif mode == 1:
        arr = np.array(flag)
        bad_indices = np.where(arr == 1)[0].tolist()
        random.shuffle(bad_indices)
        badcount = len(bad_indices)
        num = int(badcount * 0.8)
        train = bad_indices[:num]
        val = bad_indices[num:]

        selected_good = list(good)
        random.shuffle(selected_good)
        goodcount = len(selected_good)
        num_good = int(goodcount * 0.66)
        train.extend(selected_good[:num_good])
        val.extend(selected_good[num_good:])
        count = goodcount + badcount

    return train, val, count


def normalize_rel_path(path):
    return os.path.normpath(str(path)).replace("\\", "/").lower()


def dedupe_indices_by_path(indices, relative_paths):
    result = []
    seen = set()
    for index in indices:
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(relative_paths):
            continue
        key = normalize_rel_path(relative_paths[index])
        if key in seen:
            continue
        seen.add(key)
        result.append(index)
    return result


def validate_split_no_overlap(relative_paths, train_indices, val_indices):
    train_paths = {
        normalize_rel_path(relative_paths[index])
        for index in train_indices
        if 0 <= index < len(relative_paths)
    }
    val_paths = {
        normalize_rel_path(relative_paths[index])
        for index in val_indices
        if 0 <= index < len(relative_paths)
    }
    path_overlap = sorted(train_paths & val_paths)

    train_names = {
        os.path.basename(str(relative_paths[index])).lower()
        for index in train_indices
        if 0 <= index < len(relative_paths)
    }
    val_names = {
        os.path.basename(str(relative_paths[index])).lower()
        for index in val_indices
        if 0 <= index < len(relative_paths)
    }
    name_overlap = sorted(train_names & val_names)

    if path_overlap or name_overlap:
        details = []
        if path_overlap:
            details.append("same relative image path: " + ", ".join(path_overlap[:10]))
        if name_overlap:
            details.append("same image filename: " + ", ".join(name_overlap[:10]))
        raise RuntimeError("train/val split overlap detected; " + " | ".join(details))


def resolve_bad_indices(flag, mode):
    if mode == 2:
        return list(flag)
    if mode == 1:
        arr = np.array(flag)
        return np.where(arr == 1)[0].tolist()
    return []


def assignment_lookup(sample_assignments):
    return {
        normalize_rel_path(path): assignment
        for path, assignment in (sample_assignments or {}).items()
        if isinstance(assignment, dict)
    }


def get_assignment_for_index(index, relative_paths, assignments_by_path):
    if index < 0 or index >= len(relative_paths):
        return {}
    return assignments_by_path.get(normalize_rel_path(relative_paths[index]), {})


def image_strata(index, relative_paths, categories, sample_assignments, image_labels_by_index=None):
    categories = list(categories or [])
    category_set = set(categories)
    assignments_by_path = assignment_lookup(sample_assignments)
    assignment = get_assignment_for_index(index, relative_paths, assignments_by_path)
    labels = set(image_labels_by_index.get(index, set())) if image_labels_by_index else set(assignment.keys())

    strata = []
    for label in categories:
        if label not in labels and label not in assignment:
            continue
        subclass = str(assignment.get(label, "") or "无子分类")
        strata.append((str(label), subclass))

    if not strata:
        for label, subclass in assignment.items():
            if category_set and label not in category_set:
                continue
            strata.append((str(label), str(subclass or "无子分类")))

    return tuple(sorted(set(strata))) or (("未分层样本", "无子分类"),)


def validation_quota(total):
    if total >= 10:
        return min(total - 1, max(2, int(round(total * 0.2))))
    if total >= 3:
        return 1
    return 0


def split_good_indices(good_indices, relative_paths, rng=None):
    good_indices = dedupe_indices_by_path(good_indices or [], relative_paths)
    (rng or random).shuffle(good_indices)
    train_count = int(len(good_indices) * 0.66)
    return good_indices[:train_count], good_indices[train_count:]


def build_subsample_stratified_split(
    flag,
    mode,
    relative_paths,
    categories,
    sample_assignments=None,
    good=None,
    image_labels_by_index=None,
    val_ratio=0.2,
    rng=None,
):
    rng = rng or random
    bad_indices = dedupe_indices_by_path(resolve_bad_indices(flag, mode), relative_paths)
    good_train, good_val = split_good_indices(good or [], relative_paths, rng)

    index_to_strata = {
        index: image_strata(index, relative_paths, categories, sample_assignments, image_labels_by_index)
        for index in bad_indices
    }
    strata_to_indices = defaultdict(list)
    for index, strata in index_to_strata.items():
        for key in strata:
            strata_to_indices[key].append(index)

    remaining_quota = {
        key: validation_quota(len(indices))
        for key, indices in strata_to_indices.items()
    }
    val_set = set()

    while any(quota > 0 for quota in remaining_quota.values()):
        active = [
            key for key, quota in remaining_quota.items()
            if quota > 0 and any(index not in val_set for index in strata_to_indices[key])
        ]
        if not active:
            break
        active.sort(key=lambda key: (len(strata_to_indices[key]), -remaining_quota[key], key[0], key[1]))
        key = active[0]
        candidates = [index for index in strata_to_indices[key] if index not in val_set]
        if not candidates:
            remaining_quota[key] = 0
            continue
        rng.shuffle(candidates)
        best_index = max(
            candidates,
            key=lambda index: (
                sum(max(0, remaining_quota.get(stratum, 0)) for stratum in index_to_strata[index]),
                len(index_to_strata[index]),
            )
        )
        val_set.add(best_index)
        for stratum in index_to_strata[best_index]:
            if remaining_quota.get(stratum, 0) > 0:
                remaining_quota[stratum] -= 1

    target_val_count = int(round(len(bad_indices) * val_ratio))
    if len(bad_indices) >= 5:
        target_val_count = max(1, target_val_count)
    remaining_bad = [index for index in bad_indices if index not in val_set]
    rng.shuffle(remaining_bad)
    while len(val_set) < target_val_count and remaining_bad:
        val_set.add(remaining_bad.pop())

    val_bad = [index for index in bad_indices if index in val_set]
    train_bad = [index for index in bad_indices if index not in val_set]

    bad_path_keys = {normalize_rel_path(relative_paths[index]) for index in bad_indices}
    good_train = [index for index in good_train if normalize_rel_path(relative_paths[index]) not in bad_path_keys]
    good_val = [index for index in good_val if normalize_rel_path(relative_paths[index]) not in bad_path_keys]

    train = dedupe_indices_by_path(train_bad + good_train, relative_paths)
    val = dedupe_indices_by_path(val_bad + good_val, relative_paths)
    validate_split_no_overlap(relative_paths, train, val)
    return train, val, len(train) + len(val)


class Worker(QRunnable):
    def __init__(self, project_dir, relative_paths, flag, categories, mode, good=None, sample_groups=None,
                 split_indices=None, augment_config=None, sample_assignments=None):
        super().__init__()
        self.project_dir = project_dir
        self.relative_paths = relative_paths
        self.flag = flag
        self.categories = categories
        self.workSingals = WorkerSignals()
        self.mode = mode
        self.good = good if good is not None else []
        self.good_set = set(self.good)
        self.sample_groups = sample_groups or {}
        self.split_indices = split_indices
        self.augment_config = augment_config or {}
        self.sample_assignments = sample_assignments or {}
        self.sample_assignment_lookup = {
            self.normalize_rel_path(path): assignment
            for path, assignment in self.sample_assignments.items()
            if isinstance(assignment, dict)
        }

    @staticmethod
    def normalize_rel_path(path):
        return os.path.normpath(str(path)).replace("\\", "/").lower()

    @staticmethod
    def clamp01(value):
        return max(0.0, min(1.0, float(value)))

    def get_subclass(self, rel_path, label_name):
        assignment = self.sample_assignments.get(rel_path)
        if assignment is None:
            assignment = self.sample_assignment_lookup.get(self.normalize_rel_path(rel_path), {})
        if not isinstance(assignment, dict):
            return ""
        return str(assignment.get(label_name, "") or "")

    def get_normalized_points(self, annotation, content):
        points = annotation.get('points', [])
        normalized_points = []
        raw_points = []
        for point in points:
            if 'x' not in point or 'y' not in point:
                continue
            try:
                raw_points.append((float(point['x']), float(point['y'])))
            except (TypeError, ValueError):
                continue
        if not raw_points:
            return []

        max_abs = max(max(abs(x), abs(y)) for x, y in raw_points)
        if max_abs <= 1.5:
            for x, y in raw_points:
                normalized_points.append((self.clamp01(x), self.clamp01(y)))
            return normalized_points

        width = float(content.get('image_width') or content.get('width') or 0)
        height = float(content.get('image_height') or content.get('height') or 0)
        if width <= 0 or height <= 0:
            return []
        for x, y in raw_points:
            normalized_points.append((self.clamp01(x / width), self.clamp01(y / height)))
        return normalized_points

    def bbox_from_normalized_points(self, points):
        if len(points) < 2:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None
        return x1, y1, x2, y2

    def bbox_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)

    def build_manifest_row(
        self,
        split_name,
        image_name,
        rel_path,
        pre_name,
        ann_index,
        label_line_index,
        label_name,
        subclass,
        label_idx,
        annotation_type,
        bbox,
    ):
        cx, cy, width, height = self.bbox_to_xywh(bbox)
        return {
            'split': split_name,
            'dataset_image': f'images/{split_name}/{image_name}',
            'dataset_label': f'labels/{split_name}/{pre_name}.txt',
            'original_image': str(rel_path).replace("\\", "/"),
            'json_file': f'jsons/{pre_name}.json',
            'ann_index': ann_index,
            'label_line_index': label_line_index,
            'main_label': label_name,
            'subclass': subclass,
            'subclass_status': 'assigned' if subclass else 'missing_subclass',
            'class_id': label_idx,
            'annotation_type': annotation_type,
            'bbox_x1': f'{bbox[0]:.6f}',
            'bbox_y1': f'{bbox[1]:.6f}',
            'bbox_x2': f'{bbox[2]:.6f}',
            'bbox_y2': f'{bbox[3]:.6f}',
            'bbox_cx': f'{cx:.6f}',
            'bbox_cy': f'{cy:.6f}',
            'bbox_w': f'{width:.6f}',
            'bbox_h': f'{height:.6f}',
        }

    def write_subclass_manifest(self, rows, dataset_dir=None):
        dataset_dir = dataset_dir or os.path.join(self.project_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        fieldnames = [
            'split',
            'dataset_image',
            'dataset_label',
            'original_image',
            'json_file',
            'ann_index',
            'label_line_index',
            'main_label',
            'subclass',
            'subclass_status',
            'class_id',
            'annotation_type',
            'bbox_x1',
            'bbox_y1',
            'bbox_x2',
            'bbox_y2',
            'bbox_cx',
            'bbox_cy',
            'bbox_w',
            'bbox_h',
        ]

        def write_csv(path, csv_rows):
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        write_csv(os.path.join(dataset_dir, 'subclass_manifest.csv'), rows)
        write_csv(
            os.path.join(dataset_dir, 'subclass_manifest_train.csv'),
            [row for row in rows if row['split'] == 'train']
        )
        write_csv(
            os.path.join(dataset_dir, 'subclass_manifest_val.csv'),
            [row for row in rows if row['split'] == 'val']
        )

        summary = defaultdict(lambda: {'images': set(), 'gt_boxes': 0})
        for row in rows:
            key = (
                row.get('split', ''),
                row.get('main_label', ''),
                row.get('subclass', ''),
                row.get('subclass_status', ''),
            )
            summary[key]['images'].add(row.get('original_image', ''))
            summary[key]['gt_boxes'] += 1

        summary_path = os.path.join(dataset_dir, 'subclass_split_summary.csv')
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['split', 'main_label', 'subclass', 'subclass_status', 'image_count', 'gt_boxes']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(summary):
                split, main_label, subclass, subclass_status = key
                writer.writerow({
                    'split': split,
                    'main_label': main_label,
                    'subclass': subclass,
                    'subclass_status': subclass_status,
                    'image_count': len(summary[key]['images']),
                    'gt_boxes': summary[key]['gt_boxes'],
                })

    def create_yolo_yaml(self, dataset_dir=None):
        try:
            # 配置数据（根据用户输入调整）
            classes_line = 'names:\n'
            for i, cls in enumerate(self.categories):
                classes_line += f"  {i}: {cls}\n"

            lines = f'train: images/train\nval: images/val\n\nnc: {len(self.categories)}\n{classes_line}'
            yamlPath = os.path.join(dataset_dir or os.path.join(self.project_dir, 'dataset'), 'data.yaml')
            # 必须指定 utf-8，否则有中文类别名会乱码
            with open(yamlPath, 'w', encoding='utf-8') as f:
                f.write(lines)
        except Exception as e:
            if dataset_dir is not None:
                raise
            print(f"生成 yaml 文件失败: {e}")

    def create_dataset_description_files(self, dataset_dir=None):
        explicit_output = dataset_dir is not None
        try:
            dataset_dir = dataset_dir or os.path.join(self.project_dir, 'dataset')
            os.makedirs(dataset_dir, exist_ok=True)

            classes_path = os.path.join(dataset_dir, 'defect_classes.txt')
            with open(classes_path, 'w', encoding='utf-8') as f:
                for cls in self.categories:
                    f.write(f'{cls}\n')

            subclasses_path = os.path.join(dataset_dir, 'defect_subclasses.txt')
            with open(subclasses_path, 'w', encoding='utf-8') as f:
                for cls in self.categories:
                    f.write(f'{cls}\n')
                    children = self.sample_groups.get(cls, [])
                    if children:
                        for child in children:
                            f.write(f'  {child}\n')
                    else:
                        f.write('  无子分类\n')
        except Exception as e:
            if explicit_output:
                raise
            print(f"生成数据集说明文件失败: {e}")

    def run(self):
        try:
            i = 0
            dataset_dir = os.path.join(self.project_dir, "dataset")
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)

            if self.split_indices:
                train, val = self.split_indices
            else:
                train, val, _ = build_dataset_split(self.flag, self.mode, self.good)

            train = dedupe_indices_by_path(train, self.relative_paths)
            val = dedupe_indices_by_path(val, self.relative_paths)
            validate_split_no_overlap(self.relative_paths, train, val)
            count = len(train) + len(val)

            # 防御除以零
            if count == 0:
                self.workSingals.sendValue.emit(100)
                return

            manifest_rows = []
            dirs = [
                ('train', 'dataset/images/train', 'dataset/labels/train', train),
                ('val', 'dataset/images/val', 'dataset/labels/val', val)
            ]

            for split_name, img_folder, lbl_folder, file_indices in dirs:
                image_path = os.path.join(self.project_dir, img_folder)
                label_path = os.path.join(self.project_dir, lbl_folder)
                os.makedirs(image_path, exist_ok=True)
                os.makedirs(label_path, exist_ok=True)

                for x in file_indices:
                    try:
                        # 确保索引不越界
                        if x >= len(self.relative_paths):
                            continue

                        rel_path = self.relative_paths[x]
                        src = os.path.join(self.project_dir, rel_path)
                        # 防御幽灵文件：如果原图不存在，直接跳过
                        if not os.path.exists(src):
                            continue

                        name = os.path.basename(src)
                        dst = os.path.join(image_path, name)
                        shutil.copy(src, dst)

                        pre_name = os.path.splitext(name)[0]
                        json_file = os.path.join(self.project_dir, f"jsons/{pre_name}.json")
                        txt_file_path = os.path.join(label_path, f'{pre_name}.txt')

                        if x in self.good_set:
                            open(txt_file_path, 'w', encoding='utf-8').close()
                            continue

                        if not os.path.exists(json_file):
                            continue

                        # 防御中文乱码：强制 UTF-8 读取
                        with open(json_file, 'r', encoding='utf-8') as f:
                            content = json.load(f)

                        # 强制 UTF-8 写入
                        with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
                            label_line_index = 0
                            for ann_index, annotation in enumerate(content.get('annotations', []), start=1):
                                label_name = annotation.get('lable') or annotation.get('label') or annotation.get('category') or ''
                                if label_name not in self.categories:
                                    continue

                                label_idx = self.categories.index(label_name)
                                annotation_type = annotation.get('type')
                                normalized_points = self.get_normalized_points(annotation, content)
                                bbox = self.bbox_from_normalized_points(normalized_points)
                                if bbox is None:
                                    continue
                                subclass = self.get_subclass(rel_path, label_name)

                                if annotation_type == 'rect':
                                    cx, cy, w, h = self.bbox_to_xywh(bbox)
                                    txtfile.write(f'{label_idx} {cx} {cy} {w} {h}\n')
                                    label_line_index += 1
                                    manifest_rows.append(self.build_manifest_row(
                                        split_name,
                                        name,
                                        rel_path,
                                        pre_name,
                                        ann_index,
                                        label_line_index,
                                        label_name,
                                        subclass,
                                        label_idx,
                                        annotation_type,
                                        bbox,
                                    ))

                                elif annotation_type == 'polygon':
                                    if len(normalized_points) < 3:
                                        continue
                                    s = f'{label_idx} '
                                    for point_x, point_y in normalized_points:
                                        s += f"{point_x} {point_y} "
                                    txtfile.write(s.strip() + '\n')
                                    label_line_index += 1
                                    manifest_rows.append(self.build_manifest_row(
                                        split_name,
                                        name,
                                        rel_path,
                                        pre_name,
                                        ann_index,
                                        label_line_index,
                                        label_name,
                                        subclass,
                                        label_idx,
                                        annotation_type,
                                        bbox,
                                    ))

                    except Exception as item_error:
                        print(f"处理单张图片时发生跳过 (索引 {x}): {item_error}")
                    finally:
                        # 无论这部分成没成功，必须推动进度条！
                        i += 1
                        progress = int((i / count) * 100)
                        if self.augment_config.get("target_counts"):
                            progress = int(progress * 0.8)
                        # 限制进度条不超过 99%，给后续 ymal 文件留空间
                        self.workSingals.sendValue.emit(min(progress, 99))

            self.create_yolo_yaml()
            self.create_dataset_description_files()
            self.write_subclass_manifest(manifest_rows)

            target_counts = self.augment_config.get("target_counts")
            if target_counts:
                dataset_dir = os.path.join(self.project_dir, "dataset")
                output_root = self.augment_config.get("output_root") or os.path.join(self.project_dir, "dataset_aug")

                def on_augment_progress(value):
                    self.workSingals.sendValue.emit(80 + int(value * 0.19))

                DetectionAugment.augment_dataset(dataset_dir, output_root, target_counts, on_augment_progress)

        except Exception as e:
            print(f"数据集生成线程发生致命错误: {e}")
        finally:
            # 无论发生什么天灾人祸，必须让进度条满 100，解除界面的卡死状态
            self.workSingals.sendValue.emit(100)


class Data_Worker(QRunnable):
    def __init__(self, project_dir, relative_paths, tag, category, categorys):
        super().__init__()
        self.project_dir = project_dir
        self.tag = tag
        self.category = category
        self.relative_paths = relative_paths
        self.categorys = categorys

    def run(self):
        try:
            image_dir = os.path.join(self.project_dir, f'defect_{self.category}', 'images')
            label_dir = os.path.join(self.project_dir, f'defect_{self.category}', 'labels')
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            for x in self.tag:
                try:
                    if x >= len(self.relative_paths): continue
                    src = os.path.join(self.project_dir, self.relative_paths[x])
                    if not os.path.exists(src): continue

                    name = os.path.basename(src)
                    dst = os.path.join(image_dir, name)
                    shutil.copy(src, dst)

                    pre_name = os.path.splitext(name)[0]
                    json_file = os.path.join(self.project_dir, f"jsons/{pre_name}.json")
                    txt_file_path = os.path.join(label_dir, f'{pre_name}.txt')

                    if not os.path.exists(json_file): continue

                    # 强制 UTF-8 编码读取
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = json.load(f)

                    with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
                        for annotation in content.get('annotations', []):
                            label_name = annotation.get('lable', '')
                            if label_name not in self.categorys: continue

                            label_idx = self.categorys.index(label_name)

                            if annotation.get('type') == 'rect':
                                points = annotation.get('points', [])
                                if len(points) < 2: continue
                                x1, y1 = points[0]['x'], points[0]['y']
                                x2, y2 = points[1]['x'], points[1]['y']
                                cx = (x1 + x2) / 2
                                cy = (y1 + y2) / 2
                                w = abs(x1 - x2)
                                h = abs(y1 - y2)
                                txtfile.write(f'{label_idx} {cx} {cy} {w} {h}\n')

                            elif annotation.get("type") == "polygon":
                                points = annotation.get("points", [])
                                if len(points) < 3:
                                    continue
                                seg_str = " ".join([f"{p['x']:.6f} {p['y']:.6f}" for p in points])

                                # 使用 SAM 兼容的边界计算
                                xs = [p["x"] for p in points]
                                ys = [p["y"] for p in points]
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                cx = (min_x + max_x) / 2
                                cy = (min_y + max_y) / 2
                                w = max_x - min_x
                                h = max_y - min_y

                                txtfile.write(f"{label_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {seg_str}\n")
                except Exception as loop_e:
                    print(f"数据整理单项错误: {loop_e}")

        except Exception as main_e:
            print(f"数据整理线程致命错误: {main_e}")
