"""Read saved prediction facts without running a model or judging image pixels."""
import math
from collections import Counter, defaultdict
from pathlib import Path

from Utils.AIModelIdentity import model_label, recorded_model
from Utils.AIWorkspace import file_sha, read_json


def _batches(tools):
    root = tools.store.root / 'artifacts'
    if not root.is_dir():
        return []
    batches = []
    for path in root.iterdir():
        tools.check()
        if not path.is_dir():
            continue
        try:
            _, manifest = tools.store.artifact(path.name, 'candidates')
            batches.append(manifest)
        except (OSError, ValueError):
            continue
    return sorted(batches, key=lambda row: (row.get('created_at', ''), row['id']), reverse=True)


def _choice(store, manifest):
    metadata = recorded_model(store, manifest)
    return {'candidate_id': manifest['id'], 'title': manifest['title'],
            'created_at': manifest.get('created_at'), 'images': manifest.get('count'),
            'model': model_label(metadata), 'model_sha256': metadata.get('model_sha256'),
            'conf': metadata.get('conf'), 'imgsz': metadata.get('imgsz')}


def prediction_results(tools, candidate_id=None, model=None, offset=0, limit=20):
    from Utils.AIWorkTools import numeric
    offset = numeric(offset, 0, 10000000, '分页偏移', True)
    limit = numeric(limit, 1, 20, '分页数量', True)
    current = tools.context.get('viewing_artifact')
    if candidate_id is None and model is None and current:
        candidate_id = current
    if candidate_id == 'current':
        if not current:
            raise ValueError('当前未打开推理批次，请指定 candidate_id 或模型资源 model。')
        candidate_id = current
    digest = None
    if model is not None:
        resource = tools.resources.get(model)
        if not resource or resource.get('kind') != 'model':
            raise ValueError('model 必须引用已有模型资源编号；推理批次编号使用 candidate_id。')
        path = Path(resource['path'])
        if not path.is_file():
            raise ValueError('模型文件已不可用，无法核对该资源的当前版本；请用已保存批次的 candidate_id 分析历史结果。')
        before = path.stat()
        digest = file_sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('模型文件正在变化，无法可靠关联推理批次。')
    if candidate_id:
        folder, manifest = tools.store.artifact(candidate_id, 'candidates')
        if digest and recorded_model(tools.store, manifest).get('model_sha256') != digest:
            raise ValueError('指定批次与模型当前内容指纹不匹配，不能把其他模型或旧版本的结果当成本次结果。')
    else:
        batches = _batches(tools)
        matches = [row for row in batches if not digest or
                   recorded_model(tools.store, row).get('model_sha256') == digest]
        selected = next((row for row in matches if row['id'] == current), None)
        if selected is None and len(matches) == 1:
            selected = matches[0]
        if selected is None:
            return {'kind': 'prediction_summary', 'status': 'selection_required' if matches else 'not_found',
                    'model_resource': model, 'matching_batches': len(matches),
                    'batches': [_choice(tools.store, row) for row in matches[offset:offset + limit]],
                    'offset': offset, 'has_more': offset + limit < len(matches),
                    'note': ('多个推理批次，先请用户确定批次，不自动选择最新或合并。' if matches else
                             '没有找到该范围内已完成的推理批次；不转查训练 results.csv，不自动重新推理。'
                             '模型被替换或旧记录缺少指纹时，可明确指定历史 candidate_id 查询。')}
        candidate_id = selected['id']
        folder, manifest = tools.store.artifact(candidate_id, 'candidates')
    metadata = recorded_model(tools.store, manifest)
    rows = read_json(folder / 'candidates.json')
    if not isinstance(rows, list):
        raise ValueError('推理记录格式无效，不能计算统计。')
    counts, image_counts = Counter(), Counter()
    confidence = defaultdict(list)
    images, with_predictions, ignored, unmapped_images = [], 0, 0, 0
    for index, row in enumerate(rows):
        tools.check()
        annotations = row['document']['annotations']
        labels = Counter()
        for ann in annotations:
            label = ann.get('lable') or ann.get('label') or ann.get('category') or '未记录类别'
            labels[label] += 1
            score = ann.get('confidence')
            if type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1:
                confidence[label].append(score)
        counts.update(labels)
        image_counts.update(labels.keys())
        with_predictions += bool(annotations)
        ignored += row.get('ignored_classes', 0)
        unmapped_images += bool(row.get('unmapped_classes'))
        if offset <= index < offset + limit:
            images.append({'image': row['path'], 'predictions': len(annotations),
                           'classes': dict(labels), 'ignored_predictions': row.get('ignored_classes', 0)})
    classes = []
    for label, count in sorted(counts.items()):
        scores = confidence[label]
        classes.append({'label': label, 'predictions': count, 'images': image_counts[label],
                        'confidence_count': len(scores), 'confidence_missing': count - len(scores),
                        'confidence_min': min(scores) if scores else None,
                        'confidence_max': max(scores) if scores else None,
                        'confidence_mean': sum(scores) / len(scores) if scores else None})
    return {'kind': 'prediction_summary', 'status': 'ready', 'candidate_id': candidate_id,
            'title': manifest['title'], 'created_at': manifest.get('created_at'),
            'model': {'label': model_label(metadata), 'sha256': metadata.get('model_sha256'),
                      'recorded_resource': metadata.get('model_resource'), 'source': metadata.get('model_path')},
            'parameters': {key: metadata.get(key) for key in ('conf', 'imgsz', 'imgsz_source', 'class_map', 'class_filter')},
            'summary': {'successful_images': len(rows), 'failed_images': manifest.get('failed'),
                        'images_with_predictions': with_predictions, 'images_without_predictions': len(rows) - with_predictions,
                        'predictions': sum(counts.values()), 'ignored_predictions': ignored,
                        'legacy_unmapped_images': unmapped_images},
            'classes': classes, 'class_count': len(classes),
            'images': images, 'offset': offset, 'has_more': offset + limit < len(rows),
            'evidence': 'saved_prediction_records', 'image_pixels_sent': False,
            'note': '统计基于推理时保存的结果和参数，未重新预测、未看图；历史模型身份不读取今天的权重来改写。'
                    '无预测不等于良品；置信度不等于准确率，预测数量不能证明误检/漏检。'
                    '其他类别按项目要求正常过滤。需要与已有标注对照时使用 evaluate，结论依赖标注完整性。'}
