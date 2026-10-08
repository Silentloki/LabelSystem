"""Chat export using the desktop dataset split and metadata writers."""
import copy
import random
import shutil
from pathlib import Path

from Utils.Task import Worker, build_subsample_stratified_split
from Utils.AIWorkspace import file_sha, read_json, write_json


def export_dataset(tools, paths, seed, rows=None, input_context=None):
    context = tools.context if input_context is None else input_context
    flags = context.get('flags', {})
    selected = [p for p in paths if p not in flags or flags[p] in (1, 2, 3)]
    selected_set = set(selected)
    rows = tools._rows(selected) if rows is None else [r for r in rows if r['source'] in selected_set]
    if not rows:
        raise ValueError('此范围没有已标注缺陷或已确认良品，无法生成数据集。')
    stems = [Path(p).stem.lower() for p in selected]
    if len(stems) != len(set(stems)):
        raise ValueError('范围中存在同名图片或同名不同扩展名，请先处理重名，避免标签覆盖。')
    tree_path = tools.store.source('sample_tree.json')
    tree = read_json(tree_path) if tree_path.exists() else {}
    assignments = context.get('sample_assignments', tree.get('assignments', {}))
    groups = context.get('sample_groups', tree.get('groups', {}))
    label_path = tools.store.source('label.txt')
    classes = list(context.get('classes', []))
    if not classes and label_path.exists():
        classes = label_path.read_text(encoding='utf-8-sig').splitlines()
    classes = list(dict.fromkeys(c for c in classes if c))
    good = [i for i, row in enumerate(rows) if flags.get(row['source']) in (2, 3)]
    good_set = set(good)
    labels = {}
    for i, row in enumerate(rows):
        row['document'] = copy.deepcopy(row['document'])
        if i in good_set:
            row['document']['annotations'] = []
        labels[i] = set()
        for ann in row['document']['annotations']:
            name = ann.get('lable') or ann.get('label') or ann.get('category')
            labels[i].add(name)
            if name not in classes:
                classes.append(name)
    bad = [i for i in range(len(rows)) if i not in good_set]
    if input_context is not None:
        buckets = {}
        for i, row in enumerate(rows):
            buckets.setdefault(row['split_group'], []).append(i)
        grouped = list(buckets.values())
        representatives = [selected[indices[0]] for indices in grouped]
        group_labels = {i: set().union(*(labels[j] for j in indices)) for i, indices in enumerate(grouped)}
        group_good = [i for i in group_labels if not group_labels[i]]
        group_bad = [i for i in group_labels if group_labels[i]]
        train_groups, val_groups, _ = build_subsample_stratified_split(
            group_bad, 2, representatives, classes, assignments, group_good, group_labels, rng=random.Random(seed))
        train = [j for i in train_groups for j in grouped[i]]
        val = [j for i in val_groups for j in grouped[i]]
        count = len(train) + len(val)
    else:
        train, val, count = build_subsample_stratified_split(
            bad, 2, selected, classes, assignments, good, labels, rng=random.Random(seed))
    writer = Worker(str(tools.store.project), selected, bad, classes, 2,
                    good=good, sample_groups=groups, sample_assignments=assignments)
    _, folder, manifest = tools.store.create('dataset', 'YOLO数据集（AI）',
        {'format': 'yolo', 'split_policy': 'desktop_subsample_stratified', 'seed': seed,
         'skipped_unreviewed': len(paths) - len(selected),
         'source_artifact': context.get('source_artifact'), 'group_by_source': input_context is not None})
    dataset = folder / 'dataset_AI'
    (folder / 'jsons').mkdir()
    manifest_rows = []
    for split, indices in [('train', train), ('val', val)]:
        images, annotations = dataset / 'images' / split, dataset / 'labels' / split
        images.mkdir(parents=True)
        annotations.mkdir(parents=True)
        for i in indices:
            tools.check()
            row = rows[i]
            source = tools.store.source(row['source'])
            dest = images / source.name
            shutil.copyfile(source, dest)
            if file_sha(dest) != row['image_hash']:
                raise ValueError('复制期间原图片发生变化，产物未完成。')
            lines = []
            doc = row['document']
            for ann_index, ann in enumerate(doc['annotations'], 1):
                name = ann.get('lable') or ann.get('label') or ann.get('category')
                idx = classes.index(name)
                points = writer.get_normalized_points(ann, doc)
                bbox = writer.bbox_from_normalized_points(points)
                if bbox is None:
                    raise ValueError('无法导出零面积标注：' + row['source'])
                if ann['type'] == 'rect':
                    values = writer.bbox_to_xywh(bbox)
                elif ann['type'] == 'polygon' and len(points) >= 3:
                    values = [v for point in points for v in point]
                else:
                    raise ValueError('不支持的标注类型：' + row['source'])
                lines.append(str(idx) + ' ' + ' '.join(str(v) for v in values))
                manifest_rows.append(writer.build_manifest_row(
                    split, source.name, row['source'], source.stem, ann_index, len(lines),
                    name, writer.get_subclass(row['source'], name), idx, ann['type'], bbox))
            (annotations / (source.stem + '.txt')).write_text(
                '\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            row.update(split=split, image=dest.relative_to(folder).as_posix(),
                       json='jsons/' + source.stem + '.json')
            write_json(folder / row['json'], doc)
    tools.check()
    writer.create_yolo_yaml(str(dataset))
    writer.create_dataset_description_files(str(dataset))
    writer.write_subclass_manifest(manifest_rows, str(dataset))
    write_json(folder / 'sources.json', rows)
    return tools.store.finish_export(folder, manifest, count=count, classes=classes,
        splits={'train': len(train), 'val': len(val)}, dataset_dir=str(dataset),
        data_yaml=str(dataset / 'data.yaml'), skipped_unreviewed=len(paths) - len(selected), format='yolo',
        source_artifact=context.get('source_artifact'),
        note=f'已按软件规则导出YOLO数据集：训练{len(train)}张，验证{len(val)}张。'
             f'缺陷按类别/子分类分层，验证集目标约20%；良品训练占66%。'
             f'未标注/未确认样本跳过{len(paths) - len(selected)}张。'
             + ('同源切片整体划分，避免同一原图跨训练集/验证集。' if input_context is not None else '')
             + f'数据集目录：{dataset}')
