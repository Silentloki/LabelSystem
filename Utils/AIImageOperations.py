"""Shared action scopes and recoverable project sample-state changes."""
import copy
import json
import pickle
from pathlib import Path

from Utils.AIWorkspace import read_project_list, json_bytes, document_token, file_sha, sha


def original_scope(tools, scope='selected', trace_sources=False):
    source = tools.export_scope(scope, allow_predictions=True)
    if 'paths' in source:
        return source['paths']
    from Utils.AIArtifactExport import export_rows
    rows, context = export_rows(tools, source['artifact_id'], source.get('selected'), allow_predictions=True)
    if context.get('failed_images') and source.get('selected') is None:
        raise ValueError('此批次包含失败图片，请明确选择有效照片，不能静默跳过失败项后修改整批。')
    known = {tools.store.source_key(p): p for p in tools.paths}
    images = []
    for row in rows:
        key = tools.store.source_key(row['source'])
        if key not in known and trace_sources:
            original = row.get('original_source')
            if original:
                key = tools.store.source_key(original)
        if key not in known:
            raise ValueError('当前对象包含切图或派生图片，不能直接改工程原图状态/标注；如要处理来源原图，请明确要求定位来源原图后再操作。')
        images.append(known[key])
    if not images:
        raise ValueError('此范围没有图片，不会回退全工程。')
    return list(dict.fromkeys(images))


def set_sample_status(tools, status, scope='selected'):
    names = {'good': '完全良品', 'overkill': '过杀品', 'clear_good': '清除良品/过杀品状态'}
    if status not in names:
        raise ValueError('状态仅支持good（完全良品）、overkill（过杀品）、clear_good（清除良品/过杀品状态）。')
    paths = original_scope(tools, scope)
    if not paths:
        raise ValueError('没有选中照片，未生成修改方案。')
    store = tools.store
    data_file, flags_file = store.project / 'datafile.dat', store.project / 'flagfile.dat'
    data_bytes, flags_bytes = data_file.read_bytes(), flags_file.read_bytes()
    project_paths = read_project_list(data_file, data_bytes)
    flags = read_project_list(flags_file, flags_bytes)
    if len(flags) != len(project_paths) or any(type(f) is not int or f not in (0, 1, 2, 3) for f in flags):
        raise ValueError('工程图片列表与状态不一致，请先重新打开工程核对。')
    index = {store.source_key(p): i for i, p in enumerate(project_paths)}
    if len(index) != len(project_paths):
        raise ValueError('工程包含重复图片引用，不能可靠修改状态。')
    tree_path = store.project / 'sample_tree.json'
    tree_bytes = tree_path.read_bytes() if tree_path.exists() else None
    tree = json.loads(tree_bytes.decode('utf-8-sig')) if tree_bytes is not None else {'groups': {}, 'assignments': {}}
    original_tree = copy.deepcopy(tree)
    assignments = tree.setdefault('assignments', {})
    changes, tokens, preview, affected, before_flags, after_flags, guards, canvas = {}, {}, [], [], {}, {}, {}, {}
    for path in paths:
        tools.check()
        key = store.source_key(path)
        if key not in index:
            raise ValueError('工程图片列表已变化，请重新选择。')
        i = index[key]
        previous = flags[i]
        if status == 'clear_good' and previous not in (2, 3):
            continue
        annotation = 'jsons/' + Path(path).stem + '.json'
        target = store.source(annotation)
        initial = target.read_bytes() if target.exists() else None
        doc = tools.document(path, missing_ok=True)
        canvas[path] = copy.deepcopy(doc)
        count = len(doc['annotations'])
        removed = [p for p in assignments if store.source_key(p) == key] if status != 'clear_good' else []
        subclass_count = sum(len(assignments[p]) for p in removed)
        new_flag = {'good': 2, 'overkill': 3, 'clear_good': 0}[status]
        if previous == new_flag and not count and not removed:
            continue
        if status != 'clear_good':
            doc['annotations'] = []
            changes[annotation] = json_bytes(doc)
            tokens[annotation] = document_token(target, initial)
            for p in removed:
                assignments.pop(p)
        flags[i] = new_flag
        affected.append(path)
        before_flags[path], after_flags[path] = previous, new_flag
        guards[path] = file_sha(store.source(path))
        preview.append({'image': path, 'action': ('设为' + names[status] if status != 'clear_good' else '清除良品/过杀品状态，恢复未标注状态') +
            (f'；清除 {count} 处缺陷标注、{subclass_count} 项子样本归类' if status != 'clear_good' else '；保留图片和标注内容'),
            'old_status': previous, 'new_status': new_flag})
    if not affected:
        return {'kind': 'sample_status', 'status': 'unchanged', 'count': 0,
                'note': '选中图片已是目标状态，或没有需要清除的良品/过杀品状态；没有修改。'}
    changes['flagfile.dat'] = pickle.dumps(flags)
    tokens['flagfile.dat'] = document_token(flags_file, flags_bytes)
    if tree != original_tree:
        changes['sample_tree.json'] = json_bytes(tree)
        tokens['sample_tree.json'] = document_token(tree_path, tree_bytes)
    pending = store.prepare_transaction(names[status] + f' · {len(affected)} 张', changes,
        {'changes': preview, 'affected_images': affected, 'set_flags': after_flags, 'previous_flags': before_flags,
         'expected_tokens': tokens, 'image_hashes': guards, 'state_hashes': {'datafile.dat': sha(data_bytes)},
         'canvas_inputs': canvas, 'source_artifact': tools.context.get('viewing_artifact'),
         'operation': 'set_sample_status', 'status': status})
    return {'pending': {'kind': 'transaction', **pending}, 'images': len(affected),
            'note': '修改的是工程原图的正式状态及预览中列出的标注；已有推理批次保留。确认后执行，可恢复。'}


def available_actions(tools, scope='selected'):
    source = tools.export_scope(scope, allow_predictions=True)
    if source.get('artifact_id'):
        from Utils.AIArtifactExport import export_rows
        rows, _ = export_rows(tools, source['artifact_id'], source.get('selected'), allow_predictions=True)
        predicted = any(r.get('prediction') for r in rows)
    else:
        rows = [{'source': p, 'unannotated': not tools.store.source('jsons/' + Path(p).stem + '.json').exists()
                 and p not in tools.overrides} for p in source['paths']]
        predicted = False
    known = {tools.store.source_key(p) for p in tools.paths}
    original = bool(rows) and all(tools.store.source_key(r['source']) in known for r in rows)
    originals_available = bool(rows) and all(tools.store.source_key(r.get('original_source', r['source'])) in known for r in rows)
    formal = bool(rows) and not predicted and not any(r.get('unannotated') for r in rows)
    return {'count': len(rows), 'source_artifact': source.get('artifact_id'),
            'actions': {'inspect_selection': True, 'record_feedback': True, 'add_error_set': True,
                        'select': True, 'stats': True, 'export_images': True,
                        'export_predictions_or_preview': bool(rows) and all(r.get('prediction') for r in rows),
                        'export_dataset_or_annotations': formal, 'crop': formal,
                        'set_sample_status': original, 'relabel': original,
                        'preannotate': original, 'snapshot': original,
                        'duplicates': original, 'similar': original, 'original_images': originals_available},
            'notes': ['设为良品/过杀品会清除原图缺陷标注与子样本归类，先预览确认并支持恢复。',
                      '预测正常的人工记录与样本良品状态不是同一操作。',
                      '未采用的预测只能导出图片/预测记录/带框预览；可明确定位原图后使用其正式标注切图或导出。',
                      '派生图没有工程样本状态，不直接将其状态套给来源原图。',
                      '不支持直接修改Mask、删除原图或启动训练；不要替换成记录问题，也不要承诺已执行。']}
