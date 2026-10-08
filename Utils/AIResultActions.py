"""Explicit UI targets, human instructions and reference-only problem sets."""
import copy
import re
from pathlib import Path

from Utils.AIWorkspace import (file_sha, read_json, read_project_list, write_json,
                               json_bytes, sha)


def ui_target(host):
    from PyQt5.QtCore import Qt
    artifact = host.current_tree_filter[1] if host.current_tree_filter[0] == 'artifact' else None
    browser = host.result_browser
    widget = browser.list if artifact else host.listWidget
    selected = widget.selectedItems()
    current = widget.currentItem()
    items = selected or ([current] if current else [])
    paths, invalid = [], False
    for item in items:
        if artifact:
            index = item.data(Qt.UserRole)
            if browser.active_id != artifact or not isinstance(index, int) or not 0 <= index < len(browser.rows):
                invalid = True
                continue
            row = browser.rows[index]
            if not row.get('image') or row.get('error') or row.get('chart'):
                invalid = True
                continue
            try:
                path = Path(row['image']).resolve().relative_to(Path(host.project_dir).resolve()).as_posix()
            except ValueError:
                invalid = True
                continue
        else:
            path = item.toolTip()
        paths.append(path)
    return {'artifact_id': artifact, 'images': list(dict.fromkeys(paths)),
            'mode': 'selected' if selected else 'current_image', 'invalid': invalid,
            'count': len(items)}


def target_rows(tools):
    """Never fall back to a whole batch or the hidden original selection."""
    from Utils.AIResultBrowser import visual_rows
    target = tools.context.get('interaction_target')
    if target is None:
        artifact = tools.context.get('viewing_artifact')
        images = tools.context.get('selected_result_paths' if artifact else 'selected_paths', [])
        if not images and not artifact and tools.context.get('current_image'):
            images = [tools.context['current_image']]
        target = {'artifact_id': artifact, 'images': images}
    if target.get('invalid') or not target.get('images'):
        raise ValueError('当前对象为空或包含失败/不可用图片，请重新选择；不会改用整批或原图。')
    artifact, images = target.get('artifact_id'), target['images']
    if artifact != tools.context.get('viewing_artifact'):
        raise ValueError('图片与处理批次不一致，请重新选择。')
    if artifact:
        manifest, rows = visual_rows(tools.store, artifact)
        by_path = {Path(r['image']).relative_to(tools.store.project).as_posix(): r
                   for r in rows if r.get('image')}
        if any(p not in by_path for p in images):
            raise ValueError('选中图片不属于指定结果，不能扩大范围。')
        rows = [copy.deepcopy(by_path[p]) for p in images]
    else:
        if any(p not in tools.known for p in images):
            raise ValueError('选中图片已不在工程中。')
        rows = [{'image': str(tools.store.source(p)), 'source': p,
                 'annotation': str(tools.store.source('jsons/' + Path(p).stem + '.json')),
                 'document': tools.document(p, missing_ok=True)} for p in images]
    for row, path in zip(rows, images):
        tools.check()
        if row.get('error') or row.get('chart'):
            raise ValueError('选中项不是可处理的样本图片。')
        digest = file_sha(tools.store.source(path))
        if row.get('hash') and row['hash'] != digest:
            raise ValueError('图片在生成结果后已变化，不能使用旧结果。')
        row.update(image_path=path, hash=digest)
        row.setdefault('source_artifact', artifact)
        if 'document' not in row:
            row['document'] = read_json(row['annotation'])
    return rows


def signature(row):
    return sha(json_bytes([row['hash'], row['document'], row.get('source'), row.get('source_artifact')]))


def error_set_rows(store, artifact_id):
    from Utils.AIResultBrowser import visual_rows, artifact_file
    folder, manifest = store.artifact(artifact_id, 'error_set')
    entries = read_json(artifact_file(folder, 'references.json'))
    if not isinstance(entries, list) or len(entries) != manifest.get('count'):
        raise ValueError('问题样本集引用数量不一致。')
    rows, sources, seen = [], {}, set()
    for entry in entries:
        path, source = entry['image'], entry.get('source_artifact')
        if (source, path) in seen:
            raise ValueError('问题样本集包含重复引用。')
        seen.add((source, path))
        if source:
            if source not in sources:
                _, parent = store.artifact(source)
                if parent['kind'] not in {'candidates', 'crops', 'dataset', 'converted', 'selection'}:
                    raise ValueError('问题样本集来源不支持，不能递归引用。')
                _, values = visual_rows(store, source)
                sources[source] = {Path(r['image']).relative_to(store.project).as_posix(): r
                                   for r in values if r.get('image')}
            if path not in sources[source]:
                raise ValueError('问题样本集来源已缺失。')
            row = copy.deepcopy(sources[source][path])
        else:
            if path not in read_project_list(store.project / 'datafile.dat'):
                raise ValueError('问题样本图片已不在工程中。')
            annotation = store.source('jsons/' + Path(path).stem + '.json')
            if annotation.exists() != entry.get('annotation_exists', True):
                raise ValueError('问题样本的标注文件状态已变化，请重新核对。')
            row = {'image': str(store.source(path)), 'name': Path(path).name, 'source': path,
                   'annotation': str(annotation), 'document': read_json(annotation) if annotation.exists() else entry['document']}
        row['source_artifact'] = source
        row['hash'] = file_sha(store.source(path))
        if 'document' not in row:
            row['document'] = read_json(row['annotation'])
        if signature(row) != entry['signature']:
            raise ValueError('问题样本的图片或结果已变化，请从原批次重新核对。')
        rows.append(row)
    return rows


def add_error_set(tools, title='问题样本集'):
    message = tools.context.get('user_request', '')
    if (not re.search(r'加入|添加|放入|放到|建立|创建|整理|收集', message)
            or not re.search(r'error\s*set|问题(?:样本)?集', message, re.I)
            or re.search(r'不要|别|暂不|先不|是否|能否|如果|[?？]', message)):
        raise ValueError('仅按用户明确的加入问题样本集指令保存；分析疑点不会自动入集。')
    if not isinstance(title, str) or not title.strip() or len(title) > 80:
        raise ValueError('问题样本集名称须为1到80个字。')
    rows = target_rows(tools)
    entries = [{'image': r['image_path'], 'source_artifact': r['source_artifact'],
                'annotation_exists': Path(r['annotation']).exists() if not r['source_artifact'] else None,
                'signature': signature(r), 'document': r['document'] if not r['source_artifact'] else None}
               for r in rows]
    # Canvas edits have not become a durable source yet.
    for row in rows:
        if not row['source_artifact'] and row['image_path'] in tools.overrides:
            disk = read_json(row['annotation']) if Path(row['annotation']).exists() else None
            if disk != row['document']:
                raise ValueError('当前图片有未保存标注，请保存后再加入问题样本集。')
    token = sha(json_bytes([entries, title, message]))
    for item in tools.store.list_entries(limit=1000):
        if item['kind'] == 'error_set' and item['status'] == 'completed':
            _, existing = tools.store.artifact(item['id'])
            if existing.get('metadata', {}).get('signature') == token:
                error_set_rows(tools.store, item['id'])
                return {'artifact_id': item['id'], 'kind': 'error_set', 'title': title,
                        'count': len(rows), 'reused_result': True}
    _, folder, manifest = tools.store.create('error_set', title, {
        'signature': token, 'user_instruction': message, 'run_id': tools.run_id,
        'source_artifacts': list(dict.fromkeys(r['source_artifact'] for r in rows if r['source_artifact']))})
    write_json(folder / 'references.json', entries)
    return tools.store.finish(folder, manifest, count=len(rows),
        note='仅保存图片与结果引用、用户原话和来源；没有复制图片或修改标注。')


def record_feedback(tools, category):
    from Utils.ProjectContext import ProjectContext, ISSUES
    message = tools.context.get('user_request', '')
    names = {'false_positive': '误检', 'false_negative': '漏检', 'image_quality': '图片质量问题',
             'annotation_question': '标注待核对', 'correct': '正常'}
    # Only direct recording instructions are eligible, never an AI finding,
    # conditional instruction, negation or a question about possible errors.
    if (category not in names or not re.search(r'(?:记为|标记为|记录为|记成)\s*[“"\']?' + re.escape(names.get(category, 'INVALID')), message)
            or names[category] not in message
            or re.search(r'不要|别|暂不|先不|是否|能否|如果|可能|疑似|有没有|不是|不算|不正常|[?？]', message)):
        raise ValueError('尚无明确人工记录指令。请直接说“把选中的图片记为漏检”等；AI分析不能代替人工判断。')
    mentioned = [k for k, name in names.items() if name in message]
    if len(mentioned) != 1:
        raise ValueError('本条指令含多种判断，请分别选择图片并记录，不能把同一判断套给所有图片。')
    rows = target_rows(tools)
    context = ProjectContext(tools.store.project)
    with context.db() as db:
        ids = []
        for row in rows:
            tools.check()
            ids.append(context.add_feedback(row['image_path'], category, message,
                row['source_artifact'], _db=db, provenance={'via': 'chat_user_instruction',
                                                         'run_id': tools.run_id}))
    return {'kind': 'human_feedback', 'title': '人工记录 · ' + ISSUES[category],
            'count': len(ids), 'feedback_ids': ids,
            'source_artifact': tools.context.get('viewing_artifact'),
            'note': '已保存用户本次明确判断，可在项目状态查看或标为已处理；没有修改标注。'}
