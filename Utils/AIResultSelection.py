"""Persistent subsets reference a completed data artifact; never copy images."""
from Utils.AIWorkspace import read_json, write_json, sha, json_bytes
from Utils.AIResultBrowser import artifact_file


def selection_source(store, artifact_id):
    folder, manifest = store.artifact(artifact_id, 'selection')
    value = read_json(artifact_file(folder, 'selection.json'))
    parent = value['source_artifact']
    _, source = store.artifact(parent)
    if source['kind'] not in {'crops', 'dataset', 'converted', 'candidates', 'error_set'}:
        raise ValueError('筛选来源不是可读取的数据产物。')
    images = value['images']
    if (not isinstance(images, list) or any(not isinstance(p, str) for p in images)
            or len(set(images)) != len(images) or len(images) != manifest.get('count')):
        raise ValueError('结果筛选引用清单无效。')
    return parent, images


def create_selection(tools, parent, images, title, criteria):
    signature = sha(json_bytes([parent, images, title, criteria]))
    for entry in tools.store.list_entries(limit=1000):
        if entry['kind'] != 'selection' or entry['status'] != 'completed':
            continue
        folder, manifest = tools.store.artifact(entry['id'])
        if manifest.get('metadata', {}).get('selection_signature') == signature:
            return {'artifact_id': entry['id'], 'kind': 'selection', 'title': title,
                    'count': len(images), 'source_artifact': parent, 'reused_result': True}
    if not images:
        return {'kind': 'selection', 'title': title, 'count': 0, 'note': '此范围没有符合条件的结果图片。'}
    _, folder, manifest = tools.store.create('selection', title, {
        'criteria': criteria, 'selection_signature': signature, 'source_artifact': parent})
    write_json(folder / 'selection.json', {'source_artifact': parent, 'images': images})
    return tools.store.finish(folder, manifest, count=len(images), source_artifact=parent,
                              note='处理结果筛选快照，只保存引用，不复制图片。')
