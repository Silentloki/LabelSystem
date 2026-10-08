"""Read generated images and their own labels without touching original samples."""
import copy
from pathlib import Path

from PIL import Image

from Utils.AIChatSelection import annotation_features
from Utils.AIResultBrowser import artifact_file
from Utils.AIWorkspace import file_sha, read_json, read_project_list
from Utils.AIExportStorage import result_file


def export_rows(tools, artifact_id, selected=None, allow_predictions=False):
    rows, context = _export_rows(tools, artifact_id)
    if selected is not None:
        by_path = {tools.store.source_key(row['source']): row for row in rows}
        keys = [tools.store.source_key(p) for p in selected]
        if not keys or len(set(keys)) != len(keys) or any(p not in by_path for p in keys):
            raise ValueError('选中的结果图片不属于当前结果，不能扩大范围导出。')
        rows = [by_path[p] for p in keys]
    if not allow_predictions and any(row.get('prediction') for row in rows):
        raise ValueError('预测结果尚未采用，不能当正式标注切图或导出训练数据；可导出图片、预测记录或带框预览。')
    if not allow_predictions and any(row.get('unannotated') for row in rows):
        raise ValueError('范围内有未保存标注的原图，不能将其当作良品导出或切图；可先仅导出图片。')
    return rows, context


def _export_rows(tools, artifact_id):
    folder, manifest = tools.store.artifact(artifact_id)
    if manifest['kind'] == 'selection':
        from Utils.AIResultSelection import selection_source
        parent, images = selection_source(tools.store, artifact_id)
        rows, context = export_rows(tools, parent, images, allow_predictions=True)
        context.update(source_artifact=artifact_id, base_artifact=parent)
        return rows, context
    if manifest['kind'] in {'candidates', 'error_set'}:
        from Utils.AIResultBrowser import visual_rows
        _, values = visual_rows(tools.store, artifact_id)
        context = copy.deepcopy(tools.context)
        context.update(flags={}, sample_assignments={}, source_artifact=artifact_id, base_artifact=artifact_id)
        tree_file = tools.store.project / 'sample_tree.json'
        tree = read_json(tree_file) if tree_file.exists() else {}
        assignments = {tools.store.source_key(p): v for p, v in
                       tools.context.get('sample_assignments', tree.get('assignments', {})).items()}
        context['sample_groups'] = tools.context.get('sample_groups', tree.get('groups', {}))
        paths = read_project_list(tools.store.project / 'datafile.dat')
        flags = read_project_list(tools.store.project / 'flagfile.dat')
        original_flags = {tools.store.source_key(p): f for p, f in zip(paths, flags)}
        rows, seen = [], set()
        for item in values:
            tools.check()
            if item.get('error') or not item.get('image'):
                continue  # Failed inference rows have no prediction document.
            image = tools.store.source(str(Path(item['image']).relative_to(tools.store.project)))
            relative = image.relative_to(tools.store.project).as_posix()
            key = tools.store.source_key(relative)
            if key in seen:
                raise ValueError('处理结果清单包含重复图片。')
            seen.add(key)
            expected = file_sha(image)
            if item.get('hash') and item['hash'] != expected:
                raise ValueError('图片在生成结果后已变化，请重新核对。')
            doc = copy.deepcopy(item['document']) if 'document' in item else read_json(item['annotation'])
            with Image.open(image) as im:
                width, height = im.size
            if (doc.get('image_width'), doc.get('image_height')) != (width, height):
                raise ValueError('结果图片与标注尺寸不一致。')
            for ann in doc['annotations']:
                annotation_features(ann, width, height)
            if file_sha(image) != expected:
                raise ValueError('读取期间图片发生变化。')
            predicted = item.get('predictions') is not None
            rows.append({'source': relative, 'document': doc, 'image_hash': expected,
                         'prediction': predicted, 'original_source': item.get('original_source') or item.get('source') or relative,
                         'unannotated': not predicted and bool(item.get('annotation')) and not Path(item['annotation']).exists(),
                         'split_group': item.get('split_group') or expected})
            context['flags'][relative] = original_flags.get(key, 1 if doc['annotations'] else 2)
            origin = rows[-1]['original_source']
            context['sample_assignments'][relative] = copy.deepcopy(assignments.get(tools.store.source_key(origin), {}))
        context['failed_images'] = sum(bool(v.get('error')) for v in values)
        return rows, context
    if manifest['kind'] not in {'crops', 'converted', 'dataset'}:
        raise ValueError('此产物不支持图片与标注导出。')
    sources = read_json(artifact_file(folder, 'sources.json'))
    if not isinstance(sources, list) or not sources:
        raise ValueError('处理结果中没有可导出的图片。')
    if len(sources) != manifest.get('count'):
        raise ValueError('处理结果清单数量不一致，请检查结果文件。')
    context = copy.deepcopy(tools.context)
    context.update(flags={}, sample_assignments={}, source_artifact=artifact_id, base_artifact=artifact_id)
    tree_path = tools.store.source('sample_tree.json')
    tree = read_json(tree_path) if tree_path.exists() else {}
    assignments = tools.context.get('sample_assignments', tree.get('assignments', {}))
    assignments = {str(key).replace('\\', '/'): value for key, value in assignments.items()}
    context['sample_groups'] = tools.context.get('sample_groups', tree.get('groups', {}))
    rows, seen = [], set()
    for item in sources:
        tools.check()
        image = result_file(tools.store, folder, manifest, item, 'image')
        annotation = result_file(tools.store, folder, manifest, item, 'json')
        relative = image.relative_to(tools.store.project).as_posix()
        if relative in seen:
            raise ValueError('处理结果清单包含重复图片。')
        seen.add(relative)
        expected = file_sha(image)
        doc = read_json(annotation)
        if not isinstance(doc, dict) or not isinstance(doc.get('annotations'), list):
            raise ValueError('处理结果标注无效：' + image.name)
        with Image.open(image) as im:
            width, height = im.size
        if (doc.get('image_width'), doc.get('image_height')) != (width, height):
            raise ValueError('处理结果图片与标注尺寸不一致：' + image.name)
        for ann in doc['annotations']:
            annotation_features(ann, width, height)
        if file_sha(image) != expected:
            raise ValueError('读取期间结果图片发生变化。')
        origin = item.get('original_source', item.get('source', relative))
        # Crop image_hash belongs to the ORIGINAL image, not the saved tile.
        # Keep sibling tiles together when splitting a training dataset.
        group = item.get('split_group') or (item.get('image_hash') if manifest['kind'] == 'crops' else None) or expected
        rows.append({'source': relative, 'document': doc, 'image_hash': expected,
                     'original_source': origin, 'split_group': group})
        context['flags'][relative] = 1 if doc['annotations'] else 2
        context['sample_assignments'][relative] = copy.deepcopy(assignments.get(str(origin).replace('\\', '/'), {}))
    return rows, context
