"""Publish deliberate exports inside the project; keep work records separate."""
import re
import shutil
from pathlib import Path

from Utils.AIWorkspace import read_json, write_json


CATEGORIES = {'dataset', 'images', 'labels', 'annotated', 'predictions'}


def export_root(store, manifest):
    category = manifest.get('export_category')
    if category not in CATEGORIES:
        raise ValueError('无效导出分类。')
    aid = manifest['id']
    store.location('artifacts', aid)
    fmt = manifest.get('format', manifest.get('metadata', {}).get('format'))
    content = manifest.get('content', manifest.get('metadata', {}).get('content'))
    expected = ('annotated' if fmt == 'native' else 'dataset') if manifest.get('kind') == 'dataset' else (
        'predictions' if content in {'preview', 'predictions'} else content)
    if expected != category or manifest.get('kind') not in {'dataset', 'export'}:
        raise ValueError('导出分类与记录内容不一致。')
    if category == 'dataset':
        if not aid.startswith('YOLO数据集_'):
            raise ValueError('数据集批次标识无效。')
        batch = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?$', aid)
        if not batch:
            raise ValueError('无效数据集批次名称。')
        relative = 'dataset_AI_' + batch.group()
    else:
        relative = 'exports/' + category + '/' + aid
    path = store.project / relative
    # Never follow a junction or symlink, including one on exports/category.
    from Utils.AIResultStorage import _checked_path
    for part in (path, *path.parents):
        if part == store.project:
            break
        if part.exists() or part.is_symlink():
            _checked_path(part, store.project)
    if manifest.get('export_relative') not in (None, relative):
        raise ValueError('导出目录与所属批次不一致。')
    return path


def result_file(store, folder, manifest, row, field):
    from Utils.AIResultBrowser import artifact_file
    external = row.get('export_' + field)
    if external is not None:
        return artifact_file(export_root(store, manifest), external)
    return artifact_file(folder, row[field])


def publish_dataset(store, folder, manifest, result):
    """Called before completion. Only this new artifact's own files are moved."""
    fmt = result.get('format', manifest['metadata'].get('format'))
    category = 'annotated' if fmt == 'native' else 'dataset'
    manifest['export_category'] = category
    target = export_root(store, manifest)
    # Independent batches never merge, including pre-existing orphan exports.
    if target.exists():
        raise ValueError('导出批次目录已存在，未覆盖；请重新导出。')
    rows = read_json(folder / 'sources.json')
    if fmt == 'native':
        staged = folder / 'export_payload'
        staged.mkdir()
        (folder / 'images').rename(staged / 'images')
        shutil.copytree(folder / 'jsons', staged / 'jsons')
        for row in rows:
            row.update(export_image=row['image'], export_json=row['json'])
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(target)
    else:
        source = folder / ('dataset_AI' if fmt == 'yolo' else 'yolo')
        old = str(source)
        source.rename(target)
        for row in rows:
            row['export_image'] = (Path(row['image']).relative_to('dataset_AI').as_posix()
                                   if fmt == 'yolo' else 'images/' + row['split'] + '/' + Path(row['image']).name)
            if fmt != 'yolo':
                # The custom writer staged a second image copy; remove only it.
                staged = folder / row['image']
                staged.resolve().relative_to(folder.resolve())
                staged.unlink()
        if fmt != 'yolo':
            (folder / 'images').rmdir()
        result.update(dataset_dir=str(target), data_yaml=str(target / 'data.yaml'))
        result['note'] = result.get('note', '').replace(old, str(target))
    manifest['export_relative'] = target.relative_to(store.project).as_posix()
    write_json(folder / 'sources.json', rows)
    result.update(export_dir=str(target), export_relative=manifest['export_relative'], export_category=category)
    return result
