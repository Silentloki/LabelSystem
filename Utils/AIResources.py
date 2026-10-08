"""Typed, read-only inventory of user-added resources (not artifact IDs)."""
from pathlib import Path


def training_folders(directory, limit=40):
    """Recognize one run or a bounded collection within the selected directory."""
    root = Path(directory).resolve()
    if not root.is_dir():
        return []
    found, pending, visited = [], [(root, 0)], 0
    while pending and len(found) < limit and visited < 500:
        folder, depth = pending.pop(0)
        visited += 1
        try:
            folder.resolve().relative_to(root)
            csv = folder / 'results.csv'
            csv.resolve().relative_to(root)
            if csv.is_file():
                found.append(folder)
                continue
            if depth < 2:
                for child in sorted(folder.iterdir()):
                    if child.is_dir() and not child.is_symlink():
                        pending.append((child, depth + 1))
                    if len(pending) + visited >= 500:
                        break
        except (OSError, ValueError):
            continue
    return found


def resource_catalog(resources):
    catalog = {}
    for key, resource in resources.items():
        row = {'id': key, 'name': resource.get('name', ''), 'kind': resource.get('kind')}
        path = Path(resource['path'])
        try:
            row['available'] = path.is_file() if row['kind'] == 'model' else path.is_dir()
            if row['kind'] == 'directory':
                folders = training_folders(path) if row['available'] else []
                row['training_runs'] = len(folders)
                row['role'] = 'training_results' if folders else 'directory'
                row['description'] = (f'可读取{len(folders)}个训练结果；training_runs参数resource使用{key}' if folders
                                      else '此目录及两层子目录未找到results.csv' if row['available'] else '目录不存在或不可访问')
            else:
                row['role'] = 'local_model'
                from Utils.AIModelIdentity import model_source
                row['source'] = model_source(str(path), row['name'])
                row['description'] = f'本地推理模型；分析已保存推理结果用prediction_results(model="{key}")，不是训练结果目录'
        except OSError as exc:
            row.update(available=False, error=str(exc)[:300])
        catalog[key] = row
    return catalog


def resource_choices(resources, kind=None):
    return '；'.join(f"{key}：{value.get('name', key)}（{value.get('kind', '')}）"
                    for key, value in resources.items() if kind is None or value.get('kind') == kind)
