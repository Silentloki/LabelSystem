"""Model labels use recorded inference identity, never today's weight file."""
from pathlib import PurePosixPath


def model_source(path, name='本地模型'):
    path = PurePosixPath(str(path).replace('\\', '/'))
    if not path.name:
        return name
    parent = path.parent
    if parent.name.lower() == 'weights':
        parent = parent.parent
    return f'{parent.name}/{path.name}' if parent.name else path.name


def model_label(metadata):
    source = model_source(metadata.get('model_path', ''), metadata.get('model', '本地模型'))
    digest = metadata.get('model_sha256', '')
    # Put the version first so narrow result trees still distinguish best.pt files.
    return f'{digest[:8]} · {source}' if digest else source


def recorded_model(store, manifest):
    metadata = dict(manifest.get('metadata', {}))
    if metadata.get('plan_id') and not metadata.get('model_path'):
        try:
            _, plan = store.artifact(metadata['plan_id'], 'prediction_plan')
            previous = plan['metadata']
            if not metadata.get('model_sha256') or metadata['model_sha256'] == previous.get('model_sha256'):
                for key in ('model_path', 'model_resource', 'model_sha256'):
                    if key not in metadata and key in previous:
                        metadata[key] = previous[key]
        except (OSError, ValueError, KeyError):
            pass
    return metadata


def model_details(metadata):
    rows = ['模型：' + model_label(metadata)]
    for key, label in (('model_path', '来源'), ('model_sha256', '完整版本指纹')):
        if metadata.get(key):
            rows.append(label + '：' + str(metadata[key]))
    return '\n'.join(rows)
