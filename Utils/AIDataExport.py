"""Exports with independent scope and content, without changing source data."""
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from Utils.AIWorkspace import file_sha, write_json, read_json
from Utils.AIExportStorage import export_root


def export_data(tools, scope, content):
    predictions = content in {'predictions', 'preview'}
    source = tools.export_scope(scope, allow_predictions=predictions or content == 'images')
    aid = source.get('artifact_id')
    rows = []
    if aid:
        from Utils.AIArtifactExport import export_rows
        rows, _ = export_rows(tools, aid, source.get('selected'), allow_predictions=predictions or content == 'images')
        if predictions and any(not row.get('prediction') for row in rows):
            raise ValueError('请选择模型推理结果；不能把正式标注当作预测导出。')
    else:
        if predictions:
            raise ValueError('导出预测结果或带预测框图片，请指定已有模型推理结果。')
        for path in source['paths']:
            tools.check()
            expected = file_sha(tools.store.source(path))
            # Images-only export also accepts unannotated samples.
            doc = None if content == 'images' else tools.document(path)
            if file_sha(tools.store.source(path)) != expected:
                raise ValueError('读取期间图片发生变化。')
            rows.append({'source': path, 'document': doc, 'image_hash': expected})
    if not rows:
        raise ValueError('此范围没有可导出的数据。')
    titles = {'images': '仅图片', 'labels': '仅标注', 'predictions': '预测记录', 'preview': '带预测框图片'}
    category = 'predictions' if predictions else content
    _, folder, manifest = tools.store.create('export', titles[content], {'content': content, 'source_artifact': aid})
    manifest['export_category'] = category
    target = export_root(tools.store, manifest)
    if target.exists():
        raise ValueError('导出批次已存在，未覆盖；请重新导出。')
    payload = folder / 'export_payload'
    payload.mkdir()
    manifest['export_relative'] = target.relative_to(tools.store.project).as_posix()
    write_json(folder / 'manifest.json', manifest)
    used = set()
    for index, row in enumerate(rows, 1):
        tools.check()
        image = tools.store.source(row['source'])
        if file_sha(image) != row['image_hash']:
            raise ValueError('来源图片已变化，请重新处理后导出。')
        stem = image.stem
        if stem.casefold() in used:
            stem = f'{index:06d}_' + stem
        while stem.casefold() in used:
            stem = '_' + stem
        used.add(stem.casefold())
        if content == 'images':
            dest = payload / (stem + image.suffix.lower())
            shutil.copyfile(image, dest)
            if file_sha(dest) != row['image_hash']:
                raise ValueError('复制期间图片发生变化，导出未完成。')
        elif content == 'labels':
            dest = payload / (stem + '.json')
            write_json(dest, row['document'])
        elif content == 'predictions':
            dest = payload / (stem + '.json')
            write_json(dest, {'kind': 'predictions', 'source': row['source'],
                              'image_sha256': row['image_hash'], 'adopted': False,
                              'document': row['document']})
        else:
            dest = payload / (stem + '.png')
            render_predictions(image, row['document'], dest)
        if file_sha(image) != row['image_hash']:
            raise ValueError('导出期间来源图片发生变化，导出未完成。')
        row['export_file'] = dest.name
    write_json(folder / 'sources.json', rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload.rename(target)
    return tools.store.finish(folder, manifest, count=len(rows), content=content,
        source_artifact=aid, export_dir=str(target), export_relative=manifest['export_relative'],
        export_category=category, note=f'已导出{len(rows)}项{titles[content]}，未划分训练集/验证集。')


def render_predictions(source, document, target):
    with Image.open(source) as original:
        image = original.convert('RGB')
    if image.size != (document.get('image_width'), document.get('image_height')):
        raise ValueError('预测结果与图片尺寸不一致。')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    size = max(12, min(40, round(min(width, height) / 40)))
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()
    for ann in document['annotations']:
        from Utils.AIChatSelection import annotation_features
        annotation_features(ann, width, height)
        points = [(p['x'] * width, p['y'] * height) for p in ann['points']]
        if ann['type'] == 'rect':
            xs, ys = zip(*points)
            draw.rectangle((min(xs), min(ys), max(xs), max(ys)), outline='#00ff70', width=max(2, size // 8))
        else:
            draw.line(points + points[:1], fill='#00ff70', width=max(2, size // 8))
        label = ann.get('lable') or ann.get('label') or ann.get('category') or ''
        if 'confidence' in ann:
            label += f" {ann['confidence']:.1%}"
        draw.text((max(0, min(p[0] for p in points)), max(0, min(p[1] for p in points) - size - 3)),
                  label, fill='#00ff70', font=font, stroke_width=1, stroke_fill='black')
    image.save(target)
