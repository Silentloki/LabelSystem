"""Reuse Route 1 evidence, chart validation and report rendering in chat."""
import json
import re
import shutil
from pathlib import Path

from Utils import AITrainingAnalysis as analysis
from Utils import TrainingReport
from Utils.AIWorkspace import file_sha, write_json


def prepare(folder, run):
    images = []
    charts = []
    for chart in run['charts']:
        source = Path(chart['path']).resolve()
        source.relative_to(Path(run['folder']).resolve())
        target = folder / 'charts' / chart['filename']
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha(target) != run['hashes'][chart['filename']]:
            raise ValueError('训练图表在读取期间发生变化，请重新分析。')
        images.append(str(target))
        charts.append({'image': target.relative_to(folder).as_posix(), 'name': chart['filename'],
                       'source': run['name'], 'sha256': file_sha(target)})
    write_json(folder / 'charts.json', charts)
    evidence = analysis.prepare_evidence(run)
    write_json(folder / 'evidence.json', evidence)
    return {'folder': folder, 'run': run, 'evidence': evidence, 'images': images, 'charts': charts}


def prompt(bundle):
    return ('\n本轮已读取训练结果，训练分析是用户已授权的工作，请直接分析随附图表，不再询问是否查看。'
            '最终回复必须使用 {"action":"reply","message":"分析完成","analysis":报告JSON}；'
            'analysis必须符合以下已有训练分析报告协议，程序将校验并展示。'
            '不要编造命令行参数，不要建议 --save-class-metrics（本工具未提供此选项）。\n'
            + analysis.build_prompt(bundle['evidence']))


def image_paths(bundle):
    for path, chart in zip(bundle['images'], bundle['charts']):
        if file_sha(path) != chart['sha256']:
            raise ValueError('待分析图表已改变，请重新读取训练结果。')
    return bundle['images']


def complete(bundle, value):
    report = analysis.validate_report(value, bundle['evidence'], bundle['run']['args'])
    if '--save-class-metrics' in json.dumps(report, ensure_ascii=False):
        raise ValueError('分析包含未提供的训练参数，请根据现有图表给出可执行建议，不要求虚构参数。')
    for before, direction, after in re.findall(
            r'从\s*([0-9]+(?:\.[0-9]+)?)\s*(上升|下降|升|降)(?:至|到|为)\s*([0-9]+(?:\.[0-9]+)?)',
            json.dumps(report, ensure_ascii=False)):
        delta = float(after) - float(before)
        if (direction in ('上升', '升') and delta <= 0) or (direction in ('下降', '降') and delta >= 0):
            raise ValueError('数值变化方向写反，请按程序relations纠正后返回报告。')
    envelope = {'status': 'completed', 'result': report}
    folder, run = bundle['folder'], bundle['run']
    write_json(folder / 'analysis.json', envelope)
    (folder / 'report.html').write_text(TrainingReport.render_html(run, None, envelope), encoding='utf-8')
    lines = [run['context']['display_name'] + ' · 训练分析', report['summary']['text'], '', '各类表现：']
    for row in TrainingReport.class_rows(run, report.get('chart_readings', [])):
        score = f" AP50：{row['ap50']:.3f}。" if row.get('ap50') is not None else ''
        lines.append(row['name'] + '：' + TrainingReport.class_message(row) + score)
    for item in report.get('chart_readings', []):
        if not item['readable']:
            lines.append('图表未读清：' + item.get('reason', '无法可靠读取'))
    for item in report['class_findings']:
        lines.append(item['name'] + '建议：' + item['text'])
    lines += ['', '最佳轮次到最后一轮（程序核对）：']
    for metric, values in TrainingReport.training_relations(run)['best_to_last'].items():
        lines.append(f"{metric}：{values['from']:.5g} → {values['to']:.5g}（{values['direction']}）")
    lines += [item['text'] for item in report['findings']]
    next_step = report['next_experiment']
    lines += ['', '下一步：' + next_step['title'], next_step['action'],
              '观察：' + next_step['observe'], '停止条件：' + next_step['stop'],
              '图表读数可在左侧临时结果核对；验证结果不等于生产线表现。']
    return '\n'.join(lines)
