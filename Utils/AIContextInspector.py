"""Per-request evidence, captured before transport; never reconstruct old prompts."""
import base64
import copy
import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from Utils.AIWorkspace import Workspace, read_json, stamp, write_json

ACTIVE = ContextVar('ai_context_inspector', default=None)
SENSITIVE = {'apikey', 'authorization', 'password', 'secret', 'accesstoken', 'refreshtoken'}
SOURCES = {
    'project_state': '项目状态查询摘要（本轮开始时）',
    'active_models': '本次窗口推理模型编号、训练初始权重',
    'summary': '原图标注快照汇总（含当前画布覆盖）',
    'current_view': '发送时界面筛选/结果视图',
    'current_count': '本次原图范围数量', 'selected_count': '发送时选中图片数量',
    'current_result': '发送时查看的产物编号', 'current_result_count': '当前产物图片数量',
    'has_current_image': '发送时是否存在当前图片',
    'groups': '本轮工具可见的临时分组摘要',
    'resources': '项目及对话合并后的资源目录摘要',
    'latest_resource_id': '当前资源目录末项编号（不等于已确认模型）',
    'recent_artifacts': '最近产物索引，最多40条',
    'recent_workflows': '最近常用步骤索引，最多40条',
    'recent_transactions': '最近标注事务索引，最多40条',
    'recent_conversation': '调用方传入的对话历史，最多取最近12条',
    'request': '本轮用户原话', 'tool_results_this_turn': '本轮已执行工具结果，经compact压缩',
    'remaining_calls': '当前请求预算',
}


def sanitize(value, secrets=()):
    """No credentials or image bytes on disk, even inside text/model responses."""
    if isinstance(value, dict):
        return {sanitize(str(k), secrets): ('[敏感字段已隐藏]' if re.sub(r'[^a-z]', '', str(k).lower()) in SENSITIVE
                else sanitize(v, secrets)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, secrets) for v in value]
    if not isinstance(value, str):
        return value
    for secret in sorted({str(s) for s in secrets if s}, key=len, reverse=True):
        for form in {secret, quote(secret, safe=''), json.dumps(secret, ensure_ascii=True)[1:-1]}:
            value = value.replace(form, '[API Key 已隐藏]')
    value = re.sub(r'data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+', '[图片内容未保存]', value)
    value = re.sub(r'(?i)\bBearer\s+[A-Za-z0-9_.+/=-]+', 'Bearer [已隐藏]', value)
    value = re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b', '[API Key 已隐藏]', value)
    value = re.sub(r'(?i)([?&](?:api[_-]?key|token|access_token|key)=)[^&\s"<>]+', r'\1[已隐藏]', value)
    value = re.sub(r'''(?i)((?:api[_-]?key|access_token|password|authorization)[\\"'\s]*[:=][\\"'\s]*)([^\\"'\s,;}]+)''', r'\1[已隐藏]', value)
    return value


def endpoint_label(url):
    """Endpoint identity only; URL credentials/query strings never enter evidence."""
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc.rsplit('@', 1)[-1], parts.path, '', ''))
    except ValueError:
        return '[接口地址无法解析]'


def changed_fields(original, sent, path='tool_results_this_turn'):
    rows = []
    if isinstance(original, dict) and isinstance(sent, dict):
        for key, value in original.items():
            if key not in sent:
                rows.append(path + '.' + key + '：未注入')
            else:
                rows.extend(changed_fields(value, sent[key], path + '.' + key))
    elif isinstance(original, list) and isinstance(sent, list):
        for index, (a, b) in enumerate(zip(original[:20], sent[:20])):
            rows.extend(changed_fields(a, b, f'{path}[{index}]'))
        if len(original) > 20:
            rows.append(f'{path}：仅注入前20项及剩余数量，原{len(original)}项')
    elif original != sent:
        rows.append(path + '：已截断/摘要化')
    return rows


class ContextRecorder:
    def __init__(self, project, run_id, secret='', session_secrets=()):
        self.store = Workspace(project)
        self.folder = self.store.location('runs', run_id) / 'context_inspector'
        self.folder.resolve().relative_to(self.store.project)
        self.secrets = tuple(session_secrets) + (secret,)
        self.warnings = []
        self.current = None
        index_file = self.folder / 'index.json'
        if index_file.exists():
            try:
                self.index = read_json(index_file)
                if (self.index.get('version') != 1 or self.index.get('run_id') != run_id or
                        not all(isinstance(self.index.get(key), list) and
                                all(type(n) is int and n > 0 for n in self.index[key]) for key in ('requests', 'events'))):
                    raise ValueError('上下文记录版本不支持')
            except (OSError, ValueError, TypeError, AttributeError):
                # Never overwrite older/corrupt evidence on resume.
                self.index = None
                self.warnings.append('上下文记录无法续写，原记录已保留。')
                return
        else:
            self.index = {'version': 1, 'run_id': run_id, 'created_at': stamp(), 'status': 'running',
                          'requests': [], 'events': [], 'warnings': []}

    def _write(self, name, value):
        if self.index is None:
            return
        try:
            target = self.folder / name
            target.resolve().relative_to(self.folder.resolve())
            write_json(target, sanitize(value, self.secrets))
        except (OSError, ValueError, TypeError) as exc:
            warning = '上下文快照未完整保存：' + sanitize(str(exc), self.secrets)
            if warning not in self.warnings:
                self.warnings.append(warning)

    def save_current(self):
        if self.current is not None:
            self._write(f"request_{self.current['number']:04d}.json", self.current)
        if self.index is not None:
            self.index['warnings'] = list(self.warnings)
            self._write('index.json', self.index)

    def begin(self, number, public, prompt, images, known, observations, history_count):
        if self.index is None:
            return
        if number in self.index['requests'] or (self.folder / f'request_{number:04d}.json').exists():
            self.current = None
            self.warnings.append(f'第{number}轮快照已存在，未覆盖。')
            return
        warnings = []
        if public.get('project_state', {}).get('unavailable'):
            warnings.append('项目状态读取失败：' + public['project_state']['unavailable'])
        unavailable = [str(k) for k, v in public.get('resources', {}).items() if not v.get('available')]
        if unavailable:
            warnings.append('不可用资源：' + '、'.join(unavailable))
        if not public.get('active_models', {}).get('local_inference_resource'):
            warnings.append('未注入本次窗口实际加载的推理模型编号；已添加模型不等于已选用。')
        warnings.append('资源目录未自动读取每个.pt的训练类别；不能仅凭已添加模型判断适配性。')
        omissions = changed_fields(observations, public.get('tool_results_this_turn', []))
        if history_count > 12:
            omissions.append(f'recent_conversation：仅注入最近12条，传入{history_count}条')
        self.current = copy.deepcopy({'number': number, 'created_at': stamp(), 'status': 'prepared',
            'injected_context': public, 'prompt': prompt, 'known_local': known,
            'requested_images': [str(p) for p in images], 'field_sources': SOURCES,
            'omissions': omissions, 'warnings': warnings, 'transports': [],
            'evidence_note': '准备快照不代表已发送；实际提交证据见transports。图片仅保存字节指纹与大小，不保存base64。'})
        self.index['requests'].append(number)
        self.index['status'] = 'running'
        self.save_current()

    @contextmanager
    def capture(self):
        token = ACTIVE.set(self)
        try:
            yield
        finally:
            ACTIVE.reset(token)

    def transport(self, data, endpoint, image_paths, attempt):
        if self.current is None:
            return
        payload = json.loads(data.decode('utf-8'))
        attachments = []
        for message in payload.get('messages', []):
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get('type') != 'image_url':
                    continue
                url = part.get('image_url', {}).get('url', '')
                if url.startswith('data:') and ';base64,' in url:
                    header, encoded = url.split(';base64,', 1)
                    blob = base64.b64decode(encoded)
                    image = {'mime': header[5:], 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(),
                             'path': str(image_paths[len(attachments)]) if len(attachments) < len(image_paths) else None}
                    attachments.append(image)
                    part['image_url']['url'] = {'omitted': '图片base64未保存', **image}
        self.current['transports'].append({'attempt': attempt, 'submitted_at': stamp(),
            'status': 'attempted', 'endpoint': endpoint_label(endpoint), 'body_bytes': len(data),
            'body_sha256': hashlib.sha256(data).hexdigest(), 'payload': payload, 'images': attachments})
        self.current['status'] = 'attempted'
        self.save_current()

    def transport_outcome(self, status, detail=None):
        if self.current is not None and self.current['transports']:
            self.current['transports'][-1].update(status=status, finished_at=stamp(), detail=detail)
            self.save_current()

    def finish_request(self, result=None, response=None, error=None):
        if self.current is None:
            return
        self.current.update(status='failed' if error else 'returned', finished_at=stamp(),
                            response={'parsed': result, 'provider_response': response}, error=error)
        self.save_current()

    def event(self, observation, request_number=None, origin='model'):
        if self.index is None:
            return
        number = len(self.index['events']) + 1
        event = {'number': number, 'created_at': stamp(), 'after_request': request_number,
                 'origin': origin, 'observation': copy.deepcopy(observation),
                 'note': '本地执行结果；是否发送及压缩范围，以后续请求injected_context为准。'}
        self._write(f'event_{number:04d}.json', event)
        self.index['events'].append(number)
        self.save_current()

    def finish(self, status):
        if self.index is not None:
            self.index.update(status=status, finished_at=stamp())
        self.save_current()


def capture_transport(data, endpoint, paths, attempt):
    recorder = ACTIVE.get()
    if recorder is not None:
        try:
            recorder.transport(data, endpoint, paths, attempt)
        except Exception as exc:
            recorder.warnings.append('请求传输快照捕获失败：' + sanitize(str(exc), recorder.secrets))


def capture_transport_outcome(status, detail=None):
    recorder = ACTIVE.get()
    if recorder is not None:
        recorder.transport_outcome(status, detail)


def load_inspection(project, run_id):
    store = Workspace(project)
    folder = store.location('runs', run_id) / 'context_inspector'
    folder.resolve().relative_to(store.project)
    index_file = folder / 'index.json'
    if not index_file.exists():
        return {'missing': True, 'run_id': run_id, 'note': '本次没有上下文快照（旧记录或捕获未完成），不能重建当时发送内容。'}
    index = read_json(index_file)
    if index.get('version') != 1 or index.get('run_id') != run_id:
        raise ValueError('上下文快照版本或任务编号不匹配，原文件保留。')
    value = {'index': index, 'requests': [], 'events': [], 'read_warnings': []}
    for kind in ('requests', 'events'):
        for number in index[kind]:
            if type(number) is not int or number < 1:
                raise ValueError('上下文记录索引无效。')
            path = folder / f'{kind[:-1]}_{number:04d}.json'
            path.resolve().relative_to(folder.resolve())
            try:
                entry = read_json(path)
                if not isinstance(entry, dict) or entry.get('number') != number:
                    raise ValueError('快照编号或结构无效')
                if kind == 'requests':
                    if (not isinstance(entry.get('injected_context'), dict) or
                            not isinstance(entry.get('known_local'), dict) or
                            not isinstance(entry.get('prompt'), str) or
                            not isinstance(entry.get('transports'), list) or
                            any(not isinstance(t, dict) or 'status' not in t for t in entry['transports'])):
                        raise ValueError('请求快照结构无效')
                elif not isinstance(entry.get('observation'), dict):
                    raise ValueError('工具记录结构无效')
                value[kind].append(entry)
            except (OSError, ValueError, TypeError) as exc:
                value['read_warnings'].append(f'{path.name} 读取失败：{exc}')
    return value


def share_copy(value, project):
    """Export strips local paths; business text/file names still require user review."""
    paths = {str(Path(project).resolve()): '[项目目录]', str(Path.home()): '[用户目录]'}
    for request in value.get('requests', []):
        for resource in request.get('known_local', {}).get('resources', {}).values():
            if resource.get('path'):
                paths[resource['path']] = f"[资源/{Path(resource['path']).name}]"
        for path in request.get('requested_images', []):
            paths[path] = f'[图片/{Path(path).name}]'
    encoded = json.dumps(sanitize(value), ensure_ascii=False)
    for path, replacement in sorted(paths.items(), key=lambda item: len(item[0]), reverse=True):
        for variant in {path, path.replace('\\', '/'), path.replace('/', '\\')}:
            # Values may also be embedded inside the serialized prompt.
            for form in (json.dumps(json.dumps(variant, ensure_ascii=False)[1:-1], ensure_ascii=False)[1:-1],
                         json.dumps(variant, ensure_ascii=False)[1:-1]):
                encoded = encoded.replace(form, replacement)
    result = json.loads(encoded)
    def remove_absolute_paths(value):
        if isinstance(value, dict):
            return {remove_absolute_paths(k): remove_absolute_paths(v) for k, v in value.items()}
        if isinstance(value, list):
            return [remove_absolute_paths(v) for v in value]
        if isinstance(value, str):
            value = re.sub(r'(?i)\b[a-z]:[\\/]+[^"\r\n<>|]*', '[外部路径]', value)
            value = re.sub(r'(?<![:\w/])/(?:[^/\s"<>]+/)+[^\s"<>]*', '[外部路径]', value)
            value = re.sub(r'\\\\[^"\r\n<>|]+', '[外部路径]', value)
        return value
    result = remove_absolute_paths(result)
    result['export_note'] = '密钥和已知本地路径已脱敏，图片内容未导出；文件名、类别及业务文字仍保留，请分享前检查。'
    return result
