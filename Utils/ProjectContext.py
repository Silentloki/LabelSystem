"""Project-owned resource registry, derived inventory and human feedback.

Existing files remain authoritative. The SQLite inventory is a query cache;
resources and human decisions are durable records, never inferred by a model.
"""
import copy
import hashlib
import json
import sqlite3
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path

from Utils.AIWorkspace import Workspace, document_token, file_sha, identifier, read_json, read_project_list, stamp


ISSUES = {'false_positive': '误检', 'false_negative': '漏检', 'image_quality': '图片质量问题',
          'annotation_question': '标注待核对', 'correct': '已人工核对正常'}


class ProjectContext:
    def __init__(self, project):
        self.store = Workspace(project)
        self.project = self.store.project
        self.path = self.store.source('ai_workbench/project_context.sqlite')

    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('项目状态数据库版本较新，原记录已保留，请使用匹配版本的软件。')
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, signature TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, image TEXT NOT NULL, artifact TEXT NOT NULL,
                    category TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS feedback_lookup ON feedback(category, artifact, created_at);
                PRAGMA user_version = 1;
            ''')
            with connection:
                yield connection
        finally:
            connection.close()

    def location(self, path):
        path = Path(path).resolve()
        try:
            return {'project_relative': path.relative_to(self.project).as_posix()}
        except ValueError:
            return {'path': str(path)}

    def resolve(self, value):
        return self.store.source(value['project_relative']) if 'project_relative' in value else Path(value['path']).resolve()

    def resource_id(self, path, kind):
        data = json.dumps([kind, self.location(path)], sort_keys=True).encode()
        return 'p' + hashlib.sha256(data).hexdigest()[:20]

    def register_resource(self, path, kind, role=None):
        if kind not in {'model', 'directory'}:
            raise ValueError('项目资源类型无效。')
        path = Path(path).resolve()
        key = self.resource_id(path, kind)
        payload = {'id': key, 'name': path.name, 'kind': kind, **self.location(path)}
        if role:
            payload['role'] = role
        with self.db() as db:
            old = db.execute('SELECT payload FROM resources WHERE id=?', (key,)).fetchone()
            previous = json.loads(old[0]) if old else {}
            payload = {**previous, **payload, 'registered_at': previous.get('registered_at', stamp())}
            db.execute('INSERT OR REPLACE INTO resources VALUES (?,?)', (key, json.dumps(payload, ensure_ascii=False)))
        return key

    def resources(self):
        with self.db() as db:
            rows = db.execute('SELECT payload FROM resources ORDER BY rowid').fetchall()
        result = {}
        for row in rows:
            value = json.loads(row[0])
            path = self.resolve(value)
            result[value['id']] = {**value, 'path': str(path)}
        return result

    def merge_resources(self, local):
        """Keep historical r1 bindings; expose other resources through stable project IDs."""
        result = copy.deepcopy(local)
        seen = set()
        for value in local.values():
            key = self.register_resource(value['path'], value['kind'], value.get('role'))
            seen.add(key)
        for key, value in self.resources().items():
            if key not in seen and key not in result:
                result[key] = value
        return result

    def set_model(self, path, role='last_loaded_inference'):
        if role not in {'last_loaded_inference', 'preferred_model'}:
            raise ValueError('模型用途无效。')
        if not Path(path).is_file():
            raise ValueError('模型文件不存在。')
        key = self.register_resource(path, 'model', 'local_model')
        state = Path(path).stat()
        self.set_setting(role, {'resource_id': key, 'updated_at': stamp(), 'sha256': file_sha(path),
                               'file_state': [state.st_size, state.st_mtime_ns]})
        return key

    def set_setting(self, name, value):
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (name, json.dumps(value, ensure_ascii=False)))

    def settings(self):
        with self.db() as db:
            return {r['name']: json.loads(r['payload']) for r in db.execute('SELECT * FROM settings')}

    def _cached(self, key, kind, files, build):
        signature = json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.exists()
                                else (str(p), None) for p in files])
        with self.db() as db:
            row = db.execute('SELECT signature,payload FROM entities WHERE id=?', (key,)).fetchone()
        if row and row['signature'] == signature:
            return json.loads(row['payload'])
        value = build()
        value.update(id=key, kind=kind)
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO entities VALUES (?,?,?,?)',
                       (key, kind, signature, json.dumps(value, ensure_ascii=False)))
        return value

    def artifact(self, aid):
        folder = self.store.location('artifacts', aid)
        manifest_path = folder / 'manifest.json'
        def build():
            m = read_json(manifest_path)
            metadata = m.get('metadata') or {}
            source = m.get('source_artifact') or metadata.get('source_artifact')
            return {'artifact_id': aid, 'title': m.get('title', aid), 'artifact_kind': m.get('kind'),
                    'created_at': m.get('created_at'), 'finished_at': m.get('finished_at'),
                    'status': m.get('status'), 'hidden': bool(m.get('hidden')), 'count': m.get('count'),
                    'source_artifact': source, 'plan_id': metadata.get('plan_id'),
                    'export_relative': m.get('export_relative'),
                    'parameters': {k: metadata[k] for k in ('mode', 'tile_size', 'overlap', 'padding', 'label',
                        'model', 'model_resource', 'model_path', 'model_sha256', 'conf', 'imgsz', 'format',
                        'content', 'seed', 'resource', 'criteria') if k in metadata},
                    'source_count': len(metadata['paths']) if isinstance(metadata.get('paths'), list) else m.get('source_count'),
                    'evidence': f'ai_workbench/artifacts/{aid}/manifest.json'}
        value = self._cached('artifact:' + aid, 'result', [manifest_path], build)
        value['available'] = value['status'] == 'completed'
        if value.get('export_relative'):
            value['available'] &= self.store.source(value['export_relative']).is_dir()
        if value.get('source_artifact'):
            try:
                self.store.artifact(value['source_artifact'])
            except (OSError, ValueError):
                value['available'] = False
        return value

    def dataset(self, yaml_path):
        import yaml
        yaml_path = Path(yaml_path).resolve()
        key = 'dataset:' + hashlib.sha256(json.dumps(self.location(yaml_path), sort_keys=True).encode()).hexdigest()[:20]
        def build():
            if yaml_path.stat().st_size > 1024 * 1024:
                raise ValueError('数据配置文件超过 1 MB。')
            data = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict):
                raise ValueError('数据配置格式无效。')
            root = Path(data.get('path') or yaml_path.parent)
            if not root.is_absolute():
                root = yaml_path.parent / root
            splits = {}
            for split in ('train', 'val', 'test'):
                raw = data.get(split)
                entries = raw if isinstance(raw, list) else [raw] if isinstance(raw, str) else []
                splits[split] = [self.location(root / entry) for entry in entries if isinstance(entry, str)]
            return {'title': yaml_path.parent.name, 'config': self.location(yaml_path), 'classes': data.get('names', []),
                    'splits': splits, 'path_basis': 'data.yaml 所在目录解析相对 path；保留声明，缺失不猜测',
                    'evidence': self.location(yaml_path)}
        value = self._cached(key, 'dataset', [yaml_path], build)
        value['split_status'] = {name: [{'location': entry, 'available': self.resolve(entry).exists()}
                                      for entry in entries] for name, entries in value['splits'].items()}
        return value

    def experiment(self, folder):
        from Utils.AITrainingAnalysis import load_run, RECORD
        folder = Path(folder).resolve()
        resource_id = self.register_resource(folder, 'directory', 'training_results')
        def build():
            run = load_run(folder)
            record = run['record']
            comparable = [record.get('validation_fingerprint'), record.get('names'), record.get('evaluation_config')]
            group = (hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()
                     if all(comparable) and record.get('status') == 'completed' else None)
            data_path = run['args'].get('data')
            return {'title': folder.name, 'resource_id': resource_id, 'folder': self.location(folder),
                    'created_at': record.get('started_at'), 'finished_at': record.get('finished_at'),
                    'status': record.get('status', '历史记录'), 'primary_metric': run['primary'], 'best': run['best'],
                    'last': run['last'], 'comparison_group': group, 'training_data': data_path,
                    'weights': [self.location(folder / 'weights' / name) for name in ('best.pt', 'last.pt')
                                if (folder / 'weights' / name).is_file()],
                    'evidence': self.location(folder / 'results.csv'),
                    'metric_basis': 'CSV 中主指标最高轮次，不等于最终权重再次评估；不可跨验证集直接排名'}
        files = [folder / name for name in ('results.csv', 'args.yaml', RECORD, 'weights/best.pt', 'weights/last.pt')]
        value = self._cached('experiment:' + resource_id, 'experiment', files, build)
        for weight in value['weights']:
            self.register_resource(self.resolve(weight), 'model', 'local_model')
        return value

    def inventory(self, paths=None, flags=None, classes=None, cancelled=lambda: False):
        """Reconcile small metadata files; never crawl arbitrary machine directories or image bytes."""
        from Utils.AIResources import training_folders
        warnings, datasets, experiments, results = [], [], [], []
        if paths is None:
            path = self.project / 'datafile.dat'
            paths = read_project_list(path) if path.is_file() else []
        if flags is None:
            path = self.project / 'flagfile.dat'
            flags = read_project_list(path) if path.is_file() else []
        if classes is None:
            path = self.project / 'label.txt'
            classes = path.read_text(encoding='utf-8-sig').splitlines() if path.is_file() else []
        configs = set()
        folders = set()
        # Only project-owned conventional locations and deliberately associated resources.
        for child in self.project.iterdir():
            if child.is_dir() and (child.name in {'dataset', 'dataset_aug'} or child.name.startswith('dataset_AI_')):
                if (child / 'data.yaml').is_file():
                    configs.add(child / 'data.yaml')
        if (self.project / 'runs').is_dir():
            folders.update(training_folders(self.project / 'runs'))
        index = self.project / 'ai_training_analysis/runs.json'
        if index.is_file():
            try:
                folders.update(Path(p) for p in read_json(index).get('runs', []))
            except (OSError, ValueError, TypeError) as exc:
                warnings.append('训练索引未能读取：' + str(exc))
        for value in list(self.resources().values()):
            path = Path(value['path'])
            if value['kind'] == 'directory' and path.is_dir():
                folders.update(training_folders(path))
                if (path / 'data.yaml').is_file():
                    configs.add(path / 'data.yaml')
        artifacts = self.store.root / 'artifacts'
        if artifacts.is_dir():
            for folder in artifacts.iterdir():
                if cancelled():
                    raise ValueError('已停止读取项目状态。')
                if not folder.is_dir():
                    continue
                try:
                    value = self.artifact(folder.name)
                    results.append(value)
                    if value.get('export_relative') and value['available']:
                        yaml_path = self.store.source(value['export_relative']) / 'data.yaml'
                        if yaml_path.is_file():
                            configs.add(yaml_path)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    warnings.append('结果记录未能读取：' + folder.name + '：' + str(exc)[:150])
        for folder in sorted(folders):
            if cancelled():
                raise ValueError('已停止读取项目状态。')
            try:
                experiment = self.experiment(folder)
                experiments.append(experiment)
                source = experiment.get('training_data')
                if isinstance(source, str) and Path(source).is_absolute() and Path(source).is_file():
                    configs.add(Path(source))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                warnings.append('实验不可用：' + folder.name + '：' + str(exc)[:150])
        for config in sorted(configs):
            try:
                datasets.append(self.dataset(config))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                warnings.append('数据集不可用：' + str(config) + '：' + str(exc)[:150])
        resources = self.resources()
        dataset_ids = {str(self.resolve(d['config'])): d['id'] for d in datasets}
        for experiment in experiments:
            experiment['dataset_id'] = dataset_ids.get(experiment.get('training_data'))
            for weight in experiment['weights']:
                key = self.resource_id(self.resolve(weight), 'model')
                if key in resources:
                    resources[key]['produced_by'] = experiment['id']
        for value in resources.values():
            p = Path(value['path'])
            value['available'] = p.is_file() if value['kind'] == 'model' else p.is_dir()
        preferences = self.settings()
        for name in ('preferred_model', 'last_loaded_inference'):
            if name in preferences:
                value = preferences[name]
                resource = resources.get(value.get('resource_id'))
                value['available'] = bool(resource and resource['available'])
                if value['available']:
                    stat = Path(resource['path']).stat()
                    value['changed_since_selected'] = value.get('file_state') != [stat.st_size, stat.st_mtime_ns]
                value['meaning'] = '人工选定的项目主用模型' if name == 'preferred_model' else '上次在本地模型入口加载；不代表已自动加载到本次窗口'
        results.sort(key=lambda x: x.get('created_at') or '', reverse=True)
        experiments.sort(key=lambda x: x.get('created_at') or '', reverse=True)
        groups = {}
        for experiment in experiments:
            if experiment['comparison_group'] and experiment.get('best'):
                groups.setdefault((experiment['comparison_group'], experiment['primary_metric']), []).append(experiment)
        comparisons = []
        for (group, metric), runs in groups.items():
            if len(runs) < 2:
                continue
            best_value = max(r['best'][metric] for r in runs)
            comparisons.append({'group': group, 'metric': metric, 'experiments': [r['id'] for r in runs],
                                'highest': [r['id'] for r in runs if r['best'][metric] == best_value], 'value': best_value,
                                'basis': '同验证集、类别、评估条件；比较各次CSV最高轮次，不等于业务最佳模型'})
        summary = {'project_name': self.project.name, 'as_of': stamp(), 'images': len(paths),
                   'classes': classes, 'sample_states': {'unlabeled': flags.count(0), 'bad': flags.count(1),
                                                       'good': flags.count(2), 'overkill': flags.count(3)},
                   'locations': {'images': 'images/', 'annotations': 'jsons/',
                                 'masks': list(dict.fromkeys(p.resolve().as_posix().lower() for p in
                                     (self.project / 'masks', self.project / 'Masks') if p.is_dir()))},
                   'mask_note': '未登记逐图 Mask 关系；多边形标注不等于独立 Mask 文件',
                   'datasets': len(datasets), 'experiments': len(experiments), 'results': len(results),
                   'model_roles': preferences, 'warnings': warnings,
                   'context_policy': '自动整理现有事实；外部资源首次关联后项目共享，缺失/未知不推断'}
        self.set_setting('last_inventory', {'updated_at': summary['as_of'], 'images': len(paths)})
        return {'summary': summary, 'datasets': datasets, 'experiments': experiments, 'comparisons': comparisons,
                'results': results, 'resources': list(resources.values())}

    def add_feedback(self, image, category, note='', artifact_id=None, _db=None, provenance=None):
        if category not in ISSUES or not isinstance(note, str) or len(note) > 2000:
            raise ValueError('问题类型或备注无效。')
        path = self.store.source(image)
        if not path.is_file():
            raise ValueError('图片已不存在。')
        evidence_file = None
        if artifact_id:
            folder, manifest = self.store.artifact(artifact_id)
            from Utils.AIResultBrowser import visual_rows
            _, rows = visual_rows(self.store, artifact_id)
            row = next((r for r in rows if r.get('image') and Path(r['image']).resolve() == path), None)
            if row is None or row.get('chart') or row.get('error'):
                raise ValueError('当前结果中没有可记录的这张图片。')
            if category in {'false_positive', 'false_negative', 'correct'} and manifest['kind'] != 'candidates':
                raise ValueError('误检、漏检和预测正常须关联具体推理结果。')
            if row.get('hash') and row['hash'] != file_sha(path):
                raise ValueError('图片在推理后已改变，不能对旧结果记录新判断。')
            evidence_file = folder / 'candidates.json' if manifest['kind'] == 'candidates' else Path(row['annotation'])
        else:
            if category in {'false_positive', 'false_negative', 'correct'}:
                raise ValueError('请在具体推理结果中记录误检、漏检或预测正常。')
            if image not in read_project_list(self.project / 'datafile.dat'):
                raise ValueError('图片不在当前工程中。')
            evidence_file = self.project / 'jsons' / (path.stem + '.json')
        payload = {'note': note, 'image_sha256': file_sha(path), 'author': 'human',
                   'evidence': self.location(evidence_file),
                   'evidence_sha256': document_token(evidence_file, evidence_file.read_bytes()) if evidence_file.is_file() else None}
        aid = artifact_id or ''
        if provenance:
            payload['provenance'] = provenance
        with (self.db() if _db is None else nullcontext(_db)) as db:
            conflicting = ('false_positive', 'false_negative') if category == 'correct' else ('correct',) if category in {'false_positive', 'false_negative'} else ()
            if conflicting and db.execute('SELECT id FROM feedback WHERE image=? AND artifact=? AND status=? AND category IN (' +
                    ','.join('?' for _ in conflicting) + ')', (image, aid, 'open', *conflicting)).fetchone():
                raise ValueError('同一推理结果已有相反的人工判断，请先在项目状态中处理旧记录。')
            existing = db.execute('SELECT id,payload FROM feedback WHERE image=? AND artifact=? AND category=? AND status=? ORDER BY updated_at DESC LIMIT 1',
                                  (image, aid, category, 'open')).fetchone()
            if existing:
                old = json.loads(existing['payload'])
                if any(old.get(k) != payload[k] for k in ('image_sha256', 'evidence_sha256')):
                    existing = None
            fid = existing['id'] if existing else identifier()
            now = stamp()
            if existing:
                db.execute('UPDATE feedback SET updated_at=?,payload=? WHERE id=?',
                           (now, json.dumps(payload, ensure_ascii=False), fid))
            else:
                db.execute('INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?)',
                           (fid, image, aid, category, 'open', now, now, json.dumps(payload, ensure_ascii=False)))
        return fid

    def resolve_feedback(self, fid):
        with self.db() as db:
            if not db.execute('UPDATE feedback SET status=?,updated_at=? WHERE id=?', ('resolved', stamp(), fid)).rowcount:
                raise ValueError('问题记录不存在。')

    def feedback(self, category=None, day=None, artifact_id=None, status='open'):
        if category and category not in ISSUES:
            raise ValueError('问题类型无效。')
        if status not in {'open', 'resolved', 'all'}:
            raise ValueError('问题状态无效。')
        if day in {'today', 'yesterday'}:
            day = (datetime.now() - timedelta(days=day == 'yesterday')).date().isoformat()
        if day:
            datetime.strptime(day, '%Y-%m-%d')
        conditions, values = [], []
        for column, value in (('category', category), ('artifact', artifact_id), ('substr(created_at,1,10)', day),
                              ('status', status if status != 'all' else None)):
            if value is not None:
                conditions.append(column + '=?')
                values.append(value)
        with self.db() as db:
            rows = db.execute('SELECT * FROM feedback' + (' WHERE ' + ' AND '.join(conditions) if conditions else '') +
                              ' ORDER BY created_at DESC', values).fetchall()
        result, hashes = [], {}
        def digest(path, document=False):
            key = (str(path), document)
            if key not in hashes:
                hashes[key] = ((document_token(path, path.read_bytes()) if document else file_sha(path))
                               if path.is_file() else None)
            return hashes[key]
        for row in rows:
            value = dict(row)
            payload = json.loads(value.pop('payload'))
            value.update(payload, category_name=ISSUES[value['category']], stale=False)
            try:
                if value['artifact']:
                    if not self.artifact(value['artifact'])['available']:
                        raise ValueError('来源已不可用。')
                value['stale'] = (digest(self.store.source(value['image'])) != value['image_sha256'] or
                                  digest(self.resolve(value['evidence']), True) != value['evidence_sha256'])
            except (OSError, ValueError):
                value['stale'] = True
            result.append(value)
        return result

    def query(self, topic='overview', day=None, category=None, artifact_id=None, offset=0, limit=20,
              paths=None, flags=None, classes=None, cancelled=lambda: False):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError('分页参数无效。')
        if topic == 'lineage':
            if not artifact_id:
                raise ValueError('查看图片来源须指定处理批次。')
            from Utils.AIResultBrowser import visual_rows
            manifest, rows = visual_rows(self.store, artifact_id)
            items = [{'image': self.location(row['image']) if row.get('image') else None,
                      'source_image': row.get('source'),
                      'annotation': self.location(row['annotation']) if row.get('annotation') else None,
                      'prediction_snapshot': manifest['kind'] == 'candidates', 'error': row.get('error')}
                     for row in rows]
            extra = {'target_artifact': artifact_id, 'batch': self.artifact(artifact_id),
                     'basis': '该批次保存的逐图来源记录；预测快照不等于正式标注'}
        elif topic == 'feedback':
            items = self.feedback(category, day, artifact_id)
            batches = {}
            for row in items:
                if not row['stale']:
                    batches.setdefault(row['artifact'] or 'originals', set()).add(row['image'])
            counts = sorted([{'artifact_id': aid, 'marked_images': len(images)} for aid, images in batches.items()],
                            key=lambda r: r['marked_images'], reverse=True)
            extra = {'batches': counts, 'basis': '按人工记录日期及标记图片数；不是全批误检率，未标记图片不算正确'}
        else:
            inventory = self.inventory(paths, flags, classes, cancelled)
            if topic == 'overview':
                return {**inventory['summary'], 'recent_results': inventory['results'][:5],
                        'note': '详细数据集/实验/资源/人工问题请继续查询；没有统一最佳模型结论'}
            if topic not in {'datasets', 'experiments', 'results', 'resources'}:
                raise ValueError('未知项目状态查询类型。')
            items = inventory[topic]
            extra = {'warnings': inventory['summary']['warnings']}
            if topic == 'experiments':
                extra['comparable_groups'] = inventory['comparisons']
            if day:
                if day in {'today', 'yesterday'}:
                    day = (datetime.now() - timedelta(days=day == 'yesterday')).date().isoformat()
                datetime.strptime(day, '%Y-%m-%d')
                items = [v for v in items if (v.get('created_at') or '')[:10] == day]
            if artifact_id:
                items = [v for v in items if v.get('artifact_id') == artifact_id]
        return {'topic': topic, 'items': items[offset:offset + limit], 'total': len(items),
                'next_offset': offset + limit if offset + limit < len(items) else None, **extra}
