"""Actual provider serialization / Qt UI, with the HTTP boundary stubbed only."""
import base64
import copy
import hashlib
import io
import json
import os
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch
from PyQt5.QtWidgets import QApplication, QPushButton

from test_ai_chat_selection import Fixtures
from Utils.AIContextInspector import (ContextRecorder, load_inspection, sanitize, share_copy)
from Utils.AIContextInspectorDialog import ContextInspectorDialog
from Utils.AIAugment import request_qwen_json
from Utils.AIWorkAgent import run_agent, compact
from Utils.AIWorkspace import read_json, write_json


def response(action):
    class Response(io.BytesIO):
        status = 200
    return Response(json.dumps({'choices': [{'message': {'content': json.dumps(action, ensure_ascii=False)},
        'finish_reason': 'stop'}], 'usage': {'total_tokens': 13}}, ensure_ascii=False).encode())


class InspectorTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.context = {'current_paths': self.paths[:2], 'selected_paths': self.paths[:1],
                        'classes': ['脏污', '破损'], 'max_calls': 4}
        self.api = ('sk-secret-current123', 'https://example.invalid/chat?api_key=private', 'fixture-model')

    def agent(self, message='查看项目资源', history=None, context=None, request=None):
        return run_agent(self.root, self.paths, {}, {}, history or [], message, ('all', None, None),
                         self.api, context=context or self.context, request=request)

    def recorder(self, run_id='test_run'):
        recorder = ContextRecorder(self.root, run_id, self.api[0])
        recorder.begin(1, {'request': '示例', 'resources': {}, 'tool_results_this_turn': []},
                       '示例', [], {'project': str(self.root)}, [], 0)
        return recorder

    def test_provider_payload_and_tool_followup_are_actual_snapshots(self):
        weight = self.root / 'training/best.pt'
        weight.parent.mkdir()
        weight.write_bytes(b'fixture; never loaded')
        self.context['resources'] = {'r1': {'path': str(weight), 'name': 'best.pt', 'kind': 'model'}}
        sent = []
        actions = [{'action': 'tool', 'tool': 'project_context', 'args': {'topic': 'resources'}},
                   {'action': 'reply', 'message': '已读取'}]
        def http(request, **kwargs):
            sent.append(json.loads(request.data))
            return response(actions[len(sent) - 1])
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=http):
            result = self.agent(message='检查 ' + self.api[0])
        record = load_inspection(self.root, result['run_id'])
        self.assertEqual(len(record['requests']), 2)
        self.assertEqual(len(record['events']), 1)
        for request, payload in zip(record['requests'], sent):
            self.assertEqual(request['transports'][0]['payload'], sanitize(payload, [self.api[0]]))
            self.assertEqual(request['transports'][0]['status'], 'received')
            self.assertNotIn('?', request['transports'][0]['endpoint'])
        self.assertEqual(record['requests'][0]['injected_context']['tool_results_this_turn'], [])
        self.assertEqual(record['requests'][1]['injected_context']['tool_results_this_turn'][0]['tool'], 'project_context')
        self.assertEqual(record['events'][0]['after_request'], 1)
        self.assertIn('r1', record['requests'][0]['known_local']['resources'])
        frozen = copy.deepcopy(record)
        weight.unlink()
        self.context['current_paths'].clear()
        self.assertEqual(load_inspection(self.root, result['run_id']), frozen)
        self.assertNotIn(self.api[0], json.dumps(record))

    def test_images_identify_submitted_bytes_not_later_files(self):
        recorder = self.recorder()
        image = self.root / self.paths[0]
        original = image.read_bytes()
        def http(request, **kwargs):
            image.write_bytes(b'changed after submission')
            return response({'action': 'reply', 'message': 'ok'})
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=http), recorder.capture():
            request_qwen_json(*self.api, 'image request', [image], retries=1)
        record = load_inspection(self.root, 'test_run')['requests'][0]
        attachment = record['transports'][0]['images'][0]
        self.assertEqual(attachment['sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(attachment['bytes'], len(original))
        raw = json.dumps(record)
        self.assertNotIn(base64.b64encode(original).decode(), raw)
        self.assertNotIn('data:image', raw)

    def test_http_failure_keeps_attempt_and_scrubs_echoed_credentials(self):
        error = urllib.error.HTTPError(self.api[1], 401, 'denied', {}, io.BytesIO(self.api[0].encode()))
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=error):
            result = self.agent()
        record = load_inspection(self.root, result['run_id'])
        self.assertEqual(record['index']['status'], 'failed')
        self.assertEqual(record['requests'][0]['status'], 'failed')
        self.assertEqual(record['requests'][0]['transports'][0]['status'], 'http_error')
        self.assertNotIn(self.api[0], json.dumps(record))

    def test_retries_are_distinct_transport_attempts(self):
        recorder = self.recorder()
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=[urllib.error.URLError('offline'),
                response({'action': 'reply', 'message': 'ok'})]), patch('Utils.AIAugment.time.sleep'), recorder.capture():
            request_qwen_json(*self.api, 'retry', retries=2)
        attempts = load_inspection(self.root, 'test_run')['requests'][0]['transports']
        self.assertEqual([a['attempt'] for a in attempts], [1, 2])
        self.assertEqual([a['status'] for a in attempts], ['connection_error', 'received'])

    def test_tool_error_is_recorded_and_sent_in_next_request(self):
        actions = [{'action': 'tool', 'tool': 'does_not_exist', 'args': {}},
                   {'action': 'reply', 'message': '无法执行'}]
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=lambda *a, **k: response(actions.pop(0))):
            result = self.agent()
        record = load_inspection(self.root, result['run_id'])
        self.assertEqual(record['events'][0]['observation']['status'], 'failed')
        self.assertEqual(record['requests'][1]['injected_context']['tool_results_this_turn'][0]['status'], 'failed')

    def test_mock_transport_not_claimed_sent_and_resume_never_rewrites_first_request(self):
        request = lambda *a, **k: ({'action': 'reply', 'message': 'ok'}, {})
        first = self.agent(request=request)
        original = load_inspection(self.root, first['run_id'])['requests'][0]
        self.assertEqual(original['transports'], [])
        context = {**self.context, 'resume': first['work']['continuation']}
        second = self.agent(context=context, request=request)
        record = load_inspection(self.root, second['run_id'])
        self.assertEqual([r['number'] for r in record['requests']], [1, 2])
        self.assertEqual(original, record['requests'][0])

    def test_compaction_and_known_but_unsent_fields_are_explicit(self):
        observations = [{'result': {'paths': ['hidden.png'], 'document': {'annotations': []},
                                     'text': 'x' * 2500, 'items': list(range(30))}}]
        recorder = ContextRecorder(self.root, 'compact')
        public = {'tool_results_this_turn': compact(observations)}
        recorder.begin(1, public, 'prompt', [], {'resources': {'r1': {'path': 'local-only.pt'}}}, observations, 22)
        request = load_inspection(self.root, 'compact')['requests'][0]
        self.assertEqual(request['injected_context'], public)
        text = '\n'.join(request['omissions'])
        self.assertIn('paths：未注入', text)
        self.assertIn('text：已截断', text)
        self.assertIn('仅注入前20项', text)
        self.assertIn('最近12条', text)
        self.assertNotIn('resources', request['injected_context'])

    def test_export_scrubs_local_paths_and_secrets_inside_serialized_prompts(self):
        external = 'D:\\private\\customer\\best.pt'
        secret = 'test-secret-with-"quote'
        value = {'requests': [{'known_local': {'resources': {'r1': {'path': external}}},
                  'requested_images': [], 'prompt': json.dumps({'resource': external, 'project': str(self.root), 'api_key': secret})}],
                 'Authorization': 'Bearer token123', 'api_key': secret}
        clean = sanitize(value, [secret])
        exported = share_copy(clean, self.root)
        serialized = json.dumps(exported)
        self.assertNotIn('customer', serialized)
        self.assertNotIn(self.root.name, serialized)
        self.assertNotIn('test-secret', serialized)
        self.assertNotIn('token123', serialized)
        self.assertIn('best.pt', serialized)

    def test_observer_disk_error_does_not_change_agent_outcome(self):
        with patch('Utils.AIContextInspector.write_json', side_effect=PermissionError('read only')):
            result = self.agent(request=lambda *a, **k: ({'action': 'reply', 'message': 'ok'}, {}))
        self.assertEqual(result['work']['status'], 'completed')
        self.assertTrue(result['context_inspector_warnings'])

    def test_missing_and_partial_records_are_not_reconstructed(self):
        self.assertTrue(load_inspection(self.root, 'old')['missing'])
        recorder = self.recorder()
        (recorder.folder / 'request_0001.json').unlink()
        record = load_inspection(self.root, 'test_run')
        self.assertEqual(record['requests'], [])
        self.assertTrue(record['read_warnings'])
        with self.assertRaises(ValueError):
            load_inspection(self.root, '../escape')

    def test_malformed_json_response_and_cancellation_keep_request_evidence(self):
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=[
                response('this is not an action'), response({'action': 'reply', 'message': 'recovered'})]):
            result = self.agent()
        record = load_inspection(self.root, result['run_id'])
        self.assertEqual(len(record['requests']), 2)
        self.assertEqual(record['events'][0]['observation']['kind'], 'protocol')
        stopped = [False]
        def http(*args, **kwargs):
            stopped[0] = True
            return response({'action': 'reply', 'message': 'late response'})
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=http):
            result = run_agent(self.root, self.paths, {}, {}, [], 'stop', ('all', None, None), self.api,
                               context=self.context, cancelled=lambda: stopped[0])
        record = load_inspection(self.root, result['run_id'])
        self.assertEqual(record['index']['status'], 'stopped')
        self.assertEqual(record['requests'][0]['transports'][0]['status'], 'received')

    def test_corrupt_evidence_preserved_and_unknown_export_paths_removed(self):
        recorder = self.recorder()
        path = recorder.folder / 'request_0001.json'
        write_json(path, {'number': 1, 'injected_context': 'corrupt'})
        record = load_inspection(self.root, 'test_run')
        self.assertTrue(record['read_warnings'])
        self.assertEqual(record['requests'], [])
        self.assertEqual(read_json(path)['injected_context'], 'corrupt')
        exported = share_copy({'external': 'E:/sensitive/customer/results.csv',
                               'other': '/private/customer/results.csv',
                               'remote': 'https://example.org/v1/chat'}, self.root)
        self.assertNotIn('customer', json.dumps(exported))
        self.assertEqual(exported['remote'], 'https://example.org/v1/chat')

    def test_thread_local_capture_does_not_mix_models(self):
        def run(index):
            recorder = self.recorder('thread_' + str(index))
            with recorder.capture():
                request_qwen_json(self.api[0], self.api[1], f'model-{index}', 'thread', retries=1)
        with patch('Utils.AIAugment.urllib.request.urlopen', side_effect=lambda *a, **k: response({'action': 'reply', 'message': 'ok'})):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(run, [1, 2]))
        for index in (1, 2):
            record = load_inspection(self.root, f'thread_{index}')
            self.assertEqual(record['requests'][0]['transports'][0]['payload']['model'], f'model-{index}')

    def test_dialog_entry_history_and_export_are_read_only(self):
        from Utils.AIChatView import ConversationTranscript
        app = QApplication.instance() or QApplication([])
        with patch('Utils.AIAugment.urllib.request.urlopen', return_value=response({'action': 'reply', 'message': 'ok'})):
            result = self.agent()
        before = self.hashes()
        dialog = ContextInspectorDialog(self.root, result['run_id'])
        transcript = ConversationTranscript()
        try:
            self.assertEqual(dialog.requests.count(), 1)
            self.assertIn('fixture-model', dialog.raw.toPlainText())
            self.assertTrue(dialog.raw.isReadOnly())
            messages = [{'id': 'task', 'role': 'task', 'status': 'completed', 'created_at': '',
                         'text': '完成', 'run_id': result['run_id']}]
            with patch.dict(os.environ, {'LABELSYSTEM_CONTEXT_DEBUG': ''}):
                transcript.render(messages, {})
            self.assertFalse(any(b.text() == '查看本次上下文' for b in transcript.findChildren(QPushButton)))
            with patch.dict(os.environ, {'LABELSYSTEM_CONTEXT_DEBUG': '1'}):
                transcript.render(messages, {})
            events = []
            transcript.action.connect(lambda *args: events.append(args))
            button = next(b for b in transcript.findChildren(QPushButton) if b.text() == '查看本次上下文')
            button.click()
            self.assertEqual(events, [('context', result['run_id'])])
            destination = self.root / 'inspection-export.json'
            with patch('Utils.AIContextInspectorDialog.QFileDialog.getSaveFileName', return_value=(str(destination), 'JSON')):
                dialog.export()
            self.assertIn('export_note', read_json(destination))
            self.assertEqual(before, self.hashes())
            missing = ContextInspectorDialog(self.root, 'old')
            self.assertIn('没有上下文快照', missing.overview.toPlainText())
            self.assertFalse(missing.export_button.isEnabled())
            missing.close()
        finally:
            dialog.close()
            transcript.close()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
