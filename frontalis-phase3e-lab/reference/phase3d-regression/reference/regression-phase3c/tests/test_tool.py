"""Read-only tool boundary/JSONL tests: no LLM or external API calls."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from npn.model import init, update, digest, evaluate
from npn.store import save
from tools.evaluate import decision, TOOL

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [.85, .2, .55, .2, .65, .7, .45, .75]


def arguments():
    return {'query_id': 'case-0', 'space': 'relational',
            'features': FEATURES.copy(), 'basis': 'affinity'}


def choice(i='a', f=None):
    return {'id': i, 'label': '研究用の選択肢', 'features': FEATURES.copy() if f is None else f}


def request(i, method, params=None):
    result = {'jsonrpc': '2.0', 'id': i, 'method': method}
    if params is not None:
        result['params'] = params
    return result


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.state = init(4311)

    def test_input_object_and_required_fields(self):
        for bad in (None, [], 'text', {}, {'query_id': 'x'}, {**arguments(), 'features': None}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decision(self.state, bad)

    def test_unknown_fields_paths_and_emotion_targets_rejected(self):
        for key in ('statePath', 'path', 'update', 'emotion', 'emotionTarget', 'reward', 'praise', 'personalityTarget'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                decision(self.state, {**arguments(), key: 'prohibited'})

    def test_features_require_finite_numeric_nonboolean_eight_values(self):
        invalid = [[.5] * 7, [.5] * 9, [True] * 8, ['0.5'] * 8,
                   [float('nan')] * 8, [float('inf')] * 8, [-.01] * 8,
                   [1.01] * 8, (.5,) * 8]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decision(self.state, {**arguments(), 'features': value})
        for value in ([0] * 8, [1] * 8, [.5] * 8):
            self.assertEqual(decision(self.state, {**arguments(), 'features': value})['query_id'], 'case-0')

    def test_unknown_space_basis_and_empty_query_rejected(self):
        for patch_value in ({'space': 'anything'}, {'basis': 'anger'},
                            {'query_id': ''}, {'query_id': ' '}, {'query_id': 'x' * 65}):
            with self.assertRaises(ValueError):
                decision(self.state, {**arguments(), **patch_value})

    def test_choice_container_preflight_before_any_inference(self):
        for bad in (None, {}, 'a', [choice(str(i)) for i in range(5)]):
            with patch('tools.evaluate.evaluate') as inference:
                with self.assertRaises(ValueError):
                    decision(self.state, {**arguments(), 'choices': bad})
                inference.assert_not_called()

    def test_choice_unknown_fields_and_bad_values_rejected(self):
        invalid = [None, 'a', {'id': 'a'}, {**choice(), 'path': 'private'},
                   {**choice(), 'id': ''}, {**choice(), 'id': 'hold'},
                   {**choice(), 'id': 'x' * 65}, {**choice(), 'label': ''},
                   {**choice(), 'label': 'x' * 241}, {**choice(), 'features': [False] * 8}]
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decision(self.state, {**arguments(), 'choices': [bad]})

    def test_duplicate_choice_ids_rejected_before_inference(self):
        with patch('tools.evaluate.evaluate') as inference:
            with self.assertRaises(ValueError):
                decision(self.state, {**arguments(), 'choices': [choice(), choice()]})
            inference.assert_not_called()

    def test_choice_tie_is_hold(self):
        output = decision(self.state, {**arguments(), 'choices': [choice('a'), choice('b')]})
        self.assertEqual(output['selection'], 'hold')
        self.assertEqual(output['choiceMargin'], 0.)

    def test_choice_basis_is_separate_from_scene_affinity_stance(self):
        def fixture(state, features, space):
            affinity = .6 if features[0] == .85 else (.9 if features[0] == .1 else -.9)
            value = -.4 if features[0] == .85 else (-.7 if features[0] == .1 else .7)
            return {'channels': {'affinity': affinity, 'value': value}, 'signals': {}, 'predictedOutcome': [.5] * 8}
        options = [choice('a', [.1] * 8), choice('b', [.9] * 8)]
        with patch('tools.evaluate.evaluate', side_effect=fixture):
            ordinary = decision(self.state, {**arguments(), 'choices': options})
            ethical = decision(self.state, {**arguments(), 'basis': 'value', 'choices': options})
        self.assertEqual(ordinary['selection'], 'a')
        self.assertEqual(ethical['selection'], 'b')
        self.assertEqual(ordinary['stance'], ethical['stance'])
        self.assertEqual(ethical['stanceBasis'], 'affinity')
        self.assertEqual(ethical['decisionBasis'], 'value')

    def test_repeated_same_input_same_signal_no_state_mutation(self):
        state = update(self.state, {'id': 'tool-test-event', 'text': '研究用観測',
            'space': 'relational', 'cue': FEATURES, 'observed': [.2, .8, .4, .7, .6, .3, .8, .4],
            'confirmed': True, 'origin': 'synthetic'})
        before = digest(state)
        first = decision(state, arguments())
        second = decision(state, arguments())
        self.assertEqual(first, second)
        self.assertEqual(digest(state), before)
        self.assertEqual(first['stateHash'], before)
        self.assertEqual(first['predictionGuard'], [.5] * 8)

    def test_choice_label_does_not_change_signal(self):
        first = decision(self.state, {**arguments(), 'choices': [choice()]})
        second = decision(self.state, {**arguments(), 'choices': [{**choice(), 'label': '違う表示文'}]})
        self.assertEqual(first['choiceScores'][0]['score'], second['choiceScores'][0]['score'])
        self.assertEqual(first['signals'], second['signals'])

    def test_schema_advertises_readonly_and_closed_input(self):
        self.assertTrue(TOOL['annotations']['readOnlyHint'])
        self.assertFalse(TOOL['annotations']['destructiveHint'])
        self.assertFalse(TOOL['inputSchema']['additionalProperties'])
        self.assertNotIn('statePath', TOOL['inputSchema']['properties'])
        self.assertEqual(TOOL['inputSchema']['properties']['features']['minItems'], 8)
        self.assertEqual(TOOL['inputSchema']['properties']['choices']['maxItems'], 4)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        model_directory = ROOT / 'models'
        model_directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=model_directory, prefix='test-tool-')
        self.path = Path(self.temporary.name) / 'raw-state.json'
        self.state = init(4311)
        self.path.write_text(json.dumps(self.state), encoding='utf8')
        self.trace_id = 'test-' + uuid.uuid4().hex
        self.trace = ROOT / 'reports' / 'tool-traces' / (self.trace_id + '.jsonl')

    def tearDown(self):
        if self.trace.exists():
            self.trace.unlink()
        self.temporary.cleanup()

    def run_server(self, requests, state_path=None, trace_id=None):
        env = {**os.environ, 'FRONTALIS_STATE_PATH': str(state_path or self.path),
               'FRONTALIS_RUN_ID': trace_id or self.trace_id, 'PYTHONUTF8': '1',
               'PYTHONDONTWRITEBYTECODE': '1'}
        incoming = ''.join((value if isinstance(value, str) else json.dumps(value, allow_nan=True)) + '\n'
                           for value in requests)
        result = subprocess.run([sys.executable, '-u', str(ROOT / 'tools' / 'evaluate.py')],
                                cwd=ROOT, input=incoming, text=True, capture_output=True,
                                encoding='utf8', env=env, timeout=15)
        replies = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        return result, replies

    def test_initialize_ping_list_and_notification_protocol(self):
        result, replies = self.run_server([
            request(0, 'initialize', {'protocolVersion': '2024-11-05'}),
            {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
            request(1, 'ping'), request(2, 'tools/list')])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row['id'] for row in replies], [0, 1, 2])
        self.assertEqual(replies[0]['result']['protocolVersion'], '2024-11-05')
        self.assertEqual(replies[1]['result'], {})
        self.assertEqual(len(replies[2]['result']['tools']), 1)
        self.assertEqual(replies[2]['result']['tools'][0]['name'], 'evaluate')

    def test_readonly_call_repeat_and_trace_correspondence(self):
        before_bytes = self.path.read_bytes()
        params = {'name': 'evaluate', 'arguments': arguments()}
        result, replies = self.run_server([request(1, 'tools/call', params), request(2, 'tools/call', params)])
        self.assertEqual(result.returncode, 0, result.stderr)
        first, second = [json.loads(row['result']['content'][0]['text']) for row in replies]
        self.assertEqual(first, second)
        self.assertEqual(first['stateHash'], digest(self.state))
        self.assertEqual(self.path.read_bytes(), before_bytes)
        traces = [json.loads(line) for line in self.trace.read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0]['input'], arguments())
        self.assertEqual(traces[0]['output'], first)

    def test_wrapper_snapshot_supported_and_integrity_checked(self):
        wrapper = Path(self.temporary.name) / 'wrapper-state.json'
        save(wrapper, self.state)
        result, replies = self.run_server([request(1, 'tools/call', {'name': 'evaluate', 'arguments': arguments()})], wrapper)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(replies[0]['result']['content'][0]['text'])['stateHash'], digest(self.state))
        corrupt = json.loads(wrapper.read_text(encoding='utf8'))
        corrupt['payload']['branches']['relational']['b2'][0] = .1
        wrapper.write_text(json.dumps(corrupt), encoding='utf8')
        result, replies = self.run_server([request(1, 'ping')], wrapper)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(replies, [])
        self.assertIn('Snapshot integrity mismatch', result.stderr)

    def test_raw_malformed_parameter_snapshot_rejected(self):
        corrupted = copy.deepcopy(self.state)
        corrupted['branches']['relational']['w1'] = [[0.]]
        self.path.write_text(json.dumps(corrupted), encoding='utf8')
        result, replies = self.run_server([request(1, 'ping')])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(replies, [])
        self.assertIn('Invalid trainable parameter', result.stderr)

    def test_outside_model_directory_rejected(self):
        result, replies = self.run_server([request(1, 'ping')], ROOT / 'docs' / '03-design.md')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(replies, [])
        self.assertIn('State must be inside', result.stderr)

    def test_trace_path_injection_rejected(self):
        result, replies = self.run_server([request(1, 'ping')], trace_id='../invalid-path')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(replies, [])
        self.assertIn('Invalid trace identity', result.stderr)

    def test_parse_and_invalid_request_errors_do_not_reuse_prior_id(self):
        result, replies = self.run_server([
            request(7, 'ping'), '{bad json', '[]',
            {'jsonrpc': '2.0', 'id': True, 'method': 'ping'},
            request(8, 'ping')])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row['id'] for row in replies], [7, None, None, None, 8])
        self.assertEqual([row['error']['code'] for row in replies[1:4]], [-32700, -32600, -32600])
        self.assertEqual(replies[-1]['result'], {})

    def test_unknown_tool_and_method_and_bad_args_leave_state_unchanged(self):
        before = self.path.read_bytes()
        result, replies = self.run_server([
            request(1, 'tools/call', {'name': 'update', 'arguments': arguments()}),
            request(2, 'execute'),
            request(3, 'tools/call', {'name': 'evaluate', 'arguments': {**arguments(), 'path': 'private'}}),
            request(4, 'tools/call', {'name': 'evaluate', 'arguments': {**arguments(), 'features': [True] * 8}}),
            request(5, 'tools/call', {'name': 'evaluate', 'arguments': {**arguments(), 'features': [float('nan')] * 8}}),
            request(6, 'tools/call', {'name': 'evaluate', 'arguments': arguments()})])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row['error']['code'] for row in replies[:5]], [-32602, -32601, -32602, -32602, -32602])
        self.assertFalse(replies[-1]['result']['isError'])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(self.trace.read_text(encoding='utf8').splitlines()), 1)

    def test_paid_model_rejected_without_cli_launch(self):
        node = Path('D:/node.exe')
        if not node.exists():
            self.skipTest('Known Node runtime unavailable.')
        source = "import {runTool} from './scripts/tool_cli.mjs'; try { await runTool('paid/model', '', '', ''); process.exit(7); } catch(error) { if(!String(error).includes('Only confirmed free model ids')) throw error; console.log('rejected-before-launch'); }"
        result = subprocess.run([str(node), '--input-type=module', '-e', source], cwd=ROOT,
                                text=True, capture_output=True, encoding='utf8', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('rejected-before-launch', result.stdout)


if __name__ == '__main__':
    unittest.main()

