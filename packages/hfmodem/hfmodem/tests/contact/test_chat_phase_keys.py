# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import ast
import inspect
import sys
import textwrap
from types import SimpleNamespace

import pytest

from hfmodem.contact.chat_state import Phase, State
from hfmodem.contact.mail_ack import MailAckOwner
from hfmodem.contact.varac import CALLING_HZ, Session
from hfmodem.contact.varac import State as BackendState


CASES = (
    ('closing', {'_closing': True}),
    ('closing', {'_quitting': True}),
    ('host-closed', {'client.closed': True}),
    ('settling', {'client.ptt': True, 'flow.state': BackendState.CONNECTED}),
    ('settling', {'line_state': lambda: True, 'flow.state': BackendState.RETURNED}),
    ('settling', {'client.ptt': True, 'flow.state': BackendState.FAILED}),
    ('broadcast', {'broadcast.active': True}),
    ('broadcast-slot', {'broadcast.selected': CALLING_HZ[20]}),
    ('band-fault', {'_band_fault': True}),
    ('email-compose', {'email.draft': ('draft',)}),
    (('email', 'waiting for ER'), {'email.waiting': True, 'email.phase': 'waiting for ER'}),
    ('composing', {'compose': ('subject', [])}),
    (('qsy', 'readiness'), {'flow.qsy_phase': 'readiness'}),
    ('bbs-send', {'bbs_host.active': True}),
    ('file-download', {'files.active': True, 'files.downloading': True}),
    ('file-capture', {'files.active': True}),
    ('bbs', {'_bbs_waiting': True}),
    ('opening', {'_opening': True}),
    ('vmail-ack', {'mail_ack.state': 'active', 'mail_ack.kind': 'direct', 'mail_ack.generation': 0, 'client.connected': True}),
    ('cq', {'steps': iter(()), 'intent': 'cq'}),
    ('beacon', {'steps': iter(()), 'intent': 'beacon'}),
    ('connecting', {'client.pending': True}),
    ('connecting', {'flow.state': BackendState.CONNECTING}),
    ('exchange', {'client.connected': True, 'steps': iter(())}),
    ('ready-cf', {'client.connected': True}),
    ('ready-slot', {'client.connected': True, 'flow.dial': 14106500}),
    ('monitoring', {'_follow_tuned': True}),
    ('monitoring', {}),
)


def session(patch):
    s = object.__new__(Session)
    s._closing = s._quitting = s._band_fault = False
    s._bbs_waiting = s._opening = s._follow_tuned = False
    s.client = SimpleNamespace(closed=False, connected=False, ptt=False, pending=False, link_generation=0)
    s.mail_ack = MailAckOwner(s)
    s.flow = SimpleNamespace(state=BackendState.MONITORING, dial=CALLING_HZ[20],
                             qsy_phase=None, qsy_slot=3)
    s.broadcast = SimpleNamespace(active=False, phase='waiting for admission', selected=None)
    s.email = SimpleNamespace(draft=None, waiting=False, phase=None)
    s.bbs_host = SimpleNamespace(active=False, name='example.txt')
    s.files = SimpleNamespace(active=False, downloading=False)
    s.hopping = SimpleNamespace(active=False)
    s.compose = s.steps = s.intent = None
    s.auto = True
    s.line_state = lambda: False
    for name, value in patch.items():
        target = s
        *parents, attr = name.split('.')
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, attr, value)
    return s


@pytest.mark.parametrize('expected,patch', CASES)
def test_operator_phase_branches(expected, patch):
    key, description, action = session(patch)._operator_phase()
    assert key == expected
    assert isinstance(description, str) and description
    assert isinstance(action, str) and action
    normalized = key[0] if isinstance(key, tuple) else key
    assert Phase(normalized).value == normalized


def test_phase_inventory_covers_every_return_and_dynamic_key():
    source, start = inspect.getsourcelines(Session._operator_phase)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    return_lines = {start + node.lineno - 1 for node in returns}
    observed_lines = set()
    observed_keys = set()
    code = Session._operator_phase.__code__

    def trace(frame, event, arg):
        if frame.f_code is code and event == 'return':
            observed_lines.add(frame.f_lineno)
        return trace

    old_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        for _, patch in CASES:
            key, _, _ = session(patch)._operator_phase()
            observed_keys.add(key[0] if isinstance(key, tuple) else key)
    finally:
        sys.settrace(old_trace)
    assert observed_lines == return_lines
    assert observed_keys == {phase.value for phase in Phase}

    literal_keys = set()
    dynamic_keys = []
    for node in returns:
        assert isinstance(node.value, ast.Tuple) and len(node.value.elts) == 3
        key = node.value.elts[0]
        if isinstance(key, ast.Tuple):
            assert len(key.elts) == 2
            key = key.elts[0]
        if isinstance(key, ast.Constant):
            literal_keys.add(key.value)
        else:
            dynamic_keys.append(ast.unparse(key))
    assert dynamic_keys == ['self.intent']
    intent_tests = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
                    and ast.unparse(node.left) == 'self.intent']
    assert len(intent_tests) == 1
    intent_test = intent_tests[0]
    assert len(intent_test.ops) == 1 and isinstance(intent_test.ops[0], ast.In)
    dynamic_values = set(ast.literal_eval(intent_test.comparators[0]))
    assert dynamic_values == {'cq', 'beacon'}
    assert literal_keys | dynamic_values == observed_keys


def test_contract_state_matches_backend_state():
    assert {state.name: state.value for state in State} == {
        state.name: state.value for state in BackendState}


def test_phase_precedence_remains_backend_owned():
    s = session({'_closing': True, 'client.closed': True, 'broadcast.active': True,
                 'email.draft': ('draft',), 'flow.qsy_phase': 'readiness',
                 'client.connected': True})
    assert s._operator_phase()[0] == 'closing'
    s._closing = False
    assert s._operator_phase()[0] == 'host-closed'
    s.client.closed = False
    assert s._operator_phase()[0] == 'broadcast'
    s.broadcast.active = False
    assert s._operator_phase()[0] == 'email-compose'
    s.email.draft = None
    assert s._operator_phase()[0] == ('qsy', 'readiness')
