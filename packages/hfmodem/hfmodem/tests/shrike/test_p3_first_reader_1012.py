# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Untimed reader-selection observables, with the old expression as reference.

Selection fixtures never run a packet decoder. Receiver consequences below use
small existing fixtures and scoped seams; the real ACK, narrow fallback, entry,
ownership, P1/P2 and QRT modules remain separate root-owned validation.
"""
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, rxfront
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_granted_entry_retry import granted
from hfmodem.tests.shrike.test_p3_input_loss import LOST, corroborated
from hfmodem.tests.shrike.test_p3level_session import _Session


AUDIO = object()
PACKETS = ('_p3_packet', '_p2_packet', '_p1_packet')


def legacy_readers(rx):
    readers = {Protocol.PACTOR3: rx._p3_packet,
               Protocol.PACTOR2: rx._p2_packet,
               Protocol.PACTOR1: rx._p1_packet}
    return [readers.pop(rx.host.protocol), *readers.values()]


def legacy_deep(rx, audio):
    rx._scan(audio, rx._readers()[:1])


def bare(host=None, cls=onair._SessionRx):
    rx = object.__new__(cls)
    rx.host = host if host is not None else SimpleNamespace(protocol=Protocol.PACTOR3)
    calls = []
    rx._scan = lambda audio, readers: calls.append((audio, readers))
    return rx, calls


def identities(readers):
    return [(reader.__self__, reader.__func__) if type(reader) is MethodType
            else reader for reader in readers]


@pytest.mark.parametrize('protocol,order', [
    (Protocol.PACTOR1, ('_p1_packet', '_p3_packet', '_p2_packet')),
    (Protocol.PACTOR2, ('_p2_packet', '_p3_packet', '_p1_packet')),
    (Protocol.PACTOR3, ('_p3_packet', '_p2_packet', '_p1_packet')),
])
@pytest.mark.parametrize('tracked', [False, True])
def test_protocol_orders_upgrade_tail_and_fresh_lists(protocol, order, tracked):
    rx, calls = bare(SimpleNamespace(protocol=protocol))
    rx._tracked_only = tracked
    expected = identities([getattr(rx, name) for name in order])
    full, another = rx._readers(), rx._readers()
    first, next_first = rx._readers(first_only=True), rx._readers(first_only=True)
    assert identities(full) == identities(legacy_readers(rx)) == expected
    assert identities(first) == expected[:1]
    assert full is not another and first is not next_first
    full.clear()
    first.append(object())
    assert identities(another) == expected and identities(next_first) == expected[:1]
    assert identities(rx._readers()) == expected
    rx.deep_scan(AUDIO)
    rx.upgrade_scan(AUDIO)
    assert calls[0][0] is AUDIO and identities(calls[0][1]) == expected[:1]
    assert calls[1][0] is AUDIO and identities(calls[1][1]) == expected[1:]
    rx.deep_scan(AUDIO)
    assert calls[2][1] is not calls[0][1]


def test_protocol_and_each_packet_binding_stay_live_between_calls():
    rx, calls = bare()
    for protocol, name in zip((Protocol.PACTOR3, Protocol.PACTOR2, Protocol.PACTOR1), PACKETS):
        rx.host.protocol = protocol
        for _ in range(2):
            replacement = lambda audio: pytest.fail('selection called a decoder')
            setattr(rx, name, replacement)
            expected = legacy_readers(rx)
            assert rx._readers()[0] is replacement
            rx.deep_scan(AUDIO)
            assert calls[-1][1] == expected[:1] == [replacement]
            rx.upgrade_scan(AUDIO)
            assert identities(calls[-1][1]) == identities(expected[1:])


def test_alternating_protocol_getter_is_read_once_by_each_selector():
    class Host:
        reads = 0

        @property
        def protocol(self):
            value = (Protocol.PACTOR2, Protocol.PACTOR3)[self.reads % 2]
            self.reads += 1
            return value

    host = Host()
    rx, calls = bare(host)
    for name in ('_p2_packet', '_p3_packet', '_p2_packet', '_p3_packet'):
        before = host.reads
        rx.deep_scan(AUDIO)
        assert host.reads == before + 1
        assert identities(calls[-1][1]) == identities([getattr(rx, name)])


def test_getter_replacement_keeps_this_calls_packet_scan_and_selector():
    trace, replacements = [], [object(), object(), object()]
    replacement_calls = []

    class Host:
        @property
        def protocol(self):
            trace.append('protocol')
            for name, value in zip(PACKETS, replacements):
                setattr(rx, name, value)
            rx._scan = lambda audio, readers: replacement_calls.append((audio, readers))
            rx._readers = lambda: [replacements[1], replacements[0]]
            return Protocol.PACTOR3

    rx, calls = bare(Host())
    old = identities([rx._p3_packet])
    rx.deep_scan(AUDIO)
    assert trace == ['protocol'] and identities(calls[0][1]) == old
    assert not replacement_calls
    rx.deep_scan(AUDIO)
    assert trace == ['protocol']
    assert replacement_calls == [(AUDIO, [replacements[1]])]
    del rx._readers
    rx.deep_scan(AUDIO)
    assert trace == ['protocol', 'protocol']
    assert replacement_calls[-1] == (AUDIO, [replacements[0]])


def test_scan_resolution_precedes_single_selector_resolution(monkeypatch):
    trace, selected = [], object()
    rx, _ = bare()
    del rx._scan
    original = onair._SessionRx._readers

    def scan_get(receiver):
        trace.append('scan-get')
        # Mutation during scan lookup must be visible to selector lookup.
        receiver._p3_packet = selected
        return lambda audio, readers: trace.append(('scan', audio, readers))

    def readers_get(receiver):
        trace.append('readers-get')
        return original.__get__(receiver, type(receiver))

    with monkeypatch.context() as scoped:
        scoped.setattr(onair._SessionRx, '_scan', property(scan_get))
        scoped.setattr(onair._SessionRx, '_readers', property(readers_get))
        rx.deep_scan(AUDIO)
    assert trace == ['scan-get', 'readers-get', ('scan', AUDIO, [selected])]
    assert onair._SessionRx._readers is original


def test_interleaved_key_binding_dictionary_hash_and_getter_order(monkeypatch):
    original = onair._SessionRx._readers

    def run(select):
        trace = []

        class Key:
            def __init__(self, level):
                self.level = level

            def __hash__(self):
                trace.append(('hash', self.level))
                return self.level

        keys = {level: Key(level) for level in (3, 2, 1)}

        class Keys:
            def __getattr__(self, name):
                level = int(name[-1])
                trace.append(('key', level))
                return keys[level]

        class Host:
            @property
            def protocol(self):
                trace.append('protocol')
                return keys[2]

        rx, _ = bare(Host())
        with monkeypatch.context() as scoped:
            scoped.setattr(onair, 'Protocol', Keys())
            for level, name in zip((3, 2, 1), PACKETS):
                def bind(receiver, level=level):
                    trace.append(('bind', level))
                    return level
                scoped.setattr(onair._SessionRx, name, property(bind))
            if select == 'legacy':
                # The same old expression, with the scoped live keys.
                readers = {onair.Protocol.PACTOR3: rx._p3_packet,
                           onair.Protocol.PACTOR2: rx._p2_packet,
                           onair.Protocol.PACTOR1: rx._p1_packet}
                result = [readers.pop(rx.host.protocol), *readers.values()][:1]
            else:
                result = original(rx, first_only=True)
        return trace, result

    reference = run('legacy')
    assert run('candidate') == reference
    assert reference == ([('key', 3), ('bind', 3), ('key', 2), ('bind', 2),
                          ('key', 1), ('bind', 1), ('hash', 3), ('hash', 2),
                          ('hash', 1), 'protocol', ('hash', 2)], [2])


@pytest.mark.parametrize('fault', ['scan', 'selector', *PACKETS, 'protocol'])
def test_lookup_and_getter_errors_keep_order_payload_and_no_extra_reads(monkeypatch, fault):
    error = RuntimeError('lookup failed', fault)

    def run(candidate):
        trace = []

        class Host:
            @property
            def protocol(self):
                trace.append('protocol')
                if fault == 'protocol':
                    raise error
                return Protocol.PACTOR3

        rx, calls = bare(Host())
        original = onair._SessionRx._readers
        del rx._scan

        def scan_get(receiver):
            trace.append('scan')
            if fault == 'scan':
                raise error
            return lambda audio, readers: calls.append(readers)

        def selector_get(receiver):
            trace.append('selector')
            if fault == 'selector':
                raise error
            return (original if candidate else legacy_readers).__get__(receiver, type(receiver))

        with monkeypatch.context() as scoped:
            scoped.setattr(onair._SessionRx, '_scan', property(scan_get))
            scoped.setattr(onair._SessionRx, '_readers', property(selector_get))
            for name in PACKETS:
                def bind(receiver, name=name):
                    trace.append(name)
                    if fault == name:
                        raise error
                    return name
                scoped.setattr(onair._SessionRx, name, property(bind))
            with pytest.raises(RuntimeError) as caught:
                (rx.deep_scan(AUDIO) if candidate else legacy_deep(rx, AUDIO))
            assert caught.value is error and not calls
        assert onair._SessionRx._readers is original
        return trace

    assert run(True) == run(False)


class EqualString(str):
    pass


@pytest.mark.parametrize('kind', ['enum', 'string', 'string-subclass', 'equal',
                                  'collision', 'invalid', 'unhashable',
                                  'hash-error', 'eq-error', 'hash-mutation', 'eq-mutation'])
@pytest.mark.parametrize('selection', ['default', 'first', 'deep'])
def test_dictionary_lookup_observables_match_untimed_legacy(kind, selection):
    error = RuntimeError('dictionary lookup', kind)

    def run(candidate):
        trace = []
        rx, calls = bare()
        captured = [object(), object(), object()]
        replacements = [object(), object(), object()]
        for name, value in zip(PACKETS, captured):
            setattr(rx, name, value)

        def mutate():
            trace.append('replace-packets')
            for name, value in zip(PACKETS, replacements):
                setattr(rx, name, value)

        class Lookup:
            def __hash__(self):
                trace.append('hash')
                if kind == 'hash-error':
                    raise error
                if kind == 'hash-mutation':
                    mutate()
                return hash(Protocol.PACTOR2)

            def __eq__(self, other):
                trace.append(('eq', str(other)))
                if kind == 'eq-error':
                    raise error
                if kind == 'eq-mutation':
                    mutate()
                return kind != 'collision' and str(other) == str(Protocol.PACTOR2)

        key = {'enum': Protocol.PACTOR2, 'string': str(Protocol.PACTOR2),
               'string-subclass': EqualString(Protocol.PACTOR2),
               'invalid': 'not-a-protocol', 'unhashable': []}.get(kind)
        if key is None:
            key = Lookup()

        class Host:
            @property
            def protocol(self):
                trace.append('protocol')
                return key

        rx.host = Host()
        try:
            if not candidate:
                result = legacy_readers(rx)
                if selection != 'default':
                    result = result[:1]
            elif selection == 'deep':
                rx.deep_scan(AUDIO)
                result = calls[0][1]
            else:
                result = rx._readers() if selection == 'default' else rx._readers(first_only=True)
        except (KeyError, TypeError, RuntimeError) as caught:
            if isinstance(caught, KeyError):
                assert caught.args[0] is key
                outcome = ('error', KeyError, 'same lookup object')
            else:
                if isinstance(caught, RuntimeError):
                    assert caught is error
                outcome = ('error', type(caught), caught.args)
            assert not calls
        else:
            assert result[0] is captured[1]
            outcome = ('readers', [captured.index(value) for value in result])
            if selection == 'deep' and candidate:
                assert calls[0][0] is AUDIO
        return trace, outcome

    reference = run(False)
    assert run(True) == reference
    expected_error = {'collision': KeyError, 'invalid': KeyError,
                      'unhashable': TypeError, 'hash-error': RuntimeError,
                      'eq-error': RuntimeError}.get(kind)
    assert reference[0].count('protocol') == 1
    if expected_error is not None:
        assert reference[1][:2] == ('error', expected_error)
    else:
        assert reference[1] == ('readers', [1, 0, 2] if selection == 'default' else [1])


def test_first_only_truth_test_occurs_after_successful_pop():
    trace = []
    error = RuntimeError('flag truth test')

    class Flag:
        def __bool__(self):
            trace.append('flag')
            raise error

    class Host:
        @property
        def protocol(self):
            trace.append('protocol')
            return value

    rx, _ = bare(Host())
    value = 'invalid'
    with pytest.raises(KeyError):
        rx._readers(first_only=Flag())
    assert trace == ['protocol']
    value = Protocol.PACTOR1
    with pytest.raises(RuntimeError) as caught:
        rx._readers(first_only=Flag())
    assert caught.value is error and trace == ['protocol', 'protocol', 'flag']


@pytest.mark.parametrize('kind', ['subclass', 'super', 'instance', 'class', 'spoof',
                                  'attribute-trap', 'other-receiver'])
@pytest.mark.parametrize('result_kind', ['custom', 'empty', 'sliceable', 'error'])
def test_no_argument_override_dispatch_and_original_slice(monkeypatch, kind, result_kind):
    trace, one, two, sliced = [], object(), object(), object()
    error = RuntimeError('override failed')
    original = onair._SessionRx._readers

    class Sliceable:
        def __getitem__(self, key):
            trace.append(('slice', key.start, key.stop, key.step))
            return sliced

    def select():
        trace.append('select')
        if result_kind == 'error':
            raise error
        return [] if result_kind == 'empty' else Sliceable() if result_kind == 'sliceable' else [two, one]

    class Subclass(onair._SessionRx):
        def _readers(self):
            return select()

    class SuperSubclass(onair._SessionRx):
        def _readers(self):
            trace.append('super')
            full = super()._readers()
            assert len(full) == 3
            return select()

    class Spoof:
        def __call__(self):
            return select()

    class AttributeTrap:
        @property
        def __self__(self):
            pytest.fail('spoof attributes were inspected')

        @property
        def __func__(self):
            pytest.fail('spoof attributes were inspected')

        def __call__(self):
            return select()

    with monkeypatch.context() as scoped:
        rx, calls = bare(cls=Subclass if kind == 'subclass' else
                         SuperSubclass if kind == 'super' else onair._SessionRx)
        if kind == 'instance':
            rx._readers = select
        elif kind == 'class':
            scoped.setattr(onair._SessionRx, '_readers', lambda self: select())
        elif kind == 'spoof':
            spoof = Spoof()
            spoof.__self__ = rx
            spoof.__func__ = original
            rx._readers = spoof
        elif kind == 'attribute-trap':
            rx._readers = AttributeTrap()
        elif kind == 'other-receiver':
            # Keep the other receiver's live host and packet bindings.
            other, _ = bare()

            class Host:
                @property
                def protocol(self):
                    trace.append('other-protocol')
                    if result_kind == 'error':
                        raise error
                    return Protocol.PACTOR2

            other.host = Host()
            other._p2_packet = two
            rx._readers = original.__get__(other, type(other))
        if result_kind == 'error':
            with pytest.raises(RuntimeError) as caught:
                rx.deep_scan(AUDIO)
            assert caught.value is error and not calls
        else:
            rx.deep_scan(AUDIO)
            expected = [two] if kind == 'other-receiver' else (
                [] if result_kind == 'empty' else sliced if result_kind == 'sliceable' else [two])
            assert calls == [(AUDIO, expected)]
    assert onair._SessionRx._readers is original
    assert trace.count('select') == (0 if kind == 'other-receiver' else 1)
    if result_kind == 'sliceable' and kind != 'other-receiver':
        assert trace[-1] == ('slice', None, 1, None)


def test_other_receiver_full_result_still_uses_slice():
    rx, calls = bare()
    other, _ = bare()
    trace, selected = [], object()

    class FullResult(list):
        def __getitem__(self, key):
            trace.append(key)
            return super().__getitem__(key)

    def custom(self):
        assert self is other
        return FullResult([selected, object()])

    rx._readers = MethodType(custom, other)
    rx.deep_scan(AUDIO)
    assert calls == [(AUDIO, [selected])] and trace == [slice(None, 1)]


def test_override_slice_error_and_selected_reader_error_propagate():
    rx, calls = bare()
    error = RuntimeError('custom slice')

    class Result:
        def __getitem__(self, key):
            raise error

    rx._readers = lambda: Result()
    with pytest.raises(RuntimeError) as caught:
        rx.deep_scan(AUDIO)
    assert caught.value is error and not calls
    del rx._readers
    error = RuntimeError('selected packet reader')

    def packet(audio):
        assert audio is AUDIO
        raise error

    rx._p3_packet = packet
    rx._scan = lambda audio, readers: readers[0](audio)
    with pytest.raises(RuntimeError) as caught:
        rx.deep_scan(AUDIO)
    assert caught.value is error


@pytest.mark.parametrize('raises', [False, True])
@pytest.mark.parametrize('upgrade', [False, True])
def test_existing_receiver_scan_frame_scopes_and_restores_after_success_or_error(raises, upgrade):
    session = _Session()
    rx = session.rx
    old = (rx._scan_origin, rx._tracked_only, getattr(rx, '_p3_target_row0', None))
    original_scan = rx._scan
    expected = identities(legacy_readers(rx)[1:] if upgrade else legacy_readers(rx)[:1])
    audio, trace = np.zeros(0, np.float32), []
    error = RuntimeError('scoped scan')

    def scan(samples, readers):
        assert samples is audio and identities(readers) == expected
        trace.append((rx._scan_origin, rx._tracked_only, rx._p3_target_row0))
        if raises:
            raise error

    with patch.object(rx, '_scan', scan):
        if raises:
            with pytest.raises(RuntimeError) as caught:
                onair._scan_frame(rx, audio, 7300, tracked_only=True,
                                  upgrade=upgrade, p3_row0=7800)
            assert caught.value is error
        else:
            onair._scan_frame(rx, audio, 7300, tracked_only=True,
                              upgrade=upgrade, p3_row0=7800)
    assert trace == [(7300, True, 7800)]
    assert (rx._scan_origin, rx._tracked_only, rx._p3_target_row0) == old
    assert identities([rx._scan]) == identities([original_scan])
    assert '_scan' not in vars(rx)
    assert session.events == [] and session.received == b'' and rx.count == 0


@pytest.mark.parametrize('route', ['ordinary', 'admitted', 'expected', 'consumed', 'connecting', 'probe'])
def test_existing_receiver_p4_probe_and_input_loss_routing_order(route):
    session = _Session()
    rx, host = session.rx, session.host
    old = (rx._scan_origin, rx._tracked_only, getattr(rx, '_p3_target_row0', None))
    had_target_row0 = '_p3_target_row0' in vars(rx)
    trace, audio = [], np.zeros(0, np.float32)
    p4 = SimpleNamespace(admitted=route in ('admitted', 'consumed', 'connecting'),
                         expected=route == 'expected', consumed=None)
    peer = SimpleNamespace(tx_end=1900, slot=2,
                           p4_probe=SimpleNamespace(requested=route == 'probe'),
                           _observe_input_loss=lambda where: trace.append(('loss', where)))

    def p4_read(samples, origin, **kwargs):
        assert samples is audio and origin == 1700
        assert kwargs == {'carrier_end': 1900, 'whole': True}
        trace.append('p4')
        if route == 'consumed':
            p4.consumed = object()

    def scan(samples, readers):
        assert samples is audio
        assert identities(readers) == identities(legacy_readers(rx)[:1])
        trace.append('scan')

    with patch.object(host, 'peer', peer), patch.object(rx, 'p4', p4), \
            patch.object(rx, '_p4_read', p4_read), patch.object(rx, '_scan', scan), \
            patch.object(host.arq, 'state', arq.State.CONNECTING if route == 'connecting'
                         else arq.State.CONNECTED):
        onair._scan_frame(rx, audio, 1700)
    expected = [('loss', 'frame-scan')]
    if route in ('admitted', 'expected', 'consumed'):
        expected.append('p4')
    if route in ('ordinary', 'admitted', 'connecting'):
        expected.append('scan')
    assert trace == expected
    assert (rx._scan_origin, rx._tracked_only, getattr(rx, '_p3_target_row0', None)) == old
    if route in ('expected', 'consumed', 'probe'):
        # These routes return before entering the P3 scope in _scan_frame.
        assert ('_p3_target_row0' in vars(rx)) == had_target_row0
    assert session.events == [] and session.received == b''


def test_input_loss_invalidates_existing_raster_before_selected_reader():
    session, grid = _Session(), corroborated()
    rx, peer = session.rx, session.host.peer
    trace = []

    def loss(where):
        trace.append(where)
        grid.note_capture_loss(LOST)

    def reader(audio):
        assert grid.p3_control_refusal(32) is not None
        trace.append('reader')
        return None

    with patch.object(peer, '_observe_input_loss', loss, create=True), \
            patch.object(rx, '_p3_packet', reader):
        onair._scan_frame(rx, np.zeros(onair.FS // 2, np.float32), 1000)
    assert trace == ['frame-scan', 'reader']
    assert grid.p3_reply_phase() is None
    assert session.events == [] and session.received == b'' and rx.count == 0


@pytest.mark.parametrize('condition,expected', [
    ('ordinary', ['p3']), ('short', []), ('duplicate', []),
    ('tracked-newer', ['p3']), ('closed', []), ('other-protocol', ['p2']),
])
def test_existing_receiver_guards_and_protocol_ownership_keep_clean_fingerprint(condition, expected):
    session = _Session(Protocol.PACTOR2 if condition == 'other-protocol' else Protocol.PACTOR3)
    rx, host = session.rx, session.host
    audio = np.zeros(onair.FS // 2 - (condition == 'short'), np.float32)
    trace, sent = [], list(host.peer.sent)
    rx.frame_seen = condition in ('duplicate', 'tracked-newer')
    if condition == 'closed':
        host.arq.state = arq.State.DISCONNECTED

    def read(name):
        def packet(samples):
            assert samples is audio
            trace.append(name)
            return None
        return packet

    with patch.object(rx, '_p3_packet', read('p3')), \
            patch.object(rx, '_p2_packet', read('p2')), \
            patch.object(rx, '_p1_packet', read('p1')):
        onair._scan_frame(rx, audio, 6100, tracked_only=condition == 'tracked-newer')
    assert trace == expected
    assert session.events == [] and session.received == b'' and rx.count == 0
    assert host.peer.sent == sent


def test_entry_pending_selection_keeps_existing_host_and_emission_ownership():
    host, keyed = granted()
    rx = onair._SessionRx(host, tag='FIRST')
    packet, queued = host.arq._inflight, bytes(host.arq._outbuf)
    emitted = list(keyed.packets)
    expected = identities(legacy_readers(rx)[:1])
    calls = []
    audio = np.zeros(0, np.float32)
    with patch.object(rx, '_scan', lambda samples, readers: calls.append((samples, readers))):
        onair._scan_frame(rx, audio, 4100, tracked_only=True, p3_row0=4600)
    assert calls[0][0] is audio and identities(calls[0][1]) == expected
    assert host.protocol is Protocol.PACTOR3 and host.arq.entry_pending
    assert host.arq._inflight is packet and bytes(host.arq._outbuf) == queued
    assert keyed.packets == emitted and rx.count == 0 and not rx.frame_seen


@pytest.mark.parametrize('condition', ['duplicate', 'overdue'])
def test_selected_packet_cannot_deliver_duplicate_or_precommand_frame(condition):
    session = _Session()
    rx, host = session.rx, session.host
    origin, start = 10000, 1000
    absolute = origin + start
    if condition == 'duplicate':
        rx._p3_delivered_at = absolute
    else:
        host.peer.raster = SimpleNamespace(_p3_command_row0=absolute + 2 * rxfront.SPS)
    before = (rx._p3_delivered_at, rx._p3_row0, rx._p3_expected_levels)
    event = rxfront.Event(start / onair.FS, 'packet', 'selector negative',
                         protocol=Protocol.PACTOR3, start=start,
                         packet=(2, 0x23, b'undeliverable', True))
    trace = []

    def packet(audio):
        trace.append('p3')
        return event, 'untimed seam'

    with patch.object(rx, '_p3_packet', packet):
        onair._scan_frame(rx, np.zeros(onair.FS // 2, np.float32), origin,
                          tracked_only=True)
    assert trace == ['p3']
    assert (rx._p3_delivered_at, rx._p3_row0, rx._p3_expected_levels) == before
    assert session.events == [] and session.received == b'' and rx.count == 0
    assert not rx.frame_seen and host.peer.sent == []
