# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Charged-time link admission: a cached chirp must key its original slot."""
import numpy as np
import pytest

from hfmodem.shrike import onair, p4chirp, p4breakintx, p4robusttx
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4turn import P4Turn, State
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx


@pytest.mark.parametrize('deadline,name', [
    (38.766, 'U27'), (54.414, 'X27'), (81.263, 'U27'),
    (29.416, 'X27'), (62.516, 'U27'),
])
@pytest.mark.parametrize('late_source', [False, True])
def test_first_chirp_keys_admitting_slot_with_charged_misses(monkeypatch, deadline, name, late_source):
    anchor = round((deadline-3.75)*FS)
    audio = np.zeros(round((deadline+5)*FS), np.float32)
    for index, answer in enumerate(('X27', 'X27', name)):
        at = anchor+index*60000+round(.930*FS)
        wave = p4robusttx.robust_word(answer)
        audio[at:at+len(wave)] = wave
    stream = Stream(audio)
    stream.wall = 0
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.transmit = True
    stream.key_notice = 1796
    stream.tx_latency_n = 144
    stream._blk = 128
    stream._callback_timing = [(0, 128, 0., 0), (128, 128, .0001, 0),
                               (256, 128, .0002, 0)]
    stream.clamp_late = lambda at: max(0, stream.sample_now()+stream.key_notice-at)
    tx = Tx(stream)
    tx.settle = .037
    lead = onair._prekey_lead(stream, round(tx.settle*FS))
    assert lead == 2276
    tx.p4_link_options.p4_ladder = tx.p4_link_options.p4_data = 'chirp'
    tx.p4_link_options.p4_link_timeout = 8
    tx.p4_probe.emissions = [dict(audio_start=anchor, audio_end=anchor+36367)]
    stream.pos = anchor+36367
    source = b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: 61139009\r' + b'next field'*8
    if not late_source:
        tx.p4_link_host.arq.on_host_data(source)
    original = len(source)
    machines, misses, confirmations = [], [], []
    critical = False

    def seed(m, *a, **kw):
        machines.append(m)
        m._word_yield = name == 'X27'
        m.start(anchor)
        assert m.keyed(anchor).value == 13
    monkeypatch.setattr(P4Turn, 'seed_opening', seed)

    def charge(seconds):
        stream.wall = stream.sample_now()+round(seconds*FS)

    for module, method, seconds in ((p4chirp, 'entry_packet', .106/16),
                                    (p4breakintx, 'changeover_packet', .0085/5)):
        build = getattr(module, method)
        def timed(*a, _build=build, _seconds=seconds, **kw):
            if late_source and critical:
                pytest.fail('waveform builder entered decision-to-key window')
            result = _build(*a, **kw)
            if a[0]:
                misses.append((stream.sample_now(), _build.__name__, bytes(a[0])))
                charge(_seconds)
            return result
        monkeypatch.setattr(module, method, timed)
    rx = tx.sessrx.p4
    for method, seconds in (('read_partial', .003), ('read_window', .004)):
        read = getattr(rx, method)
        def timed_read(*a, _read=read, _seconds=seconds, **kw):
            nonlocal critical
            records = _read(*a, **kw)
            charge(_seconds)
            for r in records:
                o = r.observation
                if o.kind == Kind.WORD and not o.provisional:
                    confirmations.append((o.stream_sample, stream.sample_now()))
                    if late_source and abs(o.stream_sample-(anchor+120000+round(.930*FS))) < 480:
                        if not tx.p4_link_host.arq._buffer_raw:
                            tx.p4_link_host.arq.on_host_data(source)
                            critical = True
            return records
        monkeypatch.setattr(rx, method, timed_read)
    key = tx._tx
    def emit(audio, what):
        nonlocal critical
        assert stream.sample_now() <= tx.raster.boundary(tx.slot)-lead
        key(audio, what)
        stream.pos = max(stream.pos, tx.tx_end)
        stream.wall = max(stream.wall, stream.pos)
        critical = False
    tx._tx = emit
    refuse = P4Turn.refused
    def deferred(m, reason):
        nonlocal critical
        refuse(m, reason)
        critical = False
    monkeypatch.setattr(P4Turn, 'refused', deferred)
    onair._run_p4_link(tx, stream, onair._MasterGrid(anchor, 60000, 0,
        packet_n=36000, cs_n=12000, d_max_n=4800))
    chirps = [e for e in tx.p4_probe.emissions if e.get('status', 0) in range(0x60, 0x64)]
    refusals = [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']
    expected = anchor+3*60000
    first_key = expected+180000 if late_source else expected
    assert chirps and chirps[0]['proposed_at'] == first_key, (chirps, refusals)
    assert chirps[0]['keyed_at'] == first_key
    assert chirps[0]['status'] == (0x60 if name == 'X27' else 0x61)
    if late_source:
        assert [(r['at'], r['reason']) for r in refusals] == [(expected, 'waveform not prepared')]
    else:
        assert not refusals
    admission = next(now for at, now in confirmations if abs(at-(expected-round(.320*FS))) < 480)
    assert not [row for row in misses if admission <= row[0] <= expected-lead]
    assert any(row[1] == 'entry_packet' for row in misses)
    assert len({(builder, field) for _, builder, field in misses}) >= 4
    from collections import Counter
    counts = Counter((builder, field) for _, builder, field in misses)
    assert all(count == (16 if builder == 'entry_packet' else 5)
               for (builder, field), count in counts.items())
    m = machines[0]
    assert original == tx.p4_link_host.arq._buffer_raw
    assert len(tx.p4_link_host.arq._outbuf)+len(m.outbox.inflight) == original


def test_r2_cache_keypath_never_builds(monkeypatch):
    test_first_chirp_keys_admitting_slot_with_charged_misses(
        monkeypatch, 54.414, 'X27', late_source=True)


@pytest.mark.parametrize('is_request', [False, True])
def test_recorded_replay_checks_device_admission_for_every_emission(is_request):
    import ast
    import inspect
    from types import SimpleNamespace
    from hfmodem.tests.shrike import p4_chirp_deadline_support

    source = ast.parse(inspect.getsource(p4_chirp_deadline_support.replay))
    emit = next(node for node in ast.walk(source)
                if isinstance(node, ast.FunctionDef) and node.name == 'emit')
    admission_lead = 1796
    now = 100000-admission_lead+1
    stream = SimpleNamespace(sample_now=lambda: now, pos=now, wall=now,
                             key_notice=1796)
    tx = SimpleNamespace(raster=SimpleNamespace(boundary=lambda slot: 100000),
                         slot=0, tx_end=111712)
    keyed = []
    scope = dict(stream=stream, tx=tx, lead=2640,
                 key=lambda *args: keyed.append(args))
    exec(compile(ast.Module(body=[emit], type_ignores=[]), '<replay emit>', 'exec'), scope)
    with pytest.raises(AssertionError):
        scope['emit'](None, 'P4 LINK word X27' + (' repeat-request' if is_request else ''))
    assert not keyed

    stream.sample_now = lambda: 100000-admission_lead
    scope['emit'](None, 'P4 LINK word X27' + (' repeat-request' if is_request else ''))
    assert len(keyed) == 1


def test_1045_recorded_grant_keys_a_body():
    """Both-pass control: bounded recorded prefix, actual receiver association.

    Beyond the changed first chirp this is a conditional cache/receive control,
    not a claim about what the peer would transmit in a changed exchange.
    """
    from hfmodem.tests.shrike.p4_chirp_deadline_support import replay
    result = replay('onair-1001-1045', stop=round(80*FS))
    grant = next(e for e in result['events'] if e['event'] == 'Keyed'
                 and e.get('kind') == 'body' and e.get('state') == 'ISS_DATA_ACK')
    assert abs(grant['at']/FS-76.26295833333333) < 1/FS
    assert grant['status'] == 0
    source = bytes.fromhex(result['source'])
    assert bytes.fromhex(grant['payload']) == source[:31]
    assert result['buffered'] == len(source)
    assert not [e for e in result['events'] if e['event'] == 'Refused'
                and e.get('kind') == 'body' and e['reason'] == 'waveform not prepared']


def driver_prepare(m, cache):
    import ast
    import inspect
    from types import SimpleNamespace
    from hfmodem.shrike import p4shorttx, p4waveforms
    tree = ast.parse(inspect.getsource(onair._run_p4_link))
    definitions = [node for node in tree.body[0].body
                   if isinstance(node, ast.FunctionDef) and node.name in ('prepare', 'cache_key')]
    scope = dict(machine=m, cache=cache, host=SimpleNamespace(arq=m.outbox.arq),
                 Kind=Kind, p4chirp=p4chirp, p4breakintx=p4breakintx, p4shorttx=p4shorttx,
                 cache_key=p4waveforms.cache_key, prepare_waveforms=p4waveforms.prepare,
                 word_render='off')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<driver prepare>', 'exec'), scope)
    scope['prepare']()


def test_r2_cache_form_roundtrip(monkeypatch):
    from hfmodem.shrike import p4shorttx, p4chirprx, p4waveforms
    from hfmodem.shrike.p4turn import Proposal, State
    from hfmodem.tests.shrike.test_p4turn import machine
    m, arq = machine(data='body')
    arq.on_host_data(b'A'*21+b'\x1e'+b'B'*8+b'\x1c'+b'C'*60)
    raw = arq._buffer_raw
    inflight = m.outbox.transfer(40)
    queued = bytes(arq._outbuf)
    m._data_packet = Proposal(Kind.NORMAL, FS, payload=inflight, status=2, state=State.ISS_DATA)
    body = p4waveforms.prefix(inflight, 31)
    m._next = m._data_packet
    cache = {}
    driver_prepare(m, cache)
    normal_key = p4waveforms.cache_key(Kind.NORMAL, payload=inflight, status=2)
    kept_normal = cache[normal_key]
    m._next = Proposal(Kind.BODY, 2*FS, payload=body, status=0, state=State.ISS_GRANT)
    driver_prepare(m, cache)
    assert normal_key in cache
    assert cache[normal_key] is kept_normal
    body_keys = [p4waveforms.cache_key(Kind.BODY, payload=body, status=s) for s in (*range(4), 0x40)]
    kept_body = [cache[key] for key in body_keys]
    from hfmodem.shrike import p4breakin, p4rx
    from hfmodem.tests.shrike.test_p4shorttx import read_back
    for key in body_keys:
        wave = cache[key]
        z = p4rx.audio_to_baseband(np.r_[wave, np.zeros(FS//10)], FS)
        frame = p4breakin.decode_body(z, first_training_start=432*16)
        assert frame is not None and frame.status.byte == key[5]
        assert frame.data == body.ljust(31, b'\x1e')
    for status in range(4):
        key = p4waveforms.cache_key(Kind.NORMAL, payload=inflight, status=status)
        frame = read_back(cache[key])
        assert frame is not None and frame.status.byte == status
        assert frame.data == inflight.ljust(40, b'\x1e')
    m.data_kind = Kind.CHIRP
    chirp = p4waveforms.prefix(inflight, 22)
    assert len(chirp) == 21 and not chirp.endswith(b'\x1c')
    m._next = Proposal(Kind.CHIRP, 3*FS, payload=chirp, status=0x61, state=State.ISS_DATA)
    m._chirp_origin = 3*FS
    driver_prepare(m, cache)
    assert cache[normal_key] is kept_normal
    assert all(cache[key] is wave for key, wave in zip(body_keys, kept_body))
    for status in range(0x60, 0x64):
        for repeat in (False, True):
            for slot in (0, 1):
                key = p4waveforms.cache_key(Kind.CHIRP, payload=chirp, status=status,
                                          repeat=repeat, chirp_slot=slot)
                r = p4chirprx.decode_packet(cache[key])
                assert r is not None and r.frame.status.byte == status
                assert r.marker == 8+repeat and r.rank0_upper == bool(slot)
                assert r.frame.data[:len(chirp)] == chirp
    m.refused('deadline')
    pending_key = p4waveforms.cache_key(Kind.CHIRP, payload=chirp, status=0x61, chirp_slot=1)
    pending_wave = cache[pending_key]
    m.outbox.requeue()
    m.data_kind = Kind.BODY
    driver_prepare(m, cache)
    assert cache[pending_key] is pending_wave
    assert arq._buffer_raw == raw and bytes(arq._outbuf) == inflight+queued
    assert len([key for key in cache if key[4]]) <= 5*(5+4+16)
    def forbidden(*a, **kw):
        pytest.fail('rebuilt a prepared current/next waveform')
    for module, method in ((p4chirp, 'entry_packet'), (p4breakintx, 'changeover_packet'),
                           (p4shorttx, 'short_packet')):
        monkeypatch.setattr(module, method, forbidden)
    arq.on_host_data(b'tail mutation')
    driver_prepare(m, cache)
    m._next = m._data_packet = None
    driver_prepare(m, cache)
    assert not [key for key in cache if key[0] == Kind.CHIRP]


@pytest.mark.parametrize('arm,stop,expected,status', [
    ('1042', 40, 1860780, 0x61),
    ('1045', 56, 2611874, 0x60),
    ('1103', 31, 1411958, 0x60),
    ('1103', 64, 3000755, 0x61),
])
def test_recorded_chirp_prefix(arm, stop, expected, status):
    """The second 1103 turn is conditional on the earlier changed exchange."""
    from hfmodem.tests.shrike.p4_chirp_deadline_support import replay
    result = replay('onair-1001-'+arm, stop=round(stop*FS))
    chirps = [e for e in result['emissions'] if e.get('status') == status]
    keyed = next((e for e in chirps if e['proposed_at'] >= expected-480), None)
    assert keyed is not None, (chirps, result['events'])
    assert keyed['proposed_at'] == keyed['keyed_at'] == expected
    assert not [e for e in result['events'] if e['event'] == 'Refused'
                and e['at'] == expected]
    original = next(e for e in result['recorded_emissions']
                    if e.get('status') == status and e['audio_start'] >= expected)
    assert keyed['payload'] == original['payload']
    assert not [m for m in result['misses'] if expected-result['lead']-1200 <= m['at'] <= expected]


def test_1056_unread_normal_cannot_start_a_queued_turn():
    from hfmodem.tests.shrike.p4_chirp_deadline_support import replay
    result = replay('onair-1001-1056', stop=round(54*FS))
    unread = [h for h in result['heard'] if h['kind'] == Kind.NORMAL
              and abs(h['onset_sample']-round(50.993*FS)) <= 48
              and 'status' not in h]
    assert unread
    assert not [e for e in result['events'] if e['event'] == 'Keyed'
                and e.get('state') == State.IRS_BREAKIN
                and abs(e['at']-round(51.913*FS)) <= 48]
