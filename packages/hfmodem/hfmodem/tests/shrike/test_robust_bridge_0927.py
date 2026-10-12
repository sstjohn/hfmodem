"""The recorded robust-call P2 bridge, including failure and stop handling."""
from hfmodem.tests.evidence import require_input
from pathlib import Path
import numpy as np
import pytest
from hfmodem.shrike import compress, onair, p2rx, p3acquire, pactor2, placement, session, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_robust_answer_0923 import _air, _keyed, _k4mck
from hfmodem.tests.shrike.test_p3_offer import Keyed

FS = 48000
SLOT = 60000
FIELD = bytes.fromhex('0f0f2a3ab44512320f8f87c7c3e31958eb')


def bridge_host():
    peer = Keyed()
    host = PtcHost(peer=peer, mycall='K4MCK')
    host.arq.on_host_connect('K4MCK', '%AI4Y')
    host.arq.on_rx_robust_answer(0)
    host.enter_robust_bridge()
    return host, peer


def test_transition_from_p2():
    host, peer = bridge_host()
    assert host.protocol is spec.Protocol.PACTOR2
    assert host.take_bridge_grant()
    assert host.protocol is spec.Protocol.PACTOR3
    assert host.arq.speed_level == 1 and host.arq.entry_pending
    assert host.arq.tx_seq == 2 and host._grant_taken and peer.p3
    assert host.arq._inflight.status == 0x1a
    count = len(peer.p3)
    assert not host.take_bridge_grant()
    assert len(peer.p3) == count


def test_terminal_intent_blocks_transition():
    host, peer = bridge_host()
    host.arq._qrt_pending = True
    before = (host.arq.speed_level, host.arq.entry_pending, host.arq.tx_seq)
    assert not host.take_bridge_grant()
    assert host.protocol is spec.Protocol.PACTOR2 and host.arq.terminal_pending
    assert (host.arq.speed_level, host.arq.entry_pending, host.arq.tx_seq) == before
    assert not peer.p3 and not host._grant_taken


def test_recorded_replay(tmp_path, monkeypatch):
    peer = _k4mck()
    for k in range(4):
        peer[k*SLOT:k*SLOT+round(.88*FS)] = 0
    session.write_wav(str(tmp_path/'peer.wav'), np.r_[peer, np.zeros(8*SLOT)])
    from hfmodem.core import levels
    keyed, timing, answers = [], [], []
    key, read = onair.RadioTx._tx, onair._SessionRx._p3_cs

    def capture_key(tx, audio, what, *args, **kwargs):
        leveled = levels.at_drive(audio, tx.drive)
        trim = int(np.flatnonzero(abs(leveled) > .02*max(abs(leveled)))[0])
        result = key(tx, audio, what, *args, **kwargs)
        assert not tx.refused
        keyed.append((tx.slot, what, len(audio)))
        boundary = None if tx.raster is None else tx.raster.boundary(tx.slot)
        timing.append((boundary, tx.tx_audio_start, trim))
        return result

    def capture_answer(rx, *args, **kwargs):
        before = rx._p3_answer_at
        hint = rx._p3_preferred_phase
        result = read(rx, *args, **kwargs)
        if before is None and rx._p3_answer_at is not None:
            answers.append((hint, rx._p3_answer_at, rx._p3_preferred_phase, result.cs))
        return result

    monkeypatch.setattr(onair.RadioTx, '_tx', capture_key)
    monkeypatch.setattr(onair._SessionRx, '_p3_cs', capture_answer)
    log = _air(tmp_path, '--after-robust', 'p2', '--mycall', 'K4MCK')
    (tmp_path/'replay.log').write_text(log)
    assert [(s, w.split()[0]) for s,w,n in keyed[:4]] == [(0,'connect->K7ABC'),(1,'P2'),(2,'P2'),(3,'SL1')], log
    assert 'phase-delay=18.75ms' in keyed[1][1]
    assert f'field={FIELD.hex()}' in keyed[1][1]
    assert 'status=0x1a' in keyed[3][1]
    assert 'RX P3 CS4 SPEED-UP' in log
    # On the emitted timeline, distinguish the untrimmed render origin,
    # trimmed audio onset, and reference pulse. The trim is measured above.
    for boundary, onset, trim in timing[1:3]:
        assert abs((onset-trim)-boundary) <= 1
        assert onset-boundary == trim
        assert onset+pactor2.pulse_lead()-trim-boundary == round(.01875*FS)
    boundary, onset, _ = timing[3]
    assert onset-boundary == round(.004875*FS)
    hint, measured, remaining, cs = answers[0]
    assert hint is not None and cs == 3  # the first actual P3 read is CS4
    assert abs((hint-round(.005*FS)) % SLOT / FS-.971) <= .0025
    assert abs(measured % SLOT / FS-.975) <= .0025
    assert remaining is None


@pytest.mark.parametrize('call', ['w9ssj', 've3kpg', 'kb5lzk', 'w9ssj-10', 've3kpg-15'])
def test_pmc_full_field_has_no_spurious_text(call):
    from hfmodem.shrike.robust import Bridge
    payload = Bridge(call, 0).payload
    assert len(payload) == 14
    assert compress.decompress(payload, 6, spec.Protocol.PACTOR2) == (call+'\r').encode()


def _bridge_peer(tmp_path, entries):
    from hfmodem.tests.shrike.test_robust_answer_0923 import _burst
    x = np.zeros((len(entries)+12)*SLOT)
    answer = _burst(spec.CONTROL_SIGNALS[0], bits=17)
    at = round(.991*FS)
    x[at:at+len(answer)] = answer
    for slot, entry in enumerate(entries, 1):
        if entry is None:
            continue
        wave = pactor2.control_signal(entry, swapped=bool((slot-1)%2))
        at = slot*SLOT+round(.971*FS)-pactor2.pulse_lead()
        x[at:at+len(wave)] = wave*.15
    session.write_wav(str(tmp_path/'peer.wav'), x)


@pytest.mark.parametrize('entries,force,p3slot', [
    ([0,6],None,3), ([6,6],None,3), ([6,0,0,0],None,None),
    ([6,None,None,None],None,None), ([0]*4,None,None),
    ([None]*4,None,None), ([0]*4,2,3),
])
def test_bounded_bridge_orders(tmp_path, monkeypatch, entries, force, p3slot):
    _bridge_peer(tmp_path, entries)
    keyed = _keyed(monkeypatch)
    flags = ['--after-robust','p2','--robust-p2-max','4']
    if force is not None:
        flags += ['--robust-p3-without-grant',str(force)]
    log = _air(tmp_path, *flags)
    p3 = [(s,w) for s,w,n in keyed if 'ENTRY' in w]
    if p3slot is None:
        assert not p3, log
        assert 'P2 call packet never taken: 4 cycles' in log
    else:
        assert p3 and p3[0][0] == p3slot, log
        assert ('FORCED P3 ENTRY (no entry 6)' in log) == (force is not None)
    fields = [w.split('field=')[1].split()[0] for s,w,n in keyed if 'ROBUST BRIDGE' in w]
    assert len(set(fields)) == 1
    assert len(fields) == (p3slot-1 if p3slot is not None else 4)


def test_p2_phase_applies_only_in_bridge(monkeypatch):
    from hfmodem.core import levels
    host, _ = bridge_host()
    tx = onair.RadioTx.__new__(onair.RadioTx)
    tx.drive = .2
    tx.host = host
    tx.p2_phase_delay_n = 900
    wave = pactor2.data_burst(FIELD, pactor2.PATHS[1])
    leveled = levels.at_drive(wave, tx.drive)
    trim = np.flatnonzero(abs(leveled) > .02*max(abs(leveled)))[0]
    seen = []
    tx._tx = lambda audio, what, **kw: seen.append(kw['lead_n'])
    tx._tx_p2(wave,'bridge')
    assert -seen[-1] + pactor2.pulse_lead()-trim == 900
    host.protocol = spec.Protocol.PACTOR3
    tx._tx_p2(wave,'outside bridge')
    assert -seen[-1] + pactor2.pulse_lead()-trim == 0


def test_noise_does_not_grant():
    from hfmodem.shrike.robust import Bridge
    rng = np.random.default_rng(927)
    bridge = Bridge('K4MCK',round(.991*FS))
    for slot in range(100):
        x = rng.normal(0,.01,round(.40*FS))
        granted, _ = bridge.read(x,slot*SLOT+round(.82*FS),slot*SLOT,slot)
        assert not granted
    assert all(h['entry'] is None for h in bridge.heard)


def test_recorded_no_answer_windows():
    from hfmodem.shrike.robust import Bridge
    path = Path(__file__).parent/'fixtures/k4mck-0925-robust-no-answer.wav'
    require_input(path)
    x = session.load_wav(str(path),FS)
    bridge = Bridge('K4MCK',round(.991*FS),offset_hz=82)
    for slot in range(18):
        audio = x[slot*16800:(slot+1)*16800]
        grant, report = bridge.read(audio,slot*SLOT+round(.9*FS),slot*SLOT,slot)
        assert not grant and report == 'silence'


def _traffic_peer():
    x = np.zeros(32*SLOT)
    a = _k4mck()
    x[:len(a)] = a
    path = Path(__file__).parent/'fixtures/k4mck-0925-kiwi-robust-p3-traffic.wav'
    require_input(path)
    tail = p3acquire.compensate(session.load_wav(str(path),FS),82)
    at = round((285-281.431)*FS)
    x[at:at+len(tail)] = tail
    for k in range(6):
        x[k*SLOT:k*SLOT+round(.88*FS)] = 0
    for k in range(6,13):
        x[k*SLOT+round(.59*FS):k*SLOT+round(.87*FS)] = 0
    x[13*SLOT:] = 0
    return x


def test_continuation_banner_and_reply_clock(tmp_path):
    import re
    session.write_wav(str(tmp_path/'peer.wav'),_traffic_peer())
    log = _air(tmp_path,'--after-robust','p2','--mycall','K4MCK','--hold','16')
    assert 'Trimode 1.4.4.0 Welcome to AI4Y' in log
    assert 'CMS via AI4Y >' in log
    # Compare the codeword phase, not the shaped carrier's first sample.
    packet = int(re.search(r'CRC packet @ (\d+)',log)[1])
    control = int(re.search(r'emitted control phase (\d+)',log)[1])
    assert abs((control-packet)/FS-.909) <= .002


def test_peer_grants_only_a_decoded_callsign_then_sends_banner(tmp_path, monkeypatch):
    from hfmodem.shrike import p3rx
    source = _traffic_peer()
    peer = source.copy()
    peer[SLOT:] = 0
    session.write_wav(str(tmp_path/'peer.wav'),peer)
    key = onair.RadioTx._tx
    seen = []
    def respond(tx,audio,what,*a,**kw):
        if 'ROBUST BRIDGE' in what:
            decoded = p2rx.decode_expected_burst(np.pad(audio,(2400,4800)),FS)
            assert decoded is not None
            _,path,field,_ = decoded
            text = compress.decompress(field[:-3],(field[-3]>>2)&7,spec.Protocol.PACTOR2)
            seen.append((path.level+1,field[-3],text))
            entry = 6 if text == b'w9ssj\r' and field[-3]&3 == 1 else 0
            reply = pactor2.control_signal(entry,swapped=bool((len(seen)-1)%2))*.15
            at = tx.slot*SLOT+round(.971*FS)-pactor2.pulse_lead()
            tx.live.audio[at:at+len(reply)] = reply
        elif 'SL1 ENTRY' in what and len(seen)>=2:
            decoded = p3rx.decode_p3_packets(np.pad(audio,(2400,4800)))
            assert any(p.status == 0x1a for p in decoded.packets)
            at = tx.slot*SLOT
            tail = source[3*SLOT:13*SLOT]
            tx.live.audio[at:at+len(tail)] = tail
        return key(tx,audio,what,*a,**kw)
    monkeypatch.setattr(onair.RadioTx,'_tx',respond)
    log = _air(tmp_path,'--after-robust','p2','--hold','16')
    assert seen == [(2,0x19,b'w9ssj\r')]*2
    assert 'Trimode 1.4.4.0 Welcome to AI4Y' in log
    assert 'CMS via AI4Y >' in log


def test_unanswered_p3_entry_listens_without_p1_fallback(tmp_path, monkeypatch):
    _bridge_peer(tmp_path,[6,6])
    keyed = _keyed(monkeypatch)
    log = _air(tmp_path,'--after-robust','p2','--hold','12')
    assert 'robust P3 entry unanswered; listening for the observe window' in log
    assert not any('P1 pkt' in w for _,w,_ in keyed)
    assert sum('SL1 ENTRY' in w for _,w,_ in keyed) == 4


def test_bridge_phase_hint_expires_after_corroborated_p3_acquisition(monkeypatch):
    from hfmodem.tests.shrike.test_entry_answer import _Session
    rx = _Session(entry_pending=True).rx
    hint = 2*SLOT+round(.976*FS)
    rx._p3_preferred_phase = hint
    assert rx._p3_answer_at is None
    audio = _k4mck()[3*SLOT:4*SLOT]
    preferred, search = [], p3acquire.control_signal

    def capture(audio, **kwargs):
        preferred.append(kwargs['preferred_at'])
        return search(audio, **kwargs)

    monkeypatch.setattr(p3acquire, 'control_signal', capture)
    rx.new_cycle()
    assert rx._acquire_answer(audio, 3*SLOT, round(.975*FS)) is None
    assert rx._p3_answer_at is None and rx._p3_preferred_phase == hint
    rx.new_cycle()
    got = rx._acquire_answer(audio, 4*SLOT, round(.975*FS))
    assert got is not None and got.cs == 3
    assert abs(rx._p3_answer_at % SLOT / FS-.975) <= .0025
    assert rx._p3_preferred_phase is None
    assert preferred[:2] == [round(.221*FS)]*2
    # Losing the measured anchor later must not resurrect the bridge prediction.
    rx._p3_answer_at = None
    rx.new_cycle()
    rx._p3_acquisition_search(audio, 5*SLOT, round(.975*FS))
    assert preferred[-1] is None


def _bridge_keys(bridge, cycles):
    tx = onair.RadioTx.__new__(onair.RadioTx)
    tx.seq_sent, tx.breakin_at, tx.refused = [], set(), False
    seen = []
    tx._tx_p2 = lambda audio, what: seen.append((audio, what))
    for n in range(cycles):
        bridge.cycles = n
        onair._key_robust_bridge(tx, bridge)
    return seen


def _marker(audio):
    top = max(p2rx.find_markers(np.r_[np.zeros(4800), audio, np.zeros(9600)], FS, .8),
              key=lambda m: m[4])
    return top[2], top[5]


def test_default_marker_matches_head_render():
    from hfmodem.shrike.robust import Bridge
    bridge = Bridge('W9SSJ', 0)
    seen = _bridge_keys(bridge, 2)
    path = pactor2.PATHS[1]
    field = pactor2.build_field(placement.field_info(bridge.payload, 14, 0x19), path)
    for n, (audio, what) in enumerate(seen):
        assert np.array_equal(audio, pactor2.data_burst(field, path, swapped=bool(n)))
        assert what == f'P2 ROBUST BRIDGE counter=1 field={field.hex()} marker=3 swap={n}'
    assert _marker(seen[0][0]) == (3, False)


def test_inverted_callsign_marker_is_k4mcks():
    from hfmodem.shrike.robust import Bridge
    (_, what), (audio, _) = _bridge_keys(Bridge('W9SSJ', 0, marker_flag='inverted'), 2)
    assert 'P2 ROBUST BRIDGE counter=1 (inverted)' in what and 'marker=2 swap=0' in what
    recorded = _k4mck()[2*SLOT-4800:3*SLOT+4800]
    k4mck = max(p2rx.find_markers(recorded, FS, .8), key=lambda m: m[4])
    assert (k4mck[2], k4mck[5]) == (2, True)
    assert _marker(audio) == (2, True)


def test_k9_opening_precedes_the_callsign_packet():
    from hfmodem.shrike.robust import Bridge
    seen = _bridge_keys(Bridge('W9SSJ', 0, opening='k9'), 3)
    path = pactor2.PATHS[1]
    template = pactor2.build_field(placement.field_info(b'', 14, 0x18), path)
    (first, opening), (_, call), (_, again) = seen
    assert opening == f'P2 ROBUST OPENING counter=0 field={template.hex()} marker=9 swap=0'
    assert call.startswith('P2 ROBUST BRIDGE counter=1 ') and 'marker=3 swap=1' in call
    assert again.startswith('P2 ROBUST BRIDGE counter=1 ') and 'swap=0' in again
    recorded = _k4mck()[SLOT-4800:2*SLOT+4800]
    k4mck = max(p2rx.find_markers(recorded, FS, .8), key=lambda m: m[4])
    assert _marker(first) == (k4mck[2], k4mck[5]) == (9, False)


def test_k9_opening_counts_as_a_bridge_slot(tmp_path, monkeypatch):
    _bridge_peer(tmp_path, [0]*4)
    keyed = _keyed(monkeypatch)
    log = _air(tmp_path, '--after-robust', 'p2', '--robust-p2-max', '4',
               '--robust-p2-opening', 'k9', '--robust-p2-marker-flag', 'inverted')
    p2 = [(s, w) for s, w, n in keyed if w.startswith('P2 ROBUST')]
    assert [s for s, _ in p2] == [1, 2, 3, 4], log
    assert 'OPENING counter=0' in p2[0][1] and 'marker=9 swap=0' in p2[0][1]
    assert all('(inverted)' in w and 'marker=2' in w for _, w in p2[1:])
    assert 'P2 call packet never taken: 4 cycles' in log
