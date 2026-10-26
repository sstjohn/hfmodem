# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real turn controls, with explicitly synthetic qualifying payload ACKs."""
import hashlib
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, hilbert
from hfmodem.tests.evidence import WORKING, CAPTURES, require_input
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Emission, Role
from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State, Proposal
from hfmodem.tests.shrike.test_p4_part_a_1011 import setup
from hfmodem.tests.shrike.test_p4turn import FS, word
from hfmodem.tests.shrike.test_p4_bare_y27_0929 import K0003, ROOT as K0003_ROOT


def synthetic_payload_ack(m, a, p):
    """Synthetic extension: the recording demonstrates turn, not mail completion."""
    assert a.tx_acked_bytes == 0
    m.keyed()
    m.observe(word('X27', p.stream_sample+44880, responds_to=p.stream_sample, extent=.245),
              now=p.stream_sample+58000)
    assert a.tx_acked_bytes == len(p.payload)


def test_real_kiwi_long_y27_checked_body_payload_and_synthetic_ack():
    path = require_input(WORKING / K0003.relative_to(K0003_ROOT / 'working'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '35070a38bfd877ee7fb7a91007809fb894b2a65b17405f1b1553ea4ea83fd4d4'
    _, pcm = wavfile.read(path, mmap=True)
    base, end, rate = 1529.406, 1537.2, 11998.90779
    x = resample_poly(np.asarray(pcm[round(base*rate):round(end*rate)], np.float64)/32768, 4, 1)
    audio = (hilbert(x)*np.exp(-2j*np.pi*910*np.arange(x.size)/FS)).real
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=3.)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, FS//2, .330, 3.75, 3.)
    long = rx._packet(audio[:round((1533.23-base)*FS)], 0)
    assert long is not None and long.kind == Kind.LONG and long.frame.status.byte == 0x42
    assert long.frame.data.startswith(b'FF\r')
    sid = b'[Pat-1.0.0-B2FHM$]\r;FW: W9SSJ\r;PR: 12345678\r'
    m, a = setup(sid, breakin='body')
    m.observe(RegisteredObservation(long, Registered.NEW), now=long.stream_sample+158880)
    due = long.stream_sample+163200
    p = m.decide(due-m.lead)
    assert (p.kind, p.word, p.stream_sample) == (Kind.WORD, 'Y27', due)
    m.keyed()
    rx.sync_peer_watch(m.peer_watch)
    rx.note_emission(Emission(Kind.WORD, Role.ISS, due, .240))
    origin, finish = round((1533.65-base)*FS), round((1535.08-base)*FS)
    regs = rx.read_window(audio[origin:finish], origin, whole=True)
    for r in regs: m.observe(r, now=finish)
    grant = next(r.observation for r in regs if r.observation.kind == Kind.BODY)
    assert grant.frame.status.byte == 0x40 and abs(grant.stream_sample-round((1534.261-base)*FS)) <= 48
    p = m.decide(grant.stream_sample+44160-m.lead)
    assert (p.kind, p.status, p.payload, p.stream_sample) == (Kind.BODY, 0, sid[:31], grant.stream_sample+44160)
    assert m.peer_watch is None
    synthetic_payload_ack(m, a, p)
    print({'source': str(path), 'long': long.stream_sample, 'request': due, 'grant': grant.stream_sample,
           'payload': p.stream_sample, 'ACK': 'synthetic'})


def test_real_g_short_grant_v27_payload_and_synthetic_ack():
    path = require_input(CAPTURES/'onair-20261005T002601583605Z-6e2ff851-3bcd-4d2e-b30c-e7258f59cac1'/'p4-probe.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'b70387648555cb822d734c9fd674d3b2392012df5bcd328b09dcc0be56866ff7'
    tape = json.loads(path.read_text())
    grant = next(e for e in tape['emissions'] if e['audio_start'] == 2011832)
    assert grant['status'] == 0x40 and grant['payload'] == ''
    sid = b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;'
    m, a = setup(sid)
    m._next = Proposal(Kind.BODY, 2011832, status=0x40, state=State.IRS_BREAKIN)
    m.keyed()
    observed = [r for r in tape['heard'] if r.get('responds_to') == 2011832 and r.get('word') == 'V27']
    assert [r['onset_sample'] for r in observed] == [2056457, 2056453]
    assert abs(observed[-1]['onset_sample']-round(42.843*FS)) <= 12
    provisional = None
    for row in observed:
        r = word('V27', row['onset_sample'], partial=row['provisional'], responds_to=2011832,
                 outcome=Registered(row['outcome']), replaced=provisional)
        m.observe(r, now=row['received_at_sample'])
        provisional = r.observation
    p = m.decide(2071832-m.lead)
    assert (p.kind, p.state, p.status, p.payload, p.stream_sample) == (Kind.BODY, State.ISS_DATA, 0x40, sid, 2071832)
    assert m.peer_watch is None and any(e['event'] == 'turn' for e in m.link_events)
    synthetic_payload_ack(m, a, p)
