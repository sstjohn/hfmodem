# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.shrike import onair, p4frame, p4turn
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, Observation
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.p4_nak_qam_support import neutral_keys, without_neutral_requests


def exchange(monkeypatch, value=53, *, refused=False, repeats=0, late=0., status=0,
             checked=True):
    live = Stream(np.zeros(16*FS))
    live.pos = round(.9*FS)
    tx = Tx(live)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 14. if repeats else 6.
    receiver = tx.sessrx.p4
    receiver.clock = Clock(Expect.PACKET, FS, .330, 1.25, 0.)
    machine = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: machine(*a, **{**kw, 'role': 'irs'}))
    monkeypatch.setitem(p4turn.PEER_ACK, 53, 'U27')
    monkeypatch.setattr(receiver, '_word', lambda *a, **kw: None)
    if refused:
        monkeypatch.setattr(onair, '_keyable_slot', lambda *a, **kw: None)
    payload = b'gateway chirp'.ljust(22, b'\x1e')
    frame = p4frame.decode_field(p4frame.encode_field(payload, p4frame.Status.from_byte(status)), data_bytes=22)
    received = []
    tx.p4_link_host.deliver = received.append
    chirp_at = FS+round((1.25+late)*FS)
    def packet(audio, origin, partial=False, long_fallback=False):
        at = receiver.clock.next_due
        short_at = next((FS+60000*i for i in range(repeats or 1)
                         if abs(at-FS-60000*i) < FS//100), None)
        if short_at is not None:
            return Observation(Kind.SHORT, FS, origin, short_at-origin, short_at, .758, 0., .8,
                               provisional=partial, value=value, header='A', walsh_peak=64)
        if not repeats and receiver.chirp and abs(at-chirp_at) < FS//10:
            available = (origin+len(audio)-chirp_at)/FS
            if partial and available < .80:
                return Observation(Kind.NORMAL, FS, origin, chirp_at-origin, chirp_at, available,
                                   0., .1, provisional=True)
            if available < (.80 if partial else 3.30):
                return None
            if not partial and not checked:
                return None
            return Observation(Kind.CHIRP, FS, origin, chirp_at-origin, chirp_at, 3.28, 0., .8,
                               provisional=partial, frame=None if partial else frame)
        return None
    monkeypatch.setattr(receiver, '_packet', packet)
    keyed = Tx._tx
    def key(audio, what):
        keyed(tx, audio, what)
        live.pos = tx.tx_end
    tx._tx = key
    onair._run_p4_link(tx, live, onair._MasterGrid(0,60000,0,packet_n=36000,cs_n=12000,d_max_n=4800))
    return tx, receiver, received, chirp_at


@pytest.mark.parametrize('value', [53, 60])
def test_checked_chirp_receive_is_independent_of_peer_control_answer(monkeypatch, value):
    tx, receiver, received, chirp_at = exchange(monkeypatch, value)
    assert tx.p4_probe.emissions[0]['word'] == 'U27'
    assert tx.p4_probe.emissions[0]['audio_start'] == FS+round(.920*FS)
    assert receiver.chirp
    assert received == [b'gateway chirp']
    assert tx.p4_probe.emissions[1]['audio_start'] == chirp_at+round(3.400*FS)
    assert tx.p4_probe.emissions[1]['word'] == 'V27'
    mode = [e for e in tx.p4_probe.link_events if e['event'] == 'peer-mode']
    assert bool(mode) == (value == 53)
    assert all(e['state'] == 'IRS_ANSWER' and e['kind'] == Kind.WORD
               for e in tx.p4_probe.link_events if e['event'] == 'Keyed')
    assert all(not e['payload'] for e in tx.p4_probe.emissions)


def test_refused_u27_keeps_key_refusal_without_disabling_checked_receive(monkeypatch):
    tx, receiver, received, _ = exchange(monkeypatch, refused=True)
    assert receiver.chirp and not tx.p4_probe.emissions
    assert received == [b'gateway chirp']
    assert not [e for e in tx.p4_probe.link_events if e['event'] in ('Keyed', 'peer-mode')]
    refused = [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']
    assert len(refused) == 2
    assert all(e['kind'] == Kind.WORD and e['reason'] == 'deadline'
               and e['state'] == 'IRS_ANSWER' for e in refused)


@pytest.mark.parametrize('value,refused', [(60, False), (53, True)])
def test_unread_chirp_has_no_answer_delivery_or_turn_credit(monkeypatch, value, refused):
    with neutral_keys() as neutral:
        tx, receiver, received, _ = exchange(monkeypatch, value, refused=refused, checked=False)
    assert receiver.chirp and received == []
    expected = [] if refused else [
        dict(at=at,word='U27',purpose=None,kind=Kind.WORD,state='IRS_ANSWER')
        for at in (271200,332160)]
    assert neutral == expected
    neutral_at = {e['at'] for e in neutral}
    ordinary = [e for e in tx.p4_probe.emissions if e['audio_start'] not in neutral_at]
    assert [e['word'] for e in ordinary] == ([] if refused else ['U27'])
    assert [e['audio_start'] for e in ordinary] == ([] if refused else [92160])
    assert [e['word'] for e in tx.p4_probe.emissions] == ([] if refused else ['U27']*3)
    assert all(not e['payload'] for e in tx.p4_probe.emissions)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'peer-mode']
    assert all(e['state'] == 'IRS_ANSWER' and e['kind'] == Kind.WORD
               for e in tx.p4_probe.link_events if e['event'] == 'Keyed')


@pytest.mark.parametrize('value,refused', [(60, False), (53, True)])
def test_unread_chirp_without_neutral_requests_keeps_original_keys(monkeypatch,value,refused):
    without_neutral_requests(monkeypatch)
    with neutral_keys() as neutral:
        tx,receiver,received,_=exchange(monkeypatch,value,refused=refused,checked=False)
    assert receiver.chirp and received==[] and neutral==[]
    assert [e['word'] for e in tx.p4_probe.emissions]==([] if refused else ['U27'])
    assert not [e for e in tx.p4_probe.link_events if e['event']=='peer-mode']
    assert all(e['state']=='IRS_ANSWER' and e['kind']==Kind.WORD
               for e in tx.p4_probe.link_events if e['event']=='Keyed')
    assert all(not e['payload'] for e in tx.p4_probe.emissions)


def test_repeated_peer_53_uses_short_deadlines_after_u27(monkeypatch):
    tx, receiver, received, _ = exchange(monkeypatch, repeats=4)
    assert [e['audio_start'] for e in tx.p4_probe.emissions[:4]] == [
        FS+60000*i+round(.920*FS) for i in range(4)]
    assert all(e['word'] == 'U27' for e in tx.p4_probe.emissions[:4])
    assert all(e['word'] == 'U27' for e in tx.p4_probe.emissions[4:])
    assert all(not e['current_slot_accepted'] for e in tx.p4_probe.link_events
               if e['event'] == 'Keyed' and e['at'] >= FS+4*60000+round(.920*FS))
    assert receiver.chirp and not received


def test_late_chirp_replaces_ambiguous_short_partial_before_the_deadline(monkeypatch):
    tx, receiver, received, chirp_at = exchange(monkeypatch, late=.025)
    assert received == [b'gateway chirp']
    assert tx.p4_probe.emissions[1]['audio_start'] == chirp_at+round(3.400*FS)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'BREAKIN_UNREAD']


@pytest.mark.parametrize('status,word', [(0x00, 'V27'), (0x61, 'X27')])
def test_first_chirp_answer_follows_its_protocol(monkeypatch, status, word):
    tx, receiver, received, chirp_at = exchange(monkeypatch, status=status)
    assert received == [b'gateway chirp']
    assert tx.p4_probe.emissions[1]['audio_start'] == chirp_at+round(3.400*FS)
    assert tx.p4_probe.emissions[1]['word'] == word
