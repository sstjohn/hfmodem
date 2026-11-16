# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Captured span refusals classify the actual proposal at both boundaries."""
from dataclasses import replace
from unittest.mock import patch
import pytest
from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import Clock, Expect
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx
from hfmodem.tests.shrike.test_p4_opus_driver_1011 import (
    ChargedStream, GRID, LONG42_ONSETS, assert_long42_identity,
)
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, CYCLE
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.p4_declared_clock_1111 import declared_time


@pytest.mark.parametrize('boundary', ['initial', 'final'])
@pytest.mark.parametrize('kind', ['acquisition', 'policy', 'deferred', 'close'])
def test_f1_captured_span_refusal_keeps_answer_or_close_owner(boundary, kind):
    start, stop = 8016000, KEY+3*CYCLE+FS
    path = require_input(CAPTURES/
                         'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590/stream.flac')
    source = onair.session.load_wav(str(path), FS)
    live = ChargedStream(source[start:stop], start)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    args = tx.p4_link_options
    args.p4_data, args.p4_link_timeout = 'body', (stop - start) / FS
    if kind == 'acquisition': tx.p4_link_host.arq.on_host_data(b'TEST\r')
    rx = tx.sessrx.p4
    rx.clock, rx.cfo_hz = Clock(Expect.PACKET, AT, .330, 3.75, -.19), -.19
    rx.expected = True
    if kind == 'deferred':
        read = rx.read_chirp
        def slow(*a, **kw):
            result = read(*a, **kw)
            live.charge('real chirp CRC work', .100)
            return result
        rx.read_chirp = slow
    cls, machines, refused, keys, acquisitions = p4turn.P4Turn, [], [], [], []
    retained = []
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        machines.append(m)
        decide, request, fits = m.decide, m._break_in, m.peer_span_fits
        observe = m.observe
        def observed(reg, **kw):
            obs = reg.observation
            if obs.frame is not None and obs.frame.status.byte == 0x42:
                held, accepted, deferred = m._held_answer, m._accepted_proposal, m._deferred
                owners = ([held.onset] if held else []) + ([accepted[1]] if accepted else []) + ([deferred[0]] if deferred else [])
                retained.append((tuple(owners), None if held is None else held.answer.word))
            return observe(reg, **kw)
        m.observe = observed
        selected = False
        def decision(now):
            nonlocal selected
            p = decide(now)
            if (p is not None and not selected and (kind != 'deferred'
                    or m._accepted_proposal is not None and m._accepted_proposal[2].get('fallback'))):
                selected = True
                if kind == 'acquisition' and boundary == 'initial':
                    m._next = replace(p, kind=Kind.BODY, word=None, status=0x40, state=p4turn.State.ISS_GRANT)
                elif kind == 'close':
                    # Synthetic captured close/control word; it cannot acquire
                    # data ownership merely because the long watch is present.
                    m._next = replace(p, word='V27', state=p4turn.State.IRS_CLOSE)
                if kind == 'deferred': assert m._accepted_proposal[2].get('fallback')
                p = m._next
            return p
        def break_in(due):
            acquisitions.append((due, m._next, m._accepted_proposal))
            return request(due)
        def span(p, length, allowance):
            if boundary == 'initial' and selected and not refused:
                phase = m.peer_phase
                # A real guard sees a synthetic protected phase displacement.
                if kind != 'acquisition': m.peer_phase = replace(phase, onset=phase.onset-960)
                result = fits(p, length, allowance)
                m.peer_phase = phase
                assert not result
                refused.append(p)
                return result
            return fits(p, length, allowance)
        m.decide, m._break_in, m.peer_span_fits = decision, break_in, span
        return m
    emit = tx._tx
    def key(audio, what):
        m = machines[0]
        p = m._next
        if boundary == 'final' and not refused:
            phase = m.peer_phase
            m.peer_phase = replace(phase, onset=phase.onset-960)
            refused.append(p)
            try:
                tx.p4_tx_admit()
                pytest.fail('the captured waveform unexpectedly fit the protected span')
            finally:
                m.peer_phase = phase
        tx.p4_tx_admit()
        assert live.sample_now()+live.key_notice <= p.stream_sample
        emit(audio, what)
        keys.append(p)
    tx._tx = key
    # This stream charges explicit protocol work. Physical
    # Python CPU must not become a second planning clock or a native deadline.
    with declared_time(live.sample_now, fs=FS), patch.object(p4turn, 'P4Turn', factory):
        onair._run_p4_link(tx, live, GRID)
    assert refused and not any(p is refused[0] for p in keys)
    after_refusal = [p for p in keys if p.stream_sample >= refused[0].stream_sample]
    # Design 9/148/194/203: the exact engine now reads the distinct 0x42
    # packet in this capture. Refusal retains the original policy owner
    # until that new CRC evidence; the new packet then owns its own ACK.
    frames = [row for row in tx.p4_probe.heard if row['kind'] == Kind.LONG
              and row.get('status') == 0x42 and not row['provisional']]
    assert [row['onset_sample'] for row in frames] == pytest.approx([LONG42_ONSETS[0]], abs=48)
    assert_long42_identity(frames[0])
    fresh_at = frames[0]['onset_sample']
    before = [p for p in after_refusal if p.stream_sample < fresh_at+163200]
    fresh = [p for p in after_refusal if p.stream_sample >= fresh_at+163200]
    if kind == 'acquisition':
        assert before and all(p.kind == Kind.WORD and p.word == 'Y27' for p in before)
    else:
        assert retained and all(owners and set(owners) == {AT} and word == 'X27'
                                for owners, word in retained)
        assert all(p.word == 'X27' and p.state == p4turn.State.IRS_ANSWER
                   and p.accepted_target == AT for p in before)
        if not before:
            # A new CRC can precede the deferred answer's next retry slot.
            # Its earlier answer may then supersede that refused proposal.
            assert (kind, boundary) == ('deferred', 'final')
            assert refused[0].word == 'X27' and refused[0].accepted_target == AT
            assert fresh_at+163200 < refused[0].stream_sample+CYCLE
        assert not acquisitions
    # The 0x42 is robust LONG: its toggle partner is V27. A refused
    # acquisition/deferred-final answer never keys the preceding X27, so
    # those chains still start with X27; otherwise the new packet toggles
    # the already keyed X27 to V27. Repeats must retain that packet's ACK.
    fresh_word = ('X27' if kind == 'acquisition' or (kind, boundary) == ('deferred', 'final')
                  else 'V27')
    assert [(p.stream_sample, p.word, p.accepted_target) for p in fresh] == [
        (8731525, fresh_word, LONG42_ONSETS[0])]
    assert all(p.word == fresh_word and p.state == p4turn.State.IRS_ANSWER
                         and p.accepted_target == fresh_at and p.current_slot_accepted
                         and p.ack_identity == (0x42, bytes.fromhex(frames[0]['data']))
                         for p in fresh)
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    print(dict(kind=kind, boundary=boundary, refused=(id(refused[0]), refused[0]),
               keys=[(id(p), p.stream_sample, p.word, p.accepted_target) for p in keys],
               acquisitions=[at for at, p, ctx in acquisitions]))
