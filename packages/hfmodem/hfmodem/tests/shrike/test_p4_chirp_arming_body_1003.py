# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Admitted none/body links receive the peer's existing SL1 chirp form."""

from dataclasses import replace
from contextlib import redirect_stdout
from inspect import getsource
from os import devnull
from time import perf_counter
import json
from pathlib import Path

import pytest

from hfmodem.shrike import onair, p4chirp, p4chirprx
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.shrike.p4turn import P4Turn
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike import p4_chirp_deadline_support as support
from hfmodem.tests.shrike.test_p4turn import ack_frame
from hfmodem.tests.shrike.p4_nak_qam_support import (
    neutral_keys, waveform_keys, without_neutral_requests,
)


STARTS = (1248792, 1608702, 1788749, 1968723)
FIELD = b'RMS Trimode 1.4.4.0\r\nW'


def recorded(monkeypatch):
    """Charge every packet read, including the selected chirp CRC, to sample time."""
    calls = []
    deliver = []
    receivers = []
    headers = []
    decoding = [False]
    original_tx = support.Tx
    streams = []
    refusal_clock = []

    class ChargedTx(original_tx):
        def __init__(self, stream):
            super().__init__(stream)
            streams.append(stream)
            self.p4_link_host.deliver = deliver.append
            rx = self.sessrx.p4
            receivers.append(rx)
            assert not rx.chirp
            read = rx.read_chirp

            def timed(*args, **kwargs):
                started = perf_counter()
                decoding[0] = True
                try:
                    records = read(*args, **kwargs)
                finally:
                    decoding[0] = False
                seconds = perf_counter()-started
                stream.wall = stream.sample_now()+round(seconds*support.FS)
                calls.append((kwargs['onset'], seconds, records is not None))
                return records

            rx.read_chirp = timed

    seed = P4Turn.seed_opening

    def waiting(machine, *args, **kwargs):
        result = seed(machine, *args, **kwargs)
        machine.allow_breakin = False
        return result

    acquire = p4chirprx.acquire_header
    refuse = P4Turn.refused
    def timed_refusal(machine,reason):
        proposal=machine._next
        if proposal is not None and not proposal.current_slot_accepted:
            stream=streams[0]
            now=stream.sample_now()
            refusal_clock.append(dict(at=proposal.stream_sample,word=proposal.word,
                purpose=proposal.purpose,kind=proposal.kind,state=proposal.state,reason=reason,
                now_sample=now,key_notice=stream.key_notice,
                margin_samples=proposal.stream_sample-stream.key_notice-now))
        return refuse(machine,reason)

    def tracked_header(*args, **kwargs):
        before = receivers[0].clock.cycle_s
        header = acquire(*args, **kwargs)
        if header is not None and not decoding[0]:
            at = receivers[0].clock.next_due+header.start_sample-kwargs['nominal_sample']
            if not any(abs(prior-at) < 100 for prior, _, _ in headers):
                headers.append((at, header.marker, before))
        return header

    with monkeypatch.context() as patch, open(devnull, 'w') as sink, redirect_stdout(sink):
        patch.setattr(support, 'Tx', ChargedTx)
        patch.setattr(P4Turn, 'seed_opening', waiting)
        patch.setattr(P4Turn, 'refused', timed_refusal)
        patch.setattr(p4chirprx, 'acquire_header', tracked_header)
        result = support.replay('onair-1003-1110', measured_work=True)
    result['headers'] = headers
    result['neutral_refusal_clock'] = refusal_clock
    return result, calls, deliver


def assert_charged_chirp_receive(result, calls, delivered, neutral):
    complete = [row for row in result['heard']
                if row['kind'] == 'chirp' and not row['provisional']]
    assert [row['onset_sample'] for row in complete] == list(STARTS)
    assert all(row['status'] == 0x61 and bytes.fromhex(row['data']) == FIELD
               for row in complete)
    assert result['headers'][0] == (STARTS[0], 8, 1.25)
    assert [marker for _, marker, _ in result['headers']] == [8, 9, 9, 9, 9]
    assert [at for at, _, read in calls if read] == list(STARTS)
    assert all(seconds > 0 for _, seconds, _ in calls)
    # Actual decode work is charged to sample_now before the real admission check.
    ordinary = {row['at'] for row in result['events']
                if row['event'] == 'Keyed' and row.get('current_slot_accepted', True)}
    replies = [row['audio_start'] for row in result['emissions']
               if row.get('word') == 'X27' and row['audio_start'] >= STARTS[0]
               and row['audio_start'] in ordinary]
    expected = [at+round(3.4*support.FS) for at in STARTS]
    assert replies and set(replies) <= set(expected)
    assert set(expected[1:]) <= set(replies)
    assert expected[0] in replies or expected[1] in replies
    refusals = result['neutral_refusal_clock']
    # Any new refusal must belong to a captured neutral request. Ordinary
    # receive ACKs retain the original no-refusal condition.
    assert [(e['at'],e['kind'],e['state'],e['reason'])
            for e in result['events'] if e['event']=='Refused'] == [
        (e['at'],e['kind'],e['state'],e['reason']) for e in refusals]
    assert all(e['purpose'] is None and e['margin_samples']<0
               and e['reason']=='deadline' for e in refusals)
    assert not {e['at'] for e in refusals} & {e['audio_start'] for e in result['emissions']}
    assert all(not (at < row['audio_start'] < at+round(p4chirp.PACKET_S*support.FS))
               for at in STARTS for row in result['emissions'])
    ticks = [at for at, _ in result['cycles'] if at >= STARTS[0]]
    assert len(ticks) == 5
    assert all(abs((right-left)-round(3.75*support.FS)) < 100
               for left, right in zip(ticks, ticks[1:]))
    assert result['reason'] != 'peer silent'
    assert delivered == [b'RMS Trimode 1.4.4.0\r\nW9SSJ has ', FIELD]
    # The next complete local copy is unread; the final marker is truncated.
    assert not any(abs(row['onset_sample']-1428792) < 100 for row in complete)
    assert any(row['kind'] == 'chirp' and row['provisional']
               and abs(row['onset_sample']-2148862) < 100 for row in result['heard'])


def test_none_body_driver_receives_chirps_with_charged_work(monkeypatch):
    refused=[]
    with neutral_keys(refused=refused) as neutral, waveform_keys() as waves:
        result, calls, delivered = recorded(monkeypatch)
    pins = json.loads((Path(__file__).with_name('fixtures')/
                       'nak-stack1003e-controls.json').read_text())['onair-1003-1110']
    assert neutral == pins['neutral']
    assert refused == [dict(at=751084,word='X27',purpose=None,
                            kind='word',state='IRS_ANSWER',reason='deadline')]
    assert result['neutral_refusal_clock'] == [{**refused[0],'now_sample':749980,
                                              'key_notice':1796,'margin_samples':-692}]
    assert_charged_chirp_receive(result, calls, delivered, neutral)
    # Both real ACKs and neutral requests key the frozen X27 waveform; only
    # whether the keyed slot was accepted distinguishes the two, never the word's name.
    words = {row['at'] for row in result['events']
             if row['event'] == 'Keyed' and row.get('word') == 'X27'}
    assert words == {row['at'] for row in waves}
    assert all({k:row[k] for k in ('samples', 'dtype', 'sha256')}
               == pins['word_waveform'] for row in waves)


def test_charged_chirp_without_neutral_requests_keeps_original_receive(monkeypatch):
    without_neutral_requests(monkeypatch)
    with neutral_keys() as neutral:
        result, calls, delivered = recorded(monkeypatch)
    assert neutral == []
    assert result['neutral_refusal_clock'] == []
    assert_charged_chirp_receive(result, calls, delivered, neutral)


def test_recording_pin_fails_with_previous_transmit_option_gate(monkeypatch):
    source = getsource(onair._run_p4_link)
    original = 'receiver.chirp = True'
    assert original in source
    source = source.replace(original,
        "receiver.chirp |= args.p4_ladder == 'chirp' or args.p4_data == 'chirp'", 1)
    namespace = dict(vars(onair))
    exec(compile(source, 'previous_chirp_arming', 'exec'), namespace)
    monkeypatch.setattr(onair, '_run_p4_link', namespace['_run_p4_link'])
    result, calls, delivered = recorded(monkeypatch)
    assert not calls
    assert not [row for row in result['heard'] if row['kind'] == 'chirp']
    assert result['reason'] == 'peer silent'
    assert delivered == [b'RMS Trimode 1.4.4.0\r\nW9SSJ has ']


@pytest.mark.parametrize('enabled', [False, True])
def test_standalone_disabled_receiver_keeps_its_gate(enabled):
    path = CAPTURES/'onair-1003-1110/stream.flac'
    require_input(path)
    audio = onair.session.load_wav(str(path), support.FS)
    nominal = 1248463
    rx = P4Receiver(support.FS, admitted=True, chirp=enabled)
    rx.expected = True
    rx.clock = Clock(Expect.PACKET, nominal, .330, 1.25, 2.13481)
    rx.read_partial(audio[nominal-2400:nominal+round(.81*support.FS)],
                    nominal-2400, deadline=nominal+round(.81*support.FS))
    assert bool(rx._chirp_headers) is enabled
    assert rx.clock.cycle_s == (3.75 if enabled else 1.25)


@pytest.fixture(scope='module')
def unread_marker_audio():
    path = CAPTURES/'onair-0929-0936/stream.flac'
    require_input(path)
    return onair.session.load_wav(str(path), support.FS)


def unread_marker_receiver(audio, cycle=1.25):
    nominal, onset = 2147800, 2149224
    origin = nominal-round(.5*support.FS)
    rx = P4Receiver(support.FS, admitted=True, chirp=True)
    rx.expected = True
    rx.clock = Clock(Expect.PACKET, nominal, .330, cycle, -2.5703200253367777)
    early = nominal+round(.845*support.FS)
    heard = rx.read_partial(audio[origin:early], origin, deadline=early)
    assert any(reg.observation.kind == 'chirp' and reg.observation.provisional
               and reg.observation.stream_sample == onset for reg in heard)
    assert rx.has_chirp_header(onset) and rx.clock.cycle_s == 3.75
    return rx, origin, onset


@pytest.mark.parametrize('previous', [1.25, 3.75])
@pytest.mark.parametrize('path', ['fast', 'ordinary', 'expiry'])
def test_recorded_unread_marker_releases_only_its_provisional_cadence(
        unread_marker_audio, monkeypatch, previous, path):
    audio = unread_marker_audio
    rx, origin, onset = unread_marker_receiver(audio, previous)
    end = onset+round(p4chirp.PACKET_S*support.FS)
    if path == 'fast':
        assert rx.read_chirp(audio[origin:end], origin, onset=onset) is None
    elif path == 'ordinary':
        rx.read_window(audio[origin:end], origin, whole=True)
    else:
        # Expiry must release the marker even when a decoder is not dispatched.
        monkeypatch.setattr(rx, '_packet', lambda *a, **kw: None)
        rx.read_window(audio[origin:onset+round(3.36*support.FS)], origin, whole=True)
    assert not rx.has_chirp_header(onset)
    assert rx.clock.cycle_s == previous
    assert not any(o.kind == 'chirp' and o.frame is not None for o in rx.observations)
    assert [row['restored_cycle_s'] for row in rx.live_events
            if row['event'] == 'chirp header unread'] == [previous]
    # The same retained buffer must not reacquire the rejected marker.
    rx._packet(audio[origin:end], origin)
    assert rx.clock.cycle_s == previous


@pytest.mark.parametrize('new_owner', ['clock', 'generation'])
def test_expired_chirp_marker_cannot_restore_a_stale_owner_cadence(
        unread_marker_audio, new_owner):
    rx, _, onset = unread_marker_receiver(unread_marker_audio)
    if new_owner == 'clock':
        rx.clock = replace(rx.clock)
    else:
        rx._train_generation += 1
    rx._reject_chirp_header(onset)
    assert rx.clock.cycle_s == 3.75
    assert not [row for row in rx.live_events if row['event'] == 'chirp header unread']


@pytest.mark.parametrize('kind,cycle', [('long', 3.75), ('normal', 1.25)])
def test_expired_chirp_marker_cannot_displace_a_checked_packet_cadence(
        unread_marker_audio, kind, cycle):
    rx, _, onset = unread_marker_receiver(unread_marker_audio, cycle=3.75)
    checked = ack_frame(onset, b'checked source', 0x61, kind=kind).observation
    rx._emit(checked, [])
    assert rx.clock.cycle_s == cycle
    rx._reject_chirp_header(onset)
    assert rx.clock.cycle_s == cycle
    assert not [row for row in rx.live_events if row['event'] == 'chirp header unread']
