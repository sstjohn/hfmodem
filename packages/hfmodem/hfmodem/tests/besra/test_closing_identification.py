# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Native identification geometry and production close/cancellation effects.

Frame-level handshakes here establish owner semantics, not stock RF timing.
"""
import numpy as np
import pytest

from hfmodem.besra.arq import session as S
from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.host.protocol import ArdopState
from hfmodem.besra.phy import modulator
from hfmodem.besra.phy.cwid import render_cwid
from hfmodem.besra.phy.demodulator import DecodedFrame
from hfmodem.station.link import BesraLink
from hfmodem.tests.besra.test_auxiliary_controls import setup
from hfmodem.tests.besra.test_station_runtime import Observer, station, wait_for


@pytest.mark.parametrize('on_off', [False, True])
def test_morse_has_standard_de_base_call_and_native_mark_space_tones(on_off):
    # Independently spelled ITU Morse: DE W9SSJ, with ordinary 1/3/7 gaps.
    words = [('-..', '.'), ('.--', '----.', '...', '...', '.---')]
    units = '000000' + '0000000'.join('000'.join(
        '0'.join('111' if element == '-' else '1' for element in letter)
        for letter in word) for word in words) + '000000'
    samples = render_cwid('W9SSJ-7', on_off=on_off)
    assert samples.dtype == np.dtype('<i2') and len(samples) == len(units)*768
    assert np.array_equal(samples, render_cwid('W9SSJ', on_off=on_off))
    assert np.max(np.abs(samples)) <= 32700
    t = np.arange(384)/12000
    for index, mark in enumerate(units):
        middle = samples[index*768+192:index*768+576].astype(float)
        if on_off and mark == '0':
            assert np.sqrt(np.mean(middle**2)) < 2
            continue
        frequencies = (1390.625, 1609.375)
        power = [abs(np.sum(middle*np.exp(-2j*np.pi*f*t))) for f in frequencies]
        assert power[int(mark)] > 10*power[1-int(mark)]


@pytest.mark.parametrize('on_off', [False, True])
def test_filtered_cw_stays_within_its_declared_500_hz_emission(on_off):
    from hfmodem.core.occupied import keyed_hz
    from hfmodem.tests.core.test_occupied import _minus_26db_span
    actual = _minus_26db_span(render_cwid('W9SSJ', on_off=on_off), fs=12000)
    lo, hi = keyed_hz('ardop', '500')
    assert lo <= actual[0] < actual[1] <= hi


def test_cw_enabled_id_keeps_digital_identity_and_applies_drive_to_both_parts():
    from hfmodem.besra.phy.demodulator import decode
    modem, host, _, received, samples = setup()
    host._dispatch('SENDID')
    digital_length = len(samples[-1])
    host._dispatch('CWID TRUE')
    host._dispatch('SENDID')
    full = samples[-1]
    frames = decode(np.pad(full[:digital_length], (2400, 4800)))
    assert frames[0].ok and frames[0].caller == 'W9SSJ-7'
    host._dispatch('DRIVELEVEL 25')
    host._dispatch('SENDID')
    assert np.array_equal(samples[-1], np.rint(full.astype(float)*.25).astype('<i2'))
    assert received[-1] == ('IDF', b' W9SSJ-7:[NO-GS] ')
    assert modem._session.accepted_total == modem._session.confirmed_total == 0


def connect(modem):
    modem.connect('K7ABC')
    session = modem._session
    sid = session._session
    session.on_receive(S.CONACK[500], bytes([24])*3, sid, True)
    session.on_receive(0xff, b'', sid, True)
    assert modem.connected
    return session, sid


def test_periodic_id_uses_an_iss_turn_and_does_not_spend_payload_credit(monkeypatch):
    frames = []
    def render(ft, *args, **kwargs):
        frames.append(ft)
        return np.ones(1200, dtype='<i2')
    monkeypatch.setattr(modulator, 'render_frame', render)
    modem, _, _, received, _ = setup()
    _, sid = connect(modem)
    frames.clear()
    modem._clock = 600
    modem.send(0xff, b'', sid)
    assert frames == [0xff] and not received
    modem.transmit(b'original')
    assert frames[-2:] == [0x30, 0x48]
    assert modem._session.confirmed_total == 0
    assert modem._session.accepted_total == len(b'original')
    # The ID's duration counts toward the next data-repeat deadline, and a
    # further ISS turn before the next interval cannot insert another ID.
    modem._clock = 601
    assert modem.send(0x24, b'', sid) == pytest.approx(.1)
    assert frames.count(0x30) == 1
    modem._clock = 1201
    assert modem.send(0x24, b'', sid) == pytest.approx(.2)
    assert frames.count(0x30) == 2


def test_end_emits_id_before_terminal_and_never_credits_unconfirmed_originals():
    modem, host, events, received, samples = setup()
    host._dispatch('CWID TRUE')
    session, sid = connect(modem)
    modem.transmit(b'unconfirmed final originals')
    sequence = []
    modem.audio_out = lambda data: sequence.append(('samples', len(data)))
    host._send_cmd = lambda text: sequence.append(('command', text))
    session.on_receive(S.END, b'', sid, True)
    assert sequence[0] == ('command', 'PTT TRUE')
    assert sequence.index(('command', 'PTT FALSE')) < sequence.index(('command', 'DISCONNECTED'))
    assert received[-1] == ('IDF', b' W9SSJ-7:[NO-GS] ')
    assert session.last_unconfirmed == b'unconfirmed final originals'
    assert session.confirmed_total == 0
    sent = len(sequence)
    session.on_receive(S.END, b'', sid, True)
    modem.abort()
    assert len(sequence) == sent


def test_peer_disc_defers_id_and_blocks_replacement_until_completion():
    modem, _, _, received, samples = setup()
    session, sid = connect(modem)
    samples.clear()
    session.on_receive(S.DISC, b'', sid, True)
    assert modem.state == ArdopState.DISC and modem.identification_pending
    assert len(samples) == 1  # END, not the closing ID
    with pytest.raises(ValueError, match='identification'):
        modem.connect('W1AW')
    with pytest.raises(ValueError, match='identification'):
        modem.set_mycall('W1AW')
    with pytest.raises(ValueError, match='idle lane'):
        modem.set_codec(False)
    modem.tick(4.99)
    assert len(samples) == 1
    modem.tick(5.)
    assert len(samples) == 2 and not modem.identification_pending
    assert received == [('IDF', b' W9SSJ-7:[NO-GS] ')]
    modem.tick(100)
    assert len(samples) == 2


@pytest.mark.parametrize('cancel', ['abort', 'stop'])
def test_cancelled_closing_id_cannot_transmit_on_later_contact(cancel):
    modem, _, _, received, samples = setup()
    session, sid = connect(modem)
    session.on_receive(S.DISC, b'', sid, True)
    before = len(samples)
    getattr(modem, cancel)()
    modem.tick(100)
    assert len(samples) == before and not received
    assert not modem.identification_pending


def test_refused_closing_id_retires_originals_and_reports_no_success():
    modem, _, _, received, _ = setup()
    session, sid = connect(modem)
    modem.transmit(b'unconfirmed')
    def refuse(samples):
        raise OSError('ID refused')
    modem.audio_out = refuse
    with pytest.raises(OSError, match='ID refused'):
        session.on_receive(S.END, b'', sid, True)
    assert session.state == ArdopState.DISC
    assert session.last_unconfirmed == b'unconfirmed'
    assert session.confirmed_total == 0 and not received
    modem.abort()
    assert session.last_unconfirmed == b'unconfirmed'


def test_station_declares_cw_emission_and_holds_delayed_identification(monkeypatch):
    # A short frame double keeps this owner/lifetime test independent of DSP.
    monkeypatch.setattr(modulator, 'render_frame',
                        lambda *a, **kw: np.array([-1000, 0, 1000], dtype='<i2'))
    st, sent = station()
    link = BesraLink(st)
    now = [0.]
    link.modem.clock = lambda: now[0]
    link.modem.start(Observer())
    try:
        link.modem.set_mycall('W9SSJ')
        link.modem.set_bandwidth(200, False)
        link.modem.set_cwid('TRUE')
        link.modem.send_id()
        assert sent[-1].emission.audio_lo_hz <= 1390.625
        assert sent[-1].emission.audio_hi_hz >= 1609.375
        assert link.waveform.bw == '500' and sent[-1].rate == 12000
        link.modem.set_bandwidth(500, False)
        link.modem.connect('K7ABC')
        sid = link.modem.expected_session()
        link.modem.deliver_frames([DecodedFrame(S.CONACK[500], sid, True, bytes([24])*3)])
        wait_for(lambda: link.modem.core._session._sub == S._Sub.ISS_CONACK)
        link.modem.deliver_frames([DecodedFrame(0xff, sid, True)])
        wait_for(lambda: link.modem.connected)
        link.modem.deliver_frames([DecodedFrame(S.DISC, sid, True)])
        wait_for(lambda: link.modem.state == ArdopState.DISC)
        assert link.active
        before = len(sent)
        now[0] = 5.
        wait_for(lambda: not link.active)
        assert len(sent) == before+1
    finally:
        link.stop()


def test_standalone_stop_joins_old_pump_and_restart_has_a_fresh_receive_queue():
    modem = BesraModem()
    observer = Observer()
    modem.start(observer, threaded=True)
    old_pump, old_queue = modem._pump, modem._rx_q
    try:
        with pytest.raises(RuntimeError, match='already started'):
            modem.start(observer, threaded=True)
        modem.stop()
        assert not old_pump.is_alive() and modem._pump is None
        old_queue.put(np.ones(12000))
        modem.start(observer, threaded=True)
        assert modem._rx_q is not old_queue and modem._rx_q.empty()
        assert modem._pump is not old_pump and modem._pump.is_alive()
    finally:
        modem.stop()
    assert modem._pump is None


def test_reentrant_stop_before_submission_cannot_emit_or_report_an_id():
    modem, host, events, received, samples = setup()
    ptt = []
    def observe(on):
        ptt.append(on)
        if on:
            modem.stop()
    host.modem_ptt = observe
    host._dispatch('SENDID')
    assert ptt == [True, False]
    assert events[-1].startswith('FAULT')
    assert not samples and not received
    assert modem._session is None


def test_refused_timed_id_reports_fault_without_losing_the_standalone_pump():
    modem = BesraModem()
    observer = Observer()
    def refuse(samples):
        raise OSError('timed ID refused')
    modem.audio_out = refuse
    modem.set_mycall('W9SSJ')
    modem.start(observer, threaded=True)
    try:
        with modem._lock:
            modem._closing_id = (0., ('W9SSJ', '', 'FALSE'))
        wait_for(lambda: any(name == 'modem_fault' for name, _, _ in observer.events))
        assert modem._pump.is_alive() and not modem.identification_pending
        assert modem.state == ArdopState.DISC
        assert not any(name == 'modem_data_received' for name, _, _ in observer.events)
    finally:
        modem.stop()
