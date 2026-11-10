# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Unconnected ARDOP producers must emit samples without payload credit."""
import threading

import numpy as np
import pytest

from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.host.server import HostServer
from hfmodem.besra.phy.demodulator import decode
from hfmodem.besra.station_runtime import StationArdopModem
from hfmodem.tests.besra.test_station_runtime import Observer


def setup():
    modem = BesraModem()
    host = HostServer(modem, control_port=0, data_port=0, quiet=True)
    events, received, samples = [], [], []
    host._send_cmd = events.append
    host._send_data = lambda tag, data: received.append((tag, data))
    modem.audio_out = samples.append
    modem.start(host)
    host._dispatch('MYCALL W9SSJ-7')
    return modem, host, events, received, samples


def test_sendid_renders_a_decodable_frame_and_reports_the_same_identity():
    modem, host, events, received, samples = setup()
    host._dispatch('GRIDSQUARE EN51')
    host._dispatch('SENDID')
    assert events[-1] == 'SENDID'
    assert len(samples) == 1 and samples[0].dtype == np.dtype('<i2')
    frames = decode(np.pad(samples[0], (2400, 4800)))
    assert len(frames) == 1
    assert frames[0].ok and frames[0].name == 'IDFrame'
    assert frames[0].caller == 'W9SSJ-7' and frames[0].grid == 'EN51'
    assert received == [('IDF', b' W9SSJ-7:[EN51] ')]
    assert modem._session.accepted_total == modem._session.confirmed_total == 0


def test_missing_locator_uses_the_stock_no_grid_marker():
    _, host, events, received, samples = setup()
    host._dispatch('SENDID')
    frames = decode(np.pad(samples[0], (2400, 4800)))
    assert frames[0].ok and frames[0].grid == 'NO-GS'
    assert received == [('IDF', b' W9SSJ-7:[NO-GS] ')]
    assert events[-1] == 'SENDID'


def test_leader_trailer_controls_reach_rendered_frame_geometry():
    modem, host, _, _, samples = setup()
    host._dispatch('SENDID')
    baseline = len(samples[-1])
    host._dispatch('LEADER 400')
    host._dispatch('TRAILER 80')
    host._dispatch('SENDID')
    assert modem._leader_ms == 400 and modem._trailer_ms == 80
    assert len(samples[-1])-baseline == 12000*(160+60)//1000
    assert decode(np.pad(samples[-1], (2400, 4800)))[0].ok


def test_two_tone_has_native_five_second_geometry_and_effective_drive():
    modem, host, events, _, samples = setup()
    modem.transmit(b'waiting original')
    host._dispatch('DRIVELEVEL 25')
    host._dispatch('TWOTONETEST')
    assert events[-1] == 'TWOTONETEST'
    tone = samples[-1]
    assert len(tone) == 5*12000 and tone.dtype == np.dtype('<i2')
    assert 0 < np.max(np.abs(tone)) < 10000
    spectrum = np.abs(np.fft.rfft(tone))
    frequencies = np.fft.rfftfreq(len(tone), 1/12000)
    strongest = frequencies[np.argsort(spectrum)[-2:]]
    assert np.allclose(sorted(strongest), [1475, 1525], atol=.3)
    assert modem._session.accepted_total == len(b'waiting original')
    assert modem._session.confirmed_total == 0


def test_offline_producers_and_failed_transmission_never_report_sent_id():
    modem, host, events, received, samples = setup()
    host._dispatch('CODEC FALSE')
    host._dispatch('SENDID')
    assert events[-1].startswith('FAULT') and not samples
    host._dispatch('TWOTONETEST')
    assert events[-1].startswith('FAULT') and not samples
    host._dispatch('CODEC TRUE')
    def refuse(samples):
        raise OSError('station submission refused')
    modem.audio_out = refuse
    host._dispatch('SENDID')
    assert events[-1].startswith('FAULT')
    assert not received
    assert 'PTT FALSE' in events


def test_cw_modes_are_effective_and_preserve_queued_originals():
    modem, host, events, _, samples = setup()
    modem.transmit(b'pending original')
    lengths = []
    for mode in ('TRUE', 'ONOFF', 'FALSE'):
        host._dispatch(f'CWID {mode}')
        assert events[-1] == f'CWID now {mode}'
        assert modem._cwid == mode
        host._dispatch('SENDID')
        lengths.append(len(samples[-1]))
        assert modem.queued == len(b'pending original')
        assert modem._session.confirmed_total == 0
    assert lengths[0] == lengths[1] > lengths[2]
    host._dispatch('CWID FALSE')
    assert events[-1] == 'CWID now FALSE'


def test_auxiliary_owner_protects_station_from_admission_until_completion():
    modem = StationArdopModem(lambda samples: None)
    modem.start(Observer())
    entered, release = threading.Event(), threading.Event()
    errors = []
    def hold():
        entered.set()
        assert release.wait(3)
    modem.core.two_tone_test = hold
    def submit():
        try:
            modem.two_tone_test()
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=submit)
    worker.start()
    try:
        assert entered.wait(3)
        assert modem.active and not modem.connected
        release.set()
        worker.join(3)
        assert not worker.is_alive() and not errors
        assert not modem.active
    finally:
        release.set()
        worker.join(3)
        modem.stop()


def test_refused_auxiliary_burst_preserves_unsent_arq_payload():
    def refuse(samples):
        raise OSError('station arbiter refused auxiliary burst')
    modem = StationArdopModem(refuse)
    modem.start(Observer())
    try:
        modem.set_mycall('W9SSJ')
        modem.transmit(b'preserved original')
        with pytest.raises(OSError, match='arbiter refused'):
            modem.send_id()
        assert modem.queued == len(b'preserved original')
        assert modem.core._session.confirmed_total == 0
        assert modem.core._session.discarded_total == 0
        assert not modem.active
    finally:
        modem.stop()


def test_repeated_abort_and_owner_stop_preserve_original_retirement():
    modem = StationArdopModem(lambda samples: None)
    modem.start(Observer())
    original = bytes(range(256))
    modem.transmit(original)
    modem.abort()
    modem.abort()
    assert modem.core._session.last_unconfirmed == original
    assert len(modem.core._session.retired) == 1
    modem.stop()
    assert modem.core._session is None
    assert modem.core.retired[-1]['unconfirmed'] == original
    assert modem.core.last_payload_status == dict(accepted=256, confirmed=0,
                                                 discarded=256, unconfirmed=original)


def test_reentrant_terminal_callback_cannot_erase_or_duplicate_retirement():
    from hfmodem.tests.besra.test_arq_session import Pair
    pair = Pair()
    pair.connect()
    original = b'unconfirmed final original'
    pair.caller.queue_data(original)
    pair.obs_a.disconnected = pair.caller.abort
    pair.caller.abort()
    assert pair.caller.last_unconfirmed == original
    assert len(pair.caller.retired) == 1
    assert pair.caller.retired[-1]['unconfirmed'] == original
    assert pair.caller.discarded_total == len(original)
    assert not pair.caller.connected
