# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Actual WS8EOC accepted entry which the live receiver missed on 2026-09-08.

The compact fixtures are real received audio, including the production listen
windows. No transmit waveform or guessed application response is substituted.
"""
import json

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, p3acquire, pactor1, placement, spec
from hfmodem.tests.shrike.archive import P3_FIXTURES, requires_ws8eoc_p3
from hfmodem.tests.shrike.test_entry_answer import _Session

FS = 48000


def audio(name):
    fs, x = wavfile.read(P3_FIXTURES / name)
    assert fs == FS
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float) / (np.iinfo(x.dtype).max + 1)
    return x[:, 0] if x.ndim == 2 else x


@requires_ws8eoc_p3
def test_real_short_reply_windows_take_the_granted_p3_breakin():
    sess = _Session(entry_pending=True)
    # The old P1 answer anchor supplied by the live grid, not a hand-picked
    # correct P3 head. New frequency/phase acquisition must find the latter.
    meta = json.loads((P3_FIXTURES / "ws8eoc-0908-p3.json").read_text())
    results = []
    for n, end, due in [(4, meta[0]["end_stream_sample"], 653749),
                        (5, meta[1]["end_stream_sample"], 713690)]:
        x = audio(f"ws8eoc-0908-p3-head-{n}.wav")
        begin = end-len(x)
        sess.rx.new_cycle()
        results.append(sess.rx.control_signal(x, begin, due))
    assert results[-1] == arq.CS_BREAKIN
    assert sess.host.protocol == spec.Protocol.PACTOR3
    assert sess.host.arq.role == arq.IRS
    assert not sess.host.arq.entry_pending
    sess.rx.new_cycle()
    sess.rx.deep_scan(audio("ws8eoc-0908-p3-rms.wav"))
    assert bytes(sess.host.channel(sess.host.ptchn).rx) == b"RMS"


@requires_ws8eoc_p3
def test_real_complete_changeover_delivers_rms_to_the_host(monkeypatch):
    sess = _Session(entry_pending=True)
    def no_normal_packet_search(*args):
        pytest.fail("expected changeover must decode before the normal-data scan")
    monkeypatch.setattr(sess.rx.sync, "packet", no_normal_packet_search)
    sess.rx.new_cycle()
    sess.rx.deep_scan(audio("ws8eoc-0908-p3-rms.wav"))
    assert [ev.packet[2] for ev in sess.events if ev.packet] == [b"RMS"]
    assert bytes(sess.host.channel(sess.host.ptchn).rx) == b"RMS"
    assert sess.host.arq.role == arq.IRS
    assert not sess.host.arq.entry_pending


@requires_ws8eoc_p3
def test_repeated_same_window_cannot_confirm_a_new_acquisition():
    sess = _Session(entry_pending=True)
    x = audio("ws8eoc-0908-p3-head-4.wav")
    sess.rx.new_cycle()
    for _ in range(3):
        assert sess.rx.control_signal(x, 660096-len(x), 653749) is None
    assert sess.host.arq.role == arq.ISS


def test_silent_acquisition_does_not_change_role():
    sess = _Session(entry_pending=True)
    for _ in range(4):
        sess.rx.new_cycle()
        assert sess.rx.control_signal(np.zeros(17000), 0, 10000) is None
    assert sess.host.arq.role == arq.ISS


@pytest.mark.parametrize("seed", range(12))
def test_noise_is_not_a_changeover(seed):
    x = np.random.default_rng(seed).normal(0, .1, round(1.3*FS))
    assert p3acquire.changeover(x) is None


@pytest.mark.parametrize("cs", [0, 1, 3, 4, 5])
@pytest.mark.parametrize("offset", [-60, 0, 60])
def test_other_p3_controls_are_not_changeovers(cs, offset):
    x = np.pad(placement.control_signal(cs), (4800, 4800))
    assert p3acquire.changeover(p3acquire.compensate(x, -offset)) is None


@pytest.mark.parametrize("cs", range(5))
@pytest.mark.parametrize("inverted", [False, True])
def test_p1_controls_are_not_p3_changeovers(cs, inverted):
    x = np.pad(pactor1.control_signal(cs, invert=inverted), (4800, 4800))
    assert p3acquire.changeover(x) is None


@pytest.mark.parametrize("offset", [-60, 60])
def test_acquired_offset_reads_following_data(offset):
    sess = _Session(entry_pending=True)
    head = np.pad(placement.changeover_packet(b"RMS", 0), (4800, 4800))
    sess.rx.new_cycle()
    sess.rx.deep_scan(p3acquire.compensate(head, -offset))
    assert bytes(sess.host.channel(sess.host.ptchn).rx) == b"RMS"
    packet = np.pad(placement.link_packet(1, b"abc", 1), (4800, 4800))
    sess.rx.new_cycle()
    sess.rx.deep_scan(p3acquire.compensate(packet, -offset))
    assert bytes(sess.host.channel(sess.host.ptchn).rx) == b"RMSabc"


def test_acquisition_state_does_not_outlive_the_link():
    sess = _Session(entry_pending=True)
    sess.rx.p3_receive_offset_hz = 50
    sess.rx._p3_head_candidate = (1, 0, 50)
    sess.rx._p3_changeover_pending = True
    sess.host.arq.state = arq.State.DISCONNECTED
    sess.rx.new_cycle()
    assert sess.rx.p3_receive_offset_hz == 0
    assert sess.rx._p3_head_candidate is None
    assert not sess.rx._p3_changeover_pending
