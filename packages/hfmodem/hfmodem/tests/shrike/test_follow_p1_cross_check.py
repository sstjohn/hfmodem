# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-3 transmit follow answers to the PACTOR-1 leg of the same link.

KB5LZK, 40 m, 2026-09-15: the peer's FSK tones read within 0.4 Hz of nominal and
its PACTOR-3 control acquired at +74.6 Hz, and `follow-offset=all` keyed 39 data
packets there, none of them read. A differential acquisition cannot tell a
displaced carrier from a per-symbol phase convention -- `_refined` takes the data
off with plain +-1 signs, so an injected +45 deg/symbol convention on a control
keyed at exactly 0 Hz acquires at +12.6 Hz and +270 deg at -25.1. The PACTOR-1
reader has no such ambiguity: it correlates FSK lines at a fixed 1400/1600 Hz and
searches time only, and a zero-error codeword therefore bounds the peer to
`p3acquire.P1_CROSS_CHECK_HZ`. Where the two disagree past that bound, the
transmitter stays put and the receive raster still follows.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import modem, onair, p1rx, p3acquire, pactor1, rxfront, spec
from hfmodem.tests.shrike.test_granted_entry_retry import granted

FS = spec.SAMPLE_RATE


def sessrx():
    host, _ = granted()
    return onair._SessionRx(host, tag='CROSS')


def p1_word(rx, *, errors=0):
    rx.cs_log.append(rxfront.Event(
        1.0, 'cs', f'CS1  ({errors} bit errors, PACTOR-1)',
        protocol=spec.Protocol.PACTOR1, cs=0))


# -- the cross-check ------------------------------------------------------

def test_a_follow_the_pactor1_read_contradicts_does_not_reach_the_transmitter():
    rx = sessrx()
    p1_word(rx)
    rx._follow_p3_offset(74.6)
    assert rx.p3_receive_offset_hz == 74.6      # the frames decoded there
    assert rx.p3_transmit_offset_hz == 0.0      # ...and we do not key there


def test_a_follow_the_pactor1_read_allows_is_applied():
    rx = sessrx()
    p1_word(rx)
    rx._follow_p3_offset(-25.0)
    assert rx.p3_receive_offset_hz == -25.0
    assert rx.p3_transmit_offset_hz == -25.0


def test_a_session_with_no_pactor1_codeword_follows_unchecked():
    rx = sessrx()
    rx._follow_p3_offset(74.6)
    assert rx.p3_transmit_offset_hz == 74.6


def test_a_codeword_read_with_bit_errors_is_not_a_frequency_measurement():
    rx = sessrx()
    p1_word(rx, errors=2)
    rx._follow_p3_offset(74.6)
    assert rx.p3_transmit_offset_hz == 74.6


def test_the_log_line_names_both_offsets(capsys):
    rx = sessrx()
    p1_word(rx)
    rx._follow_p3_offset(74.6)
    line = capsys.readouterr().out
    assert '+74.6' in line and 'PACTOR-1' in line and '60' in line


# -- the bound itself, measured on the reader that sets it -----------------

@pytest.mark.parametrize('cs', range(4))
def test_the_pactor1_reader_spans_the_bound_it_is_given(cs):
    """A codeword is a frequency measurement only as far as its filter reaches."""
    burst = np.concatenate([np.zeros(round(.05 * FS)),
                            pactor1.control_signal(cs),
                            np.zeros(round(.10 * FS))])
    inside = p3acquire.P1_CROSS_CHECK_HZ - 2.5
    for hz in (inside, -inside):
        got = p1rx.decode_control_signal(p3acquire.compensate(burst, -hz), .05, .12)
        assert got is not None and got.errors == 0 and got.index == cs
    # The span is asymmetric per codeword, so the far edge is where every one of
    # them has given out rather than where the first does.
    for hz in (80.0, -80.0):
        got = p1rx.decode_control_signal(p3acquire.compensate(burst, -hz), .05, .12)
        assert got is None or got.errors or got.index != cs


# -- 2b: what a phase convention looks like to the acquisition -------------

def _control(cs, *, offset_hz=0.0, convention_deg=0.0):
    word = spec.CONTROL_SIGNALS[cs]
    bits = np.array([(word >> i) & 1 for i in range(spec.CS_BITS_PER_TONE)], np.uint8)
    steps = np.where(bits == 1, np.pi, 0.0) + np.deg2rad(convention_deg)
    syms = np.exp(1j * np.r_[0.0, np.cumsum(steps)])
    audio = modem.modulate_tones(
        {5: syms, 12: syms}, modem.ModConfig(sample_rate=FS, matched_pulse=True),
        delay={12: round(.005 * FS)})
    audio = np.concatenate([np.zeros(round(.05 * FS)), audio, np.zeros(round(.10 * FS))])
    return p3acquire.compensate(audio, -offset_hz) if offset_hz else audio


@pytest.mark.parametrize('convention_deg,reported_hz', [(0.0, 0.0), (45.0, 12.6),
                                                        (-45.0, -12.6), (90.0, 25.1)])
def test_the_acquisition_reports_a_phase_convention_as_a_frequency(
        convention_deg, reported_hz):
    """A control keyed at exactly 0 Hz, read as if it were displaced."""
    got = p3acquire.control_signal(_control(1, convention_deg=convention_deg))
    assert got is not None
    assert got.offset_hz == pytest.approx(reported_hz, abs=.5)


def test_a_control_signal_carries_no_convention_so_the_follow_carries_no_bias():
    """Which is why the 40 m +74.6 Hz reading is a frequency and not this."""
    for offset in (0.0, 25.0, -50.0):
        got = p3acquire.control_signal(_control(1, offset_hz=offset))
        assert got is not None
        assert got.offset_hz == pytest.approx(offset, abs=.5)


# -- and the transmitter reads the qualified number, not the raster --------

def _tx_stub(sessrx, follow='all'):
    return SimpleNamespace(p3_follow_offset=follow, sessrx=sessrx,
                           _p3_shifted={}, _tx_offset_hz=None)


@pytest.mark.parametrize('transmit_hz,keyed_hz', [(0.0, 0.0), (-25.0, -25.0)])
def test_the_transmitter_keys_the_qualified_offset(transmit_hz, keyed_hz):
    tx = _tx_stub(SimpleNamespace(p3_receive_offset_hz=74.6,
                                  p3_transmit_offset_hz=transmit_hz))
    onair.RadioTx._p3_offset(tx, np.zeros(4800), control=False)
    assert tx._tx_offset_hz == keyed_hz


def test_a_session_object_without_the_qualifier_keys_its_receive_offset():
    """Every existing caller and test double, unchanged."""
    tx = _tx_stub(SimpleNamespace(p3_receive_offset_hz=-25.0))
    onair.RadioTx._p3_offset(tx, np.zeros(4800), control=False)
    assert tx._tx_offset_hz == -25.0
