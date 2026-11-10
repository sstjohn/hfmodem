# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`--p3-control-phase`: our controls on the axis, or on stock's diagonal.

KB2PCN, 20 m, 2026-11-08 23:13: its CS3 and CS4 step -48.4 degrees a symbol and
its PACTOR-1 tones read -1.0 Hz; our follow read its controls at -13.6. Our
frames step -45 at +0 Hz and our controls 0/180, so the follow carried the
diagonal as -12.5 Hz into everything we keyed after it. `diagonal` keys the
controls on 315/135 and takes the convention back off a follow read off a word.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import modem, onair, p3acquire, placement, rx, spec
from hfmodem.tests.shrike.test_follow_p1_cross_check import p1_word, sessrx

FS = spec.SAMPLE_RATE
SPS = FS // 100
LEAD = 2400
START = 3180
"""Where a control padded by `LEAD` has its phase reference, as
`rx.decode_control_signal` counts it."""

SHAPES = [(tail, swapped) for tail in (False, True) for swapped in (None, False, True)]


@pytest.fixture
def phase(monkeypatch):
    def set_to(value):
        monkeypatch.setattr(placement, "CONTROL_PHASE", value)
    set_to("axis")
    return set_to


def _padded(cs, **shape):
    burst = placement.control_burst(cs, **(shape or dict(tail=False, swapped=None)))
    return np.concatenate([np.zeros(LEAD), burst, np.zeros(2 * LEAD)])


def _steps(audio, start=START):
    """Each header tone's differential products, one row per channel."""
    pulse = rx._pulse(SPS)
    idx = start + np.arange(spec.CS_BITS_PER_TONE + 1) * SPS + (len(pulse) - 1) // 2
    out = []
    for cn in rx.HEADER_TONES:
        y = rx._baseband(audio, cn, FS, pulse)[idx]
        out.append(y[1:] * np.conj(y[:-1]))
    return np.array(out)


def _bits(cs):
    w = spec.CONTROL_SIGNALS[cs]
    return np.array([(w >> i) & 1 for i in range(spec.CS_BITS_PER_TONE)], bool)


@pytest.mark.parametrize("cs", range(6))
def test_diagonal_controls_step_315_and_135_and_read_as_the_same_word(phase, cs):
    phase("diagonal")
    audio = _padded(cs)
    d = _steps(audio)
    # Each step within the pulse's own intersymbol spread of its point (stock's
    # spread as much, -37 to -55 on KB2PCN), and the word's mean on it exactly.
    want = np.where(_bits(cs), 135.0, -45.0)
    got = np.degrees(np.angle(d))
    assert np.abs((got - want + 180) % 360 - 180).max() < 20.0
    signs = np.where(_bits(cs), -1.0, 1.0)
    assert np.degrees(np.angle(np.sum(d * signs))) == pytest.approx(-45.0, abs=1.5)
    assert rx.decode_control_signal(audio, START) == (cs, 0)
    found = p3acquire.control_signal(audio)
    assert found is not None and found.event.cs == cs
    # ...and the acquisition reads the convention as the frequency it is.
    assert found.offset_hz == pytest.approx(placement.DIAGONAL_HZ, abs=.2)


@pytest.mark.parametrize("cs", range(6))
@pytest.mark.parametrize("tail,swapped", SHAPES)
def test_diagonal_holds_every_control_shape_to_its_word(phase, cs, tail, swapped):
    phase("diagonal")
    burst = placement.control_burst(cs, tail=tail, swapped=swapped)
    assert len(burst) == len(placement._control_burst(cs, tail, swapped, None, "axis"))
    found = p3acquire.control_signal(np.concatenate([np.zeros(LEAD), burst,
                                                     np.zeros(2 * LEAD)]))
    assert found is not None and found.event.cs == cs


@pytest.mark.parametrize("cs", range(6))
@pytest.mark.parametrize("tail,swapped", SHAPES)
def test_axis_is_the_control_we_have_always_keyed(cs, tail, swapped):
    syms = modem.differential_encode(_bits(cs).astype(np.uint8), 1)
    if tail:
        syms = np.r_[syms, syms[-1]]
    built = modem.modulate_tones(
        {5: syms, 12: syms},
        modem.ModConfig(sample_rate=FS, matched_pulse=True),
        delay=None if swapped is None else
        {5 if swapped else 12: round(placement.CONTROL_STAGGER_S * FS)})
    burst = placement.control_burst(cs, tail=tail, swapped=swapped)
    assert burst.tobytes() == built.tobytes()
    assert np.degrees(np.angle(_steps(_padded(cs)))).round() % 180 == pytest.approx(0)


def test_frames_and_the_changeover_do_not_move_with_the_controls(phase):
    axis = (placement.changeover_packet(b";FW", 0, swapped=False),
            placement.link_packet(2, b"fresh", 0x23),
            placement.link_packet(1, b"fresh", 0x21))
    phase("diagonal")
    diagonal = (placement.changeover_packet(b";FW", 0, swapped=False),
                placement.link_packet(2, b"fresh", 0x23),
                placement.link_packet(1, b"fresh", 0x21))
    for a, d in zip(axis, diagonal):
        assert a.tobytes() == d.tobytes()


def test_the_diagonal_control_shares_the_changeover_heads_steps(phase):
    """The head was already on the diagonal; now the bare word is too."""
    head = placement.changeover_packet(b";FW", 0, swapped=False)
    window = np.concatenate([np.zeros(LEAD), head])[:LEAD + round(.3 * FS)]
    found = p3acquire.control_signal(window)
    assert found is not None and found.event.cs == placement.BREAKIN_CS
    assert p3acquire.control_signal(_padded(placement.BREAKIN_CS)).offset_hz == 0.0
    phase("diagonal")
    bare = p3acquire.control_signal(_padded(placement.BREAKIN_CS))
    assert bare.offset_hz == pytest.approx(found.offset_hz, abs=.5)


# -- the follow -------------------------------------------------------------

def test_kb2pcn_follow_keys_its_carrier_under_diagonal(phase):
    phase("diagonal")
    s = sessrx()
    p1_word(s)
    s._follow_p3_offset(-13.6)
    assert s.p3_receive_offset_hz == -13.6
    assert s.p3_transmit_offset_hz == pytest.approx(-1.1, abs=1e-9)


def test_kb2pcn_follow_keys_the_raster_under_axis():
    s = sessrx()
    p1_word(s)
    s._follow_p3_offset(-13.6)
    assert s.p3_transmit_offset_hz == -13.6


def test_a_follow_read_off_a_frame_carries_no_convention(phase):
    phase("diagonal")
    s = sessrx()
    p1_word(s)
    s._follow_p3_offset(-1.0, word=False)
    assert s.p3_transmit_offset_hz == -1.0
    s._follow_p3_offset(-13.6)
    assert s.p3_transmit_offset_hz == pytest.approx(-1.1, abs=1e-9)
    s._follow_p3_offset(-1.4, word=False)
    assert s.p3_transmit_offset_hz == -1.4


def test_a_session_reset_forgets_the_convention(phase):
    phase("diagonal")
    s = sessrx()
    s._follow_p3_offset(-13.6)
    s.host.arq.state = onair.State.DISCONNECTED
    s.new_cycle()
    assert s.p3_receive_offset_hz == 0.0 == s.p3_transmit_offset_hz


def test_the_line_names_the_carrier_and_the_bound(phase, capsys):
    phase("diagonal")
    s = sessrx()
    p1_word(s)
    s._follow_p3_offset(-13.6)
    line = capsys.readouterr().out
    assert "-13.6" in line and "carrier -1.1 Hz" in line and "+-8 Hz" in line
    assert "within" in line and "agreed, TX follows" in line
    s._follow_p3_offset(-25.0)
    line = capsys.readouterr().out
    assert "past the diagonal's +-8 Hz" in line and "TX stays at +0 Hz" in line
    assert "CONTRADICTED" not in line


# -- the agreement bound ----------------------------------------------------

@pytest.mark.parametrize("carrier,agreed", [(0.0, True), (7.5, True), (-7.5, True),
                                            (8.5, False), (-8.5, False),
                                            (-50.0, False)])
def test_diagonal_agrees_within_eight_hertz_of_the_carrier(phase, carrier, agreed):
    phase("diagonal")
    s = sessrx()
    p1_word(s)
    s._follow_p3_offset(carrier + placement.DIAGONAL_HZ)
    assert s._p3_offset_contradicts_p1 is not agreed
    assert s.p3_transmit_offset_hz == pytest.approx(carrier if agreed else 0.0, abs=1e-9)


def test_axis_keeps_the_pactor1_bound():
    s = sessrx()
    p1_word(s)
    s._follow_p3_offset(-50.0)
    assert not s._p3_offset_contradicts_p1
    assert s.p3_transmit_offset_hz == -50.0


def test_the_keyed_control_reads_where_kb2pcn_did(phase):
    """Followed at KB2PCN's -13.6, our diagonal CS4 keys at its -1.1 Hz carrier,
    and our own reader finds it where it found KB2PCN's: -13.5 and -13.6."""
    phase("diagonal")
    s = sessrx()
    p1_word(s)
    s._follow_p3_offset(-13.6)
    tx = SimpleNamespace(p3_follow_offset="all", sessrx=s, _p3_shifted={},
                         _tx_offset_hz=None)
    keyed = onair.RadioTx._p3_offset(tx, _padded(3), control=True)
    assert tx._tx_offset_hz == pytest.approx(-1.1, abs=1e-9)
    found = p3acquire.control_signal(keyed)
    assert found is not None and found.event.cs == 3
    assert found.offset_hz == pytest.approx(-13.6, abs=.3)


# -- the flag ---------------------------------------------------------------

def test_the_flag_defaults_to_axis():
    parser = onair.build_parser()
    assert parser.parse_args([]).p3_control_phase == "axis"
    assert parser.parse_args(["--p3-control-phase", "diagonal"]).p3_control_phase == "diagonal"
    with pytest.raises(SystemExit):
        parser.parse_args(["--p3-control-phase", "both"])


@pytest.mark.parametrize("argv,named", [((), "phase=axis"),
                                        (("--p3-control-phase", "diagonal"), "phase=diagonal")])
def test_a_session_states_the_choice(phase, argv, named):
    from hfmodem.tests.shrike.test_grid import _session
    got = _session(cycles=1, hold=0, charge=0, peer=False, extra_argv=argv)
    assert named in got["log"]
    assert ("--p3-control-phase diagonal:" in got["log"]) is bool(argv)
    assert placement.CONTROL_PHASE == ("diagonal" if argv else "axis")
