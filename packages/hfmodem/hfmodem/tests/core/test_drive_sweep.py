# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The drive sweep: the numbers it reports, and the transmitter it must not leave up.

`tools/drive_sweep.py` is the instrument for one measurement nobody has taken. The
station's power fell about tenfold on the upgrade from PACTOR-1 to PACTOR-3, and
the cause is in this tree rather than in the radio: `levels.at_drive` normalises
every burst to the same PEAK, PACTOR-1 is near constant-envelope and PACTOR-3 has
a 12.8 dB crest factor, so the same peak is 9.8 dB less average power. How much of
that a higher drive buys back before the rig's ALC starts acting is a wattmeter
question, and the tool exists to key for it.

It read 14.5 dB and 11.5 until `placement.START_PHASE`: the packet's whole peak
was one symbol every carrier entered at the same angle, and the drive was being
set by it.

So two things are tested here, and the second is why the file is long:

  * **The figures.** The waveforms are the modems' own renderers and the drive is
    the modems' own normaliser -- a sweep measured on a hand-rolled tone would be
    a measurement of something the station never transmits -- and the crest and
    relative-power arithmetic printed beside the operator's reading is the
    arithmetic those waveforms actually produce.
  * **The keying.** Every bound is refused before the rig object exists, the
    playback is unkeyed by a `finally` and by a deadman that needs nothing from
    the playback, and the run proves the line low on the way out. A transmitting
    tool that fails these is worse than no tool.

Off the air entirely: no serial port, no sound card, no rigctld. The rig is a fake
that records what was asked of it, and the play call is a parameter.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import levels
from hfmodem.shrike import pactor1, placement

REPO = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO / "tools")]
DS = pytest.importorskip("drive_sweep", reason="tools/ is not beside the package")

#: A character device the arm gate accepts and nothing here opens.
PTT_LINE = "/dev/null"

STATION_DRIVE = levels.TX_DRIVE


@pytest.fixture
def reference() -> float:
    return DS.reference_rms(STATION_DRIVE)


def _armed_argv(*extra: str) -> list[str]:
    return ["drive_sweep", "--seconds", "0.2", "--dummy-load", "--arm",
            "--rigctld", "127.0.0.1:4532", "--expect-model", "FT-891",
            "--ptt-device", PTT_LINE, *extra]


def _args(*argv: str):
    return DS.parser().parse_args(list(argv))


class FakeRig:
    """A rig that answers and remembers, and nothing that reaches hardware."""

    def __init__(self, *a, key_up: bool = True, **kw):
        self.keys: list[bool] = []
        self.forced: list[str] = []
        self.shutdowns = 0
        self.retired = False
        self._key_up = key_up

    def key(self, on: bool, why: str = "") -> bool:
        self.keys.append(on)
        return self._key_up if on else True

    def unkey_hard(self, why: str) -> bool:
        self.forced.append(why)
        self.keys.append(False)
        return True

    def shutdown(self) -> None:
        self.shutdowns += 1


# -- the waveforms ----------------------------------------------------------

def test_the_waveforms_are_the_modems_own_renderers():
    """A sweep against a hand-rolled tone measures a waveform this station never
    transmits, and the whole question is what the PACTOR-3 envelope does."""
    _, one = DS.burst("pactor1")
    assert np.array_equal(one, np.trim_zeros(np.asarray(
        pactor1.data_signal(DS._filler(DS._P1_BYTES), 100, repeats=1,
                            lead_s=0.0, tail_s=0.0), float)))

    path = placement.SPEED_PATHS[3]
    _, three = DS.burst("pactor3", 3)
    assert np.array_equal(three, np.trim_zeros(np.asarray(
        placement.data_packet(DS._filler(path.crc_bytes - 2), path), float)))


def test_the_pactor1_burst_is_keyed_without_the_silence_of_its_own_arq_slot():
    """`data_signal` puts one 0.96 s burst at the head of a 1.25 s slot. Keying
    the rest of the slot would measure a duty cycle rather than an envelope, and
    the reference would read 1.2 dB low for a reason nothing in the table shows."""
    _, one = DS.burst("pactor1")
    assert 0.9 < len(one) / DS.FS < 1.0


@pytest.mark.parametrize("kind,level,crest,rms", [("pactor1", 0, 3.01, 0.4243),
                                                  ("pactor3", 3, 12.82, 0.1372)])
def test_the_two_waveforms_carry_the_crest_factors_the_drop_was_measured_at(
        kind, level, crest, rms, reference):
    """The figures in the tool's own docstring, which are the reason it exists:
    one packet of each, at the drive this station transmits at. The PACTOR-3
    figures moved from 11.77 / 0.1548 when data packets gained the 82nd symbol
    every recorded station keys (2026-09-02), and from 11.76 / 0.1550 when
    `placement.PROTOCOL_RISE` carried the protocol's own pulse onto them
    (2026-09-13) -- three symbols of kernel where the raised cosine spans eight,
    which a fourteen-carrier comb pays about a decibel of crest for."""
    _, one = DS.burst(kind, level)
    s = DS.Step(STATION_DRIVE, levels.at_drive(one, STATION_DRIVE), reference)
    assert s.crest_db == pytest.approx(crest, abs=0.01)
    assert s.rms == pytest.approx(rms, abs=0.0001)


def test_the_speed_level_selects_the_path_it_names(reference):
    """Each level is a different tone count, so each is a different envelope --
    and a sweep that silently keyed level 3 whatever was asked for would report
    one waveform's numbers under another's name.

    The crests used to climb with the level and no longer do:
    `placement.START_PHASE` takes 2 to 2.7 dB off every level wide enough for its
    phase reference to have set the peak, which is all of them but the two-carrier
    one. Two carriers cannot be spread -- they sweep through alignment inside every
    symbol whatever angle they start at -- so speed level 1 is what still stands
    apart, and the ordering above it is now the payload's rather than the comb's.
    """
    crests = [DS.step("pactor3", sl, STATION_DRIVE, 1.0, reference).crest_db
              for sl in placement.SPEED_PATHS]
    assert len(set(round(c, 3) for c in crests)) == len(crests), crests
    assert crests[0] < min(crests[1:]) - 2, crests


# -- the drive --------------------------------------------------------------

@pytest.mark.parametrize("kind,level", [("pactor1", 0), ("pactor3", 3)])
@pytest.mark.parametrize("drive", [0.2, 0.6, 1.0])
def test_every_step_leaves_at_the_peak_it_was_asked_for(kind, level, drive,
                                                        reference):
    s = DS.step(kind, level, drive, 0.5, reference)
    assert s.peak == pytest.approx(drive, abs=1e-6)
    assert s.audio.dtype == np.float32


def test_the_drive_is_applied_through_the_modems_own_normaliser(monkeypatch,
                                                                reference):
    """`levels.at_drive` and not a second copy of it: the point of the sweep is
    to move the number the on-air path uses, so a private scaling here would be
    measuring a level nothing else can be set to."""
    seen: list[float] = []
    real = levels.at_drive
    monkeypatch.setattr(DS.levels, "at_drive",
                        lambda a, d: seen.append(d) or real(a, d))
    DS.step("pactor3", 3, 0.42, 0.5, reference)
    assert seen == [0.42]


def test_a_step_is_exactly_the_key_down_it_was_asked_for(reference):
    """The burst is 0.89 s and the step is whatever the operator asked for, so
    the tiling has to end mid-packet rather than at a packet boundary. A step
    that ran to the next boundary would key past its own bound."""
    for seconds in (0.5, 1.0, 4.0):
        assert DS.step("pactor3", 3, 0.6, seconds, reference).seconds == seconds


def test_a_tiled_step_measures_the_same_envelope_as_one_packet(reference):
    """What is reported is measured on what goes to the card, and the tiling must
    not change it -- otherwise the table would drift with the step length."""
    short = DS.step("pactor3", 3, 0.6, 1.0, reference)
    long = DS.step("pactor3", 3, 0.6, 4.0, reference)
    assert short.crest_db == pytest.approx(long.crest_db, abs=0.2)


# -- the arithmetic beside the wattmeter reading ----------------------------

def test_the_crest_factor_is_the_peak_over_the_rms_in_dB(reference):
    s = DS.step("pactor3", 3, 0.6, 1.0, reference)
    a = np.asarray(s.audio, float)
    assert s.crest_db == pytest.approx(
        20 * np.log10(np.abs(a).max() / np.sqrt((a ** 2).mean())))
    assert s.dbfs == pytest.approx(20 * np.log10(np.sqrt((a ** 2).mean())))


def test_the_reference_is_pactor1_at_the_drive_the_station_transmits_at(reference):
    """Zero on the PACTOR-1 row at the station's own drive: the column reads how
    far this step is from what goes out today, so today has to read zero."""
    assert DS.step("pactor1", 0, STATION_DRIVE, 1.0, reference).vs_pactor1_db \
        == pytest.approx(0.0, abs=1e-9)


def test_pactor3_reports_the_drop_that_is_left_after_the_phase_reference(reference):
    """The tenfold drop this tool was written for, less the part that was a defect.

    -11.5 dB is what the operator measured and what this read until
    `placement.START_PHASE` stopped fourteen carriers entering the phase-reference
    symbol at the same angle. That one symbol was the whole packet's peak, so
    `levels.at_drive` set the drive by it and keyed the other eighty-eight below
    what was asked for. What is left is the gap a fourteen-tone comb has against
    one FSK tone, and no phase set reaches it.
    """
    s = DS.step("pactor3", 3, STATION_DRIVE, 1.0, reference)
    assert s.vs_pactor1_db == pytest.approx(-9.83, abs=0.2)


def test_the_relative_power_follows_the_drive_it_was_sent_at(reference):
    """Twice the peak is 6 dB more average power, since the envelope is unchanged
    -- which is what says how much of the drop the headroom can buy back."""
    at_low = DS.step("pactor3", 3, 0.4, 1.0, reference).vs_pactor1_db
    at_high = DS.step("pactor3", 3, 0.8, 1.0, reference).vs_pactor1_db
    assert at_high - at_low == pytest.approx(6.02, abs=0.01)


def test_the_report_prints_the_numbers_the_reading_is_written_beside(reference):
    lines: list[str] = []
    steps = [DS.step("pactor3", 3, d, 1.0, reference) for d in (0.6, 0.8)]
    DS.report("PACTOR-3 speed level 3", steps, STATION_DRIVE, lines.append)
    body = [ln for ln in lines if ln.strip().startswith(("1", "2"))]
    assert len(body) == 2
    for line, s in zip(body, steps):
        for figure in (f"{s.drive:.2f}", f"{s.peak:.3f}", f"{s.rms:.4f}",
                       f"{s.crest_db:.2f}", f"{s.vs_pactor1_db:.2f}"):
            assert figure in line
    assert "wattmeter" in "\n".join(lines)


# -- what it refuses to do --------------------------------------------------

@pytest.mark.parametrize("argv,expect", [
    (("--seconds", "30"), "--seconds"),
    (("--seconds", "0"), "--seconds"),
    (("--drive", "1.4"), "--drive"),
    (("--drive", "0"), "--drive"),
])
def test_a_step_outside_its_bounds_is_refused(argv, expect):
    args = _args(*argv)
    why = DS.refusal(args, args.drive or [STATION_DRIVE])
    assert why and expect in why


def test_a_sweep_longer_than_the_total_bound_is_refused():
    """Each step is legal and the run is not: the cap is on key-down across the
    whole sweep, or a long enough list of drives walks around the per-step one."""
    drives = [0.1 * i for i in range(1, 10)]
    args = _args("--seconds", str(DS.MAX_STEP_S))
    why = DS.refusal(args, drives)
    assert why and f"{DS.MAX_TOTAL_S:g}s" in why


def test_the_bounds_leave_room_for_the_run_the_tool_is_for():
    """A cap that refused the ordinary sweep would be a cap nobody keeps."""
    args = _args("--seconds", str(DS.DEFAULT_STEP_S))
    assert DS.refusal(args, [0.6, 0.7, 0.8, 0.9, 1.0]) is None


def test_arming_is_refused_without_a_load_acknowledgement():
    """The tool keys at drives chosen to find where the ALC acts. Into an antenna
    that is a splattering transmitter on a shared band."""
    args = _args(*_armed_argv()[1:], )
    args.dummy_load = False
    args.clear_frequency = False
    why = DS.refusal(args, [STATION_DRIVE])
    assert why and "--dummy-load" in why and "--clear-frequency" in why


def test_a_dry_run_needs_no_acknowledgement_because_it_keys_nothing():
    assert DS.refusal(_args("--waveform", "pactor3"), [STATION_DRIVE]) is None


@pytest.mark.parametrize("drop", ["--rigctld", "--expect-model", "--ptt-device"])
def test_arming_is_refused_without_the_radio_it_would_key(drop):
    """A radio nobody can name, or a keying line nothing can pull down when
    rigctld stops answering -- `core.ptt.arming_refusal`'s gate, the one every
    transmit tool here runs."""
    argv = _armed_argv()[1:]
    i = argv.index(drop)
    del argv[i:i + 2]
    why = DS.refusal(_args(*argv), [STATION_DRIVE])
    assert why and (drop in why or "keying line" in why)


def test_an_armed_run_with_everything_named_is_not_refused():
    assert DS.refusal(_args(*_armed_argv()[1:]), [STATION_DRIVE]) is None


def test_a_refused_run_never_builds_a_rig(monkeypatch):
    """The refusal has to happen before anything that can key exists. A gate
    downstream of the rig is a gate with a transmitter behind it."""
    def explode(*a, **kw):
        raise AssertionError("a refused run reached the rig")

    monkeypatch.setattr(DS, "Rig", explode)
    monkeypatch.setattr(DS, "_install_unkey_handlers", explode)
    argv = _armed_argv()
    argv.remove("--dummy-load")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        DS.main()
    assert exc.value.code == 2


# -- the keying ------------------------------------------------------------

def _sweep(rig, steps, play, **kw):
    lines: list[str] = []
    code = DS.sweep(steps, rig, lines.append, device="fake", pause=0.0,
                    play=play, **kw)
    return code, lines


def test_every_step_is_keyed_and_unkeyed_in_turn(reference):
    rig = FakeRig()
    steps = [DS.step("pactor3", 3, d, 0.2, reference) for d in (0.6, 0.8)]
    played: list[int] = []
    code, _ = _sweep(rig, steps, lambda a, fs, dev: played.append(len(a)))
    assert code == 0
    assert rig.keys == [True, False, True, False]
    assert played == [len(s.audio) for s in steps]


def test_the_key_comes_down_even_when_the_playback_raises(reference):
    """The one failure this tool must not have. A card that throws mid-step is
    exactly when nothing else is going to bring the transmitter down."""
    rig = FakeRig()
    steps = [DS.step("pactor1", 0, 0.6, 0.2, reference)]

    def boom(*a):
        raise OSError("the codec went away")

    with pytest.raises(OSError):
        _sweep(rig, steps, boom)
    assert rig.keys == [True, False]


def test_a_step_that_outlives_its_audio_is_unkeyed_by_the_deadman(monkeypatch,
                                                                  reference):
    """The deadman asks nothing of the playback it is guarding: a hung `play` and
    a card that has stopped are one failure seen from two sides."""
    monkeypatch.setattr(DS, "DEADMAN_MARGIN_S", 0.01)
    rig = FakeRig()
    fired = threading.Event()
    real_unkey = rig.unkey_hard
    rig.unkey_hard = lambda why: (fired.set(), real_unkey(why))[1]

    def stall(*a):
        assert fired.wait(10.0), "the deadman never fired on a stalled step"

    code, lines = _sweep(rig, [DS.step("pactor1", 0, 0.6, 0.05, reference)], stall)
    assert code == 0
    assert rig.forced and "deadman" in rig.forced[0]
    assert any("DEADMAN" in ln for ln in lines)


def test_the_deadman_stands_down_when_the_step_ends_on_time(monkeypatch,
                                                            reference):
    """An alarm that fires on the ordinary case is one an operator reads past."""
    monkeypatch.setattr(DS, "DEADMAN_MARGIN_S", 0.05)
    rig = FakeRig()
    code, _ = _sweep(rig, [DS.step("pactor1", 0, 0.6, 0.05, reference)],
                     lambda *a: None)
    threading.Event().wait(0.3)
    assert code == 0 and rig.forced == []


def test_a_transmitter_that_never_came_up_ends_the_sweep(reference):
    """`Keyer` has already dropped PTT to be sure by the time a key-up reports
    False, and there is nothing to measure with the rig down, so the remaining
    steps are not keyed."""
    rig = FakeRig(key_up=False)
    steps = [DS.step("pactor1", 0, 0.6, 0.2, reference)] * 3
    code, lines = _sweep(rig, steps, lambda *a: None)
    assert code == 4
    assert rig.keys == [True]
    assert any("never came up" in ln for ln in lines)


def test_a_retired_rig_ends_the_sweep(reference):
    """Something took the rig down mid-sweep -- a forced unkey, a transport that
    stopped answering. Nothing may key through it again, remaining steps or not."""
    rig = FakeRig()
    steps = [DS.step("pactor1", 0, 0.6, 0.2, reference)] * 3

    def play(*a):
        rig.retired = True

    code, _ = _sweep(rig, steps, play)
    assert code == 4
    assert rig.keys == [True, False]


# -- the verdict on the way out --------------------------------------------

def test_the_line_is_proved_low_off_the_line_itself(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(DS, "drop_rts", lambda dev, log: True)
    assert DS.prove_down(PTT_LINE, lines.append) is True
    assert "PTT VERDICT: down" in "\n".join(lines)


def test_a_line_that_will_not_read_low_says_so_where_it_cannot_be_missed(
        monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(DS, "drop_rts", lambda dev, log: False)
    assert DS.prove_down(PTT_LINE, lines.append) is False
    assert "NOT CONFIRMED DOWN" in "\n".join(lines)


def test_an_absent_keying_line_is_not_an_alarm(tmp_path, monkeypatch):
    """Nothing was ever keyed through a line that is not there, and an alarm that
    fires whenever the adapter is unplugged is one nobody reads."""
    monkeypatch.setattr(DS, "drop_rts",
                        lambda dev, log: pytest.fail("opened an absent line"))
    assert DS.prove_down(None, lambda _: None) is True
    assert DS.prove_down(str(tmp_path / "not-a-device"), lambda _: None) is True


# -- the run, end to end, with the radio faked out -------------------------

@pytest.fixture
def armed(monkeypatch):
    """An armed run whose rig, codec and keying line are all fakes."""
    rig = FakeRig()
    monkeypatch.delenv("HFMODEM_STATION", raising=False)
    monkeypatch.setattr(DS, "Rig", lambda *a, **kw: rig)
    monkeypatch.setattr(DS, "prepare_rig", lambda *a: None)
    monkeypatch.setattr(DS, "warm_output", lambda dev: None)
    monkeypatch.setattr(DS, "_install_unkey_handlers", lambda r: None)
    monkeypatch.setattr(DS, "prove_down", lambda dev, log: True)
    monkeypatch.setattr(sys, "argv", _armed_argv("--waveform", "pactor3"))
    return rig


def test_an_armed_run_hands_the_rig_to_the_trees_own_signal_handlers(monkeypatch,
                                                                     armed):
    """SIGINT, SIGTERM and SIGHUP through `Rig.panic` -- retire, arm the deadman,
    unkey -- rather than a second set of handlers to keep in step by hand."""
    handed: list = []
    monkeypatch.setattr(DS, "_install_unkey_handlers", handed.append)
    monkeypatch.setattr(DS, "sweep", lambda *a, **kw: 0)
    assert DS.main() == 0
    assert handed == [armed]


def test_a_run_that_fails_mid_sweep_still_shuts_the_rig_down_and_proves_the_line(
        monkeypatch, armed):
    proved: list = []
    monkeypatch.setattr(DS, "prove_down", lambda dev, log: proved.append(dev) or True)

    def boom(*a, **kw):
        raise RuntimeError("the sweep fell over")

    monkeypatch.setattr(DS, "sweep", boom)
    with pytest.raises(RuntimeError):
        DS.main()
    assert armed.shutdowns == 1
    assert proved == [PTT_LINE]


def test_a_run_that_cannot_prove_the_line_low_does_not_exit_clean(monkeypatch,
                                                                  armed):
    """The sweep itself succeeded; the line did not read low. A zero here would
    tell a script the transmitter is down when nothing established that."""
    monkeypatch.setattr(DS, "prove_down", lambda dev, log: False)
    monkeypatch.setattr(DS, "sweep", lambda *a, **kw: 0)
    assert DS.main() == 3


def test_a_dry_run_keys_nothing_and_says_the_figures_stand_alone(monkeypatch,
                                                                 capsys):
    monkeypatch.delenv("HFMODEM_STATION", raising=False)
    monkeypatch.setattr(DS, "Rig", lambda *a, **kw: pytest.fail("a dry run built a rig"))
    monkeypatch.setattr(sys, "argv", ["drive_sweep", "--waveform", "pactor3"])
    assert DS.main() == 2
    out = capsys.readouterr().out
    assert "DISARMED" in out and "wattmeter" in out
