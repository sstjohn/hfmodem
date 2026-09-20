# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the channel-occupancy check must not get wrong before we key a transmitter.

Four defects are pinned here, each of which a previous version of this check
actually had:

  * **A dead receiver read as a clear channel.** Both earlier scores were
    ``max - min`` over the window, which is 0.0 dB on digital silence — the most
    confidently clear verdict the detector could produce, on no evidence at all.
    This is the dangerous one: a muted or unplugged input is exactly the state in
    which we cannot hear the station we would be transmitting over.
  * **A steady occupant read as clear.** ``max - min`` measured 0.9 dB on a
    wideband signal that never stops keying and 0.9 dB on an unmodulated carrier,
    because neither varies. A detector that only sees things switch on and off is
    blind to everything that is already on.
  * **Calibration on one signal class.** The first version was tuned on a narrow
    carrier and went blind to a 2300 Hz data occupant; the metric that replaced it
    was tuned on wideband traffic. Both the wide and the narrow case are asserted
    here so neither can be traded away for the other again.
  * **Calibration on one quiet capture.** ``burst`` was a plain percentile spread
    of in-band level, and it was measured against 30 s of one quiet frequency.
    Static crashes put 3.1-3.8 dB of that spread into an empty 40 m evening, so on
    2026-08-09 every channel of the slot read occupied by +0.6 to +0.7 dB — and a
    guard that fires on everything is one the operator overrides on everything.
    Both sides of that slot are recordings now, and both are asserted below.

Measured populations behind the thresholds, over 8 s windows:

    quiet band audio, 40 m and out of band   burst <= 5.93  shape <= 4.73  tone <= 3.33
    the same on 80 m, six windows            burst <= 9.98  shape <= 3.99  tone <= 1.87
    every verified-empty window, 78 of them  burst <= 9.99  shape <= 8.47  tone <= 3.57
    the occupant each score catches          burst    7.86  shape    6.96  tone    8.89
    thresholds                               burst    6.6   shape    6.0   tone    5.0
    off-air VARA gateway sessions            no two consecutive clear windows

The 80 m row is why ``burst`` is not in ``core.busy.DECIDING``: it is the only
one of the three whose quiet side runs past the occupant it alone catches, and no
hold length pulls the two apart.

The 78-window row is why ``shape`` is pinned on a rate and not on a gap. The two
rows above it are the seven windows its 6.0 was set against, and on all the empty
audio the corpus holds it reaches 8.47 — past the threshold, not short of it. Two
of the 78 read busy, and that is the price paid for the occupant classes only
``shape`` sees; raising it clear of the tail would give back more than half of
them.

A fifth case is pinned below that is not a defect of this code but a defect in
reading it: a channel carrying a bare carrier and nothing else reads busy, on
`tone` alone, and looks like a false alarm to anyone who checks it against a modem
classifier — which is silent on a carrier by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora

# `channel_busy` moved into the package as `core.busy`; loading it through the
# out-of-package harness made this whole module importorskip, so the
# listen-before-transmit guard had no executing tests at all.
from hfmodem.core import busy as CB
from hfmodem.core.occupied import occupied_hz
FS = 48000
WIN = 8 * FS


def _wav(path):
    x = corpora.wav_mono(path)
    return x / 32768.0 if np.abs(x).max() > 1.5 else x


def _windows(x, n=WIN):
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)]


def _inband_power(x):
    f = np.fft.rfftfreq(len(x), 1 / FS)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return p[(f > 400) & (f < 2700)].mean()


def _mix(noise, signal, snr_db):
    """``signal`` scaled to sit ``snr_db`` above ``noise`` in the passband."""
    g = np.sqrt(_inband_power(noise) / _inband_power(signal) * 10 ** (snr_db / 10))
    return noise + g * signal


def _pretx(path):
    """The receive audio of a connect attempt that precedes our own first burst.

    Our transmissions are in these captures, heard through this rig's own mute,
    and they are not an occupant. The guard reads the window before the first one
    — which is also the only part of the recording it ever saw."""
    return _wav(path)[:int(corpora.SENSE_PRETX_S * FS)]


@pytest.fixture(scope="module")
def clear():
    if not corpora.CLEAR_CHANNEL.exists():
        pytest.skip("verified clear-channel capture not present")
    return _windows(_wav(corpora.CLEAR_CHANNEL))


# --------------------------------------------------------------- liveness first
# No corpus needed: silence is silence.

@pytest.mark.parametrize("name,audio", [
    ("digital silence", np.zeros(WIN)),
    ("codec floor at -75 dBFS",
     np.random.default_rng(0).normal(0, 10 ** (-75 / 20), WIN)),
    ("codec floor at -55 dBFS",
     np.random.default_rng(1).normal(0, 10 ** (-55 / 20), WIN)),
])
def test_a_deaf_receiver_is_never_reported_clear(name, audio):
    """A dead input must fail closed. The old score returned 0.0 dB here, below any
    threshold, so silence was the clearest frequency on the band."""
    assert CB.scores(audio) is None, f"{name} was scored as if it were band audio"
    busy, _levels, margin = CB.is_busy(audio)
    assert busy, f"{name} reported clear"
    assert margin == 0.0, "a refusal must not claim a margin it did not measure"


def test_a_window_with_no_samples_is_a_receiver_that_cannot_be_believed():
    """The capture that arrives from a device which opened and delivered nothing.

    `is_busy` already answers this one — no frames, no verdict, busy. The pair the
    operator's tools read to *name* the fault did not: the mean of an empty window
    is nan, nan compares false against both thresholds, and `receiver_fault`
    cleared a receiver that had produced not one sample.
    """
    empty = np.zeros(0, np.float32)
    assert CB.is_busy(empty)[0], "a capture with no samples read as a clear channel"
    fault = CB.receiver_fault(*CB.level_range(empty), seconds=6.0,
                              device="BlackHole 16ch")
    assert fault is not None and fault.kind == "deaf", (
        f"a device that delivered nothing was reported trustworthy: {fault}")


def test_the_bridge_reads_an_empty_capture_as_a_receiver_fault():
    """`tools/vara_rig_bridge.py` senses the channel through its rig->VARA leg, and
    that leg is empty when `--rig-device` names a device that opens and delivers
    nothing: the wrong index after a USB re-enumeration, or BlackHole with nothing
    routed into it. Short-circuiting to clear there prints `+0.0 dB ... (clear)`,
    which reads as a measurement, and then calls the gateway.
    """
    bridge = corpora.harness("vara_rig_bridge")

    class _Silent:
        """The leg a device that opened and never delivered leaves behind."""
        name, src, recorded = "rig->vara", "USB Audio Device", []

    busy, margin, fault = bridge.channel_busy(_Silent(), 0.0, "2750")
    assert fault is not None and fault.kind == "deaf", (
        "the bridge sensed occupancy through a receiver that delivered no samples")
    assert busy and margin == 0.0


def test_a_receiver_muted_for_most_of_the_window_yields_no_verdict(clear):
    """A transmitting radio mutes its own receive audio, so when another modem
    shares this rig the listen window is mostly our own station's mute. What is
    left is real band audio but too little of it to judge; saying so beats
    guessing from a second of noise."""
    x = clear[0].copy()
    x[int(0.6 * FS):] *= 10 ** (-40 / 20)     # 7.4 of 8 s muted, as a keyed rig does
    assert CB.scores(x) is None
    assert CB.is_busy(x)[0]


def test_a_brief_dropout_does_not_by_itself_make_the_channel_busy(clear):
    """The audio path drops out for 100-400 ms at a time, about 2% of blocks. Those
    are a codec fault, not a signal, and a detector that reads them as structure
    would call every frequency busy."""
    rng = np.random.default_rng(2)
    for w in clear:
        x = w.copy()
        for _ in range(6):
            i = rng.integers(0, len(x) - int(0.4 * FS))
            x[i:i + int(rng.uniform(0.1, 0.4) * FS)] = 0.0
        busy, _levels, margin = CB.is_busy(x)
        assert not busy, f"punched-out dropouts read as occupancy (margin {margin:+.2f} dB)"


# ------------------------------------------------------------ the clear side
@corpora.requires_clear_channel
def test_the_verified_clear_capture_reads_clear(clear):
    """The operator's own 40 m frequency, listened to and judged empty, with the
    station's other modem idle for the whole capture. If this goes busy the
    15-minute window gets spent skipping every target."""
    for i, w in enumerate(clear):
        s = CB.scores(w)
        assert s is not None, f"window {i} was not judged live"
        busy, _levels, margin = CB.is_busy(w)
        assert not busy, (
            f"window {i} reported busy at margin {margin:+.2f} dB "
            f"(burst {s['burst']:.2f}, shape {s['shape']:.2f}, tone {s['tone']:.2f})")


@corpora.requires_channel_sense
def test_the_frequency_that_was_listened_to_and_was_silent_reads_clear():
    """7108.5 kHz on 2026-08-09, and the recording that condemned the old
    calibration: the operator listened to this channel for 45 s and heard nothing,
    and the guard called it occupied anyway at +0.7 dB — as it did every other
    channel of that slot, until ``--force`` was reflex and it could not be heard on
    the two that mattered. A guard that fires always is a guard that is switched
    off."""
    for path in corpora.SENSE_CLEAR:
        s = CB.scores(_pretx(path))
        assert s is not None, f"{path.name} was not judged live"
        busy, _levels, margin = CB.is_busy(_pretx(path))
        assert not busy, (
            f"{path.name} reported busy at margin {margin:+.2f} dB "
            f"(burst {s['burst']:.2f}, shape {s['shape']:.2f}, tone {s['tone']:.2f})")


@corpora.requires_channel_sense
def test_the_two_frequencies_carrying_another_station_read_busy():
    """The other half of that slot, and what the false alarms cost: 7103.5 and
    7102.0 kHz were carrying somebody else's VARA session — a 4.9 s over and
    repeated session-control bursts on the monitor — and we transmitted over both.

    Asserted over the whole of each recording rather than its first window,
    because a caller keys on a *run*: two consecutive clear windows, and a third
    that decides. 7102.0 refuses all eight of its windows. 7103.5 refuses seven —
    the eighth is the eight seconds before the occupant's first over, which holds
    one 0.30 s flat lift and nothing else, and is followed straight away by
    +10.4 dB of `shape`. Neither recording gives anybody two clear windows in a
    row, which is the figure this has to hold at.
    """
    for path in corpora.SENSE_BUSY:
        wins = _windows(_wav(path))
        verdicts = [CB.is_busy(w)[0] for w in wins]
        assert _longest_clear_run(verdicts) < 2, (
            f"{path.name} offers {_longest_clear_run(verdicts)} consecutive clear "
            f"windows of {len(wins)} — enough to key over the session on it")


def _longest_clear_run(verdicts):
    best = run = 0
    for busy in verdicts:
        run = 0 if busy else run + 1
        best = max(best, run)
    return best


def _qrn_80m():
    """The six windows of 80 m band audio, as ``(path, window, judged band)``.

    Two hold one static crash each, in the eight seconds after our own
    transmission stopped. Four are what the gate refused a PACTOR-1 call to 3588
    on across 353 s, before it was forced past and the channel turned out to hold
    nothing. Every clear-side figure the thresholds were set on came off 40 m or
    out of band; this is the same measurement on the band that broke them.
    """
    return [(p, _wav(p)[int(t0 * FS):int(t1 * FS)], band) for p, t0, t1, band in (
        *((p, *corpora.QRN_80M_WINDOW, CB.FULL_BAND) for p in corpora.QRN_80M),
        *((p, 0.0, CB.WINDOW_S, corpora.SENSE_REFUSED_EMPTY_BAND)
          for p in corpora.SENSE_REFUSED_EMPTY))]


def _held_for(w, over_db):
    """Longest unbroken stretch of ``w`` whose in-band level sits ``over_db``
    above the window's own floor, in seconds."""
    _nfft, hop = CB._sizes(FS)
    P, f = CB._frames(np.asarray(w, float), FS)
    lv = 10 * np.log10(P[:, (f >= 400) & (f <= 2700)].mean(axis=1) + 1e-30)
    run = best = 0
    for v in lv:
        run = run + 1 if v >= np.percentile(lv, 10) + over_db else 0
        best = max(best, run)
    return best * hop / FS


@corpora.requires_channel_sense
def test_the_specimen_the_hold_is_calibrated_on_is_a_third_of_a_second():
    """What ``burst`` alone catches on 7103.5 is not an over, whatever the length
    the calibration above it claims.

    ``_BURST_HOLD_S``'s comment used to call this "the 0.9 s over on 7103.5" and
    pick the hold to sit under it. The recording says 0.30 s, and the comment
    says 0.30 s now; holding the two in step is what this measures. It matters
    because a *median* over the hold is exceeded once half its frames are up, so
    a 0.5 s hold does not reject what is shorter than 0.5 s — it rejects what is
    shorter than about 0.25 s, and the distance between those two numbers is the
    whole population of static crashes on a summer 80 m evening.
    """
    over = _pretx(corpora.SENSE_BUSY[0])
    assert 0.2 < _held_for(over, 6.0) < 0.5, (
        "the lift this hold is calibrated against has changed length, so the "
        "figure the hold is chosen against has to be measured again")
    s = CB.scores(over)
    assert s["burst"] >= CB.BURST_DB > max(s["shape"], s["tone"]) - 3, (
        "burst is no longer the only score that catches this window, which is "
        "the premise the whole hold argument rests on")


@corpora.requires_qrn_80m
@corpora.requires_sense_refused_empty
@corpora.requires_channel_sense
def test_burst_cannot_tell_that_lift_from_an_80_m_static_crash(monkeypatch):
    """The overlap that no threshold and no hold length can be moved out of.

    Each of the six windows is band noise: `shape` 2.78-3.99 against its 6.0 and
    `tone` 0.67-1.87 against its 5.0. Every one reads `burst` 7.42-9.98 against
    its 6.6, past the 7.86 of the occupant `burst` alone catches — so the two
    populations are not merely close, they are inverted.

    They are inverted in the length of the lift too, which is what the hold was
    supposed to separate: these hold 0.17 to 0.64 s over the window's own floor
    and the occupant holds 0.30, inside them. Swept from one frame to half the
    window the median filter therefore reaches both together or neither, and the
    occupant never leaves the clear population at any hold. `BURST_DB` is not
    sitting in the wrong place, it is sitting in a gap that does not exist, and
    nothing is bought by moving it or by re-cutting the hold. What separates
    these two is evidence — which is why the gate keeps the window it refused
    on — and not another constant.
    """
    windows = _qrn_80m()
    over = _pretx(corpora.SENSE_BUSY[0])
    for path, w, band in windows:
        s = CB.scores(w, FS, band)
        assert s is not None, f"{path.name} was not judged live"
        assert s["shape"] < CB.SHAPE_DB and s["tone"] < CB.TONE_DB, (
            f"{path.name} has structure in it after all (shape {s['shape']:.2f}, "
            f"tone {s['tone']:.2f}) — it is no longer band noise and cannot "
            f"stand as the clear side of anything")
        assert s["burst"] >= CB.BURST_DB, (
            f"{path.name} reads burst {s['burst']:.2f}, under the {CB.BURST_DB} "
            f"this window is kept for — it is no longer the case burst is making")
    held = [_held_for(w, 6.0) for _, w, _band in windows]
    assert min(held) <= _held_for(over, 6.0) <= max(held), (
        f"the occupant's lift holds {_held_for(over, 6.0):.2f} s against the "
        f"{min(held):.2f}-{max(held):.2f} s of band noise — it has left the "
        f"population, so a hold could tell them apart after all")

    for hold in (0.05, 0.1, 0.5, 1.0, 2.0, 4.0):
        monkeypatch.setattr(CB, "_BURST_HOLD_S", hold)
        noise = [CB.scores(w, FS, band)["burst"] for _, w, band in windows]
        occupant = CB.scores(over)["burst"]
        assert min(noise) <= occupant <= max(noise), (
            f"at a {hold} s hold the occupant burst alone catches reads "
            f"{occupant:.2f}, outside the {min(noise):.2f}-{max(noise):.2f} of 80 m "
            f"band noise — the two populations have come apart and burst can be "
            f"calibrated after all; measure it and say so here")


@corpora.requires_qrn_80m
@corpora.requires_sense_refused_empty
def test_a_static_crash_is_not_a_refusal():
    """None of those six may cost a slot, which is what they were costing.

    Seventeen of seventeen windows the gate kept on 80 m that evening read `burst`
    over its 6.6 — four of them here, on a channel keyed over immediately
    afterwards that held nothing. A refusal costs a slot and a pass keys over
    somebody, so this is the cheap side; it is asserted because it is the side the
    gate was losing every window of.
    """
    for path, w, band in _qrn_80m():
        s = CB.scores(w, FS, band)
        busy, _levels, margin = CB.is_busy(w, FS, band)
        assert not busy, (
            f"{path.name} refused a channel that was band noise, at {margin:+.2f} dB "
            f"(burst {s['burst']:.2f}, shape {s['shape']:.2f}, tone {s['tone']:.2f})")


@corpora.requires_channel_sense
def test_narrowing_the_judged_band_waives_the_7102_carrier():
    """A refusal a narrower emission is entitled to, made visible before it is
    load-bearing.

    7102.0 was carrying another station and the occupant is a narrowband feature
    at 2613 Hz. Judged over the full passband it is refused on `tone`; judged
    over what VARA BW2300 or ARDOP will actually fill, it sits outside the
    emission and the same window reads clear. That is what `core.occupied`
    narrowing the sense is *for*, and it is also the one way this gate can lose a
    channel the corpus labels occupied — so it is asserted rather than
    discovered, and `dominant`'s `tone_hz` is what an operator re-reads it by.
    """
    w = _pretx(corpora.SENSE_BUSY[1])
    wide = CB.scores(w)
    assert CB.dominant(wide)[0] == "tone" and wide["tone"] >= CB.TONE_DB
    assert 2500 < wide["tone_hz"] < 2700, (
        f"the 7102.0 occupant now reads at {wide['tone_hz']:.0f} Hz, so which "
        f"emissions are entitled to walk past it has changed")
    for modem, bw in (("vara", "2300"), ("ardop", "2000")):
        band = occupied_hz(modem, bw)
        assert not CB.is_busy(w, FS, band)[0], (
            f"{modem} {bw} now reads this window busy over {band}, which is a "
            f"stricter answer than the corpus was measured with — welcome, but "
            f"the figures in this module's docstring were taken without it")
        assert band[1] < wide["tone_hz"], (
            f"{modem} {bw} reaches {band[1]:.0f} Hz, past the carrier at "
            f"{wide['tone_hz']:.0f} — this window is inside the emission now and "
            f"waiving it is a station keying over somebody")


@corpora.requires_clear_channel
@corpora.requires_channel_sense
@corpora.requires_qrn_80m
@corpora.requires_monitored_quiet
@corpora.requires_gateway_session
def test_each_threshold_sits_between_the_two_populations_that_pin_it(clear):
    """Neither threshold is free to move. Below the first figure the guard starts
    calling empty frequencies busy; above the second it goes quiet on a frequency
    somebody is using. Both figures are measured here, on the recordings, so the
    calibration travels with the corpus instead of with a docstring.

    The quiet population is every window of verified-empty band audio the corpus
    holds: the verified clear capture, the 2026-08-09 channel that was listened to
    and was silent, the 80 m band noise no threshold was ever measured against, the
    four 80 m refusals that were keyed over and held nothing, and — 67 of the 78,
    and for a long time in the corpus without reaching this test — the ten minutes
    of 7103.5 kHz the feed and every classifier agree were empty. The occupied side
    is one window per score, each picked because that score carries most of the
    weight on it: a turnaround-heavy window of a live gateway session whose only
    mark is spectral shape, and the narrow occupant parked near 2600 Hz on 7102.0.

    ONLY ``tone`` HAS A GAP. ``burst`` never had one — its quiet side runs past
    every occupant it can see, which is why :data:`core.busy.DECIDING` does not
    hold it — and ``shape`` was only believed to have one because its clear side
    had been measured on three windows of a single capture. Over 78 it reaches
    8.47, which is 2.5 dB the wrong side of the threshold, so what is pinned for
    ``shape`` is the false-busy RATE the bias buys the occupied side with. Let that
    rate grow and the guard becomes the one an operator overrides on reflex, which
    is the failure that recalibrated it once already.
    """
    t0, t1 = corpora.QRN_80M_WINDOW
    quiet = (clear + [_pretx(p) for p in corpora.SENSE_CLEAR]
             + [_wav(p)[int(t0 * FS):int(t1 * FS)] for p in corpora.QRN_80M]
             + [_wav(p) for p in corpora.SENSE_REFUSED_EMPTY]
             + _windows(_wav(corpora.MONITORED_QUIET)))
    scored = [CB.scores(w) for w in quiet]
    worst = {k: max(s[k] for s in scored) for k in ("burst", "shape", "tone")}

    session = _windows(_wav(corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav"))
    occupied = {"burst": CB.scores(_pretx(corpora.SENSE_BUSY[0]))["burst"],
                "shape": CB.scores(session[7])["shape"],       # 56-64 s of the session
                "tone": CB.scores(_pretx(corpora.SENSE_BUSY[1]))["tone"]}

    assert worst["tone"] < CB.TONE_DB <= occupied["tone"], (
        f"tone threshold {CB.TONE_DB:.2f} dB is outside the gap it has to sit in: "
        f"quiet band audio reaches {worst['tone']:.2f} dB and the carrier this score "
        f"is what catches reads {occupied['tone']:.2f} dB")
    assert CB.SHAPE_DB <= occupied["shape"], (
        f"shape threshold {CB.SHAPE_DB:.2f} dB is above the {occupied['shape']:.2f} of "
        "the occupant only shape catches — nothing is left holding that window busy")
    refused = sum(s["shape"] >= CB.SHAPE_DB for s in scored)
    assert refused <= 2, (
        f"{refused} of {len(scored)} windows of verified-empty band audio now read "
        f"busy on shape, against the 2 this threshold was kept at {CB.SHAPE_DB:.2f} "
        "dB for — a guard that fires this often on an empty channel gets forced")
    assert worst["burst"] > occupied["burst"], (
        f"80 m band noise now reaches {worst['burst']:.2f} dB of burst against the "
        f"{occupied['burst']:.2f} of the occupant it alone catches — the populations "
        f"have come apart and burst could rejoin DECIDING")


# ------------------------------------------------------------- the busy side
@pytest.mark.parametrize("session", ["NS0A_2300", "KC9GHZ_2300"])
def test_real_gateway_traffic_reads_busy(session):
    """Whole off-air sessions with a live Winlink gateway: ARQ bursts and the
    turnaround gaps between them, through this same rig and codec.

    NS0A refuses 13 of 13 windows and KC9GHZ 12 of 13, the one exception reading
    `shape` 5.89 against its 6.0 between windows of 6.07 and 19.04. What a caller
    needs is two clear in a row, and neither session offers it."""
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    wins = _windows(_wav(path))
    verdicts = [CB.is_busy(w)[0] for w in wins]
    assert _longest_clear_run(verdicts) < 2, (
        f"{_longest_clear_run(verdicts)} consecutive of {len(verdicts)} windows of a "
        "real VARA session reported clear — enough to key over it")


@corpora.requires_clear_channel
@corpora.requires_gateway_session
def test_a_wideband_occupant_that_never_stops_keying_is_still_heard(clear):
    """The defect that most needed a new metric: a signal filling the passband with
    no gaps has almost no temporal contrast, so ``max - min`` measured 0.9 dB on it
    — clear. Real VARA burst audio, tiled to remove the gaps, mixed into the
    verified clear capture."""
    x = _wav(corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav")
    burst = x[int(92.6 * FS):int(93.8 * FS)]
    steady = np.tile(burst, int(np.ceil(WIN / len(burst))))[:WIN]
    for i, w in enumerate(clear):
        busy, _levels, margin = CB.is_busy(_mix(w, steady, 0.0))
        assert busy, f"steady wideband occupant read clear on window {i} ({margin:+.2f} dB)"


@corpora.requires_clear_channel
def test_a_steady_carrier_is_still_heard(clear):
    """The narrow half of the same defect, and the class the very first version of
    this check was calibrated on before a wideband metric replaced it: an
    unmodulated carrier varies in neither level nor spectral shape over time. Only
    a per-frequency test sees it."""
    t = np.arange(WIN) / FS
    carrier = np.sin(2 * np.pi * 1450 * t)
    for i, w in enumerate(clear):
        busy, _levels, margin = CB.is_busy(_mix(w, carrier, -3.0))
        assert busy, f"steady carrier read clear on window {i} ({margin:+.2f} dB)"


@corpora.requires_carrier_only_channel
def test_the_recorded_carrier_only_channel_reads_busy_on_tone_alone():
    """The off-air version of the case above, and the one that gets mis-filed.

    6800 kHz on 2026-08-14 was recorded as that slot's negative control and its 21
    BUSY status lines were written up as firing on an empty channel, on the evidence
    that no classifier named anything in 211.6 s. Two steady carriers are in the
    passband — see ``corpora.CARRIER_ONLY_CHANNEL`` for the spectrum and for the
    check that they move with the dial — so the verdict is right, and nothing here
    is free to move to make that complaint go away.

    What this recording pins is *which* score is carrying it. Over 8 s windows
    `tone` reads 7.64 to 9.57 against its 5.0, in all 26 of them, while `burst`
    stays at 0.49-0.99 and `shape` at 3.03-4.61 — both well inside their own
    thresholds, because a carrier varies in neither level nor spectral shape. Raise
    TONE_DB to quiet this channel and the only score that can see a carrier at all
    goes with it, which is the trade the very first version of this check made.
    """
    for i, w in enumerate(_windows(_wav(corpora.CARRIER_ONLY_CHANNEL))):
        s = CB.scores(w)
        assert s is not None, f"window {i} was not judged live"
        busy, _levels, margin = CB.is_busy(w)
        assert busy, (
            f"window {i} read clear at {margin:+.2f} dB on a channel carrying a "
            f"+30 dB carrier (burst {s['burst']:.2f}, shape {s['shape']:.2f}, "
            f"tone {s['tone']:.2f})")
        assert s["tone"] > CB.TONE_DB, (
            f"window {i}: tone {s['tone']:.2f} dB is no longer what finds this, so "
            "the busy verdict here is riding on a score that cannot see a carrier")
        assert s["burst"] < CB.BURST_DB and s["shape"] < CB.SHAPE_DB, (
            f"window {i}: burst {s['burst']:.2f}, shape {s['shape']:.2f} — the wide "
            "scores have started seeing a bare carrier, so this capture has stopped "
            "being the narrow-only case it is kept for")


@corpora.requires_clear_channel
def test_keyed_cw_is_still_heard(clear):
    """A CW operator calling CQ. Weaker than the carrier case because the key is up
    half the time, which is also why it must be checked separately."""
    t = np.arange(WIN) / FS
    cw = np.sin(2 * np.pi * 800 * t) * ((np.sin(2 * np.pi * 0.8 * t) > 0) * 0.99 + 0.01)
    for i, w in enumerate(clear):
        busy, _levels, margin = CB.is_busy(_mix(w, cw, -6.0))
        assert busy, f"keyed CW read clear on window {i} ({margin:+.2f} dB)"


# ------------------------------------------------------------------- reporting
def test_occupancy_db_reports_dbfs():
    """The middle return value is an absolute level now, not a ratio against the
    spectrum past the filter skirt — that reference was codec noise, and dividing
    by it put its second-to-second jitter straight into the verdict."""
    t = np.arange(3 * FS) / FS
    for amp in (1.0, 0.1, 0.01):
        got = CB.occupancy_db(amp * np.sin(2 * np.pi * 1000 * t))
        assert got == pytest.approx(20 * np.log10(amp / np.sqrt(2)), abs=0.05)


@corpora.requires_clear_channel
def test_the_margin_is_signed_and_agrees_with_the_verdict(clear):
    """Callers print the margin and act on the boolean; they must never disagree."""
    path = corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip("off-air recording not present")
    for w in clear + _windows(_wav(path))[:4]:
        busy, _levels, margin = CB.is_busy(w)
        assert busy == (margin >= 0.0)


# ------------------------------------------------------------- the sample rate
def test_the_analysis_frame_is_a_duration_not_a_sample_count():
    """4096 samples is 85 ms only at 48 kHz. A KiwiSDR delivers 11999 Hz, where
    the same figure buys a 341 ms frame that averages straight across the keying
    gaps ``burst`` exists to measure — and every occupancy number
    ``tools/gwsurvey.py`` has ever printed was read that way."""
    assert CB._sizes(FS) == (4096, 2048), "the calibrated rate must not move"
    for fs in (48000, 12000, 11999, 8000):
        nfft, hop = CB._sizes(fs)
        assert nfft == 2 * hop
        assert nfft / fs == pytest.approx(4096 / 48000, rel=0.01)


def test_the_scores_do_not_depend_on_the_sample_rate():
    """One signal, two rates, the same three numbers. Keyed 250 ms on and 250 ms
    off — an ARQ cadence, and short enough that a frame four times too long
    smears the gaps away: before the frame became a duration this read 4.67 dB of
    burst contrast at 12 kHz against 5.41 dB at 48 kHz on the same audio."""
    from scipy.signal import firwin, lfilter, resample_poly

    rng = np.random.default_rng(3)
    n = 10 * FS
    taps = firwin(255, [400 / (FS / 2), 2700 / (FS / 2)], pass_zero=False)
    noise = lfilter(taps, 1, rng.normal(0, 1, n))
    sig = lfilter(taps, 1, rng.normal(0, 1, n))
    keyed = (np.arange(n) // int(0.25 * FS)) % 2 == 0
    g = np.sqrt(10 ** 0.3 * (noise @ noise) / (sig @ sig))
    x48 = 0.05 * (noise + g * sig * keyed)

    at48 = CB.scores(x48, FS)
    at12 = CB.scores(resample_poly(x48, 1, 4), FS // 4)
    for k in ("burst", "shape", "tone"):
        assert at12[k] == pytest.approx(at48[k], abs=0.1), (
            f"{k} reads {at12[k]:.2f} dB at 12 kHz against {at48[k]:.2f} at 48 kHz")
