# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the rig's filter has to pass, and what a filter that will not is called.

Three paths set the mode and none of them read the width back, so on 2026-08-15
the front panel read 1700 Hz while one caller believed it had asked for 3000 and
another for 2700. 1700 is not an arbitrary number: it is what an FT-891 selects
for PKTUSB when hamlib is handed `M PKTUSB 0`, and 0 is what those callers sent.

The refusal is about the emission rather than the mode name. Centred at 1500 Hz,
a filter has to reach the further edge on both sides, so the width it needs is
twice the greater distance from centre — 2000 Hz for ARDOP 2000 (500-2500), 250
for ARDOP 200 (1375-1625). At 1700 Hz the first is clipped 300 Hz deep on each
side and the second is not touched, which is why "the rig is in a data mode" was
never enough to key on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hfmodem.besra.arq import session as besra_session
from hfmodem.besra.dsp import templates
from hfmodem.besra.frame import frame as F
from hfmodem.besra.phy.modulator import carrier_indices
from hfmodem.core.busy import FULL_BAND, MIN_SHAPE_HZ, _widen
from hfmodem.core.occupied import (
    FILTER_HZ, OCCUPIED_HZ, clipped, keyed_hz, occupied_hz,
)
from hfmodem.shrike import onair, session, spec

REPO = Path(__file__).resolve().parents[5]
LAUNCHER = REPO / "tools" / "onair.sh"


def test_a_1700_hz_filter_clips_ardop_2000_and_says_by_how_much():
    why = clipped(1700, occupied_hz("ardop", "2000"))
    assert why, "the filter the rig was actually found in passed as wide enough"
    assert "1700" in why and "2000" in why       # what it is, what it needs
    assert "650-2350" in why, why                # and what that leaves of us


def test_the_same_1700_hz_filter_clips_a_pactor_session_too():
    """It passes the PACTOR-1 connect and not the session that follows, which is
    the whole reason the table answers for the session: 2240 Hz needed against
    1700 found, so the SL6 tones at 480 and 2520 Hz leave the radio cut."""
    assert clipped(1700, session.occupied_hz(sl=1, data=False)) == ""
    why = clipped(1700, occupied_hz("pactor"))
    assert why and "2240" in why, why


def test_a_1700_hz_filter_passes_ardop_200_whole():
    """A refusal that fires on every emission is one an operator learns past.
    ARDOP 200 is 4FSK across 1425-1575 Hz at 50 Bd — it needs 250 Hz of filter,
    and the one the rig was found in has 1450 to spare."""
    assert clipped(1700, occupied_hz("ardop", "200")) == ""


def test_the_width_every_path_asks_for_covers_every_emission_listed():
    """FILTER_HZ is asked for rather than the emission's own width, because the
    `core.busy` thresholds were calibrated on audio taken through a wide filter
    and no capture exists at any other. Nothing we transmit may then be clipped
    by the width we chose."""
    for key in OCCUPIED_HZ:
        assert clipped(FILTER_HZ, occupied_hz(*key)) == "", key


def test_the_band_judged_for_occupancy_reaches_the_width_the_rig_is_set_to():
    """`busy._widen` clips every declared emission to `FULL_BAND`, so a top edge
    under the filter we ask for leaves air inside the receiver that nothing
    scores. The gear this bites is sabir, whose 140-2860 Hz row is the widest on
    the station: 265 Hz of what it is about to key sat outside the band it senses.

    The bottom edge is a separate question and is not asserted here — 400 still
    clips the pactor row's 380, and no measurement of the receiver's low corner
    has been taken.
    """
    assert FULL_BAND[1] == FILTER_HZ
    for key, band in OCCUPIED_HZ.items():
        assert _widen(band, MIN_SHAPE_HZ)[1] >= band[1], key


def test_the_pactor_entry_covers_every_burst_a_session_can_key():
    """The table is quoted and `shrike.session.occupied_hz` is computed from the
    waveform, so the two can drift — and did. Held at the connect's 1200-1800 Hz
    it was 2240 Hz narrower than an SL6 data burst, which is a sense that reads
    clear across an occupant at 500 Hz and a filter check that passes 1700 Hz for
    an emission needing 2240.

    A launcher cannot name the speed level a link will reach: it senses before
    there is a link, and the ARQ climbs to `arq.ArqConfig.max_sl` from inside it.
    So the entry is the widest burst and the drift is closed here rather than by
    teaching `core` a protocol's name — `tests/gates/test_import_direction.py`
    forbids that, and this test is the seam it leaves open.
    """
    bursts = [session.occupied_hz(sl=sl, data=data)
              for sl in spec.SPEED_LEVELS for data in (False, True)]
    assert occupied_hz("pactor") == (min(lo for lo, _ in bursts),
                                     max(hi for _, hi in bursts))


def test_a_link_held_in_pactor1_is_judged_on_the_connect_alone():
    """What a call CAN key is the ceiling it is run under, and a link that never
    offers the upgrade never leaves the connect's 1400/1600 Hz. Computed from the
    waveform beside the row, the same way the SL6 answer above is."""
    assert occupied_hz("pactor", "1") == session.occupied_hz(sl=1, data=False)
    assert occupied_hz("pactor", "1") != occupied_hz("pactor")


def test_the_flag_that_narrows_the_sense_really_holds_the_link():
    """The hold exercised rather than described. `core` may not know a protocol's
    name, so what keeps the narrow row honest is this: the ARQ offers the upgrade
    on the first acknowledged packet, and under the flag it is refused every
    time. Without that the sense would measure 1200-1800 Hz and the session would
    go on to fill 2240."""
    from hfmodem.shrike.ptc import Protocol, PtcHost

    held = PtcHost(mycall="W9SSJ")
    held.stay_in_pactor1 = True
    assert not held.upgrade(payload_waiting=True) \
        and held.protocol is Protocol.PACTOR1
    assert PtcHost(mycall="W9SSJ").upgrade(payload_waiting=True), (
        "an unheld link no longer upgrades, so the wide row is measuring a "
        "session that cannot happen")


def test_the_narrow_pactor_sense_is_only_reachable_through_the_flag_that_holds_it():
    """A declaration the modem does not honour is a sense measuring a band we
    are not going to stay in. `--pactor1-only` is what holds the link there, so
    the launcher may reach the narrow row only by reading that same flag, and it
    passes it on rather than consuming it — the ARQ never hears about `--force`
    and must hear about this one.

    Asserted as text, like every other keying rule in the launchers, because
    neither the verb nor shrike's parser can be run without a radio."""
    if not LAUNCHER.exists():
        pytest.skip(f"{LAUNCHER} is not present (installed-wheel run)")
    verb = LAUNCHER.read_text().split("\npactor)", 1)[1].split("\n  ;;", 1)[0]
    narrow, = [ln for ln in verb.splitlines() if "gate_channel" in ln]
    assert "--pactor1-only" in verb, (
        "the pactor verb no longer reads the flag that holds the link in "
        "PACTOR-1, so the band it senses is not the band it will key")
    assert '"$pbw"' in narrow, narrow
    assert '"${@:4}"' in narrow, (
        f"the flag is consumed instead of reaching shrike: {narrow}")
    assert "stay_in_pactor1 = args.pactor1_only" in \
        Path(onair.__file__).read_text(), (
        "shrike parses the flag and does not act on it: the sense would narrow "
        "to the connect while the ARQ went on offering the upgrade")


@pytest.mark.parametrize("bw", sorted(besra_session._DATA_LADDER))
def test_the_ardop_entries_cover_every_frame_besra_keys(bw):
    """Same drift, caught 25 Hz short: the 200 Hz entry said 1400-1600, the
    single-carrier PSK band, while every frame besra actually sends at that width
    is 4FSK across 1425-1575 at 50 Bd — 1375-1625 with its sidebands.

    What besra keys is every 50 Bd single-carrier frame — the whole control set —
    plus every rung of the width's transmit ladder (`arq.session._DATA_LADDER`),
    which climbs into 4PSK at every width and spreads that across 2, 4 and 8
    carriers at the wider ones. So the emission is read off the tones for FSK and
    off the carriers the mode actually rides for PSK, and either way the sidebands
    run one baud out.
    """
    ladder = {t for base, _ in besra_session._DATA_LADDER[bw] for t in (base, base + 1)}
    keyed = [f for f in F.FRAMES.values()
             if f.type in ladder or (f.carriers == 1 and f.baud == 50)]
    assert keyed, bw
    lo, hi = occupied_hz("ardop", str(bw))
    for f in keyed:
        tones = (templates.FSK_TONES_HZ[f.baud] if f.mod is F.Mod.FSK4
                 else tuple(templates.PSK_CARRIERS_HZ[c]
                            for c in carrier_indices(f.carriers)))
        assert lo <= min(tones) - f.baud and max(tones) + f.baud <= hi, (
            f"{f.name} fills {min(tones) - f.baud}-{max(tones) + f.baud} Hz, "
            f"outside the {lo:.0f}-{hi:.0f} Hz ARDOP {bw} is sensed and filtered at")


def test_a_rig_that_will_not_say_is_not_thereby_wide_enough():
    """`state()` answers None when the passband field is missing or unreadable,
    and an unknown width is the one case a comparison cannot decide."""
    assert clipped(None, occupied_hz("vara", "2300"))


def test_an_unlisted_mode_is_answered_wide():
    assert occupied_hz("nobody", "9999") == FULL_BAND
    assert occupied_hz("VARA", "2300") == occupied_hz("vara", "2300")


def _minus_26db_span(x, fs=48000):
    """Where a burst's spectrum last stands within 26 dB of its own peak.

    §97.3(a)(8) writes bandwidth in 26 dB of attenuation, so this is the measure
    the table has to survive. Rectangular and zero-padded rather than windowed:
    the DTFT of the finite burst is the burst's real spectrum, and a window would
    hide skirts the transmitter still emits.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = 1 << (int(np.ceil(np.log2(len(x)))) + 2)
    mag = np.abs(np.fft.rfft(x, n))
    inband = np.flatnonzero(20 * np.log10(mag / mag.max() + 1e-18) >= -26.0)
    f = np.fft.rfftfreq(n, 1 / fs)
    return f[inband[0]], f[inband[-1]]


def test_the_vara_rows_hold_against_vara_s_own_modulators():
    """Measured, because deriving these from the carrier extremes got them wrong.

    Three numbers are all true of BW2300 and only one is the gate's business: the
    carrier alphabet is bins 29-98 (679.7-2296.9 Hz), 99 % of the energy sits in
    bins 34-98 (773-2320 Hz), and the -26 dB span is 612.4-2401.3. The middle one
    spent time in this tree described as the first, and the row built on it
    declared a bottom edge 68 Hz inside the emission.
    """
    from hfmodem.kestrel.tx import varahf500_tx as t5, varahf2300_tx as t2
    from hfmodem.kestrel.vara import vara_frames as vf, vara_mfsk as mfsk

    from hfmodem.kestrel.rx import varahf2300 as r2
    wide = r2.BASE_LEVELS["2750"]
    for bw, alphabet, over in (("2300", vf.BW2300_TONES, t2.synth_burst(bytes(90))),
                               ("2750", vf.BW2750_TONES,
                                t2.synth_burst(bytes(90), wide)),
                               ("500", vf.BW500_TONES, t5.synth_burst(bytes(46)))):
        lo, hi = occupied_hz("vara", bw)
        # The handshake sweeps its whole alphabet one tone per symbol; the DATA
        # over is the other population, and a session emits both.
        for what, burst in (("handshake alphabet",
                             mfsk.synth_tones(sorted(alphabet.carriers) * 2)),
                            ("DATA over", over)):
            got = _minus_26db_span(burst)
            assert lo <= got[0] and got[1] <= hi, (
                f"VARA BW{bw} {what} measures {got[0]:.1f}-{got[1]:.1f} Hz at "
                f"-26 dB, outside the {lo:.0f}-{hi:.0f} Hz declared for it")


def test_the_sabir_row_holds_against_every_gear_including_the_ones_that_understate():
    """`Gear.passband_hz` is the ladder's own declaration and seven of the eight
    gears emit outside theirs — `wide256` measures 209 Hz against a declared 240
    and 2962 against a declared 2925. So the row is measured off the gears rather
    than taken from what they say about themselves.

    Swept, because one burst per gear is not a bandwidth: that is how the row
    came to be 220-2925, which `fast` crosses at both edges on draws the row was
    never measured over.
    """
    from hfmodem.sabir.phy import GEARS, Phy

    lo, hi = occupied_hz("sabir")
    for name, gear in GEARS.items():
        phy = Phy(gear)
        for seed in range(8):
            rng = np.random.default_rng(seed)
            audio = np.asarray(phy.to_audio(
                phy.transmit(rng.integers(0, 2, phy.capacity(8)).astype(np.uint8))))
            got = _minus_26db_span(audio.real if np.iscomplexobj(audio) else audio)
            assert lo <= got[0] and got[1] <= hi, (
                f"sabir gear {name} on draw {seed} measures {got[0]:.1f}-{got[1]:.1f} "
                f"Hz at -26 dB, outside the {lo:.0f}-{hi:.0f} Hz declared for it")


def test_the_gate_refuses_a_waveform_it_has_no_passband_for():
    """`occupied_hz` may guess and `keyed_hz` may not. FULL_BAND reaches the
    filter at the top, so what a gate reaching for it would still get wrong is the
    bottom: sabir 260 Hz narrow there, pactor 20."""
    assert FULL_BAND[0] - OCCUPIED_HZ[("sabir", "")][0] == 260.0
    assert OCCUPIED_HZ[("sabir", "")][1] <= FULL_BAND[1]
    assert FULL_BAND[0] - OCCUPIED_HZ[("pactor", "")][0] == 20.0
    with pytest.raises(ValueError, match="no passband on file"):
        keyed_hz("nobody")
    with pytest.raises(ValueError, match="no passband on file"):
        keyed_hz("vara", "9999")


def test_an_unpinned_call_is_judged_on_the_widest_burst_it_could_reach():
    """The pactor rows' reasoning, applied to every modem: a launcher that has
    not pinned a bandwidth may reach any of them, so the gate is owed the union.
    `occupied_hz` keeps answering FULL_BAND there because a sense that guesses
    wrong waits, and a gate that guesses wrong transmits outside the segment."""
    assert keyed_hz("vara") == OCCUPIED_HZ[("vara", "2750")]
    assert keyed_hz("ardop") == OCCUPIED_HZ[("ardop", "2000")]
    assert keyed_hz("pactor") == OCCUPIED_HZ[("pactor", "")]
    assert occupied_hz("vara") == FULL_BAND
