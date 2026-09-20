# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Regulatory profiles: the mechanism, and the two profiles shipped.

Several of these exist because an independent review of an earlier version found
the gate permitting emissions it should have refused. Each such case is marked.
"""
from __future__ import annotations

import pytest

from hfmodem.core import band
from hfmodem.core.regulatory import (
    Control, Emission, NotPermitted, Profile, Unregulated, centred, profile,
)
from hfmodem.core.regulatory.part97 import Licence, Part97

GENERAL = Part97(licence=Licence.GENERAL)
EXTRA = Part97(licence=Licence.EXTRA)


def em(centre_khz: float, bw: float, **kw) -> Emission:
    """A symmetric emission at a published channel centre."""
    return centred(band.dial_hz(centre_khz * 1000), bw, **kw)


# --- the mechanism ------------------------------------------------------------

def test_there_is_no_default_profile():
    with pytest.raises(ValueError, match="unknown regulatory profile"):
        profile("")
    with pytest.raises(ValueError, match="unknown regulatory profile"):
        profile("fcc")


def test_both_shipped_profiles_satisfy_the_interface():
    assert isinstance(profile("unregulated", because="bench"), Profile)
    assert isinstance(profile("part97", licence="general"), Profile)


def test_a_profile_reports_permission_by_raising_not_returning():
    """A check that returns a boolean gets used in an ``if`` and then ignored."""
    assert GENERAL.check(em(14105, 500), control=Control.LOCAL) is None


def test_the_span_comes_from_the_real_passband_not_an_assumed_centre():
    """An asymmetric emission must retain both stated edges."""
    dial = 14_100_000
    lo, hi = 205.0, 2965.0
    asymmetric = Emission(dial, lo, hi, drift_hz=0)
    symmetric = centred(dial, hi - lo, drift_hz=0)
    assert asymmetric.span != symmetric.span
    assert asymmetric.span[1] - symmetric.span[1] == pytest.approx(85.0)


def test_drift_is_charged_against_the_emission_on_both_sides():
    """A few ppm on a portable HF radio is 40–90 Hz at 14 MHz."""
    e = centred(14_100_000, 2300, drift_hz=100)
    lo, hi = e.span
    assert lo == 14_100_000 + 350 - 100
    assert hi == 14_100_000 + 2650 + 100


# --- the unregulated profile --------------------------------------------------

def test_the_unregulated_profile_costs_a_stated_reason():
    """It is the override flag. A station running unchecked should not look, in
    its config and its logs, like one that made a considered choice."""
    with pytest.raises(TypeError):
        Unregulated()
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="stated reason"):
            Unregulated(because=blank)


def test_the_unregulated_profile_permits_everything_and_records_why():
    p = Unregulated(because="dummy load, no antenna connected")
    assert p.name == "unregulated"
    assert "dummy load" in p.because
    for control in Control:
        assert p.check(em(14080, 2750), control=control) is None
    assert p.check(Emission(5_000_000, 0, 60_000), control=Control.AUTOMATIC) is None


def test_it_is_not_called_open():
    """In a project built on open protocols that word reads as a virtue and
    would be doing unearned work for what is in effect the override flag."""
    with pytest.raises(ValueError, match="unknown regulatory profile"):
        profile("open", because="bench")


# --- part97: bandwidth --------------------------------------------------------

def test_hf_data_is_capped_at_2800_hz():
    """Review finding: the gate had no upper bandwidth limit and permitted a
    20 kHz HF data emission. §97.307(f)(3)."""
    EXTRA.check(em(14105, 2750), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match=r"97\.307"):
        EXTRA.check(em(14105, 20_000), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match=r"97\.307"):
        EXTRA.check(em(28155, 60_000), control=Control.AUTOMATIC)


def test_2800_exactly_is_permitted_and_2801_is_not():
    EXTRA.check(centred(14_100_000, 2800, drift_hz=0), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match=r"97\.307"):
        EXTRA.check(centred(14_100_000, 2801, drift_hz=0), control=Control.LOCAL)


# --- part97: §97.309 is the operator's, not the gate's ------------------------

@pytest.mark.parametrize("technique", ["pactor", "ardop", "sabir", "vara", "something-new"])
def test_no_technique_is_refused_on_documentation_grounds(technique):
    """An earlier version refused any technique it could not place among those
    "documented publicly" under §97.309(a)(4), which would have refused VARA —
    a mode that has carried Winlink traffic daily for over a decade without
    enforcement action.

    Whether a technique's characteristics are documented publicly is a judgement
    about the state of the world, not a computation, and it is the one question
    in this profile that is not arithmetic. Software imposing a contested reading
    nobody else applies protects no one.

    There is a resolution rather than a standoff, and it is part of why this
    project exists: publishing a normative description of a waveform *makes* it
    documented publicly."""
    GENERAL.check(em(14105, 500, technique=technique), control=Control.LOCAL)


# --- part97: privileges -------------------------------------------------------

def test_a_technician_has_no_hf_data_below_ten_metres():
    """§97.307(f)(9), not §97.301 — Technicians do hold 80/40/15 m."""
    tech = Part97(licence=Licence.TECHNICIAN)
    with pytest.raises(NotPermitted, match=r"97\.301"):
        tech.check(em(14105, 500), control=Control.LOCAL)
    tech.check(em(28125, 500, power_w=100), control=Control.LOCAL)


def test_a_general_is_refused_the_extra_only_segment():
    with pytest.raises(NotPermitted, match=r"97\.301"):
        GENERAL.check(em(14010, 500), control=Control.LOCAL)
    EXTRA.check(em(14010, 500), control=Control.LOCAL)


def test_one_sixty_metres_is_available():
    """Review finding: 160 m was missing and a legal emission was refused."""
    GENERAL.check(em(1840, 500), control=Control.LOCAL)


@pytest.mark.parametrize("licence", list(Licence))
def test_outside_the_amateur_bands_entirely_is_refused_for_everyone(licence):
    with pytest.raises(NotPermitted, match=r"97\.301"):
        Part97(licence=licence).check(em(13000, 500), control=Control.LOCAL)


# --- part97: 60 m, as rewritten effective 13 February 2026 --------------------

def test_the_sixty_metre_band_is_an_ordinary_data_segment():
    """§97.303(h)(3), rewritten by FCC 25-60 (91 FR 1405): amateur stations may
    transmit "in the 5351.5-5366.5 kHz band" as well as on four channels. A
    2.8 kHz data emission there needs no channelisation at all."""
    GENERAL.check(em(5359, 2800, power_w=9.15), control=Control.LOCAL)
    EXTRA.check(em(5359, 2800, power_w=9.15), control=Control.LOCAL)


def test_the_retired_fifth_channel_is_now_inside_the_band():
    """5358.5 kHz was one of the five discrete channels; the new allocation
    swallowed it."""
    GENERAL.check(em(5358.5, 2800, power_w=5), control=Control.LOCAL)


def test_the_band_edges_hold():
    GENERAL.check(em(5353, 2800, power_w=5), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match=r"97\.301"):
        GENERAL.check(em(5352.9, 2800, power_w=5), control=Control.LOCAL)
    GENERAL.check(em(5365, 2800, power_w=5), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match=r"97\.301"):
        GENERAL.check(em(5365.1, 2800, power_w=5), control=Control.LOCAL)


@pytest.mark.parametrize("centre", [5332.0, 5348.0, 5373.0, 5405.0])
def test_the_four_discrete_channels_are_still_not_offered(centre):
    """Data on a channel is 2K80J2D with the carrier set 1.5 kHz below the
    centre — an exact placement, not a range, which a segment table cannot
    express. Four now, not five."""
    with pytest.raises(NotPermitted, match=r"97\.301"):
        GENERAL.check(em(centre, 2800, power_w=100), control=Control.LOCAL)


def test_the_sixty_metre_band_is_capped_at_nine_point_one_five_watts():
    """§97.313(i) — 9.15 W ERP in the band against 100 W ERP on the channels.
    ERP is PEP times gain over a dipole, and a dipole is presumed 0 dBd, so this
    is the limit for the antenna the rule presumes; a gain antenna must come
    down further, and the profile cannot know the gain.

    The refusal must say all of that itself: this test once pinned "9.15 W PEP",
    and pinning the wrong unit is exactly how it survived."""
    with pytest.raises(NotPermitted, match=r"9\.15 W ERP.*presumed 0 dBd dipole"):
        GENERAL.check(em(5359, 2800, power_w=100), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match="does not state its power"):
        GENERAL.check(em(5359, 2800), control=Control.LOCAL)


def test_automatic_control_is_still_out_of_the_whole_of_sixty_metres():
    """§97.221(c) excepts "channels specified in §97.303(h)", which since
    13 February 2026 specifies a band as well as four channels. Whether the
    exception reaches the band is unsettled on the text, so it stays excluded —
    including the narrow-response route, which is the one that would otherwise
    have opened."""
    with pytest.raises(NotPermitted, match=r"97\.221"):
        GENERAL.check(em(5359, 2800, power_w=5), control=Control.AUTOMATIC)
    with pytest.raises(NotPermitted, match=r"97\.221"):
        GENERAL.check(em(5359, 500, power_w=5), control=Control.AUTOMATIC,
                      responding=True)


def test_a_technician_has_no_sixty_metres():
    """Not §97.307(f)(9) this time — §97.301(a) simply does not grant it."""
    with pytest.raises(NotPermitted, match=r"97\.301"):
        Part97(licence=Licence.TECHNICIAN).check(
            em(5359, 2800, power_w=5), control=Control.LOCAL)


# --- part97: power ------------------------------------------------------------

def test_thirty_metres_is_capped_at_200_watts():
    """§97.313(c)(1), and 30 m carries most of this station's traffic."""
    GENERAL.check(em(10145, 500, power_w=100), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match=r"97\.313"):
        GENERAL.check(em(10145, 500, power_w=500), control=Control.LOCAL)


def test_a_power_limited_band_refuses_an_emission_that_will_not_say():
    with pytest.raises(NotPermitted, match="does not state its power"):
        GENERAL.check(em(10145, 500), control=Control.LOCAL)


def test_a_technician_is_capped_on_the_bands_the_rule_names():
    tech = Part97(licence=Licence.TECHNICIAN)
    tech.check(em(28125, 500, power_w=200), control=Control.LOCAL)
    with pytest.raises(NotPermitted, match=r"97\.313"):
        tech.check(em(28125, 500, power_w=1000), control=Control.LOCAL)


# --- part97: automatic control ------------------------------------------------

def test_local_control_is_not_confined_to_the_automatic_sub_bands():
    GENERAL.check(em(14080, 2300), control=Control.LOCAL)


def test_automatic_control_is_confined():
    with pytest.raises(NotPermitted, match=r"97\.221"):
        GENERAL.check(em(14080, 2300), control=Control.AUTOMATIC)


def test_the_narrow_sub_bands_are_what_actually_bite():
    """7.100–7.105 and 18.105–18.110 are 5 kHz wide."""
    GENERAL.check(em(7104, 500), control=Control.AUTOMATIC)
    with pytest.raises(NotPermitted, match=r"97\.221"):
        GENERAL.check(em(7104, 2300), control=Control.AUTOMATIC)
    for bw in (500, 2300, 2750):
        GENERAL.check(em(14110, bw), control=Control.AUTOMATIC)


def test_an_emission_may_not_straddle_the_gap_at_14_100():
    """§97.221(b) lists 14.0950–14.0995 and 14.1005–14.112 with a deliberate
    1 kHz gap. Merging them would permit an emission across it."""
    with pytest.raises(NotPermitted, match=r"97\.221"):
        GENERAL.check(em(14100, 2300), control=Control.AUTOMATIC)


def test_responding_narrow_is_allowed_outside_the_sub_bands():
    GENERAL.check(em(14080, 500), control=Control.AUTOMATIC, responding=True)
    with pytest.raises(NotPermitted, match="exceeds"):
        GENERAL.check(em(14080, 2300), control=Control.AUTOMATIC, responding=True)


def test_the_profile_name_is_not_caller_settable():
    """It is what a log says the rules were. It should not be an argument."""
    with pytest.raises(TypeError):
        Part97(licence=Licence.GENERAL, name="part97-audited")


def test_a_capped_band_is_reachable_once_the_station_states_its_power():
    """30 m is 200 W PEP for everyone, and the gate rightly refuses an emission
    that does not say what it runs — the alternative is an unknown amount into a
    limited band.

    But there was no way to say it. No configuration key, nothing passing one, so
    every protocol was refused on 30 m and no setting could fix it: a correct rule
    with no way to satisfy it, which reads to an operator as a broken radio.
    """
    import tomllib
    from pathlib import Path
    from hfmodem.core import config
    from hfmodem.station.link import LINKS
    root = Path(__file__).resolve().parents[5]
    raw = tomllib.loads((root / "examples" / "station.toml").read_text())
    raw["rig"]["power_w"] = 100.0
    cfg = config.parse(raw)
    assert cfg.rig.power_w == 100.0, "the key exists but did not parse"

    p = profile("part97", licence="general")
    dial = 10_145_500 - 1500                       # a 30 m gateway, centre minus offset
    for name, cls in LINKS.items():
        stated = cls.waveform.emission(dial, power_w=cfg.rig.power_w)
        p.check(stated, control=Control.AUTOMATIC, responding=False)

        silent = cls.waveform.emission(dial)
        with pytest.raises(NotPermitted, match="97.313"):
            p.check(silent, control=Control.AUTOMATIC, responding=False)

        with pytest.raises(NotPermitted, match="97.313"):
            p.check(cls.waveform.emission(dial, power_w=500.0),
                    control=Control.AUTOMATIC, responding=False)
