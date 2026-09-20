# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The arbiter's policy, which is five rules and no knobs.

Each one corresponds to something that has gone wrong on this station or to a
mistake a priority scheme makes by default. None of it needs a radio.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.core import config
from hfmodem.core.audio import ReplayAudio
from hfmodem.core.regulatory import Control, Unregulated, centred
from hfmodem.core.rig import Cat, Rig
from hfmodem.station.arbiter import LIVE, Identity, Refused, TxArbiter, TxRequest
from hfmodem.station.process import Station
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld

BENCH = Unregulated(because="unit test, no radio")


@pytest.fixture
def rig():
    fake = FakeRigctld()
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=FakePtt(),
            profile=BENCH, control=Control.LOCAL, mycall="N0CALL", transmit=True)
    r.cat.open()
    r._armed = True
    r._dial_hz = fake.freq
    yield r
    r.close()
    fake.close()


@pytest.fixture
def arb(rig):
    audio = ReplayAudio(np.zeros(48000 * 4, np.int16))
    return TxArbiter(audio, rig, identity=Identity("N0CALL", interval_s=600.0))


def req(**kw) -> TxRequest:
    kw.setdefault("priority", LIVE)
    kw.setdefault("seq", 1)
    kw.setdefault("protocol", "vara")
    kw.setdefault("audio", np.zeros(4800, np.float32))
    kw.setdefault("emission", centred(7_100_000, 500.0))
    return TxRequest(**kw)


def test_a_burst_goes_out_and_the_lanes_are_advanced_past_it(arb):
    end = arb.submit(req(at=1000))
    assert end == 1000 + 4800
    assert arb.sent == 1


def test_a_late_burst_is_dropped_rather_than_sent(arb):
    """A cycle that overran is absorbed, instead of displacing every cycle after
    it — shrike's take_until discipline applied to transmit."""
    arb.audio.samples = 50_000
    with pytest.raises(Refused, match="has passed"):
        arb.submit(req(at=1000))
    assert arb.dropped_late == 1 and arb.sent == 0


def test_over_long_audio_is_refused_before_keying_not_truncated(arb):
    """A clipped transmission is the worse artefact."""
    long = np.zeros(48000 * 60, np.float32)
    with pytest.raises(Refused, match="Refused rather than truncated"):
        arb.submit(req(audio=long, at=1000))
    assert arb.sent == 0


def test_an_undescribed_emission_is_refused(arb):
    """The rig will not key without a checked emission, so the arbiter will not
    ask it to."""
    with pytest.raises(Refused, match="no emission described"):
        arb.submit(req(emission=None, at=1000))


def test_identification_blocks_traffic_rather_than_queueing_behind_it(arb):
    """A priority scheme starves exactly the station that is busiest, which is the
    one most likely to owe a callsign. So this is an interlock, not a request."""
    arb.identity._last -= 10_000          # overdue
    assert arb.identity.due
    with pytest.raises(Refused, match="identification is due"):
        arb.submit(req(at=1000))
    arb.identify(centred(7_100_000, 500.0))
    assert not arb.identity.due
    arb.submit(req(at=int(arb.audio.samples) + 100))
    assert arb.sent == 2, "traffic did not resume after the callsign went out"


def test_the_arbiter_does_not_judge_channel_occupancy(arb):
    """Listening before transmitting is the operator's duty, served by the tools
    that consult `core.busy`. The arbiter once claimed to gate errands on it, but
    the receive window it judged was a stub returning no samples, so the gate
    read every channel as clear behind comments promising protection — worse
    than no gate, because an operator who read them could rely on it."""
    assert not hasattr(arb, "check_busy")
    assert not hasattr(arb, "_channel_busy")


def test_disabling_one_protocol_leaves_the_others_running(arb):
    """The property four processes gave for free and one has to build."""
    arb.disable("vara", "wedged")
    with pytest.raises(Refused, match="disabled"):
        arb.submit(req(protocol="vara", at=1000))
    arb.submit(req(protocol="pactor", at=1000))
    assert arb.sent == 1


def test_a_rig_refusal_becomes_a_refusal_and_leaves_nothing_armed(rig):
    audio = ReplayAudio(np.zeros(48000, np.int16))
    a = TxArbiter(audio, rig, identity=None)
    rig.transmit = False
    with pytest.raises(Refused, match="transmit is disabled"):
        a.submit(req(at=1000))
    assert a.sent == 0
    assert a.audio.transmitted == [], "a refused burst was still recorded as sent"


# --- the station ------------------------------------------------------------

def test_a_receive_only_station_opens_no_cat_port_and_holds_no_line(tmp_path):
    """transmit = false is a hard interlock, so the radio is never touched — which
    is cheaper than an interlock that has to be right in five places."""
    cfg = config.load(_repo() / "examples" / "replay.toml")
    st = Station(cfg, replay=np.zeros(4800, np.int16))
    st.open_audio()
    assert st.open_rig() is None
    assert st.rig is None and st.arbiter is None
    st.shutdown()


def test_the_station_says_when_it_is_automatically_controlled(tmp_path):
    """And says nothing under local control, where §97.221 does not apply — the
    shipped example is local, so asserting the note appears for it was wrong."""
    import tomllib
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())

    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    st.open_audio()
    assert "§97.221" not in st.describe()
    st.shutdown()

    raw["station"]["control"] = "automatic"
    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    st.open_audio()
    assert "§97.221" in st.describe()
    st.shutdown()


def test_a_station_with_nothing_enabled_refuses_to_run(tmp_path):
    import tomllib
    raw = tomllib.loads((_repo() / "examples" / "replay.toml").read_text())
    for p in raw["protocols"].values():
        p["enabled"] = False
    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    assert st.run() == 2


def _repo():
    from pathlib import Path
    return Path(__file__).resolve().parents[5]


def test_a_protocol_that_composes_at_its_own_rate_reaches_the_card_at_the_cards(arb):
    """ARDOP works at 12 kHz in int16 and the card plays 48 kHz float.

    Armed as composed, a 1.75 s frame goes out in 0.44 s with every tone two
    octaves high — 1500 Hz landing at 6000, outside the passband the emission gate
    checked a moment earlier. The gate cannot catch it: it is handed the
    waveform's declared band, never the samples. Nothing else caught it because no
    test here had ever transmitted at anything but the card rate.
    """
    from hfmodem.core.rates import CARD_RATE_HZ
    tone = (np.sin(2 * np.pi * 1500 * np.arange(21000) / 12000) * 27000).astype(np.int16)
    arb.submit(req(protocol="ardop", audio=tone, rate=12000))
    _, sent = arb.audio.transmitted[-1]
    assert abs(len(sent) / CARD_RATE_HZ - tone.size / 12000) < 0.01, (
        f"{tone.size / 12000:.2f} s composed, {len(sent) / CARD_RATE_HZ:.2f} s emitted")
    spec = np.abs(np.fft.rfft(np.asarray(sent, float)))
    peak = float(np.fft.rfftfreq(len(sent), 1 / CARD_RATE_HZ)[int(np.argmax(spec))])
    assert abs(peak - 1500) < 20, f"1500 Hz composed, {peak:.0f} Hz emitted"
    assert np.abs(sent).max() <= 1.0, f"card scale is [-1, 1]; peak {np.abs(sent).max()}"


def test_every_burst_leaves_at_the_configured_drive(arb):
    """sabir hands over a peak of 1.415 and the card rails at 1.0, so without
    this every sabir transmission clips — and ALC on a squared-off OFDM peak
    splatters across the band. The four protocols also disagree about level by
    more than a factor of two, so an operator setting the rig's input gain
    against one would be overdriving on the next."""
    arb.drive = 0.6
    for name, audio, rate in (
        ("sabir", np.sin(2 * np.pi * 1000 * np.arange(9600) / 48000) * 1.415, 48000),
        ("ardop", (np.sin(2 * np.pi * 1500 * np.arange(2400) / 12000) * 27000
                   ).astype(np.int16), 12000),
        ("ident", np.sin(2 * np.pi * 800 * np.arange(9600) / 48000) * 0.5, 48000),
    ):
        arb.submit(req(protocol=name, audio=audio, rate=rate))
        _, sent = arb.audio.transmitted[-1]
        peak = float(np.abs(sent).max())
        assert abs(peak - 0.6) < 0.01, f"{name} left at {peak:.3f}, not 0.600"


def test_the_drive_knob_is_the_one_in_the_config(tmp_path):
    """It was declared in `[audio]` and read by nothing, so an operator could set
    it, watch nothing change, and reasonably conclude the level was fine."""
    import tomllib
    from hfmodem.core import config
    from hfmodem.station.process import Station
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())
    raw["audio"]["tx_drive"] = 0.35
    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    st.open_audio()
    st.arbiter = TxArbiter(st.audio, None, drive=st.cfg.audio.tx_drive)
    try:
        assert st.arbiter.drive == 0.35
    finally:
        st.audio.close()
