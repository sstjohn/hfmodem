# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The arbiter's policy, which is five rules and no knobs.

Each one corresponds to something that has gone wrong on this station or to a
mistake a priority scheme makes by default. None of it needs a radio.
"""
from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import config
from hfmodem.core.audio import BurstInvalidated, ReplayAudio, TxStream
from hfmodem.core.operations import LeaseTimeout, ResourceClosed
from hfmodem.core.regulatory import Control, Unregulated, centred
from hfmodem.core.rig import Cat, Rig
from hfmodem.station.arbiter import LIVE, Identity, Refused, TxArbiter, TxRequest, longest_burst_s, RequestState
from hfmodem.station.process import Station
from hfmodem.tests.core.fakerig import FakePtt
from hfmodem.tests.core.test_rig import SocketPairRig

BENCH = Unregulated(because="unit test, no radio")


@pytest.fixture
def rig(monkeypatch):
    fake = SocketPairRig(ptt_type="None")
    monkeypatch.setattr(Cat, "_connect", lambda self, deadline: fake.connect(deadline))
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=FakePtt(),
            profile=BENCH, control=Control.LOCAL, mycall="N0CALL", transmit=True)
    # Establish the confirmed-low line and timestamped dial used by key().
    r.arm(prove_ptt=False)
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


def test_output_notice_cannot_move_a_scheduled_burst_after_policy_admission(arb, monkeypatch):
    keys, notices, cancels = [], [], []
    arm = arb.audio.arm_burst
    cancel = arb.audio.cancel_burst

    def late_arm(audio, at):
        return arm(audio, at=at + 384)

    def cancelled():
        cancels.append(True)
        cancel()

    monkeypatch.setattr(arb.audio, 'arm_burst', late_arm)
    monkeypatch.setattr(arb.audio, 'cancel_burst', cancelled)
    monkeypatch.setattr(arb.rig, 'key', lambda *a, **kw: keys.append(True))
    with pytest.raises(Refused, match='output cannot meet card index 1000'):
        arb.submit(req(at=1000, on_ptt=notices.append))
    assert cancels == [True]
    assert not keys and not notices and not arb.audio.transmitted
    assert arb.dropped_late == 1 and arb.sent == 0


def test_ungridded_burst_still_uses_the_outputs_available_index(arb, monkeypatch):
    arm = arb.audio.arm_burst
    monkeypatch.setattr(arb.audio, 'arm_burst',
                        lambda audio, at: arm(audio, at=at + 384))
    assert arb.submit(req()) == 384 + 4800
    assert arb.dropped_late == 0 and arb.sent == 1


@pytest.mark.parametrize('changed_at', ['arm', 'admit', 'reserved-admit'])
def test_another_lane_becoming_active_before_key_cancels_admitted_output(arb, monkeypatch,
                                                                      changed_at):
    owners, keys, notices, cancels = [], [], [], []
    arb.active_protocols = lambda: tuple(owners)
    arm, key, cancel = arb.audio.arm_burst, arb.rig.key, arb.audio.cancel_burst

    def armed(audio, at):
        result = arm(audio, at=at)
        if changed_at == 'arm':
            owners.append('pactor')
        return result

    def admit():
        if changed_at in ('admit', 'reserved-admit'):
            owners.append('pactor')

    def keyed(*args, **options):
        keys.append(True)
        return key(*args, **options)

    def cancelled():
        cancels.append(True)
        cancel()

    monkeypatch.setattr(arb.audio, 'arm_burst', armed)
    monkeypatch.setattr(arb.audio, 'cancel_burst', cancelled)
    monkeypatch.setattr(arb.rig, 'key', keyed)
    before = arb.transmit_history()
    options = (dict(lead_s=.02, state=RequestState()) if changed_at == 'reserved-admit' else {})
    request = req(at=None if changed_at == 'reserved-admit' else 1000,
                  admit=admit, on_ptt=notices.append, **options)
    if changed_at == 'reserved-admit':
        result = arb.submit_result(request)
        assert result.outcome == 'refused', result
        assert 'station link active on pactor' in result.reason
    else:
        with pytest.raises(Refused, match='station link active on pactor'):
            arb.submit(request)
    assert not keys and not notices and not arb.rig.keyed
    assert cancels == [True] and not arb.audio.transmitted
    assert arb.sent == 0 and arb.transmit_history() == before


def test_sample_zero_is_an_explicit_transmission_deadline(arb, monkeypatch):
    # Nonzero settle must not turn at=0 into an ungridded request.
    starts = []
    arm = arb.audio.arm_burst
    monkeypatch.setattr(arb.audio, 'arm_burst',
                        lambda audio, at: starts.append(at) or arm(audio, at=at))
    assert arb.submit(req(at=0, settle_s=.04)) == 4800
    assert starts == [0]


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
    from hfmodem.core.virtualrig import VirtualRig
    assert isinstance(st.open_rig(), VirtualRig)
    assert st.rig.cat is st.rig.ptt is None
    assert not st.rig.transmit and st.arbiter is None
    st.shutdown()


def test_the_station_says_when_it_is_automatically_controlled(tmp_path):
    """And says nothing under local control, where the automatic setting is absent — the
    shipped example is local, so asserting the note appears for it was wrong."""
    import tomllib
    raw = tomllib.loads((_repo() / "examples" / "station.toml").read_text())

    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    st.open_audio()
    assert "profile uses automatic control" not in st.describe()
    st.shutdown()

    raw["station"]["control"] = "automatic"
    st = Station(config.parse(raw), replay=np.zeros(4800, np.int16))
    st.open_audio()
    assert "profile uses automatic control" in st.describe()
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


@pytest.mark.parametrize('gain', [0., .1, .25, 1.])
def test_host_gain_survives_renderer_normalization(arb, gain):
    arb.drive = .6
    # A renderer may already apply its host control, or choose another scale.
    # The station's normalization must preserve the declared relative gain.
    for peak in (.01, .9, 27000):
        arb.submit(req(audio=np.array([-peak, 0, peak]), gain=gain))
        assert np.max(np.abs(arb.audio.transmitted[-1][1])) == pytest.approx(.6*gain)


@pytest.mark.parametrize('gain', [-.01, 1.01, float('nan'), float('inf')])
def test_invalid_gain_is_refused_before_publication_and_keying(arb, gain):
    with pytest.raises(Refused, match='gain'):
        arb.submit(req(audio=np.ones(48), gain=gain))
    assert not arb.audio.transmitted and not arb.rig.keyed


def test_ardop_drivelevel_reaches_the_recorded_station_output(arb, monkeypatch):
    from types import SimpleNamespace
    from hfmodem.besra.host.server import HostServer
    from hfmodem.station.link import BesraLink
    monkeypatch.setattr('hfmodem.besra.arq.modem.modulator.render_frame',
                        lambda *a, **kw: np.array([-10000, 0, 10000], dtype='<i2'))
    station = SimpleNamespace(audio=arb.audio, rig=arb.rig, arbiter=arb,
                              cfg=SimpleNamespace(declared_power_w=1, rig=SimpleNamespace(
                                  power_w=1, ptt=SimpleNamespace(settle_s=0))))
    link = BesraLink(station)
    host = HostServer(link.modem, control_port=0, data_port=0, quiet=True)
    events = []
    host._send_cmd = events.append
    host._send_data = lambda *args: None
    link.modem.start(host)
    try:
        host._dispatch('MYCALL W9SSJ')
        for gain in (100, 25, 0):
            host._dispatch(f'DRIVELEVEL {gain}')
            host._dispatch('SENDID')
            assert events[-1] == 'SENDID'
            assert np.max(np.abs(arb.audio.transmitted[-1][1])) == pytest.approx(.6*gain/100)
    finally:
        link.stop()


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


class Card:
    def __init__(self, lat=1152, tx=960):
        self.lat, self.tx_latency_n = lat, tx
        self.loop_lat = lat + tx
        self.samples = self.n = 10000
        self.blocksize = 128
        self.edges = []
        self.pending = self.raw = None
        self.reserved = False
        self.finished = self.cancelled = 0
        self.bad = False
        self.wait_entered = threading.Event()
        self.wait_gate = None
        self.cancel_gate = None

    def sample_now(self):
        return self.n

    def alive(self):
        pass

    def check_callbacks(self):
        if self.bad:
            raise BurstInvalidated("callbacks stopped")

    def dac_frame_at(self, timestamp):
        return self.n - self.lat + (timestamp - time.monotonic()) * 48000

    def arm_burst(self, samples, at):
        self.reserved = False
        at = max(at, self.samples + self.lat + 3 * self.blocksize)
        self.pending = samples, at
        return at

    def reserve_burst(self, samples, at, deadline):
        self.reserved = True
        actual = max(at, self.samples + self.loop_lat + 3 * self.blocksize)
        self.pending = samples, actual
        assert actual - self.tx_latency_n + len(samples) + 4800 <= deadline
        return actual

    def wait_until_dac(self, frame, **kw):
        self.edges.append(frame)
        if len(self.edges) == 2 and self.wait_gate is not None:
            self.wait_entered.set()
            while not self.wait_gate.wait(0.002):
                if kw['cancelled']():
                    raise BurstInvalidated("cancelled")
        if kw.get('cancelled', lambda: False)():
            raise BurstInvalidated("cancelled")
        self.n = max(self.n, frame + self.lat)

    def wait_until(self, index, **kw):
        self.n = max(self.n, index)

    def finish_burst(self):
        self.finished += 1
        end = self.pending[1] + len(self.pending[0]) if self.pending else (
            self.raw.end if self.raw else self.n)
        if self.pending and not self.reserved:
            end += self.tx_latency_n
        self.pending = self.raw = None
        return end

    def cancel_burst(self):
        if self.cancel_gate is not None:
            self.cancel_gate.wait(3)
        self.cancelled += 1
        self.pending = self.raw = None

    def arm_stream(self, at, capacity):
        self.raw = TxStream(at, capacity)
        return self.raw

    def emit(self):
        raw = self.raw
        data = raw._read(raw.queued)
        if len(data):
            raw._emitted(len(data), max(self.n, raw.at))
        self.n = max(self.n, raw.end)


class LineRig:
    def __init__(self, card):
        self.card = card
        self.max_key_s = 30
        self.retired = self.keyed = False
        self.line_state = "low"
        self.calls = []
        self.key_gate = self.unkey_gate = None
        self.entered = threading.Event()
        self.panicked = threading.Event()
        self.fatal = None
        self.unconfirmed = False

    def hold(self, deadline_index=None):
        return nullcontext(True)

    def key(self, emission, **kw):
        self.calls.append(("key", self.card.n, kw))
        self.entered.set()
        if self.key_gate is not None:
            self.key_gate.wait(3)
        self.keyed = True
        self.line_state = "high"
        return SimpleNamespace(card_index=self.card.n, card_deadline=self.card.n + 30 * 48000)

    def unkey(self, token):
        self.calls.append(("unkey", self.card.n, threading.current_thread().name))
        if self.unkey_gate is not None:
            self.unkey_gate.wait(3)
        self.keyed = False
        self.line_state = "unconfirmed" if self.unconfirmed else "low"

    def panic_unkey(self, why):
        self.calls.append(("panic", why))
        self.retired, self.keyed, self.line_state = True, False, "low"
        self.panicked.set()


@pytest.fixture
def bench():
    card = Card()
    rig = LineRig(card)
    arbiter = TxArbiter(card, rig)
    yield arbiter
    if rig.key_gate:
        rig.key_gate.set()
    if rig.unkey_gate:
        rig.unkey_gate.set()
    if card.cancel_gate:
        card.cancel_gate.set()
    arbiter.close()


def background(fn):
    result = []
    def run():
        try:
            result.append(fn())
        except BaseException as exc:
            result.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def joined(thread):
    thread.join(4)
    assert not thread.is_alive()


def lease_of(arb, **kw):
    return arb.lease(protocol="rtty", emission=centred(7_100_000, 500), **kw)


@pytest.mark.parametrize('lat', [128, 1152, 3000])
@pytest.mark.parametrize('tx', [0, 960])
@pytest.mark.parametrize('reserved', [False, True])
def test_finite_edges_use_dac_frames_and_keep_emission_endpoint(bench, lat, tx, reserved):
    card = bench.audio
    card.tx_latency_n = tx
    card.lat, card.loop_lat = lat, lat + card.tx_latency_n
    request = req(settle_s=0.04, lead_s=0.08 if reserved else None)
    result = bench.submit_result(request)
    assert result.outcome == 'sent'
    end = result.end
    if reserved:
        assert card.edges == [end - card.loop_lat]
    else:
        assert card.edges == [end - 4800 - card.loop_lat - 1920, end - card.loop_lat]
    assert result.emission_start == end - 4800
    assert result.emission_end == end
    record = bench.records.get_nowait()
    assert record['emission_start'] == result.emission_start
    assert record['emission_end'] == end
    assert bench.rig.calls[-1][1] == end - card.tx_latency_n
    assert bench.last_sent['vara'] == end


@pytest.mark.parametrize('tx', [0, 960])
def test_scheduled_finite_at_is_on_air_capture_index(bench, tx):
    card = bench.audio
    card.tx_latency_n = tx
    card.loop_lat = card.lat + tx
    card.wait_gate = threading.Event()
    at = card.n + 10000
    thread, result = background(lambda: bench.submit_result(req(at=at, settle_s=.04)))
    try:
        assert card.wait_entered.wait(1)
        assert card.pending[1] == at - tx
        assert card.edges == [at - card.loop_lat - 1920, at - card.loop_lat + 4800]
    finally:
        card.wait_gate.set()
        joined(thread)
    assert result[0].emission_start == at
    assert result[0].emission_end == at + 4800


def test_settle_and_watchdog_slack_must_fit_before_any_key(bench):
    bench.rig.max_key_s = 2.5
    with pytest.raises(Refused, match='complete keyed interval'):
        bench.submit(req(settle_s=1))
    assert not bench.rig.calls


@pytest.mark.parametrize("max_key_s, budget", [(6.0, 3.957), (10.043, 8.0), (20.0, 15.9656)])
def test_a_burst_sized_by_the_budget_is_admitted_and_one_sample_more_is_refused(bench, max_key_s, budget):
    assert longest_burst_s(max_key_s, 0.043) == pytest.approx(budget)
    bench.rig.max_key_s = max_key_s
    n = int(longest_burst_s(max_key_s, 0.043) * 9600)
    bench.submit(req(audio=np.zeros(n, np.float32), rate=9600, settle_s=0.043))
    with pytest.raises(Refused, match="complete keyed interval"):
        bench.submit(req(audio=np.zeros(n + 1, np.float32), rate=9600, settle_s=0.043, seq=2))
    assert bench.sent == 1


def test_station_requests_are_stamped_at_creation_not_admission(bench):
    old = req()
    bench.stop()
    with pytest.raises(Refused, match='authority'):
        bench.submit(old)
    assert bench.submit(req())


def test_explicit_session_authority_revoked_before_admission(bench):
    old = req(authority=bench.authority('session'))
    bench.stop('session')
    result = bench.submit_result(old)
    assert result.outcome == 'refused'
    assert not bench.rig.calls


@pytest.mark.parametrize('stateful', [False, True])
def test_live_arrival_during_lease_is_refused_immediately(bench, stateful):
    with lease_of(bench) as lease:
        if stateful:
            assert bench.submit_result(req()).outcome == 'refused'
        else:
            with pytest.raises(Refused, match='leased'):
                bench.submit(req())
        assert lease.end() == lease.key_index


@pytest.mark.parametrize('ended', [False, True])
def test_waiter_cannot_cross_an_intervening_lease(bench, monkeypatch, ended):
    flight = bench._acquire(req(protocol='hold'), wait=False)
    sleeping, wake = threading.Event(), threading.Event()
    original = bench._admit.wait
    def delayed_wait(timeout):
        sleeping.set()
        original(timeout)
        bench._admit.release()
        try:
            assert wake.wait(2)
        finally:
            bench._admit.acquire()
    monkeypatch.setattr(bench._admit, 'wait', delayed_wait)
    waiter, result = background(lambda: bench.submit(req()))
    assert sleeping.wait(1)
    bench._release(flight)
    lease = lease_of(bench).__enter__()
    assert 'rtty' in bench.active_protocols()
    if ended:
        lease.end()
    wake.set()
    joined(waiter)
    assert isinstance(result[0], Refused)
    assert 'intervening lease' in str(result[0])
    if not ended:
        lease.end()


@pytest.mark.parametrize('reserved', [False, True])
def test_stop_during_keying_is_contained_by_worker(bench, reserved):
    bench.rig.key_gate = threading.Event()
    request = req(lead_s=0.04 if reserved else None)
    thread, result = background(lambda: bench.submit_result(request))
    assert bench.rig.entered.wait(1)
    bench.stop()
    bench.rig.key_gate.set()
    joined(thread)
    assert result[0].outcome == 'invalidated'
    assert result[0].reason == 'stopped'
    assert bench.sent == 0 and not bench.rig.keyed
    assert all(call[2] == 'tx-unkey' for call in bench.rig.calls if call[0] == 'unkey')


def test_stop_overtaking_a_real_key_up_reports_stopped_with_the_line_down(arb, monkeypatch):
    rig = arb.rig
    entered, contained, resume = threading.Event(), threading.Event(), threading.Event()
    original = rig.ptt.assert_
    def assert_line(on):
        if on:
            entered.set()
            assert resume.wait(2)
        elif threading.current_thread().name == 'tx-unkey':
            contained.set()
        original(on)
    monkeypatch.setattr(rig.ptt, 'assert_', assert_line)
    thread, result = background(lambda: arb.submit_result(req()))
    try:
        assert entered.wait(2)
        arb.stop()
        assert contained.wait(2)
    finally:
        resume.set()
    joined(thread)
    assert (result[0].outcome, result[0].reason) == ('invalidated', 'stopped')
    assert rig.ptt.line is False and not rig.keyed and rig._timer is None
    assert rig.line_state == 'low' and not rig.retired
    assert arb.sent == 0 and arb.refused == 0


def test_a_stopped_key_up_that_fails_otherwise_still_propagates(bench, monkeypatch):
    bench.rig.key_gate = threading.Event()
    key = bench.rig.key
    def closing(emission, **kw):
        key(emission, **kw)
        raise ResourceClosed("resource is closing or closed")
    monkeypatch.setattr(bench.rig, 'key', closing)
    thread, result = background(lambda: bench.submit(req()))
    assert bench.rig.entered.wait(1)
    bench.stop()
    bench.rig.key_gate.set()
    joined(thread)
    assert isinstance(result[0], ResourceClosed)
    assert bench.sent == 0 and not bench.rig.keyed


def test_stop_in_flight_cuts_once_and_never_reports_sent(bench):
    bench.audio.wait_gate = threading.Event()
    thread, result = background(lambda: bench.submit_result(req()))
    assert bench.audio.wait_entered.wait(1)
    bench.stop()
    cut = bench._flight.cut
    bench.audio.n += 100
    bench.stop()
    joined(thread)
    assert result[0].outcome == 'invalidated'
    assert bench.transmit_history().end == cut
    assert bench.audio.cancelled and bench.audio.pending is None


def test_native_contacts_and_scanning_share_admission_with_streaming_leases(bench):
    boundary = bench.receive_boundary
    with lease_of(bench):
        assert bench.claim_contact('vara') is None
        assert bench.acquire_scan_hold('pactor', eligible=lambda: True) is None
        with bench.try_hold() as held:
            assert not held
        assert bench.receive_boundary == boundary
    contact = bench.claim_contact('vara')
    assert contact is not None
    with pytest.raises(Refused, match='station link active on vara'):
        with lease_of(bench):
            pytest.fail('streaming lease crossed the native contact')
    bench.release_contact('vara', contact)
    scan = bench.acquire_scan_hold('pactor', eligible=lambda: True)
    assert scan is not None
    with pytest.raises(Refused, match='scanner lease'):
        with lease_of(bench):
            pytest.fail('streaming lease crossed the scanner hold')
    bench.release_scan_hold(scan)


@pytest.mark.parametrize('blocked', ['unkey', 'cancel'])
@pytest.mark.wall_clock
def test_stop_handler_does_not_wait_for_blocked_containment(bench, blocked):
    bench.audio.wait_gate = threading.Event()
    gate = threading.Event()
    if blocked == 'unkey':
        bench.rig.unkey_gate = gate
    else:
        bench.audio.cancel_gate = gate
    thread, result = background(lambda: bench.submit_result(req()))
    assert bench.audio.wait_entered.wait(1)
    start = time.monotonic()
    bench.stop()
    assert time.monotonic() - start < 0.1
    assert bench.rig.panicked.wait(1.5)
    assert bench._owner is not None
    gate.set()
    joined(thread)
    # A bounded caller may return while ownership waits for the old worker.
    deadline = time.monotonic() + 1
    while bench._flight is not None and not bench._flight.drained.is_set() and time.monotonic() < deadline:
        time.sleep(.005)
    if bench._flight is not None:
        bench._release_when_safe(bench._flight)
    assert bench.rig.retired


def test_unreadable_unkey_escalates(bench):
    bench.rig.unconfirmed = True
    with pytest.raises(BurstInvalidated, match='retired'):
        bench.submit(req())
    assert bench.rig.panicked.is_set()


def test_complete_marks_do_not_close_or_unkey(bench):
    with lease_of(bench) as lease:
        stream = lease.stream()
        stream.write(np.ones(10))
        bench.audio.emit()
        endpoint = lease.complete(10)
        assert endpoint == lease.key_index + 10
        assert bench.rig.keyed and not stream.closed
        assert bench.audio.finished == 0
        stream.write(np.ones(7))
        bench.audio.emit()
        assert lease.end() == endpoint + 7
    assert bench.audio.finished == 1 and bench._owner is None


def test_empty_over_waits_for_settle(bench):
    with lease_of(bench, settle_s=.04) as lease:
        bench.audio.n += 1920
        assert lease.complete(0) == lease.key_index + 1920
        assert bench.rig.keyed
        assert lease.end() == lease.key_index + 1920


def test_lease_heartbeat_and_automatic_ceiling(bench):
    lease = lease_of(bench).__enter__()
    lease._touched -= 2
    assert lease._flight.done.wait(1)
    assert lease._flight.reason == 'holder heartbeat expired'
    assert bench.rig.keyed is False
    lease = lease_of(bench).__enter__()
    lease._flight.key_started -= 31
    assert lease._flight.done.wait(1)
    assert lease._flight.reason == 'ceiling'


def test_attended_policy_comes_from_registry_callback(bench):
    bench._attended = lambda session: session in (None, 'human')
    bench.identity = Identity('N0CALL', interval_s=0)
    with lease_of(bench, authority=bench.authority('human')) as lease:
        assert lease.ceiling_s is None
        assert bench.rig.calls[0][2]['open'] is True
        lease.end()
    assert bench.identify_if_free(centred(7_100_000, 500)) is False


@pytest.mark.parametrize('attended', [True, False])
def test_holder_heartbeat_is_one_second_for_every_lease(bench, monkeypatch, attended):
    bench._attended = lambda session: attended
    with lease_of(bench) as lease:
        assert lease.heartbeat_s == 1.0
        touched = lease._touched
        with bench._admit:
            monkeypatch.setattr(time, 'monotonic', lambda: touched + .999)
            assert lease._fault() is None
            monkeypatch.setattr(time, 'monotonic', lambda: touched + 1.0)
            assert lease._fault() == 'holder heartbeat expired'
            monkeypatch.undo()
        lease.end()


def test_attended_silent_holder_stall_unkeys_after_one_second(bench):
    bench._attended = lambda session: True
    authority = bench.authority('human')
    lease = lease_of(bench, idle='silence', authority=authority).__enter__()
    lease.stream()
    bench.audio.raw.starved_at = bench.audio.n
    lease._touched = time.monotonic() - 1.01
    bench._wake.set()
    assert lease._flight.drained.wait(1)
    assert lease._flight.reason == 'holder heartbeat expired'
    assert not bench.rig.keyed
    assert bench.authority('human') != authority


@pytest.mark.parametrize('idle', ['fault', 'silence'])
def test_starvation_policy(bench, idle):
    lease = lease_of(bench, idle=idle).__enter__()
    lease.stream()
    bench.audio.raw.starved_at = bench.audio.n
    lease._starved_since = time.monotonic() - 2
    assert lease._fault() == ('stream starved' if idle == 'fault' else None)
    lease.end()


# Another thread must refresh its real heartbeat within 30ms.
@pytest.mark.wall_clock
def test_completion_wait_touches_but_requires_progress(bench):
    lease = lease_of(bench).__enter__()
    stream = lease.stream()
    stream.write(np.ones(100))
    thread, result = background(lambda: lease.complete(100))
    time.sleep(.06)
    assert time.monotonic() - lease._touched < .03
    lease._progress_at -= 2
    assert lease._flight.done.wait(1)
    joined(thread)
    assert isinstance(result[0], BurstInvalidated)
    assert lease._flight.reason == 'card progress stalled'


@pytest.mark.parametrize('phase', ['keying', 'filling', 'waiting'])
def test_takeover_drains_only_accepted_stream_data(bench, phase):
    lease = lease_of(bench, authority=bench.authority('human'))
    if phase == 'keying':
        bench.rig.key_gate = threading.Event()
        thread, result = background(lease.__enter__)
        assert bench.rig.entered.wait(1)
        bench.takeover('human')
        bench.rig.key_gate.set()
        joined(thread)
        assert result == [lease]
    else:
        lease.__enter__()
        stream = lease.stream()
        stream.write(np.ones(100))
        if phase == 'waiting':
            thread, result = background(lambda: lease.complete(100))
        bench.takeover('human')
        with pytest.raises((Refused, BurstInvalidated)):
            stream.write(np.ones(1))
        bench.audio.emit()
        if phase == 'waiting':
            joined(thread)
            assert result == [lease.key_index + 100]
    assert lease._flight.done.wait(1)
    assert not lease._flight.stop.is_set()
    assert bench.audio.finished == 1


def test_takeover_allows_committed_finite_to_finish(bench):
    bench.audio.wait_gate = threading.Event()
    thread, result = background(lambda: bench.submit_result(req(authority=bench.authority('human'))))
    assert bench.audio.wait_entered.wait(1)
    bench.takeover('human')
    bench.audio.wait_gate.set()
    joined(thread)
    assert result[0].outcome == 'sent'
    assert bench.sent == 1


def test_release_revokes_unadmitted_session_requests(bench):
    old = req(authority=('not-yet-admitted', 0))
    bench.revoke()
    with pytest.raises(Refused, match='authority'):
        bench.submit(old)
    assert bench.submit(req())


# The blocked writer must refresh its real heartbeat within 30ms.
@pytest.mark.wall_clock
def test_blocking_stream_write_touches_and_preserves_capacity(bench):
    with lease_of(bench, capacity=100) as lease:
        stream = lease.stream()
        stream.write(np.ones(100))
        thread, result = background(lambda: stream.write_blocking(np.ones(40)))
        time.sleep(.05)
        assert time.monotonic() - lease._touched < .03
        assert stream.queued == 100
        bench.audio.emit()
        joined(thread)
        assert result == [None] and stream.queued == 40
        bench.audio.emit()
        lease.end()


def test_callback_failure_halts_a_lease(bench):
    lease = lease_of(bench).__enter__()
    bench.audio.bad = True
    assert lease._flight.done.wait(1)
    assert lease._flight.reason == 'callbacks stopped'
    assert not bench.rig.keyed


def test_supervisor_fault_exception_contains_open_key_and_keeps_running(bench, monkeypatch, caplog):
    authority = bench.authority('failed-session')
    other_authority = bench.authority('other-session')
    lease = lease_of(bench, authority=authority).__enter__()
    supervisor = bench._supervisor
    error = RuntimeError('fault inspection failed')
    def broken_fault():
        raise error
    monkeypatch.setattr(lease, '_fault', broken_fault)
    bench.rig.unkey_gate = threading.Event()
    try:
        bench._wake.set()
        deadline = time.monotonic() + 1
        while lease._flight.worker is None and time.monotonic() < deadline:
            time.sleep(.002)
        assert lease._flight.worker is not None
        assert supervisor.is_alive()
        assert lease._flight.stop.is_set()
        assert lease._flight.error is error
    finally:
        bench.rig.unkey_gate.set()
    assert lease._flight.drained.wait(1)
    assert not bench.rig.keyed
    with pytest.raises(RuntimeError, match='fault inspection failed'):
        lease.end()
    assert 'transmit supervisor failed' in caplog.text
    with pytest.raises(Refused, match='authority revoked'):
        bench.submit(req(authority=authority))
    assert bench.authority('other-session') == other_authority
    with lease_of(bench) as next_lease:
        next_lease.end()


@pytest.mark.parametrize('new_owner', [False, True])
def test_containment_completion_during_worker_exit_check_does_not_escalate(
        bench, monkeypatch, new_owner):
    # Drive the supervisor synchronously so completion happens exactly between
    # its initial done check and the worker's is_alive() result.
    monkeypatch.setattr(bench, '_ensure_supervisor', lambda: None)
    authority = bench.authority('finishing-session')
    other_authority = bench.authority('other-session')
    request = req(authority=authority)
    flight = bench._acquire(request, wait=False)
    bench._key(request, flight)
    flight.finish.set()
    fatal_calls = []
    bench.rig.fatal = fatal_calls.append
    replacement = []
    checks = []

    def complete_at_exit_check():
        assert not flight.done.is_set()
        bench._contain(flight)
        if new_owner:
            bench._release_when_safe(flight)
            replacement.append(lease_of(bench, authority=other_authority).__enter__())
        return False

    def next_pass(timeout):
        if flight.done.is_set():
            checks.append((bench._owner, bench.rig.keyed))
            if replacement:
                bench._contain(replacement[0]._flight)
            else:
                bench._release_when_safe(flight)
        return True

    flight.worker = SimpleNamespace(is_alive=complete_at_exit_check)
    monkeypatch.setattr(bench._wake, 'wait', next_pass)
    try:
        bench._supervise()
        assert checks == [(replacement[0] if new_owner else request, new_owner)]
        assert flight.done.is_set() and flight.drained.is_set()
        assert flight.error is None and not flight.stop.is_set()
        assert not bench._closed and not bench.rig.panicked.is_set()
        assert fatal_calls == []
        assert bench.authority('finishing-session') == authority
        assert bench.authority('other-session') == other_authority
    finally:
        remaining = bench._flight
        if remaining is not None:
            bench._contain(remaining)
            bench._release_when_safe(remaining)


@pytest.mark.parametrize('worker_name', ['tx-containment', 'tx-unkey'])
@pytest.mark.parametrize('panic_raises', [False, True])
def test_containment_start_failure_escalation_is_bounded(monkeypatch, worker_name, panic_raises):
    card = Card()
    rig = LineRig(card)
    arb = TxArbiter(card, rig)
    lease = lease_of(arb).__enter__()
    supervisor = arb._supervisor
    authority = lease.authority
    attempts = []
    fatal_calls = []
    start = threading.Thread.start
    def broken_start(thread):
        if thread.name == worker_name:
            attempts.append(time.monotonic())
            raise RuntimeError('cannot start containment worker')
        return start(thread)
    def panic(why):
        rig.calls.append(('panic', why))
        if panic_raises:
            raise RuntimeError('cannot start panic worker')
        rig.retired, rig.keyed, rig.line_state = True, False, 'low'
    monkeypatch.setattr(threading.Thread, 'start', broken_start)
    monkeypatch.setattr(rig, 'panic_unkey', panic)
    rig.fatal = fatal_calls.append
    began = time.monotonic()
    lease.halt('operator stop')
    assert lease._flight.done.wait(.5)
    joined(supervisor)
    assert time.monotonic() - began < .5
    assert len(attempts) == (2 if worker_name == 'tx-containment' else 1)
    assert [call[0] for call in rig.calls] == ['key', 'panic']
    assert fatal_calls == ['transmit supervisor recovery failed']
    assert isinstance(lease._flight.error, LeaseTimeout)
    assert lease.state == 'halted'
    assert arb.authority(authority[0]) != authority
    assert not lease._flight.drained.is_set()
    arb._release_when_safe(lease._flight)
    assert arb._owner is lease
    with pytest.raises(Refused, match='closed'):
        arb.submit(req())


@pytest.mark.parametrize('inspection_fails', [False, True])
def test_single_containment_start_failure_recovers_and_revokes(bench, monkeypatch, inspection_fails):
    authority = bench.authority('failed-session')
    other_authority = bench.authority('other-session')
    lease = lease_of(bench, authority=authority).__enter__()
    supervisor = bench._supervisor
    attempts = []
    inspections = []
    fatal_calls = []
    bench.rig.fatal = fatal_calls.append
    start = threading.Thread.start
    def fail_once(thread):
        if thread.name == 'tx-containment':
            attempts.append(thread)
            if len(attempts) == 1:
                raise RuntimeError('cannot start containment worker')
        return start(thread)
    monkeypatch.setattr(threading.Thread, 'start', fail_once)
    if inspection_fails:
        def broken_fault():
            inspections.append(lease._flight)
            raise RuntimeError('fault inspection failed')
        monkeypatch.setattr(lease, '_fault', broken_fault)
    else:
        lease._flight.finish.set()
    bench._wake.set()
    assert lease._flight.drained.wait(1)
    joined(supervisor)
    assert inspections == ([lease._flight] if inspection_fails else [])
    assert len(attempts) == 2
    assert not bench.rig.keyed and not bench.rig.panicked.is_set()
    assert not bench._closed and fatal_calls == []
    assert lease.state == 'halted'
    with pytest.raises(Refused, match='authority revoked'):
        bench.submit(req(authority=authority))
    assert bench.authority('other-session') == other_authority
    with lease_of(bench, authority=bench.authority('failed-session')) as next_lease:
        next_lease.end()


@pytest.mark.parametrize('has_fatal', [False, True])
def test_supervisor_contains_exception_from_halt_handler(monkeypatch, has_fatal):
    card = Card()
    rig = LineRig(card)
    arb = TxArbiter(card, rig)
    lease = lease_of(arb).__enter__()
    supervisor = arb._supervisor
    authority = lease.authority
    fatal_calls = []
    rig.fatal = fatal_calls.append if has_fatal else None
    def broken_fault():
        raise RuntimeError('fault inspection failed')
    def broken_clock():
        raise RuntimeError('cut clock failed')
    monkeypatch.setattr(card, 'sample_now', broken_clock)
    monkeypatch.setattr(lease, '_fault', broken_fault)
    began = time.monotonic()
    arb._wake.set()
    assert lease._flight.done.wait(.5)
    joined(supervisor)
    assert time.monotonic() - began < .5
    assert rig.panicked.is_set() and not rig.keyed
    assert fatal_calls == (['transmit supervisor recovery failed'] if has_fatal else [])
    assert lease._flight.stop.is_set() and lease.state == 'halted'
    assert isinstance(lease._flight.error, LeaseTimeout)
    assert arb.authority(authority[0]) != authority
    assert not lease._flight.drained.is_set()
    arb._release_when_safe(lease._flight)
    assert arb._owner is lease
    with pytest.raises(Refused, match='closed'):
        arb.submit(req())


@pytest.mark.parametrize('blocked', ['unkey', 'cancel'])
def test_containment_without_fatal_completes_with_timeout_but_retains_owner(bench, monkeypatch, blocked):
    lease = lease_of(bench).__enter__()
    gate = threading.Event()
    original = bench.rig.unkey if blocked == 'unkey' else bench.audio.cancel_burst
    def stuck(*args):
        gate.wait()
        return original(*args)
    monkeypatch.setattr(bench.rig if blocked == 'unkey' else bench.audio,
                        'unkey' if blocked == 'unkey' else 'cancel_burst', stuck)
    try:
        lease.halt('operator stop')
        assert lease._flight.done.wait(2)
        assert isinstance(lease._flight.error, LeaseTimeout)
        assert not lease._flight.drained.is_set()
        assert bench.rig.retired and bench.rig.panicked.is_set()
        with pytest.raises(LeaseTimeout, match='did not drain'):
            lease.end()
        bench._release_when_safe(lease._flight)
        assert bench._owner is lease
        with pytest.raises(Refused, match='leased'):
            bench.submit(req())
    finally:
        gate.set()
    assert lease._flight.drained.wait(1)
    deadline = time.monotonic() + 1
    while bench._owner is not None and time.monotonic() < deadline:
        time.sleep(.002)
    assert bench._owner is None


def test_keyup_timeout_keeps_ownership_until_late_key_is_contained(bench):
    bench.rig.key_gate = threading.Event()
    lease = lease_of(bench)
    thread, result = background(lease.__enter__)
    assert bench.rig.entered.wait(1)
    lease._flight.key_started -= 2
    assert bench.rig.panicked.wait(1.5)
    assert bench._owner is lease
    bench.rig.key_gate.set()
    joined(thread)
    assert isinstance(result[0], Exception)
    assert lease._flight.drained.wait(1)
    assert not bench.rig.keyed and bench._owner is None


def test_stop_while_waiting_for_ownership_revalidates_authority(bench):
    flight = bench._acquire(req(protocol='hold'), wait=False)
    request = req(authority=bench.authority('waiting'))
    thread, result = background(lambda: bench.submit(request))
    time.sleep(.03)
    bench.stop('waiting')
    joined(thread)
    assert isinstance(result[0], Refused)
    assert not bench.rig.calls
    bench._release(flight)


@pytest.mark.wall_clock
def test_identity_errand_keeps_caller_free_and_skips_lease(bench):
    bench.identity = Identity('N0CALL', interval_s=600)
    lease = lease_of(bench).__enter__()
    bench.identity._last -= 1000
    assert not bench.identify_if_free(centred(7_100_000, 500))
    lease.end()
    bench.audio.wait_gate = threading.Event()
    start = time.monotonic()
    assert bench.identify_if_free(centred(7_100_000, 500))
    assert time.monotonic() - start < .1
    assert bench.audio.wait_entered.wait(1)
    assert not bench.identify_if_free(centred(7_100_000, 500))
    bench.audio.wait_gate.set()
    joined(bench._errand)
    assert not bench.identity.due


def test_late_halt_of_released_lease_does_not_revoke_next_over(bench):
    lease = lease_of(bench).__enter__()
    lease.end()
    next_request = req(authority=bench.authority())
    lease.halt('late callback')
    assert bench.submit(next_request)


def test_unkey_exception_requires_panic_even_if_cached_state_is_low(bench, monkeypatch):
    def broken_unkey(token):
        bench.rig.line_state = 'low'
        raise OSError('ioctl failed')
    monkeypatch.setattr(bench.rig, 'unkey', broken_unkey)
    with pytest.raises(OSError, match='ioctl failed'):
        bench.submit(req())
    assert bench.rig.panicked.is_set()
    assert bench._owner is None


def test_halt_publishes_cut_before_blocked_worker_returns(bench):
    lease = lease_of(bench).__enter__()
    bench.rig.unkey_gate = threading.Event()
    lease.halt('operator stop')
    record = bench.records.get_nowait()
    assert record['outcome'] == 'operator stop'
    assert record['cut'] == bench.audio.n
    lease.halt('later stop')
    assert bench.records.empty()
    bench.rig.unkey_gate.set()
    assert lease._flight.done.wait(1)


def test_unconfirmed_lease_release_is_not_success(bench):
    lease = lease_of(bench).__enter__()
    bench.rig.unconfirmed = True
    with pytest.raises(BurstInvalidated, match='retired'):
        lease.end()
    assert lease.state == 'halted' and bench.rig.retired
def test_station_identification_declares_the_passband_of_its_actual_cw_audio(arb, monkeypatch):
    from types import SimpleNamespace
    from hfmodem.core.rates import CARD_RATE_HZ
    station = Station.__new__(Station)
    station.arbiter, station.rig = arb, arb.rig
    station.cfg = SimpleNamespace(declared_power_w=10, rig=SimpleNamespace(power_w=10))
    arb.identity.end()
    emissions = []
    identify = arb.identify
    def record(emission):
        emissions.append(emission)
        return identify(emission)
    monkeypatch.setattr(arb, 'identify', record)
    station._identify_if_due()
    assert len(emissions) == len(arb.audio.transmitted) == 1
    samples = arb.audio.transmitted[0][1]
    peak = np.fft.rfftfreq(len(samples), 1/CARD_RATE_HZ)[np.abs(np.fft.rfft(samples)).argmax()]
    emission = emissions[0]
    assert emission.audio_lo_hz <= peak <= emission.audio_hi_hz
    assert emission.power_w == 10 and not arb.identity.due
