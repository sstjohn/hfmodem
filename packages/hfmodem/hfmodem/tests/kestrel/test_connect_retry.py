# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The connect tool's transmit cadence after the link-setup.

The CR train stops the moment a connect-response comes back, and until now
nothing replaced it: the link-setup went out once and the attempt spent the rest
of its window listening. The handshake re-sends when the gateway *repeats* its
connect-response, which covers a gateway that heard us badly — not one that heard
nothing at all and therefore says nothing at all. These drive the real
``connect()`` loop over a fake transport: no device, no rig, no transmitter.

Behind the link-setup ladder is the CR ladder it falls back to. A gateway that
answers the request and then says nothing is what a fade looks like from this end
— the 4.36 s over dies in fades the 1.77 s request rides through, for a real VARA
as much as for kestrel (``working/bench-findings-20260814`` §5) — so the attempt
goes back to step 1 and calls again rather than ending on it.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (
    _CONNECT_MAX_RESTARTS,
    _I_CR_SENT,
    _LINKSETUP_MAX_TX,
    PeerAnswer,
    VaraIO,
    VaraStationHandshake,
)

kc = corpora.harness("kestrel_connect")

FS = MK.FS
_GATEWAY = "NS0A"
_LINK_SETUP_MIN = 2 * FS          # the wideband over is 4.4 s; every MFSK burst is under 2
_RESPONSE = MK.synth_burst(_GATEWAY, VF.CONNECT_RESPONSE)
_CR_LEN = len(MK.synth_burst(_GATEWAY, VF.connect_request("2300")))


class _FakeIO(VaraIO):
    """A transport that answers the CR once and then goes quiet, like a gateway
    that heard the connect-request and missed the link-setup.

    ``refuses`` is `AudioVaraIO`'s key-up refusal in miniature: the rig does not come
    up, so the samples are dropped and `VaraIO.tx_went_out` — the real verdict the
    handshake reads — says nothing went out.

    ``answers_every_cr`` is the gateway of the bench measurement instead: it hears
    every request and no link-setup at all, so it answers each CR and acks nothing.
    That is the station the restart ladder is aimed at, and the one that would keep
    an unbounded ladder keying for the whole operating window.
    """

    def __init__(self, replies: list[np.ndarray], refuses: bool = False,
                 answers_every_cr: bool = False):
        self.replies = list(replies)
        self.refuses = refuses
        self.answers_every_cr = answers_every_cr
        self.offered: list[int] = []    # handed to tx(), transmitted or not
        self.sent: list[int] = []       # what actually reached the transmitter
        self.rig = None                 # no transmitter behind this transport

    def start(self): ...
    def pending(self): ...
    def connected(self, *a): ...
    def log(self, msg): ...

    def key(self, on):
        if on:                          # decided afresh at every key-up, as on the air
            self._refused = self.refuses

    def tx(self, samples):
        self.offered.append(len(samples))
        if self._refused:
            return
        self.sent.append(len(samples))
        if self.answers_every_cr and len(samples) == _CR_LEN:
            self.replies.append(_RESPONSE)

    def next_rx_burst(self, timeout, hs=None):
        """``hs`` is where the live transport feeds the raw receive stream. This
        one hands bursts over ready-made and has no stream to offer, so it does not
        — which leaves the answer to the bracket route, the case that has to keep
        working for every transport that is not an audio device."""
        import time
        time.sleep(0.02)
        return self.replies.pop(0) if self.replies else None

    @property
    def link_setups(self) -> int:
        return sum(n >= _LINK_SETUP_MIN for n in self.sent)

    @property
    def link_setups_offered(self) -> int:
        return sum(n >= _LINK_SETUP_MIN for n in self.offered)

    @property
    def connect_requests(self) -> int:
        return sum(n == _CR_LEN for n in self.sent)


def _run(replies, refuses=False, answers_every_cr=False,
         **kw) -> tuple[_FakeIO, VaraStationHandshake]:
    io = _FakeIO(replies, refuses, answers_every_cr)
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300",
                              max_link_setups=kw.pop("max_link_setups", _LINKSETUP_MAX_TX))
    kc.connect(_GATEWAY, "W9SSJ", "2300", io, timeout=kw.pop("timeout", 1.2),
               listen_first=0, ls_interval=kw.pop("ls_interval", 0.25), hs=hs, **kw)
    return io, hs


def test_the_transport_is_handed_the_handshake_it_is_serving():
    """`AudioVaraIO.log` decides what to print from the turn state, and there is
    no other route to it: the transport is constructed before any handshake
    exists and the handshake logs through the transport, not the other way."""
    io, hs = _run([_RESPONSE])
    assert io.hs is hs


def test_the_link_setup_is_repeated_while_the_gateway_stays_quiet():
    """One link-setup and then silence for the rest of the window was the whole
    behaviour; a gateway that never heard the over had no way to be told again."""
    io, _ = _run([_RESPONSE])
    assert io.link_setups > 1, (
        f"the link-setup went out {io.link_setups}x in a window worth "
        f"{_LINKSETUP_MAX_TX} — the cadence after the connect-response is still dead")


def test_the_repeats_stop_at_the_cap_the_handshake_enforces():
    """The clock must not talk the state machine past its own retry limit: a
    fifteen-minute operating window is not licence to keep keying.

    The cap is one exchange's. What follows a spent one is a restart, and the
    gateway here answers a single request, so the exchange it opens draws nothing
    and keys no link-setup at all.
    """
    io, _ = _run([_RESPONSE], timeout=1.2, ls_interval=0.2)
    assert io.link_setups == _LINKSETUP_MAX_TX, (
        f"{io.link_setups} link-setups transmitted, cap is {_LINKSETUP_MAX_TX}")


@pytest.mark.parametrize("limit", [1, 6])
def test_configured_link_setup_limit_reaches_the_timer_and_handshake(limit):
    """A larger cap must buy actual transmissions; both resend routes share it."""
    io, hs = _run([_RESPONSE], max_link_setups=limit, max_cr=1,
                  timeout=3.0, ls_interval=0.08)
    assert io.link_setups == limit
    # Neither a late gateway response nor a caller bypassing the timer can
    # exceed the same per-exchange budget.
    hs.on_rx_audio(_RESPONSE)
    assert not hs.resend_link_setup()
    assert io.link_setups == limit


def test_configured_limit_allows_gateway_requested_repeats_past_three():
    io = _FakeIO([])
    hs = VaraStationHandshake(["W9SSJ"], io, max_link_setups=6)
    hs.originate(_GATEWAY)
    for _ in range(8):
        hs.on_rx_audio(_RESPONSE)
    assert io.link_setups == 6
    assert hs._linksetup_tx == 6


@pytest.mark.parametrize("limit", [0, -1, 1.5])
def test_invalid_link_setup_limit_is_rejected_before_keying(limit):
    io = _FakeIO([])
    with pytest.raises(ValueError, match="positive integer"):
        VaraStationHandshake(["W9SSJ"], io, max_link_setups=limit)
    assert not io.sent


def test_cli_link_setup_limit_reaches_the_live_connect_seam(monkeypatch):
    import sys

    seen = []

    def connect_without_devices(*args, **kwargs):
        seen.append(kwargs["hs"].max_link_setups)
        return False

    monkeypatch.setattr(kc, "connect", connect_without_devices)
    monkeypatch.setattr(sys, "argv", ["kestrel_connect", "--gateway", _GATEWAY,
                                    "--mycall", "W9SSJ", "--dry-run",
                                    "--max-link-setups", "6"])
    assert kc.main() == 1
    assert seen == [6]


def test_a_refused_key_up_spends_none_of_the_link_setup_budget():
    """The cap counts transmissions, not intentions.

    The handshake's own resend path charges the budget only when the transport says
    the burst went out; the tool's clock-driven resend is the same burst under the
    same cap and has to charge it the same way. A rig that refuses the key-up puts
    nothing on the band, and three refusals must not leave the attempt out of
    link-setups it never sent.
    """
    io, hs = _run([_RESPONSE], refuses=True, timeout=1.2, ls_interval=0.2)
    assert io.link_setups == 0, "a refused key-up still put a link-setup on the air"
    assert io.link_setups_offered > 1, "the cadence gave up after the first refusal"
    assert hs._linksetup_tx == 0, (
        f"{hs._linksetup_tx} of {_LINKSETUP_MAX_TX} link-setups charged to the budget "
        f"for {io.link_setups} that reached the transmitter")


def test_nothing_is_re_sent_while_the_connect_request_is_still_unanswered():
    """Before an answer the CR train is the cadence, and it owns the transmitter."""
    io, _ = _run([], timeout=0.6)
    assert io.link_setups == 0, "a link-setup went out before the gateway answered"
    assert io.sent, "the connect-request itself was never transmitted"


def test_a_link_setup_the_gateway_never_acks_sends_the_attempt_back_to_the_cr():
    """The failure the bench reproduced, and the answer to it.

    A gateway that answers the request and then goes silent has not refused us: it
    heard 1.77 s of MFSK and never heard the 4.36 s over behind it, which is what a
    fade does to those two bursts. Spending the link-setup budget and then listening
    out the window ends the attempt on the one burst that was still getting through.
    """
    io, hs = _run([_RESPONSE], timeout=1.2, ls_interval=0.15, cr_interval=0.15)
    assert io.link_setups == _LINKSETUP_MAX_TX
    assert hs.step == _I_CR_SENT, (
        "the link-setups went unanswered and the attempt stayed at step 4")
    assert io.connect_requests > 1, (
        "nothing was called again — the CR exchange was never re-entered")


@pytest.mark.realtime
def test_the_restart_ladder_has_a_top_and_the_attempt_ends_at_it():
    """A gateway that answers every request and acks nothing is a station nothing we
    key will reach, and this one keys a 4.36 s over per try on a shared band.

    So the ladder is bounded twice over: at most ``_CONNECT_MAX_RESTARTS`` restarts,
    each worth one exchange's link-setups, and the attempt ends there rather than
    holding the frequency for the rest of its window.
    """
    t0 = time.time()
    io, hs = _run([], answers_every_cr=True, timeout=8.0, ls_interval=0.1,
                  cr_interval=0.1)
    assert hs._cr_restarts == _CONNECT_MAX_RESTARTS
    assert io.link_setups == (1 + _CONNECT_MAX_RESTARTS) * _LINKSETUP_MAX_TX, (
        f"{io.link_setups} link-setups keyed at a gateway that acks nothing")
    assert time.time() - t0 < 8.0, (
        "the attempt sat out the rest of its window with both ladders spent")


def test_the_request_ladder_is_spread_over_the_window_it_is_given():
    """Arm 7 of 2026-08-23: eight requests inside the first 37 s of a 180 s window,
    then 143 s with nothing left to call with.

    The interval is dead time after a request, so a flat one spends the budget at a
    rate the window never enters into — and the gateway this station was calling
    answered its neighbours' arms at 40, 46 and 48 s. Raising ``--timeout`` has to
    buy calling time and not listening time.
    """
    for timeout in (60.0, 180.0):
        rung = kc.CR_BURST_S + kc.cr_cadence(timeout, 8)
        assert 8 * rung >= min(timeout, 8 * (kc.CR_BURST_S + kc._CR_INTERVAL_MAX_S)), (
            f"a {timeout:.0f} s window still spends its requests in {8 * rung:.0f} s")
    assert kc.cr_cadence(30.0, 8) < kc.cr_cadence(60.0, 8) < kc.cr_cadence(90.0, 8), (
        "a longer window did not buy a longer listen between requests")
    assert kc.cr_cadence(600.0, 8) == kc._CR_INTERVAL_MAX_S, (
        "a long window spreads past the widest spacing that has drawn an answer")


def test_the_cadence_is_never_shorter_than_the_burst_it_spaces():
    """The one interval the waveform forbids: below the burst a repeat would key
    before the previous request had left the air, and the gateway would be
    answering into our own next transmission."""
    assert _CR_LEN / FS == pytest.approx(kc.CR_BURST_S), (
        "the constant no longer describes the audio the tool keys")
    for timeout in (0.0, 1.0, 5.0, 30.0):
        assert kc.cr_cadence(timeout, 8) >= kc.CR_BURST_S


def _lattice_answer(position: int, called: str = _GATEWAY) -> np.ndarray:
    """A bare payload burst off ``called``'s own stream, with the run-out behind it
    that a stream search needs before it will score the alignment it sits on."""
    kind = replace(VF.CONNECT_RESPONSE,
                   preadv=1 + VF.lattice_step(VF.CONNECT_RESPONSE) * position)
    burst = MK.synth_tones(VF.payload_bins(called, kind))
    return np.concatenate([np.zeros(4800), burst, np.zeros(70000)])


class _AnswerStreamIO(_FakeIO):
    """A lower-speed offer arriving through the raw receive stream."""

    def __init__(self, position: int = 14):
        super().__init__([])
        self.stream = _lattice_answer(position)
        self.answered_at = 0.0
        self.keyed_at: list[float] = []

    def tx(self, samples):
        self.keyed_at.append(time.time())
        super().tx(samples)

    def next_rx_burst(self, timeout, hs=None):
        if self.stream is None or not self.sent:
            time.sleep(0.02)
            return None
        for i in range(0, len(self.stream), 4800):
            hs.on_rx_stream(self.stream[i:i + 4800])
        self.stream = None
        self.answered_at = time.time()
        return None


@pytest.mark.realtime
def test_lower_speed_offer_sends_setup_instead_of_another_request(capsys):
    io = _AnswerStreamIO(14)
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300")
    kc.connect(_GATEWAY, "W9SSJ", "2300", io, timeout=4.0, listen_first=0,
               cr_interval=30.0, hs=hs)
    assert [a.position for a in hs.answers] == [14]
    assert io.connect_requests == 1
    assert io.link_setups == 1 and hs._setup_level == 1
    assert io.keyed_at[-1] <= io.answered_at
    assert not hs.answer_retry
    out = capsys.readouterr().out
    assert "could not read" not in out and "resending CR" not in out


def test_the_verdict_distinguishes_offers_without_setup_confirmation(monkeypatch,
                                                                   capsys):
    """A silent frequency and a gateway that answered every request without being
    able to read one both end as "NOT connected", and they are different findings
    about the band."""
    def fake_connect(*a, hs=None, **kw):
        hs.answers = [PeerAnswer(t, 14, 0, 15, 15, ()) for t in (1.0, 2.0, 3.0)]
        return False

    monkeypatch.setattr(kc, "connect", fake_connect)
    monkeypatch.setattr(sys, "argv",
                        ["kestrel_connect.py", "--gateway", "K0SI", "--mycall",
                         "W9SSJ", "--dry-run", "--listen-first", "0", "--timeout",
                         "1", "--no-record"])
    assert kc.main() == 1
    out = capsys.readouterr().out
    assert "NOT connected (no/!=expected response)" in out
    assert "K0SI sent 3 confirmed BW2300 connect offers (frames 14, 14, 14)" in out
    assert "setup was not confirmed" in out
    assert "could not read" not in out


# --------------------------------------------------------------------------- #
# What the CR train does while the gate says the channel is in use. Calling K5FIT
# on 2026-09-11 it did nothing at all: `_BracketSegmenter` latched open on a step
# in the receiver's own level and handed the loop 14 brackets of empty 40 m band
# (`working/vara-evening-en63bc-0910/analysis/k5fit-burst/`), and requests 4 and 5
# waited 35.72 s against a 3.0 s cadence inside a 65 s attempt. The gate fix is in
# `test_rx_segmenter.py`; this is the bound behind it, because no floor follows a
# step instantly and a caller must not be parked by one that does not.
class _KeyedChannel(VaraIO):
    """A transport whose receive gate is open for the first ``busy_s`` seconds.

    While it is open the loop is handed a bracket on every poll and `receiving`
    reads true, which is exactly what a peer's over and a latched gate look like
    from inside `connect()`. Nothing ever answers: what is under test is what the
    caller does while it is being told the channel is in use.
    """

    progress = kc.ProgressWatch(None)   # the side channel, writing nowhere

    def __init__(self, busy_s: float, burst_s: float = 0.235):
        self.busy_s = busy_s
        self.burst = np.random.default_rng(0).standard_normal(int(burst_s * FS)) * 1e-3
        self.rig = None
        self.t0 = time.time()
        self.keyed: list[tuple[float, int]] = []

    def start(self): ...
    def pending(self): ...
    def connected(self, *a): ...
    def log(self, msg): ...
    def key(self, on): ...

    def tx(self, samples):
        self.keyed.append((time.time() - self.t0, len(samples)))

    @property
    def receiving(self) -> bool:
        return time.time() - self.t0 < self.busy_s

    def next_rx_burst(self, timeout, hs=None):
        time.sleep(0.02)
        return self.burst if self.receiving else None

    @property
    def requests(self) -> list[float]:
        return [t for t, n in self.keyed if n == _CR_LEN]


def _called(busy_s, monkeypatch, defer_max, timeout=2.0, cr_interval=0.15):
    monkeypatch.setattr(kc, "KA_DEFER_MAX_S", defer_max)
    io = _KeyedChannel(busy_s)
    kc.connect(_GATEWAY, "W9SSJ", "2300", io, timeout=timeout, listen_first=0,
               cr_interval=cr_interval, max_cr=10,
               hs=VaraStationHandshake(["W9SSJ"], io, bw="2300"))
    return io


@pytest.mark.realtime
def test_a_gate_that_never_closes_does_not_park_the_call(monkeypatch):
    """The K5FIT pause. A gate that hands over a bracket every poll and never
    reads clear used to defer the CR train for as long as it went on — there was
    no bound on this side at all, where the keepalive cadence has had one all
    along. One bracket's worth past a request's due time the channel is the band,
    and the request goes out."""
    io = _called(99.0, monkeypatch, defer_max=0.3)
    assert len(io.requests) > 1, (
        "the call was parked for the whole window by a gate that never closed; "
        f"{len(io.requests)} request(s) reached the transmitter")
    waited = io.requests[1] - io.requests[0]
    assert 0.15 + 0.3 <= waited < 0.15 + 0.3 + 0.4, (
        f"the second request came {waited:.2f} s after the first, against a "
        f"0.15 s cadence deferred by a 0.30 s bound")


@pytest.mark.realtime
def test_a_peer_transmission_shorter_than_the_bound_still_defers_the_call(monkeypatch):
    """And the half that must survive: the bound is not permission to key over a
    peer. Six requests come due inside this reception and none of them goes out,
    because the channel is keyed and the bound has not run. The caller picks the
    cadence up the moment the channel is clear."""
    io = _called(0.7, monkeypatch, defer_max=2.0, cr_interval=0.1, timeout=1.6)
    during = [t for t in io.requests[1:] if t < 0.7]     # [0] is the call itself
    assert not during, (
        f"{len(during)} request(s) keyed at {[round(t, 2) for t in during]} while "
        f"the peer was transmitting, inside a {2.0:g} s deferral bound")
    assert len(io.requests) > 1, (
        "the cadence never came back after the channel cleared")
