# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 2026-08-05 KE8LVA session: a real Winlink greeting, read six seconds late.

W9SSJ called KE8LVA on 7103.5 kHz at 23:57Z, connected at BW2000, took the IRS
role on the gateway's BREAK, and was sent the head of an `RMS Trimode 1.4.2.0`
greeting — the same 4PSK.500.100.O frame, over and over, every 6.087 s for two
minutes. The session read exactly one of them and answered it **2.24 s** after its
last sample, keying on top of a transmission that was already running; the
greeting never completed and no mail moved. That figure is measured off the whole
recording by `test_how_late_the_answer_went_out` below — the 2.9 s this file used
to quote is not a figure anything in the recording supports.

The reason it was read late is that acquisition triggers on energy climbing out
of silence, and this band never went quiet: its noise floor alone sat above the
2% silence gate, so each rolling window offered exactly one onset — the window's
own first sample — and a window whose first sample was not the frame's leader
found nothing at all. The frame surfaced only in the last window that could still
hold it whole, 1.83 s after it ended, against a 1.67 s gap to the gateway's next
transmission. `Demodulator._next_leader` is the repair: where energy gives no
edge, the walk continues on the two-tone leader's own scale-free signature.

The fixture is the *first* rolling window that holds the frame whole — 6.25 s of
that recording, ending 0.23 s after the frame's last sample, exactly what
`RollingDecoder` had in hand at that moment. The whole 178.7 s session is in the
corpus at ``offair/ke8lva_greeting_20260805/rig_rx.wav``.

**What the repair is worth, measured rather than counted.** It took the
whole-capture pass over that session from 8 frames to 29, and 4 ok to 21 — a
figure this file used to quote as if it were the live improvement, which it is
not. Seventeen of the ok frames read 19.4 to 22.2 dB *under* the band floor, in
the stretches where this station's own rig was keyed: five ``ConReq2000M``
decoding ``caller='W9SSJ'``, then our ConAcks, IDLEs, DATAACKs. They are our own
transmissions bleeding into a muted receiver, and the audio they sit in is what
`RadioAudio._capture` drops by design — no live session can ever hear them.

The gateway sent its greeting over and over and this pass reaches four of the
arrivals; **one** of them is read. The other three were readable only out of the
first one's validated carrier put back whole, which `BlockMemory` stopped keeping
once a block has come out complete — the same putting-back spliced one W6IDS
proposal block into the place of another on 2026-08-29 and cost a Winlink
message, and the derivation is at the foot of `test_memory_arq.py`. They are the
same bytes as the first either way, and the session delivered them once.

Through the live path — `RollingDecoder` fed 0.1 s blocks with those keyed
stretches dropped — both walks find the same five ok frames, at the same
positions, the greeting among them, reported in the same window. **The live-path
improvement on this recording is zero**, and the one real frame the repair gains
is the greeting on the whole-capture pass, where a capture-length silence gate
buries it and a 6 s window does not.

So the shipped fixture does not demonstrate the repair: it decodes the greeting
under either walk, which is what makes it a fair fixture and not a demonstration.
The two cases the old walk genuinely fails are `test_how_late_the_answer_went_out`
below, which finds no greeting at all in the whole session under it, and
`test_a_louder_burst_after_the_greeting_does_not_hide_it`, which loses the
greeting the moment anything louder follows it. The derivation is
``working/besra-audit/acquisition_walk_two_detectors.py``.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.besra.dsp import detect
from hfmodem.besra.monitor import _render
from hfmodem.besra.phy.demodulator import (SAMPLE_RATE, DecodedFrame, Demodulator,
                                           _HEADER_LEN, _SYM_50, _body_span)
from hfmodem.core.resample import from_card
from hfmodem.winlink.session import B2FSession

_FIXTURE = Path(__file__).parent / "fixtures" / "offair_ke8lva_greeting.wav"
_CORPUS = evidence.CORPUS
_SESSION = _CORPUS / "offair" / "ke8lva_greeting_20260805" / "rig_rx.wav"

#: What the gateway actually sent, recovered byte-exact and corroborated against
#: the reference ardopcf's own (RS-uncorrectable) reads of the other repeats.
_GREETING = (b"RMS Trimode 1.4.2.0 KE8LVA Winlink Gateway\r\n"
             b"W9SSJ has 1437 daily minutes remaining with KE8LVA (EN80XS)\r"
             b"[WL2K-5.0-B2FWIHJM$]\r;PQ")


def _audio() -> np.ndarray:
    with wave.open(str(_FIXTURE)) as w:
        assert w.getframerate() == 12000
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def test_the_greeting_is_read_from_the_first_window_that_holds_it():
    """That the payload survives this window at all — not that the repair is what
    recovers it. This window decodes under the pre-repair walk too; see the file
    docstring for which tests do not."""
    frames = Demodulator().decode(_audio())
    hit = [f for f in frames if f.ok and f.name == "4PSK.500.100.O"]
    assert len(hit) == 1, f"expected the greeting, got {[(f.name, f.ok) for f in frames]}"
    assert bytes(hit[0].payload) == _GREETING


def test_a_busy_channel_still_offers_only_one_energy_onset():
    """The defect itself, so the repair cannot be quietly undone: on this audio the
    energy walk has one shot per window and it is the window's first sample."""
    au = _audio().astype(np.float64)
    energy = detect.boxcar(au * au, _SYM_50)
    peak = float(energy.max())
    assert Demodulator._next_onset(energy, peak, 0) == 0
    assert Demodulator._next_onset(energy, peak, 1) == -1


def test_a_louder_burst_after_the_greeting_does_not_hide_it():
    """The other half of the same defect, and the reason the leader walk cannot be
    a last resort.

    The silence gate is 2% of the *window's* peak, so anything loud raises it over
    everything quieter: append 0.2 s of noise ten times the recording's own level
    and the greeting stops clearing the gate altogether. The energy walk still
    "succeeds" — on the noise — and a leader walk consulted only when energy found
    nothing is never asked. Both detectors have to be asked; the earlier candidate
    is tried first.
    """
    au = _audio().astype(np.float64)
    rms = float(np.sqrt((au * au).mean()))
    rng = np.random.default_rng(7)

    for scale in (1, 10):
        loud = np.concatenate([au, rng.normal(0.0, rms * scale, int(0.2 * SAMPLE_RATE))])
        hit = [f for f in Demodulator().decode(loud)
               if f.ok and f.name == "4PSK.500.100.O"]
        assert len(hit) == 1, f"greeting lost under noise x{scale}"
        assert bytes(hit[0].payload) == _GREETING

    # The gate really is what moves: at x10 the greeting's own energy no longer
    # reaches it, so the one place the energy walk points is past the frame.
    loud = np.concatenate([au, rng.normal(0.0, rms * 10, int(0.2 * SAMPLE_RATE))])
    energy = detect.boxcar(loud * loud, _SYM_50)
    frame = Demodulator().decode(au)[0]
    onset = Demodulator._next_onset(energy, float(energy.max()), 0)
    assert onset > frame.offset + _HEADER_LEN + _body_span(frame.type)


def _session_audio() -> np.ndarray:
    """The whole 178.7 s session at 12 kHz, from the corpus."""
    if not _SESSION.exists():
        pytest.skip(f"{_SESSION} absent")
    with wave.open(str(_SESSION)) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1
        card = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return from_card(card.astype(np.float32) / 32768.0, SAMPLE_RATE).astype(np.float64)


def _keyed_intervals(au: np.ndarray) -> list[tuple[float, float]]:
    """When this station was transmitting, in seconds. A keyed rig mutes its own
    receiver, so a stretch two orders under the band's own floor is ours and
    nothing else — the recording has seventeen of them, all 0.41-0.46 s."""
    n = SAMPLE_RATE // 100                               # 10 ms
    rms = np.sqrt(detect.boxcar(au * au, n)[::n] / n)
    quiet = rms < 0.05 * np.median(rms)
    edge = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    return [(a * n / SAMPLE_RATE, b * n / SAMPLE_RATE)
            for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))
            if (b - a) * n / SAMPLE_RATE >= 0.2]


def test_how_late_the_answer_went_out():
    """The derivation behind this file's headline figure, re-runnable.

    Both quantities are read off the recording itself. The greeting is the first
    ``4PSK.500.100.O`` the whole-capture pass recovers, and its last sample is its
    decoded position plus its own frame length. Our reply is visible because a
    keyed rig mutes its own receiver: the capture drops two orders of magnitude
    under the band floor from key-down, and the next such interval after the
    greeting is the ACK it provoked (the one after that, 49 s later, is the
    session's own teardown).

    That gives **2.24 s** from the greeting's last sample to key-down. It does not
    give 2.9 s, which is the number this file carried and which nothing in the
    recording produces.

    The gateway sent that greeting over and over and this pass reaches four of
    the arrivals; the one it reads is the first, which is the only one a live
    session would have reached before it answered anyway. The other three were
    read off memory ARQ until `BlockMemory` stopped keeping what a completed block
    leaves — see this file's header.
    """
    au = _session_audio()
    greeting = [f for f in Demodulator().decode(au)
                if f.ok and f.name == "4PSK.500.100.O"]
    assert len(greeting) == 1, f"expected the one read greeting, got {len(greeting)}"
    ends = (greeting[0].offset + _HEADER_LEN + _body_span(greeting[0].type)) / SAMPLE_RATE
    assert ends == pytest.approx(126.27, abs=0.01)

    keyed = _keyed_intervals(au)
    assert len(keyed) == 17 and all(0.4 < b - a < 0.5 for a, b in keyed), \
        "these are not our own transmissions"
    ack = next(k for k, _ in keyed if k > ends)
    assert ack - ends == pytest.approx(2.24, abs=0.02)


def test_the_whole_capture_count_is_mostly_our_own_transmissions():
    """Why "8 frames to 29" is not a live figure, pinned so it cannot come back.

    The whole-capture pass finds 21 ok frames where the pre-repair walk found 4.
    Seventeen of them are this station's own — a keyed rig mutes its own receiver,
    but not to nothing, and what leaks through sits about 20 dB under the band we
    were hearing a moment before. `RadioAudio._capture` never pushes that audio at
    the decoder, so those seventeen are unreachable on the live path by
    construction.

    The split is measured, not assumed: every ok frame in this recording is either
    within 2.2 dB of the band floor or 19.4 to 22.2 dB under it, nothing between,
    and every one on the low side that carries a callsign carries *ours*.
    """
    au = _session_audio()
    n = SAMPLE_RATE // 100
    rms = np.sqrt(detect.boxcar(au * au, n)[::n] / n)
    floor = float(np.median(rms))

    ours, on_air = [], []
    for f in Demodulator().decode(au):
        if not f.ok:
            continue
        at = f.offset / SAMPLE_RATE
        db = 20 * np.log10(rms[int(at * 100):int(at * 100) + 30].mean() / floor)
        assert not -19.0 < db < -3.0, f"{f.name} at {at:.2f}s reads {db:+.1f} dB — no split"
        (ours if db < -19.0 else on_air).append(f)

    assert len(ours) == 17 and len(on_air) == 4
    assert {f.caller for f in ours if f.caller} == {"W9SSJ"}
    assert {f.caller for f in on_air if f.caller} == {"KE8LVA"}
    # One of the four on the gateway's side is its greeting, sent over and over
    # and read on its own arrival. The three repeats this pass reaches are read
    # by nothing now: see the header, and `test_memory_arq.py`.
    assert [f.name for f in on_air].count("4PSK.500.100.O") == 1


def test_the_answer_landed_on_a_gateway_already_transmitting():
    """Why 2.24 s was too late rather than merely slow. Between its frames this
    band reads about half the level the gateway's own carrier does; in the third
    of a second before we keyed it is back at carrier level, so the gateway had
    started its next retransmission and could not hear us."""
    au = _session_audio()
    n = SAMPLE_RATE // 100
    rms = np.sqrt(detect.boxcar(au * au, n)[::n] / n)

    def level(a: float, b: float) -> float:
        return float(np.median(rms[int(a * 100):int(b * 100)]))

    key = [k for k, _ in _keyed_intervals(au) if k > 126.27][0]
    carrier = level(122.5, 126.0)                 # the greeting itself
    between = level(126.5, 127.5)                 # the band between transmissions
    assert between < 0.75 * carrier, "no headroom to tell a carrier from the band"
    assert level(key - 0.35, key - 0.05) > 0.9 * carrier


def test_one_frame_is_not_a_greeting():
    """Why the run reported `awaiting greeting` with the remote SID already
    printed: the gateway's greeting does not fit in one 128-byte frame. The SID
    line arrived whole, the `;PQ:` challenge was cut mid-line and the prompt that
    ends a B2F greeting never came, so the session was correctly still waiting —
    and had nothing to say until it did."""
    s = B2FSession("W9SSJ", role="calling", target="KE8LVA")
    out = s.feed(_GREETING)
    assert s.remote_sid == "[WL2K-5.0-B2FWIHJM$]"
    assert s.stage == "awaiting greeting"
    assert out == b""


def test_a_frame_whose_body_is_unreadable_is_still_named_from_its_header():
    """The 2026-08-09 session, in miniature: what happens when the same gateway's
    frame arrives with a header this receiver can read and a body it cannot.

    On 2026-08-09 KE8LVA sent this same greeting thirteen times and none of it
    survived the channel — the reference ardopcf, given the same recording, read
    `4PSK.500.100.O` thirteen times and failed RS on every one, so the body was
    genuinely gone. besra failed them too, but it also *named them wrong*: every
    one came back `4PSK.200.100.E` with a session id that was not the session's,
    which is a frame nobody sent.

    The header is where the naming comes from, and this gateway does not put it
    where the default 240 ms leader would: it sits 2018 samples past the leader
    edge here, and 1985-2058 across the thirteen bursts of that session. Damage
    the body of this fixture and the frame must still come back as the
    `4PSK.500.100.O` for session 0x51 that it is, `ok` False, carrying a quality
    the ARQ layer can NAK with.
    """
    au = _audio().astype(np.float64)
    good = Demodulator().decode(au)[0]
    assert (good.name, good.session_id, good.ok) == ("4PSK.500.100.O", 0x51, True)

    body = good.offset + _HEADER_LEN
    span = _body_span(good.type)
    assert good.offset - 20160 == 2018, "the fixture's leader is no longer the short one"

    rng = np.random.default_rng(20260809)
    level = float(np.sqrt((au[body:body + span] ** 2).mean()))
    for scale in (0.5, 1.0, 2.0):
        hurt = au.copy()
        hurt[body:body + span] += rng.normal(0.0, scale * level, span)
        got = Demodulator(expect_session=lambda: 0x51).decode(hurt)
        assert len(got) == 1, f"x{scale}: {[(f.name, f.ok) for f in got]}"
        assert (got[0].name, got[0].session_id) == ("4PSK.500.100.O", 0x51), \
            f"x{scale}: named {got[0].name} for session 0x{got[0].session_id:02X}"
        assert not got[0].ok and got[0].quality is not None
        # Named off its header and nothing else, and the frame says so — see
        # `test_a_clipped_leader_does_not_mint_a_frame_type_nobody_sent`.
        assert got[0].header_only


#: How much of its leader an ARQ reply loses to the receiving station's own post-TX
#: recovery. 80 ms was measured on KE8LVA's ConAck2000 answers; 160 takes the front
#: off this fixture's ~168 ms leader as well, leaving the header where the
#: full-leader geometry does not look for it.
_RECOVERY_CLIP_MS = 160

#: What a session-blind receiver makes of the clipped frame as its body is buried.
#: At half the body's own level the payload still validates and the frame is a
#: decode; above that there is nothing behind the header, and the honest report is
#: silence rather than a guess.
#:
#: The 0.5 row used to be silence too, and what turned it over is not the receiver:
#: the damage below is anchored on the frame's own offset, `detect._CFO_WINDOW`
#: moved that offset five samples, and the damage moved with it. Fed the same array
#: both windows read the same thing at every scale.
_BLIND = {0.5: [("4PSK.500.100.O", True)], 1.0: [], 2.0: [], 3.0: []}


def test_a_clipped_leader_does_not_mint_a_frame_type_nobody_sent():
    """A repeat read as a different frame type — the phantom, on shipped audio.

    A wrong tone grid does not look wrong to the header scan. `crisp` measures how
    concentrated ten symbols are in one of four bins, so a narrowband signal sitting
    near a *shifted* bin scores as crisply as a 4FSK symbol on the right one: this
    frame's real header, read on a grid four bins out, spelled types nobody sent —
    0x40 on session 0x40, 0x42, 0x4C across the sweep below — and cleared the 8.0
    floor `_crisp_floor` grants a data frame on the promise of an RS and a payload
    CRC behind it. In the fallback path that promise has already been broken.

    The grid the phantom was read on did not come from the offset estimate on this
    clip, whatever the guard here used to say: it asserted the estimate was 40+ Hz
    out over `au[clip:clip + 3120]`, which is 260 ms of the recording's opening
    noise and not the frame at 22178, so it passed on noise and proved nothing. At
    the onset the walk actually anchors on, the estimate reads +7 Hz — under the
    200 ms window this used to take and under `detect._CFO_WINDOW` alike. What the
    sweep proves is the corroboration.

    The same shape was measured live on 2026-08-15 (W4RJG, 7101.9) as
    `16QAM.500.100.O` between our DATANAKs, and read for a day as a gateway gearing
    two rungs *up* in a session whose mode table has no such frame. Replayed through
    `RollingDecoder`, the 2026-08-05 session produces three of them — sessions 0x00,
    0x05 and 0x11 — out of this gateway's own `4PSK.500.100.O` repeats, each one in
    place of the real frame at that position. So on-air sightings of it are not
    evidence of anyone else's traffic until this end can be ruled out.

    Two things have to hold, and they are different things. With no session in
    progress there is nothing behind the header at all, and the honest report is
    silence rather than a guess. With a session in progress the session id is eight
    bits of corroboration the tone grid cannot fake, so the frame comes back as what
    it is — and marked `header_only`, because its type was still read off ten tones
    and nothing else.
    """
    au = _audio().astype(np.float64)
    good = Demodulator().decode(au)[0]
    body = good.offset + _HEADER_LEN
    span = _body_span(good.type)
    clip = _RECOVERY_CLIP_MS * SAMPLE_RATE // 1000

    for scale in (0.5, 1.0, 2.0, 3.0):
        rng = np.random.default_rng(20260815)
        hurt = au.copy()
        level = float(np.sqrt((hurt[body:body + span] ** 2).mean()))
        hurt[body:body + span] += rng.normal(0.0, scale * level, span)
        hurt = hurt[clip:]

        blind = Demodulator().decode(hurt)
        assert [(f.name, f.ok) for f in blind] == _BLIND[scale], f"x{scale}: {blind}"

        got = Demodulator(expect_session=lambda: 0x51).decode(hurt)
        assert [(f.name, f.session_id, f.ok, f.header_only) for f in got] == \
            [("4PSK.500.100.O", 0x51, False, True)], f"x{scale}: {got}"


def test_the_log_does_not_render_an_unconfirmed_type_as_a_decode():
    """A phantom that announces itself is harmless; one that reads like a sighting
    is not. `ok=False` alone cannot carry the difference — it is also what a frame
    says when this receiver accepted it and the channel destroyed its body."""
    frames = [DecodedFrame(type=0x51, session_id=0x51, ok=False, name="4PSK.500.100.O"),
              DecodedFrame(type=0x51, session_id=0x51, ok=False, name="4PSK.500.100.O",
                           header_only=True)]
    decoded, guessed = (_render(f) for f in frames)
    assert "CRC/RS FAIL" in decoded and "HEADER ONLY" not in decoded
    assert "HEADER ONLY" in guessed
