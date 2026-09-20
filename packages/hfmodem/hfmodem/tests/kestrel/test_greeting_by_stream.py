# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The gateway's DATA overs, found without an energy gate to hand them over.

``test_response_by_stream`` settled this for the connect-response. The same fact
holds one step later and cost this project its first held VARA link: on
2026-08-06 W9SSJ connected to KC9GHZ, the gateway sent its Winlink greeting
0.257 s after our session-confirm, and the mail client sat in "awaiting greeting"
for 227 s while that over sat unread in the receive stream.

Measured on that recording. The over is 4.21 s of BW2300 rec3 at 24 of 24
reference columns with a clean CRC-16, and our receiver mute lifted 0.011 s
before it began — we were awake for every sample of it. It is 4.1 dB over the
band noise, and the level is not what an energy gate could have been made to see:
give the tracker the best floor it could possibly have had — a q25 over the band
audio that FOLLOWS the over, which no live tracker has yet — and the loudest of
the over's 198 frames reaches 1.83-1.99x it across nine window lengths from 5 s
to 90 s, never the 2.0x a bracket opens on. Handed the same audio directly, the
receive chain decodes and answers it perfectly at every bracket boundary tried,
so nothing downstream of the gate was ever wrong.

So an over is looked for the way the connect-response is: over the raw receive
stream, by the structure it is bound to carry. The reference columns of a base
BW2300 frame are fixed by column class rather than by payload, which makes "is
there a wideband frame in this audio" answerable without decoding anything and
without reference to how loud it is.

The second half is the other greeting, from the same gateway on 2026-08-19, and
the other half of the same lesson: finding the over is not reading it. That one
was found — 18 of 24 reference columns — and would not decode, because a fade
across the band, not the level of the burst, was moving the argmax.
"""
from __future__ import annotations

import numpy as np

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.rx import varahf2300 as RX
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_mfsk as MK

FS = MK.FS
MYCALL, GATEWAY = "W9SSJ", "KC9GHZ"

# The greeting over's extent in the recording, off the band-limited envelope: hard
# edges 0.011 s after our own receiver mute lifts and 4.224 s later.
_GREETING = (58.083, 62.307)

# What the gateway said, byte-exact off the air — the first block of an RMS
# Trimode SID banner, as `vara_payload` delivers it to the host.
GREETING = (b"RMS Trimode 1.4.2.0\r\n"
            b"W9SSJ has 120 daily minutes remaining with KC9GHZ (EN62BK)\r{SFI = 10")

# The same gateway on 2026-08-19: one base BW2300 over starting at sample 3254580
# of that recording, carrying the first block of a fuller RMS Trimode SID banner.
# `body[89]` is 0x51 there and 0x89 in the two overs below that end mid-sentence,
# which is what a per-frame field looks like and not a banner's next character
# [see arq.phy.vara_body].
_FADED_START = 3254580
FADED_GREETING = (b"RMS Trimode 1.4.2.0 Wadsworth IL - NVIS inverted V at 25 feet "
                  b"with 50 watts\r\nW9SSJ has 12")

# The same banner two days earlier, whole, over the three overs of the 2026-08-16
# session — the one session in the logs kept here whose overs were read as they
# arrived. Its first over carries the payload 2026-08-19 died on.
_OVERS_START = (1987362, 2298340, 2608992)
OVERS_GREETING = (
    FADED_GREETING,
    b"0 daily minutes remaining with KC9GHZ (EN62BK)\r{SFI = 129 On 2026-08-17 "
    b"00:00 UTC}\r\n[WL2K",
    b"-5.0-B2FWIHJM$]\r;PQ: 99623086\rCMS via KC9GHZ >\r",
)

# Where the replay starts: inside the tail of our own session-confirm mute, which
# is where the live transport resumes after keying (`AudioVaraIO.tx` advances its
# cursor to the end of the transmission plus a codec tail). Feeding from here
# reproduces what the stream search is actually handed.
_RESUME = 58.0

#: Device-sized blocks, which is the granularity anything keyed here can be timed
#: to: a burst goes out on the block that completed the frame, so the instant is
#: that block's END and never its start.
_FEED = 0.1


class _IO(VA.VaraIO):
    """Records what was keyed, and when in the replay. Nothing here opens a
    device or reaches a radio."""

    def __init__(self):
        self.clock = 0.0            # end of the block just fed, into the recording
        self.keyed: list[float] = []
        self.msgs: list[str] = []
        self.logged: list[tuple[float, str]] = []   # when in the replay each line fell
        self.payloads: list[bytes] = []

    def key(self, on):
        if on:
            self.keyed.append(self.clock)

    def tx(self, samples): ...

    def pending(self): ...

    def connected(self, *a): ...

    def data(self, payload): self.payloads.append(bytes(payload))

    def log(self, msg):
        self.msgs.append(msg)
        self.logged.append((self.clock, msg))


def _connected():
    """The station exactly as it stood at 57.1 s: linked, initiator, the turn with
    the peer and nothing queued — a mail client waiting to be spoken to."""
    io = _IO()
    hs = VA.VaraStationHandshake([MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", GATEWAY, MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def _replay(hs, io, x, start=0.0, block=_FEED):
    """Feed a recording to the stream search in device-sized blocks."""
    n = int(block * FS)
    for i in range(int(start * FS), len(x) - n, n):
        io.clock = (i + n) / FS
        hs.on_rx_stream(x[i:i + n])


#: Most bursts a stalled gateway can draw out of this station, whatever the replay
#: runs for. `mail_session` ticks the give-up budget every 10 s and nothing the
#: peer's cadence draws defers that tick any more [see vara_arq._answer_peer_idle],
#: so a peer that never makes progress closes the link after
#: `_MAX_WITHOUT_PROGRESS` of them; the fastest responder idle cadence on tape is
#: 3.155 s [see vara_arq._peer_responder_idle]. One answer to the over, and one
#: burst per gap inside that window.
_STALL_CAP = 1 + int(VA._MAX_WITHOUT_PROGRESS * 10.0 / 3.155)


def _occupancy(io):
    """Assert every keying sat in a gap the peer opened, and that a stall cannot
    draw an unbounded number of them.

    PLACEMENT is the real claim: the answer to an idle goes out on the block that
    named the idle, so each keying past the first shares its instant with a naming
    and nothing is keyed anywhere else. A count alone cannot say that — it would
    pass a build that keyed the same total into the middle of the peer's overs.
    """
    named = {at for at, m in io.logged if "session-responder-idle" in m}
    stray = [k for k in io.keyed[1:] if k not in named]
    assert not stray, (
        f"keyed at {stray} with no idle named there — outside the peer's gaps",
        io.msgs)
    assert len(io.keyed) <= _STALL_CAP, (
        f"keyed {len(io.keyed)}x against a ceiling of {_STALL_CAP} for a peer "
        f"that never made progress", io.msgs)


@corpora.requires_onair_gateway_greeting
def test_the_gateways_greeting_is_read_off_the_receive_stream():
    """The whole point: 168 s of a held session replayed as the receiver
    delivered it, and the greeting comes out of it."""
    x = corpora.wav_mono(corpora.ONAIR_GATEWAY_GREETING) / 32768.0
    hs, io = _connected()
    _replay(hs, io, x, start=_RESUME)

    assert io.payloads == [GREETING], (
        f"the gateway's greeting is at {_GREETING[0]}-{_GREETING[1]} s of this "
        f"recording and {len(io.payloads)} payload(s) came out", io.msgs)


@corpora.requires_onair_gateway_greeting
def test_the_greeting_is_answered_once_and_the_rest_of_the_session_is_silent():
    """An over that is read has to be answered, or the gateway stops — and the
    164 s behind it must not key anything at all.

    Two of those seconds are a third station calling two gateways in turn
    (`corpora.ONAIR_STRANGER_REQUESTS`), which is the harder half of the claim:
    somebody else's handshake on our frequency is real VARA, and answering it
    would key at a station we are not in session with.

    One keying, and it is the per-over response that leaves the gateway sending
    [spec 05 §5.3.3]: the turn was the peer's and nothing was queued here, so
    claiming it would be the turn-idle mistake that killed the earlier sessions.

    That answer is the 11-symbol two-tone control burst. Two stock VARA HF 4.9.0
    instances over a fake cable on 2026-08-26, one recording per direction,
    answered every DATA over of three sessions with it and with nothing else —
    thirteen occurrences, both directions — and neither keyed a 32-symbol frame
    anywhere in any of the three, which is what `session-over-response` would
    have been.
    """
    x = corpora.wav_mono(corpora.ONAIR_GATEWAY_GREETING) / 32768.0
    hs, io = _connected()
    _replay(hs, io, x, start=_RESUME)

    answers = [m for m in io.msgs if "tx per-over response" in m]
    assert len(answers) == 1, (
        f"{len(answers)} answers to one over, replaying 164 s of 40 m and a "
        f"stranger's two connect-requests", io.msgs)
    # ONE BURST PER GAP THE PEER OPENED, and not one anywhere else — including
    # none into the stranger's two connect-requests  [see _occupancy].
    _occupancy(io)
    # THE REST OF THE SESSION IS NOT SILENCE, which is what this used to assert.
    # The gateway keys `session-responder-idle` fourteen times, 3.4-3.9 s apart,
    # after our answer to a FULL over — the over that says another follows it —
    # and never sends the one behind it. That is the stall the 2026-09-08 KE8LVA
    # mailbox ended in, on a recording three weeks older, and every keying past
    # the first here is a rung of the ladder that answers it, each into a
    # turnaround this build named  [see `VaraStationHandshake._reack`].
    assert any("over #1 unacknowledged after" in m for m in io.msgs), (
        "the ladder never finished on a peer that answered none of it", io.msgs)
    assert hs.turn == VA._TURN_PEER, "took a turn nobody offered"
    # Answered while the gateway is still listening for it, and on the very block
    # that completed the over: the over route's scan step is finer than a device
    # block, so the frame is taken by the first scan that can hold it whole and
    # nothing of the turnaround is spent on grid.
    assert _GREETING[1] <= io.keyed[0] <= _GREETING[1] + _FEED


@corpora.requires_onair_faded_greeting
def test_a_greeting_arriving_through_a_selective_fade_is_read_too():
    """The 2026-08-19 session, replayed whole: 127 s that hold our own eight
    connect-requests, our own link-setup back through the rig monitor at 61.3 s,
    the gateway's greeting at 67.8 s and 55 s of the gateway going on transmitting
    after we stopped.

    Live, this over scored 18 of the 24 reference columns and would not decode,
    and the session ended there. What was wrong with it is in the next test.

    Those 55 s are the gateway's own idle cadence, eleven keyings of
    `session-responder-idle` behind a FULL over it never sent the rest of — the
    third recording to hold that shape — so one answer goes out and the
    re-acknowledgement ladder spends its four rungs into it  [see
    `VaraStationHandshake._reack`].
    """
    x = corpora.wav_mono(corpora.ONAIR_FADED_GREETING) / 32768.0
    hs, io = _connected()
    _replay(hs, io, x)

    assert io.payloads == [FADED_GREETING], (
        f"{len(io.payloads)} payload(s) came out of the session", io.msgs)
    answers = [m for m in io.msgs if "tx per-over response" in m]
    assert len(answers) == 1, (f"{len(answers)} answers to one over", io.msgs)
    # One answer, then one burst into each gap the gateway's own cadence opened:
    # four ladder rungs and then the idle answer  [see _occupancy].
    _occupancy(io)
    assert any("link-setup naming " + MYCALL in m for m in io.msgs), (
        "our own link-setup at 61.3 s was not recognised as ours", io.msgs)


@corpora.requires_onair_faded_greeting
def test_the_fade_and_not_the_level_is_what_stopped_that_greeting_decoding():
    """Why it failed, as the two numbers that differ.

    A base over lights one bin of sixteen per column, so the median down a bin is
    what that bin reads while unlit. Here those medians span 2.8% to 12.0% of the
    frame's peak — a fade with four bins around 1.3 kHz standing 2-4x over the rest
    of the band for the whole 4.21 s — and a bare argmax follows it: all six of the
    reference columns that miss miss *into* those four bins. Equalising by the
    medians is payload-blind, costs one division, and is the whole of the repair.
    """
    x = corpora.wav_mono(corpora.ONAIR_FADED_GREETING) / 32768.0
    r = RX.RECORDS[RX.BASE_LEVEL]
    frame = x[_FADED_START:_FADED_START + RX._BASE_NCOLS * r.dw50]
    raw = np.abs(np.fft.rfft(frame.reshape(RX._BASE_NCOLS, r.dw50), axis=1))[
        :, r.first_bin:r.first_bin + r.span]
    med = np.median(raw, axis=0) / raw.max()
    assert med.max() / med.min() > 3.0, (
        f"the band reads flat now ({med.min():.3f}-{med.max():.3f} of peak) — this "
        f"is no longer the recording the fade was measured on")

    def read(mag):
        bins = mag.argmax(1) + r.first_bin
        hits = int(sum(bins[c] == b for c, b in zip(RX._REF_COL, RX._REF_BIN)))
        onair = RX._onair_from_bins(bins, RX.BASE_LEVEL)
        return hits, RX.check_frame(RX.onair_to_frame(onair, RX.BASE_LEVEL),
                                    RX.BASE_LEVEL)

    hits, fr = read(raw)
    assert (hits, fr.crc_ok) == (18, False), f"unequalised: {hits}/24, crc {fr.crc_ok}"
    hits, fr = read(RX._band_mag(frame, RX.BASE_LEVEL, RX._BASE_NCOLS))
    assert hits >= VA._OVER_GUARD_MIN and fr.crc_ok, f"equalised: {hits}/24"
    assert bytes(fr.payload).startswith(FADED_GREETING)


@corpora.requires_onair_gateway_overs
def test_the_overs_that_already_decoded_still_do_and_score_higher():
    """The control the fade is read against: the 2026-08-16 session, whose three
    overs were read as they arrived at 22, 23 and 23 of 24 reference columns.

    Its band is flat — per-bin medians within 1.7x, against 3.9x on 2026-08-19 —
    which is the whole of the difference between a frame that decoded and a frame
    that did not, since the peer, the channel, the waveform and this receiver are
    the same on both nights. Equalising a flat band changes little and takes
    nothing away: every one of the three still decodes, and each scores at least as
    well as it did unequalised.
    """
    x = corpora.wav_mono(corpora.ONAIR_GATEWAY_OVERS) / 32768.0
    r = RX.RECORDS[RX.BASE_LEVEL]
    for at, said in zip(_OVERS_START, OVERS_GREETING):
        frame = x[at:at + RX._BASE_NCOLS * r.dw50]
        raw = np.abs(np.fft.rfft(frame.reshape(RX._BASE_NCOLS, r.dw50), axis=1))[
            :, r.first_bin:r.first_bin + r.span]
        med = np.median(raw, axis=0)
        assert med.max() / med.min() < 2.0, (
            f"the over at {at / FS:.3f} s now reads {med.max() / med.min():.2f}x "
            f"across the band — it is no longer the flat-band control")

        def score(mag):
            bins = mag.argmax(1) + r.first_bin
            hits = sum(bins[c] == b for c, b in zip(RX._REF_COL, RX._REF_BIN))
            onair = RX._onair_from_bins(bins, RX.BASE_LEVEL)
            return int(hits), RX.check_frame(
                RX.onair_to_frame(onair, RX.BASE_LEVEL), RX.BASE_LEVEL)

        was, _ = score(raw)
        hits, fr = score(RX._band_mag(frame, RX.BASE_LEVEL, RX._BASE_NCOLS))
        assert fr.crc_ok and hits >= was, (
            f"the over at {at / FS:.3f} s scored {was}/24 unequalised and "
            f"{hits}/24 equalised, crc {fr.crc_ok}")
        assert bytes(fr.payload).startswith(said)


@corpora.requires_onair_silent_calls
def test_the_over_search_keys_nothing_on_band_noise():
    """The counterexample, on the population that would cost most if it were
    wrong: a search that keys the transmitter at band noise is worse than one
    that misses an over. The two calls of the same evening that nobody answered
    — 141 s of 40 m and 80 m through the same receiver, one with a narrowband
    signal parked on the channel centre for half the run."""
    for p in corpora.ONAIR_SILENT_CALLS:
        hs, io = _connected()
        _replay(hs, io, corpora.wav_mono(p) / 32768.0)
        assert io.keyed == [], f"{p.name} keyed the transmitter"
        assert io.payloads == [], f"{p.name} delivered payload to the host"


@corpora.requires_onair_gateway_greeting
def test_no_floor_a_tracker_could_have_had_puts_the_over_over_the_gate():
    """The reason this search exists, stated as a number the suite checks.

    "Lower the enter threshold" is the obvious alternative to finding the over by its
    structure, and it does not work. Here the tracker is handed a floor it could not
    have had at the time — a q25 over the band audio that FOLLOWS the over, its own
    transmit mutes dropped — and the over still fails to clear the gate at every
    window length from 5 s to 90 s.

    The figure that stood here before, 1.18x, was in two files and asserted in
    neither. It is what a floor read a hundred seconds later gives; against the
    floor `_BracketSegmenter` actually tracks the same frame reads 21x to 3735x
    depending on where a replay starts, because the mutes collapse a quantile.
    """
    import numpy as np

    seg = corpora.harness("vara_monitor")
    x = corpora.wav_mono(corpora.ONAIR_GATEWAY_GREETING) / 32768.0
    frame = seg.SEG_FRAME
    rms = lambda a: float(np.sqrt(np.mean(x[a:a + frame] ** 2)))   # noqa: E731

    over = range(int(_GREETING[0] * FS), int(_GREETING[1] * FS) - frame + 1, frame)
    assert len(over) == 198, f"the over is {len(over)} frames of {frame}, not 198"
    peak = max(rms(s) for s in over)

    reached = {}
    for secs in (5, 10, 15, 20, 25, 30, 45, 60, 90):
        after = np.array([rms(s) for s in
                          range(int(_GREETING[1] * FS),
                                min(int((_GREETING[1] + secs) * FS), len(x)) - frame + 1,
                                frame)])
        band = after[after > np.median(after) * seg.SEG_MUTE_RATIO]
        reached[secs] = peak / float(np.quantile(band, seg.SEG_FLOOR_Q))

    crossed = {s: r for s, r in reached.items() if r >= 2.0}
    assert not crossed, (
        f"the greeting clears an enter threshold of 2.0x against a floor read over "
        f"{crossed} — an energy gate could have been made to see it after all, and "
        f"this whole search is redundant")
    assert max(reached.values()) > 1.5, (
        f"the over now reads {max(reached.values()):.2f}x its own band; it used to "
        f"read 1.83-1.99x, so either the recording or the floor arithmetic moved")


# KB3AC-10's whole greeting, 2026-08-31, in the two overs it arrived in. The join
# is the proof that they are one message: the callsign is split across it.
TWO_OVER_GREETING = (
    b"RMS Trimode 1.4.2.0 KB3AC-HF-GATEWAY\r\n"
    b"W9SSJ has 1438 daily minutes remaining with KB3AC-1",
    b"0 (FN10PV)\r[WL2K-5.0-B2FWIHJM$]\r;PQ: 13825159\rCMS via KB3AC-10 >\r")


@corpora.requires_onair_two_over_greeting
def test_the_second_over_of_a_greeting_is_read_off_the_columns_not_the_argmax():
    """The 2026-08-31 KB3AC-10 session, whose two overs carry one greeting.

    Over 2 is why this receiver weighs a column's band reading instead of taking
    its argmax. Live, it scored 18 of 24 reference columns and the link closed
    without it; 71 of its 395 columns are won by the wrong bin, but the true bin
    is the runner-up in 23 of those and loses by under 3 dB in 45, so the hard
    read hands the turbo decoder hundreds of confidently wrong bits and the soft
    one does not. The bytes below are also what a stock modem returns off this
    same audio, and the two decoders agree on all 154.
    """
    path, first, second = corpora.ONAIR_TWO_OVER_GREETING
    x = corpora.wav_mono(path) / 32768.0
    got = []
    for t0, t1 in (first, second):
        fr = RX.decode_over(x, int(t0 * FS), int(t1 * FS))
        assert fr.crc_ok, f"the over at {t0:.2f} s did not decode"
        got.append(bytes(fr.payload))

    said = [g[:len(w)] for g, w in zip(got, TWO_OVER_GREETING)]
    assert tuple(said) == TWO_OVER_GREETING, said
    assert b"".join(said).count(b"KB3AC-10 (FN10PV)") == 1, (
        "the callsign split across the two overs did not close", said)


@corpora.requires_onair_two_over_greeting
def test_the_over_that_needed_it_does_not_decode_off_its_argmax():
    """The half of the pair that says the change was needed and not free.

    Over 2 clears the payload-blind guard — it is positively an over — and then
    fails every hard-decision alignment, which is exactly the state
    `_undecoded_over` was written for and the one this recording spent its
    turnaround in. Over 1 decodes either way, so the guard is not what moved.
    """
    import numpy as np

    path, first, second = corpora.ONAIR_TWO_OVER_GREETING
    x = corpora.wav_mono(path) / 32768.0
    r = RX.RECORDS[RX.BASE_LEVEL]

    def hard(t0, t1):
        best = (0, False)
        for onset, g, mag in RX._alignments(
                np.asarray(x[int(t0 * FS):int(t1 * FS)], float), 6):
            bins = mag[g:].argmax(1) + r.first_bin
            hits = sum(bins[c] == b for c, b in zip(RX._REF_COL, RX._REF_BIN))
            fr = RX.check_frame(RX.onair_to_frame(
                RX._onair_from_bins(bins, RX.BASE_LEVEL), RX.BASE_LEVEL),
                RX.BASE_LEVEL)
            best = max(best, (int(hits), fr.crc_ok))
        return best

    hits, ok = hard(*first)
    assert (hits, ok) == (24, True), f"over 1 reads {hits}/24 hard, crc {ok}"
    hits, ok = hard(*second)
    assert hits >= VA._OVER_GUARD_MIN and not ok, (
        f"over 2 reads {hits}/24 hard, crc {ok} — this is no longer the over the "
        f"soft read was measured on")


@corpora.requires_onair_two_over_greeting
def test_weighing_the_band_does_not_make_a_frame_out_of_the_band_noise():
    """The null the soft read needs, on the recording it was measured on.

    A demodulator that never says "no bin" could manufacture a frame out of
    anything, so the same decoder is walked across 40 s of the same session's 80 m
    band noise — every gap between our own keyings while nobody was answering —
    and has to come back empty. The CRC-16 is the only thing refusing it.
    """
    path, _, _ = corpora.ONAIR_TWO_OVER_GREETING
    x = corpora.wav_mono(path) / 32768.0
    for t0 in (110.0, 122.0, 135.0, 148.0, 161.0, 173.0, 186.0, 199.0):
        fr = RX.decode_over(x, int(t0 * FS), int((t0 + 4.6) * FS))
        assert not fr.crc_ok, f"band noise at {t0:.0f} s decoded to a frame"
