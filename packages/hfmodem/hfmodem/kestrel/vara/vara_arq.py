# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VARA-station connection handshake over the MFSK bursts.

Implements the observable connect handshake of [spec 05 §5.3, §5.3.1, §5.3.2]
against a real VARA station, using the spec-derived MFSK burst family
([spec 04 §4.2], via :mod:`vara_frames` + :mod:`vara_mfsk`):

  INITIATOR (host ``CONNECT mycall gateway``)  [spec 05 §5.3 steps 1,2,4,5,6]:
    1. originate CR keyed to the CALLED (gateway) callsign        (step 1)
    2. recognize the connect-response keyed to the gateway        (step 2)
    3. TX the OFDM link-setup carrying MYCALL (the caller)         (step 4) *
    4. recognize the connected-ack by its fixed preamble           (step 5)
       -> CONNECTED                                               (step 6)

  RESPONDER (host ``LISTEN ON``)  [spec 05 §5.3 steps 1,2,4,5]:
    1. recognize an inbound CR keyed to one of MYCALL             (step 1)
    2. TX the connect-response keyed to MYCALL                    (step 2)
    3. RX + FEC-decode the OFDM link-setup to learn the CALLER    (step 4) *
    4. TX the connected-ack -> CONNECTED                          (step 5)

  * The link-setup (step 4) is an ordinary BW2300 rec3 wideband DATA burst
    keyed to the CALLER  [spec 04 §4.2A; spec 05 §5.3.2], synthesised and
    decoded by :mod:`vara_ofdm` over the interop-validated rec3 chain. An
    initiator emits it byte-exact; a responder decodes it to learn the caller
    before keying the connected-ack. Set ``mfsk_only=True`` for the MFSK-only
    kestrel<->kestrel path, which skips the link-setup in both directions (the
    responder then reports the caller as ``CALLER_UNKNOWN``).

Once CONNECTED, an initiator identifies each gateway DATA over (reference-column
alignment + CRC-clean rec3 decode), delivers its payload bytes through
:meth:`VaraIO.data` (VARA over framing per ``arq.phy.vara_payload``, validated
byte-exact against a gateway's own host-port output), and answers it with the
per-over response  [spec 05 §5.3.3].

  DATA PHASE — who may transmit  [spec 05 §5.3.3, §5.4]:
    The link is strictly one over each way, and the session frames say whose over
    it is. Answering a peer's over with the per-over response leaves the peer
    sending. Answering it with the TURN-REQUEST — the one frame in the family
    keyed to the caller rather than to the called station — claims the turn, and
    once the peer has answered that, this station sends the DATA overs and the
    peer answers each. With nothing left to send it keys the turn-idle frame on
    the idle cadence and keeps the turn  [see vara_frames, where each frame's
    generator parameters and the recordings behind them are set out]. It keeps it
    only while the peer keeps answering: a run of DATA overs keyed into our turn
    gives it back  [see ``_TURN_YIELD_AFTER``], because a turn nothing can take
    back is a queue that never drains and a transmitter that keys over a gateway.

    One of those frames per over, and one only: an over opens a turnaround wide
    enough for a single burst  [see :meth:`_answer_data_over`].

    ``send()`` is the whole host-side entry: it queues, asks an idle link for the
    turn, and :meth:`on_rx_audio` drains the queue an over at a time. Called from
    inside a delivery — which is where the Winlink client answers a greeting — it
    keys nothing, and what it queues chooses the frame that answers that over.

Every step traces to the cited spec fact or is marked ``[ours]``. No VARA
identifiers. The FSM is pure logic driven through an injected :class:`VaraIO`.
"""
from __future__ import annotations

import time

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..arq import phy as _phy
from ..rx import varahf500 as _RX500
from ..rx import varahf2300 as _RX2300
from . import vara_control as VC
from . import vara_frames as VF
from . import vara_mfsk as MK
from . import vara_ofdm as OF


class VaraState:
    DISCONNECTED = "DISCONNECTED"
    LISTENING = "LISTENING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DISCONNECTING = "DISCONNECTING"


# Handshake sub-steps (internal)  [spec 05 §5.3].
_I_CR_SENT = "I_CR_SENT"
_I_LINKSETUP_SENT = "I_LINKSETUP_SENT"
_I_CONNECTED = "I_CONNECTED"
# A DATA over is a ~4.4 s wideband burst; the longest MFSK burst is the 1.77 s CR.
# Length is a precondition for answering one, never the reason: see _peer_data_over.
_DATA_OVER_MIN = int(2.5 * MK.FS)
_R_RESP_SENT = "R_RESP_SENT"

#: Bandwidths this station can key a DATA over at: the wideband family's base
#: records, and BW500's level 4  [see arq.phy.render, vara_ofdm.data_over_tx].
_OVER_BW = frozenset(_RX2300.BASE_LEVELS) | {"500"}

# Who may key a DATA over. VARA is strictly one-over-each-way: through a whole
# logged session the two stations never keyed together and never twice in a row,
# so "our turn" is a single flag and not a window  [spec 05 §5.4].
#: Payload tones of the peer's turn-request the receiver has to have delivered
#: before it is answered — three quarters of 31  [see _peer_wants_turn].
_ASK_MIN_HEARD = 24

_TURN_PEER = "peer"       # the peer sends DATA overs; we answer each one
_TURN_ASKED = "asked"     # our turn-request is on the air, awaiting its answer
_TURN_OURS = "ours"       # we send DATA overs; the peer answers each one
# Times the turn is asked for before the queue is abandoned to keep the link up.
# Same shape and the same reason as _LINKSETUP_MAX_TX: one lost burst must not end
# an attempt, and an unanswerable one must not hold the frequency.
#
# TWO IS THE NORMAL COST OF A GRANT AND THE FIRST ASK IS USUALLY SPENT. 35 asks
# over ten fetches at a stock VARA HF 4.9.0 responder, and the outcome is decided
# by one thing with no exceptions either way: 21 of 21 grants came when the peer's
# next key-down fell 0.08-0.14 s AFTER our last sample, and all 14 ignored asks
# were still on the air when it keyed, by 0.01 to 0.34 s. This frame is 32
# symbols, 1.366 s, and with the scan grid and the device start cut the ask stands
# complete at the peer's unkey +1.42 to +1.50 — while the gap after one of the
# peer's own DATA overs is 0.47-1.69 s (median 1.43) and the gap after one of its
# idle bursts is 1.71-1.86. So an ask into an over's gap is a coin toss the frame
# is too long to win, an ask into an idle gap was granted every time, and the
# budget is what carries the first ask's loss. Nothing on our side of the ask is
# left to cut: 0.06 of read, 0.018 of device start, 1.366 of frame.
_TURN_MAX_ASK = 3
# CONSECUTIVE DATA overs keyed into our turn before we stop believing it is ours.
# Holding the turn means the peer answers our overs with short control bursts and
# keys none of its own [spec 05 §5.4], so an over arriving then contradicts the
# law — and nothing in the frame says whether the transmitter is our peer taking
# the turn back or a station we are not in session with [see _peer_data_over]. This
# does not tell them apart; it prices the two mistakes. Yielding on the first costs
# a stranger's single transmission our turn and the queue behind it; never yielding
# leaves the turn a state with no way out. So one over is answered and concedes
# nothing, and a run of them yields — the count is CONSECUTIVE, and any lawful
# answer from the peer puts it back to zero.
_TURN_YIELD_AFTER = 3
# Idle-cadence bursts keyed with the session standing still before the link is
# closed [ours — spec 05 §5.5 leaves the idle-disconnect timeout unfixed, "sessions
# were closed by the host"].
#
# WHAT RESETS IT IS PROGRESS AND NOT TRAFFIC, which is the whole of why the count
# is trustworthy: a payload delivered to the host, an answer from the peer to a
# burst of ours, or the link coming up. A burst we key resets nothing, and neither
# does an over that repeats the last one delivered — a station stuck resending is
# not a session advancing, and an activity clock would hold that link up forever.
#
# Six, because on record a live VARA answers the idle cadence every time: the two
# off-air BW2300 gateway sessions hold ten answered idle frames, 0.112-0.150 s
# behind the initiator's burst, and the loopback corpus's responder answers every
# keepalive, turn-request and over across 479 s. No recording holds a healthy link
# leaving one unanswered, so a run of six — about 70 s at the ~10-12 s cadence — is
# already well past anything measured. It is also where `station.narrate` tells the
# operator the channel is empty, and a modem that raises that alarm and goes on
# keying is the complaint itself: on 2026-08-16 the KC9GHZ session answered its last
# over and then keyed three turn-requests and twenty-three keepalives — twenty-six
# of the session's thirty-four keyings — into a channel the operator could hear was
# silent. It was stopped from outside: that session's log ends mid-cadence with no
# teardown line, and no timeout fired, because there was nothing in this file to end it.
_MAX_WITHOUT_PROGRESS = 6

# What the peer still owes us for the answer we keyed at its last over, while it
# owes it  [see _reack]. An acknowledgement is one burst into one turnaround and
# nothing repeats it, so a peer that missed it waits on its own idle cadence with
# the rest of the delivery in hand: on 2026-09-08 KE8LVA answered seven overs of
# a four-message mailbox, missed our answer to the seventh, idled at 3.55 s for
# 57 s and closed with three messages unsent.
_OWED_OVER = "over"          # an intermediate over was answered, another is due
_OWED_RELEASE = "release"    # the delivery's last over was answered, the turn is due
# Rungs of the re-acknowledgement ladder before the owed answer is abandoned.
# Four because that is what the ladder holds: the frame again, the two other
# continue-class answers, and the NAK  [see OVER_CONTINUE_ANSWERS, _tx_nak]. It
# has to fit inside the give-up budget or the link closes mid-ladder with rungs
# unspent, which is asserted rather than trusted [see tests/kestrel/test_reack_budget].
_REACK_MAX = 4

# --- DATA-over recognition (step 6 onward) ---------------------------------- #
# Reference columns of a base (rec3) BW2300 over, whose lit bins are fixed by class
# and so are known before anything is decoded [hfmodem.kestrel.rx.varahf2300._guard_scores].
_OVER_REF_COLS = len(_RX2300._REF_COL)                       # 24
# Reference columns that identify an over even if its CRC fails. Measured
# over the 246 s of off-air HF in kestrel/tests/corpora: the five real gateway overs
# in it score 20, 23, 24, 24, 24 of 24, everything else in the same recordings tops
# out at 8, and 9 s of synthetic gaussian noise reaches 9. 16 sits between the two
# populations with eight columns of room on each side.
_OVER_GUARD_MIN = 16
# Weaker complete lower-speed frames may still decode: KC9GHZ's BW2750 record
# 101 greeting on 2026-09-20 has 12/24 reference hits and a clean CRC. This lower
# threshold licenses a bounded decode attempt only, never an unread-over/NAK.
_OVER_CRC_MIN = 12
# Sample grid for the alignment search. The RX's offline decoder steps 4, which
# costs 4x for nothing here: at 16 every one of those five overs scores exactly what
# it scores at 4, and the whole check fits in ~35 ms of a live turnaround.
_OVER_ONSET_STEP = 16
# Audio a whole base over occupies, and so the shortest buffer that holds an
# alignment to score at all: 395 columns of 512 samples, 4.21 s.
_OVER_NEED = _RX2300._BASE_NCOLS * _RX2300.RECORDS[_RX2300.BASE_LEVEL].dw50
# Audio carried between scans, so that no over is split across two of them. The
# base level is the SHORTEST of the index family, not the longest: record 2 runs
# 228 columns of 1024 samples, 4.86 s, and a buffer that carries only the base
# level's 4.21 s forward leaves every record-2 over straddling a scan boundary and
# whole in none of them. Sized off the family for that reason, and separately from
# the threshold above: a buffer that had to reach the LONGEST record before its
# first scan would put 0.65 s on the front of every turnaround, including the base
# ones  [see _stream_over].
_PROBE_COLS = 32
_OVER_FRAME_MAX = max(_RX2300.RECORDS[lv].ncols * _RX2300.RECORDS[lv].dw50
                      for family in _RX2300.INDEX_LEVELS_BW.values() for lv in family)
# Keep the longest frame AND the tail used by _window_state. Keeping only
# the body evicts the floor record's head while waiting for its six quiet
# columns, so a CRC-clean L1 frame never reaches the live DATA path.
_OVER_CARRY = max((_RX2300.RECORDS[lv].ncols + _PROBE_COLS + 1)
                  * _RX2300.RECORDS[lv].dw50
                  for family in _RX2300.INDEX_LEVELS_BW.values() for lv in family)
# Longest an acknowledgement waits for the peer to stop transmitting
# [see _answer_over]. Two overs and a second: a stock sender keys two of them
# back to back once a delivery has had a run of successes, so the hold has to
# cover a whole window of that shape, and a carrier that never falls away is not
# a turnaround this end can wait out. Past it the answer goes out and the block
# it may have been keyed across is owed  [see _release_held_answer].
_ANSWER_HOLD_MAX = 2 * _OVER_FRAME_MAX + MK.FS
# Shortest gap between two of the peer's idles that says its WINDOW HAS ENDED
# rather than that one emission was named twice. A responder's idle cadence runs
# 3.3-3.75 s and a window's own second block is a 4.4 s DATA over, so a genuine
# pair is always at least this far apart — while the 32-symbol recogniser is named
# off a third of itself and can take an over-idle out of a block it is sitting in
# [see _a_gap]. Counting namings alone would release on that pair; counting the
# audio between them cannot  [see _release_held_answer].
_IDLE_PAIR_MIN_S = 3.0
# How long after our own final over a responder's turn-request can still be the
# ANSWER to it  [see _final_over_was_read]. NOT the query-reply bound: that one
# measures a query's unkey to its reply, and a responder does not answer an over
# that fast. On the K0SI 40 m tape of 2026-09-16 our final over's last sample is
# at 95.648 s and its two turn-requests at 101.262 and 104.922 s — +5.6 s and
# +9.3 s — and at the stock bench the train starts +5.7 s behind the frame it
# follows and cycles every 3.64 s. Three of those slots covers the train's first
# three requests and stops well short of the peer's own next cadence
# [docs/protocols/vara/20-peer-turn-measurements.md].
_TURN_REQUEST_ANSWER_S = 12.0
# Columns behind a decoded frame that say whether the peer is still keying, and
# the share of the frame's own lit-ness they have to reach  [see _window_state].
# Six columns is 64 ms, and they are read on the frame's own column grid and its
# own equalisation, which is what makes the two sides comparable: behind the
# second block of the 2026-09-09 bench window they read 0.98 of the frame, and
# behind the greeting over the 2026-08-20 KC9GHZ gateway unkeyed after, 0.10.
# A factor of ten, and the share sits in the middle.
#
# THE ANSWER WAITS FOR THEM. Sixty-four milliseconds is added to every
# acknowledgement this station keys, and it buys the only reading that separates
# a peer mid-window from one that has finished: judged on the audio's own level
# instead, an emission reads anywhere from 0.20 to 3.6 of a six-column reference
# inside one block, and a busy band reads higher between two of them than the
# over does.
_KEYING_COLS = 6
_KEYING_FRAC = 0.4
# Below this the band holds nothing at all and the answer goes out on the six
# columns alone: a cable between two transmissions reads 0.000, and no emission
# measured anywhere reads under a tenth of the frame in front of it.
_QUIET_FRAC = 0.05
# Columns the structural probe needs, and the reference columns of the frame that
# would start where they do that it scores  [see _next_frame_here]. A record's
# reference columns sit at 15, 31, 48 and 64, so two of them is 32 columns —
# 341 ms, which is what an answer waits when the level cannot decide.
_PROBE_REFS = 2
# Columns of the audio behind our own answer that say it was keyed across a block
# [see _watch_after_answer]. Twelve, where the level reading steadies: over the
# 2026-09-09 captures a peer still keying reads 0.93 and up of the frame in front
# of it and a band it has stopped keying into reads 0.04.
_AFTER_COLS = 12
# The narrow bandwidth's over is found by its span, not by a reference-column
# score: a burst holds one frame or two and its measured length is what says
# which  [rx.varahf500.frames_carried]. The live track is the offline detector's
# own law — a column of band energy against the burst's peak
# [rx.varahf500.burst_spans] — kept as the audio arrives  [see _stream_over_500].
_OVER500_COL = 1024
_OVER500_MIN = -(-_RX500.BURST_MIN // _OVER500_COL)
# Two level-1 frames last 10.67 s, longer than the 8.50 s base pair. Keeping
# only the base geometry clipped the first frame before the falling edge,
# turning a clean low-level burst into an unread window on the live stream.
_OVER500_LONGEST = max(
    (2 * _RX500.NSYM + _RX500._PREAMBLE) * _RX500.H,
    *((record.lead + 2 * record.ncols) * record.dw
      for record in _RX500.INDEX_RECORDS.values()))
_OVER500_CARRY = _OVER500_LONGEST + MK.FS
_OVER500_LEAD = 4 * _RX500._PAD
_OVER500_TAIL = 3 * _OVER500_COL
_OVER500_HZ = np.fft.rfftfreq(_OVER500_COL, 1 / MK.FS)
_OVER500_BAND = (_OVER500_HZ >= _RX500.BURST_BAND[0]) & (_OVER500_HZ <= _RX500.BURST_BAND[1])


def _span(kind: VF.BurstKind) -> int:
    """Audio one burst of ``kind`` occupies, first sample to last."""
    return ((len(kind.preamble) + kind.n_payload - 1) * MK.HOP + MK.STRIDE)


# The same for the short frames a peer keys during the data phase, and both ends of
# the family: the longest is 32 symbols on the hop with the last one whole, 1.37 s,
# and the shortest is the responder's release at 17, 0.73 s. The first is what a
# buffer carries between scans; the second is when a fresh one is first worth
# scanning  [see _stream_grant, _stream_answer].
_SESSION_NEED = _span(VF.SESSION_DRAINED_RESPONDER)
_SESSION_MIN = _span(VF.SESSION_TURN_RELEASE_RESPONDER)
# And the 11-symbol control burst, whole: the shortest frame a peer keys at us
# and the one it polls with after our release  [see _took_poll].
_POLL_MIN = (VF.CONNECTED_ACK_NSYM - 1) * MK.HOP + MK.STRIDE
# Polls per answer: the ~10 s idle cadence in units of the responder's 2.3 s poll.
_POLLS_PER_ANSWER = 4
# `_peer_data_over`'s third answer, told apart by identity: the audio IS an over —
# its reference columns line up — and the frame will not decode, so there is no
# body to hand back and the peer is still owed a reply. `None` keeps its meaning of
# not an over, and nothing owed.
_UNDECODED = b"<over identified, frame not decoded>"
_UNREAD_OVER = b"<over behind an acknowledged one, nothing in it decoded>"
#: Columns of 512 an all-bad BW500 burst spans before it is claimed as the
#: peer's over behind one we acknowledged  [see _peer_data_over_500]. Our own
#: answer can come back through the receiver and fail to decode, and the longest
#: of them is the 32-symbol session frame, 130 columns; the shortest over the
#: ladder keys is one base-level frame, 403.
_UNREAD_OVER_MIN_COLS = 300
# Overs identified by their reference columns and not decodable, answered with the
# NAK token, before the link is closed instead.
#
# Two, because a NAK asks for the over again and that repair has exactly one thing
# to fix: a fade over the frame we happened to be handed. A repeat that fails the
# same way puts the fault on this station's decoder, which the peer cannot mend by
# sending it a third time — so a third ask is a keying loop with a known answer.
# Both ways of getting this wrong end the same, with a peer transmitting into a
# shared channel to a station that has stopped taking part: on 2026-08-18 an
# 18-of-24 over would not decode, this station went quiet, and KC9GHZ went on
# transmitting. The budget is what makes answering safe rather than a second way
# to be that station.
_OVER_NAK_MAX = 2
_OVER_RETRY_MAX = 3  # Confirmed retransmissions of one unacknowledged over.
_FINAL_QUERY_MAX = 3  # Successfully keyed answer solicitations per pending frame.
_FINAL_QUERY_REPLY_S = 2.0  # Stock reply ended 1.46 s after query unkey.
# How long an outstanding DATA over waits before this station keys for it again.
# A stock caller holding one it has had no answer for keys at 1.822 s past its own
# unkey  [bench 2026-09-12], and what makes that safe is that the answer is long
# finished by then: a peer opens
# 0.09-0.17 s behind our last sample, the longest thing it keys there is the 0.47 s
# control burst, so 0.64 s ends it and the 0.8 s of audio `GAP_WINDOW` holds is what
# the reader gets to decide on.
#
# WHAT IT IS NOT IS A CLEAR CHANNEL. A peer's turnaround is not one burst: behind
# our own retries on 2026-09-18 K0SI keyed a 1.37 s turn-request opening at
# +0.875 s, another at +2.325 s, and more out to +9.6 s. No interval dodges those,
# and ten seconds dodged nothing either — that turnaround's last peer frame ended
# at +9.557 s and the retry behind it keyed at +10.01. What keeps this off the peer
# is the gate, read at the instant of the decision  [kestrel_connect.mail_session,
# AudioVaraIO.receiving], and the interval only decides when to start asking.
#
# The ten seconds this used to take were the IDLE cadence's, reached because
# `_retry_data_over` hangs off `idle_keepalive`: 10 s answers "I have nothing to say,
# are you there", and a retransmission asks something else. On 2026-09-18 KC9GHZ
# answered our second DATA over at +0.12 s, asked for the next one in a frame this
# build could not read, and never transmitted again — the three retries that followed
# went into a channel it had left, at +10.24, +10.24 and +10.16 s.
_OVER_RETRY_S = 1.8
# Where a stock caller keys `session-over-nak` behind the responder's idle: 0.242 s
# after the idle's last sample on both arms of 2026-09-11, inside the 1.7-1.9 s
# the idle's listening gap runs  [see _stream_recovery_cue]. The scan that keys it
# is due there exactly: on the quarter-block grid it would land anywhere up to
# 0.12 s later, and the frame's recogniser names it before its tail has arrived.
_OVER_NAK_LEAD_S = 0.24

# --- connected-ack recognition (step 5) ------------------------------------- #
# Tone-track resolution. Every alignment hypothesis reads the same analysis
# windows, so they are computed once on this lattice instead of once per
# hypothesis; the exact-match plateau is ~1400 samples wide [vara_mfsk], so 32
# samples costs nothing.
_ACK_GRID = 32
# The ack's fixed preamble, and where its symbols land on the lattice.
_ACK_PRE = np.array(VF.CONNECTED_ACK_PREAMBLE, dtype=np.int32)
_ACK_PRE_OFF = np.rint((np.arange(len(_ACK_PRE)) * MK.HOP + MK._WOFF)
                       / _ACK_GRID).astype(np.int64)
# The preamble symbols actually compared, and their offsets from the first of them.
# The leading symbol is not required. Two things take it and nothing else: the tail
# of our own transmission, which leaves this station's input dead past the 0.12-0.17 s
# a peer answers in, and an ordinary fade. Both are measured, the second in the
# corpus — qso_vara_ns0a.wav carries a real gateway ack at 19.379 s whose symbols
# 1-3 are exact and whose symbol 0 reads (61, 64) for (64, 67), one carrier of the
# pair lost. The four-symbol form rejects it, which is a recogniser that fails on
# real HF rather than a gate doing its job.
_ACK_MATCH = _ACK_PRE[1:]
_ACK_MATCH_OFF = _ACK_PRE_OFF[1:] - _ACK_PRE_OFF[1]
_ACK_SPAN = int(_ACK_MATCH_OFF[-1])
# Consecutive lattice offsets the preamble must hold for. A real symbol grid reads
# the same way over most of a symbol, so a genuine ack matches over a wide plateau:
# measured, the nine recordings that hold ``_ACK_MATCH`` at all hold it for 7 to 59
# lattice points -- the list is below -- and the four loopback acks for 62-63, at
# BW500, BW2300 and BW2750 alike. Chance does not survive a step.
#
# Measured over every real recording the project holds at every alignment on the
# lattice — 243 of them, this station's own connect attempts (one 83 minutes long),
# the shared regression corpus (PACTOR-1/2/3, ARDOP, FT8, WSPR and band noise from
# four continents), the off-air gateway sessions, the verified clear channel and the
# witnessed attempts: 31,393,466 alignments. Nine recordings hold ``_ACK_MATCH`` at
# any alignment and all nine carry a VARA session's own two-tone control bursts;
# the other 234 hold it nowhere. Those nine hold for 7, 24, 31, 48, 50, 50, 52, 52
# and 59, so three is under half the narrowest and the floor under it is zero.
_ACK_PLATEAU = 3
# Carriers of the three compared preamble symbols that must be comparable before an
# alignment can hold at all  [see :func:`_ack_hold`]. Six is every carrier of every
# symbol, and requiring six is what an exact match asks for.
#
# Six is too many, and the recordings say so twice. `qso_vara_ns0a.wav` carries a
# real gateway ack whose symbol 0 is already conceded to a fade (``_ACK_MATCH``);
# under the clearance rule its symbol 1 loses a carrier too, and at six it is
# refused — a recogniser failing on the one real ack the corpus was kept for. On
# 2026-08-19 W8MW answered a link-setup on 80 m with an ack whose lower carriers sit
# under a busy channel: four of its six stand clear, all four are the preamble's,
# and it holds for 24 alignments while nothing else in that recording reaches four
# at all.
#
# Four is where the population puts the cut. Over the 31,393,466 alignments above,
# no recording without a VARA session on it holds four comparable carriers clean at
# any alignment, and the best any of them reaches is three. Every genuine ack in the
# corpus clears it: 7, 24, 31, 48, 50, 50, 52, 52 and 59 lattice points. Five would
# lose W8MW's and shorten NS0A's to 14; three is one exact pair and one dominant
# carrier, which is not enough evidence to bring a link up on.
_ACK_MIN_CARRIERS = 4
# Symbols of the three that may have a carrier under another station and still hold
# [see :func:`_ack_held`]. One, for the reason ``_ACK_MATCH`` concedes one preamble
# symbol and no more: each concession is a symbol the alignment is no longer pinned
# by, and two of three leaves one.
#
# It is the whole of what a busy frequency costs, and it is what six unconnected
# arms were. On 2026-08-26 KD0PYG answered a link-setup on 40 m over an occupant
# 8.9 dB up: its preamble symbols 1 and 2 read (56, 74) and (64, 69) exactly, and
# symbol 3's own upper carrier 78 sits 5.9 dB under the occupant's 95/96, which
# takes its place in the pair. Two carriers right and one displaced, at every
# alignment across the burst — `preamble holds for 0 alignments`. Read as one
# crowded symbol it holds for 27.
#
# One rather than two is what the population pays for. Measured over every
# recording this station holds — 331 of them, 28,577 s, 42,802,089 alignments, the
# shared regression corpus and 283 on-air captures included: at one crowded symbol
# the recordings that hold are the sixteen the exact-pair rule already held plus
# the KD0PYG ack, and every one of the seventeen carries a VARA session's own
# two-tone control bursts. Nothing else reaches four comparable carriers clean at
# any alignment and the best anything else reaches is three, which is the floor the
# exact-pair rule was measured against and it has not moved.
#
# At two, `pos_p1_twosided_14110.wav` — a two-tone PACTOR-1 exchange with no VARA
# anywhere in it, kept because it holds NOTHING at zero offset — reaches four
# carriers clean over two consecutive alignments, one short of :data:`_ACK_PLATEAU`.
# A floor of zero and a floor of one alignment under the bar are not the same
# thing to bring a link up on.
_ACK_MAX_CROWDED = 1
# How late after the link-setup an ack may still begin. The ack names nobody, so
# what makes it ours is only that it answers the over we just sent; the wider the
# window, the more of a busy frequency's other sessions we are exposed to. Measured
# off air: this gateway starts its connect-response 0.121-0.158 s after our last
# transmitted sample (six attempts) and real VARA's own session with it was answered
# 0.171 s after the link-setup's last sample. A second is six times that, and it
# holds the exposure to 3 s an attempt — against the ~710 s of listening on this
# frequency that turned up one unrelated station's ack.
_ACK_WINDOW = int(1.0 * MK.FS)
# --- the same burst read whole -------------------------------------------- #
# Everything above scores three preamble symbols: six carriers of the twenty-two
# an 11-symbol burst carries. The other sixteen are known as well — a link's
# tails are keyed to the callsign that CALLED it, and every link this station
# has is one it called  [vara_frames, control_bursts] — and no reader here has
# ever looked at them.
#
# That cost five of the thirty-two bursts nine gateways keyed on 2026-09-17/18.
# NS0A answered two DATA overs and VE3WLR one with all seven tail symbols exact
# and preamble symbols 1-2 taken by the channel; K9KDJ and KB8AY answered a
# link-setup the same way. The preamble test refused all five at full strength,
# and loosening it does not reach them: at three comparable carriers and three
# crowded symbols — past every floor the constants above are measured against —
# four of the five are still refused. The gates are not mistuned. They are
# reading a sixth of the frame.
#
# So this reads the same evidence whole, and softly. Per symbol the expected
# pair's share of its own analysis window's band power is compared with the
# 2/N a flat band would leave there, and the frame's score is the mean of the
# logs  [see :func:`_burst_match`]. No symbol has a veto: one under this
# station's own mute contributes a ratio near 2 and a term near 0, one carrier
# lost contributes about half a clean symbol's, and fourteen good carriers
# behind a broken head carry the frame. It is the argument `_onair_llr` already
# made for a column's whole band over its argmax, at a symbol instead of a bin.
#
# IT IS A SECOND ROUTE AND NEVER A REPLACEMENT. It knows one tail per link and
# the preamble belongs to every station: the captured `CONNECTED_ACK_2300`
# scores 0.44 here and holds the preamble for 59 alignments, so a peer keying a
# tail this file does not hold is read by the test above and by nothing else.
_BURST_MATCH = 1.5
# Where the population puts the cut. Swept the way `_ACK_PLATEAU` was, over the
# 28 recordings of the shared regression corpus that carry no VARA session,
# nine grids from -0.5 to +0.5, 11,144,889 alignments per table:
#
#   bar          0.6     0.8     1.0     1.2     1.5
#   responder 2300     4934     310       0       0       0
#   responder 2750     8368     689       0       0       0
#   responder 500    292515   91102   12738    1011      94
#
# and on 1,106,539 alignments of this station's own 2026-09-17/18 on-air tapes,
# away from any answer window, at each session's own grid, shift and table, the
# ceiling is 0.976. Nothing without a VARA control burst in it has reached 1.0
# anywhere. Real bursts off the air run 1.96 to 3.21 and a synthesised one
# scores 3.14, so 1.5 is half-way between the two populations in the log the
# statistic is taken in, and 0.5 clear of every alignment ever measured under
# it. Against `_ack_plateau` on the 23 bursts BOTH readers take it is worth a
# median +1.0 dB of added noise (-1.5 to +8.5); the five above it recovers
# outright.
#
# 1.2 would be worth +3.5 dB instead and the corpus holds nothing there either.
# It is not taken, because a false accept here is not recoverable: a VARA DATA
# over carries no block index, its acknowledgement carries no sequence field
# [spec 05 §5.7, and the seven state symbols are identical at every occurrence],
# and a NAK can only ask for the outstanding over — which, once we have wrongly
# advanced, is the next one. An over retired early is a block neither end can
# name again. So the bar buys margin rather than dB.
#
# THE BW500 TAIL IS NOT IN THIS. Its seven state symbols lie on carriers 50-76,
# inside the span a two-tone PACTOR-1 exchange works, and the table shows what
# that costs: `pos_p1_twosided_14110.wav` — the recording that already sets this
# file's floor one carrier low — reaches 1.596 against it. The wide tails reach
# 1.0 nowhere. So this route runs at the two bandwidths whose tails stand clear
# of the corpus, and BW500 keeps the reader it has  [see _peer_burst_match].
_BURST_MATCH_BW = ("2300", "2750")
# Times we will re-send the link-setup when the gateway repeats its
# connect-response (it did not hear the first one).
_LINKSETUP_MAX_TX = 3
# Times an unanswered link-setup sends the attempt back to step 1. Raise it only
# on evidence that a later restart completes a connect; each one is worth another
# _LINKSETUP_MAX_TX transmissions of a 4.36 s over.
_CONNECT_MAX_RESTARTS = 2

# --- the peer's answer to an intermediate over (step 6) --------------------- #
# The control burst above answers the LAST over of a delivery and only that one.
# Every over before it draws an eight-symbol burst instead, and until this was read
# an outbound message longer than one over had nothing to advance it: we would key
# the first 89 bytes and wait out the answer that arrived. It has never bitten on
# the air only because the B2F login block is one over.
#
# THERE IS NO PREAMBLE HERE TO LOCK. Fourteen copies across five two-cable bench
# sessions share exactly one symbol — the first, which is the control burst's own
# opening pair — and the seven behind it move from over to over: the two a stock
# responder keyed into one 178-byte delivery differ in all seven. So what names the
# burst is its shape, and the shape is what our own frames do not have: eight
# consecutive symbols each carrying a clean two-carrier pair, opening on that pair,
# where every generated frame in this family is one tone per symbol.
_CONT_OFF = np.rint((np.arange(VF.OVER_CONTINUE_NSYM) * MK.HOP + MK._WOFF)
                    / _ACK_GRID).astype(np.int64)
_CONT_SPAN = int(_CONT_OFF[-1])
# Consecutive lattice offsets the shape must hold for, measured the way
# `_ACK_PLATEAU` is. Over the 31 recordings of the shared regression corpus, the
# five declared quiet stretches and 3404 s of this station's own off-air captures —
# band noise on four bands, two channel senses, ten gateway sessions and the 80 m
# QRN pair — nothing holds past 5, and the worst of those is 80 m band noise. The
# one recording that reaches further is `qso_vara_ns0a.wav` at 17, which is a real
# VARA session's own control burst and is claimed by the reader above before this
# one runs. Genuine copies hold for 54 to 60, and still for 53 under additive noise
# at 0 dB SNR. Twelve is over twice the floor and under a quarter of the narrowest
# copy.
_CONT_PLATEAU = 12
#: Audio the answer search takes before its first scan, which is this burst's own
#: span: the shortest frame the search can name is what says when it is worth
#: looking, and the family's shortest is now 0.34 s rather than 0.73. Every tenth
#: of a second here is a tenth the peer waits for the over it asked for. This
#: burst alone is looked for that early — it is the only one of the six that has
#: to be whole before its own recogniser will take it  [see _stream_answer].
_ANSWER_MIN = (VF.OVER_CONTINUE_NSYM - 1) * MK.HOP + MK.STRIDE


def _after_frame(x: np.ndarray, mag: np.ndarray, r) -> np.ndarray:
    """The audio behind the frame ``mag`` was read at, to within the column its
    alignment was found on.

    ``mag`` runs from the frame's first column to the end of ``x``, so what is
    left behind the frame is everything past its own record's worth of that —
    plus the column the onset search swept inside, which is kept rather than cut:
    a whole record has to line up before anything is read out of it, and one
    column of the frame just read cannot be mistaken for the start of another.
    """
    keep = (len(mag) - r.ncols + 1) * r.dw50
    return x[-keep:] if 0 < keep < len(x) else x[:0]


#: What the audio behind a decoded frame says about the peer  [see _window_state].
_KEYING, _QUIET, _UNKNOWN = "keying", "quiet", "unknown"


def _next_frame_here(tail: np.ndarray, r) -> int:
    """Reference columns of a frame starting where ``tail`` does that light the
    bin their own class fixes  [see varahf2300._column_roles].

    THE STRUCTURE AND NOT THE LEVEL. A record's reference columns are lit by
    class rather than by payload, so each one is a sixteenth of a chance on
    noise and about nine in ten on a real frame — which is what separates a
    second block from an empty band where the level cannot. On the window the
    2026-09-09 fetch lost, the level read 0.304 of the frame in front of it and
    every reference column of the block behind it hit.

    THE CALLER REQUIRES ALL OF THEM, and the two readings are held the other way
    round from each other on purpose. Missing a block costs mail; holding an
    answer for a window that is not there costs the whole bound, which is far
    worse than the 341 ms the probe itself spends. So the level holds on its own
    where it is confident and the probe holds only on a full house: one in 256 on
    noise against four in five on a real frame, and between them they miss about
    one window in forty where the level alone missed one in eight.
    """
    cols, want = _RX2300._ROLES[r.level][:2]
    return sum(int(np.argmax(tail[c]) + r.first_bin == w)
               for c, w in zip(cols[:_PROBE_REFS], want[:_PROBE_REFS])
               if c < len(tail))


def _window_state(mag: np.ndarray, r) -> str:
    """Is the peer still keying behind the frame ``mag`` was read at?

    Three answers, because two do not fit the evidence. The level reading over
    six columns is fast and right at its ends — a cable between transmissions
    reads 0.000 and a block under way reads 1.0 — and its middle is where the
    populations meet: a peer still keying has been measured at 0.28 and a quiet
    band at 0.41, so a threshold there misses about one window in eight, which is
    what the 2026-09-09 draws show. In that middle the answer waits for
    :data:`_PROBE_COLS` columns and the structure decides
    [see :func:`_next_frame_here`].
    """
    tail = mag[r.ncols:]
    if len(tail) < _KEYING_COLS:
        return _UNKNOWN
    ref = _lit(mag[:r.ncols] ** 2)
    now = _lit(tail[:_KEYING_COLS] ** 2)
    if now >= _KEYING_FRAC * ref:
        return _KEYING
    if now <= _QUIET_FRAC * ref:
        return _QUIET
    if len(tail) < _PROBE_COLS:
        return _UNKNOWN
    return _KEYING if _next_frame_here(tail, r) == _PROBE_REFS else _QUIET


def _lit(p: np.ndarray) -> float:
    """How strongly these column powers read as an over: the median column's
    peak against its own band median  [see _window_state]."""
    return float(np.median(p.max(1) / (np.median(p, axis=1) + 1e-18)))



def _delivery_stage(n: int) -> str:
    """What the peer's per-over field says about how much of its delivery is left
    [see arq.phy.overs_after].

    A stage and not a count. The field saturates, its bottom step is 8 where the
    rest are 4, and the first over of a delivery does not sit on the ladder at
    all — so a number read off it states more than the bench measured, and would
    be wrong by one over the middle of the delivery it was measured on.
    """
    if not n:
        return "this is its last full over"
    return ("the end of the delivery is not near" if n == _phy.OVERS_AFTER_MAX
            else "the end of the delivery is near")


def _frame_field(overs_after: int, base_close: bool = False) -> int:
    """The per-frame field a FULL body carries, given the full overs still to come
    behind it in the same delivery  [see arq.phy.vara_body].

    The last byte of a full body is not payload, and until this was read it went
    out as zero at every over. What that costs is the whole outbound direction: a
    stock responder handed two full overs at zero delivers nothing at all to its
    host — it answers the first with the 32-symbol frame, ignores the second and
    the closing over, and the message never ends. The same two overs carrying this
    field deliver 178 of 178 bytes, and each of them draws the eight-symbol
    continue burst  [see _peer_over_continue].

    Five values are on the recordings and they are one arithmetic: ``0x80`` over
    ``4k + 1``. The last full over of a delivery takes ``0x81`` — a 178-byte
    delivery's second over, a 200-byte BW500 one's fourth, and the only full over
    of two one-over deliveries — and the ones in front of it count up from
    ``0x8d`` by four: ``0x8d`` at one over to come, then ``0x91``, then ``0x95``,
    which is a stock caller's own 200-byte BW500 delivery read back off the tape.
    Flown at BW2300 as ``0x95 0x91 0x8d 0x81``: 356 of 356 bytes byte-exact,
    every intermediate over answered and the delivery closed.

    THE LAST FULL OVER ALSO ANNOUNCES THE RECORD ITS CLOSE COMES AT, which is
    what ``base_close`` says and what the send leg turned on. A stock 4.9.0
    responder will not take a closing over keyed at the base record behind a full
    over carrying ``0x81``: on the cables of 2026-09-09, same tree and same
    cables, a base-record close behind ``0x81`` drew nothing 0 of 4 and an
    identical one at record 2 was byte-exact 3 of 3. The sender's own convention
    is the pairing ``close_level`` already reads off eight deliveries — ``0x81``
    before a record-2 close, ``0x89`` before a base-record one, which is where a
    close too long for record 2's 48-byte body has to stay.

    Bodies at other levels carry the same field in their own last byte, and the
    short body's ``0x82`` is written by the trailer rather than by this.
    """
    if not overs_after:
        return 0x89 if base_close else 0x81
    return 0x80 | (4 * (min(overs_after, _phy.OVERS_AFTER_MAX) + 2) + 1)

# --- connect-response located by payload (step 2) --------------------------- #
# Payload tones a connect-response must confirm when it is found by what it says
# instead of by its preamble. The preamble path needs no such number: it locks 8
# fixed tones first and then judges the payload by the module's acceptance
# fraction (12 of 15). This path has no preamble to lock and searches thousands of
# alignments instead of one, so it is held to a stricter bar than a whole burst
# ever was.
#
# Measured over 397 s of real off-air HF — the 150 s call to a gateway (our own
# eight transmissions heard back through the receiver, the gateway's two answers,
# and a third station's narrowband traffic from ~99 s on), the two BW2300
# gateway sessions, and the verified clear-channel capture — scored against five
# callsigns each at every alignment on the 32-sample grid, negative starts
# included: 2,937,720 alignments, of which the best non-response reaches 5 of 15.
# (288 s stood here until 2026-08-19 — those four recordings less the NS0A
# session, against an alignment count only the four of them reach.)
# All four genuine answers in the corpus (KB9MMT twice, NS0A twice — repeating is
# what gateways do) reach 15 of 15. 13 leaves eight tones of room on either side of
# that gap, and cannot be met by fewer than 13 comparable symbols.
#
# Both searches multiply that population by :data:`_RESP_SHIFTS` since 2026-08-15,
# and the floor is re-measured across the shift axis rather than assumed to carry;
# :meth:`VaraStationHandshake.on_rx_stream` is held to the same number against a
# far larger population again, measured separately in its docstring.
_RESP_MIN_TONES = 13

# Tones a muted receiver never delivered are not tones the gateway got wrong, and
# on this station most of a connect-response's payload arrives while the receiver
# is still muted. Measured on 2026-08-06, calling KC9GHZ on 7103.5 kHz with a
# KiwiSDR witnessing the same minutes from another site: the gateway answered three
# of our eight requests, and its payload opens ~0.18 s after our own last
# transmitted sample (six detections across the two receivers, 14 consecutive
# correct tones at the witness). This station's input was dead from TX_end+0.02 to
# TX_end+0.44 that night, so six payload tones landed in the mute and nine reached
# the demodulator. All nine are correct in all three answers, and 9 < 13.
#
# So a symbol whose window the receiver was deaf through is scored as not
# comparable, and an alignment that confirms every one of at least this many
# comparable tones is accepted on that.
#
# The mute has since been most of the way repaired, and the ceiling that number
# implied went with it. Read off the recordings themselves as the interval between
# the last transmitted sample and the first frame back over _DEAF_DBFS, this
# station's mute runs 430 ms on 2026-08-02, 420 on 2026-08-06 00:59, 224 three
# minutes later, 120-130 on 2026-08-09/10, and 167-180 from 2026-08-15 on. 250 ms
# of that was a deliberate hold past the last sample and is gone; what remains is
# the rig's own T/R recovery and the codec under it. At 167 ms all fifteen payload
# tones of an answer 0.18 s behind us reach the demodulator, so nothing here is
# choosing between nine and thirteen any more, and the number below is the corpus
# floor and only that.
#
# The floor is what the corpus says. Measured over 212 s of this station's own
# off-air audio — the three calls of 2026-08-06 and the receiver mutes in them —
# scored against the 340 published gateway callsigns not on any of them, with the
# live mask applied: 106,394,234 alignments, and not one wrong callsign sweeps ANY
# comparable set clean. What that measurement bounds is the confirmed count, not
# the threshold: the best a wrong callsign reaches is 6 of 15 comparable tones, 5
# at nine or ten comparable, and 3 where eight are comparable. It does not locate a
# knee — 4 would sweep this corpus exactly as clean — so what it supports is that
# eight sits two tones above anything the population has ever confirmed, on the
# same footing as the 13-of-15 above sitting eight tones above its own best of 5.
# The per-tone arithmetic is the other half: a payload symbol draws from 35 tones
# (measured across that panel), so a clean sweep of eight is 35**-8 = 4.4e-13 per
# alignment against the ~90,000 an attempt searches.
_RESP_MIN_HEARD = 8

# Tone shifts a response is scored at, in carriers of 23.4375 Hz. A station tuned
# off frequency moves every tone by the same amount, so a matcher that compares bin
# indices at zero offset scores its answer at chance and reports silence.
#
# Measured on air 2026-08-15, calling N0LCR-1 on 7103.5 kHz
# (logs/onair/20260815T020415Z-W9SSJ-N0LCR-1.wav): the gateway answered our second
# connect-request at 18.35 s and its whole burst sits one carrier low, -23 Hz. At
# zero offset no alignment anywhere in those 60 s confirms more than 4 of 15 payload
# tones, and the attempt resent the request until it ran out; at -1 the answer
# confirms 14 of 15 with all fifteen comparable. `tools/vara_monitor.py` reads that
# burst off the same file, because it has searched +-3 carriers since it was
# written.
#
# Two carriers rather than three, and the bound is our own next transmission rather
# than this search: 2 carriers is 47 Hz, and a real VARA stops decoding a link-setup
# somewhere between +60 and +90 Hz of offset (bench, 2026-08-14 §4) — so an answer
# taken at +-3 can come from a station that cannot read the link-setup we would
# answer it with.
#
# It costs nothing measurable. Over 2342 s of real off-air HF in 37 recordings —
# this station's own calls, the gateway sessions, the verified clear channel, the
# 2026-08-14 slot's 6800 kHz and quiet-40 m controls, and the 31 of the shared
# regression corpus — scored against every panel callsign not on the recording at
# every alignment and every one of these five shifts: 134,094,970 alignment-shifts,
# best non-response 5 of 15. That is the number the zero-offset population alone
# already reached, so the shift axis multiplies the search without raising the floor
# it is measured against, and no wrong callsign sweeps any comparable set clean at
# any shift (``test_no_wrong_callsign_sweeps_a_comparable_tone_set_clean``, 0
# against the 8 of ``_RESP_MIN_HEARD``).
_RESP_SHIFTS = tuple(range(-2, 3))

#: Below this the receiver was not hearing the band — it is muted, not quiet. The
#: mute reads -83 dBFS on this station against a band noise of -13 to -21.
_DEAF_DBFS = -50.0

#: How far a symbol's tone has to stand clear of the next peak outside its own main
#: lobe, in dB, before it is scored as a tone at all.
#:
#: :data:`_DEAF_DBFS` catches the window a muted receiver delivered nothing
#: through. It cannot catch the one either side of it, where the receiver is
#: delivering the band at a fraction of the level it settles to a few symbols
#: later: an argmax always returns a bin, that bin reads well over an absolute
#: floor, and the difference between a tone and a noise peak is not in the level
#: but in whether anything else in the band came close to it.
#:
#: Measured against exact ground truth over 446,900 labelled symbol reads — this
#: station's own connect-requests lifted out of 113 of its own recordings, where
#: they read 41 of 41 back through the transmit mute, then buried in the verified
#: clear-channel capture at ten levels from +30 to -6 dB. P(the tone read is the
#: tone sent) runs 0.26, 0.33, 0.42, 0.51, 0.61, 0.69, 0.75, 0.82 over half-dB
#: steps up from zero, and crosses even odds at 1.9 dB. Under 2 dB a read is
#: likelier wrong than right, so it is evidence for nothing and is scored as
#: nothing; over it, 95.0% of reads are the tone that was sent, rising past 99% by
#: 6 dB.
#:
#: This does not lower the bar. A symbol dropped here leaves both the confirmed
#: count and the comparable count, so the clean-sweep route still has to sweep
#: :data:`_RESP_MIN_HEARD` of them and the 13-tone route gets no easier. Measured
#: over the false-accept population of :meth:`VaraStationHandshake.on_rx_stream`:
#: 110,319,450 alignment-shifts of real off-air HF scored against every panel
#: callsign not on the recording, best non-response 5 of 15 with the rule and 5 of
#: 15 without it, and not one alignment sweeps a comparable set of eight or wider
#: clean at 1, 1.5, 2 or 3 dB.
_CLEAR_DB = 2.0

# --- an answer that names nobody (step 2) ----------------------------------- #
# Preamble symbols of the eight that have to be comparable before a burst may be
# read as a connect-response addressed to somebody we cannot name. Every
# comparable one must be exact; this is how many of them there have to be.
#
# Seven, and the eighth is conceded for the reason ``_ACK_MATCH`` concedes one:
# the answer opens inside this station's own post-transmit mute. Measured on the
# two 2026-08-26 KB3AC-10 arms, where the burst begins 4 ms after the receiver
# comes back — preamble symbol 0 reads 0.80 and 0.96 dB of clearance on the two
# recordings and is not comparable at any alignment, and symbols 1 through 7 are
# exact at every alignment across the burst.
#
# It is a strong signature and not a loose one: the preamble is eight fixed
# carriers taking seven distinct values out of the seventy the tone track can
# return, so no steady tone and no slow drift can hold it. Measured over the
# negative corpus — the 31 shared regression fixtures, the verified clear
# channel and the two 2026-08-06 calls nobody answered, 1362 s and 9,967,325
# alignment-shifts — nothing reaches seven comparable preamble symbols exact at
# any shift. Six is reached four times, all four in recordings of real VARA
# sessions, and none of the four carries a payload this generator could have
# emitted.
_UNATTR_MIN_PRE = 7


@dataclass(frozen=True)
class PeerAnswer:
    """A burst carrying the dialled station's OWN payload, at whichever frame of
    its stream it sits on.

    Every one carries the CRC-16/GENIBUS of the callsign this station had just
    dialled, exactly; only the pre-advance ever differed from the connect-response
    the search was regenerating, and that is what read them as answers from nobody.

    An observation. The bandwidth's own connect-response is accepted by the
    callsign route ahead of this one; nothing here brings a link up.
    """
    at: float                       #: seconds into the attempt the payload began
    position: int                   #: frame of the callsign's stream  [VF.payload_position]
    shift: int                      #: carriers the burst arrived off frequency by
    tones: int                      #: payload tones confirmed
    comparable: int                 #: payload tones the receiver delivered clear
    payload: tuple[int, ...]        #: the tones read, -1 where not comparable


@dataclass(frozen=True)
class UnattributedAnswer:
    """A connect-response this station heard and cannot put a name to.

    Both routes that accept an answer are keyed to the callsign we dialled, so a
    burst that is unmistakably a connect-response and is addressed to somebody
    else reaches neither and is reported as silence. That happened twice in one
    slot on 2026-08-26 and was found by ear.

    An observation, never an authorisation. Nothing here advances the handshake,
    keys anything, or is offered to :meth:`VaraStationHandshake.on_rx_tones` — the
    callsign check is what brings a link up, and this route exists precisely
    because the callsign check said no.
    """
    shift: int                      #: carriers the burst arrived off frequency by
    preamble: int                   #: comparable preamble symbols, all of them exact
    payload: tuple[int, ...]        #: the payload tones read, -1 where not comparable
    states: tuple[int, ...]         #: generator states that emit them  [VF.payload_states]
    best_call: str                  #: the panel callsign its payload comes nearest
    best_tones: int                 #: how many of the payload's tones that one has

    @property
    def heard(self) -> int:
        """Payload tones the receiver delivered clear."""
        return sum(1 for t in self.payload if t >= 0)


# --- connect-response found on the receive stream (step 2) ------------------ #
# Grid offsets of a connect-response's symbols on the _ACK_GRID lattice, and the
# number of lattice points one burst covers.
_RESP_NSYM = len(VF.CONNECT_RESPONSE.preamble) + VF.CONNECT_RESPONSE.n_payload
_RESP_OFF = np.rint((np.arange(_RESP_NSYM) * MK.HOP + MK._WOFF)
                    / _ACK_GRID).astype(np.int64)
_RESP_SPAN = int(_RESP_OFF[-1])
# Audio gathered before the search runs again. Each pass re-transforms the NFFT-1
# samples straddling its end, so a short block pays that overhead over and over;
# half a second keeps it under 5% and still reads a burst within ~0.5 s of its
# last symbol, against the up-to-6 s a bracket waits for its gate to close.
_STREAM_BLOCK = MK.FS // 2
# ...and the grid a buffer emptied by our own keying is scanned on until the peer's
# answer to that keying has had its chance to arrive whole.
#
# THE TURNAROUND IS NOT A PLACE FOR HALF-SECOND QUANTISATION. Two stock 4.9.0s
# answer each other's bursts 0.075-0.126 s after the last sample [vara_frames,
# SESSION_TURN_RELEASE_RESPONDER], so a scan due one whole block behind the frame
# cannot put our answer inside the window the peer is listening in, however early
# the first scan comes. On 2026-08-29 that cost 1.10 s of the 1.31-1.36 s this
# station took to answer a gateway's handover, and six greetings never resumed.
#
# A quarter block holds the wait under the peer's own turnaround at four scans
# spent per changeover, and only across the one block where an answer to a burst
# of ours can be  [see _next_scan].
_TURNAROUND_STEP = _STREAM_BLOCK // 4
# ...and the over route's, which is finer again. An over's last sample IS the
# peer's unkey, so the scan that takes it cannot come before the frame ends and
# every sample of grid behind it is dead time in front of our answer: swept over
# where our own keying can leave the cursor, a quarter block answers three real
# gateway overs at +0.037, +0.087 or +0.137 s decided by nothing else, while the
# peer keys its own repeat 1.43-1.65 s later against an answer 1.37 s long.
# Under the 0.02 s the transport polls the stream on, so the poll is the grid in
# a live session and this is only the bound on what a caller feeding smaller
# pieces can spend: 9 ms of alignment per scan, 4-11 of them per turnaround.
_OVER_TURNAROUND_STEP = _STREAM_BLOCK // 16
# How far behind a buffer's newest audio the peer's answer to our turn-request may
# be named. A frame that closed stays fittable after the peer has keyed again —
# the fit is allowed a start outside the audio and the carry keeps just under one
# frame — so nothing in the waveform separates the answer to our request from an
# older burst of the peer's, and what does is its age. Over the 222 grants of the
# 2026-09-03/04 bench, the 210 whose DATA over the peer then read were named
# 0.00-0.37 s behind the newest audio and the twelve whose over it did not read
# 0.91-2.33 s. One block is the longest a frame complete in the buffer can wait
# for the scan that names it  [see _next_scan, _stream_grant].
_GRANT_FRESH_S = _STREAM_BLOCK / MK.FS


def _next_scan(seen: int, first: int, step: int = _TURNAROUND_STEP) -> int:
    """Length the buffer must reach before the scan after this one.

    ``seen`` is the audio the buffer has taken since our own keying emptied it,
    which is its own length: every carry here is longer than the turnaround window
    below, so nothing is trimmed away while that window is open.
    """
    return seen + (step if seen < first + _STREAM_BLOCK else _STREAM_BLOCK)


# Grid offsets of a bare payload's fifteen symbols  [see _peer_answer].
_PAY_OFF = _RESP_OFF[:VF.CONNECT_RESPONSE.n_payload]

# The three lower-speed offers, alongside the canonical level-4 descriptor.
# Stock-caller replay on 2026-09-20 establishes speed selection, not rejection:
# BW2300 positions 14..17, BW2750 0..3, BW500 21..24 select host levels 1..4.
_ANSWER_LATTICE = (14, 15, 16)
_ANSWER_LATTICE_BY_BW = {"2300": _ANSWER_LATTICE, "2750": (0, 1, 2), "500": (21, 22, 23)}
#: One payload burst's span in seconds — the plateau a single burst is read over,
#: and how far apart two detections have to be to be two bursts.
_ANSWER_SPAN_S = (_PAY_OFF[-1] * _ACK_GRID) / MK.FS

# --- the link's own frequency, between our bins -------------------------------- #
# EVERY STATION ON HF IS SOMEWHERE BETWEEN OUR BINS. Dial error, a drifting
# reference and Doppler all land in the same place, and 23.4375 Hz of carrier
# spacing makes half a bin about 12 Hz — inside what a correctly tuned pair of
# radios differ by on an ordinary day. A single-tone burst shrugs that off,
# because the peak bin still wins below half a bin and every one of this file's
# payload readers is single-tone. A two-tone symbol does not: its second carrier
# splits its energy over two bins and the neighbour takes the slot, which reads
# as a preamble the peer got wrong. K0SI transmitted 8 Hz low of us on
# 2026-09-18 and its 11-symbol control burst held for 0 alignments on our grid
# and 35 half a bin below it, while its single-tone frames read 16 of 16 in the
# same turnarounds — one gateway readable one way and deaf the other, for want
# of a number every gateway needs.
#
# So the grid is measured once and everything is read on it, rather than each
# reader searching either side of its own. A measurement costs no alignments:
# one grid, one hypothesis per reader, exactly as at zero. A search costs three
# times the alignments a recording gets to hold on, against thresholds swept at
# zero offset only.
#
# Symbols a read must have confirmed before its residual is allowed to move the
# grid. Six is half a connect-response's payload and the whole of the shortest
# single-tone frame here worth taking one from; below it a median is one or two
# windows and a fade decides the link's frequency.
_OFFSET_MIN_TONES = 6
# How far toward each reading the grid moves. Measured over the two K0SI arms of
# 2026-09-18 — fifteen clean single-tone reads across 71 s and 102 s — the
# station's offset sits at -0.35 of a bin and every individual read lands within
# 0.06 of that, with no trend either way: a value fixed at the connect would have
# served both sessions. What it would NOT have served is the connect itself,
# where both sessions read shallowest (-0.315 and -0.333, against -0.41 later in
# the same minute), so a grid pinned there starts a tenth of a bin short of the
# frames it is for. Half-way to each new reading tracks that out in two frames,
# halves the scatter of any one of them, and leaves a wrong read worth half a
# grid rather than a whole one — which matters on a band where the thing being
# tracked really does move.
_OFFSET_SLEW = 0.5
# How far the grid has to have moved before the transcript says so, and says so
# again. An eighth of a bin is 3 Hz, under what any reader here notices and over
# what a clean read's own scatter reaches, so the line appears when the peer is
# genuinely off our frequency and not once per frame after that.
_OFFSET_SAY = 0.125


def _top3_track(x: np.ndarray, grid: int = _ACK_GRID,
                band: tuple[int, int] | None = None, *,
                bin_offset: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three strongest tone-band bins at every ``grid``-spaced offset in ``x``,
    each outside the Hann main lobe of the ones above it, the two ratios between
    them in dB, and where the strongest one really sits between bins.

    :func:`vara_mfsk.demod_tones` reads symbol *k* of a burst at a fixed offset
    from the burst start, so a scan over candidate burst starts re-transforms the
    same windows once per symbol — twenty-three times over, for a connect-response.
    Transforming each window once and indexing into the result gives the same
    tones for every hypothesis at a fraction of the cost, which is what makes the
    widened search below affordable. Band-limited to the tone alphabet for the
    same reason :func:`vara_mfsk.demod_tones` is: an unbounded peak reads a
    received tone's own 3f image, not the tone. ``band`` is the session's
    [vara_mfsk, band_for]; omitted, it is BW2300's.

    One transform serves both families: the single-tone bursts read column 0, the
    two-tone connected-ack [spec 04 §4.2C] reads both, and an initiator waiting
    out its link-setup is looking for one of each on the same audio.

    The clearances fall out of the same peaks and cost one more log apiece. They
    are what :data:`_CLEAR_DB` is read against, and each search reads the one that
    answers its own question: an argmax always returns a bin, so a single-tone
    burst asks whether the peak stood clear of the next one (column 0), while a
    two-tone symbol emits its carriers at equal amplitude and asks instead whether
    the SECOND of them stood clear of the rest of the band (column 1). Reading the
    first ratio for a pair would score every clean two-tone symbol as noise.

    THE RESIDUAL IS A MEASUREMENT AND NOT A SEARCH, which is the whole of why this
    file is allowed to act on it. A Hann window's main lobe spans three bins and
    its shape is known, so a tone at ``b + d`` leaves ``2(R - L) / (L + 2C + R)``
    over those three magnitudes equal to ``d`` — to better than 0.001 of a bin on
    a clean tone, at every ``d`` in the half-bin either side. It reads against
    whatever grid the transform was taken on, so a track taken at ``bin_offset``
    reports how much further the signal still is from there, and a reader can
    close on a peer's real frequency instead of hoping it lands on a bin
    [see :meth:`VaraStationHandshake._note_peer_offset`].
    """
    lo, hi = band or (MK.BIN_LO, MK.BIN_HI)
    n = len(x) - MK.NFFT + 1
    if n <= 0:
        return (np.zeros((0, 3), dtype=np.int32), np.zeros((0, 2)),
                np.zeros(0))
    starts = np.arange(0, n, grid)
    win = np.hanning(MK.NFFT)
    # Evaluate the same bins on an adjacent fractional grid. Returned indices
    # stay on the integer lattice; physical frequency is index + bin_offset.
    # The ordinary path keeps its real FFT and its existing output unchanged.
    if bin_offset:
        win = win * np.exp(-2j * np.pi * bin_offset
                           * np.arange(MK.NFFT) / MK.NFFT)
    span = np.arange(MK.NFFT)
    out = np.empty((len(starts), 3), dtype=np.int32)
    clear = np.empty((len(starts), 2))
    resid = np.empty(len(starts))
    for i in range(0, len(starts), 512):          # bounded working set
        blk = x[starts[i:i + 512, None] + span] * win
        spectrum = (np.fft.fft(blk, axis=1) if bin_offset
                    else np.fft.rfft(blk, axis=1))
        full = np.abs(spectrum)
        mag = full[:, lo:hi + 1]                  # a view; zeroed below
        cols = np.arange(mag.shape[1])
        rows = np.arange(len(blk))
        peak = []
        for k in range(3):
            j = np.argmax(mag, axis=1)
            out[i:i + len(blk), k] = lo + j
            if k == 0:                            # before the skirts are zeroed
                b = lo + j
                left, mid, right = (full[rows, b - 1], full[rows, b],
                                    full[rows, b + 1])
                resid[i:i + len(blk)] = (2 * (right - left)
                                         / (left + 2 * mid + right + 1e-20))
            peak.append(mag[rows, j])
            mag[np.abs(cols[None, :] - j[:, None]) <= MK._PAIR_SKIRT] = 0
        for k in range(2):
            clear[i:i + len(blk), k] = 20 * np.log10(
                (peak[k] + 1e-20) / (peak[k + 1] + 1e-20))
    return out, clear, resid


def _tone_track(x: np.ndarray, grid: int = _ACK_GRID) -> np.ndarray:
    """Dominant tone at every ``grid``-spaced offset in ``x``."""
    return _top3_track(x, grid)[0][:, 0]


def _live_track(x: np.ndarray, grid: int = _ACK_GRID) -> np.ndarray:
    """Was the receiver hearing the band in each analysis window of ``x``?

    One flag per entry of :func:`_tone_track`, from the window's own RMS. A
    transmitting rig mutes its own receive audio and this station's stays muted
    for 167-180 ms after the last transmitted sample — 420 ms until the 250 ms
    hold past the last sample came out on 2026-08-06 — so the samples exist and
    the band is not in them, and a tone read there is not the gateway getting one
    wrong. Cumulative sums, because the mask has to cost nothing beside the
    transform.

    This is the coarse half of comparability and it only catches the windows with
    nothing in them at all. :data:`_CLEAR_DB` is the other half, and it is the one
    that catches the ramp out of the mute.
    """
    p = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    n = len(x) - MK.NFFT + 1
    if n <= 0:
        return np.zeros(0, dtype=bool)
    i = np.arange(0, n, grid)
    rms = np.sqrt(np.maximum(p[i + MK.NFFT] - p[i], 0.0) / MK.NFFT)
    return 20 * np.log10(rms + 1e-15) > _DEAF_DBFS


def _pair_track(x: np.ndarray, grid: int = _ACK_GRID) -> np.ndarray:
    """The connected-ack's two carriers at every offset, sorted so a comparison
    need not care which of them came out on top  [spec 04 §4.2C]."""
    return np.sort(_top3_track(x, grid)[0][:, :2], axis=1)


def _ack_held(bins: np.ndarray, clear: np.ndarray, live: np.ndarray,
              shift: int = 0) -> np.ndarray:
    """Where in a tone-pair track the connected-ack's fixed preamble stands
    [spec 04 §4.2C]: one flag per lattice offset.

    Offsets index preamble symbol 1, not the burst start — see ``_ACK_MATCH``.

    A carrier is compared only where it was delivered and stood clear. Both halves
    of comparability apply here exactly as they do to a connect-response's payload
    [see :data:`_CLEAR_DB`], and this reader needs them more: a two-tone symbol's
    second carrier is a second argmax, so where one carrier of the pair is under
    the band the reader still returns a bin for it, and the symbol was being scored
    as a preamble the peer got wrong rather than as one this receiver did not
    deliver. Every ack refused on 2026-08-19 was refused that way.

    So a symbol contributes two carriers when the pair stands :data:`_CLEAR_DB`
    clear of the rest of the band, one when only its stronger carrier does, and
    none through a dead window. An alignment holds when at least
    :data:`_ACK_MIN_CARRIERS` carriers were comparable and every comparable one is
    the preamble's — the same shape as :func:`_answers`, which is what keeps a fade
    from being read as a contradiction without letting an unread band be read as
    agreement.

    A shared channel takes the second carrier the other way, and that is a third
    case rather than either of those two. The pair a symbol reads is the two
    strongest bins of the band, so a station louder than one of the ack's own
    carriers takes its slot: the pair then stands clear, holds one carrier that IS
    the preamble's, and was being scored as two comparable carriers and two wrong.
    It reads as a contradiction where the truth is one carrier delivered and one
    occupied, and the occupied one says nothing either way — so it counts as one
    carrier, confirmed. :data:`_ACK_MAX_CROWDED` is what bounds it.

    The first of the three has to be one of them, and that is what holds the
    relaxation at one symbol. ``_ACK_MATCH`` has already conceded the leading
    preamble symbol to the mute; conceding the one behind it too leaves the burst
    unpinned in time, and a stretch of audio carrying only the last two symbols
    would be accepted however clean those two are. A notch inside the burst is the
    other case and it is not the same one: there the opening symbol still says
    where the burst began, and the corpus fade sweep drops a symbol out of the
    middle without losing the ack.

    ``shift`` is the peer's measured tuning offset in carriers, never a search:
    see ``VaraStationHandshake._peer_shift`` for why this one is read off the
    connect-response rather than looked for here.
    """
    n = max(len(bins) - _ACK_SPAN, 0)
    idx = np.arange(n)[:, None] + _ACK_MATCH_OFF               # (n, 3)
    want = _ACK_MATCH + shift
    pair = np.sort(bins[idx][:, :, :2], axis=2)                # (n, 3, 2)
    top = bins[idx][:, :, 0]
    c12, c23 = clear[idx][:, :, 0], clear[idx][:, :, 1]
    on = live[idx]
    both = on & (c23 >= _CLEAR_DB)
    lone = on & ~both & (c12 + c23 >= _CLEAR_DB)
    found = (pair[:, :, :, None] == want[None, :, None, :]).any(3).sum(2)
    crowded = both & (found == 1)
    per_symbol = np.where(both, np.where(crowded, 1, 2), lone.astype(np.int64))
    confirmed = np.where(both, found,
                         (lone & ((top == want[:, 0])
                                  | (top == want[:, 1]))).astype(np.int64))
    comparable = per_symbol.sum(1)
    return ((per_symbol[:, 0] > 0) & (comparable >= _ACK_MIN_CARRIERS)
            & (confirmed.sum(1) == comparable)
            & (crowded.sum(1) <= _ACK_MAX_CROWDED))


def _widest(held: np.ndarray) -> tuple[int, int]:
    """Start and width of the widest run of True in ``held``, or ``(0, 0)``."""
    ok = np.concatenate([[False], held, [False]])
    edge = np.flatnonzero(np.diff(ok.astype(np.int8)))
    if not len(edge):
        return 0, 0
    start, stop = edge[::2], edge[1::2]
    i = int(np.argmax(stop - start))
    return int(start[i]), int(stop[i] - start[i])


def _ack_hold(bins: np.ndarray, clear: np.ndarray, live: np.ndarray,
              shift: int = 0) -> int:
    """Width of the widest run of offsets :func:`_ack_held` holds over."""
    return _widest(_ack_held(bins, clear, live, shift))[1]


def _ack_plateau(x: np.ndarray, shift: int = 0, track: tuple | None = None,
                 band: tuple[int, int] | None = None,
                 bin_offset: float = 0.0) -> int:
    """:func:`_ack_hold` over raw audio, or over a transform already taken of it."""
    x = np.asarray(x, dtype=np.float64)
    bins, clear, _ = (_top3_track(x, band=band, bin_offset=bin_offset)
                      if track is None else track)
    return _ack_hold(bins, clear, _live_track(x), shift)


def _band_power(x: np.ndarray, band: tuple[int, int], bin_offset: float = 0.0,
                grid: int = _ACK_GRID) -> np.ndarray:
    """Power in every tone-alphabet bin at every ``grid``-spaced offset in ``x``.

    :func:`_top3_track` keeps the three strongest bins of each window, which is
    all a reader comparing argmaxes can use. A reader that knows which bins it
    expects wants the magnitudes themselves, including the ones a stronger
    neighbour would have hidden — that is the whole of what makes the score
    below soft — so this keeps the band and throws the ranking away.

    Same windows, same Hann taper and the same fractional grid as the track, so
    the two read one signal the same way  [see :func:`_top3_track`].
    """
    lo, hi = band
    n = len(x) - MK.NFFT + 1
    if n <= 0:
        return np.zeros((0, hi - lo + 1))
    starts = np.arange(0, n, grid)
    win = np.hanning(MK.NFFT)
    if bin_offset:
        win = win * np.exp(-2j * np.pi * bin_offset
                           * np.arange(MK.NFFT) / MK.NFFT)
    span = np.arange(MK.NFFT)
    out = np.empty((len(starts), hi - lo + 1))
    for i in range(0, len(starts), 512):          # bounded working set
        blk = x[starts[i:i + 512, None] + span] * win
        spectrum = (np.fft.fft(blk, axis=1) if bin_offset
                    else np.fft.rfft(blk, axis=1))
        out[i:i + len(blk)] = np.abs(spectrum[:, lo:hi + 1]) ** 2
    return out


def _burst_match(x: np.ndarray, tones, band: tuple[int, int], shift: int = 0,
                 bin_offset: float = 0.0) -> float:
    """How well ``x`` carries the two-tone frame ``tones``, at its best alignment.

    One number per lattice offset and the largest of them, in nats per symbol.
    For each symbol the expected pair's share of its window's band power is put
    against the ``2/N`` a flat band would leave there; the frame's score is the
    mean of those logs. Band noise sits at 0, a symbol whose pair took the whole
    band at ``log(N/2)``, and everything real in between.

    Nothing here is a gate and nothing here is a search. Symbols are not scored
    against a clearance and not dropped for failing one, so a fade or a mute
    costs the frame that symbol's share of the evidence and no more, and one
    louder station on one carrier costs half of one. The alignment is the only
    free parameter and the frame's own span pins it: ``tones`` is eleven symbols
    over 0.47 s, which chance does not walk into  [see :data:`_BURST_MATCH`].
    """
    tones = np.asarray(tones, dtype=np.int64)
    offs = np.rint((np.arange(len(tones)) * MK.HOP + MK._WOFF)
                   / _ACK_GRID).astype(np.int64)
    mag = _band_power(np.asarray(x, dtype=np.float64), band, bin_offset)
    nb = band[1] - band[0] + 1
    n = max(len(mag) - int(offs[-1]), 0)
    if n <= 0:
        return 0.0
    idx = np.arange(n)[:, None] + offs
    cols = np.clip(tones + shift - band[0], 0, nb - 1)
    pair = mag[idx[:, :, None], cols[None, :, :]].sum(2)
    ratio = nb * pair / (mag.sum(1)[idx] + 1e-30)
    return float(np.log(np.maximum(ratio, 1e-9) / 2.0).mean(1).max())


def _cont_held(bins: np.ndarray, clear: np.ndarray, live: np.ndarray,
               shift: int = 0) -> np.ndarray:
    """One flag per lattice offset where all eight continue symbols stand.

    Every symbol must be live and its pair at least :data:`_CLEAR_DB` clear, and
    the lead must be the fixed opening pair exactly. A peer whose carriers sit
    between our bins rounds that lead to the wrong pair and is read on its own
    grid instead, which is a measurement of the link and not a tolerance here
    [see :meth:`VaraStationHandshake._note_peer_offset`].
    """
    n = max(len(bins) - _CONT_SPAN, 0)
    idx = np.arange(n)[:, None] + _CONT_OFF
    pair = np.sort(bins[idx][:, :, :2], axis=2)
    want = _ACK_PRE[0] + shift
    both = live[idx] & (clear[idx][:, :, 1] >= _CLEAR_DB)
    return both.all(1) & (pair[:, 0] == want).all(1)


def _cont_plateau(x: np.ndarray, shift: int = 0, track: tuple | None = None,
                  band: tuple[int, int] | None = None,
                  bin_offset: float = 0.0) -> int:
    """Width of the widest run of offsets :func:`_cont_held` holds over."""
    x = np.asarray(x, dtype=np.float64)
    bins, clear, _ = (_top3_track(x, band=band, bin_offset=bin_offset)
                      if track is None else track)
    return _widest(_cont_held(bins, clear, _live_track(x), shift))[1]


def _ack_lock(x: np.ndarray, shift: int = 0) -> int | None:
    """Sample at which a connected-ack's first symbol begins in ``x``, or None.

    :func:`_ack_plateau` answers whether the preamble stands and how widely;
    reading the seven state symbols behind it needs where as well. The lattice
    indexes preamble symbol 1 [see ``_ACK_MATCH``], so the burst opens one symbol
    and one analysis-window offset ahead of the alignment that held — and the
    plateau's midpoint is the alignment furthest from either edge of it.

    A burst with less than a symbol of audio in front of it has nowhere to be
    located from and returns None: pass the samples before key-up as well.
    """
    x = np.asarray(x, dtype=np.float64)
    bins, clear, _ = _top3_track(x)
    start, wide = _widest(_ack_held(bins, clear, _live_track(x), shift))
    if wide < _ACK_PLATEAU:
        return None
    a = (start + wide // 2) * _ACK_GRID - (MK.HOP + MK._WOFF)
    return a if a >= 0 else None


def _answers(m, c):
    """Do ``m`` confirmed tones of ``c`` comparable ones say the gateway answered?

    Either route on its own: thirteen tones keyed to the callsign we dialled, or a
    comparable set of at least :data:`_RESP_MIN_HEARD` with nothing in it wrong.
    Scalars or arrays; one definition, because the search that ranks hypotheses and
    the two call sites that judge the winner have to mean the same thing by it. They
    did not, and a clean sweep the rule would have taken was being beaten to the one
    slot per pass by an alignment scoring higher and accepting nothing.
    """
    return (m >= _RESP_MIN_TONES) | ((c >= _RESP_MIN_HEARD) & (m == c))


def _best_fit(heard: np.ndarray, comparable: np.ndarray, tones: np.ndarray,
              npre: int) -> tuple[int, int, int, int]:
    """``(row, shift, confirmed, comparable)`` of the hypothesis that fits best.

    A hypothesis is an alignment — one row of ``heard``, holding the tones that
    alignment reads and masked by the symbols the receiver actually delivered —
    together with a tone shift from :data:`_RESP_SHIFTS`, which is what a peer
    tuned off frequency does to all of them at once.

    Ranked by :func:`_answers` first, then most tones confirmed, then fewest left
    unconfirmed, across alignments and shifts alike: an answer that arrived late
    and one that arrived off frequency are the same kind of hypothesis and are
    chosen between on the same evidence.

    The preamble is carried but not scored, because for the connect-response it is
    precisely the part that goes missing when the answer overlaps our own
    transmission.
    """
    hd, ok, want = heard[:, npre:], comparable[:, npre:], tones[npre:]
    c = ok.sum(1)
    best = (0, 0, -1, 0)
    for shift in _RESP_SHIFTS:
        m = ((hd == want + shift) & ok).sum(1)
        r = int(np.lexsort((c - m, -m, ~_answers(m, c)))[0])
        rank = (bool(_answers(m[r], c[r])), int(m[r]), int(m[r] - c[r]))
        if rank > (bool(_answers(best[2], best[3])), best[2], best[2] - best[3]):
            best = (r, shift, int(m[r]), int(c[r]))
    return best


def _payload_fit(x: np.ndarray, kind: VF.BurstKind, tones: np.ndarray,
                 track: tuple | None = None,
                 band: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
    """:func:`_payload_alignment` and the sample offset into ``x`` it put the
    burst's first symbol at.

    The offset is negative where the fit runs off the front of ``x`` and past
    ``len(x)`` where it runs off the back — both are ordinary, for the reason the
    tones themselves may be incomparable — so a caller wanting the audio held past
    the burst subtracts ``offset + _span(kind)`` and accepts either sign  [see
    :meth:`VaraStationHandshake._peer_responder_release`].
    """
    n_sym = len(tones)
    npre = len(kind.preamble)
    bins, clear, _ = _top3_track(x, band=band) if track is None else track
    track, clear = bins[:, 0], clear[:, 0]
    lo, hi = -(n_sym - 1) * MK.HOP, len(x) - MK.STRIDE
    if len(track) == 0 or hi < lo:
        return np.full(n_sym, -1, dtype=np.int32), 0
    s = np.arange(lo, hi + 1, _ACK_GRID)
    gi = np.rint((s[:, None] + np.arange(n_sym) * MK.HOP + MK._WOFF)
                 / _ACK_GRID).astype(np.int64)
    j = np.clip(gi, 0, len(track) - 1)
    ok = (gi >= 0) & (gi < len(track)) & (clear[j] >= _CLEAR_DB)
    heard = np.where(ok, track[j], -1).astype(np.int32)
    row, shift, _m, _c = _best_fit(heard, ok, tones, npre)
    return (np.where(heard[row] >= 0, heard[row] - shift, -1).astype(np.int32),
            int(s[row]))


def _payload_alignment(x: np.ndarray, kind: VF.BurstKind, tones: np.ndarray,
                       track: tuple | None = None,
                       band: tuple[int, int] | None = None) -> np.ndarray:
    """The tones ``x`` carries at the alignment that best fits ``tones``, read back
    at zero frequency offset whatever offset they arrived on.

    One entry per symbol of the burst, ``-1`` where that symbol's analysis window
    falls outside the audio, or read a tone that never stood :data:`_CLEAR_DB`
    clear of the band — so a caller reads ``heard == tones`` for the tones
    confirmed and ``heard >= 0`` for the tones that were comparable at all.

    A hypothesis is a single number: the virtual burst start ``s``, with symbol *k*
    read at ``s + k*HOP``. ``s`` is allowed to be negative, because the piece the
    segmenter hands us may be the tail of a burst whose head it already cut away —
    or, when the peer starts answering before we have finished transmitting, of a
    burst whose head our own receiver was muted through. A symbol outside the audio
    is not comparable rather than wrong, which is what lets those two cases be
    scored at all.

    ``track`` is a :func:`_top3_track` already taken of ``x``. Several of these
    frames may be looked for in one piece of audio and the transform is the whole
    cost of doing it  [see :meth:`VaraStationHandshake._stream_answer`].
    """
    return _payload_fit(x, kind, tones, track, band)[0]


# Map burst symbol-count -> BurstKind (counts are distinct: 41/23/16/32/17)
# [spec 04 §4.2.2, spec 05 §5.3.3, §5.6]. The connected-ack is deliberately absent:
# it is not a member of this family and is recognised by :func:`_ack_plateau`.
# The 16-symbol disconnect-final shares SESSION_CONFIRM's count; on_rx_tones
# separates them by what the tones say, in the one state both can arrive in.
_NSYM = {len(k.preamble) + k.n_payload: k
         for k in (VF.CR, VF.CONNECT_RESPONSE, VF.SESSION_CONFIRM,
                   VF.SESSION_KEEPALIVE_A, VF.SESSION_TURN_RELEASE)}
# The two keepalives share a symbol count; _NSYM resolves to A and the recognizer
# falls through to B  [spec 05 §5.3.3].
#
# The three handshake pairs keep one another's counts exactly — every request is
# 41 and every response 23 — so a symbol count cannot separate them on its own;
# they differ in the generator state behind the preamble, and at BW500 in the
# alphabet. The session bandwidth supplies what the count cannot, and
# :func:`_kind_for` is where the two meet. The session frames are the SAME kinds
# at every bandwidth — one descriptor read over that bandwidth's alphabet
# [vara_frames, for_bw] — so the tables carry them unchanged and only the
# handshake pair is swapped out. They hold the CANONICAL kinds so the ``is``
# comparisons on_rx_tones runs against what it received go on holding.
_NSYM_BW = {bw: {**_NSYM, **{len(k.preamble) + k.n_payload: k
                             for k in pair.values()}}
            for bw, pair in VF.HANDSHAKE_BY_BW.items()}


#: Every kind a symbol count can stand for, for the reader that names a bracket
#: by recognising it rather than by dividing its length  [see on_rx_audio]. The
#: counts collide — the two keepalives share 32, the confirm and the
#: disconnect-final share 16 — and the callsign is what separates them.
_CANDIDATES = {}
for _k in (VF.CR, VF.CONNECT_RESPONSE, VF.SESSION_CONFIRM,
           VF.SESSION_DISCONNECT_FINAL, VF.SESSION_KEEPALIVE_A,
           VF.SESSION_KEEPALIVE_B, VF.SESSION_TURN_RELEASE,
           VF.SESSION_TURN_RELEASE_RESPONDER):
    _CANDIDATES.setdefault(len(_k.preamble) + _k.n_payload, []).append(_k)
_CANDIDATES_BW = {
    bw: {n: [pair.get(k, k) for k in ks] for n, ks in _CANDIDATES.items()}
    for bw, pair in VF.HANDSHAKE_BY_BW.items()}


def _candidates_for(bw: str) -> dict:
    """The symbol counts this bandwidth can carry, and what each can stand for."""
    return _CANDIDATES_BW.get(bw, _CANDIDATES)


def _padded(x: np.ndarray, n_sym: int) -> np.ndarray:
    """``x``, zero-extended to hold ``n_sym`` symbols if it is short of them."""
    need = (n_sym - 1) * MK.HOP + MK.STRIDE
    return x if len(x) >= need else np.concatenate([x, np.zeros(need - len(x))])


def _kind_for(n_sym: int, bw: str) -> VF.BurstKind | None:
    """The handshake burst a station running at BW keys with N_SYM symbols."""
    return _NSYM_BW.get(bw, _NSYM).get(n_sym)

CALLER_UNKNOWN = "UNKNOWN"      # mfsk_only responder: no link-setup ever arrives

# Which of the two continue-class frames answers an INTERMEDIATE over
# [see :meth:`VaraStationHandshake._tx_over_response`]. Both continue a delivery
# at the bench and neither ever ends one, so what separates them is where they
# came from — and that is why the setting exists rather than a constant.
#
# GENERATED is built per callsign from the solved lattice, so it is correct for a
# station nobody has recorded, and it is the only continue-class answer that has
# drawn a further over out of a real gateway: three overs at two gateways,
# KB3AC-10 twice on 2026-08-31 and VE3KPG on 2026-09-03. CAPTURED is a copy of
# one link's tail with seven unpinned state symbols behind its lead; it has a
# bench and no gateway, 0 of 2 on the air.
#
# This air evidence originally selected GENERATED for a session's first
# delivery at BW2300; it remains explicitly selectable. A gateway keys its next over 0.15-0.25 s
# after the generated frame's last symbol — K0SI, KE8LVA and VE3KPG, off their
# own tapes — so it waits the frame out, reads it and answers it; the captured
# burst, keyed at VE3KPG on the same channel six minutes earlier at the same
# +0.16 s, read back 8 of 8 at an independent receiver and drew the idle cadence.
#
# AFTER A HANDOVER THE ANSWER IS ITS OWN SETTING  [see
# VaraStationHandshake._tx_over_response]. The first over a gateway keys after
# taking the turn back drew the idle cadence from the generated frame five times
# at three gateways on 2026-09-06, the idle keyed at our last symbol, where the
# next over comes 0.15-0.25 s after it before a handover. At the bench the phase
# is measurable: a stock 4.9.0 responder waits the 1.366 s frame out during its
# first delivery and re-keys 1.37-1.43 s after its own unkey once it is sending
# again, before the frame has ended, eight runs of eight. So the frame that
# answers there has to end early, and two do. CAPTURED ends at +0.5 s and
# carried six consecutive post-handover overs at the bench, 732 bytes
# byte-exact, in every fetch of 2026-09-03/04 — but it is the copied constant
# VE3KPG refused. SHORT is the 32-symbol frame's 16-symbol sibling on the same
# lattice (SEED_OFF 61, the callsign-keyed family every gateway reads), off this
# station's own tape at NS0A; it ends at about +0.85 s and carried the same six
# overs byte-exact on 2026-09-07. It is the default because the one thing the
# air has said about content favours the lattice family; no gateway has been
# keyed either frame after a handover.
#
# AT BW500 THE SAME LENGTH DECIDES THE OTHER WAY, and the default follows the
# bandwidth. A BW500 sender re-keys 0.520-0.556 s after its own unkey on every
# intermediate over, so the generated frame's last sample would land 0.9 s
# inside the sender's next transmission; the 8-symbol burst clears the re-key
# by 0.073-0.104 s, and it is what a 65-byte delivery completed behind, 65 of
# 65 byte-exact with the reply drawn  [spec 05 §5.5].
OVER_CONTINUE_GENERATED = "generated"
OVER_CONTINUE_CAPTURED = "captured"
OVER_CONTINUE_SHORT = "short"
OVER_CONTINUE_ANSWERS = (OVER_CONTINUE_GENERATED, OVER_CONTINUE_CAPTURED,
                         OVER_CONTINUE_SHORT)
#: What the log calls each of them, so the operator reads the frame and not the
#: setting  [see VaraStationHandshake._tx_continue_answer].
OVER_CONTINUE_NAMED = {
    OVER_CONTINUE_GENERATED: "generated 32-symbol session-over-response",
    OVER_CONTINUE_CAPTURED: "captured 8-symbol continue burst",
    OVER_CONTINUE_SHORT: "generated 16-symbol session-over-response-short",
}
#: What answers an intermediate over once the turn has been ours and gone back,
#: when nobody picked one.
OVER_CONTINUE_AFTER_DEFAULT = OVER_CONTINUE_SHORT


#: Initial continuation choices have separate bandwidth evidence. BW500's
#: 0.52-0.56 s repeat window needs the 0.341 s captured burst. At BW2750 the
#: captured tail worked for a prior W1AW/267-byte bench greeting, but the exact
#: KC9GHZ/225-byte greeting required idle/re-ACK recovery on both full frames.
#: Its generated 16-symbol answer advances those frames immediately and takes
#: half the generated 32-symbol frame's duration. The captured
#: tail's dependence on callsign/session state remains unproved; this changes
#: the measured greeting latency, not the diagnosis of live outbound silence.
#: At BW2300 the exact greeting also advances directly with generated16; with
#: generated32 its late re-ACK overlapped the next stock DATA on the host clocks
#: and only 136 of 225 greeting bytes reached the caller. Explicit generated32
#: retains the prior real-gateway initial-delivery option.
def over_continue_default(bw: str) -> str:
    """The initial continuation answer unless the caller explicitly selects one."""
    if bw in ("2300", "2750"):
        return OVER_CONTINUE_SHORT
    return OVER_CONTINUE_CAPTURED if bw == "500" else OVER_CONTINUE_GENERATED


class VaraIO:
    """Sink the handshake drives; the modem implements it over PHY + host API."""
    # The live transport's tx suppression, made readable. `AudioVaraIO` decides at
    # key-up whether the transmitter came up (`_refused`) and stops calling on a
    # rig whose unkey never confirmed (`rig.retired`); its `tx()` plays nothing
    # while either stands, and `tx()` itself returns None either way. These class
    # defaults say the rest: a transport without a rig transmits everything.
    rig = None
    _refused = False

    # A keyed region is open on the channel right now, so anything the idle
    # cadence keys lands on top of it. False on a transport that cannot tell.
    receiving = False

    def key(self, on: bool) -> None: ...             # PTT ON/OFF around a burst
    def tx(self, samples: np.ndarray) -> None: ...   # transmit one burst of audio
    def pending(self) -> None: ...                   # host PENDING notification
    def connected(self, caller: str, called: str, bw: str) -> None: ...
    def log(self, msg: str) -> None: ...

    def tx_went_out(self) -> bool:
        """Whether the burst just handed to :meth:`tx` reached the transmitter.

        Read it inside the keyed region it belongs to — after ``tx()``, before
        ``key(False)`` — because ``_refused`` is decided afresh at every key-up
        and an unkey that fails can retire the rig only after the burst played."""
        return not self._refused and not (self.rig is not None and self.rig.retired)

    def data(self, payload: bytes) -> None:
        """Payload decoded off a peer's DATA over -> the host's data port.

        The default reports rather than discards: a transport that never
        implemented this would otherwise drop received mail without a trace,
        and silence is indistinguishable from a gateway that sent nothing."""
        self.log(f"[unrouted] {len(payload)} decoded payload bytes: no data sink")


class VaraStationHandshake:
    """Transport-agnostic VARA connect-handshake driver (one station)."""

    def __init__(self, mycalls: Sequence[str], io: VaraIO,
                 bw: str = "2300", mfsk_only: bool = False,
                 panel: Sequence[str] = (),
                 over_continue: str | None = None,
                 over_continue_after: str | None = None,
                 max_link_setups: int = _LINKSETUP_MAX_TX,
                 cr_retry_form: str = "full",
                 allow_data_retries: bool = True,
                 probe_intermediate_query: bool = True,
                 tx_level: int | None = None):
        if tx_level is not None and (bw != "2750" or type(tx_level) is not int
                                     or tx_level not in (1, 2, 3, 4)):
            raise ValueError("tx_level requires BW2750 and host level 1, 2, 3 or 4")
        # The host's level number is one above the wideband index record.
        # An explicit choice fixes DATA geometry; link setup follows the peer's
        # offer independently.
        self.tx_level = tx_level
        self._tx_level = 99 + tx_level if tx_level is not None else _phy.base_level(bw)
        if not isinstance(max_link_setups, int) or max_link_setups < 1:
            raise ValueError("max_link_setups must be a positive integer")
        self.max_link_setups = max_link_setups
        #: Retained for API compatibility. Missing feedback never licenses
        #: DATA retransmission; a decoded NAK does, independently of this flag.
        self.allow_data_retries = allow_data_retries
        #: Bounded stock intermediate-answer recovery; False is retained for
        #: diagnostic callers that deliberately disable solicitation.
        self.probe_intermediate_query = probe_intermediate_query
        self._pending_answer_at = None
        self._over_keyed_at = None
        self._intermediate_query_attempted = False
        self._intermediate_query_attempts = 0
        self._intermediate_query_for = None
        self._intermediate_query_at = 0.0
        self._intermediate_query_samples = 0
        self._matched_intermediate_answer = None
        self._matched_data_nak = None
        self._unclassified_continue = None
        self._confirmed_continue_pairs = set()
        self._data_nak_wait_samples = 0
        #: Which request goes out on a re-key: "full" is our 41-symbol one,
        #: "stock" the 32-symbol form a real caller uses  [VF.connect_request_retry].
        self.cr_retry_form = cr_retry_form
        self.mycalls = [c.upper() for c in mycalls]
        self.io = io
        self.bw = bw
        self._band = MK.band_for(bw)
        self.mfsk_only = mfsk_only
        #: Which continue-class frame answers an intermediate over of the peer's
        #: first delivery; None takes the bandwidth's own  [see
        #: OVER_CONTINUE_ANSWERS, _tx_over_response].
        self.over_continue = over_continue or over_continue_default(bw)
        #: The same, once the turn has been ours and gone back
        #: [see OVER_CONTINUE_AFTER_DEFAULT].
        self.over_continue_after = over_continue_after or OVER_CONTINUE_AFTER_DEFAULT
        # Every station this end could put a name to. It is the published gateway
        # panel where a driver has one, and what it is for is the answer that
        # arrives from somebody we did not dial: an empty panel leaves such a burst
        # reported by its tones alone  [see _unattributed_answer].
        self.panel = [c.upper() for c in panel]
        #: Offers found by the payload-only search, in arrival order.
        self.answers: list[PeerAnswer] = []
        #: Compatibility with older drivers. An offer now sends setup directly;
        #: it never asks the driver to send another connect request.
        self.answer_retry = False
        #: Connect-responses heard that name nobody, in the order they arrived.
        #: Read by whatever reports the attempt; nothing in this class acts on it.
        self.unattributed: list[UnattributedAnswer] = []
        self.state = VaraState.DISCONNECTED
        self.role: str | None = None
        self.step: str | None = None
        self.caller = ""            # our own call (initiator) / learned (responder)
        self.called = ""            # the gateway/destination call
        self._linksetup_tx = 0
        self._setup_level = 4
        # Restarts spent  [see _CONNECT_MAX_RESTARTS]. Not cleared by `originate`,
        # which is also how the CR train re-keys.
        self._cr_restarts = 0
# Bodies of the last DATA over delivered to the host, in emission order. A
        # gateway that missed our per-over response repeats the over; the repeat is
        # answered again but must not reach the host twice. Equality of the WHOLE
        # over is the duplicate test [ours — INFERRED, no retransmission has been
        # captured off air yet]: distinct overs differ at least in the per-frame
        # field of their last body, observed stepping across one real session's
        # overs. Per over rather than per body, because a BW500 over carries two
        # blocks about as often as one and a repeat of it repeats both — tested
        # against a first block that does not equal the second.
        self._delivered: tuple[bytes, ...] = ()   # the last one, for the log
        # Bodies of the DATA overs we have keyed SINCE THE PEER LAST TRANSMITTED.
        # A DATA over names nobody, so this is the only thing that tells our own
        # transmission returning through the rig's monitor from a burst the peer
        # sent: the ninety bytes are ours and we have them. What the link-setup's
        # callsign does for the connect phase [see _peer_data_over], this does for
        # the data phase.
        #
        # The window is what keeps it from refusing real traffic. A body is not
        # ours by nature — it is ours by having just been keyed. Short payloads
        # frame identically in both directions, because arq.phy.vara_body's whole
        # trailer (0x14, the CALLER callsign's CRC-16 high byte, zeros) is the
        # caller's from either end: a gateway echoing `FQ\r` back at us produces
        # byte-for-byte the body we sent, and a set that never emptied refused it
        # for the rest of the session. Bounding it to the last thing we said
        # leaves the filter exactly where the echo it catches can be — our own
        # transmission comes back through the monitor before the peer's next
        # burst, never after it — and lets the byte pattern be reused afterwards.
        self._keyed_bodies: set[bytes] = set()
        # Transmit turn and the queue behind it. A connect starts with the turn at
        # the peer: the gateway speaks first in every session held.
        self.turn = _TURN_PEER
        self._tx_recovery_levels: list[int] = []
        self._txq: list[bytes] = []
        self._tx_pending: tuple[bytes, int, int] | None = None  # body, over, level
        self._tx_retries = 0
        self._reset_final_ack_recovery()
        self._over = 0              # position in the session's preamble stream
        self._query_retry_phase = 0
        self._full_keyed = 0        # full overs keyed in the delivery going out
        self._asked = 0             # turn-requests keyed since the last grant
        self._ask_owed = False      # a send() that arrived mid-burst, still to ask
        self._into_our_turn = 0     # DATA overs keyed into our turn since the last
        #                             one the peer answered  [see _TURN_YIELD_AFTER]
        self._released = False      # our release is on the air and the peer has
        #                             spent nothing of what it was handed
        self._release_owed = False  # the transport declined that handover
        self._handed_over = False   # the turn has been ours and gone back, and
        #                             `over_continue_after` answers from here
        #                             [see _tx_over_response]
        self._polls = 0             # its control bursts since  [see _took_poll]
        # Idle-cadence bursts keyed since the session last moved forward
        # [see _MAX_WITHOUT_PROGRESS]. Named for progress rather than for traffic
        # because the difference is the bug: everything this end keys is traffic,
        # and a clock our own transmitter winds keeps a dead link up by definition.
        self._since_progress = 0
        # Raised only across the delivery inside _answer_data_over, where the mail
        # client re-enters through send(). One burst answers one over, and while
        # this is up send() queues and keys nothing.
        self._answering = False
        # Raised by the first call to on_rx_stream and never lowered: a transport
        # that feeds the raw receive stream hands the connect-response to that
        # search, and on_rx_audio stops accepting one from a bracket. Both routes
        # see the same burst — the stream within a pass of its last symbol, the
        # bracket up to SEG_MAX_S later, when the energy gate finally closes — and
        # nothing downstream can tell a second detection of one answer from a
        # gateway genuinely repeating itself: a real repeat comes 2.4 s apart
        # (NS0A) and a bracket can lag by more, so no guard window separates them.
        # Measured on the 2026-07-26 recording: brackets alone send 2 link-setups,
        # the stream alone sends 2, both together send 3 — one wasted 4.4 s
        # transmission out of a budget of _LINKSETUP_MAX_TX.
        self._stream_owns_response = False
        self._stream_owns_connect_ask = False
        self._stream_owns_turn_request = False
        self._reset_connect_ask_search()
        # Carriers the peer's connect-response arrived off frequency by, which is
        # how far off frequency its connected-ack will arrive too. Measured, not
        # searched, and that distinction is the whole of why the ack is allowed to
        # use it: a shift the ack looked for on its own would be six bins of fixed
        # preamble against a whole band (see on_rx_stream, where a PACTOR-1 exchange
        # holds that preamble for 51 alignments one carrier low), while a shift the
        # response established came with thirteen or more tones keyed to the callsign
        # we dialled. Cleared by :meth:`originate`, so it lives for one attempt.
        self._peer_shift = 0
        # Said once a session, the first time a burst needed its tail to be read
        # [see :meth:`_peer_burst_match`].
        self._burst_match_said = False
        # ...and the fraction of a carrier under it, which is the rest of the same
        # measurement and is what every two-tone reader here is scored on
        # [see :meth:`_note_peer_offset`]. Cleared by :meth:`originate` too.
        self._peer_offset: float | None = None
        self._peer_offset_said = 0.0
        # The same hand-off for the data phase, and raised the same way: by the
        # over search actually running, never by a promise that it will. A
        # transport that feeds the stream through the connect and then stops —
        # which is exactly what `mail_session` used to do — leaves this False and
        # keeps the bracket route it is still relying on.
        self._stream_owns_over = False
        # ...and for the short frames of the data phase, which the stream owns on
        # the wide bandwidths only  [see _stream_answer].
        self._stream_owns_answer = False
        self._reset_over_search()
        # And the same again for the short frames in the connected state that had
        # no stream route at all  [see _stream_grant, _stream_answer].
        self._reset_grant_search()
        self._reset_answer_search()
        # Audio held past the peer's answer to our turn — its release or its
        # drained frame — when it was last named, in seconds  [see _peer_drained,
        # _peer_responder_release, _stream_grant, _took_responder_release].
        self._grant_held = 0.0
        # The same for its answer to an over of ours  [see _peer_over_answer].
        self._answer_held = 0.0
        # And for its NAK of one  [see _peer_responder_nak].
        self._nak_held = 0.0
        # And for its own idle cadence, with the frame of the two that was named
        # [see _peer_responder_idle]. The figure is what says the peer's listening
        # gap is still open, which is the only instant a re-acknowledgement is
        # worth keying into  [see _reack].
        self._idle_held = 0.0
        self._idle_kind = VF.SESSION_RESPONDER_IDLE
        # What the peer still owes us for the answer we keyed at its last over,
        # and where the ladder that asks again has got to  [see _OWED_OVER,
        # _reack]. Nothing here records an answer the peer took: an over that
        # decodes behind it, its release, or its control burst clear all of it.
        self._answer_owed: str | None = None
        self._reacks = 0
        # The continue-class frame that answer went out as, so the first rung
        # repeats it rather than guessing  [see OVER_CONTINUE_ANSWERS].
        self._reack_frame: str | None = None
        # A rung keyed into the peer's own turnaround since the last cadence tick.
        # The cadence is the backstop behind that and not a second ladder.
        self._keyed_on_peer_burst = False
        # Full overs the peer says are still behind the one it just keyed, or None
        # where the field is not one the arithmetic writes  [see arq.phy.overs_after].
        self._overs_hint: int | None = None
        # ``(speed level, per-frame field)`` of the over the next continue
        # answers, which is what selects its seven state symbols at BW500 —
        # neither is on the countdown `_overs_hint` reads  [see
        # vara_frames.over_continue_state].
        self._peer_over_state: tuple[int, int] | None = None
        # DATA transmissions of the peer's this session has read, repeats aside:
        # what the log calls the over a stall is at. One of them may carry two
        # overs  [see _peer_data_over].
        self._peer_over = 0
        # Whether the peer is mid-delivery, which is the one thing that places its
        # per-frame field: the FIRST over of a delivery is not on the countdown
        # [see arq.phy.overs_after].
        self._peer_delivery_open = False
        # An acknowledgement decided and not yet keyed, as (last, a release is
        # owed behind it), with the audio taken since  [see _answer_over]. A
        # station that is transmitting cannot hear us, and a stock sender keys two
        # overs in one window once it has had a run of successes.
        self._held_answer: tuple[bool, bool] | None = None
        self._held_samples = 0
        # Idle frames named while that answer was held. They do not make the
        # emission a gap to key into, but they do say what was inside it
        # [see _a_gap, _release_held_answer].
        self._held_idles = 0
        self._held_idle_at = 0
        self._idle_pair_seen = False
        self._held_idle_early = False
        # This tick was asked for by the peer's poll rather than by our own
        # clock  [see _took_poll, idle_keepalive].
        self._answering_poll = False
        # A block of the peer's this station keyed across and never read. The peer
        # thinks the window was taken whole; asking for it again is the only thing
        # that gets it  [see _release_held_answer].
        self._owed_block = False
        # Ordered bodies of this receive window and the preceding one. A shared
        # prefix is held until the window ends or differs [see _deliver].
        self._window_bodies: list[bytes] = []
        self._window_delivered = 0
        self._last_window: tuple[bytes, ...] = ()
        self._last_window_complete = True
        # Overs identified and not decoded since the last one that was, and the
        # budget behind the NAKs answering them  [see _OVER_NAK_MAX].
        self._undecoded = 0
        # An over of the peer's would not decode and nothing is keyed into its
        # turnaround — its first, or one behind an over we acknowledged — and the
        # ask waits for the peer's idle  [see _undecoded_over,
        # _stream_recovery_cue].
        self._owed_recovery = False
        # Times this session has moved forward, for anything pacing itself on the
        # link from outside  [see _progressed].
        self.progress = 0
        # Idle-cadence bursts this end keyed, for the clock that drives the
        # cadence: an answer keyed into the peer's turnaround is the cadence's
        # burst keyed early, not one more on top of it  [see _took_poll].
        self.idle_keyed = 0
        # Answers keyed back at the PEER's idle cadence, counted apart from
        # `idle_keyed` because they must not restart the driver's keepalive clock
        # [see _answer_peer_idle, kestrel_connect.mail_session].
        self.idle_answers = 0
        self._reset_stream()

    def _progressed(self) -> None:
        """The session moved forward, by the one definition `_MAX_WITHOUT_PROGRESS`
        is written against: a payload delivered, an answer from the peer, or the
        link coming up. Never a burst of our own.

        The counter is public because the idle cadence is driven from outside this
        object and had no other way to tell the two apart. `mail_session` used to
        restart its ~10 s clock on any burst the segmenter bracketed, which on a
        live band is a clock that never comes due: replaying the 102 s after the
        2026-08-20 NAK hands the gate 42 brackets at a longest gap of 8.50 s, none
        of them anything this station could name. The cadence never ran, so
        `_since_progress` never rose, so the budget that closes a link nobody is
        answering on sat behind a branch nothing could reach.
        """
        self._since_progress = 0
        self._released = False
        self.progress += 1

    def _owe_nothing(self) -> None:
        """The peer took the answer we keyed, so the ladder behind it is spent
        and its state goes with it  [see _reack]."""
        self._answer_owed = None
        self._reacks = 0
        self._reack_frame = None
        self._keyed_on_peer_burst = False
        self._owed_recovery = False

    def _first_over_owed(self) -> bool:
        """Is the peer's first DATA over still owed, with nothing of ours on the
        air that could come back as one?

        The one window an all-bad narrow burst can be claimed in. Such a burst
        names nothing, and everywhere else it is refused for that
        [see _peer_data_over_500]; here no DATA body has been keyed, none
        delivered, the turn is the peer's and the link is up, so the burst is
        the over this station could not read. The K5FIT greeting of 2026-09-11
        sat in this state for 78 s while nothing retained the debt.
        """
        return (self.role == "initiator" and self.state == VaraState.CONNECTED
                and self.turn == _TURN_PEER and self._peer_over == 0
                and not self._peer_delivery_open and not self._keyed_bodies)

    def _unread_over_follows_ack(self) -> bool:
        """Has the peer keyed an over behind one we acknowledged, while the
        ladder for that acknowledgement is still live?

        The other window an all-bad narrow burst is claimed in, and the window
        where no rung may go out. The over behind the acknowledgement says the
        peer took it; a continue-class frame keyed now is read as the
        acknowledgement of the burst we could not read. On the BW500 cables of
        2026-09-11 a stock sender took rung 3 for over #6 that way — `BUFFER
        162, 153` — and 18 bytes of its greeting never reached the host.
        """
        return (self.role == "initiator" and self.state == VaraState.CONNECTED
                and self.turn == _TURN_PEER and self._peer_delivery_open
                and self._answer_owed == _OWED_OVER and not self._owed_block
                and self._held_answer is None)

    def _stalled(self) -> bool:
        """Is this end waiting on the peer for something it has said it wants?

        The turn law reads the peer's NAK and its continue burst only while the
        turn is ours, which is where they belong in a delivery of ours. It is also
        where they were unreachable on 2026-09-08: the frames a peer keys when it
        did not read our acknowledgement arrive while the turn is ITS own, and the
        recognisers for them sat behind a turn state that could not be true
        [see _reack, _stream_answer, _read_burst].
        """
        return self._answer_owed is not None or self.turn == _TURN_ASKED

    def _key(self, on: bool) -> None:
        """Key or unkey, and drop what the over search was holding.

        A half-duplex transport stops delivering the channel while we transmit —
        `AudioVaraIO.tx` advances its receive cursor past everything that arrived
        under the burst and the echo guard behind it — so the samples either side
        of a keying are not one signal and cannot be scored as one. `_ov_buf` used
        to run straight across the join.

        What that costs is measured on 2026-08-20, where a keepalive went out 3.7 s
        into a gateway's over: the front of the frame spliced to the band behind
        the keying holds 16-19 of the 24 reference columns — over `_OVER_GUARD_MIN`,
        so it is claimed as an over — and fails its CRC, which is `_UNDECODED` and
        a NAK. The same audio unbroken decodes at 19 of 24 with a clean CRC. A
        missed over costs a retry; a NAK for a frame we broke ourselves asks the
        peer to repair a channel that was fine and spends a budget meant for a
        decoder that cannot read what arrived.
        """
        if on:
            self._reset_over_search()
            self._reset_grant_search()
            self._reset_answer_search()
            self._reset_connect_ask_search()
            # Whatever is going out now is the burst the queue was waiting behind,
            # so nothing is still owed to a clear channel. An over answered under
            # an owed ask puts the queue in `_answer_data_over`'s hands, which asks
            # at the end of the delivery rather than on top of the rest of it.
            self._ask_owed = False
        self.io.key(on)

    # ---- transmit turn ---------------------------------------------------
    def _tx_turn_request(self) -> bool:
        """Ask for the transmit turn: the caller-keyed turn-request frame. True
        when it reached the air.

        A real VARA keys this the moment its host hands it a payload, without
        waiting for the idle cadence, and starts its DATA overs once the peer has
        answered."""
        # `_asked` counts requests that reached the air — it is the budget that
        # eventually gives the turn back as unanswered, and a request the
        # transport declined was never put where the peer could answer it. The
        # turn state still advances: the _TURN_ASKED idle cadence is the retry
        # engine for exactly this kind of lost request.
        sent = self._send_burst(VF.SESSION_TURN_REQUEST)
        if sent:
            self._asked += 1
        self.turn = _TURN_ASKED
        return sent

    def _ask_for_turn(self) -> bool:
        """Ask for the turn from a clear channel, or owe the ask to one.

        The check the 2026-08-26 bench put on :meth:`send` belongs to every ask
        and not to the host's alone. Of the 182 requests keyed across 32 bench
        fetches, 26 went out while the peer was already transmitting: the ask that
        ends a delivery is composed inside the host's answer to the over it
        delivered, the mail client takes 1.4 s over a gateway's greeting, and
        nothing was reading the channel while it did.

        Owed rather than dropped: :meth:`_ask_if_owed` keys it in the turnaround
        the burst under it opens, :meth:`on_rx_stream` keys it at the first poll
        that finds the channel clear, and :meth:`idle_keepalive` is the backstop
        behind both. It costs one request a turn against this responder — 8 or 9
        an exchange where 6 did before — because the moment the channel clears is
        the peer's own 0.96 s turnaround and this frame is 1.39 s, so the request
        that takes the turn is the one after it. That is inside ``_TURN_MAX_ASK``
        and it is a burst that is not on top of anybody.
        """
        if getattr(self.io, "receiving", False):
            if not self._ask_owed:
                self._ask_owed = True
                self.io.log(f"{self.called} is transmitting — {len(self._txq)} "
                            "block(s) queued, asking in its turnaround")
            return False
        self._ask_owed = False
        return self._tx_turn_request()

    def _ask_if_owed(self) -> None:
        """Key the ask a mid-burst :meth:`send` owed, in the turnaround that burst
        opened.

        The bracket closing IS the clear channel :meth:`_ask_for_turn` deferred
        to, and it is the only gap wide enough. Measured against a stock responder
        on 2026-09-03: idling between our overs it keys every 3.4 s and listens for
        1.96 s, and this frame is 1.37 s, so an ask left to the 12 s cadence went
        out 6.3 s behind an unkey — into a gap that was the peer's again. A gate
        open again by the time the bracket closes belongs to the next burst and the
        ask waits for that turnaround instead; a channel that never reads clear
        still has :meth:`idle_keepalive` behind it.
        """
        if not (self._ask_owed and (self._txq or self._tx_pending is not None)
                and self.turn == _TURN_PEER
                and self.state == VaraState.CONNECTED
                and self.role == "initiator"
                and not getattr(self.io, "receiving", False)):
            return
        self._ask_owed = False
        self.io.log(f"{self.called} unkeyed — asking for the turn in its "
                    f"turnaround, {len(self._txq)} block(s) queued")
        self._tx_turn_request()

    def _release_turn(self) -> bool:
        """Hand the channel back: key the release, then stop holding the turn.

        The peer may key a DATA over only while it holds the turn  [spec 05 §5.4],
        so this burst is the whole of what stands between a drained sender and the
        reply already sitting on the far side's data port. Two stock VARA
        instances passing the turn to each other on 2026-08-26 keyed it
        0.13-0.14 s after the answer to their own over and drew the peer's next
        over 0.07-0.10 s later, in every handover of every session
        [vara_frames, SESSION_TURN_RELEASE].

        Until that capture the turn went back as an assignment and nothing went on
        the air, which is a release only this process can read: KE8LVA held the
        link to its own inactivity timeout on it, and a bench VARA reproduced that
        seven times with its answer queued and unsendable.
        """
        keyed = self._send_burst(VF.SESSION_TURN_RELEASE)
        self._release_owed = not keyed
        if not keyed:
            self.io.log("turn release was not transmitted — retaining the turn "
                        "for a retry")
            return False
        self.turn = _TURN_PEER
        self._released = keyed
        self._handed_over = True
        self._polls = 0
        self.io.log(f"nothing left queued — turn released to {self.called}"
                    + ("" if keyed else ", but the transport declined the burst"))
        return keyed

    def _tx_turn_idle(self) -> bool:
        """Hold the turn with nothing to fill it — what a real VARA keyed on the
        idle cadence through both off-air sessions after taking the turn. True when
        it reached the air."""
        return self._send_burst(VF.SESSION_TURN_IDLE)

    def _tx_payload_size(self) -> int:
        """Payload capacity of our selected DATA record, excluding its field."""
        return _phy.body_size(self.bw, self._tx_level) - 1

    def _full_overs_after(self) -> int:
        """Full overs still queued behind the one about to go out, in this same
        delivery  [see _frame_field].

        The delivery ends on the first short block, so the count stops there. A
        `send` behind it is a delivery of its own and its own first over counts
        from the top.
        """
        n = self._tx_payload_size()
        after = 0
        for block in self._txq[1:]:
            if len(block) < n:
                break
            after += 1
        return after

    def _close_stays_at_base(self) -> bool:
        """Will the over that ends this delivery be keyed at the base record?

        Read one over early, because the full over in front of the close is where
        the sender announces which record it is coming at  [see _frame_field].
        The close is the first short block behind the head of the queue, and by
        the time it is keyed a full over has gone out, so :meth:`_over_level`'s
        other term is already settled.
        """
        if self.tx_level is not None:
            return True
        n = self._tx_payload_size()
        close = next((b for b in self._txq[1:] if len(b) < n), None)
        lv = _phy.close_level(self.bw)
        return close is None or not (lv != _phy.base_level(self.bw)
                                     and len(close) < _phy.body_size(self.bw, lv))

    def _over_level(self, payload: bytes) -> int:
        """The record the over carrying ``payload`` is keyed at.

        The base level, except for the over that CLOSES a delivery that ran past
        one over, which drops to record 2 the way every stock delivery on tape
        does behind a full over announcing ``0x81``  [see arq.phy.close_level].
        A close too long for record 2's 48-byte body stays where it is, and so
        does a delivery that fits in a single short over — no full over went out
        in front of that one, and a stock responder has read 36 of them
        byte-exact off the base level.
        """
        if self._tx_recovery_levels:
            return self._tx_recovery_levels[0]
        if self.tx_level is not None:
            # A new delivery starts at the negotiated base. Each full frame
            # announces one downward step until the requested target; a short
            # close resets _full_keyed for the next delivery. Stock rejects a
            # first lower-level DATA frame even when its waveform is valid.
            return max(self._tx_level, _phy.base_level(self.bw) - self._full_keyed)
        close = _phy.close_level(self.bw)
        if (close != _phy.base_level(self.bw) and self._full_keyed
                and len(payload) < _phy.body_size(self.bw, close)):
            return close
        return _phy.base_level(self.bw)

    def _tx_data_over(self) -> bool:
        """Transmit the next queued DATA over. False when there is nothing to key.

        The over rides the same base-level burst as the link-setup, at whichever
        record the bandwidth's base level is — rec3 at BW2300, its 20-bin twin at
        BW2750, level 4 at BW500 — and carries that level's own body length
        [see arq.phy.body_size]. The one that closes a delivery drops a record
        [see :meth:`_over_level`].

        One block an over, at every bandwidth. A stock BW500 pair keys two in a
        burst once a long delivery has run a while, and this station does not:
        what a second block would buy is throughput on a link that has never
        carried a message, and what it costs is a burst twice as long across a
        turnaround measured at half a second  [spec 05 §5.5]. The receive side
        reads both  [see :meth:`_peer_data_over`].

        Nothing commits until the transport confirms the burst went out: the
        block is somebody's mail, and an over that never happened must cost
        nothing — not the queue entry, not the over number, not the duplicate
        gate. Committing after the verdict, rather than popping and undoing,
        leaves no undo path to get wrong. A held over still returns True — the
        transmit slot was taken, and the callers' else-branch keys an idle frame
        that would claim an empty queue. Once transmitted, the block moves from
        the unsent queue into `_tx_pending`; only an acknowledgment retires it.
        A later call with a pending frame solicits its answer instead of advancing.
        """
        if self._tx_pending is not None:
            self._retry_data_over()
            return True
        if not self._txq or self.bw not in _OVER_BW:
            return False
        payload = self._txq[0]
        level = self._over_level(payload)
        # Native stock accepts 81 as a one-record descent and 89 as holding
        # the current record, including multiple full lower-level frames.
        # Keep the default sender's measured countdown policy unchanged.
        field = 0x89 if self._tx_recovery_levels else (
            (0x81 if level > self._tx_level else 0x89) if self.tx_level in (1, 2, 3) else (
            _frame_field(self._full_overs_after(), self._close_stays_at_base())))
        body = _phy.vara_body(payload, self.caller, tail=field,
                              body_len=_phy.body_size(self.bw, level))
        over = self._over + 1
        self._key(True)
        try:
            self.io.tx(OF.data_over_tx(body, over=over, bw=self.bw, level=level))
            sent = self._tx_went_out()
        finally:
            self._key(False)
        if not sent:
            self.io.log(f"DATA over was not transmitted — {len(self._txq)} "
                        "block(s) stay queued")
            return True
        self._txq.pop(0)
        if self._tx_recovery_levels:
            self._tx_recovery_levels.pop(0)
        self._reset_final_ack_recovery()
        self._tx_pending = (body, over, level)
        # A new, successfully keyed frame owns a new recovery opportunity.
        # Repeated queries and refused transmissions never renew this budget.
        self._intermediate_query_attempted = False
        self._intermediate_query_attempts = 0
        self._intermediate_query_for = None
        self._intermediate_query_at = 0.0
        self._intermediate_query_samples = 0
        self._pending_answer_at = self._over_keyed_at = time.monotonic()
        self._tx_retries = 0
        self._keyed_bodies.add(body)
        self._over = over
        self._full_keyed = (self._full_keyed + 1
                            if len(payload) == len(body) - 1 else 0)
        if not self._full_keyed:
            self._query_retry_phase = 0
        at = "" if level == _phy.base_level(self.bw) else f" at record {level}"
        self.io.log(f"tx DATA over #{self._over}{at} ({len(payload)} payload "
                    f"bytes, {len(self._txq)} block(s) still queued)")
        return True

    def _pending_context(self) -> str:
        """Delivery ownership for logs, without exposing payload contents."""
        queued = sum(map(len, self._txq))
        if self._tx_pending is None:
            return f"no pending DATA, {queued} queued bytes"
        body, over, level = self._tx_pending
        payload = _phy.vara_payload(body, caller=self.caller, body_len=len(body))
        final = _phy.over_is_last(body, self.caller)
        queries = self._final_query_attempts if final else self._intermediate_query_attempts
        return (f"over #{over}, record {level}, {len(payload)} pending bytes, "
                f"{queued} queued bytes, {'final' if final else 'intermediate'} "
                f"queries {queries}/{_FINAL_QUERY_MAX}, "
                f"NAK retries {self._tx_retries}/{_OVER_RETRY_MAX}")

    def _reset_final_ack_recovery(self) -> None:
        self._final_query_attempts = 0
        self._final_query_for = None
        self._final_query_at = 0.0
        self._final_query_samples = 0
        self._final_ack_confirmed = None

    def _final_short_over(self) -> bool:
        """A final, short DATA over of ours is the only thing outstanding: the last
        over of a delivery with the queue behind it drained  [see phy.over_is_last]."""
        return (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.turn == _TURN_OURS
                and self._tx_pending is not None and not self._txq
                and _phy.over_is_last(self._tx_pending[0], self.caller))

    def _query_record_candidate(self) -> bool:
        """Recovery uses the outstanding record's geometry, including rate changes."""
        if self._tx_pending is None:
            return False
        body, _, level = self._tx_pending
        records = {"500": (0, 1, 2, 3, 4), "2300": (0, 1, 2, 3),
                   "2750": (100, 101, 102, 103)}
        return (level in records.get(self.bw, ())
                and len(body) == _phy.body_size(self.bw, level))

    def _final_ack_candidate(self) -> bool:
        """Short-final query/request/drained at supported wideband records."""
        return (self.bw in ("2300", "2750") and self._final_short_over()
                and self._query_record_candidate())

    def _final_over_was_read(self) -> bool:
        """Is this turn-request the responder's ANSWER to our final short over?

        The BW2300 equivalent of the BW2750 query/drained confirmation, and it
        takes the same caution, because the frame alone does not say what it is
        answering. A responder asks for the turn on its own schedule too, and a
        final over that FADED draws exactly that ask a cadence later — retiring on
        it would drop the last block of the operator's message unread.

        So three things beyond the shape  [see _final_short_over]: the request is
        the stream's own complete, fresh frame rather than a bracket's guess
        [see _stream_connect_ask]; our final over has actually gone out, which is
        what ``_pending_answer_at`` records at key-down; and the request arrived
        inside the window a responder answers an over in, which is measured off
        the K0SI tape and the stock bench rather than borrowed from the query
        reply  [see ``_TURN_REQUEST_ANSWER_S``].
        """
        return (self._final_short_over() and self.bw == "2300"
                and self._stream_owns_turn_request
                and self._pending_answer_at is not None
                and time.monotonic() - self._pending_answer_at
                <= _TURN_REQUEST_ANSWER_S)

    def _final_query_fresh(self) -> bool:
        return (self._final_query_for is not None
                and self._final_query_for == self._tx_pending
                and time.monotonic() - self._final_query_at <= _FINAL_QUERY_REPLY_S
                and self._final_query_samples <= _FINAL_QUERY_REPLY_S * MK.FS)

    def _query_final_answer(self) -> bool:
        """Solicit a final answer; a successful query alone retires no bytes."""
        if not self._final_ack_candidate():
            return False
        if self._final_ack_confirmed == self._tx_pending:
            return self._finish_final_answer()
        if self._final_query_fresh():
            return False  # One query per open response window.
        if self._final_query_attempts >= _FINAL_QUERY_MAX:
            self.io.log("final-answer query budget exhausted — closing with "
                        f"delivery unconfirmed; {self._pending_context()}")
            self.disconnect()
            return False
        self._final_query_for = None
        if not self._send_burst(VF.SESSION_FINAL_ANSWER_QUERY):
            self.io.log(f"final-answer query refused — retaining {self._pending_context()}")
            return False
        self._final_query_for = self._tx_pending
        self._final_query_at = time.monotonic()
        self._final_query_samples = 0
        self._final_query_attempts += 1
        self._since_progress += 1
        self.idle_keyed += 1
        self.io.log(f"final-answer query {self._final_query_attempts}/"
                    f"{_FINAL_QUERY_MAX} (60/683) — retaining {self._pending_context()}")
        return True

    def _finish_final_answer(self) -> bool:
        """Commit a solicited final answer only after the measured drained TX."""
        if not (self._final_ack_candidate()
                and self._final_ack_confirmed == self._tx_pending):
            return False
        if not self._send_burst(VF.SESSION_DRAINED):
            self.io.log("final-answer handover was not transmitted — "
                        f"retaining the confirmed answer and {self._pending_context()}")
            return False
        self._close_echo_window()
        self._owe_nothing()
        self._tx_pending = None
        self._tx_retries = 0
        self._reset_final_ack_recovery()
        self._into_our_turn = 0
        self._progressed()
        self.turn = _TURN_PEER
        self._released = self._handed_over = True
        self._release_owed = False
        self._polls = 0
        self.io.log("solicited final answer confirmed — final over retired, "
                    f"drained handover transmitted to {self.called}")
        return True

    def _intermediate_answer_candidate(self) -> bool:
        """A full pending record with further queued DATA at a supported bandwidth."""
        if (not self.probe_intermediate_query
                or self.state != VaraState.CONNECTED or self.role != "initiator"
                or self.bw not in ("500", "2300", "2750") or self.turn != _TURN_OURS
                or self._tx_pending is None
                or not self._query_record_candidate()
                or not self._txq):
            return False
        body = self._tx_pending[0]
        return (not _phy.over_is_last(body, self.caller)
                and len(_phy.vara_payload(body, caller=self.caller, body_len=len(body)))
                == len(body) - 1)

    def intermediate_query_due_in(self) -> float | None:
        """Solicit after DATA, then allow each query its complete reply window."""
        if (not self._intermediate_answer_candidate()
                or self._pending_answer_at is None
                or self._held_answer is not None or self._release_owed):
            return None
        deadline = (self._intermediate_query_at + 4.5
                    if self._intermediate_query_attempted
                    else self._pending_answer_at + _OVER_RETRY_S)
        return max(0.0, deadline - time.monotonic())

    def query_intermediate_answer(self) -> bool:
        """Driver entry: call when due and the transport is not receiving."""
        due = self.intermediate_query_due_in()
        if due is None or due > 0:
            return False
        before = self._intermediate_query_attempts
        self._retry_data_over()
        return self._intermediate_query_attempts > before

    def data_retry_due_in(self) -> float | None:
        """Time until the outstanding DATA answer needs recovery, or None.

        Its own clock rather than the idle cadence's  [see ``_OVER_RETRY_S``].
        None wherever `idle_keepalive` would do something else with the tick: a
        deferred answer, an owed release, a turn that is not ours, or the budget
        that closes a link nobody is answering on, which stays that method's to
        spend. None too while a final-answer query has a reply window open, so the
        solicitation this would otherwise repeat gets the 2 s it was measured with.
        """
        if (self._tx_pending is None or self._intermediate_answer_candidate()
                or self._over_keyed_at is None
                or self.state is not VaraState.CONNECTED
                or self.role != "initiator"
                or self.turn != _TURN_OURS
                or self._held_answer is not None or self._release_owed
                or self._since_progress >= _MAX_WITHOUT_PROGRESS
                or self._final_query_fresh()):
            return None
        return max(0.0, self._over_keyed_at + _OVER_RETRY_S - time.monotonic())

    def retry_data_over(self) -> bool:
        """Driver entry: call when due and the transport is not receiving.

        The clock moves even on a refused query to avoid a tight retry loop.
        The legacy return value counts DATA repeats only; timeout recovery now
        solicits feedback or closes, and never authorizes such a repeat.
        """
        due = self.data_retry_due_in()
        if due is None or due > 0:
            return False
        before = self._tx_retries
        self._retry_data_over(final_query=True)
        self._over_keyed_at = time.monotonic()
        if self._tx_retries == before:
            return False
        self._since_progress += 1
        self.idle_keyed += 1
        return True

    def _probe_intermediate_answer(self) -> bool:
        """Ask for state without replaying DATA or retiring any pending bytes."""
        if (not self._intermediate_answer_candidate()
                or self._intermediate_query_fresh()
                or self._intermediate_query_attempts >= _FINAL_QUERY_MAX):
            return False
        pending = self._tx_pending
        # Every successfully keyed full DATA frame, including a NAK retry,
        # advances phase. Repeated queries and refused writes retain it.
        kind = VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[
            (self._full_keyed + self._query_retry_phase - 1) % 2]
        self._intermediate_query_for = None
        self._intermediate_query_attempted = True
        self._intermediate_query_at = time.monotonic()
        if not self._send_burst(kind):
            self.io.log(f"intermediate query refused — retaining {self._pending_context()}")
            return False
        self._intermediate_query_for = pending
        self._intermediate_query_at = time.monotonic()
        self._intermediate_query_samples = 0
        self._intermediate_query_attempts += 1
        self._reset_answer_search()
        self._since_progress += 1
        self.idle_keyed += 1
        self.io.log(f"intermediate query {self._intermediate_query_attempts}/"
                    f"{_FINAL_QUERY_MAX} ({kind.seed_off}/{kind.preadv}) — "
                    f"retaining {self._pending_context()}")
        return True

    def _intermediate_query_fresh(self) -> bool:
        return (self._intermediate_answer_candidate()
                and self._intermediate_query_for is not None
                and self._intermediate_query_for == self._tx_pending
                and 0 <= time.monotonic() - self._intermediate_query_at <= 4.5
                and self._intermediate_query_samples <= 4.5 * MK.FS)

    def _peer_intermediate_query_answer(self, samples, track=None) -> bool:
        """Called-keyed query ACK, complete tail, >=20 exact clear payload symbols.

        Stock acknowledged an intermediate DATA that reached its host with this
        frame after its ordinary eight-symbol continue was erased. The live
        KC9GHZ query answer has 26 exact clear payload symbols. This recognizer
        is reachable only after our successful query for the same pending body;
        neither an unsolicited turn request nor a final-short answer qualifies.
        """
        self._matched_intermediate_answer = None
        self._intermediate_answer_wait_samples = 0
        if not self._intermediate_query_fresh():
            return False
        kinds = [VF.SESSION_INTERMEDIATE_QUERY_ANSWER]
        if (self.tx_level is None and self._tx_retries > 0
                and self._tx_pending[2] == _phy.base_level(self.bw) - 2):
            kinds.append(VF.SESSION_RETRY_QUERY_ANSWER)
        # 288/311 acknowledges DATA and selects a lower next record. Stock
        # caller replay of KC9GHZ's native reply also proves it with four blocks
        # still queued; it is not restricted to a short closing boundary.
        if (self._can_accept_downshift()
                or (len(self._txq) == 1
                    and len(self._txq[0]) < len(self._tx_pending[0]) - 1)):
            kinds.append(VF.SESSION_RESPONDER_OVER_ANSWER)
        for template in kinds:
            kind = VF.for_bw(template, self.bw)
            tones = np.asarray(VF.handshake_tones(self.called, kind))
            heard, at = _payload_fit(samples, kind, tones, track, self._band)
            clear = heard[1:] >= 0
            held = len(samples) - at - _span(kind)
            exact = clear.sum() >= 20 and np.all(heard[1:][clear] == tones[1:][clear])
            if exact and held < 0:
                # A scan just before the final symbol must resume at its tail.
                # The ordinary half-second stride can next see a complete reply
                # only after the half-second freshness window has expired.
                wait = int(np.ceil(-held))
                if (not self._intermediate_answer_wait_samples
                        or wait < self._intermediate_answer_wait_samples):
                    self._intermediate_answer_wait_samples = wait
            # Alignment is fitted on the payload; the independently unaligned
            # lead must not veto an otherwise exact, complete called-keyed reply.
            if exact and clear[-1] and 0 <= held <= _GRANT_FRESH_S * MK.FS:
                self._matched_intermediate_answer = (self._tx_pending, kind)
                return True
        return False

    def _took_intermediate_query_answer(self) -> None:
        # Revalidate immediately before retirement: a bracket replay or another
        # callback must not apply this answer to the next distinct pending body.
        if not self._intermediate_query_fresh():
            return
        match = self._matched_intermediate_answer
        observed = self._unclassified_continue
        if (match and match[0] == self._tx_pending and observed is not None
                and match[1] == VF.for_bw(VF.SESSION_INTERMEDIATE_QUERY_ANSWER, self.bw)
                and observed[:2] == (self._tx_pending, self._pending_answer_at)):
            # The query confirms this exact pending frame, with no intervening
            # DATA transmission. Its earlier eight-symbol answer can now be
            # recognized on this link; neither shape nor silence teaches an ACK.
            self._confirmed_continue_pairs.add(
                (self.caller, self.called, self.bw, observed[2]))
        self._unclassified_continue = None
        if (match and match[0] == self._tx_pending
                and match[1] == VF.for_bw(VF.SESSION_RETRY_QUERY_ANSWER, self.bw)):
            # Stock's 288/249 answer confirms the small retry AND returns the
            # remaining short delivery to base. Keep later host writes separate.
            count = len(self._tx_recovery_levels)
            if count:
                self._txq = [b"".join(self._txq[:count])] + self._txq[count:]
                self._tx_recovery_levels.clear()
        match = self._matched_intermediate_answer
        kind = match[1] if match and match[0] == self._tx_pending else None
        name = (f"{kind.name} ({kind.seed_off}/{kind.preadv})" if kind
                else "solicited intermediate answer")
        self.io.log(f"rx {name} — acknowledged {self._pending_context()}")
        self._matched_intermediate_answer = None
        if kind == VF.for_bw(VF.SESSION_RESPONDER_OVER_ANSWER, self.bw):
            self._lower_acked_delivery()
        self._took_control_burst(continue_reply=True)

    def _can_accept_downshift(self) -> bool:
        """Measured wideband ACK/downshift with a complete queued delivery."""
        if (self.bw not in ("2300", "2750") or self.tx_level not in (None, 4)
                or self._tx_pending is None or not self._txq
                or self._tx_pending[2] not in _RX2300.INDEX_LEVELS_BW[self.bw]):
            return False
        capacity = len(self._tx_pending[0]) - 1
        return any(len(block) < capacity for block in self._txq)

    def _lower_acked_delivery(self) -> None:
        """Retain only unsent bytes, repacked for the acknowledged speed change.

        Both generated 288/311 and KC9GHZ's September 20 native reply made
        stock retire 89 pending bytes and send the next 47 at record 102.
        Later host writes remain separate deliveries and keep their own plan.
        """
        if not self._can_accept_downshift():
            return
        capacity = len(self._tx_pending[0]) - 1
        stop = next(i + 1 for i, b in enumerate(self._txq) if len(b) < capacity)
        delivery = b"".join(self._txq[:stop])
        lower = _phy.lower_level(self.bw, self._tx_pending[2])
        take = _phy.body_size(self.bw, lower) - 1
        queue = [delivery[i:i + take] for i in range(0, len(delivery), take)]
        if not queue or len(queue[-1]) == take:
            queue.append(b"")
        self._txq = queue + self._txq[stop:]
        self._tx_recovery_levels = [lower] * len(queue) + self._tx_recovery_levels[stop:]
        self.io.log(f"peer requested record {lower} after ACK — "
                    f"{len(delivery)} unsent bytes repacked into {len(queue)} blocks")

    def _data_nak_fresh(self, kind) -> bool:
        """Full DATA rejection with its complete remaining delivery preserved."""
        if (self.state != VaraState.CONNECTED or self.role != "initiator"
                or self.turn != _TURN_OURS or self.bw not in ("500", "2300", "2750")
                or self.tx_level is not None or self._tx_pending is None
                or not self._txq):
            return False
        records = {"500": (4, 3, 2, 1, 0), "2300": (3, 2, 1, 0),
                   "2750": (103, 102, 101, 100)}
        if self._tx_pending[2] not in records[self.bw]:
            return False
        body = self._tx_pending[0]
        capacity = len(body) - 1
        if (len(body) != _phy.body_size(self.bw, self._tx_pending[2])
                or _phy.over_is_last(body, self.caller)
                or len(_phy.vara_payload(body, caller=self.caller, body_len=len(body))) != capacity
                or not any(len(block) < capacity for block in self._txq)):
            return False
        if (kind == VF.for_bw(VF.SESSION_DATA_NAK_QUERY, self.bw)
                and self._intermediate_query_fresh()):
            return True
        # N0XYZ also sends 288/63 directly after rejected DATA. An expired
        # query cannot borrow this direct window: attempted queries exclude it.
        return (not self._intermediate_query_attempted
                and self._pending_answer_at is not None
                and 0 <= time.monotonic() - self._pending_answer_at <= 4.5)

    def _pair_data_nak_at(self, samples, track=None, *, min_symbols=8):
        """Fit a measured full-link NACK; near matches only veto positive feedback."""
        pairs = VF.DATA_NAK_RESPONDER_BY_LINK.get((self.caller, self.called, self.bw))
        if pairs is None or self.role != "initiator" or len(samples) >= _DATA_OVER_MIN:
            return None
        bins, clear, _ = (_top3_track(samples, band=self._band, bin_offset=self._grid)
                          if track is None else track)
        idx = np.arange(max(len(bins) - _CONT_SPAN, 0))[:, None] + _CONT_OFF
        found = np.sort(bins[idx, :2], axis=2)
        wanted = np.asarray(pairs) + self._peer_shift
        matches = (found == wanted).all(axis=2)
        held = ((_live_track(samples)[idx] & (clear[idx, 1] >= _CLEAR_DB)).all(1)
                & matches[:, 0] & (matches.sum(1) >= min_symbols))
        start, width = _widest(held)
        return ((start + width // 2) * _ACK_GRID if width >= _CONT_PLATEAU else None)

    def _peer_data_nak(self, samples, track=None) -> bool:
        """Called-keyed negative feedback, with a complete clear payload tail."""
        self._matched_data_nak = None
        self._data_nak_wait_samples = 0
        if len(samples) >= _DATA_OVER_MIN:
            return False
        if self._data_nak_fresh("session-data-pair-nak"):
            at = self._pair_data_nak_at(samples, track)
            if at is not None:
                held = len(samples) - at - 8 * MK.HOP
                if 0 <= held < .06 * MK.FS:
                    self._data_nak_wait_samples = int(np.ceil(.06 * MK.FS - held))
                elif .06 * MK.FS <= held <= _GRANT_FRESH_S * MK.FS:
                    self._matched_data_nak = (self._tx_pending, self._pending_answer_at,
                                              "session-data-pair-nak")
                    return True
        for template in (VF.SESSION_DATA_NAK_SHORT, VF.SESSION_DATA_NAK_QUERY):
            kind = VF.for_bw(template, self.bw)
            if not self._data_nak_fresh(kind):
                continue
            tones = np.asarray(VF.handshake_tones(self.called, kind))
            heard, at = _payload_fit(samples, kind, tones, track, self._band)
            clear = heard[1:] >= 0
            held = len(samples) - at - _span(kind)
            if (clear.sum() >= (13 if kind.n_payload == 15 else 20) and clear[-1]
                    and np.all(heard[1:][clear] == tones[1:][clear])):
                if 0 <= held < .06 * MK.FS:
                    # The full payload arrived just before this scan. Wait out
                    # its tail guard without sleeping through the reply window.
                    self._data_nak_wait_samples = int(np.ceil(.06 * MK.FS - held))
                elif .06 * MK.FS <= held <= _GRANT_FRESH_S * MK.FS:
                    self._matched_data_nak = (self._tx_pending, self._pending_answer_at, kind)
                    return True
        return False

    def _data_nak_level(self) -> int:
        """Stock's CRC-rejection fallback, distinct from an announced descent."""
        level = self._tx_pending[2]
        # BW500 rejects repeated record-1 attempts after record-2 rejection;
        # the stock-confirmed fallback is its lowest record, 0.
        if self.bw == "500" and level == 2:
            return 0
        return _phy.lower_level(self.bw, level)

    def _took_data_nak(self) -> None:
        match = self._matched_data_nak
        if (match is None or match[:2] != (self._tx_pending, self._pending_answer_at)
                or not self._data_nak_fresh(match[2])):
            self._matched_data_nak = None
            return
        kind = match[2]
        label = kind if isinstance(kind, str) else f"{kind.name} ({kind.seed_off}/{kind.preadv})"
        level = self._data_nak_level()
        self.io.log(f"rx {label} — NAK; retaining {self._pending_context()}, "
                    f"retrying at record {level}")
        self._close_echo_window()
        try:
            self._retry_data_over(peer_asked=True, adapt=True, nak_step=1)
        finally:
            self._matched_data_nak = None

    def _nak_repacketization(self, step: int = 2) -> tuple[tuple[bytes, int, int], list[bytes], list[int]] | None:
        """Prepare a measured NAK retry without committing any bytes.

        The 289/31 and 288/63 DATA NACKs take one step down and hold that
        record through the current delivery. The missing-DATA 288/1117 NAK
        takes two steps down, then resumes base if a full base frame remains.
        Each action has its own qualified feedback window; preserve separate
        host deliveries, explicit TX ladders and all unconfirmed data.
        """
        if step != 1 and self.bw == "500":
            return None  # Missing-DATA fallback geometry still needs its own oracle.
        final = (step == 2 and self.bw == "2300"
                 and self._final_ack_candidate() and self._final_query_fresh())
        if step == 1:
            match = self._matched_data_nak
            if (match is None or match[:2] != (self._tx_pending, self._pending_answer_at)
                    or not self._data_nak_fresh(match[2])):
                return None
        elif not (final or self._intermediate_query_fresh()):
            return None
        if (self.tx_level is not None
                or (step != 1 and self._tx_pending[2] != _phy.base_level(self.bw))
                or (step != 1 and not final and not self._intermediate_answer_candidate())):
            return None
        body, over, level = self._tx_pending
        capacity = _phy.body_size(self.bw, level) - 1
        payload = _phy.vara_payload(body, caller=self.caller, body_len=len(body))
        # A queried short final already contains the complete delivery boundary.
        stop = (0 if final else next((i + 1 for i, block in enumerate(self._txq)
                                     if len(block) < capacity), None))
        if stop is None:
            return None  # No complete delivery boundary to preserve.
        delivery = payload + b"".join(self._txq[:stop])
        lower = self._data_nak_level() if step == 1 else _phy.lower_level(self.bw, level, step)
        take = _phy.body_size(self.bw, lower) - 1
        remainder = delivery[take:]
        # One-step DATA rejection holds the lower record through this delivery.
        # The two-step missing-DATA path also holds it for a short remainder:
        # an unannounced short base close cannot follow the lower retry.
        hold_lower = step == 1 or len(remainder) < capacity
        chunk = take if hold_lower else capacity
        queue = [remainder[i:i + chunk] for i in range(0, len(remainder), chunk)]
        if len(delivery) >= take and (not queue or len(queue[-1]) == chunk):
            queue.append(b"")
        close = _phy.close_level(self.bw)
        close_at_base = close == level or bool(queue and len(queue[-1]) >= _phy.body_size(self.bw, close))
        field = 0x89 if hold_lower else _frame_field(len(remainder) // capacity, close_at_base)
        retry = _phy.vara_body(delivery[:take], self.caller, tail=field,
                               body_len=_phy.body_size(self.bw, lower))
        levels = [lower] * len(queue) if hold_lower else []
        return (retry, over, lower), queue + self._txq[stop:], levels

    def _retry_data_over(self, *, final_query: bool = False,
                         peer_asked: bool = False, adapt: bool = False,
                         nak_step: int = 2) -> None:
        """Solicit missing feedback; repeat DATA only for a decoded peer NAK.

        A timeout does not say whether the peer received the block. A state
        query retains it until positive feedback arrives. Unsupported recovery
        and exhausted query budgets close with delivery unconfirmed.
        A qualified DATA NAK can repacketize the current delivery into its
        measured lower retry. Other NAKs retain the original geometry.
        A transport refusal spends no transmit budget.

        ``peer_asked`` is a NAK the peer keyed rather than a timeout of ours, and
        it resends even where host retries are disabled: the budget still bounds
        it, and a keyed ask is not the silence that gate guards against  [see
        :meth:`_took_responder_nak`].
        """
        if self._tx_pending is None:
            return
        if self.state != VaraState.CONNECTED:
            return
        if not peer_asked:
            if self._final_ack_candidate():
                self._query_final_answer()
                return
            if self._intermediate_answer_candidate():
                if (self._intermediate_query_fresh()
                        or self._intermediate_query_attempts < _FINAL_QUERY_MAX):
                    self._probe_intermediate_answer()
                    return
            self.io.log("DATA answer unconfirmed — closing without blind DATA "
                        f"retransmission; retaining {self._pending_context()}")
            if self.role == "initiator":
                self.disconnect()
            else:
                self.state = VaraState.DISCONNECTED
            return
        if self._tx_retries >= _OVER_RETRY_MAX:
            self.io.log("DATA over retry budget exhausted — closing with "
                        f"delivery unconfirmed; {self._pending_context()}")
            self.disconnect()
            return
        replacement = self._nak_repacketization(nak_step) if adapt else None
        body, over, level = replacement[0] if replacement else self._tx_pending
        self._final_query_for = self._final_ack_confirmed = None
        self._intermediate_query_for = None
        self._key(True)
        try:
            self.io.tx(OF.data_over_tx(body, over=over, bw=self.bw, level=level))
            sent = self._tx_went_out()
        finally:
            self._key(False)
        if sent:
            if replacement is not None:
                self._tx_pending, self._txq, self._tx_recovery_levels = replacement
                self.io.log(f"NAK recovery repacketized at record {level}: "
                            f"{len(body) - 1} bytes pending, remaining bytes queued")
            self._tx_retries += 1
            if not _phy.over_is_last(body, self.caller):
                # Native stock toggles the query phase after a keyed full retry
                # too. Keep this separate from the sender's rate-ladder index.
                self._query_retry_phase ^= 1
            # A NAK-requested retry has its own reply window. The previous
            # query's deadline may already have elapsed during DATA playback.
            self._pending_answer_at = self._over_keyed_at = time.monotonic()
            self._intermediate_query_attempted = False
            self._keyed_bodies.add(body)
            self.io.log(f"retry DATA over #{over} "
                        f"({self._tx_retries}/{_OVER_RETRY_MAX}); {self._pending_context()}")
        else:
            self.io.log(f"NAK retry was not transmitted — retaining {self._pending_context()}")

    def _peer_nak(self, samples, track: tuple | None = None, *,
                  min_symbols: int = 8) -> bool:
        """Match the measured NAK. Partial matches only veto a continue;
        all eight symbols are required to trigger an immediate retry."""
        pair = VF.nak(self.caller, self.bw)
        if pair is None or len(samples) >= _DATA_OVER_MIN:
            return False
        want = np.asarray(pair[0] if self.role == "responder" else pair[1])
        want = np.sort(want + self._peer_shift, axis=1)
        x = np.asarray(samples, dtype=np.float64)
        bins, clear, _ = (_top3_track(x, band=self._band, bin_offset=self._grid)
                          if track is None else track)
        idx = np.arange(max(len(bins) - _CONT_SPAN, 0))[:, None] + _CONT_OFF
        heard = np.sort(bins[idx][:, :, :2], axis=2)
        matches = (heard == want).all(axis=2)
        held = ((_live_track(x)[idx] & (clear[idx][:, :, 1] >= _CLEAR_DB)).all(1)
                & matches[:, 0] & (matches.sum(axis=1) >= min_symbols))
        return _widest(held)[1] >= _CONT_PLATEAU

    def _took_nak(self) -> None:
        """The peer failed to read something of ours and is asking for it again.

        Which thing depends on what is outstanding: an over of ours is repeated,
        and with none it is the acknowledgement we keyed at the peer's own over
        that did not arrive — the frame the ladder is for  [see :meth:`_reack`].
        Answering the second case with an over retry keys nothing at all, because
        there is nothing pending to retry.
        """
        self._close_echo_window()
        if self._tx_pending is None:
            self.io.log(f"rx NAK from {self.called} — it did not read our answer")
            self._reack()
            return
        self.io.log(f"rx NAK from {self.called} — repeating the outstanding over")
        self._retry_data_over(peer_asked=True)

    def _took_responder_nak(self) -> None:
        """The peer could not read a DATA over of ours and is asking for it again
        [vara_frames, SESSION_OVER_NAK_RESPONDER].

        A NAK the peer keyed is a positive ask, so it licenses one bounded resend
        of the outstanding over even where host retries are disabled: the silence
        ``allow_data_retries=False`` guards against is exactly what a keyed NAK is
        not. The retry budget still bounds it  [see :meth:`_retry_data_over`].
        """
        self._close_echo_window()
        if self._tx_pending is None:
            self.io.log(f"rx responder NAK from {self.called} — nothing "
                        "outstanding to repeat")
            return
        self.io.log(f"rx responder NAK from {self.called} — repeating the "
                    f"outstanding over, named {self._nak_held:+.3f} s from its "
                    "last symbol")
        self._retry_data_over(peer_asked=True, adapt=True)

    def _took_responder_nak_out_of_turn(self) -> None:
        """The same NAK while the turn is the peer's: what it could not read is
        the acknowledgement we keyed at ITS over, not an over of ours.

        So the answer goes out again and no DATA does — an over keyed into the
        peer's turn is what :meth:`_over_into_our_turn` counts against a station,
        and the frame does not ask for one. This is the caller-side NAK's own
        split  [see :meth:`_took_nak`, which re-acks when nothing is pending].

        NOT :meth:`_took_stall_answer`, which is the shape this used to borrow and
        the one thing that cannot happen here: it calls :meth:`_owe_nothing`, and
        with the debt cleared the ladder is spent — the answer the peer has just
        said it could not read would never be keyed again. Measured: a station
        owing `_OWED_OVER` keys four ladder rungs, and four became none.

        NOR :meth:`_progressed`, which this reached for and which is the opposite
        of what the frame says: a peer keying "I could not read you" has read
        nothing of ours. It zeroes `_since_progress` and moves the pair
        `kestrel_connect.mail_session` restarts its keepalive deadline on, so a
        responder keying one NAK per tick interval held a CONNECTED link open for
        ever — measured, 13 consecutive NAKs behind a spent ladder left
        `_since_progress` at 0 with nothing keyed. What a rung costs is
        :meth:`_reack`'s own charge, and the ladder's budget bounds it.

        WITH NOTHING OWED the turn is `_TURN_ASKED` — that is the only other way
        :meth:`_stalled` is true — and there is no answer to say again. The frame
        is then an observation: the request this station is waiting on is the
        turn-request, and that goes again on its own cadence.
        """
        self._close_echo_window()
        if self._answer_owed is None:
            self.io.log(f"rx responder NAK from {self.called} while our "
                        f"turn-request is outstanding, named "
                        f"{self._nak_held:+.3f} s from its last symbol — the "
                        "request goes again on its cadence")
            return
        self.io.log(f"rx responder NAK from {self.called} — it did not read our "
                    f"answer, named {self._nak_held:+.3f} s from its last symbol")
        self._reack()

    @property
    def data_pending(self) -> bool:
        """Host bytes still queued or awaiting the peer's acknowledgement."""
        return bool(self._txq or self._tx_pending is not None)

    def send(self, payload: bytes) -> None:
        """Host data to put on the air, split into DATA-over blocks.

        Queueing is always allowed; keying is not, and there are two ways it is
        not. An idle link keys at once: a real VARA put its turn-request on the
        air 0.17 s after its own host handed it a payload, breaking a 12 s
        keepalive cadence to do so, with its peer nine seconds from transmitting
        [vara_frames, SESSION_TURN_REQUEST]. Inside the delivery of an over the
        peer holds the turn for, it keys nothing: the host is answering a burst
        this station is still composing its own answer to, and that answer is one
        frame  [see :meth:`_answer_data_over`]. What the host hands us there goes
        into the queue and into the choice of that frame instead.
        """
        if self.tx_level is not None and payload:
            # Every host write is a separately closed delivery. Plan its entry
            # ladder from base, even when it queues behind a pending delivery.
            # Never squeeze an already queued base-size block into a lower PHY.
            blocks = []
            offset, level = 0, _phy.base_level(self.bw)
            while True:
                n = _phy.body_size(self.bw, level) - 1
                block = bytes(payload[offset:offset + n])
                blocks.append(block)
                offset += len(block)
                if len(block) < n:
                    break
                level = max(self._tx_level, level - 1)
        else:
            n = self._tx_payload_size()
            blocks = [bytes(payload[i:i + n]) for i in range(0, len(payload), n)]
            if blocks and len(blocks[-1]) == n:
                # A full over carries no trailer and says another one follows it,
                # so a delivery cannot end on one and a payload that is an exact
                # multiple of the block owes an empty over to close on
                # [see phy.over_is_last]. Without it the peer is never told the
                # message ended: 356 bytes went out as four full overs and the
                # turn never came back.
                blocks.append(b"")
        self._txq += blocks
        self.io.log(f"queued {len(payload)} bytes as {len(self._txq)} block(s)")
        if (self.state is not VaraState.CONNECTED or self._answering
                or self._release_owed):
            return
        # A third way keying is not allowed, and the one the bench of 2026-08-26
        # found: the host writes on its own clock, and a burst keyed while the
        # peer is mid-burst lands on top of it. That run put 1.63 s of
        # turn-request across the middle of a 1.4 s frame, drew nothing, and spent
        # the whole ask budget into a peer that had stopped listening. The queue is
        # set either way; a request waits for a quiet channel and is keyed there
        # [see _ask_for_turn], which is where a real VARA's 0.17 s ask was
        # measured: the turnaround the burst under it opens [see _ask_if_owed].
        # An over we already hold the turn for waits for the cadence.
        if self.turn == _TURN_OURS:
            if self._tx_pending is not None:
                return  # Host writes cannot bypass the outstanding acknowledgment.
            if getattr(self.io, "receiving", False):
                self.io.log(f"{self.called} is transmitting — "
                            f"{len(self._txq)} block(s) queued, keying on the cadence")
                return
            self._tx_data_over()
        elif self.turn == _TURN_PEER:
            self._ask_for_turn()

    def _close_echo_window(self) -> None:
        """Some other station has transmitted, so nothing we keyed before it can
        still be arriving as an echo  [see ``_keyed_bodies``].

        *Some other station*, deliberately, and not *the peer*: neither recogniser
        that calls this can name the transmitter. A DATA over carries no address
        [see :meth:`_peer_data_over`] and the short control burst is accepted on a
        fixed preamble with a measured, non-zero false-accept floor [see
        :meth:`_peer_control_burst`]. So a stranger's burst, or an accept on band
        noise, closes the window too, and our own transmission arriving after it
        would be delivered to the host as received. What bounds that is the
        transport rather than this: :class:`AudioVaraIO` skips the receive stream
        through the end of every transmission plus 0.1 s, so on the live audio path
        the echo this filters never reaches the state machine at all. This is the
        layer behind that one.
        """
        self._keyed_bodies.clear()

    def _over_into_our_turn(self) -> None:
        """A DATA over arrived while we hold the turn. Count it, and give the turn
        back once the count says our model of it is the thing that is wrong.

        One over settles nothing: it is the peer taking the turn back or a station
        we are not in session with, and the frame says which of those it is nowhere
        [see :meth:`_peer_data_over`]. Conceding on the first would cost a
        stranger's single transmission the turn we hold and the queue behind it.
        Never conceding costs more, and costs it to the traffic that matters: the
        turn would be a state with no way out, the queue would never drain, a later
        ``send`` would key a 5.4 s over on top of a gateway mid-delivery, and the
        idle cadence would tell a station that is transmitting that we hold the turn
        and have nothing, which is how the sessions behind this file died before the
        turn law was read off the recordings.

        So the turn is not conceded on one over and not held against a run of them.
        Each over answered in the ordinary way — by whichever burst the over itself
        asks for  [see :meth:`_tx_over_response`] — and at ``_TURN_YIELD_AFTER`` of
        them, with no lawful answer from the peer in between, the turn goes back
        where the traffic says it is. The queue is kept: :meth:`_answer_data_over`
        asks for the turn again on the very over that yields it, and
        :meth:`idle_keepalive` carries the request from there, which is the
        recovery an unanswered turn-request already had.

        What a stranger costs, then: one per-over response each, and
        ``_TURN_YIELD_AFTER`` consecutive ones — with our own peer answering nothing
        in between — before the turn goes back. Back, and not across: yielding puts
        the turn where it started the session and keys a turn-request for it, which
        is the state a stranger cannot profit from, and the queue stays where it is
        through all of it.
        """
        self._into_our_turn += 1
        if self._into_our_turn < _TURN_YIELD_AFTER:
            return
        self.io.log(f"{self._into_our_turn} DATA overs keyed into our turn and no "
                    f"answer to ours — the turn is the other station's, "
                    f"{len(self._txq)} block(s) still queued")
        self.turn = _TURN_PEER
        self._into_our_turn = 0

    def _peer_control_burst(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the peer answering us while we hold the turn.

        Length alone would key the transmitter at our own frames coming back
        through the receiver — a turn-idle answering a turn-idle, forever — so
        the audio has to be positively a control burst. Two recognisers, both
        already measured: the 11-symbol two-tone burst by its fixed four-pair
        preamble (:func:`_ack_plateau`, whose false-accept floor is measured over
        9319 s of real off-air HF), and — at BW500, the only bandwidth whose
        vocabulary a recording fixes — the short DBPSK token by its pattern. Our
        own session frames are single-tone on the same grid and match neither.

        The token table is not consulted at BW2300. Everything it could name there
        it names wrongly: it reads none of the thirteen short control bursts of the
        one two-sided BW2300 recording held, and off air it returns 1-2 hits per
        session, the same rate it returns on 73.7 s of DATA overs holding no
        responder burst at all  [see vara_control].

        The preamble is matched at the peer's measured offset, whole carriers and
        the fraction under them alike, as step 5's reader matches it  [see
        ``_peer_shift``, :meth:`_note_peer_offset`, :meth:`_ack_evidence`]. The
        fraction is what this reader could least do without: K0SI's copy of this
        burst was delivered whole and clean on 2026-09-18 and refused by one
        symbol, because two of its three compared preamble pairs had lost their
        second carrier to the neighbouring bin 8 Hz away.
        """
        x = np.asarray(samples, dtype=np.float64)
        if len(x) >= _DATA_OVER_MIN:
            return False
        if (_ack_plateau(x, self._peer_shift, track, self._band, self._grid)
                >= _ACK_PLATEAU):
            return True
        if self._peer_burst_match(x):
            return True
        return self.bw == "500" and VC.detect_token(x) is not None

    def _peer_burst_match(self, x: np.ndarray) -> bool:
        """The peer's own 11-symbol burst, scored whole rather than by its head.

        The preamble belongs to every station and the seven behind it belong to
        this link, so this is the one reader here with a frame keyed to the
        callsign we dialled rather than a shape  [see :data:`_BURST_MATCH`]. It
        answers where the test in front of it cannot: NS0A's answers to two DATA
        overs and VE3WLR's to one, 2026-09-18, seven of seven tail symbols exact
        behind a preamble the channel had taken.

        Role picks the peer's tail out of the link's two, the same way
        :meth:`_tx_control_burst` picks ours, and a link whose tails are not
        measured has nothing to score  [vara_frames, control_bursts] — the
        fallback that method keys is deliberately not taken here, because a tail
        keyed at a peer is a best effort and a tail matched against one is
        evidence.

        Our OWN burst does not pass: it is the other half of the same pair, and
        this station's copy bleeding through its own mute scores 0.39 where the
        peer's scores 3.14. The preamble test cannot tell them apart at all.
        """
        if self.bw not in _BURST_MATCH_BW:
            return False
        pair = VF.control_bursts(self.caller, self.bw)
        if pair is None:
            return False
        theirs = pair[0] if self.role == "responder" else pair[1]
        if _burst_match(x, theirs, self._band, self._peer_shift,
                        self._grid) < _BURST_MATCH:
            return False
        if not self._burst_match_said:
            self._burst_match_said = True
            # NOT an `rx <frame>` line: those say a burst was taken, and this
            # says how it had to be read  [see :meth:`_note_peer_offset`].
            self.io.log(f"{self.called}'s control burst reads whole and not by "
                        "its preamble — the head of its bursts is not reaching "
                        "us")
        return True

    def _peer_over_continue(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the peer answering an INTERMEDIATE over of ours
        [vara_frames, OVER_CONTINUE_CALLER_2300].

        What it asks for is the next over, and reading it is what lets an outbound
        message be longer than one over. A build that locks the 11-symbol burst
        alone reads the end of its own delivery and nothing before it.

        Read off the responder's cable of two two-cable bench sessions, the only
        audio held anywhere in which a stock responder answers a stock CALLER's
        intermediate overs: three copies, eight symbols, 0.340-0.345 s at the key,
        0.170-0.175 s behind the caller's last transmitted sample, and two of the
        three tone for tone identical. It sits on no callsign's lattice — every
        generated frame here is one tone per symbol over 15 or 31 payload symbols,
        and this is eight two-tone ones  [see :func:`_cont_held`].

        THE CONTROL BURST IS DECIDED FIRST, here rather than by the caller's
        ordering, because this shape is that burst's own first eight symbols.
        Their four-pair preamble separates them outright: over every copy on the
        bench cables the 11-symbol bursts hold it for 59-60 alignments and the
        eight-symbol ones for none. The two answers ask for different things, so
        guessing between them is not available.

        THE HALF-BIN CASE IS NOT THIS READER'S TO SOLVE and used to be: KB5LZK's
        carriers lay between our bins on 2026-09-11, its lead pair rounded to
        (64,67) where the frame keys (63,66), and a real request for the next body
        frame was discarded. What stood here was a search of the half bin either
        side, which is three grids for one frame against a plateau swept on one —
        and it answered for this reader alone, so the same station's control burst
        went on being refused. The link's frequency is measured once now and every
        reader is scored on it  [see :meth:`_note_peer_offset`].
        """
        self._continue_down_for = None
        x = np.asarray(samples, dtype=np.float64)
        if len(x) >= _DATA_OVER_MIN:
            return False
        # A damaged NAK must not become an acknowledgment just because its
        # common lead survived. Leave near matches to the timeout retry.
        if (self._peer_nak(x, track, min_symbols=6)
                or self._pair_data_nak_at(x, track, min_symbols=6) is not None):
            return False
        if (_ack_plateau(x, self._peer_shift, track, self._band, self._grid)
                >= _ACK_PLATEAU):
            return False
        if track is None:
            track = _top3_track(x, band=self._band, bin_offset=self._grid)
        at, width = _widest(_cont_held(track[0], track[1], _live_track(x), self._peer_shift))
        if width >= _CONT_PLATEAU:
            offsets = at + width // 2 + _CONT_OFF
            signature = tuple(tuple(int(v - self._peer_shift) for v in sorted(pair))
                              for pair in track[0][offsets, :2])
            link = (self.caller, self.called, self.bw)
            known = (VF.OVER_CONTINUE_RESPONDER_BY_LINK.get(link),
                     VF.OVER_CONTINUE_RESPONDER_DOWN_BY_LINK.get(link))
            if (self.role != "initiator" or self._tx_pending is None
                    or signature in known
                    or (*link, signature) in self._confirmed_continue_pairs):
                if signature == known[1]:
                    self._continue_down_for = self._tx_pending
                return True
            # ACK and NACK share this entire eight-symbol shape. Unknown tails
            # require called-keyed query confirmation before any DATA is retired.
            if (self.turn == _TURN_OURS and self._pending_answer_at is not None
                    and 0 <= time.monotonic() - self._pending_answer_at <= 4.5):
                observed = (self._tx_pending, self._pending_answer_at, signature)
                if self._unclassified_continue != observed:
                    self.io.log("unclassified eight-symbol DATA reply — retaining "
                                f"{self._pending_context()} for query confirmation")
                self._unclassified_continue = observed
            return False
        return (self._peer_known_continue(x, track)
                or self._peer_head_cut_continue(x, track))

    def _peer_known_continue(self, samples, track: tuple) -> bool:
        """Read a measured ACK through weak pairs, without learning a new tail.

        K0SI, 2026-09-20: all eight expected pairs reached the receiver after
        DATA #4 and #6, but one/two fell below the pair-clearance threshold.
        The generic shape requires eight clear pairs because it knows no tail.
        Here all seven tail pairs must match the independent link measurement,
        remain live, and include a clear final pair; at least six pairs must be
        clear. Only an unclear leading pair may disagree. No interior erasure
        or clear contradictory pair is waived.
        """
        if (self.state != VaraState.CONNECTED or self.role != "initiator"
                or self.turn != _TURN_OURS or not self._txq
                or self._tx_pending is None
                or self._tx_pending[2] != _phy.base_level(self.bw)
                or _phy.over_is_last(self._tx_pending[0], self.caller)):
            return False
        link = (self.caller, self.called, self.bw)
        known = [p for p in (VF.OVER_CONTINUE_RESPONDER_BY_LINK.get(link),
                            VF.OVER_CONTINUE_RESPONDER_DOWN_BY_LINK.get(link))
                 if p is not None]
        if not known:
            return False
        bins, clear, _ = track
        idx = np.arange(max(len(bins) - _CONT_SPAN, 0))[:, None] + _CONT_OFF
        live = _live_track(samples)[idx]
        strong = live & (clear[idx, 1] >= _CLEAR_DB)
        heard = np.sort(bins[idx, :2], axis=2)
        for pairs in known:
            exact = (heard == np.asarray(pairs) + self._peer_shift).all(2)
            held = (exact[:, 1:].all(1) & live[:, 1:].all(1)
                    & strong[:, -1] & (strong.sum(1) >= 6)
                    & (exact[:, 0] | ~strong[:, 0]))
            # Eight offsets span 5.3 ms. The weak native #6 holds nine;
            # unlike the generic shape, every tail pair is fixed here.
            if _widest(held)[1] >= 8:
                if pairs == VF.OVER_CONTINUE_RESPONDER_DOWN_BY_LINK.get(link):
                    self._continue_down_for = self._tx_pending
                return True
        return False

    def _peer_head_cut_continue(self, samples, track: tuple) -> bool:
        """One measured full-link continue whose lead the RX mute removed.

        KC9GHZ's 2026-09-12 reply retained seven pairs identical to four stock
        intermediate answers. Its first analysis window was below -75 dBFS.
        A loud wrong lead, any lost interior symbol, or another link has no
        exception. The caller applies the ordinary ACK and known-NAK vetoes
        before reaching this path; the generic eight-symbol shape stays strict.
        """
        pairs = VF.OVER_CONTINUE_RESPONDER_BY_LINK.get(
            (self.caller, self.called, self.bw))
        if (pairs is None or self.bw == "500" or self.state != VaraState.CONNECTED
                or self.role != "initiator" or self.turn != _TURN_OURS
                or self._tx_pending is None
                or self._tx_pending[2] != _phy.base_level(self.bw)
                or not self._txq):
            return False
        body = self._tx_pending[0]
        if (_phy.over_is_last(body, self.caller)
                or len(_phy.vara_payload(body, caller=self.caller, body_len=len(body)))
                != self._tx_payload_size()):
            return False
        # The transport's post-TX guard may omit the lead altogether. Restore
        # its missing time, not its evidence: all seven measured tail pairs
        # must still be present, live and clear in actual received samples.
        # Without this, an onset before sample zero cannot enter the search.
        samples = np.pad(np.asarray(samples), (MK.HOP, 0))
        bins, clear, _ = _top3_track(samples, band=self._band,
                                     bin_offset=self._grid)
        live = _live_track(samples)
        n = max(len(bins) - _CONT_SPAN, 0)
        idx = np.arange(n)[:, None] + _CONT_OFF
        found = np.sort(bins[idx][:, 1:, :2], axis=2)
        expected = np.asarray(pairs[1:], dtype=np.int32) + self._peer_shift
        exact_tail = (found == expected).all(axis=(1, 2))
        complete_tail = (live[idx[:, 1:]]
                         & (clear[idx[:, 1:], 1] >= _CLEAR_DB)).all(1)
        dead_lead = ~live[idx[:, 0]]
        return _widest(dead_lead & complete_tail & exact_tail)[1] >= _CONT_PLATEAU

    def _peer_idle_response(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the gateway's answer to an idle frame of ours
        [vara_frames, SESSION_IDLE_RESPONSE].

        Located by payload rather than by preamble: the frame opens on a single
        tone, which identifies nothing and gives :func:`vara_mfsk.lock_preamble`
        nothing to lock. Its 31 payload tones are fixed by the call we dialled, so
        the same alignment search that recovers a connect-response out from under
        our own transmission finds this one too, and the ordinary recogniser then
        decides  [see :meth:`_response_by_payload`].

        Measured both ways. It accepts all ten occurrences across the two off-air
        gateway sessions, at 31/31 tones where the burst lies whole inside the
        window; over the shared regression corpus — 33 recordings of real off-air
        HF from four continents, none of it addressed to us — it accepts nothing,
        0 of 4975 (window, callsign) trials, and no wrong callsign accepts anywhere
        in the 216 s of the two sessions that do hold it.
        """
        x = np.asarray(samples, dtype=np.float64)
        if not self.called or len(x) >= _DATA_OVER_MIN:
            return False
        kind = VF.SESSION_IDLE_RESPONSE
        return self._peer_frame(x, kind, self.called, track)[2]

    def _peer_responder_idle(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the gateway keying its own idle cadence
        [vara_frames, SESSION_RESPONDER_IDLE, SESSION_RESPONDER_OVER_IDLE].

        It is not addressed to us and asks for nothing: a gateway keys it while IT
        holds the turn, on its own clock, whether or not we have said anything.
        What it settles is the one thing `_MAX_WITHOUT_PROGRESS` is counting the
        absence of — a peer that is still transmitting — and until this was read
        nothing did. KE8LVA keyed it thirteen times in 47.7 s on 2026-08-23, every
        3.406 s, and the give-up budget closed the link on a station that had never
        stopped talking.

        Located by payload, and with no length guard, for :meth:`_peer_drained`'s
        reasons. Measured over both 2026-08-23 tapes, sliding 1.9 s windows every
        0.5 s: 73 accepting windows on KE8LVA, in sixteen runs 3.4 s apart, and
        NONE at all on KB3AC-10, which keyed this frame never. Nothing on either
        tape reads it as `session-idle-response` or `session-drained-responder`, and
        the ten windows that do take a drained-responder are KB3AC-10's two turn
        grants and are not these.

        TWO FRAMES ARE THIS CADENCE AND NEITHER NAMES THE SITUATION. 683 is what a
        stock responder keys when its last over went unacknowledged — sixteen
        keyings on a 3.155 s cadence with 1587 bytes still in its own queue, off
        the cables of 2026-09-09 — and it is the same frame it keys with nothing
        left at all, so the frame cannot tell a stall from an idle. 745 is what it
        answers a turn-request with while it still holds the turn, four keyings
        3.2 s apart before the grant on that bench; on the air KC9GHZ keyed it
        thirteen times after our answer to its second greeting over on 2026-08-20,
        which is the stall position. Both mean the peer is transmitting and
        listening between bursts, and 745 in particular is NOT a grant  [see
        _peer_drained]. What separates a stall from an idle is the turnaround
        geometry and what this end is owed, never the frame  [see _reack].

        An accept records which of the two it was and how much audio was already
        held past its last symbol, which is what says whether the peer's own
        listening gap is still open  [see :meth:`_reack`, :meth:`_peer_drained`].
        """
        x = np.asarray(samples, dtype=np.float64)
        self._idle_complete_exact = False
        if not self.called:
            return False
        for kind in (VF.SESSION_RESPONDER_IDLE, VF.SESSION_RESPONDER_OVER_IDLE):
            _heard, at, matched = self._peer_frame(x, kind, self.called, track)
            if matched:
                self._idle_kind = kind
                self._idle_held = (len(x) - at - _span(kind)) / MK.FS
                if self._held_answer is not None:
                    # The ordinary idle reader accepts a partial payload and
                    # can name DATA as an idle. Releasing an ACK hold needs an
                    # independent full-payload fit, complete and fresh.
                    native = VF.for_bw(kind, self.bw)
                    want = np.asarray(VF.handshake_tones(self.called, native))
                    heard, exact_at = _payload_fit(x, native, want, track, self._band)
                    clear = heard[1:] >= 0
                    held = (len(x) - exact_at - _span(native)) / MK.FS
                    self._idle_complete_exact = bool(
                        clear.sum() >= 24 and clear[-1]
                        and np.all(heard[1:][clear] == want[1:][clear])
                        and 0 <= held <= _GRANT_FRESH_S)
                return True
        return False

    def _a_gap(self) -> bool:
        """May this station key into the turnaround the idle just named?

        A NAMED FRAME IS NOT A GAP BY ITSELF. On the fetch arm of 2026-09-09 the
        32-symbol recogniser took a `session-responder-over-idle` out of the
        SECOND BLOCK of a two-block window, at +0.257 s, and the ladder keyed a
        rung on top of it: 89 bytes the peer had already sent, gone, and nothing
        left that asks for them. What says it was not a gap is not the audio —
        the six-column reading is too unsteady inside one emission to veto on,
        0.25-0.92 of that block's own reference — but this station's own state:
        it had read the window's first block and was holding its answer for the
        end of the window, which is a peer mid-transmission on the one reading
        that decoded a frame  [see :meth:`_answer_over`].
        """
        if self._held_answer is None:
            return True
        # A complete called-keyed idle before a second DATA block could finish
        # rules out a two-block window. Reserve half a second for scan/tail
        # latency so a later idle cannot acknowledge an unread second block.
        # KC9GHZ's first idles on 2026-09-20 fit 31/31 and arrive 3.5 s after
        # the held decode; the shortest wide DATA frame needs 4.21 s.
        if (self.bw in ("2300", "2750")
                and getattr(self, "_idle_complete_exact", False)
                and _IDLE_PAIR_MIN_S * MK.FS <= self._held_samples
                < _OVER_NEED - _STREAM_BLOCK):
            self._held_idle_early = True
            return False  # Release after this receive callback's window scan.
        # WHETHER THIS IS A SECOND EMISSION IS DECIDED HERE, at the naming, and
        # not later as audio piles up: the gap that matters is the one between two
        # NAMINGS, and a pair 0.3 s apart out of one block stays one emission
        # however long the hold then runs  [see _IDLE_PAIR_MIN_S,
        # _release_held_answer].
        if (self._held_idles
                and self._held_samples - self._held_idle_at
                >= _IDLE_PAIR_MIN_S * MK.FS):
            self._idle_pair_seen = True
        self._held_idle_at = self._held_samples
        self._held_idles += 1
        self.io.log(f"{self._idle_kind.name} named inside {self.called}'s "
                    "emission, not a gap — a window of its own is still open")
        return False

    def _peer_drained(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the gateway saying its own queue has drained
        [vara_frames, SESSION_DRAINED_RESPONDER] — what it answers a turn-request
        with.

        Three occurrences, two sessions, one gateway: KC9GHZ keyed this frame
        0.100-0.105 s after our turn-request's last sample on 2026-08-17 (73.923 s)
        and twice on 2026-08-22 (79.141 s, 90.916 s), at 29-31 of 32 tones through
        our own receiver and 32/32 through an independent receiver 103 mi away.
        Located by payload for :meth:`_peer_idle_response`'s reason: the frame
        opens on a single tone that identifies nothing.

        NO LENGTH GUARD, unlike every other recogniser here, and that is the half
        of this the waveform does not supply. On the 2026-08-22 session the gate
        rode the band noise open past every one of these answers and force-closed
        at ``SEG_MAX_S``, so what the segmenter handed the handshake was a 6.016 s
        bracket holding a 1.387 s frame — over :data:`_DATA_OVER_MIN`, which is
        what silently declined it. Sliding that whole session in 6 s windows
        accepts only the two that hold an answer, best clean window 7 of 32.

        WHAT IS GUARDED IS THE BURST'S LAST SYMBOL. A start may sit before the
        audio — the head a bracket cut away, or the head our own mute swallowed,
        is what a fit is allowed to score around — but a last symbol past its end
        is a burst still arriving, and this frame is answered by keying a 4.4 s
        DATA over at a half-duplex peer. What :meth:`_stream_grant` puts in front
        of this is a window length, which is a proxy: it measures the buffer, and
        a buffer carrying turnaround silence ahead of the answer passes it while
        the answer is still on the air. Measured with the buffer opened at our own
        key-down rather than at our last sample, which is that much silence: the
        grant was taken 0.49 s into a 1.39 s answer and the over went out across
        the rest of it, and the peer read none of it in 3 fetches of 3.
        """
        x = np.asarray(samples, dtype=np.float64)
        if not self.called:
            return False
        kind = VF.SESSION_DRAINED_RESPONDER
        _heard, at, matched = self._peer_frame(x, kind, self.called, track)
        if not matched:
            return False
        self._grant_held = (len(x) - at - _span(kind)) / MK.FS
        return self._grant_held >= 0     # the last symbol is in the audio

    def _drained_hands_over(self, samples, track: tuple | None = None) -> bool:
        """True when the frame :meth:`_peer_drained` names is the handover this
        station is already waiting for.

        THE FRAME SAYS THE PEER'S QUEUE AND NOT THE TURN, so what makes it a
        handover is this end's own state: the last over of the peer's delivery
        was acknowledged, its release is owed, and a block of ours stands behind
        it  [see :meth:`_key_over_answer`]. Both BW500 arms of 2026-09-11 put
        stock 4.9.0 there — this frame 0.74 s behind our final ACK and again
        every 11.4 s, where the run before it keyed the 17-symbol release — and
        it is the frame stock keys to its own stock caller after a delivery, at
        55.586-56.963 s of the two-stock tape. That caller answered 0.242 s
        after its last sample; this one spent two rungs of the release ladder,
        25.9 s per arm, before the turn-request drew the next copy
        [analysis/stock500-chain2, analysis/stock500-recovery].

        Nothing else here moves. An empty queue owes the peer no transmission,
        a release that is not owed leaves the frame to :meth:`_stream_grant`'s
        own window — between a request of ours and its answer, which is the
        `_TURN_ASKED` state this one is not — and a ladder already spent owes
        nothing to take it with.

        IT IS THE STATE THAT MAKES IT A HANDOVER, NOT THE BANDWIDTH. The narrow
        bandwidth is where it was first read, but K0SI on 40 m keyed the
        drained-responder at BW2300 twice on 2026-09-16 — 67.3 and 76.7 s, 31/31
        tones, 0.12 s after our final ACK — and the run named "no release in
        cadence" and re-keyed. So the same state test and freshness bound apply
        at every session bandwidth; a wide gateway also keys it at a turn-request
        of ours, which is `_stream_grant`'s own window and the `_TURN_ASKED`
        state this end is not in here.

        The length guard and the age bound are `_stream_grant`'s, for its
        reasons: this frame is answered by keying a 4.4 s DATA over, so it is
        taken neither from audio too short to hold the frame — `_peer_drained`
        carries no length guard of its own — nor from audio whose newest sample
        is a block past its last symbol, where the peer's own gap has gone.
        """
        return (self._answer_owed == _OWED_RELEASE
                and bool(self._txq or self._tx_pending is not None)
                and len(samples) >= _SESSION_NEED
                and self._peer_drained(samples, track)
                and self._grant_held <= _GRANT_FRESH_S)

    def _peer_responder_release(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the peer handing the turn over
        [vara_frames, SESSION_TURN_RELEASE_RESPONDER].

        The responder's counterpart of the release this station keys, and until
        the two-VARA bench of 2026-08-26 read it whole it was in this file as a
        refusal. Two gateways answered our FIRST turn-request with it — KE8LVA at
        50.603 s of the 2026-08-26 12:59z recording, KB3AC-10 at 75.575 s of the
        2026-08-23 04:52z one — and this station logged "did not grant the turn"
        and asked again, at both. The second ask drew SESSION_DRAINED_RESPONDER,
        which is the same state at 32 symbols, and that one was taken. So both
        sessions granted the first request and neither grant was acted on.

        Located by payload and with no length guard, for :meth:`_peer_drained`'s
        reasons. Over the three sessions on disk that hold no such answer — 550 s,
        sliding 1.9 s windows every 0.5 s — it accepts none of 947.

        An accept also records how much audio was already held past the frame's
        last symbol, which is the part of the answer's own delay this file owns
        and the part that had to be read back off a recording the one time the
        fix flew  [see :meth:`_took_responder_release`].
        """
        x = np.asarray(samples, dtype=np.float64)
        if not self.called:
            return False
        kind = VF.SESSION_TURN_RELEASE_RESPONDER
        _heard, at, matched = self._peer_frame(x, kind, self.called, track)
        if not matched:
            return False
        self._grant_held = (len(x) - at - _span(kind)) / MK.FS
        return self._grant_held >= 0      # the last symbol is in the audio [_peer_drained]

    def _peer_wants_turn(self, samples, track: tuple | None = None) -> bool:
        """True when the peer is asking to become the sender
        [vara_frames, SESSION_TURN_REQUEST_RESPONDER].

        Keyed to the CALLER, as both turn frames in that file are, so it is scored
        against our own callsign and not the peer's.

        The frame the bench spent whole sessions not holding: a stock VARA HF
        4.9.0 with a reply on its data port keyed it eleven times into the gap
        after our release on 2026-08-26 and never transmitted, and three earlier
        sessions of the same shape ended with twelve askings, no greeting and no
        payload either way. Answering it is what a caller with an empty queue owes
        a peer that has something to send.

        ALONE AMONG THE `_peer_*` RECOGNISERS THIS ONE IS ANSWERED BY
        TRANSMITTING, so it carries the two guards the others can do without. A
        6 s bracket of 40 m band noise off the KC9GHZ session takes it at three
        consecutive windows without one: `recognize` scores the comparable tones
        and `_payload_alignment` sweeps a start every 32 samples, so the wider the
        window the more chances noise has to put eight of them in agreement. The
        others survive that because a grant is only looked for between a request of
        ours and its answer, and this one is asked at any time — so what bounds it
        is its own shape: a bracket that can hold this frame and little else, and
        a payload most of which the receiver actually delivered. `recognize` scores
        the comparable tones and lets the rest go, which is right for a frame read
        through our own transmit mute and wrong for one that keys back. Swept every
        half second over 971 windows of a clear 40 m channel and two gateway
        sessions, at both the bare frame length and the padded bracket, it accepts
        three — and all three are one burst, at 15.88 s of the KC9GHZ 2300 session,
        seen from the starts either side of it. That one is the frame: 30 of the 30
        tones its alignment finds comparable, where nothing else in the recording
        reaches 20. Without the payload bound the same sweep takes band noise.
        """
        return self._turn_request_at(samples, track) is not None

    def _turn_request_at(self, samples, track: tuple | None = None) -> int | None:
        """The existing turn-request evidence, with its fitted first sample."""
        x = np.asarray(samples, dtype=np.float64)
        if not self.caller:
            return None
        kind = VF.SESSION_TURN_REQUEST_RESPONDER
        n_sym = len(kind.preamble) + kind.n_payload
        frame = (n_sym - 1) * MK.HOP + MK.STRIDE
        # The gate's own padding, in symbols: four frames of pre-roll and six of
        # hangover is ten 1024-sample frames, which is five HOPs, and the closing
        # frame can add a sixth. Below the frame length the burst is not all here.
        if not frame - MK.HOP <= len(x) <= frame + 6 * MK.HOP:
            return None
        fitted, at, matched = self._peer_frame(x, kind, self.caller, track)
        if sum(1 for t in fitted if t >= 0) < _ASK_MIN_HEARD:
            return None
        return at if matched else None

    def _reset_connect_ask_search(self) -> None:
        self._connect_ask_buf = np.zeros(0)
        self._connect_ask_due = _SESSION_NEED

    def _stream_connect_ask(self, samples) -> bool:
        """Read a complete, recent turn request during setup or a held link.

        K0SI's 2026-09-09 replies match the existing caller-keyed recognizer,
        including 31/31 payload tones, but its energy brackets omit the frame
        or cut it below the length guard. Search a bounded raw-audio window with
        the SAME evidence. A fitted frame must have ended a symbol ago: accepting
        a partial payload immediately would transmit across the peer's tail.
        Only the newest window is searched, so a delayed poll cannot answer an
        old request after the peer has resumed transmitting.

        KC9GHZ's 2026-09-11 morning recording also holds five complete requests
        after our final DATA frame, while CONNECTED. The energy gate kept the
        first two inside one ten-second bracket, so neither reached the bounded
        bracket reader. Use the same complete-and-fresh evidence in that state.
        A request with our DATA still pending is observed but keys nothing; the
        rest of this audio must still reach the ACK and DATA searches.
        """
        connecting = (self.state == VaraState.CONNECTING
                      and self.step == _I_LINKSETUP_SENT)
        connected = self.state == VaraState.CONNECTED
        if self.role != "initiator" or not (connecting or connected):
            self._reset_connect_ask_search()
            return False
        if connecting:
            self._stream_owns_connect_ask = True
            if not len(self._connect_ask_buf):
                # A setup answer can be the 16-symbol confirmation. Waiting
                # for a 32-symbol turn request makes a prompt confirmation
                # stale before its first scan (K0SI, 2026-09-20 04:08 UTC).
                # Subsequent scans keep the existing turnaround cadence and
                # both recognizers retain their complete-and-fresh guards.
                self._connect_ask_due = _span(
                    VF.for_bw(VF.SESSION_CONNECT_CONFIRM, self.bw))
        self._stream_owns_turn_request = True
        buf = np.concatenate([self._connect_ask_buf, np.asarray(samples, float)])
        if len(buf) < self._connect_ask_due:
            self._connect_ask_buf = buf
            return False
        buf = buf[-(_SESSION_NEED + 6 * MK.HOP):]
        self._connect_ask_buf = buf[-(_SESSION_NEED + MK.HOP):]
        self._connect_ask_due = len(self._connect_ask_buf) + _TURNAROUND_STEP
        at = self._turn_request_at(buf)
        if at is None:
            # The same fresh answer slot can hold a Trimode responder's connect
            # confirmation instead — a different frame, taken the same way.
            return connecting and self._connect_confirmed(buf)
        held = len(buf) - at - _span(VF.SESSION_TURN_REQUEST_RESPONDER)
        if held < MK.HOP:
            # A request already named needs its remaining tail/guard, not a
            # whole additional scan interval (notably on 0.5 s input blocks).
            self._connect_ask_due = len(self._connect_ask_buf) + min(
                _TURNAROUND_STEP, MK.HOP - held)
            return False
        if held > 6 * MK.HOP:
            return False
        # Consuming the complete request drops its whole match plateau, even
        # when pending DATA prevents us from transmitting. The stream owns the
        # bracket route too, so the same burst cannot be reported again later.
        self._reset_connect_ask_search()
        if connecting:
            self.io.log(f"rx turn-request from {self.called} on the receive stream "
                        f"({held / MK.FS:.3f} s past its last symbol) — the link is up")
            self._connected(confirm=False)
            self._took_turn_request()
            # A QUEUE IS STILL OWED A CHANNEL, the same debt the confirmation
            # route carries  [see _connect_confirmed]. With a block queued the
            # request is declined — "retaining the turn" — but the turn came up
            # the PEER's, so retaining it retains nothing and no ask is ever
            # keyed: on the 2026-09-16 K0SI tape the link came up at 2.9 s and
            # the block was still queued through six keepalives at the close.
            # Answered here rather than in `_took_turn_request`, whose contract
            # is the frame and not the state a connect leaves behind.
            if self.turn == _TURN_PEER and self._txq:
                self._ask_for_turn()
            return True
        self.io.log(f"rx turn-request from {self.called} on the receive stream "
                    f"({held / MK.FS:.3f} s past its last symbol) — already connected")
        return self._took_turn_request(raw=True)

    def _connect_confirmed(self, buf) -> bool:
        """A Trimode responder's connect confirmation, on the connect-response
        stream at lattice position 1 and keyed to the called station
        [vara_frames, SESSION_CONNECT_CONFIRM].

        K0SI answered our FIRST link-setup with it at +0.12 s on 80 m and then
        ignored our retries. Eight stock 4.9.0 link-setup runs of 2026-09-16
        across both wide bandwidths, clean, retried and mismatched, draw it in
        none. Stock does key the same payload as a DATA NACK, so this use
        remains confined to an outstanding link setup and never relaxes the connect
        gate the two-tone ack and turn-request routes hold: it is read only where a
        link-setup of ours is outstanding, in the same fresh answer slot the
        turn-request-before-connected route reads. Taking it brings the link up
        and stops the link-setups.

        NO SESSION-CONFIRM GOES BACK, and the reason is this frame's own and not
        the turn-request route's: the peer has confirmed the link and gone quiet,
        so nothing is owed on the air and a confirm keyed here would answer a
        station that has stopped asking.

        A QUEUE IS STILL OWED A CHANNEL. The turn starts the peer's, so a host
        block queued before the link came up leaves on an ask like any other  [see
        :meth:`_ask_for_turn`] — without one it sits there, and the session of
        2026-09-16 closed with the block still queued and not one request keyed.
        With nothing queued the turn stays where it is: the gateway's own
        turn-request train follows ~5.7 s later and is answered where those are.
        """
        if not self.called:
            return False
        kind = VF.for_bw(VF.SESSION_CONNECT_CONFIRM, self.bw)
        frame = _span(kind)
        window = buf[-(frame + 6 * MK.HOP):]
        if len(window) < frame - MK.HOP:
            return False
        fitted, at, matched = self._peer_frame(window, kind, self.called)
        if sum(1 for t in fitted if t >= 0) < _RESP_MIN_TONES or not matched:
            return False
        held = len(window) - at - frame
        # Like the turn-request route, leave one symbol beyond the fitted end.
        # The tone fit can precede the actual tail slightly; a queued host block
        # asks for the channel immediately when this route confirms the link.
        if not MK.HOP <= held <= 6 * MK.HOP:
            return False
        self._reset_connect_ask_search()
        self.io.log(f"rx connect-confirm from {self.called} on the receive stream "
                    f"({held / MK.FS:.3f} s past its last symbol) — the link is up")
        self._connected(confirm=False)
        if self._txq:
            self._ask_for_turn()
        return True

    def _peer_over_answer(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the gateway answering a DATA over of ours
        [vara_frames, SESSION_RESPONDER_OVER_ANSWER].

        One occurrence, because one over: KE8LVA keyed it 0.17 s after the last
        sample of the first payload over this station ever put on the air, at 32 of
        32 tones, and the session's remaining 110 s hold nothing else from it. The
        log called that whole stretch "nothing back from KE8LVA" while the answer
        to the over was on the tape.

        It settles that the over was heard, which with a queue is the same thing
        the two shorter answers settle and continues the delivery. With nothing
        left to send it stays what it was: the one recording that holds it holds
        no second gateway transmission afterwards, so no recording says what a
        station keys here with an empty queue  [see _stream_answer]. Swept the
        same way as :meth:`_peer_responder_release`, it accepts none of the 947
        windows of gateway audio that hold no over of ours.

        An accept also records how much audio was already held past the frame's
        last symbol. The frame is named off a third of it — 0.4 s of the 1.366 s —
        and keying there puts our next over across a second of a peer that is
        still transmitting, which a half-duplex peer decodes none of.
        """
        x = np.asarray(samples, dtype=np.float64)
        if not self.called:
            return False
        kind = VF.SESSION_RESPONDER_OVER_ANSWER
        _heard, at, matched = self._peer_frame(x, kind, self.called, track)
        if not matched:
            return False
        self._answer_held = (len(x) - at - _span(kind)) / MK.FS
        return True

    def _peer_responder_nak(self, samples, track: tuple | None = None) -> bool:
        """True when this audio is the peer NAKing a DATA over of ours
        [vara_frames, SESSION_OVER_NAK_RESPONDER].

        The responder counterpart of the NAK this station keys during recovery,
        keyed to the CALLED station and at the family's responder SEED_OFF. KC9GHZ
        answered every final-answer query of ours with it on the 2026-09-16 BW2750
        tape — three copies, 30-31 of 32 payload tones each. Located by payload for
        :meth:`_peer_over_answer`'s reasons, and it is a positive ask rather than a
        cadence: taking it resends the outstanding over  [see
        :meth:`_took_responder_nak`].

        THE LAST SYMBOL HAS TO BE IN THE AUDIO, because what this draws is a 4.4 s
        DATA over keyed at a half-duplex peer. The frame runs 1.366 s and the arm
        above it opens at ``_SESSION_MIN``, 0.726 s, so the fit names it off its
        first two thirds: driven over the KC9GHZ tape in 0.125 s blocks the accept
        lands 0.616 s INSIDE the burst and the over goes out on top of the NAK it
        is answering, spending two rungs of the retry budget on one ask. The next
        scan names the same NAK 0.134 s past its last symbol, so waiting costs
        nothing  [see :meth:`_peer_drained`, which guards the same way for the same
        reason]. Past ``_GRANT_FRESH_S`` the peer's own listening gap has gone and
        the ask is stale.
        """
        x = np.asarray(samples, dtype=np.float64)
        if not self.called:
            return False
        self._responder_nak_wait_samples = 0
        kind = VF.SESSION_OVER_NAK_RESPONDER
        _heard, at, matched = self._peer_frame(x, kind, self.called, track)
        if not matched:
            return False
        self._nak_held = (len(x) - at - _span(kind)) / MK.FS
        if self._nak_held > _GRANT_FRESH_S:
            # Say so. A NAK the scan reached too late is the peer asking and
            # being answered with nothing, and a transcript that prints only the
            # asks it acted on reads that as a peer which never spoke  [see
            # _stream_grant, which reports its own stale grants the same way].
            if not self._stale_nak_logged:
                self._stale_nak_logged = True
                self.io.log(f"{kind.name} from {self.called} named "
                            f"{self._nak_held:.2f} s behind the newest audio — "
                            "the peer's listening gap has gone, so the over it "
                            "asks for is not keyed on it")
            return False
        if self._nak_held < 0:
            # A payload fit can identify the NAK before its tail arrives. Wake
            # at that tail instead of the next half-second scan: the latter
            # rejected K0SI's complete queries at +0.51 s on 2026-09-19.
            self._responder_nak_wait_samples = int(np.ceil(-self._nak_held * MK.FS))
            return False
        return True

    def _took_turn_request(self, *, raw: bool = False) -> bool:
        """The peer asked to become the sender [vara_frames,
        SESSION_TURN_REQUEST_RESPONDER].

        With nothing of our own queued the answer is the release, whatever we
        believe the turn already is — a peer that is still asking has not read the
        one we keyed, and the frame is the only thing that says so. With a queue
        the turn stays here: our own over drains it and the release follows on the
        peer's answer to that, which is where the two-VARA bench puts it.
        """
        if self._txq or self._tx_pending is not None:
            if self._final_ack_candidate():
                self.io.log(f"rx turn-request from {self.called} — "
                            "a final short over is still unacknowledged")
                if (raw and self._stream_owns_turn_request
                        and self._final_query_samples >= _SESSION_NEED + MK.HOP
                        and self._final_query_fresh()):
                    self._final_ack_confirmed = self._tx_pending
                    return self._finish_final_answer()
                return self._query_final_answer()
            if raw and self._final_over_was_read():
                # A responder always keys the ACK before a turn-request, and no
                # stock tape holds a turn-request without a preceding ACK, so a
                # turn-request IN THE TURNAROUND OF our final short over is proof
                # it was read — the BW2300 equivalent of the BW2750 final-answer
                # confirmation the 2026-09-10 two-stock BW2300/BW2750 handover
                # bench measured, where every wide handover released on the
                # 17-symbol form [K0SI 40 m 2026-09-16]. Grant the turn rather
                # than retaining it and closing.
                self.io.log(f"rx turn-request from {self.called} behind our final "
                            "over — a responder acks before it asks, so the over "
                            "was read; granting the turn")
                changes_turn = self.turn != _TURN_PEER
                if not self._release_turn():
                    # The release never reached the air, so the turn is still ours
                    # and so is the over: retiring it here would drop the last
                    # block of the message on a burst the peer never heard. And
                    # the release is NOT owed — `_release_turn` owes it on every
                    # refusal, and that branch runs ahead of the retry and hands
                    # the turn away with the over still pending, which strands it
                    # where nothing retries it  [see idle_keepalive]. The over
                    # comes first; the turn goes back when it is acknowledged.
                    self._release_owed = False
                    self.io.log("the release was not transmitted — retaining the "
                                "turn and the final over for the retry")
                    return False
                self._close_echo_window()
                self._owe_nothing()
                self._tx_pending = None
                self._tx_retries = 0
                self._reset_final_ack_recovery()
                self._into_our_turn = 0
                if changes_turn:
                    self._progressed()
                    self._released = True
                return True
            pending = int(self._tx_pending is not None)
            self.io.log(f"rx turn-request from {self.called} — "
                        f"{pending} unacknowledged block(s), "
                        f"{len(self._txq)} queued; retaining the turn (not progress)")
            return False
        changes_turn = self.turn != _TURN_PEER
        keyed = self._release_turn()
        if keyed and changes_turn:
            self._progressed()
            # _progressed clears an earlier release's poll state; this release
            # has just reached the air and still owns the next peer poll.
            self._released = True
        return keyed

    def _took_idle_response(self) -> None:
        """The peer answered an idle frame of ours [vara_frames,
        SESSION_IDLE_RESPONSE].

        What it settles is that the peer is listening, and while the turn is ours
        that is the precondition for the next over — so a queue drains on it
        exactly as it drains on the peer's control burst. What it does NOT settle
        is the turn itself: no recording holds this frame answering a turn-request.

        An empty queue keys nothing, and gives nothing back either. The real VARA
        that recorded these sessions took this answer and said nothing for 11.7 s,
        and that silence is a station which had ALREADY RELEASED THE TURN: the
        consequence, not the act. Reading it as "do not key this instant" is what
        left the 2026-08-22 session holding a turn it could not spend and probing
        for an answer the gateway had no way to give. The release is the idle
        cadence's [see idle_keepalive], which reads the queue once a tick rather
        than once per burst the peer chooses to answer with.
        """
        self._close_echo_window()
        self._into_our_turn = 0
        # The peer answered a burst of ours, which is what the idle budget is
        # counting the absence of  [see _MAX_WITHOUT_PROGRESS].
        self._progressed()
        if self.turn == _TURN_OURS and (self._txq or self._tx_pending is not None):
            action = ("soliciting the outstanding over's answer" if self._tx_pending else
                      f"keying the next of {len(self._txq)} queued over(s)")
            self.io.log("rx session-idle-response — the peer is listening; "
                        + action)
            self._tx_data_over()
        else:
            self.io.log("rx session-idle-response — the peer is listening")

    def _took_over_continue(self) -> None:
        """The continue decoder, rather than a query's arbitrary response, won."""
        if (self._tx_pending is not None
                and getattr(self, "_continue_down_for", None) == self._tx_pending):
            self._lower_acked_delivery()
        self._continue_down_for = None
        self._took_control_burst(continue_reply=True)

    def _took_control_burst(self, *, continue_reply: bool = False) -> None:
        """The lawful answer to an over of ours: the turn is ours after all,
        whatever arrived into it before this  [see _over_into_our_turn]."""
        if (self._intermediate_query_for is not None
                and self._intermediate_query_for == self._tx_pending
                and not continue_reply):
            self.io.log("control after intermediate query probe is not a "
                        "validated continue — retaining pending DATA (not progress)")
            return
        self._intermediate_query_for = None
        self._close_echo_window()
        self._into_our_turn = 0
        self._progressed()
        self._owe_nothing()
        self._tx_pending = None
        self._tx_retries = 0
        self._reset_final_ack_recovery()
        if not self._tx_data_over():
            self._release_turn()

    def _took_stall_answer(self) -> None:
        """The peer answered while it still owed us something: whatever the ladder
        was asking for, it read us  [see :meth:`_reack`, :meth:`_stalled`].

        Nothing is keyed. The frame settles that the peer is listening, which is
        what the give-up budget counts, and the turn is where it already was.
        """
        if self._owed_block:
            self.io.log(f"{self.called} answered a control, but the unread "
                        "window still needs retransmission")
            return
        self._close_echo_window()
        self._progressed()
        self._owe_nothing()

    def _took_responder_release(self) -> None:
        """The peer handed the channel over without being asked: take it, and key
        into the turnaround it opened  [vara_frames, SESSION_TURN_RELEASE_RESPONDER].

        A gateway releases after its own over has been answered, whether or not
        anything asked — three arms of 2026-08-29 keyed it 0.13-0.15 s behind our
        control burst, and it is the same handover the bench read between two stock
        4.9.0s. Nothing ran the search there: `_stream_grant` owns this frame only
        between a request of ours and its answer, and the turn was the peer's.

        What is owed is a transmission, for `_release_turn`'s reason. With a queue
        we send it; without one the channel goes straight back, which is what lets
        a gateway finish a greeting it broke off mid-word to hand over.

        WHAT AN EMPTY QUEUE ANSWERS WITH is the release, and that is measured
        rather than chosen. No held recording shows a caller keying a DATA over
        with nothing to put in it — in all seventeen bench instances the caller had
        payload — but two of the same bench sessions show what a caller with
        nothing keys instead, and it is this frame: SESSION_TURN_RELEASE, keyed
        once the peer has answered its over and it has nothing more queued, and in
        both the peer answered by transmitting what was sitting on its data port
        [vara_frames, SESSION_TURN_RELEASE]. SESSION_DRAINED says the same at 32
        symbols over two more sessions and two more callsign pairs. So the frame
        was never the fault. It went out 1.31-1.36 s after the gateway's release on
        2026-08-29, against the 0.075-0.126 s a caller answers in, and by then the
        gateway had stopped listening  [see :func:`_next_scan`].

        The figure is logged because that fix flew once with its own number
        nowhere in the log, and had to be read back off the recording: on
        2026-08-30 at 05:00z the gateway's last symbol landed at 100.693 s and
        this station keyed at 100.770, so the whole turnaround was 77 ms against
        a window of 126.

        WHAT IS PRINTED IS THE FIRST HALF OF THAT and not the whole of it: the
        audio already held past the release's last symbol when the frame was
        named, which is the term :func:`_next_scan` moves and the only one this
        file can shorten. The rest is the transmit path, which no counter here
        sees. A negative figure is a frame named before its own tail arrived —
        the partial-payload tolerance allows it and it costs nothing.
        """
        self.turn = _TURN_OURS
        self._into_our_turn = 0
        self._progressed()
        self._owe_nothing()
        self.io.log(f"rx session-turn-release-responder — {self.called} handed "
                    f"the channel over unasked, named {self._grant_held:+.3f} s "
                    "from its last symbol")
        if not self._tx_data_over():
            self._release_turn()

    def _took_drained_handover(self) -> None:
        """The peer's queue has drained while its release was owed and ours was
        not: the turn is ours and the queue goes into the turnaround the frame
        opened  [see :meth:`_drained_hands_over`].

        What :meth:`_took_responder_release` does with the 17-symbol form of the
        same handover, because the two frames are one state keyed at 17 symbols
        and at 32  [see :meth:`_peer_responder_release`].
        """
        self.turn = _TURN_OURS
        self._into_our_turn = 0
        self._progressed()
        self._owe_nothing()
        self.io.log(f"rx session-drained-responder — {self.called} has nothing "
                    f"more to send, taking the turn, named "
                    f"{self._grant_held:+.3f} s from its last symbol")
        if not self._tx_data_over():
            self._release_turn()

    def _took_poll(self) -> None:
        """The peer's control burst while the turn is its own and nothing is
        queued here: it is polling, and the answer is the idle cadence's burst
        keyed into its turnaround.

        A stock VARA HF 4.9.0 responder handed the channel keys this burst every
        2.3 s from 0.4 s after our release, for 60 s if nothing answers it, with
        its reply queued the whole while. On 2026-09-03 the reply came in every
        run where a keepalive of ours landed whole in the gap behind one poll: the
        poll after it was the last, the responder keyed its turn-request 5.5 s
        later, our release answered that and the over followed — 5 of 5. Keyed
        across a poll the keepalive was not heard and the polling ran on; keyed
        on our own cadence it lands across one two times in five. A release in
        the poll's turnaround restarts the polling: 38 of them answered 28 polls
        on 2026-09-03 and drew nothing.

        One answer per cadence and not one per poll, because a responder that
        cannot hear us keeps polling and every answer is charged to the budget
        that closes a dead link.
        """
        self._polls += 1
        if self._polls % _POLLS_PER_ANSWER != 1:
            return
        self.io.log(f"rx control-burst poll from {self.called} — answering in "
                    "its turnaround")
        # Through the cadence, because everything else it decides still applies
        # here — the give-up budget that closes a peer we cannot reach, an owed
        # release, an owed ask, a rung of the ladder. What the flag says is that
        # this tick was ASKED FOR: the peer's turn keys nothing on our own clock,
        # and a poll is not our own clock  [see idle_keepalive].
        self._answering_poll = True
        try:
            self.idle_keepalive()
        finally:
            self._answering_poll = False

    def _turn_granted(self, samples, said: str | None = None) -> None:
        """The peer answered our turn-request. Take the turn and start sending.

        What the answer *says* is read and logged, never required: an answer named
        by its own waveform passes ``said``, and the control-burst route reads the
        two-tone tail, which carries session state no other route has. What is
        required is that an answer arrived at all — the same positional evidence
        that accepts the connected-ack.
        """
        payload_grant = said is not None
        self.turn = _TURN_OURS
        self._asked = 0
        self._into_our_turn = 0
        self._progressed()
        self._owe_nothing()
        if said is None and len(samples):
            said = VF.control_state(MK.demod_tone_pairs(
                samples, VF.CONNECTED_ACK_NSYM, self._band))
        age = (f", named {self._grant_held:+.3f} s from its last symbol"
               if payload_grant else "")
        self.io.log(f"turn granted (peer answered with {said or 'unread'}{age})")
        if not self._tx_data_over():
            self._tx_turn_idle()

    # ---- burst rendering -------------------------------------------------
    def _tx_went_out(self) -> bool:
        """The transport's verdict on the burst just handed to ``tx()``
        [see :meth:`VaraIO.tx_went_out`].

        Asked rather than required, because the seam is duck-typed by design —
        ``station.mail``'s transports never import this module — and a transport
        without the verdict transmits everything it is handed."""
        verdict = getattr(self.io, "tx_went_out", None)
        return verdict() if verdict is not None else True

    def _burst_tones(self, callsign: str, kind: VF.BurstKind) -> list[int]:
        """``kind``'s carriers for ``callsign`` on this session's own alphabet."""
        return VF.handshake_tones(callsign, VF.for_bw(kind, self.bw))

    def _matches(self, tones, callsign: str, kind: VF.BurstKind) -> bool:
        """True when ``tones`` are ``kind`` keyed to ``callsign``, read on this
        session's own alphabet  [vara_frames, for_bw]."""
        return VF.recognize(tones, callsign, VF.for_bw(kind, self.bw))

    @property
    def _grid(self) -> float:
        """The bin offset every reader on this link is scored on — the peer's own
        measured frequency, or ours until it has been measured."""
        return 0.0 if self._peer_offset is None else self._peer_offset

    def _note_peer_offset(self, resid: np.ndarray, idx: np.ndarray,
                          grid: float) -> None:
        """Take the link's own frequency off a frame this station has just read.

        ``idx`` are the lattice offsets of the symbols that came back confirmed and
        ``resid`` the track they were read from at ``grid``, so each one says how
        far that symbol's carrier stood from the bin it was counted at  [see
        :func:`_top3_track`]. Their median puts the link at ``grid + residual``,
        and the first frame to say so sets the grid outright: the alternative is a
        reader scored half-way between two stations' idea of the frequency, which
        is the one place a two-tone symbol is worse off than at either. After that
        the grid moves :data:`_OFFSET_SLEW` of the way to each new reading — a
        first-order track of the peer's frequency, the way the PACTOR receiver in
        this tree tracks its own.

        WHAT MAY MOVE THE GRID IS A FRAME ALREADY IDENTIFIED, never audio a search
        liked the look of. Every caller here has a burst keyed to a callsign in
        hand and :data:`_OFFSET_MIN_TONES` of its symbols confirmed on that
        callsign's own tones, which is the same evidence ``_peer_shift`` is taken
        on and for the same reason: a frequency this file went looking for would
        have nothing to keep it honest, and the readers downstream would then be
        scored on a grid the band chose.

        It is single-tone frames that pay for this and two-tone frames that spend
        it. A half-bin error costs the first nothing, so the reads that measure it
        are exactly the reads it cannot damage.
        """
        if len(idx) < _OFFSET_MIN_TONES:
            return
        measured = grid + float(np.median(resid[idx]))
        self._peer_offset = float(np.clip(
            measured if self._peer_offset is None
            else self._peer_offset + _OFFSET_SLEW * (measured - self._peer_offset),
            -0.5, 0.5))
        if abs(self._peer_offset - self._peer_offset_said) < _OFFSET_SAY:
            return
        self._peer_offset_said = self._peer_offset
        # NOT an `rx <frame>` line, and the transcript's own convention is why:
        # those say a burst was taken, and this says where the station is. A
        # reader counting what a session took reads one as the other.
        self.io.log(f"{self.called}'s carriers sit "
                    f"{MK.carrier_to_hz(self._peer_offset):+.1f} Hz off our grid "
                    "— reading it there")

    def _peer_frame(self, x: np.ndarray, kind: VF.BurstKind, callsign: str,
                    track: tuple | None = None):
        """``(heard, at, matched)`` for ``kind`` keyed to ``callsign`` in ``x``.

        The one way this file reads a peer's payload frame, so that every clean
        read of one also measures where the peer is transmitting
        [see :meth:`_note_peer_offset`].
        """
        if track is None:
            track = _top3_track(x, band=self._band, bin_offset=self._grid)
        tones = np.asarray(self._burst_tones(callsign, kind), dtype=np.int32)
        heard, at = _payload_fit(x, kind, tones, track, self._band)
        matched = self._matches([int(t) for t in heard], callsign, kind)
        if matched:
            got = np.flatnonzero(heard == tones)
            self._note_peer_offset(track[2], np.rint(
                (at + got * MK.HOP + MK._WOFF) / _ACK_GRID).astype(np.int64),
                self._grid)
        return heard, at, matched

    def _send_burst(self, kind: VF.BurstKind) -> bool:
        """Key up, synth the MFSK burst, transmit, key down. True when the burst
        reached the transmitter.

        Which callsign keys the burst is the kind's own property and never the
        caller's to choose: every frame in the family is keyed to the CALLED
        station except the two turn frames, which name the station transmitting
        them  [spec 05 §5.3.3]. Which alphabet it goes out on is the session's
        [vara_frames, for_bw], and this is the one place every generated burst
        passes through."""
        callsign = self.caller if kind.keyed_by == "caller" else self.called
        kind = VF.for_bw(kind, self.bw)
        # `finally`, always: an exception between key-up and key-down leaves the
        # transmitter keyed. On a shared band, with an unattended run, that is the
        # one failure here with consequences outside this process.
        self._key(True)
        try:
            self.io.tx(MK.synth_burst(callsign, kind))
            sent = self._tx_went_out()
        finally:
            self._key(False)
        # The tx line must say what happened: logging `tx ... keyed-by=` after a
        # tx() the transport declined puts a transmission in the transcript and
        # silence on the band, and `keyed-by=` is what onair_session reads as
        # proof the attempt reached the air.
        self.io.log(f"tx {kind.name} keyed-by={callsign}" if sent
                    else f"{kind.name} was not transmitted")
        return sent

    def _send_token(self, name: str) -> bool:
        """Key one of BW500's DBPSK control tokens  [spec 02 §2.6]. True when it
        reached the transmitter. BW500 is the only bandwidth with such a
        vocabulary — see :mod:`vara_control`."""
        self._key(True)
        try:
            self.io.tx(VC.synth_token(name))
            sent = self._tx_went_out()
        finally:
            self._key(False)
        if not sent:
            self.io.log(f"{name} token was not transmitted")
        return sent

    def _tx_session_confirm(self) -> None:
        """Step 6: the initiator's post-CONNECTED confirm burst, keyed to the CALLED
        callsign  [spec 05 §5.3.3]. A real VARA expects this after the connected-ack;
        without it the peer sees a silent link."""
        self._send_burst(VF.SESSION_CONFIRM)

    def _tx_connected_ack(self) -> bool:
        """Step 5: the responder's connected-ack — 11 two-tone symbols, no callsign
        [spec 04 §4.2C]. True when it reached the transmitter.

        Its four preamble symbols are fixed; the seven behind them carry session
        state whose encoding is not reversed, so what goes out is one captured
        BW2300 frame, as for the per-over response. A peer that reads more than
        the preamble will see stale state.
        """
        sent = self._tx_control_burst()
        self.io.log("tx connected-ack (captured BW2300 frame)" if sent
                    else "connected-ack was not transmitted")
        return sent

    def _tx_control_burst(self) -> bool:
        """The 11-symbol two-tone control burst for our end of the link. True when
        it reached the air.

        A link's two bursts are keyed to the station that CALLED it AND to the
        bandwidth it came up at, and role picks ours out of the two
        [vara_frames, control_bursts]. The difference is read: the responder's
        keyed from this end leaves a real VARA idling to its own timeout, and the
        caller's frees it in the turnaround.

        Only a link W9SSJ called is measured, at either bandwidth. Answering
        somebody else's call keys another link's tail, which is worth saying out
        loud rather than keying silently — and worth keying anyway, because a peer
        answered with nothing at all has never been measured and one answered with
        the wrong tail has. The fallback stays on this session's own bandwidth:
        at BW500 the wide tails put most of their carriers outside a 500 Hz
        receiver, so keying them is not a stale state but silence.
        """
        pair = VF.control_bursts(self.caller, self.bw)
        if pair is None:
            self.io.log(f"no control burst is measured for a link {self.caller} "
                        f"called at BW{self.bw} — keying the one pair held for "
                        f"this bandwidth, which is another link's tail")
            pair = VF.control_bursts(VF.CONTROL_BURST_CALLER, self.bw)
        burst = pair[1] if self.role == "responder" else pair[0]
        self._key(True)
        try:
            self.io.tx(MK.synth_tone_pairs(burst))
            return self._tx_went_out()
        finally:
            self._key(False)

    def _tx_over_continue(self) -> bool:
        """The captured 8-symbol answer to an over with another one behind it
        [vara_frames, over_continue]. True when it reached the air.

        A shorter two-tone burst than the one below, and either it or the
        32-symbol `session-over-response` is the whole difference between reading
        a delivery and taking its first over: answered with the 11-symbol burst at
        its FIRST over a stock responder released the turn with 267 of a 356-byte
        greeting still queued, which is the shape seven gateways returned,
        byte-count and all.

        Its six state symbols are a captured copy, as the 11-symbol burst's seven
        are, so a peer that reads past the lead sees stale state — and the link
        they were copied off is one W9SSJ called, not the link they are keyed
        into. That is why this is behind `OVER_CONTINUE_CAPTURED` at BW2300
        rather than the default for a session's first delivery: at KB5LZK on
        2026-08-30 it drew sixteen `session-responder-idle` and no over 2. After
        a handover it answers whatever is set  [see _tx_over_response].

        It has a copy per bandwidth because only the common lead pair survives
        an alphabet change. BW500 retains this as its initial default: a sender
        re-keys 0.52-0.56 s after its unkey. BW2750 uses the generated 16-symbol
        initial answer after the exact KC9GHZ greeting exposed retries on this
        captured tail. Explicit selection and recovery still retain the captured
        variant at every measured bandwidth [see over_continue_default].

        AT BW500 THE SEVEN ARE A REPORT AND THE STORED COPY IS THE WRONG LINK'S,
        which is why the table comes first  [vara_frames, over_continue_state]:
        they move with the callsign pair, the role and the state of the over just
        answered, and stock at KC9GHZ ignored the W1AW copy at every over of the
        chain run. Where the link and that state are on tape the measured tail
        goes out. An initiator without that link/state measurement declines
        the captured form; `_tx_continue_answer` selects generated short16,
        which stock K5FIT accepts on its first keying. Recovery rungs use the
        same selection, so none keys another link's report. Other bandwidths
        and the responder role retain their existing selection.
        """
        state = self._peer_over_state
        burst = (VF.over_continue_state(self.caller, self.called, *state)
                 if state and self.bw == "500" and self.role != "responder"
                 else None)
        if burst is not None:
            self.io.log(f"continue tail measured for {self.caller}"
                        f"→{self.called} after a level {state[0]} over "
                        f"carrying {state[1]:#04x}")
        else:
            if self.bw == "500" and self.role == "initiator":
                self.io.log(f"no continue tail is measured for {self.caller}"
                            f"→{self.called} in this state — a generated answer is owed")
                return False
            burst = VF.over_continue(self.caller, self.bw) or VF.over_continue(
                VF.CONTROL_BURST_CALLER, self.bw)
            if burst is None:
                return False
            if self.bw == "500":
                self.io.log(f"no continue tail is measured for {self.caller}"
                            f"→{self.called} in this state — keying the stored "
                            "copy, which is another link's report")
        self._key(True)
        try:
            self.io.tx(MK.synth_tone_pairs(burst))
            return self._tx_went_out()
        finally:
            self._key(False)

    def _tx_over_response(self, last: bool) -> None:
        """Answer one identified DATA over and leave the peer transmitting
        [spec 05 §5.3.3].

        ``last`` picks the burst, and the over itself is what says which
        [see :func:`arq.phy.over_is_last`]: a full body has another over behind it
        and draws a continue-class frame, a short body closes the delivery and
        draws the 11-symbol control burst below.

        Three continuation forms remain selectable: generated 32-symbol
        (1.366 s), generated 16-symbol (0.683 s), and captured 8-symbol (0.341 s).
        ``over_continue`` governs the initial delivery and
        ``over_continue_after`` the deliveries after a handover. Defaults are
        generated16 at BW2300/BW2750 and captured8 at BW500;
        post-handover defaults remain generated16. The BW2750 choice avoids the
        repeated-idle recovery seen with the captured tail on the exact KC9GHZ
        greeting; its original W1AW bench success did not establish that tail's
        portability between sessions. The final ACK below is unchanged by this
        choice [see OVER_CONTINUE_ANSWERS, over_continue_default].

        The control burst's own arbiter is two stock VARA HF 4.9.0 instances
        passing the turn to each other over a fake cable on 2026-08-26, one cable
        per direction so every burst is attributed by which recording holds it. Across three sessions each station answered
        every DATA over with this burst and with nothing else — thirteen
        occurrences, 0.470 s, 0.16 s into the turnaround, preamble 4 of 4 at every
        one — and the peer's release followed 0.13-0.14 s later. Not one
        32-symbol frame appears on either cable in any of the three, so
        `session-over-response` is not what a real caller keys here.

        Every over in those three was a LAST over: no delivery in them ran past
        one, which is why the burst above did not appear until a bench moved 600
        bytes. "Every DATA over" was true of what was on those tapes and not of
        the protocol, and reading it as the protocol is what took one over out of
        every gateway greeting this station has ever been sent.

        What that cost, measured on the same bench: answered with the 32-symbol
        frame a responder repeats its over four times and then falls into its own
        idle cadence; answered with this one it hands the turn straight back.

        The seven state symbols behind the fixed preamble are the captured
        BW2300 frame's, as at step 5. At BW2300 they are keyed to the station:
        each of the two real VARAs answered every over of the three sessions with
        its own unchanging tail, so nothing here is read out of them and a peer
        that reads more than the preamble will see stale state.

        Whatever the last over still owed is settled here: the peer sending on is
        the answer to it having landed, since a sender that was not acknowledged
        repeats instead  [see :meth:`_reack`]. What this one owes is set behind
        the burst that goes out.
        """
        self._owe_nothing()
        if last:
            if self._tx_control_burst():
                self.io.log("tx per-over response — 11-symbol control burst "
                            "(the over is short, so the delivery ends here)")
            return
        answer = self.over_continue_after if self._handed_over else self.over_continue
        # Accepting DATA creates the ACK obligation even when the transport
        # cannot key now. A fresh peer gap must retry that ACK, not a keepalive.
        self._answer_owed = _OWED_OVER
        self._reack_frame = answer
        why = ("(the over is full, so another follows it"
               + (", and the peer is sending again after a handover)"
                  if self._handed_over else ")"))
        went = self._tx_continue_answer(answer)
        if went is None:
            self.io.log("per-over ACK transmission declined — acknowledgement still owed")
            return
        self._reack_frame = went
        self.io.log(f"tx per-over response — {OVER_CONTINUE_NAMED[went]} {why}")

    def _tx_continue_answer(self, answer: str) -> str | None:
        """Key one continue-class answer to an intermediate over, and return which
        of them reached the air  [see OVER_CONTINUE_ANSWERS]. None when none did.

        The generated frame is the fallback rather than a fourth setting: a link
        with no captured copy of its own still owes the peer an answer, and that
        one is built from the callsign.
        """
        # BW500's captured tail reports link and rate state. A foreign copy
        # was ignored twice per greeting over at K5FIT; generated short16 is
        # the bounded fallback, including when the recovery ladder selects
        # captured again. Keep the measured link/state table where available.
        if (answer == OVER_CONTINUE_CAPTURED and self.bw == "500"
                and self.role == "initiator"):
            state = self._peer_over_state
            if (state is None or VF.over_continue_state(
                    self.caller, self.called, *state) is None):
                self.io.log(f"no continue tail is measured for {self.caller}"
                            f"→{self.called} in this state — using the generated short answer")
                answer = OVER_CONTINUE_SHORT
        if answer == OVER_CONTINUE_CAPTURED and self._tx_over_continue():
            return OVER_CONTINUE_CAPTURED
        if (answer == OVER_CONTINUE_SHORT
                and self._send_burst(VF.SESSION_OVER_RESPONSE_SHORT)):
            return OVER_CONTINUE_SHORT
        if self._send_burst(VF.SESSION_OVER_RESPONSE):
            return OVER_CONTINUE_GENERATED
        return None

    def _reack(self) -> bool:
        """Say again what the peer did not hear us say. True when a burst reached
        the air.

        AN ACKNOWLEDGEMENT IS ONE BURST INTO ONE TURNAROUND AND NOTHING REPEATED
        IT. On 2026-09-08 KE8LVA proposed four messages, sent seven full overs of
        the first, and did not take our answer to the seventh — keyed at the same
        +0.07 s as the eleven it did take, into a channel with a narrowband
        occupant sitting where that frame's tones fall. The gateway then ran its
        own idle cadence for 57 s with the rest of the mailbox in hand and closed;
        the field its overs carry says four or more were still to come. Nothing in
        this file could reach that state: the cadence fell to keepalive-a/b, which
        a stock responder answers 0 of 6, and re-ack, NAK and ask were all
        unreachable from it.

        SO THE PEER'S OWN CADENCE IS THE CLOCK. Each idle burst it keys opens a
        1.7-1.9 s listening gap behind it — measured over 35 asks at a stock
        responder, where every grant came from a burst that ended inside one — and
        a rung goes out there rather than on our ~10 s clock, which lands in the
        middle of the peer's next transmission about as often as not
        [see :meth:`_stream_answer`, :meth:`idle_keepalive`].

        WHICH RUNG depends on what is owed. An intermediate over draws the same
        continue-class frame again, then the other two in turn — the three differ
        in length by a factor of four, and a peer that read none of one may read
        another — and last the NAK, which a stock sender answers by re-sending the
        over rather than by waiting out its own timer  [see :meth:`_tx_nak`]. The
        repeat that answer draws is absorbed by the duplicate gate
        [see :meth:`_deliver`]. An owed release draws the control burst again and
        then the turn-request, which takes the channel rather than waiting for it.

        One burst per call, and the budget is spent transmissions: a peer that
        cannot hear us must not be keyed at forever, and `_REACK_MAX` sits inside
        the give-up budget so the ladder finishes before the link closes.
        """
        if self._answer_owed is None or self._held_answer is not None:
            return False
        keyed = (self._reack_release() if self._answer_owed == _OWED_RELEASE
                 else self._reack_over())
        if keyed:
            self._keyed_on_peer_burst = True
            self._since_progress += 1
            self.idle_keyed += 1
        return keyed

    def _has_over_nak_frame(self) -> bool:
        """The called-keyed ``SESSION_OVER_NAK`` is generated for this link.

        The CALLSIGN generates the frame, and the bandwidth does not: stock W9SSJ
        keys it behind the responder's 683 idle at both wide bandwidths — all 32
        symbols on the 2026-09-16 two-stock BW2750 tape, and 31 of 31 across 20
        keyings at BW2300  [docs/protocols/vara/20-peer-turn-measurements.md].
        Says nothing about the responder role, where no such frame is on tape.
        """
        return self.role == "initiator" and self.caller.upper() == "W9SSJ"

    def _has_idle_over_nak(self) -> bool:
        """Measured wideband caller RECOVERY, distinct from the short NAK and from
        the idle answer above.

        Stock W9SSJ at BW2750 requests unread DATA behind the responder's idle
        using the called-keyed SESSION_OVER_NAK, and the responder returns the
        missing bytes at levels 2,3,4
        [see _reack_over, _stream_recovery_cue].

        Only that bandwidth has the retransmission ladder on tape, so only it
        drives the recovery.

        Widening this to BW2300 on 2026-09-18 was wrong and is recorded here so it
        is not tried again from the same reasoning. The unread-window recovery and
        the caller's own NAK are different mechanisms: at BW2300 an over that will
        not decode is answered with the measured `NAK_CALLER_2300`
        [test_vara_nak.py::test_the_caller_keys_its_own_nak_when_an_over_will_not_decode],
        and routing that case through this predicate keyed nothing at all. Eight
        tests across four files caught it. Whether BW2300 should ALSO have an
        idle-anchored ask is open, but it is a second frame in a second situation,
        not this predicate.
        """
        return self._has_over_nak_frame() and self.bw == "2750"

    def _answer_peer_idle(self) -> bool:
        """The one burst a stock caller keys back at the peer's own idle cadence,
        with the turn the peer's and nothing owed. True when it went out.

        Measured 2026-09-16 against stock VARA HF 4.9.0, 45 of 45 idles answered at
        both bandwidths: `session-responder-idle` (683) draws one
        `session-over-nak` and `session-responder-over-idle` (745) one
        `session-keepalive-a`, each 0.11-0.14 s behind the idle's tail, one per
        idle, on the peer's cadence rather than ours
        [docs/protocols/vara/20-peer-turn-measurements.md].

        NEITHER IS CHARGED TO THE GIVE-UP BUDGET. The budget counts the peer's
        silence, and a frame this station read and answered is the opposite of
        silence; charging it closed a live link in twenty seconds at the peer's own
        3.5 s idle rate, over a log holding the six bursts it had just read
        [see idle_keepalive, ``_MAX_WITHOUT_PROGRESS``]. Nor is either progress:
        the frame says the peer is transmitting and nothing more, which is what a
        stalled session looks like too  [see _peer_responder_idle].

        AND NEITHER MAY DEFER THE TICK THAT SPENDS THAT BUDGET. ``idle_keyed`` is
        not a statistic: :func:`kestrel_connect.mail_session` restarts its
        keepalive deadline whenever ``(progress, idle_keyed)`` moves, so counting
        an answer there pushes the give-up tick back by a whole cadence. The
        answers go out at the PEER's rate, which is faster than ours — modelled
        against a 745 every 3.5 s, no tick ever fired, `_since_progress` stayed at
        zero and the link held for 115 transmissions in 400 s, which is the
        2026-08-16 failure `_MAX_WITHOUT_PROGRESS` exists to stop. So this path
        keeps its own count and leaves the driver's clock alone.
        """
        if self._idle_kind is VF.SESSION_RESPONDER_OVER_IDLE:
            kind = VF.SESSION_KEEPALIVE_A
        elif self._has_over_nak_frame():
            kind = VF.SESSION_OVER_NAK
        else:
            return False
        if not self._send_burst(kind):
            return False
        # One rung per turnaround: the idle tick must not key a second burst into
        # the gap this one already answered  [see _reack, idle_keepalive].
        self._keyed_on_peer_burst = True
        self.idle_answers += 1
        return True

    def _reack_over(self) -> bool:
        """The ladder for an intermediate over: the frame that answered it, the
        other two continue-class frames, then the NAK  [see :meth:`_reack`].

        An unread window repeats the NAK alone, and an over owed the measured
        idle recovery repeats `session-over-nak`, the frame a stock caller keys there
        [see :meth:`_stream_recovery_cue`]."""
        if self._owed_block:
            # An incomplete window owes DATA, not an ACK. Retain this debt after
            # every request: a lost NAK is retried in the next peer-idle gap.
            # Control traffic alone cannot supply the missing bytes.
            if self._has_idle_over_nak():
                self._owed_recovery = True
            generated = self._has_idle_over_nak() or (
                self.bw == "500" and self._owed_recovery)
            if (self.bw != "500" and not generated
                    and VF.nak(self.caller, self.bw) is None):
                self.io.log(f"cannot recover the unread window: no NAK is measured "
                            f"for a link {self.caller} called at BW{self.bw} — "
                            "closing the link with incomplete receive data")
                self.disconnect()
                return False
            if self._reacks >= _REACK_MAX:
                self.io.log(f"unread window still missing after {self._reacks} "
                            "NAK transmissions — closing the link")
                self.disconnect()
                return False
            keyed = (self._send_burst(VF.SESSION_OVER_NAK) if generated
                     else self._send_token("nak") if self.bw == "500"
                     else self._tx_nak())
            if not keyed:
                return False
            self._reacks += 1
            self.io.log(f"tx NAK ({self._reacks}/{_REACK_MAX}) — asking "
                        f"{self.called} for "
                        + (f"over #{self._peer_over + 1} again, from its lowest level"
                           if self._owed_recovery and self.bw == "500"
                           else "the unread window again"))
            return True
        rungs = [self._reack_frame] + [a for a in OVER_CONTINUE_ANSWERS
                                       if a != self._reack_frame]
        if self._reacks < len(rungs):
            went = self._tx_continue_answer(rungs[self._reacks])
            if went is None:
                return False
            self._reacks += 1
            self.io.log(f"re-acknowledging over #{self._peer_over} "
                        f"({self._reacks}/{_REACK_MAX}) in {self.called}'s "
                        f"turnaround — {OVER_CONTINUE_NAMED[went]}")
            return True
        if self._reacks < _REACK_MAX and self._tx_nak():
            self._reacks += 1
            self.io.log(f"over #{self._peer_over} re-acknowledged "
                        f"{self._reacks - 1}x unanswered — keying the NAK, which "
                        f"asks {self.called} to send the over again")
            return True
        return self._reack_spent()

    def _reack_release(self) -> bool:
        """The ladder for the release a delivery's last over owes us: the control
        burst again, then the turn-request  [see :meth:`_reack`].

        The peer's release follows our acknowledgement of its last over
        [vara_frames, SESSION_TURN_RELEASE_RESPONDER], so the first rung is that
        acknowledgement again and the rest is asking. Bounded by the ask budget as
        well as by the ladder's: the turn-request is the same burst either way and
        it is charged once  [see _TURN_MAX_ASK].
        """
        empty = not (self._txq or self._tx_pending is not None)
        if not self._reacks or (empty and self._reacks < _REACK_MAX):
            if not self._tx_control_burst():
                return False
            self._reacks += 1
            self.io.log(f"no release in {self.called}'s cadence — re-keying the "
                        f"11-symbol control burst ({self._reacks}/{_REACK_MAX})")
            return True
        if empty:
            return self._reack_spent()
        if (self._reacks < _REACK_MAX and self._asked < _TURN_MAX_ASK
                and self._ask_for_turn()):
            self._reacks += 1
            self.io.log(f"no release in {self.called}'s cadence — asking for the "
                        f"turn ({self._asked}/{_TURN_MAX_ASK})")
            return True
        return self._reack_spent()

    def _reack_spent(self) -> bool:
        """The ladder is out of rungs: say so, stop owing, and put the queue where
        the ordinary cadence can still reach it."""
        owed, spent = self._answer_owed, self._reacks
        self._owe_nothing()
        if owed == _OWED_RELEASE:
            self.io.log(f"{self.called} never released the turn, {spent} attempt(s) "
                        f"spent — {len(self._txq)} block(s) still queued")
            return False
        left = ("" if self._overs_hint is None else
                f" (the peer had said {_delivery_stage(self._overs_hint)})")
        self.io.log(f"over #{self._peer_over} unacknowledged after {spent} "
                    f"attempt(s){left} — {len(self._txq)} block(s) queued")
        if self._txq or self._tx_pending is not None:
            return self._ask_for_turn()
        return False

    def idle_keepalive(self) -> None:
        """Key the idle-cadence burst for the turn we are in  [spec 05 §5.3.3].
        Call on the ~10-12 s idle cadence of spec 05 §5.5.

        With an unacknowledged over, solicit its state within a bounded budget. An
        idle response is evidence the peer is listening, not an acknowledgment
        of the data. Only after the outstanding frame is acknowledged may an
        empty queue release the turn.

        Holding the turn WITH SOMETHING STILL QUEUED but no outstanding over,
        that is the turn-idle frame
        — 12 of them, 13.2 s apart and unchanging, across the two off-air
        sessions — which prompts the peer for the answer that draws the next over.
        Holding it drained and acknowledged, the turn goes back on the air
        [see _release_turn]. A refused release is retried on this cadence too.

        With an answer owed for the peer's last over and no rung keyed into a gap
        of the peer's since the last tick, that is the next rung of the
        re-acknowledgement ladder  [see :meth:`_reack`].

        Otherwise, in the peer's turn with nothing owed, it keys NOTHING. A stock
        caller left alone there keys nothing for 60 s and then disconnects on a
        silence timer  [bench 2026-09-16]; the one prompted burst is the answer to
        the peer's control-burst poll, which is a keepalive-A  [see _took_poll].
        Keepalive-B is never keyed at all. The session read as keepalive-A at
        10.2 s and keepalive-B at 12.3, 24.4 and 36.5 s is the same law seen from
        the other side: that ~12 s spacing is the responder's own poll cadence of
        12.07 s, so those are four answers rather than a cadence of the caller's.

        THE 11-SYMBOL POLL BELONGS HERE AND IT DOES NOT WORK. Two stock 4.9.0s
        cross-wired on 2026-08-30 say a caller with an empty queue keys the
        11-symbol control burst every 12.07 s and its responder answers every one,
        while our keepalives drew no key-up at all across 59 s. Flown on the bench
        against a stock 4.9.0 on 2026-09-02 it answers the poll and **does not key
        the over it is
        holding**: the responder's queued reply reached our host in 4 of 4 sessions
        on this cadence and in 1 of 5 on the poll, over the same code and the same
        cable. Keying a `session-turn-release` by hand at the same instant does not
        draw it either (0 of 1). So what the poll is missing is not the frame, and
        the frame that draws a held over is still open.

        With a request outstanding the cadence carries the request again. No
        recording holds a gateway answering one, so a peer that never does is a
        live possibility rather than a hypothetical: after ``_TURN_MAX_ASK`` the
        queue is left where it is and the turn goes back to the peer, which keeps
        the link up and puts the failure in the log instead of in a silence.

        AND KEEPING THE LINK UP IS NOT FREE. Every branch here keys, so the cadence
        that holds a live link open is also what a station talking to nobody spends
        the channel on: giving the turn back drops into the keepalive branch, which
        had no counter, no deadline and no way out, and on 2026-08-16 it keyed
        twenty-three times into a 40 m channel the operator could hear was empty
        [see ``_MAX_WITHOUT_PROGRESS``].
        ``_MAX_WITHOUT_PROGRESS`` bounds that, and closes the link when it is spent
        rather than leaving the decision to whoever is listening.
        """
        if self.state is not VaraState.CONNECTED or self.role != "initiator":
            return
        if self._held_answer is not None:
            # Neither a keepalive nor a retransmission request belongs inside
            # the receive window whose acknowledgement is still deferred.
            return
        idle_before = self.idle_keyed
        if self._since_progress >= _MAX_WITHOUT_PROGRESS:
            # The counter spans the run of bursts since the session last moved,
            # never the session, and "nothing this build can read back" read as a
            # verdict on both: it closed a KB3AC-10 link on 2026-08-30 over a log
            # holding fifteen frames named from that gateway. What is measured is
            # that no turnaround in THIS run read as an answer — not that none
            # carried one, and not that none ever did — so the session's own
            # count goes out beside it.
            self.io.log(f"{self._since_progress} idle checks without protocol "
                        f"progress — closing the stalled link to {self.called}. "
                        "Recognized idle traffic does not advance the exchange; "
                        f"{self.progress} progress event(s) in this session, "
                        f"{len(self._txq)} block(s) still queued")
            self.disconnect()
            return
        if self._release_owed:
            keyed = self._release_turn()
            if keyed and self._txq:
                self._ask_owed = True
        elif self.turn == _TURN_OURS:
            if self._tx_pending is not None:
                before = self._tx_retries
                self._retry_data_over(final_query=True)
                keyed = self._tx_retries > before
            elif self._txq:
                keyed = self._tx_turn_idle()
            else:
                keyed = self._release_turn()
        elif self.turn == _TURN_ASKED:
            if self._asked < _TURN_MAX_ASK:
                self.io.log(f"turn-request {self._asked}x unanswered — "
                            "asking again")
                keyed = self._tx_turn_request()
            else:
                self.io.log(f"turn-request unanswered {self._asked}x — giving the "
                            f"turn back, {len(self._txq)} block(s) still queued")
                self.turn = _TURN_PEER
                self._asked = 0
                self._ask_owed = False
                keyed = False
        elif self._ask_owed:
            # The backstop behind the clear-channel poll  [see _ask_for_turn]: a
            # channel that never reads clear still owes the ask, and this is where
            # it goes. It is a single owed ask and not a standing rule: asking
            # whenever the queue is non-empty and the turn is the peer's turns the
            # give-back below into a loop — three asks, the turn handed back, three
            # more — and the budget
            # that closes a dead link never comes round.
            self._ask_owed = False
            keyed = self._tx_turn_request()
        elif self._answer_owed is not None:
            # The backstop behind the peer's own turnaround  [see _reack]: a peer
            # whose cadence this build cannot name still owes an answer, and a
            # keepalive is not what asks for one. Only when no rung went out in a
            # gap of the peer's since the last tick — a ladder keyed twice per
            # turnaround is two stations transmitting at once.
            keyed = False if self._keyed_on_peer_burst else self._reack()
        else:
            # In the peer's turn a stock caller keys NOTHING on its own clock: with
            # no peer frames it stays silent for ~60 s and then disconnects on a
            # silence timer, not on a cadence of its own [bench 2026-09-16]. The
            # peer's own idles draw their single answer in the gap they open
            # [see _answer_peer_idle], and an owed over draws the ladder above.
            #
            # UNLESS THE PEER ASKED. Its control-burst poll is a prompt, not our
            # clock: a stock responder handed the channel with a reply queued keys
            # that burst every 2.3 s and releases only once one of ours lands in
            # the gap behind one, 5 of 5 on 2026-09-03  [see _took_poll]. That
            # answer is charged, and `_POLLS_PER_ANSWER` holds it to one per
            # cadence, so a responder that cannot hear us still closes the link.
            if self._answering_poll:
                keyed = self._send_burst(VF.SESSION_KEEPALIVE_A)
            else:
                # WHAT THIS TICK SPENDS IS THE PEER'S SILENCE, and that is the
                # peer's whatever this station's transmitter is doing — so it is
                # charged unconditionally. `tx_went_out` cannot be asked here:
                # `_refused` is decided afresh at each key-up and is stale outside
                # a keyed region  [see VaraIO.tx_went_out], so reading it latched a
                # dead rig's refusal and left the budget at zero on a link nothing
                # was answering. The transmissions-not-intentions rule still
                # governs every branch that keys; this one does not key.
                keyed = False
                self._since_progress += 1
        # A budget spends transmissions, not intentions [see originate]: a burst the
        # transport declined put nothing on the air for the peer to answer, so the
        # silence it left is not the peer's.
        if keyed and self.idle_keyed == idle_before:
            self._since_progress += 1
            self.idle_keyed += 1
        self._keyed_on_peer_burst = False

    def _tx_link_setup(self) -> bool:
        """Step 4: TX the link-setup over carrying the caller (MYCALL)  [spec 05
        §5.3 step 4]. True when it went out (or was skipped by policy).

        The caller-ID frame rides the session bandwidth's ordinary wideband burst
        (spec 04 §4.2A), synthesised byte-exact by ``varahf2300_tx`` or
        ``varahf500_tx``. A real VARA station requires this over to complete the
        connect, and on 2026-08-15 a real VARA HF 4.9.0 completed to CONNECTED on
        the BW500 one.
        """
        if self.mfsk_only:
            self.io.log("link-setup SKIPPED (mfsk_only) — real VARA will not complete")
            return True
        if self.bw not in OF.LINK_SETUP_BW:
            # Keying another bandwidth's frame instead would put its whole
            # occupied width into this channel: the peer cannot decode it either
            # way, and only one of the two splatters across its neighbours.
            self.io.log(f"link-setup SKIPPED at BW{self.bw} — no link-setup "
                        "waveform for this bandwidth; the connect cannot complete")
            return True
        self._key(True)
        try:
            self.io.tx(OF.link_setup_tx(self.caller, bw=self.bw, level=self._setup_level))
            sent = self._tx_went_out()
        finally:
            self._key(False)
        self.io.log(f"tx link-setup caller={self.caller} level={self._setup_level}" if sent
                    else "link-setup was not transmitted")
        return sent

    def _rx_link_setup(self, samples) -> bool:
        """Step 4 (responder): decode the initiator's link-setup over to learn the
        caller, then complete the connect  [spec 05 §5.3.2, §5.3 step 5].

        :func:`vara_ofdm.link_setup_rx` reads each bandwidth's own base-level
        burst: rec3 at BW2300, its 20-bin twin at BW2750, level 4 at BW500. A
        bandwidth with no decoder leaves the responder CONNECTING rather than
        acking a caller it could not identify."""
        if self.bw not in OF.LINK_SETUP_BW:
            self.io.log(f"rx wideband burst at BW{self.bw}: no link-setup decoder "
                        "for this bandwidth — waiting")
            return False
        caller = OF.link_setup_rx(np.asarray(samples, dtype=np.float64), bw=self.bw)
        if caller is None:
            self.io.log("rx wideband burst but no link-setup decoded — waiting")
            return False
        self.caller = caller
        self.io.log(f"rx link-setup — caller is {caller}")
        # The link is up when the ack is on the air, and not before. An ack the
        # transport declined leaves the initiator re-sending its link-setup, and
        # this is the state that answers those repeats — from CONNECTED a repeat
        # arrives as a data-phase burst and is refused for being a link-setup, so
        # a connect both ends want cannot recover from either.
        if not self._tx_connected_ack():             # step 5
            return False
        self.state = VaraState.CONNECTED
        self.io.connected(self.caller, self.called, self.bw)
        return True

    # ---- host-driven entry points ---------------------------------------
    def originate(self, called: str, caller: str | None = None,
                  retry: bool = False) -> bool:
        """Host ``CONNECT``: begin an outbound session to ``called`` (gateway).
        True when the connect-request reached the transmitter.

        ``retry`` is a re-key of a request already made, which under
        ``cr_retry_form="stock"`` goes out in the shorter form a real caller uses
        [VF.connect_request_retry].

        The caller drives the CR train, so the caller owns its budget — and a
        budget spends transmissions, not intentions: a request the transport
        declined put nothing where the gateway could answer it and must not cost
        one of the attempts. What it does cost is the interval, exactly as a sent
        one does; a rig refusing every key-up is re-keyed on the cadence rather
        than as fast as the loop comes round.

        The step still advances, because the state that says *waiting for a
        connect-response* is what the caller's resend fires on: a refusal that
        left it behind would end the attempt on its first silent burst.

        AND NOTHING FROM A PREVIOUS CONTACT COMES WITH US. Everything below
        describes a link — whose turn it is, what we last delivered, which half of
        the keepalive alternation is owed, where in the over stream we are — and a
        call being placed has no link to describe. Left standing, the last body
        delivered to the host suppresses an identical first over from the next
        session, which is exactly what a Winlink gateway sends when you call it
        twice: the same SID greeting, dropped as a repeat, and the exchange dies
        before it starts. The turn is the same defect with the channel at stake —
        a session inherits `ours`, keys turn-idle at a gateway that is mid-greeting
        and answers its overs as intrusions.
        [The queue is NOT one of these: bytes the host wrote before the link came
        up are the caller's, and `_connected` transmits them. Neither is
        `_cr_restarts`, which is this attempt's own ladder.]
        """
        self.role = "initiator"
        self.called = called.upper()
        self.caller = (caller or (self.mycalls[0] if self.mycalls else "")).upper()
        self.state = VaraState.CONNECTING
        self._linksetup_tx = 0
        self._setup_level = 4
        self._peer_shift = 0
        self._peer_offset = None
        self._peer_offset_said = 0.0
        self._burst_match_said = False
        self._delivered = ()
        self._keyed_bodies.clear()
        self._over = 0
        self._query_retry_phase = 0
        self._full_keyed = 0
        self._tx_recovery_levels.clear()
        self._tx_pending = None
        self._tx_retries = 0
        self._reset_final_ack_recovery()
        self._pending_answer_at = None
        self._over_keyed_at = None
        self._intermediate_query_attempted = False
        self._intermediate_query_attempts = 0
        self._intermediate_query_for = None
        self._intermediate_query_at = 0.0
        self._intermediate_query_samples = 0
        self._release_owed = False
        self._owe_nothing()
        self._overs_hint = None
        self._peer_over_state = None
        self._peer_over = 0
        self._peer_delivery_open = False
        self._held_answer = None
        self._owed_block = False
        self._window_bodies, self._last_window = [], ()
        self._window_delivered = 0
        self._last_window_complete = True
        self.turn = _TURN_PEER
        self._handed_over = False
        self._asked = 0
        self._into_our_turn = 0
        self._progressed()
        self.answer_retry = False
        kind = (VF.connect_request_retry(self.bw)
                if retry and self.cr_retry_form == "stock"
                else VF.connect_request(self.bw))
        sent = self._send_burst(kind)                    # step 1, keyed to CALLED
        self.step = _I_CR_SENT
        return sent

    def resend_link_setup(self) -> bool:
        """Key step 4 again on a driver's clock. True when it went out.

        The handshake's own resend answers a REPEATED connect-response; a gateway
        that heard nothing repeats nothing, and that silence is what a clock is for.
        Same ``max_link_setups`` budget, charged only by a burst that went out,
        and the ack window reopens with it as it does on that path.
        """
        if self._linksetup_tx >= self.max_link_setups:
            return False
        sent = self._tx_link_setup()
        if sent:
            self._linksetup_tx += 1
            self._reset_stream()
        return sent

    def link_setup_unanswered(self) -> bool:
        """The link-setups drew no connected-ack: go back to the connect-request
        exchange, step 4's budget fresh. False when the ladder is spent
        [see ``_CONNECT_MAX_RESTARTS``] and the attempt is over.

        The CR itself is the driver's cadence to key, which is where requests that
        reach the air are counted.
        """
        if self.role != "initiator" or self.step != _I_LINKSETUP_SENT:
            return False
        if self._cr_restarts >= _CONNECT_MAX_RESTARTS:
            self.io.log(f"no connected-ack for {self._linksetup_tx} link-setup(s) "
                        f"after {self._cr_restarts} restart(s) — the attempt is over")
            return False
        self._cr_restarts += 1
        self.io.log(f"no connected-ack for {self._linksetup_tx} link-setup(s) — back "
                    f"to the connect-request exchange (restart {self._cr_restarts} "
                    f"of {_CONNECT_MAX_RESTARTS})")
        self._linksetup_tx = 0
        self._setup_level = 4
        self.step = _I_CR_SENT
        return True

    def listen(self, on: bool = True) -> None:
        """Host ``LISTEN ON/OFF``: arm/disarm the responder.

        Re-arming is also the way out of a close whose final burst was lost:
        a responder that acked a disconnect-request and heard nothing more is
        DISCONNECTING forever on its own, because nothing it may key ends the
        wait."""
        if on and self.state in (VaraState.DISCONNECTED,
                                 VaraState.DISCONNECTING):
            self.role = None
            self.state = VaraState.LISTENING
        elif not on and self.state == VaraState.LISTENING:
            self.state = VaraState.DISCONNECTED

    def disconnect(self) -> None:
        """Host ``DISCONNECT``: key the close  [spec 05 §5.6].

        **One burst, and nothing answers it.** A stock 4.9.0 keys the 32-symbol
        `session-disconnect-request` once, immediately ahead of its own CW ident,
        and the only thing that follows on the cable is the peer's ident
        [vara_frames, SESSION_DISCONNECT_REQ]. The bench run of 2026-09-02
        confirms it from the other side: after our close its host reports
        DISCONNECTED on its own inactivity timer and its PTT ledger holds no
        key-up at all, through four more closes of ours. So the acknowledgement
        this used to wait for — and the `session-disconnect-final` it answered
        with — is a reading off a loopback tape and not a frame any peer sends,
        and waiting for it ended every session `NOT DISCONNECTED`.

        The close is therefore complete when the burst reaches the air. A burst
        the transport declined puts nothing on the band, so the session stays
        DISCONNECTING and calling this again re-keys it — the driver's retry.
        Initiator-only, like the rest of the session frames here.
        """
        if (self.role != "initiator"
                or self.state not in (VaraState.CONNECTED,
                                      VaraState.DISCONNECTING)):
            return
        self.state = VaraState.DISCONNECTING
        # The burst no longer says it: what goes out is the release, and the log
        # is where a close is legible now.
        self.io.log(f"disconnecting from {self.called}")
        if self._send_burst(VF.SESSION_DISCONNECT_REQ):
            self.state = VaraState.DISCONNECTED
            self.io.log(f"close keyed to {self.called} — session closed")

    # ---- RX ---------------------------------------------------------------
    def _expected_kind(self) -> VF.BurstKind | None:
        """The MFSK handshake burst we are waiting for in the current state.

        The connected-ack (step 5) is deliberately absent: it is not a member of
        this family and on_rx_audio recognises it separately  [spec 04 §4.2C]. A
        gateway that did not hear our link-setup repeats its connect-response,
        which is why that burst stays expected after we have sent one.
        """
        if self.state == VaraState.LISTENING:
            return VF.connect_request(self.bw)
        if self.role == "initiator" and self.step in (_I_CR_SENT, _I_LINKSETUP_SENT):
            return VF.connect_response(self.bw)
        return None

    def _ack_evidence(self, samples) -> bool:
        """True when this audio holds the responder's connected-ack  [spec 04 §4.2C].

        The ack names nobody. It is 11 two-tone symbols whose first four are a
        fixed preamble and whose last seven carry session state, so unlike every
        other burst in the handshake it cannot be checked against the callsign we
        dialled — there is nothing in it that depends on one. What can be checked
        is the preamble: eight carriers, identical in every capture held, at
        BW500, BW2300 and BW2750, from two live gateways and from loopback.

        The evidence that it is *our* ack and not somebody else's is therefore
        positional, not cryptographic: we sent a link-setup naming ourselves, and
        this arrived in the turnaround window that answers it. That is weaker than
        the 15-tone callsign match that confirms step 2, and it is what the
        waveform offers.

        This is the segmented-burst route, kept for transports that only bracket.
        A rejection here says nothing about whether the gateway answered: over a
        real audio path the ack is found by :meth:`on_rx_stream`, which reads audio
        no energy gate brackets.
        """
        wide = _ack_plateau(samples, self._peer_shift, band=self._band,
                            bin_offset=self._grid)
        if wide < _ACK_PLATEAU:
            self.io.log(f"bracketed burst is not the connected-ack: preamble holds "
                        f"for {wide} alignments (need {_ACK_PLATEAU})")
            return False
        return True

    def _response_by_payload(self, samples) -> list[int] | None:
        """The gateway's connect-response found by what it says, or None.

        A gateway that hears no link-setup answers again, and the second answer is
        the one an attempt most needs: the first has already been lost or the
        gateway would not be repeating itself. Off air on 2026-07-26 KB9MMT
        answered twice, and its repeat began 138 ms before our own eighth
        connect-request ended — so all eight of its preamble symbols, and the first
        of its payload, lay under our own transmission and the receiver mute that
        follows it. :func:`vara_mfsk.lock_preamble` has nothing to lock onto and
        returns None, and a burst whose fifteen payload tones are all intact is
        thrown away as silence.

        The preamble is not the only thing about the burst that is known in
        advance. Every payload tone is fixed by the callsign we dialled, so the
        response can be located by the payload instead, scoring every virtual burst
        start including the negative ones. Returns the tones read at the best alignment — the preamble
        positions included, whatever they turned out to be — so the state machine
        judges this burst by exactly the recogniser it judges a preamble-locked one
        by, with no evidence invented on the way.

        Strictly a fallback, and reached only once the lock has already failed, so
        every millisecond it costs is added to the turnaround: 71 ms on a 6 s
        bracket, against 67 ms for the failed lock ahead of it. A preamble that does
        lock is better evidence anyway — eight fixed tones and one alignment, where
        this has fifteen tones and ten thousand alignments to choose the best of.
        """
        for level in (4, 3, 2, 1):
            kind = VF.connect_response(self.bw, level)
            npre = len(kind.preamble)
            tones = np.asarray(self._burst_tones(self.called, kind), dtype=np.int32)
            heard = _payload_alignment(np.asarray(samples, dtype=np.float64), kind,
                                       tones, band=self._band)
            c = int((heard[npre:] >= 0).sum())
            m = int((heard[npre:] == tones[npre:]).sum())
            if _answers(m, c):
                self.io.log(f"rx connect-response for {self.called} located by payload "
                            f"({m}/{c} tones, level {level}) — its preamble did not lock")
                return [int(t) for t in heard]
        return None

    def _accept_connect_response(self, level: int) -> None:
        """Answer a callsign-validated offer at its requested setup speed."""
        if self.role != "initiator" or self.step not in (_I_CR_SENT, _I_LINKSETUP_SENT):
            return
        if self._linksetup_tx >= self.max_link_setups:
            self.io.log(f"rx repeated connect-response — link-setup already sent "
                        f"{self._linksetup_tx}x, not repeating")
            self._reset_stream()
            return
        self._setup_level = level
        self.answer_retry = False
        self.io.log(f"rx connect-response confirmed for {self.called}, level {level}"
                    + (" (repeat — resending link-setup)"
                       if self.step == _I_LINKSETUP_SENT else ""))
        if self._tx_link_setup():
            self._linksetup_tx += 1
        self.step = _I_LINKSETUP_SENT
        self._reset_stream()

    def _peer_answer(self, track: np.ndarray, clear: np.ndarray, live: np.ndarray,
                     n: int, lat: int, resid: np.ndarray | None = None,
                     include_base: bool = False) -> bool:
        """Recognize lower-speed connect offers by the called station's payload.

        The stock caller accepts these responses and sends its caller-ID frame
        at the selected speed. A damaged request can provoke a lower-speed offer;
        that observation never established the previous rejection interpretation.
        """
        if not self.called or self.role != "initiator" or self.step not in (
                _I_CR_SENT, _I_LINKSETUP_SENT):
            return False
        kind = VF.connect_response(self.bw)
        idx = np.arange(n)[:, None] + _PAY_OFF
        heard = track[idx][:, :, 0]
        comparable = live[idx] & (clear[idx][:, :, 0] >= _CLEAR_DB)
        c = comparable.sum(1)
        top = (kind.preadv - 1) // VF.lattice_step(kind)
        positions = _ANSWER_LATTICE_BY_BW.get(self.bw, ()) + ((top,) if include_base else ())
        for position in positions:
            level = 4 + position - top
            if level not in (1, 2, 3, 4):
                continue
            want = np.asarray(VF.payload_bins(self.called,
                              VF.connect_response(self.bw, level)), dtype=np.int32)
            for shift in _RESP_SHIFTS:
                m = ((heard == want + shift) & comparable).sum(1)
                rows = np.flatnonzero(_answers(m, c))
                if not len(rows):
                    continue
                r = int(rows[np.lexsort((c[rows] - m[rows], -m[rows]))[0]])
                at = (lat + r) * _ACK_GRID / MK.FS
                self.answers.append(PeerAnswer(
                    at, position, shift, int(m[r]), int(c[r]),
                    tuple(int(t) - shift if lv else -1
                          for t, lv in zip(heard[r], comparable[r]))))
                self._peer_shift = shift
                if resid is not None:
                    took = comparable[r] & (heard[r] == want + shift)
                    self._note_peer_offset(resid, idx[r][took], 0.0)
                self.io.log(f"rx {self.called} BW{self.bw} connect-response "
                            f"at lattice frame {position}: level {level} "
                            f"({int(m[r])}/{int(c[r])} tones)")
                self._accept_connect_response(level)
                return True
        return False

    def _unattributed_answer(self, heard: np.ndarray, comparable: np.ndarray,
                             kind: VF.BurstKind) -> None:
        """Record a connect-response that answered nobody we can name, if this
        pass holds one. Reports; never advances anything.

        THE TWO ROUTES THAT ACCEPT AN ANSWER ARE BOTH KEYED TO THE CALLSIGN WE
        DIALLED. :meth:`on_rx_stream` and :meth:`_response_by_payload` regenerate
        the fifteen payload tones for ``self.called`` and look for them, so a burst
        that is unmistakably a connect-response and is addressed to somebody else
        reaches neither, and the attempt reports silence. "No connect-response"
        then covers three different things — nobody transmitted, the gateway
        transmitted and we could not read it, and somebody transmitted a
        well-formed answer we cannot attribute — and an operator in a live slot has
        no way to tell them apart. On 2026-08-26 the third happened on both
        KB3AC-10 arms and was found by ear.

        So the burst is read by its own structure instead. Two things about a
        connect-response are fixed for every station on the band:

          * the eight-tone preamble, which takes seven distinct values out of the
            seventy carriers the tone track can return, so no steady tone and no
            drift holds it  [see :data:`_UNATTR_MIN_PRE` for the population];
          * the payload is the output of ONE generator whose only free parameter
            is the callsign's own start state, so :func:`vara_frames.payload_states`
            asks whether ANY station's connect-response carries these tones —
            the whole callsign space at once, rather than the few hundred a panel
            holds.

        That second half is the rejection this route stands on, and it is what
        keeps a sighting from being minted out of noise the way ten symbols of a
        steady tone once minted an ARDOP frame type. A payload symbol draws from
        35 of the alphabet's 70 carriers, so the eight comparable tones
        :data:`_RESP_MIN_HEARD` asks for are one of 35**8 sequences against 2**24
        states: an arbitrary eight land on a state 7e-6 of the time, and a burst
        that reaches a state has 14 or 15 tones behind it in practice.

        Measured with a callsign dialled that is on none of them — so that every
        genuine answer falls through to this route instead of being taken ahead of
        it — over the 16 real off-air gateway recordings and the two KB3AC-10 arms,
        3030 s and 22,595,490 alignment-shifts: it fires three times. NS0A's own
        connect-response, which a station that had dialled NS0A would never see
        here, and the two KB3AC-10 bursts. Over the negative
        corpus — the 31 shared regression fixtures, the verified clear channel
        and the two 2026-08-06 calls nobody answered, 1362 s and 9,967,325
        alignment-shifts — it fires not at all. The one thing in the whole
        population that holds the preamble and not the generator is a burst at
        44.888 s of the 2026-08-19 05:12z call, whose payload reads five to ten
        drifting tones and lands on no state at any alignment; the generator is
        what rejects it and the preamble alone would not.

        The panel search is what turns the reading into a name where there is one:
        a connect-response is keyed to the station being CALLED, so an answer
        addressed to another gateway is that gateway answering somebody else's
        call in our turnaround, and saying which gateway is worth more than saying
        none. A non-match means the station is not on the panel at all — not a
        published gateway, or not published under the callsign it is answering
        with — which is what both KB3AC-10 bursts turned out to be.
        """
        npre = len(kind.preamble)
        pre = np.asarray(kind.preamble, dtype=np.int32)
        hp, cp = heard[:, :npre], comparable[:, :npre]
        npay = comparable[:, npre:].sum(1)
        # A burst reads the same tones over most of a symbol, so the alignments
        # that hold the preamble carry a handful of distinct payload readings
        # between them and the edges of the plateau are where the misreads are.
        # Every distinct reading is offered to the generator, best first, because
        # comparability does not separate an alignment half a symbol early from the
        # one in the middle: both deliver fifteen tones and only one of them is what
        # the peer sent. Measured over the corpus below, the widest set of readings
        # one pass has ever held is eight, and the negative corpus reaches this
        # line at all in none of its 1362 s.
        readings: dict[tuple[int, ...], tuple[int, int, int]] = {}
        for shift in _RESP_SHIFTS:
            ok = (((hp == pre + shift) | ~cp).all(1)
                  & (cp.sum(1) >= _UNATTR_MIN_PRE) & (npay >= _RESP_MIN_HEARD))
            for r in np.flatnonzero(ok):
                payload = tuple(int(t) - shift if lv else -1
                                for t, lv in zip(heard[r, npre:], comparable[r, npre:]))
                rank = (int(cp[r].sum()), int(npay[r]), shift)
                if rank > readings.get(payload, (-1, -1, 0)):
                    readings[payload] = rank
        for payload, (n_pre, n_pay, shift) in sorted(
                readings.items(), key=lambda kv: kv[1], reverse=True):
            states = VF.payload_states(payload, kind)
            if states:
                break
        else:
            return
        # An answer from the station we dialled belongs to :meth:`_peer_answer`,
        # whichever frame of that station's stream it is. This route is for the
        # answer that is somebody else's, and every burst it was built on turned
        # out to be the dialled station's own.
        if VF.payload_position(payload, self.called, kind) is not None:
            return
        # The scan window overlaps its predecessor by one burst's span, so a burst
        # arriving on the seam is offered twice; the state is what tells the two
        # detections apart from two bursts. It also folds a station that answered
        # twice into one record, which is the right trade here — what this reports
        # is that somebody answered, and a repeat says nothing further.
        if any(a.states == states for a in self.unattributed):
            return
        call, m, _n = VF.best_match(payload, self.panel, kind) if self.panel else ("", 0, 0)
        self.unattributed.append(UnattributedAnswer(
            shift, n_pre, payload, states, call, m))
        off = f", {MK.carrier_to_hz(shift):+.0f} Hz off frequency" if shift else ""
        named = (f"the nearest of {len(self.panel)} panel callsigns is {call} at "
                 f"{m} of {kind.n_payload}" if self.panel else
                 "there is no panel here to search")
        self.io.log(
            f"rx a connect-response that names nobody on the receive stream "
            f"({n_pre}/{npre} preamble tones exact, {n_pay}/{kind.n_payload} payload "
            f"tones the receiver delivered{off}) — its payload is this generator's "
            f"own output and {named}. Somebody answered and it is not "
            f"{self.called or 'the station we called'}; nothing is taken from it")

    def _connected(self, confirm: bool = True) -> None:
        """Step 5 accepted: the link is up, and the initiator confirms it
        [spec 05 §5.3 steps 5-6].

        ``confirm`` is False where the link came up on a frame that owes an answer
        of its own: the turnaround holds one burst, and keying the confirm into it
        would have the peer answering that while we key what the frame asked for
        on top of the answer  [see :meth:`on_rx_audio`].
        """
        self._unclassified_continue = None
        self._confirmed_continue_pairs.clear()
        self.state = VaraState.CONNECTED
        self.step = _I_CONNECTED
        self._undecoded = 0
        self._progressed()
        self.io.connected(self.caller, self.called, self.bw)
        if not confirm:
            return
        self._tx_session_confirm()
        if self._txq:                    # host queued data before the link came up
            self._tx_turn_request()

    def _reset_stream(self) -> None:
        self._st_buf = np.zeros(0)
        self._st_track = np.zeros((0, 3), dtype=np.int32)
        self._st_clear = np.zeros((0, 2))
        self._st_resid = np.zeros(0)                # sub-bin, where each tone sat
        self._st_live = np.zeros(0, dtype=bool)     # the receiver was hearing there
        self._st_lat = 0                 # lattice offsets already scanned and dropped
        self._near_miss_at = None        # last below-offer connect-response near-miss

    def on_rx_stream(self, samples) -> None:
        """Search the raw receive stream for the gateway's answers, gate or no gate.

        Feed this every sample the receiver produces, ahead of and independently of
        whatever the burst segmenter brackets. It runs only while an initiator is
        waiting for a connect-response — :meth:`_expected_kind` decides, so there is
        one definition of that window — and drops its state the moment it is not.

        Calling it at all supersedes the bracket route for this one burst:
        :meth:`on_rx_audio` stops accepting a connect-response from a segmented
        burst (see ``_stream_owns_response``), and keeps every other thing it
        recognises. Both routes are handed the same audio, so leaving both live
        means answering one answer twice.

        An energy gate cannot see this burst. Measured on the 2026-07-26 recording
        inside the 400-2700 Hz SSB passband, KB9MMT's reply is 1.6 dB *below* the
        band noise a second after it; 75% of its energy sits outside the passband,
        in the odd-harmonic images of an overdriven receive chain, and it is that
        splatter the segmenter's broadband frame RMS reads. The gate opens on it
        only because our own transmission ends 138 ms earlier and the 2.1 s receiver
        mute that follows pulls the tracked floor 20x below the reply; against a
        floor read from the band noise one second later the margin is 2.18x on an
        enter threshold of 2.0. That is 0.8 dB, none of it earned by the signal, and
        no threshold recovers it — a correctly-configured receiver, which is to say
        one that does not splatter, hands the gate nothing at all.

        So the burst is found by what it says. Every payload tone is fixed by the
        callsign we dialled, so the stream is scored the way :func:`_payload_alignment`
        scores a bracket — the same 32-sample lattice, the same tone track, the same
        :func:`_best_fit`, which searches where the answer sits in time and where the
        peer sits in frequency together — and the winner goes to :meth:`on_rx_tones`,
        the recogniser a preamble-locked burst goes to. There is no second way to
        accept a response.

        Two differences from the bracket search, both narrowing:

          * only alignments whose every symbol is inside the retained track are
            scored. A bracket may be the tail of a burst whose head was cut away, so
            there a symbol outside the audio is *not comparable*; a continuous stream
            has no missing head, so 13 confirmed tones here always means 13 of 15
            rather than 13 of 13 visible. The bracket path keeps that tolerance and
            so keeps covering the case this one cannot — a transport that drops the
            audio under our own transmission.
          * the retained track is dropped on an accept, so the ~44 lattice points
            across one burst's match plateau cannot each advance the handshake.

        False accepts. A stream search examines far more alignments than a
        per-burst one, and :data:`_RESP_SHIFTS` multiplies them again, so its floor
        is measured rather than inherited. Over 2342 s of real off-air HF in 37
        recordings — the on-air calls, the BW2300 gateway sessions, the verified
        clear-channel capture, the 2026-08-14 slot's two negative controls, and the
        31 of the shared regression corpus (PACTOR-1/2/3, ARDOP, FT8, WSPR and band
        noise from four continents), each band-limited to the SSB passband and
        scored against every panel callsign not on it at every shift the search
        uses: 134,094,970 alignment-shifts, of which the best non-response reaches
        5 of 15 — the same 5 the zero-offset population alone reaches. The score
        distribution is geometric: survival falls by 11-27x per extra confirmed tone
        (0.093, 0.055, 0.037, 0.038 measured at 2..5), so extrapolating the eight
        tones from 5 to ``_RESP_MIN_TONES`` puts a false accept far below 1e-15 per
        alignment against the ~450,000 alignment-shifts a 60 s connect attempt
        searches. The genuine answers in that corpus — KB9MMT at 56.01 s and NS0A
        at 9.78 s — reach 15 of 15, and N0LCR-1's of 2026-08-15, one carrier low,
        reaches 14 of 15.

        :data:`_CLEAR_DB` widens what this search will take without moving that
        floor, and the second half is measured rather than argued: over the same
        population, 110,319,450 alignment-shifts, no alignment sweeps a comparable
        set of :data:`_RESP_MIN_HEARD` or wider clean once the low-clearance reads
        are dropped, and the best non-response is the same 5 of 15 either way. What
        it buys, over every recording this station holds: 583 connect-requests on
        file, 63 answers accepted against 56 before, none lost.

        The connected-ack rides here too, once the link-setup has gone out, and it
        is the burst that most needs to. Measured on the on-air recordings: NS0A
        starts its connect-response 0.12-0.16 s after our last transmitted sample
        (six attempts), and it answered a real VARA's link-setup 0.17 s after that
        burst's last sample — while this station's input stayed dead for 0.4 s after
        every transmission, and stays dead for 167-180 ms of it now. What reaches an
        energy gate in that window is band noise against a floor the dropout has
        collapsed, so the ack was being judged on audio it was never in. The stream
        carries every sample the receiver did produce, which is all that is ours to
        fix from here.

        Cost: ~1.1% of real time, in one batched transform per pass, shared by both
        searches. The track is band-limited to the tone alphabet by
        :func:`_top3_track`, which sits inside the SSB passband, so band-limited
        audio and splattering audio give the same answer and no filter is needed
        here.
        """
        self._stream_owns_response = True
        if self._final_query_for is not None:
            self._final_query_samples += len(samples)
        if self._intermediate_query_for is not None:
            self._intermediate_query_samples += len(samples)
        if self._stream_connect_ask(samples):
            return
        # Once the link is up the burst worth finding is the peer's DATA over, and
        # it needs finding for the same reason and against the same gate.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.bw in _RX2300.BASE_LEVELS):
            self._stream_owns_over = self._stream_owns_answer = True
            # Before the over search, and instead of it when either fires: an
            # answer taken keys, and the audio either side of that keying is not
            # one signal for `_stream_over` to score  [see _key]. An ask owed to a
            # clear channel is due here for the same reason and on the same terms.
            if self._ask_owed and self._ask_for_turn():
                return
            if self._owed_recovery and self._has_idle_over_nak():
                if not self._stream_recovery_cue(samples):
                    self._stream_over(samples)
            elif not self._stream_grant(samples) and not self._stream_answer(samples):
                self._stream_over(samples)
            # Last, and after the over search this block fed: an answer held for
            # the peer's turnaround is keyed the moment the channel goes quiet,
            # and a second over of the same window replaces it before that
            # [see _answer_over, _release_held_answer].
            self._release_held_answer(len(samples))
            return
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.bw == "500"):
            self._stream_owns_over = True
            if not self._stream_recovery_cue(samples):
                self._stream_over_500(samples)
            return
        self._reset_over_search()
        if self._expected_kind() is not VF.connect_response(self.bw):
            self._reset_stream()
            return
        self._st_buf = np.concatenate(
            [self._st_buf, np.asarray(samples, dtype=np.float64)])
        if len(self._st_buf) < MK.NFFT + _STREAM_BLOCK:
            return
        # Windows start at buf[0], which stays lattice-aligned because exactly the
        # samples consumed by the windows just taken are dropped.
        nwin = (len(self._st_buf) - MK.NFFT) // _ACK_GRID + 1
        span = MK.NFFT + (nwin - 1) * _ACK_GRID
        # ON OUR OWN GRID, WHOEVER THE PEER TURNS OUT TO BE. This lattice
        # accumulates across blocks, so it can only be taken at one offset, and
        # the one it has to be taken at is the one the link has not been measured
        # on yet. The response is what measures it  [see _note_peer_offset], and
        # a single-tone payload is read the same either way.
        bins, clear, resid = _top3_track(self._st_buf[:span], _ACK_GRID,
                                         self._band)
        self._st_track = np.concatenate([self._st_track, bins])
        self._st_clear = np.concatenate([self._st_clear, clear])
        self._st_resid = np.concatenate([self._st_resid, resid])
        self._st_live = np.concatenate(
            [self._st_live, _live_track(self._st_buf[:span], _ACK_GRID)])
        self._st_buf = self._st_buf[nwin * _ACK_GRID:]
        n = len(self._st_track) - _RESP_SPAN
        if n <= 0:
            return
        track, lat, live = self._st_track, self._st_lat, self._st_live
        clear, resid = self._st_clear, self._st_resid
        self._st_track = track[n:]               # keep only the unscored tail
        self._st_clear = clear[n:]
        self._st_resid = resid[n:]
        self._st_live = live[n:]
        self._st_lat = lat + n
        # The ack first: a gateway that repeats its connect-response is telling us
        # it never heard the link-setup, so the two cannot both be true, and this
        # is the one whose preamble no other burst on the band has ever matched.
        #
        # It is matched at the offset the response came in on, and at that one only.
        # The gateway that answers one carrier low acks one carrier low, so a zero-only
        # ack cannot complete the connect the response search has just opened — but
        # the ack names nobody, six bins over three symbols is the whole of its
        # evidence, and a shift IT went looking for would have nothing to keep it
        # honest.
        # Measured over the corpus below: at zero offset the preamble holds only
        # where a VARA session's own control bursts are, while at -1 carrier it holds
        # for 52 consecutive alignments in pos_p1_twosided_14110.wav — a two-tone
        # PACTOR-1 exchange with no VARA in it, and a wider plateau than the ack of
        # 2026-08-19 this recogniser was rebuilt to find.
        # ``_peer_shift`` is not that search: it is thirteen or more tones keyed to
        # the callsign we dialled, already confirmed on this attempt.
        if self.step == _I_LINKSETUP_SENT and lat * _ACK_GRID < _ACK_WINDOW:
            last = _ACK_WINDOW // _ACK_GRID - lat        # alignments still in window
            keep = slice(None, last + _ACK_SPAN)
            wide = _ack_hold(track[keep], clear[keep], live[keep], self._peer_shift)
            if wide >= _ACK_PLATEAU:
                self._reset_stream()
                self.io.log(f"rx connected-ack on the receive stream (preamble "
                            f"holds for {wide} alignments) — CONNECTED")
                self._connected()
                return
        idx = np.arange(n)[:, None] + _RESP_OFF
        heard = track[idx][:, :, 0]
        comparable = live[idx] & (clear[idx][:, :, 0] >= _CLEAR_DB)
        kind = VF.connect_response(self.bw)
        npre = len(kind.preamble)
        tones = np.asarray(self._burst_tones(self.called, kind), dtype=np.int32)
        best, shift, m, c = _best_fit(heard, comparable, tones, npre)
        if not _answers(m, c):
            if self._peer_answer(track, clear, live, n, lat, resid, include_base=True):
                return
            if m >= _RESP_MIN_TONES - 1:
                # The called station keyed the connect-response itself and it fell
                # one tone short of the offer gate. This describes our reception
                # of the reply, not the peer's reception of our request. Previously the
                # attempt read silence; a 12/13 K7EK-10 frame is what that hid.
                at = (lat + best) * _ACK_GRID / MK.FS
                if (self._near_miss_at is None
                        or abs(at - self._near_miss_at) >= _ANSWER_SPAN_S):
                    self._near_miss_at = at
                    self.io.log(
                        f"rx {self.called} keyed the connect-response but only "
                        f"{m}/{c} of {kind.n_payload} tones reached us — one short "
                        "of confirming the response")
            else:
                self._unattributed_answer(heard, comparable, kind)
            return
        self._reset_stream()                     # one accept per burst
        self._peer_shift = shift                 # where its ack will arrive too
        off = f", {MK.carrier_to_hz(shift):+.0f} Hz off frequency" if shift else ""
        self.io.log(f"rx connect-response for {self.called} found on the receive "
                    f"stream ({m}/{c} tones the receiver delivered, of "
                    f"{kind.n_payload}{off}) — no burst was gated")
        # ...and the fraction of a carrier under that shift, off the same symbols,
        # which is the first thing on the link that can say where the peer is
        # transmitting rather than which bin it lands nearest.
        took = comparable[best] & (heard[best] == tones + shift)
        self._note_peer_offset(resid, idx[best][npre:][took[npre:]], 0.0)
        # Symbols the receiver delivered no tone through go on as -1, not as
        # whatever the argmax happened to land on: the recogniser behind this scores
        # them as not comparable, and handing it the mute's own constant carrier —
        # or a noise peak out of the recovery — would be inventing evidence. The
        # rest go on at zero offset, so the recogniser judges what the peer said
        # rather than where it was tuned.
        self.on_rx_tones([int(t) - shift if lv else -1
                          for t, lv in zip(heard[best], comparable[best])], kind)

    # ---- DATA-over recognition ------------------------------------------- #
    def _reset_over_search(self) -> None:
        """Drop the over-search buffer and put its next scan back where a buffer
        with nothing carried into it belongs.

        The two are one act. A buffer starting empty carries nothing, so its first
        scan is due the instant it can hold a whole base over and not a sample
        later — making it wait for ``_OVER_CARRY`` would spend the longest record's
        span on a turnaround that may hold the shortest, and the block that used to
        sit on top of ``_OVER_NEED`` here spent half a second of every turnaround
        on a frame that was already whole in the buffer.
        """
        self._ov_buf = np.zeros(0)
        self._ov_due = _OVER_NEED
        self._ov_env = np.zeros(0)
        self._ov_framed = 0

    def _reset_grant_search(self) -> None:
        """The same for the turn-grant buffer  [see _stream_grant]."""
        self._grant_buf = np.zeros(0)
        self._grant_due = _SESSION_MIN

    def _reset_answer_search(self, skip: int = 0) -> None:
        """The same for the short-frame buffer  [see _stream_answer].

        ``skip`` discards the audio still arriving from a burst just named. A
        frame this end does not key back at leaves its own tail on the stream, and
        a fresh buffer takes that tail as a second frame: KB3AC-10 keyed fifteen
        responder-idles on 2026-08-30, 3.770 s apart, and thirty went into the
        log. The shortest gap any recording holds between two of these is 3.4 s,
        so dropping one frame's span costs nothing that has ever been seen.
        """
        self._ans_buf = np.zeros(0)
        self._ans_due = _ANSWER_MIN
        self._ans_skip = skip
        self._stale_nak_logged = False
        self._responder_nak_wait_samples = 0

    def _stream_over(self, samples) -> None:
        """Search the raw receive stream for the peer's DATA over, gate or no gate.

        The data-phase half of :meth:`on_rx_stream`, and there for the reason its
        connect-phase half is there: an energy gate cannot see these bursts.
        Measured on the 2026-08-06 session with KC9GHZ, the one recording this
        project holds of a held VARA link, the gateway's greeting over is 4.1 dB
        over the band noise, and the mail client sat in "awaiting greeting" for
        227 s while the receive chain behind the gate — which decodes that same
        audio 24/24 with a clean CRC and answers it correctly — never ran.

        Lowering the threshold is not the repair, and the way to see that is to
        give the tracker a floor it could not have had: a q25 over the band audio
        that FOLLOWS the over. Across nine window lengths from 5 s to 90 s the
        loudest of the over's 198 frames reaches 1.83-1.99x that floor and not one
        window puts it at the 2.0x a bracket opens on. A gate set low enough to
        take it takes the band it is sitting in as well.

        What the burst does carry is structure. :meth:`_rec3_alignment` scores the
        24 reference columns, whose lit bins are fixed by column class rather than
        by payload, and that is a question about the audio rather than about its
        level. It is the accept here, and :meth:`_peer_data_over` behind it still
        has to turbo-decode the frame to a clean CRC before anything is delivered
        or keyed.

        False accepts. Over the whole 227 s of that recording at ``_OVER_ONSET_STEP``
        — 667,809 alignments of real 40 m HF, our own twenty transmissions and a
        third station's two connect-requests at 159.9 and 163.5 s included
        [`corpora.ONAIR_STRANGER_REQUESTS`] — the two genuine frames in it score 24
        of 24, and every alignment reaching
        12 or more is inside one of their two match plateaux: 51.589-51.599 s, our
        own link-setup back through the monitor, and 58.088-58.098 s, the gateway's
        greeting. Each plateau is 10.3 and 10.0 ms of onset — one column, since the
        record's ``dw50`` is 10.67 ms and that is the whole span this scan sweeps.
        Elsewhere the distribution stops at 11, twice, against six at 10 and eleven
        at 9.
        ``_OVER_GUARD_MIN`` is 16, five columns clear of that, and the CRC is behind
        it.

        A decline is not silence. A peer that drops a speed level goes on keying
        overs, and one of those scores 5 or 6 of 24 here — under the 11 this
        recording's noise reaches, so the base-level score alone cannot tell the
        drop from an empty band, and a stretch of log with our keepalives and no
        `rx` line reads the same either way. ``_RX2300.unread_over`` scores the
        rest of the index-law family on the same buffer and says so when one of
        them holds a frame: on the 2026-08-14 two-sided bench session both
        stations' record-2 overs report their reference columns against nothing at
        all on noise, on the base-level overs of the same session, or on its
        control bursts. It costs 33 ms of the block below.

        A named over is then answered like any other, through the same
        :meth:`_answer_data_over` — the record it was named at is what
        :meth:`_peer_data_over` decodes it at. Both record-2 overs of that session
        read through here into their known plaintext and draw one per-over
        response  [see tests/kestrel/test_bw2300_rec2.py].

        Cost: 9% of real time on a quiet band — one :meth:`_rec3_alignment` and
        one family scan per ``_STREAM_BLOCK`` of new audio, 12 ms and 33 ms of the
        500 (measured on the buffer this now carries; the alignment alone is 2.3%,
        and the family scan only runs where the base level declines, which on a
        quiet band is every block). Across the turnaround the step is a sixteenth
        of that, which is the longest an over can then wait between ending and
        being read, and it is what keeps the answer inside the window the gateway
        is listening in  [see ``_OVER_TURNAROUND_STEP``]. The scan rate stayed
        the block's when the buffer grew to hold a record-2 over, because what
        grew is what is carried between scans and not the threshold the first
        scan waits for  [see ``_OVER_CARRY``].
        """
        self._ov_buf = np.concatenate(
            [self._ov_buf, np.asarray(samples, dtype=np.float64)])
        if len(self._ov_buf) < self._ov_due:
            return
        buf = self._ov_buf
        # Only what cannot yet hold a whole frame is carried forward; everything
        # else has been scored at every alignment it has. The next scan is due
        # behind that rather than behind the whole buffer, which is what holds the
        # scan rate where it belongs now that what is carried is longer than the
        # shortest frame worth scoring.
        self._ov_buf = buf[-(_OVER_CARRY - 1):]
        self._ov_due = _next_scan(len(self._ov_buf), _OVER_NEED,
                                  _OVER_TURNAROUND_STEP)
        lv, hits, of, mag = self._index_alignment(buf)
        if mag is None:
            return
        r = _RX2300.RECORDS[lv]
        if not len(mag) % r.ncols:
            # AN ALIGNMENT ENDING AT THE BUFFER'S LAST COLUMN IS NOT A VERDICT ON
            # THIS ROUTE, because here the buffer is still growing. A base over is
            # 4.36 s of emission around a 4.21 s frame and this rescans every
            # 31 ms, so a scan lands inside the 0.15 s between: the true alignment
            # is past where the buffer reaches, the scan takes the last one that
            # fits and reads the frame a column early, and that scores 18-20 of 24
            # — over `_OVER_GUARD_MIN`, so it is claimed — and cannot decode. That
            # is `_UNDECODED`: the NAK budget spent on a frame nobody broke, and
            # the search reset under audio that reads 24 of 24 one scan later. Six
            # of the 286 peer overs across 32 bench fetches were read there, five
            # of them costing the delivery. The bracket route has no such wait to
            # make — what it hands over is a burst that has closed — so the check
            # is here rather than in :meth:`_peer_data_over`. A whole multiple of
            # the record is the same reading for a window holding two overs, which
            # is a thing a stock sender keys  [see _peer_data_over].
            return
        state = _window_state(mag, r)
        if state is _UNKNOWN:
            # The frame is whole and what follows it does not say yet whether the
            # peer has finished. Waiting is bounded by the reading itself — at
            # `_PROBE_COLS` the structure decides — and it is the difference
            # between answering a window and answering the first block of one
            # [see _window_state]. On the FINE grid while it lasts: the buffer is
            # long by now and the standing schedule is half a second, which would
            # spend a whole block of turnaround waiting for 341 ms of columns.
            self._ov_due = len(self._ov_buf) + _OVER_TURNAROUND_STEP
            return
        if not self._answer_data_over(buf, hold=state is _KEYING):
            return
        if self._held_answer is None:
            self._reset_over_search()    # consumed, and we have just transmitted
            return
        # Nothing was keyed, so the audio behind the over we just read is still
        # the peer's and still arriving — the second over of a window it has not
        # unkeyed from. The ordinary carry would leave the FIRST over whole in the
        # buffer for another scan or two, where it reads as the peer repeating
        # itself and would replace the held answer with the one that ends a
        # delivery. So the buffer restarts at the frame's own end, to within the
        # column the alignment was found on.
        self._ov_buf = _after_frame(buf, mag, r)
        self._ov_due = _next_scan(len(self._ov_buf), _OVER_NEED,
                                  _OVER_TURNAROUND_STEP)

    def _stream_over_500(self, samples) -> None:
        """:meth:`_stream_over` at BW500, where the over is found by its span.

        A base narrow over is one frame or two — 403 columns or 796, 4.31 s or
        8.51 s — and a bracket force-closed at 6 s holds neither whole of the
        second. Two level-1 frames need 10.67 s; the carry covers the longest
        pair at every supported level, with the same one-second margin.
        A stock responder holding more than one block keys the two-frame
        over: three fetches on 2026-09-03 each read six single-frame overs and
        then cut the first two-frame one into a 6.0 s piece and a 2.7 s piece
        with no data frame in either, nothing answered it, the responder never
        repeated it, and the link closed on idle with the message at the far end.

        The span is measured by the receiver's own law  [rx.varahf500.burst_spans]
        on a per-column energy track kept as the audio arrives, so the scan runs
        on every 20 ms poll at 0.1 ms — 0.5% of real time on a quiet band — and
        the decode, one call of 53-63 ms on a real burst, runs once per closed
        run. Over the 52 bursts of the loopback session fed a poll at a time,
        every over is read and its answer keyed 0.054-0.063 s after the last
        sample, where a stock station answers at 0.083-0.107 s. A run that will
        not decode is dropped rather than scored again on the next poll.
        """
        buf = self._ov_buf = np.concatenate(
            [self._ov_buf, np.asarray(samples, dtype=np.float64)])
        if (fresh := (len(buf) - self._ov_framed) // _OVER500_COL):
            cols = buf[self._ov_framed:self._ov_framed + fresh * _OVER500_COL]
            spec = np.fft.rfft(cols.reshape(fresh, _OVER500_COL), axis=1)
            e = (np.abs(spec[:, _OVER500_BAND]) ** 2).sum(axis=1)
            self._ov_env = np.concatenate([self._ov_env, e])
            self._ov_framed += fresh * _OVER500_COL
        env = self._ov_env
        if len(env) < _OVER500_MIN + 1:
            return
        sm = 0.5 * (env[1:] + env[:-1])
        on = sm > 0.15 ** 2 * sm.max()
        edge = np.diff(on.view(np.int8), prepend=np.int8(0))
        rise, fall = np.flatnonzero(edge > 0), np.flatnonzero(edge < 0)
        runs = [(int(a), int(b)) for a, b in zip(rise, fall) if b - a >= _OVER500_MIN]
        if not runs:
            if (spare := len(buf) - _OVER500_CARRY) >= _OVER500_COL:
                self._drop_over_500(spare - spare % _OVER500_COL)
            return
        a, b = runs[0]
        stop = (b + 1) * _OVER500_COL
        if len(buf) < stop + _OVER500_TAIL:
            return
        x = buf[max(0, a * _OVER500_COL - _OVER500_LEAD):stop + _OVER500_TAIL]
        self._answer_data_over(x)
        if len(self._ov_buf):            # not emptied by our own keying
            self._drop_over_500(stop)

    def _drop_over_500(self, n: int) -> None:
        """Forget the first ``n`` samples of the narrow over buffer, a whole
        number of columns of its track."""
        self._ov_buf = self._ov_buf[n:]
        self._ov_env = self._ov_env[n // _OVER500_COL:]
        self._ov_framed -= n

    def _stream_recovery_cue(self, samples) -> bool:
        """The responder's idle on the raw stream while an over of its is
        owed the recovery, answered with `session-over-nak` where a stock caller
        keys it. True when that answer went out.

        The measured form, off the two-stock BW500 cables of 2026-09-11: a
        caller that could not read the greeting keys nothing into the over's
        turnaround, the responder idles 1.5 s after its unkey, the caller keys
        this frame 0.242 s after the idle's last sample, and the responder
        re-sends from speed level 1 and climbs — twice, to the symbol. On a
        transport that only brackets the idle is `_read_burst`'s, at whatever
        offset its gate closes on; here the scan that answers it is due at the
        lead itself rather than on the quarter-block grid  [see _OVER_NAK_LEAD_S].
        Owns the idle only while the debt is owed, so every other BW500 idle
        stays the bracket route's.

        BW2750 stock uses the same generated request behind its idle after
        unread DATA (two-stock captures 2026-09-16). Its retransmission starts
        at host level 2; the receive search retains that bandwidth's own ladder.
        """
        self._stream_owns_answer = self._owed_recovery
        if not self._owed_recovery:
            self._reset_answer_search()
            return False
        samples = np.asarray(samples, dtype=np.float64)
        if self._ans_skip:
            drop = min(self._ans_skip, len(samples))
            self._ans_skip -= drop
            samples = samples[drop:]
        self._ans_buf = np.concatenate([self._ans_buf, samples])
        if len(self._ans_buf) < self._ans_due:
            return False
        buf = self._ans_buf
        self._ans_buf = buf[-(_SESSION_NEED - 1):]
        self._ans_due = _next_scan(len(self._ans_buf), _SESSION_MIN)
        if (len(buf) < _SESSION_MIN or not self._peer_responder_idle(
                buf, _top3_track(buf, band=self._band,
                                 bin_offset=self._grid))):
            return False
        if self._idle_held < _OVER_NAK_LEAD_S:
            self._ans_due = len(self._ans_buf) + int(
                (_OVER_NAK_LEAD_S - self._idle_held) * MK.FS)
            return False
        self._reset_answer_search(_SESSION_NEED)
        self.io.log(f"rx {self._idle_kind.name} — {self.called} is still "
                    f"transmitting, named {self._idle_held:+.3f} s from its "
                    "last symbol (not progress)")
        return (self._idle_held <= _GRANT_FRESH_S and self._a_gap()
                and self._reack())

    def _stream_grant(self, samples) -> bool:
        """Search the raw receive stream for the peer's answer to our turn-request,
        gate or no gate. True when the turn was taken on it.

        THE LAST BURST IN THE CONNECTED STATE THAT AN ENERGY GATE STILL HAD TO
        FIND, and it deadlocked two gateways before this. The connect-response, the
        connected-ack and the DATA over each got a stream route for the same
        reason: the gate opens at 2.0x its tracked floor, which is 6.02 dB, and a
        gateway's burst does not stand that far over the band it arrives in.

        Measured on the 2026-08-23 KB3AC-10 session, against the floor the live
        gate was actually tracking at each changeover: the gateway's three DATA
        overs stand +1.66, +1.02 and +0.90 dB over it and its three answers to our
        turn-requests +1.67, +4.37 and +5.00. Not one of the six could open the
        gate, and replaying that recording through the tool's own segmenter
        produces one bracket in 187 s. The overs still reached the host, because
        `_stream_over` owns them; the turn grants had nowhere to go, and the
        session ended `turn-request unanswered 3x — giving the turn back, 2
        block(s) still queued` with the gateway's answer on the tape.

        On the same tape :meth:`_peer_drained` takes those answers at 30/30 and
        30/31 payload tones, and takes them with anything from 0.00 to 0.60 s of
        their heads cut away — so what was missing was never the recogniser or the
        changeover blackout in front of it. It was that nothing ever ran it.

        Bounded to the window it is asked in: the turn is `_TURN_ASKED` only
        between a request of ours and its answer, which is where a gateway's grant
        can be, so the search neither runs nor can accept anywhere else.

        AND BOUNDED IN AGE. The last-symbol guard below asks whether the frame had
        closed; it cannot ask whether it had closed RECENTLY, and a buffer that
        keeps growing across a whole turnaround holds a burst the peer has since
        keyed over. On 2026-09-03 at 22:01z the grant was named at 120.06 out of
        audio whose newest sample was the peer's next burst, from a poll of its
        own that ended at 119.23 and with the peer keyed again since 119.58: the
        4.4 s DATA over went out across that and the responder read none of it,
        twice that evening. So a grant is refused once it is a block old and the
        buffer is dropped, which asks again on audio recorded since. Across the
        222 grants of that bench the rule refuses those twelve — every one of them
        followed by an over the peer never logged — and none of the 210 whose over
        it read, which were named 0.00-0.37 s behind the newest audio they were
        found in.
        """
        if self.turn != _TURN_ASKED:
            self._reset_grant_search()
            return False
        self._grant_buf = np.concatenate(
            [self._grant_buf, np.asarray(samples, dtype=np.float64)])
        if len(self._grant_buf) < self._grant_due:
            return False
        buf = self._grant_buf
        # Only what cannot yet hold a whole frame is carried forward; everything
        # else has been scored at every alignment it has.
        self._grant_buf = buf[-(_SESSION_NEED - 1):]
        self._grant_due = _next_scan(len(self._grant_buf), _SESSION_MIN)
        # Both answers to a turn-request are read off one transform, for the reason
        # `_stream_answer` takes three off one.
        track = _top3_track(buf, band=self._band, bin_offset=self._grid)
        granted = VF.SESSION_DRAINED_RESPONDER.name
        # A grant is acted on by keying a 4.4 s DATA over, so it cannot be taken
        # from a buffer too short to hold the frame: `_peer_drained` carries no
        # length guard of its own, and the turnaround grid's first scans come
        # while a 32-symbol answer is still arriving — accepting there keys the
        # over across the tail of a peer that is still transmitting, and a
        # half-duplex peer decodes none of it.
        if len(buf) < _SESSION_NEED or not self._peer_drained(buf, track):
            if not self._peer_responder_release(buf, track):
                return False
            granted = VF.SESSION_TURN_RELEASE_RESPONDER.name
        self._reset_grant_search()
        if self._grant_held > _GRANT_FRESH_S:
            self.io.log(f"{granted} named {self._grant_held:.2f} s behind the "
                        "newest audio in the buffer — the peer has keyed since, "
                        "so the turn is not taken on it")
            return False
        self._close_echo_window()
        self._turn_granted(buf, granted)
        return True

    def _stream_answer(self, samples) -> bool:
        """Everything else the peer keys at us in the connected state, found on the
        raw receive stream. True when one of them was.

        `_stream_over` owns the wideband bursts and :meth:`_stream_grant` the answer
        to a turn-request. Between them they left the short frames a gateway keys
        during the data phase with no route but the energy gate — which is the gate
        that cannot see any of them, for the reason `_stream_grant` measures — and
        behind a ``_DATA_OVER_MIN`` length guard, which is what silently declined
        the ones it did bracket on 2026-08-22.

        Counted across the two 2026-08-23 sessions: seventeen short gateway bursts
        are on the tapes and two were recognised live — KE8LVA's connected-ack
        through the bracket and KB3AC-10's through the stream. The other fifteen,
        thirteen responder-idles and two turn grants, reached nothing at all.

        Seven frames, one buffer, because they are all 32 symbols or shorter and
        a peer keys only one of them at a time:

        * the eight-symbol continue burst while the turn is ours, which answers an
          INTERMEDIATE over of a delivery of ours and asks for the next one — the
          frame an outbound message longer than one over lives on  [see
          _peer_over_continue];
        * the 11-symbol two-tone control burst and ``SESSION_IDLE_RESPONSE`` while
          the turn is ours — both are the peer answering an over or an idle of
          ours, and both drain the queue  [see _took_control_burst,
          _took_idle_response] — and the same control burst while the turn is
          the peer's behind a release of ours, which is its poll  [see _took_poll];
        * ``SESSION_TURN_RELEASE_RESPONDER`` while the turn is the peer's, which is
          a gateway handing the channel over with nothing having asked for it, and
          ``SESSION_DRAINED_RESPONDER`` in the state that makes it the same
          handover at 32 symbols — the two frames here that owe an answer on the
          air  [see _took_responder_release, _drained_hands_over];
        * ``SESSION_RESPONDER_IDLE`` and ``SESSION_RESPONDER_OVER_IDLE`` in any
          turn state, which are the gateway's own cadence rather than an answer to
          anything — and the gap one of them opens is where a re-acknowledgement
          goes  [see _reack]; and
          ``SESSION_RESPONDER_OVER_ANSWER``, which is its answer to an over of
          ours — the same answer the continue burst carries when a queue is behind
          it, and otherwise nothing but the peer still transmitting, which is what
          the give-up budget counts  [see _peer_responder_idle, _peer_over_answer].

        The buffer runs from 0.34 s to 1.87 s — the shortest of these frames at its
        first scan, the longest plus a block at its widest — and stays under
        :data:`_DATA_OVER_MIN` throughout, so the length guard that makes the
        bracket route decline an over does not decline this. It used to wait for
        1.87 s before its FIRST scan, which is the longest frame's span plus a
        block spent looking for a 0.73 s one that was already whole: 1.10 s of the
        1.31-1.36 s this station took to answer the 2026-08-29 handovers was spent
        there, on audio it was holding  [see :func:`_next_scan`].

        One transform serves all four searches, for :func:`_top3_track`'s own
        reason: measured over a minute of band noise, the connected-state stream
        costs 5.9% of real time with this route and 1.8% without, against 12% when
        each search took its own. The fourth added 1.1 ms to a 0.5 s block, which
        is what a search that reuses the transform costs.
        """
        if self.state != VaraState.CONNECTED or self.role != "initiator":
            self._reset_answer_search()
            return False
        samples = np.asarray(samples, dtype=np.float64)
        if self._ans_skip:
            drop = min(self._ans_skip, len(samples))
            self._ans_skip -= drop
            samples = samples[drop:]
        self._ans_buf = np.concatenate([self._ans_buf, samples])
        if len(self._ans_buf) < self._ans_due:
            return False
        buf = self._ans_buf
        self._ans_buf = buf[-(_SESSION_NEED - 1):]
        self._ans_due = _next_scan(len(self._ans_buf), _SESSION_MIN)
        track = _top3_track(buf, band=self._band, bin_offset=self._grid)
        took = None
        self._responder_nak_wait_samples = 0
        self._intermediate_answer_wait_samples = 0
        data_nak = self._peer_data_nak(buf, track)
        if self._data_nak_wait_samples:
            self._ans_due = min(self._ans_due,
                                len(self._ans_buf) + self._data_nak_wait_samples)
        if data_nak:
            took = self._took_data_nak
        elif self._peer_intermediate_query_answer(buf, track):
            took = self._took_intermediate_query_answer
        elif (self.turn == _TURN_OURS or self._stalled()) and self._peer_nak(buf, track):
            took = self._took_nak
        elif ((self.turn == _TURN_OURS or self._stalled())
              and self._peer_over_continue(buf, track)):
            self.io.log(f"rx over-continue from {self.called} — the over was "
                        "heard and the next one is asked for")
            took = (self._took_over_continue if self.turn == _TURN_OURS
                    else self._took_stall_answer)
        elif len(buf) < _SESSION_MIN:
            # The continue burst is the only frame here short enough to be whole
            # in a buffer this size, and its own recogniser needs all eight of its
            # symbols delivered. The others are named off a third of themselves —
            # the control burst off its four-pair preamble, the 32-symbol frames
            # off nine of their tones — and every one of them is answered by
            # keying, which at this length would go out across a peer that is
            # still transmitting.
            return False
        elif ((self.turn == _TURN_OURS or self._stalled())
              and self._peer_responder_nak(buf, track)):
            # Out of turn what the peer could not read is our acknowledgement, so
            # the ladder re-keys that and no DATA goes out  [see _took_nak, the
            # same split on the caller-side NAK].
            took = (self._took_responder_nak if self.turn == _TURN_OURS
                    else self._took_responder_nak_out_of_turn)
        elif self.turn == _TURN_OURS and self._peer_idle_response(buf, track):
            took = self._took_idle_response
        elif self.turn == _TURN_OURS and self._peer_control_burst(buf, track):
            took = self._took_control_burst
        elif self.turn == _TURN_PEER and self._peer_responder_release(buf, track):
            took = self._took_responder_release
        elif self.turn == _TURN_PEER and self._drained_hands_over(buf, track):
            # The same handover at 32 symbols. `on_rx_audio` has owned it since
            # the BW500 arms of 2026-09-11, and at 2300/2750 the bracket route is
            # not where it arrives: K0SI keyed it at 67.334 and 76.734 s of the
            # 2026-09-16 40 m tape, 31/31 tones, and the energy gate that has
            # never seen a gateway's short burst missed both — the run named "no
            # release in K0SI's cadence" and re-keyed the control burst with the
            # handover on the tape  [see _drained_hands_over].
            took = self._took_drained_handover
        elif (self.turn == _TURN_PEER and self._released and not self._txq
              and self._peer_control_burst(buf, track)):
            took = self._took_poll
        elif self._peer_responder_idle(buf, track):
            if self._idle_held < 0:
                self._ans_due = len(self._ans_buf) + max(
                    1, int(np.ceil(-self._idle_held * MK.FS)))
                # Its last symbol has not arrived, and this frame is named off
                # about a third of itself: the fit is allowed a start outside the
                # audio, which is what reads a burst our own mute cut the head
                # from. The buffer keeps it and the scan 0.34 s behind this one
                # names it whole, where taking it here would leave the gap it
                # opens unmeasurable until the peer keys another 3.4 s later.
                return False
            # NOT `_progressed`. The frame says the peer is transmitting and
            # nothing else, and a peer transmitting is exactly what a stalled
            # session looks like: at the bench on 2026-08-31 a reply over that
            # would not decode was followed by eighteen of these, each one zeroing
            # the give-up budget, and the session hung ~100 s on the mail timeout
            # with no route out. The budget counts turnarounds this build read an
            # ANSWER in  [see _MAX_WITHOUT_PROGRESS].
            self._reset_answer_search(_SESSION_NEED)
            self.io.log(f"rx {self._idle_kind.name} — {self.called} is still "
                        f"transmitting, named {self._idle_held:+.3f} s from its "
                        "last symbol (not progress)")
            # WHAT THE FIGURE BUYS is the one instant a rung is worth keying. The
            # frame opens the peer's own 1.7-1.9 s listening gap, and the audio
            # held past its last symbol is how much of that gap has already gone;
            # a scan that arrives a block or more late is naming a burst the peer
            # has keyed past  [see _reack, _GRANT_FRESH_S].
            if not (self._idle_held <= _GRANT_FRESH_S and self._a_gap()):
                return False       # stale, or a window of the peer's is still open
            if self._reack():
                return True
            # Nothing owed. Only in the peer's turn: holding the turn, the burst
            # due in this gap is our own DATA over or the grant we are reading for,
            # and a keepalive keyed on top of either is the frame that replaces it.
            if self.turn == _TURN_PEER and self._answer_peer_idle():
                return True
        elif self._peer_over_answer(buf, track):
            # With a queue this is the same answer the two shorter frames carry
            # and it continues the delivery. With none it is what it always was:
            # no recording says what a station keys at this frame with nothing
            # left to send, and the one that holds it holds no second gateway
            # transmission afterwards  [see _peer_over_answer].
            continues = self.turn == _TURN_OURS and bool(
                self._txq or self._tx_pending is not None)
            if continues and self._answer_held < 0:
                self._ans_due = min(self._ans_due, len(self._ans_buf)
                                    + max(1, int(np.ceil(-self._answer_held * MK.FS))))
                # Its last symbol has not arrived yet. Nothing is owed until it
                # has: the two shorter answers are whole when they are named and
                # this one is named off its first third.
                return False
            self.io.log(f"rx session-responder-over-answer — {self.called} "
                        "answered the DATA over")
            if continues:
                took = self._took_control_burst
            else:
                self._reset_answer_search(_SESSION_NEED)
                self._progressed()
        if took is not None and self._held_answer is not None:
            # A WINDOW OF THE PEER'S IS STILL OPEN. Whatever this frame was, the
            # peer cannot be answering us in the middle of its own transmission,
            # and the burst that would go out here lands on the block this
            # station is still waiting to decode  [see _answer_over]. The over
            # search keeps the audio either way.
            self.io.log(f"{self.called} is mid-window — not answering the frame "
                        "named in it")
            return False
        if took is None:
            if self._intermediate_answer_wait_samples:
                self._ans_due = min(self._ans_due, len(self._ans_buf)
                                    + self._intermediate_answer_wait_samples)
            if self._responder_nak_wait_samples:
                self._ans_due = min(self._ans_due, len(self._ans_buf)
                                    + self._responder_nak_wait_samples)
            # False even where the cadence was named: only a keying makes the
            # audio either side of this block two signals, and the over search is
            # still owed every sample of a block that keyed nothing  [see _key].
            return False
        self._reset_answer_search()
        took()
        return True

    def _rec3_alignment(self, x: np.ndarray) -> tuple[int, np.ndarray | None]:
        """Best base-level frame alignment in ``x``: (reference-column hits, band
        magnitudes from the frame's first column).

        The base level is the session bandwidth's, so the name is BW2300's: at
        BW2750 this scores the same 395 columns on a 20-bin comb.

        Of the 395 emission columns of a base over, 24 are reference columns
        whose lit bin is fixed by the column's class rather than by the payload, so
        at the true frame start they all land where they are supposed to and
        anywhere else the match falls to chance (~1.5 of 24). That makes "is this
        audio a VARA wideband frame at all" a question that can be answered without
        decoding anything, which is what lets it run in front of a keying decision.

        Returns -1 when the audio cannot hold a whole frame — a piece shorter than
        395 columns (4.21 s) has no alignment to score, however long the segmenter's
        bracket was.

        THE FRAME MAY HAVE STARTED BEFORE THE BUFFER DID, which is why the record's
        own training columns are stood in as silence ahead of it. This station is
        deaf for 0.17-0.19 s after every keying — PTT held 0.14-0.15 s past the last
        sample, rig audio back 0.03 s after PTT-off — and :meth:`_key` empties the
        over-search buffer at each keying, so a peer that keys at or before our
        PTT-off is first heard 16-18 columns into its burst with nothing ahead of
        it. The search reaches no further back than the first sample it is given, so
        the frame's column 0 falls outside the buffer and the 24 reference columns
        fall to chance: the 2026-09-11 KC9GHZ greetings, whole and CRC-clean off the
        tape, score 5 and 6 of 24 from 128 ms and 85 ms in and read as an empty
        band. The reach is the record's own ``lead``, the same bound
        :func:`varahf2300.decode_over` stands in on the bracket route — 24 of 24 and
        a clean CRC on both of those greetings from 213 ms and 181 ms in. It cannot
        flatter the score, because no record carries a reference column inside its
        own lead (15 at both base records), and the columns it stands in read as
        erasures the turbo pass carries.

        The pad is local to this scan: ``mag`` is columns from the frame's first,
        which is what every caller consumes, and :func:`_after_frame` measures the
        audio behind the frame from the buffer's end, which the pad does not move.

        Onsets are ranked by hits and then by the reference bins' energy share, the
        way :func:`varahf2300._alignments` ranks them. On a clean cable the hits
        saturate a third of a column either side of the true onset, and the
        earliest onset to reach them reads its data columns across two symbols.
        """
        base = _RX2300.BASE_LEVELS[self.bw]
        r = _RX2300.RECORDS[base]
        x = np.concatenate([np.zeros(r.lead * r.dw50), np.asarray(x, float)])
        best, best_mag = (-1, 0.0), None
        for onset in range(0, r.dw50, _OVER_ONSET_STEP):
            ncol = (len(x) - onset) // r.dw50
            if ncol < _RX2300._BASE_NCOLS:
                continue
            mag = _RX2300._band_mag(x[onset:], base, ncol)
            hits, share, _ = _RX2300._guard_scores(mag, base)
            if len(hits) == 0:
                continue
            g = int(np.lexsort((-share, -hits))[0])   # most hits, then energy share
            if (int(hits[g]), float(share[g])) > best:
                best, best_mag = (int(hits[g]), float(share[g])), mag[g:]
        return best[0], best_mag

    def _index_alignment(self, x) -> tuple[int, int, int, np.ndarray | None]:
        """Which index-law record this audio holds a frame of, and where its
        columns light: ``(level, hits, of, band magnitudes)``.

        The base level is asked first and on its own terms — it is what a healthy
        link runs at, :meth:`_rec3_alignment` is the cheap score, and every
        alignment it takes holds all 395 columns. Only when it declines is the rest
        of the family scored, which costs 26 ms and is where a peer that dropped a
        gear turns up: a record-2 over reaches 24 of 24 of its own reference
        columns on audio the base level reads as 5. The family is the session
        bandwidth's: BW2750 includes its independently measured records 101/102,
        not the differently allocated BW2300 records 1/2.

        The magnitudes are None for audio that is not an over, and also for a
        record named in a window too short to hold it whole. Those are not the same thing and
        neither is a frame that failed — a window that does not reach the last
        column has shown nothing about the frame, so it is declined rather than
        counted against the NAK budget  [see :meth:`_undecoded_over`].

        ``hits`` and ``of`` are the record's own reference columns, not the base
        level's. Both records tabled here happen to carry 24, so the guard reads on
        one scale; the comparison is written as a ratio anyway, because that is the
        thing being claimed and the coincidence is not.
        """
        base = _RX2300.BASE_LEVELS[self.bw]
        hits, mag = self._rec3_alignment(x)
        if hits >= _OVER_GUARD_MIN:
            return base, hits, _OVER_REF_COLS, mag
        others = tuple(lv for lv in _RX2300.INDEX_LEVELS_BW[self.bw] if lv != base)
        partial = None
        weak = []
        for lv, ref, of, at in _RX2300.index_guard(x, others):
            if not of or ref * _OVER_REF_COLS < _OVER_GUARD_MIN * of:
                r = _RX2300.RECORDS[lv]
                if (of == _OVER_REF_COLS and ref >= _OVER_CRC_MIN
                        and len(x) - at >= r.ncols * r.dw50):
                    weak.append(lv)
                continue
            r = _RX2300.RECORDS[lv]
            seg = np.asarray(x[at:], dtype=np.float64)
            ncols = len(seg) // r.dw50
            if ncols < r.ncols:
                partial = partial or (lv, ref, of, None)
                continue
            # Keep the tail: the raw-stream route must distinguish a completed
            # low-speed window from a peer still keying its next frame.
            return lv, ref, of, _RX2300._band_mag(seg, lv, ncols)
        # Reference loss is not necessarily payload loss: the robust decoder
        # can correct a frame that this fast detector would discard. Refine at
        # most two alignments per nominated record; only CRC-clean candidates
        # enter the ordinary tail, link-setup, echo and delivery checks. A failed
        # speculative decode remains unrecognized audio and must never key a NAK.
        for lv in weak:
            for _, g, mag in _RX2300._alignments(np.asarray(x, float), 2, lv):
                ref, _, _ = _RX2300._guard_scores(mag, lv)
                if ref[g] < _OVER_CRC_MIN:
                    continue
                fr = _RX2300.check_frame(
                    _RX2300.onair_to_frame(_RX2300._onair_llr(mag[g:], lv), lv), lv)
                if fr.crc_ok:
                    return lv, int(ref[g]), _OVER_REF_COLS, mag[g:]
        return partial or (base, max(hits, 0), _OVER_REF_COLS, None)

    def _peer_data_over(self, samples) -> list[bytes]:
        """The bodies of the blocks a DATA over in this audio carries, or empty.

        **An over is one or two blocks** [spec 04 §4.1], and which is a property
        of the burst rather than of the bandwidth: at BW500 a 796-column burst
        carries two frames and a 403-column one carries one — 45 of one and 6 of
        the other over a whole logged session, and the burst's own measured length
        is what says which  [rx.varahf500.frames_carried]. The wide bandwidths key
        one block an over on every recording held, so their branch returns a list
        of one and the caller does not care which bandwidth it is reading.

        Ninety bytes at the wide base level, forty-eight at the record below it,
        forty-four at BW500 level 4 and thirty-five at its record 2: the level is
        not signalled anywhere but in the waveform, so it is
        :meth:`_index_alignment` — and at BW500 the reference-column score of
        ``decode_burst_frames`` — that names it and the body's length that
        follows. The host is handed the payload either way  [see :meth:`_deliver`].

        A recogniser and nothing else: it decides whether the audio is the one
        received burst an initiator answers by keying, and hands the body back for
        :meth:`_answer_data_over` to deliver and answer. Delivering from here was
        the fault: :meth:`VaraIO.data` runs the host's mail client, which answers
        the greeting by calling :meth:`send` straight back into this object, so
        the host would be answering a burst still being identified, with the echo
        window and the turn accounting untouched.

        Length used to be the whole test, and length is not evidence of anything.
        The segmenter force-closes a bracket at 6 s, so on a busy channel, or with
        the station's other modem running, or on band noise above the gate's enter
        threshold, it hands over 6-second pieces of whatever is on the frequency;
        answering those means keying the transmitter at an unrelated station.

        So the audio has to be positively identified, cheapest test first. A
        bandwidth with no validated codec is answered no, which is the fail-closed
        reading of having no recogniser rather than the old reading of having no
        objection.

          1. it carries an index-law frame — ``_OVER_GUARD_MIN`` of the 24
             reference columns of one of them line up (~35 ms base, +26 ms for the
             rest of the family, no decoding);
             a complete lower-speed frame with at least ``_OVER_CRC_MIN`` hits
             may instead be nominated for a CRC-only decode attempt;
          2. that frame turbo-decodes with a clean CRC-16;
          3. it is not a link-setup. A link-setup is step 4 of a *connect* and never
             an over to answer, and it is the only wideband frame that names a
             station — so this is where a stranger opening a session on our
             frequency is caught (it names them), and our own link-setup coming
             back through the receiver with it (it names us);
          4. it is not an over we keyed. A DATA over names nobody, so the frame
             carries nothing to tell ours from the peer's — but we have the ninety
             bytes we put on the air, and they are exact. Without this the whole
             connect-phase argument above has no data-phase counterpart: our own
             wideband bursts return through the rig's monitor decoding perfectly,
             because they are real and they are ours — measured on the 2026-08-06
             KC9GHZ recording, where our own link-setup comes back just after 51 s
             at 24 of 24 reference columns with a clean CRC and the callsign it
             names is this station's. A DATA over travels that path with nothing to
             name it, so delivering one hands our own outgoing mail back to the
             host as received mail.

        Anything that fails is not answered, with one exception: a burst that
        passes the strong reference gate in (1) and fails (2) has been identified as an over and cannot be
        read, and it returns ``[_UNDECODED]`` rather than nothing so that
        :meth:`_answer_data_over` can say so on the air. A missed over costs a
        retry; keying on someone else's transmission costs rather more than that.

        **BW500 needs a CRC-clean anchor for an unread-window verdict.** Its gates
        are one call — ``decode_stream`` scores the reference columns of both
        speed levels, decodes, and drops a link-setup for itself. Our own
        emission decodes from its detected span — the alignment lock is
        differential and reaches back over the nine lead-in columns the
        transmitter renders silent  [``tests/kestrel/test_bw500_cut_head``] —
        and is refused as a body we keyed (4). An all-bad burst names nothing
        and is not reported as an over we could not read: a CRC-clean non-echo
        frame inside an incomplete peer window identifies that window, and its
        missing partner draws the measured narrow NAK. The one window that
        needs no anchor is the peer's first over, where nothing of ours has
        been keyed and an all-bad burst cannot be ours  [see _first_over_owed].

        Whose turn it is is deliberately *not* a gate here. The turn law is strict
        alternation  [spec 05 §5.4], so an over arriving while we hold the turn is
        one of three things and not two: ours (4), a peer that has taken the turn
        back, or a station we are not in session with. No recording shows how a
        gateway asks for the turn — the one turn-request ever captured is the
        caller's, and our own has never been answered on air — so a peer that
        simply starts sending again is a live possibility, and refusing its overs
        on a turn state that stale would be a station that has stopped listening to
        a gateway still talking. Nothing in the frame separates that peer from the
        stranger, so :meth:`_answer_data_over` answers both alike and reads the
        turn off how many of them arrive  [see :meth:`_over_into_our_turn`].

        What this still cannot do is tell a stranger's mid-session DATA over from
        our peer's. Decoded off air, a base DATA over is 90 bytes of the session's
        byte stream and a 2-byte CRC — no address field, no session id, nothing
        outside the payload that names a station. So an over is not claimed to be
        addressed to us, only to be an over rather than a connect and not one of
        ours; what narrows it further is session state the caller holds (CONNECTED,
        initiator), not anything the frame carries.
        """
        x = np.asarray(samples, dtype=np.float64)
        secs = len(x) / MK.FS
        if self.bw == "500":
            return self._peer_data_over_500(x, secs)
        if self.bw not in _RX2300.BASE_LEVELS:
            self.io.log(f"rx {secs:.1f} s burst at BW{self.bw}: no over recogniser "
                        "for this bandwidth — not answering")
            return []
        base = _RX2300.BASE_LEVELS[self.bw]
        # ONE WINDOW MAY HOLD TWO OVERS. A stock 4.9.0 keys them back to back
        # inside one PTT once a delivery has had a run of successes: on the cables
        # of 2026-09-09 its keyings run 4.38 s until the seventh over of a
        # four-message fetch, where it keys 8.60 s. Read off that capture the two
        # frames are column-contiguous, 24 of 24 reference columns and CRC-clean
        # each, 89 payload bytes each — and a station that reads only the first
        # loses the other's 89 bytes outright and acknowledges a window it has
        # half of. What follows an over is searched rather than assumed to start
        # at the record's own next column: nothing measures the gap between two
        # overs of one window, and the alignment search is what would have to find
        # it either way.
        bodies: list[bytes] = []
        while True:
            lv, hits, of, mag = self._index_alignment(x)
            rec = "" if lv == base else f" at record {lv}"
            if mag is None:
                if bodies:
                    break
                why = ("not a wideband over" if lv == base
                       else "an over this buffer does not hold whole")
                self.io.log(f"rx {secs:.1f} s burst: {hits}/{of} reference columns"
                            f"{rec} — {why}, not answering")
                return []
            fr = _RX2300.check_frame(
                _RX2300.onair_to_frame(_RX2300._onair_llr(mag, lv), lv), lv)
            if not fr.crc_ok:
                if bodies:
                    break
                self.io.log(f"rx {secs:.1f} s burst: {hits}/{of} reference columns"
                            f"{rec} but the frame will not decode")
                return [_UNDECODED]
            frame = bytes(fr.frame_bytes)
            if VF.is_link_setup(frame):
                if bodies:
                    break
                who = VF.caller_from_link_setup(frame)
                whose = ("our own transmission back through the receiver"
                         if who == self.caller else f"{who} opening a session")
                self.io.log(f"rx wideband link-setup naming {who} — {whose}, "
                            "not answering")
                return []
            body = bytes(fr.payload)
            if body in self._keyed_bodies:
                if bodies:
                    break
                self.io.log(f"rx wideband over: {hits}/{of} reference columns{rec} "
                            "and a body we keyed — our own transmission back through "
                            "the receiver, not answering")
                return []
            bodies.append(body)
            self.io.log(f"rx wideband frame: {hits}/{of} reference columns{rec}, "
                        "CRC clean, not a connect — a DATA over of this session"
                        + (f" ({len(bodies)} in this window)" if len(bodies) > 1
                           else ""))
            r = _RX2300.RECORDS[lv]
            x = _after_frame(x, mag, r)
            if len(x) < r.ncols * r.dw50:
                break
        return bodies

    def _peer_data_over_500(self, x: np.ndarray, secs: float) -> list[bytes]:
        """:meth:`_peer_data_over` at BW500, where one call is all four gates.

        ``decode_stream`` finds the burst inside the bracket by its own energy
        span, scores the record-2 reference columns to name the speed level,
        decodes every frame the span says the burst holds, and reports a frame it
        could not reach rather than dropping it  [rx.varahf500.DecodeResult]. A
        link-setup decodes CRC-clean and is not a data frame, so gate 3 is that
        classification and needs nothing here.

        A partial read is not delivered. When a CRC-clean non-echo frame
        identifies the peer window, its missing partner creates unread DATA
        debt rather than leaving the preceding positive ACK outstanding.

        The peer's first over has no CRC-clean anchor to identify it by and is
        the one window that needs none: nothing of ours has been keyed, so an
        all-bad burst there is not our echo  [see _first_over_owed]. It is
        reported as an over that would not decode, which owes the measured
        recovery rather than a token  [see _undecoded_over].
        """
        res = _RX500.decode_stream(x)
        got = res.data_frames
        if not got:
            self.io.log(f"rx {secs:.1f} s burst at BW500: no data frame in it — "
                        "not answering")
            return []
        if not res.complete:
            # A CRC-clean peer frame identifies an incomplete multi-frame
            # window. It cannot leave the previous positive ACK outstanding:
            # that ACK would let the sender discard the unread remainder.
            # Failed CRC/shape alone does not identify DATA; in particular our
            # own narrow emission can fail when bracketed at its energy edge.
            good = [bytes(f.payload) + bytes([f.marker])
                    for f in got if f.crc_ok]
            if (self.turn == _TURN_PEER and self._peer_delivery_open
                    and self._answer_owed == _OWED_OVER and good
                    and not any(b in self._keyed_bodies for b in good)):
                self.io.log("rx incomplete BW500 peer window with a CRC-clean "
                            "frame — unread DATA is owed, not a positive ACK")
                return [_UNDECODED]
            if not good and self._first_over_owed():
                self.io.log(f"rx {secs:.1f} s burst at BW500 that will not decode "
                            f"— {self.called}'s first over is owed and this "
                            "burst is not ours")
                return [_UNDECODED]
            if (not good and self._unread_over_follows_ack()
                    and got[0].burst_columns >= _UNREAD_OVER_MIN_COLS):
                self.io.log(f"rx {secs:.1f} s burst at BW500 that will not decode "
                            f"behind over #{self._peer_over}, which {self.called} "
                            "has taken our answer to — the unread over is owed, "
                            "not another acknowledgement")
                return [_UNREAD_OVER]
            self.io.log(f"rx {secs:.1f} s burst at BW500: {len(got)} frame(s) read "
                        f"of {got[0].frames_in_burst} the burst holds, "
                        f"{sum(1 for f in got if not f.crc_ok)} that will not "
                        "decode — not answering")
            return []
        bodies = [bytes(f.payload) + bytes([f.marker]) for f in got]
        if any(b in self._keyed_bodies for b in bodies):
            self.io.log(f"rx {secs:.1f} s burst at BW500 carries a body we keyed "
                        "— our own transmission back through the receiver, not "
                        "answering")
            return []
        self._peer_over_state = (got[-1].level, got[-1].marker)
        self.io.log(f"rx BW500 burst: {len(bodies)} frame(s) at level "
                    f"{got[0].level}, CRC clean, not a connect — a DATA over of "
                    "this session")
        return bodies

    def _tx_nak(self) -> bool:
        """Key the 8-symbol NAK for our end of the link, or nothing when none is
        measured for it. True when a burst reached the air.

        Role picks ours out of the link's pair, as the control bursts do
        [vara_frames, nak]: the receiver that failed the over keys it — the caller
        in RESPONDER, the responder in CALLER. No fallback to another link's tail:
        the NAK is measured for a W9SSJ-called link only, and this station always
        originates as W9SSJ, so the one case that would key a stranger's burst is
        one it never reaches.
        """
        pair = VF.nak(self.caller, self.bw)
        if pair is None:
            return False
        burst = pair[1] if self.role == "responder" else pair[0]
        self._key(True)
        try:
            self.io.tx(MK.synth_tone_pairs(burst))
            return self._tx_went_out()
        finally:
            self._key(False)

    def _owe_unread_over(self) -> bool:
        """An all-bad over behind one the peer has taken our answer to: the
        acknowledgement ladder stops, and the over is owed the BW500 recovery
        [see _unread_over_follows_ack].

        The debt replaces the ladder that was re-keying the acknowledgement: the
        over says the peer took it, and a rung keyed after it is what a stock
        sender reads as the acknowledgement of the over we did not read — 18
        bytes discarded unread on the cables of 2026-09-11. The new over's
        ladder starts from zero, keys nothing into its turnaround and asks
        behind the peer's idle as the first over does  [see _undecoded_over].
        """
        self._reacks = 0
        self._answer_owed = _OWED_OVER
        self._owed_block = True
        self._owed_recovery = True
        self.io.log(f"{self.called}'s over behind #{self._peer_over} will not "
                    "decode — the acknowledgement ladder stops here; keying "
                    "nothing into its turnaround and asking behind its idle")
        return True

    def _undecoded_over(self) -> bool:
        """Answer an over that would not decode where the bandwidth has an answer,
        and close the link once asking again cannot help
        [spec 02 §2.6 NAK, spec 05 §5.4].

        Always True: whatever this keys, the audio has been dealt with and the
        buffer that held it is spent.

        **BW2300's NAK is measured now** [vara_frames, nak]. The one this used to
        key came out of a 2026-07-23 table in no corpus and was refuted — it
        matched none of a real BW2300 session's control bursts and KC9GHZ did not
        repeat the over it went out on. The one keyed here instead is off two
        stock 4.9.0s on the cables, 2026-09-07: noise was injected into a
        receiver's input until it registered a burst and failed its CRC, and the
        burst it keyed in the turnaround is this one — four copies each side,
        identical across overs, caller and responder sharing only the lead pair.
        The sender answered it by dropping a speed level and re-sending, and both
        deliveries closed byte-exact, so the NAK draws the same recovery an
        unacknowledged over does and does not wait on the peer's own timeout. A
        link with no measured NAK still leaves the turnaround empty, where
        :meth:`_answer_data_over` records that a gateway repeats an over it was
        not acknowledged for.

        BW500 reaches this branch for an incomplete peer window identified
        by a CRC-clean non-echo DATA frame and for the peer's first over — the
        one window where an all-bad narrow burst cannot be anything of ours
        [see _first_over_owed]. For the first over the measured stock form
        keys nothing into the turnaround: the responder idles 1.5 s after its
        unkey and the ask goes out behind that idle as `session-over-nak`,
        which it answers by re-sending from its lowest speed level
        [see _stream_recovery_cue, _reack_over]. The NAK token keyed into the
        turnaround instead drew four idles and no resend (2026-09-11). A
        turnaround nothing was keyed into charges nothing here — the budget
        counts asks — and the ladder behind the idle is the bound, kept
        across resends that will not decode either: a stranger's wide session
        on the frequency puts three undecodable bursts into a greeting window
        and must close nothing. An all-bad over behind one we acknowledged is
        owed the same recovery  [see _owe_unread_over].

        What this replaces at BW500 is silence, and silence there is not neutral. On
        2026-08-18 this station identified a 4.8 s over at 18 of 24 reference
        columns, failed to decode it, transmitted nothing further and held the
        link — while the operator heard KC9GHZ go on transmitting into a shared
        40 m channel. A NAK is what the peer is listening for, it is what
        kestrel's own ARQ has keyed on a CRC failure since it had a receive
        engine [arq.fsm._nak_reset], and it costs 0.4 s of a turnaround that
        already belongs to the peer.

        The echo window is deliberately left open. Every other caller of
        :meth:`_close_echo_window` has positively recognised somebody's waveform
        first; a frame that will not decode names nobody, our own transmission
        back through the monitor included, so clearing the bodies we keyed on the
        strength of it would let the next echo reach the host as received mail.
        The turn is left alone for the same reason.
        """
        if self.bw == "500" and self._first_over_owed():
            self._answer_owed = _OWED_OVER
            self._owed_block = True
            self._owed_recovery = True
            self.io.log(f"{self.called}'s first over will not decode — keying "
                        "nothing into its turnaround; asking behind its idle "
                        f"({self._reacks}/{_REACK_MAX} asked)")
            return True
        self._undecoded += 1
        if self._undecoded > _OVER_NAK_MAX:
            self.io.log(f"{_OVER_NAK_MAX} turnarounds have gone by and the overs "
                        "still will not decode — closing the link rather than "
                        "leaving the peer transmitting to nobody")
            self.disconnect()
            return True
        self._answer_owed = _OWED_OVER
        self._owed_block = True
        if not (self._has_idle_over_nak() and self._owed_recovery):
            self._reacks = 0
        if self._has_idle_over_nak():
            if self._held_answer is not None:
                # A decoded prefix does not license an ACK of its unread
                # suffix. Cancel the positive answer, retaining the prefix for
                # duplicate suppression when the peer retransmits the window.
                self._held_answer = None
                self._held_samples = self._held_idles = self._held_idle_at = 0
                self._idle_pair_seen = False
                self._finish_delivery_window(complete=False)
            self._owed_recovery = True
            self.io.log(f"{self.called}'s BW2750 over will not decode — "
                        "retaining the unread window; asking for retransmission "
                        "behind its next idle")
            return True
        if self.bw == "500":
            if self._send_token("nak"):
                self._reacks += 1
                self.io.log(f"tx NAK ({self._undecoded}/{_OVER_NAK_MAX}) — asking "
                            "for the over again")
        elif self._tx_nak():
            self._reacks += 1
            self.io.log(f"tx NAK ({self._undecoded}/{_OVER_NAK_MAX}) — asking for "
                        "the over again")
        elif VF.nak(self.caller, self.bw) is not None:
            self.io.log("NAK transmission declined — unread window still owed")
        else:
            self.io.log(f"rx an over this station cannot decode "
                        f"({self._undecoded}/{_OVER_NAK_MAX}) — no NAK is measured "
                        f"for a link {self.caller} called at BW{self.bw}, so the "
                        "turnaround is left to the peer to repeat the over into")
        if self._reacks:
            self._keyed_on_peer_burst = True
            self._since_progress += 1
            self.idle_keyed += 1
        return True

    def _deliver(self, bodies: list[bytes], hold: bool = False) -> None:
        """Deliver a window in order, suppressing only a complete replay.

        Streamed frames arrive separately. If a new window starts like the last
        one, retain that prefix until it differs or ends: [A, B] followed by
        [A, C] must deliver A again, while [A, B] repeated must deliver neither.
        Order, length and multiplicity all matter. Distinct prefixes can reach
        the host immediately; only the ambiguous prefix needs buffering.

        A timed-out, partially delivered window is different: its known prefix
        already reached the host, and the NAK requests the missing remainder.
        Preserve that prefix when reading the retransmission.

        The host may enqueue a reply here. The peer-turn caller sets _answering
        so that the reply influences the acknowledgement without keying early.
        """
        repeat = self._a_repeat(bodies, hold)
        self._window_bodies += bodies
        window = tuple(self._window_bodies)
        prefix = window == self._last_window[:len(window)]
        if not self._last_window_complete:
            known = min(len(window), len(self._last_window))
            if window[:known] == self._last_window[:known]:
                self._window_delivered = max(self._window_delivered, known)
        if repeat:
            self.io.log("rx over repeats one already delivered — answered "
                        "again, not re-delivered")
        elif not (hold and prefix and self._last_window_complete):
            ready = window[self._window_delivered:]
            self._window_delivered = len(window)
            self._delivered = ready
            for body in ready:
                self._deliver_block(body)
        if not hold:
            self._finish_delivery_window()

    def _a_repeat(self, bodies: list[bytes], hold: bool = False) -> bool:
        """Only a whole ordered window can establish a replay."""
        return (not hold and bool(bodies)
                and tuple(self._window_bodies + bodies) == self._last_window)

    def _finish_delivery_window(self, complete: bool = True) -> None:
        """Retain the window, or its delivered prefix after a receive timeout."""
        if self._window_bodies and (complete or self._window_delivered):
            self._last_window = tuple(self._window_bodies)
            self._last_window_complete = complete
        self._window_bodies = []
        self._window_delivered = 0

    def _deliver_block(self, body: bytes) -> None:
        """Hand one block's payload to the host."""
        # The payload end is found by the whole trailer pattern, keyed to the
        # caller's callsign — exact for every over captured off air, and what
        # lets compressed mail carry a 0x14 without being cut at it. The real
        # length field (body[-2:-1]) is not reversed. See arq.phy.vara_payload.
        # The body's own length is the speed level's, which is why it is passed:
        # the trailer sits at the end of a 48-byte record-2 body, not at the end
        # of a base one.
        payload = _phy.vara_payload(body, caller=self.caller, body_len=len(body))
        self._progressed()                  # the one unambiguous kind of progress
        self.io.data(payload)
        self.io.log(f"delivered {len(payload)} payload bytes to the host")

    def _answer_data_over(self, samples, hold: bool = False) -> bool:
        """Identify a DATA over in this audio, deliver it, and answer it with ONE
        burst.

        False when the audio is not one, and then nothing has been keyed. Shared
        by the two routes that can hand over an over — the energy segmenter's
        bracket and :meth:`_stream_over` — so that which one found it cannot
        change what we say back.

        **One burst.** The turnaround an over opens holds a single frame. Across
        three complete VARA-to-VARA sessions in the loopback corpus — 120 keyings
        each, 51 DATA overs each, read off the capture harness's own PTT ledger —
        the two stations alternate at every one of the 118 changes of transmitter,
        median 88 ms apart. The only two consecutive keyings by one station in any
        of the three are an idle-cadence answer and, 1.2 s behind it, the
        turn-request that station keyed when its host injected a payload — with
        its peer nine seconds from transmitting. On air the gap is longer and the
        conclusion is the same: measured off the two BW2300 gateway sessions, a
        gateway keys its next over 0.27-0.29 s after our answer's last symbol
        (n = 3, both gateways), and a second frame here runs 1.4 s.

        **Which burst** says what we want, and the over says which of them is due.
        A full body has another over behind it and draws a continue-class frame; a
        short body closes the delivery and draws the control burst, which is what
        frees the turn  [see :func:`arq.phy.over_is_last`, :meth:`_tx_over_response`].
        Keying the control burst at both cost every multi-over delivery this
        station was ever sent: a stock responder handed 356 bytes released the
        turn with 267 still queued after its first over, and seven gateways
        stopped at exactly one over's worth.

        **EVERY over is acknowledged, and the ask waits.** A queue of our own used
        to replace the acknowledgement of a last over with a turn-request. Whether
        that frame also acknowledges the over is in no recording, and on
        2026-09-08 it did not: the peer's release follows our acknowledgement
        [vara_frames, SESSION_TURN_RELEASE_RESPONDER], so with none keyed no
        release came, the peer idled on its own cadence, and our asks went out on
        a 10 s clock that put two of the three across its transmissions. So the
        over draws the frame it asks for, the release is what the ask waits for,
        and the peer's own cadence is where it is asked from
        [see :meth:`_reack`, :meth:`_reack_release`].

        **The host is asked first, and keys nothing, where its answer decides.**
        :meth:`VaraIO.data` is not a sink: on the mail path it runs the Winlink
        client, which answers a greeting by calling :meth:`send` on this object
        before it returns. With the turn the peer's, what the host hands back is
        the whole difference between the two frames — so it is asked before the
        choice is made, and ``send`` keys nothing while it is being asked
        [see :meth:`send`]. Holding the turn there is nothing to ask it for: the
        answer is the per-over response whatever the host does, and the older
        order stands — answer, then deliver, then the host keys its own over into
        the turn we hold. Either way the echo window is closed ahead of the
        delivery: it is what tells our own outgoing mail from a gateway's, and an
        over forgotten in the same call that records it comes back through the
        monitor as received mail.
        """
        bodies = self._peer_data_over(samples)
        if not bodies:
            return False
        if bodies[0] is _UNDECODED:
            return self._undecoded_over()
        if bodies[0] is _UNREAD_OVER:
            return self._owe_unread_over()
        self._undecoded = 0
        self._close_echo_window()
        # Read before `_deliver` moves its duplicate record on. A lost ACK can
        # make the sender repeat a full window; that does not end its delivery.
        # Only the decoded body's framing chooses continue versus final ACK.
        # The over's LAST block is what says it, whether it carried one or two:
        # a short block closes the delivery and only the final one can be short.
        body = bodies[-1]
        repeat = self._a_repeat(bodies, hold)
        last = _phy.over_is_last(body, self.caller)
        if not repeat:
            self._peer_over += 1
        self._overs_hint = _phy.overs_after(_phy.over_field(body),
                                            first=not self._peer_delivery_open)
        if not repeat:
            self._peer_delivery_open = not last
        says = ("" if self._overs_hint is None else
                f" — the peer says {_delivery_stage(self._overs_hint)}")
        self.io.log(f"rx DATA over #{self._peer_over} "
                    f"({len(samples) / MK.FS:.1f} s) — answering{says}")
        if self.turn == _TURN_OURS:
            self._over_into_our_turn()          # may hand the turn back
        if self.turn != _TURN_PEER:
            self._answer_over(last, False, hold)
            self._deliver(bodies, hold)
            return True
        self._answering = True
        try:
            self._deliver(bodies, hold)
        finally:
            self._answering = False
        self._answer_over(last, last and bool(self._txq or self._tx_pending),
                          hold)
        return True

    def _answer_over(self, last: bool, owes_release: bool, hold: bool) -> None:
        """Key the answer this over drew, or hold it until the peer stops
        transmitting.

        **A STATION THAT IS TRANSMITTING IS NOT LISTENING**, and a stock sender
        does not key one over per transmission for long. Measured on the cables,
        2026-09-09: B's keyings run 4.38 s through a four-message fetch until the
        seventh over of the delivery, where it keys for 8.60 s and puts two overs
        inside one window. This station named the first as soon as it was whole
        and answered 0.16-0.24 s later, 4 s inside B's own key-down; the
        acknowledgement was lost to the peer's transmitter rather than to the
        channel, and B stopped with 1587 bytes still queued. It is deterministic:
        the same over, in both arms, with nothing injected.

        So the answer waits for the window, and what says the peer is still in
        one is the six columns behind the frame it just read, on that frame's own
        grid  [see _window_state]. The held answer goes out on the over that ends
        the window, which is where the peer is listening — its own next over is
        decoded, delivered, and takes the acknowledgement over. One window, one
        answer.

        Only the stream route holds — a bracket is handed over after it has
        closed, so the peer has already unkeyed and the answer is due; what a
        bracket needs instead is to be read for every over it holds
        [see _peer_data_over].
        """
        if hold:
            self._held_answer = (last, owes_release)
            self._held_samples = self._held_idles = self._held_idle_at = 0
            self._idle_pair_seen = False
            self._held_idle_early = self._idle_complete_exact = False
            self.io.log(f"{self.called} is still keying — holding the answer "
                        "for the end of its window")
            return
        self._key_over_answer(last, owes_release)

    def _key_over_answer(self, last: bool, owes_release: bool) -> None:
        """The answer itself, and what the peer owes behind it. Whatever was
        being held for this window went out as this, not behind it."""
        self._held_answer = None
        self._owed_block = False
        self._tx_over_response(last)
        if owes_release or (last and self.turn == _TURN_PEER):
            self._answer_owed = _OWED_RELEASE
            self.io.log(f"answered the last over — {self.called}'s release is "
                        f"owed, holding the ask for its cadence, "
                        f"{len(self._txq)} block(s) queued")

    def _release_held_answer(self, taken: int) -> None:
        """Give up on a held answer whose window never ended  [see _answer_over].

        A decoded next over can complete the window and replace its answer.
        On timeout the missing remainder instead needs retransmission. Retain
        the delivered prefix and request the window with NAK in the next gap;
        positively acknowledging half a window can discard its missing bytes
        at the sender before a later NAK has any chance to recover them.

        Unless the emission said what was in it. :meth:`_a_gap` will not key into
        a named idle — a six-column reading inside one emission is too unsteady
        to call a turnaround on — but an emission that spent this whole budget
        producing idles and no frame is the peer waiting for its answer, not a
        block this station failed to read. KC9GHZ held that state for 102 s on
        2026-08-20 [test_over_splice]: thirteen `session-responder-over-idle`
        keyings on a 3.6 s cadence and no repeat of anything. There the answer is
        what moves the delivery on, and withholding it stalls both stations.
        """
        if self._held_answer is None:
            return
        self._held_samples += taken
        if self._held_idle_early or self._idle_pair_seen:
            # A full early idle can disprove the window by itself [_a_gap].
            # Otherwise TWO IDLES A CADENCE APART were named under the hold, so they are two
            # EMISSIONS: a window's own second block is a 4.4 s DATA over and
            # cannot present as a pair of short idles. The peer has finished its
            # window and is waiting for our ACK, which goes out now rather than at
            # the `_ANSWER_HOLD_MAX` ceiling. The first idle alone can be the tail
            # of an over still mid-window, and two namings close together can be
            # one emission read twice, so the pair is judged at naming time
            # [see _a_gap, ``_IDLE_PAIR_MIN_S``]. Evidence: K0SI 40 m, three 683
            # idles 3.75 s apart, and NS0A, three 745s — an 11.6 s late ACK in both
            # [docs/protocols/vara/20-peer-turn-measurements.md].
            (last, owes_release), self._held_answer = self._held_answer, None
            if self._held_idle_early:
                self.io.log(f"complete early idle from {self.called} — no second "
                            "DATA block fits before it; answering the held over")
                self._held_idle_early = False
            else:
                self.io.log(f"{self.called} keyed {self._held_idles} idles a cadence "
                            "apart under the held answer and no second over — its "
                            "window has ended, answering it now")
            # The window closes with the answer, exactly as it does at the ceiling:
            # left open, its bodies join the NEXT over's window and a peer repeat
            # of that over is delivered to the host twice  [see _finish_delivery_window].
            self._finish_delivery_window()
            self._key_over_answer(last, owes_release)
            return
        if self._held_samples < _ANSWER_HOLD_MAX:
            return
        (last, owes_release), self._held_answer = self._held_answer, None
        self._finish_delivery_window(complete=False)
        if self._held_idles:
            self.io.log(f"{self.called} has been keying for "
                        f"{self._held_samples / MK.FS:.1f} s with "
                        f"{self._held_idles} idle(s) and no second over in it — "
                        "answering the window it is waiting on")
            self._key_over_answer(last, owes_release)
            return
        self._owed_block = True
        self._answer_owed = _OWED_OVER
        self._reacks = 0
        self.io.log(f"{self.called} has been keying for "
                    f"{self._held_samples / MK.FS:.1f} s with no second over in "
                    "it — retaining the unread window for retransmission; "
                    "no positive acknowledgement sent")

    def on_rx_audio(self, samples: np.ndarray) -> None:
        """Read one received burst, then key anything its turnaround owes."""
        self._read_burst(samples)
        self._ask_if_owed()

    def _read_burst(self, samples: np.ndarray) -> None:
        """Demodulate one received MFSK burst and advance the handshake.

        Over a real audio path burst segmentation is a couple of symbols
        imprecise, so when we are waiting for a specific handshake burst we demod
        at that burst's known symbol count if the measured length is close —
        rather than dropping it on an exact-length miss.
        """
        # The graceful close is over the moment the burst is on the air
        # [see disconnect], so this state is only ever the one a close the
        # transport declined leaves behind — and the recovery for that is
        # re-keying it, not answering the peer. Nothing is keyed into a link we
        # have already said goodbye on.
        if self.state == VaraState.DISCONNECTING and self.role == "initiator":
            return

        # Step 5: the connected-ack is 11 two-tone symbols opening on a fixed
        # preamble  [spec 04 §4.2C] — not a member of the MFSK burst family, so it
        # is recognised here, ahead of the sizing that would try to read it as one.
        # This branch used to accept ANY audio as the ack: ten zero samples, an
        # empty array, a carrier from someone tuning up, our own link-setup tail
        # arriving late off the segmenter. On a live band the segmenter supplies
        # such a trigger almost immediately, so the connect could not fail, and
        # "CONNECTED" carried no evidence the gateway had answered at all.
        if self.role == "initiator" and self.step == _I_LINKSETUP_SENT:
            if self._ack_evidence(samples):
                self.io.log("rx connected-ack (preamble confirmed) — CONNECTED")
                self._connected()
                return
            # Deliberately NOT a return. This branch used to swallow every burst that
            # arrived in this step, so one missed ack was terminal even while the
            # gateway was still talking: its repeated connect-response — which means
            # it did not hear our link-setup — could never be recognised.

        # Step 4 (responder): after our connect-response, the next wideband burst
        # should be the initiator's link-setup naming the caller. Only a decode
        # completes the connect; over-length audio that is not a link-setup is
        # waited out rather than acked  [spec 05 §5.3.2].
        if (self.role == "responder" and self.state == VaraState.CONNECTING
                and self.step == _R_RESP_SENT and len(samples) >= _DATA_OVER_MIN):
            self._rx_link_setup(samples)
            return

        # Data phase: a gateway DATA over is a wideband OFDM burst, far longer than
        # any MFSK burst. An initiator answers each one with the per-over response
        # (spec 05 §5.3.3) — a responder uses the short DBPSK data-ack instead, so
        # this is deliberately initiator-only. Without an answer the gateway stops.
        # Length is the cheap precondition and _peer_data_over is the decision:
        # answering on length alone keys the transmitter at whatever the segmenter
        # happened to bracket. Not a return — a piece that is not an over may still
        # be a handshake burst, and used to be swallowed here.
        #
        # This is the bracket route, kept for transports that only bracket (the
        # loopback is one). Where the receive stream is fed, _stream_over owns the
        # burst instead and both routes would answer the same over twice — one
        # wasted keying, and one the peer reads as a station transmitting out of
        # turn.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and not self._stream_owns_over
                and len(samples) >= _DATA_OVER_MIN
                and self._answer_data_over(samples)):
            return

        # The gateway granting the turn we asked for. First of the connected-state
        # branches because it is the only one that reads a burst longer than an
        # over, and because everything below would decline it on length alone —
        # which is what a gate riding the band noise open cost this station on
        # 2026-08-22, twice in one session  [see _peer_drained].
        #
        # The length guard is `_stream_grant`'s, for its reason: `_peer_drained`
        # carries none of its own, so a bracket too short to hold the frame can
        # take a grant out of a 32-symbol answer still arriving and key the over
        # across the peer's tail.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.turn == _TURN_ASKED and len(samples) >= _SESSION_NEED
                and self._peer_drained(samples)):
            self._close_echo_window()
            self._turn_granted(samples, VF.SESSION_DRAINED_RESPONDER.name)
            return

        # The gateway handing the channel over unasked, at the end of its own
        # delivery. `_stream_grant` owns this frame on the wide bandwidths and
        # only between a request of ours and its answer; at BW500 the stream route
        # owns the over and nothing shorter, so without this branch the release
        # reaches nothing
        # and the log says "rx burst with 17 symbols — not an MFSK handshake
        # burst". Measured against a stock 4.9.0 on 2026-09-02: the responder
        # answered our control burst at its last over with
        # `session-turn-release-responder`, 17 of 17 carriers keyed to the station
        # we dialled, and then said nothing for 59 s while this station keyed
        # keepalives at a peer that had already given it the channel.
        #
        # The guard is a WINDOW round this frame's own span, the way
        # `_peer_wants_turn` bounds its own: a floor because the release is 17
        # symbols where the 32-symbol frames are twice that, so `_SESSION_NEED`
        # would decline a bracket holding the whole of it — and a ceiling because
        # the recogniser carries none and sweeping a six-second bracket for a
        # 0.73 s payload finds it in band noise. Unbounded it took six windows of
        # the 2026-08-06 KC9GHZ recording that hold no gateway over at all.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.turn == _TURN_PEER
                and _SESSION_MIN <= len(samples) <= _SESSION_MIN + 6 * MK.HOP
                and self._peer_responder_release(samples)):
            self._took_responder_release()
            return

        # The same handover at 32 symbols. Stock 4.9.0 keyed the release above at
        # this position in the BW500 run of 2026-09-11 06:37z and
        # `session-drained-responder` at it in the 13:35z rerun, both arms, every
        # ~11.4 s until something took it; the branch above declines it on length
        # and the grant branch above that on the turn, so it reached the release
        # ladder instead and each arm paid 25.9 s of it
        # [analysis/stock500-chain2]. What makes it the handover rather than the
        # peer's own cadence is this end's state, and that is `_drained_hands_over`'s
        # [see _took_drained_handover]. `_stream_answer` names it first at the wide
        # bandwidths, where a gateway's short burst never opens the gate that
        # feeds this route at all; here it is BW500's and any transport that only
        # brackets.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.turn == _TURN_PEER
                and self._drained_hands_over(samples)):
            self._took_drained_handover()
            return

        # The same peer polling with its control burst after our release. The
        # stream route names it first where it runs; here it is the bracket
        # route's, so a transport that only brackets answers it too, and the
        # window is the frame's own span for the release's reason above.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.turn == _TURN_PEER and self._released and not self._txq
                and not self._stream_owns_answer
                and _POLL_MIN <= len(samples) <= _POLL_MIN + 6 * MK.HOP
                and self._peer_control_burst(samples)):
            self._took_poll()
            return

        # The gateway answering an idle frame of ours. Named before the positional
        # branch below can guess at it, because this one is an identification: 31
        # tones keyed to the call we dialled, not a length and a turnaround.
        #
        # What it settles is that the peer is listening, and while the turn is ours
        # that is the precondition for the next over — so a queue drains on it
        # exactly as it drains on the peer's control burst. What it does NOT settle
        # is the turn itself, and it is deliberately not wired to _turn_granted: no
        # recording holds it answering a turn-request, and what those requests drew
        # is the branch above  [vara_frames, SESSION_IDLE_RESPONSE].
        #
        # An empty queue keys nothing, and gives nothing back either. The real
        # VARA that recorded these sessions took this answer and said nothing for
        # 11.7 s, and that silence is a station which had ALREADY RELEASED THE
        # TURN: the consequence, not the act. Reading it as "do not key this
        # instant" is what left the 2026-08-22 session holding a turn it could not
        # spend and probing for an answer the gateway had no way to give. The
        # release is the idle cadence's [see idle_keepalive], which reads the queue
        # once a tick rather than once per burst the peer chooses to answer with.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self._peer_idle_response(samples)):
            self._took_idle_response()
            return

        # The gateway's own idle cadence, which asks for nothing and is answered
        # only when it owes us something. `_stream_answer` owns it where the raw
        # stream is fed; here it is the bracket route's, and without this branch a
        # transport that only brackets logs every one of them as "not an MFSK
        # handshake burst" — which is what the loopback did through a whole
        # delivery the peer had stopped mid-way.
        #
        # THE BRACKET IS THE TURNAROUND. On the stream the frame's age decides
        # whether the peer is still listening; a bracket closes six quiet frames
        # after the last keyed sample, so a named idle in one is a gap that has
        # just opened  [see _ask_if_owed]. The window is the frame's own span, for
        # the release branch's reason above: this recogniser carries no length
        # guard and a six-second sweep finds a 32-symbol payload in band noise.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and not self._stream_owns_answer
                and _SESSION_NEED <= len(samples) <= _SESSION_NEED + 6 * MK.HOP
                and self._peer_responder_idle(samples)):
            self.io.log(f"rx {self._idle_kind.name} — {self.called} is still "
                        "transmitting (not progress)")
            if self._a_gap():
                self._reack()
            return

        # Bracket-only transports retain this route. The raw reader waits for
        # the fitted frame's complete tail and guard before taking the request;
        # it then owns this route to avoid acting twice on one received burst.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and not self._stream_owns_turn_request
                and self._peer_wants_turn(samples)):
            self._took_turn_request()
            return

        if (not self._stream_owns_answer
                and self._held_answer is None
                and self._peer_data_nak(samples)):
            self._took_data_nak()
            return

        # BW500 has no wideband answer scanner. Native stock can report a
        # failed lowest-record retry through the queried 288/1117 missing-DATA
        # answer instead of a direct CRC NACK. Keep this bracket route bound to
        # a solicited full lowest record; its existing retry preserves the rate.
        if (self.bw == "500" and not self._stream_owns_answer
                and self._held_answer is None
                and self._intermediate_answer_candidate()
                and self._intermediate_query_fresh()
                and self._tx_pending[2] == 0
                and self._peer_responder_nak(samples)):
            self._took_responder_nak()
            return

        if (not self._stream_owns_answer
                and self._held_answer is None
                and self._peer_intermediate_query_answer(samples)):
            self._took_intermediate_query_answer()
            return

        # The same frame arriving BEFORE we consider ourselves connected is the
        # peer telling us the link is up: it is keyed only on a session, and a
        # station with a queue and no channel is what a gateway is the instant it
        # answers a call. A stock caller reads it that way — keyed into the
        # turnaround behind its link setup it confirms the connect as often as the
        # connected-ack itself does (5 of 10 against 5 of 11), where the release
        # keyed into the same slot confirms nothing (0 of 9) and an empty slot
        # leaves it keying link setups to its own timeout.
        #
        # Which is what AJ4GU spent four keyings saying on 2026-08-29 — 72.857,
        # 96.618, 106.667 and 111.688 s, up to 32 of 32 tones — while this station
        # resent link setups and then fell back to connect requests at a gateway
        # that had already connected it.
        if (self.state == VaraState.CONNECTING and self.role == "initiator"
                and self.step == _I_LINKSETUP_SENT
                and not self._stream_owns_connect_ask
                and self._peer_wants_turn(samples)):
            self.io.log(f"rx turn-request from {self.called} before the "
                        "connected-ack — the link is up")
            self._connected(confirm=False)
            self._took_turn_request()
            return

        # The peer asking for the next over of a delivery of ours. The 8-symbol
        # burst, which the branch below cannot take: it carries only the lead pair
        # of the 11-symbol frame's four-pair preamble, and its own recogniser is
        # the one that reads it. `_stream_answer` owns it on the wide bandwidths
        # and the stream route does not run at BW500, so without this a delivery
        # of ours waits out the idle cadence between overs — 12 s where a stock
        # pair turns round in half of one, measured against a stock 4.9.0 on
        # 2026-09-02, and by the second over the peer had stopped reading.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and (self.turn == _TURN_OURS or self._stalled())
                and not self._stream_owns_answer
                and self._peer_nak(samples)):
            self._took_nak()
            return
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and (self.turn == _TURN_OURS or self._stalled())
                and not self._stream_owns_answer
                and self._peer_over_continue(samples)):
            self.io.log(f"rx over-continue from {self.called} — the over was "
                        "heard and the next one is asked for")
            if self.turn == _TURN_OURS:
                self._took_over_continue()
            else:
                self._took_stall_answer()
            return

        # Our turn-request, or one of our own DATA overs, has been answered. Any
        # burst the peer keys in the turnaround is that answer: it is far shorter
        # than an over, and the peer transmits nothing else while we hold the turn.
        # What it says is read for the log by _turn_granted; what it settles is
        # that the peer is listening, which is the whole precondition for keying.
        if (self.state == VaraState.CONNECTED and self.role == "initiator"
                and self.turn != _TURN_PEER and self._peer_control_burst(samples)):
            if self.turn == _TURN_ASKED:
                self._close_echo_window()
                self._turn_granted(samples)
            else:
                self._took_control_burst()
            return

        # Lock onto the burst we are waiting for by its fixed preamble, rather than
        # trusting where the envelope segmenter thought the burst began. Measured on
        # real gateway audio: the segmenter handed us a 0.11 s fragment 176 ms before
        # a real gateway connect-response, and demodulating that fragment from sample 0
        # matched 0/8 preamble tones — which the log reports as the gateway not
        # answering. Locking first gives 8/8 and the response is recognised.
        exp = self._expected_kind()
        if exp is VF.connect_response(self.bw) and self._stream_owns_response:
            exp = None                  # found on the stream or not at all
        if exp is not None and len(exp.preamble) >= 2:
            at = MK.lock_preamble(samples, exp, band=self._band)
            if at is not None:
                n_sym = len(exp.preamble) + exp.n_payload
                need = (n_sym - 1) * MK.HOP + MK.STRIDE
                seg = samples[at:]
                if len(seg) < need:
                    seg = np.concatenate([seg, np.zeros(need - len(seg))])
                self.io.log(f"rx {exp.name} preamble locked at +{at} samples")
                self.on_rx_tones(MK.demod_tones(seg, n_sym, self._band), exp)
                return
            # No preamble to lock. A connect-response can still be found by its
            # payload, which is what recovers an answer that overlapped our own
            # transmission; nothing else here can be, so nothing else tries.
            if exp is VF.connect_response(self.bw):
                heard = self._response_by_payload(samples)
                if heard is not None:
                    self.on_rx_tones(heard, exp)
                    return

        n = len(samples)
        n_sym = max(1, round((n - MK.STRIDE) / MK.HOP) + 1)
        # NAMED BY RECOGNITION, NOT BY LENGTH. A bracket is the keyed region plus
        # the gate's pre-roll ahead of it and its hangover behind — ten frames of
        # 1024 samples between them, which is exactly five symbols — so dividing
        # its length puts every session frame five symbols long. Live on the bench
        # of 2026-08-26 a peer's 32-symbol frames arrived as 37, a count `_NSYM`
        # has no key for, and every turn grant and turn-request of the session was
        # logged "not an MFSK handshake burst".
        #
        # Trimming the padding away was the wrong half to fix: the pre-roll is the
        # head of a burst the gate opened late on, and the hangover is the
        # trailing window `demod_tones` needs for the last symbol of a burst that
        # ends at the bracket's edge — without it a real VARA's connect-response
        # stops being readable and kestrel's own loopback stops connecting. So the
        # padding stays and the length stops being the name: each kind this
        # bandwidth can carry is demodulated at each start the gate could have
        # opened at, and the one the CALLSIGN takes is the burst. That is strictly
        # narrower than naming by length and recognising afterwards, which is what
        # let a 0.7 s noise swell be logged as a session-confirm.
        found = None
        for off in range(0, 2 * MK.HOP + 1, MK.HOP // 2):
            tail = samples[off:]
            if len(tail) < MK.STRIDE:
                break
            for want, cands in _candidates_for(self.bw).items():
                heard = MK.demod_tones(_padded(tail, want), want, self._band)
                if any(self._matches(heard, cs, c) for c in cands
                       for cs in (self.called, *self.mycalls) if cs):
                    found = (want, off)
                    break
            if found is not None:
                break
        if found is not None:
            n_sym, at = found
            samples = samples[at:]
        kind = _kind_for(n_sym, self.bw)
        if kind is None and exp is not None:
            en = len(exp.preamble) + exp.n_payload
            if abs(n_sym - en) <= 3:                     # segmentation slop
                kind, n_sym = exp, en
        if kind is None:
            self.io.log(f"rx burst with {n_sym} symbols — not an MFSK handshake burst")
            return
        if kind is VF.connect_response(self.bw) and self._stream_owns_response:
            return                      # the same answer, arriving the slow way
        self.on_rx_tones(MK.demod_tones(_padded(samples, n_sym), n_sym,
                                        self._band), kind)

    def on_rx_tones(self, tones: Sequence[int], kind: VF.BurstKind) -> None:
        """Advance the handshake given a demodulated burst of ``kind``."""
        # -- responder: inbound CR keyed to one of MYCALL (step 1) ----------
        if kind is VF.connect_request(self.bw) and self.state == VaraState.LISTENING:
            call, m, nn = VF.best_match(tones, self.mycalls,
                                       VF.for_bw(kind, self.bw))
            if self._matches(tones, call, kind):
                self.role = "responder"
                self.called = call                   # CR is keyed to the CALLED
                self.state = VaraState.CONNECTING
                self.io.pending()                    # PENDING [spec 05 §5.2]
                self.io.log(f"rx CR matched MYCALL={call} ({m}/{nn})")
                self._send_burst(VF.connect_response(self.bw))   # step 2
                self.step = _R_RESP_SENT
                if self.mfsk_only:
                    # kestrel<->kestrel MFSK-only: no link-setup will come, so
                    # ack now and report the caller unlearned  [spec 05 §5.3.2].
                    self.caller = CALLER_UNKNOWN
                    if self._tx_connected_ack():     # step 5
                        self.state = VaraState.CONNECTED
                        self.io.connected(self.caller, self.called, self.bw)
                # Otherwise stay CONNECTING: the ack answers the link-setup that
                # names the caller (step 4), which on_rx_audio decodes next.
            else:
                self.io.log(f"rx CR not for MYCALL (best {call} {m}/{nn}) — ignored")
            return

        # -- responder: an initiator that did not hear our connect-response
        # repeats its CR; answering the repeat is the only way this connect
        # recovers, exactly as the initiator treats a repeated connect-response.
        if (kind is VF.connect_request(self.bw) and self.role == "responder"
                and self.state == VaraState.CONNECTING
                and self.step == _R_RESP_SENT):
            if self._matches(tones, self.called, kind):
                self.io.log(f"rx repeated CR for {self.called} — resending "
                            "connect-response")
                self._send_burst(VF.connect_response(self.bw))
            else:
                self.io.log("rx CR while awaiting the link-setup, not keyed to "
                            f"{self.called} — ignored")
            return

        # -- initiator: connect-response keyed to the gateway (step 2) ------
        # Also accepted after we have already sent the link-setup: a gateway that
        # did not hear it repeats this burst, and answering the repeat is the only
        # way an attempt recovers from a link-setup lost to the channel.
        if (kind is VF.connect_response(self.bw) and self.role == "initiator"
                and self.step in (_I_CR_SENT, _I_LINKSETUP_SENT)):
            level = next((level for level in (4, 3, 2, 1)
                          if self._matches(tones, self.called,
                                           VF.connect_response(self.bw, level))), None)
            if level is None:
                self.io.log("rx connect-response did NOT match dialled call — ignored")
            else:
                self._accept_connect_response(level)
            return

        # -- responder: the initiator's disconnect-request ends the session, and
        # nothing goes back  [see disconnect]. The acknowledgement this used to
        # key, and the final it waited for behind it, are a reading off a loopback
        # tape: on the 2026-08-26 bench the closing station keys the request once
        # and the only thing that follows on either cable is the peer's own ident.
        # It shares the keepalives' symbol count, so it arrives here labelled as
        # one and the tones decide; anything else at that count falls through to
        # the keepalive branch below.
        if (self.role == "responder"
                and kind in (VF.SESSION_KEEPALIVE_A, VF.SESSION_DISCONNECT_REQ)
                and self.state in (VaraState.CONNECTED, VaraState.DISCONNECTING)
                and self._matches(tones, self.called, VF.SESSION_DISCONNECT_REQ)):
            self.io.log("rx disconnect-request — session closed")
            self.state = VaraState.DISCONNECTED
            return

        # -- responder: the initiator has handed the channel back. Nothing is
        # owed on the air — what the release settles is that this end may key an
        # over now  [vara_frames, SESSION_TURN_RELEASE].
        if (kind is VF.SESSION_TURN_RELEASE and self.role == "responder"
                and self.state == VaraState.CONNECTED):
            if self._matches(tones, self.called, kind):
                self.io.log(f"rx turn-release — {self.caller} has handed the "
                            "channel back")
            return

        # -- responder: the initiator's post-connect session bursts (step 6 confirm
        # and the idle keepalives). VARA answers each with its connect-response
        # burst  [spec 05 §5.3.3].
        if (kind in (VF.SESSION_CONFIRM, VF.SESSION_KEEPALIVE_A)
                and self.state == VaraState.CONNECTED and self.role == "responder"):
            alt = (VF.SESSION_KEEPALIVE_B if kind is VF.SESSION_KEEPALIVE_A else None)
            if (self._matches(tones, self.called, kind)
                    or (alt is not None and self._matches(tones, self.called, alt))):
                self.io.log(f"rx {kind.name} for {self.called} — answering")
                self._send_burst(VF.connect_response(self.bw))
            else:
                self.io.log(f"rx {kind.name} not keyed to {self.called} — ignored")
            return

        # -- initiator: the peer's answer to our confirm/keepalive is its ordinary
        # connect-response burst; expected on a live link, not an error.
        if (kind is VF.CONNECT_RESPONSE and self.state == VaraState.CONNECTED
                and self.role == "initiator"):
            self.io.log("rx session answer (connect-response) — link alive")
            return

        # ``kind`` may be nothing but a symbol count. on_rx_audio names a bracket by
        # its length when no preamble locked, and _NSYM has no tone in it: measured
        # over the 2026-08-06 call to KD9USW, a band-noise swell bracketed 0.70 s
        # long arrived here as a "session-confirm" and was logged as one — a frame
        # name, a role and a rejection, off a duration. Every branch above earns its
        # name with VF.recognize before acting on it; this one has to earn its name
        # before reporting it, or the log reads as an answer this station refused.
        if not any(self._matches(tones, cs, kind)
                   for cs in (self.called, *self.mycalls) if cs):
            self.io.log(f"rx burst with {len(tones)} symbols — not an MFSK "
                        "handshake burst")
            return
        self.io.log(f"rx {kind.name} unexpected in state={self.state} "
                    f"role={self.role} step={self.step} — ignored")
