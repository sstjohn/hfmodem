# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Test-time access to the material the suite is validated against but does not ship.

Two kinds of thing live outside the package and stay there:

  * the **corpora** — real captured audio, tens of megabytes each, git-ignored.
    They are the receiver's evidence, not its code.
  * the **harness** — ``oracle/`` drives a reference modem, ``tools/`` holds the
    probes, ``analysis/`` the offline segmenters. Working tools, not shipped code.

Every test that reads either declares it here, and skips when it is absent. That
is what makes a run from an installed wheel meaningful: everything that needs
neither runs, and the rest is reported as skipped rather than as an error. On a
source tree the same skip is a silent loss of coverage, so ``RF_CORPUS_RECORDINGS``
below is checked there instead — see ``tests/gates/test_corpus_present.py``.

``KESTREL_CORPUS`` points the search at a source tree, for an installed wheel that
does have the captures to hand.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from hfmodem.tests import evidence

PKG_ROOT = Path(__file__).resolve().parents[3]

ROOT = Path(os.environ.get("KESTREL_CORPUS") or evidence.WORKING / "vara")
TOOLS = evidence.TREE / "tools"

#: Where a checkout's own tables are. The gateway list read from here is the one
#: `kestrel_connect.GATEWAY_LIST` dials from, so a floor measured against "every
#: published gateway" is measured against the panel the station can actually call.
REPO = evidence.RECORD

#: Whether anyone has asserted where the logged sessions are. The default above is
#: the record of the merged-from tree and is the right place to look, but the
#: sessions themselves are 92 MB apiece and the logged-audio directory is in no
#: clone — so on the default the BW500 arbiters skip, and nothing said so. They
#: are reached by
#: pointing `KESTREL_CORPUS` at the archive that holds them, which is a path
#: private to one machine and therefore belongs in `tools/arbiters.env` beside
#: `PMON_ORACLE` rather than written down here: shipping files may not name the
#: repositories this was merged from (`gates/test_import_direction.py`).
#:
#: Set, it is a claim the gate holds to account; unset, it is an absence the gate
#: reports. Both beat a silent skip.
ROOT_DECLARED = bool(os.environ.get("KESTREL_CORPUS"))

BW2300_CAPTURE = (ROOT / "analysis" / "caps" / "bw2300"
                  / "AAAA1-BBBB2-bw2300-counter512__a2b.wav")

# The only loopback capture of a BW2750 session, 2026-07-21. Its one wideband OFDM
# burst is the link-setup, whose frame is known in advance — which is what makes it
# the arbiter of BW2750's base record. A second, unlogged session 50 s later on the
# same tape is BW2300, the control on the same modems and callsigns.
#
# READ IT RAW FROM BYTE 44. The harness died before closing the file, so the RIFF
# data-size field is 0 and every wav reader returns an empty array; the 4,177,920
# float32 samples are there.
BW2750_CAPTURE = (ROOT / "analysis" / "caps" / "bw2750"
                  / "20260721-161424-AAAA1-BBBB2-bw2750-counter512"
                  / "AAAA1-BBBB2-bw2750-counter512__a2b.wav")
# The 2026-09-04 gear-down ladder. A stock pair walked down the speed levels by
# band-limited noise on the *caller's* input -- the transmitter's level follows its
# receiver's feedback, so the overs that drop a gear are the responder's -- with the
# noise cut mid-session so the record it dropped to is also on tape clean, which the
# gear-shift's own hysteresis leaves keyed for another over or two. It is the only
# recording here that holds host `BITRATE (1)` and `BITRATE (2)`, records 0 and 1,
# and every over in its manifest is stamped with the level its host announced.
BW2300_LADDER = ROOT / "analysis" / "caps" / "bw2300" / "20260904-ladder-release"

# A second ladder run the same night, driven with no release at all. The noise it
# was driven by is the harness's own -- a seeded buffer, sample-aligned on the
# cable that records it -- so it comes back off the tape and every over of every
# session is readable, not only the two or three a quiet window held. The two
# records' bin tables were not measured from this one, which is what makes it the
# check on them: six sessions, all byte-exact, 78 record-0 and 11 record-1 overs.
BW2300_LADDER_REFS = ROOT / "analysis" / "caps" / "bw2300" / "20260904-ladder-refs"

BW500_CAPTURES = ROOT / "analysis" / "bw500_close3_2026-07-15"

# Off-air sessions with a live Winlink gateway: <call>_<bandwidth>/rig_rx.wav is
# the RF as received, payload.bin the plaintext VARA delivered on its host port
# during that same session — independent ground truth for the receive chain.
OFFAIR = ROOT / "offair"
GATEWAY_SESSION = OFFAIR / "KC9GHZ_2300"

# 30 s of an empty 40 m frequency through the same rig and codec, and the only
# capture of that night whose envelope never collapses (3.1 dB span end to end),
# so the station's other modem was idle throughout and this is band noise rather
# than our own receiver being muted. It is what the channel-occupancy thresholds
# are measured against on the clear side; captures made minutes earlier and later
# are contaminated and are deliberately not kept.
CLEAR_CHANNEL = OFFAIR / "clear_7101800" / "rig_rx.wav"

# The off-air recordings that are not this repo's own captures live in the shared
# corpus beside it. 150 s recorded through the rig while kestrel called KB9MMT on
# 2026-07-26: eight of our own connect-requests, each heard at -33 dBFS through
# our own muted receiver, against a band noise of -12. It is the segmenter's
# worst case — a 2.1 s dropout 35 dB down every 8.3 s — and the recording the
# noise-floor estimator is measured on.
#
# It is also the first gateway answer addressed to KESTREL rather than to VARA,
# and the only recording we hold whose in-band power sits just 9.0 dB above the
# out-of-band power (19.1 dB for NS0A_2300, 15.5 dB for KC9GHZ_2300) — enough
# out-of-band energy to catch a tone search that forgot to band-limit itself.
#: `HFMODEM_CORPUS` was documented in conftest.py, in pyproject.toml and in
#: ARCHITECTURE.md while three different variables did the actual work — so a
#: reader who set the documented one saw no change and had no way to tell.
RF_CORPUS = evidence.CORPUS
ONAIR_CONNECT_ATTEMPT = RF_CORPUS / "offair" / "kb9mmt_reply_20260726" / "rig_rx.wav"

# The published gateway panel a negative is scanned against, fetched at run time
# and therefore in no clone -- so it resolves against the checkout the recordings
# are in, exactly as they do, and a worktree reads the one its origin fetched.
VARA_GATEWAY_PANEL = evidence.RECORD / "winlink-vara-gateways.csv"

# Where a tool is RUN from, as against where evidence is READ from. A subprocess
# under test has to start in the tree under test; a recording has to come from the
# checkout that holds it. They are the same directory in the main tree and not in a
# worktree, which is why one name for both went wrong.
REPO = evidence.TREE

# The shared regression corpus beside it: 31 real off-air recordings kept for the
# regressions each one catches — four more VARA sessions, PACTOR-1/2/3, ARDOP,
# FT8, WSPR and band noise from four continents, some through a rig and some
# through a websdr at 11999 Hz. None of it is addressed to us, which is what makes
# it the population a wide search measures its false-accept floor against.
REGRESS_FIXTURES = RF_CORPUS / "regress" / "fixtures"

# The first recording of a real gateway answering a connect-request of OUR OWN, and
# with it the reason every such answer had been thrown away. Calling KC9GHZ on
# 7103.5 kHz on 2026-08-06, the gateway replied to three of our eight requests; a
# KiwiSDR witnessing the same minutes reads those replies 10/15 to 14/15, and this
# station reads 9 of 15 because the other six of each reply's payload tones arrive
# inside its own post-transmit mute. `kiwi_witness.wav` beside it is those same
# minutes heard from Michigan, which is how the transmit side was cleared: our own
# requests read 41/41 tones there.
ONAIR_GATEWAY_ANSWER = RF_CORPUS / "offair" / "kc9ghz_answer_20260806" / "rig_rx.wav"

# The other two calls of that same evening, through the same receiver, that nobody
# answered — 141 s of 40 m and 80 m band noise, one of them with a narrowband signal
# parked on the channel centre for half the run. They are the negative controls for
# both stream searches, and they are named here rather than derived at the point of
# use: two tests read them while declaring a dependency on `rig_rx.wav` above, so
# their own `assert p.exists()` sat behind a skipif on a different file.
ONAIR_SILENT_CALLS = tuple(ONAIR_GATEWAY_ANSWER.with_name(n)
                           for n in ("silent_kd9usw.wav", "silent_w8mw.wav"))

# Forty minutes later, the same gateway, and the first VARA link this station ever
# originated: 227 s of receive audio across a whole session with KC9GHZ, CONNECTED
# at 57 s. It holds exactly one thing the gateway transmitted after the
# connected-ack — its Winlink greeting, as an ordinary BW2300 DATA over at
# 58.083-62.307 s, 24 of 24 reference columns and a clean CRC — and the mail
# client never left "awaiting greeting", because that over is 4.1 dB over the band
# noise and no energy gate opens on it.
#
# It is therefore the recording the DATA-over stream search is measured on, both
# ways: the greeting is the one positive, and the 164 s behind it are 40 m that
# must not key the transmitter. Our own twenty transmissions are in it too, read
# back at 31/31 and 15/15 tones through the rig's own monitor.
ONAIR_GATEWAY_GREETING = (RF_CORPUS / "offair" / "kc9ghz_connect_20260806"
                          / "rig_rx.wav")

# Those 164 s were called band noise until 2026-08-19, and two of them are a third
# station's traffic: a client working down its gateway list, two connect-requests
# 3.6 s apart to two different gateways, at a moment this station's receiver is
# live throughout and its own attempt has been over for 97 s.
#
# They are the only VARA a station other than this one or its peer has left on any
# recording here, which makes them the corpus's specimen of the case a false-accept
# floor is most easily wrong about: a burst that is real, addressed to somebody
# else, and named by a callsign the floor scores as wrong. Each stands ~20 tones
# clear of the next-best of the 340 published gateway callsigns, so what they are
# is not a matter of opinion.
#
#   159.905-161.654 s  CR to N3HYM-10, on frequency, 32 of 41 tones (7/10 preamble),
#                      -15.3 dBFS against a band of -16.8; runner-up 12 of 41
#   163.523-165.273 s  CR to W9FE, three carriers high (+70 Hz), 39 of 41 tones
#                      (10/10 preamble), -13.0 dBFS against -15.7; runner-up 15
ONAIR_STRANGER_REQUESTS = ((159.905, 161.654, "N3HYM-10"),
                           (163.523, 165.273, "W9FE"))

# The 2026-08-09 slot, and the reason the listen-before-transmit guard was
# recalibrated: it declared every channel it was pointed at occupied, by +0.6 to
# +0.7 dB, until `--force` became reflex — and then it could not be heard on the
# two that were genuinely carrying somebody else's session.
#
# Each file is receive audio through this station's rig and codec. The three
# 70 s ones are whole connect attempts, and the first transmission of each begins
# at 8.30 s, so the audio before ``SENSE_PRETX_S`` is exactly what the guard was
# looking at when it decided. `_listen` never keys at all.
#
# The labels are the operator's, from the same evening: 7108.5 kHz was listened
# to for 45 s and was silent; 7102.0 and 7103.5 were carrying another station's
# VARA session, with a 4.9 s over and repeated session-control bursts logged on
# them. The narrowband occupant on 7102.0 sits around 2600 Hz.
CHANNEL_SENSE = RF_CORPUS / "offair" / "channel_sense_20260809"
SENSE_CLEAR = (CHANNEL_SENSE / "clear_7108500_ve4wsc_listen.wav",
               CHANNEL_SENSE / "clear_7108500_ve4wsc.wav")
SENSE_BUSY = (CHANNEL_SENSE / "busy_7103500_kc9ghz.wav",
              CHANNEL_SENSE / "busy_7102000_kd9udl.wav")
#: Seconds of each capture that precede this station's first transmission.
SENSE_PRETX_S = 8.0

# 6800.0 kHz on 2026-08-14, recorded as that slot's negative control — no amateur
# allocation, no Winlink channel — and filed as a false alarm because `core.busy`
# called it occupied on all 21 of its status lines while every modem classifier in
# the slot stayed silent on it: 0 detections live, 0 from `creance --deep`, 0 CR
# from the handshake scanner, over 211.6 s.
#
# It is not a false alarm. Two steady carriers sit in the passband at audio 1133.0
# and 1229.9 Hz (RF 6799.633 and 6799.730 kHz), 30.4 and 25.1 dB over a 700 Hz
# local baseline, present in all 52 of its four-second slices and holding 1133.00 Hz
# to within 2 Hz in 44 of them. Neither is a codec birdie: the same audio bin reads
# +0.07, -0.08 and -0.66 dB on the three other windows of that hour, through the
# same rig and codec at 7103.5, 14103.2 and 7103.5 kHz, so it moves with the dial.
#
# So this is the only off-air capture held of a channel carrying a bare carrier and
# nothing else — the one occupant a VARA/PACTOR/ARDOP classifier is silent on by
# construction, and therefore the one an operator is most likely to read as empty
# because nothing named it. It lives in the monitoring slot's own output rather
# than in the curated corpus, so it skips once that directory is cleared.
CARRIER_ONLY_CHANNEL = (evidence.WORKING / "rx-20260814-114505-6800000"
                        / "audio" / "rx-000.wav")

# The two windows of that same slot that `MonitorGate`'s held-lift path was built
# against, and the quiet 40 m window that has to stay shut beside them. All three
# are the monitoring slot's own output rather than curated corpus, so they skip
# once it is cleared; the figures either side of the threshold are in
# `tools/vara_monitor.MONITOR_HELD_ENTER`.
#
#   7103.5 kHz, 17:34 UTC: eleven minutes of a 25-minute VARA session between
#   W9SEM-10 and KD9OSV, named by the Winlink RMS feed and on channel. The gate's
#   peak test handed `classify` nothing at all across both segments.
#   7096.5 kHz, 16:34 UTC: the VARA 500 exchange whose two callsigns and mode a
#   second source names and whose spectrum measures 600 Hz wide on the registered
#   channel. Also nothing from the peak test.
#   7103.5 kHz, 16:14 UTC: the ten minutes the feed and every classifier agree
#   were empty — no session on the channel after 16:11:41, zero deep detections.
MONITORED_SESSIONS = (evidence.WORKING / "rx-20260814-123414-7103500" / "audio" / "rx-000.wav",
                      evidence.WORKING / "rx-20260814-113359-7096500" / "audio" / "rx-000.wav")
MONITORED_QUIET = (evidence.WORKING / "rx-20260814-111357-7103500"
                   / "audio" / "rx-000.wav")

# 80 m QRN through this station's own receiver, which no capture behind the
# occupancy thresholds holds: every clear-side figure they are set on was taken on
# 40 m or out of band. These are the eight seconds after our own transmission
# stops in each of two besra listens -- 24.0-32.0 s of a 37.5 s recording rather
# than its tail -- and each holds one static crash and nothing else — `shape` and `tone` are at their noise floors in both.
# What `burst` makes of them is asserted in `tests/kestrel/test_channel_busy.py`,
# and it is the reason the 2026-08-19 refusals on 3595 and 3596.5 never cleared.
QRN_80M = tuple(evidence.LOGS / "onair" / n for n in
                ("20260816T002744Z-besra-3585000.wav",
                 "20260816T002623Z-besra-3595000.wav"))
#: Where in those two the transmission has ended and only the band is left.
QRN_80M_WINDOW = (24.0, 32.0)

# The four windows a PACTOR-1 call to 3588 kHz was refused on across 353 s on
# 2026-08-19, kept by the gate itself: `burst` 7.42-8.00 against its 6.6, `shape`
# 2.78-3.99 against 6.0, `tone` 0.67-1.87 against 5.0. The run then keyed over
# them and read peak 1.9-3.0x median with no run at the FSK tones over eight
# cycles, so the channel under all four is band noise and the refusals are false.
#
# They are the clear side `burst` has never had: every capture the thresholds are
# set on was taken on 40 m or out of band, and these are 80 m through this
# station's own receiver at the length and in the band the gate listens in.
SENSE_REFUSED_EMPTY = tuple(evidence.LOGS / "sense" / f"20260820T{t}Z-sense-3586500.wav"
                            for t in ("021251", "021344", "021524", "021625"))
#: What those four were judged over, which is not the full band: PACTOR-1.
SENSE_REFUSED_EMPTY_BAND = (380.0, 2620.0)

# The 2026-08-15 call to N0LCR-1 on 7103.5 kHz, and the first recording held of a
# gateway answering OFF FREQUENCY. Seven connect-requests went out; the tool printed
# "no answer — resending CR" after six of them and the attempt ran out at 60 s.
#
# One answer is in the audio, at 18.36 s, and it sits one carrier low (-23 Hz): 14
# of 15 payload tones at that shift, with all fifteen comparable. At zero frequency
# offset nothing anywhere in the recording confirms more than 4 of 15 — that is the
# whole of why the search walked past it, and why `tools/vara_monitor.py`, which has
# searched +-3 carriers since it was written, names it off the same file. The other
# six windows reach 4 of 15 at best at every shift: the operator heard the gateway
# answering more than once, but only this one answer is in the recording.
#
# It is this station's own receive recording of a live attempt, so it also carries
# the transport's geometry: our own seven transmissions and the receiver mute behind
# each of them. It lives in `logs/onair` rather than the curated corpus, so it skips
# where that directory has not been kept.
ONAIR_OFFSET_ANSWER = (evidence.LOGS / "onair"
                       / "20260815T020415Z-W9SSJ-N0LCR-1.wav")

# The 2026-08-19 call to KC9GHZ on 7103.5 kHz: 127 s, connected at 61 s, and one
# ordinary base BW2300 over at 67.804-72.017 s carrying the first block of the
# gateway's RMS Trimode SID banner.
#
# It is the recording the argmax equalisation is measured on. A frequency-selective
# fade runs through it: over the whole 4.21 s frame the four bins around 1.3 kHz
# read 2-4x the rest of the band, so six reference columns land in those four bins
# instead of their own, the frame scores 18 of 24 and the turbo decode fails. It is
# the same gateway, the same waveform and the same receiver as
# `ONAIR_GATEWAY_GREETING` above, which reads 24 of 24 — so the two together are
# what separate the level of an over from the shape of the channel under it.
#
# The gateway also answered three of our eight connect-requests here, at 45.580,
# 53.205 and 59.880 s, each 0.10-0.18 s after our own transmission ended and each
# with its first payload tones inside the mute release [see
# `test_response_by_stream`].
ONAIR_FADED_GREETING = (evidence.LOGS / "onair"
                        / "20260819T034047Z-W9SSJ-KC9GHZ.wav")

# The same gateway on the same channel 92 minutes later, eight connect-requests,
# `NOT connected (no/!=expected response)` (`working/t5-vara-kc9ghz.log`) — and
# this time nobody answered. Scored at every alignment, every shift and every
# sample phase, the best any of the 70 s reaches for KC9GHZ is 5 of 15, which is
# what a callsign that is not on a recording reaches.
#
# It is the control the clearance rule needs, because it holds everything the
# session above holds except an answer: the same peer, the same channel, the same
# receiver, and a mute of 171-189 ms against that session's 167-180. A rule that
# recovers three answers there and finds none here is reading the gateway rather
# than the changeover.
ONAIR_UNANSWERED_CALL = (evidence.LOGS / "onair"
                         / "20260819T051249Z-W9SSJ-KC9GHZ.wav")

# The control for it, two days earlier: 526 s with the same gateway on the same
# channel, and the one session in the logs kept here whose overs were read as they
# arrived (`working/s2-vara-kc9ghz-force.log`). Three base BW2300 overs at 41.403,
# 47.882 and 54.354 s carry the whole of the gateway's greeting, and the first of
# the three carries the same 89 payload bytes as the frame the 2026-08-19 session
# died on.
#
# What differs is the band under them. Per-bin medians span 1.60x, 1.63x and 1.68x
# across those three overs against 3.88x on 2026-08-19 — so the two sessions
# together say the fade and not the level, the peer, the waveform or the receiver
# is what separated a frame that decoded from one that did not.
ONAIR_GATEWAY_OVERS = (evidence.LOGS / "onair"
                       / "20260817T021239Z-W9SSJ-KC9GHZ.wav")

# The 2026-08-19 call to W8MW on 3595.0 kHz, and the first recording held of a
# gateway's connected-ack refused by our own recogniser. The connect-response came
# back 15 of 15, the link-setup went out three times, and every burst in the three
# turnaround windows was logged `preamble holds for 0 alignments (need 3)`
# (`working/t5-vara-w8mw-force2.log`).
#
# The ack is at 23.280 s, 0.14 s after our first link-setup's last sample. The
# channel was occupied — the call went out over an occupant on `--force` — and its
# lower carriers are under that: symbol 3 reads (68, 78) exactly, symbols 1 and 2
# deliver only 74 and 69, and the second carrier the pair reader returns for those
# two is whatever the band left. Matched on all six bins it holds nowhere; scored on
# the four that stand clear it holds for 24 alignments, and nothing else in the 70 s
# reaches four comparable carriers at all.
ONAIR_REFUSED_ACK = (evidence.LOGS / "onair"
                     / "20260819T054405Z-W9SSJ-W8MW.wav")

# The 2026-08-26 call to KD0PYG on 7101.2 kHz, and the other way a busy channel
# takes a carrier. The gateway answered the first link-setup 0.19 s after our last
# transmitted sample; the burst opens at 41.096 s and the receiver delivered all
# eleven symbols of it.
#
# Preamble symbols 1 and 2 read (56, 74) and (64, 69) exactly. Symbol 3's lower
# carrier 68 is the strongest bin in the band and its upper carrier 78 is 5.9 dB
# down — under an occupant at 95/96 that is 0.2 dB ABOVE 68, so the two strongest
# bins of that window are 68 and the occupant's, and the pair reader returns them.
# `ONAIR_REFUSED_ACK` is the same recogniser losing a carrier the band was over;
# this is it losing one another station was on top of, which reads as a symbol the
# peer got wrong rather than as a symbol slot that says nothing. Matched on the
# pair alone the ack holds at NO alignment -- five bracketed bursts, five
# `preamble holds for 0 alignments (need 3)`, and the whole of six unconnected arms
# that night. Read as one crowded symbol it holds for 27.
#
# Nothing else in the 190 s comes near it: the same rule holds nowhere else in the
# recording, and the gateway's own connect-response is there too, 15 of 15 payload
# tones at 35.121 s.
ONAIR_CROWDED_ACK = (evidence.LOGS / "onair"
                     / "20260826T013002Z-W9SSJ-KD0PYG.wav")

# The two 2026-08-26 calls to KB3AC-10 on 3596.5 kHz, and the recordings that say
# what "no connect-response" was covering. The operator heard the gateway answer
# calls 2 and 3 on both arms; the first arm's log names one answer and the second
# arm's names none.
#
# One burst answers the second call on BOTH arms, 4 ms after this station's
# receiver comes back out of its own transmit mute. Each carries the
# connect-response's fixed eight-tone preamble — symbol 0 inside the mute at 0.96
# and 0.80 dB of clearance, symbols 1 to 7 exact at every alignment across the
# burst — and 14 and 13 clean payload tones behind it.
#
# WHO KEYED THEM WAS SETTLED ON 2026-08-29, AND IT IS KB3AC-10. Each payload pins
# exactly one of the 2**24 states of the payload generator; discrete-logged over
# what a connect-response leaves free, each reaches one (seed, mult) pair, and both
# reach seed 14821 — which is the seed KB3AC-10's own CRC-16/GENIBUS of 14532
# gives. The two states sit 60 draws — two whole 15-symbol payloads — apart on that
# one stream, because they are frames 16 and 14 of it where the connect-response is
# frame 17. Only ``mult`` ever disagreed with KB3AC-10, and ``mult`` is where the
# pre-advance enters. See :data:`ONAIR_PEER_ANSWERS`.
#
# Times are the alignment the route accepts, in the recording's own clock.
ONAIR_UNATTRIBUTED_ANSWERS = (
    (evidence.LOGS / "onair" / "20260826T011747Z-W9SSJ-KB3AC-10.wav", "KB3AC-10",
     22.925, 7, 14, 0xcf9ecf),
    (evidence.LOGS / "onair" / "20260826T012205Z-W9SSJ-KB3AC-10.wav", "KB3AC-10",
     22.873, 7, 13, 0x7b5703))

# Four attempts this station reported as unanswered, and the answers in them. Each
# entry is (recording, the callsign dialled, how many answers, how many of them sit
# at each position on the callsign's own payload lattice).
#
# Every burst counted here regenerates the DIALLED station's fifteen payload tones
# exactly, at a frame of its stream that is not the connect-response's, and lands
# 0.07-0.14 s behind one of this station's own unkeys — measured against the slot
# tap's key edges for the N5WAJ arm, whose 26 answers all fall in that band.
#
# The KB3AC-10 arm of 01:17 is the control: three answers at 16, 15 and 15, then
# the connect-response itself at 60.84 s, which the callsign route takes and brings
# the link up on. So frame 17 still means what it meant, and 14-16 are answers that
# are not it. What they say, this station cannot yet read.
ONAIR_PEER_ANSWERS = (
    (evidence.LOGS / "onair" / "20260826T011747Z-W9SSJ-KB3AC-10.wav", "KB3AC-10",
     3, {16: 1, 15: 2}),
    (evidence.LOGS / "onair" / "20260826T012205Z-W9SSJ-KB3AC-10.wav", "KB3AC-10",
     6, {14: 5, 16: 1}),
    (evidence.LOGS / "onair" / "20260829T022350Z-W9SSJ-N5MDT.wav", "N5MDT",
     6, {14: 5, 15: 1}),
    (evidence.LOGS / "onair" / "20260829T023527Z-W9SSJ-N5WAJ.wav", "N5WAJ",
     26, {14: 18, 15: 8}))

# The 2026-08-29 02:18z call to KE8LVA and the 02:31z call to N5WAJ, from the same
# slot as the two above and on the same instrument: nothing on either channel
# regenerates the dialled station's payload at any position. They are what says the
# lattice search reports an absence as an absence.
ONAIR_UNANSWERED_ARMS = (
    (evidence.LOGS / "onair" / "20260829T021824Z-W9SSJ-KE8LVA.wav", "KE8LVA"),
    (evidence.LOGS / "onair" / "20260829T023146Z-W9SSJ-N5WAJ.wav", "N5WAJ"))

# THE UNASKED HANDOVER. Three arms of the 2026-08-29 day slot, two gateways, that
# each took one greeting over of 89 bytes cut mid-word and then went quiet for
# 100-200 s without disconnecting. What the gateway keyed 0.13-0.15 s behind this
# station's control burst is `SESSION_TURN_RELEASE_RESPONDER`: 17 of 17 tones at
# both KB5LZK arms and 15 of 15 payload at the N5TW one, whose leading symbol falls
# under our own unmute. Each payload pins one 24-bit state — 0xdd833c for KB5LZK,
# 0xfec4e1 for N5TW — and both sit at position 13 of their own callsign's lattice,
# which is the position (289, 391) names. A second callsign at the same position is
# what separates the state from the identity: regenerated against the wrong one of
# the two, every arm scores 2 of 17.
#
# So the greeting was not cut off. The gateway broke mid-word to hand the channel
# over, and this station owed it a transmission — an over, or the caller's release
# straight back. Nothing ran the search: `_stream_grant` owns this frame only
# between a turn-request of ours and its answer, and the turn was the peer's.
#
# The negative population is the recordings themselves: 1159 windows of 1.87 s
# every 0.5 s clear of the burst, over all three, reach 3 of 15 and accept none.
#
# Times are the burst's own start in the recording's clock, and the state is what
# the payload pins — `None` on the 14:44 arm, whose three misread tones are wrong
# rather than absent, so they name no generator state at all. It is carried here as
# the arm that clears the cut without pinning anything, which is the difference
# between recognising a frame and solving one.
ONAIR_UNASKED_HANDOVER = (
    (evidence.LOGS / "onair" / "20260829T141915Z-W9SSJ-KB5LZK.wav", "KB5LZK",
     72.7113, 15, 0xdd833c),
    (evidence.LOGS / "onair" / "20260829T143002Z-W9SSJ-N5TW.wav", "N5TW",
     110.7300, 15, 0xfec4e1),
    (evidence.LOGS / "onair" / "20260829T144437Z-W9SSJ-KB5LZK.wav", "KB5LZK",
     98.7020, 12, None))

#: The lattice position `SESSION_TURN_RELEASE_RESPONDER` occupies, which every arm
#: of :data:`ONAIR_UNASKED_HANDOVER` is read at.
UNASKED_HANDOVER_POSITION = 13

# The 2026-08-20 02:52 call to KC9GHZ on 3595.0 kHz, 177 s, and a gateway's whole
# Winlink greeting in two ordinary base BW2300 overs, 63.693-67.907 s and
# 70.174-74.388 s. Their payloads join at "has 11" + "8 daily minutes" into the SID
# banner and the `;PQ:` secure-login challenge behind it, and both decode off this
# recording with a clean CRC, at 22 and 19 of 24 reference columns. The second is
# also what settles who the peer was: it names KC9GHZ and EN62BK in its own text,
# on a channel that station's published list does not carry.
#
# Live, that second over drew `tx NAK (1/2)`. What differed is that the station
# keyed a keepalive not four seconds into it. `AudioVaraIO.tx` drops every sample
# received under a transmission, so the over search concatenated the 3.2-3.9 s in
# front of that keying to the band behind it, and spliced there the wreckage still
# scores 16-19 reference columns — above `_OVER_GUARD_MIN`, so it is claimed as an
# over — and fails its CRC, which is `_UNDECODED` and a NAK for a frame this
# station broke itself. Both halves of that are here: the over is good, and the
# splice is what made it look otherwise.
#
# The 102 s after the NAK are the rest of the session. The gateway never repeated
# the over — the best any alignment in them scores is 10 of 24, against a guard of
# 16 — so the ladder was never asked a second question, while the channel went on
# handing the bracket gate 42 bursts at a longest gap of 8.50 s.
ONAIR_NAKED_OVER = (evidence.LOGS / "onair"
                    / "20260820T025204Z-W9SSJ-KC9GHZ.wav")

# The same evening's 02:41 call to the same gateway, 236 s, and the same ending
# with nothing in the log to mark it. One base BW2300 over at 17.90-22.85 s, 24 of
# 24 reference columns, carrying the front of the same RMS Trimode banner; this
# station answered it and then keyed nothing for the remaining 213 s of the
# recording. That session's log ends on `tx per-over response`, with no NAK to
# mark the fault and no teardown line at all.
ONAIR_STANDING_LINK = (evidence.LOGS / "onair"
                       / "20260820T024156Z-W9SSJ-KC9GHZ.wav")

# The 2026-08-28 call to K5FIT on 7105.4 kHz, and the only recording held of a
# gateway answering this station at BW500. Eight connect-requests over 190 s,
# `NOT connected (no/!=expected response)`
# (the 2026-08-28 slot, arm 8) — and K5FIT answered four of them, at
# 35.977, 48.127, 60.196 and 98.732 s, 15/15, 13/13, 13/13 and 13/13 tones against
# `CONNECT_RESPONSE_500`. One of the four opened the energy gate and was logged
# `unexpected in state=CONNECTING role=initiator step=I_CR_SENT`; the other three
# were never bracketed at all, and the stream search that would have carried them
# was regenerating BW2300's alphabet.
#
# It is the fix's own control as well as its positive. Scored the same way over the
# same 190 s, BW2300's connect-response reaches 3 of 15 for K5FIT, and the four
# other callsigns the slot dialled reach 5 of 15 at best — so what separates the
# four accepts from the recording around them is the callsign AND the bandwidth.
ONAIR_BW500_ANSWER = (evidence.LOGS / "onair"
                      / "20260828T233326Z-W9SSJ-K5FIT.wav")

# Every link-setup this station has sent that did not end in a connect, as
# `(recording, link-setups in it)`: 27 across three evenings and six callsigns,
# 861 s of receive audio. It is the negative population for anything that would
# read "the peer has taken the link" out of the turnaround instead of out of the
# connected-ack, and it is declared here because the count is the evidence — a
# window that quietly went missing would relax the floor measured over it.
#
# The 2026-08-19 pair are the third and fourth calls to KC9GHZ that evening
# (`working/t5-vara-kc9ghz-3.log`, `-4.log`), the two that got a connect-response
# and no further. `ONAIR_REFUSED_ACK` is in the tuple as the case that is not
# quite like the others: W8MW did ack, our recogniser refused it at the time, and
# W8MW still sent nothing afterwards — a peer that has taken the link waits for
# the session-confirm rather than starting to send.
#
# The 2026-08-03 three live under `captures/` rather than `logs/onair`, which is
# the same reading of the same thing: written here by a run, in no clone.
ONAIR_UNCONNECTED_LINKSETUPS = (
    (evidence.CAPTURES / "campaign-0803-0945" / "vara-KB8AY-7101500"
     / "20260803T145917Z-W9SSJ-KB8AY.wav", 2),
    (evidence.CAPTURES / "campaign-0803-0945" / "vara-NS0A-7102000"
     / "20260803T150107Z-W9SSJ-NS0A.wav", 1),
    (evidence.CAPTURES / "kb8ay-witness-1006" / "attempt1"
     / "20260803T150701Z-W9SSJ-KB8AY.wav", 3),
    (evidence.LOGS / "onair" / "20260814T053239Z-W9SSJ-KC2OUR.wav", 3),
    (evidence.LOGS / "onair" / "20260814T053556Z-W9SSJ-KC2OUR.wav", 3),
    (evidence.LOGS / "onair" / "20260814T053817Z-W9SSJ-W2KBF.wav", 1),
    (evidence.LOGS / "onair" / "20260814T054216Z-W9SSJ-KC2OUR.wav", 3),
    (evidence.LOGS / "onair" / "20260814T054727Z-W9SSJ-KC2OUR.wav", 3),
    (evidence.LOGS / "onair" / "20260819T053010Z-W9SSJ-KC9GHZ.wav", 3),
    (evidence.LOGS / "onair" / "20260819T053413Z-W9SSJ-KC9GHZ.wav", 2),
    (ONAIR_REFUSED_ACK, 3),
)

# W0LON on 40 m, 2026-09-04, eleven minutes apart, and the pair that separates the
# handshake from what follows it. The link came up two different ways and the
# gateway did the same thing after both.
#
# On the 19:56 arm the gateway keyed `SESSION_TURN_REQUEST_RESPONDER` addressed to
# W9SSJ before its connected-ack was read, and this station answered it with
# `SESSION_TURN_RELEASE` and no session-confirm. On the 19:45 arm the ack was read,
# the confirm went out, and no turn-request was keyed at all. Both are here as
# `(tape, our answering burst's key-down and key-up, what we keyed, the gateway's
# next transmitted sample)`, off the receiver's own transmit mute.
#
# What follows is the same on both to within a twentieth of a second: nothing for
# 7.3 and 7.6 s, then `SESSION_RESPONDER_IDLE` on a metronome for 49.35 and 49.30 s,
# then silence to our own give-up. No DATA over at any index record, no greeting,
# and in neither session's turnarounds a single 11-symbol control burst — which is
# what a stock responder handed the turn keys every 2.3 s until it is answered.
ONAIR_W0LON_ARMS = (
    (evidence.LOGS / "onair" / "20260904T195626Z-W9SSJ-W0LON.wav",
     23.71, 24.70, "turn-release", 32.00),
    (evidence.LOGS / "onair" / "20260904T194552Z-W9SSJ-W0LON.wav",
     117.38, 118.24, "session-confirm", 125.85))

#: The gateway's turn-request on the 19:56 arm: its first payload symbol, and the
#: station it is addressed to. It ends 0.24 s before our own key-down, so the
#: release was not keyed across it.
W0LON_TURN_REQUEST = (22.104, "W9SSJ")

#: First and last idle of each arm's train, in the tape's own clock. The two spans
#: agree to 50 ms on sessions that started differently.
W0LON_IDLE_TRAINS = ((32.00, 81.35), (125.85, 175.15))

#: Where a connected-ack would be on each arm, in :data:`ONAIR_W0LON_ARMS` order:
#: the turnaround our link-setup opens, and whether the gateway keyed one into it.
#: The 19:45 arm's is 2 of 2 windows at the peer offset the connect-response
#: fixed; the 19:56 arm holds none at any offset anywhere on 107.9 s of tape,
#: which is what leaves the turn-request as the only thing it answered our
#: link-setup with.
W0LON_ACK_WINDOWS = ((16.43, 23.71, False), (116.19, 117.38, True))

# KB3AC-10, 2026-08-23 04:52z: stock RMS Trimode 1.4.2.0, the deepest a VARA
# session from this station has run. Three turn-requests went out and the log
# reported all three unanswered; the recording holds the gateway answering two of
# them, at 30 of 30 and 30 of 31 payload tones. It is the negative population for
# a turn grant as much as the positive one — 187 s of live 80 m in which exactly
# two changeovers carry that frame and sixteen do not.
#
# Windows are `(our last transmitted sample, the next key-up, the grant is in it)`,
# read off the hard mute that brackets each of our own transmissions in the
# recording. The four listed are one grant each way and the two nearest things to
# one on the tape: the gateway's DATA over, and the turnaround where it keyed
# something the grant recogniser scores at 3 of 18. That something is named as of
# 2026-08-26 — SESSION_TURN_RELEASE_RESPONDER, what both gateways answer a first request
# with — and it has to go on reading here as no grant.
ONAIR_TURN_GRANTS = (evidence.LOGS / "onair"
                     / "20260823T045251Z-W9SSJ-KB3AC-10.wav", "KB3AC-10",
                     ((87.350, 97.845, True),      # turn-request 2, answered
                      (99.235, 120.125, True),     # turn-request 3, answered
                      (75.370, 85.960, False),     # turn-request 1, not this frame
                      (56.045, 61.055, False)))    # the gateway's greeting over

# KE8LVA, 2026-08-23 04:57z: the one recording holding SESSION_RESPONDER_IDLE. The
# gateway answered the connect, keyed its greeting over, took our per-over response
# — and then keyed that frame thirteen times, 3.406 s apart, through our own
# keepalive cadence and answering nothing. Burst starts, to the 5 ms census
# resolution; three further slots of the same cadence fall inside a stretch of
# gateway transmission and are not read.
# The two real BW2300 acknowledgements the 2026-08-23 arms drew, and the only two:
# the 11-symbol two-tone burst [spec 04 §4.2C] each gateway answered our link-setup
# with, 0.76 s and 1.0 s before our own session-confirm went out. `_ack_plateau`
# holds them for 34 and 29 alignments against a threshold of 3; the BW500 DBPSK
# table reads neither, and reads nothing in 400 random 0.75 s windows of the same
# two recordings. The third arm of that night never connected and has none.
ONAIR_BW2300_ACKS = ((evidence.LOGS / "onair"
                      / "20260823T045702Z-W9SSJ-KE8LVA.wav", 42.221),
                     (evidence.LOGS / "onair"
                      / "20260823T045251Z-W9SSJ-KB3AC-10.wav", 54.169))

ONAIR_RESPONDER_IDLE = (evidence.LOGS / "onair"
                        / "20260823T045702Z-W9SSJ-KE8LVA.wav", "KE8LVA",
                        (53.371, 56.768, 60.184, 63.588, 66.995, 70.408, 73.805,
                         77.222, 80.633, 84.047, 94.266, 97.669, 101.086))

# KE8LVA, 2026-08-26 12:59z: the first session in which this station took the turn
# and keyed a payload over, and the only recording holding either of the two frames
# the gateway answered with. Both are 0.17-0.18 s turnarounds off our own keying, so
# both open inside our receiver's mute; the times are their first PAYLOAD symbol.
#
# Segmented on absolute in-band level, this session holds eight gateway
# transmissions and no more: the connect-response, the connected-ack, three
# record-3 DATA overs, a turn-refusal, a turn grant and this over-answer. The
# twelve the live log recorded as high-speed-level overs are the last two of those
# and 110 s of band noise the energy gate held open.
ONAIR_OVER_ANSWERED = (evidence.LOGS / "onair"
                       / "20260826T125939Z-W9SSJ-KE8LVA.wav", "KE8LVA",
                       50.603,      # session-turn-release-responder, answering request 1
                       69.160)      # session-responder-over-answer, answering our over

# KB3AC-10 on 80 m, 2026-08-31 03:24z: the first session in this station's record
# in which a gateway sent a SECOND greeting over, and the recording that separates
# finding an over from reading one for the second time.
#
# Both overs are base BW2300 and both are whole. Over 1 reads 24 of 24 reference
# columns; over 2 reads 18, and every hard-decision build failed it -- 71 of its
# 395 columns are won by the wrong bin, and the reference columns that miss are
# scattered rather than contiguous, with no constant offset between the bin that
# won and the bin that should have. What separates the two overs is not level (the
# lit bins carry the same power to within 0.11 dB), not the band's shape (per-bin
# medians span 1.38x on over 1 and 1.32x on over 2 -- both flat, unlike
# `ONAIR_FADED_GREETING`) and not the structure. It is per-column spread: the
# reference columns' own SNR has a standard deviation of 4.4 dB on over 1 and
# 6.6 dB on over 2, around the same mean.
#
# The two payloads are the halves of one RMS Trimode greeting, and the join is what
# proves the pair: over 1 ends "...with KB3AC-1" and over 2 opens "0 (FN10PV)".
ONAIR_TWO_OVER_GREETING = (evidence.LOGS / "onair"
                           / "20260831T032427Z-W9SSJ-KB3AC-10.wav",
                           (217.87, 222.32),    # over 1, 89 payload bytes
                           (223.96, 228.40))    # over 2, 65 and a short-block end

# Every logged loopback session lives under one directory, named for when it ran
# and who called whom, holding both directions plus the PTT ledger that says which
# station keyed each burst. `_CAPS` is the same harness's bench captures, kept
# beside them.
_HARNESS_LOGS = ROOT / "oracle" / "logs"
_LOGGED = _HARNESS_LOGS / "audio"
_CAPS = _HARNESS_LOGS / "caps"


#: The responder's cable of the 2026-09-09 four-message fetch, where a stock
#: 4.9.0 keys TWO overs inside one PTT window: it transmits 81.330-89.880 s and
#: the two 89-byte blocks in that window are blob offsets 534 and 623 of the
#: 2117 bytes the gateway handed VARA. The station under test read one of them
#: and acknowledged 4.5 s inside the window, and the B2F parser met the hole at
#: the next block boundary  [see tests/kestrel/test_two_over_window].
FETCH_TWO_BLOCK = _CAPS / "0909-final-clean" / "final-clean__b2a.wav"
#: The window to replay, and the instant the responder unkeys inside it.
FETCH_TWO_BLOCK_AT = (81.0, 90.5, 89.880)

#: The same shape one arm later, where a session recogniser took a
#: `session-responder-over-idle` out of the SECOND block of the window and the
#: re-acknowledgement ladder keyed a rung on top of it: the block at blob offset
#: 1513 was never delivered  [see tests/kestrel/test_two_over_window].
FETCH_IDLE_IN_WINDOW = _CAPS / "0910-clean" / "g-clean__b2a.wav"
#: Its window: the responder keys 132.170 s for 8.55 s.
FETCH_IDLE_IN_WINDOW_AT = (132.0, 141.5, 140.720)

#: One draw later, the window where the hold never engaged at all: the six
#: columns behind the first block's frame read 0.304 of that frame — under the
#: level test's threshold, over a block that was there, every reference column of
#: it hitting — so the station answered 4.5 s into an 8.55 s transmission and the
#: block at blob offset 1691 was never read.
FETCH_LEVEL_MISSES = _CAPS / "0910-h-clean2" / "h-clean2__b2a.wav"
#: Its window: the responder keys 135.295 s for 8.555 s.
FETCH_LEVEL_MISSES_AT = (135.0, 144.8, 143.850)


def _side(session: str, direction: str):
    """One direction of a logged session, by the session's own naming."""
    return _LOGGED / session / f"{session.split('-', 2)[2]}__{direction}.wav"


# A logged BW500 session, station A's side, opening with the two handshake bursts
# a real VARA keys at that bandwidth: the connect request AAAA1 sends to BBBB2 at
# sample 24576, and BBBB2's connect-response behind it. The ledger says which
# station keyed which, so nothing here is inferred from the audio. This is the
# arbiter for the BW500 payload alphabet — a real VARA's own tones rather than a
# round trip through our own generator.
BW500_HANDSHAKE = _side("20260713-105241-AAAA1-BBBB2-bw500-zeros256", "a2b")
#: The same session run the other way round, so the request is keyed to the other
#: callsign — which is what separates the alphabet from one lucky seeding.
BW500_HANDSHAKE_REV = _side("20260713-101147-BBBB2-AAAA1-bw500-prbs91024", "a2b")

# One logged BW500 session, station B's side: a long transfer, so it is dense in
# the control bursts the token detector is measured against.
_SESSION = "20260713-143228-AAAA1-BBBB2-bw500-prbs94096"
CONTROL_BURSTS = _side(_SESSION, "b2a")

# Station A's side of that same session, and the only recording held anywhere that
# has bursts carrying more than one frame: 4096 PRBS9 bytes as 52 bursts, 45 of them
# 796 columns (preamble + two 394-column frames, 8.50 s) and 7 of them 403 (one
# frame, 4.31 s) — 96 data frames against the 51 the receiver demodulates.
#
# The whole session is 92 MB and a single 8.50 s burst cut out of it is still 816 kB
# as int16, so it stays corpus rather than becoming package data — and it has to be
# this session, because the b2a side kept here is all control bursts, none over 3 s.
# Point `KESTREL_CORPUS` at the vara source tree to measure the two-frame case.
MULTIFRAME_SESSION = _side(_SESSION, "a2b")


# The one BW2300 session held with a real VARA at BOTH ends and both directions
# on tape: VARA HF v4.9.0 against itself over the Wine/BlackHole bench, 126 payload
# bytes each way, with the host-port event log on the recorder's own clock. Every
# short frame in it is index-modulated MFSK, four 512-sample columns per two-tone
# symbol; nothing in either direction is the single-carrier DBPSK the BW2300 token
# table asserts.
#
# The acknowledgement is the 45-column burst listed below -- 0.489 s, eleven
# two-tone symbols on the connected-ack's own fixed preamble [spec 04 4.2C], which
# `vara_arq._ack_plateau` reads in all eleven of them and in neither 33-column one.
# `b2a` 8.567 is the connected-ack and 9.943 the first per-over answer, and they are
# the same waveform to correlation 1.00 -- which is what says the two are not
# distinct tokens at this bandwidth.
BW2300_REFERENCE = _LOGGED / "20260814-065053-W9SSJ-W1AW-bw2300-ref128"
BW2300_REF_SIDES = tuple(BW2300_REFERENCE / f"W9SSJ-W1AW-bw2300-ref128__{d}.wav"
                         for d in ("a2b", "b2a"))

#: ``(side, start_s, columns)`` for every short control burst in that session, read
#: off its own event log. 45 columns is the acknowledgement family, 33 the turn
#: exchange; both are within the span `vara_control`'s BW2300 patterns demodulate.
BW2300_REF_CONTROL = (("a2b", 36.325, 33), ("a2b", 41.973, 45), ("a2b", 44.037, 45),
                      ("a2b", 56.106, 45), ("a2b", 68.154, 45),
                      ("b2a", 8.567, 45), ("b2a", 9.943, 45), ("b2a", 12.028, 45),
                      ("b2a", 14.087, 45), ("b2a", 19.170, 33),
                      ("b2a", 24.812, 45), ("b2a", 26.882, 45), ("b2a", 29.730, 45))


#: The two record-2 (host ``BITRATE (3)``) overs of that session: ``(side, PTT-on,
#: PTT-off, payload)``, read off its own event log. Each carries the 37-byte tail of
#: its direction's 126-byte transfer — both stations dropped a speed level for the
#: short block — and they are the only recording of the robust level this project
#: holds with the plaintext to go with it.
BW2300_REF_ROBUST = (
    ("a2b", 19.621, 24.650,
     (b"KESTREL REFERENCE SESSION A->B 0123456789 " * 3)[89:126]),
    ("b2a", 36.768, 41.800,
     (b"KESTREL REFERENCE SESSION B->A abcdefghij " * 3)[89:126]),
)


# The 2026-08-26 13:39z W6IDS link, and the log the station wrote while it ran: the
# pair `tools/rehear` is graded against, because between them they hold both halves
# of a replay. The recording has what arrived; the log has the session id the live
# decoder was polling, the `TX`/`PTT OFF` edges of its own mute, and the state
# changes that reset memory ARQ — none of which is in the audio.
#
# The session reported 51 frames. A replay built the way this project had been
# building them hears 28 of them; one carrying the log's session id and the log's
# own edges reproduces all 51. That is the acceptance test, and it is the reason
# both files are declared rather than discovered: `logs/` is git-ignored and the
# session logs are written outside the tree, so a run that has neither must skip
# and say so rather than pass a replay of half a session.
BESRA_LOGGED_SESSION = (evidence.LOGS / "onair"
                        / "20260826T133925Z-besra-7060000.wav")

#: Where the runner writes a session's log. `tools/rehear` searches the same places
#: and for the same reason: nothing has ever moved these into the tree.
SESSION_LOGS = tuple(
    Path(d).expanduser()
    for d in os.environ.get("HFMODEM_SESSION_LOGS", "").split(os.pathsep) if d
) or (Path.home(),)
BESRA_SESSION_LOG = next(
    (d / "ardop-day-08-w6ids.log" for d in SESSION_LOGS
     if (d / "ardop-day-08-w6ids.log").exists()),
    SESSION_LOGS[0] / "ardop-day-08-w6ids.log")


# The 2026-08-29 15:15z W4UC call, and the transcript beside it: six identical
# 1.99 s ConReqs on a 3.85 s cadence and nothing back. A call nobody answered is
# the shape that breaks `sessionlog.align` — every transmission leaves TWO silence
# brackets in the capture and on a regular cadence the comb built on the key-ups
# pairs with the log's key-downs exactly as often as the comb built on the
# key-downs does. The tie used to go to the lower offset, which put this session at
# -4.058 s where the truth is +1.681, and rehear then withheld the gaps and read
# our own six ConReqs back as the gateway's.
#
# The truth is not a preference. The rig mutes its own receiver while we transmit,
# and the capture's level drops for exactly the six intervals `+1.681` implies.
BESRA_REGULAR_CALL = (evidence.LOGS / "onair"
                      / "20260829T151512Z-besra-10145000.wav")
BESRA_REGULAR_CALL_LOG = (evidence.WORKING / "onair-0829-1014"
                          / "ardop-day-04-w4uc-30m-send.log")

#: What the capture goes quiet across, in seconds from its first sample: `PTT ON` to
#: `PTT OFF` for each of the six keyings, read off the level trace and not off the
#: log. The offset that produces them is the one the alignment has to find.
BESRA_REGULAR_CALL_MUTED = ((0.39, 2.38), (3.62, 5.61), (7.47, 9.47),
                            (11.31, 13.30), (15.16, 17.15), (19.00, 21.01))
BESRA_REGULAR_CALL_OFFSET = 1.681


#: The two-VARA sessions of 2026-08-26 that say what the 11-symbol control
#: burst's seven state symbols are keyed to: ``(session, caller, called)``, one
#: recording per direction. The bench, the shape and both payloads are identical
#: across them and the callsign assignment is the only thing that moves — three
#: callees against one caller, one changed caller, one role swap, and two
#: unrelated pairs.
CONTROL_BURST_CALLER_KEY = (
    ("20260826-141036-handover3", "W9SSJ", "W1AW"),
    ("20260826-214913-pair1ctl", "W9SSJ", "W1AW"),
    ("20260826-215332-calleddiff", "W9SSJ", "W1AX"),
    ("20260826-215507-callee2", "W9SSJ", "KI7QQQ"),
    ("20260826-215246-callerdiff", "W9SSK", "W1AW"),
    ("20260826-214828-swap", "W1AW", "W9SSJ"),
    ("20260826-214740-pair2", "N0XYZ", "KI7QQQ"),
    ("20260826-214952-pair3", "VE3ABC", "G0XYZ"),
)


def control_burst_sides(session: str):
    """``(caller's cable, responder's cable)`` for one of those sessions."""
    return _side(session, "a2b"), _side(session, "b2a")


# The responder's cable of two 2026-08-30 sessions, and the only audio on disk
# holding what a stock responder answers a CALLER's intermediate over with — the
# turnaround an outbound message of ours longer than one over lands in. Both
# cables were recorded, so the burst is the responder's by which file holds it.
#
# Each entry is the burst's own start; the caller's over ends 0.170-0.175 s ahead
# of it. `handover-2300` is 178 bytes each way and holds two, `handover-deaf` the
# first of a delivery the caller then closed, and two of the three are tone for
# tone identical.
OVER_CONTINUE_ANSWERS = (
    (_side("20260830-091442-handover-2300", "b2a"), (35.845, 40.825)),
    (_side("20260830-092607-handover-deaf", "b2a"), (28.890,)),
)
#: The 11-symbol control bursts on the first of those cables, which the same
#: reader has to keep refusing: they are the answer to a LAST over and ask for
#: nothing more.
OVER_CONTINUE_NEGATIVES = (46.485, 48.540, 60.610, 72.670, 84.750)

# The same burst at BW2750, off two 2026-09-04 sessions of the same pair with each
# delivery padded past one over. The CALLER's cable this time — that is the end
# whose copy the transmit table holds — and the responder's cable of the same two
# sessions behind it, which is where the alphabet shows: two of its symbols land
# on bins no BW2300 burst can reach.
OVER_CONTINUE_2750 = (
    (_side("20260904-031917-continue-2750-1", "a2b"), (14.605, 19.565, 24.525)),
    (_side("20260904-032037-continue-2750-2", "a2b"), (14.520, 19.485, 24.440)),
)
OVER_CONTINUE_2750_RESPONDER = (
    (_side("20260904-031917-continue-2750-1", "b2a"), (35.410, 40.370, 45.315, 50.275)),
    (_side("20260904-032037-continue-2750-2", "b2a"), (35.305, 40.260, 45.220, 50.170)),
)

# The last byte of every DATA body a stock 4.9.0 keyed on the two-cable sessions,
# in the order it keyed them: a FULL body's is the per-frame field and a short
# one's is the 0x82 its trailer ends on  [see arq.phy.vara_body]. Each cable is
# one station's transmitter, so the sequence is one station's own.
#
# `handover-2300` is a 178-byte delivery each way — two full overs and the empty
# one that closes them — and `ref128` is a one-over delivery, which is what pins
# the last full over's own value apart from the count in front of it.
OVER_FRAME_FIELDS = (
    (_side("20260830-091442-handover-2300", "b2a"), (0x8D, 0x81, 0x82)),
    (_side("20260814-065053-W9SSJ-W1AW-bw2300-ref128", "b2a"), (0x81, 0x82)),
)

# ``(start_s, record, per-frame field)`` for every DATA over of eight whole
# deliveries a stock 4.9.0 keyed on its own cable, six sessions, both stations,
# both directions. The record is read off the waveform and cross-checks the
# session's host log, where ``BITRATE (4)`` is the base level and ``BITRATE (3)``
# record 2.
#
# What the set is for is the pair: the over that CLOSES a delivery drops to
# record 2 wherever the full over in front of it announced ``0x81``, and stays at
# the base level in the two deliveries whose full over announced ``0x89``
# [see arq.phy.close_level]. Six and two, and no delivery here mixes them.
OVER_CLOSE_RECORDS = (
    (_side("20260830-091442-handover-2300", "a2b"),
     ((31.33, 3, 0x8D), (36.30, 3, 0x81), (41.29, 2, 0x82))),
    (_side("20260830-091442-handover-2300", "b2a"),
     ((9.94, 3, 0x8D), (14.92, 3, 0x81), (19.91, 2, 0x82))),
    (_side("20260830-092607-handover-deaf", "a2b"),
     ((24.37, 3, 0x81), (29.35, 2, 0x82))),
    (_side("20260830-092607-handover-deaf", "b2a"),
     ((9.95, 3, 0x89), (14.92, 3, 0x82))),
    (_side("20260830-092248-requeue-2300", "b2a"),
     ((9.96, 3, 0x89), (14.94, 3, 0x82), (59.06, 3, 0x81), (64.04, 2, 0x82))),
    (_side("20260814-065053-W9SSJ-W1AW-bw2300-ref128", "a2b"),
     ((14.60, 3, 0x81), (19.67, 2, 0x82))),
    (_side("20260814-065053-W9SSJ-W1AW-bw2300-ref128", "b2a"),
     ((31.60, 3, 0x81), (36.82, 2, 0x82))),
)


def _requires(*paths: Path, what: str):
    missing = [p for p in paths if not p.exists()]
    return pytest.mark.skipif(bool(missing),
                              reason=f"{what} not present ({missing or paths[0]})")


#: What the shared corpus has to hold for the receive side to be measured at all.
#: A skip is the right answer for a run from an installed wheel and the wrong one on
#: a source tree, and nothing here can tell those apart — so the distinction is drawn
#: once, in `tests/gates/test_corpus_present.py`, which reads this and fails.
RF_CORPUS_RECORDINGS = (ONAIR_CONNECT_ATTEMPT, ONAIR_GATEWAY_ANSWER,
                        *ONAIR_SILENT_CALLS, ONAIR_GATEWAY_GREETING,
                        *SENSE_CLEAR, *SENSE_BUSY, REGRESS_FIXTURES)

#: What `ROOT` has to hold for the BW500 and BW2300 results to be measured against
#: a real VARA's own audio rather than against a round trip through our own
#: generator. Same gate, same reason: a tree that means to have these and does not
#: is a stale checkout, not a wheel.
VARA_TREE_RECORDINGS = (BW2300_CAPTURE, BW500_CAPTURES, CLEAR_CHANNEL,
                        GATEWAY_SESSION / "rig_rx.wav", BW500_HANDSHAKE,
                        BW500_HANDSHAKE_REV, CONTROL_BURSTS, MULTIFRAME_SESSION,
                        *BW2300_REF_SIDES)

# WW2MI, 2026-08-18: two ARDOP connects four minutes apart, one authenticated and
# offered a message, one refused with `Invalid login challenge response -- 2
# attempts remaining`. The account's allowance is three, so the difference between
# them is not a thing another connect may be spent on establishing.
#
# The recorder sits upstream of the half-duplex mute, so these hold this station's
# own transmissions as well as the gateway's — the `;PR:` secure-login answer as it
# left the transmitter, beside the `;PQ:` it answers, in one file. That is the only
# reading of that answer that costs nothing, and it is what says whether a refused
# login was a wrong number or a right one refused.
WW2MI_ACCEPTED = evidence.LOGS / "onair" / "20260819T024321Z-besra-7102100.wav"
WW2MI_REFUSED = evidence.LOGS / "onair" / "20260819T024752Z-besra-7102100.wav"

#: This station's own session records, which are neither of the above: they were
#: written here by a run rather than curated, and `working/`, `logs/` and
#: `captures/` are all gitignored, so a clone has none of them and this machine
#: has all of them. They
#: were the one body with no gate over them at all -- and `ONAIR_OFFSET_ANSWER` is
#: the sole evidence for widening the response search, so its silent absence took
#: that result with it.
STATION_RECORDS = (CARRIER_ONLY_CHANNEL, *MONITORED_SESSIONS, MONITORED_QUIET,
                   ONAIR_OFFSET_ANSWER, ONAIR_FADED_GREETING,
                   ONAIR_UNANSWERED_CALL, ONAIR_REFUSED_ACK, ONAIR_CROWDED_ACK,
                   ONAIR_GATEWAY_OVERS, ONAIR_NAKED_OVER, ONAIR_STANDING_LINK,
                   ONAIR_BW500_ANSWER,
                   *QRN_80M, *SENSE_REFUSED_EMPTY,
                   WW2MI_ACCEPTED, WW2MI_REFUSED,
                   *(p for p, _ in ONAIR_UNCONNECTED_LINKSETUPS))

#: Every stretch a false-accept floor here is measured over, as
#: ``(recording, from, to, callsigns a burst in it may name)``. The callsigns are
#: this station and whoever it was calling; a handshake burst naming anyone else is
#: traffic, and traffic in a negative understates every floor scored against it by
#: an amount nobody can recover afterwards.
#:
#: It is a declaration rather than a description because prose could not be
#: checked: `test_negative_corpus` scans these with the monitor's own handshake
#: scanner and fails on anything undeclared, which is how the two strangers in
#: `ONAIR_STRANGER_REQUESTS` were found in 164 s a comment called band noise.
#:
#: The 2026-08-14 controls are deliberately not here. `CARRIER_ONLY_CHANNEL` and
#: `MONITORED_QUIET` are 754 s that `test_response_by_stream` already drives the
#: recogniser over at every shift, and a scan costs about a twentieth of real time.
QUIET_STRETCHES = (
    (ONAIR_GATEWAY_ANSWER, 0.0, 70.7, ("W9SSJ", "KC9GHZ")),
    (ONAIR_SILENT_CALLS[0], 0.0, 70.7, ("W9SSJ", "KD9USW")),
    (ONAIR_SILENT_CALLS[1], 0.0, 70.7, ("W9SSJ", "W8MW")),
    (ONAIR_GATEWAY_GREETING, 62.4, 226.9, ("W9SSJ", "KC9GHZ")),
    (CLEAR_CHANNEL, 0.0, 30.0, ()),
)


requires_bw2300_capture = _requires(BW2300_CAPTURE, what="staged BW2300 capture")
requires_bw2750_capture = _requires(BW2750_CAPTURE, what="staged BW2750 capture")
requires_bw500_captures = _requires(BW500_CAPTURES, what="staged BW500 captures")
requires_bw2300_ladder = _requires(BW2300_LADDER / "manifest.json",
                                   what="the 2026-09-04 gear-down ladder capture")
requires_bw2300_ladder_refs = _requires(
    BW2300_LADDER_REFS / "manifest.json",
    what="the 2026-09-04 ladder capture driven with no release")
requires_control_bursts = _requires(CONTROL_BURSTS, what="BW500 control-burst session")
requires_bw2300_reference = _requires(
    *BW2300_REF_SIDES, what="the two-sided real-VARA BW2300 reference session")
requires_bw500_handshake = _requires(
    BW500_HANDSHAKE, BW500_HANDSHAKE_REV,
    what="the two BW500 loopback sessions that open with a real connect request "
         "(KESTREL_CORPUS=<vara tree>)")
requires_multiframe_session = _requires(
    MULTIFRAME_SESSION,
    what="the BW500 session whose bursts carry two frames (KESTREL_CORPUS=<vara tree>)")
requires_gateway_session = _requires(GATEWAY_SESSION / "rig_rx.wav",
                                     what="off-air gateway recording")
requires_clear_channel = _requires(CLEAR_CHANNEL, what="verified clear-channel capture")
requires_control_burst_key = _requires(
    *[p for name, _, _ in CONTROL_BURST_CALLER_KEY
      for p in control_burst_sides(name)],
    what="the eight two-VARA sessions that key the control burst to a caller")
requires_fetch_two_block = _requires(
    FETCH_TWO_BLOCK, what="the 2026-09-09 fetch capture's two-block window")
requires_fetch_level_misses = _requires(
    FETCH_LEVEL_MISSES,
    what="the 2026-09-09 fetch capture whose window the level test read as over")
requires_fetch_idle_in_window = _requires(
    FETCH_IDLE_IN_WINDOW,
    what="the 2026-09-09 fetch capture whose window a session frame was named in")
requires_over_continue_answers = _requires(
    *[p for p, _ in OVER_CONTINUE_ANSWERS],
    what="the two-cable sessions holding a responder's continue bursts")
requires_over_continue_2750 = _requires(
    *[p for p, _ in OVER_CONTINUE_2750 + OVER_CONTINUE_2750_RESPONDER],
    what="the two-cable BW2750 sessions holding the continue burst")
requires_over_frame_fields = _requires(
    *[p for p, _ in OVER_FRAME_FIELDS],
    what="the two-cable sessions holding a stock station's own DATA bodies")
requires_over_close_records = _requires(
    *[p for p, _ in OVER_CLOSE_RECORDS],
    what="the two-cable sessions holding a stock station's whole delivery")
requires_ww2mi_arms = _requires(
    WW2MI_ACCEPTED, WW2MI_REFUSED,
    what="the two 2026-08-18 WW2MI connects, one authenticated and one refused")
#: The markers gated on `VARA_TREE_RECORDINGS`, beside the declaration they read.
#: What the gate wants to report is what the absence costs, and files missing is
#: not that: `tests/gates/test_corpus_present.py` counts the collected tests
#: carrying one of these instead. Counted rather than written down because the
#: number is not the number of markers — `requires_bw500_handshake` guards two
#: test functions and parametrisation makes four tests of them — and a figure
#: restated by hand is one that goes stale where nobody is looking.
VARA_TREE_MARKS = (requires_bw2300_capture, requires_bw500_captures,
                   requires_control_bursts, requires_bw2300_reference,
                   requires_bw500_handshake, requires_multiframe_session,
                   requires_gateway_session, requires_clear_channel)

requires_onair_connect_attempt = _requires(
    ONAIR_CONNECT_ATTEMPT, what="off-air recording of kestrel calling a gateway")
requires_regress_fixtures = _requires(REGRESS_FIXTURES, what="shared regression corpus")
requires_onair_gateway_answer = _requires(
    ONAIR_GATEWAY_ANSWER, what="recording of a gateway answering our call")
requires_onair_silent_calls = _requires(
    *ONAIR_SILENT_CALLS, what="the same night's calls that nobody answered")
requires_onair_gateway_greeting = _requires(
    ONAIR_GATEWAY_GREETING, what="recording of a held gateway session")
requires_channel_sense = _requires(
    *SENSE_CLEAR, *SENSE_BUSY,
    what="the 2026-08-09 slot's channel-sense captures")
requires_carrier_only_channel = _requires(
    CARRIER_ONLY_CHANNEL, what="the 2026-08-14 slot's 6800 kHz control")
requires_monitored_sessions = _requires(
    *MONITORED_SESSIONS, MONITORED_QUIET,
    what="the 2026-08-14 slot's confirmed sessions and its quiet 40 m window")
requires_monitored_quiet = _requires(
    MONITORED_QUIET, what="the 2026-08-14 slot's quiet 40 m window")
requires_qrn_80m = _requires(*QRN_80M, what="the 2026-08-16 80 m listens")
requires_sense_refused_empty = _requires(
    *SENSE_REFUSED_EMPTY, what="the 2026-08-19 refusals on a channel that was empty")
requires_onair_offset_answer = _requires(
    ONAIR_OFFSET_ANSWER, what="the 2026-08-15 call a gateway answered off frequency")
requires_onair_bw500_answer = _requires(
    ONAIR_BW500_ANSWER, ONAIR_UNANSWERED_CALL,
    what="the 2026-08-28 BW500 call K5FIT answered, and a call nobody answered")
requires_onair_faded_greeting = _requires(
    ONAIR_FADED_GREETING, what="the 2026-08-19 greeting arriving through a fade")
requires_onair_unanswered_call = _requires(
    ONAIR_UNANSWERED_CALL,
    what="the 2026-08-19 05:12 call to the same gateway that nobody answered")
requires_onair_refused_ack = _requires(
    ONAIR_REFUSED_ACK,
    what="the 2026-08-19 call to W8MW whose connected-ack was refused")
requires_onair_crowded_ack = _requires(
    ONAIR_CROWDED_ACK,
    what="the 2026-08-26 call to KD0PYG whose ack had a carrier under an occupant")
requires_onair_unattributed_answers = _requires(
    *[p for p, *_ in ONAIR_UNATTRIBUTED_ANSWERS],
    what="the two 2026-08-26 KB3AC-10 arms a station nobody can name answered")
requires_onair_peer_answers = _requires(
    *[p for p, *_ in ONAIR_PEER_ANSWERS], *[p for p, _ in ONAIR_UNANSWERED_ARMS],
    what="the four attempts whose gateways answered off the connect-response's "
         "own frame, and the two from the same slot that nobody answered")
requires_onair_unasked_handover = _requires(
    *[p for p, *_ in ONAIR_UNASKED_HANDOVER],
    what="the three 2026-08-29 arms whose gateways handed the channel over unasked")
requires_onair_gateway_overs = _requires(
    ONAIR_GATEWAY_OVERS, what="the 2026-08-16 session whose overs were read")
requires_onair_naked_over = _requires(
    ONAIR_NAKED_OVER, what="the 2026-08-20 session whose second over drew a NAK")
requires_onair_standing_link = _requires(
    ONAIR_STANDING_LINK,
    what="the 2026-08-20 02:41 session that answered one over and then stood")
requires_besra_logged_session = _requires(
    BESRA_LOGGED_SESSION, BESRA_SESSION_LOG,
    what="the 2026-08-26 W6IDS session and the log the station wrote beside it")
requires_besra_regular_call = _requires(
    BESRA_REGULAR_CALL, BESRA_REGULAR_CALL_LOG,
    what="the 2026-08-29 W4UC call whose six keyings are evenly spaced")
requires_onair_two_over_greeting = _requires(
    ONAIR_TWO_OVER_GREETING[0],
    what="the 2026-08-31 KB3AC-10 session that sent a second greeting over")
requires_unconnected_linksetups = _requires(
    *(p for p, _ in ONAIR_UNCONNECTED_LINKSETUPS),
    what="the 27 link-setups that did not end in a connect")
requires_onair_w0lon_arms = _requires(
    *(p for p, *_ in ONAIR_W0LON_ARMS),
    what="the two 2026-09-04 W0LON sessions of the same afternoon")

# The 2026-09-16 slot — naming what the peer keys. Each recording holds a burst
# a live run reached nothing at: a responder over-NAK, a mute-cut control burst,
# a below-offer connect answer, a seed-289 connect confirmation, and a
# drained-responder handover at a wide bandwidth.
ONAIR_RESPONDER_NAK_2750 = (evidence.LOGS / "onair"
                            / "20260916T024805Z-W9SSJ-KC9GHZ.wav")
ONAIR_MUTECUT_CONTROL = (evidence.LOGS / "onair"
                         / "20260916T023202Z-W9SSJ-K0SI.wav")
ONAIR_SHORT_ACK_ARMS = (
    evidence.LOGS / "onair" / "20260912T133843Z-W9SSJ-KB5LZK.wav",
    evidence.LOGS / "onair" / "20260912T132319Z-W9SSJ-KC9GHZ.wav")
ONAIR_BELOW_OFFER_2750 = (evidence.LOGS / "onair"
                          / "20260916T041401Z-W9SSJ-K7EK-10.wav")
ONAIR_SEED289_CONFIRM = (
    evidence.LOGS / "onair" / "20260916T034218Z-W9SSJ-K0SI.wav",
    evidence.LOGS / "onair" / "20260916T034443Z-W9SSJ-K0SI.wav")
ONAIR_DRAINED_HANDOVER_2300 = ONAIR_MUTECUT_CONTROL
ONAIR_NS0A_RECORD101_OVER = (evidence.LOGS / "onair"
                             / "20260916T043220Z-W9SSJ-NS0A.wav")

requires_onair_responder_nak_2750 = _requires(
    ONAIR_RESPONDER_NAK_2750,
    what="the 2026-09-16 KC9GHZ BW2750 tape whose queries drew a responder NAK")
requires_onair_mutecut_control = _requires(
    ONAIR_MUTECUT_CONTROL, *ONAIR_SHORT_ACK_ARMS,
    what="the 2026-09-16 K0SI 40 m tape and the two 09-12 short-ack arms")
requires_onair_below_offer_2750 = _requires(
    ONAIR_BELOW_OFFER_2750,
    what="the 2026-09-16 K7EK-10 BW2750 tape answered below the offer gate")
requires_onair_seed289_confirm = _requires(
    *ONAIR_SEED289_CONFIRM,
    what="the two 2026-09-16 K0SI 80 m tapes whose first link-setup was confirmed")
requires_onair_drained_handover_2300 = _requires(
    ONAIR_DRAINED_HANDOVER_2300,
    what="the 2026-09-16 K0SI 40 m tape whose drained-responder handed the turn over")
requires_onair_ns0a_record101 = _requires(
    ONAIR_NS0A_RECORD101_OVER,
    what="the 2026-09-16 NS0A tape whose record-101 over needs the session carry")


def harness(name: str):
    """A module from the out-of-package harness, or skip the test module.

    ``oracle/`` (drives a reference modem), ``tools/`` (probes) and ``analysis/``
    (offline scratch analysis) sit beside the package in the source tree and are
    not in the wheel, so this is also where they join ``sys.path``.
    """
    for d in (ROOT / "oracle", TOOLS, ROOT):
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    return pytest.importorskip(name, reason=f"harness module {name!r} not present under {ROOT}")


#: How long a stopped tool may take to put its transmitter down and go. Raise it if
#: `onair_session._CHILD_GRACE_S` or the unkey budget behind it grows.
STOP_GRACE_S = 20.0
_REAP_S = 5.0


def child_env(**extra) -> dict:
    """The environment a spawned tool needs to import the package under test.

    A subprocess inherits `sys.path` from nothing: pytest puts ``PKG_ROOT`` on its
    own path and the child resolves `hfmodem` through whatever is installed. In a
    git worktree those are two different trees, so the tool file under test ran
    against another checkout's package -- and reported that as the tool being
    broken, which is the one answer that is never true.
    """
    path = [str(PKG_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    return {**os.environ, **extra, "PYTHONPATH": os.pathsep.join(path)}


def launch(argv, **kw) -> subprocess.Popen:
    """Start a tool as its own process group leader, so :func:`stop_tool` can reach
    everything it goes on to spawn."""
    kw.setdefault("env", child_env())
    return subprocess.Popen(argv, start_new_session=True, **kw)


def _stopper(proc: subprocess.Popen):
    """Signal the whole group the tool leads. Resolved once, up front: the tool can
    exit under the first signal while its children carry on, and its group still has
    to be reachable afterwards. A tool leading no group is signalled alone, because
    the group it is in is pytest's own."""
    pgid = os.getpgid(proc.pid)
    if pgid != proc.pid:
        return proc.send_signal

    def kill_group(sig: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)

    return kill_group


def _drain(proc: subprocess.Popen, timeout: float) -> str | None:
    try:
        return proc.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        return None


def stop_tool(proc: subprocess.Popen, grace: float = STOP_GRACE_S) -> str:
    """Stop a tool :func:`launch` started, and return what it said.

    SIGTERM to the whole group first: a tool that spawns a modem which owns the
    transmitter must be given its own shutdown, and so must the modem — a bare
    ``kill()`` reaches the supervisor and leaves the child keying with nobody
    reading it. SIGKILL the group once the grace runs out.

    Every read is bounded. ``communicate()`` with no timeout waits on the pipe
    until the last descendant holding it closes, so a survivor does not fail the
    test, it stops the run: no children, no output, no end.
    """
    stop = _stopper(proc)
    stop(signal.SIGTERM)
    out = _drain(proc, grace)
    if out is None:
        stop(signal.SIGKILL)
        out = _drain(proc, _REAP_S)
    if out is None:
        proc.stdout.close()
        proc.wait(timeout=_REAP_S)
        raise AssertionError(
            f"pid {proc.pid} was killed with its whole process group and something "
            f"still holds its output open — that is what wedges a run")
    return out


def wav_mono(path: Path | str):
    """A captured WAV as float samples, first channel."""
    import numpy as np
    from scipy.io import wavfile

    a = np.asarray(wavfile.read(str(path))[1], float)
    return a[:, 0] if a.ndim > 1 else a


def bw500_capture(name: str):
    """``(recording, transmitted bytes)`` for one BW500 capture, or ``(None, None)``.

    ``cap6``'s payload is the counter that names it; the other two carry the exact
    byte stream in ``sent.bin`` beside the recording.
    """
    import numpy as np

    rec = BW500_CAPTURES / name / "rec_d16.npy"
    sent = BW500_CAPTURES / name / "sent.bin"
    if not rec.exists():
        return None, None
    if name == "cap6":
        return np.load(rec), bytes(range(256))
    return (np.load(rec), sent.read_bytes()) if sent.exists() else (None, None)
