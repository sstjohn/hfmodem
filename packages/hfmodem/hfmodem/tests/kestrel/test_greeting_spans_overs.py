# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Eighty-nine bytes is an over, not a truncation.

Both connects of the 2026-08-28 slot stalled in `awaiting greeting` having
delivered exactly 89 payload bytes, cutting mid-callsign at `remaining with K`,
and the 2026-08-28 slot read the identical ceiling on two separate
sessions as a decode limit. It is not one. 89 is `arq.phy._VARA_OVER_PAYLOAD`, the
whole payload a base BW2300 body carries once `body[89]` is set aside as the
per-frame field, and an RMS Trimode greeting is 240 bytes — so it arrives as
three overs and cutting mid-word is what the first two do.

What separates the two readings is whether the rest of the greeting was on the
air, and that is measured here rather than argued: the same guard scan over the
same window after the first over finds the second over in the session that
completed and finds nothing in the session that stalled, down to a threshold six
columns under the live gate's. The gateway sent one over and stopped.

So neither the decode nor `B2FSession`'s completion test is at fault, and the
last test says why the completion test cannot simply accept what it has: the
greeting is complete at the prompt, and on 2026-08-26 the prompt was in the third
over.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf2300 as RX
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.winlink.session import B2FSession

FS = RX.FS
MYCALL, GATEWAY = "W9SSJ", "KE8LVA"

# Surviving home taps: a different stalled KE8LVA session on 08-26, and
# the original completed session. These taps carry our TX without receiver mute.
# Exact source coordinates and hashes are in fixtures/greetings-a2-0930/provenance.json.
_FIXTURES = Path(__file__).with_name("fixtures") / "greetings-a2-0930"
STALLED = _FIXTURES / "ke8lva-stalled.flac"
COMPLETED = _FIXTURES / "ke8lva-completed.flac"
_LINK_SETUP = _FIXTURES / "ke8lva-link-setup.flac"
_FIRST = {STALLED: 0.5, COMPLETED: 0.5}

#: What KE8LVA says, over by over, off the 2026-08-26 air. The first block ends
#: mid-callsign and the second mid-sentence; only the third carries the prompt.
GREETING = (
    b"RMS Trimode 1.4.2.0 KE8LVA Winlink Gateway\r\n"
    b"W9SSJ has 1438 daily minutes remaining with K",
    b"E8LVA (EN80XS)\rSessions for users running unregistered versions of "
    b"Vara are limited to 20",
    b" minutes.\r[WL2K-5.0-B2FWIHJM$]\r;PQ: 85808086\rCMS via KE8LVA >\r",
)

#: Six columns under `_OVER_GUARD_MIN`, and still well clear of the 9 of 24 that
#: is the most band noise has reached in any recording kept here. A window scored
#: this low and finding nothing has not merely failed to decode an over — there
#: is no wideband frame in it to decode.
_FLOOR = 10

requires_stalled_greeting = pytest.mark.skipif(
    not (STALLED.exists() and COMPLETED.exists()),
    reason=f"KE8LVA stalled/completed greeting fixtures ({_FIXTURES})")


def _overs(path, t0: float, t1: float, guard: int = VA._OVER_GUARD_MIN):
    """Every base-level frame between ``t0`` and ``t1``, best alignment first.

    The live gate's own search, run over a window instead of over a bracket:
    `_rec3_alignment` scores one onset grid across whatever it is handed, so
    sweeping the onsets and keeping every start column that clears the guard asks
    the same question of a whole stretch of recording that the station asks of
    one segmenter bracket. Alignments inside a frame of one another describe the
    same burst and only the best is kept.
    """
    r = RX.RECORDS[RX.BASE_LEVEL]
    x = np.asarray(corpora.wav_mono(path)[int(t0 * FS):int(t1 * FS)], float)
    scored = []
    for onset in range(0, r.dw50, VA._OVER_ONSET_STEP):
        ncol = (len(x) - onset) // r.dw50
        if ncol < r.ncols:
            continue
        hits, share, bins = RX._guard_scores(RX._band_mag(x[onset:], RX.BASE_LEVEL,
                                                          ncol), RX.BASE_LEVEL)
        for g in np.flatnonzero(hits >= guard):
            scored.append((int(hits[g]), float(share[g]),
                           onset + int(g) * r.dw50, bins[int(g):]))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    out = []
    for h, _, at, bins in scored:
        if any(abs(at - p) < r.ncols * r.dw50 for p, _, _ in out):
            continue
        out.append((at, h, RX.check_frame(
            RX.onair_to_frame(RX._onair_from_bins(bins, RX.BASE_LEVEL),
                              RX.BASE_LEVEL), RX.BASE_LEVEL)))
    return out


def _payload(fr):
    """What the host is handed for one decoded over, as `_deliver` hands it."""
    body = bytes(fr.payload)
    return phy.vara_payload(body, caller=MYCALL, body_len=len(body))


@requires_stalled_greeting
@pytest.mark.parametrize("path", [STALLED, COMPLETED])
def test_the_first_over_is_whole_at_eighty_nine_bytes(path):
    """The stalled session's over carries everything a base over can carry.

    Byte-identical to the session that went on to complete, bar the minutes
    counter the gateway prints from its own clock — so the 89 is the frame's
    capacity and not the point a decode gave out.
    """
    at = _FIRST[path]
    (_, hits, fr), = _overs(path, at - 0.5, at + 5.0)
    assert hits >= VA._OVER_GUARD_MIN and fr.crc_ok
    assert len(bytes(fr.payload)) == RX.payload_bytes(RX.BASE_LEVEL)

    payload = _payload(fr)
    assert len(payload) == phy._VARA_OVER_PAYLOAD == 89
    assert payload.endswith(b"remaining with K")
    assert payload[:32] == GREETING[0][:32]


@requires_stalled_greeting
def test_the_rest_of_the_greeting_never_reached_the_air():
    """The measurement the two readings of the stall divide on.

    One window, one method, two recordings: from the end of the first over to
    eight seconds past it, the session that completed holds the second over and
    the session that stalled holds no wideband frame at all. Scored at `_FLOOR`,
    six columns below what the live gate demands, so this is not a decode that
    was tried and failed.

    The completed tap is the positive control for the same post-over scan.
    The stalled tap also holds our CRC-clean link-setup; neither tap measures
    receiver mute, which was specific to the lost rig recording.
    """
    span = RX._BASE_NCOLS * RX.RECORDS[RX.BASE_LEVEL].dw50 / FS

    (_, hits, fr), = _overs(COMPLETED, _FIRST[COMPLETED] + span,
                            _FIRST[COMPLETED] + span + 8.0, guard=_FLOOR)
    assert hits >= VA._OVER_GUARD_MIN and fr.crc_ok
    assert _payload(fr) == GREETING[1]

    assert _overs(STALLED, _FIRST[STALLED] + span,
                  _FIRST[STALLED] + span + 8.0, guard=_FLOOR) == []


@corpora._requires(_LINK_SETUP, what="KE8LVA link-setup control")
def test_the_link_setup_decodes_in_the_stalled_recording():
    """The control above comes from a separate clip of the same stalled tap.

    This checks that the stalled tap carries decodable frames. It is a weaker
    control than the original rig-rx pin: the tap cannot measure receiver mute.
    """
    frames = _overs(_LINK_SETUP, 0.0, (RX._BASE_NCOLS + 2) * 512 / FS)
    assert [h for _, h, _ in frames] == [24]
    assert all(fr.crc_ok for _, _, fr in frames)


def test_a_greeting_is_complete_at_the_prompt_and_not_before():
    """What `awaiting greeting` is waiting for, and why the first block is not it.

    Fed the block the 2026-08-28 slot actually received, the session holds — the
    trailing `with K` is not a line yet, there is no SID, and answering would put
    a login on the air addressed to a callsign we had not finished reading. Fed
    all three, it leaves the greeting on the `>` the third one carries and the
    login block is queued.

    So the completion test cannot be "accept what the first over brought": the
    prompt is the rule, and on this gateway the prompt was 151 bytes further on.
    """
    held = B2FSession(MYCALL, target=GATEWAY, password="mock")
    assert held.feed(GREETING[0]) == b""
    assert held.stage == "awaiting greeting"
    assert not held.remote_sid and not held.challenge

    whole = B2FSession(MYCALL, target=GATEWAY, password="mock")
    out = b"".join(whole.feed(block) for block in GREETING)
    assert whole.stage != "awaiting greeting" and not whole.failure
    assert whole.remote_sid == "[WL2K-5.0-B2FWIHJM$]"
    assert whole.challenge == "85808086"
    assert out.startswith(b";FW: W9SSJ\r") and b";PR: " in out
