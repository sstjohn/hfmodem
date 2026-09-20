# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The two W0LON arms of 2026-09-04, and what the handshake did not change.

One link came up on the gateway's own turn-request and was answered with
`SESSION_TURN_RELEASE`; the other came up on the connected-ack and was answered
with `SESSION_CONFIRM`. Everything after that instant is the same session twice:
the same silence, the same idle train of the same length, no greeting, and no
poll. So the release is not what cost the greeting.

Skipped when the recordings are absent; they are gitignored.
"""
import numpy as np
from scipy.io import wavfile

from hfmodem.tests.kestrel.corpora import (
    ONAIR_W0LON_ARMS,
    W0LON_ACK_WINDOWS,
    W0LON_IDLE_TRAINS,
    W0LON_TURN_REQUEST,
    requires_onair_w0lon_arms,
)
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

pytestmark = requires_onair_w0lon_arms

_FS = 48000
_CALL = "W0LON"
_BAND = MK.band_for("2300")
_IDLE = VF.for_bw(VF.SESSION_RESPONDER_IDLE, "2300")
_ASK = VF.for_bw(VF.SESSION_TURN_REQUEST_RESPONDER, "2300")


def _span(kind):
    return (len(kind.preamble) + kind.n_payload - 1) * MK.HOP + MK.STRIDE


def _load(path):
    _, a = wavfile.read(path)
    a = np.asarray(a, float)
    a = a[:, 0] if a.ndim > 1 else a
    return a / (np.abs(a).max() or 1.0)


def _find(a, kind, call, t0, t1, hop=0.25):
    """Each keying of ``kind`` in [t0, t1), located by its payload.

    The recogniser drops the tones a window has not delivered yet, so it takes a
    frame from about a second either side of it; the alignment that fits the most
    payload tones is the keying, and near neighbours are that same one seen from
    a different window.
    """
    tones = np.asarray(VF.handshake_tones(call, kind), dtype=np.int32)
    exp = VF.payload_bins(call, kind)
    span, win = _span(kind), int(1.9 * _FS)
    keep = []
    for i in range(int(t0 * _FS), min(int(t1 * _FS), len(a) - win), int(hop * _FS)):
        heard, at = VA._payload_fit(a[i:i + win], kind, tones, None, _BAND)
        heard = [int(t) for t in heard]
        if not VF.recognize(heard, call, kind):
            continue
        pairs = [(p, q) for p, q in zip(VF._strip_preamble(heard, kind), exp)
                 if p >= 0]
        if not pairs:
            continue
        rec = (i + at, sum(1 for p, q in pairs if p == q), len(pairs))
        near = [k for k in keep if abs(k[0] - rec[0]) < span]
        if near:
            if (rec[1] / rec[2], rec[2]) > (near[0][1] / near[0][2], near[0][2]):
                keep[keep.index(near[0])] = rec
            continue
        keep.append(rec)
    return [k[0] / _FS for k in sorted(keep)]


def _bursty(a, t0, t1, hop=0.1, win=1.0):
    """Windows in [t0, t1) holding an 11-symbol control burst or the 8-symbol
    continue burst — what a stock responder polls with after a release."""
    ack = cont = total = 0
    for at in range(int(t0 * _FS), int(t1 * _FS) - int(win * _FS),
                    int(hop * _FS)):
        w = a[at:at + int(win * _FS)]
        total += 1
        if VA._ack_plateau(w, 0, None, _BAND) >= VA._ACK_PLATEAU:
            ack += 1
        elif VA._cont_plateau(w, 0, None, _BAND) >= VA._CONT_PLATEAU:
            cont += 1
    return ack, cont, total


def _turnarounds(path, kdn, kup):
    """Every stretch of the tape between one of our keyings and the next,
    starting at the one this arm's answering burst opened."""
    a = _load(path)
    edges = _mute_edges(a)
    out, prev = [], kup
    for s, e in edges:
        if s <= kdn + 0.5:
            continue
        out.append((prev, s))
        prev = e
    return a, out


def _mute_edges(a, depth_db=-6.0, min_s=0.30, frame=512):
    """Our own transmissions: the receiver is muted under every one of them."""
    from scipy.signal import fftconvolve, firwin
    h = firwin(401, [600 / (_FS / 2), 2400 / (_FS / 2)], pass_zero=False)
    y = fftconvolve(a, h, "same")
    n = len(y) // frame
    r = np.sqrt((y[:n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-12
    low = 20 * np.log10(r / np.median(r)) < depth_db
    d = np.diff(low.astype(np.int8))
    st = list(np.flatnonzero(d == 1) + 1)
    en = list(np.flatnonzero(d == -1) + 1)
    if low[0]:
        st = [0] + st
    if low[-1]:
        en = en + [len(low)]
    return [(s * frame / _FS, e * frame / _FS) for s, e in zip(st, en)
            if (e - s) * frame / _FS >= min_s]


def test_our_keyings_are_where_the_transcripts_say():
    for path, kdn, kup, _what, _next in ONAIR_W0LON_ARMS:
        edges = _mute_edges(_load(path))
        assert any(abs(s - kdn) < 0.1 and abs(e - kup) < 0.1 for s, e in edges), \
            f"{path.name}: no keying at {kdn}-{kup}"


def test_the_release_was_not_keyed_across_the_turn_request():
    """The 19:56 arm's ask, and the 0.24 s of clear turnaround behind it."""
    path, kdn, _kup, what, _next = ONAIR_W0LON_ARMS[0]
    assert what == "turn-release"
    at, addressed = W0LON_TURN_REQUEST
    a = _load(path)
    found = _find(a, _ASK, addressed, at - 1.0, at + 1.0)
    assert len(found) == 1 and abs(found[0] - at) < 0.1
    ends = found[0] + _span(_ASK) / _FS
    assert 0.15 < kdn - ends < 0.40, f"key-down {kdn - ends:+.3f} s off its tail"


def test_the_confirm_arm_holds_no_turn_request():
    """The control: the link that came up the ordinary way was never asked."""
    path, _kdn, kup, what, _next = ONAIR_W0LON_ARMS[1]
    assert what == "session-confirm"
    a = _load(path)
    for call in ("W9SSJ", _CALL):
        assert _find(a, _ASK, call, kup - 20.0, kup + 20.0) == []


def test_the_next_thing_the_gateway_keyed_was_an_idle_on_both_arms():
    for (path, _kdn, kup, _what, nxt), (first, _last) in zip(ONAIR_W0LON_ARMS,
                                                             W0LON_IDLE_TRAINS):
        a = _load(path)
        found = _find(a, _IDLE, _CALL, kup, kup + 8.0)
        assert len(found) == 1, f"{path.name}: {len(found)} frames, want 1"
        assert abs(found[0] - nxt) < 0.1 and abs(found[0] - first) < 0.1
        assert 7.0 < found[0] - kup < 8.0


def test_both_idle_trains_run_the_same_length():
    spans = []
    for (path, *_rest), (first, last) in zip(ONAIR_W0LON_ARMS, W0LON_IDLE_TRAINS):
        a = _load(path)
        assert _find(a, _IDLE, _CALL, first - 0.5, first + 0.5)
        assert _find(a, _IDLE, _CALL, last - 0.5, last + 0.5)
        assert _find(a, _IDLE, _CALL, last + _span(_IDLE) / _FS, len(a) / _FS) == []
        spans.append(last - first)
    assert all(abs(s - 49.3) < 0.2 for s in spans), spans
    assert abs(spans[0] - spans[1]) < 0.2


def test_neither_gateway_turnaround_holds_a_poll():
    for path, kdn, kup, _what, _next in ONAIR_W0LON_ARMS:
        a, spans = _turnarounds(path, kdn, kup)
        ack = cont = total = 0
        for t0, t1 in spans:
            x, y, n = _bursty(a, t0, t1)
            ack, cont, total = ack + x, cont + y, total + n
        assert total > 400
        assert (ack, cont) == (0, 0), f"{path.name}: {ack} control, {cont} continue"


def test_the_poll_sweep_would_have_seen_one():
    """The positive control: the same sweep over the same audio with a control
    burst spliced into it, at the level the gateway's own frames arrived at."""
    path = ONAIR_W0LON_ARMS[0][0]
    a = _load(path)
    quiet = a[int(25.5 * _FS):int(31.5 * _FS)].copy()
    assert _bursty(quiet, 0.0, len(quiet) / _FS)[:2] == (0, 0)
    burst = MK.synth_tone_pairs(VF.control_bursts("W9SSJ", "2300")[1])
    scale = np.sqrt((quiet ** 2).mean() / (burst ** 2).mean())
    at = int(2.0 * _FS)
    quiet[at:at + len(burst)] += scale * burst
    ack, _cont, _n = _bursty(quiet, 0.0, len(quiet) / _FS)
    assert ack > 0


def _ack_windows(a, t0, t1, shift, win=1.0, hop=0.1):
    """Windows in [t0, t1) the connected-ack's fixed preamble stands in, at one
    offset. The offset is read off the peer's connect-response live, so an
    absence is only an absence when it is swept."""
    n = held = 0
    for at in range(int(t0 * _FS), int(t1 * _FS) - int(win * _FS), int(hop * _FS)):
        n += 1
        if VA._ack_plateau(a[at:at + int(win * _FS)], shift, None,
                           _BAND) >= VA._ACK_PLATEAU:
            held += 1
    return held, n


def test_only_the_arm_that_never_asked_keyed_a_connected_ack():
    """What the two handshakes differ by, swept over every offset a peer keys at."""
    for (path, *_rest), (t0, t1, want) in zip(ONAIR_W0LON_ARMS, W0LON_ACK_WINDOWS):
        a = _load(path)
        best = max((_ack_windows(a, t0, t1, s) for s in range(-6, 7)),
                   key=lambda r: r[0])
        assert bool(best[0]) is want, f"{path.name}: {best[0]} of {best[1]}"


def test_the_arm_that_asked_holds_no_ack_anywhere_on_the_tape():
    path = ONAIR_W0LON_ARMS[0][0]
    a = _load(path)
    for shift in range(-6, 7):
        held, n = _ack_windows(a, 0.0, len(a) / _FS - 1.0, shift, hop=0.25)
        assert held == 0, f"shift {shift:+d}: {held} of {n}"
