# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Replay a real gateway session through the driver and check it keeps it alive.

The audio is a live Winlink RMS gateway recorded off air on 40 m (BW2300). Feeding
its transmissions to a connected initiator must produce an answer to every DATA
over — an unanswered over stops the gateway, which is the difference between
completing a connect and carrying multipart traffic  [spec 05 §5.3.3].

Skipped when the recording isn't present; the audio is gitignored, so this is a
bench test, not a CI gate.
"""
import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.tests.kestrel.corpora import (
    GATEWAY_SESSION,
    OFFAIR,
    harness,
    requires_gateway_session,
)
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

_CAP = GATEWAY_SESSION / "rig_rx.wav"
_FS = 48000

pytestmark = requires_gateway_session


class _IO(VA.VaraIO):
    def __init__(self):
        self.tx_bursts = []
        self.msgs = []

    def key(self, on): pass

    def tx(self, samples): self.tx_bursts.append(np.asarray(samples))

    def pending(self): pass

    def connected(self, caller, called, bw): pass

    def log(self, msg): self.msgs.append(msg)


def _gateway_bursts(path):
    """Keyed regions of the recording, by energy gate (self-contained so the test
    does not depend on the offline segmenter's tuning)."""
    fs, a = wavfile.read(path)
    a = a.astype(float)
    a = a[:, 0] if a.ndim > 1 else a
    a /= np.abs(a).max() or 1.0
    w = fs // 50
    p = np.empty(len(a) + 1)
    p[0] = 0.0
    np.cumsum(a * a, out=p[1:])
    lo = np.clip(np.arange(len(a)) - w // 2, 0, len(a))
    hi = np.clip(lo + w, 0, len(a))
    env = np.sqrt((p[hi] - p[lo]) / w)
    on = env > env.max() * 0.12
    d = np.diff(on.astype(int))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(a)]
    return [a[s:e] for s, e in zip(starts, ends) if (e - s) / fs >= 0.15]


def _connected_initiator(over_continue: str = VA.OVER_CONTINUE_GENERATED):
    io = _IO()
    d = VA.VaraStationHandshake(["W9SSJ"], io, bw="2300",
                                over_continue=over_continue)
    d.role, d.called, d.caller = "initiator", "KC9GHZ", "W9SSJ"
    d.state, d.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return d, io


def test_every_data_over_in_a_real_session_is_answered_and_nothing_else_is():
    """Both halves of the per-over response, on the recording itself.

    This used to count only the first half — every keyed region longer than 2.5 s
    was answered, and every one was expected to be. Six pieces of this recording
    clear that bar and two of them are gateway overs; the rest are fragments and
    other traffic, and answering those means keying at whoever they belong to
    (kestrel/tests/kestrel/test_data_over_gate.py). So the count is now against what
    ``hfmodem.kestrel.rx.varahf2300`` can decode as an over, not against what is long.
    """
    d, io = _connected_initiator()
    long = [b for b in _gateway_bursts(_CAP) if len(b) >= int(2.5 * _FS)]
    assert long, "recording should contain wideband DATA overs"
    overs = [b for b in long if rx.decode_over(b, 0, len(b)).crc_ok]
    assert overs, "recording should contain DECODABLE wideband DATA overs"
    for burst in long:
        d.on_rx_audio(burst)
    answered = sum(1 for m in io.msgs if "tx per-over response" in m)
    assert answered == len(overs), (
        f"answered {answered} of the {len(overs)} decodable overs among "
        f"{len(long)} pieces over 2.5 s")


def _decodable_overs():
    """The gateway's own DATA overs off the recording, each with what it says
    about itself: True where it closes the delivery."""
    out = []
    for b in _gateway_bursts(_CAP):
        if len(b) < int(2.5 * _FS):
            continue
        fr = rx.decode_over(b, 0, len(b))
        if fr.crc_ok:
            body = bytes(fr.payload)
            out.append((b, phy.over_is_last(body, "W9SSJ")))
    return out


def test_a_real_greeting_carries_both_kinds_of_over():
    """The discriminator, off air rather than off a bench: this gateway's
    greeting is two overs, and they are not the same shape. The first is filled
    to capacity and carries no trailer; the second is 43 bytes ending on the
    prompt and carries the one that closes a delivery."""
    kinds = [last for _, last in _decodable_overs()]
    assert kinds == [False, True], kinds


@pytest.mark.parametrize("setting", VA.OVER_CONTINUE_ANSWERS)
def test_each_over_draws_the_burst_it_asks_for(setting):
    """One burst per over, and which one is the over's own property — against a
    real gateway's own two overs rather than a synthesised pair.

    The lengths are measured and not chosen: a bench of 2026-08-30, one cable
    per direction, put the 11-symbol two-tone control burst at 0.470 s behind
    the LAST over of a delivery and a continue-class frame behind every
    intermediate one. This asserted 0.470 s for both until then, which is the
    frame a station keys only to end a delivery — and answering a gateway's
    first over with it is what took every greeting after 89 bytes.

    The closing over is not the setting's to move, which is what this asserts
    twice: only the intermediate answer changes with `over_continue`.
    """
    d, io = _connected_initiator(over_continue=setting)
    for burst, last in _decodable_overs():
        io.tx_bursts.clear()
        d.on_rx_audio(burst)
        assert len(io.tx_bursts) == 1
        answer = io.tx_bursts[0]
        secs = len(answer) / _FS
        assert np.abs(answer).max() > 0.1
        if last:
            assert 0.45 <= secs <= 0.50, secs
            pairs = MK.demod_tone_pairs(answer, VF.CONNECTED_ACK_NSYM)
            assert ([tuple(sorted(x)) for x in pairs[:4]]
                    == [tuple(sorted(x)) for x in VF.CONNECTED_ACK_PREAMBLE])
        elif setting == VA.OVER_CONTINUE_CAPTURED:
            assert 0.32 <= secs <= 0.38, secs
            pairs = MK.demod_tone_pairs(answer, VF.OVER_CONTINUE_NSYM)
            assert ([tuple(sorted(x)) for x in pairs]
                    == [tuple(sorted(x)) for x in VF.OVER_CONTINUE_CALLER_2300])
        elif setting == VA.OVER_CONTINUE_SHORT:
            assert 0.66 <= secs <= 0.70, secs
            assert (MK.demod_tones(answer, 16)
                    == VF.handshake_tones("KC9GHZ", VF.SESSION_OVER_RESPONSE_SHORT))
        else:
            assert 1.34 <= secs <= 1.40, secs
            assert (MK.demod_tones(answer, 32)
                    == VF.handshake_tones("KC9GHZ", VF.SESSION_OVER_RESPONSE))


# --------------------------------------------------------------------------- #
# End-to-end receive: the gateway's own payload, reassembled from off-air RF.
@pytest.mark.parametrize("session", ["NS0A_2300", "KC9GHZ_2300"])
def test_gateway_payload_reassembles_byte_exact(session):
    """Decode every DATA over in a real session and concatenate. The result must
    equal the plaintext VARA delivered on its host port during that same session —
    independent ground truth, so this pins the whole receive chain: segmentation,
    alignment search, rec3 demod, turbo decode, CRC and the payload boundary."""
    from hfmodem.kestrel.arq import phy
    from hfmodem.kestrel.rx import varahf2300 as RX

    find_bursts = harness("analysis.control_burst").find_bursts

    cap, truth = OFFAIR / session / "rig_rx.wav", OFFAIR / session / "payload.bin"
    if not (cap.exists() and truth.exists()):
        pytest.skip(f"off-air session {session} not present")
    fs, a = wavfile.read(cap)
    a = a.astype(float)
    a = a[:, 0] if a.ndim > 1 else a
    a /= np.abs(a).max() or 1.0

    frames = []
    for s, e in find_bursts(a, lo_s=0.15):
        seg = a[s:e]
        for s0, e0 in (RX.detect_overs(seg) or [(0, len(seg))]):
            if (e0 - s0) // 512 < 396:
                continue
            fr = RX.decode_over(seg, s0, e0)
            if fr.crc_ok:
                frames.append(phy.vara_payload(bytes(fr.frame_bytes)[:90]))
    assert b"".join(frames) == truth.read_bytes()
